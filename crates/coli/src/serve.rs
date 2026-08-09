//! `coli serve [port] [warm-up prompt...]` — an OpenAI-compatible HTTP inference
//! server for the streaming GLM-5.2 engine.
//!
//! Endpoints:
//!   - `GET  /health`, `GET /`         liveness + model id
//!   - `GET  /v1/models`               list the one served model
//!   - `POST /v1/completions`          text prompt → completion
//!   - `POST /v1/chat/completions`     chat messages → reply
//!
//! Both completion endpoints honor `"stream": true` and reply with Server-Sent
//! Events (the OpenAI chunk protocol, terminated by `data: [DONE]`), so tokens
//! appear live — which matters at ~1 tok/s. There is no external HTTP or JSON
//! dependency: a minimal HTTP/1.1 server on `std::net`, requests parsed with
//! `colibri-json`, responses hand-emitted.
//!
//! Concurrency: one generation at a time. A single GPU streaming a 744B model can
//! only run one forward pass anyway, so connections are served sequentially — no
//! shared-state locking, no half-interleaved KV caches. A `read` timeout keeps a
//! silent client from wedging the accept loop.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use colibri_engine::{ExpertCache, KvCache, Model, ShardsExpertProvider};
use colibri_json::Json;
use colibri_tokenizer::Tokenizer;

/// Default listen port; overridden by a positional arg or `COLI_PORT`.
const DEFAULT_PORT: u16 = 8080;
/// Default number of tokens generated when a request omits `max_tokens`.
const DEFAULT_MAX_TOKENS: usize = 128;
/// Default served context length (prompt + completion) when `COLI_CTX` is unset.
/// A small default keeps the KV reservation tiny so the most RAM goes to resident
/// experts (and latency is lowest). `COLI_CTX` can be raised to the model max safely:
/// the adaptive expert cache's OOM guard evicts experts to fit a larger KV, so a big
/// window can't drive the box into swap (GLM-5.2's MLA KV, ~175 KB/token, becomes the
/// practical ceiling before the 1M architectural max; the GQA models' KV is far smaller).
const DEFAULT_CTX: usize = 32_768;
/// Tokens generated per warm-up prompt (enough to route a spread of experts).
const WARMUP_TOKENS: usize = 8;

/// How long a request waits for KV room before the server answers 503.
///
/// Two admissible requests can collide simply by overlapping, and the second only needs
/// the first to finish. Rejecting on first try would make the node's capacity look far
/// smaller than it is, so contention queues. A request that could never fit is separated
/// out and answered 507 immediately — waiting cannot help it, and leaving it queued would
/// block the requests behind it that *can* be served.
///
/// 30 s is well beyond a typical decode and well inside any sane client timeout.
const KV_QUEUE_SECS: u64 = 30;

/// Parse a token count like `32k`, `1m`, or `131072`.
fn parse_ctx(s: &str) -> Option<usize> {
    let s = s.trim().to_lowercase();
    let (num, mul) = if let Some(n) = s.strip_suffix('k') {
        (n, 1024usize)
    } else if let Some(n) = s.strip_suffix('m') {
        (n, 1024 * 1024)
    } else {
        (s.as_str(), 1)
    };
    num.parse::<f64>().ok().map(|v| (v * mul as f64) as usize)
}

/// Resident KV bytes per token. Thin delegate to [`KvCache::bytes_per_token`], which
/// lives beside the allocation it accounts for — every past error in this figure came
/// from a copy here drifting from what `KvCache::for_model` actually allocates.
///
/// Per-token only: [`KvCache::fixed_bytes`] (the per-sequence Mamba2 recurrent state)
/// is a separate term, so callers should prefer [`KvCache::bytes_for`].
fn kv_bytes_per_token(cfg: &colibri_core::Config) -> usize {
    KvCache::bytes_per_token(cfg)
}

type Provider<'a> = ExpertCache<ShardsExpertProvider<'a>>;

/// RAII release for [`ExpertCache::reserve_ram`], held for the life of a request.
///
/// The reservation must come back on *every* exit path — the two rejection arms, the normal
/// completion, and any early return added later. A leak is not transient: the adaptive
/// monitor subtracts `reserved` from the expert-cache budget on every 100 ms tick, so a
/// forgotten release shrinks the cache permanently for the life of the process. That failure
/// would be invisible in a smoke test and show up as a slow, unexplained decay under load,
/// which is precisely the kind of bug worth spending a Drop impl to make impossible.
struct KvRoom<'p, 'a> {
    provider: &'p Provider<'a>,
    bytes: u64,
}

impl Drop for KvRoom<'_, '_> {
    fn drop(&mut self) {
        self.provider.release_ram(self.bytes);
    }
}

/// `coli serve <snap> [port] [warm-up prompt...]`. `snap` is injected by the CLI
/// dispatcher (position 2). An optional pure-integer next arg is the port; any
/// remaining args are one warm-up prompt. `COLI_PORT` / `COLI_WARMUP` (the latter
/// `|`-separated for several prompts) are the env equivalents.
pub fn cmd_serve(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: coli serve <snapshot-dir> [port] [warm-up prompt...]");
            return ExitCode::from(2);
        }
    };
    crate::note_model_switch(&snap);

    // Port: a leading bare integer arg, else COLI_PORT, else the default.
    let mut rest = &args[3.min(args.len())..];
    let port = match rest.first().and_then(|s| s.parse::<u16>().ok()) {
        Some(p) => {
            rest = &rest[1..];
            p
        }
        None => std::env::var("COLI_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_PORT),
    };

    // Warm-up prompts: the remaining positional args as one prompt, plus each
    // `|`-separated entry of COLI_WARMUP.
    let mut warmups: Vec<String> = Vec::new();
    if !rest.is_empty() {
        warmups.push(rest.join(" "));
    }
    if let Ok(w) = std::env::var("COLI_WARMUP") {
        warmups.extend(
            w.split('|')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
        );
    }

    // ---- load model + tokenizer -------------------------------------------
    let model = match colibri_engine::load_model(&snap) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("coli serve: load model: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Leak to 'static so the optional background prefetch loader can hold the cache
    // (a server owns the model for its whole lifetime).
    let model: &'static colibri_engine::Model = Box::leak(Box::new(model));
    let tok_path = format!("{snap}/tokenizer.json");
    let tok = match Tokenizer::load(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("coli serve: load tokenizer ({tok_path}): {e}");
            return ExitCode::FAILURE;
        }
    };
    let model_id = model_id_from(&snap);
    // Served context length (prompt + completion). The served value is COLI_CTX (else a
    // memory-safe default), clamped to the model ceiling. Requests are validated against it.
    //
    // Computed *before* the expert-cache budget: the budget has to reserve the KV
    // this window can allocate, and KV is sized from ctx_len.
    //
    // `max_position_embeddings` is NOT always a hard architectural ceiling. For a NoPE
    // model there is no positional-embedding table to overflow — Nemotron-H's 8 attention
    // layers apply no rope at all (position comes from the Mamba layers), so its 262144 is
    // an advisory default while the model card documents support "up to 1M tokens". The
    // upstream runtimes treat it the same way: vLLM needs VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    // and SGLang SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 to exceed it.
    //
    // COLI_ALLOW_LONG_CTX=1 is our equivalent. It stays opt-in rather than automatic:
    // running past the length a model was validated at is a quality decision, not a
    // memory one, and the RAM clamp below still applies either way.
    let allow_long = std::env::var("COLI_ALLOW_LONG_CTX").is_ok_and(|v| v != "0");
    let model_max = match (model.cfg.max_ctx, allow_long) {
        (m, false) if m > 0 => m as usize,
        _ => usize::MAX,
    };
    let requested_ctx = std::env::var("COLI_CTX")
        .ok()
        .and_then(|s| parse_ctx(&s))
        .unwrap_or(DEFAULT_CTX)
        .clamp(1, model_max);
    if allow_long && model.cfg.max_ctx > 0 && requested_ctx > model.cfg.max_ctx as usize {
        eprintln!(
            "[serve] COLI_ALLOW_LONG_CTX: serving {requested_ctx} tokens, past this model's \
             advertised max_position_embeddings ({}). Valid for NoPE models (no positional \
             table to overflow); output quality past the validated length is not guaranteed.",
            model.cfg.max_ctx
        );
    }
    // Also clamp to what RAM can actually hold: a full-window request's KV
    // (`kv_bytes_per_token * ctx * copies`, non-evictable) must fit alongside the dense
    // tier + a minimal streaming expert cache + the OOM floor. Reserve ~18 GB for those;
    // the rest is available to KV (experts evict to make room). This is the real ceiling —
    // for the GQA models it lands well below their architectural max, because full K/V is
    // stored per kv-head. Without this a big `COLI_CTX` looks fine at startup and only OOMs
    // when a large request finally allocates its KV.
    const CTX_RAM_RESERVE: u64 = 18 << 30;
    let kv_pt = (kv_bytes_per_token(&model.cfg) as u64).max(1); // already includes device shadow
                                                                // Subtract the fixed per-sequence state (Mamba2) before dividing: it is not a
                                                                // per-token cost, so folding it into `kv_pt` would scale it with context.
    let kv_fixed = KvCache::fixed_bytes(&model.cfg) as u64;
    let ram_ctx = colibri_engine::total_ram_bytes()
        .map(|t| (t.saturating_sub(CTX_RAM_RESERVE).saturating_sub(kv_fixed) / kv_pt) as usize)
        .unwrap_or(usize::MAX)
        .max(1);
    let ctx_len = requested_ctx.min(ram_ctx);
    if requested_ctx > ctx_len {
        eprintln!(
            "[serve] COLI_CTX {requested_ctx} exceeds what RAM can hold as KV; clamped to \
             {ctx_len} tokens (a full-window KV would not fit alongside the model)."
        );
    }

    // Worst-case KV for the served window (a single full-context request). Not reserved up
    // front — each request commits its own KV — so this is only the ceiling we quote.
    //
    // **This comment used to say requests evict experts just-in-time "see
    // `ExpertCache::reserve_ram`". That is false and always was**: `reserve_ram` has no
    // callers anywhere in the tree, and `COLI_GUARD_TRACE=1` on a serve run shows
    // `reserved=0.00 GB` on all 998 ticks. It is the second comment found citing that dead
    // function as a live mechanism (the first was `RUNTIME_RESERVE`, corrected earlier).
    //
    // What actually happens: `handle` commits through `commit_or_wait(Class::Kv, kv_bytes,
    // rigid, …)` against `rigid = ceiling − Dense − Experts`. Nothing evicts experts for a
    // specific pending request. `Experts` is the CURRENT cache — memory that is entirely
    // evictable — so a prompt can be refused while tens of GB of reclaimable cache sits
    // resident. The adaptive monitor does evict, but against *its own* ceiling on a 100 ms
    // tick, not on behalf of a waiting request. See task #39.
    let kv_worst_case = KvCache::bytes_for(&model.cfg, ctx_len) as u64;
    let budget = crate::ram_budget_reserving(kv_worst_case.min(8 << 30));
    let gib = (1u64 << 30) as f64;
    if budget == u64::MAX {
        // No /proc/meminfo (non-Linux dev box): the budget is unbounded, and printing
        // it as a number renders "17179869184 GB". Say what actually happened.
        println!("[serve] expert cache: unbounded (no MemAvailable to budget from)");
    } else {
        println!(
            "[serve] expert cache: {:.0} GB initial budget (adaptive); KV reserved per \
             request (≤ {:.1} GB at the full {} ctx{})",
            budget as f64 / gib,
            kv_worst_case as f64 / gib,
            ctx_len,
            if cfg!(feature = "cuda") {
                ", incl. device shadow"
            } else {
                ""
            }
        );
    }

    let usage_path = std::env::var("COLI_USAGE").unwrap_or_else(|_| format!("{snap}/.coli_usage"));
    let history = colibri_engine::UsageHistory::load(&usage_path).unwrap_or_default();

    // The expert->node map comes first: it gates what this node may load (below), and
    // a cluster that disagrees about it must fail before we pay for the AUTOPIN
    // warm-up (verification is seconds, pinning can be minutes). Single-node collapses
    // to "everything is local".
    let cluster = colibri_cluster::ClusterConfig::from_env();
    let sharding = if cluster.is_single_node() {
        colibri_cluster::ExpertSharding::single(model.cfg.n_experts as u32)
    } else {
        crate::build_sharding(&cluster, model.cfg.n_experts as u32, &history)
    };

    // Ownership is enforced at the load layer too, not just at dispatch: the provider
    // refuses experts this node doesn't own, so a routing bug fails loudly instead of
    // silently streaming a peer's expert off disk.
    let base = ShardsExpertProvider::with_sharding(
        &model.shards,
        &model.cfg,
        model.ebits as u32,
        sharding.clone(),
        cluster.this_node,
    );
    let provider = std::sync::Arc::new(ExpertCache::new(base, budget));
    let owned_ids: Vec<u32> = sharding.local_experts(cluster.this_node).collect();
    let maxres = crate::wire_adaptive_cache(
        &provider,
        &model.cfg,
        model.ebits as u32,
        &owned_ids,
        model.resident_bytes(),
    );
    crate::preload_all_experts(&provider, &model.cfg, maxres, &owned_ids);
    if let Some(topn) = crate::prefetch_topn() {
        provider.enable_prefetch(topn, model.cfg.n_experts as u64);
        println!("[serve] speculative next-layer prefetch on (top-{topn}/layer)");
    }

    // Multi-node: install the expert-parallel context so moe() splits experts by
    // ownership — this node computes its own shard, and peers' experts are fetched
    // from their `worker` servers over TCP/RoCE. Single-node leaves it unset.
    if !cluster.is_single_node() {
        let peers = match crate::cluster_peers(&cluster) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("coli serve: {e}");
                return ExitCode::FAILURE;
            }
        };
        let n_peers = peers.len();
        let owned = sharding.count_for(cluster.this_node);
        let transport =
            colibri_cluster::TcpTransport::new(cluster.this_node, peers, sharding.fingerprint());

        // Handshake with every worker up front: if any disagrees about the expert map
        // (or isn't up yet), fail here rather than silently mis-routing experts once
        // tokens start flowing.
        use colibri_cluster::Transport as _;
        if let Err(e) = transport.verify_peers() {
            eprintln!("coli serve: cluster verification failed: {e}");
            return ExitCode::FAILURE;
        }
        println!(
            "[serve] expert-parallel: {} nodes, rank {} owns {} experts; \
             {} peer(s) agreed on sharding {:#018x}",
            cluster.num_nodes,
            cluster.this_node.0,
            owned,
            n_peers,
            sharding.fingerprint()
        );
        colibri_engine::set_cluster(colibri_engine::ClusterCtx {
            sharding: sharding.clone(),
            transport: Box::new(transport),
        });
    }

    // Pinned hot-store (AUTOPIN) from the persistent usage history: routing is heavily
    // skewed, so keeping the hot head resident stops it churning through the LRU.
    // `COLI_PIN_GB=auto` sizes it to the usage curve's knee. Can take minutes (it reads
    // every pinned expert), so it runs only once the cluster is known-good.
    //
    // Restricted to the experts we own: every node reads the same history, so an
    // unfiltered pin would spend this node's cache on a peer's shard (and now be
    // rejected outright by the provider's ownership gate).
    let own_history = crate::owned_history(&history, &sharding, cluster.this_node);
    crate::apply_autopin(&provider, &own_history, budget);

    // ---- warm-up ----------------------------------------------------------
    for (i, w) in warmups.iter().enumerate() {
        let ids = tok.encode(w);
        if ids.is_empty() {
            continue;
        }
        eprintln!(
            "[serve] warm-up {}/{}: {} tokens",
            i + 1,
            warmups.len(),
            ids.len()
        );
        let mut kv = mk_kv(model, ids.len() + WARMUP_TOKENS);
        if let Err(e) =
            colibri_engine::generate_greedy(model, &mut kv, &*provider, &ids, WARMUP_TOKENS)
        {
            eprintln!("[serve] warm-up failed: {e}");
        }
    }
    if !warmups.is_empty() {
        let s = provider.stats();
        eprintln!(
            "[serve] warm-up done: {} experts resident ({:.1} GB)",
            s.resident,
            s.bytes as f64 / (1u64 << 30) as f64
        );
    }

    // ---- discover peers, BEFORE binding -----------------------------------
    //
    // Scan the ConnectX/RoCE fabric and print the other Sparks we can see, so the operator
    // can verify the multi-node wiring at startup. COLI_DISCOVER_SECS=0 skips it.
    //
    // This runs before the bind, and the ordering is the point. It used to sit between the
    // bind and the accept loop, which meant the socket was LISTENING for ~3 s while nothing
    // was accepting. Everything in this repo treats a live listener as "ready" —
    // `scripts/serve.sh` polls `ss -ltn`, and the comment in that script says so outright —
    // so "ready" was announced 3 s early and the first real request stalled for the
    // remainder of the scan. Measured 2026-08-08: first `GET /health` after startup took
    // **3.315 s**, against 0.0003 s with COLI_DISCOVER_SECS=0.
    //
    // Scanning first costs 3 s before the port opens, which is nothing beside a model load
    // that already ran for seconds to minutes, and it makes the invariant true instead of
    // nearly true. The beacon stays AFTER the bind on purpose: announcing this node's port
    // before anything can accept on it invites a peer to connect into the same gap.
    //
    // The signal handlers go in before the scan, not after the bind where they used to sit.
    // Moving the scan earlier would otherwise have left a 3 s window running under the
    // DEFAULT signal disposition. That still exits — it is not a hang — but it skips this
    // binary's deliberate path, and the window is exactly when an operator notices they
    // started the wrong model. Measured after the move: SIGTERM mid-scan exits in 1 ms.
    crate::install_shutdown_handlers();
    let disc_secs = std::env::var("COLI_DISCOVER_SECS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(3.0);
    let rank: u32 = std::env::var("COLI_NODE_RANK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if disc_secs > 0.0 {
        let d = colibri_cluster::discover(rank, port, Duration::from_secs_f64(disc_secs));
        let _ = colibri_cluster::discovery::print_report(&d, &mut std::io::stdout());
    }

    // ---- listen -----------------------------------------------------------
    let addr = format!("0.0.0.0:{port}");
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("coli serve: bind {addr}: {e}");
            return ExitCode::FAILURE;
        }
    };
    // SIGINT/SIGTERM handling (installed above, before the fabric scan) is what stops the
    // server — often PID 1 under Docker — on Ctrl-C or `docker stop` instead of hanging
    // until SIGKILL. A nonblocking listener plus the poll below is what lets this loop
    // actually notice the flag.
    if let Err(e) = listener.set_nonblocking(true) {
        eprintln!("[serve] warning: set_nonblocking failed ({e}); Ctrl-C may be slow");
    }
    println!(
        "[serve] coli {} — OpenAI-compatible server on http://{addr}  (model: {model_id})",
        crate::version_string()
    );
    let kv_at_ctx = KvCache::bytes_for(&model.cfg, ctx_len) as f64 / (1u64 << 30) as f64;
    let model_max_str = if model.cfg.max_ctx > 0 {
        model.cfg.max_ctx.to_string()
    } else {
        "unknown".to_string()
    };
    println!(
        "[serve]   context length: {ctx_len} tokens (model max {model_max_str}; up to {:.1} GB KV) — set COLI_CTX to change",
        kv_at_ctx
    );
    println!(
        "[serve]   POST /v1/chat/completions   POST /v1/completions   GET /v1/models   GET /health"
    );

    // Keep beaconing so peers that start later discover this node too. The scan itself
    // already ran above; only the announcement waits for a socket that can answer.
    if disc_secs > 0.0 {
        colibri_cluster::discovery::spawn_beacon(rank, port);
    }

    while !crate::shutdown_requested() {
        match listener.accept() {
            // handle() does blocking, timeout-bounded reads; the listener is
            // nonblocking but the accepted socket must not be, or reads spin.
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                handle(stream, model, &*provider, &tok, &model_id, ctx_len);
            }
            // No pending connection: WAIT ON THE SOCKET, not on the clock.
            //
            // This used to be `sleep(100ms)`, justified as "100 ms of accept latency is
            // nothing next to a multi-second generation". That was true of the model it
            // was written for — GLM spends ~1100 ms per token — and false of maple, which
            // spends 8.2. A connection arriving just after a nap waited out the rest of
            // it, so `GET /health`, a route that returns a constant and never touches the
            // model, measured 63-97 ms (2026-08-08, curl and urllib agreeing, TCP connect
            // 0.1 ms). On a 32-token maple request that was 15-24% of the wall clock, and
            // it is most of the gap between maple's 112.8 tok/s decode and its 70.5 serve.
            //
            // `poll` gives back both properties at once: it returns the instant a
            // connection lands, AND it returns after the timeout so SHUTDOWN is still
            // re-checked ~10x/second. Nothing else about the loop changes.
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                wait_for_connection(&listener, 100);
            }
            // EINTR from the signal itself: loop and let the SHUTDOWN check handle it.
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => eprintln!("[serve] accept: {e}"),
        }
    }
    println!("[serve] shutdown signal received — stopping");
    // Loop noticed the signal in ~50ms (measured); shutdown_exit then skips Drop. The
    // remaining ~2s is the kernel reclaiming this ~60 GB process, unavoidable and well
    // inside docker's grace — see shutdown_exit's docs.
    crate::shutdown_exit()
}

/// Block until the listener has a pending connection or `timeout_ms` elapses.
///
/// The accept loop needs both: no latency when a client is waiting, and a bounded nap so
/// the shutdown flag gets re-checked. A sleep only gives the second. `poll` gives both.
/// EINTR is not an error here — a signal arriving is precisely what the caller wants to
/// return and re-check for.
#[cfg(unix)]
fn wait_for_connection(listener: &TcpListener, timeout_ms: i32) {
    use std::os::unix::io::AsRawFd;
    let mut pfd = libc::pollfd {
        fd: listener.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
}

/// Non-unix fallback: the old behaviour. Windows is not a target this repo builds for,
/// so this exists to keep the code compiling rather than because it is good.
#[cfg(not(unix))]
fn wait_for_connection(_listener: &TcpListener, timeout_ms: i32) {
    std::thread::sleep(Duration::from_millis(timeout_ms.max(0) as u64));
}

fn mk_kv(model: &Model, max_t: usize) -> KvCache {
    KvCache::for_model(model, max_t)
}

/// Per-request stage timing. Off unless `COLI_SERVE_TIMING=1`, same convention as
/// `COLI_TIMING` in the engine.
///
/// This exists because the serve column is a **32-token request rate**, so anything paid
/// once per request is divided across only 32 tokens and shows up as a slower engine. The
/// first 100 ms of that turned out to be the accept loop sleeping, and it was found by
/// timing `GET /health` — a route with no model in it — rather than by reading the request
/// path. What remains after that fix (51 ms on maple, 617 ms on nemotron, measured
/// 2026-08-08) is still unattributed, and guessing at it from the source has a poor record
/// in this repo. So: measure per stage, and let the numbers say which one it is.
struct Stages {
    on: bool,
    start: Instant,
    last: Instant,
    marks: Vec<(&'static str, f64)>,
}

impl Stages {
    fn new() -> Self {
        let on = std::env::var("COLI_SERVE_TIMING").ok().as_deref() == Some("1");
        let now = Instant::now();
        Stages {
            on,
            start: now,
            last: now,
            marks: Vec::new(),
        }
    }

    fn mark(&mut self, name: &'static str) {
        if !self.on {
            return;
        }
        let now = Instant::now();
        self.marks
            .push((name, (now - self.last).as_secs_f64() * 1e3));
        self.last = now;
    }

    /// One line per request. `total` is measured independently of the marks rather than
    /// summed from them, so any stage that is not covered shows up as a gap instead of
    /// being silently absorbed — the failure mode that made the head-cost figure wrong
    /// twice.
    fn report(&self, prompt_tokens: usize, gen_tokens: usize) {
        if !self.on {
            return;
        }
        let total = self.start.elapsed().as_secs_f64() * 1e3;
        let summed: f64 = self.marks.iter().map(|(_, ms)| ms).sum();
        let mut s = String::new();
        for (n, ms) in &self.marks {
            s.push_str(&format!("  {n} {ms:.2}"));
        }
        eprintln!(
            "[serve][timing] prompt {prompt_tokens} tok, gen {gen_tokens} tok, \
             total {total:.2} ms |{s}  (unaccounted {:.2})",
            total - summed
        );
    }
}

/// Derive a display model id from the snapshot path (the HF repo dir name, or the
/// leaf directory).
fn model_id_from(snap: &str) -> String {
    let trimmed = snap.trim_end_matches('/');
    // HF cache layout: .../models--org--name/snapshots/<hash>
    if let Some(pos) = trimmed.find("models--") {
        let seg = &trimmed[pos + "models--".len()..];
        let name = seg.split('/').next().unwrap_or(seg).replace("--", "/");
        if !name.is_empty() {
            return name;
        }
    }
    trimmed.rsplit('/').next().unwrap_or("glm-5.2").to_string()
}

// ---- request handling -----------------------------------------------------

struct Request {
    method: String,
    path: String,
    body: String,
}

#[allow(clippy::too_many_arguments)]
fn handle(
    mut stream: TcpStream,
    model: &Model,
    provider: &Provider,
    tok: &Tokenizer,
    model_id: &str,
    ctx_len: usize,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_nodelay(true);
    let mut reader = match stream.try_clone() {
        Ok(s) => BufReader::new(s),
        Err(_) => return,
    };
    let req = match read_request(&mut reader) {
        Some(r) => r,
        None => return, // malformed / timed out / client closed
    };

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") | ("GET", "/health") => {
            send_json(
                &mut stream,
                200,
                &format!(
                    "{{\"status\":\"ok\",\"model\":{},\"max_model_len\":{ctx_len}}}",
                    jstr(model_id)
                ),
            );
        }
        ("GET", "/v1/models") => {
            // `max_model_len` mirrors the field vLLM/others expose so clients can
            // discover the served context window.
            let body = format!(
                "{{\"object\":\"list\",\"data\":[{{\"id\":{},\"object\":\"model\",\"owned_by\":\"colibri\",\"max_model_len\":{ctx_len}}}]}}",
                jstr(model_id)
            );
            send_json(&mut stream, 200, &body);
        }
        ("POST", "/v1/completions") => complete(
            &mut stream,
            model,
            provider,
            tok,
            model_id,
            &req.body,
            false,
            ctx_len,
        ),
        ("POST", "/v1/chat/completions") => complete(
            &mut stream,
            model,
            provider,
            tok,
            model_id,
            &req.body,
            true,
            ctx_len,
        ),
        ("OPTIONS", _) => send_json(&mut stream, 204, ""),
        _ => send_json(
            &mut stream,
            404,
            "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}",
        ),
    }
}

/// Read an HTTP/1.1 request: request line, headers, and a `Content-Length` body.
/// Generic over the reader so it can be exercised with an in-memory buffer.
fn read_request<R: BufRead>(reader: &mut R) -> Option<Request> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    let mut content_length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).ok()? == 0 {
            break;
        }
        let t = h.trim_end();
        if t.is_empty() {
            break; // end of headers
        }
        if let Some((k, v)) = t.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body).ok()?;
    }
    Some(Request {
        method,
        path,
        body: String::from_utf8_lossy(&body).into_owned(),
    })
}

/// Shared handler for `/v1/completions` (chat=false) and `/v1/chat/completions`
/// (chat=true): parse the request, build the prompt token ids, then either stream
/// SSE chunks or return one JSON object.
#[allow(clippy::too_many_arguments)]
fn complete(
    stream: &mut TcpStream,
    model: &Model,
    provider: &Provider,
    tok: &Tokenizer,
    model_id: &str,
    body: &str,
    chat: bool,
    ctx_len: usize,
) {
    let mut stages = Stages::new();
    let req = match Json::parse(body) {
        Some(j) => j,
        None => {
            send_json(stream, 400, "{\"error\":{\"message\":\"invalid JSON body\",\"type\":\"invalid_request_error\"}}");
            return;
        }
    };
    let obj = match req.as_object() {
        Some(o) => o,
        None => {
            send_json(stream, 400, "{\"error\":{\"message\":\"body must be a JSON object\",\"type\":\"invalid_request_error\"}}");
            return;
        }
    };

    stages.mark("json");
    let requested_max = obj
        .get("max_tokens")
        .and_then(|v| v.as_i64())
        .map(|n| n.max(1) as usize)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    let stream_mode = obj.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // Build the prompt token ids.
    let ids = if chat {
        let msgs = obj.get("messages").and_then(|v| v.as_array());
        match msgs {
            Some(m) => build_chat_prompt(tok, m, model.cfg.arch),
            None => {
                send_json(stream, 400, "{\"error\":{\"message\":\"missing 'messages'\",\"type\":\"invalid_request_error\"}}");
                return;
            }
        }
    } else {
        match obj.get("prompt").and_then(|v| v.as_str()) {
            Some(p) => tok.encode(p),
            None => {
                send_json(stream, 400, "{\"error\":{\"message\":\"missing 'prompt' (string)\",\"type\":\"invalid_request_error\"}}");
                return;
            }
        }
    };
    stages.mark("tokenize");
    if ids.is_empty() {
        send_json(
            stream,
            400,
            "{\"error\":{\"message\":\"empty prompt\",\"type\":\"invalid_request_error\"}}",
        );
        return;
    }

    // Enforce the served context window (prompt + completion). A prompt at/over
    // the limit is rejected (OpenAI-style); otherwise the completion is clamped to
    // the remaining room so it always fits the KV cache.
    if ids.len() >= ctx_len {
        let msg = format!(
            "This model's maximum context length is {ctx_len} tokens, but the prompt is {} tokens. Shorten the prompt or raise COLI_CTX.",
            ids.len()
        );
        send_json(
            stream,
            400,
            &format!("{{\"error\":{{\"message\":{},\"type\":\"invalid_request_error\",\"code\":\"context_length_exceeded\"}}}}", jstr(&msg)),
        );
        return;
    }
    let max_tokens = requested_max.min(ctx_len - ids.len());

    let object = if chat {
        "chat.completion"
    } else {
        "text_completion"
    };
    let id = format!("cmpl-{}", ids.len().wrapping_mul(2654435761) ^ max_tokens);
    // The KV cache commits lazily (grows with tokens produced), so we reserve only the
    // PROMPT's KV — what prefill commits at once. The generation tail grows one token at a
    // time (~KB/token), which the adaptive monitor evicts experts against gradually; no
    // giant eager allocation to race, so short generations never pay for a big `max_tokens`.
    // If even the prompt's KV can't fit after evicting every expert, reject rather than OOM.
    // `bytes_for` adds the fixed per-sequence term (Nemotron-H's ~174 MB of Mamba2 conv+scan
    // state). That is O(1) in context, so counting only per-token bytes under-reserves worst
    // for SHORT prompts — where the per-token figure is far too small to cover it.
    let kv_bytes = KvCache::bytes_for(&model.cfg, ids.len()) as u64;

    // MAKE ROOM BEFORE ASKING WHETHER IT FITS.
    //
    // `rigid` below subtracts the CURRENT expert cache — memory that is entirely
    // reclaimable. Measured 2026-08-02 on m2.7: a 186828-token prompt needing 93.9 GB was
    // refused against a 37.7 GB budget while 71.4 GB of evictable cache sat resident. It
    // would have fit by 15.2 GB. `reserve_ram` evicts LRU experts down to a floor, holds the
    // bytes against the monitor (which subtracts `reserved` from the cache budget every
    // tick, so experts cannot refill into the space), and rolls back if even a full eviction
    // cannot cover the request.
    //
    // This is the call a comment two hundred lines up claimed already existed. It did not:
    // `reserve_ram` had ZERO callers, and `COLI_GUARD_TRACE=1` showed reserved=0.00 GB on
    // every tick of a serve run.
    //
    // COST, stated plainly: admitting one very large prompt can evict most of the expert
    // cache, and every concurrent request then streams from disk until the monitor refills.
    // That is the right trade against the alternative — refusing outright a request the box
    // can serve — but it is a real cost, not a free win.
    let experts_before = colibri_engine::ram::manager()
        .map(|m| m.committed_in(colibri_engine::ram::Class::Experts))
        .unwrap_or(0);
    if !provider.reserve_ram(kv_bytes) {
        let gib = (1u64 << 30) as f64;
        eprintln!(
            "[serve] REFUSED 507: {} tokens need {:.1} GiB KV — does not fit even after \
             evicting the expert cache to its floor",
            ids.len(),
            kv_bytes as f64 / gib,
        );
        let msg = format!(
            "Prompt too large for this node: {} tokens need ~{:.1} GiB of KV cache, which \
             does not fit even after evicting the expert cache. Use a shorter prompt.",
            ids.len(),
            kv_bytes as f64 / gib
        );
        send_json(
            stream,
            507,
            &format!("{{\"error\":{{\"message\":{},\"type\":\"insufficient_memory\",\"code\":\"kv_cache_too_large\"}}}}", jstr(&msg)),
        );
        return;
    }
    // Release on EVERY exit path. A leaked reservation is permanent: the monitor subtracts
    // `reserved` from the cache budget on every 100 ms tick, so the expert cache would stay
    // that much smaller for the life of the process.
    let _room = KvRoom {
        provider,
        bytes: kv_bytes,
    };
    // Say so when a request cost the cache something. Without this the fix is invisible:
    // an admitted request looks the same as one that always fitted, and a client timeout
    // cannot distinguish "admitted and prefilling" from "queued" — which is precisely how
    // an earlier attempt to observe this mis-reported itself.
    stages.mark("reserve_ram");
    let experts_after = colibri_engine::ram::manager()
        .map(|m| m.committed_in(colibri_engine::ram::Class::Experts))
        .unwrap_or(0);
    if experts_after < experts_before {
        let gib = (1u64 << 30) as f64;
        eprintln!(
            "[serve] ADMITTED {} tokens ({:.1} GiB KV) by evicting {:.1} GiB of expert cache \
             ({:.1} -> {:.1} GiB) — concurrent requests will stream until the monitor refills",
            ids.len(),
            kv_bytes as f64 / gib,
            (experts_before - experts_after) as f64 / gib,
            experts_before as f64 / gib,
            experts_after as f64 / gib,
        );
    }

    // Admit through the RAM ledger, which knows the dense tier and the expert arena.
    // A request that merely collides with another in flight WAITS — both may be
    // individually admissible, and rejecting the second would make capacity look far
    // smaller than it is. A request too large for the whole rigid budget is rejected at
    // once, because waiting cannot help it and it would block the queue behind it.
    //
    // Computed AFTER `reserve_ram`, so `Class::Experts` already reflects the eviction
    // (`evict_to_protecting` publishes the ledger on both its exit paths).
    //
    // Scratch and ReadBuf are subtracted too. They were omitted, which made this budget
    // optimistic by ~4.4 GB (measured: ReadBuf 0-2.1 GB, CUDA scratch 0.17-2.30 GB across
    // the fleet — decimal GB, as forward.rs's `[profile]` lines divide by 1e9, unlike the
    // GiB used for the runtime figures below).
    // That error points the OPPOSITE way to the expert one and is far smaller,
    // so it was masked — but with experts now evicted out of the way it is what remains.
    let rigid = colibri_engine::ram::manager()
        .map(|m| {
            m.ceiling()
                .saturating_sub(m.committed_in(colibri_engine::ram::Class::Dense))
                .saturating_sub(m.committed_in(colibri_engine::ram::Class::Experts))
                .saturating_sub(m.committed_in(colibri_engine::ram::Class::Scratch))
                .saturating_sub(m.committed_in(colibri_engine::ram::Class::ReadBuf))
        })
        .unwrap_or(u64::MAX);
    let (verdict, kv_commit) = colibri_engine::ram::commit_or_wait(
        colibri_engine::ram::Class::Kv,
        kv_bytes,
        rigid,
        std::time::Duration::from_secs(KV_QUEUE_SECS),
    );
    let _kv_commit = match verdict {
        colibri_engine::ram::Admission::Ok => kv_commit,
        colibri_engine::ram::Admission::TooLarge => {
            let gib = (1u64 << 30) as f64;
            // Both refusal paths used to answer the client and record NOTHING, so a node
            // refusing every long prompt looked identical to one nobody was calling — and a
            // black-box probe cannot tell a rejection from a slow prefill, which is exactly
            // how an attempt to observe this from outside failed.
            //
            // Reaching here now means the LEDGER refused after `reserve_ram` already evicted
            // to fit — i.e. the accounting says no even though physical memory said yes.
            // `still_resident` is what survived the eviction (pinned entries and the cache
            // floor), so a large value here is a genuine signal, not the old #39 bug: it
            // would mean eviction is not reaching what the ledger is counting.
            let still_resident = colibri_engine::ram::manager()
                .map(|m| m.committed_in(colibri_engine::ram::Class::Experts))
                .unwrap_or(0);
            eprintln!(
                "[serve] REFUSED 507 (ledger): {} tokens need {:.1} GiB KV, rigid budget \
                 {:.1} GiB after evicting to fit ({:.1} GiB experts still resident)",
                ids.len(),
                kv_bytes as f64 / gib,
                rigid as f64 / gib,
                still_resident as f64 / gib,
            );
            let msg = format!(
                "Prompt too large for this node: {} tokens need ~{:.1} GiB of KV cache, \
                 but only ~{:.1} GiB is available for requests after the model's own \
                 memory. Use a shorter prompt.",
                ids.len(),
                kv_bytes as f64 / gib,
                rigid as f64 / gib
            );
            send_json(
                stream,
                507,
                &format!("{{\"error\":{{\"message\":{},\"type\":\"insufficient_memory\",\"code\":\"kv_cache_too_large\"}}}}", jstr(&msg)),
            );
            return;
        }
        colibri_engine::ram::Admission::Busy => {
            let gib = (1u64 << 30) as f64;
            eprintln!(
                "[serve] REFUSED 503: {} tokens need {:.1} GiB KV; waited {KV_QUEUE_SECS}s and \
                 other requests did not free it (rigid budget {:.1} GiB)",
                ids.len(),
                kv_bytes as f64 / gib,
                rigid as f64 / gib,
            );
            let msg = format!(
                "Server busy: this prompt needs ~{:.1} GiB of KV cache and other requests \
                 did not free it within {KV_QUEUE_SECS}s. Retry shortly.",
                kv_bytes as f64 / (1u64 << 30) as f64
            );
            send_json(
                stream,
                503,
                &format!("{{\"error\":{{\"message\":{},\"type\":\"server_busy\",\"code\":\"kv_cache_contended\"}}}}", jstr(&msg)),
            );
            return;
        }
    };
    stages.mark("admit");
    let mut kv = mk_kv(model, ids.len() + max_tokens);
    stages.mark("kv_alloc");

    if stream_mode {
        stream_completion(
            stream, model, provider, tok, &ids, max_tokens, &id, model_id, object, chat, &mut kv,
        );
    } else {
        block_completion(
            stream, model, provider, tok, &ids, max_tokens, &id, model_id, object, chat, &mut kv,
        );
    }
    stages.mark("generate+send");
    stages.report(ids.len(), max_tokens);
    drop(kv); // free the KV before the commitment that covers it
}

/// Official GLM-5.2 chat template (byte-matches `chat_template.jinja`, mirrored
/// from the C reference): `[gMASK]<sop>` then `<|role|>{content}` per message with
/// **no** separators, ending with `<|assistant|><think></think>` — the empty think
/// block disables reasoning so the model answers directly. The control tokens
/// (`<|user|>`, `<|assistant|>`, …) are added-vocab entries, so encoding the
/// assembled string resolves them to their ids exactly as the C engine does.
fn build_chat_prompt(tok: &Tokenizer, messages: &[Json], arch: colibri_core::Arch) -> Vec<i32> {
    match arch {
        colibri_core::Arch::MinimaxM2 => build_chat_prompt_minimax(tok, messages),
        colibri_core::Arch::NemotronH => build_chat_prompt_nemotron(tok, messages),
        colibri_core::Arch::Maple => build_chat_prompt_maple(tok, messages),
        // GLM-5.2 / MiniMax-M3: GLM-style chat markers.
        _ => {
            let mut s = String::from("[gMASK]<sop>");
            for m in messages {
                let o = match m.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                let role = o.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let content = o.get("content").and_then(|v| v.as_str()).unwrap_or("");
                s.push_str(&format!("<|{role}|>{content}"));
            }
            s.push_str("<|assistant|><think></think>");
            tok.encode(&s)
        }
    }
}

/// Nemotron-H chat format (from its `chat_template.jinja`): plain **ChatML** —
/// each turn is `<|im_start|>{role}\n{content}<|im_end|>\n`, with no conversation
/// opener and no default system turn (the template emits a system block only when
/// the caller supplies one). The generation prompt is `<|im_start|>assistant\n`
/// followed by the template's `enable_thinking` branch; we emit the disabled form
/// `<think></think>` so the model answers directly, matching what the GLM path does.
///
/// Getting here matters: without this arm Nemotron fell through to the GLM branch
/// and was served `[gMASK]<sop><|user|>…` — control tokens absent from its vocab.
/// `<|im_start|>`/`<|im_end|>` are single added-vocab ids (10/11), and 11 is already
/// in `stop_ids` via `generation_config.json`, so turn termination needs no extra
/// wiring — only the prompt was wrong.
fn build_chat_prompt_nemotron(tok: &Tokenizer, messages: &[Json]) -> Vec<i32> {
    tok.encode(&nemotron_chat_string(messages))
}

/// The ChatML assembly for [`build_chat_prompt_nemotron`], split out so the exact
/// marker layout is unit-testable without loading a 67 GB model's tokenizer.
fn nemotron_chat_string(messages: &[Json]) -> String {
    let mut s = String::new();
    for m in messages {
        let o = match m.as_object() {
            Some(o) => o,
            None => continue,
        };
        let role = o.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = o.get("content").and_then(|v| v.as_str()).unwrap_or("");
        s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n<think></think>");
    s
}

/// MiniMax-M2 chat format (from its `chat_template.jinja`): `]~!b[` opens the
/// conversation, each turn is `]~b]<role>\n<content>[e~[` with roles
/// system/user/ai, and the trailing `]~b]ai\n` is the generation prompt. M2 is a
/// reasoning model — it emits `<think>…</think>` before its answer on its own, so we
/// do not force an empty think block. Generation halts at `[e~[` (200020), which
/// `Config::load` folds into `stop_ids` from `generation_config.json`.
fn build_chat_prompt_minimax(tok: &Tokenizer, messages: &[Json]) -> Vec<i32> {
    let role_tag = |r: &str| match r {
        "system" => "system",
        "assistant" => "ai",
        _ => "user",
    };
    let mut s = String::from("]~!b[");
    let first_is_system = messages
        .first()
        .and_then(|m| m.as_object())
        .and_then(|o| o.get("role"))
        .and_then(|v| v.as_str())
        == Some("system");
    if !first_is_system {
        s.push_str("]~b]system\nYou are a helpful assistant. Your name is MiniMax-M2.7 and is built by MiniMax.[e~[\n");
    }
    for m in messages {
        let o = match m.as_object() {
            Some(o) => o,
            None => continue,
        };
        let role = o.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = o.get("content").and_then(|v| v.as_str()).unwrap_or("");
        s.push_str(&format!("]~b]{}\n{}[e~[\n", role_tag(role), content));
    }
    s.push_str("]~b]ai\n");
    tok.encode(&s)
}

/// Maple ChatML (`chat_template.jinja`): `<|im_start|>{role}\n{content}<|im_end|>\n` per
/// message, ending with `<|im_start|>assistant\n<think>\n`.
///
/// Two details that are easy to get wrong and produce plausible output either way:
///
/// - The generation prompt **opens** a `<think>` block rather than closing an empty one.
///   Maple is a reasoning model, so a completion begins mid-reasoning and emits `</think>`
///   before its answer. GLM's template does the opposite (`<think></think>` to suppress
///   reasoning), and the shared fallthrough here is GLM's — so a Maple served by the
///   default arm gets GLM's control tokens, which are not even in its vocabulary.
/// - A **uniform** loop is correct here even though the template has two branches for
///   `system`. A leading system message is emitted by the preamble and skipped by the
///   loop (`'user' or (system and not loop.first)` is false for it); a later one is
///   emitted by the loop. Both spell it `<|im_start|>system\n…<|im_end|>\n`, so each
///   system turn renders exactly once, identically, either way.
///
/// Tools are not rendered: this server has no tool-call path, and emitting the tools
/// preamble with nothing to fill it would change the prompt for no benefit.
fn build_chat_prompt_maple(tok: &Tokenizer, messages: &[Json]) -> Vec<i32> {
    let mut s = String::new();
    for m in messages {
        let o = match m.as_object() {
            Some(o) => o,
            None => continue,
        };
        let role = o.get("role").and_then(|v| v.as_str()).unwrap_or("user");
        let content = o.get("content").and_then(|v| v.as_str()).unwrap_or("");
        // The template renders only these three roles and drops anything else; mirror
        // that rather than silently relabelling an unknown role as `user`.
        if !matches!(role, "system" | "user" | "assistant") {
            continue;
        }
        s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n<think>\n");
    tok.encode(&s)
}

/// Non-streaming: generate everything, then send one JSON object.
#[allow(clippy::too_many_arguments)]
fn block_completion(
    stream: &mut TcpStream,
    model: &Model,
    provider: &Provider,
    tok: &Tokenizer,
    prompt: &[i32],
    max_tokens: usize,
    id: &str,
    model_id: &str,
    object: &str,
    chat: bool,
    kv: &mut KvCache,
) {
    // `COLI_PROFILE=1`: per-REQUEST phase breakdown. The engine's counters are global
    // running totals, which suits `gen` (one run per process) and tells the server nothing
    // — so snapshot around the request and print the difference.
    //
    // This existed nowhere until now, and its absence is exactly why the f32-IO serve
    // regression could be bounded by A/B (per-token, not the forward math, not the KV size,
    // not threading) without ever being NAMED. `serve` is the production path; it should
    // not be the one path with no instrument.
    let prof0 = colibri_engine::profile_snapshot();
    let wall0 = std::time::Instant::now();
    let seq = match colibri_engine::generate_greedy(model, kv, provider, prompt, max_tokens) {
        Ok(s) => s,
        Err(e) => {
            send_json(
                stream,
                500,
                &format!(
                    "{{\"error\":{{\"message\":{},\"type\":\"internal_error\"}}}}",
                    jstr(&e.to_string())
                ),
            );
            return;
        }
    };
    if std::env::var("COLI_PROFILE").ok().as_deref() == Some("1") {
        let d = colibri_engine::profile_snapshot().since(&prof0);
        let ms = |u: u64| u as f64 / 1e3;
        let n = seq.len().max(1) as f64;
        eprintln!(
            "[serve-profile] {} tok in {:.0} ms | attn {:.0} | moe {:.0} (load {:.0}) | dense {:.0} \
             | embed {:.0} | logits {:.0} || per-tok {:.2} ms (attn {:.2} moe {:.2} logits {:.2})",
            seq.len(),
            wall0.elapsed().as_secs_f64() * 1e3,
            ms(d.attn_us), ms(d.moe_us), ms(d.expert_load_us), ms(d.dense_us),
            ms(d.embed_us), ms(d.logits_us),
            wall0.elapsed().as_secs_f64() * 1e3 / n,
            ms(d.attn_us) / n, ms(d.moe_us) / n, ms(d.logits_us) / n,
        );
    }

    // Drop the trailing stop token (e.g. GLM's `<|user|>`) from the visible text —
    // generation halts right after emitting it, so it's always the last token. The
    // streaming path already excludes it; keep the two consistent.
    let full = &seq[prompt.len()..];
    let hit_stop = full.last().is_some_and(|t| model.cfg.stop_ids.contains(t));
    let cont = if hit_stop {
        &full[..full.len() - 1]
    } else {
        full
    };
    let text = tok.decode(cont);
    let finish = if hit_stop { "stop" } else { "length" };
    let choice = if chat {
        format!("{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":{}}}", jstr(&text), jstr(finish))
    } else {
        format!(
            "{{\"index\":0,\"text\":{},\"finish_reason\":{}}}",
            jstr(&text),
            jstr(finish)
        )
    };
    let usage = format!(
        "{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}",
        prompt.len(),
        cont.len(),
        prompt.len() + cont.len()
    );
    let body = format!(
        "{{\"id\":{},\"object\":{},\"model\":{},\"choices\":[{}],\"usage\":{}}}",
        jstr(id),
        jstr(object),
        jstr(model_id),
        choice,
        usage
    );
    send_json(stream, 200, &body);
}

/// Streaming: emit an SSE chunk per token (the OpenAI delta protocol). Aborts
/// generation if the client disconnects (a chunk write fails).
#[allow(clippy::too_many_arguments)]
fn stream_completion(
    stream: &mut TcpStream,
    model: &Model,
    provider: &Provider,
    tok: &Tokenizer,
    prompt: &[i32],
    max_tokens: usize,
    id: &str,
    model_id: &str,
    object: &str,
    chat: bool,
    kv: &mut KvCache,
) {
    let chunk_obj = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    // SSE response headers.
    let headers = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/event-stream\r\n\
        Cache-Control: no-cache\r\n\
        Connection: close\r\n\
        Access-Control-Allow-Origin: *\r\n\r\n";
    if stream.write_all(headers.as_bytes()).is_err() {
        return;
    }

    // Decode the growing continuation each step and emit the *new* suffix, so
    // multi-byte tokens never split a UTF-8 boundary mid-chunk.
    let mut out_ids: Vec<i32> = Vec::with_capacity(max_tokens);
    let mut sent = String::new();
    let mut finish = "length";

    let _ = colibri_engine::generate_stream(model, kv, provider, prompt, max_tokens, |t| {
        if model.cfg.stop_ids.contains(&t) {
            finish = "stop";
            return true; // deliver nothing for the stop token; loop ends after this
        }
        out_ids.push(t);
        let full = tok.decode(&out_ids);
        if full.len() <= sent.len() {
            return true; // no new complete text yet
        }
        let delta = &full[sent.len()..];
        let payload = if chat {
            format!("{{\"role\":\"assistant\",\"content\":{}}}", jstr(delta))
        } else {
            jstr(delta) // /v1/completions puts the string directly in "text"
        };
        let field = if chat { "delta" } else { "text" };
        let chunk = format!(
            "{{\"id\":{},\"object\":{},\"model\":{},\"choices\":[{{\"index\":0,\"{}\":{},\"finish_reason\":null}}]}}",
            jstr(id),
            jstr(chunk_obj),
            jstr(model_id),
            field,
            payload
        );
        let ok = write_sse(stream, &chunk);
        if ok {
            sent = full;
        }
        ok // false → client gone → stop generating
    });

    // Terminal chunk carrying finish_reason, then the OpenAI [DONE] sentinel.
    let last = if chat {
        format!(
            "{{\"id\":{},\"object\":{},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":{}}}]}}",
            jstr(id), jstr(object_chunk(chat, object)), jstr(model_id), jstr(finish)
        )
    } else {
        format!(
            "{{\"id\":{},\"object\":{},\"model\":{},\"choices\":[{{\"index\":0,\"text\":\"\",\"finish_reason\":{}}}]}}",
            jstr(id), jstr(object_chunk(chat, object)), jstr(model_id), jstr(finish)
        )
    };
    let _ = write_sse(stream, &last);
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
}

fn object_chunk(chat: bool, object: &str) -> &str {
    if chat {
        "chat.completion.chunk"
    } else {
        object
    }
}

fn write_sse(stream: &mut TcpStream, data: &str) -> bool {
    stream.write_all(b"data: ").is_ok()
        && stream.write_all(data.as_bytes()).is_ok()
        && stream.write_all(b"\n\n").is_ok()
        && stream.flush().is_ok()
}

/// Send a fixed JSON (or empty) body with CORS + `Content-Length`. Status 200/204/
/// 4xx/5xx.
fn send_json(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    };
    let resp = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: *\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

/// Minimal JSON string literal: wraps in quotes and escapes per RFC 8259.
fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn jstr_escapes() {
        assert_eq!(jstr("hi"), "\"hi\"");
        assert_eq!(jstr("a\"b"), "\"a\\\"b\"");
        assert_eq!(jstr("a\\b"), "\"a\\\\b\"");
        assert_eq!(jstr("line\nbreak"), "\"line\\nbreak\"");
        assert_eq!(jstr("tab\tx"), "\"tab\\tx\"");
        assert_eq!(jstr("\u{0007}"), "\"\\u0007\""); // bell → 
        assert_eq!(jstr("café ☕"), "\"café ☕\""); // multibyte passes through
    }

    /// Nemotron-H is ChatML, NOT the GLM markers it used to fall through to.
    /// Mirrors `chat_template.jinja`: no opener, no injected system turn, one
    /// `<|im_start|>{role}\n{content}<|im_end|>\n` per message, and a generation
    /// prompt whose `<think></think>` is the template's disabled-thinking branch.
    #[test]
    fn nemotron_chat_is_chatml() {
        // Built by parsing, so the test walks the same shape a real request does.
        let msgs = match Json::parse(
            r#"[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]"#,
        ) {
            Some(Json::Arr(a)) => a,
            other => panic!("expected a JSON array, got {other:?}"),
        };
        let s = nemotron_chat_string(&msgs);
        assert_eq!(
            s,
            "<|im_start|>system\nBe brief.<|im_end|>\n\
             <|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\n<think></think>"
        );
        // The regression this fixes: none of the GLM control tokens may appear.
        assert!(
            !s.contains("[gMASK]"),
            "GLM opener leaked into the Nemotron prompt"
        );
        assert!(
            !s.contains("<|user|>"),
            "GLM role markers leaked into the Nemotron prompt"
        );
    }

    /// No messages still yields a valid generation prompt (not an empty string),
    /// so a malformed request cannot make the model continue arbitrary text.
    #[test]
    fn nemotron_chat_empty_messages_still_prompts() {
        assert_eq!(
            nemotron_chat_string(&[]),
            "<|im_start|>assistant\n<think></think>"
        );
    }

    #[test]
    fn model_id_from_hf_cache_path() {
        assert_eq!(
            model_id_from(
                "/root/.cache/huggingface/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/abc123"
            ),
            "nvidia/GLM-5.2-NVFP4"
        );
        assert_eq!(model_id_from("/data/glm52-nvfp4/"), "glm52-nvfp4");
        assert_eq!(model_id_from("/model"), "model");
    }

    #[test]
    fn read_request_parses_post_with_body() {
        let raw = "POST /v1/chat/completions HTTP/1.1\r\n\
                   Host: localhost\r\n\
                   Content-Type: application/json\r\n\
                   Content-Length: 13\r\n\r\n\
                   {\"a\":\"hello\"}";
        let mut r = Cursor::new(raw.as_bytes());
        let req = read_request(&mut r).unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/v1/chat/completions");
        assert_eq!(req.body, "{\"a\":\"hello\"}");
    }

    #[test]
    fn read_request_get_no_body() {
        let raw = "GET /health HTTP/1.1\r\nHost: x\r\n\r\n";
        let mut r = Cursor::new(raw.as_bytes());
        let req = read_request(&mut r).unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/health");
        assert!(req.body.is_empty());
    }

    #[test]
    fn read_request_content_length_case_insensitive() {
        let raw = "POST /v1/completions HTTP/1.1\r\ncontent-length: 3\r\n\r\nabc";
        let mut r = Cursor::new(raw.as_bytes());
        let req = read_request(&mut r).unwrap();
        assert_eq!(req.body, "abc");
    }

    #[test]
    fn read_request_empty_is_none() {
        let mut r = Cursor::new(&b""[..]);
        assert!(read_request(&mut r).is_none());
    }

    #[test]
    fn parse_ctx_units() {
        assert_eq!(parse_ctx("32768"), Some(32768));
        assert_eq!(parse_ctx("32k"), Some(32768));
        assert_eq!(parse_ctx("128K"), Some(131072));
        assert_eq!(parse_ctx("1m"), Some(1024 * 1024));
        assert_eq!(parse_ctx("0.5m"), Some(512 * 1024));
        assert_eq!(parse_ctx("  8k "), Some(8192));
        assert_eq!(parse_ctx("nope"), None);
    }

    #[test]
    fn ctx_clamp_to_model_max() {
        // COLI_CTX request is bounded by the model's max_position_embeddings.
        let model_max = 1_048_576usize;
        assert_eq!(parse_ctx("2m").unwrap().clamp(1, model_max), model_max);
        assert_eq!(parse_ctx("128k").unwrap().clamp(1, model_max), 131072);
    }
}
