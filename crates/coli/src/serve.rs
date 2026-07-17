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
use std::time::Duration;

use colibri_engine::{ExpertCache, KvCache, Model, ShardsExpertProvider};
use colibri_json::Json;
use colibri_tokenizer::Tokenizer;

/// Default listen port; overridden by a positional arg or `COLI_PORT`.
const DEFAULT_PORT: u16 = 8080;
/// Default number of tokens generated when a request omits `max_tokens`.
const DEFAULT_MAX_TOKENS: usize = 128;
/// Default served context length (prompt + completion) when `COLI_CTX` is unset.
/// GLM-5.2 supports up to 1M positions, but the KV cache is ~175 KB/token
/// (~5.6 GB at 32K), so we cap to a memory-safe default and let `COLI_CTX` raise
/// it as far as the model max.
const DEFAULT_CTX: usize = 32_768;
/// Tokens generated per warm-up prompt (enough to route a spread of experts).
const WARMUP_TOKENS: usize = 8;

/// How many copies of the KV cache to reserve for. `KvCache` lazily allocates a
/// full-size device-side shadow per layer (`model.rs`'s `DeviceKv`) on the GPU decode
/// path — and on GB10's unified memory that shadow comes out of the *same* pool as
/// the host copy, so under CUDA the KV genuinely costs twice.
#[cfg(feature = "cuda")]
const KV_COPIES: u64 = 2;
#[cfg(not(feature = "cuda"))]
const KV_COPIES: u64 = 1;

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

/// KV-cache bytes per token: two f32 latent/rotary buffers per layer.
fn kv_bytes_per_token(cfg: &colibri_core::Config) -> usize {
    cfg.n_layers as usize * (cfg.kv_lora as usize + cfg.qk_rope as usize) * 4
}

type Provider<'a> = ExpertCache<ShardsExpertProvider<'a>>;

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

    // Port: a leading bare integer arg, else COLI_PORT, else the default.
    let mut rest = &args[3.min(args.len())..];
    let port = match rest.first().and_then(|s| s.parse::<u16>().ok()) {
        Some(p) => {
            rest = &rest[1..];
            p
        }
        None => std::env::var("COLI_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_PORT),
    };

    // Warm-up prompts: the remaining positional args as one prompt, plus each
    // `|`-separated entry of COLI_WARMUP.
    let mut warmups: Vec<String> = Vec::new();
    if !rest.is_empty() {
        warmups.push(rest.join(" "));
    }
    if let Ok(w) = std::env::var("COLI_WARMUP") {
        warmups.extend(w.split('|').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()));
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
    // Served context length (prompt + completion). The model's hard ceiling is
    // `max_position_embeddings`; the served value is COLI_CTX (else a memory-safe
    // default), clamped to that ceiling. Requests are validated against it.
    //
    // Computed *before* the expert-cache budget: the budget has to reserve the KV
    // this window can allocate, and KV is sized from ctx_len.
    let model_max = if model.cfg.max_ctx > 0 { model.cfg.max_ctx as usize } else { usize::MAX };
    let ctx_len = std::env::var("COLI_CTX")
        .ok()
        .and_then(|s| parse_ctx(&s))
        .unwrap_or(DEFAULT_CTX)
        .clamp(1, model_max);

    // Worst-case KV for the served window — a single full-context request allocates
    // exactly this, and the expert cache must not have already eaten it.
    let kv_reserve = (kv_bytes_per_token(&model.cfg) as u64)
        .saturating_mul(ctx_len as u64)
        .saturating_mul(KV_COPIES);
    let budget = crate::ram_budget_reserving(kv_reserve);
    let gib = (1u64 << 30) as f64;
    if budget == u64::MAX {
        // No /proc/meminfo (non-Linux dev box): the budget is unbounded, and printing
        // it as a number renders "17179869184 GB". Say what actually happened.
        println!(
            "[serve] expert cache: unbounded (no MemAvailable to budget from) \
             — set COLI_RAM_GB to cap it"
        );
    } else {
        println!(
            "[serve] expert cache: {:.0} GB budget (reserved {:.1} GB KV for {} ctx{}) \
             — set COLI_RAM_GB to override",
            budget as f64 / gib,
            kv_reserve as f64 / gib,
            ctx_len,
            if KV_COPIES > 1 { ", incl. device shadow" } else { "" }
        );
    }

    let usage_path =
        std::env::var("COLI_USAGE").unwrap_or_else(|_| format!("{snap}/.coli_usage"));
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
    if let Some(topn) = crate::prefetch_topn() {
        provider.enable_prefetch(topn);
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
        eprintln!("[serve] warm-up {}/{}: {} tokens", i + 1, warmups.len(), ids.len());
        let mut kv = mk_kv(model, ids.len() + WARMUP_TOKENS);
        if let Err(e) = colibri_engine::generate_greedy(model, &mut kv, &*provider, &ids, WARMUP_TOKENS) {
            eprintln!("[serve] warm-up failed: {e}");
        }
    }
    if !warmups.is_empty() {
        let s = provider.stats();
        eprintln!("[serve] warm-up done: {} experts resident ({:.1} GB)", s.resident, s.bytes as f64 / (1u64 << 30) as f64);
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
    // Handle SIGINT/SIGTERM so the server (often PID 1 under Docker) stops on Ctrl-C or
    // `docker stop` instead of hanging until SIGKILL. Nonblocking accept + a poll of
    // the shutdown flag is what lets the blocking loop below actually notice it.
    crate::install_shutdown_handlers();
    if let Err(e) = listener.set_nonblocking(true) {
        eprintln!("[serve] warning: set_nonblocking failed ({e}); Ctrl-C may be slow");
    }
    println!(
        "[serve] coli {} — OpenAI-compatible server on http://{addr}  (model: {model_id})",
        crate::version_string()
    );
    let kv_at_ctx = kv_bytes_per_token(&model.cfg).saturating_mul(ctx_len) as f64 / (1u64 << 30) as f64;
    let model_max_str =
        if model.cfg.max_ctx > 0 { model.cfg.max_ctx.to_string() } else { "unknown".to_string() };
    println!(
        "[serve]   context length: {ctx_len} tokens (model max {model_max_str}; up to {:.1} GB KV) — set COLI_CTX to change",
        kv_at_ctx
    );
    println!("[serve]   POST /v1/chat/completions   POST /v1/completions   GET /v1/models   GET /health");

    // Scan the ConnectX/RoCE fabric and print the other Sparks we can see, so the
    // operator can verify the multi-node wiring at startup, then keep beaconing so
    // peers that start later discover this node too. COLI_DISCOVER_SECS=0 skips.
    let disc_secs = std::env::var("COLI_DISCOVER_SECS")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(3.0);
    if disc_secs > 0.0 {
        let rank: u32 = std::env::var("COLI_NODE_RANK").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let d = colibri_cluster::discover(rank, port, Duration::from_secs_f64(disc_secs));
        let _ = colibri_cluster::discovery::print_report(&d, &mut std::io::stdout());
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
            // No pending connection: nap briefly, then re-check SHUTDOWN. 100 ms of
            // accept latency is nothing next to a multi-second generation, and it
            // bounds how long Ctrl-C takes to be noticed.
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
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

fn mk_kv(model: &Model, max_t: usize) -> KvCache {
    KvCache::for_model(model, max_t)
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
        ("POST", "/v1/completions") => {
            complete(&mut stream, model, provider, tok, model_id, &req.body, false, ctx_len)
        }
        ("POST", "/v1/chat/completions") => {
            complete(&mut stream, model, provider, tok, model_id, &req.body, true, ctx_len)
        }
        ("OPTIONS", _) => send_json(&mut stream, 204, ""),
        _ => send_json(&mut stream, 404, "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}"),
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
    Some(Request { method, path, body: String::from_utf8_lossy(&body).into_owned() })
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
            Some(m) => build_chat_prompt(tok, m),
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
    if ids.is_empty() {
        send_json(stream, 400, "{\"error\":{\"message\":\"empty prompt\",\"type\":\"invalid_request_error\"}}");
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

    let object = if chat { "chat.completion" } else { "text_completion" };
    let id = format!("cmpl-{}", ids.len().wrapping_mul(2654435761) ^ max_tokens);
    let mut kv = mk_kv(model, ids.len() + max_tokens);

    if stream_mode {
        stream_completion(stream, model, provider, tok, &ids, max_tokens, &id, model_id, object, chat, &mut kv);
    } else {
        block_completion(stream, model, provider, tok, &ids, max_tokens, &id, model_id, object, chat, &mut kv);
    }
}

/// Official GLM-5.2 chat template (byte-matches `chat_template.jinja`, mirrored
/// from the C reference): `[gMASK]<sop>` then `<|role|>{content}` per message with
/// **no** separators, ending with `<|assistant|><think></think>` — the empty think
/// block disables reasoning so the model answers directly. The control tokens
/// (`<|user|>`, `<|assistant|>`, …) are added-vocab entries, so encoding the
/// assembled string resolves them to their ids exactly as the C engine does.
fn build_chat_prompt(tok: &Tokenizer, messages: &[Json]) -> Vec<i32> {
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
    let seq = match colibri_engine::generate_greedy(model, kv, provider, prompt, max_tokens) {
        Ok(s) => s,
        Err(e) => {
            send_json(stream, 500, &format!("{{\"error\":{{\"message\":{},\"type\":\"internal_error\"}}}}", jstr(&e.to_string())));
            return;
        }
    };
    // Drop the trailing stop token (e.g. GLM's `<|user|>`) from the visible text —
    // generation halts right after emitting it, so it's always the last token. The
    // streaming path already excludes it; keep the two consistent.
    let full = &seq[prompt.len()..];
    let hit_stop = full.last().is_some_and(|t| model.cfg.stop_ids.contains(t));
    let cont = if hit_stop { &full[..full.len() - 1] } else { full };
    let text = tok.decode(cont);
    let finish = if hit_stop { "stop" } else { "length" };
    let choice = if chat {
        format!("{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":{}}},\"finish_reason\":{}}}", jstr(&text), jstr(finish))
    } else {
        format!("{{\"index\":0,\"text\":{},\"finish_reason\":{}}}", jstr(&text), jstr(finish))
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
    let chunk_obj = if chat { "chat.completion.chunk" } else { "text_completion" };
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

    #[test]
    fn model_id_from_hf_cache_path() {
        assert_eq!(
            model_id_from("/root/.cache/huggingface/hub/models--mateogrgic--GLM-5.2-colibri-int4-with-int8-mtp/snapshots/abc123"),
            "mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp"
        );
        assert_eq!(model_id_from("/data/glm52-int4/"), "glm52-int4");
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
