//! `coli` — colibrì command-line entry point.
//!
//! Port target: the `main()` dispatch in `c/glm.c` plus the `c/coli` launcher.
//! The subcommands that depend on the not-yet-ported forward pass print an
//! honest "pending" message; `tokenize` and `config` already work end-to-end
//! against the ported crates.

use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};

mod serve;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Set by the SIGINT/SIGTERM handler; polled by the long-running server loops
/// (`serve`'s accept loop, `worker`'s park loop) so they stop instead of hanging.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// True once a shutdown signal has been received.
pub(crate) fn shutdown_requested() -> bool {
    SHUTDOWN.load(Ordering::SeqCst)
}

/// Signal handler. First signal → request graceful shutdown; a second (impatient
/// operator, or a graceful stop that's wedged mid-request) → immediate `_exit`.
///
/// Async-signal-safe: an atomic swap and `_exit` are the only operations, both on the
/// POSIX allowlist. `std::process::exit` is NOT safe here — it runs atexit hooks that
/// can deadlock if the signal interrupted an allocation — so the hard path uses
/// `_exit`, which the kernel guarantees is safe from a handler.
#[cfg(unix)]
extern "C" fn on_shutdown_signal(_sig: libc::c_int) {
    if SHUTDOWN.swap(true, Ordering::SeqCst) {
        unsafe { libc::_exit(130) };
    }
}

/// Install SIGINT/SIGTERM handlers so the long-running servers stop cleanly.
///
/// This is also what makes shutdown work **as PID 1 in a container**: the kernel
/// discards a signal sent to PID 1 unless PID 1 has installed a handler for it, so
/// without this `docker stop` (SIGTERM) and Ctrl-C (SIGINT) are ignored, the server
/// loop blocks forever, and the terminal never returns — the reported bug. The
/// entrypoint `exec`s `coli`, so `coli` really is PID 1.
///
/// No `SA_RESTART`: a signal landing during a blocking syscall should return `EINTR`
/// so the loop wakes and checks [`shutdown_requested`], rather than silently resuming.
#[cfg(unix)]
pub(crate) fn install_shutdown_handlers() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_shutdown_signal as extern "C" fn(libc::c_int) as usize;
        libc::sigemptyset(&mut sa.sa_mask);
        sa.sa_flags = 0;
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
    }
}

#[cfg(not(unix))]
pub(crate) fn install_shutdown_handlers() {}

/// Exit on shutdown, skipping the Drop of the model / expert cache / CUDA context.
///
/// Measured on the box (`COLI_DISCOVER_SECS=0`, so no startup-scan confound): the loop
/// notices the signal in ~50ms, then exit takes ~2s. That ~2s is the **kernel**
/// reclaiming this ~60 GB process's mappings (resident weights + 40 GB expert cache +
/// KV + device shadow); it is NOT userspace atexit — `libc::_exit(0)` was measured no
/// faster (2.6s vs 2.0s, within noise) — so there is nothing to skip by going lower
/// than `process::exit`. Drop *is* worth skipping (redundant explicit frees on a dying
/// process), which is the one thing this does. ~2s sits well inside docker's 10s grace.
pub(crate) fn shutdown_exit() -> ! {
    std::process::exit(0)
}

/// Source revision this binary was built from (see `build.rs`). Printed by `version`
/// and by the `serve`/`worker` banners: a container image built from stale source
/// looks identical at runtime otherwise, and on a cluster every node should show the
/// same value.
const BUILD_REV: &str = env!("COLI_BUILD_REV");

/// `v0.1.0 (abc1234)` — the identity string for logs and `version`.
pub(crate) fn version_string() -> String {
    format!("v{VERSION} ({BUILD_REV})")
}

fn usage() {
    eprintln!(
        r#"colibrì (SpeedyColibri, Rust port) v{VERSION} ({BUILD_REV})
  tiny engine, immense model

USAGE:
  coli <command> [args]

COMMANDS:
  cluster [seconds]        scan the ConnectX/RoCE fabric for other Sparks  [working]
  serve <snap> [port] [warm-up prompt...]  OpenAI-compatible HTTP server  [working]
  worker <snap> [port]     expert-shard server for a peer node (multi-node)  [working]
  bench <snap>             throughput benchmark        [pending]
  convert <src-snap> <out-snap>  FP8/NVFP4 -> int4 container converter  [working]
  probe <snap>             print snapshot format (container|fp8|nvfp4|unknown)  [working]
  qerr <src-snap> [bits] [n] [experts|resident]  requant error vs the FP8 source  [working]
  tokenize <tok.json> <text>   encode/decode round-trip   [working]
  config <snap>            print parsed hyperparameters   [working]
  load <snap>              load dense weights, print structure  [working]
  gen <snap> [ids...]      greedy-generate from token ids       [working]
  tf <snap> <ids...>       teacher-forcing argmax per position  [working]
  ppl <snap> <text-file> [n]   perplexity on held-out text (quality yardstick)  [working]
  capacity <snap> [ram_gb] expert residency / RAM planning      [working]
  loadbench <snap> [n] [layer]  decompose warm per-expert load cost  [working]
  repack <snap> <out> [n]  repack experts into n core-sharded binary files [working]
  backend                  show the selected compute backend (cpu/cuda)   [working]
  version                  print version
  help                     show this help

The <snap> is a model snapshot directory (config.json + *.safetensors).
See PORTING.md for the C->Rust port status."#
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");

    match cmd {
        "version" | "--version" | "-V" => {
            println!("coli {}", version_string());
            ExitCode::SUCCESS
        }
        "help" | "--help" | "-h" => {
            usage();
            ExitCode::SUCCESS
        }
        "tokenize" => cmd_tokenize(&args),
        "config" => cmd_config(&args),
        "load" => cmd_load(&args),
        "gen" => cmd_gen(&args),
        "tf" => cmd_tf(&args),
        "ppl" => cmd_ppl(&args),
        "capacity" => cmd_capacity(&args),
        "loadbench" => cmd_loadbench(&args),
        "repack" => cmd_repack(&args),
        "shard-export" => cmd_shard_export(&args),
        "shard-serve" => cmd_shard_serve(&args),
        "shard-pull" => cmd_shard_pull(&args),
        "backend" => cmd_backend(),
        "cluster" => cmd_cluster(&args),
        "worker" => cmd_worker(&args),
        "serve" => serve::cmd_serve(&args),
        "convert" => cmd_convert(&args),
        "probe" => cmd_probe(&args),
        "qerr" => cmd_qerr(&args),
        "bench" => {
            eprintln!("coli {cmd}: not yet ported. See PORTING.md for status.");
            ExitCode::from(2)
        }
        other => {
            eprintln!("coli: unknown command '{other}'\n");
            usage();
            ExitCode::from(2)
        }
    }
}

/// `coli tokenize <tokenizer.json> <text...>` — encode the text, print the ids,
/// and verify decode round-trips. Exercises the ported tokenizer end-to-end.
fn cmd_tokenize(args: &[String]) -> ExitCode {
    let tok_path = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli tokenize <tokenizer.json> <text...>");
            return ExitCode::from(2);
        }
    };
    let text = args.get(3..).map(|s| s.join(" ")).unwrap_or_default();

    let tok = match colibri_tokenizer::Tokenizer::load(tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("coli tokenize: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ids = tok.encode(&text);
    let decoded = tok.decode(&ids);
    println!("ids ({}): {:?}", ids.len(), ids);
    println!("decoded: {decoded:?}");
    if decoded == text {
        println!("round-trip: ok");
        ExitCode::SUCCESS
    } else {
        println!("round-trip: MISMATCH (expected {text:?})");
        ExitCode::FAILURE
    }
}

/// `coli probe <snap>` — print the snapshot's format on stdout, one word:
/// `container` (already ours — serve directly), `fp8` / `nvfp4` (needs `convert`),
/// or `unknown`. Scripting hook for the container entrypoint.
fn cmd_probe(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli probe <snapshot-dir>");
            return ExitCode::from(2);
        }
    };
    match colibri_engine::detect_format(snap) {
        Ok(f) => {
            println!("{}", f.as_str());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("coli probe: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `coli qerr <src-snapshot> [bits] [n]` — what re-quantizing the source at `bits`
/// costs, per resident tensor, measured against the checkpoint's own values.
///
/// The converter reads block-scaled FP8 and re-quantizes with its own per-row scales;
/// a native path would pass the source bytes through untouched. This scores what that
/// round trip costs. Reads a strided sample of tensors; no conversion, no GPU.
///
/// Reports weight-reconstruction error only — not perplexity, not throughput. A lower
/// number here does not imply a better model or a faster one.
fn cmd_qerr(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli qerr <src-snapshot> [bits=8] [n=8]");
            return ExitCode::from(2);
        }
    };
    let scheme = match args.get(3).map(|s| s.as_str()) {
        Some("nvfp4") => colibri_engine::Scheme::Nvfp4,
        Some(s) => match s.parse::<u32>() {
            Ok(b) => colibri_engine::Scheme::Int(b),
            Err(_) => {
                eprintln!("coli qerr: bits must be a number or `nvfp4`, got {s:?}");
                return ExitCode::from(2);
            }
        },
        None => colibri_engine::Scheme::Int(8),
    };
    let label = match scheme {
        colibri_engine::Scheme::Nvfp4 => "nvfp4 (e2m1 + ue4m3/16)".to_string(),
        colibri_engine::Scheme::Int(b) => format!("{b}-bit per-row int"),
    };
    let limit: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);
    let experts = matches!(args.get(5).map(|s| s.as_str()), Some("experts" | "x"));
    let n_layers = std::env::var("COLI_NLAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(78usize);
    let pop = if experts { "routed experts" } else { "resident" };

    match colibri_engine::quant_error(snap, scheme, n_layers, limit, experts) {
        Ok(errs) if errs.is_empty() => {
            eprintln!("coli qerr: no {pop} 2-D weights found in {snap}");
            ExitCode::FAILURE
        }
        Ok(errs) => {
            println!("requant error, {label} vs the source's own values [{pop}]");
            println!("{:>9} {:>9} {:>8}  tensor", "rms_rel", "max_rel", "snr_dB");
            let mut worst = 0f64;
            let mut sum = 0f64;
            for e in &errs {
                println!(
                    "{:>9.5} {:>9.3} {:>8.1}  {} [{}x{}]",
                    e.rms_rel, e.max_rel, e.snr_db, e.name, e.o, e.i
                );
                worst = worst.max(e.rms_rel);
                sum += e.rms_rel;
            }
            println!(
                "\nmean rms_rel {:.5} over {} tensors; worst {:.5}",
                sum / errs.len() as f64,
                errs.len(),
                worst
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("coli qerr: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `coli convert <src-snapshot> <out-snapshot>` — rewrite a block-scaled FP8 or
/// modelopt-NVFP4 GLM-5.2 snapshot as the colibrì int4/int8 container the engine loads.
///
/// Bit-widths default to the measured sweet spot — **8-bit resident, 4-bit experts**
/// (`ebits=8 xbits=4 io_bits=8`): 7.9x better perplexity than all-int4 (6.189 vs
/// 48.665) for ~33% throughput. Override via `COLI_EBITS` / `COLI_IO_BITS` /
/// `COLI_XBITS` / `COLI_NLAYERS`; see `ConvertOpts`.
fn cmd_convert(args: &[String]) -> ExitCode {
    let (indir, outdir) = match (args.get(2), args.get(3)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("usage: coli convert <fp8|nvfp4-snapshot-dir> <output-snapshot-dir>");
            eprintln!("  env: COLI_EBITS(8) COLI_IO_BITS(8) COLI_XBITS(4) COLI_NLAYERS(78) COLI_KEEP_INDEXER(0)");
            return ExitCode::from(2);
        }
    };
    let env_u32 = |k: &str, d: u32| {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    // NB: xbits does NOT default to ebits. It used to (mirroring the reference
    // converter), which would make the new `ebits=8` default silently mean 8/8 —
    // doubling bytes-per-token and needing a 0.74 TB container that does not fit on
    // the box, for quality that fixing attention already recovers.
    let opts = colibri_engine::ConvertOpts {
        ebits: env_u32("COLI_EBITS", 8),
        io_bits: env_u32("COLI_IO_BITS", 8),
        xbits: env_u32("COLI_XBITS", 4),
        n_layers: env_u32("COLI_NLAYERS", 78) as usize,
        // COLI_KEEP_INDEXER=1 keeps the DSA lightning-indexer weights so the container
        // can run DSA sparse attention (dropped by default, matching the reference).
        keep_indexer: env_u32("COLI_KEEP_INDEXER", 0) != 0,
        // COLI_XFP8=1 emits routed experts as per-row e4m3 fp8 (8-bit) instead of xbits
        // int — preserves the source FP8 precision for the tiled FP8 expert kernel.
        xfp8: env_u32("COLI_XFP8", 0) != 0,
    };

    eprintln!(
        "[convert] {indir} -> {outdir}  (ebits={} io_bits={} xbits={} xfp8={} n_layers={})",
        opts.ebits, opts.io_bits, opts.xbits, opts.xfp8, opts.n_layers
    );
    let t0 = std::time::Instant::now();
    let res = colibri_engine::convert_snapshot(indir, outdir, opts, |fi, n, st| {
        let secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "[convert] shard {:>3}/{n}  quantized={} f32={} skipped={}  out={:.1} GB  {:.0}s",
            fi + 1,
            st.tensors_quantized,
            st.tensors_f32,
            st.tensors_skipped,
            st.bytes_out as f64 / 1e9,
            secs
        );
    });
    match res {
        Ok(st) => {
            eprintln!(
                "[convert] done: {} shards, {} weights quantized, {} f32, {} skipped, {:.1} GB out, {:.0}s",
                st.shards_written,
                st.tensors_quantized,
                st.tensors_f32,
                st.tensors_skipped,
                st.bytes_out as f64 / 1e9,
                t0.elapsed().as_secs_f64()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("[convert] error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `coli load <snap>` — materialize the dense weights and print a structural
/// summary. Streams no experts; this just proves the snapshot loads.
fn cmd_load(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli load <snapshot-dir>");
            return ExitCode::from(2);
        }
    };
    match colibri_engine::load_model(snap) {
        Ok(m) => {
            let dense = m.layers.iter().filter(|l| !l.sparse).count();
            let sparse = m.layers.len() - dense;
            println!("loaded {} layers ({dense} dense, {sparse} MoE)", m.layers.len());
            println!(
                "embed [{},{}] fmt={}  lm_head [{},{}]  final_norm[{}]",
                m.embed.o,
                m.embed.i,
                m.embed.fmt_code,
                m.lm_head.o,
                m.lm_head.i,
                m.final_norm.len()
            );
            println!(
                "dense bits={}  expert bits={}  has_dsa={}  has_mtp={}",
                m.dbits, m.ebits, m.has_dsa, m.has_mtp
            );
            println!("(routed experts stream on demand; not resident)");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("coli load: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `coli gen <snap> [id...]` — load a model and greedy-generate from the given
/// token ids (default `[1]`), printing the continuation ids. Runs the full CPU
/// forward pass; experts stream from the snapshot on demand.
fn cmd_gen(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli gen <snapshot-dir> [token_id ...]");
            return ExitCode::from(2);
        }
    };
    let prompt: Vec<i32> = args
        .get(3..)
        .map(|a| a.iter().filter_map(|s| s.parse().ok()).collect())
        .filter(|v: &Vec<i32>| !v.is_empty())
        .unwrap_or_else(|| vec![1]);

    // Bit-widths default to int4; overridable via env for the C-vs-Rust
    // validation harness (e.g. COLI_DBITS=16 for the exact f32 path).
    let envbits = |k: &str, d: u32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let opts = colibri_engine::LoadOptions {
        dbits: envbits("COLI_DBITS", 4),
        ebits: envbits("COLI_EBITS", 4),
    };
    let model = match colibri_engine::load_model_with(snap, opts) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("coli gen: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Leak to 'static so an optional background prefetch loader can hold the cache
    // (the process owns the model for its lifetime anyway).
    let model: &'static colibri_engine::Model = Box::leak(Box::new(model));
    let n_new = envbits("COLI_NGEN", 16) as usize;

    // COLI_PRELOAD: parallel-load experts into RAM, then serve with no per-token
    // disk I/O. If the value is a dir with a manifest.json it uses repacked
    // shards; otherwise (e.g. `COLI_PRELOAD=1`) it reads directly from the
    // original model in parallel — no repack, no second copy on disk.
    if let Ok(v) = std::env::var("COLI_PRELOAD") {
        let repacked = std::path::Path::new(&v).join("manifest.json").exists();
        if repacked {
            return cmd_gen_preload_repacked(model, &v, &prompt, n_new);
        }
        return cmd_gen_preload_direct(model, &prompt, n_new);
    }

    // Usage history first — both the hot-aware sharding and AUTOPIN read it.
    let usage_path =
        std::env::var("COLI_USAGE").unwrap_or_else(|_| format!("{snap}/.coli_usage"));
    let mut history = colibri_engine::UsageHistory::load(&usage_path).unwrap_or_default();

    // Cluster-aware expert sharding. Single-node keeps every expert local; multi-node
    // splits experts by owner so `moe()` computes this node's shard and dispatches the
    // rest to their owners over the transport. Wiring this into `gen` (not just
    // `serve`) is what makes the token-identity oracle — `coli gen <snap> 100 200 300`
    // — runnable across nodes, which is the RDMA-A correctness gate.
    let cluster = colibri_cluster::ClusterConfig::from_env();
    let sharding = if cluster.is_single_node() {
        colibri_cluster::ExpertSharding::single(model.cfg.n_experts as u32)
    } else {
        build_sharding(&cluster, model.cfg.n_experts as u32, &history)
    };

    // Resident expert cache, restricted to this node's shard (the provider refuses a
    // non-owned expert, so a routing bug fails loudly instead of streaming a peer's
    // expert off disk). Budget from COLI_RAM_GB, else the auto cap.
    let base = colibri_engine::ShardsExpertProvider::with_sharding(
        &model.shards,
        &model.cfg,
        model.ebits as u32,
        sharding.clone(),
        cluster.this_node,
    );
    let budget = ram_budget();
    let provider = std::sync::Arc::new(colibri_engine::ExpertCache::new(base, budget));
    if let Some(topn) = prefetch_topn() {
        provider.enable_prefetch(topn);
        println!("prefetch: speculative next-layer prefetch on (top-{topn}/layer)");
    }

    // Multi-node: install the expert-parallel context so `moe()` dispatches non-local
    // experts over TCP/RoCE. verify_peers() handshakes every worker up front, so a
    // mismatched sharding map or a peer that isn't up fails here rather than
    // mid-generation. Single-node leaves the context unset (everything local).
    if !cluster.is_single_node() {
        let peers = match cluster_peers(&cluster) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("coli gen: {e}");
                return ExitCode::FAILURE;
            }
        };
        let owned = sharding.count_for(cluster.this_node);
        let transport =
            colibri_cluster::TcpTransport::new(cluster.this_node, peers, sharding.fingerprint());
        use colibri_cluster::Transport as _;
        if let Err(e) = transport.verify_peers() {
            eprintln!("coli gen: cluster verification failed: {e}");
            return ExitCode::FAILURE;
        }
        eprintln!(
            "[gen] expert-parallel: {} nodes, rank {} owns {} experts, sharding {:#018x}",
            cluster.num_nodes,
            cluster.this_node.0,
            owned,
            sharding.fingerprint()
        );
        colibri_engine::set_cluster(colibri_engine::ClusterCtx {
            sharding: sharding.clone(),
            transport: Box::new(transport),
        });
    }

    // AUTOPIN the hottest experts, restricted to this node's shard in a cluster.
    let own_history = owned_history(&history, &sharding, cluster.this_node);
    apply_autopin(&provider, &own_history, budget);

    // `for_model` sizes the KV for the MTP head too (rows = n_layers + has_mtp);
    // hand-rolling `KvCache::new(n_layers, ..)` would under-allocate on an MTP model.
    let mut kv = colibri_engine::KvCache::for_model(model, prompt.len() + n_new);
    match colibri_engine::generate_greedy(model, &mut kv, &*provider, &prompt, n_new) {
        Ok(seq) => {
            let cont: Vec<i32> = seq[prompt.len()..].to_vec();
            println!("prompt: {prompt:?}");
            println!("generated ({} tok): {cont:?}", cont.len());
            let s = provider.stats();
            println!(
                "expert cache: {} resident ({:.1} MB), {} pinned, {} hits / {} misses, {} evictions",
                s.resident,
                s.bytes as f64 / (1024.0 * 1024.0),
                provider.pinned_count(),
                s.hits,
                s.misses,
                s.evictions
            );
            #[cfg(feature = "cuda")]
            {
                let (n, bytes, evict, budget) = colibri_engine::gpu::ffn_cache_stats();
                let gib = 1u64 << 30;
                println!(
                    "gpu: {} matmuls, {} fused expert FFNs, {} attention cores",
                    colibri_engine::gpu::matmul_count(),
                    colibri_engine::gpu::ffn_count(),
                    colibri_engine::gpu::attn_count()
                );
                println!(
                    "gpu vram (experts): {} resident ({:.1} GB / {:.0} GB budget), {} evictions",
                    n,
                    bytes as f64 / gib as f64,
                    budget as f64 / gib as f64,
                    evict
                );
            }
            // Persist this session's selections into the usage history for the
            // next run's warm-up.
            history.merge(&provider.usage_snapshot());
            if let Err(e) = history.save(&usage_path) {
                eprintln!("coli gen: could not save usage history to {usage_path}: {e}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("coli gen: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `coli backend` — report the selected compute backend. On a CUDA build/host it
/// prints the GPU and its free/total memory; otherwise CPU.
fn cmd_backend() -> ExitCode {
    let b = colibri_backend::autoselect();
    println!("backend: {} ({:?})", b.name(), b.device());
    #[cfg(feature = "cuda")]
    {
        let n = colibri_backend::cuda::device_count();
        println!("cuda devices: {n}");
        for d in 0..n {
            if let Some((free, total)) = colibri_backend::cuda::mem_info(d) {
                let gib = 1u64 << 30;
                println!(
                    "  device {d}: {:.1} / {:.1} GB free",
                    free as f64 / gib as f64,
                    total as f64 / gib as f64
                );
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    println!("(built without `cuda` — CPU only; rebuild with --features cuda on a DGX Spark)");
    ExitCode::SUCCESS
}

/// `coli cluster [seconds]` — scan the ConnectX/RoCE fabric and print the other
/// DGX Sparks it can see (local links + peers, whether or not they run colibrì),
/// for the operator to verify the multi-node wiring. Advertises this node's
/// `COLI_NODE_RANK` / `COLI_PORT` in its beacon so peers see them too.
fn cmd_cluster(args: &[String]) -> ExitCode {
    let window = args
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .map(|s| std::time::Duration::from_secs_f64(s.clamp(0.5, 60.0)))
        .unwrap_or_else(|| std::time::Duration::from_secs(4));
    let rank: u32 = std::env::var("COLI_NODE_RANK").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let port: u16 = std::env::var("COLI_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8080);

    eprintln!("[cluster] scanning the fabric for {:.0}s (UDP :{}) ...", window.as_secs_f64(), colibri_cluster::discovery::DISC_PORT);
    let d = colibri_cluster::discover(rank, port, window);
    let mut out = std::io::stdout();
    if let Err(e) = colibri_cluster::discovery::print_report(&d, &mut out) {
        eprintln!("coli cluster: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Port a `worker` binds (and a `serve` peer connects to) for expert exchange.
fn expert_port() -> u16 {
    std::env::var("COLI_EXPERT_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(48800)
}

/// Peer addresses for every *other* rank in the cluster, validated.
///
/// A multi-node run needs an address for every rank but our own. Missing entries are
/// fatal: with an empty/partial peer map the startup handshake has nothing to talk to,
/// so it "verifies" vacuously and the failure only surfaces on the first token as
/// `no address for node N`. Catch it here instead.
pub(crate) fn cluster_peers(
    cluster: &colibri_cluster::ClusterConfig,
) -> Result<std::collections::HashMap<colibri_cluster::NodeId, std::net::SocketAddr>, String> {
    let peers = parse_peers()?;
    let missing = missing_peer_ranks(cluster.num_nodes, cluster.this_node, &peers);
    if !missing.is_empty() {
        return Err(format!(
            "COLI_NUM_NODES={} but COLI_PEERS has no address for rank(s) {missing:?}. \
             Every other node needs one: COLI_PEERS=\"<rank>=<host:port>,...\" \
             (e.g. COLI_PEERS=\"1=192.168.100.10:48800\").",
            cluster.num_nodes
        ));
    }
    Ok(peers)
}

/// Ranks other than `this` with no configured address. Non-empty ⇒ the cluster is
/// misconfigured and must not start.
fn missing_peer_ranks(
    num_nodes: u32,
    this: colibri_cluster::NodeId,
    peers: &std::collections::HashMap<colibri_cluster::NodeId, std::net::SocketAddr>,
) -> Vec<u32> {
    (0..num_nodes)
        .filter(|&r| colibri_cluster::NodeId(r) != this)
        .filter(|&r| !peers.contains_key(&colibri_cluster::NodeId(r)))
        .collect()
}

/// Parse `COLI_PEERS="1=host:port,2=host:port"` into a node→address map (the
/// expert servers of the other nodes).
fn parse_peers() -> Result<std::collections::HashMap<colibri_cluster::NodeId, std::net::SocketAddr>, String> {
    use std::net::ToSocketAddrs;
    let mut map = std::collections::HashMap::new();
    let s = std::env::var("COLI_PEERS").unwrap_or_default();
    for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let (rank, addr) = entry
            .split_once('=')
            .ok_or_else(|| format!("bad COLI_PEERS entry '{entry}' (want rank=host:port)"))?;
        let rank: u32 = rank.trim().parse().map_err(|_| format!("bad rank in '{entry}'"))?;
        let sa = addr
            .trim()
            .to_socket_addrs()
            .map_err(|e| format!("resolve '{addr}': {e}"))?
            .next()
            .ok_or_else(|| format!("no address for '{addr}'"))?;
        map.insert(colibri_cluster::NodeId(rank), sa);
    }
    Ok(map)
}

/// `coli worker <snap> [port]` — a headless expert-shard server for a peer node.
/// Loads the model, then answers `serve`'s expert-exchange requests over TCP
/// (RoCE Ethernet): for each request it computes `Σ w·expert(x)` over the experts
/// this node owns and returns the partial MoE sum. `COLI_NODE_RANK`/`COLI_NUM_NODES`
/// set which shard this node owns; only that shard is ever loaded/cached.
fn cmd_worker(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: coli worker <snapshot-dir> [port]  (set COLI_NODE_RANK / COLI_NUM_NODES)");
            return ExitCode::from(2);
        }
    };
    let cluster = colibri_cluster::ClusterConfig::from_env();
    let port = args.get(3).and_then(|s| s.parse().ok()).unwrap_or_else(expert_port);

    // Leak the model to 'static so the (process-lifetime) expert server thread can
    // hold a persistent cache of this node's shard.
    let model: &'static colibri_engine::Model = match colibri_engine::load_model(&snap) {
        Ok(m) => Box::leak(Box::new(m)),
        Err(e) => {
            eprintln!("coli worker: {e}");
            return ExitCode::FAILURE;
        }
    };
    // The map comes first: it decides which experts this node may load. The same
    // history feeds it, so its fingerprint must match every other node's — the
    // driver's handshake enforces that, and the printed values let you eyeball it.
    let usage_path =
        std::env::var("COLI_USAGE").unwrap_or_else(|_| format!("{snap}/.coli_usage"));
    let history = colibri_engine::UsageHistory::load(&usage_path).unwrap_or_default();
    let sharding = build_sharding(&cluster, model.cfg.n_experts as u32, &history);
    let owned = sharding.count_for(cluster.this_node);

    // Ownership enforced at the load layer: the driver should only ever send us
    // experts we own, so a request for someone else's is a bug worth surfacing
    // rather than quietly serving from disk.
    let base = colibri_engine::ShardsExpertProvider::with_sharding(
        &model.shards,
        &model.cfg,
        model.ebits as u32,
        sharding.clone(),
        cluster.this_node,
    );
    let budget = ram_budget();
    let provider = std::sync::Arc::new(colibri_engine::ExpertCache::new(base, budget));
    if let Some(topn) = prefetch_topn() {
        provider.enable_prefetch(topn);
    }

    // AUTOPIN our shard's hot experts, before the provider moves into the server
    // closure. Filtered to what we own — the history covers the whole cluster.
    let own_history = owned_history(&history, &sharding, cluster.this_node);
    apply_autopin(&provider, &own_history, budget);
    // Peers must present this exact fingerprint on connect, or they're refused before
    // any activation is computed — disagreeing maps corrupt results silently.
    let fingerprint = sharding.fingerprint();
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    // Attention head-slice handler (tensor-parallel attention): compute this node's
    // heads over the shipped layer input + the driver's DSA selection. Stateless — a
    // fresh scratch KV per request (single-shot prefill), the real layer's resident
    // weights via `model.layers`.
    let attn = move |req: &colibri_cluster::AttnRequest| {
        let mut outputs = vec![0.0f32; req.n_tokens * req.hidden];
        if let Some(l) = model.layers.get(req.layer as usize) {
            colibri_engine::compute_attention_partial(
                &model.cfg,
                l,
                &req.activations,
                req.n_tokens,
                req.pos_base as usize,
                req.h_start as usize,
                req.h_count as usize,
                &req.sel,
                &mut outputs,
            );
        } else {
            eprintln!("[worker] attention request for out-of-range layer {}", req.layer);
        }
        colibri_cluster::AttnResponse { outputs, n_tokens: req.n_tokens, hidden: req.hidden }
    };
    let bound = match colibri_cluster::serve_cluster(addr, fingerprint, move |req| {
        match colibri_engine::compute_experts_partial(
            &*provider,
            req.layer as usize,
            &req.experts,
            &req.weights,
            &req.activations,
            req.n_tokens,
            req.hidden,
        ) {
            Ok(outputs) => {
                colibri_cluster::ExpertResponse { outputs, n_tokens: req.n_tokens, hidden: req.hidden }
            }
            Err(e) => {
                eprintln!("[worker] expert compute error: {e}");
                colibri_cluster::ExpertResponse {
                    outputs: vec![0.0; req.n_tokens * req.hidden],
                    n_tokens: req.n_tokens,
                    hidden: req.hidden,
                }
            }
        }
    }, attn) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("coli worker: bind {addr}: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "[worker] coli {} — rank {} of {} — serving {} experts on {} (TCP/RoCE)",
        version_string(),
        cluster.this_node.0,
        cluster.num_nodes,
        owned,
        bound
    );
    // Advertise on the discovery beacon so `cluster` scans see this worker.
    colibri_cluster::discovery::spawn_beacon(cluster.this_node.0, port);
    // serve_experts runs the accept loop on its own thread; this thread just waits for
    // a shutdown signal. Handle SIGINT/SIGTERM (as PID 1 under Docker the kernel
    // ignores them without a handler) and poll instead of parking forever, so
    // `docker stop` / Ctrl-C actually return the terminal.
    install_shutdown_handlers();
    while !shutdown_requested() {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    eprintln!("[worker] shutdown signal received — stopping");
    shutdown_exit()
}

/// Direct parallel preload from the original model (no repack). One thread per
/// core reads a contiguous, offset-ordered slice of the experts into RAM.
fn cmd_gen_preload_direct(model: &colibri_engine::Model, prompt: &[i32], n_new: usize) -> ExitCode {
    let cores = colibri_engine::default_num_files();
    let budget = ram_budget();
    let t0 = std::time::Instant::now();
    let store = match colibri_engine::preload_parallel(
        &model.shards,
        &model.cfg,
        model.ebits as u32,
        cores,
        budget,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("coli gen: preload_parallel: {e}");
            return ExitCode::FAILURE;
        }
    };
    let secs = t0.elapsed().as_secs_f64().max(1e-9);
    let bpe = colibri_engine::capacity::bytes_per_expert(
        model.cfg.hidden as u64,
        model.cfg.moe_inter as u64,
        model.ebits as u32,
    );
    let gb = (store.len() as u64 * bpe) as f64 / (1u64 << 30) as f64;
    println!(
        "preload (direct from model, {cores} threads): {} experts in {:.2}s ({:.2} GB, {:.2} GB/s)",
        store.len(),
        secs,
        gb,
        gb / secs
    );
    finish_gen(model, &store, prompt, n_new)
}

/// Shared tail: build the KV cache, generate, print.
fn finish_gen(
    model: &colibri_engine::Model,
    provider: &impl colibri_engine::ExpertProvider,
    prompt: &[i32],
    n_new: usize,
) -> ExitCode {
    let mut kv = colibri_engine::KvCache::for_model(&model, prompt.len() + n_new);
    match colibri_engine::generate_greedy(model, &mut kv, provider, prompt, n_new) {
        Ok(seq) => {
            println!("prompt: {prompt:?}");
            println!("generated ({} tok): {:?}", seq.len() - prompt.len(), &seq[prompt.len()..]);
            #[cfg(feature = "cuda")]
            {
                println!(
                    "gpu: {} matmuls, {} fused expert FFNs, {} attention cores",
                    colibri_engine::gpu::matmul_count(),
                    colibri_engine::gpu::ffn_count(),
                    colibri_engine::gpu::attn_count()
                );
                let (n, bytes, evict, budget) = colibri_engine::gpu::ffn_cache_stats();
                let gib = 1u64 << 30;
                println!(
                    "gpu vram (experts): {} resident ({:.1} GB / {:.0} GB budget), {} evictions",
                    n,
                    bytes as f64 / gib as f64,
                    budget as f64 / gib as f64,
                    evict
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("coli gen: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `coli gen` with a repacked shard dir — parallel-load the shards, then generate.
fn cmd_gen_preload_repacked(
    model: &colibri_engine::Model,
    pre_dir: &str,
    prompt: &[i32],
    n_new: usize,
) -> ExitCode {
    use std::path::Path;
    let manifest = match colibri_engine::Manifest::load(Path::new(pre_dir).join("manifest.json")) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("coli gen: preload manifest: {e}");
            return ExitCode::FAILURE;
        }
    };
    // per-shard budget so total ~= the RAM budget (loads "as many as fit").
    let per_file = (ram_budget() / manifest.num_files.max(1) as u64).max(1);
    let t0 = std::time::Instant::now();
    let store = match colibri_engine::PreloadStore::load(&manifest, Path::new(pre_dir), per_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("coli gen: preload: {e}");
            return ExitCode::FAILURE;
        }
    };
    let secs = t0.elapsed().as_secs_f64().max(1e-9);
    let gb = manifest.total_bytes() as f64 / (1u64 << 30) as f64;
    let loaded_gb = gb * store.len() as f64 / manifest.experts.len().max(1) as f64;
    println!(
        "preload (repacked, {} shards): {} experts in {:.2}s ({:.2} GB, {:.2} GB/s across cores)",
        manifest.num_files,
        store.len(),
        secs,
        loaded_gb,
        loaded_gb / secs
    );
    finish_gen(model, &store, prompt, n_new)
}

/// `coli repack <snap> <out_dir> [num_files]` — repack every routed expert into
/// `num_files` (default: CPU cores) contiguous binary shards + a manifest, for
/// The routed-expert id embedded in a tensor name
/// (`model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight[.qs]`), or `None`
/// for a non-expert (resident) tensor.
fn expert_id_of(name: &str) -> Option<u32> {
    const M: &str = ".mlp.experts.";
    let i = name.find(M)?;
    let rest = &name[i + M.len()..];
    let end = rest.find('.').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Tensor names for node `me`'s shard: every resident (non-expert) tensor plus the
/// routed experts it owns under `sharding`. Shared by `shard-export` and `shard-serve`.
fn select_shard_names(
    shards: &colibri_safetensors::Shards,
    sharding: &colibri_cluster::ExpertSharding,
    me: colibri_cluster::NodeId,
    n_experts: u32,
) -> Vec<String> {
    shards
        .tensors()
        .iter()
        .filter(|t| match expert_id_of(&t.name) {
            Some(e) => e < n_experts && sharding.is_local(me, e),
            None => true,
        })
        .map(|t| t.name.clone())
        .collect()
}

// Little-endian framing for the shard-distribute wire protocol.
fn wr_u64<W: std::io::Write>(w: &mut W, v: u64) -> std::io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn wr_u32<W: std::io::Write>(w: &mut W, v: u32) -> std::io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn rd_u64<R: std::io::Read>(r: &mut R) -> std::io::Result<u64> { let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn rd_u32<R: std::io::Read>(r: &mut R) -> std::io::Result<u32> { let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn wr_path<W: std::io::Write>(w: &mut W, s: &str) -> std::io::Result<()> { wr_u32(w, s.len() as u32)?; w.write_all(s.as_bytes()) }
fn rd_path<R: std::io::Read>(r: &mut R) -> std::io::Result<String> {
    let n = rd_u32(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// `coli shard-serve <src-snap> [port]` — stream each connecting peer *its* expert
/// shard over **raw TCP** (no SSH crypto → full RoCE bandwidth), reading the source
/// **in parallel**. The peer runs `coli shard-pull`. Serves until killed. Replaces
/// the slow single-threaded `shard-export` + `rsync -e ssh` bootstrap (~0.35 GB/s,
/// ~20 min) with a direct source→peer-disk stream (no intermediate file).
///
/// Wire form (all little-endian): peer sends `nodes,rank`; server replies
/// `n_files`, then per file `path_len,path,size,<size bytes>`. Each file is either a
/// complete `out-NNNNN.safetensors` the server builds on the fly, or a metadata file
/// (config/tokenizer) copied verbatim — so the receiver is a dumb byte sink.
fn cmd_shard_serve(args: &[String]) -> ExitCode {
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let src = match args.get(2) {
        Some(s) => s.clone(),
        None => {
            eprintln!("usage: coli shard-serve <src-snap> [port]");
            return ExitCode::from(2);
        }
    };
    let port: u16 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(48900);
    let cfg = match colibri_core::Config::load(&src) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("shard-serve: config: {e}");
            return ExitCode::FAILURE;
        }
    };
    let n_experts = cfg.n_experts as u32;
    let shards = match colibri_safetensors::Shards::open(&src) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("shard-serve: open {src}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let read_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let listener = match std::net::TcpListener::bind(("0.0.0.0", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("shard-serve: bind :{port}: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("[shard-serve] {n_experts} experts, serving on 0.0.0.0:{port} ({read_threads} read threads)");
    for conn in listener.incoming() {
        let mut stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[shard-serve] accept: {e}");
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        let nodes = match rd_u32(&mut stream) { Ok(v) => v, Err(e) => { eprintln!("[shard-serve] handshake: {e}"); continue; } };
        let rank = match rd_u32(&mut stream) { Ok(v) => v, Err(_) => continue };
        if nodes < 1 || rank >= nodes {
            eprintln!("[shard-serve] bad request nodes={nodes} rank={rank}");
            continue;
        }
        let sharding = colibri_cluster::ExpertSharding::new(nodes, n_experts);
        let me = colibri_cluster::NodeId(rank);
        let names = select_shard_names(&shards, &sharding, me, n_experts);
        let items: Vec<&colibri_safetensors::StTensor> =
            names.iter().filter_map(|n| shards.find(n)).collect();
        // Group tensors into ~5 GB out-*.safetensors, same packing as write_subset.
        let max_file = 5_000_000_000u64;
        let mut groups: Vec<(usize, usize)> = Vec::new();
        {
            let mut i = 0;
            while i < items.len() {
                let start = i;
                let mut acc = 0u64;
                while i < items.len() && (i == start || acc + items[i].nbytes <= max_file) {
                    acc += items[i].nbytes;
                    i += 1;
                }
                groups.push((start, i));
            }
        }
        // Metadata files (config/tokenizer/generation_config) shipped verbatim.
        let mut meta: Vec<(String, std::path::PathBuf, u64)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&src) {
            for ent in rd.flatten() {
                let p = ent.path();
                let is_st = p.extension().map(|e| e == "safetensors").unwrap_or(false);
                let is_usage = p.file_name().map(|f| f == ".coli_usage").unwrap_or(false);
                if p.is_file() && !is_st && !is_usage {
                    if let (Some(fname), Ok(m)) =
                        (p.file_name().and_then(|f| f.to_str()).map(String::from), std::fs::metadata(&p))
                    {
                        meta.push((fname, p, m.len()));
                    }
                }
            }
        }
        let gb: f64 = (items.iter().map(|t| t.nbytes).sum::<u64>()
            + meta.iter().map(|(_, _, s)| *s).sum::<u64>()) as f64 / 1e9;
        eprintln!(
            "[shard-serve] peer rank {rank}/{nodes}: {} tensors, {} shard files + {} meta, {gb:.1} GB",
            items.len(), groups.len(), meta.len()
        );
        let t0 = std::time::Instant::now();
        let res = (|| -> std::io::Result<()> {
            let mut w = std::io::BufWriter::with_capacity(8 << 20, stream.try_clone()?);
            wr_u64(&mut w, (groups.len() + meta.len()) as u64)?;
            for (fi, &(start, end)) in groups.iter().enumerate() {
                let grp = &items[start..end];
                // safetensors header (relative data_offsets), then the tensor bytes.
                let mut header = String::from("{");
                let mut rel = 0u64;
                for (gi, t) in grp.iter().enumerate() {
                    if gi > 0 { header.push(','); }
                    let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
                    header.push_str(&format!(
                        "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                        t.name, t.dtype.safetensors_str(), shape.join(","), rel, rel + t.nbytes
                    ));
                    rel += t.nbytes;
                }
                header.push('}');
                let group_bytes: u64 = grp.iter().map(|t| t.nbytes).sum();
                let file_size = 8 + header.len() as u64 + group_bytes;
                wr_path(&mut w, &format!("out-{fi:05}.safetensors"))?;
                wr_u64(&mut w, file_size)?;
                w.write_all(&(header.len() as u64).to_le_bytes())?;
                w.write_all(header.as_bytes())?;
                // Parallel read of this group's tensors into ordered buffers, then send.
                let n = grp.len();
                let mut bufs: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
                let cursor = AtomicUsize::new(0);
                let (tx, rx) = std::sync::mpsc::channel::<(usize, Vec<u8>)>();
                std::thread::scope(|scope| {
                    let nt = read_threads.min(n).max(1);
                    for _ in 0..nt {
                        let tx = tx.clone();
                        let cursor = &cursor;
                        let shards = &shards;
                        scope.spawn(move || loop {
                            let i = cursor.fetch_add(1, Ordering::Relaxed);
                            if i >= grp.len() { break; }
                            let mut buf = vec![0u8; grp[i].nbytes as usize];
                            if shards.read_raw(&grp[i].name, &mut buf).is_ok() {
                                let _ = tx.send((i, buf));
                            }
                        });
                    }
                    drop(tx);
                    for (i, buf) in rx { bufs[i] = Some(buf); }
                });
                for b in &bufs {
                    match b {
                        Some(bytes) => w.write_all(bytes)?,
                        None => return Err(std::io::Error::new(std::io::ErrorKind::Other, "shard-serve: tensor read failed")),
                    }
                }
            }
            for (fname, path, size) in &meta {
                wr_path(&mut w, fname)?;
                wr_u64(&mut w, *size)?;
                let mut f = std::fs::File::open(path)?;
                std::io::copy(&mut f, &mut w)?;
            }
            w.flush()
        })();
        match res {
            Ok(()) => eprintln!(
                "[shard-serve] peer rank {rank} done: {gb:.1} GB in {:.1}s ({:.2} GB/s)",
                t0.elapsed().as_secs_f64(), gb / t0.elapsed().as_secs_f64().max(1e-9)
            ),
            Err(e) => eprintln!("[shard-serve] peer rank {rank} error: {e}"),
        }
    }
    ExitCode::SUCCESS
}

/// `coli shard-pull <out-dir> <host:port> --nodes N --rank R` — pull this node's
/// shard from a `coli shard-serve` peer over raw TCP, writing it to `out-dir` as a
/// self-contained snapshot. A dumb byte sink: the server frames complete files.
fn cmd_shard_pull(args: &[String]) -> ExitCode {
    use std::io::{Read, Write};
    let mut pos: Vec<&String> = Vec::new();
    let (mut nodes, mut rank): (u32, u32) = (0, u32::MAX);
    let mut it = args.iter().skip(2);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--nodes" => nodes = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--rank" => rank = it.next().and_then(|s| s.parse().ok()).unwrap_or(u32::MAX),
            _ => pos.push(a),
        }
    }
    let (out, addr) = match (pos.first(), pos.get(1)) {
        (Some(o), Some(a)) => (o.as_str(), a.as_str()),
        _ => {
            eprintln!("usage: coli shard-pull <out-dir> <host:port> --nodes N --rank R");
            return ExitCode::from(2);
        }
    };
    if nodes < 1 || rank >= nodes {
        eprintln!("shard-pull: need --nodes >= 1 and 0 <= --rank < nodes");
        return ExitCode::from(2);
    }
    let out_dir = std::path::Path::new(out);
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("shard-pull: mkdir {out}: {e}");
        return ExitCode::FAILURE;
    }
    let res = (|| -> std::io::Result<(usize, u64)> {
        let stream = std::net::TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        let mut w = stream.try_clone()?;
        wr_u32(&mut w, nodes)?;
        wr_u32(&mut w, rank)?;
        let mut r = std::io::BufReader::with_capacity(8 << 20, stream);
        let n_files = rd_u64(&mut r)?;
        let mut total = 0u64;
        for _ in 0..n_files {
            let path = rd_path(&mut r)?;
            let size = rd_u64(&mut r)?;
            let mut f = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(out_dir.join(&path))?);
            let mut left = size;
            let mut buf = vec![0u8; 8 << 20];
            while left > 0 {
                let want = (buf.len() as u64).min(left) as usize;
                r.read_exact(&mut buf[..want])?;
                f.write_all(&buf[..want])?;
                left -= want as u64;
            }
            f.flush()?;
            total += size;
        }
        Ok((n_files as usize, total))
    })();
    match res {
        Ok((nf, total)) => {
            eprintln!("shard-pull: received {nf} files, {:.1} GB → {out}", total as f64 / 1e9);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("shard-pull: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `coli shard-export <src-snap> <out-dir> --nodes N --rank R` — write a snapshot
/// containing ONLY node R's owned routed experts plus every resident (non-expert)
/// tensor, so a peer can load its shard from local disk instead of holding the full
/// model. This is the multispark distribution primitive: rank 0 exports each rank's
/// shard, then ships it over. Bytes are copied verbatim (the e4m3 container
/// round-trips), and non-safetensors files (config/tokenizer) are copied too. The
/// ownership map is the contiguous default — the same one the runtime uses, so the
/// exported experts are exactly what `ShardsExpertProvider` will ask this node for.
fn cmd_shard_export(args: &[String]) -> ExitCode {
    use std::path::Path;
    let mut pos: Vec<&String> = Vec::new();
    let (mut nodes, mut rank): (u32, u32) = (0, u32::MAX);
    let mut it = args.iter().skip(2);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--nodes" => nodes = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--rank" => rank = it.next().and_then(|s| s.parse().ok()).unwrap_or(u32::MAX),
            _ => pos.push(a),
        }
    }
    let (src, out) = match (pos.first(), pos.get(1)) {
        (Some(s), Some(o)) => (s.as_str(), o.as_str()),
        _ => {
            eprintln!("usage: coli shard-export <src-snap> <out-dir> --nodes N --rank R");
            return ExitCode::from(2);
        }
    };
    if nodes < 1 || rank >= nodes {
        eprintln!("shard-export: need --nodes >= 1 and 0 <= --rank < nodes (got nodes={nodes} rank={rank})");
        return ExitCode::from(2);
    }
    let cfg = match colibri_core::Config::load(src) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("shard-export: config load: {e}");
            return ExitCode::FAILURE;
        }
    };
    let n_experts = cfg.n_experts as u32;
    let shards = match colibri_safetensors::Shards::open(src) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("shard-export: open {src}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let sharding = colibri_cluster::ExpertSharding::new(nodes, n_experts);
    let me = colibri_cluster::NodeId(rank);

    let names = select_shard_names(&shards, &sharding, me, n_experts);
    let bytes: u64 = names.iter().filter_map(|n| shards.find(n)).map(|t| t.nbytes).sum();
    let n_exp = names.iter().filter(|n| expert_id_of(n).is_some()).count();
    eprintln!(
        "shard-export rank {rank}/{nodes}: {} tensors ({} resident + {n_exp} expert), {:.1} GB, owns {} experts",
        names.len(),
        names.len() - n_exp,
        bytes as f64 / 1e9,
        sharding.count_for(me),
    );
    // ~5 GB/file, matching the source snapshot's shard size.
    let out_path = Path::new(out);
    let files = match shards.write_subset(&names, out_path, 5_000_000_000) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("shard-export: write: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Copy the non-safetensors metadata (config.json, generation_config.json,
    // tokenizer*) so the shard is a self-contained, loadable snapshot.
    let mut copied = 0;
    if let Ok(rd) = std::fs::read_dir(src) {
        for ent in rd.flatten() {
            let p = ent.path();
            let is_st = p.extension().map(|e| e == "safetensors").unwrap_or(false);
            let is_usage = p.file_name().map(|f| f == ".coli_usage").unwrap_or(false);
            if p.is_file() && !is_st && !is_usage {
                if let Some(fname) = p.file_name() {
                    if std::fs::copy(&p, out_path.join(fname)).is_ok() {
                        copied += 1;
                    }
                }
            }
        }
    }
    eprintln!("shard-export: wrote {files} safetensors files + {copied} metadata files to {out}");
    ExitCode::SUCCESS
}

/// fast parallel preloading (`COLI_PRELOAD`).
fn cmd_repack(args: &[String]) -> ExitCode {
    use std::path::Path;
    let (snap, out) = match (args.get(2), args.get(3)) {
        (Some(s), Some(o)) => (s, o),
        _ => {
            eprintln!("usage: coli repack <snapshot-dir> <out-dir> [num_files]");
            return ExitCode::from(2);
        }
    };
    let num_files = args
        .get(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(colibri_engine::default_num_files);

    let model = match colibri_engine::load_model(snap) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("coli repack: {e}");
            return ExitCode::FAILURE;
        }
    };
    let provider =
        colibri_engine::ShardsExpertProvider::new(&model.shards, &model.cfg, model.ebits as u32);
    println!(
        "repacking experts into {num_files} shards (one per core: {} available)...",
        colibri_engine::default_num_files()
    );
    let t0 = std::time::Instant::now();
    match colibri_engine::repack(&provider, &model.cfg, Path::new(out), num_files) {
        Ok(m) => {
            let secs = t0.elapsed().as_secs_f64();
            let gb = m.total_bytes() as f64 / (1u64 << 30) as f64;
            println!(
                "repacked {} experts → {} shards, {:.1} GB in {:.1}s. manifest: {}/manifest.json",
                m.experts.len(),
                m.num_files,
                gb,
                secs,
                out
            );
            println!("run: COLI_PRELOAD={out} coli gen {snap} <ids...>");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("coli repack: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `coli loadbench <snap> [n_experts] [layer]` — decompose the *warm* (page-cache
/// hot) per-expert load cost. Steady-state decode is bound by expert loading even
/// when every byte is in the page cache, so this isolates where the time goes:
/// the chunked read's thread spawns, the fresh 18 MB allocation (mmap + zero-fill
/// page faults), the coalesced read itself, and the 3 small scale reads.
fn cmd_loadbench(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli loadbench <snapshot-dir> [n_experts] [layer]");
            return ExitCode::from(2);
        }
    };
    let n_experts: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    let cfg = match colibri_core::Config::load(snap) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("coli loadbench: {e}");
            return ExitCode::FAILURE;
        }
    };
    let shards = match colibri_safetensors::Shards::open(snap) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("coli loadbench: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (hidden, moe_inter) = (cfg.hidden as usize, cfg.moe_inter as usize);
    let wn = |l: usize, e: usize, suf: &str| {
        format!("model.layers.{l}.mlp.experts.{e}.{suf}.weight")
    };
    // Default to the first layer that actually has routed experts (GLM: layer 3).
    let layer = match args.get(4).and_then(|s| s.parse().ok()) {
        Some(l) => l,
        None => match (0..cfg.n_layers as usize).find(|&l| shards.has(&wn(l, 0, "gate_proj"))) {
            Some(l) => l,
            None => {
                eprintln!("coli loadbench: no routed experts found in snapshot");
                return ExitCode::FAILURE;
            }
        },
    };
    let n_experts = n_experts.min(cfg.n_experts as usize);
    let threads: usize = std::env::var("COLI_LOAD_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));

    // Per-expert byte counts (gate+up+down weights; scales are the tiny sidecars).
    let names = |e: usize| [wn(layer, e, "gate_proj"), wn(layer, e, "up_proj"), wn(layer, e, "down_proj")];
    let sizes: Vec<usize> = names(0).iter().map(|n| shards.nbytes(n) as usize).collect();
    let span: usize = sizes.iter().sum();
    let total_bytes = (span * n_experts) as f64;
    println!(
        "loadbench: layer {layer}, {n_experts} experts, {:.1} MB/expert, T={threads}",
        span as f64 / (1 << 20) as f64
    );

    // Warm the page cache: read every byte of the set once (results discarded).
    let t0 = std::time::Instant::now();
    for e in 0..n_experts {
        let nm = names(e);
        let nr: Vec<&str> = nm.iter().map(String::as_str).collect();
        if let Err(err) = shards.read_raw_shared(&nr, threads) {
            eprintln!("coli loadbench: warm-up read failed: {err}");
            return ExitCode::FAILURE;
        }
    }
    println!(
        "warm-up pass: {:.2} s ({:.2} GB/s cold-ish)\n",
        t0.elapsed().as_secs_f64(),
        total_bytes / t0.elapsed().as_secs_f64() / 1e9
    );

    println!("{:<34} {:>9} {:>10} {:>8}", "phase", "total ms", "ms/expert", "GB/s");
    let report = |name: &str, secs: f64, bytes: f64| {
        let gbs = if bytes > 0.0 { format!("{:.2}", bytes / secs / 1e9) } else { "-".into() };
        println!(
            "{:<34} {:>9.1} {:>10.3} {:>8}",
            name,
            secs * 1e3,
            secs * 1e3 / n_experts as f64,
            gbs
        );
        secs
    };

    // 1+2. Full production path, chunked (T) vs single-thread.
    let mut full = [0f64; 2];
    for (i, t) in [threads, 1].into_iter().enumerate() {
        let t0 = std::time::Instant::now();
        for e in 0..n_experts {
            let ex = colibri_engine::moe::load_expert(&shards, hidden, moe_inter, 4, layer, e, t)
                .expect("load_expert");
            std::hint::black_box(&ex);
        }
        full[i] = report(&format!("full load_expert (T={t})"), t0.elapsed().as_secs_f64(), total_bytes);
    }

    // 2b. Pooled batch: all experts through one continuously-streaming worker set
    //     (the COLI_READER_POOL path) vs the per-expert spawn/join above. Same
    //     work (full Experts, scales + QTensor), so this isolates the loader shape.
    {
        let eids: Vec<usize> = (0..n_experts).collect();
        let t0 = std::time::Instant::now();
        let exps =
            colibri_engine::moe::load_experts_batch(&shards, hidden, moe_inter, 4, layer, &eids, threads)
                .expect("load_experts_batch");
        std::hint::black_box(&exps);
        report(
            &format!("pooled batch load (T={threads})"),
            t0.elapsed().as_secs_f64(),
            total_bytes,
        );
    }

    // 3+4. Coalesced read only (fresh alloc + read; no scales, no QTensor).
    let mut read = [0f64; 2];
    for (i, t) in [threads, 1].into_iter().enumerate() {
        let t0 = std::time::Instant::now();
        for e in 0..n_experts {
            let nm = names(e);
            let nr: Vec<&str> = nm.iter().map(String::as_str).collect();
            let ws = shards.read_raw_shared(&nr, t).expect("read_raw_shared");
            std::hint::black_box(&ws);
        }
        read[i] = report(&format!("coalesced read, fresh alloc (T={t})"), t0.elapsed().as_secs_f64(), total_bytes);
    }

    // 5. Read into one REUSED, pre-faulted buffer (single-thread) — no allocation.
    let mut reused = vec![1u8; span];
    let t0 = std::time::Instant::now();
    for e in 0..n_experts {
        let nm = names(e);
        let mut off = 0;
        for (j, n) in nm.iter().enumerate() {
            shards.read_raw(n, &mut reused[off..off + sizes[j]]).expect("read_raw");
            off += sizes[j];
        }
        std::hint::black_box(&reused);
    }
    let reused_s = report("read into reused buffer (T=1)", t0.elapsed().as_secs_f64(), total_bytes);

    // 6. Fresh allocation + touch one byte per page — mmap/munmap churn + zero-fill
    //    faults, the allocation cost the read path pays before any byte arrives.
    let t0 = std::time::Instant::now();
    for _ in 0..n_experts {
        let mut v = Vec::<u8>::with_capacity(span);
        #[allow(clippy::uninit_vec)]
        unsafe {
            v.set_len(span)
        };
        let p = v.as_mut_ptr();
        let mut i = 0;
        while i < span {
            unsafe { p.add(i).write(1) };
            i += 4096;
        }
        std::hint::black_box(&v);
    }
    let alloc_s = report("fresh alloc + page-touch only", t0.elapsed().as_secs_f64(), total_bytes);

    // 7. Scale sidecar reads only (3 small preads + f32 convert, fresh vecs).
    let t0 = std::time::Instant::now();
    for e in 0..n_experts {
        for (n, o) in [
            (format!("{}.qs", wn(layer, e, "gate_proj")), moe_inter),
            (format!("{}.qs", wn(layer, e, "up_proj")), moe_inter),
            (format!("{}.qs", wn(layer, e, "down_proj")), hidden),
        ] {
            let mut s = vec![0f32; o];
            shards.read_f32(&n, &mut s).expect("read_f32");
            std::hint::black_box(&s);
        }
    }
    let scales_s = report("scale reads only (3x .qs)", t0.elapsed().as_secs_f64(), 0.0);

    // 8. Bare thread spawn/join of T no-op scoped threads — the fixed price
    //    pread_chunked pays per expert regardless of what the disk does.
    let nt = threads.min(span >> 20).max(1);
    let t0 = std::time::Instant::now();
    for _ in 0..n_experts {
        std::thread::scope(|s| {
            for _ in 0..nt {
                s.spawn(|| std::hint::black_box(0));
            }
        });
    }
    let spawn_s = report(&format!("thread spawn/join only ({nt} thr)"), t0.elapsed().as_secs_f64(), 0.0);

    let ms = |s: f64| s * 1e3 / n_experts as f64;
    println!("\nattribution (ms/expert, warm):");
    println!("  chunking delta (full T={threads} vs T=1): {:+.3}", ms(full[0]) - ms(full[1]));
    println!("  alloc cost (fresh vs reused read):        {:+.3}  (direct alloc+fault: {:.3})",
        ms(read[1]) - ms(reused_s), ms(alloc_s));
    println!("  scales + QTensor (full - read, T=1):      {:+.3}  (scales alone: {:.3})",
        ms(full[1]) - ms(read[1]), ms(scales_s));
    println!("  spawn overhead ({nt} threads):              {:.3}", ms(spawn_s));
    println!("  pure read, reused buf:                    {:.3}", ms(reused_s));
    ExitCode::SUCCESS
}

/// Held back from the expert cache on top of any caller-specific reserve: the CUDA
/// context and its workspaces, expert read buffers, activations and HTTP scratch, and
/// allocator slack. Mirrors the VRAM-side reserve in `colibri_engine::gpu::ffn_budget`
/// — on the Spark's unified pool both draw from the same ~121 GB.
///
/// This alone is **not** what keeps the box out of swap — see [`CACHE_CAP_DIVISOR`],
/// which is the real guard. Subtracting a constant from `MemAvailable` cannot express
/// "don't take so much that the kernel pages you out", because `MemAvailable` counts
/// the page cache as free: on a 121 GiB Spark it reads ~99 GiB once dense weights are
/// resident, so any small constant still yields a budget far past the cliff.
const WORKING_RESERVE: u64 = 10 << 30;

/// The expert cache is capped at `MemTotal / CACHE_CAP_DIVISOR`.
///
/// **Measured on a 121 GiB Spark, 8/4 model (~19 GiB resident), 12 diverse prompts,
/// counterbalanced order** (each config run at mirrored positions so a drift cannot
/// masquerade as a config effect — every earlier ascending-order sweep here was
/// uninterpretable for exactly that reason):
///
/// | `COLI_RAM_GB` | RSS   | swap  | tok/s |
/// |---------------|-------|-------|-------|
/// | 20            | 39 GB | 0     | 0.46  |
/// | 40            | 57 GB | 0     | 0.45  |
/// | 55            | 74 GB | 0     | 0.44  |
/// | 70            | 89 GB | 15 GB | 0.38  |
/// | 87 (old auto) | 95 GB | 15 GB | 0.11  |
///
/// Two facts set the rule. **The plateau is flat**: 20 GiB serves as fast as 55, so
/// a bigger cache buys nothing on diverse traffic — routed experts are barely reused
/// across unrelated prompts, so the hit rate stays near zero whatever the size.
/// **The cliff is a wall**: RSS tracks `budget + resident`, and once that crowds out
/// the page cache the kernel pages out the very cache we just filled, on an engine
/// whose whole cost is already disk I/O.
///
/// So the ceiling is chosen for margin, not throughput — there is no throughput to
/// win. `/3` lands at ~40 GiB here: mid-plateau, ~15 GiB clear of the last known-good
/// point and ~30 clear of the cliff. That margin absorbs what this table does not
/// cover — longer contexts, other tenants, larger resident footprints. Repeated or
/// shared-prefix traffic *does* reuse experts and would prefer a larger cache; `/3`
/// leaves room for that without approaching the wall.
const CACHE_CAP_DIVISOR: u64 = 3;

/// Floor so a small/busy box still gets a usable cache rather than 0 (a 0 budget
/// evicts every expert immediately and thrashes). Set `COLI_RAM_GB` explicitly if
/// even this doesn't fit.
const MIN_BUDGET: u64 = 4 << 30;

/// Expert-cache byte budget, reserving `reserve` bytes the caller knows it will
/// still allocate (e.g. `serve`'s KV cache) on top of [`WORKING_RESERVE`].
///
/// `COLI_RAM_GB` remains an **exact override** — the point of the default is to pick
/// a safe maximum automatically, and the knob is there to go lower (or higher, if you
/// know better).
///
/// Dense weights are deliberately *not* subtracted: `load_model_with` materializes
/// them eagerly and every caller budgets *after* the load, so `MemAvailable` has
/// already excluded them. Subtracting again would double-count ~11 GiB.
fn ram_budget_reserving(reserve: u64) -> u64 {
    if let Ok(gb) = std::env::var("COLI_RAM_GB") {
        if let Ok(g) = gb.parse::<u64>() {
            let asked = g << 30;
            // Exact override, but say so when it's past the measured cliff: 70 GiB on a
            // 121 GiB box swaps 15 GiB and costs ~20%, 85 costs ~4x. Silently obeying a
            // number that guarantees thrash is how the old default hurt people.
            if let Some(cap) = colibri_engine::total_ram_bytes().map(|t| t / CACHE_CAP_DIVISOR) {
                if asked > cap {
                    eprintln!(
                        "[warn] COLI_RAM_GB={g} exceeds the safe cache ceiling of {} GB \
                         (MemTotal/{CACHE_CAP_DIVISOR}); measured: budgets past ~55 GB on a \
                         121 GB box swap and lose throughput. Using {g} GB as asked.",
                        cap >> 30
                    );
                }
            }
            return asked;
        }
    }
    match colibri_engine::available_ram_bytes() {
        Some(avail) => budget_from(avail, reserve, colibri_engine::total_ram_bytes()),
        None => u64::MAX, // non-Linux: no /proc/meminfo, stay unbounded
    }
}

/// Pure arithmetic behind [`ram_budget_reserving`], split out to be testable.
///
/// Two independent guards, because they fail in different directions:
/// - subtracting the reserves keeps a *busy* box from overcommitting what's left;
/// - the [`CACHE_CAP_DIVISOR`] cap keeps an *idle* box from taking memory the kernel
///   needs for page cache. This is the one that matters in practice: on an idle Spark
///   `MemAvailable` reads ~99 GiB and the subtraction alone yields 87 GiB — measured
///   at 0.11 tok/s against 0.46 at 40 GiB.
///
/// Saturating on purpose: a plain `avail - reserve - WORKING_RESERVE` underflows on a
/// small box and wraps to ~16 EiB — an effectively unbounded budget, causing exactly
/// the OOM this exists to prevent.
fn budget_from(avail: u64, reserve: u64, total: Option<u64>) -> u64 {
    let by_avail = avail
        .saturating_sub(reserve)
        .saturating_sub(WORKING_RESERVE);
    // `total` is None only off-Linux, where there's no /proc/meminfo to cap against.
    let capped = match total {
        Some(t) => by_avail.min(t / CACHE_CAP_DIVISOR),
        None => by_avail,
    };
    capped.max(MIN_BUDGET)
}

/// Expert-cache byte budget for callers with nothing extra to reserve.
fn ram_budget() -> u64 {
    ram_budget_reserving(0)
}

/// Speculative-prefetch setting from `COLI_PREFETCH`: unset/`0` → off; `1` → on
/// with `COLI_PREFETCH_N` experts/layer (default 16); a bare number → that many.
/// Off by default and best left off with local-NVMe experts: a controlled A/B
/// regressed decode tok/s at every degree (speculative loads evict the working set
/// and contend for the saturated drive). Retained opt-in for the RDMA case where
/// the prefetch source is a peer's RAM. See `ExpertCache::enable_prefetch` and
/// `scripts/expert_prefetch_analysis.py`.
fn prefetch_topn() -> Option<usize> {
    // The prefill prefetch-ahead reuses the same background loader thread + channel
    // (it bypasses the predictor), so it must enable the loader even when COLI_PREFETCH
    // is off. `Some(0)` wires the loader with a no-op predictor.
    let ahead = std::env::var("COLI_PREFETCH_AHEAD").ok().as_deref() == Some("1");
    match std::env::var("COLI_PREFETCH").ok().as_deref() {
        None | Some("") | Some("0") => ahead.then_some(0),
        Some("1") => {
            Some(std::env::var("COLI_PREFETCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(16))
        }
        Some(v) => Some(v.parse().unwrap_or(16)),
    }
}

/// AUTOPIN sizing from `COLI_PIN_GB`: unset/`0` → off; `auto` → size to the knee of
/// the usage-coverage curve (pin the hot head, stream the tail); a number `N` → pin
/// up to `N` GB of the hottest experts.
pub(crate) enum PinMode {
    Off,
    Auto,
    Gb(u64),
}

pub(crate) fn pin_mode() -> PinMode {
    match std::env::var("COLI_PIN_GB").ok().as_deref().map(str::trim) {
        None | Some("") | Some("0") => PinMode::Off,
        Some(v) if v.eq_ignore_ascii_case("auto") => PinMode::Auto,
        Some(v) => match v.parse::<u64>() {
            Ok(0) | Err(_) => PinMode::Off,
            Ok(n) => PinMode::Gb(n),
        },
    }
}

/// Apply AUTOPIN to `provider` from the persistent usage `history`, honoring
/// `COLI_PIN_GB` (see [`pin_mode`]). `cache_budget` is the cache's byte budget, used
/// by the `auto` path to leave streaming headroom. Logs what it pinned. Shared by
/// `coli gen` and `coli serve`.
pub(crate) fn apply_autopin<P: colibri_engine::ExpertProvider>(
    provider: &colibri_engine::ExpertCache<P>,
    history: &colibri_engine::UsageHistory,
    cache_budget: u64,
) {
    let mode = pin_mode();
    if matches!(mode, PinMode::Off) {
        return;
    }
    if history.is_empty() {
        eprintln!(
            "hot-store: COLI_PIN_GB set but usage history is empty — it builds as you \
             run; nothing to pin yet"
        );
        return;
    }
    let gib = (1u64 << 30) as f64;
    match mode {
        PinMode::Off => unreachable!(),
        PinMode::Auto => match provider.warm_pin_auto(history, cache_budget) {
            Ok((n, bytes, cov)) => println!(
                "hot-store: AUTOPIN pinned {n} experts ({:.1} GB) at the usage-curve knee \
                 — {:.0}% of historical routing kept resident",
                bytes as f64 / gib,
                cov * 100.0
            ),
            Err(e) => eprintln!("coli: warm_pin_auto: {e}"),
        },
        PinMode::Gb(gb) => match provider.warm_pin(history, gb << 30) {
            Ok(n) => println!(
                "hot-store: pinned {n} experts from usage history ({} entries, {} selections)",
                history.len(),
                history.total()
            ),
            Err(e) => eprintln!("coli: warm_pin: {e}"),
        },
    }
}

/// The usage history restricted to the experts `this_node` owns.
///
/// Every node loads the *same* history — the hot-aware map is derived from it, so it
/// has to be identical — but each only ever computes its own shard. Pinning from the
/// unfiltered history would spend this node's cache on experts it is never asked for
/// (up to half of it on 2 nodes), and the provider's ownership gate rejects them
/// outright. Single-node owns everything, so this is a no-op there.
pub(crate) fn owned_history(
    history: &colibri_engine::UsageHistory,
    sharding: &colibri_cluster::ExpertSharding,
    this_node: colibri_cluster::NodeId,
) -> colibri_engine::UsageHistory {
    history.filter_experts(|eid| sharding.is_local(this_node, eid as u32))
}

/// Build the expert→node sharding for a multi-node run. `COLI_SHARD=hot` selects the
/// hot-aware, traffic-balanced map built from the shared usage `history` (spreads the
/// popular experts across nodes); anything else (or an empty history) uses contiguous
/// blocks. Logs the map fingerprint and the balance achieved — **all nodes must print
/// the same fingerprint**, or the activation exchange is misrouting.
pub(crate) fn build_sharding(
    cluster: &colibri_cluster::ClusterConfig,
    n_experts: u32,
    history: &colibri_engine::UsageHistory,
) -> colibri_cluster::ExpertSharding {
    let hot = std::env::var("COLI_SHARD")
        .ok()
        .is_some_and(|v| v.eq_ignore_ascii_case("hot"));
    if hot && !history.is_empty() {
        let weights = history.expert_weights(n_experts as usize);
        let sharding = cluster.expert_sharding_balanced(n_experts, &weights);
        let nw = sharding.node_weights(&weights);
        let (min, max) = (
            nw.iter().copied().min().unwrap_or(0),
            nw.iter().copied().max().unwrap_or(0),
        );
        let imbalance = if min > 0 { max as f64 / min as f64 } else { f64::INFINITY };
        println!(
            "sharding: hot-aware (traffic-balanced), fingerprint {:#018x} — per-node load \
             max/min {:.2}x (contiguous would cluster hot experts). Verify this fingerprint \
             matches on every node.",
            sharding.fingerprint(),
            imbalance
        );
        sharding
    } else {
        let sharding = cluster.expert_sharding(n_experts);
        if hot {
            eprintln!(
                "sharding: COLI_SHARD=hot but usage history is empty — falling back to \
                 contiguous. Build history with a warm-up run first."
            );
        }
        println!(
            "sharding: contiguous blocks, fingerprint {:#018x}",
            sharding.fingerprint()
        );
        sharding
    }
}

/// Parse a token count like `256k`, `1m`, or `262144`.
fn parse_ctx(s: &str) -> Option<u64> {
    let s = s.trim().to_lowercase();
    let (num, mul) = if let Some(n) = s.strip_suffix('k') {
        (n, 1024u64)
    } else if let Some(n) = s.strip_suffix('m') {
        (n, 1024 * 1024)
    } else {
        (s.as_str(), 1)
    };
    num.parse::<f64>().ok().map(|v| (v * mul as f64) as u64)
}

/// `coli capacity <snap> [ram_gb] [ctx]` — using the model's real dimensions,
/// report per-expert size and how many experts fit resident in a RAM budget
/// after reserving the dense weights, working headroom, and the KV cache for a
/// given context length (`ctx`, e.g. `256k`). Answers "how many experts can a
/// Spark hold while keeping N context".
fn cmd_capacity(args: &[String]) -> ExitCode {
    use colibri_engine::capacity::{
        bytes_per_expert, context_in_kv_budget, experts_in_budget, kv_bytes_per_token,
    };
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli capacity <snapshot-dir> [ram_gb] [ctx e.g. 256k]");
            return ExitCode::from(2);
        }
    };
    let cfg = match colibri_core::Config::load(snap) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("coli capacity: {e}");
            return ExitCode::FAILURE;
        }
    };
    let gib = 1u64 << 30;
    let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
    let gb = |b: u64| b as f64 / gib as f64;

    let bpe = bytes_per_expert(cfg.hidden as u64, cfg.moe_inter as u64, 4);
    let sparse_layers = (cfg.n_layers - cfg.first_dense).max(0) as u64;
    let total_experts = sparse_layers * cfg.n_experts as u64;

    let ram_gb: u64 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .or_else(|| colibri_engine::available_ram_bytes().map(|b| b / gib))
        .unwrap_or(128);
    let ctx = args.get(4).and_then(|s| parse_ctx(s)).unwrap_or(0);

    // Fixed reserves (GLM-5.2 int4 estimates): resident dense ~10 GB, working
    // buffers / OS headroom ~4 GB.
    let dense_gb = 10u64;
    let working_gb = 4u64;
    let kv_per_tok = kv_bytes_per_token(cfg.kv_lora as u64, cfg.qk_rope as u64, cfg.n_layers as u64);
    let kv_bytes = kv_per_tok * ctx;

    let ram_bytes = ram_gb * gib;
    let expert_budget =
        ram_bytes.saturating_sub((dense_gb + working_gb) * gib).saturating_sub(kv_bytes);
    let per_node = experts_in_budget(expert_budget, bpe).min(total_experts);
    let pct = |n: u64| if total_experts > 0 { 100.0 * n as f64 / total_experts as f64 } else { 0.0 };

    println!("model: hidden={} moe_inter={} experts/layer={} attn_layers={} sparse_layers={}",
        cfg.hidden, cfg.moe_inter, cfg.n_experts, cfg.n_layers, sparse_layers);
    println!("per expert (int4): {:.2} MB   total routed: {total_experts} → {:.0} GB",
        mb(bpe), gb(total_experts * bpe));
    println!("KV cache: {:.1} KB/token (compressed MLA, {} layers)",
        kv_per_tok as f64 / 1024.0, cfg.n_layers);
    println!("  8 GB KV holds ~{} tokens ({}K)",
        context_in_kv_budget(8 * gib, kv_per_tok),
        context_in_kv_budget(8 * gib, kv_per_tok) / 1024);
    for &c in &[131072u64, 262144, 524288] {
        println!("  {}K context → {:.1} GB KV", c / 1024, gb(kv_per_tok * c));
    }
    println!();
    println!("budget: {ram_gb} GB − {dense_gb} dense − {working_gb} working{} → {:.0} GB for experts",
        if ctx > 0 { format!(" − {:.0} KV({}K ctx)", gb(kv_bytes), ctx / 1024) } else { String::new() },
        gb(expert_budget));
    println!("==> resident experts per node: {per_node} ({:.0}% of all {total_experts})", pct(per_node));
    if ctx > 0 {
        println!("    (keeping {}K-token context in a {}-GB KV cache)", ctx / 1024, kv_bytes.div_ceil(gib));
    }
    ExitCode::SUCCESS
}

/// Bits per weight actually stored in a loaded tensor, from its container format
/// (`fmt_code`: 0 = f32, 1 = int8, 2 = packed int4).
///
/// `model.dbits`/`model.ebits` are the **LoadOptions** the caller asked for, which
/// only bite when quantizing a full-precision snapshot at load. For a pre-quantized
/// container the bits are fixed on disk and those fields are just the env defaults —
/// reporting them made `ppl` claim "4 / 4 bits" while measuring an 8-bit container.
fn tensor_bits(t: &colibri_core::QTensor) -> &'static str {
    match t.fmt_code {
        0 => "f32",
        1 => "int8",
        2 => "int4",
        _ => "?",
    }
}

/// Log-probability the distribution `logits` assigns to token `t`, in nats.
///
/// Stable log-softmax: `logit[t] - logsumexp(logits)`, shifted by the max so the
/// exponentials can't overflow. A naive `ln(exp(l[t]) / sum(exp(l)))` overflows to
/// inf/NaN on real logits and would silently poison the perplexity.
fn logprob_of(logits: &[f32], t: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    logits[t] - (max + sum_exp.ln())
}

/// `coli ppl <snap> <text-file> [max_tokens]` — teacher-forcing perplexity over
/// held-out text.
///
/// The quality yardstick this repo otherwise lacks: `VALIDATION.md` proves the port
/// is token-exact *against the C engine at the same quantization*, which says nothing
/// about fidelity to the original model. Perplexity does — run the same file through
/// two builds and the lower number is the better model.
///
/// Its reason to exist: choosing quantization by intuition is expensive. `COLI_EBITS
/// 4->8` (attention + dense + shared expert — resident, never streamed) is worth 7.9x
/// the perplexity for ~9 GB of RAM. `COLI_XBITS 4->8` (routed experts) doubles the
/// bytes streamed per token; its cost has **not** been measured, because an 8-bit-
/// expert container needs 0.74 TB and does not fit on the box. Measure which one
/// actually buys the quality before paying for it.
///
/// One forward over the whole sequence (prefill), then the mean negative
/// log-likelihood of each *actual* next token — not the argmax, which only says
/// whether the top pick matched and is blind to how much probability mass moved.
/// Honors `COLI_DBITS`/`COLI_EBITS`, which only bite on a full-precision snapshot: a
/// pre-quantized int4 container is already 4-bit on disk and cannot be un-rounded, so
/// comparing bit-widths on the real model means converting the FP8 source at each
/// setting.
fn cmd_ppl(args: &[String]) -> ExitCode {
    let (snap, text_path) = match (args.get(2), args.get(3)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("usage: coli ppl <snapshot-dir> <text-file|-> [max_tokens]");
            eprintln!("  compares quantization quality: lower perplexity is better");
            return ExitCode::from(2);
        }
    };
    let max_tokens: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(512);

    let text = if text_path == "-" {
        let mut s = String::new();
        match std::io::Read::read_to_string(&mut std::io::stdin(), &mut s) {
            Ok(_) => s,
            Err(e) => {
                eprintln!("coli ppl: stdin: {e}");
                return ExitCode::FAILURE;
            }
        }
    } else {
        match std::fs::read_to_string(text_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("coli ppl: {text_path}: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    let tok_path = format!("{snap}/tokenizer.json");
    let tok = match colibri_tokenizer::Tokenizer::load(&tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("coli ppl: load tokenizer ({tok_path}): {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut ids = tok.encode(&text);
    ids.truncate(max_tokens);
    if ids.len() < 2 {
        eprintln!("coli ppl: need >= 2 tokens (got {}), give it more text", ids.len());
        return ExitCode::from(2);
    }

    let envbits = |k: &str, d: u32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let opts = colibri_engine::LoadOptions {
        dbits: envbits("COLI_DBITS", 4),
        ebits: envbits("COLI_EBITS", 4),
    };
    let model = match colibri_engine::load_model_with(snap, opts) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("coli ppl: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Cache the experts: teacher-forcing a few hundred positions re-touches the same
    // experts many times, and the uncached provider would re-read each from disk.
    let base =
        colibri_engine::ShardsExpertProvider::new(&model.shards, &model.cfg, model.ebits as u32);
    let provider = colibri_engine::ExpertCache::new(base, ram_budget());

    let d = model.cfg.hidden as usize;
    let mut kv = colibri_engine::KvCache::for_model(&model, ids.len());
    let mut hidden = vec![0f32; ids.len() * d];
    // Report what the container actually holds, per class — `ebits` governs the
    // resident path, `xbits` the streamed experts, and they differ independently.
    let resident_fmt = model
        .layers
        .iter()
        .find(|l| l.sparse)
        .map(|l| tensor_bits(&l.sh_gate))
        .unwrap_or("?");
    let expert_fmt = colibri_engine::ExpertProvider::expert(
        &provider,
        model.layers.iter().position(|l| l.sparse).unwrap_or(0),
        0,
    )
    .map(|e| tensor_bits(&e.gate))
    .unwrap_or("?");
    eprintln!(
        "[ppl] {} tokens from {text_path} (resident {resident_fmt}, experts {expert_fmt}) \
         — one forward...",
        ids.len()
    );
    let t0 = std::time::Instant::now();
    if let Err(e) = colibri_engine::forward(&model, &mut kv, &provider, &ids, 0, &mut hidden) {
        eprintln!("coli ppl: {e}");
        return ExitCode::FAILURE;
    }

    // NLL of the token that actually followed each position.
    let mut sum = 0f64;
    let mut top1 = 0usize;
    let n = ids.len() - 1;
    for pos in 0..n {
        let lg = colibri_engine::logits(&model, &hidden[pos * d..(pos + 1) * d]);
        let target = ids[pos + 1] as usize;
        if target >= lg.len() {
            eprintln!("coli ppl: token id {target} out of range for vocab {}", lg.len());
            return ExitCode::FAILURE;
        }
        sum += -logprob_of(&lg, target) as f64;
        if colibri_engine::argmax(&lg) == target {
            top1 += 1;
        }
    }
    let nll = sum / n as f64;
    println!("tokens        : {n}");
    println!("resident/expert: {resident_fmt} / {expert_fmt}   (as stored in the container)");
    println!("mean NLL      : {nll:.4} nats/token");
    println!("perplexity    : {:.3}   <- lower is better", nll.exp());
    println!("top-1 match   : {:.1}%  ({top1}/{n})", top1 as f64 / n as f64 * 100.0);
    eprintln!("[ppl] {:.1}s", t0.elapsed().as_secs_f64());
    ExitCode::SUCCESS
}

/// `coli tf <snap> <id...>` — teacher-forcing: one forward over the token ids,
/// print the argmax prediction at each position. Mirrors the C engine's `TF=1`
/// mode (`forward_all`), for the validation harness. Honors COLI_DBITS/EBITS.
fn cmd_tf(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli tf <snapshot-dir> <token_id ...>");
            return ExitCode::from(2);
        }
    };
    let ids: Vec<i32> = args
        .get(3..)
        .map(|a| a.iter().filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_default();
    if ids.is_empty() {
        eprintln!("coli tf: provide at least one token id");
        return ExitCode::from(2);
    }
    let envbits = |k: &str, d: u32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let opts = colibri_engine::LoadOptions {
        dbits: envbits("COLI_DBITS", 4),
        ebits: envbits("COLI_EBITS", 4),
    };
    let model = match colibri_engine::load_model_with(snap, opts) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("coli tf: {e}");
            return ExitCode::FAILURE;
        }
    };
    let provider =
        colibri_engine::ShardsExpertProvider::new(&model.shards, &model.cfg, model.ebits as u32);
    let d = model.cfg.hidden as usize;
    let mut kv = colibri_engine::KvCache::for_model(&model, ids.len());
    let mut hidden = vec![0f32; ids.len() * d];
    if let Err(e) = colibri_engine::forward(&model, &mut kv, &provider, &ids, 0, &mut hidden) {
        eprintln!("coli tf: {e}");
        return ExitCode::FAILURE;
    }
    let preds: Vec<i32> = (0..ids.len())
        .map(|pos| colibri_engine::argmax(&colibri_engine::logits(&model, &hidden[pos * d..(pos + 1) * d])) as i32)
        .collect();
    println!("tf preds ({}): {preds:?}", preds.len());
    ExitCode::SUCCESS
}

/// `coli config <snap>` — load and print the parsed, validated hyperparameters.
fn cmd_config(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli config <snapshot-dir>");
            return ExitCode::from(2);
        }
    };
    match colibri_core::Config::load(snap) {
        Ok(c) => {
            println!("hidden={}  layers={}  heads={}", c.hidden, c.n_layers, c.n_heads);
            println!(
                "experts={}  topk={}  moe_inter={}  shared={}",
                c.n_experts, c.topk, c.moe_inter, c.n_shared
            );
            println!(
                "q_lora={}  kv_lora={}  qk_head={}  v_head={}",
                c.q_lora, c.kv_lora, c.qk_head, c.v_head
            );
            println!("vocab={}  eps={}  theta={}", c.vocab, c.eps, c.theta);
            println!("stop_ids={:?}", c.stop_ids);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("coli config: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use colibri_cluster::NodeId;
    use std::collections::HashMap;
    use std::net::SocketAddr;

    fn addr(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    const GB: u64 = 1 << 30;

    // The ONLY test that touches the process-global signal state — kept single so it
    // cannot race another test between the reset and the raise. A raise while SHUTDOWN
    // is already true takes the `_exit` path and would kill the test runner; the reset
    // + single raise keeps this on the first-signal (flag-only) path.
    #[cfg(unix)]
    #[test]
    fn sigterm_requests_shutdown_without_exiting() {
        SHUTDOWN.store(false, Ordering::SeqCst);
        install_shutdown_handlers();
        assert!(!shutdown_requested(), "flag should start clear");
        // raise() runs the handler synchronously before returning. If the handler had
        // taken the _exit path this process would die and the run would fail as a
        // crash — so reaching the assert at all is half the test.
        unsafe { libc::raise(libc::SIGTERM) };
        assert!(shutdown_requested(), "SIGTERM must set the shutdown flag");
        SHUTDOWN.store(false, Ordering::SeqCst); // leave clean for any later code
    }

    #[test]
    fn logprob_matches_hand_computed_softmax() {
        // Uniform over 4 -> each has p=0.25 -> ln(0.25).
        let lg = [1.0f32, 1.0, 1.0, 1.0];
        for t in 0..4 {
            assert!((logprob_of(&lg, t) - 0.25f32.ln()).abs() < 1e-6);
        }
        // Asymmetric: check against an explicit softmax.
        let lg = [0.0f32, 1.0, 2.0];
        let denom: f32 = lg.iter().map(|x| x.exp()).sum();
        for t in 0..3 {
            let want = (lg[t].exp() / denom).ln();
            assert!((logprob_of(&lg, t) - want).abs() < 1e-5, "t={t}");
        }
    }

    #[test]
    fn logprob_is_stable_on_huge_logits() {
        // The bug this guards: a naive ln(exp(l[t]) / sum(exp(l))) overflows to
        // inf/NaN here and would silently poison every perplexity we report.
        let lg = [900.0f32, 901.0, 899.0];
        for t in 0..3 {
            let p = logprob_of(&lg, t);
            assert!(p.is_finite(), "logprob {p} not finite for t={t}");
            assert!(p <= 0.0, "log-prob must be <= 0, got {p}");
        }
        // Shifting all logits by a constant must not change the distribution.
        let a: Vec<f32> = vec![0.3, -1.2, 4.5, 2.0];
        let b: Vec<f32> = a.iter().map(|x| x + 500.0).collect();
        for t in 0..a.len() {
            assert!((logprob_of(&a, t) - logprob_of(&b, t)).abs() < 1e-4, "not shift-invariant");
        }
    }

    #[test]
    fn logprobs_form_a_distribution() {
        // exp of the log-probs must sum to 1 — catches a wrong normalizer.
        let lg = [0.5f32, -2.0, 3.25, 1.0, -0.75];
        let mass: f32 = (0..lg.len()).map(|t| logprob_of(&lg, t).exp()).sum();
        assert!((mass - 1.0).abs() < 1e-5, "mass {mass} != 1");
    }

    #[test]
    fn confident_prediction_beats_uniform() {
        // Sanity on the direction: perplexity should fall as the model gets it right.
        let uniform = [0.0f32; 8];
        let confident = [10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(logprob_of(&confident, 0) > logprob_of(&uniform, 0));
        // ...and a confident *wrong* answer is worse than uniform.
        assert!(logprob_of(&confident, 1) < logprob_of(&uniform, 1));
    }

    /// The Spark this was all measured on: 121 GiB total, ~99 GiB `MemAvailable` once
    /// the 8/4 dense weights are resident.
    const SPARK_TOTAL: u64 = 121 * GB;
    const SPARK_AVAIL: u64 = 99 * GB;

    #[test]
    fn budget_stays_on_the_measured_plateau() {
        // This test previously asserted 70..=80 GB and PASSED — encoding the 4/4 model's
        // cliff. The 8/4 model made it wrong: measured counterbalanced on the box, 70 GB
        // swaps 15 GiB at 0.38 tok/s and the old auto-pick of 87 GB manages 0.11, against
        // 0.46 at 40. Throughput is flat from 20..55, so the target is the plateau with
        // margin, not the highest number that fits.
        let got = budget_from(SPARK_AVAIL, 1 * GB, Some(SPARK_TOTAL)) / GB;
        assert!(
            (20..=55).contains(&got),
            "picked {got} GB; must land on the measured 20-55 GB plateau, clear of the \
             cliff between 55 and 70"
        );
    }

    #[test]
    fn idle_box_does_not_get_a_cliff_sized_budget() {
        // The regression that shipped: on an idle box MemAvailable counts page cache as
        // free, so the reserves alone pick 87 GB — past the cliff, 4x slower. The cap is
        // what actually prevents this; the subtraction never could.
        let no_cap = SPARK_AVAIL.saturating_sub(1 * GB).saturating_sub(WORKING_RESERVE) / GB;
        assert!(no_cap > 70, "premise: reserves alone pick {no_cap} GB, past the cliff");
        let with_cap = budget_from(SPARK_AVAIL, 1 * GB, Some(SPARK_TOTAL)) / GB;
        assert!(with_cap < 70, "cap failed to pull {with_cap} GB back below the cliff");
    }

    #[test]
    fn budget_never_underflows_into_an_unbounded_cache() {
        // The bug this guards: `avail - reserve - WORKING_RESERVE` on a small box
        // wraps to ~16 EiB — an effectively unlimited budget, i.e. exactly the OOM
        // the reserve exists to prevent.
        for (avail, reserve) in [(0, 0), (1 * GB, 0), (8 * GB, 64 * GB), (0, u64::MAX)] {
            let b = budget_from(avail, reserve, Some(SPARK_TOTAL));
            assert!(b <= MIN_BUDGET, "underflowed to {b} bytes for avail={avail} reserve={reserve}");
        }
    }

    #[test]
    fn budget_subtracts_both_reserves() {
        // With the cap out of the way (huge total), the reserves still bind exactly:
        // 100 - 20 caller reserve - 10 working = 70.
        let uncapped = Some(u64::MAX);
        assert_eq!(budget_from(100 * GB, 20 * GB, uncapped), 70 * GB);
        // A bigger KV window (longer ctx) must shrink the cache one-for-one.
        assert_eq!(budget_from(100 * GB, 40 * GB, uncapped), 50 * GB);
    }

    #[test]
    fn cap_scales_with_the_machine_not_a_constant() {
        // A fixed reserve can't work across machine sizes: the same 10 GiB that is
        // reasonable on a 32 GiB box is meaningless on a 512 GiB one. The ceiling has
        // to track MemTotal.
        for total_gb in [32u64, 121, 512] {
            let total = total_gb * GB;
            // Idle: MemAvailable ~= total, so only the cap can bind.
            let b = budget_from(total, 0, Some(total)) / GB;
            assert_eq!(b, total_gb / CACHE_CAP_DIVISOR, "cap did not track a {total_gb} GB box");
        }
    }

    #[test]
    fn missing_meminfo_total_still_bounded_by_reserves() {
        // Off-Linux there's no MemTotal to cap against; the reserves must still apply
        // rather than silently going unbounded.
        assert_eq!(budget_from(100 * GB, 20 * GB, None), 70 * GB);
    }

    #[test]
    fn bigger_context_never_grows_the_cache() {
        // Monotonic in the reserve — a longer served window can only take from the
        // cache, never give to it.
        let mut prev = u64::MAX;
        for ctx_gb in [0u64, 5, 11, 22, 44] {
            let b = budget_from(SPARK_AVAIL, ctx_gb * GB, Some(SPARK_TOTAL));
            assert!(b <= prev, "budget grew when ctx grew");
            prev = b;
        }
    }

    #[test]
    fn single_node_needs_no_peers() {
        assert!(missing_peer_ranks(1, NodeId(0), &HashMap::new()).is_empty());
    }

    #[test]
    fn multi_node_without_peers_is_missing_all_others() {
        // The regression this guards: COLI_NUM_NODES=2 with an empty COLI_PEERS used
        // to sail through startup verification (nothing to verify) and only fail on
        // the first token with "no address for node 1".
        assert_eq!(missing_peer_ranks(2, NodeId(0), &HashMap::new()), vec![1]);
        assert_eq!(missing_peer_ranks(4, NodeId(2), &HashMap::new()), vec![0, 1, 3]);
    }

    #[test]
    fn complete_peer_set_is_accepted() {
        let mut p = HashMap::new();
        p.insert(NodeId(1), addr("192.168.100.10:48800"));
        assert!(missing_peer_ranks(2, NodeId(0), &p).is_empty());
    }

    #[test]
    fn partial_peer_set_reports_only_the_gaps() {
        let mut p = HashMap::new();
        p.insert(NodeId(1), addr("192.168.100.10:48800"));
        p.insert(NodeId(3), addr("192.168.100.12:48800"));
        assert_eq!(missing_peer_ranks(4, NodeId(0), &p), vec![2]);
    }
}
