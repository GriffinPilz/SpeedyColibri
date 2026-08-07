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
  convert <src-snap> <out-snap>  FP8/NVFP4 -> int8/nvfp4 container converter  [working]
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

/// Make `free()` actually return expert-sized allocations to the OS.
///
/// **glibc's mmap threshold is dynamic.** It starts at 128 KiB and ratchets *upward* every
/// time an mmap'd block is freed, up to 32 MiB. So a process that recycles large buffers
/// silently migrates them from `mmap` (where `free` is `munmap` and the pages go back to
/// the kernel) into the malloc arena (where `free` retains them). Nothing reports this: the
/// allocation succeeds, the free succeeds, and RSS simply never comes down again.
///
/// That is not a theoretical concern here — it is what made the expert cache's central
/// promise false. Measured on Kimi-K3 (2026-08-01, `COLI_GUARD_TRACE=1`): the OOM guard
/// evicted the cache from **45.31 GB to 18.53 GB** — 26.8 GB of experts — while process RSS
/// climbed **85 GB to 122 GB** and `MemAvailable` fell the entire time, until earlyoom
/// SIGTERMed it. The guard was working perfectly (`gap=100ms` on every tick); eviction
/// decrements `state.bytes` and the memory does not come back. K3's ~17.5 MB MXFP4 spans
/// are exactly the size that gets captured by the ratchet.
///
/// Setting `M_MMAP_THRESHOLD` explicitly **disables the dynamic adjustment** (documented
/// glibc behaviour), pinning expert-sized allocations to `mmap` so a free is a `munmap`.
/// With it, `MemAvailable` recovers across a guard fire (3.99 -> 6.84 GB) and the cache
/// regrows (37.57 -> 54.15 GB) — neither of which happened before.
///
/// The cost is a syscall and page faults per large alloc, which is exactly what the buffer
/// pool in `colibri_core::quant` exists to amortise: hot-path allocations hit the pool and
/// never reach the allocator, so this is paid only on a pool miss. 2 MiB rather than the
/// pool's 1 MiB notion so ordinary sub-huge-page allocations are unaffected.
///
/// `M_TRIM_THRESHOLD` is set for the same reason in the other direction: it governs how
/// much free top-of-heap is retained before `sbrk` gives it back.
/// Tell the ledger about a `gen`-path KV cache.
///
/// `Class::Kv` was charged in exactly ONE place — `serve`'s admission check — so every
/// `coli gen` ran with the ledger believing KV was zero. Small at benchmark sizes (~55 MB
/// for K3's 108 KiB/token over 512 tokens) but it scales with context, and both
/// `RUNTIME_RESERVE` and the cap margin are meant to be derived from what the ledger knows.
/// A ledger that is silently wrong for a whole command is worse than one that is
/// approximate.
///
/// `set_usage`, not `commit`: gen holds exactly one KV cache for the life of the process,
/// so an absolute figure is right and needs no RAII to release.
///
/// **Serve deliberately does not use this.** It charges at admission via
/// `commit_or_wait(Class::Kv, ..)`, which must *hold* the bytes between the fit check and
/// the allocation — otherwise a second request slips into the space between them. Charging
/// again at allocation would double-count. Unifying the two (admission converting its
/// reservation instead of adding a second charge) is a real refactor of serve's admission
/// path and is not attempted here.
/// `n_prompt`/`n_new` rather than a single total: on DeepSeek-V4 the raw KV is a ring, so
/// generated tokens add no raw rows and charging them would take the difference straight
/// out of the expert cache's budget.
fn charge_gen_kv(model: &colibri_engine::Model, n_prompt: usize, n_new: usize) {
    let bytes = colibri_engine::KvCache::bytes_for_split(&model.cfg, n_prompt, n_new) as u64;
    colibri_engine::ram::set_usage(colibri_engine::ram::Class::Kv, bytes);
}

/// 2 MiB: above ordinary sub-huge-page allocations, below every expert span on the fleet
/// (K3's MXFP4 spans are ~17.5 MB, the NVFP4 models' larger still).
const MMAP_THRESHOLD: i32 = 2 * 1024 * 1024;

#[cfg(target_os = "linux")]
fn return_freed_memory_to_the_os() {
    // NOT configurable, deliberately. This shipped with a `COLI_MMAP_THRESHOLD_MB` override
    // on the theory that the right value is span-size dependent. Measured, raising it far
    // enough to stop mmap-forcing costs **nemotron serve 8.40 -> 3.50 tok/s (2.4x)** and
    // **m2.7 decode 3.50 -> 1.89 (1.85x)** — it is not a tuning knob, it is a way to break
    // the engine quietly.
    //
    // Worse, it breaks a second thing implicitly: the per-tick `supported_cap` ceiling
    // evicts *expecting* freed expert buffers to return to the OS. Without mmap-forcing
    // they do not, so eviction becomes pure cost — which is why mallopt-off on the current
    // binary (3.50) measured far worse than the pre-rework binary (8.60) even though
    // neither has mallopt. The allocator setting and the cache ceiling are COUPLED and have
    // to ship together.
    //
    // A future model wanting a different threshold should change this constant and A/B it,
    // the same way the value was arrived at — not flip an env var in production.
    unsafe {
        libc::mallopt(libc::M_MMAP_THRESHOLD, MMAP_THRESHOLD);
        libc::mallopt(libc::M_TRIM_THRESHOLD, MMAP_THRESHOLD);
    }
}

#[cfg(not(target_os = "linux"))]
fn return_freed_memory_to_the_os() {}

fn main() -> ExitCode {
    // Before any allocation that matters. See the function's own doc: without this,
    // evicting an expert does not give its memory back and the cache's "LRU-evict under
    // pressure — never OOM" guarantee is false.
    return_freed_memory_to_the_os();
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
        "genbatch" => cmd_genbatch(&args),
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
        "requant-nvfp4" => cmd_requant_nvfp4(&args),
        "probe" => cmd_probe(&args),
        "qerr" => cmd_qerr(&args),
        "iobench" => cmd_iobench(&args),
        "dropcache" => cmd_dropcache(&args),
        "gpubench" => cmd_gpubench(&args),
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
    let pop = if experts {
        "routed experts"
    } else {
        "resident"
    };

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
/// modelopt-NVFP4 GLM-5.2 snapshot as the colibrì container the engine loads (int8
/// resident, NVFP4 experts).
///
/// Bit-widths default to the measured sweet spot — **8-bit resident** (`ebits=8
/// io_bits=8`): 7.9x better perplexity than all-4-bit resident (6.189 vs 48.665).
/// Routed experts ship NVFP4 (4-bit block-scaled), or e4m3 under `COLI_XFP8`.
/// Override via `COLI_EBITS` / `COLI_IO_BITS` / `COLI_NLAYERS`; see `ConvertOpts`.
/// `coli requant-nvfp4 <container-in> <container-out>` — re-quantize the routed experts
/// of an existing e4m3 colibrì container to NVFP4 (e2m1 nibbles + per-16 ue4m3 block
/// scales + f32 global), copying every other tensor through byte-for-byte. hidden /
/// moe_inter / n_layers come from the input's config.json.
fn cmd_requant_nvfp4(args: &[String]) -> ExitCode {
    let (indir, outdir) = match (args.get(2), args.get(3)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("usage: coli requant-nvfp4 <e4m3-container-dir> <output-container-dir>");
            eprintln!(
                "  re-quantizes routed experts e4m3 -> NVFP4; everything else copied through"
            );
            return ExitCode::from(2);
        }
    };
    let cfg = match colibri_core::Config::load(indir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[requant-nvfp4] load config ({indir}): {e}");
            return ExitCode::FAILURE;
        }
    };
    let (hidden, moe_inter, n_layers) = (
        cfg.hidden as usize,
        cfg.moe_inter as usize,
        cfg.n_layers as usize,
    );
    eprintln!(
        "[requant-nvfp4] {indir} -> {outdir}  (hidden={hidden} moe_inter={moe_inter} n_layers={n_layers})"
    );
    // `COLI_RESIDENT_NVFP4=1` also re-encodes the RESIDENT weights (Kind::Q: attention
    // q/k/v/o, Mamba in_proj/out_proj, fc1/fc2_latent, shared experts) from int8 to NVFP4.
    // Embeddings/lm_head (Kind::Io) are never touched. Measured 0 ± 1.5% perplexity across
    // three corpora, and it is the single-box decode lever: 8.87 -> 6.18 GB/token.
    //
    // NOTE: there is no GPU NVFP4 dense kernel yet, so a container built this way runs the
    // resident matmuls on the single-threaded CPU path. It is CORRECT but slow — convert it
    // to validate tokens/perplexity, do NOT benchmark speed against it.
    let resident_nvfp4 = std::env::var("COLI_RESIDENT_NVFP4").ok().as_deref() == Some("1");
    if resident_nvfp4 {
        eprintln!(
            "[requant-nvfp4] RESIDENT weights -> NVFP4 too (embeddings/lm_head untouched). \
             No GPU dense NVFP4 kernel exists yet: this container is for correctness and \
             perplexity only, NOT for speed measurement."
        );
    }
    let t0 = std::time::Instant::now();
    let res = colibri_engine::requant_experts_nvfp4(
        indir,
        outdir,
        n_layers,
        hidden,
        moe_inter,
        resident_nvfp4,
        |fi, n, st| {
            eprintln!(
                "[requant-nvfp4] shard {:>3}/{n}  experts_nvfp4={} copied={} skipped={}  out={:.1} GB  {:.0}s",
                fi + 1,
                st.tensors_quantized,
                st.tensors_f32,
                st.tensors_skipped,
                st.bytes_out as f64 / 1e9,
                t0.elapsed().as_secs_f64()
            );
        },
    );
    match res {
        Ok(st) => {
            eprintln!(
                "[requant-nvfp4] done: {} shards, {} expert weights -> NVFP4, {} copied, {} dropped, {:.1} GB out, {:.0}s",
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
            eprintln!("[requant-nvfp4] error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_convert(args: &[String]) -> ExitCode {
    let (indir, outdir) = match (args.get(2), args.get(3)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("usage: coli convert <fp8|nvfp4-snapshot-dir> <output-snapshot-dir>");
            eprintln!("  env: COLI_EBITS(8) COLI_IO_BITS(8) COLI_NLAYERS(78) COLI_KEEP_INDEXER(0) COLI_XFP8(0)");
            return ExitCode::from(2);
        }
    };
    let env_u32 = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    // Detect the source architecture from its config: the MiniMax GQA family (M3/M2)
    // needs the block_sparse_moe→mlp / w1w2w3 name remapping, its layer count comes from
    // the config (env COLI_NLAYERS still overrides for GLM), and Gemma-norm folding is
    // per-model (M3 yes, M2 no — read from the config). Missing config → falls back to GLM.
    let src_cfg = colibri_core::Config::load(indir).ok();
    let minimax = src_cfg.as_ref().map(|c| c.arch.is_gqa()).unwrap_or(false);
    let nemotron = src_cfg
        .as_ref()
        .map(|c| c.arch == colibri_core::Arch::NemotronH)
        .unwrap_or(false);
    let kimi = src_cfg
        .as_ref()
        .map(|c| c.arch == colibri_core::Arch::KimiK3)
        .unwrap_or(false);
    let deepseek_v4 = src_cfg
        .as_ref()
        .map(|c| c.arch == colibri_core::Arch::DeepseekV4)
        .unwrap_or(false);
    let gemma_norm = src_cfg.as_ref().map(|c| c.gemma_norm).unwrap_or(false);
    let n_layers = if minimax || nemotron || kimi || deepseek_v4 {
        src_cfg.as_ref().map(|c| c.n_layers as usize).unwrap_or(60)
    } else {
        env_u32("COLI_NLAYERS", 78) as usize
    };
    // Routed experts are NVFP4 (4-bit block-scaled) regardless of `ebits`, or e4m3
    // under COLI_XFP8 — they do not inherit the resident bit width, so raising
    // `ebits` never drags the streamed experts (or the 0.74 TB container) up with it.
    let opts = colibri_engine::ConvertOpts {
        ebits: env_u32("COLI_EBITS", 8),
        io_bits: env_u32("COLI_IO_BITS", 8),
        n_layers,
        // COLI_KEEP_INDEXER=1 keeps the DSA lightning-indexer weights so the container
        // can run DSA sparse attention (dropped by default, matching the reference).
        keep_indexer: env_u32("COLI_KEEP_INDEXER", 0) != 0,
        // Experts are NVFP4 (4-bit block-scaled) by default; COLI_XFP8=1 opts into 8-bit
        // e4m3 instead. int4 experts are no longer produced.
        xfp8: env_u32("COLI_XFP8", 0) != 0,
        // COLI_MTP_ONLY=1 converts ONLY the MTP speculative head (layer n_layers).
        // Drop the resulting shard into an existing container (Shards::open indexes
        // every *.safetensors in the dir) to enable drafting without re-converting.
        mtp_only: env_u32("COLI_MTP_ONLY", 0) != 0,
        deepseek_v4,
        // MiniMax-M3 name/norm handling (auto-detected above).
        minimax,
        gemma_norm,
        // Nemotron-H hybrid: backbone.*→model.* remap + `.mixer.` classification.
        nemotron,
        // Kimi-K3 hybrid: M3-style remap + the latent-MoE projection rename. Its routed
        // experts are already MXFP4 and pass through without requantizing.
        kimi,
        // How many layer indices above the stack the MTP head occupies: 1 for GLM/M3's
        // single sparse block, 2 for Nemotron-H's `"*E"` attention+latent-MoE pair. Read
        // from the SOURCE config's `mtp_hybrid_override_pattern`; falls back to 1.
        mtp_layers: src_cfg.as_ref().map(|c| c.mtp_head_layers()).unwrap_or(1),
    };

    eprintln!(
        "[convert] {indir} -> {outdir}  (experts={} ebits={} io_bits={} n_layers={})",
        // A K3 source is already MXFP4 and its experts are COPIED, not requantized —
        // saying "nvfp4" here would misreport what the container actually holds, which
        // is the one thing this line exists to tell you.
        match (opts.xfp8, opts.kimi) {
            (true, _) => "e4m3",
            (false, true) => "mxfp4 (passthrough)",
            (false, false) => "nvfp4",
        },
        opts.ebits,
        opts.io_bits,
        opts.n_layers
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
            println!(
                "loaded {} layers ({dense} dense, {sparse} MoE)",
                m.layers.len()
            );
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

/// Warn when the last model loaded on this box was a DIFFERENT one, and say what to do.
///
/// `coli dropcache` fadvises only the containers it is given, so switching models leaves the
/// outgoing model's pages resident and the incoming one evicts them while streaming.
/// Measured 2026-08-02 on K3: 95 MB/s at the device and still unfinished at 69 minutes,
/// against 456 s for the very next run once those pages were gone — **~30x**, from nothing
/// but whose pages were resident.
///
/// `scripts/lib.sh`'s `mem_reset` now handles this automatically, so the harness is covered.
/// This exists for the case that is not: a bare `coli gen ...` typed by hand. I wrote the
/// warning about this trap and then walked into it an hour later with a one-liner, spent ten
/// minutes misreading the result as a hang, and only then remembered. A line of output at
/// load time would have ended it immediately.
///
/// Advisory only — it never evicts. Dropping another model's cache without being asked would
/// be the wrong call on a shared box, and the state file is best-effort: a missing or
/// unreadable one simply means no warning.
fn note_model_switch(container: &str) {
    const LAST: &str = "/tmp/colibri-last-container";
    let prev = std::fs::read_to_string(LAST).unwrap_or_default();
    let prev = prev.trim();
    if !prev.is_empty() && prev != container && std::path::Path::new(prev).is_dir() {
        eprintln!(
            "[cache] NOTE: the previous model loaded here was {prev}, and its pages are \
             probably still resident — this run may be far slower than steady state. \
             `coli dropcache {container} {prev}` first, or use scripts/lib.sh's mem_reset."
        );
    }
    let _ = std::fs::write(LAST, container);
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
    note_model_switch(snap);
    let prompt: Vec<i32> = args
        .get(3..)
        .map(|a| a.iter().filter_map(|s| s.parse().ok()).collect())
        .filter(|v: &Vec<i32>| !v.is_empty())
        .unwrap_or_else(|| vec![1]);

    // Bit-widths default to int8 (int8-resident container); overridable via env for
    // the C-vs-Rust validation harness (e.g. COLI_DBITS=16 for the exact f32 path).
    let envbits = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let opts = colibri_engine::LoadOptions {
        dbits: envbits("COLI_DBITS", 8),
        ebits: envbits("COLI_EBITS", 8),
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
    let usage_path = std::env::var("COLI_USAGE").unwrap_or_else(|_| format!("{snap}/.coli_usage"));
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
    // expert off disk). Budget from the adaptive cap.
    let base = colibri_engine::ShardsExpertProvider::with_sharding(
        &model.shards,
        &model.cfg,
        model.ebits as u32,
        sharding.clone(),
        cluster.this_node,
    );
    let budget = ram_budget();
    let provider = std::sync::Arc::new(colibri_engine::ExpertCache::new(base, budget));
    let owned_ids: Vec<u32> = sharding.local_experts(cluster.this_node).collect();
    let _maxres = wire_adaptive_cache(
        &provider,
        &model.cfg,
        model.ebits as u32,
        &owned_ids,
        model.resident_bytes(),
    );
    preload_all_experts(&provider, &model.cfg, _maxres, &owned_ids);
    if let Some(topn) = prefetch_topn() {
        provider.enable_prefetch(topn, model.cfg.n_experts as u64);
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

    // `for_model` sizes the KV for the MTP head too (one extra row per head SUBLAYER:
    // 1 on GLM/M3, 2 on Nemotron-H's `"*E"` head); hand-rolling
    // `KvCache::new(n_layers, ..)` would under-allocate on an MTP model.
    let mut kv = colibri_engine::KvCache::for_model(model, prompt.len() + n_new);
    charge_gen_kv(model, prompt.len(), n_new);
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
            // DeepSeek-V4 Indexer pruning. Not CUDA-gated: it is a model mechanism, and
            // its success case is "tokens unchanged", so without this line a skipped
            // Indexer and a working one look identical from the outside.
            {
                let (scored, seen, kept) = colibri_engine::forward::dsv4_indexer_stats();
                let (skipped, skip_max) = colibri_engine::forward::dsv4_indexer_skips();
                if scored > 0 {
                    println!(
                        "indexer: {scored} queries scored, {seen} candidate rows -> {kept} kept \
                         ({:.1}% pruned)",
                        100.0 * (seen - kept) as f64 / seen as f64
                    );
                }
                if skipped > 0 {
                    println!(
                        "indexer: {skipped} queries kept everything (largest candidate set {skip_max})"
                    );
                }
            }
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
    let rank: u32 = std::env::var("COLI_NODE_RANK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let port: u16 = std::env::var("COLI_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    eprintln!(
        "[cluster] scanning the fabric for {:.0}s (UDP :{}) ...",
        window.as_secs_f64(),
        colibri_cluster::discovery::DISC_PORT
    );
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
    std::env::var("COLI_EXPERT_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48800)
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
fn parse_peers(
) -> Result<std::collections::HashMap<colibri_cluster::NodeId, std::net::SocketAddr>, String> {
    use std::net::ToSocketAddrs;
    let mut map = std::collections::HashMap::new();
    let s = std::env::var("COLI_PEERS").unwrap_or_default();
    for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let (rank, addr) = entry
            .split_once('=')
            .ok_or_else(|| format!("bad COLI_PEERS entry '{entry}' (want rank=host:port)"))?;
        let rank: u32 = rank
            .trim()
            .parse()
            .map_err(|_| format!("bad rank in '{entry}'"))?;
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
            eprintln!(
                "usage: coli worker <snapshot-dir> [port]  (set COLI_NODE_RANK / COLI_NUM_NODES)"
            );
            return ExitCode::from(2);
        }
    };
    let cluster = colibri_cluster::ClusterConfig::from_env();
    let port = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(expert_port);

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
    let usage_path = std::env::var("COLI_USAGE").unwrap_or_else(|_| format!("{snap}/.coli_usage"));
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
    let owned_ids: Vec<u32> = sharding.local_experts(cluster.this_node).collect();
    let _maxres = wire_adaptive_cache(
        &provider,
        &model.cfg,
        model.ebits as u32,
        &owned_ids,
        model.resident_bytes(),
    );
    preload_all_experts(&provider, &model.cfg, _maxres, &owned_ids);
    if let Some(topn) = prefetch_topn() {
        provider.enable_prefetch(topn, model.cfg.n_experts as u64);
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
            eprintln!(
                "[worker] attention request for out-of-range layer {}",
                req.layer
            );
        }
        colibri_cluster::AttnResponse {
            outputs,
            n_tokens: req.n_tokens,
            hidden: req.hidden,
        }
    };
    let bound = match colibri_cluster::serve_cluster(
        addr,
        fingerprint,
        move |req| match colibri_engine::compute_experts_partial(
            &*provider,
            req.layer as usize,
            &req.experts,
            &req.weights,
            &req.activations,
            req.n_tokens,
            req.hidden,
        ) {
            Ok(outputs) => colibri_cluster::ExpertResponse {
                outputs,
                n_tokens: req.n_tokens,
                hidden: req.hidden,
            },
            Err(e) => {
                eprintln!("[worker] expert compute error: {e}");
                colibri_cluster::ExpertResponse {
                    outputs: vec![0.0; req.n_tokens * req.hidden],
                    n_tokens: req.n_tokens,
                    hidden: req.hidden,
                }
            }
        },
        attn,
    ) {
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

/// `coli genbatch <snap> <B> <ngen> [base token id...]` — batched multi-sequence
/// decode benchmark. B sequences advance one token per `forward_batched`, so the
/// routed-expert union streams from disk ONCE per step and amortizes across the
/// batch (decode is bytes-bound — this is the throughput lever). The base prompt is
/// diversified per slot so routing genuinely spreads (synthetic ids, kept in-vocab).
/// Reports aggregate decode tok/s (B tokens/step). `COLI_BATCH_VERIFY=1` also decodes
/// slot 0 single-sequence and asserts identical tokens (batching must not change
/// output). Single-node measurement path; mirrors `cmd_gen`'s loader + autopin.
fn cmd_genbatch(args: &[String]) -> ExitCode {
    let snap = match args.get(2) {
        Some(p) => p,
        None => {
            eprintln!("usage: coli genbatch <snapshot-dir> <B> <ngen> [token_id ...]");
            return ExitCode::from(2);
        }
    };
    let b: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let ngen: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(16);
    let base: Vec<i32> = args
        .get(5..)
        .map(|a| a.iter().filter_map(|s| s.parse().ok()).collect())
        .filter(|v: &Vec<i32>| !v.is_empty())
        .unwrap_or_else(|| vec![1]);
    if b == 0 {
        eprintln!("genbatch: B must be >= 1");
        return ExitCode::from(2);
    }

    let envbits = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let opts = colibri_engine::LoadOptions {
        dbits: envbits("COLI_DBITS", 8),
        ebits: envbits("COLI_EBITS", 8),
    };
    let model = match colibri_engine::load_model_with(snap, opts) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("coli genbatch: {e}");
            return ExitCode::FAILURE;
        }
    };
    let model: &'static colibri_engine::Model = Box::leak(Box::new(model));
    let d = model.cfg.hidden as usize;
    let vocab = model.cfg.vocab.max(1);

    // Single-node provider (the measurement regime), mirroring cmd_gen's setup.
    let usage_path = std::env::var("COLI_USAGE").unwrap_or_else(|_| format!("{snap}/.coli_usage"));
    let history = colibri_engine::UsageHistory::load(&usage_path).unwrap_or_default();
    let sharding = colibri_cluster::ExpertSharding::single(model.cfg.n_experts as u32);
    let base_p = colibri_engine::ShardsExpertProvider::with_sharding(
        &model.shards,
        &model.cfg,
        model.ebits as u32,
        sharding,
        colibri_cluster::NodeId(0),
    );
    let budget = ram_budget();
    let provider = std::sync::Arc::new(colibri_engine::ExpertCache::new(base_p, budget));
    apply_autopin(&provider, &history, budget);

    // Diversify the base prompt per slot so each sequence routes differently (this is
    // a synthetic benchmark — like `gen`, ids need only be valid; the shift keeps them
    // in-vocab). Identical prompts would route identically and hide the amortization.
    let seqs: Vec<Vec<i32>> = (0..b)
        .map(|n| {
            base.iter()
                .enumerate()
                .map(|(i, &t)| (t + (n as i32 * 149 + i as i32 * 7) % vocab).rem_euclid(vocab))
                .collect()
        })
        .collect();

    use colibri_engine::{argmax, forward, forward_batched, logits, KvCache};

    // Prefill each sequence into its own KV cache.
    let cap = base.len() + ngen + 2;
    let mut kvs: Vec<KvCache> = Vec::with_capacity(b);
    let mut pos: Vec<usize> = Vec::with_capacity(b);
    let mut logit: Vec<Vec<f32>> = Vec::with_capacity(b);
    let t_pre = std::time::Instant::now();
    for n in 0..b {
        let mut kv = KvCache::for_model(model, cap);
        let s = seqs[n].len();
        let mut hidden = vec![0f32; s * d];
        if let Err(e) = forward(model, &mut kv, &*provider, &seqs[n], 0, &mut hidden) {
            eprintln!("genbatch prefill seq {n}: {e}");
            return ExitCode::FAILURE;
        }
        logit.push(logits(model, &hidden[(s - 1) * d..s * d]));
        pos.push(s);
        kvs.push(kv);
    }
    eprintln!(
        "[genbatch] prefilled B={b} seqs ({} tok each) in {:.1}s",
        base.len(),
        t_pre.elapsed().as_secs_f64()
    );

    // Batched decode: one token per sequence per step.
    let mut outs: Vec<Vec<i32>> = vec![Vec::with_capacity(ngen); b];
    let mut step_ms: Vec<f64> = Vec::with_capacity(ngen);
    for _ in 0..ngen {
        let ids: Vec<i32> = (0..b).map(|n| argmax(&logit[n]) as i32).collect();
        for n in 0..b {
            outs[n].push(ids[n]);
        }
        let t = std::time::Instant::now();
        let mut hidden = vec![0f32; b * d];
        if let Err(e) = forward_batched(model, &mut kvs, &*provider, &ids, &pos, &mut hidden) {
            eprintln!("genbatch step: {e}");
            return ExitCode::FAILURE;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        step_ms.push(ms);
        for n in 0..b {
            logit[n] = logits(model, &hidden[n * d..(n + 1) * d]);
            pos[n] += 1;
        }
        eprintln!(
            "[genbatch] step {}/{ngen}: {ms:.1} ms  -> {:.2} tok/s aggregate",
            step_ms.len(),
            b as f64 / (ms / 1e3)
        );
    }

    // Steady-state = drop the first 2 warm-up steps.
    let warm = 2.min(step_ms.len().saturating_sub(1));
    let ss = &step_ms[warm..];
    let mean = ss.iter().sum::<f64>() / ss.len().max(1) as f64;
    println!(
        "genbatch B={b} ngen={ngen}: steady-state {:.1} ms/step  aggregate {:.2} tok/s  per-seq {:.3} tok/s",
        mean,
        b as f64 / (mean / 1e3),
        1.0 / (mean / 1e3)
    );
    let s0 = &outs[0];
    println!("slot0 tokens: {:?}", &s0[..s0.len().min(12)]);
    let st = provider.stats();
    println!(
        "expert cache: {} resident ({:.1} MB), {} hits / {} misses, {} evictions",
        st.resident,
        st.bytes as f64 / (1024.0 * 1024.0),
        st.hits,
        st.misses,
        st.evictions
    );

    // Token-identity gate: slot 0 batched must equal slot 0 decoded alone.
    if std::env::var("COLI_BATCH_VERIFY").ok().as_deref() == Some("1") {
        let mut kv = KvCache::for_model(model, cap);
        let s = seqs[0].len();
        let mut hidden = vec![0f32; s * d];
        forward(model, &mut kv, &*provider, &seqs[0], 0, &mut hidden).unwrap();
        let mut lg = logits(model, &hidden[(s - 1) * d..s * d]);
        let mut p = s;
        let mut single: Vec<i32> = Vec::with_capacity(ngen);
        for _ in 0..ngen {
            let nx = argmax(&lg) as i32;
            single.push(nx);
            let mut h = vec![0f32; d];
            forward(model, &mut kv, &*provider, &[nx], p, &mut h).unwrap();
            lg = logits(model, &h);
            p += 1;
        }
        if single == outs[0] {
            println!("VERIFY PASS: slot0 batched == single-sequence ({ngen} tok)");
        } else {
            println!(
                "VERIFY FAIL: slot0 diverged\n single:  {single:?}\n batched: {:?}",
                outs[0]
            );
        }
    }
    ExitCode::SUCCESS
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
    let bpe = colibri_engine::capacity::bytes_per_expert_of(&model.cfg, model.ebits as u32);
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
    charge_gen_kv(&model, prompt.len(), n_new);
    match colibri_engine::generate_greedy(model, &mut kv, provider, prompt, n_new) {
        Ok(seq) => {
            println!("prompt: {prompt:?}");
            println!(
                "generated ({} tok): {:?}",
                seq.len() - prompt.len(),
                &seq[prompt.len()..]
            );
            // DeepSeek-V4 Indexer pruning. Printed whenever it actually scored, because
            // "tokens unchanged" is the SUCCESS case for a mechanism that drops the least
            // relevant rows — and is indistinguishable from it never having run.
            {
                let (scored, seen, kept) = colibri_engine::forward::dsv4_indexer_stats();
                if scored > 0 {
                    println!(
                        "indexer: {scored} queries scored, {seen} candidate rows -> {kept} kept \
                         ({:.1}% pruned)",
                        100.0 * (seen - kept) as f64 / seen as f64
                    );
                }
            }
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
fn wr_u64<W: std::io::Write>(w: &mut W, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn wr_u32<W: std::io::Write>(w: &mut W, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn rd_u64<R: std::io::Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn rd_u32<R: std::io::Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn wr_path<W: std::io::Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    wr_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}
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
    let read_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
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
        let nodes = match rd_u32(&mut stream) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[shard-serve] handshake: {e}");
                continue;
            }
        };
        let rank = match rd_u32(&mut stream) {
            Ok(v) => v,
            Err(_) => continue,
        };
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
                    if let (Some(fname), Ok(m)) = (
                        p.file_name().and_then(|f| f.to_str()).map(String::from),
                        std::fs::metadata(&p),
                    ) {
                        meta.push((fname, p, m.len()));
                    }
                }
            }
        }
        let gb: f64 = (items.iter().map(|t| t.nbytes).sum::<u64>()
            + meta.iter().map(|(_, _, s)| *s).sum::<u64>()) as f64
            / 1e9;
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
                    if gi > 0 {
                        header.push(',');
                    }
                    let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
                    header.push_str(&format!(
                        "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                        t.name,
                        t.dtype.safetensors_str(),
                        shape.join(","),
                        rel,
                        rel + t.nbytes
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
                            if i >= grp.len() {
                                break;
                            }
                            let mut buf = vec![0u8; grp[i].nbytes as usize];
                            if shards.read_raw(&grp[i].name, &mut buf).is_ok() {
                                let _ = tx.send((i, buf));
                            }
                        });
                    }
                    drop(tx);
                    for (i, buf) in rx {
                        bufs[i] = Some(buf);
                    }
                });
                for b in &bufs {
                    match b {
                        Some(bytes) => w.write_all(bytes)?,
                        None => {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                "shard-serve: tensor read failed",
                            ))
                        }
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
                t0.elapsed().as_secs_f64(),
                gb / t0.elapsed().as_secs_f64().max(1e-9)
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
            let mut f = std::io::BufWriter::with_capacity(
                8 << 20,
                std::fs::File::create(out_dir.join(&path))?,
            );
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
            eprintln!(
                "shard-pull: received {nf} files, {:.1} GB → {out}",
                total as f64 / 1e9
            );
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
    let bytes: u64 = names
        .iter()
        .filter_map(|n| shards.find(n))
        .map(|t| t.nbytes)
        .sum();
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

/// `coli gpubench [S=1] [reps=200]` — per-call cost of a GPU matmul, isolated from
/// any model. Decode issues one of these per dense weight per layer, each paying a
/// blocking HtoD copy, a launch, a DtoH copy and a stream sync. The sweep includes a
/// deliberately tiny shape whose arithmetic is negligible, so its time IS the fixed
/// per-call floor and every real shape can be reported against it.
///
/// Prints two tables. The second sweeps the fused routed-expert FFN for nvfp4 AND mxfp4,
/// which is the only way to measure MXFP4 at all — fmt 6 is expert-only, so it has no
/// dense matmul the first table could call.
#[cfg(feature = "cuda")]
fn cmd_gpubench(args: &[String]) -> ExitCode {
    let s: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let reps: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(200);
    if !colibri_engine::gpu::available() {
        eprintln!("coli gpubench: no CUDA backend — this measures GPU dispatch, so there is nothing to report");
        return ExitCode::FAILURE;
    }
    colibri_engine::gpubench::report(s, reps);
    colibri_engine::gpubench::report_experts(s, reps);
    ExitCode::SUCCESS
}

/// Without the `cuda` feature there is no dispatch to measure — say so rather than
/// timing the CPU fallback, which would report a number for a different code path.
#[cfg(not(feature = "cuda"))]
fn cmd_gpubench(_args: &[String]) -> ExitCode {
    eprintln!("coli gpubench: built without the `cuda` feature — nothing to measure");
    ExitCode::FAILURE
}

/// `coli iobench <file> [total_mb=2048] [bs_kb=2048] [qd=20]` — the atlas io_uring
/// question, measured. Reads `total_mb` of **cold** data (page cache dropped via
/// `posix_fadvise(DONTNEED)` before each pass) from `file` at random block-aligned
/// offsets, block size `bs_kb`, comparing the read engines that matter for decode:
///   (A) our model — `qd` threads each doing blocking `pread`,
///   (B) io_uring with SQPOLL at depth `qd`, buffered (page-cache-served, like us),
///   (C) io_uring + O_DIRECT at depth `qd` — the drive's ceiling (bypasses cache).
/// If (A) already ≈ (C), the read engine is not the bottleneck and io_uring can't help.
/// `coli dropcache <container> [container ...]` — drop those models' cached pages, and
/// report memory state.
///
/// Run this **between benchmark arms**. Page-cache carry-over is the largest source of
/// contamination in this repo's measurements: the same configuration has read 2.27 tok/s
/// early in a sequence and 0.23 tok/s late, purely from what an earlier arm left warm. An
/// A/B that does not reset between arms is measuring arm order as much as arm content.
///
/// **Pass the PREVIOUS model too when switching models.** `fadvise` only touches the
/// containers named, so dropping just the incoming model leaves the outgoing one resident
/// and the new model has to evict it while streaming. Measured 2026-08-02: the first K3 run
/// after nine Nemotron runs (172% coverage, fills the cache) moved 95 MB/s and had not
/// finished at 69 minutes; the very next K3 run, with those pages displaced, completed in
/// 456 s. ~30x, from nothing but whose pages were resident.
///
/// `scripts/bench.sh` works around this with a discarded warmup run, which costs a full run
/// per model. Naming both containers is the cheaper fix and is why this takes a list.
///
/// Uses `posix_fadvise(DONTNEED)`, so it needs **no root** and only touches this model's
/// pages. Two things it deliberately does not do, because they require privileges this
/// process does not have:
///
///   - `/proc/sys/vm/drop_caches` — would clear the whole machine's cache
///   - `swapoff -a && swapon -a` — the only way to reclaim already-swapped pages
///
/// If a run has driven the box into swap, that swap **stays** until someone with root
/// clears it, and every subsequent measurement is taken on a degraded box. This command
/// reports swap usage precisely so that state is visible rather than silently spoiling the
/// next twenty runs — which is exactly what happened here before it existed.
fn cmd_dropcache(args: &[String]) -> ExitCode {
    // `args` is the whole argv — every other command indexes from 2 (argv[0], subcommand).
    // This read `args.first()`, so it took the *binary path* as the container, failed with
    // "Not a directory", and exited 1. `mem_reset` pipes it under `set -euo pipefail`, so
    // every `scripts/bench.sh` suite aborted immediately after printing its header and
    // produced no output at all. A benchmark harness that silently measures nothing is
    // worse than one that crashes.
    let containers = &args[2.min(args.len())..];
    if containers.is_empty() {
        eprintln!("usage: coli dropcache <container-dir> [container-dir ...]");
        return ExitCode::from(2);
    }
    let mem = |key: &str| -> u64 {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines().find(|l| l.starts_with(key)).and_then(|l| {
                    l.split_whitespace()
                        .nth(1)
                        .and_then(|v| v.parse::<u64>().ok())
                })
            })
            .unwrap_or(0)
            / 1024 // MiB
    };
    let cached_before = mem("Cached:");
    let mut dropped = 0u64;
    let mut nfiles = 0usize;
    for container in containers {
        let shards = match colibri_safetensors::Shards::open(container) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("dropcache: {container}: {e}");
                return ExitCode::FAILURE;
            }
        };
        dropped += shards.drop_page_cache();
        nfiles += shards.num_files();
    }
    let cached_after = mem("Cached:");
    let swap_used = mem("SwapTotal:").saturating_sub(mem("SwapFree:"));
    println!(
        "dropcache: advised {} GB across {} shards in {} container(s) | page cache {} -> {} MiB (freed {})",
        dropped >> 30,
        nfiles,
        containers.len(),
        cached_before,
        cached_after,
        cached_before.saturating_sub(cached_after),
    );
    if swap_used > 0 {
        println!(
            "dropcache: WARNING {swap_used} MiB still in SWAP — fadvise cannot reclaim it. \
             Every measurement on this box is degraded until a root user runs \
             `swapoff -a && swapon -a` (or the box is rebooted)."
        );
    }
    ExitCode::SUCCESS
}

#[cfg(target_os = "linux")]
fn cmd_iobench(args: &[String]) -> ExitCode {
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;
    let path = match args.get(2) {
        Some(p) => p.clone(),
        None => {
            eprintln!("usage: coli iobench <file> [total_mb=2048] [bs_kb=2048] [qd=20]");
            return ExitCode::from(2);
        }
    };
    let total_mb: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2048);
    let bs: usize = args
        .get(4)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048)
        * 1024;
    let qd: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(20);

    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("coli iobench: open {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let fd = file.as_raw_fd();
    let flen = file.metadata().map(|m| m.len()).unwrap_or(0);
    if flen < bs as u64 * 2 {
        eprintln!("coli iobench: file too small ({flen} B) for bs {bs}");
        return ExitCode::FAILURE;
    }
    let nblocks = ((total_mb << 20) / bs as u64).max(1) as usize;
    let maxblk = flen / bs as u64;
    // deterministic block-aligned pseudo-random offsets (splitmix64)
    let mut sm = 0x1234_5678_9abc_def0u64;
    let offsets: Vec<u64> = (0..nblocks)
        .map(|_| {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z % maxblk) * bs as u64
        })
        .collect();
    let total_bytes = (nblocks * bs) as f64;
    let drop_cache =
        || unsafe { libc::posix_fadvise(fd, 0, flen as libc::off_t, libc::POSIX_FADV_DONTNEED) };
    println!(
        "iobench {path}: {nblocks} × {} KiB = {:.2} GB cold random reads, qd={qd}",
        bs / 1024,
        total_bytes / 1e9
    );

    // (A) our model: `qd` threads, blocking pread.
    drop_cache();
    let ta = std::time::Instant::now();
    std::thread::scope(|s| {
        for t in 0..qd {
            let offsets = &offsets;
            s.spawn(move || {
                let mut buf = vec![0u8; bs];
                let mut k = t;
                while k < nblocks {
                    let base = offsets[k] as libc::off_t;
                    let mut done = 0usize;
                    while done < bs {
                        let r = unsafe {
                            libc::pread(
                                fd,
                                buf.as_mut_ptr().add(done) as *mut libc::c_void,
                                bs - done,
                                base + done as libc::off_t,
                            )
                        };
                        if r <= 0 {
                            break;
                        }
                        done += r as usize;
                    }
                    k += qd;
                }
            });
        }
    });
    let a = ta.elapsed().as_secs_f64();
    println!(
        "  (A) threaded pread ×{qd}   : {:.2} GB/s  ({a:.2}s)",
        total_bytes / a / 1e9
    );

    // (B) io_uring, buffered.
    drop_cache();
    let tb = std::time::Instant::now();
    match iouring_read(fd, bs, qd, &offsets, false) {
        Ok(sqpoll) => {
            let b = tb.elapsed().as_secs_f64();
            println!(
                "  (B) io_uring {} buffered: {:.2} GB/s  ({b:.2}s)",
                if sqpoll { "SQPOLL" } else { "plain " },
                total_bytes / b / 1e9
            );
        }
        Err(e) => println!("  (B) io_uring buffered FAILED: {e}"),
    }

    // (C) io_uring + O_DIRECT — the drive ceiling (opens a second O_DIRECT fd).
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(&path)
    {
        Ok(df) => {
            let dfd = df.as_raw_fd();
            let tc = std::time::Instant::now();
            match iouring_read(dfd, bs, qd, &offsets, true) {
                Ok(sqpoll) => {
                    let c = tc.elapsed().as_secs_f64();
                    println!(
                        "  (C) io_uring {} O_DIRECT: {:.2} GB/s  ({c:.2}s)  [drive ceiling]",
                        if sqpoll { "SQPOLL" } else { "plain " },
                        total_bytes / c / 1e9
                    );
                }
                Err(e) => println!("  (C) io_uring O_DIRECT FAILED: {e}"),
            }
        }
        Err(e) => println!("  (C) O_DIRECT open failed ({e}) — skipping ceiling"),
    }
    ExitCode::SUCCESS
}

#[cfg(not(target_os = "linux"))]
fn cmd_iobench(_args: &[String]) -> ExitCode {
    eprintln!("coli iobench: Linux only (io_uring)");
    ExitCode::from(2)
}

/// Read every offset once through an io_uring of depth `qd`. `direct` requires the
/// buffers and offsets to be 512-aligned (they are: block-aligned offsets, page-aligned
/// buffers). Returns whether SQPOLL was successfully enabled.
#[cfg(target_os = "linux")]
fn iouring_read(
    fd: i32,
    bs: usize,
    qd: usize,
    offsets: &[u64],
    direct: bool,
) -> std::io::Result<bool> {
    use io_uring::{opcode, types, IoUring};
    let (mut ring, sqpoll) = match IoUring::builder().setup_sqpoll(2000).build(qd as u32) {
        Ok(r) => (r, true),
        Err(_) => (IoUring::new(qd as u32)?, false),
    };
    // Page-aligned buffers (required for O_DIRECT; harmless otherwise).
    let align = 4096usize;
    let mut bufs: Vec<Vec<u8>> = (0..qd)
        .map(|_| {
            let mut v = vec![0u8; bs + align];
            let off = v.as_ptr() as usize % align;
            if off != 0 {
                v.drain(0..(align - off));
            }
            v
        })
        .collect();
    let _ = direct;
    let n = offsets.len();
    let mut next = 0usize;
    let mut inflight = 0usize;
    let mk = |slot: usize, off: u64, bufs: &mut [Vec<u8>]| {
        opcode::Read::new(types::Fd(fd), bufs[slot].as_mut_ptr(), bs as u32)
            .offset(off)
            .build()
            .user_data(slot as u64)
    };
    while next < n && inflight < qd {
        let e = mk(inflight, offsets[next], &mut bufs);
        unsafe {
            ring.submission()
                .push(&e)
                .map_err(|_| std::io::Error::other("sq push"))?;
        }
        next += 1;
        inflight += 1;
    }
    ring.submit()?;
    while inflight > 0 {
        ring.submit_and_wait(1)?;
        let done: Vec<usize> = ring.completion().map(|c| c.user_data() as usize).collect();
        for slot in done {
            inflight -= 1;
            if next < n {
                let e = mk(slot, offsets[next], &mut bufs);
                unsafe {
                    ring.submission()
                        .push(&e)
                        .map_err(|_| std::io::Error::other("sq push"))?;
                }
                next += 1;
                inflight += 1;
            }
        }
        ring.submit()?;
    }
    Ok(sqpoll)
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
    let elayout = colibri_engine::moe::ExpertLayout::for_arch(cfg.arch);
    let wn =
        |l: usize, e: usize, suf: &str| format!("model.layers.{l}.mlp.experts.{e}.{suf}.weight");
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
    let names = |e: usize| {
        [
            wn(layer, e, "gate_proj"),
            wn(layer, e, "up_proj"),
            wn(layer, e, "down_proj"),
        ]
    };
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

    println!(
        "{:<34} {:>9} {:>10} {:>8}",
        "phase", "total ms", "ms/expert", "GB/s"
    );
    let report = |name: &str, secs: f64, bytes: f64| {
        let gbs = if bytes > 0.0 {
            format!("{:.2}", bytes / secs / 1e9)
        } else {
            "-".into()
        };
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
            let ex = colibri_engine::moe::load_expert(
                &shards, elayout, hidden, moe_inter, 4, layer, e, t,
            )
            .expect("load_expert");
            std::hint::black_box(&ex);
        }
        full[i] = report(
            &format!("full load_expert (T={t})"),
            t0.elapsed().as_secs_f64(),
            total_bytes,
        );
    }

    // 2b. Pooled batch: all experts through one continuously-streaming worker set
    //     (the COLI_READER_POOL path) vs the per-expert spawn/join above. Same
    //     work (full Experts, scales + QTensor), so this isolates the loader shape.
    {
        let eids: Vec<usize> = (0..n_experts).collect();
        let t0 = std::time::Instant::now();
        let exps = colibri_engine::moe::load_experts_batch(
            &shards, elayout, hidden, moe_inter, 4, layer, &eids, threads,
        )
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
        read[i] = report(
            &format!("coalesced read, fresh alloc (T={t})"),
            t0.elapsed().as_secs_f64(),
            total_bytes,
        );
    }

    // 5. Read into one REUSED, pre-faulted buffer (single-thread) — no allocation.
    let mut reused = vec![1u8; span];
    let t0 = std::time::Instant::now();
    for e in 0..n_experts {
        let nm = names(e);
        let mut off = 0;
        for (j, n) in nm.iter().enumerate() {
            shards
                .read_raw(n, &mut reused[off..off + sizes[j]])
                .expect("read_raw");
            off += sizes[j];
        }
        std::hint::black_box(&reused);
    }
    let reused_s = report(
        "read into reused buffer (T=1)",
        t0.elapsed().as_secs_f64(),
        total_bytes,
    );

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
    let alloc_s = report(
        "fresh alloc + page-touch only",
        t0.elapsed().as_secs_f64(),
        total_bytes,
    );

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
    let spawn_s = report(
        &format!("thread spawn/join only ({nt} thr)"),
        t0.elapsed().as_secs_f64(),
        0.0,
    );

    let ms = |s: f64| s * 1e3 / n_experts as f64;
    println!("\nattribution (ms/expert, warm):");
    println!(
        "  chunking delta (full T={threads} vs T=1): {:+.3}",
        ms(full[0]) - ms(full[1])
    );
    println!(
        "  alloc cost (fresh vs reused read):        {:+.3}  (direct alloc+fault: {:.3})",
        ms(read[1]) - ms(reused_s),
        ms(alloc_s)
    );
    println!(
        "  scales + QTensor (full - read, T=1):      {:+.3}  (scales alone: {:.3})",
        ms(full[1]) - ms(read[1]),
        ms(scales_s)
    );
    println!(
        "  spawn overhead ({nt} threads):              {:.3}",
        ms(spawn_s)
    );
    println!(
        "  pure read, reused buf:                    {:.3}",
        ms(reused_s)
    );
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
///
/// **This is [`RUNTIME_RESERVE`], not a second quantity.** They were declared separately,
/// both at 10 GiB, with near-identical doc comments — "CUDA context, expert read buffers,
/// activations, allocator slack" versus "prefill activations, GPU host staging, expert
/// read buffers, the CUDA context and allocator slack" — and sat on two different live
/// paths: this one under `ram_budget` → [`budget_from`], the other under
/// [`clamp_fill_to_headroom`]. Anyone tuning "the reserve" would fix one and silently miss
/// the other. Aliased rather than deleted so both names keep their (different, both
/// accurate) explanations of what the reserve is for.
const WORKING_RESERVE: u64 = RUNTIME_RESERVE;

/// Share of `MemTotal` the process aims to occupy in total — experts, dense weights and
/// its own runtime combined. Experts take whatever is left after the other two.
///
/// The goal is **maximum expert residency**: RAM held back is RAM not caching a model.
/// Only an *external* tenant should push us off it, which is what
/// [`ADAPTIVE_DANGER_FLOOR`] and the swap guard in `spawn_adaptive_budget` are for.
///
/// The remaining 4% is left to the kernel itself — slab, page tables (a 100 GB mapping
/// needs ~200 MB of PTEs alone), network and block buffers, and the few hundred MB of
/// userland already running. Taking it is what turns a fill into paging.
const TARGET_RAM_PCT: u64 = 96;

/// Non-expert RAM the serving process needs at peak, **excluding KV**: prefill
/// activations, GPU host staging, expert read buffers, the CUDA context and allocator
/// slack.
///
/// This was 20 GB and conflated two different things: KV is not a constant — it scales
/// with context and with live request count — so a flat allowance for it double-charged
/// against the dynamic handling and cost ~15 GB of expert residency on every model.
///
/// **The original justification named the wrong mechanism, and that matters if you are
/// thinking of shrinking this further.** It cited `ExpertCache::reserve_ram` "having
/// callers subtract a request's KV from the standing budget". `reserve_ram` has **no
/// callers** — every reference to it in the tree is a doc comment or its own rollback
/// path, and `COLI_GUARD_TRACE=1` on a serve run shows `reserved=0.00 GB` on every tick.
///
/// What actually admits KV is the ledger, in `serve::handle_completion`:
/// `commit_or_wait(Class::Kv, kv_bytes, rigid, …)` against
/// `rigid = ceiling − Dense − Experts`, with a real 507 and a queue path. So KV *is*
/// bounded and cannot OOM the box — but it is bounded *after* experts rather than by
/// evicting them, which is a separate gap (experts win over KV at long context).
///
/// The 20 → 10 GB cut may still be right; it is the *reasoning* that was unfounded. Since
/// the per-tick `supported_cap` ceiling landed, the monitor also corrects a too-small
/// reserve dynamically by capping the cache against real free memory — which is arguably
/// the better mechanism and an argument for this constant mattering less, not more.
const RUNTIME_RESERVE: u64 = 10 << 30;

/// Ceiling on total process footprint: [`TARGET_RAM_PCT`] of RAM.
fn ram_target(total: u64) -> u64 {
    total / 100 * TARGET_RAM_PCT
}

/// Bound any requested expert-cache fill by what is left after the model's **non-expert**
/// RAM, within the [`TARGET_RAM_PCT`] footprint: the resident dense tier plus
/// [`RUNTIME_RESERVE`].
///
/// There is deliberately no way to bypass this. A hand-set byte budget (the removed
/// `COLI_RAM_GB`) used to skip it: on a 121 GiB Spark a 110 GB request drove the serve
/// process to 108.7 GiB RSS **and into swap** (3 GB paged out) with throughput at
/// 0.06-0.24 tok/s — worse than the 40 GB default it was meant to beat. The caller cannot
/// know the resident dense tier, the KV for the live context, or the GPU's share of a
/// unified pool; all three are known here.
///
/// `resident` is the **host** dense tier; the GPU's duplicate of it is added here rather
/// than at the call sites. Charging it to the ledger but not to this clamp is exactly the
/// bug that shipped once: GLM-5.2 reported `dense 34 GB` (17 host + 17 duplicate) and, in
/// the same breath, `fill to ~89 GB` — 123 GB of intent on a 121 GB box. It did not swap,
/// because swap is off; earlyoom SIGTERMed it 43 s in. Deriving the duplicate inside the
/// clamp means a future call site cannot forget it.
fn clamp_fill_to_headroom(requested: u64, resident: u64, total: u64) -> u64 {
    let dense = resident.saturating_add(colibri_engine::ram::device_duplicate_bytes());
    fill_within_headroom(requested, dense, total)
}

/// The arithmetic of [`clamp_fill_to_headroom`], taking the *full* dense cost (host copy
/// plus any device duplicate) explicitly.
///
/// Split out so the invariant can be tested without writing the process-global duplicate:
/// tests run in parallel, and a test that mutates shared state to set up its own case has
/// silently changed another test's math in this repo before.
fn fill_within_headroom(requested: u64, dense_ram: u64, total: u64) -> u64 {
    let headroom = ram_target(total)
        .saturating_sub(dense_ram)
        .saturating_sub(RUNTIME_RESERVE);
    requested.min(headroom)
}

/// Adaptive max-residency danger line: only when `MemAvailable` stays under this (a real
/// other tenant, since the reserve absorbs our own runtime) does the monitor cede gradually.
const ADAPTIVE_DANGER_FLOOR: u64 = 4 << 30;

/// Floor of the hard OOM-guard line. The *effective* floor is [`oom_guard_floor`], which
/// raises this to clear whatever will actually kill us.
///
/// The monitor evicts LRU experts immediately (no hysteresis) whenever `MemAvailable` falls
/// below the effective floor — whatever consumed the memory, including the GPU on GB10's
/// unified pool. This is the guarantee that filling RAM can never OOM the box; it sits below
/// [`ADAPTIVE_DANGER_FLOOR`] (the softer, external-tenant line).
const ADAPTIVE_HARD_FLOOR: u64 = 3 << 30;

/// earlyoom's SIGTERM threshold as a percent of `MemTotal`, discovered from the running
/// process, or `None` if earlyoom is not running.
///
/// Read from `/proc/*/cmdline` rather than by shelling out to `ps`: this runs during
/// startup on a box we are about to fill, and forking is exactly what a nearly-full box
/// handles worst.
///
/// **A running earlyoom whose flags cannot be parsed yields its documented default of
/// 10%, not `None`.** Guessing low here means the guard sits under the kill line and never
/// fires, which is the failure this whole function exists to prevent; guessing high only
/// costs residency.
#[cfg(target_os = "linux")]
fn earlyoom_sigterm_pct() -> Option<u64> {
    const DEFAULT_PCT: u64 = 10; // `earlyoom --help`: "default 10 %"
    let entries = std::fs::read_dir("/proc").ok()?;
    for e in entries.flatten() {
        let raw = match std::fs::read(e.path().join("cmdline")) {
            Ok(r) => r,
            Err(_) => continue,
        };
        // cmdline is NUL-separated argv.
        let args: Vec<String> = raw
            .split(|b| *b == 0)
            .filter(|s| !s.is_empty())
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .collect();
        if args.is_empty() || !args[0].contains("earlyoom") {
            continue;
        }
        // `-m PERCENT` or `-m PERCENT,KILL_PERCENT`; the SIGTERM line is the first field.
        let pct = args
            .iter()
            .position(|a| a == "-m")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.split(',').next())
            .and_then(|v| v.trim().parse::<u64>().ok());
        return Some(pct.unwrap_or(DEFAULT_PCT).clamp(1, 99));
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn earlyoom_sigterm_pct() -> Option<u64> {
    None
}

/// The OOM guard's floor, derived from what will actually kill this process.
///
/// **[`ADAPTIVE_HARD_FLOOR`] alone is only safe because this one box was hand-tuned.**
/// earlyoom's threshold is a *percentage of MemTotal*; ours was an absolute 3 GiB. Those
/// cross:
///
/// | config | earlyoom SIGTERMs at | old 3 GiB floor | result |
/// |---|---|---|---|
/// | gx10-42b2, `-m 2` (tuned by `sparkrun`) | 2.61 GB | 3.22 GB | safe by 0.61 GB |
/// | **stock earlyoom, `-m 10` (the default)** | **13.07 GB** | 3.22 GB | **killed; guard idle** |
/// | 256 GB host at `-m 2` | 5.12 GB | 3.22 GB | **killed; guard idle** |
///
/// It is fatal rather than merely suboptimal because [`TARGET_RAM_PCT`] deliberately drives
/// toward leaving only ~4% free. On a stock-earlyoom host that means crossing the kill line
/// on essentially every model, every run, while `MemAvailable` is still nowhere near the
/// guard's own 3.22 GB — so the entire adaptive OOM guard is decorative on any untuned host.
/// Even at `-m 2` it breaks above ~161 GB of RAM.
///
/// The margin above the trigger is a quarter of it, floored at 1 GiB: the *rate*-dependent
/// part of the stopping distance is already handled by `braking_floor` in the monitor, so
/// this only has to cover the gap between "we noticed" and "earlyoom noticed".
fn oom_guard_floor(total: u64, earlyoom_pct: Option<u64>) -> u64 {
    match earlyoom_pct {
        // No earlyoom: the kernel OOM killer is the backstop. It is far less eager (it
        // fires on allocation failure, not on a percentage), so the static floor stands.
        None => ADAPTIVE_HARD_FLOOR,
        Some(pct) => {
            let trigger = total / 100 * pct;
            let margin = (trigger / 4).max(1 << 30);
            ADAPTIVE_HARD_FLOOR.max(trigger.saturating_add(margin))
        }
    }
}

/// The expert cache is capped at `MemTotal / CACHE_CAP_DIVISOR`.
///
/// **Measured on a 121 GiB Spark, 8/4 model (~19 GiB resident), 12 diverse prompts,
/// counterbalanced order** (each config run at mirrored positions so a drift cannot
/// masquerade as a config effect — every earlier ascending-order sweep here was
/// uninterpretable for exactly that reason):
///
/// | budget (GB)   | RSS   | swap  | tok/s |
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
/// evicts every expert immediately and thrashes).
const MIN_BUDGET: u64 = 4 << 30;

/// Expert-cache byte budget, reserving `reserve` bytes the caller knows it will
/// still allocate (e.g. `serve`'s KV cache) on top of [`WORKING_RESERVE`].
///
/// There is deliberately NO manual override. The budget is chosen adaptively from
/// `MemAvailable` and the [`CACHE_CAP_DIVISOR`] ceiling, and a background monitor evicts
/// under pressure — a fixed number cannot react to any of that. Every measured sweep of
/// the old `COLI_RAM_GB` knob came out flat or worse, and the one thing it reliably did
/// was let a caller pin a budget past the thrash cliff.
///
/// Dense weights are deliberately *not* subtracted: `load_model_with` materializes
/// them eagerly and every caller budgets *after* the load, so `MemAvailable` has
/// already excluded them. Subtracting again would double-count ~11 GiB.
fn ram_budget_reserving(reserve: u64) -> u64 {
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

/// A model whose experts fit in the achievable cache to at least this % is "near-fit": it
/// gets `fadvise` plus a fill-to-`natural_fill` residency hold, collapsing the page-cache
/// double-hold so the working set stays resident. Below it, a model keeps the page cache
/// as a second tier and streams against the `MemTotal/CACHE_CAP_DIVISOR` cap.
///
/// **Was 118, which is above 100% — i.e. residency was reserved for models that fit
/// entirely, and an 86%-coverage model got the 40 GB streaming cap instead. That cost 8.5×.**
///
/// Measured on a rebooted 121 GiB Spark, MiniMax-M2.7 (86% coverage, 117 GB of experts),
/// `bench_serve` 12 distinct prompts x 32 tok, two binaries differing only in this
/// constant, arms alternating, 4 passes each:
///
/// | | pass 1 | pass 2 | pass 3 | pass 4 | swap |
/// |---|---|---|---|---|---|
/// | 40 GB + page cache (118) | 0.30 | 0.51 | 0.81 | 0.62 tok/s | 0 |
/// | **max residency (80)** | **4.51** | **5.26** | **5.23** | **5.30 tok/s** | **0** |
///
/// The old comment recorded this same configuration as *bimodal 0.9–4.3 tok/s* and chose
/// the streaming cap for stability. That instability is gone — passes 2-4 span 1.3% — and
/// three things changed in between: mmap serves resident spans as views instead of heap
/// copies (so residency no longer double-holds), the buffer-pool cap fix removed the
/// alloc/free churn, and the fill can no longer page the box out (unconditional headroom
/// clamp + a swap guard that evicts on swap growth). `swap=0M` across every pass.
///
/// 80 sits below M2.7's 86% and above MiniMax-M3's 46%.
///
/// **The gate is necessary — extending max residency to every model was tried and it
/// fails catastrophically below this line.** Filling M3 (43% coverage) to its 94 GB
/// headroom exhausted all 16 GB of swap and the model generated **zero tokens**; the run
/// had to be killed. GLM and K3 never got to run. This is not "slower", it is broken, and
/// it is the failure the original bar existed to prevent — it was only ever set at the
/// wrong value (118, above 100%, which excluded a model that measured 8.5x faster with
/// residency).
///
/// Why the guards did not save it, and the part worth remembering: **for a mapped model,
/// evicting from our cache does not promptly return memory.** Dropping the `Arc` releases
/// a *view*; the underlying file pages stay in the page cache until the kernel reclaims
/// them on its own schedule. So the swap guard in `spawn_adaptive_budget` fires, evicts,
/// and the pressure does not fall — meanwhile the monitor's own 100 ms tick is competing
/// with a thrashing box. Our eviction is not an effective pressure-relief valve once the
/// resident set is mapped, which makes staying out of that state the only real defence.
///
/// So the shape is: residency is a large win when the set nearly fits (M2.7 87%: 8.5x) and
/// a hard failure when it does not (M3 43%). The crossover between 43 and 87 is **not**
/// measured; 80 is bracketed by them and deliberately nearer the safe end.
const NEARFIT_COVERAGE_PCT: u64 = 80;

/// Coverage at or above which shard reads stay **buffered**; below it we switch to
/// `O_DIRECT`. `COLI_O_DIRECT=0|1` overrides in either direction.
///
/// The page cache is a real second tier — but only if it can hold a useful share of the
/// expert set. Coverage (`natural_fill / total_expert_bytes`) is exactly that ratio, so
/// it is the right axis, and the crossover is sharp. Measured on 42b2, ABBA-mirrored,
/// 2 passes per arm, 12-token prompt, ngen 6, tokens byte-identical in every arm
/// (`expert-load` ms — wall carries a one-time model load and swings ~20%):
///
/// | model | coverage | buffered | `O_DIRECT` | |
/// |---|---|---|---|---|
/// | Kimi-K3 | 7% | 31350 | **28788** | O_DIRECT 1.089× |
/// | GLM-5.2 | 27% | 16957 | **14812** | O_DIRECT 1.145× |
/// | MiniMax-M3 | 47% | **6232** | 6755 | buffered |
/// | MiniMax-M2.7 | 86% | **4166** | 4444 | buffered |
///
/// Monotonic, and the mechanism is visible in the byte counters rather than inferred:
/// K3's two arms read the SAME device bytes (200.6 vs 202.1 GiB) because at 7% the page
/// cache serves ~nothing, so bypassing it is free; M2.7's buffered rep read **zero**
/// device bytes because at 86% the page cache holds essentially the whole working set,
/// and bypassing it forfeits all of that.
///
/// 35% sits between the two measured neighbours (27% and 47%). The exact crossover
/// inside that interval is not measured — the two nearest points are what bound it.
///
/// This supersedes the older "leave off" advice, which came from ONE model measured
/// before the coverage axis was understood (and on GLM's older, 735 GB e4m3 container —
/// nearly double today's 379 GB, i.e. a different coverage entirely).
const O_DIRECT_MAX_COVERAGE_PCT: u64 = 35;

/// Coverage **at or above** which the expert FFN stops staging its weights host→device.
/// `COLI_FFN_DEVCOPY` overrides. The fourth decision off the coverage axis, and like the
/// others it has its own measurement and its own threshold.
///
/// Staging costs GPU time and buys I/O decoupling; only the second half is model-dependent.
/// Measured 2026-08-06, one binary, env only, ABBA, tokens identical in every arm:
///
/// | model | coverage | staging OFF vs ON | n/arm |
/// |---|---|---|---|
/// | nemotron-3-super | ~155% | **−10.3%** | 4 |
/// | minimax-m2.7 | ~97% | **−8.6%** | 4 |
/// | minimax-m3 @512 | ~37% | −2.5% | 6 |
/// | minimax-m3 @128 | ~37% | −1.45% | 4 |
/// | **glm-5.2** | **~18%** | **+11.5% WORSE** | 2 |
///
/// Monotonic in coverage. At 18% nearly every expert streams from NVMe *during* the forward
/// pass, so a zero-copy kernel reads the very host pages the loader is writing — GLM pays
/// `expert-load` +10.4 s to save `gpu-ffn` 1.5 s. At 97–155% there is no live streaming to
/// collide with, and the copy is pure waste.
///
/// **BOUNDED, NOT DERIVED.** The crossover lies somewhere in (18%, 37%); 35 is the existing
/// neighbour threshold and both measured points fall on the correct side of it. Do not read
/// it as a measured optimum — the same caveat the rows-per-expert gate carries.
///
/// The trap worth remembering: CUDA events show the kernel is *faster* without the copy
/// (6.21 s vs 6.70 s — unified memory has no faster tier to copy into), so a compute-only
/// benchmark concludes "always disable" and regresses GLM by 11.5%. The benefit is invisible
/// to any measurement that excludes the loader.
const FFN_DEVCOPY_MAX_COVERAGE_PCT: u64 = 35;

/// Coverage below which the expert reader uses a **narrow** thread pool
/// ([`DISK_BOUND_READ_THREADS`]) instead of `2 x cores`. `COLI_LOAD_THREADS` overrides.
///
/// A constant thread count is wrong for the fleet. Measured on 42b2, decode, interleaved,
/// token-identical, 34 threads vs 8:
///
/// | model | coverage | 34 | 8 | |
/// |---|---|---|---|---|
/// | Kimi-K3 | 7% | 26364 ms | **22531 ms** | 8 wins 1.17x |
/// | GLM-5.2 | 26% | 12017 ms | **10402 ms** | 8 wins 1.15x |
/// | MiniMax-M3 | 46% | **3718 ms** | 8611-16092 ms | 8 loses **2-4x** |
/// | MiniMax-M2.7 | 86% | 481 ms | 485 ms | neutral (all mmap views) |
/// | Nemotron-3 | 172% | **12.0 s preload** | 29.5 s preload | 8 loses **2.5x** |
///
/// The two winners are the two models whose bytes genuinely come off the drive: GLM reads
/// device bytes equal to reader bytes in *both* arms (ratio 1.00, invariant under thread
/// count), so it saturates the NVMe at low queue depth and extra threads only cost. M3's
/// ratio swings 0.26 -> 1.13 with thread count, so it is not disk-bound in the same sense.
///
/// 35 sits between the measured neighbours 26 (GLM, wants narrow) and 46 (M3, wants wide);
/// the exact crossover in that interval is **not** measured. It coincides with
/// [`O_DIRECT_MAX_COVERAGE_PCT`] by accident, not by derivation — do not merge them, the
/// same mistake was already made and caught with the mmap threshold.
const DISK_BOUND_COVERAGE_PCT: u64 = 35;

/// Reader threads for a disk-bound model. The GLM sweep is flat across 8-16
/// (11.40 / 11.42 / 11.32 GB/s at 8 / 12 / 16) and falls off hard outside it
/// (5.82 at 2, 9.87 at 34), so 12 is the middle of the plateau rather than the single
/// fastest sample.
const DISK_BOUND_READ_THREADS: usize = 12;

/// Coverage at or above which resident expert spans are served as **mapped views**
/// instead of being copied into heap buffers. `COLI_MMAP_EXPERTS=0|1` overrides.
///
/// Mapping only pays when spans are *actually resident when touched* — otherwise every
/// span pays a `mincore` and then reads anyway. Coverage is a loose proxy for that, and
/// the crossover is much higher than [`O_DIRECT_MAX_COVERAGE_PCT`]. Measured on 42b2,
/// ABBA, 4 runs/arm, tokens identical in every run:
///
/// | model | coverage | pread | mapped | |
/// |---|---|---|---|---|
/// | MiniMax-M2.7 | 86% | 2987 ms | **529 ms** | **5.65×** |
/// | MiniMax-M3 | 47% | **7851 ms** | 8605 ms | 0.91× |
/// | GLM-5.2 | 26% | **14694 ms** | 15080 ms | 0.97× |
///
/// The byte counters explain the split better than coverage does. M2.7 touches 33.6 GB
/// and both arms read **zero** from the drive — it stays resident, so mapping deletes a
/// pure memcpy. M3 touches 60.6 GB and reads **67 GiB in both arms**: at 47% coverage
/// almost nothing survives in page cache next to the 40 GB expert cache, so the residency
/// gate mostly fails and its `mincore` is dead weight.
///
/// **I initially gated this on `O_DIRECT_MAX_COVERAGE_PCT` (35%), reasoning that one
/// number should decide both. That was wrong** — it would have enabled mapping for M3 and
/// cost 9.6%. 80% sits between the measured loss (47%) and win (86%); the crossover inside
/// that interval is not measured, and it is deliberately conservative because the downside
/// is paid on every span while the upside needs residency.
///
/// **This threshold now collides with [`NEARFIT_COVERAGE_PCT`], which is also 80.** The two
/// fire on exactly the same models and want the same RAM in different tiers: near-fit fills
/// the *heap* expert cache to the 96% ceiling, this serves spans from the *page cache*. The
/// heap wins, because it is explicit. Measured on M2.7 after max residency shipped: the heap
/// took ~110 GB, page cache was left ~11 GB, and the mapped path served **1505 of 18571
/// spans (8%)** while the run drained 135.9 GB from the drive.
///
/// So the 5.65× above is **stale for the configuration we actually run**, and so is "M2.7
/// reads zero from the drive". Max residency is still the right trade — it measured 8.5× on
/// M2.7 serve against the 40 GB streaming cap, and a small heap next to a large page cache
/// is precisely the arm that lost. But do not cite this table as current, and do not tune
/// this constant expecting it to matter until the tier competition is resolved.
///
/// **Raised 80 → 100 on 2026-07-31: at 80 it was a measured 1.27× REGRESSION on M2.7.**
///
/// The threshold is no longer a proxy — it is the condition this doc already states two
/// paragraphs up, that mapping only pays when spans are *actually resident when touched*.
/// 100% is that condition. Anything less means some fraction of touches is a synchronous
/// disk read, and the grouped expert path turned that from a reader's problem into
/// everyone's: the pack memcpys every staged expert on the CPU, so a non-resident page is a
/// **major fault inside the copy**.
///
/// M2.7 at 84% coverage, 128-tok prefill, 2 reps ABBA, token-identical:
///
/// | | wall | pack | pgmajfault | moe | expert-load |
/// |---|---|---|---|---|---|
/// | **mapped off** | **39-43 s** | **1.96-2.13 s** | **0** | **18.9 s** | 12.1 s |
/// | mapped on (80) | 49-56 s | 7.79-15.56 s | 159k-311k | 28.8 s | 13.5 s |
///
/// Note `expert-load` is no better mapped — the tier this gate exists to help does not gain,
/// while `moe` loses 1.52×. It also removes a variance problem, not just a mean one: mapped
/// off spans 1.96-2.13 s, mapped on 7.79-15.56 s. That spread is the "M2.7 bimodality" that
/// two other hypotheses (pack thread count, buffer-pool cap) were wrongly blamed for.
///
/// Nemotron (155% coverage) still qualifies and is unaffected in practice — it preloads, so
/// its expert-load is ~2 ms either way. GLM (18%) and M3 (37%) were already below the gate.
const MMAP_MIN_COVERAGE_PCT: u64 = 100;

/// Wire the expert cache to **fill RAM safely, for every model**. It grows toward a fill
/// target and a background monitor evicts LRU experts under memory pressure so the box can
/// never OOM — which is what lets us point a fill-RAM policy at a model of *any* size:
///
/// - **near-fit** (experts ≈ RAM) → fill to `total − reserve` **plus fadvise** — the whole
///   working set resident, no page-cache double-hold.
/// - **≫ RAM** (experts ≫ RAM) → hold only `MemTotal / CACHE_CAP_DIVISOR` with **fadvise
///   off**, letting the OS page cache serve the streaming tail as a second tier. Holding more
///   *thrashes* (a ~101 GB hold collapsed M3 decode to ~0.7 tok/s); `MemTotal/3` is the settled
///   sustainable ceiling — see `memory-ceiling-is-real` / `autopin-single-node-negative`.
///
/// Off-Linux (no `/proc/meminfo`) it no-ops — there is no live pressure signal to evict on.
/// Returns whether the model is being held at MAX residency
/// (near-fit) — i.e. the whole expert set fits and the cache
/// Bytes of routed experts this node holds: `per-expert × experts-owned × MoE-layers`.
///
/// Split out as a pure function purely so the MoE-layer count is testable without a
/// provider, RAM probe or container. That count is the part that has broken twice, and
/// it breaks SILENTLY — deriving it from `layer_kind.is_empty()` returns 0 for Kimi-K3,
/// whose `layer_kind` is populated with mixer kinds (Kda/Attn) while every layer past
/// `first_dense` still carries experts. A 0 here reports "~0 GB experts / 0% coverage"
/// and hands every coverage-derived decision a model that appears to have no experts.
/// Always via [`colibri_core::Config::moe_layers`], never `layer_kind` directly.
fn expert_footprint(cfg: &colibri_core::Config, per_expert: u64, owned_experts: u64) -> u64 {
    let (n_moe, _probe) = cfg.moe_layers();
    per_expert
        .saturating_mul(owned_experts)
        .saturating_mul(n_moe as u64)
}

/// will hold it with no eviction. `preload_all_experts` uses this to decide whether an
/// eager preload is safe (it must never fire for a ≫-RAM model).
fn wire_adaptive_cache<P>(
    provider: &std::sync::Arc<colibri_engine::ExpertCache<P>>,
    cfg: &colibri_core::Config,
    ebits: u32,
    owned: &[u32],
    resident: u64,
) -> bool
where
    P: colibri_engine::ExpertProvider + Send + Sync + 'static,
{
    use colibri_engine::ExpertProvider as _; // bring `.expert()` into method scope
    let total = match colibri_engine::total_ram_bytes() {
        Some(t) => t,
        None => return false, // non-Linux: no live pressure signal, leave the static budget
    };
    // Number of MoE layers and the index of one, to size the streamed-expert footprint.
    // Homogeneous arches (GLM/MiniMax): every layer at/after `first_dense` is MoE. Nemotron-H
    // is hybrid (Mamba/attn/MoE by index) — layer `first_dense` (0) is a *Mamba* layer with
    // NO experts, so probing it falls to the dense-width fallback and both the count and the
    // per-expert size come out wildly wrong (~17× → misclassifies a RAM-fitting model as
    // ≫-RAM, leaving experts non-resident). Probe an actual MoE layer and count only MoE ones.
    // Ask `Config::moe_layers`, never `layer_kind` directly. Branching on
    // `layer_kind.is_empty()` is right for Nemotron-H (populated, one kind per layer)
    // and for homogeneous arches (empty), but WRONG for Kimi-K3: its `layer_kind` is
    // non-empty yet carries only mixer kinds (Kda/Attn) because every K3 layer has BOTH
    // a mixer and an FFN. Counting `Moe` entries there yields 0, so K3 reported
    // "~0 GB experts / 0% coverage" and any coverage-derived decision saw a model with
    // no experts at all. `moe_layers` branches on whether a `Moe` entry exists and
    // falls back to the `first_dense` prefix rule, which is correct for all three.
    let (n_moe, probe_layer) = cfg.moe_layers();
    let n_moe = n_moe as u64;
    // Size an expert from a real one on disk — its QTensors carry the true format
    // (NVFP4 fmt=5, e4m3 fmt=4, …), so block-scale overhead and the actual bit-width
    // are exact (for Nemotron this also captures the latent input dim, not `hidden`). The
    // `ebits` estimate is only a fallback: it reflects the *requested* resident dense width
    // (default 8), not the streamed experts' real format, and would mis-decide coverage.
    //
    // Probe an expert this node **owns**: the sharded provider refuses a peer's expert, so
    // probing a hardcoded 0 fails on every rank but 0 and silently drops to the `ebits`
    // fallback — which measured 15.79 MB against Nemotron's real 3.1 MB (5× over).
    let probe_expert = owned.first().copied().unwrap_or(0) as usize;
    let per = match provider.expert(probe_layer, probe_expert) {
        Ok(e) => (e.gate.bytes() + e.up.bytes() + e.down.bytes()) as u64,
        Err(_) => colibri_engine::capacity::bytes_per_expert_of(cfg, ebits),
    };
    // Only the experts THIS node owns are ever loaded here, so coverage must be measured
    // against the shard — not `cfg.n_experts`. Using the whole model's count made a 2-rank
    // worker see ~630 GB against 121 GB of RAM (16% "coverage"), pick the ≫-RAM regime, cap
    // at 40 GB and never preload — reintroducing the lazy cold-load that #42 fixed, on every
    // worker, in a way that would only show up as bad 2-node numbers.
    let owned_experts = if owned.is_empty() {
        cfg.n_experts as u64
    } else {
        owned.len() as u64
    };
    let total_expert_bytes = expert_footprint(cfg, per, owned_experts);
    debug_assert_eq!(
        total_expert_bytes,
        per.saturating_mul(owned_experts).saturating_mul(n_moe)
    );
    // Achievable expert residency: the `TARGET_RAM_PCT` footprint minus the dense tier and
    // the non-KV runtime. Coverage is then model-intrinsic — "what share of this model's
    // experts can actually stay resident on this box?" — and drives the cache regime, the
    // read path (O_DIRECT), the mapped-view path, and the reader thread count.
    //
    // This is deliberately the SAME expression as `clamp_fill_to_headroom`, so the number
    // coverage is computed from is the number that will actually be held. It previously
    // was not: coverage used `total - 20 GB` while the clamp used
    // `total - resident - 20 GB`, so a model with a large dense tier (Kimi-K3, 63 GB) was
    // classified against a fill it could never reach.
    let natural_fill = clamp_fill_to_headroom(u64::MAX, resident, total);
    let covers_pct = if total_expert_bytes > 0 {
        natural_fill.saturating_mul(100) / total_expert_bytes
    } else {
        0
    };
    let near_fit = covers_pct >= NEARFIT_COVERAGE_PCT;
    // Pick the read path from the same coverage number, before any expert is read.
    // O_DIRECT pays exactly when the page cache is too small to be a useful second tier
    // for this model's expert set; see `O_DIRECT_MAX_COVERAGE_PCT`.
    // Two decisions off the same axis, but NOT the same threshold — they were measured
    // separately and the crossovers do not coincide (see each constant).
    colibri_safetensors::set_o_direct(covers_pct < O_DIRECT_MAX_COVERAGE_PCT);
    colibri_safetensors::set_mmap_experts(covers_pct >= MMAP_MIN_COVERAGE_PCT);
    // Fourth decision off the same axis: stage expert weights H2D only while the model is
    // genuinely streaming them, because staging's whole value is decoupling the kernel from
    // pages the loader is still writing. See `FFN_DEVCOPY_MAX_COVERAGE_PCT`.
    #[cfg(feature = "cuda")]
    colibri_backend::cuda::set_ffn_devcopy(covers_pct < FFN_DEVCOPY_MAX_COVERAGE_PCT);
    // Third decision off the same axis, with its own threshold and its own measurement.
    // A model whose experts genuinely stream off the drive saturates the NVMe at a low
    // queue depth, and every reader thread past that only adds spawn cost (the drain
    // builds a fresh pool per batch) and contention. Models that do not stream want the
    // wide pool — for them, narrowing it is a 2-4x regression. See
    // `colibri_engine::moe::set_disk_bound_read_threads` for the fleet table.
    if covers_pct < DISK_BOUND_COVERAGE_PCT {
        colibri_engine::moe::set_disk_bound_read_threads(DISK_BOUND_READ_THREADS);
    }
    // Print what the coverage number actually decided. Four path choices now hang off it and
    // every one of them is silent when it goes the wrong way — the failure mode that has
    // cost this project the most time is a dispatch that quietly picks the slow arm. One
    // line, so a profile can be read against the paths it actually ran.
    // Report the EFFECTIVE state, not the coverage-derived one: `COLI_FFN_DEVCOPY` overrides
    // the gate, and a line that still printed the gate's opinion would be actively wrong in
    // exactly the runs someone is debugging. Mirrors `ffn_devcopy_on()` in backend_cuda.cu.
    let staging_gate = covers_pct < FFN_DEVCOPY_MAX_COVERAGE_PCT;
    let staging = match std::env::var("COLI_FFN_DEVCOPY").ok().as_deref() {
        Some("0") => "off (COLI_FFN_DEVCOPY)",
        Some(_) => "on (COLI_FFN_DEVCOPY)",
        None if staging_gate => "on",
        None => "off",
    };
    eprintln!(
        "[cache] coverage {covers_pct}% → o_direct {} | mmap-experts {} | reader-pool {} | \
         ffn-staging {staging}",
        if covers_pct < O_DIRECT_MAX_COVERAGE_PCT { "on" } else { "off" },
        if covers_pct >= MMAP_MIN_COVERAGE_PCT { "on" } else { "off" },
        if covers_pct < DISK_BOUND_COVERAGE_PCT { "narrow" } else { "wide" },
    );
    // Fill target by regime — **fully derived, no manual override**. Near-fit: fill to
    // `natural_fill`, the whole set nearly fits and the fadvise below keeps MemAvailable
    // honest. ≫-RAM: hold the settled `MemTotal / CACHE_CAP_DIVISOR` ceiling and let the
    // OS page cache serve the streaming tail as a second tier. Holding more *thrashes*:
    // filling ~101 GB collapsed M3 decode to ~0.7 tok/s.
    //
    // `COLI_RAM_GB` used to override this and has been **removed**. It was the one path
    // that skipped the headroom clamp below, and on a 121 GiB Spark `COLI_RAM_GB=110`
    // drove the serve process to 108.7 GiB RSS and into swap (3 GB paged out) with
    // throughput at 0.06-0.24 tok/s. A hand-set byte budget cannot know the resident dense
    // tier, the KV for the live context, or the GPU's share of a unified pool — the three
    // terms that decide whether a fill is safe. Every one of those is available here, so
    // the budget is computed, not asked for.
    // **Max residency for every model.** `natural_fill` is what is left of the 96% ceiling
    // after the dense tier (now including its device duplicate) and `RUNTIME_RESERVE`, so
    // asking for it is asking for every byte that is actually free.
    //
    // This was previously gated to `near_fit` models because extending it to all of them
    // failed hard: MiniMax-M3 filled toward its "94 GB headroom", exhausted 16 GB of swap
    // and generated zero tokens. That fill was never 94 GB of *free* memory. M3's dense
    // tier is 12 GB and `WeightResidency::Upload` duplicates it, and the duplicate was
    // charged to nobody — so the real total was 94 + 12 + 12 = 118 GB on a 121 GB box,
    // before a single KV byte. Three things changed underneath that failure: the duplicate
    // is now in the ledger (so `natural_fill` is ~82 GB for M3, not 94), `Class::Experts`
    // is committed so KV admission can see the cache, and swap is off, so an accounting
    // mistake fails loudly instead of degrading everything for hours.
    //
    // `clamp_fill_to_headroom` below is what bounds this; there is no second regime left to
    // pick, so `near_fit` no longer selects a budget — it only reports.
    let requested = natural_fill;
    // `headroom` accounts for everything in RAM that is NOT experts: the resident dense
    // tier plus the serving process's own runtime (KV, activations, GPU staging, read
    // buffers). It bounds the fill unconditionally. No-op for a small-resident model
    // (GLM: 121-19-20 = 82, so the CACHE_CAP_DIVISOR ceiling still binds); binding for
    // Kimi-K3 (121-63-20 = 38), where without it the cache overcommits and earlyoom
    // SIGTERMs the process mid-forward.
    let fill_target = clamp_fill_to_headroom(requested, resident, total);
    if fill_target < requested {
        eprintln!(
            "[cache] regime fill {} GB exceeds what is left after the model's own RAM \
             ({} GB dense weights + {} GB runtime reserve on {} GB total) — holding {} GB \
             so the box cannot page out",
            requested >> 30,
            resident >> 30,
            RUNTIME_RESERVE >> 30,
            total >> 30,
            fill_target >> 30,
        );
    }
    // **Page-locking expert buffers is OFF: a measured net loss at every budget.** No
    // `set_pin_budget` call, so the default of zero stands and nothing is locked.
    //
    // The idea was sound and it works exactly as designed — a page-locked buffer is DMA'd
    // where it lies instead of being packed into a pinned intermediate first, which took
    // GLM to 93% DMA-direct and all but deleted the pack (7.75 s -> 0.92 s, 300 GB/s). It
    // still loses, monotonically, because the two costs are not the same currency. Measured
    // on GLM, 128-tok prefill, tokens identical in every arm:
    //
    // | COLI_PIN_MAX_MB | locked | pack | expert-load | moe | wall |
    // |---|---|---|---|---|---|
    // | **0 (this)** | 0 | 7.72 s | **18.1 s** | **42.3 s** | **77 s** |
    // | 2048 | 2.2 GB | 7.41 s | 22.4 s | 45.7 s | 79 s |
    // | 8192 | 4.8 GB | 4.73 s | 39.3 s | 59.9 s | 97 s |
    // | 32768 | 12.8 GB | 2.96 s | 47.1 s | 67.1 s | 100 s |
    //
    // The pack costs CPU — ~7 s, and cheap. Locked pages cannot be reclaimed, so they
    // evict page cache, and the reader pays that back as NVMe I/O at +29 s. Every GB
    // locked buys pack time at a worse exchange rate than it sells reader time. Even 2 GB
    // is a loss, so there is no small budget that wins.
    //
    // This also corrects the reasoning that removed an earlier flat cap: locking live
    // anonymous pages is *not* free when swap is off. It does not cost swap — it costs the
    // page cache, which is the tier the streaming models actually run on.
    //
    // Kept behind `COLI_PIN_MAX_MB` rather than deleted: the machinery is what proves the
    // pack is removable, and a future reader that does not lean on the page cache
    // (O_DIRECT throughout, or genuinely resident experts) would flip this trade.
    // Register with the RAM ledger before a single expert is read, so every later
    // allocation — KV, activations, staging — is admitted against a total that already
    // knows what the model itself costs. This is the arbiter that did not exist when the
    // expert cache filled MiniMax-M3 to 94 GB and inference then allocated on top of it.
    let mgr = colibri_engine::ram::init_manager(ram_target(total));
    // Charge the device duplicate too. On GB10 `WeightResidency::Upload` copies the dense
    // tier into the *same* 121 GB pool, so it costs `resident` a second time — 17 GB on
    // GLM-5.2. Load-time sizing already budgets `resident * 2`; the ledger did not, so it
    // was handing that memory out again to KV.
    let dense_ram = resident.saturating_add(colibri_engine::ram::device_duplicate_bytes());
    if let Some(c) = mgr.commit(colibri_engine::ram::Class::Dense, dense_ram) {
        c.hold_forever(); // dense weights live for the process
    }
    // The expert arena is **not** wired here yet. It was, and the first real-model run
    // showed the sizing is wrong in two ways that each make things worse, not better:
    //
    //   - **Mapped models do not use it at all.** MiniMax-M2.7 and Nemotron serve resident
    //     spans as mmap *views*, so they never allocate an expert-sized heap buffer. The
    //     arena was ~102 GB of pre-faulted memory nothing would ever ask for, and touching
    //     it at startup on a box with 84 GB free hung the load before the first token.
    //   - **The slot size did not match real spans.** Slots were sized at `per` (one
    //     expert), but `read_raw_shared_batched` coalesces an expert's projections into one
    //     span that is *larger* than that — GLM asked for ~21 MB against 20 MB slots, so
    //     every request bypassed the arena and allocated fresh while 40 GB sat idle
    //     alongside it (peak RSS 80 GiB).
    //
    // The arena itself is sound and tested (`crates/colibri-core/tests/arena.rs`); what is
    // missing is a slot size derived from the *span* sizes the reader actually requests,
    // and a decision to skip it entirely when spans are served as views. Until then, the
    // committed budget below is the honest one: the ledger tracks the dense tier and
    // admits requests against it, without claiming an arena that does not exist.
    eprintln!(
        // GiB, not GB. These are `>> 30` and were labelled "GB", which reads as a
        // disagreement with the ledger's `[profile] ram peak` line (that one divides by
        // 1e9 and is genuinely GB). Same bytes, two units, one label — it cost an hour of
        // chasing a planner-vs-ledger accounting bug that did not exist.
        "[ram] ceiling {} GiB = {}% of {} GiB | dense {} GiB{} | {} GiB for experts + KV + \
         activations (arena not yet wired: expert memory is still pooled, not pre-granted)",
        mgr.ceiling() >> 30,
        TARGET_RAM_PCT,
        total >> 30,
        dense_ram >> 30,
        if colibri_engine::ram::device_duplicate_bytes() > 0 {
            format!(" (incl. {} GiB device duplicate)", resident >> 30)
        } else {
            String::new()
        },
        mgr.ceiling().saturating_sub(mgr.committed()) >> 30,
    );
    // Swap state changes what a memory-accounting bug looks like, so say which mode this
    // box is in rather than leaving it to be discovered from a mystery slowdown.
    //
    // Swap OFF is the intended configuration for a dedicated inference box. Paging expert
    // weights is strictly worse than the fallback we already have: a miss re-reads one
    // coalesced ~21 MB span at ~11.6 GB/s from the reader, where a swapped page faults
    // back 4 KiB at a time from the same drive, inside a forward pass, un-batchable. Swap
    // also turns an over-commit into a silent 100x slowdown instead of a loud failure —
    // MiniMax-M3 once limped at 0.06 tok/s producing nothing, and left 16 GB paged out
    // that degraded every measurement after it until the box was rebooted.
    match colibri_engine::cache::swap_used_bytes() {
        Some(0) | None => {}
        Some(used) => eprintln!(
            "[ram] WARNING: {} MB is already in swap. Paging is worse than a cache miss \
             here (4 KiB faults inside the forward pass vs one coalesced span in the \
             reader), and it masks accounting bugs as slowdowns rather than failures. \
             `sudo swapoff -a` on a dedicated box; already-swapped pages need \
             `swapoff -a && swapon -a` or a reboot to reclaim.",
            used >> 20
        ),
    }
    // fadvise auto-engages only for a near-fit model: there the whole set is resident, so
    // the page cache is a pure duplicate and dropping it frees RAM for experts. A ≫-RAM
    // model needs the page cache as its second tier. `COLI_FADVISE` still overrides.
    if near_fit {
        colibri_safetensors::set_fadvise(true);
    }
    // Derive the guard's floor from what will actually kill us, not from a constant that
    // happens to suit this host's tuned earlyoom. Say so when it moves: on a stock-earlyoom
    // box this costs real residency, and the operator's better fix is to tune earlyoom down
    // rather than let us hold back 10+ GB.
    let eo_pct = earlyoom_sigterm_pct();
    let hard_floor = oom_guard_floor(total, eo_pct);
    if hard_floor > ADAPTIVE_HARD_FLOOR {
        eprintln!(
            "[ram] OOM-guard floor raised {} -> {} GiB: earlyoom SIGTERMs at -m {}% = {} GiB \
             of this host's {} GiB, which is above the built-in floor. Holding back that \
             much less for experts. `earlyoom -m 2` (see /etc/default/earlyoom) would \
             return it.",
            ADAPTIVE_HARD_FLOOR >> 30,
            hard_floor >> 30,
            eo_pct.unwrap_or(0),
            (total / 100 * eo_pct.unwrap_or(0)) >> 30,
            total >> 30,
        );
    }
    let danger_floor = ADAPTIVE_DANGER_FLOOR.max(hard_floor);
    provider.spawn_adaptive_budget(fill_target, danger_floor, hard_floor);
    // Both regimes now fill to the SAME max-residency target; they differ only in what
    // happens to the page cache. A model whose set fits drops it (`fadvise`) because it is
    // a pure duplicate of what we already hold; a model that still streams a tail keeps it
    // as the second tier for the part that could not be made resident.
    let regime = if near_fit {
        "max residency, fadvise (set fits)"
    } else {
        "max residency, page cache kept for the streaming tail"
    };
    // Name the shard when this node owns only part of the model — otherwise a 2-node log
    // reads as though the box holds the whole thing and the coverage number looks like a bug.
    let scope = if owned_experts < cfg.n_experts as u64 {
        format!(" (shard: {owned_experts}/{} experts)", cfg.n_experts)
    } else {
        String::new()
    };
    eprintln!(
        // GiB throughout — see the `[ram]` line above for why the label matters.
        "[cache] {regime}: ~{} GiB experts{scope} / {} GiB RAM ({covers_pct}% coverage), {} GiB \
         resident weights → fill to ~{} GiB, LRU-evict under pressure (hard floor {} GiB) — never OOM",
        total_expert_bytes >> 30,
        total >> 30,
        resident >> 30,
        fill_target >> 30,
        ADAPTIVE_HARD_FLOOR >> 30
    );
    // Preload needs a STRICTER test than residency-fill, and since `NEARFIT_COVERAGE_PCT`
    // dropped to 80 the two deliberately diverge.
    //
    // Residency-fill is worthwhile whenever most of the set can stay resident — MiniMax-M2.7
    // at 86% coverage holds ~103 of its 117 GB and measured 5.3 tok/s against 0.6 for the
    // streaming cap. Eager preload is a different promise: it loads EVERY expert up front, so
    // it is only safe when the whole set fits with room to spare. M2.7 fails that (117 GB of
    // experts against ~103 GB of headroom), and forcing it would fill the cache and then
    // evict from it during the preload itself.
    //
    // So: fill aggressively, preload only when the set genuinely fits. `near_fit` is retained
    // as a necessary condition — a ≫-RAM model must never attempt this.
    let fits_with_headroom = total_expert_bytes <= natural_fill / 100 * 85;
    near_fit && fits_with_headroom
}

/// Eagerly load EVERY routed expert into the cache at startup, so decode never pays a cold
/// miss. A nemotron decode touches ~all experts within a couple hundred tokens (diverse
/// routing, no hot set), so lazy loading otherwise spreads a ~59 GB cold read across the
/// generation and dominates it (measured: expert-load 26 s vs gpu-ffn 0.4 s over 256 tok);
/// preloading moves that to a one-time startup and lifts warm decode 1.7× (5.86→9.95 tok/s
/// on nemotron-3-super), token-identical. `max_residency` (from `wire_adaptive_cache`) gates
/// it to models that fit, so a ≫-RAM model never tries to preload its whole set.
/// `COLI_PRELOAD_EXPERTS` overrides: `1` forces on, `0` forces off.
fn preload_all_experts<P>(
    provider: &std::sync::Arc<colibri_engine::ExpertCache<P>>,
    cfg: &colibri_core::Config,
    max_residency: bool,
    owned: &[u32],
) where
    P: colibri_engine::ExpertProvider + Send + Sync + 'static,
{
    use colibri_engine::ExpertProvider as _;
    // `max_residency` means "the expert set fits", which is the safe auto-on condition.
    // DeepSeek-V4 does NOT fit — 137 GiB of experts against a ~90 GiB budget, 66% coverage
    // — and preloading it evicts as it goes, yet it is still a clear win. Measured on
    // gx10-42b2, page cache dropped between arms (without that, a preload run leaves ~98 GB
    // warm and the NEXT lazy run reads 4.66 — pure carry-over, which is how this nearly got
    // recorded as noise):
    //
    //     preload off   3.12 / 3.16 tok/s        preload on   4.08 / 4.12   -> 1.31x
    //
    // The hit rate gets WORSE (90% -> 73%) and 5306 evictions happen; it wins anyway,
    // because preload converts thousands of scattered decode-time reads into one sequential
    // 20 s bulk read. That mechanism does not depend on fitting.
    //
    // Enabled for V4 specifically rather than fleet-wide: the argument generalises but the
    // MEASUREMENT does not, and GLM/K3/M3 sit in different coverage regimes that were not
    // tested here. Widen it only with numbers for those.
    let want = match std::env::var("COLI_PRELOAD_EXPERTS").ok().as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => max_residency || cfg.arch == colibri_core::Arch::DeepseekV4,
    };
    if !want {
        return;
    }
    // Branch on whether a `Moe` entry EXISTS, not on whether `layer_kind` is populated —
    // same distinction as `Config::moe_layers`. Kimi-K3 populates `layer_kind` with mixer
    // kinds only (Kda/Attn) while every layer past `first_dense` still carries experts, so
    // the `is_empty()` form built an EMPTY preload list for it and would have preloaded
    // nothing had K3 ever reached the near-fit regime (a K3 slice can).
    let has_moe_kind = cfg
        .layer_kind
        .iter()
        .any(|k| matches!(k, colibri_core::LayerKind::Moe));
    let moe_layers: Vec<usize> = if has_moe_kind {
        cfg.layer_kind
            .iter()
            .enumerate()
            .filter(|(_, k)| matches!(k, colibri_core::LayerKind::Moe))
            .map(|(i, _)| i)
            .collect()
    } else {
        (cfg.first_dense as usize..cfg.n_layers as usize).collect()
    };
    // Only this node's own experts: the sharded provider refuses a peer's, so preloading
    // `0..n_experts` on a worker fails on the first unowned id and drops the WHOLE node back
    // to lazy loading.
    let all: Vec<usize> = if owned.is_empty() {
        (0..cfg.n_experts as usize).collect()
    } else {
        owned.iter().map(|&e| e as usize).collect()
    };
    let t = std::time::Instant::now();
    for &l in &moe_layers {
        if let Err(e) = provider.prefetch(l, &all) {
            eprintln!("[preload] layer {l} failed: {e} — falling back to lazy load");
            return;
        }
    }
    let s = provider.stats();
    eprintln!(
        "[preload] {} experts resident ({:.1} GB) across {} MoE layers in {:.1}s",
        s.resident,
        s.bytes as f64 / 1e9,
        moe_layers.len(),
        t.elapsed().as_secs_f64()
    );
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
    // is off. `Some(0)` wires the loader with a no-op predictor. Prefetch-ahead is ON
    // by default (measured prefill win, tokens identical; see cache::prefetch_ahead_enabled);
    // `COLI_PREFETCH_AHEAD=0` disables it. It self-gates to prefill via PREFETCH_AHEAD_MIN,
    // so wiring the loader here costs decode nothing but one idle background thread.
    let ahead = std::env::var("COLI_PREFETCH_AHEAD").ok().as_deref() != Some("0");
    match std::env::var("COLI_PREFETCH").ok().as_deref() {
        None | Some("") | Some("0") => ahead.then_some(0),
        Some("1") => Some(
            std::env::var("COLI_PREFETCH_N")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(16),
        ),
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
        let imbalance = if min > 0 {
            max as f64 / min as f64
        } else {
            f64::INFINITY
        };
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
        bytes_per_expert_of, context_in_kv_budget, experts_in_budget, kv_bytes_per_token,
        kv_fixed_bytes,
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

    // Both of these ask the config rather than assuming GLM's shape. `capacity` reported
    // Nemotron-H's routed experts as 695 GB inside a 69 GB container by getting all three
    // wrong at once: expert width (latent, not hidden), tensor count (gateless relu2 has no
    // gate), and layer count (`n_layers - first_dense` counts Nemotron's Mamba2 and GQA
    // layers as MoE — `moe_layers()` reads the layer_kind axis, and its own doc notes the
    // naive form "mis-sizes the expert cache by the entire model").
    let bpe = bytes_per_expert_of(&cfg, 4);
    let (sparse_layers, _first_moe) = cfg.moe_layers();
    let sparse_layers = sparse_layers as u64;
    let total_experts = sparse_layers * cfg.n_experts as u64;

    let ram_gb: u64 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .or_else(|| colibri_engine::available_ram_bytes().map(|b| b / gib))
        .unwrap_or(128);
    let ctx = args.get(4).and_then(|s| parse_ctx(s)).unwrap_or(0);

    // Fixed reserves (GLM-5.2 int8-resident estimates): resident dense ~10 GB,
    // working buffers / OS headroom ~4 GB.
    let dense_gb = 10u64;
    let working_gb = 4u64;
    let kv_per_tok = kv_bytes_per_token(&cfg);
    let kv_fixed = kv_fixed_bytes(&cfg); // per-sequence Mamba2 state; 0 unless hybrid
    let kv_bytes = kv_per_tok * ctx + kv_fixed;

    let ram_bytes = ram_gb * gib;
    let expert_budget = ram_bytes
        .saturating_sub((dense_gb + working_gb) * gib)
        .saturating_sub(kv_bytes);
    let per_node = experts_in_budget(expert_budget, bpe).min(total_experts);
    let pct = |n: u64| {
        if total_experts > 0 {
            100.0 * n as f64 / total_experts as f64
        } else {
            0.0
        }
    };

    println!(
        "model: hidden={} moe_inter={} experts/layer={} attn_layers={} sparse_layers={}",
        cfg.hidden, cfg.moe_inter, cfg.n_experts, cfg.n_layers, sparse_layers
    );
    println!(
        "per expert (nvfp4 ~4-bit): {:.2} MB   total routed: {total_experts} → {:.0} GB",
        mb(bpe),
        gb(total_experts * bpe)
    );
    // Report the layers that actually hold KV, not the total: a hybrid stack caches on
    // only its attention layers (Nemotron-H: 8 of 88; Kimi-K3: 24 of 93). Shared with the
    // reservation rather than reimplemented — this used to be a second copy of the
    // predicate, which is how the earlier KV accounting bugs got in.
    let kv_layers = colibri_engine::KvCache::kv_layers(&cfg);
    println!(
        "KV cache: {:.1} KB/token ({} of {} layers cache KV)",
        kv_per_tok as f64 / 1024.0,
        kv_layers,
        cfg.n_layers
    );
    if kv_fixed > 0 {
        println!(
            "  + {:.0} MB/sequence fixed recurrent state (O(1) in context, per concurrent \
             sequence)",
            mb(kv_fixed)
        );
    }
    println!(
        "  8 GB KV holds ~{} tokens ({}K)",
        context_in_kv_budget(8 * gib, kv_per_tok),
        context_in_kv_budget(8 * gib, kv_per_tok) / 1024
    );
    for &c in &[131072u64, 262144, 524288] {
        println!("  {}K context → {:.1} GB KV", c / 1024, gb(kv_per_tok * c));
    }
    println!();
    println!(
        "budget: {ram_gb} GB − {dense_gb} dense − {working_gb} working{} → {:.0} GB for experts",
        if ctx > 0 {
            format!(" − {:.0} KV({}K ctx)", gb(kv_bytes), ctx / 1024)
        } else {
            String::new()
        },
        gb(expert_budget)
    );
    println!(
        "==> resident experts per node: {per_node} ({:.0}% of all {total_experts})",
        pct(per_node)
    );
    if ctx > 0 {
        println!(
            "    (keeping {}K-token context in a {}-GB KV cache)",
            ctx / 1024,
            kv_bytes.div_ceil(gib)
        );
    }
    ExitCode::SUCCESS
}

/// Bits per weight actually stored in a loaded tensor, from its container format
/// (`fmt_code`: 0 = f32, 1 = int8, 3 = packed int2, 4 = e4m3, 5 = nvfp4).
///
/// `model.dbits`/`model.ebits` are the **LoadOptions** the caller asked for, which
/// only bite when quantizing a full-precision snapshot at load. For a pre-quantized
/// container the bits are fixed on disk and those fields are just the env defaults —
/// reporting them made `ppl` claim "4 / 4 bits" while measuring an 8-bit container.
fn tensor_bits(t: &colibri_core::QTensor) -> &'static str {
    match t.fmt_code {
        0 => "f32",
        1 => "int8",
        3 => "int2",
        4 => "e4m3",
        5 => "nvfp4",
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
/// the perplexity for ~9 GB of RAM. Routed experts ship NVFP4 (4-bit block-scaled);
/// the 8-bit e4m3 alternative doubles the bytes streamed per token and needs a
/// 0.74 TB container that does not fit on the box. Measure which knob actually buys
/// the quality before paying for it.
///
/// One forward over the whole sequence (prefill), then the mean negative
/// log-likelihood of each *actual* next token — not the argmax, which only says
/// whether the top pick matched and is blind to how much probability mass moved.
/// Honors `COLI_DBITS`/`COLI_EBITS`, which only bite on a full-precision snapshot: a
/// pre-quantized container is already fixed on disk and cannot be un-rounded, so
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
        eprintln!(
            "coli ppl: need >= 2 tokens (got {}), give it more text",
            ids.len()
        );
        return ExitCode::from(2);
    }

    let envbits = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let opts = colibri_engine::LoadOptions {
        dbits: envbits("COLI_DBITS", 8),
        ebits: envbits("COLI_EBITS", 8),
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
    // resident path; the streamed experts are NVFP4/e4m3, independent of it.
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
            eprintln!(
                "coli ppl: token id {target} out of range for vocab {}",
                lg.len()
            );
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
    println!(
        "top-1 match   : {:.1}%  ({top1}/{n})",
        top1 as f64 / n as f64 * 100.0
    );
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
    let envbits = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(d)
    };
    let opts = colibri_engine::LoadOptions {
        dbits: envbits("COLI_DBITS", 8),
        ebits: envbits("COLI_EBITS", 8),
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
        .map(|pos| {
            colibri_engine::argmax(&colibri_engine::logits(
                &model,
                &hidden[pos * d..(pos + 1) * d],
            )) as i32
        })
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
            println!(
                "hidden={}  layers={}  heads={}",
                c.hidden, c.n_layers, c.n_heads
            );
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

    /// No fill target, however derived, may exceed what is left after the model's own RAM.
    ///
    /// A regression test for a measured failure. The clamp used to be applied only on the
    /// automatic path, so the (now removed) `COLI_RAM_GB=110` on a 121 GiB Spark set a
    /// 110 GB fill target with nothing subtracted for the dense tier or the runtime. The
    /// serve process reached 108.7 GiB RSS, 3 GB went to swap, and serve throughput fell to
    /// 0.06-0.24 tok/s — worse than the 40 GB default it was meant to beat. The knob is
    /// gone; this pins the invariant that outlived it.
    /// The GPU's duplicate of the dense tier must come out of the fill, not just the ledger.
    ///
    /// GLM-5.2 shipped for one commit with `dense 34 GB` charged (17 host + 17 duplicate)
    /// and `fill to ~89 GB` planned in the same run — 123 GB of intent on a 121 GB box.
    /// With swap off that is not a slowdown: earlyoom SIGTERMed it 43 s in, and a 12-token
    /// smoke test had passed because the cache never grew far enough to reach the target.
    #[test]
    fn fill_target_subtracts_the_device_weight_duplicate() {
        let gb = |n: u64| n << 30;
        let total = gb(121);
        let resident = gb(17); // GLM-5.2's host dense tier

        // Zero-copy: only the host copy is charged.
        assert_eq!(fill_within_headroom(u64::MAX, resident, total) >> 30, 89);

        // Upload: the duplicate is real RAM on GB10 and must come out of the fill too.
        let uploaded = fill_within_headroom(u64::MAX, resident * 2, total);
        assert_eq!(uploaded >> 30, 72, "116 - (17 + 17) dense - 10 runtime");
        assert!(
            uploaded + resident * 2 + RUNTIME_RESERVE <= ram_target(total),
            "fill + both weight copies + reserve must fit the 96% ceiling"
        );
    }

    /// The guard's floor must clear whatever will actually SIGTERM us, on every host —
    /// not just the one whose earlyoom someone tuned down to 2%.
    ///
    /// This cannot be verified end-to-end on gx10-42b2 without rewriting its
    /// /etc/default/earlyoom (root, and the operator's call), so the arithmetic is the
    /// deliverable. Each row is a real configuration.
    #[test]
    fn oom_guard_floor_clears_earlyoom_on_every_host() {
        let gb = |n: f64| (n * 1e9) as u64;
        let cases = [
            // (MemTotal, earlyoom -m pct, label)
            (gb(130.66), 2u64, "gx10-42b2 as tuned by sparkrun"),
            (gb(130.66), 10, "same box with STOCK earlyoom"),
            (gb(256.0), 2, "a 256 GB host at -m 2"),
            (gb(512.0), 10, "a large host, stock"),
        ];
        for (total, pct, label) in cases {
            let trigger = total / 100 * pct;
            let floor = oom_guard_floor(total, Some(pct));
            assert!(
                floor > trigger,
                "{label}: floor {floor} must sit ABOVE earlyoom's {trigger}, or the guard \
                 never fires before the kill"
            );
            assert!(
                floor >= ADAPTIVE_HARD_FLOOR,
                "{label}: never drop below the built-in minimum"
            );
        }

        // The old absolute constant fails exactly where this function was written to help:
        // stock earlyoom on this very box would SIGTERM at 13 GB with the guard idle at 3.
        let stock_trigger = gb(130.66) / 100 * 10;
        assert!(
            ADAPTIVE_HARD_FLOOR < stock_trigger,
            "the bug this replaces: a 3 GiB absolute floor is below a stock 10% kill line"
        );

        // No earlyoom => the kernel OOM killer is the backstop; keep the static floor
        // rather than inventing headroom nobody needs.
        assert_eq!(oom_guard_floor(gb(130.66), None), ADAPTIVE_HARD_FLOOR);
    }

    #[test]
    fn fill_target_cannot_exceed_non_expert_headroom() {
        let gb = |n: u64| n << 30;
        let total = gb(121); // the Spark
        let resident = gb(3); // M2.7's dense tier

        // The exact case that swapped. 96% of 121 = 116; 116 - 3 dense - 10 runtime = 103.
        let clamped = clamp_fill_to_headroom(gb(110), resident, total);
        assert!(
            clamped <= ram_target(total) - resident - RUNTIME_RESERVE,
            "clamped fill {} GB still exceeds headroom",
            clamped >> 30
        );
        assert_eq!(
            clamped,
            ram_target(total) - resident - RUNTIME_RESERVE,
            "the clamp must BE the headroom, not merely respect it"
        );
        assert_eq!(
            clamped >> 30,
            103,
            "96% of 121 GiB, less 3 dense and 10 runtime"
        );

        // The whole point of the target: nearly all of RAM ends up holding model, not
        // held back "just in case". Experts + dense must reach at least 85% of MemTotal.
        let footprint = clamped + resident;
        assert!(
            footprint * 100 / total >= 85,
            "only {}% of RAM would hold model — the reserve is too conservative",
            footprint * 100 / total
        );
        // ...but never all of it: the kernel needs slab, page tables (~200 MB of PTEs for
        // a 100 GB mapping alone) and block/network buffers. Taking those is what turns a
        // fill into paging.
        assert!(
            footprint <= ram_target(total),
            "footprint exceeds the {TARGET_RAM_PCT}% target"
        );

        // A request that already fits is passed through untouched — the clamp is a
        // ceiling, not a rewrite.
        assert_eq!(clamp_fill_to_headroom(gb(40), resident, total), gb(40));

        // A big dense tier eats the headroom: Kimi-K3 keeps ~63 GB resident, so the same
        // request must come down much further. Without this the cache overcommits and the
        // box OOM-kills the process mid-forward.
        assert_eq!(clamp_fill_to_headroom(gb(110), gb(63), total) >> 30, 43);

        // Degenerate: dense tier alone exceeds RAM. Must saturate to 0, not underflow.
        assert_eq!(clamp_fill_to_headroom(gb(110), gb(200), total), 0);
    }

    /// The narrow-reader gate, pinned to the fleet A/B that produced it (34 vs 8 threads,
    /// decode, interleaved, token-identical). This one is worth pinning hard: a constant
    /// thread count of 8 is a 1.15-1.17x WIN on the two low-coverage models and a 2-4x
    /// REGRESSION on the two high-coverage ones, so getting the threshold wrong is not a
    /// small miss in either direction.
    #[test]
    fn narrow_reader_gate_matches_the_measured_models() {
        for (name, cov, want_narrow) in [
            ("kimi-k3", 7u64, true),          // 26364 -> 22531 ms at 8 threads
            ("glm-5.2", 26, true),            // 12017 -> 10402 ms at 8 threads
            ("minimax-m3", 46, false),        // 3718 -> 8611..16092 ms at 8 threads: 2-4x WORSE
            ("minimax-m2.7", 86, false),      // 481 -> 485 ms: neutral, all mmap views
            ("nemotron-3-super", 172, false), // preload 12.0 s -> 29.5 s at 8 threads
        ] {
            assert_eq!(
                cov < DISK_BOUND_COVERAGE_PCT,
                want_narrow,
                "{name} at {cov}% coverage should have narrow-reader={want_narrow}"
            );
        }
        // Bracketed by the measured neighbours (GLM 26 wants narrow, M3 46 wants wide) so
        // the threshold cannot be tuned onto a known result.
        assert!(
            (27..=46).contains(&DISK_BOUND_COVERAGE_PCT),
            "narrow-reader threshold {DISK_BOUND_COVERAGE_PCT} escaped the measured interval (26%, 46%)"
        );
        // Inside the flat part of the GLM thread curve: 8/12/16 gave 11.40/11.42/11.32
        // GB/s, while 2 and 34 collapse to 5.82 and 9.87.
        assert!(
            (8..=16).contains(&DISK_BOUND_READ_THREADS),
            "{DISK_BOUND_READ_THREADS} threads is outside the measured 8-16 plateau"
        );
    }

    /// One constant now drives BOTH the read path (O_DIRECT) and the mapped-view path,
    /// in opposite directions, so a change to it moves two behaviours at once. Both are
    /// measured, so pin both.
    ///
    /// mmap, ABBA, 4 runs/arm, tokens identical throughout:
    ///   MiniMax-M2.7 (86%): pread 2987 ms -> mapped 529 ms   = 5.65x WIN
    ///   GLM-5.2      (26%): pread 14694 ms -> mapped 15080 ms = 0.97x loss
    #[test]
    fn mmap_gate_matches_the_measured_models() {
        // Every measured mmap arm. M3 is the one that matters: it sits ABOVE the O_DIRECT
        // threshold, so gating mmap on that constant (as this first did) turns mapping on
        // for a model where it measured 9.6% SLOWER.
        // **M2.7 flipped to `false` on 2026-07-31.** The 5.65x above is superseded: it was
        // measured when the READER was the mapping's only consumer. The grouped expert path
        // added a second one that memcpys every staged expert on the CPU, so a non-resident
        // page became a major fault *inside the copy*. Re-measured at 84% coverage, 2 reps
        // ABBA, token-identical: mapped off 39-43 s wall / pack 1.96-2.13 s / 0 major
        // faults, mapped on 49-56 s / pack 7.79-15.56 s / 159k-311k faults. `expert-load`,
        // the tier this gate exists to help, does not gain (12.1 vs 13.5 s) while `moe`
        // loses 1.52x.
        for (name, cov, want_map) in [
            ("minimax-m2.7", 86u64, false), // was 5.65x; now 1.27x SLOWER, see above
            ("minimax-m3", 47, false),      // 7851 -> 8605 ms, 0.91x
            ("glm-5.2", 26, false),         // 14694 -> 15080 ms, 0.97x
            ("kimi-k3", 7, false),
            ("nemotron-3-super", 155, true), // fits outright; preloads, so ~2 ms either way
        ] {
            assert_eq!(
                cov >= MMAP_MIN_COVERAGE_PCT,
                want_map,
                "{name} at {cov}% coverage should have mmap={want_map}"
            );
        }
        // The threshold is no longer a tuned proxy, so it is no longer bracketed by a
        // measured interval. It is the CONDITION mapping requires — that a span is resident
        // when touched — and 100% is the only value that states it. Below 100 some fraction
        // of touches is a synchronous disk read, which is exactly what 84% cost M2.7.
        assert_eq!(
            MMAP_MIN_COVERAGE_PCT, 100,
            "mapping pays only when the set is fully resident; anything less admits faults"
        );
        // The two paths are MUTUALLY EXCLUSIVE, and not merely by taste: O_DIRECT never
        // populates the page cache, so with both on the residency gate can never succeed
        // and every span pays a `mincore` before falling back to `pread` anyway. Measured
        // on Kimi-K3 (7% coverage, so O_DIRECT is on) with mapping forced: 28604 -> 29304 ms
        // steady state (2.4%, matching GLM's 2.6% at 26%), plus a ONE-OFF 65191 ms first
        // run while 94 shards / 1.4 TB of mappings are established and first-touched.
        // Ordering the thresholds prevents the engine from ever selecting the combination.
        assert!(
            MMAP_MIN_COVERAGE_PCT > O_DIRECT_MAX_COVERAGE_PCT,
            "thresholds must not overlap: O_DIRECT + mmap is all cost and no benefit"
        );
        for cov in [0u64, 7, 26, 34, 35, 47, 79, 80, 86, 172, 1000] {
            let direct = cov < O_DIRECT_MAX_COVERAGE_PCT;
            let mapped = cov >= MMAP_MIN_COVERAGE_PCT;
            assert!(!(direct && mapped), "at {cov}% both paths would be active");
        }
    }

    /// The O_DIRECT threshold must keep classifying the four models it was measured on.
    /// Each row is a real ABBA-mirrored result, so a threshold change that flips any of
    /// them is a change against evidence and should fail here first.
    #[test]
    fn o_direct_threshold_matches_the_measured_models() {
        // (name, coverage %, O_DIRECT should be ON)
        let measured = [
            ("kimi-k3", 7u64, true),     // 31350 -> 28788 ms, 1.089x
            ("glm-5.2", 27, true),       // 16957 -> 14812 ms, 1.145x
            ("minimax-m3", 47, false),   // buffered 6232 vs 6755
            ("minimax-m2.7", 86, false), // buffered 4166 vs 4444; warm rep read 0 bytes
        ];
        for (name, cov, want_direct) in measured {
            assert_eq!(
                cov < O_DIRECT_MAX_COVERAGE_PCT,
                want_direct,
                "{name} at {cov}% coverage should have O_DIRECT={want_direct}"
            );
        }
        // The threshold must sit strictly between the two measured neighbours, so it
        // cannot be "tuned" past a model whose result we actually have.
        assert!(
            (28..=46).contains(&O_DIRECT_MAX_COVERAGE_PCT),
            "threshold {O_DIRECT_MAX_COVERAGE_PCT} escaped the measured interval (27%, 47%)"
        );
        // Nemotron preloads (172% coverage) and must never take the direct path.
        assert!(!(172 < O_DIRECT_MAX_COVERAGE_PCT));
    }

    /// The cache-sizing footprint must not collapse to zero on an arch whose
    /// `layer_kind` carries mixer kinds only.
    ///
    /// Kimi-K3 populates `layer_kind` with Kda/Attn and never `Moe`, because every K3
    /// layer has BOTH a mixer and an FFN. Counting `Moe` entries there gives 0 MoE
    /// layers, so the footprint came out as 0 bytes and the `[cache]` line reported
    /// "~0 GB experts / 0% coverage" for a model with 1.4 TB of experts — which is
    /// exactly what an upstream merge reintroduced here after it had been fixed once.
    /// It is silent: nothing panics, the run completes, and only the coverage-derived
    /// decisions are wrong.
    #[test]
    fn expert_footprint_survives_a_mixer_only_layer_kind() {
        // 93 layers, first_k_dense_replace = 1 => 92 MoE layers, 896 experts.
        let json = colibri_json::Json::parse(
            r#"{"model_type":"kimi_k3","architectures":["KimiK3ForConditionalGeneration"],
                "text_config":{"hidden_size":7168,"num_hidden_layers":93,
                "num_attention_heads":96,"num_key_value_heads":96,"num_experts":896,
                "num_experts_per_token":16,"num_shared_experts":2,
                "moe_intermediate_size":3072,"intermediate_size":6144,
                "routed_expert_hidden_size":3584,"first_k_dense_replace":1,
                "q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_head_dim":128,
                "qk_rope_head_dim":64,"v_head_dim":128,"vocab_size":163840,
                "max_position_embeddings":16384,"rms_norm_eps":1e-5,"rope_theta":50000.0,
                "moe_renormalize":true,"moe_router_activation_func":"sigmoid",
                "num_expert_group":1,"topk_group":1,"routed_scaling_factor":1.0,
                "eos_token_id":5,"mla_use_nope":true,"hidden_act":"situ",
                "activation_situ_beta":4.0,"activation_situ_linear_beta":25.0,
                "attn_res_block_size":12,
                "linear_attn_config":{"head_dim":128,"num_heads":64,
                  "short_conv_kernel_size":4,"full_attn_layers":[4,8,93]}}}"#,
        )
        .unwrap();
        let cfg = colibri_core::Config::from_json(&json).expect("kimi_k3 parse");

        // Precondition: populated, but carrying no `Moe` — the shape that breaks the
        // naive predicate.
        assert!(!cfg.layer_kind.is_empty());
        assert_eq!(
            cfg.layer_kind
                .iter()
                .filter(|k| **k == colibri_core::LayerKind::Moe)
                .count(),
            0
        );

        const PER: u64 = 17_547_264; // one real K3 expert, MXFP4
        let got = expert_footprint(&cfg, PER, cfg.n_experts as u64);
        assert_eq!(got, PER * 896 * 92, "must charge all 92 MoE layers");
        assert!(
            got > 1_400_000_000_000,
            "K3's expert set is ~1.4 TB, got {got}"
        );

        // A sharded node charges only what it owns, so coverage reflects the shard.
        assert_eq!(expert_footprint(&cfg, PER, 448), PER * 448 * 92);
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
            assert!(
                (logprob_of(&a, t) - logprob_of(&b, t)).abs() < 1e-4,
                "not shift-invariant"
            );
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
        let no_cap = SPARK_AVAIL
            .saturating_sub(1 * GB)
            .saturating_sub(WORKING_RESERVE)
            / GB;
        assert!(
            no_cap > 70,
            "premise: reserves alone pick {no_cap} GB, past the cliff"
        );
        let with_cap = budget_from(SPARK_AVAIL, 1 * GB, Some(SPARK_TOTAL)) / GB;
        assert!(
            with_cap < 70,
            "cap failed to pull {with_cap} GB back below the cliff"
        );
    }

    #[test]
    fn budget_never_underflows_into_an_unbounded_cache() {
        // The bug this guards: `avail - reserve - WORKING_RESERVE` on a small box
        // wraps to ~16 EiB — an effectively unlimited budget, i.e. exactly the OOM
        // the reserve exists to prevent.
        for (avail, reserve) in [(0, 0), (1 * GB, 0), (8 * GB, 64 * GB), (0, u64::MAX)] {
            let b = budget_from(avail, reserve, Some(SPARK_TOTAL));
            assert!(
                b <= MIN_BUDGET,
                "underflowed to {b} bytes for avail={avail} reserve={reserve}"
            );
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
            assert_eq!(
                b,
                total_gb / CACHE_CAP_DIVISOR,
                "cap did not track a {total_gb} GB box"
            );
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
        assert_eq!(
            missing_peer_ranks(4, NodeId(2), &HashMap::new()),
            vec![0, 1, 3]
        );
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
