//! `coli` — colibrì command-line entry point.
//!
//! Port target: the `main()` dispatch in `c/glm.c` plus the `c/coli` launcher.
//! The subcommands that depend on the not-yet-ported forward pass print an
//! honest "pending" message; `tokenize` and `config` already work end-to-end
//! against the ported crates.

use std::process::ExitCode;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn usage() {
    eprintln!(
        r#"colibrì (SpeedyColibri, Rust port) v{VERSION}
  tiny engine, immense model

USAGE:
  coli <command> [args]

COMMANDS:
  chat <snap>              interactive chat            [pending: forward pass]
  web <snap>               web dashboard               [pending: forward pass]
  serve <snap>             OpenAI-compatible server    [pending: forward pass]
  bench <snap>             throughput benchmark        [pending: forward pass]
  convert ...              FP8 -> int4 converter       [pending: tools port]
  tokenize <tok.json> <text>   encode/decode round-trip   [working]
  config <snap>            print parsed hyperparameters   [working]
  load <snap>              load dense weights, print structure  [working]
  gen <snap> [ids...]      greedy-generate from token ids       [working]
  tf <snap> <ids...>       teacher-forcing argmax per position  [working]
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
            println!("coli {VERSION}");
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
        "capacity" => cmd_capacity(&args),
        "loadbench" => cmd_loadbench(&args),
        "repack" => cmd_repack(&args),
        "backend" => cmd_backend(),
        "chat" | "web" | "serve" | "bench" | "convert" => {
            eprintln!(
                "coli {cmd}: not yet ported — the CPU forward pass is still being \
                 converted from c/glm.c. See PORTING.md for status."
            );
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
    let n_new = envbits("COLI_NGEN", 16) as usize;

    // COLI_PRELOAD: parallel-load experts into RAM, then serve with no per-token
    // disk I/O. If the value is a dir with a manifest.json it uses repacked
    // shards; otherwise (e.g. `COLI_PRELOAD=1`) it reads directly from the
    // original model in parallel — no repack, no second copy on disk.
    if let Ok(v) = std::env::var("COLI_PRELOAD") {
        let repacked = std::path::Path::new(&v).join("manifest.json").exists();
        if repacked {
            return cmd_gen_preload_repacked(&model, &v, &prompt, n_new);
        }
        return cmd_gen_preload_direct(&model, &prompt, n_new);
    }

    // Resident expert cache: experts loaded once stay in RAM until the budget
    // forces an eviction (see `coli capacity`). Budget from COLI_RAM_GB, else
    // available RAM, else unbounded.
    let base =
        colibri_engine::ShardsExpertProvider::new(&model.shards, &model.cfg, model.ebits as u32);
    let budget = ram_budget();
    let provider = colibri_engine::ExpertCache::new(base, budget);

    // Pinned hot-store warm-up (AUTOPIN): read the persistent usage history and
    // pin the hottest experts (COLI_PIN_GB budget) so they stay resident.
    let usage_path =
        std::env::var("COLI_USAGE").unwrap_or_else(|_| format!("{snap}/.coli_usage"));
    let mut history = colibri_engine::UsageHistory::load(&usage_path).unwrap_or_default();
    let pin_gb: u64 = std::env::var("COLI_PIN_GB").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    if pin_gb > 0 && !history.is_empty() {
        match provider.warm_pin(&history, pin_gb << 30) {
            Ok(n) => println!(
                "hot-store: pinned {n} experts from usage history ({} entries, {} selections)",
                history.len(),
                history.total()
            ),
            Err(e) => eprintln!("coli gen: warm_pin: {e}"),
        }
    }

    let mut kv = colibri_engine::KvCache::new(
        model.cfg.n_layers as usize,
        model.cfg.kv_lora as usize,
        model.cfg.qk_rope as usize,
        prompt.len() + n_new,
    );
    match colibri_engine::generate_greedy(&model, &mut kv, &provider, &prompt, n_new) {
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
    let mut kv = colibri_engine::KvCache::new(
        model.cfg.n_layers as usize,
        model.cfg.kv_lora as usize,
        model.cfg.qk_rope as usize,
        prompt.len() + n_new,
    );
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

/// Expert-cache byte budget: `COLI_RAM_GB` if set, else available RAM (Linux),
/// else unbounded.
fn ram_budget() -> u64 {
    if let Ok(gb) = std::env::var("COLI_RAM_GB") {
        if let Ok(g) = gb.parse::<u64>() {
            return g << 30;
        }
    }
    colibri_engine::available_ram_bytes().unwrap_or(u64::MAX)
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
    let mut kv = colibri_engine::KvCache::new(
        model.cfg.n_layers as usize,
        model.cfg.kv_lora as usize,
        model.cfg.qk_rope as usize,
        ids.len(),
    );
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
