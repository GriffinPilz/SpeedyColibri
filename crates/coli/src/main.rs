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
    // Resident expert cache: experts loaded once stay in RAM until the budget
    // forces an eviction (see `coli capacity`). Budget from COLI_RAM_GB, else
    // available RAM, else unbounded.
    let base =
        colibri_engine::ShardsExpertProvider::new(&model.shards, &model.cfg, model.ebits as u32);
    let budget = ram_budget();
    let provider = colibri_engine::ExpertCache::new(base, budget);

    let n_new = envbits("COLI_NGEN", 16) as usize;
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
                "expert cache: {} resident ({:.1} MB), {} hits / {} misses, {} evictions",
                s.resident,
                s.bytes as f64 / (1024.0 * 1024.0),
                s.hits,
                s.misses,
                s.evictions
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("coli gen: {e}");
            ExitCode::FAILURE
        }
    }
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
