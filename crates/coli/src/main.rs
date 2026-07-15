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
