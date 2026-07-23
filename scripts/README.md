# Harness — model-parameterized build / convert / benchmark

Every script here takes a **model name** from the registry instead of a hardcoded
path or prompt, so the same tooling covers every model and adding one is a data edit,
not a rewrite of five scripts.

## Registry — `models.toml`

The single source of truth. One `[<name>]` block per model:

```toml
models_root = "/home/dgx1/models"        # per-host; override with $COLI_MODELS_ROOT

[minimax-m3]
container = "MiniMax-M3-container"        # joined under models_root unless absolute
source    = "MiniMax-M3-NVFP4-src"        # HF checkpoint for `convert`
arch      = "minimax-m3"
prompt    = "100..611"                    # canonical bench prompt (token-id range or list)
notes     = "…"
[minimax-m3.convert_env]                  # COLI_* flags coli convert needs (optional)
```

`model.py` resolves it for both bash (`eval "$(model.py env <name>)"`) and Python
(`from model import resolve`). Path-joining and prompt-range expansion live there and
nowhere else. Quick checks: `model.py list`, `model.py path <name>`.

## Scripts

| script | what |
|---|---|
| `build.sh` | Build+verify the CUDA `coli` binary (guards the silent CPU-only-binary trap). Model-agnostic. |
| `convert.sh <model>` | `coli convert` source → container using the model's registry paths + `convert_env`. |
| `bench.sh <model> [suite]` | Benchmark battery. Suites: `prefill`, `decode`, `batch`, `serve`, `all` (default). |
| `pipeline.sh <model> [phases]` | One entry point: `convert,build,bench` (default `build,bench`). |
| `bench_serve.py` / `paired_compare.py` | Real-HTTP throughput over diverse prompts + paired stats (used by the `serve` suite). |

Every gen-based suite runs a **token-identity gate**: repeated runs (and the batched
path, via `COLI_BATCH_VERIFY`) must emit identical tokens, so a faster number that
changed the output fails instead of being reported as a win. Suites report a **median**
over `BENCH_REPS` (default 3) because cold expert-load cache variance is ~±40% run to run.

Useful knobs: `COLI_BIN`, `COLI_MODELS_ROOT`, `BENCH_REPS`, `DECODE_NGEN`,
`BATCH_SIZES`, `BATCH_NGEN`, `SERVE_PORT`, `SERVE_NTOK`.

### Examples
```bash
scripts/build.sh                          # build the binary
scripts/bench.sh minimax-m3 prefill       # just the prefill profile
scripts/bench.sh minimax-m3               # full battery (prefill+decode+batch)
BENCH_SUITE=prefill scripts/pipeline.sh minimax-m3   # build then prefill-bench
scripts/pipeline.sh glm-5.2 convert,build,bench       # from HF checkpoint to numbers
```

## Adding a model — the whole checklist

1. **`crates/colibri-core/src/config.rs`** — add an `Arch` variant and detect it from
   `config.json` (`model_type` / `architectures`).
2. **`crates/colibri-engine/src/convert.rs`** — the per-arch tensor mapping
   (names, quant formats) so `coli convert` produces a container.
3. **`crates/colibri-engine`** forward path — wire the arch's attention/router/activation
   (most infra — NVFP4 experts, expert streaming, prefetch-ahead — is arch-agnostic and
   comes for free).
4. **`scripts/models.toml`** — one `[<name>]` block.

Then `scripts/pipeline.sh <name> convert,build,bench` runs end to end, and every
measurement script already understands `<name>`.
