# Harness — model-parameterized build / convert / benchmark / serve

Every script takes a **model name** from the registry instead of a hardcoded path or
prompt. Switching models is therefore just changing the name you pass — the same
tooling covers every model, and adding one is a data edit, not a rewrite of six scripts.

## Registry — `models.toml`

The single source of truth. One `[<name>]` block per model:

```toml
models_root = "/home/dgx1/models"        # per-host; override with $COLI_MODELS_ROOT

[minimax-m3]
container = "MiniMax-M3-container"        # joined under models_root unless absolute
source    = "MiniMax-M3-NVFP4-src"        # HF checkpoint for `convert`
arch      = "minimax-m3"
prompt    = "100..611"                    # canonical bench prompt (token-id range or list)
hf_repo   = "Org/Repo"                    # optional: HF container to re-materialize from
notes     = "…"
[minimax-m3.convert_env]                  # COLI_* flags coli convert needs (optional)
```

`model.py` resolves it for both bash (`eval "$(model.py env <name>)"`) and Python
(`from model import resolve`). Path-joining and prompt-range expansion live there and
nowhere else. Quick checks: `model.py list`, `model.py path <name>`.
(Quote dotted names in TOML — `["glm-5.2"]`, not `[glm-5.2]`.)

## Scripts

| script | what |
|---|---|
| `build.sh` | Build+verify the CUDA `coli` binary (guards the silent CPU-only-binary trap). Model-agnostic. |
| `convert.sh <model>` | `coli convert` source → container using the model's registry paths + `convert_env`. |
| `bench.sh <model> [suite]` | Benchmark battery. Suites: `prefill`, `decode`, `batch`, `serve`, `all` (default). |
| `serve.sh <model> [port]` | Serve a model over HTTP (container from the registry). Foreground; `SERVE_DETACH=1` to background. |
| `pipeline.sh <model> [phases]` | One entry point: `convert,build,bench` (default `build,bench`). |
| `bench_serve.py` / `paired_compare.py` | Real-HTTP throughput over diverse prompts + paired stats (used by the `serve` suite). |

Every gen-based suite runs a **token-identity gate**: repeated runs (and the batched
path, via `COLI_BATCH_VERIFY`) must emit identical tokens, so a faster number that
changed the output fails instead of being reported as a win. Suites report a **median**
over `BENCH_REPS` (default 3) because cold expert-load cache variance is ~±40% run to run.

Useful knobs: `COLI_BIN`, `COLI_MODELS_ROOT`, `BENCH_REPS`, `DECODE_NGEN`,
`BATCH_SIZES`, `BATCH_NGEN`, `SERVE_PORT`, `SERVE_DETACH`.

## Switching models

The model name is the **first argument to every action script** — switching is nothing
more than changing it. Nothing else (no paths, no prompts, no env) has to change.

```bash
scripts/model.py list                 # what's registered
scripts/bench.sh minimax-m3 prefill   # bench one model…
scripts/bench.sh glm-5.2   prefill    # …same command, different model
scripts/pipeline.sh glm-5.2 convert,build,bench   # convert→build→bench a different model end to end
```

`build.sh` is the one model-agnostic script — the `coli` binary is shared, so you build
**once** and then switch freely between models with `bench.sh`/`serve.sh`.

### Serve a model and query it from anywhere
`serve.sh` resolves the container from the registry, so you never type a container path:

```bash
# on the box — start a server for whichever model you want (SERVE_DETACH=1 → background + wait):
SERVE_DETACH=1 scripts/serve.sh minimax-m3 8080
```

`coli serve` binds `0.0.0.0` **after** the model finishes loading, so a live listener
means it's ready. Then query it from your Mac (or anywhere on the tailnet) — no tunnel,
the `model` field is just a label:

```bash
curl -s http://gx10-42b2:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":32}'
```

Switch the served model by pointing a server at a different name (use a **different port**
to run two at once), and stop one with `pkill -f 'coli serve'`:

```bash
SERVE_DETACH=1 scripts/serve.sh glm-5.2 8081     # second model, second port
```

Decode is ~1.4 tok/s, so size `max_tokens` accordingly (64 ≈ 45 s).

### Bringing a model online before you can switch to it
A registry entry only names a model; the container has to exist on the host. If
`bench.sh`/`serve.sh` says *"container missing"*, materialize it first:

```bash
# from its HF container (if the entry has hf_repo):
huggingface-cli download "$(scripts/model.py get glm-5.2 hf_repo)" \
  --local-dir "$(scripts/model.py path glm-5.2)"

# …or convert it from the source checkpoint:
scripts/convert.sh glm-5.2
```

Point the whole registry at a different host/disk with `COLI_MODELS_ROOT=/other/path`;
point a single model elsewhere by giving its `container`/`source` an absolute path.

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
script above already understands `<name>`.
