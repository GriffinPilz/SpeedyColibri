# Contributing

SpeedyColibri is a Rust rewrite of [JustVugg/colibri](https://github.com/JustVugg/colibri).
Keep changes focused, and keep the Rust port **token-exact** with the C engine —
`c/` is retained as the reference oracle for exactly that reason.

## Workflow

The Rust work lives on `main`. Keep it building clean (`cargo build --workspace`
with zero warnings) and green (`cargo test --workspace`).

## Local checks

```sh
cargo build --workspace          # GPU build: -p coli --features cuda
cargo test  --workspace          # unit + integration tests
```

Correctness against the original C engine (the oracle) — see
[VALIDATION.md](VALIDATION.md):

```sh
python3 scripts/validate_c_vs_rust.py     # builds `make -C c glm`, diffs f32 + int4
```

The CUDA backend must additionally be checked on a CUDA-capable host (the DGX
Spark); the build compiles the `c/backend_cuda.cu` kernels via `nvcc`:

```sh
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121 cargo test -p colibri-backend --features cuda
```

## Reports

Benchmark reports should include the commit, exact commands, hardware and storage
details, warm-up policy, run count, and median throughput. Machine load is noisy —
prefer same-run relative numbers.
