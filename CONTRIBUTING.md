# Contributing

SpeedyColibri is a Rust rewrite of [JustVugg/colibri](https://github.com/JustVugg/colibri).
Keep changes focused. The Rust port is now the engine (the original C reference has
been removed); correctness is held by the per-crate unit and integration tests, which
encode the reference behavior.

## Workflow

The Rust work lives on `main`. Keep it building clean (`cargo build --workspace`
with zero warnings) and green (`cargo test --workspace`).

## Local checks

```sh
cargo build --workspace          # GPU build: -p coli --features cuda
cargo test  --workspace          # unit + integration tests
```

The CUDA backend must additionally be checked on a CUDA-capable host (the DGX
Spark); the build compiles the `crates/colibri-backend/cuda/backend_cuda.cu` kernels
via `nvcc`:

```sh
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121 cargo test -p colibri-backend --features cuda
```

## Reports

Benchmark reports should include the commit, exact commands, hardware and storage
details, warm-up policy, run count, and median throughput. Machine load is noisy —
prefer same-run relative numbers.
