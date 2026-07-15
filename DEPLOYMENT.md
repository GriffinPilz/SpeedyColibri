# Deploying SpeedyColibri on DGX Spark

Target: a Docker container on **NVIDIA DGX Spark** (GB10 Grace Blackwell) —
aarch64 Grace CPU + Blackwell GPU, 128 GB unified LPDDR5X per node, CUDA. The
engine streams the 744B model's experts from disk, so a single Spark can serve
it; multiple Sparks split the experts across nodes.

## Current status

| Piece | Status |
|---|---|
| Container image (aarch64 + CUDA 13 base) | ✅ CUDA backend compiled in by default (`sm_121`) |
| Single-node run | ✅ full pipeline in-container: zero-copy GPU experts, flash-attention decode, pooled streaming loads — ~1.2 tok/s steady disk-streaming on one Spark |
| Model provisioning | ✅ entrypoint resolves `/model` mount → HF cache → `hf download` (token as first arg or `HF_TOKEN`) |
| GPU passthrough without root | ✅ `docker/run-dgx.sh` (manual device+driver-lib binds when CDI/nvidia-runtime absent) |
| CUDA (Blackwell) backend | ✅ GB10-validated, token-exact vs CPU |
| Expert-parallel sharding math | ✅ `colibri-cluster` (tested) |
| Multi-node RDMA transport | ⬜ designed, not wired (next up) |

The **compute backend priority is CUDA → CPU**. Apple-Silicon Metal is off the
critical path (kept only as an optional stub). The CPU fallback targets the Grace
ARM cores (aarch64 NEON is the CPU SIMD target).

## Quick start (DGX Spark)

One command; the launcher builds the image on first use, wires up the GPU, and
the container fetches the model if the host's HF cache doesn't have it:

```bash
# token from the environment...
HF_TOKEN=hf_abc123 docker/run-dgx.sh gen 100 200 300
# ...or as the first argument
docker/run-dgx.sh hf_abc123 gen 100 200 300
# already downloaded (or public repo)? no token needed:
docker/run-dgx.sh gen 100 200 300
```

The model resolves in this order: a snapshot mounted at `/model`
(`COLI_MODEL_DIR=<dir> docker/run-dgx.sh ...`), the HF cache (the launcher
mounts the host's `~/.cache/huggingface`, so the 358 GB download happens at most
once and is shared with non-container runs), else `hf download
$COLI_MODEL_REPO` (default `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`).

`run-dgx.sh` picks the GPU passthrough that the host supports: CDI
(`--device nvidia.com/gpu=all`), the nvidia runtime (`--gpus all`), or — on a
stock shared DGX where neither is configured and you have no root — it binds
`/dev/nvidia*` and the driver's user-space libraries directly (the entrypoint
re-runs `ldconfig` so `libcuda.so.1` resolves). Tuning env vars (`COLI_RAM_GB`,
`COLI_PROFILE`, ...) pass through automatically.

## Build the image

The workspace has **no crates.io dependencies** (std + path crates only), so the
build needs no registry access. Build on an aarch64 host (or with buildx for
`linux/arm64`):

```bash
docker build -f docker/Dockerfile -t speedycolibri:latest .
# CPU-only image: --build-arg CUDA_FEATURE=0
# cross-build for DGX Spark from another arch:
# docker buildx build --platform linux/arm64 -f docker/Dockerfile -t speedycolibri:latest .
```

`CUDA_TAG` (default `13.0.1`) is an `ARG` — match it to your Spark's driver
(driver 580+ for CUDA 13). The CUDA backend builds in by default with
`CUDA_ARCH=sm_121` (GB10); `native` doesn't work inside `docker build` (no GPU
to probe).

## Run by hand (without run-dgx.sh)

```bash
docker run --rm -it --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  speedycolibri:latest gen 100 200 300
```

or with compose (requires CDI or the nvidia runtime on the host):

```bash
HF_TOKEN=hf_... docker compose -f docker/docker-compose.yml run --rm coli gen 100 200 300
```

### Environment variables

| Var | Meaning | Default |
|---|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Hugging Face token for the model download (or pass `hf_...` as the first argument) | unset — fine if the model is cached/public |
| `COLI_MODEL_REPO` | HF repo to fetch when no snapshot is mounted/cached | `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp` |
| `COLI_MODEL_DIR` | host path to a pre-resolved snapshot → mounted at `/model` | unset |
| `COLI_RAM_GB` / `COLI_VRAM_GB` | expert-cache budgets (cap on a shared box to avoid the OOM killer) | available RAM / VRAM |
| `COLI_NUM_NODES` | cluster size (expert-parallel) | `1` |
| `COLI_NODE_RANK` | this node's rank `0..NUM_NODES` | `0` |
| `COLI_CACHE_DIR` | writable path for `.coli_kv` / `.coli_usage` (compose) | `./.coli-cache` |
| `COLI_IMAGE` | image tag used by `run-dgx.sh` | `speedycolibri:latest` |
| `COLI_DOCKER_ARGS` | extra `docker run` flags injected by `run-dgx.sh` | unset |

`colibri-cluster::ClusterConfig::from_env` reads `COLI_NUM_NODES` / `COLI_NODE_RANK`.

## Multi-node plan (expert-parallel)

Chosen split: **expert-parallel MoE**. Each node owns a contiguous block of the
256 experts/layer (`colibri-cluster::ExpertSharding`, e.g. two nodes → 0..128 and
128..256), streams and computes only its block, and the router dispatches each
token's chosen experts to their owner. The dense part (attention, shared expert,
embeddings — ~10 GB int4) is replicated per node so attention stays local and
only expert I/O crosses the wire.

Transport target: **RDMA/RoCE over the ConnectX-7 200 GbE link**, ideally with
GPUDirect RDMA so the Blackwell GPU DMAs activations straight to the wire
(`colibri-cluster::transport`, `--features rdma`). This is designed but not
wired — single node comes first.

To bring up two nodes later:

1. Build the split model layout so each node has its expert shard on local NVMe.
2. Launch node 0 with `COLI_NUM_NODES=2 COLI_NODE_RANK=0`, node 1 with rank `1`.
3. Wire `RdmaTransport::connect` (queue pairs + registered MRs over ConnectX).

On real hardware the two Sparks talk over the direct 200 GbE link with host
networking, not a compose bridge (the disabled `node1` service in the compose
file is only a placeholder for the topology).

## CUDA (Blackwell) backend

`colibri-backend` binds the reference CUDA kernels (`c/backend_cuda.cu`) over FFI.
It is **off by default**; enable it on a CUDA host:

```bash
# in the image (nvcc comes from the -devel base):
docker build --build-arg CUDA_FEATURE=1 --build-arg CUDA_ARCH=sm_121 \
  -f docker/Dockerfile -t speedycolibri:cuda .

# or directly on a DGX Spark:
CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121 cargo build --release -p coli --features cuda
coli backend            # -> "backend: cuda (Cuda(0))" + per-device free/total memory
```

`build.rs` compiles `backend_cuda.cu` with `nvcc` and links `cudart` + `stdc++`.
`CUDA_ARCH` defaults to `native`; use `sm_121` for the GB10.

**Status:** GPU-verified on a DGX Spark (GB10, sm_121, CUDA 13.0) — the FFI
surface (`coli_cuda_init` / `mem_info` / `tensor_upload` / `matmul` /
`expert_mlp` / lifecycle) builds, links, initializes the GPU (130.7 GB VRAM), and
a GPU matmul smoke test passes (`cargo test -p colibri-backend --features cuda`).
`coli backend` reports the GB10. The forward pass is **wired to the GPU**:
`matmul_qt` routes resident weights (dense + preloaded experts) to
`coli_cuda_matmul`, CPU fallback otherwise. Validated on the GB10 —
`COLI_PRELOAD=1 coli gen` runs matmuls on the GPU with output identical to the CPU
path, and an int4 `[8192,6144]` matmul measured **~18× faster** than one Grace
core (429–448 vs 24 GFLOP/s). The expert / shared / dense FFN is **fused** onto
the GPU (`coli_cuda_expert_mlp`, one on-device `down(silu(gate·x)⊙up·x)`) —
measured **19.2×** vs one Grace core (165 µs vs 3171 µs per expert at hidden 6144
/ moe_inter 2048). The MLA attention core also runs on the GPU
(`coli_cuda_attention_absorb_batch`) — **31.9×** vs one Grace core (1349 µs vs
43 ms at H=64 / T=2048), matching the CPU to ~1e-10. **The whole hot path —
attention projections, the absorb core, the fused expert FFN, and lm_head — is
on-device.** Still to do: a persistent device KV cache (avoid re-uploading the
compressed KV each token) and VRAM eviction for the full model.

## Fast cold-start: parallel preload

To saturate the DGX Spark NVMe on cold start, read the experts across all CPU
cores instead of one scattered `pread` at a time. The recommended path needs **no
repack and no second copy** — the safetensors index already knows every tensor's
`(file, offset)`:

```bash
# direct parallel preload from the original model (one thread per core)
COLI_PRELOAD=1 coli gen /model <ids...>
```

`preload_parallel` sorts experts by on-disk offset, gives each core a contiguous
slice to scan, and loads up to the `COLI_RAM_GB` budget — near-sequential reads
with a deep NVMe queue. Output is byte-identical to the serial disk path.

**Optional** — repack for maximum sequentiality (contiguous per-expert blobs),
at the cost of a one-time ~model-sized second copy on disk:

```bash
coli repack /model /model/repacked           # writes experts_NNNN.bin + manifest.json
COLI_PRELOAD=/model/repacked coli gen /model <ids...>
```

`COLI_PRELOAD` picks the repacked path automatically when the value is a directory
containing `manifest.json`; otherwise it preloads directly.

Rule of thumb: N cores keeping the NVMe queue deep approaches the drive's
aggregate bandwidth — several× a single random stream. Pair with `COLI_PIN_GB`
(AUTOPIN) so the *hottest* experts fill the resident set.

## Notes

- The image ships only the `coli` binary; the model, KV-cache, and usage cache
  live on mounted volumes.
- `--gpus all` (or the compose `deploy.resources` block) requires the NVIDIA
  Container Toolkit on the host — standard on DGX OS.
