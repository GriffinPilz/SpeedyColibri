# Deploying SpeedyColibri on DGX Spark

Target: a Docker container on **NVIDIA DGX Spark** (GB10 Grace Blackwell) —
aarch64 Grace CPU + Blackwell GPU, 128 GB unified LPDDR5X per node, CUDA. The
engine streams the 744B model's experts from disk, so a single Spark can serve
it; multiple Sparks split the experts across nodes.

## Current status

| Piece | Status |
|---|---|
| Container image (aarch64 + CUDA base) | ✅ Dockerfile ready |
| Single-node run | 🟡 image runs `coli` (`tokenize`/`config` today; `chat` pending the forward-pass port) |
| CUDA (Blackwell) backend | ⬜ stub — CPU path only for now |
| Expert-parallel sharding math | ✅ `colibri-cluster` (tested) |
| Multi-node RDMA transport | ⬜ designed, not wired (single-node first) |

The **compute backend priority is CUDA → CPU**. Apple-Silicon Metal is off the
critical path (kept only as an optional stub). The CPU fallback targets the Grace
ARM cores (aarch64 NEON is the CPU SIMD target).

## Build the image

The workspace has **no crates.io dependencies** (std + path crates only), so the
build needs no registry access. Build on an aarch64 host (or with buildx for
`linux/arm64`):

```bash
docker build -f docker/Dockerfile -t speedycolibri:latest .
# cross-build for DGX Spark from another arch:
# docker buildx build --platform linux/arm64 -f docker/Dockerfile -t speedycolibri:latest .
```

`CUDA_TAG` (default `12.8.0`) is an `ARG` — bump it to match your Spark's CUDA/
driver. Blackwell (GB10, sm_121) needs CUDA ≥ 12.8.

## Run (single node)

The model snapshot is **mounted, never baked in**:

```bash
docker run --rm -it --gpus all \
  -v /data/glm52-int4:/model:ro \
  -v "$PWD/.coli-cache":/cache \
  speedycolibri:latest config /model
```

or with compose:

```bash
COLI_MODEL_DIR=/data/glm52-int4 \
  docker compose -f docker/docker-compose.yml run --rm coli config /model
```

### Environment variables

| Var | Meaning | Default |
|---|---|---|
| `COLI_NUM_NODES` | cluster size (expert-parallel) | `1` |
| `COLI_NODE_RANK` | this node's rank `0..NUM_NODES` | `0` |
| `COLI_CUDA` | enable CUDA backend when compiled in | on if present |
| `COLI_MODEL_DIR` | host path to the model snapshot (compose) | `/data/glm52-int4` |
| `COLI_CACHE_DIR` | writable path for `.coli_kv` / `.coli_usage` (compose) | `./.coli-cache` |

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

## Fast cold-start: repack + parallel preload

Loading experts from the original safetensors means many small, scattered
`pread`s. To saturate the DGX Spark NVMe on cold start, repack the experts into
one contiguous shard file per CPU core, then read all shards in parallel:

```bash
# one-time: repack experts into <cores> byte-balanced shards + a manifest
coli repack /model /model/repacked            # or: coli repack /model /model/repacked 20

# serve with a parallel preload (one reader thread per shard, no per-token I/O)
COLI_PRELOAD=/model/repacked coli gen /model <ids...>
```

`coli repack` writes `experts_NNNN.bin` shards + `manifest.json`
(`(layer,eid) → (file,offset)`). `COLI_PRELOAD` makes `coli gen` read every shard
in parallel into RAM up to the `COLI_RAM_GB` budget (per-shard = budget/cores),
then serve with the experts resident. Output is byte-identical to the disk path.

Rule of thumb: N cores reading N sequential streams approaches the drive's
aggregate bandwidth — e.g. a drive that does ~1 GB/s per random stream but
~5–10 GB/s sequential aggregate loads the resident set several times faster.
Pair with `COLI_PIN_GB` (AUTOPIN) so the *hottest* experts fill the resident set.

## Notes

- The image ships only the `coli` binary; the model, KV-cache, and usage cache
  live on mounted volumes.
- `--gpus all` (or the compose `deploy.resources` block) requires the NVIDIA
  Container Toolkit on the host — standard on DGX OS.
