# SpeedyColibri container for NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64).
#
# Two stages on a matched Ubuntu 24.04 / CUDA base so glibc lines up:
#   - builder: CUDA -devel + rustup, compiles `coli` with the CUDA backend (nvcc)
#   - runtime: CUDA -runtime + the `coli` binary + the `hf` downloader CLI
#
# The 744B model is NOT baked into the image. At run time the entrypoint resolves
# it in this order (see docker/entrypoint.sh):
#   1. a snapshot mounted at /model,
#   2. the Hugging Face cache (mount the host's ~/.cache/huggingface),
#   3. download from the Hub — pass an HF token as the first argument (hf_...)
#      or via HF_TOKEN / HUGGING_FACE_HUB_TOKEN in the environment.
#
# Build (on/for aarch64) — pass GIT_SHA so the image can report what it was built
# from (`coli version`); .git is not in the build context:
#   docker build -f docker/Dockerfile \
#     --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t speedycolibri:latest .
# Run on a DGX Spark host (GPU passthrough + HF cache handled for you):
#   docker/run-dgx.sh [hf_TOKEN] gen 100 200 300
# Or by hand:
#   docker run --rm -it --gpus all \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -e HF_TOKEN speedycolibri:latest gen 100 200 300

ARG CUDA_TAG=13.0.1
ARG UBUNTU=ubuntu24.04

# ---- builder ---------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:${CUDA_TAG}-devel-${UBUNTU} AS builder

# The workspace has no crates.io dependencies (std + path crates only), so the
# build needs no network beyond installing the toolchain.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Pinned, reproducible Rust toolchain.
ARG RUST_VERSION=1.83.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain ${RUST_VERSION}
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
COPY . .

# Source revision stamp. `.dockerignore` excludes `.git`, so the build cannot derive
# it — pass it in:
#   docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) ...
# It surfaces in `coli version` and the serve/worker banners, so an image built from
# stale source is visible rather than silently running old code (and two cluster
# nodes can be compared at a glance). Defaults to `unknown` if not supplied.
ARG GIT_SHA=unknown
ENV COLI_GIT_SHA=${GIT_SHA}

# The CUDA (Blackwell) backend is the default. CUDA_ARCH must be an explicit
# sm_* here — `native` probes a GPU, and there is none during `docker build`.
# CPU-only image: --build-arg CUDA_FEATURE=0.
#
# `sm_121a`, not `sm_121`: the `a` (architecture-specific) target is where NVIDIA
# exposes GB10's block-scaled tensor-core instructions. ptxas rejects NVFP4's
# `mma.sync...kind::mxf4nvf4.block_scale` on plain `sm_121` and accepts it on
# `sm_121a` (measured on the box). Both target compute 12.1 only, so this costs no
# device coverage.
ARG CUDA_FEATURE=1
ARG CUDA_ARCH=sm_121a
ENV CUDA_HOME=/usr/local/cuda
RUN if [ "$CUDA_FEATURE" = "1" ]; then \
        CUDA_ARCH="$CUDA_ARCH" cargo build --release -p coli --features cuda; \
    else \
        cargo build --release -p coli; \
    fi \
    && cp target/release/coli /usr/local/bin/coli \
    && strip /usr/local/bin/coli

# ---- runtime ---------------------------------------------------------------
FROM nvcr.io/nvidia/cuda:${CUDA_TAG}-runtime-${UBUNTU} AS runtime

LABEL org.opencontainers.image.title="SpeedyColibri" \
      org.opencontainers.image.description="GLM-5.2 744B MoE engine (Rust port) for DGX Spark" \
      org.opencontainers.image.source="https://github.com/GriffinPilz/SpeedyColibri"

# Let the NVIDIA container runtime expose the Blackwell GPU.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    COLI_NUM_NODES=1 \
    COLI_NODE_RANK=0 \
    HF_HOME=/root/.cache/huggingface

# `hf` CLI for pulling the model snapshot on first run (Ubuntu 24.04 is
# PEP 668 externally-managed, so it lives in its own venv). `iproute2` gives the
# `ip` tool that ConnectX discovery reads interface addresses from.
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv iproute2 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/hf \
    && /opt/hf/bin/pip install --no-cache-dir -q huggingface_hub \
    && for cli in hf huggingface-cli; do \
         [ -x "/opt/hf/bin/$cli" ] && ln -s "/opt/hf/bin/$cli" "/usr/local/bin/$cli"; \
       done \
    && { [ -x /usr/local/bin/hf ] || [ -x /usr/local/bin/huggingface-cli ]; }

COPY --from=builder /usr/local/bin/coli /usr/local/bin/coli
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Optional pre-resolved snapshot mount (read-only); otherwise the HF cache
# (mount the host's) or a fresh download provides the model.
VOLUME ["/model"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["help"]
