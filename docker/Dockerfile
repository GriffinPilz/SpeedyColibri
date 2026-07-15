# SpeedyColibri container for NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64).
#
# Two stages on a matched Ubuntu 24.04 / CUDA base so glibc lines up and nvcc is
# available for the CUDA backend when it lands:
#   - builder: CUDA -devel + rustup, compiles the Rust workspace
#   - runtime: CUDA -runtime + just the `coli` binary
#
# The 744B model is NOT baked into the image — mount the snapshot as a volume at
# run time (see docker/README and DEPLOYMENT.md). The engine streams experts from
# that mount.
#
# Build (on/for aarch64):
#   docker build -f docker/Dockerfile -t speedycolibri:latest .
# Run (single node, GPU + model mount):
#   docker run --rm -it --gpus all \
#     -v /data/glm52-int4:/model:ro speedycolibri:latest config /model

ARG CUDA_TAG=12.8.0
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

# Build the `coli` release binary. Set CUDA_FEATURE=1 to compile the CUDA
# (Blackwell) backend via nvcc from this -devel base; CUDA_ARCH=sm_121 for GB10.
#   docker build --build-arg CUDA_FEATURE=1 --build-arg CUDA_ARCH=sm_121 ...
ARG CUDA_FEATURE=0
ARG CUDA_ARCH=native
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
    COLI_NODE_RANK=0

COPY --from=builder /usr/local/bin/coli /usr/local/bin/coli
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Model snapshot is mounted here at run time (read-only); KV/usage caches go to a
# writable path the entrypoint can point at.
VOLUME ["/model"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["help"]
