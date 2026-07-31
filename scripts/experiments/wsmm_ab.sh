#!/usr/bin/env bash
# OBSOLETE — kept only so the result is findable. Do not run it expecting an A/B.
#
# This toggled COLI_NVFP4_WSMM to compare the weight-stationary NVFP4 expert GEMM against
# the WMMA tile. That knob no longer exists: `nvfp4_wsmm_launch` selects the smallest MT
# bucket >= S and declines above 32, and S = tokens*top_k/n_experts already carries both the
# model and the cluster shape, so there was nothing for an override to decide that the
# launcher was not deciding better.
#
# It is stubbed rather than deleted because if it still ran, both arms would set an ignored
# variable, produce identical configurations, and report ~1.00x — a false negative that
# reads exactly like "the kernel does not help".
#
# Results it produced, both token-identical:
#   Nemotron (relu2, serve, warm prefill) : 1.24x wall, 1.48x on the kernel   (#90)
#   MiniMax-M2.7 (SwiGLU, prefill)        : 1.16x gpu-ffn, 258623 -> 222521 ms
#
# The larger M2.7 lever turned out to be elsewhere: expert weight staging was gated on
# S >= 16 and a routed expert only ever sees S ~ 4, so it read dirty host pages on every
# expert call. Removing that gate is 1.24x gpu-ffn on its own. See
# `coli_cuda_expert_mlp_nvfp4` in crates/colibri-backend/cuda/backend_cuda.cu.
echo "wsmm_ab.sh is obsolete: COLI_NVFP4_WSMM was removed — the kernel is chosen from S." >&2
echo "See the header of this file for the measurements it produced." >&2
exit 1
