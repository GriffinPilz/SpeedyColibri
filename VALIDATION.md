# Validating the Rust forward pass against the C engine

The Rust port must reproduce the reference C engine (`c/glm.c`) token-for-token.
The C engine is itself validated **token-exact against a `transformers` oracle**
(teacher-forcing 32/32, greedy 20/20 — see the README and
`c/tools/make_glm_oracle.py`), so matching the C engine transitively validates
against transformers.

Crucially, this needs **no 370 GB model and no torch**: both engines run on the
same tiny synthetic model that uses the *real* GLM-5.2 architecture (MLA,
sigmoid/noaux_tc router, shared + routed experts) with random weights.

## What the harness checks

`scripts/validate_c_vs_rust.py` feeds both engines the identical model + inputs
and diffs their outputs, in two modes and at two bit-widths:

- **generation** — greedy decode from a prompt (`coli gen` vs the C engine's
  default validation mode);
- **teacher-forcing** — one forward over a diverse fixed sequence, argmax at
  every position (`coli tf` vs the C engine's `TF=1` mode). This exercises many
  distinct hidden states, not just the greedy fixed point.

- **dbits = 16 (f32, no quantization):** both engines use pure-scalar matmuls in
  the same order, so the streams must match **byte-exactly**. This is the strong
  correctness proof and is **hard-failed** on any divergence.
- **dbits = 4 (int4):** confirms the Rust quantizer is byte-identical to the C
  one (`round_ties_even`/`lrintf`, same clamps and scale floor). The C int4
  kernel may vectorize (different summation order), so a near-tie *could* flip;
  a divergence here is **reported but not hard-failed**.

The C engine is forced onto the exact CPU path with `IDOT=0 ABSORB=0 DRAFT=0`.

## Running it

```bash
# 1. Build the reference C engine (needs libomp for OpenMP):
make -C c glm                 # produces c/glm
# ...or without libomp, single-threaded (deterministic), with an omp.h shim:
#   printf '#ifndef OMP_H\n#define OMP_H\nstatic inline int omp_get_max_threads(void){return 1;}\nstatic inline int omp_get_thread_num(void){return 0;}\nstatic inline int omp_in_parallel(void){return 0;}\n#endif\n' > /tmp/omp/omp.h
#   clang -O3 -I/tmp/omp -Wno-unknown-pragmas c/glm.c -o c/glm -lm

# 2. Run the harness (builds the Rust CLI itself):
GLM_REF=c/glm python3 scripts/validate_c_vs_rust.py
```

Environment knobs: `GLM_REF` (C binary path), `BITS` (default `16,4`), `NGEN`
(default `12`), `PROMPT` (default `1 5 2 7 3`).

## Current result

Both modes match exactly at both bit-widths on the tiny model:

```
--- 16-bit (f32, byte-exact expected) ---
[gen 16b] MATCH 12/12
[tf  16b] MATCH 10/10   preds [0, 3, 2, 2, 0, 3, 4, 0, 0, 4]  (diverse, not degenerate)
--- 4-bit (int4, tokens expected to match) ---
[gen  4b] MATCH 12/12
[tf   4b] MATCH 10/10
VALIDATION PASSED
```

## Scope and honest caveats

- This validates the **dense CPU path** the Rust engine currently implements. It
  does **not** yet cover the DSA sparse indexer, speculative decoding (MTP), or
  the CUDA backend — none of which are ported yet.
- The tiny model has no DSA/MTP tensors, so both engines run the dense path; a
  full-architecture oracle (with the indexer) would additionally exercise DSA
  once that is ported. Regenerate one with `c/tools/make_glm_oracle.py` (needs
  `torch` + the GLM-MoE-DSA `transformers`) when validating those paths.
- Byte-exactness at int4 against the C engine's *SIMD* kernels is not guaranteed
  (summation order); the real bar there is token agreement, which holds.
