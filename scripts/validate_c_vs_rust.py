#!/usr/bin/env python3
"""C-vs-Rust validation harness for the SpeedyColibri forward pass.

The reference C engine (c/glm.c) is validated token-exact against a transformers
oracle. This harness checks the Rust port against that C engine on an identical
synthetic tiny model (real GLM architecture, random weights), so no 370 GB model
or torch is needed. It feeds both engines the same prompt and diffs the greedy
token streams.

At dbits=16 (f32, no quantization) both engines use pure-scalar matmuls in the
same order, so the streams must match EXACTLY. At dbits=4 (int4) the C engine's
kernel may vectorize (different summation order), so small logit differences can
occasionally flip a near-tie; tokens are expected to match but a divergence there
is reported, not hard-failed.

Usage:
  GLM_REF=/path/to/glm  python3 scripts/validate_c_vs_rust.py
    GLM_REF   path to the built C engine (default: c/glm)
    BITS      comma list of bit-widths to test (default: 16,4)
    NGEN      tokens to generate (default: 12)
    PROMPT    space/comma-separated prompt ids (default: 1 5 2 7 3)
"""
import json, os, re, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLM_REF = os.environ.get("GLM_REF", os.path.join(REPO, "c", "glm"))
BITS = [int(b) for b in os.environ.get("BITS", "16,4").split(",")]
NGEN = int(os.environ.get("NGEN", "12"))
PROMPT = [int(x) for x in re.split(r"[ ,]+", os.environ.get("PROMPT", "1 5 2 7 3").strip())]


def sh(cmd, env=None, cwd=None):
    e = dict(os.environ)
    if env:
        e.update({k: str(v) for k, v in env.items()})
    return subprocess.run(cmd, env=e, cwd=cwd, capture_output=True, text=True)


def run_c(model_dir, ref_path, bits):
    """Run the C reference; return the greedy continuation token list."""
    r = sh([GLM_REF, "64", str(bits), str(bits)],
           env={"SNAP": model_dir, "REF": ref_path, "IDOT": 0, "ABSORB": 0,
                "DRAFT": 0, "REF_FORCE": 1})
    for line in r.stdout.splitlines():
        if line.startswith("GLM C engine"):
            return [int(t) for t in re.findall(r"-?\d+", line.split(":", 1)[1])]
    sys.stderr.write("C engine produced no token line:\n" + r.stdout + r.stderr + "\n")
    return None


def run_rust(coli, model_dir, bits, ngen):
    """Run the Rust `coli gen`; return the greedy continuation token list."""
    r = sh([coli, "gen", model_dir] + [str(t) for t in PROMPT],
           env={"COLI_DBITS": bits, "COLI_EBITS": bits, "COLI_NGEN": ngen})
    m = re.search(r"generated \(\d+ tok\): \[([^\]]*)\]", r.stdout)
    if not m:
        sys.stderr.write("Rust coli produced no token line:\n" + r.stdout + r.stderr + "\n")
        return None
    body = m.group(1).strip()
    return [int(t) for t in re.findall(r"-?\d+", body)] if body else []


# A diverse, non-degenerate sequence for teacher-forcing (exercises many
# distinct hidden states, unlike the greedy fixed point).
TF_IDS = [1, 5, 2, 7, 3, 0, 4, 8, 6, 9]


def run_c_tf(model_dir, ref_path, bits):
    """Run the C engine in TF mode; return per-position argmax predictions.
    With tf_pred all -1 every position 'mismatches', so the C engine prints its
    prediction for each position to stderr — which we parse."""
    r = sh([GLM_REF, "64", str(bits), str(bits)],
           env={"SNAP": model_dir, "REF": ref_path, "TF": 1, "IDOT": 0,
                "ABSORB": 0, "DRAFT": 0, "REF_FORCE": 1})
    got = {}
    for m in re.finditer(r"pos=(\d+) expected=-?\d+ got=(-?\d+)", r.stderr):
        got[int(m.group(1))] = int(m.group(2))
    if not got:
        sys.stderr.write("C TF produced no preds:\n" + r.stdout + r.stderr + "\n")
        return None
    return [got[i] for i in sorted(got)]


def run_rust_tf(coli, model_dir, ids, bits):
    r = sh([coli, "tf", model_dir] + [str(t) for t in ids],
           env={"COLI_DBITS": bits, "COLI_EBITS": bits})
    m = re.search(r"tf preds \(\d+\): \[([^\]]*)\]", r.stdout)
    if not m:
        sys.stderr.write("Rust tf produced no preds:\n" + r.stdout + r.stderr + "\n")
        return None
    return [int(t) for t in re.findall(r"-?\d+", m.group(1))]


def main():
    if not os.path.exists(GLM_REF):
        sys.exit(f"C reference engine not found at {GLM_REF}. Build it "
                 f"(`make -C c glm`, needs libomp) or set GLM_REF.")

    # build the Rust CLI once
    b = sh(["cargo", "build", "-q", "-p", "coli"], cwd=REPO)
    if b.returncode != 0:
        sys.exit("cargo build failed:\n" + b.stderr)
    coli = os.path.join(REPO, "target", "debug", "coli")

    tmp = tempfile.mkdtemp(prefix="colibri-validate-")
    model_dir = os.path.join(tmp, "model")
    sh([sys.executable, os.path.join(REPO, "scripts", "gen_tiny_model.py"), model_dir])

    # generation ref: prompt + NGEN placeholders so C generates NGEN tokens.
    ref_gen = os.path.join(tmp, "ref_gen.json")
    json.dump({"prompt_ids": PROMPT, "full_ids": PROMPT + [0] * NGEN}, open(ref_gen, "w"))
    # teacher-forcing ref: a diverse sequence, tf_pred all -1 to force printing.
    ref_tf = os.path.join(tmp, "ref_tf.json")
    json.dump({"prompt_ids": TF_IDS, "full_ids": TF_IDS, "tf_pred": [-1] * len(TF_IDS)},
              open(ref_tf, "w"))

    print(f"prompt: {PROMPT}   ngen: {NGEN}   tf_ids: {TF_IDS}   bits: {BITS}")
    print(f"C ref: {GLM_REF}\n")

    failures = 0

    def compare(tag, c, rs, exact_required):
        nonlocal failures
        if c is None or rs is None:
            print(f"{tag} ERROR: an engine produced no output")
            failures += 1
            return
        n = min(len(c), len(rs))
        if c == rs:
            print(f"{tag} MATCH {len(c)}/{len(c)}")
            print(f"        C:    {c}")
            print(f"        Rust: {rs}")
        else:
            div = next((i for i in range(n) if c[i] != rs[i]), n)
            print(f"{tag} DIVERGE at position {div}")
            print(f"        C:    {c}")
            print(f"        Rust: {rs}")
            if exact_required:
                failures += 1

    for bits in BITS:
        exact = bits >= 16
        note = "f32, byte-exact expected" if exact else "int4, tokens expected to match"
        print(f"--- {bits}-bit ({note}) ---")
        compare(f"[gen {bits:>2}b]", run_c(model_dir, ref_gen, bits),
                run_rust(coli, model_dir, bits, NGEN), exact)
        compare(f"[tf  {bits:>2}b]", run_c_tf(model_dir, ref_tf, bits),
                run_rust_tf(coli, model_dir, TF_IDS, bits), exact)
        print()

    if failures:
        print(f"VALIDATION FAILED ({failures} exact-mode mismatch)")
        sys.exit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
