#!/usr/bin/env python3
"""Fuzz colibrì's Kimi-K3 pre-tokenizer against the reference `pat_str`.

The curated fixture in `crates/colibri-tokenizer/tests/k3_pretok.rs` only covers cases
someone thought of. This generates text from an alphabet chosen to stress the parts of
the pattern that are easy to get wrong — in particular the OVERLAP between the two letter
classes (`Lm`, `Lo`, `M` are in both), which is what forces `[U]*` to backtrack — and
diffs every split against the reference.

Usage:
    cargo build -p colibri-tokenizer --example k3split --release
    python3 scripts/fuzz_k3_pretok.py [n_cases]

Any mismatch is printed with the input, the reference split and colibrì's; exit code is
non-zero if any case differs.
"""
import json
import random
import subprocess
import sys

try:
    import regex
except ImportError:
    raise SystemExit("needs the `regex` module: pip install regex")

K3_PAT = "|".join([
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]&&[^\p{Han}]]*"
    r"[[\p{Ll}\p{Lm}\p{Lo}\p{M}]&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]&&[^\p{Han}]]+"
    r"[[\p{Ll}\p{Lm}\p{Lo}\p{M}]&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
])

# An alphabet picked so the hard interactions actually occur, not just ASCII prose.
ALPHABET = (
    list("abcdefghijkxyz")                 # Ll
    + list("ABCDEFGHIJKXYZ")               # Lu
    + list("0123456789")                   # Nd
    + list(" \t\n\r")                      # \s, incl. the newline alternatives
    + list("'\"!?.,-_/\\@#$%&*()[]{}+=<>|~`^:;")   # punctuation, and the contraction quote
    + list("あいうえおカタカナぁ")            # Lo (in BOTH letter classes) — the backtracking driver
    + list("中国北京漢字")                    # Han (excluded from both letter classes)
    + list("ǅǆǄ")                          # Lt (titlecase) — U only
    + list("̧́̈")           # Mn combining marks — in BOTH classes
    + list("ʰʲˤ")                          # Lm modifier letters — in BOTH classes
    + list("αβγΑΒΓ")                       # Greek Ll/Lu
    + list("абвАБВ")                       # Cyrillic Ll/Lu
    + list("  　")           # exotic whitespace
    + list("😀🌍→")                         # So — neither L nor N
    + ["ß", "ẞ", "ﬁ", "ı", "İ"]            # casing oddities
)


def gen(rng, max_len=24):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, max_len)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rng = random.Random(0xC0FFEE)
    pat = regex.compile(K3_PAT, regex.V1)

    cases = [gen(rng) for _ in range(n)]
    # Reference splits first, so a crash in the Rust side cannot be mistaken for a pass.
    want = [[m.group() for m in pat.finditer(c)] for c in cases]

    exe = "target/debug/examples/k3split"
    import os
    if not os.path.exists(exe):
        exe = "target/release/examples/k3split"
    if not os.path.exists(exe):
        raise SystemExit("build it first: cargo build -p colibri-tokenizer --example k3split")

    # ensure_ascii=False: non-BMP characters (emoji) go through as literal UTF-8. With
    # the default they become \uXXXX surrogate PAIRS, which the helper's simple
    # unescaper turns into U+FFFD — that showed up as a wave of fake "mismatches" that
    # were entirely an artifact of this harness.
    payload = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
    res = subprocess.run([exe], input=payload, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit("k3split failed: %s" % res.stderr[:2000])
    got = [json.loads(line) for line in res.stdout.splitlines()]

    if len(got) != len(cases):
        raise SystemExit("expected %d output lines, got %d" % (len(cases), len(got)))

    bad = 0
    for c, w, g in zip(cases, want, got):
        if w != g:
            bad += 1
            if bad <= 12:
                print("MISMATCH input=%r\n   ref  %r\n   coli %r" % (c, w, g))
    print("\n%d/%d cases match" % (len(cases) - bad, len(cases)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
