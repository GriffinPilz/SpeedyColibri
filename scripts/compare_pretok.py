#!/usr/bin/env python3
"""Where Kimi-K3's pre-tokenizer disagrees with the cl100k one colibrì implements.

colibrì hard-codes the cl100k split pattern as a hand-written matcher
(`colibri-tokenizer::pretok_chunk`); it does not read `pre_tokenizer` out of
`tokenizer.json`. K3's `pat_str` (from `tokenization_kimi.py`) is a different
pattern, so until `pretok_chunk` gains a K3 variant, any K3 text colibrì tokenizes
is split by the wrong rule.

This script says exactly how wrong. Run it before trusting a K3 tokenization:

    scripts/compare_pretok.py [file-of-sample-lines]

Measured on the built-in cases (2026-07-28): 5/8 agree. The three that differ:

  * contractions — K3 attaches `(?i:'s|'t|'re|...)` to the letter run, so `it's` is
    ONE piece; cl100k has it as a leading alternative, so it splits `it` + `'s`.
  * case transitions — K3 splits the letter run into an uppercase* + lowercase+
    form, so `getHTTPResponse` becomes `get` + `HTTPResponse`; cl100k's plain
    `\\p{L}+` keeps it whole.
  * Han adjacency — K3 matches `[\\p{Han}]+` first and excludes Han from the letter
    classes, so a space before CJK is its own piece; cl100k's `[^\\r\\n\\p{L}\\p{N}]?\\p{L}+`
    absorbs it.

Both patterns agree on plain lowercase English, ALLCAPS, digits+punctuation, and
pure-CJK runs — which is why the `"The capital of France is"` MVP prompt tokenizes
identically under both. That agreement is verified, not assumed, but it is a
property of that prompt, NOT a license to use cl100k for K3 generally.

Needs the `regex` module (stdlib `re` has no `&&` character-class intersection).
"""
import sys

try:
    import regex
except ImportError:
    raise SystemExit("needs the `regex` module: pip install regex")

# From tokenization_kimi.py. The `&&` intersections are ICU/Java syntax; `regex`
# accepts them in V1 mode with the operands bracketed, which stdlib `re` cannot do.
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

# What colibri-tokenizer's `pretok_chunk` implements.
CL100K_PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CASES = [
    ("MVP prompt", "The capital of France is"),
    ("plain English", "the quick brown fox jumps over the lazy dog"),
    ("contraction", "it's don't we're"),
    ("CamelCase", "getHTTPResponse CamelCase XMLHttpRequest"),
    ("ALLCAPS", "NASA AND THE USA"),
    ("Chinese", "中国的首都是北京"),
    ("mixed CJK/latin", "Beijing 北京 is the capital"),
    ("numbers+punct", "In 2026, GPT-4o cost $1,234.56!"),
]


def main(argv):
    k3 = regex.compile(K3_PAT, regex.V1)
    cl = regex.compile(CL100K_PAT, regex.V1)
    split = lambda r, s: [m.group() for m in r.finditer(s)]

    cases = CASES
    if len(argv) > 1:
        with open(argv[1]) as fh:
            cases = [(f"line {i}", ln.rstrip("\n")) for i, ln in enumerate(fh, 1) if ln.strip()]

    agree = 0
    for name, s in cases:
        a, b = split(k3, s), split(cl, s)
        if a == b:
            agree += 1
            print(f"SAME  {name}")
        else:
            print(f"DIFF  {name}")
            print(f"        K3    : {a}")
            print(f"        cl100k: {b}")
    print(f"\n{agree}/{len(cases)} agree")
    return 0 if agree == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
