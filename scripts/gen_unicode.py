#!/usr/bin/env python3
"""Convert c/tok_unicode.h into a Rust source file with the same range tables.

The C header is machine-generated (tools/gen_unicode.py); we re-emit its
uni_L/uni_N/uni_S range arrays as Rust `&[(u32,u32)]` slices plus binary-search
predicates, so the Rust tokenizer's pre-tokenizer classifies codepoints
byte-identically to the C engine.
"""
import re
import sys

src = open(sys.argv[1]).read()

def extract(name):
    # match `static const uint32_t <name>[][2] = { ... };`
    m = re.search(r"%s\[\]\[2\]\s*=\s*\{(.*?)\};" % re.escape(name), src, re.S)
    if not m:
        raise SystemExit("could not find %s" % name)
    body = m.group(1)
    pairs = re.findall(r"\{\s*(0x[0-9A-Fa-f]+)\s*,\s*(0x[0-9A-Fa-f]+)\s*\}", body)
    return [(a, b) for a, b in pairs]

L = extract("uni_L")
N = extract("uni_N")
S = extract("uni_S")

out = []
out.append("//! Unicode property range tables for the tokenizer's pre-tokenizer.")
out.append("//!")
out.append("//! GENERATED from c/tok_unicode.h by scripts/gen_unicode.py — do not edit by hand.")
out.append("//! `is_l`/`is_n`/`is_s` mirror the C `is_L`/`is_N`/`is_S` (\\p{L}, \\p{N}, \\s).")
out.append("")

def emit(name, pairs):
    out.append("#[rustfmt::skip]")
    out.append("pub static %s: &[(u32, u32)] = &[" % name)
    line = "    "
    for i, (a, b) in enumerate(pairs):
        tok = "(%s, %s)," % (a, b)
        if len(line) + len(tok) + 1 > 100:
            out.append(line.rstrip())
            line = "    "
        line += tok + " "
    if line.strip():
        out.append(line.rstrip())
    out.append("];")
    out.append("")

emit("UNI_L", L)
emit("UNI_N", N)
emit("UNI_S", S)

out.append("/// Binary-search a codepoint in a sorted, non-overlapping range table.")
out.append("fn uni_in(table: &[(u32, u32)], cp: u32) -> bool {")
out.append("    let (mut lo, mut hi) = (0isize, table.len() as isize - 1);")
out.append("    while lo <= hi {")
out.append("        let m = ((lo + hi) >> 1) as usize;")
out.append("        let (a, b) = table[m];")
out.append("        if cp < a {")
out.append("            hi = m as isize - 1;")
out.append("        } else if cp > b {")
out.append("            lo = m as isize + 1;")
out.append("        } else {")
out.append("            return true;")
out.append("        }")
out.append("    }")
out.append("    false")
out.append("}")
out.append("")
out.append("/// `\\p{L}` — Unicode letter.")
out.append("#[inline] pub fn is_l(c: u32) -> bool { uni_in(UNI_L, c) }")
out.append("/// `\\p{N}` — Unicode number.")
out.append("#[inline] pub fn is_n(c: u32) -> bool { uni_in(UNI_N, c) }")
out.append("/// `\\s` — whitespace (the cl100k set).")
out.append("#[inline] pub fn is_s(c: u32) -> bool { uni_in(UNI_S, c) }")
out.append("")

open(sys.argv[2], "w").write("\n".join(out))
print("wrote %s: L=%d N=%d S=%d ranges" % (sys.argv[2], len(L), len(N), len(S)))
