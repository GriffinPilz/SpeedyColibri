#!/usr/bin/env python3
"""Convert a tiktoken BPE rank table into an HF `tokenizer.json`.

Kimi-K3 ships `tiktoken.model` (base64(token_bytes) + rank, one per line) plus a
Python `tokenization_kimi.py`, and NO `tokenizer.json`. colibrì's tokenizer parses
`tokenizer.json` exclusively (`Tokenizer::from_json`: `model.vocab`, `model.merges`,
`added_tokens`), so the ranks have to be turned into a vocab + merge list once.

Usage:
    tiktoken_to_tokenizer_json.py <tiktoken.model> <out/tokenizer.json> \
        [--specials-from <tokenizer_config.json>] [--reserved N] [--verify]

# Deriving merges from ranks

tiktoken stores only `bytes -> rank`; HF wants an ordered merge list. The rank IS
the order a token was created, so for each multi-byte token we recover the pair it
came from: among every split into two tokens that both already exist, take the one
minimising `max(rank(left), rank(right))` — the pair that became mergeable latest,
which is what BPE training would have merged to produce this token. Emitting those
pairs in increasing rank of the merged token reproduces the training order.

`--verify` gates the result properly: it re-runs BPE with the derived merges over
every token's own bytes and asserts each collapses back to exactly that one id. A
merge list that is subtly mis-ordered fails this on the tokens it affects, which a
spot check of a few strings would miss.

# What this does NOT do

The pre-tokenizer. colibrì hard-codes the cl100k split pattern as a hand-written
matcher (`pretok_chunk`), and K3's `pat_str` is different — it adds a leading
`[\\p{Han}]+` alternative, splits the letter alternative by case with Han excluded,
and attaches the `(?i:'s|'t|...)` contraction suffix to the letter run instead of
making it a leading alternative. Those disagree on CJK, on case transitions, and on
contractions. This script emits the K3 pattern into `pre_tokenizer` for the record,
but colibrì does not read it yet — see `compare_pretok.py` for where they diverge.
"""
import base64
import json
import sys


def load_ranks(path):
    """`base64(bytes) rank` per line -> {bytes: rank}."""
    ranks = {}
    with open(path, "rb") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                b64, rank = line.split()
            except ValueError:
                raise SystemExit(f"{path}:{lineno}: expected '<base64> <rank>', got {line!r}")
            ranks[base64.b64decode(b64)] = int(rank)
    return ranks


def byte_to_unicode():
    """GPT-2 ByteLevel byte<->printable-codepoint map, matching `build_bytemap`."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("\xa1"), ord("\xac") + 1))
    bs += list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = byte_to_unicode()


def enc(token: bytes) -> str:
    return "".join(B2U[b] for b in token)


def derive_merges(ranks):
    """Recover the ordered merge list by REPLAYING BPE training.

    A token `T` with rank `r` was produced by merging some pair `(A, B)` whose own
    ranks are both `< r`. So process tokens in rank order and, for each, run BPE over
    its bytes using only the merges discovered so far: the result is exactly `[A, B]`,
    and that pair is `T`'s merge. Appending in rank order reproduces the training
    order, which is what the merge list encodes.

    An earlier version instead picked, among all splits into two known tokens, the one
    minimising `max(rank(a), rank(b))`. That is a plausible-sounding shortcut and it is
    WRONG: 11,108 of K3's 163,584 tokens failed to round-trip under it (` out` split to
    `' ' + 'out'` but the emitted merge was a different pair, so the piece never
    reformed). Only the replay gets the pair right, because only the replay knows which
    pieces actually survive the earlier merges.
    """
    merges = []
    mrank = {}
    unmerged = []
    for token, _ in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(token) < 2:
            continue
        parts = bpe(token, mrank)
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
            mrank[(parts[0], parts[1])] = len(merges) - 1
        else:
            # 1 piece: already reachable, so no new merge is needed. >2: the token
            # cannot be formed from existing pieces at all (tokens injected outside
            # training). Either way it stays in the vocab, just not via a merge.
            unmerged.append(token)
    return merges, unmerged


def bpe(token: bytes, mrank):
    """Apply the derived merges to `token`'s bytes; return the resulting pieces."""
    parts = [bytes([b]) for b in token]
    while len(parts) > 1:
        best, at = None, -1
        for i in range(len(parts) - 1):
            r = mrank.get((parts[i], parts[i + 1]))
            if r is not None and (best is None or r < best):
                best, at = r, i
        if at < 0:
            break
        parts[at : at + 2] = [parts[at] + parts[at + 1]]
    return parts


def verify(ranks, merges):
    """Every token must collapse to itself under the derived merges."""
    mrank = {(a, b): i for i, (a, b) in enumerate(merges)}
    bad = []
    for token in ranks:
        if len(token) < 2:
            continue
        parts = bpe(token, mrank)
        if len(parts) != 1 or parts[0] != token:
            bad.append((token, parts))
    return bad


# K3's pre-tokenizer, from `tokenization_kimi.py`. Emitted for the record; note the
# `&&` character-class intersection is ICU/Java syntax, not PCRE — it is NOT usable
# as-is by a Rust `regex`/`fancy-regex` engine.
K3_PAT = "|".join([
    r"[\p{Han}]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
])


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    src, dst = argv[1], argv[2]
    cfg_path = None
    reserved = 256
    do_verify = "--verify" in argv
    if "--specials-from" in argv:
        cfg_path = argv[argv.index("--specials-from") + 1]
    if "--reserved" in argv:
        reserved = int(argv[argv.index("--reserved") + 1])

    ranks = load_ranks(src)
    n_base = len(ranks)
    print(f"[tok] {n_base} base tokens from {src}")

    merges, unmerged = derive_merges(ranks)
    print(f"[tok] {len(merges)} merges derived; {len(unmerged)} multi-byte tokens unmergeable")

    if do_verify:
        bad = verify(ranks, merges)
        if bad:
            for t, p in bad[:10]:
                print(f"  MISMATCH {t!r} -> {p!r}", file=sys.stderr)
            raise SystemExit(f"[tok] FAIL: {len(bad)} tokens do not round-trip")
        print(f"[tok] verified: all {n_base} tokens round-trip under the derived merges")

    # Specials occupy [n_base, n_base + reserved). `tokenizer_config.json`'s
    # added_tokens_decoder names the ones that are not reserved placeholders.
    names = {}
    if cfg_path:
        with open(cfg_path) as fh:
            dec = json.load(fh).get("added_tokens_decoder", {}) or {}
        for sid, meta in dec.items():
            names[int(sid)] = meta["content"] if isinstance(meta, dict) else str(meta)

    added = []
    for i in range(n_base, n_base + reserved):
        added.append({
            "id": i,
            "content": names.get(i, f"<|reserved_token_{i}|>"),
            "single_word": False, "lstrip": False, "rstrip": False,
            "normalized": False, "special": True,
        })
    # Anything named beyond the reserved block (chat tokens) is kept too, so the
    # chat template can encode. NOTE these can exceed the embedding table — see
    # the README note emitted below.
    for sid in sorted(names):
        if sid >= n_base + reserved:
            added.append({
                "id": sid, "content": names[sid],
                "single_word": False, "lstrip": False, "rstrip": False,
                "normalized": False, "special": True,
            })

    out = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        # Recorded for provenance; colibrì still applies its built-in cl100k split.
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": K3_PAT},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {enc(t): r for t, r in sorted(ranks.items(), key=lambda kv: kv[1])},
            "merges": [f"{enc(a)} {enc(b)}" for a, b in merges],
        },
    }
    with open(dst, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"[tok] wrote {dst}  (vocab {len(out['model']['vocab'])}, "
          f"merges {len(merges)}, added {len(added)})")


if __name__ == "__main__":
    main(sys.argv)
