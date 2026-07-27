#!/usr/bin/env bash
# MTP Mamba-state rollback correctness gate (Nemotron-H).
#
# Speculative decode must be EXACT: DRAFT>0 has to emit the same greedy tokens as
# DRAFT=0. The Mamba2 recurrent state is corrupted by rejected drafts unless it is
# rolled back, so this compares three arms of ONE binary:
#   base   : DRAFT=0                          (no drafting — the oracle)
#   fixed  : DRAFT=N  COLI_MTP_ROLLBACK=1     (rollback on, default)
#   buggy  : DRAFT=N  COLI_MTP_ROLLBACK=0     (rollback off — should diverge)
# PASS iff  fixed == base  AND  buggy != base  (the second half proves the gate has
# teeth; if buggy also matched, the drafts never got rejected and the test is vacuous).
set -uo pipefail
cd /home/dgx1/SpeedyColibri-nemotron
BIN=./target/release/coli
C=/home/dgx1/models/Nemotron-3-Super-120B-container
NGEN="${NGEN:-48}"
DRAFT_N="${DRAFT_N:-6}"

# First 32 tokens of the registry passage (real language — avoids the degenerate
# synthetic-token trap that makes benign FP noise trip identity gates).
PROMPT="1784 6330 1307 23716 6609 1454 17054 22028 10483 1046 8810 1398 23558 1541 9543 1278 12243 92709 14861 1044 1261 4352 10485 17054 8648 1046 63212 41355 1299 1771 9956 1278"

run() { # $1=label  $2=extra-env
  local log="/tmp/mtp_${1}.log"
  env $2 COLI_NGEN="$NGEN" $BIN gen "$C" $PROMPT >"$log" 2>&1
  local ids; ids=$(grep -oE 'generated \([0-9]+ tok\): \[[^]]*\]' "$log" | sed -E 's/.*\[//; s/\]//')
  local mtp; mtp=$(grep -E '\[MTP\].*accepted' "$log" | tail -1)
  echo "== $1 =="
  echo "  ids: $ids"
  [ -n "$mtp" ] && echo "  $mtp"
  echo "$ids" > "/tmp/mtp_${1}.ids"
}

run base  "DRAFT=0"
run fixed "DRAFT=$DRAFT_N COLI_MTP_ROLLBACK=1"
run buggy "DRAFT=$DRAFT_N COLI_MTP_ROLLBACK=0"

echo
echo "=== verdict ==="
if diff -q /tmp/mtp_base.ids /tmp/mtp_fixed.ids >/dev/null; then
  echo "fixed == base : PASS (speculative decode is token-identical)"
else
  echo "fixed != base : FAIL (rollback did not make drafting exact)"
fi
if diff -q /tmp/mtp_base.ids /tmp/mtp_buggy.ids >/dev/null; then
  echo "buggy == base : (gate vacuous — no drafts were rejected; raise NGEN/DRAFT_N)"
else
  echo "buggy != base : the bug is real and this gate detects it"
fi
