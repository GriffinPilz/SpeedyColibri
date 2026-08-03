#!/usr/bin/env bash
# Fetch and convert the full unsloth/Kimi-K3 checkpoint ONE SHARD AT A TIME.
#
# Why not just download it and run `coli convert`: the source is 1561 GB and the
# container is ~1505 GB (measured: 94 text shards, 16.0 GB each), so holding both needs
# ~3.0 TB — more than a Spark's 3.6 TB root has spare once any other model is present.
# Converting shard-by-shard and deleting each source as it lands keeps the peak at
# `container-so-far + 2 shards` (~1.54 TB). `preflight_space` below checks the real
# figure at start-up rather than trusting a number written here, which goes stale.
#
# This is only sound because of three properties of the checkpoint, verified against
# `model.safetensors.index.json` before the script was written:
#   1. every `*.weight_packed` is in the SAME shard as its `*.weight_scale`
#      (247,296 pairs, zero cross-shard) — so a shard converts without its neighbours;
#   2. no layer spans more than one shard (93 layers, one per shard);
#   3. the non-layer tensors (embed / lm_head / norms / output_attn_res) live in
#      shards 94-96.
# If a future repack breaks (1), per-shard conversion silently loses scales. Re-check
# it before pointing this at a different repo.
#
# `convert_snapshot` names its output `out-<input-file-index>.safetensors`, and the index
# is assigned over whatever is in the input dir — so converting one shard at a time always
# yields `out-00000`. Each is renamed to the true shard number on the way into the
# container. The loader globs `*.safetensors` and the shards are self-describing, so only
# uniqueness matters.
#
# Resumable: a shard whose output already exists is skipped, so re-running after an
# interruption picks up where it stopped.
#
# Shard layout, read off the index (layer N is in shard N+1):
#   1..93  one transformer layer each
#   94     embed_tokens + lm_head + model.norm + output_attn_res_{norm,proj}
#   95..96 vision_tower (165 tensors) + mm_projector — VISION ONLY
# `convert` drops every vision tensor, so shards 95-96 convert to nothing at all and are
# not fetched by default: the text-only container is shards 1..94. Ask for them
# explicitly if a VL container is ever wanted; the loop tolerates their empty output
# rather than treating it as a failed conversion.
#
#   Usage: scripts/k3_fetch_convert.sh [first_shard] [last_shard]
#   Env:   K3_SRC, K3_OUT, COLI_BIN, MIN_FREE_GB (default 150),
#          OUT_GB_PER_SHARD / SRC_GB_PER_SHARD (preflight sizing, measured defaults)
set -euo pipefail

SRC=${K3_SRC:-$HOME/models/Kimi-K3-src}
OUT=${K3_OUT:-$HOME/models/Kimi-K3-container}
COLI_BIN=${COLI_BIN:-$(dirname "$0")/../target/release/coli}
BASE=https://huggingface.co/unsloth/Kimi-K3/resolve/main
NSHARD=96
# Last shard carrying text-model weights; 95-96 are vision-only (see the header).
TEXT_LAST=94
MIN_FREE_GB=${MIN_FREE_GB:-150}
# Concurrent shard downloads to keep in flight ahead of the converter. See the loop.
PREFETCH_DEPTH=${PREFETCH_DEPTH:-3}
FIRST=${1:-1}
LAST=${2:-$TEXT_LAST}

META=(config.json generation_config.json tokenizer_config.json added_tokens.json
      tiktoken.model chat_template.jinja tokenization_kimi.py configuration_kimi_k3.py)

die() { echo "[k3] FATAL: $*" >&2; exit 1; }
log() { echo "[k3] $*"; }

# `declare -A` (the in-flight download table) needs bash 4+. macOS ships bash 3.2, so
# say so plainly rather than failing with "declare: -A: invalid option".
(( ${BASH_VERSINFO[0]:-0} >= 4 )) || die "needs bash >= 4 (found ${BASH_VERSION:-unknown})"

# Don't leave a prefetch writing into $SRC after the script dies — a later run would
# find a stale `.part` and `curl -C -` would resume it against a possibly different file.
cleanup() { local j; j=$(jobs -p); [[ -n $j ]] && kill $j 2>/dev/null; return 0; }
trap cleanup EXIT INT TERM

[[ -x $COLI_BIN ]] || die "coli binary not found/executable: $COLI_BIN (build it first)"
mkdir -p "$SRC" "$OUT"

shard_name() { printf 'model-%05d-of-000096.safetensors' "$1"; }
out_name()   { printf 'out-%05d.safetensors' "$1"; }
free_gb()    { df -BG --output=avail "$SRC" | tail -1 | tr -dc '0-9'; }
avail_gb()   { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
fs_of()      { df --output=source "$1" | tail -1 | tr -d '[:space:]'; }

# ---- preflight: does the remaining work actually fit? --------------------------------
# The `free_gb` check inside the loop is a FLOOR guard — it aborts once space is nearly
# gone, which on this script means after hours of downloading at ~16 GB a shard. This
# answers the other question, before anything is fetched: is there room for the shards
# still to convert?
#
# Sizes are measured off the finished container rather than guessed: 94 text shards /
# 1505 GB = 16.0 GB per output shard; source shards are ~16.3 GB each.
#
# Resume-aware on purpose. A finished container plus a re-run must NOT be refused — the
# 1.5 TB is already spent, so `todo` is 0 and the requirement collapses to the staging
# floor. Charging the whole range every time would make the script refuse to re-verify
# its own completed work on the very box that produced it, which is precisely when you
# re-run it.
OUT_GB_PER_SHARD=${OUT_GB_PER_SHARD:-16}
SRC_GB_PER_SHARD=${SRC_GB_PER_SHARD:-17}

preflight_space() {
  local n todo=0 total=$(( LAST - FIRST + 1 )) need_out need_src
  for ((n=FIRST; n<=LAST; n++)); do
    # Same predicate the loop uses to skip, so this estimate cannot disagree with it.
    [[ -s "$OUT/$(out_name "$n")" ]] || todo=$(( todo + 1 ))
  done
  need_out=$(( todo * OUT_GB_PER_SHARD ))
  # Staging peaks at PREFETCH_DEPTH downloads in flight plus the one being converted.
  need_src=$(( (PREFETCH_DEPTH + 1) * SRC_GB_PER_SHARD ))
  (( todo == 0 )) && need_src=0

  local out_fs src_fs out_avail src_avail need
  out_fs=$(fs_of "$OUT"); src_fs=$(fs_of "$SRC")
  out_avail=$(avail_gb "$OUT"); src_avail=$(avail_gb "$SRC")
  log "preflight: $todo of $total shard(s) still to convert (~${OUT_GB_PER_SHARD}GB each)"

  local hint="Free the space, or do it in pieces — the range is resumable:
    scripts/k3_fetch_convert.sh <first> <last>     # e.g. $FIRST $(( FIRST + 19 ))
  Each converted shard costs ~${OUT_GB_PER_SHARD}GB and is kept, so progress is not lost."

  if [[ "$out_fs" == "$src_fs" ]]; then
    # One filesystem holds both, so the peak is additive — this is the default layout.
    need=$(( need_out + need_src + MIN_FREE_GB ))
    log "preflight: need ~${need}GB on $out_fs (output ${need_out} + staging ${need_src} + ${MIN_FREE_GB} floor), have ${out_avail}GB"
    (( out_avail >= need )) || die "not enough disk: need ~${need}GB on $out_fs, have ${out_avail}GB (short ~$(( need - out_avail ))GB).
  $hint"
  else
    log "preflight: \$K3_OUT on $out_fs needs ~$(( need_out + MIN_FREE_GB ))GB, have ${out_avail}GB"
    log "preflight: \$K3_SRC on $src_fs needs ~$(( need_src + MIN_FREE_GB ))GB, have ${src_avail}GB"
    (( out_avail >= need_out + MIN_FREE_GB )) || die "not enough disk for output on $out_fs: need ~$(( need_out + MIN_FREE_GB ))GB, have ${out_avail}GB.
  $hint"
    (( src_avail >= need_src + MIN_FREE_GB )) || die "not enough disk for staging on $src_fs: need ~$(( need_src + MIN_FREE_GB ))GB, have ${src_avail}GB."
  fi
}

# Run before ANY network activity: the point of a preflight is that nothing has been
# fetched yet when it refuses. It first sat below the metadata fetch, which already put
# eight files on disk before the check ran.
preflight_space

# Content-Length for a remote file, so a truncated download is caught rather than
# converted into garbage. A short safetensors file can still parse its header and yield
# silently wrong tensors, so this is a real gate, not belt-and-braces.
remote_size() {
  curl -sIL "$BASE/$1" | awk 'tolower($1)=="content-length:"{v=$2} END{printf "%d", v+0}'
}

# Download to `.part` and rename only on success: the loop below treats "final name
# exists" as "complete", so an in-flight prefetch must never occupy that name.
fetch_shard() {
  local n=$1 f want got
  f=$(shard_name "$n")
  [[ -s $SRC/$f ]] && return 0
  want=$(remote_size "$f")
  [[ $want -gt 0 ]] || die "could not get Content-Length for $f"
  curl -sL --retry 8 --retry-delay 5 --retry-all-errors -C - \
       -o "$SRC/$f.part" "$BASE/$f" || die "download failed: $f"
  got=$(stat -c %s "$SRC/$f.part")
  [[ "$got" == "$want" ]] || die "$f truncated: got $got want $want (delete the .part and rerun)"
  mv "$SRC/$f.part" "$SRC/$f"
}

# ---- metadata, once --------------------------------------------------------------
for f in "${META[@]}"; do
  [[ -s $SRC/$f ]] && continue
  log "fetch $f"
  curl -sL --retry 8 --retry-all-errors -o "$SRC/$f" "$BASE/$f" || die "fetch $f"
done

# ---- shard loop ------------------------------------------------------------------
# Downloads in flight, keyed by shard index.
declare -A DL_PID

# Start a download for shard $1 unless one is already running, the file is already here,
# or the shard is already converted. Bounded by PREFETCH_DEPTH at the call site.
ensure_dl() {
  local n=$1
  (( n > LAST )) && return 0
  [[ -n ${DL_PID[$n]:-} ]] && return 0
  [[ -s $SRC/$(shard_name "$n") ]] && return 0
  [[ -s $OUT/$(out_name "$n") ]] && return 0
  fetch_shard "$n" &
  DL_PID[$n]=$!
  return 0
}

# Block until shard $1's download (if any) has finished.
wait_dl() {
  local n=$1
  [[ -n ${DL_PID[$n]:-} ]] || return 0
  wait "${DL_PID[$n]}" || die "download of shard $n failed"
  unset "DL_PID[$n]"
  return 0
}

t_start=$(date +%s)
for ((n=FIRST; n<=LAST; n++)); do
  o=$OUT/$(out_name "$n")
  if [[ -s $o ]]; then
    log "shard $n/$NSHARD: already converted, skipping"
    # An in-flight download for this shard is wasted work. Let it finish rather than
    # killing it mid-write (which strands a .part), then drop the file.
    if [[ -n ${DL_PID[$n]:-} ]]; then
      wait "${DL_PID[$n]}" || true
      unset "DL_PID[$n]"
      rm -f "$SRC/$(shard_name "$n")" "$SRC/$(shard_name "$n").part"
    fi
    continue
  fi

  avail=$(free_gb)
  (( avail >= MIN_FREE_GB )) || die "only ${avail}GB free, need >= ${MIN_FREE_GB}GB"

  # Keep PREFETCH_DEPTH downloads in flight. A single connection is throttled well below
  # the link: measured on 42b2, a second concurrent stream pulled 32.8 MB/s while the
  # first held ~46 MB/s, so depth is what sets the wall-clock here, not bandwidth.
  # Each extra depth costs one shard of disk (~17 GB) while in flight.
  for ((k=n; k<=n+PREFETCH_DEPTH && k<=LAST; k++)); do ensure_dl "$k"; done

  log "shard $n/$NSHARD: fetching (${avail}GB free, ${#DL_PID[@]} in flight)"
  wait_dl "$n"
  fetch_shard "$n"   # no-op when the prefetch already landed it

  # Stage exactly one shard plus the config the converter needs. Hardlink the shard so
  # staging costs no space (same filesystem).
  stage=$SRC/.stage; tmp=$SRC/.out
  rm -rf "$stage" "$tmp"; mkdir -p "$stage"
  ln "$SRC/$(shard_name "$n")" "$stage/" || die "hardlink failed (same filesystem?)"
  for f in "${META[@]}"; do [[ -s $SRC/$f ]] && cp "$SRC/$f" "$stage/"; done

  log "shard $n/$NSHARD: converting"
  "$COLI_BIN" convert "$stage" "$tmp" || die "convert failed on shard $n"

  produced=("$tmp"/out-*.safetensors)
  if [[ ! -s ${produced[0]} ]]; then
    # A vision-only shard converts to nothing — every tensor is dropped — which is a
    # correct outcome, not a failure. Anywhere in the text range it means the mapping
    # skipped tensors it should have kept, and that must be loud.
    if (( n > TEXT_LAST )); then
      log "shard $n/$NSHARD: vision-only, nothing to convert (expected)"
      rm -rf "$stage" "$tmp"; rm -f "$SRC/$(shard_name "$n")"
      continue
    fi
    die "convert produced no shard for $n (text shard: the name mapping dropped everything)"
  fi
  (( ${#produced[@]} == 1 )) || die "expected 1 output shard for input $n, got ${#produced[@]}"
  mv "${produced[0]}" "$o"

  # config/tokenizer land in the container on the first pass; harmless to refresh.
  for f in config.json generation_config.json tokenizer_config.json added_tokens.json \
           chat_template.jinja; do
    [[ -s $tmp/$f ]] && cp "$tmp/$f" "$OUT/" || true
  done

  rm -rf "$stage" "$tmp"
  rm -f "$SRC/$(shard_name "$n")"
  log "shard $n/$NSHARD: done -> $(basename "$o") ($(du -h "$o" | cut -f1)), $(free_gb)GB free, $(( ($(date +%s)-t_start)/60 ))min elapsed"
done

# ---- tokenizer.json --------------------------------------------------------------
# K3 ships tiktoken.model and no tokenizer.json; colibrì's tokenizer needs the latter.
# The engine picks the K3 pre-tokenizer off the pattern this script writes.
if [[ ! -s $OUT/tokenizer.json && -s $SRC/tiktoken.model ]]; then
  log "building tokenizer.json from tiktoken.model"
  python3 "$(dirname "$0")/tiktoken_to_tokenizer_json.py" \
      "$SRC/tiktoken.model" "$OUT/tokenizer.json" \
      --specials-from "$SRC/tokenizer_config.json" --verify \
    || log "WARNING: tokenizer.json generation failed — run it by hand"
fi

# Count against TEXT_LAST, not NSHARD: a complete text-only container is 94 shards, and
# reporting "94/96" would read like two are missing.
log "complete: $(ls "$OUT"/out-*.safetensors 2>/dev/null | wc -l)/$TEXT_LAST text shards in $OUT"
log "container size: $(du -sh "$OUT" | cut -f1)"
