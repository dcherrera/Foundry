#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# ingot_measure_chain.sh — measure a whole multi-shard safetensors model end to end in format
# version 4 (match mode with refs across shards). Shards go in name order:
#   1. encode shard k with --self-check, --refs = shards 1..k-1 of the source;
#   2. decode the written .ingot with --refs = the shards 1..k-1 *rebuilt* by this run;
#   3. check the rebuilt sha256 against the expected one (Hugging Face LFS) and keep the rebuilt
#      shard (later shards may refer to it) until the whole model is done.
#
#   ingot/scripts/ingot_measure_chain.sh [options] <model-dir> <sha256-file> <work-dir> [<ingot-dir>]
#
# <sha256-file>: lines "sha256  shard-name" (sha256sum format; the LFS hashes). <work-dir> holds
# the logs and the rebuilt shards (deleted at the end); <ingot-dir> (default <work-dir>/ingot)
# keeps the .ingot files.
# Options: -j N threads (default 8); --only REGEX shards to encode (refs still use all earlier
# shards); --keep-rebuilt; -- ARGS... extra `ingot encode` arguments (e.g. -- --pool-layers 1).
# Output: <work-dir>/chain.tsv (one line per shard), <shard>.tensors.tsv, <shard>.log, TOTAL.
# Resumable: shards already in chain.tsv are skipped (their rebuilt shard must still exist).
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
INGOT="$HERE/build/ingot"
[ -x "$INGOT" ] || { echo "build the CLI first: make -C ingot" >&2; exit 2; }
THREADS=8 ONLY=. KEEP=0
EXTRA=()
ARGS=()
while [ $# -gt 0 ]; do
    case $1 in
        -j) THREADS=$2; shift 2 ;;
        --only) ONLY=$2; shift 2 ;;
        --keep-rebuilt) KEEP=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
[ ${#ARGS[@]} -ge 3 ] || { sed -n '4,20p' "$0" >&2; exit 2; }
SRC=${ARGS[0]} SHAS=${ARGS[1]} WORK=${ARGS[2]}
INGOTS=${ARGS[3]:-$WORK/ingot}
mkdir -p "$WORK/rebuilt" "$INGOTS"
LOG="$WORK/chain.tsv"
MIN_FREE_GB=${INGOT_MIN_FREE_GB:-15}

free_gb() { df -Pk "$1" | awk 'NR == 2 {print int($4 / 1048576)}'; }
field() { tr '\t' '\n' | awk -v k="$1" '$1 == k {print $2; exit}'; }
file_size() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

mapfile -t SHARDS < <(cd "$SRC" && ls -1 *.safetensors | sort)
[ ${#SHARDS[@]} -gt 0 ] || { echo "no shards in $SRC" >&2; exit 3; }
[ -s "$LOG" ] || printf '#shard\tsource_bytes\telements\toutput_bytes\tv3style_v2_bytes\tencode_s\tdecode_s\tsha256\tmatch_tensors\tmatch_elements\tmatch_bytes\n' > "$LOG"

for k in "${!SHARDS[@]}"; do
    name=${SHARDS[$k]}
    stem=${name%.safetensors}
    [[ $name =~ $ONLY ]] || continue
    grep -q "^$stem	" "$LOG" && continue
    want=$(awk -v n="$name" '$2 == n || $2 == "*" n {print $1}' "$SHAS")
    [ -n "$want" ] || { echo "no expected sha256 for $name" >&2; exit 3; }
    size=$(file_size "$SRC/$name")
    [ "$(free_gb "$WORK")" -ge $((MIN_FREE_GB + 2 * (size >> 30) + 1)) ] ||
        { echo "refusing $stem: free disk $(free_gb "$WORK") GB" >&2; exit 4; }
    enc_refs=() dec_refs=()
    for ((j = 0; j < k; j++)); do
        enc_refs+=(--refs "$SRC/${SHARDS[$j]}")
        dec_refs+=(--refs "$WORK/rebuilt/${SHARDS[$j]}")
        [ -s "$WORK/rebuilt/${SHARDS[$j]}" ] || { echo "missing rebuilt ${SHARDS[$j]}" >&2; exit 5; }
    done
    dst="$INGOTS/$name.ingot"
    rm -f "$dst" "$WORK/rebuilt/$name"
    t0=$(date +%s)
    enc=$(nice -n 10 "$INGOT" encode -q -j "$THREADS" --self-check \
              --tensor-log "$WORK/$stem.tensors.tsv" "${enc_refs[@]+"${enc_refs[@]}"}" \
              "${EXTRA[@]+"${EXTRA[@]}"}" "$SRC/$name" "$dst" | tee "$WORK/$stem.log")
    t1=$(date +%s)
    line=$(echo "$enc" | grep "^encoded") || { echo "encode failed: $stem" >&2; exit 6; }
    out_bytes=$(echo "$line" | field output_bytes)
    v2=$(echo "$line" | field v2_output_bytes)
    elements=$(echo "$enc" | grep '^mode' | tr '\t' '\n' | awk '$1 == "elements" {s += $2} END {print s}')
    mline=$(echo "$enc" | grep -P '^mode\tmatch\t' || true)
    m_t=$(echo "$mline" | field tensors) m_e=$(echo "$mline" | field elements)
    m_b=$(echo "$mline" | field payload_bytes)
    nice -n 10 "$INGOT" decode -q -j "$THREADS" "${dec_refs[@]+"${dec_refs[@]}"}" \
        "$dst" "$WORK/rebuilt/$name" | tee -a "$WORK/$stem.log"
    t2=$(date +%s)
    got=$(sha256sum "$WORK/rebuilt/$name" | awk '{print $1}')
    [ "$got" = "$want" ] || { echo "REBUILT SHA256 MISMATCH $stem: $got != $want" >&2; exit 7; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stem" "$size" "$elements" \
        "$out_bytes" "${v2:--}" "$((t1 - t0))" "$((t2 - t1))" ok "${m_t:-0}" "${m_e:-0}" \
        "${m_b:-0}" >> "$LOG"
    echo "$(date +%T) $stem $(awk -v o="$out_bytes" -v e="$elements" 'BEGIN {printf "%.4f", 8 * o / e}') bits/weight, encode $((t1 - t0)) s, decode $((t2 - t1)) s"
done
awk -F'\t' '!/^#/ {s += $2; e += $3; o += $4; v += $5; te += $6; td += $7; n++}
    END {printf "shards %d\tsource_bytes %d\telements %d\toutput_bytes %d\tbits_per_weight %.4f\tv2_output_bytes %d\tv2_bits_per_weight %.4f\tencode_s %d\tdecode_s %d\n",
         n, s, e, o, 8 * o / e, v, 8 * v / e, te, td}' "$LOG" | tee "$WORK/TOTAL"
done_all=$(grep -vc "^#" "$LOG" || true)
[ "$KEEP" = 1 ] || [ "$done_all" -ne ${#SHARDS[@]} ] || rm -rf "$WORK/rebuilt"
