#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# ingot_measure.sh — measure a whole safetensors model end to end without ever storing it: each
# shard streams (ingot_fetch.sh) into `ingot encode --self-check` (every coded tensor decoded
# back and compared in memory), the source SHA-256 is checked against the Hugging Face LFS hash,
# and with --rebuild the .ingot is written, rebuilt by the streaming decoder (`ingot verify`)
# whose SHA-256 is checked against the LFS hash too, then deleted. One shard on disk at most.
#
#   ingot/scripts/ingot_measure.sh [options] <org/repo> <out-dir>
#
# Options:
#   --rev REV           HF revision (default: main); the LFS hashes always come from HF
#   --src SRC           where the shards are read: hf (default), ssh:HOST:DIR or a local DIR
#   --wait              wait for each shard to appear at SRC with its full size (a download
#                       still filling SRC)
#   --rebuild           also write each .ingot, rebuild it with the streaming decoder and check
#                       the rebuilt SHA-256 (needs disk for one .ingot)
#   --only REGEX        shards whose name matches (default: all)
#   -j N                threads (default 4)
#   -- ARGS...          extra `ingot encode` arguments (e.g. -- --policy plain)
# Output: <out-dir>/measure.tsv (one line per shard), <out-dir>/<shard>.tensors.tsv (when the
# encoder supports --tensor-log), <out-dir>/<shard>.log, <out-dir>/TOTAL. Resumable: shards
# already in measure.tsv are skipped.
# Env: INGOT_MIN_FREE_GB (default 15), GUARD_* (metal/run_guarded.sh limits).
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
INGOT="$HERE/build/ingot"
GUARD="$HERE/../metal/run_guarded.sh"
[ -x "$INGOT" ] || { echo "build the CLI first: make -C ingot" >&2; exit 2; }

REV=main SRC=hf WAIT=0 REBUILD=0 ONLY=. THREADS=4
EXTRA=()
while [ $# -gt 0 ]; do
    case $1 in
        --rev) REV=$2; shift 2 ;;
        --src) SRC=$2; shift 2 ;;
        --wait) WAIT=1; shift ;;
        --rebuild) REBUILD=1; shift ;;
        --only) ONLY=$2; shift 2 ;;
        -j) THREADS=$2; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) [ -z "${REPO:-}" ] && REPO=$1 || OUT=$1; shift ;;
    esac
done
[ -n "${REPO:-}" ] && [ -n "${OUT:-}" ] || { sed -n '4,27p' "$0" >&2; exit 2; }
MIN_FREE_GB=${INGOT_MIN_FREE_GB:-15}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
LOG="$OUT/measure.tsv"
CURL=(curl -sfL --retry 10 --retry-all-errors)
[ -n "${HF_TOKEN:-}" ] && CURL+=(-H "Authorization: Bearer $HF_TOKEN")
TENSOR_LOG=0
USAGE=$("$INGOT" 2>&1 || true)
[[ $USAGE == *--tensor-log* ]] && TENSOR_LOG=1
export GUARD_TIMEOUT_SEC=${GUARD_TIMEOUT_SEC:-7200}

free_gb() { df -g "$OUT" | awk 'NR==2 {print $4}'; }
field() { tr '\t' '\n' | awk -v k="$1" '$1 == k {print $2; exit}'; } # "key value" fields

# Shard list with LFS size and sha256 (from HF, whatever SRC is).
"${CURL[@]}" "https://huggingface.co/api/models/$REPO/tree/$REV" |
    jq -r '.[] | select(.path | endswith(".safetensors")) | "\(.path)\t\(.lfs.size)\t\(.lfs.oid)"' |
    grep -E "$ONLY" > "$OUT/shards.tsv"
[ -s "$OUT/shards.tsv" ] || { echo "no shards for $REPO@$REV" >&2; exit 3; }

source_of() { # shard name -> fetch argument
    case $SRC in
        hf) echo "https://huggingface.co/$REPO/resolve/$REV/$1" ;;
        ssh:*) echo "$SRC/$1" ;;
        *) echo "$SRC/$1" ;;
    esac
}
remote_size() { # shard name -> its size at SRC (empty if absent)
    case $SRC in
        hf) echo "$2" ;;
        ssh:*) local r=${SRC#ssh:}
               ssh -n -o ConnectTimeout=20 "${r%%:*}" "stat -c %s $(printf '%q' "${r#*:}/$1")" \
                   2>/dev/null || true ;;
        *) stat -c %s "$SRC/$1" 2>/dev/null || stat -f %z "$SRC/$1" 2>/dev/null || true ;;
    esac
}
fetch() { # shard name -> bytes on stdout
    case $SRC in
        hf|ssh:*) "$HERE/scripts/ingot_fetch.sh" "$(source_of "$1")" ;;
        *) cat "$SRC/$1" ;;
    esac
}

measure_shard() { # name size sha
    local name=$1 size=$2 sha=$3 stem fifo dst tlog=() enc src_sha out_bytes elements v2 t0 t1
    stem=$(basename "$name" .safetensors)
    if [ "$WAIT" = 1 ]; then
        until [ "$(remote_size "$name" "$size")" = "$size" ]; do sleep 60; done
    fi
    [ "$(free_gb)" -ge $((MIN_FREE_GB + (REBUILD ? (size >> 30) + 1 : 0))) ] ||
        { echo "refusing $stem: free disk $(free_gb) GB" >&2; exit 4; }
    fifo="$OUT/.$stem.fifo"
    dst=""
    [ "$REBUILD" = 1 ] && dst="$OUT/$name.ingot"
    [ "$TENSOR_LOG" = 1 ] && tlog=(--tensor-log "$OUT/$stem.tensors.tsv")
    rm -f "$fifo" "$dst"
    mkfifo "$fifo"
    fetch "$name" > "$fifo" &
    local fetch_pid=$!
    t0=$(date +%s)
    local cmd=("$INGOT" encode -q -j "$THREADS" --self-check "${tlog[@]+"${tlog[@]}"}"
               "${EXTRA[@]+"${EXTRA[@]}"}")
    if [ -n "$dst" ]; then cmd+=(- "$dst"); else cmd+=(--measure -); fi
    # the guard backgrounds its child (stdin /dev/null), so the child opens the fifo itself
    enc=$(INGOT_IN="$fifo" nice -n 10 "$GUARD" bash -c 'exec "$@" < "$INGOT_IN"' _ "${cmd[@]}" |
              tee "$OUT/$stem.log") || true
    t1=$(date +%s)
    wait "$fetch_pid" || { echo "fetch failed: $stem" >&2; rm -f "$fifo"; exit 5; }
    rm -f "$fifo"
    local line
    line=$(echo "$enc" | grep "^encoded") || { echo "encode failed: $stem" >&2; kill "$fetch_pid" 2>/dev/null; rm -f "$fifo"; exit 5; }
    src_sha=$(echo "$line" | field sha256)
    out_bytes=$(echo "$line" | field output_bytes)
    v2=$(echo "$line" | field v2_output_bytes)
    elements=$(echo "$enc" | grep '^mode' | tr '\t' '\n' | awk '$1 == "elements" {s += $2} END {print s}')
    [ "$src_sha" = "$sha" ] || { echo "SOURCE SHA256 MISMATCH $stem: $src_sha != $sha" >&2; exit 6; }
    local vs="-" rebuilt="-"
    if [ -n "$dst" ]; then
        local ver
        ver=$(nice -n 10 "$GUARD" "$INGOT" verify -q -j "$THREADS" "$dst" | tee -a "$OUT/$stem.log")
        rebuilt=$(echo "$ver" | grep '^verify' | field sha256)
        vs=$(echo "$ver" | grep '^verify' | field seconds)
        [ "$rebuilt" = "$sha" ] || { echo "REBUILT SHA256 MISMATCH $stem: $rebuilt != $sha" >&2; exit 6; }
        rebuilt=ok
        rm -f "$dst"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stem" "$size" "$elements" "$out_bytes" \
        "${v2:--}" "$((t1 - t0))" "$vs" sha-ok "$rebuilt" >> "$LOG"
    echo "$(date +%T) $stem $(awk -v o="$out_bytes" -v e="$elements" 'BEGIN {printf "%.4f", 8 * o / e}') bits/weight"
}

[ -s "$LOG" ] || printf '#shard\tsource_bytes\telements\toutput_bytes\tv2_output_bytes\tencode_s\tverify_s\tsource_sha256\trebuilt_sha256\n' > "$LOG"
while IFS=$'\t' read -r name size sha <&3; do
    grep -q "^$(basename "$name" .safetensors)	" "$LOG" && continue
    measure_shard "$name" "$size" "$sha"
done 3< "$OUT/shards.tsv"
awk -F'\t' '!/^#/ {s += $2; e += $3; o += $4; if ($5 != "-") v += $5; else nv = 1; n++}
    END {printf "shards %d\tsource_bytes %d\telements %d\toutput_bytes %d\tbits_per_weight %.4f", n, s, e, o, 8 * o / e;
         if (!nv) printf "\tv2_output_bytes %d\tv2_bits_per_weight %.4f", v, 8 * v / e; printf "\n"}' \
    "$LOG" | tee "$OUT/TOTAL"
