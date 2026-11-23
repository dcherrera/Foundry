#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# ingot_gguf.sh — convert one GGUF from Hugging Face to .ingot without keeping the GGUF on disk:
# stream it into `ingot encode -`, then stream-verify the result and compare the rebuilt file's
# sha256 with the LFS sha256 Hugging Face reports. Optionally code it as a delta against a base
# GGUF (a local file, or another HF file streamed alongside: its tensors must come in the same
# order, which holds for files from the same converter).
#
#   ingot_gguf.sh [-j N] [--base FILE | --base-hf REPO REV FILE] REPO REV FILE OUTDIR
#
# Writes OUTDIR/FILE.ingot and appends a line to OUTDIR/gguf.tsv:
#   file  source_bytes  ingot_bytes  ratio  sha256  encode_s  verify_s  result
# Needs: curl, the ingot CLI (make -C ingot). HF_TOKEN is used for gated repos.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
INGOT=${INGOT:-$HERE/../build/ingot}
THREADS=4
BASE_FILE="" BASE_HF=()
while [ $# -gt 0 ]; do
    case "$1" in
        -j) THREADS=$2; shift 2 ;;
        --base) BASE_FILE=$2; shift 2 ;;
        --base-hf) BASE_HF=("$2" "$3" "$4"); shift 4 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) break ;;
    esac
done
[ $# -eq 4 ] || { sed -n '4,15p' "$0" >&2; exit 2; }
REPO=$1 REV=$2 FILE=$3 OUT=$4
[ -x "$INGOT" ] || { echo "build the CLI first: make -C ingot" >&2; exit 2; }
mkdir -p "$OUT"
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
url() { echo "https://huggingface.co/$1/resolve/$2/$3"; }
stream() { "$HERE/ingot_fetch.sh" "$(url "$@")"; } # Range chunks, retried

# LFS sha256 and size from the resolve headers.
HDRS=$(curl -sIL "${AUTH[@]+"${AUTH[@]}"}" "$(url "$REPO" "$REV" "$FILE")")
WANT_SHA=$(printf '%s' "$HDRS" | tr -d '\r' | awk -F'"' 'tolower($0) ~ /^x-linked-etag:/ {print $2}' | tail -1)
WANT_SIZE=$(printf '%s' "$HDRS" | tr -d '\r' | awk 'tolower($1) == "x-linked-size:" {print $2}' | tail -1)
[ ${#WANT_SHA} -eq 64 ] || { echo "no LFS sha256 for $REPO/$FILE" >&2; exit 1; }

BASE_ARGS=()
[ -n "$BASE_FILE" ] && BASE_ARGS=(--base "$BASE_FILE" --check-base)
[ ${#BASE_HF[@]} -eq 3 ] && BASE_ARGS=(--base /dev/fd/3) # a fresh base stream per pass

DST="$OUT/$FILE.ingot"
t0=$(date +%s)
base_stream() { if [ ${#BASE_HF[@]} -eq 3 ]; then stream "${BASE_HF[@]}"; fi; }
stream "$REPO" "$REV" "$FILE" | nice -n 10 "$INGOT" encode -q -j "$THREADS" \
    ${BASE_ARGS[@]+"${BASE_ARGS[@]}"} - "$DST" 3< <(base_stream) > "$OUT/$FILE.encode.log"
enc=$?
t1=$(date +%s)
if [ $enc -ne 0 ]; then
    echo "encode failed for $FILE" >&2
    exit 1
fi
nice -n 10 "$INGOT" verify -q -j "$THREADS" ${BASE_ARGS[@]+"${BASE_ARGS[@]}"} "$DST" 3< <(base_stream) \
    > "$OUT/$FILE.verify.log"
ver=$?
t2=$(date +%s)
GOT_SHA=$(awk -F'\t' '{for (i = 1; i <= NF; i++) if ($i ~ /^sha256 /) print substr($i, 8)}' "$OUT/$FILE.verify.log")
SIZE=$(wc -c < "$DST" | tr -d ' ')
RESULT=FAIL
[ $ver -eq 0 ] && [ "$GOT_SHA" = "$WANT_SHA" ] && RESULT=sha256-ok
printf '%s\t%s\t%s\t%.4f\t%s\t%d\t%d\t%s\n' "$FILE" "$WANT_SIZE" "$SIZE" \
    "$(echo "$SIZE / $WANT_SIZE" | bc -l)" "$GOT_SHA" $((t1 - t0)) $((t2 - t1)) "$RESULT" \
    | tee -a "$OUT/gguf.tsv"
[ "$RESULT" = sha256-ok ]
