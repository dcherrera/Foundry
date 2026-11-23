#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# ingot_convert.sh — convert a safetensors model to .ingot, one shard at a time (each shard X
# becomes X.ingot, e.g. model-00001-of-00018.safetensors.ingot).
#
#   ingot/scripts/ingot_convert.sh [options] <org/repo | local-dir> <out-dir>
#
# Hugging Face repo: shards are downloaded one at a time (sha256-checked against the LFS hash),
# encoded, verified bit-identical against the shard, then deleted. Small non-weight files
# (config, tokenizer, index) are copied to <out-dir>. Local dir: every *.safetensors is encoded
# and verified; nothing is deleted. Resumable: shards already verified are skipped.
#
# Options:
#   --rev REV            HF revision (default: main)
#   --base PATH          delta mode against local base files or dirs (repeatable)
#   --base-repo REPO     delta mode against a HF repo: the base tensors each shard needs are
#   --base-rev REV         fetched by HTTP Range (default rev: main) and deleted after the shard
#   --rebuild-check      also rebuild each shard from the .ingot and compare its sha256
#   --keep-source        keep downloaded shards
#   --discard-output     delete each .ingot after it verifies (measurement runs; sizes stay logged)
#   -j N                 threads (default 4)
# Env: INGOT_MIN_FREE_GB (default 15): refuse a download that would leave less free disk.
#      HF_TOKEN: sent as a bearer token if set (gated repos).
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
INGOT="$HERE/build/ingot"
[ -x "$INGOT" ] || { echo "build the CLI first: make -C ingot" >&2; exit 2; }

REV=main BASE_REPO="" BASE_REV=main REBUILD=0 KEEP=0 DISCARD=0 THREADS=4
BASES=()
while [ $# -gt 2 ]; do
    case $1 in
        --rev) REV=$2; shift 2 ;;
        --base) BASES+=(--base "$2"); shift 2 ;;
        --base-repo) BASE_REPO=$2; shift 2 ;;
        --base-rev) BASE_REV=$2; shift 2 ;;
        --rebuild-check) REBUILD=1; shift ;;
        --keep-source) KEEP=1; shift ;;
        --discard-output) DISCARD=1; shift ;;
        -j) THREADS=$2; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[ $# -eq 2 ] || { sed -n '4,24p' "$0" >&2; exit 2; }
SRC=$1 OUT=$2
MIN_FREE_GB=${INGOT_MIN_FREE_GB:-15}
WORK="$OUT/.work"
mkdir -p "$OUT" "$WORK"
LOG="$OUT/convert.tsv"

CURL=(curl -sfL --retry 10 --retry-all-errors)
[ -n "${HF_TOKEN:-}" ] && CURL+=(-H "Authorization: Bearer $HF_TOKEN")

free_gb() { df -g "$OUT" 2>/dev/null | awk 'NR==2 {print $4}' || df -BG "$OUT" | awk 'NR==2 {print $4+0}'; }
sha256_of() {
    if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
file_size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
need_disk() {  # bytes about to land on disk
    local need=$(( ($1 >> 30) + MIN_FREE_GB + 1 ))
    [ "$(free_gb)" -ge "$need" ] || { echo "refusing: free disk $(free_gb) GB < $need GB" >&2; exit 3; }
}

# --- base tensors by HTTP Range (delta against a remote base) ---------------------------------
BASE_URL="https://huggingface.co/$BASE_REPO/resolve/$BASE_REV"
base_map() {  # name -> shard of the base repo, once
    [ -s "$WORK/base_map.tsv" ] && return
    if "${CURL[@]}" "$BASE_URL/model.safetensors.index.json" -o "$WORK/base_index.json"; then
        jq -r '.weight_map | to_entries[] | "\(.key)\t\(.value)"' "$WORK/base_index.json" \
            > "$WORK/base_map.tsv"
    else
        printf '*\tmodel.safetensors\n' > "$WORK/base_map.tsv"
    fi
}
base_header() {  # base shard -> tensor table "name dtype shape begin end"
    local shard=$1 hdr="$WORK/base_hdr/$1"
    mkdir -p "$WORK/base_hdr"
    if [ ! -s "$hdr.tsv" ]; then
        local n
        n=$("${CURL[@]}" -r 0-7 "$BASE_URL/$shard" | od -An -tu8 | tr -d ' ')
        "${CURL[@]}" -r 0-$((n + 7)) "$BASE_URL/$shard" -o "$hdr"
        "$INGOT" st-ls "$hdr" > "$hdr.tsv"
    fi
    echo "$hdr.tsv"
}
fetch_base_for() {  # shard file -> base tensors with the same names in $WORK/base/
    rm -rf "$WORK/base"
    mkdir -p "$WORK/base"
    base_map
    "$INGOT" st-ls "$1" | cut -f1 | while read -r name; do
        local shard
        shard=$(awk -F'\t' -v n="$name" '$1 == n || $1 == "*" {print $2; exit}' "$WORK/base_map.tsv")
        [ -n "$shard" ] || continue
        grep -F "$name	" "$(base_header "$shard")" | awk -F'\t' -v n="$name" '$1 == n' |
        while IFS=$'\t' read -r _ dtype shape begin end; do
            [ "$dtype" = BF16 ] || continue
            local dst="$WORK/base/$(echo "$name" | tr '/' '_').safetensors" nbytes=$((end - begin))
            need_disk "$nbytes"
            "$INGOT" st-wrap "$name" "$dtype" "$shape" "$nbytes" > "$dst.part"
            "${CURL[@]}" -r "$begin-$((end - 1))" "$BASE_URL/$shard" >> "$dst.part"
            [ "$(file_size "$dst.part")" -gt "$nbytes" ] || { echo "short base read $name" >&2; exit 4; }
            mv "$dst.part" "$dst"
        done
    done
}

# --- one shard ---------------------------------------------------------------------------------
convert_shard() {  # shard path, expected sha256 (or "")
    local shard=$1 sha=$2 stem dst base_args=("${BASES[@]+"${BASES[@]}"}")
    stem=$(basename "$shard" .safetensors)
    dst="$OUT/$(basename "$shard").ingot"   # model-00001-of-00018.safetensors.ingot
    if [ -n "$BASE_REPO" ]; then
        fetch_base_for "$shard"
        base_args+=(--base "$WORK/base")
    fi
    nice -n 10 "$INGOT" encode -j "$THREADS" "${base_args[@]+"${base_args[@]}"}" "$shard" "$dst.part" \
        | tee -a "$OUT/encode.log"
    mv "$dst.part" "$dst"
    nice -n 10 "$INGOT" verify -j "$THREADS" "${base_args[@]+"${base_args[@]}"}" "$dst" "$shard" \
        | tee -a "$OUT/verify.log"
    local rebuilt="skipped"
    if [ "$REBUILD" = 1 ]; then
        [ -n "$sha" ] || sha=$(sha256_of "$shard")
        need_disk "$(file_size "$shard")"
        nice -n 10 "$INGOT" decode -j "$THREADS" "${base_args[@]+"${base_args[@]}"}" "$dst" "$WORK/rebuilt.safetensors"
        rebuilt=$(sha256_of "$WORK/rebuilt.safetensors")
        rm -f "$WORK/rebuilt.safetensors"
        [ "$rebuilt" = "$sha" ] || { echo "REBUILD MISMATCH $stem: $rebuilt != $sha" >&2; exit 5; }
        rebuilt="sha256-ok"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$stem" "$(file_size "$shard")" "$(file_size "$dst")" verified \
        "$rebuilt" >> "$LOG"
    rm -rf "$WORK/base"
    [ "$DISCARD" = 1 ] && rm -f "$dst"
    return 0
}

# One file as $PARTS parallel Range requests (one connection is often throttled well below the
# link), each resumable; parts are appended in order and deleted as they go.
PARTS=${INGOT_DOWNLOAD_PARTS:-4}
download_parts() {  # url, size, dst
    local url=$1 size=$2 dst=$3 chunk=$(( ($2 + PARTS - 1) / PARTS )) k pids=()
    [ -f "$dst" ] && [ "$(file_size "$dst")" -eq "$size" ] && return 0  # resumed run
    for ((k = 0; k < PARTS; k++)); do
        local lo=$((k * chunk)) hi=$(( (k + 1) * chunk - 1 ))
        [ "$hi" -lt "$size" ] || hi=$((size - 1))
        [ "$lo" -le "$hi" ] || continue
        local part="$dst.p$k" want=$((hi - lo + 1)) have=0
        [ -f "$part" ] && have=$(file_size "$part")
        if [ "$have" -lt "$want" ]; then
            "${CURL[@]}" -r "$((lo + have))-$hi" "$url" >> "$part" &
            pids+=($!)
        fi
    done
    for p in "${pids[@]+"${pids[@]}"}"; do wait "$p" || { echo "download failed: $url" >&2; exit 4; }; done
    : > "$dst"
    for ((k = 0; k < PARTS; k++)); do
        [ -f "$dst.p$k" ] || continue
        cat "$dst.p$k" >> "$dst"
        rm -f "$dst.p$k"
    done
}

done_already() { [ -s "$LOG" ] && grep -q "^$1	.*	verified	" "$LOG"; }

if [ -d "$SRC" ]; then
    for shard in "$SRC"/*.safetensors; do
        stem=$(basename "$shard" .safetensors)
        done_already "$stem" && continue
        convert_shard "$shard" ""
    done
else
    API="https://huggingface.co/api/models/$SRC/revision/$REV?blobs=true"
    "${CURL[@]}" "$API" -o "$WORK/siblings.json"
    jq -r '.siblings[] | select(.rfilename|endswith(".safetensors")) |
           "\(.rfilename)\t\(.size)\t\(.lfs.sha256)"' "$WORK/siblings.json" > "$WORK/shards.tsv"
    jq -r '.siblings[] | select((.rfilename|endswith(".safetensors")|not) and
           ((.size // 0) < 67108864) and (.rfilename|contains("/")|not)) | .rfilename' \
        "$WORK/siblings.json" | while read -r f; do
        [ -s "$OUT/$f" ] || "${CURL[@]}" "https://huggingface.co/$SRC/resolve/$REV/$f" -o "$OUT/$f"
    done
    while IFS=$'\t' read -r name size sha; do
        stem=$(basename "$name" .safetensors)
        done_already "$stem" && continue
        need_disk "$size"
        local_path="$WORK/$(basename "$name")"
        download_parts "https://huggingface.co/$SRC/resolve/$REV/$name" "$size" "$local_path"
        [ "$(file_size "$local_path")" -eq "$size" ] || { echo "size mismatch $name" >&2; exit 4; }
        [ "$(sha256_of "$local_path")" = "$sha" ] || { echo "sha256 mismatch $name" >&2; rm -f "$local_path"; exit 5; }
        echo "$(date +%T) downloaded $name"
        convert_shard "$local_path" "$sha"
        [ "$KEEP" = 1 ] || rm -f "$local_path"
    done < "$WORK/shards.tsv"
fi
awk -F'\t' '{s += $2; o += $3} END {printf "total\tsource %d\tingot %d\tratio %.4f\n", s, o, o / s}' \
    "$LOG" | tee "$OUT/DONE"
