#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# ingot_fetch.sh — write a remote file to stdout, in order, as a series of range requests (each
# retried), so a long download survives dropped connections without storing the file:
#   ingot_fetch.sh URL | ingot encode - model.gguf.ingot
#   ingot_fetch.sh ssh:HOST:/path/model.safetensors | ingot encode - out.ingot
# An ssh: source reads byte ranges with tail/head on HOST over one multiplexed connection.
# Env: INGOT_FETCH_CHUNK (bytes per request, default 256 MiB), HF_TOKEN (gated repos).
set -uo pipefail
[ $# -eq 1 ] || { echo "usage: ingot_fetch.sh URL | ssh:HOST:PATH" >&2; exit 2; }
URL=$1
CHUNK=${INGOT_FETCH_CHUNK:-268435456}
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")

if [[ $URL == ssh:* ]]; then
    rest=${URL#ssh:}
    HOST=${rest%%:*}
    RPATH=$(printf '%q' "${rest#*:}")
    mkdir -p "$HOME/.ssh/cm"
    SSH=(ssh -n -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ControlMaster=auto
         -o "ControlPath=$HOME/.ssh/cm/%r@%h:%p" -o ControlPersist=120 "$HOST")
    size_of() { "${SSH[@]}" "stat -c %s $RPATH" 2>/dev/null | tr -d '\r'; }
    range() { "${SSH[@]}" "tail -c +$(($1 + 1)) $RPATH | head -c $(($2 - $1 + 1))"; }
else
    size_of() {
        curl -sIL "${AUTH[@]+"${AUTH[@]}"}" "$URL" | tr -d '\r' |
            awk 'tolower($1) == "content-length:" {n = $2} END {print n}'
    }
    range() {
        curl -sfL --connect-timeout 30 --speed-time 60 --speed-limit 1024 \
            "${AUTH[@]+"${AUTH[@]}"}" -r "$1-$2" "$URL"
    }
fi

SIZE=""
for try in 1 2 3 4 5; do
    SIZE=$(size_of)
    [ -n "$SIZE" ] && [ "$SIZE" -gt 0 ] && break
    sleep $((try * 5))
done
[ -n "$SIZE" ] && [ "$SIZE" -gt 0 ] || { echo "ingot_fetch: no size for $URL" >&2; exit 1; }
exec 3>&1 # the real stdout; tee writes the bytes there and wc counts them
at=0
while [ "$at" -lt "$SIZE" ]; do
    end=$((at + CHUNK - 1))
    [ "$end" -ge "$SIZE" ] && end=$((SIZE - 1))
    want=$((end - at + 1))
    ok=0
    for try in 1 2 3 4 5 6 7 8; do
        # count the bytes as they pass; a short chunk is retried from where it stopped
        got=$(range "$at" "$end" | tee /dev/fd/3 | wc -c | tr -d " ")
        at=$((at + got))
        want=$((want - got))
        [ "$want" -eq 0 ] && { ok=1; break; }
        echo "ingot_fetch: retry $try at byte $at" >&2
        sleep $((try * 5))
    done
    [ "$ok" -eq 1 ] || { echo "ingot_fetch: giving up at byte $at of $SIZE" >&2; exit 1; }
done
