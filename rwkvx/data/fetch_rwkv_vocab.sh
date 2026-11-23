#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# fetch_rwkv_vocab.sh — fetch the RWKV World tokenizer vocabulary at build time.
#
# The vocab is BlinkDL's (ChatRWKV, Apache-2.0), so foundry does not commit it.
# This downloads it from a PINNED commit into rwkvx/data/ (gitignored) and
# verifies its sha256. A missing download or a wrong checksum is a hard error,
# and a file with the wrong checksum is deleted, never used.
#
#   Usage: rwkvx/data/fetch_rwkv_vocab.sh          fetch if missing, then verify
#          rwkvx/data/fetch_rwkv_vocab.sh --check  verify only (no network)
#
# Status: added 2026-10-08 (the vocab was removed from the tree).
# Issues: None
# Todo:   None
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
OUT="$HERE/rwkv_vocab_v20230424.txt"
URL="https://raw.githubusercontent.com/BlinkDL/ChatRWKV/02058ba0624a77c20f0913f83550835eb03a8db4/tokenizer/rwkv_vocab_v20230424.txt"
SHA256="e6dee3d4e31b4d5c40ac99508ac6c701ceef4bed681bf2167ce9a908552bca89"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

if [ ! -f "$OUT" ]; then
    if [ "${1:-}" = --check ]; then
        echo "fetch_rwkv_vocab: $OUT is missing (run without --check to fetch it)" >&2
        exit 1
    fi
    echo "fetch_rwkv_vocab: downloading $URL" >&2
    curl -fsSL -o "$OUT.part" "$URL" || { rm -f "$OUT.part"; echo "fetch_rwkv_vocab: download FAILED" >&2; exit 1; }
    mv "$OUT.part" "$OUT"
fi

got=$(sha256_of "$OUT")
if [ "$got" != "$SHA256" ]; then
    rm -f "$OUT"
    echo "fetch_rwkv_vocab: CHECKSUM MISMATCH for $OUT (got $got, want $SHA256); deleted" >&2
    exit 1
fi
echo "fetch_rwkv_vocab: $OUT ok (sha256 $SHA256)"
