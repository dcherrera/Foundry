#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Build and run the k-quant reference generator / cross-check.
#
# This is a ONE-TIME offline step. Its product — ft_gguf_kquant_reference_vectors.h —
# lives in the tree so that tests/test_convert.c needs no ggml checkout. Every block it
# freezes is synthetic (seeded; see generate_kquant_reference.c); no checkpoint is read.
#
# What it links, and why each one:
#   libggml-base       upstream ggml's OWN compiled dequantize_row_qN_K and
#                      quantize_row_qN_K_ref. The oracle. Not a transcription of
#                      ggml — it is ggml.
#   foundry convert    the copy under test.
#   Pantheon formats   OPTIONAL third copy, cross-checked when PANTHEON_DIR is set.
#
# Usage (from anywhere):
#   reference/generate_kquant_reference.sh <llama.cpp checkout>
#   LLAMA_CPP=<llama.cpp checkout> reference/generate_kquant_reference.sh
#   PANTHEON_DIR=<Pantheon checkout> ... also cross-checks Pantheon's reader
#
# The llama.cpp checkout must already be built (build/bin/libggml-base.{dylib,so}).

set -e

here=$(cd "$(dirname "$0")" && pwd)
convert="$here/.."
llama=${1:-${LLAMA_CPP:-}}
[ -n "$llama" ] || { echo "usage: $0 <llama.cpp checkout>   (or set LLAMA_CPP)" >&2; exit 2; }

ggml_lib="$llama/build/bin"
out="$here/ft_gguf_kquant_reference_vectors.h"
bin="$here/build/generate_kquant_reference"

[ -e "$ggml_lib/libggml-base.dylib" ] || [ -e "$ggml_lib/libggml-base.so" ] || {
    echo "missing: $ggml_lib/libggml-base.{dylib,so} (build llama.cpp first)" >&2; exit 1; }

pantheon_flags=""
pantheon_sources=""
if [ -n "${PANTHEON_DIR:-}" ]; then
    [ -e "$PANTHEON_DIR/src/formats/gguf_dequant.c" ] || {
        echo "missing: $PANTHEON_DIR/src/formats/gguf_dequant.c" >&2; exit 1; }
    pantheon_flags="-DPANTHEON -I$PANTHEON_DIR/src/formats -I$PANTHEON_DIR/include"
    pantheon_sources="$PANTHEON_DIR/src/formats/gguf_dequant.c $PANTHEON_DIR/src/formats/gguf.c"
fi

mkdir -p "$here/build"

# -O1 and no fast-math: the point is to reproduce ggml's fp32 words exactly, so
# nothing here may reassociate `scale*q - min`.
# shellcheck disable=SC2086 # the optional Pantheon flags/sources are word lists
cc -std=c99 -O1 -g -Wall -Wextra \
   -I"$convert" -I"$convert/.." $pantheon_flags \
   "$here/generate_kquant_reference.c" \
   "$convert/ft_gguf_dequant.c" \
   "$convert/ft_gguf_map.c" \
   "$convert/ft_float_convert.c" \
   $pantheon_sources \
   -L"$ggml_lib" -lggml-base -Wl,-rpath,"$ggml_lib" \
   -lm -o "$bin"

"$bin" "$out"
