#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# make_attention_golden_words.sh
#   Encode attention_golden_requests.inc with the attention encoder of git ref <ref> (default
#   the merge base with origin/main) and print the words as the C table
#   test_attention_splitk.c pins (k_golden_words). Run from tt/tests; writes only under
#   build-attn-golden/ (gitignored).
#
#   Usage:  bash make_attention_golden_words.sh [ref]
#   Status: added 2026-11-16 with split-K (Qwen3.8 TT usable plan 3.1).
#   Issues: None
#   Todo:   None
#
set -euo pipefail
cd "$(dirname "$0")"
REF=${1:-$(git merge-base HEAD origin/main)}
OUT=build-attn-golden
rm -rf "$OUT" && mkdir -p "$OUT"
git show "$REF:tt/ft_tt_op_attention.h" > "$OUT/ft_tt_op_attention.h"
git show "$REF:tt/ft_tt_op_attention.c" > "$OUT/ft_tt_op_attention.c"
cp print_attention_golden_words.c attention_golden_requests.inc "$OUT/"
# The rest of the off-hardware link is Makefile.attention's (the fenced backend).
BACKEND="kmd arch arch_blackhole device tlb arc_telemetry coord noc buffer elf kernel_blob
         udrv_claim tensix_claim launch tilize op_dispatch op_memory queue group watchdog"
SOURCES=()
for unit in $BACKEND; do SOURCES+=("../ft_tt_$unit.c"); done
cc -std=c99 -O1 -I"$OUT" -I.. -o "$OUT/print_golden" "$OUT/print_attention_golden_words.c" \
    "$OUT/ft_tt_op_attention.c" "${SOURCES[@]}" ../convert/ft_float_convert.c -lm
echo "/* encoded by $REF ($(git rev-parse --short "$REF")) */"
"$OUT/print_golden"
