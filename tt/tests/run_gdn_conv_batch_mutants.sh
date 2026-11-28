#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_gdn_conv_batch_mutants.sh
#   Mutation reds for test_gdn_conv_batch.c (the CONV row batch): break one piece at a time —
#   the kernels' field and predicate (kernels/resident_gdn_protocol.h), the host mirrors
#   (ft_tt_op_gdn.h) or the host encoder (ft_tt_op_gdn.c) — rebuild test_gdn_conv_batch
#   (binary deleted first) and require it RED, naming the check that caught it.
#   Every mutated file is restored byte for byte (cmp) on exit; then the clean build must be
#   GREEN again.
#
#   bash tt/tests/run_gdn_conv_batch_mutants.sh
#
#   Output: tt/tests/build-gdn/mutants_batch/ (per-mutant logs, summary.txt). Exit 0 = every mutant
#   red and the clean build green; 1 otherwise; 2 setup error.
#
#   Status: added with the CONV row batch.
#   Issues: None
#   Todo:   None
#
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
OUT=$HERE/build-gdn/mutants_batch
TEST=$HERE/build-gdn/test_gdn_conv_batch
SUMMARY=$OUT/summary.txt
SAVED=() # "original|copy" pairs, restored on exit whatever happens
red=0

mkdir -p "$OUT/orig"
: > "$SUMMARY"

restore_all() {
    local pair
    for pair in "${SAVED[@]+"${SAVED[@]}"}"; do
        cp -p "${pair#*|}" "${pair%%|*}"
    done
}
trap restore_all EXIT

build_and_run() {
    rm -f "$TEST"
    make -C "$HERE" -f Makefile.gdn build-gdn/test_gdn_conv_batch > "$1.build" 2>&1 || return 3
    (cd "$HERE" && "$TEST") > "$1" 2>&1 # its kernel-source checks read relative paths
}

# mutant <name> <file> <the check expected to catch it> <perl substitution, applied -0pi>
mutant() {
    local name=$1 file=$2 expect=$3 expr=$4 saved rc caught
    saved=$OUT/orig/$name
    cp -p "$file" "$saved"
    SAVED+=("$file|$saved")
    perl -0pi -e "$expr" "$file"
    if cmp -s "$file" "$saved"; then
        printf 'RED-SCRIPT %-26s the mutation did not apply\n' "$name" | tee -a "$SUMMARY"
        red=1
        return
    fi
    build_and_run "$OUT/$name.log"
    rc=$?
    cp -p "$saved" "$file"
    cmp -s "$file" "$saved" || { echo "restore of $file failed" >&2; exit 2; }
    caught=$(grep -c '^  FAIL' "$OUT/$name.log" 2>/dev/null)
    if [ "$rc" = 3 ]; then
        printf 'GREEN %-26s the mutant does not compile (%s.build)\n' "$name" "$OUT/$name.log" | tee -a "$SUMMARY"
        red=1
    elif [ "$rc" != 0 ] && grep -qF "$expect" <(grep '^  FAIL' "$OUT/$name.log"); then
        printf 'RED   %-26s %s FAIL line(s), incl. "%s"\n' "$name" "$caught" "$expect" | tee -a "$SUMMARY"
    else
        printf 'GREEN %-26s rc=%s — the mutant SURVIVED (expected "%s")\n' "$name" "$rc" "$expect" | tee -a "$SUMMARY"
        red=1
    fi
}

P=$TT/kernels/resident_gdn_protocol.h
H=$TT/ft_tt_op_gdn.h
E=$TT/ft_tt_op_gdn.c
mutant shift_moved "$P" "the batch shift agrees" \
    's/enum \{ kGdnConvBatchShift = 10 \};/enum { kGdnConvBatchShift = 9 };/'
mutant limit_past_face1 "$H" "rows [13, 17) past face 1 refused" \
    's/#define FT_TT_OP_GDN_CONV_BATCH_ROW_LIMIT 16u/#define FT_TT_OP_GDN_CONV_BATCH_ROW_LIMIT 32u/'
mutant host_bf16_allowed "$E" "a batch over the G2a bf16 stream refused" \
    's/ \|\|\n\s+operand_is_set\(&request->stream_bf16\)\)\)\n\s+return 0;/))\n        return 0;/'
mutant host_one_row_allowed "$E" "a one-row batch refused" \
    's/request->row_count < 2u \|\|\n(\s+)request->row \+ request->row_count > FT_TT_OP_GDN_CONV_BATCH_ROW_LIMIT/request->row_count < 1u ||\n$1request->row + request->row_count > FT_TT_OP_GDN_CONV_BATCH_ROW_LIMIT/'
mutant host_rows_as_count "$E" "the compute sees ONE row" \
    's/request->row_count > 1u && request->conv_row_batch == 0u \?/request->row_count > 1u ?/'
mutant host_pass_per_row "$E" "the writer is told ONE pass of pages" \
    's/if \(request->conv_row_batch != 0u \|\| request->row_count < 2u\)/if (request->row_count < 2u)/'
mutant device_bf16_allowed "$P" "the bf16 input flag beside a batch refused on the device" \
    's/\(flags & \(uint32_t\)kGdnFlagConvBf16In\) == 0u &&\n//'
mutant device_count_allowed "$P" "a batch beside a multi-row count refused on the device" \
    's/shape->row_count == 1u &&\n\s+//'
mutant device_strip_unchecked "$P" "a batch without the strip flag refused on the device" \
    's/\(flags & \(uint32_t\)kGdnFlagConvStrip\) != 0u &&\n\s+//'
mutant device_any_phase "$P" "a DELTA descriptor with batch bits refused on the device" \
    's/return shape->phase == \(uint32_t\)kGdnPhaseConv && shape->row_count == 1u/return shape->row_count == 1u/'

restore_all
trap - EXIT
build_and_run "$OUT/clean.log"
rc=$?
printf 'clean rebuild: rc=%s, %s\n' "$rc" "$(grep 'checks failed' "$OUT/clean.log")" | tee -a "$SUMMARY"
[ "$rc" = 0 ] || red=1
[ "$red" = 0 ] && echo "ALL MUTANTS RED, CLEAN GREEN" | tee -a "$SUMMARY"
exit "$red"
