#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_gdnc_table_mutants.sh
#   Mutation reds for the exact card gates: break one piece at a time — the table front ends and
#   builder (kernels/ftgc_table.h) or the host encoders (ft_tt_op_gdnc.c, ft_tt_op_gdn.c) —
#   rebuild test_gdnc_table, test_gdnc_ops and test_gdn_ops (binaries deleted first) and
#   require them RED, naming the
#   check that caught it. Every mutated file is restored byte for byte (cmp) on exit; then the
#   clean build must be GREEN again.
#
#   bash tt/tests/run_gdnc_table_mutants.sh
#
#   Output: tt/tests/build-gdnc/mutants/ (per-mutant logs, summary.txt). Exit 0 = every mutant
#   red and the clean build green; 1 otherwise; 2 setup error.
#
#   Status: added with the exact table gates.
#   Issues: None
#   Todo:   None
#
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
OUT=$HERE/build-gdnc/mutants
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
    local a b c
    rm -f "$HERE/build-gdnc/test_gdnc_table" "$HERE/build-gdnc/test_gdnc_ops" \
        "$HERE/build-gdn/test_gdn_ops"
    make -C "$HERE" -f Makefile.gdnc all > "$1.build" 2>&1 || return 3
    make -C "$HERE" -f Makefile.gdn all >> "$1.build" 2>&1 || return 3
    (cd "$HERE" && ./build-gdnc/test_gdnc_table) > "$1" 2>&1
    a=$?
    (cd "$HERE" && ./build-gdnc/test_gdnc_ops) >> "$1" 2>&1 # reads kernel sources relatively
    b=$?
    (cd "$HERE" && ./build-gdn/test_gdn_ops) >> "$1" 2>&1
    c=$?
    [ "$a" = 0 ] && [ "$b" = 0 ] && [ "$c" = 0 ]
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

T=$TT/kernels/ftgc_table.h
E=$TT/ft_tt_op_gdnc.c
mutant resolve_low_is_high "$T" "decay table builds and proves" \
    's/\*value = header\[kFtgtLowPos \+ 2u \* side\];/*value = header[kFtgtHighPos + 2u * side];/'
mutant resolve_neg_start_lost "$T" "decay table builds and proves" \
    's/return \(side \? header\[kFtgtNegStart\] : 0u\) \+ \(m - lo\);/return (m - lo);/'
mutant builder_empty_window "$T" "a constant table is its header" \
    's/    if \(first > end\)\n        first = end;\n//'
mutant face_cache_never_restaged "$T" "every head's compact run is the host's" \
    's/if \(state->staged != key\) \{/if (state->staged == 0u) {/'
mutant band_off_by_one "$T" "every head's compact run is the host's" \
    's/const uint32_t band_offset = band \* block\[kFtgtBlockBandBytes\];/const uint32_t band_offset = (band + 1u) * block[kFtgtBlockBandBytes];/'
mutant element_row_column_swapped "$T" "every head's compact run is the host's" \
    's/\(strip_row % 16u\) \* 16u \+ \(column % 16u\)/(column % 16u) * 16u + (strip_row % 16u)/'
mutant line_not_64_aligned "$T" "every DRAM read is 64 B aligned" \
    's/address & ~63u,/address & ~15u,/'
mutant scale_word_lost "$T" "every head's compact run is the host's" \
    's/compact\[0\] = block\[kFtgtBlockScale\];/compact[0] = 0u;/'
mutant stale_run_not_zeroed "$T" "every head's compact run is the host's" \
    's/compact\[word\] = 0u;/(void)compact;/'
mutant lower_face_skipped "$T" "every head's compact run is the host's" \
    's/\(rows == 1u && first_row \+ chunk_rows <= 16u\)/(rows == 1u)/'
mutant encode_table_flag_lost "$E" "a table DELTA descriptor encodes its flag and band" \
    's/\(FT_TT_OP_GDNC_DELTA_FLAG_COMPACT \|\n\s+FT_TT_OP_GDNC_DELTA_FLAG_TABLE\)/FT_TT_OP_GDNC_DELTA_FLAG_COMPACT/'
mutant encode_band_lost "$E" "a table DELTA descriptor encodes its flag and band" \
    's/\? request->gate_band : 0u;/? 0u : 0u;/'
mutant encode_band_unguarded "$E" "a band without the table flag" \
    's/request->gate_band != 0u\)\n\s+return 0;/0)\n            return 0;/'
mutant stripe_lost_in_head "$T" "every head's compact run is the host's" \
    's/const uint32_t stripe = kFtgtBlockStripe \+ 4u \* \(head % stripes\);/const uint32_t stripe = kFtgtBlockStripe;/'
mutant stripe_slot_index_lost "$T" "every head's compact run is the host's" \
    's/block\[stripe \+ kFtgtStripeDecayBase\] \+ \(head \/ stripes\) \* slot;/block[stripe + kFtgtStripeDecayBase] + head * slot;/'
mutant block_stripes_unchecked "$T" "a block with 9 stripes refuses" \
    's/stripes > kFtgtMaxStripes \|\|//'

G=$TT/ft_tt_op_gdn.c
mutant rows_face_key_wrong "$T" "every member's gate words are the host's" \
    's/\(\(head \/ 16u\) \* 2u \+ row \/ 16u\) \* 1024u/((head \/ 32u) * 2u + row \/ 16u) * 1024u/'
mutant rows_face_half_lost "$T" "every member's gate words are the host's" \
    's/\(group \* 2u \+ key % 2u\) \* FTGT_FACE_BYTES/(group * 2u) * FTGT_FACE_BYTES/'
mutant rows_beta_offset_lost "$T" "every member's gate words are the host's" \
    's/const uint32_t dest = r \* \(stride \/ 4u\) \+ which \* v_heads \+ head;/const uint32_t dest = r * (stride \/ 4u) + which * 0u * v_heads + head;/'
mutant rows_stride_lost "$T" "every member's gate words are the host's" \
    's/const uint32_t dest = r \* \(stride \/ 4u\) \+/const uint32_t dest = r * 0u * (stride \/ 4u) +/'
mutant rows_tail_flush_dropped "$T" "every member's gate words are the host's" \
    's/\(pending == FTGT_ROWS_SLOT_COUNT \|\| n == 1u\)/(pending == FTGT_ROWS_SLOT_COUNT)/'
mutant rows_heads_unbounded "$T" "more heads than the header area refuse" \
    's/head_count > FTGT_ROWS_MAX_HEADS \|\| //'
mutant encode_gdn_table_lost "$G" "a DELTA descriptor encodes the flag" \
    's/FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0 \| FT_TT_OP_GDN_FLAG_TABLE\)/FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0)/'
mutant encode_gdn_table_conv "$G" "...and a CONV descriptor with it refuses" \
    's/if \(request->phase == FT_TT_OP_GDN_PHASE_CONV &&\n\s+\(request->flags & FT_TT_OP_GDN_FLAG_TABLE\) != 0u\)/if (0)/'

restore_all
trap - EXIT
build_and_run "$OUT/clean.log"
rc=$?
printf 'clean rebuild: rc=%s, %s\n' "$rc" "$(grep 'checks failed' "$OUT/clean.log")" | tee -a "$SUMMARY"
[ "$rc" = 0 ] || red=1
[ "$red" = 0 ] && echo "ALL MUTANTS RED, CLEAN GREEN" | tee -a "$SUMMARY"
exit "$red"
