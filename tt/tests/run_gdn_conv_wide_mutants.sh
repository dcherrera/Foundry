#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_gdn_conv_wide_mutants.sh
#   Mutation reds for test_gdn_ops.c's wide CONV column slice checks (test_wire_constants,
#   test_conv_wide_column_slice): break one piece of the wide encoding at a time — the kernels'
#   decode (kernels/resident_gdn_protocol.h) or the host encoder (ft_tt_op_gdn.c) — rebuild
#   test_gdn_ops (binary deleted first) and require it RED, naming the check that caught it.
#   Every mutated file is restored byte for byte (cmp) on exit; then the clean build must be
#   GREEN again.
#
#   bash tt/tests/run_gdn_conv_wide_mutants.sh
#
#   Output: tt/tests/build-gdn/mutants/ (per-mutant logs, summary.txt). Exit 0 = every mutant
#   red and the clean build green; 1 otherwise; 2 setup error.
#
#   Status: added with the wide CONV column slice; granule mutants with the slice granule.
#   Issues: None
#   Todo:   None
#
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
OUT=$HERE/build-gdn/mutants
TEST=$HERE/build-gdn/test_gdn_ops
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
    make -C "$HERE" -f Makefile.gdn build-gdn/test_gdn_ops > "$1.build" 2>&1 || return 3
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
E=$TT/ft_tt_op_gdn.c
mutant mask_pin_moved "$P" "kGdnConvTileColsMask" \
    's/enum \{ kGdnConvTileColsMask = 0xFFFF \};/enum { kGdnConvTileColsMask = 0xFFFFF };/'
mutant decode_first_high_lost "$P" "round-trips through ftg_decode" \
    's/shape\.col_first = shape\.head_first \|\s+\(ftg_byte\(parameters\[kGdnConvTileCols\], kGdnByteColFirstHigh\) << 8\);/shape.col_first = shape.head_first;/'
mutant decode_count_from_first "$P" "partition the 320 columns" \
    's/kGdnByteColCountHigh\) << 8\)/kGdnByteColFirstHigh) << 8)/'
mutant decode_count_high_lost "$P" "an explicit [0, 320)" \
    's/const uint32_t count = shape\.head_count \|\s+\(ftg_byte\(parameters\[kGdnConvTileCols\], kGdnByteColCountHigh\) << 8\);/const uint32_t count = shape.head_count;/'
mutant decode_count_unguarded "$P" "unsliced one is still the whole stream" \
    's/if \(count != 0u\)\n\s+shape\.col_count = count;/shape.col_count = count;/'
mutant decode_cols_unmasked "$P" "round-trips through ftg_decode" \
    's/parameters\[kGdnConvTileCols\] & \(uint32_t\)kGdnConvTileColsMask;/parameters[kGdnConvTileCols];/'
mutant encode_first_high_lost "$E" "[300, 320)" \
    's/\(\(\(request->col_first >> 8\) & 0xFFu\) << 16\)/0u/'
mutant encode_narrow_moved "$E" "every slice under column 256" \
    's/\(\(\(request->col_first >> 8\) & 0xFFu\) << 16\)/((request->col_first & 0xFFu) << 16)/'
mutant encode_stream_cap_lost "$E" "wider than the tile-column half-word" \
    's/if \(request->conv_dim \/ FT_TT_OP_TILE_DIM > FT_TT_OP_GDN_COL_MAX\)\n\s+return 0;\n//'
mutant encode_bounds_lost "$E" "a wide slice that runs off the stream" \
    's/if \(request->col_count != 0u &&\n\s+request->col_first \+ request->col_count > request->conv_dim \/ FT_TT_OP_TILE_DIM\)\n\s+return 0;\n//'
mutant encode_word1_every_phase "$E" "still leaves the tile-column word zero" \
    's/request->phase == FT_TT_OP_GDN_PHASE_CONV\n(\s+)\? \(request->conv_dim/1\n$1? (request->conv_dim/'

# The slice granule (FT_TT_OP_GDN_CONV_SLICE_GRANULE): the 18-column slice that broke DELTA.
mutant granule_guard_lost "$E" "[300, 318) is REFUSED" \
    's/request->col_count != 0u && !conv_slice_is_granular\(request\)/request->col_count != 0u \&\& !conv_slice_is_granular(request) \&\& 0/'
mutant granule_is_depth "$E" "the granule at S 128" \
    's/return gdn_lcm\(depth, gdn_lcm\(window_pages, state_tiles\) \/ window_pages\);/return gdn_lcm(depth, (window_pages \/ window_pages) * (state_tiles \/ state_tiles));/'
mutant granule_window_count "$E" "admitted == on grid" \
    's/const uint32_t window_pages = 2u \* conv_taps - 3u;/const uint32_t window_pages = 2u * conv_taps - 2u;/'
mutant granule_reads_first "$E" "[300, 318) is REFUSED" \
    's/request->col_count % granule == 0u/request->col_first % granule == 0u/'

restore_all
trap - EXIT
build_and_run "$OUT/clean.log"
rc=$?
printf 'clean rebuild: rc=%s, %s\n' "$rc" "$(grep 'checks failed' "$OUT/clean.log")" | tee -a "$SUMMARY"
[ "$rc" = 0 ] || red=1
[ "$red" = 0 ] && echo "ALL MUTANTS RED, CLEAN GREEN" | tee -a "$SUMMARY"
exit "$red"
