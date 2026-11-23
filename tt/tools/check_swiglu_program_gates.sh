#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_swiglu_program_gates.sh
#   The compile-side gates of `rtriad_bfp8_swiglu`, the fused-SwiGLU resident matmul
#   (kernels/resident_matmul_swiglu_protocol.h).
#   Needs the SFPI toolchain (7.68.0, the blob's), no card:
#
#     1. explode the committed blob (ft_tt_programs.h) into $WORK/blob: the shipped images
#     2. rebuild EVERY shipped program from its own provenance recipe and THIS tree's sources
#        into ../programs, and require every image LOAD-IDENTICAL to the shipped one: the fused
#        code is all under -DFT_RTRIAD_SWIGLU, so no shipped program may move
#     3. build rtriad_bfp8_swiglu (rtriad_bfp8's recipe plus -DFT_RTRIAD_SWIGLU=1 and the
#        matmul_bfp8_b_swiglu CB table) into ../programs; every kernel .text under the
#        12,288 B slab and DIFFERENT from rtriad_bfp8's (else the define never reached the
#        compiler), every firmware image IDENTICAL to rtriad_bfp8's (Pantheon boots
#        rtriad_bfp8's firmware family: pn_tt_firmware_matches)
#     4. regenerate the blob from ../programs: CHECK=1 (diff only) unless REGEN=1, which
#        writes ../ft_tt_programs.h and then runs the provenance check and the verifier
#
#   Usage (from tt/):  bash tools/check_swiglu_program_gates.sh
#   Env:  SFPI   toolchain root (default /opt/tenstorrent/sfpi)
#         WORK   scratch (default ../build/swiglu-gates, gitignored)
#         REGEN  1 = write the regenerated ft_tt_programs.h
#   Any host with SFPI 7.68.0 built for it, bash >= 4, python3, make and cc. On macOS:
#     export SFPI=<install root> PATH=/opt/homebrew/bin:$PATH
#
#   Output: $WORK/gates.txt (the verdicts), $WORK/rebuild/ (per-program logs and summary)
#   Exit:   0 every gate green; 1 any gate red; 2 setup error
#
#   Status: added 2026-11-17 with rtriad_bfp8_swiglu (plan 3.3).
#   Issues: None
#   Todo:   None
#
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
WORK=${WORK:-$TT/build/swiglu-gates}
REGEN=${REGEN:-0}
PROGRAMS=$TT/programs
K=$TT/kernels
NEW=rtriad_bfp8_swiglu
BASE=rtriad_bfp8
SLAB_BYTES=12288
GATES=$WORK/gates.txt
red=0

say() { printf '%s\n' "$*" | tee -a "$GATES"; }
fail() { say "RED: $*"; red=1; }
text_bytes() { "$SFPI/compiler/bin/riscv-tt-elf-size" -A "$1" | awk '$1 == ".text" {print $2}'; }
same_load() { python3 "$HERE/compare_tensix_load_images.py" "$1" "$2" | cut -d: -f1; }

[ -x "$SFPI/compiler/bin/riscv-tt-elf-gcc" ] || { echo "no SFPI compiler under $SFPI" >&2; exit 2; }
export SFPI
mkdir -p "$WORK"
: > "$GATES"
say "=== $NEW program gates  $(date -u +%FT%TZ)  sfpi $("$SFPI/compiler/bin/riscv-tt-elf-gcc" -dumpversion)"

# 1. The shipped images, out of the committed blob.
make -C "$HERE" explode-programs > "$WORK/explode_build.log" 2>&1 || { echo "explode build failed" >&2; exit 2; }
rm -rf "$WORK/blob"
"$HERE/build/explode_tensix_program_blob" "$WORK/blob" --verify > "$WORK/explode.log" 2>&1 ||
    { echo "explode failed (see $WORK/explode.log)" >&2; exit 2; }
say "exploded $(ls "$WORK/blob" | wc -l | tr -d ' ') programs from ft_tt_programs.h"

# 2. Every shipped program from this tree's sources, LOAD-IDENTICAL to the shipped images.
rm -rf "$PROGRAMS" "$WORK/rebuild"
bash "$HERE/rebuild_all_tensix_programs.sh" "$WORK/blob" "$PROGRAMS" $(ls "$WORK/blob" | grep -v "^$NEW\$") > /dev/null 2>&1
mkdir -p "$WORK/rebuild"
mv "$PROGRAMS"/*.rebuild.log "$PROGRAMS"/*.diff "$PROGRAMS"/summary.txt "$WORK/rebuild/" 2>/dev/null
rm -f "$PROGRAMS/DONE"
images=$(awk 'NF >= 4 && $2 ~ /\.elf$/' "$WORK/rebuild/summary.txt" | wc -l | tr -d ' ')
identical=$(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 == "LOAD-IDENTICAL"' "$WORK/rebuild/summary.txt" | wc -l | tr -d ' ')
say "LOAD-IDENTICAL: $identical of $images images (every shipped program, kernels + firmware)"
grep -q "REBUILD FAILED" "$WORK/rebuild/summary.txt" && fail "$(grep -c 'REBUILD FAILED' "$WORK/rebuild/summary.txt") program(s) failed to rebuild"
[ "$images" -gt 0 ] && [ "$identical" = "$images" ] ||
    fail "images moved: $(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 != "LOAD-IDENTICAL" {print $1"/"$2}' "$WORK/rebuild/summary.txt" | tr '\n' ' ')"

# 3. rtriad_bfp8_swiglu: rtriad_bfp8's ctargs and recorded defines, plus FT_RTRIAD_SWIGLU and
#    (S3a) FT_RTRIAD_RESADD.
KERNEL_DEFINES="-DFT_TT_MATMUL_DST_SYNC=1 -DFT_RTRIAD_SWIGLU=1 -DFT_RTRIAD_RESADD=1" \
    RELOCATE_KERNEL_TEXT=1 READER_CTARGS=0,1,2048,1088,1 COMPUTE_CTARGS=0,1,16,1 WRITER_CTARGS=16,2048,1 \
    CB_TABLE=$K/cb_tables/matmul_bfp8_b_swiglu.cbtable bash "$TT/build_tensix_program.sh" \
    "$K/resident_matmul_reader.c" "$K/resident_matmul_compute.c" "$K/resident_matmul_writer.c" \
    "$PROGRAMS/$NEW" > "$WORK/$NEW.build.log" 2>&1 ||
    fail "$NEW build failed (see $WORK/$NEW.build.log)"
for image in brisc ncrisc trisc0 trisc1 trisc2; do
    elf=$PROGRAMS/$NEW/$image.elf
    [ -f "$elf" ] || { fail "$NEW/$image.elf missing"; continue; }
    size=$(text_bytes "$elf")
    same=$(same_load "$PROGRAMS/$BASE/$image.elf" "$elf")
    say "$NEW $image.elf text $size B ($BASE $(text_bytes "$PROGRAMS/$BASE/$image.elf") B) vs $BASE: $same"
    [ "$size" -lt "$SLAB_BYTES" ] || fail "$image.elf text $size B does not fit the $SLAB_BYTES B slab"
    [ "$same" != LOAD-IDENTICAL ] || fail "$image.elf equals $BASE's: FT_RTRIAD_SWIGLU did not take"
done
for image in fw_brisc fw_ncrisc fw_trisc0 fw_trisc1 fw_trisc2; do
    same=$(same_load "$PROGRAMS/$BASE/$image.elf" "$PROGRAMS/$NEW/$image.elf")
    say "$NEW $image.elf vs $BASE: $same"
    [ "$same" = LOAD-IDENTICAL ] || fail "$image.elf differs from $BASE's: Pantheon would refuse the firmware family"
done

# 4. The blob from ../programs.
check=1
[ "$REGEN" = 1 ] && check=0
(cd "$HERE" && PROGRAMS=$PROGRAMS CHECK=$check bash regen_tensix_program_header.sh) > "$WORK/regen.log" 2>&1
regen_rc=$?
say "regen (CHECK=$check) rc=$regen_rc: $(grep -E 'IDENTICAL|DIFFERS|distinct firmware' "$WORK/regen.log" | tr -s ' ' | tr '\n' ' ')"
if [ "$REGEN" = 1 ]; then
    [ "$regen_rc" = 0 ] || fail "regen refused (see $WORK/regen.log)"
    python3 "$HERE/check_tensix_program_provenance.py" --header "$TT/ft_tt_programs.h" > "$WORK/provenance.log" 2>&1 ||
        fail "provenance --header (see $WORK/provenance.log)"
    say "provenance --header: $(tail -1 "$WORK/provenance.log")"
    make -C "$HERE" verify-programs > "$WORK/verify_build.log" 2>&1
    "$HERE/build/verify_tensix_program_blob" "$PROGRAMS" > "$WORK/verify.log" 2>&1 || fail "blob verify (see $WORK/verify.log)"
    say "verify: $(tail -1 "$WORK/verify.log")"
fi

[ "$red" = 0 ] && say "ALL GATES GREEN" || say "GATES RED"
exit "$red"
