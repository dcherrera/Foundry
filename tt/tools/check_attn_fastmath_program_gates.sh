#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# check_attn_fastmath_program_gates.sh
#   The compile-side gates of the attention FAST-MATH variants (kernels/
#   resident_attention_compute.c, FAST-MATH BUILDS): six programs, each its base's recipe plus
#   the switch(es), so the two knobs can be measured apart on the card:
#
#     attention_k8v4_fexp        attention_k8v4        + -DATTN_FAST_EXP=1
#     attention_k8v4_lofi        attention_k8v4        + -DATTN_FAST_LOFI=1
#     attention_k8v4_fast        attention_k8v4        + both
#     attention_k8v4_split_fexp  attention_k8v4_split  + -DATTN_FAST_EXP=1
#     attention_k8v4_split_lofi  attention_k8v4_split  + -DATTN_FAST_LOFI=1
#     attention_k8v4_split_fast  attention_k8v4_split  + both
#
#   Needs the SFPI toolchain (7.68.0, the blob's), no card:
#
#     1. explode the committed blob (ft_tt_programs.h) into $WORK/blob: the shipped images
#     2. rebuild EVERY shipped program from its own provenance recipe and THIS tree's sources
#        into ../programs, and require every image LOAD-IDENTICAL to the shipped one: both
#        switches are defines no shipped recipe sets, so no shipped program may move
#     3. build the six variants into ../programs and require, against the base in the same
#        root: trisc1 (MATH, the only thread either switch touches) DIFFERS and fits the
#        12,288 B slab; brisc / ncrisc / trisc0 / trisc2 and the five firmware images
#        LOAD-IDENTICAL; every recipe equal to the base's but for the added define(s); and in
#        trisc1's disassembly the accurate exp's SFPSTOCHRND present exactly when
#        ATTN_FAST_EXP is not (the define reached the exp, not just the image)
#     4. regenerate the blob from ../programs with CHECK=1: a diff, nothing written
#        (tests/run_attn_fastmath_regen_mac.sh writes the header)
#
#   Usage (from tt/):  bash tools/check_attn_fastmath_program_gates.sh
#   Env:  SFPI           toolchain root (default /opt/tenstorrent/sfpi)
#         WORK           scratch (default ../build/attn-fastmath-gates, gitignored)
#         PROGRAMS       programs root (default ../programs)
#         VARIANTS_ONLY  1 = step 3 only, against the bases already in $PROGRAMS (the regen
#                        script's call: it has laid out $PROGRAMS from the blob itself)
#   Any host with SFPI 7.68.0 built for it, bash >= 4, python3, make and cc. On macOS:
#     export SFPI=<install root> PATH=/opt/homebrew/bin:$PATH
#
#   Output: $WORK/gates.txt (the verdicts), $WORK/rebuild/ (per-program logs and summary)
#   Exit:   0 every gate green; 1 any gate red; 2 setup error
#
#   Status: added with the attention fast-math variants (Qwen3.8 prefill, card 0112).
#   Issues: None
#   Todo:   None
#
set -u

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
WORK=${WORK:-$TT/build/attn-fastmath-gates}
PROGRAMS=${PROGRAMS:-$TT/programs}
VARIANTS_ONLY=${VARIANTS_ONLY:-0}
K=$TT/kernels
SLAB_BYTES=12288
GATES=$WORK/gates.txt
FEXP=-DATTN_FAST_EXP=1
LOFI=-DATTN_FAST_LOFI=1
# name  base  added defines (comma-joined, as the recipe records them)
VARIANTS=(
    "attention_k8v4_fexp attention_k8v4 $FEXP"
    "attention_k8v4_lofi attention_k8v4 $LOFI"
    "attention_k8v4_fast attention_k8v4 $FEXP,$LOFI"
    "attention_k8v4_split_fexp attention_k8v4_split $FEXP"
    "attention_k8v4_split_lofi attention_k8v4_split $LOFI"
    "attention_k8v4_split_fast attention_k8v4_split $FEXP,$LOFI"
)
red=0

say() { printf '%s\n' "$*" | tee -a "$GATES"; }
fail() { say "RED: $*"; red=1; }
text_bytes() { "$SFPI/compiler/bin/riscv-tt-elf-size" -A "$1" | awk '$1 == ".text" {print $2}'; }
same_load() { python3 "$HERE/compare_tensix_load_images.py" "$1" "$2" | cut -d: -f1; }
recipe_of() { awk '$1 == "recipe"' "$1"; }
stochrnd_count() { "$SFPI/compiler/bin/riscv-tt-elf-objdump" -d "$1" | grep -c sfpstochrnd; }

[ -x "$SFPI/compiler/bin/riscv-tt-elf-gcc" ] || { echo "no SFPI compiler under $SFPI" >&2; exit 2; }
export SFPI
mkdir -p "$WORK"
: > "$GATES"
say "=== attention fast-math program gates  $(date -u +%FT%TZ)  sfpi $("$SFPI/compiler/bin/riscv-tt-elf-gcc" -dumpversion)"

variant_names() { for v in "${VARIANTS[@]}"; do set -- $v; echo "$1"; done; }

if [ "$VARIANTS_ONLY" != 1 ]; then
    # 1. The shipped images, out of the committed blob.
    make -C "$HERE" explode-programs > "$WORK/explode_build.log" 2>&1 || { echo "explode build failed" >&2; exit 2; }
    rm -rf "$WORK/blob"
    "$HERE/build/explode_tensix_program_blob" "$WORK/blob" --verify > "$WORK/explode.log" 2>&1 ||
        { echo "explode failed (see $WORK/explode.log)" >&2; exit 2; }
    say "exploded $(ls "$WORK/blob" | wc -l | tr -d ' ') programs from ft_tt_programs.h"

    # 2. Every shipped program (the variants excepted) from this tree's sources.
    rm -rf "$PROGRAMS" "$WORK/rebuild"
    shipped=$(ls "$WORK/blob" | grep -vxF "$(variant_names)")
    bash "$HERE/rebuild_all_tensix_programs.sh" "$WORK/blob" "$PROGRAMS" $shipped > /dev/null 2>&1
    mkdir -p "$WORK/rebuild"
    mv "$PROGRAMS"/*.rebuild.log "$PROGRAMS"/*.diff "$PROGRAMS"/summary.txt "$WORK/rebuild/" 2>/dev/null
    rm -f "$PROGRAMS/DONE"
    images=$(awk 'NF >= 4 && $2 ~ /\.elf$/' "$WORK/rebuild/summary.txt" | wc -l | tr -d ' ')
    identical=$(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 == "LOAD-IDENTICAL"' "$WORK/rebuild/summary.txt" | wc -l | tr -d ' ')
    say "LOAD-IDENTICAL: $identical of $images images (every shipped program, kernels + firmware)"
    grep -q "REBUILD FAILED" "$WORK/rebuild/summary.txt" && fail "$(grep -c 'REBUILD FAILED' "$WORK/rebuild/summary.txt") program(s) failed to rebuild"
    [ "$images" -gt 0 ] && [ "$identical" = "$images" ] ||
        fail "images moved: $(awk 'NF >= 4 && $2 ~ /\.elf$/ && $4 != "LOAD-IDENTICAL" {print $1"/"$2}' "$WORK/rebuild/summary.txt" | tr '\n' ' ')"
fi

# 3. The variants: the base's recipe (attention_k8v4's ctargs and CB table, its defines and
#    the split's) plus the switches.
for v in "${VARIANTS[@]}"; do
    set -- $v
    name=$1 base=$2 added=$3
    defines="-DATTN_K_TILE_BYTES=1088 -DATTN_V_TILE_BYTES=576"
    [ "$base" = attention_k8v4_split ] && defines="$defines -DATTN_SPLITK=1"
    [ -f "$PROGRAMS/$base/trisc1.elf" ] || { fail "$base is not in $PROGRAMS"; continue; }
    rm -rf "${PROGRAMS:?}/$name"
    TMPDIR=$WORK KERNEL_DEFINES="$defines ${added//,/ }" \
        RELOCATE_KERNEL_TEXT=1 READER_CTARGS=0,1,2,3,13,2048,1 \
        COMPUTE_CTARGS=0,1,2,3,4,5,6,7,8,9,10,11,12,16,14,1 WRITER_CTARGS=16,2048,1 \
        CB_TABLE=$K/cb_tables/attention_k8v4.cbtable bash "$TT/build_tensix_program.sh" \
        "$K/resident_attention_reader.c" "$K/resident_attention_compute.c" \
        "$K/resident_attention_writer.c" "$PROGRAMS/$name" > "$WORK/$name.build.log" 2>&1 ||
        { fail "$name build failed (see $WORK/$name.build.log)"; continue; }
    for image in brisc ncrisc trisc0 trisc1 trisc2 fw_brisc fw_ncrisc fw_trisc0 fw_trisc1 fw_trisc2; do
        elf=$PROGRAMS/$name/$image.elf
        [ -f "$elf" ] || { fail "$name/$image.elf missing"; continue; }
        same=$(same_load "$PROGRAMS/$base/$image.elf" "$elf")
        case $image in
        trisc1)
            size=$(text_bytes "$elf")
            say "$name trisc1.elf text $size B ($base $(text_bytes "$PROGRAMS/$base/trisc1.elf") B) vs $base: $same"
            [ "$size" -lt "$SLAB_BYTES" ] || fail "$name/trisc1.elf text $size B does not fit the $SLAB_BYTES B slab"
            [ "$same" != LOAD-IDENTICAL ] || fail "$name/trisc1.elf equals $base's: the switch did not take"
            ;;
        *)
            [ "$same" = LOAD-IDENTICAL ] || fail "$name/$image.elf differs from $base's: a switch leaked out of MATH"
            ;;
        esac
        case $image in fw_*) continue ;; esac
        want="$(recipe_of "$PROGRAMS/$base/$image.elf.prov" | sed 's/[[:space:]]*$//'),$added"
        got=$(recipe_of "$elf.prov" | sed 's/[[:space:]]*$//')
        [ "$got" = "$want" ] || fail "$name/$image.elf recipe is not $base's plus $added: $got"
    done
    base_exp=$(stochrnd_count "$PROGRAMS/$base/trisc1.elf")
    exp=$(stochrnd_count "$PROGRAMS/$name/trisc1.elf")
    case $added in
    *FAST_EXP*) [ "$exp" = 0 ] || fail "$name trisc1 still has the accurate exp ($exp SFPSTOCHRND)" ;;
    *)          [ "$exp" = "$base_exp" ] && [ "$exp" -gt 0 ] || fail "$name trisc1 lost the accurate exp ($exp vs $base_exp)" ;;
    esac
    say "$name: 4 other kernels + 5 firmware LOAD-IDENTICAL to $base, recipes $base + $added, accurate-exp SFPSTOCHRND $exp (base $base_exp)"
done

# 4. The blob from ../programs: a diff only.
if [ "$VARIANTS_ONLY" != 1 ]; then
    (cd "$HERE" && TMPDIR=$WORK PROGRAMS=$PROGRAMS CHECK=1 bash regen_tensix_program_header.sh) > "$WORK/regen.log" 2>&1
    say "regen (CHECK=1) rc=$?: $(grep -E 'IDENTICAL|DIFFERS|distinct firmware' "$WORK/regen.log" | tr -s ' ' | tr '\n' ' ')"
fi

[ "$red" = 0 ] && say "ALL GATES GREEN" || say "GATES RED"
exit "$red"
