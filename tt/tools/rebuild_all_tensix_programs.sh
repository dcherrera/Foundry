#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# rebuild_all_tensix_programs.sh
#   Rebuild EVERY program directory under a programs root from its own provenance records
#   into a new root, then compare each image against the original with the instruction-stream
#   diff. The build-side gate for a toolchain-level change (clean-room phases): it answers
#   "does everything still build, and exactly which images moved".
#
#   Usage:  rebuild_all_tensix_programs.sh <programs-root> <out-root> [program ...]
#           EXTRA_KERNEL_DEFINES  passed through to every rebuild (a red build)
#   Output: <out-root>/<program>/ images, <out-root>/<program>.rebuild.log,
#           <out-root>/<program>.diff, <out-root>/summary.txt
#
#   Per image the summary says IDENTICAL (byte-equal .text), DIFFERS, or MISSING; whether it
#   is LOAD-IDENTICAL (tools/compare_tensix_load_images.py: what reaches the card); and the
#   kernel's .text size, so a change that pushes a kernel toward its 12 KiB text slab is
#   visible here before it reaches a card.
#
#   Status: added 2026-10-08 (clean-room Phase 2).
#   Issues: None
#   Todo:   None
#
set -uo pipefail

SRC=${1:?usage: rebuild_all_tensix_programs.sh <programs-root> <out-root> [program ...]}
OUT=${2:?usage: rebuild_all_tensix_programs.sh <programs-root> <out-root> [program ...]}
shift 2
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SIZE=${SFPI:-/opt/tenstorrent/sfpi}/compiler/bin/riscv-tt-elf-size
OBJCOPY=${SFPI:-/opt/tenstorrent/sfpi}/compiler/bin/riscv-tt-elf-objcopy

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
PROGRAMS=("$@")
if [ ${#PROGRAMS[@]} -eq 0 ]; then
    for dir in "$SRC"/*/; do [ -f "$dir/brisc.elf.prov" ] && PROGRAMS+=("$(basename "$dir")"); done
fi

failed=0 identical=0 differing=0 missing=0 load_identical=0
: > "$OUT/summary.txt"
for name in "${PROGRAMS[@]}"; do
    rm -rf "$OUT/$name"; mkdir -p "$OUT/$name"
    if ! bash "$HERE/rebuild_tensix_program.sh" "$SRC/$name" "$OUT/$name" > "$OUT/$name.rebuild.log" 2>&1; then
        echo "$name REBUILD FAILED (see $name.rebuild.log)" >> "$OUT/summary.txt"; failed=$((failed + 1)); continue
    fi
    bash "$HERE/diff_tensix_kernel_instructions.sh" --dir "$SRC/$name" "$OUT/$name" > "$OUT/$name.diff" 2>&1
    for image in "$SRC/$name"/*.elf; do
        file=$(basename "$image")
        new="$OUT/$name/$file"
        if [ ! -f "$new" ]; then verdict=MISSING; missing=$((missing + 1))
        elif "$OBJCOPY" -O binary -j .text "$image" "$OUT/.a.text" 2>/dev/null &&
             "$OBJCOPY" -O binary -j .text "$new" "$OUT/.b.text" 2>/dev/null &&
             cmp -s "$OUT/.a.text" "$OUT/.b.text"; then
            verdict=IDENTICAL; identical=$((identical + 1))
        else verdict=DIFFERS; differing=$((differing + 1)); fi
        text=$("$SIZE" -A "$new" 2>/dev/null | awk '$1 == ".text" {print $2}')
        load=MISSING
        [ -f "$new" ] && load=$(python3 "$HERE/compare_tensix_load_images.py" "$image" "$new" | cut -d: -f1)
        [ "$load" = LOAD-IDENTICAL ] && load_identical=$((load_identical + 1))
        printf '%-18s %-16s %-9s %-14s text %s\n' "$name" "$file" "$verdict" "$load" "${text:-?}" >> "$OUT/summary.txt"
    done
done
{
    echo
    echo "programs: ${#PROGRAMS[@]}  rebuild failures: $failed"
    echo "images: $identical .text-identical, $differing differing, $missing missing"
    echo "images LOAD-IDENTICAL (entry, every PT_LOAD, .segments): $load_identical"
    echo "largest kernel .text: $(grep -v ' fw_' "$OUT/summary.txt" | awk '/ text [0-9]/ {print $NF, $1"/"$2}' | sort -n | tail -1)"
} >> "$OUT/summary.txt"
rm -f "$OUT/.a.text" "$OUT/.b.text"
echo DONE > "$OUT/DONE"
