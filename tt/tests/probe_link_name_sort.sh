#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# probe_link_name_sort.sh
#   Search for link names for the two TRISC firmware globals that leave every image
#   LOAD-IDENTICAL. Renaming a global can move code (GCC orders some objects by symbol, so
#   register allocation follows the name). For each candidate pair, patch a copy of the tree,
#   rebuild the named programs and compare every image to the reference. Run DETACHED.
#
#   Usage:  probe_link_name_sort.sh <tree> <ref-programs> <out> <program,...> "cfg dest" ...
#   Output: <out>/RESULT (one line per candidate), <out>/DONE
#
#   Status: added 2026-10-08 (clean-room Phase 7).
#   Issues: None
#   Todo:   None
#
set -u
TREE=${1:?tree}; REF=${2:?ref}; OUT=${3:?out}; PROGS=${4:?programs}; shift 4
mkdir -p "$OUT"; : > "$OUT/RESULT"; rm -f "$OUT/DONE"
OLD_CFG=$(grep -o '__asm__("[^"]*cfg_state_id[^"]*")' "$TREE/tt/fw/ft_trisc_fw.c" | cut -d'"' -f2)
OLD_DEST=$(grep -o '__asm__("[^"]*dest_offset_id[^"]*")' "$TREE/tt/fw/ft_trisc_fw.c" | cut -d'"' -f2)
n=0
for pair in "$@"; do
    n=$((n + 1)); set -- $pair; cfg=$1; dest=$2
    work=$OUT/t$n; rm -rf "$work"; cp -r "$TREE" "$work"
    sed -i "s/$OLD_CFG/$cfg/; s/$OLD_DEST/$dest/" "$work/tt/fw/ft_trisc_fw.c" "$work/tt/kernels/ft_tensix_compute.h"
    moved=0
    for p in ${PROGS//,/ }; do
        bash "$work/tt/tools/rebuild_tensix_program.sh" "$REF/$p" "$work/out/$p" > "$work/$p.log" 2>&1 || moved=$((moved + 100))
        for image in "$REF/$p"/*.elf; do
            v=$(python3 "$work/tt/tools/compare_tensix_load_images.py" "$image" "$work/out/$p/$(basename "$image")" | cut -d: -f1)
            [ "$v" = LOAD-IDENTICAL ] || moved=$((moved + 1))
        done
    done
    echo "$cfg $dest moved=$moved" >> "$OUT/RESULT"
    rm -rf "$work"
    [ $moved = 0 ] && break
done
echo DONE > "$OUT/DONE"
