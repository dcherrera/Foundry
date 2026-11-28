#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_gdn_round_regen_mac.sh
#   Regenerate ft_tt_programs.h with an SFPI 7.68.0 toolchain built for macOS:
#     1. explode the shipped blob and rebuild every program from its recipe against this tree's
#        kernels in tt/build/gdnround/rebuilt, to learn which images MOVE;
#     2. tt/programs/ = the exploded blob plus the generated headers its recipe names;
#     3. rebuild only the movers IN PLACE in tt/programs/;
#     4. emit, then verify the blob byte-for-byte and the provenance.
#   Everything stays under tt/build/gdnround/ and tt/programs/ (gitignored).
#
#   SFPI=<sfpi install> bash tt/tests/run_gdn_round_regen_mac.sh
#
#   d1, the GDN state round bias (kGdnFlagStateRound / kGdncDeltaFlagStateRound): the movers
#   must be exactly gdn's and gdnc's compute images (trisc0-2) and every other image IDENTICAL.
#   The script REFUSES any other mover set before writing the header.
#
#   MOVERS / IMAGES override the expected mover set and the image-name pattern that may move: dswap
#   (kGdnFlagDecaySwap) is MOVERS=gdn IMAGES='^(brisc|trisc[012])\.elf$'.
#
#   RED=1 builds the red programs instead: gdn and gdnc with a wrong bias (0x0800), into
#   tt/build/gdnround/red/ for PN_TT_PROGRAM_DIR, and leaves the header alone.
#
#   Status: added with d1.
#   Issues: None
#   Todo:   None
set -euo pipefail
: "${SFPI:?set SFPI to the SFPI 7.68.0 install}"
export SFPI
export PATH="$(brew --prefix)/bin:$PATH" # bash 5: the builders use empty arrays under set -u
TT=$(cd "$(dirname "$0")/.." && pwd)
W=$TT/build/gdnround
cd "$TT"
rm -rf "$W/tmp" "$W/blob" "$W/rebuilt" programs
mkdir -p "$W/tmp" programs
make -C tools explode-programs > "$W/explode_build.log" 2>&1
tools/build/explode_tensix_program_blob "$W/blob" --verify --recipes "$W/recipes.txt" > "$W/explode.log" 2>&1
bash tools/rebuild_all_tensix_programs.sh "$W/blob" "$W/rebuilt" > "$W/rebuild_all.out" 2>&1 || true
movers=$(awk '$3 == "DIFFERS" {print $1}' "$W/rebuilt/summary.txt" | sort -u)
echo "movers: ${movers:-none}"
awk '$3 == "DIFFERS" {print $1"/"$2}' "$W/rebuilt/summary.txt"
echo "images: $(awk '$3 == "IDENTICAL"' "$W/rebuilt/summary.txt" | wc -l | tr -d ' ') identical, $(awk '$3 == "DIFFERS"' "$W/rebuilt/summary.txt" | wc -l | tr -d ' ') differ"
MOVERS=${MOVERS:-gdn gdnc}
IMAGES=${IMAGES:-^trisc[012]\.elf$}
[ "$(echo $movers)" = "$MOVERS" ] || { echo "REFUSED: expected only $MOVERS to move" >&2; exit 1; }
other=$(awk -v re="$IMAGES" '$3 == "DIFFERS" && $2 !~ re' "$W/rebuilt/summary.txt")
[ -z "$other" ] || { echo "REFUSED: a non-compute image moved: $other" >&2; exit 1; }
for d in "$W"/blob/*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    cp -R "$d" programs/
    for h in "$W/rebuilt/$n"/*.h; do [ -f "$h" ] && cp -p "$h" "programs/$n/"; done
done
if [ "${RED:-0}" = 1 ]; then
    rm -rf "$W/red"
    cp -R programs "$W/red"
    for n in gdn gdnc; do
        EXTRA_KERNEL_DEFINES=-DFT_SFPU_TF32_ROUND_BIAS=0x0800u \
            bash tools/rebuild_tensix_program.sh "programs/$n" "$W/red/$n" > "$W/rebuild_red_$n.log" 2>&1
    done
    for n in gdn gdnc; do (cd "$W/red/$n" && md5 -r *.elf); done
    exit 0
fi
for n in $MOVERS; do
    bash tools/rebuild_tensix_program.sh "programs/$n" "programs/$n" > "$W/rebuild_$n.log" 2>&1
done
TMPDIR=$W/tmp bash tools/regen_tensix_program_header.sh > "$W/regen.log" 2>&1
rm -f tools/build/verify_tensix_program_blob
make -C tools verify-programs > "$W/verify_build.log" 2>&1
tools/build/verify_tensix_program_blob "$TT/programs" | tail -1
make -C tools check-program-provenance 2>&1 | tail -1
for image in $(for n in $MOVERS; do ls "$TT"/programs/$n/*.elf; done); do
    echo "$(basename "$(dirname "$image")")/$(basename "$image") text $("$SFPI/compiler/bin/riscv-tt-elf-size" -A "$image" | awk '$1 == ".text" {print $2}') B md5 $(md5 -q "$image")"
done
