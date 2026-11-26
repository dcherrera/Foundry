#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_puller_await_regen_mac.sh
#   Regenerate ft_tt_programs.h with an SFPI 7.68.0 toolchain built for macOS:
#     1. explode the shipped blob and rebuild every program from its recipe against this tree's
#        kernels in tt/build/pullerawait/rebuilt, to learn which images MOVE;
#     2. tt/programs/ = the exploded blob plus the generated headers its recipe names;
#     3. rebuild only the movers IN PLACE in tt/programs/;
#     4. emit, then verify the blob byte-for-byte and the provenance.
#   Everything stays under tt/build/pullerawait/ and tt/programs/ (gitignored).
#
#   SFPI=<sfpi install> bash tt/tests/run_puller_await_regen_mac.sh
#
#   The puller's AWAIT polls a torn line again (resident_puller_abi.h ft_tt_puller_await_step):
#   the movers must be exactly the puller's images and every other image IDENTICAL. The script
#   REFUSES any other mover set before writing the header.
#
#   Status: added with the step-mark AWAIT retry.
#   Issues: None
#   Todo:   None
set -euo pipefail
: "${SFPI:?set SFPI to the SFPI 7.68.0 install}"
export SFPI
export PATH="$(brew --prefix)/bin:$PATH" # bash 5: the builders use empty arrays under set -u
TT=$(cd "$(dirname "$0")/.." && pwd)
W=$TT/build/pullerawait
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
[ "$movers" = puller ] || { echo "REFUSED: expected only puller to move" >&2; exit 1; }
for d in "$W"/blob/*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    cp -R "$d" programs/
    for h in "$W/rebuilt/$n"/*.h; do [ -f "$h" ] && cp -p "$h" "programs/$n/"; done
done
bash tools/rebuild_tensix_program.sh programs/puller programs/puller > "$W/rebuild_puller.log" 2>&1
TMPDIR=$W/tmp bash tools/regen_tensix_program_header.sh > "$W/regen.log" 2>&1
rm -f tools/build/verify_tensix_program_blob
make -C tools verify-programs > "$W/verify_build.log" 2>&1
tools/build/verify_tensix_program_blob "$TT/programs" | tail -1
make -C tools check-program-provenance 2>&1 | tail -1
for image in "$TT"/programs/puller/*.elf; do
    echo "puller/$(basename "$image") text $("$SFPI/compiler/bin/riscv-tt-elf-size" -A "$image" | awk '$1 == ".text" {print $2}') B"
done
