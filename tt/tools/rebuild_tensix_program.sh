#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# rebuild_tensix_program.sh
#   Rebuild a Tensix PROGRAM DIRECTORY from ITS OWN provenance records.
#
#   Usage:  rebuild_tensix_program.sh <program-dir> [output-dir]
#   Env:    KERNELS  where the named sources live (default ../kernels beside this script)
#           Output defaults to <program-dir>, i.e. rebuild in place.
#
#   ------------------------------------------------------------------------------
#   WHY THIS EXISTS, AND WHY IT IS NOT A SECOND SOURCE OF TRUTH
#   ------------------------------------------------------------------------------
#   Until 2026-08-19 the recipe for each of the 21 shipped programs lived only in prose --
#   spread across kernel headers, four test Makefiles and DEV_NOTES -- and for THREE of them
#   (matmul_assert0, matmul_assert1, reduce_fp32) it was written down nowhere at all. So
#   "rebuild the program that is stale" meant reconstructing a command line by reading, and
#   guessing one is not a theoretical risk: a rebuild of `rope` with invented compile-time
#   args linked a smaller, functionally different kernel that reported the CONTROL as stale
#   (DEV_NOTES.md 5), because a short `CTARGS` list folds to a literal 0 under SFPI
#   GCC 15.1.0 with no diagnostic.
#
#   `regen_tensix_program_header.sh` refuses to own the recipes for exactly that reason:
#   inventing one recipe for all of them would be a second source of truth. This script does
#   not invent anything. It READS the recipe out of the `.prov` record the builder wrote
#   beside each image, which is the recipe that produced the image and therefore the first
#   source of truth, not a second one. If the directory has no records it says so and stops.
#
#   THE CHECK THAT MAKES IT TRUSTWORTHY IS NOT ITS OWN: rebuild into a scratch directory and
#   `cmp` against the original. A faithful rebuild of an unchanged program is byte-identical
#   on all ten images -- that is how the 21-program audit established which directories were
#   really stale and which merely looked it.
#
#   Status: added 2026-08-19. Proven by rebuilding `rope` -- the byte-identity control --
#           from its own records: all ten images identical.
#   Issues: None
#   Todo:   None
#
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
KERNELS=${KERNELS:-$TT/kernels}

SOURCE_DIR=${1:?usage: rebuild_tensix_program.sh <program-dir> [output-dir]}
OUT=${2:-$SOURCE_DIR}
SOURCE_DIR=$(cd -- "$SOURCE_DIR" && pwd)

# `recipe` is one line of `key=value` words; `source` is one word. Both are the format
# tools/tensix_image_provenance.py writes, read here with awk so this script needs nothing
# but a shell.
recipe_value() {
    awk -v want="$2" '$1 == "recipe" {
        for (i = 2; i <= NF; i++) {
            equals = index($i, "=")
            if (substr($i, 1, equals - 1) == want) { print substr($i, equals + 1); exit }
        }
    }' "$1"
}
record_source() { awk '$1 == "source" { print $2; exit }' "$1"; }

for image in brisc ncrisc trisc0; do
    if [ ! -f "$SOURCE_DIR/$image.elf.prov" ]; then
        echo "rebuild: $SOURCE_DIR has no $image.elf.prov -- this directory does not record" \
             "how it was built, so it cannot be rebuilt from itself. Build it with" \
             "build_tensix_program.sh and its documented recipe once; every rebuild after" \
             "that comes from here." >&2
        exit 1
    fi
done

READER_SRC=$(record_source "$SOURCE_DIR/brisc.elf.prov")
WRITER_SRC=$(record_source "$SOURCE_DIR/ncrisc.elf.prov")
COMPUTE_SRC=$(record_source "$SOURCE_DIR/trisc0.elf.prov")

READER_CTARGS=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" ctargs)
WRITER_CTARGS=$(recipe_value "$SOURCE_DIR/ncrisc.elf.prov" ctargs)
COMPUTE_CTARGS=$(recipe_value "$SOURCE_DIR/trisc0.elf.prov" ctargs)
CB_TABLE_NAME=$(recipe_value "$SOURCE_DIR/trisc0.elf.prov" cb_table)
WATCHER=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" watcher)
LIGHTWEIGHT_ASSERTS=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" lightweight_asserts)
RELOCATE_KERNEL_TEXT=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" relocate_kernel_text)
# Empty for every program built before KERNEL_DEFINES existed, which is what makes those
# rebuilds byte-identical rather than merely equivalent.
#
# `EXTRA_KERNEL_DEFINES` is the ONE knob this script takes, and it is APPENDED rather than
# substituted: a rebuild always applies what the directory recorded, so it can never come
# out as something other than what it was. Appending is how a VARIANT directory is derived
# from a shipped one -- the output records the combined string, so the variant rebuilds
# itself faithfully from then on and needs the knob exactly once.
KERNEL_DEFINES=$(recipe_value "$SOURCE_DIR/trisc0.elf.prov" defines | tr ',' ' ')  # recorded comma-joined
if [ -n "${EXTRA_KERNEL_DEFINES:-}" ]; then
    KERNEL_DEFINES="${KERNEL_DEFINES:+$KERNEL_DEFINES }$EXTRA_KERNEL_DEFINES"
    echo "=== NOT A FAITHFUL REBUILD: EXTRA_KERNEL_DEFINES=$EXTRA_KERNEL_DEFINES appended," \
         "so $OUT is a VARIANT of $(basename "$SOURCE_DIR"), not a copy of it" >&2
fi

# A program is ONE build: a kernel's text is linked at its own firmware's
# __fw_export_text_end, and watcher changes the firmware's size. Three images that disagree
# about it are three builds wearing one directory's name, so this refuses rather than
# picking the reader's answer for all five.
for image in ncrisc trisc0 trisc1 trisc2; do
    [ -f "$SOURCE_DIR/$image.elf.prov" ] || continue
    for key in watcher lightweight_asserts defines; do
        mine=$(recipe_value "$SOURCE_DIR/$image.elf.prov" "$key")
        theirs=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" "$key")
        if [ "$mine" != "$theirs" ]; then
            echo "rebuild: $SOURCE_DIR disagrees with itself -- brisc has $key=$theirs and" \
                 "$image has $key=$mine. A program is one build; these are two." >&2
            exit 1
        fi
    done
done

CB_TABLE=""
[ -n "$CB_TABLE_NAME" ] && CB_TABLE=$KERNELS/cb_tables/$CB_TABLE_NAME

echo "=== rebuilding $(basename "$SOURCE_DIR") into $OUT from its own provenance"
printf '    reader  %-38s CTARGS=%s\n' "$READER_SRC"  "$READER_CTARGS"
printf '    compute %-38s CTARGS=%s\n' "$COMPUTE_SRC" "$COMPUTE_CTARGS"
printf '    writer  %-38s CTARGS=%s\n' "$WRITER_SRC"  "$WRITER_CTARGS"
printf '    CB_TABLE=%s WATCHER=%s LIGHTWEIGHT_ASSERTS=%s RELOCATE_KERNEL_TEXT=%s\n' \
       "${CB_TABLE_NAME:-<none>}" "$WATCHER" "$LIGHTWEIGHT_ASSERTS" "$RELOCATE_KERNEL_TEXT"
printf '    KERNEL_DEFINES=%s\n' "${KERNEL_DEFINES:-<none>}"

mkdir -p "$OUT"
env READER_CTARGS="$READER_CTARGS" COMPUTE_CTARGS="$COMPUTE_CTARGS" \
    WRITER_CTARGS="$WRITER_CTARGS" CB_TABLE="$CB_TABLE" \
    WATCHER="$WATCHER" LIGHTWEIGHT_ASSERTS="$LIGHTWEIGHT_ASSERTS" \
    RELOCATE_KERNEL_TEXT="$RELOCATE_KERNEL_TEXT" \
    KERNEL_DEFINES="$KERNEL_DEFINES" \
    bash "$TT/build_tensix_program.sh" \
        "$KERNELS/$READER_SRC" "$KERNELS/$COMPUTE_SRC" "$KERNELS/$WRITER_SRC" "$OUT"
