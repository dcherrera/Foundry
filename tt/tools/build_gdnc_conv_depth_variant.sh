#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# build_gdnc_conv_depth_variant.sh
#   Build a `gdnc` program directory at a DIFFERENT CONV BLOCK DEPTH, derived from a shipped
#   one's own provenance records.
#
#   Usage:  build_gdnc_conv_depth_variant.sh <source-gdnc-dir> <depth> <output-dir>
#   Env:    KERNELS  where the named sources live (default ../kernels beside this script)
#
#   ------------------------------------------------------------------------------
#   WHY THIS IS NOT `rebuild_tensix_program.sh` WITH A KNOB
#   ------------------------------------------------------------------------------
#   `rebuild_tensix_program.sh` takes exactly one knob, `EXTRA_KERNEL_DEFINES`, and it
#   APPENDS: a rebuild always applies what the directory recorded, so it can never come out as
#   something other than what it was. The CONV depth is not a define, it is compile-time arg 11
#   of the reader and arg 10 of the compute kernel, and moving it is a SUBSTITUTION. A
#   substitution knob on the general rebuilder would be exactly the "invent a recipe" hazard
#   that tool exists to refuse — a short or wrong CTARGS list folds to a literal 0 under SFPI
#   GCC 15.1.0 with no diagnostic, which is how a `rope` rebuild once linked a smaller,
#   functionally different kernel and reported the CONTROL as stale.
#
#   So the substitution lives here, where it can be family-specific and checked:
#     - the source directory must actually be a `gdnc` program (its recorded sources are named)
#     - the two ctargs that carry the depth must AGREE in the source before anything moves,
#       because a source directory that already disagrees is not a base to derive from
#     - both are rewritten to the same new value, and the arg COUNT is required to be
#       unchanged, so a short list cannot come out of here
#     - everything else — the CB table, watcher, relocation, defines, the writer's ctargs —
#       comes from the records untouched
#   The output records the substituted recipe, so the variant rebuilds itself faithfully from
#   then on with `rebuild_tensix_program.sh` and needs this script exactly once.
#
#   THE THIRD SIDE IS NOT THIS SCRIPT'S. The host must be told the same depth
#   (`FT_TT_OP_GDNC_CONV_DEPTH`), and it DECLARES that number on every CONV descriptor; both
#   kernels refuse a descriptor whose declared depth is not the one they were built with. So a
#   variant armed against a host at another depth REFUSES rather than running quietly at its
#   own — see kGdncByteConvDepth in kernels/resident_gdnc_protocol.h.
#
#   Status: added 2026-08-28 for the gdnc CONV depth sweep (PX2). Derived from
#           build_gdn_conv_depth_variant.sh, which does the same job one family over; the two
#           differ ONLY in the source names and the two arg indices.
#   Issues: None
#   Todo:   None
#
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TT=$(cd -- "$HERE/.." && pwd)
KERNELS=${KERNELS:-$TT/kernels}

SOURCE_DIR=${1:?usage: build_gdnc_conv_depth_variant.sh <source-gdnc-dir> <depth> <output-dir>}
DEPTH=${2:?usage: build_gdnc_conv_depth_variant.sh <source-gdnc-dir> <depth> <output-dir>}
OUT=${3:?usage: build_gdnc_conv_depth_variant.sh <source-gdnc-dir> <depth> <output-dir>}
SOURCE_DIR=$(cd -- "$SOURCE_DIR" && pwd)

# Which compile-time arg carries the depth on each processor. Stated once here and asserted
# against the source's own values below; test_gdnc_ops.c pins the same two indices against the
# kernel sources' documented build lines.
READER_DEPTH_INDEX=11
COMPUTE_DEPTH_INDEX=10

case "$DEPTH" in
    ''|*[!0-9]*) echo "build-gdnc-depth: depth '$DEPTH' is not a number" >&2; exit 1;;
esac
if [ "$DEPTH" -lt 1 ] || [ "$DEPTH" -gt 255 ]; then
    echo "build-gdnc-depth: depth $DEPTH is outside 1..255 -- the host declares it in ONE BYTE" \
         "of the FLAGS word" >&2
    exit 1
fi

recipe_value() {
    awk -v want="$2" '$1 == "recipe" {
        for (i = 2; i <= NF; i++) {
            equals = index($i, "=")
            if (substr($i, 1, equals - 1) == want) { print substr($i, equals + 1); exit }
        }
    }' "$1"
}
record_source() { awk '$1 == "source" { print $2; exit }' "$1"; }

# One comma list, one index, one new value -- and the LENGTH must not change, because a
# CTARGS list one short is the failure mode that folds to a literal 0 with no diagnostic.
substitute_ctarg() {
    awk -v list="$1" -v index_wanted="$2" -v value="$3" '
        BEGIN {
            n = split(list, a, ",")
            if (index_wanted + 1 > n) { print "SHORT"; exit }
            a[index_wanted + 1] = value
            out = a[1]
            for (i = 2; i <= n; i++) out = out "," a[i]
            print out
        }'
}
ctarg_at() {
    awk -v list="$1" -v index_wanted="$2" '
        BEGIN { n = split(list, a, ","); if (index_wanted + 1 > n) print "SHORT"; else print a[index_wanted + 1] }'
}

for image in brisc ncrisc trisc0; do
    [ -f "$SOURCE_DIR/$image.elf.prov" ] || {
        echo "build-gdnc-depth: $SOURCE_DIR has no $image.elf.prov -- it does not record how it" \
             "was built, so nothing can be derived from it" >&2
        exit 1
    }
done

READER_SRC=$(record_source "$SOURCE_DIR/brisc.elf.prov")
WRITER_SRC=$(record_source "$SOURCE_DIR/ncrisc.elf.prov")
COMPUTE_SRC=$(record_source "$SOURCE_DIR/trisc0.elf.prov")
if [ "$READER_SRC" != "resident_gdnc_reader.c" ] || [ "$COMPUTE_SRC" != "resident_gdnc_compute.c" ]; then
    echo "build-gdnc-depth: $SOURCE_DIR is not a gdnc program -- it records $READER_SRC /" \
         "$COMPUTE_SRC. The depth's arg indices are this family's and nothing else's." >&2
    exit 1
fi

READER_CTARGS=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" ctargs)
WRITER_CTARGS=$(recipe_value "$SOURCE_DIR/ncrisc.elf.prov" ctargs)
COMPUTE_CTARGS=$(recipe_value "$SOURCE_DIR/trisc0.elf.prov" ctargs)
CB_TABLE_NAME=$(recipe_value "$SOURCE_DIR/trisc0.elf.prov" cb_table)
WATCHER=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" watcher)
LIGHTWEIGHT_ASSERTS=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" lightweight_asserts)
RELOCATE_KERNEL_TEXT=$(recipe_value "$SOURCE_DIR/brisc.elf.prov" relocate_kernel_text)
KERNEL_DEFINES=$(recipe_value "$SOURCE_DIR/trisc0.elf.prov" defines | tr ',' ' ')  # recorded comma-joined

SOURCE_READER_DEPTH=$(ctarg_at "$READER_CTARGS" "$READER_DEPTH_INDEX")
SOURCE_COMPUTE_DEPTH=$(ctarg_at "$COMPUTE_CTARGS" "$COMPUTE_DEPTH_INDEX")
if [ "$SOURCE_READER_DEPTH" = SHORT ] || [ "$SOURCE_COMPUTE_DEPTH" = SHORT ]; then
    echo "build-gdnc-depth: $SOURCE_DIR's CTARGS are too short to carry a depth" \
         "(reader[$READER_DEPTH_INDEX], compute[$COMPUTE_DEPTH_INDEX])" >&2
    exit 1
fi
if [ "$SOURCE_READER_DEPTH" != "$SOURCE_COMPUTE_DEPTH" ]; then
    echo "build-gdnc-depth: $SOURCE_DIR ALREADY DISAGREES WITH ITSELF -- reader depth" \
         "$SOURCE_READER_DEPTH, compute depth $SOURCE_COMPUTE_DEPTH. That is not a base to" \
         "derive a variant from; fix the source directory first." >&2
    exit 1
fi

NEW_READER_CTARGS=$(substitute_ctarg "$READER_CTARGS" "$READER_DEPTH_INDEX" "$DEPTH")
NEW_COMPUTE_CTARGS=$(substitute_ctarg "$COMPUTE_CTARGS" "$COMPUTE_DEPTH_INDEX" "$DEPTH")
CB_TABLE=""
[ -n "$CB_TABLE_NAME" ] && CB_TABLE=$KERNELS/cb_tables/$CB_TABLE_NAME

echo "=== gdnc CONV depth variant: $SOURCE_DIR ($SOURCE_READER_DEPTH) -> $OUT ($DEPTH)"
printf '    reader  CTARGS %s\n            ->     %s\n' "$READER_CTARGS" "$NEW_READER_CTARGS"
printf '    compute CTARGS %s\n            ->     %s\n' "$COMPUTE_CTARGS" "$NEW_COMPUTE_CTARGS"
printf '    writer  CTARGS %s (untouched)\n' "$WRITER_CTARGS"
printf '    CB_TABLE=%s WATCHER=%s LIGHTWEIGHT_ASSERTS=%s RELOCATE_KERNEL_TEXT=%s DEFINES=%s\n' \
       "${CB_TABLE_NAME:-<none>}" "$WATCHER" "$LIGHTWEIGHT_ASSERTS" "$RELOCATE_KERNEL_TEXT" \
       "${KERNEL_DEFINES:-<none>}"
echo "    THE HOST MUST BE TOLD THE SAME NUMBER: FT_TT_OP_GDNC_CONV_DEPTH=$DEPTH."
echo "    A host at any other depth is REFUSED by both kernels, not silently run at $DEPTH."

mkdir -p "$OUT"
env READER_CTARGS="$NEW_READER_CTARGS" COMPUTE_CTARGS="$NEW_COMPUTE_CTARGS" \
    WRITER_CTARGS="$WRITER_CTARGS" CB_TABLE="$CB_TABLE" \
    WATCHER="$WATCHER" LIGHTWEIGHT_ASSERTS="$LIGHTWEIGHT_ASSERTS" \
    RELOCATE_KERNEL_TEXT="$RELOCATE_KERNEL_TEXT" \
    KERNEL_DEFINES="$KERNEL_DEFINES" \
    bash "$TT/build_tensix_program.sh" \
        "$KERNELS/$READER_SRC" "$KERNELS/$COMPUTE_SRC" "$KERNELS/$WRITER_SRC" "$OUT"
