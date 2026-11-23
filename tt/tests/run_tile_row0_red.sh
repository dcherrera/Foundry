#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
# Green-then-red for ft_tt_tile_row0.c.
#
# A test nobody has watched fail is not evidence. This builds the real source once and
# requires 0 failures, then applies each mutation to a COPY of the source, rebuilds
# whole-program into a scratch directory of its own, and requires the run to fail.
#
# Whole-program into a fresh directory each time is not caution, it is the documented
# hazard: GNU make 3.81 (what Apple ships) compares mtimes at one-second granularity and
# silently serves the previous object for a source edited in the same second.
#
#   ./run_tile_row0_red.sh
#
# Exit 0 only if green is green AND every mutation went red.

set -u
cd "$(dirname "$0")" || exit 1

CC=${CC:-cc}
CFLAGS="-std=c99 -O1 -g -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion \
        -Wstrict-prototypes -Wformat=2 -fstack-protector-strong -I.."
SOURCES="../ft_tt_kmd.c ../ft_tt_arch.c ../ft_tt_arch_blackhole.c ../ft_tt_device.c \
         ../ft_tt_tlb.c ../ft_tt_coord.c ../ft_tt_noc.c ../ft_tt_tilize.c"
SCRATCH=build-row0-red
rc_total=0

rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"

echo "=== GREEN: the real source"
# shellcheck disable=SC2086
$CC $CFLAGS $SOURCES ../ft_tt_tile_row0.c test_tile_row0.c -o "$SCRATCH/green" || exit 1
if "$SCRATCH/green"; then
    echo "GREEN OK"
else
    echo "GREEN FAILED — stop, the baseline is broken"
    exit 1
fi

# name | sed expression applied to ft_tt_tile_row0.c
run_mutation() {
    name=$1
    expr=$2
    dir="$SCRATCH/$name"

    mkdir -p "$dir"
    sed "$expr" ../ft_tt_tile_row0.c > "$dir/ft_tt_tile_row0.c"
    if cmp -s "$dir/ft_tt_tile_row0.c" ../ft_tt_tile_row0.c; then
        echo "MUTATION $name: sed CHANGED NOTHING — the expression is stale, not the code"
        rc_total=1
        return
    fi
    echo "=== RED: $name"
    # shellcheck disable=SC2086
    $CC $CFLAGS -I. $SOURCES "$dir/ft_tt_tile_row0.c" test_tile_row0.c -o "$dir/mutant" || {
        echo "MUTATION $name: did not build"
        rc_total=1
        return
    }
    if "$dir/mutant"; then
        echo "MUTATION $name: STILL GREEN — the test does not cover it"
        rc_total=1
    else
        echo "MUTATION $name: RED, as required"
    fi
}

# The element-index stride (256 elements between a face's halves) mistaken for the BYTE
# stride. Reads the middle of face 0 and returns a plausible wrong answer.
run_mutation face_stride_256 \
    's/chunk \* FT_TT_TILE_ROW0_FACE_BYTES/chunk * 256u/'

# The two chunks swapped: columns 16-31 where columns 0-15 belong.
run_mutation chunks_swapped \
    's/chunk \* FT_TT_TILE_ROW0_FACE_BYTES/(1u - chunk) * FT_TT_TILE_ROW0_FACE_BYTES/'

# A third, cheap: the tile stride wrong. Included because it is the other constant in the
# same expression and a test that catches two of three would be a coincidence.
run_mutation tile_stride_1024 \
    's/tile_index \* FT_TT_TILE_ROW0_TILE_BYTES/tile_index * 1024u/'

echo
if [ "$rc_total" -eq 0 ]; then
    echo "ALL MUTATIONS RED, GREEN GREEN"
else
    echo "SOMETHING DID NOT GO RED — see above"
fi
exit "$rc_total"
