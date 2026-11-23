#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# diff_tensix_kernel_instructions.sh
#   Compares the INSTRUCTION STREAM of two builds of the same Tensix kernel image. This is
#   the acceptance gate for plan Phase 6.6 (kernels in C rather than C++): identical streams
#   mean identical hardware behaviour, and nothing else needs arguing.
#
#   Usage:  diff_tensix_kernel_instructions.sh <a.elf> <b.elf> [more pairs...]
#           diff_tensix_kernel_instructions.sh --dir <a_dir> <b_dir>
#   Env:    SFPI   toolchain prefix (default /opt/tenstorrent/sfpi)
#
#   WHY THIS AND NOT A PCC. A correctness score computed through the device answers "did
#   the number come out close", which in this project has been wrong four distinct ways
#   (blind to a permutation, wrong for index-returning ops, blind to a 29% scale error on a
#   norm, and unable to see what a mathematical invariance absorbs -- docs/tt_kernel_model.md
#   17-18). A rewrite that changes the language but not the emitted instructions cannot
#   change behaviour at all, so the instruction stream is a STRONGER claim than any score,
#   and it is checkable without the card.
#
#   WHAT IT REPORTS, in increasing order of what a difference costs you to explain:
#     1. .text byte comparison        -- the whole gate when it passes
#     2. section sizes                -- names WHICH section moved when it does not
#     3. disassembly diff             -- the readable version, for a stream that differs
#     4. entry point and _start class -- catches the --just-symbols trap (a kernel that
#                                        links clean and is empty; docs/tt_kernel_toolchain.md 4)
#
#   A DIFFERING STREAM IS NOT AUTOMATICALLY A FAILURE, but it is automatically something
#   that has to be named. Register allocation and instruction scheduling are allowed to move
#   if the flags moved; a different STORE is not, because that is a different transaction.
#
#   Status: used for the vector_add triad's C conversion, 2026-08-13.
#   Issues: None
#   Todo:   None
#
set -uo pipefail
SFPI=${SFPI:-/opt/tenstorrent/sfpi}
OBJDUMP=$SFPI/compiler/bin/riscv-tt-elf-objdump
OBJCOPY=$SFPI/compiler/bin/riscv-tt-elf-objcopy
READELF=$SFPI/compiler/bin/riscv-tt-elf-readelf
NM=$SFPI/compiler/bin/riscv-tt-elf-nm

failures=0
pairs=0

compare_one() {
    local a=$1 b=$2 label=$3
    local scratch
    scratch=$(mktemp -d)
    pairs=$((pairs + 1))

    echo "=== $label"
    echo "    A $a"
    echo "    B $b"

    if [ ! -f "$a" ] || [ ! -f "$b" ]; then
        echo "    MISSING: one side does not exist"
        failures=$((failures + 1))
        rm -rf "$scratch"
        return
    fi

    # 4. The --just-symbols trap first: an empty kernel compares equal to another empty
    #    kernel, so prove both are real before believing any comparison of the two.
    local entry_a entry_b start_a start_b
    entry_a=$("$READELF" -h "$a" | awk '/Entry point/{print $NF}')
    entry_b=$("$READELF" -h "$b" | awk '/Entry point/{print $NF}')
    start_a=$("$NM" "$a" | awk '/ _start$/{print $2}')
    start_b=$("$NM" "$b" | awk '/ _start$/{print $2}')
    echo "    entry  A=$entry_a  B=$entry_b        _start class A=$start_a B=$start_b"
    if [ "$start_a" != "T" ] || [ "$start_b" != "T" ]; then
        echo "    REFUSED: _start is not T on both sides -- one of these is the empty-kernel trap"
        failures=$((failures + 1))
        rm -rf "$scratch"
        return
    fi

    # 2. Section sizes, so a difference says WHERE.
    local size_a size_b
    size_a=$("$READELF" -S "$a" | awk '/\.text|\.data|\.bss/{printf "%s=%s ", $2, $6}')
    size_b=$("$READELF" -S "$b" | awk '/\.text|\.data|\.bss/{printf "%s=%s ", $2, $6}')
    echo "    sections A: $size_a"
    echo "    sections B: $size_b"

    # 1. The gate.
    "$OBJCOPY" -O binary -j .text "$a" "$scratch/a.text" 2>/dev/null
    "$OBJCOPY" -O binary -j .text "$b" "$scratch/b.text" 2>/dev/null
    if cmp -s "$scratch/a.text" "$scratch/b.text"; then
        echo "    IDENTICAL .text ($(wc -c < "$scratch/a.text") bytes)"
    else
        echo "    .TEXT DIFFERS -- disassembly diff follows"
        failures=$((failures + 1))
        # 3. The readable version. Strip the address column so a pure relocation does not
        #    drown the real difference; the addresses are printed above anyway.
        "$OBJDUMP" -d "$a" | sed -e '1,4d' -e 's/^ *[0-9a-f]*:\t[0-9a-f ]*\t//' > "$scratch/a.dis"
        "$OBJDUMP" -d "$b" | sed -e '1,4d' -e 's/^ *[0-9a-f]*:\t[0-9a-f ]*\t//' > "$scratch/b.dis"
        diff -u "$scratch/a.dis" "$scratch/b.dis" | head -200
    fi
    rm -rf "$scratch"
}

if [ "${1:-}" = "--dir" ]; then
    dir_a=${2:?usage: --dir <a_dir> <b_dir>}
    dir_b=${3:?usage: --dir <a_dir> <b_dir>}
    for image in brisc ncrisc trisc0 trisc1 trisc2; do
        [ -f "$dir_a/$image.elf" ] || continue
        compare_one "$dir_a/$image.elf" "$dir_b/$image.elf" "$image"
    done
else
    while [ $# -ge 2 ]; do
        compare_one "$1" "$2" "$(basename "$1")"
        shift 2
    done
fi

echo "=== $pairs image pair(s) compared, $failures differing"
[ "$failures" -eq 0 ] || exit 1
