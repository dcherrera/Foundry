#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# readelf_tensix_kernel_dump.sh
#   Prints the entry point and PT_LOAD segments of a set of Tensix kernel ELFs in exactly
#   the format dump_tensix_kernel_header.c prints them from the embedded blobs in
#   ft_tt_kernels.h. `diff` between the two answers "does the committed header still
#   describe the binaries the linker produced?" using the toolchain as the oracle rather
#   than our own ELF parser.
#
#   Usage:  readelf_tensix_kernel_dump.sh <elfdir>
#           Expects <elfdir>/{brisc,ncrisc,trisc0,trisc1,trisc2}.elf and maps them to the
#           FT_TT_PROC_* names the header uses.
#   Env:    READELF  path to readelf (default the SFPI one; any binutils readelf that
#                    understands elf32-littleriscv will do)
#
#   The round trip:
#     cc -std=c99 -I.. -o /tmp/dump_kernels dump_tensix_kernel_header.c ../ft_tt_elf.c \
#        ../ft_tt_kernel_blob.c ../ft_tt_device.c ../ft_tt_kmd.c ../ft_tt_tlb.c
#     diff <(/tmp/dump_kernels | grep -v '^#') <(bash readelf_tensix_kernel_dump.sh /tmp/ttk_va)
#
#   Status: run on the p150a card host 2026-08-11 -- diff was empty against the committed header.
#   Issues: None
#   Todo:   None
#
set -euo pipefail
READELF=${READELF:-${SFPI:-/opt/tenstorrent/sfpi}/compiler/bin/riscv-tt-elf-readelf}
ELFDIR=${1:?usage: readelf_tensix_kernel_dump.sh <elfdir>}

# ELF file name -> the FT_TT_PROC_* name ft_tt_memmap.h gives that processor.
IMAGES="brisc:DM0 ncrisc:DM1 trisc0:MATH0 trisc1:MATH1 trisc2:MATH2"

for entry in $IMAGES; do
    file=${entry%%:*}
    name=${entry##*:}
    elf=$ELFDIR/$file.elf
    [ -f "$elf" ] || { echo "missing $elf" >&2; exit 1; }

    $READELF -h "$elf" | awk -v n="$name" '/Entry point address:/ {printf "%s entry=0x%08x\n", n, strtonum($NF)}'

    # readelf -lW PT_LOAD columns: Type Offset VirtAddr PhysAddr FileSiz MemSiz Flg Align.
    # ft_tt_elf_parse rounds memory_bytes up to a 4-byte multiple, so round here too --
    # otherwise the diff would flag a difference our loader deliberately introduces.
    $READELF -lW "$elf" | awk -v n="$name" '
        $1 == "LOAD" {
            printf "%s seg%u exec=0x%08x load=0x%08x file=%u mem=%u\n",
                   n, seg++, strtonum($3), strtonum($4), strtonum($5),
                   int((strtonum($6) + 3) / 4) * 4
        }'
done
