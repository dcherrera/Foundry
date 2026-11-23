#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# build_tensix_program.sh
#   Builds a whole Tensix program -- all five processor images of one reader/compute/writer
#   triad -- into a single output directory, by driving the two per-class builders:
#
#       build_tensix_kernel.sh           compute  -> trisc0.elf trisc1.elf trisc2.elf
#       build_tensix_dataflow_kernel.sh  reader   -> brisc.elf   (NoC 0)
#       build_tensix_dataflow_kernel.sh  writer   -> ncrisc.elf  (NoC 1)
#
#   Reader on BRISC and writer on NCRISC is a choice, not a constraint: what matters is
#   that the two data-movement kernels sit on DIFFERENT NoCs so they do not contend
#   (docs/tt_kernel_model.md 1). NOC_INDEX is baked into each ELF at compile time and the
#   host must pass the matching brisc_noc_id in the launch message.
#
#   Usage:  build_tensix_program.sh <reader.cpp> <compute.cpp> <writer.cpp> [outdir]
#   Env:    READER_CTARGS  compile-time args for the reader  (default 0,1,2048)
#           COMPUTE_CTARGS compile-time args for the compute (default 0,1,16)
#           WRITER_CTARGS  compile-time args for the writer  (default 16,2048)
#           CB_TABLE       CB table driving the compute descriptors; unset means the
#                          legacy uniform Float16_b 32x32 set. Data movement has no
#                          descriptor header at all, so a CB table changes three of the
#                          five images -- the other two learn the tile size through
#                          READER_CTARGS / WRITER_CTARGS, and the two MUST agree
#           WATCHER        must be 0: Foundry's firmware has no watcher, and both
#                          per-class builders refuse anything else
#           RELOCATE_KERNEL_TEXT
#                          1 links all five kernels into the free L1 arena instead of the
#                          tail of their own firmware holes. Required for every op bigger
#                          than vector_add -- build_tensix_kernel.sh's KERNEL TEXT
#                          RELOCATION block has the measurements that make it mandatory.
#                          The five addresses are PARSED out of ft_tt_op_l1_layout.h, which
#                          is the same header that declares them to the host — so there is
#                          one source of truth and no second copy to keep in step
#   Output: <outdir>/{brisc,ncrisc,trisc0,trisc1,trisc2}.elf, each with a .src sidecar,
#           plus the fw_*.elf they were linked against.
#
#   Feed the result to tools/emit_tensix_kernel_header.py to get ft_tt_kernels.h.
#
#   Status: verified on the p150a card host 2026-08-11 building kernels/vector_add_{reader,compute,
#           writer}.cpp; that build produced the committed ../ft_tt_kernels.h.
#   Issues: None
#   Todo:   None
#
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
READER=${1:?usage: build_tensix_program.sh <reader.cpp> <compute.cpp> <writer.cpp> [outdir]}
COMPUTE=${2:?usage: build_tensix_program.sh <reader.cpp> <compute.cpp> <writer.cpp> [outdir]}
WRITER=${3:?usage: build_tensix_program.sh <reader.cpp> <compute.cpp> <writer.cpp> [outdir]}
OUT=${4:-/tmp/ttk_program}
mkdir -p "$OUT"

# Passed through explicitly rather than relied on being inherited, so that reading this
# file tells you every knob that reaches the per-class builders.
WATCHER=${WATCHER:-0}

# ---- THE RELOCATION MAP: READ FROM THE HEADER, NOT RETYPED HERE ------------------------
# Per-processor slabs inside the KERNEL_CONFIG region, which is where the stock loader puts kernel
# binaries too (each at kernel_config_base + a per-processor offset; the Blackhole region
# is 69 KiB from MEM_MAP_END, measured). Ours starts above MEM_MAP_END so it
# does not sit on the firmware .data staging; ft_tt_op_l1_layout.h has the full note.
#
# These addresses used to be TYPED HERE as well as declared in that header, with nothing
# enforcing agreement -- the same class of duplicated contract that FT_TT_KERNEL_BUILD_*
# has a check for. They are now PARSED out of the header, so there is one source of truth
# and a disagreement is impossible rather than merely unlikely. A missing header or a
# missing macro stops the build; it never falls back to a literal, because a build that
# quietly links kernel text where the launch message does not point produces five images
# that load fine and a core that runs nothing.
kernel_text_base_from_header() {
    local header="$HERE/ft_tt_op_l1_layout.h"
    local macro="FT_TT_OP_TEXT_OFFSET_$1"
    local config_base offset

    [ -f "$header" ] || { echo "missing $header — cannot derive the relocation map" >&2; exit 1; }
    # `#define NAME 0x1234u` -> 0x1234. The u suffix and any trailing comment go.
    config_base=$(awk '$1=="#define" && $2=="FT_TT_OP_KERNEL_CONFIG_BASE" {gsub(/[uU]$/,"",$3); print $3; exit}' "$header")
    offset=$(awk -v m="$macro" '$1=="#define" && $2==m {gsub(/[uU]$/,"",$3); print $3; exit}' "$header")
    if [ -z "$config_base" ] || [ -z "$offset" ]; then
        echo "$header does not define FT_TT_OP_KERNEL_CONFIG_BASE and $macro" >&2
        exit 1
    fi
    printf '0x%X\n' $(( config_base + offset ))
}

# Empty means "link in the firmware hole", i.e. exactly the pre-2026-08-12 behaviour.
BRISC_TEXT=""; NCRISC_TEXT=""; TRISC0_TEXT=""; TRISC1_TEXT=""; TRISC2_TEXT=""
if [ "${RELOCATE_KERNEL_TEXT:-0}" = 1 ]; then
    BRISC_TEXT=$(kernel_text_base_from_header DM0)
    NCRISC_TEXT=$(kernel_text_base_from_header DM1)
    TRISC0_TEXT=$(kernel_text_base_from_header MATH0)
    TRISC1_TEXT=$(kernel_text_base_from_header MATH1)
    TRISC2_TEXT=$(kernel_text_base_from_header MATH2)
    echo "### KERNEL TEXT RELOCATED (from ft_tt_op_l1_layout.h):" \
         "brisc=$BRISC_TEXT ncrisc=$NCRISC_TEXT" \
         "trisc0=$TRISC0_TEXT trisc1=$TRISC1_TEXT trisc2=$TRISC2_TEXT"
fi

echo "===== reader -> BRISC (NoC 0) ====="
CTARGS=${READER_CTARGS:-0,1,2048} NOC=0 WATCHER=$WATCHER KERNEL_TEXT_BASE=$BRISC_TEXT \
    bash "$HERE/build_tensix_dataflow_kernel.sh" "$READER" brisc "$OUT"

echo "===== writer -> NCRISC (NoC 1) ====="
CTARGS=${WRITER_CTARGS:-16,2048} NOC=1 WATCHER=$WATCHER KERNEL_TEXT_BASE=$NCRISC_TEXT \
    bash "$HERE/build_tensix_dataflow_kernel.sh" "$WRITER" ncrisc "$OUT"

echo "===== compute -> TRISC0/1/2 ====="
CTARGS=${COMPUTE_CTARGS:-0,1,16} WATCHER=$WATCHER CB_TABLE=${CB_TABLE:-} \
    KERNEL_TEXT_BASE_TRISC0=$TRISC0_TEXT KERNEL_TEXT_BASE_TRISC1=$TRISC1_TEXT \
    KERNEL_TEXT_BASE_TRISC2=$TRISC2_TEXT \
    bash "$HERE/build_tensix_kernel.sh" "$COMPUTE" "$OUT"

echo "===== program images ====="
for image in brisc ncrisc trisc0 trisc1 trisc2; do
    if [ ! -f "$OUT/$image.elf" ]; then
        echo "MISSING $OUT/$image.elf" >&2
        exit 1
    fi
    "${SFPI:-/opt/tenstorrent/sfpi}/compiler/bin/riscv-tt-elf-size" "$OUT/$image.elf" | tail -1 \
        | awk -v n="$image.elf" -v s="$(cat "$OUT/$image.elf.src")" \
              '{printf "%-12s text=%-6s data=%-4s bss=%-5s  from %s\n", n, $1, $2, $3, s}'
done
echo __PROGRAM_DONE__
