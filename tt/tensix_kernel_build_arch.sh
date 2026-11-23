#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# tensix_kernel_build_arch.sh
#   The ONE place the Tensix kernel builders name an architecture. Sourced by
#   build_tensix_kernel.sh (compute) and build_tensix_dataflow_kernel.sh (data movement);
#   it exports the -mcpu names, the arch define and the board-shaped defines that every
#   image is compiled with.
#
#   Usage:  . tensix_kernel_build_arch.sh
#   Env:    TT_ARCH            part to build for (default blackhole)
#           TT_NUM_L1_BANKS    live Tensix core count of the BOARD (default 140)
#           TT_NUM_DRAM_BANKS  live DRAM bank count of the BOARD (default 8)
#   Exports:
#           TT_ARCH_DEFINE     -DARCH_BLACKHOLE
#           TT_MCPU_COMPUTE    -mcpu for TRISC (has the Tensix co-processor instructions)
#           TT_MCPU_DM         -mcpu for BRISC/NCRISC (does not)
#           TT_BOARD_DEFINES   the -D block describing the board's live geometry
#
#   Nothing here names a Tenstorrent source tree: since clean-room plan Phases 2-6 every
#   image is built from Foundry sources only (tt/fw/, tt/kernels/) with the SFPI compiler.
#
#   Status: verified on the p150a card host as a refactor of the literals the two builders used to
#           carry inline; staged-tree paths removed 2026-10-08 (clean-room Phase 7).
#   Issues: None
#   Todo:   None
#
# ---------------------------------------------------------------------------
# ARCHITECTURE: ONE IS IMPLEMENTED AND THE REST ARE REFUSED BY NAME
# ---------------------------------------------------------------------------
# This is the same rule ft_tt_arch_select() applies on the host side, for the same
# reason: we have exactly one Tenstorrent part and can verify nothing for any other, and
# a plausible-looking wrong build is worse than no build at all.
#
# What a porter owes, all of it verifiable only against that silicon:
#   1. the two -mcpu names for the part (SFPI's riscv-tt-elf target list),
#   2. -DARCH_<NAME>,
#   3. that part's firmware (tt/fw/) and linker constants (ft_tt_memmap.h),
#   4. and on the HOST side, an ft_tt_arch_<name>.c plus that part's ft_tt_memmap.h /
#      ft_tt_launch.h facts. Building kernels for a part the host cannot address is not
#      progress, it is a wedged core waiting to happen.
set -euo pipefail

TT_ARCH=${TT_ARCH:-blackhole}

case $TT_ARCH in
blackhole)
    TT_ARCH_DEFINE="-DARCH_BLACKHOLE"
    TT_MCPU_COMPUTE="tt-bh-tensix"
    TT_MCPU_DM="tt-bh"
    ;;
*)
    echo "tensix_kernel_build_arch.sh: TT_ARCH='$TT_ARCH' is not implemented." >&2
    echo "  Only 'blackhole' is. This refuses rather than guessing a toolchain" >&2
    echo "  configuration, for the same reason ft_tt_arch_select() refuses an" >&2
    echo "  unverified part: a wrong image is a wedged core and a walk to the machine." >&2
    echo "  See the ARCHITECTURE block in this file for what a port must supply." >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# BOARD: what the images are told the LIVE grid looks like
# ---------------------------------------------------------------------------
# These are BOARD facts, not architecture facts, and they are the ones that would ship
# silently wrong. NUM_L1_BANKS is literally the Tensix CORE COUNT -- 140 on this card, but
# 120-core p150s are a shipping configuration (dev notes §5) whose images must say 120.
# Nothing in a wrong image complains; the host-side guard is
# ft_tt_coord_check_kernel_build_assumptions(), which compares live telemetry against
# FT_TT_KERNEL_BUILD_L1_BANK_COUNT / _DRAM_BANK_COUNT in ft_tt_memmap.h.
#
# SO: CHANGE THESE AND THOSE TWO MACROS TOGETHER. They are one duplicated contract and
# the check is the only thing that makes a duplicated contract safe.
#
# The two derived defines are DERIVED here rather than typed, because they are exactly
# the pair that would silently disagree with the counts above.
TT_NUM_L1_BANKS=${TT_NUM_L1_BANKS:-140}
TT_NUM_DRAM_BANKS=${TT_NUM_DRAM_BANKS:-8}

# LOG_BASE_2_OF_NUM_DRAM_BANKS only exists because the DRAM bank index is computed with a
# shift, so a non-power-of-two bank count has no valid value -- refuse rather than round.
case $TT_NUM_DRAM_BANKS in
1)  TT_LOG2_DRAM_BANKS=0 ;;
2)  TT_LOG2_DRAM_BANKS=1 ;;
4)  TT_LOG2_DRAM_BANKS=2 ;;
8)  TT_LOG2_DRAM_BANKS=3 ;;
16) TT_LOG2_DRAM_BANKS=4 ;;
*)
    echo "tensix_kernel_build_arch.sh: TT_NUM_DRAM_BANKS=$TT_NUM_DRAM_BANKS is not a" >&2
    echo "  power of two, and LOG_BASE_2_OF_NUM_DRAM_BANKS has no honest value for it." >&2
    exit 2
    ;;
esac

# IS_NOT_POW2_NUM_L1_BANKS follows from the core count and must never be typed beside it.
case $TT_NUM_L1_BANKS in
1|2|4|8|16|32|64|128|256) TT_IS_NOT_POW2_L1_BANKS=0 ;;
*)                        TT_IS_NOT_POW2_L1_BANKS=1 ;;
esac

TT_BOARD_DEFINES="-DNUM_NOCS=2 -DNUM_DRAM_BANKS=$TT_NUM_DRAM_BANKS \
-DLOG_BASE_2_OF_NUM_DRAM_BANKS=$TT_LOG2_DRAM_BANKS \
-DNUM_L1_BANKS=$TT_NUM_L1_BANKS \
-DIS_NOT_POW2_NUM_L1_BANKS=$TT_IS_NOT_POW2_L1_BANKS"
