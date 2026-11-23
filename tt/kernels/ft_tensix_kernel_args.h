/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tensix_kernel_args.h
 * @brief The two argument channels of a Tensix kernel: runtime arguments in L1 and
 *        compile-time arguments baked in by the build.
 *
 * Runtime arguments are 32-bit words in an L1 block; the firmware publishes the block's base
 * in `rta_l1_base` before it calls the kernel. Compile-time arguments arrive as the
 * comma-separated macro KERNEL_COMPILE_TIME_ARGS and fold to constants. Only 4-byte arguments
 * exist.
 *
 * tools/check_tensix_kernel_compile_time_arg_counts.py scans kernel sources for calls of the
 * form `ft_tensix_compile_time_arg(<integer literal>)`, so that name and call shape are fixed.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#ifndef FT_TENSIX_KERNEL_ARGS_H
#define FT_TENSIX_KERNEL_ARGS_H

#include <stdint.h>

#include "../fw/ft_bh_regs.h"

/** Base of this launch's runtime arguments (an L1 word array; set by the firmware). */
extern uint32_t FT_BH_L1_PTR *rta_l1_base;

/** Runtime argument `index`. Deliberately non-volatile: repeated reads may be merged. */
static FT_BH_INLINE uint32_t ft_tensix_runtime_arg(int index)
{
    return *(FT_BH_L1_PTR uint32_t *)(uintptr_t)&rta_l1_base[index];
}

#ifndef KERNEL_COMPILE_TIME_ARGS
#define KERNEL_COMPILE_TIME_ARGS 0
#endif

/** Compile-time argument `index`. Callers pass literals, so the table folds away; an
 *  out-of-range index is not diagnosed here (the python checker above is the guard). */
static FT_BH_INLINE uint32_t ft_tensix_compile_time_arg(unsigned index)
{
    static const uint32_t compile_time_args[] = {KERNEL_COMPILE_TIME_ARGS};

    return compile_time_args[index];
}

#endif /* FT_TENSIX_KERNEL_ARGS_H */
