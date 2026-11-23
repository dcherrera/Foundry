/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bfp6_expand_compute.c
 * @brief Route 3 of qwen36 spec §4.1b, IDLE TRISCs as expanders: the same scalar SWAR
 *   expansion (bfp6_expand_tile.h) run by each of the three compute RISC-Vs on its own share
 *   of tiles, L1 -> L1. No Tensix instruction is issued — the coprocessor stays idle — so this
 *   is the TRISCs' RISC-V throughput, which is what the top-k helpers used (Ornith spec
 *   "MULTI-ROW TOP-K and TOP-K HELPERS"). The data movers would feed L1 in the real route.
 *
 * Runtime args (each TRISC its own block): 0 layout  1 tiles  2 repeats  3 l1_in  4 l1_out
 *   5 out slots  6 status
 * Status words: as bfp6_expand_dataflow.c.
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4 price bench).
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "bfp6_expand_tile.h"
#include "ft_tensix_compute.h"
#include "ft_tensix_kernel_args.h"

#define EXPAND_STATUS_MAGIC 0x51364558u
/** Compile-time arg 0: 0 the real expander, 1 the RED build (datum << 1). */
#define BFP6_EXPAND_SHIFT (ft_tensix_compile_time_arg(0) == 1u ? 1u : 2u)

void kernel_main(void)
{
    const uint32_t layout = ft_tensix_runtime_arg(0);
    const uint32_t tiles = ft_tensix_runtime_arg(1);
    const uint32_t repeats = ft_tensix_runtime_arg(2);
    const uint32_t l1_in = ft_tensix_runtime_arg(3);
    const uint32_t l1_out = ft_tensix_runtime_arg(4);
    const uint32_t slots = ft_tensix_runtime_arg(5);
    volatile uint32_t *status = (volatile uint32_t *)(unsigned long)ft_tensix_runtime_arg(6);
    uint32_t slot = 0;

    if (tiles == 0u || slots == 0u) {
        status[3] = 0u;
        status[0] = EXPAND_STATUS_MAGIC;
        return;
    }
    status[1] = ft_bh_wall_clock_low();
    for (uint32_t r = 0; r < repeats; r++) {
        for (uint32_t t = 0; t < tiles; t++) {
            bfp6_expand_one_tile((const volatile uint32_t *)(unsigned long)(l1_in + t * BFP6_TILE_BYTES),
                                 (volatile uint32_t *)(unsigned long)(l1_out + slot * BFP8_TILE_BYTES), layout, BFP6_EXPAND_SHIFT);
            slot = slot + 1u == slots ? 0u : slot + 1u;
        }
    }
    status[2] = ft_bh_wall_clock_low();
    status[3] = repeats * tiles;
    status[0] = EXPAND_STATUS_MAGIC;
}
