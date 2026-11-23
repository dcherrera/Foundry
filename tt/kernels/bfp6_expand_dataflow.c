/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bfp6_expand_dataflow.c
 * @brief Route 1 of qwen36 spec §4.1b, the DATA-MOVER expander, on BRISC (NoC 0) or NCRISC
 *   (NoC 1): BFP6_B tiles in, BFP8_B tiles out, scalar SWAR (bfp6_expand_tile.h). Three modes,
 *   one binary:
 *     0  L1 -> L1      the expansion alone: `repeats` passes over tiles already in L1
 *     1  DRAM -> L1    the route as it would run: `block`-tile reads, double-buffered by
 *                      transaction id (dram_bandwidth_reader.c's M4 shape), each block expanded
 *                      while the next is in flight
 *     2  verify        DRAM -> L1 -> expand -> DRAM, one tile at a time, for the byte compare
 *
 * The timed interval is wall clock around the loop, stamped into the status block (magic last).
 *
 * Runtime args: 0 mode  1 layout  2 tiles  3 repeats  4 l1_in  5 l1_out  6 out slots
 *   7 dram x  8 dram y  9 dram in base  10 out x  11 out y  12 out base  13 status  14 block
 * Status words: 0 magic 0x51364558 ("Q6EX")  1 start  2 end  3 tiles expanded
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4 price bench).
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "bfp6_expand_tile.h"
#include "ft_tensix_dataflow.h"

#define EXPAND_STATUS_MAGIC 0x51364558u
/** Compile-time arg 0: 0 the real expander, 1 the RED build (datum << 1). */
#define BFP6_EXPAND_SHIFT (ft_tensix_compile_time_arg(0) == 1u ? 1u : 2u)

static void expand_block(uint32_t in, uint32_t out, uint32_t *slot, uint32_t slots, uint32_t tiles,
                         uint32_t layout)
{
    for (uint32_t t = 0; t < tiles; t++) {
        bfp6_expand_one_tile((const volatile uint32_t *)(unsigned long)(in + t * BFP6_TILE_BYTES),
                             (volatile uint32_t *)(unsigned long)(out + *slot * BFP8_TILE_BYTES), layout,
                             BFP6_EXPAND_SHIFT);
        *slot = *slot + 1u == slots ? 0u : *slot + 1u;
    }
}

/** Mode 1: two `block`-tile buffers at l1_in, read k+1 in flight while k expands. */
static uint32_t run_pipelined(uint32_t layout, uint32_t tiles, uint32_t l1_in, uint32_t l1_out,
                              uint32_t slots, uint64_t source, uint32_t block)
{
    uint32_t blocks = tiles / block, slot = 0, trid[2] = {1u, 2u};

    ft_tensix_noc_read_set_trid(trid[0]);
    ft_tensix_noc_read(source, l1_in, block * BFP6_TILE_BYTES);
    for (uint32_t b = 0; b < blocks; b++) {
        uint32_t here = l1_in + (b & 1u) * block * BFP6_TILE_BYTES;

        if (b + 1u < blocks) {
            ft_tensix_noc_read_set_trid(trid[(b + 1u) & 1u]);
            ft_tensix_noc_read(source + (uint64_t)(b + 1u) * block * BFP6_TILE_BYTES,
                               l1_in + ((b + 1u) & 1u) * block * BFP6_TILE_BYTES, block * BFP6_TILE_BYTES);
        }
        ft_tensix_noc_read_barrier_trid(trid[b & 1u]);
        expand_block(here, l1_out, &slot, slots, block, layout);
    }
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_read_set_trid(0u);
    return blocks * block;
}

/** Mode 2: every tile read, expanded and written back, so the host can compare every byte. */
static uint32_t run_verify(uint32_t layout, uint32_t tiles, uint32_t l1_in, uint32_t l1_out, uint64_t source,
                           uint64_t destination)
{
    uint32_t slot = 0;

    for (uint32_t t = 0; t < tiles; t++) {
        ft_tensix_noc_read(source + (uint64_t)t * BFP6_TILE_BYTES, l1_in, BFP6_TILE_BYTES);
        ft_tensix_noc_read_barrier();
        expand_block(l1_in, l1_out, &slot, 1u, 1u, layout);
        ft_tensix_noc_write(l1_out, destination + (uint64_t)t * BFP8_TILE_BYTES, BFP8_TILE_BYTES);
        ft_tensix_noc_write_barrier();
    }
    return tiles;
}

void kernel_main(void)
{
    const uint32_t mode = ft_tensix_runtime_arg(0);
    const uint32_t layout = ft_tensix_runtime_arg(1);
    const uint32_t tiles = ft_tensix_runtime_arg(2);
    const uint32_t repeats = ft_tensix_runtime_arg(3);
    const uint32_t l1_in = ft_tensix_runtime_arg(4);
    const uint32_t l1_out = ft_tensix_runtime_arg(5);
    const uint32_t slots = ft_tensix_runtime_arg(6);
    const uint64_t source = ft_tensix_noc_address(ft_tensix_runtime_arg(7), ft_tensix_runtime_arg(8),
                                                  ft_tensix_runtime_arg(9));
    const uint64_t destination = ft_tensix_noc_address(ft_tensix_runtime_arg(10), ft_tensix_runtime_arg(11),
                                                       ft_tensix_runtime_arg(12));
    volatile uint32_t *status = (volatile uint32_t *)(unsigned long)ft_tensix_runtime_arg(13);
    const uint32_t block = ft_tensix_runtime_arg(14);
    uint32_t done = 0, slot = 0;

    status[1] = ft_bh_wall_clock_low();
    if (mode == 0u) {
        for (uint32_t r = 0; r < repeats; r++)
            expand_block(l1_in, l1_out, &slot, slots, tiles, layout);
        done = repeats * tiles;
    } else if (mode == 1u && block != 0u) {
        for (uint32_t r = 0; r < repeats; r++)
            done += run_pipelined(layout, tiles, l1_in, l1_out, slots, source, block);
    } else if (mode == 2u) {
        done = run_verify(layout, tiles, l1_in, l1_out, source, destination);
    }
    status[2] = ft_bh_wall_clock_low();
    ft_tensix_noc_write_barrier();
    status[3] = done;
    status[0] = EXPAND_STATUS_MAGIC;
}
