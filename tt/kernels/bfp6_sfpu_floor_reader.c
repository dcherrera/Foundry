/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bfp6_sfpu_floor_reader.c
 * @brief Reader of the route-2 floor bench (bfp6_sfpu_floor_compute.c): `tiles` input tiles
 *   from one DRAM bank into cb_in, `depth` per read and barrier, wall clock at entry into
 *   status word 0 (bfp_split_matmul_writer.c stamps word 1).
 *
 * Compile-time args: 0 cb_in  1 tile bytes
 * Runtime args: 0 tiles  1 x  2 y  3 base  4 depth  5 status L1 address
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4 price bench).
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_dataflow.h"

void kernel_main(void)
{
    const uint32_t cb_in = ft_tensix_compile_time_arg(0);
    const uint32_t tile_bytes = ft_tensix_compile_time_arg(1);
    const uint32_t tiles = ft_tensix_runtime_arg(0);
    const uint32_t depth = ft_tensix_runtime_arg(4);
    volatile uint32_t *status = (volatile uint32_t *)(unsigned long)ft_tensix_runtime_arg(5);

    status[0] = ft_bh_wall_clock_low();
    for (uint32_t t = 0; t < tiles; t += depth) {
        ft_tensix_cb_reserve_back(cb_in, depth);
        ft_tensix_noc_read(ft_tensix_noc_address(ft_tensix_runtime_arg(1), ft_tensix_runtime_arg(2),
                                                 ft_tensix_runtime_arg(3) + t * tile_bytes),
                           ft_tensix_cb_write_pointer(cb_in), depth * tile_bytes);
        ft_tensix_noc_read_barrier();
        ft_tensix_cb_push_back(cb_in, depth);
    }
}
