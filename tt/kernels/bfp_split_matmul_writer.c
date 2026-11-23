/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bfp_split_matmul_writer.c
 * @brief Writer of the qwen36 plan 1.4b price bench: vector_add_writer.c's loop (one output
 *   tile at a time to DRAM), then the wall clock into status word 1 once the last write has
 *   landed — the end of the measured interval the reader started.
 *
 * Compile-time args: 0 cb_out  1 tile bytes
 * Runtime args: 0 tile count  1 x  2 y  3 base  4 status L1 address
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4b price bench).
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_dataflow.h"

void kernel_main(void)
{
    const uint32_t cb_out = ft_tensix_compile_time_arg(0);
    const uint32_t tile_bytes = ft_tensix_compile_time_arg(1);
    const uint32_t tile_count = ft_tensix_runtime_arg(0);
    const uint32_t x = ft_tensix_runtime_arg(1);
    const uint32_t y = ft_tensix_runtime_arg(2);
    const uint32_t base = ft_tensix_runtime_arg(3);
    volatile uint32_t *status = (volatile uint32_t *)(unsigned long)ft_tensix_runtime_arg(4);

    for (uint32_t tile = 0; tile < tile_count; tile++) {
        ft_tensix_cb_wait_front(cb_out, 1);
        ft_tensix_noc_write(ft_tensix_cb_read_pointer(cb_out),
                            ft_tensix_noc_address(x, y, base + tile * tile_bytes), tile_bytes);
        ft_tensix_noc_write_barrier();
        ft_tensix_cb_pop_front(cb_out, 1);
    }
    status[1] = ft_bh_wall_clock_low();
}
