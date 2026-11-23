/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bfp_split_matmul_reader.c
 * @brief Reader of the qwen36 plan 1.4b price bench: for each output tile (mt, nt) it streams
 *   the HIGH pass (A[mt][kt], B_high[kt][nt]) then the LOW pass (A[mt][kt], B_low[kt][nt]),
 *   each pass's weight in its own CB (cb_in1 / cb_in2) so the two native formats never share
 *   a page size. `mode` picks the passes: 1 high only, 2 low only, 3 both.
 *
 * Weights are stored N-MAJOR (tile index nt*Kt + kt), so a block of `depth` B tiles is one
 * read, as matmul_reader.c's N-major arm. `depth` must divide Kt (the host makes it so) and
 * every CB holds 2*depth pages, matmul_reader.c's contiguity invariant.
 *
 * Wall clock at entry goes to status word 0 (the writer stamps the end), so the price is on
 * the device's own clock, outside the launch.
 *
 * Compile-time args: 0 cb_in0  1 cb_in1  2 cb_in2  3 in0 tile bytes  4 in1 tile bytes
 *                    5 in2 tile bytes
 * Runtime args: 0 Mt  1 Kt  2 Nt  3 A x  4 A y  5 A base  6 B1 x  7 B1 y  8 B1 base
 *               9 B2 x  10 B2 y  11 B2 base  12 depth  13 mode  14 status L1 address
 *
 * @status New 2026-10-28 (qwen36 Phase 1.4b price bench).
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_dataflow.h"

/** Stream one pass of Kt (A, B) pairs for output tile (mt, nt) into cb_a / cb_b. */
static void read_pass(uint32_t cb_a, uint32_t cb_b, uint32_t a_bytes, uint32_t b_bytes, uint32_t mt,
                      uint32_t nt, uint32_t tile_inner, uint32_t depth, uint32_t a_x, uint32_t a_y,
                      uint32_t a_base, uint32_t b_x, uint32_t b_y, uint32_t b_base)
{
    for (uint32_t kt = 0; kt < tile_inner; kt += depth) {
        ft_tensix_cb_reserve_back(cb_a, depth);
        ft_tensix_cb_reserve_back(cb_b, depth);
        ft_tensix_noc_read(ft_tensix_noc_address(a_x, a_y, a_base + (mt * tile_inner + kt) * a_bytes),
                           ft_tensix_cb_write_pointer(cb_a), depth * a_bytes);
        ft_tensix_noc_read(ft_tensix_noc_address(b_x, b_y, b_base + (nt * tile_inner + kt) * b_bytes),
                           ft_tensix_cb_write_pointer(cb_b), depth * b_bytes);
        ft_tensix_noc_read_barrier();
        ft_tensix_cb_push_back(cb_a, depth);
        ft_tensix_cb_push_back(cb_b, depth);
    }
}

void kernel_main(void)
{
    const uint32_t cb_in0 = ft_tensix_compile_time_arg(0);
    const uint32_t cb_in1 = ft_tensix_compile_time_arg(1);
    const uint32_t cb_in2 = ft_tensix_compile_time_arg(2);
    const uint32_t in0_bytes = ft_tensix_compile_time_arg(3);
    const uint32_t in1_bytes = ft_tensix_compile_time_arg(4);
    const uint32_t in2_bytes = ft_tensix_compile_time_arg(5);
    const uint32_t tile_rows = ft_tensix_runtime_arg(0);
    const uint32_t tile_inner = ft_tensix_runtime_arg(1);
    const uint32_t tile_cols = ft_tensix_runtime_arg(2);
    const uint32_t depth = ft_tensix_runtime_arg(12);
    const uint32_t mode = ft_tensix_runtime_arg(13);
    volatile uint32_t *status = (volatile uint32_t *)(unsigned long)ft_tensix_runtime_arg(14);

    status[0] = ft_bh_wall_clock_low();
    for (uint32_t mt = 0; mt < tile_rows; mt++) {
        for (uint32_t nt = 0; nt < tile_cols; nt++) {
            if (mode & 1u)
                read_pass(cb_in0, cb_in1, in0_bytes, in1_bytes, mt, nt, tile_inner, depth,
                          ft_tensix_runtime_arg(3), ft_tensix_runtime_arg(4), ft_tensix_runtime_arg(5),
                          ft_tensix_runtime_arg(6), ft_tensix_runtime_arg(7), ft_tensix_runtime_arg(8));
            if (mode & 2u)
                read_pass(cb_in0, cb_in2, in0_bytes, in2_bytes, mt, nt, tile_inner, depth,
                          ft_tensix_runtime_arg(3), ft_tensix_runtime_arg(4), ft_tensix_runtime_arg(5),
                          ft_tensix_runtime_arg(9), ft_tensix_runtime_arg(10), ft_tensix_runtime_arg(11));
        }
    }
}
