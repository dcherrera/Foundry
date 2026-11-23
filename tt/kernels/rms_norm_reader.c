/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file rms_norm_reader.c
 * @brief Reader third of Foundry's RMS-norm triad. Pushes the three resident operands once —
 *   the 1/W scaler tile, the eps tile, and the Wt expanded gamma tiles — and then streams Wt
 *   activation tiles per row block.
 *
 * C99, on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 *
 * "Once" is load-bearing. The compute kernel waits on the scaler, eps and gamma CBs before
 * its block loop and never pops them, so they stay in front for every block. Re-pushing them
 * per block would deadlock against a CB that is never drained.
 *
 * The scaler and eps tiles are adjacent in one two-tile constant buffer, so they cost one
 * address rather than two.
 *
 * Compile-time args:
 *   0: cb_in index   1: cb_gamma index   2: cb_scaler index   3: cb_eps index
 *   4: tile size in bytes
 * Runtime args:
 *   0: row block count   1: Wt
 *   2: x NoC x        3: x NoC y        4: x base address
 *   5: gamma NoC x    6: gamma NoC y    7: gamma base address
 *   8: const NoC x    9: const NoC y   10: const base address (scaler tile, then eps)
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,2,3,2048 NOC=0 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/rms_norm_reader.c brisc /tmp/ttk_rms_norm
 *
 * @status See docs/tt_bringup_log.md §14.
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_dataflow.h"

void kernel_main(void)
{
    const uint32_t cb_in = ft_tensix_compile_time_arg(0);
    const uint32_t cb_gamma = ft_tensix_compile_time_arg(1);
    const uint32_t cb_scaler = ft_tensix_compile_time_arg(2);
    const uint32_t cb_eps = ft_tensix_compile_time_arg(3);
    const uint32_t tile_bytes = ft_tensix_compile_time_arg(4);

    const uint32_t row_blocks = ft_tensix_runtime_arg(0);
    const uint32_t tile_cols = ft_tensix_runtime_arg(1);
    const uint32_t x_x = ft_tensix_runtime_arg(2);
    const uint32_t x_y = ft_tensix_runtime_arg(3);
    const uint32_t x_base = ft_tensix_runtime_arg(4);
    const uint32_t gamma_x = ft_tensix_runtime_arg(5);
    const uint32_t gamma_y = ft_tensix_runtime_arg(6);
    const uint32_t gamma_base = ft_tensix_runtime_arg(7);
    const uint32_t const_x = ft_tensix_runtime_arg(8);
    const uint32_t const_y = ft_tensix_runtime_arg(9);
    const uint32_t const_base = ft_tensix_runtime_arg(10);

    ft_tensix_cb_reserve_back(cb_scaler, 1);
    ft_tensix_cb_reserve_back(cb_eps, 1);
    ft_tensix_noc_read(ft_tensix_noc_address(const_x, const_y, const_base),
                       ft_tensix_cb_write_pointer(cb_scaler), tile_bytes);
    ft_tensix_noc_read(ft_tensix_noc_address(const_x, const_y, const_base + tile_bytes),
                       ft_tensix_cb_write_pointer(cb_eps), tile_bytes);
    ft_tensix_noc_read_barrier();
    ft_tensix_cb_push_back(cb_scaler, 1);
    ft_tensix_cb_push_back(cb_eps, 1);

    for (uint32_t wt = 0; wt < tile_cols; wt++) {
        ft_tensix_cb_reserve_back(cb_gamma, 1);
        ft_tensix_noc_read(ft_tensix_noc_address(gamma_x, gamma_y, gamma_base + wt * tile_bytes),
                           ft_tensix_cb_write_pointer(cb_gamma), tile_bytes);
        ft_tensix_noc_read_barrier();
        ft_tensix_cb_push_back(cb_gamma, 1);
    }

    for (uint32_t block = 0; block < row_blocks; block++) {
        for (uint32_t wt = 0; wt < tile_cols; wt++) {
            const uint32_t tile = block * tile_cols + wt;

            ft_tensix_cb_reserve_back(cb_in, 1);
            ft_tensix_noc_read(ft_tensix_noc_address(x_x, x_y, x_base + tile * tile_bytes),
                               ft_tensix_cb_write_pointer(cb_in), tile_bytes);
            ft_tensix_noc_read_barrier();
            ft_tensix_cb_push_back(cb_in, 1);
        }
    }
}
