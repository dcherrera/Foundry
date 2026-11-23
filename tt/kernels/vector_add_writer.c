/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file vector_add_writer.c
 * @brief Writer half of Foundry's vector-add triad. Drains the output circular buffer to a
 *   destination buffer, one tile at a time.
 *
 * C99, on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 *
 * THE WIDEST-REACH FILE IN tt/kernels/. It is the writer of `rms_norm`, `matmul`,
 * `matmul_bfp8`, `matmul_bfp4`, all four fidelity arms, `resident`, `oneshot` and the three
 * phase-2 vector_add programs — eleven program directories. A change here obliges rebuilding
 * all of them.
 *
 * Runs on the OTHER data-movement core from the reader, on the OTHER NoC. Which of
 * BRISC/NCRISC is "reader" is arbitrary; what matters is that the two use different NoCs so
 * they do not contend (docs/tt_kernel_model.md §1).
 *
 * Compile-time args:
 *   0: cb_out index
 *   1: tile size in bytes
 * Runtime args:
 *   0: number of tiles
 *   1: destination NoC x    2: destination NoC y   3: destination base address
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=16,2048 NOC=1 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/vector_add_writer.c ncrisc /tmp/ttk_vector_add
 *
 * Unlike BRISC, NCRISC's NoC is not named anywhere in the launch message: the NCRISC
 * firmware does no NoC setup at all, and the kernel's own _start calls
 * noc_local_state_init(NOC_INDEX). So NCRISC's NoC is whatever this was
 * compiled with, full stop.
 *
 * @status None
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
    const uint32_t destination_x = ft_tensix_runtime_arg(1);
    const uint32_t destination_y = ft_tensix_runtime_arg(2);
    const uint32_t destination_base = ft_tensix_runtime_arg(3);

    for (uint32_t tile = 0; tile < tile_count; tile++) {
        ft_tensix_cb_wait_front(cb_out, 1);
        ft_tensix_noc_write(ft_tensix_cb_read_pointer(cb_out),
                            ft_tensix_noc_address(destination_x, destination_y,
                                                  destination_base + tile * tile_bytes),
                            tile_bytes);
        /* Barrier per tile rather than a flush plus one barrier at the end: correctness
         * first for a bring-up kernel, latency later (docs/tt_kernel_model.md §1). */
        ft_tensix_noc_write_barrier();
        ft_tensix_cb_pop_front(cb_out, 1);
    }
}
