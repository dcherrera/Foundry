/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file vector_add_reader.c
 * @brief Reader half of Foundry's vector-add triad. Streams tile pairs from two source
 *   buffers into the two input circular buffers the compute kernel consumes.
 *
 * C99, on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6). The
 * C++ predecessor's `.text` is the acceptance baseline: this must emit the same bytes.
 *
 * Deliberately NOT built on TensorAccessor. Upstream's examples derive NoC addresses from a
 * compile-time buffer layout appended host-side by TensorAccessorArgs
 * (docs/tt_kernel_model.md §6), which drags tt-metal's buffer abstraction into Foundry's
 * host code. Foundry already knows exactly where it put every buffer, so the addresses
 * arrive as runtime args and the kernel builds the NoC address itself.
 *
 * Compile-time args:
 *   0: cb_in0 index
 *   1: cb_in1 index
 *   2: tile size in bytes (2048 for a 32x32 Float16_b tile)
 * Runtime args:
 *   0: number of tiles
 *   1: source-0 NoC x       2: source-0 NoC y      3: source-0 base address
 *   4: source-1 NoC x       5: source-1 NoC y      6: source-1 base address
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,2048 NOC=0 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/vector_add_reader.c brisc /tmp/ttk_vector_add
 *
 * NOC_INDEX is fixed at compile time, but the firmware picks BRISC's NoC at run time from
 * the launch message's brisc_noc_id. The host must pass the same value this
 * was built with, or the kernel issues on a NoC whose command buffers the firmware never
 * initialised.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_dataflow.h"

void kernel_main(void)
{
    const uint32_t cb_in0 = ft_tensix_compile_time_arg(0);
    const uint32_t cb_in1 = ft_tensix_compile_time_arg(1);
    const uint32_t tile_bytes = ft_tensix_compile_time_arg(2);

    const uint32_t tile_count = ft_tensix_runtime_arg(0);
    const uint32_t source0_x = ft_tensix_runtime_arg(1);
    const uint32_t source0_y = ft_tensix_runtime_arg(2);
    const uint32_t source0_base = ft_tensix_runtime_arg(3);
    const uint32_t source1_x = ft_tensix_runtime_arg(4);
    const uint32_t source1_y = ft_tensix_runtime_arg(5);
    const uint32_t source1_base = ft_tensix_runtime_arg(6);

    for (uint32_t tile = 0; tile < tile_count; tile++) {
        const uint32_t byte_offset = tile * tile_bytes;

        ft_tensix_cb_reserve_back(cb_in0, 1);
        ft_tensix_cb_reserve_back(cb_in1, 1);

        ft_tensix_noc_read(ft_tensix_noc_address(source0_x, source0_y, source0_base + byte_offset),
                           ft_tensix_cb_write_pointer(cb_in0), tile_bytes);
        ft_tensix_noc_read(ft_tensix_noc_address(source1_x, source1_y, source1_base + byte_offset),
                           ft_tensix_cb_write_pointer(cb_in1), tile_bytes);
        /* One barrier for both reads: it is per-core, not per-transaction. */
        ft_tensix_noc_read_barrier();

        ft_tensix_cb_push_back(cb_in0, 1);
        ft_tensix_cb_push_back(cb_in1, 1);
    }
}
