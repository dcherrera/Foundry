/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file matmul_reader.c
 * @brief Reader third of Foundry's matmul triad. Streams the (A tile, B tile) pairs the
 *   compute kernel consumes, in exactly its consumption order: for each output tile
 *   (mt, nt), the Kt pairs A[mt][kt], B[kt][nt].
 *
 * C99, on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 *
 * Both operands are addressed as a row-major grid of 32x32 tiles, so A's tile index is
 * mt*Kt + kt and B's is kt*Nt + nt. That is the whole of the layout contract — deliberately
 * NOT TensorAccessor, for the reason the vector-add reader gives: the host already knows
 * where it put every buffer, so addresses arrive as runtime args.
 *
 * An A tile is re-read Nt times and a B tile Mt times. That is the single-core blocked outer
 * product's cost and it is known: Phase 4's job is a correct op, and reuse (multi-core with
 * an in0 broadcast, or keeping a row block resident) is Phase 5/8's.
 *
 * PREFETCH DEPTH — the Phase 4 follow-up this kernel answers.
 * The first version issued two reads and a full read barrier PER TILE PAIR, so every pair
 * paid a whole DRAM round trip that nothing overlapped: 849 ns a pair of which 80 ns was
 * arithmetic. The barrier is per-core, not per-transaction, so the fix is to have more than
 * one transaction outstanding when it is taken: reserve a BLOCK of `depth` K-tiles, issue all
 * 2*depth reads, take ONE barrier, push the block. The latency is then paid once per block
 * instead of once per pair, and with 2*depth pages in each CB the compute thread chews on one
 * half while the reader fills the other.
 *
 * AND THE A SIDE OF A BLOCK IS ONE TRANSACTION, NOT `depth` OF THEM. Within a block the A
 * tiles are mt*Kt + kt .. kt+depth-1 — CONSECUTIVE tile indices, so consecutive bytes in DRAM
 * — and the CB block they land in is contiguous L1 by the invariant below. So the whole A
 * block is a single read of depth*in0_tile_bytes. Measured at depth 48 it is worth 27-37%
 * (365.9 -> 466.3 GFLOP/s on ffn_gate/up) for BYTE-IDENTICAL traffic and bit-identical PCC,
 * which is the evidence that what remains after the barrier fix is per-TRANSACTION issue cost
 * rather than bandwidth.
 *
 * THE B SIDE COALESCES TOO, BUT ONLY IF THE WEIGHT IS STORED N-MAJOR. In the K-major tile
 * grid the artifact shipped first, B's tile index is kt*Nt + nt, so a block's B tiles are
 * STRIDED by Nt and each one has to be its own transaction. Store the weight's tile grid
 * COLUMN-major instead (tile index nt*Kt + kt) and a block's B tiles become consecutive, so
 * the whole B block is one read as well, and the reader issues 2 transactions per block
 * instead of depth+1. Byte-identical traffic again; only the count changes.
 *
 * The layout is a RUNTIME arg, not a build, so both arms of that A/B are the SAME binary and
 * a layout measurement cannot be a build difference in disguise. It is the host's job to say
 * which layout the weight in DRAM actually has — a wrong value here is a wrong answer, not an
 * error, because both index formulas are in range.
 *
 * `depth` is a RUNTIME arg for the same reason the shapes are: one binary serves every depth,
 * and a rebuild is ~4 minutes on the card host. The host is responsible for two invariants and
 * ft_tt_op_matmul.c enforces both:
 *
 *   1. depth DIVIDES Kt          -- so every block is the same size
 *   2. CB page count == 2*depth  -- so a reserved block is contiguous
 *
 * Both are needed because reserving does not guarantee a contiguous block: the fifo write
 * pointer wraps only when it lands EXACTLY on the limit ("producer always writes into
 * contiguous memory, it cannot wrap"). Pushes of a constant `depth` into a 2*depth-page CB
 * land the pointer on 0 or depth and never straddle. A ragged final block would break that
 * for every later block, which is why the host shrinks depth to a divisor rather than
 * clamping it.
 *
 * THE TWO OPERANDS HAVE DIFFERENT TILE SIZES, which is the whole of Route B's cost on this
 * side: a Float16_b activation tile is 2048 B while a Bfp8_b weight tile is 1088 B and a
 * Bfp4_b one 576 B. A data-movement kernel has no `ft_tensix_cb_formats.h` to read a format
 * from, so both sizes are compile-time args and they MUST match the CB table the compute
 * third was built with.
 *
 * Compile-time args:
 *   0: cb_in0 index   1: cb_in1 index
 *   2: in0 tile size in bytes   3: in1 tile size in bytes
 * Runtime args:
 *   0: Mt   1: Kt   2: Nt
 *   3: A NoC x   4: A NoC y   5: A base address
 *   6: B NoC x   7: B NoC y   8: B base address
 *   9: prefetch depth in K-tiles (0 or 1 = the old per-pair barrier)
 *  10: B tile-grid layout — 0 = K-major (index kt*Nt + nt), 1 = N-major (nt*Kt + kt)
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,2048,2048 NOC=0 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/matmul_reader.c brisc /tmp/ttk_matmul
 *
 * @status See docs/tt_bringup_log.md §14 and §16.
 * @issues None
 * @todo None
 */

#include <stdint.h>

#include "ft_tensix_dataflow.h"

void kernel_main(void)
{
    const uint32_t cb_in0 = ft_tensix_compile_time_arg(0);
    const uint32_t cb_in1 = ft_tensix_compile_time_arg(1);
    const uint32_t in0_tile_bytes = ft_tensix_compile_time_arg(2);
    const uint32_t in1_tile_bytes = ft_tensix_compile_time_arg(3);

    const uint32_t tile_rows = ft_tensix_runtime_arg(0);
    const uint32_t tile_inner = ft_tensix_runtime_arg(1);
    const uint32_t tile_cols = ft_tensix_runtime_arg(2);
    const uint32_t a_x = ft_tensix_runtime_arg(3);
    const uint32_t a_y = ft_tensix_runtime_arg(4);
    const uint32_t a_base = ft_tensix_runtime_arg(5);
    const uint32_t b_x = ft_tensix_runtime_arg(6);
    const uint32_t b_y = ft_tensix_runtime_arg(7);
    const uint32_t b_base = ft_tensix_runtime_arg(8);
    const uint32_t requested_depth = ft_tensix_runtime_arg(9);
    const uint32_t b_n_major = ft_tensix_runtime_arg(10);

    /* A depth that does not divide Kt would make a later block straddle the CB wrap, so the
     * kernel refuses to guess: anything the host did not make legal collapses to the per-pair
     * behaviour, which is always correct. */
    const uint32_t depth =
        (requested_depth > 0u && tile_inner % requested_depth == 0u) ? requested_depth : 1u;

    for (uint32_t mt = 0; mt < tile_rows; mt++) {
        for (uint32_t nt = 0; nt < tile_cols; nt++) {
            for (uint32_t kt = 0; kt < tile_inner; kt += depth) {
                uint32_t a_write;
                uint32_t b_write;

                ft_tensix_cb_reserve_back(cb_in0, depth);
                ft_tensix_cb_reserve_back(cb_in1, depth);
                a_write = ft_tensix_cb_write_pointer(cb_in0);
                b_write = ft_tensix_cb_write_pointer(cb_in1);

                /* The block's A tiles are consecutive, so they are ONE read. */
                ft_tensix_noc_read(
                    ft_tensix_noc_address(a_x, a_y,
                                          a_base + (mt * tile_inner + kt) * in0_tile_bytes),
                    a_write, depth * in0_tile_bytes);

                if (b_n_major) {
                    /* Column-major tile grid: this block's B tiles are nt*Kt + kt ..
                     * +depth-1, consecutive in DRAM, so they are ONE read as well. */
                    ft_tensix_noc_read(
                        ft_tensix_noc_address(b_x, b_y,
                                              b_base + (nt * tile_inner + kt) * in1_tile_bytes),
                        b_write, depth * in1_tile_bytes);
                } else {
                    /* Row-major tile grid: strided by Nt, one transaction each. */
                    for (uint32_t step = 0; step < depth; step++) {
                        const uint32_t b_tile = (kt + step) * tile_cols + nt;

                        ft_tensix_noc_read(
                            ft_tensix_noc_address(b_x, b_y, b_base + b_tile * in1_tile_bytes),
                            b_write + step * in1_tile_bytes, in1_tile_bytes);
                    }
                }

                /* ONE barrier for the whole block: it is per-core, not per-transaction,
                 * which is exactly why depth converts. */
                ft_tensix_noc_read_barrier();
                ft_tensix_cb_push_back(cb_in0, depth);
                ft_tensix_cb_push_back(cb_in1, depth);
            }
        }
    }
}
