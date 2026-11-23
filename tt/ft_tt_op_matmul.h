/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_matmul.h
 * @brief Host driver for the matmul triad — the load-bearing op (spec §5.1 fc_sgemm).
 *   Places the circular buffers, plants the shapes, kicks the core, waits.
 *
 * ---------------------------------------------------------------------------
 * THE ORIENTATION DECISION — Phase 3 follow-up, settled here
 * ---------------------------------------------------------------------------
 * The question was: "the converter stores GGUF's dims[1] as rows and dims[0] as columns,
 * untransposed. Does the Tensix matmul want the weight transposed?" — and it had to be
 * answered before artifacts proliferate.
 *
 * **Decision: activations are [tokens x in_features] row-major, and a 2-D matmul weight is
 * stored as K x N = [in_features x out_features], which is the TRANSPOSE of what GGUF and
 * the current converter store.** The converter should transpose 2-D weights before
 * tilizing them; nothing else changes.
 *
 * Four reasons, in the order that decided it:
 *
 *  1. **Block float forces the transpose upstream of the packer.** A Bfp8_b tile shares
 *     one exponent across 16 elements ALONG A ROW. Transposing a packed tile therefore
 *     means unpacking and repacking it — so a loader-side or artifact-side transpose is
 *     not a byte shuffle, it is conversion work. The only place a transpose is free is
 *     before ft_bfp packs, i.e. inside the converter. This one is decisive: it rules out
 *     the option that would have avoided touching the converter.
 *  2. **It keeps the runtime path the textbook one.** in0 -> SrcB, in1 -> SrcA, C[m][n] +=
 *     A[m][k] * B[k][n] with A the activation and B the weight, which is what every TT
 *     reader, every reference kernel and `matmul_single_core` do. Ops 4-40 copy this; a
 *     bespoke orientation would be re-derived forty times.
 *  3. **The alternative is a whole-model commitment, not a local one.** Keeping GGUF's
 *     [out, in] means computing y^T = W @ x^T, i.e. activations stored [features x tokens]
 *     for the entire model — RMS norm reduces along columns, softmax transposes, and every
 *     op inherits it. Tile waste is symmetric either way (a 1-token decode wastes 31/32
 *     of a tile in whichever axis), so there is nothing to buy with that.
 *  4. **`matmul_init(transpose=1)` was the tempting third option and it is not free
 *     enough.** It would transpose in1 in the unpacker and let the reader index the tile
 *     grid the other way round, costing zero conversion — but it puts an unvalidated
 *     hardware mode on the critical path of the one op everything depends on, to save work
 *     that happens once, offline, on the Mac, in seven seconds.
 *
 * Phase 4 tests transpose host-side before uploading, so the kernels are exercised in the
 * decided orientation without the converter changing under another agent's feet. The
 * converter change is a plan item.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS DRIVER DOES NOT DO
 * ---------------------------------------------------------------------------
 * Single core, and every A tile is re-read Nt times and every B tile Mt times. That is the
 * blocked outer product's cost, it is known, and Phase 4's acceptance is correctness. Reuse
 * — multi-core with an in0 broadcast, or a resident row block — is Phase 5/8.
 *
 * @status PCC and shape coverage in docs/tt_bringup_log.md §14; the prefetch-depth work
 *   and the fidelity decision are §16. 468.8 GFLOP/s on one core at depth 48.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_MATMUL_H
#define FT_TT_OP_MATMUL_H

#include <stdint.h>

#include "ft_tt_kernel_blob.h"
#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"

/* ft_tt_op_operand moved to ft_tt_op_dispatch.h when the vector op family needed the same
 * three fields — every op driver needs it and none of them owns it. */

/** One matmul: C[Mt*32 x Nt*32] = A[Mt*32 x Kt*32] * B[Kt*32 x Nt*32], in tiles. */
typedef struct ft_tt_op_matmul_request {
    uint32_t tile_rows;  /**< Mt */
    uint32_t tile_inner; /**< Kt */
    uint32_t tile_cols;  /**< Nt */
    /** How many K-tiles the reader fetches per NoC barrier. 0 or 1 is the per-pair
     *  barrier Phase 4 measured at 849 ns a tile-pair; higher pipelines the reads behind
     *  the math. Shrunk to a divisor of tile_inner by
     *  ft_tt_op_matmul_prefetch_depth() — see there for why a ragged block is illegal. */
    uint32_t prefetch_tiles;
    /** How the WEIGHT's tile grid is ordered in DRAM. 0 = K-major (tile index kt*Nt + nt),
     *  which is the row-major order the converter emitted first; 1 = N-major
     *  (nt*Kt + kt), which makes a prefetch block's B tiles consecutive so the reader can
     *  take them in ONE transaction instead of `depth`. THIS IS A STATEMENT ABOUT DRAM,
     *  not a request: both index formulas are in range, so a wrong value here is a wrong
     *  answer with no error anywhere. It must match the artifact's `layout` field. */
    uint32_t weight_n_major;
    ft_tt_op_operand activation;
    ft_tt_op_operand weight;
    ft_tt_op_operand output;
} ft_tt_op_matmul_request;

/**
 * The prefetch depth this request will actually run at: ft_tt_op_block_depth() over Kt,
 * because the matmul's K loop restarts per output tile and a block may not span two of
 * them. See that function for why a ragged block is illegal rather than merely wasteful.
 *
 * Kept as a named matmul entry point because the caller thinks in Kt, not in "the run".
 *
 * @param tile_inner  Kt
 * @param requested   what the caller asked for; 0 means 1
 * @return            the effective depth, always >= 1 and always dividing tile_inner
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_op_matmul_prefetch_depth(uint32_t tile_inner, uint32_t requested);

/**
 * Run one matmul on a core whose firmware is already booted.
 *
 * The three CBs are placed in the arena at the request's own page sizes. The two input CBs
 * get TWO BLOCKS of `prefetch_tiles` pages, so the reader can fill one block while the
 * compute thread consumes the other; the output CB keeps two pages. The weight CB's page
 * size is where a block-float operand differs from a bf16 one, and the only place.
 *
 * @param program     a RELOCATE_KERNEL_TEXT=1 build of matmul_{reader,compute}.c +
 *                    vector_add_writer.c, whose compile-time tile sizes must match the
 *                    request's. Nothing checks that: a data-movement kernel has no
 *                    descriptor header, so its tile size is a compile-time arg
 * @param timeout_ms  bounds the wait; there is no untimed path (spec §7.3)
 * @return            FT_TT_OK; FT_TT_EIO on timeout with the core dumped; FT_TT_EINVAL
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_op_matmul_run(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                        const ft_tt_kernel_program *program,
                        const ft_tt_op_matmul_request *request, uint32_t timeout_ms);

#endif /* FT_TT_OP_MATMUL_H */
