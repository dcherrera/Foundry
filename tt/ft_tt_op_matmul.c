/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_matmul.c
 * @brief Implementation of the matmul host driver. See ft_tt_op_matmul.h, which carries
 *   the orientation decision this file assumes.
 *
 * @status Runs every MiniCPM5-1B shape on the card, at any prefetch depth.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_matmul.h"

#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"

/** CB indices, matching the compile-time args the triad was built with (0,1,16). */
#define MATMUL_CB_IN0 0u
#define MATMUL_CB_IN1 1u
#define MATMUL_CB_OUT 16u

/** Pages per input CB, as a multiple of the prefetch depth. TWO blocks: the reader fills
 *  one while the compute thread consumes the other, which is the whole point of a depth.
 *  It is also the invariant that keeps a reserved block contiguous — see
 *  ft_tt_op_matmul_prefetch_depth(). The output CB keeps a plain two pages. */
#define MATMUL_CB_BLOCKS  2u
#define MATMUL_CB_PAGES   2u

uint32_t ft_tt_op_matmul_prefetch_depth(uint32_t tile_inner, uint32_t requested)
{
    return ft_tt_op_block_depth(tile_inner, requested);
}

int ft_tt_op_matmul_run(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                        const ft_tt_kernel_program *program,
                        const ft_tt_op_matmul_request *request, uint32_t timeout_ms)
{
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    ft_tt_launch_cb_entry placed[3];
    ft_tt_op_cb_request requests[3];
    ft_tt_launch_config config;
    ft_tt_op_args args[5];
    uint32_t reader_args[11];
    uint32_t writer_args[4];
    uint32_t compute_args[3];
    uint32_t depth;
    int rc;

    if (!noc || !program || !request)
        return FT_TT_EINVAL;
    if (request->tile_rows == 0u || request->tile_inner == 0u || request->tile_cols == 0u)
        return FT_TT_EINVAL;

    depth = ft_tt_op_matmul_prefetch_depth(request->tile_inner, request->prefetch_tiles);

    rc = ft_tt_op_program_load(noc, x, y, program, NULL, &config);
    if (rc != FT_TT_OK)
        return rc;

    requests[0].index = MATMUL_CB_IN0;
    requests[0].page_bytes = request->activation.tile_bytes;
    requests[0].page_count = depth * MATMUL_CB_BLOCKS;
    requests[1].index = MATMUL_CB_IN1;
    requests[1].page_bytes = request->weight.tile_bytes;
    requests[1].page_count = depth * MATMUL_CB_BLOCKS;
    requests[2].index = MATMUL_CB_OUT;
    requests[2].page_bytes = request->output.tile_bytes;
    requests[2].page_count = MATMUL_CB_PAGES;

    rc = ft_tt_op_cb_place(requests, 3u, placed, cb_buffer, sizeof(cb_buffer),
                           &config.local_cb_mask);
    if (rc != FT_TT_OK)
        return rc;
    config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    /* The reader is on BRISC and was built with NOC_INDEX=0; the firmware re-reads this
     * from the launch message at run time, so the two must agree. */
    config.brisc_noc_id = 0u;

    reader_args[0] = request->tile_rows;
    reader_args[1] = request->tile_inner;
    reader_args[2] = request->tile_cols;
    reader_args[3] = request->activation.x;
    reader_args[4] = request->activation.y;
    reader_args[5] = request->activation.address;
    reader_args[6] = request->weight.x;
    reader_args[7] = request->weight.y;
    reader_args[8] = request->weight.address;
    reader_args[9] = depth;
    reader_args[10] = request->weight_n_major ? 1u : 0u;

    writer_args[0] = request->tile_rows * request->tile_cols;
    writer_args[1] = request->output.x;
    writer_args[2] = request->output.y;
    writer_args[3] = request->output.address;

    compute_args[0] = request->tile_rows;
    compute_args[1] = request->tile_inner;
    compute_args[2] = request->tile_cols;

    args[0].processor = FT_TT_PROC_DM0;
    args[0].values = reader_args;
    args[0].count = 11u;
    args[1].processor = FT_TT_PROC_DM1;
    args[1].values = writer_args;
    args[1].count = 4u;
    args[2].processor = FT_TT_PROC_MATH0;
    args[2].values = compute_args;
    args[2].count = 3u;
    args[3] = args[2];
    args[3].processor = FT_TT_PROC_MATH1;
    args[4] = args[2];
    args[4].processor = FT_TT_PROC_MATH2;

    return ft_tt_op_launch(noc, x, y, &config, cb_buffer, sizeof(cb_buffer), args, 5u, timeout_ms,
                           NULL);
}
