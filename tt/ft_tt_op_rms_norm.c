/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_rms_norm.c
 * @brief Implementation of the RMS-norm host driver. See ft_tt_op_rms_norm.h.
 *
 * @status Runs at MiniCPM5-1B's 1536-wide shape on the card.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_rms_norm.h"

#include <string.h>

#include "convert/ft_float_convert.h"
#include "ft_tt_device.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_tilize.h"

/** CB indices, matching the compile-time args the triad was built with. */
#define RMS_CB_IN     0u
#define RMS_CB_GAMMA  1u
#define RMS_CB_SCALER 2u
#define RMS_CB_EPS    3u
#define RMS_CB_XX     4u
#define RMS_CB_SCALE  5u
#define RMS_CB_XG     6u
#define RMS_CB_OUT    16u

/** How many CBs the request places. */
#define RMS_CB_COUNT 8u

/** Elements in one 32x32 tile. */
#define RMS_TILE_ELEMENTS (FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM)

int ft_tt_op_rms_norm_build_constants(uint32_t width, float eps, void *out, size_t out_bytes)
{
    uint16_t plain[RMS_TILE_ELEMENTS];
    unsigned char *bytes = out;
    uint16_t scaler;
    uint16_t eps_value;
    size_t index;

    if (!out || width == 0u || out_bytes < 2u * FT_TT_OP_TILE_BYTES_BF16)
        return FT_TT_EINVAL;

    /* The reduce multiplies every row sum by this, so 1/W turns a sum of squares into a
     * mean square. It only has to be present in the first row of each face, which is what
     * upstream's generate_reduce_scaler writes — filling the whole tile is a superset and
     * costs nothing, because only those positions are read. */
    scaler = ft_bf16_from_float(1.0f / (float)width);
    for (index = 0; index < RMS_TILE_ELEMENTS; index++)
        plain[index] = scaler;
    if (ft_tt_tilize_strip(plain, sizeof(uint16_t), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                           FT_TT_OP_TILE_DIM * sizeof(uint16_t), bytes) != 0)
        return FT_TT_EINVAL;

    /* One extra reduce over this tile adds exactly eps to every row's mean square:
     * 32 columns * (eps*W/32) * (1/W) = eps. */
    eps_value = ft_bf16_from_float(eps * (float)width / (float)FT_TT_OP_TILE_DIM);
    for (index = 0; index < RMS_TILE_ELEMENTS; index++)
        plain[index] = eps_value;
    if (ft_tt_tilize_strip(plain, sizeof(uint16_t), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                           FT_TT_OP_TILE_DIM * sizeof(uint16_t),
                           bytes + FT_TT_OP_TILE_BYTES_BF16) != 0)
        return FT_TT_EINVAL;

    return FT_TT_OK;
}

int ft_tt_op_rms_norm_run(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                          const ft_tt_kernel_program *program,
                          const ft_tt_op_rms_norm_request *request, uint32_t timeout_ms)
{
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    ft_tt_launch_cb_entry placed[RMS_CB_COUNT];
    ft_tt_op_cb_request requests[RMS_CB_COUNT];
    ft_tt_launch_config config;
    ft_tt_op_args args[5];
    uint32_t reader_args[11];
    uint32_t writer_args[4];
    uint32_t compute_args[2];
    uint32_t page;
    int rc;

    if (!noc || !program || !request)
        return FT_TT_EINVAL;
    if (request->row_blocks == 0u || request->tile_cols == 0u)
        return FT_TT_EINVAL;

    rc = ft_tt_op_program_load(noc, x, y, program, NULL, &config);
    if (rc != FT_TT_OK)
        return rc;

    /* A whole row block has to be resident twice over: once to square and reduce, once to
     * scale. So cb_in, cb_gamma and cb_xx are all tile_cols deep. The rest are one or two
     * pages because they are produced and consumed a tile at a time. */
    page = request->activation.tile_bytes;
    requests[0].index = RMS_CB_IN;
    requests[0].page_bytes = page;
    requests[0].page_count = request->tile_cols;
    requests[1].index = RMS_CB_GAMMA;
    requests[1].page_bytes = request->gamma.tile_bytes;
    requests[1].page_count = request->tile_cols;
    requests[2].index = RMS_CB_SCALER;
    requests[2].page_bytes = page;
    requests[2].page_count = 1u;
    requests[3].index = RMS_CB_EPS;
    requests[3].page_bytes = page;
    requests[3].page_count = 1u;
    requests[4].index = RMS_CB_XX;
    requests[4].page_bytes = page;
    requests[4].page_count = request->tile_cols;
    requests[5].index = RMS_CB_SCALE;
    requests[5].page_bytes = page;
    requests[5].page_count = 1u;
    requests[6].index = RMS_CB_XG;
    requests[6].page_bytes = page;
    requests[6].page_count = 1u;
    requests[7].index = RMS_CB_OUT;
    requests[7].page_bytes = request->output.tile_bytes;
    requests[7].page_count = 2u;

    rc = ft_tt_op_cb_place(requests, RMS_CB_COUNT, placed, cb_buffer, sizeof(cb_buffer),
                           &config.local_cb_mask);
    if (rc != FT_TT_OK)
        return rc;
    config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    config.brisc_noc_id = 0u;

    reader_args[0] = request->row_blocks;
    reader_args[1] = request->tile_cols;
    reader_args[2] = request->activation.x;
    reader_args[3] = request->activation.y;
    reader_args[4] = request->activation.address;
    reader_args[5] = request->gamma.x;
    reader_args[6] = request->gamma.y;
    reader_args[7] = request->gamma.address;
    reader_args[8] = request->constants.x;
    reader_args[9] = request->constants.y;
    reader_args[10] = request->constants.address;

    writer_args[0] = request->row_blocks * request->tile_cols;
    writer_args[1] = request->output.x;
    writer_args[2] = request->output.y;
    writer_args[3] = request->output.address;

    compute_args[0] = request->row_blocks;
    compute_args[1] = request->tile_cols;

    args[0].processor = FT_TT_PROC_DM0;
    args[0].values = reader_args;
    args[0].count = 11u;
    args[1].processor = FT_TT_PROC_DM1;
    args[1].values = writer_args;
    args[1].count = 4u;
    args[2].processor = FT_TT_PROC_MATH0;
    args[2].values = compute_args;
    args[2].count = 2u;
    args[3] = args[2];
    args[3].processor = FT_TT_PROC_MATH1;
    args[4] = args[2];
    args[4].processor = FT_TT_PROC_MATH2;

    return ft_tt_op_launch(noc, x, y, &config, cb_buffer, sizeof(cb_buffer), args, 5u, timeout_ms,
                           NULL);
}
