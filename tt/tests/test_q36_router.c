/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_q36_router.c
 * @brief qwen36 plan Phase 5.3 on the card: the MoE ROUTER program (`rtriad_router`) — the
 *   router matmul at fp32 output plus the routing step in its writer (kernels/ft_moe_route.h):
 *   top-8 of 256 with the lowest-index tie rule, renormalised det-exp weights and the
 *   shared-expert gate, written to the routing slot and the scale strip.
 *
 * The weights are the REAL layer-0 router and shared-expert gate of Qwen3.6-35B-A3B, read from
 * the metadata GGUF that keeps them at source values (`.meta+router.gguf`). They are F32 in the
 * GGUF and every value is exactly representable in bf16 (checked here, refused otherwise), so
 * the card holds them as bf16 tiles with no loss: B = [2048 x 288] (256 router columns, the
 * shared-expert gate at column 256, zero padding), N-major, Kt = 64, Nt = 9.
 *
 * Gates, three rounds of fresh activations (bf16 row 0, rms-normed and scaled by the layer's
 * post-attention norm, rows 1-31 zero):
 *   - logits: the card's fp32 logits vs a double reference on the same bf16 operands
 *     (reported; bounded by fp32 accumulation), and the card's top-8 SET == the reference's;
 *   - routing step EXACT: the slot (8 indices + 8 weights) and the scale strip's row 0
 *     BIT-IDENTICAL to ft_moe_route.h run on the host over the card's own logits; rows 1-31 of
 *     the strip untouched (zero);
 * Reds: a crafted exact tie (column 9 := column 5, both scaled into the top-8) must put 5
 *   before 9, and FTMR_RED_TIE_FLIP must put 9 first (DIFFERS); FTMR_RED_UNNORM and
 *   FTMR_RED_RANK_PERMUTE slots DIFFER from the green slot.
 *
 * Usage (run_q36_phase5_router.sh builds the program):
 *   FT_TT_FIRMWARE_DIR=<prog>/rtriad_router FT_TT_TENSIX_INDEX=75 \
 *     ./build-q36router/test_q36_router <prog>/rtriad_router <meta+router.gguf>
 *
 * @status New 2026-10-28 (qwen36 Phase 5.3).
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../convert/ft_gguf_map.h"
#include "../kernels/ft_moe_route.h"
#include "../kernels/resident_matmul_moe_protocol.h"
#include "../kernels/resident_matmul_multicast_protocol.h"
#include "ft_tt_group.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "ft_tt_tilize.h"
#include "ft_tt_watchdog.h"

#define HIDDEN 2048u
#define EXPERTS 256u
#define KT (HIDDEN / 32u)
#define NT 9u
#define NCOLS (NT * 32u)
#define SHEXP_COLUMN 256u
#define TOPK 8u
#define EXPERT_TILES 16u
#define SHEXP_TILES 16u
#define SCALE_TILES (TOPK * EXPERT_TILES + SHEXP_TILES)
#define DEPTH 16u
#define BF16_TILE 2048u
#define FP32_TILE 4096u

#define BANK_A 0u
#define BANK_B 1u
#define BANK_C 2u
#define BANK_SLOT 3u
#define BANK_SCALE 4u
#define DRAM_BASE 0x00200000u

#define CB_IN0 0u
#define CB_IN1 1u
#define CB_OUT 16u
#define CB_SCRATCH 31u
#define QUEUE_CAPACITY 64u
#define TIMEOUT_MS 20000u
#define SPIN_LIMIT 400000000u
#define TRIAD_MASK                                                                    \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) |     \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

typedef struct {
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_noc_coord dram[8];
    uint32_t index;
    ft_tt_launch_core_placement placement[1];
    ft_tt_queue queue[1];
    ft_tt_core_group group;
    ft_tt_program_directory router;
    ft_tt_launch_config config;
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    uint32_t scratch;
    float *w;            /* [HIDDEN x NCOLS] row-major, K x N, bf16-exact values */
    uint16_t *b_tiles;   /* N-major bf16 tiles */
    uint16_t *a_tiles;
    float x[HIDDEN];     /* bf16-exact activations, row 0 */
    float norm[HIDDEN];
    unsigned char logits_tiles[NT * FP32_TILE];
    unsigned char scale[SCALE_TILES * BF16_TILE];
    uint32_t slot[16];
    /* ROWS mode (plan S1) */
    float *x_rows;            /* [ROWS_MAX x HIDDEN] bf16-exact */
    unsigned char *logits_rows; /* ROWS_BANDS x NT fp32 tiles */
    unsigned char *scale_rows;  /* ROWS_BANDS x 4096 routed scale tiles */
    unsigned char *scale_shexp; /* ROWS_BANDS x 16 shared-expert scale tiles */
    uint32_t list[FT_TT_MOE_SLOT_COPY_BYTES / 4u];
} router_test;

#define ROWS_BANDS 4u
#define ROWS_MAX (ROWS_BANDS * 32u)
#define ROWS_A4 5u
#define ROUTED_TILES (ROWS_BANDS * FT_TT_MOE_SCALE_LIST_TILES)
#define POISON16 0x5A5Au

static uint32_t g_random = 7u;

static float uniform(void)
{
    g_random = g_random * 1664525u + 1013904223u;
    return (float)(g_random >> 8) / (float)(1u << 24);
}

static uint32_t bits_of(float f)
{
    uint32_t u;

    memcpy(&u, &f, 4);
    return u;
}

static float float_of(uint32_t u)
{
    float f;

    memcpy(&f, &u, 4);
    return f;
}

static int dram_io(router_test *t, uint32_t bank, uint32_t address, void *bytes, size_t count, bool write)
{
    for (size_t offset = 0; offset < count;) {
        size_t chunk = count - offset > 0x10000u ? 0x10000u : count - offset;
        int rc = write ? t->session.noc.write(t->session.noc.user, (uint16_t)t->dram[bank].x, (uint16_t)t->dram[bank].y,
                                              (uint64_t)address + offset, (unsigned char *)bytes + offset, chunk)
                       : t->session.noc.read(t->session.noc.user, (uint16_t)t->dram[bank].x, (uint16_t)t->dram[bank].y,
                                             (uint64_t)address + offset, (unsigned char *)bytes + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

/** Real layer-0 router + shared-expert gate into W [HIDDEN x NCOLS]; false unless bf16-exact. */
static bool load_weights(router_test *t, const char *gguf)
{
    ft_gguf_map *map = NULL;
    const ft_gguf_tensor *router, *gate, *norm;
    size_t inexact = 0;

    if (ft_gguf_map_open(gguf, &map) != 0)
        return false;
    router = ft_gguf_map_find(map, "blk.0.ffn_gate_inp.weight");
    gate = ft_gguf_map_find(map, "blk.0.ffn_gate_inp_shexp.weight");
    norm = ft_gguf_map_find(map, "blk.0.post_attention_norm.weight");
    if (!router || !gate || !norm || router->ggml_type != 0u || gate->ggml_type != 0u || norm->ggml_type != 0u ||
        router->dims[0] != HIDDEN || router->dims[1] != EXPERTS) {
        ft_gguf_map_close(map);
        return false;
    }
    memset(t->w, 0, (size_t)HIDDEN * NCOLS * sizeof(float));
    for (uint32_t e = 0; e < EXPERTS; e++)
        for (uint32_t k = 0; k < HIDDEN; k++)
            t->w[(size_t)k * NCOLS + e] = ((const float *)router->data)[(size_t)e * HIDDEN + k];
    for (uint32_t k = 0; k < HIDDEN; k++) {
        t->w[(size_t)k * NCOLS + SHEXP_COLUMN] = ((const float *)gate->data)[k];
        t->norm[k] = ((const float *)norm->data)[k];
    }
    for (size_t i = 0; i < (size_t)HIDDEN * NCOLS; i++)
        inexact += (bits_of(t->w[i]) & 0xFFFFu) != 0u;
    ft_gguf_map_close(map);
    ft_tt_test_check_fmt("router + shared-expert gate are bf16-exact (F32 in the GGUF)", inexact == 0u,
                         "%zu of %u values have low bits", inexact, HIDDEN * NCOLS);
    return inexact == 0u;
}

/** W -> N-major bf16 tiles in DRAM. */
static int upload_weights(router_test *t)
{
    uint16_t *row_major = malloc((size_t)HIDDEN * NCOLS * 2u);
    uint16_t *k_major = malloc((size_t)HIDDEN * NCOLS * 2u);
    int rc = FT_TT_ENOMEM;

    if (row_major && k_major) {
        for (size_t i = 0; i < (size_t)HIDDEN * NCOLS; i++)
            row_major[i] = (uint16_t)(bits_of(t->w[i]) >> 16);
        rc = ft_tt_tilize_matrix(row_major, 2u, HIDDEN, NCOLS, k_major);
        for (uint32_t kt = 0; rc == FT_TT_OK && kt < KT; kt++)
            for (uint32_t nt = 0; nt < NT; nt++)
                memcpy(t->b_tiles + ((size_t)nt * KT + kt) * 1024u, k_major + ((size_t)kt * NT + nt) * 1024u,
                       BF16_TILE);
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_B, DRAM_BASE, t->b_tiles, (size_t)KT * NT * BF16_TILE, true);
    }
    free(row_major);
    free(k_major);
    return rc;
}

/** Fresh x: N(0,1)-ish, rms-normed, times the norm weight, rounded to bf16. Row 0 only. */
static int upload_activation(router_test *t, uint32_t seed)
{
    uint16_t row_major[32u * HIDDEN];
    double ss = 0.0;

    g_random = seed;
    for (uint32_t k = 0; k < HIDDEN; k++) {
        t->x[k] = (uniform() + uniform() + uniform() - 1.5f) * 2.0f;
        ss += (double)t->x[k] * t->x[k];
    }
    memset(row_major, 0, sizeof(row_major));
    for (uint32_t k = 0; k < HIDDEN; k++) {
        uint32_t b = ftmr_bf16(bits_of(t->x[k] / (float)sqrt(ss / HIDDEN) * t->norm[k]));

        row_major[k] = (uint16_t)b;
        t->x[k] = float_of(b << 16);
    }
    if (ft_tt_tilize_matrix(row_major, 2u, 32u, HIDDEN, t->a_tiles) != FT_TT_OK)
        return FT_TT_ERROR;
    return dram_io(t, BANK_A, DRAM_BASE, t->a_tiles, (size_t)KT * BF16_TILE, true);
}

static int place_cbs(router_test *t)
{
    ft_tt_op_cb_request requests[4] = {
        { CB_IN0, BF16_TILE, DEPTH * 2u },
        { CB_IN1, BF16_TILE, DEPTH * 2u },
        { CB_OUT, FP32_TILE, 2u },
        { CB_SCRATCH, FT_TT_MOE_ROUTER_SCRATCH_BYTES, 1u },
    };
    ft_tt_launch_cb_entry placed[4];
    int rc = ft_tt_op_cb_place(requests, 4u, placed, t->cb_buffer, sizeof(t->cb_buffer), &t->config.local_cb_mask);

    t->scratch = placed[3].address;
    t->config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    t->config.brisc_noc_id = 0u;
    return rc;
}

/** One router descriptor with writer red bits `red` over `bands` tile rows (ROWS mode when
 *  `rows`, the group's first band `a4`); reads back logits, slot and strip. */
static int run_router_ex(router_test *t, uint32_t red, uint32_t bands, bool rows, uint32_t a4,
                         uint64_t *microseconds)
{
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0, FT_TT_PROC_MATH1,
                                            FT_TT_PROC_MATH2 };
    const ft_tt_launch_core_placement *core = &t->placement[0];
    uint32_t args[5][32];
    ft_tt_op_args blocks[5];
    uint32_t words[FT_TT_QUEUE_MATMUL_WORDS], address = 0u, completed = 0u;
    ft_tt_queue_descriptor descriptor;
    struct timespec t0, t1;
    int rc;

    memset(args, 0, sizeof(args));
    memset(t->scale, 0, sizeof(t->scale));
    memset(t->slot, 0xA5, sizeof(t->slot));
    rc = ft_tt_op_program_load(&t->session.noc, core->x, core->y, &t->router.program, NULL, &t->config);
    if (rc == FT_TT_OK)
        rc = place_cbs(t);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&t->session.noc, core->x, core->y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                              ft_tt_queue_make_nonce(), &t->queue[0]);
    for (unsigned p = 0; p < 5u && rc == FT_TT_OK; p++) {
        ft_tt_queue_kernel_binding binding;

        rc = ft_tt_group_kernel_binding(&t->group, 0u, processors[p], FT_TT_PROC_DM1, SPIN_LIMIT, SPIN_LIMIT,
                                        &binding);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_kernel_args(&t->queue[0], &binding, args[p]);
        blocks[p].processor = processors[p];
        blocks[p].values = args[p];
        blocks[p].count = p == 0u ? kArgCoopCount : (p == 1u ? kArgRouteCount : FT_TT_QUEUE_KERNEL_ARG_COUNT);
    }
    args[0][kArgCoopRole] = FT_TT_RESIDENT_COOP_ROLE_SOLO;
    args[1][kArgRouteLogits] = EXPERTS;
    args[1][kArgRouteShexpColumn] = SHEXP_COLUMN;
    args[1][kArgRouteK] = TOPK;
    args[1][kArgRouteSlotX] = t->dram[BANK_SLOT].x;
    args[1][kArgRouteSlotY] = t->dram[BANK_SLOT].y;
    args[1][kArgRouteSlotAddress] = DRAM_BASE;
    args[1][kArgRouteScaleX] = t->dram[BANK_SCALE].x;
    args[1][kArgRouteScaleY] = t->dram[BANK_SCALE].y;
    args[1][kArgRouteScaleAddress] = DRAM_BASE;
    args[1][kArgRouteExpertTiles] = EXPERT_TILES;
    args[1][kArgRouteShexpTiles] = SHEXP_TILES;
    args[1][kArgRouteScratch] = t->scratch;
    args[1][kArgRouteRed] = red;
    if (rc == FT_TT_OK && !rows)
        rc = dram_io(t, BANK_SCALE, DRAM_BASE, t->scale, sizeof(t->scale), true);
    if (rc == FT_TT_OK && rows) {
        memset(t->scale_rows, 0x5A, (size_t)ROUTED_TILES * BF16_TILE);
        memset(t->scale_shexp, 0x5A, (size_t)ROWS_BANDS * EXPERT_TILES * BF16_TILE);
        memset(t->list, 0xA5, sizeof(t->list));
        rc = dram_io(t, BANK_SCALE, DRAM_BASE, t->scale_rows, (size_t)ROUTED_TILES * BF16_TILE, true);
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_SCALE,
                         DRAM_BASE + (FT_TT_MOE_SHEXP_SCALE_TILE0 + (a4 & 0xFFFFu) * EXPERT_TILES) * BF16_TILE,
                         t->scale_shexp, (size_t)ROWS_BANDS * EXPERT_TILES * BF16_TILE, true);
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_SLOT, DRAM_BASE, t->list, sizeof(t->list), true);
    }
    if (rc == FT_TT_OK && !rows)
        rc = dram_io(t, BANK_SLOT, DRAM_BASE, t->slot, sizeof(t->slot), true);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_stage(&t->session.noc, core->x, core->y, &t->config, t->cb_buffer, sizeof(t->cb_buffer),
                            blocks, 5u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_send_go(&t->session.noc, core->x, core->y);
    if (rc != FT_TT_OK)
        return rc;

    memset(words, 0, sizeof(words));
    words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = bands;
    words[FT_TT_QUEUE_MATMUL_TILE_INNER] = KT;
    words[FT_TT_QUEUE_MATMUL_TILE_COLS] = NT;
    words[FT_TT_QUEUE_MATMUL_A_X] = t->dram[BANK_A].x;
    words[FT_TT_QUEUE_MATMUL_A_Y] = t->dram[BANK_A].y;
    words[FT_TT_QUEUE_MATMUL_A_BASE] = DRAM_BASE;
    words[FT_TT_QUEUE_MATMUL_B_X] = t->dram[BANK_B].x;
    words[FT_TT_QUEUE_MATMUL_B_Y] = t->dram[BANK_B].y;
    words[FT_TT_QUEUE_MATMUL_B_BASE] = DRAM_BASE;
    words[FT_TT_QUEUE_MATMUL_C_X] = t->dram[BANK_C].x;
    words[FT_TT_QUEUE_MATMUL_C_Y] = t->dram[BANK_C].y;
    words[FT_TT_QUEUE_MATMUL_C_BASE] = DRAM_BASE;
    words[FT_TT_QUEUE_MATMUL_DEPTH] = DEPTH;
    words[FT_TT_QUEUE_MATMUL_B_N_MAJOR] = 1u | (1u << 8) | (rows ? FT_TT_MOE_FLAG_ROUTE_ROWS : 0u);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    rc = ft_tt_queue_write_matmul_parameters(&t->session.noc, &t->queue[0], 0u, words, &address);
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
    descriptor.a0 = address;
    descriptor.a4 = rows ? a4 : 0u;
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_enqueue(&t->session.noc, &t->queue[0], &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&t->session.noc, &t->queue[0], 1u, TIMEOUT_MS, &completed);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *microseconds = (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_nsec - t0.tv_nsec) / 1000);
    ft_tt_test_check_fmt("router descriptor completes", rc == FT_TT_OK, "rc=%d completed=%u", rc, completed);
    {
        ft_tt_core_group solo = t->group;
        uint32_t unstopped = 0u;
        int stop = ft_tt_watchdog_shutdown(&t->session.noc, &solo, TIMEOUT_MS, &unstopped);

        ft_tt_test_check_fmt("the triad retires when asked", stop == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=%u",
                             stop, unstopped);
    }
    if (rows) {
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_C, DRAM_BASE, t->logits_rows, (size_t)bands * NT * FP32_TILE, false);
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_SLOT, DRAM_BASE, t->list, sizeof(t->list), false);
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_SCALE, DRAM_BASE, t->scale_rows, (size_t)ROUTED_TILES * BF16_TILE, false);
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_SCALE,
                         DRAM_BASE + (FT_TT_MOE_SHEXP_SCALE_TILE0 + (a4 & 0xFFFFu) * EXPERT_TILES) * BF16_TILE,
                         t->scale_shexp, (size_t)ROWS_BANDS * EXPERT_TILES * BF16_TILE, false);
        return rc;
    }
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_C, DRAM_BASE, t->logits_tiles, sizeof(t->logits_tiles), false);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_SLOT, DRAM_BASE, t->slot, sizeof(t->slot), false);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_SCALE, DRAM_BASE, t->scale, sizeof(t->scale), false);
    return rc;
}

static int run_router(router_test *t, uint32_t red, uint64_t *microseconds)
{
    return run_router_ex(t, red, 1u, false, 0u, microseconds);
}

/** Row-0 fp32 value of logical column `c` in the card's logits tiles. */
static uint32_t card_logit(const router_test *t, uint32_t c)
{
    uint32_t bits;
    const uint32_t j = c % 32u;

    memcpy(&bits, t->logits_tiles + (c / 32u) * FP32_TILE + (j < 16u ? j * 4u : 1024u + (j - 16u) * 4u), 4);
    return bits;
}

/** The host's routing step over the card's logits: expected slot + strip row values. */
static void host_route(const router_test *t, uint32_t red, uint32_t slot[16], uint32_t *shexp_bits)
{
    ftmr_top top;
    uint32_t index[TOPK], weight[TOPK];

    ftmr_init(&top, TOPK);
    for (uint32_t c = 0; c < EXPERTS; c++)
        ftmr_offer(&top, c, card_logit(t, c), red);
    ftmr_weights(&top, red, index, weight);
    memset(slot, 0, 16u * sizeof(uint32_t));
    for (uint32_t r = 0; r < top.count; r++) {
        slot[r] = index[r];
        slot[FT_TT_MOE_SLOT_WEIGHT0 + r] = weight[r];
    }
    *shexp_bits = ft_det_sigmoid(card_logit(t, SHEXP_COLUMN));
}

/** Row 0 of every scale tile == the expected bf16, rows 1-31 still zero. */
static bool scale_strip_matches(const router_test *t, const uint32_t slot[16], uint32_t shexp_bits, size_t *bad)
{
    *bad = 0u;
    for (uint32_t tile = 0; tile < SCALE_TILES; tile++) {
        const uint16_t *v = (const uint16_t *)(t->scale + (size_t)tile * BF16_TILE);
        const uint32_t want = ftmr_bf16(tile < TOPK * EXPERT_TILES ? slot[FT_TT_MOE_SLOT_WEIGHT0 + tile / EXPERT_TILES]
                                                                   : shexp_bits);

        for (uint32_t i = 0; i < 1024u; i++) {
            const uint32_t face = i / 256u, within = i % 256u;
            const bool row0 = within < 16u && face < 2u;

            *bad += row0 ? v[i] != want : v[i] != 0u;
        }
    }
    return *bad == 0u;
}

static void stage_rounds(router_test *t)
{
    double worst = 0.0;
    uint64_t us = 0, total = 0;

    for (uint32_t round = 0; round < 3u; round++) {
        uint32_t want[16], shexp_bits;
        size_t bad = 0u;
        double reference[NCOLS], dmax = 0.0;
        int ref_top[TOPK], card_top_equal = 1;

        ft_tt_test_step("round %u", round);
        if (upload_activation(t, 0x5E000000u + round) != FT_TT_OK || run_router(t, 0u, &us) != FT_TT_OK)
            return;
        total += us;
        for (uint32_t c = 0; c < NCOLS; c++) {
            reference[c] = 0.0;
            for (uint32_t k = 0; k < HIDDEN; k++)
                reference[c] += (double)t->x[k] * t->w[(size_t)k * NCOLS + c];
            if (c < EXPERTS || c == SHEXP_COLUMN) {
                const double d = fabs((double)float_of(card_logit(t, c)) - reference[c]);

                dmax = d > dmax ? d : dmax;
            }
        }
        worst = dmax > worst ? dmax : worst;
        for (uint32_t r = 0; r < TOPK; r++) { /* the reference's top-8 (selection, lowest index on ties) */
            double best = -1e300;
            int at = -1;

            for (uint32_t c = 0; c < EXPERTS; c++) {
                int taken = 0;

                for (uint32_t q = 0; q < r; q++)
                    taken |= ref_top[q] == (int)c;
                if (!taken && reference[c] > best) {
                    best = reference[c];
                    at = (int)c;
                }
            }
            ref_top[r] = at;
        }
        for (uint32_t r = 0; r < TOPK; r++) {
            int found = 0;

            for (uint32_t q = 0; q < TOPK; q++)
                found |= t->slot[q] == (uint32_t)ref_top[r];
            card_top_equal &= found;
        }
        /* A SANITY bound, not the quality bar: a misplaced tile or face is O(1). The bar is the
         * routing (the set check below; spec §8 B's agreement rates in Phase 7). The FPU's
         * in-tile reduction is not an exact fp32 sum: measured ~1e-3 on logits of a few units,
         * against 3.5e-2 for a Bfp8_b router (evals/expert_quant/results/router_card_formats.log). */
        ft_tt_test_check_fmt("logits: card fp32 vs the double reference (sanity < 1e-2)", dmax < 1e-2,
                             "max |d| %.3g", dmax);
        ft_tt_test_check_fmt("the card's top-8 SET == the reference's", card_top_equal, "experts %u %u %u %u %u %u %u %u",
                             t->slot[0], t->slot[1], t->slot[2], t->slot[3], t->slot[4], t->slot[5], t->slot[6],
                             t->slot[7]);
        host_route(t, 0u, want, &shexp_bits);
        ft_tt_test_check_fmt("GATE: routing slot BIT-IDENTICAL to ft_moe_route.h on the card's logits",
                             memcmp(want, t->slot, sizeof(want)) == 0, "w0 card %08x host %08x", t->slot[8], want[8]);
        ft_tt_test_check_fmt("GATE: scale strip row 0 == bf16(weights, sigmoid gate), rows 1-31 untouched",
                             scale_strip_matches(t, want, shexp_bits, &bad), "%zu values wrong (gate %.6f)", bad,
                             (double)float_of(shexp_bits));
    }
    printf("LOGITS: worst max |dlogit| vs double over 3 rounds %.3g\n", worst);
    printf("PRICE: router descriptor (Mt=1 Kt=64 Nt=9 + route + slot + %u-tile strip) %.1f us host wall, mean of 3\n",
           SCALE_TILES, (double)total / 3.0);
}

static void stage_reds(router_test *t)
{
    uint32_t green[16], want[16], shexp_bits;
    uint64_t us;
    size_t bad = 0u;

    ft_tt_test_step("reds: a crafted tie, and each FTMR red bit");
    /* Columns 5 and 9 := x * 2^-6 (bf16-exact): an EXACT tie (same values, same products, same
     * accumulation order) at logit 2^-6 * |x|^2, far above every real router logit, so both
     * are selected and the tie rule alone orders them. */
    if (upload_activation(t, 0x7E000000u) != FT_TT_OK)
        return;
    for (uint32_t k = 0; k < HIDDEN; k++) {
        t->w[(size_t)k * NCOLS + 5u] = t->x[k] * 0.015625f;
        t->w[(size_t)k * NCOLS + 9u] = t->x[k] * 0.015625f;
    }
    if (upload_weights(t) != FT_TT_OK || run_router(t, 0u, &us) != FT_TT_OK)
        return;
    memcpy(green, t->slot, sizeof(green));
    {
        int at5 = -1, at9 = -1;

        for (int r = 0; r < (int)TOPK; r++) {
            at5 = green[r] == 5u ? r : at5;
            at9 = green[r] == 9u ? r : at9;
        }
        ft_tt_test_check_fmt("tie: logits 5 and 9 bit-equal on the card",
                             card_logit(t, 5u) == card_logit(t, 9u), "%08x %08x", card_logit(t, 5u), card_logit(t, 9u));
        ft_tt_test_check_fmt("tie: both selected, 5 ranked before 9 (lowest index wins)",
                             at5 >= 0 && at9 >= 0 && at5 < at9, "rank of 5 = %d, of 9 = %d", at5, at9);
    }
    host_route(t, 0u, want, &shexp_bits);
    ft_tt_test_check_fmt("tie round: slot BIT-IDENTICAL to the host", memcmp(want, green, sizeof(want)) == 0, "%s", "ok");

    static const struct {
        uint32_t red;
        const char *name;
    } reds[3] = { { FTMR_RED_TIE_FLIP, "moe_tie_flip" },
                  { FTMR_RED_UNNORM, "moe_weights_unnorm" },
                  { FTMR_RED_RANK_PERMUTE, "moe_rank_permute" } };
    for (uint32_t i = 0; i < 3u; i++) {
        if (run_router(t, reds[i].red, &us) != FT_TT_OK)
            return;
        host_route(t, reds[i].red, want, &shexp_bits);
        ft_tt_test_check_fmt("RED: slot DIFFERS from the green slot", memcmp(t->slot, green, sizeof(green)) != 0,
                             "%s", reds[i].name);
        ft_tt_test_check_fmt("...and equals the host's red (the red is the one intended)",
                             memcmp(want, t->slot, sizeof(want)) == 0 && scale_strip_matches(t, want, shexp_bits, &bad),
                             "%s", reds[i].name);
    }
}

/** ROWS_MAX activation rows (bf16-exact, rms-normed), tilized tile-row-major into A. */
static int upload_rows(router_test *t, uint32_t seed)
{
    uint16_t *row_major = calloc((size_t)ROWS_MAX * HIDDEN, 2u);
    int rc = FT_TT_ENOMEM;

    if (!row_major)
        return rc;
    g_random = seed;
    for (uint32_t r = 0; r < ROWS_MAX; r++) {
        double ss = 0.0;
        float *x = t->x_rows + (size_t)r * HIDDEN;

        for (uint32_t k = 0; k < HIDDEN; k++) {
            x[k] = (uniform() + uniform() + uniform() - 1.5f) * 2.0f;
            ss += (double)x[k] * x[k];
        }
        for (uint32_t k = 0; k < HIDDEN; k++) {
            const uint32_t b = ftmr_bf16(bits_of(x[k] / (float)sqrt(ss / HIDDEN) * t->norm[k]));

            row_major[(size_t)r * HIDDEN + k] = (uint16_t)b;
            x[k] = float_of(b << 16);
        }
    }
    rc = FT_TT_OK;
    for (uint32_t b = 0; b < ROWS_BANDS && rc == FT_TT_OK; b++)
        rc = ft_tt_tilize_matrix(row_major + (size_t)b * 32u * HIDDEN, 2u, 32u, HIDDEN,
                                 t->a_tiles + (size_t)b * KT * 1024u);
    free(row_major);
    return rc == FT_TT_OK ? dram_io(t, BANK_A, DRAM_BASE, t->a_tiles, (size_t)ROWS_BANDS * KT * BF16_TILE, true)
                          : rc;
}

/** Row `r`'s fp32 logit at logical column `c` in the card's rows-mode logits tiles. */
static uint32_t card_logit_row(const router_test *t, uint32_t r, uint32_t c)
{
    const uint32_t j = r % 32u, col = c % 32u;
    const size_t tile = (size_t)(r / 32u) * NT + c / 32u;
    const size_t at = (j / 16u) * 2048u + (j % 16u) * 64u + (col / 16u) * 1024u + (col % 16u) * 4u;
    uint32_t bits;

    memcpy(&bits, t->logits_rows + tile * FP32_TILE + at, 4);
    return bits;
}

/** Value at (row j, column c) of a bf16 face-order tile. */
static uint16_t tile_value(const unsigned char *tile, uint32_t j, uint32_t c)
{
    uint16_t v;

    memcpy(&v, tile + (((j / 16u) * 2u + c / 16u) * 256u + (j % 16u) * 16u + c % 16u) * 2u, 2);
    return v;
}

/**
 * The host's rows mode over the card's own logits: per-row top-8 and weights, the union list,
 * then every scale tile of every list entry (E copies) and the shared-expert tiles; tiles of
 * entries >= n must still hold the poison. Returns the count of wrong values; *out_n = n.
 */
static size_t rows_mismatches(const router_test *t, uint32_t red, uint32_t *out_n, size_t *list_bad)
{
    ftmr_row rows[ROWS_MAX];
    uint32_t bitmap[FTMR_EXPERTS_MAX / 32u] = { 0u }, list[FTMR_EXPERTS_MAX], n;
    size_t bad = 0u;

    for (uint32_t r = 0; r < ROWS_MAX; r++) {
        ftmr_top top;

        ftmr_init(&top, TOPK);
        for (uint32_t c = 0; c < EXPERTS; c++)
            ftmr_offer(&top, c, card_logit_row(t, r, c), red);
        ftmr_weights(&top, red, rows[r].index, rows[r].weight);
        ftmr_union_add(bitmap, &rows[r], TOPK);
    }
    n = ftmr_union_list(bitmap, EXPERTS, list);
    *out_n = n;
    *list_bad = t->list[FT_TT_MOE_LIST_COUNT_ENTRY] != n;
    for (uint32_t s = 0; s < n; s++)
        *list_bad += t->list[s] != list[s];
    for (uint32_t b = 0; b < ROWS_BANDS; b++) {
        for (uint32_t s = 0; s < EXPERTS; s++)
            for (uint32_t c = 0; c < EXPERT_TILES; c++) {
                const unsigned char *tile =
                    t->scale_rows + ((size_t)b * FT_TT_MOE_SCALE_LIST_TILES + s * EXPERT_TILES + c) * BF16_TILE;

                for (uint32_t j = 0; j < 32u; j++)
                    for (uint32_t col = 0; col < 32u; col++) {
                        const uint16_t want =
                            s < n ? (uint16_t)ftmr_bf16(ftmr_row_weight(&rows[b * 32u + j], TOPK, list[s])) : POISON16;

                        bad += tile_value(tile, j, col) != want;
                    }
            }
        for (uint32_t c = 0; c < EXPERT_TILES; c++)
            for (uint32_t j = 0; j < 32u; j++)
                for (uint32_t col = 0; col < 32u; col++) {
                    const uint16_t want = (uint16_t)ftmr_bf16(ft_det_sigmoid(card_logit_row(t, b * 32u + j, SHEXP_COLUMN)));

                    bad += tile_value(t->scale_shexp + ((size_t)b * EXPERT_TILES + c) * BF16_TILE, j, col) != want;
                }
    }
    return bad;
}

static void stage_rows(router_test *t)
{
    uint64_t us = 0;
    uint32_t n = 0;
    size_t bad, list_bad = 0;

    ft_tt_test_step("ROWS mode (plan S1): %u rows routed, union list + scale tiles", ROWS_MAX);
    if (upload_rows(t, 0x5F000000u) != FT_TT_OK || run_router_ex(t, 0u, ROWS_BANDS, true, ROWS_A4, &us) != FT_TT_OK)
        return;
    bad = rows_mismatches(t, 0u, &n, &list_bad);
    ft_tt_test_check_fmt("GATE: the list (union, ascending) and its count BIT-IDENTICAL to the host's", list_bad == 0u,
                         "n %u (card %u), %zu entries wrong", n, t->list[FT_TT_MOE_LIST_COUNT_ENTRY], list_bad);
    ft_tt_test_check_fmt("GATE: every scale tile (entries < n, E copies) and shared-expert tile == the host's; "
                         "entries >= n untouched", bad == 0u, "%zu values wrong of %zu", bad,
                         (size_t)ROWS_BANDS * (EXPERTS + 1u) * EXPERT_TILES * 1024u);
    printf("PRICE: rows-mode router descriptor (Mt=%u, %u rows, n = %u list entries) %.1f us host wall\n", ROWS_BANDS,
           ROWS_MAX, n, (double)us);
    if (run_router_ex(t, FTMR_RED_UNNORM, ROWS_BANDS, true, ROWS_A4, &us) != FT_TT_OK)
        return;
    bad = rows_mismatches(t, 0u, &n, &list_bad);
    ft_tt_test_check_fmt("RED moe_weights_unnorm (rows): scale tiles DIFFER from the green host's", bad > 0u,
                         "%zu values differ", bad);
    bad = rows_mismatches(t, FTMR_RED_UNNORM, &n, &list_bad);
    ft_tt_test_check_fmt("...and equal the host's red", bad == 0u && list_bad == 0u, "%zu values wrong", bad);
}

/**
 * The host's CONCAT mode (verify rows, plan S6) over the card's band-0 logits: entry r*k + i is
 * row r's i-th ranked expert, n = rows * k, entries n..255 zero; entry e's E scale tiles hold
 * row e/k's weight in row e/k and 0 elsewhere; tiles of entries >= n, and of bands 1..3, still
 * the poison; the band's shared-expert tiles as rows mode, the other bands' untouched.
 */
static size_t concat_mismatches(const router_test *t, uint32_t red, uint32_t rows, size_t *list_bad)
{
    ftmr_row routed[32];
    const uint32_t n = rows * TOPK;
    size_t bad = 0u;

    for (uint32_t r = 0; r < rows; r++) {
        ftmr_top top;

        ftmr_init(&top, TOPK);
        for (uint32_t c = 0; c < EXPERTS; c++)
            ftmr_offer(&top, c, card_logit_row(t, r, c), red);
        ftmr_weights(&top, red, routed[r].index, routed[r].weight);
    }
    *list_bad = t->list[FT_TT_MOE_LIST_COUNT_ENTRY] != n;
    for (uint32_t e = 0; e < FT_TT_MOE_LIST_MAX; e++)
        *list_bad += t->list[e] != (e < n ? routed[e / TOPK].index[e % TOPK] : 0u);
    for (uint32_t b = 0; b < ROWS_BANDS; b++)
        for (uint32_t e = 0; e < EXPERTS; e++)
            for (uint32_t c = 0; c < EXPERT_TILES; c++) {
                const unsigned char *tile =
                    t->scale_rows + ((size_t)b * FT_TT_MOE_SCALE_LIST_TILES + e * EXPERT_TILES + c) * BF16_TILE;

                for (uint32_t j = 0; j < 32u; j++) {
                    uint16_t want = POISON16;

                    if (b == 0u && e < n)
                        want = j == e / TOPK ? (uint16_t)ftmr_bf16(routed[e / TOPK].weight[e % TOPK]) : 0u;
                    for (uint32_t col = 0; col < 32u; col++)
                        bad += tile_value(tile, j, col) != want;
                }
            }
    for (uint32_t b = 0; b < ROWS_BANDS; b++)
        for (uint32_t c = 0; c < EXPERT_TILES; c++)
            for (uint32_t j = 0; j < 32u; j++)
                for (uint32_t col = 0; col < 32u; col++) {
                    const uint16_t want =
                        b == 0u ? (uint16_t)ftmr_bf16(ft_det_sigmoid(card_logit_row(t, j, SHEXP_COLUMN))) : POISON16;

                    bad += tile_value(t->scale_shexp + ((size_t)b * EXPERT_TILES + c) * BF16_TILE, j, col) != want;
                }
    return bad;
}

static void stage_concat(router_test *t)
{
    static const uint32_t row_counts[] = { 1u, 2u, 5u, 16u };
    uint64_t us = 0;
    size_t bad, list_bad = 0;

    if (upload_rows(t, 0x5E000000u) != FT_TT_OK)
        return;
    for (size_t i = 0; i < sizeof(row_counts) / sizeof(row_counts[0]); i++) {
        const uint32_t rows = row_counts[i];

        ft_tt_test_step("CONCAT mode (plan S6 verify rows): %u row(s), per-row rank-ordered top-%u", rows, TOPK);
        if (run_router_ex(t, 0u, 1u, true, ftmoe_route_a4(ROWS_A4, rows), &us) != FT_TT_OK)
            return;
        bad = concat_mismatches(t, 0u, rows, &list_bad);
        ft_tt_test_check_fmt("GATE: the concatenated list and its count BIT-IDENTICAL to the host's", list_bad == 0u,
                             "count %u (want %u), %zu entries wrong", t->list[FT_TT_MOE_LIST_COUNT_ENTRY], rows * TOPK,
                             list_bad);
        ft_tt_test_check_fmt("GATE: every scale tile (owner row only) and shared-expert tile == the host's; the rest "
                             "untouched", bad == 0u, "%zu values wrong", bad);
        printf("PRICE: CONCAT router descriptor (%u rows, n = %u) %.1f us host wall\n", rows, rows * TOPK,
               (double)us);
    }
    /* Reds: the identity rests on the rank ORDER and the weights, so both must be seen. */
    if (run_router_ex(t, FTMR_RED_RANK_PERMUTE, 1u, true, ftmoe_route_a4(ROWS_A4, 5u), &us) != FT_TT_OK)
        return;
    bad = concat_mismatches(t, 0u, 5u, &list_bad);
    ft_tt_test_check_fmt("RED rank_permute (concat): the list DIFFERS from the green host's", list_bad > 0u,
                         "%zu entries differ", list_bad);
    bad = concat_mismatches(t, FTMR_RED_RANK_PERMUTE, 5u, &list_bad);
    ft_tt_test_check_fmt("...and equals the host's red", bad == 0u && list_bad == 0u, "%zu values, %zu entries wrong",
                         bad, list_bad);
    if (run_router_ex(t, FTMR_RED_UNNORM, 1u, true, ftmoe_route_a4(ROWS_A4, 5u), &us) != FT_TT_OK)
        return;
    bad = concat_mismatches(t, 0u, 5u, &list_bad);
    ft_tt_test_check_fmt("RED moe_weights_unnorm (concat): scale tiles DIFFER from the green host's", bad > 0u,
                         "%zu values differ", bad);
}

int main(int argc, char **argv)
{
    static router_test t;
    int rc;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <rtriad_router-dir> <meta+router.gguf>\n", argv[0]);
        return 2;
    }
    t.index = 75u;
    t.w = malloc((size_t)HIDDEN * NCOLS * sizeof(float));
    t.b_tiles = malloc((size_t)KT * NT * BF16_TILE);
    t.a_tiles = malloc((size_t)ROWS_BANDS * KT * BF16_TILE);
    t.x_rows = malloc((size_t)ROWS_MAX * HIDDEN * sizeof(float));
    t.logits_rows = malloc((size_t)ROWS_BANDS * NT * FP32_TILE);
    t.scale_rows = malloc((size_t)ROUTED_TILES * BF16_TILE);
    t.scale_shexp = malloc((size_t)ROWS_BANDS * EXPERT_TILES * BF16_TILE);
    if (!t.w || !t.b_tiles || !t.a_tiles || !t.x_rows || !t.logits_rows || !t.scale_rows || !t.scale_shexp ||
        !load_weights(&t, argv[2]))
        return ft_tt_test_finish();

    rc = ft_tt_hardware_session_open(&t.session);
    ft_tt_test_check_fmt("card opens", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return ft_tt_test_finish();
    (void)ft_tt_tensix_index_from_environment(&t.index);
    rc = ft_tt_hardware_session_claim_cores(&t.session, &t.index, 1u, "qwen36 phase 5 router", NULL, NULL);
    for (uint32_t bank = 0; bank < 8u && rc == FT_TT_OK; bank++)
        rc = ft_tt_coord_dram_bank(&t.session.grid, bank, FT_TT_COORD_TRANSLATED, &t.dram[bank]);
    if (rc == FT_TT_OK)
        rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &t.firmware);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &t.router);
    if (rc == FT_TT_OK) {
        memset(&t.group, 0, sizeof(t.group));
        t.group.first_tensix_index = t.index;
        t.group.tensix_index_count = 1u;
        t.group.queues = t.queue;
        t.group.resident_processor_mask = TRIAD_MASK;
        rc = ft_tt_launch_core_set_init(&t.group.cores, t.placement, 1u);
        if (rc == FT_TT_OK)
            rc = ft_tt_launch_core_set_add_tensix(&t.group.cores, &t.session.grid, t.index, NULL);
    }
    if (rc == FT_TT_OK)
        rc = ft_tt_core_boot_firmware(&t.session.noc, t.placement[0].x, t.placement[0].y, &t.firmware, NULL);
    if (rc == FT_TT_OK)
        rc = upload_weights(&t);
    ft_tt_test_check_fmt("core claimed, program loaded, firmware up, weights uploaded", rc == FT_TT_OK,
                         "index %u: %s", t.index, ft_tt_error_string(rc));
    if (rc == FT_TT_OK) {
        stage_rounds(&t);
        stage_rows(&t);
        stage_concat(&t);
        stage_reds(&t);
    }
    ft_tt_hardware_session_close(&t.session);
    return ft_tt_test_finish();
}
