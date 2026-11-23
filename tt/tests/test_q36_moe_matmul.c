/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_q36_moe_matmul.c
 * @brief qwen36 plan Phase 5.1 / 5.2 on the card: the resident matmul triad reading a ROUTED
 *   EXPERT out of a BFP6_B stack — the in-reader expander (`rtriad_bfp6`) and the indirect
 *   expert operand (B base + slot[rank] * stride, read on the card).
 *
 * THE GATE IS BYTE IDENTITY, NOT PCC. Expansion is exact (BFP8 datum = BFP6 datum << 2, same
 * exponent), so the MoE arm — BFP6_B stack in DRAM, expert chosen on the card from the routing
 * slot — must produce the SAME BYTES as the shipped dense arm (`rtriad_bfp8`) reading the
 * host-expanded BFP8_B copy of the chosen expert at a host-computed address. Same compute
 * source, same CB format, same DST: any difference is the reader's bug.
 *
 * One core (FT_TT_TENSIX_INDEX, default 75). Shape: Qwen3.6's expert gate/up at decode,
 * Mt = 1, Kt = 64, Nt = 16, N-major, depth 16, pipelined reader; a stack of 8 synthetic experts
 * (random values, packed by foundry's own ft_bfp_pack_tiles). Each arm runs EIGHT descriptors,
 * one per slot rank, each to its own output block; three rounds of fresh data and a fresh,
 * different routing slot, output poisoned before every arm.
 *
 * Reds (each must DIFFER, or be refused):
 *   moe_idx_shift   the slot holds idx + 1 (a wrong routing decision reaching the reader)
 *   moe_stride_off  the residency's stride is one tile long (expert addressing off)
 *   bfp6_expand_bit the reader built with the expander shifting by 1 (rtriad_bfp6_red)
 *   moe_idx_bound   an index == the expert count: REFUSED with kErrMoeExpertIndex, nothing
 *                   retired, later descriptors never run, and the core runs the gate again
 * K-experts: the routed down + weighted sum as ONE descriptor (FT_TT_MOE_FLAG_K_EXPERTS,
 *   Kt = 8 experts x 16), byte-identical to rtriad_bfp8 over a host-built K-concatenation of
 *   the same experts; red moe_idx_shift DIFFERS.
 * Price: us a descriptor, MoE arm vs the dense BFP8 arm on the same core (the expansion is
 * the difference), from the host's wall clock around eight descriptors.
 *
 * Usage (run_q36_phase5_moe_matmul.sh builds the programs and probes the core):
 *   FT_TT_FIRMWARE_DIR=<prog>/rtriad_bfp8 FT_TT_TENSIX_INDEX=75 \
 *     ./build-q36moe/test_q36_moe_matmul <prog>/rtriad_bfp8 <prog>/rtriad_bfp6 <prog>/rtriad_bfp6_red
 *
 * @status New 2026-10-28 (qwen36 Phase 5.1 / 5.2).
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../convert/ft_bfp.h"
#include "../kernels/resident_matmul_moe_protocol.h"
#include "../kernels/resident_matmul_multicast_protocol.h"
#include "ft_tt_group.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "ft_tt_watchdog.h"

#define KT 64u
#define NT 16u
#define DEPTH 16u
#define EXPERTS 8u
#define RANKS 8u
#define TILE_VALUES 1024u
#define A_TILE 2048u
#define C_TILE 2048u
#define BFP6_TILE 832u
#define BFP8_TILE 1088u
#define EXPERT_TILES (KT * NT)
#define STRIDE6 (EXPERT_TILES * BFP6_TILE) /* 851,968 B: the artifact's expert stride */
#define STRIDE8 (EXPERT_TILES * BFP8_TILE)
#define C_BYTES (RANKS * NT * C_TILE)
/* The routed down (K-concatenated experts): each expert Kt = 16 (512), Nt = NT here. */
#define DOWN_KT 16u
#define DOWN_TILES (DOWN_KT * NT)
#define DOWN_STRIDE6 (DOWN_TILES * BFP6_TILE)
#define A_DOWN_BASE (DRAM_BASE + 0x00080000u)   /* bank A: 128 K-tiles of scaled SwiGLU output */
#define DOWN_BASE (DRAM_BASE + 0x01000000u)     /* bank B6: the BFP6 down stack */
#define CAT_BASE (DRAM_BASE + 0x01000000u)      /* bank B8: the host-built K-concatenated BFP8 B */

/* DRAM: everything in bank 0..3 at a high offset clear of anything a test leaves. */
#define BANK_A 0u
#define BANK_B6 1u
#define BANK_B8 2u
#define BANK_C 3u
#define BANK_SLOT 0u
#define DRAM_BASE 0x00200000u
#define SLOT_ADDRESS (DRAM_BASE + 0x00100000u)

#define CB_IN0 0u
#define CB_IN1 1u
#define CB_OUT 16u
#define CB_SCRATCH 31u
#define QUEUE_CAPACITY 64u
#define TIMEOUT_MS 20000u
#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT 400000000u
#define POISON 0x5Au
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
    ft_tt_program_directory bfp8, bfp6, bfp6_red;
    ft_tt_launch_config config;
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    uint32_t scratch;
    unsigned char *b6, *b8, *reference, *observed;
    uint16_t *a;
    uint32_t idx[RANKS];
} moe_test;

typedef struct {
    const ft_tt_program_directory *program;
    bool indirect;
    uint32_t stride;          /* the residency's kArgMoeExpertStride */
    uint32_t count;           /* kArgMoeExpertCount */
    uint32_t descriptors;     /* ranks 0..descriptors-1 are enqueued */
    bool kexp;                /* ONE K-concatenated-experts descriptor (the routed down) */
    uint64_t microseconds;    /* out: eight descriptors, host wall clock */
    uint32_t error;           /* out: the queue's error field */
    bool completed;           /* out */
    bool custom;              /* ONE descriptor of `words`, descriptor word a1 = `a1` (LIST mode) */
    uint32_t words[FT_TT_QUEUE_MATMUL_WORDS];
    uint32_t a1;
} moe_arm;

static uint32_t g_random = 1u;

static float random_value(void)
{
    g_random = g_random * 1664525u + 1013904223u;
    return ((float)(g_random >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static uint64_t now_us(void)
{
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
}

static int dram_io(moe_test *t, uint32_t bank, uint32_t address, void *bytes, size_t count, bool write)
{
    size_t offset = 0;

    while (offset < count) {
        size_t chunk = count - offset > 0x10000u ? 0x10000u : count - offset;
        int rc = write ? t->session.noc.write(t->session.noc.user, (uint16_t)t->dram[bank].x,
                                              (uint16_t)t->dram[bank].y, (uint64_t)address + offset,
                                              (unsigned char *)bytes + offset, chunk)
                       : t->session.noc.read(t->session.noc.user, (uint16_t)t->dram[bank].x,
                                             (uint16_t)t->dram[bank].y, (uint64_t)address + offset,
                                             (unsigned char *)bytes + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

static uint16_t bf16_of(float value)
{
    uint32_t u;

    memcpy(&u, &value, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

/** The card holds expert stacks in the FOLD layout (plan S3): stream tiles -> fold, in place
 *  through a scratch copy. The BFP8 reference stays the stream expansion. */
static int fold_in_place(unsigned char *tiles, size_t count)
{
    unsigned char *folded = malloc(count * BFP6_TILE);
    int rc;

    if (!folded)
        return -1;
    rc = ft_bfp6_fold_from_stream(tiles, count, folded);
    if (rc == FT_BFP_OK)
        memcpy(tiles, folded, count * BFP6_TILE);
    free(folded);
    return rc == FT_BFP_OK ? 0 : -1;
}

/** Fresh activation, a fresh 8-expert stack (BFP6 + its exact BFP8 expansion), uploaded. */
static int upload_operands(moe_test *t, uint32_t seed)
{
    float *values = malloc((size_t)EXPERTS * EXPERT_TILES * TILE_VALUES * sizeof(float));
    int rc = FT_TT_ENOMEM;

    if (!values)
        return rc;
    g_random = seed;
    for (size_t i = 0; i < (size_t)KT * TILE_VALUES; i++)
        t->a[i] = bf16_of(random_value());
    for (size_t i = 0; i < (size_t)EXPERTS * EXPERT_TILES * TILE_VALUES; i++) {
        float v = random_value();
        /* A few octaves per tile, so shared exponents differ block to block. */
        values[i] = v * 0.02f * (float)(1u << ((i / 16u) % 4u));
    }
    rc = ft_bfp_pack_tiles(FT_BFP_BFP6_B, values, (size_t)EXPERTS * EXPERT_TILES, t->b6) == 0 &&
                 ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, t->b6, (size_t)EXPERTS * EXPERT_TILES, t->b8) == 0 &&
                 fold_in_place(t->b6, (size_t)EXPERTS * EXPERT_TILES) == 0
             ? FT_TT_OK
             : FT_TT_ERROR;
    free(values);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_A, DRAM_BASE, t->a, (size_t)KT * A_TILE, true);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_B6, DRAM_BASE, t->b6, (size_t)EXPERTS * STRIDE6, true);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_B8, DRAM_BASE, t->b8, (size_t)EXPERTS * STRIDE8, true);
    return rc;
}

static int write_slot(moe_test *t, const uint32_t *idx)
{
    uint32_t slot[FT_TT_MOE_SLOT_ENTRIES];

    memset(slot, 0, sizeof(slot));
    memcpy(slot, idx, RANKS * sizeof(uint32_t));
    return dram_io(t, BANK_SLOT, SLOT_ADDRESS, slot, sizeof(slot), true);
}

static int poison_output(moe_test *t)
{
    memset(t->observed, POISON, C_BYTES);
    return dram_io(t, BANK_C, DRAM_BASE, t->observed, C_BYTES, true);
}

static int place_cbs(moe_test *t)
{
    ft_tt_op_cb_request requests[4] = {
        { CB_IN0, A_TILE, DEPTH * 2u },
        { CB_IN1, BFP8_TILE, DEPTH * 2u },
        { CB_OUT, C_TILE, 2u },
        { CB_SCRATCH, FT_TT_MOE_SCRATCH_BYTES, 1u },
    };
    ft_tt_launch_cb_entry placed[4];
    int rc = ft_tt_op_cb_place(requests, 4u, placed, t->cb_buffer, sizeof(t->cb_buffer),
                               &t->config.local_cb_mask);

    t->scratch = placed[3].address;
    t->config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    t->config.brisc_noc_id = 0u;
    return rc != FT_TT_OK ? rc : ((t->scratch & 63u) != 0u ? FT_TT_EINVAL : FT_TT_OK);
}

/** Load, stage and kick one triad with the reader's MoE tail; then eight descriptors. */
static int run_arm(moe_test *t, moe_arm *arm)
{
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0, FT_TT_PROC_MATH1,
                                            FT_TT_PROC_MATH2 };
    const ft_tt_launch_core_placement *core = &t->placement[0];
    uint32_t args[5][kArgMoeCount];
    ft_tt_op_args blocks[5];
    uint32_t completed = 0u;
    uint64_t started;
    int rc;

    memset(args, 0, sizeof(args));
    rc = ft_tt_op_program_load(&t->session.noc, core->x, core->y, &arm->program->program, NULL, &t->config);
    if (rc == FT_TT_OK)
        rc = place_cbs(t);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&t->session.noc, core->x, core->y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                              ft_tt_queue_make_nonce(), &t->queue[0]);
    for (unsigned p = 0; p < 5u && rc == FT_TT_OK; p++) {
        ft_tt_queue_kernel_binding binding;

        rc = ft_tt_group_kernel_binding(&t->group, 0u, processors[p], FT_TT_PROC_DM1, IDLE_SPIN_LIMIT,
                                        CB_SPIN_LIMIT, &binding);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_kernel_args(&t->queue[0], &binding, args[p]);
        blocks[p].processor = processors[p];
        blocks[p].values = args[p];
        blocks[p].count = p == 0u ? kArgMoeCount : FT_TT_QUEUE_KERNEL_ARG_COUNT;
    }
    /* The reader's tail: cooperation off (SOLO, zeros), then the MoE words. */
    args[0][kArgCoopRole] = FT_TT_RESIDENT_COOP_ROLE_SOLO;
    args[0][kArgMoeSlotX] = t->dram[BANK_SLOT].x;
    args[0][kArgMoeSlotY] = t->dram[BANK_SLOT].y;
    args[0][kArgMoeSlotAddress] = SLOT_ADDRESS;
    args[0][kArgMoeExpertStride] = arm->stride;
    args[0][kArgMoeExpertCount] = arm->count;
    args[0][kArgMoeScratch] = t->scratch;
    if (rc == FT_TT_OK) {
        /* The expansion mailbox restarts with the residency (resident_matmul_moe_helpers.h). */
        const uint32_t zeros[FT_TT_MOE_MAILBOX_BYTES / 4u] = { 0u };

        rc = t->session.noc.write(t->session.noc.user, core->x, core->y,
                                  FT_TT_OP_QUEUE_BASE + FT_TT_MOE_MAILBOX_OFFSET, zeros, sizeof(zeros));
    }
    if (rc == FT_TT_OK)
        rc = poison_output(t);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_stage(&t->session.noc, core->x, core->y, &t->config, t->cb_buffer, sizeof(t->cb_buffer),
                            blocks, 5u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_send_go(&t->session.noc, core->x, core->y);
    if (rc != FT_TT_OK)
        return rc;

    started = now_us();
    for (uint32_t k = 0; k < arm->descriptors && rc == FT_TT_OK; k++) {
        uint32_t words[FT_TT_QUEUE_MATMUL_WORDS];
        ft_tt_queue_descriptor descriptor;
        uint32_t address = 0u;

        memset(words, 0, sizeof(words));
        words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = 1u;
        words[FT_TT_QUEUE_MATMUL_TILE_INNER] = KT;
        words[FT_TT_QUEUE_MATMUL_TILE_COLS] = NT;
        words[FT_TT_QUEUE_MATMUL_A_X] = t->dram[BANK_A].x;
        words[FT_TT_QUEUE_MATMUL_A_Y] = t->dram[BANK_A].y;
        words[FT_TT_QUEUE_MATMUL_A_BASE] = DRAM_BASE;
        words[FT_TT_QUEUE_MATMUL_C_X] = t->dram[BANK_C].x;
        words[FT_TT_QUEUE_MATMUL_C_Y] = t->dram[BANK_C].y;
        words[FT_TT_QUEUE_MATMUL_C_BASE] = DRAM_BASE + k * NT * C_TILE;
        words[FT_TT_QUEUE_MATMUL_DEPTH] = DEPTH;
        words[FT_TT_QUEUE_MATMUL_B_N_MAJOR] = 1u | (1u << 8);
        if (arm->indirect) {
            /* The stack's base, the SAME word for every rank and every token: the expert is
             * the slot's business. */
            words[FT_TT_QUEUE_MATMUL_B_X] = t->dram[BANK_B6].x;
            words[FT_TT_QUEUE_MATMUL_B_Y] = t->dram[BANK_B6].y;
            words[FT_TT_QUEUE_MATMUL_B_BASE] = DRAM_BASE;
            words[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= ftmoe_flags(k);
        } else {
            words[FT_TT_QUEUE_MATMUL_B_X] = t->dram[BANK_B8].x;
            words[FT_TT_QUEUE_MATMUL_B_Y] = t->dram[BANK_B8].y;
            words[FT_TT_QUEUE_MATMUL_B_BASE] = DRAM_BASE + t->idx[k] * STRIDE8;
        }
        if (arm->kexp) {
            /* The routed down + weighted sum: K = 8 experts x 16 tiles, one descriptor. */
            words[FT_TT_QUEUE_MATMUL_TILE_INNER] = RANKS * DOWN_KT;
            words[FT_TT_QUEUE_MATMUL_A_BASE] = A_DOWN_BASE;
            words[FT_TT_QUEUE_MATMUL_C_BASE] = DRAM_BASE;
            words[FT_TT_QUEUE_MATMUL_B_BASE] = arm->indirect ? DOWN_BASE : CAT_BASE;
            if (arm->indirect)
                words[FT_TT_QUEUE_MATMUL_B_N_MAJOR] |= FT_TT_MOE_FLAG_K_EXPERTS;
        }
        if (arm->custom)
            memcpy(words, arm->words, sizeof(words));
        rc = ft_tt_queue_write_matmul_parameters(&t->session.noc, &t->queue[0], k, words, &address);
        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
        descriptor.a0 = address;
        descriptor.a1 = arm->custom ? arm->a1 : 0u;
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_enqueue(&t->session.noc, &t->queue[0], &descriptor, 1u);
    }
    arm->completed = rc == FT_TT_OK && ft_tt_queue_wait_completed(&t->session.noc, &t->queue[0], arm->descriptors, TIMEOUT_MS,
                                                                  &completed) == FT_TT_OK;
    arm->microseconds = now_us() - started;
    (void)ft_tt_queue_read_field(&t->session.noc, &t->queue[0], FT_TT_QUEUE_FIELD_ERROR, &arm->error);
    {
        ft_tt_core_group solo = t->group;
        uint32_t unstopped = 0u;

        solo.cores.count = 1u;
        rc = ft_tt_watchdog_shutdown(&t->session.noc, &solo, TIMEOUT_MS, &unstopped);
        ft_tt_test_check_fmt("the triad retires when asked", rc == FT_TT_OK && unstopped == 0u,
                             "rc=%d unstopped=%u", rc, unstopped);
    }
    return dram_io(t, BANK_C, DRAM_BASE, t->observed, C_BYTES, false);
}

static size_t differing_bytes(const unsigned char *a, const unsigned char *b, size_t count)
{
    size_t n = 0;

    for (size_t i = 0; i < count; i++)
        n += a[i] != b[i];
    return n;
}

static size_t poison_bytes(const unsigned char *a, size_t count)
{
    size_t n = 0;

    for (size_t i = 0; i < count; i++)
        n += a[i] == POISON;
    return n;
}

static void stage_rounds(moe_test *t)
{
    static const uint32_t routes[3][RANKS] = {
        { 5u, 2u, 7u, 0u, 3u, 6u, 1u, 4u },
        { 1u, 1u, 6u, 3u, 7u, 0u, 2u, 5u },
        { 7u, 4u, 0u, 2u, 6u, 5u, 3u, 1u },
    };
    uint64_t dense_us = 0u, moe_us = 0u;

    for (uint32_t round = 0; round < 3u; round++) {
        moe_arm dense = { &t->bfp8, false, 0u, 0u, RANKS, false, 0u, 0u, false, false, {0u}, 0u };
        moe_arm moe = { &t->bfp6, true, STRIDE6, EXPERTS, RANKS, false, 0u, 0u, false, false, {0u}, 0u };
        int rc;

        ft_tt_test_step("round %u: experts %u %u %u %u %u %u %u %u", round, routes[round][0], routes[round][1],
                        routes[round][2], routes[round][3], routes[round][4], routes[round][5], routes[round][6],
                        routes[round][7]);
        memcpy(t->idx, routes[round], sizeof(t->idx));
        rc = upload_operands(t, 0xB6000000u + round);
        if (rc == FT_TT_OK)
            rc = write_slot(t, t->idx);
        ft_tt_test_check_fmt("operands and routing slot uploaded", rc == FT_TT_OK, "rc=%d", rc);
        if (rc != FT_TT_OK)
            return;

        rc = run_arm(t, &dense);
        memcpy(t->reference, t->observed, C_BYTES);
        ft_tt_test_check_fmt("DENSE arm (rtriad_bfp8, host-expanded expert at a host address) completes",
                             rc == FT_TT_OK && dense.completed && dense.error == 0u,
                             "rc=%d completed=%d error=%u", rc, dense.completed, dense.error);
        ft_tt_test_check_fmt("and wrote over the poison", poison_bytes(t->reference, C_BYTES) * 100u < C_BYTES,
                             "%zu of %u bytes still poison", poison_bytes(t->reference, C_BYTES), C_BYTES);

        rc = run_arm(t, &moe);
        ft_tt_test_check_fmt("MoE arm (rtriad_bfp6, BFP6 stack, expert from the slot) completes",
                             rc == FT_TT_OK && moe.completed && moe.error == 0u, "rc=%d completed=%d error=%u", rc,
                             moe.completed, moe.error);
        ft_tt_test_check_fmt("GATE: MoE output BYTE-IDENTICAL to the dense arm (8 experts x 16 tiles)",
                             differing_bytes(t->observed, t->reference, C_BYTES) == 0u, "%zu of %u bytes differ",
                             differing_bytes(t->observed, t->reference, C_BYTES), C_BYTES);
        dense_us += dense.microseconds;
        moe_us += moe.microseconds;
    }
    printf("PRICE: 8 descriptors (Mt=1 Kt=64 Nt=16), mean of 3 rounds, host wall: dense BFP8 %.1f us, "
           "MoE BFP6+indirect %.1f us (%.2f us a descriptor more; %.3f us a weight tile)\n",
           (double)dense_us / 3.0, (double)moe_us / 3.0, ((double)moe_us - (double)dense_us) / 3.0 / RANKS,
           ((double)moe_us - (double)dense_us) / 3.0 / (RANKS * EXPERT_TILES));
}

static void stage_reds(moe_test *t)
{
    uint32_t shifted[RANKS], bound[RANKS];
    moe_arm arm;
    int rc;

    ft_tt_test_step("reds: each must DIFFER from the last round's reference, or be refused");
    /* moe_idx_shift: the slot routes every rank to the next expert. */
    for (uint32_t k = 0; k < RANKS; k++)
        shifted[k] = (t->idx[k] + 1u) % EXPERTS;
    arm = (moe_arm){ &t->bfp6, true, STRIDE6, EXPERTS, RANKS, false, 0u, 0u, false, false, {0u}, 0u };
    rc = write_slot(t, shifted);
    if (rc == FT_TT_OK)
        rc = run_arm(t, &arm);
    ft_tt_test_check_fmt("RED moe_idx_shift: completes and DIFFERS",
                         rc == FT_TT_OK && arm.completed && differing_bytes(t->observed, t->reference, C_BYTES) > 0u,
                         "rc=%d completed=%d, %zu bytes differ", rc, arm.completed,
                         differing_bytes(t->observed, t->reference, C_BYTES));

    /* moe_stride_off: the residency's stride one tile long. */
    arm = (moe_arm){ &t->bfp6, true, STRIDE6 + BFP6_TILE, EXPERTS, RANKS, false, 0u, 0u, false, false, {0u}, 0u };
    rc = write_slot(t, t->idx);
    if (rc == FT_TT_OK)
        rc = run_arm(t, &arm);
    ft_tt_test_check_fmt("RED moe_stride_off: completes and DIFFERS",
                         rc == FT_TT_OK && arm.completed && differing_bytes(t->observed, t->reference, C_BYTES) > 0u,
                         "rc=%d completed=%d, %zu bytes differ", rc, arm.completed,
                         differing_bytes(t->observed, t->reference, C_BYTES));

    /* bfp6_expand_bit: the expander shifting by 1. */
    arm = (moe_arm){ &t->bfp6_red, true, STRIDE6, EXPERTS, RANKS, false, 0u, 0u, false, false, {0u}, 0u };
    rc = run_arm(t, &arm);
    ft_tt_test_check_fmt("RED bfp6_expand_bit: completes and DIFFERS",
                         rc == FT_TT_OK && arm.completed && differing_bytes(t->observed, t->reference, C_BYTES) > 0u,
                         "rc=%d completed=%d, %zu bytes differ", rc, arm.completed,
                         differing_bytes(t->observed, t->reference, C_BYTES));

    /* moe_idx_bound: rank 0 names expert EXPERTS (one past the stack). ONE descriptor: the
     * refusal ends the reader, and a stop with later descriptors queued finds MATH and PACK
     * already inside the next one with UNPACK gone (the resident triad's drain covers a stop
     * during a wait, not a reader that has exited) — that wedged core 75's backend once. */
    memcpy(bound, t->idx, sizeof(bound));
    bound[0] = EXPERTS;
    arm = (moe_arm){ &t->bfp6, true, STRIDE6, EXPERTS, 1u, false, 0u, 0u, false, false, {0u}, 0u };
    rc = write_slot(t, bound);
    if (rc == FT_TT_OK)
        rc = run_arm(t, &arm);
    ft_tt_test_check_fmt("RED moe_idx_bound: REFUSED with kErrMoeExpertIndex, nothing retired",
                         rc == FT_TT_OK && !arm.completed && arm.error == (uint32_t)kErrMoeExpertIndex,
                         "rc=%d completed=%d error=%u", rc, arm.completed, arm.error);
    /* Ranks 1..7 were never enqueued: their blocks must still be poison (rank 0's holds the
     * drained descriptor's stale-page output, which is not a result). */
    ft_tt_test_check_fmt("...and nothing outside its own block was written",
                         poison_bytes(t->observed + NT * C_TILE, C_BYTES - NT * C_TILE) == C_BYTES - NT * C_TILE,
                         "%zu of %u bytes of ranks 1..7 poison", poison_bytes(t->observed + NT * C_TILE,
                                                                               C_BYTES - NT * C_TILE),
                         C_BYTES - NT * C_TILE);

    /* AND THE GOOD PATH STILL PASSES on the same core: a refusal must not cost the core. */
    arm = (moe_arm){ &t->bfp6, true, STRIDE6, EXPERTS, RANKS, false, 0u, 0u, false, false, {0u}, 0u };
    rc = write_slot(t, t->idx);
    if (rc == FT_TT_OK)
        rc = run_arm(t, &arm);
    ft_tt_test_check_fmt("after the refusal the MoE arm runs again on the same core, IDENTICAL",
                         rc == FT_TT_OK && arm.completed && arm.error == 0u &&
                             differing_bytes(t->observed, t->reference, C_BYTES) == 0u,
                         "rc=%d completed=%d error=%u, %zu bytes differ", rc, arm.completed, arm.error,
                         differing_bytes(t->observed, t->reference, C_BYTES));
}

/**
 * The routed DOWN as ONE descriptor: K-block b reads expert slot[b]'s panel
 * (FT_TT_MOE_FLAG_K_EXPERTS), against the dense arm reading a host-built K-concatenation of
 * the same experts' BFP8 tiles. Byte-identical, or the K walk is wrong.
 */
static void stage_k_experts(moe_test *t)
{
    float *values = malloc((size_t)EXPERTS * DOWN_TILES * TILE_VALUES * sizeof(float));
    unsigned char *b6 = malloc((size_t)EXPERTS * DOWN_STRIDE6);
    unsigned char *b8 = malloc((size_t)EXPERTS * DOWN_TILES * BFP8_TILE);
    unsigned char *cat = malloc((size_t)RANKS * DOWN_TILES * BFP8_TILE);
    uint16_t *a = malloc((size_t)RANKS * DOWN_KT * A_TILE);
    uint32_t shifted[RANKS];
    moe_arm arm;
    int rc = FT_TT_ENOMEM;

    ft_tt_test_step("K-concatenated experts: the routed down + weighted sum as one descriptor (Kt = 8 x 16)");
    if (values && b6 && b8 && cat && a) {
        g_random = 0xD0000000u;
        for (size_t i = 0; i < (size_t)RANKS * DOWN_KT * TILE_VALUES; i++)
            a[i] = bf16_of(random_value());
        for (size_t i = 0; i < (size_t)EXPERTS * DOWN_TILES * TILE_VALUES; i++)
            values[i] = random_value() * 0.02f * (float)(1u << ((i / 16u) % 4u));
        rc = ft_bfp_pack_tiles(FT_BFP_BFP6_B, values, (size_t)EXPERTS * DOWN_TILES, b6) == 0 &&
                     ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, b6, (size_t)EXPERTS * DOWN_TILES, b8) == 0 &&
                     fold_in_place(b6, (size_t)EXPERTS * DOWN_TILES) == 0
                 ? FT_TT_OK
                 : FT_TT_ERROR;
        /* N-major K-concatenation: column n, block b = expert idx[b]'s column n, 16 K-tiles. */
        for (uint32_t n = 0; n < NT; n++)
            for (uint32_t b = 0; b < RANKS; b++)
                memcpy(cat + ((size_t)n * RANKS * DOWN_KT + b * DOWN_KT) * BFP8_TILE,
                       b8 + ((size_t)t->idx[b] * DOWN_TILES + n * DOWN_KT) * BFP8_TILE, DOWN_KT * BFP8_TILE);
    }
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_A, A_DOWN_BASE, a, (size_t)RANKS * DOWN_KT * A_TILE, true);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_B6, DOWN_BASE, b6, (size_t)EXPERTS * DOWN_STRIDE6, true);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_B8, CAT_BASE, cat, (size_t)RANKS * DOWN_TILES * BFP8_TILE, true);
    if (rc == FT_TT_OK)
        rc = write_slot(t, t->idx);
    ft_tt_test_check_fmt("down operands uploaded", rc == FT_TT_OK, "rc=%d", rc);
    if (rc == FT_TT_OK) {
        arm = (moe_arm){ &t->bfp8, false, 0u, 0u, 1u, true, 0u, 0u, false, false, {0u}, 0u };
        rc = run_arm(t, &arm);
        memcpy(t->reference, t->observed, NT * C_TILE);
        ft_tt_test_check_fmt("DENSE K-concatenation (rtriad_bfp8, Kt = 128) completes",
                             rc == FT_TT_OK && arm.completed && poison_bytes(t->reference, NT * C_TILE) * 100u <
                                                                  NT * C_TILE,
                             "rc=%d completed=%d", rc, arm.completed);
        arm = (moe_arm){ &t->bfp6, true, DOWN_STRIDE6, EXPERTS, 1u, true, 0u, 0u, false, false, {0u}, 0u };
        rc = run_arm(t, &arm);
        ft_tt_test_check_fmt("GATE: K-experts descriptor BYTE-IDENTICAL to the dense K-concatenation",
                             rc == FT_TT_OK && arm.completed && differing_bytes(t->observed, t->reference, NT * C_TILE) == 0u,
                             "rc=%d completed=%d, %zu bytes differ", rc, arm.completed,
                             differing_bytes(t->observed, t->reference, NT * C_TILE));
        printf("PRICE: K-experts down (Kt=128 Nt=%u, %u BFP6 tiles) %.1f us host wall\n", NT, RANKS * DOWN_TILES,
               (double)arm.microseconds);
        for (uint32_t k = 0; k < RANKS; k++)
            shifted[k] = (t->idx[k] + 1u) % EXPERTS;
        rc = write_slot(t, shifted);
        if (rc == FT_TT_OK)
            rc = run_arm(t, &arm);
        ft_tt_test_check_fmt("RED moe_idx_shift on the K-experts descriptor: DIFFERS",
                             rc == FT_TT_OK && arm.completed && differing_bytes(t->observed, t->reference, NT * C_TILE) > 0u,
                             "rc=%d, %zu bytes differ", rc, differing_bytes(t->observed, t->reference, NT * C_TILE));
    }
    free(values);
    free(b6);
    free(b8);
    free(cat);
    free(a);
}

/* LIST mode (plan S1): the batched-prefill expert descriptors. */
#define LIST8_BASE (DRAM_BASE + 0x02000000u) /* bank B8: the host-built dense equivalent */
#define LIST_N 3u
#define LIST_CELL0 8u
#define LIST_NT 56u    /* cells 8..63: entries 0..2 real (from column 8 of entry 0), entry 3 past n */
#define LIST_BLOCKS 5u /* K-concat: entries 0..2 real, 3..4 past n */

static int write_list(moe_test *t, const uint32_t *list, uint32_t n)
{
    uint32_t slot[FT_TT_MOE_SLOT_ENTRIES];

    memset(slot, 0xEE, sizeof(slot)); /* entries past n hold garbage: they must never be read */
    memcpy(slot, list, n * sizeof(uint32_t));
    slot[FT_TT_MOE_LIST_COUNT_ENTRY] = n;
    return dram_io(t, BANK_SLOT, SLOT_ADDRESS, slot, sizeof(slot), true);
}

static void list_words(moe_test *t, uint32_t *w, uint32_t kt, uint32_t nt, uint32_t a_base, uint32_t b_bank,
                       uint32_t b_base, uint32_t flags)
{
    memset(w, 0, FT_TT_QUEUE_MATMUL_WORDS * sizeof(uint32_t));
    w[FT_TT_QUEUE_MATMUL_TILE_ROWS] = 1u;
    w[FT_TT_QUEUE_MATMUL_TILE_INNER] = kt;
    w[FT_TT_QUEUE_MATMUL_TILE_COLS] = nt;
    w[FT_TT_QUEUE_MATMUL_A_X] = t->dram[BANK_A].x;
    w[FT_TT_QUEUE_MATMUL_A_Y] = t->dram[BANK_A].y;
    w[FT_TT_QUEUE_MATMUL_A_BASE] = a_base;
    w[FT_TT_QUEUE_MATMUL_B_X] = t->dram[b_bank].x;
    w[FT_TT_QUEUE_MATMUL_B_Y] = t->dram[b_bank].y;
    w[FT_TT_QUEUE_MATMUL_B_BASE] = b_base;
    w[FT_TT_QUEUE_MATMUL_C_X] = t->dram[BANK_C].x;
    w[FT_TT_QUEUE_MATMUL_C_Y] = t->dram[BANK_C].y;
    w[FT_TT_QUEUE_MATMUL_C_BASE] = DRAM_BASE;
    w[FT_TT_QUEUE_MATMUL_DEPTH] = DEPTH;
    w[FT_TT_QUEUE_MATMUL_B_N_MAJOR] = 1u | (1u << 8) | flags;
}

/** Gate/up LIST descriptor (cells 8..63 of the entry x column grid, entry 3 past n) and the
 *  K-concatenated LIST down (5 blocks, 2 past n): each BYTE-IDENTICAL to rtriad_bfp8 over the
 *  host-built dense equivalent (expanded experts, zero columns / blocks past n). */
static void stage_list(moe_test *t)
{
    static const uint32_t list[LIST_N] = { 3u, 0u, 6u };
    unsigned char *dense = calloc((size_t)LIST_NT * KT, BFP8_TILE);
    unsigned char *reference = malloc(C_BYTES);
    moe_arm arm;
    int rc;

    ft_tt_test_step("LIST mode (plan S1): gate/up over cells %u..%u of list {3, 0, 6}, n = %u; K-concat %u blocks",
                    LIST_CELL0, LIST_CELL0 + LIST_NT - 1u, LIST_N, LIST_BLOCKS);
    if (!dense || !reference || upload_operands(t, 0xA1000000u) != FT_TT_OK) {
        free(dense);
        free(reference);
        return;
    }
    for (uint32_t j = 0; j < LIST_NT; j++) {
        const uint32_t cell = LIST_CELL0 + j, s = cell / NT, col = cell % NT;

        if (s < LIST_N)
            memcpy(dense + (size_t)j * KT * BFP8_TILE,
                   t->b8 + ((size_t)list[s] * EXPERT_TILES + (size_t)col * KT) * BFP8_TILE, (size_t)KT * BFP8_TILE);
    }
    rc = dram_io(t, BANK_B8, LIST8_BASE, dense, (size_t)LIST_NT * KT * BFP8_TILE, true);
    if (rc == FT_TT_OK)
        rc = write_list(t, list, LIST_N);
    memset(&arm, 0, sizeof(arm));
    arm.program = &t->bfp8;
    arm.custom = true;
    arm.descriptors = 1u;
    list_words(t, arm.words, KT, LIST_NT, DRAM_BASE, BANK_B8, LIST8_BASE, 0u);
    if (rc == FT_TT_OK)
        rc = run_arm(t, &arm);
    memcpy(reference, t->observed, C_BYTES);
    ft_tt_test_check_fmt("dense reference arm completes", rc == FT_TT_OK && arm.completed, "rc=%d", rc);
    memset(&arm, 0, sizeof(arm));
    arm.program = &t->bfp6;
    arm.custom = true;
    arm.descriptors = 1u;
    arm.indirect = true;
    arm.stride = STRIDE6;
    arm.count = EXPERTS;
    arm.a1 = ftmoe_list_a1(0u, LIST_CELL0, NT);
    list_words(t, arm.words, KT, LIST_NT, DRAM_BASE, BANK_B6, DRAM_BASE, 0u);
    if (rc == FT_TT_OK)
        rc = run_arm(t, &arm);
    ft_tt_test_check_fmt("GATE: LIST gate/up BYTE-IDENTICAL to the dense equivalent (zero columns past n)",
                         rc == FT_TT_OK && arm.completed &&
                             differing_bytes(reference, t->observed, (size_t)LIST_NT * C_TILE) == 0u,
                         "rc=%d completed=%d error=%u, %zu bytes differ", rc, arm.completed, arm.error,
                         differing_bytes(reference, t->observed, (size_t)LIST_NT * C_TILE));
    printf("PRICE: LIST gate/up (Kt=64 Nt=%u, %u real BFP6 columns) %.1f us host wall\n", LIST_NT,
           (LIST_N * NT - LIST_CELL0), (double)arm.microseconds);
    {
        static const uint32_t shifted[LIST_N] = { 4u, 0u, 6u };

        if (write_list(t, shifted, LIST_N) == FT_TT_OK && run_arm(t, &arm) == FT_TT_OK)
            ft_tt_test_check_fmt("RED list_entry_shift: completes and DIFFERS",
                                 arm.completed && differing_bytes(reference, t->observed, (size_t)LIST_NT * C_TILE) > 0u,
                                 "%zu bytes differ", differing_bytes(reference, t->observed, (size_t)LIST_NT * C_TILE));
    }
    {
        /* K-concat LIST down over the down stack the K-experts stage uploaded? No: build one here. */
        float *values = malloc((size_t)EXPERTS * DOWN_TILES * TILE_VALUES * sizeof(float));
        unsigned char *b6 = malloc((size_t)EXPERTS * DOWN_TILES * BFP6_TILE);
        unsigned char *b8 = malloc((size_t)EXPERTS * DOWN_TILES * BFP8_TILE);
        unsigned char *cat = calloc((size_t)LIST_BLOCKS * DOWN_TILES, BFP8_TILE);
        uint16_t *a = malloc((size_t)LIST_BLOCKS * DOWN_KT * A_TILE);
        static const uint32_t down_list[LIST_N] = { 2u, 7u, 1u };

        rc = FT_TT_ENOMEM;
        if (values && b6 && b8 && cat && a) {
            g_random = 0xA2000000u;
            for (size_t i = 0; i < (size_t)LIST_BLOCKS * DOWN_KT * TILE_VALUES; i++)
                a[i] = bf16_of(random_value());
            for (size_t i = 0; i < (size_t)EXPERTS * DOWN_TILES * TILE_VALUES; i++)
                values[i] = random_value() * 0.02f * (float)(1u << ((i / 16u) % 4u));
            rc = ft_bfp_pack_tiles(FT_BFP_BFP6_B, values, (size_t)EXPERTS * DOWN_TILES, b6) == 0 &&
                         ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, b6, (size_t)EXPERTS * DOWN_TILES, b8) == 0 &&
                         fold_in_place(b6, (size_t)EXPERTS * DOWN_TILES) == 0
                     ? FT_TT_OK
                     : FT_TT_ERROR;
            for (uint32_t n = 0; n < NT; n++)
                for (uint32_t b = 0; b < LIST_N; b++)
                    memcpy(cat + ((size_t)n * LIST_BLOCKS * DOWN_KT + b * DOWN_KT) * BFP8_TILE,
                           b8 + ((size_t)down_list[b] * DOWN_TILES + n * DOWN_KT) * BFP8_TILE, DOWN_KT * BFP8_TILE);
        }
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_A, A_DOWN_BASE, a, (size_t)LIST_BLOCKS * DOWN_KT * A_TILE, true);
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_B6, DOWN_BASE, b6, (size_t)EXPERTS * DOWN_STRIDE6, true);
        if (rc == FT_TT_OK)
            rc = dram_io(t, BANK_B8, LIST8_BASE, cat, (size_t)LIST_BLOCKS * DOWN_TILES * BFP8_TILE, true);
        if (rc == FT_TT_OK)
            rc = write_list(t, down_list, LIST_N);
        memset(&arm, 0, sizeof(arm));
        arm.program = &t->bfp8;
        arm.custom = true;
        arm.descriptors = 1u;
        list_words(t, arm.words, LIST_BLOCKS * DOWN_KT, NT, A_DOWN_BASE, BANK_B8, LIST8_BASE, 0u);
        if (rc == FT_TT_OK)
            rc = run_arm(t, &arm);
        memcpy(reference, t->observed, C_BYTES);
        memset(&arm, 0, sizeof(arm));
        arm.program = &t->bfp6;
        arm.custom = true;
        arm.descriptors = 1u;
        arm.indirect = true;
        arm.stride = DOWN_STRIDE6;
        arm.count = EXPERTS;
        arm.a1 = ftmoe_list_a1(0u, 0u, NT);
        list_words(t, arm.words, LIST_BLOCKS * DOWN_KT, NT, A_DOWN_BASE, BANK_B6, DOWN_BASE,
                   FT_TT_MOE_FLAG_K_EXPERTS);
        if (rc == FT_TT_OK)
            rc = run_arm(t, &arm);
        ft_tt_test_check_fmt("GATE: LIST K-concat down BYTE-IDENTICAL to the dense K-concatenation (blocks past n zero)",
                             rc == FT_TT_OK && arm.completed &&
                                 differing_bytes(reference, t->observed, (size_t)NT * C_TILE) == 0u,
                             "rc=%d completed=%d error=%u, %zu bytes differ", rc, arm.completed, arm.error,
                             differing_bytes(reference, t->observed, (size_t)NT * C_TILE));
        free(values);
        free(b6);
        free(b8);
        free(cat);
        free(a);
    }
    free(dense);
    free(reference);
}

int main(int argc, char **argv)
{
    static moe_test t;
    int rc;

    if (argc < 4) {
        fprintf(stderr, "usage: %s <rtriad_bfp8-dir> <rtriad_bfp6-dir> <rtriad_bfp6_red-dir>\n", argv[0]);
        return 2;
    }
    t.index = 75u;
    t.a = malloc((size_t)KT * A_TILE);
    t.b6 = malloc((size_t)EXPERTS * STRIDE6);
    t.b8 = malloc((size_t)EXPERTS * STRIDE8);
    t.reference = malloc(C_BYTES);
    t.observed = malloc(C_BYTES);
    if (!t.a || !t.b6 || !t.b8 || !t.reference || !t.observed)
        return 1;

    rc = ft_tt_hardware_session_open(&t.session);
    ft_tt_test_check_fmt("card opens", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return ft_tt_test_finish();
    (void)ft_tt_tensix_index_from_environment(&t.index);
    rc = ft_tt_hardware_session_claim_cores(&t.session, &t.index, 1u, "qwen36 phase 5 moe matmul", NULL, NULL);
    for (uint32_t bank = 0; bank < 8u && rc == FT_TT_OK; bank++)
        rc = ft_tt_coord_dram_bank(&t.session.grid, bank, FT_TT_COORD_TRANSLATED, &t.dram[bank]);
    if (rc == FT_TT_OK)
        rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &t.firmware);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &t.bfp8);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[2], TRIAD_MASK, &t.bfp6);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[3], TRIAD_MASK, &t.bfp6_red);
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
    ft_tt_test_check_fmt("core claimed, programs loaded, firmware up", rc == FT_TT_OK, "index %u at (%u,%u): %s",
                         t.index, t.placement[0].x, t.placement[0].y, ft_tt_error_string(rc));
    if (rc == FT_TT_OK) {
        stage_rounds(&t);
        stage_reds(&t);
        stage_k_experts(&t);
        stage_list(&t);
    }
    ft_tt_hardware_session_close(&t.session);
    return ft_tt_test_finish();
}
