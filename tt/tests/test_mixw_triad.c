/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_mixw_triad.c
 * @brief On the card: `rtriad_mixw`, the resident matmul
 *   triad that takes Bfp8_b OR Bfp4_b weights per descriptor (word 13 bit 30) in one residency.
 *
 * THE GATE IS BYTE IDENTITY, NOT PCC, against the two programs each format already ships on:
 *   Bfp8 descriptors under rtriad_mixw = rtriad_bfp8 on the same descriptors (same compute
 *     source and CB table; the only difference is the mixw code path, so any byte is its bug)
 *   Bfp4 descriptors under rtriad_mixw = the one-shot `matmul_bfp4` on the same weight
 *     (spec 6.1: Bfp4 resident outputs equal one-shot; if they differ, REPORT, don't tune --
 *     the resident NB-block summation order is the suspect, not the format switch)
 *
 * One core (FT_TT_TENSIX_INDEX, default 75). The shipped Qwen3.8 decode shape: Mt = 1,
 * Nt = 16, depth 4, NB 8, N-major, pipelined reader; Kt 160 (hidden) and 544 (ffn_down). Each
 * arm runs FOUR descriptors, one output block each, formats alternating 8/4/8/4 so every
 * descriptor switches SrcA (the lazy switch's worst case); two rounds of fresh weights per Kt,
 * output poisoned before every arm. A uniform-Bfp4 arm (one switch, then no-ops) too.
 *
 * Red (must DIFFER): descriptor 1's Bfp4 weight sent WITHOUT bit 30, so the card reads 576 B
 * tiles as 1,088 B ones -- a wrong answer with a clean run, which is why the host check
 * (ft_tt_queue_matmul_check_weight) is the only guard; the test asserts that check refuses the
 * word. The descriptors after it must still be right (the format state recovers).
 *
 * Usage (detached, cores idle; the blob's programs exploded or built under <prog>):
 *   FT_TT_FIRMWARE_DIR=<prog>/rtriad_bfp8 FT_TT_TENSIX_INDEX=75 \
 *     ./build-mixw/test_mixw_triad <prog>/rtriad_bfp8 <prog>/rtriad_mixw <prog>/matmul_bfp4
 *
 * @status New 2026-11-13 (plan 3.2); NOT RUN: the card proof is plan 3.4.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_bfp.h"
#include "../kernels/resident_matmul_block_protocol.h"
#include "../kernels/resident_matmul_multicast_protocol.h"
#include "ft_tt_group.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_op_matmul.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "ft_tt_watchdog.h"

#define KT_MAX 544u
#define NT 16u
#define DEPTH 4u
#define COLUMN_BLOCK 8u
#define DESCRIPTORS 4u
#define WEIGHT_SETS 2u /* per format: descriptors 0/2 read Bfp8 set 0/1, 1/3 Bfp4 set 0/1 */
#define TILE_VALUES 1024u
#define A_TILE 2048u
#define C_TILE 2048u
#define BFP8_TILE FT_TT_QUEUE_MATMUL_WEIGHT_TILE_BYTES_BFP8
#define BFP4_TILE FT_TT_QUEUE_MATMUL_WEIGHT_TILE_BYTES_BFP4
#define BLOCK_BYTES (NT * C_TILE)
#define C_BYTES (DESCRIPTORS * BLOCK_BYTES)
#define SET8_BYTES (KT_MAX * NT * BFP8_TILE) /* 9,469,952 B: one set's stride in bank B8 */
#define SET4_BYTES (KT_MAX * NT * BFP4_TILE)

#define BANK_A 0u
#define BANK_B8 1u
#define BANK_B4 2u
#define BANK_C 3u
#define DRAM_BASE 0x00200000u

#define CB_IN0 0u
#define CB_IN1 1u
#define CB_OUT 16u
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
    ft_tt_program_directory bfp8, mixw, oneshot4;
    ft_tt_launch_config config;
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    uint32_t kt;
    uint16_t *a;
    unsigned char *b8, *b4, *observed, *ref8, *ref4;
} mixw_test;

/** One resident arm: which program, and per descriptor the weight format it reads and the
 *  flag bit it is sent with (they disagree only in the red). */
typedef struct {
    const ft_tt_program_directory *program;
    bool bfp4[DESCRIPTORS];
    bool flagged[DESCRIPTORS];
    bool completed; /* out */
    uint32_t error; /* out: the queue's error field */
} mixw_arm;

static uint32_t g_random = 1u;

static float random_value(void)
{
    g_random = g_random * 1664525u + 1013904223u;
    return ((float)(g_random >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static uint16_t bf16_of(float value)
{
    uint32_t u;

    memcpy(&u, &value, 4);
    return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

static int dram_io(mixw_test *t, uint32_t bank, uint32_t address, void *bytes, size_t count, bool write)
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

/** Fresh activation and WEIGHT_SETS fresh sets per format (Kt x Nt tiles each, N-major in
 *  DRAM: the reader and the one-shot request both read them that way), uploaded. */
static int upload_operands(mixw_test *t, uint32_t seed)
{
    const size_t tiles = (size_t)t->kt * NT;
    float *values = malloc(tiles * TILE_VALUES * sizeof(float));
    int rc = values ? FT_TT_OK : FT_TT_ENOMEM;

    g_random = seed;
    for (size_t i = 0; i < (size_t)t->kt * TILE_VALUES; i++)
        t->a[i] = bf16_of(random_value());
    for (uint32_t set = 0; set < 2u * WEIGHT_SETS && rc == FT_TT_OK; set++) {
        const bool bfp4 = set >= WEIGHT_SETS;
        unsigned char *out = bfp4 ? t->b4 + (set - WEIGHT_SETS) * (size_t)SET4_BYTES
                                  : t->b8 + set * (size_t)SET8_BYTES;

        /* A few octaves per tile, so shared exponents differ block to block. */
        for (size_t i = 0; i < tiles * TILE_VALUES; i++)
            values[i] = random_value() * 0.02f * (float)(1u << ((i / 16u) % 4u));
        if (ft_bfp_pack_tiles(bfp4 ? FT_BFP_BFP4_B : FT_BFP_BFP8_B, values, tiles, out) != FT_BFP_OK)
            rc = FT_TT_ERROR;
    }
    free(values);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_A, DRAM_BASE, t->a, (size_t)t->kt * A_TILE, true);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_B8, DRAM_BASE, t->b8, WEIGHT_SETS * (size_t)SET8_BYTES, true);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_B4, DRAM_BASE, t->b4, WEIGHT_SETS * (size_t)SET4_BYTES, true);
    return rc;
}

static int poison_output(mixw_test *t)
{
    memset(t->observed, POISON, C_BYTES);
    return dram_io(t, BANK_C, DRAM_BASE, t->observed, C_BYTES, true);
}

/** rtriad_bfp8's CBs: two chunks of depth x NB 1,088 B pages for the weight, whichever format
 *  a descriptor's tiles are (spec 6.1: one CB, no reconfigure). */
static int place_cbs(mixw_test *t)
{
    ft_tt_op_cb_request requests[3] = {
        { CB_IN0, A_TILE, DEPTH * 2u },
        { CB_IN1, BFP8_TILE, DEPTH * COLUMN_BLOCK * 2u },
        { CB_OUT, C_TILE, 2u },
    };
    ft_tt_launch_cb_entry placed[3];

    t->config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    t->config.brisc_noc_id = 0u;
    return ft_tt_op_cb_place(requests, 3u, placed, t->cb_buffer, sizeof(t->cb_buffer),
                             &t->config.local_cb_mask);
}

/** Load, stage and kick one triad: SOLO cooperative tail, ten queue args on every processor. */
static int start_triad(mixw_test *t, const ft_tt_program_directory *program)
{
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0, FT_TT_PROC_MATH1,
                                            FT_TT_PROC_MATH2 };
    const ft_tt_launch_core_placement *core = &t->placement[0];
    uint32_t args[5][kArgCoopCount];
    ft_tt_op_args blocks[5];
    int rc;

    memset(args, 0, sizeof(args));
    rc = ft_tt_op_program_load(&t->session.noc, core->x, core->y, &program->program, NULL, &t->config);
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
        blocks[p].count = p == 0u ? kArgCoopCount : FT_TT_QUEUE_KERNEL_ARG_COUNT;
    }
    args[0][kArgCoopRole] = FT_TT_RESIDENT_COOP_ROLE_SOLO;
    if (rc == FT_TT_OK)
        rc = poison_output(t);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_stage(&t->session.noc, core->x, core->y, &t->config, t->cb_buffer, sizeof(t->cb_buffer),
                            blocks, 5u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_send_go(&t->session.noc, core->x, core->y);
    return rc;
}

/** Descriptor k's sixteen words: the decode shape, weight set k/2 of its format, block k out. */
static void descriptor_words(const mixw_test *t, const mixw_arm *arm, uint32_t k,
                             uint32_t words[FT_TT_QUEUE_MATMUL_WORDS])
{
    const uint32_t bank = arm->bfp4[k] ? BANK_B4 : BANK_B8;

    memset(words, 0, FT_TT_QUEUE_MATMUL_WORDS * sizeof(uint32_t));
    words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = 1u;
    words[FT_TT_QUEUE_MATMUL_TILE_INNER] = t->kt;
    words[FT_TT_QUEUE_MATMUL_TILE_COLS] = NT;
    words[FT_TT_QUEUE_MATMUL_A_X] = t->dram[BANK_A].x;
    words[FT_TT_QUEUE_MATMUL_A_Y] = t->dram[BANK_A].y;
    words[FT_TT_QUEUE_MATMUL_A_BASE] = DRAM_BASE;
    words[FT_TT_QUEUE_MATMUL_B_X] = t->dram[bank].x;
    words[FT_TT_QUEUE_MATMUL_B_Y] = t->dram[bank].y;
    words[FT_TT_QUEUE_MATMUL_B_BASE] =
        DRAM_BASE + (k / 2u) * (arm->bfp4[k] ? (uint32_t)SET4_BYTES : (uint32_t)SET8_BYTES);
    words[FT_TT_QUEUE_MATMUL_C_X] = t->dram[BANK_C].x;
    words[FT_TT_QUEUE_MATMUL_C_Y] = t->dram[BANK_C].y;
    words[FT_TT_QUEUE_MATMUL_C_BASE] = DRAM_BASE + k * BLOCK_BYTES;
    words[FT_TT_QUEUE_MATMUL_DEPTH] = DEPTH;
    words[FT_TT_QUEUE_MATMUL_B_N_MAJOR] =
        1u | (1u << 8) | (arm->flagged[k] ? FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4 : 0u);
    words[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] = ftmm_block_word(1u, COLUMN_BLOCK);
}

/** Run one arm's four descriptors and read the four output blocks into `observed`. */
static int run_arm(mixw_test *t, mixw_arm *arm)
{
    uint32_t completed = 0u;
    int rc = start_triad(t, arm->program);

    for (uint32_t k = 0; k < DESCRIPTORS && rc == FT_TT_OK; k++) {
        uint32_t words[FT_TT_QUEUE_MATMUL_WORDS];
        ft_tt_queue_descriptor descriptor;
        uint32_t address = 0u;

        descriptor_words(t, arm, k, words);
        rc = ft_tt_queue_write_matmul_parameters(&t->session.noc, &t->queue[0], k, words, &address);
        memset(&descriptor, 0, sizeof(descriptor));
        descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
        descriptor.a0 = address;
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_enqueue(&t->session.noc, &t->queue[0], &descriptor, 1u);
    }
    arm->completed = rc == FT_TT_OK && ft_tt_queue_wait_completed(&t->session.noc, &t->queue[0], DESCRIPTORS,
                                                                  TIMEOUT_MS, &completed) == FT_TT_OK;
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

/** The Bfp4 reference: one-shot matmul_bfp4 on weight set `set`, written to block `block`. */
static int run_oneshot_bfp4(mixw_test *t, uint32_t set, uint32_t block)
{
    ft_tt_op_matmul_request request;

    memset(&request, 0, sizeof(request));
    request.tile_rows = 1u;
    request.tile_inner = t->kt;
    request.tile_cols = NT;
    request.prefetch_tiles = DEPTH;
    request.weight_n_major = 1u;
    request.activation = (ft_tt_op_operand){ (uint16_t)t->dram[BANK_A].x, (uint16_t)t->dram[BANK_A].y,
                                             DRAM_BASE, A_TILE };
    request.weight = (ft_tt_op_operand){ (uint16_t)t->dram[BANK_B4].x, (uint16_t)t->dram[BANK_B4].y,
                                         DRAM_BASE + set * (uint32_t)SET4_BYTES, BFP4_TILE };
    request.output = (ft_tt_op_operand){ (uint16_t)t->dram[BANK_C].x, (uint16_t)t->dram[BANK_C].y,
                                         DRAM_BASE + block * BLOCK_BYTES, C_TILE };
    return ft_tt_op_matmul_run(&t->session.noc, t->placement[0].x, t->placement[0].y, &t->oneshot4.program,
                               &request, TIMEOUT_MS);
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

/** Block k of `observed` against block k of `reference`; returns the differing bytes. */
static size_t block_diff(const mixw_test *t, const unsigned char *reference, uint32_t k)
{
    return differing_bytes(t->observed + k * BLOCK_BYTES, reference + k * BLOCK_BYTES, BLOCK_BYTES);
}

/** The two references for this round: rtriad_bfp8 (every descriptor Bfp8) and the one-shot
 *  matmul_bfp4 into the odd blocks. */
static bool build_references(mixw_test *t)
{
    mixw_arm ref8 = { &t->bfp8, { false, false, false, false }, { false, false, false, false }, false, 0u };
    int rc = run_arm(t, &ref8);

    memcpy(t->ref8, t->observed, C_BYTES);
    ft_tt_test_check_fmt("REFERENCE rtriad_bfp8 (4 Bfp8 descriptors) completes and overwrote the poison",
                         rc == FT_TT_OK && ref8.completed && ref8.error == 0u &&
                             poison_bytes(t->ref8, C_BYTES) * 100u < C_BYTES,
                         "rc=%d completed=%d error=%u poison=%zu", rc, ref8.completed, ref8.error,
                         poison_bytes(t->ref8, C_BYTES));
    rc = poison_output(t);
    for (uint32_t k = 1u; k < DESCRIPTORS && rc == FT_TT_OK; k += 2u)
        rc = run_oneshot_bfp4(t, k / 2u, k);
    if (rc == FT_TT_OK)
        rc = dram_io(t, BANK_C, DRAM_BASE, t->ref4, C_BYTES, false);
    ft_tt_test_check_fmt("REFERENCE one-shot matmul_bfp4 (blocks 1, 3) completes and overwrote the poison",
                         rc == FT_TT_OK && poison_bytes(t->ref4 + BLOCK_BYTES, BLOCK_BYTES) * 100u < BLOCK_BYTES,
                         "rc=%d", rc);
    return rc == FT_TT_OK;
}

/** The host guard on every green word, as Pantheon will call it (plan 3.3). */
static bool words_pass_host_check(const mixw_arm *arm, bool mixw_residency)
{
    for (uint32_t k = 0; k < DESCRIPTORS; k++) {
        const uint32_t word = 1u | (1u << 8) | (arm->flagged[k] ? FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4 : 0u);

        if (ft_tt_queue_matmul_check_weight(arm->bfp4[k] ? BFP4_TILE : BFP8_TILE, word, mixw_residency) != FT_TT_OK)
            return false;
    }
    return true;
}

static void stage_round(mixw_test *t, uint32_t round)
{
    mixw_arm mixed = { &t->mixw, { false, true, false, true }, { false, true, false, true }, false, 0u };
    mixw_arm uniform4 = { &t->mixw, { true, true, true, true }, { true, true, true, true }, false, 0u };
    int rc;

    ft_tt_test_step("Kt %u round %u", t->kt, round);
    rc = upload_operands(t, 0x38000000u + t->kt * 16u + round);
    ft_tt_test_check_fmt("operands uploaded", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK || !build_references(t))
        return;

    ft_tt_test_check("the green words pass the host check", words_pass_host_check(&mixed, true),
                     "ft_tt_queue_matmul_check_weight, mixw residency");
    rc = run_arm(t, &mixed);
    ft_tt_test_check_fmt("MIXED arm (rtriad_mixw, formats 8/4/8/4) completes",
                         rc == FT_TT_OK && mixed.completed && mixed.error == 0u, "rc=%d completed=%d error=%u", rc,
                         mixed.completed, mixed.error);
    ft_tt_test_check_fmt("GATE: Bfp8 descriptors 0, 2 BYTE-IDENTICAL to rtriad_bfp8",
                         block_diff(t, t->ref8, 0u) == 0u && block_diff(t, t->ref8, 2u) == 0u,
                         "%zu + %zu bytes differ", block_diff(t, t->ref8, 0u), block_diff(t, t->ref8, 2u));
    ft_tt_test_check_fmt("GATE: Bfp4 descriptors 1, 3 BYTE-IDENTICAL to one-shot matmul_bfp4",
                         block_diff(t, t->ref4, 1u) == 0u && block_diff(t, t->ref4, 3u) == 0u,
                         "%zu + %zu bytes differ (report, don't tune)", block_diff(t, t->ref4, 1u),
                         block_diff(t, t->ref4, 3u));

    /* Uniform Bfp4: descriptors 0/2 read Bfp4 set 0/1 too, so blocks 0/2 equal blocks 1/3's
     * references; only the odd blocks have a one-shot reference written at their address. */
    rc = run_arm(t, &uniform4);
    ft_tt_test_check_fmt("GATE: UNIFORM Bfp4 arm, descriptors 1, 3 BYTE-IDENTICAL to one-shot matmul_bfp4",
                         rc == FT_TT_OK && uniform4.completed && block_diff(t, t->ref4, 1u) == 0u &&
                             block_diff(t, t->ref4, 3u) == 0u,
                         "rc=%d completed=%d, %zu + %zu bytes differ", rc, uniform4.completed,
                         block_diff(t, t->ref4, 1u), block_diff(t, t->ref4, 3u));
}

static void stage_red(mixw_test *t)
{
    mixw_arm red = { &t->mixw, { false, true, false, true }, { false, false, false, true }, false, 0u };
    int rc;

    ft_tt_test_step("red: descriptor 1's Bfp4 weight sent WITHOUT bit 30");
    ft_tt_test_check("the host check REFUSES that word (the only guard there is)",
                     !words_pass_host_check(&red, true), "ft_tt_queue_matmul_check_weight, mixw residency");
    rc = run_arm(t, &red);
    ft_tt_test_check_fmt("RED: completes clean (DRAM tiles carry no header) and block 1 DIFFERS",
                         rc == FT_TT_OK && red.completed && red.error == 0u && block_diff(t, t->ref4, 1u) > 0u,
                         "rc=%d completed=%d error=%u, %zu bytes differ", rc, red.completed, red.error,
                         block_diff(t, t->ref4, 1u));
    ft_tt_test_check_fmt("and the format state recovers: blocks 2 (Bfp8) and 3 (Bfp4) still match",
                         block_diff(t, t->ref8, 2u) == 0u && block_diff(t, t->ref4, 3u) == 0u,
                         "%zu + %zu bytes differ", block_diff(t, t->ref8, 2u), block_diff(t, t->ref4, 3u));
}

/** Claim the core, load the three program directories, boot the firmware. */
static int open_card(mixw_test *t, char **argv)
{
    int rc = ft_tt_hardware_session_open(&t->session);

    if (rc != FT_TT_OK)
        return rc;
    (void)ft_tt_tensix_index_from_environment(&t->index);
    rc = ft_tt_hardware_session_claim_cores(&t->session, &t->index, 1u, "qwen38 mixw triad", NULL, NULL);
    for (uint32_t bank = 0; bank < 8u && rc == FT_TT_OK; bank++)
        rc = ft_tt_coord_dram_bank(&t->session.grid, bank, FT_TT_COORD_TRANSLATED, &t->dram[bank]);
    if (rc == FT_TT_OK)
        rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &t->firmware);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &t->bfp8);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[2], TRIAD_MASK, &t->mixw);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[3], TRIAD_MASK, &t->oneshot4);
    if (rc == FT_TT_OK) {
        memset(&t->group, 0, sizeof(t->group));
        t->group.first_tensix_index = t->index;
        t->group.tensix_index_count = 1u;
        t->group.queues = t->queue;
        t->group.resident_processor_mask = TRIAD_MASK;
        rc = ft_tt_launch_core_set_init(&t->group.cores, t->placement, 1u);
        if (rc == FT_TT_OK)
            rc = ft_tt_launch_core_set_add_tensix(&t->group.cores, &t->session.grid, t->index, NULL);
    }
    if (rc == FT_TT_OK)
        rc = ft_tt_core_boot_firmware(&t->session.noc, t->placement[0].x, t->placement[0].y, &t->firmware, NULL);
    return rc;
}

int main(int argc, char **argv)
{
    static const uint32_t shapes[2] = { 160u, KT_MAX };
    static mixw_test t;
    int rc;

    if (argc < 4) {
        fprintf(stderr, "usage: %s <rtriad_bfp8-dir> <rtriad_mixw-dir> <matmul_bfp4-dir>\n", argv[0]);
        return 2;
    }
    t.index = 75u;
    t.a = malloc((size_t)KT_MAX * A_TILE);
    t.b8 = malloc(WEIGHT_SETS * (size_t)SET8_BYTES);
    t.b4 = malloc(WEIGHT_SETS * (size_t)SET4_BYTES);
    t.observed = malloc(C_BYTES);
    t.ref8 = malloc(C_BYTES);
    t.ref4 = malloc(C_BYTES);
    if (!t.a || !t.b8 || !t.b4 || !t.observed || !t.ref8 || !t.ref4)
        return 1;

    rc = open_card(&t, argv);
    ft_tt_test_check_fmt("core claimed, programs loaded, firmware up", rc == FT_TT_OK, "index %u: %s", t.index,
                         ft_tt_error_string(rc));
    for (uint32_t s = 0; s < 2u && rc == FT_TT_OK; s++) {
        t.kt = shapes[s];
        for (uint32_t round = 0; round < 2u; round++)
            stage_round(&t, round);
    }
    if (rc == FT_TT_OK)
        stage_red(&t);
    ft_tt_hardware_session_close(&t.session);
    return ft_tt_test_finish();
}
