/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_triad.c
 * @brief The RESIDENT TRIAD on the card: a real matmul running through a reader/compute/
 *   writer triad that never returns between ops, that can be asked to stop while it is
 *   blocked in `cb_wait_front`, and that leaves the core reusable afterwards.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS FILE HAS TO SETTLE, AND HOW EACH STAGE SETTLES IT
 * ---------------------------------------------------------------------------
 * Phase 5 left one design question: its resident kernel is BRISC-only and CB-free on
 * purpose, because `cb_wait_front` is the one wait the host cannot break and killing a
 * kernel blocked in it is how Tensix 0 was wedged (dev notes §1). A real op needs a triad.
 * So a resident triad either makes the CB wait stop-aware or accepts `assert_reset` as its
 * only exit, and the second is not acceptable.
 *
 *   stage 1  the triad computes a REAL matmul, byte-for-byte identical to the Phase 4
 *            slow-dispatch arm on the same inputs, over three different input sets with the
 *            destination poisoned before each — so "identical" cannot be stale DRAM
 *   stage 2  the dispatch A/B on a REAL op. Every Phase 5 figure is for a NOP payload; this
 *            is what a resident triad does to a matmul's 0.245 ms fixed cost, in GFLOP/s
 *            and in % of one Tensix's 1.64 TFLOP/s derived peak
 *   stage 3  what the stoppable wait COSTS: the same triad built with upstream's own
 *            blocking waits versus the stoppable ones, interleaved, same shape
 *   stage 4  the acceptance. A DELIBERATE CB DEADLOCK — the reader withholds the last block
 *            of a real matmul, so the compute thread is stuck in the op's own
 *            `cb_wait_front` — caught by the watchdog, which names WHICH PROCESSOR and
 *            WHICH WAIT; retired by ONE WRITE; and then the SAME CORE runs a fresh matmul
 *            correctly
 *   stage 5  the RED PROOF for stage 4: the same deadlock against a build whose stop check
 *            is deliberately removed. It must NOT stop when asked. This is safe to run only
 *            because every wait is bounded independently of the ask, so the worst case is
 *            `kErrCbWaitTimeout` and not a wedged core
 *   stage 6  group-level completion: two cores, a sequence number the KERNELS write, and
 *            the check refusing when one member is behind
 *
 * ---------------------------------------------------------------------------
 * WHY BYTE-EXACTNESS AND NOT PCC
 * ---------------------------------------------------------------------------
 * PCC is blind to a consistent permutation and degrades gently under a smooth corruption
 * (dev notes §3), and both arms here run the SAME compute source — so a difference between
 * them is a plumbing bug, and the sensitive instrument for a plumbing bug is a byte
 * comparison, not a correlation. The Phase 4 arm is already PCC-validated against a CPU
 * fp32 reference (`tests/test_ops.c`, bring-up log §14); this file's job is to show the
 * resident path does not change the answer. "Nine identical digits is evidence; an
 * identical file is proof."
 *
 * ---------------------------------------------------------------------------
 * SAFETY
 * ---------------------------------------------------------------------------
 * NOTHING HERE ASSERTS RESET. Every recovery is `stop_request`. Cores are parked only after
 * every processor has reported EXIT — and for a triad that means all five status slots, not
 * the control block's single `state` word, which has one writer and would happily read EXIT
 * while a peer was still spinning.
 *
 * The kWaitUpstream build is NEVER deadlocked. It has no escape and no bound; deadlocking it
 * is exactly the wedge this phase exists to prevent, and it exists only as the cost arm.
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=../programs/rtriad FT_TT_TENSIX_INDICES=60,61 \
 *     ./build-phase6/test_phase6_triad ../programs/rtriad ../programs/rtriad_upstream \
 *         ../programs/rtriad_boundonly ../programs/matmul
 *
 * @status
 * @issues None
 * @todo None
 */

/* clock_gettime and CLOCK_MONOTONIC are POSIX.1-2008; -std=c99 turns off glibc's default
 * feature set, so ask for them by name or the Linux card host's gcc hides them. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_group.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_op_matmul.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "ft_tt_tilize.h"
#include "ft_tt_watchdog.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================
 * The shape, and the constants the numbers are quoted against
 * ================================================================ */

/** One Tensix's derived peak at this card's measured 800 MHz busy clock, HiFi4. Every
 *  figure in this file is PER CORE and a fraction of THIS — which is what makes it portable
 *  to a 120-core p150 (dev notes §6). */
#define TENSIX_PEAK_GFLOPS 1640.0

/** MiniCPM5-1B's `ffn_gate/up` family: 32 tokens, 1536 in, 4608 out. Chosen because it is
 *  the shape §16 and §18 measured, so the numbers here sit beside those rather than beside
 *  nothing. depth 48 divides Kt exactly, which the reader requires. */
#define TILE_ROWS  1u
#define TILE_INNER 48u
#define TILE_COLS  144u
#define PREFETCH_DEPTH 48u

#define TILE_BYTES FT_TT_OP_TILE_BYTES_BF16
#define A_TILES    (TILE_ROWS * TILE_INNER)
#define B_TILES    (TILE_INNER * TILE_COLS)
#define C_TILES    (TILE_ROWS * TILE_COLS)

/** Where the operands live. Separate banks so a read and a write cannot alias. */
#define DRAM_BANK_A 0u
#define DRAM_BANK_B 1u
#define DRAM_BANK_C 2u
#define DRAM_BUFFER_BASE 0x00100000u

#define CORE_COUNT 2u
#define QUEUE_CAPACITY 64u
#define RUN_TIMEOUT_MS 20000u
#define POISON_BYTE 0x5Au

/** The two bounds.
 *
 * The red-proof bound has to be LONGER than the window the ask is given, not shorter, and
 * getting that backwards made the first red run green for the wrong reason: at 40M spins
 * (~0.4 s here) every processor had already retired ITSELF on the bound before the shutdown
 * was even called, so `unstopped` came back 0 and the case passed against a build whose ask
 * does nothing. A red proof that the bound can satisfy is not a proof of the ask.
 *
 * So: ~400M spins is ~10 s of blocked polling, the ask is given 3 s, and the two outcomes
 * are then distinguishable — "it stopped in milliseconds because it was asked" versus "the
 * shutdown timed out and the BOUND cleaned up afterwards". */
#define IDLE_SPIN_LIMIT     400000000u
#define CB_SPIN_LIMIT       400000000u
#define RED_PROOF_SHUTDOWN_MS 3000u

/** Every processor of the triad is resident. */
#define TRIAD_MASK                                                            \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) | \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

#define CB_IN0 0u
#define CB_IN1 1u
#define CB_OUT 16u

#define CORE_INDICES_ENVIRONMENT "FT_TT_TENSIX_INDICES"
#define AB_ITERATIONS_ENVIRONMENT "FT_TT_TRIAD_AB_ITERATIONS"

static uint32_t g_core_indices[CORE_COUNT] = { 60u, 61u };
static uint32_t g_ab_iterations = 8u;

static ft_tt_launch_core_placement g_placements[CORE_COUNT];
static ft_tt_queue g_queues[CORE_COUNT];
static ft_tt_core_group g_group;
static ft_tt_noc_coord g_dram[8];

static ft_tt_program_directory g_triad;          /* stoppable — what ships */
static ft_tt_program_directory g_triad_upstream; /* upstream's blocking waits — cost arm */
static ft_tt_program_directory g_triad_boundonly;/* the ask removed — the red proof */
static ft_tt_program_directory g_matmul;         /* the Phase 4 slow-dispatch arm */

static ft_tt_launch_config g_triad_config[CORE_COUNT];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
static ft_tt_launch_cb_entry g_cb_placed[3];

static uint16_t *g_a_tiles;
static uint16_t *g_b_tiles;
static uint16_t *g_reference; /* the Phase 4 arm's output, byte-for-byte */
static uint16_t *g_observed;

/* ================================================================
 * Small helpers
 * ================================================================ */

static uint64_t monotonic_microseconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0u;
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)(now.tv_nsec / 1000);
}

static uint32_t g_random_state = 1u;

static void random_seed(uint32_t seed)
{
    g_random_state = seed ? seed : 1u;
}

/** A full-mantissa value. Small integers are the WRONG choice for a layout or plumbing
 *  test: a bf16-exact array is bit-identical to several wrong renderings of itself, which
 *  made two deliberately-different builds byte-exact once (dev notes §3). */
static float random_value(void)
{
    g_random_state = g_random_state * 1664525u + 1013904223u;
    return ((float)(g_random_state >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static uint16_t bf16_from_float(float value)
{
    union {
        float f;
        uint32_t u;
    } bits;

    bits.f = value;
    /* Round-to-nearest-even on the truncated 16 bits, the same rule the converter uses. */
    return (uint16_t)((bits.u + 0x7FFFu + ((bits.u >> 16) & 1u)) >> 16);
}

static int device_write(const ft_tt_noc_access *noc, ft_tt_noc_coord where, uint32_t address,
                        const void *source, size_t bytes)
{
    const unsigned char *cursor = source;
    size_t offset = 0;

    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > 0x10000u)
            chunk = 0x10000u;
        rc = noc->write(noc->user, (uint16_t)where.x, (uint16_t)where.y,
                        (uint64_t)address + offset, cursor + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

static int device_read(const ft_tt_noc_access *noc, ft_tt_noc_coord where, uint32_t address,
                       void *destination, size_t bytes)
{
    unsigned char *cursor = destination;
    size_t offset = 0;

    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > 0x10000u)
            chunk = 0x10000u;
        rc = noc->read(noc->user, (uint16_t)where.x, (uint16_t)where.y,
                       (uint64_t)address + offset, cursor + offset, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

/** Poison the output bank, so an arm that computed nothing is visible as such rather than
 *  as whatever the previous arm left there. */
static int poison_output(const ft_tt_noc_access *noc)
{
    unsigned char pattern[0x1000];
    size_t bytes = (size_t)C_TILES * TILE_BYTES;
    size_t offset = 0;

    memset(pattern, POISON_BYTE, sizeof(pattern));
    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > sizeof(pattern))
            chunk = sizeof(pattern);
        rc = noc->write(noc->user, (uint16_t)g_dram[DRAM_BANK_C].x, (uint16_t)g_dram[DRAM_BANK_C].y,
                        (uint64_t)DRAM_BUFFER_BASE + offset, pattern, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

/** GFLOP/s for this shape at a measured wall time. 2 flops per multiply-accumulate. */
static double gflops_at(uint64_t microseconds)
{
    const double flops = 2.0 * (double)(TILE_ROWS * FT_TT_OP_TILE_DIM) *
                         (double)(TILE_INNER * FT_TT_OP_TILE_DIM) *
                         (double)(TILE_COLS * FT_TT_OP_TILE_DIM);

    if (microseconds == 0u)
        return 0.0;
    return flops / ((double)microseconds * 1e3);
}

/* ================================================================
 * Operands
 * ================================================================ */

/** Upload a fresh pair of operands, seeded so each round is a DIFFERENT matrix. Byte
 *  equality between two arms means nothing if both arms could be reading the same stale
 *  DRAM as the previous round — so the data has to move. */
static int upload_operands(const ft_tt_noc_access *noc, uint32_t seed)
{
    const size_t a_elements = (size_t)TILE_ROWS * FT_TT_OP_TILE_DIM * TILE_INNER * FT_TT_OP_TILE_DIM;
    const size_t b_elements = (size_t)TILE_INNER * FT_TT_OP_TILE_DIM * TILE_COLS * FT_TT_OP_TILE_DIM;
    uint16_t *row_major = malloc((a_elements > b_elements ? b_elements : b_elements) *
                                 sizeof(uint16_t));
    size_t index;
    int rc;

    if (!row_major)
        return FT_TT_ENOMEM;

    random_seed(seed);
    for (index = 0; index < a_elements; index++)
        row_major[index] = bf16_from_float(random_value());
    rc = ft_tt_tilize_matrix(row_major, sizeof(uint16_t), TILE_ROWS * FT_TT_OP_TILE_DIM,
                             TILE_INNER * FT_TT_OP_TILE_DIM, g_a_tiles);
    if (rc != FT_TT_OK)
        goto cleanup;

    for (index = 0; index < b_elements; index++)
        row_major[index] = bf16_from_float(random_value());
    rc = ft_tt_tilize_matrix(row_major, sizeof(uint16_t), TILE_INNER * FT_TT_OP_TILE_DIM,
                             TILE_COLS * FT_TT_OP_TILE_DIM, g_b_tiles);
    if (rc != FT_TT_OK)
        goto cleanup;

    rc = device_write(noc, g_dram[DRAM_BANK_A], DRAM_BUFFER_BASE, g_a_tiles,
                      (size_t)A_TILES * TILE_BYTES);
    if (rc == FT_TT_OK)
        rc = device_write(noc, g_dram[DRAM_BANK_B], DRAM_BUFFER_BASE, g_b_tiles,
                          (size_t)B_TILES * TILE_BYTES);

cleanup:
    free(row_major);
    return rc;
}

/* ================================================================
 * The Phase 4 arm
 * ================================================================ */

static void fill_matmul_request(ft_tt_op_matmul_request *request)
{
    memset(request, 0, sizeof(*request));
    request->tile_rows = TILE_ROWS;
    request->tile_inner = TILE_INNER;
    request->tile_cols = TILE_COLS;
    request->prefetch_tiles = PREFETCH_DEPTH;
    /* K-major, because upload_operands() tilizes B in its natural [K x N] order. The field
     * is a STATEMENT ABOUT DRAM, not a request: both index formulas are in range, so a
     * wrong value here is a wrong answer with no error anywhere. */
    request->weight_n_major = 0u;
    request->activation.x = (uint16_t)g_dram[DRAM_BANK_A].x;
    request->activation.y = (uint16_t)g_dram[DRAM_BANK_A].y;
    request->activation.address = DRAM_BUFFER_BASE;
    request->activation.tile_bytes = TILE_BYTES;
    request->weight.x = (uint16_t)g_dram[DRAM_BANK_B].x;
    request->weight.y = (uint16_t)g_dram[DRAM_BANK_B].y;
    request->weight.address = DRAM_BUFFER_BASE;
    request->weight.tile_bytes = TILE_BYTES;
    request->output.x = (uint16_t)g_dram[DRAM_BANK_C].x;
    request->output.y = (uint16_t)g_dram[DRAM_BANK_C].y;
    request->output.address = DRAM_BUFFER_BASE;
    request->output.tile_bytes = TILE_BYTES;
}

/* ================================================================
 * The resident triad
 * ================================================================ */

/** Place the triad's three circular buffers, exactly as ft_tt_op_matmul.c does: two blocks
 *  of `depth` pages on each input so the reader fills one while the compute thread consumes
 *  the other, and two pages out. */
static int place_circular_buffers(ft_tt_launch_config *config)
{
    ft_tt_op_cb_request requests[3];

    requests[0].index = CB_IN0;
    requests[0].page_bytes = TILE_BYTES;
    requests[0].page_count = PREFETCH_DEPTH * 2u;
    requests[1].index = CB_IN1;
    requests[1].page_bytes = TILE_BYTES;
    requests[1].page_count = PREFETCH_DEPTH * 2u;
    requests[2].index = CB_OUT;
    requests[2].page_bytes = TILE_BYTES;
    requests[2].page_count = 2u;
    return ft_tt_op_cb_place(requests, 3u, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer),
                             &config->local_cb_mask);
}

/**
 * Stage and KICK a resident triad: five images already in L1, five runtime-arg blocks, the
 * CB descriptor array, the launch message, GO — and then the host walks away. Everything
 * after this is `ft_tt_queue_enqueue()`.
 */
static int start_triad(const ft_tt_noc_access *noc, const ft_tt_core_group *group, uint32_t slot,
                       const ft_tt_launch_config *config, uint32_t cb_spin_limit)
{
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0,
                                            FT_TT_PROC_MATH1, FT_TT_PROC_MATH2 };
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&group->cores, slot);
    uint32_t args[5][FT_TT_QUEUE_KERNEL_ARG_COUNT];
    ft_tt_op_args blocks[5];
    unsigned index;
    int rc;

    for (index = 0; index < 5u; index++) {
        ft_tt_queue_kernel_binding binding;

        /* DM1 is the publisher: the writer owns completion, per core and per group. Giving
         * a second processor the group address would give one slot two writers, which is
         * the one property the slot array exists to have. */
        rc = ft_tt_group_kernel_binding(group, slot, processors[index], FT_TT_PROC_DM1,
                                        IDLE_SPIN_LIMIT, cb_spin_limit, &binding);
        if (rc != FT_TT_OK)
            return rc;
        rc = ft_tt_queue_kernel_args(&group->queues[slot], &binding, args[index]);
        if (rc != FT_TT_OK)
            return rc;
        blocks[index].processor = processors[index];
        blocks[index].values = args[index];
        blocks[index].count = FT_TT_QUEUE_KERNEL_ARG_COUNT;
    }

    rc = ft_tt_op_stage(noc, placement->x, placement->y, config, g_cb_buffer, sizeof(g_cb_buffer),
                        blocks, 5u);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_send_go(noc, placement->x, placement->y);
}

/** Load all five images of one program build into a core, replacing whatever is there. */
static int load_triad_images(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                             const ft_tt_program_directory *program)
{
    unsigned index;

    for (index = 0; index < program->program.image_count; index++) {
        const ft_tt_kernel_image *image = &program->program.images[index];
        int rc = ft_tt_launch_load_image(noc, x, y, image->processor, FT_TT_IMAGE_KERNEL,
                                         image->elf_data, image->elf_size, NULL);

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/** Build one matmul descriptor plus the parameter block it points at, at an EXPLICIT
 *  output-tile shape. `enqueue_matmul()` is this at the shipped shape; the shape is a
 *  parameter only so the CB-cycle red proof can name one the shipped path never uses. */
static int enqueue_matmul_shaped(const ft_tt_noc_access *noc, ft_tt_queue *queue, uint32_t slot,
                                 uint32_t starve_blocks, uint32_t group_sequence,
                                 uint32_t tile_rows, uint32_t tile_cols)
{
    uint32_t parameters[FT_TT_QUEUE_MATMUL_WORDS];
    ft_tt_queue_descriptor descriptor;
    uint32_t address = 0u;
    int rc;

    memset(parameters, 0, sizeof(parameters));
    parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] = tile_rows;
    parameters[FT_TT_QUEUE_MATMUL_TILE_INNER] = TILE_INNER;
    parameters[FT_TT_QUEUE_MATMUL_TILE_COLS] = tile_cols;
    parameters[FT_TT_QUEUE_MATMUL_A_X] = g_dram[DRAM_BANK_A].x;
    parameters[FT_TT_QUEUE_MATMUL_A_Y] = g_dram[DRAM_BANK_A].y;
    parameters[FT_TT_QUEUE_MATMUL_A_BASE] = DRAM_BUFFER_BASE;
    parameters[FT_TT_QUEUE_MATMUL_B_X] = g_dram[DRAM_BANK_B].x;
    parameters[FT_TT_QUEUE_MATMUL_B_Y] = g_dram[DRAM_BANK_B].y;
    parameters[FT_TT_QUEUE_MATMUL_B_BASE] = DRAM_BUFFER_BASE;
    parameters[FT_TT_QUEUE_MATMUL_C_X] = g_dram[DRAM_BANK_C].x;
    parameters[FT_TT_QUEUE_MATMUL_C_Y] = g_dram[DRAM_BANK_C].y;
    parameters[FT_TT_QUEUE_MATMUL_C_BASE] = DRAM_BUFFER_BASE;
    parameters[FT_TT_QUEUE_MATMUL_DEPTH] = PREFETCH_DEPTH;
    parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] = 0u;

    rc = ft_tt_queue_write_matmul_parameters(noc, queue, slot, parameters, &address);
    if (rc != FT_TT_OK)
        return rc;

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
    descriptor.a0 = address;
    descriptor.a1 = starve_blocks;
    descriptor.a5 = group_sequence;
    return ft_tt_queue_enqueue(noc, queue, &descriptor, 1u);
}

/** Build one matmul descriptor at the shipped shape. */
static int enqueue_matmul(const ft_tt_noc_access *noc, ft_tt_queue *queue, uint32_t slot,
                          uint32_t starve_blocks, uint32_t group_sequence)
{
    return enqueue_matmul_shaped(noc, queue, slot, starve_blocks, group_sequence, TILE_ROWS,
                                 TILE_COLS);
}

/* ================================================================
 * Setup
 * ================================================================ */

static bool load_environment(void)
{
    const char *indices = getenv(CORE_INDICES_ENVIRONMENT);
    const char *iterations = getenv(AB_ITERATIONS_ENVIRONMENT);

    if (indices && indices[0] != '\0') {
        unsigned count = 0;
        const char *cursor = indices;

        while (*cursor != '\0') {
            char *end = NULL;
            unsigned long value = strtoul(cursor, &end, 10);

            if (end == cursor || count >= CORE_COUNT) {
                ft_tt_test_check_fmt(CORE_INDICES_ENVIRONMENT " parses", false, "at \"%s\"", cursor);
                return false;
            }
            g_core_indices[count++] = (uint32_t)value;
            cursor = (*end == ',') ? end + 1 : end;
        }
        if (count != CORE_COUNT) {
            ft_tt_test_check_fmt(CORE_INDICES_ENVIRONMENT " names every core", false,
                                 "got %u, need %u", count, CORE_COUNT);
            return false;
        }
    }
    if (iterations && iterations[0] != '\0')
        g_ab_iterations = (uint32_t)strtoul(iterations, NULL, 10);
    if (g_ab_iterations == 0u || g_ab_iterations > 64u)
        g_ab_iterations = 8u;
    return true;
}

static bool build_group(const ft_tt_hardware_session *session)
{
    unsigned slot;
    int rc;

    memset(&g_group, 0, sizeof(g_group));
    g_group.id = 0u;
    g_group.first_tensix_index = g_core_indices[0];
    g_group.tensix_index_count = CORE_COUNT;
    g_group.queues = g_queues;
    g_group.resident_processor_mask = TRIAD_MASK;

    rc = ft_tt_launch_core_set_init(&g_group.cores, g_placements, CORE_COUNT);
    if (rc != FT_TT_OK)
        return false;

    for (slot = 0; slot < CORE_COUNT; slot++) {
        const ft_tt_launch_core_placement *placement;

        rc = ft_tt_launch_core_set_add_tensix(&g_group.cores, &session->grid, g_core_indices[slot],
                                              NULL);
        placement = ft_tt_launch_core_set_at(&g_group.cores, slot);
        ft_tt_test_check_fmt("coordinate derives and classifies as a Tensix",
                             rc == FT_TT_OK && placement != NULL, "index %u -> (%u, %u), rc=%d",
                             g_core_indices[slot], placement ? placement->x : 0u,
                             placement ? placement->y : 0u, rc);
        if (rc != FT_TT_OK)
            return false;
    }
    return true;
}

static bool boot_and_place(const ft_tt_hardware_session *session,
                           const ft_tt_firmware_set *firmware)
{
    unsigned slot;

    ft_tt_test_step("boot firmware, place the circular buffers, load the triad");

    for (slot = 0; slot < CORE_COUNT; slot++) {
        const ft_tt_launch_core_placement *placement =
            ft_tt_launch_core_set_at(&g_group.cores, slot);
        int rc = ft_tt_core_boot_firmware(&session->noc, placement->x, placement->y, firmware, NULL);

        ft_tt_test_check_fmt("firmware booted", rc == FT_TT_OK, "(%u, %u) rc=%d", placement->x,
                             placement->y, rc);
        if (rc != FT_TT_OK)
            return false;

        rc = ft_tt_op_program_load(&session->noc, placement->x, placement->y, &g_triad.program,
                                   NULL, &g_triad_config[slot]);
        ft_tt_test_check_fmt("all five triad images load into their slabs", rc == FT_TT_OK,
                             "(%u, %u) rc=%d", placement->x, placement->y, rc);
        if (rc != FT_TT_OK)
            return false;

        rc = place_circular_buffers(&g_triad_config[slot]);
        ft_tt_test_check_fmt("the three circular buffers place in the arena", rc == FT_TT_OK,
                             "rc=%d, depth %u", rc, PREFETCH_DEPTH);
        if (rc != FT_TT_OK)
            return false;
        g_triad_config[slot].local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
        g_triad_config[slot].brisc_noc_id = 0u;
    }
    return true;
}

/* ================================================================
 * Stage 1 — the triad computes a real matmul, byte-identical
 * ================================================================ */

static bool stage_1_real_op(const ft_tt_hardware_session *session)
{
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&g_group.cores, 0u);
    ft_tt_op_matmul_request request;
    unsigned round;
    int rc;

    ft_tt_test_step("stage 1 — the resident triad runs a REAL matmul, byte-identical to the "
                    "Phase 4 arm (Mt=%u Kt=%u Nt=%u, depth %u)",
                    TILE_ROWS, TILE_INNER, TILE_COLS, PREFETCH_DEPTH);

    fill_matmul_request(&request);

    for (round = 0; round < 3u; round++) {
        size_t bytes = (size_t)C_TILES * TILE_BYTES;
        uint32_t completed = 0u;

        /* Fresh operands every round: byte equality between two arms proves nothing if a
         * stale DRAM buffer could satisfy both. */
        rc = upload_operands(&session->noc, 0xA5000000u + round);
        ft_tt_test_check_fmt("operands uploaded", rc == FT_TT_OK, "round %u rc=%d", round, rc);
        if (rc != FT_TT_OK)
            return false;

        /* --- the Phase 4 arm, which test_ops.c has already scored against a CPU fp32
         * reference. Its output is this round's reference. --- */
        rc = poison_output(&session->noc);
        if (rc == FT_TT_OK)
            rc = ft_tt_op_matmul_run(&session->noc, placement->x, placement->y, &g_matmul.program,
                                     &request, RUN_TIMEOUT_MS);
        ft_tt_test_check_fmt("the Phase 4 slow-dispatch arm completes", rc == FT_TT_OK,
                             "round %u rc=%d", round, rc);
        if (rc != FT_TT_OK)
            return false;
        rc = device_read(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, g_reference, bytes);
        if (rc != FT_TT_OK)
            return false;
        {
            size_t poisoned = 0;
            size_t index;

            for (index = 0; index < bytes; index++)
                if (((const unsigned char *)g_reference)[index] == POISON_BYTE)
                    poisoned++;
            ft_tt_test_check_fmt("and it wrote over the poison", poisoned * 100u < bytes,
                                 "round %u: %zu of %zu bytes still poison", round, poisoned, bytes);
        }

        /* --- the resident triad, on the SAME core and the SAME inputs --- */
        rc = load_triad_images(&session->noc, placement->x, placement->y, &g_triad);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                                  QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &g_queues[0]);
        if (rc == FT_TT_OK)
            rc = poison_output(&session->noc);
        if (rc == FT_TT_OK)
            rc = start_triad(&session->noc, &g_group, 0u, &g_triad_config[0], CB_SPIN_LIMIT);
        ft_tt_test_check_fmt("the triad is staged and kicked", rc == FT_TT_OK, "round %u rc=%d",
                             round, rc);
        if (rc != FT_TT_OK)
            return false;

        rc = enqueue_matmul(&session->noc, &g_queues[0], 0u, 0u, 0u);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &g_queues[0], 1u, RUN_TIMEOUT_MS,
                                            &completed);
        ft_tt_test_check_fmt("the triad completed the matmul descriptor", rc == FT_TT_OK,
                             "round %u completed=%u", round, completed);

        {
            uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
            const char *reason = NULL;
            bool evidenced;

            (void)ft_tt_queue_read_control(&session->noc, &g_queues[0], fields);
            evidenced = ft_tt_queue_check_completion(&g_queues[0], fields, 1u, &reason) == FT_TT_OK;
            ft_tt_test_check_fmt("completion is evidenced by the WRITER, not by a run message",
                                 evidenced, "round %u: %s", round, reason ? reason : "ok");
        }

        rc = device_read(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, g_observed, bytes);
        ft_tt_test_check_fmt("the triad's output reads back", rc == FT_TT_OK, "round %u rc=%d",
                             round, rc);

        /* THE CHECK. Not PCC: both arms run the same compute source, so a difference is a
         * plumbing bug and a byte comparison is the instrument for one. */
        {
            size_t index;
            size_t first = bytes;
            size_t mismatches = 0;

            for (index = 0; index < bytes; index++) {
                if (((const unsigned char *)g_observed)[index] !=
                    ((const unsigned char *)g_reference)[index]) {
                    mismatches++;
                    if (first == bytes)
                        first = index;
                }
            }
            ft_tt_test_check_fmt("and it is BYTE-IDENTICAL to the Phase 4 arm", mismatches == 0,
                                 "round %u: %zu of %zu bytes differ, first at %zu", round,
                                 mismatches, bytes, first);
        }

        /* The triad is STILL RESIDENT — that is what resident means. A second descriptor
         * with no launch of any kind is the proof. */
        rc = enqueue_matmul(&session->noc, &g_queues[0], 1u, 0u, 0u);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &g_queues[0], 2u, RUN_TIMEOUT_MS,
                                            &completed);
        ft_tt_test_check_fmt("a SECOND matmul runs with no launch of any kind", rc == FT_TT_OK,
                             "round %u completed=%u", round, completed);

        {
            uint32_t unstopped = 0u;
            ft_tt_core_group solo = g_group;

            solo.cores.count = 1u;
            rc = ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
            ft_tt_test_check_fmt("all five processors retire when asked, with no reset anywhere",
                                 rc == FT_TT_OK && unstopped == 0u, "round %u rc=%d unstopped=%u",
                                 round, rc, unstopped);
        }
    }
    return ft_tt_test_failure_count() == 0;
}

/* ================================================================
 * Stage 2 — the REAL-OP dispatch A/B
 * ================================================================ */

/** The Phase 4 shape without the image reload: CB config + five arg blocks + launch message
 *  + GO + poll DONE. Splitting it out is what separates "the images crossed the x1 link"
 *  from "the launch protocol ran", and those are very different fractions of 0.245 ms. */
static int slow_dispatch_staged(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                                const ft_tt_launch_config *config, const ft_tt_op_args *args,
                                unsigned arg_count)
{
    int rc = ft_tt_op_stage(&session->noc, x, y, config, g_cb_buffer, sizeof(g_cb_buffer), args,
                            arg_count);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go(&session->noc, x, y);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_wait_done(&session->noc, x, y, RUN_TIMEOUT_MS, NULL);
}

static bool stage_2_real_op_dispatch(const ft_tt_hardware_session *session)
{
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&g_group.cores, 0u);
    ft_tt_op_matmul_request request;
    uint64_t full_total = 0u;
    uint64_t staged_total = 0u;
    uint64_t queue_total = 0u;
    uint32_t round;
    int rc = FT_TT_OK;

    ft_tt_test_step("stage 2 — dispatch on a REAL op, %u matmuls per arm, arms interleaved",
                    g_ab_iterations);

    fill_matmul_request(&request);
    rc = upload_operands(&session->noc, 0xBEEF0001u);
    if (rc != FT_TT_OK)
        return false;

    for (round = 0; round < g_ab_iterations; round++) {
        uint64_t started;

        /* ---- arm A1: the whole Phase 4 op call, images and all ---- */
        started = monotonic_microseconds();
        rc = ft_tt_op_matmul_run(&session->noc, placement->x, placement->y, &g_matmul.program,
                                 &request, RUN_TIMEOUT_MS);
        full_total += monotonic_microseconds() - started;
        ft_tt_test_check_fmt("arm A1 completed its round", rc == FT_TT_OK, "round %u rc=%d", round,
                             rc);
        if (rc != FT_TT_OK)
            return false;

        /* ---- arm A2: the same launch protocol with the images already in L1 ---- */
        {
            uint32_t reader_args[11];
            uint32_t writer_args[4];
            uint32_t compute_args[3];
            ft_tt_op_args args[5];
            ft_tt_launch_config config;

            rc = ft_tt_op_program_load(&session->noc, placement->x, placement->y,
                                       &g_matmul.program, NULL, &config);
            if (rc == FT_TT_OK)
                rc = place_circular_buffers(&config);
            if (rc != FT_TT_OK)
                return false;
            config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
            config.brisc_noc_id = 0u;

            reader_args[0] = TILE_ROWS;
            reader_args[1] = TILE_INNER;
            reader_args[2] = TILE_COLS;
            reader_args[3] = g_dram[DRAM_BANK_A].x;
            reader_args[4] = g_dram[DRAM_BANK_A].y;
            reader_args[5] = DRAM_BUFFER_BASE;
            reader_args[6] = g_dram[DRAM_BANK_B].x;
            reader_args[7] = g_dram[DRAM_BANK_B].y;
            reader_args[8] = DRAM_BUFFER_BASE;
            reader_args[9] = PREFETCH_DEPTH;
            reader_args[10] = 0u;
            writer_args[0] = TILE_ROWS * TILE_COLS;
            writer_args[1] = g_dram[DRAM_BANK_C].x;
            writer_args[2] = g_dram[DRAM_BANK_C].y;
            writer_args[3] = DRAM_BUFFER_BASE;
            compute_args[0] = TILE_ROWS;
            compute_args[1] = TILE_INNER;
            compute_args[2] = TILE_COLS;
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

            started = monotonic_microseconds();
            rc = slow_dispatch_staged(session, placement->x, placement->y, &config, args, 5u);
            staged_total += monotonic_microseconds() - started;
            ft_tt_test_check_fmt("arm A2 completed its round", rc == FT_TT_OK, "round %u rc=%d",
                                 round, rc);
            if (rc != FT_TT_OK)
                return false;
        }

        /* ---- arm B: the resident triad. One parameter block, one descriptor, one head
         * write, one completion poll. No images, no CB config, no launch message. ---- */
        rc = load_triad_images(&session->noc, placement->x, placement->y, &g_triad);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                                  QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &g_queues[0]);
        if (rc == FT_TT_OK)
            rc = start_triad(&session->noc, &g_group, 0u, &g_triad_config[0], CB_SPIN_LIMIT);
        if (rc != FT_TT_OK)
            return false;

        started = monotonic_microseconds();
        rc = enqueue_matmul(&session->noc, &g_queues[0], 0u, 0u, 0u);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &g_queues[0], 1u, RUN_TIMEOUT_MS, NULL);
        queue_total += monotonic_microseconds() - started;
        ft_tt_test_check_fmt("arm B completed its round", rc == FT_TT_OK, "round %u rc=%d", round,
                             rc);
        if (rc != FT_TT_OK)
            return false;

        {
            uint32_t unstopped = 0u;
            ft_tt_core_group solo = g_group;

            solo.cores.count = 1u;
            (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
        }
    }

    {
        double rounds = (double)g_ab_iterations;
        uint64_t full_us = (uint64_t)((double)full_total / rounds);
        uint64_t staged_us = (uint64_t)((double)staged_total / rounds);
        uint64_t queue_us = (uint64_t)((double)queue_total / rounds);

        printf("\n  ONE REAL MATMUL, Mt=%u Kt=%u Nt=%u bf16, depth %u, %u per arm, interleaved\n",
               TILE_ROWS, TILE_INNER, TILE_COLS, PREFETCH_DEPTH, g_ab_iterations);
        printf("    arm                                       wall (ms)   GFLOP/s   %% of one Tensix\n");
        printf("    A1  Phase 4 op call, images reloaded  : %9.3f  %8.1f   %6.2f%%\n",
               (double)full_us / 1000.0, gflops_at(full_us),
               100.0 * gflops_at(full_us) / TENSIX_PEAK_GFLOPS);
        printf("    A2  same, images already in L1        : %9.3f  %8.1f   %6.2f%%\n",
               (double)staged_us / 1000.0, gflops_at(staged_us),
               100.0 * gflops_at(staged_us) / TENSIX_PEAK_GFLOPS);
        printf("    B   resident triad, one enqueue       : %9.3f  %8.1f   %6.2f%%\n",
               (double)queue_us / 1000.0, gflops_at(queue_us),
               100.0 * gflops_at(queue_us) / TENSIX_PEAK_GFLOPS);
        printf("    fixed cost removed: A1 - B = %.3f ms,  A2 - B = %.3f ms\n",
               (double)(full_us - queue_us) / 1000.0, (double)(staged_us - queue_us) / 1000.0);
        printf("    (peak is one Tensix at 800 MHz HiFi4 = %.0f GFLOP/s; every figure here is "
               "PER CORE)\n", TENSIX_PEAK_GFLOPS);

        ft_tt_test_check_fmt("the resident triad beats the full Phase 4 op call",
                             queue_us < full_us, "%.3f ms vs %.3f ms", (double)queue_us / 1000.0,
                             (double)full_us / 1000.0);
        ft_tt_test_check_fmt("and beats it with the images already loaded too",
                             queue_us < staged_us, "%.3f ms vs %.3f ms", (double)queue_us / 1000.0,
                             (double)staged_us / 1000.0);
    }
    return ft_tt_test_failure_count() == 0;
}

/* ================================================================
 * Stage 3 — what the stoppable wait costs
 * ================================================================ */

/** Run `count` matmuls on a resident triad built from `program`, and return the total. */
static uint64_t time_resident_matmuls(const ft_tt_hardware_session *session,
                                      const ft_tt_program_directory *program, uint32_t count,
                                      int *out_rc)
{
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&g_group.cores, 0u);
    uint64_t started;
    uint64_t elapsed = 0u;
    uint32_t index;
    int rc;

    rc = load_triad_images(&session->noc, placement->x, placement->y, program);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                              QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &g_queues[0]);
    if (rc == FT_TT_OK)
        rc = start_triad(&session->noc, &g_group, 0u, &g_triad_config[0], CB_SPIN_LIMIT);
    if (rc != FT_TT_OK) {
        *out_rc = rc;
        return 0u;
    }

    started = monotonic_microseconds();
    for (index = 0; index < count; index++) {
        rc = enqueue_matmul(&session->noc, &g_queues[0], index % 8u, 0u, 0u);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &g_queues[0], index + 1u, RUN_TIMEOUT_MS,
                                            NULL);
        if (rc != FT_TT_OK)
            break;
    }
    elapsed = monotonic_microseconds() - started;

    {
        uint32_t unstopped = 0u;
        ft_tt_core_group solo = g_group;

        solo.cores.count = 1u;
        (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
    }
    *out_rc = rc;
    return elapsed;
}

static bool stage_3_stoppable_cost(const ft_tt_hardware_session *session)
{
    uint64_t upstream_total = 0u;
    uint64_t stoppable_total = 0u;
    uint32_t round;
    int rc = FT_TT_OK;

    ft_tt_test_step("stage 3 — what the stoppable CB wait COSTS: upstream's blocking waits "
                    "versus ours, same source, same shape, interleaved");

    rc = upload_operands(&session->noc, 0xC0DE0002u);
    if (rc != FT_TT_OK)
        return false;

    /* Interleaved, because two 1.8 GHz cores also driving eight K80s drift enough over ten
     * seconds to swamp the difference being measured (dev notes §8). */
    for (round = 0; round < g_ab_iterations; round++) {
        int arm_rc = FT_TT_OK;

        upstream_total += time_resident_matmuls(session, &g_triad_upstream, 1u, &arm_rc);
        ft_tt_test_check_fmt("the upstream-wait arm ran", arm_rc == FT_TT_OK, "round %u rc=%d",
                             round, arm_rc);
        if (arm_rc != FT_TT_OK)
            return false;

        stoppable_total += time_resident_matmuls(session, &g_triad, 1u, &arm_rc);
        ft_tt_test_check_fmt("the stoppable-wait arm ran", arm_rc == FT_TT_OK, "round %u rc=%d",
                             round, arm_rc);
        if (arm_rc != FT_TT_OK)
            return false;
    }

    {
        double rounds = (double)g_ab_iterations;
        uint64_t upstream_us = (uint64_t)((double)upstream_total / rounds);
        uint64_t stoppable_us = (uint64_t)((double)stoppable_total / rounds);
        double cost = upstream_us ? 100.0 * ((double)stoppable_us - (double)upstream_us) /
                                        (double)upstream_us
                                  : 0.0;

        printf("\n  THE COST OF A STOPPABLE WAIT, one real matmul, %u per arm, interleaved\n",
               g_ab_iterations);
        printf("    upstream cb_wait_front / cb_reserve_back : %9.3f ms  %8.1f GFLOP/s\n",
               (double)upstream_us / 1000.0, gflops_at(upstream_us));
        printf("    stoppable (ask + bound)                  : %9.3f ms  %8.1f GFLOP/s\n",
               (double)stoppable_us / 1000.0, gflops_at(stoppable_us));
        printf("    cost of the escape: %+.2f%%\n", cost);
        printf("    (the escape sits INSIDE the loop body, which only runs while BLOCKED, so a\n"
               "     wait that is already satisfied pays one predictable branch and nothing else)\n");

        /* A band, not a point: this host cannot resolve better than a few percent, and a
         * tighter assertion would be measuring its mood. What would fail is a REGRESSION —
         * an escape that polled somewhere it should not. */
        ft_tt_test_check_fmt("the stoppable wait costs less than 5%", cost < 5.0, "%+.2f%%", cost);
    }
    return ft_tt_test_failure_count() == 0;
}

/* ================================================================
 * Stage 4 — the acceptance: a deliberate CB deadlock
 * ================================================================ */

/**
 * The deadlock is a REAL matmul with its last block withheld.
 *
 * That matters. A synthetic wait built to be easy to break proves nothing about
 * `cb_wait_front`; here the compute thread is blocked in the op's own
 * `cb_wait_front(cb_in0, 1)` inside the real K loop, the writer is blocked in its own
 * `cb_wait_front(cb_out, 1)`, and the reader is idle at the queue poll — which is exactly
 * the state a starved pipeline reaches in production.
 */
static bool stage_4_deadlock_and_recovery(const ft_tt_hardware_session *session,
                                          const ft_tt_program_directory *program,
                                          uint32_t shutdown_ms, bool expect_stop_honoured)
{
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&g_group.cores, 0u);
    ft_tt_core_group solo = g_group;
    ft_tt_watchdog_config config;
    ft_tt_watchdog_report report;
    uint32_t unstopped = 0u;
    uint64_t asked_at;
    uint64_t stopped_at;
    int rc;

    solo.cores.count = 1u;

    rc = upload_operands(&session->noc, 0xDEAD0003u);
    if (rc == FT_TT_OK)
        rc = load_triad_images(&session->noc, placement->x, placement->y, program);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                              QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &g_queues[0]);
    if (rc == FT_TT_OK)
        rc = start_triad(&session->noc, &g_group, 0u, &g_triad_config[0], CB_SPIN_LIMIT);
    ft_tt_test_check_fmt("the triad is resident", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    /* One block withheld: every tile pair but the last arrives, and the compute thread waits
     * for a tile that will never be pushed. */
    rc = enqueue_matmul(&session->noc, &g_queues[0], 0u, 1u, 0u);
    if (rc != FT_TT_OK)
        return false;

    config.stall_timeout_ms = 1000u;
    config.overall_timeout_ms = 5000u;
    rc = ft_tt_watchdog_wait(&session->noc, &solo, 1u, &config, &report);
    ft_tt_test_check_fmt("the watchdog returns instead of hanging the session", rc == FT_TT_EIO,
                         "rc=%d after %u ms", rc, report.elapsed_ms);
    ft_tt_test_check_fmt("it says the stalled core is ALIVE, not dead", report.stalled_core_is_alive,
                         "state=%s", ft_tt_queue_state_string(report.stalled_state));
    /* THE DIAGNOSIS. Five processors share this queue; "the core is blocked" is not an
     * answer. Both halves are asserted separately so a green run cannot be carried by one. */
    ft_tt_test_check_fmt("and it NAMES THE PROCESSOR that is blocked",
                         report.stalled_processor < FT_TT_PROC_COUNT, "processor=%u",
                         report.stalled_processor);
    /* Asserted only in the arm where the ask works. In the red-proof arm the bound is
     * running out underneath the dump, so which wait a processor is in at the moment the
     * watchdog looks is a race — asserting it there would make the RED case fail for a
     * reason that has nothing to do with what it is testing. */
    if (expect_stop_honoured)
        ft_tt_test_check_fmt("and NAMES THE WAIT it is blocked in",
                             report.stalled_processor_state ==
                                     (uint32_t)FT_TT_QUEUE_STATE_CB_WAIT_FRONT ||
                                 report.stalled_processor_state ==
                                     (uint32_t)FT_TT_QUEUE_STATE_CB_RESERVE_BACK,
                             "%s in %s", ft_tt_processor_name(report.stalled_processor),
                             ft_tt_queue_state_string(report.stalled_processor_state));

    /* Recovery is a WRITE. No reset is asserted anywhere on this path. */
    asked_at = monotonic_microseconds();
    rc = ft_tt_watchdog_shutdown(&session->noc, &solo, shutdown_ms, &unstopped);
    stopped_at = monotonic_microseconds();

    if (expect_stop_honoured) {
        ft_tt_test_check_fmt("the deadlocked TRIAD is retired by ASKING, not by reset",
                             rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=%u", rc, unstopped);
        /* It has to be the ASK that retired it, not the bound quietly expiring: the bound is
         * ~0.5 s and the ask should land in milliseconds. Without this the stage would pass
         * against a kernel whose stop check does nothing, which is the whole point of it. */
        ft_tt_test_check_fmt("and it stopped because it was ASKED, not because a bound expired",
                             stopped_at - asked_at < 200000u, "%llu us after the ask",
                             (unsigned long long)(stopped_at - asked_at));
        {
            uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];

            (void)ft_tt_queue_read_control(&session->noc, &g_queues[0], fields);
            ft_tt_test_check_fmt("the abandoned descriptor was NOT retired",
                                 fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] == 0u, "completed=%u",
                                 fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT]);
        }
    } else {
        /* THE RED PROOF. The same deadlock against a build whose stop check is removed: the
         * ask must NOT retire it. It is safe to run only because the bound is independent of
         * the ask, so the worst case is a timeout and not a wedged core. */
        ft_tt_test_check_fmt("WITH THE ASK REMOVED, the shutdown REFUSES instead of passing",
                             rc == FT_TT_EIO && unstopped == 1u, "rc=%d unstopped=%u after %llu us",
                             rc, unstopped, (unsigned long long)(stopped_at - asked_at));
        /* And it recovers anyway — on the bound. That is rule 4 earning its place: a build
         * with a broken ask still leaves the core usable. */
        {
            uint32_t pending = 0u;
            uint64_t deadline = monotonic_microseconds() + 30000000u;

            while (monotonic_microseconds() < deadline) {
                if (ft_tt_queue_all_processors_exited(&session->noc, &g_queues[0], TRIAD_MASK,
                                                      &pending) == FT_TT_OK)
                    break;
            }
            ft_tt_test_check_fmt("but the BOUND still retires it, so even a broken ask is not a "
                                 "wedged core",
                                 pending == 0u, "pending mask 0x%02x", pending);
        }
        {
            uint32_t status[FT_TT_QUEUE_STATUS_WORDS];

            (void)ft_tt_queue_read_status(&session->noc, &g_queues[0], FT_TT_PROC_MATH0, status);
            ft_tt_test_check_fmt("and it says WHY: the CB wait exhausted its bound",
                                 status[FT_TT_QUEUE_STATUS_ERROR] ==
                                     (uint32_t)FT_TT_QUEUE_KERR_CB_WAIT_TIMEOUT,
                                 "%s",
                                 ft_tt_queue_kernel_error_string(status[FT_TT_QUEUE_STATUS_ERROR]));
        }
    }

    /* THE ACCEPTANCE: the SAME core runs a fresh triad afterwards and gets the right answer.
     * If this needed a power cycle, none of the above is worth anything. */
    {
        size_t bytes = (size_t)C_TILES * TILE_BYTES;
        uint32_t completed = 0u;
        ft_tt_op_matmul_request request;

        fill_matmul_request(&request);
        rc = poison_output(&session->noc);
        if (rc == FT_TT_OK)
            rc = ft_tt_op_matmul_run(&session->noc, placement->x, placement->y, &g_matmul.program,
                                     &request, RUN_TIMEOUT_MS);
        if (rc == FT_TT_OK)
            rc = device_read(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, g_reference,
                             bytes);
        ft_tt_test_check_fmt("the reference arm still runs on this core", rc == FT_TT_OK, "rc=%d",
                             rc);

        rc = load_triad_images(&session->noc, placement->x, placement->y, &g_triad);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                                  QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &g_queues[0]);
        if (rc == FT_TT_OK)
            rc = poison_output(&session->noc);
        if (rc == FT_TT_OK)
            rc = start_triad(&session->noc, &g_group, 0u, &g_triad_config[0], CB_SPIN_LIMIT);
        if (rc == FT_TT_OK)
            rc = enqueue_matmul(&session->noc, &g_queues[0], 0u, 0u, 0u);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_wait_completed(&session->noc, &g_queues[0], 1u, RUN_TIMEOUT_MS,
                                            &completed);
        if (rc == FT_TT_OK)
            rc = device_read(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, g_observed,
                             bytes);
        ft_tt_test_check_fmt("THE CARD SURVIVED: the SAME core runs a fresh resident triad",
                             rc == FT_TT_OK, "rc=%d completed=%u", rc, completed);
        ft_tt_test_check_fmt("and its answer is byte-identical to the reference",
                             rc == FT_TT_OK && memcmp(g_observed, g_reference, bytes) == 0,
                             "%zu bytes", bytes);

        {
            uint32_t stopped = 0u;

            (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &stopped);
        }
    }
    return ft_tt_test_failure_count() == 0;
}

/* ================================================================
 * Stage 6 — group-level completion
 * ================================================================ */

/**
 * A DESCRIPTOR THIS PROGRAM CANNOT EXECUTE IS REFUSED, ON THE CARD, AND FOR THE STATED
 * REASON.
 *
 * `tile_inner = 0` is the shape every one of the reader's derived lengths collapses on:
 * `depth * IN0_TILE_BYTES` is fine but the block loop never runs, `blocks_per_tile` is a
 * divide by the collapsed depth, and a descriptor that reads nothing still hands the writer
 * a completion. A zero-shaped descriptor used to walk straight through.
 *
 * TWO THINGS ARE ASSERTED AND THEY FAIL DIFFERENTLY: that it is REFUSED, and that it is
 * refused FOR THE STATED REASON (`BAD_PARAMETERS`) rather than by whichever ground is
 * cheapest — a check with several sufficient grounds tests the cheapest one, and the one
 * you care about may never run.
 *
 * **This is NOT the red proof for the CB block-grid guard.** That guard cannot be exercised
 * on this op: its circular buffers hold `2 * depth` pages, every legal depth divides both
 * Kt = 48 and 96, and the write pointer is therefore always on the grid of whatever depth
 * the next descriptor asks for. The op that CAN express it is the memory family, where one
 * residency serves a 4-tile KV group and a 48-tile embedding group through one CB_OUT — and
 * that is where the guard was proved, on silicon, refusing a real off-grid descriptor with
 * `BAD_PARAMETERS` while the pre-round control passed the same stage beside it.
 *
 * The shipped shape is then run on a FRESH residency of the same core, so "the guard
 * refuses" and "the guard refuses everything" are distinguishable.
 */
static bool stage_7_zero_shape_refusal(const ft_tt_hardware_session *session)
{
    const ft_tt_launch_core_placement *placement = ft_tt_launch_core_set_at(&g_group.cores, 0u);
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t completed = 0u;
    uint32_t unstopped = 0u;
    ft_tt_core_group solo = g_group;
    int rc;

    ft_tt_test_step("stage 7 — a zero-shaped descriptor is REFUSED on the card, with a reason");

    solo.cores.count = 1u;
    solo.queues = &g_queues[0];

    rc = upload_operands(&session->noc, 0xC0FFEE01u);
    if (rc == FT_TT_OK)
        rc = load_triad_images(&session->noc, placement->x, placement->y, &g_triad);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                              QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &g_queues[0]);
    if (rc == FT_TT_OK)
        rc = poison_output(&session->noc);
    if (rc == FT_TT_OK)
        rc = start_triad(&session->noc, &g_group, 0u, &g_triad_config[0], CB_SPIN_LIMIT);
    ft_tt_test_check_fmt("a fresh residency for the bad descriptor", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    rc = enqueue_matmul_shaped(&session->noc, &g_queues[0], 0u, 0u, 0u, 0u, TILE_COLS);
    ft_tt_test_check_fmt("the zero-shaped descriptor enqueues — the HOST does not catch it",
                         rc == FT_TT_OK, "rc=%d", rc);

    rc = ft_tt_queue_wait_completed(&session->noc, &g_queues[0], 1u, RED_PROOF_SHUTDOWN_MS,
                                    &completed);
    ft_tt_test_check_fmt("and the kernel REFUSES it rather than retiring empty work",
                         rc != FT_TT_OK && completed == 0u, "rc=%d completed=%u", rc, completed);

    memset(fields, 0, sizeof(fields));
    (void)ft_tt_queue_read_control(&session->noc, &g_queues[0], fields);
    ft_tt_test_check_fmt("...for the STATED reason, not whichever ground was cheapest",
                         fields[FT_TT_QUEUE_FIELD_ERROR] ==
                             (uint32_t)FT_TT_QUEUE_KERR_BAD_PARAMETERS,
                         "error=%u (%s)", fields[FT_TT_QUEUE_FIELD_ERROR],
                         ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
    ft_tt_test_check_fmt("...and it left the loop rather than blocking in it",
                         fields[FT_TT_QUEUE_FIELD_STATE] == (uint32_t)FT_TT_QUEUE_STATE_EXIT,
                         "state=0x%08x", fields[FT_TT_QUEUE_FIELD_STATE]);

    (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);

    /* AND THE GOOD PATH STILL PASSES. A guard that refuses everything is not a guard. */
    rc = load_triad_images(&session->noc, placement->x, placement->y, &g_triad);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                              QUEUE_CAPACITY, ft_tt_queue_make_nonce(), &g_queues[0]);
    if (rc == FT_TT_OK)
        rc = poison_output(&session->noc);
    if (rc == FT_TT_OK)
        rc = start_triad(&session->noc, &g_group, 0u, &g_triad_config[0], CB_SPIN_LIMIT);
    if (rc == FT_TT_OK)
        rc = enqueue_matmul(&session->noc, &g_queues[0], 0u, 0u, 0u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queues[0], 1u, RUN_TIMEOUT_MS,
                                        &completed);
    ft_tt_test_check_fmt("the SHIPPED shape still completes on the same core", rc == FT_TT_OK,
                         "rc=%d completed=%u", rc, completed);

    memset(fields, 0, sizeof(fields));
    (void)ft_tt_queue_read_control(&session->noc, &g_queues[0], fields);
    ft_tt_test_check_fmt("...with no error recorded", fields[FT_TT_QUEUE_FIELD_ERROR] == 0u,
                         "error=%u", fields[FT_TT_QUEUE_FIELD_ERROR]);

    unstopped = 0u;
    (void)ft_tt_watchdog_shutdown(&session->noc, &solo, RUN_TIMEOUT_MS, &unstopped);
    return true;
}

static bool stage_6_group_completion(const ft_tt_hardware_session *session)
{
    ft_tt_group_member_completion members[CORE_COUNT];
    const char *reason = NULL;
    uint32_t short_slot = 0u;
    uint32_t reached = 0u;
    uint32_t group_nonce;
    unsigned slot;
    int rc;

    ft_tt_test_step("stage 6 — group-level completion: a sequence number the KERNELS write");

    rc = upload_operands(&session->noc, 0x6E000004u);
    if (rc != FT_TT_OK)
        return false;

    rc = ft_tt_group_completion_init(&session->noc, &g_group,
                                     FT_TT_OP_QUEUE_BASE + FT_TT_QUEUE_GROUP_OFFSET);
    ft_tt_test_check_fmt("the group completion block is written on the leader", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    /* ONE NONCE FOR THE WHOLE GROUP, which is what ft_tt_group_queues_init() does and why it
     * takes a single one. A group-level token check asks "did every member finish item N",
     * and that only has an answer if item N has ONE token across the group — per-member
     * nonces give per-member tokens and the check can never pass. Getting this wrong is a
     * green-looking failure: every member posts, every sequence is right, and the check
     * refuses on the token. */
    group_nonce = ft_tt_queue_make_nonce();
    for (slot = 0; slot < CORE_COUNT; slot++) {
        const ft_tt_launch_core_placement *placement =
            ft_tt_launch_core_set_at(&g_group.cores, slot);

        rc = load_triad_images(&session->noc, placement->x, placement->y, &g_triad);
        if (rc == FT_TT_OK)
            rc = ft_tt_queue_init(&session->noc, placement->x, placement->y, FT_TT_OP_QUEUE_BASE,
                                  QUEUE_CAPACITY, group_nonce, &g_queues[slot]);
        if (rc == FT_TT_OK)
            rc = start_triad(&session->noc, &g_group, slot, &g_triad_config[slot], CB_SPIN_LIMIT);
        ft_tt_test_check_fmt("member is resident", rc == FT_TT_OK, "slot %u rc=%d", slot, rc);
        if (rc != FT_TT_OK)
            return false;
    }

    /* Feed ONE member only. The whole point is that a group-level check must refuse this —
     * and a SUM of completed_count across the group would not, because one member doing the
     * work of two sums to the same number. */
    rc = enqueue_matmul(&session->noc, &g_queues[0], 0u, 0u, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queues[0], 1u, RUN_TIMEOUT_MS, NULL);
    ft_tt_test_check_fmt("member 0 finished item 1", rc == FT_TT_OK, "rc=%d", rc);

    rc = ft_tt_group_completion_read(&session->noc, &g_group, members, CORE_COUNT);
    ft_tt_test_check_fmt("the group block reads back", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("member 0's slot carries sequence 1", members[0].sequence == 1u,
                         "sequence=%u token=0x%08x, want sequence 1 token 0x%08x",
                         members[0].sequence, members[0].token,
                         ft_tt_queue_token_at(&g_queues[0], 0u));
    ft_tt_test_check_fmt("and the token beside it is the one the host stamped, so the pair "
                         "landed together",
                         members[0].token == ft_tt_queue_token_at(&g_queues[0], 0u), "0x%08x",
                         members[0].token);
    ft_tt_test_check_fmt("and member 1's slot is untouched", members[1].sequence == 0u,
                         "sequence=%u", members[1].sequence);

    rc = ft_tt_group_check_completion(&g_group, members, CORE_COUNT, 1u,
                                      ft_tt_queue_token_at(&g_queues[0], 0u), &reason, &short_slot);
    ft_tt_test_check_fmt("THE GROUP CHECK REFUSES with one member behind", rc == FT_TT_ERROR, "%s",
                         reason ? reason : "(it passed, which is the bug)");
    ft_tt_test_check_fmt("and it refuses BECAUSE a member has not reached the sequence",
                         reason != NULL && strstr(reason, "has not reached") != NULL, "%s",
                         reason ? reason : "(no reason)");
    ft_tt_test_check_fmt("and it names WHICH member", short_slot == 1u, "slot %u", short_slot);

    /* Now feed the other member, and it must pass. */
    rc = enqueue_matmul(&session->noc, &g_queues[1], 0u, 0u, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_group_wait_completion(&session->noc, &g_group, 1u, RUN_TIMEOUT_MS, &reached);
    ft_tt_test_check_fmt("with both members fed, the group reaches sequence 1", rc == FT_TT_OK,
                         "reached=%u rc=%d", reached, rc);

    rc = ft_tt_group_completion_read(&session->noc, &g_group, members, CORE_COUNT);
    if (rc == FT_TT_OK)
        rc = ft_tt_group_check_completion(&g_group, members, CORE_COUNT, 1u,
                                          ft_tt_queue_token_at(&g_queues[0], 0u), &reason,
                                          &short_slot);
    ft_tt_test_check_fmt("and the group check now passes", rc == FT_TT_OK, "%s",
                         reason ? reason : "ok");

    /* A WRONG TOKEN must refuse, or the sequence is doing all the work and a previous run's
     * residue would satisfy the check. */
    rc = ft_tt_group_check_completion(&g_group, members, CORE_COUNT, 1u, 0x1234ABCDu, &reason,
                                      &short_slot);
    ft_tt_test_check_fmt("a group check with the WRONG token refuses", rc == FT_TT_ERROR, "%s",
                         reason ? reason : "(it passed, which is the bug)");
    ft_tt_test_check_fmt("and it refuses BECAUSE of the token, not the sequence",
                         reason != NULL && strstr(reason, "token") != NULL, "%s",
                         reason ? reason : "(no reason)");

    {
        uint32_t unstopped = 0u;

        rc = ft_tt_watchdog_shutdown(&session->noc, &g_group, RUN_TIMEOUT_MS, &unstopped);
        ft_tt_test_check_fmt("the whole group retires by asking", rc == FT_TT_OK && unstopped == 0u,
                             "rc=%d unstopped=%u", rc, unstopped);
    }
    return ft_tt_test_failure_count() == 0;
}

/* ================================================================ */

static bool allocate_host_buffers(void)
{
    g_a_tiles = malloc((size_t)A_TILES * TILE_BYTES);
    g_b_tiles = malloc((size_t)B_TILES * TILE_BYTES);
    g_reference = malloc((size_t)C_TILES * TILE_BYTES);
    g_observed = malloc((size_t)C_TILES * TILE_BYTES);
    return g_a_tiles && g_b_tiles && g_reference && g_observed;
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint32_t bank;
    int rc;

    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <triad-dir> <triad-upstream-dir> <triad-boundonly-dir> <matmul-dir>\n",
                argv[0]);
        return 2;
    }
    if (!load_environment())
        return ft_tt_test_finish();
    if (!allocate_host_buffers()) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        return ft_tt_test_finish();
    }

    printf("Phase 6 resident triad — cores %u, %u\n", g_core_indices[0], g_core_indices[1]);

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;

    /* Per-core, not card-wide: this pair can run while another agent works a
     * different pair. A refusal names the holder's pid instead of blocking. */
    rc = ft_tt_hardware_session_claim_cores(&session, g_core_indices, CORE_COUNT,
                                            "phase6 resident triad", NULL, NULL);
    ft_tt_test_check_fmt("both cores are claimed for this run", rc == FT_TT_OK, "%s",
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return ft_tt_test_finish();
    }

    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    for (bank = 0; bank < 3u; bank++) {
        rc = ft_tt_coord_dram_bank(&session.grid, bank, FT_TT_COORD_TRANSLATED, &g_dram[bank]);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("DRAM bank coordinates derive", false, "bank %u rc=%d", bank, rc);
            goto shutdown;
        }
    }

    rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &g_triad);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[2], TRIAD_MASK, &g_triad_upstream);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[3], TRIAD_MASK, &g_triad_boundonly);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[4], TRIAD_MASK, &g_matmul);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("all four program directories load", false, "rc=%d", rc);
        goto release;
    }

    if (build_group(&session) && boot_and_place(&session, &firmware)) {
        if (stage_1_real_op(&session) && stage_2_real_op_dispatch(&session) &&
            stage_3_stoppable_cost(&session)) {
            ft_tt_test_step("stage 4 — the acceptance: a REAL matmul deadlocked in cb_wait_front");
            if (stage_4_deadlock_and_recovery(&session, &g_triad, RUN_TIMEOUT_MS, true)) {
                ft_tt_test_step("stage 5 — THE RED PROOF: the same deadlock with the ask removed");
                if (stage_4_deadlock_and_recovery(&session, &g_triad_boundonly,
                                                  RED_PROOF_SHUTDOWN_MS, false)) {
                    (void)stage_6_group_completion(&session);
                    (void)stage_7_zero_shape_refusal(&session);
                }
            }
        }
    }

    /* DO NOT PARK. AT ALL.
     *
     * Two versions of this ran and both cost a core. The first parked on every exit path,
     * including the failure one, so a core whose BRISC was still executing had its TRISCs
     * reset underneath it and BRISC hung in the firmware's `tensix_sync()` forever
     * (`soft reset 0x00047000: BRISC=running NCRISC=held TRISC0..2=held`, every later boot
     * timing out). The second parked only after confirming all five processors had reported
     * EXIT — and a core whose kernel exited that way ALSO failed its next
     * `ft_tt_core_boot_firmware` until the host was rebooted.
     *
     * So the measured rule is stronger than "park carefully": on a five-processor resident
     * program, PARKING IS THE HAZARD. There is no need for it either. `ft_tt_watchdog_shutdown()`
     * already leaves the core with its firmware up and ready for the next launch, and its own
     * header says it deliberately does not park because "a parked core needs its firmware
     * re-booted before it can run anything". Retiring the kernel and stopping using the core
     * are different decisions; this test only needs the first.
     *
     * What is left behind is a core sitting in its firmware's GO wait, which is exactly where
     * a healthy idle core sits. */
    {
        uint32_t slot;

        for (slot = 0u; slot < ft_tt_launch_core_set_count(&g_group.cores); slot++) {
            ft_tt_core_group solo = g_group;
            uint32_t control_state = 0u;
            uint32_t unstopped = 0u;

            solo.cores.count = 1u;
            solo.queues = &g_queues[slot];
            (void)ft_tt_queue_read_field(&session.noc, &g_queues[slot], FT_TT_QUEUE_FIELD_STATE,
                                         &control_state);
            if (control_state == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
                continue; /* nothing was ever launched here */
            (void)ft_tt_watchdog_shutdown(&session.noc, &solo, RUN_TIMEOUT_MS, &unstopped);
        }
    }

release:
    ft_tt_program_directory_release(&g_matmul);
    ft_tt_program_directory_release(&g_triad_boundonly);
    ft_tt_program_directory_release(&g_triad_upstream);
    ft_tt_program_directory_release(&g_triad);
    ft_tt_firmware_set_release(&firmware);
shutdown:
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
