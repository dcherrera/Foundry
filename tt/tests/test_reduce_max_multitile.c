/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_reduce_max_multitile.c
 * @brief ON-CARD probe answering ONE question: may a row-wise MAX reduce WIDER THAN ONE
 *   TILE be expressed as repeated `reduce_tile<PoolType::MAX>` calls accumulating into the
 *   same DST? `kernels/ft_tensix_compute_reduce.h` (ft_tensix_math_reduce_row_max) records
 *   the doubt; `kernels/resident_reduce_compute.c` already ships the loop.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS NOT A SOFTMAX TEST
 * ---------------------------------------------------------------------------
 * softmax(x - c) = softmax(x) for ANY c, so a wrong row maximum is structurally invisible
 * in softmax's OUTPUT unless it drives exp out of range. test_phase6_vector.c's outlier
 * check plants its +200 in the LAST tile — and the failure mode suspected here is that
 * EARLIER tiles' accumulators are dropped, which leaves the LAST tile's maximum intact.
 * That test therefore passes whether or not the defect exists. This probe does not use
 * softmax at all.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT RUNS INSTEAD, AND WHY IT IS STILL THE SHIPPING KERNEL
 * ---------------------------------------------------------------------------
 * The SHIPPING `resident_reduce_compute.c` triad, unmodified, driven with a program the
 * host's four named members do not name:
 *
 *     pass count 1 · pass0 = MAX / map NONE / verb NONE / no eps · source IN · chain SCALE
 *
 * i.e. y[r][c] = x[r][c] * rowmax(r). That is byte-for-byte the ROUTER NORMALISE program
 * shape (1 pass, source IN, chain SCALE, no gamma/beta/eps/mid, scaler 1.0) with two bytes
 * changed: pool SUM->MAX and verb RECIP->NONE. So every dataflow path it exercises is one
 * already green on card, and nothing but the pool differs. The host request is built for
 * FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE and the three program words are overwritten after
 * ft_tt_op_reduce_parameters() — a TEST-side override; no kernel and no backend source is
 * touched.
 *
 * With the row's background set to exactly 1.0, y[r][0] IS rowmax(r), read straight out of
 * DRAM. No correlation, no budget, no invariance: a number, per row, compared to a number.
 *
 * ---------------------------------------------------------------------------
 * WHAT DISTINGUISHES THE TWO OUTCOMES
 * ---------------------------------------------------------------------------
 * Each row carries ONE spike per tile, and the spikes are per-row distinct so a wrong
 * answer can never coincide with a right one:
 *
 *   case B (the question):  tile 0 spike = 100+r   tile 1 spike = 40+r
 *   case C (the mirror):    tile 0 spike =  40+r   tile 1 spike = 100+r
 *   case D (depth 4):       tile 0 = 100+r, tile 1 = 40+r, tile 2 = 30+r, tile 3 = 20+r
 *
 * The true row max is 100+r in all three. If the first reduce_tile's accumulator does not
 * survive the second's transpose, case B reports ~40+r and case C still reports 100+r —
 * the two are 60 apart at every row and the ARM that failed is named by which case moved.
 * Case D says whether it is "only the last tile survives" or something else again.
 *
 * ---------------------------------------------------------------------------
 * THE NEGATIVE CONTROL RUNS FIRST, AND IT IS A REAL DEVICE RUN
 * ---------------------------------------------------------------------------
 * Stage 1 uploads case C's data (64 columns, the true max in tile 1) and then asks the card
 * for `tile_cols = 1`. The kernel genuinely reduces one tile of a two-tile row — which IS
 * the defect under investigation, induced by a descriptor rather than by a mutation — and
 * the harness must report it RED against the 64-wide reference and GREEN against the
 * 32-wide one. Until that pair prints, a green stage 2 means nothing.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX  which enabled Tensix to drive
 *   FT_TT_FIRMWARE_DIR  the program directory whose fw_*.elf to boot
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=<dir> FT_TT_TENSIX_INDEX=60 \
 *     ./build-vector-card/test_reduce_max_multitile <reduce-program-dir>
 *
 * @status Probe. Not part of any acceptance suite.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_coord.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_vector.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define POISON_BYTE    0x5A
#define TILE_BYTES     FT_TT_OP_TILE_BYTES_BF16
#define TILE_DIM       FT_TT_OP_TILE_DIM
#define RUN_TIMEOUT_MS 20000u
#define QUEUE_CAPACITY 64u

#define DRAM_BANK_A     2u
#define DRAM_BANK_C     4u
#define DRAM_BANK_CONST 5u
#define DRAM_BANK_COUNT 6u
#define DRAM_BUFFER_BASE 0x00100000u

#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

#define TRIAD_MASK                                                                \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) | \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

/** Rows: ONE row block, so the only accumulation under test is the one across tiles. */
#define PROBE_ROWS 32u
/** The deepest row any case runs; the residency is placed for it once. */
#define PROBE_MAX_TILE_COLS 4u
/** A max is a max: the readback must be EXACT in bf16, not close. */
#define EXACT_TOLERANCE 0.0f

/** The carrier the probe reuses: one pass, source IN, chain SCALE, scaler 1.0, no gamma,
 *  no beta, no eps, no intermediate — the same dataflow the MAX program needs. */
#define CARRIER_FUNCTION FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_reduce;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

/** How many descriptors have completed on the resident triad so far. */
static uint32_t g_slot;

/**
 * Run the DIAGNOSTIC stages (5-10) as well as the verdict ones?
 *
 * Off by default, and that is a safety property rather than a preference: stages 8-10 send
 * descriptors whose width does not divide the residency's buffers, `descriptor_is_executable()`
 * REFUSES them, and a refused descriptor deliberately does not retire its slot -- so the
 * host's wait times out and the triad is left holding a queue it will never drain. That is
 * correct kernel behaviour and a stuck core for whoever runs it. The verdict stages size
 * every residency to its own descriptor and are refused by nothing.
 */
static bool diagnostics_enabled(void)
{
    const char *value = getenv("FT_PROBE_DIAGNOSTICS");

    return value && value[0] == '1';
}

/* ================================================================
 * Device access
 * ================================================================ */

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

static int poison_output(const ft_tt_noc_access *noc, size_t bytes)
{
    unsigned char pattern[0x1000];
    size_t offset = 0;

    memset(pattern, POISON_BYTE, sizeof(pattern));
    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > sizeof(pattern))
            chunk = sizeof(pattern);
        rc = device_write(noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE + (uint32_t)offset, pattern,
                          chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

static size_t poison_remaining(const void *data, size_t bytes)
{
    const unsigned char *cursor = data;
    size_t remaining = 0;
    size_t index;

    for (index = 0; index < bytes; index++)
        if (cursor[index] == POISON_BYTE)
            remaining++;
    return remaining;
}

/* ================================================================
 * Tilize / untilize
 * ================================================================ */

static int to_bf16_tiles(const float *matrix, uint32_t rows, uint32_t cols, uint16_t *out)
{
    uint16_t *plain = malloc((size_t)rows * cols * sizeof(uint16_t));
    size_t index;
    int rc;

    if (!plain)
        return FT_TT_ENOMEM;
    for (index = 0; index < (size_t)rows * cols; index++)
        plain[index] = ft_bf16_from_float(matrix[index]);
    rc = ft_tt_tilize_matrix(plain, sizeof(uint16_t), rows, cols, out);
    free(plain);
    return (rc == 0) ? FT_TT_OK : FT_TT_EINVAL;
}

static int from_bf16_tiles(const uint16_t *tiles, uint32_t rows, uint32_t cols, float *out)
{
    uint16_t *plain = malloc((size_t)rows * cols * sizeof(uint16_t));
    size_t index;
    int rc;

    if (!plain)
        return FT_TT_ENOMEM;
    rc = ft_tt_untilize_matrix(tiles, sizeof(uint16_t), rows, cols, plain);
    if (rc == 0) {
        for (index = 0; index < (size_t)rows * cols; index++)
            out[index] = ft_float_from_bf16(plain[index]);
    }
    free(plain);
    return (rc == 0) ? FT_TT_OK : FT_TT_EINVAL;
}

/* ================================================================
 * Bringing the resident reduce triad up
 * ================================================================ */

static int load_images(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
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

static int start_triad(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0,
                                            FT_TT_PROC_MATH1, FT_TT_PROC_MATH2 };
    uint32_t args[5][FT_TT_QUEUE_KERNEL_ARG_COUNT];
    ft_tt_op_args blocks[5];
    unsigned index;
    int rc;

    for (index = 0; index < 5u; index++) {
        ft_tt_queue_kernel_binding binding;

        memset(&binding, 0, sizeof(binding));
        binding.idle_spin_limit = IDLE_SPIN_LIMIT;
        binding.cb_wait_spin_limit = CB_SPIN_LIMIT;
        binding.status_address =
            FT_TT_QUEUE_STATUS_ADDRESS(g_queue.control_address, processors[index]);
        rc = ft_tt_queue_kernel_args(&g_queue, &binding, args[index]);
        if (rc != FT_TT_OK)
            return rc;
        blocks[index].processor = processors[index];
        blocks[index].values = args[index];
        blocks[index].count = FT_TT_QUEUE_KERNEL_ARG_COUNT;
    }

    rc = ft_tt_op_stage(noc, x, y, &g_config, g_cb_buffer, sizeof(g_cb_buffer), blocks, 5u);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_send_go(noc, x, y);
}

static void retire_triad(const ft_tt_noc_access *noc, const char *label)
{
    uint32_t unstopped = 0u;
    int rc = ft_tt_watchdog_shutdown(noc, &g_group, RUN_TIMEOUT_MS, &unstopped);

    ft_tt_test_check_fmt(label, rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=0x%X", rc,
                         unstopped);
}

static int start_reduce_triad(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                              const ft_tt_op_reduce_residency *residency)
{
    uint64_t mask = 0;
    int rc = load_images(&session->noc, x, y, &g_reduce);

    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, x, y, &g_reduce.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_reduce_place_cbs(residency, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer),
                                       &mask);
    if (rc != FT_TT_OK)
        return rc;

    g_config.local_cb_mask = mask;
    g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    g_config.brisc_noc_id = 0u;
    rc = ft_tt_queue_init(&session->noc, x, y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          ft_tt_queue_make_nonce(), &g_queue);
    if (rc != FT_TT_OK)
        return rc;
    return start_triad(&session->noc, x, y);
}

/* ================================================================
 * The operands
 * ================================================================ */

/** One spike: at `column`, every row r carries `base + r`. */
typedef struct spike {
    uint32_t column;
    float base;
} spike;

/**
 * A row of 1.0 with one spike per tile.
 *
 * 1.0 as the background is load-bearing twice over: y = x * rowmax means y[r][0] IS
 * rowmax(r) with no division to blur it, and every value here (1, 20..131) is exact in
 * bf16, so a mismatch is a wrong answer and never a rounding.
 */
static void build_input(float *x, uint32_t rows, uint32_t cols, const spike *spikes,
                        size_t spike_count)
{
    size_t index;
    uint32_t row;

    for (index = 0; index < (size_t)rows * cols; index++)
        x[index] = 1.0f;
    for (row = 0; row < rows; row++) {
        size_t which;

        for (which = 0; which < spike_count; which++)
            x[(size_t)row * cols + spikes[which].column] = spikes[which].base + (float)row;
    }
    /* Round through bf16 HERE, so the reference is computed from the values the card will
     * actually see. Without it a base of 300 reads as a defect: bf16 steps by 2 above 256,
     * so 301 is not representable and a correct card returns 300 or 302 against a reference
     * that says 301. The bases below are all bf16-exact anyway; this makes that a property
     * of the code rather than of the person who chose them. */
    for (index = 0; index < (size_t)rows * cols; index++)
        x[index] = ft_float_from_bf16(ft_bf16_from_float(x[index]));
}

/** The row maxima over columns [first_col, last_col). The reference, computed from the
 *  DEFINITION of a max over the host's own copy of the data — never from the kernel. */
static void reference_max(const float *x, uint32_t rows, uint32_t cols, uint32_t first_col,
                          uint32_t last_col, float *out)
{
    uint32_t row;

    for (row = 0; row < rows; row++) {
        float best = x[(size_t)row * cols + first_col];
        uint32_t col;

        for (col = first_col + 1u; col < last_col; col++) {
            const float value = x[(size_t)row * cols + col];

            if (value > best)
                best = value;
        }
        out[row] = best;
    }
}

static int upload_operands(const ft_tt_hardware_session *session, const float *x, uint32_t rows,
                           uint32_t cols, uint16_t *tiles, size_t output_bytes)
{
    const size_t bytes = (size_t)(rows / TILE_DIM) * (cols / TILE_DIM) * TILE_BYTES;
    const size_t constant_bytes =
        ft_tt_op_reduce_constants_bytes(CARRIER_FUNCTION, cols / TILE_DIM, TILE_BYTES);
    unsigned char *constants = malloc(constant_bytes);
    int rc;

    if (!constants)
        return FT_TT_ENOMEM;
    /* The carrier's scaler is 1.0, which is also what upstream uses for MAX
     * (ttnn/kernel_lib/reduce_helpers_dataflow.hpp: for SUM/MAX the scaler is 1.0). */
    rc = ft_tt_op_reduce_build_constants(CARRIER_FUNCTION, cols, 0.0f, NULL, NULL, constants,
                                         constant_bytes);
    if (rc == FT_TT_OK)
        rc = to_bf16_tiles(x, rows, cols, tiles);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_A], DRAM_BUFFER_BASE, tiles, bytes);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_CONST], DRAM_BUFFER_BASE, constants,
                          constant_bytes);
    if (rc == FT_TT_OK)
        rc = poison_output(&session->noc, output_bytes);
    free(constants);
    return rc;
}

/* ================================================================
 * One device run
 * ================================================================ */

/**
 * Enqueue ONE descriptor whose program is MAX / none / none, source IN, chain SCALE.
 *
 * The three program words are overwritten AFTER ft_tt_op_reduce_parameters() has filled the
 * shape and address words from the carrier request. Nothing but bytes 0 and 2 of the pass
 * word and byte 0 of the final word differ from the router normalise the card already runs.
 */
static bool run_max_reduce(const ft_tt_hardware_session *session, uint32_t rows,
                           uint32_t reduce_tile_cols, uint32_t input_tile_cols, float *actual,
                           uint16_t *tiles, size_t *out_poison)
{
    const uint32_t out_cols = reduce_tile_cols * TILE_DIM;
    const size_t bytes = (size_t)(rows / TILE_DIM) * reduce_tile_cols * TILE_BYTES;
    ft_tt_op_reduce_request request;
    ft_tt_queue_descriptor descriptor;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    char label[160];
    int rc;

    (void)input_tile_cols;

    memset(&request, 0, sizeof(request));
    request.function = CARRIER_FUNCTION;
    request.row_blocks = rows / TILE_DIM;
    request.tile_cols = reduce_tile_cols;
    request.prefetch_tiles = reduce_tile_cols;
    request.input.x = (uint16_t)g_dram[DRAM_BANK_A].x;
    request.input.y = (uint16_t)g_dram[DRAM_BANK_A].y;
    request.input.address = DRAM_BUFFER_BASE;
    request.input.tile_bytes = TILE_BYTES;
    request.constants.x = (uint16_t)g_dram[DRAM_BANK_CONST].x;
    request.constants.y = (uint16_t)g_dram[DRAM_BANK_CONST].y;
    request.constants.address = DRAM_BUFFER_BASE;
    request.constants.tile_bytes = TILE_BYTES;
    request.output.x = (uint16_t)g_dram[DRAM_BANK_C].x;
    request.output.y = (uint16_t)g_dram[DRAM_BANK_C].y;
    request.output.address = DRAM_BUFFER_BASE;
    request.output.tile_bytes = TILE_BYTES;

    rc = ft_tt_op_reduce_parameters(&request, words);
    if (rc == FT_TT_OK) {
        words[FT_TT_OP_REDUCE_PASS0] =
            (FT_TT_OP_REDUCE_POOL_MAX << (FT_TT_OP_REDUCE_BYTE_POOL * 8u)) |
            (FT_TT_OP_REDUCE_MAP_NONE << (FT_TT_OP_REDUCE_BYTE_MAP * 8u)) |
            (FT_TT_OP_REDUCE_VERB_NONE << (FT_TT_OP_REDUCE_BYTE_VERB * 8u));
        words[FT_TT_OP_REDUCE_PASS1] = 0u;
        words[FT_TT_OP_REDUCE_FINAL] =
            (1u << (FT_TT_OP_REDUCE_BYTE_PASS_COUNT * 8u)) |
            (FT_TT_OP_REDUCE_SOURCE_IN << (FT_TT_OP_REDUCE_BYTE_SOURCE * 8u)) |
            (FT_TT_OP_REDUCE_CHAIN_SCALE << (FT_TT_OP_REDUCE_BYTE_CHAIN * 8u));
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, g_slot, words,
                                             &parameter_address);
    }
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the MAX descriptor is prepared", false, "rc=%d", rc);
        return false;
    }

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_REDUCE;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    snprintf(label, sizeof(label), "MAX*scale %ux%u (Wt = %u) completes", rows, out_cols,
             reduce_tile_cols);
    ft_tt_test_check_fmt(label, rc == FT_TT_OK, "rc=%d completed=%u of %u", rc, completed,
                         g_slot + 1u);
    if (rc != FT_TT_OK)
        return false;
    g_slot++;

    /* DID IT EXECUTE, OR WAS IT REFUSED? An advancing completion count is NOT the same as a
     * descriptor that ran. `descriptor_is_executable()` sets kErrBadParameters and
     * `ftq_abort()` DRAINS -- every push and pop still happens so the page counters stay
     * balanced (resident_reduce_reader.c, "Rule 3") -- so a REFUSED descriptor still retires
     * its slot and still lets the writer produce an output tile. The kernel error word and
     * the five per-processor status words are the only things that separate the two, and
     * without them a wrong answer cannot be attributed to anything. */
    {
        static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1,
                                                FT_TT_PROC_MATH0, FT_TT_PROC_MATH1,
                                                FT_TT_PROC_MATH2 };
        static const char *names[5] = { "DM0", "DM1", "M0", "M1", "M2" };
        uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
        const char *reason = NULL;
        unsigned which;
        int completion_rc;

        (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
        completion_rc = ft_tt_queue_check_completion(&g_queue, fields, g_slot, &reason);
        printf("      control: state=%s error=%s(0x%08X) tail=%u completed=%u check=%s\n",
               ft_tt_queue_state_string(fields[FT_TT_QUEUE_FIELD_STATE]),
               ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]),
               fields[FT_TT_QUEUE_FIELD_ERROR], fields[FT_TT_QUEUE_FIELD_TAIL],
               fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT],
               (completion_rc == FT_TT_OK) ? "ok" : (reason ? reason : "?"));
        printf("      per-processor:");
        for (which = 0; which < 5u; which++) {
            uint32_t status[FT_TT_QUEUE_STATUS_WORDS];

            if (ft_tt_queue_read_status(&session->noc, &g_queue, processors[which], status) !=
                FT_TT_OK) {
                printf("  %s=<unread>", names[which]);
                continue;
            }
            printf("  %s=%s/%s", names[which],
                   ft_tt_queue_state_string(status[FT_TT_QUEUE_STATUS_STATE]),
                   ft_tt_queue_kernel_error_string(status[FT_TT_QUEUE_STATUS_ERROR]));
        }
        printf("\n");
    }

    rc = device_read(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, tiles, bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the output reads back", false, "rc=%d", rc);
        return false;
    }
    *out_poison = poison_remaining(tiles, bytes);
    /* Printed for EVERY descriptor, not only the ones that are scored: a refused descriptor
     * that never wrote its output reads back as pure poison, and that is the difference
     * between "the kernel computed this" and "this is what was in DRAM already". */
    printf("      output: %zu of %zu bytes still poison\n", *out_poison, bytes);
    rc = from_bf16_tiles(tiles, rows, out_cols, actual);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the output untilizes", false, "rc=%d", rc);
        return false;
    }
    return true;
}

/**
 * Recover the row maxima the CARD computed.
 *
 * y[r][c] = x[r][c] * rowmax(r) and the background is 1.0, so column 0 of the output is the
 * maximum itself. Divided by the input anyway, so a background that was ever not 1.0 would
 * be a wrong number rather than a silently rescaled one.
 */
static void extract_device_max(const float *actual, const float *input, uint32_t rows,
                               uint32_t out_cols, uint32_t in_cols, float *out)
{
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float denominator = input[(size_t)row * in_cols];

        out[row] = (denominator != 0.0f) ? actual[(size_t)row * out_cols] / denominator : NAN;
    }
}

/** Report every row that disagrees, and by how much. A summary that collapses to one
 *  number cannot say WHICH rows moved, and which rows moved is the whole characterisation. */
static void compare_maxima(const char *label, const float *expected, const float *device,
                           uint32_t rows, bool expect_pass)
{
    uint32_t mismatches = 0;
    uint32_t row;
    uint32_t first_bad = 0;
    double worst = 0.0;
    char detail[256];

    for (row = 0; row < rows; row++) {
        const double difference = fabs((double)device[row] - (double)expected[row]);

        if (!(difference <= (double)EXACT_TOLERANCE)) {
            if (mismatches == 0u)
                first_bad = row;
            mismatches++;
            if (difference > worst)
                worst = difference;
        }
    }

    if (mismatches == 0u)
        snprintf(detail, sizeof(detail), "all %u rows exact", rows);
    else
        snprintf(detail, sizeof(detail),
                 "%u of %u rows wrong, first row %u: device %.6g, reference %.6g (worst |d| "
                 "= %.6g)",
                 mismatches, rows, first_bad, (double)device[first_bad],
                 (double)expected[first_bad], worst);

    ft_tt_test_check_fmt(label, (mismatches == 0u) == expect_pass, "%s", detail);
}

/** The per-row table, printed unconditionally. The verdict is a claim; this is evidence. */
static void print_table(const char *title, const float *device, const float *const *candidates,
                        const char *const *names, size_t candidate_count, uint32_t rows)
{
    uint32_t row;
    size_t which;

    printf("\n  %s\n    row | device", title);
    for (which = 0; which < candidate_count; which++)
        printf(" | %s", names[which]);
    printf("\n");
    for (row = 0; row < rows; row++) {
        printf("    %3u | %6.1f", row, (double)device[row]);
        for (which = 0; which < candidate_count; which++)
            printf(" | %6.1f", (double)candidates[which][row]);
        printf("\n");
    }
    printf("\n");
}

/* ================================================================
 * The cases
 * ================================================================ */

/** Case B — the max lives in the FIRST of two tiles. The question, stated as data. */
static const spike g_case_b[] = { { 3u, 100.0f }, { 39u, 40.0f } };
/** Case C — the mirror: the max lives in the SECOND tile. */
static const spike g_case_c[] = { { 3u, 40.0f }, { 39u, 100.0f } };
/** Case D — four tiles, strictly decreasing spikes, so the answer names WHICH tile won. */
static const spike g_case_d[] = {
    { 3u, 100.0f }, { 39u, 40.0f }, { 75u, 30.0f }, { 111u, 20.0f }
};

typedef struct probe_buffers {
    float *input;
    float *actual;
    uint16_t *tiles;
    float *device_max;
    float *reference_all;
    float *reference_first;
    float *reference_last;
} probe_buffers;

static void free_buffers(probe_buffers *buffers)
{
    free(buffers->input);
    free(buffers->actual);
    free(buffers->tiles);
    free(buffers->device_max);
    free(buffers->reference_all);
    free(buffers->reference_first);
    free(buffers->reference_last);
    memset(buffers, 0, sizeof(*buffers));
}

static bool allocate_buffers(probe_buffers *buffers, uint32_t rows, uint32_t max_cols)
{
    const size_t count = (size_t)rows * max_cols;

    memset(buffers, 0, sizeof(*buffers));
    buffers->input = malloc(count * sizeof(float));
    buffers->actual = malloc(count * sizeof(float));
    buffers->tiles = malloc(count * sizeof(uint16_t));
    buffers->device_max = malloc(rows * sizeof(float));
    buffers->reference_all = malloc(rows * sizeof(float));
    buffers->reference_first = malloc(rows * sizeof(float));
    buffers->reference_last = malloc(rows * sizeof(float));
    if (!buffers->input || !buffers->actual || !buffers->tiles || !buffers->device_max ||
        !buffers->reference_all || !buffers->reference_first || !buffers->reference_last) {
        free_buffers(buffers);
        return false;
    }
    return true;
}

/**
 * STAGE 1 — the negative control, and it must go RED.
 *
 * Case C's data (true max in tile 1) uploaded at 64 columns, reduced with `tile_cols = 1`.
 * That is a row reduce that only ever saw the FIRST tile — the exact defect this probe is
 * hunting, induced by a descriptor. The pair of checks below is what makes a later green
 * mean something: the harness must call the 64-wide comparison FAILED and the 32-wide one
 * PASSED, off the same device output.
 */
static bool stage_1_negative_control(const ft_tt_hardware_session *session,
                                     probe_buffers *buffers)
{
    const uint32_t cols = 64u;
    size_t poison = 0;
    const float *candidates[2];
    const char *names[2] = { "true(64)", "tile0(32)" };

    ft_tt_test_step("stage 1 — NEGATIVE CONTROL: reduce ONE tile of a TWO-tile row. The "
                    "64-wide check MUST fail; if it does not, nothing after it is evidence");

    build_input(buffers->input, PROBE_ROWS, cols, g_case_c,
                sizeof(g_case_c) / sizeof(g_case_c[0]));
    reference_max(buffers->input, PROBE_ROWS, cols, 0u, cols, buffers->reference_all);
    reference_max(buffers->input, PROBE_ROWS, cols, 0u, TILE_DIM, buffers->reference_first);

    if (upload_operands(session, buffers->input, PROBE_ROWS, cols, buffers->tiles,
                        (size_t)(PROBE_ROWS / TILE_DIM) * 1u * TILE_BYTES) != FT_TT_OK) {
        ft_tt_test_check("the negative control's operands upload", false, "");
        return false;
    }
    if (!run_max_reduce(session, PROBE_ROWS, 1u, cols / TILE_DIM, buffers->actual,
                        buffers->tiles, &poison))
        return false;

    /* A handful of surviving poison BYTES is not a kernel that did not run: 0x5A5A is a
     * representable bf16 and the answers legitimately contain it. The check is that the
     * output is not still POISON, i.e. essentially all of it was overwritten. */
    ft_tt_test_check_fmt("the kernel wrote over the poison", poison * 100u < 2048u,
                         "%zu bytes still poison", poison);
    extract_device_max(buffers->actual, buffers->input, PROBE_ROWS, TILE_DIM, cols,
                       buffers->device_max);

    candidates[0] = buffers->reference_all;
    candidates[1] = buffers->reference_first;
    print_table("stage 1 (Wt = 1 over a 64-wide row)", buffers->device_max, candidates, names,
                2u, PROBE_ROWS);

    compare_maxima("A ONE-TILE reduce of a TWO-TILE row is REPORTED WRONG (this check must "
                   "FAIL for the probe to mean anything)",
                   buffers->reference_all, buffers->device_max, PROBE_ROWS, false);
    compare_maxima("and the same output is EXACTLY the first tile's max — the harness reads "
                   "a real number, not noise",
                   buffers->reference_first, buffers->device_max, PROBE_ROWS, true);
    return true;
}

/** One two-tile or four-tile case, scored against all three candidate answers. */
static bool run_case(const ft_tt_hardware_session *session, probe_buffers *buffers,
                     const char *title, const spike *spikes, size_t spike_count, uint32_t cols,
                     const char *verdict_label)
{
    const uint32_t tile_cols = cols / TILE_DIM;
    const size_t bytes = (size_t)(PROBE_ROWS / TILE_DIM) * tile_cols * TILE_BYTES;
    size_t poison = 0;
    const float *candidates[3];
    const char *names[3] = { "true", "tile0only", "lastonly" };

    build_input(buffers->input, PROBE_ROWS, cols, spikes, spike_count);
    reference_max(buffers->input, PROBE_ROWS, cols, 0u, cols, buffers->reference_all);
    reference_max(buffers->input, PROBE_ROWS, cols, 0u, TILE_DIM, buffers->reference_first);
    reference_max(buffers->input, PROBE_ROWS, cols, cols - TILE_DIM, cols,
                  buffers->reference_last);

    if (upload_operands(session, buffers->input, PROBE_ROWS, cols, buffers->tiles, bytes) !=
        FT_TT_OK) {
        ft_tt_test_check_fmt("the case's operands upload", false, "%s", title);
        return false;
    }
    if (!run_max_reduce(session, PROBE_ROWS, tile_cols, tile_cols, buffers->actual,
                        buffers->tiles, &poison))
        return false;

    ft_tt_test_check_fmt("the kernel wrote over the poison", poison * 100u < bytes,
                         "%zu bytes still poison", poison);
    extract_device_max(buffers->actual, buffers->input, PROBE_ROWS, cols, cols,
                       buffers->device_max);

    candidates[0] = buffers->reference_all;
    candidates[1] = buffers->reference_first;
    candidates[2] = buffers->reference_last;
    print_table(title, buffers->device_max, candidates, names, 3u, PROBE_ROWS);

    compare_maxima(verdict_label, buffers->reference_all, buffers->device_max, PROBE_ROWS,
                   true);
    return true;
}

/* ================================================================
 * Stage 5 — characterisation
 * ================================================================ */

/**
 * ONE spike, swept over (tile index x in-face column x row width).
 *
 * Stage 3 came back with a row max that was EXACTLY 64x the right answer while stages 2 and
 * 4 were exact, and its case differs from theirs in TWO ways at once — the winning value was
 * in a later tile AND at a different column inside its face. Two changes is not a finding.
 * These cases move ONE at a time.
 *
 * A single spike per case, so "which value should have won" has no second answer, and the
 * OBSERVED RATIO is printed rather than a pass/fail: the number is the characterisation.
 */
typedef struct sweep_case {
    const char *name;
    uint32_t cols;
    uint32_t spike_column;
} sweep_case;

static const sweep_case g_sweep[] = {
    /* Wt = 2: tile 0 vs tile 1, at the SAME in-face column each time. */
    { "Wt=2 spike in tile 0, face col 3", 64u, 3u },
    { "Wt=2 spike in tile 1, face col 3", 64u, 35u },
    /* Wt = 2: the same two tiles, at in-face column 7 — stage 3's column. */
    { "Wt=2 spike in tile 0, face col 7", 64u, 7u },
    { "Wt=2 spike in tile 1, face col 7", 64u, 39u },
    /* The RIGHT face of a tile (columns 16..31), which no case has touched. */
    { "Wt=2 spike in tile 0, face col 19", 64u, 19u },
    { "Wt=2 spike in tile 1, face col 19", 64u, 51u },
    /* Wt = 4: does the factor follow the row width, the tile index, or neither? */
    { "Wt=4 spike in tile 0, face col 7", 128u, 7u },
    { "Wt=4 spike in tile 2, face col 7", 128u, 71u },
    { "Wt=4 spike in tile 3, face col 7", 128u, 103u },
    /* Wt = 1: one tile, so no accumulation at all — the control for the sweep. */
    { "Wt=1 spike in tile 0, face col 7", 32u, 7u }
};

#define SWEEP_COUNT (sizeof(g_sweep) / sizeof(g_sweep[0]))

/**
 * Run one swept case and print the ratio device/reference, plus the ratio at four OUTPUT
 * columns.
 *
 * The second half matters: y = x * col0, so if the whole row is off by the same factor the
 * defect is in the REDUCE (col0 itself is wrong). If only column 0 is off, it is in the pack
 * or in the read, and the reduce is fine. One print separates those.
 */
static bool run_sweep_case(const ft_tt_hardware_session *session, probe_buffers *buffers,
                           const sweep_case *item)
{
    const uint32_t cols = item->cols;
    const uint32_t tile_cols = cols / TILE_DIM;
    const size_t bytes = (size_t)(PROBE_ROWS / TILE_DIM) * tile_cols * TILE_BYTES;
    const spike one = { 0u, 100.0f };
    spike spikes[1];
    size_t poison = 0;
    uint32_t row;
    double first_ratio = 0.0;
    bool uniform = true;
    bool exact = true;

    spikes[0] = one;
    spikes[0].column = item->spike_column;

    build_input(buffers->input, PROBE_ROWS, cols, spikes, 1u);
    reference_max(buffers->input, PROBE_ROWS, cols, 0u, cols, buffers->reference_all);

    if (upload_operands(session, buffers->input, PROBE_ROWS, cols, buffers->tiles, bytes) !=
        FT_TT_OK) {
        ft_tt_test_check_fmt("the swept case's operands upload", false, "%s", item->name);
        return false;
    }
    if (!run_max_reduce(session, PROBE_ROWS, tile_cols, tile_cols, buffers->actual,
                        buffers->tiles, &poison))
        return false;

    extract_device_max(buffers->actual, buffers->input, PROBE_ROWS, cols, cols,
                       buffers->device_max);

    for (row = 0; row < PROBE_ROWS; row++) {
        const double ratio = (double)buffers->device_max[row] / (double)buffers->reference_all[row];

        if (row == 0u)
            first_ratio = ratio;
        else if (fabs(ratio - first_ratio) > 1e-3)
            uniform = false;
        if (fabs((double)buffers->device_max[row] - (double)buffers->reference_all[row]) > 0.0)
            exact = false;
    }

    printf("    %-36s ratio %s%.6g   (row 0: device %.6g, reference %.6g)\n", item->name,
           uniform ? "= " : "~ ", first_ratio, (double)buffers->device_max[0],
           (double)buffers->reference_all[0]);
    if (!exact) {
        /* y = x * col0. If every column carries the same factor the REDUCE is wrong; if
         * only column 0 does, the reduce is right and the pack or the read is not. */
        static const uint32_t probe_columns[4] = { 0u, 1u, 16u, 31u };
        unsigned which;

        printf("      per-output-column y/x at row 0:");
        for (which = 0; which < 4u; which++) {
            const uint32_t column = probe_columns[which];

            printf("  c%u=%.6g", column,
                   (double)buffers->actual[column] / (double)buffers->input[column]);
        }
        printf("\n");
    }
    return true;
}

static void stage_5_characterise(const ft_tt_hardware_session *session, probe_buffers *buffers)
{
    unsigned index;
    bool all_exact = true;

    ft_tt_test_step("stage 5 — characterisation: ONE spike, swept over tile index and "
                    "in-face column, so the two things stage 3 changed at once are "
                    "separated");
    printf("\n");
    for (index = 0; index < SWEEP_COUNT; index++) {
        if (!run_sweep_case(session, buffers, &g_sweep[index]))
            return;
        {
            uint32_t row;

            for (row = 0; row < PROBE_ROWS; row++) {
                if (fabs((double)buffers->device_max[row] -
                         (double)buffers->reference_all[row]) > 0.0)
                    all_exact = false;
            }
        }
    }
    printf("\n");
    ft_tt_test_check("every swept single-spike row max is EXACT", all_exact,
                     "see the ratio table above for the ones that are not");
}

/* ================================================================
 * Stages 6-7 — is the odd case about the REDUCE, or about the RESIDENCY?
 * ================================================================ */

/**
 * The sweep's shape is not "a later tile loses". Wt = 4 finds a spike in tile 2 and in
 * tile 3 exactly; Wt = 1 is exact; only Wt = 2 -- the one width that is LESS than the
 * residency's max_tile_cols and still has more than one tile -- gets it wrong. That is a
 * statement about the CIRCULAR BUFFERS, which are placed once per residency, and not about
 * reduce_tile.
 *
 * Two runs settle it, and both use the SAME descriptor:
 *   stage 6 -- the same call three times in a row. Same answer three times means the
 *              defect is deterministic in the data, not a stale page left by whatever ran
 *              before it.
 *   stage 7 -- the same call on a residency placed for max_tile_cols = 2, so Wt EQUALS the
 *              depth the buffers were placed for. If that is exact, the defect is the
 *              residency/descriptor width mismatch and reduce_tile is not implicated at
 *              all.
 */
static void run_probe_case_wide(const ft_tt_hardware_session *session, probe_buffers *buffers,
                                const char *label, float base, uint32_t tile_cols)
{
    const uint32_t cols = tile_cols * TILE_DIM;
    const spike one = { (tile_cols > 1u) ? 39u : 7u, base };
    const size_t bytes = (size_t)(PROBE_ROWS / TILE_DIM) * tile_cols * TILE_BYTES;
    size_t poison = 0;
    uint32_t row;
    uint32_t wrong = 0;

    build_input(buffers->input, PROBE_ROWS, cols, &one, 1u);
    reference_max(buffers->input, PROBE_ROWS, cols, 0u, cols, buffers->reference_all);

    if (upload_operands(session, buffers->input, PROBE_ROWS, cols, buffers->tiles, bytes) !=
        FT_TT_OK) {
        ft_tt_test_check_fmt("operands upload", false, "%s", label);
        return;
    }
    if (!run_max_reduce(session, PROBE_ROWS, tile_cols, tile_cols, buffers->actual,
                        buffers->tiles, &poison))
        return;
    extract_device_max(buffers->actual, buffers->input, PROBE_ROWS, cols, cols,
                       buffers->device_max);

    printf("    %-30s device rows 0..7:", label);
    for (row = 0; row < 8u; row++)
        printf(" %.6g", (double)buffers->device_max[row]);
    printf("   (reference %.6g..%.6g)\n", (double)buffers->reference_all[0],
           (double)buffers->reference_all[7]);
    for (row = 0; row < PROBE_ROWS; row++) {
        if (buffers->device_max[row] != buffers->reference_all[row])
            wrong++;
    }
    ft_tt_test_check_fmt(label, wrong == 0u, "%u of %u rows wrong", wrong, PROBE_ROWS);
}

static void run_probe_case_valued(const ft_tt_hardware_session *session,
                                  probe_buffers *buffers, const char *label, float base)
{
    run_probe_case_wide(session, buffers, label, base, 2u);
}

static void run_probe_case(const ft_tt_hardware_session *session, probe_buffers *buffers,
                           const char *label)
{
    run_probe_case_wide(session, buffers, label, 100.0f, 2u);
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_op_reduce_residency residency;
    ft_tt_op_reduce_function carrier = CARRIER_FUNCTION;
    probe_buffers buffers;
    uint16_t x = 0;
    uint16_t y = 0;
    uint32_t tensix_index = 0;
    uint32_t bank;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <reduce-program-dir>\n", argv[0]);
        return 2;
    }

    ft_tt_test_step("multi-tile row MAX — does a second reduce_tile see the first one's "
                    "accumulator?");

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    rc = ft_tt_hardware_session_tensix_from_environment(&session, &x, &y, &tensix_index);
    if (rc != FT_TT_OK)
        goto release;

    for (bank = 0; bank < DRAM_BANK_COUNT; bank++) {
        rc = ft_tt_coord_dram_bank(&session.grid, bank, FT_TT_COORD_TRANSLATED, &g_dram[bank]);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("DRAM bank coordinates derive", false, "bank %u rc=%d", bank,
                                 rc);
            goto release;
        }
    }

    rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &g_reduce);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the reduce program directory loads", false, "rc=%d", rc);
        goto release;
    }

    rc = ft_tt_launch_core_set_init(&g_group.cores, g_placement_storage, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_core_set_add_tensix(&g_group.cores, &session.grid, tensix_index, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the core set derives and classifies the target", false,
                             "index %u rc=%d", tensix_index, rc);
        goto release_program;
    }
    g_group.id = 0u;
    g_group.first_tensix_index = tensix_index;
    g_group.tensix_index_count = 1u;
    g_group.queues = &g_queue;
    g_group.resident_processor_mask = TRIAD_MASK;

    rc = ft_tt_core_boot_firmware(&session.noc, x, y, &firmware, NULL);
    ft_tt_test_check_fmt("firmware boots on the target core", rc == FT_TT_OK,
                         "index %u -> (%u, %u) rc=%d", tensix_index, x, y, rc);
    if (rc != FT_TT_OK)
        goto release_program;

    if (!allocate_buffers(&buffers, PROBE_ROWS, PROBE_MAX_TILE_COLS * TILE_DIM)) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto release_program;
    }

    /* EVERY VERDICT STAGE GETS A RESIDENCY SIZED TO ITS OWN DESCRIPTOR, which is also what
     * every shipped caller does (test_phase6_vector.c places the residency at the shape's
     * own tile_cols). A residency wider than the descriptor is refused by
     * `descriptor_is_executable()` as soon as the page pointer leaves the descriptor's grid,
     * and a refused descriptor does not retire -- so mixing widths on one residency is a
     * stuck triad, not a wrong answer. Stages 8-10 explore exactly that and are opt-in. */
    memset(&residency, 0, sizeof(residency));
    rc = ft_tt_op_reduce_residency_for(&carrier, 1u, 1u, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(&session, x, y, &residency);
    ft_tt_test_check_fmt("the reduce triad starts, placed for Wt <= 1", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto free_host;

    if (stage_1_negative_control(&session, &buffers)) {
        retire_triad(&session.noc, "the triad retires between widths");
        g_slot = 0u;
        memset(&residency, 0, sizeof(residency));
        rc = ft_tt_op_reduce_residency_for(&carrier, 1u, 2u, TILE_BYTES, &residency);
        if (rc == FT_TT_OK)
            rc = start_reduce_triad(&session, x, y, &residency);
        ft_tt_test_check_fmt("the reduce triad restarts, placed for Wt <= 2", rc == FT_TT_OK,
                             "rc=%d", rc);
    }
    if (rc == FT_TT_OK) {
        ft_tt_test_step("stage 2 — the question: max in the FIRST of TWO tiles. If the first "
                        "reduce_tile's accumulator does not survive the second's transpose, "
                        "this reports ~40+r instead of 100+r");
        if (run_case(&session, &buffers, "case B (Wt = 2, max in tile 0)", g_case_b,
                     sizeof(g_case_b) / sizeof(g_case_b[0]), 64u,
                     "MULTI-TILE MAX: a max in the FIRST tile survives the SECOND tile's "
                     "reduce_tile")) {
            ft_tt_test_step("stage 3 — the mirror: max in the SECOND tile. Green either way "
                            "if the defect exists, so its job is to NAME the direction");
            if (run_case(&session, &buffers, "case C (Wt = 2, max in tile 1)", g_case_c,
                         sizeof(g_case_c) / sizeof(g_case_c[0]), 64u,
                         "and a max in the LAST tile survives too")) {
                retire_triad(&session.noc, "the triad retires between widths");
                g_slot = 0u;
                memset(&residency, 0, sizeof(residency));
                rc = ft_tt_op_reduce_residency_for(&carrier, 1u, 4u, TILE_BYTES, &residency);
                if (rc == FT_TT_OK)
                    rc = start_reduce_triad(&session, x, y, &residency);
                ft_tt_test_check_fmt("the reduce triad restarts, placed for Wt <= 4",
                                     rc == FT_TT_OK, "rc=%d", rc);
                if (rc == FT_TT_OK) {
                    ft_tt_test_step("stage 4 — depth: four tiles, strictly decreasing spikes, "
                                    "so the reported value NAMES the tile that won");
                    (void)run_case(&session, &buffers, "case D (Wt = 4, max in tile 0)",
                                   g_case_d, sizeof(g_case_d) / sizeof(g_case_d[0]), 128u,
                                   "and it survives THREE more reduce_tile calls at Wt = 4");
                    ft_tt_test_step("stage 5 — Wt = 4 with a SINGLE spike, in a tile that is "
                                    "NOT the first: nothing else in the row can supply the "
                                    "answer, so a dropped accumulator shows as the 1.0 "
                                    "background");
                    printf("\n");
                    run_probe_case_wide(&session, &buffers, "Wt=4, single spike in tile 1",
                                        100.0f, 4u);
                    printf("\n");
                }
            }
        }
    }

    retire_triad(&session.noc, "all five processors retire when ASKED");

    if (!diagnostics_enabled()) {
        ft_tt_test_step("stages 6-10 (the residency-width diagnostics) are OPT-IN: they send "
                        "descriptors the kernel correctly REFUSES, and a refused descriptor "
                        "does not retire, so they leave the triad holding a queue it will "
                        "never drain. Set FT_PROBE_DIAGNOSTICS=1 to run them");
        goto free_host;
    }

    g_slot = 0u;
    memset(&residency, 0, sizeof(residency));
    rc = ft_tt_op_reduce_residency_for(&carrier, 1u, 4u, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(&session, x, y, &residency);
    if (rc == FT_TT_OK)
        stage_5_characterise(&session, &buffers);

    /* stage 7 — the same descriptor on a residency whose buffers are placed for exactly the
     * width the descriptor uses. Everything else is held constant. */
    ft_tt_test_step("stage 7 — the SAME Wt = 2 call on a residency placed for "
                    "max_tile_cols = 2, so the descriptor's width EQUALS the depth the "
                    "circular buffers were placed for");
    g_slot = 0u;
    memset(&residency, 0, sizeof(residency));
    rc = ft_tt_op_reduce_residency_for(&carrier, 1u, 2u, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(&session, x, y, &residency);
    ft_tt_test_check_fmt("the reduce triad restarts, placed for Wt <= 2", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc == FT_TT_OK) {
        printf("\n");
        run_probe_case(&session, &buffers, "Wt=2 on a Wt<=2 residency");
        printf("\n");
        retire_triad(&session.noc, "and the triad retires again when ASKED");
    }

    /* stage 8 — name the LAG. Four consecutive Wt = 2 descriptors on a Wt <= 4 residency,
     * each with a DIFFERENT spike value. If a wrong run returns a value that was never in
     * its own input but WAS in an earlier descriptor's, the defect is a stale circular-
     * buffer page and not a reduction of the wrong data. Which earlier descriptor names the
     * lag, and the lag is what makes the fix mechanical. */
    ft_tt_test_step("stage 8 — four consecutive Wt = 2 descriptors on a Wt <= 4 residency, "
                    "each with a DIFFERENT maximum: which input does a wrong run return?");
    g_slot = 0u;
    memset(&residency, 0, sizeof(residency));
    rc = ft_tt_op_reduce_residency_for(&carrier, 1u, 4u, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(&session, x, y, &residency);
    ft_tt_test_check_fmt("the reduce triad restarts, placed for Wt <= 4", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc == FT_TT_OK) {
        printf("\n");
        run_probe_case_valued(&session, &buffers, "descriptor 1, max 100+r", 100.0f);
        run_probe_case_valued(&session, &buffers, "descriptor 2, max 200+r", 200.0f);
        run_probe_case_valued(&session, &buffers, "descriptor 3, max 300+r", 300.0f);
        run_probe_case_valued(&session, &buffers, "descriptor 4, max 200+r", 200.0f);
        printf("\n");
        retire_triad(&session.noc, "and the triad retires a third time when ASKED");
    }

    /* stage 9 — THE PREDICTION.
     *
     * Stage 8 shows that Wt = 2 descriptors on a Wt <= 4 residency are correct when they
     * are all the SAME width; stage 6 shows them alternating once other widths have run.
     * So the trigger is a width CHANGE, and the reader says why: it reserves `depth` pages
     * and then issues ONE CONTIGUOUS NoC read of `depth * TILE_BYTES` at the write pointer
     * (resident_reduce_reader.c, the cb_in block loop). That is only correct while the
     * `depth` pages do not WRAP the circular buffer -- and a descriptor of a different
     * width leaves the page pointer at an offset that is not a multiple of the next
     * descriptor's depth, after which every other block wraps.
     *
     * Concretely, on a 4-page cb_in: two Wt = 2 descriptors leave the pointer at page 0. A
     * Wt = 1 descriptor leaves it at page 1. The next Wt = 2 takes pages 1-2 (contiguous,
     * fine); the one after takes pages 3-0, WRAPS, and the single contiguous read lands
     * past the end of the buffer instead. So the prediction is: correct, correct, [Wt=1],
     * correct, WRONG, correct, WRONG.
     *
     * If that sequence prints, the defect is the reader's block read and not reduce_tile.
     */
    ft_tt_test_step("stage 9 — the PREDICTION: a Wt = 1 descriptor leaves the cb_in page "
                    "pointer on an odd page, after which every OTHER Wt = 2 block wraps the "
                    "4-page buffer. Expect: ok, ok, [Wt=1], ok, WRONG, ok, WRONG");
    g_slot = 0u;
    memset(&residency, 0, sizeof(residency));
    rc = ft_tt_op_reduce_residency_for(&carrier, 1u, 4u, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(&session, x, y, &residency);
    ft_tt_test_check_fmt("the reduce triad restarts, placed for Wt <= 4", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc == FT_TT_OK) {
        printf("\n");
        run_probe_case_wide(&session, &buffers, "1: Wt=2 (pages 0-1)", 100.0f, 2u);
        run_probe_case_wide(&session, &buffers, "2: Wt=2 (pages 2-3)", 200.0f, 2u);
        run_probe_case_wide(&session, &buffers, "3: Wt=1 MISALIGNS (page 0)", 100.0f, 1u);
        run_probe_case_wide(&session, &buffers, "4: Wt=2 (pages 1-2, no wrap)", 200.0f, 2u);
        run_probe_case_wide(&session, &buffers, "5: Wt=2 (pages 3-0, WRAPS)", 100.0f, 2u);
        run_probe_case_wide(&session, &buffers, "6: Wt=2 (pages 1-2, no wrap)", 200.0f, 2u);
        run_probe_case_wide(&session, &buffers, "7: Wt=2 (pages 3-0, WRAPS)", 100.0f, 2u);
        printf("\n");
        retire_triad(&session.noc, "and the triad retires a fourth time when ASKED");
    }

    /* stage 10 — CAN THE GUARD FIRE AT ALL?
     *
     * Everything above depends on where the cb_in write pointer HAPPENS to be, which is
     * history and therefore arguable. This does not. `ft_tensix_cb_block_grid_is_safe()`
     * refuses on three terms, and the FIRST -- `fifo_num_pages % chunk_pages != 0` -- needs
     * no history at all: place the residency at max_tile_cols = 3 and send a Wt = 2
     * descriptor, and 3 % 2 = 1 on the very first descriptor after a fresh launch, pointer
     * at the base.
     *
     * So this is the guard's own red/green, on one residency:
     *   Wt = 3 and Wt = 1 divide 3  -> accepted, and must compute the right answer;
     *   Wt = 2 does not             -> MUST report kErrBadParameters.
     * If the Wt = 2 descriptor comes back with no error, the guard is INERT on this path and
     * that is the finding, not the wrap. */
    ft_tt_test_step("stage 10 — the GUARD's own red/green: a residency of THREE pages, where "
                    "Wt=3 and Wt=1 divide the buffer and Wt=2 does not. No pointer history "
                    "is involved -- three is not a multiple of two, on the first descriptor "
                    "after launch");
    g_slot = 0u;
    memset(&residency, 0, sizeof(residency));
    rc = ft_tt_op_reduce_residency_for(&carrier, 1u, 3u, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(&session, x, y, &residency);
    ft_tt_test_check_fmt("the reduce triad restarts, placed for Wt <= 3", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc == FT_TT_OK) {
        printf("\n");
        run_probe_case_wide(&session, &buffers, "Wt=2 on 3 pages (3%%2!=0, MUST refuse)",
                            100.0f, 2u);
        run_probe_case_wide(&session, &buffers, "Wt=3 on 3 pages (divides, ok)", 200.0f, 3u);
        run_probe_case_wide(&session, &buffers, "Wt=1 on 3 pages (divides, ok)", 100.0f, 1u);
        printf("\n");
        retire_triad(&session.noc, "and the triad retires a fifth time when ASKED");
    }

free_host:
    free_buffers(&buffers);
release_program:
    ft_tt_program_directory_release(&g_reduce);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
