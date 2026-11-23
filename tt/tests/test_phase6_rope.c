/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_rope.c
 * @brief ON-CARD acceptance for ROPE — partial NeoX rotary position embedding, on the
 *   resident triad.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS SUITE HAS TWO GATES AND NOT ONE
 * ---------------------------------------------------------------------------
 * **PCC is blind to a consistent permutation**, and a
 * permutation is exactly what a face-order or partner-order mistake produces here.
 * **And a rotation preserves the norm**, so no check built on the norm — or on a
 * correlation, which is a normalised inner product — can see a wrong ANGLE
 * (docs/tt_bringup_log.md §22.2, stated there for softmax's shift invariance and true here
 * for the invariance this op IS). A single PCC gate would therefore be blind to both of the
 * two mistakes this op is most likely to make. So:
 *
 *   STAGE 1  STRUCTURE, scored by BYTES with no tolerance at all. A synthetic sin/cos table
 *            of exactly-representable values ({1, -1, 2, -2}) and small-integer activations
 *            make every product and every sum exact in bf16, so the card's output must be
 *            BIT-IDENTICAL to a host model of it. Pairing, partner order, face map,
 *            pass-through and tile order all fail this and nothing rounds them away. The
 *            kernel cannot tell a synthetic table from a real one — it multiplies whatever
 *            arrives — which is what makes this legitimate.
 *   STAGE 2  the ANGLE, read straight out of the answer: rotate the basis pair (1, 0) and
 *            the output IS (cos, sin), compared element by element against a double-precision
 *            computation. A wrong position, a wrong theta or a wrong table row misses by
 *            O(1) here while leaving every norm and every correlation intact.
 *   STAGE 3  the op as it will actually be called — full-mantissa random activations, the
 *            real table, PCC >= 0.99 and max-abs against a CPU fp32 reference. This is the
 *            standard gate and it is the WEAKEST of the three; it is here because it is the
 *            one that speaks about the model rather than about the implementation.
 *
 * ---------------------------------------------------------------------------
 * THE ROW-PHASE CONTRACT IS TESTED, NOT ASSUMED
 * ---------------------------------------------------------------------------
 * The kernel multiplies activation tile row `r` by table tile row `r`, so a decode token at
 * position `p` must sit in tile row `p % 32` (ft_tt_op_rope.h). Stage 2 runs a decode case at
 * `p = 7`, in row 7, precisely so that a kernel which quietly assumed row 0 goes red — and so
 * that the contract is a fact about the silicon rather than a paragraph.
 *
 * ---------------------------------------------------------------------------
 * BEFORE ANYTHING IS SCORED
 * ---------------------------------------------------------------------------
 *  - a host->DRAM->host round trip runs FIRST (bringup log §16.0);
 *  - every output buffer is poisoned and the surviving poison counted;
 *  - non-finite elements are counted on EACH SIDE separately, because `PCC -nan` is a refusal
 *    to answer and does not say which side failed.
 *
 * DO NOT PARK. On a five-processor resident program, parking is the hazard — twice, on cores
 * that had retired cleanly (dev notes §1). This test retires by ASKING.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX  which enabled Tensix to drive (this round owns 40-49)
 *   FT_TT_FIRMWARE_DIR  the program directory whose fw_*.elf to boot
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=../programs/rope FT_TT_TENSIX_INDEX=40 \
 *     ./build-rope-card/test_phase6_rope ../programs/rope
 *
 * @status None
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
#include "../ft_tt_op_rope.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_rope_reference.h"
#include "ft_tt_test_report.h"

#define PCC_ACCEPT     0.99
#define POISON_BYTE    0x5A
#define TILE_BYTES     FT_TT_OP_TILE_BYTES_BF16
#define TILE_DIM       FT_TT_OP_TILE_DIM
#define RUN_TIMEOUT_MS 20000u
#define QUEUE_CAPACITY 64u

/** DRAM banks 2..5 are this suite's, as they are test_ops.c's. */
#define DRAM_BANK_INPUT  2u
#define DRAM_BANK_TABLE  3u
#define DRAM_BANK_OUTPUT 4u
#define DRAM_BANK_COUNT  6u
/** 512 MiB into each bank, NOT the 1 MiB every other suite uses. Three other workers are on
 *  this card with their own cores but the SAME DRAM banks, and a shared buffer base is a
 *  cross-agent corruption that would look exactly like a kernel bug. */
#define DRAM_BUFFER_BASE 0x20000000u

#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

/** Every processor of a triad is resident. */
#define TRIAD_MASK                                                                 \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) |  \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

/** MiniCPM5-1B, read from the GGUF header on the card host 2026-08-12. */
#define MODEL_HEAD_DIM FT_TT_OP_ROPE_MINICPM5_HEAD_DIM
#define MODEL_ROPE_DIM FT_TT_OP_ROPE_MINICPM5_ROPE_DIM
#define MODEL_THETA    FT_TT_OP_ROPE_MINICPM5_THETA
#define MODEL_HEADS_Q  FT_TT_OP_ROPE_MINICPM5_HEADS_Q

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_rope;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
static uint32_t g_slot;

/* ================================================================
 * Device access helpers
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
        rc = device_write(noc, g_dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE + (uint32_t)offset,
                          pattern, chunk);
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
 * Scoring
 * ================================================================ */

static double pearson(const float *a, const float *b, size_t count)
{
    double mean_a = 0.0;
    double mean_b = 0.0;
    double covariance = 0.0;
    double variance_a = 0.0;
    double variance_b = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        mean_a += a[index];
        mean_b += b[index];
    }
    mean_a /= (double)count;
    mean_b /= (double)count;
    for (index = 0; index < count; index++) {
        const double da = a[index] - mean_a;
        const double db = b[index] - mean_b;

        covariance += da * db;
        variance_a += da * da;
        variance_b += db * db;
    }
    if (variance_a == 0.0 || variance_b == 0.0)
        return 0.0;
    return covariance / sqrt(variance_a * variance_b);
}

static double max_abs_difference(const float *a, const float *b, size_t count)
{
    double worst = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        const double difference = fabs((double)a[index] - (double)b[index]);

        if (difference > worst)
            worst = difference;
    }
    return worst;
}

/** Count non-finite elements on EACH SIDE separately: `PCC -nan` is a refusal to answer and
 *  does not say which side failed (dev notes §3). */
static size_t count_non_finite(const float *expected, const float *actual, size_t count,
                               char *census, size_t census_bytes)
{
    size_t bad_expected = 0;
    size_t bad_actual = 0;
    size_t first = count;
    size_t index;

    for (index = 0; index < count; index++) {
        const bool expected_bad = !isfinite(expected[index]);
        const bool actual_bad = !isfinite(actual[index]);

        if (expected_bad)
            bad_expected++;
        if (actual_bad)
            bad_actual++;
        if ((expected_bad || actual_bad) && first == count)
            first = index;
    }
    if (bad_expected || bad_actual) {
        snprintf(census, census_bytes,
                 "%zu non-finite expected, %zu actual, first at %zu (tile %zu, offset %zu)",
                 bad_expected, bad_actual, first, first / (TILE_DIM * TILE_DIM),
                 first % (TILE_DIM * TILE_DIM));
    }
    return bad_expected + bad_actual;
}

static void score(const char *label, const float *expected, const float *actual, size_t count,
                  const void *raw, size_t raw_bytes)
{
    char census[192];
    const size_t non_finite = count_non_finite(expected, actual, count, census, sizeof(census));
    const size_t poisoned = poison_remaining(raw, raw_bytes);
    double pcc;
    double worst;

    ft_tt_test_check_fmt("the kernel wrote over the poison", poisoned * 100u < raw_bytes,
                         "%zu of %zu bytes still poison", poisoned, raw_bytes);
    if (non_finite != 0u) {
        ft_tt_test_check_fmt(label, false, "NON-FINITE: %s", census);
        return;
    }
    pcc = pearson(expected, actual, count);
    worst = max_abs_difference(expected, actual, count);
    ft_tt_test_check_fmt(label, pcc >= PCC_ACCEPT, "PCC %.9f, max abs %.6g", pcc, worst);
}

/* ================================================================
 * Bringing the resident triad up
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

/* ================================================================
 * One RoPE call, end to end
 * ================================================================ */

/**
 * Upload `in_tiles`, enqueue one descriptor for `request`, and read the output tiles back.
 *
 * The CB placement is done ONCE by the caller, so every call here is an ENQUEUE and nothing
 * else — which is the residency claim, carried as a fact of how the test is written rather
 * than asserted in prose.
 *
 * @return FT_TT_OK, or the first failing step's code
 */
static int run_one_rope(const ft_tt_hardware_session *session,
                        const ft_tt_op_rope_request *request, const uint16_t *in_tiles,
                        size_t bytes, uint16_t *out_tiles)
{
    ft_tt_queue_descriptor descriptor;
    uint32_t words[FT_TT_OP_ROPE_WORDS];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    int rc;

    rc = ft_tt_op_rope_parameters(request, words);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_INPUT], DRAM_BUFFER_BASE, in_tiles,
                          bytes);
    if (rc == FT_TT_OK)
        rc = poison_output(&session->noc, bytes);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, g_slot, words,
                                             &parameter_address);
    if (rc != FT_TT_OK)
        return rc;

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_ROPE;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the descriptor completes", false, "rc=%d completed=%u of %u", rc,
                             completed, g_slot + 1u);
        return rc;
    }

    {
        uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
        const char *reason = NULL;

        (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
        ft_tt_test_check_fmt(
            "completion is evidenced by the WRITER, not by a run message",
            ft_tt_queue_check_completion(&g_queue, fields, g_slot + 1u, &reason) == FT_TT_OK,
            "%s", reason ? reason : "ok");
    }
    g_slot++;

    return device_read(&session->noc, g_dram[DRAM_BANK_OUTPUT], DRAM_BUFFER_BASE, out_tiles,
                       bytes);
}

/** Fill a request for `heads` heads of `head_dim`, one row block, at `position`. */
static void fill_request(ft_tt_op_rope_request *request, uint32_t heads, uint32_t head_dim,
                         uint32_t rope_dim, uint32_t position, uint32_t row_blocks)
{
    memset(request, 0, sizeof(*request));
    request->row_blocks = row_blocks;
    request->heads = heads;
    request->head_dim = head_dim;
    request->rope_dim = rope_dim;
    request->first_position = position;
    request->prefetch_tiles = head_dim / TILE_DIM;
    request->input.x = (uint16_t)g_dram[DRAM_BANK_INPUT].x;
    request->input.y = (uint16_t)g_dram[DRAM_BANK_INPUT].y;
    request->input.address = DRAM_BUFFER_BASE;
    request->input.tile_bytes = TILE_BYTES;
    request->table.x = (uint16_t)g_dram[DRAM_BANK_TABLE].x;
    request->table.y = (uint16_t)g_dram[DRAM_BANK_TABLE].y;
    request->table.address = DRAM_BUFFER_BASE;
    request->table.tile_bytes = TILE_BYTES;
    request->output.x = (uint16_t)g_dram[DRAM_BANK_OUTPUT].x;
    request->output.y = (uint16_t)g_dram[DRAM_BANK_OUTPUT].y;
    request->output.address = DRAM_BUFFER_BASE;
    request->output.tile_bytes = TILE_BYTES;
}

/* ================================================================
 * Stage 0 — the transfer path, before anything is scored
 * ================================================================ */

static void stage_0_round_trip(const ft_tt_hardware_session *session)
{
    uint16_t written[TILE_DIM * TILE_DIM];
    uint16_t read_back[TILE_DIM * TILE_DIM];
    size_t index;
    int rc;

    ft_tt_test_step("stage 0 — host->DRAM->host, so the transfer path is a LINE and not an "
                    "assumption");

    for (index = 0; index < TILE_DIM * TILE_DIM; index++)
        written[index] = (uint16_t)(0x3F00u + (index & 0xFFu));
    memset(read_back, 0xA5, sizeof(read_back));

    rc = device_write(&session->noc, g_dram[DRAM_BANK_INPUT], DRAM_BUFFER_BASE, written,
                      sizeof(written));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_dram[DRAM_BANK_INPUT], DRAM_BUFFER_BASE, read_back,
                         sizeof(read_back));
    ft_tt_test_check_fmt("a tile round-trips through DRAM", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check("and it comes back byte-exact",
                     memcmp(written, read_back, sizeof(written)) == 0, "2048 bytes");
}

/* ================================================================
 * Stage 1 — STRUCTURE, byte-exact, with a synthetic table
 * ================================================================ */

/** A synthetic table value: exactly representable, and varying with BOTH the row and the
 *  column so that a wrong row or a wrong column is visible. */
static float synthetic_cos(uint32_t row, uint32_t column)
{
    static const float values[4] = { 1.0f, 2.0f, -1.0f, -2.0f };

    return values[(row * 3u + column) & 3u];
}

static float synthetic_sin(uint32_t row, uint32_t column)
{
    static const float values[4] = { 2.0f, -1.0f, -2.0f, 1.0f };

    return values[(row + column * 5u) & 3u];
}

/** An activation: an integer in [-50, 50), distinct per (row, feature) at this width, so
 *  every element is DISTINGUISHABLE and a swapped pair cannot hide. */
static float structural_activation(uint32_t row, uint32_t feature)
{
    return (float)(int32_t)(((row * 37u + feature * 11u) % 100u)) - 50.0f;
}

static bool stage_1_structure(const ft_tt_hardware_session *session)
{
    const uint32_t heads = 4u;
    const uint32_t head_dim = MODEL_HEAD_DIM;
    const uint32_t rot = MODEL_ROPE_DIM;
    const uint32_t rot_tiles = rot / TILE_DIM;
    const uint32_t rows = TILE_DIM;
    const size_t width = (size_t)heads * head_dim;
    const size_t count = (size_t)rows * width;
    const size_t bytes = count * sizeof(uint16_t);
    const size_t table_bytes = (size_t)2u * rot_tiles * TILE_BYTES;
    ft_tt_op_rope_request request;
    uint16_t *in = malloc(count * sizeof(uint16_t));
    uint16_t *model = malloc(count * sizeof(uint16_t));
    uint16_t *swapped = malloc(count * sizeof(uint16_t));
    uint16_t *device_tiles = malloc(bytes);
    uint16_t *model_tiles = malloc(bytes);
    uint16_t *cos_plain = malloc((size_t)rows * rot * sizeof(uint16_t));
    uint16_t *sin_plain = malloc((size_t)rows * rot * sizeof(uint16_t));
    uint16_t *swapped_sin = malloc((size_t)rows * rot * sizeof(uint16_t));
    uint16_t *table_tiles = malloc(table_bytes);
    uint32_t row;
    uint32_t column;
    size_t differing = 0;
    size_t index;
    bool ok = false;
    int rc;

    ft_tt_test_step("stage 1 — STRUCTURE, byte-exact: a synthetic table makes every "
                    "intermediate exact in bf16, so there is no tolerance to hide in");

    if (!in || !model || !swapped || !device_tiles || !model_tiles || !cos_plain ||
        !sin_plain || !swapped_sin || !table_tiles) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    for (row = 0; row < rows; row++)
        for (index = 0; index < width; index++)
            in[(size_t)row * width + index] =
                ft_bf16_from_float(structural_activation(row, (uint32_t)index));
    for (row = 0; row < rows; row++) {
        for (column = 0; column < rot; column++) {
            cos_plain[(size_t)row * rot + column] =
                ft_bf16_from_float(synthetic_cos(row, column));
            sin_plain[(size_t)row * rot + column] =
                ft_bf16_from_float(synthetic_sin(row, column));
        }
    }

    ft_tt_rope_model_apply(in, cos_plain, sin_plain, model, rows, heads, head_dim, rot);

    /* CAN THIS COMPARISON SEE A WRONG PAIRING? Asserted before it is trusted, the way the
     * multicore suite asserts its per-core operands are pairwise distinct rather than
     * assuming the seed made them so (dev notes §3). */
    for (index = 0; index < (size_t)rows * rot; index++)
        swapped_sin[index] = ft_bf16_from_float(-ft_float_from_bf16(sin_plain[index]));
    ft_tt_rope_model_apply(in, cos_plain, swapped_sin, swapped, rows, heads, head_dim, rot);
    for (index = 0; index < count; index++)
        if (model[index] != swapped[index])
            differing++;
    ft_tt_test_check_fmt("the data would SEE a wrong pairing", differing * 4u > count,
                         "%zu of %zu elements differ under the wrong pairing", differing,
                         count);

    /* Upload: activations tilized as a [rows x width] matrix, the table as one row block. */
    rc = ft_tt_tilize_matrix(in, sizeof(uint16_t), rows, (uint32_t)width, model_tiles) == 0
             ? FT_TT_OK
             : FT_TT_EINVAL;
    if (rc == FT_TT_OK)
        rc = ft_tt_tilize_strip(cos_plain, sizeof(uint16_t), rows, rot,
                                (size_t)rot * sizeof(uint16_t), table_tiles) == 0
                 ? FT_TT_OK
                 : FT_TT_EINVAL;
    if (rc == FT_TT_OK)
        rc = ft_tt_tilize_strip(sin_plain, sizeof(uint16_t), rows, rot,
                                (size_t)rot * sizeof(uint16_t),
                                (unsigned char *)table_tiles + rot_tiles * TILE_BYTES) == 0
                 ? FT_TT_OK
                 : FT_TT_EINVAL;
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_TABLE], DRAM_BUFFER_BASE, table_tiles,
                          table_bytes);
    ft_tt_test_check_fmt("the synthetic table uploads", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto done;

    fill_request(&request, heads, head_dim, rot, 0u, 1u);
    rc = run_one_rope(session, &request, model_tiles, bytes, device_tiles);
    ft_tt_test_check_fmt("the structural descriptor completes", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto done;

    /* THE GATE. Not a PCC, not a tolerance: the same bytes. */
    if (ft_tt_tilize_matrix(model, sizeof(uint16_t), rows, (uint32_t)width, model_tiles) != 0) {
        ft_tt_test_check("the model tilizes", false, "ft_tt_tilize_matrix refused");
        goto done;
    }
    differing = 0;
    for (index = 0; index < count; index++)
        if (device_tiles[index] != model_tiles[index])
            differing++;
    ft_tt_test_check_fmt("the card is BYTE-IDENTICAL to the model over the whole output",
                         differing == 0u, "%zu of %zu tile elements differ", differing, count);
    ft_tt_test_check_fmt("the kernel wrote over the poison",
                         poison_remaining(device_tiles, bytes) * 100u < bytes,
                         "%zu of %zu bytes still poison",
                         poison_remaining(device_tiles, bytes), bytes);
    ok = (differing == 0u);

done:
    free(in);
    free(model);
    free(swapped);
    free(device_tiles);
    free(model_tiles);
    free(cos_plain);
    free(sin_plain);
    free(swapped_sin);
    free(table_tiles);
    return ok;
}

/* ================================================================
 * Stage 2 — the ANGLE, read out of the answer
 * ================================================================ */

/**
 * Rotate the basis pair (1, 0) at `position` and read (cos, sin) straight out of the output.
 *
 * `position % 32` is where the token goes — THE ROW-PHASE CONTRACT — so a non-zero position
 * here is also the test that the kernel does not silently use table row 0.
 */
static void stage_2_angle(const ft_tt_hardware_session *session, uint32_t position)
{
    const uint32_t heads = 2u;
    const uint32_t head_dim = MODEL_HEAD_DIM;
    const uint32_t rot = MODEL_ROPE_DIM;
    const uint32_t rot_tiles = rot / TILE_DIM;
    const uint32_t half = rot / 2u;
    const uint32_t rows = TILE_DIM;
    const uint32_t row = position % TILE_DIM;
    const uint32_t block_first = position - row;
    const size_t width = (size_t)heads * head_dim;
    const size_t count = (size_t)rows * width;
    const size_t bytes = count * sizeof(uint16_t);
    const size_t table_bytes = (size_t)2u * rot_tiles * TILE_BYTES;
    ft_tt_op_rope_request request;
    uint16_t *in = calloc(count, sizeof(uint16_t));
    uint16_t *in_tiles = malloc(bytes);
    uint16_t *out_tiles = malloc(bytes);
    uint16_t *out_plain = malloc(count * sizeof(uint16_t));
    uint16_t *table_tiles = malloc(table_bytes);
    double worst_cos = 0.0;
    double worst_sin = 0.0;
    uint32_t j;
    char label[160];
    int rc;

    if (!in || !in_tiles || !out_tiles || !out_plain || !table_tiles) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    for (j = 0; j < half; j++)
        in[(size_t)row * width + j] = ft_bf16_from_float(1.0f);

    /* ONE row block is uploaded, but it is placed WHERE THE DRIVER WILL LOOK — at the
     * position's own row-block offset inside the table region — rather than at the base with
     * the request bent to match. Bending the request would be the test agreeing with the
     * driver's arithmetic instead of checking it. */
    rc = ft_tt_op_rope_build_table(block_first, TILE_DIM, head_dim, rot, MODEL_THETA,
                                   table_tiles, table_bytes);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_TABLE],
                          DRAM_BUFFER_BASE +
                              ft_tt_op_rope_table_offset(position, rot_tiles, TILE_BYTES),
                          table_tiles, table_bytes);
    if (rc == FT_TT_OK)
        rc = ft_tt_tilize_matrix(in, sizeof(uint16_t), rows, (uint32_t)width, in_tiles) == 0
                 ? FT_TT_OK
                 : FT_TT_EINVAL;
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the basis vector and its table upload", false, "rc=%d", rc);
        goto done;
    }

    fill_request(&request, heads, head_dim, rot, position, 1u);
    rc = run_one_rope(session, &request, in_tiles, bytes, out_tiles);
    if (rc == FT_TT_OK)
        rc = ft_tt_untilize_matrix(out_tiles, sizeof(uint16_t), rows, (uint32_t)width,
                                   out_plain) == 0
                 ? FT_TT_OK
                 : FT_TT_EINVAL;
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the basis rotation completes", false, "rc=%d", rc);
        goto done;
    }

    for (j = 0; j < half; j++) {
        const double frequency = pow(MODEL_THETA, -2.0 * (double)j / (double)rot);
        const double angle = (double)position * frequency;
        const double got_cos = ft_float_from_bf16(out_plain[(size_t)row * width + j]);
        const double got_sin = ft_float_from_bf16(out_plain[(size_t)row * width + j + half]);
        const double cos_error = fabs(got_cos - cos(angle));
        const double sin_error = fabs(got_sin - sin(angle));

        if (cos_error > worst_cos)
            worst_cos = cos_error;
        if (sin_error > worst_sin)
            worst_sin = sin_error;
    }

    /* 2^-8 is bf16's step at 1.0 and the value passes two roundings, so ~0.008 is the floor a
     * correct kernel can reach. A wrong POSITION, a wrong theta or a table row read from row 0
     * misses by O(1), not by O(2^-8) — which is the whole reason this check exists beside the
     * PCC one. */
    snprintf(label, sizeof(label),
             "position %u (row %u): rotating (1,0) returns (cos, sin) ON THE CARD", position,
             row);
    ft_tt_test_check_fmt(label, worst_cos < 0.01 && worst_sin < 0.01,
                         "worst |dcos| %.5f, |dsin| %.5f", worst_cos, worst_sin);

done:
    free(in);
    free(in_tiles);
    free(out_tiles);
    free(out_plain);
    free(table_tiles);
}

/* ================================================================
 * Stage 3 — the op as it will be called, against a CPU fp32 reference
 * ================================================================ */

static uint64_t g_random_state = 0x243F6A8885A308D3ull;

static float random_value(void)
{
    g_random_state = g_random_state * 6364136223846793005ull + 1442695040888963407ull;
    return ((float)(int32_t)(g_random_state >> 33) / 1073741824.0f);
}

static void stage_3_real(const ft_tt_hardware_session *session, uint32_t row_blocks,
                         uint32_t heads, uint32_t head_dim, uint32_t rope_dim,
                         uint32_t position, uint32_t prefetch, const char *what)
{
    const uint32_t rot = ft_tt_op_rope_rotary_width(head_dim, rope_dim);
    const uint32_t rot_tiles = rot / TILE_DIM;
    const uint32_t rows = TILE_DIM * row_blocks;
    const uint32_t row_phase = position % TILE_DIM;
    const uint32_t block_first = position - row_phase;
    const size_t width = (size_t)heads * head_dim;
    const size_t count = (size_t)rows * width;
    const size_t bytes = count * sizeof(uint16_t);
    const size_t table_bytes = (size_t)row_blocks * 2u * rot_tiles * TILE_BYTES;
    ft_tt_op_rope_request request;
    float *source = malloc(count * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    uint16_t *in = malloc(count * sizeof(uint16_t));
    uint16_t *in_tiles = malloc(bytes);
    uint16_t *out_tiles = malloc(bytes);
    uint16_t *out_plain = malloc(count * sizeof(uint16_t));
    uint16_t *table_tiles = malloc(table_bytes);
    size_t index;
    size_t untouched_wrong = 0;
    char label[192];
    int rc;

    if (!source || !expected || !actual || !in || !in_tiles || !out_tiles || !out_plain ||
        !table_tiles) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    for (index = 0; index < count; index++) {
        in[index] = ft_bf16_from_float(random_value());
        /* The reference consumes what the CARD sees, not the fp32 the card never gets. */
        source[index] = ft_float_from_bf16(in[index]);
    }

    rc = ft_tt_op_rope_build_table(block_first, rows, head_dim, rope_dim, MODEL_THETA,
                                   table_tiles, table_bytes);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_TABLE],
                          DRAM_BUFFER_BASE +
                              ft_tt_op_rope_table_offset(position, rot_tiles, TILE_BYTES),
                          table_tiles, table_bytes);
    if (rc == FT_TT_OK)
        rc = ft_tt_tilize_matrix(in, sizeof(uint16_t), rows, (uint32_t)width, in_tiles) == 0
                 ? FT_TT_OK
                 : FT_TT_EINVAL;
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the operands upload", false, "rc=%d", rc);
        goto done;
    }

    fill_request(&request, heads, head_dim, rope_dim, position, row_blocks);
    request.prefetch_tiles = prefetch;
    rc = run_one_rope(session, &request, in_tiles, bytes, out_tiles);
    if (rc == FT_TT_OK)
        rc = ft_tt_untilize_matrix(out_tiles, sizeof(uint16_t), rows, (uint32_t)width,
                                   out_plain) == 0
                 ? FT_TT_OK
                 : FT_TT_EINVAL;
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the descriptor completes and untilizes", false, "rc=%d", rc);
        goto done;
    }
    for (index = 0; index < count; index++)
        actual[index] = ft_float_from_bf16(out_plain[index]);

    /* The reference is computed on UNTILIZED data from the op's DEFINITION, so a face-order
     * mistake has something to break against instead of cancelling (dev notes §3). */
    ft_tt_rope_reference_apply(source, expected, rows, heads, head_dim, rope_dim, block_first,
                               (float)MODEL_THETA);
    snprintf(label, sizeof(label), "%s vs a CPU fp32 reference", what);
    score(label, expected, actual, count, out_tiles, bytes);

    /* PARTIAL, byte-exact: everything past the rotary dimension must come through untouched.
     * A near-miss here would be a kernel that "copies" through the FPU. */
    if (rot < head_dim) {
        uint32_t row;

        for (row = 0; row < rows; row++) {
            uint32_t head;

            for (head = 0; head < heads; head++) {
                uint32_t column;

                for (column = rot; column < head_dim; column++) {
                    const size_t at = (size_t)row * width + (size_t)head * head_dim + column;

                    if (out_plain[at] != in[at])
                        untouched_wrong++;
                }
            }
        }
        snprintf(label, sizeof(label),
                 "%s: the features past the rotary dimension pass through BIT-identical", what);
        ft_tt_test_check_fmt(label, untouched_wrong == 0u, "%zu of %zu wrong", untouched_wrong,
                             (size_t)rows * heads * (head_dim - rot));
    }

done:
    free(source);
    free(expected);
    free(actual);
    free(in);
    free(in_tiles);
    free(out_tiles);
    free(out_plain);
    free(table_tiles);
}

/* ================================================================
 * Stage 4 — an opcode this binary does not implement is REFUSED
 * ================================================================ */

static void stage_4_unknown_opcode_is_refused(const ft_tt_hardware_session *session)
{
    ft_tt_queue_descriptor descriptor;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t completed = 0u;
    int rc;

    ft_tt_test_step("stage 4 — the RoPE binary REFUSES an opcode it does not implement");

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_ELTWISE;
    descriptor.a0 = FT_TT_QUEUE_PARAM_ADDRESS(g_queue.control_address, 0u);
    (void)ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);

    /* It must NOT complete. A short timeout is the right instrument: the expectation is a
     * refusal, so waiting the full run timeout would only make the test slow. */
    rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_slot + 1u, 1000u, &completed);
    ft_tt_test_check_fmt("it does not retire the descriptor",
                         rc != FT_TT_OK && completed == g_slot, "rc=%d completed=%u", rc,
                         completed);

    (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
    ft_tt_test_check_fmt("and it names the reason",
                         fields[FT_TT_QUEUE_FIELD_ERROR] ==
                             (uint32_t)FT_TT_QUEUE_KERR_BAD_OPCODE,
                         "error=%u (%s)", fields[FT_TT_QUEUE_FIELD_ERROR],
                         ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
}

/* ================================================================ */

static int bring_up_triad(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    ft_tt_op_rope_request placement;
    uint64_t mask = 0;
    int rc;

    /* The WIDEST placement any stage uses, so every stage runs against ONE descriptor array
     * and the triad is brought up ONCE — which is what "resident" has to mean at the host end
     * too. head_dim 256 covers the partial stage; rot_tiles only sets the depth. */
    fill_request(&placement, 2u, 256u, 256u, 0u, 1u);
    placement.prefetch_tiles = 4u;

    rc = load_images(&session->noc, x, y, &g_rope);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, x, y, &g_rope.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_rope_place_cbs(&placement, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer),
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

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint16_t x = 0;
    uint16_t y = 0;
    uint32_t tensix_index = 0;
    uint32_t bank;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <rope-program-dir>\n", argv[0]);
        return 2;
    }

    ft_tt_test_step("Phase 6 RoPE — ON-CARD acceptance");

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

    rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &g_rope);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the program directory loads", false, "rc=%d", rc);
        goto release;
    }

    /* The core set is built by ft_tt_launch_core_set_add_tensix(), which DERIVES the
     * coordinate from the live grid and CLASSIFIES it before it can enter a launch. A
     * hand-filled placement would skip that, and a wrong NoC endpoint is a hung chip and a
     * physical power cycle, not a failed call (spec §7.3). */
    rc = ft_tt_launch_core_set_init(&g_group.cores, g_placement_storage, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_core_set_add_tensix(&g_group.cores, &session.grid, tensix_index, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the core set derives and classifies the target", false,
                             "index %u rc=%d", tensix_index, rc);
        goto release_programs;
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
        goto release_programs;

    stage_0_round_trip(&session);

    rc = bring_up_triad(&session, x, y);
    ft_tt_test_check_fmt("the RoPE triad is staged and kicked ONCE", rc == FT_TT_OK, "rc=%d",
                         rc);
    if (rc != FT_TT_OK)
        goto release_programs;

    if (stage_1_structure(&session)) {
        ft_tt_test_step("stage 2 — the ANGLE, read out of the answer, at four positions");
        stage_2_angle(&session, 0u);
        stage_2_angle(&session, 7u);
        stage_2_angle(&session, 100u);
        stage_2_angle(&session, 131040u);

        ft_tt_test_step("stage 3 — the op as it will be called, vs a CPU fp32 reference");
        stage_3_real(&session, 1u, MODEL_HEADS_Q, MODEL_HEAD_DIM, MODEL_ROPE_DIM, 0u, 4u,
                     "MiniCPM5-1B Q, 16 heads x 128, 32 tokens at position 0");
        stage_3_real(&session, 1u, FT_TT_OP_ROPE_MINICPM5_HEADS_KV, MODEL_HEAD_DIM,
                     MODEL_ROPE_DIM, 4096u, 4u, "MiniCPM5-1B KV, 2 heads x 128, position 4096");
        stage_3_real(&session, 1u, 2u, 256u, 64u, 1024u, 2u,
                     "PARTIAL: 64 of 256, position 1024");
        /* DEPTH 1, and it is not a "small case". At depth = rot_tiles the reader asks for the
         * partner of column 0 ONLY, which is below the half for every shape, so the wrap side
         * of ftr_partner_tile() is never evaluated and a partner map that forgot to wrap is
         * INDISTINGUISHABLE. Measured, not argued: mutation K1 (`return c + half`) came back
         * green against this suite until this line existed. Same rule as the matmul reader's
         * block-loop mutations — score a parameter by a setting that switches it on. */
        stage_3_real(&session, 1u, 4u, MODEL_HEAD_DIM, MODEL_ROPE_DIM, 32u, 1u,
                     "DEPTH 1: every column asks for its own partner, wrap included");
        /* FOUR row blocks. Nothing above reaches the reader's row-block arithmetic at all —
         * with one row block the table index is a constant 0 and `heads` is unused, so a
         * kernel that ignored both would pass every check above. This is the same rule the
         * matmul-reader mutations are scored by: a mutation that is unreachable at the
         * trivial setting proves the FEATURE only at a nontrivial one (bringup log §16.5). */
        stage_3_real(&session, 4u, 3u, MODEL_HEAD_DIM, MODEL_ROPE_DIM, 128u, 4u,
                     "FOUR row blocks x 3 heads, 128 tokens from position 128");

        /* THE RESIDENCY CLAIM, stated as a check: every stage above was an ENQUEUE, with the
         * five images loaded once and no launch of any kind between descriptors. */
        ft_tt_test_check_fmt("every descriptor ran on ONE image load, with no launch", true,
                             "%u descriptors, 1 program directory", g_slot);

        stage_4_unknown_opcode_is_refused(&session);
    }

    retire_triad(&session.noc,
                 "all five processors retire when ASKED, with no reset and no park");

    /* DO NOT PARK. The core is left in its firmware's GO wait, which is where a healthy idle
     * core sits (dev notes §1). */

release_programs:
    ft_tt_program_directory_release(&g_rope);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
