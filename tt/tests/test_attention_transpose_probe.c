/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_attention_transpose_probe.c
 * @brief ON-CARD: does `matmul_init(in0, in1, transpose)` actually transpose B on
 *   Blackhole? This is the ONE assumption ft_tt_op_memory.h PART 1 §1.3 records as
 *   INFERRED, and flash-decode attention's inner loop is written differently depending on
 *   the answer.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A SEPARATE, TINY PROGRAM
 * ---------------------------------------------------------------------------
 * The attention triad is ~1000 lines across three kernels. If the transpose flag does not
 * work, its scores come out COLUMN-oriented and its softmax reduces along the other axis —
 * a different inner loop, written before anything can be scored. So the flag is measured
 * first, on a program in which everything except the flag is already green on silicon: the
 * UNMODIFIED resident_matmul_{reader,writer}.cpp, plus a compute kernel that is
 * resident_matmul_compute.c with the flag taken from a descriptor word.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES THE ANSWER UNAMBIGUOUS
 * ---------------------------------------------------------------------------
 * The two arms differ by ONE WORD of one descriptor — same binary, same core, same
 * operands, same run — so nothing but the flag can explain a difference (the standard
 * docs/tt_bringup_log.md §18.4 held the N-major layout A/B to).
 *
 * Each arm is scored against BOTH candidate answers, A*B and A*B^T, with a NON-SYMMETRIC
 * random A and B. That is what makes it a measurement rather than a hope: a green
 * "transpose works" that was really "the flag was ignored and B happened to be symmetric"
 * cannot happen, and the report says which product each arm actually computed.
 *
 * Byte-exactness is the gate, not PCC. Both candidates are computed from the SAME bf16
 * operands the card sees, and a tile matmul over Kt=1 is a sum of 32 products — so an
 * exact match is a fair demand only for the ARGUMENT ORDER question, which is what this
 * is. PCC and max-abs are reported beside it, because a near-miss and a transposed answer
 * are different diagnoses.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX  which enabled Tensix to drive (60-79 are this worker's)
 *   FT_TT_FIRMWARE_DIR  the program directory whose fw_*.elf to boot
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=../programs/probe FT_TT_TENSIX_INDEX=60 \
 *     ./build-attn-card/test_attention_transpose_probe ../programs/probe
 *
 * @status See the Phase 6 attention report.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_coord.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_DIM       32u
#define TILE_BYTES     2048u
#define TILE_ELEMENTS  (TILE_DIM * TILE_DIM)
#define QUEUE_CAPACITY 64u
#define RUN_TIMEOUT_MS 20000u
#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u
#define POISON_BYTE     0xA5

#define DRAM_BANK_A 2u
#define DRAM_BANK_B 3u
#define DRAM_BANK_C 4u
#define DRAM_BANK_COUNT 6u
#define DRAM_BUFFER_BASE 0x00200000u

/** Word 14 of the matmul parameter block, which the matmul family leaves unused. */
#define PROBE_TRANSPOSE_WORD 14u

#define TRIAD_MASK                                                                \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) | \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_probe;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

static uint64_t g_random_state = 0x9E3779B97F4A7C15ull;

/** Full-mantissa random in [-1, 1): the operands have to carry low bits, or a wrong
 *  argument order can reproduce the right answer (dev notes §3). */
static float random_value(void)
{
    g_random_state = g_random_state * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((double)(g_random_state >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

static int device_write(const ft_tt_noc_access *noc, ft_tt_noc_coord where, uint32_t address,
                        const void *source, size_t bytes)
{
    const unsigned char *cursor = (const unsigned char *)source;
    size_t done = 0;

    while (done < bytes) {
        size_t chunk = bytes - done;
        int rc;

        if (chunk > 65536u)
            chunk = 65536u;
        rc = noc->write(noc->user, (uint16_t)where.x, (uint16_t)where.y,
                        (uint64_t)address + done, cursor + done, chunk);
        if (rc != FT_TT_OK)
            return rc;
        done += chunk;
    }
    return FT_TT_OK;
}

static int device_read(const ft_tt_noc_access *noc, ft_tt_noc_coord where, uint32_t address,
                       void *destination, size_t bytes)
{
    unsigned char *cursor = (unsigned char *)destination;
    size_t done = 0;

    while (done < bytes) {
        size_t chunk = bytes - done;
        int rc;

        if (chunk > 65536u)
            chunk = 65536u;
        rc = noc->read(noc->user, (uint16_t)where.x, (uint16_t)where.y,
                       (uint64_t)address + done, cursor + done, chunk);
        if (rc != FT_TT_OK)
            return rc;
        done += chunk;
    }
    return FT_TT_OK;
}

/** fp32 row-major -> one face-ordered bf16 tile. */
static void to_tile(const float *matrix, uint16_t *out)
{
    uint32_t row;

    for (row = 0; row < TILE_DIM; row++) {
        uint32_t column;

        for (column = 0; column < TILE_DIM; column++)
            out[ft_tt_tile_face_index(row, column)] =
                ft_bf16_from_float(matrix[row * TILE_DIM + column]);
    }
}

/** One face-ordered bf16 tile -> fp32 row-major. */
static void from_tile(const uint16_t *tile, float *out)
{
    uint32_t row;

    for (row = 0; row < TILE_DIM; row++) {
        uint32_t column;

        for (column = 0; column < TILE_DIM; column++)
            out[row * TILE_DIM + column] =
                ft_float_from_bf16(tile[ft_tt_tile_face_index(row, column)]);
    }
}

/** C = A * B, or C = A * B^T, in fp32 from bf16-rounded operands. */
static void reference_product(const float *a, const float *b, float *out, bool transpose_b)
{
    uint32_t row;

    for (row = 0; row < TILE_DIM; row++) {
        uint32_t column;

        for (column = 0; column < TILE_DIM; column++) {
            float sum = 0.0f;
            uint32_t k;

            for (k = 0; k < TILE_DIM; k++) {
                const float bv = transpose_b ? b[column * TILE_DIM + k] : b[k * TILE_DIM + column];

                sum += a[row * TILE_DIM + k] * bv;
            }
            out[row * TILE_DIM + column] = sum;
        }
    }
}

typedef struct match_score {
    uint32_t exact;
    double pcc;
    double max_abs;
} match_score;

/** How well `actual` matches `expected`: exact bf16 element count, PCC, max-abs. */
static match_score score_against(const float *expected, const float *actual)
{
    match_score result = { 0u, 0.0, 0.0 };
    double mean_e = 0.0;
    double mean_a = 0.0;
    double cov = 0.0;
    double var_e = 0.0;
    double var_a = 0.0;
    size_t index;

    for (index = 0; index < TILE_ELEMENTS; index++) {
        const double difference = fabs((double)expected[index] - (double)actual[index]);

        if (ft_bf16_from_float(expected[index]) == ft_bf16_from_float(actual[index]))
            result.exact++;
        if (difference > result.max_abs)
            result.max_abs = difference;
        mean_e += (double)expected[index];
        mean_a += (double)actual[index];
    }
    mean_e /= (double)TILE_ELEMENTS;
    mean_a /= (double)TILE_ELEMENTS;
    for (index = 0; index < TILE_ELEMENTS; index++) {
        const double de = (double)expected[index] - mean_e;
        const double da = (double)actual[index] - mean_a;

        cov += de * da;
        var_e += de * de;
        var_a += da * da;
    }
    result.pcc = (var_e > 0.0 && var_a > 0.0) ? cov / sqrt(var_e * var_a) : 0.0;
    return result;
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

/** Three CBs, two pages each: the matmul triad's contract at Mt=Kt=Nt=1. */
static int place_cbs(uint64_t *out_mask)
{
    ft_tt_op_cb_request requests[3];

    requests[0].index = 0u;
    requests[0].page_bytes = TILE_BYTES;
    requests[0].page_count = 2u;
    requests[1].index = 1u;
    requests[1].page_bytes = TILE_BYTES;
    requests[1].page_count = 2u;
    requests[2].index = 16u;
    requests[2].page_bytes = TILE_BYTES;
    requests[2].page_count = 2u;
    return ft_tt_op_cb_place(requests, 3u, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer),
                             out_mask);
}

/** One arm: run the 1x1x1 matmul with `transpose`, and score it against both candidates. */
static void run_arm(const ft_tt_hardware_session *session, uint32_t transpose, uint32_t slot,
                    const float *a, const float *b)
{
    float plain[TILE_ELEMENTS];
    float transposed[TILE_ELEMENTS];
    float actual[TILE_ELEMENTS];
    uint16_t tile[TILE_ELEMENTS];
    uint32_t words[FT_TT_QUEUE_MATMUL_WORDS];
    ft_tt_queue_descriptor descriptor;
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    match_score plain_score;
    match_score transposed_score;
    char label[128];
    int rc;

    memset(words, 0, sizeof(words));
    words[FT_TT_QUEUE_MATMUL_TILE_ROWS] = 1u;
    words[FT_TT_QUEUE_MATMUL_TILE_INNER] = 1u;
    words[FT_TT_QUEUE_MATMUL_TILE_COLS] = 1u;
    words[FT_TT_QUEUE_MATMUL_A_X] = g_dram[DRAM_BANK_A].x;
    words[FT_TT_QUEUE_MATMUL_A_Y] = g_dram[DRAM_BANK_A].y;
    words[FT_TT_QUEUE_MATMUL_A_BASE] = DRAM_BUFFER_BASE;
    words[FT_TT_QUEUE_MATMUL_B_X] = g_dram[DRAM_BANK_B].x;
    words[FT_TT_QUEUE_MATMUL_B_Y] = g_dram[DRAM_BANK_B].y;
    words[FT_TT_QUEUE_MATMUL_B_BASE] = DRAM_BUFFER_BASE;
    words[FT_TT_QUEUE_MATMUL_C_X] = g_dram[DRAM_BANK_C].x;
    words[FT_TT_QUEUE_MATMUL_C_Y] = g_dram[DRAM_BANK_C].y;
    words[FT_TT_QUEUE_MATMUL_C_BASE] = DRAM_BUFFER_BASE;
    words[FT_TT_QUEUE_MATMUL_DEPTH] = 1u;
    words[FT_TT_QUEUE_MATMUL_B_N_MAJOR] = 0u;
    words[PROBE_TRANSPOSE_WORD] = transpose;

    memset(tile, POISON_BYTE, sizeof(tile));
    rc = device_write(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, tile, sizeof(tile));
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, slot, words,
                                             &parameter_address);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the arm is prepared", false, "transpose=%u rc=%d", transpose, rc);
        return;
    }

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    snprintf(label, sizeof(label), "transpose=%u completes", transpose);
    ft_tt_test_check_fmt(label, rc == FT_TT_OK, "rc=%d completed=%u of %u", rc, completed,
                         slot + 1u);
    if (rc != FT_TT_OK)
        return;

    rc = device_read(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, tile, sizeof(tile));
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the result reads back", false, "rc=%d", rc);
        return;
    }
    from_tile(tile, actual);
    reference_product(a, b, plain, false);
    reference_product(a, b, transposed, true);
    plain_score = score_against(plain, actual);
    transposed_score = score_against(transposed, actual);

    printf("      transpose=%u : vs A*B    exact %u/1024 pcc %.9f max-abs %.6f\n", transpose,
           plain_score.exact, plain_score.pcc, plain_score.max_abs);
    printf("      transpose=%u : vs A*B^T  exact %u/1024 pcc %.9f max-abs %.6f\n", transpose,
           transposed_score.exact, transposed_score.pcc, transposed_score.max_abs);

    if (transpose == 0u) {
        ft_tt_test_check_fmt("transpose=0 computes A*B", plain_score.exact == TILE_ELEMENTS,
                             "exact %u/1024 pcc %.9f", plain_score.exact, plain_score.pcc);
        ft_tt_test_check_fmt("transpose=0 is NOT A*B^T (so B is not symmetric)",
                             transposed_score.exact < TILE_ELEMENTS, "exact %u/1024",
                             transposed_score.exact);
    } else {
        ft_tt_test_check_fmt("THE ASSUMPTION: transpose=1 computes A*B^T",
                             transposed_score.exact == TILE_ELEMENTS, "exact %u/1024 pcc %.9f",
                             transposed_score.exact, transposed_score.pcc);
        ft_tt_test_check_fmt("transpose=1 is NOT the plain product",
                             plain_score.exact < TILE_ELEMENTS, "exact %u/1024",
                             plain_score.exact);
    }
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    float a[TILE_ELEMENTS];
    float b[TILE_ELEMENTS];
    uint16_t tile[TILE_ELEMENTS];
    uint16_t x = 0;
    uint16_t y = 0;
    uint32_t tensix_index = 0;
    uint32_t bank;
    uint64_t mask = 0;
    size_t index;
    uint32_t unstopped = 0u;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <probe-program-dir>\n", argv[0]);
        return 2;
    }

    ft_tt_test_step("matmul_init(transpose) on Blackhole — the ONE inferred assumption "
                    "behind the KV layout");

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
            ft_tt_test_check_fmt("DRAM bank coordinates derive", false, "bank %u rc=%d", bank, rc);
            goto release;
        }
    }

    rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &g_probe);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the probe program loads", false, "rc=%d", rc);
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

    /* Operands: full-mantissa random, and DELIBERATELY not symmetric. */
    for (index = 0; index < TILE_ELEMENTS; index++) {
        a[index] = random_value();
        b[index] = random_value();
    }
    for (index = 0; index < TILE_ELEMENTS; index++) {
        a[index] = ft_float_from_bf16(ft_bf16_from_float(a[index]));
        b[index] = ft_float_from_bf16(ft_bf16_from_float(b[index]));
    }
    to_tile(a, tile);
    rc = device_write(&session.noc, g_dram[DRAM_BANK_A], DRAM_BUFFER_BASE, tile, sizeof(tile));
    if (rc == FT_TT_OK) {
        to_tile(b, tile);
        rc = device_write(&session.noc, g_dram[DRAM_BANK_B], DRAM_BUFFER_BASE, tile, sizeof(tile));
    }
    ft_tt_test_check_fmt("both operands upload", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto release_program;

    rc = ft_tt_op_program_load(&session.noc, x, y, &g_probe.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = place_cbs(&mask);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the probe program stages", false, "rc=%d", rc);
        goto release_program;
    }
    g_config.local_cb_mask = mask;
    g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    g_config.brisc_noc_id = 0u;
    rc = ft_tt_queue_init(&session.noc, x, y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          ft_tt_queue_make_nonce(), &g_queue);
    if (rc == FT_TT_OK)
        rc = start_triad(&session.noc, x, y);
    ft_tt_test_check_fmt("the probe triad goes resident", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto release_program;

    run_arm(&session, 0u, 0u, a, b);
    run_arm(&session, 1u, 1u, a, b);

    rc = ft_tt_watchdog_shutdown(&session.noc, &g_group, RUN_TIMEOUT_MS, &unstopped);
    ft_tt_test_check_fmt("every processor retires by ASKING", rc == FT_TT_OK && unstopped == 0u,
                         "rc=%d unstopped=0x%X", rc, unstopped);

release_program:
    ft_tt_program_directory_release(&g_probe);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
