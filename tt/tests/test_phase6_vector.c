/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_vector.c
 * @brief ON-CARD acceptance for the vector op family: the resident elementwise triad
 *   (SiLU, sigmoid, add, mul, sub, SiLU-gated multiply) and the resident REDUCE triad
 *   (softmax, RMS norm, layer norm, router normalise), each scored PCC >= 0.99 against a
 *   CPU fp32 reference — and, for the reduce family, a relative MAX-ABS budget as well,
 *   because PCC cannot see a global scale and three of its four members are scale-shaped.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES THE REFERENCE INDEPENDENT
 * ---------------------------------------------------------------------------
 * The same three things test_ops.c establishes for Phase 4, because they are the same
 * three traps (dev notes §3):
 *
 *  - the reference is a plain fp32 computation written from each op's DEFINITION
 *    (ft_tt_vector_reference.c), never from the kernel;
 *  - it consumes the values the CARD will actually see — bf16-rounded — so the score is
 *    about the op and not about a quantization the kernel is not responsible for;
 *  - the comparison runs on UNTILIZED data, so a face-ordering mistake has something to
 *    break against instead of cancelling. PCC is blind to a consistent permutation, and a
 *    reference that went through ft_tt_tilize would give it nothing to cancel against.
 *
 * ---------------------------------------------------------------------------
 * THE RESIDENCY CLAIM IS THE POINT OF STAGE 2, NOT A DETAIL
 * ---------------------------------------------------------------------------
 * All six elementwise functions run on ONE triad with the images loaded ONCE and no launch
 * of any kind between them. That is the whole argument for making them one binary rather
 * than six: a binary is bound to one slab, so six programs would mean five image reloads
 * — ~0.19 ms each over the x1 link — inside every SwiGLU.
 * The check that proves it is that descriptors 2..6 complete after no host action but an
 * enqueue.
 *
 * ---------------------------------------------------------------------------
 * BEFORE ANYTHING IS SCORED
 * ---------------------------------------------------------------------------
 *  - a host->DRAM->host round trip runs FIRST, so "the transfer path is fine" is a line in
 *    the report rather than an assumption (bringup log §16.0);
 *  - every output buffer is poisoned and the surviving poison counted, so a kernel that
 *    never ran scores ~0 rather than reading back the previous stage's answer;
 *  - non-finite elements are counted on EACH SIDE separately, because `PCC -nan` is a
 *    refusal to answer and does not say which side failed.
 *
 * ---------------------------------------------------------------------------
 * DO NOT PARK
 * ---------------------------------------------------------------------------
 * On a five-processor resident program, parking is the hazard — twice, on cores that had
 * retired cleanly (dev notes §1, bringup log §20.3). This test retires by ASKING and leaves
 * the core in its firmware's GO wait, which is where a healthy idle core sits.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX  which enabled Tensix to drive (default 0; pick a healthy one)
 *   FT_TT_FIRMWARE_DIR  the program directory whose fw_*.elf to boot
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=../programs/eltwise FT_TT_TENSIX_INDEX=101 \
 *     ./build-vector-card/test_phase6_vector ../programs/eltwise ../programs/softmax
 *
 * @status Green on Tensix 20, 0 checks failed. The two program directories it wants are
 *   tt/programs/{eltwise,reduce}.
 * @issues None
 * @todo None
 */

/* clock_gettime(CLOCK_MONOTONIC) is POSIX.1-2008, not C99. */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
#include "ft_tt_vector_reference.h"

#define PCC_ACCEPT     0.99
#define POISON_BYTE    0x5A
#define TILE_BYTES     FT_TT_OP_TILE_BYTES_BF16
#define TILE_DIM       FT_TT_OP_TILE_DIM
#define RUN_TIMEOUT_MS 20000u
#define QUEUE_CAPACITY 64u

/** DRAM banks 2..5 are this suite's, as they are test_ops.c's; 0 and 1 belong to the
 *  Phase 2 follow-ups and the triad gate. */
#define DRAM_BANK_A       2u
#define DRAM_BANK_B       3u
#define DRAM_BANK_C       4u
#define DRAM_BANK_CONST   5u
#define DRAM_BANK_COUNT   6u
#define DRAM_BUFFER_BASE  0x00100000u

/** The two bounds. ~400M spins is ~10 s of blocked polling, which is there for a host that
 *  DIED and not for flow control; the shutdown ask is what a normal exit uses. Neither may
 *  be 0 — an unbounded wait is the one thing a resident kernel may not be configured into
 *  (kernels/resident_queue_protocol.h rule 4). */
#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

/** Every processor of a triad is resident. */
#define TRIAD_MASK                                                                 \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) |  \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

/** The elementwise shape: 48 tiles, i.e. a 32 x 1536 activation block — MiniCPM5-1B's
 *  hidden width, so the number sits beside the matmul shapes rather than beside nothing. */
#define ELTWISE_TILES 48u
#define ELTWISE_ROWS  TILE_DIM
#define ELTWISE_COLS  (ELTWISE_TILES * TILE_DIM)

/** How many softmax shapes stage 3 runs. */
#define SOFTMAX_SHAPE_COUNT 3u

typedef struct softmax_shape {
    uint32_t rows;
    uint32_t cols;
    uint32_t prefetch_tiles;
} softmax_shape;

/**
 * WHERE the +200 outlier goes, and why the position is the whole test.
 *
 * A spike in tile T is invisible to any partial row max whose tile set CONTAINS T, and
 * catastrophic to one whose set does not. So a single placement covers exactly one half of
 * the failure space, and the half it covers is decided by where you put it:
 *
 *   LAST  catches a reduce that keeps only the FIRST tile (the loop truncated to one trip)
 *   FIRST catches a reduce that keeps only the LAST  tile (an overwrite where an accumulate
 *         was intended — the DST section reset on every trip)
 *
 * Both mutations are one character of `resident_reduce_compute.c`'s `reduce_row_max()` loop,
 * and each is GREEN against the other's placement. MEASURED 2026-08-14 at Wt = 48, one
 * mutation site (`reduce_row_max()`'s loop and nothing else), each arm on a core the other
 * arms never touched:
 *
 *   build                                     core  spike LAST  spike FIRST  suite
 *   shipped                                    21   ok          ok           0 failed
 *   for (wt = 0; wt < 1u; wt++)                22   FAIL x2     ok           2 failed
 *   for (wt = tile_cols-1u; wt < tile_cols;)   23   ok          FAIL x2      2 failed
 *
 * A red arm reads `PCC 0.000000000, max abs 1` with `worst |sum - 1| = 1`; a green one reads
 * `PCC 1.000000000, max abs 0`. The outlier case is a SATURATION detector, not a precision
 * one — with the spike dominating, reference and device agree bit for bit or not at all,
 * which is why it scores exactly 1 where the plain shapes score 0.99998.
 *
 * Which is why stage 3 runs BOTH and neither is redundant. The shipped LAST-only version was
 * not vacuous — it is the placement bringup log §22.2 measured red — but it was half a check
 * described as a whole one.
 */
typedef enum outlier_position {
    OUTLIER_NONE = 0,
    OUTLIER_FIRST_TILE,
    OUTLIER_LAST_TILE
} outlier_position;

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_eltwise;
static ft_tt_program_directory g_reduce;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

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

/** Fill the output bank with the poison, so a kernel that never ran is visible as such
 *  rather than as whatever the previous stage left there. */
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
 * Host-side tilize / untilize
 * ================================================================ */

/** fp32 row-major -> face-ordered bf16 tiles. */
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

/** Face-ordered bf16 tiles -> fp32 row-major. */
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

/** Round a matrix through bf16 in place: what the card will actually see. */
static void round_to_bf16(float *matrix, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++)
        matrix[index] = ft_float_from_bf16(ft_bf16_from_float(matrix[index]));
}

/* ================================================================
 * Scoring
 * ================================================================ */

/** Score one result the way the acceptance demands: BOTH metrics, and a non-finite census
 *  before either, because a summary that collapses under one bad element is a claim about
 *  the evidence rather than the evidence (dev notes §3). */
static void score(const char *label, const float *expected, const float *actual, size_t count,
                  const void *raw, size_t raw_bytes)
{
    char census[192];
    const size_t non_finite =
        ft_tt_vector_count_non_finite(expected, actual, count, census, sizeof(census));
    const size_t poisoned = poison_remaining(raw, raw_bytes);
    double pcc;
    double worst;

    ft_tt_test_check_fmt("the kernel wrote over the poison", poisoned * 100u < raw_bytes,
                         "%zu of %zu bytes still poison", poisoned, raw_bytes);
    if (non_finite != 0u) {
        ft_tt_test_check_fmt(label, false, "NON-FINITE: %s", census);
        return;
    }
    pcc = ft_tt_vector_pearson(expected, actual, count);
    worst = ft_tt_vector_max_abs_difference(expected, actual, count);
    ft_tt_test_check_fmt(label, pcc >= PCC_ACCEPT, "PCC %.9f, max abs %.6g", pcc, worst);
}

/* ================================================================
 * Bringing a resident triad up
 * ================================================================ */

/** Load all five images of one program into a core, replacing whatever is there. */
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

/**
 * Stage and KICK a resident triad: five runtime-arg blocks, the CB descriptor array, the
 * launch message, GO — and then the host walks away. Everything after this is an enqueue.
 *
 * No group block: group completion is proven in test_phase6_triad.c and a second proof
 * here would only add a way for this test to fail for a reason that is not its subject.
 */
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

/**
 * Ask the resident kernel to leave, and report whether every processor did.
 *
 * NEVER a reset and NEVER a park — see the file header. `resident_processor_mask` is
 * TRIAD_MASK and not 0 on purpose: 0 means "ask the control block", whose `state` word has
 * exactly ONE writer, so a reader that exited while the compute thread is still spinning
 * out its bound would read EXIT and be a lie (ft_tt_group.h).
 */
static void retire_triad(const ft_tt_noc_access *noc, const char *label)
{
    uint32_t unstopped = 0u;
    int rc = ft_tt_watchdog_shutdown(noc, &g_group, RUN_TIMEOUT_MS, &unstopped);

    ft_tt_test_check_fmt(label, rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=0x%X", rc,
                         unstopped);
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
        written[index] = ft_bf16_from_float(ft_tt_vector_random_value());
    memset(read_back, 0xA5, sizeof(read_back));

    rc = device_write(&session->noc, g_dram[DRAM_BANK_A], DRAM_BUFFER_BASE, written,
                      sizeof(written));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_dram[DRAM_BANK_A], DRAM_BUFFER_BASE, read_back,
                         sizeof(read_back));
    ft_tt_test_check_fmt("a tile round-trips through DRAM", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check("and it comes back byte-exact",
                     memcmp(written, read_back, sizeof(written)) == 0, "");
}

/* ================================================================
 * Stage 1/2 — the elementwise triad, six functions, ONE residency
 * ================================================================ */

/** One elementwise function on the already-resident triad. */
static bool run_one_eltwise(const ft_tt_hardware_session *session,
                            ft_tt_op_vector_function function, uint32_t slot,
                            const float *a, const float *b, float *expected, float *actual,
                            uint16_t *tiles)
{
    const size_t count = (size_t)ELTWISE_ROWS * ELTWISE_COLS;
    const size_t bytes = (size_t)ELTWISE_TILES * TILE_BYTES;
    const bool binary = ft_tt_op_vector_function_is_binary(function);
    ft_tt_op_eltwise_request request;
    ft_tt_queue_descriptor descriptor;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    char label[96];
    int rc;

    memset(&request, 0, sizeof(request));
    request.function = function;
    request.tile_count = ELTWISE_TILES;
    request.prefetch_tiles = ELTWISE_TILES;
    request.input_a.x = (uint16_t)g_dram[DRAM_BANK_A].x;
    request.input_a.y = (uint16_t)g_dram[DRAM_BANK_A].y;
    request.input_a.address = DRAM_BUFFER_BASE;
    request.input_a.tile_bytes = TILE_BYTES;
    if (binary) {
        request.input_b.x = (uint16_t)g_dram[DRAM_BANK_B].x;
        request.input_b.y = (uint16_t)g_dram[DRAM_BANK_B].y;
        request.input_b.address = DRAM_BUFFER_BASE;
        request.input_b.tile_bytes = TILE_BYTES;
    }
    request.output.x = (uint16_t)g_dram[DRAM_BANK_C].x;
    request.output.y = (uint16_t)g_dram[DRAM_BANK_C].y;
    request.output.address = DRAM_BUFFER_BASE;
    request.output.tile_bytes = TILE_BYTES;

    rc = ft_tt_op_eltwise_parameters(&request, words);
    if (rc == FT_TT_OK)
        rc = poison_output(&session->noc, bytes);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, slot, words,
                                             &parameter_address);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the descriptor is prepared", false, "%s rc=%d",
                             ft_tt_op_vector_function_name(function), rc);
        return false;
    }

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_ELTWISE;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    snprintf(label, sizeof(label), "%s completes with NO launch, only an enqueue",
             ft_tt_op_vector_function_name(function));
    ft_tt_test_check_fmt(label, rc == FT_TT_OK, "rc=%d completed=%u of %u", rc, completed,
                         slot + 1u);
    if (rc != FT_TT_OK)
        return false;

    {
        uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
        const char *reason = NULL;
        bool evidenced;

        (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
        evidenced =
            ft_tt_queue_check_completion(&g_queue, fields, slot + 1u, &reason) == FT_TT_OK;
        ft_tt_test_check_fmt("completion is evidenced by the WRITER, not by a run message",
                             evidenced, "%s: %s", ft_tt_op_vector_function_name(function),
                             reason ? reason : "ok");
    }

    rc = device_read(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, tiles, bytes);
    if (rc == FT_TT_OK)
        rc = from_bf16_tiles(tiles, ELTWISE_ROWS, ELTWISE_COLS, actual);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the output reads back and untilizes", false, "rc=%d", rc);
        return false;
    }

    ft_tt_vector_reference_eltwise(function, a, binary ? b : NULL, expected, count);
    snprintf(label, sizeof(label), "%s vs a CPU fp32 reference over %u tiles",
             ft_tt_op_vector_function_name(function), ELTWISE_TILES);
    score(label, expected, actual, count, tiles, bytes);
    return true;
}

/**
 * Fill both operand banks with full-mantissa random tiles, and leave the HOST copies
 * rounded through bf16 — because the reference must consume the values the card will see,
 * not the fp32 ones it never gets (dev notes §3).
 *
 * `tiles` is scratch: it holds A's tiles, then B's.
 */
static int upload_eltwise_operands(const ft_tt_hardware_session *session, float *a, float *b,
                                   uint16_t *tiles, size_t count, size_t bytes)
{
    size_t element;
    int rc;

    ft_tt_vector_random_seed(0xE17E15EULL);
    for (element = 0; element < count; element++) {
        a[element] = ft_tt_vector_random_value();
        b[element] = ft_tt_vector_random_value();
    }
    round_to_bf16(a, count);
    round_to_bf16(b, count);

    rc = to_bf16_tiles(a, ELTWISE_ROWS, ELTWISE_COLS, tiles);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_A], DRAM_BUFFER_BASE, tiles, bytes);
    if (rc == FT_TT_OK)
        rc = to_bf16_tiles(b, ELTWISE_ROWS, ELTWISE_COLS, tiles);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_B], DRAM_BUFFER_BASE, tiles, bytes);
    return rc;
}

static bool stage_1_eltwise(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    static const ft_tt_op_vector_function functions[] = {
        FT_TT_OP_VECTOR_FN_SILU, FT_TT_OP_VECTOR_FN_SIGMOID, FT_TT_OP_VECTOR_FN_ADD,
        FT_TT_OP_VECTOR_FN_MUL,  FT_TT_OP_VECTOR_FN_SUB,     FT_TT_OP_VECTOR_FN_SILU_MUL
    };
    const size_t count = (size_t)ELTWISE_ROWS * ELTWISE_COLS;
    const size_t bytes = (size_t)ELTWISE_TILES * TILE_BYTES;
    ft_tt_op_eltwise_request placement_request;
    float *a = malloc(count * sizeof(float));
    float *b = malloc(count * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    uint16_t *tiles = malloc(bytes);
    uint64_t mask = 0;
    unsigned index;
    bool ok = true;
    int rc;

    ft_tt_test_step("stage 1 — SIX elementwise functions on ONE resident triad, images "
                    "loaded ONCE");

    if (!a || !b || !expected || !actual || !tiles) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        ok = false;
        goto done;
    }

    rc = upload_eltwise_operands(session, a, b, tiles, count, bytes);
    ft_tt_test_check_fmt("both operands upload", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK) {
        ok = false;
        goto done;
    }

    /* The CB placement is the SILU_MUL one — the widest of the six — so every function in
     * the sweep runs against one descriptor array, which is what "one binary" means at the
     * host end too. */
    memset(&placement_request, 0, sizeof(placement_request));
    placement_request.function = FT_TT_OP_VECTOR_FN_SILU_MUL;
    placement_request.tile_count = ELTWISE_TILES;
    placement_request.prefetch_tiles = ELTWISE_TILES;
    placement_request.input_a.tile_bytes = TILE_BYTES;
    placement_request.input_b.tile_bytes = TILE_BYTES;
    placement_request.output.tile_bytes = TILE_BYTES;

    rc = load_images(&session->noc, x, y, &g_eltwise);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, x, y, &g_eltwise.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_eltwise_place_cbs(&placement_request, g_cb_placed, g_cb_buffer,
                                        sizeof(g_cb_buffer), &mask);
    if (rc == FT_TT_OK) {
        g_config.local_cb_mask = mask;
        g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
        g_config.brisc_noc_id = 0u;
        rc = ft_tt_queue_init(&session->noc, x, y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                              ft_tt_queue_make_nonce(), &g_queue);
    }
    if (rc == FT_TT_OK)
        rc = start_triad(&session->noc, x, y);
    ft_tt_test_check_fmt("the elementwise triad is staged and kicked ONCE", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc != FT_TT_OK) {
        ok = false;
        goto done;
    }

    for (index = 0; index < sizeof(functions) / sizeof(functions[0]); index++) {
        if (!run_one_eltwise(session, functions[index], index, a, b, expected, actual, tiles)) {
            ok = false;
            break;
        }
    }

    /* THE RESIDENCY CLAIM, stated as a check rather than left implicit: six descriptors
     * completed and the images crossed the link once. */
    ft_tt_test_check_fmt("six functions ran on one image load", ok,
                         "%u descriptors, 1 program directory",
                         (unsigned)(sizeof(functions) / sizeof(functions[0])));

    retire_triad(&session->noc,
                 "all five processors retire when ASKED, with no reset and no park");

done:
    free(a);
    free(b);
    free(expected);
    free(actual);
    free(tiles);
    return ok;
}

/* ================================================================
 * Stages 3-6 — the REDUCE family
 * ================================================================ */

/**
 * The maximum-absolute-error budget, as a FRACTION of the reference's own peak magnitude.
 *
 * This is the §22.2 lesson applied to this family, and it is the reason the reduce stages
 * do not score on PCC alone. **PCC is invariant under a global scale factor**, and every
 * member except softmax is scale-related: a reduce that summed the row twice, a scaler tile
 * off by two, an rsqrt where an rcp belongs — all of them multiply the whole output by a
 * constant and move a correlation by nothing at all. A relative max-abs budget sees exactly
 * that class, and 5% is chosen to sit well above bf16's own ~0.4% relative rounding and well
 * below the smallest scale error worth catching (1/sqrt(2) is 29%).
 *
 * MEASURED, four kernel mutations on Tensix 21, driven by this same unmodified binary. **PCC
 * caught none of them on the norms.** Every red below came from this check or from a
 * definitional one (the row sums, the standardised rows):
 *
 *   | mutation                          | RMS norm PCC | layer norm PCC | caught by      |
 *   |-----------------------------------|--------------|----------------|----------------|
 *   | reduce ONE tile instead of Wt     | 0.996863     | 0.992082       | max-abs (9.16) |
 *   | accumulate every tile TWICE       | 0.999995     | 0.998633       | max-abs (0.39) |
 *   | drop the eps term                 | 0.999972     | -              | max-abs (0.60) |
 *   | skip the beta chain step          | -            | 0.993583       | max-abs (0.11) |
 *
 * The second row is the one to remember: a kernel with a 29% scale error reproduced
 * softmax's three acceptance PCCs to NINE DIGITS (0.999983709 / 0.999985087 / 0.999980702,
 * bit-identical to the correct build) and would have shipped green against a PCC-only gate.
 */
#define MAX_ABS_BUDGET_FRACTION 0.05

/** Wt = 1 is where a row reduce that only ever saw one tile would pass; Wt = 48 is the
 *  model's hidden width; Wt = 128 is a 4096-wide attention row and the widest that leaves
 *  the arena comfortable. */
static const softmax_shape g_softmax_shapes[SOFTMAX_SHAPE_COUNT] = {
    { 32u, 32u, 1u },
    { 64u, 1536u, 48u },
    { 64u, 4096u, 64u }
};

/** The shape every member of the family stage runs at: MiniCPM5-1B's hidden width. */
#define FAMILY_ROWS 64u
#define FAMILY_COLS 1536u
#define FAMILY_TILE_COLS (FAMILY_COLS / TILE_DIM)
#define FAMILY_TILES ((FAMILY_ROWS / TILE_DIM) * FAMILY_TILE_COLS)
#define FAMILY_EPS 1e-5f

/** Largest |expected|, for the relative max-abs budget. */
static double peak_magnitude(const float *values, size_t count)
{
    double peak = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        if (fabs((double)values[index]) > peak)
            peak = fabs((double)values[index]);
    }
    return peak;
}

/**
 * Score a reduce result: the poison, the non-finite census, PCC — and MAX-ABS AS A CHECK.
 *
 * score() asserts PCC and merely PRINTS max-abs, which is right for an op whose errors are
 * not scale-shaped. For this family it is not: see MAX_ABS_BUDGET_FRACTION.
 */
static void score_reduce(const char *label, const float *expected, const float *actual,
                         size_t count, const void *raw, size_t raw_bytes)
{
    char census[192];
    const size_t non_finite =
        ft_tt_vector_count_non_finite(expected, actual, count, census, sizeof(census));
    const size_t poisoned = poison_remaining(raw, raw_bytes);
    const double budget = MAX_ABS_BUDGET_FRACTION * peak_magnitude(expected, count);
    double pcc;
    double worst;
    char detail[160];

    ft_tt_test_check_fmt("the kernel wrote over the poison", poisoned * 100u < raw_bytes,
                         "%zu of %zu bytes still poison", poisoned, raw_bytes);
    if (non_finite != 0u) {
        ft_tt_test_check_fmt(label, false, "NON-FINITE: %s", census);
        return;
    }
    pcc = ft_tt_vector_pearson(expected, actual, count);
    worst = ft_tt_vector_max_abs_difference(expected, actual, count);
    ft_tt_test_check_fmt(label, pcc >= PCC_ACCEPT, "PCC %.9f, max abs %.6g", pcc, worst);
    snprintf(detail, sizeof(detail), "%s: max abs %.6g of a %.4g budget (5%% of peak %.4g)",
             label, worst, budget, budget / MAX_ABS_BUDGET_FRACTION);
    ft_tt_test_check("and its SCALE is right, which PCC cannot see", worst <= budget, detail);
}

/**
 * Bring up a reduce residency: place its eleven circular buffers, arm the queue, kick once.
 *
 * The residency — not the descriptor — fixes the deepest row and whether gamma and beta are
 * carried, because circular buffers are configured at launch. A residency sized for several
 * members is what makes the merge worth having: four ops, ONE image load.
 */
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

/**
 * Upload one reduce call's operands: the activation into bank A, the constants — scaler,
 * eps, gamma, beta — into the constants bank, and a poisoned output.
 */
static int upload_reduce_operands(const ft_tt_hardware_session *session,
                                  ft_tt_op_reduce_function function, const float *x,
                                  const float *gamma, const float *beta, uint32_t rows,
                                  uint32_t cols, float eps, uint16_t *tiles)
{
    const size_t bytes = (size_t)(rows / TILE_DIM) * (cols / TILE_DIM) * TILE_BYTES;
    const size_t constant_bytes =
        ft_tt_op_reduce_constants_bytes(function, cols / TILE_DIM, TILE_BYTES);
    unsigned char *constants = malloc(constant_bytes);
    int rc;

    if (!constants)
        return FT_TT_ENOMEM;
    rc = ft_tt_op_reduce_build_constants(function, cols, eps, gamma, beta, constants,
                                         constant_bytes);
    if (rc == FT_TT_OK)
        rc = to_bf16_tiles(x, rows, cols, tiles);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_A], DRAM_BUFFER_BASE, tiles, bytes);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_CONST], DRAM_BUFFER_BASE, constants,
                          constant_bytes);
    if (rc == FT_TT_OK)
        rc = poison_output(&session->noc, bytes);
    free(constants);
    return rc;
}

/** Fill `request` for one call. Shapes and addresses only; the PROGRAM comes from the
 *  function, host-side, and is the only thing the kernel reads. */
static void fill_reduce_request(ft_tt_op_reduce_request *request,
                                ft_tt_op_reduce_function function, uint32_t rows, uint32_t cols,
                                uint32_t prefetch_tiles)
{
    memset(request, 0, sizeof(*request));
    request->function = function;
    request->row_blocks = rows / TILE_DIM;
    request->tile_cols = cols / TILE_DIM;
    request->prefetch_tiles = prefetch_tiles;
    request->input.x = (uint16_t)g_dram[DRAM_BANK_A].x;
    request->input.y = (uint16_t)g_dram[DRAM_BANK_A].y;
    request->input.address = DRAM_BUFFER_BASE;
    request->input.tile_bytes = TILE_BYTES;
    request->constants.x = (uint16_t)g_dram[DRAM_BANK_CONST].x;
    request->constants.y = (uint16_t)g_dram[DRAM_BANK_CONST].y;
    request->constants.address = DRAM_BUFFER_BASE;
    request->constants.tile_bytes = TILE_BYTES;
    request->output.x = (uint16_t)g_dram[DRAM_BANK_C].x;
    request->output.y = (uint16_t)g_dram[DRAM_BANK_C].y;
    request->output.address = DRAM_BUFFER_BASE;
    request->output.tile_bytes = TILE_BYTES;
}

/**
 * Enqueue ONE reduce descriptor on an already-resident triad and read the result back.
 *
 * `slot` is both the parameter-block slot and the expected completion count, so every call
 * after the first proves the residency: no launch, no image load, only an enqueue.
 *
 * @return false on a protocol failure; a bad SCORE is reported and still returns true, so
 *         the sweep keeps going and reports every member rather than the first bad one
 */
static bool run_one_reduce(const ft_tt_hardware_session *session,
                           const ft_tt_op_reduce_request *request, uint32_t slot, float *actual,
                           uint16_t *tiles)
{
    const uint32_t rows = request->row_blocks * TILE_DIM;
    const uint32_t cols = request->tile_cols * TILE_DIM;
    const size_t bytes = (size_t)request->row_blocks * request->tile_cols * TILE_BYTES;
    ft_tt_queue_descriptor descriptor;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    char label[128];
    int rc;

    rc = ft_tt_op_reduce_parameters(request, words);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, slot, words,
                                             &parameter_address);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the descriptor is prepared", false, "%s rc=%d",
                             ft_tt_op_reduce_function_name(request->function), rc);
        return false;
    }

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_REDUCE;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    snprintf(label, sizeof(label), "%s %ux%u completes with NO launch, only an enqueue",
             ft_tt_op_reduce_function_name(request->function), rows, cols);
    ft_tt_test_check_fmt(label, rc == FT_TT_OK, "rc=%d completed=%u of %u", rc, completed,
                         slot + 1u);
    if (rc != FT_TT_OK)
        return false;

    {
        uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
        const char *reason = NULL;

        (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
        ft_tt_test_check_fmt("completion is evidenced by the WRITER, not by a run message",
                             ft_tt_queue_check_completion(&g_queue, fields, slot + 1u,
                                                          &reason) == FT_TT_OK,
                             "%s: %s", ft_tt_op_reduce_function_name(request->function),
                             reason ? reason : "ok");
    }

    rc = device_read(&session->noc, g_dram[DRAM_BANK_C], DRAM_BUFFER_BASE, tiles, bytes);
    if (rc == FT_TT_OK)
        rc = from_bf16_tiles(tiles, rows, cols, actual);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the output reads back and untilizes", false, "rc=%d", rc);
        return false;
    }
    return true;
}

/**
 * Every row sums to 1 — the check a correlation cannot make, for softmax and the router.
 *
 * PCC is unchanged by a whole row being scaled by a constant, and "the rows are a
 * probability distribution" is the DEFINITION of both ops rather than a property of them. So
 * a reciprocal applied to the wrong row, or a broadcast that read the wrong column, is
 * visible here and nowhere else in this file.
 */
static void check_row_sums(const float *actual, uint32_t rows, uint32_t cols)
{
    double worst = 0.0;
    uint32_t row;
    char label[96];

    for (row = 0; row < rows; row++) {
        double total = 0.0;
        uint32_t column;

        for (column = 0; column < cols; column++)
            total += actual[(size_t)row * cols + column];
        if (fabs(total - 1.0) > worst)
            worst = fabs(total - 1.0);
    }
    snprintf(label, sizeof(label), "every row of %u x %u sums to 1", rows, cols);
    ft_tt_test_check_fmt(label, worst < 0.02, "worst |sum - 1| = %.6g", worst);
}

/**
 * Every row is standardised: mean 0, variance 1.
 *
 * THE definitional check for layer norm, and the one that survives its invariances. Layer
 * norm is invariant to BOTH an affine shift and a scale of its input, so with zero-mean test
 * data a kernel that reduced the mean over one tile instead of the whole row is nearly
 * invisible to PCC — the same trap softmax's shift-invariance set (bringup log §22.2). The
 * caller plants a large per-row offset so a wrong mean cannot hide, and this is what sees it.
 */
static void check_rows_are_standardised(const float *actual, uint32_t rows, uint32_t cols)
{
    double worst_mean = 0.0;
    double worst_deviation = 0.0;
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = actual + (size_t)row * cols;
        double mean = 0.0;
        double variance = 0.0;
        uint32_t column;

        for (column = 0; column < cols; column++)
            mean += (double)source[column];
        mean /= (double)cols;
        for (column = 0; column < cols; column++)
            variance += ((double)source[column] - mean) * ((double)source[column] - mean);
        variance /= (double)cols;
        if (fabs(mean) > worst_mean)
            worst_mean = fabs(mean);
        if (fabs(sqrt(variance) - 1.0) > worst_deviation)
            worst_deviation = fabs(sqrt(variance) - 1.0);
    }
    ft_tt_test_check_fmt("every layer-normed row has mean 0", worst_mean < 0.02,
                         "worst |mean| = %.6g", worst_mean);
    ft_tt_test_check_fmt("and standard deviation 1", worst_deviation < 0.02,
                         "worst |sd - 1| = %.6g", worst_deviation);
}

/* ---------------------------------------------------------------- */
/* Stage 3 — softmax, three widths, on the merged binary             */
/* ---------------------------------------------------------------- */

static bool run_one_softmax(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                            const softmax_shape *shape, outlier_position outlier)
{
    const uint32_t tile_cols = shape->cols / TILE_DIM;
    const size_t count = (size_t)shape->rows * shape->cols;
    const size_t bytes = (size_t)(shape->rows / TILE_DIM) * tile_cols * TILE_BYTES;
    ft_tt_op_reduce_residency residency;
    ft_tt_op_reduce_request request;
    ft_tt_op_reduce_function function = FT_TT_OP_REDUCE_FN_SOFTMAX;
    float *input = malloc(count * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    uint16_t *tiles = malloc(bytes);
    size_t element;
    char label[128];
    bool ok = false;
    int rc;

    if (!input || !expected || !actual || !tiles) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    ft_tt_vector_random_seed(0x50F7ULL + tile_cols);
    for (element = 0; element < count; element++)
        input[element] = ft_tt_vector_random_value();
    if (outlier != OUTLIER_NONE) {
        /* THE SHIFT INVARIANCE, MADE VISIBLE. softmax(x - c) = softmax(x) for any c, so a
         * wrong row maximum is structurally invisible to correlation on well-scaled data —
         * one of seven off-hardware mutations came back green for exactly this reason
         * (bringup log §22.2). A +200 outside the tiles a partial max reduced pushes it out
         * of representable range: it must produce non-finite output, not a slightly worse
         * score. The COLUMN is the test — see `outlier_position`. */
        const uint32_t column = (outlier == OUTLIER_LAST_TILE) ? shape->cols - 1u : 0u;
        uint32_t row;

        for (row = 0; row < shape->rows; row++)
            input[(size_t)row * shape->cols + column] = 200.0f;
    }
    round_to_bf16(input, count);

    memset(&residency, 0, sizeof(residency));
    rc = ft_tt_op_reduce_residency_for(&function, 1u, tile_cols, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = upload_reduce_operands(session, function, input, NULL, NULL, shape->rows,
                                    shape->cols, 0.0f, tiles);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(session, x, y, &residency);
    snprintf(label, sizeof(label), "the reduce triad starts at Wt = %u%s", tile_cols,
             outlier == OUTLIER_LAST_TILE  ? " with a +200 outlier in the LAST tile"
             : outlier == OUTLIER_FIRST_TILE ? " with a +200 outlier in the FIRST tile"
                                             : "");
    ft_tt_test_check_fmt(label, rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto done;

    fill_reduce_request(&request, function, shape->rows, shape->cols, shape->prefetch_tiles);
    if (!run_one_reduce(session, &request, 0u, actual, tiles)) {
        retire_triad(&session->noc, "the core retires after the failure");
        goto done;
    }

    ft_tt_vector_reference_softmax(input, expected, shape->rows, shape->cols);
    snprintf(label, sizeof(label), "softmax %u x %u (Wt = %u)%s vs a CPU fp32 reference",
             shape->rows, shape->cols, tile_cols,
             outlier == OUTLIER_LAST_TILE    ? " + outlier in the LAST tile"
             : outlier == OUTLIER_FIRST_TILE ? " + outlier in the FIRST tile"
                                             : "");
    score(label, expected, actual, count, tiles, bytes);
    check_row_sums(actual, shape->rows, shape->cols);

    retire_triad(&session->noc, "all five processors retire when ASKED");
    ok = true;

done:
    free(input);
    free(expected);
    free(actual);
    free(tiles);
    return ok;
}

static bool stage_3_softmax(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    unsigned index;

    ft_tt_test_step("stage 3 — softmax at Wt = 1, 48 and 128, now on the MERGED reduce "
                    "binary: the numbers must not move");

    for (index = 0; index < SOFTMAX_SHAPE_COUNT; index++) {
        if (!run_one_softmax(session, x, y, &g_softmax_shapes[index], OUTLIER_NONE))
            return false;
    }
    /* Then twice with the outlier — the shift-invariance test made explicit, at BOTH ends of
     * the row. One placement is half a check: see `outlier_position` for the two mutations
     * and which placement each of them is green against. */
    if (!run_one_softmax(session, x, y, &g_softmax_shapes[1], OUTLIER_LAST_TILE))
        return false;
    return run_one_softmax(session, x, y, &g_softmax_shapes[1], OUTLIER_FIRST_TILE);
}

/* ---------------------------------------------------------------- */
/* Stage 5 — all four members, ONE residency, ONE image load          */
/* ---------------------------------------------------------------- */

/**
 * test_ops.c's generator, reproduced bit for bit.
 *
 * The acceptance for the merged RMS norm is that it matches what the Phase 4 one-shot
 * scored ON THE SAME INPUT, and "the same input" has to mean the same bits — a different
 * random stream would make the two PCCs incomparable to the digit they need to be compared
 * at. So this is test_ops.c's `random_value()`, not ft_tt_vector_random_value(), and it is
 * duplicated deliberately rather than shared: the Phase 4 suite is not this suite's to
 * change, and a generator that drifted would silently stop reproducing the baseline.
 */
static uint64_t g_phase4_random_state = 1u;

static void phase4_random_seed(uint64_t seed)
{
    g_phase4_random_state = seed ? seed : 1u;
}

static float phase4_random_value(void)
{
    g_phase4_random_state ^= g_phase4_random_state >> 12;
    g_phase4_random_state ^= g_phase4_random_state << 25;
    g_phase4_random_state ^= g_phase4_random_state >> 27;
    return (float)((double)(g_phase4_random_state * 0x2545F4914F6CDD1Dull >> 11) /
                       (double)(1ull << 53) * 2.0 -
                   1.0);
}

/** The Phase 4 rms_norm operands: seed 20260812, x in [-4, 4), gamma in [0.25, 0.75). */
static void build_phase4_rms_operands(float *x, float *gamma, uint32_t rows, uint32_t cols)
{
    size_t index;

    phase4_random_seed(20260812u);
    for (index = 0; index < (size_t)rows * cols; index++)
        x[index] = phase4_random_value() * 4.0f;
    for (index = 0; index < cols; index++)
        gamma[index] = 0.5f + phase4_random_value() * 0.25f;
    round_to_bf16(x, (size_t)rows * cols);
    round_to_bf16(gamma, cols);
}

/**
 * All four members of the family on ONE residency, with the images loaded ONCE.
 *
 * THIS IS THE MERGE, STATED AS A CHECK. Four ops that were two binaries (softmax's resident
 * triad and RMS norm's Phase 4 one-shot) and would have been four run on one core with no
 * launch and no image upload between them — which is the whole reason the family exists,
 * because a transformer block norms, softmaxes and norms again in sequence and each swap
 * would otherwise cost ~0.19 ms of image upload over the x1 link (bringup log §20.4).
 */
static bool stage_5_family(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    static const ft_tt_op_reduce_function members[] = {
        FT_TT_OP_REDUCE_FN_SOFTMAX, FT_TT_OP_REDUCE_FN_RMS_NORM,
        FT_TT_OP_REDUCE_FN_LAYER_NORM, FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE
    };
    const size_t count = (size_t)FAMILY_ROWS * FAMILY_COLS;
    const size_t bytes = (size_t)FAMILY_TILES * TILE_BYTES;
    ft_tt_op_reduce_residency residency;
    ft_tt_op_reduce_request request;
    float *input = malloc(count * sizeof(float));
    float *gamma = malloc(FAMILY_COLS * sizeof(float));
    float *beta = malloc(FAMILY_COLS * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    uint16_t *tiles = malloc(bytes);
    unsigned index;
    bool ok = false;
    int rc;

    ft_tt_test_step("stage 5 — ALL FOUR members on ONE residency, images loaded ONCE");

    if (!input || !gamma || !beta || !expected || !actual || !tiles) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    rc = ft_tt_op_reduce_residency_for(members, sizeof(members) / sizeof(members[0]),
                                       FAMILY_TILE_COLS, TILE_BYTES, &residency);
    ft_tt_test_check_fmt("a residency serving all four members places", rc == FT_TT_OK,
                         "Wt=%u gamma=%d beta=%d ceiling=%u", FAMILY_TILE_COLS,
                         (int)residency.carries_gamma, (int)residency.carries_beta,
                         ft_tt_op_reduce_max_tile_cols(&residency));
    if (rc != FT_TT_OK)
        goto done;

    rc = start_reduce_triad(session, x, y, &residency);
    ft_tt_test_check_fmt("the reduce triad is staged and kicked ONCE", rc == FT_TT_OK, "rc=%d",
                         rc);
    if (rc != FT_TT_OK)
        goto done;

    for (index = 0; index < sizeof(members) / sizeof(members[0]); index++) {
        const ft_tt_op_reduce_function function = members[index];
        char label[128];
        size_t element;

        /* Each member gets the operands its definition needs, and the ones that matter are
         * the two that are NOT random noise: RMS norm reproduces Phase 4's exact input so
         * its score is comparable to the one-shot's, and the router gets POSITIVE gates,
         * because x/sum(x) on signed noise is a division by something near zero. */
        if (function == FT_TT_OP_REDUCE_FN_RMS_NORM) {
            build_phase4_rms_operands(input, gamma, FAMILY_ROWS, FAMILY_COLS);
        } else {
            ft_tt_vector_random_seed(0x4EDCE1ULL + index);
            for (element = 0; element < count; element++)
                input[element] = ft_tt_vector_random_value();
            if (function == FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE) {
                for (element = 0; element < count; element++)
                    input[element] = 0.05f + fabsf(input[element]);
            }
            for (element = 0; element < FAMILY_COLS; element++) {
                gamma[element] = 0.5f + ft_tt_vector_random_value() * 0.25f;
                beta[element] = ft_tt_vector_random_value() * 0.1f;
            }
            round_to_bf16(input, count);
            round_to_bf16(gamma, FAMILY_COLS);
            round_to_bf16(beta, FAMILY_COLS);
        }

        rc = upload_reduce_operands(session, function, input, gamma, beta, FAMILY_ROWS,
                                    FAMILY_COLS, FAMILY_EPS, tiles);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the operands upload", false, "%s rc=%d",
                                 ft_tt_op_reduce_function_name(function), rc);
            break;
        }

        fill_reduce_request(&request, function, FAMILY_ROWS, FAMILY_COLS, FAMILY_TILE_COLS);
        if (!run_one_reduce(session, &request, index, actual, tiles))
            break;

        switch (function) {
        case FT_TT_OP_REDUCE_FN_SOFTMAX:
            ft_tt_vector_reference_softmax(input, expected, FAMILY_ROWS, FAMILY_COLS);
            break;
        case FT_TT_OP_REDUCE_FN_RMS_NORM:
            ft_tt_vector_reference_rms_norm(input, gamma, FAMILY_EPS, expected, FAMILY_ROWS,
                                            FAMILY_COLS);
            break;
        case FT_TT_OP_REDUCE_FN_LAYER_NORM:
            ft_tt_vector_reference_layer_norm(input, gamma, beta, FAMILY_EPS, expected,
                                              FAMILY_ROWS, FAMILY_COLS);
            break;
        default:
            ft_tt_vector_reference_router_normalise(input, expected, FAMILY_ROWS, FAMILY_COLS);
            break;
        }
        snprintf(label, sizeof(label), "%s %ux%u vs a CPU fp32 reference",
                 ft_tt_op_reduce_function_name(function), FAMILY_ROWS, FAMILY_COLS);
        score_reduce(label, expected, actual, count, tiles, bytes);
        if (function == FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE)
            check_row_sums(actual, FAMILY_ROWS, FAMILY_COLS);
        ok = (index + 1u == sizeof(members) / sizeof(members[0]));
    }

    /* THE RESIDENCY CLAIM, as a check rather than a comment: four members, one image load. */
    ft_tt_test_check_fmt("four members ran on ONE image load and ONE CB placement", ok,
                         "%u descriptors, 1 program directory",
                         (unsigned)(sizeof(members) / sizeof(members[0])));
    retire_triad(&session->noc,
                 "all five processors retire when ASKED, with no reset and no park");

done:
    free(input);
    free(gamma);
    free(beta);
    free(expected);
    free(actual);
    free(tiles);
    return ok;
}

/* ---------------------------------------------------------------- */
/* Stage 6 — the invariances, tested EXPLICITLY                       */
/* ---------------------------------------------------------------- */

/**
 * The two cases the family's own invariances would otherwise absorb.
 *
 * Bringup log §22.2 in general form: *if the operation has an invariance, no check built on
 * that invariance can see errors inside it.* Softmax is shift-invariant, so stage 3 plants a
 * +200 outlier. This family has two more:
 *
 *  - **LAYER NORM IS AFFINE-INVARIANT**: LN(a*x + b) = LN(x). On zero-mean test data the row
 *    mean is ~0, so a kernel that reduced the mean over ONE tile computes ~0 too and every
 *    score stays green. The case below plants a large offset in ONE TILE of each row, ONCE
 *    AT EACH END — the same shape as softmax's outlier, including the reason there have to
 *    be two of them — and for a sharper reason than it looks:
 *
 *    A FLAT per-row offset does NOT work, and finding out why is worth more than the test.
 *    Measured on Tensix 20 with every row shifted by 20-34: the output rows came back with
 *    a mean of up to **0.191** instead of 0, and the run went red. That is not a bug in the
 *    kernel, it is the column tiles being **bf16**: a mean of magnitude M is packed with
 *    half an ulp of error, M * 2^-9, so centering a row whose mean is 27 and whose spread is
 *    0.58 costs 0.0625/0.58 = 0.11 of a standard deviation. Softmax has the identical
 *    limitation on its row max and is BLIND to it, because subtracting a slightly wrong
 *    constant is exactly what shift-invariance absorbs — the same invariance, hiding the
 *    same hardware fact, in the op next door.
 *
 *    And it made the check useless as well as red: a mean reduced over one tile of 48 is
 *    wrong by about sd/sqrt(32) = 0.10, the SAME order as the bf16 error, so no threshold
 *    separates the mutation from the arithmetic. With the offset in ONE tile the true mean
 *    is small (bf16 is then precise to ~4e-4 of a standard deviation) while a partial-row
 *    mean misses nearly the whole offset — a 300x margin instead of none.
 *  - **RMS NORM IS SCALE-INVARIANT WHEN eps IS NEGLIGIBLE**: at |x| ~ 1 and eps = 1e-5, eps
 *    is 1e-5 of the mean square, so a kernel that dropped the eps term entirely moves the
 *    answer by ~5e-6 and NOTHING sees it. The case below uses |x| ~ 3e-3, where the mean
 *    square is ~1e-5 and eps is half the denominator: dropping it is then a 40% error.
 */
static bool stage_6_invariances(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    static const ft_tt_op_reduce_function members[] = { FT_TT_OP_REDUCE_FN_RMS_NORM,
                                                        FT_TT_OP_REDUCE_FN_LAYER_NORM };
    const size_t count = (size_t)FAMILY_ROWS * FAMILY_COLS;
    const size_t bytes = (size_t)FAMILY_TILES * TILE_BYTES;
    /* eps big enough beside the data that dropping it is a 40% error, not a rounding one. */
    const float small_eps = 1e-5f;
    ft_tt_op_reduce_residency residency;
    ft_tt_op_reduce_request request;
    float *input = malloc(count * sizeof(float));
    float *gamma = malloc(FAMILY_COLS * sizeof(float));
    float *beta = malloc(FAMILY_COLS * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    uint16_t *tiles = malloc(bytes);
    size_t element;
    uint32_t placement;
    bool ok = false;
    int rc;

    ft_tt_test_step("stage 6 — the invariances this family would otherwise hide behind");

    if (!input || !gamma || !beta || !expected || !actual || !tiles) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    rc = ft_tt_op_reduce_residency_for(members, sizeof(members) / sizeof(members[0]),
                                       FAMILY_TILE_COLS, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(session, x, y, &residency);
    ft_tt_test_check_fmt("the reduce triad starts", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto done;

    /* --- case 1: layer norm with a +300 block in ONE tile, at BOTH ends of the row ------
     *
     * The placement is one-ended for the same reason softmax's outlier is (see
     * `outlier_position`), and for the mean the argument is even sharper: a partial row sum
     * is wrong by exactly the terms it omitted, and the offset block carries essentially the
     * whole row sum. So a mean reduced over a tile SET CONTAINING the offset reproduces the
     * true mean to four digits, while one that omits it is out by the whole offset.
     *
     * MEASURED 2026-08-14, one mutation site (`reduce_row_sum()`'s loop), each arm on a core
     * the others never touched. The cell is `check_rows_are_standardised()`'s worst |mean|:
     *
     *   build                          core   offset LAST     offset FIRST
     *   shipped                         26    0.00110944      0.0010802
     *   sum over tile 0 only            27    RED, 98.0856    0.00107129
     *   sum over the last tile only     29    0.000902494     RED, 99.2234
     *
     * Each green cell is green to the digit the shipped build prints, which is the whole
     * point: a placement cannot see a mutation that keeps the tile it was planted in. Note
     * also that PCC stayed at 0.9961 THROUGH a max-abs of 4697 — max-abs is the sensitive
     * column here (dev notes §3) and the PCC gate alone would have passed both red arms. */
    for (placement = 0; placement < 2u; placement++) {
        const bool offset_in_first_tile = (placement == 1u);
        const uint32_t offset_base = offset_in_first_tile ? 0u : FAMILY_COLS - TILE_DIM;
        uint32_t line;

        ft_tt_vector_random_seed(0x1A4E12ULL);
        for (element = 0; element < count; element++)
            input[element] = ft_tt_vector_random_value();
        for (line = 0; line < FAMILY_ROWS; line++) {
            const float offset = 300.0f + 40.0f * (float)(line / TILE_DIM);
            uint32_t column;

            for (column = offset_base; column < offset_base + TILE_DIM; column++)
                input[(size_t)line * FAMILY_COLS + column] += offset;
        }
        for (element = 0; element < FAMILY_COLS; element++) {
            gamma[element] = 1.0f;
            beta[element] = 0.0f;
        }
        round_to_bf16(input, count);

        rc = upload_reduce_operands(session, FT_TT_OP_REDUCE_FN_LAYER_NORM, input, gamma, beta,
                                    FAMILY_ROWS, FAMILY_COLS, small_eps, tiles);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the offset activation uploads", false, "rc=%d", rc);
            goto retire;
        }
        fill_reduce_request(&request, FT_TT_OP_REDUCE_FN_LAYER_NORM, FAMILY_ROWS, FAMILY_COLS,
                            FAMILY_TILE_COLS);
        if (!run_one_reduce(session, &request, 0u, actual, tiles))
            goto retire;

        ft_tt_vector_reference_layer_norm(input, gamma, beta, small_eps, expected, FAMILY_ROWS,
                                          FAMILY_COLS);
        score_reduce(offset_in_first_tile
                         ? "layer norm where the row mean lives in the FIRST tile of 48"
                         : "layer norm where the row mean lives in the LAST tile of 48",
                     expected, actual, count, tiles, bytes);
        check_rows_are_standardised(actual, FAMILY_ROWS, FAMILY_COLS);
    }

    /* --- case 2: RMS norm where eps is half the denominator ---------------------------- */
    ft_tt_vector_random_seed(0xE95ULL);
    for (element = 0; element < count; element++)
        input[element] = ft_tt_vector_random_value_scaled(0.005f);
    for (element = 0; element < FAMILY_COLS; element++)
        gamma[element] = 1.0f;
    round_to_bf16(input, count);

    rc = upload_reduce_operands(session, FT_TT_OP_REDUCE_FN_RMS_NORM, input, gamma, NULL,
                               FAMILY_ROWS, FAMILY_COLS, small_eps, tiles);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the small activation uploads", false, "rc=%d", rc);
        goto retire;
    }
    fill_reduce_request(&request, FT_TT_OP_REDUCE_FN_RMS_NORM, FAMILY_ROWS, FAMILY_COLS,
                        FAMILY_TILE_COLS);
    if (!run_one_reduce(session, &request, 1u, actual, tiles))
        goto retire;

    ft_tt_vector_reference_rms_norm(input, gamma, small_eps, expected, FAMILY_ROWS, FAMILY_COLS);
    score_reduce("RMS norm where eps is HALF the denominator, so it cannot hide", expected,
                 actual, count, tiles, bytes);
    ok = true;

retire:
    retire_triad(&session->noc, "all five processors retire when ASKED");

done:
    free(input);
    free(gamma);
    free(beta);
    free(expected);
    free(actual);
    free(tiles);
    return ok;
}

/* ================================================================
 * Stage 7 — an opcode this binary does not implement is REFUSED
 * ================================================================ */

/**
 * A binary is bound to one slab, so "the resident program does not implement this opcode"
 * is a real state and not a hypothetical. It has to be a refusal: a kernel that silently
 * skipped an opcode it does not know would advance `tail` and let the writer publish
 * completion for work nobody did — the exact lie the whole completion protocol exists to
 * make impossible (ft_tt_queue.h rule 1).
 *
 * The core must then survive it, which is the second half of the check.
 */
static void stage_7_unknown_opcode_is_refused(const ft_tt_hardware_session *session, uint16_t x,
                                              uint16_t y)
{
    const ft_tt_op_reduce_function function = FT_TT_OP_REDUCE_FN_SOFTMAX;
    ft_tt_op_reduce_residency residency;
    ft_tt_op_eltwise_request request;
    ft_tt_queue_descriptor descriptor;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    int rc;

    ft_tt_test_step("stage 7 — the reduce binary REFUSES an elementwise descriptor");

    memset(&request, 0, sizeof(request));
    request.function = FT_TT_OP_VECTOR_FN_ADD;
    request.tile_count = 1u;
    request.prefetch_tiles = 1u;
    request.input_a.x = (uint16_t)g_dram[DRAM_BANK_A].x;
    request.input_a.y = (uint16_t)g_dram[DRAM_BANK_A].y;
    request.input_a.address = DRAM_BUFFER_BASE;
    request.input_a.tile_bytes = TILE_BYTES;
    request.input_b = request.input_a;
    request.output = request.input_a;
    request.output.x = (uint16_t)g_dram[DRAM_BANK_C].x;
    request.output.y = (uint16_t)g_dram[DRAM_BANK_C].y;

    rc = ft_tt_op_reduce_residency_for(&function, 1u, 1u, TILE_BYTES, &residency);
    if (rc == FT_TT_OK)
        rc = start_reduce_triad(session, x, y, &residency);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_eltwise_parameters(&request, words);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, 0u, words,
                                             &parameter_address);
    ft_tt_test_check_fmt("the reduce triad starts", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_ELTWISE;
    descriptor.a0 = parameter_address;
    (void)ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);

    /* It must NOT complete. A short timeout is the right instrument: the expectation is a
     * refusal, so waiting the full run timeout would only make the test slow. */
    rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, 1u, 1000u, &completed);
    ft_tt_test_check_fmt("it does not retire the descriptor", rc != FT_TT_OK && completed == 0u,
                         "rc=%d completed=%u", rc, completed);

    (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
    ft_tt_test_check_fmt("and it names the reason",
                         fields[FT_TT_QUEUE_FIELD_ERROR] ==
                             (uint32_t)FT_TT_QUEUE_KERR_BAD_OPCODE,
                         "error=%u (%s)", fields[FT_TT_QUEUE_FIELD_ERROR],
                         ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));

    retire_triad(&session->noc, "the core retires by ASKING after the refusal");
}

/* ================================================================ */

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint16_t x = 0;
    uint16_t y = 0;
    uint32_t tensix_index = 0;
    uint32_t bank;
    int rc;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <eltwise-program-dir> <reduce-program-dir>\n", argv[0]);
        return 2;
    }

    ft_tt_test_step("Phase 6 vector ops — ON-CARD acceptance");

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

    rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &g_eltwise);
    if (rc == FT_TT_OK)
        rc = ft_tt_program_directory_load(argv[2], TRIAD_MASK, &g_reduce);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("both program directories load", false, "rc=%d", rc);
        goto release;
    }

    /* The core set is built by ft_tt_launch_core_set_add_tensix(), which DERIVES the
     * coordinate from the live grid and CLASSIFIES it before it can enter a launch. A
     * hand-filled placement would skip that, and a wrong NoC endpoint is a hung chip and a
     * physical power cycle, not a failed call (spec 7.3). */
    rc = ft_tt_launch_core_set_init(&g_group.cores, g_placement_storage, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_core_set_add_tensix(&g_group.cores, &session.grid, tensix_index, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the core set derives and classifies the target", false,
                             "index %u rc=%d", tensix_index, rc);
        goto release;
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
    if (stage_1_eltwise(&session, x, y) && stage_3_softmax(&session, x, y) &&
        stage_5_family(&session, x, y) && stage_6_invariances(&session, x, y))
        stage_7_unknown_opcode_is_refused(&session, x, y);

    /* DO NOT PARK — see the file header. The core is left in its firmware's GO wait, which
     * is where a healthy idle core sits. */

release_programs:
    ft_tt_program_directory_release(&g_eltwise);
    ft_tt_program_directory_release(&g_reduce);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
