/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_attention.c
 * @brief ON-CARD acceptance for FLASH-DECODE ATTENTION: PCC and max-abs against a CPU
 *   reference, the two invariance tests correlation cannot see, a byte check that the
 *   writer scattered where it was told, the residency claim, and the timing that turns all
 *   of it into tok/s.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES THE REFERENCE INDEPENDENT
 * ---------------------------------------------------------------------------
 * The same three things test_ops.c established for Phase 4 (dev notes §3): it is attention
 * from its DEFINITION and knows nothing about blocks or masks; it consumes the values the
 * CARD sees, bf16-rounded, including the query scale applied the way the kernel applies it;
 * and it is compared on UNTILIZED data, so a face-ordering mistake has something to break
 * against instead of cancelling.
 *
 * ---------------------------------------------------------------------------
 * PCC IS THE WEAK GATE HERE AND IT HAS ALREADY FAILED THIS PROJECT TWICE
 * ---------------------------------------------------------------------------
 * Attention is softmax-shaped, so it inherits shift invariance: softmax(x - c) = softmax(x)
 * makes a WRONG RUNNING MAX — the most likely bug in the kernel — structurally invisible to
 * correlation. That exact mutation came back green on seven checks in the reduce suite.
 * So this suite gates on three things PCC is not:
 *
 *  - **an outlier planted in a LATE KV block.** A block-local max then puts the early
 *    blocks' exponentials outside representable range instead of producing a
 *    shifted-but-equal answer.
 *  - **garbage poisoned into the cache past the live length.** A mask that does not mask
 *    changes the answer rather than being absorbed. Stale KV is not hypothetical — a
 *    reused cache is exactly what a server does.
 *  - **a byte comparison of the tiles the writer must NOT have touched.** PCC is blind to a
 *    consistent permutation, so where an op is a placed copy, the evidence is the bytes
 *    (§18.4: nine identical digits is evidence, an identical file is proof).
 *
 * MAX-ABS IS REPORTED BESIDE PCC ALWAYS, because different corruptions are visible to
 * different metrics: a per-element scale error moved PCC 0.99998 -> 0.929 while max-abs
 * moved 35x on the same data.
 *
 * ---------------------------------------------------------------------------
 * DO NOT PARK
 * ---------------------------------------------------------------------------
 * On a five-processor resident program, parking is the hazard — twice, on cores that had
 * retired cleanly (dev notes §1). This test retires by ASKING and leaves the core in its
 * firmware's GO wait, where a healthy idle core sits.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX  which enabled Tensix to drive (60-79 are this worker's)
 *   FT_TT_FIRMWARE_DIR  the program directory whose fw_*.elf to boot
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=../programs/attention FT_TT_TENSIX_INDEX=70 \
 *     ./build-attn-card/test_phase6_attention ../programs/attention
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
#include <time.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_coord.h"
#include "../ft_tt_op_attention.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_memory.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_attention_reference.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_DIM       FT_TT_OP_TILE_DIM
#define TILE_BYTES     FT_TT_OP_TILE_BYTES_BF16
#define TILE_ELEMENTS  (TILE_DIM * TILE_DIM)
#define QUEUE_CAPACITY 64u
#define RUN_TIMEOUT_MS 60000u
#define PCC_ACCEPT     0.99
#define POISON_BYTE    0x5A

/** MiniCPM5-1B, read from the GGUF header and not
 *  assumed: 16 Q heads, 2 KV heads, head_dim 128, 24 layers, context 131072.
 *
 *  **Overridable at COMPILE time, so one source serves two models.** Ornith is 32 Q heads /
 *  16 KV groups / head_dim 256 over 8 attention layers, i.e. a different `q_rows` AND a
 *  different `dim_tiles` — a different RESIDENCY rather than a different descriptor — so it
 *  cannot be an environment variable on a running binary the way the block depth can.
 *  `Makefile.attention`'s `ornith` target passes them; the defaults leave the MiniCPM
 *  acceptance exactly what it was. */
#ifndef MODEL_Q_HEADS
#define MODEL_Q_HEADS   16u
#endif
#ifndef MODEL_KV_HEADS
#define MODEL_KV_HEADS  2u
#endif
#ifndef MODEL_HEAD_DIM
#define MODEL_HEAD_DIM  128u
#endif
#ifndef MODEL_LAYERS
#define MODEL_LAYERS    24u
#endif
#define MODEL_DIM_TILES (MODEL_HEAD_DIM / TILE_DIM)
#define GROUP_ROWS      (MODEL_Q_HEADS / MODEL_KV_HEADS)
/** Which KV group this test drives. NOT zero: a Q base that ignored the group offset would
 *  read head 0's query and still produce a plausible answer. */
#define TEST_GROUP      1u

/** Position tiles per KV block. A RESIDENCY parameter, overridable from the environment so
 *  the depth can be swept without rebuilding — the same reason FT_TT_TENSIX_INDEX exists.
 *  It is read once, into g_block_tiles, because the placement and every descriptor have to
 *  agree about it or the kernel refuses them. */
#define BLOCK_TILES_DEFAULT 8u
#define BLOCK_TILES_ENVIRONMENT "FT_TT_ATTN_BLOCK_TILES"

/** The deepest context this suite uploads. 8192 positions is 2 MiB of K and 2 MiB of V.
 *  Overridable with the shape, because the phase whose value is at 8k+ has to be measured
 *  there and not extrapolated to it. */
#ifndef MAX_KV
#define MAX_KV       8192u
#endif
#define MAX_KV_TILES (MAX_KV / TILE_DIM)

/** Banks 2..5, at a base no other worker's suite uses — the card is shared. */
#define DRAM_BANK_Q      2u
#define DRAM_BANK_K      3u
#define DRAM_BANK_V      4u
#define DRAM_BANK_OUT    5u
#define DRAM_BANK_CONST  2u
#define DRAM_BANK_COUNT  6u
#define DRAM_BASE        0x02000000u
#define DRAM_CONST_BASE  0x02400000u

#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

#define TRIAD_MASK                                                                \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) | \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_program;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
static ft_tt_kv_cache_layout g_cache;
static uint32_t g_descriptors_done;
static uint32_t g_block_tiles = BLOCK_TILES_DEFAULT;

/* Host-side operands. Static because 8192 positions of K and V is 8 MiB of fp32 and a
 * stack frame is not where that goes. */
static float g_query[MODEL_Q_HEADS * MODEL_HEAD_DIM];
static float g_keys[MAX_KV * MODEL_HEAD_DIM];
static float g_values[MAX_KV * MODEL_HEAD_DIM];
static float g_expected[GROUP_ROWS * MODEL_HEAD_DIM];
static float g_actual[GROUP_ROWS * MODEL_HEAD_DIM];
static uint16_t g_query_tiles[MODEL_Q_HEADS * MODEL_DIM_TILES * TILE_ELEMENTS];
static uint16_t g_output_tiles[MODEL_Q_HEADS * MODEL_DIM_TILES * TILE_ELEMENTS];
static uint16_t g_region_tiles[MAX_KV_TILES * MODEL_DIM_TILES * TILE_ELEMENTS];
static uint16_t g_constants[FT_TT_OP_ATTENTION_CONST_TILES * TILE_ELEMENTS];

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

static double now_milliseconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* ================================================================
 * Layout helpers
 * ================================================================ */

/** Row-major [rows x cols] fp32 -> a face-ordered bf16 tile grid `cols/32` wide. */
static void to_tiles(const float *matrix, uint32_t rows, uint32_t cols, uint16_t *out)
{
    const uint32_t tile_cols = cols / TILE_DIM;
    uint32_t row;

    for (row = 0; row < rows; row++) {
        uint32_t column;

        for (column = 0; column < cols; column++) {
            const size_t tile = (size_t)(row / TILE_DIM) * tile_cols + column / TILE_DIM;

            out[tile * TILE_ELEMENTS +
                ft_tt_tile_face_index(row % TILE_DIM, column % TILE_DIM)] =
                ft_bf16_from_float(matrix[(size_t)row * cols + column]);
        }
    }
}

/** One row of a face-ordered tile grid -> fp32. */
static void row_from_tiles(const uint16_t *tiles, uint32_t first_tile, uint32_t tile_count,
                           uint32_t row, float *out)
{
    uint32_t tile;

    for (tile = 0; tile < tile_count; tile++) {
        uint32_t column;

        for (column = 0; column < TILE_DIM; column++)
            out[tile * TILE_DIM + column] = ft_float_from_bf16(
                tiles[(size_t)(first_tile + tile) * TILE_ELEMENTS +
                      ft_tt_tile_face_index(row, column)]);
    }
}

/**
 * Build ONE KV head's region from a row-major [positions x head_dim] array, THROUGH the
 * layout contract's own address function rather than through a second copy of it.
 *
 * `ft_tt_kv_cache_tile_offset()` is ft_tt_op_memory.h PART 1 §1.1 as code. Using it here is
 * what makes this test a check of "attention reads what KV_APPEND writes" instead of a
 * check that two independently-written layouts happen to agree.
 */
static void build_kv_region(const float *rows, uint32_t positions, uint16_t *out)
{
    uint32_t position;

    memset(out, 0, (size_t)g_cache.capacity_tiles * g_cache.dim_tiles * TILE_ELEMENTS *
                       sizeof(out[0]));
    for (position = 0; position < positions; position++) {
        uint32_t element;

        for (element = 0; element < MODEL_HEAD_DIM; element++) {
            const uint64_t offset =
                ft_tt_kv_cache_tile_offset(&g_cache, 0u, position, element / TILE_DIM);

            out[offset / sizeof(out[0]) +
                ft_tt_tile_face_index(ft_tt_op_memory_position_row(position),
                                      element % TILE_DIM)] =
                ft_bf16_from_float(rows[(size_t)position * MODEL_HEAD_DIM + element]);
        }
    }
}

/* ================================================================
 * Bringing the residency up
 * ================================================================ */

static int load_images(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    unsigned index;

    for (index = 0; index < g_program.program.image_count; index++) {
        const ft_tt_kernel_image *image = &g_program.program.images[index];
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

static int start_residency(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    ft_tt_op_attention_residency residency;
    uint64_t mask = 0;
    int rc;

    residency.q_rows = GROUP_ROWS;
    residency.dim_tiles = MODEL_DIM_TILES;
    residency.block_tiles = g_block_tiles;
    residency.tile_bytes = TILE_BYTES;

    rc = load_images(&session->noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, x, y, &g_program.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_attention_place_cbs(&residency, g_cb_placed, g_cb_buffer,
                                          sizeof(g_cb_buffer), &mask);
    if (rc != FT_TT_OK)
        return rc;

    ft_tt_test_check_fmt("the residency's L1 is flat in the context and fits the arena",
                         ft_tt_op_attention_l1_bytes(&residency) <
                             (size_t)(FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE),
                         "%zu bytes of %u", ft_tt_op_attention_l1_bytes(&residency),
                         FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE);

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
 * One call
 * ================================================================ */

/** Fill the request for one call. Shapes and addresses only. */
static void fill_request(ft_tt_op_attention_request *request, uint32_t kv_positions,
                         uint32_t query_row)
{
    memset(request, 0, sizeof(*request));
    request->q_rows = GROUP_ROWS;
    request->dim_tiles = MODEL_DIM_TILES;
    request->block_tiles = g_block_tiles;
    request->kv_positions = kv_positions;
    request->capacity_tiles = g_cache.capacity_tiles;
    request->query_row = query_row;

    request->query.x = (uint16_t)g_dram[DRAM_BANK_Q].x;
    request->query.y = (uint16_t)g_dram[DRAM_BANK_Q].y;
    /* The group's FIRST Q tile, not the buffer's — a base that ignored the group offset
     * would read head 0 and still produce a plausible answer. */
    request->query.address =
        DRAM_BASE + TEST_GROUP * GROUP_ROWS * MODEL_DIM_TILES * TILE_BYTES;
    request->query.tile_bytes = TILE_BYTES;

    request->keys.x = (uint16_t)g_dram[DRAM_BANK_K].x;
    request->keys.y = (uint16_t)g_dram[DRAM_BANK_K].y;
    request->keys.address = DRAM_BASE;
    request->keys.tile_bytes = TILE_BYTES;

    request->values.x = (uint16_t)g_dram[DRAM_BANK_V].x;
    request->values.y = (uint16_t)g_dram[DRAM_BANK_V].y;
    request->values.address = DRAM_BASE;
    request->values.tile_bytes = TILE_BYTES;

    request->constants.x = (uint16_t)g_dram[DRAM_BANK_CONST].x;
    request->constants.y = (uint16_t)g_dram[DRAM_BANK_CONST].y;
    request->constants.address = DRAM_CONST_BASE;
    request->constants.tile_bytes = TILE_BYTES;

    request->output.x = (uint16_t)g_dram[DRAM_BANK_OUT].x;
    request->output.y = (uint16_t)g_dram[DRAM_BANK_OUT].y;
    request->output.address =
        DRAM_BASE + TEST_GROUP * GROUP_ROWS * MODEL_DIM_TILES * TILE_BYTES;
    request->output.tile_bytes = TILE_BYTES;
}

/** Enqueue one descriptor and wait for it. @return FT_TT_OK, or the first failing step. */
static int run_one(const ft_tt_hardware_session *session,
                   const ft_tt_op_attention_request *request, uint32_t *out_completed)
{
    uint32_t words[FT_TT_OP_ATTENTION_WORDS];
    ft_tt_queue_descriptor descriptor;
    uint32_t parameter_address = 0u;
    int rc = ft_tt_op_attention_parameters(request, words);

    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue,
                                             g_descriptors_done % 32u, words,
                                             &parameter_address);
    if (rc != FT_TT_OK)
        return rc;

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_ATTENTION;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc != FT_TT_OK)
        return rc;
    g_descriptors_done++;
    return ft_tt_queue_wait_completed(&session->noc, &g_queue, g_descriptors_done,
                                      RUN_TIMEOUT_MS, out_completed);
}

/** Poison the whole output buffer, so a kernel that never ran scores as such. */
static int poison_output(const ft_tt_hardware_session *session)
{
    memset(g_output_tiles, POISON_BYTE, sizeof(g_output_tiles));
    return device_write(&session->noc, g_dram[DRAM_BANK_OUT], DRAM_BASE, g_output_tiles,
                        sizeof(g_output_tiles));
}

/**
 * Upload one problem: the query (whole 16-head activation), the two regions, and a poisoned
 * output. `dead_value` fills the cache PAST the live length — the stale data a real server's
 * reused cache would hold, and what the additive mask has to defeat.
 */
static int upload_problem(const ft_tt_hardware_session *session, uint32_t kv_positions,
                          uint64_t seed, float dead_value, float outlier)
{
    const uint32_t padded = ft_tt_op_attention_padded_tiles(kv_positions, g_block_tiles) * TILE_DIM;
    const size_t region_bytes = (size_t)g_cache.capacity_tiles * g_cache.dim_tiles * TILE_BYTES;
    uint32_t index;
    int rc;

    ft_tt_attention_seed(seed);
    for (index = 0; index < MODEL_Q_HEADS * MODEL_HEAD_DIM; index++)
        g_query[index] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_attention_random()));
    for (index = 0; index < padded * MODEL_HEAD_DIM; index++) {
        const bool dead = (index / MODEL_HEAD_DIM) >= kv_positions;

        g_keys[index] =
            ft_float_from_bf16(ft_bf16_from_float(dead ? dead_value : ft_tt_attention_random()));
        g_values[index] =
            ft_float_from_bf16(ft_bf16_from_float(dead ? dead_value : ft_tt_attention_random()));
    }
    /* The outlier goes in the LAST live tile — a late block — which is what makes a
     * block-local running max produce non-finite values instead of a shifted answer. */
    if (outlier != 0.0f && kv_positions > TILE_DIM) {
        const uint32_t position = kv_positions - 1u;

        for (index = 0; index < MODEL_HEAD_DIM; index++)
            g_keys[(size_t)position * MODEL_HEAD_DIM + index] =
                ft_float_from_bf16(ft_bf16_from_float(outlier));
    }

    /* The query buffer carries ALL 16 heads, and only ONE row of it is this token's.
     * Every other row is different random data, so a reader that packed the wrong row
     * cannot pass — and only the GROUP's 8 heads are attended to, so a Q base that ignored
     * the group offset cannot either. */
    rc = FT_TT_OK;
    {
        /* Query activation: [32 rows x (16 heads * 128)] — the model's real layout. */
        static float activation[TILE_DIM * MODEL_Q_HEADS * MODEL_HEAD_DIM];
        const uint32_t width = MODEL_Q_HEADS * MODEL_HEAD_DIM;
        uint32_t row;

        for (row = 0; row < TILE_DIM; row++) {
            uint32_t column;

            for (column = 0; column < width; column++)
                activation[(size_t)row * width + column] =
                    ft_float_from_bf16(ft_bf16_from_float(ft_tt_attention_random()));
        }
        /* This token's row carries the query the reference will use. */
        for (index = 0; index < width; index++)
            activation[(size_t)ft_tt_op_memory_position_row(kv_positions - 1u) * width + index] =
                g_query[index % (MODEL_Q_HEADS * MODEL_HEAD_DIM)];
        to_tiles(activation, TILE_DIM, width, g_query_tiles);
        rc = device_write(&session->noc, g_dram[DRAM_BANK_Q], DRAM_BASE, g_query_tiles,
                          sizeof(g_query_tiles));
    }

    if (rc == FT_TT_OK) {
        build_kv_region(g_keys, padded, g_region_tiles);
        rc = device_write(&session->noc, g_dram[DRAM_BANK_K], DRAM_BASE, g_region_tiles,
                          region_bytes);
    }
    if (rc == FT_TT_OK) {
        build_kv_region(g_values, padded, g_region_tiles);
        rc = device_write(&session->noc, g_dram[DRAM_BANK_V], DRAM_BASE, g_region_tiles,
                          region_bytes);
    }
    if (rc == FT_TT_OK)
        rc = poison_output(session);
    return rc;
}

/** Score one completed call: read the output back, untilize the group's rows, compare. */
static void score_call(const ft_tt_hardware_session *session, const char *label,
                       uint32_t kv_positions, uint32_t query_row)
{
    ft_tt_attention_problem problem;
    const uint32_t first_tile = TEST_GROUP * GROUP_ROWS * MODEL_DIM_TILES;
    double pcc;
    double max_abs;
    size_t untouched = 0;
    size_t index;
    uint32_t head;
    char text[192];
    int rc;

    rc = device_read(&session->noc, g_dram[DRAM_BANK_OUT], DRAM_BASE, g_output_tiles,
                     sizeof(g_output_tiles));
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the output reads back", false, "rc=%d", rc);
        return;
    }

    /* Row `query_row` of the group's tiles is the answer; every OTHER tile in the buffer
     * belongs to the other group and must still hold the poison. That is a byte check, and
     * it is the only kind that can see a writer scattering to the wrong tile — PCC cannot. */
    for (index = 0; index < MODEL_Q_HEADS * MODEL_DIM_TILES; index++) {
        const bool mine = index >= first_tile && index < first_tile + GROUP_ROWS * MODEL_DIM_TILES;
        const unsigned char *bytes =
            (const unsigned char *)(g_output_tiles + index * TILE_ELEMENTS);
        size_t byte;

        if (mine)
            continue;
        for (byte = 0; byte < TILE_BYTES; byte++) {
            if (bytes[byte] != POISON_BYTE)
                untouched++;
        }
    }
    snprintf(text, sizeof(text), "%s: the OTHER group's tiles are byte-untouched", label);
    ft_tt_test_check_fmt(text, untouched == 0u, "%zu bytes disturbed", untouched);

    for (head = 0; head < GROUP_ROWS; head++)
        row_from_tiles(g_output_tiles, first_tile + head * MODEL_DIM_TILES, MODEL_DIM_TILES,
                       query_row, g_actual + (size_t)head * MODEL_HEAD_DIM);

    problem.q_rows = GROUP_ROWS;
    problem.head_dim = MODEL_HEAD_DIM;
    problem.kv_positions = kv_positions;
    problem.query = g_query + (size_t)TEST_GROUP * GROUP_ROWS * MODEL_HEAD_DIM;
    problem.keys = g_keys;
    problem.values = g_values;
    if (!ft_tt_attention_reference(&problem, g_expected)) {
        ft_tt_test_check_fmt("the reference computes", false, "kv=%u", kv_positions);
        return;
    }

    pcc = ft_tt_attention_pcc(g_expected, g_actual, GROUP_ROWS * MODEL_HEAD_DIM);
    max_abs = ft_tt_attention_max_abs(g_expected, g_actual, GROUP_ROWS * MODEL_HEAD_DIM);
    snprintf(text, sizeof(text), "%s: PCC >= %.2f vs a CPU fp32 reference", label, PCC_ACCEPT);
    ft_tt_test_check_fmt(text, pcc >= PCC_ACCEPT, "pcc %.9f max-abs %.6f", pcc, max_abs);
    snprintf(text, sizeof(text), "%s: max-abs is small (PCC is the coarse instrument)", label);
    ft_tt_test_check_fmt(text, max_abs < 0.05, "max-abs %.6f", max_abs);
    snprintf(text, sizeof(text), "%s: neither side is non-finite", label);
    ft_tt_test_check_fmt(
        text,
        ft_tt_attention_non_finite(g_expected, GROUP_ROWS * MODEL_HEAD_DIM) == 0u &&
            ft_tt_attention_non_finite(g_actual, GROUP_ROWS * MODEL_HEAD_DIM) == 0u,
        "reference %zu card %zu",
        ft_tt_attention_non_finite(g_expected, GROUP_ROWS * MODEL_HEAD_DIM),
        ft_tt_attention_non_finite(g_actual, GROUP_ROWS * MODEL_HEAD_DIM));
    printf("      %-46s pcc %.9f  max-abs %.6f\n", label, pcc, max_abs);
}

/** One whole scored call: upload, enqueue, wait, score. */
static bool run_and_score(const ft_tt_hardware_session *session, const char *label,
                          uint32_t kv_positions, uint64_t seed, float dead_value, float outlier)
{
    ft_tt_op_attention_request request;
    const uint32_t query_row = ft_tt_op_memory_position_row(kv_positions - 1u);
    uint32_t completed = 0u;
    char text[160];
    int rc = upload_problem(session, kv_positions, seed, dead_value, outlier);

    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the problem uploads", false, "rc=%d", rc);
        return false;
    }
    fill_request(&request, kv_positions, query_row);
    rc = run_one(session, &request, &completed);
    snprintf(text, sizeof(text), "%s completes with NO launch, only an enqueue", label);
    ft_tt_test_check_fmt(text, rc == FT_TT_OK, "rc=%d completed=%u of %u", rc, completed,
                         g_descriptors_done);
    if (rc != FT_TT_OK)
        return false;

    {
        uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
        const char *reason = NULL;

        (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
        snprintf(text, sizeof(text), "%s: completion is evidenced by the WRITER", label);
        ft_tt_test_check_fmt(text,
                             ft_tt_queue_check_completion(&g_queue, fields, g_descriptors_done,
                                                          &reason) == FT_TT_OK,
                             "%s", reason ? reason : "ok");
    }
    score_call(session, label, kv_positions, query_row);
    return true;
}

/* ================================================================
 * The stages
 * ================================================================ */

static void stage_round_trip(const ft_tt_hardware_session *session)
{
    uint16_t written[TILE_ELEMENTS];
    uint16_t read_back[TILE_ELEMENTS];
    size_t index;
    int rc;

    ft_tt_test_step("stage 0 — host->DRAM->host, so the transfer path is a LINE and not an "
                    "assumption");

    ft_tt_attention_seed(1u);
    for (index = 0; index < TILE_ELEMENTS; index++)
        written[index] = ft_bf16_from_float(ft_tt_attention_random());
    memset(read_back, 0xA5, sizeof(read_back));
    rc = device_write(&session->noc, g_dram[DRAM_BANK_K], DRAM_BASE, written, sizeof(written));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_dram[DRAM_BANK_K], DRAM_BASE, read_back,
                         sizeof(read_back));
    ft_tt_test_check_fmt("a tile round-trips through DRAM", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check("and it comes back byte-exact",
                     memcmp(written, read_back, sizeof(written)) == 0, "");
}

static void stage_shapes(const ft_tt_hardware_session *session)
{
    ft_tt_test_step("stage 1 — the shapes, each scored against a CPU fp32 reference");

    (void)run_and_score(session, "kv=256, four whole blocks", 256u, 20260812ull, 0.0f, 0.0f);
    (void)run_and_score(session, "kv=1000, a partial last tile", 1000u, 20260813ull, 0.0f, 0.0f);
    (void)run_and_score(session, "kv=33, two live tiles in a padded window", 33u, 20260814ull,
                        0.0f, 0.0f);
    (void)run_and_score(session, "kv=1, one live position", 1u, 20260815ull, 0.0f, 0.0f);
    (void)run_and_score(session, "kv=2048, sixty-four blocks", 2048u, 20260816ull, 0.0f, 0.0f);
}

static void stage_invariances(const ft_tt_hardware_session *session)
{
    ft_tt_test_step("stage 2 — the two things PCC cannot see");

    /* softmax(x - c) = softmax(x): a stale running max is invisible to correlation on
     * well-scaled data, because the rescale keeps the answer algebraically right whatever
     * reference point it uses. What is NOT invisible is a max so stale that the exponential
     * leaves representable range — so the outlier has to be BIG. +200 is the size the reduce
     * suite settled on for the same reason: a kernel that
     * kept an early block's max computes exp(+200) here, which is inf in bf16 and in fp32
     * alike, while a correct one computes exp(0) = 1.
     *
     * MEASURED, and it is why this number is 200 and not 30: at +30 the mutation that
     * freezes the max after the first block stays GREEN on every check in this file. */
    (void)run_and_score(session, "a +200 outlier planted in a LATE KV block", 2048u,
                        20260817ull, 0.0f, 200.0f);

    /* Stale KV past the live length, which is what a reused cache holds. If the additive
     * mask is absent or applied after the exponential, this dominates the row max and the
     * answer collapses. */
    (void)run_and_score(session, "garbage poisoned past the live length", 1000u, 20260818ull,
                        60.0f, 0.0f);
}

/* ================================================================
 * stage 2b — CHUNK MODE: one Q head, C consecutive positions, one descriptor
 * ================================================================ */

/** The established split. 103 positions as 32 + 32 + 32 + 7 is the smallest shape that
 *  exercises a chunk with no prior context, two chunks with it, and a ragged chunk — and
 *  both GDN mixers recorded their boundary bug as INVISIBLE IN CHUNK 0 (spec, 2026-08-21).
 *  For attention the analogue is "attend to prior chunks", which chunk 0 cannot see at all,
 *  because at chunk 0 there is nothing before the chunk. */
#define CHUNK_POSITIONS 103u
#define CHUNK_MAX_ROWS  32u
#define CHUNK_COUNT     4u

/** NOT head 0, and not a head at the start of a KV group: a Q base that dropped the head
 *  offset, or one that kept the group offset, would still read a plausible query. */
#define CHUNK_HEAD 5u

/** Max-norm relative, the metric the chunkwise GDN gates settled on: per-element relative
 *  is meaningless for an output that crosses zero. The absolute is reported beside it.
 *
 *  1.5e-2 IS CALIBRATED AGAINST THE SHIPPED DECODE DESCRIPTOR, NOT CHOSEN. Measured on this
 *  problem, 2026-08-21: the chunk arm reads 4.6e-3 .. 7.7e-3 over the four chunks, and the
 *  DECODE descriptor — which chunk_cross_check() shows is bit-identical to the chunk's last
 *  row, max-abs 0.000e+00 — reads 3.1e-3 .. 9.5e-3 against the same reference on the same
 *  row. So this residue is the op's own bf16/Tf32 residue and not the chunk path's, and a
 *  bar below the decode arm's own figure would fail a bit-exact arm — the same mistake the
 *  1e-3 intermediate bar made on the GDN norms. The bar that actually discriminates is
 *  `chunk_relative <= decode_relative`, which is checked directly beside this one; this
 *  number exists so a gross regression still trips something. */
#define CHUNK_RELATIVE_GATE 1.5e-2
/** What a red arm has to clear. Two orders above the gate, so "red" is not "noisy". */
#define CHUNK_RED_FLOOR 1.0e-2
/** The chunk's LAST row is the one a decode descriptor would have computed, so the two must
 *  agree to about the bf16 rounding of one extra pack. */
#define CHUNK_CROSS_ABSOLUTE 5.0e-3

/** A second constants operand whose 32 causal tiles are ALL LIVE. */
#define DRAM_CONST_RED_BASE (DRAM_CONST_BASE + 0x40000u)

static float g_chunk_query[CHUNK_POSITIONS * MODEL_Q_HEADS * MODEL_HEAD_DIM];
static float g_chunk_expected[CHUNK_MAX_ROWS * MODEL_HEAD_DIM];
static float g_chunk_actual[CHUNK_MAX_ROWS * MODEL_HEAD_DIM];
static float g_chunk_head_query[CHUNK_MAX_ROWS * MODEL_HEAD_DIM];
static float g_chunk_last_row[MODEL_HEAD_DIM];
static float g_arm_last_row[MODEL_HEAD_DIM];
static float g_chunk_arm[CHUNK_MAX_ROWS * MODEL_HEAD_DIM];
static uint16_t g_constants_red[FT_TT_OP_ATTENTION_CONST_TILES * TILE_ELEMENTS];

/** Filled by chunk_score(): the max-norm relative over the whole chunk, and the max-abs of
 *  the LAST row alone — which is the row a decode descriptor computes and the row causality
 *  cannot move. */
static double g_chunk_relative;
static double g_chunk_absolute;
static double g_chunk_last_relative;
/** The SHIPPED decode descriptors' own max-norm relative over the WHOLE chunk, filled by
 *  chunk_cross_check(). The chunk arm's bar, and the only honest one. */
static double g_decode_last_relative;

/** Positions the reader actually fetches for this stage, whatever the chunk. */
static uint32_t chunk_padded_positions(void)
{
    return ft_tt_op_attention_padded_tiles(CHUNK_POSITIONS, g_block_tiles) * TILE_DIM;
}

/**
 * Fill the host operands once.
 *
 * PAST THE 103 POSITIONS THE CACHE HOLDS ORDINARY RANDOM DATA, not a bland constant. For
 * chunk `c` every position after it is a FUTURE position of this same sequence, so a causal
 * mask that leaks reads a perfectly plausible value — which is what makes the causality red
 * arm a real question rather than a check that a 1e30 shows up somewhere.
 */
static void chunk_fill_host(void)
{
    const uint32_t padded = chunk_padded_positions();
    uint32_t index;

    ft_tt_attention_seed(0xC401BEEFull);
    for (index = 0; index < padded * MODEL_HEAD_DIM; index++) {
        g_keys[index] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_attention_random()));
        g_values[index] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_attention_random()));
    }
    for (index = 0; index < CHUNK_POSITIONS * MODEL_Q_HEADS * MODEL_HEAD_DIM; index++)
        g_chunk_query[index] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_attention_random()));
}

/**
 * Zero the UPLOADED tiles of positions [0, positions).
 *
 * The host arrays are untouched, so the reference still reads the prior context this arm
 * removes. A red arm that mutated `g_keys` would change what BOTH sides compute and score
 * 0.000 — the ninth recorded vacuous-gate class, and the one that cost the delta rule a
 * round (spec, 2026-08-21).
 */
static void chunk_wipe_region(uint32_t positions)
{
    uint32_t position;

    for (position = 0; position < positions; position += TILE_DIM) {
        uint32_t dim_tile;

        for (dim_tile = 0; dim_tile < MODEL_DIM_TILES; dim_tile++) {
            const uint64_t offset = ft_tt_kv_cache_tile_offset(&g_cache, 0u, position, dim_tile);

            memset(g_region_tiles + offset / sizeof(g_region_tiles[0]), 0, TILE_BYTES);
        }
    }
}

/** Both regions, optionally with the positions before `wipe_positions` blanked on the card. */
static int chunk_upload_regions(const ft_tt_hardware_session *session, uint32_t wipe_positions)
{
    const size_t region_bytes = (size_t)g_cache.capacity_tiles * g_cache.dim_tiles * TILE_BYTES;
    int rc;

    build_kv_region(g_keys, chunk_padded_positions(), g_region_tiles);
    chunk_wipe_region(wipe_positions);
    rc = device_write(&session->noc, g_dram[DRAM_BANK_K], DRAM_BASE, g_region_tiles,
                      region_bytes);
    if (rc != FT_TT_OK)
        return rc;
    build_kv_region(g_values, chunk_padded_positions(), g_region_tiles);
    chunk_wipe_region(wipe_positions);
    return device_write(&session->noc, g_dram[DRAM_BANK_V], DRAM_BASE, g_region_tiles,
                        region_bytes);
}

/**
 * The chunk's Q activation: rows 0..rows-1 are positions `first..first+rows-1` for EVERY
 * head, rows `rows`..31 are zero.
 *
 * This is the whole geometry claim, as data: a chunk is the SAME strip a single position is,
 * with rows 1..C-1 carrying positions instead of padding. Nothing about the tile grid moves.
 */
static int chunk_upload_query(const ft_tt_hardware_session *session, uint32_t first,
                              uint32_t rows)
{
    static float activation[TILE_DIM * MODEL_Q_HEADS * MODEL_HEAD_DIM];
    const uint32_t width = MODEL_Q_HEADS * MODEL_HEAD_DIM;
    uint32_t row;

    memset(activation, 0, sizeof(activation));
    for (row = 0; row < rows; row++)
        memcpy(activation + (size_t)row * width,
               g_chunk_query + (size_t)(first + row) * width, (size_t)width * sizeof(float));
    to_tiles(activation, TILE_DIM, width, g_query_tiles);
    return device_write(&session->noc, g_dram[DRAM_BANK_Q], DRAM_BASE, g_query_tiles,
                        sizeof(g_query_tiles));
}

/** One chunk descriptor. `head` is a Q head, not a KV group — that is the mode. */
static void chunk_fill_request(ft_tt_op_attention_request *request, uint32_t first,
                               uint32_t rows, uint32_t head, uint32_t constants_base)
{
    const uint32_t head_offset = head * MODEL_DIM_TILES * TILE_BYTES;

    memset(request, 0, sizeof(*request));
    request->q_rows = rows;
    request->dim_tiles = MODEL_DIM_TILES;
    request->block_tiles = g_block_tiles;
    request->kv_positions = first + rows;
    request->capacity_tiles = g_cache.capacity_tiles;
    request->query_row = 0u;
    request->flags = FT_TT_OP_ATTENTION_FLAG_CHUNK;

    request->query.x = (uint16_t)g_dram[DRAM_BANK_Q].x;
    request->query.y = (uint16_t)g_dram[DRAM_BANK_Q].y;
    request->query.address = DRAM_BASE + head_offset;
    request->query.tile_bytes = TILE_BYTES;

    request->keys.x = (uint16_t)g_dram[DRAM_BANK_K].x;
    request->keys.y = (uint16_t)g_dram[DRAM_BANK_K].y;
    request->keys.address = DRAM_BASE;
    request->keys.tile_bytes = TILE_BYTES;

    request->values.x = (uint16_t)g_dram[DRAM_BANK_V].x;
    request->values.y = (uint16_t)g_dram[DRAM_BANK_V].y;
    request->values.address = DRAM_BASE;
    request->values.tile_bytes = TILE_BYTES;

    request->constants.x = (uint16_t)g_dram[DRAM_BANK_CONST].x;
    request->constants.y = (uint16_t)g_dram[DRAM_BANK_CONST].y;
    request->constants.address = constants_base;
    request->constants.tile_bytes = TILE_BYTES;

    request->output.x = (uint16_t)g_dram[DRAM_BANK_OUT].x;
    request->output.y = (uint16_t)g_dram[DRAM_BANK_OUT].y;
    request->output.address = DRAM_BASE + head_offset;
    request->output.tile_bytes = TILE_BYTES;
}

/**
 * Read the chunk's rows back and score them against the DEFINITIONAL chunk reference.
 *
 * MAX-NORM RELATIVE, with the absolute beside it. Attention's output crosses zero, so a
 * per-element relative is 100% wherever the answer is small and says nothing about the
 * kernel (the same defect the chunkwise convolution's first gate had).
 *
 * @return true if the readback and the reference both succeeded
 */
static bool chunk_score(const ft_tt_hardware_session *session, const char *label,
                        uint32_t first, uint32_t rows, uint32_t head, bool check_poison)
{
    ft_tt_attention_problem problem;
    const uint32_t first_tile = head * MODEL_DIM_TILES;
    const uint32_t width = MODEL_Q_HEADS * MODEL_HEAD_DIM;
    double worst_absolute = 0.0;
    double scale = 0.0;
    double last_absolute = 0.0;
    double last_scale = 0.0;
    size_t untouched = 0;
    size_t index;
    uint32_t row;
    char text[224];
    int rc = device_read(&session->noc, g_dram[DRAM_BANK_OUT], DRAM_BASE, g_output_tiles,
                         sizeof(g_output_tiles));

    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the chunk output reads back", false, "rc=%d", rc);
        return false;
    }

    if (check_poison) {
        for (index = 0; index < MODEL_Q_HEADS * MODEL_DIM_TILES; index++) {
            const unsigned char *bytes =
                (const unsigned char *)(g_output_tiles + index * TILE_ELEMENTS);
            size_t byte;

            if (index >= first_tile && index < first_tile + MODEL_DIM_TILES)
                continue;
            for (byte = 0; byte < TILE_BYTES; byte++) {
                if (bytes[byte] != POISON_BYTE)
                    untouched++;
            }
        }
        snprintf(text, sizeof(text), "%s: every OTHER head's tiles are byte-untouched", label);
        ft_tt_test_check_fmt(text, untouched == 0u, "%zu bytes disturbed", untouched);
    }

    for (row = 0; row < rows; row++) {
        row_from_tiles(g_output_tiles, first_tile, MODEL_DIM_TILES, row,
                       g_chunk_actual + (size_t)row * MODEL_HEAD_DIM);
        memcpy(g_chunk_head_query + (size_t)row * MODEL_HEAD_DIM,
               g_chunk_query + (size_t)(first + row) * width + (size_t)head * MODEL_HEAD_DIM,
               MODEL_HEAD_DIM * sizeof(float));
    }

    problem.q_rows = rows;
    problem.head_dim = MODEL_HEAD_DIM;
    problem.kv_positions = first + rows;
    problem.query = g_chunk_head_query;
    problem.keys = g_keys;
    problem.values = g_values;
    if (!ft_tt_attention_chunk_reference(&problem, first, g_chunk_expected)) {
        ft_tt_test_check_fmt("the chunk reference computes", false, "first=%u rows=%u", first,
                             rows);
        return false;
    }

    for (row = 0; row < rows; row++) {
        uint32_t element;

        for (element = 0; element < MODEL_HEAD_DIM; element++) {
            const size_t at = (size_t)row * MODEL_HEAD_DIM + element;
            const double want = (double)g_chunk_expected[at];
            const double difference = fabs((double)g_chunk_actual[at] - want);

            if (difference > worst_absolute)
                worst_absolute = difference;
            if (fabs(want) > scale)
                scale = fabs(want);
            if (row + 1u == rows) {
                if (difference > last_absolute)
                    last_absolute = difference;
                if (fabs(want) > last_scale)
                    last_scale = fabs(want);
            }
        }
    }
    memcpy(g_chunk_last_row, g_chunk_actual + (size_t)(rows - 1u) * MODEL_HEAD_DIM,
           MODEL_HEAD_DIM * sizeof(float));

    g_chunk_absolute = worst_absolute;
    g_chunk_relative = worst_absolute / (scale > 1.0e-6 ? scale : 1.0e-6);
    g_chunk_last_relative = last_absolute / (last_scale > 1.0e-6 ? last_scale : 1.0e-6);
    printf("      %-52s max-norm rel %.3e  abs %.3e  last-row rel %.3e\n", label,
           g_chunk_relative, g_chunk_absolute, g_chunk_last_relative);
    return true;
}

/** Upload the query for a chunk, enqueue its descriptor, and score it. */
static bool chunk_run(const ft_tt_hardware_session *session, const char *label, uint32_t first,
                      uint32_t rows, uint32_t head, uint32_t constants_base, bool check_poison)
{
    ft_tt_op_attention_request request;
    uint32_t completed = 0u;
    char text[192];
    int rc = chunk_upload_query(session, first, rows);

    if (rc == FT_TT_OK)
        rc = poison_output(session);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the chunk problem uploads", false, "rc=%d", rc);
        return false;
    }
    chunk_fill_request(&request, first, rows, head, constants_base);
    rc = run_one(session, &request, &completed);
    snprintf(text, sizeof(text), "%s: the chunk descriptor completes", label);
    ft_tt_test_check_fmt(text, rc == FT_TT_OK, "rc=%d completed=%u of %u", rc, completed,
                         g_descriptors_done);
    if (rc != FT_TT_OK)
        return false;
    return chunk_score(session, label, first, rows, head, check_poison);
}

/**
 * THE C DECODE DESCRIPTORS THE CHUNK REPLACES, RUN AND COMPARED ROW FOR ROW.
 *
 * THIS IS THE STRONGEST CHECK IN THE STAGE AND IT NEEDS NO REFERENCE AT ALL. Position
 * `first + r` is served by the SHIPPED decode path at query row `r` of the SAME uploaded Q
 * activation, and by the chunk path at row `r` of the same output tiles. The two go through
 * different readers, a different mask and different writers — a face-row pack and scatter
 * against two whole-tile transfers — and they must produce the same numbers.
 *
 * Every decode descriptor writes ROW r of the same tiles, so C of them leave the buffer
 * holding the whole chunk and ONE readback recovers it.
 *
 * It also gives the chunk arm its only honest bar. A definitional fp32 reference measures
 * this op's bf16 KV, its Tf32 unpack and its online rescale — none of which chunk mode
 * touches — so "is the chunk arm accurate" is really "is it as accurate as the descriptor it
 * replaces", and that is a number, not a constant.
 */
static bool chunk_cross_check(const ft_tt_hardware_session *session, uint32_t first,
                              uint32_t rows)
{
    const uint32_t group = CHUNK_HEAD / GROUP_ROWS;
    const uint32_t group_offset = group * GROUP_ROWS * MODEL_DIM_TILES * TILE_BYTES;
    double worst_card = 0.0;
    double worst_absolute = 0.0;
    double scale = 0.0;
    uint32_t row;
    int rc;

    memcpy(g_chunk_arm, g_chunk_actual, (size_t)rows * MODEL_HEAD_DIM * sizeof(float));

    rc = poison_output(session);
    if (rc != FT_TT_OK)
        return false;

    for (row = 0; row < rows; row++) {
        ft_tt_op_attention_request request;
        uint32_t completed = 0u;

        fill_request(&request, first + row + 1u, row);
        /* fill_request points at TEST_GROUP; this call is about CHUNK_HEAD's group. */
        request.query.address = DRAM_BASE + group_offset;
        request.output.address = DRAM_BASE + group_offset;
        if (run_one(session, &request, &completed) != FT_TT_OK) {
            ft_tt_test_check_fmt("the decode descriptors for the chunk's positions complete",
                                 false, "position %u", first + row);
            return false;
        }
    }
    rc = device_read(&session->noc, g_dram[DRAM_BANK_OUT], DRAM_BASE, g_output_tiles,
                     sizeof(g_output_tiles));
    if (rc != FT_TT_OK)
        return false;

    for (row = 0; row < rows; row++) {
        uint32_t element;

        row_from_tiles(g_output_tiles, CHUNK_HEAD * MODEL_DIM_TILES, MODEL_DIM_TILES, row,
                       g_chunk_actual + (size_t)row * MODEL_HEAD_DIM);
        for (element = 0; element < MODEL_HEAD_DIM; element++) {
            const size_t at = (size_t)row * MODEL_HEAD_DIM + element;
            const double want = (double)g_chunk_expected[at];
            const double card = fabs((double)g_chunk_actual[at] - (double)g_chunk_arm[at]);
            const double against = fabs((double)g_chunk_actual[at] - want);

            if (card > worst_card)
                worst_card = card;
            if (against > worst_absolute)
                worst_absolute = against;
            if (fabs(want) > scale)
                scale = fabs(want);
        }
    }
    g_decode_last_relative = worst_absolute / (scale > 1.0e-6 ? scale : 1.0e-6);

    ft_tt_test_check_fmt("CARD vs CARD: ONE chunk descriptor IS the decode descriptors it "
                         "replaces, row for row",
                         worst_card < CHUNK_CROSS_ABSOLUTE, "rows %u max-abs %.3e (bar %.1e)",
                         rows, worst_card, CHUNK_CROSS_ABSOLUTE);
    printf("      %-52s max-abs %.3e   (decode\'s own rel vs the reference %.3e)\n",
           "chunk vs the C decode descriptors it replaces", worst_card,
           g_decode_last_relative);
    /* Leave the chunk arm's rows where the caller found them. */
    memcpy(g_chunk_actual, g_chunk_arm, (size_t)rows * MODEL_HEAD_DIM * sizeof(float));
    return true;
}

static void stage_chunk(const ft_tt_hardware_session *session)
{
    static const uint32_t k_first[CHUNK_COUNT] = { 0u, 32u, 64u, 96u };
    static const uint32_t k_rows[CHUNK_COUNT] = { 32u, 32u, 32u, 7u };
    double arm_relative[CHUNK_COUNT];
    double decode_relative[CHUNK_COUNT];
    double causal_relative[CHUNK_COUNT];
    double causal_last_delta[CHUNK_COUNT];
    double prior_relative[CHUNK_COUNT];
    uint32_t chunk;
    int rc;

    ft_tt_test_step("stage 2b — CHUNK MODE: %u positions as 32+32+32+7, one Q head a "
                    "descriptor", CHUNK_POSITIONS);

    chunk_fill_host();
    rc = chunk_upload_regions(session, 0u);
    ft_tt_test_check_fmt("the chunk stage's K and V regions upload", rc == FT_TT_OK, "rc=%d",
                         rc);
    if (rc != FT_TT_OK)
        return;

    /* The RED constants: identical to the shipped ones except that all 32 causal tiles are
     * ALL LIVE. A descriptor pointed at them is causality switched off and NOTHING else —
     * same images, same K, same V, same Q, one address word different. */
    memcpy(g_constants_red, g_constants, sizeof(g_constants_red));
    memset(g_constants_red + (size_t)FT_TT_OP_ATTENTION_CONST_CAUSAL_TILE * TILE_ELEMENTS, 0,
           (size_t)TILE_DIM * TILE_ELEMENTS * sizeof(g_constants_red[0]));
    rc = device_write(&session->noc, g_dram[DRAM_BANK_CONST], DRAM_CONST_RED_BASE,
                      g_constants_red, sizeof(g_constants_red));
    ft_tt_test_check_fmt("the all-live causal constants upload beside the real ones",
                         rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    for (chunk = 0; chunk < CHUNK_COUNT; chunk++) {
        char label[96];

        snprintf(label, sizeof(label), "chunk %u  positions %u..%u  head %u", chunk,
                 k_first[chunk], k_first[chunk] + k_rows[chunk] - 1u, CHUNK_HEAD);
        if (!chunk_run(session, label, k_first[chunk], k_rows[chunk], CHUNK_HEAD,
                       DRAM_CONST_BASE, true))
            return;
        arm_relative[chunk] = g_chunk_relative;
        memcpy(g_arm_last_row, g_chunk_last_row, sizeof(g_arm_last_row));
        ft_tt_test_check_fmt(label, g_chunk_relative <= CHUNK_RELATIVE_GATE,
                             "max-norm relative %.3e (gate %.1e), abs %.3e", g_chunk_relative,
                             CHUNK_RELATIVE_GATE, g_chunk_absolute);

        g_decode_last_relative = 0.0;
        (void)chunk_cross_check(session, k_first[chunk], k_rows[chunk]);
        decode_relative[chunk] = g_decode_last_relative;

        /* RED 1 — CAUSALITY OFF. */
        snprintf(label, sizeof(label), "RED causal_off  chunk %u", chunk);
        if (!chunk_run(session, label, k_first[chunk], k_rows[chunk], CHUNK_HEAD,
                       DRAM_CONST_RED_BASE, false))
            return;
        causal_relative[chunk] = g_chunk_relative;
        {
            double worst = 0.0;
            uint32_t element;

            for (element = 0; element < MODEL_HEAD_DIM; element++) {
                const double difference =
                    fabs((double)g_chunk_last_row[element] - (double)g_arm_last_row[element]);

                if (difference > worst)
                    worst = difference;
            }
            causal_last_delta[chunk] = worst;
        }

        /* RED 2 — PRIOR CONTEXT WIPED ON THE CARD ONLY. */
        rc = chunk_upload_regions(session, k_first[chunk]);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the wiped regions upload", false, "rc=%d", rc);
            return;
        }
        snprintf(label, sizeof(label), "RED prior_wipe  chunk %u", chunk);
        if (!chunk_run(session, label, k_first[chunk], k_rows[chunk], CHUNK_HEAD,
                       DRAM_CONST_BASE, false))
            return;
        prior_relative[chunk] = g_chunk_relative;
        rc = chunk_upload_regions(session, 0u);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the regions restore", false, "rc=%d", rc);
            return;
        }
    }

    /* RED 3 — the head offset. Chunk 1's descriptor pointed at head 0's query, everything
     * else identical: one address word, and the answer must move. */
    {
        ft_tt_op_attention_request request;
        uint32_t completed = 0u;

        if (chunk_upload_query(session, k_first[1], k_rows[1]) == FT_TT_OK &&
            poison_output(session) == FT_TT_OK) {
            chunk_fill_request(&request, k_first[1], k_rows[1], CHUNK_HEAD, DRAM_CONST_BASE);
            request.query.address = DRAM_BASE; /* head 0, not CHUNK_HEAD */
            if (run_one(session, &request, &completed) == FT_TT_OK &&
                chunk_score(session, "RED wrong_head  chunk 1 reads head 0's Q", k_first[1],
                            k_rows[1], CHUNK_HEAD, false))
                ft_tt_test_check_fmt("RED wrong_head: a chunk reading head 0's query goes red",
                                     g_chunk_relative > CHUNK_RED_FLOOR,
                                     "relative %.3e (floor %.1e)", g_chunk_relative,
                                     CHUNK_RED_FLOOR);
        }
    }

    /* ---- The findings this stage exists for --------------------------------------- */

    for (chunk = 0; chunk < CHUNK_COUNT; chunk++) {
        char label[192];

        /* CAUSALITY. Every chunk here is at least 7 rows, so every one of them has a future
         * to leak. */
        snprintf(label, sizeof(label),
                 "RED causal_off is RED for chunk %u — a chunk attending to its own future",
                 chunk);
        ft_tt_test_check_fmt(label, causal_relative[chunk] > CHUNK_RED_FLOOR,
                             "arm %.3e -> red %.3e (floor %.1e)", arm_relative[chunk],
                             causal_relative[chunk], CHUNK_RED_FLOOR);

        /* ...AND ON A FULL CHUNK IT LEAVES THE LAST ROW ALONE. The last position of a full
         * chunk attends to the whole chunk, so causality cannot move its answer — which is
         * exactly why a gate scoring only the last position would call this bug green.
         *
         * ON THE RAGGED CHUNK IT MOVES THE LAST ROW TOO, and that is a second finding rather
         * than an exception: the causal tile also does the TAIL mask's job (`c < C`), so at
         * C = 7 switching it off lets even the last row read 25 positions the sequence does
         * not have. */
        if (k_rows[chunk] == CHUNK_MAX_ROWS) {
            snprintf(label, sizeof(label),
                     "RED causal_off leaves FULL chunk %u's LAST row BIT-UNCHANGED — a "
                     "last-position gate cannot see it at all", chunk);
            ft_tt_test_check_fmt(label, causal_last_delta[chunk] == 0.0,
                                 "arm vs red on the last row, max-abs %.3e",
                                 causal_last_delta[chunk]);
        } else {
            snprintf(label, sizeof(label),
                     "RED causal_off moves the RAGGED chunk %u's last row too — the causal "
                     "tile is also the tail mask", chunk);
            ft_tt_test_check_fmt(label, causal_last_delta[chunk] > 0.0,
                                 "arm vs red on the last row, max-abs %.3e",
                                 causal_last_delta[chunk]);
        }

        /* THE BAR THAT ACTUALLY DISCRIMINATES. Whatever the reference residue is, the chunk
         * arm may not be worse than the SHIPPED decode descriptor measured on the same row
         * of the same problem — a comparison between two things that ran on this card, with
         * the definitional reference used only to give both of them a common yardstick. */
        snprintf(label, sizeof(label),
                 "chunk %u is no worse against the reference than the DECODE descriptor is",
                 chunk);
        ft_tt_test_check_fmt(label,
                             arm_relative[chunk] <= decode_relative[chunk] * 1.05 + 1.0e-9,
                             "chunk %.3e vs decode %.3e", arm_relative[chunk],
                             decode_relative[chunk]);

        /* THE BOUNDARY. The arm itself is the positive evidence — chunks 1..3 score inside
         * the gate against a reference that includes every prior position, which they could
         * not do without reading them. prior_wipe is what proves that check CAN fail, and it
         * is INVISIBLE IN CHUNK 0 for the same structural reason both GDN mixers recorded:
         * at chunk 0 there is nothing before the chunk, so the path is not exercised at all
         * and a chunk-0-only gate is blind to it. */
        if (chunk == 0u) {
            ft_tt_test_check_fmt("RED prior_wipe is INVISIBLE in chunk 0 — nothing precedes "
                                 "it, so a chunk-0-only gate cannot see this path",
                                 prior_relative[0] <= CHUNK_RELATIVE_GATE,
                                 "relative %.3e (gate %.1e)", prior_relative[0],
                                 CHUNK_RELATIVE_GATE);
        } else {
            snprintf(label, sizeof(label),
                     "RED prior_wipe is RED for chunk %u — the chunk DOES attend to what "
                     "came before it", chunk);
            ft_tt_test_check_fmt(label, prior_relative[chunk] > CHUNK_RED_FLOOR,
                                 "arm %.3e -> red %.3e (floor %.1e)", arm_relative[chunk],
                                 prior_relative[chunk], CHUNK_RED_FLOOR);
        }
    }
}

/** How many contexts the timing stage will sweep, and how it is told which. */
#define TIMING_CONTEXTS_MAX 16u
#define CONTEXTS_ENVIRONMENT "FT_TT_ATTN_CONTEXTS"
#define REPEATS_ENVIRONMENT  "FT_TT_ATTN_REPEATS"

/**
 * What a chunk descriptor COSTS, against a decode descriptor at the same context.
 *
 * THE RATIO IS THE WHOLE PRICE OF THE RUNG, and it is an arithmetic that has to be measured
 * rather than asserted:
 *
 *   decode  — `kv_heads` descriptors serve ONE position, so a position costs
 *             `kv_heads * t_decode`;
 *   chunk   — `q_heads` descriptors serve THIRTY-TWO, so a position costs
 *             `q_heads * t_chunk / 32`.
 *
 * The claim being tested is that `t_chunk ~= t_decode`: a decode descriptor's score matmul
 * already computes all 32 rows of its tile and throws 32 - q_rows of them away, so filling
 * those rows with positions is free. If that holds, a position costs
 * `32 * kv_heads / q_heads` times less — 8x at Ornith's 16/4 — and the chunk arm ALSO reads
 * the KV cache 8x less often, because 16 walks replace 128.
 *
 * A chunk descriptor reads the cache once per Q HEAD where decode reads it once per KV
 * GROUP, so if this op ever becomes reader bound rather than compute bound the ratio falls
 * towards `32 / q_rows`. That is why it is swept over context and not taken at one point.
 */
static void stage_chunk_timing(const ft_tt_hardware_session *session)
{
    static const uint32_t k_context[] = { 1024u, 8192u };
    const char *repeats_text = getenv(REPEATS_ENVIRONMENT);
    const uint32_t repeats = repeats_text ? (uint32_t)strtoul(repeats_text, NULL, 0) : 8u;
    size_t index;

    ft_tt_test_step("stage 2c — what a chunk descriptor costs against a decode one");
    if (repeats == 0u)
        return;

    for (index = 0; index < sizeof(k_context) / sizeof(k_context[0]); index++) {
        const uint32_t context = k_context[index];
        ft_tt_op_attention_request request;
        uint32_t completed = 0u;
        double chunk_ms;
        double decode_ms;
        double started;
        uint32_t repeat;
        double decode_position;
        double chunk_position;

        if (context > MAX_KV || context < CHUNK_MAX_ROWS)
            continue;

        chunk_fill_request(&request, context - CHUNK_MAX_ROWS, CHUNK_MAX_ROWS, CHUNK_HEAD,
                           DRAM_CONST_BASE);
        started = now_milliseconds();
        for (repeat = 0; repeat < repeats; repeat++) {
            if (run_one(session, &request, &completed) != FT_TT_OK) {
                ft_tt_test_check_fmt("the timed chunk descriptor completes", false, "kv=%u",
                                     context);
                return;
            }
        }
        chunk_ms = (now_milliseconds() - started) / (double)repeats;

        fill_request(&request, context, CHUNK_MAX_ROWS - 1u);
        started = now_milliseconds();
        for (repeat = 0; repeat < repeats; repeat++) {
            if (run_one(session, &request, &completed) != FT_TT_OK) {
                ft_tt_test_check_fmt("the timed decode descriptor completes", false, "kv=%u",
                                     context);
                return;
            }
        }
        decode_ms = (now_milliseconds() - started) / (double)repeats;

        decode_position = decode_ms * (double)MODEL_KV_HEADS;
        chunk_position = chunk_ms * (double)MODEL_Q_HEADS / (double)CHUNK_MAX_ROWS;
        printf("      kv=%-6u chunk %.3f ms/desc  decode %.3f ms/desc  |  a POSITION: "
               "%.4f -> %.4f ms  (%.2fx)\n",
               context, chunk_ms, decode_ms, decode_position, chunk_position,
               chunk_position > 0.0 ? decode_position / chunk_position : 0.0);
        ft_tt_test_check_fmt("a chunk descriptor costs about what a decode descriptor costs "
                             "— the padded rows were free",
                             chunk_ms < decode_ms * 2.0, "kv=%u chunk %.3f ms decode %.3f ms",
                             context, chunk_ms, decode_ms);
    }
}


/**
 * Read the context ladder from the environment, defaulting to what this suite always swept.
 *
 * **The SLOPE is the finding, not any single point.** A phase whose whole value is at 8k+
 * cannot be priced from one descriptor at bench length, and a `block_tiles` big enough to
 * pad a short context does the padded work rather than the real work — so the ladder has to
 * cross the quantum in both directions. Entries above MAX_KV are dropped rather than
 * clamped, because a clamped entry would silently duplicate its neighbour.
 */
static uint32_t timing_contexts(uint32_t *out)
{
    static const uint32_t fallback[] = { 1024u, 2048u, 8192u };
    const char *text = getenv(CONTEXTS_ENVIRONMENT);
    uint32_t count = 0u;

    if (!text) {
        for (count = 0; count < sizeof(fallback) / sizeof(fallback[0]); count++)
            out[count] = fallback[count];
        return count;
    }
    while (*text && count < TIMING_CONTEXTS_MAX) {
        char *end = NULL;
        unsigned long value = strtoul(text, &end, 0);

        if (end == text)
            break;
        if (value >= TILE_DIM && value <= MAX_KV)
            out[count++] = (uint32_t)value;
        text = (*end == ',') ? end + 1 : end;
    }
    return count;
}

/** ms per descriptor at a given context, and what it projects to. */
static void stage_timing(const ft_tt_hardware_session *session)
{
    uint32_t contexts[TIMING_CONTEXTS_MAX];
    const uint32_t context_count = timing_contexts(contexts);
    const char *repeats_text = getenv(REPEATS_ENVIRONMENT);
    const uint32_t repeats = repeats_text ? (uint32_t)strtoul(repeats_text, NULL, 0) : 8u;
    double per_descriptor[TIMING_CONTEXTS_MAX];
    size_t index;

    ft_tt_test_step("stage 3 — ms per descriptor, per token, and the tok/s it projects to");
    if (context_count == 0u || repeats == 0u) {
        ft_tt_test_check_fmt("the timing ladder parses", false, "%u contexts, %u repeats",
                             context_count, repeats);
        return;
    }

    for (index = 0; index < context_count; index++) {
        ft_tt_op_attention_request request;
        uint32_t completed = 0u;
        double start;
        double elapsed;
        uint32_t repeat;
        int rc = upload_problem(session, contexts[index], 20260819ull, 0.0f, 0.0f);

        per_descriptor[index] = 0.0;
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the timing problem uploads", false, "rc=%d", rc);
            continue;
        }
        fill_request(&request, contexts[index],
                     ft_tt_op_memory_position_row(contexts[index] - 1u));

        /* One warm call outside the window, so the measurement is of steady-state
         * descriptors and not of the first one's cold circular buffers. */
        rc = run_one(session, &request, &completed);
        start = now_milliseconds();
        for (repeat = 0; repeat < repeats && rc == FT_TT_OK; repeat++)
            rc = run_one(session, &request, &completed);
        elapsed = now_milliseconds() - start;
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the timing run completes", false, "rc=%d kv=%u", rc,
                                 contexts[index]);
            continue;
        }
        per_descriptor[index] = elapsed / (double)repeats;
        ft_tt_test_check_fmt("the timing run completes", true, "kv=%u %.3f ms per descriptor",
                             contexts[index], per_descriptor[index]);
    }

    printf("\n      context   padded   ms/descriptor   us/position   ms/token (1 core, %u x "
           "%u)\n",
           MODEL_LAYERS, MODEL_KV_HEADS);
    for (index = 0; index < context_count; index++) {
        const double per_token =
            per_descriptor[index] * (double)MODEL_LAYERS * (double)MODEL_KV_HEADS;

        if (per_descriptor[index] <= 0.0)
            continue;
        printf("      %7u   %6u   %13.3f   %11.3f   %10.1f\n", contexts[index],
               ft_tt_op_attention_padded_tiles(contexts[index], g_block_tiles) * TILE_DIM,
               per_descriptor[index], 1000.0 * per_descriptor[index] / (double)contexts[index],
               per_token);
    }

    /* THE SLOPE, by least squares over every point that ran. `a + b*context` is the model
     * the whole phase is priced on: `b` is what host attention's 0.53 ms/position is
     * compared against, and `a` is the fixed cost that decides whether a short context is
     * worth moving at all. A per-point us/position figure carries the intercept and is NOT
     * the slope — quoting one as the other is how a padded window reads as fast work. */
    {
        double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, n = 0.0;
        double slope, intercept;

        for (index = 0; index < context_count; index++) {
            const double x = (double)contexts[index];
            const double y = per_descriptor[index];

            if (y <= 0.0)
                continue;
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        if (n >= 2.0 && (n * sxx - sx * sx) != 0.0) {
            slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            intercept = (sy - slope * sx) / n;
            printf("\n      FIT over %.0f points: %.4f ms + %.4f us/position a descriptor "
                   "-> %.4f ms + %.4f ms/position a TOKEN on one core (%u x %u "
                   "descriptors)\n",
                   n, intercept, 1000.0 * slope,
                   intercept * (double)MODEL_LAYERS * (double)MODEL_KV_HEADS,
                   slope * (double)MODEL_LAYERS * (double)MODEL_KV_HEADS, MODEL_LAYERS,
                   MODEL_KV_HEADS);
        }
    }
    printf("\n");
}

static void stage_unknown_opcode(const ft_tt_hardware_session *session)
{
    ft_tt_queue_descriptor descriptor;
    uint32_t completed = 0u;
    int rc;

    ft_tt_test_step("stage 4 — an opcode this binary does not implement is REFUSED, not "
                    "ignored");

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_ROPE;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_descriptors_done + 1u, 2000u,
                                        &completed);
    ft_tt_test_check_fmt("a ROPE descriptor is NOT completed by the attention triad",
                         rc != FT_TT_OK && completed == g_descriptors_done,
                         "rc=%d completed=%u of %u", rc, completed, g_descriptors_done + 1u);
    {
        uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];

        (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
        ft_tt_test_check_fmt("...and the reason it reports is a BAD OPCODE, not a timeout",
                             fields[FT_TT_QUEUE_FIELD_ERROR] == FT_TT_QUEUE_KERR_BAD_OPCODE,
                             "error=%u (%s)", fields[FT_TT_QUEUE_FIELD_ERROR],
                             ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
    }
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint16_t x = 0;
    uint16_t y = 0;
    uint32_t tensix_index = 0;
    uint32_t bank;
    uint32_t unstopped = 0u;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <attention-program-dir>\n", argv[0]);
        return 2;
    }

    {
        const char *depth = getenv(BLOCK_TILES_ENVIRONMENT);

        if (depth) {
            const unsigned long value = strtoul(depth, NULL, 0);

            if (value == 0u || value > 64u) {
                fprintf(stderr, "%s must be 1..64\n", BLOCK_TILES_ENVIRONMENT);
                return 2;
            }
            g_block_tiles = (uint32_t)value;
        }
    }

    ft_tt_test_step("Phase 6 — FLASH-DECODE ATTENTION, ON-CARD acceptance, block_tiles=%u",
                    g_block_tiles);

    rc = ft_tt_kv_cache_layout_init(1u, MODEL_HEAD_DIM, MAX_KV, TILE_BYTES, &g_cache);
    ft_tt_test_check_fmt("the KV layout contract derives (one head, 8192 positions)",
                         rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return 1;

    rc = ft_tt_op_attention_build_constants(MODEL_HEAD_DIM, TILE_BYTES, g_constants,
                                            sizeof(g_constants));
    if (rc != FT_TT_OK)
        return 1;

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

    rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &g_program);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the attention program loads", false, "rc=%d", rc);
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

    stage_round_trip(&session);

    rc = device_write(&session.noc, g_dram[DRAM_BANK_CONST], DRAM_CONST_BASE, g_constants,
                      sizeof(g_constants));
    ft_tt_test_check_fmt("the constants upload (one scaler, one scale, 32 masks)",
                         rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto release_program;

    rc = start_residency(&session, x, y);
    ft_tt_test_check_fmt("the attention triad goes resident", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto release_program;

    stage_shapes(&session);
    stage_invariances(&session);
    stage_chunk(&session);
    stage_chunk_timing(&session);
    stage_timing(&session);
    stage_unknown_opcode(&session);

    rc = ft_tt_watchdog_shutdown(&session.noc, &g_group, RUN_TIMEOUT_MS, &unstopped);
    ft_tt_test_check_fmt("every processor retires by ASKING, and none is parked",
                         rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=0x%X", rc,
                         unstopped);

    /* DO NOT PARK — the core is left in its firmware's GO wait, which is where a healthy
     * idle core sits (dev notes §1). */

release_program:
    ft_tt_program_directory_release(&g_program);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
