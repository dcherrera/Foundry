/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_memory_ops.c
 * @brief OFF-HARDWARE gate for the memory op family — EMBEDDING and KV_APPEND — and for
 *   the KV cache layout contract (ft_tt_op_memory.h PART 1).
 *
 * ---------------------------------------------------------------------------
 * A COPY OP IS WHERE A TEST LIES MOST EASILY
 * ---------------------------------------------------------------------------
 * A gather that returns the right BYTES from the WRONG ROW looks right whenever every row
 * holds similar data, and a face-order mistake is invisible to any metric computed through
 * the same wrong arithmetic. So this suite does four things
 * deliberately, and every one of them was needed to make some mutation go red:
 *
 *  1. **Every source row holds different content**, from a per-row generator whose values
 *     are full-mantissa bit patterns — never small integers and never a per-row constant.
 *  2. **Every destination is poisoned first**, and the poison is checked to SURVIVE
 *     everywhere the op was not supposed to write. That half is what proves an APPEND
 *     preserves the positions already in the cache, and no PCC-shaped check can see it.
 *  3. **The comparison is BYTE-FOR-BYTE, never a correlation.** PCC is blind to a
 *     consistent permutation, which is exactly the failure mode of a layout op (dev notes
 *     §3). The expected image is built element by element through
 *     `ft_tt_tile_face_index()` — ft_tt_tilize.c, written for the weight converter and
 *     verified against tt-metal's own tilize_utils.cpp — so it shares no code with the
 *     thing under test.
 *  4. **The functions under test are THE KERNEL'S**, not a C copy of them. The arithmetic
 *     half of kernels/resident_memory_protocol.h compiles on the host and is reached
 *     through tests/ft_tt_memory_device_shim.c. A transcription would have agreed with
 *     itself.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS CANNOT PROVE
 * ---------------------------------------------------------------------------
 * That a 32-byte DRAM write is legal, which is the fact the whole family rests on. Nothing
 * off-hardware can — "off-hardware suites cannot cover the copy path" (dev notes §3), and
 * here they cannot cover the NoC's alignment rules either. tests/test_phase6_memory.c
 * measures it on silicon, against neighbours as well as against the row.
 *
 * Environment:
 *   FT_TT_KERNEL_DIR  where the device headers live (default ../kernels)
 *
 * @status GREEN: 95 checks, 0 failures, on the Mac (clang) and on the card host (gcc). TWELVE
 *   mutations were run against it and ALL TWELVE went red — 14, 11, 10, 2, 6, 9, 15, 3, 6,
 *   2, 1 and 1 failures. SIX were in the KERNEL's own header (the face index, a run that
 *   emits one column half, a run one face row too long, the face pairing of a row-major
 *   source, a place that reads the wrong source row, and a run split that ignores the
 *   16-row face group boundary); TWO in the host layout arithmetic (the cache axis order,
 *   and a split rule that ignores the source tile row); FOUR in the model's control flow
 *   (the reader's fast path, the gather's index array, the whole-tiles flag, the
 *   zero-fill).
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_memory.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_test_report.h"

/** Where the device headers live, so the wire-constant pin reads the real file. */
#define KERNEL_DIRECTORY_ENVIRONMENT "FT_TT_KERNEL_DIR"
#define KERNEL_DIRECTORY_DEFAULT "../kernels"
#define PATH_MAX_BYTES 512u

#define TILE_BYTES FT_TT_OP_TILE_BYTES_BF16
#define TILE_DIM   FT_TT_OP_TILE_DIM

/** Two patterns, and they answer different questions. POISON is what a destination holds
 *  before the op, so "the op did not write here" and "the op wrote a zero" are different
 *  bit patterns. STALE is what a circular-buffer page holds before the reader fills it, so
 *  a missing zero-fill publishes something recognisable instead of a plausible zero. */
#define POISON_BYTE 0xA5u
#define STALE_BYTE  0xCCu

/* Declared by tests/ft_tt_memory_device_shim.c — the KERNEL's own arithmetic. */
uint32_t ft_tt_device_chunk_bytes(uint32_t tile_bytes);
uint32_t ft_tt_device_face_bytes(uint32_t tile_bytes);
uint32_t ft_tt_device_chunk_offset(uint32_t row, uint32_t half, uint32_t tile_bytes);
uint32_t ft_tt_device_group_row_bytes(uint32_t group_tiles, uint32_t tile_bytes);
void ft_tt_device_place_row_from_row_major(uint32_t *destination_block,
                                           const uint32_t *source_row, uint32_t destination_row,
                                           uint32_t group_tiles, uint32_t tile_bytes);
void ft_tt_device_place_row_from_tiles(uint32_t *destination_block, const uint32_t *source_block,
                                       uint32_t source_row, uint32_t destination_row,
                                       uint32_t group_tiles, uint32_t tile_bytes);
void ft_tt_device_zero_words(uint32_t *destination, uint32_t words);
uint32_t ft_tt_device_destination_runs(uint32_t row0, uint32_t row_count, uint32_t tile_bytes,
                                       uint32_t *offsets, uint32_t *bytes);
uint32_t ft_tt_device_max_runs(void);
int ft_tt_device_banked_destination(uint32_t flags, uint32_t packed_x, uint32_t packed_y,
                                    uint32_t group, uint32_t *x, uint32_t *y, uint32_t *slot);
int ft_tt_device_length_is_issuable(uint32_t bytes);

static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

/* ================================================================
 * Test data that cannot be confused between rows
 * ================================================================ */

/**
 * The bf16 bit pattern for logical element (row, column) of source `stream`.
 *
 * Every (stream, row, column) gets a DIFFERENT value, and the row contributes a large
 * multiplier, so "the right bytes from the wrong row" is a mismatch in every element rather
 * than in none. The pattern also avoids POISON_BYTE and STALE_BYTE in both halves, so a
 * value can never be mistaken for un-written memory.
 */
static uint16_t source_pattern(uint32_t stream, uint32_t row, uint32_t column)
{
    uint32_t mixed = (stream * 2654435761u) ^ (row * 40503u) ^ (column * 2246822519u);

    mixed ^= mixed >> 13;
    mixed = (mixed * 1103515245u) + 12345u;
    mixed >>= 9;
    mixed &= 0xFFFFu;
    if ((mixed & 0xFFu) == POISON_BYTE || (mixed & 0xFFu) == STALE_BYTE)
        mixed ^= 0x0001u;
    if (((mixed >> 8) & 0xFFu) == POISON_BYTE || ((mixed >> 8) & 0xFFu) == STALE_BYTE)
        mixed ^= 0x0100u;
    return (uint16_t)mixed;
}

/** Write the bf16 element (row, column) of a face-ordered tile grid `tiles` wide. */
static void put_tiled_element(unsigned char *base, uint32_t tiles_wide, uint32_t row,
                              uint32_t column, uint16_t value)
{
    const uint32_t tile = column / TILE_DIM;
    const uint32_t tile_row = row / TILE_DIM;
    const size_t offset = ((size_t)tile_row * tiles_wide + tile) * TILE_BYTES +
                          (size_t)ft_tt_tile_face_index(row % TILE_DIM, column % TILE_DIM) * 2u;

    (void)tiles_wide;
    memcpy(base + offset, &value, sizeof(value));
}

/* ================================================================
 * A model of the two kernels, driven by the KERNEL's own arithmetic
 * ================================================================ */

/**
 * Run one descriptor the way resident_gather_reader.c and resident_place_writer.c do,
 * over simulated device memory.
 *
 * The control flow is a transcription — that much is unavoidable off-hardware — but every
 * address it computes comes from the kernel's own inline functions through the shim, and
 * the ANSWER is scored against an expectation built out of ft_tt_tilize.c. So a bug in the
 * face arithmetic goes red here; a bug in the control flow goes red on the card.
 */
static void model_run(const uint32_t words[FT_TT_OP_MEMORY_WORDS], const unsigned char *source,
                      unsigned char *destination, const uint32_t *indices,
                      unsigned char *const *banks)
{
    const uint32_t flags = words[FT_TT_OP_MEMORY_FLAGS];
    const uint32_t row_count = words[FT_TT_OP_MEMORY_ROW_COUNT];
    const uint32_t group_count = words[FT_TT_OP_MEMORY_GROUP_COUNT];
    const uint32_t group_tiles = words[FT_TT_OP_MEMORY_GROUP_TILES];
    const uint32_t source_base = words[FT_TT_OP_MEMORY_SRC_BASE];
    const uint32_t source_row_stride = words[FT_TT_OP_MEMORY_SRC_ROW_STRIDE];
    const uint32_t source_row0 = words[FT_TT_OP_MEMORY_SRC_ROW0];
    const uint32_t source_group_stride = words[FT_TT_OP_MEMORY_SRC_GROUP_STRIDE];
    const uint32_t destination_base = words[FT_TT_OP_MEMORY_DST_BASE];
    const uint32_t destination_group_stride = words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE];
    const uint32_t destination_row0 = words[FT_TT_OP_MEMORY_DST_ROW0];
    const uint32_t row_in_tile = destination_row0 % TILE_DIM;
    const uint32_t tile_row = destination_row0 / TILE_DIM;
    const uint32_t group_bytes = group_tiles * TILE_BYTES;
    const uint32_t row_bytes = ft_tt_device_group_row_bytes(group_tiles, TILE_BYTES);
    const bool whole_tiles = (flags & (uint32_t)FT_TT_OP_MEMORY_FLAG_DST_WHOLE_TILES) != 0u;
    const bool tiled_source = (flags & (uint32_t)FT_TT_OP_MEMORY_FLAG_SRC_TILED) != 0u;
    const bool gather = (flags & (uint32_t)FT_TT_OP_MEMORY_FLAG_GATHER) != 0u;
    unsigned char *block = malloc(group_bytes);
    unsigned char *stage = malloc(group_bytes);
    uint32_t group;

    if (!block || !stage) {
        free(block);
        free(stage);
        return;
    }

    for (group = 0; group < group_count; group++) {
        const uint32_t source_group = source_base + group * source_group_stride;
        uint32_t bank_x = 0u, bank_y = 0u, slot = group;
        unsigned char *bank_memory = destination;
        uint32_t destination_group;
        uint32_t row;

        /* BANKED: the kernel's own decode picks the bank (the test's banks are indexed by x
         * and carry y = 100 + x, so a swapped byte lands nowhere sensible). */
        if ((flags & (uint32_t)FT_TT_OP_MEMORY_FLAG_DST_BANKED) != 0u) {
            if (!banks || !ft_tt_device_banked_destination(flags, words[FT_TT_OP_MEMORY_DST_X],
                                                           words[FT_TT_OP_MEMORY_DST_Y], group,
                                                           &bank_x, &bank_y, &slot) ||
                bank_y != 100u + bank_x)
                continue;
            bank_memory = banks[bank_x];
        }
        destination_group = destination_base + slot * destination_group_stride +
                            tile_row * group_bytes;

        /* A circular-buffer page is REUSED, so it starts as whatever the previous descriptor
         * left. Modelling that is what makes a missing zero-fill visible. */
        memset(block, (int)STALE_BYTE, group_bytes);
        if (whole_tiles && row_count < TILE_DIM)
            ft_tt_device_zero_words((uint32_t *)(void *)block, group_bytes / 4u);

        if (tiled_source) {
            const bool direct =
                source_row0 == row_in_tile && !(whole_tiles && row_count < TILE_DIM);

            memcpy(direct ? block : stage, source + source_group, group_bytes);
            if (!direct) {
                for (row = 0; row < row_count; row++) {
                    ft_tt_device_place_row_from_tiles((uint32_t *)(void *)block,
                                                      (const uint32_t *)(const void *)stage,
                                                      source_row0 + row, row_in_tile + row,
                                                      group_tiles, TILE_BYTES);
                }
            }
        } else {
            for (row = 0; row < row_count; row++) {
                const uint32_t index = gather ? indices[row] : source_row0 + row;

                memcpy(stage + row * row_bytes,
                       source + source_group + index * source_row_stride, row_bytes);
            }
            for (row = 0; row < row_count; row++) {
                ft_tt_device_place_row_from_row_major(
                    (uint32_t *)(void *)block,
                    (const uint32_t *)(const void *)(stage + row * row_bytes),
                    row_in_tile + row, group_tiles, TILE_BYTES);
            }
        }

        if (whole_tiles || (row_in_tile == 0u && row_count == TILE_DIM)) {
            memcpy(bank_memory + destination_group, block, group_bytes);
        } else {
            uint32_t offsets[8];
            uint32_t lengths[8];
            const uint32_t run_count =
                ft_tt_device_destination_runs(row_in_tile, row_count, TILE_BYTES, offsets,
                                              lengths);
            uint32_t tile;

            for (tile = 0; tile < group_tiles; tile++) {
                uint32_t index;

                for (index = 0; index < run_count; index++) {
                    const uint32_t offset = tile * TILE_BYTES + offsets[index];

                    memcpy(bank_memory + destination_group + offset, block + offset,
                           lengths[index]);
                }
            }
        }
    }
    free(block);
    free(stage);
}

/* ================================================================
 * Reading a constant out of a device header
 * ================================================================ */

static int read_device_constant(const char *path, const char *name, uint32_t *out_value)
{
    char line[512];
    FILE *file = fopen(path, "r");
    int found = -1;

    if (!file)
        return -1;
    while (fgets(line, sizeof(line), file)) {
        char parsed_name[128];
        char parsed_value[64];

        if (sscanf(line, " enum { %127[A-Za-z0-9_] = %63[^}] };", parsed_name,
                   parsed_value) != 2)
            continue;
        if (strcmp(parsed_name, name) != 0)
            continue;
        *out_value = (uint32_t)strtoul(parsed_value, NULL, 0);
        found = 0;
        break;
    }
    fclose(file);
    return found;
}

static void device_header_path(char *out, size_t out_bytes, const char *file)
{
    const char *directory = getenv(KERNEL_DIRECTORY_ENVIRONMENT);

    if (!directory || directory[0] == '\0')
        directory = KERNEL_DIRECTORY_DEFAULT;
    snprintf(out, out_bytes, "%s/%s", directory, file);
}

static void check_device_constant(const char *path, const char *name, uint32_t host_value)
{
    uint32_t device_value = 0;

    if (read_device_constant(path, name, &device_value) != 0) {
        ft_tt_test_check_fmt(name, false, "not declared in %s", path);
        return;
    }
    ft_tt_test_check_fmt(name, device_value == host_value, "host %u, kernel %u", host_value,
                         device_value);
}

static void test_wire_constants_agree(void)
{
    char memory_path[PATH_MAX_BYTES];
    char queue_path[PATH_MAX_BYTES];

    ft_tt_test_step("wire constants: the host enum against the kernel's own header");
    device_header_path(memory_path, sizeof(memory_path), "resident_memory_protocol.h");
    device_header_path(queue_path, sizeof(queue_path), "resident_queue_protocol.h");

    check_device_constant(memory_path, "kMemFlags", FT_TT_OP_MEMORY_FLAGS);
    check_device_constant(memory_path, "kMemRowCount", FT_TT_OP_MEMORY_ROW_COUNT);
    check_device_constant(memory_path, "kMemGroupCount", FT_TT_OP_MEMORY_GROUP_COUNT);
    check_device_constant(memory_path, "kMemGroupTiles", FT_TT_OP_MEMORY_GROUP_TILES);
    check_device_constant(memory_path, "kMemSrcX", FT_TT_OP_MEMORY_SRC_X);
    check_device_constant(memory_path, "kMemSrcY", FT_TT_OP_MEMORY_SRC_Y);
    check_device_constant(memory_path, "kMemSrcBase", FT_TT_OP_MEMORY_SRC_BASE);
    check_device_constant(memory_path, "kMemSrcRowStride", FT_TT_OP_MEMORY_SRC_ROW_STRIDE);
    check_device_constant(memory_path, "kMemSrcRow0", FT_TT_OP_MEMORY_SRC_ROW0);
    check_device_constant(memory_path, "kMemSrcGroupStride", FT_TT_OP_MEMORY_SRC_GROUP_STRIDE);
    check_device_constant(memory_path, "kMemDstX", FT_TT_OP_MEMORY_DST_X);
    check_device_constant(memory_path, "kMemDstY", FT_TT_OP_MEMORY_DST_Y);
    check_device_constant(memory_path, "kMemDstBase", FT_TT_OP_MEMORY_DST_BASE);
    check_device_constant(memory_path, "kMemDstGroupStride", FT_TT_OP_MEMORY_DST_GROUP_STRIDE);
    check_device_constant(memory_path, "kMemDstRow0", FT_TT_OP_MEMORY_DST_ROW0);
    check_device_constant(memory_path, "kMemWords", FT_TT_OP_MEMORY_WORDS);
    check_device_constant(memory_path, "kMemMaxRows", FT_TT_OP_MEMORY_MAX_ROWS);
    check_device_constant(memory_path, "kMemFlagSrcTiled", FT_TT_OP_MEMORY_FLAG_SRC_TILED);
    check_device_constant(memory_path, "kMemFlagDstWholeTiles",
                          FT_TT_OP_MEMORY_FLAG_DST_WHOLE_TILES);
    check_device_constant(memory_path, "kMemFlagGather", FT_TT_OP_MEMORY_FLAG_GATHER);
    check_device_constant(memory_path, "kMemFlagDstBanked", FT_TT_OP_MEMORY_FLAG_DST_BANKED);
    check_device_constant(memory_path, "kMemBankHeadsShift", FT_TT_OP_MEMORY_BANK_HEADS_SHIFT);
    check_device_constant(memory_path, "kMemMaxBanks", FT_TT_OP_MEMORY_MAX_BANKS);
    check_device_constant(memory_path, "kMemFaceRows", FT_TT_OP_MEMORY_FACE_ROWS);

    check_device_constant(queue_path, "kOpEmbedding", FT_TT_QUEUE_OP_EMBEDDING);
    check_device_constant(queue_path, "kOpKvAppend", FT_TT_QUEUE_OP_KV_APPEND);
}

/* ================================================================
 * The face arithmetic, against an independent implementation
 * ================================================================ */

static void test_chunk_offset_is_the_tilize_layout(void)
{
    uint32_t row;
    unsigned mismatches = 0;
    unsigned host_mismatches = 0;

    ft_tt_test_step("face arithmetic: the KERNEL's chunk offsets against ft_tt_tilize.c");

    for (row = 0; row < TILE_DIM; row++) {
        uint32_t half;

        for (half = 0; half < 2u; half++) {
            const uint32_t expected =
                ft_tt_tile_face_index(row, half * 16u) * 2u; /* bf16 */
            const uint32_t device = ft_tt_device_chunk_offset(row, half, TILE_BYTES);
            const uint32_t host = ft_tt_op_memory_chunk_offset(row, half, TILE_BYTES);

            if (device != expected)
                mismatches++;
            if (host != device)
                host_mismatches++;
        }
    }
    ft_tt_test_check_fmt("all 64 face rows land where ft_tt_tilize puts them",
                         mismatches == 0u, "%u of 64 wrong", mismatches);
    ft_tt_test_check_fmt("and the host copy agrees with the kernel's", host_mismatches == 0u,
                         "%u of 64 differ", host_mismatches);
    ft_tt_test_check_fmt("a face row is 32 bytes at bf16",
                         ft_tt_device_chunk_bytes(TILE_BYTES) == 32u, "%u",
                         ft_tt_device_chunk_bytes(TILE_BYTES));
    ft_tt_test_check_fmt("and 64 at fp32",
                         ft_tt_device_chunk_bytes(FT_TT_OP_TILE_BYTES_FP32) == 64u, "%u",
                         ft_tt_device_chunk_bytes(FT_TT_OP_TILE_BYTES_FP32));
    ft_tt_test_check_fmt("a face is a quarter of a tile",
                         ft_tt_device_face_bytes(TILE_BYTES) == 512u, "%u",
                         ft_tt_device_face_bytes(TILE_BYTES));
    /* The property the whole family rests on: every partial-write address and length is a
     * multiple of the 16-byte DRAM write alignment. */
    ft_tt_test_check("every face row is a multiple of the 16-byte DRAM write alignment",
                     ft_tt_device_chunk_bytes(TILE_BYTES) % 16u == 0u &&
                         ft_tt_device_chunk_bytes(FT_TT_OP_TILE_BYTES_FP32) % 16u == 0u,
                     "32 and 64");
}

/**
 * The runs of a row range cover EXACTLY that range's bytes — no byte of another row, and
 * no byte of the range missed.
 *
 * Checked by painting: mark every byte the runs claim, then walk the tile element by
 * element through ft_tt_tile_face_index() and demand that a byte is marked if and only if
 * its row is in the range. A run that spilled into row 16 would be invisible to any check
 * that only looked at the rows it meant to write.
 */
static void test_destination_runs_cover_exactly_the_rows(void)
{
    uint32_t row0;
    unsigned bad_ranges = 0;
    unsigned worst_runs = 0;

    ft_tt_test_step("destination runs: exactly the rows asked for, and nothing beside them");

    for (row0 = 0; row0 < TILE_DIM; row0++) {
        uint32_t count;

        for (count = 1u; row0 + count <= TILE_DIM; count++) {
            unsigned char marked[FT_TT_OP_TILE_BYTES_BF16];
            uint32_t offsets[8];
            uint32_t lengths[8];
            uint32_t run_count =
                ft_tt_device_destination_runs(row0, count, TILE_BYTES, offsets, lengths);
            uint32_t index;
            uint32_t row;
            bool ok = true;

            memset(marked, 0, sizeof(marked));
            if (run_count == 0u || run_count > ft_tt_device_max_runs()) {
                bad_ranges++;
                continue;
            }
            if (run_count > worst_runs)
                worst_runs = run_count;
            for (index = 0; index < run_count; index++) {
                uint32_t byte;

                for (byte = 0; byte < lengths[index]; byte++)
                    marked[offsets[index] + byte] = 1u;
            }
            for (row = 0; row < TILE_DIM && ok; row++) {
                const bool wanted = row >= row0 && row < row0 + count;
                uint32_t column;

                for (column = 0; column < TILE_DIM; column++) {
                    const uint32_t at = ft_tt_tile_face_index(row, column) * 2u;

                    if ((marked[at] != 0u) != wanted || (marked[at + 1u] != 0u) != wanted) {
                        ok = false;
                        break;
                    }
                }
            }
            if (!ok)
                bad_ranges++;
        }
    }
    ft_tt_test_check_fmt("every one of the 528 row ranges is covered exactly",
                         bad_ranges == 0u, "%u wrong", bad_ranges);
    ft_tt_test_check_fmt("and never needs more than four runs", worst_runs <= 4u, "worst %u",
                         worst_runs);
    ft_tt_test_check("a zero-length or out-of-range request yields no runs",
                     ft_tt_device_destination_runs(0u, 0u, TILE_BYTES, NULL, NULL) == 0u &&
                         ft_tt_device_destination_runs(30u, 4u, TILE_BYTES, NULL, NULL) == 0u,
                     "");
}

/* ================================================================
 * The KV cache layout contract
 * ================================================================ */

static void test_kv_cache_layout(void)
{
    ft_tt_kv_cache_layout layout;
    ft_tt_kv_cache_layout model;
    uint64_t offset;
    uint32_t bytes;
    int rc;

    ft_tt_test_step("KV cache layout: the contract as arithmetic");

    /* MiniCPM5-1B, read from the GGUF header and never
     * assumed: 2 KV heads, head_dim 128, context 131072. */
    rc = ft_tt_kv_cache_layout_init(2u, 128u, 131072u, TILE_BYTES, &model);
    ft_tt_test_check_fmt("MiniCPM5-1B's shape derives", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("dim_tiles is 4 and capacity_tiles 4096",
                         model.dim_tiles == 4u && model.capacity_tiles == 4096u, "%u, %u",
                         model.dim_tiles, model.capacity_tiles);
    ft_tt_test_check_fmt("one region is 64 MiB, so K+V over 24 layers is 3 GiB",
                         model.region_bytes == 64ull * 1024ull * 1024ull, "%llu bytes",
                         (unsigned long long)model.region_bytes);

    rc = ft_tt_kv_cache_layout_init(2u, 128u, 200u, TILE_BYTES, &layout);
    ft_tt_test_check_fmt("a capacity is rounded UP to a whole position tile",
                         rc == FT_TT_OK && layout.capacity_positions == 224u &&
                             layout.capacity_tiles == 7u,
                         "rc=%d capacity=%u tiles=%u", rc, layout.capacity_positions,
                         layout.capacity_tiles);

    /* F2: the axis order. Consecutive dim tiles are adjacent; the next position tile
     * follows the last dim tile; a head is head_stride away. */
    ft_tt_test_check(
        "dim tiles are adjacent",
        ft_tt_kv_cache_tile_offset(&layout, 0u, 0u, 1u) -
                ft_tt_kv_cache_tile_offset(&layout, 0u, 0u, 0u) == TILE_BYTES,
        "");
    ft_tt_test_check("the next position tile follows the last dim tile",
                     ft_tt_kv_cache_tile_offset(&layout, 0u, 32u, 0u) -
                             ft_tt_kv_cache_tile_offset(&layout, 0u, 31u, 3u) == TILE_BYTES,
                     "");
    ft_tt_test_check("a head is head_stride away",
                     ft_tt_kv_cache_tile_offset(&layout, 1u, 0u, 0u) == layout.head_stride,
                     "");
    ft_tt_test_check("an out-of-range index refuses rather than wrapping",
                     ft_tt_kv_cache_tile_offset(&layout, 2u, 0u, 0u) == UINT64_MAX &&
                         ft_tt_kv_cache_tile_offset(&layout, 0u, 224u, 0u) == UINT64_MAX &&
                         ft_tt_kv_cache_tile_offset(&layout, 0u, 0u, 4u) == UINT64_MAX,
                     "");

    /* §1.2: the property attention is written against. */
    rc = ft_tt_kv_cache_run(&layout, 1u, 64u, 96u, &offset, &bytes);
    ft_tt_test_check_fmt("a position range is ONE run", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("it starts at the first position's tile",
                         offset == ft_tt_kv_cache_tile_offset(&layout, 1u, 64u, 0u),
                         "%llu", (unsigned long long)offset);
    ft_tt_test_check_fmt("and it is exactly the positions asked for",
                         bytes == 3u * layout.position_tile_bytes, "%u bytes", bytes);
    ft_tt_test_check("the run ends where the next one begins — no gap, no stride",
                     offset + bytes == ft_tt_kv_cache_tile_offset(&layout, 1u, 160u, 0u), "");
    ft_tt_test_check(
        "a run that is not a whole number of position tiles is refused, not rounded",
        ft_tt_kv_cache_run(&layout, 0u, 16u, 32u, &offset, &bytes) == FT_TT_EINVAL &&
            ft_tt_kv_cache_run(&layout, 0u, 0u, 33u, &offset, &bytes) == FT_TT_EINVAL,
        "");
    ft_tt_test_check("a run past the capacity is refused",
                     ft_tt_kv_cache_run(&layout, 0u, 192u, 64u, &offset, &bytes) ==
                         FT_TT_EINVAL,
                     "");

    /* §1.6: a Bfp8_b / Bfp4_b cache is opt-in (the writer packs); other sizes are refused. */
    ft_tt_test_check("a Bfp8_b and a Bfp4_b cache lay out; an unknown tile size is refused",
                     ft_tt_kv_cache_layout_init(2u, 128u, 64u, FT_TT_OP_TILE_BYTES_BFP8_B,
                                                &layout) == FT_TT_OK &&
                         layout.position_tile_bytes == 4u * FT_TT_OP_TILE_BYTES_BFP8_B &&
                         ft_tt_kv_cache_format(&layout) == FT_TT_KV_FORMAT_BFP8 &&
                         ft_tt_kv_cache_layout_init(2u, 128u, 64u, FT_TT_OP_TILE_BYTES_BFP4_B,
                                                    &layout) == FT_TT_OK &&
                         ft_tt_kv_cache_format(&layout) == FT_TT_KV_FORMAT_BFP4 &&
                         ft_tt_kv_cache_layout_init(2u, 128u, 64u, 1024u, &layout) ==
                             FT_TT_EINVAL,
                     "");
    ft_tt_test_check("a head dimension that is not a whole number of tiles is refused",
                     ft_tt_kv_cache_layout_init(2u, 100u, 64u, TILE_BYTES, &layout) ==
                         FT_TT_EINVAL,
                     "");
}

/* ================================================================
 * KV_APPEND, byte for byte
 * ================================================================ */

#define APPEND_HEADS     2u
#define APPEND_HEAD_DIM  128u
#define APPEND_DIM_TILES (APPEND_HEAD_DIM / TILE_DIM)
#define APPEND_SOURCE_TILES (APPEND_HEADS * APPEND_DIM_TILES)
#define APPEND_CAPACITY  128u

/**
 * One append, scored against an expectation built element by element.
 *
 * `stream` seeds the source content so no two calls in this file can pass on each other's
 * data.
 */
static bool run_append_case(const char *label, uint32_t position, uint32_t source_row0,
                            uint32_t token_count, uint32_t stream)
{
    ft_tt_kv_cache_layout layout;
    ft_tt_op_kv_append_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];
    unsigned char *source = NULL;
    unsigned char *cache = NULL;
    unsigned char *expected = NULL;
    size_t source_bytes;
    size_t cache_bytes;
    uint32_t row;
    uint32_t head;
    size_t wrong = 0;
    size_t disturbed = 0;
    bool ok = false;
    int rc;

    rc = ft_tt_kv_cache_layout_init(APPEND_HEADS, APPEND_HEAD_DIM, APPEND_CAPACITY, TILE_BYTES,
                                    &layout);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "layout rc=%d", rc);
        return false;
    }

    source_bytes = (size_t)APPEND_SOURCE_TILES * TILE_BYTES * 2u; /* two source tile rows */
    cache_bytes = (size_t)layout.region_bytes;
    source = malloc(source_bytes);
    cache = malloc(cache_bytes);
    expected = malloc(cache_bytes);
    if (!source || !cache || !expected) {
        ft_tt_test_check(label, false, "out of memory");
        goto done;
    }

    /* Per-ROW distinguishable content: a gather that returned the right bytes from the
     * wrong row cannot pass this. */
    memset(source, 0, source_bytes);
    for (row = 0; row < TILE_DIM * 2u; row++) {
        uint32_t column;

        for (column = 0; column < APPEND_HEADS * APPEND_HEAD_DIM; column++)
            put_tiled_element(source, APPEND_SOURCE_TILES, row, column,
                              source_pattern(stream, row, column));
    }

    memset(cache, (int)POISON_BYTE, cache_bytes);
    memcpy(expected, cache, cache_bytes);
    for (head = 0; head < APPEND_HEADS; head++) {
        uint32_t token;

        for (token = 0; token < token_count; token++) {
            uint32_t dimension;

            for (dimension = 0; dimension < APPEND_HEAD_DIM; dimension++) {
                const uint64_t tile = ft_tt_kv_cache_tile_offset(&layout, head, position + token,
                                                                 dimension / TILE_DIM);
                const size_t at = (size_t)tile + (size_t)ft_tt_tile_face_index(
                                                     (position + token) % TILE_DIM,
                                                     dimension % TILE_DIM) *
                                                     2u;
                const uint16_t value =
                    source_pattern(stream, source_row0 + token, head * APPEND_HEAD_DIM + dimension);

                memcpy(expected + at, &value, sizeof(value));
            }
        }
    }

    memset(&request, 0, sizeof(request));
    request.cache = &layout;
    request.position = position;
    request.token_count = token_count;
    request.source.x = 1u;
    request.source.y = 2u;
    request.source.address = 0u;
    request.source.tile_bytes = TILE_BYTES;
    request.source_row0 = source_row0;
    request.source_tiles = APPEND_SOURCE_TILES;
    request.region.x = 3u;
    request.region.y = 4u;
    request.region.address = 0u;
    request.region.tile_bytes = TILE_BYTES;
    rc = ft_tt_op_kv_append_parameters(&request, words);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "encode rc=%d", rc);
        goto done;
    }

    model_run(words, source, cache, NULL, NULL);

    {
        size_t index;

        for (index = 0; index < cache_bytes; index++) {
            if (cache[index] == expected[index])
                continue;
            if (expected[index] == POISON_BYTE)
                disturbed++;
            else
                wrong++;
        }
    }
    ft_tt_test_check_fmt(label, wrong == 0u && disturbed == 0u,
                         "%zu bytes wrong, %zu bytes of an untouched position disturbed",
                         wrong, disturbed);
    ok = wrong == 0u && disturbed == 0u;

done:
    free(source);
    free(cache);
    free(expected);
    return ok;
}

static void test_kv_append(void)
{
    ft_tt_test_step("KV_APPEND: byte for byte, against a poisoned cache");

    /* Decode: one token, at four positions that exercise both face row groups and both
     * position tiles. */
    (void)run_append_case("one token at position 0", 0u, 0u, 1u, 1u);
    (void)run_append_case("one token at position 7 (first face group)", 7u, 0u, 1u, 2u);
    (void)run_append_case("one token at position 20 (second face group)", 20u, 0u, 1u, 3u);
    (void)run_append_case("one token at position 63 (last row of the second tile)", 63u, 0u,
                          1u, 4u);
    /* The source row need not be the destination row — decode reads row 0 of a projection
     * whose position is anything. */
    (void)run_append_case("source row 5 into position 19", 19u, 5u, 1u, 5u);
    /* A range that CROSSES the face row group boundary: the case with four runs. */
    (void)run_append_case("six tokens across rows 14..19", 14u, 14u, 6u, 6u);
    (void)run_append_case("six tokens from source row 0 into rows 14..19", 14u, 0u, 6u, 7u);
    /* Prefill: a whole aligned tile row, which collapses to one transaction per head. */
    (void)run_append_case("a whole 32-token block at position 32", 32u, 32u, 32u, 8u);
    (void)run_append_case("a whole 32-token block from source row 0", 96u, 0u, 32u, 9u);
    /* The last position of the cache. */
    (void)run_append_case("the last position of the cache", APPEND_CAPACITY - 1u, 3u, 1u, 10u);
}

/**
 * BANKED: one descriptor appends heads that live in `bank_count` banks at one address, and
 * each bank must hold exactly the bytes the one-bank region holds for its heads.
 *
 * `red_rotate` swaps the banks' x/y bytes (bank group b written into bank b+1): it must go
 * red, or the check proves nothing.
 */
static bool run_banked_case(const char *label, uint32_t bank_count, uint32_t position,
                            uint32_t token_count, bool red_rotate)
{
    enum { HEADS = 4u, HEAD_DIM = 64u, CAPACITY = 64u };
    const uint32_t source_tiles = HEADS * HEAD_DIM / TILE_DIM;
    ft_tt_kv_cache_layout layout;
    ft_tt_op_kv_append_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];
    unsigned char *banks[FT_TT_OP_MEMORY_MAX_BANKS] = {NULL};
    unsigned char *source = NULL, *reference = NULL;
    size_t source_bytes, region_bytes, bank_bytes, wrong = 0u;
    uint32_t row, column, b, per;
    bool ok = false;

    if (ft_tt_kv_cache_layout_init(HEADS, HEAD_DIM, CAPACITY, TILE_BYTES, &layout) != FT_TT_OK)
        return ft_tt_test_check(label, false, "layout"), false;
    per = HEADS / bank_count;
    source_bytes = (size_t)source_tiles * TILE_BYTES;
    region_bytes = (size_t)layout.region_bytes;
    bank_bytes = (size_t)per * (size_t)layout.head_stride;
    source = calloc(1, source_bytes);
    reference = malloc(region_bytes);
    for (b = 0; b < bank_count; b++)
        banks[b] = malloc(bank_bytes);
    if (!source || !reference || (bank_count && !banks[bank_count - 1u])) {
        ft_tt_test_check(label, false, "out of memory");
        goto done;
    }
    for (row = 0; row < TILE_DIM; row++)
        for (column = 0; column < HEADS * HEAD_DIM; column++)
            put_tiled_element(source, source_tiles, row, column, source_pattern(11u, row, column));
    memset(reference, (int)POISON_BYTE, region_bytes);
    for (b = 0; b < bank_count; b++)
        memset(banks[b], (int)POISON_BYTE, bank_bytes);

    memset(&request, 0, sizeof(request));
    request.cache = &layout;
    request.position = position;
    request.token_count = token_count;
    request.source.x = 1u;
    request.source.y = 2u;
    request.source.tile_bytes = TILE_BYTES;
    request.source_row0 = position % TILE_DIM;
    request.source_tiles = source_tiles;
    request.region.x = 3u;
    request.region.y = 4u;
    request.region.tile_bytes = TILE_BYTES;
    if (ft_tt_op_kv_append_parameters(&request, words) != FT_TT_OK) {
        ft_tt_test_check(label, false, "one-bank encode refused");
        goto done;
    }
    model_run(words, source, reference, NULL, NULL);

    request.bank_count = bank_count;
    for (b = 0; b < bank_count; b++) {
        const uint32_t x = red_rotate ? (b + 1u) % bank_count : b;

        request.bank_region[b] = request.region;
        request.bank_region[b].x = (uint16_t)x;
        request.bank_region[b].y = (uint16_t)(100u + x);
    }
    request.region.x = request.bank_region[0].x;
    request.region.y = request.bank_region[0].y;
    if (ft_tt_op_kv_append_parameters(&request, words) != FT_TT_OK) {
        ft_tt_test_check(label, false, "banked encode refused");
        goto done;
    }
    model_run(words, source, NULL, NULL, banks);
    for (b = 0; b < bank_count; b++)
        for (size_t i = 0; i < bank_bytes; i++)
            wrong += banks[b][i] != reference[(size_t)b * bank_bytes + i];
    ok = wrong == 0u;
    ft_tt_test_check_fmt(label, red_rotate ? !ok : ok, "%zu bytes differ from the one-bank "
                         "region%s", wrong, red_rotate ? " (red: must differ)" : "");
done:
    free(source);
    free(reference);
    for (b = 0; b < FT_TT_OP_MEMORY_MAX_BANKS; b++)
        free(banks[b]);
    return ok;
}

static void test_kv_append_banked(void)
{
    ft_tt_kv_cache_layout layout;
    ft_tt_op_kv_append_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];

    ft_tt_test_step("KV_APPEND BANKED: one descriptor over the heads' banks == one bank");
    (void)run_banked_case("4 banks, one token at position 5", 4u, 5u, 1u, false);
    (void)run_banked_case("4 banks, one token at position 40", 4u, 40u, 1u, false);
    (void)run_banked_case("2 banks, 6 tokens across rows 14..19", 2u, 14u, 6u, false);
    (void)run_banked_case("4 banks, a whole 32-token block", 4u, 32u, 32u, false);
    (void)run_banked_case("RED 4 banks rotated", 4u, 5u, 1u, true);

    /* Refusals: a bank at another address, a count that does not divide the heads, > 4. */
    (void)ft_tt_kv_cache_layout_init(4u, 64u, 64u, TILE_BYTES, &layout);
    memset(&request, 0, sizeof(request));
    request.cache = &layout;
    request.token_count = 1u;
    request.source.tile_bytes = TILE_BYTES;
    request.source.x = 1u;
    request.source_tiles = 8u;
    request.region.tile_bytes = TILE_BYTES;
    request.region.x = 1u;
    request.bank_count = 2u;
    request.bank_region[0] = request.region;
    request.bank_region[1] = request.region;
    request.bank_region[1].address = 64u;
    ft_tt_test_check("banked: a bank at another address is refused",
                     ft_tt_op_kv_append_parameters(&request, words) == FT_TT_EINVAL, "");
    request.bank_region[1].address = 0u;
    request.bank_count = 3u;
    ft_tt_test_check("banked: 3 banks over 4 heads is refused",
                     ft_tt_op_kv_append_parameters(&request, words) == FT_TT_EINVAL, "");
    request.bank_count = 5u;
    ft_tt_test_check("banked: 5 banks is refused",
                     ft_tt_op_kv_append_parameters(&request, words) == FT_TT_EINVAL, "");
}

/**
 * Two appends in a row: the second must not disturb the first.
 *
 * This is the check that makes "an append PRESERVES" more than a claim about poison — the
 * bytes it must preserve are real data at a neighbouring position, in the SAME tile, in the
 * same face rows.
 */
static void test_kv_append_preserves_earlier_positions(void)
{
    ft_tt_kv_cache_layout layout;
    ft_tt_op_kv_append_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];
    unsigned char *source = NULL;
    unsigned char *cache = NULL;
    size_t source_bytes;
    size_t cache_bytes;
    uint32_t position;
    uint32_t row;
    size_t wrong = 0;
    int rc;

    ft_tt_test_step("KV_APPEND: appending position p+1 leaves position p alone");

    rc = ft_tt_kv_cache_layout_init(APPEND_HEADS, APPEND_HEAD_DIM, APPEND_CAPACITY, TILE_BYTES,
                                    &layout);
    source_bytes = (size_t)APPEND_SOURCE_TILES * TILE_BYTES;
    cache_bytes = (size_t)layout.region_bytes;
    source = malloc(source_bytes);
    cache = malloc(cache_bytes);
    if (rc != FT_TT_OK || !source || !cache) {
        ft_tt_test_check("the case sets up", false, "");
        goto done;
    }

    memset(source, 0, source_bytes);
    for (row = 0; row < TILE_DIM; row++) {
        uint32_t column;

        for (column = 0; column < APPEND_HEADS * APPEND_HEAD_DIM; column++)
            put_tiled_element(source, APPEND_SOURCE_TILES, row, column,
                              source_pattern(99u, row, column));
    }
    memset(cache, (int)POISON_BYTE, cache_bytes);

    /* Append 24 positions, ONE AT A TIME, exactly as decode does. Source row p is what
     * position p receives, so every position holds different content. */
    for (position = 0; position < 24u; position++) {
        memset(&request, 0, sizeof(request));
        request.cache = &layout;
        request.position = position;
        request.token_count = 1u;
        request.source.address = 0u;
        request.source.tile_bytes = TILE_BYTES;
        request.source_row0 = position;
        request.source_tiles = APPEND_SOURCE_TILES;
        request.region.address = 0u;
        request.region.tile_bytes = TILE_BYTES;
        if (ft_tt_op_kv_append_parameters(&request, words) != FT_TT_OK) {
            ft_tt_test_check_fmt("every step encodes", false, "position %u", position);
            goto done;
        }
        model_run(words, source, cache, NULL, NULL);
    }

    for (position = 0; position < 24u; position++) {
        uint32_t head;

        for (head = 0; head < APPEND_HEADS; head++) {
            uint32_t dimension;

            for (dimension = 0; dimension < APPEND_HEAD_DIM; dimension++) {
                const uint64_t tile =
                    ft_tt_kv_cache_tile_offset(&layout, head, position, dimension / TILE_DIM);
                const size_t at =
                    (size_t)tile +
                    (size_t)ft_tt_tile_face_index(position % TILE_DIM, dimension % TILE_DIM) * 2u;
                const uint16_t want =
                    source_pattern(99u, position, head * APPEND_HEAD_DIM + dimension);
                uint16_t got;

                memcpy(&got, cache + at, sizeof(got));
                if (got != want)
                    wrong++;
            }
        }
    }
    ft_tt_test_check_fmt("all 24 positions survive 24 single-position appends", wrong == 0u,
                         "%zu of %u elements wrong", wrong,
                         24u * APPEND_HEADS * APPEND_HEAD_DIM);

done:
    free(source);
    free(cache);
}

/* ================================================================
 * EMBEDDING, byte for byte
 * ================================================================ */

#define EMBEDDING_VOCABULARY 64u
#define EMBEDDING_DIM        64u
#define EMBEDDING_TILES      (EMBEDDING_DIM / TILE_DIM)

/**
 * A gather, with a PERMUTED, repeating id list.
 *
 * The permutation is the point: a kernel that ignored the index array and read rows in
 * order would return the right bytes from the wrong rows, which is the single most likely
 * bug in a gather and the one a similar-looking table hides.
 */
static void test_embedding(bool zero_fill)
{
    static const uint32_t ids[8] = { 40u, 3u, 17u, 3u, 63u, 0u, 55u, 8u };
    ft_tt_op_embedding_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];
    unsigned char *table = NULL;
    unsigned char *output = NULL;
    unsigned char *expected = NULL;
    const uint32_t row_bytes = EMBEDDING_DIM * 2u;
    const uint32_t output_row0 = zero_fill ? 3u : 11u;
    size_t table_bytes = (size_t)EMBEDDING_VOCABULARY * row_bytes;
    size_t output_bytes = (size_t)EMBEDDING_TILES * TILE_BYTES;
    uint32_t token;
    uint32_t row;
    size_t wrong = 0;
    int rc;

    ft_tt_test_step("EMBEDDING: a permuted gather with a repeat, %s",
                    zero_fill ? "zero-filling the rest of the tiles"
                              : "preserving the rest of the tiles");

    table = malloc(table_bytes);
    output = malloc(output_bytes);
    expected = malloc(output_bytes);
    if (!table || !output || !expected) {
        ft_tt_test_check("the case sets up", false, "out of memory");
        goto done;
    }

    /* A ROW-MAJOR table: row `id` is one contiguous run, which is the whole reason the
     * table is not tilized (a face row is 32 B and a 32-byte DRAM READ is not legal). */
    for (row = 0; row < EMBEDDING_VOCABULARY; row++) {
        uint32_t column;

        for (column = 0; column < EMBEDDING_DIM; column++) {
            const uint16_t value = source_pattern(0x1234u, row, column);

            memcpy(table + (size_t)row * row_bytes + (size_t)column * 2u, &value, sizeof(value));
        }
    }

    memset(output, (int)POISON_BYTE, output_bytes);
    memcpy(expected, output, output_bytes);
    if (zero_fill)
        memset(expected, 0, output_bytes);
    for (token = 0; token < 8u; token++) {
        uint32_t column;

        for (column = 0; column < EMBEDDING_DIM; column++) {
            const size_t at = (size_t)(column / TILE_DIM) * TILE_BYTES +
                              (size_t)ft_tt_tile_face_index(output_row0 + token,
                                                            column % TILE_DIM) *
                                  2u;
            const uint16_t value = source_pattern(0x1234u, ids[token], column);

            memcpy(expected + at, &value, sizeof(value));
        }
    }

    memset(&request, 0, sizeof(request));
    request.token_count = 8u;
    request.embedding_dim = EMBEDDING_DIM;
    request.table.address = 0u;
    request.table.tile_bytes = TILE_BYTES;
    request.table_row_stride = row_bytes;
    request.table_row_count = EMBEDDING_VOCABULARY;
    request.output.address = 0u;
    request.output.tile_bytes = TILE_BYTES;
    request.output_bytes = (uint32_t)output_bytes;
    request.output_row0 = output_row0;
    request.zero_fill = zero_fill;
    rc = ft_tt_op_embedding_parameters(&request, words);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the descriptor encodes", false, "rc=%d", rc);
        goto done;
    }

    model_run(words, table, output, ids, NULL);

    {
        size_t index;

        for (index = 0; index < output_bytes; index++) {
            if (output[index] != expected[index])
                wrong++;
        }
    }
    ft_tt_test_check_fmt("the gathered rows land byte for byte, and nothing else moves",
                         wrong == 0u, "%zu of %zu bytes differ", wrong, output_bytes);

done:
    free(table);
    free(output);
    free(expected);
}

/* ================================================================
 * The split rule, the encoders' refusals, and the placement
 * ================================================================ */

static void test_split_rule(void)
{
    uint32_t positions[64];
    uint32_t source_rows[64];
    uint32_t counts[64];
    uint32_t count = 0;
    uint32_t index;
    uint32_t total = 0;
    bool contiguous = true;
    int rc;

    ft_tt_test_step("the split rule: a long append becomes descriptors that each fit one "
                    "tile row");

    rc = ft_tt_op_kv_append_split(0u, 32u, 0u, positions, source_rows, counts, 64u, &count);
    ft_tt_test_check_fmt("an aligned 32-token block is ONE descriptor",
                         rc == FT_TT_OK && count == 1u && counts[0] == 32u, "rc=%d count=%u",
                         rc, count);

    rc = ft_tt_op_kv_append_split(5u, 100u, 0u, positions, source_rows, counts, 64u, &count);
    ft_tt_test_check_fmt("100 tokens from position 5 split", rc == FT_TT_OK, "rc=%d", rc);
    for (index = 0; index < count; index++) {
        if ((positions[index] % TILE_DIM) + counts[index] > TILE_DIM ||
            (source_rows[index] % TILE_DIM) + counts[index] > TILE_DIM)
            contiguous = false;
        if (index > 0 && (positions[index] != positions[index - 1] + counts[index - 1] ||
                          source_rows[index] != source_rows[index - 1] + counts[index - 1]))
            contiguous = false;
        total += counts[index];
    }
    ft_tt_test_check_fmt("every piece fits one source AND one destination tile row",
                         contiguous, "%u pieces", count);
    ft_tt_test_check_fmt("and they account for every token", total == 100u, "%u of 100",
                         total);
    /* Both ends bind independently: a source row and a position that are out of phase split
     * more often than either would alone. */
    rc = ft_tt_op_kv_append_split(0u, 64u, 16u, positions, source_rows, counts, 64u, &count);
    ft_tt_test_check_fmt("an out-of-phase source splits on the SOURCE boundary too",
                         rc == FT_TT_OK && count == 4u && counts[0] == 16u, "count=%u first=%u",
                         count, counts[0]);
    /* And keeping the two in phase is worth doing: 64 aligned tokens are 2 descriptors, the
     * same 64 out of phase are 4. A projection output whose row 0 is position p0 is in
     * phase by construction, so this is a property to preserve rather than a cost to pay. */
    rc = ft_tt_op_kv_append_split(0u, 64u, 0u, positions, source_rows, counts, 64u, &count);
    ft_tt_test_check_fmt("in phase, the same 64 tokens are half as many descriptors",
                         rc == FT_TT_OK && count == 2u, "count=%u", count);
    rc = ft_tt_op_kv_append_split(5u, 100u, 0u, positions, source_rows, counts, 2u, &count);
    ft_tt_test_check_fmt("too small an output array is refused, and still says how many",
                         rc == FT_TT_ENOMEM && count == 7u, "rc=%d count=%u", rc, count);
}

static void test_encoders_refuse_what_the_kernel_cannot_do(void)
{
    ft_tt_kv_cache_layout layout;
    ft_tt_op_kv_append_request append;
    ft_tt_op_embedding_request embedding;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];

    ft_tt_test_step("the encoders refuse what the kernel cannot do");

    (void)ft_tt_kv_cache_layout_init(APPEND_HEADS, APPEND_HEAD_DIM, APPEND_CAPACITY, TILE_BYTES,
                                     &layout);
    memset(&append, 0, sizeof(append));
    append.cache = &layout;
    append.position = 0u;
    append.token_count = 1u;
    append.source.tile_bytes = TILE_BYTES;
    append.source_tiles = APPEND_SOURCE_TILES;
    append.region.tile_bytes = TILE_BYTES;
    ft_tt_test_check("the baseline request encodes",
                     ft_tt_op_kv_append_parameters(&append, words) == FT_TT_OK, "");

    append.position = 30u;
    append.token_count = 4u;
    ft_tt_test_check("a range crossing a destination tile row is refused",
                     ft_tt_op_kv_append_parameters(&append, words) == FT_TT_EINVAL, "");
    append.position = 0u;
    append.source_row0 = 30u;
    ft_tt_test_check("a range crossing a SOURCE tile row is refused",
                     ft_tt_op_kv_append_parameters(&append, words) == FT_TT_EINVAL, "");
    append.source_row0 = 0u;
    append.token_count = 33u;
    ft_tt_test_check("more than 32 rows is refused",
                     ft_tt_op_kv_append_parameters(&append, words) == FT_TT_EINVAL, "");
    append.token_count = 1u;
    append.position = APPEND_CAPACITY;
    ft_tt_test_check("a position past the capacity is refused",
                     ft_tt_op_kv_append_parameters(&append, words) == FT_TT_EINVAL, "");
    append.position = 0u;
    append.source_tiles = APPEND_SOURCE_TILES - 1u;
    ft_tt_test_check("a source too narrow for every head is refused",
                     ft_tt_op_kv_append_parameters(&append, words) == FT_TT_EINVAL, "");
    append.source_tiles = APPEND_SOURCE_TILES;
    append.source.tile_bytes = FT_TT_OP_TILE_BYTES_BFP8_B;
    ft_tt_test_check("a source whose tile size disagrees with the cache is refused",
                     ft_tt_op_kv_append_parameters(&append, words) == FT_TT_EINVAL, "");

    memset(&embedding, 0, sizeof(embedding));
    embedding.token_count = 4u;
    embedding.embedding_dim = 1536u;
    embedding.table.tile_bytes = TILE_BYTES;
    embedding.table_row_stride = 1536u * 2u;
    embedding.table_row_count = 73448u;
    embedding.output.tile_bytes = TILE_BYTES;
    embedding.output_bytes = 48u * TILE_BYTES;
    ft_tt_test_check("MiniCPM5-1B's 1536-wide embedding encodes",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_OK, "");
    ft_tt_test_check_fmt("and asks for 48 tiles in one group",
                         words[FT_TT_OP_MEMORY_GROUP_TILES] == 48u &&
                             words[FT_TT_OP_MEMORY_GROUP_COUNT] == 1u,
                         "%u tiles, %u groups", words[FT_TT_OP_MEMORY_GROUP_TILES],
                         words[FT_TT_OP_MEMORY_GROUP_COUNT]);
    ft_tt_test_check("every embedding descriptor is a GATHER",
                     (words[FT_TT_OP_MEMORY_FLAGS] &
                      (uint32_t)FT_TT_OP_MEMORY_FLAG_GATHER) != 0u,
                     "");
    embedding.table_row_stride = 1536u * 2u + 32u;
    ft_tt_test_check("a table row stride that breaks the 64-byte DRAM read alignment is "
                     "refused",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_EINVAL, "");
    embedding.table_row_stride = 1536u * 2u;
    embedding.embedding_dim = 1500u;
    ft_tt_test_check("a width that is not a whole number of tiles is refused",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_EINVAL, "");
    embedding.embedding_dim = 1536u;
    embedding.output_row0 = 30u;
    embedding.token_count = 4u;
    ft_tt_test_check("a token block crossing a destination tile row is refused",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_EINVAL, "");

    /* THE TWO BOUNDS THE KERNEL REFUSES WITHOUT (gaps 4 and 5). Each is asserted against
     * FT_TT_EINVAL specifically and re-armed afterwards, so the case that follows it is not
     * being carried by the previous one still being broken — the failure mode a check with
     * several sufficient grounds always has. */
    embedding.output_row0 = 0u;
    embedding.token_count = 4u;
    ft_tt_test_check("the request is green again before the bound cases",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_OK, "");
    embedding.table_row_count = 0u;
    ft_tt_test_check("a table with no declared row count is refused — an unbounded token id "
                     "is an arbitrary DRAM read",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_EINVAL, "");
    embedding.table_row_count = 73448u;
    embedding.output_bytes = 0u;
    ft_tt_test_check("an output with no declared extent is refused — an absolute row0 is an "
                     "unbounded byte offset",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_EINVAL, "");
    embedding.output_bytes = 48u * TILE_BYTES;
    embedding.output_row0 = 32u;
    ft_tt_test_check("and a destination tile row past that extent is refused, not clamped",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_EINVAL, "");
    embedding.output_bytes = 2u * 48u * TILE_BYTES;
    ft_tt_test_check("the same row0 against an extent that HOLDS it encodes",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_OK, "");
    ft_tt_test_check_fmt("and both bounds ride the descriptor",
                         words[FT_TT_OP_MEMORY_SRC_ROW_LIMIT] == 73448u &&
                             words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE] == 2u * 48u * TILE_BYTES,
                         "row limit %u, group bytes %u", words[FT_TT_OP_MEMORY_SRC_ROW_LIMIT],
                         words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE]);
}

static void test_circular_buffers_fit_the_arena(void)
{
    ft_tt_launch_cb_entry entries[FT_TT_OP_MEMORY_CB_COUNT];
    uint64_t mask = 0;
    uint32_t index_address;
    int rc;

    ft_tt_test_step("circular buffers: ONE OUTPUT CB PER GROUP SIZE, and each is a whole "
                    "multiple of its own group");

    /* MiniCPM5-1B: 48 tiles of hidden width for the embedding, 4 dim-tiles a KV head. */
    rc = ft_tt_op_memory_place_cbs(48u, 4u, TILE_BYTES, entries, g_cb_buffer,
                                   sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("a 1536-wide embedding + 4-tile KV residency places", rc == FT_TT_OK,
                         "rc=%d", rc);
    index_address = ft_tt_op_memory_index_address(entries, FT_TT_OP_MEMORY_CB_COUNT);
    ft_tt_test_check_fmt("and the index array has an address in the arena",
                         index_address >= FT_TT_OP_CB_ARENA_BASE &&
                             index_address < FT_TT_OP_CB_ARENA_END,
                         "0x%X", index_address);
    ft_tt_test_check_fmt("the mask names exactly the five circular buffers this op uses",
                         mask == ((1ull << FT_TT_OP_MEMORY_CB_STAGE) |
                                  (1ull << FT_TT_OP_MEMORY_CB_INDEX) |
                                  (1ull << FT_TT_OP_MEMORY_CB_KV_PACK) |
                                  (1ull << FT_TT_OP_MEMORY_CB_OUT_EMBEDDING) |
                                  (1ull << FT_TT_OP_MEMORY_CB_OUT_KV)),
                         "0x%llX", (unsigned long long)mask);
    /* THE INVARIANT THE SPLIT EXISTS FOR, stated as the arithmetic the kernel's guard runs:
     * each output CB holds a whole number of ITS OWN blocks, so its pointer is on that grid
     * at every descriptor boundary. The two are DIFFERENT buffers — the check that would
     * have failed before the split is that they do not overlap. */
    {
        const ft_tt_launch_cb_entry *embedding_out =
            ft_tt_op_cb_find(entries, FT_TT_OP_MEMORY_CB_COUNT, FT_TT_OP_MEMORY_CB_OUT_EMBEDDING);
        const ft_tt_launch_cb_entry *kv_out =
            ft_tt_op_cb_find(entries, FT_TT_OP_MEMORY_CB_COUNT, FT_TT_OP_MEMORY_CB_OUT_KV);

        ft_tt_test_check("both output circular buffers are placed",
                         embedding_out != NULL && kv_out != NULL, "");
        if (embedding_out && kv_out) {
            ft_tt_test_check_fmt("the embedding output holds a whole number of 48-tile blocks",
                                 embedding_out->page_count % 48u == 0u, "%u pages",
                                 embedding_out->page_count);
            ft_tt_test_check_fmt("the KV output holds a whole number of 4-tile blocks",
                                 kv_out->page_count % 4u == 0u, "%u pages", kv_out->page_count);
            ft_tt_test_check_fmt("and they are DISJOINT byte ranges — the whole point of the "
                                 "split, since one buffer cannot carry two group sizes",
                                 embedding_out->address + embedding_out->size_bytes <=
                                         kv_out->address ||
                                     kv_out->address + kv_out->size_bytes <=
                                         embedding_out->address,
                                 "embedding 0x%X+%u kv 0x%X+%u", embedding_out->address,
                                 embedding_out->size_bytes, kv_out->address,
                                 kv_out->size_bytes);
        }
    }

    rc = ft_tt_op_memory_place_cbs(0u, 4u, TILE_BYTES, entries, g_cb_buffer,
                                   sizeof(g_cb_buffer), &mask);
    ft_tt_test_check("a zero-width embedding group is refused", rc == FT_TT_EINVAL, "");
    rc = ft_tt_op_memory_place_cbs(48u, 0u, TILE_BYTES, entries, g_cb_buffer,
                                   sizeof(g_cb_buffer), &mask);
    ft_tt_test_check("a zero-width KV group is refused", rc == FT_TT_EINVAL, "");
    rc = ft_tt_op_memory_place_cbs(48u, 4u, FT_TT_OP_TILE_BYTES_BFP8_B, entries, g_cb_buffer,
                                   sizeof(g_cb_buffer), &mask);
    ft_tt_test_check("a block-float page size is refused", rc == FT_TT_EINVAL, "");
}

/* ================================================================
 * The zero-length guard, and the cross-op row phase
 * ================================================================ */

/**
 * Nothing this family issues can be zero bytes long.
 *
 * A zero-length NoC read is suspected of taking the WHOLE CARD down — the RoPE round's
 * card-down on 2026-08-12 happened at exactly the first descriptor that computed one
 * (INFERRED, and deliberately not re-confirmed, because confirming it costs the card for
 * every worker on it). Both ops here are gathers built out of computed lengths, so a zero
 * arrives by arithmetic rather than by a typo: a token count of zero, a collapsed range, an
 * underflowed subtraction.
 *
 * So the guard is asserted three ways — the kernel HAS one, every length it can compute is
 * non-zero, and the host REFUSES the requests that would produce one.
 */
static void test_no_transfer_can_be_zero_length(void)
{
    ft_tt_kv_cache_layout layout;
    ft_tt_op_kv_append_request append;
    ft_tt_op_embedding_request embedding;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];
    uint32_t row0;
    unsigned zero_runs = 0;
    unsigned zero_rows = 0;
    uint32_t group_tiles;

    ft_tt_test_step("no transfer this family issues can be zero bytes long");

    ft_tt_test_check("the kernel has the guard, and it answers",
                     ft_tt_device_length_is_issuable(0u) == 0 &&
                         ft_tt_device_length_is_issuable(32u) == 1,
                     "");

    for (row0 = 0; row0 < TILE_DIM; row0++) {
        uint32_t count;

        for (count = 1u; row0 + count <= TILE_DIM; count++) {
            uint32_t offsets[8];
            uint32_t lengths[8];
            const uint32_t run_count =
                ft_tt_device_destination_runs(row0, count, TILE_BYTES, offsets, lengths);
            uint32_t index;

            for (index = 0; index < run_count; index++) {
                if (!ft_tt_device_length_is_issuable(lengths[index]))
                    zero_runs++;
            }
        }
    }
    ft_tt_test_check_fmt("every run of every one of the 528 row ranges is non-empty",
                         zero_runs == 0u, "%u empty", zero_runs);

    for (group_tiles = 1u; group_tiles <= 64u; group_tiles++) {
        if (!ft_tt_device_length_is_issuable(
                ft_tt_device_group_row_bytes(group_tiles, TILE_BYTES)))
            zero_rows++;
    }
    ft_tt_test_check_fmt("a source row is non-empty at every group width 1..64",
                         zero_rows == 0u, "%u empty", zero_rows);

    /* And the host refuses the requests that would compute one, rather than sending a
     * descriptor the kernel has to refuse on the card. */
    (void)ft_tt_kv_cache_layout_init(APPEND_HEADS, APPEND_HEAD_DIM, APPEND_CAPACITY, TILE_BYTES,
                                     &layout);
    memset(&append, 0, sizeof(append));
    append.cache = &layout;
    append.token_count = 0u;
    append.source.tile_bytes = TILE_BYTES;
    append.source_tiles = APPEND_SOURCE_TILES;
    append.region.tile_bytes = TILE_BYTES;
    ft_tt_test_check("a zero-token append is REFUSED, not sent as a no-op",
                     ft_tt_op_kv_append_parameters(&append, words) == FT_TT_EINVAL, "");

    memset(&embedding, 0, sizeof(embedding));
    embedding.token_count = 0u;
    embedding.embedding_dim = 1536u;
    embedding.table.tile_bytes = TILE_BYTES;
    embedding.table_row_stride = 1536u * 2u;
    embedding.output.tile_bytes = TILE_BYTES;
    ft_tt_test_check("a zero-token gather is REFUSED",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_EINVAL, "");
    embedding.token_count = 4u;
    embedding.embedding_dim = 0u;
    ft_tt_test_check("a zero-width gather is REFUSED",
                     ft_tt_op_embedding_parameters(&embedding, words) == FT_TT_EINVAL, "");
}

/**
 * §1.5b: a token at position p occupies tile row p % 32 — in BOTH ops, and RoPE depends on
 * it.
 *
 * This is the check that keeps a cross-op constraint from being lost between two authors:
 * RoPE multiplies activation tile row `r` by sin/cos table row `r`, so a hidden state that
 * the embedding op put at the wrong row is rotated by the wrong angle — a plausible wrong
 * answer with no error anywhere.
 */
static void test_row_phase_invariant(void)
{
    ft_tt_kv_cache_layout layout;
    ft_tt_op_kv_append_request append;
    ft_tt_op_embedding_request embedding;
    uint32_t append_words[FT_TT_OP_MEMORY_WORDS];
    uint32_t embedding_words[FT_TT_OP_MEMORY_WORDS];
    uint32_t position;
    unsigned disagreements = 0;

    ft_tt_test_step("the row-phase invariant: position p lands at tile row p %% 32, in BOTH "
                    "ops");

    for (position = 0; position < 100u; position++) {
        if (ft_tt_op_memory_position_row(position) != position % TILE_DIM)
            disagreements++;
    }
    ft_tt_test_check_fmt("the helper is position modulo 32", disagreements == 0u, "%u wrong",
                         disagreements);

    (void)ft_tt_kv_cache_layout_init(APPEND_HEADS, APPEND_HEAD_DIM, APPEND_CAPACITY, TILE_BYTES,
                                     &layout);
    for (position = 0; position < APPEND_CAPACITY; position++) {
        memset(&append, 0, sizeof(append));
        append.cache = &layout;
        append.position = position;
        append.token_count = 1u;
        append.source.tile_bytes = TILE_BYTES;
        append.source_tiles = APPEND_SOURCE_TILES;
        append.region.tile_bytes = TILE_BYTES;
        if (ft_tt_op_kv_append_parameters(&append, append_words) != FT_TT_OK) {
            disagreements++;
            continue;
        }

        memset(&embedding, 0, sizeof(embedding));
        embedding.token_count = 1u;
        embedding.embedding_dim = 1536u;
        embedding.table.tile_bytes = TILE_BYTES;
        embedding.table_row_stride = 1536u * 2u;
        embedding.table_row_count = 73448u;
        embedding.output.tile_bytes = TILE_BYTES;
        embedding.output_bytes = 48u * TILE_BYTES;
        embedding.output_row0 = ft_tt_op_memory_position_row(position);
        if (ft_tt_op_embedding_parameters(&embedding, embedding_words) != FT_TT_OK) {
            disagreements++;
            continue;
        }

        /* The two ops must put the same position on the same tile row — that is the whole
         * statement, and it is what RoPE reads. */
        if (append_words[FT_TT_OP_MEMORY_DST_ROW0] % TILE_DIM !=
            embedding_words[FT_TT_OP_MEMORY_DST_ROW0] % TILE_DIM)
            disagreements++;
        if (embedding_words[FT_TT_OP_MEMORY_DST_ROW0] % TILE_DIM != position % TILE_DIM)
            disagreements++;
    }
    ft_tt_test_check_fmt("EMBEDDING and KV_APPEND agree on the row for all 128 positions",
                         disagreements == 0u, "%u disagreements", disagreements);
}

/* ================================================================ */

int main(void)
{
    ft_tt_test_step("Phase 6 memory ops — OFF-HARDWARE gate");

    test_wire_constants_agree();
    test_chunk_offset_is_the_tilize_layout();
    test_destination_runs_cover_exactly_the_rows();
    test_kv_cache_layout();
    test_kv_append();
    test_kv_append_preserves_earlier_positions();
    test_kv_append_banked();
    test_embedding(true);
    test_embedding(false);
    test_split_rule();
    test_encoders_refuse_what_the_kernel_cannot_do();
    test_circular_buffers_fit_the_arena();
    test_no_transfer_can_be_zero_length();
    test_row_phase_invariant();

    return ft_tt_test_finish();
}
