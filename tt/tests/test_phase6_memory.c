/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_memory.c
 * @brief ON-CARD acceptance for the memory op family — EMBEDDING and KV_APPEND — scored
 *   BYTE FOR BYTE against a host expectation, never by a correlation.
 *
 * ---------------------------------------------------------------------------
 * WHY THERE IS NO PCC IN THIS FILE
 * ---------------------------------------------------------------------------
 * Both ops are placed COPIES. PCC is blind to a consistent permutation (dev notes §3), so
 * on a layout op it is not a weak instrument, it is the wrong one: a gather that returned
 * every row shifted by one, or a tile whose faces were exchanged, would score 1.0. The
 * acceptance is therefore `memcmp` against an image built element by element through
 * `ft_tt_tile_face_index()`, and the source content is per-ROW distinguishable so "the
 * right bytes from the wrong row" is a mismatch in every element rather than in none.
 *
 * ---------------------------------------------------------------------------
 * STAGE 1 IS A MEASUREMENT, NOT A TEST
 * ---------------------------------------------------------------------------
 * The whole family rests on one fact: **a 32-byte write to DRAM, at a 32-byte-aligned
 * address, from a kernel's NoC, lands.** Our own alignment table says it should — DRAM
 * writes take the 16-byte L1 alignment and only READS need 64 (ft_tt_noc.c,
 * ft_tt_noc_alignment()) — but that is a host-side transliteration and nothing has driven
 * it from a Tensix. If it is false, a one-position KV append cannot be a copy and this
 * family needs a read-modify-write path.
 *
 * So stage 1 poisons a whole cache region, appends ONE position, and checks two things that
 * fail differently: the 256 bytes it meant to write are byte-exact, AND every other byte of
 * the region is still poison. The second half is the one that matters — a write that
 * quietly rounded up to a 64-byte granule would pass a check that only looked at what it
 * meant to write, and would corrupt a neighbouring position in production.
 *
 * ---------------------------------------------------------------------------
 * THE RESIDENCY CLAIM, AND THE DEGENERATE TRIAD
 * ---------------------------------------------------------------------------
 * Both opcodes run on ONE resident program with the images loaded ONCE and no launch of any
 * kind between them — and that program has **two processors, not five**: a reader on BRISC
 * and a writer on NCRISC, with `enables` DM0|DM1 and no compute thread at all
 * (kernels/resident_memory_protocol.h says why the CB page counters are safe without one).
 * This is the first program in the tree of that shape, so "the triad still works with the
 * TRISCs parked in their GO wait" is a result of this file and not an assumption of it.
 *
 * ---------------------------------------------------------------------------
 * DO NOT PARK
 * ---------------------------------------------------------------------------
 * On a resident program, parking is the hazard — twice, on cores that had retired cleanly
 * (dev notes §1, bringup log §20.3). This test retires by ASKING and leaves the core in its
 * firmware's GO wait, which is where a healthy idle core sits.
 *
 * Environment:
 *   FT_TT_TENSIX_INDEX  which enabled Tensix to drive (this round owns 30-39)
 *   FT_TT_FIRMWARE_DIR  the program directory whose fw_*.elf to boot
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=../programs/memory FT_TT_TENSIX_INDEX=30 \
 *     ./build-memory-card/test_phase6_memory ../programs/memory
 *
 * @status GREEN on Tensix 30 and again on Tensix 32 — 17 checks, 0 failures each, and both
 *   cores probe PASS 4/4 afterwards. Stage 1's measurement came back clean: 0 bytes wrong
 *   and 0 bytes of an untouched position disturbed, so a 32-byte DRAM write from a kernel
 *   lands and does not round up to a wider granule. The program directory is
 *   tt/programs/memory.
 * @issues None
 * @todo
 *   - [ ] `ft_tt_core probe --program <this program dir>` WEDGES A CORE, and it is not this
 *         test's fault: probe runs the EMBEDDED vector_add triad, linked against the
 *         COMMITTED firmware, so pointing it at this directory mixes a kernel with a foreign
 *         firmware — the exact hazard ft_tt_program_directory.h names. It cost Tensix 30 and
 *         31 on 2026-08-12. Probe with the DEFAULT program, always.
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ft_tt_coord.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_memory.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_BYTES     FT_TT_OP_TILE_BYTES_BF16
#define TILE_DIM       FT_TT_OP_TILE_DIM
#define RUN_TIMEOUT_MS 20000u
#define QUEUE_CAPACITY 64u
#define POISON_BYTE    0xA5u

/** The two bounds. Neither may be 0 — an unbounded wait is the one thing a resident kernel
 *  may not be configured into (kernels/resident_queue_protocol.h rule 4). */
#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

/** THE DEGENERATE TRIAD: a reader and a writer, and no compute thread. */
#define MEMORY_MASK ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1))

/** Banks 6 and 7, at an offset no other suite in this tree uses, because three other
 *  workers own their own cores on the same card. */
#define DRAM_BANK_SOURCE 6u
#define DRAM_BANK_OUTPUT 7u
#define DRAM_BANK_COUNT  8u
#define SOURCE_BASE      0x02000000u
#define TABLE_BASE       0x02100000u
#define CACHE_BASE       0x02000000u
#define EMBEDDING_BASE   0x02200000u

/** MiniCPM5-1B: 2 KV heads, head_dim 128. The capacity is
 *  the test's, not the model's — 128 positions is four position tiles, enough to exercise
 *  both face row groups, a tile boundary and the last position. */
#define KV_HEADS        2u
#define KV_HEAD_DIM     128u
#define KV_DIM_TILES    (KV_HEAD_DIM / TILE_DIM)
#define KV_SOURCE_TILES (KV_HEADS * KV_DIM_TILES)
#define KV_CAPACITY     128u
#define KV_SOURCE_ROWS  (TILE_DIM * 2u)

/** MiniCPM5-1B's hidden width, so the residency is the one a real embedding core holds. */
#define EMBEDDING_DIM        1536u
#define EMBEDDING_TILES      (EMBEDDING_DIM / TILE_DIM)
#define EMBEDDING_VOCABULARY 64u
#define EMBEDDING_TOKENS     8u
#define EMBEDDING_ROW_BYTES  (EMBEDDING_DIM * 2u)

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_program;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
static ft_tt_kv_cache_layout g_cache;
static uint32_t g_index_address;
static uint32_t g_slot;
static uint16_t g_core_x;
static uint16_t g_core_y;

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

static int poison_region(const ft_tt_noc_access *noc, ft_tt_noc_coord where, uint32_t address,
                         size_t bytes)
{
    unsigned char pattern[0x1000];
    size_t offset = 0;

    memset(pattern, (int)POISON_BYTE, sizeof(pattern));
    while (offset < bytes) {
        size_t chunk = bytes - offset;
        int rc;

        if (chunk > sizeof(pattern))
            chunk = sizeof(pattern);
        rc = device_write(noc, where, address + (uint32_t)offset, pattern, chunk);
        if (rc != FT_TT_OK)
            return rc;
        offset += chunk;
    }
    return FT_TT_OK;
}

/* ================================================================
 * Test data that cannot be confused between rows
 * ================================================================ */

/** The same generator the off-hardware suite uses: every (stream, row, column) is a
 *  different full-mantissa bit pattern, and none of them is the poison. */
static uint16_t source_pattern(uint32_t stream, uint32_t row, uint32_t column)
{
    uint32_t mixed = (stream * 2654435761u) ^ (row * 40503u) ^ (column * 2246822519u);

    mixed ^= mixed >> 13;
    mixed = (mixed * 1103515245u) + 12345u;
    mixed >>= 9;
    mixed &= 0xFFFFu;
    if ((mixed & 0xFFu) == POISON_BYTE)
        mixed ^= 0x0001u;
    if (((mixed >> 8) & 0xFFu) == POISON_BYTE)
        mixed ^= 0x0100u;
    return (uint16_t)mixed;
}

static void put_tiled_element(unsigned char *base, uint32_t tiles_wide, uint32_t row,
                              uint32_t column, uint16_t value)
{
    const size_t offset =
        ((size_t)(row / TILE_DIM) * tiles_wide + column / TILE_DIM) * TILE_BYTES +
        (size_t)ft_tt_tile_face_index(row % TILE_DIM, column % TILE_DIM) * 2u;

    memcpy(base + offset, &value, sizeof(value));
}

/* ================================================================
 * Bringing the resident pair up
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

/**
 * Stage and KICK the resident pair: two runtime-arg blocks, the CB descriptor array, the
 * launch message, GO — and then the host walks away. Everything after this is an enqueue.
 */
static int start_pair(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    static const unsigned processors[2] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1 };
    uint32_t args[2][FT_TT_QUEUE_KERNEL_ARG_COUNT];
    ft_tt_op_args blocks[2];
    unsigned index;
    int rc;

    for (index = 0; index < 2u; index++) {
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

    rc = ft_tt_op_stage(noc, x, y, &g_config, g_cb_buffer, sizeof(g_cb_buffer), blocks, 2u);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_send_go(noc, x, y);
}

static void retire_pair(const ft_tt_noc_access *noc, const char *label)
{
    uint32_t unstopped = 0u;
    int rc = ft_tt_watchdog_shutdown(noc, &g_group, RUN_TIMEOUT_MS, &unstopped);

    ft_tt_test_check_fmt(label, rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=0x%X", rc,
                         unstopped);
}

/** Enqueue one already-encoded parameter block and wait for the WRITER to publish it. */
static bool submit(const ft_tt_hardware_session *session, uint32_t opcode,
                   const uint32_t words[FT_TT_OP_MEMORY_WORDS], const char *label)
{
    ft_tt_queue_descriptor descriptor;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    const char *reason = NULL;
    int rc;

    rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, g_slot % 64u, words,
                                         &parameter_address);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "parameters rc=%d", rc);
        return false;
    }
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = opcode;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    if (rc != FT_TT_OK) {
        /* A timeout says nothing about WHY. The kernel writes its reason into the control
         * block before it leaves, and a refusal and a hang are different diagnoses that a
         * bare rc cannot tell apart — which cost a whole round trip to the card host once. */
        uint32_t why[FT_TT_QUEUE_FIELD_COUNT];

        memset(why, 0, sizeof(why));
        (void)ft_tt_queue_read_control(&session->noc, &g_queue, why);
        ft_tt_test_check_fmt(label, false,
                             "rc=%d completed=%u of %u, kernel error=%u (%s) state=0x%08x "
                             "tail=%u head=%u idle=%u lastop=0x%08x",
                             rc, completed, g_slot + 1u, why[FT_TT_QUEUE_FIELD_ERROR],
                             ft_tt_queue_kernel_error_string(why[FT_TT_QUEUE_FIELD_ERROR]),
                             why[FT_TT_QUEUE_FIELD_STATE], why[FT_TT_QUEUE_FIELD_TAIL],
                             why[FT_TT_QUEUE_FIELD_HEAD], why[FT_TT_QUEUE_FIELD_IDLE_SPINS],
                             why[FT_TT_QUEUE_FIELD_LAST_OPCODE]);
        return false;
    }
    (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
    if (ft_tt_queue_check_completion(&g_queue, fields, g_slot + 1u, &reason) != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "completion not evidenced: %s",
                             reason ? reason : "?");
        return false;
    }
    g_slot++;
    return true;
}

/* ================================================================
 * Stage 0 — the transfer path, before anything is scored
 * ================================================================ */

static void stage_0_round_trip(const ft_tt_hardware_session *session)
{
    unsigned char written[TILE_BYTES];
    unsigned char read_back[TILE_BYTES];
    size_t index;
    int rc;

    ft_tt_test_step("stage 0 — host->DRAM->host, so the transfer path is a LINE and not an "
                    "assumption");

    for (index = 0; index < sizeof(written); index++)
        written[index] = (unsigned char)(source_pattern(7u, (uint32_t)index, 1u) & 0xFFu);
    memset(read_back, 0, sizeof(read_back));

    rc = device_write(&session->noc, g_dram[DRAM_BANK_OUTPUT], CACHE_BASE, written,
                      sizeof(written));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_dram[DRAM_BANK_OUTPUT], CACHE_BASE, read_back,
                         sizeof(read_back));
    ft_tt_test_check_fmt("a tile round-trips through DRAM", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check("and it comes back byte-exact",
                     memcmp(written, read_back, sizeof(written)) == 0, "");
}

/* ================================================================
 * The KV source, and the expectation
 * ================================================================ */

/** Upload two source tile rows of a KV projection output, per-row distinguishable. */
static int upload_kv_source(const ft_tt_hardware_session *session, unsigned char *scratch)
{
    uint32_t row;

    memset(scratch, 0, (size_t)KV_SOURCE_TILES * TILE_BYTES * 2u);
    for (row = 0; row < KV_SOURCE_ROWS; row++) {
        uint32_t column;

        for (column = 0; column < KV_HEADS * KV_HEAD_DIM; column++)
            put_tiled_element(scratch, KV_SOURCE_TILES, row, column,
                              source_pattern(0x4B56u, row, column));
    }
    return device_write(&session->noc, g_dram[DRAM_BANK_SOURCE], SOURCE_BASE, scratch,
                        (size_t)KV_SOURCE_TILES * TILE_BYTES * 2u);
}

/** Write into `expected` what the cache must hold after `token_count` rows starting at
 *  `source_row0` have been appended at `position`. */
static void expect_appended(unsigned char *expected, uint32_t position, uint32_t source_row0,
                            uint32_t token_count)
{
    uint32_t head;

    for (head = 0; head < KV_HEADS; head++) {
        uint32_t token;

        for (token = 0; token < token_count; token++) {
            uint32_t dimension;

            for (dimension = 0; dimension < KV_HEAD_DIM; dimension++) {
                const uint64_t tile = ft_tt_kv_cache_tile_offset(&g_cache, head,
                                                                 position + token,
                                                                 dimension / TILE_DIM);
                const size_t at =
                    (size_t)tile + (size_t)ft_tt_tile_face_index((position + token) % TILE_DIM,
                                                                 dimension % TILE_DIM) *
                                       2u;
                const uint16_t value =
                    source_pattern(0x4B56u, source_row0 + token, head * KV_HEAD_DIM + dimension);

                memcpy(expected + at, &value, sizeof(value));
            }
        }
    }
}

static int append_once(const ft_tt_hardware_session *session, uint32_t position,
                       uint32_t source_row0, uint32_t token_count, const char *label)
{
    ft_tt_op_kv_append_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];
    int rc;

    memset(&request, 0, sizeof(request));
    request.cache = &g_cache;
    request.position = position;
    request.token_count = token_count;
    request.source.x = (uint16_t)g_dram[DRAM_BANK_SOURCE].x;
    request.source.y = (uint16_t)g_dram[DRAM_BANK_SOURCE].y;
    request.source.address = SOURCE_BASE;
    request.source.tile_bytes = TILE_BYTES;
    request.source_row0 = source_row0;
    request.source_tiles = KV_SOURCE_TILES;
    request.region.x = (uint16_t)g_dram[DRAM_BANK_OUTPUT].x;
    request.region.y = (uint16_t)g_dram[DRAM_BANK_OUTPUT].y;
    request.region.address = CACHE_BASE;
    request.region.tile_bytes = TILE_BYTES;
    rc = ft_tt_op_kv_append_parameters(&request, words);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "encode rc=%d", rc);
        return rc;
    }
    return submit(session, FT_TT_QUEUE_OP_KV_APPEND, words, label) ? FT_TT_OK : FT_TT_EIO;
}

/** Read the cache back and compare it with `expected`, counting the two failure kinds
 *  separately because they mean different things. */
static void score_cache(const ft_tt_hardware_session *session, const unsigned char *expected,
                        unsigned char *scratch, const char *label)
{
    const size_t bytes = (size_t)g_cache.region_bytes;
    size_t wrong = 0;
    size_t disturbed = 0;
    size_t index;
    int rc = device_read(&session->noc, g_dram[DRAM_BANK_OUTPUT], CACHE_BASE, scratch, bytes);

    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "read back rc=%d", rc);
        return;
    }
    for (index = 0; index < bytes; index++) {
        if (scratch[index] == expected[index])
            continue;
        if (expected[index] == POISON_BYTE)
            disturbed++;
        else
            wrong++;
    }
    ft_tt_test_check_fmt(label, wrong == 0u && disturbed == 0u,
                         "%zu bytes wrong, %zu bytes of an untouched position DISTURBED",
                         wrong, disturbed);
}

/* ================================================================
 * Stage 1 — the measurement: one position, 32-byte DRAM writes
 * ================================================================ */

static bool stage_1_single_position(const ft_tt_hardware_session *session,
                                    unsigned char *expected, unsigned char *scratch)
{
    const size_t bytes = (size_t)g_cache.region_bytes;
    int rc;

    ft_tt_test_step("stage 1 — MEASUREMENT: one position appended with 32-byte DRAM writes");

    rc = poison_region(&session->noc, g_dram[DRAM_BANK_OUTPUT], CACHE_BASE, bytes);
    ft_tt_test_check_fmt("the cache region poisons", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;
    memset(expected, (int)POISON_BYTE, bytes);

    /* Position 7: inside the first face row group, and not at a tile boundary, so every
     * write is 32 bytes at an address that is 32-byte but NOT 64-byte aligned. That is the
     * exact case the alignment table permits and nothing has driven. */
    if (append_once(session, 7u, 0u, 1u, "one position appends with no launch, only an "
                                         "enqueue") != FT_TT_OK)
        return false;
    expect_appended(expected, 7u, 0u, 1u);
    score_cache(session, expected, scratch,
                "the 256 bytes land byte-exact AND every neighbouring position is untouched");
    return true;
}

/* ================================================================
 * Stage 2 — decode and prefill shapes
 * ================================================================ */

static bool stage_2_append_shapes(const ft_tt_hardware_session *session,
                                  unsigned char *expected, unsigned char *scratch)
{
    uint32_t position;

    ft_tt_test_step("stage 2 — 24 single-position appends, then a 32-token block");

    /* One position at a time, exactly as decode does, each from its own source row. Every
     * step must leave every earlier position alone — which is what a face-row run that
     * spilled by one row would break, and what no summary metric would see. */
    for (position = 0; position < 24u; position++) {
        char label[96];

        snprintf(label, sizeof(label), "position %u appends", position);
        if (append_once(session, position, position, 1u, label) != FT_TT_OK)
            return false;
        expect_appended(expected, position, position, 1u);
    }
    score_cache(session, expected, scratch,
                "all 24 positions survive 24 single-position appends");

    /* Prefill: a whole aligned tile row, which collapses to ONE transaction per head. */
    if (append_once(session, 32u, 32u, 32u, "an aligned 32-token block appends") != FT_TT_OK)
        return false;
    expect_appended(expected, 32u, 32u, 32u);
    score_cache(session, expected, scratch, "the block lands and the 24 earlier positions "
                                            "are still exact");

    /* A range that crosses the face row group boundary — the four-run case. */
    if (append_once(session, 78u, 14u, 6u, "a 6-token range across rows 14..19 appends") !=
        FT_TT_OK)
        return false;
    expect_appended(expected, 78u, 14u, 6u);
    score_cache(session, expected, scratch, "a range crossing the face row boundary lands");
    return true;
}

/* ================================================================
 * Stage 3 — the gather
 * ================================================================ */

static bool stage_3_embedding(const ft_tt_hardware_session *session)
{
    static const uint32_t ids[EMBEDDING_TOKENS] = { 40u, 3u, 17u, 3u, 63u, 0u, 55u, 8u };
    ft_tt_op_embedding_request request;
    uint32_t words[FT_TT_OP_MEMORY_WORDS];
    const uint32_t output_row0 = 3u;
    const size_t table_bytes = (size_t)EMBEDDING_VOCABULARY * EMBEDDING_ROW_BYTES;
    const size_t output_bytes = (size_t)EMBEDDING_TILES * TILE_BYTES;
    unsigned char *table = malloc(table_bytes);
    unsigned char *expected = malloc(output_bytes);
    unsigned char *actual = malloc(output_bytes);
    uint32_t row;
    uint32_t token;
    size_t wrong = 0;
    size_t index;
    bool ok = false;
    int rc;

    ft_tt_test_step("stage 3 — EMBEDDING: a permuted gather with a repeat, 1536 wide");

    if (!table || !expected || !actual) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }

    /* A ROW-MAJOR table: row `id` is one contiguous 3072-byte read. A tilized table could
     * not be gathered from at all — its rows are 32-byte face rows and a 32-byte DRAM READ
     * is not legal (64 B, ft_tt_noc_alignment()). */
    for (row = 0; row < EMBEDDING_VOCABULARY; row++) {
        uint32_t column;

        for (column = 0; column < EMBEDDING_DIM; column++) {
            const uint16_t value = source_pattern(0x454Du, row, column);

            memcpy(table + (size_t)row * EMBEDDING_ROW_BYTES + (size_t)column * 2u, &value,
                   sizeof(value));
        }
    }
    rc = device_write(&session->noc, g_dram[DRAM_BANK_SOURCE], TABLE_BASE, table, table_bytes);
    if (rc == FT_TT_OK)
        rc = poison_region(&session->noc, g_dram[DRAM_BANK_OUTPUT], EMBEDDING_BASE,
                           output_bytes);
    ft_tt_test_check_fmt("the table uploads and the output poisons", rc == FT_TT_OK, "rc=%d",
                         rc);
    if (rc != FT_TT_OK)
        goto done;

    /* The ids go into the index circular buffer's page, on the CORE — derived from the
     * placement, never remembered, so they cannot be written where the kernel does not
     * look. */
    rc = session->noc.write(session->noc.user, g_core_x, g_core_y, (uint64_t)g_index_address,
                            ids, sizeof(ids));
    ft_tt_test_check_fmt("the token ids reach the index circular buffer", rc == FT_TT_OK,
                         "0x%X rc=%d", g_index_address, rc);
    if (rc != FT_TT_OK)
        goto done;

    memset(expected, 0, output_bytes);
    for (token = 0; token < EMBEDDING_TOKENS; token++) {
        uint32_t column;

        for (column = 0; column < EMBEDDING_DIM; column++) {
            const size_t at = (size_t)(column / TILE_DIM) * TILE_BYTES +
                              (size_t)ft_tt_tile_face_index(output_row0 + token,
                                                            column % TILE_DIM) *
                                  2u;
            const uint16_t value = source_pattern(0x454Du, ids[token], column);

            memcpy(expected + at, &value, sizeof(value));
        }
    }

    memset(&request, 0, sizeof(request));
    request.token_count = EMBEDDING_TOKENS;
    request.embedding_dim = EMBEDDING_DIM;
    request.table.x = (uint16_t)g_dram[DRAM_BANK_SOURCE].x;
    request.table.y = (uint16_t)g_dram[DRAM_BANK_SOURCE].y;
    request.table.address = TABLE_BASE;
    request.table.tile_bytes = TILE_BYTES;
    request.table_row_stride = EMBEDDING_ROW_BYTES;
    request.table_row_count = EMBEDDING_VOCABULARY;
    request.output.x = (uint16_t)g_dram[DRAM_BANK_OUTPUT].x;
    request.output.y = (uint16_t)g_dram[DRAM_BANK_OUTPUT].y;
    request.output.address = EMBEDDING_BASE;
    request.output.tile_bytes = TILE_BYTES;
    request.output_bytes = (uint32_t)output_bytes;
    request.output_row0 = output_row0;
    request.zero_fill = true;
    rc = ft_tt_op_embedding_parameters(&request, words);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the descriptor encodes", false, "rc=%d", rc);
        goto done;
    }
    if (!submit(session, FT_TT_QUEUE_OP_EMBEDDING, words,
                "the gather completes on the SAME residency as the appends"))
        goto done;

    rc = device_read(&session->noc, g_dram[DRAM_BANK_OUTPUT], EMBEDDING_BASE, actual,
                     output_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the output reads back", false, "rc=%d", rc);
        goto done;
    }
    for (index = 0; index < output_bytes; index++) {
        if (actual[index] != expected[index])
            wrong++;
    }
    ft_tt_test_check_fmt("eight permuted rows land byte for byte, and the rest is zero",
                         wrong == 0u, "%zu of %zu bytes differ", wrong, output_bytes);
    ok = wrong == 0u;

done:
    free(table);
    free(expected);
    free(actual);
    return ok;
}

/* ================================================================
 * Stage 4 — an opcode this binary does not implement is REFUSED
 * ================================================================ */

static void stage_4_unknown_opcode_is_refused(const ft_tt_hardware_session *session)
{
    ft_tt_queue_descriptor descriptor;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    uint32_t completed = 0u;

    ft_tt_test_step("stage 4 — the memory binary REFUSES an opcode it does not implement");

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_MATMUL;
    descriptor.a0 = FT_TT_QUEUE_PARAM_ADDRESS(g_queue.control_address, 0u);
    (void)ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);

    /* It must NOT complete. A short timeout is the right instrument: the expectation is a
     * refusal, so waiting the full run timeout would only make the test slow. */
    ft_tt_test_check_fmt(
        "it does not retire the descriptor",
        ft_tt_queue_wait_completed(&session->noc, &g_queue, g_slot + 1u, 1000u, &completed) !=
                FT_TT_OK &&
            completed == g_slot,
        "completed=%u of %u", completed, g_slot + 1u);
    (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
    ft_tt_test_check_fmt("and it names the reason",
                         fields[FT_TT_QUEUE_FIELD_ERROR] ==
                             (uint32_t)FT_TT_QUEUE_KERR_BAD_OPCODE,
                         "error=%u (%s)", fields[FT_TT_QUEUE_FIELD_ERROR],
                         ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
}

/* ================================================================ */

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    unsigned char *expected = NULL;
    unsigned char *scratch = NULL;
    uint16_t x = 0;
    uint16_t y = 0;
    uint32_t tensix_index = 0;
    uint32_t bank;
    uint64_t mask = 0;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <memory-program-dir>\n", argv[0]);
        return 2;
    }

    ft_tt_test_step("Phase 6 memory ops — ON-CARD acceptance");

    rc = ft_tt_kv_cache_layout_init(KV_HEADS, KV_HEAD_DIM, KV_CAPACITY, TILE_BYTES, &g_cache);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the cache layout derives", false, "rc=%d", rc);
        return 1;
    }

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

    /* DM0 and DM1 ONLY: the program directory's mask IS the enables mask, so a two-processor
     * program is expressed here and nowhere else. */
    rc = ft_tt_program_directory_load(argv[1], MEMORY_MASK, &g_program);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the program directory loads (brisc + ncrisc only)", false,
                             "rc=%d", rc);
        goto release;
    }

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
    g_group.resident_processor_mask = MEMORY_MASK;

    g_core_x = x;
    g_core_y = y;
    rc = ft_tt_core_boot_firmware(&session.noc, x, y, &firmware, NULL);
    ft_tt_test_check_fmt("firmware boots on the target core", rc == FT_TT_OK,
                         "index %u -> (%u, %u) rc=%d", tensix_index, x, y, rc);
    if (rc != FT_TT_OK)
        goto release_programs;

    expected = malloc((size_t)g_cache.region_bytes);
    scratch = malloc((size_t)KV_SOURCE_TILES * TILE_BYTES * 2u > g_cache.region_bytes
                         ? (size_t)KV_SOURCE_TILES * TILE_BYTES * 2u
                         : (size_t)g_cache.region_bytes);
    if (!expected || !scratch) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto release_programs;
    }

    stage_0_round_trip(&session);
    rc = upload_kv_source(&session, scratch);
    ft_tt_test_check_fmt("the KV projection output uploads", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto release_programs;

    /* ONE residency for both opcodes, and ONE OUTPUT CB PER GROUP SIZE. Sizing a single
     * output CB for the widest group and running both group widths through it OVERRUNS L1 —
     * measured here, by this suite, at stage 3: 27 KV descriptors of 2 x 4 tiles leave the
     * write pointer 24 pages into a 96-page buffer and the 48-page embedding block is
     * refused. ft_tt_op_memory.h has the arithmetic showing no host rule about group sizes
     * can fix that. The two CBs make it unreachable instead. */
    rc = load_images(&session.noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session.noc, x, y, &g_program.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_memory_place_cbs(EMBEDDING_TILES, KV_DIM_TILES, TILE_BYTES, g_cb_placed,
                                       g_cb_buffer, sizeof(g_cb_buffer), &mask);
    if (rc == FT_TT_OK) {
        g_index_address = ft_tt_op_memory_index_address(g_cb_placed, FT_TT_OP_MEMORY_CB_COUNT);
        g_config.local_cb_mask = mask;
        g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
        g_config.brisc_noc_id = 0u;
        rc = ft_tt_queue_init(&session.noc, x, y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                              ft_tt_queue_make_nonce(), &g_queue);
    }
    if (rc == FT_TT_OK)
        rc = start_pair(&session.noc, x, y);
    ft_tt_test_check_fmt("the resident READER+WRITER pair is staged and kicked ONCE, with no "
                         "compute thread",
                         rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        goto release_programs;

    if (stage_1_single_position(&session, expected, scratch) &&
        stage_2_append_shapes(&session, expected, scratch) && stage_3_embedding(&session)) {
        ft_tt_test_check_fmt("both opcodes ran on ONE image load", true, "%u descriptors",
                             g_slot);
        stage_4_unknown_opcode_is_refused(&session);
    }

    retire_pair(&session.noc, "both processors retire when ASKED, with no reset and no park");

    /* DO NOT PARK — see the file header. The core is left in its firmware's GO wait, which
     * is where a healthy idle core sits. */

release_programs:
    free(expected);
    free(scratch);
    ft_tt_program_directory_release(&g_program);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
