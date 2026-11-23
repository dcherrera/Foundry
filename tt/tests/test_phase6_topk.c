/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_topk.c
 * @brief ON-CARD acceptance for the sampling op pair — exact top-N over a vocabulary logit
 *   row and the token draw that consumes it, on one resident program.
 *
 * ---------------------------------------------------------------------------
 * THE GATE IS EXACT INDEX AGREEMENT, NOT PCC — AND NOT BY PREFERENCE
 * ---------------------------------------------------------------------------
 * Every other Phase 6 op is accepted at PCC ≥ 0.99 against a CPU fp32 reference. That gate
 * is BLIND to this op's failure mode: top-k returns INDICES, and a result whose N values
 * are all correct while its N-th index resolved a tie the other way correlates at 1.0 with
 * the truth and makes the model emit a different token. So every check here is an integer
 * comparison — pair for pair, token for token.
 *
 * bf16 has EIGHT mantissa bits against MiniCPM5-1B's 130,560 vocabulary, so ties are the
 * common case and not an edge one. The rows stage 2 builds plant exact ties, adjacent-
 * representable near-ties and an all-identical plateau AT the selection boundary, because
 * that is the only place a reference and an implementation disagree without either looking
 * wrong.
 *
 * ---------------------------------------------------------------------------
 * THE DECOY ROWS, AND WHY THEY ARE THE MOST IMPORTANT INPUT HERE
 * ---------------------------------------------------------------------------
 * The LM head runs at Mt = 1, so it writes 32 token rows and decode selects over ONE. A
 * tile is four 16x16 faces, so that row is TWO runs of 16 elements 512 bytes apart, not one
 * run of 32. A kernel that got either the row or the face
 * split wrong would still return real logits with plausible values — the failure is
 * invisible to any value metric. Every row this test uploads therefore fills the OTHER 31
 * token rows with values that would WIN if they were read, so a row/face bug is a wrong
 * INDEX and a red check rather than a slightly different answer.
 *
 * ---------------------------------------------------------------------------
 * THE COST SPLIT IS MEASURED, NOT INFERRED
 * ---------------------------------------------------------------------------
 * Stage 3 runs the full MiniCPM5-1B row twice, interleaved, differing only by a RUNTIME
 * FLAG: the full arm and a drain-only arm that pops every logit block without selecting
 * over it. Same binary, same descriptors, same DRAM traffic — so the difference IS the
 * scalar selection's cost and the drain arm IS the DRAM stream's. Interleaved because two
 * 1.8 GHz cores also driving eight K80s drift enough over ten seconds to swamp the
 * difference (dev notes §8).
 *
 * ---------------------------------------------------------------------------
 * SAFETY
 * ---------------------------------------------------------------------------
 * The resident program is retired by ASKING (`ft_tt_watchdog_shutdown`). It is never
 * parked and never reset: parking a resident triad breaks the core's NEXT boot even after
 * a textbook-clean retirement (dev notes §1, cores 60 and 65).
 *
 * ---------------------------------------------------------------------------
 * AND THE SUITE SCORES THE CORE, NOT ONLY THE ANSWER — stage 7
 * ---------------------------------------------------------------------------
 * Every stage below stage 7 grades an OUTPUT. That is not enough and this suite is the proof:
 * it reported `0 checks failed`, retired its residency with `unstopped=0x0`, and left the
 * Tensix it ran on unable to boot — four cores, before anyone noticed, because a wedged core
 * READS exactly like a healthy idle one (bringup log §21.1). Stage 7 therefore ends the run
 * by BOOTING the core again and running a fresh residency on it. It runs unconditionally,
 * including after an earlier stage has failed.
 *
 * Usage:
 *   FT_TT_FIRMWARE_DIR=<program-dir> FT_TT_TENSIX_INDEX=<n> \
 *     ./test_phase6_topk <program-dir>
 *
 * @status 40 checks, 0 failures on Tensix 51, 57 and 59. Five kernel mutations were driven
 *   through THIS binary unmodified and all five go red: the tie rule (8 failures), the face
 *   split (10), the token row (10), the accept interval (3), and the repetition penalty
 *   moved before the greedy argmax (1). That last one came back GREEN the first time and is
 *   why stage 4 now takes a SECOND greedy draw — see the note there.
 *   Stage 7 (the core still boots and still computes) added 2026-08-13 and shown RED against
 *   the pre-fix kernel, which it names with the reset register.
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
#include "../ft_tt_op_sample.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"
#include "ft_tt_topk_reference.h"

#define TILE_BYTES     FT_TT_OP_TILE_BYTES_BF16
#define TILE_DIM       FT_TT_OP_TILE_DIM
#define RUN_TIMEOUT_MS 60000u
#define QUEUE_CAPACITY 64u
#define POISON_WORD    0xA5A5A5A5u

/** DRAM banks 6 and 7 are this suite's; 0-5 belong to the other Phase 6 workers. */
#define DRAM_BANK_LOGITS 6u
#define DRAM_BANK_TOKEN  7u
#define DRAM_BANK_COUNT  8u
#define DRAM_BUFFER_BASE 0x00100000u
#define DRAM_TOKEN_BASE  0x00200000u

/** Every processor of the program is resident, including the deliberately idle TRISCs —
 *  they follow the queue so the stop protocol is the normal one. */
#define TRIAD_MASK                                                                \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) | \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

/** The small shape stage 2 scores on: 5120 entries, 160 tiles. Big enough that the
 *  selection is not trivially the whole row, small enough that a case costs a second — and
 *  160 is a whole number of PLACEMENT_DEPTH*2 = 80-page buffer cycles, which is the
 *  invariant a resident reader actually needs and the one that overwrote L1 when this test
 *  used 128 (see ft_tt_op_sample.h). 4080 is 51 of the same cycles. */
#define SMALL_VOCAB 5120u
#define SMALL_TILES (SMALL_VOCAB / 32u)

/** The real shape: MiniCPM5-1B's vocabulary, MEASURED from the production artifact. */
#define FULL_VOCAB 130560u
#define FULL_TILES (FULL_VOCAB / 32u)

/** Which of a tile's 32 rows carries the decoded token. Deliberately NOT 0: row 0 is the
 *  one a face/row bug is most likely to land on by accident. */
#define TOKEN_ROW 13u

#define SELECT_COUNT 40u

/** The depth the CBs are PLACED at, once, for the whole residency, giving a 2 x 40 = 80-page
 *  buffer. Every depth this test uses (40, 20, 8, 1) divides 80 AND divides both tile counts,
 *  and 80 divides BOTH tile counts (160 and 4080) so every descriptor hands the FIFO write
 *  pointer back where it found it. That last clause is the one the kernel model does not
 *  state and the one that cost this test a run — ft_tt_op_sample.h carries the derivation. */
#define PLACEMENT_DEPTH 40u

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_program;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
static uint32_t g_result_address;
static uint32_t g_state_address;
static uint32_t g_descriptor_slot;

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

/** L1 on the target core. The result and state blocks live there, so every read of them
 *  goes through the core's own coordinate rather than a DRAM bank's. */
static ft_tt_noc_coord core_coord(uint16_t x, uint16_t y)
{
    ft_tt_noc_coord where;

    where.x = x;
    where.y = y;
    return where;
}

static double elapsed_ms(const struct timespec *start, const struct timespec *end)
{
    return (double)(end->tv_sec - start->tv_sec) * 1000.0 +
           (double)(end->tv_nsec - start->tv_nsec) / 1.0e6;
}

/* ================================================================
 * Bringing the resident program up
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

static int start_program(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
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

/** NEVER a reset and NEVER a park — dev notes §1. */
static void retire_program(const ft_tt_noc_access *noc)
{
    uint32_t unstopped = 0u;
    int rc = ft_tt_watchdog_shutdown(noc, &g_group, RUN_TIMEOUT_MS, &unstopped);

    ft_tt_test_check_fmt("the resident program retires by ASKING", rc == FT_TT_OK &&
                                                                      unstopped == 0u,
                         "rc=%d unstopped=0x%X", rc, unstopped);
}

/* ================================================================
 * Enqueue and wait
 * ================================================================ */

/**
 * Enqueue one descriptor and wait for the WRITER to publish its completion.
 *
 * The completion check is `ft_tt_queue_check_completion`, which agrees four things — count,
 * token, state and error — because a count alone is exactly the lie Phase 5 was built
 * against (dev notes §3, "group completion is not evidence that a group ran").
 */
static bool run_descriptor(const ft_tt_hardware_session *session, uint32_t opcode,
                           const ft_tt_op_topk_request *request, double *out_ms)
{
    uint32_t words[FT_TT_OP_SAMPLE_WORDS];
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    ft_tt_queue_descriptor descriptor;
    uint32_t completed = 0u;
    uint32_t parameter_address = 0u;
    const char *reason = NULL;
    struct timespec start;
    struct timespec finish;
    uint32_t slot = g_descriptor_slot;
    int rc;

    rc = ft_tt_op_sample_parameters(request, words);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the parameter block encodes", false, "rc=%d", rc);
        return false;
    }
    rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, slot % 64u, words,
                                         &parameter_address);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the parameter block uploads", false, "rc=%d", rc);
        return false;
    }

    /* The token is stamped by ft_tt_queue_enqueue() from the run nonce, deliberately — a
     * caller that supplied its own could check the token it was TOLD rather than the one
     * that was SENT. */
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = opcode;
    descriptor.a0 = parameter_address;

    clock_gettime(CLOCK_MONOTONIC, &start);
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    if (out_ms)
        *out_ms = elapsed_ms(&start, &finish);
    if (rc != FT_TT_OK) {
        (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
        ft_tt_test_check_fmt("the descriptor completes", false,
                             "rc=%d completed=%u state=0x%08X error=%s", rc, completed,
                             fields[FT_TT_QUEUE_FIELD_STATE],
                             ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
        return false;
    }
    (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
    if (ft_tt_queue_check_completion(&g_queue, fields, slot + 1u, &reason) != FT_TT_OK) {
        /* Print the words, not just the verdict. A completion check that reports WHICH of
         * its four grounds failed but not the values it read is a failure you have to re-run
         * to diagnose, and re-running costs a core. */
        ft_tt_test_check_fmt("the completion agrees on all four words", false,
                             "%s — expected %u, count %u, token 0x%08X (want 0x%08X), "
                             "state 0x%08X, error %s",
                             reason ? reason : "unknown", slot + 1u,
                             fields[FT_TT_QUEUE_FIELD_COMPLETED_COUNT],
                             fields[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN],
                             ft_tt_queue_token_at(&g_queue, slot),
                             fields[FT_TT_QUEUE_FIELD_STATE],
                             ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
        return false;
    }
    g_descriptor_slot = slot + 1u;
    return true;
}

/* ================================================================
 * Building and uploading a logit row
 * ================================================================ */

static uint32_t g_random_state = 0x9E3779B9u;

static uint32_t next_random(void)
{
    g_random_state ^= g_random_state << 13;
    g_random_state ^= g_random_state >> 17;
    g_random_state ^= g_random_state << 5;
    return g_random_state;
}

static uint16_t random_logit(void)
{
    float value = ((float)(next_random() >> 8) / 16777216.0f) * 40.0f - 20.0f;

    return ft_bf16_from_float(value);
}

/**
 * Tilize a row and upload it, with every OTHER token row filled with a value that would
 * beat every real logit if the kernel read the wrong row or the wrong face.
 *
 * The filler is +1000.0f and the logits are in [-20, 20), so a decoy that leaks into the
 * selection is not a subtle score change — it is the WHOLE answer, and the indices come
 * back wrong. That is the point: a row bug must be a red check, not a plausible number.
 */
static int upload_row(const ft_tt_hardware_session *session, const uint16_t *row,
                      uint32_t vocab, uint8_t *scratch)
{
    size_t bytes = (size_t)(vocab / 32u) * TILE_BYTES;
    int rc = ft_tt_topk_reference_tilize_row(row, vocab, TOKEN_ROW,
                                             ft_bf16_from_float(1000.0f), scratch, bytes);

    if (rc != 0)
        return FT_TT_EINVAL;
    return device_write(&session->noc, g_dram[DRAM_BANK_LOGITS], DRAM_BUFFER_BASE, scratch,
                        bytes);
}

static ft_tt_op_topk_request base_request(uint32_t vocab, uint32_t select_count,
                                          uint32_t prefetch, uint32_t flags)
{
    ft_tt_op_topk_request request;

    memset(&request, 0, sizeof(request));
    request.tile_count = vocab / 32u;
    request.row = TOKEN_ROW;
    request.select_count = select_count;
    request.prefetch_tiles = prefetch;
    request.flags = flags;
    request.logits.x = (uint16_t)g_dram[DRAM_BANK_LOGITS].x;
    request.logits.y = (uint16_t)g_dram[DRAM_BANK_LOGITS].y;
    request.logits.address = DRAM_BUFFER_BASE;
    request.logits.tile_bytes = TILE_BYTES;
    request.result_address = g_result_address;
    request.state_address = g_state_address;
    return request;
}

/** Poison the result block, so "the kernel never wrote it" and "the kernel answered" are
 *  different bit patterns rather than the same stale one. */
static int poison_result(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    uint32_t poison[FT_TT_OP_SAMPLE_RESULT_BYTES / 4u];
    size_t index;

    for (index = 0; index < sizeof(poison) / sizeof(poison[0]); index++)
        poison[index] = POISON_WORD;
    return device_write(&session->noc, core_coord(x, y), g_result_address, poison,
                        sizeof(poison));
}

/**
 * Run TOPK over `row` and compare the card's answer, pair for pair, against the reference.
 *
 * @return true when every pair matched
 */
static bool score_topk(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                       const char *label, const uint16_t *row, uint32_t vocab,
                       uint32_t select_count, uint32_t prefetch, uint8_t *scratch,
                       double *out_ms)
{
    uint32_t block[FT_TT_OP_SAMPLE_RESULT_BYTES / 4u];
    uint32_t card_indices[FT_TT_OP_SAMPLE_MAX_SELECT];
    float card_values[FT_TT_OP_SAMPLE_MAX_SELECT];
    uint32_t reference_indices[FT_TT_OP_SAMPLE_MAX_SELECT];
    float reference_values[FT_TT_OP_SAMPLE_MAX_SELECT];
    ft_tt_op_topk_request request;
    uint32_t count = 0u;
    uint32_t first_bad = select_count;
    uint32_t entry;
    int rc;

    if (row) {
        uint32_t before[FT_TT_QUEUE_FIELD_COUNT];
        uint32_t after[FT_TT_QUEUE_FIELD_COUNT];

        /* Read the queue control block ACROSS the upload. A bulk DRAM write and the resident
         * kernel's own control words share nothing by design, so this is not a suspicion —
         * it is the instrument that says WHICH link failed instead of leaving a corrupted
         * word to be attributed to the kernel (dev notes §3: a harness that cannot say which
         * link failed will point at the wrong one). */
        (void)ft_tt_queue_read_control(&session->noc, &g_queue, before);
        rc = upload_row(session, row, vocab, scratch);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt(label, false, "the row failed to upload, rc=%d", rc);
            return false;
        }
        (void)ft_tt_queue_read_control(&session->noc, &g_queue, after);
        if (memcmp(before, after, sizeof(before)) != 0)
            ft_tt_test_check_fmt("the row upload did NOT disturb the queue control block",
                                 false,
                                 "%s: count %u -> %u, token 0x%08X -> 0x%08X, head %u -> %u",
                                 label, before[FT_TT_QUEUE_FIELD_COMPLETED_COUNT],
                                 after[FT_TT_QUEUE_FIELD_COMPLETED_COUNT],
                                 before[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN],
                                 after[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN],
                                 before[FT_TT_QUEUE_FIELD_HEAD], after[FT_TT_QUEUE_FIELD_HEAD]);
    }
    if (poison_result(session, x, y) != FT_TT_OK) {
        ft_tt_test_check(label, false, "the result block failed to poison");
        return false;
    }

    request = base_request(vocab, select_count, prefetch, 0u);
    if (!run_descriptor(session, FT_TT_QUEUE_OP_TOPK, &request, out_ms))
        return false;

    rc = device_read(&session->noc, core_coord(x, y), g_result_address, block, sizeof(block));
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "the result block failed to read, rc=%d", rc);
        return false;
    }
    if (block[FT_TT_OP_SAMPLE_RESULT_MAGIC] == POISON_WORD) {
        ft_tt_test_check(label, false, "the poison survived — the kernel never wrote the block");
        return false;
    }
    rc = ft_tt_op_sample_result_decode(block, FT_TT_OP_SAMPLE_MAX_SELECT, card_indices,
                                       card_values, &count, NULL);
    if (rc != FT_TT_OK || count != select_count) {
        ft_tt_test_check_fmt(label, false, "decode rc=%d, count %u of %u", rc, count,
                             select_count);
        return false;
    }
    if (ft_tt_topk_reference_select(row, vocab, select_count, reference_indices,
                                    reference_values) != 0) {
        ft_tt_test_check(label, false, "the reference refused the row");
        return false;
    }
    for (entry = 0u; entry < select_count; entry++) {
        if (card_indices[entry] != reference_indices[entry] ||
            card_values[entry] != reference_values[entry]) {
            first_bad = entry;
            break;
        }
    }
    if (first_bad == select_count) {
        ft_tt_test_check_fmt(label, true, "%u pairs EXACT, top id %u = %.6g", select_count,
                             card_indices[0], (double)card_values[0]);
        return true;
    }
    ft_tt_test_check_fmt(label, false, "pair %u: card (%u, %.9g), reference (%u, %.9g)",
                         first_bad, card_indices[first_bad], (double)card_values[first_bad],
                         reference_indices[first_bad], (double)reference_values[first_bad]);
    return false;
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

    ft_tt_test_step("stage 0 — host -> DRAM -> host, so the transfer path is not a suspect");
    for (index = 0; index < TILE_DIM * TILE_DIM; index++)
        written[index] = (uint16_t)(0x3F00u + (index & 0x7Fu));
    memset(read_back, 0, sizeof(read_back));

    rc = device_write(&session->noc, g_dram[DRAM_BANK_LOGITS], DRAM_BUFFER_BASE, written,
                      sizeof(written));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_dram[DRAM_BANK_LOGITS], DRAM_BUFFER_BASE, read_back,
                         sizeof(read_back));
    ft_tt_test_check_fmt("a tile round-trips through DRAM byte-exact",
                         rc == FT_TT_OK && memcmp(written, read_back, sizeof(written)) == 0,
                         "rc=%d", rc);
}

/* ================================================================
 * Stage 1 — bring the resident program up
 * ================================================================ */

/**
 * Bring the resident program up: images, CB placement, queue, go.
 *
 * Separate from the stage that announces it because stage 7 starts a SECOND residency on the
 * same core, and a step header there would claim to be stage 1.
 */
static bool start_residency(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    ft_tt_op_topk_request placement = base_request(FULL_VOCAB, SELECT_COUNT, PLACEMENT_DEPTH, 0u);
    const ft_tt_launch_cb_entry *result_entry;
    const ft_tt_launch_cb_entry *state_entry;
    uint64_t mask = 0u;
    int rc;

    /* The placement is the FULL shape's, so one descriptor array serves every stage —
     * which is what "one binary, one residency" means at the host end too. */
    placement.result_address = 1u;
    placement.state_address = 1u;

    rc = load_images(&session->noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, x, y, &g_program.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_sample_place_cbs(&placement, g_cb_placed, g_cb_buffer,
                                       sizeof(g_cb_buffer), &mask);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("images load and buffers place", false, "rc=%d", rc);
        return false;
    }

    result_entry = ft_tt_op_cb_find(g_cb_placed, FT_TT_OP_SAMPLE_CB_COUNT, 1u);
    state_entry = ft_tt_op_cb_find(g_cb_placed, FT_TT_OP_SAMPLE_CB_COUNT, 2u);
    if (!result_entry || !state_entry) {
        ft_tt_test_check("the result and state blocks were placed", false, "missing entry");
        return false;
    }
    g_result_address = result_entry->address;
    g_state_address = state_entry->address;
    ft_tt_test_check_fmt("the two L1 blocks are placed by the arena, not hardcoded", true,
                         "result 0x%X, state 0x%X", g_result_address, g_state_address);

    g_config.local_cb_mask = mask;
    g_config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    g_config.brisc_noc_id = 0u;
    rc = ft_tt_queue_init(&session->noc, x, y, FT_TT_OP_QUEUE_BASE, QUEUE_CAPACITY,
                          ft_tt_queue_make_nonce(), &g_queue);
    if (rc == FT_TT_OK)
        rc = start_program(&session->noc, x, y);
    ft_tt_test_check_fmt("the program is staged and kicked", rc == FT_TT_OK, "rc=%d", rc);
    return rc == FT_TT_OK;
}

static bool stage_1_start(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    ft_tt_test_step("stage 1 — the resident TOPK/SAMPLE program, staged and kicked ONCE");
    return start_residency(session, x, y);
}

/* ================================================================
 * Stage 2 — exactness, on the cases that break a top-k
 * ================================================================ */

static bool stage_2_exactness(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                              uint16_t *row, uint8_t *scratch)
{
    uint32_t index;
    bool ok = true;

    ft_tt_test_step("stage 2 — EXACT index agreement, including the tie cases");

    for (index = 0u; index < SMALL_VOCAB; index++)
        row[index] = random_logit();
    ok = score_topk(session, x, y, "random row, N = 40", row, SMALL_VOCAB, SELECT_COUNT, 20u,
                    scratch, NULL) &&
         ok;
    ok = score_topk(session, x, y, "the same row, N = 1 (greedy's shape)", row, SMALL_VOCAB,
                    1u, 20u, scratch, NULL) &&
         ok;
    ok = score_topk(session, x, y, "the same row, N = 64 (the block's capacity)", row,
                    SMALL_VOCAB, 64u, 20u, scratch, NULL) &&
         ok;

    /* The prefetch depth must not change the ANSWER, only the transaction count. Depth 1 is
     * the per-tile barrier; a block-loop bug that straddles the CB wrap shows up as a wrong
     * index at one depth and not another. Every depth here DIVIDES the placed page count
     * (PLACEMENT_DEPTH * 2) and is no larger than it — the invariant a resident op makes
     * easy to break, and the one that cost this test its first hardware run. */
    ok = score_topk(session, x, y, "the same row at depth 1", row, SMALL_VOCAB, SELECT_COUNT,
                    1u, scratch, NULL) &&
         ok;
    ok = score_topk(session, x, y, "the same row at depth 8", row, SMALL_VOCAB, SELECT_COUNT,
                    8u, scratch, NULL) &&
         ok;

    /* A 20-wide plateau straddling N = 10: the answer is decided ENTIRELY by the tie rule. */
    for (index = 0u; index < SMALL_VOCAB; index++)
        row[index] = ft_bf16_from_float(1.0f);
    for (index = 0u; index < 20u; index++)
        row[index * 173u % SMALL_VOCAB] = ft_bf16_from_float(3.5f);
    ok = score_topk(session, x, y, "a 20-wide tie plateau straddling N = 10", row, SMALL_VOCAB,
                    10u, 20u, scratch, NULL) &&
         ok;

    /* Adjacent representable bf16 values, which is where a reference that computed in fp32
     * and an implementation that compared bits disagree by exactly one index. */
    for (index = 0u; index < SMALL_VOCAB; index++)
        row[index] = ft_bf16_from_float(1.0f);
    for (index = 0u; index < 24u; index++) {
        uint16_t base = ft_bf16_from_float(7.0f);

        row[index * 191u % SMALL_VOCAB] = (uint16_t)(base + (uint16_t)(index % 3u) - 1u);
    }
    ok = score_topk(session, x, y, "adjacent-representable near-ties at N = 12", row,
                    SMALL_VOCAB, 12u, 20u, scratch, NULL) &&
         ok;

    /* Every value identical: the only determinate answer is indices 0..N-1, and a kernel
     * that inserted on >= instead of > returns the LAST N instead. */
    for (index = 0u; index < SMALL_VOCAB; index++)
        row[index] = ft_bf16_from_float(2.0f);
    ok = score_topk(session, x, y, "an all-identical row at N = 8", row, SMALL_VOCAB, 8u, 20u,
                    scratch, NULL) &&
         ok;

    /* A negative plateau, because the accept interval takes a DIFFERENT branch for a
     * negative incumbent and every case above leaves it on the positive one. */
    for (index = 0u; index < SMALL_VOCAB; index++)
        row[index] = ft_bf16_from_float(-9.0f);
    for (index = 0u; index < 16u; index++)
        row[index * 211u % SMALL_VOCAB] = ft_bf16_from_float(-2.5f);
    ok = score_topk(session, x, y, "an all-negative row with a plateau at N = 8", row,
                    SMALL_VOCAB, 8u, 20u, scratch, NULL) &&
         ok;

    /* The winners at the very ends of the row, so an off-by-one in the block loop or in the
     * face arithmetic drops one of them. */
    for (index = 0u; index < SMALL_VOCAB; index++)
        row[index] = ft_bf16_from_float(-5.0f);
    row[0] = ft_bf16_from_float(9.0f);
    row[SMALL_VOCAB - 1u] = ft_bf16_from_float(8.0f);
    row[31] = ft_bf16_from_float(7.0f);
    row[32] = ft_bf16_from_float(6.0f);
    ok = score_topk(session, x, y, "winners at both ends and at a tile boundary", row,
                    SMALL_VOCAB, 4u, 20u, scratch, NULL) &&
         ok;

    return ok;
}

/* ================================================================
 * Stage 3 — the real shape, and the cost split
 * ================================================================ */

static bool stage_3_full_row(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                             uint16_t *row, uint8_t *scratch)
{
    double full_ms[4];
    double drain_ms[4];
    double full_total = 0.0;
    double drain_total = 0.0;
    uint32_t index;
    uint32_t arm;
    bool ok;

    ft_tt_test_step("stage 3 — MiniCPM5-1B's real row: 130,560 logits, 4080 tiles, 8.36 MB");

    for (index = 0u; index < FULL_VOCAB; index++)
        row[index] = random_logit();
    ok = score_topk(session, x, y, "the full vocabulary row, N = 40, EXACT", row, FULL_VOCAB,
                    SELECT_COUNT, PLACEMENT_DEPTH, scratch, &full_ms[0]);
    if (!ok)
        return false;

    /* The cost split. Interleaved arms, same binary, same descriptors, differing only by a
     * RUNTIME flag — the drain arm pops every block without selecting, so full minus drain
     * IS the scalar selection and drain IS the DRAM stream. Interleaved because two 1.8 GHz
     * cores also driving eight K80s drift enough over ten seconds to swamp the difference
     * (dev notes §8, which once reported a NEGATIVE cost for CONFIGURE_TLB). */
    for (arm = 0u; arm < 4u; arm++) {
        ft_tt_op_topk_request request = base_request(FULL_VOCAB, SELECT_COUNT, PLACEMENT_DEPTH, 0u);

        if (!run_descriptor(session, FT_TT_QUEUE_OP_TOPK, &request, &full_ms[arm]))
            return false;
        request = base_request(FULL_VOCAB, SELECT_COUNT, PLACEMENT_DEPTH,
                               FT_TT_OP_SAMPLE_FLAG_DRAIN_ONLY);
        if (!run_descriptor(session, FT_TT_QUEUE_OP_TOPK, &request, &drain_ms[arm]))
            return false;
        full_total += full_ms[arm];
        drain_total += drain_ms[arm];
    }
    ft_tt_test_check_fmt("the full row selects in a time worth reporting", full_total > 0.0,
                         "full %.3f ms/token, drain-only %.3f ms/token, selection %.3f ms",
                         full_total / 4.0, drain_total / 4.0,
                         (full_total - drain_total) / 4.0);

    /* The whole reason the op exists, stated as a number rather than as prose: the same row
     * fetched to the host at the MEASURED card->host rate (dev notes §6, 4.7 MB/s at 16-byte
     * loads) instead of reduced on the card. */
    ft_tt_test_check_fmt("on-card selection beats fetching the row to the host",
                         full_total / 4.0 < 55.0,
                         "on card %.3f ms, host fetch of 261,120 B at 4.7 MB/s = %.1f ms",
                         full_total / 4.0, 261120.0 / 4.7e6 * 1000.0);

    /* And the drain arm must be a DIFFERENT answer, or the flag did nothing and the split
     * above is two measurements of the same thing. */
    {
        uint32_t block[FT_TT_OP_SAMPLE_RESULT_BYTES / 4u];
        int rc = device_read(&session->noc, core_coord(x, y), g_result_address, block,
                             sizeof(block));

        ft_tt_test_check_fmt("the drain-only arm writes an EMPTY result, not an answer",
                             rc == FT_TT_OK &&
                                 block[FT_TT_OP_SAMPLE_RESULT_MAGIC] ==
                                     FT_TT_OP_SAMPLE_RESULT_MAGIC_VALUE &&
                                 block[FT_TT_OP_SAMPLE_RESULT_COUNT] == 0u,
                             "rc=%d magic 0x%08X count %u", rc,
                             block[FT_TT_OP_SAMPLE_RESULT_MAGIC],
                             block[FT_TT_OP_SAMPLE_RESULT_COUNT]);
    }
    return true;
}

/* ================================================================
 * Stage 4 — the token draw
 * ================================================================ */

static ft_tt_op_sample_params default_params(void)
{
    ft_tt_op_sample_params params;

    memset(&params, 0, sizeof(params));
    params.temperature = 0.0f;
    params.top_p = 1.0f;
    params.repeat_penalty = 1.0f;
    params.top_k = SELECT_COUNT;
    params.repeat_window = 0u;
    params.rng_seed = 0x2545F491u;
    return params;
}

static int upload_state(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                        const ft_tt_op_sample_params *params)
{
    uint32_t state[FT_TT_OP_SAMPLE_STATE_WORDS];
    int rc = ft_tt_op_sample_state_build(params, state, FT_TT_OP_SAMPLE_STATE_WORDS);

    if (rc != FT_TT_OK)
        return rc;
    return device_write(&session->noc, core_coord(x, y), g_state_address, state,
                        sizeof(state));
}

static bool stage_4_sampling(const ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                             uint16_t *row, uint8_t *scratch)
{
    uint32_t block[FT_TT_OP_SAMPLE_RESULT_BYTES / 4u];
    uint32_t state[FT_TT_OP_SAMPLE_STATE_WORDS];
    uint32_t card_indices[FT_TT_OP_SAMPLE_MAX_SELECT];
    float card_values[FT_TT_OP_SAMPLE_MAX_SELECT];
    ft_tt_op_sample_params params = default_params();
    ft_tt_op_topk_request request;
    uint32_t count = 0u;
    uint32_t card_token = 0u;
    uint32_t reference_token;
    uint32_t rng;
    uint32_t index;
    uint32_t draw;
    bool ok = true;
    int rc;

    ft_tt_test_step("stage 4 — the token draw, against Pantheon's own sparse sampler");

    for (index = 0u; index < SMALL_VOCAB; index++)
        row[index] = random_logit();
    if (!score_topk(session, x, y, "the row this stage samples from", row, SMALL_VOCAB,
                    SELECT_COUNT, 20u, scratch, NULL))
        return false;
    rc = device_read(&session->noc, core_coord(x, y), g_result_address, block, sizeof(block));
    if (rc == FT_TT_OK)
        rc = ft_tt_op_sample_result_decode(block, FT_TT_OP_SAMPLE_MAX_SELECT, card_indices,
                                           card_values, &count, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the survivors read back", false, "rc=%d", rc);
        return false;
    }

    /* GREEDY. Temperature 0, and a repetition penalty that WOULD change the answer if it
     * were applied first — which is the shipped K80 bug this ordering exists to prevent. */
    params.temperature = 0.0f;
    params.repeat_penalty = 1.5f;
    params.repeat_window = 8u;
    rc = upload_state(session, x, y, &params);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the sampler state uploads", false, "rc=%d", rc);
        return false;
    }
    request = base_request(SMALL_VOCAB, SELECT_COUNT, 20u, 0u);
    if (!run_descriptor(session, FT_TT_QUEUE_OP_SAMPLE, &request, NULL))
        return false;
    rc = device_read(&session->noc, core_coord(x, y), g_result_address, block, sizeof(block));
    card_token = block[FT_TT_OP_SAMPLE_RESULT_TOKEN];
    ft_tt_test_check_fmt("SAMPLE marks the token valid",
                         rc == FT_TT_OK && block[FT_TT_OP_SAMPLE_RESULT_TOKEN_VALID] == 1u,
                         "valid %u", block[FT_TT_OP_SAMPLE_RESULT_TOKEN_VALID]);
    rng = params.rng_seed;
    reference_token = ft_tt_topk_reference_sample(card_indices, card_values, count, &params,
                                                  NULL, 0u, &rng, 0);
    ft_tt_test_check_fmt("greedy returns the reference's token EXACTLY",
                         card_token == reference_token, "card %u, reference %u", card_token,
                         reference_token);
    /* The discriminating half: greedy must be the argmax of the RAW survivors, so it is the
     * FIRST pair of the descending-by-value result block and nothing else. */
    ft_tt_test_check_fmt("greedy is the top pair of the result block, so no penalty was applied",
                         card_token == card_indices[0], "card %u, top pair %u", card_token,
                         card_indices[0]);
    ok = ok && (card_token == reference_token) && (card_token == card_indices[0]);

    /* THE SECOND GREEDY DRAW IS THE ONE THAT CAN GO RED, AND THE FIRST ONE CANNOT.
     *
     * This was found by mutation, not by review: a build with the repetition penalty moved
     * BEFORE the greedy argmax — the exact shipped K80 bug this ordering exists to prevent —
     * passed the check above with ZERO failures. The reason is that the state was freshly
     * uploaded, so the recent-token ring was EMPTY and `apply_repeat_penalty` returned
     * without touching anything. The check was asserting an ordering whose two arms were
     * identical on that input, which is dev notes §3's "a test can be vacuous because the
     * input never exercises it" in its purest form.
     *
     * The fix is not a bigger assertion, it is the right INPUT: the first draw appended its
     * own token to the ring, so a SECOND draw now has a penalty that lands on exactly the
     * token greedy must still return. A penalty-first implementation demotes the top logit
     * by the 1.5x factor and returns the runner-up. Nothing else changed. */
    {
        uint32_t recent_after[FT_TT_OP_SAMPLE_STATE_WORDS];
        uint32_t second_token;
        uint32_t second_reference;
        uint32_t second_rng;

        rc = device_read(&session->noc, core_coord(x, y), g_state_address, recent_after,
                         sizeof(recent_after));
        ft_tt_test_check_fmt("the case is NOT vacuous: the ring now holds the top token and "
                             "the penalty is live",
                             rc == FT_TT_OK &&
                                 recent_after[FT_TT_OP_SAMPLE_STATE_RECENT_COUNT] == 1u &&
                                 recent_after[FT_TT_OP_SAMPLE_RECENT_OFFSET_WORDS] == card_token &&
                                 params.repeat_penalty != 1.0f,
                             "recent %u entries, ring[0] %u, penalty %.2f",
                             recent_after[FT_TT_OP_SAMPLE_STATE_RECENT_COUNT],
                             recent_after[FT_TT_OP_SAMPLE_RECENT_OFFSET_WORDS],
                             (double)params.repeat_penalty);

        request = base_request(SMALL_VOCAB, SELECT_COUNT, 20u, 0u);
        if (!run_descriptor(session, FT_TT_QUEUE_OP_SAMPLE, &request, NULL))
            return false;
        rc = device_read(&session->noc, core_coord(x, y), g_result_address, block,
                         sizeof(block));
        second_token = block[FT_TT_OP_SAMPLE_RESULT_TOKEN];
        second_rng = params.rng_seed;
        second_reference = ft_tt_topk_reference_sample(card_indices, card_values, count,
                                                       &params, &card_token, 1u, &second_rng,
                                                       0);
        ft_tt_test_check_fmt("greedy STILL returns the true argmax with that token penalised",
                             rc == FT_TT_OK && second_token == card_indices[0] &&
                                 second_token == second_reference,
                             "card %u, reference %u, true argmax %u", second_token,
                             second_reference, card_indices[0]);
        /* And name what a penalty-first build would have returned, so the case's
         * discriminating power is on the transcript rather than asserted in a comment. */
        ft_tt_test_check_fmt("a penalty-first build would have returned the runner-up instead",
                             card_values[0] / params.repeat_penalty < card_values[1],
                             "top %.6g penalised to %.6g, runner-up %.6g",
                             (double)card_values[0],
                             (double)(card_values[0] / params.repeat_penalty),
                             (double)card_values[1]);
        ok = ok && second_token == card_indices[0];
    }

    /* The state a closed decode loop carries: the token was appended to BOTH rings and the
     * emitted count moved, without the host doing any of it. */
    rc = device_read(&session->noc, core_coord(x, y), g_state_address, state, sizeof(state));
    ft_tt_test_check_fmt("the sampler appended the token to its own recent ring",
                         rc == FT_TT_OK && state[FT_TT_OP_SAMPLE_RECENT_OFFSET_WORDS] ==
                                               card_token,
                         "ring[0] %u, token %u", state[FT_TT_OP_SAMPLE_RECENT_OFFSET_WORDS],
                         card_token);
    /* TWO, because the non-vacuous ordering case above needed a second greedy draw and both
     * went into this ring. The ring's FIRST entry is still the first token, which is the
     * property being checked — the host drains a SEQUENCE, not a single slot. */
    ft_tt_test_check_fmt("and to the emitted ring the host drains lazily",
                         state[FT_TT_OP_SAMPLE_STATE_EMITTED_COUNT] == 2u &&
                             state[FT_TT_OP_SAMPLE_EMITTED_OFFSET_WORDS] == card_token,
                         "emitted %u, ring[0] %u", state[FT_TT_OP_SAMPLE_STATE_EMITTED_COUNT],
                         state[FT_TT_OP_SAMPLE_EMITTED_OFFSET_WORDS]);
    ft_tt_test_check_fmt("greedy did NOT advance the generator",
                         state[FT_TT_OP_SAMPLE_STATE_RNG] == params.rng_seed, "state 0x%08X",
                         state[FT_TT_OP_SAMPLE_STATE_RNG]);

    /* STOCHASTIC. Eight successive draws off the same survivor set, each compared against
     * the reference driven by the SAME generator — so this is an exact sequence match and
     * not a distribution argument. A single draw would prove nothing: any bug that permutes
     * the survivors still returns one of them. */
    params = default_params();
    params.temperature = 0.8f;
    params.top_p = 0.95f;
    params.top_k = 20u;
    rc = upload_state(session, x, y, &params);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the stochastic state uploads", false, "rc=%d", rc);
        return false;
    }
    rng = params.rng_seed;
    {
        uint32_t recent[16];
        uint32_t recent_count = 0u;
        uint32_t matched = 0u;
        uint32_t distinct = 0u;
        uint32_t first_token = 0u;

        for (draw = 0u; draw < 8u; draw++) {
            uint32_t expected;

            request = base_request(SMALL_VOCAB, SELECT_COUNT, 20u, 0u);
            if (!run_descriptor(session, FT_TT_QUEUE_OP_SAMPLE, &request, NULL))
                return false;
            rc = device_read(&session->noc, core_coord(x, y), g_result_address, block,
                             sizeof(block));
            if (rc != FT_TT_OK)
                return false;
            card_token = block[FT_TT_OP_SAMPLE_RESULT_TOKEN];
            expected = ft_tt_topk_reference_sample(card_indices, card_values, count, &params,
                                                   recent, recent_count, &rng, 0);
            if (card_token == expected)
                matched++;
            else
                ft_tt_test_check_fmt("draw disagrees", false, "draw %u: card %u, reference %u",
                                     draw, card_token, expected);
            if (draw == 0u)
                first_token = card_token;
            else if (card_token != first_token)
                distinct = 1u;
            if (recent_count < 16u)
                recent[recent_count++] = card_token;
        }
        ft_tt_test_check_fmt("eight stochastic draws match the reference sequence EXACTLY",
                             matched == 8u, "%u of 8", matched);
        /* The control that stops the check above passing for free: a sampler stuck on the
         * argmax would also "match" if the reference were stuck the same way, so the draws
         * have to be seen to VARY. */
        ft_tt_test_check_fmt("the draws are not all the same token", distinct == 1u,
                             "first token %u", first_token);
        ft_tt_test_check_fmt("the generator advanced on the card", rc == FT_TT_OK, "rc=%d", rc);
        ok = ok && matched == 8u && distinct == 1u;
    }

    rc = device_read(&session->noc, core_coord(x, y), g_state_address, state, sizeof(state));
    ft_tt_test_check_fmt("the card's generator state matches the host's after eight draws",
                         rc == FT_TT_OK && state[FT_TT_OP_SAMPLE_STATE_RNG] == rng,
                         "card 0x%08X, host 0x%08X", state[FT_TT_OP_SAMPLE_STATE_RNG], rng);
    /* EIGHT, not nine. The stochastic knobs were uploaded as a fresh state block, which is
     * what a new REQUEST does, and a new request starts its own emitted ring; the greedy
     * token above belongs to the previous state. Worth leaving in the comment because the
     * first version of this check asserted nine and was wrong about WHOSE counter it is —
     * the count is per STATE BLOCK, not per core and not per residency. */
    ft_tt_test_check_fmt("eight tokens are in the emitted ring, none of which crossed the "
                         "link as logits",
                         state[FT_TT_OP_SAMPLE_STATE_EMITTED_COUNT] == 8u, "emitted %u",
                         state[FT_TT_OP_SAMPLE_STATE_EMITTED_COUNT]);
    return ok;
}

/* ================================================================
 * Stage 5 — the loop closes on the card
 * ================================================================ */

/**
 * The whole point of the op, demonstrated rather than argued: the sampled id is written to
 * a DRAM address the descriptor names — which is where the next step's embedding lookup
 * reads its input — and the host does not touch anything between the two descriptors.
 */
static bool stage_5_closed_loop(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    ft_tt_op_topk_request request;
    ft_tt_op_sample_params params = default_params();
    uint32_t token_slot = POISON_WORD;
    uint32_t block[FT_TT_OP_SAMPLE_RESULT_BYTES / 4u];
    uint32_t state[FT_TT_OP_SAMPLE_STATE_WORDS];
    uint32_t step;
    int rc;

    ft_tt_test_step("stage 5 — the decode loop closes ON THE CARD");

    params.eos_count = 1u;
    params.eos_ids[0] = 0xFFFFFFFFu; /* never fires, so the budget is what stops it */
    params.token_budget = 3u;
    rc = upload_state(session, x, y, &params);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_TOKEN], DRAM_TOKEN_BASE, &token_slot,
                          sizeof(token_slot));
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the state and the token slot upload", false, "rc=%d", rc);
        return false;
    }

    /* THREE full TOPK -> SAMPLE pairs with NO host read between them. The host writes six
     * descriptors and reads once at the end; on the K80 path this would have been six
     * logit-row fetches. */
    for (step = 0u; step < 3u; step++) {
        request = base_request(SMALL_VOCAB, SELECT_COUNT, 20u, 0u);
        if (!run_descriptor(session, FT_TT_QUEUE_OP_TOPK, &request, NULL))
            return false;
        request = base_request(SMALL_VOCAB, SELECT_COUNT, 20u, 0u);
        request.token_out.x = (uint16_t)g_dram[DRAM_BANK_TOKEN].x;
        request.token_out.y = (uint16_t)g_dram[DRAM_BANK_TOKEN].y;
        request.token_out.address = DRAM_TOKEN_BASE;
        request.token_out.tile_bytes = 4u;
        if (!run_descriptor(session, FT_TT_QUEUE_OP_SAMPLE, &request, NULL))
            return false;
    }

    rc = device_read(&session->noc, g_dram[DRAM_BANK_TOKEN], DRAM_TOKEN_BASE, &token_slot,
                     sizeof(token_slot));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, core_coord(x, y), g_result_address, block,
                         sizeof(block));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, core_coord(x, y), g_state_address, state,
                         sizeof(state));
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the loop's outputs read back", false, "rc=%d", rc);
        return false;
    }

    ft_tt_test_check_fmt("the token id reached the address the NEXT op would read",
                         token_slot != POISON_WORD &&
                             token_slot == block[FT_TT_OP_SAMPLE_RESULT_TOKEN],
                         "slot %u, result block %u", token_slot,
                         block[FT_TT_OP_SAMPLE_RESULT_TOKEN]);
    ft_tt_test_check_fmt("three tokens were produced with no host read between them",
                         state[FT_TT_OP_SAMPLE_STATE_EMITTED_COUNT] == 3u, "emitted %u",
                         state[FT_TT_OP_SAMPLE_STATE_EMITTED_COUNT]);
    ft_tt_test_check_fmt("the card noticed its own token budget and stopped itself",
                         state[FT_TT_OP_SAMPLE_STATE_STOPPED] == 1u, "stopped %u",
                         state[FT_TT_OP_SAMPLE_STATE_STOPPED]);
    ft_tt_test_check_fmt("the recent ring carries all three, ready for the next penalty",
                         state[FT_TT_OP_SAMPLE_STATE_RECENT_COUNT] == 3u, "recent %u",
                         state[FT_TT_OP_SAMPLE_STATE_RECENT_COUNT]);
    return true;
}

/* ================================================================
 * Stage 6 — a descriptor the program CANNOT execute is refused, not hung on
 * ================================================================ */

/**
 * A block deeper than the circular buffer, which `cb_reserve_back` can never satisfy.
 *
 * THIS CASE IS HERE BECAUSE IT HAPPENED. The first hardware run of this suite asked for a
 * 128-tile block against a 96-page CB and got six red checks, ten seconds of blocked
 * polling and a dead residency — with the failure reported as an idle timeout, which is the
 * one thing it was not. The kernel can read `fifo_num_pages` and the host's placement is
 * exactly what it cannot see, so the refusal belongs there.
 *
 * It is the LAST stage because a refusal ends the residency by design: `kErrBadParameters`
 * is a reason to leave the loop, not to continue. That is the same shape as
 * `stage_7_unknown_opcode_is_refused` in the vector suite.
 */
static void stage_6_an_impossible_block_is_refused(const ft_tt_hardware_session *session)
{
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    ft_tt_op_topk_request request = base_request(SMALL_VOCAB, SELECT_COUNT, SMALL_TILES, 0u);
    uint32_t words[FT_TT_OP_SAMPLE_WORDS];
    ft_tt_queue_descriptor descriptor;
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    int rc;

    ft_tt_test_step("stage 6 — a block deeper than the CB is REFUSED, not blocked on");

    /* SMALL_TILES is 128 and the CB holds 2 x PLACEMENT_DEPTH = 96 pages, so this is
     * unsatisfiable by construction. It is written as a deliberate over-ask rather than as a
     * zero, because a zero-length NoC transfer is suspected of taking the CARD down rather
     * than failing (a sibling op's card-down on 2026-08-12; INFERRED, and not re-confirmed
     * because confirming it costs the card for everyone). A red proof must not be able to
     * reach that value. */
    rc = ft_tt_op_sample_parameters(&request, words);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue,
                                             g_descriptor_slot % 64u, words,
                                             &parameter_address);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the over-deep descriptor is prepared", false, "rc=%d", rc);
        return;
    }
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_TOPK;
    descriptor.a0 = parameter_address;

    (void)ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_descriptor_slot + 1u, 5000u,
                                    &completed);
    (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
    ft_tt_test_check_fmt("it is NOT retired as complete", rc != FT_TT_OK, "rc=%d completed=%u",
                         rc, completed);
    ft_tt_test_check_fmt("and the reader names the REASON rather than a timeout",
                         fields[FT_TT_QUEUE_FIELD_ERROR] == FT_TT_QUEUE_KERR_BAD_PARAMETERS,
                         "%s", ft_tt_queue_kernel_error_string(fields[FT_TT_QUEUE_FIELD_ERROR]));
}

/* ================================================================
 * Stage 7 — the core this suite ran on still BOOTS, and still COMPUTES
 * ================================================================ */

/**
 * THIS STAGE EXISTS BECAUSE THE SUITE ONCE REPORTED `0 checks failed` WHILE DESTROYING THE
 * CORE IT RAN ON.
 *
 * Every check above scores an ANSWER. None of them scores the core, and the two are
 * independent: the token hand-off's `noc_inline_dw_write` wrote the right value to the right
 * address — the stage-5 read-back passed on every run — and left the Tensix unable to boot
 * again (`soft reset 0x00047000`, BRISC out of reset and never releasing its subordinates).
 * It cost Tensix 50 and 52, then 20 and 21 reproducing it, and nothing in a suite that only
 * grades output could ever have seen it. `ft_tt_core status` cannot see it either: a wedged
 * core reads exactly like a healthy idle one (bringup log §21.1), which is why this has to be
 * a BOOT and not a read.
 *
 * Two checks, because there are two independent ways to fail and a single "it still works"
 * would be carried by whichever is cheaper (dev notes §3):
 *   1. the firmware boots again at all — the one the wedge trips;
 *   2. a fresh residency on that core still returns EXACT pairs — which a core that boots but
 *      computes garbage would fail.
 *
 * The boot is the same `assert_reset` + load + deassert the NEXT user of this core performs,
 * so it tests what they will hit rather than a proxy for it. It is not a park: the residency
 * was retired by ASKING first, and four cores (22-25, 27) came back PASS through exactly this
 * sequence.
 */
static bool stage_7_the_core_survives(const ft_tt_hardware_session *session,
                                      const ft_tt_firmware_set *firmware, uint16_t x, uint16_t y,
                                      uint16_t *row, uint8_t *scratch)
{
    ft_tt_launch_state state;
    uint32_t index;
    int rc;
    bool ok;

    ft_tt_test_step("stage 7 — the core this suite ran on still BOOTS, and still COMPUTES");

    rc = ft_tt_core_boot_firmware(&session->noc, x, y, firmware, NULL);
    if (rc != FT_TT_OK) {
        /* Print the reset register, because "the boot failed" is not a diagnosis and a
         * re-run to get it costs another core. 0x00047000 with BRISC running is
         * ft_tt_core's "boot handshake never finished". */
        memset(&state, 0, sizeof(state));
        (void)ft_tt_launch_read_state(&session->noc, x, y, &state);
        ft_tt_test_check_fmt("the firmware BOOTS AGAIN on the core the op ran on", false,
                             "rc=%d, soft reset 0x%08X — the op left this core unbootable", rc,
                             state.reset_register);
        return false;
    }
    ft_tt_test_check_fmt("the firmware BOOTS AGAIN on the core the op ran on", true, "rc=%d", rc);

    /* And it still computes. A core can boot and answer garbage — that is the OTHER wedge
     * signature this backend has seen (dev notes §1), so booting is not the whole claim. */
    g_descriptor_slot = 0u;
    if (!start_residency(session, x, y))
        return false;
    for (index = 0u; index < SMALL_VOCAB; index++)
        row[index] = random_logit();
    ok = score_topk(session, x, y, "and a FRESH residency on it still selects EXACTLY", row,
                    SMALL_VOCAB, SELECT_COUNT, 20u, scratch, NULL);
    retire_program(&session->noc);
    return ok;
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint16_t x = 0;
    uint16_t y = 0;
    uint32_t tensix_index = 0;
    uint32_t bank;
    uint16_t *row = NULL;
    uint8_t *scratch = NULL;
    bool ok = true;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <topk-program-dir>\n", argv[0]);
        return 2;
    }

    ft_tt_test_step("Phase 6 top-k and sampling — ON-CARD acceptance");

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

    rc = ft_tt_program_directory_load(argv[1], TRIAD_MASK, &g_program);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the program directory loads", false, "rc=%d", rc);
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

    row = malloc((size_t)FULL_VOCAB * sizeof(*row));
    scratch = malloc((size_t)FULL_TILES * TILE_BYTES);
    if (!row || !scratch) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto release_program;
    }

    rc = ft_tt_core_boot_firmware(&session.noc, x, y, &firmware, NULL);
    ft_tt_test_check_fmt("firmware boots on the target core", rc == FT_TT_OK,
                         "index %u -> (%u, %u) rc=%d", tensix_index, x, y, rc);
    if (rc != FT_TT_OK)
        goto release_program;

    stage_0_round_trip(&session);
    if (stage_1_start(&session, x, y)) {
        ok = stage_2_exactness(&session, x, y, row, scratch);
        if (ok)
            ok = stage_3_full_row(&session, x, y, row, scratch);
        if (ok)
            ok = stage_4_sampling(&session, x, y, row, scratch);
        if (ok)
            ok = stage_5_closed_loop(&session, x, y);
        if (ok)
            stage_6_an_impossible_block_is_refused(&session);
        /* Retire by ASKING whatever happened above: a program left resident on a failure is
         * a core the next worker cannot use, and the ask is the only exit protocol
         * (dev notes §1 — never park, never reset). */
        retire_program(&session.noc);
        /* UNCONDITIONALLY, and last: whatever the answers were, the core has to still be a
         * core. This runs even when a stage above failed, because a suite that gives up
         * early is exactly when nobody finds out what it left behind. */
        ok = stage_7_the_core_survives(&session, &firmware, x, y, row, scratch) && ok;
    }

    /* DO NOT PARK. The core is left in its firmware's GO wait, which is where a healthy
     * idle core sits (dev notes §1: parking a resident triad breaks its NEXT boot). */

release_program:
    free(row);
    free(scratch);
    ft_tt_program_directory_release(&g_program);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
