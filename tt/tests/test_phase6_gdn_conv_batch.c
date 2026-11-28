/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_gdn_conv_batch.c
 * @brief ON-CARD acceptance of the gdn CONV ROW BATCH: the batched descriptor against the
 *   multi-row walk on the same core, the same images and the same uploaded operands,
 *   bit for bit, at 2, 3 and 4 rows, at row 0 and at rows [12, 16), whole stream and one
 *   column slice. No model, no oracle tolerance: the walk IS the reference, and every byte
 *   of both outputs is compared.
 *
 * What each arm checks, after poisoning every output buffer to the same byte:
 *   - conv_out: every tile, every row — rows the batch must not touch stay poison in both;
 *   - row r's rolled window: every tile of row r's slot;
 *   - the live window: unchanged by both (the verify keeps it; the commit picks a slot), so
 *     a row past the accepted ones cannot leak into it.
 *
 * Red (data only, legal): the batched reader's rows red (kGdnFlagRedRowsRow0) stages row 0's
 * stream for every row — the batch must then DIFFER from the walk, or the comparison is blind.
 *
 * Every exit after bring-up retires the triad by asking it (ft_tt_watchdog_shutdown), timeouts
 * included: a descriptor that does not complete is reported and the remaining arms are skipped,
 * then the same retire runs.
 *
 *   make -f Makefile.gdn PHASE1B=1 card
 *   FT_TT_DRIVER=udrv FT_TT_TENSIX_INDEX=<n> ./build-gdn-card/test_phase6_gdn_conv_batch ../programs/gdn
 *
 * @status added with the row batch.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ft_tt_coord.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_gdn.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_BYTES     FT_TT_OP_GDN_FP32_TILE_BYTES
#define RUN_TIMEOUT_MS 30000u
#define QUEUE_CAPACITY 64u
#define POISON_BYTE    0x5A

/** Banks 2..4 at 576 MiB: away from every other suite's base (they share these banks). */
#define DRAM_BANK_ACTIVATION 2u
#define DRAM_BANK_CONSTANT   3u
#define DRAM_BANK_STATE      4u
#define DRAM_BANK_COUNT      5u
#define DRAM_BASE            0x24000000u
/** Inside the activation bank: the output above the input. Inside the state bank: the live
 *  window, then two sets of row slots (walk, batch), each MAX_ROWS slots. */
#define DRAM_OUTPUT_GAP 0x01000000u
#define DRAM_SLOT_GAP   0x00400000u

#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

#define TRIAD_MASK                                                                \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) | \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

/** Qwen3.8's head shape (the placement is the family's); a 2048-wide stream (64 tile columns)
 *  keeps the readback short. */
#define STATE_SIZE 128u
#define V_HEADS    48u
#define K_HEADS    16u
#define TAPS       4u
#define CONV_DIM   2048u
#define CONV_COLS  (CONV_DIM / 32u)
#define MAX_ROWS   4u

#define STREAM_BYTES ((size_t)CONV_COLS * TILE_BYTES)
#define WEIGHT_BYTES ((size_t)TAPS * CONV_COLS * TILE_BYTES)
#define WINDOW_BYTES ((size_t)(TAPS - 1u) * CONV_COLS * TILE_BYTES)

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_program;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_GDN_CB_COUNT];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG *
                                 4u];
static uint32_t g_slot;
/** A descriptor timed out: later arms are skipped, the retire still runs. */
static bool g_stuck;

static unsigned char g_upload[WEIGHT_BYTES];
static unsigned char g_walk_out[STREAM_BYTES], g_batch_out[STREAM_BYTES];
static unsigned char g_walk_window[WINDOW_BYTES], g_batch_window[WINDOW_BYTES];
static unsigned char g_live_before[WINDOW_BYTES], g_live_after[WINDOW_BYTES];

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

/** Finite fp32 values in [-2, 2), never denormal, from a fixed seed. */
static void fill_random(unsigned char *bytes, size_t count, uint32_t seed)
{
    uint32_t state = 0x9E3779B9u ^ seed;
    size_t index;

    for (index = 0; index + 4u <= count; index += 4u) {
        float value;

        state = state * 1664525u + 1013904223u;
        value = (float)((int32_t)(state >> 8) - 0x800000) / 4194304.0f;
        memcpy(bytes + index, &value, sizeof(value));
    }
}

/* ================================================================
 * Requests and descriptors
 * ================================================================ */

static ft_tt_op_operand bank_operand(uint32_t bank, uint32_t address)
{
    ft_tt_op_operand operand;

    operand.x = (uint16_t)g_dram[bank].x;
    operand.y = (uint16_t)g_dram[bank].y;
    operand.address = address;
    operand.tile_bytes = TILE_BYTES;
    return operand;
}

/** Row r's window slot of slot set `set` (0 walk, 1 batch). */
static uint32_t slot_address(uint32_t set, uint32_t row)
{
    return DRAM_BASE + DRAM_SLOT_GAP * (1u + set * MAX_ROWS + row);
}

/** A strip CONV over rows [row, row + rows), each row's window to slot set `set`. */
static void fill_request(ft_tt_op_gdn_request *request, uint32_t row, uint32_t rows, uint32_t set,
                         bool batch)
{
    uint32_t r;

    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDN_PHASE_CONV;
    request->state_size = STATE_SIZE;
    request->v_heads = V_HEADS;
    request->k_heads = K_HEADS;
    request->conv_dim = CONV_DIM;
    request->conv_taps = TAPS;
    request->activation_in = bank_operand(DRAM_BANK_ACTIVATION, DRAM_BASE);
    request->activation_out = bank_operand(DRAM_BANK_ACTIVATION, DRAM_BASE + DRAM_OUTPUT_GAP);
    request->conv_weight = bank_operand(DRAM_BANK_CONSTANT, DRAM_BASE);
    request->state_conv = bank_operand(DRAM_BANK_STATE, DRAM_BASE);
    request->flags = FT_TT_OP_GDN_FLAG_CONV_STRIP;
    request->row = row;
    request->row_count = rows;
    request->conv_row_batch = batch ? 1u : 0u;
    for (r = 0; r < rows; r++)
        request->state_rows_out[r] = bank_operand(DRAM_BANK_STATE, slot_address(set, r));
    request->state_out = request->state_rows_out[0];
}

/** Print every processor's status slot: which one is blocked, in which wait. */
static void report_stuck(const ft_tt_hardware_session *session)
{
    static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0,
                                            FT_TT_PROC_MATH1, FT_TT_PROC_MATH2 };
    static const char *const names[5] = { "BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2" };
    ft_tt_noc_coord where;
    unsigned which;

    where.x = g_queue.x;
    where.y = g_queue.y;
    for (which = 0; which < 5u; which++) {
        uint32_t slot[FT_TT_QUEUE_STATUS_WORDS];

        if (device_read(&session->noc, where,
                        FT_TT_QUEUE_STATUS_ADDRESS(g_queue.control_address, processors[which]),
                        slot, sizeof(slot)) != FT_TT_OK)
            continue;
        fprintf(stderr, "    %-6s state 0x%08x cursor %u error 0x%08x opcode 0x%02x\n",
                names[which], slot[FT_TT_QUEUE_STATUS_STATE], slot[FT_TT_QUEUE_STATUS_CURSOR],
                slot[FT_TT_QUEUE_STATUS_ERROR], slot[FT_TT_QUEUE_STATUS_LAST_OPCODE]);
    }
}

/**
 * Encode, write the parameters (and the row extension into the NEXT slot), enqueue, and wait
 * for the writer's completion. Slots are taken in pairs from an even index, so the extension
 * never wraps; the harness waits before the next enqueue, so a rotating slot is safe.
 */
static int run_descriptor(const ft_tt_hardware_session *session,
                          const ft_tt_op_gdn_request *request)
{
    const uint32_t slot = (g_slot * 2u) % FT_TT_QUEUE_MATMUL_PARAM_SLOTS;
    ft_tt_queue_descriptor descriptor;
    uint32_t words[FT_TT_OP_GDN_WORDS];
    uint32_t extension[FT_TT_OP_GDN_WORDS];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    int rc;

    if (g_stuck)
        return FT_TT_EINVAL;
    rc = ft_tt_op_gdn_parameters(request, words);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_gdn_row_extension(request, extension);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, slot + 1u, extension,
                                             NULL);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, slot, words,
                                             &parameter_address);
    if (rc != FT_TT_OK)
        return rc;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_GDN;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    if (rc != FT_TT_OK) {
        g_stuck = true;
        ft_tt_test_check_fmt("the descriptor completes", false, "rc=%d completed=%u of %u", rc,
                             completed, g_slot + 1u);
        report_stuck(session);
        return rc;
    }
    g_slot++;
    return FT_TT_OK;
}

/* ================================================================
 * The arms
 * ================================================================ */

/** Poison conv_out and slot set `set`'s first `rows` slots. */
static int poison_outputs(const ft_tt_noc_access *noc, uint32_t set, uint32_t rows)
{
    static unsigned char poison[WINDOW_BYTES];
    uint32_t r;
    int rc;

    memset(poison, POISON_BYTE, sizeof(poison));
    rc = device_write(noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE + DRAM_OUTPUT_GAP, poison,
                      STREAM_BYTES);
    for (r = 0; rc == FT_TT_OK && r < rows; r++)
        rc = device_write(noc, g_dram[DRAM_BANK_STATE], slot_address(set, r), poison,
                          WINDOW_BYTES);
    return rc;
}

/** Run one arm (walk or batch) and read back its conv_out. */
static int run_arm(const ft_tt_hardware_session *session, const ft_tt_op_gdn_request *request,
                   unsigned char *out)
{
    int rc = poison_outputs(&session->noc, request->conv_row_batch ? 1u : 0u,
                            request->row_count);

    if (rc == FT_TT_OK)
        rc = run_descriptor(session, request);
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE + DRAM_OUTPUT_GAP,
                         out, STREAM_BYTES);
    return rc;
}

/**
 * The batch against the walk at `rows` rows from `row`, optionally on a column slice and under
 * the rows red. Returns true when conv_out and every row's window are bit-identical.
 */
static bool compare_arm(const ft_tt_hardware_session *session, uint32_t row, uint32_t rows,
                        uint32_t col_first, uint32_t col_count, bool red, const char *label)
{
    ft_tt_op_gdn_request walk, batch;
    uint32_t r;
    bool same;
    int rc;

    fill_request(&walk, row, rows, 0u, false);
    fill_request(&batch, row, rows, 1u, true);
    walk.col_first = batch.col_first = col_first;
    walk.col_count = batch.col_count = col_count;
    if (red)
        batch.flags |= FT_TT_OP_GDN_FLAG_RED_ROWS_ROW0;
    rc = run_arm(session, &walk, g_walk_out);
    if (rc == FT_TT_OK)
        rc = run_arm(session, &batch, g_batch_out);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "rc=%d", rc);
        return false;
    }
    same = memcmp(g_walk_out, g_batch_out, STREAM_BYTES) == 0;
    for (r = 0; rc == FT_TT_OK && r < rows; r++) {
        rc = device_read(&session->noc, g_dram[DRAM_BANK_STATE], slot_address(0u, r),
                         g_walk_window, WINDOW_BYTES);
        if (rc == FT_TT_OK)
            rc = device_read(&session->noc, g_dram[DRAM_BANK_STATE], slot_address(1u, r),
                             g_batch_window, WINDOW_BYTES);
        same = same && memcmp(g_walk_window, g_batch_window, WINDOW_BYTES) == 0;
    }
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_dram[DRAM_BANK_STATE], DRAM_BASE, g_live_after,
                         WINDOW_BYTES);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "readback rc=%d", rc);
        return false;
    }
    ft_tt_test_check_fmt("the live window is untouched by both arms",
                         memcmp(g_live_before, g_live_after, WINDOW_BYTES) == 0, "%s", label);
    if (red)
        ft_tt_test_check_fmt(label, !same, "rows %u from %u: the red must DIFFER", rows, row);
    else
        ft_tt_test_check_fmt(label, same, "rows %u from %u, columns [%u, +%u)", rows, row,
                             col_first, col_count ? col_count : CONV_COLS);
    return same;
}

/** Upload the stream (every row of every tile), the weights and the live window. */
static int upload_operands(const ft_tt_noc_access *noc)
{
    int rc;

    fill_random(g_upload, STREAM_BYTES, 1u);
    rc = device_write(noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE, g_upload, STREAM_BYTES);
    fill_random(g_upload, WEIGHT_BYTES, 2u);
    if (rc == FT_TT_OK)
        rc = device_write(noc, g_dram[DRAM_BANK_CONSTANT], DRAM_BASE, g_upload, WEIGHT_BYTES);
    fill_random(g_live_before, WINDOW_BYTES, 3u);
    if (rc == FT_TT_OK)
        rc = device_write(noc, g_dram[DRAM_BANK_STATE], DRAM_BASE, g_live_before, WINDOW_BYTES);
    return rc;
}

static void stage_batch(const ft_tt_hardware_session *session)
{
    const uint32_t granule = ft_tt_op_gdn_conv_slice_granule(STATE_SIZE, TAPS);
    uint32_t rows;
    int rc;

    ft_tt_test_step("CONV row batch == the multi-row walk, bit for bit");
    rc = upload_operands(&session->noc);
    ft_tt_test_check_fmt("operands upload", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    for (rows = 2u; rows <= MAX_ROWS && !g_stuck; rows++)
        (void)compare_arm(session, 0u, rows, 0u, 0u, false, "batched rows == walked rows");
    if (!g_stuck)
        (void)compare_arm(session, 12u, 4u, 0u, 0u, false, "rows [12, 16) batched == walked");
    if (!g_stuck && granule != 0u && 2u * granule <= CONV_COLS)
        (void)compare_arm(session, 0u, 4u, granule, granule, false,
                          "a column slice batched == walked");
    ft_tt_test_step("red: the rows red must DIFFER");
    if (!g_stuck)
        (void)compare_arm(session, 0u, 4u, 0u, 0u, true, "rows red (row 0's stream for every row)");
}

/* ================================================================
 * Bring-up
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

static int bring_up_triad(const ft_tt_hardware_session *session, uint16_t x, uint16_t y)
{
    ft_tt_op_gdn_request placement;
    uint64_t mask = 0;
    int rc;

    fill_request(&placement, 0u, MAX_ROWS, 0u, false);
    rc = load_images(&session->noc, x, y, &g_program);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, x, y, &g_program.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_gdn_place_cbs(&placement, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer),
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

static void retire_triad(const ft_tt_noc_access *noc)
{
    uint32_t unstopped = 0u;
    int rc = ft_tt_watchdog_shutdown(noc, &g_group, RUN_TIMEOUT_MS, &unstopped);

    ft_tt_test_check_fmt("all five processors retire when ASKED", rc == FT_TT_OK && unstopped == 0u,
                         "rc=%d unstopped=0x%X", rc, unstopped);
}

/** The core set, from the live grid (classified, never hand-filled). */
static int claim_core(ft_tt_hardware_session *session, uint32_t tensix_index)
{
    int rc = ft_tt_launch_core_set_init(&g_group.cores, g_placement_storage, 1u);

    if (rc == FT_TT_OK)
        rc = ft_tt_launch_core_set_add_tensix(&g_group.cores, &session->grid, tensix_index, NULL);
    if (rc != FT_TT_OK)
        return rc;
    g_group.id = 0u;
    g_group.first_tensix_index = tensix_index;
    g_group.tensix_index_count = 1u;
    g_group.queues = &g_queue;
    g_group.resident_processor_mask = TRIAD_MASK;
    return FT_TT_OK;
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
        fprintf(stderr, "usage: %s <gdn-program-dir>\n", argv[0]);
        return 2;
    }
    ft_tt_test_step("gdn CONV ROW BATCH — ON-CARD acceptance");
    if (ft_tt_hardware_session_open(&session) != FT_TT_OK)
        return 1;
    if (ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware) != FT_TT_OK) {
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
        ft_tt_test_check_fmt("the program directory loads", false, "rc=%d", rc);
        goto release;
    }
    rc = claim_core(&session, tensix_index);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the core set derives and classifies the target", false,
                             "index %u rc=%d", tensix_index, rc);
        goto release_programs;
    }
    rc = ft_tt_core_boot_firmware(&session.noc, x, y, &firmware, NULL);
    ft_tt_test_check_fmt("firmware boots on the target core", rc == FT_TT_OK,
                         "index %u -> (%u, %u) rc=%d", tensix_index, x, y, rc);
    if (rc != FT_TT_OK)
        goto release_programs;
    rc = bring_up_triad(&session, x, y);
    ft_tt_test_check_fmt("the gdn triad is staged and kicked ONCE", rc == FT_TT_OK, "rc=%d", rc);
    if (rc == FT_TT_OK)
        stage_batch(&session);
    /* Whatever happened above — a refusal, a timeout, a failed check — the triad is retired
     * by asking it, never left running. */
    retire_triad(&session.noc);

release_programs:
    ft_tt_program_directory_release(&g_program);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
