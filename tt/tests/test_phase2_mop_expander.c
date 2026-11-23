/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_mop_expander.c — WHAT DOES THE TENSIX MOP EXPANDER DO WITH
 * `loop0_last_instr` / `loop1_last_instr`?
 *
 * The nine-word double-loop template (ported in
 * kernels/ft_tensix_compute.h) carries two "last instruction" words. Everything this
 * backend counts — 32 ELWMUL per mul_tiles, 32 MVMUL per reduce_tile, 64 MVMUL per HiFi4
 * matmul tile — rests on a rule that was inferred from a self-consistent story and never
 * measured: that the substitution REPLACES rather than ADDS. A C-vs-C++ instruction diff
 * cannot settle it, because both models program the SAME nine words.
 *
 * THE MEASUREMENT. mop_expander_probe_compute.c turns DST into a TAPE: every issued op writes
 * one 8-row DST slot and the ADDR_MOD advances DST by 8, so the k-th instruction the expander
 * issues lands in slot k. With SrcA = 5.0 and SrcB = 2.0 uniformly, ELWADD writes 7, ELWSUB
 * writes +/-3 and ELWMUL writes 10 — so one packed 32x32 output tile reads back as the
 * LITERAL SEQUENCE of ops the MOP expanded to, eight slots at a time, and the tape LENGTH is
 * the instruction COUNT. DST is zeroed by the math thread before every trial, so an unwritten
 * slot reads 0 and a count is a count rather than a maximum.
 *
 * THE FOUR CANDIDATE RULES, all of which program identical nine words:
 *
 *   H1 REPLACE-LAST  the substitution replaces the LAST instruction of the inner body:
 *                    loop_op1 when loop_op1 is a real op, loop_op0 when loop_op1 is NOP.
 *                    (This is what the backend assumes; it is the only rule under which BOTH
 *                    upstream constructors are an identity.)
 *   H2 REPLACE-OP0   the substitution always replaces the loop_op0 slot.
 *   H3 REPLACE-OP1   the substitution always replaces the loop_op1 slot. This is the LITERAL
 *                    pseudocode in the upstream class comment.
 *   H4 ADD           the substitution is APPENDED; both the body and the last-instr issue.
 *
 * H1 and H3 agree whenever loop_op1 is a real op; H1 and H2 agree whenever loop_op1 is NOP.
 * So it takes BOTH kinds of trial to separate all four, and the sweep runs both.
 *
 * Predictions are not typed in per trial: expand_template() below IS each model, and the
 * expected tape for every trial is computed from it. A trial can therefore come out matching
 * a model nobody wanted, which is the point.
 *
 * Build (on the card host):
 *   cc -std=c99 -O1 -g -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion \
 *      -Wstrict-prototypes -Wformat=2 -I.. -DFT_TT_PHASE1B_NOC_ENABLED \
 *      ../ft_tt_kmd.c ../ft_tt_arch.c ../ft_tt_arch_blackhole.c ../ft_tt_device.c \
 *      ../ft_tt_tlb.c ../ft_tt_arc_telemetry.c ../ft_tt_coord.c ../ft_tt_noc.c \
 *      ../ft_tt_elf.c ../ft_tt_kernel_blob.c ../ft_tt_launch.c \
 *      ft_tt_test_report.c ft_tt_hardware_session.c ft_tt_phase2_l1_layout.c \
 *      ft_tt_program_directory.c test_phase2_mop_expander.c \
 *      -o build/test_phase2_mop_expander
 *
 * Run:
 *   FT_TT_TENSIX_INDEX=n FT_TT_FIRMWARE_DIR=<program dir> \
 *       ./build/test_phase2_mop_expander <program dir>
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_hardware_session.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

#define RUN_TIMEOUT_MS 10000u
#define TILE_ELEMENTS  (FT_TT_PHASE2_TILE_BYTES / 2u)
#define CB_CONFIG_BYTES (FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4u)
#define COUNTER_STAGE_ATTEMPTS 64u

/** Rows a tape slot covers, and therefore elements: MAX_FPU_ROWS(8) * 16 datums. */
#define SLOT_ELEMENTS 128u
/** Slots in one packed 32x32 tile. The tape is read one output tile per trial. */
#define TAPE_SLOTS (TILE_ELEMENTS / SLOT_ELEMENTS)

/** Output CB pages, and so trials per launch: 0x6000 B between CB_OUT and SOURCE0. */
#define MAX_TRIALS 12u

/** The two operands. Chosen so ADD/SUB/MUL land on three distinct exact bfloat16 values. */
#define OPERAND_A 5.0f
#define OPERAND_B 2.0f

/* Symbols, matching the kernel's trial encoding. */
#define SYM_NOP 0u
#define SYM_ADD 1u
#define SYM_SUB 2u
#define SYM_MUL 3u
#define SYM_NONE 7u /**< host-only: "no substitution applies here" */

/* Hypotheses. */
#define H_REPLACE_LAST 0u
#define H_REPLACE_OP0  1u
#define H_REPLACE_OP1  2u
#define H_ADD          3u
#define HYPOTHESIS_COUNT 4u

static const char *const hypothesis_name[HYPOTHESIS_COUNT] = {
    "H1 REPLACE-last-of-body", "H2 REPLACE-op0-slot", "H3 REPLACE-op1-slot", "H4 ADD-append"};

/* Kernel trial modes. */
#define MODE_MOP           0u
#define MODE_LADDER_UNIFORM 1u
#define MODE_LADDER_CYCLE   2u
#define MODE_SILENT         3u

typedef struct trial {
    const char *name;
    unsigned mode;
    unsigned outer;
    unsigned inner;
    unsigned ladder;
    unsigned addr_mod; /**< 1 = every op writes slot 0; the deliberately-wrong control */
    unsigned start0;
    unsigned end0;
    unsigned end1;
    unsigned op0;
    unsigned op1;
    unsigned l0;
    unsigned l1;
} trial;

static ft_tt_phase2_l1_layout g_layout;

/* ------------------------------------------------------------------------- */
/* The models                                                                */
/* ------------------------------------------------------------------------- */

/** Append one symbol to a tape, counting past the end rather than writing past it. */
static void tape_emit(unsigned symbol, unsigned char *tape, unsigned capacity, unsigned *length)
{
    if (symbol == SYM_NOP || symbol == SYM_NONE) {
        return; /* a NOP issues but writes no DST slot, so it leaves no mark on the tape */
    }
    if (*length < capacity)
        tape[*length] = (unsigned char)symbol;
    (*length)++;
}

/**
 * Expand one trial under one hypothesis into the symbol sequence the DST tape would show.
 *
 * Returns the FULL length; only the first `capacity` symbols are written. The outer-loop
 * shape is the upstream class comment's, which is not in dispute:
 *   LOOP_OUTER: { START_OP; LOOP_INNER: { LOOP_OP0; LOOP_OP1 }; END_OP0; END_OP1 }
 * The hypotheses differ only in what the last inner iteration does with loop0/loop1_last.
 */
static unsigned expand_template(const trial *t, unsigned hypothesis, unsigned char *tape,
                                unsigned capacity)
{
    unsigned length = 0;
    unsigned k;
    unsigned j;

    if (t->mode == MODE_SILENT)
        return 0;
    if (t->mode == MODE_LADDER_UNIFORM) {
        for (k = 0; k < t->ladder; k++)
            tape_emit(t->op0, tape, capacity, &length);
        return length;
    }
    if (t->mode == MODE_LADDER_CYCLE) {
        const unsigned cycle[3] = {t->op0, t->op1, t->l0};

        for (k = 0; k < t->ladder; k++)
            tape_emit(cycle[k % 3u], tape, capacity, &length);
        return length;
    }

    for (k = 0; k < t->outer; k++) {
        tape_emit(t->start0, tape, capacity, &length);
        for (j = 0; j < t->inner; j++) {
            int last_inner = (j + 1u == t->inner);
            /* The class comment's own rule for WHICH last-instr applies, which this
             * experiment also gets to check: loop0_last on the final inner iteration of the
             * final outer iteration, loop1_last on every other final inner iteration. */
            unsigned substitution =
                last_inner ? ((k + 1u == t->outer) ? t->l0 : t->l1) : SYM_NONE;

            switch (hypothesis) {
                case H_REPLACE_LAST:
                    if (t->op1 != SYM_NOP) {
                        tape_emit(t->op0, tape, capacity, &length);
                        tape_emit(last_inner ? substitution : t->op1, tape, capacity, &length);
                    } else {
                        tape_emit(last_inner ? substitution : t->op0, tape, capacity, &length);
                        tape_emit(t->op1, tape, capacity, &length);
                    }
                    break;
                case H_REPLACE_OP0:
                    tape_emit(last_inner ? substitution : t->op0, tape, capacity, &length);
                    tape_emit(t->op1, tape, capacity, &length);
                    break;
                case H_REPLACE_OP1:
                    tape_emit(t->op0, tape, capacity, &length);
                    tape_emit(last_inner ? substitution : t->op1, tape, capacity, &length);
                    break;
                default: /* H_ADD */
                    tape_emit(t->op0, tape, capacity, &length);
                    tape_emit(t->op1, tape, capacity, &length);
                    if (last_inner)
                        tape_emit(substitution, tape, capacity, &length);
                    break;
            }
        }
        tape_emit(t->end0, tape, capacity, &length);
        tape_emit(t->end1, tape, capacity, &length);
    }
    return length;
}

/**
 * What the OUTPUT TILE should show for a trial under a hypothesis.
 *
 * Identical to the expansion except when the trial names the zero-increment ADDR_MOD: then
 * every op overwrites slot 0, so all that survives is the last symbol issued. That is the
 * deliberately-wrong control, and its predicted observation is a wrong-looking one slot.
 */
static unsigned predict_observation(const trial *t, unsigned hypothesis, unsigned char *tape,
                                    unsigned capacity)
{
    unsigned char full[64];
    unsigned length = expand_template(t, hypothesis, full, 64u);

    if (t->addr_mod == 0u) {
        unsigned copy = (length < capacity) ? length : capacity;

        memcpy(tape, full, copy);
        return (length < capacity) ? length : capacity;
    }
    if (length == 0u)
        return 0u;
    if (length > 64u)
        length = 64u;
    if (capacity > 0u)
        tape[0] = full[length - 1u];
    return 1u;
}

/* ------------------------------------------------------------------------- */
/* Trial encoding                                                            */
/* ------------------------------------------------------------------------- */

static uint32_t trial_shape_word(const trial *t)
{
    return (t->mode & 0xFu) | ((t->outer & 0xFu) << 4) | ((t->inner & 0xFu) << 8) |
           ((t->ladder & 0x1Fu) << 12) | ((t->addr_mod & 0x3u) << 17);
}

static uint32_t trial_ops_word(const trial *t)
{
    return (t->start0 & 7u) | ((t->end0 & 7u) << 3) | ((t->end1 & 7u) << 6) |
           ((t->op0 & 7u) << 9) | ((t->op1 & 7u) << 12) | ((t->l0 & 7u) << 15) |
           ((t->l1 & 7u) << 18);
}

/* ------------------------------------------------------------------------- */
/* Tiles                                                                     */
/* ------------------------------------------------------------------------- */

static uint16_t tile_a[TILE_ELEMENTS];
static uint16_t tile_b[TILE_ELEMENTS];
static uint16_t tile_zero[TILE_ELEMENTS];
static uint16_t tile_readback[TILE_ELEMENTS];

static uint16_t bfloat16_from_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float float_from_bfloat16(uint16_t value)
{
    uint32_t bits = (uint32_t)value << 16;
    float result;

    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void build_operand_tiles(void)
{
    unsigned element;

    for (element = 0; element < TILE_ELEMENTS; element++) {
        tile_a[element] = bfloat16_from_float(OPERAND_A);
        tile_b[element] = bfloat16_from_float(OPERAND_B);
        tile_zero[element] = 0u;
    }
}

/* ------------------------------------------------------------------------- */
/* Calibration: which bfloat16 pattern each op family writes                  */
/* ------------------------------------------------------------------------- */

/** Filled by the calibration trial. 0 means "not yet known". */
static uint16_t g_symbol_pattern[4];

static const char *symbol_text(unsigned symbol)
{
    switch (symbol) {
        case SYM_ADD: return "A";
        case SYM_SUB: return "S";
        case SYM_MUL: return "M";
        case SYM_NOP: return "n";
        default: return "?";
    }
}

/** Decode one slot of a packed tile: its uniform value, or a refusal. */
static int slot_value(const uint16_t *tile, unsigned slot, uint16_t *out_value)
{
    unsigned base = slot * SLOT_ELEMENTS;
    uint16_t first = tile[base];
    unsigned index;

    for (index = 1; index < SLOT_ELEMENTS; index++) {
        if (tile[base + index] != first)
            return 0; /* not uniform: the slot is not a clean symbol */
    }
    *out_value = first;
    return 1;
}

/** Map a slot's value to a symbol using the calibration; SYM_NONE if unrecognised. */
static unsigned symbol_of_value(uint16_t value)
{
    unsigned symbol;

    if (value == 0u)
        return SYM_NOP; /* an untouched slot */
    for (symbol = SYM_ADD; symbol <= SYM_MUL; symbol++) {
        if (g_symbol_pattern[symbol] != 0u && g_symbol_pattern[symbol] == value)
            return symbol;
    }
    return SYM_NONE;
}

/**
 * Read a tape out of a packed tile.
 *
 * The tape ENDS at the first zero slot: a written slot is never zero (7, +/-3 and 10 are all
 * nonzero) and the math thread zeroed DST before the trial, so a zero slot is an instruction
 * that was never issued.
 */
static unsigned read_tape(const uint16_t *tile, unsigned char *tape, uint16_t *values,
                          int *out_clean)
{
    unsigned length = 0;
    unsigned slot;

    *out_clean = 1;
    for (slot = 0; slot < TAPE_SLOTS; slot++) {
        uint16_t value = 0;

        if (!slot_value(tile, slot, &value)) {
            *out_clean = 0;
            values[slot] = 0xFFFFu;
            tape[slot] = (unsigned char)SYM_NONE;
            length = slot + 1u;
            continue;
        }
        values[slot] = value;
        tape[slot] = (unsigned char)symbol_of_value(value);
        if (value != 0u)
            length = slot + 1u;
    }
    return length;
}

/* ------------------------------------------------------------------------- */
/* Device plumbing                                                           */
/* ------------------------------------------------------------------------- */

static int boot_and_derive_layout(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  const ft_tt_firmware_set *firmware)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    int rc = ft_tt_core_boot_firmware(noc, x, y, firmware, placements);

    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_phase2_l1_layout_derive(placements, FT_TT_PROC_COUNT, &g_layout);
}

static void describe_circular_buffers(ft_tt_launch_cb_entry entries[3])
{
    entries[0].index = FT_TT_PHASE2_CB_IN0;
    entries[0].address = g_layout.cb_in0_address;
    entries[0].size_bytes = 2u * FT_TT_PHASE2_TILE_BYTES;
    entries[0].page_size_bytes = FT_TT_PHASE2_TILE_BYTES;
    entries[0].page_count = 2u;

    entries[1].index = FT_TT_PHASE2_CB_IN1;
    entries[1].address = g_layout.cb_in1_address;
    entries[1].size_bytes = 2u * FT_TT_PHASE2_TILE_BYTES;
    entries[1].page_size_bytes = FT_TT_PHASE2_TILE_BYTES;
    entries[1].page_count = 2u;

    /* One output page per trial: the tape for trial i is output tile i. */
    entries[2].index = FT_TT_PHASE2_CB_OUT;
    entries[2].address = g_layout.cb_out_address;
    entries[2].size_bytes = MAX_TRIALS * FT_TT_PHASE2_TILE_BYTES;
    entries[2].page_size_bytes = FT_TT_PHASE2_TILE_BYTES;
    entries[2].page_count = MAX_TRIALS;
}

static int write_cb_config(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           uint64_t *out_mask)
{
    static uint8_t buffer[CB_CONFIG_BYTES];
    ft_tt_launch_cb_entry entries[3];
    int rc;

    describe_circular_buffers(entries);
    rc = ft_tt_launch_encode_cb_config(entries, 3u, buffer, sizeof(buffer), out_mask);
    if (rc != FT_TT_OK)
        return rc;
    return noc->write(noc->user, x, y, g_layout.cb_config_address, buffer, sizeof(buffer));
}

static int stage_cb_counters(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                             unsigned cb_index, uint32_t received, uint32_t acked)
{
    uint32_t received_address = FT_TT_CB_TILES_RECEIVED_ADDR(cb_index);
    uint32_t acked_address = FT_TT_CB_TILES_ACKED_ADDR(cb_index);
    unsigned attempt;

    for (attempt = 0; attempt < COUNTER_STAGE_ATTEMPTS; attempt++) {
        uint32_t observed_received = 0;
        uint32_t observed_acked = 0;
        int rc = noc->write(noc->user, x, y, acked_address, &acked, sizeof(acked));

        if (rc != FT_TT_OK)
            return rc;
        rc = noc->write(noc->user, x, y, received_address, &received, sizeof(received));
        if (rc != FT_TT_OK)
            return rc;
        rc = noc->read(noc->user, x, y, received_address, &observed_received,
                       sizeof(observed_received));
        if (rc != FT_TT_OK)
            return rc;
        rc = noc->read(noc->user, x, y, acked_address, &observed_acked, sizeof(observed_acked));
        if (rc != FT_TT_OK)
            return rc;
        if (observed_received == received && observed_acked == acked)
            return FT_TT_OK;
    }
    ft_tt_log_error("mop probe: CB %u counters would not hold their value", cb_index);
    return FT_TT_EIO;
}

static void build_config(const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                         uint64_t cb_mask, ft_tt_launch_config *out_config)
{
    unsigned processor;

    memset(out_config, 0, sizeof(*out_config));
    out_config->kernel_config_base = g_layout.kernel_config_base;
    out_config->enables = 0x1cu; /* the compute triad only — no device NoC traffic at all */
    out_config->local_cb_config_address = g_layout.cb_config_address;
    out_config->local_cb_mask = cb_mask;
    out_config->brisc_noc_id = 0u;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if ((out_config->enables & (1u << processor)) == 0u)
            continue;
        out_config->kernel_text_address[processor] = placements[processor].entry_address;
        out_config->kernel_text_bytes[processor] = placements[processor].text_bytes;
        out_config->runtime_args_address[processor] =
            ft_tt_phase2_runtime_args_address(&g_layout, processor);
    }
}

/* ------------------------------------------------------------------------- */
/* One batch                                                                 */
/* ------------------------------------------------------------------------- */

/** Per-hypothesis tally across every scored trial in the whole run. */
static unsigned g_hypothesis_matches[HYPOTHESIS_COUNT];
static unsigned g_hypothesis_scored;

static void print_tape(const char *label, const unsigned char *tape, unsigned length)
{
    unsigned index;

    printf("%s[", label);
    for (index = 0; index < length; index++)
        printf("%s", symbol_text(tape[index]));
    printf("]%u", length);
}

static int run_batch(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                     const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                     const trial *trials, unsigned trial_count, const char *batch_name)
{
    ft_tt_launch_config config;
    uint64_t cb_mask = 0;
    uint32_t arguments[1u + 2u * MAX_TRIALS];
    unsigned processor;
    unsigned index;
    int rc;

    ft_tt_test_step("%s", batch_name);

    if (trial_count > MAX_TRIALS) {
        ft_tt_test_check("batch fits the output CB", 0, "too many trials");
        return FT_TT_EINVAL;
    }

    rc = write_cb_config(noc, x, y, &cb_mask);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("CB config written", 0, ft_tt_error_string(rc));
        return rc;
    }

    build_config(placements, cb_mask, &config);

    arguments[0] = trial_count;
    for (index = 0; index < trial_count; index++) {
        arguments[1u + 2u * index] = trial_shape_word(&trials[index]);
        arguments[2u + 2u * index] = trial_ops_word(&trials[index]);
    }
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (config.runtime_args_address[processor] == 0u)
            continue;
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config.runtime_args_address[processor],
                                             arguments, 1u + 2u * trial_count);
        if (rc != FT_TT_OK)
            break;
    }
    if (rc != FT_TT_OK) {
        ft_tt_test_check("trial table planted on each TRISC", 0, ft_tt_error_string(rc));
        return rc;
    }

    rc = noc->write(noc->user, x, y, g_layout.cb_in0_address, tile_a, sizeof(tile_a));
    if (rc == FT_TT_OK)
        rc = noc->write(noc->user, x, y, g_layout.cb_in1_address, tile_b, sizeof(tile_b));
    for (index = 0; index < trial_count && rc == FT_TT_OK; index++) {
        rc = noc->write(noc->user, x, y,
                        g_layout.cb_out_address + index * FT_TT_PHASE2_TILE_BYTES, tile_zero,
                        sizeof(tile_zero));
    }
    if (rc != FT_TT_OK) {
        ft_tt_test_check("operands staged and output wiped", 0, ft_tt_error_string(rc));
        return rc;
    }

    rc = ft_tt_core_wait_sync_registers_ready(noc, x, y, 1000u);
    if (rc == FT_TT_OK)
        rc = stage_cb_counters(noc, x, y, FT_TT_PHASE2_CB_IN0, 1u, 0u);
    if (rc == FT_TT_OK)
        rc = stage_cb_counters(noc, x, y, FT_TT_PHASE2_CB_IN1, 1u, 0u);
    if (rc == FT_TT_OK)
        rc = stage_cb_counters(noc, x, y, FT_TT_PHASE2_CB_OUT, 0u, 0u);
    if (rc != FT_TT_OK) {
        ft_tt_test_check("CB page counters primed", 0, ft_tt_error_string(rc));
        return rc;
    }

    rc = ft_tt_launch_write_launch_message(noc, x, y, &config);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_send_go(noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_wait_done(noc, x, y, RUN_TIMEOUT_MS, NULL);
    ft_tt_test_check_fmt("compute triad completed", rc == FT_TT_OK, "rc=%d (%s), %u trials", rc,
                         ft_tt_error_string(rc), trial_count);
    /* Deliberately NOT a `return rc`. A thread that does not retire does not unwrite the tape
     * the packer already put in L1: the first run of this experiment hung the UNPACK thread
     * with all of MATH's and PACK's work complete and threw the whole measurement away. Read
     * the tapes and let each trial be judged on whether its slots are sane. */
    if (rc != FT_TT_OK)
        printf("        launch did not retire — reading the tapes anyway\n");

    for (index = 0; index < trial_count; index++) {
        const trial *t = &trials[index];
        unsigned char measured[TAPE_SLOTS];
        uint16_t values[TAPE_SLOTS];
        unsigned measured_length;
        unsigned hypothesis;
        int clean = 0;
        unsigned matches = 0;
        unsigned slot;

        rc = noc->read(noc->user, x, y,
                       g_layout.cb_out_address + index * FT_TT_PHASE2_TILE_BYTES, tile_readback,
                       sizeof(tile_readback));
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("tape read back", 0, "%s: %s", t->name, ft_tt_error_string(rc));
            return rc;
        }

        /* The calibration trial DEFINES the three signature values; run it first in a batch. */
        if (t->mode == MODE_LADDER_CYCLE) {
            const unsigned cycle[3] = {t->op0, t->op1, t->l0};

            for (slot = 0; slot < 3u && slot < TAPE_SLOTS; slot++) {
                uint16_t value = 0;

                if (slot_value(tile_readback, slot, &value) && value != 0u &&
                    cycle[slot] <= SYM_MUL)
                    g_symbol_pattern[cycle[slot]] = value;
            }
        }

        measured_length = read_tape(tile_readback, measured, values, &clean);

        printf("        %-22s ", t->name);
        if (t->mode == MODE_MOP)
            printf("O=%u I=%u am=%u ops=[s%s e%s/%s | %s %s | L0=%s L1=%s]  ", t->outer, t->inner,
                   t->addr_mod, symbol_text(t->start0), symbol_text(t->end0),
                   symbol_text(t->end1), symbol_text(t->op0), symbol_text(t->op1),
                   symbol_text(t->l0), symbol_text(t->l1));
        else
            printf("mode=%u k=%u  ", t->mode, t->ladder);
        print_tape("got=", measured, measured_length);

        for (hypothesis = 0; hypothesis < HYPOTHESIS_COUNT; hypothesis++) {
            unsigned char predicted[TAPE_SLOTS];
            unsigned predicted_length =
                predict_observation(t, hypothesis, predicted, TAPE_SLOTS);
            int same = (predicted_length == measured_length) &&
                       (memcmp(predicted, measured, predicted_length) == 0);

            printf("  ");
            print_tape(hypothesis_name[hypothesis], predicted, predicted_length);
            printf("%s", same ? "=OK" : "=NO");
            if (same)
                matches |= 1u << hypothesis;
        }
        printf("%s\n", clean ? "" : "   [SLOT NOT UNIFORM]");

        printf("          raw slots:");
        for (slot = 0; slot < TAPE_SLOTS; slot++)
            printf(" %g", (double)float_from_bfloat16(values[slot]));
        printf("\n");

        /* Only score a trial that actually discriminates: if every hypothesis predicts the
         * same tape, agreement is not evidence. */
        {
            unsigned char reference[TAPE_SLOTS];
            unsigned reference_length = predict_observation(t, 0u, reference, TAPE_SLOTS);
            int discriminating = 0;

            for (hypothesis = 1; hypothesis < HYPOTHESIS_COUNT; hypothesis++) {
                unsigned char other[TAPE_SLOTS];
                unsigned other_length = predict_observation(t, hypothesis, other, TAPE_SLOTS);

                if (other_length != reference_length ||
                    memcmp(other, reference, other_length) != 0)
                    discriminating = 1;
            }
            if (discriminating) {
                g_hypothesis_scored++;
                for (hypothesis = 0; hypothesis < HYPOTHESIS_COUNT; hypothesis++) {
                    if ((matches & (1u << hypothesis)) != 0u)
                        g_hypothesis_matches[hypothesis]++;
                }
            }
            ft_tt_test_check_fmt(t->name, matches != 0u, "%s, matched mask 0x%x",
                                 discriminating ? "discriminating" : "control (all models agree)",
                                 matches);
        }
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* The sweep                                                                 */
/* ------------------------------------------------------------------------- */

#define MOP(nm, o, i, s, e0, e1, a, b, la, lb) \
    {nm, MODE_MOP, (o), (i), 0u, 0u, (s), (e0), (e1), (a), (b), (la), (lb)}

static const trial batch_one[] = {
    /* Calibration FIRST: it defines what bfloat16 pattern each op family writes, and it uses
     * no MOP at all, so it cannot be contaminated by the question under test. */
    {"cal-cycle-k6", MODE_LADDER_CYCLE, 0u, 0u, 6u, 0u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD,
     SYM_SUB, SYM_MUL, SYM_NOP},
    /* The harness must be able to count. Hand-issued ladders, no MOP. */
    {"ctl-ladder-k1", MODE_LADDER_UNIFORM, 0u, 0u, 1u, 0u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD,
     SYM_NOP, SYM_NOP, SYM_NOP},
    {"ctl-ladder-k3", MODE_LADDER_UNIFORM, 0u, 0u, 3u, 0u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD,
     SYM_NOP, SYM_NOP, SYM_NOP},
    {"ctl-ladder-k6", MODE_LADDER_UNIFORM, 0u, 0u, 6u, 0u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD,
     SYM_NOP, SYM_NOP, SYM_NOP},
    {"ctl-ladder-k8", MODE_LADDER_UNIFORM, 0u, 0u, 8u, 0u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD,
     SYM_NOP, SYM_NOP, SYM_NOP},
    /* Nothing is issued at all: the tape must be empty, which is what proves the math thread's
     * ZEROACC really established the initial state and a count is not a leftover. */
    {"ctl-silent", MODE_SILENT, 0u, 0u, 0u, 0u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_NOP, SYM_NOP,
     SYM_NOP, SYM_NOP},
    /* Deliberately wrong: the zero-increment ADDR_MOD, so four issued ops pile into slot 0.
     * Predicted, and wanted, answer: ONE slot. */
    {"ctl-destincr0", MODE_MOP, 1u, 4u, 0u, 1u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP,
     SYM_ADD, SYM_ADD},
    /* The 1-argument constructor, swept in the inner loop. */
    MOP("1arg-O1-I1", 1u, 1u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD, SYM_ADD),
    MOP("1arg-O1-I2", 1u, 2u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD, SYM_ADD),
    MOP("1arg-O1-I4", 1u, 4u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD, SYM_ADD),
    MOP("1arg-O1-I7", 1u, 7u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD, SYM_ADD),
    MOP("1arg-O2-I1", 2u, 1u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD, SYM_ADD),
};

static const trial batch_two[] = {
    /* Re-run the calibration so the symbol map is established in this launch too. */
    {"cal-cycle-k6", MODE_LADDER_CYCLE, 0u, 0u, 6u, 0u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD,
     SYM_SUB, SYM_MUL, SYM_NOP},
    /* 1-argument constructor, swept in the OUTER loop: the two models diverge by O here, not
     * by 1, so the two predictions are different LINES and not one offset. */
    MOP("1arg-O4-I1", 4u, 1u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD, SYM_ADD),
    MOP("1arg-O2-I2", 2u, 2u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD, SYM_ADD),
    MOP("1arg-O2-I3", 2u, 3u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD, SYM_ADD),
    /* The 2-argument constructor, where the substitution is claimed to be an IDENTITY by
     * construction. Separates REPLACE-op0 and ADD from the other two. */
    MOP("2arg-O1-I3", 1u, 3u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_SUB, SYM_SUB, SYM_SUB),
    MOP("2arg-O2-I2", 2u, 2u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_SUB, SYM_SUB, SYM_SUB),
    /* Two real inner ops with the substitution NOT an identity: this is the trial that
     * separates REPLACE-op0 from REPLACE-op1 outright. */
    MOP("2arg-subst-A", 1u, 3u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_SUB, SYM_ADD, SYM_ADD),
    MOP("2arg-subst-M", 2u, 2u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_SUB, SYM_MUL, SYM_MUL),
    /* Distinct loop0_last and loop1_last: also checks the class comment's claim that the
     * OUTER one wins on the final inner iteration of the final outer iteration. */
    MOP("outer-vs-inner-O3", 3u, 2u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_MUL,
        SYM_SUB),
    MOP("outer-vs-inner-O1", 1u, 3u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_MUL,
        SYM_SUB),
    /* NEGATIVE CONTROL WITH TEETH. Both last-instr words are NOP. Under REPLACE the template
     * LOSES one op per outer iteration (2*(3-1) = 4); under ADD nothing changes (2*3 = 6).
     * 4 is a number nobody wants, and it is what this backend's own model predicts. */
    MOP("ctl-lastinstr-NOP", 2u, 3u, SYM_NOP, SYM_NOP, SYM_NOP, SYM_ADD, SYM_NOP, SYM_NOP,
        SYM_NOP),
    /* Where START_OP and END_OP sit relative to the loops — free, and Phase 7/8 needs it. */
    MOP("start-end-place", 2u, 2u, SYM_MUL, SYM_SUB, SYM_NOP, SYM_ADD, SYM_NOP, SYM_ADD,
        SYM_ADD),
};

/* ------------------------------------------------------------------------- */

/**
 * Print what each model predicts, WITHOUT touching the card.
 *
 * Run this before the run and keep the output: it is the same expand_template() that scores
 * the measurement, so the predictions cannot be adjusted after seeing the data.
 */
static void print_predictions(const trial *trials, unsigned trial_count, const char *batch_name)
{
    unsigned index;
    unsigned hypothesis;

    printf("=== PREDICTIONS: %s ===\n", batch_name);
    for (index = 0; index < trial_count; index++) {
        const trial *t = &trials[index];

        printf("  %-22s ", t->name);
        if (t->mode == MODE_MOP)
            printf("O=%u I=%u am=%u [s%s e%s/%s | %s %s | L0=%s L1=%s]", t->outer, t->inner,
                   t->addr_mod, symbol_text(t->start0), symbol_text(t->end0),
                   symbol_text(t->end1), symbol_text(t->op0), symbol_text(t->op1),
                   symbol_text(t->l0), symbol_text(t->l1));
        else
            printf("mode=%u k=%u", t->mode, t->ladder);
        for (hypothesis = 0; hypothesis < HYPOTHESIS_COUNT; hypothesis++) {
            unsigned char predicted[TAPE_SLOTS];
            unsigned predicted_length =
                predict_observation(t, hypothesis, predicted, TAPE_SLOTS);

            printf("   ");
            print_tape(hypothesis_name[hypothesis], predicted, predicted_length);
        }
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_program_directory loaded;
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    const char *directory = (argc > 1) ? argv[1] : NULL;
    uint16_t x = 0;
    uint16_t y = 0;
    unsigned hypothesis;
    int rc;

    if (directory == NULL) {
        fprintf(stderr, "usage: %s <program-directory> | --predict\n", argv[0]);
        return 2;
    }
    if (strcmp(directory, "--predict") == 0) {
        print_predictions(batch_one, (unsigned)(sizeof(batch_one) / sizeof(batch_one[0])),
                          "batch 1");
        print_predictions(batch_two, (unsigned)(sizeof(batch_two) / sizeof(batch_two[0])),
                          "batch 2");
        return 0;
    }

    build_operand_tiles();

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    rc = ft_tt_hardware_session_tensix_from_environment(&session, &x, &y, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    ft_tt_test_step("setup — boot the firmware and load the MOP-expander probe triad");
    rc = boot_and_derive_layout(&session.noc, x, y, &firmware);
    ft_tt_test_check_fmt("firmware booted", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    rc = ft_tt_program_directory_load(directory, 0x1cu, &loaded);
    ft_tt_test_check_fmt("probe triad images read", rc == FT_TT_OK, "%s: %s", directory,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    rc = ft_tt_core_load_program(&session.noc, x, y, &loaded.program, placements);
    ft_tt_test_check_fmt("probe triad loaded", rc == FT_TT_OK, "%u images",
                         loaded.program.image_count);
    if (rc != FT_TT_OK)
        goto done;

    rc = run_batch(&session.noc, x, y, placements, batch_one,
                   (unsigned)(sizeof(batch_one) / sizeof(batch_one[0])),
                   "batch 1 — calibration, harness controls, and the inner-loop sweep");
    if (rc != FT_TT_OK)
        goto done;

    rc = run_batch(&session.noc, x, y, placements, batch_two,
                   (unsigned)(sizeof(batch_two) / sizeof(batch_two[0])),
                   "batch 2 — outer-loop sweep, the 2-argument form, and the negative control");
    if (rc != FT_TT_OK)
        goto done;

    ft_tt_test_step("verdict");
    printf("        symbol calibration: ADD=%g SUB=%g MUL=%g\n",
           (double)float_from_bfloat16(g_symbol_pattern[SYM_ADD]),
           (double)float_from_bfloat16(g_symbol_pattern[SYM_SUB]),
           (double)float_from_bfloat16(g_symbol_pattern[SYM_MUL]));
    for (hypothesis = 0; hypothesis < HYPOTHESIS_COUNT; hypothesis++)
        printf("        %-26s %u / %u discriminating trials\n", hypothesis_name[hypothesis],
               g_hypothesis_matches[hypothesis], g_hypothesis_scored);
    {
        unsigned survivors = 0;

        for (hypothesis = 0; hypothesis < HYPOTHESIS_COUNT; hypothesis++) {
            if (g_hypothesis_matches[hypothesis] == g_hypothesis_scored && g_hypothesis_scored > 0u)
                survivors++;
        }
        ft_tt_test_check_fmt("exactly one rule survives every discriminating trial",
                             survivors == 1u, "%u survivor(s) of %u over %u trials", survivors,
                             HYPOTHESIS_COUNT, g_hypothesis_scored);
    }

done:
    (void)ft_tt_core_park(&session.noc, x, y);
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
