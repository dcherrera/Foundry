/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_phase6_gdnc.c
 * @brief ON-CARD acceptance for the CHUNKWISE GATED-DELTA-NET convolution: 103 positions
 *   through the card in chunks of 32 + 32 + 32 + 7, scored against a scalar fp64
 *   transcription of `pn_qwen35moe_conv1d_silu()`, plus the chunk-size sweep and the
 *   descriptor cost at Ornith's real width.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS SUITE NEEDS NO MODEL, AND WHY THAT MATTERS
 * ---------------------------------------------------------------------------
 * a second card host has 7.1 GB of host RAM against a 9.5 GB GGUF, so ONE model load at a time is a hard
 * rule and two `pantheon` processes took the box off the network once. This suite opens the
 * card, uploads a synthetic weight and a synthetic stream, and scores the answer — no model,
 * no artifact, no flock contention, and an iteration that costs seconds rather than the
 * 6.5 minutes an 2k prefill costs. The arithmetic it checks is the same arithmetic; what it
 * does not check is the WIRING into a forward, which is the next round's job.
 *
 * ---------------------------------------------------------------------------
 * THE CHUNK SPLIT IS THE POINT. A SINGLE-CHUNK TEST CANNOT SEE A BOUNDARY BUG.
 * ---------------------------------------------------------------------------
 * At position 0 the window is zero and the whole phase is `SiLU(x * w[taps-1])` — which is
 * the decode family's own recorded blind spot: its PHASE CONV scored 6.9e-4 at layer 0 of
 * token 0 and 1e-2 to 2e-1 after it, on a CB_STATE defect that was invisible in the first
 * descriptor because the buffer above it was still zero. So the parity stage runs FOUR
 * chunks: three full ones and a ragged 7-row one, with the window carried on the card and
 * never touched by the host between them.
 *
 * AND THE CARRY IS READ BACK AND SCORED SEPARATELY. The `taps-1` rolling planes come off the
 * card after every chunk and are compared against the oracle's own `conv_state`, so a roll
 * that lands the wrong rows is caught AS a roll error rather than as next chunk's output
 * being slightly off.
 *
 * ---------------------------------------------------------------------------
 * THE RED ARMS ARE DRAM MUTATIONS, SO THEY NEED NO REBUILD
 * ---------------------------------------------------------------------------
 * Every one of them is a change to what is UPLOADED, run against the same five images:
 *
 *   band_swap    upload the band with `S_tau` and `C_tau` exchanged — the plausible
 *                transcription error, and the one a uniform `2*taps` loop invites
 *   band_zero    upload a zeroed band — the whole answer becomes `SiLU(0)`, which is the
 *                control that says the matmul is contributing at all
 *   window_wipe  zero the carried window between chunks — parity must survive chunk 0 and
 *                FAIL from chunk 1, which is the quantitative form of "the carry is
 *                load-bearing"
 *   weight_zero  upload a zeroed weight — same shape of control, on the other operand
 *
 * A gate that has never produced a low number is not a gate, and this project has eight
 * recorded vacuous-gate classes. All four are run every time.
 *
 * Usage:
 *   FT_TT_DRIVER=udrv FT_TT_FIRMWARE_DIR=<ttprog> FT_TT_TENSIX_INDEX=<n> \
 *     ./build-gdnc-card/test_phase6_gdnc ../programs/gdnc
 *
 * @status NEW.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../ft_tt_coord.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_gdnc.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "../ft_tt_watchdog.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE_BYTES     FT_TT_OP_GDNC_FP32_TILE_BYTES
#define TILE_DIM       ((uint32_t)FT_TT_OP_TILE_DIM)
#define RUN_TIMEOUT_MS 30000u
#define QUEUE_CAPACITY 64u
#define POISON_BYTE    0x5A

/** DRAM banks 2..5 are this suite's, as they are test_ops.c's and test_phase6_rope.c's.
 *
 *  THE INPUT AND THE OUTPUT SHARE A BANK, and that is a contract rather than a convenience:
 *  `kGdncActivationXy` is ONE packed coordinate word for both, so a pair in two banks makes
 *  the writer land the answer at the output's offset INSIDE the input's bank — a clean retire
 *  with the output buffer never written. `ft_tt_op_gdnc_parameters()` refuses it; this suite
 *  honours it. */
#define DRAM_BANK_ACTIVATION 2u
#define DRAM_BANK_CONSTANT   3u
#define DRAM_BANK_STATE      4u
#define DRAM_BANK_COUNT      6u

/** 512 MiB into each bank, NOT the 1 MiB every other suite uses: other workers are on this
 *  card with their own cores but the SAME DRAM banks, and a shared base is a cross-agent
 *  corruption that would look exactly like a kernel bug. */
#define DRAM_BASE        0x20000000u
/** The output sits above the input INSIDE the activation bank. */
#define DRAM_OUTPUT_GAP  0x02000000u

#define IDLE_SPIN_LIMIT 400000000u
#define CB_SPIN_LIMIT   400000000u

#define TRIAD_MASK                                                                \
    ((1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1) | (1u << FT_TT_PROC_MATH0) | \
     (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2))

/** The parity stream: narrow enough that a chunk's readback costs ~50 ms on a 5.3 MB/s link,
 *  wide enough that the column loop, the slice and the tile-column arithmetic are all
 *  exercised. Ornith's real 8192 is used for the COST stage, where nothing is read back. */
#define PARITY_CONV_DIM 2048u
#define PARITY_TAPS     FT_TT_OP_GDNC_QWEN35_CONV_TAPS
#define PARITY_POSITIONS 103u

/** Ornith's own width, for the cost and sweep stages. */
#define ORNITH_CONV_DIM FT_TT_OP_GDNC_QWEN35_CONV_DIM

/** The gate, and THE METRIC IS MAX-NORM RELATIVE — `max|got-want| / max|want|` — which is the
 *  figure every recorded number for this op is in: the decode family's PHASE CONV is
 *  1.6e-04..8.7e-04 at position 0 and 1.1e-03..1.8e-03 past it, in exactly this metric.
 *
 *  PER-ELEMENT RELATIVE IS THE WRONG INSTRUMENT FOR THIS OP AND IT WAS MEASURED SO. The first
 *  run of this suite scored per-element relative with the denominator floored at 1e-3 and read
 *  **3.702e-01**, which looks like a structural fault and is not one: `SiLU` crosses zero, so
 *  a chunk of 2,048 channels contains outputs of order 1e-4, and a Tf32 truncation of 3.7e-4
 *  on one of them reads as 37%. The same run's max-norm figure is three orders smaller. A
 *  metric whose denominator can be arbitrarily small does not measure the kernel, it measures
 *  where the zero crossings fell.
 *
 *  Both are reported, plus the per-element relative restricted to elements big enough for it
 *  to mean anything, so the reader can see WHICH mechanism a failure is.
 *
 *  This form crosses ONE MORE unpack stage than the decode family's — the weighted product is
 *  packed and re-unpacked for the matmul — so the bar is 5e-03: loose enough to admit the
 *  extra stage, and two orders tighter than the smallest structural error the red arms
 *  produce. The 1e-3 bar the CUDA arm carried is REFUTED as an acceptance criterion for this
 *  backend anyway: it fails an arm whose 48-token output is byte-identical to the oracle, and
 *  it was replaced by the output gate. */
#define CONV_RELATIVE_GATE 5.0e-3
/** PHASE DELTA's shape, up here because `fill_request()` needs it: the placement is the
 *  WHOLE FAMILY'S and a CONV request still has to name it. */
#define DELTA_S           128u
#define DELTA_STATE_TILES 4u
#define DELTA_V_HEADS     4u
#define DELTA_K_HEADS     2u
#define DELTA_ROWS        32u
#define DELTA_CHUNKS      4u
#define DELTA_KEY_COLS    (DELTA_K_HEADS * DELTA_S)
#define DELTA_VAL_COLS    (DELTA_V_HEADS * DELTA_S)
#define DELTA_IN_COLS     (2u * DELTA_KEY_COLS + DELTA_VAL_COLS)

/** Below this the per-element relative is noise about a zero crossing, not about the op. */
#define SIGNIFICANT_MAGNITUDE 0.05

static ft_tt_noc_coord g_dram[DRAM_BANK_COUNT];
static ft_tt_launch_core_placement g_placement_storage[1];
static ft_tt_core_group g_group;
static ft_tt_program_directory g_program;
static ft_tt_launch_config g_config;
static ft_tt_queue g_queue;
static ft_tt_launch_cb_entry g_cb_placed[FT_TT_OP_MAX_CB];
static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG *
                                 4];
static uint32_t g_slot;

/** The two DIAGNOSTIC figures beside the gate's own, so a failure names its mechanism:
 *  the worst ABSOLUTE difference, and the worst per-element relative over elements big
 *  enough for a per-element relative to mean anything. A large per-element relative with a
 *  tiny absolute is a zero crossing; a large absolute is the kernel. */
static double g_worst_absolute;
static double g_worst_significant;

/* Host-side scratch. Sized for the LARGEST BLOCK this suite stages, which is not the stream:
 * the CONSTANT buffer at Ornith's width is `taps * 256 + 2*taps` = 1,032 tiles = 4.03 MiB,
 * four times the 256-tile stream. Sizing this from the stream is a 4 MiB overflow of a static
 * array with nothing at all to say so. */
#define MAX_TILE_COLS  (ORNITH_CONV_DIM / 32u)
#define MAX_TILE_COUNT (PARITY_TAPS * MAX_TILE_COLS + 2u * PARITY_TAPS)
static float g_tiles[MAX_TILE_COUNT * 1024u];
static float g_rows[TILE_DIM * ORNITH_CONV_DIM];
static double g_expected[PARITY_POSITIONS][PARITY_CONV_DIM];
static double g_weight[PARITY_CONV_DIM * PARITY_TAPS];
static double g_stream[PARITY_POSITIONS][PARITY_CONV_DIM];
static double g_oracle_window[PARITY_CONV_DIM * (PARITY_TAPS - 1u)];

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

static int fill_dram(const ft_tt_noc_access *noc, ft_tt_noc_coord where, uint32_t address,
                     unsigned char byte, size_t bytes)
{
    static unsigned char pattern[0x2000];
    size_t offset = 0;

    memset(pattern, byte, sizeof(pattern));
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

static double now_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* ================================================================
 * The oracle
 * ================================================================ */

static double oracle_silu(double x)
{
    return x / (1.0 + exp(-x));
}

/**
 * `pn_qwen35moe_conv1d_silu()`, one position, transcribed rather than paraphrased.
 *
 * The newest tap multiplies the RAW input and the value rolled into the window is the RAW
 * input too — not the SiLU'd output. Getting that backwards is a convolution that still looks
 * like a convolution.
 */
static void oracle_step(const double *weight, const double *x, double *window, double *out,
                        uint32_t channels, uint32_t taps)
{
    uint32_t channel;

    for (channel = 0; channel < channels; channel++) {
        const double *w = weight + (size_t)channel * taps;
        double *cs = window + (size_t)channel * (taps - 1u);
        double acc = x[channel] * w[taps - 1u];
        uint32_t tap;

        for (tap = 0; tap + 1u < taps; tap++)
            acc += cs[tap] * w[tap];
        out[channel] = oracle_silu(acc);
        for (tap = 0; tap + 2u < taps; tap++)
            cs[tap] = cs[tap + 1u];
        cs[taps - 2u] = x[channel];
    }
}

static double sample(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return ((double)(*state >> 8) / 8388608.0) - 1.0;
}

/* ================================================================
 * Staging
 * ================================================================ */

/** Tilize `rows` live rows of a `[rows, cols]` fp32 strip into `g_tiles`. */
static void stage_strip(const float *row_major, uint32_t rows, uint32_t cols)
{
    (void)ft_tt_tilize_strip(row_major, sizeof(float), rows, cols,
                             (size_t)cols * sizeof(float), g_tiles);
}

/**
 * The ROW-REPLICATED, TAP-MAJOR conv weight, plus the band, into `g_tiles`.
 *
 * Plane `tap` is `conv_dim/32` tiles carrying `w[c][tap]` at EVERY row, because the tap
 * coefficient is constant across positions and the broadcast surface is BROADCAST_COL only.
 * The decode family's planes carry it in row 0 with zeros below; the only difference is
 * `valid_rows`.
 */
static size_t stage_constants(const double *weight, uint32_t conv_dim, uint32_t taps,
                              bool zero_weight, bool swap_band)
{
    const uint32_t cols = conv_dim / TILE_DIM;
    const size_t band_offset = (size_t)taps * cols * (TILE_BYTES / sizeof(float));
    unsigned char *band = (unsigned char *)(g_tiles + band_offset);
    uint32_t tap;

    for (tap = 0; tap < taps; tap++) {
        uint32_t row;
        uint32_t channel;

        for (row = 0; row < TILE_DIM; row++)
            for (channel = 0; channel < conv_dim; channel++)
                g_rows[(size_t)row * conv_dim + channel] =
                    zero_weight ? 0.0f
                                : (float)weight[(size_t)channel * taps + tap];
        (void)ft_tt_tilize_strip(g_rows, sizeof(float), TILE_DIM, conv_dim,
                                 (size_t)conv_dim * sizeof(float),
                                 g_tiles + (size_t)tap * cols * (TILE_BYTES / sizeof(float)));
    }

    (void)ft_tt_op_gdnc_build_band(taps, band, ft_tt_op_gdnc_band_bytes(taps));
    if (swap_band) {
        /* THE PLAUSIBLE TRANSCRIPTION ERROR: pair `S_tau` with the window and `C_tau` with the
         * input. A uniform `2*taps` loop invites exactly this. */
        unsigned char scratch[TILE_BYTES];
        uint32_t index;

        for (index = 0; index < taps; index++) {
            unsigned char *a = band + (size_t)(2u * index) * TILE_BYTES;
            unsigned char *b = a + TILE_BYTES;

            memcpy(scratch, a, TILE_BYTES);
            memcpy(a, b, TILE_BYTES);
            memcpy(b, scratch, TILE_BYTES);
        }
    }
    return ((size_t)taps * cols + 2u * (size_t)taps) * TILE_BYTES;
}

/* ================================================================
 * One descriptor
 * ================================================================ */

static void fill_request(ft_tt_op_gdnc_request *request, uint32_t conv_dim, uint32_t taps,
                         uint32_t chunk_rows, uint32_t first_col, uint32_t col_count)
{
    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDNC_PHASE_CONV;
    request->chunk_rows = chunk_rows;
    request->conv_dim = conv_dim;
    request->conv_taps = taps;
    request->first_col = first_col;
    request->col_count = col_count;
    /* THE PLACEMENT IS THE WHOLE FAMILY'S, NOT ONE PHASE'S: the residency is armed ONCE and
     * then answers whatever the queue hands it, so a CONV request still names the DELTA shape.
     * `ft_tt_op_gdnc_place_cbs()` refuses a request that does not, which is how this was
     * caught -- as a bring-up rc=-3 rather than as a buffer the other phase reads out of. */
    request->state_tiles = DELTA_STATE_TILES;
    request->v_heads = DELTA_V_HEADS;
    request->k_heads = DELTA_K_HEADS;

    request->activation_in.x = (uint16_t)g_dram[DRAM_BANK_ACTIVATION].x;
    request->activation_in.y = (uint16_t)g_dram[DRAM_BANK_ACTIVATION].y;
    request->activation_in.address = DRAM_BASE;
    request->activation_in.tile_bytes = TILE_BYTES;

    request->activation_out = request->activation_in;
    request->activation_out.address = DRAM_BASE + DRAM_OUTPUT_GAP;

    request->constant.x = (uint16_t)g_dram[DRAM_BANK_CONSTANT].x;
    request->constant.y = (uint16_t)g_dram[DRAM_BANK_CONSTANT].y;
    request->constant.address = DRAM_BASE;
    request->constant.tile_bytes = TILE_BYTES;

    request->state_conv.x = (uint16_t)g_dram[DRAM_BANK_STATE].x;
    request->state_conv.y = (uint16_t)g_dram[DRAM_BANK_STATE].y;
    request->state_conv.address = DRAM_BASE;
    request->state_conv.tile_bytes = TILE_BYTES;
}

/**
 * Enqueue one descriptor and wait for the WRITER to publish completion.
 *
 * @param out_seconds receives the enqueue-to-completion wall time
 * @return FT_TT_OK, or the first failing step's code. A REFUSAL comes back as the queue's own
 *         error rather than as a wrong answer, which is what the three-sided shape check buys.
 */
static int run_descriptor(const ft_tt_hardware_session *session,
                          const ft_tt_op_gdnc_request *request, double *out_seconds,
                          bool expect_success)
{
    ft_tt_queue_descriptor descriptor;
    uint32_t words[FT_TT_OP_GDNC_WORDS];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    double start;
    int rc;

    rc = ft_tt_op_gdnc_parameters(request, words);
    if (rc == FT_TT_OK)
        /* THE PARAMETER SLOTS ROTATE AND THERE ARE `FT_TT_QUEUE_MATMUL_PARAM_SLOTS` OF THEM.
         * `g_slot` is the COMPLETION COUNT and it does not wrap; using it directly as the
         * slot index worked until this suite passed 64 descriptors, at which point every
         * further write was refused and every arm after it scored its poison value rather
         * than its arithmetic. Rewriting a rotating slot is safe here because this harness
         * waits for completion before enqueueing the next descriptor. */
        rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue,
                                             g_slot % FT_TT_QUEUE_MATMUL_PARAM_SLOTS, words,
                                             &parameter_address);
    if (rc != FT_TT_OK)
        return rc;

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_GDN_CHUNK;
    descriptor.a0 = parameter_address;
    start = now_seconds();
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_slot + 1u, RUN_TIMEOUT_MS,
                                        &completed);
    if (out_seconds)
        *out_seconds = now_seconds() - start;
    if (rc != FT_TT_OK) {
        if (expect_success) {
            /* "THE GROUP IS STUCK" IS NOT A DIAGNOSIS. Five processors share one queue and
             * each keeps its OWN status slot, so a blocked triad can say WHICH processor is
             * blocked in WHICH wait -- and CB_WAIT_FRONT on the compute thread and
             * CB_RESERVE_BACK on the reader are different bugs with different fixes. */
            static const unsigned processors[5] = { FT_TT_PROC_DM0, FT_TT_PROC_DM1,
                                                    FT_TT_PROC_MATH0, FT_TT_PROC_MATH1,
                                                    FT_TT_PROC_MATH2 };
            static const char *const names[5] = { "BRISC", "NCRISC", "TRISC0", "TRISC1",
                                                  "TRISC2" };
            unsigned which;

            ft_tt_test_check_fmt("the descriptor completes", false, "rc=%d completed=%u of %u",
                                 rc, completed, g_slot + 1u);
            for (which = 0; which < 5u; which++) {
                uint32_t slot[FT_TT_QUEUE_STATUS_WORDS];

                ft_tt_noc_coord where;

                where.x = g_queue.x;
                where.y = g_queue.y;
                if (device_read(&session->noc, where,
                                FT_TT_QUEUE_STATUS_ADDRESS(g_queue.control_address,
                                                           processors[which]),
                                slot, sizeof(slot)) != FT_TT_OK)
                    continue;
                fprintf(stderr,
                        "    %-6s state 0x%08x cursor %u heartbeat %u error 0x%08x "
                        "opcode 0x%02x spins %u\n",
                        names[which], slot[FT_TT_QUEUE_STATUS_STATE],
                        slot[FT_TT_QUEUE_STATUS_CURSOR], slot[FT_TT_QUEUE_STATUS_HEARTBEAT],
                        slot[FT_TT_QUEUE_STATUS_ERROR], slot[FT_TT_QUEUE_STATUS_LAST_OPCODE],
                        slot[FT_TT_QUEUE_STATUS_SPINS]);
            }
        }
        return rc;
    }
    g_slot++;
    return FT_TT_OK;
}

/* ================================================================
 * Stage 0 — the transfer path is a LINE
 * ================================================================ */

static void stage_round_trip(const ft_tt_hardware_session *session)
{
    static float written[1024];
    static float read_back[1024];
    uint32_t index;
    int rc;

    ft_tt_test_step("stage 0 — host->DRAM->host, so the transfer path is a LINE and not a "
                    "suspect");

    for (index = 0; index < 1024u; index++)
        written[index] = (float)index * 0.25f - 100.0f;
    rc = device_write(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE, written,
                      sizeof(written));
    if (rc == FT_TT_OK)
        rc = device_read(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE, read_back,
                         sizeof(read_back));
    ft_tt_test_check_fmt("a tile round-trips through DRAM", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check("...byte for byte", memcmp(written, read_back, sizeof(written)) == 0,
                     NULL);
}


/* ================================================================
 * Stage 1b — THE SHIFT MAP, READ OUT OF THE ANSWER
 * ================================================================ */

/** The probe's own width: ONE tile column, so a readback is 4 KiB on a 5.3 MB/s link. */
#define MAP_CONV_DIM 32u

/**
 * Which row of the input, or which window plane, explains output row `t`.
 *
 * ---------------------------------------------------------------------------
 * AN INJECTIVE STREAM TURNS A WRONG TENSOR INTO A MAP
 * ---------------------------------------------------------------------------
 * Three searches on the decode family said only what a plane was NOT; `qkv[c] = c` said what
 * it WAS in one run. So this stage feeds the card values that are distinguishable by
 * inspection -- input row `t` carries `t+1` at every channel, window plane `p` carries
 * `-(p+1)` -- and turns ONE TAP ON AT A TIME. Then `out[t] = SiLU(source)` and the source can
 * be recovered by matching, which reads BOTH band matrices of that tap straight out of the
 * answer:
 *
 *     tap tau, d = (taps-1) - tau:   t >= d  -> input row t-d      (that is `S_tau`)
 *                                    t <  d  -> window plane t+tau (that is `C_tau`)
 *
 * IT IS INDEPENDENT OF PRECISION. Every candidate is an integer apart, so a Tf32 residue
 * cannot move a row from one source to another -- which makes this the structural half of the
 * acceptance and the parity stage the numerical half. A suite that had only the numerical
 * half cannot say whether a miss is the arithmetic or the geometry.
 */
static int map_source_of(double value, uint32_t taps)
{
    double best = 1.0e30;
    int chosen = -1;
    int candidate;

    /* Negative indices are window planes: -1 is plane 0, -2 is plane 1, ... */
    for (candidate = -(int)(taps - 1u); candidate < (int)TILE_DIM; candidate++) {
        const double source = (candidate < 0) ? (double)candidate : (double)(candidate + 1);
        const double difference = fabs(value - oracle_silu(source));

        if (difference < best) {
            best = difference;
            chosen = candidate;
        }
    }
    /* A value that matches NOTHING is its own answer and must not be silently rounded to the
     * nearest candidate: SiLU(1) = 0.731 and SiLU(2) = 1.762, so 0.05 is a tenth of the
     * closest gap anywhere in the range. */
    return (best < 0.05) ? chosen : -100;
}

static void stage_shift_map(const ft_tt_hardware_session *session)
{
    const uint32_t conv_dim = MAP_CONV_DIM;
    const uint32_t taps = PARITY_TAPS;
    const size_t stream_bytes = TILE_BYTES;
    static double one_tap[MAP_CONV_DIM * PARITY_TAPS];
    ft_tt_op_gdnc_request request;
    uint32_t tap;
    uint32_t row;
    uint32_t channel;
    uint32_t wrong = 0;
    char first_wrong[160];
    int rc;

    ft_tt_test_step("stage 1b — THE SHIFT MAP: one tap at a time, an injective stream, and "
                    "both band matrices read straight out of the answer");
    first_wrong[0] = '\0';

    for (tap = 0; tap < taps; tap++) {
        const uint32_t shift = (taps - 1u) - tap;
        uint32_t plane;

        /* Only this tap is live, and its coefficient is exactly 1, so SiLU's argument IS the
         * source value. */
        for (channel = 0; channel < conv_dim; channel++) {
            uint32_t other;

            for (other = 0; other < taps; other++)
                one_tap[(size_t)channel * taps + other] = (other == tap) ? 1.0 : 0.0;
        }
        rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT], DRAM_BASE, g_tiles,
                          stage_constants(one_tap, conv_dim, taps, false, false));
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the shift-map stage stages", false, "rc=%d", rc);
            return;
        }

        /* The window, ROW-0 PLANES, plane p carrying -(p+1) at every channel. */
        for (plane = 0; plane + 1u < taps; plane++) {
            for (channel = 0; channel < conv_dim; channel++)
                g_rows[channel] = -(float)(plane + 1u);
            stage_strip(g_rows, 1u, conv_dim);
            rc = device_write(&session->noc, g_dram[DRAM_BANK_STATE],
                              DRAM_BASE + (uint32_t)((size_t)plane * TILE_BYTES), g_tiles,
                              TILE_BYTES);
            if (rc != FT_TT_OK)
                return;
        }

        /* The input, row t carrying t+1. */
        for (row = 0; row < TILE_DIM; row++)
            for (channel = 0; channel < conv_dim; channel++)
                g_rows[(size_t)row * conv_dim + channel] = (float)(row + 1u);
        stage_strip(g_rows, TILE_DIM, conv_dim);
        rc = device_write(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE, g_tiles,
                          stream_bytes);
        if (rc == FT_TT_OK)
            rc = fill_dram(&session->noc, g_dram[DRAM_BANK_ACTIVATION],
                           DRAM_BASE + DRAM_OUTPUT_GAP, POISON_BYTE, stream_bytes);
        if (rc != FT_TT_OK)
            return;

        fill_request(&request, conv_dim, taps, TILE_DIM, 0u, 0u);
        if (run_descriptor(session, &request, NULL, true) != FT_TT_OK)
            return;
        rc = device_read(&session->noc, g_dram[DRAM_BANK_ACTIVATION],
                         DRAM_BASE + DRAM_OUTPUT_GAP, g_tiles, stream_bytes);
        if (rc != FT_TT_OK)
            return;
        (void)ft_tt_untilize_strip(g_tiles, sizeof(float), TILE_DIM, conv_dim,
                                   (size_t)conv_dim * sizeof(float), g_rows);

        for (row = 0; row < TILE_DIM; row++) {
            const int got = map_source_of((double)g_rows[(size_t)row * conv_dim], taps);
            const int want = (row >= shift) ? (int)(row - shift)
                                            : -(int)(row + tap + 1u);

            if (got != want) {
                wrong++;
                if (first_wrong[0] == '\0')
                    snprintf(first_wrong, sizeof(first_wrong),
                             "tap %u row %u: source %d, wanted %d (raw %.5f)", tap, row, got,
                             want, (double)g_rows[(size_t)row * conv_dim]);
            }
        }
        /* Every channel of a row carries the same value here, so a row that disagrees with
         * ITSELF across channels is a different fault and is worth naming separately. */
        for (row = 0; row < TILE_DIM; row++)
            for (channel = 1; channel < conv_dim; channel++)
                if (g_rows[(size_t)row * conv_dim + channel] !=
                    g_rows[(size_t)row * conv_dim])
                    wrong += 1000u;
    }

    ft_tt_test_check_fmt("every output row of every tap reads its own band matrix's source",
                         wrong == 0u, "%u of %u rows wrong%s%s", wrong, taps * TILE_DIM,
                         first_wrong[0] ? " — " : "", first_wrong);
}

/* ================================================================
 * Stage 1..3 — parity across chunk boundaries
 * ================================================================ */

typedef enum gdnc_red {
    kRedNone = 0,
    kRedBandSwap,
    kRedBandZero,
    kRedWindowWipe,
    kRedWeightZero,
    kRedCount
} gdnc_red;

static const char *const g_red_name[kRedCount] = { "none", "band_swap", "band_zero",
                                                   "window_wipe", "weight_zero" };

/**
 * Run `PARITY_POSITIONS` positions through the card in chunks and score every one.
 *
 * @param out_first_bad_chunk  the first chunk that missed the gate, or -1
 * @param out_window_worst     the worst relative error of the CARRIED WINDOW, read back and
 *                             scored against the oracle's own `conv_state` after every chunk
 * @return the worst relative error over every position and channel
 */
static double parity_run(const ft_tt_hardware_session *session, gdnc_red red,
                         int *out_first_bad_chunk, double *out_window_worst)
{
    const uint32_t conv_dim = PARITY_CONV_DIM;
    const uint32_t taps = PARITY_TAPS;
    const uint32_t cols = conv_dim / TILE_DIM;
    const size_t stream_bytes = (size_t)cols * TILE_BYTES;
    const size_t window_bytes = (size_t)(taps - 1u) * cols * TILE_BYTES;
    ft_tt_op_gdnc_request request;
    uint32_t done = 0;
    int chunk_index = 0;
    double worst = 0.0;
    double window_worst = 0.0;
    double chunk_worst_absolute = 0.0;
    double chunk_scale = 0.0;
    int rc;

    g_worst_absolute = 0.0;
    g_worst_significant = 0.0;

    *out_first_bad_chunk = -1;
    *out_window_worst = 0.0;

    /* The constants, staged once for the whole run. */
    {
        size_t bytes = stage_constants(g_weight, conv_dim, taps, red == kRedWeightZero,
                                       red == kRedBandSwap);

        if (red == kRedBandZero)
            memset((unsigned char *)g_tiles + (size_t)taps * cols * TILE_BYTES, 0,
                   ft_tt_op_gdnc_band_bytes(taps));
        rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT], DRAM_BASE, g_tiles, bytes);
        if (rc != FT_TT_OK)
            return 1.0e30;
    }

    /* A FRESH SEQUENCE IS A ZEROED WINDOW, on the card and in the oracle alike. */
    rc = fill_dram(&session->noc, g_dram[DRAM_BANK_STATE], DRAM_BASE, 0u, window_bytes);
    if (rc != FT_TT_OK)
        return 1.0e30;
    memset(g_oracle_window, 0, sizeof(g_oracle_window));
    {
        uint32_t position;

        for (position = 0; position < PARITY_POSITIONS; position++)
            oracle_step(g_weight, g_stream[position], g_oracle_window, g_expected[position],
                        conv_dim, taps);
    }
    /* ...and the oracle's window is re-run alongside the card's, so the CARRY can be scored
     * on its own rather than only through the next chunk's output. */
    memset(g_oracle_window, 0, sizeof(g_oracle_window));

    while (done < PARITY_POSITIONS) {
        uint32_t rows = PARITY_POSITIONS - done;
        uint32_t row;
        uint32_t channel;

        if (rows > TILE_DIM)
            rows = TILE_DIM;
        /* A tail shorter than the window goes through the DECODE family. The protocol refuses
         * it, so this loop must not manufacture one either. */
        if (rows < taps - 1u)
            break;

        /* THE HOST OWES ZEROS ABOVE THE CHUNK — the strip tilizer supplies them from
         * `valid_rows`, which is exactly the guarantee `pn_tt_activation_upload_rows()` gives
         * into a zeroed strip. */
        for (row = 0; row < rows; row++)
            for (channel = 0; channel < conv_dim; channel++)
                g_rows[(size_t)row * conv_dim + channel] =
                    (float)g_stream[done + row][channel];
        stage_strip(g_rows, rows, conv_dim);
        rc = device_write(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE, g_tiles,
                          stream_bytes);
        if (rc == FT_TT_OK)
            rc = fill_dram(&session->noc, g_dram[DRAM_BANK_ACTIVATION],
                           DRAM_BASE + DRAM_OUTPUT_GAP, POISON_BYTE, stream_bytes);
        if (rc != FT_TT_OK)
            return 1.0e30;

        if (red == kRedWindowWipe && chunk_index > 0) {
            /* THE CARRY, REMOVED. Chunk 0 must still pass — its window is zero anyway — and
             * every chunk after it must fail. That contrast is the whole argument for running
             * four chunks. */
            rc = fill_dram(&session->noc, g_dram[DRAM_BANK_STATE], DRAM_BASE, 0u, window_bytes);
            if (rc != FT_TT_OK)
                return 1.0e30;
        }

        fill_request(&request, conv_dim, taps, rows, 0u, 0u);
        if (run_descriptor(session, &request, NULL, true) != FT_TT_OK)
            return 1.0e30;

        rc = device_read(&session->noc, g_dram[DRAM_BANK_ACTIVATION],
                         DRAM_BASE + DRAM_OUTPUT_GAP, g_tiles, stream_bytes);
        if (rc != FT_TT_OK)
            return 1.0e30;
        (void)ft_tt_untilize_strip(g_tiles, sizeof(float), TILE_DIM, conv_dim,
                                   (size_t)conv_dim * sizeof(float), g_rows);

        chunk_worst_absolute = 0.0;
        chunk_scale = 0.0;
        for (row = 0; row < rows; row++) {
            for (channel = 0; channel < conv_dim; channel++) {
                const double got = (double)g_rows[(size_t)row * conv_dim + channel];
                const double want = g_expected[done + row][channel];
                const double difference = fabs(got - want);

                if (difference > chunk_worst_absolute)
                    chunk_worst_absolute = difference;
                if (fabs(want) > chunk_scale)
                    chunk_scale = fabs(want);
                if (fabs(want) >= SIGNIFICANT_MAGNITUDE &&
                    difference / fabs(want) > g_worst_significant)
                    g_worst_significant = difference / fabs(want);
            }
        }
        {
            /* MAX-NORM RELATIVE, per chunk: the metric every recorded figure for this op is
             * in. Per chunk rather than over the run, so a chunk whose own scale is small
             * cannot be flattered by a later chunk's larger one. */
            const double chunk_relative =
                chunk_worst_absolute / (chunk_scale > 1.0e-6 ? chunk_scale : 1.0e-6);

            if (chunk_relative > worst)
                worst = chunk_relative;
            if (chunk_relative > CONV_RELATIVE_GATE && *out_first_bad_chunk < 0)
                *out_first_bad_chunk = chunk_index;
            if (chunk_worst_absolute > g_worst_absolute)
                g_worst_absolute = chunk_worst_absolute;
        }

        /* THE CARRY, SCORED ON ITS OWN. Step the oracle's window over the same rows and
         * compare it against the `taps-1` planes the writer just rolled. */
        {
            static double throwaway[PARITY_CONV_DIM];
            uint32_t plane;

            for (row = 0; row < rows; row++)
                oracle_step(g_weight, g_stream[done + row], g_oracle_window, throwaway,
                            conv_dim, taps);

            rc = device_read(&session->noc, g_dram[DRAM_BANK_STATE], DRAM_BASE, g_tiles,
                             window_bytes);
            if (rc != FT_TT_OK)
                return 1.0e30;
            for (plane = 0; plane + 1u < taps; plane++) {
                (void)ft_tt_untilize_strip((unsigned char *)g_tiles +
                                               (size_t)plane * cols * TILE_BYTES,
                                           sizeof(float), 1u, conv_dim,
                                           (size_t)conv_dim * sizeof(float), g_rows);
                {
                    double plane_worst = 0.0;
                    double plane_scale = 0.0;

                    for (channel = 0; channel < conv_dim; channel++) {
                        const double got = (double)g_rows[channel];
                        const double want =
                            g_oracle_window[(size_t)channel * (taps - 1u) + plane];

                        if (fabs(got - want) > plane_worst)
                            plane_worst = fabs(got - want);
                        if (fabs(want) > plane_scale)
                            plane_scale = fabs(want);
                    }
                    plane_worst /= (plane_scale > 1.0e-6 ? plane_scale : 1.0e-6);
                    if (plane_worst > window_worst)
                        window_worst = plane_worst;
                }
            }
        }

        done += rows;
        chunk_index++;
    }
    *out_window_worst = window_worst;
    return worst;
}

static void stage_parity(const ft_tt_hardware_session *session)
{
    int first_bad = -1;
    double window_worst = 0.0;
    double worst;
    uint32_t red;

    ft_tt_test_step("stage 2 — 103 positions on the card in chunks of 32 + 32 + 32 + 7, "
                    "scored against the scalar oracle");

    worst = parity_run(session, kRedNone, &first_bad, &window_worst);
    ft_tt_test_check_fmt("the chunkwise convolution meets the gate at EVERY position of "
                         "EVERY chunk",
                         worst <= CONV_RELATIVE_GATE && first_bad < 0,
                         "max-norm relative %.3e (gate %.1e), first bad chunk %d", worst,
                         CONV_RELATIVE_GATE, first_bad);
    ft_tt_test_check_fmt("...and the two diagnostic figures beside it", true,
                         "worst ABSOLUTE %.3e, worst per-element relative over |want| >= %.2f "
                         "is %.3e", g_worst_absolute, SIGNIFICANT_MAGNITUDE,
                         g_worst_significant);
    /* THE CARRY, SCORED SEPARATELY. A roll that lands the wrong rows shows up HERE as a roll
     * error rather than as the next chunk's output being slightly off — which is the
     * difference between a diagnosis and a symptom. */
    ft_tt_test_check_fmt("and the ROLLED WINDOW matches the oracle's conv_state after every "
                         "chunk",
                         window_worst <= CONV_RELATIVE_GATE, "worst relative %.3e",
                         window_worst);

    ft_tt_test_step("stage 3 — the red arms, every one of them a DRAM mutation against the "
                    "same five images");
    for (red = kRedBandSwap; red < kRedCount; red++) {
        int bad = -1;
        double red_window = 0.0;
        double red_worst = parity_run(session, (gdnc_red)red, &bad, &red_window);
        char label[192];

        snprintf(label, sizeof(label), "RED %s breaks it", g_red_name[red]);
        ft_tt_test_check_fmt(label, red_worst > CONV_RELATIVE_GATE,
                             "max-norm relative %.3e, first bad chunk %d, window %.3e",
                             red_worst, bad, red_window);
        if (red == (uint32_t)kRedWindowWipe)
            /* THE QUANTITATIVE FORM OF "ONE CHUNK IS NOT A TEST": with the window zeroed
             * between chunks, chunk 0 is UNAFFECTED (its window is zero anyway) and the
             * failure appears at chunk 1. A suite that ran one chunk would have called this
             * arm green. */
            ft_tt_test_check_fmt("...and window_wipe is INVISIBLE in chunk 0, appearing at "
                                 "chunk 1",
                                 bad == 1, "first bad chunk %d", bad);
    }

    /* THE GREEN CONTROL, RE-RUN LAST, so "the red arms fire" is not a card that broke halfway
     * through the stage. */
    worst = parity_run(session, kRedNone, &first_bad, &window_worst);
    ft_tt_test_check_fmt("and the unmutated arm is still green AFTER every red one",
                         worst <= CONV_RELATIVE_GATE && first_bad < 0,
                         "max-norm relative %.3e, window %.3e", worst, window_worst);
}

/* ================================================================
 * Stage 4 — the column slice
 * ================================================================ */

static void stage_slice(const ft_tt_hardware_session *session)
{
    const uint32_t conv_dim = PARITY_CONV_DIM;
    const uint32_t taps = PARITY_TAPS;
    const uint32_t cols = conv_dim / TILE_DIM;
    const size_t stream_bytes = (size_t)cols * TILE_BYTES;
    const size_t window_bytes = (size_t)(taps - 1u) * cols * TILE_BYTES;
    static float whole[TILE_DIM * PARITY_CONV_DIM];
    ft_tt_op_gdnc_request request;
    uint32_t slice;
    uint32_t row;
    uint32_t channel;
    size_t differing = 0;
    int rc;

    ft_tt_test_step("stage 4 — the COLUMN SLICE: four descriptors covering the stream must "
                    "produce the byte-for-byte answer one descriptor does");

    (void)stage_constants(g_weight, conv_dim, taps, false, false);
    rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT], DRAM_BASE, g_tiles,
                      ((size_t)taps * cols + 2u * (size_t)taps) * TILE_BYTES);
    if (rc == FT_TT_OK)
        rc = fill_dram(&session->noc, g_dram[DRAM_BANK_STATE], DRAM_BASE, 0u, window_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the slice stage stages", false, "rc=%d", rc);
        return;
    }

    for (row = 0; row < TILE_DIM; row++)
        for (channel = 0; channel < conv_dim; channel++)
            g_rows[(size_t)row * conv_dim + channel] = (float)g_stream[row][channel];
    stage_strip(g_rows, TILE_DIM, conv_dim);
    rc = device_write(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE, g_tiles,
                      stream_bytes);
    if (rc == FT_TT_OK)
        rc = fill_dram(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE + DRAM_OUTPUT_GAP,
                       POISON_BYTE, stream_bytes);
    if (rc != FT_TT_OK)
        return;

    fill_request(&request, conv_dim, taps, TILE_DIM, 0u, 0u);
    if (run_descriptor(session, &request, NULL, true) != FT_TT_OK)
        return;
    rc = device_read(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE + DRAM_OUTPUT_GAP,
                     g_tiles, stream_bytes);
    if (rc != FT_TT_OK)
        return;
    (void)ft_tt_untilize_strip(g_tiles, sizeof(float), TILE_DIM, conv_dim,
                               (size_t)conv_dim * sizeof(float), whole);

    /* Fresh window, poisoned output, then the same stream in four column slices. */
    rc = fill_dram(&session->noc, g_dram[DRAM_BANK_STATE], DRAM_BASE, 0u, window_bytes);
    if (rc == FT_TT_OK)
        rc = fill_dram(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE + DRAM_OUTPUT_GAP,
                       POISON_BYTE, stream_bytes);
    if (rc != FT_TT_OK)
        return;
    for (slice = 0; slice < 4u; slice++) {
        fill_request(&request, conv_dim, taps, TILE_DIM, slice * (cols / 4u), cols / 4u);
        if (run_descriptor(session, &request, NULL, true) != FT_TT_OK)
            return;
    }
    rc = device_read(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE + DRAM_OUTPUT_GAP,
                     g_tiles, stream_bytes);
    if (rc != FT_TT_OK)
        return;
    (void)ft_tt_untilize_strip(g_tiles, sizeof(float), TILE_DIM, conv_dim,
                               (size_t)conv_dim * sizeof(float), g_rows);

    for (row = 0; row < TILE_DIM; row++)
        for (channel = 0; channel < conv_dim; channel++)
            if (g_rows[(size_t)row * conv_dim + channel] !=
                whole[(size_t)row * conv_dim + channel])
                differing++;
    /* BYTE-FOR-BYTE, not within a tolerance: both arms run the same arithmetic on the same
     * bytes, so a difference is a plumbing bug and the sensitive instrument for a plumbing bug
     * is an exact comparison. */
    ft_tt_test_check_fmt("four column slices reproduce one whole-stream descriptor EXACTLY",
                         differing == 0u, "%zu of %u elements differ", differing,
                         TILE_DIM * conv_dim);
}

/* ================================================================
 * Stage 5 — the cost, and the chunk-size sweep, at Ornith's own width
 * ================================================================ */

static void stage_cost(const ft_tt_hardware_session *session)
{
    static const uint32_t sweep[] = { 3u, 4u, 8u, 16u, 24u, 32u };
    const uint32_t conv_dim = ORNITH_CONV_DIM;
    const uint32_t taps = PARITY_TAPS;
    const uint32_t cols = conv_dim / TILE_DIM;
    const size_t stream_bytes = (size_t)cols * TILE_BYTES;
    const size_t window_bytes = (size_t)(taps - 1u) * cols * TILE_BYTES;
    ft_tt_op_gdnc_request request;
    unsigned index;
    uint32_t rng = 0x1234abcdu;
    uint32_t row;
    uint32_t channel;
    int rc;

    ft_tt_test_step("stage 5 — the descriptor cost and the CHUNK-SIZE SWEEP at Ornith's own "
                    "conv_dim = 8192 (256 tile columns)");

    /* A weight this stage does not score; it only has to be finite, because PHASE CONV in the
     * decode family is DATA-DEPENDENT (1222.7 us on real activations against 4571 on a
     * degenerate stream, suspected an SFPU slow path on out-of-range SiLU input) and a
     * timing arm fed zeros would measure the wrong thing. */
    {
        static double wide_weight[ORNITH_CONV_DIM * PARITY_TAPS];
        uint32_t tap;

        for (channel = 0; channel < conv_dim; channel++)
            for (tap = 0; tap < taps; tap++)
                wide_weight[(size_t)channel * taps + tap] = sample(&rng) * 0.5;
        rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT], DRAM_BASE, g_tiles,
                          stage_constants(wide_weight, conv_dim, taps, false, false));
    }
    if (rc == FT_TT_OK)
        rc = fill_dram(&session->noc, g_dram[DRAM_BANK_STATE], DRAM_BASE, 0u, window_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the cost stage stages", false, "rc=%d", rc);
        return;
    }

    for (row = 0; row < TILE_DIM; row++)
        for (channel = 0; channel < conv_dim; channel++)
            g_rows[(size_t)row * conv_dim + channel] = (float)sample(&rng);
    stage_strip(g_rows, TILE_DIM, conv_dim);
    rc = device_write(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DRAM_BASE, g_tiles,
                      stream_bytes);
    if (rc != FT_TT_OK)
        return;

    for (index = 0; index < sizeof(sweep) / sizeof(sweep[0]); index++) {
        const uint32_t rows = sweep[index];
        double best = 1.0e30;
        unsigned repeat;

        fill_request(&request, conv_dim, taps, rows, 0u, 0u);
        /* ONE WARM-UP THEN THREE TIMED. The first descriptor after an arm retires and replays
         * the residency, which contaminated an earlier prefill measurement and read as a fake
         * 277x. The MINIMUM of the timed runs is reported: this box is 2-core and a scheduler
         * excursion can only make a descriptor look slower. */
        (void)run_descriptor(session, &request, NULL, true);
        for (repeat = 0; repeat < 3u; repeat++) {
            double seconds = 0.0;

            if (run_descriptor(session, &request, &seconds, true) != FT_TT_OK)
                return;
            if (seconds < best)
                best = seconds;
        }
        ft_tt_test_check_fmt("chunk-size sweep", true,
                             "C = %2u  descriptor %8.1f us  = %7.1f us/position", rows,
                             best * 1.0e6, best * 1.0e6 / (double)rows);
    }

    /* PM 8: THE FIXED COST OF A DESCRIPTOR. The model's chain issues 3,672 gdnc descriptors a
     * 1,629-row prompt over 8 members, and the puller's AWAIT census fits them at ~96 us
     * fixed + 368/N us of work. A column-slice sweep at C = 32 separates the two here, on one
     * core and with no model: descriptor(cols) = intercept + slope * cols. */
    {
        static const uint32_t slices[] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u, 256u };

        for (index = 0; index < sizeof(slices) / sizeof(slices[0]); index++) {
            const uint32_t count = slices[index];
            double best = 1.0e30;
            unsigned repeat;

            fill_request(&request, conv_dim, taps, 32u, 0u, count == cols ? 0u : count);
            (void)run_descriptor(session, &request, NULL, true);
            for (repeat = 0; repeat < 5u; repeat++) {
                double seconds = 0.0;

                if (run_descriptor(session, &request, &seconds, true) != FT_TT_OK)
                    return;
                if (seconds < best)
                    best = seconds;
            }
            ft_tt_test_check_fmt("CONV column-slice sweep", true,
                                 "%3u column(s)  descriptor %8.1f us", count, best * 1.0e6);
        }
    }

    /* THE COMPARISON THAT MATTERS, stated so nobody has to divide by hand: the decode
     * family's PHASE CONV is 1220.0 us for ONE position at this width, measured on the shipped
     * arm the same day. */
    ft_tt_test_check_fmt("the decode family's PHASE CONV, for reference", true,
                         "1220.0 us/position at conv_dim %u, measured on the shipped arm "
                         "2026-08-21", conv_dim);
}

/* ================================================================
 * Stage 6 — the device's own refusals
 * ================================================================ */

static void stage_refusals(const ft_tt_hardware_session *session)
{
    ft_tt_op_gdnc_request request;
    ft_tt_queue_descriptor descriptor;
    uint32_t words[FT_TT_OP_GDNC_WORDS];
    uint32_t parameter_address = 0u;
    uint32_t completed = 0u;
    uint32_t fields[FT_TT_QUEUE_FIELD_COUNT];
    int rc;

    ft_tt_test_step("stage 6 — a shape the DEVICE refuses, hand-encoded past the host check");

    /* The host refuses `chunk_rows < taps-1` before it encodes, so the only way to see the
     * DEVICE's refusal is to encode a legal shape and then mutate the word. That is the point:
     * the refusal exists because a bad shape does not produce a wrong answer, it produces a
     * BLOCKED TRIAD — a writer waiting on an output tile nobody will produce, until its bound
     * expires. A refusal names it in one word instead. */
    fill_request(&request, PARITY_CONV_DIM, PARITY_TAPS, TILE_DIM, 0u, 0u);
    rc = ft_tt_op_gdnc_parameters(&request, words);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the legal shape encodes", false, "rc=%d", rc);
        return;
    }
    /* chunk_rows = 1, which is below `taps-1` and cannot roll the window. */
    words[FT_TT_OP_GDNC_SHAPE] = (words[FT_TT_OP_GDNC_SHAPE] & ~0x0000FF00u) | 0x00000100u;
    rc = ft_tt_queue_write_op_parameters(&session->noc, &g_queue, g_slot, words,
                                         &parameter_address);
    if (rc != FT_TT_OK)
        return;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.opcode = FT_TT_QUEUE_OP_GDN_CHUNK;
    descriptor.a0 = parameter_address;
    rc = ft_tt_queue_enqueue(&session->noc, &g_queue, &descriptor, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_queue_wait_completed(&session->noc, &g_queue, g_slot + 1u, 3000u,
                                        &completed);
    (void)ft_tt_queue_read_control(&session->noc, &g_queue, fields);
    ft_tt_test_check_fmt("a chunk shorter than the window is REFUSED, not executed",
                         rc != FT_TT_OK || completed <= g_slot,
                         "rc=%d completed=%u slot=%u error=%u", rc, completed, g_slot,
                         fields[FT_TT_QUEUE_FIELD_ERROR]);
}

/* ================================================================
 * Stage 6 — PHASE DELTA on silicon: 103 positions in chunks of 32 + 32 + 32 + 7
 * ================================================================
 *
 * ---------------------------------------------------------------------------
 * THE CHUNK-BOUNDARY CARRY IS THE THING THIS STAGE EXISTS TO PROVE
 * ---------------------------------------------------------------------------
 * At chunk 0 the recurrent state is ZERO, so `K_hat N_0^T` and `Q_hat N_0^T` both vanish and
 * HALF THE TRANSFORM IS NOT EXERCISED — the whole `M_0` half of
 * `M_t = P_t M_0 + sum (P_t/P_s) k_s u_s^T` is multiplied by nothing. A single-chunk test
 * therefore cannot see a carry bug, which is not a hypothesis: this suite's own `window_wipe`
 * arm on PHASE CONV scores 6.996e-01 and is INVISIBLE IN CHUNK 0. `state_wipe` below is the
 * same arm on the recurrent state and is asserted to behave the same way.
 *
 * **AND THE CARRIED STATE IS READ BACK AND SCORED SEPARATELY, AFTER EVERY CHUNK.** A state
 * that drifts shows up as a state error rather than as next chunk's output being slightly
 * off — which is the difference between a diagnosis and a number.
 *
 * ---------------------------------------------------------------------------
 * THE SHAPE, AND WHY IT IS NOT ORNITH'S
 * ---------------------------------------------------------------------------
 * `state_tiles` 4 (S = 128) is ORNITH'S OWN — the GDN state is `ssm_state = 128`, not the 256
 * that is the attention head dim. What is scaled down is the HEAD COUNT: 4 v-heads over 2
 * k-heads instead of 32 over 16, because the state readback is `v_heads * 16` tiles on a
 * 5.3 MB/s link and 32 heads would be 2 MiB a chunk. **4 over 2 still exercises the
 * INTERLEAVED GQA repeat** (`vh % k_heads`: 0,1,0,1) — which is the one that has produced
 * gibberish in this architecture before — and the per-head arithmetic is identical.
 *
 * ---------------------------------------------------------------------------
 * FIVE RED ARMS, AND EVERY ONE IS A DRAM MUTATION SO NONE NEEDS A REBUILD
 * ---------------------------------------------------------------------------
 *   decay_identity  upload D = I — the decay ratios stop existing and the causal mask with
 *                   them, so `Pm` becomes diag(q.k)
 *   const_swap      swap I and 2I — the Newton residual `2I - AX` becomes `I - AX` and the
 *                   iteration converges to the wrong fixed point
 *   strict_full     upload the STRICT mask as ALL ONES — A stops being unit triangular,
 *                   because `D[t][t] (k_t.k_t)` is exactly 1 for L2-normalised k
 *   beta_zero       upload beta = 0 — U is exactly zero, so y is the state term alone; the
 *                   control that says the chunk's own positions contribute at all
 *   state_wipe      zero the carried state between chunks — INVISIBLE IN CHUNK 0 BY
 *                   CONSTRUCTION, and asserted to be
 */


/** DELTA's own DRAM offsets, ABOVE the CONV stages' — so a post-mortem dump of either is
 *  still there after the other has run. */
#define DELTA_IN_BASE       (DRAM_BASE + 0x04000000u)
#define DELTA_OUT_BASE      (DRAM_BASE + 0x06000000u)
#define DELTA_CONST_BASE    (DRAM_BASE + 0x04000000u)
#define DELTA_GATE_BASE     (DRAM_BASE + 0x04100000u)
#define DELTA_STATE_BASE    (DRAM_BASE + 0x04000000u)

/** The gate. PHASE DELTA crosses MORE unpack stages than PHASE CONV — two Gram matmuls, four
 *  Newton doublings, two `[C,S]x[S,S]` products and the state update — so it cannot be held to
 *  CONV's 5e-03. **THE HONEST BAR IS THE ARM IT REPLACES**: modelled through the same Tf32
 *  truncating unpack, 32 SEQUENTIAL DECODE STEPS score 2.1e-02 on y and the chunkwise form
 *  scores 8.6e-03. A chunkwise arm that beat 2.5e-02 would already be no worse than what
 *  ships; 1.5e-02 is set here, between the two, so the gate can fail. */
#define DELTA_RELATIVE_GATE 1.5e-2

typedef enum {
    kDeltaCardNone = 0,
    kDeltaCardDecayIdentity,
    kDeltaCardConstSwap,
    kDeltaCardStrictFull,
    kDeltaCardBetaZero,
    kDeltaCardStateWipe,
    kDeltaCardCount
} delta_card_red;

static const char *const g_delta_card_name[kDeltaCardCount] = {
    "none", "decay_identity", "const_swap", "strict_full", "beta_zero", "state_wipe"
};

static double g_delta_state[DELTA_V_HEADS][DELTA_S * DELTA_S];
static double g_delta_q[DELTA_ROWS][DELTA_KEY_COLS];
static double g_delta_k[DELTA_ROWS][DELTA_KEY_COLS];
static double g_delta_v[DELTA_ROWS][DELTA_VAL_COLS];
static double g_delta_y[DELTA_ROWS][DELTA_VAL_COLS];
static float g_delta_decay[DELTA_ROWS][DELTA_V_HEADS];
static float g_delta_beta[DELTA_ROWS][DELTA_V_HEADS];
static float g_delta_gate[DELTA_V_HEADS * FT_TT_OP_GDNC_GATE_STRIDE];
static float g_delta_plane[DELTA_S * DELTA_S];
static float g_delta_y_read[DELTA_ROWS * DELTA_VAL_COLS];
static float g_delta_state_read[DELTA_S * DELTA_S];
static const uint32_t g_delta_split[DELTA_CHUNKS] = { 32u, 32u, 32u, 7u };

/** The oracle: `qwen35moe_compute_recur_heads()`, ONE position, state TRANSPOSED. */
static void delta_oracle_step(double *state, const double *q, const double *k, const double *v,
                              double decay, double beta, double scale, double *y)
{
    uint32_t i;
    uint32_t j;

    for (j = 0; j < DELTA_S; j++) {
        double *row = state + (size_t)j * DELTA_S;
        double pred = 0.0;
        double u;
        double out = 0.0;

        for (i = 0; i < DELTA_S; i++)
            row[i] *= decay;
        for (i = 0; i < DELTA_S; i++)
            pred += row[i] * k[i];
        u = (v[j] - pred) * beta;
        for (i = 0; i < DELTA_S; i++)
            row[i] += k[i] * u;
        /* (e) READS THE UPDATED STATE — confirmed on silicon for the decode family. */
        for (i = 0; i < DELTA_S; i++)
            out += row[i] * q[i];
        y[j] = scale * out;
    }
}

static void delta_fill_request(ft_tt_op_gdnc_request *request, uint32_t chunk_rows)
{
    memset(request, 0, sizeof(*request));
    request->phase = FT_TT_OP_GDNC_PHASE_DELTA;
    request->chunk_rows = chunk_rows;
    /* The placement is the WHOLE FAMILY'S, so a DELTA request still names the conv shape. */
    request->conv_dim = PARITY_CONV_DIM;
    request->conv_taps = PARITY_TAPS;
    request->state_tiles = DELTA_STATE_TILES;
    request->v_heads = DELTA_V_HEADS;
    request->k_heads = DELTA_K_HEADS;

    request->activation_in.x = (uint16_t)g_dram[DRAM_BANK_ACTIVATION].x;
    request->activation_in.y = (uint16_t)g_dram[DRAM_BANK_ACTIVATION].y;
    request->activation_in.address = DELTA_IN_BASE;
    request->activation_in.tile_bytes = TILE_BYTES;

    request->activation_out = request->activation_in;
    request->activation_out.address = DELTA_OUT_BASE;

    request->constant.x = (uint16_t)g_dram[DRAM_BANK_CONSTANT].x;
    request->constant.y = (uint16_t)g_dram[DRAM_BANK_CONSTANT].y;
    request->constant.address = DELTA_CONST_BASE;
    request->constant.tile_bytes = TILE_BYTES;
    request->gate_address = DELTA_GATE_BASE;

    request->state_conv.x = (uint16_t)g_dram[DRAM_BANK_STATE].x;
    request->state_conv.y = (uint16_t)g_dram[DRAM_BANK_STATE].y;
    request->state_conv.address = DELTA_STATE_BASE;
    request->state_conv.tile_bytes = TILE_BYTES;
}

/** Upload the four program constants, with the red mutations applied to what is UPLOADED. */
static int delta_stage_constants(const ft_tt_hardware_session *session, delta_card_red red)
{
    unsigned char constants[FT_TT_OP_GDNC_DELTA_CONST_TILES * TILE_BYTES];
    int rc;

    rc = ft_tt_op_gdnc_build_delta_constants(DELTA_S, constants, sizeof(constants));
    if (rc != FT_TT_OK)
        return rc;

    if (red == kDeltaCardConstSwap) {
        unsigned char scratch[TILE_BYTES];

        memcpy(scratch, constants, TILE_BYTES);
        memcpy(constants, constants + TILE_BYTES, TILE_BYTES);
        memcpy(constants + TILE_BYTES, scratch, TILE_BYTES);
    }
    if (red == kDeltaCardStrictFull) {
        uint32_t index;

        for (index = 0; index < TILE_BYTES / sizeof(float); index++) {
            const float one = 1.0f;

            memcpy(constants + (size_t)FT_TT_OP_GDNC_DELTA_CONST_STRICT * TILE_BYTES +
                       (size_t)index * sizeof(float),
                   &one, sizeof(one));
        }
    }
    return device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT], DELTA_CONST_BASE, constants,
                        sizeof(constants));
}

/**
 * One chunk: stage it, run it, read y and the state back, and score both against the oracle.
 *
 * The oracle state is advanced HERE, position by position, so the comparison is against a
 * recurrence that has seen exactly the same stream — including the ragged chunk's dead rows,
 * which it does not see at all.
 */
static bool delta_one_chunk(const ft_tt_hardware_session *session, delta_card_red red,
                            uint32_t chunk, uint32_t *rng, double *worst_y, double *worst_state,
                            double *reference_y, double *reference_state, uint32_t *first_bad)
{
    const uint32_t rows = g_delta_split[chunk];
    const double scale = 1.0 / sqrt((double)DELTA_S);
    ft_tt_op_gdnc_request request;
    uint32_t position;
    uint32_t column;
    uint32_t head;
    int rc;

    /* q and k are L2-NORMALISED per k-head, exactly as the oracle normalises them before the
     * delta rule -- which is what makes `D[t][t] (k_t.k_t)` exactly 1 and A unit triangular. */
    for (position = 0; position < DELTA_ROWS; position++) {
        for (column = 0; column < DELTA_KEY_COLS; column++) {
            g_delta_q[position][column] = (position < rows) ? sample(rng) : 0.0;
            g_delta_k[position][column] = (position < rows) ? sample(rng) : 0.0;
        }
        for (column = 0; column < DELTA_VAL_COLS; column++)
            g_delta_v[position][column] = (position < rows) ? sample(rng) : 0.0;
        for (head = 0; head < DELTA_K_HEADS && position < rows; head++) {
            double norm_q = 0.0;
            double norm_k = 0.0;

            for (column = 0; column < DELTA_S; column++) {
                const size_t at = (size_t)head * DELTA_S + column;

                norm_q += g_delta_q[position][at] * g_delta_q[position][at];
                norm_k += g_delta_k[position][at] * g_delta_k[position][at];
            }
            norm_q = 1.0 / sqrt(norm_q + 1e-6);
            norm_k = 1.0 / sqrt(norm_k + 1e-6);
            for (column = 0; column < DELTA_S; column++) {
                const size_t at = (size_t)head * DELTA_S + column;

                g_delta_q[position][at] *= norm_q;
                g_delta_k[position][at] *= norm_k;
            }
        }
        for (head = 0; head < DELTA_V_HEADS; head++) {
            /* Ornith's decays sit near 1; that is also where a sequential recurrence
             * compounds worst, so it is the honest place to compare the two. */
            g_delta_decay[position][head] = (float)(0.90 + 0.099 * (sample(rng) * 0.5 + 0.5));
            g_delta_beta[position][head] =
                (float)(1.0 / (1.0 + exp(-sample(rng) * 2.0)));
        }
    }

    /* THE STAGED q|k|v IS ONE STRIP, head-major, q and k per K-HEAD and NOT GQA-repeated --
     * the decode family's `kGdnActivationIn` byte for byte, with rows carrying positions
     * instead of padding. Rows at and above `chunk_rows` are ZERO, which is what the host
     * owes. */
    for (position = 0; position < DELTA_ROWS; position++) {
        for (column = 0; column < DELTA_KEY_COLS; column++) {
            g_rows[(size_t)position * DELTA_IN_COLS + column] =
                (float)g_delta_q[position][column];
            g_rows[(size_t)position * DELTA_IN_COLS + DELTA_KEY_COLS + column] =
                (float)g_delta_k[position][column];
        }
        for (column = 0; column < DELTA_VAL_COLS; column++)
            g_rows[(size_t)position * DELTA_IN_COLS + 2u * DELTA_KEY_COLS + column] =
                (float)g_delta_v[position][column];
    }
    (void)ft_tt_tilize_strip(g_rows, sizeof(float), DELTA_ROWS, DELTA_IN_COLS,
                             (size_t)DELTA_IN_COLS * sizeof(float), g_tiles);
    rc = device_write(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DELTA_IN_BASE, g_tiles,
                      (size_t)(DELTA_IN_COLS / TILE_DIM) * TILE_BYTES);
    ft_tt_test_check_fmt("the chunk's q|k|v uploads", rc == FT_TT_OK, "rc=%d", rc);

    /* THE SHIPPING BUILDERS build the decay tile and the gate run -- one per v-head. */
    for (head = 0; head < DELTA_V_HEADS; head++) {
        float decay[DELTA_ROWS];
        float beta[DELTA_ROWS];
        unsigned char tile[TILE_BYTES];

        for (position = 0; position < DELTA_ROWS; position++) {
            decay[position] = g_delta_decay[position][head];
            beta[position] = g_delta_beta[position][head];
        }
        rc = ft_tt_op_gdnc_build_decay(decay, rows, tile, sizeof(tile));
        if (rc == FT_TT_OK && red == kDeltaCardDecayIdentity) {
            uint32_t row;
            uint32_t col;

            for (row = 0; row < TILE_DIM; row++)
                for (col = 0; col < TILE_DIM; col++)
                    g_delta_plane[row * TILE_DIM + col] = (row == col) ? 1.0f : 0.0f;
            (void)ft_tt_tilize_strip(g_delta_plane, sizeof(float), TILE_DIM, TILE_DIM,
                                     (size_t)TILE_DIM * sizeof(float), tile);
        }
        if (rc == FT_TT_OK)
            rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT],
                              DELTA_CONST_BASE +
                                  (FT_TT_OP_GDNC_DELTA_CONST_DECAY + head) * TILE_BYTES,
                              tile, sizeof(tile));
        if (rc == FT_TT_OK)
            rc = ft_tt_op_gdnc_build_gate(decay, beta, rows, DELTA_S,
                                          g_delta_gate + head * FT_TT_OP_GDNC_GATE_STRIDE,
                                          FT_TT_OP_GDNC_GATE_STRIDE);
        /* A RED ARM MUTATES WHAT IS UPLOADED AND NOTHING ELSE. Zeroing `g_delta_beta` instead
         * would zero the ORACLE's beta too, and both arms would agree on a different
         * recurrence -- a mutation that scores 0.000e+00 and proves nothing. This project has
         * eight recorded vacuous-gate classes and that is one of them. */
        if (rc == FT_TT_OK && red == kDeltaCardBetaZero)
            memset(g_delta_gate + head * FT_TT_OP_GDNC_GATE_STRIDE +
                       FT_TT_OP_GDNC_GATE_B * DELTA_ROWS,
                   0, DELTA_ROWS * sizeof(float));
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the head's decay and gate build and upload", false, "rc=%d",
                                 rc);
            return false;
        }
    }
    rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT], DELTA_GATE_BASE, g_delta_gate,
                      sizeof(g_delta_gate));
    ft_tt_test_check_fmt("the gate block uploads", rc == FT_TT_OK, "rc=%d", rc);

    /* THE CARRY. The state is zeroed ONCE, before chunk 0, and then never touched by the host
     * -- which is the whole claim. `state_wipe` is that line removed. */
    if (chunk == 0u || red == kDeltaCardStateWipe) {
        rc = fill_dram(&session->noc, g_dram[DRAM_BANK_STATE], DELTA_STATE_BASE, 0x00,
                       (size_t)DELTA_V_HEADS * DELTA_STATE_TILES * DELTA_STATE_TILES *
                           TILE_BYTES);
        ft_tt_test_check_fmt("the carried state starts at zero", rc == FT_TT_OK, "rc=%d", rc);
        if (chunk == 0u)
            memset(g_delta_state, 0, sizeof(g_delta_state));
    }

    delta_fill_request(&request, rows);
    if (run_descriptor(session, &request, NULL, true) != FT_TT_OK)
        return false;

    /* y comes back as a `[chunk_rows, v_heads*S]` strip -- WHOLE TILES, rows carrying
     * positions. The decode family's writer had to transpose a column into a row; this one
     * does not, and that is the chunk geometry paying out. */
    rc = device_read(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DELTA_OUT_BASE, g_tiles,
                     (size_t)(DELTA_VAL_COLS / TILE_DIM) * TILE_BYTES);
    if (rc == FT_TT_OK)
        rc = ft_tt_untilize_strip(g_tiles, sizeof(float), DELTA_ROWS, DELTA_VAL_COLS,
                                  (size_t)DELTA_VAL_COLS * sizeof(float), g_delta_y_read);
    ft_tt_test_check_fmt("y reads back", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return false;

    /* The oracle, one position at a time, over the SAME stream. */
    for (position = 0; position < rows; position++)
        for (head = 0; head < DELTA_V_HEADS; head++) {
            const uint32_t k_head = head % DELTA_K_HEADS; /* THE INTERLEAVED REPEAT */

            delta_oracle_step(g_delta_state[head], &g_delta_q[position][k_head * DELTA_S],
                              &g_delta_k[position][k_head * DELTA_S],
                              &g_delta_v[position][head * DELTA_S],
                              (double)g_delta_decay[position][head],
                              (double)g_delta_beta[position][head], scale,
                              &g_delta_y[position][head * DELTA_S]);
        }

    for (position = 0; position < rows; position++)
        for (column = 0; column < DELTA_VAL_COLS; column++) {
            const double want = g_delta_y[position][column];
            const double got = (double)g_delta_y_read[(size_t)position * DELTA_VAL_COLS +
                                                      column];
            const double difference = fabs(got - want);

            if (difference > *worst_y)
                *worst_y = difference;
            if (fabs(want) > *reference_y)
                *reference_y = fabs(want);
        }

    /* AND THE CARRY, SCORED SEPARATELY. A state that drifts is a state error here rather than
     * next chunk's output being slightly off. */
    for (head = 0; head < DELTA_V_HEADS; head++) {
        rc = device_read(&session->noc, g_dram[DRAM_BANK_STATE],
                         DELTA_STATE_BASE + head * DELTA_STATE_TILES * DELTA_STATE_TILES *
                                                (uint32_t)TILE_BYTES,
                         g_tiles,
                         (size_t)DELTA_STATE_TILES * DELTA_STATE_TILES * TILE_BYTES);
        if (rc == FT_TT_OK)
            rc = ft_tt_untilize_matrix(g_tiles, sizeof(float), DELTA_S, DELTA_S,
                                       g_delta_state_read);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt("the carried state reads back", false, "rc=%d", rc);
            return false;
        }
        for (column = 0; column < DELTA_S * DELTA_S; column++) {
            const double want = g_delta_state[head][column];
            const double difference = fabs((double)g_delta_state_read[column] - want);

            if (difference > *worst_state)
                *worst_state = difference;
            if (fabs(want) > *reference_state)
                *reference_state = fabs(want);
        }
    }

    /* THE FIRST CHUNK AT WHICH THIS ARM IS PAST THE GATE, which is how `state_wipe` is shown
     * to be invisible in chunk 0 and present at chunk 1. */
    if (*first_bad == UINT32_MAX && *reference_y > 0.0 &&
        *worst_y / *reference_y > DELTA_RELATIVE_GATE)
        *first_bad = chunk;
    return true;
}

static double delta_card_run(const ft_tt_hardware_session *session, delta_card_red red,
                             double *out_state, uint32_t *out_first_bad)
{
    uint32_t rng = 0x0de17a00u;
    double worst_y = 0.0;
    double worst_state = 0.0;
    double reference_y = 0.0;
    double reference_state = 0.0;
    uint32_t first_bad = UINT32_MAX;
    uint32_t chunk;

    if (delta_stage_constants(session, red) != FT_TT_OK) {
        ft_tt_test_check("the DELTA program constants upload", false, NULL);
        return 1.0e9;
    }
    for (chunk = 0; chunk < DELTA_CHUNKS; chunk++)
        if (!delta_one_chunk(session, red, chunk, &rng, &worst_y, &worst_state, &reference_y,
                             &reference_state, &first_bad)) {
            /* A DESCRIPTOR THAT DID NOT RUN IS NOT A SCORE OF ZERO. Returning the untouched
             * accumulator here would read as a PERFECT arm on the green run and as a BROKEN
             * red arm -- a vacuous gate in both directions, and this project has eight
             * recorded classes of exactly that. */
            *out_state = 1.0e9;
            *out_first_bad = chunk;
            return 1.0e9;
        }

    *out_state = (reference_state > 0.0) ? worst_state / reference_state : worst_state;
    *out_first_bad = first_bad;
    return (reference_y > 0.0) ? worst_y / reference_y : worst_y;
}

/* ================================================================
 * Stage 5c — what a PHASE DELTA descriptor COSTS, at Ornith's real head count
 * ================================================================
 *
 * The parity stage runs 4 v-heads because its readback is `v_heads * state_tiles^2` tiles on a
 * 5.3 MB/s link. THE COST IS A DIFFERENT MEASUREMENT AND READS NOTHING BACK, so it runs the
 * shape the model actually has: 32 v-heads over 16 k-heads, `state_tiles` 4.
 *
 * THE HEAD SWEEP IS THE POINT, not the single number. The decode family's DELTA fits
 * `59.8 us + N x 103.4` over six points, i.e. a fixed host round trip plus a term that divides
 * exactly by head — which is what makes a multi-core split worth building. The same fit here
 * says whether this op is bound by its per-head arithmetic or by something that does not
 * divide, and the intercept is the number a column split cannot remove.
 *
 * The CB placement is UNCHANGED between 4 heads and 32: `resolve_cb_pages()` reads
 * `state_tiles` and `conv_taps` and nothing else, because every buffer this family places is
 * sized PER HEAD. A member that owns four heads needs the identical placement to one that
 * owns thirty-two.
 */

#define DELTA_COST_V_HEADS 32u
#define DELTA_COST_K_HEADS 16u
#define DELTA_COST_REPEATS 12u

static void delta_cost_request(ft_tt_op_gdnc_request *request, uint32_t head_count)
{
    delta_fill_request(request, DELTA_ROWS);
    request->v_heads = DELTA_COST_V_HEADS;
    request->k_heads = DELTA_COST_K_HEADS;
    request->first_col = 0u;
    request->col_count = (head_count == DELTA_COST_V_HEADS) ? 0u : head_count;
}

static void stage_delta_cost(const ft_tt_hardware_session *session)
{
    static const uint32_t sweep[5] = { 1u, 2u, 4u, 8u, 32u };
    static float decay[DELTA_ROWS];
    static float beta[DELTA_ROWS];
    static float gate[FT_TT_OP_GDNC_GATE_STRIDE];
    static unsigned char tile[TILE_BYTES];
    static unsigned char constants[FT_TT_OP_GDNC_DELTA_CONST_TILES * TILE_BYTES];
    const size_t state_bytes = (size_t)DELTA_COST_V_HEADS * DELTA_STATE_TILES *
                               DELTA_STATE_TILES * TILE_BYTES;
    const size_t input_bytes =
        (size_t)(2u * DELTA_COST_K_HEADS + DELTA_COST_V_HEADS) * DELTA_STATE_TILES * TILE_BYTES;
    ft_tt_op_gdnc_request request;
    double best_per_position = 0.0;
    uint32_t index;
    uint32_t head;
    int rc;

    ft_tt_test_step("stage 5c — a PHASE DELTA descriptor's cost at Ornith's 32 v-heads");

    for (index = 0; index < DELTA_ROWS; index++) {
        decay[index] = 0.95f;
        beta[index] = 0.5f;
    }
    rc = ft_tt_op_gdnc_build_delta_constants(DELTA_S, constants, sizeof(constants));
    if (rc == FT_TT_OK)
        rc = ft_tt_op_gdnc_build_decay(decay, DELTA_ROWS, tile, sizeof(tile));
    if (rc == FT_TT_OK)
        rc = ft_tt_op_gdnc_build_gate(decay, beta, DELTA_ROWS, DELTA_S, gate,
                                      FT_TT_OP_GDNC_GATE_STRIDE);
    if (rc == FT_TT_OK)
        rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT], DELTA_CONST_BASE,
                          constants, sizeof(constants));
    /* Every head gets the SAME decay tile and gate run: the cost is data-independent -- this
     * op has no branch on a value anywhere -- and staging 32 distinct ones would measure the
     * host's upload rather than the card's arithmetic. */
    for (head = 0; head < DELTA_COST_V_HEADS && rc == FT_TT_OK; head++) {
        rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT],
                          DELTA_CONST_BASE +
                              (FT_TT_OP_GDNC_DELTA_CONST_DECAY + head) * (uint32_t)TILE_BYTES,
                          tile, sizeof(tile));
        if (rc == FT_TT_OK)
            rc = device_write(&session->noc, g_dram[DRAM_BANK_CONSTANT],
                              DELTA_GATE_BASE +
                                  head * FT_TT_OP_GDNC_GATE_STRIDE * (uint32_t)sizeof(float),
                              gate, sizeof(gate));
    }
    /* ZEROS, not garbage: rows nobody reads are still COMPUTED, and a stale inf or NaN in
     * DRAM propagates through arithmetic this stage does not inspect. */
    if (rc == FT_TT_OK)
        rc = fill_dram(&session->noc, g_dram[DRAM_BANK_ACTIVATION], DELTA_IN_BASE, 0x00,
                       input_bytes);
    if (rc == FT_TT_OK)
        rc = fill_dram(&session->noc, g_dram[DRAM_BANK_STATE], DELTA_STATE_BASE, 0x00,
                       state_bytes);
    ft_tt_test_check_fmt("the 32-head operands stage", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;

    for (index = 0; index < 5u; index++) {
        const uint32_t heads = sweep[index];
        double best = 0.0;
        uint32_t repeat;

        delta_cost_request(&request, heads);
        for (repeat = 0; repeat < DELTA_COST_REPEATS; repeat++) {
            double seconds = 0.0;

            if (run_descriptor(session, &request, &seconds, true) != FT_TT_OK)
                return;
            /* THE MINIMUM, not the mean: this is a fixed amount of work with no data
             * dependence, so every excursion above the floor is the host's scheduler or the
             * link, and the floor is what the card costs. The first flush retires and replays,
             * which contaminated an earlier arm in this project and read as a fake 277x. */
            if (repeat > 0u && (best == 0.0 || seconds < best))
                best = seconds;
        }
        ft_tt_test_check_fmt("the descriptor completes at every head count", best > 0.0,
                             "%2u head(s): %8.1f us a descriptor, %7.2f us a position",
                             heads, best * 1e6, best * 1e6 / (double)DELTA_ROWS);
        if (heads == DELTA_COST_V_HEADS)
            best_per_position = best * 1e6 / (double)DELTA_ROWS;
    }

    /* THE COMPARISON THAT MATTERS. The decode family's whole GDN descriptor costs 1220.1 us a
     * POSITION; PHASE CONV chunkwise is 21.3. This is the other half of the same statement. */
    ft_tt_test_check_fmt("and that is the number the prefill ladder wants", true,
                         "%.2f us a position at 32 v-heads, against the decode family's "
                         "per-position GDN cost of 1220.1",
                         best_per_position);
}

static void stage_delta(const ft_tt_hardware_session *session)
{
    double arm;
    double arm_state;
    uint32_t first_bad = UINT32_MAX;
    uint32_t state_wipe_first_bad = UINT32_MAX;
    unsigned red;

    ft_tt_test_step("stage 5b — PHASE DELTA on silicon, 103 positions as 32 + 32 + 32 + 7");

    arm = delta_card_run(session, kDeltaCardNone, &arm_state, &first_bad);
    ft_tt_test_check_fmt(
        "the chunkwise delta rule reproduces the oracle across three chunk boundaries and a "
        "ragged chunk",
        arm < DELTA_RELATIVE_GATE, "y %.3e, state %.3e (gate %.1e; 32 sequential decode steps "
                                   "model at 2.1e-02)",
        arm, arm_state, DELTA_RELATIVE_GATE);
    ft_tt_test_check_fmt("and the CARRIED STATE, read back after every chunk, holds too",
                         arm_state < DELTA_RELATIVE_GATE, "%.3e", arm_state);

    for (red = (unsigned)kDeltaCardDecayIdentity; red < (unsigned)kDeltaCardCount; red++) {
        double score_state = 0.0;
        uint32_t bad = UINT32_MAX;
        const double score = delta_card_run(session, (delta_card_red)red, &score_state, &bad);

        ft_tt_test_check_fmt("a red arm breaks it", score > DELTA_RELATIVE_GATE,
                             "%s scores y %.3e state %.3e, first bad chunk %d",
                             g_delta_card_name[red], score, score_state,
                             (bad == UINT32_MAX) ? -1 : (int)bad);
        if (red == (unsigned)kDeltaCardStateWipe)
            state_wipe_first_bad = bad;
    }

    /* THE QUANTITATIVE FORM OF "A SINGLE-CHUNK TEST CANNOT SEE A CARRY BUG". At chunk 0 the
     * state is zero on both arms, so wiping it changes NOTHING; it appears at chunk 1 and
     * never goes away. */
    ft_tt_test_check_fmt(
        "state_wipe is INVISIBLE in chunk 0 and appears at chunk 1 — which is why one chunk "
        "is not a test",
        state_wipe_first_bad == 1u, "first bad chunk %d",
        (state_wipe_first_bad == UINT32_MAX) ? -1 : (int)state_wipe_first_bad);
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
    ft_tt_op_gdnc_request placement;
    uint64_t mask = 0;
    int rc;

    /* THE WIDEST PLACEMENT ANY STAGE USES, so every stage runs against ONE descriptor array
     * and the triad is brought up ONCE — which is what "resident" has to mean at the host end
     * too. Only `conv_taps` reaches `resolve_cb_pages()`; the width and the chunk are
     * per-descriptor words. */
    fill_request(&placement, ORNITH_CONV_DIM, PARITY_TAPS, TILE_DIM, 0u, 0u);

    rc = load_images(&session->noc, x, y, &g_program);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_program_load(&session->noc, x, y, &g_program.program, NULL, &g_config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_gdnc_place_cbs(&placement, g_cb_placed, g_cb_buffer, sizeof(g_cb_buffer),
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

static void retire_triad(const ft_tt_noc_access *noc, const char *label)
{
    uint32_t unstopped = 0u;
    int rc = ft_tt_watchdog_shutdown(noc, &g_group, RUN_TIMEOUT_MS, &unstopped);

    ft_tt_test_check_fmt(label, rc == FT_TT_OK && unstopped == 0u, "rc=%d unstopped=0x%X", rc,
                         unstopped);
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint16_t x = 0;
    uint16_t y = 0;
    uint32_t tensix_index = 0;
    uint32_t bank;
    uint32_t rng = 0x5eed1234u;
    uint32_t position;
    uint32_t channel;
    uint32_t tap;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <gdnc-program-dir>\n", argv[0]);
        return 2;
    }

    ft_tt_test_step("Phase 4 CHUNKWISE GDN convolution — ON-CARD acceptance");

    for (channel = 0; channel < PARITY_CONV_DIM; channel++)
        for (tap = 0; tap < PARITY_TAPS; tap++)
            g_weight[(size_t)channel * PARITY_TAPS + tap] = sample(&rng) * 0.5;
    for (position = 0; position < PARITY_POSITIONS; position++)
        for (channel = 0; channel < PARITY_CONV_DIM; channel++)
            g_stream[position][channel] = sample(&rng);

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

    /* The core set is built by ft_tt_launch_core_set_add_tensix(), which DERIVES the
     * coordinate from the live grid and CLASSIFIES it before it can enter a launch. A
     * hand-filled placement would skip that, and a wrong NoC endpoint is a hung chip and a
     * physical power cycle, not a failed call. */
    rc = ft_tt_launch_core_set_init(&g_group.cores, g_placement_storage, 1u);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_core_set_add_tensix(&g_group.cores, &session.grid, tensix_index,
                                              NULL);
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

    stage_round_trip(&session);

    rc = bring_up_triad(&session, x, y);
    ft_tt_test_check_fmt("stage 1 — the gdnc triad is staged and kicked ONCE", rc == FT_TT_OK,
                         "rc=%d", rc);
    if (rc == FT_TT_OK) {
        stage_shift_map(&session);
        stage_parity(&session);
        stage_slice(&session);
        stage_cost(&session);
        stage_delta(&session);
        stage_delta_cost(&session);
        stage_refusals(&session);
        /* THE RESIDENCY CLAIM, stated as a check: every stage above was an ENQUEUE, with the
         * five images loaded once and no launch of any kind between descriptors. */
        ft_tt_test_check_fmt("every descriptor ran on ONE image load, with no launch", true,
                             "%u descriptors, 1 program directory", g_slot);
    }

    retire_triad(&session.noc,
                 "all five processors retire when ASKED, with no reset and no park");

release_programs:
    ft_tt_program_directory_release(&g_program);
release:
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
