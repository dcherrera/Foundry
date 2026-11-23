/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_gdnc_table.c
 * @brief The gdnc DELTA reader's EXACT gate front end (kernels/ftgc_table.h), off hardware.
 *
 *   stage 1  ftgc_table_build() over libm gates at extreme (a, dt): sizes, and the
 *            builder's own every-code proof.
 *   stage 2  the CARD'S OWN front end (ftgc_table_begin/head) over an emulated DRAM and L1,
 *            decay tables striped over 1, 3 and 4 banks with a beta copy in each, for 1-16
 *            members' head ranges: reads land only at the barrier (async, as the NoC), every
 *            DRAM read is 64 B aligned at both ends and inside the scratch page, and the
 *            compact header + run it builds equal, word for word, what gdnc_compact_stage()
 *            writes from host gates (header word 0 the scale, run[row] / run[32 + row], zeros
 *            elsewhere).
 *   stage 3  reds: a barrier that never lands the reads and a wrong band must DIFFER; a
 *            malformed block refuses.
 *   stage 4  DECODE (ftgc_table_rows, the gdn reader's kGdnFlagTable): rows of band 0 for
 *            1-16 members' head ranges into the decode gate layout, word for word against
 *            the host gates, other heads' words untouched; reds and refusals.
 *
 *   make -f Makefile.gdnc table && ./build-gdnc/test_gdnc_table
 *
 * @status None
 * @issues None
 * @todo None
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_op_gdnc.h"
#include "ft_tt_test_report.h"
#include "kernels/ftgc_table.h"

#define CODES FTGT_CODES
#define HEADS 48u
#define LAYERS 3u
#define BANDS 10u
#define STRIP_TILES 2u

/* ----- the gates, spelled as the host spells them (qwen35moe_ssm.c) ----- */

static float bits_float(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float softplus(float x)
{
    return x > 20.0f ? x : log1pf(expf(x));
}

static uint32_t decay_gate(uint32_t code, float a, float dt)
{
    return float_bits(expf(a * softplus(bits_float(code << 16) + dt)));
}

static uint32_t beta_gate(uint32_t code)
{
    return float_bits(1.0f / (1.0f + expf(-bits_float(code << 16))));
}

/* ----- emulated DRAM (five endpoints) and L1 ----- */

#define ENDPOINTS 5u
#define DRAM_BYTES (24u << 20)
#define L1_BYTES (4u << 20)
#define SCRATCH 4096u /* the scratch page's L1 address */
#define QUEUE 512u

typedef struct {
    uint32_t xy, address, l1, bytes;
} pending_read;

typedef struct {
    uint8_t *dram[ENDPOINTS];
    uint8_t *l1;
    uint32_t scratch_bytes; /* the placed CB_SCRATCH */
    pending_read queue[QUEUE];
    uint32_t queued;
    uint32_t reads, misaligned, outside, overflow;
    bool land;           /* red: false = the barrier never lands a read */
} emulator;

static uint32_t endpoint_xy(uint32_t endpoint)
{
    return (endpoint + 1u) | ((endpoint + 17u) << 16);
}

static int endpoint_of(uint32_t xy)
{
    for (uint32_t e = 0; e < ENDPOINTS; e++)
        if (endpoint_xy(e) == xy)
            return (int)e;
    return -1;
}

static void emu_read(void *user, uint32_t xy, uint32_t address, uint32_t l1, uint32_t bytes)
{
    emulator *emu = (emulator *)user;

    emu->reads++;
    if ((address % 64u) != 0u || (l1 % 64u) != 0u || (bytes % 64u) != 0u)
        emu->misaligned++;
    if (endpoint_of(xy) < 0 || (uint64_t)address + bytes > DRAM_BYTES || l1 < SCRATCH ||
        (uint64_t)l1 + bytes > (uint64_t)SCRATCH + emu->scratch_bytes) {
        emu->outside++;
        return;
    }
    if (emu->queued == QUEUE) {
        emu->overflow++;
        return;
    }
    emu->queue[emu->queued].xy = xy;
    emu->queue[emu->queued].address = address;
    emu->queue[emu->queued].l1 = l1;
    emu->queue[emu->queued].bytes = bytes;
    emu->queued++;
}

static void emu_barrier(void *user)
{
    emulator *emu = (emulator *)user;

    for (uint32_t i = 0; i < emu->queued && emu->land; i++) {
        const pending_read *r = &emu->queue[i];

        memcpy(emu->l1 + r->l1, emu->dram[endpoint_of(r->xy)] + r->address, r->bytes);
    }
    emu->queued = 0u;
}

/* ----- the model the tables come from ----- */

typedef struct {
    float a[LAYERS][HEADS], dt[LAYERS][HEADS];
    uint32_t *decay_values[LAYERS][HEADS]; /* [code] */
    uint32_t beta_values[CODES];
    uint32_t decay_slot[LAYERS], beta_bytes;
    uint32_t alpha_base, beta_strip_base, block_base;
    uint16_t alpha[BANDS][STRIP_TILES * 1024u], beta[BANDS][STRIP_TILES * 1024u];
} model;

static uint32_t rng_state = 0x2545F491u;

static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/** A non-NaN bf16 code: mostly the realistic range, some specials. */
static uint16_t random_code(void)
{
    static const uint16_t special[] = {0x0000u, 0x8000u, 0x0001u, 0x807Fu, 0x7F80u,
                                       0xFF80u, 0x7F7Fu, 0xFF7Fu, 0x3F80u, 0xC2C8u};
    const uint32_t pick = rng() % 16u;

    if (pick == 0u)
        return special[rng() % (sizeof(special) / sizeof(special[0]))];
    if (pick < 4u) {
        uint32_t code;

        do
            code = rng() & 0xFFFFu;
        while ((code & 0x7FFFu) > FTGT_LAST_MAGNITUDE);
        return (uint16_t)code;
    }
    /* |x| in [2^-8, 2^6): the projections' own range */
    return (uint16_t)(((rng() & 1u) << 15) | ((119u + rng() % 14u) << 7) | (rng() & 0x7Fu));
}

static uint32_t align64(uint32_t bytes)
{
    return (bytes + 63u) & ~63u;
}

static uint32_t tile_element(uint32_t row, uint32_t head)
{
    const uint32_t column = head % 32u;

    return (head / 32u) * 1024u + ((row / 16u) * 2u + column / 16u) * 256u + (row % 16u) * 16u +
           (column % 16u);
}

/* ----- stage 1 ----- */

static void stage_build(void)
{
    static const float as[] = {-1e-3f, -0.05f, -1.0f, -30.0f};
    static const float dts[] = {-7.0f, -1.5f, 0.0f, 3.0f};
    static uint32_t values[CODES], table[CODES + 16u];
    size_t worst = 0u;

    ft_tt_test_step("stage 1 — the builder over libm gates, every non-NaN code proved");
    for (uint32_t i = 0; i < 4u; i++)
        for (uint32_t j = 0; j < 4u; j++) {
            size_t words, sized;

            for (uint32_t code = 0; code < CODES; code++)
                values[code] = decay_gate(code, as[i], dts[j]);
            sized = ftgc_table_build(values, NULL, 0u);
            words = ftgc_table_build(values, table, CODES + 16u);
            if (words > worst)
                worst = words;
            ft_tt_test_check_fmt("decay table builds and proves", words != 0u && words == sized,
                                 "a %g dt %g: %zu words (+%u/-%u window)", (double)as[i],
                                 (double)dts[j], words, table[kFtgtHiPos] - table[kFtgtLoPos],
                                 table[kFtgtHiNeg] - table[kFtgtLoNeg]);
        }
    for (uint32_t code = 0; code < CODES; code++)
        values[code] = beta_gate(code);
    {
        const size_t words = ftgc_table_build(values, table, CODES + 16u);

        ft_tt_test_check_fmt("beta table builds and proves", words != 0u, "%zu words", words);
    }
    ft_tt_test_check_fmt("a table is a small fraction of 65,536 words", worst < 16384u,
                         "worst %zu words", worst);
    {
        /* A constant function is an empty window on both sides. */
        for (uint32_t code = 0; code < CODES; code++)
            values[code] = 0x3F000000u;
        ft_tt_test_check_fmt("a constant table is its header",
                             ftgc_table_build(values, table, 16u) == 16u,
                             "16 words");
        ft_tt_test_check("too small an output refuses",
                         ftgc_table_build(values, table, 15u) == 0u, "0");
    }
}

/* ----- stage 2 / 3 ----- */

/** Layer l's stripe count: 3, 4, 1 (every count the reader must take, uneven included). */
static uint32_t layer_stripes(uint32_t l)
{
    static const uint32_t stripes[LAYERS] = {3u, 4u, 1u};

    return stripes[l];
}

static void model_build(model *m, emulator *emu)
{
    static uint32_t scratch_table[CODES + 16u];
    uint32_t cursor[ENDPOINTS] = {0u, 0u, 0u, 0u, 0u};
    uint32_t beta_at[ENDPOINTS];

    for (uint32_t l = 0; l < LAYERS; l++)
        for (uint32_t h = 0; h < HEADS; h++) {
            m->a[l][h] = -expf((float)(rng() % 1000u) / 250.0f - 2.0f); /* -[0.14, 7.4] */
            m->dt[l][h] = (float)(rng() % 1000u) / 125.0f - 6.0f;        /* [-6, 2) */
        }
    for (uint32_t code = 0; code < CODES; code++)
        m->beta_values[code] = beta_gate(code);

    /* endpoint 0: strips + table blocks; 1..4: decay stripes, each with a beta copy. */
    m->alpha_base = 0u;
    m->beta_strip_base = BANDS * STRIP_TILES * 2048u;
    m->block_base = 2u * BANDS * STRIP_TILES * 2048u;
    for (uint32_t band = 0; band < BANDS; band++)
        for (uint32_t e = 0; e < STRIP_TILES * 1024u; e++) {
            m->alpha[band][e] = random_code();
            m->beta[band][e] = random_code();
        }
    memcpy(emu->dram[0] + m->alpha_base, m->alpha, sizeof(m->alpha));
    memcpy(emu->dram[0] + m->beta_strip_base, m->beta, sizeof(m->beta));

    {
        const size_t words = ftgc_table_build(m->beta_values, scratch_table, CODES + 16u);

        m->beta_bytes = align64((uint32_t)words * 4u);
        for (uint32_t e = 1; e < ENDPOINTS; e++) {
            beta_at[e] = 4096u * e; /* a different address in every bank */
            memcpy(emu->dram[e] + beta_at[e], scratch_table, words * 4u);
            cursor[e] = beta_at[e] + m->beta_bytes + 4096u;
        }
    }
    for (uint32_t l = 0; l < LAYERS; l++) {
        const uint32_t stripes = layer_stripes(l);
        uint32_t block[kFtgtBlockWords] = {0};
        size_t widest = 0u;

        for (uint32_t h = 0; h < HEADS; h++) {
            uint32_t *values = (uint32_t *)malloc(CODES * sizeof(uint32_t));
            size_t words;

            for (uint32_t code = 0; code < CODES; code++)
                values[code] = decay_gate(code, m->a[l][h], m->dt[l][h]);
            m->decay_values[l][h] = values;
            words = ftgc_table_build(values, NULL, 0u);
            if (words > widest)
                widest = words;
        }
        m->decay_slot[l] = align64((uint32_t)widest * 4u);
        block[kFtgtBlockScale] = 0x3DB504F3u;
        block[kFtgtBlockAlphaXy] = endpoint_xy(0u);
        block[kFtgtBlockAlphaBase] = m->alpha_base;
        block[kFtgtBlockBetaXy] = endpoint_xy(0u);
        block[kFtgtBlockBetaBase] = m->beta_strip_base;
        block[kFtgtBlockBandBytes] = STRIP_TILES * 2048u;
        block[kFtgtBlockStripes] = stripes;
        block[kFtgtBlockDecaySlot] = m->decay_slot[l];
        block[kFtgtBlockBetaBytes] = m->beta_bytes;
        for (uint32_t s = 0; s < stripes; s++) {
            const uint32_t e = 1u + (s + l) % (ENDPOINTS - 1u);
            const uint32_t w = kFtgtBlockStripe + 4u * s;

            block[w + kFtgtStripeDecayXy] = endpoint_xy(e);
            block[w + kFtgtStripeDecayBase] = cursor[e];
            block[w + kFtgtStripeBetaXy] = endpoint_xy(e);
            block[w + kFtgtStripeBetaBase] = beta_at[e];
            for (uint32_t h = s; h < HEADS; h += stripes) {
                const size_t words =
                    ftgc_table_build(m->decay_values[l][h], scratch_table, CODES + 16u);

                memcpy(emu->dram[e] + cursor[e] + (h / stripes) * m->decay_slot[l],
                       scratch_table, words * 4u);
            }
            cursor[e] += align64(((HEADS + stripes - 1u) / stripes) * m->decay_slot[l]);
        }
        memcpy(emu->dram[0] + m->block_base + l * kFtgtBlockStride, block, sizeof(block));
    }
}

/** What gdnc_compact_stage() writes for this head from host gates. */
static void expected_run(const model *m, uint32_t layer, uint32_t head, uint32_t band,
                         uint32_t first_row, uint32_t chunk_rows, uint32_t out[80])
{
    memset(out, 0, 80u * sizeof(uint32_t));
    out[0] = 0x3DB504F3u;
    for (uint32_t row = 0; row < chunk_rows; row++) {
        const uint32_t at = tile_element(first_row + row, head);

        out[16u + row] = m->decay_values[layer][head][m->alpha[band][at]];
        out[16u + 32u + row] = m->beta_values[m->beta[band][at]];
    }
}

/** One member's descriptor (heads [first, first + count)) through the card's front end, its
 *  heads in order or shuffled. @return heads that differed, or HEADS + 1 when begin refused. */
static uint32_t run_descriptor(const model *m, emulator *emu, ftgc_table_state *state,
                               uint32_t layer, uint32_t band, uint32_t wire_band,
                               uint32_t first_row, uint32_t chunk_rows, uint32_t first,
                               uint32_t count, bool shuffled)
{
    const ftgc_table_io io = {emu, (uintptr_t)emu->l1, emu_read, emu_barrier};
    uint32_t differ = 0u, order[HEADS];

    for (uint32_t i = 0; i < count; i++)
        order[i] = first + i;
    if (shuffled)
        for (uint32_t i = count - 1u; i > 0u; i--) {
            const uint32_t k = rng() % (i + 1u), t = order[i];

            order[i] = order[k];
            order[k] = t;
        }
    memset(emu->l1, 0xA5, SCRATCH + FTGT_SCRATCH_BYTES); /* stale L1: every word must be written */
    if (!ftgc_table_begin(&io, state, SCRATCH, endpoint_xy(0u),
                          m->block_base + layer * kFtgtBlockStride, first))
        return HEADS + 1u;
    for (uint32_t i = 0; i < count; i++) {
        const uint32_t head = order[i];
        uint32_t want[80];

        ftgc_table_head(&io, state, SCRATCH, head, wire_band, first_row, chunk_rows);
        expected_run(m, layer, head, band, first_row, chunk_rows, want);
        if (memcmp(emu->l1 + SCRATCH + FTGT_SCRATCH_COMPACT, want, sizeof(want)) != 0)
            differ++;
    }
    return differ;
}

static void stage_front_end(model *m, emulator *emu)
{
    static const uint32_t windows[][2] = {{0u, 16u}, {16u, 16u}, {0u, 32u}, {0u, 7u},
                                          {16u, 5u}, {0u, 1u}, {16u, 1u}};
    static const uint32_t members[] = {1u, 4u, 8u, 16u};
    ftgc_table_state state;
    uint32_t total = 0u, differ = 0u;

    memset(&state, 0, sizeof(state));
    emu->land = true;
    emu->scratch_bytes = FTGT_SCRATCH_BYTES;
    ft_tt_test_step("stage 2 — the card's front end over emulated DRAM, striped, word for word");
    for (uint32_t w = 0; w < sizeof(windows) / sizeof(windows[0]); w++)
        for (uint32_t l = 0; l < LAYERS; l++)
            for (uint32_t k = 0; k < sizeof(members) / sizeof(members[0]); k++) {
                const uint32_t band = rng() % BANDS, per = HEADS / members[k];

                for (uint32_t first = 0; first < HEADS; first += per) {
                    differ += run_descriptor(m, emu, &state, l, band, band, windows[w][0],
                                             windows[w][1], first, per, (w + k) % 2u == 1u);
                    total += per;
                }
            }
    ft_tt_test_check_fmt("every head's compact run is the host's", differ == 0u,
                         "%u of %u heads differ", differ, total);
    ft_tt_test_check_fmt("every DRAM read is 64 B aligned at both ends", emu->misaligned == 0u,
                         "%u of %u reads misaligned", emu->misaligned, emu->reads);
    ft_tt_test_check_fmt("every read inside the scratch page and a DRAM endpoint",
                         emu->outside == 0u && emu->overflow == 0u, "%u outside, %u overflow",
                         emu->outside, emu->overflow);

    ft_tt_test_step("stage 3 — reds (must DIFFER / refuse)");
    emu->land = false;
    differ = run_descriptor(m, emu, &state, 0u, 3u, 3u, 0u, 16u, 0u, HEADS, false);
    ft_tt_test_check_fmt("red: reads never landed (refused or wrong)", differ != 0u,
                         "%u (%u = refused)", differ, HEADS + 1u);
    emu->land = true;
    differ = run_descriptor(m, emu, &state, 1u, 4u, 5u, 16u, 16u, 0u, HEADS, false);
    ft_tt_test_check_fmt("red: the wrong band on the wire", differ != 0u && differ <= HEADS,
                         "%u of %u heads differ", differ, HEADS);
    {
        uint32_t saved[kFtgtBlockWords];
        uint8_t *block = emu->dram[0] + m->block_base;

        memcpy(saved, block, sizeof(saved));
        ((uint32_t *)block)[kFtgtBlockStripes] = kFtgtMaxStripes + 1u;
        ft_tt_test_check("a block with 9 stripes refuses",
                         run_descriptor(m, emu, &state, 0u, 0u, 0u, 0u, 16u, 0u, 6u, false) ==
                             HEADS + 1u, "begin returns 0");
        memcpy(block, saved, sizeof(saved));
        ((uint32_t *)block)[kFtgtBlockDecaySlot] = 100u;
        ft_tt_test_check("a slot off the 64 B grain refuses",
                         run_descriptor(m, emu, &state, 0u, 0u, 0u, 0u, 16u, 0u, 6u, false) ==
                             HEADS + 1u, "begin returns 0");
        memcpy(block, saved, sizeof(saved));
    }
}

/* ----- stage 4: decode ----- */

#define GATE_STRIDE ((2u * HEADS * 4u + 63u) & ~63u)

/** One decode member's gate block through ftgc_table_rows. @return words wrong (own words
 *  that differ from the host gates plus other words that were written), or ~0u refused. */
static uint32_t run_rows(const model *m, emulator *emu, uint32_t layer, uint32_t first,
                         uint32_t count, uint32_t row_first, uint32_t row_count)
{
    const ftgc_table_io io = {emu, (uintptr_t)emu->l1, emu_read, emu_barrier};
    const uint32_t gate = SCRATCH + FTGT_ROWS_BYTES;
    const uint32_t *out = (const uint32_t *)(emu->l1 + gate);
    uint32_t wrong = 0u;

    memset(emu->l1, 0xA5, SCRATCH + FTGT_ROWS_BYTES + row_count * GATE_STRIDE);
    if (!ftgc_table_rows(&io, SCRATCH, endpoint_xy(0u), m->block_base + layer * kFtgtBlockStride,
                         first, count, row_first, row_count, gate, GATE_STRIDE, HEADS))
        return ~0u;
    for (uint32_t r = 0; r < row_count; r++)
        for (uint32_t h = 0; h < HEADS; h++) {
            const uint32_t at = tile_element(row_first + r, h);
            const bool own = h >= first && h < first + count;
            const uint32_t decay = out[r * (GATE_STRIDE / 4u) + h];
            const uint32_t beta = out[r * (GATE_STRIDE / 4u) + HEADS + h];

            if (own)
                wrong += (decay != m->decay_values[layer][h][m->alpha[0][at]]) +
                         (beta != m->beta_values[m->beta[0][at]]);
            else
                wrong += (decay != 0xA5A5A5A5u) + (beta != 0xA5A5A5A5u);
        }
    return wrong;
}

static void stage_decode(model *m, emulator *emu)
{
    static const uint32_t rows[][2] = {{0u, 1u}, {5u, 1u}, {0u, 4u}, {14u, 3u}, {16u, 10u},
                                       {30u, 2u}, {31u, 1u}};
    static const uint32_t members[] = {1u, 2u, 4u, 8u, 16u};
    uint32_t wrong = 0u, runs = 0u, refused = 0u;

    ft_tt_test_step("stage 4 — DECODE: ftgc_table_rows into the gate layout, word for word");
    emu->land = true;
    emu->scratch_bytes = FTGT_ROWS_BYTES;
    emu->misaligned = emu->outside = emu->overflow = 0u;
    for (uint32_t w = 0; w < sizeof(rows) / sizeof(rows[0]); w++)
        for (uint32_t l = 0; l < LAYERS; l++)
            for (uint32_t k = 0; k < sizeof(members) / sizeof(members[0]); k++) {
                const uint32_t per = HEADS / members[k];

                for (uint32_t first = 0; first < HEADS; first += per) {
                    const uint32_t bad = run_rows(m, emu, l, first, per, rows[w][0], rows[w][1]);

                    refused += bad == ~0u;
                    wrong += bad == ~0u ? 0u : bad;
                    runs++;
                }
            }
    ft_tt_test_check_fmt("every member's gate words are the host's, no other word written",
                         wrong == 0u && refused == 0u, "%u words wrong, %u of %u refused",
                         wrong, refused, runs);
    ft_tt_test_check_fmt("every DRAM read 64 B aligned, inside the scratch and an endpoint",
                         emu->misaligned == 0u && emu->outside == 0u && emu->overflow == 0u,
                         "%u misaligned, %u outside, %u overflow", emu->misaligned,
                         emu->outside, emu->overflow);
    emu->land = false;
    wrong = run_rows(m, emu, 1u, 0u, 6u, 0u, 1u);
    ft_tt_test_check_fmt("red: reads never landed", wrong != 0u, "%u", wrong);
    emu->land = true;
    {
        uint32_t saved[kFtgtBlockWords];
        uint8_t *block = emu->dram[0] + m->block_base;

        memcpy(saved, block, sizeof(saved));
        ((uint32_t *)block)[kFtgtBlockStripes] = 0u;
        ft_tt_test_check("decode: a block with 0 stripes refuses",
                         run_rows(m, emu, 0u, 0u, 6u, 0u, 1u) == ~0u, NULL);
        memcpy(block, saved, sizeof(saved));
    }
    ft_tt_test_check("decode: rows past the tile refuse", run_rows(m, emu, 0u, 0u, 6u, 31u, 2u) == ~0u,
                     NULL);
    ft_tt_test_check("decode: more heads than the header area refuse",
                     run_rows(m, emu, 0u, 0u, FTGT_ROWS_MAX_HEADS + 1u, 0u, 1u) == ~0u, NULL);
}

int main(void)
{
    static model m;
    static emulator emu;

    ft_tt_test_step("test_gdnc_table — exact card gates through host-built tables");
    stage_build();
    for (uint32_t e = 0; e < ENDPOINTS; e++) {
        emu.dram[e] = (uint8_t *)calloc(DRAM_BYTES, 1u);
        if (!emu.dram[e])
            return 2;
    }
    emu.l1 = (uint8_t *)malloc(L1_BYTES);
    if (!emu.l1)
        return 2;
    model_build(&m, &emu);
    stage_front_end(&m, &emu);
    stage_decode(&m, &emu);
    for (uint32_t l = 0; l < LAYERS; l++)
        for (uint32_t h = 0; h < HEADS; h++)
            free(m.decay_values[l][h]);
    for (uint32_t e = 0; e < ENDPOINTS; e++)
        free(emu.dram[e]);
    free(emu.l1);
    return ft_tt_test_finish();
}
