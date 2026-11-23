/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_q36_expander_bench.c
 * @brief qwen36 plan Phase 1.4 / 1.4b on the card: the price of running BFP6_B experts.
 *
 * STAGE expand (routes 1 and 3 of spec §4.1b): the scalar BFP6_B -> BFP8_B expander
 *   (kernels/bfp6_expand_tile.h) on the data movers and on the three TRISCs.
 *   - GATE: every expanded byte IDENTICAL to ft_bfp6_expand_tiles(), both layouts, DRAM-fed on
 *     BRISC and NCRISC, and L1 -> L1 on each TRISC. RED: the same comparator against a datum<<1
 *     host expansion must DIFFER (a comparator that cannot fail proves nothing).
 *   - PRICE: tiles/us per processor on its own wall clock — BRISC alone L1 -> L1 and DRAM-fed,
 *     BRISC + NCRISC together DRAM-fed, and all five RISC-Vs of a core together.
 *
 * STAGE split (route 1.4b): a BFP4_B high tile + BFP2_B residual tile, two native matmul
 *   passes into one Dst (kernels/bfp_split_matmul_compute.c), at the decode shape Mt = 1.
 *   - GATE: PCC against an fp64 reference on the values the card holds (high + low, bf16
 *     activations), and against its own passes; a run without the low pass must score WORSE
 *     against the high+low reference (the red). Phases {0,2} must be BYTE-IDENTICAL to HiFi4
 *     for BFP4/BFP2 and for an expanded BFP6_B tile; for a 7-bit BFP8_B weight it must DIFFER.
 *   - PRICE: ns per weight tile for BFP8_B (the control: Ornith's format), high only, low
 *     only, both, at HiFi4 and at phases {0,2}.
 *
 * STAGE sfpu (route 2's floor): kernels/bfp6_sfpu_floor_compute.c — unpack the planes
 *   layout's high plane (a valid BFP4_B tile), copy to Dst, pack BFP8_B: everything route 2
 *   does EXCEPT the decode, so a lower bound on its per-tile cost. GATE: the packed values
 *   equal the plane's.
 *
 * Environment: FT_TT_TENSIX_INDEX (default 75), FT_TT_FIRMWARE_DIR (one of the program
 * directories), FT_TT_PROGRAM_DIR (holds q36_expand, q36_split_hf4, q36_split_hf2,
 * q36_bfp8_hf4, q36_bfp8_hf2), Q36_BENCH_STAGES (default "expand,split"), Q36_BENCH_REPEATS.
 * Built by Makefile.q36bench; programs by run_q36_phase1_bench.sh.
 *
 * @status New 2026-10-28 (qwen36 Phase 1).
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../convert/ft_bfp.h"
#include "../convert/ft_bfp_split.h"
#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"
#include "ft_tt_test_report.h"

#define TILE 32u
#define TILE_VALUES 1024u
#define BF16_TILE 2048u
#define BFP6_TILE 832u
#define BFP8_TILE 1088u
#define TIMEOUT_MS 20000u
#define STATUS_MAGIC 0x51364558u
#define DRAM_IN_BANK 2u
#define DRAM_OUT_BANK 4u
#define DRAM_BASE 0x00200000u
#define EXPAND_TILES 96u
#define PIPE_BLOCK 4u

typedef struct bench_context {
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    uint16_t x, y;
    ft_tt_noc_coord dram[8];
    double cycles_per_us;
    uint32_t repeats;
} bench_context;

/* ================================================================ host helpers */

static uint64_t random_state = 0x9E3779B97F4A7C15ull;

static float random_value(void)
{
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (float)((double)(random_state * 0x2545F4914F6CDD1Dull >> 11) / (double)(1ull << 53) * 2.0 - 1.0);
}

/** Expert-like weight: sum of four uniforms, scaled to an expert matrix's spread. */
static float random_weight(void)
{
    return (random_value() + random_value() + random_value() + random_value()) * 0.025f;
}

static double now_us(void)
{
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec * 1e-3;
}

static int noc_write(const bench_context *c, ft_tt_noc_coord at, uint32_t address, const void *src, size_t bytes)
{
    for (size_t off = 0; off < bytes; off += 0x10000u) {
        size_t chunk = bytes - off < 0x10000u ? bytes - off : 0x10000u;
        int rc = c->session.noc.write(c->session.noc.user, (uint16_t)at.x, (uint16_t)at.y, address + off,
                                      (const unsigned char *)src + off, chunk);

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

static int noc_read(const bench_context *c, ft_tt_noc_coord at, uint32_t address, void *dst, size_t bytes)
{
    for (size_t off = 0; off < bytes; off += 0x10000u) {
        size_t chunk = bytes - off < 0x10000u ? bytes - off : 0x10000u;
        int rc = c->session.noc.read(c->session.noc.user, (uint16_t)at.x, (uint16_t)at.y, address + off,
                                     (unsigned char *)dst + off, chunk);

        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

static ft_tt_noc_coord core_coord(const bench_context *c)
{
    ft_tt_noc_coord at;

    memset(&at, 0, sizeof(at));
    at.x = c->x;
    at.y = c->y;
    return at;
}

static int load_program(const char *name, ft_tt_program_directory *out)
{
    char path[512];
    const char *root = getenv("FT_TT_PROGRAM_DIR");

    snprintf(path, sizeof(path), "%s/%s", root && *root ? root : ".", name);
    return ft_tt_program_directory_load(path, 0x1Fu, out);
}

/* ================================================================ stage expand */

#define LAYOUT_NAME(l) ((l) == 2u ? "fold" : (l) == 1u ? "planes" : "stream")

/** Per-processor L1 plan of the expander program: shared input tiles, each processor's out
 *  slots, one status block (64 B a processor). Placed as CBs only to get arena addresses. */
typedef struct expand_plan {
    ft_tt_launch_cb_entry entries[7];
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    uint64_t cb_mask;
    uint32_t in, out[5], status;
} expand_plan;

#define OUT_SLOTS 8u

static int expand_plan_place(expand_plan *plan)
{
    ft_tt_op_cb_request requests[7];
    int rc;

    requests[0].index = 0;
    requests[0].page_bytes = BFP6_TILE;
    requests[0].page_count = EXPAND_TILES;
    for (unsigned p = 1; p < 6u; p++) {
        requests[p].index = p;
        requests[p].page_bytes = BFP8_TILE;
        requests[p].page_count = OUT_SLOTS;
    }
    requests[6].index = 6;
    requests[6].page_bytes = 320u;
    requests[6].page_count = 1u;
    rc = ft_tt_op_cb_place(requests, 7u, plan->entries, plan->cb_buffer, sizeof(plan->cb_buffer), &plan->cb_mask);
    plan->in = plan->entries[0].address;
    for (unsigned p = 0; p < 5u; p++)
        plan->out[p] = plan->entries[1u + p].address;
    plan->status = plan->entries[6].address;
    return rc;
}

/** Arguments of one expander launch, per processor: tiles 0 means "idle". */
typedef struct expand_arm {
    uint32_t mode, layout, repeats;
    uint32_t tiles[5];     /* DM0, DM1, TRISC0, TRISC1, TRISC2 */
    uint32_t in_offset[5]; /* tile offset into the shared input / DRAM source */
} expand_arm;

static int expand_launch(bench_context *c, const ft_tt_kernel_program *program, expand_plan *plan,
                         const expand_arm *arm, uint32_t status_out[5][4])
{
    static const unsigned processors[5] = {FT_TT_PROC_DM0, FT_TT_PROC_DM1, FT_TT_PROC_MATH0, FT_TT_PROC_MATH1,
                                           FT_TT_PROC_MATH2};
    uint32_t mover[2][15], trisc[3][7], poison[80];
    ft_tt_launch_config config;
    ft_tt_op_args args[5];
    int rc;

    rc = ft_tt_op_program_load(&c->session.noc, c->x, c->y, program, NULL, &config);
    if (rc != FT_TT_OK)
        return rc;
    config.local_cb_mask = plan->cb_mask;
    config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    config.brisc_noc_id = 0u;
    for (unsigned i = 0; i < 80u; i++)
        poison[i] = 0x7B1B1B1Bu;
    rc = noc_write(c, core_coord(c), plan->status, poison, sizeof(poison));
    for (unsigned p = 0; p < 2u; p++) {
        uint32_t *a = mover[p];

        a[0] = arm->mode;
        a[1] = arm->layout;
        a[2] = arm->tiles[p];
        a[3] = arm->repeats;
        a[4] = plan->in + (arm->mode == 0u ? arm->in_offset[p] * BFP6_TILE : (p == 0 ? 0u : 2u * PIPE_BLOCK * BFP6_TILE));
        a[5] = plan->out[p];
        a[6] = OUT_SLOTS;
        a[7] = c->dram[DRAM_IN_BANK].x;
        a[8] = c->dram[DRAM_IN_BANK].y;
        a[9] = DRAM_BASE + arm->in_offset[p] * BFP6_TILE;
        a[10] = c->dram[DRAM_OUT_BANK].x;
        a[11] = c->dram[DRAM_OUT_BANK].y;
        a[12] = DRAM_BASE + arm->in_offset[p] * BFP8_TILE;
        a[13] = plan->status + p * 64u;
        a[14] = PIPE_BLOCK;
        args[p].processor = processors[p];
        args[p].values = a;
        args[p].count = 15u;
    }
    for (unsigned t = 0; t < 3u; t++) {
        uint32_t *a = trisc[t];

        a[0] = arm->layout;
        a[1] = arm->tiles[2u + t];
        a[2] = arm->repeats;
        a[3] = plan->in + arm->in_offset[2u + t] * BFP6_TILE;
        a[4] = plan->out[2u + t];
        a[5] = OUT_SLOTS;
        a[6] = plan->status + (2u + t) * 64u;
        args[2u + t].processor = processors[2u + t];
        args[2u + t].values = a;
        args[2u + t].count = 7u;
    }
    if (rc == FT_TT_OK)
        rc = ft_tt_op_launch(&c->session.noc, c->x, c->y, &config, plan->cb_buffer, sizeof(plan->cb_buffer), args,
                             5u, TIMEOUT_MS, NULL);
    for (unsigned p = 0; p < 5u && rc == FT_TT_OK; p++)
        rc = noc_read(c, core_coord(c), plan->status + p * 64u, status_out[p], 16u);
    return rc;
}

/** Tiles/us of one processor's status block, or 0 if it never ran or did nothing. */
static double status_rate(const bench_context *c, const uint32_t status[4])
{
    uint32_t cycles = status[2] - status[1];

    if (status[0] != STATUS_MAGIC || status[3] == 0u || cycles == 0u)
        return 0.0;
    return (double)status[3] / ((double)cycles / c->cycles_per_us);
}

/** Byte compare of `tiles` expanded tiles against the host expansion; returns differing tiles. */
static uint32_t tiles_differing(const unsigned char *card, const unsigned char *host, uint32_t tiles)
{
    uint32_t differ = 0;

    for (uint32_t t = 0; t < tiles; t++)
        differ += memcmp(card + (size_t)t * BFP8_TILE, host + (size_t)t * BFP8_TILE, BFP8_TILE) != 0;
    return differ;
}

/** The byte gates: DRAM-fed on BRISC + NCRISC, L1 -> L1 on each TRISC, plus the comparator red. */
static void expand_gates(bench_context *c, const ft_tt_kernel_program *program, expand_plan *plan,
                         uint32_t layout, const unsigned char *bfp6, const unsigned char *expanded)
{
    unsigned char *card = calloc(EXPAND_TILES, BFP8_TILE), *mutant = malloc((size_t)EXPAND_TILES * BFP8_TILE);
    uint32_t status[5][4];
    expand_arm arm;
    int rc;

    memset(&arm, 0, sizeof(arm));
    arm.mode = 2u;
    arm.layout = layout;
    arm.repeats = 1u;
    arm.tiles[0] = EXPAND_TILES / 2u;
    arm.tiles[1] = EXPAND_TILES / 2u;
    arm.in_offset[1] = EXPAND_TILES / 2u;
    rc = noc_write(c, c->dram[DRAM_IN_BANK], DRAM_BASE, bfp6, (size_t)EXPAND_TILES * BFP6_TILE);
    memset(card, 0xA5, (size_t)EXPAND_TILES * BFP8_TILE);
    rc |= noc_write(c, c->dram[DRAM_OUT_BANK], DRAM_BASE, card, (size_t)EXPAND_TILES * BFP8_TILE);
    rc |= expand_launch(c, program, plan, &arm, status);
    rc |= noc_read(c, c->dram[DRAM_OUT_BANK], DRAM_BASE, card, (size_t)EXPAND_TILES * BFP8_TILE);
    ft_tt_test_check_fmt("route 1 GATE: DRAM-fed BRISC + NCRISC expansion IDENTICAL to ft_bfp6_expand_tiles",
                         rc == FT_TT_OK && status[0][3] == EXPAND_TILES / 2u && status[1][3] == EXPAND_TILES / 2u &&
                             tiles_differing(card, expanded, EXPAND_TILES) == 0u,
                         "layout %s: %u of %u tiles differ (rc %d, done %u + %u)", LAYOUT_NAME(layout),
                         tiles_differing(card, expanded, EXPAND_TILES), EXPAND_TILES, rc, status[0][3], status[1][3]);
    memcpy(mutant, expanded, (size_t)EXPAND_TILES * BFP8_TILE);
    for (size_t t = 0; t < EXPAND_TILES; t++)
        for (size_t i = 64; i < BFP8_TILE; i++)
            mutant[t * BFP8_TILE + i] = (unsigned char)(mutant[t * BFP8_TILE + i] >> 1);
    ft_tt_test_check_fmt("RED: the same card bytes against a datum<<1 host expansion DIFFER", tiles_differing(card, mutant, EXPAND_TILES) > EXPAND_TILES / 2u,
                         "%u of %u tiles differ", tiles_differing(card, mutant, EXPAND_TILES), EXPAND_TILES);

    /* TRISCs: L1 -> L1, one tile each per slot, read back from each TRISC's out slots. */
    memset(&arm, 0, sizeof(arm));
    arm.mode = 0u;
    arm.layout = layout;
    arm.repeats = 1u;
    for (unsigned t = 0; t < 3u; t++) {
        arm.tiles[2u + t] = 4u;
        arm.in_offset[2u + t] = 10u + 4u * t;
    }
    rc = noc_write(c, core_coord(c), plan->in, bfp6, (size_t)EXPAND_TILES * BFP6_TILE);
    rc |= expand_launch(c, program, plan, &arm, status);
    for (unsigned t = 0; t < 3u && rc == FT_TT_OK; t++) {
        rc = noc_read(c, core_coord(c), plan->out[2u + t], card, 4u * BFP8_TILE);
        ft_tt_test_check_fmt("route 3 GATE: TRISC L1 -> L1 expansion IDENTICAL", rc == FT_TT_OK &&
                             status[2u + t][3] == 4u && tiles_differing(card, expanded + (size_t)(10u + 4u * t) * BFP8_TILE, 4u) == 0u,
                             "TRISC%u layout %s: %u of 4 tiles differ", t, LAYOUT_NAME(layout),
                             tiles_differing(card, expanded + (size_t)(10u + 4u * t) * BFP8_TILE, 4u));
    }
    free(card);
    free(mutant);
}

/** One timed arm, `repeats` launches, best per-processor rates; prints and returns the core sum. */
static double expand_time(bench_context *c, const ft_tt_kernel_program *program, expand_plan *plan,
                          const expand_arm *arm, const char *label)
{
    double best[5] = {0}, total = 0.0;
    uint32_t status[5][4];

    for (uint32_t r = 0; r < c->repeats; r++) {
        if (expand_launch(c, program, plan, arm, status) != FT_TT_OK) {
            ft_tt_test_check_fmt("expander arm runs", false, "%s", label);
            return 0.0;
        }
        for (unsigned p = 0; p < 5u; p++) {
            double rate = status_rate(c, status[p]);

            if (rate > best[p])
                best[p] = rate;
        }
    }
    for (unsigned p = 0; p < 5u; p++)
        total += best[p];
    ft_tt_test_step("PRICE %-34s tiles/us  BRISC %.4f  NCRISC %.4f  TRISC %.4f %.4f %.4f  = core %.4f (%.2f us/tile/RISC-V)",
                    label, best[0], best[1], best[2], best[3], best[4], total,
                    best[0] > 0.0 ? 1.0 / best[0] : (best[2] > 0.0 ? 1.0 / best[2] : 0.0));
    return total;
}

static void stage_expand(bench_context *c)
{
    ft_tt_program_directory program;
    expand_plan plan;
    float *values = malloc((size_t)EXPAND_TILES * TILE_VALUES * sizeof(float));
    unsigned char *stream = malloc((size_t)EXPAND_TILES * BFP6_TILE), *planes = malloc((size_t)EXPAND_TILES * BFP6_TILE);
    unsigned char *fold = malloc((size_t)EXPAND_TILES * BFP6_TILE);
    unsigned char *expanded = malloc((size_t)EXPAND_TILES * BFP8_TILE);
    int rc = load_program("q36_expand", &program);

    ft_tt_test_check_fmt("expander program loads", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK || !values || !stream || !planes || !fold || !expanded)
        goto done;
    rc = expand_plan_place(&plan);
    ft_tt_test_check_fmt("expander L1 plan places", rc == FT_TT_OK, "in %u x %u B at 0x%x", EXPAND_TILES, BFP6_TILE,
                         plan.in);
    for (size_t i = 0; i < (size_t)EXPAND_TILES * TILE_VALUES; i++)
        values[i] = random_weight();
    ft_bfp_pack_tiles(FT_BFP_BFP6_B, values, EXPAND_TILES, stream);
    ft_bfp6_relayout_tiles(FT_BFP6_LAYOUT_STREAM, FT_BFP6_LAYOUT_PLANES, stream, EXPAND_TILES, planes);
    ft_bfp6_fold_from_stream(stream, EXPAND_TILES, fold);
    ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, stream, EXPAND_TILES, expanded);
    for (uint32_t layout = 0; layout < 3u; layout++) {
        const unsigned char *input = layout == 2u ? fold : layout ? planes : stream;
        expand_arm arm;
        char label[64];

        expand_gates(c, &program.program, &plan, layout, input, expanded);
        noc_write(c, core_coord(c), plan.in, input, (size_t)EXPAND_TILES * BFP6_TILE);
        noc_write(c, c->dram[DRAM_IN_BANK], DRAM_BASE, input, (size_t)EXPAND_TILES * BFP6_TILE);
        memset(&arm, 0, sizeof(arm));
        arm.layout = layout;
        arm.repeats = 20u;
        arm.tiles[0] = EXPAND_TILES;
        snprintf(label, sizeof(label), "[%s] BRISC L1->L1", LAYOUT_NAME(layout));
        expand_time(c, &program.program, &plan, &arm, label);
        arm.mode = 1u;
        snprintf(label, sizeof(label), "[%s] BRISC DRAM-fed", LAYOUT_NAME(layout));
        expand_time(c, &program.program, &plan, &arm, label);
        arm.tiles[0] = arm.tiles[1] = EXPAND_TILES / 2u;
        arm.in_offset[1] = EXPAND_TILES / 2u;
        snprintf(label, sizeof(label), "[%s] BRISC+NCRISC DRAM-fed (route 1)", LAYOUT_NAME(layout));
        expand_time(c, &program.program, &plan, &arm, label);
        arm.mode = 0u;
        for (unsigned p = 0; p < 5u; p++) {
            arm.tiles[p] = 16u;
            arm.in_offset[p] = 16u * p;
        }
        snprintf(label, sizeof(label), "[%s] 5 RISC-Vs L1->L1 (route 3)", LAYOUT_NAME(layout));
        expand_time(c, &program.program, &plan, &arm, label);
    }
    ft_tt_program_directory_release(&program);
done:
    free(values);
    free(stream);
    free(planes);
    free(fold);
    free(expanded);
}

/** RED on the card: the expander built with datum << 1 (compile-time arg 0 = 1) must write bytes
 *  the same byte gate rejects — a red that leaves DRAM unwritten would be vacuous, so the
 *  output buffer is poisoned first and the gate counts tiles that differ from the TRUE answer. */
static void stage_expand_red(bench_context *c)
{
    ft_tt_program_directory program;
    expand_plan plan;
    expand_arm arm;
    uint32_t status[5][4];
    float *values = malloc((size_t)EXPAND_TILES * TILE_VALUES * sizeof(float));
    unsigned char *stream = malloc((size_t)EXPAND_TILES * BFP6_TILE), *expanded = malloc((size_t)EXPAND_TILES * BFP8_TILE);
    unsigned char *card = malloc((size_t)EXPAND_TILES * BFP8_TILE);
    int rc = load_program("q36_expand_red", &program);

    if (rc != FT_TT_OK || !values || !stream || !expanded || !card || expand_plan_place(&plan) != FT_TT_OK) {
        ft_tt_test_check_fmt("red expander program loads", false, "%s", ft_tt_error_string(rc));
        goto done;
    }
    for (size_t i = 0; i < (size_t)EXPAND_TILES * TILE_VALUES; i++)
        values[i] = random_weight();
    ft_bfp_pack_tiles(FT_BFP_BFP6_B, values, EXPAND_TILES, stream);
    ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, stream, EXPAND_TILES, expanded);
    memset(&arm, 0, sizeof(arm));
    arm.mode = 2u;
    arm.repeats = 1u;
    arm.tiles[0] = arm.tiles[1] = EXPAND_TILES / 2u;
    arm.in_offset[1] = EXPAND_TILES / 2u;
    memset(card, 0xA5, (size_t)EXPAND_TILES * BFP8_TILE);
    rc = noc_write(c, c->dram[DRAM_IN_BANK], DRAM_BASE, stream, (size_t)EXPAND_TILES * BFP6_TILE);
    rc |= noc_write(c, c->dram[DRAM_OUT_BANK], DRAM_BASE, card, (size_t)EXPAND_TILES * BFP8_TILE);
    rc |= expand_launch(c, &program.program, &plan, &arm, status);
    rc |= noc_read(c, c->dram[DRAM_OUT_BANK], DRAM_BASE, card, (size_t)EXPAND_TILES * BFP8_TILE);
    {
        size_t poison = 0;

        for (size_t i = 0; i < (size_t)EXPAND_TILES * BFP8_TILE; i++)
            poison += card[i] == 0xA5;
        ft_tt_test_check_fmt("RED (card): the expander built with datum << 1 writes bytes the gate REJECTS",
                             rc == FT_TT_OK && status[0][3] == EXPAND_TILES / 2u &&
                                 tiles_differing(card, expanded, EXPAND_TILES) == EXPAND_TILES &&
                                 poison < (size_t)EXPAND_TILES * BFP8_TILE / 8u,
                             "%u of %u tiles DIFFER, %zu poison bytes left (it wrote)",
                             tiles_differing(card, expanded, EXPAND_TILES), EXPAND_TILES, poison);
    }
    ft_tt_program_directory_release(&program);
done:
    free(values);
    free(stream);
    free(expanded);
    free(card);
}

/* ================================================================ stage split */

#include "q36_split_matmul_stage.inc"

/* ================================================================ stage sfpu (route-2 floor) */

#define FLOOR_TILES 1024u
#define FLOOR_DEPTH 8u

static int floor_launch(bench_context *c, const ft_tt_kernel_program *program, uint32_t *ticks)
{
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    ft_tt_launch_cb_entry placed[3];
    ft_tt_op_cb_request requests[3] = {{1, 576u, 2u * FLOOR_DEPTH}, {16, BFP8_TILE, 4u}, {24, 64u, 1u}};
    uint32_t reader[6], writer[5], compute[1] = {FLOOR_TILES}, status[2] = {0, 0};
    ft_tt_launch_config config;
    ft_tt_op_args args[5];
    int rc = ft_tt_op_program_load(&c->session.noc, c->x, c->y, program, NULL, &config);

    if (rc == FT_TT_OK)
        rc = ft_tt_op_cb_place(requests, 3u, placed, cb_buffer, sizeof(cb_buffer), &config.local_cb_mask);
    if (rc != FT_TT_OK)
        return rc;
    config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    config.brisc_noc_id = 0u;
    reader[0] = FLOOR_TILES;
    reader[1] = c->dram[DRAM_IN_BANK].x;
    reader[2] = c->dram[DRAM_IN_BANK].y;
    reader[3] = DRAM_BASE;
    reader[4] = FLOOR_DEPTH;
    reader[5] = placed[2].address;
    writer[0] = FLOOR_TILES;
    writer[1] = c->dram[DRAM_OUT_BANK].x;
    writer[2] = c->dram[DRAM_OUT_BANK].y;
    writer[3] = DRAM_BASE;
    writer[4] = placed[2].address;
    args[0] = (ft_tt_op_args){FT_TT_PROC_DM0, reader, 6u};
    args[1] = (ft_tt_op_args){FT_TT_PROC_DM1, writer, 5u};
    args[2] = (ft_tt_op_args){FT_TT_PROC_MATH0, compute, 1u};
    args[3] = (ft_tt_op_args){FT_TT_PROC_MATH1, compute, 1u};
    args[4] = (ft_tt_op_args){FT_TT_PROC_MATH2, compute, 1u};
    rc = ft_tt_op_launch(&c->session.noc, c->x, c->y, &config, cb_buffer, sizeof(cb_buffer), args, 5u, TIMEOUT_MS,
                         NULL);
    if (rc == FT_TT_OK)
        rc = noc_read(c, core_coord(c), placed[2].address, status, sizeof(status));
    *ticks = status[1] - status[0];
    return rc;
}

/** Route 2's floor: unpack the high plane (a BFP4_B tile) -> Dst -> pack BFP8_B, nothing else. */
static void stage_sfpu_floor(bench_context *c)
{
    ft_tt_program_directory program;
    float *values = malloc((size_t)FLOOR_TILES * TILE_VALUES * sizeof(float));
    float *want = malloc((size_t)FLOOR_TILES * TILE_VALUES * sizeof(float));
    float *got = malloc((size_t)FLOOR_TILES * TILE_VALUES * sizeof(float));
    unsigned char *high = malloc((size_t)FLOOR_TILES * 576u), *out = malloc((size_t)FLOOR_TILES * BFP8_TILE);
    uint32_t ticks = 0xFFFFFFFFu, best = 0xFFFFFFFFu;
    size_t differ = 0;
    int rc = load_program("q36_sfpu_floor", &program);

    ft_tt_test_check_fmt("route-2 floor program loads", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK || !values || !want || !got || !high || !out)
        goto done;
    for (size_t i = 0; i < (size_t)FLOOR_TILES * TILE_VALUES; i++)
        values[i] = random_weight();
    ft_bfp_pack_tiles(FT_BFP_BFP4_B, values, FLOOR_TILES, high);
    ft_bfp_unpack_tiles(FT_BFP_BFP4_B, high, FLOOR_TILES, want);
    memset(out, 0xA5, (size_t)FLOOR_TILES * BFP8_TILE);
    rc = noc_write(c, c->dram[DRAM_IN_BANK], DRAM_BASE, high, (size_t)FLOOR_TILES * 576u);
    rc |= noc_write(c, c->dram[DRAM_OUT_BANK], DRAM_BASE, out, (size_t)FLOOR_TILES * BFP8_TILE);
    for (uint32_t r = 0; r < c->repeats && rc == FT_TT_OK; r++) {
        rc = floor_launch(c, &program.program, &ticks);
        if (ticks < best)
            best = ticks;
    }
    if (rc == FT_TT_OK)
        rc = noc_read(c, c->dram[DRAM_OUT_BANK], DRAM_BASE, out, (size_t)FLOOR_TILES * BFP8_TILE);
    ft_bfp_unpack_tiles(FT_BFP_BFP8_B, out, FLOOR_TILES, got);
    for (size_t i = 0; i < (size_t)FLOOR_TILES * TILE_VALUES; i++)
        differ += memcmp(&want[i], &got[i], sizeof(float)) != 0;
    ft_tt_test_check_fmt("route-2 floor GATE: packed BFP8_B values == the BFP4_B plane's values", rc == FT_TT_OK &&
                         differ == 0u, "%zu of %u values differ", differ, FLOOR_TILES * TILE_VALUES);
    ft_tt_test_step("PRICE %-40s %.4f tiles/us a core (%.3f us/tile) — unpack + copy + pack, NO decode work",
                    "route 2 floor (one core)", (double)FLOOR_TILES / ((double)best / c->cycles_per_us),
                    ((double)best / c->cycles_per_us) / FLOOR_TILES);
    ft_tt_program_directory_release(&program);
done:
    free(values);
    free(want);
    free(got);
    free(high);
    free(out);
}

/* ================================================================ main */

/** Device wall-clock ticks per microsecond: one long L1 expansion timed by host and device. */
static void calibrate_clock(bench_context *c)
{
    ft_tt_program_directory program;
    expand_plan plan;
    expand_arm arm;
    uint32_t status[5][4];
    double started, elapsed;

    c->cycles_per_us = 1350.0;
    if (load_program("q36_expand", &program) != FT_TT_OK || expand_plan_place(&plan) != FT_TT_OK)
        return;
    memset(&arm, 0, sizeof(arm));
    arm.repeats = 600u;
    arm.tiles[0] = EXPAND_TILES;
    started = now_us();
    if (expand_launch(c, &program.program, &plan, &arm, status) == FT_TT_OK && status[0][0] == STATUS_MAGIC) {
        elapsed = now_us() - started;
        c->cycles_per_us = (double)(status[0][2] - status[0][1]) / elapsed;
        ft_tt_test_step("clock: %u device ticks in %.0f us of host time (launch included) = %.1f ticks/us "
                        "(an UNDER-estimate by the launch's share)",
                        status[0][2] - status[0][1], elapsed, c->cycles_per_us);
    }
    ft_tt_program_directory_release(&program);
}

int main(void)
{
    bench_context c;
    const char *stages = getenv("Q36_BENCH_STAGES");
    uint32_t index = 75u;
    int rc;

    memset(&c, 0, sizeof(c));
    c.repeats = getenv("Q36_BENCH_REPEATS") ? (uint32_t)atoi(getenv("Q36_BENCH_REPEATS")) : 5u;
    if (!stages || !*stages)
        stages = "expand,red,split,sfpu";
    rc = ft_tt_hardware_session_open(&c.session);
    ft_tt_test_check_fmt("card opens", rc == FT_TT_OK, "%s", ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return ft_tt_test_finish();
    (void)ft_tt_tensix_index_from_environment(&index);
    rc = ft_tt_hardware_session_tensix(&c.session, index, &c.x, &c.y);
    if (rc == FT_TT_OK)
        rc = ft_tt_hardware_session_claim_cores(&c.session, &index, 1u, "qwen36 phase 1 expander bench", NULL, NULL);
    for (uint32_t bank = 0; bank < 8u && rc == FT_TT_OK; bank++)
        rc = ft_tt_coord_dram_bank(&c.session.grid, bank, FT_TT_COORD_TRANSLATED, &c.dram[bank]);
    if (rc == FT_TT_OK)
        rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &c.firmware);
    if (rc == FT_TT_OK)
        rc = ft_tt_core_boot_firmware(&c.session.noc, c.x, c.y, &c.firmware, NULL);
    ft_tt_test_check_fmt("Tensix claimed, firmware up", rc == FT_TT_OK, "index %u at (%u,%u): %s", index, c.x, c.y,
                         ft_tt_error_string(rc));
    if (rc == FT_TT_OK) {
        calibrate_clock(&c);
        if (strstr(stages, "expand"))
            stage_expand(&c);
        if (strstr(stages, "red"))
            stage_expand_red(&c);
        if (strstr(stages, "split"))
            stage_split(&c);
        if (strstr(stages, "sfpu"))
            stage_sfpu_floor(&c);
    }
    ft_tt_hardware_session_close(&c.session);
    return ft_tt_test_finish();
}
