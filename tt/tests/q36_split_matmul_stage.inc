/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * q36_split_matmul_stage.inc — the `split` stage of test_q36_expander_bench.c (plan 1.4b):
 * BFP4_B high + BFP2_B residual as two native matmul passes into one Dst, priced against a
 * BFP8_B weight on the same kernel. Included by that file only (its context and helpers).
 *
 * status: new 2026-10-28.  issues: none.  todo: none.
 */

#define SPLIT_KT 64u    /* K = 2048, the hidden size: gate/up of one expert */
#define SPLIT_NT 16u    /* N = 512, one expert's ff */
#define SPLIT_DEPTH 16u
#define SPLIT_A_BANK 2u
#define SPLIT_B1_BANK 3u
#define SPLIT_OUT_BANK 4u
#define SPLIT_B2_BANK 5u

typedef struct split_operands {
    uint16_t *a_tiles;                 /* bf16, Kt tiles */
    float *a;                          /* the bf16 values, 32 x K */
    unsigned char *high, *low, *bfp8, *bfp6x8; /* N-major tile streams */
    float *w_split, *w_high, *w_low, *w_bfp8, *w_bfp6; /* the values each stream decodes to, K x N */
} split_operands;

/** Row-major K x N -> 32x32 row-major tiles in N-major order (tile nt*Kt + kt). */
static void weight_to_tiles(const float *w, float *tiles)
{
    const uint32_t n = SPLIT_NT * TILE;

    for (uint32_t nt = 0; nt < SPLIT_NT; nt++)
        for (uint32_t kt = 0; kt < SPLIT_KT; kt++)
            for (uint32_t r = 0; r < TILE; r++)
                memcpy(tiles + ((size_t)(nt * SPLIT_KT + kt) * TILE_VALUES + r * TILE),
                       w + (size_t)(kt * TILE + r) * n + nt * TILE, TILE * sizeof(float));
}

static void tiles_to_weight(const float *tiles, float *w)
{
    const uint32_t n = SPLIT_NT * TILE;

    for (uint32_t nt = 0; nt < SPLIT_NT; nt++)
        for (uint32_t kt = 0; kt < SPLIT_KT; kt++)
            for (uint32_t r = 0; r < TILE; r++)
                memcpy(w + (size_t)(kt * TILE + r) * n + nt * TILE,
                       tiles + ((size_t)(nt * SPLIT_KT + kt) * TILE_VALUES + r * TILE), TILE * sizeof(float));
}

static int split_operands_build(split_operands *o)
{
    const size_t tiles = (size_t)SPLIT_KT * SPLIT_NT, k = SPLIT_KT * TILE, weights = tiles * TILE_VALUES;
    float *w = malloc(weights * sizeof(float)), *t = malloc(weights * sizeof(float));
    unsigned char *bfp6 = malloc(tiles * BFP6_TILE);

    memset(o, 0, sizeof(*o));
    o->a_tiles = malloc(k * TILE * sizeof(uint16_t));
    o->a = malloc(k * TILE * sizeof(float));
    o->high = malloc(tiles * FT_BFP_SPLIT_HIGH_TILE_BYTES);
    o->low = malloc(tiles * FT_BFP_SPLIT_LOW_TILE_BYTES);
    o->bfp8 = malloc(tiles * BFP8_TILE);
    o->bfp6x8 = malloc(tiles * BFP8_TILE);
    o->w_split = malloc(weights * sizeof(float));
    o->w_high = malloc(weights * sizeof(float));
    o->w_low = malloc(weights * sizeof(float));
    o->w_bfp8 = malloc(weights * sizeof(float));
    o->w_bfp6 = malloc(weights * sizeof(float));
    if (!w || !t || !bfp6 || !o->a_tiles || !o->a || !o->high || !o->low || !o->bfp8 || !o->bfp6x8 || !o->w_split ||
        !o->w_high || !o->w_low || !o->w_bfp8 || !o->w_bfp6)
        return -1;
    random_state = 0x51364258ull;
    for (size_t i = 0; i < k * TILE; i++)
        o->a[i] = ft_float_from_bf16(ft_bf16_from_float(random_value() * 0.5f));
    {
        uint16_t *rows = malloc(k * TILE * sizeof(uint16_t));

        if (!rows)
            return -1;
        ft_bf16_from_float_array(o->a, rows, k * TILE);
        ft_tt_tilize_matrix(rows, sizeof(uint16_t), TILE, (uint32_t)k, o->a_tiles);
        free(rows);
    }
    for (size_t i = 0; i < weights; i++)
        w[i] = random_weight();
    weight_to_tiles(w, t);
    ft_bfp_split_pack_tiles(FT_BFP_SPLIT_RESIDUAL_E4, t, tiles, o->high, o->low);
    ft_bfp_pack_tiles(FT_BFP_BFP8_B, t, tiles, o->bfp8);
    ft_bfp_pack_tiles(FT_BFP_BFP6_B, t, tiles, bfp6);
    ft_bfp6_expand_tiles(FT_BFP6_LAYOUT_STREAM, bfp6, tiles, o->bfp6x8);
    ft_bfp_split_unpack_tiles(o->high, o->low, tiles, t);
    tiles_to_weight(t, o->w_split);
    ft_bfp_unpack_tiles(FT_BFP_BFP4_B, o->high, tiles, t);
    tiles_to_weight(t, o->w_high);
    ft_bfp_unpack_tiles(FT_BFP_BFP2_B, o->low, tiles, t);
    tiles_to_weight(t, o->w_low);
    ft_bfp_unpack_tiles(FT_BFP_BFP8_B, o->bfp8, tiles, t);
    tiles_to_weight(t, o->w_bfp8);
    ft_bfp_unpack_tiles(FT_BFP_BFP8_B, o->bfp6x8, tiles, t);
    tiles_to_weight(t, o->w_bfp6);
    free(w);
    free(t);
    free(bfp6);
    return 0;
}

static void split_operands_free(split_operands *o)
{
    free(o->a_tiles);
    free(o->a);
    free(o->high);
    free(o->low);
    free(o->bfp8);
    free(o->bfp6x8);
    free(o->w_split);
    free(o->w_high);
    free(o->w_low);
    free(o->w_bfp8);
    free(o->w_bfp6);
}

/** fp64 reference of A (32 x K) times w (K x N), the values the card holds. */
static void split_reference(const split_operands *o, const float *w, float *out)
{
    const uint32_t k = SPLIT_KT * TILE, n = SPLIT_NT * TILE;

    for (uint32_t r = 0; r < TILE; r++)
        for (uint32_t col = 0; col < n; col++) {
            double sum = 0.0;

            for (uint32_t i = 0; i < k; i++)
                sum += (double)o->a[(size_t)r * k + i] * (double)w[(size_t)i * n + col];
            out[(size_t)r * n + col] = (float)sum;
        }
}

static double split_pcc(const float *a, const float *b, size_t count, double *max_abs)
{
    double ma = 0, mb = 0, cov = 0, va = 0, vb = 0;

    *max_abs = 0.0;
    for (size_t i = 0; i < count; i++) {
        ma += a[i];
        mb += b[i];
        if (fabs((double)a[i] - (double)b[i]) > *max_abs || a[i] != a[i] || b[i] != b[i])
            *max_abs = fabs((double)a[i] - (double)b[i]);
    }
    ma /= (double)count;
    mb /= (double)count;
    for (size_t i = 0; i < count; i++) {
        cov += ((double)a[i] - ma) * ((double)b[i] - mb);
        va += ((double)a[i] - ma) * ((double)a[i] - ma);
        vb += ((double)b[i] - mb) * ((double)b[i] - mb);
    }
    return va > 0 && vb > 0 ? cov / sqrt(va * vb) : 0.0;
}

/** One launch of a split-family program; returns device ticks reader-start -> writer-end. */
static int split_launch(bench_context *c, const ft_tt_kernel_program *program, uint32_t b1_bytes, uint32_t b2_bytes,
                        uint32_t mode, uint32_t *ticks)
{
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    ft_tt_launch_cb_entry placed[5];
    ft_tt_op_cb_request requests[5] = {{0, BF16_TILE, 2u * SPLIT_DEPTH}, {1, 0, 2u * SPLIT_DEPTH},
                                       {2, 0, 2u * SPLIT_DEPTH}, {16, BF16_TILE, 2u}, {24, 64u, 1u}};
    uint32_t reader[15], writer[5], compute[4], status[2] = {0, 0};
    ft_tt_launch_config config;
    ft_tt_op_args args[5];
    int rc;

    requests[1].page_bytes = b1_bytes;
    requests[2].page_bytes = b2_bytes;
    rc = ft_tt_op_program_load(&c->session.noc, c->x, c->y, program, NULL, &config);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_cb_place(requests, 5u, placed, cb_buffer, sizeof(cb_buffer), &config.local_cb_mask);
    if (rc != FT_TT_OK)
        return rc;
    config.local_cb_config_address = FT_TT_OP_CB_CONFIG_ADDRESS;
    config.brisc_noc_id = 0u;
    reader[0] = 1u;
    reader[1] = SPLIT_KT;
    reader[2] = SPLIT_NT;
    reader[3] = c->dram[SPLIT_A_BANK].x;
    reader[4] = c->dram[SPLIT_A_BANK].y;
    reader[5] = DRAM_BASE;
    reader[6] = c->dram[SPLIT_B1_BANK].x;
    reader[7] = c->dram[SPLIT_B1_BANK].y;
    reader[8] = DRAM_BASE;
    reader[9] = c->dram[SPLIT_B2_BANK].x;
    reader[10] = c->dram[SPLIT_B2_BANK].y;
    reader[11] = DRAM_BASE;
    reader[12] = SPLIT_DEPTH;
    reader[13] = mode;
    reader[14] = placed[4].address;
    writer[0] = SPLIT_NT;
    writer[1] = c->dram[SPLIT_OUT_BANK].x;
    writer[2] = c->dram[SPLIT_OUT_BANK].y;
    writer[3] = DRAM_BASE;
    writer[4] = placed[4].address;
    compute[0] = 1u;
    compute[1] = SPLIT_KT;
    compute[2] = SPLIT_NT;
    compute[3] = mode;
    args[0] = (ft_tt_op_args){FT_TT_PROC_DM0, reader, 15u};
    args[1] = (ft_tt_op_args){FT_TT_PROC_DM1, writer, 5u};
    args[2] = (ft_tt_op_args){FT_TT_PROC_MATH0, compute, 4u};
    args[3] = (ft_tt_op_args){FT_TT_PROC_MATH1, compute, 4u};
    args[4] = (ft_tt_op_args){FT_TT_PROC_MATH2, compute, 4u};
    rc = ft_tt_op_launch(&c->session.noc, c->x, c->y, &config, cb_buffer, sizeof(cb_buffer), args, 5u, TIMEOUT_MS,
                         NULL);
    if (rc == FT_TT_OK)
        rc = noc_read(c, core_coord(c), placed[4].address, status, sizeof(status));
    *ticks = status[1] - status[0];
    return rc;
}

/** Run one arm `repeats` times: best ticks, and the last run's output (bf16 tiles + floats). */
static int split_arm(bench_context *c, const char *program_name, const unsigned char *b1, uint32_t b1_bytes,
                     const unsigned char *b2, uint32_t b2_bytes, uint32_t mode, uint16_t *out_tiles, float *out,
                     double *best_ns)
{
    ft_tt_program_directory program;
    const size_t tiles = (size_t)SPLIT_KT * SPLIT_NT;
    uint32_t ticks = 0xFFFFFFFFu, best = 0xFFFFFFFFu;
    int rc = load_program(program_name, &program);

    if (rc != FT_TT_OK)
        return rc;
    memset(out_tiles, 0xA5, SPLIT_NT * BF16_TILE);
    rc = noc_write(c, c->dram[SPLIT_OUT_BANK], DRAM_BASE, out_tiles, SPLIT_NT * BF16_TILE);
    if (b1)
        rc |= noc_write(c, c->dram[SPLIT_B1_BANK], DRAM_BASE, b1, tiles * b1_bytes);
    if (b2)
        rc |= noc_write(c, c->dram[SPLIT_B2_BANK], DRAM_BASE, b2, tiles * b2_bytes);
    for (uint32_t r = 0; r < c->repeats && rc == FT_TT_OK; r++) {
        rc = split_launch(c, &program.program, b1_bytes, b2_bytes, mode, &ticks);
        if (ticks < best)
            best = ticks;
    }
    if (rc == FT_TT_OK)
        rc = noc_read(c, c->dram[SPLIT_OUT_BANK], DRAM_BASE, out_tiles, SPLIT_NT * BF16_TILE);
    if (rc == FT_TT_OK) {
        uint16_t *rows = malloc((size_t)TILE * SPLIT_NT * TILE * sizeof(uint16_t));

        if (!rows)
            rc = FT_TT_ENOMEM;
        else {
            ft_tt_untilize_matrix(out_tiles, sizeof(uint16_t), TILE, SPLIT_NT * TILE, rows);
            ft_float_from_bf16_array(rows, out, (size_t)TILE * SPLIT_NT * TILE);
            free(rows);
        }
    }
    *best_ns = (double)best / c->cycles_per_us * 1e3;
    ft_tt_program_directory_release(&program);
    return rc;
}

/** Score `out` against the reference for weights `w`; prints and returns the PCC. */
static double split_score(const split_operands *o, const float *w, const float *out, double *max_abs)
{
    const size_t n = (size_t)TILE * SPLIT_NT * TILE;
    float *expected = malloc(n * sizeof(float));
    double pcc;

    if (!expected)
        return 0.0;
    split_reference(o, w, expected);
    pcc = split_pcc(expected, out, n, max_abs);
    free(expected);
    return pcc;
}

typedef struct split_result {
    uint16_t tiles[SPLIT_NT * BF16_TILE / 2];
    float out[TILE * SPLIT_NT * TILE];
    double ns;
    int rc;
} split_result;

static void split_report_price(const char *label, const split_result *r, uint32_t passes)
{
    const double tiles = (double)SPLIT_KT * SPLIT_NT;

    ft_tt_test_step("PRICE %-40s %9.1f us a launch, %6.1f ns a weight tile-pass, %6.1f ns a BFP6-equivalent tile",
                    label, r->ns / 1e3, r->ns / (tiles * passes), r->ns / tiles);
}

static void stage_split(bench_context *c)
{
    split_operands o;
    static split_result hf4[3], hf2[3], b8_hf4, b8_hf2, b6_hf4, b6_hf2;
    double pcc, worst, worst_both;

    if (split_operands_build(&o) != 0) {
        ft_tt_test_check("split operands build", false, "out of memory");
        return;
    }
    ft_tt_test_step("split: Mt 1, Kt %u, Nt %u (%u weight tiles), depth %u, %u repeats, best of",
                    SPLIT_KT, SPLIT_NT, SPLIT_KT * SPLIT_NT, SPLIT_DEPTH, c->repeats);
    if (noc_write(c, c->dram[SPLIT_A_BANK], DRAM_BASE, o.a_tiles, (size_t)SPLIT_KT * BF16_TILE) != FT_TT_OK)
        goto done;
    for (uint32_t mode = 1; mode <= 3u; mode++) {
        hf4[mode - 1].rc = split_arm(c, "q36_split_hf4", o.high, 576u, o.low, 320u, mode, hf4[mode - 1].tiles,
                                     hf4[mode - 1].out, &hf4[mode - 1].ns);
        hf2[mode - 1].rc = split_arm(c, "q36_split_hf2", o.high, 576u, o.low, 320u, mode, hf2[mode - 1].tiles,
                                     hf2[mode - 1].out, &hf2[mode - 1].ns);
    }
    b8_hf4.rc = split_arm(c, "q36_bfp8_hf4", o.bfp8, BFP8_TILE, o.bfp8, BFP8_TILE, 1u, b8_hf4.tiles, b8_hf4.out, &b8_hf4.ns);
    b8_hf2.rc = split_arm(c, "q36_bfp8_hf2", o.bfp8, BFP8_TILE, o.bfp8, BFP8_TILE, 1u, b8_hf2.tiles, b8_hf2.out, &b8_hf2.ns);
    b6_hf4.rc = split_arm(c, "q36_bfp8_hf4", o.bfp6x8, BFP8_TILE, o.bfp6x8, BFP8_TILE, 1u, b6_hf4.tiles, b6_hf4.out, &b6_hf4.ns);
    b6_hf2.rc = split_arm(c, "q36_bfp8_hf2", o.bfp6x8, BFP8_TILE, o.bfp6x8, BFP8_TILE, 1u, b6_hf2.tiles, b6_hf2.out, &b6_hf2.ns);

    pcc = split_score(&o, o.w_split, hf4[2].out, &worst_both);
    ft_tt_test_check_fmt("1.4b GATE: high + low passes into one Dst match the fp64 reference of high + low",
                         hf4[2].rc == FT_TT_OK && pcc >= 0.9999, "PCC %.7f, max abs %.5g", pcc, worst_both);
    pcc = split_score(&o, o.w_high, hf4[0].out, &worst);
    ft_tt_test_check_fmt("high pass alone (BFP4_B) matches its reference", hf4[0].rc == FT_TT_OK && pcc >= 0.9999,
                         "PCC %.7f, max abs %.5g", pcc, worst);
    pcc = split_score(&o, o.w_low, hf4[1].out, &worst);
    ft_tt_test_check_fmt("low pass alone (BFP2_B) matches its reference — the unpacker reads BFP2_B",
                         hf4[1].rc == FT_TT_OK && pcc >= 0.999, "PCC %.7f, max abs %.5g", pcc, worst);
    pcc = split_score(&o, o.w_split, hf4[0].out, &worst);
    ft_tt_test_check_fmt("RED: the high pass alone against the high + low reference is WORSE", worst > 4.0 * worst_both,
                         "max abs %.5g vs %.5g with both passes (PCC %.7f)", worst, worst_both, pcc);
    for (uint32_t m = 0; m < 3u; m++)
        ft_tt_test_check_fmt("phases {0,2} IDENTICAL to HiFi4 for <=5-bit weights", hf2[m].rc == FT_TT_OK &&
                             memcmp(hf2[m].tiles, hf4[m].tiles, sizeof(hf4[m].tiles)) == 0, "mode %u", m + 1u);
    ft_tt_test_check_fmt("phases {0,2} IDENTICAL to HiFi4 on an expanded BFP6_B tile", b6_hf2.rc == FT_TT_OK &&
                         memcmp(b6_hf2.tiles, b6_hf4.tiles, sizeof(b6_hf4.tiles)) == 0, "%s", "BFP8_B bytes of BFP6_B values");
    ft_tt_test_check_fmt("RED: phases {0,2} on a 7-bit BFP8_B weight DIFFERS from HiFi4", b8_hf2.rc == FT_TT_OK &&
                         memcmp(b8_hf2.tiles, b8_hf4.tiles, sizeof(b8_hf4.tiles)) != 0, "%s", "BFP8_B of the raw weight");
    pcc = split_score(&o, o.w_bfp8, b8_hf4.out, &worst);
    ft_tt_test_check_fmt("control: BFP8_B on the same kernel matches its reference", b8_hf4.rc == FT_TT_OK && pcc >= 0.9999,
                         "PCC %.7f, max abs %.5g", pcc, worst);
    pcc = split_score(&o, o.w_bfp6, b6_hf4.out, &worst);
    ft_tt_test_check_fmt("expanded BFP6_B tiles match the BFP6_B reference", b6_hf4.rc == FT_TT_OK && pcc >= 0.9999,
                         "PCC %.7f, max abs %.5g", pcc, worst);

    split_report_price("BFP8_B, HiFi4 (control, Ornith's format)", &b8_hf4, 1u);
    split_report_price("BFP8_B of BFP6_B values, phases {0,2}", &b6_hf2, 1u);
    split_report_price("BFP4_B high only, HiFi4", &hf4[0], 1u);
    split_report_price("BFP2_B low only, HiFi4", &hf4[1], 1u);
    split_report_price("high + low (1.4b), HiFi4", &hf4[2], 2u);
    split_report_price("BFP4_B high only, phases {0,2}", &hf2[0], 1u);
    split_report_price("BFP2_B low only, phases {0,2}", &hf2[1], 1u);
    split_report_price("high + low (1.4b), phases {0,2}", &hf2[2], 2u);
done:
    split_operands_free(&o);
}
