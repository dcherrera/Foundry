/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_attn_decode_rows.c
 * @brief Bit-exact gate for fc_attn_flash_decode_rows_q4_ctx (the R-row
 *        speculative-verify attention) against the decode slots kernel.
 *
 * Random Q4 ring + fp16 sink/window shadows at the Qwen3.6 shape (16 q heads,
 * 2 kv heads, head_dim 256, sink 4, win 128). For each pos0 in a sweep that
 * crosses the split-plan edges (n_splits steps at multiples of 256, the
 * 32-split cap at 8192, split_len steps every 512 past it) and the window edge:
 *   ROWS  one rows call over R rows at pos0..pos0+R-1
 *   REF   R slots calls (nslots 1, pos = pos0+r), the plain decode step
 *   DEV   the device-position form (fc_attn_flash_decode_rows_q4_dev_ctx), too
 * and memcmp row r of ROWS against REF r, ungated and gated (z folded into the
 * combine). Then the append parity the verify relies on: the batched append
 * (fc_attn_kv_append_q4_ctx, R rows) vs R slots appends must leave identical
 * ring, sink and window bytes.
 * RED arm (must FAIL): REF against the prefill flash kernel over the same rows
 * (a different reduction order) — proves the comparison sees a ~1-ulp change.
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/test_attn_decode_rows cuda/tests/test_attn_decode_rows.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run:  CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/test_attn_decode_rows [rows]
 *       (FC_ATTN_ROWS_PER_BLOCK=1..5 varies the rows per block; all must PASS)
 * Last line: "RESULT exact=PASS|FAIL red=FAIL|PASS".
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T_HEAD     16
#define T_HEAD_KV  2
#define T_HD       256
#define T_MAXSEQ   16384
#define T_SINK     4
#define T_WIN      128
#define T_WMAX     256      /* FC_KVQ4_WMAX */
#define T_SINKMAX  8        /* FC_KVQ4_SINKMAX */
#define T_KPS      256
#define T_CAP      32
#define T_ROWB     ((T_HD / 32) * 18)

static uint32_t rng_state = 0x9E3779B9u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}
static float frand(void) { return ((float)(rng() & 0xFFFFFF) / 8388608.0f) - 1.0f; }

/* fp16 bits of a value in a sane range (scale or shadow element). */
static uint16_t half_bits(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = (x >> 13) & 0x3FFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) exp = 30;
    return (uint16_t)(sign | ((uint32_t)exp << 10) | man);
}

static void fill_q4_ring(unsigned char *ring, size_t rows)
{
    for (size_t r = 0; r < rows; r++)
        for (int b = 0; b < T_HD / 32; b++) {
            unsigned char *bp = ring + r * T_ROWB + (size_t)b * 18;
            uint16_t d = half_bits(0.01f + 0.05f * (float)(rng() % 1000) / 1000.0f);
            bp[0] = (unsigned char)(d & 0xFF);
            bp[1] = (unsigned char)(d >> 8);
            for (int j = 0; j < 16; j++)
                bp[2 + j] = (unsigned char)(rng() & 0xFF);
        }
}

static void fill_half(uint16_t *h, size_t n)
{
    for (size_t i = 0; i < n; i++)
        h[i] = half_bits(0.5f * frand());
}

typedef struct {
    fc_buffer *kc, *vc, *ks, *vs, *kw, *vw;
} ring_set;

static ring_set ring_upload(const unsigned char *kc, const unsigned char *vc,
                            const uint16_t *ks, const uint16_t *vs,
                            const uint16_t *kw, const uint16_t *vw)
{
    size_t ring_b = (size_t)T_HEAD_KV * T_MAXSEQ * T_ROWB;
    size_t sink_b = (size_t)T_HEAD_KV * T_SINKMAX * T_HD * 2;
    size_t win_b = (size_t)T_HEAD_KV * T_WMAX * T_HD * 2;
    ring_set r = {
        fc_buffer_from_ptr(kc, ring_b), fc_buffer_from_ptr(vc, ring_b),
        fc_buffer_from_ptr(ks, sink_b), fc_buffer_from_ptr(vs, sink_b),
        fc_buffer_from_ptr(kw, win_b), fc_buffer_from_ptr(vw, win_b)
    };
    return r;
}

static void ring_free(ring_set *r)
{
    fc_buffer_free(r->kc); fc_buffer_free(r->vc); fc_buffer_free(r->ks);
    fc_buffer_free(r->vs); fc_buffer_free(r->kw); fc_buffer_free(r->vw);
}

/* REF row r: one slots call at pos0 + r. */
static int ref_row(fc_cuda_ctx *ctx, ring_set *rs, fc_buffer *q_row,
                   fc_buffer *z_row, fc_buffer *m, fc_buffer *l, fc_buffer *acc,
                   fc_buffer *out, int pos, float scale)
{
    int32_t p = pos;
    fc_buffer *pos_g = fc_buffer_from_ptr(&p, sizeof(p));
    int seq = pos + 1, ns = (seq + T_KPS - 1) / T_KPS;
    if (ns > T_CAP) ns = T_CAP;
    int sl = ((seq + ns - 1) / ns + 15) & ~15;
    int launch = (seq + sl - 1) / sl;
    int rc = z_row
        ? fc_attn_flash_decode_slots_q4_gated_ctx(ctx, q_row, rs->kc, rs->vc, rs->ks,
              rs->vs, rs->kw, rs->vw, m, l, acc, out, T_HEAD, T_HEAD_KV, T_HD,
              T_MAXSEQ, pos_g, NULL, 1, launch, T_KPS, T_CAP, T_SINK, T_WIN, 0, 0,
              scale, z_row)
        : fc_attn_flash_decode_slots_q4_ctx(ctx, q_row, rs->kc, rs->vc, rs->ks,
              rs->vs, rs->kw, rs->vw, m, l, acc, out, T_HEAD, T_HEAD_KV, T_HD,
              T_MAXSEQ, pos_g, NULL, 1, launch, T_KPS, T_CAP, T_SINK, T_WIN, 0, 0,
              scale);
    fc_sync();
    fc_buffer_free(pos_g);
    return rc;
}

/* Attention parity at one pos0. Returns mismatching rows (red: vs prefill). */
static int check_attention(fc_cuda_ctx *ctx, ring_set *rs, int pos0, int R,
                           int gated, int red, int dev)
{
    const size_t row_f = (size_t)T_HEAD * T_HD;
    float *q = malloc(row_f * R * 4), *z = malloc(row_f * R * 4);
    float *got = malloc(row_f * R * 4), *want = malloc(row_f * 4);
    for (size_t i = 0; i < row_f * R; i++) { q[i] = frand(); z[i] = 2.0f * frand(); }
    fc_buffer *q_g = fc_buffer_from_ptr(q, row_f * R * 4);
    fc_buffer *z_g = fc_buffer_from_ptr(z, row_f * R * 4);
    fc_buffer *out_g = fc_buffer_alloc(row_f * R * 4);
    fc_buffer *ref_g = fc_buffer_alloc(row_f * 4);
    size_t np = (size_t)R * T_HEAD * T_CAP;
    fc_buffer *m = fc_buffer_alloc(np * 4), *l = fc_buffer_alloc(np * 4);
    fc_buffer *acc = fc_buffer_alloc(np * T_HD * 4);
    float scale = 1.0f / sqrtf((float)T_HD);
    int bad = 0, rc;

    if (red)
        rc = fc_attn_flash_prefill_q4_ctx(ctx, q_g, rs->kc, rs->vc, rs->ks, rs->vs,
                                          rs->kw, rs->vw, out_g, T_HEAD, T_HEAD_KV,
                                          T_HD, T_MAXSEQ, pos0, R, T_SINK, T_WIN,
                                          0, 0, scale);
    else if (dev) {
        /* Device-position form: pos[0] on the die, grid = the max split count. */
        int32_t p = pos0, launch = 1;
        for (int r = 0; r < R; r++) {
            int seq = pos0 + r + 1, ns = (seq + T_KPS - 1) / T_KPS;
            if (ns > T_CAP) ns = T_CAP;
            int sl = ((seq + ns - 1) / ns + 15) & ~15;
            if ((seq + sl - 1) / sl > launch) launch = (seq + sl - 1) / sl;
        }
        fc_buffer *pos_g = fc_buffer_from_ptr(&p, sizeof(p));
        rc = fc_attn_flash_decode_rows_q4_dev_ctx(ctx, q_g, rs->kc, rs->vc, rs->ks, rs->vs,
                                                  rs->kw, rs->vw, m, l, acc,
                                                  gated ? z_g : NULL, out_g, T_HEAD,
                                                  T_HEAD_KV, T_HD, T_MAXSEQ, pos_g, R,
                                                  launch, T_KPS, T_CAP, T_SINK, T_WIN,
                                                  scale);
        fc_sync();
        fc_buffer_free(pos_g);
    } else
        rc = fc_attn_flash_decode_rows_q4_ctx(ctx, q_g, rs->kc, rs->vc, rs->ks,
                                              rs->vs, rs->kw, rs->vw, m, l, acc,
                                              gated ? z_g : NULL, out_g, T_HEAD,
                                              T_HEAD_KV, T_HD, T_MAXSEQ, pos0, R,
                                              T_KPS, T_CAP, T_SINK, T_WIN, 0, 0,
                                              scale);
    fc_sync();
    if (rc != 0) { printf("  pos0=%d launch rc=%d\n", pos0, rc); bad = R; goto out; }
    fc_buffer_to_cpu(out_g, got);
    for (int r = 0; r < R; r++) {
        fc_buffer *qv = fc_buffer_offset_view(q_g, (size_t)r * row_f * 4);
        fc_buffer *zv = fc_buffer_offset_view(z_g, (size_t)r * row_f * 4);
        if (ref_row(ctx, rs, qv, (gated && !red) ? zv : NULL, m, l, acc, ref_g,
                    pos0 + r, scale) != 0) { bad++; continue; }
        fc_buffer_free_view(qv);
        fc_buffer_free_view(zv);
        fc_buffer_to_cpu(ref_g, want);
        int nd = 0;
        for (size_t i = 0; i < row_f; i++)
            nd += memcmp(&got[(size_t)r * row_f + i], &want[i], 4) != 0;
        if (nd) {
            bad++;
            printf("  pos0=%d row=%d %s%s: %d floats differ\n", pos0, r,
                   red ? "RED prefill" : dev ? "rows dev" : "rows", gated ? " gated" : "",
                   nd);
        }
    }
out:
    fc_buffer_free(q_g); fc_buffer_free(z_g); fc_buffer_free(out_g);
    fc_buffer_free(ref_g); fc_buffer_free(m); fc_buffer_free(l); fc_buffer_free(acc);
    free(q); free(z); free(got); free(want);
    return bad;
}

/* Determinism: the slots reference for row 0 at pos0, run once right after
 * itself and once after a prefill-kernel launch over random q (which leaves
 * different shared memory / register state on the SMs). Must be identical. */
static int check_determinism(fc_cuda_ctx *ctx, ring_set *rs, int pos0, int gated)
{
    const size_t row_f = (size_t)T_HEAD * T_HD;
    float *q = malloc(row_f * 4 * 4), *z = malloc(row_f * 4);
    float *a = malloc(row_f * 4), *b = malloc(row_f * 4);
    for (size_t i = 0; i < row_f * 4; i++) q[i] = 50.0f * frand();
    for (size_t i = 0; i < row_f; i++) z[i] = frand();
    fc_buffer *q_g = fc_buffer_from_ptr(q, row_f * 4);
    fc_buffer *qd_g = fc_buffer_from_ptr(q, row_f * 4 * 4);
    fc_buffer *z_g = fc_buffer_from_ptr(z, row_f * 4);
    fc_buffer *o_g = fc_buffer_alloc(row_f * 4), *od_g = fc_buffer_alloc(row_f * 4 * 4);
    size_t np = (size_t)T_HEAD * T_CAP;
    fc_buffer *m = fc_buffer_alloc(np * 4), *l = fc_buffer_alloc(np * 4);
    fc_buffer *acc = fc_buffer_alloc(np * T_HD * 4);
    float scale = 1.0f / sqrtf((float)T_HD);
    ref_row(ctx, rs, q_g, gated ? z_g : NULL, m, l, acc, o_g, pos0, scale);
    ref_row(ctx, rs, q_g, gated ? z_g : NULL, m, l, acc, o_g, pos0, scale);
    fc_buffer_to_cpu(o_g, a);
    fc_attn_flash_prefill_q4_ctx(ctx, qd_g, rs->kc, rs->vc, rs->ks, rs->vs,
                                 rs->kw, rs->vw, od_g, T_HEAD, T_HEAD_KV, T_HD,
                                 T_MAXSEQ, pos0 > 3 ? pos0 - 3 : 0, 4, T_SINK,
                                 T_WIN, 0, 0, scale);
    fc_sync();
    ref_row(ctx, rs, q_g, gated ? z_g : NULL, m, l, acc, o_g, pos0, scale);
    fc_buffer_to_cpu(o_g, b);
    int nd = 0;
    for (size_t i = 0; i < row_f; i++)
        nd += memcmp(&a[i], &b[i], 4) != 0;
    if (nd)
        printf("  NONDETERMINISTIC slots ref pos0=%d%s: %d floats\n", pos0,
               gated ? " gated" : "", nd);
    fc_buffer_free(q_g); fc_buffer_free(qd_g); fc_buffer_free(z_g);
    fc_buffer_free(o_g); fc_buffer_free(od_g); fc_buffer_free(m);
    fc_buffer_free(l); fc_buffer_free(acc);
    free(q); free(z); free(a); free(b);
    return nd != 0;
}

/* Append parity: batched append of R rows vs R slots appends. */
static int check_append(fc_cuda_ctx *ctx, const unsigned char *kc0,
                        const uint16_t *s0, const uint16_t *w0, int pos0, int R)
{
    const size_t kv_f = (size_t)T_HEAD_KV * T_HD;
    float *kn = malloc(kv_f * R * 4), *vn = malloc(kv_f * R * 4);
    for (size_t i = 0; i < kv_f * R; i++) { kn[i] = frand(); vn[i] = frand(); }
    fc_buffer *k_g = fc_buffer_from_ptr(kn, kv_f * R * 4);
    fc_buffer *v_g = fc_buffer_from_ptr(vn, kv_f * R * 4);
    ring_set a = ring_upload(kc0, kc0, s0, s0, w0, w0);
    ring_set b = ring_upload(kc0, kc0, s0, s0, w0, w0);
    fc_attn_kv_append_q4_ctx(ctx, k_g, v_g, a.kc, a.vc, a.ks, a.vs, a.kw, a.vw,
                             T_HEAD_KV, T_HD, T_MAXSEQ, pos0, R, T_SINK, T_WIN);
    for (int r = 0; r < R; r++) {
        int32_t p = pos0 + r;
        fc_buffer *pos_g = fc_buffer_from_ptr(&p, 4);
        fc_buffer *kv = fc_buffer_offset_view(k_g, (size_t)r * kv_f * 4);
        fc_buffer *vv = fc_buffer_offset_view(v_g, (size_t)r * kv_f * 4);
        fc_attn_kv_append_slots_q4_ctx(ctx, kv, vv, b.kc, b.vc, b.ks, b.vs, b.kw,
                                       b.vw, T_HEAD_KV, T_HD, T_MAXSEQ, pos_g,
                                       NULL, 1, T_SINK, T_WIN);
        fc_sync();
        fc_buffer_free_view(kv);
        fc_buffer_free_view(vv);
        fc_buffer_free(pos_g);
    }
    fc_sync();
    size_t sizes[6] = {
        (size_t)T_HEAD_KV * T_MAXSEQ * T_ROWB, (size_t)T_HEAD_KV * T_MAXSEQ * T_ROWB,
        (size_t)T_HEAD_KV * T_SINKMAX * T_HD * 2, (size_t)T_HEAD_KV * T_SINKMAX * T_HD * 2,
        (size_t)T_HEAD_KV * T_WMAX * T_HD * 2, (size_t)T_HEAD_KV * T_WMAX * T_HD * 2
    };
    fc_buffer *ab[6] = { a.kc, a.vc, a.ks, a.vs, a.kw, a.vw };
    fc_buffer *bb[6] = { b.kc, b.vc, b.ks, b.vs, b.kw, b.vw };
    const char *nm[6] = { "k_ring", "v_ring", "k_sink", "v_sink", "k_win", "v_win" };
    int bad = 0;
    for (int i = 0; i < 6; i++) {
        unsigned char *x = malloc(sizes[i]), *y = malloc(sizes[i]);
        fc_buffer_to_cpu(ab[i], x);
        fc_buffer_to_cpu(bb[i], y);
        if (memcmp(x, y, sizes[i]) != 0) {
            bad++;
            printf("  append pos0=%d %s differs\n", pos0, nm[i]);
        }
        free(x); free(y);
    }
    ring_free(&a); ring_free(&b);
    fc_buffer_free(k_g); fc_buffer_free(v_g);
    free(kn); free(vn);
    return bad;
}

int main(int argc, char **argv)
{
    int R = argc > 1 ? atoi(argv[1]) : 5;
    if (R < 1 || R > FC_ATTN_ROWS_MAX) { fprintf(stderr, "rows 1..%d\n", FC_ATTN_ROWS_MAX); return 2; }
    if (fc_init() != 0) { printf("[SKIP] no CUDA\n"); return 0; }
    fc_cuda_ctx *ctx = fc_default_ctx();
    if (!fc_attn_flash_decode_rows_q4_available_ctx(ctx)) {
        printf("rows kernel unavailable (stale fc_cuda_ptx.h?)\n");
        return 1;
    }

    size_t ring_b = (size_t)T_HEAD_KV * T_MAXSEQ * T_ROWB;
    unsigned char *kc = malloc(ring_b), *vc = malloc(ring_b);
    size_t sink_n = (size_t)T_HEAD_KV * T_SINKMAX * T_HD, win_n = (size_t)T_HEAD_KV * T_WMAX * T_HD;
    uint16_t *ks = malloc(sink_n * 2), *vs = malloc(sink_n * 2);
    uint16_t *kw = malloc(win_n * 2), *vw = malloc(win_n * 2);
    fill_q4_ring(kc, (size_t)T_HEAD_KV * T_MAXSEQ);
    fill_q4_ring(vc, (size_t)T_HEAD_KV * T_MAXSEQ);
    fill_half(ks, sink_n); fill_half(vs, sink_n);
    fill_half(kw, win_n); fill_half(vw, win_n);
    ring_set rs = ring_upload(kc, vc, ks, vs, kw, vw);

    static const int pos_list[] = { 0, 1, 3, 100, 126, 127, 128, 130, 252, 254, 255,
                                    256, 300, 509, 510, 511, 1020, 1022, 1023, 1024,
                                    2047, 4093, 4095, 8187, 8189, 8191, 8192, 8700,
                                    9213, 9215, 12000, 16379 };
    int n_pos = (int)(sizeof(pos_list) / sizeof(pos_list[0]));
    const char *rpb = getenv("FC_ATTN_ROWS_PER_BLOCK");
    int dev_ok = fc_attn_flash_decode_rows_q4_dev_available_ctx(ctx);
    if (!dev_ok)
        printf("device-position rows kernel unavailable: NOT checked\n");
    printf("test_attn_decode_rows: rows=%d rows_per_block=%s positions=%d\n", R,
           rpb ? rpb : "(default)", n_pos);
    int bad = 0, red_bad = 0, app_bad = 0, det_bad = 0;
    for (int i = 0; i < n_pos; i++) {
        int pos0 = pos_list[i];
        if (pos0 + R > T_MAXSEQ) continue;
        bad += check_attention(ctx, &rs, pos0, R, 0, 0, 0);
        bad += check_attention(ctx, &rs, pos0, R, 1, 0, 0);
        if (dev_ok) {
            bad += check_attention(ctx, &rs, pos0, R, 0, 0, 1);
            bad += check_attention(ctx, &rs, pos0, R, 1, 0, 1);
        }

        red_bad += check_attention(ctx, &rs, pos0, R, 0, 1, 0);
        app_bad += check_append(ctx, kc, ks, kw, pos0, R);
        det_bad += check_determinism(ctx, &rs, pos0, 0);
        det_bad += check_determinism(ctx, &rs, pos0, 1);
    }
    printf("slots reference nondeterministic: %d\n", det_bad);
    printf("attention rows mismatched: %d | append mismatched: %d | red rows "
           "mismatched: %d\n", bad, app_bad, red_bad);
    int exact = bad == 0 && app_bad == 0;
    printf("RESULT exact=%s red=%s\n", exact ? "PASS" : "FAIL",
           red_bad ? "FAIL" : "PASS");
    ring_free(&rs);
    free(kc); free(vc); free(ks); free(vs); free(kw); free(vw);
    fc_free();
    return exact && red_bad ? 0 : 1;
}
