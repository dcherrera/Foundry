/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_attn_decode_kvf.c
 * @brief Bit-exact gate for the fewer-instruction decode slots attention
 *        (fc_attn_flash_decode_slots_q4_kvf) against the kvs kernel.
 *
 * Random Q4 ring + fp16 sink/window shadows at the Qwen3.6 shape (16 q heads,
 * 2 kv heads, head_dim 256, sink 4, win 128). For nslots 1..4 (users in
 * scattered ring rows, each at its own position) and positions that cross the
 * split-plan edges (one split below 256, the 32-split cap at 8192, split_len
 * steps past it), tile tails (seq % 16 != 0), fp16-shadow tiles and
 * the window edge, runs one decode slots call with the kvs kernel and one with
 * the kvf kernel and memcmps every output float, ungated and gated.
 * RED arm (must FAIL): the kvs output against a kvs call whose q has one
 * element nudged by one ulp — proves the comparison sees a 1-ulp change.
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/test_attn_decode_kvf cuda/tests/test_attn_decode_kvf.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run:  CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/test_attn_decode_kvf
 * Last line: "RESULT exact=PASS|FAIL red=FAIL|PASS".
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
#define T_MAXSEQ   20480
#define T_SINK     4
#define T_WIN      128
#define T_WMAX     256
#define T_SINKMAX  8
#define T_KPS      256
#define T_CAP      32
#define T_ROWB     ((T_HD / 32) * 18)
#define T_ROWS     6            /* ring rows carved; users sit in rows 5,1,3,0 */
#define T_SLOTS    4

static uint32_t rng_state = 0x2545F491u;
static uint32_t rng(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* Random q4 bytes with every block's fp16 scale small and finite. */
static void fill_ring(unsigned char *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (unsigned char)rng();
    for (size_t i = 0; i + 1 < n; i += 18) {
        p[i] = (unsigned char)(rng() & 0xFF);
        p[i + 1] = (unsigned char)(0x1C + (rng() & 3));
    }
}

static void fill_half(uint16_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint16_t)(0x3000 + (rng() & 0x07FF)) | (uint16_t)((rng() & 1) << 15);
}

static int launch_splits_for(const int *pos, int n)
{
    int lsp = 1;
    for (int u = 0; u < n; u++) {
        int seq = pos[u] + 1, ns = (seq + T_KPS - 1) / T_KPS;
        if (ns > T_CAP) ns = T_CAP;
        int sl = ((seq + ns - 1) / ns + 15) & ~15, nsp = (seq + sl - 1) / sl;
        if (nsp > lsp) lsp = nsp;
    }
    return lsp;
}

typedef struct {
    fc_buffer *kc, *vc, *ks, *vs, *kw, *vw, *q, *z, *out, *m, *l, *acc, *rows;
} t_bufs;

/* One slots call (gated when gated != 0); returns the output on the host. */
static int run_call(t_bufs *b, const int *pos, int n, int gated, float *host_out)
{
    fc_buffer *pos_g = fc_buffer_from_ptr((void *)pos, (size_t)n * 4);
    int lsp = launch_splits_for(pos, n), rc;
    const float scale = 1.0f / 16.0f;
    if (gated)
        rc = fc_attn_flash_decode_slots_q4_gated_ctx(
            fc_default_ctx(), b->q, b->kc, b->vc, b->ks, b->vs, b->kw, b->vw, b->m, b->l,
            b->acc, b->out, T_HEAD, T_HEAD_KV, T_HD, T_MAXSEQ, pos_g, b->rows, n, lsp,
            T_KPS, T_CAP, T_SINK, T_WIN, 0, 0, scale, b->z);
    else
        rc = fc_attn_flash_decode_slots_q4_ctx(
            fc_default_ctx(), b->q, b->kc, b->vc, b->ks, b->vs, b->kw, b->vw, b->m, b->l,
            b->acc, b->out, T_HEAD, T_HEAD_KV, T_HD, T_MAXSEQ, pos_g, b->rows, n, lsp,
            T_KPS, T_CAP, T_SINK, T_WIN, 0, 0, scale);
    fc_sync();
    fc_buffer_to_cpu_n(b->out, host_out, (size_t)n * T_HEAD * T_HD * 4);
    fc_buffer_free(pos_g);
    return rc;
}

static int count_diff(const float *a, const float *b, size_t n)
{
    int nd = 0;
    for (size_t i = 0; i < n; i++)
        nd += memcmp(&a[i], &b[i], 4) != 0;
    return nd;
}

int main(void)
{
    static const int pos_list[] = { 0, 15, 16, 200, 255, 256, 1000, 4095, 8190, 8191,
                                    8192, 9000, 12345, 16383, 16400, 20000 };
    const int n_pos = (int)(sizeof(pos_list) / sizeof(pos_list[0]));
    const int row_map[T_SLOTS] = { 5, 1, 3, 0 };

    if (fc_init() != 0) { printf("[SKIP] no CUDA\n"); return 0; }
    if (!fc_attn_dec_kvf_available_ctx(fc_default_ctx())) {
        printf("kvf kernel not in this PTX header: nothing to compare\n");
        printf("RESULT exact=FAIL red=FAIL\n");
        return 1;
    }
    size_t ring_b = (size_t)T_ROWS * T_HEAD_KV * T_MAXSEQ * T_ROWB;
    size_t sink_n = (size_t)T_ROWS * T_HEAD_KV * T_SINKMAX * T_HD;
    size_t win_n = (size_t)T_ROWS * T_HEAD_KV * T_WMAX * T_HD;
    size_t q_n = (size_t)T_SLOTS * T_HEAD * T_HD;
    size_t np = (size_t)T_SLOTS * T_HEAD * T_CAP;
    unsigned char *ring = malloc(ring_b);
    uint16_t *half = malloc(win_n * 2);
    float *q = malloc(q_n * 4), *z = malloc(q_n * 4);
    float *a = malloc(q_n * 4), *bb = malloc(q_n * 4);
    t_bufs b = { 0 };
    fill_ring(ring, ring_b);
    b.kc = fc_buffer_from_ptr(ring, ring_b);
    fill_ring(ring, ring_b);
    b.vc = fc_buffer_from_ptr(ring, ring_b);
    fill_half(half, sink_n); b.ks = fc_buffer_from_ptr(half, sink_n * 2);
    fill_half(half, sink_n); b.vs = fc_buffer_from_ptr(half, sink_n * 2);
    fill_half(half, win_n); b.kw = fc_buffer_from_ptr(half, win_n * 2);
    fill_half(half, win_n); b.vw = fc_buffer_from_ptr(half, win_n * 2);
    for (size_t i = 0; i < q_n; i++) {
        q[i] = (float)((int)(rng() % 2001) - 1000) / 400.0f;
        z[i] = (float)((int)(rng() % 2001) - 1000) / 300.0f;
    }
    b.q = fc_buffer_from_ptr(q, q_n * 4);
    b.z = fc_buffer_from_ptr(z, q_n * 4);
    b.out = fc_buffer_alloc(q_n * 4);
    b.m = fc_buffer_alloc(np * 4);
    b.l = fc_buffer_alloc(np * 4);
    b.acc = fc_buffer_alloc(np * T_HD * 4);
    b.rows = fc_buffer_from_ptr((void *)row_map, sizeof(row_map));

    int bad = 0, calls = 0, fail_rc = 0;
    for (int n = 1; n <= T_SLOTS; n++) {
        for (int pi = 0; pi < n_pos; pi++) {
            int pos[T_SLOTS];
            for (int u = 0; u < n; u++) {      /* users at different depths */
                int p = pos_list[(pi + 3 * u) % n_pos];
                pos[u] = p < T_MAXSEQ - 1 ? p : T_MAXSEQ - 2;
            }
            for (int gated = 0; gated < 2; gated++) {
                fc_attn_dec_kvf_set(0);
                fail_rc |= run_call(&b, pos, n, gated, a);
                fc_attn_dec_kvf_set(1);
                fail_rc |= run_call(&b, pos, n, gated, bb);
                int nd = count_diff(a, bb, (size_t)n * T_HEAD * T_HD);
                calls++;
                if (nd) {
                    bad++;
                    printf("  DIFF n=%d pos0=%d gated=%d: %d floats\n", n, pos[0], gated, nd);
                }
            }
        }
    }
    /* RED: nudge one q element by an ulp; kvs vs kvs must now differ. */
    fc_attn_dec_kvf_set(0);
    int pos_red[1] = { 9000 };
    run_call(&b, pos_red, 1, 0, a);
    uint32_t bits;
    memcpy(&bits, &q[7], 4);
    bits += 1;
    memcpy(&q[7], &bits, 4);
    fc_buffer_free(b.q);
    b.q = fc_buffer_from_ptr(q, q_n * 4);
    run_call(&b, pos_red, 1, 0, bb);
    int red_diff = count_diff(a, bb, (size_t)T_HEAD * T_HD);

    int exact = bad == 0 && fail_rc == 0;
    printf("%d call pairs, %d differ, launch rc %d; red arm %d floats differ\n", calls, bad,
           fail_rc, red_diff);
    printf("RESULT exact=%s red=%s\n", exact ? "PASS" : "FAIL", red_diff ? "FAIL" : "PASS");
    fc_free();
    return exact && red_diff ? 0 : 1;
}
