/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_attn_decode_rows.c
 * @brief Kernel-level timing of the speculative-verify attention: one R-row
 *        fc_attn_flash_decode_rows_q4_ctx call against the decode slots kernel
 *        (one plain decode step's attention) at the Qwen3.6 shape, per position.
 *
 * Prints, per position, ms per call (mean of REPS back-to-back calls, synced
 * once) for the slots kernel (1 row), the rows kernel at R = 1..5 (host plan), its
 * device-position form (dev, FC_ATTN_ROWS_PSM picks its softmax), and the
 * per-position slope of each between consecutive positions. One layer's call;
 * the model runs 10 such layers per step.
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/bench_attn_decode_rows cuda/tests/bench_attn_decode_rows.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run:  CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/bench_attn_decode_rows [reps=200]
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
#include <time.h>

#define B_HEAD    16
#define B_HEAD_KV 2
#define B_HD      256
#define B_MAXSEQ  40960
#define B_SINK    4
#define B_WIN     128
#define B_KPS     256
#define B_CAP     32
#define B_ROWB    ((B_HD / 32) * 18)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static void fill_bytes(unsigned char *p, size_t n)
{
    uint32_t s = 0x12345678u;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        p[i] = (unsigned char)s;
    }
    /* Keep every q4 block's fp16 scale small and finite. */
    for (size_t i = 0; i + 1 < n; i += 18) {
        p[i] = 0x00;
        p[i + 1] = 0x20;
    }
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 200;
    static const int pos_list[] = { 1024, 4096, 8192, 14336, 24576, 32768 };
    const int n_pos = (int)(sizeof(pos_list) / sizeof(pos_list[0]));

    if (fc_init() != 0) { printf("[SKIP] no CUDA\n"); return 0; }
    fc_cuda_ctx *ctx = fc_default_ctx();
    size_t ring_b = (size_t)B_HEAD_KV * B_MAXSEQ * B_ROWB;
    size_t sink_b = (size_t)B_HEAD_KV * 8 * B_HD * 2, win_b = (size_t)B_HEAD_KV * 256 * B_HD * 2;
    unsigned char *host = malloc(ring_b);
    fill_bytes(host, ring_b);
    fc_buffer *kc = fc_buffer_from_ptr(host, ring_b), *vc = fc_buffer_from_ptr(host, ring_b);
    memset(host, 0x11, win_b);
    fc_buffer *ks = fc_buffer_from_ptr(host, sink_b), *vs = fc_buffer_from_ptr(host, sink_b);
    fc_buffer *kw = fc_buffer_from_ptr(host, win_b), *vw = fc_buffer_from_ptr(host, win_b);
    const size_t row_f = (size_t)B_HEAD * B_HD;
    float *q = malloc(row_f * 5 * 4);
    for (size_t i = 0; i < row_f * 5; i++) q[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    fc_buffer *q_g = fc_buffer_from_ptr(q, row_f * 5 * 4), *out = fc_buffer_alloc(row_f * 5 * 4);
    size_t np = (size_t)5 * B_HEAD * B_CAP;
    fc_buffer *m = fc_buffer_alloc(np * 4), *l = fc_buffer_alloc(np * 4);
    fc_buffer *acc = fc_buffer_alloc(np * B_HD * 4);
    float scale = 1.0f / 16.0f;
    double ms[16][8];

    printf("bench_attn_decode_rows: reps=%d (ms per call, one layer)\n", reps);
    printf("   pos    slots1    rows1    rows2    rows3    rows4    rows5 |    dev1     dev2     dev3     dev4     dev5\n");
    for (int pi = 0; pi < n_pos; pi++) {
        int pos = pos_list[pi];
        int32_t p32 = pos;
        fc_buffer *pos_g = fc_buffer_from_ptr(&p32, 4);
        int seq = pos + 1, ns = (seq + B_KPS - 1) / B_KPS;
        if (ns > B_CAP) ns = B_CAP;
        int sl = ((seq + ns - 1) / ns + 15) & ~15, launch = (seq + sl - 1) / sl;
        int32_t pz = pos;
        fc_buffer *p0_g = fc_buffer_from_ptr(&pz, 4);
        for (int v = 0; v <= 10; v++) {
            int rows = v <= 5 ? v : v - 5;
            int lsp = 1;
            for (int r = 0; r < rows; r++) {
                int sq = pos + r + 1, n2 = (sq + B_KPS - 1) / B_KPS;
                if (n2 > B_CAP) n2 = B_CAP;
                int s2 = ((sq + n2 - 1) / n2 + 15) & ~15;
                if ((sq + s2 - 1) / s2 > lsp) lsp = (sq + s2 - 1) / s2;
            }
            for (int w = 0; w < 2; w++) {     /* w = 0 warm-up */
                double t0 = now_ms();
                for (int i = 0; i < (w ? reps : 5); i++) {
                    if (v == 0)
                        fc_attn_flash_decode_slots_q4_ctx(ctx, q_g, kc, vc, ks, vs, kw, vw, m, l,
                                                          acc, out, B_HEAD, B_HEAD_KV, B_HD,
                                                          B_MAXSEQ, pos_g, NULL, 1, launch, B_KPS,
                                                          B_CAP, B_SINK, B_WIN, 0, 0, scale);
                    else if (v <= 5)
                        fc_attn_flash_decode_rows_q4_ctx(ctx, q_g, kc, vc, ks, vs, kw, vw, m, l,
                                                         acc, NULL, out, B_HEAD, B_HEAD_KV, B_HD,
                                                         B_MAXSEQ, pos, v, B_KPS, B_CAP, B_SINK,
                                                         B_WIN, 0, 0, scale);
                    else
                        fc_attn_flash_decode_rows_q4_dev_ctx(
                            ctx, q_g, kc, vc, ks, vs, kw, vw, m, l, acc, NULL, out, B_HEAD,
                            B_HEAD_KV, B_HD, B_MAXSEQ, p0_g, rows, lsp, B_KPS, B_CAP, B_SINK,
                            B_WIN, scale);
                }
                fc_sync();
                ms[v][pi] = (now_ms() - t0) / (w ? reps : 5);
            }
        }
        fc_buffer_free(pos_g);
        fc_buffer_free(p0_g);
        printf("%6d ", pos);
        for (int v = 0; v <= 10; v++)
            printf(" %8.4f%s", ms[v][pi], v == 5 ? " |" : "");
        printf("\n");
        fflush(stdout);
    }
    printf("slope us/pos (1024 -> 32768):");
    const char *nm[11] = { "slots1", "rows1", "rows2", "rows3", "rows4", "rows5", "dev1", "dev2",
                           "dev3", "dev4", "dev5" };
    for (int v = 0; v <= 10; v++)
        printf(" %s %.4f", nm[v], 1e3 * (ms[v][n_pos - 1] - ms[v][0]) / (32768 - 1024));
    printf("\nextra-row slope us/pos (R rows - rows1)/(R-1):");
    for (int v = 2; v <= 5; v++)
        printf(" rows R%d %.4f", v,
               1e3 * ((ms[v][n_pos - 1] - ms[v][0]) - (ms[1][n_pos - 1] - ms[1][0])) /
                   (32768 - 1024) / (v - 1));
    for (int v = 7; v <= 10; v++)
        printf(" dev R%d %.4f", v - 5,
               1e3 * ((ms[v][n_pos - 1] - ms[v][0]) - (ms[6][n_pos - 1] - ms[6][0])) /
                   (32768 - 1024) / (v - 6));
    printf("\n");
    fc_free();
    return 0;
}
