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
 * device-position form (dev, FC_ATTN_ROWS_PSM picks its softmax), the R rows run
 * through the decode slots call as R slots of one ring (kvf: rows map 0, positions
 * pos..pos+R-1), and the per-position slope of each between consecutive positions.
 * One layer's call; the model runs 10 such layers per step. Each kvf R output is
 * byte-compared with the dev R output (same rows, same positions).
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/bench_attn_decode_rows cuda/tests/bench_attn_decode_rows.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run:  CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/bench_attn_decode_rows [reps=200]
 * Env:  FC_BENCH_TP=1 times the TP2 per-die shape (8 heads, 1 kv head).
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

#define B_HD      256
#define B_MAXSEQ  40960
#define B_SINK    4
#define B_WIN     128
#define B_KPS     256
#define B_CAP     32
#define B_ROWB    ((B_HD / 32) * 18)
#define B_ARMS    16

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
    static const int pos_list[] = { 1024, 8192, 14336, 19456, 24576, 32768 };
    const int n_pos = (int)(sizeof(pos_list) / sizeof(pos_list[0]));
    const char *tp = getenv("FC_BENCH_TP");
    const int n_head = tp && tp[0] == '1' ? 8 : 16;
    const int n_head_kv = tp && tp[0] == '1' ? 1 : 2;

    if (fc_init() != 0) { printf("[SKIP] no CUDA\n"); return 0; }
    fc_cuda_ctx *ctx = fc_default_ctx();
    size_t ring_b = (size_t)n_head_kv * B_MAXSEQ * B_ROWB;
    size_t sink_b = (size_t)n_head_kv * 8 * B_HD * 2, win_b = (size_t)n_head_kv * 256 * B_HD * 2;
    unsigned char *host = malloc(ring_b);
    fill_bytes(host, ring_b);
    fc_buffer *kc = fc_buffer_from_ptr(host, ring_b), *vc = fc_buffer_from_ptr(host, ring_b);
    memset(host, 0x11, win_b);
    fc_buffer *ks = fc_buffer_from_ptr(host, sink_b), *vs = fc_buffer_from_ptr(host, sink_b);
    fc_buffer *kw = fc_buffer_from_ptr(host, win_b), *vw = fc_buffer_from_ptr(host, win_b);
    const size_t row_f = (size_t)n_head * B_HD;
    float *q = malloc(row_f * 5 * 4);
    for (size_t i = 0; i < row_f * 5; i++) q[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    fc_buffer *q_g = fc_buffer_from_ptr(q, row_f * 5 * 4), *out = fc_buffer_alloc(row_f * 5 * 4);
    size_t np = (size_t)5 * n_head * B_CAP;
    fc_buffer *m = fc_buffer_alloc(np * 4), *l = fc_buffer_alloc(np * 4);
    fc_buffer *acc = fc_buffer_alloc(np * B_HD * 4);
    float *out_dev = malloc(row_f * 5 * 4), *out_kvf = malloc(row_f * 5 * 4);
    int32_t rows0[5] = { 0, 0, 0, 0, 0 };
    fc_buffer *rows_g = fc_buffer_from_ptr(rows0, sizeof(rows0));
    float scale = 1.0f / 16.0f;
    double ms[B_ARMS][8];
    int mismatched = 0;

    printf("bench_attn_decode_rows: reps=%d heads %d/%d (ms per call, one layer)\n", reps, n_head,
           n_head_kv);
    printf("   pos    slots1    rows1    rows2    rows3    rows4    rows5 |    dev1     dev2     dev3"
           "     dev4     dev5 |    kvf1     kvf2     kvf3     kvf4     kvf5\n");
    for (int pi = 0; pi < n_pos; pi++) {
        int pos = pos_list[pi];
        int32_t p32 = pos;
        fc_buffer *pos_g = fc_buffer_from_ptr(&p32, 4);
        int seq = pos + 1, ns = (seq + B_KPS - 1) / B_KPS;
        if (ns > B_CAP) ns = B_CAP;
        int sl = ((seq + ns - 1) / ns + 15) & ~15, launch = (seq + sl - 1) / sl;
        int32_t pz = pos;
        fc_buffer *p0_g = fc_buffer_from_ptr(&pz, 4);
        int32_t pr[5];
        for (int r = 0; r < 5; r++) pr[r] = pos + r;
        fc_buffer *pr_g = fc_buffer_from_ptr(pr, sizeof(pr));
        for (int v = 0; v < B_ARMS; v++) {
            int rows = v == 0 ? 1 : (v - 1) % 5 + 1;
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
                                                          acc, out, n_head, n_head_kv, B_HD,
                                                          B_MAXSEQ, pos_g, NULL, 1, launch, B_KPS,
                                                          B_CAP, B_SINK, B_WIN, 0, 0, scale);
                    else if (v <= 5)
                        fc_attn_flash_decode_rows_q4_ctx(ctx, q_g, kc, vc, ks, vs, kw, vw, m, l,
                                                         acc, NULL, out, n_head, n_head_kv, B_HD,
                                                         B_MAXSEQ, pos, rows, B_KPS, B_CAP, B_SINK,
                                                         B_WIN, 0, 0, scale);
                    else if (v <= 10)
                        fc_attn_flash_decode_rows_q4_dev_ctx(
                            ctx, q_g, kc, vc, ks, vs, kw, vw, m, l, acc, NULL, out, n_head,
                            n_head_kv, B_HD, B_MAXSEQ, p0_g, rows, lsp, B_KPS, B_CAP, B_SINK,
                            B_WIN, scale);
                    else
                        fc_attn_flash_decode_slots_q4_ctx(ctx, q_g, kc, vc, ks, vs, kw, vw, m, l,
                                                          acc, out, n_head, n_head_kv, B_HD,
                                                          B_MAXSEQ, pr_g, rows_g, rows, lsp,
                                                          B_KPS, B_CAP, B_SINK, B_WIN, 0, 0,
                                                          scale);
                }
                fc_sync();
                ms[v][pi] = (now_ms() - t0) / (w ? reps : 5);
            }
            /* dev5 leaves rows pos..pos+4 in out; a row's bytes do not depend on R, so kvf R
             * must reproduce the first R of them. */
            if (v == 10)
                fc_buffer_to_cpu(out, out_dev);
            if (v >= 11) {
                fc_buffer_to_cpu(out, out_kvf);
                if (memcmp(out_dev, out_kvf, (size_t)rows * row_f * 4) != 0) {
                    printf("MISMATCH pos %d rows %d (kvf vs dev)\n", pos, rows);
                    mismatched++;
                }
            }
        }
        fc_buffer_free(pos_g);
        fc_buffer_free(p0_g);
        fc_buffer_free(pr_g);
        printf("%6d ", pos);
        for (int v = 0; v < B_ARMS; v++)
            printf(" %8.4f%s", ms[v][pi], v == 5 || v == 10 ? " |" : "");
        printf("\n");
        fflush(stdout);
    }
    printf("slope us/pos (%d -> %d):", pos_list[0], pos_list[n_pos - 1]);
    const char *nm[B_ARMS] = { "slots1", "rows1", "rows2", "rows3", "rows4", "rows5", "dev1",
                               "dev2", "dev3", "dev4", "dev5", "kvf1", "kvf2", "kvf3", "kvf4",
                               "kvf5" };
    for (int v = 0; v < B_ARMS; v++)
        printf(" %s %.4f", nm[v],
               1e3 * (ms[v][n_pos - 1] - ms[v][0]) / (pos_list[n_pos - 1] - pos_list[0]));
    printf("\nkvf vs dev (bytes): %s, %d mismatched\n", mismatched ? "FAIL" : "PASS", mismatched);
    fc_free();
    free(out_dev);
    free(out_kvf);
    free(q);
    free(host);
    return mismatched ? 1 : 0;
}
