/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_attn_decode_kvf.c
 * @brief Kernel-level timing of the decode slots attention at long context:
 *        the kvs kernel vs the fewer-instruction kvf kernel, one layer's call, for
 *        nslots users at the same depth (Qwen3.6 shape, gated combine).
 *
 * Prints ms per call (mean of REPS back-to-back calls, synced once) and the KV
 * read rate it implies. The model runs 10 such layers per decode step.
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/bench_attn_decode_kvf cuda/tests/bench_attn_decode_kvf.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run:  CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/bench_attn_decode_kvf [reps=50]
 */
#include "cuda/fc_cuda.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define B_HEAD    16
#define B_HEAD_KV 2
#define B_HD      256
#define B_MAXSEQ  131072
#define B_SINK    4
#define B_WIN     128
#define B_KPS     256
#define B_CAP     32
#define B_ROWB    ((B_HD / 32) * 18)
#define B_SLOTS   4

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
    for (size_t i = 0; i + 1 < n; i += 18) {
        p[i] = 0x00;
        p[i + 1] = 0x20;
    }
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 50;
    static const int pos_list[] = { 8191, 32767, 65535, 131071 - 1 };
    static const int n_list[] = { 1, 2, 4 };

    if (fc_init() != 0) { printf("[SKIP] no CUDA\n"); return 0; }
    fc_cuda_ctx *ctx = fc_default_ctx();
    if (!fc_attn_dec_kvf_available_ctx(ctx)) {
        printf("kvf kernel not in this PTX header\n");
        return 1;
    }
    size_t ring_b = (size_t)B_SLOTS * B_HEAD_KV * B_MAXSEQ * B_ROWB;
    size_t sink_b = (size_t)B_SLOTS * B_HEAD_KV * 8 * B_HD * 2;
    size_t win_b = (size_t)B_SLOTS * B_HEAD_KV * 256 * B_HD * 2;
    unsigned char *host = malloc(ring_b);
    fill_bytes(host, ring_b);
    fc_buffer *kc = fc_buffer_from_ptr(host, ring_b), *vc = fc_buffer_from_ptr(host, ring_b);
    memset(host, 0x11, win_b);
    fc_buffer *ks = fc_buffer_from_ptr(host, sink_b), *vs = fc_buffer_from_ptr(host, sink_b);
    fc_buffer *kw = fc_buffer_from_ptr(host, win_b), *vw = fc_buffer_from_ptr(host, win_b);
    size_t q_n = (size_t)B_SLOTS * B_HEAD * B_HD;
    float *q = malloc(q_n * 4);
    for (size_t i = 0; i < q_n; i++)
        q[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    fc_buffer *q_g = fc_buffer_from_ptr(q, q_n * 4), *z_g = fc_buffer_from_ptr(q, q_n * 4);
    fc_buffer *out = fc_buffer_alloc(q_n * 4);
    size_t np = (size_t)B_SLOTS * B_HEAD * B_CAP;
    fc_buffer *m = fc_buffer_alloc(np * 4), *l = fc_buffer_alloc(np * 4);
    fc_buffer *acc = fc_buffer_alloc(np * B_HD * 4);

    printf("bench_attn_decode_kvf: reps=%d, ms per call (one layer), KV GB/s\n", reps);
    printf("    pos  n |     kvs  GB/s |     kvf  GB/s | speedup\n");
    for (size_t pi = 0; pi < sizeof(pos_list) / sizeof(pos_list[0]); pi++) {
        for (size_t ni = 0; ni < sizeof(n_list) / sizeof(n_list[0]); ni++) {
            int pos = pos_list[pi], n = n_list[ni];
            int32_t p32[B_SLOTS] = { pos, pos, pos, pos };
            fc_buffer *pos_g = fc_buffer_from_ptr(p32, sizeof(p32));
            int seq = pos + 1, ns = (seq + B_KPS - 1) / B_KPS;
            if (ns > B_CAP) ns = B_CAP;
            int sl = ((seq + ns - 1) / ns + 15) & ~15, launch = (seq + sl - 1) / sl;
            double ms[2];
            for (int v = 0; v < 2; v++) {
                fc_attn_dec_kvf_set(v);
                for (int w = 0; w < 2; w++) {
                    int it = w ? reps : 3;
                    double t0 = now_ms();
                    for (int i = 0; i < it; i++)
                        fc_attn_flash_decode_slots_q4_gated_ctx(
                            ctx, q_g, kc, vc, ks, vs, kw, vw, m, l, acc, out, B_HEAD,
                            B_HEAD_KV, B_HD, B_MAXSEQ, pos_g, NULL, n, launch, B_KPS, B_CAP,
                            B_SINK, B_WIN, 0, 0, 1.0f / 16.0f, z_g);
                    fc_sync();
                    ms[v] = (now_ms() - t0) / it;
                }
            }
            double gb = 2.0 * n * B_HEAD_KV * (double)seq * B_ROWB / 1e9;
            printf("%7d %2d | %7.3f %5.1f | %7.3f %5.1f | %5.2fx\n", pos, n, ms[0],
                   gb / (ms[0] / 1e3), ms[1], gb / (ms[1] / 1e3), ms[0] / ms[1]);
            fflush(stdout);
            fc_buffer_free(pos_g);
        }
    }
    fc_free();
    return 0;
}
