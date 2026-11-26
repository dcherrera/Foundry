/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_attn_decode_splits.c
 * @brief Decode slots attention vs split plan (cap, keys per split) at depth, on the
 *        two-die tensor-parallel shape: one kv head and 8 q heads a die, Q4 KV, gated
 *        combine, the kernel the host picks by default (kvf).
 *
 * The split cap bounds the blocks a call launches: 1 user at the default cap (32) is 32
 * blocks on a 13-SMX die. For each (depth, users) the default plan (cap 32, 256 keys a
 * split) is the reference; every other plan prints ms per call, its speedup, and the
 * largest |out - ref| and |out - ref| / max|ref| (a different plan merges the partials
 * in a different order, so it is not byte-identical).
 *
 * Build (from foundry/):
 *   gcc -O2 -I. -o cuda/tests/bench_attn_decode_splits cuda/tests/bench_attn_decode_splits.c \
 *       cuda/fc_cuda.c -lm -ldl
 * Run:  CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/bench_attn_decode_splits [reps=50]
 */
#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define B_HEAD    8
#define B_HEAD_KV 1
#define B_HD      256
#define B_MAXSEQ  65536
#define B_SINK    4
#define B_WIN     128
#define B_ROWB    ((B_HD / 32) * 18)
#define B_SLOTS   4
#define B_CAP_MAX 256

typedef struct { int cap, kps; } split_plan;

static const split_plan plans[] = {
    { 32, 256 }, { 64, 256 }, { 128, 256 }, { 256, 256 },
    { 64, 128 }, { 128, 128 }, { 128, 64 }, { 256, 64 },
};

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

/* Random q4 bytes with every block scale set to a sane fp16 (0x2000). */
static void fill_q4(unsigned char *p, size_t n)
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

/* The host plan of qwen35moe_forward_cuda.c (== fc_attn_dec_plan): grid width. */
static int launch_splits(int seq, int kps, int cap)
{
    int ns = (seq + kps - 1) / kps;
    if (ns < 1) ns = 1;
    if (ns > cap) ns = cap;
    int sl = ((seq + ns - 1) / ns + 15) & ~15;
    return (seq + sl - 1) / sl;
}

int main(int argc, char **argv)
{
    int reps = argc > 1 ? atoi(argv[1]) : 50;
    static const int pos_list[] = { 8191, 14335, 32767, 65535 - 1 };
    static const int n_list[] = { 1, 4 };
    const int n_plans = (int)(sizeof(plans) / sizeof(plans[0]));

    if (fc_init() != 0) { printf("[SKIP] no CUDA\n"); return 0; }
    fc_cuda_ctx *ctx = fc_default_ctx();
    size_t ring_b = (size_t)B_SLOTS * B_HEAD_KV * B_MAXSEQ * B_ROWB;
    size_t sink_b = (size_t)B_SLOTS * B_HEAD_KV * 8 * B_HD * 2;
    size_t win_b = (size_t)B_SLOTS * B_HEAD_KV * 256 * B_HD * 2;
    unsigned char *host = malloc(ring_b);
    if (!host) { printf("host alloc failed\n"); return 1; }
    fill_q4(host, ring_b);
    fc_buffer *kc = fc_buffer_from_ptr(host, ring_b), *vc = fc_buffer_from_ptr(host, ring_b);
    memset(host, 0x11, win_b);
    fc_buffer *ks = fc_buffer_from_ptr(host, sink_b), *vs = fc_buffer_from_ptr(host, sink_b);
    fc_buffer *kw = fc_buffer_from_ptr(host, win_b), *vw = fc_buffer_from_ptr(host, win_b);
    size_t q_n = (size_t)B_SLOTS * B_HEAD * B_HD;
    float *q = malloc(q_n * 4), *ref = malloc(q_n * 4), *got = malloc(q_n * 4);
    if (!q || !ref || !got) { printf("host alloc failed\n"); return 1; }
    for (size_t i = 0; i < q_n; i++)
        q[i] = (float)((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
    fc_buffer *q_g = fc_buffer_from_ptr(q, q_n * 4), *z_g = fc_buffer_from_ptr(q, q_n * 4);
    fc_buffer *out = fc_buffer_alloc(q_n * 4);
    size_t np = (size_t)B_SLOTS * B_HEAD * B_CAP_MAX;
    fc_buffer *m = fc_buffer_alloc(np * 4), *l = fc_buffer_alloc(np * 4);
    fc_buffer *acc = fc_buffer_alloc(np * B_HD * 4);
    if (!kc || !vc || !ks || !vs || !kw || !vw || !q_g || !z_g || !out || !m || !l || !acc) {
        printf("device alloc failed\n");
        return 1;
    }

    printf("bench_attn_decode_splits: TP shape (%d q heads, %d kv head a die), reps=%d\n",
           B_HEAD, B_HEAD_KV, reps);
    printf("    pos  n  cap  kps grid |      ms  speedup | max|d|    rel\n");
    for (size_t pi = 0; pi < sizeof(pos_list) / sizeof(pos_list[0]); pi++) {
        for (size_t ni = 0; ni < sizeof(n_list) / sizeof(n_list[0]); ni++) {
            int pos = pos_list[pi], n = n_list[ni], seq = pos + 1;
            int32_t p32[B_SLOTS] = { pos, pos, pos, pos };
            fc_buffer *pos_g = fc_buffer_from_ptr(p32, sizeof(p32));
            size_t out_n = (size_t)n * B_HEAD * B_HD;
            double ms_ref = 0.0, ref_max = 0.0;
            for (int pl = 0; pl < n_plans; pl++) {
                int cap = plans[pl].cap, kps = plans[pl].kps;
                int grid = launch_splits(seq, kps, cap);
                double ms = 0.0;
                int rc = 0;
                for (int w = 0; w < 2; w++) {
                    int it = w ? reps : 3;
                    double t0 = now_ms();
                    for (int i = 0; i < it && rc == 0; i++)
                        rc = fc_attn_flash_decode_slots_q4_gated_ctx(
                            ctx, q_g, kc, vc, ks, vs, kw, vw, m, l, acc, out, B_HEAD,
                            B_HEAD_KV, B_HD, B_MAXSEQ, pos_g, NULL, n, grid, kps, cap,
                            B_SINK, B_WIN, 0, 0, 1.0f / 16.0f, z_g);
                    fc_sync();
                    ms = (now_ms() - t0) / it;
                }
                if (rc != 0) {
                    printf("%7d %2d %4d %4d %4d | launch failed\n", pos, n, cap, kps, grid);
                    continue;
                }
                fc_buffer_to_cpu(out, pl == 0 ? ref : got);
                double dmax = 0.0;
                if (pl == 0) {
                    ms_ref = ms;
                    for (size_t i = 0; i < out_n; i++)
                        if (fabs(ref[i]) > ref_max) ref_max = fabs(ref[i]);
                } else {
                    for (size_t i = 0; i < out_n; i++)
                        if (fabs((double)got[i] - ref[i]) > dmax)
                            dmax = fabs((double)got[i] - ref[i]);
                }
                printf("%7d %2d %4d %4d %4d | %7.3f %6.2fx | %.2e %.1e\n", pos, n, cap, kps,
                       grid, ms, ms_ref / ms, dmax, ref_max > 0 ? dmax / ref_max : 0.0);
                fflush(stdout);
            }
            fc_buffer_free(pos_g);
        }
    }
    fc_free();
    return 0;
}
