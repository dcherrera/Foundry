/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * bench_attn_split.c — decode wall time vs split-K workgroup count.
 *
 * This is the measurement FM_ATTN_DECODE_TARGET_WORKGROUPS is set from, kept
 * runnable so the default can be re-derived on another machine instead of
 * trusted. It sweeps the split plan at three context lengths and prints
 * milliseconds per decoded token for the decode + reduce pair.
 *
 * Not a correctness test — it asserts nothing. Correctness lives in
 * test_metal_attn.c. Exits 77 with no Metal device.
 */

#include "metal/kernels/fm_kernel_attn.h"

#include <mach/mach_time.h>
#include <stdio.h>

#define BENCH_HEAD_DIM   256
#define BENCH_N_HEAD      16
#define BENCH_N_HEAD_KV    2
#define BENCH_KV_GROUPS   (BENCH_N_HEAD / BENCH_N_HEAD_KV)
#define BENCH_MAX_SEQ   4096
#define BENCH_WARMUP       5
#define BENCH_ITERS       50

static double bench_now_ms(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1e6;
}

/* Deterministic, non-degenerate fill. The values only have to be finite and
 * varied — this measures dispatch shape, not numerics. */
static void bench_fill(fm_buffer *buf, size_t n, int period, float step) {
    float *p = (float *)fm_buffer_data(buf);
    for (size_t i = 0; i < n; i++) p[i] = (float)(i % (size_t)period) * step;
}

static double bench_one_plan(fm_buffer *q, fm_buffer *k, fm_buffer *v,
                             fm_buffer *out, int seq_len, int n_splits,
                             int split_len, float scale) {
    size_t m_floats = 0, acc_floats = 0;
    fm_attn_decode_scratch_floats(BENCH_N_HEAD, BENCH_HEAD_DIM, n_splits,
                                  &m_floats, &acc_floats);
    fm_buffer *m = fm_buffer_alloc(m_floats * sizeof(float));
    fm_buffer *l = fm_buffer_alloc(m_floats * sizeof(float));
    fm_buffer *acc = fm_buffer_alloc(acc_floats * sizeof(float));
    if (!m || !l || !acc) return -1.0;

    double elapsed = 0.0;
    for (int i = 0; i < BENCH_WARMUP + BENCH_ITERS; i++) {
        if (i == BENCH_WARMUP) elapsed = bench_now_ms();
        fm_attn_flash_decode(q, k, v, m, l, acc, BENCH_KV_GROUPS,
                             BENCH_N_HEAD_KV, BENCH_HEAD_DIM, BENCH_MAX_SEQ,
                             seq_len, n_splits, split_len, scale);
        fm_attn_flash_decode_reduce(m, l, acc, out, BENCH_N_HEAD,
                                    BENCH_HEAD_DIM, n_splits);
        fm_sync();
    }
    elapsed = (bench_now_ms() - elapsed) / BENCH_ITERS;

    fm_buffer_free(m);
    fm_buffer_free(l);
    fm_buffer_free(acc);
    return elapsed;
}

int main(void) {
    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "Metal unavailable — skipping.\n");
        return 77;
    }

    size_t cache_n = (size_t)BENCH_N_HEAD_KV * BENCH_MAX_SEQ * BENCH_HEAD_DIM;
    size_t row_n = (size_t)BENCH_N_HEAD * BENCH_HEAD_DIM;
    fm_buffer *k = fm_buffer_alloc(cache_n * sizeof(float));
    fm_buffer *v = fm_buffer_alloc(cache_n * sizeof(float));
    fm_buffer *q = fm_buffer_alloc(row_n * sizeof(float));
    fm_buffer *out = fm_buffer_alloc(row_n * sizeof(float));
    if (!k || !v || !q || !out) return 1;
    bench_fill(k, cache_n, 17, 0.01f);
    bench_fill(v, cache_n, 13, 0.02f);
    bench_fill(q, row_n, 7, 0.03f);

    const int seq_lens[] = { 128, 512, 4096 };
    const int workgroups[] = { 2, 4, 8, 16, 32, 64 };
    const float scale = 1.0f / 16.0f;   /* 1/sqrt(256) */

    printf("decode  n_head=%d n_head_kv=%d head_dim=%d  (ms per token)\n",
           BENCH_N_HEAD, BENCH_N_HEAD_KV, BENCH_HEAD_DIM);
    for (int s = 0; s < 3; s++) {
        printf("  seq %5d:\n", seq_lens[s]);
        for (int w = 0; w < 6; w++) {
            int split_len = 0;
            int n_splits = fm_attn_decode_plan(seq_lens[s], BENCH_N_HEAD_KV,
                                               workgroups[w], &split_len);
            double ms = bench_one_plan(q, k, v, out, seq_lens[s], n_splits,
                                       split_len, scale);
            printf("    %2d workgroups (%2d splits x %5d keys)  %7.3f\n",
                   n_splits * BENCH_N_HEAD_KV, n_splits, split_len, ms);
        }
    }

    fm_buffer_free(k);
    fm_buffer_free(v);
    fm_buffer_free(q);
    fm_buffer_free(out);
    fm_free();
    return 0;
}
