/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_matvec_q80_ab.c
 * @brief Interleaved A/B of two matvec_kquants.ptx builds on the fc_matvec_q80 family.
 *
 * Loads build A and build B in ONE process and, per shape, alternates batches of A and B
 * launches (A-B then B-A each round) so clock and thermal drift cancel. Reports the median
 * per-launch time of each and B's speedup. Every shape first checks that A and B produce
 * bit-identical outputs (memcmp) and exits non-zero if they differ. Shapes are the
 * Qwen3.6-35B-A3B Q8_0 decode matvecs of the batched slots path (Q80_SHAPES=tp2: the live
 * TP2 per-die shapes) at n = 1/2/3/4/5/8/16, each on the kernel fc_cuda.c picks for n
 * (single, nx2, r4c3/4/5/8 row tiles when the build has them).
 * `A.ptx A.ptx` is the null A/B (must read 1.00x).
 *
 * Build (on the bank, from foundry/):
 *   gcc -O2 -I/usr/local/cuda-11.4/include -o cuda/tests/bench_matvec_q80_ab \
 *       cuda/tests/bench_matvec_q80_ab.c -lcuda -lm
 * Run: CUDA_VISIBLE_DEVICES=0 ./cuda/tests/bench_matvec_q80_ab A.ptx[:rpb][:legacy] B.ptx[:rpb][:legacy] [rounds]
 *      (:legacy = a build before the G4 layout; see matvec_q80_ptx_harness.h)
 *      (rpb = rows per block of that build, default 32; origin/main before this kernel = 8)
 *      env Q80_B_BANK8=1 runs build B with 8-byte shared-memory banks;
 *      Q80_ONLY_ROWS=<rows> / Q80_ONLY_N=<n> restrict the shapes (for profiling).
 *      ... --dump-cubin A.ptx out.cubin   writes the driver-JIT cubin (for cuobjdump).
 *
 * @status Working.
 * @issues None
 * @todo None
 */

#include "matvec_q80_ptx_harness.h"

#define AB_MAX_ROUNDS 64

typedef struct {
    const char *name;
    int rows, cols;
} AbShape;

static const AbShape ab_shapes[] = {
    { "ssm_qkv/attn_q", 8192, 2048 },
    { "ssm_gate", 4096, 2048 },
    { "ssm_out/attn_o", 2048, 4096 },
    { "shexp_gu/attn_kv", 512, 2048 },
    { "shexp_down", 2048, 512 },
    { "lm_head", 248320, 2048 },
};

/* Per-die shapes of the live Qwen3.6 TP2 profile (two dies a stage, rows split by
 * tp_rows; Q80_SHAPES=tp2): GDN qkv / attn q+gate half, z half, ssm_out / attn_o half,
 * shexp gate/up / attn k/v / router, shexp down half, the LM head half. */
static const AbShape ab_shapes_tp2[] = {
    { "tp2_qkv/attn_q", 4096, 2048 },
    { "tp2_ssm_z", 2048, 2048 },
    { "tp2_ssm_out/o", 1024, 4096 },
    { "tp2_shexp/kv/rtr", 256, 2048 },
    { "tp2_shexp_down", 1024, 512 },
    { "tp2_lm_head", 124160, 2048 },
};

static int ab_compare_floats(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* Time `launches` back-to-back launches of one build; returns microseconds per launch. */
static float ab_time_batch(const Q80KernelSet *set, CUdeviceptr w, CUdeviceptr x,
                           CUdeviceptr out, const AbShape *s, int stride, int n,
                           int launches, CUevent start, CUevent stop)
{
    float ms = 0.0f;
    Q80_CHECK(cuEventRecord(start, 0));
    for (int i = 0; i < launches; i++)
        q80_launch_production(set, w, x, out, s->rows, s->cols, stride, n, 0);
    Q80_CHECK(cuEventRecord(stop, 0));
    Q80_CHECK(cuEventSynchronize(stop));
    Q80_CHECK(cuEventElapsedTime(&ms, start, stop));
    return ms * 1000.0f / (float)launches;
}

/* Bit-compare A vs B on one shape, then the interleaved timing. Returns 1 on mismatch. */
static int ab_run_shape(const Q80KernelSet *a, const Q80KernelSet *b, const AbShape *s,
                        int n, int rounds)
{
    int stride = 0;
    unsigned char *wh = q80_make_weights(s->rows, s->cols, 0x9E3779B9u ^ (uint32_t)s->rows,
                                         &stride);
    float *xh = q80_make_activations(s->cols, n, 0x85EBCA6Bu ^ (uint32_t)s->cols);
    const size_t out_bytes = (size_t)s->rows * (size_t)n * sizeof(float);
    float *ya = malloc(out_bytes), *yb = malloc(out_bytes);
    CUdeviceptr w, wb, x, out;
    CUevent start, stop;
    float ta[AB_MAX_ROUNDS], tb[AB_MAX_ROUNDS];

    Q80_CHECK(cuMemAlloc(&w, (size_t)stride * (size_t)s->rows));
    Q80_CHECK(cuMemAlloc(&x, (size_t)s->cols * (size_t)n * sizeof(float)));
    Q80_CHECK(cuMemAlloc(&out, out_bytes));
    Q80_CHECK(cuMemAlloc(&wb, (size_t)stride * (size_t)s->rows));
    unsigned char *wd = q80_device_weights(a, wh, s->rows, s->cols, stride);
    Q80_CHECK(cuMemcpyHtoD(w, wd, (size_t)stride * (size_t)s->rows));
    free(wd);
    wd = q80_device_weights(b, wh, s->rows, s->cols, stride);   /* B's layout may differ */
    Q80_CHECK(cuMemcpyHtoD(wb, wd, (size_t)stride * (size_t)s->rows));
    free(wd);
    Q80_CHECK(cuMemcpyHtoD(x, xh, (size_t)s->cols * (size_t)n * sizeof(float)));
    Q80_CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    Q80_CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    const int tile = q80_launch_production(a, w, x, out, s->rows, s->cols, stride, n, 0);
    Q80_CHECK(cuMemcpyDtoH(ya, out, out_bytes));
    Q80_CHECK(cuMemsetD8(out, 0xFF, out_bytes));
    q80_launch_production(b, wb, x, out, s->rows, s->cols, stride, n, 0);
    Q80_CHECK(cuMemcpyDtoH(yb, out, out_bytes));
    const int mismatch = memcmp(ya, yb, out_bytes) != 0;

    const double weight_bytes = (double)stride * s->rows * ((n + tile - 1) / tile);
    const int launches = weight_bytes > 1e8 ? 4 : 64;
    ab_time_batch(a, w, x, out, s, stride, n, launches, start, stop);   /* warm-up */
    ab_time_batch(b, wb, x, out, s, stride, n, launches, start, stop);
    for (int r = 0; r < rounds; r++) {
        if (r & 1) {
            tb[r] = ab_time_batch(b, wb, x, out, s, stride, n, launches, start, stop);
            ta[r] = ab_time_batch(a, w, x, out, s, stride, n, launches, start, stop);
        } else {
            ta[r] = ab_time_batch(a, w, x, out, s, stride, n, launches, start, stop);
            tb[r] = ab_time_batch(b, wb, x, out, s, stride, n, launches, start, stop);
        }
    }
    qsort(ta, (size_t)rounds, sizeof(float), ab_compare_floats);
    qsort(tb, (size_t)rounds, sizeof(float), ab_compare_floats);
    const float ma = ta[rounds / 2], mb = tb[rounds / 2];
    printf("%-17s %6dx%-5d n=%d tile=%d  A %9.2f us  B %9.2f us  B/A speedup %5.3fx  "
           "A %6.1f GB/s  B %6.1f GB/s  spread A %4.1f%% B %4.1f%%  %s\n",
           s->name, s->rows, s->cols, n, tile, ma, mb, ma / mb, weight_bytes / ma / 1e3,
           weight_bytes / mb / 1e3, 100.0f * (ta[rounds - 1] - ta[0]) / ma,
           100.0f * (tb[rounds - 1] - tb[0]) / mb, mismatch ? "BITS-DIFFER" : "bit-identical");

    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuMemFree(w);
    cuMemFree(wb);
    cuMemFree(x);
    cuMemFree(out);
    free(wh);
    free(xh);
    free(ya);
    free(yb);
    return mismatch;
}

int main(int argc, char **argv)
{
    CUdevice dev;
    CUcontext ctx;
    Q80KernelSet a, b;
    int fails = 0;

    if (argc < 3) {
        fprintf(stderr, "usage: %s A.ptx B.ptx [rounds] | --dump-cubin X.ptx out.cubin\n",
                argv[0]);
        return 2;
    }
    Q80_CHECK(cuInit(0));
    Q80_CHECK(cuDeviceGet(&dev, 0));
    Q80_CHECK(cuCtxCreate(&ctx, 0, dev));
    if (strcmp(argv[1], "--dump-cubin") == 0 && argc >= 4) {
        q80_dump_jit_cubin(argv[2], argv[3]);
        return 0;
    }
    int rounds = argc > 3 ? atoi(argv[3]) : 15;
    if (rounds < 1 || rounds > AB_MAX_ROUNDS)
        rounds = 15;
    q80_load_kernel_set(argv[1], &a);
    q80_load_kernel_set(argv[2], &b);
    if (getenv("Q80_B_BANK8")) {   /* experiment: build B with 8-byte shared banks */
        CUfunction fs[4] = { b.single, b.tile2, b.tile4, b.tile8 };
        for (int i = 0; i < 4; i++)
            Q80_CHECK(cuFuncSetSharedMemConfig(fs[i], CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));
        printf("B kernels: 8-byte shared banks\n");
    }
    printf("A = %s\nB = %s\nrounds %d (median per launch)\n", argv[1], argv[2], rounds);

    static const int widths[] = { 1, 2, 3, 4, 5, 8, 16 };
    const char *only_rows = getenv("Q80_ONLY_ROWS"), *only_n = getenv("Q80_ONLY_N");
    const char *shape_set = getenv("Q80_SHAPES");
    const int tp2 = shape_set && strcmp(shape_set, "tp2") == 0;
    const AbShape *shapes = tp2 ? ab_shapes_tp2 : ab_shapes;
    const size_t nshapes = tp2 ? sizeof ab_shapes_tp2 / sizeof ab_shapes_tp2[0]
                               : sizeof ab_shapes / sizeof ab_shapes[0];
    for (size_t wi = 0; wi < sizeof widths / sizeof widths[0]; wi++) {
        if (only_n && atoi(only_n) != widths[wi])
            continue;
        for (size_t si = 0; si < nshapes; si++) {
            if (only_rows && atoi(only_rows) != shapes[si].rows)
                continue;
            fails += ab_run_shape(&a, &b, &shapes[si], widths[wi], rounds);
        }
    }

    printf(fails ? "RESULT: %d shape(s) BITS-DIFFER\n" : "RESULT: all shapes bit-identical\n",
           fails);
    cuCtxDestroy(ctx);
    return fails ? 1 : 0;
}
