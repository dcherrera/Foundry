/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_matvec_q80_parity.c
 * @brief Bit-exact gate for the fc_matvec_q80 family of one matvec_kquants.ptx build.
 *
 * Every kernel (single-column, 2-, 4- and 8-wide tiles) is run on edge shapes — rows not
 * a multiple of 8, a partial last chunk whose block count is not a multiple of 4 or 8,
 * one-block and three-block rows, an ignored partial trailing block, two full chunks —
 * at full and partial tile widths, and every output column is compared BIT FOR BIT with
 * the CPU replica of the kernel's op order (matvec_q80_ptx_harness.h). Weights include
 * the bytes -128 / 127 / 0 on every row and scales of both signs across 15 binades.
 * With a second PTX path the two builds are also memcmp'd against each other.
 *
 * Build (on the bank, from foundry/):
 *   gcc -O2 -ffp-contract=off -I/usr/local/cuda-11.4/include \
 *       -o cuda/tests/test_matvec_q80_parity cuda/tests/test_matvec_q80_parity.c -lcuda -lm
 * Run: CUDA_VISIBLE_DEVICES=0 ./cuda/tests/test_matvec_q80_parity NEW.ptx[:rpb][:legacy] [OLD.ptx[:rpb][:legacy]]
 *      (rpb = rows per block of that build, default 32; origin/main before this kernel = 8;
 *      :legacy = a build before the G4 layout, fed the plain 36-B blocks)
 *
 * @status Working.
 * @issues None
 * @todo None
 */

#include "matvec_q80_ptx_harness.h"

#define PARITY_MAX_COLUMNS 12

typedef struct {
    int rows, cols;
} ParityShape;

typedef struct {
    int tile, n;
} ParityLaunch;

static const ParityShape parity_shapes[] = {
    { 77, 2656 },   /* 10 row blocks, last half full; chunks 2048+608 (19 blocks) / 5x512+96 */
    { 64, 4096 },   /* two full single-column chunks */
    { 40, 96 },     /* three blocks per row (single-column kernel only) */
    { 16, 32 },     /* one block per row (single-column kernel only) */
    { 8, 544 },     /* tiles: one full 512 chunk + a one-block chunk */
    { 24, 1040 },   /* 32 whole blocks + an ignored 16-column partial block */
    { 512, 2048 },  /* shared-expert gate/up, attn k/v */
    { 2048, 512 },  /* shared-expert down */
    { 77, 2176 },   /* G4: chunks 2048+128 (one group) / 4x512+128 */
    { 33, 128 },    /* G4: one group per row */
};

static const ParityLaunch parity_launches[] = {
    { 1, 1 }, { 2, 1 }, { 2, 2 }, { 4, 1 }, { 4, 3 }, { 4, 4 }, { 4, 6 },
    { 8, 5 }, { 8, 8 }, { 8, 12 },
};

/* Compare n columns against the replica; prints the first bad element. Returns the
 * number of mismatching elements. */
static long parity_compare(const float *got, const float *want, int rows, int n,
                           const char *what, const ParityShape *s, const ParityLaunch *l)
{
    long bad = 0;
    for (int c = 0; c < n; c++) {
        for (int r = 0; r < rows; r++) {
            const float g = got[(size_t)c * rows + r], e = want[(size_t)c * rows + r];
            if (memcmp(&g, &e, 4) == 0)
                continue;
            if (bad == 0)
                printf("  %s %dx%d tile=%d n=%d: col %d row %d got %.9g want %.9g\n", what,
                       s->rows, s->cols, l->tile, l->n, c, r, g, e);
            bad++;
        }
    }
    return bad;
}

/* All launches of one shape for one or two builds. Returns mismatching elements. */
static long parity_run_shape(const Q80KernelSet *build, const Q80KernelSet *other,
                             const ParityShape *s)
{
    int stride = 0;
    unsigned char *wh = q80_make_weights(s->rows, s->cols, 0xC2B2AE35u ^ (uint32_t)s->cols,
                                         &stride);
    float *xh = q80_make_activations(s->cols, PARITY_MAX_COLUMNS, 0x27D4EB2Fu);
    const size_t out_bytes = (size_t)s->rows * PARITY_MAX_COLUMNS * sizeof(float);
    float *want = malloc(out_bytes), *got = malloc(out_bytes), *got2 = malloc(out_bytes);
    CUdeviceptr w, w2 = 0, x, out;
    long bad = 0;

    for (int c = 0; c < PARITY_MAX_COLUMNS; c++)
        for (int r = 0; r < s->rows; r++)
            want[(size_t)c * s->rows + r] = q80_reference_row(
                wh + (size_t)r * stride, xh + (size_t)c * s->cols, s->cols);
    Q80_CHECK(cuMemAlloc(&w, (size_t)stride * (size_t)s->rows));
    Q80_CHECK(cuMemAlloc(&x, (size_t)s->cols * PARITY_MAX_COLUMNS * sizeof(float)));
    Q80_CHECK(cuMemAlloc(&out, out_bytes));
    unsigned char *wd = q80_device_weights(build, wh, s->rows, s->cols, stride);
    Q80_CHECK(cuMemcpyHtoD(w, wd, (size_t)stride * (size_t)s->rows));
    free(wd);
    if (other) {   /* the other build may read the other layout */
        Q80_CHECK(cuMemAlloc(&w2, (size_t)stride * (size_t)s->rows));
        wd = q80_device_weights(other, wh, s->rows, s->cols, stride);
        Q80_CHECK(cuMemcpyHtoD(w2, wd, (size_t)stride * (size_t)s->rows));
        free(wd);
    }
    Q80_CHECK(cuMemcpyHtoD(x, xh, (size_t)s->cols * PARITY_MAX_COLUMNS * sizeof(float)));

    for (size_t li = 0; li < sizeof parity_launches / sizeof parity_launches[0]; li++) {
        const ParityLaunch *l = &parity_launches[li];
        if (l->tile > 1 && s->cols < Q80_TILE_CHUNK)
            continue;   /* tiles address column c at c*512: cols < 512 is never launched */
        const size_t bytes = (size_t)s->rows * (size_t)l->n * sizeof(float);
        Q80_CHECK(cuMemsetD8(out, 0xFF, out_bytes));
        q80_launch(build, l->tile, w, x, out, s->rows, s->cols, stride, l->n, 0);
        Q80_CHECK(cuMemcpyDtoH(got, out, bytes));
        long b = parity_compare(got, want, s->rows, l->n, "vs replica", s, l);
        if (other) {
            Q80_CHECK(cuMemsetD8(out, 0xFF, out_bytes));
            q80_launch(other, l->tile, w2, x, out, s->rows, s->cols, stride, l->n, 0);
            Q80_CHECK(cuMemcpyDtoH(got2, out, bytes));
            b += parity_compare(got, got2, s->rows, l->n, "vs other build", s, l);
        }
        printf("%-6s %6dx%-5d tile=%d n=%-2d  %ld mismatching\n", b ? "FAIL" : "PASS",
               s->rows, s->cols, l->tile, l->n, b);
        bad += b;
    }
    cuMemFree(w);
    if (w2)
        cuMemFree(w2);
    cuMemFree(x);
    cuMemFree(out);
    free(wh);
    free(xh);
    free(want);
    free(got);
    free(got2);
    return bad;
}

int main(int argc, char **argv)
{
    CUdevice dev;
    CUcontext ctx;
    Q80KernelSet build, other;
    long bad = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s NEW.ptx [OLD.ptx]\n", argv[0]);
        return 2;
    }
    Q80_CHECK(cuInit(0));
    Q80_CHECK(cuDeviceGet(&dev, 0));
    Q80_CHECK(cuCtxCreate(&ctx, 0, dev));
    q80_load_kernel_set(argv[1], &build);
    if (argc > 2)
        q80_load_kernel_set(argv[2], &other);
    printf("build %s%s%s\n", argv[1], argc > 2 ? " vs " : "", argc > 2 ? argv[2] : "");

    for (size_t si = 0; si < sizeof parity_shapes / sizeof parity_shapes[0]; si++)
        bad += parity_run_shape(&build, argc > 2 ? &other : NULL, &parity_shapes[si]);

    printf(bad ? "RESULT: FAIL (%ld mismatching elements)\n" : "RESULT: PASS\n", bad);
    cuCtxDestroy(ctx);
    return bad ? 1 : 0;
}
