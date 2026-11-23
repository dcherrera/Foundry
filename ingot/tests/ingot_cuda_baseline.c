/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Matvec time of foundry's dense kernels on one die (see ingot_cuda_baseline.h). */
#include "ingot/tests/ingot_cuda_baseline.h"

#include <stdlib.h>
#include <time.h>

#include "cuda/fc_cuda.h"
#include "ingot/tests/ingot_test_files.h"

enum { WARMUP = 1, RUNS = 10, Q8_0 = 8 };

struct ingot_cuda_baseline {
    fc_cuda_ctx *ctx;
};

int ingot_cuda_baseline_open(int device, ingot_cuda_baseline **out) {
    ingot_cuda_baseline *b = calloc(1, sizeof(*b));
    if (!b)
        return -1;
    b->ctx = fc_cuda_init_device(device);
    if (!b->ctx || !fc_matvec_bf16_available_ctx(b->ctx)) {
        ingot_cuda_baseline_close(b);
        return -1;
    }
    *out = b;
    return 0;
}

void ingot_cuda_baseline_close(ingot_cuda_baseline *b) {
    if (!b)
        return;
    if (b->ctx)
        fc_cuda_ctx_free(b->ctx);
    free(b);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

typedef void (*matvec_fn)(fc_cuda_ctx *, fc_buffer *, fc_buffer *, fc_buffer *, int, int, int);

/* Mean time of RUNS launches after WARMUP. */
static double time_matvec(fc_cuda_ctx *ctx, matvec_fn fn, fc_buffer *w, fc_buffer *x,
                          fc_buffer *y, int rows, int cols, int stride) {
    for (int i = 0; i < WARMUP; i++)
        fn(ctx, w, x, y, rows, cols, stride);
    fc_sync_ctx(ctx);
    double t0 = now_ms();
    for (int i = 0; i < RUNS; i++)
        fn(ctx, w, x, y, rows, cols, stride);
    fc_sync_ctx(ctx);
    return (now_ms() - t0) / RUNS;
}

int ingot_cuda_baseline_time(ingot_cuda_baseline *b, const uint16_t *w, uint64_t rows,
                             uint64_t cols, double *bf16_ms, double *q80_ms) {
    *bf16_ms = *q80_ms = -1;
    float *hx = malloc(cols * sizeof(float));
    if (!hx)
        return -1;
    for (uint64_t c = 0; c < cols; c++)
        hx[c] = (float)((c * 2654435761u % 2001) / 1000.0 - 1.0);
    fc_buffer *x = fc_buffer_from_ptr_ctx(b->ctx, hx, cols * sizeof(float));
    fc_buffer *y = fc_buffer_alloc_ctx(b->ctx, rows * sizeof(float));
    fc_buffer *wb = fc_buffer_from_ptr_ctx(b->ctx, w, rows * cols * 2);
    int rc = x && y && wb ? 0 : -1;
    if (!rc)
        *bf16_ms = time_matvec(b->ctx, fc_matvec_bf16_ctx, wb, x, y, (int)rows, (int)cols,
                               (int)(cols * 2));
    fc_buffer_free(wb);
    size_t row_bytes = cols % 32 ? 0 : fc_repack_row_bytes(Q8_0, (int)cols);
    uint8_t *q = row_bytes ? malloc(rows * cols / 32 * 34) : NULL;
    uint8_t *packed = q ? malloc(rows * row_bytes) : NULL;
    if (!rc && packed) {
        test_q8_quantize(w, rows * cols, q);
        rc = fc_repack_weight(Q8_0, q, packed, (long)rows, (int)cols) ? -1 : 0;
        fc_buffer *wq = rc ? NULL : fc_buffer_from_ptr_ctx(b->ctx, packed, rows * row_bytes);
        if (wq)
            *q80_ms = time_matvec(b->ctx, fc_matvec_q80_ctx, wq, x, y, (int)rows, (int)cols,
                                  (int)row_bytes);
        rc = rc || !wq ? -1 : 0;
        fc_buffer_free(wq);
    }
    free(q);
    free(packed);
    fc_buffer_free(x);
    fc_buffer_free(y);
    free(hx);
    return rc;
}
