/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file matvec_q80_ptx_harness.h
 * @brief Driver-API harness for the fc_matvec_q80 kernel family loaded from a PTX FILE.
 *
 * Lets a test or bench load two builds of cuda_kernels/matvec_kquants.ptx side by side
 * (old vs new, or a deliberately broken copy) and launch them with exactly the geometry
 * cuda/fc_cuda.c uses: 256 threads, 32 rows per block (8 for builds before the 8-lanes-
 * per-row kernel: pass the PTX as "path:8"), staged-x shared memory of
 * min(cols, 2048) floats for the single-column kernel and tile * min(cols, 512) floats
 * for the 2/4/8-wide tiles. Also holds the random repacked-weight generator and a CPU
 * replica of the kernel's exact op order (t = d*q rounded, per-lane fmaf chain over the
 * blocks, then the 16/8/4/2/1 shuffle-down tree), so outputs can be gated bit for bit.
 * Header-only (static functions); link with -lcuda.
 *
 * @status Used by test_matvec_q80_parity.c and bench_matvec_q80_ab.c.
 * @issues None
 * @todo None
 */
#ifndef MATVEC_Q80_PTX_HARNESS_H
#define MATVEC_Q80_PTX_HARNESS_H

#include <cuda.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q80_BLOCK_BYTES 36
#define Q80_THREADS 256
#define Q80_ROWS_PER_BLOCK_DEFAULT 32
#define Q80_SINGLE_CHUNK 2048
#define Q80_TILE_CHUNK 512

#define Q80_CHECK(call)                                                                   \
    do {                                                                                  \
        CUresult q80_rc_ = (call);                                                        \
        if (q80_rc_ != CUDA_SUCCESS) {                                                    \
            const char *q80_msg_ = NULL;                                                  \
            cuGetErrorString(q80_rc_, &q80_msg_);                                         \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call,          \
                    q80_msg_ ? q80_msg_ : "?");                                           \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

/* The four kernels of one PTX build, indexed by tile width 1/2/4/8. */
typedef struct {
    CUmodule module;
    CUfunction single;   /* fc_matvec_q80     */
    CUfunction tile2;    /* fc_matvec_q80_nx2 */
    CUfunction tile4;    /* fc_matvec_q80_nx  */
    CUfunction tile8;    /* fc_matvec_q80_nx8 */
    CUfunction row4[9];  /* fc_matvec_q80_r4c<nc> row tiles at nc = 3/4/5/8 (NULL: not in the build) */
    int rows_per_block;  /* launch geometry of this build */
    int g4;              /* reads the G4 layout when cols % 128 == 0 (0: ":legacy") */
} Q80KernelSet;

/* Read a whole text file into a NUL-terminated heap buffer (NULL on failure). */
static char *q80_read_text_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)len + 1);
    if (text && fread(text, 1, (size_t)len, f) != (size_t)len) {
        free(text);
        text = NULL;
    }
    if (text)
        text[len] = '\0';
    fclose(f);
    return text;
}

/* JIT the PTX file through the driver (as production does) and resolve the kernels.
 * spec is "path" or "path:rows_per_block". */
static void q80_load_kernel_set(const char *spec, Q80KernelSet *set)
{
    char ptx_path[1024];
    snprintf(ptx_path, sizeof ptx_path, "%s", spec);
    char *colon;
    set->rows_per_block = Q80_ROWS_PER_BLOCK_DEFAULT;
    set->g4 = 1;
    while ((colon = strrchr(ptx_path, ':')) != NULL) {
        if (colon[1] >= '1' && colon[1] <= '9')
            set->rows_per_block = atoi(colon + 1);
        else if (strcmp(colon + 1, "legacy") == 0)
            set->g4 = 0;
        else
            break;
        *colon = '\0';
    }
    if (set->rows_per_block != Q80_ROWS_PER_BLOCK_DEFAULT)
        set->g4 = 0;   /* the 8-rows-per-block kernel predates G4 */
    char *ptx = q80_read_text_file(ptx_path);
    if (!ptx)
        exit(2);
    Q80_CHECK(cuModuleLoadData(&set->module, ptx));
    free(ptx);
    Q80_CHECK(cuModuleGetFunction(&set->single, set->module, "fc_matvec_q80"));
    Q80_CHECK(cuModuleGetFunction(&set->tile2, set->module, "fc_matvec_q80_nx2"));
    Q80_CHECK(cuModuleGetFunction(&set->tile4, set->module, "fc_matvec_q80_nx"));
    Q80_CHECK(cuModuleGetFunction(&set->tile8, set->module, "fc_matvec_q80_nx8"));
    memset(set->row4, 0, sizeof set->row4);
    for (int nc = 3; nc <= 8; nc++) {
        char name[32];
        snprintf(name, sizeof name, "fc_matvec_q80_r4c%d", nc);
        if (cuModuleGetFunction(&set->row4[nc], set->module, name) != CUDA_SUCCESS)
            set->row4[nc] = NULL;
    }
}

/* Write the driver-JIT cubin of a PTX file (the SASS production really runs). */
static void q80_dump_jit_cubin(const char *ptx_path, const char *cubin_path)
{
    char *ptx = q80_read_text_file(ptx_path);
    CUlinkState link;
    void *cubin = NULL;
    size_t cubin_bytes = 0;

    if (!ptx)
        exit(2);
    Q80_CHECK(cuLinkCreate(0, NULL, NULL, &link));
    Q80_CHECK(cuLinkAddData(link, CU_JIT_INPUT_PTX, ptx, strlen(ptx) + 1, "matvec_kquants",
                            0, NULL, NULL));
    Q80_CHECK(cuLinkComplete(link, &cubin, &cubin_bytes));
    FILE *f = fopen(cubin_path, "wb");
    if (!f || fwrite(cubin, 1, cubin_bytes, f) != cubin_bytes)
        fprintf(stderr, "cannot write %s\n", cubin_path);
    if (f)
        fclose(f);
    cuLinkDestroy(link);
    free(ptx);
}

/* Tile width fc_cuda.c's fc_nx_tile_cols picks for n columns (FC_NX1_DIRECT default). */
static int q80_production_tile(int n)
{
    if (n == 1)
        return 1;
    if (n == 2)
        return 2;
    return n >= 5 ? 8 : 4;
}

/* Launch one kernel of the set with fc_cuda.c's geometry. tile 1 = the single-column
 * kernel (reads x column 0 only); tiles 2/4/8 cover n columns in ceil(n/tile) passes. */
static void q80_launch(const Q80KernelSet *set, int tile, CUdeviceptr w, CUdeviceptr x,
                       CUdeviceptr out, int rows, int cols, int row_stride, int n,
                       CUstream stream)
{
    unsigned grid_x = (unsigned)((rows + set->rows_per_block - 1) / set->rows_per_block);
    if (tile == 1) {
        void *args[] = { &w, &x, &out, &rows, &cols, &row_stride };
        unsigned sh = (unsigned)((cols < Q80_SINGLE_CHUNK ? cols : Q80_SINGLE_CHUNK) * 4);
        Q80_CHECK(cuLaunchKernel(set->single, grid_x, 1, 1, Q80_THREADS, 1, 1, sh, stream,
                                 args, NULL));
        return;
    }
    CUfunction kern = tile == 2 ? set->tile2 : tile == 4 ? set->tile4 : set->tile8;
    void *args[] = { &w, &x, &out, &rows, &cols, &row_stride, &n };
    unsigned sh = (unsigned)(tile * (cols < Q80_TILE_CHUNK ? cols : Q80_TILE_CHUNK) * 4);
    unsigned grid_y = (unsigned)((n + tile - 1) / tile);
    Q80_CHECK(cuLaunchKernel(kern, grid_x, grid_y, 1, Q80_THREADS, 1, 1, sh, stream, args,
                             NULL));
}

/* Row-tile width fc_cuda.c's fc_matvec_q80_tile_launch picks for n >= 3 (FC_Q80_TILE_MIN_N
 * default 3): r4c3 / r4c4 / r4c5 at n = 3 / 4 / 5, r4c8 past that. */
static int q80_production_row_tile(int n)
{
    return n <= 5 ? n : 8;
}

/* Launch what fc_cuda.c launches for n columns: the single-column kernel at n = 1, nx2 at
 * n = 2, the r4c<nc> row tile at n >= 3 on G4 rows (128 rows a block, nc columns a pass,
 * nc full 512-float chunks of shared), else the _nx tile. Returns the tile width used
 * (columns per pass). */
static int q80_launch_production(const Q80KernelSet *set, CUdeviceptr w, CUdeviceptr x,
                                 CUdeviceptr out, int rows, int cols, int row_stride, int n,
                                 CUstream stream)
{
    const int nc = q80_production_row_tile(n);
    if (n < 3 || (cols & 127) || !set->g4 || !set->row4[nc]) {
        q80_launch(set, q80_production_tile(n), w, x, out, rows, cols, row_stride, n, stream);
        return q80_production_tile(n);
    }
    void *args[] = { &w, &x, &out, &rows, &cols, &row_stride, &n };
    const unsigned grid_x = (unsigned)((rows + 4 * Q80_ROWS_PER_BLOCK_DEFAULT - 1) /
                                       (4 * Q80_ROWS_PER_BLOCK_DEFAULT));
    Q80_CHECK(cuLaunchKernel(set->row4[nc], grid_x, (unsigned)((n + nc - 1) / nc), 1,
                             Q80_THREADS, 1, 1, (unsigned)(nc * Q80_TILE_CHUNK * 4), stream,
                             args, NULL));
    return nc;
}

/* Deterministic xorshift so every run (and both kernel builds) sees the same data. */
static uint32_t q80_rand_next(uint32_t *state)
{
    uint32_t v = *state;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    *state = v;
    return v;
}

/* Repacked Q8_0 rows: per block [f32 d][32 x int8]. Scales span 2^-12..2^2 with both
 * signs; every 7th block forces the extreme bytes -128 / 127 / 0 into fixed lanes so the
 * int->float conversion is exercised at both ends on every row. */
static unsigned char *q80_make_weights(int rows, int cols, uint32_t seed, int *row_stride)
{
    const int nblk = cols / 32;
    const size_t stride = (size_t)nblk * Q80_BLOCK_BYTES;
    unsigned char *w = malloc(stride * (size_t)rows);
    uint32_t state = seed ? seed : 1;

    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nblk; b++) {
            unsigned char *blk = w + (size_t)r * stride + (size_t)b * Q80_BLOCK_BYTES;
            uint32_t bits = q80_rand_next(&state);
            float d = ldexpf(1.0f + (float)(bits & 0xFFFF) / 65536.0f,
                             (int)((bits >> 16) % 15) - 12);
            if (bits >> 31)
                d = -d;
            memcpy(blk, &d, 4);
            for (int i = 0; i < 32; i++)
                blk[4 + i] = (unsigned char)(q80_rand_next(&state) & 0xFF);
            if (b % 7 == 0) {
                blk[4 + (r % 32)] = 0x80;           /* -128 */
                blk[4 + ((r + 11) % 32)] = 0x7F;    /* 127  */
                blk[4 + ((r + 23) % 32)] = 0x00;
            }
        }
    }
    *row_stride = (int)stride;
    return w;
}

/* The bytes a build reads: the 36-B block rows above, regrouped into fc_repack_weight's
 * G4 layout when the build reads it and cols % 128 == 0 (per 4 blocks: f32 d[4], then
 * for s = 0..7 bytes 4s..4s+3 of blocks 0..3). Returns a malloc'd copy. */
static unsigned char *q80_device_weights(const Q80KernelSet *set, const unsigned char *w,
                                         int rows, int cols, int row_stride)
{
    const size_t bytes = (size_t)row_stride * (size_t)rows;
    unsigned char *d = malloc(bytes);
    memcpy(d, w, bytes);
    if (!set->g4 || cols % 128)
        return d;
    for (int r = 0; r < rows; r++)
        for (int g = 0; g < cols / 128; g++) {
            const unsigned char *src = w + (size_t)r * row_stride + (size_t)g * 144;
            unsigned char *dst = d + (size_t)r * row_stride + (size_t)g * 144;
            for (int j = 0; j < 4; j++) {
                memcpy(dst + 4 * j, src + 36 * j, 4);
                for (int l = 0; l < 8; l++)
                    memcpy(dst + 16 + 16 * l + 4 * j, src + 36 * j + 4 + 4 * l, 4);
            }
        }
    return d;
}

/* n columns of activations in [-2, 2), column-major (column c at c * cols). */
static float *q80_make_activations(int cols, int n, uint32_t seed)
{
    float *x = malloc((size_t)cols * (size_t)n * sizeof(float));
    uint32_t state = seed ? seed : 1;
    for (size_t i = 0; i < (size_t)cols * (size_t)n; i++)
        x[i] = (float)(q80_rand_next(&state) >> 8) / 4194304.0f - 2.0f;
    return x;
}

/* CPU replica of one output row: lane l accumulates element l of every whole block of
 * every chunk (partial trailing blocks ignored, as the kernel does), then the warp tree. */
static float q80_reference_row(const unsigned char *row, const float *x, int cols)
{
    float lane_sum[32];
    const int nblk = cols / 32;

    for (int l = 0; l < 32; l++) {
        float p = 0.0f;
        for (int b = 0; b < nblk; b++) {
            const unsigned char *blk = row + (size_t)b * Q80_BLOCK_BYTES;
            float d;
            memcpy(&d, blk, 4);
            volatile float t = d * (float)(signed char)blk[4 + l];
            p = fmaf(t, x[32 * b + l], p);
        }
        lane_sum[l] = p;
    }
    for (int off = 16; off > 0; off >>= 1)
        for (int l = 0; l < off; l++)
            lane_sum[l] = lane_sum[l] + lane_sum[l + off];
    return lane_sum[0];
}

#endif /* MATVEC_Q80_PTX_HARNESS_H */
