/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_q80_tile.c
 * @brief Q8_0 prefill-tile kernels (fc_matvec_q80_r<R>c<NC>) vs fc_matvec_q80_nx8.
 *
 * One matvec_kquants.ptx build (compile it with -DFC_Q80_TILE_SWEEP for every variant).
 * Per prefill shape (Qwen3.8-27B dense FFN / GDN / attention projections at the
 * 128-token chunk) it runs nx8 as the reference, then each tile kernel present in the
 * module: memcmp of all outputs against the production _nx tile for that n (nx2 / nx /
 * nx8, bit-identical or the run fails), and the median timing with the achieved FP32 rate.
 *
 * Build (bank, from foundry/):
 *   gcc -O2 -I/usr/local/cuda-11.4/include -o cuda/tests/bench_q80_tile \
 *       cuda/tests/bench_q80_tile.c -lcuda -lm
 * Run: CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/bench_q80_tile matvec_kquants.ptx [n[,n..]] [rounds]
 *      TB_ONLY=r4c8,r2c8 limits the tile kernels (the production tile always runs). Without the sweep
 *      define only the shipped r4c8 is in the module.
 *
 * @status Working.
 * @issues None
 * @todo None
 */

#include "matvec_q80_ptx_harness.h"

#define TB_MAX_ROUNDS 32

typedef struct {
    const char *name;
    int rows, cols;
} TbShape;

static const TbShape tb_shapes[] = {
    { "ffn_gate/up", 17408, 5120 },
    { "ffn_down", 5120, 17408 },
    { "ssm_qkv", 10240, 5120 },
    { "ssm_z/attn_qg/2", 6144, 5120 },
    { "ssm_out/attn_o", 5120, 6144 },
};

typedef struct {
    const char *name;
    int r, nc;
    CUfunction f;
} TbKern;

/* Production _nx tiles (fc_cuda.c's fc_nx_tile_cols): the reference for each n. */
static TbKern tb_prod[] = {
    { "fc_matvec_q80_nx2", 1, 2, NULL },
    { "fc_matvec_q80_nx", 1, 4, NULL },
    { "fc_matvec_q80_nx8", 1, 8, NULL },
};

static TbKern tb_kerns[] = {
    { "fc_matvec_q80_nx8", 1, 8, NULL },      /* slot 0: the production tile for this n */
    { "fc_matvec_q80_r1c8", 1, 8, NULL },
    { "fc_matvec_q80_r2c8", 2, 8, NULL },
    { "fc_matvec_q80_r2c12", 2, 12, NULL },
    { "fc_matvec_q80_r3c8", 3, 8, NULL },
    { "fc_matvec_q80_r4c4", 4, 4, NULL },
    { "fc_matvec_q80_r4c8", 4, 8, NULL },     /* shipped */
    { "fc_matvec_q80_r4c1", 4, 1, NULL },
    { "fc_matvec_q80_r4c2", 4, 2, NULL },
    { "fc_matvec_q80_r4c3", 4, 3, NULL },
    { "fc_matvec_q80_r4c5", 4, 5, NULL },
    { "fc_matvec_q80_r4c6", 4, 6, NULL },
    { "fc_matvec_q80_r2c2", 2, 2, NULL },
    { "fc_matvec_q80_r2c3", 2, 3, NULL },
    { "fc_matvec_q80_r2c5", 2, 5, NULL },
    { "fc_matvec_q80_r8c2", 8, 2, NULL },
    { "fc_matvec_q80_r8c3", 8, 3, NULL },
};
#define TB_NKERN ((int)(sizeof tb_kerns / sizeof tb_kerns[0]))

static int tb_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static void tb_launch(const TbKern *k, CUdeviceptr w, CUdeviceptr x, CUdeviceptr out,
                      int rows, int cols, int stride, int n)
{
    void *args[] = { &w, &x, &out, &rows, &cols, &stride, &n };
    unsigned grid_x = (unsigned)((rows + 32 * k->r - 1) / (32 * k->r));
    unsigned grid_y = (unsigned)((n + k->nc - 1) / k->nc);
    unsigned sh = (unsigned)(k->nc * (cols < Q80_TILE_CHUNK ? cols : Q80_TILE_CHUNK) * 4);
    Q80_CHECK(cuLaunchKernel(k->f, grid_x, grid_y, 1, Q80_THREADS, 1, 1, sh, 0, args, NULL));
}

static float tb_time(const TbKern *k, CUdeviceptr w, CUdeviceptr x, CUdeviceptr out,
                     const TbShape *s, int stride, int n, int launches, CUevent e0,
                     CUevent e1)
{
    float ms = 0.0f;
    Q80_CHECK(cuEventRecord(e0, 0));
    for (int i = 0; i < launches; i++)
        tb_launch(k, w, x, out, s->rows, s->cols, stride, n);
    Q80_CHECK(cuEventRecord(e1, 0));
    Q80_CHECK(cuEventSynchronize(e1));
    Q80_CHECK(cuEventElapsedTime(&ms, e0, e1));
    return ms * 1000.0f / (float)launches;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s matvec_kquants.ptx [n] [rounds]\n", argv[0]);
        return 2;
    }
    int ns[16], n_ns = 0;
    for (const char *p = argc > 2 ? argv[2] : "128"; *p && n_ns < 16;) {
        ns[n_ns++] = atoi(p);
        p = strchr(p, ',');
        if (!p)
            break;
        p++;
    }
    int rounds = argc > 3 ? atoi(argv[3]) : 5;
    if (rounds > TB_MAX_ROUNDS)
        rounds = TB_MAX_ROUNDS;
    CUdevice dev;
    CUcontext cu;
    CUmodule mod;
    Q80_CHECK(cuInit(0));
    Q80_CHECK(cuDeviceGet(&dev, 0));
    Q80_CHECK(cuCtxCreate(&cu, 0, dev));
    char *ptx = q80_read_text_file(argv[1]);
    if (!ptx)
        return 2;
    Q80_CHECK(cuModuleLoadData(&mod, ptx));
    free(ptx);
    const char *only_env = getenv("TB_ONLY"); /* comma list; the production tile always runs */
    char only_buf[512];
    const char *only = NULL;
    if (only_env) {
        snprintf(only_buf, sizeof only_buf, ",%s,", only_env);
        only = only_buf;
    }
    for (int i = 0; i < TB_NKERN; i++) {
        char key[40];
        snprintf(key, sizeof key, ",%s,", tb_kerns[i].name + strlen("fc_matvec_q80_"));
        if (i > 0 && only && !strstr(only, key))
            continue;
        if (cuModuleGetFunction(&tb_kerns[i].f, mod, tb_kerns[i].name) != CUDA_SUCCESS)
            tb_kerns[i].f = NULL;
    }
    for (int i = 0; i < 3; i++)
        Q80_CHECK(cuModuleGetFunction(&tb_prod[i].f, mod, tb_prod[i].name));
    Q80KernelSet g4 = { .g4 = 1, .rows_per_block = 32 };
    int fail = 0;

    for (int ni = 0; ni < n_ns; ni++)
    for (size_t si = 0; si < sizeof tb_shapes / sizeof tb_shapes[0]; si++) {
        const int n = ns[ni];
        const TbShape *s = &tb_shapes[si];
        tb_kerns[0] = tb_prod[n == 2 ? 0 : n <= 4 ? 1 : 2];
        int stride = 0;
        unsigned char *wh = q80_make_weights(s->rows, s->cols, 0x9E3779B9u ^ (uint32_t)s->rows,
                                             &stride);
        unsigned char *wd = q80_device_weights(&g4, wh, s->rows, s->cols, stride);
        float *xh = q80_make_activations(s->cols, n, 0x85EBCA6Bu ^ (uint32_t)s->cols);
        const size_t ob = (size_t)s->rows * (size_t)n * sizeof(float);
        float *yref = malloc(ob), *y = malloc(ob);
        CUdeviceptr w, x, out;
        CUevent e0, e1;
        Q80_CHECK(cuMemAlloc(&w, (size_t)stride * (size_t)s->rows));
        Q80_CHECK(cuMemAlloc(&x, (size_t)s->cols * (size_t)n * sizeof(float)));
        Q80_CHECK(cuMemAlloc(&out, ob));
        Q80_CHECK(cuMemcpyHtoD(w, wd, (size_t)stride * (size_t)s->rows));
        Q80_CHECK(cuMemcpyHtoD(x, xh, (size_t)s->cols * (size_t)n * sizeof(float)));
        Q80_CHECK(cuEventCreate(&e0, CU_EVENT_DEFAULT));
        Q80_CHECK(cuEventCreate(&e1, CU_EVENT_DEFAULT));
        tb_launch(&tb_kerns[0], w, x, out, s->rows, s->cols, stride, n);
        Q80_CHECK(cuMemcpyDtoH(yref, out, ob));
        const double flop = 2.0 * s->rows * (double)s->cols * n;
        float tref = 0.0f;

        for (int ki = 0; ki < TB_NKERN; ki++) {
            const TbKern *k = &tb_kerns[ki];
            if (!k->f)
                continue;
            Q80_CHECK(cuMemsetD8(out, 0xFF, ob));
            tb_launch(k, w, x, out, s->rows, s->cols, stride, n);
            Q80_CHECK(cuMemcpyDtoH(y, out, ob));
            const int diff = memcmp(y, yref, ob) != 0;
            fail |= diff;
            float t[TB_MAX_ROUNDS];
            tb_time(k, w, x, out, s, stride, n, 2, e0, e1);   /* warm-up */
            for (int r = 0; r < rounds; r++)
                t[r] = tb_time(k, w, x, out, s, stride, n, 4, e0, e1);
            qsort(t, (size_t)rounds, sizeof(float), tb_cmp);
            const float med = t[rounds / 2];
            if (ki == 0)
                tref = med;
            printf("%-16s %5dx%-5d n=%d %-20s %9.1f us %7.1f GFLOP/s  x%.3f vs prod  %s\n",
                   s->name, s->rows, s->cols, n, k->name, med, flop / med / 1e3,
                   tref / med, diff ? "BITS-DIFFER" : "bit-identical");
            fflush(stdout);
        }
        cuEventDestroy(e0);
        cuEventDestroy(e1);
        cuMemFree(w);
        cuMemFree(x);
        cuMemFree(out);
        free(wh);
        free(wd);
        free(xh);
        free(yref);
        free(y);
    }
    printf("%s\n", fail ? "FAIL: a tile kernel differs from the production tile" : "PASS: all bit-identical");
    return fail;
}
