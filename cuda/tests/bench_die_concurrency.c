/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_die_concurrency.c
 * @brief How much aggregate throughput N concurrently-busy K80 dies deliver.
 *
 * WHY THIS EXISTS. A pipelined prefill keeps every die of an N-die layer split
 * busy at the same instant instead of one at a time. Its ceiling is N only if N
 * busy dies each run as fast as one busy die does. On a K80 they do not: the
 * board is two GK210 dies behind one power budget, so a busy sibling drags a die
 * off its 875 MHz application clock. That throttle is the difference between a
 * measured 1.96x and a measured 1.23x on the shipped 2-die prefill pipeline, and
 * nothing in the tree has ever measured it past two dies.
 *
 * This harness answers exactly one question: run the SAME fixed load on each of
 * N dies at the same time, and what is the aggregate throughput relative to N x
 * the one-die figure? The answer prices any die-parallel scheme on this host
 * before it is built.
 *
 * WHAT IT DOES. One process, one context per die, one streaming-read kernel
 * (bench_bw.ptx fc_bw_read_u128_ldg_u4 - the geometry bench_bandwidth found
 * peaks) issued to every die's null stream, then all contexts synchronised. That
 * is the same shape a pipelined forward has: one host thread issuing async work
 * to several dies and joining. Reads dominate a memory-bound MoE/SSM prefill, so
 * a streaming read is the closest cheap proxy for its power draw.
 *
 * WHAT IT REPORTS, per die-set: wall seconds for the round, aggregate GB/s,
 * per-die GB/s, and the scaling efficiency against the one-die arm. Clocks and
 * power are NOT read here - sample them from outside with nvidia-smi while this
 * runs, so the measurement and the instrument stay independent.
 *
 * FALSIFICATION. --dies with one die must reproduce bench_bandwidth's one-die
 * figure for the same kernel and buffer. If a die-set of N reports N x the
 * one-die throughput exactly, suspect that the launches never overlapped: check
 * that wall time for N dies exceeds the one-die wall time by less than N.
 *
 * BUILD (from foundry/), no CUDA headers needed - the driver API is dlopen'd:
 *   gcc -O2 -Wall -o cuda/tests/bench_die_concurrency \
 *       cuda/tests/bench_die_concurrency.c -ldl
 * RUN:
 *   ./cuda/tests/bench_die_concurrency --ptx cuda_kernels/bench_bw.ptx \
 *       --dies 0 --dies 0,1 --dies 0,2 --dies 0,2,4,6 --dies 0,1,2,3,4,5,6,7
 *
 * @status Reports aggregate + per-die GB/s and efficiency for each --dies set.
 * @issues None
 * @todo None
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ================================================================
 * Minimal CUDA driver API surface (dlopen'd, as fc_cuda.c does)
 * ================================================================ */

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;

#define CU_SUCCESS 0

static void *g_libcuda;

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGetCount)(int *);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuCtxDestroy)(CUcontext);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemGetInfo)(size_t *, size_t *);
static CUresult (*p_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, CUstream,
                                    void **, void **);

static void *sym(const char *name, const char *v2)
{
    void *s = v2 ? dlsym(g_libcuda, v2) : NULL;
    if (!s)
        s = dlsym(g_libcuda, name);
    if (!s) {
        fprintf(stderr, "[die_conc] missing driver symbol %s\n", name);
        exit(1);
    }
    return s;
}

static void load_driver(void)
{
    g_libcuda = dlopen("libcuda.so.1", RTLD_NOW);
    if (!g_libcuda)
        g_libcuda = dlopen("libcuda.so", RTLD_NOW);
    if (!g_libcuda) {
        fprintf(stderr, "[die_conc] dlopen libcuda failed: %s\n", dlerror());
        exit(1);
    }
    p_cuInit              = sym("cuInit", NULL);
    p_cuDeviceGetCount    = sym("cuDeviceGetCount", NULL);
    p_cuDeviceGet         = sym("cuDeviceGet", NULL);
    p_cuCtxCreate         = sym("cuCtxCreate", "cuCtxCreate_v2");
    p_cuCtxDestroy        = sym("cuCtxDestroy", "cuCtxDestroy_v2");
    p_cuCtxSetCurrent     = sym("cuCtxSetCurrent", NULL);
    p_cuCtxSynchronize    = sym("cuCtxSynchronize", NULL);
    p_cuMemAlloc          = sym("cuMemAlloc", "cuMemAlloc_v2");
    p_cuMemFree           = sym("cuMemFree", "cuMemFree_v2");
    p_cuMemGetInfo        = sym("cuMemGetInfo", "cuMemGetInfo_v2");
    p_cuMemsetD32         = sym("cuMemsetD32", "cuMemsetD32_v2");
    p_cuModuleLoadData    = sym("cuModuleLoadData", NULL);
    p_cuModuleGetFunction = sym("cuModuleGetFunction", NULL);
    p_cuLaunchKernel      = sym("cuLaunchKernel", NULL);
}

static void cu_ok(CUresult r, const char *what)
{
    if (r != CU_SUCCESS) {
        fprintf(stderr, "[die_conc] %s failed: %d\n", what, r);
        exit(1);
    }
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Idle between arms so a board enters each one from a comparable power/thermal
 * state. Must not busy-wait: the reference host has two cores and the issuing thread
 * is one of them. */
static void sleep_s(double secs)
{
    struct timespec ts;
    ts.tv_sec = (time_t)secs;
    ts.tv_nsec = (long)((secs - (double)ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
}

/* ================================================================
 * Per-die resources
 * ================================================================ */

#define MAX_DIES 16

typedef struct {
    int         ordinal;      /* CUDA device ordinal */
    CUcontext   ctx;
    CUdeviceptr src;          /* the buffer the kernel streams */
    CUdeviceptr dst;          /* one u32 per thread, keeps loads alive */
    CUmodule    mod;
    CUfunction  fn;
} die_slot;

/* A die-set arm: which dies ran together and what they achieved. */
typedef struct {
    int    n_dies;
    double wall_s;
    double bytes_total;
    double agg_gbs;
    double per_die_gbs;
} arm_result;

/**
 * Bring up one die: context, module, buffers. `buf_bytes` is the streaming
 * source; `threads` sized destination keeps the compiler from eliding loads.
 * Ownership: the caller frees via die_slot_teardown.
 */
static void die_slot_init(die_slot *d, int ordinal, const char *ptx,
                          size_t buf_bytes, size_t dst_bytes)
{
    CUdevice dev;
    d->ordinal = ordinal;
    cu_ok(p_cuDeviceGet(&dev, ordinal), "cuDeviceGet");
    cu_ok(p_cuCtxCreate(&d->ctx, 0, dev), "cuCtxCreate");
    cu_ok(p_cuCtxSetCurrent(d->ctx), "cuCtxSetCurrent");

    size_t free_b = 0, total_b = 0;
    cu_ok(p_cuMemGetInfo(&free_b, &total_b), "cuMemGetInfo");
    if (free_b < buf_bytes + dst_bytes + (64u << 20)) {
        fprintf(stderr, "[die_conc] die %d: only %.0f MiB free, need %.0f MiB "
                "- lower --buf-mib\n", ordinal, free_b / 1048576.0,
                (buf_bytes + dst_bytes) / 1048576.0);
        exit(1);
    }
    cu_ok(p_cuMemAlloc(&d->src, buf_bytes), "cuMemAlloc src");
    cu_ok(p_cuMemAlloc(&d->dst, dst_bytes), "cuMemAlloc dst");
    cu_ok(p_cuMemsetD32(d->src, 0x3f800000u, buf_bytes / 4), "cuMemsetD32 src");
    cu_ok(p_cuMemsetD32(d->dst, 0u, dst_bytes / 4), "cuMemsetD32 dst");
    cu_ok(p_cuModuleLoadData(&d->mod, ptx), "cuModuleLoadData");
}

static void die_slot_bind_kernel(die_slot *d, const char *kernel)
{
    cu_ok(p_cuCtxSetCurrent(d->ctx), "cuCtxSetCurrent");
    cu_ok(p_cuModuleGetFunction(&d->fn, d->mod, kernel), "cuModuleGetFunction");
}

static void die_slot_teardown(die_slot *d)
{
    if (!d->ctx)
        return;
    p_cuCtxSetCurrent(d->ctx);
    p_cuMemFree(d->src);
    p_cuMemFree(d->dst);
    p_cuCtxDestroy(d->ctx);
    d->ctx = NULL;
}

/**
 * Issue `launches` kernel launches to every die in `set`, round-robin so no die
 * is starved by the host walking the list, then join every context.
 *
 * Round-robin matters: issuing all of die 0's launches before touching die 1
 * would let die 0 finish before die 1 starts on a short arm, and the dies would
 * never actually be busy together - which is the thing being measured.
 */
static double run_concurrent_round(die_slot *slots, const int *set, int n_set,
                                   int launches, int grid, int block, int iters)
{
    double t0 = now_s();
    for (int rep = 0; rep < launches; rep++) {
        for (int i = 0; i < n_set; i++) {
            die_slot *d = &slots[set[i]];
            void *args[] = { &d->src, &d->dst, &iters };
            cu_ok(p_cuCtxSetCurrent(d->ctx), "cuCtxSetCurrent");
            cu_ok(p_cuLaunchKernel(d->fn, grid, 1, 1, block, 1, 1, 0, NULL,
                                   args, NULL), "cuLaunchKernel");
        }
    }
    for (int i = 0; i < n_set; i++) {
        cu_ok(p_cuCtxSetCurrent(slots[set[i]].ctx), "cuCtxSetCurrent");
        cu_ok(p_cuCtxSynchronize(), "cuCtxSynchronize");
    }
    return now_s() - t0;
}

static int parse_die_list(const char *s, int *out)
{
    int n = 0;
    const char *p = s;
    while (*p && n < MAX_DIES) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p)
            break;
        out[n++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ')
            p++;
    }
    return n;
}

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[die_conc] cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "[die_conc] cannot read %s\n", path);
        exit(1);
    }
    buf[len] = '\0';
    fclose(f);
    return buf;
}

static void usage(void)
{
    printf("usage: bench_die_concurrency [--ptx PATH] [--buf-mib N] "
           "[--launches N] [--iters N] [--unroll N] [--grid N]\n"
           "                             [--kernel NAME] [--settle-s N] "
           "--dies 0 [--dies 0,1] ...\n");
}

int main(int argc, char **argv)
{
    const char *ptx_path = "cuda_kernels/bench_bw.ptx";
    const char *kernel   = "fc_bw_read_u128_ldg_u4";
    int    buf_mib   = 256;
    int    launches  = 40;
    int    iters     = 30;
    int    block     = 256;
    int    grid      = 1024;
    int    unroll    = 4;      /* the _u<N> suffix of --kernel; sets bytes/iter */
    double settle_s  = 3.0;

    int  sets[MAX_DIES][MAX_DIES];
    int  set_n[MAX_DIES];
    int  n_sets = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ptx") && i + 1 < argc)
            ptx_path = argv[++i];
        else if (!strcmp(argv[i], "--kernel") && i + 1 < argc)
            kernel = argv[++i];
        else if (!strcmp(argv[i], "--buf-mib") && i + 1 < argc)
            buf_mib = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--launches") && i + 1 < argc)
            launches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
            iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--unroll") && i + 1 < argc)
            unroll = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--grid") && i + 1 < argc)
            grid = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--settle-s") && i + 1 < argc)
            settle_s = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dies") && i + 1 < argc && n_sets < MAX_DIES)
            set_n[n_sets] = parse_die_list(argv[++i], sets[n_sets]), n_sets++;
        else {
            usage();
            return 1;
        }
    }
    if (n_sets == 0) {
        usage();
        return 1;
    }

    load_driver();
    cu_ok(p_cuInit(0), "cuInit");
    int n_dev = 0;
    cu_ok(p_cuDeviceGetCount(&n_dev), "cuDeviceGetCount");

    /* Every die named by any set gets brought up once, so a set's arm pays no
     * context-creation or module-load cost inside its timed region. */
    int want[MAX_DIES];
    memset(want, 0, sizeof(want));
    for (int s = 0; s < n_sets; s++)
        for (int i = 0; i < set_n[s]; i++) {
            if (sets[s][i] < 0 || sets[s][i] >= n_dev) {
                fprintf(stderr, "[die_conc] die %d out of range (%d visible)\n",
                        sets[s][i], n_dev);
                return 1;
            }
            want[sets[s][i]] = 1;
        }

    char *ptx = read_file(ptx_path);
    size_t buf_bytes = (size_t)buf_mib << 20;
    size_t dst_bytes = (size_t)grid * block * sizeof(unsigned int);

    /* The kernel strides forward through the buffer and never wraps, so one
     * launch touches exactly bytes_per_launch of it. Reading past the
     * allocation would not fault reliably on this driver - it would return
     * whatever is resident and quietly report a bandwidth for the wrong bytes,
     * so this is a hard refusal rather than a warning. */
    double footprint = (double)grid * block * iters * unroll * 16.0;
    if (footprint > (double)buf_bytes) {
        fprintf(stderr, "[die_conc] REFUSE: one launch walks %.1f MiB but the "
                "buffer is %d MiB. Raise --buf-mib or lower --iters/--grid.\n",
                footprint / 1048576.0, buf_mib);
        return 1;
    }

    die_slot slots[MAX_DIES];
    memset(slots, 0, sizeof(slots));
    for (int d = 0; d < MAX_DIES; d++)
        if (want[d]) {
            die_slot_init(&slots[d], d, ptx, buf_bytes, dst_bytes);
            die_slot_bind_kernel(&slots[d], kernel);
        }
    free(ptx);

    /* Bytes are derived from the iteration count, never estimated: each of the
     * grid*block threads runs `iters` iterations and each iteration loads
     * `unroll` uint4 words of 16 bytes. */
    double bytes_per_launch = (double)grid * block * iters * unroll * 16.0;

    printf("kernel=%s buf=%d MiB grid=%d block=%d iters=%d unroll=%d "
           "launches=%d\n", kernel, buf_mib, grid, block, iters, unroll,
           launches);
    printf("one launch walks %.1f MiB of the %d MiB buffer\n",
           footprint / 1048576.0, buf_mib);
    printf("bytes per die per arm = %.2f GiB\n",
           bytes_per_launch * launches / 1073741824.0);
    printf("\n%-26s %8s %10s %11s %11s %9s\n", "dies", "n", "wall_s",
           "agg_GB/s", "perdie_GB/s", "eff");
    fflush(stdout);

    arm_result base = {0};
    for (int s = 0; s < n_sets; s++) {
        /* Warm the arm, then let the boards shed the warm-up's heat/power so
         * each arm enters from a comparable state rather than the last arm's. */
        run_concurrent_round(slots, sets[s], set_n[s], 4, grid, block, iters);
        sleep_s(settle_s);

        double wall = run_concurrent_round(slots, sets[s], set_n[s], launches,
                                           grid, block, iters);
        arm_result r;
        r.n_dies      = set_n[s];
        r.wall_s      = wall;
        r.bytes_total = bytes_per_launch * launches * set_n[s];
        r.agg_gbs     = r.bytes_total / wall / 1e9;
        r.per_die_gbs = r.agg_gbs / set_n[s];
        if (s == 0)
            base = r;

        char label[128];
        int off = 0;
        for (int i = 0; i < set_n[s] && off < (int)sizeof(label) - 4; i++)
            off += snprintf(label + off, sizeof(label) - off, "%s%d",
                            i ? "," : "", sets[s][i]);

        double eff = base.per_die_gbs > 0 ? r.per_die_gbs / base.per_die_gbs : 0;
        printf("%-26s %8d %10.4f %11.2f %11.2f %8.3f\n", label, r.n_dies,
               r.wall_s, r.agg_gbs, r.per_die_gbs, eff);
        fflush(stdout);
    }

    for (int d = 0; d < MAX_DIES; d++)
        die_slot_teardown(&slots[d]);
    return 0;
}
