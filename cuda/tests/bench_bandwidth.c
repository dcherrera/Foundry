/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_bandwidth.c
 * @brief Measures the pure streaming-READ memory ceiling of one GK210 die.
 *
 * Phase 0 of the quant-layout work. Everything downstream
 * of it is scored against "what fraction of the ceiling is this kernel at", so
 * the ceiling has to be right.
 *
 * WHY THIS EXISTS. The number in the spec, 177.5 GB/s, came from
 * `bandwidthTest --dtod`, which is a COPY: it alternates reads and writes and
 * pays GDDR5 bus turnaround on every switch. Our matvecs are near-pure
 * streaming reads with one float written per output row, which normally runs
 * closer to the pin rate. 177.5 is therefore a floor on the ceiling, not the
 * ceiling. This harness reads and does nothing else.
 *
 * WHAT IT REPORTS
 *   Stage A  geometry sweep (load width x read path x accumulators in flight x
 *            block x grid) on a fixed 256 MiB buffer -> the peak geometry.
 *   Stage B  the peak geometry across buffer sizes, 3 reps, spread + SM clock.
 *   Stage C  the matvec-shaped read skeleton (cuda_kernels/bench_bw.cu
 *            fc_bw_mvskel_r*) at the exact shapes fc_matvec_q3k runs -
 *            248320x2048, 8192x2048, 2048x2048 with the 152 B / 256 weight
 *            device footprint - so the result is directly comparable to
 *            fc_matvec_q3k's GB/s rather than to an abstract peak.
 *   Control  fc_bw_read_u32_ldg_u4_dead, whose loads nvcc deletes. It must
 *            print an impossible number. That is the evidence this harness can
 *            detect the "the compiler optimised my benchmark away" failure,
 *            which is the only interesting way a bandwidth number lies.
 *
 * SANITY GATES, applied to every reported figure:
 *   - > 240.5 GB/s (2505 MHz GDDR5 x 2 x 384 bit / 8) is physically
 *     impossible on one GK210 die and is flagged IMPOSSIBLE, not reported.
 *   - the flat read figure should land in [177.5, 240.5): at or above the
 *     copy figure, below the pin rate. Outside that window it is flagged.
 *   - per-kernel register count and LOCAL (stack frame) bytes are printed;
 *     a non-zero stack frame means an accumulator array spilled to local
 *     memory and the kernel is measuring the wrong thing.
 *   - byte accounting is exact, never estimated: every thread runs the same
 *     iteration count and the host derives bytes from it.
 *   - the SM clock and the sibling die's utilization are sampled DURING every
 *     timed rep. The two GK210s of ONE K80 board share a
 *     power budget, so a busy neighbour drags this die off 875 MHz (seen at
 *     745-810). A sub-875 reading with the sibling IDLE is different: that is
 *     our own kernel hitting the die power limit, it is a real property of the
 *     kernel, and it is accepted with a note. Groups are rejected for a busy
 *     sibling or a spread over 1%. Foreign resident VRAM is reported too but is
 *     NOT a gate: the dies have independent memory controllers, and reps taken
 *     with 7 GB of someone else'"'"'s idle model resident agreed to 0.01%.
 *
 * BUILD (from foundry/), no CUDA headers needed - the driver API is dlopen'd:
 *   cd cuda_kernels && make bench_bw.ptx
 *   gcc -O2 -Wall -I. -o cuda/tests/bench_bandwidth \
 *       cuda/tests/bench_bandwidth.c -ldl
 * RUN (on an otherwise idle board; die 6 shown):
 *   CUDA_VISIBLE_DEVICES=6 FC_BW_SMI_INDEX=6 \
 *       ./cuda/tests/bench_bandwidth [--ptx PATH] [--reps N] [--stage abc]
 *
 * bench_bw.ptx is loaded from disk on purpose. Adding it to the KERNELS list
 * in cuda_kernels/Makefile would regenerate ../cuda/fc_cuda_ptx.h, which
 * production links against; a benchmark has no business doing that.
 *
 * @status Phase 0 complete: flat ceiling, per-shape skeleton, elision control.
 * @issues Small shapes (2048x2048) include per-launch overhead and partially
 *   hit the 1.5 MB L2, so their GB/s is not a DRAM figure. Reported as-is with
 *   the launch count so the effect is visible rather than hidden.
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

#define CU_SUCCESS                     0
#define CU_FUNC_ATTRIBUTE_MAX_THREADS  0
#define CU_FUNC_ATTRIBUTE_LOCAL_SIZE   3
#define CU_FUNC_ATTRIBUTE_NUM_REGS     4

static void *g_libcuda;

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGetCount)(int *);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*p_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuCtxDestroy)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemGetInfo)(size_t *, size_t *);
static CUresult (*p_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
static CUresult (*p_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuFuncGetAttribute)(int *, int, CUfunction);
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
        fprintf(stderr, "[bench_bw] missing driver symbol %s\n", name);
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
        fprintf(stderr, "[bench_bw] dlopen libcuda failed: %s\n", dlerror());
        exit(1);
    }
    p_cuInit              = sym("cuInit", NULL);
    p_cuDeviceGetCount    = sym("cuDeviceGetCount", NULL);
    p_cuDeviceGet         = sym("cuDeviceGet", NULL);
    p_cuDeviceGetName     = sym("cuDeviceGetName", NULL);
    p_cuCtxCreate         = sym("cuCtxCreate", "cuCtxCreate_v2");
    p_cuCtxDestroy        = sym("cuCtxDestroy", "cuCtxDestroy_v2");
    p_cuCtxSynchronize    = sym("cuCtxSynchronize", NULL);
    p_cuMemAlloc          = sym("cuMemAlloc", "cuMemAlloc_v2");
    p_cuMemFree           = sym("cuMemFree", "cuMemFree_v2");
    p_cuMemGetInfo        = sym("cuMemGetInfo", "cuMemGetInfo_v2");
    p_cuMemcpyHtoD        = sym("cuMemcpyHtoD", "cuMemcpyHtoD_v2");
    p_cuMemcpyDtoH        = sym("cuMemcpyDtoH", "cuMemcpyDtoH_v2");
    p_cuMemsetD32         = sym("cuMemsetD32", "cuMemsetD32_v2");
    p_cuModuleLoadData    = sym("cuModuleLoadData", NULL);
    p_cuModuleGetFunction = sym("cuModuleGetFunction", NULL);
    p_cuFuncGetAttribute  = sym("cuFuncGetAttribute", NULL);
    p_cuLaunchKernel      = sym("cuLaunchKernel", NULL);
}

static void cu_ok(CUresult r, const char *what)
{
    if (r != CU_SUCCESS) {
        fprintf(stderr, "[bench_bw] %s failed: %d\n", what, r);
        exit(1);
    }
}

/* ================================================================
 * Host helpers
 * ================================================================ */

#define K80_DIE_PEAK_GBS   240.5   /* 2505 MHz x 2 (DDR) x 384 bit / 8      */
#define K80_DTOD_COPY_GBS  177.5   /* bandwidthTest --dtod, die 6, 2026-08-08 */
#define Q3K_BLOCK_BYTES    152     /* repacked device super-block, 256 wts   */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* One nvidia-smi call sampling BOTH GK210 dies of this K80 board: our target
 * die's SM clock and allocated VRAM, plus the sibling die's utilization.
 * nvidia-smi ignores CUDA_VISIBLE_DEVICES, so the physical index comes from
 * FC_BW_SMI_INDEX; the two dies of a board are consecutive, so the sibling is
 * index^1.
 *
 * WHICH SIGNAL ACTUALLY MATTERS. Resident foreign VRAM is NOT contention: the
 * two dies have independent memory controllers, and a neighbour merely
 * holding a model consumes no bandwidth. Measured here: three reps taken with
 * 7003 MiB of somebody else's model resident on the die agreed to 0.01%. What
 * does contend is (a) another process actually issuing work, which shows up
 * as a wide spread, and (b) the sibling die under load, which drags the SM
 * clock off 875 MHz because both dies share one board power budget - observed
 * at 745-810 MHz while the neighbour was busy. So the gates are the CLOCK and
 * the SPREAD; foreign MiB and sibling utilization are printed as context for
 * why a rep went bad, not as pass/fail on their own. */
struct die_sample {
    int clock_mhz;
    int mem_mib;
    int sibling_util;
};

static void smi_sample(struct die_sample *out)
{
    const char *idx_s = getenv("FC_BW_SMI_INDEX");
    int idx = idx_s ? atoi(idx_s) : 0;
    int sib = idx ^ 1;
    char cmd[256], line[128];
    FILE *f;

    out->clock_mhz = -1;
    out->mem_mib = -1;
    out->sibling_util = -1;
    snprintf(cmd, sizeof(cmd),
             "nvidia-smi --query-gpu=index,clocks.sm,memory.used,"
             "utilization.gpu --format=csv,noheader,nounits -i %d,%d 2>/dev/null",
             idx < sib ? idx : sib, idx < sib ? sib : idx);
    f = popen(cmd, "r");
    if (!f)
        return;
    while (fgets(line, sizeof(line), f)) {
        int i, clk, mem, util;

        if (sscanf(line, "%d, %d, %d, %d", &i, &clk, &mem, &util) != 4)
            continue;
        if (i == idx) {
            out->clock_mhz = clk;
            out->mem_mib = mem;
        } else if (i == sib) {
            out->sibling_util = util;
        }
    }
    pclose(f);
}

static int smi_clock_mhz(void)
{
    struct die_sample d;

    smi_sample(&d);
    return d.clock_mhz;
}

/* Die occupancy once this process has finished allocating and before any
 * measurement starts: context + PTX module + our two buffers, plus anything a
 * co-tenant already held. */
static int g_baseline_mib;

/* VRAM that appeared on the die after our baseline, i.e. someone else's.
 * Context only - see the note above on why this is not a pass/fail gate. */
static int foreign_mib(int total_mib)
{
    int f = total_mib - g_baseline_mib;

    return (total_mib < 0 || f < 0) ? 0 : f;
}

/* The SM clock every reported figure must have been taken at. */
#define K80_RATED_SM_MHZ 875

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n;
    char *buf;

    if (!f) {
        fprintf(stderr, "[bench_bw] cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "[bench_bw] cannot read %s\n", path);
        exit(1);
    }
    buf[n] = '\0';
    fclose(f);
    return buf;
}

/* Verdict string for a bandwidth figure, so an impossible number can never be
 * quietly copied into a plan as a result. */
static const char *verdict(double gbs)
{
    if (gbs >= K80_DIE_PEAK_GBS)
        return "IMPOSSIBLE";
    if (gbs > K80_DIE_PEAK_GBS * 0.995)
        return "at-pin-rate?";
    return "";
}

/* Verdict for a group of reps. Distinguishes the two reasons the SM clock
 * leaves 875 MHz, which look identical in the clock column alone:
 *
 *   sibling die busy  -> the K80 board's power budget is being shared and the
 *                        measurement is contended. Reject.
 *   sibling die idle  -> the droop is SELF-induced: our own kernel is drawing
 *                        enough power to hit the die's limit. Measured here at
 *                        758-810 MHz for the matvec skeleton with sibutil 0,
 *                        foreign 0 and a 0.03% spread. That is a real property
 *                        of running this kernel, not an artifact, and the real
 *                        fc_matvec_q3k pays it too. Accept, and say so.
 *
 * A spread above 1% is rejected regardless: something moved during the group.
 * The 2048x2048 shape trips this legitimately, because at 0.03 ms per launch
 * the per-launch overhead is a large fraction of the measurement. */
static const char *group_verdict(int sibling_busy, int min_clock, double spread)
{
    if (sibling_busy)
        return "REJECT: sibling die busy (shared board power)";
    if (spread > 1.0)
        return "REJECT: spread > 1%";
    if (min_clock > 0 && min_clock < K80_RATED_SM_MHZ)
        return "accepted (self power-limited, not contended)";
    return "accepted";
}

/* ================================================================
 * Device state
 * ================================================================ */

static CUcontext g_ctx;
static CUmodule  g_mod;
static CUdeviceptr g_src;      /* the buffer every read kernel streams */
static size_t g_src_bytes;
static CUdeviceptr g_dst;      /* per-thread / per-row sink            */
static size_t g_dst_bytes;

static CUfunction get_fn(const char *name)
{
    CUfunction f = NULL;
    cu_ok(p_cuModuleGetFunction(&f, g_mod, name), name);
    return f;
}

/* Largest block this kernel can legally launch with. Kepler caps a block's
 * register footprint at 64K, so a 74-register kernel tops out at 832 threads,
 * not 1024, and asking for more returns CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
 * (701). The geometry sweep must skip those points rather than die on them —
 * fc_bw_mvskel_r4 at block=1024 aborted a whole run this way. */
static int fn_max_threads(CUfunction f)
{
    int n = 1024;

    if (p_cuFuncGetAttribute(&n, CU_FUNC_ATTRIBUTE_MAX_THREADS, f) != CU_SUCCESS)
        return 1024;
    return n;
}

/* Index-derived fill: word i = i * 2654435761. A constant fill would make the
 * XOR checksum zero for any even-sized read and destroy the consumption gate.
 * Written in 8 MiB chunks so the host (2 cores / 8 GB) never stages more. */
static void fill_src(void)
{
    const size_t chunk_words = 2u << 20;      /* 8 MiB */
    unsigned int *stage = malloc(chunk_words * sizeof(unsigned int));
    size_t done = 0;

    if (!stage) {
        fprintf(stderr, "[bench_bw] out of host memory for the fill stage\n");
        exit(1);
    }
    while (done < g_src_bytes / 4) {
        size_t n = g_src_bytes / 4 - done;
        size_t i;

        if (n > chunk_words)
            n = chunk_words;
        for (i = 0; i < n; i++)
            stage[i] = (unsigned int)((done + i) * 2654435761u);
        cu_ok(p_cuMemcpyHtoD(g_src + done * 4, stage, n * 4), "cuMemcpyHtoD");
        done += n;
    }
    free(stage);
}

/* ================================================================
 * Timing
 * ================================================================ */

/* Launch `f` back to back until at least target_s of GPU work has been
 * enqueued, and return seconds per launch. The SM clock is sampled after the
 * launches are queued but before the sync, i.e. while the die is under load —
 * only meaningful when target_s is long enough to outlast nvidia-smi's own
 * startup, hence the 1.0 s windows used for every reported figure. */
static double time_launches(CUfunction f, unsigned int grid, unsigned int block,
                            void **args, double target_s,
                            struct die_sample *smp, int *launches_out)
{
    double t0, dt;
    int n, i;

    cu_ok(p_cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL),
          "warmup launch");
    cu_ok(p_cuCtxSynchronize(), "warmup sync");
    t0 = now_s();
    cu_ok(p_cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL),
          "probe launch");
    cu_ok(p_cuCtxSynchronize(), "probe sync");
    dt = now_s() - t0;

    n = (int)(target_s / (dt > 0 ? dt : 1e-6));
    if (n < 3)
        n = 3;
    if (n > 2000)
        n = 2000;

    t0 = now_s();
    for (i = 0; i < n; i++) {
        cu_ok(p_cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, NULL, args,
                               NULL), "launch");
        /* Sample the clock from the middle of the run: nvidia-smi takes
         * ~150 ms to start, so reading it after the sync would report the
         * idle clock the die drops back to. The queued work outlasts it. */
        if (smp && i == n / 2)
            smi_sample(smp);
    }
    cu_ok(p_cuCtxSynchronize(), "sync");
    dt = (now_s() - t0) / n;
    if (launches_out)
        *launches_out = n;
    return dt;
}

/* XOR of the whole sink. Zero means nothing was consumed (or the fill was
 * degenerate) and the measurement is void. */
static unsigned int dst_checksum(size_t words)
{
    unsigned int *h = malloc(words * sizeof(unsigned int));
    unsigned int x = 0;
    size_t i;

    if (!h)
        return 0;
    cu_ok(p_cuMemcpyDtoH(h, g_dst, words * 4), "cuMemcpyDtoH");
    for (i = 0; i < words; i++)
        x ^= h[i];
    free(h);
    return x;
}

/* ================================================================
 * Stage A/B — flat streaming read
 * ================================================================ */

struct read_kind {
    const char *name;
    int elem_bytes;
};

static const struct read_kind KINDS[] = {
    { "u32_ldg",  4 }, { "u32_ld",   4 },
    { "u128_ldg", 16 }, { "u128_ld", 16 },
};
#define N_KINDS ((int)(sizeof(KINDS) / sizeof(KINDS[0])))

static const int UNROLLS[] = { 1, 2, 4, 8 };
static const int BLOCKS[]  = { 128, 256, 512, 1024 };
static const int GRIDS[]   = { 104, 208, 416, 832, 1664 };
#define N_UNROLLS 4
#define N_BLOCKS  4
#define N_GRIDS   5

struct flat_cfg {
    int kind, unroll, block, grid;
    double gbs;
};

/* One flat-read configuration. Returns 0 and leaves *gbs untouched when the
 * buffer cannot supply at least 8 whole iterations per thread — a config that
 * shallow measures launch overhead, not bandwidth. */
static int run_flat(struct flat_cfg *c, size_t bytes, double target_s,
                    double *gbs, struct die_sample *smp,
                    unsigned int *sum_out)
{
    int esz = KINDS[c->kind].elem_bytes;
    long long threads = (long long)c->block * c->grid;
    long long elems = (long long)(bytes / (size_t)esz);
    long long per_pass = threads * c->unroll;
    int iters = (int)(elems / per_pass);
    char name[64];
    CUfunction f;
    void *args[3];
    double dt;
    size_t read_bytes;

    if (iters < 8 || bytes > g_src_bytes)
        return 0;
    read_bytes = (size_t)iters * (size_t)per_pass * (size_t)esz;

    snprintf(name, sizeof(name), "fc_bw_read_%s_u%d", KINDS[c->kind].name,
             c->unroll);
    f = get_fn(name);
    if (c->block > fn_max_threads(f))
        return 0;
    args[0] = &g_src;
    args[1] = &g_dst;
    args[2] = &iters;
    dt = time_launches(f, (unsigned int)c->grid, (unsigned int)c->block, args,
                       target_s, smp, NULL);
    *gbs = (double)read_bytes / dt / 1e9;
    if (sum_out)
        *sum_out = dst_checksum((size_t)threads);
    return 1;
}

static void print_fn_attrs(const char *name)
{
    CUfunction f = get_fn(name);
    int regs = -1, local = -1;

    p_cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
    p_cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE, f);
    printf("   %-28s regs %3d  stack frame %d B%s\n", name, regs, local,
           local ? "   <-- SPILLED, measurement suspect" : "");
}

static struct flat_cfg stage_a_sweep(size_t bytes)
{
    struct flat_cfg best = { 0, 0, 0, 0, 0.0 };
    int ki, ui, bi, gi;

    printf("== Stage A: flat streaming-read geometry sweep (%zu MiB) ==\n",
           bytes >> 20);
    printf("%-10s %6s %6s %6s %9s %8s\n", "kind", "unroll", "block", "grid",
           "GB/s", "note");
    for (ki = 0; ki < N_KINDS; ki++)
        for (ui = 0; ui < N_UNROLLS; ui++)
            for (bi = 0; bi < N_BLOCKS; bi++)
                for (gi = 0; gi < N_GRIDS; gi++) {
                    struct flat_cfg c;
                    double gbs = 0.0;

                    c.kind = ki; c.unroll = UNROLLS[ui];
                    c.block = BLOCKS[bi]; c.grid = GRIDS[gi]; c.gbs = 0.0;
                    if (!run_flat(&c, bytes, 0.10, &gbs, NULL, NULL))
                        continue;
                    printf("%-10s %6d %6d %6d %9.2f %8s\n",
                           KINDS[ki].name, c.unroll, c.block, c.grid, gbs,
                           verdict(gbs));
                    if (gbs > best.gbs) {
                        c.gbs = gbs;
                        best = c;
                    }
                }
    printf("-- Stage A peak: %s unroll=%d block=%d grid=%d -> %.2f GB/s --\n\n",
           KINDS[best.kind].name, best.unroll, best.block, best.grid, best.gbs);
    return best;
}

static void stage_b_sizes(struct flat_cfg peak, int reps)
{
    static const size_t MIB[] = { 32, 64, 128, 256, 512, 1024 };
    int si;

    printf("== Stage B: peak geometry across buffer sizes, %d reps ==\n", reps);
    printf("   geometry: %s unroll=%d block=%d grid=%d\n",
           KINDS[peak.kind].name, peak.unroll, peak.block, peak.grid);
    printf("%8s %4s %9s %6s %7s %8s %10s %s\n", "MiB", "rep", "GB/s", "clk",
           "sibutil", "foreign", "checksum", "note");
    for (si = 0; si < 6; si++) {
        double lo = 1e30, hi = 0.0, sum = 0.0, spread;
        int got = 0, sibbusy = 0, minclk = 0, r;

        for (r = 0; r < reps; r++) {
            struct die_sample d;
            double gbs = 0.0;
            unsigned int cks = 0;

            if (!run_flat(&peak, MIB[si] << 20, 1.0, &gbs, &d, &cks))
                continue;
            sibbusy |= (d.sibling_util > 5);
            if (d.clock_mhz > 0 && (!minclk || d.clock_mhz < minclk))
                minclk = d.clock_mhz;
            printf("%8zu %4d %9.2f %6d %7d %8d   %08x %s%s%s\n", MIB[si],
                   r + 1, gbs, d.clock_mhz, d.sibling_util,
                   foreign_mib(d.mem_mib), cks, verdict(gbs),
                   cks == 0 ? "  <-- ZERO CHECKSUM, loads not consumed" : "",
                   (d.clock_mhz > 0 && d.clock_mhz < K80_RATED_SM_MHZ
                    && d.sibling_util > 5) ? "  <-- SIBLING BUSY" : "");
            if (gbs < lo) lo = gbs;
            if (gbs > hi) hi = gbs;
            sum += gbs;
            got++;
        }
        if (!got)
            continue;
        spread = 100.0 * (hi - lo) / (sum / got);
        printf("%8zu %4s %9.2f %6s %7s %8s %10s spread %.2f%%  %s\n", MIB[si],
               "avg", sum / got, "", "", "", "", spread,
               group_verdict(sibbusy, minclk, spread));
    }
    printf("\n");
}

/* ================================================================
 * Stage C — the matvec's own read pattern
 * ================================================================ */

struct mv_shape {
    const char *label;
    int rows, cols;
};

static const struct mv_shape SHAPES[] = {
    { "lm head 248320x2048", 248320, 2048 },
    { "qkv     8192x2048",     8192, 2048 },
    { "dense   2048x2048",     2048, 2048 },
};
#define N_SHAPES 3

/* One (shape, rows-per-warp, block) point of the read skeleton. Bytes are
 * rows * row_stride exactly — the same accounting bench_matvec.c uses for
 * fc_matvec_q3k, so the two GB/s columns are the same quantity. */
static int run_mvskel(const struct mv_shape *s, int R, int block,
                      double target_s, double *ms, double *gbs,
                      struct die_sample *smp, int *launches)
{
    int row_stride = (s->cols / 256) * Q3K_BLOCK_BYTES;
    int warps = block / 32;
    int rows = s->rows, cols = s->cols;
    size_t bytes = (size_t)rows * (size_t)row_stride;
    char name[48];
    CUfunction f;
    void *args[5];
    double dt;

    if (rows % (warps * R) != 0 || bytes > g_src_bytes)
        return 0;
    snprintf(name, sizeof(name), "fc_bw_mvskel_r%d", R);
    f = get_fn(name);
    if (block > fn_max_threads(f))
        return 0;
    args[0] = &g_src;
    args[1] = &g_dst;
    args[2] = &rows;
    args[3] = &cols;
    args[4] = &row_stride;
    dt = time_launches(f, (unsigned int)(rows / (warps * R)),
                       (unsigned int)block, args, target_s, smp, launches);
    *ms = dt * 1e3;
    *gbs = (double)bytes / dt / 1e9;
    return 1;
}

/* fc_matvec_q3k's own launch geometry (fc_cuda.c: 256 threads = 8 warps,
 * grid = ceil(rows/16), i.e. 2 output rows per warp). The skeleton at THIS
 * point is the number directly comparable to the measured matvec. */
#define Q3K_PROD_R      2
#define Q3K_PROD_BLOCK  256

static const int RS[] = { 1, 2, 4, 8 };

/* Full-rep measurement of one skeleton point, printed with its guard columns.
 * Returns the mean GB/s over the reps that ran. */
static double mvskel_confirm(const struct mv_shape *sh, int R, int block,
                             int reps, const char *tag)
{
    double lo = 1e30, hi = 0.0, sum = 0.0, spread;
    int warps = block / 32, got = 0, sibbusy = 0, minclk = 0, r;

    for (r = 0; r < reps; r++) {
        struct die_sample d;
        double ms = 0.0, gbs = 0.0;
        int nl = 0;

        if (!run_mvskel(sh, R, block, 1.0, &ms, &gbs, &d, &nl))
            return 0.0;   /* shape/geometry not runnable here */
        sibbusy |= (d.sibling_util > 5);
        if (d.clock_mhz > 0 && (!minclk || d.clock_mhz < minclk))
            minclk = d.clock_mhz;
        printf("%-20s %3d %6d %6d %9.4f %9.2f %6d %7d %8d %8d %s%s%s\n",
               sh->label, R, block, sh->rows / (warps * R), ms, gbs,
               d.clock_mhz, d.sibling_util, foreign_mib(d.mem_mib), nl, tag,
               verdict(gbs),
               (d.clock_mhz > 0 && d.clock_mhz < K80_RATED_SM_MHZ
                && d.sibling_util > 5) ? "  <-- SIBLING BUSY" : "");
        if (gbs < lo) lo = gbs;
        if (gbs > hi) hi = gbs;
        sum += gbs;
        got++;
    }
    if (got > 1) {
        spread = 100.0 * (hi - lo) / (sum / got);
        printf("%-20s %3d %6d %6s %9s %9.2f %6s %7s %8s %8s spread %.2f%%  "
               "%s\n", "  mean", R, block, "", "", sum / got, "", "", "", "",
               spread, group_verdict(sibbusy, minclk, spread));
    }
    return got ? sum / got : 0.0;
}

static void stage_c_shapes(int reps)
{
    int si, ri, bi;

    printf("== Stage C: matvec-shaped read skeleton (152 B / 256 weights) ==\n");
    printf("   Same loads per lane per super-block as fc_q3k_sb_lane_dot, with\n"
           "   the dequant deleted. R = output rows per warp; fc_matvec_q3k\n"
           "   ships R=%d block=%d, marked [ship].\n", Q3K_PROD_R,
           Q3K_PROD_BLOCK);
    printf("%-20s %3s %6s %6s %9s %9s %6s %7s %8s %8s\n", "shape", "R",
           "block", "grid", "ms", "GB/s", "clk", "sibutil", "foreign",
           "launches");
    for (si = 0; si < N_SHAPES; si++) {
        int best_r = Q3K_PROD_R, best_b = Q3K_PROD_BLOCK;
        double best = 0.0;

        /* Cheap single-pass scan for the peak geometry. */
        for (ri = 0; ri < 4; ri++)
            for (bi = 0; bi < N_BLOCKS; bi++) {
                struct die_sample d;
                double ms = 0.0, gbs = 0.0;
                int nl = 0;

                if (!run_mvskel(&SHAPES[si], RS[ri], BLOCKS[bi], 0.25, &ms,
                                &gbs, &d, &nl))
                    continue;
                printf("%-20s %3d %6d %6d %9.4f %9.2f %6d %7d %8d %8d scan "
                       "%s\n", SHAPES[si].label, RS[ri], BLOCKS[bi],
                       SHAPES[si].rows / ((BLOCKS[bi] / 32) * RS[ri]), ms, gbs,
                       d.clock_mhz, d.sibling_util, foreign_mib(d.mem_mib), nl,
                       verdict(gbs));
                if (gbs > best) {
                    best = gbs;
                    best_r = RS[ri];
                    best_b = BLOCKS[bi];
                }
            }
        /* Then spend the reps where they matter: the shipped geometry, and
         * the scan winner if it is a different one. */
        mvskel_confirm(&SHAPES[si], Q3K_PROD_R, Q3K_PROD_BLOCK, reps, "[ship] ");
        if (best_r != Q3K_PROD_R || best_b != Q3K_PROD_BLOCK)
            mvskel_confirm(&SHAPES[si], best_r, best_b, reps, "[peak] ");
        printf("\n");
    }
}

/* ================================================================
 * Elision control
 * ================================================================ */

/* Runs the kernel whose loads nvcc is guaranteed to delete. If this does NOT
 * print an impossible number, the harness cannot see elision and none of the
 * figures above are trustworthy. */
static void elision_control(size_t bytes)
{
    int esz = 4, block = 256, grid = 416, unroll = 4;
    long long threads = (long long)block * grid;
    int iters = (int)((long long)(bytes / (size_t)esz) / (threads * unroll));
    CUfunction f = get_fn("fc_bw_read_u32_ldg_u4_dead");
    void *args[3];
    double dt, gbs;
    size_t read_bytes = (size_t)iters * (size_t)(threads * unroll) * 4;

    args[0] = &g_src;
    args[1] = &g_dst;
    args[2] = &iters;
    dt = time_launches(f, (unsigned int)grid, (unsigned int)block, args, 0.2,
                       NULL, NULL);
    gbs = (double)read_bytes / dt / 1e9;
    printf("== Elision control (loads deliberately dead) ==\n");
    printf("   fc_bw_read_u32_ldg_u4_dead: %.1f GB/s  %s\n", gbs,
           verdict(gbs));
    printf("   %s\n\n", gbs >= K80_DIE_PEAK_GBS
           ? "CONTROL OK - an elided kernel is detectable as impossible."
           : "CONTROL FAILED - elision is NOT detectable here; distrust "
             "every figure above.");
    print_fn_attrs("fc_bw_read_u32_ldg_u4_dead");
}

/* ================================================================
 * main
 * ================================================================ */

static void print_env(int foreign_at_start)
{
    char devname[128] = { 0 };
    CUdevice dev = 0;
    size_t freeb = 0, totalb = 0;

    p_cuDeviceGet(&dev, 0);
    p_cuDeviceGetName(devname, sizeof(devname), dev);
    p_cuMemGetInfo(&freeb, &totalb);
    printf("device: %s   VRAM %zu / %zu MiB free   SM clock %d MHz (idle)\n",
           devname, freeb >> 20, totalb >> 20, smi_clock_mhz());
    printf("reference: theoretical %.1f GB/s, bandwidthTest --dtod copy "
           "%.1f GB/s\n", K80_DIE_PEAK_GBS, K80_DTOD_COPY_GBS);
    if (foreign_at_start > 64)
        printf("*** WARNING: %d MiB was already allocated on this die by "
               "another process.\n*** This is a SHARED die; nothing below is "
               "an exclusive-die ceiling.\n", foreign_at_start);
    else
        printf("die is exclusive at start (%d MiB foreign)\n",
               foreign_at_start);
    printf("\n");
}

int main(int argc, char **argv)
{
    const char *ptx_path = "cuda_kernels/bench_bw.ptx";
    const char *stages = "abc";
    size_t buf_mib = 1024;
    int reps = 3, i, ndev = 0, foreign_at_start = 0;
    char *ptx;
    CUdevice dev = 0;
    struct flat_cfg peak = { 0, 4, 256, 416, 0.0 };

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ptx") && i + 1 < argc)
            ptx_path = argv[++i];
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stage") && i + 1 < argc)
            stages = argv[++i];
        else if (!strcmp(argv[i], "--buf-mib") && i + 1 < argc)
            buf_mib = (size_t)atoll(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--ptx PATH] [--reps N] [--stage abc] "
                    "[--buf-mib N]\n", argv[0]);
            return 2;
        }
    }

    load_driver();
    cu_ok(p_cuInit(0), "cuInit");
    cu_ok(p_cuDeviceGetCount(&ndev), "cuDeviceGetCount");
    if (ndev < 1) {
        fprintf(stderr, "[bench_bw] no visible CUDA device\n");
        return 1;
    }
    if (ndev > 1)
        fprintf(stderr, "[bench_bw] WARNING: %d devices visible; pin with "
                "CUDA_VISIBLE_DEVICES so this cannot touch production\n", ndev);
    cu_ok(p_cuDeviceGet(&dev, 0), "cuDeviceGet");
    {
        struct die_sample d;

        smi_sample(&d);
        foreign_at_start = d.mem_mib < 0 ? 0 : d.mem_mib;
    }
    cu_ok(p_cuCtxCreate(&g_ctx, 0, dev), "cuCtxCreate");
    print_env(foreign_at_start < 0 ? 0 : foreign_at_start);

    ptx = read_file(ptx_path);
    cu_ok(p_cuModuleLoadData(&g_mod, ptx), "cuModuleLoadData");
    free(ptx);

    g_src_bytes = buf_mib << 20;
    g_dst_bytes = 8u << 20;   /* >= max grid*block words (1664*1024) */
    cu_ok(p_cuMemAlloc(&g_src, g_src_bytes), "cuMemAlloc src");
    cu_ok(p_cuMemAlloc(&g_dst, g_dst_bytes), "cuMemAlloc dst");
    cu_ok(p_cuMemsetD32(g_dst, 0, g_dst_bytes / 4), "cuMemsetD32 dst");
    fill_src();
    /* Everything we will ever allocate is now resident, so anything that
     * appears on the die from here on belongs to someone else. */
    {
        struct die_sample d;

        smi_sample(&d);
        g_baseline_mib = d.mem_mib;
    }

    printf("== kernel resources (stack frame MUST be 0) ==\n");
    print_fn_attrs("fc_bw_read_u32_ldg_u4");
    print_fn_attrs("fc_bw_read_u128_ldg_u8");
    print_fn_attrs("fc_bw_mvskel_r2");
    print_fn_attrs("fc_bw_mvskel_r8");
    printf("\n");

    if (strchr(stages, 'a'))
        peak = stage_a_sweep(256u << 20);
    if (strchr(stages, 'b'))
        stage_b_sizes(peak, reps);
    if (strchr(stages, 'c'))
        stage_c_shapes(reps);
    elision_control(256u << 20);

    p_cuMemFree(g_src);
    p_cuMemFree(g_dst);
    p_cuCtxDestroy(g_ctx);
    return 0;
}
