/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_attn_qtile.c
 * @brief Driver-API microbenchmark for the Q_TILE x split-KV sweep on the
 *   prefill flash-attention kernel (sm_37 / K80 / GK210).
 *
 * Same instrument as the exp14/exp25 benches that predicted in-model gains to
 * within 0.2-1.7 %: a raw driver-API program that opens exactly ONE device,
 * loads a PTX module whose baseline symbol is the production kernel #include'd
 * verbatim (cuda_kernels/tests/exp_attn_qtile.cu), and times variants with CUDA
 * events, best-of-N after a discarded warm-up. Production shape:
 *   npos=128, n_head=16, n_head_kv=2 (kv_groups=8), head_dim=256,
 *   q4 KV with sink=4 / win=128, attn_win=0 (full attention).
 *
 * Measurement discipline this file enforces, because this project has shipped
 * green-but-vacuous gates before:
 *   - the shipped kernel is re-timed as a CONTROL immediately before every arm,
 *     so drift shows up as control drift and every ratio is control-relative;
 *   - a NOISE FLOOR series times the control N times back to back first;
 *   - every timed arm can be gated on a board entry TEMPERATURE read from a
 *     file an external sampler writes (XQ_TEMP_FILE / XQ_TEMP_MAX), because a
 *     heat-soaked K80 die moves 100 MHz between runs 11 minutes apart;
 *   - a deliberately BROKEN arm (xq_broken) must fail the bit-exactness check,
 *     which is how the check itself is shown to be capable of failing.
 *
 * Build (on a K80 host with nvcc/cuobjdump):
 *   nvcc -ptx -arch=sm_37 -ccbin gcc-11 -isystem <foundry>/cuda_kernels/nvcc_shim \
 *        -I<foundry>/cuda_kernels -o exp_attn_qtile.ptx \
 *        <foundry>/cuda_kernels/tests/exp_attn_qtile.cu
 *   gcc -O2 -o bench_attn_qtile bench_attn_qtile.c -I/usr/local/cuda/include \
 *        -L/usr/lib/x86_64-linux-gnu -lcuda -lm
 * Run:
 *   ./bench_attn_qtile <ptxfile> <device> [reps] [npos]
 *
 * @status Microbenchmark only; never linked into Pantheon or libfoundry.
 * @issues None
 * @todo None
 */

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NPOSMAX   1024
#define NHEAD     16
#define NHEADKV   2
#define GROUPS    (NHEAD / NHEADKV)
#define HEADDIM   256
#define MAXSEQ    32768
#define ROWBYTES  ((HEADDIM / 32) * 18)
#define TILEKEYS  16
#define KPAD      16
#define WMAX      256
#define SINKMAX   8
#define Q4SINK    4
#define Q4WIN     128
#define THREADS   HEADDIM
#define SHARED_MAX 49152          /* sm_37 per-block dynamic-shared ceiling */

static int NPOS = 128;

#define CHECK(x)                                                              \
    do {                                                                      \
        CUresult _e = (x);                                                    \
        if (_e != CUDA_SUCCESS) {                                             \
            const char *_s = NULL;                                            \
            cuGetErrorName(_e, &_s);                                          \
            fprintf(stderr, "FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x,  \
                    _s ? _s : "?");                                           \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

static CUmodule mod;
static CUdeviceptr d_q, d_k, d_v, d_ksink, d_vsink, d_kwin, d_vwin, d_out;
static float *h_out, *h_ref;

/* One benchmark arm: a symbol plus the launch geometry it needs. */
struct arm {
    const char *name;
    int qt;              /* query rows per block (grid.x = NPOS / qt) */
    int kreg;            /* 1 = no shared K tile */
    int split;           /* grid.z KV slices; >1 means PARTIAL output */
    int pad;             /* extra dynamic shared, an occupancy dial */
    const char *note;
};

static unsigned long long rng = 88172645463325252ULL;
static double urand(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (double)(rng >> 11) / 9007199254740992.0;
}

static unsigned short f2h(float f)
{
    unsigned int x; memcpy(&x, &f, 4);
    unsigned int sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    unsigned int man = (x >> 13) & 0x3ff;
    if (exp <= 0) return (unsigned short)sign;
    if (exp >= 31) return (unsigned short)(sign | 0x7bff);
    return (unsigned short)(sign | ((unsigned)exp << 10) | man);
}

static CUfunction getf(const char *n)
{
    CUfunction f = NULL;
    if (cuModuleGetFunction(&f, mod, n) != CUDA_SUCCESS) return NULL;
    return f;
}

/* Dynamic-shared bytes one block of an arm needs. Mirrors the shared layout in
 * exp_attn_qtile.cu: q group for QT rows, the padded K tile unless it lives in
 * registers, then scores + probs + m/l/corr per (row, group). */
static int shbytes(const struct arm *a)
{
    int qg = a->qt * GROUPS;
    size_t f = (size_t)qg * HEADDIM +
               (a->kreg ? 0u : (size_t)TILEKEYS * (HEADDIM + KPAD)) +
               2u * (size_t)qg * TILEKEYS + 3u * (size_t)qg;
    return (int)(f * sizeof(float)) + a->pad;
}

static int blocks_per_smx(const char *name, int shared)
{
    CUfunction f = getf(name);
    int occ = 0;
    if (!f) return -1;
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, THREADS, shared);
    return occ;
}

/* regs / LOCAL_SIZE_BYTES / residency of the LOADED function. LOCAL_SIZE_BYTES
 * is the number that matters: a runtime-bounded array is not a "spill", so
 * ptxas prints 0 spill bytes beside a nonzero stack frame. */
static void dump_attrs(const char *name, int shared)
{
    CUfunction f = getf(name);
    if (!f) { printf("  %-14s <absent>\n", name); return; }
    int regs = 0, sh = 0, loc = 0, occ = 0;
    cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, f);
    cuFuncGetAttribute(&sh, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f);
    cuFuncGetAttribute(&loc, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f);
    cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, THREADS, shared);
    printf("  %-14s regs=%3d LOCAL=%3dB statshared=%d dynshared=%6d "
           "blocks/SMX=%d warps=%2d occ=%3.0f%%%s\n",
           name, regs, loc, sh, shared, occ, occ * 8,
           occ * 8 * 100.0 / 64.0, loc ? "   <-- LOCAL MEMORY, NOT REGISTERS" : "");
}

/* Board entry-temperature gate. An external sampler writes the die's
 * temperature (one integer, degrees C) to XQ_TEMP_FILE; nothing is timed until
 * it is at or below XQ_TEMP_MAX. Without both env vars the gate is inert and
 * says so in the header, so a run cannot silently skip it. */
static const char *temp_file = NULL;
static int temp_max = 0;
static int temp_waits = 0, temp_wait_s = 0, temp_last = -1;

static int read_temp(void)
{
    FILE *f;
    int t = -1;
    if (!temp_file) return -1;
    f = fopen(temp_file, "r");
    if (!f) return -1;
    if (fscanf(f, "%d", &t) != 1) t = -1;
    fclose(f);
    return t;
}

static void temp_gate(void)
{
    int waited = 0, t;
    if (!temp_file || temp_max <= 0) return;
    for (;;) {
        t = read_temp();
        if (t < 0 || t <= temp_max) break;
        sleep(2);
        waited += 2;
        if (waited >= 300) break;            /* never hang a run forever */
    }
    temp_last = t;
    if (waited) { temp_waits++; temp_wait_s += waited; }
}

/* Time one arm: warm-up launch, then best-of-reps with CUDA events.
 * save_out: 0 none, 1 into h_out, 2 into h_ref. */
static double time_arm(const struct arm *a, int pos0, int reps, int save_out)
{
    CUfunction f = getf(a->name);
    int kv_groups = GROUPS, head_dim = HEADDIM, max_seq = MAXSEQ, npos = NPOS;
    int sink = Q4SINK, win = Q4WIN, attn_sink = 4, attn_win = 0;
    float scale = 1.0f / sqrtf((float)HEADDIM);
    int shared = shbytes(a);
    unsigned int gx, gz;
    CUevent e0, e1;
    double best = 1e30;
    int r;
    /* The shipped kernel predates the npos argument; the variants take it. */
    void *args_prod[] = { &d_q, &d_k, &d_v, &d_ksink, &d_vsink, &d_kwin,
                          &d_vwin, &d_out, &kv_groups, &head_dim, &max_seq,
                          &pos0, &sink, &win, &attn_sink, &attn_win, &scale };
    void *args_var[]  = { &d_q, &d_k, &d_v, &d_ksink, &d_vsink, &d_kwin,
                          &d_vwin, &d_out, &kv_groups, &head_dim, &max_seq,
                          &pos0, &npos, &sink, &win, &attn_sink, &attn_win,
                          &scale };
    void **args = (strncmp(a->name, "xq_", 3) == 0) ? args_var : args_prod;

    if (!f) return -1.0;
    if (shared > SHARED_MAX) return -2.0;
    if (NPOS % a->qt) return -3.0;
    gx = (unsigned int)(NPOS / a->qt);
    gz = (unsigned int)a->split;

    temp_gate();
    CHECK(cuEventCreate(&e0, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&e1, CU_EVENT_DEFAULT));
    CHECK(cuLaunchKernel(f, gx, NHEADKV, gz, THREADS, 1, 1,
                         (unsigned int)shared, 0, args, NULL));
    CHECK(cuCtxSynchronize());
    for (r = 0; r < reps; r++) {
        float ms = 0.0f;
        CHECK(cuEventRecord(e0, 0));
        CHECK(cuLaunchKernel(f, gx, NHEADKV, gz, THREADS, 1, 1,
                             (unsigned int)shared, 0, args, NULL));
        CHECK(cuEventRecord(e1, 0));
        CHECK(cuEventSynchronize(e1));
        CHECK(cuEventElapsedTime(&ms, e0, e1));
        if (ms / 1000.0 < best) best = ms / 1000.0;
    }
    if (save_out) {
        size_t n = (size_t)NPOS * NHEAD * HEADDIM;
        CHECK(cuMemcpyDtoH(save_out == 2 ? h_ref : h_out, d_out,
                           n * sizeof(float)));
    }
    cuEventDestroy(e0); cuEventDestroy(e1);
    return best;
}

/* memcmp the full NPOS x NHEAD x HEADDIM output against the shipped kernel's. */
static int cmp_ref(const char *name)
{
    size_t n = (size_t)NPOS * NHEAD * HEADDIM, i, nbad = 0;
    double maxrel = 0.0, sum2 = 0.0, ref2 = 0.0;
    for (i = 0; i < n; i++) {
        double a = (double)h_ref[i], b = (double)h_out[i];
        sum2 += (a - b) * (a - b);
        ref2 += a * a;
        if (h_out[i] != h_ref[i]) {
            double rel = fabs(a) > 0 ? fabs(a - b) / fabs(a) : fabs(a - b);
            nbad++;
            if (rel > maxrel) maxrel = rel;
        }
    }
    printf("  %-14s %-10s mismatching floats %7zu/%zu  max rel %.3g  "
           "rel L2 %.3g\n", name, nbad == 0 ? "BIT-EXACT" : "DIFFERS",
           nbad, n, maxrel, ref2 > 0 ? sqrt(sum2 / ref2) : 0.0);
    return nbad == 0;
}

static const struct arm PROD = { "fc_attn_flash_prefill_q4", 1, 0, 1, 0,
                                 "shipped kernel" };

int main(int argc, char **argv)
{
    int dev, reps, smx, regsmx, shsmx, clk, i, ok, fails = 0;
    CUdevice cd;
    CUcontext ctx;
    char nm[128];
    FILE *fp;
    long sz;
    char *ptx;
    size_t qn, kvn, winn, sinkn;
    float *hq;
    unsigned char *hk;
    unsigned short *hw, *hs;
    double worst_lo, worst_hi;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <ptx> <device> [reps] [npos]\n", argv[0]);
        return 1;
    }
    dev = atoi(argv[2]);
    reps = argc > 3 ? atoi(argv[3]) : 3;
    NPOS = argc > 4 ? atoi(argv[4]) : 128;
    if (NPOS < 1 || NPOS > NPOSMAX) { fprintf(stderr, "npos 1..%d\n", NPOSMAX); return 1; }
    temp_file = getenv("XQ_TEMP_FILE");
    temp_max = getenv("XQ_TEMP_MAX") ? atoi(getenv("XQ_TEMP_MAX")) : 0;

    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&cd, dev));
    CHECK(cuDeviceGetName(nm, sizeof nm, cd));
    smx = regsmx = shsmx = clk = 0;
    cuDeviceGetAttribute(&smx, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cd);
    cuDeviceGetAttribute(&regsmx, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, cd);
    cuDeviceGetAttribute(&shsmx, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cd);
    cuDeviceGetAttribute(&clk, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, cd);
    CHECK(cuCtxCreate(&ctx, 0, cd));
    printf("device %d: %s  SMX=%d regs/SMX=%d shared/SMX=%d clk=%d kHz  "
           "NPOS=%d reps=%d\n", dev, nm, smx, regsmx, shsmx, clk, NPOS, reps);
    printf("temperature gate: %s (file=%s max=%dC)\n",
           (temp_file && temp_max > 0) ? "ARMED" : "INERT",
           temp_file ? temp_file : "-", temp_max);

    fp = fopen(argv[1], "rb");
    if (!fp) { perror(argv[1]); return 1; }
    fseek(fp, 0, SEEK_END); sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    ptx = malloc((size_t)sz + 1);
    if (!ptx || fread(ptx, 1, (size_t)sz, fp) != (size_t)sz) { perror("read"); return 1; }
    ptx[sz] = 0; fclose(fp);
    CHECK(cuModuleLoadData(&mod, ptx));

    qn = (size_t)NPOSMAX * NHEAD * HEADDIM;
    kvn = (size_t)NHEADKV * MAXSEQ * ROWBYTES;
    winn = (size_t)NHEADKV * WMAX * HEADDIM;
    sinkn = (size_t)NHEADKV * SINKMAX * HEADDIM;
    CHECK(cuMemAlloc(&d_q, qn * 4));
    CHECK(cuMemAlloc(&d_out, qn * 4));
    CHECK(cuMemAlloc(&d_k, kvn));
    CHECK(cuMemAlloc(&d_v, kvn));
    CHECK(cuMemAlloc(&d_kwin, winn * 2));
    CHECK(cuMemAlloc(&d_vwin, winn * 2));
    CHECK(cuMemAlloc(&d_ksink, sinkn * 2));
    CHECK(cuMemAlloc(&d_vsink, sinkn * 2));

    hq = malloc(qn * 4);
    for (size_t j = 0; j < qn; j++) hq[j] = (float)(urand() * 2.0 - 1.0) * 0.5f;
    CHECK(cuMemcpyHtoD(d_q, hq, qn * 4));
    hk = malloc(kvn);
    for (size_t b = 0; b < kvn; b += 18) {
        unsigned short d = f2h(0.05f);
        hk[b] = d & 0xff; hk[b + 1] = (d >> 8) & 0xff;
        for (int j = 0; j < 16; j++)
            hk[b + 2 + j] = (unsigned char)(rng = rng * 6364136223846793005ULL + 1, (rng >> 33) & 0xff);
    }
    CHECK(cuMemcpyHtoD(d_k, hk, kvn));
    for (size_t b = 0; b < kvn; b += 18) {
        unsigned short d = f2h(0.05f);
        hk[b] = d & 0xff; hk[b + 1] = (d >> 8) & 0xff;
        for (int j = 0; j < 16; j++)
            hk[b + 2 + j] = (unsigned char)(rng = rng * 6364136223846793005ULL + 1, (rng >> 29) & 0xff);
    }
    CHECK(cuMemcpyHtoD(d_v, hk, kvn));
    hw = malloc(winn * 2);
    for (size_t j = 0; j < winn; j++) hw[j] = f2h((float)(urand() * 2.0 - 1.0) * 0.4f);
    CHECK(cuMemcpyHtoD(d_kwin, hw, winn * 2));
    for (size_t j = 0; j < winn; j++) hw[j] = f2h((float)(urand() * 2.0 - 1.0) * 0.4f);
    CHECK(cuMemcpyHtoD(d_vwin, hw, winn * 2));
    hs = malloc(sinkn * 2);
    for (size_t j = 0; j < sinkn; j++) hs[j] = f2h((float)(urand() * 2.0 - 1.0) * 0.4f);
    CHECK(cuMemcpyHtoD(d_ksink, hs, sinkn * 2));
    for (size_t j = 0; j < sinkn; j++) hs[j] = f2h((float)(urand() * 2.0 - 1.0) * 0.4f);
    CHECK(cuMemcpyHtoD(d_vsink, hs, sinkn * 2));
    h_out = malloc(qn * 4); h_ref = malloc(qn * 4);

    /* ---- the sweep ------------------------------------------------------ */
    struct arm V[] = {
        { "xq_q1_sh",    1, 0, 1, 0, "template control, shared K" },
        { "xq_q1_krn",   1, 1, 1, 0, "K in registers, per-element dequant" },
        { "xq_q1_kr",    1, 1, 1, 0, "K in registers, hoisted dequant" },
        { "xq_q2_sh",    2, 0, 1, 0, "Q_TILE=2, K in shared" },
        { "xq_q2_krn",   2, 1, 1, 0, "Q_TILE=2, K in registers, naive dq" },
        { "xq_q2_kr",    2, 1, 1, 0, "Q_TILE=2, K in registers" },
        { "xq_q4_kr",    4, 1, 1, 0, "Q_TILE=4, K in registers" },
        { "xq_q1_kr_s2", 1, 1, 2, 0, "Q_TILE=1 + split-KV 2 (PARTIAL, bound)" },
        { "xq_q1_kr_s4", 1, 1, 4, 0, "Q_TILE=1 + split-KV 4 (PARTIAL, bound)" },
        { "xq_q2_kr_s2", 2, 1, 2, 0, "Q_TILE=2 + split-KV 2 (PARTIAL, bound)" },
        { "xq_q4_kr_s2", 4, 1, 2, 0, "Q_TILE=4 + split-KV 2 (PARTIAL, bound)" },
        { "xq_q4_kr_s4", 4, 1, 4, 0, "Q_TILE=4 + split-KV 4 (PARTIAL, bound)" },
        { "xq_broken",   1, 1, 1, 0, "FALSIFICATION arm, must not be bit-exact" },
        { NULL, 0, 0, 0, 0, NULL }
    };

    printf("\n=== driver-JIT attributes of the LOADED functions ===\n");
    dump_attrs(PROD.name, shbytes(&PROD));
    for (i = 0; V[i].name; i++) dump_attrs(V[i].name, shbytes(&V[i]));

    printf("\n=== noise floor: the shipped kernel, %d back-to-back timings "
           "at pos0=8192 ===\n", 9);
    worst_lo = 1e30; worst_hi = 0.0;
    for (i = 0; i < 9; i++) {
        double s = time_arm(&PROD, 8192, reps, 0);
        if (s < worst_lo) worst_lo = s;
        if (s > worst_hi) worst_hi = s;
        printf("  %d: %9.6f s%s\n", i, s, temp_last >= 0 ? "" : "");
    }
    printf("  noise floor: min %.6f  max %.6f  spread %+.3f %%\n",
           worst_lo, worst_hi, 100.0 * (worst_hi - worst_lo) / worst_lo);

    printf("\n=== the sweep at pos0=8192 (control re-timed before every arm) "
           "===\n");
    printf("  %-14s %6s %6s %5s %7s %4s %11s %11s %8s  %s\n", "arm", "shared",
           "blocks", "grid", "blk/SMX", "wave", "control", "arm", "gain",
           "note");
    for (i = 0; V[i].name; i++) {
        double s, c;
        int shared = shbytes(&V[i]), nblk, b, waves;
        c = time_arm(&PROD, 8192, reps, 0);
        s = time_arm(&V[i], 8192, reps, 0);
        if (s == -1.0) { printf("  %-14s <absent>\n", V[i].name); continue; }
        if (s == -2.0) {
            printf("  %-14s shared %d B EXCEEDS the sm_37 %d B per-block "
                   "ceiling -- cannot launch  (%s)\n", V[i].name, shared,
                   SHARED_MAX, V[i].note);
            continue;
        }
        if (s == -3.0) { printf("  %-14s npos %d not divisible by QT %d\n",
                                V[i].name, NPOS, V[i].qt); continue; }
        nblk = (NPOS / V[i].qt) * NHEADKV * V[i].split;
        b = blocks_per_smx(V[i].name, shared);
        waves = b > 0 ? (nblk + b * smx - 1) / (b * smx) : 0;
        printf("  %-14s %6d %6d %5d %7d %4d %11.6f %11.6f %7.3fx  %s\n",
               V[i].name, shared, nblk, NPOS / V[i].qt, b, waves, c, s, c / s,
               V[i].note);
    }

    /* ---- bit-exactness against the shipped kernel ----------------------- */
    printf("\n=== bit-exactness vs %s at pos0=8192, full %d-float output ===\n",
           PROD.name, NPOS * NHEAD * HEADDIM);
    time_arm(&PROD, 8192, 1, 2);
    for (i = 0; V[i].name; i++) {
        if (!getf(V[i].name) || shbytes(&V[i]) > SHARED_MAX) continue;
        if (V[i].split > 1) {
            printf("  %-14s (split-KV writes PARTIALS, no combine pass -- not "
                   "comparable by construction)\n", V[i].name);
            continue;
        }
        time_arm(&V[i], 8192, 1, 1);
        ok = cmp_ref(V[i].name);
        if (strcmp(V[i].name, "xq_broken") == 0) {
            printf("  ---> falsification: the broken arm MUST differ. %s\n",
                   ok ? "IT DID NOT -- THE GATE IS VACUOUS" : "It does. Gate is live.");
            if (ok) fails++;
        } else if (!ok) {
            fails++;
        }
    }

    /* ---- occupancy dial on the best Q-tiled arm ------------------------- */
    printf("\n=== shared-size -> blocks/SMX map (driver, on xq_q2_kr) ===\n");
    if (getf("xq_q2_kr")) {
        int prev = -1, shx;
        for (shx = 8192; shx <= SHARED_MAX; shx += 32) {
            int b = blocks_per_smx("xq_q2_kr", shx);
            if (b != prev) { printf("  >= %6d B : %d blocks/SMX\n", shx, b); prev = b; }
        }
    }
    printf("\n=== occupancy dial (identical code, only the dynamic-shared "
           "request moves) ===\n");
    {
        const char *dial[] = { "xq_q1_kr", "xq_q2_kr", "xq_q4_kr", NULL };
        int d;
        for (d = 0; dial[d]; d++) {
            struct arm a = { dial[d], d == 0 ? 1 : (d == 1 ? 2 : 4), 1, 1, 0, "" };
            int nat, target;
            if (!getf(dial[d])) continue;
            nat = shbytes(&a);
            printf("  %s (natural %d B, grid %d blocks)\n", dial[d], nat,
                   (NPOS / a.qt) * NHEADKV);
            for (target = 8; target >= 2; target--) {
                int shx, found = -1;
                double s;
                int waves;
                for (shx = nat; shx <= SHARED_MAX; shx += 32)
                    if (blocks_per_smx(dial[d], shx) == target) { found = shx; break; }
                if (found < 0) continue;
                a.pad = found - nat;
                s = time_arm(&a, 8192, reps, 0);
                waves = ((NPOS / a.qt) * NHEADKV + target * smx - 1) / (target * smx);
                if (s > 0)
                    printf("    blocks/SMX=%d warps=%2d shared=%6d waves=%d  "
                           "%9.6f s\n", target, target * 8, found, waves, s);
                a.pad = 0;
            }
        }
    }

    /* ---- depth sweep ---------------------------------------------------- */
    printf("\n=== depth sweep (control re-timed at every depth) ===\n");
    {
        int depths[] = { 0, 1024, 4096, 8192, 16384, 31872 };
        const char *nmv[] = { "xq_q1_kr", "xq_q2_kr", "xq_q4_kr", NULL };
        int qtv[] = { 1, 2, 4 };
        printf("  %-8s %11s", "pos0", "ship");
        for (i = 0; nmv[i]; i++) printf(" %11s %7s", nmv[i], "gain");
        printf("\n");
        for (int dp = 0; dp < 6; dp++) {
            double c = time_arm(&PROD, depths[dp], reps, 0);
            printf("  %-8d %11.6f", depths[dp], c);
            for (i = 0; nmv[i]; i++) {
                struct arm a = { nmv[i], qtv[i], 1, 1, 0, "" };
                double s = getf(nmv[i]) ? time_arm(&a, depths[dp], reps, 0) : -1.0;
                if (s > 0) printf(" %11.6f %6.3fx", s, c / s);
                else printf(" %11s %7s", "-", "-");
            }
            printf("\n");
        }
    }

    printf("\ntemperature gate: %d waits, %d s total, last reading %d C\n",
           temp_waits, temp_wait_s, temp_last);
    printf("%s\n", fails ? "RESULT: FAILURES PRESENT" : "RESULT: all gates as expected");
    printf("DONE\n");
    return fails ? 1 : 0;
}
