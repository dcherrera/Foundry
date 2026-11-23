/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tp_xboard_copy.c
 * @brief Cross-board die-to-die copies staged through host memory (see the header).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "cuda/tests/tp_xboard_copy.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* fc_buffer is opaque in fc_cuda.h; this mirrors fc_cuda.c's definition so a mapped
 * pinned host range can be handed to fc_copy_2d. Bench-only: never fc_buffer_free one. */
struct fc_buffer {
    unsigned long long dptr;
    size_t size;
    fc_cuda_ctx *owner;
};

#define XB_MAXD 4
#define XB_EVR  1024

typedef int (*pfn_ctx_get)(void **);
typedef int (*pfn_dev_get)(int *, int);
typedef int (*pfn_primary_retain)(void **, int);
typedef int (*pfn_ctx_push)(void *);
typedef int (*pfn_ctx_pop)(void **);
typedef int (*pfn_host_alloc)(void **, size_t, unsigned int);
typedef int (*pfn_host_devptr)(unsigned long long *, void *, unsigned int);

#define CU_MEMHOSTALLOC_PORTABLE   0x01
#define CU_MEMHOSTALLOC_DEVICEMAP  0x02

static const char *const XB_NAMES[XB_N] = { "drv", "ce", "zc", "zcce", "cezc" };

static fc_cuda_ctx *xb_ctx[XB_MAXD];
static int          xb_nd;
static void        *xb_host[XB_MAXD][2];     /* pinned staging, host address */
static fc_buffer   *xb_hbuf[XB_MAXD][2];     /* the same range, owned by die d */
static size_t       xb_slot_bytes;
static fc_event    *xb_ev[XB_MAXD][XB_EVR];
static int          xb_evi[XB_MAXD];

int xb_parse(const char *name)
{
    int m;
    for (m = 0; name && m < XB_N; m++)
        if (!strcmp(name, XB_NAMES[m]))
            return m;
    return -1;
}

const char *xb_name(int method)
{
    return (method >= 0 && method < XB_N) ? XB_NAMES[method] : "?";
}

static fc_buffer *xb_wrap(fc_cuda_ctx *owner, unsigned long long dptr, size_t bytes)
{
    fc_buffer *b = malloc(sizeof(*b));
    if (!b)
        return NULL;
    b->dptr = dptr;
    b->size = bytes;
    b->owner = owner;
    return b;
}

/* One portable, device-mapped pinned range. A context must be current: the caller's,
 * else device 0's primary context (portable memory serves every context). */
static void *xb_host_alloc(size_t bytes, unsigned long long *dptr)
{
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    pfn_ctx_get ctx_get;
    pfn_dev_get dev_get;
    pfn_primary_retain retain;
    pfn_ctx_push push;
    pfn_ctx_pop pop;
    pfn_host_alloc halloc;
    pfn_host_devptr hdev;
    void *cur = NULL, *h = NULL;
    int pushed = 0;

    if (!lib)
        return NULL;
    ctx_get = (pfn_ctx_get)dlsym(lib, "cuCtxGetCurrent");
    dev_get = (pfn_dev_get)dlsym(lib, "cuDeviceGet");
    retain = (pfn_primary_retain)dlsym(lib, "cuDevicePrimaryCtxRetain");
    push = (pfn_ctx_push)dlsym(lib, "cuCtxPushCurrent_v2");
    pop = (pfn_ctx_pop)dlsym(lib, "cuCtxPopCurrent_v2");
    halloc = (pfn_host_alloc)dlsym(lib, "cuMemHostAlloc");
    hdev = (pfn_host_devptr)dlsym(lib, "cuMemHostGetDevicePointer_v2");
    if (!ctx_get || !dev_get || !retain || !push || !pop || !halloc || !hdev)
        return NULL;
    if (ctx_get(&cur) != 0 || !cur) {
        int dev;
        void *prim = NULL;
        if (dev_get(&dev, 0) != 0 || retain(&prim, dev) != 0 || push(prim) != 0)
            return NULL;
        pushed = 1;
    }
    if (halloc(&h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP) != 0)
        h = NULL;
    if (h && (hdev(dptr, h, 0) != 0 || *dptr != (unsigned long long)(size_t)h)) {
        fprintf(stderr, "xb: mapped pinned memory has no UVA alias\n");
        h = NULL;   /* leaked on purpose: bench exits */
    }
    if (pushed)
        pop(&cur);
    return h;
}

int xb_init(fc_cuda_ctx *const *ctx, int nd, size_t slot_bytes)
{
    int d, s, j;
    if (nd > XB_MAXD)
        nd = XB_MAXD;
    xb_nd = nd;
    xb_slot_bytes = slot_bytes;
    for (d = 0; d < nd; d++) {
        xb_ctx[d] = ctx[d];
        for (s = 0; s < 2; s++) {
            unsigned long long dp = 0;
            xb_host[d][s] = xb_host_alloc(slot_bytes, &dp);
            if (!xb_host[d][s])
                return -1;
            memset(xb_host[d][s], 0, slot_bytes);
            xb_hbuf[d][s] = xb_wrap(ctx[d], dp, slot_bytes);
            if (!xb_hbuf[d][s])
                return -1;
        }
        for (j = 0; j < XB_EVR; j++)
            if (!(xb_ev[d][j] = fc_event_create_ctx(ctx[d])))
                return -1;
    }
    return 0;
}

static fc_event *xb_record(int d)
{
    fc_event *e = xb_ev[d][xb_evi[d]++ % XB_EVR];
    fc_event_record_ctx(xb_ctx[d], e);
    return e;
}

/* Stage write on die a: src range -> slot (copy engine or a's SMs). */
static int xb_stage_write(int method, int a, fc_buffer *src, size_t src_off, size_t bytes,
                          int slot)
{
    int rc;
    if (method == XB_CE || method == XB_CEZC) {
        fc_buffer *v = fc_buffer_offset_view(src, src_off);
        rc = v ? fc_buffer_to_cpu_n_async(v, xb_host[a][slot], bytes) : -1;
        fc_buffer_free_view(v);
        return rc;
    }
    return fc_buffer_copy_2d_kernel_async(xb_hbuf[a][slot], 0, bytes, src, src_off, bytes,
                                          bytes, 1, xb_ctx[a]);
}

/* Stage read on die b: slot -> dst range (copy engine or b's SMs). */
static int xb_stage_read(int method, int a, int b, fc_buffer *dst, size_t dst_off,
                         size_t bytes, int slot)
{
    int rc;
    if (method == XB_CE || method == XB_ZCCE) {
        fc_buffer *v = fc_buffer_offset_view(dst, dst_off);
        rc = v ? fc_buffer_from_cpu_n_async(v, xb_host[a][slot], bytes) : -1;
        fc_buffer_free_view(v);
        return rc;
    }
    return fc_buffer_copy_2d_kernel_async(dst, dst_off, bytes, xb_hbuf[a][slot], 0, bytes,
                                          bytes, 1, xb_ctx[b]);
}

int xb_copy(int method, int a, fc_buffer *src, size_t src_off, int b, fc_buffer *dst,
            size_t dst_off, size_t bytes, fc_event *ev_ready, int slot)
{
    if (a < 0 || a >= xb_nd || b < 0 || b >= xb_nd || bytes > xb_slot_bytes)
        return -1;
    slot &= 1;
    if (method == XB_DRV) {
        fc_buffer *s = fc_buffer_offset_view(src, src_off);
        fc_buffer *t = fc_buffer_offset_view(dst, dst_off);
        int rc = (s && t && fc_event_wait_ctx(xb_ctx[b], ev_ready ? ev_ready : xb_record(a)) == 0)
                     ? fc_buffer_copy_peer_async(t, s, bytes, xb_ctx[b]) : -1;
        fc_buffer_free_view(s);
        fc_buffer_free_view(t);
        return rc;
    }
    /* ev_ready is on a's own stream, so stream order already covers the write. */
    if (xb_stage_write(method, a, src, src_off, bytes, slot) != 0)
        return -1;
    if (fc_event_wait_ctx(xb_ctx[b], xb_record(a)) != 0)
        return -1;
    return xb_stage_read(method, a, b, dst, dst_off, bytes, slot);
}

/* ------------------------------------------------------------------
 * mode xbv
 * ------------------------------------------------------------------ */

static double xb_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int xb_cmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double xb_median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(double), xb_cmp);
    return v[n / 2];
}

static void xb_sync_all(void)
{
    int d;
    for (d = 0; d < xb_nd; d++)
        fc_sync_ctx(xb_ctx[d]);
}

/* One copy a -> b of bytes, byte-checked. */
static int xb_verify_one(int m, int a, int b, fc_buffer *const *src, fc_buffer *const *dst,
                         size_t bytes)
{
    size_t i, nf = bytes / sizeof(float);
    float *h = malloc(bytes), *g = malloc(bytes);
    int ok = 1;
    if (!h || !g)
        return 0;
    for (i = 0; i < nf; i++)
        h[i] = (float)(i * 3 + (size_t)m * 100000 + (size_t)a * 7);
    fc_buffer_from_cpu_n(src[a], h, bytes);
    memset(g, 0, bytes);
    fc_buffer_from_cpu_n(dst[b], g, bytes);
    xb_sync_all();
    if (xb_copy(m, a, src[a], 0, b, dst[b], 0, bytes, NULL, 0) != 0)
        ok = 0;
    xb_sync_all();
    fc_buffer_to_cpu_n(dst[b], g, bytes);
    if (ok && memcmp(g, h, bytes) != 0)
        ok = 0;
    free(h);
    free(g);
    return ok;
}

/* Dependent swap chain over npairs cross pairs (0<->2, then 1<->3): us per swap. Each
 * die's next send follows its previous receive in its own stream order. */
static double xb_swap_chain(int m, int npairs, fc_buffer *const *src, fc_buffer *const *dst,
                            size_t bytes, int len)
{
    static const int A[2] = { 0, 1 }, B[2] = { 2, 3 };
    int k, q;
    double t0;
    xb_sync_all();
    t0 = xb_now();
    for (k = 0; k < len; k++)
        for (q = 0; q < npairs; q++) {
            xb_copy(m, A[q], src[A[q]], 0, B[q], dst[B[q]], 0, bytes, NULL, k);
            xb_copy(m, B[q], src[B[q]], 0, A[q], dst[A[q]], 0, bytes, NULL, k);
        }
    xb_sync_all();
    return (xb_now() - t0) * 1e6 / len;
}

/* One-way copies 0 -> 2 back to back: GB/s (timing only; slots are reused unguarded). */
static double xb_oneway(int m, fc_buffer *const *src, fc_buffer *const *dst, size_t bytes,
                        int len)
{
    int k;
    double t0;
    xb_sync_all();
    t0 = xb_now();
    for (k = 0; k < len; k++)
        xb_copy(m, 0, src[0], 0, 2, dst[2], 0, bytes, NULL, k);
    xb_sync_all();
    return (double)bytes * len / (xb_now() - t0) / 1e9;
}

int xb_mode_verify(fc_cuda_ctx *const *ctx, int nd)
{
    /* Per-direction bytes: TP4 hierarchical cross messages at n = 1 (out 1280, core 1536,
     * reduce 2560 floats a die pair), their n = 4 sizes, then bandwidth sizes. */
    static const size_t SZ[] = { 5120, 6144, 10240, 20480, 24576, 40960, 81920, 262144,
                                 1048576 };
    const int nsz = (int)(sizeof(SZ) / sizeof(SZ[0]));
    fc_buffer *src[XB_MAXD], *dst[XB_MAXD];
    int m, si, d, r, bad = 0, reps = 5;

    if (nd < 4) {
        fprintf(stderr, "xbv needs 4 dies (two boards)\n");
        return 2;
    }
    for (d = 0; d < 4; d++) {
        src[d] = fc_buffer_alloc_ctx(ctx[d], SZ[nsz - 1]);
        dst[d] = fc_buffer_alloc_ctx(ctx[d], SZ[nsz - 1]);
        if (!src[d] || !dst[d])
            return 2;
    }
    printf("cross-board copies through host memory; dies 0,1 = board A, 2,3 = board B\n"
           "swap = both directions of a pair at once, dependent chain, median of %d\n"
           "%-5s %8s %6s %6s %10s %10s %9s\n", reps, "meth", "bytes", "0->2", "2->0",
           "swap1_us", "swap2_us", "1way_GBs");
    for (m = 0; m < XB_N; m++)
        for (si = 0; si < nsz; si++) {
            size_t bytes = SZ[si];
            int len = bytes >= 262144 ? 16 : 128;
            int ok1 = xb_verify_one(m, 0, 2, src, dst, bytes);
            int ok2 = xb_verify_one(m, 2, 0, src, dst, bytes);
            double t1[8], t2[8], bw[8];
            xb_swap_chain(m, 2, src, dst, bytes, 8);   /* warm */
            for (r = 0; r < reps; r++) {
                t1[r] = xb_swap_chain(m, 1, src, dst, bytes, len);
                t2[r] = xb_swap_chain(m, 2, src, dst, bytes, len);
                bw[r] = xb_oneway(m, src, dst, bytes, len);
            }
            bad |= !ok1 || !ok2;
            printf("%-5s %8zu %6s %6s %10.1f %10.1f %9.3f\n", xb_name(m), bytes,
                   ok1 ? "OK" : "FAIL", ok2 ? "OK" : "FAIL", xb_median(t1, reps),
                   xb_median(t2, reps), xb_median(bw, reps));
            fflush(stdout);
        }
    return bad;
}
