/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_handoff_cost.c
 * @brief What one die-to-die boundary hand-off actually costs, at prefill size.
 *
 * WHY THIS EXISTS. An N-die layer split pays one activation copy per boundary
 * per chunk. On the reference 8-card host 4 of the 7 hops are device-to-device over a board's
 * PLX switch and 3 cross boards, where cuMemcpyPeer returns success and moves
 * nothing, so they stage through host RAM instead. Any pipelined prefill
 * inherits those 3 host round trips, and the standing question is whether they
 * cap it. The tree's only figure for the host link is a code comment
 * ("0.4 GB/s"), which is not a measurement.
 *
 * This times the two hop shapes at the size prefill actually moves — a
 * [chunk][n_embd] f32 activation block, 1 MiB at the shipped 128-token chunk
 * and 2048-wide residual — and reports microseconds per hop, so the cost can be
 * put directly against a die-segment's compute time.
 *
 *   peer  : cuMemcpyPeer src -> dst, then sync the receiving context
 *           (exactly gpu_handoff_import's device-to-device branch)
 *   host  : cuMemcpyDtoH into a host buffer then cuMemcpyHtoD into the peer
 *           (exactly gpu_handoff_export + import's staged branch)
 *
 * It also VERIFIES the bytes arrive. A cross-board cuMemcpyPeer on a K80
 * reports CUDA_SUCCESS and copies nothing; timing that would report an
 * absurdly cheap hop for a copy that never happened, which is the single most
 * misleading number this file could produce. Pairs whose peer copy does not
 * move the bytes are reported as NO-PEER rather than timed.
 *
 * BUILD (from foundry/), no CUDA headers needed - the driver API is dlopen'd:
 *   gcc -O2 -Wall -o cuda/tests/bench_handoff_cost \
 *       cuda/tests/bench_handoff_cost.c -ldl
 * RUN:
 *   ./cuda/tests/bench_handoff_cost --bytes 1048576 --reps 200 \
 *       --pairs 0:1,1:2,2:3,3:4,4:5,5:6,6:7
 *
 * @status Reports per-hop us and GB/s for the peer and host-staged routes.
 * @issues None
 * @todo None
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;

#define CU_SUCCESS 0

static void *g_libcuda;

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGetCount)(int *);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuCtxDestroy)(CUcontext);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuCtxEnablePeerAccess)(CUcontext, unsigned int);
static CUresult (*p_cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
static CUresult (*p_cuMemcpyPeer)(CUdeviceptr, CUcontext, CUdeviceptr,
                                  CUcontext, size_t);
static CUresult (*p_cuMemsetD32)(CUdeviceptr, unsigned int, size_t);

static void *sym(const char *name, const char *v2)
{
    void *s = v2 ? dlsym(g_libcuda, v2) : NULL;
    if (!s)
        s = dlsym(g_libcuda, name);
    if (!s) {
        fprintf(stderr, "[handoff] missing driver symbol %s\n", name);
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
        fprintf(stderr, "[handoff] dlopen libcuda failed: %s\n", dlerror());
        exit(1);
    }
    p_cuInit                = sym("cuInit", NULL);
    p_cuDeviceGetCount      = sym("cuDeviceGetCount", NULL);
    p_cuDeviceGet           = sym("cuDeviceGet", NULL);
    p_cuCtxCreate           = sym("cuCtxCreate", "cuCtxCreate_v2");
    p_cuCtxDestroy          = sym("cuCtxDestroy", "cuCtxDestroy_v2");
    p_cuCtxSetCurrent       = sym("cuCtxSetCurrent", NULL);
    p_cuCtxSynchronize      = sym("cuCtxSynchronize", NULL);
    p_cuCtxEnablePeerAccess = sym("cuCtxEnablePeerAccess", NULL);
    p_cuMemAlloc            = sym("cuMemAlloc", "cuMemAlloc_v2");
    p_cuMemFree             = sym("cuMemFree", "cuMemFree_v2");
    p_cuMemcpyDtoH          = sym("cuMemcpyDtoH", "cuMemcpyDtoH_v2");
    p_cuMemcpyHtoD          = sym("cuMemcpyHtoD", "cuMemcpyHtoD_v2");
    p_cuMemcpyPeer          = sym("cuMemcpyPeer", NULL);
    p_cuMemsetD32           = sym("cuMemsetD32", "cuMemsetD32_v2");
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/**
 * Does a peer copy between these two contexts actually move bytes?
 *
 * A cross-board K80 pair answers CUDA_SUCCESS and copies nothing, so the only
 * trustworthy check is to poison the destination, copy, and read it back. This
 * mirrors gpu_peer_link_moves_bytes() in the engine.
 *
 * @return 1 if the bytes arrived, 0 otherwise.
 */
static int peer_moves_bytes(CUcontext a, CUcontext b, CUdeviceptr src,
                            CUdeviceptr dst, size_t bytes)
{
    unsigned int pattern = 0xA5A5A5A5u;
    size_t probe = bytes < 256 ? bytes : 256;

    p_cuCtxSetCurrent(a);
    p_cuMemsetD32(src, pattern, probe / 4);
    p_cuCtxSynchronize();
    p_cuCtxSetCurrent(b);
    p_cuMemsetD32(dst, 0xDEADBEEFu, probe / 4);
    p_cuCtxSynchronize();

    if (p_cuMemcpyPeer(dst, b, src, a, bytes) != CU_SUCCESS)
        return 0;
    p_cuCtxSetCurrent(b);
    p_cuCtxSynchronize();

    unsigned int *back = malloc(probe);
    if (!back)
        return 0;
    p_cuMemcpyDtoH(back, dst, probe);
    int ok = 1;
    for (size_t i = 0; i < probe / 4; i++)
        if (back[i] != pattern) {
            ok = 0;
            break;
        }
    free(back);
    return ok;
}

int main(int argc, char **argv)
{
    size_t bytes = 1048576;              /* 128 tokens x 2048 embd x f32 */
    int reps = 200;
    const char *pairs = "0:1,1:2,2:3,3:4,4:5,5:6,6:7";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bytes") && i + 1 < argc)
            bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pairs") && i + 1 < argc)
            pairs = argv[++i];
        else {
            printf("usage: bench_handoff_cost [--bytes N] [--reps N] "
                   "[--pairs a:b,c:d,...]\n");
            return 1;
        }
    }

    load_driver();
    if (p_cuInit(0) != CU_SUCCESS) {
        fprintf(stderr, "[handoff] cuInit failed\n");
        return 1;
    }
    int n_dev = 0;
    p_cuDeviceGetCount(&n_dev);

    void *host = malloc(bytes);
    if (!host) {
        fprintf(stderr, "[handoff] host alloc failed\n");
        return 1;
    }
    memset(host, 0, bytes);

    printf("hop size %zu bytes (%.2f MiB), %d reps\n", bytes,
           bytes / 1048576.0, reps);
    printf("%-8s %-10s %12s %10s %12s %10s\n", "pair", "peer?", "peer_us",
           "peer_GB/s", "host_us", "host_GB/s");

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", pairs);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int a = -1, b = -1;
        if (sscanf(tok, "%d:%d", &a, &b) != 2)
            continue;
        if (a < 0 || b < 0 || a >= n_dev || b >= n_dev) {
            printf("%d:%-6d out-of-range\n", a, b);
            continue;
        }

        CUdevice da, db;
        CUcontext ca, cb;
        CUdeviceptr sa, sb;
        p_cuDeviceGet(&da, a);
        p_cuDeviceGet(&db, b);
        if (p_cuCtxCreate(&ca, 0, da) != CU_SUCCESS) continue;
        if (p_cuCtxCreate(&cb, 0, db) != CU_SUCCESS) { p_cuCtxDestroy(ca); continue; }

        p_cuCtxSetCurrent(ca);
        p_cuMemAlloc(&sa, bytes);
        p_cuCtxEnablePeerAccess(cb, 0);
        p_cuCtxSetCurrent(cb);
        p_cuMemAlloc(&sb, bytes);
        p_cuCtxEnablePeerAccess(ca, 0);

        int have_peer = peer_moves_bytes(ca, cb, sa, sb, bytes);

        double peer_us = 0.0;
        if (have_peer) {
            /* warm */
            for (int r = 0; r < 5; r++) {
                p_cuMemcpyPeer(sb, cb, sa, ca, bytes);
                p_cuCtxSetCurrent(cb);
                p_cuCtxSynchronize();
            }
            double t0 = now_s();
            for (int r = 0; r < reps; r++) {
                p_cuMemcpyPeer(sb, cb, sa, ca, bytes);
                p_cuCtxSetCurrent(cb);
                p_cuCtxSynchronize();
            }
            peer_us = (now_s() - t0) / reps * 1e6;
        }

        /* Host-staged route: D2H off the sender then H2D into the receiver,
         * both blocking, which is exactly what the engine does. */
        for (int r = 0; r < 5; r++) {
            p_cuCtxSetCurrent(ca);
            p_cuMemcpyDtoH(host, sa, bytes);
            p_cuCtxSetCurrent(cb);
            p_cuMemcpyHtoD(sb, host, bytes);
        }
        double t1 = now_s();
        for (int r = 0; r < reps; r++) {
            p_cuCtxSetCurrent(ca);
            p_cuMemcpyDtoH(host, sa, bytes);
            p_cuCtxSetCurrent(cb);
            p_cuMemcpyHtoD(sb, host, bytes);
        }
        double host_us = (now_s() - t1) / reps * 1e6;

        printf("%d:%-6d %-10s %12.2f %10.2f %12.2f %10.2f\n", a, b,
               have_peer ? "yes" : "NO-PEER",
               peer_us, have_peer ? bytes / (peer_us * 1e-6) / 1e9 : 0.0,
               host_us, bytes / (host_us * 1e-6) / 1e9);
        fflush(stdout);

        p_cuCtxSetCurrent(ca);
        p_cuMemFree(sa);
        p_cuCtxSetCurrent(cb);
        p_cuMemFree(sb);
        p_cuCtxDestroy(ca);
        p_cuCtxDestroy(cb);
    }

    free(host);
    return 0;
}
