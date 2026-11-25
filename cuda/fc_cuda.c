/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* The k-quant block repack (fc_repack_*) is modelled on ggml (MIT License,
 * Copyright (c) 2023-2026 The ggml authors, https://github.com/ggml-org/ggml),
 * modified. See NOTICE. */
/*
 * fc_cuda.c — CUDA GPU compute backend for Foundry
 *
 * Zero compile-time CUDA dependency: all CUDA symbols loaded via dlopen/dlsym.
 * Targets CUDA 11.4 Driver API + cuBLAS for K80 (sm_37).
 *
 * Architecture:
 *   1. dlopen("libcuda.so.1") → CUDA driver API symbols
 *   2. dlopen("libcublas.so.11") → cuBLAS SGEMM/GEMV
 *   3. Embedded PTX (from fc_cuda_ptx.h) → cuModuleLoadData → cuLaunchKernel
 *   4. Falls back gracefully if CUDA unavailable
 *
 * Per-context model (multi-die):
 *   The dlopen handles + driver/cuBLAS function-pointer table are
 *   PROCESS-GLOBAL — they are loaded once per process and shared by every
 *   context. All per-device state (CUcontext, stream, cuBLAS handle, PTX
 *   modules, kernel functions) lives in an fc_cuda_ctx, one per K80 die,
 *   so a single process can drive several dies concurrently. Each device
 *   op pushes its context current and pops it on return, so contexts for
 *   different dies never bleed into each other.
 *
 *   Back-compat: a process-default ctx is created on first fc_init() and
 *   the original non-ctx functions delegate to it, leaving single-die
 *   callers (e.g. Pantheon rwkvx.c) unchanged.
 *
 * @status Per-context refactor complete; device paths validated on Mac by
 *         compile only — runtime validation pending on K80 hardware.
 * @issues None
 * @todo None
 */

#include "cuda/fc_cuda.h"
#include "cuda/fc_cuda_ptx.h"

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

/* ================================================================
 * CUDA Driver API types (defined locally — no cuda.h dependency)
 * ================================================================ */

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_NOT_READY 600
#define CU_EVENT_DISABLE_TIMING 0x2
#define FC_CU_MEMHOSTALLOC_PORTABLE 0x01u   /* CU_MEMHOSTALLOC_PORTABLE */

/* CUstreamCaptureMode — see fc_graph_begin_ctx for why we do not use GLOBAL. */
#define CU_STREAM_CAPTURE_MODE_GLOBAL       0
#define CU_STREAM_CAPTURE_MODE_THREAD_LOCAL 1
#define CU_EVENT_RECORD_EXTERNAL 0x1
#define CU_STREAM_CAPTURE_MODE_RELAXED      2

/* cuBLAS types */
typedef int cublasStatus_t;
typedef void *cublasHandle_t;

#define CUBLAS_STATUS_SUCCESS 0

/* cuBLAS operation types (column-major conventions) */
typedef enum {
    CUBLAS_OP_N = 0,  /* no transpose */
    CUBLAS_OP_T = 1,  /* transpose */
    CUBLAS_OP_C = 2   /* conjugate transpose */
} cublasOperation_t;

/* ================================================================
 * Function pointer typedefs
 * ================================================================ */

/* CUDA Driver API */
typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGetCount)(int *);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*pfn_cuDeviceGetName)(char *, int, CUdevice);
typedef CUresult (*pfn_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*pfn_cuCtxDestroy)(CUcontext);
typedef CUresult (*pfn_cuCtxSynchronize)(void);
typedef CUresult (*pfn_cuCtxPushCurrent)(CUcontext);
typedef CUresult (*pfn_cuCtxPopCurrent)(CUcontext *);
typedef CUresult (*pfn_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*pfn_cuMemFree)(CUdeviceptr);
typedef CUresult (*pfn_cuMemHostAlloc)(void **, size_t, unsigned int);
typedef CUresult (*pfn_cuMemFreeHost)(void *);
typedef CUresult (*pfn_cuMemGetInfo)(size_t *, size_t *);
typedef CUresult (*pfn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
typedef CUresult (*pfn_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
typedef CUresult (*pfn_cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*pfn_cuMemcpyPeer)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
typedef CUresult (*pfn_cuMemcpyPeerAsync)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);
typedef CUresult (*pfn_cuDeviceGetAttribute)(int *, int, CUdevice);
#define CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD          84
#define CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID 85
/* CUDA_MEMCPY3D_PEER (cuda.h ABI); fc_buffer_copy_2d_async uses it with Depth 1. */
#define CU_MEMORYTYPE_DEVICE 2
typedef struct {
    size_t srcXInBytes, srcY, srcZ, srcLOD;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    void *srcArray;
    CUcontext srcContext;
    size_t srcPitch, srcHeight;
    size_t dstXInBytes, dstY, dstZ, dstLOD;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    void *dstArray;
    CUcontext dstContext;
    size_t dstPitch, dstHeight;
    size_t WidthInBytes, Height, Depth;
} fc_memcpy3d_peer;
typedef CUresult (*pfn_cuMemcpy3DPeerAsync)(const fc_memcpy3d_peer *, CUstream);
typedef CUresult (*pfn_cuDeviceCanAccessPeer)(int *, CUdevice, CUdevice);
typedef CUresult (*pfn_cuCtxEnablePeerAccess)(CUcontext, unsigned int);
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*pfn_cuModuleUnload)(CUmodule);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*pfn_cuLaunchKernel)(CUfunction,
    unsigned int, unsigned int, unsigned int,   /* grid  */
    unsigned int, unsigned int, unsigned int,   /* block */
    unsigned int, CUstream, void **, void **);
typedef CUresult (*pfn_cuStreamCreate)(CUstream *, unsigned int);
typedef CUresult (*pfn_cuStreamDestroy)(CUstream);
typedef CUresult (*pfn_cuStreamSynchronize)(CUstream);

/* CUDA graph capture / replay (optional — decode is launch-bound, so a
 * captured graph replays hundreds of launches with one call). */
typedef void *CUgraph;
typedef void *CUgraphExec;
typedef void *CUgraphNode;
typedef CUresult (*pfn_cuStreamBeginCapture)(CUstream, int /*CUstreamCaptureMode*/);
typedef CUresult (*pfn_cuStreamEndCapture)(CUstream, CUgraph *);
typedef CUresult (*pfn_cuGraphInstantiate)(CUgraphExec *, CUgraph,
                                           CUgraphNode *, char *, size_t);
typedef CUresult (*pfn_cuGraphLaunch)(CUgraphExec, CUstream);
typedef CUresult (*pfn_cuGraphDestroy)(CUgraph);
typedef CUresult (*pfn_cuGraphExecDestroy)(CUgraphExec);

/* CUDA events (optional — the async serve scheduler's non-blocking
 * completion probe; C2). */
typedef CUresult (*pfn_cuEventCreate)(CUevent *, unsigned int);
typedef CUresult (*pfn_cuEventRecord)(CUevent, CUstream);
typedef CUresult (*pfn_cuEventQuery)(CUevent);
typedef CUresult (*pfn_cuEventSynchronize)(CUevent);
typedef CUresult (*pfn_cuEventDestroy)(CUevent);
typedef CUresult (*pfn_cuStreamWaitEvent)(CUstream, CUevent, unsigned int);

/* Timing events and graph node census (optional symbols). */
typedef CUresult (*pfn_cuEventRecordWithFlags)(CUevent, CUstream, unsigned int);
typedef CUresult (*pfn_cuEventElapsedTime)(float *, CUevent, CUevent);
typedef CUresult (*pfn_cuStreamIsCapturing)(CUstream, int *);
typedef CUresult (*pfn_cuGraphGetNodes)(CUgraph, CUgraphNode *, size_t *);
typedef CUresult (*pfn_cuGraphNodeGetType)(CUgraphNode, int *);

/* cuBLAS */
typedef cublasStatus_t (*pfn_cublasCreate)(cublasHandle_t *);
typedef cublasStatus_t (*pfn_cublasDestroy)(cublasHandle_t);
typedef cublasStatus_t (*pfn_cublasSetStream)(cublasHandle_t, CUstream);
typedef cublasStatus_t (*pfn_cublasSgemm)(cublasHandle_t,
    cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float *, const float *, int,
    const float *, int,
    const float *, float *, int);
typedef cublasStatus_t (*pfn_cublasSgemv)(cublasHandle_t,
    cublasOperation_t,
    int, int,
    const float *, const float *, int,
    const float *, int,
    const float *, float *, int);

/* ================================================================
 * Process-global driver state
 *
 * The dlopen handles and the entire driver/cuBLAS function-pointer
 * table are loaded once per process and shared by all contexts. They
 * are NOT per-device. fc_driver_load() populates this; it is the only
 * global mutable state in the file besides the process-default ctx.
 * ================================================================ */

static bool  _fc_driver_loaded = false;
static void *_fc_libcuda       = NULL;
static void *_fc_libcublas     = NULL;
static bool  _fc_cublas_loaded = false;   /* cuBLAS symbols resolved? */

/* CUDA Driver API function pointers */
static pfn_cuInit              p_cuInit              = NULL;
static pfn_cuDeviceGetCount    p_cuDeviceGetCount    = NULL;
static pfn_cuDeviceGet         p_cuDeviceGet         = NULL;
static pfn_cuDeviceGetName     p_cuDeviceGetName     = NULL;
static pfn_cuCtxCreate         p_cuCtxCreate         = NULL;
static pfn_cuCtxDestroy        p_cuCtxDestroy        = NULL;
static pfn_cuCtxSynchronize    p_cuCtxSynchronize    = NULL;
static pfn_cuCtxPushCurrent    p_cuCtxPushCurrent    = NULL;
static pfn_cuCtxPopCurrent     p_cuCtxPopCurrent     = NULL;
static pfn_cuMemAlloc          p_cuMemAlloc          = NULL;
static pfn_cuMemFree           p_cuMemFree           = NULL;
static pfn_cuMemHostAlloc      p_cuMemHostAlloc      = NULL;   /* optional: pinned host memory */
static pfn_cuMemFreeHost       p_cuMemFreeHost       = NULL;
static pfn_cuMemGetInfo        p_cuMemGetInfo        = NULL;
static pfn_cuMemcpyHtoD        p_cuMemcpyHtoD        = NULL;
static pfn_cuMemcpyDtoH        p_cuMemcpyDtoH        = NULL;
static pfn_cuMemcpyDtoHAsync   p_cuMemcpyDtoHAsync   = NULL;
static pfn_cuMemcpyHtoDAsync   p_cuMemcpyHtoDAsync   = NULL;
static pfn_cuModuleLoadData    p_cuModuleLoadData    = NULL;
static pfn_cuMemcpyDtoDAsync   p_cuMemcpyDtoDAsync   = NULL;
static pfn_cuMemcpyPeer         p_cuMemcpyPeer         = NULL;
static pfn_cuMemcpyPeerAsync    p_cuMemcpyPeerAsync    = NULL;
static pfn_cuDeviceGetAttribute p_cuDeviceGetAttribute = NULL;
static pfn_cuMemcpy3DPeerAsync  p_cuMemcpy3DPeerAsync  = NULL;
static pfn_cuDeviceCanAccessPeer p_cuDeviceCanAccessPeer = NULL;
static pfn_cuCtxEnablePeerAccess p_cuCtxEnablePeerAccess = NULL;
static pfn_cuModuleUnload      p_cuModuleUnload      = NULL;
static pfn_cuModuleGetFunction p_cuModuleGetFunction = NULL;
static pfn_cuLaunchKernel      p_cuLaunchKernel      = NULL;
static pfn_cuStreamCreate      p_cuStreamCreate      = NULL;
static pfn_cuStreamDestroy     p_cuStreamDestroy     = NULL;
static pfn_cuStreamSynchronize p_cuStreamSynchronize = NULL;

/* Context scheduling. The driver default (flags 0 = CU_CTX_SCHED_AUTO) picks YIELD when
 * there are more contexts than CPUs, so every synchronize spin-waits on sched_yield: on a
 * 2-core host driving 8 dies that spin took most of the serve's CPU (Pantheon
 * docs/audits/2026-10-30-host-cpu-bottleneck-audit.md, fix 4). FC_CUDA_BLOCKING_SYNC=1 makes
 * a waiting thread sleep on a driver event instead (CU_CTX_SCHED_BLOCKING_SYNC). Scheduling
 * only: results are identical. Default ON since a 2026-10-31 A/B (8 concurrent streams: host CPU
 * 165-200% -> 0-25%, same throughput); FC_CUDA_BLOCKING_SYNC=0 restores the driver default. */
#define FC_CU_CTX_SCHED_BLOCKING_SYNC 0x04u   /* CU_CTX_SCHED_BLOCKING_SYNC */

static unsigned int fc_ctx_create_flags(void)
{
    const char *e = getenv("FC_CUDA_BLOCKING_SYNC");
    return (e && e[0] == '0') ? 0u : FC_CU_CTX_SCHED_BLOCKING_SYNC;
}

/* Graph capture/replay — OPTIONAL symbols (NULL on drivers without graph
 * support; fc_graph_begin_ctx then reports unavailable and callers keep
 * their direct launch paths). */
static pfn_cuStreamBeginCapture p_cuStreamBeginCapture = NULL;
static pfn_cuStreamEndCapture   p_cuStreamEndCapture   = NULL;
static pfn_cuGraphInstantiate   p_cuGraphInstantiate   = NULL;
static pfn_cuGraphLaunch        p_cuGraphLaunch        = NULL;
static pfn_cuGraphDestroy       p_cuGraphDestroy       = NULL;
static pfn_cuGraphExecDestroy   p_cuGraphExecDestroy   = NULL;

/* CUDA events — OPTIONAL symbols (NULL on drivers without them; fc_event_*
 * then report unavailable and callers keep their blocking fc_sync_ctx). */
static pfn_cuEventCreate  p_cuEventCreate  = NULL;
static pfn_cuEventRecord  p_cuEventRecord  = NULL;
static pfn_cuEventQuery   p_cuEventQuery   = NULL;
static pfn_cuEventSynchronize p_cuEventSynchronize = NULL;
static pfn_cuEventDestroy p_cuEventDestroy = NULL;
static pfn_cuStreamWaitEvent  p_cuStreamWaitEvent  = NULL;
static pfn_cuEventRecordWithFlags p_cuEventRecordWithFlags = NULL;
static pfn_cuEventElapsedTime p_cuEventElapsedTime = NULL;
static pfn_cuStreamIsCapturing p_cuStreamIsCapturing = NULL;
static pfn_cuGraphGetNodes p_cuGraphGetNodes = NULL;
static pfn_cuGraphNodeGetType p_cuGraphNodeGetType = NULL;

/* Call census (fc_call_census_enable): counts of the driver calls that cost host time, taken
 * only while fc_call_census_enable(1) is in force. Off, each wrapper is one branch. */
static int fc_census_on;
static fc_call_census fc_census;
#define FC_CENSUS_ADD(field, v) \
    do { if (fc_census_on) __atomic_fetch_add(&fc_census.field, (unsigned long long)(v), \
                                              __ATOMIC_RELAXED); } while (0)
static CUresult fcw_cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
                                   unsigned int bx, unsigned int by, unsigned int bz,
                                   unsigned int shm, CUstream st, void **args, void **extra)
{
    FC_CENSUS_ADD(launches, 1);
    return p_cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shm, st, args, extra);
}
static CUresult fcw_cuMemcpyHtoD(CUdeviceptr d, const void *h, size_t n)
{
    FC_CENSUS_ADD(h2d_sync, 1);
    FC_CENSUS_ADD(h2d_bytes, n);
    return p_cuMemcpyHtoD(d, h, n);
}
static CUresult fcw_cuMemcpyDtoH(void *h, CUdeviceptr d, size_t n)
{
    FC_CENSUS_ADD(d2h_sync, 1);
    FC_CENSUS_ADD(d2h_bytes, n);
    return p_cuMemcpyDtoH(h, d, n);
}
static CUresult fcw_cuMemcpyHtoDAsync(CUdeviceptr d, const void *h, size_t n, CUstream st)
{
    FC_CENSUS_ADD(h2d_async, 1);
    FC_CENSUS_ADD(h2d_bytes, n);
    return p_cuMemcpyHtoDAsync(d, h, n, st);
}
static CUresult fcw_cuMemcpyDtoHAsync(void *h, CUdeviceptr d, size_t n, CUstream st)
{
    FC_CENSUS_ADD(d2h_async, 1);
    FC_CENSUS_ADD(d2h_bytes, n);
    return p_cuMemcpyDtoHAsync(h, d, n, st);
}
static CUresult fcw_cuMemcpyDtoDAsync(CUdeviceptr d, CUdeviceptr s, size_t n, CUstream st)
{
    FC_CENSUS_ADD(d2d, 1);
    FC_CENSUS_ADD(d2d_bytes, n);
    return p_cuMemcpyDtoDAsync(d, s, n, st);
}
static CUresult fcw_cuMemcpyPeer(CUdeviceptr d, CUcontext dc, CUdeviceptr s, CUcontext sc,
                                 size_t n)
{
    FC_CENSUS_ADD(peer, 1);
    FC_CENSUS_ADD(peer_bytes, n);
    return p_cuMemcpyPeer(d, dc, s, sc, n);
}
static CUresult fcw_cuMemcpyPeerAsync(CUdeviceptr d, CUcontext dc, CUdeviceptr s, CUcontext sc,
                                      size_t n, CUstream st)
{
    FC_CENSUS_ADD(peer, 1);
    FC_CENSUS_ADD(peer_bytes, n);
    return p_cuMemcpyPeerAsync(d, dc, s, sc, n, st);
}
static CUresult fcw_cuStreamSynchronize(CUstream st)
{
    FC_CENSUS_ADD(syncs, 1);
    return p_cuStreamSynchronize(st);
}
static CUresult fcw_cuCtxSynchronize(void)
{
    FC_CENSUS_ADD(syncs, 1);
    return p_cuCtxSynchronize();
}
static CUresult fcw_cuEventSynchronize(CUevent e)
{
    FC_CENSUS_ADD(syncs, 1);
    return p_cuEventSynchronize(e);
}
static CUresult fcw_cuGraphLaunch(CUgraphExec x, CUstream st)
{
    FC_CENSUS_ADD(graph_launches, 1);
    return p_cuGraphLaunch(x, st);
}

void fc_call_census_enable(int on)
{
    fc_census_on = on != 0;
}

void fc_call_census_read(fc_call_census *out)
{
    if (!out)
        return;
    unsigned long long *dst = (unsigned long long *)out;
    unsigned long long *src = (unsigned long long *)&fc_census;
    for (size_t i = 0; i < sizeof(*out) / sizeof(*dst); i++)
        dst[i] = __atomic_load_n(&src[i], __ATOMIC_RELAXED);
}

/* cuBLAS function pointers */
static pfn_cublasCreate    p_cublasCreate    = NULL;
static pfn_cublasDestroy   p_cublasDestroy   = NULL;
static pfn_cublasSetStream p_cublasSetStream = NULL;
static pfn_cublasSgemm     p_cublasSgemm     = NULL;
static pfn_cublasSgemv     p_cublasSgemv     = NULL;

/* ================================================================
 * Per-device context (one per K80 die)
 * ================================================================ */

/* Q8_0 parity tiles (fc_matvec_q80p_r<r>c<nc> in matvec_kquants.cu): rows and columns per
 * thread and the compact x floats staged per column (the kernel's CHUNK). The shipped
 * PTX holds the seven fc_q80p_tile_for picks; the rest load only from a sweep build. */
static const struct { int r, nc, chunk; } fc_q80p_tiles[] = {
    { 1, 1, 2048 }, { 2, 1, 2048 }, { 1, 2, 1024 }, { 2, 2, 1024 }, { 2, 3, 512 },
    { 2, 4, 512 }, { 2, 5, 512 }, { 4, 3, 512 }, { 4, 4, 512 }, { 4, 5, 512 },
    { 4, 8, 512 }, { 2, 8, 512 },
};
#define FC_Q80P_NKERN ((int)(sizeof(fc_q80p_tiles) / sizeof(fc_q80p_tiles[0])))

struct fc_cuda_ctx {
    int        device_id;   /* CUDA device ordinal this ctx is bound to */
    bool       ready;       /* fully initialized? */

    CUdevice   device;
    CUcontext  context;
    CUstream   stream;
    CUevent    sync_ev;     /* fc_sync_ctx's completion point under FC_CUDA_WAIT; NULL = none */

    cublasHandle_t cublas;  /* NULL if cuBLAS unavailable */

    /* PTX modules — one copy per context */
    CUmodule   mod_elem;    /* elementwise.ptx */
    CUmodule   mod_norm;    /* normalization.ptx */
    CUmodule   mod_q8;      /* matvec_q8.ptx */
    CUmodule   mod_q3k;     /* matvec_q3k.ptx */
    CUmodule   mod_kq;      /* matvec_kquants.ptx (Q4_K/Q5_K/Q6_K/Q8_0/F32) */
    CUmodule   mod_rope;    /* rope.ptx */
    CUmodule   mod_moe;     /* moe_routing.ptx */
    CUmodule   mod_moegrp;  /* moe_expert_group.ptx */
    CUmodule   mod_ssm;     /* ssm_gated_delta.ptx */
    CUmodule   mod_fused;   /* fused_layer.ptx */
    CUmodule   mod_topk;    /* topk_logits.ptx */
    CUmodule   mod_attn;    /* attention.ptx */

    /* Kernel function handles (resolved from this context's modules) */
    CUfunction kern_add;
    CUfunction kern_copy_2d;   /* optional: absent from older PTX headers */
    CUfunction kern_xchg_2d;   /* optional, as kern_copy_2d */
    CUfunction kern_mul;
    CUfunction kern_silu;
    CUfunction kern_rms_norm;
    CUfunction kern_softmax;
    CUfunction kern_qk_rms_norm;     /* from mod_norm (normalization.ptx) */
    CUfunction kern_matvec_q8;
    CUfunction kern_matvec_q3k;
    CUfunction kern_matvec_q3k_nx;   /* batched N-column variant */
    CUfunction kern_matvec_q3k_nx2;  /* 2-wide tile (L10; n == 2) */
    CUfunction kern_matvec_q3k_nx8;  /* 8-wide tile (L6; n >= 5) */
    /* peer-stream variants (9a): bit-exact, more loads in flight. NULL on an
     * older fc_cuda_ptx.h — fc_matvec_q3k_ps_ctx then reports unavailable. */
    CUfunction kern_matvec_q3k_ps[FC_Q3K_PS_COUNT];
    CUfunction kern_matvec_q4k;      /* from mod_kq (matvec_kquants.ptx) */
    CUfunction kern_matvec_q5k;
    CUfunction kern_matvec_q6k;
    CUfunction kern_matvec_q80;
    CUfunction kern_matvec_q2_0;     /* Bonsai native ternary (GGUF type 42) */
    CUfunction kern_matvec_f32;
    CUfunction kern_matvec_q4k_nx;   /* batched N-column variants */
    CUfunction kern_matvec_q5k_nx;
    CUfunction kern_matvec_q6k_nx;
    CUfunction kern_matvec_q80_nx;
    CUfunction kern_matvec_f32_nx;
    CUfunction kern_matvec_q4k_nx2;  /* 2-wide tiles (L10; n == 2) */
    CUfunction kern_matvec_q5k_nx2;
    CUfunction kern_matvec_q6k_nx2;
    CUfunction kern_matvec_q80_nx2;
    CUfunction kern_matvec_f32_nx2;
    CUfunction kern_matvec_q4k_nx8;  /* 8-wide tiles (L6; n >= 5) */
    CUfunction kern_matvec_q5k_nx8;
    CUfunction kern_matvec_q6k_nx8;
    CUfunction kern_matvec_q80_nx8;
    CUfunction kern_matvec_f32_nx8;
    CUfunction kern_matvec_q80_r4c8; /* Q8_0 prefill tile: 4 rows x 8 cols a thread */
    CUfunction kern_matvec_q80_r4c3; /* exact-width tiles for 3 / 4 / 5 columns */
    CUfunction kern_matvec_q80_r4c4;
    CUfunction kern_matvec_q80_r4c5;
    CUfunction kern_q80p[FC_Q80P_NKERN]; /* Q8_0 parity (K-split) tiles, fc_q80p_tiles order */
    /* F16 dense rows (gguf type 1) — attn q/k/v/o, shared expert, router,
     * LM head of an F16 model. OPTIONAL symbols: an fc_cuda_ptx.h from
     * before F16 leaves them NULL and gpu_matvec keeps its host fallback. */
    CUfunction kern_matvec_f16;
    CUfunction kern_matvec_f16_nx;
    CUfunction kern_matvec_f16_nx2;
    CUfunction kern_matvec_f16_nx8;
    /* BF16 dense rows (gguf type 30) — the same weights as F16 without the
     * re-rounding, since the models ship bf16. OPTIONAL for the same reason. */
    CUfunction kern_matvec_bf16;
    CUfunction kern_matvec_bf16_nx;
    CUfunction kern_matvec_bf16_nx2;
    CUfunction kern_matvec_bf16_nx8;
    CUfunction kern_rope_neox_partial;
    CUfunction kern_moe_router_topk;
    CUfunction kern_moe_weighted_add;
    CUfunction kern_moe_shared_gate;
    CUfunction kern_moe_gateup_q3k;     /* from mod_moegrp (moe_expert_group.ptx) */
    /* Q4_K gate+up (single-token and batched). OPTIONAL, like the compacted
     * dedup pair below: an fc_cuda_ptx.h regenerated before they existed leaves
     * these NULL while the whole Q3_K grouped path still loads, and
     * fc_moe_gateup_quant_available_ctx then reports Q4_K unsupported so the
     * caller declines the grouped path instead of decoding Q4_K bytes as Q3_K. */
    CUfunction kern_moe_gateup_q4k;
    CUfunction kern_moe_gateup_q4k_nx;
    /* Q6_K down (single-token and batched). OPTIONAL for the same reason. */
    CUfunction kern_moe_down_q6k;
    CUfunction kern_moe_down_q6k_nx;
    /* F16 gate+up and down (single-token and batched) — the top of this
     * hardware's precision ladder. OPTIONAL for the same reason. */
    CUfunction kern_moe_gateup_f16;
    CUfunction kern_moe_gateup_f16_nx;
    CUfunction kern_moe_down_f16;
    CUfunction kern_moe_down_f16_nx;
    /* BF16 gate+up and down — the shipped-values control for the F16 pair. */
    CUfunction kern_moe_gateup_bf16;
    CUfunction kern_moe_gateup_bf16_nx;
    CUfunction kern_moe_down_bf16;
    CUfunction kern_moe_down_bf16_nx;
    /* uint2-load variants of both 16-bit formats, selected by
     * FC_MOE_VEC_LOADS (see fc_moe_vec_loads_enabled). Same DRAM bytes, half
     * the LSU transactions, DIFFERENT lane->element map — so they are a
     * separate kernel set and not a drop-in for the scalar ones. */
    CUfunction kern_moe_gateup_f16_vec;
    CUfunction kern_moe_gateup_f16_vec_nx;
    CUfunction kern_moe_down_f16_vec;
    CUfunction kern_moe_down_f16_vec_nx;
    CUfunction kern_moe_gateup_bf16_vec;
    CUfunction kern_moe_gateup_bf16_vec_nx;
    CUfunction kern_moe_down_bf16_vec;
    CUfunction kern_moe_down_bf16_vec_nx;
    /* Q8_0 gate+up and down (single-token and batched), over the 288-B
     * expert super-blocks of fc_moe_repack_weight. OPTIONAL like the rest. */
    CUfunction kern_moe_gateup_q80;
    CUfunction kern_moe_gateup_q80_nx;
    CUfunction kern_moe_down_q80;
    CUfunction kern_moe_down_q80_nx;
    /* plain-load batched Q8_0 forms for ntok > fc_moe_q80_ldg_max_tok() */
    CUfunction kern_moe_gateup_q80_nxp;
    CUfunction kern_moe_down_q80_nxp;
    CUfunction kern_moe_down_q3k;
    CUfunction kern_moe_down_q4k;
    CUfunction kern_moe_silu_mul;
    CUfunction kern_moe_shexp_dot_gate;
    CUfunction kern_ssm_recurrence;     /* from mod_ssm (ssm_gated_delta.ptx) */
    CUfunction kern_ssm_conv1d_silu;
    CUfunction kern_ssm_gated_norm;
    CUfunction kern_sigmoid;            /* SSM glue: per-element sigmoid */
    CUfunction kern_ssm_decay_gate;     /* SSM glue: softplus decay gate */
    CUfunction kern_ssm_qknorm_gqa;     /* SSM glue: L2-norm + GQA repeat */
    CUfunction kern_fused_multi;        /* from mod_fused (fused_layer.ptx) */
    CUfunction kern_attn_qknorm_rope;
    CUfunction kern_topk_blockmax;      /* from mod_topk (topk_logits.ptx) */
    CUfunction kern_topk_filter;
    CUfunction kern_topk_rank_select;
    CUfunction kern_topk_select;

    /* Batched (prefill) kernels — one launch covers n positions/tokens. */
    CUfunction kern_rms_norm_nx;        /* normalization.ptx */
    CUfunction kern_rope_nx;            /* rope.ptx */
    CUfunction kern_moe_router_topk_nx; /* moe_routing.ptx */
    CUfunction kern_moe_router_topk_rank; /* moe_routing.ptx */
    CUfunction kern_moe_gateup_q3k_nx;  /* moe_expert_group.ptx */
    CUfunction kern_moe_down_q3k_nx;
    CUfunction kern_moe_down_q4k_nx;
    CUfunction kern_moe_shexp_dot_gate_nx;
    CUfunction kern_moe_dedup_build;    /* moe_expert_group.ptx (dedup) */
    CUfunction kern_moe_gateup_q3k_dedup;
    CUfunction kern_moe_gateup_q3k_dedup_shx;
    /* Compacted-grid dedup pair (FC_MOE_DEDUP_COMPACT, default off). OPTIONAL:
     * present only in a -DFC_MOE_DEDUP_COMPACT=1 PTX build, so with the shipped
     * fc_cuda_ptx.h both stay NULL and the env flag silently does nothing. */
    CUfunction kern_moe_dedup_build_wl;
    CUfunction kern_moe_gateup_q3k_dedup_shxc;
    /* Q8_0 expert-grouped gate+up (prefill). OPTIONAL: a header predating it
     * leaves it NULL and callers keep fc_moe_gateup_nx_ctx. */
    CUfunction kern_moe_gateup_q80_dedup;
    CUfunction kern_ssm_conv1d_silu_seq;/* ssm_gated_delta.ptx */
    CUfunction kern_ssm_recurrence_seq;
    CUfunction kern_ssm_recurrence_seq_warp; /* warp-shuffle variant (bit-exact) */
    CUfunction kern_ssm_recurrence_seq_warp8; /* 8-lane layout of it (bit-exact) */
    CUfunction kern_ssm_recurrence_seq_unroll128; /* S==128 unrolled-tree variant */
    CUfunction kern_ssm_decay_gate_nx;
    CUfunction kern_ssm_qknorm_gqa_nx;

    /* Multi-user decode (slot) kernels — K users, one token each. */
    CUfunction kern_ssm_conv1d_silu_slots;  /* ssm_gated_delta.ptx */
    CUfunction kern_ssm_recurrence_slots;
    CUfunction kern_ssm_recurrence_slots_2pass; /* T4 fused-reduction variant */
    CUfunction kern_ssm_gdn_step_slots_warp;    /* fused sigmoid+gate+recur */
    CUfunction kern_ssm_gdn_step_slots_warp8;   /* 8-lane layout of it */
    CUfunction kern_moe_shexp_gate_combine_nx;  /* dot gate + 2 adds, fused */
    CUfunction kern_rope_pos;               /* rope.ptx */

    /* On-die causal GQA flash attention (attention.ptx). */
    CUfunction kern_attn_qz_split;
    CUfunction kern_attn_kv_append;
    CUfunction kern_attn_flash_prefill;
    CUfunction kern_attn_kv_append_q4;      /* Q4 KV cache variants */
    CUfunction kern_attn_flash_prefill_q4;
    /* Narrow-staging prefill variants (FC_ATTN_KTILE_FP16, LOSSY, default
     * off). OPTIONAL: absent from a pre-#25 fc_cuda_ptx.h, in which case the
     * flag silently does nothing. */
    CUfunction kern_attn_flash_prefill_q4_kh;
    CUfunction kern_attn_flash_prefill_q4_kqh;
    /* Register-K prefill variant (FC_ATTN_KTILE_REG, BIT-EXACT, default off).
     * OPTIONAL, same tolerance as the narrow variants above. */
    CUfunction kern_attn_flash_prefill_q4_kr;
    /* Shared-V prefill variant (FC_ATTN_VSTAGE, BIT-EXACT, default on).
     * OPTIONAL, same tolerance as the variants above. */
    CUfunction kern_attn_flash_prefill_q4_kvs;
    /* kvs specialised for kv_groups 8 / head_dim 256 (FC_ATTN_G8, BIT-EXACT,
     * default on). OPTIONAL, same tolerance as the variants above. */
    CUfunction kern_attn_flash_prefill_q4_g8;
    CUfunction kern_attn_flash_decode;      /* decode split-attention */
    CUfunction kern_attn_flash_decode_q4;
    CUfunction kern_attn_flash_decode_combine;

    /* Device-position decode variants (CUDA-graph capture). OPTIONAL: looked
     * up tolerantly after the attention module loads, so a stale (pre-C1)
     * fc_cuda_ptx.h leaves these NULL and only disables graph capture instead
     * of failing the whole attention module. */
    CUfunction kern_attn_kv_append_dev;
    CUfunction kern_attn_kv_append_q4_dev;
    CUfunction kern_attn_flash_decode_dev;
    CUfunction kern_attn_flash_decode_q4_dev;
    CUfunction kern_attn_flash_decode_combine_dev;

    /* Multi-user decode slot variants (L7). OPTIONAL like the _dev kernels:
     * a stale fc_cuda_ptx.h leaves these NULL and the slots path falls back
     * to host attention via fc_attn_flash_decode_slots_available_ctx. */
    CUfunction kern_attn_kv_append_slots;
    CUfunction kern_attn_kv_append_slots_q4;
    CUfunction kern_attn_flash_decode_slots;
    CUfunction kern_attn_flash_decode_slots_q4;
    CUfunction kern_attn_flash_decode_slots_q4_kvs; /* register-K + shared-V */
    CUfunction kern_attn_flash_decode_slots_q4_kvf; /* kvs, fewer instructions */
    CUfunction kern_attn_flash_decode_combine_gated; /* combine + out gate */
    /* R-row spec verify, one instance per rows-per-block width (1..5). */
    CUfunction kern_attn_flash_decode_rows_q4_kvs[FC_ATTN_ROWS_MAX];
    CUfunction kern_attn_flash_decode_rows_q4_dev;
    CUfunction kern_attn_flash_decode_rows_q4_dev_ps;

    /* Lazily-allocated device scratch for fc_topk_logits_slots_ctx: the
     * fast path's chunk maxima and candidates, the fallback tree's two
     * ping-pong levels, and the [nslots][k] winners + status. Grown to the
     * largest (nslots, n, k) seen (topk_scratch_bytes); freed with ctx. */
    CUdeviceptr topk_scratch;
    size_t topk_scratch_bytes;

    /* MemoryAware per-die VRAM tally: running sum of
     * bytes this context has cuMemAlloc'd and not freed. Maintained at the two
     * buffer choke points; the number admission schedules against. */
    size_t     alloc_bytes;
};

/* Per-device allocation: tagged with its owning context so frees and
 * copies always run with the right context current. */
struct fc_buffer {
    CUdeviceptr  dptr;   /* GPU device pointer */
    size_t       size;   /* Size in bytes */
    fc_cuda_ctx *owner;  /* Context that allocated this buffer */
};

/* ================================================================
 * Process-default context (back-compat single-die path)
 * ================================================================ */

static fc_cuda_ctx *_fc_default_ctx = NULL;

/* ================================================================
 * Live-context registry (MemoryAware §8) — a flat table of every ready
 * context, so a signal handler can drain + destroy them all and let the
 * process exit with the /dev/nvidiaN fd closed the proven-clean way. Fixed
 * size (16 K80 dies); registration is single-threaded (contexts are created
 * at load, before the accept loop), so no lock is needed on the common path.
 * ================================================================ */

#define FC_MAX_LIVE_CTX 16
static fc_cuda_ctx *_fc_live_ctx[FC_MAX_LIVE_CTX];
static int          _fc_live_ctx_n = 0;

static void fc_registry_add(fc_cuda_ctx *ctx)
{
    if (!ctx)
        return;
    for (int i = 0; i < _fc_live_ctx_n; i++)
        if (_fc_live_ctx[i] == ctx)
            return;                 /* already registered */
    if (_fc_live_ctx_n < FC_MAX_LIVE_CTX)
        _fc_live_ctx[_fc_live_ctx_n++] = ctx;
}

static void fc_registry_remove(fc_cuda_ctx *ctx)
{
    for (int i = 0; i < _fc_live_ctx_n; i++) {
        if (_fc_live_ctx[i] == ctx) {
            _fc_live_ctx[i] = _fc_live_ctx[--_fc_live_ctx_n];
            _fc_live_ctx[_fc_live_ctx_n] = NULL;
            return;
        }
    }
}

/* ================================================================
 * Internal helpers — symbol loading
 * ================================================================ */

/* Load a symbol from a dlopen'd library. Returns NULL on failure. */
static void *fc_dlsym(void *lib, const char *name)
{
    void *sym = dlsym(lib, name);
    if (!sym)
        fprintf(stderr, "[fc_cuda] dlsym failed for %s: %s\n", name, dlerror());
    return sym;
}

/* Load all CUDA driver API symbols. Returns 0 on success. */
static int fc_load_cuda_symbols(void)
{
    #define LOAD(fn) do { \
        p_##fn = (pfn_##fn)fc_dlsym(_fc_libcuda, #fn); \
        if (!p_##fn) return -1; \
    } while(0)

    /* Many CUDA driver functions have _v2 variants (modern API since CUDA 3.2).
     * We MUST use _v2 for context/memory ops — the v1 symbols create legacy
     * contexts that cuBLAS cannot use. */
    #define LOAD_V2(fn) do { \
        p_##fn = (pfn_##fn)dlsym(_fc_libcuda, #fn "_v2"); \
        if (!p_##fn) p_##fn = (pfn_##fn)dlsym(_fc_libcuda, #fn); \
        if (!p_##fn) { \
            fprintf(stderr, "[fc_cuda] dlsym failed for %s\n", #fn); \
            return -1; \
        } \
    } while(0)

    LOAD(cuInit);
    LOAD(cuDeviceGetCount);
    LOAD(cuDeviceGet);
    LOAD(cuDeviceGetName);
    LOAD_V2(cuCtxCreate);
    LOAD_V2(cuCtxDestroy);
    LOAD(cuCtxSynchronize);
    LOAD_V2(cuCtxPushCurrent);
    LOAD_V2(cuCtxPopCurrent);
    LOAD_V2(cuMemAlloc);
    LOAD_V2(cuMemFree);
    LOAD_V2(cuMemGetInfo);
    LOAD_V2(cuMemcpyHtoD);
    LOAD_V2(cuMemcpyDtoH);
    LOAD_V2(cuMemcpyDtoHAsync);
    LOAD_V2(cuMemcpyHtoDAsync);
    LOAD_V2(cuMemcpyDtoDAsync);
    LOAD(cuModuleLoadData);
    /* Optional peer-copy symbols — absent ones just disable fc_buffer_copy_peer
     * (multi-die vocab/tensor split falls back to host staging). */
    p_cuMemcpyPeer = (pfn_cuMemcpyPeer)dlsym(_fc_libcuda, "cuMemcpyPeer");
    p_cuMemcpyPeerAsync =
        (pfn_cuMemcpyPeerAsync)dlsym(_fc_libcuda, "cuMemcpyPeerAsync");
    p_cuDeviceGetAttribute =
        (pfn_cuDeviceGetAttribute)dlsym(_fc_libcuda, "cuDeviceGetAttribute");
    p_cuMemcpy3DPeerAsync =
        (pfn_cuMemcpy3DPeerAsync)dlsym(_fc_libcuda, "cuMemcpy3DPeerAsync");
    p_cuDeviceCanAccessPeer =
        (pfn_cuDeviceCanAccessPeer)dlsym(_fc_libcuda, "cuDeviceCanAccessPeer");
    p_cuCtxEnablePeerAccess =
        (pfn_cuCtxEnablePeerAccess)dlsym(_fc_libcuda, "cuCtxEnablePeerAccess");
    /* Optional pinned host memory — absent symbols make fc_host_alloc return NULL. */
    p_cuMemHostAlloc = (pfn_cuMemHostAlloc)dlsym(_fc_libcuda, "cuMemHostAlloc");
    p_cuMemFreeHost = (pfn_cuMemFreeHost)dlsym(_fc_libcuda, "cuMemFreeHost");
    LOAD(cuModuleUnload);
    LOAD(cuModuleGetFunction);
    LOAD(cuLaunchKernel);
    LOAD(cuStreamCreate);
    LOAD_V2(cuStreamDestroy);
    LOAD(cuStreamSynchronize);

    /* Optional graph symbols — missing ones just disable fc_graph_*. */
    #define LOAD_OPT_V2(fn) do { \
        p_##fn = (pfn_##fn)dlsym(_fc_libcuda, #fn "_v2"); \
        if (!p_##fn) p_##fn = (pfn_##fn)dlsym(_fc_libcuda, #fn); \
    } while(0)

    LOAD_OPT_V2(cuStreamBeginCapture);
    LOAD_OPT_V2(cuStreamEndCapture);
    LOAD_OPT_V2(cuGraphInstantiate);
    LOAD_OPT_V2(cuGraphLaunch);
    LOAD_OPT_V2(cuGraphDestroy);
    LOAD_OPT_V2(cuGraphExecDestroy);

    /* Optional event symbols — missing ones just disable fc_event_*. */
    LOAD_OPT_V2(cuEventCreate);
    LOAD_OPT_V2(cuEventRecord);
    LOAD_OPT_V2(cuEventQuery);
    LOAD_OPT_V2(cuEventSynchronize);
    LOAD_OPT_V2(cuEventDestroy);
    LOAD_OPT_V2(cuStreamWaitEvent);
    LOAD_OPT_V2(cuEventRecordWithFlags);
    LOAD_OPT_V2(cuEventElapsedTime);
    LOAD_OPT_V2(cuStreamIsCapturing);
    LOAD_OPT_V2(cuGraphGetNodes);
    LOAD_OPT_V2(cuGraphNodeGetType);

    #undef LOAD_OPT_V2
    #undef LOAD_V2
    #undef LOAD
    return 0;
}

/* Load all cuBLAS symbols. Returns 0 on success. */
static int fc_load_cublas_symbols(void)
{
    #define LOAD(fn) do { \
        p_##fn = (pfn_##fn)fc_dlsym(_fc_libcublas, #fn "_v2"); \
        if (!p_##fn) { \
            /* Try without _v2 suffix for older cuBLAS */ \
            p_##fn = (pfn_##fn)fc_dlsym(_fc_libcublas, #fn); \
            if (!p_##fn) return -1; \
        } \
    } while(0)

    LOAD(cublasCreate);
    LOAD(cublasDestroy);
    LOAD(cublasSetStream);
    LOAD(cublasSgemm);
    LOAD(cublasSgemv);

    #undef LOAD
    return 0;
}

/* ================================================================
 * Process-wide driver loading
 * ================================================================ */

int fc_driver_load(void)
{
    if (_fc_driver_loaded)
        return 0;

    /* dlopen libcuda.so.1 (fall back to unversioned) */
    _fc_libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!_fc_libcuda)
        _fc_libcuda = dlopen("libcuda.so", RTLD_LAZY);
    if (!_fc_libcuda) {
        fprintf(stderr, "[fc_cuda] Cannot open libcuda.so: %s\n", dlerror());
        return -1;
    }

    if (fc_load_cuda_symbols() != 0) {
        dlclose(_fc_libcuda);
        _fc_libcuda = NULL;
        return -1;
    }

    CUresult err = p_cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuInit failed: %d\n", err);
        dlclose(_fc_libcuda);
        _fc_libcuda = NULL;
        return -1;
    }

    /* cuBLAS is optional — custom kernels still work without it. */
    _fc_libcublas = dlopen("libcublas.so.11", RTLD_LAZY);
    if (!_fc_libcublas)
        _fc_libcublas = dlopen("libcublas.so", RTLD_LAZY);
    if (_fc_libcublas && fc_load_cublas_symbols() == 0) {
        _fc_cublas_loaded = true;
    } else if (_fc_libcublas) {
        fprintf(stderr, "[fc_cuda] cuBLAS symbol loading failed, "
                "SGEMM/SGEMV unavailable\n");
    } else {
        fprintf(stderr, "[fc_cuda] Cannot open libcublas.so: %s\n"
                "  cuBLAS unavailable, custom kernels only\n", dlerror());
    }

    _fc_driver_loaded = true;
    return 0;
}

bool fc_driver_loaded(void)
{
    return _fc_driver_loaded;
}

/* ================================================================
 * Context current push/pop
 *
 * Every device op brackets itself with fc_ctx_enter()/fc_ctx_leave()
 * so the correct context is current for the call and the caller's
 * previous context is restored afterward. This keeps concurrent dies
 * isolated even when ops interleave.
 * ================================================================ */

/* Make ctx's context current. Returns 0 on success. */
static int fc_ctx_enter(const fc_cuda_ctx *ctx)
{
    CUresult err = p_cuCtxPushCurrent(ctx->context);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuCtxPushCurrent failed: %d\n", err);
        return -1;
    }
    return 0;
}

/* Restore the previous context (pop the one fc_ctx_enter pushed). */
static void fc_ctx_leave(void)
{
    CUcontext prev = NULL;
    p_cuCtxPopCurrent(&prev);
}

/* ================================================================
 * PTX module loading (per context)
 * ================================================================ */

/* Load a PTX module into the current context and extract kernels.
 * funcs is an array of pointers to individual CUfunction variables. */
static int fc_load_ptx_module(const char *ptx, CUmodule *mod,
                              const char **names, CUfunction **funcs, int n)
{
    if (!ptx || !ptx[0])
        return -1;  /* Empty PTX string = not yet compiled */

    CUresult err = p_cuModuleLoadData(mod, ptx);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuModuleLoadData failed: %d\n", err);
        return -1;
    }

    for (int i = 0; i < n; i++) {
        err = p_cuModuleGetFunction(funcs[i], *mod, names[i]);
        if (err != CUDA_SUCCESS) {
            fprintf(stderr, "[fc_cuda] cuModuleGetFunction(%s) failed: %d\n",
                    names[i], err);
            return -1;
        }
    }
    return 0;
}

/* Load all three embedded PTX modules into ctx (ctx must be current).
 * Missing/empty modules are non-fatal — the affected kernels stay NULL
 * and their dispatch functions no-op, matching the original behavior. */
static void fc_ctx_load_kernels(fc_cuda_ctx *ctx)
{
    {
        const char *elem_names[] = { "fc_add", "fc_mul", "fc_silu" };
        CUfunction *elem_funcs[] = {
            &ctx->kern_add, &ctx->kern_mul, &ctx->kern_silu
        };
        if (fc_load_ptx_module(FC_PTX_ELEMENTWISE, &ctx->mod_elem,
                               elem_names, elem_funcs, 3) != 0)
            fprintf(stderr, "[fc_cuda] Elementwise PTX load failed "
                    "(kernels unavailable)\n");
        else if (p_cuModuleGetFunction(&ctx->kern_copy_2d, ctx->mod_elem,
                                       "fc_copy_2d") != CUDA_SUCCESS)
            ctx->kern_copy_2d = NULL;   /* fc_buffer_copy_2d_kernel_async refuses */
        if (ctx->mod_elem && p_cuModuleGetFunction(&ctx->kern_xchg_2d, ctx->mod_elem,
                                                   "fc_xchg_2d") != CUDA_SUCCESS)
            ctx->kern_xchg_2d = NULL;   /* fc_xchg_2d_async refuses */
    }
    {
        const char *norm_names[] = { "fc_rms_norm", "fc_softmax",
                                     "fc_qk_rms_norm", "fc_rms_norm_nx" };
        CUfunction *norm_funcs[] = {
            &ctx->kern_rms_norm, &ctx->kern_softmax, &ctx->kern_qk_rms_norm,
            &ctx->kern_rms_norm_nx
        };
        if (fc_load_ptx_module(FC_PTX_NORMALIZATION, &ctx->mod_norm,
                               norm_names, norm_funcs, 4) != 0)
            fprintf(stderr, "[fc_cuda] Normalization PTX load failed "
                    "(kernels unavailable)\n");
    }
    {
        const char *q8_names[] = { "fc_matvec_q8" };
        CUfunction *q8_funcs[] = { &ctx->kern_matvec_q8 };
        if (fc_load_ptx_module(FC_PTX_MATVEC_Q8, &ctx->mod_q8,
                               q8_names, q8_funcs, 1) != 0)
            fprintf(stderr, "[fc_cuda] Matvec Q8 PTX load failed "
                    "(kernel unavailable)\n");
    }
    {
        const char *q3k_names[] = { "fc_matvec_q3k", "fc_matvec_q3k_nx" };
        CUfunction *q3k_funcs[] = { &ctx->kern_matvec_q3k,
                                    &ctx->kern_matvec_q3k_nx };
        if (fc_load_ptx_module(FC_PTX_MATVEC_Q3K, &ctx->mod_q3k,
                               q3k_names, q3k_funcs, 2) != 0) {
            fprintf(stderr, "[fc_cuda] Matvec Q3K PTX load failed "
                    "(kernel unavailable)\n");
        } else {
            /* Optional narrow/wide tiles (L6 _nx8, L10 _nx2): an older
             * fc_cuda_ptx.h just leaves them NULL and the launcher stays on
             * the 4-wide tile. */
            struct { const char *name; CUfunction *fn; } opt[] = {
                { "fc_matvec_q3k_nx2", &ctx->kern_matvec_q3k_nx2 },
                { "fc_matvec_q3k_nx8", &ctx->kern_matvec_q3k_nx8 },
                /* peer-stream variants — order MUST match the FC_Q3K_PS_*
                 * enum in fc_cuda.h */
                { "fc_matvec_q3k_ps_r2p1",
                  &ctx->kern_matvec_q3k_ps[FC_Q3K_PS_R2P1] },
                { "fc_matvec_q3k_ps_r2p2",
                  &ctx->kern_matvec_q3k_ps[FC_Q3K_PS_R2P2] },
                { "fc_matvec_q3k_ps_r2p4",
                  &ctx->kern_matvec_q3k_ps[FC_Q3K_PS_R2P4] },
                { "fc_matvec_q3k_ps_r2p8",
                  &ctx->kern_matvec_q3k_ps[FC_Q3K_PS_R2P8] },
                { "fc_matvec_q3k_ps_r1p1",
                  &ctx->kern_matvec_q3k_ps[FC_Q3K_PS_R1P1] },
                { "fc_matvec_q3k_ps_r1p4",
                  &ctx->kern_matvec_q3k_ps[FC_Q3K_PS_R1P4] },
                { "fc_matvec_q3k_ps_r1p8",
                  &ctx->kern_matvec_q3k_ps[FC_Q3K_PS_R1P8] },
            };
            for (size_t i = 0; i < sizeof(opt) / sizeof(opt[0]); i++) {
                if (p_cuModuleGetFunction(opt[i].fn, ctx->mod_q3k,
                                          opt[i].name) != CUDA_SUCCESS)
                    *opt[i].fn = NULL;
            }
        }
    }
    {
        const char *kq_names[] = { "fc_matvec_q4k", "fc_matvec_q5k",
                                   "fc_matvec_q6k", "fc_matvec_q80",
                                   "fc_matvec_f32",
                                   "fc_matvec_q4k_nx", "fc_matvec_q5k_nx",
                                   "fc_matvec_q6k_nx", "fc_matvec_q80_nx",
                                   "fc_matvec_f32_nx" };
        CUfunction *kq_funcs[] = {
            &ctx->kern_matvec_q4k, &ctx->kern_matvec_q5k,
            &ctx->kern_matvec_q6k, &ctx->kern_matvec_q80,
            &ctx->kern_matvec_f32,
            &ctx->kern_matvec_q4k_nx, &ctx->kern_matvec_q5k_nx,
            &ctx->kern_matvec_q6k_nx, &ctx->kern_matvec_q80_nx,
            &ctx->kern_matvec_f32_nx
        };
        if (fc_load_ptx_module(FC_PTX_MATVEC_KQUANTS, &ctx->mod_kq,
                               kq_names, kq_funcs, 10) != 0) {
            fprintf(stderr, "[fc_cuda] Matvec K-quants PTX load failed "
                    "(kernels unavailable)\n");
        } else {
            /* Optional 8-wide tiles (L6): tolerant lookup, NULL on a
             * pre-L6 fc_cuda_ptx.h (launcher stays on the 4-wide tile). */
            struct { const char *name; CUfunction *fn; } opt[] = {
                { "fc_matvec_q4k_nx2", &ctx->kern_matvec_q4k_nx2 },
                { "fc_matvec_q5k_nx2", &ctx->kern_matvec_q5k_nx2 },
                { "fc_matvec_q6k_nx2", &ctx->kern_matvec_q6k_nx2 },
                { "fc_matvec_q80_nx2", &ctx->kern_matvec_q80_nx2 },
                { "fc_matvec_f32_nx2", &ctx->kern_matvec_f32_nx2 },
                { "fc_matvec_q4k_nx8", &ctx->kern_matvec_q4k_nx8 },
                { "fc_matvec_q5k_nx8", &ctx->kern_matvec_q5k_nx8 },
                { "fc_matvec_q6k_nx8", &ctx->kern_matvec_q6k_nx8 },
                { "fc_matvec_q80_nx8", &ctx->kern_matvec_q80_nx8 },
                { "fc_matvec_f32_nx8", &ctx->kern_matvec_f32_nx8 },
                { "fc_matvec_q80_r4c8", &ctx->kern_matvec_q80_r4c8 },
                { "fc_matvec_q80_r4c3", &ctx->kern_matvec_q80_r4c3 },
                { "fc_matvec_q80_r4c4", &ctx->kern_matvec_q80_r4c4 },
                { "fc_matvec_q80_r4c5", &ctx->kern_matvec_q80_r4c5 },
                { "fc_matvec_f16", &ctx->kern_matvec_f16 },
                { "fc_matvec_f16_nx", &ctx->kern_matvec_f16_nx },
                { "fc_matvec_f16_nx2", &ctx->kern_matvec_f16_nx2 },
                { "fc_matvec_f16_nx8", &ctx->kern_matvec_f16_nx8 },
                { "fc_matvec_bf16", &ctx->kern_matvec_bf16 },
                { "fc_matvec_bf16_nx", &ctx->kern_matvec_bf16_nx },
                { "fc_matvec_bf16_nx2", &ctx->kern_matvec_bf16_nx2 },
                { "fc_matvec_bf16_nx8", &ctx->kern_matvec_bf16_nx8 },
                { "fc_matvec_q2_0",    &ctx->kern_matvec_q2_0 },
            };
            for (size_t i = 0; i < sizeof(opt) / sizeof(opt[0]); i++) {
                if (p_cuModuleGetFunction(opt[i].fn, ctx->mod_kq,
                                          opt[i].name) != CUDA_SUCCESS)
                    *opt[i].fn = NULL;
            }
            for (int i = 0; i < FC_Q80P_NKERN; i++) {
                char name[40];
                snprintf(name, sizeof(name), "fc_matvec_q80p_r%dc%d",
                         fc_q80p_tiles[i].r, fc_q80p_tiles[i].nc);
                if (p_cuModuleGetFunction(&ctx->kern_q80p[i], ctx->mod_kq,
                                          name) != CUDA_SUCCESS)
                    ctx->kern_q80p[i] = NULL;
            }
        }
    }
    {
        const char *rope_names[] = { "fc_rope_neox_partial",
                                     "fc_rope_neox_partial_nx",
                                     "fc_rope_neox_partial_pos" };
        CUfunction *rope_funcs[] = { &ctx->kern_rope_neox_partial,
                                     &ctx->kern_rope_nx,
                                     &ctx->kern_rope_pos };
        if (fc_load_ptx_module(FC_PTX_ROPE, &ctx->mod_rope,
                               rope_names, rope_funcs, 3) != 0)
            fprintf(stderr, "[fc_cuda] RoPE PTX load failed "
                    "(kernel unavailable)\n");
    }
    {
        const char *moe_names[] = { "fc_moe_router_topk",
                                    "fc_moe_weighted_add",
                                    "fc_moe_shared_gate",
                                    "fc_moe_router_topk_nx",
                                    "fc_moe_router_topk_rank" };
        CUfunction *moe_funcs[] = {
            &ctx->kern_moe_router_topk, &ctx->kern_moe_weighted_add,
            &ctx->kern_moe_shared_gate, &ctx->kern_moe_router_topk_nx,
            &ctx->kern_moe_router_topk_rank
        };
        if (fc_load_ptx_module(FC_PTX_MOE_ROUTING, &ctx->mod_moe,
                               moe_names, moe_funcs, 5) != 0)
            fprintf(stderr, "[fc_cuda] MoE routing PTX load failed "
                    "(kernels unavailable)\n");
    }
    {
        const char *grp_names[] = { "fc_moe_gateup_q3k",
                                    "fc_moe_down_q3k",
                                    "fc_moe_down_q4k",
                                    "fc_moe_silu_mul",
                                    "fc_moe_shexp_dot_gate",
                                    "fc_moe_gateup_q3k_nx",
                                    "fc_moe_down_q3k_nx",
                                    "fc_moe_down_q4k_nx",
                                    "fc_moe_shexp_dot_gate_nx",
                                    "fc_moe_dedup_build",
                                    "fc_moe_gateup_q3k_dedup",
                                    "fc_moe_gateup_q3k_dedup_shx" };
        CUfunction *grp_funcs[] = {
            &ctx->kern_moe_gateup_q3k, &ctx->kern_moe_down_q3k,
            &ctx->kern_moe_down_q4k, &ctx->kern_moe_silu_mul,
            &ctx->kern_moe_shexp_dot_gate,
            &ctx->kern_moe_gateup_q3k_nx, &ctx->kern_moe_down_q3k_nx,
            &ctx->kern_moe_down_q4k_nx, &ctx->kern_moe_shexp_dot_gate_nx,
            &ctx->kern_moe_dedup_build, &ctx->kern_moe_gateup_q3k_dedup,
            &ctx->kern_moe_gateup_q3k_dedup_shx
        };
        if (fc_load_ptx_module(FC_PTX_MOE_EXPERT_GROUP, &ctx->mod_moegrp,
                               grp_names, grp_funcs, 12) != 0) {
            fprintf(stderr, "[fc_cuda] MoE expert-group PTX load failed "
                    "(kernels unavailable)\n");
        } else {
            /* Optional names, looked up tolerantly — a miss leaves the
             * function NULL and the corresponding *_available_ctx probe
             * reports 0, rather than failing the whole module (and with it the
             * shipping Q3_K path) the way a required name would.
             *   - the compacted-grid dedup pair: only a
             *     -DFC_MOE_DEDUP_COMPACT=1 PTX build has them.
             *   - the Q4_K gate+up pair: only a PTX regenerated since Q4_K_M
             *     MoE support has them.
             *   - the F16 gate+up / down pairs: only a PTX regenerated since
             *     F16 MoE support has them. */
            struct { const char *name; CUfunction *fn; } opt[] = {
                { "fc_moe_dedup_build_wl", &ctx->kern_moe_dedup_build_wl },
                { "fc_moe_gateup_q3k_dedup_shxc",
                  &ctx->kern_moe_gateup_q3k_dedup_shxc },
                { "fc_moe_gateup_q80_dedup",
                  &ctx->kern_moe_gateup_q80_dedup },
                { "fc_moe_gateup_q4k", &ctx->kern_moe_gateup_q4k },
                { "fc_moe_gateup_q4k_nx", &ctx->kern_moe_gateup_q4k_nx },
                { "fc_moe_down_q6k", &ctx->kern_moe_down_q6k },
                { "fc_moe_down_q6k_nx", &ctx->kern_moe_down_q6k_nx },
                { "fc_moe_gateup_f16", &ctx->kern_moe_gateup_f16 },
                { "fc_moe_gateup_f16_nx", &ctx->kern_moe_gateup_f16_nx },
                { "fc_moe_down_f16", &ctx->kern_moe_down_f16 },
                { "fc_moe_down_f16_nx", &ctx->kern_moe_down_f16_nx },
                { "fc_moe_gateup_bf16", &ctx->kern_moe_gateup_bf16 },
                { "fc_moe_gateup_bf16_nx", &ctx->kern_moe_gateup_bf16_nx },
                { "fc_moe_down_bf16", &ctx->kern_moe_down_bf16 },
                { "fc_moe_down_bf16_nx", &ctx->kern_moe_down_bf16_nx },
                { "fc_moe_gateup_f16_vec", &ctx->kern_moe_gateup_f16_vec },
                { "fc_moe_gateup_f16_vec_nx",
                  &ctx->kern_moe_gateup_f16_vec_nx },
                { "fc_moe_down_f16_vec", &ctx->kern_moe_down_f16_vec },
                { "fc_moe_down_f16_vec_nx", &ctx->kern_moe_down_f16_vec_nx },
                { "fc_moe_gateup_bf16_vec", &ctx->kern_moe_gateup_bf16_vec },
                { "fc_moe_gateup_bf16_vec_nx",
                  &ctx->kern_moe_gateup_bf16_vec_nx },
                { "fc_moe_down_bf16_vec", &ctx->kern_moe_down_bf16_vec },
                { "fc_moe_down_bf16_vec_nx", &ctx->kern_moe_down_bf16_vec_nx },
                { "fc_moe_gateup_q80", &ctx->kern_moe_gateup_q80 },
                { "fc_moe_gateup_q80_nx", &ctx->kern_moe_gateup_q80_nx },
                { "fc_moe_down_q80", &ctx->kern_moe_down_q80 },
                { "fc_moe_down_q80_nx", &ctx->kern_moe_down_q80_nx },
                { "fc_moe_gateup_q80_nxp", &ctx->kern_moe_gateup_q80_nxp },
                { "fc_moe_down_q80_nxp", &ctx->kern_moe_down_q80_nxp },
                { "fc_moe_shexp_gate_combine_nx",
                  &ctx->kern_moe_shexp_gate_combine_nx },
            };
            for (size_t i = 0; i < sizeof(opt) / sizeof(opt[0]); i++) {
                if (p_cuModuleGetFunction(opt[i].fn, ctx->mod_moegrp,
                                          opt[i].name) != CUDA_SUCCESS)
                    *opt[i].fn = NULL;
            }
        }
    }
    {
        const char *ssm_names[] = { "fc_ssm_recurrence",
                                    "fc_ssm_conv1d_silu",
                                    "fc_ssm_gated_norm",
                                    "fc_sigmoid",
                                    "fc_ssm_decay_gate",
                                    "fc_ssm_qknorm_gqa",
                                    "fc_ssm_conv1d_silu_seq",
                                    "fc_ssm_recurrence_seq",
                                    "fc_ssm_decay_gate_nx",
                                    "fc_ssm_qknorm_gqa_nx",
                                    "fc_ssm_conv1d_silu_slots",
                                    "fc_ssm_recurrence_slots",
                                    "fc_ssm_recurrence_slots_2pass",
                                    "fc_ssm_recurrence_seq_warp",
                                    "fc_ssm_recurrence_seq_unroll128",
                                    "fc_ssm_gdn_step_slots_warp",
                                    "fc_ssm_recurrence_seq_warp8",
                                    "fc_ssm_gdn_step_slots_warp8" };
        CUfunction *ssm_funcs[] = {
            &ctx->kern_ssm_recurrence, &ctx->kern_ssm_conv1d_silu,
            &ctx->kern_ssm_gated_norm, &ctx->kern_sigmoid,
            &ctx->kern_ssm_decay_gate, &ctx->kern_ssm_qknorm_gqa,
            &ctx->kern_ssm_conv1d_silu_seq, &ctx->kern_ssm_recurrence_seq,
            &ctx->kern_ssm_decay_gate_nx, &ctx->kern_ssm_qknorm_gqa_nx,
            &ctx->kern_ssm_conv1d_silu_slots, &ctx->kern_ssm_recurrence_slots,
            &ctx->kern_ssm_recurrence_slots_2pass,
            &ctx->kern_ssm_recurrence_seq_warp,
            &ctx->kern_ssm_recurrence_seq_unroll128,
            &ctx->kern_ssm_gdn_step_slots_warp,
            &ctx->kern_ssm_recurrence_seq_warp8,
            &ctx->kern_ssm_gdn_step_slots_warp8
        };
        if (fc_load_ptx_module(FC_PTX_SSM_GATED_DELTA, &ctx->mod_ssm,
                               ssm_names, ssm_funcs, 18) != 0)
            fprintf(stderr, "[fc_cuda] SSM gated-delta PTX load failed "
                    "(kernels unavailable)\n");
    }
    {
        const char *fu_names[] = { "fc_fused_multi", "fc_attn_qknorm_rope" };
        CUfunction *fu_funcs[] = {
            &ctx->kern_fused_multi, &ctx->kern_attn_qknorm_rope
        };
        if (fc_load_ptx_module(FC_PTX_FUSED_LAYER, &ctx->mod_fused,
                               fu_names, fu_funcs, 2) != 0)
            fprintf(stderr, "[fc_cuda] Fused-layer PTX load failed "
                    "(kernels unavailable)\n");
    }
    {
        const char *tk_names[] = { "fc_topk_blockmax", "fc_topk_filter",
                                   "fc_topk_rank_select", "fc_topk_select" };
        CUfunction *tk_funcs[] = {
            &ctx->kern_topk_blockmax, &ctx->kern_topk_filter,
            &ctx->kern_topk_rank_select, &ctx->kern_topk_select
        };
        if (fc_load_ptx_module(FC_PTX_TOPK_LOGITS, &ctx->mod_topk,
                               tk_names, tk_funcs, 4) != 0)
            fprintf(stderr, "[fc_cuda] Top-k logits PTX load failed "
                    "(kernels unavailable)\n");
    }
    {
        const char *at_names[] = { "fc_attn_qz_split", "fc_attn_kv_append",
                                   "fc_attn_flash_prefill",
                                   "fc_attn_kv_append_q4",
                                   "fc_attn_flash_prefill_q4",
                                   "fc_attn_flash_decode",
                                   "fc_attn_flash_decode_q4",
                                   "fc_attn_flash_decode_combine" };
        CUfunction *at_funcs[] = {
            &ctx->kern_attn_qz_split, &ctx->kern_attn_kv_append,
            &ctx->kern_attn_flash_prefill,
            &ctx->kern_attn_kv_append_q4, &ctx->kern_attn_flash_prefill_q4,
            &ctx->kern_attn_flash_decode, &ctx->kern_attn_flash_decode_q4,
            &ctx->kern_attn_flash_decode_combine
        };
        if (fc_load_ptx_module(FC_PTX_ATTENTION, &ctx->mod_attn,
                               at_names, at_funcs, 8) != 0) {
            fprintf(stderr, "[fc_cuda] Attention PTX load failed "
                    "(kernels unavailable)\n");
        } else {
            /* Optional device-position (graph-capture) kernels: absent from
             * a pre-C1 fc_cuda_ptx.h. Look them up tolerantly — a miss just
             * leaves the function NULL (graph capture self-disables via
             * fc_attn_decode_graph_available_ctx) rather than failing the
             * whole attention module. */
            struct { const char *name; CUfunction *fn; } opt[] = {
                { "fc_attn_flash_prefill_q4_kh",
                  &ctx->kern_attn_flash_prefill_q4_kh },
                { "fc_attn_flash_prefill_q4_kqh",
                  &ctx->kern_attn_flash_prefill_q4_kqh },
                { "fc_attn_flash_prefill_q4_kr",
                  &ctx->kern_attn_flash_prefill_q4_kr },
                { "fc_attn_flash_prefill_q4_kvs",
                  &ctx->kern_attn_flash_prefill_q4_kvs },
                { "fc_attn_flash_prefill_q4_g8",
                  &ctx->kern_attn_flash_prefill_q4_g8 },
                { "fc_attn_kv_append_dev",
                  &ctx->kern_attn_kv_append_dev },
                { "fc_attn_kv_append_q4_dev",
                  &ctx->kern_attn_kv_append_q4_dev },
                { "fc_attn_flash_decode_dev",
                  &ctx->kern_attn_flash_decode_dev },
                { "fc_attn_flash_decode_q4_dev",
                  &ctx->kern_attn_flash_decode_q4_dev },
                { "fc_attn_flash_decode_combine_dev",
                  &ctx->kern_attn_flash_decode_combine_dev },
                { "fc_attn_kv_append_slots",
                  &ctx->kern_attn_kv_append_slots },
                { "fc_attn_kv_append_slots_q4",
                  &ctx->kern_attn_kv_append_slots_q4 },
                { "fc_attn_flash_decode_slots",
                  &ctx->kern_attn_flash_decode_slots },
                { "fc_attn_flash_decode_slots_q4",
                  &ctx->kern_attn_flash_decode_slots_q4 },
                { "fc_attn_flash_decode_slots_q4_kvs",
                  &ctx->kern_attn_flash_decode_slots_q4_kvs },
                { "fc_attn_flash_decode_slots_q4_kvf",
                  &ctx->kern_attn_flash_decode_slots_q4_kvf },
                { "fc_attn_flash_decode_combine_gated",
                  &ctx->kern_attn_flash_decode_combine_gated },
                { "fc_attn_flash_decode_rows_q4_kvs_r1",
                  &ctx->kern_attn_flash_decode_rows_q4_kvs[0] },
                { "fc_attn_flash_decode_rows_q4_kvs_r2",
                  &ctx->kern_attn_flash_decode_rows_q4_kvs[1] },
                { "fc_attn_flash_decode_rows_q4_kvs_r3",
                  &ctx->kern_attn_flash_decode_rows_q4_kvs[2] },
                { "fc_attn_flash_decode_rows_q4_kvs_r4",
                  &ctx->kern_attn_flash_decode_rows_q4_kvs[3] },
                { "fc_attn_flash_decode_rows_q4_kvs_r5",
                  &ctx->kern_attn_flash_decode_rows_q4_kvs[4] },
                { "fc_attn_flash_decode_rows_q4_dev",
                  &ctx->kern_attn_flash_decode_rows_q4_dev },
                { "fc_attn_flash_decode_rows_q4_dev_ps",
                  &ctx->kern_attn_flash_decode_rows_q4_dev_ps },
            };
            for (size_t i = 0; i < sizeof(opt) / sizeof(opt[0]); i++) {
                if (p_cuModuleGetFunction(opt[i].fn, ctx->mod_attn,
                                          opt[i].name) != CUDA_SUCCESS)
                    *opt[i].fn = NULL;
            }
        }
    }
}

/* Bring up ctx's cuBLAS handle on its stream. Non-fatal on failure. */
static void fc_ctx_init_cublas(fc_cuda_ctx *ctx)
{
    if (!_fc_cublas_loaded)
        return;

    cublasStatus_t cberr = p_cublasCreate(&ctx->cublas);
    if (cberr == CUBLAS_STATUS_SUCCESS) {
        p_cublasSetStream(ctx->cublas, ctx->stream);
    } else {
        fprintf(stderr, "[fc_cuda] cublasCreate failed: %d\n", cberr);
        ctx->cublas = NULL;
    }
}

/* ================================================================
 * Per-device context lifecycle
 * ================================================================ */

fc_cuda_ctx *fc_cuda_init_device(int device_id)
{
    fc_cuda_ctx *ctx = NULL;
    bool pushed = false;
    CUresult err;

    if (fc_driver_load() != 0)
        return NULL;

    ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;
    ctx->device_id = device_id;

    err = p_cuDeviceGet(&ctx->device, device_id);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuDeviceGet(%d) failed: %d\n", device_id, err);
        goto cleanup;
    }

    char name[256] = {0};
    p_cuDeviceGetName(name, sizeof(name), ctx->device);
    printf("[fc_cuda] Device %d: %s\n", device_id, name);

    err = p_cuCtxCreate(&ctx->context, fc_ctx_create_flags(), ctx->device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuCtxCreate(dev %d) failed: %d\n",
                device_id, err);
        goto cleanup;
    }

    /* cuCtxCreate leaves the new context current; track that we must pop. */
    pushed = true;

    err = p_cuStreamCreate(&ctx->stream, 0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuStreamCreate(dev %d) failed: %d\n",
                device_id, err);
        goto cleanup;
    }

    if (p_cuEventCreate && p_cuEventRecord && p_cuEventQuery &&
        p_cuEventCreate(&ctx->sync_ev, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS)
        ctx->sync_ev = NULL;

    fc_ctx_init_cublas(ctx);
    fc_ctx_load_kernels(ctx);

    /* Restore the caller's previous context — cuCtxCreate made ours current. */
    fc_ctx_leave();
    pushed = false;

    ctx->ready = true;
    fc_registry_add(ctx);   /* MemoryAware §8: track for crash-time teardown */
    printf("[fc_cuda] Device %d initialized\n", device_id);
    return ctx;

cleanup:
    if (pushed)
        fc_ctx_leave();
    fc_cuda_ctx_free(ctx);
    return NULL;
}

void fc_cuda_ctx_free(fc_cuda_ctx *ctx)
{
    if (!ctx)
        return;

    fc_registry_remove(ctx);   /* MemoryAware §8: no longer live */

    /* Make the context current so module/stream/cuBLAS teardown targets it. */
    bool entered = (ctx->context && p_cuCtxPushCurrent &&
                    fc_ctx_enter(ctx) == 0);

    if (ctx->stream && p_cuStreamSynchronize)
        fcw_cuStreamSynchronize(ctx->stream);

    if (ctx->mod_elem && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_elem);
    if (ctx->mod_norm && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_norm);
    if (ctx->mod_q8   && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_q8);
    if (ctx->mod_q3k  && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_q3k);
    if (ctx->mod_kq   && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_kq);
    if (ctx->mod_rope && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_rope);
    if (ctx->mod_moe  && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_moe);
    if (ctx->mod_moegrp && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_moegrp);
    if (ctx->mod_ssm  && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_ssm);
    if (ctx->mod_fused && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_fused);
    if (ctx->mod_topk && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_topk);
    if (ctx->mod_attn && p_cuModuleUnload) p_cuModuleUnload(ctx->mod_attn);

    if (ctx->topk_scratch && p_cuMemFree)
        p_cuMemFree(ctx->topk_scratch);

    if (ctx->cublas && p_cublasDestroy)
        p_cublasDestroy(ctx->cublas);

    if (ctx->sync_ev && p_cuEventDestroy)
        p_cuEventDestroy(ctx->sync_ev);

    if (ctx->stream && p_cuStreamDestroy)
        p_cuStreamDestroy(ctx->stream);

    if (entered)
        fc_ctx_leave();

    /* cuCtxDestroy removes the context regardless of current state. */
    if (ctx->context && p_cuCtxDestroy)
        p_cuCtxDestroy(ctx->context);

    free(ctx);
}

bool fc_cuda_ctx_available(const fc_cuda_ctx *ctx)
{
    return ctx && ctx->ready;
}

int fc_cuda_ctx_device_id(const fc_cuda_ctx *ctx)
{
    return ctx ? ctx->device_id : -1;
}

/* ================================================================
 * MemoryAware VRAM signals
 * ================================================================ */

int fc_mem_info_ctx(fc_cuda_ctx *ctx, size_t *free_bytes, size_t *total_bytes)
{
    if (!fc_cuda_ctx_available(ctx) || !p_cuMemGetInfo)
        return -1;

    if (fc_ctx_enter(ctx) != 0)
        return -1;

    size_t f = 0, t = 0;
    CUresult err = p_cuMemGetInfo(&f, &t);
    fc_ctx_leave();

    if (err != CUDA_SUCCESS)
        return -1;

    if (free_bytes)  *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return 0;
}

/* Read a die's free/total VRAM without leaving anything behind on it.
 *
 * cuMemGetInfo needs a current context, so one has to be created — but the
 * full fc_cuda_init_device() context is the wrong instrument for a *probe*:
 * it also builds a cuBLAS handle and JIT-compiles the twelve embedded PTX
 * modules, and the caller (pn_dm_init_cuda) then dropped it on the floor.
 * That is what stranded ~124 MiB of context on every die the process merely
 * looked at, including dies owned by another serve. This creates a BARE
 * context, reads the counters, and destroys it — the die is byte-identical
 * afterwards.
 *
 * Returns 0 on success. */
int fc_cuda_probe_device(int device_id, size_t *free_bytes, size_t *total_bytes)
{
    CUdevice dev;
    CUcontext ctx = NULL;
    size_t f = 0, t = 0;
    CUresult err;

    if (fc_driver_load() != 0)
        return -1;
    if (!p_cuDeviceGet || !p_cuCtxCreate || !p_cuCtxDestroy || !p_cuMemGetInfo)
        return -1;

    if (p_cuDeviceGet(&dev, device_id) != CUDA_SUCCESS)
        return -1;

    /* cuCtxCreate pushes the new context and leaves it current — exactly what
     * cuMemGetInfo needs — so no enter/leave bracket is required here. */
    if (p_cuCtxCreate(&ctx, fc_ctx_create_flags(), dev) != CUDA_SUCCESS)
        return -1;

    err = p_cuMemGetInfo(&f, &t);

    /* Destroying the current context pops it, restoring whatever the caller
     * had current before. */
    p_cuCtxDestroy(ctx);

    if (err != CUDA_SUCCESS || t == 0)
        return -1;

    if (free_bytes)  *free_bytes = f;
    if (total_bytes) *total_bytes = t;
    return 0;
}

size_t fc_ctx_alloc_bytes(const fc_cuda_ctx *ctx)
{
    return ctx ? ctx->alloc_bytes : 0;
}

int fc_cuda_live_ctx_count(void)
{
    return _fc_live_ctx_n;
}

void fc_cuda_emergency_teardown(void)
{
    /* MemoryAware §8.2: drain + destroy every live context so the driver frees
     * its device allocations, then the caller exits and libcuda closes the
     * /dev/nvidiaN fd — the only step that actually reclaims Kepler VRAM. This
     * is NOT async-signal-safe (it calls into libcuda), so it is only used from
     * the SIGINT/SIGTERM path (a controlled shutdown), never from SIGSEGV /
     * SIGABRT where we _exit() and rely on kernel fd-close + startup detection.
     *
     * Iterate a snapshot count from the end: cuCtxDestroy does not need the
     * context current, and fc_ctx_enter/synchronize best-effort drains in-flight
     * work first. We deliberately do minimal work (no module unload) — the fd
     * close on exit reclaims everything; this just quiesces the device. */
    for (int i = _fc_live_ctx_n - 1; i >= 0; i--) {
        fc_cuda_ctx *ctx = _fc_live_ctx[i];
        if (!ctx || !ctx->context)
            continue;
        if (p_cuCtxPushCurrent && fc_ctx_enter(ctx) == 0) {
            if (p_cuCtxSynchronize)
                fcw_cuCtxSynchronize();
            fc_ctx_leave();
        }
        if (p_cuCtxDestroy)
            p_cuCtxDestroy(ctx->context);
        ctx->context = NULL;
        ctx->ready = false;
    }
    _fc_live_ctx_n = 0;
}

/* ================================================================
 * Process-default context (back-compat)
 * ================================================================ */

int fc_init_on(int device_id)
{
    if (_fc_default_ctx)
        return 0;

    if (device_id < 0)
        device_id = 0;

    _fc_default_ctx = fc_cuda_init_device(device_id);
    if (!_fc_default_ctx)
        return -1;

    printf("[fc_cuda] Initialized successfully (device %d)\n", device_id);
    return 0;
}

int fc_init(void)
{
    return fc_init_on(0);
}

bool fc_driver_available(void)
{
    return fc_driver_load() == 0;
}

void fc_free(void)
{
    if (!_fc_default_ctx)
        return;

    fc_cuda_ctx_free(_fc_default_ctx);
    _fc_default_ctx = NULL;

    /* The driver libraries are process-global; closing them here keeps
     * fc_free() symmetric with the original behavior (full shutdown). */
    if (_fc_libcublas) { dlclose(_fc_libcublas); _fc_libcublas = NULL; }
    if (_fc_libcuda)   { dlclose(_fc_libcuda);   _fc_libcuda = NULL;   }
    _fc_driver_loaded = false;
    _fc_cublas_loaded = false;

    printf("[fc_cuda] Shutdown complete\n");
}

bool fc_available(void)
{
    return fc_cuda_ctx_available(_fc_default_ctx);
}

fc_cuda_ctx *fc_default_ctx(void)
{
    return _fc_default_ctx;
}

/* ================================================================
 * Buffers (per-context)
 * ================================================================ */

fc_buffer *fc_buffer_from_ptr_ctx(fc_cuda_ctx *ctx, const void *cpu, size_t bytes)
{
    if (!fc_cuda_ctx_available(ctx) || !cpu || bytes == 0)
        return NULL;

    fc_buffer *buf = calloc(1, sizeof(*buf));
    if (!buf)
        return NULL;
    buf->size = bytes;
    buf->owner = ctx;

    if (fc_ctx_enter(ctx) != 0) {
        free(buf);
        return NULL;
    }

    CUresult err = p_cuMemAlloc(&buf->dptr, bytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemAlloc(%zu) failed: %d\n", bytes, err);
        goto fail;
    }

    err = fcw_cuMemcpyHtoD(buf->dptr, cpu, bytes);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemcpyHtoD failed: %d\n", err);
        p_cuMemFree(buf->dptr);
        goto fail;
    }

    ctx->alloc_bytes += bytes;   /* MemoryAware per-die tally */
    fc_ctx_leave();
    return buf;

fail:
    fc_ctx_leave();
    free(buf);
    return NULL;
}

fc_buffer *fc_buffer_alloc_ctx(fc_cuda_ctx *ctx, size_t bytes)
{
    if (!fc_cuda_ctx_available(ctx) || bytes == 0)
        return NULL;

    fc_buffer *buf = calloc(1, sizeof(*buf));
    if (!buf)
        return NULL;
    buf->size = bytes;
    buf->owner = ctx;

    if (fc_ctx_enter(ctx) != 0) {
        free(buf);
        return NULL;
    }

    CUresult err = p_cuMemAlloc(&buf->dptr, bytes);
    if (err == CUDA_SUCCESS)
        ctx->alloc_bytes += bytes;   /* MemoryAware per-die tally */
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemAlloc(%zu) failed: %d\n", bytes, err);
        free(buf);
        return NULL;
    }

    return buf;
}

fc_buffer *fc_buffer_from_ptr(const void *cpu, size_t bytes)
{
    return fc_buffer_from_ptr_ctx(_fc_default_ctx, cpu, bytes);
}

fc_buffer *fc_buffer_alloc(size_t bytes)
{
    return fc_buffer_alloc_ctx(_fc_default_ctx, bytes);
}

void fc_buffer_free(fc_buffer *buf)
{
    if (!buf)
        return;

    fc_cuda_ctx *ctx = buf->owner;
    bool entered = (ctx && ctx->context && p_cuCtxPushCurrent &&
                    fc_ctx_enter(ctx) == 0);

    if (buf->dptr && p_cuMemFree)
        p_cuMemFree(buf->dptr);

    /* MemoryAware per-die tally: mirror the alloc bump. Clamp so a double-free
     * or an untracked buffer can never underflow the running sum. */
    if (ctx) {
        ctx->alloc_bytes = (ctx->alloc_bytes >= buf->size)
                               ? ctx->alloc_bytes - buf->size : 0;
    }

    if (entered)
        fc_ctx_leave();

    free(buf);
}

fc_buffer* fc_buffer_offset_view(fc_buffer *base, size_t byte_off)
{
    if (!base || byte_off > base->size)
        return NULL;
    fc_buffer *v = (fc_buffer *)malloc(sizeof(*v));
    if (!v)
        return NULL;
    v->dptr  = base->dptr + (CUdeviceptr)byte_off;
    v->size  = base->size - byte_off;
    v->owner = base->owner;   /* non-owning: never cuMemFree'd via this view */
    return v;
}

void fc_buffer_free_view(fc_buffer *view)
{
    /* Frees only the wrapper — the aliased device memory belongs to the base
     * buffer. No cuMemFree, no per-die tally adjustment. */
    free(view);
}

void fc_buffer_to_cpu(fc_buffer *buf, void *dst)
{
    if (!buf || !dst || !fc_cuda_ctx_available(buf->owner))
        return;

    if (fc_ctx_enter(buf->owner) != 0)
        return;
    CUresult err = fcw_cuMemcpyDtoH(dst, buf->dptr, buf->size);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuMemcpyDtoH failed: %d\n", err);
}

void fc_buffer_from_cpu(fc_buffer *buf, const void *src)
{
    if (!buf || !src || !fc_cuda_ctx_available(buf->owner))
        return;

    if (fc_ctx_enter(buf->owner) != 0)
        return;
    CUresult err = fcw_cuMemcpyHtoD(buf->dptr, src, buf->size);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuMemcpyHtoD failed: %d\n", err);
}

size_t fc_buffer_size(fc_buffer *buf)
{
    return buf ? buf->size : 0;
}

/* Partial copies — the batch path sizes its persistent buffers for the
 * maximum batch but moves only the live n tokens' bytes per call. */

void fc_buffer_to_cpu_n(fc_buffer *buf, void *dst, size_t bytes)
{
    if (!buf || !dst || bytes == 0 || !fc_cuda_ctx_available(buf->owner))
        return;
    if (bytes > buf->size)
        bytes = buf->size;

    if (fc_ctx_enter(buf->owner) != 0)
        return;
    CUresult err = fcw_cuMemcpyDtoH(dst, buf->dptr, bytes);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuMemcpyDtoH(n) failed: %d\n", err);
}

void fc_buffer_from_cpu_n(fc_buffer *buf, const void *src, size_t bytes)
{
    if (!buf || !src || bytes == 0 || !fc_cuda_ctx_available(buf->owner))
        return;
    if (bytes > buf->size)
        bytes = buf->size;

    if (fc_ctx_enter(buf->owner) != 0)
        return;
    CUresult err = fcw_cuMemcpyHtoD(buf->dptr, src, bytes);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuMemcpyHtoD(n) failed: %d\n", err);
}

/* Async host-to-device copy queued on the buffer's owning ctx stream: stream-ordered after
 * the prior work on that stream, without blocking the host until that work drains (the
 * blocking cuMemcpyHtoD waits for the whole stream). A pageable `src` is staged by the
 * driver before the call returns, so the caller may reuse it at once. Returns 0, or -1
 * (async HtoD unavailable / failed: the caller falls back to fc_buffer_from_cpu_n). */
int fc_buffer_from_cpu_n_async(fc_buffer *buf, const void *src, size_t bytes)
{
    if (!buf || !src || bytes == 0 || !fc_cuda_ctx_available(buf->owner))
        return -1;
    if (!p_cuMemcpyHtoDAsync)
        return -1;
    if (bytes > buf->size)
        bytes = buf->size;

    if (fc_ctx_enter(buf->owner) != 0)
        return -1;
    CUresult err = fcw_cuMemcpyHtoDAsync(buf->dptr, src, bytes, buf->owner->stream);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemcpyHtoDAsync failed: %d\n", err);
        return -1;
    }
    return 0;
}

/* Async device-to-host copy queued on the buffer's owning ctx stream. Does NOT
 * block the host: the copy is stream-ordered after prior kernel work on that
 * stream, and the caller must fc_sync_ctx(owner) (or wait on an event recorded
 * after this) before reading `dst`. `dst` should be page-locked host memory for
 * the copy to be truly asynchronous, but pageable also works (the driver may
 * stage it). Returns 0 on success, -1 if async DtoH is unavailable (caller can
 * fall back to the blocking fc_buffer_to_cpu_n). */
int fc_buffer_to_cpu_n_async(fc_buffer *buf, void *dst, size_t bytes)
{
    if (!buf || !dst || bytes == 0 || !fc_cuda_ctx_available(buf->owner))
        return -1;
    if (!p_cuMemcpyDtoHAsync)
        return -1;
    if (bytes > buf->size)
        bytes = buf->size;

    if (fc_ctx_enter(buf->owner) != 0)
        return -1;
    CUresult err = fcw_cuMemcpyDtoHAsync(dst, buf->dptr, bytes,
                                       buf->owner->stream);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemcpyDtoHAsync failed: %d\n", err);
        return -1;
    }
    return 0;
}

/* Page-locked host memory (cuMemHostAlloc, PORTABLE so every context's DMA can use it).
 * A copy to or from it skips the driver's pageable staging copy. */
void *fc_host_alloc(fc_cuda_ctx *ctx, size_t bytes)
{
    if (!bytes || !p_cuMemHostAlloc || !p_cuMemFreeHost || !fc_cuda_ctx_available(ctx))
        return NULL;
    if (fc_ctx_enter(ctx) != 0)
        return NULL;
    void *p = NULL;
    CUresult err = p_cuMemHostAlloc(&p, bytes, FC_CU_MEMHOSTALLOC_PORTABLE);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemHostAlloc(%zu) failed: %d\n", bytes, err);
        return NULL;
    }
    return p;
}

void fc_host_free(fc_cuda_ctx *ctx, void *p)
{
    if (!p || !p_cuMemFreeHost || !fc_cuda_ctx_available(ctx) || fc_ctx_enter(ctx) != 0)
        return;
    CUresult err = p_cuMemFreeHost(p);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuMemFreeHost failed: %d\n", err);
}

void fc_buffer_to_cpu_at(fc_buffer *buf, void *dst, size_t src_off,
                         size_t bytes)
{
    if (!buf || !dst || bytes == 0 || !fc_cuda_ctx_available(buf->owner))
        return;
    if (src_off + bytes > buf->size)
        return;

    if (fc_ctx_enter(buf->owner) != 0)
        return;
    CUresult err = fcw_cuMemcpyDtoH(dst, buf->dptr + src_off, bytes);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuMemcpyDtoH(at) failed: %d\n", err);
}

void fc_buffer_from_cpu_at(fc_buffer *buf, size_t dst_off, const void *src,
                           size_t bytes)
{
    if (!buf || !src || bytes == 0 || !fc_cuda_ctx_available(buf->owner))
        return;
    if (dst_off + bytes > buf->size)
        return;

    if (fc_ctx_enter(buf->owner) != 0)
        return;
    CUresult err = fcw_cuMemcpyHtoD(buf->dptr + dst_off, src, bytes);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuMemcpyHtoD(at) failed: %d\n", err);
}

/* Async device-to-device range copy on ctx's stream (both buffers must
 * belong to ctx). Used to slice one position out of a batched buffer
 * (e.g. the last position's x_norm for the LM head). */
void fc_buffer_copy_d2d_ctx(fc_cuda_ctx *ctx, fc_buffer *dst, size_t dst_off,
                            fc_buffer *src, size_t src_off, size_t bytes)
{
    if (!fc_cuda_ctx_available(ctx) || !dst || !src || bytes == 0 ||
        !p_cuMemcpyDtoDAsync)
        return;
    if (dst_off + bytes > dst->size || src_off + bytes > src->size)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    CUresult err = fcw_cuMemcpyDtoDAsync(dst->dptr + dst_off,
                                       src->dptr + src_off, bytes,
                                       ctx->stream);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuMemcpyDtoDAsync failed: %d\n", err);
}

/* ================================================================
 * Peer (die-to-die) copies
 * ================================================================ */

int fc_cuda_devices_same_board(int dev_a, int dev_b)
{
    CUdevice a, b;
    int ma = 0, mb = 0, ga = 0, gb = 0;
    if (fc_driver_load() != 0 || !p_cuDeviceGetAttribute || dev_a < 0 || dev_b < 0)
        return -1;
    if (dev_a == dev_b)
        return 1;
    if (p_cuDeviceGet(&a, dev_a) != CUDA_SUCCESS || p_cuDeviceGet(&b, dev_b) != CUDA_SUCCESS)
        return -1;
    if (p_cuDeviceGetAttribute(&ma, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, a) != CUDA_SUCCESS ||
        p_cuDeviceGetAttribute(&mb, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, b) != CUDA_SUCCESS)
        return -1;
    if (!ma || !mb)
        return 0;
    if (p_cuDeviceGetAttribute(&ga, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, a) !=
            CUDA_SUCCESS ||
        p_cuDeviceGetAttribute(&gb, CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID, b) !=
            CUDA_SUCCESS)
        return -1;
    return ga == gb;
}

int fc_cuda_enable_peer(fc_cuda_ctx *a, fc_cuda_ctx *b)
{
    if (!fc_cuda_ctx_available(a) || !fc_cuda_ctx_available(b))
        return -1;
    if (a == b)
        return 0;
    if (!p_cuDeviceCanAccessPeer || !p_cuCtxEnablePeerAccess ||
        !p_cuMemcpyPeer) {
        fprintf(stderr, "[fc_cuda] peer symbols unavailable — no P2P\n");
        return -1;
    }
    int ab = 0, ba = 0;
    if (p_cuDeviceCanAccessPeer(&ab, a->device, b->device) != CUDA_SUCCESS ||
        p_cuDeviceCanAccessPeer(&ba, b->device, a->device) != CUDA_SUCCESS ||
        !ab || !ba) {
        fprintf(stderr, "[fc_cuda] devices %d/%d cannot peer-access "
                "(%d/%d)\n", a->device_id, b->device_id, ab, ba);
        return -1;
    }
    /* Enable a->(peer b): make a current, enable access to b's context.
     * CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED (704) is benign/idempotent. */
    CUresult e1 = CUDA_SUCCESS, e2 = CUDA_SUCCESS;
    if (fc_ctx_enter(a) == 0) {
        e1 = p_cuCtxEnablePeerAccess(b->context, 0);
        fc_ctx_leave();
    }
    if (fc_ctx_enter(b) == 0) {
        e2 = p_cuCtxEnablePeerAccess(a->context, 0);
        fc_ctx_leave();
    }
    if ((e1 != CUDA_SUCCESS && e1 != 704) ||
        (e2 != CUDA_SUCCESS && e2 != 704)) {
        fprintf(stderr, "[fc_cuda] EnablePeerAccess failed (%d/%d)\n",
                (int)e1, (int)e2);
        return -1;
    }
    return 0;
}

int fc_buffer_copy_peer(fc_buffer *dst, fc_buffer *src, size_t bytes)
{
    if (!dst || !src || bytes == 0 || !p_cuMemcpyPeer)
        return -1;
    if (!fc_cuda_ctx_available(dst->owner) ||
        !fc_cuda_ctx_available(src->owner))
        return -1;
    if (bytes > dst->size || bytes > src->size)
        return -1;
    /* cuMemcpyPeer takes both contexts explicitly; it does not need a
     * current context, but we push src's to be safe on all drivers. */
    if (fc_ctx_enter(src->owner) != 0)
        return -1;
    CUresult err = fcw_cuMemcpyPeer(dst->dptr, dst->owner->context,
                                  src->dptr, src->owner->context, bytes);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemcpyPeer failed: %d\n", (int)err);
        return -1;
    }
    return 0;
}

/* Async device-to-device peer copy queued on `stream_ctx`'s stream. Copies
 * `bytes` from src (on src->owner's device) into dst (on dst->owner's device)
 * over the peer link, stream-ordered after prior work on stream_ctx's stream
 * and does NOT block the host. `stream_ctx` must be one of the two owners
 * (normally the DESTINATION owner, so the receiving die's stream both issues
 * the copy and can be event-synced before it reads dst). The caller must
 * fc_sync_ctx(stream_ctx) or wait on an event recorded after this before
 * reading dst. Requires fc_cuda_enable_peer(src->owner, dst->owner) first.
 * Returns 0 on success, -1 if cuMemcpyPeerAsync is unavailable (caller falls
 * back to the blocking fc_buffer_copy_peer). */
int fc_buffer_copy_peer_async(fc_buffer *dst, fc_buffer *src, size_t bytes,
                              fc_cuda_ctx *stream_ctx)
{
    if (!dst || !src || bytes == 0 || !p_cuMemcpyPeerAsync)
        return -1;
    if (!fc_cuda_ctx_available(dst->owner) ||
        !fc_cuda_ctx_available(src->owner) ||
        !fc_cuda_ctx_available(stream_ctx))
        return -1;
    if (bytes > dst->size || bytes > src->size)
        return -1;
    if (fc_ctx_enter(stream_ctx) != 0)
        return -1;
    CUresult err = fcw_cuMemcpyPeerAsync(dst->dptr, dst->owner->context,
                                       src->dptr, src->owner->context, bytes,
                                       stream_ctx->stream);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemcpyPeerAsync failed: %d\n", (int)err);
        return -1;
    }
    return 0;
}

int fc_buffer_copy_2d_async(fc_buffer *dst, size_t dst_off, size_t dst_pitch,
                            fc_buffer *src, size_t src_off, size_t src_pitch,
                            size_t width, size_t height, fc_cuda_ctx *stream_ctx)
{
    if (!dst || !src || width == 0 || height == 0 || !p_cuMemcpy3DPeerAsync)
        return -1;
    if (!fc_cuda_ctx_available(dst->owner) || !fc_cuda_ctx_available(src->owner) ||
        !fc_cuda_ctx_available(stream_ctx))
        return -1;
    if (width > dst_pitch || width > src_pitch ||
        dst_off + (height - 1) * dst_pitch + width > dst->size ||
        src_off + (height - 1) * src_pitch + width > src->size)
        return -1;
    fc_memcpy3d_peer cp;
    memset(&cp, 0, sizeof(cp));
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice = src->dptr + src_off;
    cp.srcContext = src->owner->context;
    cp.srcPitch = src_pitch;
    cp.srcHeight = height;
    cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice = dst->dptr + dst_off;
    cp.dstContext = dst->owner->context;
    cp.dstPitch = dst_pitch;
    cp.dstHeight = height;
    cp.WidthInBytes = width;
    cp.Height = height;
    cp.Depth = 1;
    if (fc_ctx_enter(stream_ctx) != 0)
        return -1;
    CUresult err = p_cuMemcpy3DPeerAsync(&cp, stream_ctx->stream);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuMemcpy3DPeerAsync failed: %d\n", (int)err);
        return -1;
    }
    return 0;
}

int fc_buffer_copy_2d_kernel_async(fc_buffer *dst, size_t dst_off, size_t dst_pitch,
                                   fc_buffer *src, size_t src_off, size_t src_pitch,
                                   size_t width, size_t height, fc_cuda_ctx *stream_ctx)
{
    if (!dst || !src || width == 0 || height == 0 || height > 65535)
        return -1;
    if (!fc_cuda_ctx_available(dst->owner) || !fc_cuda_ctx_available(src->owner) ||
        !fc_cuda_ctx_available(stream_ctx) || !stream_ctx->kern_copy_2d ||
        dst->owner != stream_ctx)
        return -1;
    if (width > dst_pitch || width > src_pitch ||
        dst_off + (height - 1) * dst_pitch + width > dst->size ||
        src_off + (height - 1) * src_pitch + width > src->size)
        return -1;
    if (((dst->dptr + dst_off) | (src->dptr + src_off) | dst_pitch | src_pitch | width) & 15)
        return -1;   /* 16-byte units only */
    CUdeviceptr s = src->dptr + src_off, d = dst->dptr + dst_off;
    int w16 = (int)(width / 16), sp16 = (int)(src_pitch / 16), dp16 = (int)(dst_pitch / 16);
    unsigned int threads = 256, gx = (unsigned int)((w16 + 255) / 256);
    if (gx > 8)
        gx = 8;
    if (fc_ctx_enter(stream_ctx) != 0)
        return -1;
    void *args[] = { &s, &d, &w16, &sp16, &dp16 };
    CUresult err = fcw_cuLaunchKernel(stream_ctx->kern_copy_2d, gx, (unsigned int)height, 1,
                                    threads, 1, 1, 0, stream_ctx->stream, args, NULL);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] copy_2d launch failed: %d\n", (int)err);
        return -1;
    }
    return 0;
}

int fc_xchg_2d_async(fc_buffer *dst, size_t own_off, size_t peer_off, size_t dst_pitch,
                     fc_buffer *own, fc_buffer *peer, size_t src_pitch, size_t width,
                     size_t height, fc_buffer *sync, fc_buffer *peer_sync, int slot,
                     fc_cuda_ctx *stream_ctx)
{
    if (!dst || !own || !peer || !sync || !peer_sync || width == 0 || height == 0 ||
        height > 65535 || slot < 0)
        return -1;
    if (!fc_cuda_ctx_available(stream_ctx) || !stream_ctx->kern_xchg_2d ||
        dst->owner != stream_ctx || own->owner != stream_ctx || sync->owner != stream_ctx ||
        !fc_cuda_ctx_available(peer->owner) || peer_sync->owner != peer->owner)
        return -1;
    const size_t sb = (size_t)(slot + 1) * FC_XCHG_SYNC_BYTES;
    if (sb > sync->size || sb > peer_sync->size || width > dst_pitch || width > src_pitch ||
        own_off + (height - 1) * dst_pitch + width > dst->size ||
        peer_off + (height - 1) * dst_pitch + width > dst->size ||
        (height - 1) * src_pitch + width > own->size ||
        (height - 1) * src_pitch + width > peer->size)
        return -1;
    if ((dst->dptr | own->dptr | peer->dptr | own_off | peer_off | dst_pitch | src_pitch |
         width) & 15)
        return -1;   /* 16-byte units only */
    CUdeviceptr o = own->dptr, p = peer->dptr, d = dst->dptr;
    CUdeviceptr sy = sync->dptr + (size_t)slot * FC_XCHG_SYNC_BYTES;
    CUdeviceptr ps = peer_sync->dptr + (size_t)slot * FC_XCHG_SYNC_BYTES;
    int w16 = (int)(width / 16), sp16 = (int)(src_pitch / 16), dp16 = (int)(dst_pitch / 16);
    int oo16 = (int)(own_off / 16), po16 = (int)(peer_off / 16);
    unsigned int threads = 256, gx = (unsigned int)((w16 + 255) / 256);
    if (gx > 8)
        gx = 8;
    if (fc_ctx_enter(stream_ctx) != 0)
        return -1;
    void *args[] = { &o, &p, &d, &w16, &sp16, &dp16, &oo16, &po16, &sy, &ps };
    CUresult err = fcw_cuLaunchKernel(stream_ctx->kern_xchg_2d, gx, (unsigned int)height, 2,
                                    threads, 1, 1, 0, stream_ctx->stream, args, NULL);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] xchg_2d launch failed: %d\n", (int)err);
        return -1;
    }
    return 0;
}

/* ================================================================
 * Quant-block repack (host-side, upload-time)
 *
 * The GGUF block layouts and scale unpacking follow ggml (MIT License,
 * https://github.com/ggml-org/ggml), modified; see NOTICE.
 *
 * Converts raw GGUF quant blocks into the 4-byte-aligned device
 * layouts the matvec/MoE kernels read (documented in fc_cuda.h).
 * Numerically identical: scales are unpacked with the exact bit
 * selection of Pantheon/src/formats/gguf_dequant.c, and the fp16
 * block scales are decoded with the same integer-ALU half->float as
 * the kernels used on-device (pn_half_to_float), so the kernels see
 * bit-identical f32/int values in the new positions.
 * ================================================================ */

/* Bit-identical to pn_half_to_float (gguf_dequant.c) / the device
 * fc_half_to_float the kernels previously ran per block. */
static float fc_repack_half_to_float(unsigned short h)
{
    unsigned int sign = ((unsigned int)(h & 0x8000)) << 16;
    int exp = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    unsigned int bits;
    float out;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                          /* zero */
        } else {
            /* subnormal: normalize into fp32 */
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x400u;
            bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / nan */
    } else {
        bits = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void fc_repack_store_f32(unsigned char *dst, float v)
{
    memcpy(dst, &v, sizeof(v));
}

static void fc_repack_store_u32(unsigned char *dst, unsigned int v)
{
    memcpy(dst, &v, sizeof(v));
}

/* Q3_K: hmask[32] @0, qs[64] @32, scales[12] @96, d fp16 @108  (110 B)
 *   ->  u32 qw[32] @0, u8 sc[16] @128, f32 d @144, pad @148     (152 B)
 *
 * MAGIC-DEQUANT layout (sm_37: int-mul / int->f32 cvt crawl at 32/clk, so
 * the kernels build floats via the 2^23 mantissa-OR trick instead):
 *   qw[l] nibble i (i = 4n+j, i.e. element e = 32i + l) holds
 *     u = qv | hbit<<2   with qv = (qs[32n+l] >> 2j) & 3,
 *                             hbit = (hm[l] >> (4n+j)) & 1
 *   so the dequant value (qv - hb, hb = hbit?0:4) is exactly u - 4: the
 *   kernel does __int_as_float(0x4B000000|u) - (2^23 + 4), one OR + FSUB.
 *   sc[16] stores the RAW 6-bit scales (0..63, the -32 folded into the
 *   kernel's magic constant), reordered so lane half lh reads its 8 bytes
 *   contiguously: sc[8*lh + i] = raw6[2i + lh] (one uint2 load per lane). */
static void fc_repack_block_q3k(const unsigned char *s, unsigned char *d)
{
    const unsigned char *hm = s;
    const unsigned char *qs = s + 32;
    const unsigned char *sc = s + 96;
    unsigned char raw6[16];
    int i, l, n, j;

    /* Same bit selection as fc_q3k_unpack_scales / gguf_dequant.c, minus
     * the -32 bias (folded into the kernel's magic FSUB constant). */
    for (i = 0; i < 4; i++) {
        raw6[i]      = (unsigned char)((sc[i]     & 0xF) | (((sc[8 + i] >> 0) & 3) << 4));
        raw6[i + 4]  = (unsigned char)((sc[i + 4] & 0xF) | (((sc[8 + i] >> 2) & 3) << 4));
        raw6[i + 8]  = (unsigned char)((sc[i]     >> 4)  | (((sc[8 + i] >> 4) & 3) << 4));
        raw6[i + 12] = (unsigned char)((sc[i + 4] >> 4)  | (((sc[8 + i] >> 6) & 3) << 4));
    }
    for (l = 0; l < 32; l++) {
        unsigned int w = 0;
        for (n = 0; n < 2; n++)
            for (j = 0; j < 4; j++) {
                unsigned int qv = (qs[32 * n + l] >> (2 * j)) & 3u;
                unsigned int hb = (hm[l] >> (4 * n + j)) & 1u;
                w |= (qv | (hb << 2)) << (4 * (4 * n + j));
            }
        fc_repack_store_u32(d + 4 * l, w);
    }
    for (i = 0; i < 8; i++) {
        d[128 + i]     = raw6[2 * i];             /* lane half 0 */
        d[128 + 8 + i] = raw6[2 * i + 1];         /* lane half 1 */
    }
    fc_repack_store_f32(d + 144, fc_repack_half_to_float(
        (unsigned short)(s[108] | (s[109] << 8))));
    memset(d + 148, 0, 4);                        /* pad to 152 */
}

/* Q4_K/Q5_K 6-bit scale+min unpack — value-identical to unpack_k_scales
 * (gguf_dequant.c) / the device fc_kq_scale_min. */
static void fc_repack_k_scales(const unsigned char *sr,
                               unsigned char sc8[8], unsigned char mn8[8])
{
    int t;
    for (t = 0; t < 8; t++) {
        if (t < 4) {
            sc8[t] = (unsigned char)(sr[t] & 63);
            mn8[t] = (unsigned char)(sr[t + 4] & 63);
        } else {
            sc8[t] = (unsigned char)((sr[t + 4] & 0xF) | ((sr[t - 4] >> 6) << 4));
            mn8[t] = (unsigned char)((sr[t + 4] >> 4)  | ((sr[t]     >> 6) << 4));
        }
    }
}

/* Q4_K: d fp16 @0, dmin fp16 @2, scales[12] @4, qs[128] @16      (144 B)
 *   ->  u32 qw[32] @0, u8 sc[8] @128, u8 mn[8] @136, f32 d @144,
 *       f32 dmin @148                                            (152 B) */
static void fc_repack_block_q4k(const unsigned char *s, unsigned char *d)
{
    const unsigned char *qs = s + 16;
    unsigned char sc8[8], mn8[8];
    int l;

    for (l = 0; l < 32; l++)
        fc_repack_store_u32(d + 4 * l,
                            (unsigned int)qs[l] |
                            ((unsigned int)qs[32 + l] << 8) |
                            ((unsigned int)qs[64 + l] << 16) |
                            ((unsigned int)qs[96 + l] << 24));
    fc_repack_k_scales(s + 4, sc8, mn8);
    memcpy(d + 128, sc8, 8);
    memcpy(d + 136, mn8, 8);
    fc_repack_store_f32(d + 144, fc_repack_half_to_float(
        (unsigned short)(s[0] | (s[1] << 8))));
    fc_repack_store_f32(d + 148, fc_repack_half_to_float(
        (unsigned short)(s[2] | (s[3] << 8))));
}

/* Q5_K: d fp16 @0, dmin fp16 @2, scales[12] @4, qh[32] @16,
 *       qs[128] @48                                              (176 B)
 *   ->  u32 qw[32] @0, u8 qh[32] @128, u8 sc[8] @160, u8 mn[8] @168,
 *       f32 d @176, f32 dmin @180                                (184 B) */
static void fc_repack_block_q5k(const unsigned char *s, unsigned char *d)
{
    const unsigned char *qs = s + 48;
    unsigned char sc8[8], mn8[8];
    int l;

    for (l = 0; l < 32; l++)
        fc_repack_store_u32(d + 4 * l,
                            (unsigned int)qs[l] |
                            ((unsigned int)qs[32 + l] << 8) |
                            ((unsigned int)qs[64 + l] << 16) |
                            ((unsigned int)qs[96 + l] << 24));
    memcpy(d + 128, s + 16, 32);                  /* qh */
    fc_repack_k_scales(s + 4, sc8, mn8);
    memcpy(d + 160, sc8, 8);
    memcpy(d + 168, mn8, 8);
    fc_repack_store_f32(d + 176, fc_repack_half_to_float(
        (unsigned short)(s[0] | (s[1] << 8))));
    fc_repack_store_f32(d + 180, fc_repack_half_to_float(
        (unsigned short)(s[2] | (s[3] << 8))));
}

/* Q6_K: ql[128] @0, qh[64] @128, s8 scales[16] @192, d fp16 @208 (210 B)
 *   ->  u32 qlw[32] @0, u16 qh2[32] @128, u8 sc[16] @192,
 *       f32 d @208, pad @212                                     (216 B)
 *
 * MAGIC-DEQUANT scales (see fc_repack_block_q3k): the s8 scales are stored
 * biased +128 (u8) and reordered so lane half lh reads its 8 bytes in one
 * uint2 load: sc[8*lh + i] = s8sc[2i + lh] + 128 (i = 4*half + quad). The
 * kernel recovers the signed value with one OR + FSUB of (2^23 + 128),
 * replacing the per-element s8 __ldg + int-mul + int->f32 cvt. */
static void fc_repack_block_q6k(const unsigned char *s, unsigned char *d)
{
    const unsigned char *ql = s;
    const unsigned char *qh = s + 128;
    const signed char *sc = (const signed char *)(s + 192);
    int l, i;

    for (l = 0; l < 32; l++)
        fc_repack_store_u32(d + 4 * l,
                            (unsigned int)ql[l] |
                            ((unsigned int)ql[32 + l] << 8) |
                            ((unsigned int)ql[64 + l] << 16) |
                            ((unsigned int)ql[96 + l] << 24));
    for (l = 0; l < 32; l++) {
        d[128 + 2 * l]     = qh[l];
        d[128 + 2 * l + 1] = qh[32 + l];
    }
    for (i = 0; i < 8; i++) {
        d[192 + i]     = (unsigned char)((int)sc[2 * i]     + 128);
        d[192 + 8 + i] = (unsigned char)((int)sc[2 * i + 1] + 128);
    }
    fc_repack_store_f32(d + 208, fc_repack_half_to_float(
        (unsigned short)(s[208] | (s[209] << 8))));
    memset(d + 212, 0, 4);                        /* pad to 216 */
}

/* Q8_0: d fp16 @0, s8 qs[32] @2 (34 B)  ->  f32 d @0, s8 qs[32] @4 (36 B) */
static void fc_repack_block_q80(const unsigned char *s, unsigned char *d)
{
    fc_repack_store_f32(d, fc_repack_half_to_float(
        (unsigned short)(s[0] | (s[1] << 8))));
    memcpy(d + 4, s + 2, 32);
}

/* Q8_0 G4: four 36-B repacked blocks (144 B) -> [f32 d0..d3] then, for lane
 * s = 0..7 of a row's 8-lane group, bytes 4s..4s+3 of blocks 0..3 (16 B), so
 * each lane reads its 16 quant bytes and the 4 scales with two 128-bit loads
 * (fc_q80_body, fc_fu_q80_seg). In place. */
static void fc_repack_group_q80_g4(unsigned char *g)
{
    unsigned char t[144];
    int j, s;
    for (j = 0; j < 4; j++) {
        memcpy(t + 4 * j, g + 36 * j, 4);
        for (s = 0; s < 8; s++)
            memcpy(t + 16 + 16 * s + 4 * j, g + 36 * j + 4 + 4 * s, 4);
    }
    memcpy(g, t, sizeof(t));
}

size_t fc_repack_row_bytes(int gguf_quant, int cols)
{
    switch (gguf_quant) {
    case 8:  return (cols % 32)  ? 0 : (size_t)(cols / 32)  * 36;  /* Q8_0 */
    case 42:  /* fallthrough: g64-era id */
    case 142: return (cols % 128) ? 0 : (size_t)(cols / 128) * 36;  /* Q2_0_g128 (shipped id 142) */
    case 11: return (cols % 256) ? 0 : (size_t)(cols / 256) * 152; /* Q3_K */
    case 12: return (cols % 256) ? 0 : (size_t)(cols / 256) * 152; /* Q4_K */
    case 13: return (cols % 256) ? 0 : (size_t)(cols / 256) * 184; /* Q5_K */
    case 14: return (cols % 256) ? 0 : (size_t)(cols / 256) * 216; /* Q6_K */
    default: return 0;
    }
}

int fc_repack_weight(int gguf_quant, const void *src, void *dst,
                     long rows, int cols)
{
    const unsigned char *sp = src;
    unsigned char *dp = dst;
    size_t sblk, dblk;
    void (*fn)(const unsigned char *, unsigned char *);
    long r;
    int b, nb;

    if (!src || !dst || rows <= 0 ||
        fc_repack_row_bytes(gguf_quant, cols) == 0)
        return -1;

    switch (gguf_quant) {
    case 8:  sblk = 34;  dblk = 36;  fn = fc_repack_block_q80; break;
    case 42:  /* fallthrough: g64-era id */
    case 142: sblk = 34;  dblk = 36;  fn = fc_repack_block_q80; break; /* Q2_0_g128 (shipped id 142): same [fp16 d][32B] -> [f32 d][32B] repack */
    case 11: sblk = 110; dblk = 152; fn = fc_repack_block_q3k; break;
    case 12: sblk = 144; dblk = 152; fn = fc_repack_block_q4k; break;
    case 13: sblk = 176; dblk = 184; fn = fc_repack_block_q5k; break;
    case 14: sblk = 210; dblk = 216; fn = fc_repack_block_q6k; break;
    default: return -1;
    }
    nb = (gguf_quant == 8) ? cols / 32 :
         (gguf_quant == 42 || gguf_quant == 142) ? cols / 128 : cols / 256;

    for (r = 0; r < rows; r++)
        for (b = 0; b < nb; b++)
            fn(sp + ((size_t)r * nb + b) * sblk,
               dp + ((size_t)r * nb + b) * dblk);
    if (gguf_quant == 8 && cols % 128 == 0)   /* row bytes unchanged */
        for (r = 0; r < rows; r++)
            for (b = 0; b < nb; b += 4)
                fc_repack_group_q80_g4(dp + ((size_t)r * nb + b) * 36);
    return 0;
}

size_t fc_repack_q80_parity_row_bytes(int cols)
{
    return (cols <= 0 || cols % 128) ? 0 : (size_t)(cols / 128) * 80;
}

int fc_repack_weight_q80_parity(const void *src, void *dst, long rows, int cols, int parity)
{
    const unsigned char *sp = src;
    unsigned char *dp = dst;
    const size_t row_bytes = fc_repack_q80_parity_row_bytes(cols);

    if (!src || !dst || rows <= 0 || row_bytes == 0 || (parity != 0 && parity != 1))
        return -1;
    for (long r = 0; r < rows; r++) {
        for (int g = 0; g < cols / 128; g++) {
            const unsigned char *s = sp + ((size_t)r * (cols / 32) + 4 * (size_t)g) * 34;
            unsigned char *d = dp + (size_t)r * row_bytes + (size_t)g * 80;
            for (int j = 0; j < 4; j++) {
                const unsigned char *b = s + 34 * j;
                fc_repack_store_f32(d + 4 * j, fc_repack_half_to_float(
                    (unsigned short)(b[0] | (b[1] << 8))));
                for (int sl = 0; sl < 8; sl++) {
                    d[16 + 8 * sl + 2 * j] = b[2 + 4 * sl + parity];
                    d[16 + 8 * sl + 2 * j + 1] = b[2 + 4 * sl + parity + 2];
                }
            }
        }
    }
    return 0;
}

/* Q8_0 EXPERT layout: 8 consecutive Q8_0 blocks (256 elements, 8 x 34 B) ->
 * one 288-B super-block [8 x f32 d][256 x s8 q]. Same bytes as eight dense
 * 36-B repacked blocks, regrouped so a warp reads the quants as full 128-B
 * transactions (moe_expert_group.cu, FC_DEF_Q80_GRP2). */
static void fc_moe_repack_sb_q80(const unsigned char *s, unsigned char *d)
{
    int b;
    for (b = 0; b < 8; b++) {
        const unsigned char *sb = s + 34 * b;
        fc_repack_store_f32(d + 4 * b, fc_repack_half_to_float(
            (unsigned short)(sb[0] | (sb[1] << 8))));
        memcpy(d + 32 + 32 * b, sb + 2, 32);
    }
}

size_t fc_moe_repack_row_bytes(int gguf_quant, int cols)
{
    if (gguf_quant == 8)
        return (cols % 256) ? 0 : (size_t)(cols / 256) * 288;
    return fc_repack_row_bytes(gguf_quant, cols);
}

int fc_moe_repack_weight(int gguf_quant, const void *src, void *dst,
                         long rows, int cols)
{
    const unsigned char *sp = src;
    unsigned char *dp = dst;
    long r;
    int b, nb;

    if (gguf_quant != 8)
        return fc_repack_weight(gguf_quant, src, dst, rows, cols);
    if (!src || !dst || rows <= 0 || fc_moe_repack_row_bytes(8, cols) == 0)
        return -1;
    nb = cols / 256;
    for (r = 0; r < rows; r++)
        for (b = 0; b < nb; b++)
            fc_moe_repack_sb_q80(sp + ((size_t)r * nb + b) * 272,
                                 dp + ((size_t)r * nb + b) * 288);
    return 0;
}

/* ================================================================
 * cuBLAS SGEMM / SGEMV (per-context)
 *
 * cuBLAS is column-major. For row-major C = A @ B:
 *   Column-major equivalent: C^T = B^T @ A^T
 *   Which is: cublasSgemm(N, N, N, M, K, ..., B, N, A, K, ..., C, N)
 * ================================================================ */

void fc_sgemm_ctx(fc_cuda_ctx *ctx, fc_buffer *A, fc_buffer *B, fc_buffer *C,
                  int M, int N, int K, float alpha, float beta)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->cublas || !A || !B || !C)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;

    /* Row-major C[M,N] = A[M,K] @ B[K,N]
     * → col-major: C_col[N,M] = B_col[N,K] @ A_col[K,M]
     * cublasSgemm(op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
     * where m=N, n=M, k=K, A=B_buf, lda=N, B=A_buf, ldb=K, C=C_buf, ldc=N */
    cublasStatus_t err = p_cublasSgemm(
        ctx->cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (const float *)(uintptr_t)B->dptr, N,  /* B as col-major [N,K] */
        (const float *)(uintptr_t)A->dptr, K,  /* A as col-major [K,M] */
        &beta,
        (float *)(uintptr_t)C->dptr, N          /* C as col-major [N,M] */
    );

    fc_ctx_leave();
    if (err != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[fc_cuda] cublasSgemm failed: %d\n", err);
}

void fc_sgemv_ctx(fc_cuda_ctx *ctx, fc_buffer *A, fc_buffer *x, fc_buffer *y,
                  int M, int K, float alpha, float beta)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->cublas || !A || !x || !y)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;

    /* Row-major y[M] = A[M,K] @ x[K]
     * In column-major, A stored as A^T[K,M].
     * cublasSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
     * Use trans=T: y = A^T @ x where A^T is [M,K] col-major = our row-major A
     * m=K, n=M (dimensions of the column-major matrix before transpose) */
    cublasStatus_t err = p_cublasSgemv(
        ctx->cublas,
        CUBLAS_OP_T,
        K, M,             /* m=K, n=M (col-major A is [K,M]) */
        &alpha,
        (const float *)(uintptr_t)A->dptr, K,  /* A in col-major = A^T[K,M], lda=K */
        (const float *)(uintptr_t)x->dptr, 1,  /* x, stride 1 */
        &beta,
        (float *)(uintptr_t)y->dptr, 1          /* y, stride 1 */
    );

    fc_ctx_leave();
    if (err != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "[fc_cuda] cublasSgemv failed: %d\n", err);
}

void fc_sgemm(fc_buffer *A, fc_buffer *B, fc_buffer *C,
              int M, int N, int K, float alpha, float beta)
{
    fc_sgemm_ctx(_fc_default_ctx, A, B, C, M, N, K, alpha, beta);
}

void fc_sgemv(fc_buffer *A, fc_buffer *x, fc_buffer *y,
              int M, int K, float alpha, float beta)
{
    fc_sgemv_ctx(_fc_default_ctx, A, x, y, M, K, alpha, beta);
}

/* ================================================================
 * Custom kernel dispatch helper
 *
 * Launches a 1-D grid on ctx's stream. The caller must already have
 * ctx current (the kernel wrappers bracket the whole op).
 * ================================================================ */

static void fc_dispatch_1d(const fc_cuda_ctx *ctx, CUfunction func,
                           void **args, int total_threads)
{
    if (!func || total_threads <= 0)
        return;

    int threads_per_block = 256;
    if (total_threads < threads_per_block)
        threads_per_block = total_threads;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    CUresult err = fcw_cuLaunchKernel(
        func,
        (unsigned int)num_blocks, 1, 1,            /* grid */
        (unsigned int)threads_per_block, 1, 1,      /* block */
        0, ctx->stream, args, NULL
    );

    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] cuLaunchKernel failed: %d\n", err);
}

/* ================================================================
 * Custom kernels (per-context)
 * ================================================================ */

void fc_add_ctx(fc_cuda_ctx *ctx, fc_buffer *a, fc_buffer *b, fc_buffer *out, int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_add || !a || !b || !out)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &a->dptr, &b->dptr, &out->dptr, &n };
    fc_dispatch_1d(ctx, ctx->kern_add, args, n);
    fc_ctx_leave();
}

void fc_mul_ctx(fc_cuda_ctx *ctx, fc_buffer *a, fc_buffer *b, fc_buffer *out, int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_mul || !a || !b || !out)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &a->dptr, &b->dptr, &out->dptr, &n };
    fc_dispatch_1d(ctx, ctx->kern_mul, args, n);
    fc_ctx_leave();
}

void fc_silu_ctx(fc_cuda_ctx *ctx, fc_buffer *x, fc_buffer *out, int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_silu || !x || !out)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &x->dptr, &out->dptr, &n };
    fc_dispatch_1d(ctx, ctx->kern_silu, args, n);
    fc_ctx_leave();
}

void fc_rms_norm_ctx(fc_cuda_ctx *ctx, fc_buffer *x, fc_buffer *weight,
                     fc_buffer *out, int n, float eps)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_rms_norm || !x || !out)
        return;

    /* RMS norm uses one threadgroup per normalization. For single-vector
     * (batch=1), launch 1 block with enough threads to reduce n elements. */
    int threads = 256;
    if (n < threads)
        threads = n;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));
    CUdeviceptr w_ptr = weight ? weight->dptr : 0;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &x->dptr, &w_ptr, &out->dptr, &n, &eps };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_rms_norm,
        1, 1, 1,                                /* 1 block */
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] rms_norm launch failed: %d\n", err);
}

void fc_softmax_ctx(fc_cuda_ctx *ctx, fc_buffer *x, fc_buffer *out,
                    int rows, int cols)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_softmax || !x || !out)
        return;

    /* One block per row, threads process columns cooperatively */
    int threads = 256;
    if (cols < threads)
        threads = cols;
    /* Shared memory: threads * sizeof(float) * 2 (max and sum reductions) */
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float) * 2);

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &x->dptr, &out->dptr, &rows, &cols };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_softmax,
        (unsigned int)rows, 1, 1,               /* one block per row */
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] softmax launch failed: %d\n", err);
}

void fc_matvec_q8_ctx(fc_cuda_ctx *ctx, fc_buffer *x_q, fc_buffer *w,
                      fc_buffer *w_scales, fc_buffer *out,
                      float x_scale, int rows, int cols)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_matvec_q8 ||
        !x_q || !w || !w_scales || !out)
        return;

    /* One block per output row. Within each block, threads cooperate to
     * compute the dot product of x_q[cols] with w[row, cols], then reduce. */
    int threads = 256;
    if (cols < threads)
        threads = cols;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &x_q->dptr, &w->dptr, &w_scales->dptr, &out->dptr,
        &x_scale, &rows, &cols
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_matvec_q8,
        (unsigned int)rows, 1, 1,               /* one block per row */
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_q8 launch failed: %d\n", err);
}

/* ================================================================
 * qwen35moe kernels (per-context)
 *
 * Launch configs are transcribed from the kernel .cu comments; the
 * cited line ranges in the cuda_kernels .cu files pin each launch.
 * ================================================================ */

void fc_matvec_q3k_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_matvec_q3k ||
        !w || !x || !out)
        return;

    /* matvec_q3k.cu fc_matvec_q3k: 8 warps (256 threads) per block, FOUR
     * output rows per warp (element-interleaved independent chains for
     * dual-issue ILP), __shfl_down reduction per row, staged-x shared =
     * min(cols, 2048) floats. Grid = ceil(rows / 16). */
    int threads = 256;
    int sh_floats = cols < 2048 ? cols : 2048;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_matvec_q3k,
        (unsigned int)((rows + 15) / 16), 1, 1, /* row pair per warp */
        (unsigned int)threads, 1, 1,
        (unsigned int)(sh_floats * sizeof(float)), ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_q3k launch failed: %d\n", err);
}

/* Peer-stream variants: identical output, more loads in flight. rows per
 * block = 8 warps * rows-per-warp, so the grid divisor is the only launch
 * difference from fc_matvec_q3k_ctx. */
static const struct { const char *name; int rows_per_block; }
fc_q3k_ps_tab[FC_Q3K_PS_COUNT] = {
    { "r2p1", 16 }, { "r2p2", 16 }, { "r2p4", 16 }, { "r2p8", 16 },
    { "r1p1",  8 }, { "r1p4",  8 }, { "r1p8",  8 },
};

const char *fc_matvec_q3k_ps_name(int variant)
{
    if (variant < 0 || variant >= FC_Q3K_PS_COUNT)
        return NULL;
    return fc_q3k_ps_tab[variant].name;
}

int fc_matvec_q3k_ps_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                         fc_buffer *out, int rows, int cols, int row_stride,
                         int variant)
{
    int threads = 256;
    int sh_floats = cols < 2048 ? cols : 2048;
    int rpb;
    CUfunction kern;
    CUresult err;

    if (variant < 0 || variant >= FC_Q3K_PS_COUNT)
        return -1;
    if (!fc_cuda_ctx_available(ctx) || !w || !x || !out)
        return -1;
    kern = ctx->kern_matvec_q3k_ps[variant];
    if (!kern)
        return -1;                      /* absent from this build's PTX */
    rpb = fc_q3k_ps_tab[variant].rows_per_block;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    {
        void *args[] = {
            &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride
        };
        err = fcw_cuLaunchKernel(
            kern,
            (unsigned int)((rows + rpb - 1) / rpb), 1, 1,
            (unsigned int)threads, 1, 1,
            (unsigned int)(sh_floats * sizeof(float)), ctx->stream, args, NULL
        );
    }
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] matvec_q3k_ps(%s) launch failed: %d\n",
                fc_q3k_ps_tab[variant].name, err);
        return -1;
    }
    return 0;
}

/* Extra shared floats the MIN-LIFTED K-quant matvecs need in FRONT of the
 * staged x: one activation sub-block sum per 32 staged elements (2048/32).
 * Must match FC_KQ_XSUM_FLOATS in cuda_kernels/matvec_kquants.cu. */
#define FC_KQ_XSUM_FLOATS_HOST 64

/* Generic launcher for the quantized matvec_kquants.cu family — every kernel
 * shares the (w, x, out, rows, cols, row_stride) signature and the 8-warp
 * (256-thread) block with staged-x shared = min(cols, 2048) floats.
 * rows_per_block: 16 for the K-quant kernels (element-interleaved row
 * pair per warp for dual-issue ILP), 32 for fc_matvec_q80 (8 lanes per
 * row). Grid = ceil(rows / rpb).
 * extra_sh_floats: prefix floats ahead of the staged x —
 * FC_KQ_XSUM_FLOATS_HOST for the min-lifted kernels, 0 for the rest.
 * NOT used for fc_matvec_f32, which keeps the one-block-per-row shape. */
static void fc_matvec_kq_launch(fc_cuda_ctx *ctx, CUfunction kern,
                                const char *name, fc_buffer *w, fc_buffer *x,
                                fc_buffer *out, int rows, int cols,
                                int row_stride, int rows_per_block,
                                int extra_sh_floats)
{
    if (!fc_cuda_ctx_available(ctx) || !kern || !w || !x || !out)
        return;

    int threads = 256;
    int sh_floats = (cols < 2048 ? cols : 2048) + extra_sh_floats;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)((rows + rows_per_block - 1) / rows_per_block), 1, 1,
        (unsigned int)threads, 1, 1,
        (unsigned int)(sh_floats * sizeof(float)), ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] %s launch failed: %d\n", name, err);
}

void fc_matvec_q4k_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride)
{
    /* MIN-LIFTED (matvec_kquants.cu): needs the xsum prefix in shared. */
    fc_matvec_kq_launch(ctx, ctx ? ctx->kern_matvec_q4k : NULL, "matvec_q4k",
                        w, x, out, rows, cols, row_stride, 16,
                        FC_KQ_XSUM_FLOATS_HOST);
}

void fc_matvec_q5k_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride)
{
    fc_matvec_kq_launch(ctx, ctx ? ctx->kern_matvec_q5k : NULL, "matvec_q5k",
                        w, x, out, rows, cols, row_stride, 16, 0);
}

void fc_matvec_q6k_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride)
{
    fc_matvec_kq_launch(ctx, ctx ? ctx->kern_matvec_q6k : NULL, "matvec_q6k",
                        w, x, out, rows, cols, row_stride, 16, 0);
}

/* Rows per block of the fc_matvec_q80 family: 8 lanes per row, so the 256-thread block
 * covers 32 rows. Must match FC_Q80_ROWS_PER_BLOCK in cuda_kernels/matvec_kquants.cu. */
#define FC_Q80_ROWS_PER_BLOCK_HOST 32

void fc_matvec_q80_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride)
{
    fc_matvec_kq_launch(ctx, ctx ? ctx->kern_matvec_q80 : NULL, "matvec_q80",
                        w, x, out, rows, cols, row_stride,
                        FC_Q80_ROWS_PER_BLOCK_HOST, 0);
}

/* Bonsai native-ternary Q2_0_g128 (GGUF type 42). Row pair per warp (16 rows/block);
 * conditional add/sub, no dequant. */
void fc_matvec_q2_0_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                        fc_buffer *out, int rows, int cols, int row_stride)
{
    fc_matvec_kq_launch(ctx, ctx ? ctx->kern_matvec_q2_0 : NULL, "matvec_q2_0",
                        w, x, out, rows, cols, row_stride, 16, 0);
}

void fc_matvec_f32_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride)
{
    /* fc_matvec_f32 keeps the original one-block-per-row + shared-mem tree
     * reduction shape (plain fp32 rows coalesce fine; measured 65 GB/s). */
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_matvec_f32 || !w || !x || !out)
        return;

    int threads = 256;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_matvec_f32,
        (unsigned int)rows, 1, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_f32 launch failed: %d\n", err);
}

/* F16 dense rows (gguf type 1) — the attention projections, shared expert,
 * router and LM head of an F16 model. Same one-block-per-row + shared-tree
 * shape as fc_matvec_f32; the weight is 2 bytes widened in-register, because
 * sm_37 has no fp16 arithmetic. row_stride is bytes (cols * 2), NOT repacked.
 * Silently does nothing when the kernel is absent (a pre-F16 fc_cuda_ptx.h),
 * which is why the CALLER must keep its own type switch: an F16 weight with no
 * kernel has to reach the host fallback, not a no-op. */
void fc_matvec_f16_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_matvec_f16 || !w || !x || !out)
        return;

    int threads = 256;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_matvec_f16,
        (unsigned int)rows, 1, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_f16 launch failed: %d\n", err);
}

int fc_matvec_f16_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_matvec_f16 &&
           ctx->kern_matvec_f16_nx;
}

/* BF16 dense rows (gguf type 30) — fc_matvec_f16_ctx with the shift-widen.
 * Same silent-when-absent contract, so the caller keeps its own type switch. */
void fc_matvec_bf16_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                        fc_buffer *out, int rows, int cols, int row_stride)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_matvec_bf16 ||
        !w || !x || !out)
        return;

    int threads = 256;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_matvec_bf16,
        (unsigned int)rows, 1, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_bf16 launch failed: %d\n", err);
}

int fc_matvec_bf16_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_matvec_bf16 &&
           ctx->kern_matvec_bf16_nx;
}

/* ================================================================
 * Batched (multi-column) matvecs — fc_matvec_*_nx
 *
 * out[t][rows] = dequant(w) . x[t] for t in [0, n). x is [n][cols]
 * row-major, out [n][rows]. Grid.y tiles the batch FC_NX_COLS columns
 * at a time; per column the kernels are bit-identical to their
 * single-column counterparts (see matvec_q3k.cu / matvec_kquants.cu).
 *
 * L6: two tile widths — 4 (the original _nx kernels) and 8 (_nx8).
 * n >= 5 dispatches to the 8-wide tile so a K=8 batched-decode step
 * streams the shared weights ONCE (and K=5-7 stops paying a second
 * stream); n <= 4 stays on the 4-wide tile (the 8-wide kernel's extra
 * FMA chains cost ~25% at small n on the issue-bound sm_37). Both are
 * bit-exact per column, so the dispatch never changes results.
 *
 * L10 (n == 1): the tile width is a COMPILE-time template parameter, so a
 * 4-wide tile fed one live column still runs all four FMA chains per
 * element (three of them against shared memory the staging loop zero-fills)
 * and still stages 4 * FC_NX_CHUNK_HOST floats per chunk. On the
 * issue-bound sm_37 that is ~1.6x (K-quant projections) to ~2.1x (the wide
 * Q6_K LM head) the cost of the dedicated single-column kernel — measured
 * as the dominant defect of batched decode at K=1 (results/phase0_baseline.md
 * §6). At n == 1 the _nx memory layout degenerates exactly to the
 * single-column one (x = x[0], out = out[0]), so we simply dispatch the
 * single-column kernel, which the _nx kernels are already documented and
 * tested to be bit-identical to per column. Gate FC_NX1_DIRECT=0 to A/B.
 * ================================================================ */

#define FC_NX2_COLS_HOST 2
#define FC_NX_COLS_HOST 4
#define FC_NX8_COLS_HOST 8
#define FC_NX_CHUNK_HOST 512

/* Per-column shared prefix of the MIN-LIFTED batched K-quant matvecs: one
 * activation sub-block sum per 32 staged elements of a column's chunk.
 * Must match FC_NX_XSUM_PER_COL in cuda_kernels/matvec_kquants.cu. */
#define FC_NX_XSUM_PER_COL_HOST 16

/* L10 A/B gate: 0 keeps every n on the tiled _nx kernels (the pre-L10
 * behavior). Latched once — these launchers sit inside captured graphs. */
static int fc_nx1_direct(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("FC_NX1_DIRECT");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Smallest n at which the MIN-LIFTED Q4_K batched matvec is worth the 8-wide
 * tile. Q4_K is the one quant whose _nx8 kernel pays a register price for the
 * min-lift (2*NC min accumulators -> 125 registers, 4 resident blocks instead
 * of 6), so unlike q3k/q5k/q6k/q80 its crossover is NOT at n = 5: the 8-wide
 * tile loses on narrow batches and wins on wide ones, where halving the number
 * of weight streams outweighs the occupancy loss. Measured with
 * cuda/tests/test_matvec_nx.c on die 7 at 875 MHz, alternating batteries,
 * spread < 0.5% -- 8-wide relative to 4-wide, so + means 8-wide is faster:
 *
 *              n=12    n=16    n=32    n=64    n=128   n=256
 *   qkv        -23.3%  +6.4%   +6.2%    --     +6.6%   +6.9%     8192x2048
 *   down_shexp -22.9%  +1.3%   +2.1%   +2.7%   +4.6%    --       2048x512
 *
 * n=12 is the pathology, and it is tile QUANTIZATION, not width: 12 columns
 * fill the 4-wide tile exactly (3 passes) while the 8-wide one runs 2 passes
 * with a quarter of its lanes idle, so the weight-stream saving falls to 1.5x
 * and no longer pays for the lost occupancy. That is the case `9aac7b1`
 * measured, and its numbers reproduce here exactly (0.087 vs 0.107 ms) -- it
 * was right about n=12 and wrong to generalise from it.
 *
 * The prefill chunk is QWEN35_BS_PREFILL_CHUNK = 128 tokens and its LAST chunk
 * is total_tokens % 128, so the threshold has to hold for every width in
 * between. The turn is already positive at n=16 on both shapes; 32 is the
 * conservative choice, measured positive on both, and well clear of the n=12
 * pathology and of batched decode (saber max_sessions=2, so n<=2 there).
 * Both tiles are bit-exact per column (test_matvec_nx gates it in BOTH
 * dispatch modes), so this only ever trades speed, never numerics.
 *
 * FC_Q4K_NX8_MIN_N overrides it for A/B work; 0 disables the 8-wide tile
 * entirely (the pre-fix behaviour). Latched once — these launchers sit inside
 * captured graphs and must not change shape between capture and replay. */
#define FC_Q4K_NX8_MIN_N_DEFAULT 32

static int fc_q4k_nx8_min_n(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("FC_Q4K_NX8_MIN_N");
        v = env ? atoi(env) : FC_Q4K_NX8_MIN_N_DEFAULT;
        if (v < 0)
            v = 0;
    }
    return v;
}

/* Tile width for a batch of n columns: 8-wide when n >= 5, 2-wide at n == 2,
 * 4-wide otherwise. Each optional width is used only when its kernel exists
 * (NULL on an older fc_cuda_ptx.h), so the dispatch degrades to the 4-wide
 * tile rather than failing. Every width is bit-exact per column. */
static int fc_nx_tile_cols(int n, CUfunction kern8, CUfunction kern2)
{
    if (n >= 5 && kern8)
        return FC_NX8_COLS_HOST;
    if (n == 2 && kern2 && fc_nx1_direct())
        return FC_NX2_COLS_HOST;
    return FC_NX_COLS_HOST;
}

/* Shared launcher for the K-quant/Q8_0 _nx kernels (same signature).
 * rows_per_block: 16 for the row-pair kernels, 32 for q80 (8 lanes per row).
 * kern8 (optional) is the 8-wide-tile variant, used for n >= 5.
 * kern1 (optional) is the dedicated SINGLE-column kernel, used at n == 1
 * (same (w, x, out, rows, cols, row_stride) signature and block geometry —
 * see fc_matvec_kq_launch). kern1_extra_sh_floats is that kernel's shared
 * prefix (FC_KQ_XSUM_FLOATS_HOST for the min-lifted ones, else 0);
 * extra_sh_floats_per_col is the tiled kernels' PER-COLUMN prefix
 * (FC_NX_XSUM_PER_COL_HOST for the min-lifted ones, else 0). */
static void fc_matvec_nx_launch(fc_cuda_ctx *ctx, CUfunction kern,
                                CUfunction kern8, CUfunction kern2,
                                CUfunction kern1, const char *name,
                                fc_buffer *w, fc_buffer *x,
                                fc_buffer *out, int rows, int cols,
                                int row_stride, int n, int rows_per_block,
                                int kern1_extra_sh_floats,
                                int extra_sh_floats_per_col)
{
    if (!fc_cuda_ctx_available(ctx) || !kern || !w || !x || !out || n <= 0)
        return;

    if (n == 1 && kern1 && fc_nx1_direct()) {
        fc_matvec_kq_launch(ctx, kern1, name, w, x, out, rows, cols,
                            row_stride, rows_per_block,
                            kern1_extra_sh_floats);
        return;
    }

    int tile = fc_nx_tile_cols(n, kern8, kern2);
    if (tile == FC_NX8_COLS_HOST)
        kern = kern8;
    else if (tile == FC_NX2_COLS_HOST)
        kern = kern2;
    /* The tiles stage column c at c * FC_NX_CHUNK floats (a compile-time stride),
     * so a tile wider than one column needs the full chunk per column even when
     * cols is smaller. */
    int sh_floats = (tile > 1 || cols >= FC_NX_CHUNK_HOST ? FC_NX_CHUNK_HOST : cols) +
                    extra_sh_floats_per_col;
    unsigned int grid_y = (unsigned int)((n + tile - 1) / tile);

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride, &n
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)((rows + rows_per_block - 1) / rows_per_block), grid_y, 1,
        256, 1, 1,
        (unsigned int)(tile * sh_floats * sizeof(float)),
        ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] %s launch failed: %d\n", name, err);
}

void fc_matvec_q3k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n)
{
    fc_matvec_nx_launch(ctx, ctx ? ctx->kern_matvec_q3k_nx : NULL,
                        ctx ? ctx->kern_matvec_q3k_nx8 : NULL,
                        ctx ? ctx->kern_matvec_q3k_nx2 : NULL,
                        ctx ? ctx->kern_matvec_q3k : NULL,
                        "matvec_q3k_nx", w, x, out, rows, cols, row_stride,
                        n, 16, 0, 0);
}

void fc_matvec_q4k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n)
{
    /* The 8-wide Q4_K tile is dispatched BY BATCH WIDTH, not at the shared
     * n >= 5 threshold — see FC_Q4K_NX8_MIN_N_DEFAULT. The min-lift costs
     * fc_matvec_q4k_nx8 125 registers (from 80) and drops it from 6 resident
     * blocks per SM to 4, which loses at batched-decode widths and wins at
     * the 128-token prefill chunk. `ca97bd7` disabled the tile outright on a
     * 2048x512, n=12 measurement (8-wide before the min-lift 0.096 ms, after
     * 0.108, 4-wide after 0.088); that call was right for n=12 and wrong for
     * prefill, where it cost 22-27% of the Q4_K dense matvec time and ~4% of
     * TTFT. Below the threshold kern8 is NULL and fc_nx_tile_cols falls back
     * to the 4-wide tile. Both tiles are bit-exact per column. */
    CUfunction kern8 = (ctx && fc_q4k_nx8_min_n() &&
                        n >= fc_q4k_nx8_min_n())
                     ? ctx->kern_matvec_q4k_nx8 : NULL;

    fc_matvec_nx_launch(ctx, ctx ? ctx->kern_matvec_q4k_nx : NULL,
                        kern8,
                        ctx ? ctx->kern_matvec_q4k_nx2 : NULL,
                        ctx ? ctx->kern_matvec_q4k : NULL,
                        "matvec_q4k_nx", w, x, out, rows, cols, row_stride,
                        n, 16, FC_KQ_XSUM_FLOATS_HOST, FC_NX_XSUM_PER_COL_HOST);
}

void fc_matvec_q5k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n)
{
    fc_matvec_nx_launch(ctx, ctx ? ctx->kern_matvec_q5k_nx : NULL,
                        ctx ? ctx->kern_matvec_q5k_nx8 : NULL,
                        ctx ? ctx->kern_matvec_q5k_nx2 : NULL,
                        ctx ? ctx->kern_matvec_q5k : NULL,
                        "matvec_q5k_nx", w, x, out, rows, cols, row_stride,
                        n, 16, 0, 0);
}

void fc_matvec_q6k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n)
{
    fc_matvec_nx_launch(ctx, ctx ? ctx->kern_matvec_q6k_nx : NULL,
                        ctx ? ctx->kern_matvec_q6k_nx8 : NULL,
                        ctx ? ctx->kern_matvec_q6k_nx2 : NULL,
                        ctx ? ctx->kern_matvec_q6k : NULL,
                        "matvec_q6k_nx", w, x, out, rows, cols, row_stride,
                        n, 16, 0, 0);
}

/* Q8_0 row tiles: each thread carries 4 rows x NC columns, a block 128 rows x NC.
 * Bit-identical to the _nx kernels per output (same chains, same block order, same
 * shuffle tree). n = 3 / 4 / 5 take the exact-width r4c3 / r4c4 / r4c5 (the _nx tiles
 * compute 4 or 8 columns there; 1.03-1.79x on sm_37), n >= 6 takes r4c8 (1.23-1.50x at
 * the 128-token prefill chunk). n = 1 and 2 stay on the single-column and nx2 kernels.
 * G4 rows only (cols % 128 == 0, the only layout the tiles read). FC_Q80_TILE_MIN_N=<n>
 * raises the smallest n that takes a tile; 0 = off. */
#define FC_Q80_TILE_ROWS_HOST 128
#define FC_Q80_TILE_MIN_N_DEFAULT 3

static int fc_q80_tile_min_n(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("FC_Q80_TILE_MIN_N");
        v = e ? atoi(e) : FC_Q80_TILE_MIN_N_DEFAULT;
        if (v < 0)
            v = 0;
    }
    return v;
}

/* Grid rule. A row tile launches rows / 128 blocks and a SMX holds 2-3 of them, so a small
 * matrix leaves most of a 13-SMX K80 die idle; the _nx kernels launch 4x the blocks. On the
 * Qwen3.6 TP2 per-die shapes (sm_37, die A/B, all bit-identical) the _nx kernels win at
 * n = 3 / 4 up to 2048 rows (16 tile blocks: 1.06-1.70x) and lose at 4096 (0.84x at n = 3);
 * at n = 5, where _nx runs the 8-wide tile, they win only at 256 rows (1.57x). Same bits
 * either way (per-output identity above). FC_Q80_TILE_GRID_RULE=0 turns the rule off. */
#define FC_Q80_TILE_MIN_BLOCKS_N34 17
#define FC_Q80_TILE_MIN_BLOCKS_N5 3

static int fc_q80_tile_grid_rule(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("FC_Q80_TILE_GRID_RULE");
        v = e ? atoi(e) != 0 : 1;
    }
    return v;
}

/* Launch a row tile; 0 = launched, -1 = not applicable (caller runs _nx). */
static int fc_matvec_q80_tile_launch(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                                     fc_buffer *out, int rows, int cols, int row_stride,
                                     int n)
{
    const int min_n = fc_q80_tile_min_n();
    if (!fc_cuda_ctx_available(ctx) || !min_n || n < min_n || n < 3 || (cols & 127) ||
        !w || !x || !out)
        return -1;
    CUfunction kern = n == 3 ? ctx->kern_matvec_q80_r4c3 : n == 4 ? ctx->kern_matvec_q80_r4c4
                    : n == 5 ? ctx->kern_matvec_q80_r4c5 : ctx->kern_matvec_q80_r4c8;
    const int nc = n <= 5 ? n : 8;
    if (!kern)
        return -1;
    const int blocks = (rows + FC_Q80_TILE_ROWS_HOST - 1) / FC_Q80_TILE_ROWS_HOST;
    if (n <= 5 && fc_q80_tile_grid_rule() &&
        blocks < (n == 5 ? FC_Q80_TILE_MIN_BLOCKS_N5 : FC_Q80_TILE_MIN_BLOCKS_N34))
        return -1;
    if (fc_ctx_enter(ctx) != 0)
        return 0;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride, &n
    };
    /* Column c is staged at c * FC_NX_CHUNK floats: a full chunk per column. */
    const int sh_floats = nc > 1 || cols >= FC_NX_CHUNK_HOST ? FC_NX_CHUNK_HOST : cols;
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)((rows + FC_Q80_TILE_ROWS_HOST - 1) / FC_Q80_TILE_ROWS_HOST),
        (unsigned int)((n + nc - 1) / nc), 1,
        256, 1, 1,
        (unsigned int)(nc * sh_floats * sizeof(float)),
        ctx->stream, args, NULL);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_q80 row tile (n=%d) launch failed: %d\n", n, err);
    return 0;
}

void fc_matvec_q80_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n)
{
    if (fc_matvec_q80_tile_launch(ctx, w, x, out, rows, cols, row_stride, n) == 0)
        return;
    fc_matvec_nx_launch(ctx, ctx ? ctx->kern_matvec_q80_nx : NULL,
                        ctx ? ctx->kern_matvec_q80_nx8 : NULL,
                        ctx ? ctx->kern_matvec_q80_nx2 : NULL,
                        ctx ? ctx->kern_matvec_q80 : NULL,
                        "matvec_q80_nx", w, x, out, rows, cols, row_stride,
                        n, FC_Q80_ROWS_PER_BLOCK_HOST, 0, 0);
}

/* The shipped parity tile for batch width n (bench_q80_ksplit on sm_37, FFN down):
 * r1c1, r1c2, r2c3, r2c4, r2c5, then r2c8 up to 8 columns and r4c8 past that. */
static void fc_q80p_tile_for(int n, int *r, int *nc)
{
    *r = n == 1 || n == 2 ? 1 : n <= 8 ? 2 : 4;
    *nc = n <= 5 ? n : 8;
}

int fc_matvec_q80_par_tile_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                               fc_buffer *out, int rows, int kh, int row_stride, int n,
                               int tile_r, int tile_nc)
{
    int t = -1;
    for (int i = 0; i < FC_Q80P_NKERN; i++)
        if (fc_q80p_tiles[i].r == tile_r && fc_q80p_tiles[i].nc == tile_nc)
            t = i;
    if (t < 0 || !fc_cuda_ctx_available(ctx) || !ctx->kern_q80p[t] || !w || !x || !out ||
        rows <= 0 || n <= 0 || kh <= 0 || (kh & 63))
        return -1;
    const int nc = fc_q80p_tiles[t].nc;
    const int rows_per_block = FC_Q80_ROWS_PER_BLOCK_HOST * fc_q80p_tiles[t].r;
    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = { &w->dptr, &x->dptr, &out->dptr, &rows, &kh, &row_stride, &n };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_q80p[t], (unsigned int)((rows + rows_per_block - 1) / rows_per_block),
        (unsigned int)((n + nc - 1) / nc), 1, 256, 1, 1,
        (unsigned int)(nc * fc_q80p_tiles[t].chunk * sizeof(float)), ctx->stream, args, NULL);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] matvec_q80_par r%dc%d (n=%d) launch failed: %d\n",
                tile_r, tile_nc, n, err);
        return -1;
    }
    return 0;
}

void fc_matvec_q80_par_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                              fc_buffer *out, int rows, int kh, int row_stride, int n)
{
    if (n <= 0)
        return;
    int r, nc;
    fc_q80p_tile_for(n, &r, &nc);
    if (fc_matvec_q80_par_tile_ctx(ctx, w, x, out, rows, kh, row_stride, n, r, nc) != 0)
        fprintf(stderr, "[fc_cuda] matvec_q80_par: r%dc%d unavailable (n=%d)\n", r, nc, n);
}

void fc_matvec_f32_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n)
{
    /* Block-per-row like fc_matvec_f32; shared = tile * threads. */
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_matvec_f32_nx ||
        !w || !x || !out || n <= 0)
        return;

    /* L10: one live column degenerates to fc_matvec_f32 exactly. */
    if (n == 1 && ctx->kern_matvec_f32 && fc_nx1_direct()) {
        fc_matvec_f32_ctx(ctx, w, x, out, rows, cols, row_stride);
        return;
    }

    int tile = fc_nx_tile_cols(n, ctx->kern_matvec_f32_nx8,
                               ctx->kern_matvec_f32_nx2);
    CUfunction kern = (tile == FC_NX8_COLS_HOST) ? ctx->kern_matvec_f32_nx8
                    : (tile == FC_NX2_COLS_HOST) ? ctx->kern_matvec_f32_nx2
                                                 : ctx->kern_matvec_f32_nx;
    int threads = 256;
    if (cols < threads)
        threads = cols;
    unsigned int shared_bytes =
        (unsigned int)(tile * threads * sizeof(float));
    unsigned int grid_y = (unsigned int)((n + tile - 1) / tile);

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride, &n
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)rows, grid_y, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_f32_nx launch failed: %d\n", err);
}

/* Batched F16 dense rows — fc_matvec_f32_nx_ctx's shape and tile selection,
 * with the 2-byte load. Per column bit-exact against fc_matvec_f16_ctx. */
void fc_matvec_f16_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_matvec_f16_nx ||
        !w || !x || !out || n <= 0)
        return;

    if (n == 1 && ctx->kern_matvec_f16 && fc_nx1_direct()) {
        fc_matvec_f16_ctx(ctx, w, x, out, rows, cols, row_stride);
        return;
    }

    int tile = fc_nx_tile_cols(n, ctx->kern_matvec_f16_nx8,
                               ctx->kern_matvec_f16_nx2);
    CUfunction kern = (tile == FC_NX8_COLS_HOST) ? ctx->kern_matvec_f16_nx8
                    : (tile == FC_NX2_COLS_HOST) ? ctx->kern_matvec_f16_nx2
                                                 : ctx->kern_matvec_f16_nx;
    int threads = 256;
    if (cols < threads)
        threads = cols;
    unsigned int shared_bytes =
        (unsigned int)(tile * threads * sizeof(float));
    unsigned int grid_y = (unsigned int)((n + tile - 1) / tile);

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride, &n
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)rows, grid_y, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_f16_nx launch failed: %d\n", err);
}

/* Batched BF16 dense rows — fc_matvec_f16_nx_ctx with the shift-widen.
 * Per column bit-exact against fc_matvec_bf16_ctx. */
void fc_matvec_bf16_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                           fc_buffer *out, int rows, int cols, int row_stride,
                           int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_matvec_bf16_nx ||
        !w || !x || !out || n <= 0)
        return;

    if (n == 1 && ctx->kern_matvec_bf16 && fc_nx1_direct()) {
        fc_matvec_bf16_ctx(ctx, w, x, out, rows, cols, row_stride);
        return;
    }

    int tile = fc_nx_tile_cols(n, ctx->kern_matvec_bf16_nx8,
                               ctx->kern_matvec_bf16_nx2);
    CUfunction kern = (tile == FC_NX8_COLS_HOST) ? ctx->kern_matvec_bf16_nx8
                    : (tile == FC_NX2_COLS_HOST) ? ctx->kern_matvec_bf16_nx2
                                                 : ctx->kern_matvec_bf16_nx;
    int threads = 256;
    if (cols < threads)
        threads = cols;
    unsigned int shared_bytes =
        (unsigned int)(tile * threads * sizeof(float));
    unsigned int grid_y = (unsigned int)((n + tile - 1) / tile);

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &w->dptr, &x->dptr, &out->dptr, &rows, &cols, &row_stride, &n
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)rows, grid_y, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] matvec_bf16_nx launch failed: %d\n", err);
}

void fc_rope_neox_partial_ctx(fc_cuda_ctx *ctx, fc_buffer *data,
                              int n_heads, int head_dim, int rope_dim,
                              int pos, float theta_base)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_rope_neox_partial || !data)
        return;

    /* rope.cu (lines 25-34, 49): one block per head, one thread per
     * rotation pair (half = rot/2). Block holds enough threads to cover
     * half; 256 is a safe upper bound for any head_dim we use here. No
     * shared memory. */
    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    int threads = (half < 256) ? half : 256;
    if (threads < 1)
        threads = 1;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &data->dptr, &n_heads, &head_dim, &rope_dim, &pos, &theta_base
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_rope_neox_partial,
        (unsigned int)n_heads, 1, 1,            /* one block per head */
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] rope_neox_partial launch failed: %d\n", err);
}

void fc_qk_rms_norm_ctx(fc_cuda_ctx *ctx, fc_buffer *data, fc_buffer *weight,
                        int n_heads, int head_dim, float eps)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_qk_rms_norm || !data)
        return;

    /* normalization.cu (lines 139-141): one block per head, 256 threads,
     * shared sum-of-squares reduction = blockDim.x * sizeof(float). */
    int threads = 256;
    if (head_dim < threads)
        threads = head_dim;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));
    CUdeviceptr w_ptr = weight ? weight->dptr : 0;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &data->dptr, &w_ptr, &n_heads, &head_dim, &eps };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_qk_rms_norm,
        (unsigned int)n_heads, 1, 1,            /* one block per head */
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] qk_rms_norm launch failed: %d\n", err);
}

/* The rank router (moe_routing.cu fc_moe_router_topk_rank) is bit-exact to
 * the block router for n_expert a power of two in [1, 256]; k as the block
 * kernel allows (FC_MOE_MAX_TOPK). Elsewhere the block kernels run. */
static int fc_moe_router_rank_ok(fc_cuda_ctx *ctx, int n_expert, int k)
{
    return ctx && ctx->kern_moe_router_topk_rank &&
           n_expert >= 1 && n_expert <= 256 &&
           (n_expert & (n_expert - 1)) == 0 && k >= 1 && k <= 32;
}

void fc_moe_router_topk_ctx(fc_cuda_ctx *ctx, fc_buffer *logits,
                            fc_buffer *top_idx, fc_buffer *top_wt,
                            int n_expert, int k, float expert_weights_scale)
{
    if (fc_moe_router_rank_ok(ctx, n_expert, k)) {
        fc_moe_router_topk_nx_ctx(ctx, logits, top_idx, top_wt, n_expert, k,
                                  expert_weights_scale, 1);
        return;
    }
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_router_topk ||
        !logits || !top_idx || !top_wt)
        return;

    /* moe_routing.cu (lines 62-63): 1 block, 256 threads, shared memory
     * = blockDim.x * sizeof(float) * 2 (max reduction | sum reduction). */
    int threads = 256;
    if (n_expert < threads)
        threads = n_expert;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float) * 2);

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &logits->dptr, &top_idx->dptr, &top_wt->dptr,
        &n_expert, &k, &expert_weights_scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_router_topk,
        1, 1, 1,                                /* single block */
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_router_topk launch failed: %d\n", err);
}

/* A round that found no logit above -inf (the row is all NaN / -inf, or has
 * fewer than k such values) reports index INT_MAX (topk_logits.cu
 * FC_TOPK_NO_WINNER). Those pairs are not a result: reject the whole call so
 * the caller takes its full-logits fallback or drops the draft, and say which
 * row, because NaN logits are an upstream bug worth seeing. Returns 0 when
 * every index is a position in [0, n), -1 otherwise. */
static int fc_topk_check_rows(const int *idx, int nrows, int k, int n)
{
    for (int r = 0; r < nrows; r++) {
        for (int i = 0; i < k; i++) {
            int id = idx[(size_t)r * k + i];
            if (id < 0 || id >= n) {
                fprintf(stderr, "[fc_cuda] topk_logits: row %d has %d of %d values "
                        "above -inf (NaN/-inf logits); no result\n", r, i, k);
                return -1;
            }
        }
    }
    return 0;
}

/* Device-side exact top-k over the rows of a [nslots][n] logits matrix
 * (topk_logits.cu). Fast path, three launches and no selection rounds:
 * chunk maxima, a per-row threshold (the k-th best chunk maximum) that
 * filters each row to a short candidate list, and a rank-select of that list.
 * A row whose list overflows FC_TOPK_CAND_CAP (mass ties at the threshold)
 * sends the call down the fallback: a tree of fc_topk_select launches (each
 * block emits its chunk's top-k by k rounds of block argmax; later levels
 * merge until one block per row remains). Both are compare-only: values are
 * bit-exact, ties go to the lowest index (the host's serial strict-'>'
 * argmax), and the logits are only read.
 *
 * Only the nslots*k (idx, val) pairs cross the host link (vs. the full
 * n-float rows): on a 0.4 GB/s small-BAR K80 host that turns the ~4 ms/token
 * 248k-vocab logits D2H into a few microseconds of copy. One D2H per path.
 *
 * Synchronous: out_idx/out_val (row u at [u*k, u*k+k), descending) are valid
 * on the host when it returns. Returns 0 on success, -1 on failure. */
#define FC_TOPK_SEL_E       16   /* topk_logits.cu FC_TOPK_SEL_E */
#define FC_TOPK_FAST_MAXBLK 1024 /* topk_logits.cu FC_TOPK_FAST_MAXBLK */
#define FC_TOPK_CAND_CAP    2048 /* topk_logits.cu FC_TOPK_CAND_CAP */
#define FC_TOPK_THREADS0    256
#define FC_TOPK_BLOCKS0     104  /* 13 SMX x 8 resident 256-thread blocks */
#define FC_TOPK_THREADS1    512

static int fc_topk_div_up(int a, int b) { return (a + b - 1) / b; }

/* Elements per thread for a level over m entries per row with `threads`
 * threads per block, at most FC_TOPK_SEL_E, aiming at `blocks` blocks. */
static int fc_topk_level_e(int m, int threads, int blocks)
{
    int e = fc_topk_div_up(m, threads * blocks);
    return e < 1 ? 1 : e > FC_TOPK_SEL_E ? FC_TOPK_SEL_E : e;
}

/* Copy the [nslots][k] winners (val then idx, contiguous at d_oval) plus
 * `extra` int32 words after them to the host in one stream-ordered D2H.
 * tmp holds 2*outn + extra words. */
static CUresult fc_topk_fetch(fc_cuda_ctx *ctx, CUdeviceptr d_oval,
                              size_t outn, size_t extra, int32_t *tmp)
{
    size_t bytes = (2 * outn + extra) * sizeof(int32_t);
    CUresult err;
    if (p_cuMemcpyDtoHAsync) {
        err = fcw_cuMemcpyDtoHAsync(tmp, d_oval, bytes, ctx->stream);
        if (err == CUDA_SUCCESS && p_cuStreamSynchronize)
            err = fcw_cuStreamSynchronize(ctx->stream);
    } else {
        err = p_cuStreamSynchronize ? fcw_cuStreamSynchronize(ctx->stream)
                                    : CUDA_SUCCESS;
        if (err == CUDA_SUCCESS)
            err = fcw_cuMemcpyDtoH(tmp, d_oval, bytes);
    }
    return err;
}

int fc_topk_logits_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *logits, int nslots,
                             int n, int k, int *out_idx, float *out_val)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_topk_select ||
        !logits || !out_idx || !out_val ||
        nslots <= 0 || n <= 0 || k <= 0 || k > FC_TOPK_MAX || k > n)
        return -1;

    /* Tree chunks: about one wave per row. */
    int e0 = fc_topk_level_e(n, FC_TOPK_THREADS0, FC_TOPK_BLOCKS0);
    int nblk0 = fc_topk_div_up(n, FC_TOPK_THREADS0 * e0);
    /* Fast-path chunks: the same, but at least k of them when n allows, so
     * the threshold (k-th best chunk maximum) exists. */
    int ef = e0, ek = n / (FC_TOPK_THREADS0 * k);
    if (ek >= 1 && ek < ef)
        ef = ek;
    int nblkf = fc_topk_div_up(n, FC_TOPK_THREADS0 * ef);
    int fast = ctx->kern_topk_blockmax && ctx->kern_topk_filter &&
               ctx->kern_topk_rank_select && nblkf <= FC_TOPK_FAST_MAXBLK;

    /* Scratch, int32/f32 words: tree levels 2 x [nslots][nblk0*k] (val|idx)
     * | fast maxima [nslots][nblkf] (val|idx) | candidates [nslots][CAP]
     * (val|idx) | count [nslots] | out [nslots][k] val | idx | status
     * [nslots]. Later tree levels only shrink. */
    size_t lvl = (size_t)nslots * nblk0 * k;
    size_t mxn = (size_t)nslots * nblkf;
    size_t cnd = (size_t)nslots * FC_TOPK_CAND_CAP;
    size_t outn = (size_t)nslots * k;
    size_t words = 4 * lvl + 2 * mxn + 2 * cnd + nslots + 2 * outn + nslots;
    size_t bytes = words * sizeof(int32_t);
    int32_t *tmp = malloc((2 * outn + nslots) * sizeof(int32_t));
    if (!tmp)
        return -1;

    if (fc_ctx_enter(ctx) != 0) {
        free(tmp);
        return -1;
    }
    if (!ctx->topk_scratch || ctx->topk_scratch_bytes < bytes) {
        if (ctx->topk_scratch && p_cuMemFree)
            p_cuMemFree(ctx->topk_scratch);
        ctx->topk_scratch = 0;
        ctx->topk_scratch_bytes = 0;
        CUresult aerr = p_cuMemAlloc(&ctx->topk_scratch, bytes);
        if (aerr != CUDA_SUCCESS) {
            fprintf(stderr, "[fc_cuda] topk scratch alloc failed: %d\n", aerr);
            fc_ctx_leave();
            free(tmp);
            return -1;
        }
        ctx->topk_scratch_bytes = bytes;
    }
    const size_t W = sizeof(int32_t);
    CUdeviceptr lv_val[2], lv_idx[2];
    lv_val[0] = ctx->topk_scratch;
    lv_idx[0] = lv_val[0] + lvl * W;
    lv_val[1] = lv_idx[0] + lvl * W;
    lv_idx[1] = lv_val[1] + lvl * W;
    CUdeviceptr d_bv = lv_idx[1] + lvl * W;
    CUdeviceptr d_bi = d_bv + mxn * W;
    CUdeviceptr d_cv = d_bi + mxn * W;
    CUdeviceptr d_ci = d_cv + cnd * W;
    CUdeviceptr d_cnt = d_ci + cnd * W;
    CUdeviceptr d_oval = d_cnt + (size_t)nslots * W;
    CUdeviceptr d_oidx = d_oval + outn * W;
    CUdeviceptr d_stat = d_oidx + outn * W;

    CUresult err = CUDA_SUCCESS;
    int done = 0;
    if (fast) {
        int threads = FC_TOPK_THREADS0;
        void *margs[] = { &logits->dptr, &n, &ef, &d_bv, &d_bi, &d_cnt };
        err = fcw_cuLaunchKernel(ctx->kern_topk_blockmax,
                               (unsigned int)nblkf, (unsigned int)nslots, 1,
                               (unsigned int)threads, 1, 1,
                               0, ctx->stream, margs, NULL);
        void *fargs[] = { &logits->dptr, &n, &ef, &d_bv, &d_bi, &k,
                          &d_cv, &d_ci, &d_cnt };
        if (err == CUDA_SUCCESS)
            err = fcw_cuLaunchKernel(ctx->kern_topk_filter,
                                   (unsigned int)nblkf, (unsigned int)nslots, 1,
                                   (unsigned int)threads, 1, 1,
                                   0, ctx->stream, fargs, NULL);
        void *rargs[] = { &d_cv, &d_ci, &d_cnt, &k, &d_oval, &d_oidx,
                          &d_stat };
        if (err == CUDA_SUCCESS)
            err = fcw_cuLaunchKernel(ctx->kern_topk_rank_select,
                                   1, (unsigned int)nslots, 1,
                                   (unsigned int)threads, 1, 1,
                                   0, ctx->stream, rargs, NULL);
        if (err == CUDA_SUCCESS)
            err = fc_topk_fetch(ctx, d_oval, outn, (size_t)nslots, tmp);
        done = err == CUDA_SUCCESS;
        for (int u = 0; done && u < nslots; u++)
            if (tmp[2 * outn + u] != 0)
                done = 0;               /* a row overflowed: take the tree */
    }

    CUdeviceptr in_val = logits->dptr, in_idx = 0;
    int m = n, e = e0, threads = FC_TOPK_THREADS0, nblk = nblk0;
    for (int level = 0; !done && err == CUDA_SUCCESS; level++) {
        int last = (nblk == 1);
        CUdeviceptr o_val = last ? d_oval : lv_val[level & 1];
        CUdeviceptr o_idx = last ? d_oidx : lv_idx[level & 1];
        void *args[] = { &in_val, &in_idx, &m, &e, &k, &o_val, &o_idx };
        err = fcw_cuLaunchKernel(ctx->kern_topk_select,
                               (unsigned int)nblk, (unsigned int)nslots, 1,
                               (unsigned int)threads, 1, 1,
                               0, ctx->stream, args, NULL);
        if (last) {
            if (err == CUDA_SUCCESS)
                err = fc_topk_fetch(ctx, d_oval, outn, 0, tmp);
            break;
        }
        in_val = o_val;
        in_idx = o_idx;
        m = nblk * k;
        threads = FC_TOPK_THREADS1;
        e = fc_topk_level_e(m, threads, 1);
        nblk = fc_topk_div_up(m, threads * e);
    }

    fc_ctx_leave();
    if (err == CUDA_SUCCESS) {
        memcpy(out_val, tmp, outn * sizeof(float));
        memcpy(out_idx, tmp + outn, outn * sizeof(int));
    }
    free(tmp);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] topk_logits failed: %d\n", err);
        return -1;
    }
    return fc_topk_check_rows(out_idx, nslots, k, n);
}

int fc_topk_logits_ctx(fc_cuda_ctx *ctx, fc_buffer *logits, int n, int k,
                       int *out_idx, float *out_val)
{
    return fc_topk_logits_slots_ctx(ctx, logits, 1, n, k, out_idx, out_val);
}

int fc_topk_logits(fc_buffer *logits, int n, int k,
                   int *out_idx, float *out_val)
{
    return fc_topk_logits_ctx(_fc_default_ctx, logits, n, k,
                              out_idx, out_val);
}

void fc_moe_weighted_add_ctx(fc_cuda_ctx *ctx, fc_buffer *expert_out,
                             fc_buffer *moe_out, float weight, int n_embd)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_weighted_add ||
        !expert_out || !moe_out)
        return;

    /* moe_routing.cu (lines 152-154): ceil(n_embd / blockDim.x) blocks,
     * 256 threads, no shared memory. */
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &expert_out->dptr, &moe_out->dptr, &weight, &n_embd };
    fc_dispatch_1d(ctx, ctx->kern_moe_weighted_add, args, n_embd);
    fc_ctx_leave();
}

void fc_moe_shared_gate_ctx(fc_cuda_ctx *ctx, fc_buffer *shexp_out,
                            float gate_logit, int n_embd)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_shared_gate ||
        !shexp_out)
        return;

    /* moe_routing.cu (lines 172-174): ceil(n_embd / blockDim.x) blocks,
     * 256 threads, no shared memory. */
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &shexp_out->dptr, &gate_logit, &n_embd };
    fc_dispatch_1d(ctx, ctx->kern_moe_shared_gate, args, n_embd);
    fc_ctx_leave();
}

/* FC_MOE_VEC_LOADS=1 routes the two 16-bit expert formats (F16, BF16) to
 * their uint2-load kernels instead of the scalar 16-bit-load ones. Read per
 * call, like FC_MOE_DOWN_WPP. It changes the lane->element partition and so
 * the fp association of the result — the two kernel sets are numerically
 * different (each bit-exact against its own replica), which is why this is a
 * switch and not an internal optimisation. */
int fc_moe_vec_loads_enabled(void)
{
    /* DEFAULT ON. Measured on die 7: gateup bf16 0.2400 -> 0.1840 ms and down
     * bf16 0.1299 -> 0.0986 ms, i.e. 1.30-1.33x and 139 -> 182 GB/s, which is
     * 76% of the die's 240 against 58% for the scalar map. A warp-wide 16-bit
     * load fills 64 B of a 128-B transaction, so the scalar map spends 8
     * half-empty transactions per 512-B super-block where uint2 spends 4 full
     * ones; SASS confirms LD.E.U16 x16 -> LD.E.64 x12.
     *
     * The association differs, so this is a switch rather than an internal
     * optimisation -- but it was turned on BEFORE any F16/BF16 model had ever
     * been served, which is the one moment the reproducibility cost is zero.
     * FC_MOE_VEC_LOADS=0 selects the scalar kernels; both sets are bit-exact
     * against their own replica. Q3_K/Q4_K are untouched either way. */
    const char *e = getenv("FC_MOE_VEC_LOADS");
    return (e && atoi(e) == 0) ? 0 : 1;
}

/* Is there a grouped gate+up kernel for this expert quant on this ctx?
 * Callers decide the GROUPED path on this — Q4_K's kernels are looked up
 * optionally (a pre-Q4_K fc_cuda_ptx.h leaves them NULL), and a caller that
 * skipped this probe would hand Q4_K bytes to a Q3_K decode. Under
 * FC_MOE_VEC_LOADS the probe answers for the VEC pair, so a PTX without them
 * reports unavailable here rather than failing closed at launch. */
int fc_moe_gateup_quant_available_ctx(fc_cuda_ctx *ctx, int gguf_quant)
{
    if (!fc_cuda_ctx_available(ctx))
        return 0;
    if (gguf_quant == 11)       /* GGUF_QUANT_Q3_K */
        return ctx->kern_moe_gateup_q3k && ctx->kern_moe_gateup_q3k_nx;
    if (gguf_quant == 12)       /* GGUF_QUANT_Q4_K */
        return ctx->kern_moe_gateup_q4k && ctx->kern_moe_gateup_q4k_nx;
    if (gguf_quant == 1) {      /* GGUF_QUANT_F16 */
        if (fc_moe_vec_loads_enabled())
            return ctx->kern_moe_gateup_f16_vec &&
                   ctx->kern_moe_gateup_f16_vec_nx;
        return ctx->kern_moe_gateup_f16 && ctx->kern_moe_gateup_f16_nx;
    }
    if (gguf_quant == 30) {     /* GGUF_QUANT_BF16 */
        if (fc_moe_vec_loads_enabled())
            return ctx->kern_moe_gateup_bf16_vec &&
                   ctx->kern_moe_gateup_bf16_vec_nx;
        return ctx->kern_moe_gateup_bf16 && ctx->kern_moe_gateup_bf16_nx;
    }
    if (gguf_quant == 8)        /* GGUF_QUANT_Q8_0 */
        return ctx->kern_moe_gateup_q80 && ctx->kern_moe_gateup_q80_nx;
    return 0;
}

/* Is there a grouped down kernel for this expert quant on this ctx? Q6_K is an
 * optional symbol like the Q4_K gate+up pair, so the caller must probe. */
int fc_moe_down_quant_available_ctx(fc_cuda_ctx *ctx, int gguf_quant)
{
    if (!fc_cuda_ctx_available(ctx))
        return 0;
    if (gguf_quant == 11)       /* GGUF_QUANT_Q3_K */
        return ctx->kern_moe_down_q3k && ctx->kern_moe_down_q3k_nx;
    if (gguf_quant == 12)       /* GGUF_QUANT_Q4_K */
        return ctx->kern_moe_down_q4k && ctx->kern_moe_down_q4k_nx;
    if (gguf_quant == 14)       /* GGUF_QUANT_Q6_K */
        return ctx->kern_moe_down_q6k && ctx->kern_moe_down_q6k_nx;
    if (gguf_quant == 1) {      /* GGUF_QUANT_F16 */
        if (fc_moe_vec_loads_enabled())
            return ctx->kern_moe_down_f16_vec && ctx->kern_moe_down_f16_vec_nx;
        return ctx->kern_moe_down_f16 && ctx->kern_moe_down_f16_nx;
    }
    if (gguf_quant == 30) {     /* GGUF_QUANT_BF16 */
        if (fc_moe_vec_loads_enabled())
            return ctx->kern_moe_down_bf16_vec &&
                   ctx->kern_moe_down_bf16_vec_nx;
        return ctx->kern_moe_down_bf16 && ctx->kern_moe_down_bf16_nx;
    }
    if (gguf_quant == 8)        /* GGUF_QUANT_Q8_0 */
        return ctx->kern_moe_down_q80 && ctx->kern_moe_down_q80_nx;
    return 0;
}

/* Grouped gate+up SwiGLU, dispatched on the gate/up tensor's gguf type the
 * same way fc_moe_down_accum_ctx dispatches the down tensor's: 11 = Q3_K,
 * 12 = Q4_K, 1 = F16, 30 = BF16, 8 = Q8_0. FAILS CLOSED — anything else (or a missing
 * kernel) is a named error and NO launch, because the alternative is decoding
 * one quant's bytes with another's kernel and producing plausible garbage.
 * Returns 0 on launch, -1 otherwise. */
int fc_moe_gateup_ctx(fc_cuda_ctx *ctx, int gguf_quant, fc_buffer *gate_all,
                      fc_buffer *up_all, fc_buffer *x, fc_buffer *top_idx,
                      fc_buffer *hid, int k, int ff, int cols,
                      int row_stride)
{
    if (!fc_cuda_ctx_available(ctx) ||
        !gate_all || !up_all || !x || !top_idx || !hid)
        return -1;

    int vec = fc_moe_vec_loads_enabled();
    CUfunction kern = NULL;
    if (gguf_quant == 11)       /* GGUF_QUANT_Q3_K */
        kern = ctx->kern_moe_gateup_q3k;
    else if (gguf_quant == 12)  /* GGUF_QUANT_Q4_K */
        kern = ctx->kern_moe_gateup_q4k;
    else if (gguf_quant == 1)   /* GGUF_QUANT_F16 */
        kern = vec ? ctx->kern_moe_gateup_f16_vec : ctx->kern_moe_gateup_f16;
    else if (gguf_quant == 30)  /* GGUF_QUANT_BF16 */
        kern = vec ? ctx->kern_moe_gateup_bf16_vec : ctx->kern_moe_gateup_bf16;
    else if (gguf_quant == 8)   /* GGUF_QUANT_Q8_0 (expert layout) */
        kern = ctx->kern_moe_gateup_q80;
    if (!kern) {
        fprintf(stderr, "[fc_cuda] moe_gateup: unsupported quant %d\n",
                gguf_quant);
        return -1;
    }

    /* moe_expert_group.cu fc_moe_gateup_q3k/q4k: WARP-PER-ROW — grid
     * (ceil(ff/8), k), 8 warps (256 threads) per block, one warp per
     * (expert-FF row, slot), __shfl_down reduction, staged-x shared =
     * min(cols, 2048) floats. Q3_K/Q4_K repack to 152 B per 256-weight
     * super-block and F16 is a flat 512 B, but the launch shape is a function
     * of ff/k/cols and never of row_stride, so it is type-independent. */
    int threads = 256;
    int sh_floats = cols < 2048 ? cols : 2048;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &gate_all->dptr, &up_all->dptr, &x->dptr, &top_idx->dptr, &hid->dptr,
        &ff, &cols, &row_stride
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)((ff + 7) / 8), (unsigned int)k, 1,
        (unsigned int)threads, 1, 1,
        (unsigned int)(sh_floats * sizeof(float)), ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] moe_gateup launch failed (quant %d): %d\n",
                gguf_quant, err);
        return -1;
    }
    return 0;
}

void fc_moe_gateup_q3k_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                           fc_buffer *up_all, fc_buffer *x, fc_buffer *top_idx,
                           fc_buffer *hid, int k, int ff, int cols,
                           int row_stride)
{
    (void)fc_moe_gateup_ctx(ctx, 11, gate_all, up_all, x, top_idx, hid, k, ff,
                            cols, row_stride);
}

void fc_moe_down_accum_ctx(fc_cuda_ctx *ctx, int gguf_quant, fc_buffer *down_all,
                           fc_buffer *hid, fc_buffer *top_idx, fc_buffer *top_wt,
                           fc_buffer *out, int k, int rows, int ff,
                           int row_stride)
{
    if (!fc_cuda_ctx_available(ctx) || !down_all || !hid || !top_idx ||
        !top_wt || !out)
        return;

    /* Type dispatch: the merged ffn_down_exps tensor is mixed WITHIN one file
     * — Q3_K/Q4_K by layer in the Q3_K_M mini, Q4_K/Q6_K by layer in a Q4_K_M.
     * An F16 or BF16 file is uniform (one type everywhere): at the top of the
     * ladder there are no per-layer quant choices left to make. */
    int vec = fc_moe_vec_loads_enabled();
    CUfunction kern = NULL;
    if (gguf_quant == 11)       /* GGUF_QUANT_Q3_K */
        kern = ctx->kern_moe_down_q3k;
    else if (gguf_quant == 12)  /* GGUF_QUANT_Q4_K */
        kern = ctx->kern_moe_down_q4k;
    else if (gguf_quant == 14)  /* GGUF_QUANT_Q6_K */
        kern = ctx->kern_moe_down_q6k;
    else if (gguf_quant == 1)   /* GGUF_QUANT_F16 */
        kern = vec ? ctx->kern_moe_down_f16_vec : ctx->kern_moe_down_f16;
    else if (gguf_quant == 30)  /* GGUF_QUANT_BF16 */
        kern = vec ? ctx->kern_moe_down_bf16_vec : ctx->kern_moe_down_bf16;
    else if (gguf_quant == 8)   /* GGUF_QUANT_Q8_0 (expert layout) */
        kern = ctx->kern_moe_down_q80;
    if (!kern) {
        fprintf(stderr, "[fc_cuda] moe_down_accum: unsupported quant %d\n",
                gguf_quant);
        return;
    }

    /* moe_expert_group.cu fc_moe_down_*: COOPERATIVE MULTI-WARP-PER-ROW-PAIR
     * — the block's warps run in groups of wpp cooperating warps; each
     * group covers one output row pair, splitting the k slots across its
     * warps (hid via __ldg, no staging shared), then a per-warp __shfl_down
     * tree + tiny static-shared cross-warp sum. Tuned defaults in fc_cuda.h
     * (128 threads, wpp=2). Grid = ceil(rows / rows-per-block),
     * rows-per-block = 2 * n_warps/wpp. */
    static int wpp_cached = 0, thr_cached = 0;  /* env read once per process */
    if (!wpp_cached) {
        wpp_cached = FC_MOE_DOWN_WPP_DEFAULT;
        const char *wpp_env = getenv("FC_MOE_DOWN_WPP");
        if (wpp_env) {
            int v = atoi(wpp_env);
            if (v == 1 || v == 2 || v == 4 || v == 8)
                wpp_cached = v;
        }
        thr_cached = FC_MOE_DOWN_THREADS_DEFAULT;
        const char *thr_env = getenv("FC_MOE_DOWN_THREADS");
        if (thr_env) {
            int v = atoi(thr_env);
            /* kernel's static cross-warp scratch covers <= 16 warps; the
             * warp count must be a multiple of wpp */
            if ((v == 64 || v == 128 || v == 256 || v == 512) &&
                (v / 32) % wpp_cached == 0)
                thr_cached = v;
        }
    }
    int wpp = wpp_cached;
    int threads = thr_cached;
    int rows_per_block = 2 * (threads / 32) / wpp;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &down_all->dptr, &hid->dptr, &top_idx->dptr, &top_wt->dptr, &out->dptr,
        &k, &rows, &ff, &row_stride, &wpp
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)((rows + rows_per_block - 1) / rows_per_block), 1, 1,
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_down_accum launch failed: %d\n", err);
}

void fc_moe_silu_mul_ctx(fc_cuda_ctx *ctx, fc_buffer *a, fc_buffer *b,
                         fc_buffer *out, int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_silu_mul ||
        !a || !b || !out)
        return;

    /* moe_expert_group.cu fc_moe_silu_mul: ceil(n / 256) blocks, 256
     * threads, no shared memory. In-place safe. */
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &a->dptr, &b->dptr, &out->dptr, &n };
    fc_dispatch_1d(ctx, ctx->kern_moe_silu_mul, args, n);
    fc_ctx_leave();
}

void fc_moe_shexp_dot_gate_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_w,
                               fc_buffer *x, fc_buffer *shexp, int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_shexp_dot_gate ||
        !gate_w || !x || !shexp)
        return;

    /* moe_expert_group.cu fc_moe_shexp_dot_gate: 1 block, 256 threads,
     * shared = 256 floats (dot reduction). */
    int threads = 256;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &gate_w->dptr, &x->dptr, &shexp->dptr, &n };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_shexp_dot_gate,
        1, 1, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_shexp_dot_gate launch failed: %d\n", err);
}

/* ================================================================
 * Fused per-layer launches (fused_layer.cu)
 * ================================================================ */

/* Device-side segment descriptor — must match fc_fused_seg_dev in
 * fused_layer.cu (6 x u64 then 8 x s32, 80 bytes). */
typedef struct {
    unsigned long long w, w2, out, aux0, aux1, aux2;
    int rows, cols, row_stride;
    int kind, qtype, ep;
    int block_base;
    int pad;
} fc_fused_seg_dev;

fc_buffer *fc_fused_build_ctx(fc_cuda_ctx *ctx, const fc_fused_seg *segs,
                              int nseg, int *total_blocks, int *shared_bytes)
{
    fc_fused_seg_dev dev[FC_FUSED_MAX_SEG];
    int blocks = 0, sh = 0;

    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_fused_multi || !segs ||
        !total_blocks || !shared_bytes || nseg < 1 || nseg > FC_FUSED_MAX_SEG)
        return NULL;

    for (int i = 0; i < nseg; i++) {
        const fc_fused_seg *s = &segs[i];
        int shb;
        if (!s->w || !s->out || s->rows <= 0 || s->cols <= 0 ||
            s->row_stride <= 0)
            return NULL;
        if (s->kind == 0 || s->kind == 2) {
            /* Quant warp-per-row: repacked types only, cols aligned to the
             * block granularity of the type. */
            if (s->qtype == 8) {
                if (s->cols % 32)
                    return NULL;
            } else if (s->qtype == 11 || s->qtype == 12 || s->qtype == 13 ||
                       s->qtype == 14) {
                if (s->cols % 256)
                    return NULL;
            } else {
                return NULL;
            }
            if (s->kind == 2 && !s->w2)
                return NULL;
            shb = (s->cols < 2048 ? s->cols : 2048) * (int)sizeof(float);
            dev[i].block_base = blocks;
            if (s->qtype == 8)              /* 8 lanes per row: 32/block */
                blocks += (s->rows + 31) / 32;
            else                            /* WARP-PER-ROW-PAIR: 16/block */
                blocks += (s->rows + 15) / 16;
        } else if (s->kind == 1) {
            /* F32 block-per-row: 256-thread reduction shape requires
             * cols >= 256 for exactness vs fc_matvec_f32. */
            if (s->cols < 256)
                return NULL;
            shb = 256 * (int)sizeof(float);
            dev[i].block_base = blocks;
            blocks += s->rows;
        } else {
            return NULL;
        }
        if (shb > sh)
            sh = shb;
        /* Epilogue operand presence. */
        if ((s->ep == 2 && (!s->aux0 || !s->aux1)) ||
            (s->ep == 3 && !s->aux1) ||
            (s->ep == 4 && (!s->aux0 || !s->aux1 || !s->aux2)) ||
            s->ep < 0 || s->ep > 4)
            return NULL;
        dev[i].w    = (unsigned long long)s->w->dptr;
        dev[i].w2   = s->w2 ? (unsigned long long)s->w2->dptr : 0;
        dev[i].out  = (unsigned long long)s->out->dptr;
        dev[i].aux0 = s->aux0 ? (unsigned long long)s->aux0->dptr : 0;
        dev[i].aux1 = s->aux1 ? (unsigned long long)s->aux1->dptr : 0;
        dev[i].aux2 = s->aux2 ? (unsigned long long)s->aux2->dptr : 0;
        dev[i].rows = s->rows;
        dev[i].cols = s->cols;
        dev[i].row_stride = s->row_stride;
        dev[i].kind = s->kind;
        dev[i].qtype = s->qtype;
        dev[i].ep = s->ep;
        dev[i].pad = 0;
    }

    *total_blocks = blocks;
    *shared_bytes = sh;
    return fc_buffer_from_ptr_ctx(ctx, dev,
                                  (size_t)nseg * sizeof(fc_fused_seg_dev));
}

void fc_fused_multi_ctx(fc_cuda_ctx *ctx, fc_buffer *desc, int nseg,
                        int total_blocks, int shared_bytes, fc_buffer *x)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_fused_multi || !desc || !x ||
        total_blocks <= 0)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &desc->dptr, &nseg, &x->dptr };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_fused_multi,
        (unsigned int)total_blocks, 1, 1,
        256, 1, 1,
        (unsigned int)shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] fused_multi launch failed: %d\n", err);
}

int fc_attn_qknorm_rope_ctx(fc_cuda_ctx *ctx, fc_buffer *q, fc_buffer *k,
                            fc_buffer *q_norm_w, fc_buffer *k_norm_w,
                            int n_head, int n_head_kv, int head_dim, float eps,
                            int rope_dim, int pos, float theta_base)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_qknorm_rope ||
        !q || !k)
        return -1;

    /* fused_layer.cu fc_attn_qknorm_rope: one block per (Q then K) head,
     * threads = min(256, head_dim) — the fc_qk_rms_norm shape (the rope half
     * is always <= head_dim/2 < threads). Shared = threads floats. */
    int threads = head_dim < 256 ? head_dim : 256;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));
    CUdeviceptr qw_ptr = q_norm_w ? q_norm_w->dptr : 0;
    CUdeviceptr kw_ptr = k_norm_w ? k_norm_w->dptr : 0;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &q->dptr, &k->dptr, &qw_ptr, &kw_ptr, &n_head, &n_head_kv, &head_dim,
        &eps, &rope_dim, &pos, &theta_base
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_qknorm_rope,
        (unsigned int)(n_head + n_head_kv), 1, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_qknorm_rope launch failed: %d\n", err);
        return -1;
    }
    return 0;
}

/* ================================================================
 * SSM Gated-Delta-Net kernels (per-context)
 *
 * Launch configs are transcribed from ssm_gated_delta.cu; the cited
 * line ranges there pin each kernel's grid/block layout. All three
 * mirror static helpers in Pantheon/src/models/qwen35moe_ssm.c.
 * ================================================================ */

void fc_ssm_recurrence_ctx(fc_cuda_ctx *ctx, fc_buffer *s, fc_buffer *q,
                           fc_buffer *k, fc_buffer *v, fc_buffer *g,
                           fc_buffer *beta, fc_buffer *y,
                           int n_v_heads, int state_size, float scale)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_recurrence ||
        !s || !q || !k || !v || !g || !beta || !y)
        return;

    /* ssm_gated_delta.cu (lines ~58-66): grid = (state_size, n_v_heads),
     * one block per (row j, head h); block = state_size threads (= S, one
     * per state coord i); shared reduction scratch = state_size*sizeof(float). */
    unsigned int shared_bytes = (unsigned int)((size_t)state_size * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &s->dptr, &q->dptr, &k->dptr, &v->dptr, &g->dptr, &beta->dptr,
        &y->dptr, &state_size, &scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_ssm_recurrence,
        (unsigned int)state_size, (unsigned int)n_v_heads, 1,   /* grid (j, h) */
        (unsigned int)state_size, 1, 1,                          /* block = S */
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] ssm_recurrence launch failed: %d\n", err);
}

void fc_ssm_conv1d_silu_ctx(fc_cuda_ctx *ctx, fc_buffer *qkv, fc_buffer *weight,
                            fc_buffer *conv_state, fc_buffer *out,
                            int conv_dim, int conv_kernel)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_conv1d_silu ||
        !qkv || !weight || !conv_state || !out)
        return;

    /* ssm_gated_delta.cu (lines ~138-141): one thread per channel,
     * ceil(conv_dim/256) blocks of 256 threads, no shared memory. */
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &qkv->dptr, &weight->dptr, &conv_state->dptr, &out->dptr,
        &conv_dim, &conv_kernel
    };
    fc_dispatch_1d(ctx, ctx->kern_ssm_conv1d_silu, args, conv_dim);
    fc_ctx_leave();
}

void fc_ssm_gated_norm_ctx(fc_cuda_ctx *ctx, fc_buffer *y, fc_buffer *weight,
                           fc_buffer *z, fc_buffer *gated,
                           int n_v_heads, int state_size, float eps)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_gated_norm ||
        !y || !weight || !z || !gated)
        return;

    /* ssm_gated_delta.cu (lines ~206-209): one block per V-head,
     * 256 threads (or S if smaller), shared sum-of-squares reduction
     * = blockDim.x * sizeof(float). */
    int threads = 256;
    if (state_size < threads)
        threads = state_size;
    unsigned int shared_bytes = (unsigned int)((size_t)threads * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &y->dptr, &weight->dptr, &z->dptr, &gated->dptr, &state_size, &eps
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_ssm_gated_norm,
        (unsigned int)n_v_heads, 1, 1,          /* one block per head */
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] ssm_gated_norm launch failed: %d\n", err);
}

/* --- SSM glue launchers (replace host round-trips in the qwen35moe forward) --- */

void fc_sigmoid_ctx(fc_cuda_ctx *ctx, fc_buffer *x, int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_sigmoid || !x)
        return;
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &x->dptr, &n };
    fc_dispatch_1d(ctx, ctx->kern_sigmoid, args, n);
    fc_ctx_leave();
}

void fc_ssm_decay_gate_ctx(fc_cuda_ctx *ctx, fc_buffer *alpha, fc_buffer *ssm_dt,
                           fc_buffer *ssm_a, fc_buffer *g, int n)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_decay_gate ||
        !alpha || !ssm_dt || !ssm_a || !g)
        return;
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &alpha->dptr, &ssm_dt->dptr, &ssm_a->dptr, &g->dptr, &n };
    fc_dispatch_1d(ctx, ctx->kern_ssm_decay_gate, args, n);
    fc_ctx_leave();
}

/* q/k L2-norm + GQA interleaved-repeat on device; v (conv_out[2*key_dim..]) is a
 * straight async DtoD copy to vg. grid = n_k_heads blocks, block = state_size,
 * shared = 2*state_size floats (q|k sum-of-squares). */
void fc_ssm_qknorm_gqa_ctx(fc_cuda_ctx *ctx, fc_buffer *conv_out, fc_buffer *qg,
                           fc_buffer *kg, fc_buffer *vg, int n_k_heads,
                           int kv_groups, int state_size, int key_dim,
                           int val_dim, float eps)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_qknorm_gqa ||
        !conv_out || !qg || !kg || !vg)
        return;

    unsigned int shared_bytes = (unsigned int)(2 * (size_t)state_size * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &conv_out->dptr, &qg->dptr, &kg->dptr, &n_k_heads, &kv_groups,
        &state_size, &key_dim, &eps
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_ssm_qknorm_gqa,
        (unsigned int)n_k_heads, 1, 1,          /* grid: one block per K-head */
        (unsigned int)state_size, 1, 1,         /* block: S threads */
        shared_bytes, ctx->stream, args, NULL
    );
    /* v passthrough: async DtoD (stream-ordered after conv1d + this kernel). */
    if (err == CUDA_SUCCESS && p_cuMemcpyDtoDAsync)
        err = fcw_cuMemcpyDtoDAsync(vg->dptr,
                                  conv_out->dptr + (size_t)2 * key_dim * sizeof(float),
                                  (size_t)val_dim * sizeof(float), ctx->stream);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] ssm_qknorm_gqa launch failed: %d\n", err);
}

/* ================================================================
 * Batched (prefill) launchers — n positions/tokens per launch. Kernel
 * bodies mirror their single-position counterparts per position (bit-
 * exact); see the .cu headers.
 * ================================================================ */

void fc_rms_norm_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *x, fc_buffer *weight,
                        fc_buffer *out, int n, float eps, int rows)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_rms_norm_nx || !x || !out ||
        rows <= 0)
        return;

    /* Same blockDim choice as the single-row fc_rms_norm launch (so each
     * row's reduction tree is identical); one block per row. */
    int threads = 256;
    if (n < threads)
        threads = n;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));
    CUdeviceptr w_ptr = weight ? weight->dptr : 0;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &x->dptr, &w_ptr, &out->dptr, &n, &eps, &rows };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_rms_norm_nx,
        (unsigned int)rows, 1, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] rms_norm_nx launch failed: %d\n", err);
}

void fc_rope_neox_partial_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *data,
                                 int n_heads, int head_dim, int rope_dim,
                                 int pos0, float theta_base, int npos)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_rope_nx || !data ||
        npos <= 0)
        return;

    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    int threads = (half < 256) ? half : 256;
    if (threads < 1)
        threads = 1;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &data->dptr, &n_heads, &head_dim, &rope_dim, &pos0, &theta_base
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_rope_nx,
        (unsigned int)n_heads, (unsigned int)npos, 1,
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] rope_neox_partial_nx launch failed: %d\n",
                err);
}

/* Reference block router (fc_moe_router_topk_nx, blockDim = n_expert): what
 * every launch used before the rank router; kept for the n_expert / k the
 * warp kernel does not cover and for the bit-exact tests. */
void fc_moe_router_topk_nx_block_ctx(fc_cuda_ctx *ctx, fc_buffer *logits,
                                     fc_buffer *top_idx, fc_buffer *top_wt,
                                     int n_expert, int k,
                                     float expert_weights_scale, int ntok)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_router_topk_nx ||
        !logits || !top_idx || !top_wt || ntok <= 0)
        return;

    int threads = 256;
    if (n_expert < threads)
        threads = n_expert;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float) * 2);

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &logits->dptr, &top_idx->dptr, &top_wt->dptr,
        &n_expert, &k, &expert_weights_scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_router_topk_nx,
        (unsigned int)ntok, 1, 1,               /* one block per token */
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_router_topk_nx launch failed: %d\n",
                err);
}

void fc_moe_router_topk_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *logits,
                               fc_buffer *top_idx, fc_buffer *top_wt,
                               int n_expert, int k,
                               float expert_weights_scale, int ntok)
{
    if (!fc_moe_router_rank_ok(ctx, n_expert, k)) {
        fc_moe_router_topk_nx_block_ctx(ctx, logits, top_idx, top_wt,
                                        n_expert, k, expert_weights_scale,
                                        ntok);
        return;
    }
    if (!fc_cuda_ctx_available(ctx) || !logits || !top_idx || !top_wt ||
        ntok <= 0)
        return;
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &logits->dptr, &top_idx->dptr, &top_wt->dptr,
        &n_expert, &k, &expert_weights_scale
    };
    unsigned int threads = n_expert < 32 ? 32u : (unsigned int)n_expert;
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_router_topk_rank,
        (unsigned int)ntok, 1, 1,               /* one block per token */
        threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_router_topk_rank launch failed: %d\n",
                err);
}

/* Largest token count at which the batched Q8_0 MoE kernels use __ldg weight
 * loads; above it they take the plain-load _nxp forms. Read per call, like
 * FC_MOE_DOWN_WPP: FC_MOE_Q80_LDG_MAX_TOK=<n> overrides (0 = always plain). */
int fc_moe_q80_ldg_max_tok(void)
{
    /* Qwen3.6 on K80 dies: __ldg made 2- and 4-slot decode 8% and
     * 10.6% faster but the 512-token prefill chunks 22% slower. Die 8, 256
     * experts, random routing (test_moe_q80_bw --sweep), gate+up us per call
     * __ldg/plain: 8 tok 898/1278, 16 tok 4642/2528, 512 tok 159489/80163;
     * down: 8 tok 562/668, 16 tok 1505/1359. */
    const char *e = getenv("FC_MOE_Q80_LDG_MAX_TOK");
    if (e && *e)
        return atoi(e);
    return FC_MOE_Q80_LDG_MAX_TOK_DEFAULT;
}

/* Batched grouped gate+up, dispatched on the gate/up gguf type. Same
 * fail-closed contract as fc_moe_gateup_ctx; returns 0 on launch, -1 else. */
int fc_moe_gateup_nx_ctx(fc_cuda_ctx *ctx, int gguf_quant, fc_buffer *gate_all,
                         fc_buffer *up_all, fc_buffer *x, fc_buffer *top_idx,
                         fc_buffer *hid, int k, int ff, int cols,
                         int row_stride, int ntok)
{
    if (!fc_cuda_ctx_available(ctx) ||
        !gate_all || !up_all || !x || !top_idx || !hid || ntok <= 0)
        return -1;

    int vec = fc_moe_vec_loads_enabled();
    CUfunction kern = NULL;
    if (gguf_quant == 11)       /* GGUF_QUANT_Q3_K */
        kern = ctx->kern_moe_gateup_q3k_nx;
    else if (gguf_quant == 12)  /* GGUF_QUANT_Q4_K */
        kern = ctx->kern_moe_gateup_q4k_nx;
    else if (gguf_quant == 1)   /* GGUF_QUANT_F16 */
        kern = vec ? ctx->kern_moe_gateup_f16_vec_nx
                   : ctx->kern_moe_gateup_f16_nx;
    else if (gguf_quant == 30)  /* GGUF_QUANT_BF16 */
        kern = vec ? ctx->kern_moe_gateup_bf16_vec_nx
                   : ctx->kern_moe_gateup_bf16_nx;
    else if (gguf_quant == 8)   /* GGUF_QUANT_Q8_0 (expert layout) */
        kern = (ntok > fc_moe_q80_ldg_max_tok() && ctx->kern_moe_gateup_q80_nxp)
             ? ctx->kern_moe_gateup_q80_nxp : ctx->kern_moe_gateup_q80_nx;
    if (!kern) {
        fprintf(stderr, "[fc_cuda] moe_gateup_nx: unsupported quant %d\n",
                gguf_quant);
        return -1;
    }

    int threads = 256;
    int sh_floats = cols < 2048 ? cols : 2048;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &gate_all->dptr, &up_all->dptr, &x->dptr, &top_idx->dptr, &hid->dptr,
        &k, &ff, &cols, &row_stride
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)((ff + 7) / 8), (unsigned int)k, (unsigned int)ntok,
        (unsigned int)threads, 1, 1,
        (unsigned int)(sh_floats * sizeof(float)), ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr,
                "[fc_cuda] moe_gateup_nx launch failed (quant %d): %d\n",
                gguf_quant, err);
        return -1;
    }
    return 0;
}

void fc_moe_gateup_q3k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                              fc_buffer *up_all, fc_buffer *x,
                              fc_buffer *top_idx, fc_buffer *hid, int k,
                              int ff, int cols, int row_stride, int ntok)
{
    (void)fc_moe_gateup_nx_ctx(ctx, 11, gate_all, up_all, x, top_idx, hid, k,
                               ff, cols, row_stride, ntok);
}

/* Both dedup kernels present? They live at the end of the MoE expert-group
 * module's symbol list, so an fc_cuda_ptx.h regenerated before they existed
 * leaves exactly these two NULL while the rest of the grouped path still
 * loads. Callers must probe this and fall back to the _nx path. */
int fc_moe_dedup_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_moe_dedup_build &&
           ctx->kern_moe_gateup_q3k_dedup;
}

/* Build per-expert request lists (cross-user dedup). top_idx is [ntok*k]
 * int32; ecount [n_expert] and ereq [n_expert*maxr] are device scratch. */
void fc_moe_dedup_build_ctx(fc_cuda_ctx *ctx, fc_buffer *top_idx,
                            fc_buffer *ecount, fc_buffer *ereq, int n_expert,
                            int k, int ntok, int maxr)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_dedup_build ||
        !top_idx || !ecount || !ereq || ntok <= 0)
        return;

    int threads = 256;
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &top_idx->dptr, &ecount->dptr, &ereq->dptr,
        &n_expert, &k, &ntok, &maxr
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_dedup_build,
        1, 1, 1,                                /* single block */
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_dedup_build launch failed: %d\n", err);
}

/* Grouped gate+up over the union of routed experts: reads each unique
 * expert's rows once and applies them to every token that routed to it.
 * Bit-exact with fc_moe_gateup_q3k_nx (see the kernel). grid.y spans all
 * n_expert (experts nothing routed to exit on their first load); grid.z tiles
 * the per-expert request list by FC_MOE_DEDUP_RT, sized off maxr because the
 * real per-expert counts only exist on the device. */
/* The Q8_0 grouped gate+up kernel loaded (plus both list builders it reads)? */
int fc_moe_dedup_q80_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_moe_dedup_build &&
           ctx->kern_moe_dedup_build_wl && ctx->kern_moe_gateup_q80_dedup;
}

/* Q8_0 expert-grouped gate+up over the compacted work list
 * (fc_moe_dedup_build_ctx then fc_moe_dedup_build_wl_ctx, same stream).
 * Bit-exact with fc_moe_gateup_nx_ctx (GGUF Q8_0) per request. wmax is
 * fc_moe_dedup_wmax(n_expert, k, ntok, maxr). Returns 0 on launch. */
int fc_moe_gateup_q80_dedup_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                                fc_buffer *up_all, fc_buffer *x,
                                fc_buffer *ereq, fc_buffer *wlist,
                                fc_buffer *hid, int k, int ff, int cols,
                                int row_stride, int wmax, int maxr)
{
    if (!fc_moe_dedup_q80_available_ctx(ctx) || !gate_all || !up_all || !x ||
        !ereq || !wlist || !hid || wmax <= 0 || maxr <= 0 || cols % 256 != 0)
        return -1;
    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &gate_all->dptr, &up_all->dptr, &x->dptr, &ereq->dptr, &wlist->dptr,
        &hid->dptr, &k, &ff, &cols, &row_stride, &maxr
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_gateup_q80_dedup,
        (unsigned int)((ff + 7) / 8), (unsigned int)wmax, 1,
        256, 1, 1, 0, ctx->stream, args, NULL);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] moe_gateup_q80_dedup launch failed: %d\n", err);
        return -1;
    }
    return 0;
}

void fc_moe_gateup_q3k_dedup_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                                 fc_buffer *up_all, fc_buffer *x,
                                 fc_buffer *ecount, fc_buffer *ereq,
                                 fc_buffer *hid, int k, int ff, int cols,
                                 int row_stride, int n_expert, int maxr)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_gateup_q3k_dedup ||
        !gate_all || !up_all || !x || !ecount || !ereq || !hid ||
        n_expert <= 0 || maxr <= 0)
        return;

    int threads = 256;
    unsigned int zt = (unsigned int)((maxr + FC_MOE_DEDUP_RT - 1) /
                                     FC_MOE_DEDUP_RT);

    /* FC_MOE_DEDUP_SHX=1 selects the shared-x variant: same grid, same args,
     * bit-exact, measured +20% on the kernel. Default off. It falls back
     * silently when the symbol is missing — it is the LAST name in the
     * module's list, so an fc_cuda_ptx.h predating it leaves exactly this
     * CUfunction NULL while every other kernel still loads. */
    CUfunction kern = ctx->kern_moe_gateup_q3k_dedup;
    static int shx_env = -1;
    if (shx_env < 0) {
        const char *e = getenv("FC_MOE_DEDUP_SHX");
        shx_env = (e && *e && *e != '0') ? 1 : 0;
    }
    if (shx_env && ctx->kern_moe_gateup_q3k_dedup_shx)
        kern = ctx->kern_moe_gateup_q3k_dedup_shx;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &gate_all->dptr, &up_all->dptr, &x->dptr, &ecount->dptr, &ereq->dptr,
        &hid->dptr, &k, &ff, &cols, &row_stride, &maxr
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)((ff + 7) / 8), (unsigned int)n_expert, zt,
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_gateup_q3k_dedup launch failed: %d\n",
                err);
}

/* Expert count fc_moe_dedup_build_wl's single-block scan can hold. Must match
 * FC_MOE_WL_EXPERT_CAP in cuda_kernels/moe_expert_group.cu. */
#define FC_MOE_WL_EXPERT_CAP 1024

/* Exact upper bound on the compacted work-item count. Derivation and the
 * attaining router are in fc_cuda.h; the three guards below are the encoding
 * limits of the device work item and of the builder's scan. */
int fc_moe_dedup_wmax(int n_expert, int k, int ntok, int maxr)
{
    if (n_expert <= 0 || k <= 0 || ntok <= 0 || maxr <= 0)
        return 0;
    if (n_expert > FC_MOE_WL_EXPERT_CAP)
        return 0;                     /* beyond the builder's shared-mem scan */
    if (maxr > 65535)
        return 0;                     /* r0 no longer fits the packed item */

    long long total = (long long)ntok * k;
    long long uniq = (total < (long long)n_expert) ? total : (long long)n_expert;
    long long b1 = (total + (FC_MOE_DEDUP_RT - 1) * uniq) / FC_MOE_DEDUP_RT;
    long long b2 = uniq * ((maxr + FC_MOE_DEDUP_RT - 1) / FC_MOE_DEDUP_RT);
    long long w = (b1 < b2) ? b1 : b2;
    if (total < w)
        w = total;
    return (int)w;
}

int fc_moe_dedup_compact_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_moe_dedup_build_wl &&
           ctx->kern_moe_gateup_q3k_dedup_shxc;
}

int fc_moe_dedup_compact_enabled_ctx(fc_cuda_ctx *ctx)
{
    static int compact_env = -1;
    if (compact_env < 0) {
        const char *e = getenv("FC_MOE_DEDUP_COMPACT");
        compact_env = (e && *e && *e != '0') ? 1 : 0;
    }
    return compact_env && fc_moe_dedup_compact_available_ctx(ctx);
}

/* Compact ecount into the (expert, request-tile) work list. Single block, like
 * fc_moe_dedup_build; must be launched on the same stream between it and the
 * compacted gate+up. */
void fc_moe_dedup_build_wl_ctx(fc_cuda_ctx *ctx, fc_buffer *ecount,
                               fc_buffer *wlist, int n_expert, int wmax)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_dedup_build_wl ||
        !ecount || !wlist || n_expert <= 0 || wmax <= 0)
        return;

    int threads = 256;
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &ecount->dptr, &wlist->dptr, &n_expert, &wmax };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_dedup_build_wl,
        1, 1, 1,                                /* single block */
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_dedup_build_wl launch failed: %d\n",
                err);
}

/* Grouped gate+up over the COMPACTED work list: grid (ceil(ff/8), wmax, 1)
 * instead of (ceil(ff/8), n_expert, ceil(maxr/RT)). Bit-exact with
 * fc_moe_gateup_q3k_dedup{,_shx} — same work items, same body, one plain store
 * per output element. */
void fc_moe_gateup_q3k_dedup_compact_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                                         fc_buffer *up_all, fc_buffer *x,
                                         fc_buffer *ereq, fc_buffer *wlist,
                                         fc_buffer *hid, int k, int ff,
                                         int cols, int row_stride, int wmax,
                                         int maxr)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_gateup_q3k_dedup_shxc ||
        !gate_all || !up_all || !x || !ereq || !wlist || !hid || wmax <= 0 ||
        maxr <= 0)
        return;

    int threads = 256;
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &gate_all->dptr, &up_all->dptr, &x->dptr, &ereq->dptr, &wlist->dptr,
        &hid->dptr, &k, &ff, &cols, &row_stride, &maxr
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_gateup_q3k_dedup_shxc,
        (unsigned int)((ff + 7) / 8), (unsigned int)wmax, 1,
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr,
                "[fc_cuda] moe_gateup_q3k_dedup_compact launch failed: %d\n",
                err);
}

void fc_moe_down_accum_nx_ctx(fc_cuda_ctx *ctx, int gguf_quant,
                              fc_buffer *down_all, fc_buffer *hid,
                              fc_buffer *top_idx, fc_buffer *top_wt,
                              fc_buffer *out, int k, int rows, int ff,
                              int row_stride, int ntok)
{
    if (!fc_cuda_ctx_available(ctx) || !down_all || !hid || !top_idx ||
        !top_wt || !out || ntok <= 0)
        return;

    int vec = fc_moe_vec_loads_enabled();
    CUfunction kern = NULL;
    if (gguf_quant == 11)
        kern = ctx->kern_moe_down_q3k_nx;
    else if (gguf_quant == 12)
        kern = ctx->kern_moe_down_q4k_nx;
    else if (gguf_quant == 14)
        kern = ctx->kern_moe_down_q6k_nx;
    else if (gguf_quant == 1)   /* GGUF_QUANT_F16 */
        kern = vec ? ctx->kern_moe_down_f16_vec_nx : ctx->kern_moe_down_f16_nx;
    else if (gguf_quant == 30)  /* GGUF_QUANT_BF16 */
        kern = vec ? ctx->kern_moe_down_bf16_vec_nx
                   : ctx->kern_moe_down_bf16_nx;
    else if (gguf_quant == 8)   /* GGUF_QUANT_Q8_0 (expert layout) */
        kern = (ntok > fc_moe_q80_ldg_max_tok() && ctx->kern_moe_down_q80_nxp)
             ? ctx->kern_moe_down_q80_nxp : ctx->kern_moe_down_q80_nx;
    if (!kern) {
        fprintf(stderr, "[fc_cuda] moe_down_accum_nx: unsupported quant %d\n",
                gguf_quant);
        return;
    }

    /* Same tuned shape as fc_moe_down_accum_ctx (wpp/threads defaults). */
    int wpp = FC_MOE_DOWN_WPP_DEFAULT;
    int threads = FC_MOE_DOWN_THREADS_DEFAULT;
    int rows_per_block = 2 * (threads / 32) / wpp;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &down_all->dptr, &hid->dptr, &top_idx->dptr, &top_wt->dptr, &out->dptr,
        &k, &rows, &ff, &row_stride, &wpp
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)((rows + rows_per_block - 1) / rows_per_block),
        (unsigned int)ntok, 1,
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_down_accum_nx launch failed: %d\n",
                err);
}

void fc_moe_shexp_dot_gate_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_w,
                                  fc_buffer *x, fc_buffer *shexp, int n,
                                  int ntok)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_shexp_dot_gate_nx ||
        !gate_w || !x || !shexp || ntok <= 0)
        return;

    int threads = 256;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &gate_w->dptr, &x->dptr, &shexp->dptr, &n };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_shexp_dot_gate_nx,
        (unsigned int)ntok, 1, 1,               /* one block per token */
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] moe_shexp_dot_gate_nx launch failed: %d\n",
                err);
}

int fc_moe_shexp_gate_combine_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_w,
                                     fc_buffer *x, fc_buffer *shexp,
                                     fc_buffer *moe_out, fc_buffer *block_out,
                                     fc_buffer *hidden, int n, int ntok)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_moe_shexp_gate_combine_nx ||
        !gate_w || !x || !shexp || !moe_out || !block_out || !hidden ||
        ntok <= 0)
        return -1;

    /* 256 threads: the reduction tree of fc_moe_shexp_dot_gate_nx. */
    int threads = 256;
    unsigned int shared_bytes = (unsigned int)(threads * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = { &gate_w->dptr, &x->dptr, &shexp->dptr, &moe_out->dptr,
                     &block_out->dptr, &hidden->dptr, &n };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_moe_shexp_gate_combine_nx,
        (unsigned int)ntok, 1, 1,
        (unsigned int)threads, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] moe_shexp_gate_combine_nx launch failed: "
                "%d\n", err);
        return -1;
    }
    return 0;
}

void fc_ssm_conv1d_silu_seq_ctx(fc_cuda_ctx *ctx, fc_buffer *qkv,
                                fc_buffer *weight, fc_buffer *conv_state,
                                fc_buffer *out, int conv_dim, int conv_kernel,
                                int npos, fc_buffer *snaps)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_conv1d_silu_seq ||
        !qkv || !weight || !conv_state || !out || npos <= 0 ||
        conv_kernel > 4)
        return;

    CUdeviceptr snaps_ptr = snaps ? snaps->dptr : 0;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &qkv->dptr, &weight->dptr, &conv_state->dptr, &out->dptr,
        &conv_dim, &conv_kernel, &npos, &snaps_ptr
    };
    fc_dispatch_1d(ctx, ctx->kern_ssm_conv1d_silu_seq, args, conv_dim);
    fc_ctx_leave();
}

/* The 8-lane layout of the S == 128 warp kernels (fc_ssm_recurrence_seq_warp8,
 * fc_ssm_gdn_step_slots_warp8): 16 columns a 128-thread block, state / k / q
 * (and snaps) as 16-B float4 accesses. Same bits as the 32-lane kernels;
 * 1.26-1.65x on a K80 die. DEFAULT ON; FC_SSM_WARP8=0 keeps the 32-lane ones.
 * Taken only when the kernel loaded and every float4 pointer is 16-B aligned. */
#define FC_SSM_W8_COLS_HOST 16

static int fc_ssm_warp8_usable(CUfunction kern8, const fc_buffer *s,
                               const fc_buffer *q, const fc_buffer *k,
                               const fc_buffer *snaps)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("FC_SSM_WARP8");
        on = (e && e[0] == '0') ? 0 : 1;
    }
    const CUdeviceptr any = s->dptr | q->dptr | k->dptr | (snaps ? snaps->dptr : 0);
    return on && kern8 && (any & 15) == 0;
}

void fc_ssm_recurrence_seq_ctx(fc_cuda_ctx *ctx, fc_buffer *s, fc_buffer *q,
                               fc_buffer *k, fc_buffer *v, fc_buffer *g,
                               fc_buffer *beta, fc_buffer *y,
                               int n_v_heads, int state_size, int npos,
                               float scale, fc_buffer *snaps)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_recurrence_seq ||
        !s || !q || !k || !v || !g || !beta || !y || npos <= 0)
        return;

    unsigned int shared_bytes = (unsigned int)((size_t)state_size * sizeof(float));
    CUdeviceptr snaps_ptr = snaps ? snaps->dptr : 0;

    /* fc_ssm_recurrence_seq_warp (DEFAULT ON; QWEN35_SSM_WARP=0 restores the
     * baseline): one warp per state column, the length-S dot reduced by a
     * tree-preserving __shfl_xor butterfly instead of a barrier-chained
     * shared-memory sweep. BIT-EXACT (same reduction tree, contraction-barrier
     * intrinsics, same expf): Qwen3.8-27B's 1,024-token prefill logits hash the
     * same with and without it on a K80, and the recurrence fell 7.4x (13.5 s ->
     * 1.8 s of a 93 s sequential prefill). The K80 fleet scripts already set it;
     * Pantheon Local's serve did not. Specialised to state_size == 128; the
     * staging buffer is 2*npos floats, so npos is bounded below (else baseline).
     * Process-constant, same pattern as QWEN35_SSM_2PASS. */
    static int use_warp = -1;
    if (use_warp < 0) {
        const char *e = getenv("QWEN35_SSM_WARP");
        use_warp = (e && e[0] == '0') ? 0 : 1;
    }

    /* QWEN35_SSM_UNROLL=1 routes to fc_ssm_recurrence_seq_unroll128: the SAME
     * barrier-chained shared-memory sweep as the baseline with the tree's trip
     * count written as a literal so ptxas can unroll it. Strictly weaker than
     * the warp variant (it keeps the barriers and the shared traffic; it only
     * deletes the 45 % of issue slots that were loop control) but it cannot cost
     * occupancy, so it is the fallback if the warp kernel's register count
     * disappoints. QWEN35_SSM_WARP wins if both are set. Same op order as the
     * baseline, so bit-exact by construction. */
    static int use_unroll = -1;
    if (use_unroll < 0) {
        const char *e = getenv("QWEN35_SSM_UNROLL");
        use_unroll = (e && e[0] && e[0] != '0') ? 1 : 0;
    }

    CUfunction kern = ctx->kern_ssm_recurrence_seq;
    unsigned int grid_x = (unsigned int)state_size;
    unsigned int block_x = (unsigned int)state_size;
    const int warp_cols = 4;             /* must match FC_SSM_WARP_COLS */
    if (use_warp && ctx->kern_ssm_recurrence_seq_warp && state_size == 128 &&
        (size_t)2 * npos * sizeof(float) <= 4096) {
        kern = ctx->kern_ssm_recurrence_seq_warp;
        grid_x = (unsigned int)(state_size / warp_cols);
        block_x = (unsigned int)(warp_cols * 32);
        shared_bytes = (unsigned int)((size_t)2 * npos * sizeof(float));
        if (fc_ssm_warp8_usable(ctx->kern_ssm_recurrence_seq_warp8, s, q, k,
                                snaps)) {
            kern = ctx->kern_ssm_recurrence_seq_warp8;
            grid_x = (unsigned int)(state_size / FC_SSM_W8_COLS_HOST);
        }
    } else if (use_unroll && ctx->kern_ssm_recurrence_seq_unroll128 &&
               state_size == 128) {
        kern = ctx->kern_ssm_recurrence_seq_unroll128;
    }

    /* Announce ONCE per process which kernel the flags actually selected. An
     * A/B has to be able to prove the flag changed the code path from the
     * process's own mouth, not from the plan document — and a DECLINED line is
     * what catches a guard silently falling back (state_size != 128, or npos so
     * large the staging buffer would wreck occupancy). */
    static int announced = 0;
    if (!announced) {
        announced = 1;
        if (kern == ctx->kern_ssm_recurrence_seq_warp ||
            kern == ctx->kern_ssm_recurrence_seq_warp8)
            fprintf(stderr, "[fc_cuda] QWEN35_SSM_WARP ACTIVE: warp-shuffle "
                    "SSM recurrence (%d lanes a column), %u B shared, %u thr "
                    "— BIT-EXACT\n",
                    kern == ctx->kern_ssm_recurrence_seq_warp8 ? 8 : 32,
                    shared_bytes, block_x);
        else if (kern == ctx->kern_ssm_recurrence_seq_unroll128)
            fprintf(stderr, "[fc_cuda] QWEN35_SSM_UNROLL=1 ACTIVE: unrolled-tree "
                    "SSM recurrence (S=128) — BIT-EXACT\n");
        else if (use_warp || use_unroll)
            fprintf(stderr, "[fc_cuda] SSM recurrence variant DECLINED "
                    "(warp=%d kern=%d unroll=%d kern=%d state_size=%d npos=%d)\n",
                    use_warp, ctx->kern_ssm_recurrence_seq_warp != NULL,
                    use_unroll, ctx->kern_ssm_recurrence_seq_unroll128 != NULL,
                    state_size, npos);
    }

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &s->dptr, &q->dptr, &k->dptr, &v->dptr, &g->dptr, &beta->dptr,
        &y->dptr, &state_size, &n_v_heads, &npos, &scale, &snaps_ptr
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        grid_x, (unsigned int)n_v_heads, 1,
        block_x, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] ssm_recurrence_seq launch failed: %d\n",
                err);
}

void fc_ssm_decay_gate_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *alpha,
                              fc_buffer *ssm_dt, fc_buffer *ssm_a,
                              fc_buffer *g, int n_heads, int npos)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_decay_gate_nx ||
        !alpha || !ssm_dt || !ssm_a || !g || npos <= 0)
        return;
    int total = n_heads * npos;
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &alpha->dptr, &ssm_dt->dptr, &ssm_a->dptr, &g->dptr, &n_heads, &total
    };
    fc_dispatch_1d(ctx, ctx->kern_ssm_decay_gate_nx, args, total);
    fc_ctx_leave();
}

void fc_ssm_qknorm_gqa_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *conv_out,
                              fc_buffer *qg, fc_buffer *kg, fc_buffer *vg,
                              int n_k_heads, int kv_groups, int state_size,
                              int key_dim, int val_dim, float eps, int npos)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_qknorm_gqa_nx ||
        !conv_out || !qg || !kg || !vg || npos <= 0)
        return;

    unsigned int shared_bytes =
        (unsigned int)(2 * (size_t)state_size * sizeof(float));

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &conv_out->dptr, &qg->dptr, &kg->dptr, &vg->dptr, &n_k_heads,
        &kv_groups, &state_size, &key_dim, &val_dim, &eps
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_ssm_qknorm_gqa_nx,
        (unsigned int)n_k_heads, (unsigned int)npos, 1,
        (unsigned int)state_size, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] ssm_qknorm_gqa_nx launch failed: %d\n",
                err);
}

/* ================================================================
 * Multi-user decode (slot) kernels — K users, one token each
 *
 * Every slot kernel takes an optional ROW MAP: rows[u] is the persistent
 * state/ring row that user u of this launch reads and writes. NULL (device
 * pointer 0) = identity, rows[u] == u. Per-user activations (q, partials,
 * qkv, y, ...) stay indexed by the launch position u.
 * ================================================================ */

static CUdeviceptr fc_rows_dptr(const fc_buffer *rows)
{
    return rows ? rows->dptr : (CUdeviceptr)0;
}

void fc_ssm_conv1d_silu_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *qkv,
                                  fc_buffer *weight, fc_buffer *conv_state,
                                  fc_buffer *out, int conv_dim,
                                  int conv_kernel, int nslots,
                                  fc_buffer *rows)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_conv1d_silu_slots ||
        !qkv || !weight || !conv_state || !out || nslots <= 0 ||
        conv_kernel > 4)
        return;

    int threads = 256;
    if (conv_dim < threads)
        threads = conv_dim;
    unsigned int blocks_x = (unsigned int)((conv_dim + threads - 1) / threads);

    if (fc_ctx_enter(ctx) != 0)
        return;
    CUdeviceptr rows_d = fc_rows_dptr(rows);
    void *args[] = {
        &qkv->dptr, &weight->dptr, &conv_state->dptr, &out->dptr,
        &conv_dim, &conv_kernel, &rows_d
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_ssm_conv1d_silu_slots,
        blocks_x, (unsigned int)nslots, 1,
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] ssm_conv1d_silu_slots launch failed: %d\n",
                err);
}

void fc_ssm_recurrence_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *s, fc_buffer *q,
                                 fc_buffer *k, fc_buffer *v, fc_buffer *g,
                                 fc_buffer *beta, fc_buffer *y,
                                 int n_v_heads, int state_size, int nslots,
                                 float scale, fc_buffer *rows)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_recurrence_slots ||
        !s || !q || !k || !v || !g || !beta || !y || nslots <= 0)
        return;

    /* T4 (fable2): QWEN35_SSM_2PASS=1 routes to the fused-reduction variant
     * (one LDS sweep, 3*S shared) — lossy-tiny by reassociation, default OFF so
     * the serve path stays the bit-exact baseline kernel. Process-constant. */
    static int use_2pass = -1;
    if (use_2pass < 0) {
        const char *e = getenv("QWEN35_SSM_2PASS");
        use_2pass = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    CUfunction kern = ctx->kern_ssm_recurrence_slots;
    unsigned int shared_bytes =
        (unsigned int)((size_t)state_size * sizeof(float));
    if (use_2pass && ctx->kern_ssm_recurrence_slots_2pass) {
        kern = ctx->kern_ssm_recurrence_slots_2pass;
        shared_bytes = (unsigned int)((size_t)3 * state_size * sizeof(float));
    }

    if (fc_ctx_enter(ctx) != 0)
        return;
    CUdeviceptr rows_d = fc_rows_dptr(rows);
    void *args[] = {
        &s->dptr, &q->dptr, &k->dptr, &v->dptr, &g->dptr, &beta->dptr,
        &y->dptr, &state_size, &n_v_heads, &scale, &rows_d
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)state_size, (unsigned int)n_v_heads,
        (unsigned int)nslots,
        (unsigned int)state_size, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] ssm_recurrence_slots launch failed: %d\n",
                err);
}

int fc_ssm_gdn_step_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *s, fc_buffer *q,
                              fc_buffer *k, fc_buffer *v, fc_buffer *alpha,
                              fc_buffer *ssm_dt, fc_buffer *ssm_a,
                              fc_buffer *beta_raw, fc_buffer *y,
                              int n_v_heads, int state_size, int nslots,
                              float scale, fc_buffer *rows)
{
    /* The kernels' warp layouts are specialised to S == 128. */
    const int warp_cols = 4;
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_ssm_gdn_step_slots_warp ||
        state_size != 128 || !s || !q || !k || !v || !alpha || !ssm_dt ||
        !ssm_a || !beta_raw || !y || nslots <= 0)
        return -1;
    if (fc_ctx_enter(ctx) != 0)
        return -1;
    CUdeviceptr rows_d = fc_rows_dptr(rows);
    void *args[] = {
        &s->dptr, &q->dptr, &k->dptr, &v->dptr, &alpha->dptr, &ssm_dt->dptr,
        &ssm_a->dptr, &beta_raw->dptr, &y->dptr, &state_size, &n_v_heads,
        &scale, &rows_d
    };
    const int w8 = fc_ssm_warp8_usable(ctx->kern_ssm_gdn_step_slots_warp8, s, q, k,
                                       NULL);
    CUresult err = fcw_cuLaunchKernel(
        w8 ? ctx->kern_ssm_gdn_step_slots_warp8 : ctx->kern_ssm_gdn_step_slots_warp,
        (unsigned int)(state_size / (w8 ? FC_SSM_W8_COLS_HOST : warp_cols)),
        (unsigned int)n_v_heads,
        (unsigned int)nslots,
        (unsigned int)(warp_cols * 32), 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] ssm_gdn_step_slots_warp launch failed: %d\n",
                err);
        return -1;
    }
    return 0;
}

void fc_rope_neox_partial_pos_ctx(fc_cuda_ctx *ctx, fc_buffer *data,
                                  int n_heads, int head_dim, int rope_dim,
                                  fc_buffer *pos, float theta_base,
                                  int nrows)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_rope_pos || !data ||
        !pos || nrows <= 0)
        return;

    int rot = (rope_dim > 0 && rope_dim < head_dim) ? rope_dim : head_dim;
    int half = rot / 2;
    int threads = (half < 256) ? half : 256;
    if (threads < 1)
        threads = 1;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &data->dptr, &n_heads, &head_dim, &rope_dim, &pos->dptr, &theta_base
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_rope_pos,
        (unsigned int)n_heads, (unsigned int)nrows, 1,
        (unsigned int)threads, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] rope_neox_partial_pos launch failed: %d\n",
                err);
}

/* ================================================================
 * On-die causal GQA flash attention (attention.ptx)
 * ================================================================ */

/* Must match attention.cu. */
#define FC_ATTN_TILE_KEYS  16
#define FC_ATTN_DOT_LANES  16
#define FC_ATTN_MAX_GROUPS 8
#define FC_ATTN_K_PAD      16
#define FC_ATTN_K_STRIDE(hd) ((hd) + FC_ATTN_K_PAD)
/* Compile-time bound on head_dim / FC_ATTN_DOT_LANES for the register-K prefill
 * variant. MUST match FC_ATTN_KREG_MAX_CHUNK in cuda_kernels/attention.cu. */
#define FC_ATTN_KREG_MAX_CHUNK 16
/* Q4_0 KV-cache block geometry, host mirror of cuda_kernels/attention.cu's
 * FC_KVQ4_BLK / FC_KVQ4_BLK_BYTES and fc_kvq4_row_bytes(). */
#define FC_ATTN_KVQ4_BLK       32
#define FC_ATTN_KVQ4_BLK_BYTES 18
#define FC_ATTN_KVQ4_ROW_BYTES(hd) (((hd) / FC_ATTN_KVQ4_BLK) * FC_ATTN_KVQ4_BLK_BYTES)

/* Dynamic-shared bytes for ONE flash-attention block, prefill or decode:
 * q group + the padded K tile + scores + probs + m/l/corr. MUST match the
 * shared layout and FC_ATTN_K_STRIDE in cuda_kernels/attention.cu. Returns 0
 * when the block would exceed sm_37's 48 KB per-block dynamic-shared ceiling,
 * so an out-of-contract shape declines here (host fallback) instead of dying
 * in cuLaunchKernel. */
static unsigned int fc_attn_flash_shared_bytes(int kv_groups, int head_dim)
{
    size_t floats =
        (size_t)kv_groups * head_dim +                              /* q group */
        (size_t)FC_ATTN_TILE_KEYS * FC_ATTN_K_STRIDE(head_dim) +    /* K tile */
        2u * (size_t)kv_groups * FC_ATTN_TILE_KEYS +           /* scores+probs */
        3u * (size_t)kv_groups;                                   /* m,l,corr */
    size_t bytes = floats * sizeof(float);
    if (bytes > 48u * 1024u)
        return 0;
    return (unsigned int)bytes;
}

/* Dynamic-shared bytes for the NARROW-STAGING prefill variants
 * (fc_attn_flash_prefill_q4_kh / _kqh): the K tile is __half, and at level 2
 * so is the q group; the softmax state stays fp32. Shrinking the footprint is
 * the whole point — residency on this kernel is shared-capped, so 26 720 B ->
 * 18 016 B is 4 -> 6 blocks/SMX and -> 13 920 B is 7. MUST match the shared
 * layout in fc_attn_flash_prefill_q4_narrow() in cuda_kernels/attention.cu. */
static unsigned int fc_attn_flash_narrow_shared_bytes(int kv_groups,
                                                      int head_dim,
                                                      int q_fp16)
{
    size_t bytes =
        (size_t)kv_groups * head_dim * (q_fp16 ? 2u : 4u) +          /* q group */
        (size_t)FC_ATTN_TILE_KEYS * FC_ATTN_K_STRIDE(head_dim) * 2u +  /* K tile */
        ((size_t)2u * kv_groups * FC_ATTN_TILE_KEYS +
         3u * (size_t)kv_groups) * 4u;                       /* scores+probs+state */
    if (bytes > 48u * 1024u)
        return 0;
    return (unsigned int)bytes;
}

/* Dynamic-shared bytes for the REGISTER-K prefill variant
 * (fc_attn_flash_prefill_q4_kr): the K tile is GONE — held in per-thread
 * registers — so only the q group and the fp32 softmax state remain. 26 720 ->
 * 9 312 B at the shipped shape, which lifts residency 4 -> 7 blocks/SMX (7 is
 * then the REGISTER cap at 72 regs x 256 threads, not the shared cap, so extra
 * shared padding below ~16 KB cannot buy or cost a block — this path
 * deliberately ignores FC_ATTN_KTILE_SHARED_PAD). MUST match the shared layout
 * in fc_attn_flash_prefill_q4_kr() in cuda_kernels/attention.cu. */
static unsigned int fc_attn_flash_kreg_shared_bytes(int kv_groups, int head_dim)
{
    size_t floats =
        (size_t)kv_groups * head_dim +                              /* q group */
        2u * (size_t)kv_groups * FC_ATTN_TILE_KEYS +           /* scores+probs */
        3u * (size_t)kv_groups;                                   /* m,l,corr */
    size_t bytes = floats * sizeof(float);
    if (bytes > 48u * 1024u)
        return 0;
    return (unsigned int)bytes;
}

/* Dynamic-shared bytes for the SHARED-V prefill variant
 * (fc_attn_flash_prefill_q4_kvs): the register-K layout plus the key tile's V
 * rows staged as raw q4 bytes. 9 312 -> 11 616 B at the shipped shape, and
 * residency is UNCHANGED at 7 blocks/SMX because 7 is the REGISTER cap at 72
 * registers x 256 threads, not the shared cap (114 688/7 = 16 384 B is the
 * budget). MUST match the shared layout in fc_attn_flash_prefill_q4_kvs() in
 * cuda_kernels/attention.cu. */
static unsigned int fc_attn_flash_kvs_shared_bytes(int kv_groups, int head_dim)
{
    unsigned int base = fc_attn_flash_kreg_shared_bytes(kv_groups, head_dim);
    if (base == 0)
        return 0;
    size_t bytes = (size_t)base +
                   (size_t)FC_ATTN_TILE_KEYS * (size_t)FC_ATTN_KVQ4_ROW_BYTES(head_dim);
    if (bytes > 48u * 1024u)
        return 0;
    return (unsigned int)bytes;
}

/* Dynamic-shared bytes for fc_attn_flash_prefill_q4_g8 (kv_groups 8, head_dim
 * 256 only): q group, scores, probs, m / l / corr / m_new, then the V tile and
 * the staged K tile. 13 952 B (7 blocks/SMX stay register-capped). MUST match
 * FC_ATTN_G8_K_OFF + one tile in cuda_kernels/attention_prefill_g8.cuh. */
static unsigned int fc_attn_flash_g8_shared_bytes(void)
{
    return (unsigned int)((8u * 256u + 2u * 8u * FC_ATTN_TILE_KEYS + 4u * 8u) *
                              sizeof(float) +
                          2u * (size_t)FC_ATTN_TILE_KEYS * FC_ATTN_KVQ4_ROW_BYTES(256));
}

/* FC_ATTN_G8 (default on; =0 keeps kvs): the kv_groups 8 / head_dim 256
 * specialisation of the shared-V prefill kernel. BIT-EXACT against kvs (0 of
 * 262 144 outputs differ at depth 8k-32k, cuda_kernels/tests/bench_attn_prefill_g8.cu),
 * 1.60-1.70x on the kernel. */
static int fc_attn_g8_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("FC_ATTN_G8");
        on = (e && *e && atoi(e) == 0) ? 0 : 1;
    }
    return on;
}

/* FC_ATTN_KTILE_REG=1: hold the prefill K tile in per-thread registers instead
 * of shared memory (fc_attn_flash_prefill_q4_kr). BIT-EXACT — 0 of 524 288
 * output floats differ — and measured 1.51x on the standalone microbenchmark.
 * Default OFF pending the in-model ship decision. Takes PRECEDENCE over
 * FC_ATTN_KTILE_FP16, which it strictly dominates: more shared freed, higher
 * residency, and lossless instead of lossy. */
static int fc_attn_ktile_reg_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("FC_ATTN_KTILE_REG");
        on = (e && *e && atoi(e) != 0) ? 1 : 0;
    }
    return on;
}

/* FC_ATTN_VSTAGE: additionally stage the prefill V tile's raw q4 bytes in
 * shared (fc_attn_flash_prefill_q4_kvs). BIT-EXACT, and MEASURED 1.1497x on the
 * kernel over fc_attn_flash_prefill_q4_kr — die 6 pinned 666 MHz, noise floor
 * 0.301 %, depth-flat 8 k -> 32 k, 0 of 524 288 outputs differ
 * (cuda_kernels/tests/bench_attn_vstage.cu). Implies FC_ATTN_KTILE_REG: the kvs
 * kernel IS the register-K kernel plus the V tile, so this flag alone is enough.
 *
 * SINGLE GATE, deliberately. There is no compile-time switch: the kernel is
 * always present in the embedded PTX and selection is purely this env var. A
 * dual compile-time-plus-getenv gate is how FC_MOE_DEDUP_COMPACT ended up able
 * to look enabled while shipping a dead flag. DEFAULT ON (FC_ATTN_VSTAGE=0
 * restores the shipped kernel): the K80 fleet scripts set it, Pantheon Local's
 * serve did not, and Qwen3.8-27B's 4,096-token prefill logits hash the same
 * with it on and off. */
static int fc_attn_vstage_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("FC_ATTN_VSTAGE");
        on = (e && *e && atoi(e) == 0) ? 0 : 1;
    }
    return on;
}

/* FC_ATTN_KTILE_FP16: 0/unset = off (the shipped fp32-staged kernel),
 * 1 = __half K tile, 2 = __half K tile AND q group. LOSSY — see
 * Pantheon results/prefill_attention_ktile_fp16.md for the eval gate. */
static int fc_attn_ktile_fp16_level(void)
{
    static int level = -1;
    if (level < 0) {
        const char *e = getenv("FC_ATTN_KTILE_FP16");
        level = (e && *e) ? atoi(e) : 0;
        if (level < 0 || level > 2)
            level = 0;
    }
    return level;
}

/* FC_ATTN_KTILE_SHARED_PAD: extra dynamic-shared bytes requested by the
 * narrow-staging variants, i.e. a residency dial. Sounds backwards and is not:
 * the prefill grid is npos * n_head_kv = 128 * 2 = 256 blocks against 13 SMX,
 * so residency quantises the launch into ceil(256 / (13*b)) waves — 5 waves at
 * b=4, FOUR at both b=5 and b=6. Six resident blocks therefore buys no wave
 * over five and only adds contention: at the shipped grid, PADDING 18 016 ->
 * 18 976 B (b=6 -> b=5) is measured FASTER. Default 0 (natural footprint);
 * the tuning is grid-specific and belongs in an env var, not in a constant. */
static unsigned int fc_attn_ktile_shared_pad(void)
{
    static long pad = -1;
    if (pad < 0) {
        const char *e = getenv("FC_ATTN_KTILE_SHARED_PAD");
        pad = (e && *e) ? atol(e) : 0;
        if (pad < 0 || pad > 48L * 1024L)
            pad = 0;
    }
    return (unsigned int)pad;
}

int fc_attn_flash_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_attn_qz_split &&
           ctx->kern_attn_kv_append && ctx->kern_attn_flash_prefill;
}

void fc_attn_qz_split_ctx(fc_cuda_ctx *ctx, fc_buffer *qg, fc_buffer *q,
                          fc_buffer *z, int n_head, int head_dim, int npos)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_qz_split || !qg ||
        !q || !z || npos <= 0)
        return;

    int total = npos * n_head * head_dim;
    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = { &qg->dptr, &q->dptr, &z->dptr, &head_dim, &total };
    fc_dispatch_1d(ctx, ctx->kern_attn_qz_split, args, total);
    fc_ctx_leave();
}

void fc_attn_kv_append_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                           fc_buffer *v_new, fc_buffer *k_cache,
                           fc_buffer *v_cache, int n_head_kv, int head_dim,
                           int max_seq, int pos0, int npos)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_kv_append || !k_new ||
        !v_new || !k_cache || !v_cache || npos <= 0)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &k_new->dptr, &v_new->dptr, &k_cache->dptr, &v_cache->dptr,
        &n_head_kv, &head_dim, &max_seq, &pos0
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_kv_append,
        (unsigned int)npos, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] attn_kv_append launch failed: %d\n", err);
}

int fc_attn_flash_prefill_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                              fc_buffer *k_cache, fc_buffer *v_cache,
                              fc_buffer *out, int n_head, int n_head_kv,
                              int head_dim, int max_seq, int pos0, int npos,
                              int attn_sink, int attn_win, float scale)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_flash_prefill || !q ||
        !k_cache || !v_cache || !out || npos <= 0 || n_head_kv <= 0)
        return -1;

    /* Kernel contract: 16-lane dots need head_dim % 16 == 0; one thread per
     * output dim needs head_dim <= 1024; the register accumulator caps the
     * GQA group; causal indexing assumes no ring wrap in the window. */
    int kv_groups = n_head / n_head_kv;
    if (head_dim % FC_ATTN_DOT_LANES != 0 || head_dim > 1024 ||
        kv_groups < 1 || kv_groups > FC_ATTN_MAX_GROUPS ||
        kv_groups * n_head_kv != n_head || pos0 + npos > max_seq)
        return -1;

    unsigned int shared_bytes = fc_attn_flash_shared_bytes(kv_groups, head_dim);
    if (shared_bytes == 0)
        return -1;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr, &out->dptr,
        &kv_groups, &head_dim, &max_seq, &pos0, &attn_sink, &attn_win, &scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_prefill,
        (unsigned int)npos, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_prefill launch failed: %d\n",
                err);
        return -1;
    }
    return 0;
}

int fc_attn_flash_q4_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_attn_kv_append_q4 &&
           ctx->kern_attn_flash_prefill_q4;
}

void fc_attn_kv_append_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                              fc_buffer *v_new, fc_buffer *k_cache,
                              fc_buffer *v_cache, fc_buffer *k_sink,
                              fc_buffer *v_sink, fc_buffer *k_win,
                              fc_buffer *v_win, int n_head_kv, int head_dim,
                              int max_seq, int pos0, int npos, int sink,
                              int win)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_kv_append_q4 ||
        !k_new || !v_new || !k_cache || !v_cache || !k_sink || !v_sink ||
        !k_win || !v_win || npos <= 0 || (head_dim % 32) != 0)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &k_new->dptr, &v_new->dptr, &k_cache->dptr, &v_cache->dptr,
        &k_sink->dptr, &v_sink->dptr, &k_win->dptr, &v_win->dptr,
        &n_head_kv, &head_dim, &max_seq, &pos0, &sink, &win
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_kv_append_q4,
        (unsigned int)npos, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] attn_kv_append_q4 launch failed: %d\n", err);
}

int fc_attn_flash_prefill_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                 fc_buffer *k_cache, fc_buffer *v_cache,
                                 fc_buffer *k_sink, fc_buffer *v_sink,
                                 fc_buffer *k_win, fc_buffer *v_win,
                                 fc_buffer *out, int n_head, int n_head_kv,
                                 int head_dim, int max_seq, int pos0, int npos,
                                 int sink, int win, int attn_sink, int attn_win,
                                 float scale)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_flash_prefill_q4 || !q ||
        !k_cache || !v_cache || !k_sink || !v_sink || !k_win || !v_win ||
        !out || npos <= 0 || n_head_kv <= 0)
        return -1;

    int kv_groups = n_head / n_head_kv;
    if (head_dim % FC_ATTN_DOT_LANES != 0 || (head_dim % 32) != 0 ||
        head_dim > 1024 || kv_groups < 1 || kv_groups > FC_ATTN_MAX_GROUPS ||
        kv_groups * n_head_kv != n_head || pos0 + npos > max_seq)
        return -1;

    unsigned int shared_bytes = fc_attn_flash_shared_bytes(kv_groups, head_dim);
    if (shared_bytes == 0)
        return -1;

    /* Variant selection. Both flags keep the args and the grid and change only
     * the kernel and its dynamic-shared request, and both fall back to the
     * shipped kernel rather than launching something out of contract.
     *
     * FC_ATTN_KTILE_REG is checked FIRST and wins when both are set: it is
     * bit-exact where FC_ATTN_KTILE_FP16 is lossy, and it frees more shared. Its
     * contract is head_dim in [TILE_KEYS*DOT_LANES, KREG_MAX_CHUNK*DOT_LANES]
     * (= exactly 256 today) and full attention. The lower bound is not a
     * register-file limit but the pre-existing rule that a block has head_dim
     * threads and must therefore supply at least FC_ATTN_TILE_KEYS distinct
     * `ky` values to score a full key tile; attn_win > 0 is declined because
     * the windowed path is unmeasured on this variant, not because it is known
     * wrong (the tile predicate is the shipped one). */
    CUfunction kern = ctx->kern_attn_flash_prefill_q4;
    int narrow = fc_attn_ktile_fp16_level();
    if (fc_attn_ktile_reg_enabled() || fc_attn_vstage_enabled()) {
        unsigned int rb = fc_attn_flash_kreg_shared_bytes(kv_groups, head_dim);
        int shape_ok = head_dim <= FC_ATTN_KREG_MAX_CHUNK * FC_ATTN_DOT_LANES &&
                       head_dim >= FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES &&
                       attn_win <= 0;
        /* FC_ATTN_VSTAGE additionally stages the V tile in shared. It IS the
         * register-K kernel plus that tile, so it implies FC_ATTN_KTILE_REG and
         * is tried FIRST; it needs the same contract plus a word-aligned q4 row,
         * and it falls back to the register-K kernel (then to the shipped one)
         * rather than launching out of contract. */
        if (fc_attn_vstage_enabled()) {
            unsigned int vb = fc_attn_flash_kvs_shared_bytes(kv_groups, head_dim);
            int v_ok = shape_ok &&
                       FC_ATTN_KVQ4_ROW_BYTES(head_dim) % 4 == 0 &&
                       head_dim % FC_ATTN_KVQ4_BLK == 0;
            if (ctx->kern_attn_flash_prefill_q4_kvs && vb && v_ok) {
                kern = ctx->kern_attn_flash_prefill_q4_kvs;
                shared_bytes = vb;
                narrow = 0;
                /* The G8 specialisation is kvs at its one hot shape. */
                if (fc_attn_g8_enabled() && ctx->kern_attn_flash_prefill_q4_g8 &&
                    kv_groups == 8 && head_dim == 256) {
                    kern = ctx->kern_attn_flash_prefill_q4_g8;
                    shared_bytes = fc_attn_flash_g8_shared_bytes();
                }
            }
        }
        if (kern == ctx->kern_attn_flash_prefill_q4 &&
            ctx->kern_attn_flash_prefill_q4_kr && rb && shape_ok) {
            kern = ctx->kern_attn_flash_prefill_q4_kr;
            shared_bytes = rb;
            narrow = 0;               /* the register path supersedes fp16 staging */
        }
        static int vs_noted = 0;
        if (fc_attn_vstage_enabled() && !vs_noted) {
            vs_noted = 1;
            if (kern == ctx->kern_attn_flash_prefill_q4_g8)
                fprintf(stderr, "[fc_cuda] FC_ATTN_VSTAGE ACTIVE: shared-V "
                        "prefill attention, G8 form, %u B shared — BIT-EXACT\n",
                        shared_bytes);
            else if (kern == ctx->kern_attn_flash_prefill_q4_kvs)
                fprintf(stderr, "[fc_cuda] FC_ATTN_VSTAGE ACTIVE: shared-V "
                        "prefill attention, %u B shared — BIT-EXACT\n",
                        shared_bytes);
            else
                fprintf(stderr, "[fc_cuda] FC_ATTN_VSTAGE requested but "
                        "DECLINED (kernel %s, head_dim %d, row_bytes %d, "
                        "attn_win %d) — falling back\n",
                        ctx->kern_attn_flash_prefill_q4_kvs ? "present"
                                                            : "unavailable",
                        head_dim, FC_ATTN_KVQ4_ROW_BYTES(head_dim), attn_win);
        }
        /* Only report on FC_ATTN_KTILE_REG when the shared-V path did NOT take
         * over. With both flags set — the deployed steady state — the kvs kernel wins
         * and IS the register-K kernel plus the V tile, so the notice below would
         * otherwise print "DECLINED ... using the shipped kernel", which is wrong
         * on both counts and misleads exactly the provenance audit these banners
         * exist for. Observed once in a deployment's log on the 2026-08-11 deploy. */
        static int reg_noted = 0;
        if (fc_attn_ktile_reg_enabled() &&
            kern != ctx->kern_attn_flash_prefill_q4_kvs &&
            kern != ctx->kern_attn_flash_prefill_q4_g8 && !reg_noted) {
            reg_noted = 1;
            if (kern == ctx->kern_attn_flash_prefill_q4_kr)
                fprintf(stderr, "[fc_cuda] FC_ATTN_KTILE_REG=1 ACTIVE: "
                        "register-K prefill attention, %u B shared — BIT-EXACT"
                        "%s\n", shared_bytes,
                        fc_attn_ktile_fp16_level()
                            ? " (supersedes FC_ATTN_KTILE_FP16)" : "");
            else
                fprintf(stderr, "[fc_cuda] FC_ATTN_KTILE_REG=1 requested but "
                        "DECLINED (kernel %s, head_dim %d, attn_win %d) — "
                        "using the shipped kernel\n",
                        ctx->kern_attn_flash_prefill_q4_kr ? "present"
                                                           : "unavailable",
                        head_dim, attn_win);
        }
    }
    if (narrow) {
        CUfunction nk = narrow == 2 ? ctx->kern_attn_flash_prefill_q4_kqh
                                    : ctx->kern_attn_flash_prefill_q4_kh;
        unsigned int nb = fc_attn_flash_narrow_shared_bytes(kv_groups, head_dim,
                                                            narrow == 2);
        if (nb)
            nb += fc_attn_ktile_shared_pad();
        if (nk && nb && nb <= 48u * 1024u) {
            kern = nk;
            shared_bytes = nb;
        }
        /* A lossy path must announce itself. Once per process, and it also
         * says when the flag asked for something the module cannot give. */
        static int narrow_noted = 0;
        if (!narrow_noted) {
            narrow_noted = 1;
            fprintf(stderr, kern == ctx->kern_attn_flash_prefill_q4
                    ? "[fc_cuda] FC_ATTN_KTILE_FP16=%d requested but the "
                      "narrow prefill kernel is unavailable (shared %u B) — "
                      "using the exact kernel\n"
                    : "[fc_cuda] FC_ATTN_KTILE_FP16=%d ACTIVE: fp16-staged "
                      "prefill attention, %u B shared — LOSSY\n",
                    narrow, nb);
        }
    }

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr,
        &k_sink->dptr, &v_sink->dptr, &k_win->dptr, &v_win->dptr,
        &out->dptr,
        &kv_groups, &head_dim, &max_seq, &pos0, &sink, &win,
        &attn_sink, &attn_win, &scale
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)npos, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_prefill_q4 launch failed: %d\n",
                err);
        return -1;
    }
    return 0;
}

/* ================================================================
 * Decode-mode split flash attention (attention.ptx)
 * ================================================================ */

int fc_attn_flash_decode_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_attn_flash_decode &&
           ctx->kern_attn_flash_decode_combine;
}

int fc_attn_flash_decode_q4_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_attn_flash_decode_q4 &&
           ctx->kern_attn_flash_decode_combine;
}

/* Shared: dynamic-shared bytes for one decode/prefill block
 * (fc_attn_flash_shared_bytes) plus the argument checks common to both decode
 * variants. Returns kv_groups (>0) or -1 on a contract violation. */
static int fc_attn_decode_check(int n_head, int n_head_kv, int head_dim,
                                int max_seq, int seq_len, int n_splits,
                                int split_len, unsigned int *shared_bytes)
{
    int kv_groups = n_head / n_head_kv;
    if (head_dim % FC_ATTN_DOT_LANES != 0 || head_dim > 1024 ||
        kv_groups < 1 || kv_groups > FC_ATTN_MAX_GROUPS ||
        kv_groups * n_head_kv != n_head || seq_len < 1 || seq_len > max_seq ||
        n_splits < 1 || split_len < 1)
        return -1;
    *shared_bytes = fc_attn_flash_shared_bytes(kv_groups, head_dim);
    if (*shared_bytes == 0)
        return -1;
    return kv_groups;
}

/* Launch fc_attn_flash_decode_combine on this ctx's stream (ordered after the
 * split kernel). out: [n_head][head_dim]. */
static int fc_attn_decode_combine_launch(fc_cuda_ctx *ctx, fc_buffer *m_scr,
                                         fc_buffer *l_scr, fc_buffer *acc_scr,
                                         fc_buffer *out, int n_head,
                                         int head_dim, int n_splits)
{
    void *args[] = {
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr, &out->dptr,
        &head_dim, &n_splits
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode_combine,
        (unsigned int)n_head, 1, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_combine launch failed: "
                "%d\n", err);
        return -1;
    }
    return 0;
}

int fc_attn_flash_decode_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                             fc_buffer *k_cache, fc_buffer *v_cache,
                             fc_buffer *m_scr, fc_buffer *l_scr,
                             fc_buffer *acc_scr, fc_buffer *out, int n_head,
                             int n_head_kv, int head_dim, int max_seq,
                             int seq_len, int n_splits, int split_len,
                             int attn_sink, int attn_win, float scale)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_flash_decode ||
        !ctx->kern_attn_flash_decode_combine || !q || !k_cache || !v_cache ||
        !m_scr || !l_scr || !acc_scr || !out || n_head_kv <= 0)
        return -1;

    unsigned int shared_bytes = 0;
    int kv_groups = fc_attn_decode_check(n_head, n_head_kv, head_dim, max_seq,
                                         seq_len, n_splits, split_len,
                                         &shared_bytes);
    if (kv_groups < 0)
        return -1;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr,
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr,
        &kv_groups, &head_dim, &max_seq, &seq_len, &n_splits, &split_len,
        &attn_sink, &attn_win, &scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode,
        (unsigned int)n_splits, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode launch failed: %d\n", err);
        fc_ctx_leave();
        return -1;
    }
    int rc = fc_attn_decode_combine_launch(ctx, m_scr, l_scr, acc_scr, out,
                                           n_head, head_dim, n_splits);
    fc_ctx_leave();
    return rc;
}

int fc_attn_flash_decode_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                fc_buffer *k_cache, fc_buffer *v_cache,
                                fc_buffer *k_sink, fc_buffer *v_sink,
                                fc_buffer *k_win, fc_buffer *v_win,
                                fc_buffer *m_scr, fc_buffer *l_scr,
                                fc_buffer *acc_scr, fc_buffer *out, int n_head,
                                int n_head_kv, int head_dim, int max_seq,
                                int seq_len, int n_splits, int split_len,
                                int sink, int win, int attn_sink, int attn_win,
                                float scale)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_flash_decode_q4 ||
        !ctx->kern_attn_flash_decode_combine || !q || !k_cache || !v_cache ||
        !k_sink || !v_sink || !k_win || !v_win || !m_scr || !l_scr ||
        !acc_scr || !out || n_head_kv <= 0 || (head_dim % 32) != 0)
        return -1;

    unsigned int shared_bytes = 0;
    int kv_groups = fc_attn_decode_check(n_head, n_head_kv, head_dim, max_seq,
                                         seq_len, n_splits, split_len,
                                         &shared_bytes);
    if (kv_groups < 0)
        return -1;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr,
        &k_sink->dptr, &v_sink->dptr, &k_win->dptr, &v_win->dptr,
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr,
        &kv_groups, &head_dim, &max_seq, &seq_len, &n_splits, &split_len,
        &sink, &win, &attn_sink, &attn_win, &scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode_q4,
        (unsigned int)n_splits, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_q4 launch failed: %d\n",
                err);
        fc_ctx_leave();
        return -1;
    }
    int rc = fc_attn_decode_combine_launch(ctx, m_scr, l_scr, acc_scr, out,
                                           n_head, head_dim, n_splits);
    fc_ctx_leave();
    return rc;
}

/* ================================================================
 * Multi-user decode slot attention (L7)
 *
 * K users, one decode token each at its OWN position (a device int32
 * array — the RoPE position buffer), against per-slot KV rings laid out
 * [slot][n_head_kv][max_seq][row]. Each user's split plan is recomputed
 * on-device from its own position with the single-user formula, so the
 * per-split partials are bit-identical to fc_attn_flash_decode at that
 * position; the launch is uniform at launch_splits (>= every user's
 * plan) and off-plan blocks write empty partials the combine ignores.
 * The combine is the ordinary decode combine over nslots*n_head heads.
 * ================================================================ */

int fc_attn_flash_decode_slots_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_attn_kv_append_slots &&
           ctx->kern_attn_flash_decode_slots &&
           ctx->kern_attn_flash_decode_combine;
}

int fc_attn_flash_decode_slots_q4_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_attn_kv_append_slots_q4 &&
           ctx->kern_attn_flash_decode_slots_q4 &&
           ctx->kern_attn_flash_decode_combine;
}

void fc_attn_kv_append_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                                 fc_buffer *v_new, fc_buffer *k_cache,
                                 fc_buffer *v_cache, int n_head_kv,
                                 int head_dim, int max_seq, fc_buffer *pos,
                                 fc_buffer *rows, int nslots)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_kv_append_slots ||
        !k_new || !v_new || !k_cache || !v_cache || !pos || nslots <= 0)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    CUdeviceptr rows_d = fc_rows_dptr(rows);
    void *args[] = {
        &k_new->dptr, &v_new->dptr, &k_cache->dptr, &v_cache->dptr,
        &n_head_kv, &head_dim, &max_seq, &pos->dptr, &rows_d
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_kv_append_slots,
        (unsigned int)nslots, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] attn_kv_append_slots launch failed: %d\n",
                err);
}

void fc_attn_kv_append_slots_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                                    fc_buffer *v_new, fc_buffer *k_cache,
                                    fc_buffer *v_cache, fc_buffer *k_sink,
                                    fc_buffer *v_sink, fc_buffer *k_win,
                                    fc_buffer *v_win, int n_head_kv,
                                    int head_dim, int max_seq, fc_buffer *pos,
                                    fc_buffer *rows, int nslots, int sink,
                                    int win)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_kv_append_slots_q4 ||
        !k_new || !v_new || !k_cache || !v_cache || !k_sink || !v_sink ||
        !k_win || !v_win || !pos || nslots <= 0 || (head_dim % 32) != 0)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    CUdeviceptr rows_d = fc_rows_dptr(rows);
    void *args[] = {
        &k_new->dptr, &v_new->dptr, &k_cache->dptr, &v_cache->dptr,
        &k_sink->dptr, &v_sink->dptr, &k_win->dptr, &v_win->dptr,
        &n_head_kv, &head_dim, &max_seq, &pos->dptr, &sink, &win, &rows_d
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_kv_append_slots_q4,
        (unsigned int)nslots, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] attn_kv_append_slots_q4 launch failed: "
                "%d\n", err);
}

int fc_attn_flash_decode_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                   fc_buffer *k_cache, fc_buffer *v_cache,
                                   fc_buffer *m_scr, fc_buffer *l_scr,
                                   fc_buffer *acc_scr, fc_buffer *out,
                                   int n_head, int n_head_kv, int head_dim,
                                   int max_seq, fc_buffer *pos,
                                   fc_buffer *rows, int nslots,
                                   int launch_splits, int kps, int splits_cap,
                                   int attn_sink, int attn_win, float scale)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_flash_decode_slots ||
        !ctx->kern_attn_flash_decode_combine || !q || !k_cache || !v_cache ||
        !m_scr || !l_scr || !acc_scr || !out || !pos || nslots <= 0 ||
        n_head_kv <= 0)
        return -1;

    unsigned int shared_bytes = 0;
    int kv_groups = fc_attn_decode_check(n_head, n_head_kv, head_dim, max_seq,
                                         max_seq, launch_splits, 1,
                                         &shared_bytes);
    if (kv_groups < 0)
        return -1;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    CUdeviceptr rows_d = fc_rows_dptr(rows);
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr,
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr,
        &kv_groups, &head_dim, &max_seq, &pos->dptr, &kps, &splits_cap,
        &attn_sink, &attn_win, &scale, &rows_d
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode_slots,
        (unsigned int)launch_splits, (unsigned int)n_head_kv,
        (unsigned int)nslots,
        (unsigned int)head_dim, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_slots launch failed: "
                "%d\n", err);
        fc_ctx_leave();
        return -1;
    }
    int rc = fc_attn_decode_combine_launch(ctx, m_scr, l_scr, acc_scr, out,
                                           nslots * n_head, head_dim,
                                           launch_splits);
    fc_ctx_leave();
    return rc;
}

/* FC_ATTN_DEC_KVS (default ON; =0 restores the shipped kernel for A/B): the
 * decode slots Q4 split kernel runs as fc_attn_flash_decode_slots_q4_kvs —
 * register-K + shared-V, BIT-EXACT, same grid and partials. Latched once, so
 * a captured graph never sees the kernel change. */
static int fc_attn_dec_kvs_enabled(void)
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("FC_ATTN_DEC_KVS");
        on = (e && e[0] == '0') ? 0 : 1;
    }
    return on;
}

/* FC_ATTN_DEC_KVF (default ON since 2026-11-02; =0 turns it off): the decode slots call runs the
 * fewer-instruction kernel (fc_attn_flash_decode_slots_q4_kvf,
 * cuda_kernels/attention_decode_fast.cuh) — bit-exact with kvs, gated by
 * cuda/tests/test_attn_decode_kvf. Needs kv_groups == 8. Latched; tests and
 * benches override it with fc_attn_dec_kvf_set. */
#define FC_ATTN_KVF_GROUPS_HOST 8

static int fc_attn_dec_kvf_on = -1;

static int fc_attn_dec_kvf_enabled(void)
{
    if (fc_attn_dec_kvf_on < 0) {
        const char *e = getenv("FC_ATTN_DEC_KVF");
        fc_attn_dec_kvf_on = (e && e[0] == '0') ? 0 : 1;
    }
    return fc_attn_dec_kvf_on;
}

int fc_attn_dec_kvf_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_attn_flash_decode_slots_q4_kvf != NULL;
}

int fc_attn_dec_kvf_set(int on)
{
    int prev = fc_attn_dec_kvf_enabled();
    fc_attn_dec_kvf_on = on ? 1 : 0;
    return prev;
}

/* fc_attn_decode_combine_launch with the sigmoid output gate folded in
 * (fc_attn_flash_decode_combine_gated). z: [n_head][head_dim], read only. */
static int fc_attn_decode_combine_gated_launch(fc_cuda_ctx *ctx,
                                               fc_buffer *m_scr,
                                               fc_buffer *l_scr,
                                               fc_buffer *acc_scr,
                                               fc_buffer *z, fc_buffer *out,
                                               int n_head, int head_dim,
                                               int n_splits)
{
    void *args[] = {
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr, &z->dptr, &out->dptr,
        &head_dim, &n_splits
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode_combine_gated,
        (unsigned int)n_head, 1, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_combine_gated launch "
                "failed: %d\n", err);
        return -1;
    }
    return 0;
}

static int fc_attn_flash_decode_slots_q4_impl(fc_cuda_ctx *ctx, fc_buffer *q,
                                      fc_buffer *k_cache, fc_buffer *v_cache,
                                      fc_buffer *k_sink, fc_buffer *v_sink,
                                      fc_buffer *k_win, fc_buffer *v_win,
                                      fc_buffer *m_scr, fc_buffer *l_scr,
                                      fc_buffer *acc_scr, fc_buffer *out,
                                      int n_head, int n_head_kv, int head_dim,
                                      int max_seq, fc_buffer *pos,
                                      fc_buffer *rows, int nslots,
                                      int launch_splits, int kps,
                                      int splits_cap, int sink, int win,
                                      int attn_sink, int attn_win, float scale, fc_buffer *z)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_flash_decode_slots_q4 ||
        !ctx->kern_attn_flash_decode_combine || !q || !k_cache || !v_cache ||
        !k_sink || !v_sink || !k_win || !v_win || !m_scr || !l_scr ||
        !acc_scr || !out || !pos || nslots <= 0 || n_head_kv <= 0 ||
        (head_dim % 32) != 0)
        return -1;

    unsigned int shared_bytes = 0;
    int kv_groups = fc_attn_decode_check(n_head, n_head_kv, head_dim, max_seq,
                                         max_seq, launch_splits, 1,
                                         &shared_bytes);
    if (kv_groups < 0)
        return -1;

    /* kvs contract: a head_dim-thread block scores a full key tile and
     * stages its V rows in FC_ATTN_VS_ROUNDS word rounds (head_dim == 256). */
    CUfunction kern = ctx->kern_attn_flash_decode_slots_q4;
    if (fc_attn_dec_kvs_enabled() && ctx->kern_attn_flash_decode_slots_q4_kvs &&
        head_dim == FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES &&
        FC_ATTN_KVQ4_ROW_BYTES(head_dim) % 4 == 0) {
        unsigned int vb = fc_attn_flash_kvs_shared_bytes(kv_groups, head_dim);
        if (vb) {
            kern = ctx->kern_attn_flash_decode_slots_q4_kvs;
            shared_bytes = vb;
            if (fc_attn_dec_kvf_enabled() && ctx->kern_attn_flash_decode_slots_q4_kvf &&
                kv_groups == FC_ATTN_KVF_GROUPS_HOST)
                kern = ctx->kern_attn_flash_decode_slots_q4_kvf;
        }
    }

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    CUdeviceptr rows_d = fc_rows_dptr(rows);
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr,
        &k_sink->dptr, &v_sink->dptr, &k_win->dptr, &v_win->dptr,
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr,
        &kv_groups, &head_dim, &max_seq, &pos->dptr, &kps, &splits_cap,
        &sink, &win, &attn_sink, &attn_win, &scale, &rows_d
    };
    CUresult err = fcw_cuLaunchKernel(
        kern,
        (unsigned int)launch_splits, (unsigned int)n_head_kv,
        (unsigned int)nslots,
        (unsigned int)head_dim, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_slots_q4 launch failed: "
                "%d\n", err);
        fc_ctx_leave();
        return -1;
    }
    int rc = z ? fc_attn_decode_combine_gated_launch(ctx, m_scr, l_scr, acc_scr,
                                                    z, out, nslots * n_head,
                                                    head_dim, launch_splits)
               : fc_attn_decode_combine_launch(ctx, m_scr, l_scr, acc_scr, out,
                                               nslots * n_head, head_dim,
                                               launch_splits);
    fc_ctx_leave();
    return rc;
}

int fc_attn_flash_decode_slots_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                      fc_buffer *k_cache, fc_buffer *v_cache,
                                      fc_buffer *k_sink, fc_buffer *v_sink,
                                      fc_buffer *k_win, fc_buffer *v_win,
                                      fc_buffer *m_scr, fc_buffer *l_scr,
                                      fc_buffer *acc_scr, fc_buffer *out,
                                      int n_head, int n_head_kv, int head_dim,
                                      int max_seq, fc_buffer *pos,
                                      fc_buffer *rows, int nslots,
                                      int launch_splits, int kps,
                                      int splits_cap, int sink, int win,
                                      int attn_sink, int attn_win, float scale)
{
    return fc_attn_flash_decode_slots_q4_impl(
        ctx, q, k_cache, v_cache, k_sink, v_sink, k_win, v_win, m_scr, l_scr,
        acc_scr, out, n_head, n_head_kv, head_dim, max_seq, pos, rows, nslots,
        launch_splits, kps, splits_cap, sink, win, attn_sink, attn_win, scale,
        NULL);
}

/* fc_attn_flash_decode_slots_q4_ctx with the sigmoid output gate folded into
 * the combine: out = attention * sigmoid(z). -1 if the gated combine kernel is
 * absent (caller runs the ungated call + fc_sigmoid + fc_mul). */
int fc_attn_flash_decode_slots_q4_gated_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                      fc_buffer *k_cache, fc_buffer *v_cache,
                                      fc_buffer *k_sink, fc_buffer *v_sink,
                                      fc_buffer *k_win, fc_buffer *v_win,
                                      fc_buffer *m_scr, fc_buffer *l_scr,
                                      fc_buffer *acc_scr, fc_buffer *out,
                                      int n_head, int n_head_kv, int head_dim,
                                      int max_seq, fc_buffer *pos,
                                      fc_buffer *rows, int nslots,
                                      int launch_splits, int kps,
                                      int splits_cap, int sink, int win,
                                      int attn_sink, int attn_win, float scale,
                                      fc_buffer *z)
{
    if (!ctx || !ctx->kern_attn_flash_decode_combine_gated || !z)
        return -1;
    return fc_attn_flash_decode_slots_q4_impl(
        ctx, q, k_cache, v_cache, k_sink, v_sink, k_win, v_win, m_scr, l_scr,
        acc_scr, out, n_head, n_head_kv, head_dim, max_seq, pos, rows, nslots,
        launch_splits, kps, splits_cap, sink, win, attn_sink, attn_win, scale,
        z);
}

/* ================================================================
 * Speculative-verify decode attention (fc_attn_flash_decode_rows_q4_kvs)
 * ================================================================ */

/* Dynamic-shared bytes for one rows block carrying `rows` rows: the q set,
 * scores + probabilities and m/l/corr per (row, group), plus the staged V tile.
 * MUST match the layout in cuda_kernels/attention_decode_rows.cuh. 48 864 B at
 * 5 rows x 8 groups x head_dim 256 — inside sm_37's 48 KB. */
static unsigned int fc_attn_flash_rows_shared_bytes(int rows, int kv_groups,
                                                    int head_dim)
{
    size_t rg = (size_t)rows * kv_groups;
    size_t bytes = (rg * head_dim + 2u * rg * FC_ATTN_TILE_KEYS + 3u * rg) *
                   sizeof(float) +
                   (size_t)FC_ATTN_TILE_KEYS * FC_ATTN_KVQ4_ROW_BYTES(head_dim);
    if (bytes > 48u * 1024u)
        return 0;
    return (unsigned int)bytes;
}

/* Rows per block (FC_ATTN_ROWS_PER_BLOCK, default 1). Latched. Measured on the
 * K80 (Qwen3.6 verify, 2026-10-22): one row per block is fastest or tied at
 * every R and depth — the split kernel is latency-bound, so 5x the blocks beats
 * reading each K/V tile once for 5 rows (rpb 5: 1-row verify 57.2 ms at 8k,
 * rpb 1: 49.1). */
static int fc_attn_rows_per_block(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("FC_ATTN_ROWS_PER_BLOCK");
        v = e ? atoi(e) : 1;
        if (v < 1) v = 1;
        if (v > FC_ATTN_ROWS_MAX) v = FC_ATTN_ROWS_MAX;
    }
    return v;
}

/* Host copy of fc_attn_dec_plan (attention.cu): n_splits, and split_len out. */
static int fc_attn_rows_plan(int seq_len, int kps, int cap, int *split_len)
{
    int n_splits = (seq_len + kps - 1) / kps;
    if (n_splits < 1)
        n_splits = 1;
    if (n_splits > cap)
        n_splits = cap;
    int sl = (seq_len + n_splits - 1) / n_splits;
    sl = (sl + 15) & ~15;
    *split_len = sl;
    return (seq_len + sl - 1) / sl;
}

int fc_attn_flash_decode_rows_q4_available_ctx(fc_cuda_ctx *ctx)
{
    if (!fc_cuda_ctx_available(ctx))
        return 0;
    for (int i = 0; i < FC_ATTN_ROWS_MAX; i++)
        if (!ctx->kern_attn_flash_decode_rows_q4_kvs[i])
            return 0;
    return ctx->kern_attn_flash_decode_combine &&
           ctx->kern_attn_flash_decode_combine_gated;
}

/* Launch the rows kernel for rows [r_lo, r_hi), which share split_len. */
static int fc_attn_rows_launch(fc_cuda_ctx *ctx, void **bufs, int kv_groups,
                               int head_dim, int max_seq, int n_head_kv,
                               int pos0, int r_lo, int r_hi, int split_len,
                               int launch_splits, int sink, int win,
                               float scale)
{
    int nrows = r_hi - r_lo;
    int rpb = fc_attn_rows_per_block();
    if (rpb > nrows)
        rpb = nrows;
    unsigned int shared = fc_attn_flash_rows_shared_bytes(rpb, kv_groups,
                                                          head_dim);
    if (shared == 0)
        return -1;
    int pos_lo = pos0 + r_lo;
    void *args[] = {
        bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], bufs[5], bufs[6],
        bufs[7], bufs[8], bufs[9],
        &kv_groups, &head_dim, &max_seq, &pos_lo, &r_lo, &nrows,
        &split_len, &sink, &win, &scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode_rows_q4_kvs[rpb - 1],
        (unsigned int)launch_splits, (unsigned int)n_head_kv,
        (unsigned int)((nrows + rpb - 1) / rpb),
        (unsigned int)head_dim, 1, 1, shared, ctx->stream, args, NULL);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_rows_q4 launch failed: "
                "%d\n", err);
        return -1;
    }
    return 0;
}

int fc_attn_flash_decode_rows_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                     fc_buffer *k_cache, fc_buffer *v_cache,
                                     fc_buffer *k_sink, fc_buffer *v_sink,
                                     fc_buffer *k_win, fc_buffer *v_win,
                                     fc_buffer *m_scr, fc_buffer *l_scr,
                                     fc_buffer *acc_scr, fc_buffer *z,
                                     fc_buffer *out, int n_head, int n_head_kv,
                                     int head_dim, int max_seq, int pos0,
                                     int nrows, int kps, int splits_cap,
                                     int sink, int win, int attn_sink,
                                     int attn_win, float scale)
{
    (void)attn_sink;   /* full attention only: attn_win <= 0 attends all */
    if (!fc_attn_flash_decode_rows_q4_available_ctx(ctx) || !q || !k_cache ||
        !v_cache || !k_sink || !v_sink || !k_win || !v_win || !m_scr ||
        !l_scr || !acc_scr || !out || n_head_kv <= 0 || nrows < 1 ||
        nrows > FC_ATTN_ROWS_MAX || pos0 < 0 || pos0 + nrows > max_seq ||
        attn_win > 0 || kps < 1 || splits_cap < 1 ||
        head_dim != FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES ||
        FC_ATTN_KVQ4_ROW_BYTES(head_dim) % 4 != 0)
        return -1;
    int kv_groups = n_head / n_head_kv;
    if (kv_groups < 1 || kv_groups > FC_ATTN_MAX_GROUPS ||
        kv_groups * n_head_kv != n_head)
        return -1;

    int split_len[FC_ATTN_ROWS_MAX];
    int launch_splits = 1;
    for (int r = 0; r < nrows; r++) {
        int ns = fc_attn_rows_plan(pos0 + r + 1, kps, splits_cap, &split_len[r]);
        if (ns > launch_splits)
            launch_splits = ns;
    }

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *bufs[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr, &k_sink->dptr,
        &v_sink->dptr, &k_win->dptr, &v_win->dptr, &m_scr->dptr,
        &l_scr->dptr, &acc_scr->dptr
    };
    /* One launch per run of rows sharing a split_len (almost always one). */
    int rc = 0;
    for (int r_lo = 0; r_lo < nrows && rc == 0;) {
        int r_hi = r_lo + 1;
        while (r_hi < nrows && split_len[r_hi] == split_len[r_lo])
            r_hi++;
        rc = fc_attn_rows_launch(ctx, bufs, kv_groups, head_dim, max_seq,
                                 n_head_kv, pos0, r_lo, r_hi, split_len[r_lo],
                                 launch_splits, sink, win, scale);
        r_lo = r_hi;
    }
    if (rc == 0)
        rc = z ? fc_attn_decode_combine_gated_launch(ctx, m_scr, l_scr,
                                                     acc_scr, z, out,
                                                     nrows * n_head, head_dim,
                                                     launch_splits)
               : fc_attn_decode_combine_launch(ctx, m_scr, l_scr, acc_scr,
                                               out, nrows * n_head, head_dim,
                                               launch_splits);
    fc_ctx_leave();
    return rc;
}

int fc_attn_flash_decode_rows_q4_dev_available_ctx(fc_cuda_ctx *ctx)
{
    return fc_cuda_ctx_available(ctx) && ctx->kern_attn_flash_decode_rows_q4_dev &&
           ctx->kern_attn_flash_decode_combine &&
           ctx->kern_attn_flash_decode_combine_gated;
}

int fc_attn_flash_decode_rows_q4_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                         fc_buffer *k_cache, fc_buffer *v_cache,
                                         fc_buffer *k_sink, fc_buffer *v_sink,
                                         fc_buffer *k_win, fc_buffer *v_win,
                                         fc_buffer *m_scr, fc_buffer *l_scr,
                                         fc_buffer *acc_scr, fc_buffer *z,
                                         fc_buffer *out, int n_head, int n_head_kv,
                                         int head_dim, int max_seq, fc_buffer *pos,
                                         int nrows, int launch_splits, int kps,
                                         int splits_cap, int sink, int win,
                                         float scale)
{
    if (!fc_attn_flash_decode_rows_q4_dev_available_ctx(ctx) || !q || !k_cache ||
        !v_cache || !k_sink || !v_sink || !k_win || !v_win || !m_scr || !l_scr ||
        !acc_scr || !out || !pos || n_head_kv <= 0 || nrows < 1 ||
        nrows > FC_ATTN_ROWS_MAX || kps < 1 || splits_cap < 1 || launch_splits < 1 ||
        launch_splits > splits_cap ||
        head_dim != FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES ||
        FC_ATTN_KVQ4_ROW_BYTES(head_dim) % 4 != 0)
        return -1;
    int kv_groups = n_head / n_head_kv;
    if (kv_groups < 1 || kv_groups > FC_ATTN_MAX_GROUPS ||
        kv_groups * n_head_kv != n_head)
        return -1;
    unsigned int shared = fc_attn_flash_rows_shared_bytes(1, kv_groups, head_dim);
    /* FC_ATTN_ROWS_PSM=0 keeps the serial-expf softmax (the A/B arm); same bytes. */
    static int psm = -1;
    if (psm < 0) {
        const char *e = getenv("FC_ATTN_ROWS_PSM");
        psm = (e && e[0] == '0') ? 0 : 1;
    }
    CUfunction kern = (psm && ctx->kern_attn_flash_decode_rows_q4_dev_ps)
                          ? ctx->kern_attn_flash_decode_rows_q4_dev_ps
                          : ctx->kern_attn_flash_decode_rows_q4_dev;
    if (kern == ctx->kern_attn_flash_decode_rows_q4_dev_ps)
        shared += (unsigned int)(kv_groups * sizeof(float));
    if (shared == 0 || fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr, &k_sink->dptr, &v_sink->dptr,
        &k_win->dptr, &v_win->dptr, &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr,
        &kv_groups, &head_dim, &max_seq, &pos->dptr, &kps, &splits_cap, &sink,
        &win, &scale
    };
    CUresult err = fcw_cuLaunchKernel(kern,
                                    (unsigned int)launch_splits, (unsigned int)n_head_kv,
                                    (unsigned int)nrows, (unsigned int)head_dim, 1, 1,
                                    shared, ctx->stream, args, NULL);
    int rc = 0;
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_rows_q4_dev launch failed: %d\n", err);
        rc = -1;
    }
    if (rc == 0)
        rc = z ? fc_attn_decode_combine_gated_launch(ctx, m_scr, l_scr, acc_scr, z, out,
                                                     nrows * n_head, head_dim, launch_splits)
               : fc_attn_decode_combine_launch(ctx, m_scr, l_scr, acc_scr, out,
                                               nrows * n_head, head_dim, launch_splits);
    fc_ctx_leave();
    return rc;
}

/* ================================================================
 * Device-position decode variants (CUDA-graph capture)
 *
 * The position is a device int32 buffer (pos[0]); the split plan is
 * recomputed on-device from kps/splits_cap with the caller's exact
 * formula. The decode grid is FIXED at splits_cap blocks so the
 * recorded launch replays at any position; dead blocks exit early.
 * See fc_cuda.h.
 * ================================================================ */

int fc_attn_decode_graph_available_ctx(fc_cuda_ctx *ctx, int q4)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_rope_pos ||
        !ctx->kern_attn_qz_split || !ctx->kern_qk_rms_norm ||
        !ctx->kern_sigmoid || !ctx->kern_mul ||
        !ctx->kern_attn_flash_decode_combine_dev)
        return 0;
    if (q4)
        return (ctx->kern_attn_kv_append_q4_dev &&
                ctx->kern_attn_flash_decode_q4_dev) ? 1 : 0;
    return (ctx->kern_attn_kv_append_dev &&
            ctx->kern_attn_flash_decode_dev) ? 1 : 0;
}

void fc_attn_kv_append_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                               fc_buffer *v_new, fc_buffer *k_cache,
                               fc_buffer *v_cache, int n_head_kv,
                               int head_dim, int max_seq, fc_buffer *pos)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_kv_append_dev ||
        !k_new || !v_new || !k_cache || !v_cache || !pos)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &k_new->dptr, &v_new->dptr, &k_cache->dptr, &v_cache->dptr,
        &n_head_kv, &head_dim, &max_seq, &pos->dptr
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_kv_append_dev,
        1u, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] attn_kv_append_dev launch failed: %d\n",
                err);
}

void fc_attn_kv_append_q4_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                                  fc_buffer *v_new, fc_buffer *k_cache,
                                  fc_buffer *v_cache, fc_buffer *k_sink,
                                  fc_buffer *v_sink, fc_buffer *k_win,
                                  fc_buffer *v_win, int n_head_kv,
                                  int head_dim, int max_seq, fc_buffer *pos,
                                  int sink, int win)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_kv_append_q4_dev ||
        !k_new || !v_new || !k_cache || !v_cache || !k_sink || !v_sink ||
        !k_win || !v_win || !pos || (head_dim % 32) != 0)
        return;

    if (fc_ctx_enter(ctx) != 0)
        return;
    void *args[] = {
        &k_new->dptr, &v_new->dptr, &k_cache->dptr, &v_cache->dptr,
        &k_sink->dptr, &v_sink->dptr, &k_win->dptr, &v_win->dptr,
        &n_head_kv, &head_dim, &max_seq, &pos->dptr, &sink, &win
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_kv_append_q4_dev,
        1u, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    fc_ctx_leave();
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] attn_kv_append_q4_dev launch failed: %d\n",
                err);
}

/* Launch fc_attn_flash_decode_combine_dev (stream-ordered after the split
 * kernel). */
static int fc_attn_decode_combine_dev_launch(fc_cuda_ctx *ctx,
                                             fc_buffer *m_scr,
                                             fc_buffer *l_scr,
                                             fc_buffer *acc_scr,
                                             fc_buffer *out, int n_head,
                                             int head_dim, fc_buffer *pos,
                                             int kps, int splits_cap)
{
    void *args[] = {
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr, &out->dptr,
        &head_dim, &pos->dptr, &kps, &splits_cap
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode_combine_dev,
        (unsigned int)n_head, 1, 1,
        (unsigned int)head_dim, 1, 1,
        0, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_combine_dev launch "
                "failed: %d\n", err);
        return -1;
    }
    return 0;
}

int fc_attn_flash_decode_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                 fc_buffer *k_cache, fc_buffer *v_cache,
                                 fc_buffer *m_scr, fc_buffer *l_scr,
                                 fc_buffer *acc_scr, fc_buffer *out,
                                 int n_head, int n_head_kv, int head_dim,
                                 int max_seq, fc_buffer *pos, int kps,
                                 int splits_cap, int attn_sink, int attn_win,
                                 float scale)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_flash_decode_dev ||
        !ctx->kern_attn_flash_decode_combine_dev || !q || !k_cache ||
        !v_cache || !m_scr || !l_scr || !acc_scr || !out || !pos ||
        n_head_kv <= 0 || kps < 1 || splits_cap < 1)
        return -1;

    /* Same contract checks as the host-arg variant; seq_len/n_splits/
     * split_len live on the device, so validate the static caps only. */
    unsigned int shared_bytes = 0;
    int kv_groups = fc_attn_decode_check(n_head, n_head_kv, head_dim, max_seq,
                                         1, 1, 1, &shared_bytes);
    if (kv_groups < 0)
        return -1;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr,
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr,
        &kv_groups, &head_dim, &max_seq, &pos->dptr, &kps, &splits_cap,
        &attn_sink, &attn_win, &scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode_dev,
        (unsigned int)splits_cap, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_dev launch failed: %d\n",
                err);
        fc_ctx_leave();
        return -1;
    }
    int rc = fc_attn_decode_combine_dev_launch(ctx, m_scr, l_scr, acc_scr,
                                               out, n_head, head_dim, pos,
                                               kps, splits_cap);
    fc_ctx_leave();
    return rc;
}

int fc_attn_flash_decode_q4_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                    fc_buffer *k_cache, fc_buffer *v_cache,
                                    fc_buffer *k_sink, fc_buffer *v_sink,
                                    fc_buffer *k_win, fc_buffer *v_win,
                                    fc_buffer *m_scr, fc_buffer *l_scr,
                                    fc_buffer *acc_scr, fc_buffer *out,
                                    int n_head, int n_head_kv, int head_dim,
                                    int max_seq, fc_buffer *pos, int kps,
                                    int splits_cap, int sink, int win,
                                    int attn_sink, int attn_win, float scale)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->kern_attn_flash_decode_q4_dev ||
        !ctx->kern_attn_flash_decode_combine_dev || !q || !k_cache ||
        !v_cache || !k_sink || !v_sink || !k_win || !v_win || !m_scr ||
        !l_scr || !acc_scr || !out || !pos || n_head_kv <= 0 ||
        (head_dim % 32) != 0 || kps < 1 || splits_cap < 1)
        return -1;

    unsigned int shared_bytes = 0;
    int kv_groups = fc_attn_decode_check(n_head, n_head_kv, head_dim, max_seq,
                                         1, 1, 1, &shared_bytes);
    if (kv_groups < 0)
        return -1;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    void *args[] = {
        &q->dptr, &k_cache->dptr, &v_cache->dptr,
        &k_sink->dptr, &v_sink->dptr, &k_win->dptr, &v_win->dptr,
        &m_scr->dptr, &l_scr->dptr, &acc_scr->dptr,
        &kv_groups, &head_dim, &max_seq, &pos->dptr, &kps, &splits_cap,
        &sink, &win, &attn_sink, &attn_win, &scale
    };
    CUresult err = fcw_cuLaunchKernel(
        ctx->kern_attn_flash_decode_q4_dev,
        (unsigned int)splits_cap, (unsigned int)n_head_kv, 1,
        (unsigned int)head_dim, 1, 1,
        shared_bytes, ctx->stream, args, NULL
    );
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] attn_flash_decode_q4_dev launch failed: "
                "%d\n", err);
        fc_ctx_leave();
        return -1;
    }
    int rc = fc_attn_decode_combine_dev_launch(ctx, m_scr, l_scr, acc_scr,
                                               out, n_head, head_dim, pos,
                                               kps, splits_cap);
    fc_ctx_leave();
    return rc;
}

void fc_add(fc_buffer *a, fc_buffer *b, fc_buffer *out, int n)
{
    fc_add_ctx(_fc_default_ctx, a, b, out, n);
}

void fc_mul(fc_buffer *a, fc_buffer *b, fc_buffer *out, int n)
{
    fc_mul_ctx(_fc_default_ctx, a, b, out, n);
}

void fc_silu(fc_buffer *x, fc_buffer *out, int n)
{
    fc_silu_ctx(_fc_default_ctx, x, out, n);
}

void fc_rms_norm(fc_buffer *x, fc_buffer *weight, fc_buffer *out,
                 int n, float eps)
{
    fc_rms_norm_ctx(_fc_default_ctx, x, weight, out, n, eps);
}

void fc_softmax(fc_buffer *x, fc_buffer *out, int rows, int cols)
{
    fc_softmax_ctx(_fc_default_ctx, x, out, rows, cols);
}

void fc_matvec_q8(fc_buffer *x_q, fc_buffer *w, fc_buffer *w_scales,
                  fc_buffer *out, float x_scale, int rows, int cols)
{
    fc_matvec_q8_ctx(_fc_default_ctx, x_q, w, w_scales, out,
                     x_scale, rows, cols);
}

void fc_matvec_q3k(fc_buffer *w, fc_buffer *x, fc_buffer *out,
                   int rows, int cols, int row_stride)
{
    fc_matvec_q3k_ctx(_fc_default_ctx, w, x, out, rows, cols, row_stride);
}

void fc_rope_neox_partial(fc_buffer *data, int n_heads, int head_dim,
                          int rope_dim, int pos, float theta_base)
{
    fc_rope_neox_partial_ctx(_fc_default_ctx, data, n_heads, head_dim,
                             rope_dim, pos, theta_base);
}

void fc_qk_rms_norm(fc_buffer *data, fc_buffer *weight,
                    int n_heads, int head_dim, float eps)
{
    fc_qk_rms_norm_ctx(_fc_default_ctx, data, weight, n_heads, head_dim, eps);
}

void fc_moe_router_topk(fc_buffer *logits, fc_buffer *top_idx,
                        fc_buffer *top_wt, int n_expert, int k,
                        float expert_weights_scale)
{
    fc_moe_router_topk_ctx(_fc_default_ctx, logits, top_idx, top_wt,
                           n_expert, k, expert_weights_scale);
}

void fc_moe_weighted_add(fc_buffer *expert_out, fc_buffer *moe_out,
                         float weight, int n_embd)
{
    fc_moe_weighted_add_ctx(_fc_default_ctx, expert_out, moe_out,
                            weight, n_embd);
}

void fc_moe_shared_gate(fc_buffer *shexp_out, float gate_logit, int n_embd)
{
    fc_moe_shared_gate_ctx(_fc_default_ctx, shexp_out, gate_logit, n_embd);
}

void fc_ssm_recurrence(fc_buffer *s, fc_buffer *q, fc_buffer *k, fc_buffer *v,
                       fc_buffer *g, fc_buffer *beta, fc_buffer *y,
                       int n_v_heads, int state_size, float scale)
{
    fc_ssm_recurrence_ctx(_fc_default_ctx, s, q, k, v, g, beta, y,
                          n_v_heads, state_size, scale);
}

void fc_ssm_conv1d_silu(fc_buffer *qkv, fc_buffer *weight, fc_buffer *conv_state,
                        fc_buffer *out, int conv_dim, int conv_kernel)
{
    fc_ssm_conv1d_silu_ctx(_fc_default_ctx, qkv, weight, conv_state, out,
                           conv_dim, conv_kernel);
}

void fc_ssm_gated_norm(fc_buffer *y, fc_buffer *weight, fc_buffer *z,
                       fc_buffer *gated, int n_v_heads, int state_size, float eps)
{
    fc_ssm_gated_norm_ctx(_fc_default_ctx, y, weight, z, gated,
                          n_v_heads, state_size, eps);
}

/* ================================================================
 * CUDA graph capture / replay (per-context)
 *
 * See fc_cuda.h for the capture contract. The graph template is
 * destroyed right after instantiation — only the executable graph
 * (CUgraphExec) is kept for replay.
 * ================================================================ */

struct fc_graph {
    CUgraphExec  exec;
    fc_cuda_ctx *owner;
    size_t       nodes_by_type[FC_GRAPH_NODE_TYPES];   /* census taken at instantiation */
};

/* Count the template's nodes by CUgraphNodeType (0 kernel, 1 memcpy, 2 memset,
 * 3 host, 4 child graph, 5 empty, 6 event wait, 7 event record); best-effort. */
static void fc_graph_census(CUgraph graph, size_t *by_type)
{
    size_t n = 0;
    if (!p_cuGraphGetNodes || !p_cuGraphNodeGetType ||
        p_cuGraphGetNodes(graph, NULL, &n) != CUDA_SUCCESS || n == 0)
        return;
    CUgraphNode *nodes = malloc(n * sizeof(*nodes));
    if (nodes && p_cuGraphGetNodes(graph, nodes, &n) == CUDA_SUCCESS) {
        for (size_t i = 0; i < n; i++) {
            int t = -1;
            if (p_cuGraphNodeGetType(nodes[i], &t) == CUDA_SUCCESS)
                by_type[t >= 0 && t < FC_GRAPH_NODE_TYPES ? t : FC_GRAPH_NODE_TYPES - 1]++;
        }
    }
    free(nodes);
}

int fc_graph_begin_ctx(fc_cuda_ctx *ctx)
{
    if (!fc_cuda_ctx_available(ctx) || !ctx->stream ||
        !p_cuStreamBeginCapture || !p_cuStreamEndCapture ||
        !p_cuGraphInstantiate || !p_cuGraphLaunch ||
        !p_cuGraphDestroy || !p_cuGraphExecDestroy)
        return -1;

    if (fc_ctx_enter(ctx) != 0)
        return -1;
    /* Mode 1 = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL.
     *
     * NOT mode 0 (GLOBAL). Under GLOBAL the driver polices the WHOLE PROCESS
     * for the duration of the capture: any other host thread that calls a
     * "potentially unsafe" API (cuMemAlloc, cuMemFree, cuModuleLoad/Unload,
     * cuStreamSynchronize, ...) gets CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
     * (900) *and* invalidates this capture, so our own next launch returns
     * CUDA_ERROR_STREAM_CAPTURE_INVALIDATED (901). It does this even when the
     * other thread is driving a DIFFERENT die behind a DIFFERENT context —
     * the dies are independent, the driver's global capture guard is not.
     *
     * A serve is not single-threaded: one driver thread per batched model,
     * one connection thread per HTTP request, and /admin/models/{load,unload}
     * allocating and freeing device memory, none of which share a GPU-wide
     * lock across models. So GLOBAL turned any second model's ordinary work
     * into a capture failure on the first model's die, silently costing the
     * graph speedup on exactly the multi-user path serving depends on.
     *
     * THREAD_LOCAL keeps the safety check where it belongs — on the thread
     * that is actually capturing — and leaves every other thread alone.
     * Measured on driver 470.256.02 (K80): GLOBAL fails a concurrent
     * two-context workload, THREAD_LOCAL runs it clean. Falls back to GLOBAL
     * only if the driver rejects the mode outright. */
    CUresult err = p_cuStreamBeginCapture(ctx->stream,
                                          CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
    if (err == CUDA_ERROR_INVALID_VALUE)
        err = p_cuStreamBeginCapture(ctx->stream,
                                     CU_STREAM_CAPTURE_MODE_GLOBAL);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuStreamBeginCapture failed: %d\n", err);
        return -1;
    }
    return 0;
}

fc_graph *fc_graph_end_ctx(fc_cuda_ctx *ctx)
{
    CUgraph graph = NULL;
    CUgraphExec exec = NULL;
    fc_graph *g = NULL;
    CUresult err;

    if (!fc_cuda_ctx_available(ctx) || !ctx->stream || !p_cuStreamEndCapture)
        return NULL;

    if (fc_ctx_enter(ctx) != 0)
        return NULL;
    err = p_cuStreamEndCapture(ctx->stream, &graph);
    if (err != CUDA_SUCCESS || !graph) {
        fprintf(stderr, "[fc_cuda] cuStreamEndCapture failed: %d\n", err);
        goto out;
    }
    size_t census[FC_GRAPH_NODE_TYPES] = { 0 };
    fc_graph_census(graph, census);
    err = p_cuGraphInstantiate(&exec, graph, NULL, NULL, 0);
    p_cuGraphDestroy(graph);
    if (err != CUDA_SUCCESS || !exec) {
        fprintf(stderr, "[fc_cuda] cuGraphInstantiate failed: %d\n", err);
        goto out;
    }
    g = calloc(1, sizeof(*g));
    if (!g) {
        p_cuGraphExecDestroy(exec);
        goto out;
    }
    g->exec = exec;
    g->owner = ctx;
    memcpy(g->nodes_by_type, census, sizeof(census));
out:
    fc_ctx_leave();
    return g;
}

int fc_graph_launch(fc_graph *g)
{
    if (!g || !g->exec || !fc_cuda_ctx_available(g->owner))
        return -1;

    if (fc_ctx_enter(g->owner) != 0)
        return -1;
    CUresult err = fcw_cuGraphLaunch(g->exec, g->owner->stream);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuGraphLaunch failed: %d\n", err);
        return -1;
    }
    return 0;
}

size_t fc_graph_node_count(const fc_graph *g, int type)
{
    size_t n = 0;
    if (!g)
        return 0;
    if (type >= 0 && type < FC_GRAPH_NODE_TYPES)
        return g->nodes_by_type[type];
    for (int t = 0; t < FC_GRAPH_NODE_TYPES; t++)
        n += g->nodes_by_type[t];
    return n;
}

void fc_graph_free(fc_graph *g)
{
    if (!g)
        return;
    if (g->exec && g->owner && g->owner->context &&
        fc_ctx_enter(g->owner) == 0) {
        p_cuGraphExecDestroy(g->exec);
        fc_ctx_leave();
    }
    free(g);
}

/* How a host thread waits for the GPU (FC_CUDA_WAIT, read once):
 *   unset / "driver": cuStreamSynchronize / cuEventSynchronize, which follow the context's
 *                     scheduling flag (blocking sync by default: the thread sleeps until the
 *                     device interrupt, ~0.1 ms wake on INTx hosts);
 *   "spin":           poll cuEventQuery without pause (lowest latency, holds a core);
 *   "yield":          poll with sched_yield between queries;
 *   "hybrid[:us]":    spin up to `us` microseconds (default 200), then the driver wait;
 *   "sleep[:us]":     poll with a `us` microsecond sleep between queries (default 20).
 * Scheduling only: every mode waits for the same completion point, so results are
 * identical. */
typedef enum { FC_WAIT_DRIVER, FC_WAIT_SPIN, FC_WAIT_YIELD, FC_WAIT_HYBRID, FC_WAIT_SLEEP } fc_wait_kind;

typedef struct {
    fc_wait_kind kind;
    unsigned int us;   /* hybrid: spin budget; sleep: pause between polls */
} fc_wait_mode;

static fc_wait_mode fc_wait_mode_parse(const char *e)
{
    fc_wait_mode m = { FC_WAIT_DRIVER, 0 };
    if (!e || !e[0] || strcmp(e, "driver") == 0)
        return m;
    if (strcmp(e, "spin") == 0) {
        m.kind = FC_WAIT_SPIN;
    } else if (strcmp(e, "yield") == 0) {
        m.kind = FC_WAIT_YIELD;
    } else if (strncmp(e, "hybrid", 6) == 0 && (e[6] == 0 || e[6] == ':')) {
        m.kind = FC_WAIT_HYBRID;
        m.us = e[6] == ':' ? (unsigned int)strtoul(e + 7, NULL, 10) : 200u;
    } else if (strncmp(e, "sleep", 5) == 0 && (e[5] == 0 || e[5] == ':')) {
        m.kind = FC_WAIT_SLEEP;
        m.us = e[5] == ':' ? (unsigned int)strtoul(e + 6, NULL, 10) : 20u;
        if (m.us == 0)
            m.us = 1;
    } else {
        fprintf(stderr, "[fc_cuda] FC_CUDA_WAIT=%s not known; using the driver wait\n", e);
    }
    return m;
}

static fc_wait_mode fc_wait_mode_get(void)
{
    static int latched;
    static fc_wait_mode mode;
    if (!__atomic_load_n(&latched, __ATOMIC_ACQUIRE)) {
        mode = fc_wait_mode_parse(getenv("FC_CUDA_WAIT"));
        __atomic_store_n(&latched, 1, __ATOMIC_RELEASE);
    }
    return mode;
}

static double fc_wait_now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec * 1e-3;
}

/* Wait for `ev` (current context entered) by polling, per `m`. Returns the driver status:
 * CUDA_SUCCESS once complete, the first error cuEventQuery reports, or (hybrid, budget
 * spent) CUDA_ERROR_NOT_READY: the caller then waits the driver's way, which honours the
 * context's blocking-sync flag (an event sync on a plain event would spin). */
static CUresult fc_wait_event_polled(CUevent ev, fc_wait_mode m)
{
    const double t0 = m.kind == FC_WAIT_HYBRID ? fc_wait_now_us() : 0.0;
    for (;;) {
        CUresult err = p_cuEventQuery(ev);
        if (err != CUDA_ERROR_NOT_READY)
            return err;
        if (m.kind == FC_WAIT_YIELD) {
            sched_yield();
        } else if (m.kind == FC_WAIT_SLEEP) {
            struct timespec d = { 0, (long)m.us * 1000L };
            nanosleep(&d, NULL);
        } else if (m.kind == FC_WAIT_HYBRID && fc_wait_now_us() - t0 >= (double)m.us) {
            return CUDA_ERROR_NOT_READY;
        }
    }
}

/* ================================================================
 * CUDA events — non-blocking completion (per-context)
 *
 * The async serve scheduler (C2) records an event after a card's
 * step is launched and POLLS it instead of blocking in
 * cuStreamSynchronize (which spins/yields a whole host core per
 * waiting thread — with 8 cards on a 2-core K80 host the spinning
 * waiters starve each other's launch turns and the GPUs serialize).
 * cuEventQuery never blocks, so a waiter costs ~nothing between polls
 * and every card's stream stays in flight simultaneously.
 * ================================================================ */

struct fc_event {
    CUevent      ev;
    fc_cuda_ctx *owner;
};

fc_event *fc_event_create_ctx(fc_cuda_ctx *ctx)
{
    if (!fc_cuda_ctx_available(ctx) ||
        !p_cuEventCreate || !p_cuEventRecord || !p_cuEventQuery ||
        !p_cuEventDestroy)
        return NULL;

    fc_event *e = calloc(1, sizeof(*e));
    if (!e)
        return NULL;
    if (fc_ctx_enter(ctx) != 0) {
        free(e);
        return NULL;
    }
    CUresult err = p_cuEventCreate(&e->ev, CU_EVENT_DISABLE_TIMING);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuEventCreate failed: %d\n", err);
        free(e);
        return NULL;
    }
    e->owner = ctx;
    return e;
}

int fc_event_record_ctx(fc_cuda_ctx *ctx, fc_event *ev)
{
    if (!ev || !ev->ev || ev->owner != ctx || !fc_cuda_ctx_available(ctx))
        return -1;
    if (fc_ctx_enter(ctx) != 0)
        return -1;
    CUresult err = p_cuEventRecord(ev->ev, ctx->stream);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuEventRecord failed: %d\n", err);
        return -1;
    }
    return 0;
}

int fc_event_query(fc_event *ev)
{
    if (!ev || !ev->ev || !fc_cuda_ctx_available(ev->owner))
        return -1;
    if (fc_ctx_enter(ev->owner) != 0)
        return -1;
    CUresult err = p_cuEventQuery(ev->ev);
    fc_ctx_leave();
    if (err == CUDA_SUCCESS)
        return 0;
    if (err == CUDA_ERROR_NOT_READY)
        return 1;
    fprintf(stderr, "[fc_cuda] cuEventQuery failed: %d\n", err);
    return -1;
}

int fc_event_sync(fc_event *ev)
{
    if (!ev || !ev->ev || !p_cuEventSynchronize ||
        !fc_cuda_ctx_available(ev->owner))
        return -1;
    if (fc_ctx_enter(ev->owner) != 0)
        return -1;
    const fc_wait_mode m = fc_wait_mode_get();
    CUresult err;
    if (m.kind != FC_WAIT_DRIVER) {
        FC_CENSUS_ADD(syncs, 1);
        err = fc_wait_event_polled(ev->ev, m);
        if (err == CUDA_ERROR_NOT_READY)
            err = p_cuEventSynchronize(ev->ev);
    } else {
        err = fcw_cuEventSynchronize(ev->ev);
    }
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuEventSynchronize failed: %d\n", err);
        return -1;
    }
    return 0;
}

int fc_event_wait_ctx(fc_cuda_ctx *waiter, fc_event *ev)
{
    if (!ev || !ev->ev || !p_cuStreamWaitEvent ||
        !fc_cuda_ctx_available(waiter))
        return -1;
    /* cuStreamWaitEvent is issued in the WAITER's context — the event may have
     * been recorded on another device entirely, which is the whole point: this
     * is the only cross-die ordering primitive here that does not route through
     * the host. */
    if (fc_ctx_enter(waiter) != 0)
        return -1;
    CUresult err = p_cuStreamWaitEvent(waiter->stream, ev->ev, 0);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "[fc_cuda] cuStreamWaitEvent failed: %d\n", err);
        return -1;
    }
    return 0;
}

void fc_event_free(fc_event *ev)
{
    if (!ev)
        return;
    if (ev->ev && ev->owner && ev->owner->context &&
        fc_ctx_enter(ev->owner) == 0) {
        p_cuEventDestroy(ev->ev);
        fc_ctx_leave();
    }
    free(ev);
}

/* ================================================================
 * Timing events
 *
 * Same handle as fc_event but created WITH timing. Recorded inside a stream
 * capture they become event-record nodes (CU_EVENT_RECORD_EXTERNAL), so every
 * replay of the graph re-records them at that point.
 * ================================================================ */

fc_event *fc_tevent_create_ctx(fc_cuda_ctx *ctx)
{
    if (!fc_cuda_ctx_available(ctx) || !p_cuEventCreate || !p_cuEventRecord ||
        !p_cuEventElapsedTime || !p_cuEventDestroy)
        return NULL;
    fc_event *e = calloc(1, sizeof(*e));
    if (!e)
        return NULL;
    if (fc_ctx_enter(ctx) != 0) {
        free(e);
        return NULL;
    }
    CUresult err = p_cuEventCreate(&e->ev, 0);
    fc_ctx_leave();
    if (err != CUDA_SUCCESS) {
        free(e);
        return NULL;
    }
    e->owner = ctx;
    return e;
}

int fc_tevent_record_ctx(fc_cuda_ctx *ctx, fc_event *ev)
{
    if (!ev || !ev->ev || ev->owner != ctx || !fc_cuda_ctx_available(ctx))
        return -1;
    if (fc_ctx_enter(ctx) != 0)
        return -1;
    int capturing = 0;
    if (p_cuStreamIsCapturing)
        p_cuStreamIsCapturing(ctx->stream, &capturing);
    CUresult err;
    if (capturing && p_cuEventRecordWithFlags)
        err = p_cuEventRecordWithFlags(ev->ev, ctx->stream, CU_EVENT_RECORD_EXTERNAL);
    else if (capturing)
        err = CUDA_ERROR_INVALID_VALUE;
    else
        err = p_cuEventRecord(ev->ev, ctx->stream);
    fc_ctx_leave();
    return err == CUDA_SUCCESS ? 0 : -1;
}

int fc_tevent_elapsed_ms(fc_event *from, fc_event *to, float *ms)
{
    if (!from || !to || !ms || from->owner != to->owner || !p_cuEventElapsedTime ||
        !fc_cuda_ctx_available(from->owner))
        return -1;
    if (fc_ctx_enter(from->owner) != 0)
        return -1;
    CUresult err = p_cuEventElapsedTime(ms, from->ev, to->ev);
    fc_ctx_leave();
    return err == CUDA_SUCCESS ? 0 : -1;
}

/* ================================================================
 * Synchronization (per-context)
 * ================================================================ */

void fc_sync_ctx(fc_cuda_ctx *ctx)
{
    if (!fc_cuda_ctx_available(ctx))
        return;

    const fc_wait_mode m = fc_wait_mode_get();
    if (fc_ctx_enter(ctx) != 0)
        return;
    CUresult err;
    if (m.kind != FC_WAIT_DRIVER && ctx->stream && ctx->sync_ev) {
        FC_CENSUS_ADD(syncs, 1);
        err = p_cuEventRecord(ctx->sync_ev, ctx->stream);
        if (err == CUDA_SUCCESS)
            err = fc_wait_event_polled(ctx->sync_ev, m);
        if (err == CUDA_ERROR_NOT_READY)
            err = p_cuStreamSynchronize(ctx->stream);
    } else {
        err = ctx->stream ? fcw_cuStreamSynchronize(ctx->stream) : fcw_cuCtxSynchronize();
    }
    fc_ctx_leave();
    /* A sync that did NOT wait is a correctness problem, not a slow path:
     * every caller's next line reads a host buffer it believes the device has
     * finished writing. Discarding this return code turned that into silent
     * stale data. Callers take fc_sync_ctx as void (53 call sites), so the
     * least we owe is a loud line rather than nothing at all. */
    if (err != CUDA_SUCCESS)
        fprintf(stderr, "[fc_cuda] SYNC DID NOT COMPLETE: %d "
                        "(host may now read stale device results)\n", err);
}

void fc_sync(void)
{
    fc_sync_ctx(_fc_default_ctx);
}
