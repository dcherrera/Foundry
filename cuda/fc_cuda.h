/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * fc_cuda.h — CUDA GPU compute backend for Foundry
 *
 * Module prefix: fc_ (foundry cuda)
 * Build flag: USE_CUDA=1
 *
 * Provides GPU-accelerated tensor operations via NVIDIA CUDA.
 * Uses dlopen() for zero compile-time CUDA dependency — falls back
 * to CPU if CUDA is unavailable at runtime.
 *
 * Key difference from Metal: CUDA targets discrete GPUs with separate
 * VRAM, so buffers require explicit Host-to-Device / Device-to-Host
 * copies. Weights should be uploaded once and kept resident.
 *
 * Target: CUDA 11.4 / sm_37 (K80) — no FP16 math, no tensor cores,
 * uses __shfl_down() not __shfl_down_sync().
 */

#ifndef FOUNDRY_CUDA_H
#define FOUNDRY_CUDA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Opaque types
 * ================================================================ */

typedef struct fc_buffer fc_buffer;

/* Per-device CUDA context. Holds one device's context, stream,
 * cuBLAS handle, PTX modules and kernel function handles. One
 * instance per K80 die lets a single process drive multiple dies
 * concurrently. The dlopen'd driver function table is process-wide
 * and shared by all contexts. Opaque — see fc_cuda.c for fields. */
typedef struct fc_cuda_ctx fc_cuda_ctx;

/* ================================================================
 * Process-wide driver loading
 *
 * The CUDA driver / cuBLAS shared libraries are dlopen'd once per
 * process; their function pointers are shared by all contexts.
 * fc_cuda_init_device() loads them lazily on first call, so most
 * callers never need fc_driver_load() directly.
 * ================================================================ */

/* dlopen libcuda + libcublas and resolve all driver symbols, then
 * cuInit(0). Idempotent. Returns 0 on success, -1 on failure
 * (e.g. no CUDA driver present — the Mac case). */
int fc_driver_load(void);

/* Returns true once fc_driver_load() has succeeded. */
bool fc_driver_loaded(void);

/* ================================================================
 * Per-device context lifecycle (multi-die)
 *
 * Each context owns one device's CUcontext, stream, cuBLAS handle,
 * and a private copy of the PTX modules/kernels. Operations taking
 * an fc_cuda_ctx* make that context current (push) for the call and
 * restore the previous context (pop) on return, so contexts for
 * different dies never bleed into each other.
 * ================================================================ */

/* Initialize CUDA for a specific device id: dlopen driver (if not
 * already), cuDeviceGet, cuCtxCreate, stream + cuBLAS create, and
 * per-context cuModuleLoadData of the embedded PTX.
 *
 * @param device_id  CUDA ordinal of the die to bind (0-based)
 * @return           Owned context handle, or NULL on failure.
 *                   Free with fc_cuda_ctx_free().
 *
 * Thread-safe: no (driver load + cuCtxCreate are not serialized here)
 * Reentrant: no */
fc_cuda_ctx* fc_cuda_init_device(int device_id);

/* Destroy a per-device context: sync, unload modules, destroy
 * cuBLAS handle, stream and context. Safe to call with NULL. */
void fc_cuda_ctx_free(fc_cuda_ctx *ctx);

/* Returns true if the context is initialized and ready. */
bool fc_cuda_ctx_available(const fc_cuda_ctx *ctx);

/* Returns the CUDA device ordinal this context is bound to, or -1. */
int fc_cuda_ctx_device_id(const fc_cuda_ctx *ctx);

/* ================================================================
 * MemoryAware VRAM signals
 * ================================================================ */

/* Query this die's free/total VRAM via cuMemGetInfo, with ctx made current for
 * the call. Authoritative at the moment of the call but racy under concurrent
 * alloc, so it is a decision-point cross-check, not a per-token input. Either
 * output pointer may be NULL. Returns 0 on success, -1 if unavailable.
 *
 * Thread-safe: no (makes ctx current; serialize with other ctx ops)
 * Reentrant: no */
int fc_mem_info_ctx(fc_cuda_ctx *ctx, size_t *free_bytes, size_t *total_bytes);

/* Same reading, but for a die this process does NOT own: creates a bare
 * context (no cuBLAS handle, no PTX modules), reads cuMemGetInfo, destroys it.
 * The die is left byte-identical. Use this for enumeration/probing; use
 * fc_mem_info_ctx for a die you already hold a context on.
 *
 * Probing with fc_cuda_init_device() instead is what stranded a full ~124 MiB
 * context on every die a process merely looked at — including dies belonging
 * to another serve.
 *
 * Thread-safe: no (creates/destroys a context on the calling thread)
 * Reentrant: no */
int fc_cuda_probe_device(int device_id, size_t *free_bytes, size_t *total_bytes);

/* Running sum of bytes this context has allocated and not yet freed, tallied at
 * the buffer choke points. Free + predictive — the number admission schedules
 * against. Returns 0 for a NULL ctx.
 *
 * Thread-safe: no (plain read of a counter mutated on the alloc path)
 * Reentrant: yes */
size_t fc_ctx_alloc_bytes(const fc_cuda_ctx *ctx);

/* ================================================================
 * Crash-time teardown
 * ================================================================ */

/* Number of ready contexts currently registered (live dies). */
int fc_cuda_live_ctx_count(void);

/* Drain + cuCtxDestroy every live context, then the caller MUST exit so
 * libcuda closes the /dev/nvidiaN fd — the only step that reclaims Kepler VRAM
 * (cuCtxDestroy alone does not). Intended for the SIGINT/SIGTERM shutdown path.
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: NO — calls into libcuda. Use only from a controlled-shutdown
 *   signal (SIGINT/SIGTERM), never from SIGSEGV/SIGABRT (there, _exit() and
 *   lean on kernel fd-close + startup stranded-VRAM detection). */
void fc_cuda_emergency_teardown(void);

/* ================================================================
 * Lifecycle (process-default context — back-compat single-die path)
 *
 * These keep every existing single-context call site working. The
 * first fc_init_on() creates a process-default fc_cuda_ctx bound to
 * the device it names; all the non-ctx functions below (fc_buffer_*,
 * fc_sgemm, kernels, fc_sync) delegate to it. Multi-die callers should
 * use the fc_cuda_ctx* variants instead.
 * ================================================================ */

/* Initialize CUDA on `device_id` into the process-default context:
 * dlopen libcuda + libcublas, cuInit, cuCtxCreate, load embedded PTX
 * kernels. Returns 0 on success, -1 on failure. Idempotent — the FIRST
 * call wins, so a caller that knows its placement must call this before
 * anything else forces a default context onto a die it does not own. */
int fc_init_on(int device_id);

/* fc_init_on(0). Only correct when the process genuinely owns device 0
 * (or CUDA_VISIBLE_DEVICES has pinned its die to ordinal 0). Callers that
 * have a placement MUST use fc_init_on() — this is how a run pinned to
 * two other dies ended up holding a context on die 0. */
int fc_init(void);

/* True once libcuda is loadable and cuInit has succeeded. Unlike
 * fc_available() this creates NO context, so it is the right gate for
 * "is there a GPU at all" questions asked before placement is known. */
bool fc_driver_available(void);

/* Cleanup the process-default context and unload driver libraries. */
void fc_free(void);

/* Returns true if the process-default context is initialized. */
bool fc_available(void);

/* Returns the process-default context, or NULL if fc_init() has not
 * succeeded. Lets a caller bridge from the back-compat path to the
 * explicit-ctx API without re-initializing. */
fc_cuda_ctx* fc_default_ctx(void);

/* ================================================================
 * Buffers (discrete GPU memory — explicit H2D/D2H copies)
 *
 * Unlike Metal's unified memory, CUDA buffers live in GPU VRAM.
 * fc_buffer_from_ptr() allocates GPU memory and copies data from
 * CPU. Use fc_buffer_to_cpu() to read results back.
 *
 * For persistent data (model weights), create the buffer once at
 * load time and reuse it across inference steps.
 * ================================================================ */

/* --- Per-context variants (multi-die) ---
 * Each allocation is tagged with its owning context; frees and copies
 * automatically make that context current. The buffer must only be
 * used with the context that created it. */

/* Allocate a GPU buffer on ctx's device and copy CPU data into it. */
fc_buffer* fc_buffer_from_ptr_ctx(fc_cuda_ctx *ctx, const void *cpu, size_t bytes);

/* Allocate an uninitialized GPU buffer on ctx's device. */
fc_buffer* fc_buffer_alloc_ctx(fc_cuda_ctx *ctx, size_t bytes);

/* --- Process-default-context variants (back-compat) --- */

/* Allocate GPU buffer and copy CPU data to it (Host-to-Device). */
fc_buffer* fc_buffer_from_ptr(const void *cpu, size_t bytes);

/* Allocate GPU buffer without initializing (for output buffers). */
fc_buffer* fc_buffer_alloc(size_t bytes);

/* Free a GPU buffer (uses the buffer's owning context). */
void fc_buffer_free(fc_buffer *buf);

/* Non-owning offset view: returns a heap fc_buffer whose device pointer is
 * base's + byte_off (size shrunk by byte_off), sharing base's owning context.
 * It does NOT own the device memory — free ONLY with fc_buffer_free_view
 * (which frees the wrapper struct, never the underlying allocation). Used by
 * the die-interleave (L5) to launch a slot kernel over a sub-range
 * [slot_base, slot_base+count) of a persistent per-slot ring while the batch
 * scratch stays group-local [0,count): the ring buffer is offset by
 * slot_base*row_bytes so the kernel's slot index 0 lands at slot_base.
 * Returns NULL on bad args (byte_off past the end). */
fc_buffer* fc_buffer_offset_view(fc_buffer *base, size_t byte_off);

/* Free a view created by fc_buffer_offset_view: frees the wrapper struct only,
 * NOT the aliased device memory. A no-op on NULL. */
void fc_buffer_free_view(fc_buffer *view);

/* Copy GPU buffer contents back to CPU (Device-to-Host). */
void fc_buffer_to_cpu(fc_buffer *buf, void *dst);

/* Copy CPU data into an existing GPU buffer (Host-to-Device). */
void fc_buffer_from_cpu(fc_buffer *buf, const void *src);

/* Get buffer size in bytes. */
size_t fc_buffer_size(fc_buffer *buf);

/* Partial copies (first `bytes` of the buffer; clamped to its size). The
 * batch path sizes persistent buffers for the max batch but moves only the
 * live n tokens' bytes per call. */
void fc_buffer_to_cpu_n(fc_buffer *buf, void *dst, size_t bytes);
void fc_buffer_from_cpu_n(fc_buffer *buf, const void *src, size_t bytes);

/* Async host-to-device copy of the first `bytes` queued on the buffer's owning ctx stream
 * (stream-ordered; does not wait for the stream to drain). A pageable `src` may be reused
 * as soon as the call returns. Returns 0, or -1 (unavailable: use fc_buffer_from_cpu_n). */
int fc_buffer_from_cpu_n_async(fc_buffer *buf, const void *src, size_t bytes);

/* Async device-to-host copy queued on the buffer's owning ctx stream (non-
 * blocking). Stream-ordered after prior kernel work; the caller MUST
 * fc_sync_ctx(owner) before reading `dst`. Returns 0 on success, -1 if async
 * DtoH is unavailable (caller should fall back to the blocking variant). */
int fc_buffer_to_cpu_n_async(fc_buffer *buf, void *dst, size_t bytes);

/**
 * Allocates page-locked (pinned) host memory usable by every context's DMA
 * (cuMemHostAlloc, PORTABLE). Copies to or from it skip the driver's pageable
 * staging copy, and async copies from it are truly async: the DMA reads the
 * memory when the stream reaches the copy, so the caller must not rewrite it
 * until that copy has completed (an event recorded after it, or a sync).
 *
 * @param ctx    Any live context (the allocation needs one current)
 * @param bytes  Bytes to allocate
 * @return       Pointer, or NULL (no driver symbol, no context, or the
 *               allocation failed): the caller falls back to malloc.
 *
 * Thread-safe: yes (enters/leaves ctx)
 */
void *fc_host_alloc(fc_cuda_ctx *ctx, size_t bytes);

/** Frees memory from fc_host_alloc (NULL is a no-op). Never pass malloc'd memory. */
void fc_host_free(fc_cuda_ctx *ctx, void *p);

/* Host-to-device copy into a byte offset of the buffer (bounds-checked
 * no-op on overflow). Used to re-sync a row range of the GPU KV cache
 * from the host mirror. */
void fc_buffer_from_cpu_at(fc_buffer *buf, size_t dst_off, const void *src,
                           size_t bytes);

/* Device-to-host copy from a byte offset of the buffer (bounds-checked
 * no-op on overflow). Used to refresh a row range of the host KV mirror
 * from the GPU cache byte-for-byte. */
void fc_buffer_to_cpu_at(fc_buffer *buf, void *dst, size_t src_off,
                         size_t bytes);

/* Async device-to-device range copy on ctx's stream (both buffers must
 * belong to ctx; bounds-checked no-op otherwise). */
void fc_buffer_copy_d2d_ctx(fc_cuda_ctx *ctx, fc_buffer *dst, size_t dst_off,
                            fc_buffer *src, size_t src_off, size_t bytes);

/* ================================================================
 * Peer (die-to-die) copies — for multi-die tensor/vocab split on cards
 * whose dies are separate CUDA devices (K80: two GK210 dies behind an
 * on-card PLX switch, cudaDeviceCanAccessPeer==1, PIX single-bridge path).
 * ================================================================ */

/* Enable P2P access between two contexts' devices in BOTH directions.
 * Idempotent: a second call for an already-enabled pair returns 0. Returns
 * 0 on success (or if the pair is already enabled), -1 if either context is
 * unavailable, the driver lacks the peer symbols, or the devices cannot
 * access each other (cuDeviceCanAccessPeer==0). Call once at model load. */
int fc_cuda_enable_peer(fc_cuda_ctx *a, fc_cuda_ctx *b);

/* Whether CUDA devices dev_a and dev_b (ordinals) sit on ONE multi-GPU board
 * (the two GK210 dies of a K80: CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD and the same
 * MULTI_GPU_BOARD_GROUP_ID). No context needed. 1 = same board, 0 = not
 * (separate boards, or a single-GPU card), -1 = unknown (no driver or no
 * attribute query). Peer copies between the dies of different K80 boards report
 * success and deliver wrong bytes on some hosts, so callers enable peer
 * access only on 1. */
int fc_cuda_devices_same_board(int dev_a, int dev_b);

/* Blocking device-to-device peer copy: copy `bytes` from src (owned by
 * src's context/device) into dst (owned by dst's context/device), across
 * the peer link (cuMemcpyPeer). Both buffers may live on different devices.
 * Synchronous (uses cuMemcpyPeer, which blocks until complete). Bounds-
 * checked no-op if bytes exceeds either buffer. Requires fc_cuda_enable_peer
 * to have succeeded for the pair. Returns 0 on success, -1 on error. */
int fc_buffer_copy_peer(fc_buffer *dst, fc_buffer *src, size_t bytes);

/* Async (non-blocking) device-to-device peer copy queued on stream_ctx's
 * stream — stream-ordered after prior work on that stream, does NOT block the
 * host. stream_ctx must be one of the two buffers' owners (normally dst's, so
 * the receiving die both issues the copy and can event-sync before reading
 * dst). Caller must fc_sync_ctx(stream_ctx) or wait on an event recorded after
 * this before reading dst. Requires fc_cuda_enable_peer for the pair. Returns 0
 * on success, -1 if cuMemcpyPeerAsync is unavailable (fall back to the blocking
 * fc_buffer_copy_peer). */
int fc_buffer_copy_peer_async(fc_buffer *dst, fc_buffer *src, size_t bytes,
                              fc_cuda_ctx *stream_ctx);

/* Async 2-D copy queued on stream_ctx's stream: `height` rows of `width`
 * bytes, from src + src_off (row pitch src_pitch) to dst + dst_off (row pitch
 * dst_pitch). The buffers may be on one die or on two (cuMemcpy3DPeerAsync,
 * Depth 1); a cross-die copy goes over the peer link only if
 * fc_cuda_enable_peer succeeded for the pair (else the driver stages it).
 * Queue a cross-die copy on the READER's stream (the destination die pulls,
 * after waiting on an event the source recorded once the data was ready):
 * completion is ordered only on the queuing stream. On K80 a copy pushed from
 * the source's stream was seen incomplete on the destination after the source
 * stream had synced (test_matvec_q80_ksplit, 2 of 9 rows landed).
 * Bounds-checked: returns -1 (nothing queued) on overflow, a missing driver
 * symbol or a launch error, 0 when queued. */
int fc_buffer_copy_2d_async(fc_buffer *dst, size_t dst_off, size_t dst_pitch,
                            fc_buffer *src, size_t src_off, size_t src_pitch,
                            size_t width, size_t height, fc_cuda_ctx *stream_ctx);

/* fc_buffer_copy_2d_async as a kernel (fc_copy_2d) on stream_ctx's die, which must own
 * dst: the reader's SMs pull src, which may be a peer die's buffer (fc_cuda_enable_peer
 * for the pair). A CUDA graph can capture it and keeps it on this die; the peer copy
 * cuMemcpy3DPeerAsync cannot be captured, and a captured cross-die memcpy node is issued
 * from the graph's launching die (a push, which lands incomplete on K80). Offsets,
 * pitches and width in 16-byte units, height <= 65535. Returns -1 (nothing queued) on
 * overflow, misalignment, a dst not owned by stream_ctx or a PTX without fc_copy_2d. */
int fc_buffer_copy_2d_kernel_async(fc_buffer *dst, size_t dst_off, size_t dst_pitch,
                                   fc_buffer *src, size_t src_off, size_t src_pitch,
                                   size_t width, size_t height, fc_cuda_ctx *stream_ctx);

/* One die's half of a two-die exchange (fc_xchg_2d), ordered by device counters, not
 * events: on stream_ctx (which owns dst, own and sync) it copies own's rows to dst at
 * own_off and pulls the partner's rows (peer, a peer die's buffer of the same shape) to
 * dst at peer_off once the partner's launch for the same slot has started. Both dies
 * must call it once per exchange, in the same slot order, each with its own sync buffer
 * and the partner's (FC_XCHG_SYNC_BYTES per slot, zeroed together before first use and
 * after any failed step). A captured graph needs no cross-die edge for it. Rows: height
 * <= 65535, src_pitch for own and peer, dst_pitch for dst; offsets, pitches and width in
 * 16-byte units; fc_cuda_enable_peer for the pair. Returns -1 (nothing queued) on
 * overflow, misalignment, wrong owners or a PTX without fc_xchg_2d. */
#define FC_XCHG_SYNC_BYTES 16
int fc_xchg_2d_async(fc_buffer *dst, size_t own_off, size_t peer_off, size_t dst_pitch,
                     fc_buffer *own, fc_buffer *peer, size_t src_pitch, size_t width,
                     size_t height, fc_buffer *sync, fc_buffer *peer_sync, int slot,
                     fc_cuda_ctx *stream_ctx);

/* ================================================================
 * Dispatch thresholds
 *
 * On discrete GPU, even single-token decode can benefit from GPU
 * since GPU memory bandwidth (240 GB/s on K80) exceeds CPU
 * (~20 GB/s on server). However, PCIe transfer overhead means
 * we need sufficient work to amortize the copy cost.
 *
 * These thresholds assume weights are already resident in VRAM
 * (only activations transfer per token).
 * ================================================================ */

#define FC_GEMM_THRESHOLD    (64L * 1024 * 1024)   /* ~64M ops */
#define FC_GEMV_THRESHOLD    (4096L * 4096)         /* ~16M ops */
#define FC_Q8_THRESHOLD      (0)                    /* always use GPU for Q8 matvec */
#define FC_ELEM_THRESHOLD    (1024 * 1024)          /* 1M elements */

/* ================================================================
 * BLAS via cuBLAS (dlopen'd)
 *
 * All matrices are row-major float32. cuBLAS is column-major,
 * so we swap A/B and transpose the problem internally.
 * ================================================================ */

/* C = alpha * A @ B + beta * C
 * A: [M, K], B: [K, N], C: [M, N] — all row-major float32. */
void fc_sgemm_ctx(fc_cuda_ctx *ctx, fc_buffer *A, fc_buffer *B, fc_buffer *C,
                  int M, int N, int K, float alpha, float beta);
void fc_sgemm(fc_buffer *A, fc_buffer *B, fc_buffer *C,
              int M, int N, int K, float alpha, float beta);

/* y = alpha * A @ x + beta * y
 * A: [M, K] row-major, x: [K], y: [M] — all float32. */
void fc_sgemv_ctx(fc_cuda_ctx *ctx, fc_buffer *A, fc_buffer *x, fc_buffer *y,
                  int M, int K, float alpha, float beta);
void fc_sgemv(fc_buffer *A, fc_buffer *x, fc_buffer *y,
              int M, int K, float alpha, float beta);

/* ================================================================
 * Custom kernels (loaded from embedded PTX)
 * ================================================================ */

/* RMS normalization over last dimension.
 * x: [n], weight: [n] (may be NULL), out: [n] */
void fc_rms_norm_ctx(fc_cuda_ctx *ctx, fc_buffer *x, fc_buffer *weight,
                     fc_buffer *out, int n, float eps);
void fc_rms_norm(fc_buffer *x, fc_buffer *weight, fc_buffer *out,
                 int n, float eps);

/* Softmax over last dimension.
 * x: [rows, cols], out: [rows, cols] */
void fc_softmax_ctx(fc_cuda_ctx *ctx, fc_buffer *x, fc_buffer *out,
                    int rows, int cols);
void fc_softmax(fc_buffer *x, fc_buffer *out, int rows, int cols);

/* SiLU activation: out = x * sigmoid(x) */
void fc_silu_ctx(fc_cuda_ctx *ctx, fc_buffer *x, fc_buffer *out, int n);
void fc_silu(fc_buffer *x, fc_buffer *out, int n);

/* Elementwise addition: out = a + b */
void fc_add_ctx(fc_cuda_ctx *ctx, fc_buffer *a, fc_buffer *b, fc_buffer *out, int n);
void fc_add(fc_buffer *a, fc_buffer *b, fc_buffer *out, int n);

/* Elementwise multiplication: out = a * b */
void fc_mul_ctx(fc_cuda_ctx *ctx, fc_buffer *a, fc_buffer *b, fc_buffer *out, int n);
void fc_mul(fc_buffer *a, fc_buffer *b, fc_buffer *out, int n);

/* INT8 quantized matrix-vector product.
 * x_q: [cols] int8, w: [rows, cols] int8, w_scales: [rows] float32,
 * out: [rows] float32.
 * Computes: out[i] = x_scale * w_scales[i] * dot(x_q, w[i]) */
void fc_matvec_q8_ctx(fc_cuda_ctx *ctx, fc_buffer *x_q, fc_buffer *w,
                      fc_buffer *w_scales, fc_buffer *out,
                      float x_scale, int rows, int cols);
void fc_matvec_q8(fc_buffer *x_q, fc_buffer *w, fc_buffer *w_scales,
                  fc_buffer *out, float x_scale, int rows, int cols);

/* ================================================================
 * qwen35moe kernels (loaded from embedded PTX)
 *
 * The CUDA halves of the qwen35moe forward pass. Each mirrors a CPU
 * reference (cited in its cuda_kernels .cu source) bit-for-bit and is
 * validated against the golden vectors in Pantheon/tests/golden/.
 * ================================================================ */

/* Q3_K dequant + matrix-vector product.
 * w: raw Q3_K weight bytes (row-major, row_stride bytes/row),
 * x: [cols] float32 activation, out: [rows] float32.
 * cols must be a multiple of 256; row_stride = (cols/256) * 110.
 * Computes: out[r] = dequant_q3k(w[r]) · x. */
void fc_matvec_q3k_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride);
void fc_matvec_q3k(fc_buffer *w, fc_buffer *x, fc_buffer *out,
                   int rows, int cols, int row_stride);

/* ---- Q3_K peer-stream variants (scheme 9a) ----
 * BIT-EXACT alternatives to fc_matvec_q3k that differ only in how many weight
 * loads a warp keeps in flight. They exist for the case where the die running
 * the kernel reads ANOTHER die's VRAM over the K80's on-card peer link, which
 * is LATENCY-bound (9.3-12.4 GB/s available, 2.7-4.2 GB/s achieved) — see
 * plans/results/phase9a_peer_streaming.md. On a die's OWN memory the kernel is
 * issue-bound and the stock shape is the right one.
 *   r2 = row pair per warp (the stock shape), r1 = one row per warp (2x warps)
 *   pN = N super-blocks' weight loads issued before the first is consumed
 * All variants also stage x with 128-bit loads. */
enum {
    FC_Q3K_PS_R2P1 = 0, FC_Q3K_PS_R2P2, FC_Q3K_PS_R2P4, FC_Q3K_PS_R2P8,
    FC_Q3K_PS_R1P1,     FC_Q3K_PS_R1P4, FC_Q3K_PS_R1P8,
    FC_Q3K_PS_COUNT
};

/* Human-readable name of a variant, or NULL if `variant` is out of range. */
const char *fc_matvec_q3k_ps_name(int variant);

/* Launch peer-stream variant `variant`. Same contract and same output bytes as
 * fc_matvec_q3k_ctx. Returns 0 if launched, -1 if the variant is out of range
 * or absent from this build's PTX — the caller must handle -1 rather than let
 * a missing kernel silently launch nothing. */
int fc_matvec_q3k_ps_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                         fc_buffer *out, int rows, int cols, int row_stride,
                         int variant);

/* Mixed-quant dequant + matvec family (matvec_kquants.cu). Same contract as
 * fc_matvec_q3k_ctx for each GGUF quant type the Q3_K_M mini mixes in:
 * Q4_K (144 B / 256 elems), Q5_K (176 B / 256), Q6_K (210 B / 256),
 * Q8_0 (34 B / 32), and plain F32 rows (row_stride = cols * 4). */
void fc_matvec_q4k_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride);
void fc_matvec_q5k_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride);
void fc_matvec_q6k_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride);
void fc_matvec_q80_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride);
/* Bonsai native ternary (GGUF type 42, Q2_0_g128): conditional add/sub, no
 * dequant. Repack via fc_repack_weight(42, ...). Row pair per warp. */
void fc_matvec_q2_0_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                        fc_buffer *out, int rows, int cols, int row_stride);
void fc_matvec_f32_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride);
/* F16 rows (GGUF type 1): the dense half of an F16 model — attention q/k/v/o,
 * shared expert, router, LM head. 2-byte load widened in-register (sm_37 has no
 * fp16 ARITHMETIC; cvt.f32.f16 is a storage conversion and predates it), fp32
 * accumulate. row_stride is bytes (cols * 2) and the weight is NOT repacked —
 * the GGUF bytes are already the device layout, which is what keeps a 70 GB
 * upload from needing a second host copy of every tensor.
 * These are OPTIONAL PTX symbols: probe fc_matvec_f16_available_ctx before
 * routing a model's dense weights here. With the kernel absent the launcher
 * does NOTHING, so a caller that skipped the probe would write no output at
 * all instead of falling back. */
int  fc_matvec_f16_available_ctx(fc_cuda_ctx *ctx);
void fc_matvec_f16_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                       fc_buffer *out, int rows, int cols, int row_stride);
/* BF16 rows (GGUF type 30): the same dense tensors as the F16 pair above,
 * carrying the values the model actually SHIPS — the checkpoints are bf16
 * safetensors, and converting bf16 -> F16 re-rounds every weight (7 stored
 * mantissa bits against 10, and neither field contains the other), so an F16
 * run measures a perturbed model. That fidelity is the only reason to prefer
 * it: the widen is a shift into the high half rather than a cvt, but ptxas
 * emits ONE instruction either way and the two decode at the same speed
 * (measured; see fc_bf16_to_f32 in moe_expert_group.cu). Identical
 * contract to the F16 launcher otherwise: row_stride in bytes (cols * 2), NOT
 * repacked, OPTIONAL PTX symbols, silent when absent — probe
 * fc_matvec_bf16_available_ctx before routing weights here. */
int  fc_matvec_bf16_available_ctx(fc_cuda_ctx *ctx);
void fc_matvec_bf16_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                        fc_buffer *out, int rows, int cols, int row_stride);

/* ---- Batched (multi-column) matvecs: the prefill/batch GEMM path ----
 * out[t][rows] = dequant(w) . x[t] for the n activation columns of a batch:
 * x is [n][cols] row-major (token-major), out is [n][rows]. Each weight
 * block is decoded ONCE per 4-column tile and reused across the columns, so
 * the dequant instruction stream (the single-token kernels' issue bound on
 * sm_37) amortizes over the batch. BIT-EXACT per column: column t of the
 * batched output is bit-identical to the corresponding single-column
 * fc_matvec_* call on x[t] (same per-lane chain order + reduction; verified
 * by cuda/tests/test_matvec_nx). Same layout/stride contracts as the
 * single-column variants. */
void fc_matvec_q3k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n);
void fc_matvec_q4k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n);
void fc_matvec_q5k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n);
void fc_matvec_q6k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n);
void fc_matvec_q80_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n);
void fc_matvec_f32_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n);

/* Q8_0 PARITY matvec: rank p's half of an input-column (K) split that is
 * bit-exact with fc_matvec_q80_nx_ctx. The Q8_0 kernels sum a row as
 * (V0 + V2) + (V1 + V3), V_m = the lane tree over inputs k % 4 == m; rank p
 * computes V_p + V_(p+2) from the inputs k % 2 == p with the same chains and
 * tree, so out_0 + out_1 (fc_add_ctx, either order) equals the full output bit
 * for bit. w = fc_repack_weight_q80_parity(..., p) rows; x = n columns of the
 * rank's compact input (x[c * kh + i] = full input element 2i + p of column c);
 * kh = full cols / 2, a multiple of 64; out[c * rows + row]. Tensor-parallel
 * FFN down (gate/up rows k % 2 == p produce exactly this x). */
void fc_matvec_q80_par_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                              fc_buffer *out, int rows, int kh, int row_stride,
                              int n);
/* The same with one named tile (fc_matvec_q80p_r<tile_r>c<tile_nc>), for the
 * test and bench sweeps. Returns -1 if that tile is not in the PTX or the
 * arguments are refused, 0 when launched. */
int fc_matvec_q80_par_tile_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                               fc_buffer *out, int rows, int kh, int row_stride, int n,
                               int tile_r, int tile_nc);
void fc_matvec_f16_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                          fc_buffer *out, int rows, int cols, int row_stride,
                          int n);
void fc_matvec_bf16_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *w, fc_buffer *x,
                           fc_buffer *out, int rows, int cols, int row_stride,
                           int n);

/* ================================================================
 * Quant-block repack (host-side, done once at weight upload)
 *
 * The raw GGUF block layouts are unaligned (110/144/176/210/34 B)
 * and interleave packed scales with the quant bits, which caps the
 * warp-per-row matvec kernels at 10-40 GB/s (per-lane byte loads +
 * per-element scale unpack). fc_repack_weight converts a weight's
 * blocks ONCE into the 4-byte-aligned device layout those kernels
 * read: numerically identical values, scales pre-unpacked, block
 * scale d (and dmin) pre-decoded to f32, and the quant bytes
 * transposed so lane l of a warp issues ONE aligned 32-bit load per
 * super-block. Per-super-block layouts (offsets in bytes):
 *
 *  Q3_K 110->148: u32 qw[32] @0   qw[l] = qs[l] | qs[32+l]<<8 | hmask[l]<<16
 *                 s8  sc[16] @128 (6-bit scales, unpacked, -32 applied)
 *                 f32 d      @144
 *  Q4_K 144->152: u32 qw[32] @0   qw[l] byte j = qs[32*j + l]
 *                 f32 d @128, f32 dmin @132, u8 sc[8] @136, u8 mn[8] @144
 *  Q5_K 176->184: u32 qw[32] @0 (as Q4_K), u8 qh[32] @128,
 *                 f32 d @160, f32 dmin @164, u8 sc[8] @168, u8 mn[8] @176
 *  Q6_K 210->212: u32 qlw[32] @0  qlw[l] byte t = ql[32*t + l]
 *                 u16 qh2[32] @128  qh2[l] = qh[l] | qh[32+l]<<8
 *                 s8 sc[16] @192, f32 d @208
 *  Q8_0  34-> 36: f32 d @0, s8 qs[32] @4    (per 32-elem block)
 *        when cols % 128 == 0, each run of 4 blocks (144 B) is regrouped
 *        "G4": f32 d[4] @0, then for s = 0..7 bytes 4s..4s+3 of blocks
 *        0..3 @16+16s (one 128-bit load per lane of an 8-lane row group).
 *        Same row bytes; every Q8_0 reader keys on cols % 128.
 *
 * fc_repack_row_bytes returns the repacked bytes-per-row for a GGUF
 * quant type id (8=Q8_0, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K), or 0
 * if the type/cols combination has no repacked form (caller must not
 * use the quant matvec kernels then).
 * ================================================================ */
size_t fc_repack_row_bytes(int gguf_quant, int cols);
int fc_repack_weight(int gguf_quant, const void *src, void *dst,
                     long rows, int cols);

/* GGUF Q8_0 rows -> rank `parity`'s fc_matvec_q80_par_nx_ctx slab: per row,
 * per 4 blocks (128 cols) 80 B = f32 d[4] @0, then for s = 0..7 the bytes
 * q_b[4s + parity], q_b[4s + parity + 2] of blocks b = 0..3 @16+8s.
 * cols % 128 == 0 (else 0 / -1). */
size_t fc_repack_q80_parity_row_bytes(int cols);
int fc_repack_weight_q80_parity(const void *src, void *dst, long rows, int cols,
                                int parity);

/* The grouped MoE kernels' upload layout for a merged expert tensor. Identical
 * to fc_repack_row_bytes / fc_repack_weight for every type EXCEPT Q8_0 (8),
 * whose experts use 288-B super-blocks [8 x f32 d][256 x s8] per 256 weights
 * (cols % 256 == 0, else 0 / -1) instead of the dense 36-B blocks. Callers
 * that upload ffn_{gate,up,down}_exps for fc_moe_gateup_ctx /
 * fc_moe_down_accum_ctx must use these; the dense matvecs must not. */
size_t fc_moe_repack_row_bytes(int gguf_quant, int cols);
int fc_moe_repack_weight(int gguf_quant, const void *src, void *dst,
                         long rows, int cols);

/* Partial NEOX rotary position embedding, in place.
 * data: [n_heads * head_dim] float32. Only the first rope_dim dims of
 * each head are rotated (NEOX pairing j vs j+rope_dim/2); dims
 * [rope_dim, head_dim) pass through unchanged. */
void fc_rope_neox_partial_ctx(fc_cuda_ctx *ctx, fc_buffer *data,
                              int n_heads, int head_dim, int rope_dim,
                              int pos, float theta_base);
void fc_rope_neox_partial(fc_buffer *data, int n_heads, int head_dim,
                          int rope_dim, int pos, float theta_base);

/* Per-head QK-RMSNorm, in place.
 * data: [n_heads * head_dim] float32; each head is RMS-normalized over
 * its own head_dim and scaled by the shared weight [head_dim].
 * weight may be NULL (no scaling). */
void fc_qk_rms_norm_ctx(fc_cuda_ctx *ctx, fc_buffer *data, fc_buffer *weight,
                        int n_heads, int head_dim, float eps);
void fc_qk_rms_norm(fc_buffer *data, fc_buffer *weight,
                    int n_heads, int head_dim, float eps);

/* MoE router: softmax over n_expert logits (in place), top-k select,
 * norm_topk_prob renormalize, apply expert_weights_scale.
 * logits: [n_expert] float32 (overwritten with softmax probs),
 * top_idx: [k] int32 output, top_wt: [k] float32 output.
 * k must be <= 32 (FC_MOE_MAX_TOPK). */
void fc_moe_router_topk_ctx(fc_cuda_ctx *ctx, fc_buffer *logits,
                            fc_buffer *top_idx, fc_buffer *top_wt,
                            int n_expert, int k, float expert_weights_scale);
void fc_moe_router_topk(fc_buffer *logits, fc_buffer *top_idx,
                        fc_buffer *top_wt, int n_expert, int k,
                        float expert_weights_scale);

/* Device-side exact top-k over LM-head logits (topk_logits.cu).
 * logits: [n] float32 device buffer (left intact on return).
 * out_idx/out_val: HOST arrays of k entries, filled in descending value
 * order; equal values order by ascending index (the host argmax tie-break,
 * so the k=1 result IS the host argmax). Values are bit-exact — selection
 * is compare-only — and only k (idx, val) pairs cross the host link
 * instead of the full n-float buffer. k must be <= FC_TOPK_MAX.
 * Synchronous; returns 0 on success, -1 on failure (caller should fall
 * back to a full-logits D2H copy). A row with fewer than k values above
 * -inf (all-NaN logits, say) is a failure: no pair is written through the
 * kernels' no-winner index and the call returns -1. */
#define FC_TOPK_MAX 512
int fc_topk_logits_ctx(fc_cuda_ctx *ctx, fc_buffer *logits, int n, int k,
                       int *out_idx, float *out_val);
int fc_topk_logits(fc_buffer *logits, int n, int k,
                   int *out_idx, float *out_val);

/* Batched (multi-slot) top-k (L8): the same exact selection over every row
 * of a [nslots][n] logits matrix IN PLACE (no per-row staging copy), one
 * sync + one nslots*k-pair D2H for the whole batch. out_idx/out_val are
 * HOST arrays of nslots*k entries (row u's pairs at [u*k .. u*k+k)); each
 * row's pairs are bit-exact to fc_topk_logits_ctx on that row. Returns 0
 * on success, -1 on failure / kernels absent / any row with fewer than k
 * values above -inf (caller falls back to the per-slot path). */
int fc_topk_logits_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *logits, int nslots,
                             int n, int k, int *out_idx, float *out_val);

/* As fc_topk_logits_slots_ctx, plus a per-row NaN check: out_nan[u] = 1 when
 * row u holds a NaN, 0 when it holds none, -1 when unknown (the fallback tree
 * ran; it does not check). The selection skips NaN, so the pairs alone cannot
 * show one; a caller that must reproduce a host scan of the whole row (where a
 * NaN changes the result) needs this. Same cost as the plain call: the check
 * rides on the filter pass. */
int fc_topk_logits_slots_nan_ctx(fc_cuda_ctx *ctx, fc_buffer *logits, int nslots,
                                 int n, int k, int *out_idx, float *out_val,
                                 int *out_nan);

/* As fc_topk_logits_slots_nan_ctx (out_nan may be NULL), with speed options; the
 * pairs, the NaN flags and the return value are the same under every option:
 *   FC_TOPK_OPT_ROW_BLOCKS  the fast path ranks its threshold and its candidates
 *                           over several blocks per row instead of one (on a K80,
 *                           k = 257 over 248320: 318 -> 159 us for 1 row, 946 ->
 *                           524 us for 4);
 *   FC_TOPK_OPT_GRAPH       the fast path and its result copy replay as a CUDA
 *                           graph, captured on the first call with this logits
 *                           buffer and shape (a few cached per ctx), copied into
 *                           pinned memory; ~25-90 us less a call. Falls back to
 *                           stream launches if a capture fails.
 * The no-option calls above are this with opts = 0. Not callable while the
 * caller's stream is being captured (it synchronizes). */
#define FC_TOPK_OPT_ROW_BLOCKS 1u
#define FC_TOPK_OPT_GRAPH      2u
int fc_topk_logits_slots_opt_ctx(fc_cuda_ctx *ctx, fc_buffer *logits, int nslots,
                                 int n, int k, int *out_idx, float *out_val,
                                 int *out_nan, unsigned int opts);

/* MoE weighted accumulate: moe_out[i] += weight * expert_out[i].
 * Both buffers are [n_embd] float32. */
void fc_moe_weighted_add_ctx(fc_cuda_ctx *ctx, fc_buffer *expert_out,
                             fc_buffer *moe_out, float weight, int n_embd);
void fc_moe_weighted_add(fc_buffer *expert_out, fc_buffer *moe_out,
                         float weight, int n_embd);

/* MoE shared-expert sigmoid gate: shexp_out[i] *= sigmoid(gate_logit).
 * shexp_out: [n_embd] float32, gated in place. */
void fc_moe_shared_gate_ctx(fc_cuda_ctx *ctx, fc_buffer *shexp_out,
                            float gate_logit, int n_embd);
void fc_moe_shared_gate(fc_buffer *shexp_out, float gate_logit, int n_embd);

/* ---- Grouped top-k expert kernels (moe_expert_group.cu) ----
 * The whole routed-expert SwiGLU block in two launches, reading the router
 * selection (top_idx/top_wt) directly on device — no host read-back. */

/* Are the 16-bit expert kernels (F16, BF16) routed to their uint2-load form?
 * Set by FC_MOE_VEC_LOADS=1 in the environment and read per call. An
 * unsupported combination FAILS CLOSED (no launch) rather than quietly running
 * the scalar kernel, so a bench cannot end up timing the wrong thing.
 *
 * The vec kernels move the SAME DRAM bytes in HALF the LSU transactions (a
 * warp-wide 16-bit load fills 64 B of a 128-B transaction; a uint2 load fills
 * it), and on die 7 that is worth 1.31x on gate+up (0.2418 -> 0.1849 ms, 139 ->
 * 181 GB/s) and 1.32x on down (0.1304 -> 0.0986 ms, 129 -> 170 GB/s), the same
 * for both formats. 181 GB/s is 76% of the die's 240 GB/s theoretical, against
 * 58% for the scalar form.
 *
 * It is opt-in and not the default because it is not a pure win: to load four
 * contiguous weights per lane the kernels must hand lane l the run {4l..4l+3}
 * instead of the scalar map's stride-32 set {32i+l}. That repartitions the
 * products across lanes, so the per-lane fmaf chains and the shfl tree
 * associate differently and the output is NOT bit-identical to the scalar
 * kernels'. Each set is bit-exact against its own CPU replica and neither is
 * more accurate; but a run that has to reproduce an earlier run's exact logits
 * must pin this setting, so flipping the default is a deliberate call, not a
 * tuning detail. Exposed so a test can pick the matching replica from the same
 * source of truth the dispatch uses. */
int fc_moe_vec_loads_enabled(void);

/* Grouped gate+up SwiGLU for the k selected experts.
 * gate_all/up_all: merged [n_expert * ff * row_stride] repacked quant bytes,
 * x: [cols] f32, top_idx: [k] int32 (device), hid: [k*ff] f32 out.
 * hid[s*ff+row] = silu(gate[e_s,row] . x) * (up[e_s,row] . x).
 *
 * gguf_quant selects the decoder (11 = Q3_K, 12 = Q4_K, 1 = F16, 30 = BF16,
 * 8 = Q8_0 on the fc_moe_repack_weight layout),
 * exactly as it does
 * for fc_moe_down_accum_ctx below. FAILS CLOSED: an unsupported quant, or a
 * supported one whose kernel is missing from the loaded PTX, logs a named
 * error and launches NOTHING (hid untouched) rather than falling through to
 * another quant's decode. Returns 0 on launch, -1 otherwise.
 *
 * Probe fc_moe_gateup_quant_available_ctx BEFORE committing a model to the
 * grouped path: the Q4_K kernels are OPTIONAL symbols, so an fc_cuda_ptx.h
 * regenerated before they existed answers 0 for Q4_K and the caller can take
 * the per-expert fallback instead of failing on every layer at run time. */
int fc_moe_gateup_quant_available_ctx(fc_cuda_ctx *ctx, int gguf_quant);
int fc_moe_gateup_ctx(fc_cuda_ctx *ctx, int gguf_quant, fc_buffer *gate_all,
                      fc_buffer *up_all, fc_buffer *x, fc_buffer *top_idx,
                      fc_buffer *hid, int k, int ff, int cols, int row_stride);

/* Q3_K-pinned shorthand for fc_moe_gateup_ctx (benches, older callers). */
void fc_moe_gateup_q3k_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                           fc_buffer *up_all, fc_buffer *x, fc_buffer *top_idx,
                           fc_buffer *hid, int k, int ff, int cols,
                           int row_stride);

/* Grouped down-projection + weighted accumulate:
 * out[row] = sum_s top_wt[s] * (down[e_s,row] . hid[s]).
 * gguf_quant selects the decoder (11 = Q3_K, 12 = Q4_K, 14 = Q6_K, 8 = Q8_0, 1 = F16,
 * 30 = BF16), and a
 * merged ffn_down_exps really is mixed WITHIN one file — Q3_K/Q4_K by layer in
 * a Q3_K_M, Q4_K/Q6_K by layer in a Q4_K_M. Unsupported quants log a named
 * error and launch nothing. out is WRITTEN.
 *
 * Q6_K is an OPTIONAL PTX symbol (a header regenerated before it existed leaves
 * it NULL); probe fc_moe_down_quant_available_ctx before committing a model to
 * the grouped path. */
int fc_moe_down_quant_available_ctx(fc_cuda_ctx *ctx, int gguf_quant);
void fc_moe_down_accum_ctx(fc_cuda_ctx *ctx, int gguf_quant, fc_buffer *down_all,
                           fc_buffer *hid, fc_buffer *top_idx, fc_buffer *top_wt,
                           fc_buffer *out, int k, int rows, int ff,
                           int row_stride);

/* Warps cooperating per down-projection output row pair (the wpp launch arg
 * of fc_moe_down_q3k/q4k; FC_MOE_DOWN_WPP env-overridable for tuning, block
 * size via FC_MOE_DOWN_THREADS). The 512-col down dot is 2 super-blocks, so
 * one warp per row-pair stalls on latency; wpp warps split the k slots and
 * shared-mem-reduce. Must divide the block's warp count. wpp=2 with
 * 128-thread blocks measured best on a K80 (the kernel is
 * instruction-issue bound at ~4.3 IPC — see moe_expert_group.cu). Mirrored
 * by the bench_matvec CPU replica. */
#define FC_MOE_DOWN_WPP_DEFAULT 2
#define FC_MOE_DOWN_THREADS_DEFAULT 128

/* MoE expert-union dedup request-tile depth. Must match FC_MOE_DEDUP_RT in
 * cuda_kernels/moe_expert_group.cu: it sets both the per-block register
 * footprint and grid.z = ceil(maxr / RT). An expert with R requests has its
 * rows read ceil(R/RT) times, so RT >= the typical R is what makes the dedup
 * pay. */
#define FC_MOE_DEDUP_RT 8

/* Fused SwiGLU elementwise: out[i] = silu(a[i]) * b[i] (in-place safe). */
void fc_moe_silu_mul_ctx(fc_cuda_ctx *ctx, fc_buffer *a, fc_buffer *b,
                         fc_buffer *out, int n);

/* Shared-expert scalar gate fully on-device:
 * shexp[i] *= sigmoid(gate_w . x). gate_w, x: [n] f32. */
void fc_moe_shexp_dot_gate_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_w,
                               fc_buffer *x, fc_buffer *shexp, int n);

/* ================================================================
 * Fused per-layer launches (fused_layer.cu)
 *
 * Decode on the K80 is launch-bound (~80 us/launch), so fc_fused_multi
 * batches up to FC_FUSED_MAX_SEG independent matvec "segments" that read
 * the SAME x into ONE launch, each with an optional fused epilogue. Every
 * segment's math is bit-identical to its stand-alone kernel (the kernel
 * bodies are duplicated verbatim; see fused_layer.cu).
 * ================================================================ */

#define FC_FUSED_MAX_SEG 8

/* One matvec segment. kind: 0 = quant warp-per-row (qtype 8/11/12/13/14,
 * repacked layout), 1 = F32 block-per-row (cols >= 256), 2 = quant gate+up
 * pair (w = gate, w2 = up; writes silu(g.x)*(u.x)). ep (epilogue applied to
 * the finished row value): 0 none; 1 sigmoid; 2 decay gate out[r] =
 * aux1[r]*softplus(v+aux0[r]); 3 residual out[r] = v, aux1[r] += v;
 * 4 moe combine out[r] = aux0[r] + v*aux2[0], aux1[r] += out[r]. */
typedef struct {
    fc_buffer *w;      /* weight bytes (repacked quant, or raw F32 rows) */
    fc_buffer *w2;     /* kind 2 only: the up weight (same type/stride) */
    fc_buffer *out;    /* [rows] f32 */
    fc_buffer *aux0;   /* epilogue operand (see ep) or NULL */
    fc_buffer *aux1;
    fc_buffer *aux2;
    int rows, cols, row_stride;
    int kind, qtype, ep;
} fc_fused_seg;

/* Validate the segments and upload their device descriptor ONCE (all
 * referenced buffers must be persistent). Returns the descriptor buffer
 * (free with fc_buffer_free) and the launch shape, or NULL if any segment
 * is unsupported (caller keeps its unfused path). */
fc_buffer *fc_fused_build_ctx(fc_cuda_ctx *ctx, const fc_fused_seg *segs,
                              int nseg, int *total_blocks, int *shared_bytes);

/* Launch the fused multi-matvec: every segment's dot(s) against x in one
 * kernel. desc/total_blocks/shared_bytes come from fc_fused_build_ctx. */
void fc_fused_multi_ctx(fc_cuda_ctx *ctx, fc_buffer *desc, int nseg,
                        int total_blocks, int shared_bytes, fc_buffer *x);

/* Fused per-head QK-RMSNorm + partial NEOX RoPE on Q and K in one launch
 * (replaces fc_qk_rms_norm x2 + fc_rope_neox_partial x2; per-head math
 * identical). Returns 0 on launch, -1 if the kernel is unavailable (caller
 * must fall back to the four stand-alone launches). */
int fc_attn_qknorm_rope_ctx(fc_cuda_ctx *ctx, fc_buffer *q, fc_buffer *k,
                            fc_buffer *q_norm_w, fc_buffer *k_norm_w,
                            int n_head, int n_head_kv, int head_dim, float eps,
                            int rope_dim, int pos, float theta_base);

/* ================================================================
 * SSM Gated-Delta-Net kernels (loaded from embedded PTX)
 *
 * The GPU halves of the single-token Gated-Delta-Net SSM block. Each
 * mirrors a static helper in Pantheon/src/models/qwen35moe_ssm.c
 * (cited in ssm_gated_delta.cu) and is validated against the SSM
 * golden vectors in Pantheon/tests/golden/.
 * ================================================================ */

/* Per-token delta-rule recurrence (the centerpiece). For one token, per
 * (head h, row j): decays the transposed state row, computes the rank-1
 * delta update in place, and reads y back after the write.
 *   s:    [n_v_heads * S * S] transposed state (s[h*S*S+j*S+i]==S_h[i][j]),
 *         updated in place; q,k,v: [n_v_heads * S] (q/k L2-normed + GQA-
 *         repeated on host, v straight); g: [n_v_heads] PRE-exp decay;
 *         beta: [n_v_heads] sigmoid'd; y: [n_v_heads * S] output.
 *   scale = 1/sqrt(state_size). Mirrors qwen35moe_compute_recur_heads. */
void fc_ssm_recurrence_ctx(fc_cuda_ctx *ctx, fc_buffer *s, fc_buffer *q,
                           fc_buffer *k, fc_buffer *v, fc_buffer *g,
                           fc_buffer *beta, fc_buffer *y,
                           int n_v_heads, int state_size, float scale);
void fc_ssm_recurrence(fc_buffer *s, fc_buffer *q, fc_buffer *k, fc_buffer *v,
                       fc_buffer *g, fc_buffer *beta, fc_buffer *y,
                       int n_v_heads, int state_size, float scale);

/* Causal depthwise conv1d (kernel size conv_kernel) + SiLU over the full
 * conv_dim stream, priming from conv_state and rolling it forward.
 *   qkv: [conv_dim] raw input, weight: [conv_dim * conv_kernel] row-major,
 *   conv_state: [conv_dim * (conv_kernel-1)] history (updated in place),
 *   out: [conv_dim] SiLU'd output. Mirrors qwen35moe_conv1d_silu. */
void fc_ssm_conv1d_silu_ctx(fc_cuda_ctx *ctx, fc_buffer *qkv, fc_buffer *weight,
                            fc_buffer *conv_state, fc_buffer *out,
                            int conv_dim, int conv_kernel);
void fc_ssm_conv1d_silu(fc_buffer *qkv, fc_buffer *weight, fc_buffer *conv_state,
                        fc_buffer *out, int conv_dim, int conv_kernel);

/* Gated RMSNorm: gated[h*S+d] = RMSNorm(y_h)[d] * weight[d] * silu(z[h*S+d]),
 * per V-head over state_size dims (norm weight shared across heads). Omits the
 * trailing ssm_out projection (reuse fc_matvec_q3k). y,z,gated: [n_v_heads*S],
 * weight: [state_size]. Mirrors qwen35moe_gated_norm. */
void fc_ssm_gated_norm_ctx(fc_cuda_ctx *ctx, fc_buffer *y, fc_buffer *weight,
                           fc_buffer *z, fc_buffer *gated,
                           int n_v_heads, int state_size, float eps);
void fc_ssm_gated_norm(fc_buffer *y, fc_buffer *weight, fc_buffer *z,
                       fc_buffer *gated, int n_v_heads, int state_size, float eps);

/* SSM "glue" kernels — replace host round-trips in the qwen35moe GPU forward. */
/* beta = sigmoid(beta), in place [n]. */
void fc_sigmoid_ctx(fc_cuda_ctx *ctx, fc_buffer *x, int n);
/* g[h] = ssm_a[h] * softplus(alpha[h] + ssm_dt[h]), per V-head [n]. */
void fc_ssm_decay_gate_ctx(fc_cuda_ctx *ctx, fc_buffer *alpha, fc_buffer *ssm_dt,
                           fc_buffer *ssm_a, fc_buffer *g, int n);
/* Per-K-head L2-norm of q,k over state_size dims + GQA interleaved-repeat to
 * qg/kg [n_v_heads*S]; v (conv_out[2*key_dim..], val_dim) straight-copied to vg.
 * conv_out layout: [q key_dim | k key_dim | v val_dim]. */
void fc_ssm_qknorm_gqa_ctx(fc_cuda_ctx *ctx, fc_buffer *conv_out, fc_buffer *qg,
                           fc_buffer *kg, fc_buffer *vg, int n_k_heads,
                           int kv_groups, int state_size, int key_dim,
                           int val_dim, float eps);

/* ================================================================
 * Batched (prefill) kernels — n positions of ONE sequence per launch
 *
 * The launch-count killers for batched prefill: each covers all n
 * positions of a batch in one launch, with per-position math BIT-EXACT
 * to its single-position counterpart (the stateful SSM kernels scan the
 * positions inside the kernel, carrying state exactly as n sequential
 * launches would). Batched activations are token-major: a [n][dim]
 * buffer holds position t's vector at offset t*dim.
 * ================================================================ */

/* Batched RMS norm: x/out [rows][n], one block per row. */
void fc_rms_norm_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *x, fc_buffer *weight,
                        fc_buffer *out, int n, float eps, int rows);

/* Batched partial NEOX RoPE: data [npos][n_heads*head_dim], row t rotated
 * at position pos0 + t, in place. */
void fc_rope_neox_partial_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *data,
                                 int n_heads, int head_dim, int rope_dim,
                                 int pos0, float theta_base, int npos);

/* Batched MoE router: logits [ntok][n_expert] (softmax'd in place),
 * top_idx [ntok][k], top_wt [ntok][k]. Rank-select router (no selection
 * rounds) when n_expert is a power of two <= 256 — bit-exact to the block
 * router — else the block router.
 * fc_moe_router_topk_ctx is this with ntok = 1. */
void fc_moe_router_topk_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *logits,
                               fc_buffer *top_idx, fc_buffer *top_wt,
                               int n_expert, int k,
                               float expert_weights_scale, int ntok);
/* The block router (one 256-thread block per token) every launch used before
 * the rank router: the fallback, and the reference the exactness tests hold
 * the rank router to. Same arguments and outputs. */
void fc_moe_router_topk_nx_block_ctx(fc_cuda_ctx *ctx, fc_buffer *logits,
                                     fc_buffer *top_idx, fc_buffer *top_wt,
                                     int n_expert, int k,
                                     float expert_weights_scale, int ntok);

/* Batched grouped gate+up SwiGLU: x [ntok][cols], top_idx [ntok][k],
 * hid [ntok][k*ff]. gguf_quant and the fail-closed contract are exactly
 * fc_moe_gateup_ctx's; per token the result is bit-exact with it. */
int fc_moe_gateup_nx_ctx(fc_cuda_ctx *ctx, int gguf_quant, fc_buffer *gate_all,
                         fc_buffer *up_all, fc_buffer *x, fc_buffer *top_idx,
                         fc_buffer *hid, int k, int ff, int cols,
                         int row_stride, int ntok);

/* Q3_K-pinned shorthand for fc_moe_gateup_nx_ctx. */
void fc_moe_gateup_q3k_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                              fc_buffer *up_all, fc_buffer *x,
                              fc_buffer *top_idx, fc_buffer *hid, int k,
                              int ff, int cols, int row_stride, int ntok);

/* Expert-union dedup for the batched gate+up. Q3_K GATE/UP ONLY — there is no
 * Q4_K twin of these kernels (dedup is an opt-in, prefill-only speed lever, not
 * a format the engine must support), so a caller with Q4_K experts must keep to
 * fc_moe_gateup_nx_ctx. fc_moe_dedup_build_ctx
 * turns top_idx [ntok*k] into per-expert request lists (ecount [n_expert],
 * ereq [n_expert*maxr]); fc_moe_gateup_q3k_dedup_ctx then runs the grouped
 * gate+up over the union, reading each unique expert's weights once and
 * applying them to every routing token. hid is bit-exact with the _nx path,
 * so the existing fc_moe_down_accum_nx_ctx consumes it unchanged. maxr must
 * be >= the max per-expert request count (== ntok).
 *
 * Both dedup calls are no-ops when their kernels are missing (an
 * fc_cuda_ptx.h predating them), which would leave hid UNWRITTEN rather than
 * wrong-but-plausible. Callers must gate on fc_moe_dedup_available_ctx and
 * fall back to fc_moe_gateup_q3k_nx_ctx.
 *
 * FC_MOE_DEDUP_SHX=1 (default off) swaps in fc_moe_gateup_q3k_dedup_shx,
 * which stages x in shared like the _nx kernel and keeps the block-uniform
 * request list in shared instead of 16 per-thread registers: 62 registers and
 * 8 blocks/SMX against 95 and 5, and bit-exact with both other paths.
 * Measured +20% on the kernel at ntok=128 / n_expert=256. */
int fc_moe_dedup_available_ctx(fc_cuda_ctx *ctx);
/* Q8_0 expert-grouped gate+up (prefill) over the compacted work list
 * (fc_moe_dedup_build_ctx, then fc_moe_dedup_build_wl_ctx with
 * wmax = fc_moe_dedup_wmax), bit-exact with fc_moe_gateup_nx_ctx on Q8_0
 * experts. Gate on fc_moe_dedup_q80_available_ctx; the launcher returns -1 if
 * it did not run (hid then unwritten — fall back to fc_moe_gateup_nx_ctx). */
int fc_moe_dedup_q80_available_ctx(fc_cuda_ctx *ctx);
int fc_moe_gateup_q80_dedup_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                                fc_buffer *up_all, fc_buffer *x,
                                fc_buffer *ereq, fc_buffer *wlist,
                                fc_buffer *hid, int k, int ff, int cols,
                                int row_stride, int wmax, int maxr);
void fc_moe_dedup_build_ctx(fc_cuda_ctx *ctx, fc_buffer *top_idx,
                            fc_buffer *ecount, fc_buffer *ereq, int n_expert,
                            int k, int ntok, int maxr);
void fc_moe_gateup_q3k_dedup_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                                 fc_buffer *up_all, fc_buffer *x,
                                 fc_buffer *ecount, fc_buffer *ereq,
                                 fc_buffer *hid, int k, int ff, int cols,
                                 int row_stride, int n_expert, int maxr);

/* ---- COMPACTED-GRID dedup gate+up (FC_MOE_DEDUP_COMPACT, default OFF) ----
 *
 * The uncompacted launch above is (ceil(ff/8), n_expert, ceil(maxr/RT)) =
 * (64, 256, 16) = 262,144 blocks at the shipped prefill shape, of which only
 * 6.3 % do any work; MEASURED, the empty ones cost 18.6 % of the kernel.
 * The compacted path folds
 * (expert, request-tile) onto ONE grid axis over a device-built work list and
 * launches grid.y at fc_moe_dedup_wmax's exact upper bound — 352 instead of
 * 4096, i.e. 22,528 blocks — with NO device->host sync. Bit-exact: same work
 * items, same kernel body, one plain store per output, so block order is not
 * observable (the full argument is in moe_expert_group.cu).
 *
 * Sequence per MoE block, replacing the two calls above:
 *   fc_moe_dedup_build_ctx      (UNCHANGED — still builds ecount/ereq)
 *   fc_moe_dedup_build_wl_ctx   (compacts ecount into wlist)
 *   fc_moe_gateup_q3k_dedup_compact_ctx
 *
 * Both new kernels exist only in a
 *   cd cuda_kernels && make NVCCFLAGS_EXTRA=-DFC_MOE_DEDUP_COMPACT=1
 * build of the PTX, so with the shipped header fc_moe_dedup_compact_enabled_ctx
 * is 0 whatever the environment says and callers keep the uncompacted launch.
 */

/* Exact host-side upper bound on the number of (expert, request-tile) work
 * items — the grid.y a compacted launch must cover, and the int2 element count
 * the wlist buffer must hold. W = sum over routed experts of
 * ceil(R_e / FC_MOE_DEDUP_RT), and each of these three bounds is individually
 * exact, with T = ntok*k requests and U = min(n_expert, T) routed experts:
 *   1. W <= (T + (RT-1)*U) / RT  — ATTAINED: at T=1024, U=256 a router giving
 *      160 experts R=1 and 96 experts R=9 produces 160 + 192 = 352 tiles, so
 *      this bound cannot be tightened without knowing the router.
 *   2. W <= U * ceil(maxr/RT)    — an expert's list is at most maxr long
 *   3. W <= T                    — every work item holds at least one request
 * At the shipped prefill shape (ntok=128, k=8, n_expert=256, maxr=128) that is
 * min(352, 4096, 1024) = 352 against the uncompacted grid.y*grid.z of 4096.
 * Returns 0 when the shape cannot be compacted (n_expert or maxr outside the
 * work-item encoding / the builder's scan capacity), which callers must treat as
 * "no compaction available". Pure function; no ctx, no device. */
int fc_moe_dedup_wmax(int n_expert, int k, int ntok, int maxr);
/* Both compaction kernels present in the loaded PTX? */
int fc_moe_dedup_compact_available_ctx(fc_cuda_ctx *ctx);
/* ...and FC_MOE_DEDUP_COMPACT=1 in the environment. Default OFF. */
int fc_moe_dedup_compact_enabled_ctx(fc_cuda_ctx *ctx);
void fc_moe_dedup_build_wl_ctx(fc_cuda_ctx *ctx, fc_buffer *ecount,
                               fc_buffer *wlist, int n_expert, int wmax);
void fc_moe_gateup_q3k_dedup_compact_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_all,
                                         fc_buffer *up_all, fc_buffer *x,
                                         fc_buffer *ereq, fc_buffer *wlist,
                                         fc_buffer *hid, int k, int ff,
                                         int cols, int row_stride, int wmax,
                                         int maxr);

/* Batched Q8_0 MoE weight loads: __ldg up to this many tokens per call (the
 * slot-decode widths), plain loads above it (prefill chunks, where tokens share
 * experts). Bit-identical either way. FC_MOE_Q80_LDG_MAX_TOK=<n> overrides per
 * call; 0 = always plain. A header without the plain _nxp kernels keeps __ldg. */
#define FC_MOE_Q80_LDG_MAX_TOK_DEFAULT 8
int fc_moe_q80_ldg_max_tok(void);

/* Batched grouped down + weighted accumulate: hid [ntok][k*ff],
 * top_idx/top_wt [ntok][k], out [ntok][rows] (WRITTEN). */
void fc_moe_down_accum_nx_ctx(fc_cuda_ctx *ctx, int gguf_quant,
                              fc_buffer *down_all, fc_buffer *hid,
                              fc_buffer *top_idx, fc_buffer *top_wt,
                              fc_buffer *out, int k, int rows, int ff,
                              int row_stride, int ntok);

/* Batched shared-expert scalar gate: x/shexp [ntok][n], per token
 * shexp[t] *= sigmoid(gate_w . x[t]). */
void fc_moe_shexp_dot_gate_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_w,
                                  fc_buffer *x, fc_buffer *shexp, int n,
                                  int ntok);

/* Shared-expert gate + combine + residual, one launch per call: per token
 * shexp *= sigmoid(gate_w . x); block_out = moe_out + shexp; hidden +=
 * block_out. BIT-EXACT vs fc_moe_shexp_dot_gate_nx + fc_add + fc_add.
 * Returns 0 when launched, -1 when unavailable (caller runs the three). */
int fc_moe_shexp_gate_combine_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *gate_w,
                                     fc_buffer *x, fc_buffer *shexp,
                                     fc_buffer *moe_out, fc_buffer *block_out,
                                     fc_buffer *hidden, int n, int ntok);

/* SSM conv1d + SiLU SCAN over npos positions: qkv/out [npos][conv_dim],
 * conv_state rolled forward through all npos (updated in place). Requires
 * conv_kernel <= 4. snaps (optional, NULL = off): per-position conv_state
 * snapshots [npos][conv_dim*(conv_kernel-1)] — snaps[t] = state after
 * position t, for speculative-decode partial-accept rollback. */
void fc_ssm_conv1d_silu_seq_ctx(fc_cuda_ctx *ctx, fc_buffer *qkv,
                                fc_buffer *weight, fc_buffer *conv_state,
                                fc_buffer *out, int conv_dim, int conv_kernel,
                                int npos, fc_buffer *snaps);

/* SSM delta-rule recurrence SCAN over npos positions: q/k/v/y
 * [npos][n_v_heads*S], g/beta [npos][n_v_heads], state s updated in place
 * through all npos. snaps (optional, NULL = off): per-position state
 * snapshots [npos][n_v_heads*S*S] — snaps[t] = state after position t, for
 * speculative-decode partial-accept rollback. */
void fc_ssm_recurrence_seq_ctx(fc_cuda_ctx *ctx, fc_buffer *s, fc_buffer *q,
                               fc_buffer *k, fc_buffer *v, fc_buffer *g,
                               fc_buffer *beta, fc_buffer *y,
                               int n_v_heads, int state_size, int npos,
                               float scale, fc_buffer *snaps);

/* Batched decay gate: alpha/g [npos][n_heads], dt/a broadcast [n_heads]. */
void fc_ssm_decay_gate_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *alpha,
                              fc_buffer *ssm_dt, fc_buffer *ssm_a,
                              fc_buffer *g, int n_heads, int npos);

/* Batched q/k L2-norm + GQA repeat (+ v passthrough): conv_out
 * [npos][conv_dim], qg/kg/vg [npos][val_dim]. */
void fc_ssm_qknorm_gqa_nx_ctx(fc_cuda_ctx *ctx, fc_buffer *conv_out,
                              fc_buffer *qg, fc_buffer *kg, fc_buffer *vg,
                              int n_k_heads, int kv_groups, int state_size,
                              int key_dim, int val_dim, float eps, int npos);

/* ================================================================
 * On-die causal GQA flash attention (attention.cu)
 *
 * Replaces the host-side O(N^2) causal softmax of the qwen35moe
 * batched prefill: the KV cache lives on the die ([n_head_kv][max_seq]
 * [head_dim], the host cache's exact layout) and one launch runs the
 * tiled QK^T -> online-softmax -> PV pipeline for all n query rows.
 * Online softmax reduces in tile order (running max + running sum), so
 * outputs drift ~1e-6 relative vs the host single-pass softmax — same
 * math, different reduction order, NOT bit-exact by design.
 * ================================================================ */

/* 1 if this ctx has the full flash-attention kernel set loaded (callers
 * pick the host-softmax path otherwise). */
int fc_attn_flash_available_ctx(fc_cuda_ctx *ctx);

/* Split the joint Q+gate projection: qg [n][n_head][2*head_dim] (per head
 * [q | z]) into contiguous q and z, each [n][n_head][head_dim]. */
void fc_attn_qz_split_ctx(fc_cuda_ctx *ctx, fc_buffer *qg, fc_buffer *q,
                          fc_buffer *z, int n_head, int head_dim, int npos);

/* Append npos roped K/V rows to the GPU ring KV cache: k_new/v_new
 * [npos][n_head_kv][head_dim], caches [n_head_kv][max_seq][head_dim],
 * row t landing at (pos0 + t) % max_seq per head. */
void fc_attn_kv_append_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                           fc_buffer *v_new, fc_buffer *k_cache,
                           fc_buffer *v_cache, int n_head_kv, int head_dim,
                           int max_seq, int pos0, int npos);

/* Causal GQA flash attention (prefill): q/out [npos][n_head][head_dim],
 * caches as above with rows [0, pos0+npos) live; query row t attends
 * cache rows [0, pos0+t]. Requires pos0+npos <= max_seq (no ring wrap),
 * head_dim a multiple of 16 (<= 1024) and n_head/n_head_kv <= 8.
 * Returns 0 on launch, -1 if unavailable/unsupported (caller falls back
 * to its host path). */
/* attn_sink/attn_win drive StreamingLLM sliding-window attention: when
 * attn_win > 0 each query attends only [0, attn_sink) U the last attn_win keys
 * (prefill O(N^2) -> O(N*W)); attn_win <= 0 is full attention (byte-identical
 * to the pre-window kernel). Orthogonal to the Q4 sink/win fp16-shadow. */
int fc_attn_flash_prefill_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                              fc_buffer *k_cache, fc_buffer *v_cache,
                              fc_buffer *out, int n_head, int n_head_kv,
                              int head_dim, int max_seq, int pos0, int npos,
                              int attn_sink, int attn_win, float scale);

/* ---- TurboQuant Q4 KV cache (QWEN35_KV_Q4) ------------------------------
 * The cache is stored ggml-Q4_0 (32-element blocks, one fp16 scale each,
 * ~7x smaller than fp32) and dequantized in-kernel by the flash path — the
 * lever for the model's full 256K context on one card. Byte layout matches
 * Pantheon's pn_block_q4_0 so host<->device copies are byte identical. Row
 * bytes = (head_dim/32)*18; head_dim must be a multiple of 32. */

/* 1 if this ctx has both Q4 KV kernels loaded. */
int fc_attn_flash_q4_available_ctx(fc_cuda_ctx *ctx);

/* Append + quantize npos roped K/V rows into the Q4 ring cache, and mirror
 * each row into the fp16 sink/recent-window shadow so greedy argmax stops
 * flipping. k_new/v_new fp32 [npos][n_head_kv][head_dim]; caches
 * [n_head_kv][max_seq][row_bytes]; k/v_sink fp16 [n_head_kv][SINKMAX][head_dim];
 * k/v_win fp16 [n_head_kv][WMAX][head_dim] ring. sink/win = live sizes. */
void fc_attn_kv_append_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                              fc_buffer *v_new, fc_buffer *k_cache,
                              fc_buffer *v_cache, fc_buffer *k_sink,
                              fc_buffer *v_sink, fc_buffer *k_win,
                              fc_buffer *v_win, int n_head_kv, int head_dim,
                              int max_seq, int pos0, int npos, int sink,
                              int win);

/* Causal GQA flash attention over the mixed Q4 + fp16-shadow cache (fp16 for
 * sink/recent keys, in-kernel q4 dequant for the middle). Same contract/returns
 * as fc_attn_flash_prefill_ctx. */
int fc_attn_flash_prefill_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                 fc_buffer *k_cache, fc_buffer *v_cache,
                                 fc_buffer *k_sink, fc_buffer *v_sink,
                                 fc_buffer *k_win, fc_buffer *v_win,
                                 fc_buffer *out, int n_head, int n_head_kv,
                                 int head_dim, int max_seq, int pos0, int npos,
                                 int sink, int win, int attn_sink, int attn_win,
                                 float scale);

/* ================================================================
 * Decode-mode split flash attention (QWEN35_FLASH_DECODE)
 *
 * A single decode query attends the full cached context [0, seq_len). The KV
 * sequence is split across n_splits blocks (grid (n_splits, n_head_kv)) so the
 * die is not left near-idle by the npos=1 prefill launch; each block writes a
 * partial {m, l, acc} to scratch and a combine pass merges them per head. The
 * combine is fused into the same ctx-stream launch, so ..._ctx returns once
 * both kernels are queued.
 *
 * Scratch sizing (caller-owned, indexed by global head qh = kvh*kv_groups+g):
 *   m_scr, l_scr : >= n_head * n_splits  floats
 *   acc_scr      : >= n_head * n_splits * head_dim  floats
 * out: [n_head][head_dim]. Returns 0 on success, -1 on a contract violation /
 * launch failure. */
int fc_attn_flash_decode_available_ctx(fc_cuda_ctx *ctx);
int fc_attn_flash_decode_q4_available_ctx(fc_cuda_ctx *ctx);

int fc_attn_flash_decode_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                             fc_buffer *k_cache, fc_buffer *v_cache,
                             fc_buffer *m_scr, fc_buffer *l_scr,
                             fc_buffer *acc_scr, fc_buffer *out, int n_head,
                             int n_head_kv, int head_dim, int max_seq,
                             int seq_len, int n_splits, int split_len,
                             int attn_sink, int attn_win, float scale);

int fc_attn_flash_decode_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                fc_buffer *k_cache, fc_buffer *v_cache,
                                fc_buffer *k_sink, fc_buffer *v_sink,
                                fc_buffer *k_win, fc_buffer *v_win,
                                fc_buffer *m_scr, fc_buffer *l_scr,
                                fc_buffer *acc_scr, fc_buffer *out, int n_head,
                                int n_head_kv, int head_dim, int max_seq,
                                int seq_len, int n_splits, int split_len,
                                int sink, int win, int attn_sink, int attn_win,
                                float scale);

/* ---- Device-position decode variants (CUDA-graph capture) ---------------
 * Same math as the host-arg append/decode calls above, but the token position
 * is read from `pos` (an int32 DEVICE buffer, pos[0] = current position;
 * seq_len = pos[0] + 1) and the split plan (n_splits/split_len) is recomputed
 * ON-DEVICE from kps/splits_cap with the identical formula the caller uses —
 * so every launch argument, buffer address and grid dim is position-invariant
 * and the whole decode attention block can live inside a captured CUDA graph:
 * the host updates pos[0] once per token (outside the graph) and replays.
 * Bit-exact per token to the host-arg variants. The decode launch uses a
 * fixed grid of splits_cap blocks; dead blocks exit immediately. Scratch must
 * be sized for splits_cap. Caller guarantees no ring wrap
 * (pos[0] + 1 <= max_seq). Kernels are OPTIONAL in the PTX blob (older
 * fc_cuda_ptx.h): probe with fc_attn_decode_graph_available_ctx before use. */

/* 1 if every kernel the graph-captured decode-attention block needs is
 * loaded (qz split, QK norm, RoPE-from-device-pos, sigmoid/mul glue, the
 * *_dev append + split + combine for the given cache mode q4 = 0/1). */
int fc_attn_decode_graph_available_ctx(fc_cuda_ctx *ctx, int q4);

void fc_attn_kv_append_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                               fc_buffer *v_new, fc_buffer *k_cache,
                               fc_buffer *v_cache, int n_head_kv,
                               int head_dim, int max_seq, fc_buffer *pos);

void fc_attn_kv_append_q4_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                                  fc_buffer *v_new, fc_buffer *k_cache,
                                  fc_buffer *v_cache, fc_buffer *k_sink,
                                  fc_buffer *v_sink, fc_buffer *k_win,
                                  fc_buffer *v_win, int n_head_kv,
                                  int head_dim, int max_seq, fc_buffer *pos,
                                  int sink, int win);

int fc_attn_flash_decode_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                 fc_buffer *k_cache, fc_buffer *v_cache,
                                 fc_buffer *m_scr, fc_buffer *l_scr,
                                 fc_buffer *acc_scr, fc_buffer *out,
                                 int n_head, int n_head_kv, int head_dim,
                                 int max_seq, fc_buffer *pos, int kps,
                                 int splits_cap, int attn_sink, int attn_win,
                                 float scale);

int fc_attn_flash_decode_q4_dev_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                    fc_buffer *k_cache, fc_buffer *v_cache,
                                    fc_buffer *k_sink, fc_buffer *v_sink,
                                    fc_buffer *k_win, fc_buffer *v_win,
                                    fc_buffer *m_scr, fc_buffer *l_scr,
                                    fc_buffer *acc_scr, fc_buffer *out,
                                    int n_head, int n_head_kv, int head_dim,
                                    int max_seq, fc_buffer *pos, int kps,
                                    int splits_cap, int sink, int win,
                                    int attn_sink, int attn_win, float scale);

/* ---- Multi-user decode slot attention (L7) ------------------------------
 * K users, one decode token each at its OWN position, against PER-SLOT KV
 * rings laid out [slot][n_head_kv][max_seq][row] (each slot slice
 * byte-identical to the single-user ring layout). `pos` is the int32 DEVICE
 * position array (the RoPE position buffer, pos[u] = slot u's position);
 * every user's split plan is recomputed on-device from its own position with
 * the single-user formula (kps/splits_cap), so each user's attention output
 * is bit-identical to fc_attn_flash_decode(_q4) at the same position.
 * launch_splits must be >= every user's plan (the caller takes the max of
 * the per-user host plans); partial scratch is strided by launch_splits:
 *   m_scr, l_scr : >= nslots * n_head * launch_splits floats
 *   acc_scr      : >= nslots * n_head * launch_splits * head_dim floats
 * q: [nslots][n_head][head_dim]; out: [nslots][n_head][head_dim].
 * Caller guarantees no ring wrap (pos[u] + 1 <= max_seq for every slot).
 * `rows` (optional int32 DEVICE buffer, NULL = identity) maps launch position
 * u to the ring row it owns, so a non-contiguous set of persistent rows can
 * decode in one launch with no data movement; q / partials / out stay indexed
 * by u. The same contract applies to every slot call below.
 * Kernels are OPTIONAL in the PTX blob (older fc_cuda_ptx.h): probe with the
 * availability calls below. */
int fc_attn_flash_decode_slots_available_ctx(fc_cuda_ctx *ctx);
int fc_attn_flash_decode_slots_q4_available_ctx(fc_cuda_ctx *ctx);

/* Append each slot's single roped K/V row ([nslots][n_head_kv][head_dim]) to
 * its own ring at pos[u] (q4: quantized in-kernel + per-slot fp16 sink /
 * recent-window shadow mirror, matching the host pn_kvq4_store_row bytes). */
void fc_attn_kv_append_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                                 fc_buffer *v_new, fc_buffer *k_cache,
                                 fc_buffer *v_cache, int n_head_kv,
                                 int head_dim, int max_seq, fc_buffer *pos,
                                 fc_buffer *rows, int nslots);

void fc_attn_kv_append_slots_q4_ctx(fc_cuda_ctx *ctx, fc_buffer *k_new,
                                    fc_buffer *v_new, fc_buffer *k_cache,
                                    fc_buffer *v_cache, fc_buffer *k_sink,
                                    fc_buffer *v_sink, fc_buffer *k_win,
                                    fc_buffer *v_win, int n_head_kv,
                                    int head_dim, int max_seq, fc_buffer *pos,
                                    fc_buffer *rows, int nslots, int sink,
                                    int win);

int fc_attn_flash_decode_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *q,
                                   fc_buffer *k_cache, fc_buffer *v_cache,
                                   fc_buffer *m_scr, fc_buffer *l_scr,
                                   fc_buffer *acc_scr, fc_buffer *out,
                                   int n_head, int n_head_kv, int head_dim,
                                   int max_seq, fc_buffer *pos,
                                   fc_buffer *rows, int nslots,
                                   int launch_splits, int kps, int splits_cap,
                                   int attn_sink, int attn_win, float scale);

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
                                      int attn_sink, int attn_win, float scale);

/**
 * Select the decode slots attention variant for this process: 1 = the
 * fewer-instruction kernel (fc_attn_flash_decode_slots_q4_kvf, bit-exact with
 * kvs), 0 = kvs. Overrides FC_ATTN_DEC_KVF. For tests and benches that compare
 * both in one process. Returns the previous setting.
 * Thread-safe: no. Reentrant: no. Signal-safe: no.
 */
int fc_attn_dec_kvf_set(int on);

/** 1 when ctx loaded the kvf kernel (a PTX header older than it lacks the
 * symbol, and the slots call then silently runs kvs). Thread-safe: yes. */
int fc_attn_dec_kvf_available_ctx(fc_cuda_ctx *ctx);

/* fc_attn_flash_decode_slots_q4_ctx with the sigmoid output gate folded into
 * the combine (out = attention * sigmoid(z), z read only). BIT-EXACT vs the
 * ungated call + fc_sigmoid + fc_mul. Returns -1 when unavailable. */
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
                                      fc_buffer *z);

/* ================================================================
 * Speculative-verify decode attention — R consecutive rows of ONE session
 *
 * Rows r = 0..nrows-1 sit at positions pos0 + r and attend the session's
 * single-user Q4 ring over [0, pos0 + r] (the rows' own K/V already appended).
 * Split-K over positions like the decode kernels, but each block reads a K/V
 * tile once for up to FC_ATTN_ROWS_MAX rows per block. BIT-EXACT per row against
 * fc_attn_flash_decode_slots_q4_ctx at that row's position (row grouping by
 * split plan, per-row shadow choice: see cuda_kernels/attention_decode_rows.cuh).
 * q, z, out: [nrows][n_head][head_dim]. z != NULL folds the sigmoid output
 * gate into the combine (as the _gated slots call). m/l/acc scratch: at least
 * nrows * n_head * splits_cap (acc: x head_dim) floats.
 * Contract: head_dim 256, attn_win <= 0, 1 <= nrows <= FC_ATTN_ROWS_MAX,
 * pos0 + nrows <= max_seq. Returns 0, or -1 (contract / kernel missing:
 * caller uses another path).
 * FC_ATTN_ROWS_PER_BLOCK (default 1) sets the rows a block carries: 1 reads
 * K/V once per row (fastest on the K80: latency-bound), up to 5 reads it once
 * per verify.
 * ================================================================ */
#define FC_ATTN_ROWS_MAX 5
int fc_attn_flash_decode_rows_q4_available_ctx(fc_cuda_ctx *ctx);
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
                                     int attn_win, float scale);

/* fc_attn_flash_decode_rows_q4_ctx with the position on the DEVICE (pos[0] =
 * row 0's position; the rows are pos[0] .. pos[0]+nrows-1), so a CUDA graph can
 * replay it at any position: each block derives its row's split plan on the
 * device. launch_splits (the grid width, = the host plan's max n_splits over the
 * rows) is the only position-derived argument; it changes once per kps tokens.
 * Bytes identical to fc_attn_flash_decode_rows_q4_ctx. Returns 0 or -1. */
int fc_attn_flash_decode_rows_q4_dev_available_ctx(fc_cuda_ctx *ctx);
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
                                         float scale);

/* ================================================================
 * Multi-user decode (slot) kernels — K independent sequences, ONE
 * token each, per launch
 *
 * The batched-decode counterparts of the _seq scans: slot u owns its
 * OWN state image (slot-major state buffers) and advances exactly one
 * step, with per-slot math bit-exact to the single-token kernels —
 * batching K users changes throughput, not any user's output.
 * Activations are slot-major [K][dim], matching the batch layout.
 * ================================================================ */

/* Conv1d + SiLU, one step per slot: qkv/out [nslots][conv_dim],
 * conv_state [rows][conv_dim*(conv_kernel-1)] (each slot's slice
 * updated in place). `rows` (int32 DEVICE, NULL = identity) maps launch
 * position u to its persistent state row, as for every SSM slot call. */
void fc_ssm_conv1d_silu_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *qkv,
                                  fc_buffer *weight, fc_buffer *conv_state,
                                  fc_buffer *out, int conv_dim,
                                  int conv_kernel, int nslots,
                                  fc_buffer *rows);

/* Delta-rule recurrence, one step per slot: s [nslots][n_v_heads*S*S]
 * (each slot's state updated in place), q/k/v/y [nslots][n_v_heads*S],
 * g/beta [nslots][n_v_heads]. */
void fc_ssm_recurrence_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *s, fc_buffer *q,
                                 fc_buffer *k, fc_buffer *v, fc_buffer *g,
                                 fc_buffer *beta, fc_buffer *y,
                                 int n_v_heads, int state_size, int nslots,
                                 float scale, fc_buffer *rows);

/* Fused single-step GDN for the slots path: beta sigmoid + decay gate +
 * delta-rule recurrence in one warp-per-row launch, BIT-EXACT vs
 * fc_sigmoid + fc_ssm_decay_gate_nx + fc_ssm_recurrence_slots. alpha and
 * beta_raw are the RAW projections [nslots][n_v_heads] (neither is written).
 * Returns 0 when launched, -1 when the kernel is absent or state_size != 128
 * (the caller then runs the three-launch chain). */
int fc_ssm_gdn_step_slots_ctx(fc_cuda_ctx *ctx, fc_buffer *s, fc_buffer *q,
                              fc_buffer *k, fc_buffer *v, fc_buffer *alpha,
                              fc_buffer *ssm_dt, fc_buffer *ssm_a,
                              fc_buffer *beta_raw, fc_buffer *y,
                              int n_v_heads, int state_size, int nslots,
                              float scale, fc_buffer *rows);

/* Partial NEOX RoPE with a PER-ROW position: data [nrows][n_heads*head_dim]
 * rotated in place, pos an int32 DEVICE buffer of nrows entries (row t at
 * position pos[t]) — each user decodes at its own context length. */
void fc_rope_neox_partial_pos_ctx(fc_cuda_ctx *ctx, fc_buffer *data,
                                  int n_heads, int head_dim, int rope_dim,
                                  fc_buffer *pos, float theta_base,
                                  int nrows);

/* ================================================================
 * CUDA graph capture / replay
 *
 * Decode on the K80 is launch-bound: each cuLaunchKernel costs far more
 * host time than the kernel it starts. A captured graph replays a whole
 * recorded launch sequence with ONE cuGraphLaunch. Bit-exact by
 * construction — the graph re-issues the exact kernels, grids and args
 * that were recorded, so the device does precisely the same work.
 *
 * Contract: between begin and end, issue ONLY stream-ordered ops on this
 * ctx's stream (kernel launches, async copies). Synchronous ops
 * (fc_sync_ctx, fc_buffer_to_cpu/from_cpu) invalidate the capture. The
 * recorded launches DO NOT EXECUTE during capture; on a NULL return from
 * fc_graph_end_ctx the caller must re-issue the sequence directly.
 * Replay requires every recorded buffer address and grid dim to still be
 * valid — capture only position-invariant sequences over persistent
 * buffers.
 *
 * Thread scope: capture is THREAD_LOCAL, so the contract above binds the
 * CAPTURING THREAD only. Other threads may allocate, free, sync and run
 * their own dies normally while a capture is open — and may hold their own
 * captures concurrently. (This was not true when capture used GLOBAL mode:
 * any thread's unsafe call anywhere in the process killed the capture.)
 * ================================================================ */

typedef struct fc_graph fc_graph;

/* Begin capturing ctx's stream. Returns 0 on success, -1 if graphs are
 * unavailable (old driver / missing symbols) — caller keeps its direct
 * launch path. */
int fc_graph_begin_ctx(fc_cuda_ctx *ctx);

/* End capture and instantiate the executable graph. NULL on failure (the
 * recorded launches never ran — re-issue them directly). */
fc_graph *fc_graph_end_ctx(fc_cuda_ctx *ctx);

/* Replay a captured graph on its owning ctx's stream (async, stream-
 * ordered like any launch). Returns 0 on success. */
int fc_graph_launch(fc_graph *g);

void fc_graph_free(fc_graph *g);

/* Node census of a captured graph, taken at instantiation: the count of nodes of
 * CUgraphNodeType `type` (0 kernel, 1 memcpy, 2 memset, 5 empty, 6 event wait,
 * 7 event record), or of all nodes for type < 0. 0 when unknown. */
#define FC_GRAPH_NODE_TYPES 12
size_t fc_graph_node_count(const fc_graph *g, int type);

/* ================================================================
 * CUDA events — non-blocking completion
 *
 * fc_sync_ctx blocks the calling host thread (the driver spins or
 * yields) until the stream drains. The async serve scheduler instead
 * records an event after launching a card's step and POLLS it, so one
 * 2-core host keeps 8 cards' streams in flight simultaneously: the
 * waiter sleeps between polls instead of burning a core per card.
 * ================================================================ */

typedef struct fc_event fc_event;

/* Create a (timing-disabled) event owned by ctx. NULL if events are
 * unavailable (missing driver symbols / ctx not ready) — caller falls
 * back to fc_sync_ctx. */
fc_event *fc_event_create_ctx(fc_cuda_ctx *ctx);

/* Record ev on ctx's stream: the event completes once all work queued
 * on the stream BEFORE this call has finished. ev must have been
 * created on the same ctx. Returns 0 on success. */
int fc_event_record_ctx(fc_cuda_ctx *ctx, fc_event *ev);

/* Non-blocking completion probe. Returns 0 = the recorded point has
 * completed, 1 = still running, -1 = error/unavailable. Never blocks. */
int fc_event_query(fc_event *ev);

/* Block the calling host thread until ev's recorded point completes
 * (cuEventSynchronize). Narrower than fc_sync_ctx, which drains the WHOLE
 * stream: this waits only for work queued before the record. Returns 0 on
 * success, -1 if events are unavailable.
 *
 * Thread-safe: yes (enters/leaves the owning context)
 * Reentrant: no
 * Signal-safe: no */
int fc_event_sync(fc_event *ev);

/* CROSS-DIE ORDERING WITHOUT THE HOST. Make `waiter`'s stream block until ev
 * completes (cuStreamWaitEvent). ev may have been recorded on a DIFFERENT
 * device's context — that is the point: every other ordering primitive here
 * (fc_sync_ctx, fc_event_query polling) routes a die-to-die dependency through
 * a host round-trip, and this one does not. The call itself returns
 * immediately; it enqueues a wait node, it does not block the host.
 *
 * Pair with fc_buffer_copy_peer_async to build a die-to-die producer/consumer
 * hand-off that costs no host synchronisation:
 *     fc_buffer_copy_peer_async(dst_on_b, src_on_a, n, ctx_b);
 *     fc_event_record_ctx(ctx_b, ev_b);
 *     fc_event_wait_ctx(ctx_a, ev_b);     // die A's stream waits on die B
 *
 * Returns 0 on success, -1 if cuStreamWaitEvent is unavailable (old driver) or
 * either handle is unusable.
 *
 * Thread-safe: yes (enters/leaves the waiter's context)
 * Reentrant: no
 * Signal-safe: no */
int fc_event_wait_ctx(fc_cuda_ctx *waiter, fc_event *ev);

void fc_event_free(fc_event *ev);

/* Timing events: a timing-enabled fc_event, for measuring GPU time between two points
 * of one stream (free with fc_event_free; fc_event_sync works on it). Recorded while
 * ctx's stream is being captured, it becomes an event-record node that every replay of
 * the graph re-records. NULL when the driver lacks the timing symbols.
 * fc_tevent_elapsed_ms: ms between two completed records of one ctx's events; returns
 * 0, or -1 (different contexts, not yet complete, or unavailable). */
fc_event *fc_tevent_create_ctx(fc_cuda_ctx *ctx);
int fc_tevent_record_ctx(fc_cuda_ctx *ctx, fc_event *ev);
int fc_tevent_elapsed_ms(fc_event *from, fc_event *to, float *ms);

/* Call census: counts of the driver calls that cost host time, taken only while enabled
 * (off by default; off, each counted call pays one branch): kernel launches (direct, or captured into a graph), graph launches, explicit
 * synchronizes (stream, context, event), and memcpys by kind with their bytes. The blocking
 * HtoD / DtoH copies (h2d_sync, d2h_sync) also wait on the host. Read by snapshot and diff. */
typedef struct {
    unsigned long long launches, graph_launches, syncs;
    unsigned long long h2d_sync, h2d_async, h2d_bytes;
    unsigned long long d2h_sync, d2h_async, d2h_bytes;
    unsigned long long d2d, d2d_bytes, peer, peer_bytes;
} fc_call_census;
void fc_call_census_enable(int on);
void fc_call_census_read(fc_call_census *out);

/* ================================================================
 * Synchronization
 *
 * CUDA operations are asynchronous by default. Call fc_sync()
 * when CPU needs GPU results (e.g., before fc_buffer_to_cpu).
 * In practice, cuMemcpyDtoH already synchronizes, so fc_sync()
 * is mainly for explicit fence points.
 * ================================================================ */

/* Wait for all pending GPU operations on ctx's stream to complete. How the
 * thread waits (here and in fc_event_sync) follows FC_CUDA_WAIT, read once:
 * unset or "driver" = the driver's wait (the context's scheduling flag);
 * "spin" = poll the completion point; "yield" = poll with sched_yield;
 * "hybrid[:us]" = spin up to us microseconds (default 200), then the driver
 * wait; "sleep[:us]" = poll with a us microsecond sleep (default 20). Every
 * mode waits for the same point, so results are identical. */
void fc_sync_ctx(fc_cuda_ctx *ctx);

/* Wait for all pending GPU operations on the process-default context. */
void fc_sync(void);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_CUDA_H */
