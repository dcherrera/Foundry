/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_metal.h
 * @brief Metal GPU compute backend for Foundry — public C API (prefix fm_).
 *   Build flag USE_METAL=1. GPU tensor ops via Apple's Metal framework on a
 *   global singleton (MTLDevice + MTLCommandQueue + specialized pipeline cache),
 *   with unified memory (MTLResourceStorageModeShared) for zero-copy on Apple Silicon.
 *
 *   This header owns the backend-wide API — lifecycle, function-constant
 *   specialization, buffers, no-copy regions, dispatch thresholds, batching —
 *   and then includes one header per kernel family from metal/kernels/. Include
 *   this file and you have the whole backend; a family's own header documents
 *   only that family. Kernel families are added without touching this file.
 *
 * @status Buffers, no-copy mmap regions with offset binding, MPS GEMM/GEMV
 *   (single + batched), custom kernels and function-constant pipeline
 *   specialization all work.
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_H
#define FOUNDRY_METAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Opaque types
 * ================================================================ */

typedef struct fm_buffer fm_buffer;

/* ================================================================
 * Lifecycle
 * ================================================================ */

/* Initialize Metal device, command queue, and pipeline cache.
 * Returns 0 on success, -1 on failure (no Metal device). */
int fm_init(void);

/* Cleanup all Metal resources. */
void fm_free(void);

/* Returns true if Metal backend is initialized and ready. */
bool fm_available(void);

/**
 * @brief True when the custom-kernel shader library actually built.
 *
 * fm_available() only reports that a Metal device was initialized. Shader
 * compilation is non-fatal, so when it fails every custom kernel silently
 * becomes a no-op while fm_available() still returns true. All families share
 * one MSL translation unit, so one bad fragment disables the whole backend.
 *
 * Tests and benchmarks MUST gate on this: a no-op backend makes GPU-vs-GPU
 * comparisons agree on all-zeros and makes bandwidth numbers fictional.
 */
bool fm_shaders_available(void);

/* ================================================================
 * Function-constant pipeline specialization
 *
 * A Metal function constant is a value bound when the pipeline is
 * built, not when the kernel is dispatched.  Inside the kernel it is a
 * compile-time constant, so loop bounds specialized this way unroll and
 * the local arrays those loops index stay in registers instead of
 * spilling to thread-local memory.  That is why every shape-dependent
 * kernel in this backend takes its shape as a function constant rather
 * than as a buffer argument.
 *
 * One pipeline is compiled and cached per (kernel name + constant
 * values) combination.  Repeating a call with the same values reuses
 * the cached pipeline; changing any value builds and caches a new one.
 * ================================================================ */

/* Maximum number of distinct constant indices in one specialization set. */
#define FM_FC_MAX_VALUES 16

/* Type tag for a function-constant value.  Must match the declared type
 * of the corresponding [[function_constant(index)]] in the shader — a
 * mismatch makes pipeline creation fail. */
typedef enum fm_fc_type {
    FM_FC_TYPE_NONE  = 0,
    FM_FC_TYPE_BOOL  = 1,   /* MSL bool  */
    FM_FC_TYPE_SHORT = 2,   /* MSL short */
    FM_FC_TYPE_INT   = 3    /* MSL int   */
} fm_fc_type;

/* A set of function-constant values, keyed by the shader's
 * [[function_constant(index)]] index.
 *
 * Plain value type: declare it on the stack, copy it freely, nothing to
 * free.  Entries are kept sorted by index, so two sets holding the same
 * values produce the same cache key regardless of the order they were
 * set in.  Treat the fields as private — use the fm_fc_* accessors. */
typedef struct fm_fc_values {
    int      count;
    uint16_t index[FM_FC_MAX_VALUES];
    uint8_t  type[FM_FC_MAX_VALUES];    /* fm_fc_type */
    int32_t  value[FM_FC_MAX_VALUES];
} fm_fc_values;

/**
 * Empties a constant set.  Call this before the first fm_fc_set_* on a
 * set that was not zero-initialized.
 *
 * @param values  Set to clear (ignored if NULL)
 *
 * Thread-safe: yes (operates only on the caller's struct)
 * Reentrant: yes
 * Signal-safe: yes
 */
void fm_fc_reset(fm_fc_values *values);

/**
 * Sets the MSL `int` constant at `index`, replacing any previous value
 * at that index.
 *
 * @param values  Constant set to modify
 * @param index   The shader's [[function_constant(index)]] index
 * @param value   Value to bind at pipeline-build time
 * @return        0 on success, -1 if the set is full or arguments are invalid
 *
 * Thread-safe: yes (operates only on the caller's struct)
 * Reentrant: yes
 * Signal-safe: yes
 */
int fm_fc_set_int(fm_fc_values *values, unsigned index, int32_t value);

/* Sets the MSL `short` constant at `index`.  See fm_fc_set_int. */
int fm_fc_set_short(fm_fc_values *values, unsigned index, int16_t value);

/* Sets the MSL `bool` constant at `index`.  See fm_fc_set_int. */
int fm_fc_set_bool(fm_fc_values *values, unsigned index, bool value);

/**
 * Builds (or fetches from cache) the pipeline for `kernel_name`
 * specialized with `constants`, and returns it as an opaque handle.
 *
 * The handle is only meaningful for identity comparison: two calls with
 * equal constant sets return the same pointer, calls with different
 * constant sets return different pointers.  The cache owns it — do not
 * release it, and do not use it after fm_free().
 *
 * @param kernel_name  Kernel name as declared in the shader source
 * @param constants    Constant set, or NULL for the unspecialized pipeline
 * @return             Opaque cached pipeline handle, or NULL on failure
 *
 * Thread-safe: no (shares the global pipeline cache)
 * Reentrant: no
 * Signal-safe: no
 */
const void *fm_pipeline_for_constants(const char *kernel_name,
                                      const fm_fc_values *constants);

/* The diagnostic kernel that proves specialization reached the shader
 * compiler lives in metal/kernels/fm_kernel_probe.h. */

/* ================================================================
 * Buffers (unified memory, zero-copy on Apple Silicon)
 * ================================================================ */

/* Wrap existing CPU pointer as a Metal shared buffer (zero-copy if page-aligned).
 * If pointer is not page-aligned, allocates a new Metal buffer and copies data.
 * The caller retains ownership of ptr — do not free it while buffer is in use. */
fm_buffer* fm_buffer_from_ptr(void *ptr, size_t bytes);

/* Allocate a new Metal buffer with shared memory. */
fm_buffer* fm_buffer_alloc(size_t bytes);

/* Free a Metal buffer wrapper. Does NOT free the underlying data
 * if created via fm_buffer_from_ptr. */
void fm_buffer_free(fm_buffer *buf);

/* Get CPU-accessible pointer to buffer data. */
void* fm_buffer_data(fm_buffer *buf);

/* Get buffer size in bytes. */
size_t fm_buffer_size(fm_buffer *buf);

/* ================================================================
 * No-copy regions and offset binding
 *
 * fm_buffer_from_ptr copies whenever the pointer is not page-aligned,
 * which is almost always for a weight inside a GGUF: the file's tensor
 * data starts at an arbitrary offset and tensors are only 32-byte
 * aligned, so binding each weight as its own buffer means duplicating
 * the whole model.  That is unaffordable and defeats the point of
 * mmap'ing the file in the first place.
 *
 * The fix is to stop making a buffer per weight.  Wrap the whole mmap
 * once — its base IS page-aligned, so that single buffer genuinely
 * takes the no-copy path — and address each tensor as a byte offset
 * into it via fm_bufref.
 *
 * Measured on Apple M4 / macOS 26:
 *   - A buffer offset only has to satisfy the alignment of the type the
 *     kernel reads (1 for uchar, 4 for float, 16 for float4/half4).
 *     There is NO page-alignment requirement: offsets that are multiples
 *     of 32 but not of the 16 KiB page read correctly for every type,
 *     under both API and GPU validation.  GGUF's 32-byte tensor
 *     alignment therefore satisfies binding with room to spare.
 *   - maxBufferLength is 14302248960 bytes (13.32 GiB).  One buffer
 *     cannot cover a larger mapping, so fm_region splits an oversized
 *     region into overlapping no-copy segments.
 * ================================================================ */

/* Byte alignment a bind offset must satisfy: the strictest element
 * alignment any kernel here reads (float4/half4).  Not a page. */
#define FM_BUFFER_OFFSET_ALIGNMENT 16

/* A bindable location: a buffer plus a byte offset into it.  Plain value
 * type — copy it freely, nothing to free.  It does not own the buffer. */
typedef struct fm_bufref {
    fm_buffer *buffer;
    size_t     offset;
} fm_bufref;

/* Builds a reference to `offset` bytes into `buffer`. */
fm_bufref fm_bufref_at(fm_buffer *buffer, size_t offset);

/* FM_BUFFER_OFFSET_ALIGNMENT, for callers that need it at runtime. */
size_t fm_buffer_offset_alignment(void);

/* Largest single buffer this device supports, in bytes.  A mapping
 * larger than this needs fm_region. */
size_t fm_buffer_max_length(void);

/**
 * Wraps an already page-aligned mapping in ONE buffer without copying.
 *
 * Unlike fm_buffer_from_ptr this never falls back to a copy: if the
 * pointer is not page-aligned, the length exceeds fm_buffer_max_length(),
 * or Metal refuses the mapping, it fails and returns NULL.  That is
 * deliberate — a silent copy of a weight region is the bug this exists
 * to prevent.
 *
 * The caller keeps ownership of the mapping and must not unmap it while
 * the buffer lives.  Free with fm_buffer_free.
 *
 * @param mapped_base   Page-aligned base of the mapping (e.g. from mmap)
 * @param mapped_bytes  Length in bytes; rounded up to a page internally,
 *                      which is safe because mmap already rounds the
 *                      mapping up to a page boundary
 * @return              Buffer aliasing the mapping, or NULL (see above)
 *
 * Thread-safe: no (touches the global device)
 * Reentrant: no
 * Signal-safe: no
 */
fm_buffer *fm_buffer_map_no_copy(void *mapped_base, size_t mapped_bytes);

/* A mapping covered by one or more no-copy buffer segments, so that any
 * byte range inside it can be bound without copying. */
typedef struct fm_region fm_region;

/**
 * Covers a page-aligned mapping with no-copy buffer segments.
 *
 * A mapping that fits in one buffer gets exactly one segment.  A larger
 * one is covered by overlapping segments: consecutive segments start
 * `segment length - max_span_bytes` apart, so any range of at most
 * `max_span_bytes` lies wholly inside a single segment and never
 * straddles a boundary.  Segments alias the same pages — they cost
 * address space, not memory, and nothing is copied.
 *
 * Never copies.  Returns NULL if the base is not page-aligned, if
 * max_span_bytes leaves no room for forward progress, or if any segment
 * fails to map.
 *
 * @param mapped_base    Page-aligned base of the mapping (e.g. from mmap)
 * @param mapped_bytes   Length of the mapping in bytes
 * @param max_span_bytes Largest single range that will ever be bound —
 *                       the biggest tensor.  0 selects a 1 GiB default.
 * @return               Region handle, or NULL on failure
 *
 * Thread-safe: no
 * Reentrant: no
 * Signal-safe: no
 */
fm_region *fm_region_open(void *mapped_base, size_t mapped_bytes,
                          size_t max_span_bytes);

/* Releases the segment buffers.  Does not unmap the caller's mapping. */
void fm_region_close(fm_region *region);

/**
 * Pretends the device's maximum buffer length is `bytes`, so the
 * multi-segment path can be exercised without a mapping larger than the
 * real 13.32 GiB limit.  Affects fm_region_open only; 0 restores the
 * device limit.  Diagnostic hook for tests — production code has no
 * reason to call it.
 *
 * Thread-safe: no (global)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_region_force_segment_span(size_t bytes);

/* Number of segments covering the region (1 when it fits in one buffer). */
int fm_region_segment_count(const fm_region *region);

/**
 * Resolves a byte range in the region to a bindable buffer + offset.
 *
 * @param region  Region to address into
 * @param offset  Byte offset of the range from the mapping base
 * @param bytes   Length of the range
 * @param out     Receives the buffer and the offset within it
 * @return        0 on success; -1 if the range runs past the region, is
 *                larger than the region's max_span_bytes, or has an
 *                offset that violates FM_BUFFER_OFFSET_ALIGNMENT
 *
 * Thread-safe: yes (read-only once the region is open)
 * Reentrant: yes
 * Signal-safe: no (may log on failure)
 */
int fm_region_bind(const fm_region *region, size_t offset, size_t bytes,
                   fm_bufref *out);

/* ================================================================
 * Dispatch thresholds
 *
 * Backend-wide policy, not a kernel family: the CPU callers in tensor/
 * and rwkvx/ test an op's size against these before choosing Metal.
 * ================================================================ */

/* Dispatch threshold: ops below this stay on CPU.
 *
 * On Apple Silicon with unified memory, single-token decode (M=1) should
 * ALWAYS stay on CPU.  Single-vector operations are memory-bandwidth bound
 * and CPU/GPU share the same bus, so Metal can only add dispatch overhead
 * (~100-200us per kernel).  Metal wins for batched operations (prefill,
 * training) where M >= ~8-16 amortizes the overhead across many rows.
 *
 * GEMM: M*K*N.  For M=1, K=2560, N=2560 → 6.5M, well under threshold.
 *       For prefill M=32, 32*2560*2560 = 210M, above threshold → GPU.
 * GEMV: M*K.  Single-vector always stays on CPU.
 * WKV7: Only the sequence variant (fm_wkv7_seq) benefits from Metal.
 *       The per-token fm_wkv7_one always stays on CPU.
 * Q8 matvec: fast_matvec_q8_preq is inherently M=1, always CPU.
 * Elementwise: softmax/rms_norm/silu on single vectors stay on CPU. */
#define FM_GEMM_THRESHOLD     (128L * 1024 * 1024)  /* ~128M ops: Metal wins for batched prefill (N>=8) */
#define FM_Q8_GEMM_THRESHOLD  (128L * 1024 * 1024)  /* ~128M ops: Metal INT8 GEMM for batched prefill */
#define FM_GEMV_THRESHOLD  (4096L * 4096)
#define FM_WKV7_THRESHOLD  (INT64_MAX)    /* disabled: fm_wkv7_one is always per-token */
#define FM_Q8_THRESHOLD    (INT64_MAX)    /* disabled: fast_matvec_q8_preq is always M=1 */
#define FM_ELEM_THRESHOLD  (1024 * 1024)  /* 1M elements: only batch softmax/norm */

/* ================================================================
 * Synchronization & Command batching
 *
 * GPU dispatches auto-accumulate in a persistent command buffer
 * (MLX-style). Call fm_sync() when CPU needs GPU results.
 * ================================================================ */

/* Commit pending GPU work and wait for completion.
 * Call this at data dependency points where CPU needs GPU output. */
void fm_sync(void);

/* Enable/disable auto-batch mode.
 * When enabled, GPU dispatches accumulate until fm_sync() is called.
 * When disabled (default), each dispatch commits and waits immediately.
 * Inference engines should enable this for performance. */
void fm_set_auto_batch(bool enable);

/* Begin accumulating GPU commands into a shared command buffer.
 * All fm_* calls between begin/end encode without committing. */
int  fm_batch_begin(void);

/* Commit the batched command buffer and wait for completion. */
void fm_batch_end(void);

/* Check if command batching is active. */
bool fm_batch_active(void);

#ifdef __cplusplus
}
#endif

/* ================================================================
 * Kernel families
 *
 * One header per unit under metal/kernels/, included last so the types
 * above are already declared.  Each is self-sufficient (it includes this
 * header back, which the guard makes a no-op), so a caller that wants
 * only one family can include just that header.
 *
 * Adding a family: create the .h/.m pair, register its MSL fragment in
 * metal/kernels/fm_kernel_registry.{h,m}, add it to METAL_SRC in the
 * Makefile, and add one line here.
 * ================================================================ */

#include "metal/kernels/fm_kernel_convert.h"
#include "metal/kernels/fm_kernel_elementwise.h"
#include "metal/kernels/fm_kernel_gemm_f32.h"
#include "metal/kernels/fm_kernel_matvec_q3_k.h"
#include "metal/kernels/fm_kernel_matvec_q4_k.h"
#include "metal/kernels/fm_kernel_matvec_q5_k.h"
#include "metal/kernels/fm_kernel_matvec_q6_k.h"
#include "metal/kernels/fm_kernel_matvec_q8.h"
#include "metal/kernels/fm_kernel_matvec_q8_0.h"
#include "metal/kernels/fm_kernel_norm.h"
#include "metal/kernels/fm_kernel_probe.h"
#include "metal/kernels/fm_kernel_rope.h"
#include "metal/kernels/fm_kernel_ssm.h"
#include "metal/kernels/fm_kernel_ssm_chunked.h"
#include "metal/kernels/fm_kernel_moe.h"
#include "metal/kernels/fm_kernel_attn.h"
#include "metal/kernels/fm_kernel_wkv7.h"

#endif /* FOUNDRY_METAL_H */
