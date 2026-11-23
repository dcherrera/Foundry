/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_attn.h
 * @brief Attention family: fp32 KV cache append, flash decode (the batch-1
 *   vector path, split-K across workgroups plus a combine kernel), and flash
 *   prefill (tiled causal GQA with online softmax).
 *
 *   All three kernels compute exactly one thing — softmax(scale * qK^T) V over
 *   a causal key range — and differ only in how that range is carved up:
 *
 *     prefill   one workgroup per (query row, kv head); keys [0, pos0+t]
 *     decode    one workgroup per (split, kv head); keys [split_lo, split_hi)
 *               of the single query row, partial (m, l, acc) per split
 *     reduce    one workgroup per query head; folds the splits back together
 *
 *   The score matrix is never materialized: each workgroup carries a running
 *   max `m` and running sum `l` and rescales its output accumulator whenever
 *   the max moves (online softmax). Tile order therefore differs from a
 *   single-pass CPU softmax and the results are NOT bit-exact — the bar is
 *   ~1e-3 with a stable argmax (spec §11).
 *
 * @status kv_append, flash_decode + flash_decode_reduce and flash_prefill are
 *   implemented and tested (metal/tests/test_metal_attn.c, 14 cases) against
 *   an independent double-precision CPU reference at context depths 1, 2, 16,
 *   17, 513 and 4097, including forced multi-split decode, empty splits, both
 *   causality probes and a constant-V normalization check. Worst drift
 *   1.19e-07 absolute, 8.9e-06 of the output RMS.
 * @issues None
 * @todo
 *   - [ ] fm_attn_blk — per-KV-block mask classification so fully-masked
 *         blocks are skipped. Long-context lever, not needed for correctness.
 *
 * FUNCTION-CONSTANT INDICES OWNED BY THIS FAMILY: 260-279.
 *   Function constants share ONE index space across the whole assembled
 *   shader library, so each family reserves a block; two families colliding
 *   on an index fails the compile of every kernel in the backend, not just
 *   theirs. Allocated so far: 0-2 probe / 30 Q3_K / 50 Q5_K / 60 Q6_K / 70 Q8_0 / 80 Q4_K /
 *   100-109 rope / 110-119 norm / 120-129 elementwise / 200-219 ssm /
 *   220-239 ssm_chunked / 240-259 moe / 260-279 attn.
 *   In use here: 260 kv_groups, 261 head_dim. 262-279 free.
 */

#ifndef FOUNDRY_METAL_KERNEL_ATTN_H
#define FOUNDRY_METAL_KERNEL_ATTN_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Shape limits
 *
 * The threadgroup staging buffers are declared at these maxima because
 * fm_metal's dispatch helpers never call setThreadgroupMemoryLength — a
 * kernel's threadgroup arrays have to be statically sized. The staged K
 * tile is FM_ATTN_TILE_KEYS * FM_ATTN_MAX_HEAD_DIM floats (16 KiB at the
 * limits below), which is what caps them.
 * ================================================================ */

/* Keys staged per tile. One tile is the unit the online softmax folds. */
#define FM_ATTN_TILE_KEYS      16

/* Largest head_dim any entry point accepts. Must be a multiple of 32
 * (the SIMD width the QK dot is split across). */
#define FM_ATTN_MAX_HEAD_DIM   256

/* Largest GQA group (query heads per kv head) any entry point accepts.
 * One SIMD group of the workgroup scores one query head, and a workgroup
 * is 8 SIMD groups wide. */
#define FM_ATTN_MAX_GROUPS     8

/* Threads per workgroup for the flash kernels: 8 SIMD groups of 32. */
#define FM_ATTN_THREADS        256

/* Function-constant indices this family binds. Both flash kernels are
 * specialized on them so the per-thread accumulator arrays have
 * compile-time loop bounds and stay in registers. */
#define FM_ATTN_FC_KV_GROUPS   260
#define FM_ATTN_FC_HEAD_DIM    261

/* Workgroups the default decode plan aims for, in total across all kv heads.
 *
 * MEASURED by metal/tests/bench_attn_split.c on M4 / 10 GPU cores,
 * n_head=16 n_head_kv=2 head_dim=256. Speedup over the unsplit kernel
 * (2 workgroups, one per kv head):
 *
 *   workgroups      2      4      8     16     32     64
 *   seq   128    1.00x  1.42x  1.66x  --- tile rounding caps this length ---
 *   seq   512    1.00x  1.34x  1.73x  2.28x  2.19x  2.10x
 *   seq  4096    1.00x  2.57x  4.54x  6.68x  9.37x  8.69x
 *
 * Absolute milliseconds are not quoted because they moved 2x between runs on
 * a shared machine; the ratios within a run were stable across five runs and
 * are what the choice rests on. 32 is the smallest count at which the curve
 * has flattened at every length tested, and 64 was never better than 32
 * outside the noise. At long context the split is not a micro-optimization
 * — it is most of the decode path's speed.
 *
 * Note the SHAPE of that table: the best split_len moves 16 -> 32 -> 256
 * across the three lengths while the best workgroup count does not move at
 * all. What the GPU wants held constant is the number of workgroups, which
 * is why this plan is parameterized by workgroups rather than by the
 * keys-per-split knob the CUDA decode path uses. */
#define FM_ATTN_DECODE_TARGET_WORKGROUPS  32

/* ================================================================
 * Split-K planning
 * ================================================================ */

/**
 * Plans the decode split: how many workgroups share the key range and how
 * many keys each one takes.
 *
 * `split_len` is rounded up to FM_ATTN_TILE_KEYS so every workgroup starts
 * on a tile boundary, and the returned split count is recomputed from the
 * rounded length — so no planned split is ever empty, and a short context
 * naturally uses fewer workgroups than asked for rather than fanning out
 * into slivers.
 *
 * @param seq_len            Live keys in the cache (>= 1)
 * @param n_head_kv          Key/value heads; the dispatch runs
 *                           n_head_kv * splits workgroups
 * @param target_workgroups  Total workgroups to aim for
 *                           (0 selects FM_ATTN_DECODE_TARGET_WORKGROUPS)
 * @param split_len_out      Receives the keys per workgroup
 * @return                   Number of splits, or 0 if the arguments are invalid
 *
 * Thread-safe: yes (pure function)
 * Reentrant: yes
 * Signal-safe: yes
 */
int fm_attn_decode_plan(int seq_len, int n_head_kv, int target_workgroups,
                        int *split_len_out);

/**
 * Floats of scratch the split-K partials need, for each of the three
 * buffers fm_attn_flash_decode_at writes.
 *
 * @param n_head    Total query heads (n_head_kv * kv_groups)
 * @param head_dim  Per-head dimension
 * @param n_splits  Splits the decode will be dispatched with
 * @param m_floats  Receives the element count for the m and l buffers
 * @param acc_floats Receives the element count for the acc buffer
 *
 * Thread-safe: yes (pure function)
 * Reentrant: yes
 * Signal-safe: yes
 */
void fm_attn_decode_scratch_floats(int n_head, int head_dim, int n_splits,
                                   size_t *m_floats, size_t *acc_floats);

/* ================================================================
 * KV cache append
 * ================================================================ */

/**
 * Appends `n` rows of post-RoPE K and V into the ring KV cache.
 *
 * Row t of the batch lands at cache position (pos0 + t) % max_seq, which is
 * the same ring rule the host cache uses, so a host<->device sync stays a
 * contiguous per-head row range.
 *
 * @param k_new     [n][n_head_kv][head_dim] fp32
 * @param v_new     [n][n_head_kv][head_dim] fp32
 * @param k_cache   [n_head_kv][max_seq][head_dim] fp32 (written)
 * @param v_cache   [n_head_kv][max_seq][head_dim] fp32 (written)
 * @param n         Rows to append (0 is a no-op, not an error)
 * @param n_head_kv Key/value heads
 * @param head_dim  Per-head dimension
 * @param max_seq   Ring capacity in rows
 * @param pos0      Absolute position of row 0
 *
 * Q4_0 KV quantization is explicitly out of scope (spec §12).
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_attn_kv_append_at(fm_bufref k_new, fm_bufref v_new,
                          fm_bufref k_cache, fm_bufref v_cache,
                          int n, int n_head_kv, int head_dim, int max_seq,
                          int pos0);

/* fm_attn_kv_append_at with every operand bound at offset 0. */
void fm_attn_kv_append(fm_buffer *k_new, fm_buffer *v_new,
                       fm_buffer *k_cache, fm_buffer *v_cache,
                       int n, int n_head_kv, int head_dim, int max_seq,
                       int pos0);

/* ================================================================
 * Flash decode (split-K vector path)
 * ================================================================ */

/**
 * Scores one query row against key slice [split*split_len, ...) of the
 * cache and writes that slice's partial online-softmax state.
 *
 * This is half of the decode path: the partials it writes are meaningless
 * until fm_attn_flash_decode_reduce_at folds them. Both take `n_splits`, and
 * it must be the same value in both calls — it is the stride of the partial
 * arrays, not just a workgroup count.
 *
 * The caller owns the split plan (see fm_attn_decode_plan). Passing a plan
 * whose splits run past seq_len is safe: an empty split writes m = -inf,
 * l = 0 and a zero accumulator, which the reduce weights to zero.
 *
 * @param q          [n_head][head_dim] fp32, the single query row
 * @param k_cache    [n_head_kv][max_seq][head_dim] fp32
 * @param v_cache    [n_head_kv][max_seq][head_dim] fp32
 * @param m_partial  [n_head][n_splits] fp32 (written) running max
 * @param l_partial  [n_head][n_splits] fp32 (written) running sum
 * @param acc_partial [n_head][n_splits][head_dim] fp32 (written) UNnormalized
 * @param kv_groups  Query heads per kv head, 1..FM_ATTN_MAX_GROUPS
 * @param n_head_kv  Key/value heads
 * @param head_dim   Per-head dimension, multiple of 32, <= FM_ATTN_MAX_HEAD_DIM
 * @param max_seq    Ring capacity in rows
 * @param seq_len    Live keys, >= 1; the query attends [0, seq_len)
 * @param n_splits   Workgroups per kv head, >= 1
 * @param split_len  Keys per split; the caller's plan, not derived here
 * @param scale      1/sqrt(head_dim)
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_attn_flash_decode_at(fm_bufref q, fm_bufref k_cache, fm_bufref v_cache,
                             fm_bufref m_partial, fm_bufref l_partial,
                             fm_bufref acc_partial,
                             int kv_groups, int n_head_kv, int head_dim,
                             int max_seq, int seq_len, int n_splits,
                             int split_len, float scale);

/* fm_attn_flash_decode_at with every operand bound at offset 0. */
void fm_attn_flash_decode(fm_buffer *q, fm_buffer *k_cache, fm_buffer *v_cache,
                          fm_buffer *m_partial, fm_buffer *l_partial,
                          fm_buffer *acc_partial,
                          int kv_groups, int n_head_kv, int head_dim,
                          int max_seq, int seq_len, int n_splits,
                          int split_len, float scale);

/**
 * Folds the per-split partials into the final attention output.
 *
 * Takes the max of the per-split maxima, reweights each split's sum and
 * accumulator by exp(m_split - M), and divides. Algebraically identical to
 * a single-pass softmax over the whole key range; a split that saw no keys
 * carries m = -inf and drops out with weight zero.
 *
 * @param m_partial   [n_head][n_splits] fp32 from fm_attn_flash_decode_at
 * @param l_partial   [n_head][n_splits] fp32
 * @param acc_partial [n_head][n_splits][head_dim] fp32
 * @param out         [n_head][head_dim] fp32 (written)
 * @param n_head      Total query heads
 * @param head_dim    Per-head dimension
 * @param n_splits    Same value passed to fm_attn_flash_decode_at
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_attn_flash_decode_reduce_at(fm_bufref m_partial, fm_bufref l_partial,
                                    fm_bufref acc_partial, fm_bufref out,
                                    int n_head, int head_dim, int n_splits);

/* fm_attn_flash_decode_reduce_at with every operand bound at offset 0. */
void fm_attn_flash_decode_reduce(fm_buffer *m_partial, fm_buffer *l_partial,
                                 fm_buffer *acc_partial, fm_buffer *out,
                                 int n_head, int head_dim, int n_splits);

/* ================================================================
 * Flash prefill (tiled causal GQA)
 * ================================================================ */

/**
 * Causal GQA attention for a batch of query rows against the cache.
 *
 * Query row t sits at absolute position pos0 + t and attends cache rows
 * [0, pos0 + t] — causality is enforced by the key loop bound and by
 * clamping the final tile to the live key count, so no future key is ever
 * read, scored or weighted. The caller must have appended rows [0, n) to
 * the cache first, and must guarantee pos0 + n <= max_seq (no ring wrap
 * inside the attended window).
 *
 * @param q         [n][n_head][head_dim] fp32, post QK-norm and RoPE
 * @param k_cache   [n_head_kv][max_seq][head_dim] fp32
 * @param v_cache   [n_head_kv][max_seq][head_dim] fp32
 * @param out       [n][n_head][head_dim] fp32 (written)
 * @param n         Query rows (0 is a no-op, not an error)
 * @param kv_groups Query heads per kv head, 1..FM_ATTN_MAX_GROUPS
 * @param n_head_kv Key/value heads
 * @param head_dim  Per-head dimension, multiple of 32, <= FM_ATTN_MAX_HEAD_DIM
 * @param max_seq   Ring capacity in rows
 * @param pos0      Absolute position of query row 0
 * @param scale     1/sqrt(head_dim)
 *
 * Thread-safe: no (shares the global command buffer)
 * Reentrant: no
 * Signal-safe: no
 */
void fm_attn_flash_prefill_at(fm_bufref q, fm_bufref k_cache, fm_bufref v_cache,
                              fm_bufref out, int n, int kv_groups,
                              int n_head_kv, int head_dim, int max_seq,
                              int pos0, float scale);

/* fm_attn_flash_prefill_at with every operand bound at offset 0. */
void fm_attn_flash_prefill(fm_buffer *q, fm_buffer *k_cache, fm_buffer *v_cache,
                           fm_buffer *out, int n, int kv_groups,
                           int n_head_kv, int head_dim, int max_seq,
                           int pos0, float scale);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_ATTN_H */
