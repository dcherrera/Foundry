/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   fm_kernel_moe.h
 * @brief  Mixture-of-experts family: router top-k, expert sort, grouped Q3_K gate+up,
 *         grouped Q3_K/Q4_K down, SiLU-multiply and the shared-expert gate.
 *
 * Written from a behaviour specification; no ggml or
 * llama.cpp source was consulted.
 *
 * @status None
 * @issues Router rounds that select nothing write index -1 / weight -1.0f and add -1 to
 *         the weight sum; its max seed is -3.0e38f, not -Inf. Sort counts a token whose
 *         slots repeat an expert once, with sorted_slot = (sum of matching slots +
 *         count) - 1. Grouped kernels guard only negative experts, not >= n_expert.
 *         Grouped down weights each lane's partial before one simd_sum, so it is not
 *         bit-exact against an expert-by-expert loop. No overflow checks on
 *         n_tokens*expert_used, rows*tiles, n_requests*ff. Sort accepts
 *         expert_used > n_expert; the router rejects it.
 * @todo   None
 */

#ifndef FOUNDRY_METAL_KERNEL_MOE_H
#define FOUNDRY_METAL_KERNEL_MOE_H

#include "metal/fm_metal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Function-constant indices are GLOBAL across the one assembled MSL library, so a
 * collision breaks every kernel. Library-wide allocation:
 *     0-  2 probe          30 Q3_K           50 Q5_K           60 Q6_K
 *    70     Q8_0           80 Q4_K      100-109 rope        110-119 norm
 *   120-129 elementwise 200-219 ssm     220-239 ssm_chunked 240-259 moe
 *   260-279 attn
 * This family uses 240 (down: experts per token), 241 (down: halves per expert dot)
 * and 242 (gate+up: halves per matrix dot). */

/* Weights in one grouped-expert lane unit: half a 256-weight super-block. */
#define FM_MOE_LANE_WEIGHTS   128

/* Threads in one grouped-expert threadgroup (one SIMD group). */
#define FM_MOE_LANES          32

/* Threads in one router / sort / shared-gate threadgroup. */
#define FM_MOE_REDUCE_THREADS 256

/**
 * Router: softmax over each token's expert logits, then the expert_used most
 * probable experts (ties to the lower index), weighted p * (expert_weights_scale /
 * sum of selected p). One 256-thread group per token. Does nothing unless the
 * backend is initialized, all buffers are non-NULL, n_tokens, n_expert, expert_used
 * > 0 and expert_used <= n_expert.
 *
 * @param logits                [n_tokens][n_expert] float32; overwritten with the
 *                              probabilities, selected entries set to -1.0f
 * @param top_idx               [n_tokens][expert_used] int32 out, descending p
 * @param top_wt                [n_tokens][expert_used] float32 out
 * @param n_tokens              Tokens
 * @param n_expert              Experts
 * @param expert_used           Experts selected per token
 * @param expert_weights_scale  Factor applied to the renormalized weights
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_router_topk_at(fm_bufref logits, fm_bufref top_idx, fm_bufref top_wt,
                           int n_tokens, int n_expert, int expert_used,
                           float expert_weights_scale);

/** fm_moe_router_topk_at at offset 0; does nothing if any buffer is NULL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_router_topk(fm_buffer *logits, fm_buffer *top_idx, fm_buffer *top_wt,
                        int n_tokens, int n_expert, int expert_used,
                        float expert_weights_scale);

/**
 * Groups the routed (token, slot) entries by expert, with no atomics: one dispatch
 * counts each expert's tokens, a second writes the offsets and scatters the entries
 * in ascending token order within each expert. Entries whose expert is negative or
 * >= n_expert are dropped. Does nothing unless initialized, all buffers non-NULL and
 * n_tokens, n_expert, expert_used > 0.
 *
 * @param top_idx         [n_tokens][expert_used] int32 routing
 * @param expert_counts   [n_expert] int32 out
 * @param expert_offsets  [n_expert + 1] int32 out (exclusive prefix, then total)
 * @param sorted_token    [n_tokens * expert_used] int32 out
 * @param sorted_slot     [n_tokens * expert_used] int32 out
 * @param sorted_expert   [n_tokens * expert_used] int32 out
 * @param n_tokens        Tokens
 * @param n_expert        Experts
 * @param expert_used     Slots per token
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_expert_sort_at(fm_bufref top_idx, fm_bufref expert_counts,
                           fm_bufref expert_offsets, fm_bufref sorted_token,
                           fm_bufref sorted_slot, fm_bufref sorted_expert,
                           int n_tokens, int n_expert, int expert_used);

/** fm_moe_expert_sort_at at offset 0; does nothing if any buffer is NULL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_expert_sort(fm_buffer *top_idx, fm_buffer *expert_counts,
                        fm_buffer *expert_offsets, fm_buffer *sorted_token,
                        fm_buffer *sorted_slot, fm_buffer *sorted_expert,
                        int n_tokens, int n_expert, int expert_used);

/**
 * Grouped Q3_K gate+up for every routed request (request = token * expert_used +
 * slot): hid[request][r] = silu(gate_e[r] . x_t) * (up_e[r] . x_t). One SIMD group
 * per (request, row). A request routed to a negative expert gets 0. Does nothing
 * unless initialized, all buffers non-NULL, n_tokens, expert_used, expert_ff, cols
 * > 0, cols % 256 == 0 and row_stride_bytes >= (cols / 256) * 110.
 *
 * @param gate_all          [n_expert][expert_ff] Q3_K rows
 * @param up_all            [n_expert][expert_ff] Q3_K rows
 * @param x                 [n_tokens][cols] float32
 * @param top_idx           [n_tokens][expert_used] int32 routing
 * @param hid               [n_tokens * expert_used][expert_ff] float32 out
 * @param n_tokens          Tokens
 * @param expert_used       Slots per token
 * @param expert_ff         Rows per expert matrix
 * @param cols              Input width, multiple of 256
 * @param row_stride_bytes  Bytes between rows
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_gateup_q3k_at(fm_bufref gate_all, fm_bufref up_all, fm_bufref x,
                          fm_bufref top_idx, fm_bufref hid,
                          int n_tokens, int expert_used, int expert_ff,
                          int cols, int row_stride_bytes);

/** fm_moe_gateup_q3k_at at offset 0; does nothing if any buffer is NULL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_gateup_q3k(fm_buffer *gate_all, fm_buffer *up_all, fm_buffer *x,
                       fm_buffer *top_idx, fm_buffer *hid,
                       int n_tokens, int expert_used, int expert_ff,
                       int cols, int row_stride_bytes);

/**
 * fm_moe_gateup_q3k_at driven by the expert-sorted entry list from
 * fm_moe_expert_sort: entry i is (sorted_token[i], sorted_slot[i],
 * sorted_expert[i]) for i < n_requests. Writes the same hid elements, bit for bit,
 * as the slot-major form. Rejects as that form, with n_requests in place of
 * n_tokens * expert_used.
 *
 * @param n_requests  Entries in the sorted list
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_gateup_q3k_sorted_at(fm_bufref gate_all, fm_bufref up_all, fm_bufref x,
                                 fm_bufref sorted_token, fm_bufref sorted_slot,
                                 fm_bufref sorted_expert, fm_bufref hid,
                                 int n_requests, int expert_used, int expert_ff,
                                 int cols, int row_stride_bytes);

/** fm_moe_gateup_q3k_sorted_at at offset 0; does nothing if any buffer is NULL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_gateup_q3k_sorted(fm_buffer *gate_all, fm_buffer *up_all, fm_buffer *x,
                              fm_buffer *sorted_token, fm_buffer *sorted_slot,
                              fm_buffer *sorted_expert, fm_buffer *hid,
                              int n_requests, int expert_used, int expert_ff,
                              int cols, int row_stride_bytes);

/**
 * Grouped Q3_K down projection, weighted and summed over a token's experts:
 * out[t][r] = sum_slot top_wt[t][slot] * (down_e[r] . hid[t * expert_used + slot]).
 * One SIMD group per (token, row); output is written, not accumulated. Slots routed
 * to a negative expert contribute nothing. Does nothing unless initialized, all
 * buffers non-NULL, n_tokens, expert_used, rows, ff > 0, ff % 256 == 0 and
 * row_stride_bytes >= (ff / 256) * 110.
 *
 * @param down_all          [n_expert][rows] Q3_K rows of ff weights
 * @param hid               [n_tokens * expert_used][ff] float32
 * @param top_idx           [n_tokens][expert_used] int32
 * @param top_wt            [n_tokens][expert_used] float32
 * @param out               [n_tokens][rows] float32 out
 * @param n_tokens          Tokens
 * @param expert_used       Slots per token
 * @param rows              Output rows
 * @param ff                Expert hidden width, multiple of 256
 * @param row_stride_bytes  Bytes between rows
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_down_q3k_at(fm_bufref down_all, fm_bufref hid, fm_bufref top_idx,
                        fm_bufref top_wt, fm_bufref out,
                        int n_tokens, int expert_used, int rows, int ff,
                        int row_stride_bytes);

/** fm_moe_down_q3k_at at offset 0; does nothing if any buffer is NULL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_down_q3k(fm_buffer *down_all, fm_buffer *hid, fm_buffer *top_idx,
                     fm_buffer *top_wt, fm_buffer *out,
                     int n_tokens, int expert_used, int rows, int ff,
                     int row_stride_bytes);

/**
 * fm_moe_down_q3k_at for Q4_K weights (144-byte blocks); the stride check uses 144.
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_down_q4k_at(fm_bufref down_all, fm_bufref hid, fm_bufref top_idx,
                        fm_bufref top_wt, fm_bufref out,
                        int n_tokens, int expert_used, int rows, int ff,
                        int row_stride_bytes);

/** fm_moe_down_q4k_at at offset 0; does nothing if any buffer is NULL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_down_q4k(fm_buffer *down_all, fm_buffer *hid, fm_buffer *top_idx,
                     fm_buffer *top_wt, fm_buffer *out,
                     int n_tokens, int expert_used, int rows, int ff,
                     int row_stride_bytes);

/**
 * out[i] = silu(a[i]) * b[i] for i < n. out may alias a or b. Does nothing unless
 * initialized, all buffers non-NULL and n > 0.
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_silu_mul_at(fm_bufref a, fm_bufref b, fm_bufref out, int n);

/** fm_moe_silu_mul_at at offset 0; does nothing if any buffer is NULL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_silu_mul(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n);

/**
 * Shared-expert gate: shexp[t][:] *= sigmoid(gate_w . x[t]) per token, one
 * 256-thread group per token. Does nothing unless initialized, all buffers non-NULL
 * and n_tokens, n_embd > 0.
 *
 * @param gate_w    [n_embd] float32
 * @param x         [n_tokens][n_embd] float32
 * @param shexp     [n_tokens][n_embd] float32, scaled in place
 * @param n_tokens  Tokens
 * @param n_embd    Width
 *
 * Thread-safe: no (encodes into the shared global command buffer)
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_shexp_dot_gate_at(fm_bufref gate_w, fm_bufref x, fm_bufref shexp,
                              int n_tokens, int n_embd);

/** fm_moe_shexp_dot_gate_at at offset 0; does nothing if any buffer is NULL.
 *
 * Thread-safe: no
 * Reentrant:   no
 * Signal-safe: no
 */
void fm_moe_shexp_dot_gate(fm_buffer *gate_w, fm_buffer *x, fm_buffer *shexp,
                           int n_tokens, int n_embd);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_METAL_KERNEL_MOE_H */
