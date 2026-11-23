/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   fm_kernel_moe.m
 * @brief  Host dispatch and MSL source for the mixture-of-experts kernels.
 *
 * Written from a behaviour specification; no ggml or
 * llama.cpp source was consulted.
 *
 * @status None
 * @issues See fm_kernel_moe.h (router -1 rounds and -3.0e38f seed, sort duplicate-expert
 *         slot, no >= n_expert guard, grouped down reassociated, no overflow checks).
 * @todo   None
 */

#include "metal/kernels/fm_kernel_moe.h"

#include <string.h>

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

/* Function-constant indices owned by this family (block 240-259). */
#define FM_MOE_FC_DOWN_TOPK     240
#define FM_MOE_FC_DOWN_HALVES   241
#define FM_MOE_FC_GATEUP_HALVES 242

/* Super-block geometry of the two weight formats the grouped kernels read. */
#define FM_MOE_BLOCK_WEIGHTS   256
#define FM_MOE_Q3_K_BLOCK_BYTES 110
#define FM_MOE_Q4_K_BLOCK_BYTES 144

/* ================================================================
 * Router and sort
 * ================================================================ */

void fm_moe_router_topk_at(fm_bufref logits, fm_bufref top_idx, fm_bufref top_wt,
                           int n_tokens, int n_expert, int expert_used,
                           float expert_weights_scale) {
    if (!fm_metal_is_initialized() || !logits.buffer || !top_idx.buffer || !top_wt.buffer)
        return;
    if (n_tokens <= 0 || n_expert <= 0 || expert_used <= 0 || expert_used > n_expert)
        return;

    /* The scale travels as raw bits so every inline argument is a uint. */
    uint32_t scale_bits = 0;
    memcpy(&scale_bits, &expert_weights_scale, sizeof(scale_bits));

    fm_bufref refs[] = { logits, top_idx, top_wt };
    uint32_t consts[] = { (uint32_t)n_tokens, (uint32_t)n_expert, (uint32_t)expert_used,
                          scale_bits };
    fm_dispatch_threadgroups_refs(@"fmmoe_router_topk", NULL, refs, 3, consts, 4,
                                  n_tokens, FM_MOE_REDUCE_THREADS);
}

void fm_moe_router_topk(fm_buffer *logits, fm_buffer *top_idx, fm_buffer *top_wt,
                        int n_tokens, int n_expert, int expert_used,
                        float expert_weights_scale) {
    if (!logits || !top_idx || !top_wt)
        return;
    fm_moe_router_topk_at(fm_bufref_at(logits, 0), fm_bufref_at(top_idx, 0),
                          fm_bufref_at(top_wt, 0), n_tokens, n_expert, expert_used,
                          expert_weights_scale);
}

void fm_moe_expert_sort_at(fm_bufref top_idx, fm_bufref expert_counts,
                           fm_bufref expert_offsets, fm_bufref sorted_token,
                           fm_bufref sorted_slot, fm_bufref sorted_expert,
                           int n_tokens, int n_expert, int expert_used) {
    if (!fm_metal_is_initialized() || !top_idx.buffer || !expert_counts.buffer ||
        !expert_offsets.buffer || !sorted_token.buffer || !sorted_slot.buffer ||
        !sorted_expert.buffer)
        return;
    if (n_tokens <= 0 || n_expert <= 0 || expert_used <= 0)
        return;

    /* Count first, then offsets + scatter: the second pass needs every expert's
     * count, and a separate dispatch is the barrier that avoids atomics. */
    fm_bufref count_refs[] = { top_idx, expert_counts };
    uint32_t count_consts[] = { (uint32_t)n_tokens, (uint32_t)expert_used };
    fm_dispatch_threadgroups_refs(@"fmmoe_expert_count", NULL, count_refs, 2,
                                  count_consts, 2, n_expert, FM_MOE_REDUCE_THREADS);

    fm_bufref scatter_refs[] = { top_idx, expert_counts, expert_offsets,
                                 sorted_token, sorted_slot, sorted_expert };
    uint32_t scatter_consts[] = { (uint32_t)n_tokens, (uint32_t)n_expert,
                                  (uint32_t)expert_used };
    fm_dispatch_threadgroups_refs(@"fmmoe_expert_scatter", NULL, scatter_refs, 6,
                                  scatter_consts, 3, n_expert, FM_MOE_REDUCE_THREADS);
}

void fm_moe_expert_sort(fm_buffer *top_idx, fm_buffer *expert_counts,
                        fm_buffer *expert_offsets, fm_buffer *sorted_token,
                        fm_buffer *sorted_slot, fm_buffer *sorted_expert,
                        int n_tokens, int n_expert, int expert_used) {
    if (!top_idx || !expert_counts || !expert_offsets || !sorted_token || !sorted_slot ||
        !sorted_expert)
        return;
    fm_moe_expert_sort_at(fm_bufref_at(top_idx, 0), fm_bufref_at(expert_counts, 0),
                          fm_bufref_at(expert_offsets, 0), fm_bufref_at(sorted_token, 0),
                          fm_bufref_at(sorted_slot, 0), fm_bufref_at(sorted_expert, 0),
                          n_tokens, n_expert, expert_used);
}

/* ================================================================
 * Grouped expert projections
 * ================================================================ */

/* Shared launch of both gate+up forms; `refs` already holds the form's buffers. */
static void fm_moe_gateup_launch(NSString *kernel_name, const fm_bufref *refs, int n_refs,
                                 int n_requests, int expert_used, int expert_ff,
                                 int cols, int row_stride_bytes) {
    if (!fm_metal_is_initialized())
        return;
    if (n_requests <= 0 || expert_used <= 0 || expert_ff <= 0 || cols <= 0 ||
        (cols % FM_MOE_BLOCK_WEIGHTS) != 0)
        return;
    if (row_stride_bytes < (cols / FM_MOE_BLOCK_WEIGHTS) * FM_MOE_Q3_K_BLOCK_BYTES)
        return;

    fm_fc_values fc;
    fm_fc_reset(&fc);
    if (fm_fc_set_int(&fc, FM_MOE_FC_GATEUP_HALVES, cols / FM_MOE_LANE_WEIGHTS) != 0)
        return;

    uint32_t consts[] = { (uint32_t)n_requests, (uint32_t)expert_used,
                          (uint32_t)expert_ff, (uint32_t)cols, (uint32_t)row_stride_bytes };
    fm_dispatch_threadgroups_refs(kernel_name, &fc, refs, n_refs, consts, 5,
                                  n_requests * expert_ff, FM_MOE_LANES);
}

void fm_moe_gateup_q3k_at(fm_bufref gate_all, fm_bufref up_all, fm_bufref x,
                          fm_bufref top_idx, fm_bufref hid,
                          int n_tokens, int expert_used, int expert_ff,
                          int cols, int row_stride_bytes) {
    if (!gate_all.buffer || !up_all.buffer || !x.buffer || !top_idx.buffer || !hid.buffer)
        return;
    if (n_tokens <= 0)
        return;

    fm_bufref refs[] = { gate_all, up_all, x, top_idx, hid };
    fm_moe_gateup_launch(@"fmmoe_gateup_q3k", refs, 5, n_tokens * expert_used,
                         expert_used, expert_ff, cols, row_stride_bytes);
}

void fm_moe_gateup_q3k(fm_buffer *gate_all, fm_buffer *up_all, fm_buffer *x,
                       fm_buffer *top_idx, fm_buffer *hid,
                       int n_tokens, int expert_used, int expert_ff,
                       int cols, int row_stride_bytes) {
    if (!gate_all || !up_all || !x || !top_idx || !hid)
        return;
    fm_moe_gateup_q3k_at(fm_bufref_at(gate_all, 0), fm_bufref_at(up_all, 0),
                         fm_bufref_at(x, 0), fm_bufref_at(top_idx, 0),
                         fm_bufref_at(hid, 0), n_tokens, expert_used, expert_ff,
                         cols, row_stride_bytes);
}

void fm_moe_gateup_q3k_sorted_at(fm_bufref gate_all, fm_bufref up_all, fm_bufref x,
                                 fm_bufref sorted_token, fm_bufref sorted_slot,
                                 fm_bufref sorted_expert, fm_bufref hid,
                                 int n_requests, int expert_used, int expert_ff,
                                 int cols, int row_stride_bytes) {
    if (!gate_all.buffer || !up_all.buffer || !x.buffer || !sorted_token.buffer ||
        !sorted_slot.buffer || !sorted_expert.buffer || !hid.buffer)
        return;

    fm_bufref refs[] = { gate_all, up_all, x, sorted_token, sorted_slot, sorted_expert,
                         hid };
    fm_moe_gateup_launch(@"fmmoe_gateup_q3k_sorted", refs, 7, n_requests,
                         expert_used, expert_ff, cols, row_stride_bytes);
}

void fm_moe_gateup_q3k_sorted(fm_buffer *gate_all, fm_buffer *up_all, fm_buffer *x,
                              fm_buffer *sorted_token, fm_buffer *sorted_slot,
                              fm_buffer *sorted_expert, fm_buffer *hid,
                              int n_requests, int expert_used, int expert_ff,
                              int cols, int row_stride_bytes) {
    if (!gate_all || !up_all || !x || !sorted_token || !sorted_slot || !sorted_expert ||
        !hid)
        return;
    fm_moe_gateup_q3k_sorted_at(fm_bufref_at(gate_all, 0), fm_bufref_at(up_all, 0),
                                fm_bufref_at(x, 0), fm_bufref_at(sorted_token, 0),
                                fm_bufref_at(sorted_slot, 0),
                                fm_bufref_at(sorted_expert, 0), fm_bufref_at(hid, 0),
                                n_requests, expert_used, expert_ff, cols,
                                row_stride_bytes);
}

/* Shared launch of both down forms; block_bytes selects the stride check. */
static void fm_moe_down_launch(NSString *kernel_name, int block_bytes,
                               fm_bufref down_all, fm_bufref hid, fm_bufref top_idx,
                               fm_bufref top_wt, fm_bufref out,
                               int n_tokens, int expert_used, int rows, int ff,
                               int row_stride_bytes) {
    if (!fm_metal_is_initialized() || !down_all.buffer || !hid.buffer ||
        !top_idx.buffer || !top_wt.buffer || !out.buffer)
        return;
    if (n_tokens <= 0 || expert_used <= 0 || rows <= 0 || ff <= 0 ||
        (ff % FM_MOE_BLOCK_WEIGHTS) != 0)
        return;
    if (row_stride_bytes < (ff / FM_MOE_BLOCK_WEIGHTS) * block_bytes)
        return;

    fm_fc_values fc;
    fm_fc_reset(&fc);
    if (fm_fc_set_int(&fc, FM_MOE_FC_DOWN_TOPK, expert_used) != 0 ||
        fm_fc_set_int(&fc, FM_MOE_FC_DOWN_HALVES, ff / FM_MOE_LANE_WEIGHTS) != 0)
        return;

    fm_bufref refs[] = { down_all, hid, top_idx, top_wt, out };
    uint32_t consts[] = { (uint32_t)n_tokens, (uint32_t)rows, (uint32_t)ff,
                          (uint32_t)row_stride_bytes };
    fm_dispatch_threadgroups_refs(kernel_name, &fc, refs, 5, consts, 4,
                                  n_tokens * rows, FM_MOE_LANES);
}

void fm_moe_down_q3k_at(fm_bufref down_all, fm_bufref hid, fm_bufref top_idx,
                        fm_bufref top_wt, fm_bufref out,
                        int n_tokens, int expert_used, int rows, int ff,
                        int row_stride_bytes) {
    fm_moe_down_launch(@"fmmoe_down_q3k", FM_MOE_Q3_K_BLOCK_BYTES, down_all, hid,
                       top_idx, top_wt, out, n_tokens, expert_used, rows, ff,
                       row_stride_bytes);
}

void fm_moe_down_q3k(fm_buffer *down_all, fm_buffer *hid, fm_buffer *top_idx,
                     fm_buffer *top_wt, fm_buffer *out,
                     int n_tokens, int expert_used, int rows, int ff,
                     int row_stride_bytes) {
    if (!down_all || !hid || !top_idx || !top_wt || !out)
        return;
    fm_moe_down_q3k_at(fm_bufref_at(down_all, 0), fm_bufref_at(hid, 0),
                       fm_bufref_at(top_idx, 0), fm_bufref_at(top_wt, 0),
                       fm_bufref_at(out, 0), n_tokens, expert_used, rows, ff,
                       row_stride_bytes);
}

void fm_moe_down_q4k_at(fm_bufref down_all, fm_bufref hid, fm_bufref top_idx,
                        fm_bufref top_wt, fm_bufref out,
                        int n_tokens, int expert_used, int rows, int ff,
                        int row_stride_bytes) {
    fm_moe_down_launch(@"fmmoe_down_q4k", FM_MOE_Q4_K_BLOCK_BYTES, down_all, hid,
                       top_idx, top_wt, out, n_tokens, expert_used, rows, ff,
                       row_stride_bytes);
}

void fm_moe_down_q4k(fm_buffer *down_all, fm_buffer *hid, fm_buffer *top_idx,
                     fm_buffer *top_wt, fm_buffer *out,
                     int n_tokens, int expert_used, int rows, int ff,
                     int row_stride_bytes) {
    if (!down_all || !hid || !top_idx || !top_wt || !out)
        return;
    fm_moe_down_q4k_at(fm_bufref_at(down_all, 0), fm_bufref_at(hid, 0),
                       fm_bufref_at(top_idx, 0), fm_bufref_at(top_wt, 0),
                       fm_bufref_at(out, 0), n_tokens, expert_used, rows, ff,
                       row_stride_bytes);
}

/* ================================================================
 * Elementwise and shared-expert gate
 * ================================================================ */

void fm_moe_silu_mul_at(fm_bufref a, fm_bufref b, fm_bufref out, int n) {
    if (!fm_metal_is_initialized() || !a.buffer || !b.buffer || !out.buffer || n <= 0)
        return;

    fm_bufref refs[] = { a, b, out };
    uint32_t consts[] = { (uint32_t)n };
    fm_dispatch_1d_refs(@"fmmoe_silu_mul", NULL, refs, 3, consts, 1, n);
}

void fm_moe_silu_mul(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n) {
    if (!a || !b || !out)
        return;
    fm_moe_silu_mul_at(fm_bufref_at(a, 0), fm_bufref_at(b, 0), fm_bufref_at(out, 0), n);
}

void fm_moe_shexp_dot_gate_at(fm_bufref gate_w, fm_bufref x, fm_bufref shexp,
                              int n_tokens, int n_embd) {
    if (!fm_metal_is_initialized() || !gate_w.buffer || !x.buffer || !shexp.buffer)
        return;
    if (n_tokens <= 0 || n_embd <= 0)
        return;

    fm_bufref refs[] = { gate_w, x, shexp };
    uint32_t consts[] = { (uint32_t)n_tokens, (uint32_t)n_embd };
    fm_dispatch_threadgroups_refs(@"fmmoe_shexp_dot_gate", NULL, refs, 3, consts, 2,
                                  n_tokens, FM_MOE_REDUCE_THREADS);
}

void fm_moe_shexp_dot_gate(fm_buffer *gate_w, fm_buffer *x, fm_buffer *shexp,
                           int n_tokens, int n_embd) {
    if (!gate_w || !x || !shexp)
        return;
    fm_moe_shexp_dot_gate_at(fm_bufref_at(gate_w, 0), fm_bufref_at(x, 0),
                             fm_bufref_at(shexp, 0), n_tokens, n_embd);
}

/* ================================================================
 * MSL
 *
 * Split by kernel group so no one function grows past a screenful; the parts
 * are concatenated in order into the family's fragment.
 * ================================================================ */

/* Function constants and the threadgroup tree reductions. */
static NSString *fm_moe_msl_common(void)
{
    return @
    "/* ============================================================\n"
    " * Mixture of experts\n"
    " *\n"
    " * Fast math makes results depend on lane ownership, visit order and\n"
    " * reduction primitives, so every partition below is fixed. Trees are\n"
    " * halving trees over threadgroup memory with a barrier per level.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fmmoe_down_topk_fc [[function_constant(240)]];\n"
    "constant int fmmoe_down_topk = is_function_constant_defined(fmmoe_down_topk_fc)\n"
    "                               ? fmmoe_down_topk_fc : 8;\n"
    "constant int fmmoe_down_halves_fc [[function_constant(241)]];\n"
    "constant int fmmoe_down_halves = is_function_constant_defined(fmmoe_down_halves_fc)\n"
    "                                 ? fmmoe_down_halves_fc : 4;\n"
    "constant int fmmoe_gateup_halves_fc [[function_constant(242)]];\n"
    "constant int fmmoe_gateup_halves = is_function_constant_defined(fmmoe_gateup_halves_fc)\n"
    "                                   ? fmmoe_gateup_halves_fc : 16;\n"
    "\n"
    "/* Halving-tree sum; the caller has stored and barriered. Total in buf[0]. */\n"
    "static inline void fmmoe_tree_sum_float(threadgroup float* buf, uint tid, uint T) {\n"
    "    for (uint s = T / 2u; s > 0u; s >>= 1) {\n"
    "        if (tid < s) buf[tid] = buf[tid] + buf[tid + s];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "}\n"
    "\n"
    "static inline void fmmoe_tree_sum_int(threadgroup int* buf, uint tid, uint T) {\n"
    "    for (uint s = T / 2u; s > 0u; s >>= 1) {\n"
    "        if (tid < s) buf[tid] = buf[tid] + buf[tid + s];\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "}\n"
    "\n"
    "static inline void fmmoe_tree_max_float(threadgroup float* buf, uint tid, uint T) {\n"
    "    for (uint s = T / 2u; s > 0u; s >>= 1) {\n"
    "        if (tid < s) buf[tid] = max(buf[tid], buf[tid + s]);\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* Best (value, index) into slot 0: larger value wins; equal values go to\n"
    " * the lower valid index. */\n"
    "static inline void fmmoe_tree_argmax(threadgroup float* sval, threadgroup int* sidx,\n"
    "                                     uint tid, uint T) {\n"
    "    for (uint s = T / 2u; s > 0u; s >>= 1) {\n"
    "        if (tid < s) {\n"
    "            float va = sval[tid];\n"
    "            int   ia = sidx[tid];\n"
    "            float vb = sval[tid + s];\n"
    "            int   ib = sidx[tid + s];\n"
    "            bool take = vb > va || (vb == va && ib >= 0 && (ia < 0 || ib < ia));\n"
    "            sval[tid] = take ? vb : va;\n"
    "            sidx[tid] = take ? ib : ia;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "}\n"
    "\n"
    ;
}

/* Router: softmax, then topk rounds of argmax, then renormalize. */
static NSString *fm_moe_msl_router(void)
{
    return @
    "kernel void fmmoe_router_topk(\n"
    "    device float*       logits     [[buffer(0)]],\n"
    "    device int*         top_idx    [[buffer(1)]],\n"
    "    device float*       top_wt     [[buffer(2)]],\n"
    "    constant uint&      n_tokens   [[buffer(3)]],\n"
    "    constant uint&      n_expert   [[buffer(4)]],\n"
    "    constant uint&      topk       [[buffer(5)]],\n"
    "    constant uint&      scale_bits [[buffer(6)]],\n"
    "    uint tok [[threadgroup_position_in_grid]],\n"
    "    uint tid [[thread_position_in_threadgroup]],\n"
    "    uint T   [[threads_per_threadgroup]]\n"
    ") {\n"
    "    threadgroup float sval[256];\n"
    "    threadgroup int   sidx[256];\n"
    "    threadgroup float swt_sum;\n"
    "    if (tok >= n_tokens) return;\n"
    "    device float* lg = logits + (ulong)tok * n_expert;\n"
    "    device int*   ti = top_idx + (ulong)tok * topk;\n"
    "    device float* tw = top_wt + (ulong)tok * topk;\n"
    "\n"
    "    float m = -3.0e38f;\n"
    "    for (uint e = tid; e < n_expert; e += T) m = max(m, lg[e]);\n"
    "    sval[tid] = m;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    fmmoe_tree_max_float(sval, tid, T);\n"
    "    float row_max = sval[0];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    float sum = 0.0f;\n"
    "    for (uint e = tid; e < n_expert; e += T) {\n"
    "        float v = exp(lg[e] - row_max);\n"
    "        lg[e] = v;\n"
    "        sum += v;\n"
    "    }\n"
    "    sval[tid] = sum;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    fmmoe_tree_sum_float(sval, tid, T);\n"
    "    float inv_sum = 1.0f / sval[0];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    for (uint e = tid; e < n_expert; e += T) lg[e] = lg[e] * inv_sum;\n"
    "    if (tid == 0u) swt_sum = 0.0f;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    for (uint sel = 0; sel < topk; sel++) {\n"
    "        float best_val = -1.0f;\n"
    "        int   best = -1;\n"
    "        for (uint e = tid; e < n_expert; e += T) {\n"
    "            float p = lg[e];\n"
    "            if (p > best_val) {\n"
    "                best_val = p;\n"
    "                best = int(e);\n"
    "            }\n"
    "        }\n"
    "        sval[tid] = best_val;\n"
    "        sidx[tid] = best;\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        fmmoe_tree_argmax(sval, sidx, tid, T);\n"
    "        if (tid == 0u) {\n"
    "            int   winner = sidx[0];\n"
    "            float winner_p = sval[0];\n"
    "            ti[sel] = winner;\n"
    "            tw[sel] = winner_p;\n"
    "            swt_sum += winner_p;\n"
    "            if (winner >= 0) lg[winner] = -1.0f;\n"
    "        }\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "\n"
    "    if (tid == 0u) {\n"
    "        float inv_wt = (swt_sum > 0.0f) ? 1.0f / swt_sum : 0.0f;\n"
    "        float scale = as_type<float>(scale_bits) * inv_wt;\n"
    "        for (uint s = 0; s < topk; s++) tw[s] = tw[s] * scale;\n"
    "    }\n"
    "}\n"
    "\n"
    ;
}

/* Expert sort: per-expert count, then offsets + ordered scatter. */
static NSString *fm_moe_msl_sort(void)
{
    return @
    "/* 1 + the slot where `expert` appears in one token's routing, 0 if absent\n"
    " * (a repeated expert sums its slots). */\n"
    "static inline int fmmoe_slot_plus_one(device const int* route, uint topk, int expert) {\n"
    "    int sp1 = 0;\n"
    "    for (uint s = 0; s < topk; s++) sp1 += int(route[s] == expert) * int(s + 1u);\n"
    "    return sp1;\n"
    "}\n"
    "\n"
    "kernel void fmmoe_expert_count(\n"
    "    device const int* top_idx       [[buffer(0)]],\n"
    "    device int*       expert_counts [[buffer(1)]],\n"
    "    constant uint&    n_tokens      [[buffer(2)]],\n"
    "    constant uint&    topk          [[buffer(3)]],\n"
    "    uint tg  [[threadgroup_position_in_grid]],\n"
    "    uint tid [[thread_position_in_threadgroup]],\n"
    "    uint T   [[threads_per_threadgroup]]\n"
    ") {\n"
    "    threadgroup int scount[256];\n"
    "    int expert = int(tg);\n"
    "    int count = 0;\n"
    "    for (uint t = tid; t < n_tokens; t += T)\n"
    "        count += int(fmmoe_slot_plus_one(top_idx + (ulong)t * topk, topk, expert) != 0);\n"
    "    scount[tid] = count;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    fmmoe_tree_sum_int(scount, tid, T);\n"
    "    if (tid == 0u) expert_counts[expert] = scount[0];\n"
    "}\n"
    "\n"
    "kernel void fmmoe_expert_scatter(\n"
    "    device const int* top_idx        [[buffer(0)]],\n"
    "    device const int* expert_counts  [[buffer(1)]],\n"
    "    device int*       expert_offsets [[buffer(2)]],\n"
    "    device int*       sorted_token   [[buffer(3)]],\n"
    "    device int*       sorted_slot    [[buffer(4)]],\n"
    "    device int*       sorted_expert  [[buffer(5)]],\n"
    "    constant uint&    n_tokens       [[buffer(6)]],\n"
    "    constant uint&    n_expert       [[buffer(7)]],\n"
    "    constant uint&    topk           [[buffer(8)]],\n"
    "    uint tg  [[threadgroup_position_in_grid]],\n"
    "    uint tid [[thread_position_in_threadgroup]],\n"
    "    uint T   [[threads_per_threadgroup]]\n"
    ") {\n"
    "    threadgroup int sred[256];\n"
    "    threadgroup int sflag[256];\n"
    "    int expert = int(tg);\n"
    "\n"
    "    int partial = 0;\n"
    "    for (uint j = tid; j < n_expert; j += T)\n"
    "        partial += (int(j) < expert) ? expert_counts[j] : 0;\n"
    "    sred[tid] = partial;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    fmmoe_tree_sum_int(sred, tid, T);\n"
    "    int base = sred[0];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    if (tid == 0u) {\n"
    "        expert_offsets[expert] = base;\n"
    "        if (tg == n_expert - 1u) expert_offsets[n_expert] = base + expert_counts[expert];\n"
    "    }\n"
    "\n"
    "    int running = 0;\n"
    "    for (uint t0 = 0; t0 < n_tokens; t0 += T) {\n"
    "        uint t = t0 + tid;\n"
    "        int sp1 = (t < n_tokens) ? fmmoe_slot_plus_one(top_idx + (ulong)t * topk, topk,\n"
    "                                                       expert) : 0;\n"
    "        sflag[tid] = int(sp1 != 0);\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        for (uint off = 1u; off < T; off <<= 1) {\n"
    "            int add = (tid >= off) ? sflag[tid - off] : 0;\n"
    "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "            sflag[tid] += add;\n"
    "            threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "        }\n"
    "        int inclusive = sflag[tid];\n"
    "        int tile_total = sflag[T - 1u];\n"
    "        if (sp1 != 0) {\n"
    "            int pos = base + running + inclusive - 1;\n"
    "            sorted_token[pos] = int(t);\n"
    "            sorted_slot[pos] = sp1 - 1;\n"
    "            sorted_expert[pos] = expert;\n"
    "        }\n"
    "        running += tile_total;\n"
    "        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    }\n"
    "}\n"
    "\n"
    ;
}

/* Half-super-block dots: the grouped kernels' lane unit. */
static NSString *fm_moe_msl_half_dots(void)
{
    return @
    "/* Six-bit Q3_K scale of sub-block s (i = s & 3, quarter q = s >> 2). */\n"
    "static inline int fmmoe_q3k_six_bit_scale(device const uchar* sc, uint s) {\n"
    "    uint i = s & 3u;\n"
    "    uint q = s >> 2;\n"
    "    int lo_byte = int(sc[i + 4u * (q & 1u)]);\n"
    "    int lo = (q < 2u) ? (lo_byte & 15) : (lo_byte >> 4);\n"
    "    int hi = (int(sc[8u + i]) >> (2u * q)) & 3;\n"
    "    return lo | (hi << 4);\n"
    "}\n"
    "\n"
    "/* Half u (128 weights) of a Q3_K block against xb, the block's 256\n"
    " * activations. One chain over level v, group g, weight l. */\n"
    "static inline float fmmoe_q3k_half_dot(device const uchar* blk, uint u,\n"
    "                                       device const float* xb) {\n"
    "    float d = float(*(device const half*)(blk + 108));\n"
    "    float acc = 0.0f;\n"
    "    for (uint v = 0; v < 4u; v++) {\n"
    "        for (uint g = 0; g < 2u; g++) {\n"
    "            float dl = d * float(fmmoe_q3k_six_bit_scale(blk + 96, 8u * u + 2u * v + g) - 32);\n"
    "            device const uchar* qs = blk + 32u + 32u * u + 16u * g;\n"
    "            device const uchar* hm = blk + 16u * g;\n"
    "            device const float* xs = xb + 128u * u + 32u * v + 16u * g;\n"
    "            uint shift = 2u * v;\n"
    "            uint mask = 1u << (4u * u + v);\n"
    "            for (uint l = 0; l < 16u; l++) {\n"
    "                int q2 = int((uint(qs[l]) >> shift) & 3u);\n"
    "                int quant = (uint(hm[l]) & mask) ? q2 : q2 - 4;\n"
    "                acc += dl * float(quant) * xs[l];\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    return acc;\n"
    "}\n"
    "\n"
    "/* Six-bit Q4_K (scale, min) pair j from the 12 packed bytes sr. */\n"
    "static inline void fmmoe_q4k_scale_min(device const uchar* sr, uint j,\n"
    "                                       thread int& sc, thread int& mn) {\n"
    "    if (j < 4u) {\n"
    "        sc = int(sr[j]) & 63;\n"
    "        mn = int(sr[j + 4u]) & 63;\n"
    "    } else {\n"
    "        sc = (int(sr[j + 4u]) & 15) | ((int(sr[j - 4u]) >> 6) << 4);\n"
    "        mn = (int(sr[j + 4u]) >> 4) | ((int(sr[j]) >> 6) << 4);\n"
    "    }\n"
    "}\n"
    "\n"
    "/* Half u of a Q4_K block: groups 2u and 2u + 1, each 32 bytes whose low\n"
    " * nibbles are weights 64p + l and high nibbles 64p + 32 + l. */\n"
    "static inline float fmmoe_q4k_half_dot(device const uchar* blk, uint u,\n"
    "                                       device const float* xb) {\n"
    "    float d = float(*(device const half*)(blk));\n"
    "    float dmin = float(*(device const half*)(blk + 2));\n"
    "    device const uchar* sr = blk + 4;\n"
    "    device const uchar* qs = blk + 16;\n"
    "    float acc = 0.0f;\n"
    "    for (uint p = 2u * u; p < 2u * u + 2u; p++) {\n"
    "        int sc_lo, mn_lo, sc_hi, mn_hi;\n"
    "        fmmoe_q4k_scale_min(sr, 2u * p, sc_lo, mn_lo);\n"
    "        fmmoe_q4k_scale_min(sr, 2u * p + 1u, sc_hi, mn_hi);\n"
    "        float d1 = d * float(sc_lo);\n"
    "        float m1 = dmin * float(mn_lo);\n"
    "        float d2 = d * float(sc_hi);\n"
    "        float m2 = dmin * float(mn_hi);\n"
    "        for (uint l = 0; l < 32u; l++) {\n"
    "            uchar q = qs[32u * p + l];\n"
    "            acc += (d1 * float(q & 15) - m1) * xb[64u * p + l];\n"
    "            acc += (d2 * float(q >> 4) - m2) * xb[64u * p + 32u + l];\n"
    "        }\n"
    "    }\n"
    "    return acc;\n"
    "}\n"
    "\n"
    ;
}

/* Grouped gate+up (slot-major and sorted) and grouped down (Q3_K, Q4_K). */
static NSString *fm_moe_msl_grouped(void)
{
    return @
    "/* One SIMD group per (request, row). Lane units are halves: the first HV\n"
    " * are gate halves, the rest up halves; both chains update every unit. */\n"
    "static inline void fmmoe_gateup_row(device const uchar* gate_all,\n"
    "                                    device const uchar* up_all,\n"
    "                                    device const float* x, device float* hid,\n"
    "                                    uint request, uint token, int expert, uint row,\n"
    "                                    uint lane, uint ff, uint cols, uint row_stride) {\n"
    "    if (expert < 0) {\n"
    "        if (lane == 0u) hid[(ulong)request * ff + row] = 0.0f;\n"
    "        return;\n"
    "    }\n"
    "    ulong row_off = (ulong)expert * ((ulong)ff * row_stride) + (ulong)row * row_stride;\n"
    "    device const uchar* gate_row = gate_all + row_off;\n"
    "    device const uchar* up_row = up_all + row_off;\n"
    "    device const float* xt = x + (ulong)token * cols;\n"
    "    const uint HV = uint(fmmoe_gateup_halves);\n"
    "\n"
    "    float ga = 0.0f;\n"
    "    float ua = 0.0f;\n"
    "    for (uint w = lane; w < 2u * HV; w += 32u) {\n"
    "        bool is_gate = w < HV;\n"
    "        uint h = is_gate ? w : w - HV;\n"
    "        device const uchar* row_base = is_gate ? gate_row : up_row;\n"
    "        float v = fmmoe_q3k_half_dot(row_base + (ulong)(h >> 1) * 110u, h & 1u,\n"
    "                                     xt + (ulong)(h >> 1) * 256u);\n"
    "        ga += is_gate ? v : 0.0f;\n"
    "        ua += is_gate ? 0.0f : v;\n"
    "    }\n"
    "    float g = simd_sum(ga);\n"
    "    float u = simd_sum(ua);\n"
    "    if (lane == 0u) hid[(ulong)request * ff + row] = (g / (1.0f + exp(-g))) * u;\n"
    "}\n"
    "\n"
    "kernel void fmmoe_gateup_q3k(\n"
    "    device const uchar* gate_all   [[buffer(0)]],\n"
    "    device const uchar* up_all     [[buffer(1)]],\n"
    "    device const float* x          [[buffer(2)]],\n"
    "    device const int*   top_idx    [[buffer(3)]],\n"
    "    device float*       hid        [[buffer(4)]],\n"
    "    constant uint&      n_requests [[buffer(5)]],\n"
    "    constant uint&      topk       [[buffer(6)]],\n"
    "    constant uint&      ff         [[buffer(7)]],\n"
    "    constant uint&      cols       [[buffer(8)]],\n"
    "    constant uint&      row_stride [[buffer(9)]],\n"
    "    uint tg   [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    uint entry = tg / ff;\n"
    "    if (entry >= n_requests) return;\n"
    "    uint row = tg - entry * ff;\n"
    "    uint request = entry;\n"
    "    fmmoe_gateup_row(gate_all, up_all, x, hid, request, request / topk,\n"
    "                     top_idx[request], row, lane, ff, cols, row_stride);\n"
    "}\n"
    "\n"
    "kernel void fmmoe_gateup_q3k_sorted(\n"
    "    device const uchar* gate_all      [[buffer(0)]],\n"
    "    device const uchar* up_all        [[buffer(1)]],\n"
    "    device const float* x             [[buffer(2)]],\n"
    "    device const int*   sorted_token  [[buffer(3)]],\n"
    "    device const int*   sorted_slot   [[buffer(4)]],\n"
    "    device const int*   sorted_expert [[buffer(5)]],\n"
    "    device float*       hid           [[buffer(6)]],\n"
    "    constant uint&      n_requests    [[buffer(7)]],\n"
    "    constant uint&      topk          [[buffer(8)]],\n"
    "    constant uint&      ff            [[buffer(9)]],\n"
    "    constant uint&      cols          [[buffer(10)]],\n"
    "    constant uint&      row_stride    [[buffer(11)]],\n"
    "    uint tg   [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    uint entry = tg / ff;\n"
    "    if (entry >= n_requests) return;\n"
    "    uint row = tg - entry * ff;\n"
    "    int expert = sorted_expert[entry];\n"
    "    uint token = uint(sorted_token[entry]);\n"
    "    uint request = token * topk + uint(sorted_slot[entry]);\n"
    "    fmmoe_gateup_row(gate_all, up_all, x, hid, request, token, expert, row, lane,\n"
    "                     ff, cols, row_stride);\n"
    "}\n"
    "\n"
    "/* One SIMD group per (token, row). Lane units are (slot, half) pairs across\n"
    " * all K slots; each unit's dot is weighted by its slot's router weight\n"
    " * before the single simd_sum. */\n"
    "template <bool IS_Q4K>\n"
    "static inline void fmmoe_down_row(device const uchar* down_all, device const float* hid,\n"
    "                                  device const int* top_idx, device const float* top_wt,\n"
    "                                  device float* out, uint n_tokens, uint rows, uint ff,\n"
    "                                  uint row_stride, uint tg, uint lane) {\n"
    "    uint token = tg / rows;\n"
    "    if (token >= n_tokens) return;\n"
    "    uint row = tg - token * rows;\n"
    "    const uint K = uint(fmmoe_down_topk);\n"
    "    const uint HV = uint(fmmoe_down_halves);\n"
    "    const uint B = IS_Q4K ? 144u : 110u;\n"
    "\n"
    "    float acc = 0.0f;\n"
    "    for (uint w = lane; w < K * HV; w += 32u) {\n"
    "        uint slot = w / HV;\n"
    "        uint h = w - slot * HV;\n"
    "        uint request = token * K + slot;\n"
    "        int expert = top_idx[request];\n"
    "        if (expert < 0) continue;\n"
    "        device const uchar* blk = down_all + (ulong)expert * ((ulong)rows * row_stride)\n"
    "                                  + (ulong)row * row_stride + (ulong)(h >> 1) * B;\n"
    "        device const float* xb = hid + (ulong)request * ff + (ulong)(h >> 1) * 256u;\n"
    "        float dot = IS_Q4K ? fmmoe_q4k_half_dot(blk, h & 1u, xb)\n"
    "                           : fmmoe_q3k_half_dot(blk, h & 1u, xb);\n"
    "        acc += top_wt[request] * dot;\n"
    "    }\n"
    "    float row_sum = simd_sum(acc);\n"
    "    if (lane == 0u) out[(ulong)token * rows + row] = row_sum;\n"
    "}\n"
    "\n"
    "kernel void fmmoe_down_q3k(\n"
    "    device const uchar* down_all   [[buffer(0)]],\n"
    "    device const float* hid        [[buffer(1)]],\n"
    "    device const int*   top_idx    [[buffer(2)]],\n"
    "    device const float* top_wt     [[buffer(3)]],\n"
    "    device float*       out        [[buffer(4)]],\n"
    "    constant uint&      n_tokens   [[buffer(5)]],\n"
    "    constant uint&      rows       [[buffer(6)]],\n"
    "    constant uint&      ff         [[buffer(7)]],\n"
    "    constant uint&      row_stride [[buffer(8)]],\n"
    "    uint tg   [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    fmmoe_down_row<false>(down_all, hid, top_idx, top_wt, out, n_tokens, rows, ff,\n"
    "                          row_stride, tg, lane);\n"
    "}\n"
    "\n"
    "kernel void fmmoe_down_q4k(\n"
    "    device const uchar* down_all   [[buffer(0)]],\n"
    "    device const float* hid        [[buffer(1)]],\n"
    "    device const int*   top_idx    [[buffer(2)]],\n"
    "    device const float* top_wt     [[buffer(3)]],\n"
    "    device float*       out        [[buffer(4)]],\n"
    "    constant uint&      n_tokens   [[buffer(5)]],\n"
    "    constant uint&      rows       [[buffer(6)]],\n"
    "    constant uint&      ff         [[buffer(7)]],\n"
    "    constant uint&      row_stride [[buffer(8)]],\n"
    "    uint tg   [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    fmmoe_down_row<true>(down_all, hid, top_idx, top_wt, out, n_tokens, rows, ff,\n"
    "                         row_stride, tg, lane);\n"
    "}\n"
    "\n"
    ;
}

/* SiLU-multiply and the shared-expert sigmoid gate. */
static NSString *fm_moe_msl_shared(void)
{
    return @
    "kernel void fmmoe_silu_mul(\n"
    "    device const float* a   [[buffer(0)]],\n"
    "    device const float* b   [[buffer(1)]],\n"
    "    device float*       out [[buffer(2)]],\n"
    "    constant uint&      n   [[buffer(3)]],\n"
    "    uint i [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (i >= n) return;\n"
    "    float v = a[i];\n"
    "    out[i] = (v / (1.0f + exp(-v))) * b[i];\n"
    "}\n"
    "\n"
    "kernel void fmmoe_shexp_dot_gate(\n"
    "    device const float* gate_w   [[buffer(0)]],\n"
    "    device const float* x        [[buffer(1)]],\n"
    "    device float*       shexp    [[buffer(2)]],\n"
    "    constant uint&      n_tokens [[buffer(3)]],\n"
    "    constant uint&      n_embd   [[buffer(4)]],\n"
    "    uint tok [[threadgroup_position_in_grid]],\n"
    "    uint tid [[thread_position_in_threadgroup]],\n"
    "    uint T   [[threads_per_threadgroup]]\n"
    ") {\n"
    "    threadgroup float sdot[256];\n"
    "    if (tok >= n_tokens) return;\n"
    "    device const float* xt = x + (ulong)tok * n_embd;\n"
    "    float partial = 0.0f;\n"
    "    for (uint i = tid; i < n_embd; i += T) partial += gate_w[i] * xt[i];\n"
    "    sdot[tid] = partial;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    fmmoe_tree_sum_float(sdot, tid, T);\n"
    "    float sg = 1.0f / (1.0f + exp(-sdot[0]));\n"
    "    device float* st = shexp + (ulong)tok * n_embd;\n"
    "    for (uint i = tid; i < n_embd; i += T) st[i] = st[i] * sg;\n"
    "}\n"
    "\n"
    ;
}

NSString *fm_kernel_msl_moe(void)
{
    NSMutableString *source = [NSMutableString string];

    [source appendString:fm_moe_msl_common()];
    [source appendString:fm_moe_msl_router()];
    [source appendString:fm_moe_msl_sort()];
    [source appendString:fm_moe_msl_half_dots()];
    [source appendString:fm_moe_msl_grouped()];
    [source appendString:fm_moe_msl_shared()];
    return source;
}
