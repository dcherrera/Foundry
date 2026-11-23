/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_norm.m
 * @brief Host dispatch and MSL source for the normalization kernels.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_norm.h"

#include <string.h>

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

/* MSL function-constant index owned by this family (see the header).
 * The shader spells the same index literally in its
 * [[function_constant(110)]] attribute — change one and change both. */
#define FM_NORM_FC_QK_HEAD_DIM 110

/* Threads per row-reduction threadgroup: enough SIMD groups to keep the
 * reduction shallow, capped so the 32-slot staging array always suffices. */
#define FM_NORM_MAX_ROW_THREADS 256

/* The dispatch helpers carry runtime scalars as uint32; a float rides
 * through as its bit pattern and the kernel reads it back with as_type.
 * Done explicitly rather than by punning the argument type so the trip
 * through an integer channel is visible on both sides. */
static uint32_t fmnorm_float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* Rows are normalized independently, so one threadgroup per row with a
 * cooperative reduction inside it. `weight` may be NULL: the slot still
 * needs a bound buffer, so x stands in and has_weight tells the kernel
 * never to read it. */
static void fmnorm_dispatch_rows(NSString *kernel_name, fm_buffer *x,
                                  fm_buffer *weight, fm_buffer *out,
                                  int rows, int n, float eps) {
    if (!fm_metal_is_initialized() || !x || !out || rows <= 0 || n <= 0) return;

    fm_buffer *bufs[] = { x, weight ? weight : x, out };
    uint32_t consts[] = { (uint32_t)n, weight ? 1u : 0u, fmnorm_float_bits(eps) };
    int tpg = n < FM_NORM_MAX_ROW_THREADS ? n : FM_NORM_MAX_ROW_THREADS;
    fm_dispatch_threadgroups(kernel_name, bufs, 3, consts, 3, rows, tpg);
}

void fm_softmax(fm_buffer *x, fm_buffer *out, int rows, int cols) {
    if (!fm_metal_is_initialized()) return;

    fm_buffer *bufs[] = { x, out };
    uint32_t consts[] = { (uint32_t)rows, (uint32_t)cols };
    /* One threadgroup per row, min(cols, 256) threads per group for reduction */
    int tpg = cols < 256 ? cols : 256;
    fm_dispatch_threadgroups(@"softmax_row", bufs, 2, consts, 2, rows, tpg);
}

void fm_rms_norm(fm_buffer *x, fm_buffer *weight, fm_buffer *out,
                 int n, float eps) {
    /* One vector is the one-row case of the batched kernel — same code path,
     * so the two forms cannot drift apart. */
    fmnorm_dispatch_rows(@"rms_norm_nx", x, weight, out, 1, n, eps);
}

void fm_rms_norm_nx(fm_buffer *x, fm_buffer *weight, fm_buffer *out,
                    int rows, int n, float eps) {
    fmnorm_dispatch_rows(@"rms_norm_nx", x, weight, out, rows, n, eps);
}

void fm_qk_rms_norm(fm_buffer *x, fm_buffer *weight, fm_buffer *out,
                    int n_heads, int head_dim, float eps) {
    if (!fm_metal_is_initialized() || !x || !out || n_heads <= 0 || head_dim <= 0)
        return;

    /* head_dim is a function constant so the per-lane loop bound is known at
     * shader-compile time and unrolls; one SIMD group per head means the
     * reduction is a single simd_sum with no threadgroup memory or barrier. */
    fm_fc_values fc;
    fm_fc_reset(&fc);
    fm_fc_set_int(&fc, FM_NORM_FC_QK_HEAD_DIM, head_dim);

    fm_bufref refs[] = { fm_bufref_at(x, 0),
                         fm_bufref_at(weight ? weight : x, 0),
                         fm_bufref_at(out, 0) };
    uint32_t consts[] = { weight ? 1u : 0u, fmnorm_float_bits(eps) };
    fm_dispatch_threadgroups_refs(@"qk_rms_norm", &fc, refs, 3, consts, 2,
                                  n_heads, 32);
}

NSString *fm_kernel_msl_norm(void)
{
    return @
    "/* ============================================================\n"
    " * Normalization family\n"
    " *\n"
    " * Every helper and constant here is prefixed fmnorm_, and the\n"
    " * fragment defines no preprocessor macro at all: the families are\n"
    " * concatenated into ONE MSL translation unit, so an unfenced macro or\n"
    " * a bare helper name would collide with whatever is assembled after\n"
    " * this fragment.\n"
    " *\n"
    " * eps arrives as a uint bit pattern because the dispatch helpers\n"
    " * carry runtime scalars as uint32; as_type<float> reads it back.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int fmnorm_fc_qk_head_dim_in [[function_constant(110)]];\n"
    "/* The host always specializes qk_rms_norm, so this default exists only\n"
    " * because MSL requires the unspecialized pipeline to still build. It is\n"
    " * the qwen35moe head_dim. */\n"
    "constant int fmnorm_fc_qk_head_dim =\n"
    "    is_function_constant_defined(fmnorm_fc_qk_head_dim_in)\n"
    "        ? fmnorm_fc_qk_head_dim_in : 256;\n"
    "\n"
    "/* ============================================================\n"
    " * Softmax (per-row, numerically stable)\n"
    " *\n"
    " * One threadgroup per row. Threads cooperate on reduction.\n"
    " * Three passes: max -> exp/sum -> normalize.\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void softmax_row(\n"
    "    device const float* x    [[buffer(0)]],\n"
    "    device float* out        [[buffer(1)]],\n"
    "    constant uint& rows      [[buffer(2)]],\n"
    "    constant uint& cols      [[buffer(3)]],\n"
    "    uint row   [[threadgroup_position_in_grid]],\n"
    "    uint tid   [[thread_position_in_threadgroup]],\n"
    "    uint tsize [[threads_per_threadgroup]]\n"
    ") {\n"
    "    if (row >= rows) return;\n"
    "    uint base = row * cols;\n"
    "\n"
    "    // Pass 1: find max (thread-local then reduce)\n"
    "    float local_max = -HUGE_VALF;\n"
    "    for (uint c = tid; c < cols; c += tsize) {\n"
    "        float val = x[base + c];\n"
    "        local_max = max(local_max, val);\n"
    "    }\n"
    "\n"
    "    // SIMD reduction for max\n"
    "    threadgroup float shared_max[32];\n"
    "    float smax = simd_max(local_max);\n"
    "    if (simd_is_first()) shared_max[tid / 32] = smax;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid < 32) {\n"
    "        uint n_simd = (tsize + 31) / 32;\n"
    "        smax = (tid < n_simd) ? shared_max[tid] : -HUGE_VALF;\n"
    "        smax = simd_max(smax);\n"
    "        shared_max[0] = smax;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    float row_max = shared_max[0];\n"
    "\n"
    "    // Pass 2: exp and sum\n"
    "    float local_sum = 0.0f;\n"
    "    for (uint c = tid; c < cols; c += tsize) {\n"
    "        float e = fast::exp(x[base + c] - row_max);\n"
    "        out[base + c] = e;\n"
    "        local_sum += e;\n"
    "    }\n"
    "\n"
    "    // SIMD reduction for sum\n"
    "    threadgroup float shared_sum[32];\n"
    "    float ssum = simd_sum(local_sum);\n"
    "    if (simd_is_first()) shared_sum[tid / 32] = ssum;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid < 32) {\n"
    "        uint n_simd = (tsize + 31) / 32;\n"
    "        ssum = (tid < n_simd) ? shared_sum[tid] : 0.0f;\n"
    "        ssum = simd_sum(ssum);\n"
    "        shared_sum[0] = ssum;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    float inv_sum = 1.0f / shared_sum[0];\n"
    "\n"
    "    // Pass 3: normalize\n"
    "    for (uint c = tid; c < cols; c += tsize) {\n"
    "        out[base + c] *= inv_sum;\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * RMS normalization, batched over rows\n"
    " *\n"
    " * out[r][i] = x[r][i] * rsqrt(mean(x[r]^2) + eps) * weight[i]\n"
    " *\n"
    " * One threadgroup per row; the whole group cooperates on that row's\n"
    " * sum of squares. The row stride IS n — rows are contiguous, with no\n"
    " * padding. `weight` is shared by every row and is n long, not\n"
    " * rows * n; when has_weight is 0 the buffer is never read (the host\n"
    " * binds x in that slot, since the argument still has to be bound).\n"
    " *\n"
    " * The single-vector case is just rows == 1, so there is deliberately\n"
    " * no separate rms_norm kernel to fall out of sync with this one.\n"
    " * ============================================================ */\n"
    "\n"
    "/* Sum of squares of `row` reduced across the threadgroup, returned as\n"
    " * the normalization factor. `scratch` needs 32 slots (one per SIMD\n"
    " * group). Every thread in the group must call this exactly once — it\n"
    " * carries threadgroup barriers. */\n"
    "inline float fmnorm_row_inv_rms(device const float* row, uint n, float eps,\n"
    "                                 uint tid, uint tsize,\n"
    "                                 threadgroup float* scratch) {\n"
    "    float local_sum_sq = 0.0f;\n"
    "    for (uint i = tid; i < n; i += tsize) {\n"
    "        float val = row[i];\n"
    "        local_sum_sq += val * val;\n"
    "    }\n"
    "\n"
    "    float ssum = simd_sum(local_sum_sq);\n"
    "    if (simd_is_first()) scratch[tid / 32] = ssum;\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    if (tid < 32) {\n"
    "        uint n_simd = (tsize + 31) / 32;\n"
    "        ssum = (tid < n_simd) ? scratch[tid] : 0.0f;\n"
    "        ssum = simd_sum(ssum);\n"
    "        scratch[0] = ssum;\n"
    "    }\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "\n"
    "    return precise::rsqrt(scratch[0] / float(n) + eps);\n"
    "}\n"
    "\n"
    "kernel void rms_norm_nx(\n"
    "    device const float* x       [[buffer(0)]],\n"
    "    device const float* weight  [[buffer(1)]],\n"
    "    device float* out           [[buffer(2)]],\n"
    "    constant uint& n            [[buffer(3)]],\n"
    "    constant uint& has_weight   [[buffer(4)]],\n"
    "    constant uint& eps_bits     [[buffer(5)]],\n"
    "    uint row   [[threadgroup_position_in_grid]],\n"
    "    uint tid   [[thread_position_in_threadgroup]],\n"
    "    uint tsize [[threads_per_threadgroup]]\n"
    ") {\n"
    "    threadgroup float scratch[32];\n"
    "    uint base = row * n;\n"
    "    float inv_rms = fmnorm_row_inv_rms(x + base, n, as_type<float>(eps_bits),\n"
    "                                        tid, tsize, scratch);\n"
    "\n"
    "    for (uint i = tid; i < n; i += tsize) {\n"
    "        float scaled = x[base + i] * inv_rms;\n"
    "        out[base + i] = has_weight ? scaled * weight[i] : scaled;\n"
    "    }\n"
    "}\n"
    "\n"
    "/* ============================================================\n"
    " * QK-RMSNorm — per-head RMS over head_dim\n"
    " *\n"
    " * One threadgroup of exactly ONE SIMD group (32 lanes) per head, so\n"
    " * the reduction is a single simd_sum: no threadgroup memory, no\n"
    " * barriers. head_dim is a function constant, which makes the lane\n"
    " * loop's trip count (head_dim / 32) compile-time and unrollable.\n"
    " *\n"
    " * The weight is SHARED ACROSS HEADS: it is head_dim long and head h\n"
    " * indexes weight[i], NOT weight[head * head_dim + i]. GGUF stores one\n"
    " * pre-baked norm weight per tensor and every head applies the same\n"
    " * one. Indexing it per head reads past the buffer for h > 0.\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void qk_rms_norm(\n"
    "    device const float* x       [[buffer(0)]],\n"
    "    device const float* weight  [[buffer(1)]],\n"
    "    device float* out           [[buffer(2)]],\n"
    "    constant uint& has_weight   [[buffer(3)]],\n"
    "    constant uint& eps_bits     [[buffer(4)]],\n"
    "    uint head [[threadgroup_position_in_grid]],\n"
    "    uint lane [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    const uint head_dim = uint(fmnorm_fc_qk_head_dim);\n"
    "    const uint base = head * head_dim;\n"
    "\n"
    "    float local_sum_sq = 0.0f;\n"
    "    for (uint i = lane; i < head_dim; i += 32u) {\n"
    "        float val = x[base + i];\n"
    "        local_sum_sq += val * val;\n"
    "    }\n"
    "    float inv_rms = precise::rsqrt(simd_sum(local_sum_sq) / float(head_dim)\n"
    "                                   + as_type<float>(eps_bits));\n"
    "\n"
    "    for (uint i = lane; i < head_dim; i += 32u) {\n"
    "        float scaled = x[base + i] * inv_rms;\n"
    "        out[base + i] = has_weight ? scaled * weight[i] : scaled;\n"
    "    }\n"
    "}\n"
    "\n"
    ;
}
