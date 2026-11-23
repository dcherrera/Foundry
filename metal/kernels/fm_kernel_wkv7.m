/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_wkv7.m
 * @brief Host dispatch and MSL source for the WKV-7 recurrence kernels.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_wkv7.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

void fm_wkv7_one(fm_buffer *r, fm_buffer *w, fm_buffer *k, fm_buffer *v,
                 fm_buffer *neg_kk, fm_buffer *kk_a,
                 fm_buffer *state, fm_buffer *y_out,
                 int n_head, int head_size) {
    if (!fm_metal_is_initialized()) return;

    fm_buffer *bufs[] = { r, w, k, v, neg_kk, kk_a, state, y_out };
    uint32_t consts[] = { (uint32_t)n_head, (uint32_t)head_size };
    /* One threadgroup per head, head_size threads per group */
    fm_dispatch_threadgroups(@"wkv7_one", bufs, 8, consts, 2,
                            n_head, head_size);
}

void fm_wkv7_seq(fm_buffer *r, fm_buffer *w, fm_buffer *k, fm_buffer *v,
                 fm_buffer *neg_kk, fm_buffer *kk_a,
                 fm_buffer *state, fm_buffer *y_out,
                 int n_head, int head_size, int t_len) {
    if (!fm_metal_is_initialized()) return;

    fm_buffer *bufs[] = { r, w, k, v, neg_kk, kk_a, state, y_out };
    uint32_t consts[] = { (uint32_t)n_head, (uint32_t)head_size, (uint32_t)t_len };
    fm_dispatch_threadgroups(@"wkv7_seq", bufs, 8, consts, 3,
                            n_head, head_size);
}

NSString *fm_kernel_msl_wkv7(void)
{
    return @
    "/* ============================================================\n"
    " * WKV-7 single timestep (Generalized Delta Rule)\n"
    " *\n"
    " * One threadgroup per head, head_size threads per group.\n"
    " * Uses threadgroup memory for cooperative loading.\n"
    " * ============================================================ */\n"
    "\n"
    "kernel void wkv7_one(\n"
    "    device const float* r        [[buffer(0)]],\n"
    "    device const float* w        [[buffer(1)]],\n"
    "    device const float* k        [[buffer(2)]],\n"
    "    device const float* v        [[buffer(3)]],\n"
    "    device const float* neg_kk   [[buffer(4)]],\n"
    "    device const float* kk_a     [[buffer(5)]],\n"
    "    device float* state          [[buffer(6)]],\n"
    "    device float* y              [[buffer(7)]],\n"
    "    constant uint& n_head        [[buffer(8)]],\n"
    "    constant uint& head_size     [[buffer(9)]],\n"
    "    uint h   [[threadgroup_position_in_grid]],\n"
    "    uint i   [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    if (h >= n_head || i >= head_size) return;\n"
    "    uint n = head_size;\n"
    "    uint offset = h * n;\n"
    "    uint sbase = (h * n + i) * n;\n"
    "\n"
    "    // Step 1: sa = dot(-kk, state_row)\n"
    "    float sa = 0.0f;\n"
    "    for (uint j = 0; j < n; ++j) {\n"
    "        sa += neg_kk[offset + j] * state[sbase + j];\n"
    "    }\n"
    "\n"
    "    float vv = v[offset + i];\n"
    "\n"
    "    // Step 2: state update + output\n"
    "    float y_out = 0.0f;\n"
    "    for (uint j = 0; j < n; ++j) {\n"
    "        float s = state[sbase + j];\n"
    "        s = s * w[offset + j] + k[offset + j] * vv + sa * kk_a[offset + j];\n"
    "        state[sbase + j] = s;\n"
    "        y_out += s * r[offset + j];\n"
    "    }\n"
    "    y[offset + i] = y_out;\n"
    "}\n"
    "\n"
    "/* WKV-7 sequence kernel */\n"
    "kernel void wkv7_seq(\n"
    "    device const float* r        [[buffer(0)]],\n"
    "    device const float* w        [[buffer(1)]],\n"
    "    device const float* k        [[buffer(2)]],\n"
    "    device const float* v        [[buffer(3)]],\n"
    "    device const float* neg_kk   [[buffer(4)]],\n"
    "    device const float* kk_a     [[buffer(5)]],\n"
    "    device float* state          [[buffer(6)]],\n"
    "    device float* y              [[buffer(7)]],\n"
    "    constant uint& n_head        [[buffer(8)]],\n"
    "    constant uint& head_size     [[buffer(9)]],\n"
    "    constant uint& t_len         [[buffer(10)]],\n"
    "    uint h   [[threadgroup_position_in_grid]],\n"
    "    uint i   [[thread_position_in_threadgroup]]\n"
    ") {\n"
    "    if (h >= n_head || i >= head_size) return;\n"
    "    uint n = head_size;\n"
    "    uint c = n_head * n;\n"
    "    uint offset = h * n;\n"
    "    uint sbase = (h * n + i) * n;\n"
    "\n"
    "    for (uint t = 0; t < t_len; ++t) {\n"
    "        uint toff = t * c + offset;\n"
    "        float sa = 0.0f;\n"
    "        for (uint j = 0; j < n; ++j)\n"
    "            sa += neg_kk[toff + j] * state[sbase + j];\n"
    "        float vv = v[toff + i];\n"
    "        float y_out = 0.0f;\n"
    "        for (uint j = 0; j < n; ++j) {\n"
    "            float s = state[sbase + j];\n"
    "            s = s * w[toff + j] + k[toff + j] * vv + sa * kk_a[toff + j];\n"
    "            state[sbase + j] = s;\n"
    "            y_out += s * r[toff + j];\n"
    "        }\n"
    "        y[t * c + offset + i] = y_out;\n"
    "    }\n"
    "}\n"
    "\n"
    ;
}
