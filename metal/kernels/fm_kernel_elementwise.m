/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_elementwise.m
 * @brief Host dispatch and MSL source for the elementwise kernels.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_elementwise.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

/* Threads per group. Deliberately NOT the device maximum: fm_dispatch_1d
 * sizes a threadgroup as min(total_threads, maxTotalThreadsPerThreadgroup),
 * which is 1024 here, so a 2048-element vector would land in TWO
 * threadgroups — two of the M4's ten GPU cores, with eight idle. A
 * threadgroup is the unit of core assignment, so a smaller group means more
 * groups and more cores. 256 = 8 SIMD groups, still enough per group to
 * hide latency. Measured on 2048 elements: 8 groups instead of 2. */
#define FM_ELEM_THREADS_PER_GROUP 256

/* Threads a float4 kernel needs to cover n elements, tail included. */
static int fm_elem_vec4_threads(int n) {
    return (n + 3) / 4;
}

static int fm_elem_groups(int threads) {
    return (threads + FM_ELEM_THREADS_PER_GROUP - 1) / FM_ELEM_THREADS_PER_GROUP;
}

/* Grid is rounded up to whole threadgroups; the kernels bound themselves on
 * n, so the trailing threads of the last group simply do nothing. */
static void fm_elem_dispatch_unary(NSString *kernel_name, fm_buffer *x,
                                   fm_buffer *out, int n, int threads) {
    if (!fm_metal_is_initialized() || !x || !out || n <= 0) return;

    fm_buffer *bufs[] = { x, out };
    uint32_t consts[] = { (uint32_t)n };
    fm_dispatch_threadgroups(kernel_name, bufs, 2, consts, 1,
                             fm_elem_groups(threads), FM_ELEM_THREADS_PER_GROUP);
}

static void fm_elem_dispatch_binary(NSString *kernel_name, fm_buffer *a,
                                    fm_buffer *b, fm_buffer *out, int n,
                                    int threads) {
    if (!fm_metal_is_initialized() || !a || !b || !out || n <= 0) return;

    fm_buffer *bufs[] = { a, b, out };
    uint32_t consts[] = { (uint32_t)n };
    fm_dispatch_threadgroups(kernel_name, bufs, 3, consts, 1,
                             fm_elem_groups(threads), FM_ELEM_THREADS_PER_GROUP);
}

void fm_silu(fm_buffer *x, fm_buffer *out, int n) {
    fm_elem_dispatch_unary(@"silu", x, out, n, n);
}

void fm_sigmoid(fm_buffer *x, fm_buffer *out, int n) {
    fm_elem_dispatch_unary(@"sigmoid", x, out, n, n);
}

void fm_add(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n) {
    fm_elem_dispatch_binary(@"elementwise_add", a, b, out, n, n);
}

void fm_mul(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n) {
    fm_elem_dispatch_binary(@"elementwise_mul", a, b, out, n, n);
}

void fm_silu_4(fm_buffer *x, fm_buffer *out, int n) {
    fm_elem_dispatch_unary(@"silu_4", x, out, n, fm_elem_vec4_threads(n));
}

void fm_sigmoid_4(fm_buffer *x, fm_buffer *out, int n) {
    fm_elem_dispatch_unary(@"sigmoid_4", x, out, n, fm_elem_vec4_threads(n));
}

void fm_add_4(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n) {
    fm_elem_dispatch_binary(@"elementwise_add_4", a, b, out, n,
                            fm_elem_vec4_threads(n));
}

void fm_mul_4(fm_buffer *a, fm_buffer *b, fm_buffer *out, int n) {
    fm_elem_dispatch_binary(@"elementwise_mul_4", a, b, out, n,
                            fm_elem_vec4_threads(n));
}

NSString *fm_kernel_msl_elementwise(void)
{
    return @
    "/* ============================================================\n"
    " * Elementwise activations and binary ops\n"
    " *\n"
    " * Two kernels per op:\n"
    " *   <name>    one element per thread\n"
    " *   <name>_4  four elements per thread (float4 load/store)\n"
    " *\n"
    " * The op itself lives in a functor and the loop/tail logic in one\n"
    " * template, so the scalar and float4 forms of an op cannot drift\n"
    " * apart and every kernel body is a single call. C++ templates, NOT\n"
    " * preprocessor macros: the fragments are concatenated into ONE MSL\n"
    " * translation unit, and a macro defined here would still be defined\n"
    " * for every family assembled afterwards. Every name is prefixed\n"
    " * fmelem_ for the same reason.\n"
    " *\n"
    " * OCCUPANCY: consecutive lanes read consecutive elements in both\n"
    " * forms, so every one of the 32 lanes in a SIMD group has work and\n"
    " * each load coalesces (128 bytes scalar, 512 bytes float4). Only the\n"
    " * final partial group of the grid is underfed. The _4 form's wider\n"
    " * loads do NOT make it faster on a 10-core M4 — see the measurements\n"
    " * in fm_kernel_elementwise.h — because it also quarters the thread\n"
    " * count, and these kernels are launch-bound below the memory roof.\n"
    " *\n"
    " * The _4 form rounds the grid up to ceil(n/4); the final thread takes\n"
    " * a scalar path over the 1-3 leftover elements, so n does not have to\n"
    " * be a multiple of 4. Buffers are bound at offset 0 and a Metal\n"
    " * allocation is page aligned, so the float4 accesses are always\n"
    " * 16-byte aligned.\n"
    " * ============================================================ */\n"
    "\n"
    "struct fmelem_silu {\n"
    "    template <typename T> inline T operator()(T v) const {\n"
    "        return v / (T(1.0f) + fast::exp(-v));\n"
    "    }\n"
    "};\n"
    "struct fmelem_sigmoid {\n"
    "    template <typename T> inline T operator()(T v) const {\n"
    "        return T(1.0f) / (T(1.0f) + fast::exp(-v));\n"
    "    }\n"
    "};\n"
    "struct fmelem_add {\n"
    "    template <typename T> inline T operator()(T a, T b) const { return a + b; }\n"
    "};\n"
    "struct fmelem_mul {\n"
    "    template <typename T> inline T operator()(T a, T b) const { return a * b; }\n"
    "};\n"
    "\n"
    "template <typename Op>\n"
    "inline void fmelem_unary_4(device const float* x, device float* out,\n"
    "                           uint n, uint idx, Op op) {\n"
    "    uint base = idx * 4u;\n"
    "    if (base + 4u <= n) {\n"
    "        ((device float4*)out)[idx] = op(((device const float4*)x)[idx]);\n"
    "    } else {\n"
    "        for (uint i = base; i < n; ++i) out[i] = op(x[i]);\n"
    "    }\n"
    "}\n"
    "\n"
    "template <typename Op>\n"
    "inline void fmelem_binary_4(device const float* a, device const float* b,\n"
    "                            device float* out, uint n, uint idx, Op op) {\n"
    "    uint base = idx * 4u;\n"
    "    if (base + 4u <= n) {\n"
    "        ((device float4*)out)[idx] = op(((device const float4*)a)[idx],\n"
    "                                        ((device const float4*)b)[idx]);\n"
    "    } else {\n"
    "        for (uint i = base; i < n; ++i) out[i] = op(a[i], b[i]);\n"
    "    }\n"
    "}\n"
    "\n"
    "kernel void silu(\n"
    "    device const float* x [[buffer(0)]],\n"
    "    device float* out     [[buffer(1)]],\n"
    "    constant uint& n      [[buffer(2)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (idx < n) out[idx] = fmelem_silu()(x[idx]);\n"
    "}\n"
    "\n"
    "kernel void silu_4(\n"
    "    device const float* x [[buffer(0)]],\n"
    "    device float* out     [[buffer(1)]],\n"
    "    constant uint& n      [[buffer(2)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    fmelem_unary_4(x, out, n, idx, fmelem_silu());\n"
    "}\n"
    "\n"
    "kernel void sigmoid(\n"
    "    device const float* x [[buffer(0)]],\n"
    "    device float* out     [[buffer(1)]],\n"
    "    constant uint& n      [[buffer(2)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (idx < n) out[idx] = fmelem_sigmoid()(x[idx]);\n"
    "}\n"
    "\n"
    "kernel void sigmoid_4(\n"
    "    device const float* x [[buffer(0)]],\n"
    "    device float* out     [[buffer(1)]],\n"
    "    constant uint& n      [[buffer(2)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    fmelem_unary_4(x, out, n, idx, fmelem_sigmoid());\n"
    "}\n"
    "\n"
    "kernel void elementwise_add(\n"
    "    device const float* a [[buffer(0)]],\n"
    "    device const float* b [[buffer(1)]],\n"
    "    device float* out     [[buffer(2)]],\n"
    "    constant uint& n      [[buffer(3)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (idx < n) out[idx] = fmelem_add()(a[idx], b[idx]);\n"
    "}\n"
    "\n"
    "kernel void elementwise_add_4(\n"
    "    device const float* a [[buffer(0)]],\n"
    "    device const float* b [[buffer(1)]],\n"
    "    device float* out     [[buffer(2)]],\n"
    "    constant uint& n      [[buffer(3)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    fmelem_binary_4(a, b, out, n, idx, fmelem_add());\n"
    "}\n"
    "\n"
    "kernel void elementwise_mul(\n"
    "    device const float* a [[buffer(0)]],\n"
    "    device const float* b [[buffer(1)]],\n"
    "    device float* out     [[buffer(2)]],\n"
    "    constant uint& n      [[buffer(3)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (idx < n) out[idx] = fmelem_mul()(a[idx], b[idx]);\n"
    "}\n"
    "\n"
    "kernel void elementwise_mul_4(\n"
    "    device const float* a [[buffer(0)]],\n"
    "    device const float* b [[buffer(1)]],\n"
    "    device float* out     [[buffer(2)]],\n"
    "    constant uint& n      [[buffer(3)]],\n"
    "    uint idx [[thread_position_in_grid]]\n"
    ") {\n"
    "    fmelem_binary_4(a, b, out, n, idx, fmelem_mul());\n"
    "}\n"
    "\n"
    ;
}
