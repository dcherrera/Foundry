/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_convert.m
 * @brief Host dispatch and MSL source for the F32 <-> F16 conversion kernels.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_convert.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

void fm_convert_f32_to_f16(fm_buffer *in_f32, fm_buffer *out_f16, int n) {
    if (!fm_metal_is_initialized() || !in_f32 || !out_f16 || n <= 0) return;

    fm_buffer *bufs[] = { in_f32, out_f16 };
    uint32_t consts[] = { (uint32_t)n };
    fm_dispatch_1d(@"cvt_f32_f16", bufs, 2, consts, 1, n);
}

void fm_convert_f16_to_f32(fm_buffer *in_f16, fm_buffer *out_f32, int n) {
    if (!fm_metal_is_initialized() || !in_f16 || !out_f32 || n <= 0) return;

    fm_buffer *bufs[] = { in_f16, out_f32 };
    uint32_t consts[] = { (uint32_t)n };
    fm_dispatch_1d(@"cvt_f16_f32", bufs, 2, consts, 1, n);
}

NSString *fm_kernel_msl_convert(void)
{
    return @
    "kernel void cvt_f32_f16(device const float* in [[buffer(0)]], device half* out [[buffer(1)]],\n"
    "                        constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
    "    if (gid < n) out[gid] = half(in[gid]);\n"
    "}\n"
    "kernel void cvt_f16_f32(device const half* in [[buffer(0)]], device float* out [[buffer(1)]],\n"
    "                        constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
    "    if (gid < n) out[gid] = float(in[gid]);\n"
    "}\n"
    ;
}
