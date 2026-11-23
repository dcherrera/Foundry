/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_probe.m
 * @brief Host dispatch and MSL source for the function-constant probe kernel.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_probe.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

void fm_constant_probe(fm_buffer *out, int count, const fm_fc_values *constants) {
    if (!fm_metal_is_initialized() || !out || count <= 0) return;

    fm_buffer *bufs[] = { out };
    uint32_t runtime_consts[] = { (uint32_t)count };
    fm_dispatch_1d_specialized(@"constant_probe", constants, bufs, 1,
                               runtime_consts, 1, count);
}

NSString *fm_kernel_msl_probe(void)
{
    return @
    "/* ============================================================\n"
    " * Function-constant probe (diagnostic, not a compute path)\n"
    " *\n"
    " * Reports the constants the pipeline was specialized with, so the\n"
    " * host can prove specialization reached the shader compiler and not\n"
    " * just the pipeline cache key. is_function_constant_defined lets an\n"
    " * unset constant fall back to a default instead of failing the build.\n"
    " * ============================================================ */\n"
    "\n"
    "constant int   fc_probe_repeat_in [[function_constant(0)]];\n"
    "constant short fc_probe_scale_in  [[function_constant(1)]];\n"
    "constant bool  fc_probe_flag_in   [[function_constant(2)]];\n"
    "\n"
    "constant int   fc_probe_repeat =\n"
    "    is_function_constant_defined(fc_probe_repeat_in) ? fc_probe_repeat_in : 0;\n"
    "constant short fc_probe_scale =\n"
    "    is_function_constant_defined(fc_probe_scale_in) ? fc_probe_scale_in : short(1);\n"
    "constant bool  fc_probe_flag =\n"
    "    is_function_constant_defined(fc_probe_flag_in) ? fc_probe_flag_in : false;\n"
    "\n"
    "kernel void constant_probe(\n"
    "    device float* out    [[buffer(0)]],\n"
    "    constant uint& count [[buffer(1)]],\n"
    "    uint gid [[thread_position_in_grid]]\n"
    ") {\n"
    "    if (gid >= count) return;\n"
    "    // Compile-time loop bound: unrolls, and a triangular sum makes an\n"
    "    // off-by-one in the constant visible in the output.\n"
    "    float sum = 0.0f;\n"
    "    for (int i = 1; i <= fc_probe_repeat; ++i) sum += float(i);\n"
    "    out[gid] = sum * float(fc_probe_scale)\n"
    "             + (fc_probe_flag ? 1000.0f : 0.0f)\n"
    "             + float(gid);\n"
    "}\n"
    "\n"
    ;
}
