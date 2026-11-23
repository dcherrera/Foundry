/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_registry.m
 * @brief Assembles the per-family MSL fragments into the one shader source
 *   fm_init() hands to newLibraryWithSource:. See fm_kernel_registry.h for
 *   why the shaders are fragments of a string rather than .metal files.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "metal/kernels/fm_kernel_registry.h"

/* Shared by every fragment. Kept here, not in a family file, because every
 * family depends on it and no family owns it. */
static NSString *fm_kernel_msl_preamble(void)
{
    return @
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    ;
}

/* Concatenation order. It only has to be a valid MSL ordering — MSL needs a
 * declaration before its use, so a family that depends on another's helpers
 * must follow it. Kernels themselves are order-independent. */
typedef NSString *(*fm_kernel_msl_fn)(void);

static const fm_kernel_msl_fn fm_kernel_msl_table[] = {
    fm_kernel_msl_preamble,
    fm_kernel_msl_probe,
    fm_kernel_msl_wkv7,
    fm_kernel_msl_norm,
    fm_kernel_msl_elementwise,
    fm_kernel_msl_rope,
    fm_kernel_msl_matvec_q8,
    fm_kernel_msl_matvec_q4_k,
    fm_kernel_msl_matvec_q3_k,
    fm_kernel_msl_matvec_q5_k,
    fm_kernel_msl_matvec_q6_k,
    fm_kernel_msl_matvec_q8_0,
    fm_kernel_msl_convert,
    fm_kernel_msl_ssm,
    fm_kernel_msl_ssm_chunked,
    fm_kernel_msl_moe,
    fm_kernel_msl_attn,
};

NSString *fm_kernel_library_source(void)
{
    const size_t count = sizeof(fm_kernel_msl_table) / sizeof(fm_kernel_msl_table[0]);
    if (count == 0) return nil;

    NSMutableString *source = [NSMutableString string];
    for (size_t i = 0; i < count; i++) {
        NSString *fragment = fm_kernel_msl_table[i]();
        if (fragment) [source appendString:fragment];
    }
    return source;
}
