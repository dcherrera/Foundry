/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_registry.h
 * @brief The list of MSL source fragments, one per kernel family, and the
 *   assembler that concatenates them into the single shader source fm_init()
 *   compiles. Objective-C only — the public C API lives in metal/fm_metal.h.
 *
 * WHY FRAGMENTS RATHER THAN .metal FILES: the backend compiles its shaders at
 * runtime from an embedded string, so nothing has to ship or locate a
 * .metallib. Keeping that property while still giving each kernel family its
 * own file means each family owns a *fragment* of that string. All fragments
 * land in ONE MTLLibrary, so function-constant specialization and pipeline
 * caching work exactly as they did when the source was one literal.
 *
 * TO ADD A KERNEL FAMILY: create metal/kernels/fm_kernel_<family>.{h,m},
 * declare its fragment below, and add it to the table in the .m. Nothing else
 * in the backend needs to change.
 *
 * SHARED PREPROCESSOR NAMESPACE (footgun): the fragments are concatenated into
 * one MSL translation unit, so a `#define` in one family is visible to every
 * family after it. Prefix macros with the family name and `#undef` them at the
 * end of the fragment.
 *
 * @status All existing kernels are fragment-backed; the Phase-1 families
 *   (rope, q3_k, q5_k, q6_k, q8_0) are registered and return an empty string
 *   until their kernels land.
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_KERNEL_REGISTRY_H
#define FOUNDRY_METAL_KERNEL_REGISTRY_H

#import <Foundation/Foundation.h>

/* The whole shader source: the common preamble followed by every registered
 * fragment, in table order. Returns nil only if the table is empty. */
NSString *fm_kernel_library_source(void);

/* ================================================================
 * Per-family fragments
 *
 * Each returns MSL text ending in a blank line, or @"" if the family has
 * no kernels yet. Defined in the matching fm_kernel_<family>.m.
 * ================================================================ */

NSString *fm_kernel_msl_probe(void);
NSString *fm_kernel_msl_ssm(void);
NSString *fm_kernel_msl_ssm_chunked(void);
NSString *fm_kernel_msl_moe(void);
NSString *fm_kernel_msl_attn(void);
NSString *fm_kernel_msl_wkv7(void);
NSString *fm_kernel_msl_norm(void);
NSString *fm_kernel_msl_elementwise(void);
NSString *fm_kernel_msl_matvec_q8(void);
NSString *fm_kernel_msl_matvec_q4_k(void);
NSString *fm_kernel_msl_convert(void);
NSString *fm_kernel_msl_rope(void);
NSString *fm_kernel_msl_matvec_q3_k(void);
NSString *fm_kernel_msl_matvec_q5_k(void);
NSString *fm_kernel_msl_matvec_q6_k(void);
NSString *fm_kernel_msl_matvec_q8_0(void);

#endif /* FOUNDRY_METAL_KERNEL_REGISTRY_H */

/* Function-constant index allocation — a SECOND shared namespace.
 *
 * Every family's MSL fragment is concatenated into ONE MTLLibrary, so two
 * families declaring a function constant at the same index is a whole-library
 * compile failure that breaks every kernel in the backend, not just theirs.
 * Claim a block here before using it, and prefix constant NAMES by family too.
 *
 *    0- 2  probe
 *   30     matvec_q3_k
 *   50     matvec_q5_k
 *   60     matvec_q6_k
 *   70     matvec_q8_0   (reserved)
 *   80     matvec_q4_k   (reserved)
 *  100     rope          (reserved)
 *  110     norm          (reserved)
 *  120     elementwise   (reserved)
 */
