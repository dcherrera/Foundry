/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_kernel_ssm_chunked.m
 * @brief Host dispatch and MSL source for the ssm_chunked kernel family.
 *
 * @status Stub. Contributes an empty MSL fragment so the library assembles;
 *   no kernels implemented yet.
 * @issues None
 * @todo Implement this family. Prefer no MSL preprocessor macros at all —
 *   fragments share one translation unit, so a stray #define leaks into
 *   every family declared after this one. Prefix shader-side identifiers
 *   with fmssm_chunked_.
 */

#include "metal/kernels/fm_kernel_ssm_chunked.h"

#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"

NSString *fm_kernel_msl_ssm_chunked(void)
{
    return @"";
}
