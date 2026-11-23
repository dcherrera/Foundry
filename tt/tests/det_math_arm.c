/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file det_math_arm.c
 * @brief Instantiates kernels/ft_det_math.h as a DetMathArm table. Compiled once per arm:
 *        no flags -> det_math_arm_production(); with DET_MATH_ARM_FN / DET_MATH_ARM_NAME and
 *        an FT_DET_* constant override -> a red arm (see Makefile.detmath). The same file is
 *        the rv32 text-size probe (`make -f Makefile.detmath size`).
 *
 * @status None
 * @issues None
 * @todo None
 */
#include "../kernels/ft_det_math.h"
#include "det_math_arm.h"

#ifndef DET_MATH_ARM_FN
#define DET_MATH_ARM_FN det_math_arm_production
#define DET_MATH_ARM_NAME "production"
#endif

static uint32_t arm_add(uint32_t a, uint32_t b) { return ft_det_add(a, b); }
static uint32_t arm_mul(uint32_t a, uint32_t b) { return ft_det_mul(a, b); }
static uint32_t arm_exp(uint32_t x) { return ft_det_exp(x); }
static uint32_t arm_softplus(uint32_t x) { return ft_det_softplus(x); }
static uint32_t arm_sigmoid(uint32_t x) { return ft_det_sigmoid(x); }
static uint32_t arm_gdn_beta(uint32_t b) { return ft_det_gdn_beta(b); }
static uint32_t arm_gdn_g(uint32_t alpha, uint32_t dt, uint32_t ssm_a)
{
    return ft_det_gdn_g(alpha, dt, ssm_a);
}
static uint32_t arm_gdn_decay(uint32_t g) { return ft_det_gdn_decay(g); }
static uint32_t arm_gdn_beta_reference(uint32_t b) { return ft_det_gdn_beta_reference(b); }
static uint32_t arm_gdn_gate(uint32_t alpha, uint32_t dt, uint32_t ssm_a)
{
    return ft_det_gdn_gate(alpha, dt, ssm_a);
}

const DetMathArm *DET_MATH_ARM_FN(void)
{
    static const DetMathArm arm = {DET_MATH_ARM_NAME,     arm_add,      arm_mul,
                                   arm_exp,               arm_softplus, arm_sigmoid,
                                   arm_gdn_beta,          arm_gdn_beta_reference,
                                   arm_gdn_g,             arm_gdn_decay, arm_gdn_gate};
    return &arm;
}
