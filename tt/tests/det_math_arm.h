/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file det_math_arm.h
 * @brief One "arm" of kernels/ft_det_math.h as a table of out-of-line functions, so
 *        test_det_math.c can run the same checks against the production header and against
 *        builds of it with one constant mutated (the red arms).
 *
 * @status None
 * @issues None
 * @todo None
 */
#ifndef DET_MATH_ARM_H
#define DET_MATH_ARM_H

#include <stdint.h>

typedef struct DetMathArm {
    const char *name;
    uint32_t (*add)(uint32_t a, uint32_t b);
    uint32_t (*mul)(uint32_t a, uint32_t b);
    uint32_t (*exp)(uint32_t x);
    uint32_t (*softplus)(uint32_t x);
    uint32_t (*sigmoid)(uint32_t x);
    uint32_t (*gdn_beta)(uint32_t b);           /* fast (card) path */
    uint32_t (*gdn_beta_reference)(uint32_t b); /* reference sigmoid */
    uint32_t (*gdn_g)(uint32_t alpha, uint32_t dt, uint32_t ssm_a);
    uint32_t (*gdn_decay)(uint32_t g);
    uint32_t (*gdn_gate)(uint32_t alpha, uint32_t dt, uint32_t ssm_a); /* fast fused decay */
} DetMathArm;

/** The header as shipped. */
const DetMathArm *det_math_arm_production(void);
/** Red arm: ft_det_pack's round-half constant 0x80 -> 0x7F (ties round up, not to even). */
const DetMathArm *det_math_arm_red_round_half(void);
/** Red arm: the exp polynomial's 1/6 coefficient replaced by 1/5 (0x33333333). */
const DetMathArm *det_math_arm_red_exp_coefficient(void);
/** Red arm: the fast log1p series' 1/3 coefficient replaced by 1/4 (0x20000000). */
const DetMathArm *det_math_arm_red_fast_log1p(void);
/** Red arm: the fast beta's Newton reciprocal stops after 2 steps instead of 3. */
const DetMathArm *det_math_arm_red_fast_newton(void);

#endif /* DET_MATH_ARM_H */
