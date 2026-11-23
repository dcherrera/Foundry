/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_predict_rules.h
 * @brief  Which tensors of a safetensors file the writer predicts from which partners
 *         (same-unit rules by tensor name and shape, within one file).
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Rules, for one layer prefix X (H = residual width; every tensor BF16 and 2-D):
 *   1. X.mlp.down_proj.weight [H, I], columns <- X.mlp.gate_proj.weight, X.mlp.up_proj.weight
 *      [I, H] rows.
 *   2. X.mlp.up_proj.weight [I, H], rows <- X.mlp.gate_proj.weight rows.
 *   3. X.linear_attn.in_proj_z.weight [V, H], rows <- X.linear_attn.in_proj_qkv.weight [R, H]
 *      rows R - V .. R - 1 (v is last).
 *   4. X.linear_attn.out_proj.weight [H, V], columns <- in_proj_qkv's v rows, in_proj_z rows.
 *   5. X.self_attn.o_proj.weight [H, nh hd], columns <- X.self_attn.v_proj.weight [nkv hd, H]
 *      rows, grouped-query mapped (hd = length of X.self_attn.q_norm.weight, g = nh / nkv).
 * A rule applies only when every partner exists in the same file with matching shapes.
 */
#ifndef FOUNDRY_INGOT_PREDICT_RULES_H
#define FOUNDRY_INGOT_PREDICT_RULES_H

#include <stddef.h>

#include "ingot/codec/ingot_predict.h"
#include "ingot/container/ingot_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The rule for tensor i of a layout, if any.
 *
 * @param l        Source layout
 * @param i        Tensor index in l
 * @param g        Out: geometry (partner entries 0: the caller fills in directory indices)
 * @param partner  Out: layout indices of the partners (g->partner_count of them)
 * @return         1 if a rule applies, else 0
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_predict_rule(const ingot_layout *l, size_t i, ingot_predict_geometry *g,
                       size_t partner[INGOT_PREDICT_MAX_PARTNERS]);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_PREDICT_RULES_H */
