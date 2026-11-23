/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_match_pools.h
 * @brief  Encoder policy for match mode (not part of the format): which tensors are layer roles,
 *         their coding order, and the pool of earlier tensors a target's partners come from.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * A layer role is a 2-D BF16 tensor named P + "layers." + L + suffix, with suffix one of (rank):
 * mlp.gate_proj (0), mlp.up_proj (1), mlp.down_proj (2), self_attn.q_proj / linear_attn.
 * in_proj_qkv (3), self_attn.k_proj (4), self_attn.v_proj (5), linear_attn.in_proj_z (6),
 * linear_attn.in_proj_a (7), linear_attn.in_proj_b (8), self_attn.o_proj / linear_attn.out_proj
 * (9), each ".weight". Readers (ranks 0, 1, 3-8) have row units, writers (2, 9) column units;
 * the unit length is the residual width H.
 *
 * Pool of target T (layer L, rank r): every role with the same P and H in layers L - o for each
 * pool offset o, the roles of layer L with rank < r, and T itself when its units are rows.
 */
#ifndef FOUNDRY_INGOT_MATCH_POOLS_H
#define FOUNDRY_INGOT_MATCH_POOLS_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/codec/ingot_predict.h"
#include "ingot/container/ingot_format.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { INGOT_MATCH_ROLE_SUFFIXES = 12, INGOT_MATCH_MAX_POOL_LAYERS = 8 };

/** What a tensor is to the match policy. */
typedef struct {
    int is_role;
    size_t prefix_len;          /* bytes of P */
    long layer;
    int rank;
    ingot_predict_axis axis;
    uint64_t units;
    uint64_t unit_len;
} ingot_match_role;

/**
 * Classify a tensor by name, dtype and shape. @return 1 for a layer role (out filled), else 0
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_match_role_of(const char *name, ingot_dtype dtype, int ndim, const uint64_t *shape,
                        ingot_match_role *out);

/**
 * Candidate names of pool member k for a target (k = 0 .. count-1, in pool order): fills name
 * (P + "layers." + layer + suffix) and the member's expected unit axis.
 * @return number of candidate names (layers x suffixes), or 0
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_match_pool_names(const char *target_name, const ingot_match_role *t,
                              const int *offsets, size_t offset_count, size_t k, char *name,
                              size_t name_size);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_MATCH_POOLS_H */
