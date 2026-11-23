/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_match_pools.c
 * @brief  Layer roles and match pools by tensor name (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_match_pools.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const struct {
    const char *suffix;
    int rank;
    ingot_predict_axis axis;
} ROLES[INGOT_MATCH_ROLE_SUFFIXES] = {
    {".mlp.gate_proj.weight", 0, INGOT_PREDICT_ROWS},
    {".mlp.up_proj.weight", 1, INGOT_PREDICT_ROWS},
    {".mlp.down_proj.weight", 2, INGOT_PREDICT_COLS},
    {".self_attn.q_proj.weight", 3, INGOT_PREDICT_ROWS},
    {".linear_attn.in_proj_qkv.weight", 3, INGOT_PREDICT_ROWS},
    {".self_attn.k_proj.weight", 4, INGOT_PREDICT_ROWS},
    {".self_attn.v_proj.weight", 5, INGOT_PREDICT_ROWS},
    {".linear_attn.in_proj_z.weight", 6, INGOT_PREDICT_ROWS},
    {".linear_attn.in_proj_a.weight", 7, INGOT_PREDICT_ROWS},
    {".linear_attn.in_proj_b.weight", 8, INGOT_PREDICT_ROWS},
    {".self_attn.o_proj.weight", 9, INGOT_PREDICT_COLS},
    {".linear_attn.out_proj.weight", 9, INGOT_PREDICT_COLS},
};

static const char LAYERS[] = "layers.";

int ingot_match_role_of(const char *name, ingot_dtype dtype, int ndim, const uint64_t *shape,
                        ingot_match_role *out) {
    if (!name || dtype != INGOT_DTYPE_BF16 || ndim != 2 || !shape[0] || !shape[1])
        return 0;
    const char *at = NULL;
    for (const char *p = strstr(name, LAYERS); p; p = strstr(p + 1, LAYERS))
        at = p; /* the last "layers." */
    if (!at || (at != name && at[-1] != '.'))
        return 0;
    const char *num = at + sizeof(LAYERS) - 1;
    char *end = NULL;
    long layer = strtol(num, &end, 10);
    if (end == num || layer < 0 || *end != '.')
        return 0;
    for (size_t k = 0; k < INGOT_MATCH_ROLE_SUFFIXES; k++) {
        if (strcmp(end, ROLES[k].suffix))
            continue;
        int rows = ROLES[k].axis == INGOT_PREDICT_ROWS;
        *out = (ingot_match_role){.is_role = 1, .prefix_len = (size_t)(at - name),
                                  .layer = layer, .rank = ROLES[k].rank, .axis = ROLES[k].axis,
                                  .units = rows ? shape[0] : shape[1],
                                  .unit_len = rows ? shape[1] : shape[0]};
        return 1;
    }
    return 0;
}

/* Pool order: for each offset (layer L - o), every suffix; then layer L's suffixes ranked below
 * the target; then the target itself (self). Index k walks that list. */
size_t ingot_match_pool_names(const char *target_name, const ingot_match_role *t,
                              const int *offsets, size_t offset_count, size_t k, char *name,
                              size_t name_size) {
    size_t per = INGOT_MATCH_ROLE_SUFFIXES;
    size_t total = (offset_count + 1) * per;
    if (!t->is_role || k >= total || offset_count > INGOT_MATCH_MAX_POOL_LAYERS)
        return 0;
    size_t group = k / per, s = k % per;
    long layer = t->layer;
    if (group < offset_count) {
        layer -= offsets[group];
        if (offsets[group] <= 0 || layer < 0) {
            name[0] = '\0';
            return total;
        }
    } else if (ROLES[s].rank >= t->rank) {
        name[0] = '\0';
        return total;
    }
    int n = snprintf(name, name_size, "%.*slayers.%ld%s", (int)t->prefix_len, target_name, layer,
                     ROLES[s].suffix);
    if (n < 0 || (size_t)n >= name_size)
        name[0] = '\0';
    return total;
}
