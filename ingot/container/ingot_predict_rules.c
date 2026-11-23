/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_predict_rules.c
 * @brief  Same-unit prediction rules by tensor name and shape (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_predict_rules.h"

#include <stdio.h>
#include <string.h>

/* Names longer than this never match a rule. */
enum { MAX_NAME = 1024 };

typedef struct {
    char prefix[MAX_NAME]; /* layer prefix X */
    const ingot_layout *layout;
} rule_ctx;

/* Tensor X + suffix if it is 2-D BF16 (or any dtype, 1-D when want_1d); *index set. */
static const ingot_layout_tensor *find(const rule_ctx *r, const char *suffix, int want_1d,
                                       size_t *index) {
    char name[2 * MAX_NAME];
    size_t at = 0;
    snprintf(name, sizeof(name), "%s%s", r->prefix, suffix);
    if (ingot_layout_find(r->layout, name, &at))
        return NULL;
    const ingot_layout_tensor *t = ingot_layout_tensor_at(r->layout, at);
    if (want_1d ? t->ndim != 1 : (t->ndim != 2 || t->dtype != INGOT_DTYPE_BF16))
        return NULL;
    *index = at;
    return t;
}

static void identity_partner(ingot_predict_partner *p, uint64_t rows, uint64_t offset) {
    *p = (ingot_predict_partner){.rows = rows, .row_offset = offset, .head_dim = 1, .group = 1};
}

static int rule_down(const rule_ctx *r, const ingot_layout_tensor *d, ingot_predict_geometry *g,
                     size_t *partner) {
    const ingot_layout_tensor *gate = find(r, ".mlp.gate_proj.weight", 0, &partner[0]);
    const ingot_layout_tensor *up = find(r, ".mlp.up_proj.weight", 0, &partner[1]);
    uint64_t hidden = d->shape[0], inner = d->shape[1];
    if (!gate || !up || gate->shape[0] != inner || gate->shape[1] != hidden ||
        up->shape[0] != inner || up->shape[1] != hidden)
        return 0;
    *g = (ingot_predict_geometry){.axis = INGOT_PREDICT_COLS, .units = inner,
                                  .unit_len = hidden, .partner_count = 2};
    identity_partner(&g->partner[0], inner, 0);
    identity_partner(&g->partner[1], inner, 0);
    return 1;
}

static int rule_up(const rule_ctx *r, const ingot_layout_tensor *d, ingot_predict_geometry *g,
                   size_t *partner) {
    const ingot_layout_tensor *gate = find(r, ".mlp.gate_proj.weight", 0, &partner[0]);
    uint64_t inner = d->shape[0], hidden = d->shape[1];
    if (!gate || gate->shape[0] != inner || gate->shape[1] != hidden)
        return 0;
    *g = (ingot_predict_geometry){.axis = INGOT_PREDICT_ROWS, .units = inner,
                                  .unit_len = hidden, .partner_count = 1};
    identity_partner(&g->partner[0], inner, 0);
    return 1;
}

static int rule_z(const rule_ctx *r, const ingot_layout_tensor *d, ingot_predict_geometry *g,
                  size_t *partner) {
    const ingot_layout_tensor *qkv = find(r, ".linear_attn.in_proj_qkv.weight", 0, &partner[0]);
    uint64_t v = d->shape[0], hidden = d->shape[1];
    if (!qkv || qkv->shape[1] != hidden || qkv->shape[0] < v)
        return 0;
    *g = (ingot_predict_geometry){.axis = INGOT_PREDICT_ROWS, .units = v, .unit_len = hidden,
                                  .partner_count = 1};
    identity_partner(&g->partner[0], qkv->shape[0], qkv->shape[0] - v);
    return 1;
}

static int rule_out(const rule_ctx *r, const ingot_layout_tensor *d, ingot_predict_geometry *g,
                    size_t *partner) {
    const ingot_layout_tensor *qkv = find(r, ".linear_attn.in_proj_qkv.weight", 0, &partner[0]);
    const ingot_layout_tensor *z = find(r, ".linear_attn.in_proj_z.weight", 0, &partner[1]);
    uint64_t hidden = d->shape[0], v = d->shape[1];
    if (!qkv || !z || qkv->shape[1] != hidden || qkv->shape[0] < v || z->shape[0] != v ||
        z->shape[1] != hidden)
        return 0;
    *g = (ingot_predict_geometry){.axis = INGOT_PREDICT_COLS, .units = v, .unit_len = hidden,
                                  .partner_count = 2};
    identity_partner(&g->partner[0], qkv->shape[0], qkv->shape[0] - v);
    identity_partner(&g->partner[1], v, 0);
    return 1;
}

static int rule_o(const rule_ctx *r, const ingot_layout_tensor *d, ingot_predict_geometry *g,
                  size_t *partner) {
    size_t norm_index = 0;
    const ingot_layout_tensor *v = find(r, ".self_attn.v_proj.weight", 0, &partner[0]);
    const ingot_layout_tensor *norm = find(r, ".self_attn.q_norm.weight", 1, &norm_index);
    uint64_t hidden = d->shape[0], q = d->shape[1];
    if (!v || !norm || !norm->shape[0] || v->shape[1] != hidden)
        return 0;
    uint64_t hd = norm->shape[0], kv = v->shape[0];
    if (q % hd || kv % hd || !kv || (q / hd) % (kv / hd))
        return 0;
    *g = (ingot_predict_geometry){.axis = INGOT_PREDICT_COLS, .units = q, .unit_len = hidden,
                                  .partner_count = 1};
    g->partner[0] = (ingot_predict_partner){.rows = kv, .head_dim = hd,
                                            .group = (q / hd) / (kv / hd)};
    return 1;
}

typedef int (*rule_fn)(const rule_ctx *, const ingot_layout_tensor *, ingot_predict_geometry *,
                       size_t *);

static const struct {
    const char *suffix;
    rule_fn fn;
} RULES[] = {
    {".mlp.down_proj.weight", rule_down},
    {".mlp.up_proj.weight", rule_up},
    {".linear_attn.in_proj_z.weight", rule_z},
    {".linear_attn.out_proj.weight", rule_out},
    {".self_attn.o_proj.weight", rule_o},
};

int ingot_predict_rule(const ingot_layout *l, size_t i, ingot_predict_geometry *g,
                       size_t partner[INGOT_PREDICT_MAX_PARTNERS]) {
    const ingot_layout_tensor *d = ingot_layout_tensor_at(l, i);
    if (!d || d->dtype != INGOT_DTYPE_BF16 || d->ndim != 2)
        return 0;
    size_t len = strlen(d->name);
    for (size_t k = 0; k < sizeof(RULES) / sizeof(RULES[0]); k++) {
        size_t sl = strlen(RULES[k].suffix);
        if (len <= sl || strcmp(d->name + len - sl, RULES[k].suffix))
            continue;
        if (len - sl >= MAX_NAME)
            return 0;
        rule_ctx r = {.layout = l};
        memcpy(r.prefix, d->name, len - sl);
        r.prefix[len - sl] = '\0';
        int hit = RULES[k].fn(&r, d, g, partner);
        return hit && !ingot_predict_geometry_check(g);
    }
    return 0;
}
