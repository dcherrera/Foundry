/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_predict.h
 * @brief  Predict mode (format version 3): a BF16 tensor coded given 1-2 partner tensors that
 *         hold the same units (an FFN neuron's gate and up rows predict its down column). Per unit
 *         u, p = a_u x1 + b_u x2 over the unit's partner vectors; each weight is coded in a
 *         context picked by |p| relative to the unit's scale.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Units. The dependent D holds U unit vectors of length H: unit u is row u (axis ROWS, D is
 * [U, H]) or column u (axis COLS, D is [H, U]). Partner j is a BF16 tensor [rows_j, H]; unit u's
 * partner vector is its row map_j(u) = off_j + ((u / hd_j) / g_j) hd_j + u % hd_j (grouped-query
 * heads; hd = g = 1 is the identity).
 *
 * Every number that selects a context is computed in integers (deterministic on every compiler
 * and platform): partner BF16 values become fixed point x << F_j (F_j = C - E_j), p = sum
 * a_q x in int64 (a_q = int16 coefficient at exponent E_j, so p is the prediction at 2^C), and
 * the bucket of |p| against thresholds t_k 2^(s_u / 4) 2^C built from integer tables.
 *
 * Coding per weight w in context z = class(s_u) * 12 + bucket: one binary symbol
 * sign(w) xor (p < 0) in context z; then the plain symbol (BF16 bits 14..5) in context (z, that
 * bit); then the low 5 mantissa bits raw. Blocks of 2^18 weights in D's row-major order.
 *
 * Payload prefix (after the frame, little-endian):
 *   u32 partner_count J, u32 axis (0 rows, 1 cols), u64 units U, u64 unit_len H,
 *   i32 C, u32 block_log2 (18), u64 block_count, u32 classes S (1..8), u32 class_width (1..16),
 *   i32 class_ref, u32 context_count (36 S), u64 param_hash,
 *   J x { u32 entry, u32 0, u64 row_offset, u64 head_dim, u64 group, i32 E, u32 0 },
 *   i8 scale[U], pad to 2, i16 coef[U][J], pad to 8, context_count tables, pad to 8.
 * param_hash = ingot_hash64 of scale[] and coef[] as stored (pad excluded), so a changed
 * coefficient is refused even where it would not change a context.
 */
#ifndef FOUNDRY_INGOT_PREDICT_H
#define FOUNDRY_INGOT_PREDICT_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_fcoder.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    INGOT_PREDICT_MAX_PARTNERS = 2,
    INGOT_PREDICT_BUCKETS = 12,      /* thresholds + 1 */
    INGOT_PREDICT_MAX_CLASSES = 8,
    INGOT_PREDICT_FIX_BITS = 40,     /* partner fixed point saturates at +-2^40 */
    INGOT_PREDICT_MAX_CLASS_WIDTH = 16,
    INGOT_PREDICT_EXP_LIMIT = 1024,  /* |C|, |E_j| */
};

/* Bound on units, unit length and map fields (keeps every row index far from overflow). */
#define INGOT_PREDICT_LIMIT ((uint64_t)1 << 40)

/* Bucket thresholds saturate here (above any |p| < 2^57). */
#define INGOT_PREDICT_THRESHOLD_MAX ((int64_t)1 << 62)

typedef enum { INGOT_PREDICT_ROWS = 0, INGOT_PREDICT_COLS = 1 } ingot_predict_axis;

/** Where the units' partner vectors sit in one partner tensor ([rows, unit_len], row-major). */
typedef struct {
    uint32_t entry;       /* directory index of the partner tensor */
    uint64_t rows;        /* its row count */
    uint64_t row_offset;
    uint64_t head_dim;    /* >= 1 */
    uint64_t group;       /* >= 1 */
} ingot_predict_partner;

/** Unit geometry of a dependent tensor. */
typedef struct {
    ingot_predict_axis axis;
    uint64_t units;       /* U */
    uint64_t unit_len;    /* H */
    uint32_t partner_count;
    ingot_predict_partner partner[INGOT_PREDICT_MAX_PARTNERS];
} ingot_predict_geometry;

/** Stored predictor parameters (the prefix fields). */
typedef struct {
    ingot_predict_geometry geom;
    int32_t scale_c;                                /* C */
    int32_t coef_exp[INGOT_PREDICT_MAX_PARTNERS];   /* E_j */
    uint32_t classes;                               /* S */
    uint32_t class_width;
    int32_t class_ref;
    int8_t *scale;                                  /* [U]: unit RMS ~ 2^(scale / 4) */
    int16_t *coef;                                  /* [U * J] */
} ingot_predict_params;

/**
 * Partner row of unit u.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_predict_map(const ingot_predict_partner *p, uint64_t u);

/**
 * Check a geometry: 1-2 partners, units and lengths > 0, every mapped row inside its partner.
 * @return 0 or -INGOT_EINVAL
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_predict_geometry_check(const ingot_predict_geometry *g);

/**
 * BF16 value as fixed point: value * 2^shift truncated toward zero, magnitude clamped to
 * 2^INGOT_PREDICT_FIX_BITS (any shift, any bit pattern; inf and NaN clamp; integers only).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int64_t ingot_predict_fixed(uint16_t bf16, int32_t shift);

/**
 * Bucket thresholds of one unit: floor(t_k r 2^(floor(scale / 4) + c)) with r = 2^((scale mod 4)
 * / 4) in Q16 (65536, 77936, 92682, 110218), for the 11 thresholds t = 1/16, 1/8, 1/4, 3/8,
 * 1/2, 3/4, 1, 5/4, 3/2, 2, 3; clamped to INGOT_PREDICT_THRESHOLD_MAX (integers only).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_predict_thresholds(int8_t scale, int32_t c, int64_t out[INGOT_PREDICT_BUCKETS - 1]);

/**
 * Scale class of a unit: clamp(floor((scale - ref) / width) + classes / 2, 0, classes - 1).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint32_t ingot_predict_class(int32_t scale, uint32_t classes, uint32_t width, int32_t ref);

/** Derived per-tensor state shared by encoder and decoder. Opaque. */
typedef struct ingot_predict_model ingot_predict_model;

/**
 * Derive thresholds, classes and the context model from params (which must outlive it).
 * @return 0, -INGOT_EINVAL (bad params) or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_predict_model_create(const ingot_predict_params *p, ingot_predict_model **out);

/**
 * Free (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_predict_model_free(ingot_predict_model *m);

/**
 * The field-coder model (context count and alphabets) of a predict tensor.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const ingot_fc_model *ingot_predict_fc_model(const ingot_predict_model *m);

/**
 * Context bytes of D's weights [lo, hi) (flat row-major index): bit 7 = p < 0, bits 0..6 the
 * context z. partners[j] are the partner tensors' BF16 values.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_predict_contexts(const ingot_predict_model *m, const uint16_t *const *partners,
                            uint64_t lo, uint64_t hi, uint8_t *ctx);

/**
 * Code n weights given their context bytes: src (count, encode) or dst (decode). The one code
 * path for both directions.
 *
 * Thread-safe: yes (caller-owned coder)   Reentrant: yes   Signal-safe: yes
 */
void ingot_predict_code(ingot_fcoder *c, const ingot_predict_model *m, const uint8_t *ctx,
                        const uint16_t *src, uint16_t *dst, size_t n);

/**
 * Bytes of the prefix up to the tables (header, scales, coefficients, pads).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_predict_params_bytes(const ingot_predict_geometry *g);

/**
 * Serialize the prefix: header, partners, scales, coefficients (params_bytes), then the tables
 * (ingot_fc_tables_size) and the pad to 8. out must be zeroed.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_predict_prefix_write(const ingot_predict_params *p, uint64_t block_count,
                                uint32_t contexts, const ingot_fc_tables *t, uint8_t *out);

/** Histograms, tables and parameters of one dependent tensor. Opaque. */
typedef struct ingot_predict_plan ingot_predict_plan;

/**
 * Fit the per-unit predictor, quantize it, count every symbol and build the tables.
 *
 * @param g         Geometry (partner entries = their directory indices)
 * @param data      D's BF16 values (U * H); must outlive the plan
 * @param partners  Partner tensors' BF16 values; must outlive the plan
 * @param threads   Worker threads
 * @param out       Out: plan, freed with ingot_predict_plan_free
 * @return          0, -INGOT_EINVAL (bad geometry) or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_predict_plan_create(const ingot_predict_geometry *g, const uint16_t *data,
                              const uint16_t *const *partners, int threads,
                              ingot_predict_plan **out);

/**
 * Payload size (after the frame) within a few bytes per block; prefix bytes; block count.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_predict_plan_bytes(const ingot_predict_plan *plan);
uint64_t ingot_predict_plan_prefix_bytes(const ingot_predict_plan *plan);
size_t ingot_predict_plan_blocks(const ingot_predict_plan *plan);

/**
 * Write the payload (prefix + block area) to the sink (8-aligned position).
 * @return 0 or the first error
 *
 * Thread-safe: no (one sink)   Reentrant: yes   Signal-safe: no
 */
int ingot_predict_encode(const ingot_predict_plan *plan, int threads, ingot_sink *sink);

/**
 * Free (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_predict_plan_free(ingot_predict_plan *plan);

/** Block decoder for one predict payload. Opaque. */
typedef struct ingot_predict_decoder ingot_predict_decoder;

/**
 * Parse the prefix and check it against the directory entry's shape (rows x cols BF16).
 *
 * @param prefix_bytes  Out: where the block area starts, relative to prefix
 * @return              0, -INGOT_EFORMAT, -INGOT_ECORRUPT (param_hash mismatch) or
 *                      -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_predict_decoder_open(uint64_t rows, uint64_t cols, const uint8_t *prefix,
                               uint64_t avail, ingot_predict_decoder **out,
                               uint64_t *prefix_bytes);

/**
 * The geometry (partner entries and shapes) the payload declares.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const ingot_predict_geometry *ingot_predict_decoder_geometry(const ingot_predict_decoder *d);

/**
 * Give partner j's decoded values (rows x cols BF16; must outlive the block decodes).
 * @return 0 or -INGOT_EFORMAT (shape does not match the declared geometry)
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: yes
 */
int ingot_predict_decoder_bind(ingot_predict_decoder *d, uint32_t j, uint64_t rows,
                               uint64_t cols, const uint16_t *values);

/**
 * Block count and the weight range [lo, hi) of block b.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_predict_decoder_blocks(const ingot_predict_decoder *d);
void ingot_predict_decoder_range(const ingot_predict_decoder *d, size_t b, uint64_t *lo,
                                 uint64_t *hi);

/**
 * Decode block b into out (hi - lo weights). Every partner must be bound.
 * @return 0, -INGOT_EINVAL (a partner not bound), -INGOT_ECORRUPT or -INGOT_ENOMEM
 *
 * Thread-safe: yes (distinct out buffers)   Reentrant: yes   Signal-safe: no
 */
int ingot_predict_decoder_block(const ingot_predict_decoder *d, size_t b,
                                const ingot_block_view *v, uint16_t *out);

/**
 * Close (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_predict_decoder_close(ingot_predict_decoder *d);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_PREDICT_H */
