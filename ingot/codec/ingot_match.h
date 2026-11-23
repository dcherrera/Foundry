/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_match.h
 * @brief  Match mode (format version 4): a BF16 tensor coded given up to 8 partner units per unit,
 *         chosen per unit from any earlier tensor (the previous layer, a sibling layer, earlier
 *         roles of the same layer, the tensor's own earlier rows, or a tensor of an earlier file).
 *         Mode 6's integer predictor and weight coder, generalised: p = sum_k q_k x_k.
 *
 * @status Active
 * @issues None
 * @todo   None
 *
 * Units. The target D holds U unit vectors of length H: unit u is row u (axis ROWS, D [U, H]) or
 * column u (axis COLS, D [H, U]). Source s is a BF16 tensor whose n_s units have length H: rows
 * of [n_s, H] or columns of [H, n_s]. A source may be D itself (rows only): unit u then uses rows
 * i < u, and the blocks decode in order (ingot_match_decoder_serial) unless they are R-row
 * strips whose self partners stay in their own strip (flag bit 1: strips; then blocks are
 * independent and each decodes its rows in order). Flag bit 2 (waves, with strips) lets a self
 * partner sit in any earlier row: the encoder keeps each strip's partners in strips of earlier
 * waves, so a device decodes the strips wave by wave (the CPU decoder runs them in order).
 *
 * Per unit u (side stream, one rANS block in the prefix): scale s_u (unit RMS ~ 2^(s_u / 4)), K_u
 * (0..8) and K_u partners (source, unit index, int16 coefficient q). Predictor, in integers:
 * X = ingot_predict_fixed(x, C - E_s) and p = sum_k q_k X_k in int64 (|p| < 2^58, so the sum is
 * exact in any order). Contexts and the weight coder are mode 6's (class of s_u x 12 buckets of
 * |p|; sign-agree bit, plain symbol, 5 raw bits).
 *
 * Payload prefix (after the frame, little-endian):
 *   u32 source_count NS, u32 axis, u64 U, u64 H, i32 C, u32 block_log2 (18), u64 block_count,
 *   u32 classes S, u32 class_width, i32 class_ref, u32 context_count (36 S), u64 param_hash,
 *   u32 flags (bit 0: self source, bit 1: strips, bit 2: waves),
 *   u32 strip rows R (0 without bit 1),
 *   u64 side_table_bytes, u64 side_block_bytes,
 *   NS x { u32 ref (0xFFFFFFFF: self), u32 axis, u64 units, i32 E, u32 0 },
 *   side tables, pad to 8, side block, pad to 8, weight tables, pad to 8.
 * Side stream, units in order: scale + 128 (ctx 0, alphabet 256); K (ctx 1, alphabet 9); per
 * partner: link = 2 src + same (ctx 2, alphabet 2 NS; same: index = u), the index as
 * ceil(log2 n_src) raw bits unless same, the coefficient as a step code (ctx 3 + src).
 * param_hash = ingot_hash64 of the decoded side values (ingot_match_param_hash).
 */
#ifndef FOUNDRY_INGOT_MATCH_H
#define FOUNDRY_INGOT_MATCH_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/codec/ingot_block.h"
#include "ingot/codec/ingot_fcoder.h"
#include "ingot/codec/ingot_predict.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    INGOT_MATCH_MAX_K = 8,
    INGOT_MATCH_MAX_SOURCES = 64,
    INGOT_MATCH_INDEX_BITS_MAX = 25, /* a source holds at most 2^25 units */
    INGOT_MATCH_COEF_MAX = 32767,
    INGOT_MATCH_FLAG_SELF = 1,
    INGOT_MATCH_FLAG_STRIPS = 2, /* blocks are R-row strips; self partners stay in their strip */
    INGOT_MATCH_FLAG_WAVES = 4,  /* with strips: self partners may sit in earlier strips */
};

/* Source ref of the self source (D itself). */
#define INGOT_MATCH_SELF_REF 0xFFFFFFFFu

/** One source tensor of a match payload. */
typedef struct {
    uint32_t ref;              /* directory index, entry_count + external ref, or SELF_REF */
    ingot_predict_axis axis;   /* units are rows ([n, H]) or columns ([H, n]) */
    uint64_t units;            /* n */
    int32_t coef_exp;          /* E: coefficients of this source are q / 2^E */
    int self;                  /* derived: ref is D itself */
} ingot_match_source;

/** Stored parameters of a match payload. Per-unit arrays have U (or U * 8) entries. */
typedef struct {
    ingot_predict_axis axis;
    uint64_t units;            /* U */
    uint64_t unit_len;         /* H */
    uint32_t source_count;
    ingot_match_source source[INGOT_MATCH_MAX_SOURCES];
    int32_t scale_c;           /* C */
    uint32_t classes;
    uint32_t class_width;
    int32_t class_ref;
    uint64_t strip_rows;       /* R: blocks of R rows of D, self partners in the unit's strip
                                  (rows i >= u - u % R); 0: blocks of 2^18 weights */
    int waves;                 /* strips decode in waves: self partners in any earlier row */
    int8_t *scale;             /* [U] */
    uint8_t *k;                /* [U] */
    uint8_t *link;             /* [U * 8]: source index of partner k */
    uint64_t *index;           /* [U * 8]: unit index in that source */
    int16_t *coef;             /* [U * 8] */
} ingot_match_params;

/**
 * Allocate the per-unit arrays of params (units set; zeroed). @return 0 or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_match_params_alloc(ingot_match_params *p);

/**
 * Free the per-unit arrays (the struct itself is the caller's).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_match_params_release(ingot_match_params *p);

/**
 * Check params: geometry, sources, every partner inside its source (self: index < u), K <= 8,
 * coefficient magnitudes. @return 0 or -INGOT_EINVAL
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_match_params_check(const ingot_match_params *p);

/**
 * ingot_hash64 of the side values as decoded: per unit u8 scale, u8 K, then per partner u8 link,
 * u64 index, i16 coefficient (little-endian).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_match_param_hash(const ingot_match_params *p);

/**
 * ingot_hash64 of a list of source refs (u32 little-endian each): a match entry's directory base
 * hash.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_match_ref_hash(const uint32_t *refs, uint32_t n);

/** Derived per-tensor state shared by encoder and decoder. Opaque. */
typedef struct ingot_match_model ingot_match_model;

/**
 * Thresholds, classes, fixed-point tables and the weight context model (params must outlive it).
 * @return 0, -INGOT_EINVAL or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_match_model_create(const ingot_match_params *p, ingot_match_model **out);

/**
 * Free (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_match_model_free(ingot_match_model *m);

/**
 * The weight field-coder model (36 S contexts).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const ingot_fc_model *ingot_match_fc_model(const ingot_match_model *m);

/**
 * Context bytes (bit 7: p < 0, bits 0..6: z) of D's weights [lo, hi) (flat row-major).
 * sources[s] are the source tensors' BF16 units, unit-major ([n][H]: the stored tensor for row
 * units, a transposed copy for column units); a self source is D itself (rows), as the encoder
 * has it or as far as the decoder has written it.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_match_contexts(const ingot_match_model *m, const uint16_t *const *sources,
                          uint64_t lo, uint64_t hi, uint8_t *ctx);

/**
 * Transpose a rows x cols BF16 matrix into out (cols x rows): column units to unit-major.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
void ingot_match_transpose(const uint16_t *in, uint64_t rows, uint64_t cols, uint16_t *out);

/**
 * Code the side stream (count, encode or decode params' per-unit arrays). The one code path
 * for both directions; decoding fills params and checks every value.
 *
 * Thread-safe: yes (caller-owned coder and params)   Reentrant: yes   Signal-safe: yes
 */
void ingot_match_side_code(ingot_fcoder *c, ingot_match_params *p);

/**
 * The side stream's field-coder model for a source count.
 * @return 0 or -INGOT_EINVAL
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_match_side_model(uint32_t source_count, ingot_fc_model *out);

/**
 * Bytes of the fixed prefix header and source records.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_match_header_bytes(uint32_t source_count);

/**
 * Serialize the prefix: header, sources, side tables and block, weight tables, pads.
 * out must hold ingot_match_prefix_bytes and be zeroed.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_match_prefix_bytes(uint32_t source_count, const ingot_fc_tables *side_tables,
                                  size_t side_block, const ingot_fc_tables *tables);
void ingot_match_prefix_write(const ingot_match_params *p, uint64_t block_count,
                              uint32_t contexts, const ingot_fc_tables *side_tables,
                              const ingot_block_buf *side_block, const ingot_fc_tables *tables,
                              uint8_t *out);

/** Block decoder for one match payload. Opaque. */
typedef struct ingot_match_decoder ingot_match_decoder;

/**
 * Parse the prefix (decoding the side stream) and check it against the entry's shape.
 *
 * @param prefix_bytes  Out: where the block area starts, relative to prefix
 * @return              0, -INGOT_EFORMAT, -INGOT_ECORRUPT (side stream or param_hash) or
 *                      -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_match_decoder_open(uint64_t rows, uint64_t cols, const uint8_t *prefix, uint64_t avail,
                             ingot_match_decoder **out, uint64_t *prefix_bytes);

/**
 * The parameters the payload declares (sources, refs, per-unit partners).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
const ingot_match_params *ingot_match_decoder_params(const ingot_match_decoder *d);

/**
 * Give source s's decoded values (rows x cols as stored, BF16; must outlive the block decodes).
 * The self source is not bound: blocks read D from the output buffer.
 * Column-unit sources are copied unit-major (one transposed copy, freed at close).
 * @return 0, -INGOT_EFORMAT (shape does not match the source record) or -INGOT_ENOMEM
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
int ingot_match_decoder_bind(ingot_match_decoder *d, uint32_t s, uint64_t rows, uint64_t cols,
                             const uint16_t *values);

/**
 * 1 when the blocks must be decoded in order into one buffer holding the whole tensor (a self
 * source), else 0.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
int ingot_match_decoder_serial(const ingot_match_decoder *d);

/**
 * Block count of a payload with these params, its largest block (weights), and the weight
 * range [lo, hi) of block b: 2^18-weight blocks, or R-row strips (strip_rows R).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_match_block_count(const ingot_match_params *p, uint64_t *max_weights);
void ingot_match_block_range(const ingot_match_params *p, size_t b, uint64_t *lo, uint64_t *hi);

/**
 * Block count and the weight range [lo, hi) of block b.
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
size_t ingot_match_decoder_blocks(const ingot_match_decoder *d);
void ingot_match_decoder_range(const ingot_match_decoder *d, size_t b, uint64_t *lo,
                               uint64_t *hi);

/**
 * Decode block b into out (hi - lo weights). Serial payloads: out must point at weight lo of a
 * buffer holding the whole tensor, with blocks 0..b-1 already decoded into it.
 * @return 0, -INGOT_EINVAL (a source not bound), -INGOT_ECORRUPT or -INGOT_ENOMEM
 *
 * Thread-safe: yes (distinct out buffers; serial payloads: one block at a time)
 * Reentrant: yes   Signal-safe: no
 */
int ingot_match_decoder_block(const ingot_match_decoder *d, size_t b, const ingot_block_view *v,
                              uint16_t *out);

/**
 * Close (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_match_decoder_close(ingot_match_decoder *d);

/* ---- Encoder (ingot_match_plan.c) ---- */

/** A source as the encoder sees it. */
typedef struct {
    uint32_t ref;
    ingot_predict_axis axis;
    uint64_t units;
    const uint16_t *data;      /* row-major as stored */
    const uint16_t *unit_major;/* [units][H]: data itself for rows, a transposed copy for cols */
    int self;
} ingot_match_input;

/** Candidate partners of every unit: [U][per_unit] (source, index) pairs; source 0xFF = none. */
typedef struct {
    uint32_t per_unit;
    uint8_t *source;
    uint64_t *index;
    uint32_t max_k;     /* partners kept per unit, 1..8 (0: 8; encoder policy, not format) */
    uint64_t strip_rows;  /* R-row strip blocks for the payload (0: 2^18-weight blocks) */
    int waves;            /* with strips: self partners may sit in earlier strips */
} ingot_match_candidates;

/** Histograms, tables and parameters of one target tensor. Opaque. */
typedef struct ingot_match_plan ingot_match_plan;

/**
 * Choose partners per unit from the candidates (exact re-rank, least squares, cost-based K),
 * quantize, code the side stream, count every symbol and build the tables.
 *
 * @param axis, units, unit_len  Target geometry
 * @param data       D's BF16 values (row-major); must outlive the plan
 * @param unit_major D's units [U][H] (data itself for rows); must outlive the plan
 * @param src        Sources (at most 64); data must outlive the plan
 * @param cand       Candidates
 * @return           0, -INGOT_EINVAL or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_match_plan_create(ingot_predict_axis axis, uint64_t units, uint64_t unit_len,
                            const uint16_t *data, const uint16_t *unit_major,
                            const ingot_match_input *src, uint32_t source_count,
                            const ingot_match_candidates *cand, int threads,
                            ingot_match_plan **out);

/**
 * Payload size after the frame (within a few bytes per block), prefix bytes, block count, and
 * the parameters chosen (which sources a unit uses).
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: yes
 */
uint64_t ingot_match_plan_bytes(const ingot_match_plan *plan);
uint64_t ingot_match_plan_prefix_bytes(const ingot_match_plan *plan);
size_t ingot_match_plan_blocks(const ingot_match_plan *plan);
const ingot_match_params *ingot_match_plan_params(const ingot_match_plan *plan);

/**
 * Write the payload (prefix + block area) to the sink (8-aligned position).
 * @return 0 or the first error
 *
 * Thread-safe: no (one sink)   Reentrant: yes   Signal-safe: no
 */
int ingot_match_encode(const ingot_match_plan *plan, int threads, ingot_sink *sink);

/**
 * Free (NULL is a no-op).
 *
 * Thread-safe: no   Reentrant: yes   Signal-safe: no
 */
void ingot_match_plan_free(ingot_match_plan *plan);

/* ---- Candidate search (ingot_match_search.c; encoder only, not part of the format) ---- */

#ifndef INGOT_MATCH_SKETCH_DIMS
#define INGOT_MATCH_SKETCH_DIMS 512 /* a multiple of 64; build-time knob for search experiments */
#endif
enum { INGOT_MATCH_SKETCH_DIM = INGOT_MATCH_SKETCH_DIMS };

/**
 * Sketch n unit vectors ([n][H] BF16): a fixed +-1 random projection of each unit-normalized
 * vector to INGOT_MATCH_SKETCH_DIM int16 values (cosine x 2^16 / DIM per product; |value| <= 2047;
 * zero vectors sketch to zero).
 * @return 0 or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_match_sketch(const uint16_t *unit_major, uint64_t n, uint64_t unit_len, int threads,
                       int16_t *out);

/** A search pool member: one source's sketches. */
typedef struct {
    const int16_t *sketch; /* [units][SKETCH_DIM] */
    uint64_t units;
    int self;              /* only units below the target unit qualify */
} ingot_match_pool;

/**
 * For every target unit, the per_unit pool units with the largest |sketch dot| (cand must have
 * per_unit set and arrays of U * per_unit entries; unfilled slots get source 0xFF).
 * @return 0 or -INGOT_ENOMEM
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ingot_match_search(const int16_t *target_sketch, uint64_t units,
                       const ingot_match_pool *pool, uint32_t pool_count, int threads,
                       ingot_match_candidates *cand);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_INGOT_MATCH_H */
