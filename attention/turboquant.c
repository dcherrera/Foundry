/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * TurboQuant KV cache compression. See turboquant.h and attention/spec.md.
 *
 * Codebook provenance
 * -------------------
 * The scalar quantizer levels are Lloyd-Max optimal for a Beta(d/2, d/2)
 * source on [0,1] (d = head_dim), which is the limiting per-coordinate law of
 * a randomized-Hadamard-rotated vector.  They are shipped as a static table in
 * tq_codebooks.h (generated once, offline, by the same Lloyd's algorithm that
 * lloyd_beta() implements below).  For a (head_dim, bits) pair not present in
 * the table, fa_tq_build_codebook() regenerates the levels at runtime via
 * lloyd_beta(): start from uniform centroids, then alternately set boundaries
 * to centroid midpoints and centroids to the Beta-conditional means
 * E[X | boundary_i < X < boundary_{i+1}] (numerical quadrature), ~converged in
 * a few hundred iterations.  Only ratios of the Beta pdf are used, so its
 * normalization constant cancels and never needs to be computed.
 */

#include "turboquant.h"
#include "tq_codebooks.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================= */
/* State                                                         */
/* ============================================================= */

struct fa_tq_state {
    int            head_dim;     /* logical dimension */
    int            rot_dim;      /* padded pow2 transform dimension */
    float          inv_sqrt;     /* 1 / sqrt(rot_dim) */
    float*         signs;        /* [rot_dim] random +-1 diagonal (D) */
    fa_tq_config   config;
    fa_tq_codebook key_cb;
    fa_tq_codebook value_cb;
};

struct fa_tq_cache {
    fa_tq_state* state;
    int          num_layers;
    int          num_heads;
    int          max_seq_len;
    int          current_len;
    size_t       key_bph;        /* key packed bytes per head */
    size_t       value_bph;      /* value packed bytes per head */
    uint8_t**    key_data;       /* [num_layers] packed key bytes */
    uint8_t**    value_data;     /* [num_layers] packed value bytes */
    float**      key_range;      /* [num_layers] 2 floats per (pos,head) */
    float**      value_range;
};

/* ============================================================= */
/* Small utilities                                               */
/* ============================================================= */

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* splitmix64: deterministic seed expansion for the sign diagonal. */
static uint64_t splitmix64(uint64_t* s) {
    uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* In-place Fast Walsh-Hadamard Transform, O(d log d), d a power of two. */
static void fwht(float* x, int d) {
    for (int len = 1; len < d; len <<= 1) {
        for (int i = 0; i < d; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
}

/* ============================================================= */
/* Lloyd-Max codebook for Beta(d/2, d/2)                         */
/* ============================================================= */

#define TQ_LLOYD_NGRID 40000
#define TQ_LLOYD_ITERS 300

static void lloyd_beta(int head_dim, int bits, float* boundaries, float* centroids) {
    int L = 1 << bits;
    double a = head_dim / 2.0;
    double cen[16];

    /* Precompute grid samples of x and (unnormalized) Beta weights. */
    static double gx[TQ_LLOYD_NGRID];
    static double gw[TQ_LLOYD_NGRID];
    double maxlog = -1e300;
    for (int k = 0; k < TQ_LLOYD_NGRID; k++) {
        double x = (k + 0.5) / TQ_LLOYD_NGRID;
        double lp = (a - 1.0) * log(x) + (a - 1.0) * log(1.0 - x);
        gx[k] = x;
        gw[k] = lp;
        if (lp > maxlog) maxlog = lp;
    }
    for (int k = 0; k < TQ_LLOYD_NGRID; k++) gw[k] = exp(gw[k] - maxlog);

    for (int j = 0; j < L; j++) cen[j] = (j + 0.5) / (double)L;

    double bnd[16];
    for (int it = 0; it < TQ_LLOYD_ITERS; it++) {
        for (int j = 0; j < L - 1; j++) bnd[j] = 0.5 * (cen[j] + cen[j + 1]);
        int k = 0;
        for (int j = 0; j < L; j++) {
            double lo = (j == 0) ? 0.0 : bnd[j - 1];
            double hi = (j == L - 1) ? 1.0 : bnd[j];
            double sw = 0.0, sxw = 0.0;
            while (k < TQ_LLOYD_NGRID && gx[k] < lo) k++;
            int k2 = k;
            while (k2 < TQ_LLOYD_NGRID && gx[k2] < hi) {
                sw += gw[k2];
                sxw += gx[k2] * gw[k2];
                k2++;
            }
            k = k2;
            cen[j] = (sw > 0.0) ? (sxw / sw) : 0.5 * (lo + hi);
        }
    }
    for (int j = 0; j < L - 1; j++) boundaries[j] = (float)(0.5 * (cen[j] + cen[j + 1]));
    for (int j = 0; j < L; j++) centroids[j] = (float)cen[j];
}

int fa_tq_build_codebook(fa_tq_codebook* cb, int head_dim, int bits) {
    if (!cb || head_dim < 2 || (bits != 2 && bits != 3 && bits != 4)) return -1;

    memset(cb, 0, sizeof(*cb));
    cb->bits = bits;
    cb->levels = 1 << bits;

    /* Prefer a shipped, offline-generated table entry. */
    for (size_t i = 0; i < FA_TQ_NUM_CODEBOOKS; i++) {
        const fa_tq_cb_entry* e = &FA_TQ_CODEBOOKS[i];
        if (e->head_dim == head_dim && e->bits == bits) {
            memcpy(cb->boundaries, e->boundaries, sizeof(cb->boundaries));
            memcpy(cb->centroids, e->centroids, sizeof(cb->centroids));
            return 0;
        }
    }

    /* Not shipped: regenerate at runtime. */
    lloyd_beta(head_dim, bits, cb->boundaries, cb->centroids);
    return 0;
}

/* ============================================================= */
/* Lifecycle                                                     */
/* ============================================================= */

fa_tq_config fa_tq_config_default(void) {
    fa_tq_config c;
    c.key_bits = 4;
    c.value_bits = 2;
    return c;
}

static int valid_bits(int b) { return b == 2 || b == 3 || b == 4; }

fa_tq_state* fa_tq_init(int head_dim, fa_tq_config config, uint64_t seed) {
    if (head_dim < 2) return NULL;
    if (!valid_bits(config.key_bits) || !valid_bits(config.value_bits)) return NULL;

    fa_tq_state* st = calloc(1, sizeof(fa_tq_state));
    if (!st) return NULL;

    st->head_dim = head_dim;
    st->rot_dim = next_pow2(head_dim);
    st->inv_sqrt = 1.0f / sqrtf((float)st->rot_dim);
    st->config = config;

    st->signs = malloc((size_t)st->rot_dim * sizeof(float));
    if (!st->signs) { free(st); return NULL; }

    uint64_t s = seed;
    for (int i = 0; i < st->rot_dim; i++) {
        st->signs[i] = (splitmix64(&s) & 1ULL) ? 1.0f : -1.0f;
    }

    /* Beta parameter follows the logical head_dim, not the padded rot_dim. */
    if (fa_tq_build_codebook(&st->key_cb, head_dim, config.key_bits) != 0 ||
        fa_tq_build_codebook(&st->value_cb, head_dim, config.value_bits) != 0) {
        free(st->signs);
        free(st);
        return NULL;
    }
    return st;
}

void fa_tq_free(fa_tq_state* state) {
    if (!state) return;
    free(state->signs);
    free(state);
}

int fa_tq_head_dim(const fa_tq_state* state) { return state ? state->head_dim : 0; }
int fa_tq_rot_dim(const fa_tq_state* state) { return state ? state->rot_dim : 0; }
const fa_tq_codebook* fa_tq_key_codebook(const fa_tq_state* state) { return state ? &state->key_cb : NULL; }
const fa_tq_codebook* fa_tq_value_codebook(const fa_tq_state* state) { return state ? &state->value_cb : NULL; }

static size_t bytes_per_head(int rot_dim, int bits) {
    int vpb = (bits == 2) ? 4 : 2;   /* values per byte: 2->4, 3->2, 4->2 */
    return (size_t)((rot_dim + vpb - 1) / vpb);
}

size_t fa_tq_key_bytes_per_head(const fa_tq_state* state) {
    return state ? bytes_per_head(state->rot_dim, state->config.key_bits) : 0;
}
size_t fa_tq_value_bytes_per_head(const fa_tq_state* state) {
    return state ? bytes_per_head(state->rot_dim, state->config.value_bits) : 0;
}

/* ============================================================= */
/* Rotation                                                      */
/* ============================================================= */

/* Forward:  y = H (D .* x) / sqrt(d).  Applying the random signs D *before*
 * the Walsh-Hadamard H is the standard SRHT ordering: it decorrelates the
 * input (in particular it destroys any DC/constant component, which a bare H
 * would otherwise pile into coordinate 0) before H spreads the energy evenly.
 * The spec writes "D * H * x", but signs-first is what actually realizes the
 * "spread information across all coordinates" property it describes. */
void fa_tq_rotate(const fa_tq_state* state, float* buf) {
    for (int i = 0; i < state->rot_dim; i++) buf[i] = state->signs[i] * buf[i];
    fwht(buf, state->rot_dim);
    for (int i = 0; i < state->rot_dim; i++) buf[i] *= state->inv_sqrt;
}

/* Inverse:  x = D .* (H y) / sqrt(d).  (H and D are each self-inverse up to
 * the 1/sqrt(d) scaling, applied in the opposite order.) */
void fa_tq_rotate_inverse(const fa_tq_state* state, float* buf) {
    fwht(buf, state->rot_dim);
    for (int i = 0; i < state->rot_dim; i++) {
        buf[i] = state->signs[i] * buf[i] * state->inv_sqrt;
    }
}

/* ============================================================= */
/* Scalar quantize / bit packing                                 */
/* ============================================================= */

/* Map a normalized value in [0,1] to a level index via the decision
 * boundaries.  Linear scan (<=15 comparisons) keeps it branch-simple. */
static int quantize_level(const fa_tq_codebook* cb, float norm) {
    int j = 0;
    while (j < cb->levels - 1 && norm >= cb->boundaries[j]) j++;
    return j;
}

static void pack_levels(const uint8_t* levels, int n, int bits, uint8_t* out) {
    int vpb = (bits == 2) ? 4 : 2;
    size_t nbytes = bytes_per_head(n, bits);
    memset(out, 0, nbytes);
    uint8_t mask = (uint8_t)((1 << bits) - 1);
    for (int i = 0; i < n; i++) {
        int bi = i / vpb;
        int slot = i % vpb;
        out[bi] |= (uint8_t)((levels[i] & mask) << (slot * bits));
    }
}

static void unpack_levels(const uint8_t* packed, int n, int bits, uint8_t* levels) {
    int vpb = (bits == 2) ? 4 : 2;
    uint8_t mask = (uint8_t)((1 << bits) - 1);
    for (int i = 0; i < n; i++) {
        int bi = i / vpb;
        int slot = i % vpb;
        levels[i] = (uint8_t)((packed[bi] >> (slot * bits)) & mask);
    }
}

/* ============================================================= */
/* Single-head quantize / dequantize                             */
/* ============================================================= */

void fa_tq_quantize_head(const fa_tq_state* state, const float* vec,
                         bool is_key, uint8_t* out_packed, float* out_range) {
    const fa_tq_codebook* cb = is_key ? &state->key_cb : &state->value_cb;
    int rd = state->rot_dim;
    int hd = state->head_dim;

    float work[512];              /* rot_dim <= 512 covers head_dim up to 512 */
    float* w = work;
    float* heap = NULL;
    if (rd > 512) { heap = malloc((size_t)rd * sizeof(float)); w = heap; }

    for (int i = 0; i < hd; i++) w[i] = vec[i];
    for (int i = hd; i < rd; i++) w[i] = 0.0f;

    fa_tq_rotate(state, w);

    /* Normalize by the L2 norm, not by min/max.  A coordinate of a rotated
     * (approximately uniformly random) direction lies in [-s, s] with
     * s = ||y||, and (y/s + 1)/2 follows Beta((d-1)/2, (d-1)/2) ~ Beta(d/2,
     * d/2), which is exactly what the codebook is optimized for.  We store the
     * range as [-s, +s] so the generic affine map below still lands in [0,1];
     * a future optimization could store the single scalar s instead. */
    double sq = 0.0;
    for (int i = 0; i < rd; i++) sq += (double)w[i] * w[i];
    float s = (float)sqrt(sq);
    out_range[0] = -s;
    out_range[1] = s;

    float mn = -s;
    float span = 2.0f * s;
    float inv_span = (span > 0.0f) ? 1.0f / span : 0.0f;

    uint8_t levels[512];
    uint8_t* lv = levels;
    uint8_t* lheap = NULL;
    if (rd > 512) { lheap = malloc((size_t)rd); lv = lheap; }

    for (int i = 0; i < rd; i++) {
        float norm = (span > 0.0f) ? (w[i] - mn) * inv_span : 0.5f;
        lv[i] = (uint8_t)quantize_level(cb, norm);
    }
    pack_levels(lv, rd, cb->bits, out_packed);

    free(heap);
    free(lheap);
}

void fa_tq_dequantize_head(const fa_tq_state* state, const uint8_t* packed,
                           const float* range, bool is_key, float* out_vec) {
    const fa_tq_codebook* cb = is_key ? &state->key_cb : &state->value_cb;
    int rd = state->rot_dim;
    int hd = state->head_dim;

    float work[512];
    float* w = work;
    float* heap = NULL;
    if (rd > 512) { heap = malloc((size_t)rd * sizeof(float)); w = heap; }

    uint8_t levels[512];
    uint8_t* lv = levels;
    uint8_t* lheap = NULL;
    if (rd > 512) { lheap = malloc((size_t)rd); lv = lheap; }

    unpack_levels(packed, rd, cb->bits, lv);

    float mn = range[0];
    float span = range[1] - range[0];
    for (int i = 0; i < rd; i++) {
        w[i] = mn + cb->centroids[lv[i]] * span;
    }

    fa_tq_rotate_inverse(state, w);

    for (int i = 0; i < hd; i++) out_vec[i] = w[i];

    free(heap);
    free(lheap);
}

/* ============================================================= */
/* Cache                                                         */
/* ============================================================= */

fa_tq_cache* fa_tq_cache_new(fa_tq_state* state, int num_layers,
                             int num_heads, int max_seq_len) {
    if (!state || num_layers <= 0 || num_heads <= 0 || max_seq_len <= 0) return NULL;

    fa_tq_cache* c = calloc(1, sizeof(fa_tq_cache));
    if (!c) return NULL;
    c->state = state;
    c->num_layers = num_layers;
    c->num_heads = num_heads;
    c->max_seq_len = max_seq_len;
    c->current_len = 0;
    c->key_bph = fa_tq_key_bytes_per_head(state);
    c->value_bph = fa_tq_value_bytes_per_head(state);

    c->key_data = calloc(num_layers, sizeof(uint8_t*));
    c->value_data = calloc(num_layers, sizeof(uint8_t*));
    c->key_range = calloc(num_layers, sizeof(float*));
    c->value_range = calloc(num_layers, sizeof(float*));
    if (!c->key_data || !c->value_data || !c->key_range || !c->value_range) {
        fa_tq_cache_free(c);
        return NULL;
    }

    size_t slots = (size_t)max_seq_len * num_heads;
    for (int l = 0; l < num_layers; l++) {
        c->key_data[l] = calloc(slots, c->key_bph);
        c->value_data[l] = calloc(slots, c->value_bph);
        c->key_range[l] = calloc(slots * 2, sizeof(float));
        c->value_range[l] = calloc(slots * 2, sizeof(float));
        if (!c->key_data[l] || !c->value_data[l] || !c->key_range[l] || !c->value_range[l]) {
            fa_tq_cache_free(c);
            return NULL;
        }
    }
    return c;
}

void fa_tq_cache_free(fa_tq_cache* cache) {
    if (!cache) return;
    for (int l = 0; l < cache->num_layers; l++) {
        if (cache->key_data) free(cache->key_data[l]);
        if (cache->value_data) free(cache->value_data[l]);
        if (cache->key_range) free(cache->key_range[l]);
        if (cache->value_range) free(cache->value_range[l]);
    }
    free(cache->key_data);
    free(cache->value_data);
    free(cache->key_range);
    free(cache->value_range);
    free(cache);
}

void fa_tq_cache_clear(fa_tq_cache* cache) {
    if (cache) cache->current_len = 0;
}

int fa_tq_cache_seq_len(const fa_tq_cache* cache) {
    return cache ? cache->current_len : 0;
}

size_t fa_tq_cache_bytes(const fa_tq_cache* cache) {
    if (!cache) return 0;
    size_t slots = (size_t)cache->max_seq_len * cache->num_heads;
    size_t per_layer = slots * cache->key_bph + slots * cache->value_bph
                     + slots * 2 * sizeof(float)   /* key ranges */
                     + slots * 2 * sizeof(float);  /* value ranges */
    return per_layer * (size_t)cache->num_layers;
}

void fa_tq_quantize(fa_tq_state* state, fa_tq_cache* cache,
                    int layer, int pos, const float* vec, bool is_key) {
    if (!state || !cache || layer < 0 || layer >= cache->num_layers) return;
    if (pos < 0 || pos >= cache->max_seq_len) return;

    int H = cache->num_heads;
    int hd = state->head_dim;
    size_t bph = is_key ? cache->key_bph : cache->value_bph;
    uint8_t* data = is_key ? cache->key_data[layer] : cache->value_data[layer];
    float* ranges = is_key ? cache->key_range[layer] : cache->value_range[layer];

    for (int h = 0; h < H; h++) {
        size_t slot = (size_t)pos * H + h;
        fa_tq_quantize_head(state, vec + (size_t)h * hd, is_key,
                            data + slot * bph, ranges + slot * 2);
    }

    if (pos + 1 > cache->current_len) cache->current_len = pos + 1;
}

float* fa_tq_dequantize(fa_tq_state* state, fa_tq_cache* cache,
                        int layer, int start, int count,
                        bool is_key, float* out) {
    if (!state || !cache || !out) return out;
    if (layer < 0 || layer >= cache->num_layers) return out;

    int H = cache->num_heads;
    int hd = state->head_dim;
    size_t bph = is_key ? cache->key_bph : cache->value_bph;
    const uint8_t* data = is_key ? cache->key_data[layer] : cache->value_data[layer];
    const float* ranges = is_key ? cache->key_range[layer] : cache->value_range[layer];

    for (int t = 0; t < count; t++) {
        int pos = start + t;
        for (int h = 0; h < H; h++) {
            size_t slot = (size_t)pos * H + h;
            float* dst = out + ((size_t)t * H + h) * hd;
            fa_tq_dequantize_head(state, data + slot * bph, ranges + slot * 2,
                                  is_key, dst);
        }
    }
    return out;
}
