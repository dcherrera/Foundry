/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "hdc.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================= */
/* Internal helpers                                              */
/* ============================================================= */

/* Number of uint64 words needed for dim bits */
static inline int fv__words(int dim) {
    return (dim + 63) / 64;
}

/* Simple xorshift64 PRNG */
static inline uint64_t fv__xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

/* Box-Muller transform: generate one standard normal from PRNG state */
static double fv__randn(uint64_t *state) {
    double u1 = (double)(fv__xorshift64(state) >> 11) / (double)(1ULL << 53);
    double u2 = (double)(fv__xorshift64(state) >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ============================================================= */
/* Config                                                        */
/* ============================================================= */

fv_config fv_config_default(void) {
    fv_config c;
    c.dim = 4096;
    c.seed = 42;
    return c;
}

/* ============================================================= */
/* Vector Lifecycle                                              */
/* ============================================================= */

fv_vector* fv_alloc(int dim) {
    if (dim <= 0) return NULL;
    fv_vector *v = calloc(1, sizeof(fv_vector));
    if (!v) return NULL;
    int words = fv__words(dim);
    v->bits = calloc(words, sizeof(uint64_t));
    if (!v->bits) { free(v); return NULL; }
    v->dim = dim;
    return v;
}

fv_vector* fv_random(int dim, uint64_t seed) {
    if (dim <= 0) return NULL;
    fv_vector *v = fv_alloc(dim);
    if (!v) return NULL;
    uint64_t state = seed ? seed : 1;
    int words = fv__words(dim);
    for (int i = 0; i < words; i++) {
        v->bits[i] = fv__xorshift64(&state);
    }
    /* Mask unused bits in last word */
    int remainder = dim % 64;
    if (remainder > 0) {
        v->bits[words - 1] &= (1ULL << remainder) - 1;
    }
    return v;
}

fv_vector* fv_zeros(int dim) {
    return fv_alloc(dim); /* calloc already zeros */
}

fv_vector* fv_clone(const fv_vector *v) {
    if (!v) return NULL;
    fv_vector *c = fv_alloc(v->dim);
    if (!c) return NULL;
    int words = fv__words(v->dim);
    memcpy(c->bits, v->bits, words * sizeof(uint64_t));
    return c;
}

void fv_free(fv_vector *v) {
    if (!v) return;
    free(v->bits);
    free(v);
}

/* ============================================================= */
/* Core HDC Operations                                           */
/* ============================================================= */

fv_vector* fv_xor(const fv_vector *a, const fv_vector *b) {
    if (!a || !b || a->dim != b->dim) return NULL;
    fv_vector *r = fv_alloc(a->dim);
    if (!r) return NULL;
    int words = fv__words(a->dim);
    for (int i = 0; i < words; i++) {
        r->bits[i] = a->bits[i] ^ b->bits[i];
    }
    return r;
}

fv_vector* fv_permute(const fv_vector *v, int shift) {
    if (!v) return NULL;
    int dim = v->dim;
    /* Normalize shift to [0, dim) */
    shift = ((shift % dim) + dim) % dim;
    if (shift == 0) return fv_clone(v);

    fv_vector *r = fv_alloc(dim);
    if (!r) return NULL;

    /* Bit-by-bit cyclic left shift */
    for (int i = 0; i < dim; i++) {
        int src = (i + dim - shift) % dim;
        if (fv_get_bit(v, src)) {
            fv_set_bit(r, i, 1);
        }
    }
    return r;
}

fv_vector* fv_bundle(fv_vector **vectors, int count) {
    if (!vectors || count <= 0) return NULL;
    int dim = vectors[0]->dim;

    /* Count 1-bits per position */
    int *counts = calloc(dim, sizeof(int));
    if (!counts) return NULL;

    for (int v = 0; v < count; v++) {
        if (!vectors[v] || vectors[v]->dim != dim) {
            free(counts);
            return NULL;
        }
        for (int i = 0; i < dim; i++) {
            if (fv_get_bit(vectors[v], i)) {
                counts[i]++;
            }
        }
    }

    /* Majority vote: threshold at count/2 */
    fv_vector *r = fv_alloc(dim);
    if (!r) { free(counts); return NULL; }

    int threshold = count / 2;
    for (int i = 0; i < dim; i++) {
        if (counts[i] > threshold) {
            fv_set_bit(r, i, 1);
        } else if (counts[i] == threshold && count % 2 == 0) {
            /* Tie-break: random based on position */
            fv_set_bit(r, i, i % 2);
        }
    }

    free(counts);
    return r;
}

fv_vector* fv_bundle_weighted(fv_vector **vectors, float *weights, int count) {
    if (!vectors || !weights || count <= 0) return NULL;
    int dim = vectors[0]->dim;

    /* Weighted count per position */
    float *sums = calloc(dim, sizeof(float));
    if (!sums) return NULL;

    for (int v = 0; v < count; v++) {
        if (!vectors[v] || vectors[v]->dim != dim) {
            free(sums);
            return NULL;
        }
        for (int i = 0; i < dim; i++) {
            if (fv_get_bit(vectors[v], i)) {
                sums[i] += weights[v];
            } else {
                sums[i] -= weights[v];
            }
        }
    }

    /* Threshold at 0 */
    fv_vector *r = fv_alloc(dim);
    if (!r) { free(sums); return NULL; }

    for (int i = 0; i < dim; i++) {
        if (sums[i] > 0.0f) {
            fv_set_bit(r, i, 1);
        }
    }

    free(sums);
    return r;
}

int fv_hamming(const fv_vector *a, const fv_vector *b) {
    if (!a || !b || a->dim != b->dim) return -1;
    int words = fv__words(a->dim);
    int dist = 0;
    for (int i = 0; i < words; i++) {
        dist += __builtin_popcountll(a->bits[i] ^ b->bits[i]);
    }
    return dist;
}

float fv_similarity(const fv_vector *a, const fv_vector *b) {
    int dist = fv_hamming(a, b);
    if (dist < 0) return -1.0f;
    return 1.0f - (float)dist / (float)a->dim;
}

/* ============================================================= */
/* Bit Operations                                                */
/* ============================================================= */

int fv_get_bit(const fv_vector *v, int idx) {
    if (!v || idx < 0 || idx >= v->dim) return -1;
    int word = idx / 64;
    int bit = idx % 64;
    return (v->bits[word] >> bit) & 1;
}

void fv_set_bit(fv_vector *v, int idx, int val) {
    if (!v || idx < 0 || idx >= v->dim) return;
    int word = idx / 64;
    int bit = idx % 64;
    if (val) {
        v->bits[word] |= (1ULL << bit);
    } else {
        v->bits[word] &= ~(1ULL << bit);
    }
}

int fv_popcount(const fv_vector *v) {
    if (!v) return -1;
    int words = fv__words(v->dim);
    int count = 0;
    for (int i = 0; i < words; i++) {
        count += __builtin_popcountll(v->bits[i]);
    }
    return count;
}

/* ============================================================= */
/* Random Projection                                             */
/* ============================================================= */

struct fv_projector {
    int input_dim;
    int hdc_dim;
    float *matrix;   /* [hdc_dim * input_dim] row-major Gaussian entries */
    uint64_t seed;
};

fv_projector* fv_projector_new(int input_dim, int hdc_dim, uint64_t seed) {
    if (input_dim <= 0 || hdc_dim <= 0) return NULL;

    fv_projector *p = calloc(1, sizeof(fv_projector));
    if (!p) return NULL;

    p->input_dim = input_dim;
    p->hdc_dim = hdc_dim;
    p->seed = seed;

    size_t n = (size_t)hdc_dim * (size_t)input_dim;
    p->matrix = malloc(n * sizeof(float));
    if (!p->matrix) { free(p); return NULL; }

    /* Generate Gaussian random projection matrix from seed */
    uint64_t state = seed ? seed : 1;
    float scale = 1.0f / sqrtf((float)input_dim);
    for (size_t i = 0; i < n; i++) {
        p->matrix[i] = (float)fv__randn(&state) * scale;
    }

    return p;
}

void fv_projector_free(fv_projector *proj) {
    if (!proj) return;
    free(proj->matrix);
    free(proj);
}

fv_vector* fv_project(const fv_projector *proj, const tm_tensor *embedding) {
    if (!proj || !embedding) return NULL;
    if (embedding->ndim != 1 || embedding->shape[0] != proj->input_dim) return NULL;
    if (embedding->dtype != TM_F32) return NULL;

    const float *x = (const float *)embedding->data;
    fv_vector *v = fv_alloc(proj->hdc_dim);
    if (!v) return NULL;

    /* Compute sign(R @ x) */
    for (int row = 0; row < proj->hdc_dim; row++) {
        float dot = 0.0f;
        const float *r = proj->matrix + (size_t)row * proj->input_dim;
        for (int col = 0; col < proj->input_dim; col++) {
            dot += r[col] * x[col];
        }
        if (dot > 0.0f) {
            fv_set_bit(v, row, 1);
        }
    }

    return v;
}

void fv_project_accum(const fv_projector *proj, const float *x, float *accum) {
    if (!proj || !x || !accum) return;
    for (int row = 0; row < proj->hdc_dim; row++) {
        float dot = 0.0f;
        const float *r = proj->matrix + (size_t)row * proj->input_dim;
        for (int col = 0; col < proj->input_dim; col++) {
            dot += r[col] * x[col];
        }
        accum[row] += dot;
    }
}

fv_vector* fv_project_threshold(const fv_projector *proj, const float *accum) {
    if (!proj || !accum) return NULL;
    fv_vector *v = fv_alloc(proj->hdc_dim);
    if (!v) return NULL;
    for (int row = 0; row < proj->hdc_dim; row++) {
        if (accum[row] > 0.0f) fv_set_bit(v, row, 1);
    }
    return v;
}

int fv_projector_hdc_dim(const fv_projector *proj) {
    return proj ? proj->hdc_dim : 0;
}

/* ============================================================= */
/* Tensor Conversion                                             */
/* ============================================================= */

tm_tensor* fv_to_tensor(const fv_vector *v) {
    if (!v) return NULL;
    int64_t shape[1] = { v->dim };
    tm_tensor *t = tm_zeros(shape, 1, TM_F32);
    if (!t) return NULL;
    float *data = (float *)t->data;
    for (int i = 0; i < v->dim; i++) {
        data[i] = (float)fv_get_bit(v, i);
    }
    return t;
}

fv_vector* fv_from_tensor(const tm_tensor *t) {
    if (!t || t->ndim != 1 || t->dtype != TM_F32) return NULL;
    int dim = (int)t->shape[0];
    fv_vector *v = fv_alloc(dim);
    if (!v) return NULL;
    const float *data = (const float *)t->data;
    for (int i = 0; i < dim; i++) {
        if (data[i] > 0.0f) {
            fv_set_bit(v, i, 1);
        }
    }
    return v;
}

/* ============================================================= */
/* Serialization                                                 */
/* ============================================================= */

int fv_save(const fv_vector *v, const char *path) {
    if (!v || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Write dimension as int32 */
    int32_t dim32 = (int32_t)v->dim;
    if (fwrite(&dim32, sizeof(int32_t), 1, f) != 1) { fclose(f); return -1; }

    /* Write bit data */
    int words = fv__words(v->dim);
    if (fwrite(v->bits, sizeof(uint64_t), words, f) != (size_t)words) {
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

fv_vector* fv_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    int32_t dim32;
    if (fread(&dim32, sizeof(int32_t), 1, f) != 1) { fclose(f); return NULL; }
    if (dim32 <= 0) { fclose(f); return NULL; }

    fv_vector *v = fv_alloc(dim32);
    if (!v) { fclose(f); return NULL; }

    int words = fv__words(dim32);
    if (fread(v->bits, sizeof(uint64_t), words, f) != (size_t)words) {
        fv_free(v);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return v;
}

/* ============================================================= */
/* Batch Search                                                  */
/* ============================================================= */

int fv_search(fv_vector **corpus, int corpus_size,
              const fv_vector *query, int top_k,
              int *indices_out, float *scores_out) {
    if (!corpus || !query || !indices_out || !scores_out) return 0;
    if (corpus_size <= 0 || top_k <= 0) return 0;

    int k = top_k < corpus_size ? top_k : corpus_size;

    /* Compute all similarities */
    typedef struct { int idx; float score; } scored;
    scored *all = malloc(corpus_size * sizeof(scored));
    if (!all) return 0;

    for (int i = 0; i < corpus_size; i++) {
        all[i].idx = i;
        all[i].score = fv_similarity(query, corpus[i]);
    }

    /* Partial sort: find top-k by selection */
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < corpus_size; j++) {
            if (all[j].score > all[best].score) {
                best = j;
            }
        }
        if (best != i) {
            scored tmp = all[i];
            all[i] = all[best];
            all[best] = tmp;
        }
        indices_out[i] = all[i].idx;
        scores_out[i] = all[i].score;
    }

    free(all);
    return k;
}
