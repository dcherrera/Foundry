/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_QUANTIZATION_EXPORT_H
#define FOUNDRY_QUANTIZATION_EXPORT_H

#include "quantization.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Offline model quantization export.
 *
 * Loads a safetensors model, quantizes weights to INT8/INT4/ternary,
 * and saves a new safetensors file with quantization metadata.
 * Clean C API for calling from a GUI.
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

typedef struct fq_export_state fq_export_state;

typedef struct {
    fq_dtype target_dtype;        /* FQ_INT8, FQ_INT4, FQ_TERNARY */
    fq_granularity granularity;   /* PER_TENSOR, PER_CHANNEL, PER_GROUP */
    int group_size;               /* for per-group (default 128) */
    float ternary_threshold;      /* 0.0 = auto absmean, >0 = manual */

    bool skip_embeddings;         /* don't quantize embed layers */
    bool skip_norms;              /* don't quantize norm layers */
    bool skip_head;               /* don't quantize lm_head */
    bool skip_biases;             /* don't quantize biases */
    bool skip_1d;                 /* don't quantize 1D tensors */
    int min_elements;             /* skip tensors < this size (default 1024) */

    /* Progress callback for UI */
    void (*progress_cb)(int current, int total, const char* name, void* ud);
    void* userdata;
} fq_export_config;

/* ============================================================= */
/* API                                                           */
/* ============================================================= */

/* Get default config: INT8, per-tensor, skip norms/embeddings/biases/1D */
fq_export_config fq_export_config_defaults(void);

/* One-shot export: quantize entire model in one call */
bool fq_export_quantized(const char* input, const char* output, fq_export_config* cfg);

/* Step-by-step API for GUI (process one tensor at a time) */
fq_export_state* fq_export_begin(const char* input, fq_export_config* cfg);
int fq_export_tensor_count(fq_export_state* s);
bool fq_export_step(fq_export_state* s);        /* true = more remain */
float fq_export_progress(fq_export_state* s);    /* 0.0 - 1.0 */
const char* fq_export_current_tensor(fq_export_state* s);
bool fq_export_finish(fq_export_state* s, const char* output);
void fq_export_abort(fq_export_state* s);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_QUANTIZATION_EXPORT_H */
