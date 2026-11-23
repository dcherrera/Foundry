/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_LORA_H
#define FOUNDRY_LORA_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "../tensor/tensor.h"
#include "../safetensors/safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Low-Rank Adaptation (LoRA) for efficient fine-tuning.
 * Replaces HuggingFace PEFT.
 *
 * For weight matrix W (d_out x d_in), LoRA adds:
 *   output = Wx + (alpha/r) * B @ A @ x
 * where A is (r x d_in) and B is (d_out x r)
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* LoRA layer configuration */
typedef struct {
    int rank;           /* LoRA rank (r) */
    float alpha;        /* Scaling factor */
    float dropout;      /* Dropout rate (0.0 = no dropout) */
    bool fan_in_fan_out;/* For Conv1D compatibility */
} fl_config;

/* Single LoRA layer */
typedef struct {
    char* name;         /* Layer name (e.g., "model.layers.0.self_attn.q_proj") */
    int d_in;           /* Input dimension */
    int d_out;          /* Output dimension */
    int rank;           /* LoRA rank */
    float scale;        /* alpha / rank */

    tm_tensor* lora_A;  /* Down projection: (rank x d_in) */
    tm_tensor* lora_B;  /* Up projection: (d_out x rank) */
} fl_layer;

/* Collection of LoRA layers (adapter) */
typedef struct fl_adapter fl_adapter;

/* ============================================================= */
/* LoRA Layer                                                    */
/* ============================================================= */

/* Create LoRA layer with random initialization */
fl_layer* fl_layer_new(const char* name, int d_in, int d_out, fl_config* config);

/* Create LoRA layer from existing A and B matrices */
fl_layer* fl_layer_from_tensors(const char* name, tm_tensor* A, tm_tensor* B, float alpha);

/* Free layer */
void fl_layer_free(fl_layer* layer);

/* Apply LoRA to input: returns BAx scaled by alpha/r */
tm_tensor* fl_layer_forward(fl_layer* layer, tm_tensor* x);

/* Merge LoRA into weight matrix W: W_new = W + scale * B @ A */
tm_tensor* fl_layer_merge(fl_layer* layer, tm_tensor* W);

/* Get merged weight without modifying original */
tm_tensor* fl_layer_get_merged(fl_layer* layer, tm_tensor* W);

/* ============================================================= */
/* LoRA Adapter                                                  */
/* ============================================================= */

/* Create empty adapter */
fl_adapter* fl_adapter_new(void);

/* Free adapter and all layers */
void fl_adapter_free(fl_adapter* adapter);

/* Add layer to adapter (adapter takes ownership) */
void fl_adapter_add(fl_adapter* adapter, fl_layer* layer);

/* Get layer by name */
fl_layer* fl_adapter_get(fl_adapter* adapter, const char* name);

/* Get number of layers */
int fl_adapter_size(fl_adapter* adapter);

/* Get layer by index */
fl_layer* fl_adapter_get_idx(fl_adapter* adapter, int idx);

/* Load adapter from safetensors file */
fl_adapter* fl_adapter_load(const char* path, fl_config* config);

/* Save adapter to safetensors file */
bool fl_adapter_save(fl_adapter* adapter, const char* path);

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Default config: rank=8, alpha=16, dropout=0 */
fl_config fl_config_default(void);

/* Get total number of trainable parameters */
int64_t fl_adapter_num_params(fl_adapter* adapter);

/* Print adapter summary */
void fl_adapter_print(fl_adapter* adapter);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_LORA_H */
