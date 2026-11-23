/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_CONVERT_H
#define FOUNDRY_CONVERT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Model format converter.
 * Converts HuggingFace safetensors models to BootAI .btw/.btv binary format.
 *
 * Supports any Llama-family model:
 *   Llama 2/3, Qwen 2/2.5/3, Mistral, Phi-3, Gemma 2
 *
 * Usage:
 *   fc_model_config *cfg = fc_parse_config("config.json");
 *   fc_export_weights("model.safetensors", cfg, "model.btw");
 *   fc_export_vocab("tokenizer.json", NULL, "vocab.btv");
 *   free(cfg);
 */

/* ============================================================= */
/* Architecture Detection                                        */
/* ============================================================= */

typedef enum {
    FC_ARCH_LLAMA,    /* Llama 2/3, Mistral, CodeLlama */
    FC_ARCH_QWEN2,    /* Qwen 2/2.5 (QKV bias) */
    FC_ARCH_QWEN3,    /* Qwen 3 (QK-norm, no bias) */
    FC_ARCH_PHI3,     /* Phi-3/3.5 (QKV bias) */
    FC_ARCH_GEMMA2,   /* Gemma 2 (QK-norm) */
    FC_ARCH_UNKNOWN,
} fc_arch;

/* ============================================================= */
/* Model Configuration                                           */
/* ============================================================= */

typedef struct {
    fc_arch arch;
    int n_layers;
    int n_embd;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int ffn_dim;
    int vocab_size;
    int max_seq_len;
    float rope_theta;
    float rms_eps;
    int has_qk_norm;
    int has_qkv_bias;
    int tie_word_embeddings;
    char model_type[32];
} fc_model_config;

/* ============================================================= */
/* Config Parsing                                                */
/* ============================================================= */

/* Parse HuggingFace config.json and auto-detect architecture */
fc_model_config* fc_parse_config(const char *path);

/* Get architecture name string */
const char* fc_arch_name(fc_arch arch);

/* ============================================================= */
/* Weight Export (.btw)                                           */
/* ============================================================= */

/*
 * Export safetensors weights to .btw binary format.
 *
 * Reads BF16/F16 weights from safetensors, quantizes projections to INT8,
 * writes contiguous binary output matching bootai/model/loader.c format.
 *
 * Returns true on success.
 */
bool fc_export_weights(const char *safetensors_path,
                       fc_model_config *config,
                       const char *output_path);

/* ============================================================= */
/* Vocab Export (.btv)                                            */
/* ============================================================= */

/*
 * Export tokenizer to .btv binary format.
 *
 * tokenizer_path: path to tokenizer.json (HF format)
 * merges_path:    path to merges.txt (NULL to extract from tokenizer.json)
 * vocab_path:     path to vocab.json (NULL to extract from tokenizer.json)
 *
 * Returns true on success.
 */
bool fc_export_vocab(const char *tokenizer_path,
                     const char *merges_path,
                     const char *vocab_path,
                     const char *output_path);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_CONVERT_H */
