/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "convert.h"
#include "../safetensors/safetensors.h"
#include "../quantization/quantization.h"
#include "../tokenizer/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================= */
/* Minimal JSON helpers (same pattern as safetensors.c)           */
/* ============================================================= */

static char* read_file_str(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(size + 1);
    size_t rd = fread(buf, 1, size, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

/* Extract integer value for a JSON key like "num_hidden_layers": 28 */
static int json_get_int(const char *json, const char *key, int default_val) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return default_val;
    pos += strlen(search);
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t' || *pos == '\n')) pos++;
    return (int)strtol(pos, NULL, 10);
}

/* Extract float value for a JSON key */
static float json_get_float(const char *json, const char *key, float default_val) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return default_val;
    pos += strlen(search);
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t' || *pos == '\n')) pos++;
    return strtof(pos, NULL);
}

/* Extract string value for a JSON key (caller must free) */
static char* json_get_string(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return NULL;
    pos += strlen(search);
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t' || *pos == '\n')) pos++;
    if (*pos != '"') return NULL;
    pos++;
    const char *end = pos;
    while (*end && *end != '"') {
        if (*end == '\\') end++;
        if (*end) end++;
    }
    size_t len = end - pos;
    char *result = malloc(len + 1);
    memcpy(result, pos, len);
    result[len] = '\0';
    return result;
}

/* Extract boolean value for a JSON key */
static int json_get_bool(const char *json, const char *key, int default_val) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return default_val;
    pos += strlen(search);
    while (*pos && (*pos == ' ' || *pos == ':' || *pos == '\t' || *pos == '\n')) pos++;
    if (*pos == 't') return 1;
    if (*pos == 'f') return 0;
    return default_val;
}

/* ============================================================= */
/* Architecture Detection                                        */
/* ============================================================= */

const char* fc_arch_name(fc_arch arch) {
    switch (arch) {
        case FC_ARCH_LLAMA:   return "Llama";
        case FC_ARCH_QWEN2:   return "Qwen2";
        case FC_ARCH_QWEN3:   return "Qwen3";
        case FC_ARCH_PHI3:    return "Phi-3";
        case FC_ARCH_GEMMA2:  return "Gemma2";
        case FC_ARCH_UNKNOWN: return "Unknown";
    }
    return "Unknown";
}

static fc_arch detect_arch(const char *model_type) {
    if (!model_type) return FC_ARCH_UNKNOWN;
    if (strstr(model_type, "qwen3")) return FC_ARCH_QWEN3;
    if (strstr(model_type, "qwen2") || strstr(model_type, "qwen")) return FC_ARCH_QWEN2;
    if (strstr(model_type, "llama") || strstr(model_type, "codellama")) return FC_ARCH_LLAMA;
    if (strstr(model_type, "mistral")) return FC_ARCH_LLAMA;
    if (strstr(model_type, "phi3") || strstr(model_type, "phi")) return FC_ARCH_PHI3;
    if (strstr(model_type, "gemma2") || strstr(model_type, "gemma")) return FC_ARCH_GEMMA2;
    return FC_ARCH_UNKNOWN;
}

/* ============================================================= */
/* Config Parsing                                                */
/* ============================================================= */

fc_model_config* fc_parse_config(const char *path) {
    char *json = read_file_str(path);
    if (!json) {
        fprintf(stderr, "ERROR: cannot read %s\n", path);
        return NULL;
    }

    fc_model_config *cfg = calloc(1, sizeof(fc_model_config));

    /* Extract model_type */
    char *mt = json_get_string(json, "model_type");
    if (mt) {
        snprintf(cfg->model_type, sizeof(cfg->model_type), "%s", mt);
        cfg->arch = detect_arch(mt);
        free(mt);
    } else {
        cfg->arch = FC_ARCH_UNKNOWN;
        strcpy(cfg->model_type, "unknown");
    }

    /* Core dimensions */
    cfg->n_layers    = json_get_int(json, "num_hidden_layers", 28);
    cfg->n_embd      = json_get_int(json, "hidden_size", 1024);
    cfg->n_heads     = json_get_int(json, "num_attention_heads", 16);
    cfg->n_kv_heads  = json_get_int(json, "num_key_value_heads", cfg->n_heads);
    cfg->ffn_dim     = json_get_int(json, "intermediate_size", 2816);
    cfg->vocab_size  = json_get_int(json, "vocab_size", 32000);
    cfg->max_seq_len = json_get_int(json, "max_position_embeddings", 4096);

    /* Float params */
    cfg->rope_theta  = json_get_float(json, "rope_theta", 10000.0f);
    cfg->rms_eps     = json_get_float(json, "rms_norm_eps", 1e-6f);

    /* Derived */
    cfg->head_dim = cfg->n_embd / cfg->n_heads;

    /* Optional features */
    cfg->tie_word_embeddings = json_get_bool(json, "tie_word_embeddings", 1);

    /* Architecture-specific defaults */
    switch (cfg->arch) {
        case FC_ARCH_QWEN3:
            cfg->has_qk_norm  = 1;
            cfg->has_qkv_bias = 0;
            break;
        case FC_ARCH_QWEN2:
        case FC_ARCH_PHI3:
            cfg->has_qk_norm  = 0;
            cfg->has_qkv_bias = 1;
            break;
        case FC_ARCH_GEMMA2:
            cfg->has_qk_norm  = 1;
            cfg->has_qkv_bias = 0;
            break;
        default:
            cfg->has_qk_norm  = 0;
            cfg->has_qkv_bias = 0;
            break;
    }

    free(json);
    return cfg;
}

/* ============================================================= */
/* Data Type Conversion                                          */
/* ============================================================= */

static void bf16_to_f32(const uint16_t *src, float *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &bits, 4);
    }
}

static void f16_to_f32(const uint16_t *src, float *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                /* Subnormal: normalize */
                exp = 1;
                while (!(mant & 0x400)) { mant <<= 1; exp--; }
                mant &= 0x3FF;
                bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mant << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000 | ((uint32_t)mant << 13);
        } else {
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | ((uint32_t)mant << 13);
        }
        memcpy(&dst[i], &bits, 4);
    }
}

static void bf16_to_f16(const uint16_t *src, uint16_t *dst, int64_t n) {
    /* BF16 → F32 → F16 (simple, not speed-critical) */
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        float f;
        memcpy(&f, &bits, 4);
        /* f32 to f16: truncate */
        uint32_t fbits;
        memcpy(&fbits, &f, 4);
        uint32_t sign = (fbits >> 16) & 0x8000;
        int32_t exp = ((fbits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (fbits >> 13) & 0x3FF;
        if (exp <= 0) {
            dst[i] = (uint16_t)sign;  /* flush to zero */
        } else if (exp >= 31) {
            dst[i] = (uint16_t)(sign | 0x7C00);  /* infinity */
        } else {
            dst[i] = (uint16_t)(sign | (exp << 10) | mant);
        }
    }
}

/*
 * Convert safetensors tensor data to f32.
 * Handles BF16, F16, F32 input types. Returns malloc'd f32 array.
 * Caller must free.
 */
static float* tensor_to_f32(const void *data, fs_dtype dtype, int64_t numel) {
    float *f32 = malloc(numel * sizeof(float));
    if (!f32) return NULL;

    switch (dtype) {
        case FS_DTYPE_BF16:
            bf16_to_f32((const uint16_t *)data, f32, numel);
            break;
        case FS_DTYPE_F16:
            f16_to_f32((const uint16_t *)data, f32, numel);
            break;
        case FS_DTYPE_F32:
            memcpy(f32, data, numel * sizeof(float));
            break;
        default:
            fprintf(stderr, "WARNING: unsupported dtype %d, zeroing\n", dtype);
            memset(f32, 0, numel * sizeof(float));
            break;
    }
    return f32;
}

/*
 * Convert safetensors tensor data to f16 (for embeddings).
 * Handles BF16, F16, F32 input. Returns malloc'd uint16_t array.
 * Caller must free.
 */
static uint16_t* tensor_to_f16(const void *data, fs_dtype dtype, int64_t numel) {
    uint16_t *f16 = malloc(numel * sizeof(uint16_t));
    if (!f16) return NULL;

    switch (dtype) {
        case FS_DTYPE_BF16:
            bf16_to_f16((const uint16_t *)data, f16, numel);
            break;
        case FS_DTYPE_F16:
            memcpy(f16, data, numel * sizeof(uint16_t));
            break;
        case FS_DTYPE_F32: {
            const float *f32 = (const float *)data;
            for (int64_t i = 0; i < numel; i++) {
                uint32_t fbits;
                memcpy(&fbits, &f32[i], 4);
                uint32_t sign = (fbits >> 16) & 0x8000;
                int32_t exp = ((fbits >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = (fbits >> 13) & 0x3FF;
                if (exp <= 0) f16[i] = (uint16_t)sign;
                else if (exp >= 31) f16[i] = (uint16_t)(sign | 0x7C00);
                else f16[i] = (uint16_t)(sign | (exp << 10) | mant);
            }
            break;
        }
        default:
            memset(f16, 0, numel * sizeof(uint16_t));
            break;
    }
    return f16;
}

/* ============================================================= */
/* .btw Header (must match bootai/model/loader.c)                */
/* ============================================================= */

#pragma pack(push, 1)
typedef struct {
    char     magic[4];       /* "BTAI" */
    uint32_t version;        /* 1 */
    uint32_t arch;           /* 0=qwen3, 1=qwen2, 2=llama, 3=rwkvx */
    uint32_t quant;          /* 0=f32, 1=f16, 2=q8, 3=q4 */
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t ffn_dim;
    uint32_t vocab_size;
    float    rope_theta;
    float    rms_eps;
    uint32_t max_seq_len;
    uint32_t has_qk_norm;
    uint32_t has_qkv_bias;
    uint8_t  reserved[64];
} btw_header;
#pragma pack(pop)

static uint32_t arch_to_btw(fc_arch arch) {
    switch (arch) {
        case FC_ARCH_QWEN3:  return 0;
        case FC_ARCH_QWEN2:  return 1;
        case FC_ARCH_LLAMA:  return 2;
        case FC_ARCH_PHI3:   return 1;  /* same layout as Qwen2 (has bias) */
        case FC_ARCH_GEMMA2: return 0;  /* same layout as Qwen3 (has qk-norm) */
        default:             return 2;  /* default to Llama */
    }
}

/* ============================================================= */
/* Weight Export                                                  */
/* ============================================================= */

/*
 * Fast per-row symmetric INT8 quantization for 2D matrices.
 * Much faster than fq_quantize_int8_per_channel which does O(channels × numel).
 * This does O(numel) — a single pass per row.
 */
static void quantize_per_row_q8(const float *f32, int8_t *q8, float *scales,
                                int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row = f32 + (int64_t)r * cols;
        float abs_max = 0.0f;
        for (int c = 0; c < cols; c++) {
            float a = fabsf(row[c]);
            if (a > abs_max) abs_max = a;
        }
        float scale = abs_max / 127.0f;
        scales[r] = scale;
        float inv_scale = (abs_max > 0.0f) ? 127.0f / abs_max : 0.0f;
        int8_t *qrow = q8 + (int64_t)r * cols;
        for (int c = 0; c < cols; c++) {
            int v = (int)roundf(row[c] * inv_scale);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qrow[c] = (int8_t)v;
        }
    }
}

/* Write a Q8 quantized weight matrix to file.
 * Reads raw data from safetensors, converts to f32, quantizes per-row, writes. */
static bool write_q8_weight(FILE *out, fs_file *sf, const char *name,
                            int rows, int cols, uint64_t *total) {
    fs_tensor_info *info = fs_get_info(sf, name);
    if (!info) {
        fprintf(stderr, "  WARNING: tensor '%s' not found, writing zeros\n", name);
        int64_t data_size = (int64_t)rows * cols;
        int8_t *zeros = calloc(data_size, 1);
        float *zero_scales = calloc(rows, sizeof(float));
        fwrite(zeros, 1, data_size, out);
        fwrite(zero_scales, sizeof(float), rows, out);
        free(zeros);
        free(zero_scales);
        *total += data_size + rows * sizeof(float);
        return true;
    }

    const void *raw = fs_get_data(sf, name);
    int64_t numel = (int64_t)rows * cols;
    float *f32 = tensor_to_f32(raw, info->dtype, numel);
    fs_free_info(info);

    if (!f32) {
        fprintf(stderr, "  ERROR: failed to convert '%s' to f32\n", name);
        return false;
    }

    /* Fast per-row quantization */
    int8_t *q8 = malloc(numel);
    float *scales = malloc(rows * sizeof(float));
    quantize_per_row_q8(f32, q8, scales, rows, cols);
    free(f32);

    fwrite(q8, 1, numel, out);
    fwrite(scales, sizeof(float), rows, out);
    *total += numel + rows * sizeof(float);

    free(q8);
    free(scales);
    return true;
}

/* Write an f32 weight to file. Converts from BF16/F16 if needed. */
static bool write_f32_weight(FILE *out, fs_file *sf, const char *name,
                             int64_t numel, uint64_t *total) {
    fs_tensor_info *info = fs_get_info(sf, name);
    if (!info) {
        fprintf(stderr, "  WARNING: tensor '%s' not found, writing zeros\n", name);
        float *zeros = calloc(numel, sizeof(float));
        fwrite(zeros, sizeof(float), numel, out);
        free(zeros);
        *total += numel * sizeof(float);
        return true;
    }

    const void *raw = fs_get_data(sf, name);
    float *f32 = tensor_to_f32(raw, info->dtype, numel);
    fs_free_info(info);

    fwrite(f32, sizeof(float), numel, out);
    *total += numel * sizeof(float);
    free(f32);
    return true;
}

/* Write an f16 embedding/lm_head weight. Converts from BF16 if needed. */
static bool write_f16_weight(FILE *out, fs_file *sf, const char *name,
                             int64_t numel, uint64_t *total) {
    fs_tensor_info *info = fs_get_info(sf, name);
    if (!info) {
        fprintf(stderr, "  WARNING: tensor '%s' not found\n", name);
        return false;
    }

    const void *raw = fs_get_data(sf, name);
    uint16_t *f16 = tensor_to_f16(raw, info->dtype, numel);
    fs_free_info(info);

    fwrite(f16, sizeof(uint16_t), numel, out);
    *total += numel * sizeof(uint16_t);
    free(f16);
    return true;
}

bool fc_export_weights(const char *safetensors_path,
                       fc_model_config *config,
                       const char *output_path) {
    fprintf(stderr, "Opening: %s\n", safetensors_path);
    fs_file *sf = fs_open(safetensors_path);
    if (!sf) {
        fprintf(stderr, "ERROR: cannot open %s\n", safetensors_path);
        return false;
    }

    int nt = fs_num_tensors(sf);
    fprintf(stderr, "  %d tensors found\n", nt);

    /* Detect naming prefix: "model.layers..." vs "layers..." */
    char prefix[64] = "";
    if (fs_has_tensor(sf, "model.embed_tokens.weight")) {
        strcpy(prefix, "model.");
    }

    FILE *out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "ERROR: cannot create %s\n", output_path);
        fs_close(sf);
        return false;
    }

    /* Write header */
    btw_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "BTAI", 4);
    hdr.version     = 1;
    hdr.arch        = arch_to_btw(config->arch);
    hdr.quant       = 1; /* F16 */
    hdr.n_layers    = config->n_layers;
    hdr.n_embd      = config->n_embd;
    hdr.n_heads     = config->n_heads;
    hdr.n_kv_heads  = config->n_kv_heads;
    hdr.head_dim    = config->head_dim;
    hdr.ffn_dim     = config->ffn_dim;
    hdr.vocab_size  = config->vocab_size;
    hdr.rope_theta  = config->rope_theta;
    hdr.rms_eps     = config->rms_eps;
    hdr.max_seq_len = config->max_seq_len;
    hdr.has_qk_norm = config->has_qk_norm;
    hdr.has_qkv_bias = config->has_qkv_bias;

    fwrite(&hdr, sizeof(hdr), 1, out);
    uint64_t total = sizeof(hdr);

    fprintf(stderr, "\nModel: %s (%s)\n", config->model_type, fc_arch_name(config->arch));
    fprintf(stderr, "  Layers: %d, Hidden: %d, Heads: %d/%d, FFN: %d\n",
            config->n_layers, config->n_embd, config->n_heads,
            config->n_kv_heads, config->ffn_dim);
    fprintf(stderr, "  QK-Norm: %d, QKV-Bias: %d, Tied: %d\n",
            config->has_qk_norm, config->has_qkv_bias,
            config->tie_word_embeddings);

    int q_dim  = config->n_heads * config->head_dim;
    int kv_dim = config->n_kv_heads * config->head_dim;

    /* 1. Embedding: [vocab, n_embd] as f16 */
    char tname[256];
    snprintf(tname, sizeof(tname), "%sembed_tokens.weight", prefix);
    int64_t embed_numel = (int64_t)config->vocab_size * config->n_embd;
    fprintf(stderr, "\nWriting embed_tokens [%d x %d] f16...\n",
            config->vocab_size, config->n_embd);
    write_f16_weight(out, sf, tname, embed_numel, &total);

    /* 2. Per-layer weights */
    for (int i = 0; i < config->n_layers; i++) {
        char lp[128];
        snprintf(lp, sizeof(lp), "%slayers.%d.", prefix, i);

        /* input_layernorm.weight: [n_embd] f32 */
        snprintf(tname, sizeof(tname), "%sinput_layernorm.weight", lp);
        write_f32_weight(out, sf, tname, config->n_embd, &total);

        /* Q projection: [q_dim, n_embd] F16 */
        snprintf(tname, sizeof(tname), "%sself_attn.q_proj.weight", lp);
        write_f16_weight(out, sf, tname, (int64_t)q_dim * config->n_embd, &total);

        /* K projection: [kv_dim, n_embd] F16 */
        snprintf(tname, sizeof(tname), "%sself_attn.k_proj.weight", lp);
        write_f16_weight(out, sf, tname, (int64_t)kv_dim * config->n_embd, &total);

        /* V projection: [kv_dim, n_embd] F16 */
        snprintf(tname, sizeof(tname), "%sself_attn.v_proj.weight", lp);
        write_f16_weight(out, sf, tname, (int64_t)kv_dim * config->n_embd, &total);

        /* O projection: [n_embd, q_dim] F16 */
        snprintf(tname, sizeof(tname), "%sself_attn.o_proj.weight", lp);
        write_f16_weight(out, sf, tname, (int64_t)config->n_embd * q_dim, &total);

        /* QK-Norm (if present) */
        if (config->has_qk_norm) {
            snprintf(tname, sizeof(tname), "%sself_attn.q_norm.weight", lp);
            write_f32_weight(out, sf, tname, config->head_dim, &total);
            snprintf(tname, sizeof(tname), "%sself_attn.k_norm.weight", lp);
            write_f32_weight(out, sf, tname, config->head_dim, &total);
        }

        /* QKV biases (if present) */
        if (config->has_qkv_bias) {
            snprintf(tname, sizeof(tname), "%sself_attn.q_proj.bias", lp);
            write_f32_weight(out, sf, tname, q_dim, &total);
            snprintf(tname, sizeof(tname), "%sself_attn.k_proj.bias", lp);
            write_f32_weight(out, sf, tname, kv_dim, &total);
            snprintf(tname, sizeof(tname), "%sself_attn.v_proj.bias", lp);
            write_f32_weight(out, sf, tname, kv_dim, &total);
        }

        /* post_attention_layernorm.weight: [n_embd] f32 */
        snprintf(tname, sizeof(tname), "%spost_attention_layernorm.weight", lp);
        write_f32_weight(out, sf, tname, config->n_embd, &total);

        /* gate_proj: [ffn_dim, n_embd] F16 */
        snprintf(tname, sizeof(tname), "%smlp.gate_proj.weight", lp);
        write_f16_weight(out, sf, tname, (int64_t)config->ffn_dim * config->n_embd, &total);

        /* up_proj: [ffn_dim, n_embd] F16 */
        snprintf(tname, sizeof(tname), "%smlp.up_proj.weight", lp);
        write_f16_weight(out, sf, tname, (int64_t)config->ffn_dim * config->n_embd, &total);

        /* down_proj: [n_embd, ffn_dim] F16 */
        snprintf(tname, sizeof(tname), "%smlp.down_proj.weight", lp);
        write_f16_weight(out, sf, tname, (int64_t)config->n_embd * config->ffn_dim, &total);

        if ((i + 1) % 4 == 0 || i == config->n_layers - 1) {
            fprintf(stderr, "  Layer %d/%d done (%llu MB)\n",
                    i + 1, config->n_layers, (unsigned long long)(total / 1024 / 1024));
        }
    }

    /* 3. Final norm: [n_embd] f32 */
    snprintf(tname, sizeof(tname), "%snorm.weight", prefix);
    write_f32_weight(out, sf, tname, config->n_embd, &total);

    /* 4. LM head: [vocab, n_embd] f16 (skip if tied with embed_tokens) */
    if (!config->tie_word_embeddings) {
        fprintf(stderr, "\nWriting lm_head [%d x %d] f16...\n",
                config->vocab_size, config->n_embd);
        write_f16_weight(out, sf, "lm_head.weight", embed_numel, &total);
    } else {
        fprintf(stderr, "\n  lm_head: tied with embed_tokens (shared)\n");
    }

    fclose(out);
    fs_close(sf);

    fprintf(stderr, "\nDone! Total: %.1f MB -> %s\n",
            (double)total / 1024.0 / 1024.0, output_path);
    return true;
}

/* ============================================================= */
/* .btv Vocab Export                                              */
/* ============================================================= */

#pragma pack(push, 1)
typedef struct {
    char     magic[4];    /* "BTAV" */
    uint32_t n_tokens;
    uint32_t n_merges;
    uint32_t n_special;
    uint8_t  reserved[16];
} btv_header;
#pragma pack(pop)

/*
 * Fast direct vocab.json parser.
 * Format: {"token_str": id, "another": id2, ...}
 * Handles JSON escapes: \n \t \r \\ \" \uXXXX
 * Returns malloc'd array of strings indexed by token ID.
 * Caller must free each string and the array.
 */
static char** parse_vocab_json(const char *path, int *n_tokens_out) {
    char *json = read_file_str(path);
    if (!json) return NULL;

    /* First pass: find max token ID to size the array */
    int max_id = -1;
    int count = 0;
    const char *p = json;
    while (*p) {
        /* Find next quoted key */
        while (*p && *p != '"') p++;
        if (!*p) break;
        p++; /* skip opening quote */
        /* Skip the key string */
        while (*p && *p != '"') {
            if (*p == '\\') p++;
            if (*p) p++;
        }
        if (!*p) break;
        p++; /* skip closing quote */
        /* Find colon and number */
        while (*p && *p != ':') p++;
        if (!*p) break;
        p++; /* skip colon */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        int id = (int)strtol(p, NULL, 10);
        if (id > max_id) max_id = id;
        count++;
        /* Skip past the number */
        while (*p && *p != ',' && *p != '}') p++;
    }

    if (max_id < 0) {
        free(json);
        *n_tokens_out = 0;
        return NULL;
    }

    int n_tokens = max_id + 1;
    char **tokens = calloc(n_tokens, sizeof(char *));

    /* Second pass: extract key strings and assign to their IDs */
    p = json;
    while (*p) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        p++; /* skip opening quote */

        /* Read key with escape handling */
        char key_buf[1024];
        int ki = 0;
        while (*p && *p != '"' && ki < (int)sizeof(key_buf) - 1) {
            if (*p == '\\') {
                p++;
                switch (*p) {
                    case 'n': key_buf[ki++] = '\n'; break;
                    case 't': key_buf[ki++] = '\t'; break;
                    case 'r': key_buf[ki++] = '\r'; break;
                    case '\\': key_buf[ki++] = '\\'; break;
                    case '"': key_buf[ki++] = '"'; break;
                    case '/': key_buf[ki++] = '/'; break;
                    case 'u': {
                        /* \uXXXX -> UTF-8 */
                        if (p[1] && p[2] && p[3] && p[4]) {
                            char hex[5] = {p[1], p[2], p[3], p[4], 0};
                            uint32_t cp = (uint32_t)strtoul(hex, NULL, 16);
                            p += 4;
                            if (cp < 0x80) {
                                key_buf[ki++] = (char)cp;
                            } else if (cp < 0x800) {
                                key_buf[ki++] = (char)(0xC0 | (cp >> 6));
                                key_buf[ki++] = (char)(0x80 | (cp & 0x3F));
                            } else {
                                key_buf[ki++] = (char)(0xE0 | (cp >> 12));
                                key_buf[ki++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                                key_buf[ki++] = (char)(0x80 | (cp & 0x3F));
                            }
                        }
                        break;
                    }
                    default: key_buf[ki++] = *p; break;
                }
                if (*p) p++;
            } else {
                key_buf[ki++] = *p++;
            }
        }
        key_buf[ki] = '\0';
        if (!*p) break;
        p++; /* skip closing quote */

        /* Find colon and number */
        while (*p && *p != ':') p++;
        if (!*p) break;
        p++;
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
        int id = (int)strtol(p, NULL, 10);

        if (id >= 0 && id < n_tokens) {
            tokens[id] = strdup(key_buf);
        }

        while (*p && *p != ',' && *p != '}') p++;
    }

    free(json);
    *n_tokens_out = n_tokens;
    return tokens;
}

/*
 * Fast merges.txt parser.
 * Format: optional "#version: 0.2\n" header, then "token_a token_b\n" per line.
 * Returns two malloc'd arrays of strings (a[] and b[]) and count.
 */
static void parse_merges_txt(const char *path, char ***a_out, char ***b_out,
                             int *n_merges_out) {
    char *data = read_file_str(path);
    if (!data) {
        *a_out = NULL;
        *b_out = NULL;
        *n_merges_out = 0;
        return;
    }

    /* Count lines (excluding header and empty lines) */
    int n_lines = 0;
    const char *p = data;
    while (*p) {
        while (*p == '\n' || *p == '\r') p++;
        if (!*p) break;
        if (*p != '#') n_lines++;
        while (*p && *p != '\n') p++;
    }

    char **a = malloc(n_lines * sizeof(char *));
    char **b = malloc(n_lines * sizeof(char *));
    int count = 0;

    p = data;
    while (*p) {
        /* Skip whitespace/newlines */
        while (*p == '\n' || *p == '\r') p++;
        if (!*p) break;

        /* Skip comment lines */
        if (*p == '#') {
            while (*p && *p != '\n') p++;
            continue;
        }

        /* Read token_a (up to space) */
        const char *start = p;
        while (*p && *p != ' ' && *p != '\n' && *p != '\r') p++;
        size_t a_len = p - start;

        if (*p != ' ') {
            /* Malformed line, skip */
            while (*p && *p != '\n') p++;
            continue;
        }
        p++; /* skip space */

        /* Read token_b (up to newline) */
        const char *b_start = p;
        while (*p && *p != '\n' && *p != '\r') p++;
        size_t b_len = p - b_start;

        /* Trim trailing spaces from b */
        while (b_len > 0 && (b_start[b_len - 1] == ' ' || b_start[b_len - 1] == '\r'))
            b_len--;

        a[count] = strndup(start, a_len);
        b[count] = strndup(b_start, b_len);
        count++;
    }

    free(data);
    *a_out = a;
    *b_out = b;
    *n_merges_out = count;
}

/*
 * Simple hash map: string → int.
 * Uses FNV-1a hash with open addressing (linear probing).
 * For 151K entries, 256K buckets gives ~60% load factor.
 */
#define VOCAB_MAP_CAPACITY (1 << 18) /* 262144 buckets */
#define VOCAB_MAP_MASK     (VOCAB_MAP_CAPACITY - 1)

typedef struct {
    const char *key;
    int value;
} vocab_map_entry;

typedef struct {
    vocab_map_entry *buckets;
} vocab_map;

static uint32_t fnv1a(const char *str) {
    uint32_t h = 2166136261u;
    while (*str) {
        h ^= (uint8_t)*str++;
        h *= 16777619u;
    }
    return h;
}

static vocab_map* vocab_map_create(void) {
    vocab_map *m = calloc(1, sizeof(vocab_map));
    m->buckets = calloc(VOCAB_MAP_CAPACITY, sizeof(vocab_map_entry));
    return m;
}

static void vocab_map_put(vocab_map *m, const char *key, int value) {
    uint32_t idx = fnv1a(key) & VOCAB_MAP_MASK;
    while (m->buckets[idx].key) {
        if (strcmp(m->buckets[idx].key, key) == 0) {
            m->buckets[idx].value = value;
            return;
        }
        idx = (idx + 1) & VOCAB_MAP_MASK;
    }
    m->buckets[idx].key = key;
    m->buckets[idx].value = value;
}

static int vocab_map_get(vocab_map *m, const char *key) {
    uint32_t idx = fnv1a(key) & VOCAB_MAP_MASK;
    while (m->buckets[idx].key) {
        if (strcmp(m->buckets[idx].key, key) == 0)
            return m->buckets[idx].value;
        idx = (idx + 1) & VOCAB_MAP_MASK;
    }
    return -1;
}

static void vocab_map_free(vocab_map *m) {
    if (m) { free(m->buckets); free(m); }
}

/*
 * GPT-2 bytes_to_unicode() mapping.
 * Maps each byte 0-255 to a unicode codepoint.
 * Printable ASCII and Latin-1 supplement map to themselves.
 * Control bytes and gaps map to U+0100+.
 */
static void build_gpt2_byte_map(uint32_t *byte_to_unicode, int *unicode_to_byte) {
    /* Initialize reverse map to -1 */
    for (int i = 0; i < 65536; i++) unicode_to_byte[i] = -1;

    /* Ranges that map to themselves: !-~ (0x21-0x7E), ¡-¬ (0xA1-0xAC), ®-ÿ (0xAE-0xFF) */
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 0x21 && b <= 0x7E) ||
            (b >= 0xA1 && b <= 0xAC) ||
            (b >= 0xAE && b <= 0xFF)) {
            byte_to_unicode[b] = (uint32_t)b;
            unicode_to_byte[b] = b;
        } else {
            byte_to_unicode[b] = 256 + n;
            unicode_to_byte[256 + n] = b;
            n++;
        }
    }
}

/*
 * Convert a GPT-2 unicode token string to raw bytes.
 * E.g., "Ġhello" (U+0120 + hello) → " hello" (space + hello)
 * Returns malloc'd string. Caller must free.
 */
static char* gpt2_unicode_to_bytes(const char *utf8_str, const int *unicode_to_byte, int *out_len) {
    if (!utf8_str) { *out_len = 0; return NULL; }

    /* Decode UTF-8 to codepoints, map each through unicode_to_byte */
    size_t slen = strlen(utf8_str);
    char *result = malloc(slen + 1); /* raw bytes can't be longer than UTF-8 */
    int ri = 0;

    const uint8_t *p = (const uint8_t *)utf8_str;
    while (*p) {
        uint32_t cp;
        if (*p < 0x80) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6;
            p++;
            if (*p) { cp |= (*p & 0x3F); p++; }
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12;
            p++;
            if (*p) { cp |= (*p & 0x3F) << 6; p++; }
            if (*p) { cp |= (*p & 0x3F); p++; }
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p & 0x07) << 18;
            p++;
            if (*p) { cp |= (*p & 0x3F) << 12; p++; }
            if (*p) { cp |= (*p & 0x3F) << 6; p++; }
            if (*p) { cp |= (*p & 0x3F); p++; }
        } else {
            p++; continue; /* invalid UTF-8, skip */
        }

        /* Map unicode codepoint to raw byte */
        if (cp < 65536 && unicode_to_byte[cp] >= 0) {
            result[ri++] = (char)(uint8_t)unicode_to_byte[cp];
        } else {
            /* Not a GPT-2 byte character — keep as UTF-8 (for multi-codepoint tokens) */
            /* Re-encode codepoint to UTF-8 */
            if (cp < 0x80) {
                result[ri++] = (char)cp;
            } else if (cp < 0x800) {
                result[ri++] = (char)(0xC0 | (cp >> 6));
                result[ri++] = (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                result[ri++] = (char)(0xE0 | (cp >> 12));
                result[ri++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                result[ri++] = (char)(0x80 | (cp & 0x3F));
            }
        }
    }
    *out_len = ri;
    return result;
}

bool fc_export_vocab(const char *tokenizer_path,
                     const char *merges_path,
                     const char *vocab_path,
                     const char *output_path) {
    /*
     * Fast path: parse vocab.json + merges.txt directly.
     * Bypasses ft_load_bpe() which is too slow for 151K+ token vocabs.
     */
    if (vocab_path && merges_path) {
        fprintf(stderr, "Loading vocab: %s + %s\n", vocab_path, merges_path);

        /* Parse vocab.json directly */
        int n_tokens = 0;
        char **tokens = parse_vocab_json(vocab_path, &n_tokens);
        if (!tokens || n_tokens == 0) {
            fprintf(stderr, "ERROR: failed to parse vocab.json\n");
            return false;
        }

        /* Parse merges.txt directly */
        char **merge_a_str = NULL, **merge_b_str = NULL;
        int n_merges = 0;
        parse_merges_txt(merges_path, &merge_a_str, &merge_b_str, &n_merges);

        fprintf(stderr, "  Vocab: %d tokens, %d merges\n", n_tokens, n_merges);

        /* Build GPT-2 byte mapping tables */
        uint32_t byte_to_unicode[256];
        int *unicode_to_byte = calloc(65536, sizeof(int));
        build_gpt2_byte_map(byte_to_unicode, unicode_to_byte);

        /* Convert all token strings from GPT-2 unicode to raw bytes.
         * This makes the string blob contain actual bytes (e.g., space instead of Ġ)
         * so decode works without reverse mapping. */
        char **raw_tokens = malloc(n_tokens * sizeof(char *));
        int *raw_lengths = malloc(n_tokens * sizeof(int));
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i]) {
                raw_tokens[i] = gpt2_unicode_to_bytes(tokens[i], unicode_to_byte, &raw_lengths[i]);
            } else {
                raw_tokens[i] = NULL;
                raw_lengths[i] = 0;
            }
        }

        /* Build string blob from raw (decoded) token bytes */
        size_t blob_size = 0;
        for (int i = 0; i < n_tokens; i++) blob_size += raw_lengths[i];

        uint32_t *offsets = malloc(n_tokens * sizeof(uint32_t));
        uint16_t *lengths = malloc(n_tokens * sizeof(uint16_t));
        char *blob = malloc(blob_size + 1);
        size_t blob_pos = 0;

        for (int i = 0; i < n_tokens; i++) {
            offsets[i] = (uint32_t)blob_pos;
            lengths[i] = (uint16_t)raw_lengths[i];
            if (raw_tokens[i] && raw_lengths[i] > 0) {
                memcpy(blob + blob_pos, raw_tokens[i], raw_lengths[i]);
                blob_pos += raw_lengths[i];
            }
        }

        /* Build hash map: ORIGINAL token string → ID for merge lookup
         * (merges.txt uses GPT-2 unicode strings, not raw bytes) */
        vocab_map *vmap = vocab_map_create();
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i]) vocab_map_put(vmap, tokens[i], i);
        }

        /* Resolve merge token IDs via hash map */
        uint32_t *merge_a_ids = NULL;
        uint32_t *merge_b_ids = NULL;
        if (n_merges > 0) {
            merge_a_ids = malloc(n_merges * sizeof(uint32_t));
            merge_b_ids = malloc(n_merges * sizeof(uint32_t));
            for (int i = 0; i < n_merges; i++) {
                int a_id = merge_a_str[i] ? vocab_map_get(vmap, merge_a_str[i]) : -1;
                int b_id = merge_b_str[i] ? vocab_map_get(vmap, merge_b_str[i]) : -1;
                merge_a_ids[i] = (a_id >= 0) ? (uint32_t)a_id : 0;
                merge_b_ids[i] = (b_id >= 0) ? (uint32_t)b_id : 0;
            }
        }

        /* Collect special tokens */
        typedef struct { uint32_t id; const char *name; } special_entry;
        special_entry specials[8];
        int n_special = 0;

        const char *special_names[] = {
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<s>", "</s>", "<|begin_of_text|>", "<|end_of_text|>",
            NULL
        };
        for (int i = 0; special_names[i] && n_special < 8; i++) {
            int id = vocab_map_get(vmap, special_names[i]);
            if (id >= 0) {
                specials[n_special].id = (uint32_t)id;
                specials[n_special].name = special_names[i];
                n_special++;
            }
        }

        fprintf(stderr, "  Special tokens: %d\n", n_special);

        /* Write .btv file */
        FILE *out = fopen(output_path, "wb");
        if (!out) {
            fprintf(stderr, "ERROR: cannot create %s\n", output_path);
            goto cleanup_direct;
        }

        btv_header hdr;
        memset(&hdr, 0, sizeof(hdr));
        memcpy(hdr.magic, "BTAV", 4);
        hdr.n_tokens = n_tokens;
        hdr.n_merges = n_merges;
        hdr.n_special = n_special;

        fwrite(&hdr, sizeof(hdr), 1, out);
        fwrite(offsets, sizeof(uint32_t), n_tokens, out);
        fwrite(lengths, sizeof(uint16_t), n_tokens, out);
        fwrite(blob, 1, blob_pos, out);

        /* Pre-compute merge result IDs: for each merge (a,b), find the token
         * whose string == concat(a_str, b_str). This lets bpe.c skip O(n²) search. */
        uint32_t *merge_result_ids = NULL;
        if (n_merges > 0) {
            merge_result_ids = malloc(n_merges * sizeof(uint32_t));
            int resolved = 0;
            for (int i = 0; i < n_merges; i++) {
                /* Concatenate a + b merge strings */
                const char *as = merge_a_str[i] ? merge_a_str[i] : "";
                const char *bs = merge_b_str[i] ? merge_b_str[i] : "";
                size_t alen = strlen(as), blen = strlen(bs);
                char *merged = malloc(alen + blen + 1);
                memcpy(merged, as, alen);
                memcpy(merged + alen, bs, blen);
                merged[alen + blen] = '\0';
                int result_id = vocab_map_get(vmap, merged);
                merge_result_ids[i] = (result_id >= 0) ? (uint32_t)result_id : 0;
                if (result_id >= 0) resolved++;
                free(merged);
            }
            fprintf(stderr, "  Merge results resolved: %d/%d\n", resolved, n_merges);
        }

        /* Write merge IDs as two flat uint32 arrays + result IDs */
        if (n_merges > 0) {
            fwrite(merge_a_ids, sizeof(uint32_t), n_merges, out);
            fwrite(merge_b_ids, sizeof(uint32_t), n_merges, out);
            fwrite(merge_result_ids, sizeof(uint32_t), n_merges, out);
        }

        for (int i = 0; i < n_special; i++) {
            uint32_t id = specials[i].id;
            const char *name = specials[i].name;
            uint16_t name_len = (uint16_t)strlen(name);
            fwrite(&id, sizeof(uint32_t), 1, out);
            fwrite(&name_len, sizeof(uint16_t), 1, out);
            fwrite(name, 1, name_len, out);
        }

        /* Write byte_to_id[256]: maps raw byte → token ID using GPT-2 mapping.
         * For each byte, find its GPT-2 unicode codepoint, look up the single-char
         * token in vocab. bpe.c uses this for the initial byte→token step. */
        uint32_t byte_to_id[256];
        int byte_mapped = 0;
        for (int b = 0; b < 256; b++) {
            uint32_t cp = byte_to_unicode[b];
            /* Encode codepoint to UTF-8 string for vocab lookup */
            char utf8[5] = {0};
            if (cp < 0x80) {
                utf8[0] = (char)cp;
            } else if (cp < 0x800) {
                utf8[0] = (char)(0xC0 | (cp >> 6));
                utf8[1] = (char)(0x80 | (cp & 0x3F));
            } else {
                utf8[0] = (char)(0xE0 | (cp >> 12));
                utf8[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                utf8[2] = (char)(0x80 | (cp & 0x3F));
            }
            int id = vocab_map_get(vmap, utf8);
            byte_to_id[b] = (id >= 0) ? (uint32_t)id : 0;
            if (id >= 0) byte_mapped++;
        }
        fprintf(stderr, "  Byte-to-ID mapped: %d/256\n", byte_mapped);
        fwrite(byte_to_id, sizeof(uint32_t), 256, out);

        fclose(out);

        long file_size = 0;
        FILE *check = fopen(output_path, "rb");
        if (check) { fseek(check, 0, SEEK_END); file_size = ftell(check); fclose(check); }

        fprintf(stderr, "  Vocab: %.1f KB -> %s\n",
                (double)file_size / 1024.0, output_path);

cleanup_direct:
        vocab_map_free(vmap);
        for (int i = 0; i < n_tokens; i++) free(tokens[i]);
        free(tokens);
        for (int i = 0; i < n_tokens; i++) free(raw_tokens[i]);
        free(raw_tokens); free(raw_lengths);
        free(unicode_to_byte);
        free(offsets); free(lengths); free(blob);
        free(merge_a_ids); free(merge_b_ids); free(merge_result_ids);
        if (merge_a_str) {
            for (int i = 0; i < n_merges; i++) { free(merge_a_str[i]); free(merge_b_str[i]); }
            free(merge_a_str); free(merge_b_str);
        }
        return true;
    }

    /* Fallback: use tokenizer.json via ft_load() */
    if (tokenizer_path) {
        fprintf(stderr, "Loading tokenizer: %s\n", tokenizer_path);
        ft_tokenizer *tok = ft_load(tokenizer_path);
        if (!tok) {
            fprintf(stderr, "ERROR: failed to load tokenizer\n");
            return false;
        }

        int n_tokens = ft_vocab_size(tok);
        int n_merges = ft_num_merges(tok);
        fprintf(stderr, "  Vocab: %d tokens, %d merges\n", n_tokens, n_merges);

        /* Build string blob */
        size_t blob_size = 0;
        for (int i = 0; i < n_tokens; i++) {
            const char *s = ft_id_to_token(tok, i);
            blob_size += s ? strlen(s) : 0;
        }

        uint32_t *offsets = malloc(n_tokens * sizeof(uint32_t));
        uint16_t *lengths = malloc(n_tokens * sizeof(uint16_t));
        char *blob = malloc(blob_size + 1);
        size_t blob_pos = 0;

        for (int i = 0; i < n_tokens; i++) {
            const char *s = ft_id_to_token(tok, i);
            size_t len = s ? strlen(s) : 0;
            offsets[i] = (uint32_t)blob_pos;
            lengths[i] = (uint16_t)len;
            if (s && len > 0) {
                memcpy(blob + blob_pos, s, len);
                blob_pos += len;
            }
        }

        /* Collect special tokens */
        typedef struct { uint32_t id; const char *name; } special_entry_ft;
        special_entry_ft specials[8];
        int n_special = 0;

        const char *special_names[] = {
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<s>", "</s>", "<|begin_of_text|>", "<|end_of_text|>",
            NULL
        };
        for (int i = 0; special_names[i] && n_special < 8; i++) {
            int id = ft_token_to_id(tok, special_names[i]);
            if (id >= 0) {
                specials[n_special].id = (uint32_t)id;
                specials[n_special].name = special_names[i];
                n_special++;
            }
        }

        fprintf(stderr, "  Special tokens: %d\n", n_special);

        FILE *out = fopen(output_path, "wb");
        if (!out) {
            fprintf(stderr, "ERROR: cannot create %s\n", output_path);
            ft_free(tok);
            free(offsets); free(lengths); free(blob);
            return false;
        }

        btv_header hdr;
        memset(&hdr, 0, sizeof(hdr));
        memcpy(hdr.magic, "BTAV", 4);
        hdr.n_tokens = n_tokens;
        hdr.n_merges = n_merges;
        hdr.n_special = n_special;

        fwrite(&hdr, sizeof(hdr), 1, out);
        fwrite(offsets, sizeof(uint32_t), n_tokens, out);
        fwrite(lengths, sizeof(uint16_t), n_tokens, out);
        fwrite(blob, 1, blob_pos, out);

        /* Write merges as string pairs */
        for (int i = 0; i < n_merges; i++) {
            const char *a_str = NULL, *b_str = NULL;
            ft_get_merge(tok, i, &a_str, &b_str);
            uint16_t a_len = a_str ? (uint16_t)strlen(a_str) : 0;
            uint16_t b_len = b_str ? (uint16_t)strlen(b_str) : 0;
            fwrite(&a_len, sizeof(uint16_t), 1, out);
            if (a_len) fwrite(a_str, 1, a_len, out);
            fwrite(&b_len, sizeof(uint16_t), 1, out);
            if (b_len) fwrite(b_str, 1, b_len, out);
        }

        for (int i = 0; i < n_special; i++) {
            uint32_t id = specials[i].id;
            const char *name = specials[i].name;
            uint16_t name_len = (uint16_t)strlen(name);
            fwrite(&id, sizeof(uint32_t), 1, out);
            fwrite(&name_len, sizeof(uint16_t), 1, out);
            fwrite(name, 1, name_len, out);
        }

        fclose(out);

        long file_size = 0;
        FILE *check = fopen(output_path, "rb");
        if (check) { fseek(check, 0, SEEK_END); file_size = ftell(check); fclose(check); }

        fprintf(stderr, "  Vocab: %.1f KB -> %s\n",
                (double)file_size / 1024.0, output_path);

        ft_free(tok);
        free(offsets); free(lengths); free(blob);
        return true;
    }

    fprintf(stderr, "ERROR: no tokenizer files provided\n");
    return false;
}
