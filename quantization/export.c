/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "export.h"
#include "quantization.h"
#include "../safetensors/safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================= */
/* Internal state                                                */
/* ============================================================= */

struct fq_export_state {
    fs_file* input;
    fs_writer* writer;
    fq_export_config config;

    const char** tensor_names;
    int num_tensors;
    int current;

    int quantized_count;
    int passthrough_count;
};

/* ============================================================= */
/* BF16/F16 -> F32 conversion (self-contained)                   */
/* ============================================================= */

static void bf16_to_f32(const uint16_t* src, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(&dst[i], &bits, 4);
    }
}

static void f16_to_f32(const uint16_t* src, float* dst, int64_t n) {
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

/* ============================================================= */
/* Tensor skip logic                                             */
/* ============================================================= */

static bool str_contains(const char* haystack, const char* needle) {
    return strstr(haystack, needle) != NULL;
}

static bool str_ends_with(const char* s, const char* suffix) {
    size_t slen = strlen(s);
    size_t sufflen = strlen(suffix);
    if (sufflen > slen) return false;
    return strcmp(s + slen - sufflen, suffix) == 0;
}

static bool should_quantize(const char* name, fs_tensor_info* info, fq_export_config* cfg) {
    /* Only quantize float types */
    if (info->dtype != FS_DTYPE_F32 && info->dtype != FS_DTYPE_F16 &&
        info->dtype != FS_DTYPE_BF16) {
        return false;
    }

    /* Compute numel */
    int64_t numel = 1;
    for (int i = 0; i < info->ndim; i++) numel *= info->shape[i];

    /* Skip small tensors */
    if (numel < cfg->min_elements) return false;

    /* Skip 1D tensors */
    if (cfg->skip_1d && info->ndim <= 1) return false;

    /* Skip embeddings */
    if (cfg->skip_embeddings) {
        if (str_contains(name, "embed") || str_contains(name, "wte") ||
            str_contains(name, "wpe")) {
            return false;
        }
    }

    /* Skip norm layers */
    if (cfg->skip_norms) {
        if (str_contains(name, "norm") || str_contains(name, "ln_") ||
            str_contains(name, "layernorm") || str_contains(name, "rmsnorm")) {
            return false;
        }
    }

    /* Skip head */
    if (cfg->skip_head) {
        if (str_contains(name, "lm_head") || str_contains(name, "head.weight") ||
            str_contains(name, "output.weight")) {
            return false;
        }
    }

    /* Skip biases */
    if (cfg->skip_biases) {
        if (str_ends_with(name, ".bias")) return false;
    }

    return true;
}

/* ============================================================= */
/* Defaults                                                      */
/* ============================================================= */

fq_export_config fq_export_config_defaults(void) {
    fq_export_config cfg = {0};
    cfg.target_dtype = FQ_INT8;
    cfg.granularity = FQ_PER_TENSOR;
    cfg.group_size = 128;
    cfg.ternary_threshold = 0.0f;  /* auto */
    cfg.skip_embeddings = true;
    cfg.skip_norms = true;
    cfg.skip_head = false;
    cfg.skip_biases = true;
    cfg.skip_1d = true;
    cfg.min_elements = 1024;
    cfg.progress_cb = NULL;
    cfg.userdata = NULL;
    return cfg;
}

/* ============================================================= */
/* Internal: process one tensor                                  */
/* ============================================================= */

static bool process_tensor(fq_export_state* s, int idx) {
    const char* name = s->tensor_names[idx];
    fs_tensor_info* info = fs_get_info(s->input, name);
    if (!info) return false;

    if (!should_quantize(name, info, &s->config)) {
        /* Passthrough: copy raw bytes with original dtype */
        const void* raw = fs_get_data(s->input, name);
        if (raw) {
            fs_writer_add_tensor(s->writer, name, info->dtype,
                                 info->shape, info->ndim, raw);
        }
        s->passthrough_count++;
        fs_free_info(info);
        return true;
    }

    /* Compute numel */
    int64_t numel = 1;
    for (int i = 0; i < info->ndim; i++) numel *= info->shape[i];

    /* Get data as F32 */
    float* f32_data = NULL;
    bool owns_f32 = false;

    if (info->dtype == FS_DTYPE_F32) {
        f32_data = (float*)fs_get_data(s->input, name);
    } else if (info->dtype == FS_DTYPE_BF16) {
        f32_data = malloc(numel * sizeof(float));
        if (!f32_data) { fs_free_info(info); return false; }
        bf16_to_f32((const uint16_t*)fs_get_data(s->input, name), f32_data, numel);
        owns_f32 = true;
    } else if (info->dtype == FS_DTYPE_F16) {
        f32_data = malloc(numel * sizeof(float));
        if (!f32_data) { fs_free_info(info); return false; }
        f16_to_f32((const uint16_t*)fs_get_data(s->input, name), f32_data, numel);
        owns_f32 = true;
    }

    if (!f32_data) {
        fs_free_info(info);
        return false;
    }

    /* Quantize */
    fq_tensor* qt = NULL;

    if (s->config.target_dtype == FQ_TERNARY) {
        if (s->config.ternary_threshold > 0.0f) {
            qt = fq_quantize_ternary_threshold(f32_data, info->shape, info->ndim,
                                                s->config.ternary_threshold);
        } else if (s->config.granularity == FQ_PER_CHANNEL) {
            qt = fq_quantize_ternary_per_channel(f32_data, info->shape, info->ndim, 0);
        } else {
            qt = fq_quantize_ternary(f32_data, info->shape, info->ndim);
        }
    } else {
        qt = fq_quantize(f32_data, info->shape, info->ndim,
                         s->config.target_dtype, FQ_SYMMETRIC,
                         s->config.granularity, 0, s->config.group_size);
    }

    if (owns_f32) free(f32_data);

    if (!qt) {
        fs_free_info(info);
        return false;
    }

    /* Write quantized data tensor */
    fs_dtype out_dtype;
    size_t data_bytes;

    if (qt->dtype == FQ_INT8) {
        out_dtype = FS_DTYPE_I8;
        data_bytes = numel;
    } else if (qt->dtype == FQ_INT4 || qt->dtype == FQ_UINT4) {
        out_dtype = FS_DTYPE_U8;
        data_bytes = (numel + 1) / 2;
    } else { /* FQ_TERNARY */
        out_dtype = FS_DTYPE_U8;
        data_bytes = (numel + 3) / 4;
    }

    /* Data tensor: shape reflects the packed byte count for INT4/ternary */
    int64_t packed_shape[] = {(int64_t)data_bytes};
    if (qt->dtype == FQ_INT8) {
        fs_writer_add_tensor(s->writer, name, out_dtype,
                             info->shape, info->ndim, qt->data);
    } else {
        fs_writer_add_tensor(s->writer, name, out_dtype,
                             packed_shape, 1, qt->data);
    }

    /* Scale tensor */
    char scale_name[512];
    snprintf(scale_name, sizeof(scale_name), "%s.__scale", name);
    int64_t scale_shape[] = {qt->num_scales};
    fs_writer_add_tensor(s->writer, scale_name, FS_DTYPE_F32,
                         scale_shape, 1, qt->scales);

    /* Shape tensor (original dimensions, needed for packed formats) */
    char shape_name[512];
    snprintf(shape_name, sizeof(shape_name), "%s.__shape", name);
    fs_writer_add_tensor(s->writer, shape_name, FS_DTYPE_I64,
                         (int64_t[]){info->ndim}, 1, info->shape);

    /* Zero point tensor (only for asymmetric) */
    if (qt->mode == FQ_ASYMMETRIC && qt->zero_points) {
        char zp_name[512];
        snprintf(zp_name, sizeof(zp_name), "%s.__zp", name);
        int64_t zp_shape[] = {qt->num_scales};
        fs_writer_add_tensor(s->writer, zp_name, FS_DTYPE_I32,
                             zp_shape, 1, qt->zero_points);
    }

    s->quantized_count++;
    fq_free(qt);
    fs_free_info(info);
    return true;
}

/* ============================================================= */
/* Step-by-step API                                              */
/* ============================================================= */

fq_export_state* fq_export_begin(const char* input, fq_export_config* cfg) {
    if (!input || !cfg) return NULL;

    fs_file* f = fs_open(input);
    if (!f) return NULL;

    fq_export_state* s = calloc(1, sizeof(fq_export_state));
    if (!s) { fs_close(f); return NULL; }

    s->input = f;
    s->config = *cfg;

    s->tensor_names = fs_tensor_names(f, &s->num_tensors);
    if (!s->tensor_names) {
        fs_close(f);
        free(s);
        return NULL;
    }

    s->writer = fs_writer_new();
    if (!s->writer) {
        free((void*)s->tensor_names);
        fs_close(f);
        free(s);
        return NULL;
    }

    /* Add quantization metadata */
    const char* dtype_str = fq_dtype_name(cfg->target_dtype);
    fs_writer_add_metadata(s->writer, "quantization.dtype", dtype_str);

    const char* gran_str = "per_tensor";
    if (cfg->granularity == FQ_PER_CHANNEL) gran_str = "per_channel";
    else if (cfg->granularity == FQ_PER_GROUP) gran_str = "per_group";
    fs_writer_add_metadata(s->writer, "quantization.granularity", gran_str);

    char gs_buf[32];
    snprintf(gs_buf, sizeof(gs_buf), "%d", cfg->group_size);
    fs_writer_add_metadata(s->writer, "quantization.group_size", gs_buf);

    fs_writer_add_metadata(s->writer, "quantization.format_version", "1");

    s->current = 0;
    return s;
}

int fq_export_tensor_count(fq_export_state* s) {
    return s ? s->num_tensors : 0;
}

bool fq_export_step(fq_export_state* s) {
    if (!s || s->current >= s->num_tensors) return false;

    process_tensor(s, s->current);

    if (s->config.progress_cb) {
        s->config.progress_cb(s->current + 1, s->num_tensors,
                              s->tensor_names[s->current], s->config.userdata);
    }

    s->current++;
    return s->current < s->num_tensors;
}

float fq_export_progress(fq_export_state* s) {
    if (!s || s->num_tensors == 0) return 0.0f;
    return (float)s->current / s->num_tensors;
}

const char* fq_export_current_tensor(fq_export_state* s) {
    if (!s || s->current >= s->num_tensors) return NULL;
    return s->tensor_names[s->current];
}

bool fq_export_finish(fq_export_state* s, const char* output) {
    if (!s || !output) return false;

    bool ok = fs_writer_save(s->writer, output);

    fs_writer_free(s->writer);
    free((void*)s->tensor_names);
    fs_close(s->input);
    free(s);

    return ok;
}

void fq_export_abort(fq_export_state* s) {
    if (!s) return;

    fs_writer_free(s->writer);
    free((void*)s->tensor_names);
    fs_close(s->input);
    free(s);
}

/* ============================================================= */
/* One-shot API                                                  */
/* ============================================================= */

bool fq_export_quantized(const char* input, const char* output, fq_export_config* cfg) {
    if (!input || !output || !cfg) return false;

    fq_export_state* s = fq_export_begin(input, cfg);
    if (!s) return false;

    while (fq_export_step(s)) {
        /* process all tensors */
    }

    return fq_export_finish(s, output);
}
