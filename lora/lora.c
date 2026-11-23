/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "lora.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================= */
/* Config                                                        */
/* ============================================================= */

fl_config fl_config_default(void) {
    fl_config config = {
        .rank = 8,
        .alpha = 16.0f,
        .dropout = 0.0f,
        .fan_in_fan_out = false
    };
    return config;
}

/* ============================================================= */
/* LoRA Layer                                                    */
/* ============================================================= */

fl_layer* fl_layer_new(const char* name, int d_in, int d_out, fl_config* config) {
    if (!name || d_in <= 0 || d_out <= 0) return NULL;

    fl_config cfg = config ? *config : fl_config_default();
    if (cfg.rank <= 0) return NULL;

    fl_layer* layer = calloc(1, sizeof(fl_layer));
    if (!layer) return NULL;

    layer->name = strdup(name);
    layer->d_in = d_in;
    layer->d_out = d_out;
    layer->rank = cfg.rank;
    layer->scale = cfg.alpha / (float)cfg.rank;

    /* Create A matrix: (rank x d_in) - initialized with kaiming uniform */
    int64_t shape_A[] = {cfg.rank, d_in};
    layer->lora_A = tm_randn(shape_A, 2, TM_F32);
    if (layer->lora_A) {
        /* Scale for kaiming initialization: sqrt(1/d_in) */
        float* data = tm_data_f32(layer->lora_A);
        float scale = sqrtf(1.0f / (float)d_in);
        for (int64_t i = 0; i < cfg.rank * d_in; i++) {
            data[i] *= scale;
        }
        tm_set_requires_grad(layer->lora_A, true);
    }

    /* Create B matrix: (d_out x rank) - initialized to zeros */
    int64_t shape_B[] = {d_out, cfg.rank};
    layer->lora_B = tm_zeros(shape_B, 2, TM_F32);
    if (layer->lora_B) {
        tm_set_requires_grad(layer->lora_B, true);
    }

    if (!layer->lora_A || !layer->lora_B) {
        fl_layer_free(layer);
        return NULL;
    }

    return layer;
}

fl_layer* fl_layer_from_tensors(const char* name, tm_tensor* A, tm_tensor* B, float alpha) {
    if (!name || !A || !B) return NULL;

    /* Validate shapes */
    if (A->ndim != 2 || B->ndim != 2) return NULL;
    if (A->shape[0] != B->shape[1]) return NULL;  /* rank must match */

    fl_layer* layer = calloc(1, sizeof(fl_layer));
    if (!layer) return NULL;

    layer->name = strdup(name);
    layer->d_in = A->shape[1];
    layer->d_out = B->shape[0];
    layer->rank = A->shape[0];
    layer->scale = alpha / (float)layer->rank;

    /* Clone tensors (take ownership) */
    layer->lora_A = tm_clone(A);
    layer->lora_B = tm_clone(B);

    if (!layer->lora_A || !layer->lora_B) {
        fl_layer_free(layer);
        return NULL;
    }

    return layer;
}

void fl_layer_free(fl_layer* layer) {
    if (!layer) return;

    free(layer->name);
    if (layer->lora_A) tm_release(layer->lora_A);
    if (layer->lora_B) tm_release(layer->lora_B);
    free(layer);
}

tm_tensor* fl_layer_forward(fl_layer* layer, tm_tensor* x) {
    if (!layer || !x) return NULL;

    /* LoRA forward: scale * B @ (A @ x)
     * For 2D input (batch x d_in): x @ A^T @ B^T
     * For 1D input (d_in): A @ x, then B @ result
     */

    /* Transpose A for matmul: A^T is (d_in x rank) */
    tm_tensor* At = tm_transpose(layer->lora_A, 0, 1);
    if (!At) return NULL;

    /* First matmul: x @ A^T -> (batch x rank) or (rank) */
    tm_tensor* hidden = tm_matmul(x, At);
    tm_release(At);
    if (!hidden) return NULL;

    /* Transpose B for matmul: B^T is (rank x d_out) */
    tm_tensor* Bt = tm_transpose(layer->lora_B, 0, 1);
    if (!Bt) {
        tm_release(hidden);
        return NULL;
    }

    /* Second matmul: hidden @ B^T -> (batch x d_out) or (d_out) */
    tm_tensor* output = tm_matmul(hidden, Bt);
    tm_release(hidden);
    tm_release(Bt);
    if (!output) return NULL;

    /* Scale by alpha/rank */
    tm_tensor* scaled = tm_mul_scalar(output, layer->scale);
    tm_release(output);

    return scaled;
}

tm_tensor* fl_layer_get_merged(fl_layer* layer, tm_tensor* W) {
    if (!layer || !W) return NULL;

    /* W_merged = W + scale * B @ A
     * W is (d_out x d_in)
     * B is (d_out x rank)
     * A is (rank x d_in)
     * B @ A is (d_out x d_in)
     */

    /* Compute B @ A */
    tm_tensor* BA = tm_matmul(layer->lora_B, layer->lora_A);
    if (!BA) return NULL;

    /* Scale by alpha/rank */
    tm_tensor* scaled = tm_mul_scalar(BA, layer->scale);
    tm_release(BA);
    if (!scaled) return NULL;

    /* Add to W */
    tm_tensor* merged = tm_add(W, scaled);
    tm_release(scaled);

    return merged;
}

tm_tensor* fl_layer_merge(fl_layer* layer, tm_tensor* W) {
    /* Get merged result and copy back to W */
    tm_tensor* merged = fl_layer_get_merged(layer, W);
    if (!merged) return NULL;

    /* Copy data from merged to W */
    float* dst = tm_data_f32(W);
    float* src = tm_data_f32(merged);
    if (dst && src) {
        memcpy(dst, src, W->numel * sizeof(float));
    }

    tm_release(merged);
    return W;
}

/* ============================================================= */
/* LoRA Adapter                                                  */
/* ============================================================= */

struct fl_adapter {
    fl_layer** layers;
    int num_layers;
    int capacity;
};

fl_adapter* fl_adapter_new(void) {
    fl_adapter* adapter = calloc(1, sizeof(fl_adapter));
    if (!adapter) return NULL;

    adapter->capacity = 16;
    adapter->layers = calloc(adapter->capacity, sizeof(fl_layer*));
    if (!adapter->layers) {
        free(adapter);
        return NULL;
    }

    return adapter;
}

void fl_adapter_free(fl_adapter* adapter) {
    if (!adapter) return;

    for (int i = 0; i < adapter->num_layers; i++) {
        fl_layer_free(adapter->layers[i]);
    }
    free(adapter->layers);
    free(adapter);
}

void fl_adapter_add(fl_adapter* adapter, fl_layer* layer) {
    if (!adapter || !layer) return;

    if (adapter->num_layers >= adapter->capacity) {
        adapter->capacity *= 2;
        adapter->layers = realloc(adapter->layers, adapter->capacity * sizeof(fl_layer*));
    }

    adapter->layers[adapter->num_layers++] = layer;
}

fl_layer* fl_adapter_get(fl_adapter* adapter, const char* name) {
    if (!adapter || !name) return NULL;

    for (int i = 0; i < adapter->num_layers; i++) {
        if (strcmp(adapter->layers[i]->name, name) == 0) {
            return adapter->layers[i];
        }
    }
    return NULL;
}

int fl_adapter_size(fl_adapter* adapter) {
    return adapter ? adapter->num_layers : 0;
}

fl_layer* fl_adapter_get_idx(fl_adapter* adapter, int idx) {
    if (!adapter || idx < 0 || idx >= adapter->num_layers) return NULL;
    return adapter->layers[idx];
}

int64_t fl_adapter_num_params(fl_adapter* adapter) {
    if (!adapter) return 0;

    int64_t total = 0;
    for (int i = 0; i < adapter->num_layers; i++) {
        fl_layer* layer = adapter->layers[i];
        /* A: rank x d_in, B: d_out x rank */
        total += (int64_t)layer->rank * layer->d_in;
        total += (int64_t)layer->d_out * layer->rank;
    }
    return total;
}

/* ============================================================= */
/* Adapter I/O                                                   */
/* ============================================================= */

fl_adapter* fl_adapter_load(const char* path, fl_config* config) {
    if (!path) return NULL;

    fs_file* f = fs_open(path);
    if (!f) return NULL;

    fl_adapter* adapter = fl_adapter_new();
    if (!adapter) {
        fs_close(f);
        return NULL;
    }

    fl_config cfg = config ? *config : fl_config_default();

    /* Get alpha from metadata if present */
    const char* alpha_str = fs_get_metadata(f, "lora_alpha");
    if (alpha_str) {
        cfg.alpha = (float)atof(alpha_str);
    }

    /* Iterate through tensors, pairing A and B matrices */
    int count;
    const char** names = fs_tensor_names(f, &count);
    if (!names) {
        fs_close(f);
        fl_adapter_free(adapter);
        return NULL;
    }

    /* Find matching A/B pairs */
    for (int i = 0; i < count; i++) {
        const char* name = names[i];
        size_t len = strlen(name);

        /* Look for ".lora_A" suffix */
        if (len > 7 && strcmp(name + len - 7, ".lora_A") == 0) {
            /* Extract base name */
            char* base = malloc(len - 6);  /* len - 7 + 1 for null */
            strncpy(base, name, len - 7);
            base[len - 7] = '\0';

            /* Find corresponding B matrix */
            size_t b_name_len = len - 7 + 8;  /* base + ".lora_B" + null */
            char* b_name = malloc(b_name_len + 1);
            snprintf(b_name, b_name_len + 1, "%s.lora_B", base);

            if (fs_has_tensor(f, b_name)) {
                /* Load both tensors */
                fs_tensor_info* info_A = fs_get_info(f, name);
                fs_tensor_info* info_B = fs_get_info(f, b_name);

                if (info_A && info_B && info_A->dtype == FS_DTYPE_F32 && info_B->dtype == FS_DTYPE_F32) {
                    const float* data_A = fs_get_f32(f, name);
                    const float* data_B = fs_get_f32(f, b_name);

                    if (data_A && data_B) {
                        /* Create tensors */
                        tm_tensor* A = tm_from_data((void*)data_A, info_A->shape, info_A->ndim, TM_F32);
                        tm_tensor* B = tm_from_data((void*)data_B, info_B->shape, info_B->ndim, TM_F32);

                        if (A && B) {
                            /* Clone since from_data doesn't copy */
                            tm_tensor* A_clone = tm_clone(A);
                            tm_tensor* B_clone = tm_clone(B);
                            tm_release(A);
                            tm_release(B);

                            if (A_clone && B_clone) {
                                fl_layer* layer = fl_layer_from_tensors(base, A_clone, B_clone, cfg.alpha);
                                if (layer) {
                                    fl_adapter_add(adapter, layer);
                                }
                                tm_release(A_clone);
                                tm_release(B_clone);
                            }
                        }
                    }
                }

                fs_free_info(info_A);
                fs_free_info(info_B);
            }

            free(base);
            free(b_name);
        }
    }

    free(names);
    fs_close(f);

    return adapter;
}

bool fl_adapter_save(fl_adapter* adapter, const char* path) {
    if (!adapter || !path) return false;

    fs_writer* w = fs_writer_new();
    if (!w) return false;

    /* Add metadata */
    if (adapter->num_layers > 0) {
        char alpha_str[32];
        float alpha = adapter->layers[0]->scale * adapter->layers[0]->rank;
        snprintf(alpha_str, sizeof(alpha_str), "%.1f", alpha);
        fs_writer_add_metadata(w, "lora_alpha", alpha_str);

        char rank_str[16];
        snprintf(rank_str, sizeof(rank_str), "%d", adapter->layers[0]->rank);
        fs_writer_add_metadata(w, "lora_rank", rank_str);
    }

    /* Save each layer's A and B matrices */
    for (int i = 0; i < adapter->num_layers; i++) {
        fl_layer* layer = adapter->layers[i];

        char name_A[256], name_B[256];
        snprintf(name_A, sizeof(name_A), "%s.lora_A", layer->name);
        snprintf(name_B, sizeof(name_B), "%s.lora_B", layer->name);

        fs_writer_add_f32(w, name_A, layer->lora_A->shape, layer->lora_A->ndim,
                          tm_data_f32(layer->lora_A));
        fs_writer_add_f32(w, name_B, layer->lora_B->shape, layer->lora_B->ndim,
                          tm_data_f32(layer->lora_B));
    }

    bool success = fs_writer_save(w, path);
    fs_writer_free(w);
    return success;
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

void fl_adapter_print(fl_adapter* adapter) {
    if (!adapter) {
        printf("(null adapter)\n");
        return;
    }

    printf("LoRA Adapter: %d layers, %lld parameters\n",
           adapter->num_layers, (long long)fl_adapter_num_params(adapter));

    for (int i = 0; i < adapter->num_layers; i++) {
        fl_layer* layer = adapter->layers[i];
        printf("  %s: (%d x %d), rank=%d, scale=%.4f\n",
               layer->name, layer->d_out, layer->d_in,
               layer->rank, layer->scale);
    }
}
