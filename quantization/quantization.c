/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "quantization.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================= */
/* Utility functions                                             */
/* ============================================================= */

const char* fq_dtype_name(fq_dtype dtype) {
    switch (dtype) {
        case FQ_INT8:    return "INT8";
        case FQ_INT4:    return "INT4";
        case FQ_UINT8:   return "UINT8";
        case FQ_UINT4:   return "UINT4";
        case FQ_TERNARY: return "TERNARY";
        default:         return "UNKNOWN";
    }
}

float fq_dtype_bytes(fq_dtype dtype) {
    switch (dtype) {
        case FQ_INT8:
        case FQ_UINT8:   return 1.0f;
        case FQ_INT4:
        case FQ_UINT4:   return 0.5f;
        case FQ_TERNARY: return 0.25f;
        default:         return 0.0f;
    }
}

static int64_t compute_numel(int64_t* shape, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        n *= shape[i];
    }
    return n;
}

/* Find min/max of data */
static void find_min_max(const float* data, int64_t n, float* min_val, float* max_val) {
    *min_val = FLT_MAX;
    *max_val = -FLT_MAX;

    for (int64_t i = 0; i < n; i++) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
    }
}

void fq_compute_scale(const float* data, int64_t n, fq_mode mode,
                      float* scale, int32_t* zero_point) {
    float min_val, max_val;
    find_min_max(data, n, &min_val, &max_val);

    if (mode == FQ_SYMMETRIC) {
        /* Symmetric: scale = max(|min|, |max|) / 127 */
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        *scale = abs_max / 127.0f;
        *zero_point = 0;
    } else {
        /* Asymmetric: map [min, max] to [0, 255] for uint8, [-128, 127] for int8 */
        float range = max_val - min_val;
        *scale = range / 255.0f;
        *zero_point = (int32_t)roundf(-min_val / *scale);
    }

    /* Avoid division by zero */
    if (*scale < 1e-10f) {
        *scale = 1e-10f;
    }
}

size_t fq_memory_usage(fq_tensor* qt) {
    if (!qt) return 0;

    size_t data_size = 0;
    if (qt->dtype == FQ_INT8 || qt->dtype == FQ_UINT8) {
        data_size = qt->numel;
    } else if (qt->dtype == FQ_TERNARY) {
        /* Ternary: 4 values packed per byte (2 bits each) */
        data_size = (qt->numel + 3) / 4;
    } else {
        /* INT4: 2 values packed per byte */
        data_size = (qt->numel + 1) / 2;
    }

    size_t scale_size = qt->num_scales * sizeof(float);
    size_t zp_size = (qt->mode == FQ_ASYMMETRIC) ? qt->num_scales * sizeof(int32_t) : 0;

    return data_size + scale_size + zp_size;
}

/* ============================================================= */
/* Tensor allocation                                             */
/* ============================================================= */

static fq_tensor* fq_alloc(int64_t* shape, int ndim, fq_dtype dtype,
                          fq_mode mode, fq_granularity granularity, int num_scales) {
    fq_tensor* qt = calloc(1, sizeof(fq_tensor));
    if (!qt) return NULL;

    qt->ndim = ndim;
    qt->numel = compute_numel(shape, ndim);
    qt->dtype = dtype;
    qt->mode = mode;
    qt->granularity = granularity;
    qt->num_scales = num_scales;

    qt->shape = malloc(ndim * sizeof(int64_t));
    memcpy(qt->shape, shape, ndim * sizeof(int64_t));

    /* Allocate data */
    if (dtype == FQ_INT8 || dtype == FQ_UINT8) {
        qt->data = calloc(qt->numel, 1);
    } else if (dtype == FQ_TERNARY) {
        /* Ternary: pack 4 values per byte (2 bits each) */
        qt->data = calloc((qt->numel + 3) / 4, 1);
    } else {
        /* INT4: pack 2 values per byte */
        qt->data = calloc((qt->numel + 1) / 2, 1);
    }

    /* Allocate scales */
    qt->scales = calloc(num_scales, sizeof(float));

    /* Allocate zero points for asymmetric */
    if (mode == FQ_ASYMMETRIC) {
        qt->zero_points = calloc(num_scales, sizeof(int32_t));
    }

    if (!qt->shape || !qt->data || !qt->scales) {
        fq_free(qt);
        return NULL;
    }

    return qt;
}

void fq_free(fq_tensor* qt) {
    if (!qt) return;

    free(qt->shape);
    free(qt->data);
    free(qt->scales);
    free(qt->zero_points);
    free(qt);
}

/* ============================================================= */
/* INT8 Quantization                                             */
/* ============================================================= */

fq_tensor* fq_quantize_int8(const float* data, int64_t* shape, int ndim) {
    if (!data || !shape || ndim <= 0) return NULL;

    fq_tensor* qt = fq_alloc(shape, ndim, FQ_INT8, FQ_SYMMETRIC, FQ_PER_TENSOR, 1);
    if (!qt) return NULL;

    /* Compute scale */
    float scale;
    int32_t zero_point;
    fq_compute_scale(data, qt->numel, FQ_SYMMETRIC, &scale, &zero_point);
    qt->scales[0] = scale;

    /* Quantize */
    int8_t* qdata = (int8_t*)qt->data;
    float inv_scale = 1.0f / scale;

    for (int64_t i = 0; i < qt->numel; i++) {
        float q = roundf(data[i] * inv_scale);
        if (q < -128.0f) q = -128.0f;
        if (q > 127.0f) q = 127.0f;
        qdata[i] = (int8_t)q;
    }

    return qt;
}

fq_tensor* fq_quantize_int8_per_channel(const float* data, int64_t* shape, int ndim, int channel_dim) {
    if (!data || !shape || ndim <= 0) return NULL;
    if (channel_dim < 0 || channel_dim >= ndim) return NULL;

    int num_channels = (int)shape[channel_dim];

    fq_tensor* qt = fq_alloc(shape, ndim, FQ_INT8, FQ_SYMMETRIC, FQ_PER_CHANNEL, num_channels);
    if (!qt) return NULL;

    /* Compute channel stride */
    int64_t channel_stride = 1;
    for (int i = channel_dim + 1; i < ndim; i++) {
        channel_stride *= shape[i];
    }

    /* First pass: compute scales per channel */
    for (int c = 0; c < num_channels; c++) {
        float abs_max = 0.0f;

        /* Find max in this channel */
        for (int64_t i = 0; i < qt->numel; i++) {
            /* Check if this element belongs to channel c */
            int64_t channel_idx = (i / channel_stride) % num_channels;
            if (channel_idx == c) {
                float abs_val = fabsf(data[i]);
                if (abs_val > abs_max) abs_max = abs_val;
            }
        }

        qt->scales[c] = abs_max / 127.0f;
        if (qt->scales[c] < 1e-10f) qt->scales[c] = 1e-10f;
    }

    /* Second pass: quantize */
    int8_t* qdata = (int8_t*)qt->data;

    for (int64_t i = 0; i < qt->numel; i++) {
        int64_t channel_idx = (i / channel_stride) % num_channels;
        float inv_scale = 1.0f / qt->scales[channel_idx];
        float q = roundf(data[i] * inv_scale);
        if (q < -128.0f) q = -128.0f;
        if (q > 127.0f) q = 127.0f;
        qdata[i] = (int8_t)q;
    }

    return qt;
}

/* ============================================================= */
/* INT4 Quantization                                             */
/* ============================================================= */

/* Pack two 4-bit values into one byte */
static void pack_int4(int8_t* packed, int64_t numel, const int8_t* values) {
    for (int64_t i = 0; i < numel; i += 2) {
        int8_t low = values[i] & 0x0F;
        int8_t high = (i + 1 < numel) ? (values[i + 1] & 0x0F) : 0;
        packed[i / 2] = (high << 4) | low;
    }
}

/* Unpack 4-bit values from packed byte */
static void unpack_int4(const int8_t* packed, int64_t numel, int8_t* values) {
    for (int64_t i = 0; i < numel; i += 2) {
        int8_t byte = packed[i / 2];
        values[i] = (byte & 0x0F);
        if (values[i] > 7) values[i] -= 16;  /* Sign extend */
        if (i + 1 < numel) {
            values[i + 1] = (byte >> 4);
            if (values[i + 1] > 7) values[i + 1] -= 16;  /* Sign extend */
        }
    }
}

/* ============================================================= */
/* Ternary Packing (2-bit: 4 values per byte)                    */
/* ============================================================= */

/* Encoding: 00 = -1, 01 = 0, 10 = +1 */
/* Byte layout: [val3:2bits][val2:2bits][val1:2bits][val0:2bits] */

static void pack_ternary(uint8_t *packed, int64_t numel, const int8_t *values) {
    int64_t n_bytes = (numel + 3) / 4;
    for (int64_t b = 0; b < n_bytes; b++) {
        uint8_t byte = 0;
        for (int j = 0; j < 4; j++) {
            int64_t idx = b * 4 + j;
            if (idx >= numel) break;
            uint8_t enc;
            if (values[idx] < 0)      enc = 0; /* -1 -> 00 */
            else if (values[idx] == 0) enc = 1; /* 0  -> 01 */
            else                       enc = 2; /* +1 -> 10 */
            byte |= (enc << (j * 2));
        }
        packed[b] = byte;
    }
}

static void unpack_ternary(const uint8_t *packed, int64_t numel, int8_t *values) {
    static const int8_t decode[4] = {-1, 0, 1, 0}; /* 00=-1, 01=0, 10=+1, 11=unused */
    for (int64_t i = 0; i < numel; i++) {
        uint8_t byte = packed[i / 4];
        int shift = (i % 4) * 2;
        uint8_t enc = (byte >> shift) & 0x03;
        values[i] = decode[enc];
    }
}

fq_tensor* fq_quantize_int4(const float* data, int64_t* shape, int ndim, int group_size) {
    if (!data || !shape || ndim <= 0) return NULL;
    if (group_size <= 0) group_size = 128;  /* Default group size */

    int64_t numel = compute_numel(shape, ndim);
    int num_groups = (int)((numel + group_size - 1) / group_size);

    fq_tensor* qt = fq_alloc(shape, ndim, FQ_INT4, FQ_SYMMETRIC, FQ_PER_GROUP, num_groups);
    if (!qt) return NULL;
    qt->group_size = group_size;

    /* Temporary buffer for unpacked values */
    int8_t* temp = malloc(numel);
    if (!temp) {
        fq_free(qt);
        return NULL;
    }

    /* Quantize per group */
    for (int g = 0; g < num_groups; g++) {
        int64_t start = g * group_size;
        int64_t end = start + group_size;
        if (end > numel) end = numel;
        /* Find max in group */
        float abs_max = 0.0f;
        for (int64_t i = start; i < end; i++) {
            float abs_val = fabsf(data[i]);
            if (abs_val > abs_max) abs_max = abs_val;
        }

        qt->scales[g] = abs_max / 7.0f;  /* INT4 range is -8 to 7 */
        if (qt->scales[g] < 1e-10f) qt->scales[g] = 1e-10f;

        float inv_scale = 1.0f / qt->scales[g];

        /* Quantize to INT4 range */
        for (int64_t i = start; i < end; i++) {
            float q = roundf(data[i] * inv_scale);
            if (q < -8.0f) q = -8.0f;
            if (q > 7.0f) q = 7.0f;
            temp[i] = (int8_t)q;
        }
    }

    /* Pack INT4 values */
    pack_int4((int8_t*)qt->data, numel, temp);

    free(temp);
    return qt;
}

/* ============================================================= */
/* Ternary Quantization (BitNet b1.58 absmean)                   */
/* ============================================================= */

fq_tensor* fq_quantize_ternary(const float* data, int64_t* shape, int ndim) {
    if (!data || !shape || ndim <= 0) return NULL;

    fq_tensor* qt = fq_alloc(shape, ndim, FQ_TERNARY, FQ_SYMMETRIC, FQ_PER_TENSOR, 1);
    if (!qt) return NULL;

    /* Compute absmean: γ = mean(|data|) */
    double abs_sum = 0.0;
    for (int64_t i = 0; i < qt->numel; i++) {
        abs_sum += fabsf(data[i]);
    }
    float gamma = (float)(abs_sum / qt->numel);
    if (gamma < 1e-10f) gamma = 1e-10f;
    qt->scales[0] = gamma;

    /* Quantize: round(data[i] / γ), clamp to {-1, 0, +1} */
    int8_t* temp = malloc(qt->numel);
    if (!temp) { fq_free(qt); return NULL; }

    float inv_gamma = 1.0f / gamma;
    for (int64_t i = 0; i < qt->numel; i++) {
        float q = roundf(data[i] * inv_gamma);
        if (q < -1.0f) q = -1.0f;
        if (q > 1.0f) q = 1.0f;
        temp[i] = (int8_t)q;
    }

    /* Pack into 2-bit format */
    pack_ternary((uint8_t*)qt->data, qt->numel, temp);

    free(temp);
    return qt;
}

fq_tensor* fq_quantize_ternary_threshold(const float* data, int64_t* shape, int ndim, float threshold) {
    if (!data || !shape || ndim <= 0) return NULL;
    if (threshold < 1e-10f) threshold = 1e-10f;

    fq_tensor* qt = fq_alloc(shape, ndim, FQ_TERNARY, FQ_SYMMETRIC, FQ_PER_TENSOR, 1);
    if (!qt) return NULL;

    qt->scales[0] = threshold; /* reconstruction: ternary_value * scale */

    /* Quantize to {-1, 0, +1} */
    int8_t* temp = malloc(qt->numel);
    if (!temp) { fq_free(qt); return NULL; }

    for (int64_t i = 0; i < qt->numel; i++) {
        if (data[i] > threshold) temp[i] = 1;
        else if (data[i] < -threshold) temp[i] = -1;
        else temp[i] = 0;
    }

    pack_ternary((uint8_t*)qt->data, qt->numel, temp);
    free(temp);

    return qt;
}

fq_tensor* fq_quantize_ternary_per_channel(const float* data, int64_t* shape, int ndim, int channel_dim) {
    if (!data || !shape || ndim <= 0) return NULL;
    if (channel_dim < 0 || channel_dim >= ndim) return NULL;

    int num_channels = (int)shape[channel_dim];

    fq_tensor* qt = fq_alloc(shape, ndim, FQ_TERNARY, FQ_SYMMETRIC, FQ_PER_CHANNEL, num_channels);
    if (!qt) return NULL;

    int64_t channel_stride = 1;
    for (int i = channel_dim + 1; i < ndim; i++) {
        channel_stride *= shape[i];
    }

    /* First pass: compute absmean threshold per channel */
    for (int c = 0; c < num_channels; c++) {
        double sum = 0.0;
        int64_t count = 0;
        for (int64_t i = 0; i < qt->numel; i++) {
            int64_t channel_idx = (i / channel_stride) % num_channels;
            if (channel_idx == c) {
                sum += fabsf(data[i]);
                count++;
            }
        }
        qt->scales[c] = (count > 0) ? (float)(sum / count) : 1e-10f;
        if (qt->scales[c] < 1e-10f) qt->scales[c] = 1e-10f;
    }

    /* Second pass: quantize */
    int8_t* temp = malloc(qt->numel);
    if (!temp) { fq_free(qt); return NULL; }

    for (int64_t i = 0; i < qt->numel; i++) {
        int64_t channel_idx = (i / channel_stride) % num_channels;
        float thresh = qt->scales[channel_idx];
        if (data[i] > thresh) temp[i] = 1;
        else if (data[i] < -thresh) temp[i] = -1;
        else temp[i] = 0;
    }

    pack_ternary((uint8_t*)qt->data, qt->numel, temp);
    free(temp);

    return qt;
}

/* ============================================================= */
/* Generic Quantization                                          */
/* ============================================================= */

fq_tensor* fq_quantize(const float* data, int64_t* shape, int ndim,
                       fq_dtype dtype, fq_mode mode, fq_granularity granularity,
                       int channel_dim, int group_size) {
    (void)mode;

    if (dtype == FQ_TERNARY) {
        if (granularity == FQ_PER_CHANNEL) {
            return fq_quantize_ternary_per_channel(data, shape, ndim, channel_dim);
        }
        return fq_quantize_ternary(data, shape, ndim);
    }

    if (dtype == FQ_INT4 || dtype == FQ_UINT4) {
        return fq_quantize_int4(data, shape, ndim, group_size);
    }

    if (granularity == FQ_PER_CHANNEL) {
        return fq_quantize_int8_per_channel(data, shape, ndim, channel_dim);
    }

    return fq_quantize_int8(data, shape, ndim);
}

/* ============================================================= */
/* Dequantization                                                */
/* ============================================================= */

float* fq_dequantize(fq_tensor* qt) {
    if (!qt) return NULL;

    float* data = malloc(qt->numel * sizeof(float));
    if (!data) return NULL;

    if (qt->dtype == FQ_TERNARY) {
        int8_t* temp = malloc(qt->numel);
        if (!temp) { free(data); return NULL; }

        unpack_ternary((const uint8_t*)qt->data, qt->numel, temp);

        if (qt->granularity == FQ_PER_CHANNEL) {
            int num_channels = qt->num_scales;
            int64_t elements_per_channel = qt->numel / num_channels;
            for (int64_t i = 0; i < qt->numel; i++) {
                int channel_idx = (int)(i / elements_per_channel);
                data[i] = temp[i] * qt->scales[channel_idx];
            }
        } else {
            float scale = qt->scales[0];
            for (int64_t i = 0; i < qt->numel; i++) {
                data[i] = temp[i] * scale;
            }
        }

        free(temp);
    } else if (qt->dtype == FQ_INT8 || qt->dtype == FQ_UINT8) {
        int8_t* qdata = (int8_t*)qt->data;

        if (qt->granularity == FQ_PER_TENSOR) {
            float scale = qt->scales[0];
            for (int64_t i = 0; i < qt->numel; i++) {
                data[i] = qdata[i] * scale;
            }
        } else if (qt->granularity == FQ_PER_CHANNEL) {
            /* Per-channel: elements are grouped by channel (first dim) */
            int num_channels = qt->num_scales;
            int64_t elements_per_channel = qt->numel / num_channels;

            for (int64_t i = 0; i < qt->numel; i++) {
                int channel_idx = (int)(i / elements_per_channel);
                data[i] = qdata[i] * qt->scales[channel_idx];
            }
        }
    } else {
        /* INT4 */
        int8_t* temp = malloc(qt->numel);
        if (!temp) {
            free(data);
            return NULL;
        }

        unpack_int4((int8_t*)qt->data, qt->numel, temp);

        if (qt->granularity == FQ_PER_GROUP) {
            int group_size = qt->group_size;
            for (int64_t i = 0; i < qt->numel; i++) {
                int group_idx = (int)(i / group_size);
                data[i] = temp[i] * qt->scales[group_idx];
            }
        } else {
            float scale = qt->scales[0];
            for (int64_t i = 0; i < qt->numel; i++) {
                data[i] = temp[i] * scale;
            }
        }

        free(temp);
    }

    return data;
}

float* fq_dequantize_range(fq_tensor* qt, int64_t start, int64_t end) {
    if (!qt || start < 0 || end > qt->numel || start >= end) return NULL;

    int64_t len = end - start;
    float* data = malloc(len * sizeof(float));
    if (!data) return NULL;

    /* For simplicity, dequantize all and copy range */
    float* full = fq_dequantize(qt);
    if (!full) {
        free(data);
        return NULL;
    }

    memcpy(data, full + start, len * sizeof(float));
    free(full);

    return data;
}

/* ============================================================= */
/* Quantized Operations                                          */
/* ============================================================= */

float* fq_matmul(const float* A, fq_tensor* B, int M, int K, int N) {
    if (!A || !B) return NULL;

    float* C = calloc(M * N, sizeof(float));
    if (!C) return NULL;

    fq_matmul_acc(A, B, C, M, K, N);
    return C;
}

void fq_matmul_acc(const float* A, fq_tensor* B, float* C, int M, int K, int N) {
    if (!A || !B || !C) return;

    /* Dispatch ternary to native integer-only kernel */
    if (B->dtype == FQ_TERNARY) {
        fq_matmul_ternary_acc(A, B, C, M, K, N);
        return;
    }

    /* Dequantize B for now (could be optimized with int8 accumulation) */
    float* B_f32 = fq_dequantize(B);
    if (!B_f32) return;

    /* Standard matmul: C[m,n] = sum_k A[m,k] * B[k,n] */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B_f32[k * N + n];
            }
            C[m * N + n] += sum;
        }
    }

    free(B_f32);
}

/* ============================================================= */
/* Native Ternary Matmul (integer-only, no dequant)              */
/* ============================================================= */

void fq_matmul_ternary_acc(const float* A, fq_tensor* B, float* C, int M, int K, int N) {
    if (!A || !B || !C) return;

    const uint8_t* packed = (const uint8_t*)B->data;
    float scale = B->scales[0];
    static const int8_t ternary_val[4] = {-1, 0, 1, 0};

    /* C[m,n] += scale * sum_k( A[m,k] * ternary(B[k,n]) ) */
    /* B is K×N row-major, element [k,n] is at linear index k*N+n */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                int64_t idx = (int64_t)k * N + n;
                uint8_t byte = packed[idx / 4];
                int shift = (idx % 4) * 2;
                int8_t w = ternary_val[(byte >> shift) & 0x03];
                /* Branch-free: w is -1, 0, or +1 so this is add/nop/sub */
                sum += A[m * K + k] * w;
            }
            C[m * N + n] += sum * scale;
        }
    }
}

float* fq_matmul_ternary(const float* A, fq_tensor* B, int M, int K, int N) {
    if (!A || !B) return NULL;

    float* C = calloc(M * N, sizeof(float));
    if (!C) return NULL;

    fq_matmul_ternary_acc(A, B, C, M, K, N);
    return C;
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

void fq_print(fq_tensor* qt) {
    if (!qt) {
        printf("(null quantized tensor)\n");
        return;
    }

    printf("Quantized tensor: %s, [", fq_dtype_name(qt->dtype));
    for (int i = 0; i < qt->ndim; i++) {
        if (i > 0) printf(", ");
        printf("%lld", (long long)qt->shape[i]);
    }
    printf("]\n");

    printf("  Mode: %s\n", qt->mode == FQ_SYMMETRIC ? "symmetric" : "asymmetric");
    printf("  Granularity: %s (%d scales)\n",
           qt->granularity == FQ_PER_TENSOR ? "per-tensor" :
           qt->granularity == FQ_PER_CHANNEL ? "per-channel" : "per-group",
           qt->num_scales);
    printf("  Memory: %.2f KB (%.1fx compression)\n",
           fq_memory_usage(qt) / 1024.0f,
           (qt->numel * 4.0f) / fq_memory_usage(qt));
}

float fq_quantization_error(const float* original, fq_tensor* quantized) {
    if (!original || !quantized) return -1.0f;

    float* dequant = fq_dequantize(quantized);
    if (!dequant) return -1.0f;

    double mse = 0.0;
    for (int64_t i = 0; i < quantized->numel; i++) {
        double diff = original[i] - dequant[i];
        mse += diff * diff;
    }
    mse /= quantized->numel;

    free(dequant);
    return (float)mse;
}
