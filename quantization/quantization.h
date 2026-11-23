/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef FOUNDRY_QUANTIZATION_H
#define FOUNDRY_QUANTIZATION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tensor quantization for model compression.
 * Replaces bitsandbytes.
 *
 * Supports INT8, INT4 quantization with per-tensor or per-channel scaling.
 */

/* ============================================================= */
/* Types                                                         */
/* ============================================================= */

/* Quantization data types */
typedef enum {
    FQ_INT8,    /* 8-bit signed integer */
    FQ_INT4,    /* 4-bit signed integer (packed) */
    FQ_UINT8,   /* 8-bit unsigned integer */
    FQ_UINT4,   /* 4-bit unsigned integer (packed) */
    FQ_TERNARY, /* 2-bit ternary {-1, 0, +1} packed, 4 values per byte */
} fq_dtype;

/* Quantization mode */
typedef enum {
    FQ_SYMMETRIC,     /* Zero-point is always 0 */
    FQ_ASYMMETRIC,    /* Zero-point can be non-zero */
} fq_mode;

/* Quantization granularity */
typedef enum {
    FQ_PER_TENSOR,    /* Single scale for entire tensor */
    FQ_PER_CHANNEL,   /* Different scale for each channel */
    FQ_PER_GROUP,     /* Group quantization (like GPTQ) */
} fq_granularity;

/* Quantized tensor */
typedef struct {
    void* data;           /* Quantized data (int8_t* or packed int4) */
    int64_t* shape;       /* Original shape */
    int ndim;
    int64_t numel;        /* Number of elements */

    fq_dtype dtype;
    fq_mode mode;
    fq_granularity granularity;

    float* scales;        /* Scale factors */
    int32_t* zero_points; /* Zero points (for asymmetric) */
    int num_scales;       /* Number of scale values */
    int group_size;       /* Elements per group (for per-group) */
} fq_tensor;

/* ============================================================= */
/* Quantization                                                  */
/* ============================================================= */

/* Quantize float32 tensor to INT8 (symmetric, per-tensor) */
fq_tensor* fq_quantize_int8(const float* data, int64_t* shape, int ndim);

/* Quantize float32 tensor to INT8 per-channel */
fq_tensor* fq_quantize_int8_per_channel(const float* data, int64_t* shape, int ndim, int channel_dim);

/* Quantize float32 tensor to INT4 (per-group) */
fq_tensor* fq_quantize_int4(const float* data, int64_t* shape, int ndim, int group_size);

/* Quantize float32 tensor to ternary {-1, 0, +1} using absmean (BitNet b1.58) */
fq_tensor* fq_quantize_ternary(const float* data, int64_t* shape, int ndim);

/* Quantize float32 tensor to ternary with manual threshold */
fq_tensor* fq_quantize_ternary_threshold(const float* data, int64_t* shape, int ndim, float threshold);

/* Quantize float32 tensor to ternary per-channel */
fq_tensor* fq_quantize_ternary_per_channel(const float* data, int64_t* shape, int ndim, int channel_dim);

/* Generic quantization with options */
fq_tensor* fq_quantize(const float* data, int64_t* shape, int ndim,
                       fq_dtype dtype, fq_mode mode, fq_granularity granularity,
                       int channel_dim, int group_size);

/* Free quantized tensor */
void fq_free(fq_tensor* qt);

/* ============================================================= */
/* Dequantization                                                */
/* ============================================================= */

/* Dequantize to float32 */
float* fq_dequantize(fq_tensor* qt);

/* Dequantize subset (for lazy loading) */
float* fq_dequantize_range(fq_tensor* qt, int64_t start, int64_t end);

/* ============================================================= */
/* Quantized Operations                                          */
/* ============================================================= */

/* Quantized matrix multiplication: C = A @ B where A is float, B is quantized */
float* fq_matmul(const float* A, fq_tensor* B, int M, int K, int N);

/* Quantized matrix multiplication with accumulator */
void fq_matmul_acc(const float* A, fq_tensor* B, float* C, int M, int K, int N);

/* Native ternary matmul: C = A @ B where B is ternary (integer-only, no dequant) */
float* fq_matmul_ternary(const float* A, fq_tensor* B, int M, int K, int N);

/* Native ternary matmul with accumulator */
void fq_matmul_ternary_acc(const float* A, fq_tensor* B, float* C, int M, int K, int N);


/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

/* Get dtype name string */
const char* fq_dtype_name(fq_dtype dtype);

/* Get bytes per element (1 for int8, 0.5 for int4) */
float fq_dtype_bytes(fq_dtype dtype);

/* Calculate quantization parameters from data */
void fq_compute_scale(const float* data, int64_t n, fq_mode mode,
                      float* scale, int32_t* zero_point);

/* Calculate memory usage */
size_t fq_memory_usage(fq_tensor* qt);

/* Print quantized tensor info */
void fq_print(fq_tensor* qt);

/* Compute quantization error (MSE) */
float fq_quantization_error(const float* original, fq_tensor* quantized);

#ifdef __cplusplus
}
#endif

#endif /* FOUNDRY_QUANTIZATION_H */
