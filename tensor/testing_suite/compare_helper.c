/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Comparison helper for Foundry vs PyTorch testing.
 *
 * This program reads input tensors from binary files, runs a Foundry operation,
 * and writes the output to a binary file for comparison with PyTorch.
 *
 * Usage:
 *   ./compare_helper <operation> <output_file> <input_file1> [input_file2] [--param value]
 *
 * Example:
 *   ./compare_helper mish output.bin input.bin
 *   ./compare_helper matmul output.bin a.bin b.bin
 *   ./compare_helper elu output.bin input.bin --alpha 1.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tensor/tensor.h"

/* Binary file format:
 *   ndim: int32 (4 bytes)
 *   dtype: int32 (4 bytes) - 0=f32, 1=f64, 2=i32, 3=i64
 *   reserved: int64 (8 bytes)
 *   shape: ndim * int64
 *   data: numel * sizeof(dtype)
 */

static tm_tensor* read_tensor_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return NULL;
    }

    /* Read header */
    int32_t ndim, dtype_code;
    int64_t reserved;

    if (fread(&ndim, sizeof(int32_t), 1, f) != 1) goto error;
    if (fread(&dtype_code, sizeof(int32_t), 1, f) != 1) goto error;
    if (fread(&reserved, sizeof(int64_t), 1, f) != 1) goto error;

    /* Read shape */
    int64_t shape[TM_MAX_DIMS];
    int64_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (fread(&shape[i], sizeof(int64_t), 1, f) != 1) goto error;
        numel *= shape[i];
    }

    /* Determine dtype */
    tm_dtype dtype;
    size_t elem_size;
    switch (dtype_code) {
        case 0: dtype = TM_F32; elem_size = 4; break;
        case 1: dtype = TM_F64; elem_size = 8; break;
        case 2: dtype = TM_I32; elem_size = 4; break;
        case 3: dtype = TM_I64; elem_size = 8; break;
        default: dtype = TM_F32; elem_size = 4; break;
    }

    /* Create tensor and read data */
    tm_tensor* t = tm_empty(shape, ndim, dtype);
    if (!t) goto error;

    if (fread(t->data, elem_size, numel, f) != (size_t)numel) {
        tm_free(t);
        goto error;
    }

    fclose(f);
    return t;

error:
    fclose(f);
    fprintf(stderr, "Error: Failed to read tensor from %s\n", filename);
    return NULL;
}

static int write_tensor_file(const char* filename, tm_tensor* t) {
    if (!t) return -1;

    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return -1;
    }

    /* Write header */
    int32_t ndim = t->ndim;
    int32_t dtype_code;
    switch (t->dtype) {
        case TM_F32: dtype_code = 0; break;
        case TM_F64: dtype_code = 1; break;
        case TM_I32: dtype_code = 2; break;
        case TM_I64: dtype_code = 3; break;
        default: dtype_code = 0; break;
    }
    int64_t reserved = 0;

    fwrite(&ndim, sizeof(int32_t), 1, f);
    fwrite(&dtype_code, sizeof(int32_t), 1, f);
    fwrite(&reserved, sizeof(int64_t), 1, f);

    /* Write shape */
    for (int i = 0; i < t->ndim; i++) {
        fwrite(&t->shape[i], sizeof(int64_t), 1, f);
    }

    /* Write data */
    size_t elem_size = tm_dtype_size(t->dtype);
    fwrite(t->data, elem_size, t->numel, f);

    fclose(f);
    return 0;
}

static float get_float_param(int argc, char** argv, const char* name, float default_val) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) {
            return (float)atof(argv[i + 1]);
        }
    }
    return default_val;
}

static int get_int_param(int argc, char** argv, const char* name, int default_val) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) {
            return atoi(argv[i + 1]);
        }
    }
    return default_val;
}

/* Check if argument looks like a file path (not a --param or param value) */
static int is_input_file(int argc, char** argv, int idx) {
    if (idx >= argc) return 0;
    if (argv[idx][0] == '-') return 0;  /* --param */
    /* Check if previous arg was --param (making this a value, not a file) */
    if (idx > 0 && strncmp(argv[idx-1], "--", 2) == 0) return 0;
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <operation> <output_file> <input_file1> [input_file2] [input_file3] [--param value]\n", argv[0]);
        return 1;
    }

    const char* operation = argv[1];
    const char* output_file = argv[2];
    const char* input1 = argv[3];
    const char* input2 = is_input_file(argc, argv, 4) ? argv[4] : NULL;
    const char* input3 = is_input_file(argc, argv, 5) ? argv[5] : NULL;

    /* Read input tensors */
    tm_tensor* t1 = read_tensor_file(input1);
    if (!t1) return 1;

    tm_tensor* t2 = NULL;
    if (input2) {
        t2 = read_tensor_file(input2);
        if (!t2) {
            tm_free(t1);
            return 1;
        }
    }

    tm_tensor* t3 = NULL;
    if (input3) {
        t3 = read_tensor_file(input3);
        if (!t3) {
            tm_free(t1);
            if (t2) tm_free(t2);
            return 1;
        }
    }

    tm_tensor* result = NULL;

    /* =============================================== */
    /* Activation functions                            */
    /* =============================================== */

    if (strcmp(operation, "relu") == 0) {
        result = tm_relu(t1);
    }
    else if (strcmp(operation, "sigmoid") == 0) {
        result = tm_sigmoid(t1);
    }
    else if (strcmp(operation, "tanh") == 0) {
        result = tm_tanh(t1);
    }
    else if (strcmp(operation, "gelu") == 0) {
        result = tm_gelu(t1);
    }
    else if (strcmp(operation, "silu") == 0) {
        result = tm_silu(t1);
    }
    else if (strcmp(operation, "mish") == 0) {
        result = tm_mish(t1);
    }
    else if (strcmp(operation, "selu") == 0) {
        result = tm_selu(t1);
    }
    else if (strcmp(operation, "celu") == 0) {
        float alpha = get_float_param(argc, argv, "--alpha", 1.0f);
        result = tm_celu(t1, alpha);
    }
    else if (strcmp(operation, "elu") == 0) {
        float alpha = get_float_param(argc, argv, "--alpha", 1.0f);
        result = tm_elu(t1, alpha);
    }
    else if (strcmp(operation, "relu6") == 0) {
        result = tm_relu6(t1);
    }
    else if (strcmp(operation, "leaky_relu") == 0) {
        float slope = get_float_param(argc, argv, "--slope", 0.01f);
        result = tm_leaky_relu(t1, slope);
    }
    else if (strcmp(operation, "softmax") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        result = tm_softmax(t1, dim);
    }
    else if (strcmp(operation, "log_softmax") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        result = tm_log_softmax(t1, dim);
    }
    else if (strcmp(operation, "hardsigmoid") == 0) {
        result = tm_hardsigmoid(t1);
    }
    else if (strcmp(operation, "hardswish") == 0) {
        result = tm_hardswish(t1);
    }
    else if (strcmp(operation, "hardtanh") == 0) {
        float min_val = get_float_param(argc, argv, "--min", -1.0f);
        float max_val = get_float_param(argc, argv, "--max", 1.0f);
        result = tm_hardtanh(t1, min_val, max_val);
    }
    else if (strcmp(operation, "softsign") == 0) {
        result = tm_softsign(t1);
    }
    else if (strcmp(operation, "tanhshrink") == 0) {
        result = tm_tanhshrink(t1);
    }
    else if (strcmp(operation, "softplus") == 0) {
        float beta = get_float_param(argc, argv, "--beta", 1.0f);
        float threshold = get_float_param(argc, argv, "--threshold", 20.0f);
        result = tm_softplus(t1, beta, threshold);
    }
    else if (strcmp(operation, "glu") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        result = tm_glu(t1, dim);
    }

    /* =============================================== */
    /* Element-wise operations                         */
    /* =============================================== */

    else if (strcmp(operation, "add") == 0) {
        if (!t2) { fprintf(stderr, "Error: add requires 2 inputs\n"); goto cleanup; }
        result = tm_add(t1, t2);
    }
    else if (strcmp(operation, "sub") == 0) {
        if (!t2) { fprintf(stderr, "Error: sub requires 2 inputs\n"); goto cleanup; }
        result = tm_sub(t1, t2);
    }
    else if (strcmp(operation, "mul") == 0) {
        if (!t2) { fprintf(stderr, "Error: mul requires 2 inputs\n"); goto cleanup; }
        result = tm_mul(t1, t2);
    }
    else if (strcmp(operation, "div") == 0) {
        if (!t2) { fprintf(stderr, "Error: div requires 2 inputs\n"); goto cleanup; }
        result = tm_div(t1, t2);
    }
    else if (strcmp(operation, "neg") == 0) {
        result = tm_neg(t1);
    }
    else if (strcmp(operation, "abs") == 0) {
        result = tm_abs(t1);
    }
    else if (strcmp(operation, "exp") == 0) {
        result = tm_exp(t1);
    }
    else if (strcmp(operation, "log") == 0) {
        result = tm_log(t1);
    }
    else if (strcmp(operation, "sqrt") == 0) {
        result = tm_sqrt(t1);
    }
    else if (strcmp(operation, "rsqrt") == 0) {
        result = tm_rsqrt(t1);
    }
    else if (strcmp(operation, "pow") == 0) {
        float exp = get_float_param(argc, argv, "--exp", 2.0f);
        result = tm_pow(t1, exp);
    }
    else if (strcmp(operation, "sin") == 0) {
        result = tm_sin(t1);
    }
    else if (strcmp(operation, "cos") == 0) {
        result = tm_cos(t1);
    }
    else if (strcmp(operation, "tan") == 0) {
        result = tm_tan(t1);
    }
    else if (strcmp(operation, "sinh") == 0) {
        result = tm_sinh(t1);
    }
    else if (strcmp(operation, "cosh") == 0) {
        result = tm_cosh(t1);
    }
    else if (strcmp(operation, "asin") == 0) {
        result = tm_asin(t1);
    }
    else if (strcmp(operation, "acos") == 0) {
        result = tm_acos(t1);
    }
    else if (strcmp(operation, "atan") == 0) {
        result = tm_atan(t1);
    }
    else if (strcmp(operation, "floor") == 0) {
        result = tm_floor(t1);
    }
    else if (strcmp(operation, "ceil") == 0) {
        result = tm_ceil(t1);
    }
    else if (strcmp(operation, "round") == 0) {
        result = tm_round(t1);
    }
    else if (strcmp(operation, "clamp") == 0) {
        float min_val = get_float_param(argc, argv, "--min", -1.0f);
        float max_val = get_float_param(argc, argv, "--max", 1.0f);
        result = tm_clamp(t1, min_val, max_val);
    }
    else if (strcmp(operation, "maximum") == 0) {
        if (!t2) { fprintf(stderr, "Error: maximum requires 2 inputs\n"); goto cleanup; }
        result = tm_maximum(t1, t2);
    }
    else if (strcmp(operation, "minimum") == 0) {
        if (!t2) { fprintf(stderr, "Error: minimum requires 2 inputs\n"); goto cleanup; }
        result = tm_minimum(t1, t2);
    }

    /* =============================================== */
    /* Matrix operations                               */
    /* =============================================== */

    else if (strcmp(operation, "matmul") == 0) {
        if (!t2) { fprintf(stderr, "Error: matmul requires 2 inputs\n"); goto cleanup; }
        result = tm_matmul(t1, t2);
    }
    else if (strcmp(operation, "bmm") == 0) {
        if (!t2) { fprintf(stderr, "Error: bmm requires 2 inputs\n"); goto cleanup; }
        result = tm_bmm(t1, t2);
    }
    else if (strcmp(operation, "mm") == 0) {
        if (!t2) { fprintf(stderr, "Error: mm requires 2 inputs\n"); goto cleanup; }
        result = tm_mm(t1, t2);
    }
    else if (strcmp(operation, "mv") == 0) {
        if (!t2) { fprintf(stderr, "Error: mv requires 2 inputs\n"); goto cleanup; }
        result = tm_mv(t1, t2);
    }
    else if (strcmp(operation, "dot") == 0) {
        if (!t2) { fprintf(stderr, "Error: dot requires 2 inputs\n"); goto cleanup; }
        result = tm_dot(t1, t2);
    }
    else if (strcmp(operation, "outer") == 0) {
        if (!t2) { fprintf(stderr, "Error: outer requires 2 inputs\n"); goto cleanup; }
        result = tm_outer(t1, t2);
    }
    else if (strcmp(operation, "linear") == 0) {
        if (!t2) { fprintf(stderr, "Error: linear requires 2 inputs (x, weight)\n"); goto cleanup; }
        result = tm_linear(t1, t2, t3);  /* t3 is bias, can be NULL */
    }
    else if (strcmp(operation, "transpose") == 0) {
        int dim0 = get_int_param(argc, argv, "--dim0", 0);
        int dim1 = get_int_param(argc, argv, "--dim1", 1);
        result = tm_transpose(t1, dim0, dim1);
    }
    /* =============================================== */
    /* Reductions                                      */
    /* =============================================== */

    else if (strcmp(operation, "sum") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_sum(t1, dim, keepdim);
    }
    else if (strcmp(operation, "mean") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_mean(t1, dim, keepdim);
    }
    else if (strcmp(operation, "max") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_max(t1, dim, keepdim);
    }
    else if (strcmp(operation, "min") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_min(t1, dim, keepdim);
    }
    else if (strcmp(operation, "norm") == 0) {
        float p = get_float_param(argc, argv, "--p", 2.0f);
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_norm(t1, p, dim, keepdim);
    }
    else if (strcmp(operation, "argmax") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_argmax(t1, dim, keepdim);
    }
    else if (strcmp(operation, "argmin") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_argmin(t1, dim, keepdim);
    }
    else if (strcmp(operation, "std") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        int unbiased = get_int_param(argc, argv, "--unbiased", 1);
        result = tm_std(t1, dim, keepdim, unbiased);
    }
    else if (strcmp(operation, "var") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        int unbiased = get_int_param(argc, argv, "--unbiased", 1);
        result = tm_var(t1, dim, keepdim, unbiased);
    }
    else if (strcmp(operation, "prod") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_prod(t1, dim, keepdim);
    }
    else if (strcmp(operation, "logsumexp") == 0) {
        int dim = get_int_param(argc, argv, "--dim", -1);
        int keepdim = get_int_param(argc, argv, "--keepdim", 0);
        result = tm_logsumexp(t1, dim, keepdim);
    }
    else if (strcmp(operation, "cumsum") == 0) {
        int dim = get_int_param(argc, argv, "--dim", 0);
        result = tm_cumsum(t1, dim);
    }
    else if (strcmp(operation, "cumprod") == 0) {
        int dim = get_int_param(argc, argv, "--dim", 0);
        result = tm_cumprod(t1, dim);
    }

    /* =============================================== */
    /* Normalization                                   */
    /* =============================================== */

    else if (strcmp(operation, "layer_norm") == 0) {
        float eps = get_float_param(argc, argv, "--eps", 1e-5f);
        result = tm_layer_norm(t1, NULL, NULL, eps);  /* No weight/bias */
    }
    else if (strcmp(operation, "rms_norm") == 0) {
        float eps = get_float_param(argc, argv, "--eps", 1e-5f);
        result = tm_rms_norm(t1, NULL, eps);  /* No weight */
    }

    /* =============================================== */
    /* Loss functions                                  */
    /* =============================================== */

    else if (strcmp(operation, "mse_loss") == 0) {
        if (!t2) { fprintf(stderr, "Error: mse_loss requires 2 inputs\n"); goto cleanup; }
        result = tm_mse_loss(t1, t2);
    }
    else if (strcmp(operation, "l1_loss") == 0) {
        if (!t2) { fprintf(stderr, "Error: l1_loss requires 2 inputs\n"); goto cleanup; }
        result = tm_l1_loss(t1, t2);
    }
    else if (strcmp(operation, "cross_entropy") == 0) {
        if (!t2) { fprintf(stderr, "Error: cross_entropy requires 2 inputs\n"); goto cleanup; }
        result = tm_cross_entropy(t1, t2);
    }
    else if (strcmp(operation, "smooth_l1_loss") == 0) {
        if (!t2) { fprintf(stderr, "Error: smooth_l1_loss requires 2 inputs\n"); goto cleanup; }
        float beta = get_float_param(argc, argv, "--beta", 1.0f);
        result = tm_smooth_l1_loss(t1, t2, beta);
    }
    else if (strcmp(operation, "nll_loss") == 0) {
        if (!t2) { fprintf(stderr, "Error: nll_loss requires 2 inputs\n"); goto cleanup; }
        result = tm_nll_loss(t1, t2);
    }
    else if (strcmp(operation, "binary_cross_entropy_with_logits") == 0) {
        if (!t2) { fprintf(stderr, "Error: binary_cross_entropy_with_logits requires 2 inputs\n"); goto cleanup; }
        result = tm_binary_cross_entropy_with_logits(t1, t2);
    }
    else if (strcmp(operation, "kl_div") == 0) {
        if (!t2) { fprintf(stderr, "Error: kl_div requires 2 inputs\n"); goto cleanup; }
        result = tm_kl_div(t1, t2);
    }
    else if (strcmp(operation, "huber_loss") == 0) {
        if (!t2) { fprintf(stderr, "Error: huber_loss requires 2 inputs\n"); goto cleanup; }
        float delta = get_float_param(argc, argv, "--delta", 1.0f);
        result = tm_huber_loss(t1, t2, delta);
    }
    else if (strcmp(operation, "cosine_embedding_loss") == 0) {
        if (!t2 || !t3) { fprintf(stderr, "Error: cosine_embedding_loss requires 3 inputs\n"); goto cleanup; }
        float margin = get_float_param(argc, argv, "--margin", 0.0f);
        result = tm_cosine_embedding_loss(t1, t2, t3, margin);
    }
    else if (strcmp(operation, "triplet_margin_loss") == 0) {
        if (!t2 || !t3) { fprintf(stderr, "Error: triplet_margin_loss requires 3 inputs\n"); goto cleanup; }
        float margin = get_float_param(argc, argv, "--margin", 1.0f);
        result = tm_triplet_margin_loss(t1, t2, t3, margin, 2.0f);
    }
    else if (strcmp(operation, "addmm") == 0) {
        if (!t2 || !t3) { fprintf(stderr, "Error: addmm requires 3 inputs\n"); goto cleanup; }
        result = tm_addmm(t1, t2, t3, 1.0, 1.0);
    }
    else if (strcmp(operation, "permute") == 0) {
        /* For now, just support common cases */
        /* TODO: parse dims string */
        int dims[4] = {0, 2, 1, 3};
        result = tm_permute(t1, dims, t1->ndim);
    }

    /* =============================================== */
    /* Unknown operation                               */
    /* =============================================== */

    else {
        fprintf(stderr, "Error: Unknown operation '%s'\n", operation);
        goto cleanup;
    }

    /* Write output */
    if (result) {
        if (write_tensor_file(output_file, result) != 0) {
            fprintf(stderr, "Error: Failed to write output\n");
            tm_free(result);
            goto cleanup;
        }
        tm_free(result);
    } else {
        fprintf(stderr, "Error: Operation produced NULL result\n");
        goto cleanup;
    }

    /* Cleanup */
    tm_free(t1);
    if (t2) tm_free(t2);
    if (t3) tm_free(t3);
    return 0;

cleanup:
    tm_free(t1);
    if (t2) tm_free(t2);
    if (t3) tm_free(t3);
    return 1;
}
