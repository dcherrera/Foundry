/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef USE_BLAS
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#ifdef USE_METAL
  #include "metal/fm_metal.h"
#endif

#ifdef USE_CUDA
  #include "cuda/fc_cuda.h"
#endif

/* ============================================================= */
/* Random Number Generator (xorshift128+)                        */
/* ============================================================= */

static uint64_t rng_state[2] = {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL};

void tm_manual_seed(uint64_t seed) {
    rng_state[0] = seed;
    rng_state[1] = seed ^ 0x853c49e6748fea9bULL;
    if (rng_state[0] == 0) rng_state[0] = 1;
    if (rng_state[1] == 0) rng_state[1] = 1;
}

static uint64_t xorshift128plus(void) {
    uint64_t s1 = rng_state[0];
    uint64_t s0 = rng_state[1];
    rng_state[0] = s0;
    s1 ^= s1 << 23;
    s1 ^= s1 >> 17;
    s1 ^= s0;
    s1 ^= s0 >> 26;
    rng_state[1] = s1;
    return rng_state[0] + rng_state[1];
}

static double rand_uniform(void) {
    return (xorshift128plus() >> 11) * (1.0 / 9007199254740992.0);
}

static double rand_normal(void) {
    /* Box-Muller transform */
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/* ============================================================= */
/* Global State                                                  */
/* ============================================================= */

static bool grad_enabled = true;

bool tm_no_grad_begin(void) {
    bool prev = grad_enabled;
    grad_enabled = false;
    return prev;
}

void tm_no_grad_end(bool prev) {
    grad_enabled = prev;
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

size_t tm_dtype_size(tm_dtype dtype) {
    switch (dtype) {
        case TM_F32: return sizeof(float);
        case TM_F64: return sizeof(double);
        case TM_F16: return 2;
        case TM_BF16: return 2;
        case TM_I32: return sizeof(int32_t);
        case TM_I64: return sizeof(int64_t);
        default: return 0;
    }
}

static const char* dtype_name(tm_dtype dtype) {
    switch (dtype) {
        case TM_F32: return "f32";
        case TM_F64: return "f64";
        case TM_F16: return "f16";
        case TM_BF16: return "bf16";
        case TM_I32: return "i32";
        case TM_I64: return "i64";
        default: return "unknown";
    }
}

static int64_t compute_numel(int64_t* shape, int ndim) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        n *= shape[i];
    }
    return n;
}

static void compute_strides(int64_t* shape, int ndim, int64_t* strides) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

/* Forward declarations for float16 conversion */
static uint16_t float_to_half(float f);
static float half_to_float(uint16_t h);

/* ============================================================= */
/* Broadcasting Utilities                                        */
/* ============================================================= */

/* Check if two shapes can be broadcast together */
static bool can_broadcast(int64_t* a_shape, int a_ndim, int64_t* b_shape, int b_ndim) {
    int max_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
    for (int i = 0; i < max_ndim; i++) {
        int a_idx = a_ndim - 1 - i;
        int b_idx = b_ndim - 1 - i;
        int64_t a_dim = (a_idx >= 0) ? a_shape[a_idx] : 1;
        int64_t b_dim = (b_idx >= 0) ? b_shape[b_idx] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            return false;
        }
    }
    return true;
}

/* Compute broadcast output shape. Returns the number of dimensions. */
static int broadcast_shape(int64_t* a_shape, int a_ndim, int64_t* b_shape, int b_ndim,
                           int64_t* out_shape) {
    int out_ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
    for (int i = 0; i < out_ndim; i++) {
        int a_idx = a_ndim - 1 - i;
        int b_idx = b_ndim - 1 - i;
        int64_t a_dim = (a_idx >= 0) ? a_shape[a_idx] : 1;
        int64_t b_dim = (b_idx >= 0) ? b_shape[b_idx] : 1;
        out_shape[out_ndim - 1 - i] = (a_dim > b_dim) ? a_dim : b_dim;
    }
    return out_ndim;
}

/* Compute broadcast strides (0 for broadcasted dimensions) */
static void broadcast_strides(int64_t* orig_shape, int orig_ndim, int64_t* orig_strides,
                              int out_ndim, int64_t* bc_strides) {
    for (int i = 0; i < out_ndim; i++) {
        int orig_idx = orig_ndim - 1 - (out_ndim - 1 - i);
        if (orig_idx < 0 || orig_shape[orig_idx] == 1) {
            bc_strides[i] = 0;  /* Broadcast: repeat this dimension */
        } else {
            bc_strides[i] = orig_strides[orig_idx];
        }
    }
}

/* Convert flat output index to input index using broadcast strides */
static int64_t broadcast_flat_index(int64_t out_idx, int64_t* out_shape, int out_ndim,
                                    int64_t* bc_strides) {
    int64_t in_idx = 0;
    for (int d = out_ndim - 1; d >= 0; d--) {
        int64_t coord = out_idx % out_shape[d];
        out_idx /= out_shape[d];
        in_idx += coord * bc_strides[d];
    }
    return in_idx;
}

/* ============================================================= */
/* Tensor Creation                                               */
/* ============================================================= */

tm_tensor* tm_empty(int64_t* shape, int ndim, tm_dtype dtype) {
    if (ndim < 0 || ndim > TM_MAX_DIMS) return NULL;

    tm_tensor* t = calloc(1, sizeof(tm_tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->dtype = dtype;
    t->numel = compute_numel(shape, ndim);

    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }
    compute_strides(shape, ndim, t->strides);

    size_t bytes = t->numel * tm_dtype_size(dtype);
    if (bytes > 0) {
        t->data = malloc(bytes);
        if (!t->data) {
            free(t);
            return NULL;
        }
    }

    t->owns_data = true;
    t->refcount = 1;
    t->requires_grad = false;
    t->grad = NULL;
    t->grad_fn = NULL;

    return t;
}

tm_tensor* tm_zeros(int64_t* shape, int ndim, tm_dtype dtype) {
    tm_tensor* t = tm_empty(shape, ndim, dtype);
    if (t && t->data) {
        memset(t->data, 0, t->numel * tm_dtype_size(dtype));
    }
    return t;
}

tm_tensor* tm_ones(int64_t* shape, int ndim, tm_dtype dtype) {
    tm_tensor* t = tm_empty(shape, ndim, dtype);
    if (!t) return NULL;

    if (dtype == TM_F32) {
        float* data = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) data[i] = 1.0f;
    } else if (dtype == TM_F64) {
        double* data = (double*)t->data;
        for (int64_t i = 0; i < t->numel; i++) data[i] = 1.0;
    } else if (dtype == TM_I32) {
        int32_t* data = (int32_t*)t->data;
        for (int64_t i = 0; i < t->numel; i++) data[i] = 1;
    }

    return t;
}

tm_tensor* tm_full(int64_t* shape, int ndim, tm_dtype dtype, double value) {
    tm_tensor* t = tm_empty(shape, ndim, dtype);
    if (!t) return NULL;

    if (dtype == TM_F32) {
        float* data = (float*)t->data;
        float v = (float)value;
        for (int64_t i = 0; i < t->numel; i++) data[i] = v;
    } else if (dtype == TM_F64) {
        double* data = (double*)t->data;
        for (int64_t i = 0; i < t->numel; i++) data[i] = value;
    } else if (dtype == TM_I32) {
        int32_t* data = (int32_t*)t->data;
        int32_t v = (int32_t)value;
        for (int64_t i = 0; i < t->numel; i++) data[i] = v;
    }

    return t;
}

tm_tensor* tm_from_data(void* data, int64_t* shape, int ndim, tm_dtype dtype) {
    if (ndim < 0 || ndim > TM_MAX_DIMS) return NULL;

    tm_tensor* t = calloc(1, sizeof(tm_tensor));
    if (!t) return NULL;

    t->ndim = ndim;
    t->dtype = dtype;
    t->numel = compute_numel(shape, ndim);
    t->data = data;
    t->owns_data = false;  /* We don't own external data */
    t->refcount = 1;

    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
    }
    compute_strides(shape, ndim, t->strides);

    return t;
}

tm_tensor* tm_randn(int64_t* shape, int ndim, tm_dtype dtype) {
    tm_tensor* t = tm_empty(shape, ndim, dtype);
    if (!t) return NULL;

    if (dtype == TM_F32) {
        float* data = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            data[i] = (float)rand_normal();
        }
    } else if (dtype == TM_F64) {
        double* data = (double*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            data[i] = rand_normal();
        }
    }

    return t;
}

tm_tensor* tm_rand(int64_t* shape, int ndim, tm_dtype dtype) {
    tm_tensor* t = tm_empty(shape, ndim, dtype);
    if (!t) return NULL;

    if (dtype == TM_F32) {
        float* data = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            data[i] = (float)rand_uniform();
        }
    } else if (dtype == TM_F64) {
        double* data = (double*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            data[i] = rand_uniform();
        }
    }

    return t;
}

tm_tensor* tm_clone(tm_tensor* t) {
    if (!t) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    memcpy(out->data, t->data, t->numel * tm_dtype_size(t->dtype));
    out->requires_grad = t->requires_grad;

    return out;
}

tm_tensor* tm_arange(double start, double end, double step, tm_dtype dtype) {
    if (step == 0) return NULL;
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) return NULL;

    int64_t count = (int64_t)ceil((end - start) / step);
    if (count <= 0) count = 0;

    int64_t shape[1] = {count};
    tm_tensor* out = tm_empty(shape, 1, dtype);
    if (!out) return NULL;

    if (dtype == TM_F32) {
        float* data = (float*)out->data;
        for (int64_t i = 0; i < count; i++) {
            data[i] = (float)(start + i * step);
        }
    } else if (dtype == TM_F64) {
        double* data = (double*)out->data;
        for (int64_t i = 0; i < count; i++) {
            data[i] = start + i * step;
        }
    } else if (dtype == TM_I32) {
        int32_t* data = (int32_t*)out->data;
        for (int64_t i = 0; i < count; i++) {
            data[i] = (int32_t)(start + i * step);
        }
    } else if (dtype == TM_I64) {
        int64_t* data = (int64_t*)out->data;
        for (int64_t i = 0; i < count; i++) {
            data[i] = (int64_t)(start + i * step);
        }
    }

    return out;
}

tm_tensor* tm_randint(int64_t low, int64_t high, int64_t* shape, int ndim) {
    if (high <= low) return NULL;

    tm_tensor* out = tm_empty(shape, ndim, TM_I32);
    if (!out) return NULL;

    int32_t* data = (int32_t*)out->data;
    int64_t range = high - low;
    for (int64_t i = 0; i < out->numel; i++) {
        data[i] = (int32_t)(low + (xorshift128plus() % range));
    }

    return out;
}

tm_tensor* tm_eye(int64_t n, tm_dtype dtype) {
    int64_t shape[2] = {n, n};
    tm_tensor* out = tm_zeros(shape, 2, dtype);
    if (!out) return NULL;

    if (dtype == TM_F32) {
        float* data = (float*)out->data;
        for (int64_t i = 0; i < n; i++) {
            data[i * n + i] = 1.0f;
        }
    } else if (dtype == TM_F64) {
        double* data = (double*)out->data;
        for (int64_t i = 0; i < n; i++) {
            data[i * n + i] = 1.0;
        }
    }

    return out;
}

tm_tensor* tm_linspace(double start, double end, int64_t steps, tm_dtype dtype) {
    if (steps < 1) return NULL;

    int64_t shape[1] = {steps};
    tm_tensor* out = tm_empty(shape, 1, dtype);
    if (!out) return NULL;

    double step = (steps > 1) ? (end - start) / (steps - 1) : 0;

    if (dtype == TM_F32) {
        float* data = (float*)out->data;
        for (int64_t i = 0; i < steps; i++) {
            data[i] = (float)(start + i * step);
        }
    } else if (dtype == TM_F64) {
        double* data = (double*)out->data;
        for (int64_t i = 0; i < steps; i++) {
            data[i] = start + i * step;
        }
    }

    return out;
}

tm_tensor* tm_full_like(tm_tensor* t, double value) {
    if (!t) return NULL;
    return tm_full(t->shape, t->ndim, t->dtype, value);
}

tm_tensor* tm_empty_like(tm_tensor* t) {
    if (!t) return NULL;
    return tm_empty(t->shape, t->ndim, t->dtype);
}

tm_tensor* tm_rand_like(tm_tensor* t) {
    if (!t) return NULL;
    return tm_rand(t->shape, t->ndim, t->dtype);
}

tm_tensor* tm_randn_like(tm_tensor* t) {
    if (!t) return NULL;
    return tm_randn(t->shape, t->ndim, t->dtype);
}

/* ============================================================= */
/* Memory Management                                             */
/* ============================================================= */

tm_tensor* tm_retain(tm_tensor* t) {
    if (t) t->refcount++;
    return t;
}

void tm_release(tm_tensor* t) {
    if (!t) return;

    t->refcount--;
    if (t->refcount <= 0) {
        if (t->owns_data && t->data) {
            free(t->data);
        }
        if (t->grad) {
            tm_release(t->grad);
        }
        if (t->grad_fn) {
            free(t->grad_fn);
        }
        /* Free sparse tensor data */
        if (t->is_sparse) {
            if (t->indices) free(t->indices);
            if (t->row_ptr) free(t->row_ptr);
            if (t->col_ind) free(t->col_ind);
        }
        free(t);
    }
}

/* ============================================================= */
/* Accessors                                                     */
/* ============================================================= */

float tm_get_f32(tm_tensor* t, int64_t i) {
    if (!t || t->dtype != TM_F32 || i < 0 || i >= t->numel) return 0.0f;
    return ((float*)t->data)[i];
}

double tm_get_f64(tm_tensor* t, int64_t i) {
    if (!t || t->dtype != TM_F64 || i < 0 || i >= t->numel) return 0.0;
    return ((double*)t->data)[i];
}

int32_t tm_get_i32(tm_tensor* t, int64_t i) {
    if (!t || t->dtype != TM_I32 || i < 0 || i >= t->numel) return 0;
    return ((int32_t*)t->data)[i];
}

void tm_set_f32(tm_tensor* t, int64_t i, float val) {
    if (!t || t->dtype != TM_F32 || i < 0 || i >= t->numel) return;
    ((float*)t->data)[i] = val;
}

void tm_set_f64(tm_tensor* t, int64_t i, double val) {
    if (!t || t->dtype != TM_F64 || i < 0 || i >= t->numel) return;
    ((double*)t->data)[i] = val;
}

void tm_set_i32(tm_tensor* t, int64_t i, int32_t val) {
    if (!t || t->dtype != TM_I32 || i < 0 || i >= t->numel) return;
    ((int32_t*)t->data)[i] = val;
}

float* tm_data_f32(tm_tensor* t) {
    if (!t || t->dtype != TM_F32) return NULL;
    return (float*)t->data;
}

double* tm_data_f64(tm_tensor* t) {
    if (!t || t->dtype != TM_F64) return NULL;
    return (double*)t->data;
}

int64_t tm_numel(tm_tensor* t) {
    return t ? t->numel : 0;
}

int tm_ndim(tm_tensor* t) {
    return t ? t->ndim : 0;
}

int64_t tm_size(tm_tensor* t, int dim) {
    if (!t || dim < 0 || dim >= t->ndim) return 0;
    return t->shape[dim];
}

bool tm_is_contiguous(tm_tensor* t) {
    if (!t || t->ndim == 0) return true;

    int64_t expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}

tm_tensor* tm_contiguous(tm_tensor* t) {
    if (!t) return NULL;
    if (tm_is_contiguous(t)) return tm_retain(t);
    return tm_clone(t);
}

/* ============================================================= */
/* Shape Operations                                              */
/* ============================================================= */

tm_tensor* tm_reshape(tm_tensor* t, int64_t* shape, int ndim) {
    if (!t) return NULL;

    /* Verify same number of elements */
    int64_t new_numel = compute_numel(shape, ndim);
    if (new_numel != t->numel) return NULL;

    /* Must be contiguous for reshape */
    if (!tm_is_contiguous(t)) {
        tm_tensor* contig = tm_contiguous(t);
        tm_tensor* result = tm_reshape(contig, shape, ndim);
        tm_release(contig);
        return result;
    }

    /* Create new tensor sharing data */
    tm_tensor* out = calloc(1, sizeof(tm_tensor));
    if (!out) return NULL;

    out->data = t->data;
    out->ndim = ndim;
    out->dtype = t->dtype;
    out->numel = t->numel;
    out->owns_data = false;
    out->refcount = 1;
    out->requires_grad = t->requires_grad;

    for (int i = 0; i < ndim; i++) {
        out->shape[i] = shape[i];
    }
    compute_strides(shape, ndim, out->strides);

    /* Keep original alive */
    tm_retain(t);

    return out;
}

tm_tensor* tm_transpose(tm_tensor* t, int dim0, int dim1) {
    if (!t || dim0 < 0 || dim0 >= t->ndim || dim1 < 0 || dim1 >= t->ndim) {
        return NULL;
    }

    /* For 2D tensors, create a contiguous transposed copy */
    if (t->ndim == 2) {
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];
        int64_t out_shape[2] = {cols, rows};

        tm_tensor* out = tm_empty(out_shape, 2, t->dtype);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* src = (float*)t->data;
            float* dst = (float*)out->data;
            for (int64_t i = 0; i < rows; i++) {
                for (int64_t j = 0; j < cols; j++) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        } else if (t->dtype == TM_F64) {
            double* src = (double*)t->data;
            double* dst = (double*)out->data;
            for (int64_t i = 0; i < rows; i++) {
                for (int64_t j = 0; j < cols; j++) {
                    dst[j * rows + i] = src[i * cols + j];
                }
            }
        }

        out->requires_grad = t->requires_grad;
        return out;
    }

    /* For higher dimensions, swap shape and strides (creates a view) */
    tm_tensor* out = calloc(1, sizeof(tm_tensor));
    if (!out) return NULL;

    out->data = t->data;
    out->ndim = t->ndim;
    out->dtype = t->dtype;
    out->numel = t->numel;
    out->owns_data = false;
    out->refcount = 1;
    out->requires_grad = t->requires_grad;

    for (int i = 0; i < t->ndim; i++) {
        out->shape[i] = t->shape[i];
        out->strides[i] = t->strides[i];
    }

    /* Swap dimensions */
    int64_t tmp = out->shape[dim0];
    out->shape[dim0] = out->shape[dim1];
    out->shape[dim1] = tmp;

    tmp = out->strides[dim0];
    out->strides[dim0] = out->strides[dim1];
    out->strides[dim1] = tmp;

    tm_retain(t);
    return out;
}

tm_tensor* tm_squeeze(tm_tensor* t, int dim) {
    if (!t) return NULL;
    if (dim >= 0 && (dim >= t->ndim || t->shape[dim] != 1)) {
        return tm_retain(t);
    }

    int new_ndim = 0;
    int64_t new_shape[TM_MAX_DIMS];

    for (int i = 0; i < t->ndim; i++) {
        if (dim < 0) {
            /* Squeeze all dims of size 1 */
            if (t->shape[i] != 1) {
                new_shape[new_ndim++] = t->shape[i];
            }
        } else {
            /* Squeeze only specified dim */
            if (i != dim) {
                new_shape[new_ndim++] = t->shape[i];
            }
        }
    }

    return tm_reshape(t, new_shape, new_ndim);
}

tm_tensor* tm_unsqueeze(tm_tensor* t, int dim) {
    if (!t || dim < 0 || dim > t->ndim) return NULL;

    int64_t new_shape[TM_MAX_DIMS];
    int new_ndim = t->ndim + 1;

    for (int i = 0, j = 0; i < new_ndim; i++) {
        if (i == dim) {
            new_shape[i] = 1;
        } else {
            new_shape[i] = t->shape[j++];
        }
    }

    return tm_reshape(t, new_shape, new_ndim);
}

tm_tensor* tm_view(tm_tensor* t, int64_t* shape, int ndim) {
    /* View is the same as reshape for contiguous tensors */
    return tm_reshape(t, shape, ndim);
}

tm_tensor* tm_permute(tm_tensor* t, int* dims, int ndim) {
    if (!t || !dims) return NULL;
    if (ndim != t->ndim) return NULL;

    /* Verify dims is a valid permutation */
    bool used[TM_MAX_DIMS] = {false};
    for (int i = 0; i < ndim; i++) {
        if (dims[i] < 0 || dims[i] >= ndim || used[dims[i]]) return NULL;
        used[dims[i]] = true;
    }

    /* Create output shape */
    int64_t out_shape[TM_MAX_DIMS];
    for (int i = 0; i < ndim; i++) {
        out_shape[i] = t->shape[dims[i]];
    }

    /* Create contiguous output tensor */
    tm_tensor* out = tm_empty(out_shape, ndim, t->dtype);
    if (!out) return NULL;

    /* Copy data with permuted indices */
    float* src = (float*)t->data;
    float* dst = (float*)out->data;

    /* For each output position, compute the source position */
    int64_t coords[TM_MAX_DIMS] = {0};
    for (int64_t i = 0; i < out->numel; i++) {
        /* Compute source linear index from permuted coordinates */
        int64_t src_idx = 0;
        for (int d = 0; d < ndim; d++) {
            /* coords[d] is the output coordinate
               dims[d] is the source dimension it came from */
            src_idx += coords[d] * t->strides[dims[d]];
        }
        dst[i] = src[src_idx];

        /* Increment output coordinates */
        for (int d = ndim - 1; d >= 0; d--) {
            coords[d]++;
            if (coords[d] < out_shape[d]) break;
            coords[d] = 0;
        }
    }

    return out;
}

tm_tensor* tm_flatten(tm_tensor* t, int start_dim, int end_dim) {
    if (!t) return NULL;

    /* Handle negative dimensions */
    if (start_dim < 0) start_dim += t->ndim;
    if (end_dim < 0) end_dim += t->ndim;

    if (start_dim < 0 || start_dim >= t->ndim) return NULL;
    if (end_dim < 0 || end_dim >= t->ndim) return NULL;
    if (start_dim > end_dim) return NULL;

    /* Calculate new shape */
    int64_t new_shape[TM_MAX_DIMS];
    int new_ndim = 0;

    /* Dimensions before start_dim */
    for (int i = 0; i < start_dim; i++) {
        new_shape[new_ndim++] = t->shape[i];
    }

    /* Flatten dimensions from start_dim to end_dim */
    int64_t flat_size = 1;
    for (int i = start_dim; i <= end_dim; i++) {
        flat_size *= t->shape[i];
    }
    new_shape[new_ndim++] = flat_size;

    /* Dimensions after end_dim */
    for (int i = end_dim + 1; i < t->ndim; i++) {
        new_shape[new_ndim++] = t->shape[i];
    }

    return tm_reshape(t, new_shape, new_ndim);
}

tm_tensor* tm_select(tm_tensor* t, int dim, int64_t index) {
    if (!t || dim < 0 || dim >= t->ndim) return NULL;
    if (index < 0 || index >= t->shape[dim]) return NULL;

    /* Build new shape (without the selected dimension) */
    int64_t new_shape[TM_MAX_DIMS];
    int new_ndim = t->ndim - 1;

    for (int i = 0, j = 0; i < t->ndim; i++) {
        if (i != dim) {
            new_shape[j++] = t->shape[i];
        }
    }

    /* For contiguous tensors, we can create a view */
    if (tm_is_contiguous(t)) {
        tm_tensor* out = calloc(1, sizeof(tm_tensor));
        if (!out) return NULL;

        /* Calculate offset to the selected slice */
        int64_t offset = index * t->strides[dim];

        out->data = (char*)t->data + offset * tm_dtype_size(t->dtype);
        out->ndim = new_ndim;
        out->dtype = t->dtype;
        out->owns_data = false;
        out->refcount = 1;
        out->requires_grad = t->requires_grad;

        for (int i = 0, j = 0; i < t->ndim; i++) {
            if (i != dim) {
                out->shape[j] = t->shape[i];
                out->strides[j] = t->strides[i];
                j++;
            }
        }
        out->numel = compute_numel(out->shape, out->ndim);

        tm_retain(t);
        return out;
    }

    /* For non-contiguous, make a copy */
    tm_tensor* out = tm_empty(new_shape, new_ndim, t->dtype);
    if (!out) return NULL;

    /* Copy the selected slice - simplified for now */
    /* TODO: handle strided tensors properly */
    return out;
}

tm_tensor* tm_cat(tm_tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0) return NULL;
    if (!tensors[0]) return NULL;

    tm_tensor* first = tensors[0];
    if (dim < 0) dim += first->ndim;
    if (dim < 0 || dim >= first->ndim) return NULL;

    /* Verify all tensors have same shape except along dim */
    for (int i = 1; i < num_tensors; i++) {
        if (!tensors[i]) return NULL;
        if (tensors[i]->ndim != first->ndim) return NULL;
        if (tensors[i]->dtype != first->dtype) return NULL;
        for (int d = 0; d < first->ndim; d++) {
            if (d != dim && tensors[i]->shape[d] != first->shape[d]) {
                return NULL;
            }
        }
    }

    /* Calculate output shape */
    int64_t out_shape[TM_MAX_DIMS];
    for (int d = 0; d < first->ndim; d++) {
        out_shape[d] = first->shape[d];
    }

    int64_t total_dim_size = 0;
    for (int i = 0; i < num_tensors; i++) {
        total_dim_size += tensors[i]->shape[dim];
    }
    out_shape[dim] = total_dim_size;

    tm_tensor* out = tm_empty(out_shape, first->ndim, first->dtype);
    if (!out) return NULL;

    /* Copy data - handle 2D and 3D cases efficiently */
    if (first->dtype == TM_F32) {
        float* od = (float*)out->data;

        if (first->ndim == 1) {
            /* 1D case: simple concatenation */
            int64_t offset = 0;
            for (int i = 0; i < num_tensors; i++) {
                float* td = (float*)tensors[i]->data;
                memcpy(od + offset, td, tensors[i]->numel * sizeof(float));
                offset += tensors[i]->numel;
            }
        } else if (first->ndim == 2) {
            if (dim == 0) {
                /* Concat along rows */
                int64_t offset = 0;
                for (int i = 0; i < num_tensors; i++) {
                    float* td = (float*)tensors[i]->data;
                    memcpy(od + offset, td, tensors[i]->numel * sizeof(float));
                    offset += tensors[i]->numel;
                }
            } else {
                /* Concat along columns */
                int64_t rows = first->shape[0];
                int64_t out_cols = out_shape[1];
                for (int64_t r = 0; r < rows; r++) {
                    int64_t col_offset = 0;
                    for (int i = 0; i < num_tensors; i++) {
                        float* td = (float*)tensors[i]->data;
                        int64_t cols = tensors[i]->shape[1];
                        memcpy(od + r * out_cols + col_offset,
                               td + r * cols,
                               cols * sizeof(float));
                        col_offset += cols;
                    }
                }
            }
        } else {
            /* General case: element by element */
            int64_t* coords = calloc(first->ndim, sizeof(int64_t));
            int64_t out_idx = 0;

            for (int64_t i = 0; i < out->numel; i++) {
                /* Find which tensor this element belongs to */
                int64_t coord_dim = coords[dim];
                int tensor_idx = 0;
                int64_t local_coord = coord_dim;

                for (tensor_idx = 0; tensor_idx < num_tensors; tensor_idx++) {
                    if (local_coord < tensors[tensor_idx]->shape[dim]) {
                        break;
                    }
                    local_coord -= tensors[tensor_idx]->shape[dim];
                }

                /* Calculate input index */
                float* td = (float*)tensors[tensor_idx]->data;
                int64_t in_idx = 0;
                for (int d = 0; d < first->ndim; d++) {
                    int64_t c = (d == dim) ? local_coord : coords[d];
                    in_idx = in_idx * tensors[tensor_idx]->shape[d] + c;
                }

                od[out_idx++] = td[in_idx];

                /* Increment coordinates */
                for (int d = first->ndim - 1; d >= 0; d--) {
                    coords[d]++;
                    if (coords[d] < out_shape[d]) break;
                    coords[d] = 0;
                }
            }
            free(coords);
        }
    }

    return out;
}

tm_tensor* tm_stack(tm_tensor** tensors, int num_tensors, int dim) {
    if (!tensors || num_tensors <= 0) return NULL;
    if (!tensors[0]) return NULL;

    tm_tensor* first = tensors[0];
    if (dim < 0) dim += first->ndim + 1;
    if (dim < 0 || dim > first->ndim) return NULL;

    /* Verify all tensors have same shape */
    for (int i = 1; i < num_tensors; i++) {
        if (!tensors[i]) return NULL;
        if (tensors[i]->ndim != first->ndim) return NULL;
        if (tensors[i]->dtype != first->dtype) return NULL;
        for (int d = 0; d < first->ndim; d++) {
            if (tensors[i]->shape[d] != first->shape[d]) return NULL;
        }
    }

    /* Calculate output shape (insert new dimension at dim) */
    int64_t out_shape[TM_MAX_DIMS];
    int out_ndim = first->ndim + 1;

    for (int d = 0, j = 0; d < out_ndim; d++) {
        if (d == dim) {
            out_shape[d] = num_tensors;
        } else {
            out_shape[d] = first->shape[j++];
        }
    }

    tm_tensor* out = tm_empty(out_shape, out_ndim, first->dtype);
    if (!out) return NULL;

    /* Copy data */
    if (first->dtype == TM_F32) {
        float* od = (float*)out->data;

        if (dim == 0) {
            /* Stack along first dimension: simple concatenation */
            int64_t offset = 0;
            for (int i = 0; i < num_tensors; i++) {
                float* td = (float*)tensors[i]->data;
                memcpy(od + offset, td, first->numel * sizeof(float));
                offset += first->numel;
            }
        } else {
            /* General case */
            int64_t slice_size = 1;
            for (int d = dim; d < out_ndim; d++) {
                slice_size *= out_shape[d];
            }
            int64_t inner_size = slice_size / num_tensors;
            int64_t outer_size = out->numel / slice_size;

            for (int64_t outer = 0; outer < outer_size; outer++) {
                for (int i = 0; i < num_tensors; i++) {
                    float* td = (float*)tensors[i]->data;
                    memcpy(od + outer * slice_size + i * inner_size,
                           td + outer * inner_size,
                           inner_size * sizeof(float));
                }
            }
        }
    }

    return out;
}

tm_tensor** tm_split(tm_tensor* t, int64_t split_size, int dim, int* num_chunks) {
    if (!t || split_size <= 0 || !num_chunks) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;

    int64_t dim_size = t->shape[dim];
    *num_chunks = (int)((dim_size + split_size - 1) / split_size);

    tm_tensor** chunks = malloc(*num_chunks * sizeof(tm_tensor*));
    if (!chunks) return NULL;

    int64_t offset = 0;
    for (int i = 0; i < *num_chunks; i++) {
        int64_t chunk_size = (offset + split_size <= dim_size) ? split_size : (dim_size - offset);

        /* Build chunk shape */
        int64_t chunk_shape[TM_MAX_DIMS];
        for (int d = 0; d < t->ndim; d++) {
            chunk_shape[d] = (d == dim) ? chunk_size : t->shape[d];
        }

        chunks[i] = tm_empty(chunk_shape, t->ndim, t->dtype);
        if (!chunks[i]) {
            /* Cleanup on failure */
            for (int j = 0; j < i; j++) tm_release(chunks[j]);
            free(chunks);
            return NULL;
        }

        /* Copy data */
        if (t->dtype == TM_F32 && t->ndim == 2) {
            float* od = (float*)chunks[i]->data;
            float* td = (float*)t->data;

            if (dim == 0) {
                memcpy(od, td + offset * t->shape[1],
                       chunk_size * t->shape[1] * sizeof(float));
            } else {
                int64_t rows = t->shape[0];
                for (int64_t r = 0; r < rows; r++) {
                    memcpy(od + r * chunk_size,
                           td + r * t->shape[1] + offset,
                           chunk_size * sizeof(float));
                }
            }
        }

        offset += chunk_size;
    }

    return chunks;
}

tm_tensor** tm_chunk(tm_tensor* t, int num_chunks, int dim, int* actual_chunks) {
    if (!t || num_chunks <= 0 || !actual_chunks) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;

    int64_t dim_size = t->shape[dim];
    int64_t chunk_size = (dim_size + num_chunks - 1) / num_chunks;

    return tm_split(t, chunk_size, dim, actual_chunks);
}

tm_tensor* tm_narrow(tm_tensor* t, int dim, int64_t start, int64_t length) {
    if (!t) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;
    if (start < 0 || start >= t->shape[dim]) return NULL;
    if (length <= 0 || start + length > t->shape[dim]) return NULL;

    /* Build new shape */
    int64_t new_shape[TM_MAX_DIMS];
    for (int i = 0; i < t->ndim; i++) {
        new_shape[i] = (i == dim) ? length : t->shape[i];
    }

    /* For contiguous tensors, create a view */
    if (tm_is_contiguous(t)) {
        tm_tensor* out = calloc(1, sizeof(tm_tensor));
        if (!out) return NULL;

        int64_t offset = start * t->strides[dim];
        out->data = (char*)t->data + offset * tm_dtype_size(t->dtype);
        out->ndim = t->ndim;
        out->dtype = t->dtype;
        out->owns_data = false;
        out->refcount = 1;
        out->requires_grad = t->requires_grad;

        for (int i = 0; i < t->ndim; i++) {
            out->shape[i] = new_shape[i];
            out->strides[i] = t->strides[i];
        }
        out->numel = compute_numel(out->shape, out->ndim);

        tm_retain(t);
        return out;
    }

    /* For non-contiguous, copy data */
    tm_tensor* out = tm_empty(new_shape, t->ndim, t->dtype);
    if (!out) return NULL;

    /* Simple copy for 1D/2D */
    if (t->dtype == TM_F32 && t->ndim <= 2) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;

        if (t->ndim == 1) {
            memcpy(od, td + start, length * sizeof(float));
        } else if (dim == 0) {
            int64_t cols = t->shape[1];
            memcpy(od, td + start * cols, length * cols * sizeof(float));
        } else {
            int64_t rows = t->shape[0];
            int64_t cols = t->shape[1];
            for (int64_t r = 0; r < rows; r++) {
                memcpy(od + r * length, td + r * cols + start, length * sizeof(float));
            }
        }
    }

    return out;
}

tm_tensor* tm_expand(tm_tensor* t, int64_t* shape, int ndim) {
    if (!t || !shape) return NULL;
    if (ndim < t->ndim) return NULL;

    /* Verify expansion is valid */
    for (int i = 0; i < ndim; i++) {
        int t_idx = t->ndim - ndim + i;
        int64_t t_dim = (t_idx >= 0) ? t->shape[t_idx] : 1;
        if (t_dim != 1 && t_dim != shape[i]) {
            return NULL;  /* Can only expand dims of size 1 */
        }
    }

    /* Create output with expanded shape */
    tm_tensor* out = tm_empty(shape, ndim, t->dtype);
    if (!out) return NULL;

    /* Copy data with broadcasting */
    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;

        /* Compute broadcast strides for input */
        int64_t t_bc_strides[TM_MAX_DIMS];
        for (int i = 0; i < ndim; i++) {
            int t_idx = t->ndim - ndim + i;
            if (t_idx < 0 || t->shape[t_idx] == 1) {
                t_bc_strides[i] = 0;
            } else {
                t_bc_strides[i] = t->strides[t_idx];
            }
        }

        for (int64_t i = 0; i < out->numel; i++) {
            int64_t t_idx = 0;
            int64_t idx = i;
            for (int d = ndim - 1; d >= 0; d--) {
                int64_t coord = idx % shape[d];
                idx /= shape[d];
                t_idx += coord * t_bc_strides[d];
            }
            od[i] = td[t_idx];
        }
    }

    return out;
}

tm_tensor* tm_repeat(tm_tensor* t, int64_t* repeats, int ndim) {
    if (!t || !repeats) return NULL;
    if (ndim != t->ndim) return NULL;

    /* Calculate output shape */
    int64_t out_shape[TM_MAX_DIMS];
    for (int i = 0; i < ndim; i++) {
        if (repeats[i] <= 0) return NULL;
        out_shape[i] = t->shape[i] * repeats[i];
    }

    tm_tensor* out = tm_empty(out_shape, ndim, t->dtype);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;

        for (int64_t i = 0; i < out->numel; i++) {
            /* Convert output index to input index */
            int64_t t_idx = 0;
            int64_t idx = i;
            int64_t mult = 1;

            for (int d = ndim - 1; d >= 0; d--) {
                int64_t coord = idx % out_shape[d];
                idx /= out_shape[d];
                int64_t t_coord = coord % t->shape[d];
                t_idx += t_coord * mult;
                mult *= t->shape[d];
            }
            od[i] = td[t_idx];
        }
    }

    return out;
}

tm_tensor* tm_flip(tm_tensor* t, int* dims, int ndim) {
    if (!t || !dims || ndim <= 0) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    /* Mark which dimensions to flip */
    bool flip_dim[TM_MAX_DIMS] = {false};
    for (int i = 0; i < ndim; i++) {
        int d = dims[i];
        if (d < 0) d += t->ndim;
        if (d >= 0 && d < t->ndim) flip_dim[d] = true;
    }

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;

        for (int64_t i = 0; i < out->numel; i++) {
            /* Convert flat index to coordinates */
            int64_t coords[TM_MAX_DIMS];
            int64_t idx = i;
            for (int d = t->ndim - 1; d >= 0; d--) {
                coords[d] = idx % t->shape[d];
                idx /= t->shape[d];
            }

            /* Flip coordinates */
            int64_t t_flat = 0;
            for (int d = 0; d < t->ndim; d++) {
                int64_t coord = flip_dim[d] ? (t->shape[d] - 1 - coords[d]) : coords[d];
                t_flat = t_flat * t->shape[d] + coord;
            }

            od[i] = td[t_flat];
        }
    }

    return out;
}

tm_tensor* tm_roll(tm_tensor* t, int64_t shift, int dim) {
    if (!t) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    int64_t dim_size = t->shape[dim];
    shift = ((shift % dim_size) + dim_size) % dim_size;  /* Handle negative shifts */

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;

        for (int64_t i = 0; i < out->numel; i++) {
            int64_t coords[TM_MAX_DIMS];
            int64_t idx = i;
            for (int d = t->ndim - 1; d >= 0; d--) {
                coords[d] = idx % t->shape[d];
                idx /= t->shape[d];
            }

            /* Roll the coordinate at dim */
            int64_t src_coord = (coords[dim] - shift + dim_size) % dim_size;
            coords[dim] = src_coord;

            /* Calculate source flat index */
            int64_t t_flat = 0;
            for (int d = 0; d < t->ndim; d++) {
                t_flat = t_flat * t->shape[d] + coords[d];
            }

            od[i] = td[t_flat];
        }
    }

    return out;
}

tm_tensor* tm_t(tm_tensor* t) {
    if (!t || t->ndim != 2) return NULL;
    return tm_transpose(t, 0, 1);
}

/* ============================================================= */
/* Indexing Operations                                           */
/* ============================================================= */

tm_tensor* tm_gather(tm_tensor* t, int dim, tm_tensor* index) {
    if (!t || !index) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;
    if (index->ndim != t->ndim) return NULL;

    /* Output has same shape as index */
    tm_tensor* out = tm_empty(index->shape, index->ndim, t->dtype);
    if (!out) return NULL;

    if (t->dtype == TM_F32 && (index->dtype == TM_I32 || index->dtype == TM_I64)) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;

        for (int64_t i = 0; i < out->numel; i++) {
            /* Get coordinates in output/index */
            int64_t coords[TM_MAX_DIMS];
            int64_t idx = i;
            for (int d = out->ndim - 1; d >= 0; d--) {
                coords[d] = idx % out->shape[d];
                idx /= out->shape[d];
            }

            /* Get index value at this position */
            int64_t gather_idx;
            if (index->dtype == TM_I32) {
                gather_idx = ((int32_t*)index->data)[i];
            } else {
                gather_idx = ((int64_t*)index->data)[i];
            }

            /* Replace coordinate at dim with gather_idx */
            coords[dim] = gather_idx;

            /* Calculate input flat index */
            int64_t t_flat = 0;
            for (int d = 0; d < t->ndim; d++) {
                t_flat = t_flat * t->shape[d] + coords[d];
            }

            od[i] = td[t_flat];
        }
    }

    return out;
}

tm_tensor* tm_scatter(tm_tensor* t, int dim, tm_tensor* index, tm_tensor* src) {
    if (!t || !index || !src) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32 && (index->dtype == TM_I32 || index->dtype == TM_I64)) {
        float* od = (float*)out->data;
        float* sd = (float*)src->data;

        for (int64_t i = 0; i < index->numel; i++) {
            int64_t coords[TM_MAX_DIMS];
            int64_t idx = i;
            for (int d = index->ndim - 1; d >= 0; d--) {
                coords[d] = idx % index->shape[d];
                idx /= index->shape[d];
            }

            int64_t scatter_idx;
            if (index->dtype == TM_I32) {
                scatter_idx = ((int32_t*)index->data)[i];
            } else {
                scatter_idx = ((int64_t*)index->data)[i];
            }

            coords[dim] = scatter_idx;

            int64_t out_flat = 0;
            for (int d = 0; d < out->ndim; d++) {
                out_flat = out_flat * out->shape[d] + coords[d];
            }

            od[out_flat] = sd[i];
        }
    }

    return out;
}

tm_tensor* tm_index_select(tm_tensor* t, int dim, tm_tensor* index) {
    if (!t || !index) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;
    if (index->ndim != 1) return NULL;

    /* Output shape: replace t->shape[dim] with index->numel */
    int64_t out_shape[TM_MAX_DIMS];
    for (int i = 0; i < t->ndim; i++) {
        out_shape[i] = (i == dim) ? index->numel : t->shape[i];
    }

    tm_tensor* out = tm_empty(out_shape, t->ndim, t->dtype);
    if (!out) return NULL;

    if (t->dtype == TM_F32 && t->ndim == 2) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];

        if (dim == 0) {
            for (int64_t i = 0; i < index->numel; i++) {
                int64_t idx = (index->dtype == TM_I32) ?
                    ((int32_t*)index->data)[i] : ((int64_t*)index->data)[i];
                memcpy(od + i * cols, td + idx * cols, cols * sizeof(float));
            }
        } else {
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t i = 0; i < index->numel; i++) {
                    int64_t idx = (index->dtype == TM_I32) ?
                        ((int32_t*)index->data)[i] : ((int64_t*)index->data)[i];
                    od[r * index->numel + i] = td[r * cols + idx];
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_scatter_add(tm_tensor* t, int dim, tm_tensor* index, tm_tensor* src) {
    if (!t || !index || !src) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32 && (index->dtype == TM_I32 || index->dtype == TM_I64)) {
        float* od = (float*)out->data;
        float* sd = (float*)src->data;

        for (int64_t i = 0; i < index->numel; i++) {
            int64_t coords[TM_MAX_DIMS];
            int64_t idx = i;
            for (int d = index->ndim - 1; d >= 0; d--) {
                coords[d] = idx % index->shape[d];
                idx /= index->shape[d];
            }

            int64_t scatter_idx;
            if (index->dtype == TM_I32) {
                scatter_idx = ((int32_t*)index->data)[i];
            } else {
                scatter_idx = ((int64_t*)index->data)[i];
            }

            coords[dim] = scatter_idx;

            int64_t out_flat = 0;
            for (int d = 0; d < out->ndim; d++) {
                out_flat = out_flat * out->shape[d] + coords[d];
            }

            od[out_flat] += sd[i];  /* Add instead of replace */
        }
    }

    return out;
}

tm_tensor* tm_masked_select(tm_tensor* t, tm_tensor* mask) {
    if (!t || !mask) return NULL;
    if (t->numel != mask->numel) return NULL;

    /* First count how many elements are selected */
    int64_t count = 0;
    if (mask->dtype == TM_I32) {
        int32_t* md = (int32_t*)mask->data;
        for (int64_t i = 0; i < mask->numel; i++) {
            if (md[i]) count++;
        }
    } else if (mask->dtype == TM_F32) {
        float* md = (float*)mask->data;
        for (int64_t i = 0; i < mask->numel; i++) {
            if (md[i] != 0.0f) count++;
        }
    }

    int64_t out_shape[1] = {count};
    tm_tensor* out = tm_empty(out_shape, 1, t->dtype);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;
        int64_t j = 0;

        if (mask->dtype == TM_I32) {
            int32_t* md = (int32_t*)mask->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (md[i]) od[j++] = td[i];
            }
        } else if (mask->dtype == TM_F32) {
            float* md = (float*)mask->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (md[i] != 0.0f) od[j++] = td[i];
            }
        }
    }

    return out;
}

tm_tensor* tm_nonzero(tm_tensor* t) {
    if (!t) return NULL;

    /* Count non-zero elements */
    int64_t count = 0;
    if (t->dtype == TM_F32) {
        float* td = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            if (td[i] != 0.0f) count++;
        }
    } else if (t->dtype == TM_I32) {
        int32_t* td = (int32_t*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            if (td[i] != 0) count++;
        }
    }

    /* Output shape is [count, ndim] */
    int64_t out_shape[2] = {count, t->ndim};
    tm_tensor* out = tm_empty(out_shape, 2, TM_I64);
    if (!out) return NULL;

    int64_t* od = (int64_t*)out->data;
    int64_t j = 0;

    if (t->dtype == TM_F32) {
        float* td = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            if (td[i] != 0.0f) {
                /* Convert flat index to coordinates */
                int64_t idx = i;
                for (int d = t->ndim - 1; d >= 0; d--) {
                    od[j * t->ndim + d] = idx % t->shape[d];
                    idx /= t->shape[d];
                }
                j++;
            }
        }
    }

    return out;
}

tm_tensor* tm_index_fill(tm_tensor* t, int dim, tm_tensor* index, float value) {
    if (!t || !index) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;
    if (index->ndim != 1) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32 && t->ndim == 2) {
        float* od = (float*)out->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];

        for (int64_t i = 0; i < index->numel; i++) {
            int64_t idx = (index->dtype == TM_I32) ?
                ((int32_t*)index->data)[i] : ((int64_t*)index->data)[i];

            if (dim == 0) {
                for (int64_t c = 0; c < cols; c++) {
                    od[idx * cols + c] = value;
                }
            } else {
                for (int64_t r = 0; r < rows; r++) {
                    od[r * cols + idx] = value;
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_index_copy(tm_tensor* t, int dim, tm_tensor* index, tm_tensor* src) {
    if (!t || !index || !src) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;
    if (index->ndim != 1) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32 && t->ndim == 2) {
        float* od = (float*)out->data;
        float* sd = (float*)src->data;
        int64_t cols = t->shape[1];

        if (dim == 0) {
            for (int64_t i = 0; i < index->numel; i++) {
                int64_t idx = (index->dtype == TM_I32) ?
                    ((int32_t*)index->data)[i] : ((int64_t*)index->data)[i];
                memcpy(od + idx * cols, sd + i * cols, cols * sizeof(float));
            }
        }
    }

    return out;
}

tm_tensor* tm_index_add(tm_tensor* t, int dim, tm_tensor* index, tm_tensor* src) {
    if (!t || !index || !src) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;
    if (index->ndim != 1) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32 && t->ndim == 2) {
        float* od = (float*)out->data;
        float* sd = (float*)src->data;
        int64_t cols = t->shape[1];

        if (dim == 0) {
            for (int64_t i = 0; i < index->numel; i++) {
                int64_t idx = (index->dtype == TM_I32) ?
                    ((int32_t*)index->data)[i] : ((int64_t*)index->data)[i];
                for (int64_t c = 0; c < cols; c++) {
                    od[idx * cols + c] += sd[i * cols + c];
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_take(tm_tensor* t, tm_tensor* indices) {
    if (!t || !indices) return NULL;

    tm_tensor* out = tm_empty(indices->shape, indices->ndim, t->dtype);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        float* td = (float*)t->data;

        if (indices->dtype == TM_I32) {
            int32_t* id = (int32_t*)indices->data;
            for (int64_t i = 0; i < indices->numel; i++) {
                od[i] = td[id[i]];
            }
        } else if (indices->dtype == TM_I64) {
            int64_t* id = (int64_t*)indices->data;
            for (int64_t i = 0; i < indices->numel; i++) {
                od[i] = td[id[i]];
            }
        }
    }

    return out;
}

/* ============================================================= */
/* Autograd Helpers                                              */
/* ============================================================= */

static tm_grad_fn* create_grad_fn(void (*backward)(tm_tensor*)) {
    tm_grad_fn* gf = calloc(1, sizeof(tm_grad_fn));
    if (gf) {
        gf->backward = backward;
        gf->num_inputs = 0;
    }
    return gf;
}

static bool should_track_grad(tm_tensor* a, tm_tensor* b) {
    if (!grad_enabled) return false;
    if (a && a->requires_grad) return true;
    if (b && b->requires_grad) return true;
    return false;
}

void tm_set_requires_grad(tm_tensor* t, bool requires) {
    if (t) t->requires_grad = requires;
}

tm_tensor* tm_detach(tm_tensor* t) {
    if (!t) return NULL;
    tm_tensor* out = tm_clone(t);
    if (out) {
        out->requires_grad = false;
        out->grad_fn = NULL;
    }
    return out;
}

/* ============================================================= */
/* Element-wise Operations                                       */
/* ============================================================= */

/* Backward for add with broadcasting: grad_a = sum of grad_out over broadcast dims */
static void backward_add(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];
    tm_tensor* b = gf->inputs[1];

    /* Compute broadcast strides for a and b relative to output */
    int64_t a_bc_strides[TM_MAX_DIMS];
    int64_t b_bc_strides[TM_MAX_DIMS];
    broadcast_strides(a->shape, a->ndim, a->strides, out->ndim, a_bc_strides);
    broadcast_strides(b->shape, b->ndim, b->strides, out->ndim, b_bc_strides);

    float* og = (float*)out->grad->data;

    if (a->requires_grad) {
        if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);
        float* ag = (float*)a->grad->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out->shape, out->ndim, a_bc_strides);
            ag[a_idx] += og[i];
        }
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = (float*)b->grad->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t b_idx = broadcast_flat_index(i, out->shape, out->ndim, b_bc_strides);
            bg[b_idx] += og[i];
        }
    }
}

tm_tensor* tm_add(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;

    /* Check if shapes are compatible (same or broadcastable) */
    bool same_shape = (a->numel == b->numel && a->ndim == b->ndim);
    if (same_shape) {
        for (int i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i]) { same_shape = false; break; }
        }
    }

    if (same_shape) {
        /* Fast path: same shape */
        tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
        if (!out) return NULL;

        if (a->dtype == TM_F32) {
            float* ad = (float*)a->data;
            float* bd = (float*)b->data;
            float* od = (float*)out->data;
            for (int64_t i = 0; i < a->numel; i++) {
                od[i] = ad[i] + bd[i];
            }
        }

        if (should_track_grad(a, b)) {
            out->requires_grad = true;
            out->grad_fn = create_grad_fn(backward_add);
            out->grad_fn->inputs[0] = tm_retain(a);
            out->grad_fn->inputs[1] = tm_retain(b);
            out->grad_fn->num_inputs = 2;
        }
        return out;
    }

    /* Broadcasting path */
    if (!can_broadcast(a->shape, a->ndim, b->shape, b->ndim)) return NULL;

    int64_t out_shape[TM_MAX_DIMS];
    int out_ndim = broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape);

    tm_tensor* out = tm_empty(out_shape, out_ndim, a->dtype);
    if (!out) return NULL;

    /* Compute broadcast strides */
    int64_t a_bc_strides[TM_MAX_DIMS];
    int64_t b_bc_strides[TM_MAX_DIMS];
    broadcast_strides(a->shape, a->ndim, a->strides, out_ndim, a_bc_strides);
    broadcast_strides(b->shape, b->ndim, b->strides, out_ndim, b_bc_strides);

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out_shape, out_ndim, a_bc_strides);
            int64_t b_idx = broadcast_flat_index(i, out_shape, out_ndim, b_bc_strides);
            od[i] = ad[a_idx] + bd[b_idx];
        }
    }

    if (should_track_grad(a, b)) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_add);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->inputs[1] = tm_retain(b);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* Backward for sub with broadcasting: grad_a = grad_out, grad_b = -grad_out */
static void backward_sub(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];
    tm_tensor* b = gf->inputs[1];

    int64_t a_bc_strides[TM_MAX_DIMS];
    int64_t b_bc_strides[TM_MAX_DIMS];
    broadcast_strides(a->shape, a->ndim, a->strides, out->ndim, a_bc_strides);
    broadcast_strides(b->shape, b->ndim, b->strides, out->ndim, b_bc_strides);

    float* og = (float*)out->grad->data;

    if (a->requires_grad) {
        if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);
        float* ag = (float*)a->grad->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out->shape, out->ndim, a_bc_strides);
            ag[a_idx] += og[i];
        }
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = (float*)b->grad->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t b_idx = broadcast_flat_index(i, out->shape, out->ndim, b_bc_strides);
            bg[b_idx] -= og[i];
        }
    }
}

tm_tensor* tm_sub(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;

    bool same_shape = (a->numel == b->numel && a->ndim == b->ndim);
    if (same_shape) {
        for (int i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i]) { same_shape = false; break; }
        }
    }

    if (same_shape) {
        tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
        if (!out) return NULL;

        if (a->dtype == TM_F32) {
            float* ad = (float*)a->data;
            float* bd = (float*)b->data;
            float* od = (float*)out->data;
            for (int64_t i = 0; i < a->numel; i++) {
                od[i] = ad[i] - bd[i];
            }
        }

        if (should_track_grad(a, b)) {
            out->requires_grad = true;
            out->grad_fn = create_grad_fn(backward_sub);
            out->grad_fn->inputs[0] = tm_retain(a);
            out->grad_fn->inputs[1] = tm_retain(b);
            out->grad_fn->num_inputs = 2;
        }
        return out;
    }

    if (!can_broadcast(a->shape, a->ndim, b->shape, b->ndim)) return NULL;

    int64_t out_shape[TM_MAX_DIMS];
    int out_ndim = broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape);

    tm_tensor* out = tm_empty(out_shape, out_ndim, a->dtype);
    if (!out) return NULL;

    int64_t a_bc_strides[TM_MAX_DIMS];
    int64_t b_bc_strides[TM_MAX_DIMS];
    broadcast_strides(a->shape, a->ndim, a->strides, out_ndim, a_bc_strides);
    broadcast_strides(b->shape, b->ndim, b->strides, out_ndim, b_bc_strides);

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out_shape, out_ndim, a_bc_strides);
            int64_t b_idx = broadcast_flat_index(i, out_shape, out_ndim, b_bc_strides);
            od[i] = ad[a_idx] - bd[b_idx];
        }
    }

    if (should_track_grad(a, b)) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_sub);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->inputs[1] = tm_retain(b);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* Backward for mul with broadcasting: grad_a = grad_out * b, grad_b = grad_out * a */
static void backward_mul(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];
    tm_tensor* b = gf->inputs[1];

    int64_t a_bc_strides[TM_MAX_DIMS];
    int64_t b_bc_strides[TM_MAX_DIMS];
    broadcast_strides(a->shape, a->ndim, a->strides, out->ndim, a_bc_strides);
    broadcast_strides(b->shape, b->ndim, b->strides, out->ndim, b_bc_strides);

    float* og = (float*)out->grad->data;
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;

    if (a->requires_grad) {
        if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);
        float* ag = (float*)a->grad->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out->shape, out->ndim, a_bc_strides);
            int64_t b_idx = broadcast_flat_index(i, out->shape, out->ndim, b_bc_strides);
            ag[a_idx] += og[i] * bd[b_idx];
        }
    }
    if (b->requires_grad) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = (float*)b->grad->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out->shape, out->ndim, a_bc_strides);
            int64_t b_idx = broadcast_flat_index(i, out->shape, out->ndim, b_bc_strides);
            bg[b_idx] += og[i] * ad[a_idx];
        }
    }
}

tm_tensor* tm_mul(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;

    bool same_shape = (a->numel == b->numel && a->ndim == b->ndim);
    if (same_shape) {
        for (int i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i]) { same_shape = false; break; }
        }
    }

    if (same_shape) {
        tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
        if (!out) return NULL;

        if (a->dtype == TM_F32) {
            float* ad = (float*)a->data;
            float* bd = (float*)b->data;
            float* od = (float*)out->data;
            for (int64_t i = 0; i < a->numel; i++) {
                od[i] = ad[i] * bd[i];
            }
        }

        if (should_track_grad(a, b)) {
            out->requires_grad = true;
            out->grad_fn = create_grad_fn(backward_mul);
            out->grad_fn->inputs[0] = tm_retain(a);
            out->grad_fn->inputs[1] = tm_retain(b);
            out->grad_fn->num_inputs = 2;
        }
        return out;
    }

    if (!can_broadcast(a->shape, a->ndim, b->shape, b->ndim)) return NULL;

    int64_t out_shape[TM_MAX_DIMS];
    int out_ndim = broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape);

    tm_tensor* out = tm_empty(out_shape, out_ndim, a->dtype);
    if (!out) return NULL;

    int64_t a_bc_strides[TM_MAX_DIMS];
    int64_t b_bc_strides[TM_MAX_DIMS];
    broadcast_strides(a->shape, a->ndim, a->strides, out_ndim, a_bc_strides);
    broadcast_strides(b->shape, b->ndim, b->strides, out_ndim, b_bc_strides);

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out_shape, out_ndim, a_bc_strides);
            int64_t b_idx = broadcast_flat_index(i, out_shape, out_ndim, b_bc_strides);
            od[i] = ad[a_idx] * bd[b_idx];
        }
    }

    if (should_track_grad(a, b)) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_mul);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->inputs[1] = tm_retain(b);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* Backward for div with broadcasting: d/da (a/b) = 1/b, d/db (a/b) = -a/b^2 */
static void backward_div(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];
    tm_tensor* b = gf->inputs[1];

    if (!out->grad) return;

    int64_t a_bc_strides[TM_MAX_DIMS];
    int64_t b_bc_strides[TM_MAX_DIMS];
    broadcast_strides(a->shape, a->ndim, a->strides, out->ndim, a_bc_strides);
    broadcast_strides(b->shape, b->ndim, b->strides, out->ndim, b_bc_strides);

    float* og = tm_data_f32(out->grad);
    float* ad = tm_data_f32(a);
    float* bd = tm_data_f32(b);

    if (a->requires_grad) {
        if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);
        float* ag = tm_data_f32(a->grad);
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out->shape, out->ndim, a_bc_strides);
            int64_t b_idx = broadcast_flat_index(i, out->shape, out->ndim, b_bc_strides);
            ag[a_idx] += og[i] / bd[b_idx];
        }
    }

    if (b->requires_grad) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = tm_data_f32(b->grad);
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out->shape, out->ndim, a_bc_strides);
            int64_t b_idx = broadcast_flat_index(i, out->shape, out->ndim, b_bc_strides);
            bg[b_idx] += -og[i] * ad[a_idx] / (bd[b_idx] * bd[b_idx]);
        }
    }
}

tm_tensor* tm_div(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;

    bool same_shape = (a->numel == b->numel && a->ndim == b->ndim);
    if (same_shape) {
        for (int i = 0; i < a->ndim; i++) {
            if (a->shape[i] != b->shape[i]) { same_shape = false; break; }
        }
    }

    if (same_shape) {
        tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
        if (!out) return NULL;

        if (a->dtype == TM_F32) {
            float* ad = (float*)a->data;
            float* bd = (float*)b->data;
            float* od = (float*)out->data;
            for (int64_t i = 0; i < a->numel; i++) {
                od[i] = ad[i] / bd[i];
            }
        }

        if (a->requires_grad || b->requires_grad) {
            out->requires_grad = true;
            out->grad_fn = create_grad_fn(backward_div);
            out->grad_fn->inputs[0] = tm_retain(a);
            out->grad_fn->inputs[1] = tm_retain(b);
            out->grad_fn->num_inputs = 2;
        }
        return out;
    }

    if (!can_broadcast(a->shape, a->ndim, b->shape, b->ndim)) return NULL;

    int64_t out_shape[TM_MAX_DIMS];
    int out_ndim = broadcast_shape(a->shape, a->ndim, b->shape, b->ndim, out_shape);

    tm_tensor* out = tm_empty(out_shape, out_ndim, a->dtype);
    if (!out) return NULL;

    int64_t a_bc_strides[TM_MAX_DIMS];
    int64_t b_bc_strides[TM_MAX_DIMS];
    broadcast_strides(a->shape, a->ndim, a->strides, out_ndim, a_bc_strides);
    broadcast_strides(b->shape, b->ndim, b->strides, out_ndim, b_bc_strides);

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < out->numel; i++) {
            int64_t a_idx = broadcast_flat_index(i, out_shape, out_ndim, a_bc_strides);
            int64_t b_idx = broadcast_flat_index(i, out_shape, out_ndim, b_bc_strides);
            od[i] = ad[a_idx] / bd[b_idx];
        }
    }

    if (a->requires_grad || b->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_div);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->inputs[1] = tm_retain(b);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

tm_tensor* tm_add_scalar(tm_tensor* a, double s) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        float sf = (float)s;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = ad[i] + sf;
        }
    }

    return out;
}

tm_tensor* tm_mul_scalar(tm_tensor* a, double s) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        float sf = (float)s;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = ad[i] * sf;
        }
    }

    return out;
}

tm_tensor* tm_sub_scalar(tm_tensor* a, double s) {
    return tm_add_scalar(a, -s);
}

tm_tensor* tm_div_scalar(tm_tensor* a, double s) {
    if (s == 0) return NULL;
    return tm_mul_scalar(a, 1.0 / s);
}

tm_tensor* tm_neg(tm_tensor* a) {
    return tm_mul_scalar(a, -1.0);
}

tm_tensor* tm_abs(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = fabsf(ad[i]);
        }
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = fabs(ad[i]);
        }
    }

    return out;
}

/* Backward for exp: d/dx exp(x) = exp(x) = output */
static void backward_exp(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];

    if (!a->requires_grad || !out->grad) return;

    if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);

    float* ag = tm_data_f32(a->grad);
    float* og = tm_data_f32(out->grad);
    float* od = tm_data_f32(out);  /* exp(x) = output */

    for (int64_t i = 0; i < a->numel; i++) {
        ag[i] += og[i] * od[i];
    }
}

tm_tensor* tm_exp(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = expf(ad[i]);
        }
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_exp);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->num_inputs = 1;
    }

    return out;
}

/* Backward for log: d/dx log(x) = 1/x */
static void backward_log(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];

    if (!a->requires_grad || !out->grad) return;

    if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);

    float* ag = tm_data_f32(a->grad);
    float* og = tm_data_f32(out->grad);
    float* ad = tm_data_f32(a);

    for (int64_t i = 0; i < a->numel; i++) {
        ag[i] += og[i] / ad[i];
    }
}

tm_tensor* tm_log(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = logf(ad[i]);
        }
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_log);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->num_inputs = 1;
    }

    return out;
}

/* Backward for sqrt: d/dx sqrt(x) = 1/(2*sqrt(x)) = 1/(2*output) */
static void backward_sqrt(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];

    if (!a->requires_grad || !out->grad) return;

    if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);

    float* ag = tm_data_f32(a->grad);
    float* og = tm_data_f32(out->grad);
    float* od = tm_data_f32(out);

    for (int64_t i = 0; i < a->numel; i++) {
        ag[i] += og[i] / (2.0f * od[i]);
    }
}

tm_tensor* tm_sqrt(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = sqrtf(ad[i]);
        }
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_sqrt);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->num_inputs = 1;
    }

    return out;
}

/* Backward for tanh: d/dx tanh(x) = 1 - tanh(x)^2 = 1 - output^2 */
static void backward_tanh(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];

    if (!a->requires_grad || !out->grad) return;

    if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);

    float* ag = tm_data_f32(a->grad);
    float* og = tm_data_f32(out->grad);
    float* od = tm_data_f32(out);

    for (int64_t i = 0; i < a->numel; i++) {
        ag[i] += og[i] * (1.0f - od[i] * od[i]);
    }
}

tm_tensor* tm_tanh(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = tanhf(ad[i]);
        }
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_tanh);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->num_inputs = 1;
    }

    return out;
}

/* Backward for relu: d/dx relu(x) = 1 if x > 0 else 0 */
static void backward_relu(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];

    if (!a->requires_grad || !out->grad) return;

    if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);

    float* ag = tm_data_f32(a->grad);
    float* og = tm_data_f32(out->grad);
    float* ad = tm_data_f32(a);

    for (int64_t i = 0; i < a->numel; i++) {
        ag[i] += ad[i] > 0 ? og[i] : 0;
    }
}

tm_tensor* tm_relu(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = ad[i] > 0 ? ad[i] : 0;
        }
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_relu);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->num_inputs = 1;
    }

    return out;
}

/* Backward for sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = out * (1 - out) */
static void backward_sigmoid(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];

    if (!a->requires_grad || !out->grad) return;

    if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);

    float* ag = tm_data_f32(a->grad);
    float* og = tm_data_f32(out->grad);
    float* od = tm_data_f32(out);

    for (int64_t i = 0; i < a->numel; i++) {
        ag[i] += og[i] * od[i] * (1.0f - od[i]);
    }
}

tm_tensor* tm_sigmoid(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = 1.0f / (1.0f + expf(-ad[i]));
        }
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_sigmoid);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->num_inputs = 1;
    }

    return out;
}

/* GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
#define GELU_CONST 0.7978845608028654f  /* sqrt(2/pi) */
#define GELU_COEF  0.044715f

static void backward_gelu(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];

    if (!a->requires_grad || !out->grad) return;

    if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);

    float* ag = tm_data_f32(a->grad);
    float* og = tm_data_f32(out->grad);
    float* ad = tm_data_f32(a);

    for (int64_t i = 0; i < a->numel; i++) {
        float x = ad[i];
        float x3 = x * x * x;
        float inner = GELU_CONST * (x + GELU_COEF * x3);
        float tanh_inner = tanhf(inner);
        float sech2 = 1.0f - tanh_inner * tanh_inner;
        float gelu_grad = 0.5f * (1.0f + tanh_inner) +
                          0.5f * x * sech2 * GELU_CONST * (1.0f + 3.0f * GELU_COEF * x * x);
        ag[i] += og[i] * gelu_grad;
    }
}

tm_tensor* tm_gelu(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            float x = ad[i];
            float inner = GELU_CONST * (x + GELU_COEF * x * x * x);
            od[i] = 0.5f * x * (1.0f + tanhf(inner));
        }
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_gelu);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->num_inputs = 1;
    }

    return out;
}

/* SiLU (Swish): x * sigmoid(x) - used in SwiGLU (Llama, etc.) */
static void backward_silu(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];

    if (!a->requires_grad || !out->grad) return;

    if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);

    float* ag = tm_data_f32(a->grad);
    float* og = tm_data_f32(out->grad);
    float* ad = tm_data_f32(a);

    for (int64_t i = 0; i < a->numel; i++) {
        float x = ad[i];
        float sig = 1.0f / (1.0f + expf(-x));
        /* d/dx (x * sigmoid(x)) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
         *                       = sigmoid(x) * (1 + x * (1 - sigmoid(x))) */
        float grad = sig * (1.0f + x * (1.0f - sig));
        ag[i] += og[i] * grad;
    }
}

tm_tensor* tm_silu(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
#ifdef USE_METAL
        if (fm_available() && a->numel >= FM_ELEM_THRESHOLD) {
            fm_buffer *bx = fm_buffer_from_ptr(ad, a->numel * sizeof(float));
            fm_buffer *bo = fm_buffer_from_ptr(od, a->numel * sizeof(float));
            fm_silu(bx, bo, (int)a->numel);
            float *gpu_out = (float*)fm_buffer_data(bo);
            if (gpu_out != od) memcpy(od, gpu_out, a->numel * sizeof(float));
            fm_buffer_free(bx);
            fm_buffer_free(bo);
        } else
#endif
#ifdef USE_CUDA
        if (fc_available() && a->numel >= FC_ELEM_THRESHOLD) {
            fc_buffer *bx = fc_buffer_from_ptr(ad, a->numel * sizeof(float));
            fc_buffer *bo = fc_buffer_alloc(a->numel * sizeof(float));
            fc_silu(bx, bo, (int)a->numel);
            fc_sync();
            fc_buffer_to_cpu(bo, od);
            fc_buffer_free(bx);
            fc_buffer_free(bo);
        } else
#endif
        for (int64_t i = 0; i < a->numel; i++) {
            float x = ad[i];
            od[i] = x / (1.0f + expf(-x));  /* x * sigmoid(x) */
        }
    }

    if (a->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_silu);
        out->grad_fn->inputs[0] = tm_retain(a);
        out->grad_fn->num_inputs = 1;
    }

    return out;
}

tm_tensor* tm_sin(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = sinf(ad[i]);
        }
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = sin(ad[i]);
        }
    }

    return out;
}

tm_tensor* tm_cos(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = cosf(ad[i]);
        }
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = cos(ad[i]);
        }
    }

    return out;
}

tm_tensor* tm_pow(tm_tensor* a, double exponent) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        float exp_f = (float)exponent;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = powf(ad[i], exp_f);
        }
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = pow(ad[i], exponent);
        }
    }

    return out;
}

tm_tensor* tm_rsqrt(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = 1.0f / sqrtf(ad[i]);
        }
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = 1.0 / sqrt(ad[i]);
        }
    }

    return out;
}

tm_tensor* tm_clamp(tm_tensor* a, double min_val, double max_val) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        float minf = (float)min_val;
        float maxf = (float)max_val;
        for (int64_t i = 0; i < a->numel; i++) {
            float v = ad[i];
            if (v < minf) v = minf;
            if (v > maxf) v = maxf;
            od[i] = v;
        }
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            double v = ad[i];
            if (v < min_val) v = min_val;
            if (v > max_val) v = max_val;
            od[i] = v;
        }
    }

    return out;
}

tm_tensor* tm_where(tm_tensor* condition, tm_tensor* a, tm_tensor* b) {
    if (!condition || !a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;

    /* For simplicity, require same shapes (broadcasting can be added later) */
    if (a->numel != b->numel || a->numel != condition->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;

        /* Condition can be bool (stored as int) or float (nonzero = true) */
        if (condition->dtype == TM_I32) {
            int32_t* cd = (int32_t*)condition->data;
            for (int64_t i = 0; i < a->numel; i++) {
                od[i] = cd[i] ? ad[i] : bd[i];
            }
        } else if (condition->dtype == TM_F32) {
            float* cd = (float*)condition->data;
            for (int64_t i = 0; i < a->numel; i++) {
                od[i] = (cd[i] != 0.0f) ? ad[i] : bd[i];
            }
        }
    }

    return out;
}

tm_tensor* tm_masked_fill(tm_tensor* t, tm_tensor* mask, float value) {
    if (!t || !mask) return NULL;
    if (t->numel != mask->numel) return NULL;  /* Require same size for simplicity */

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;

        if (mask->dtype == TM_I32) {
            int32_t* md = (int32_t*)mask->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (md[i]) od[i] = value;
            }
        } else if (mask->dtype == TM_F32) {
            float* md = (float*)mask->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (md[i] != 0.0f) od[i] = value;
            }
        }
    }

    return out;
}

tm_tensor* tm_maximum(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->numel != b->numel) return NULL;  /* Same shape for simplicity */

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] > bd[i]) ? ad[i] : bd[i];
        }
    }

    return out;
}

tm_tensor* tm_minimum(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] < bd[i]) ? ad[i] : bd[i];
        }
    }

    return out;
}

/* Comparison operations - return TM_I32 tensors (0 or 1) */
tm_tensor* tm_eq(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] == bd[i]) ? 1 : 0;
        }
    } else if (a->dtype == TM_I32) {
        int32_t* ad = (int32_t*)a->data;
        int32_t* bd = (int32_t*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] == bd[i]) ? 1 : 0;
        }
    }

    return out;
}

tm_tensor* tm_ne(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] != bd[i]) ? 1 : 0;
        }
    }

    return out;
}

tm_tensor* tm_lt(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] < bd[i]) ? 1 : 0;
        }
    }

    return out;
}

tm_tensor* tm_le(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] <= bd[i]) ? 1 : 0;
        }
    }

    return out;
}

tm_tensor* tm_gt(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] > bd[i]) ? 1 : 0;
        }
    }

    return out;
}

tm_tensor* tm_ge(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] >= bd[i]) ? 1 : 0;
        }
    }

    return out;
}

/* NaN/Inf classification by bit pattern, read straight from memory. The library builds
 * with -ffast-math: GCC folds isnan()/isinf()/isfinite() to constants under
 * -ffinite-math-only, and clang marks float values and parameters as never inf/NaN, so
 * even a bit test on a float VALUE can be folded. Loading the bits through a pointer
 * cannot be. */
static int tm_f32_class(const float *v) {
    uint32_t b; memcpy(&b, v, sizeof(b));
    if ((b & 0x7F800000u) != 0x7F800000u) return 0;  /* finite */
    return (b & 0x007FFFFFu) ? 2 : 1;                /* 2 = NaN, 1 = Inf */
}

static int tm_f64_class(const double *v) {
    uint64_t b; memcpy(&b, v, sizeof(b));
    if ((b & 0x7FF0000000000000ull) != 0x7FF0000000000000ull) return 0;
    return (b & 0x000FFFFFFFFFFFFFull) ? 2 : 1;
}

tm_tensor* tm_isnan(tm_tensor* t) {
    if (!t) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (t->dtype == TM_F32) {
        float* td = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = tm_f32_class(&td[i]) == 2;
        }
    } else if (t->dtype == TM_F64) {
        double* td = (double*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = tm_f64_class(&td[i]) == 2;
        }
    }

    return out;
}

tm_tensor* tm_isinf(tm_tensor* t) {
    if (!t) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (t->dtype == TM_F32) {
        float* td = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = tm_f32_class(&td[i]) == 1;
        }
    } else if (t->dtype == TM_F64) {
        double* td = (double*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = tm_f64_class(&td[i]) == 1;
        }
    }

    return out;
}

tm_tensor* tm_leaky_relu(tm_tensor* a, float negative_slope) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] > 0) ? ad[i] : negative_slope * ad[i];
        }
    }

    return out;
}

tm_tensor* tm_elu(tm_tensor* a, float alpha) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] > 0) ? ad[i] : alpha * (expf(ad[i]) - 1.0f);
        }
    }

    return out;
}

tm_tensor* tm_floor(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = floorf(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = floor(ad[i]);
    }
    return out;
}

tm_tensor* tm_ceil(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = ceilf(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = ceil(ad[i]);
    }
    return out;
}

tm_tensor* tm_round(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = roundf(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = round(ad[i]);
    }
    return out;
}

tm_tensor* tm_trunc(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = truncf(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = trunc(ad[i]);
    }
    return out;
}

tm_tensor* tm_sign(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] > 0) ? 1.0f : ((ad[i] < 0) ? -1.0f : 0.0f);
        }
    }
    return out;
}

tm_tensor* tm_reciprocal(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = 1.0f / ad[i];
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = 1.0 / ad[i];
    }
    return out;
}

tm_tensor* tm_log2(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = log2f(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = log2(ad[i]);
    }
    return out;
}

tm_tensor* tm_log10(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = log10f(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = log10(ad[i]);
    }
    return out;
}

tm_tensor* tm_exp2(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = exp2f(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = exp2(ad[i]);
    }
    return out;
}

tm_tensor* tm_expm1(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = expm1f(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = expm1(ad[i]);
    }
    return out;
}

tm_tensor* tm_log1p(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = log1pf(ad[i]);
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* od = (double*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = log1p(ad[i]);
    }
    return out;
}

tm_tensor* tm_tan(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = tanf(ad[i]);
    }
    return out;
}

tm_tensor* tm_asin(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = asinf(ad[i]);
    }
    return out;
}

tm_tensor* tm_acos(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = acosf(ad[i]);
    }
    return out;
}

tm_tensor* tm_atan(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = atanf(ad[i]);
    }
    return out;
}

tm_tensor* tm_sinh(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = sinhf(ad[i]);
    }
    return out;
}

tm_tensor* tm_cosh(tm_tensor* a) {
    if (!a) return NULL;
    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* od = (float*)out->data;
        for (int64_t i = 0; i < a->numel; i++) od[i] = coshf(ad[i]);
    }
    return out;
}

tm_tensor* tm_addcmul(tm_tensor* t, tm_tensor* a, tm_tensor* b, double value) {
    if (!t || !a || !b) return NULL;
    if (t->numel != a->numel || a->numel != b->numel) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float val = (float)value;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] += val * ad[i] * bd[i];
        }
    }
    return out;
}

tm_tensor* tm_addcdiv(tm_tensor* t, tm_tensor* a, tm_tensor* b, double value) {
    if (!t || !a || !b) return NULL;
    if (t->numel != a->numel || a->numel != b->numel) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float val = (float)value;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] += val * ad[i] / bd[i];
        }
    }
    return out;
}

tm_tensor* tm_lerp(tm_tensor* a, tm_tensor* b, double weight) {
    if (!a || !b) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;
        float w = (float)weight;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = ad[i] + w * (bd[i] - ad[i]);
        }
    }
    return out;
}

tm_tensor* tm_logical_and(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_I32 && b->dtype == TM_I32) {
        int32_t* ad = (int32_t*)a->data;
        int32_t* bd = (int32_t*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] && bd[i]) ? 1 : 0;
        }
    } else if (a->dtype == TM_F32 && b->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] != 0 && bd[i] != 0) ? 1 : 0;
        }
    }
    return out;
}

tm_tensor* tm_logical_or(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_I32 && b->dtype == TM_I32) {
        int32_t* ad = (int32_t*)a->data;
        int32_t* bd = (int32_t*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] || bd[i]) ? 1 : 0;
        }
    } else if (a->dtype == TM_F32 && b->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] != 0 || bd[i] != 0) ? 1 : 0;
        }
    }
    return out;
}

tm_tensor* tm_logical_not(tm_tensor* a) {
    if (!a) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_I32) {
        int32_t* ad = (int32_t*)a->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = ad[i] ? 0 : 1;
        }
    } else if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = (ad[i] == 0) ? 1 : 0;
        }
    }
    return out;
}

tm_tensor* tm_logical_xor(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->numel != b->numel) return NULL;

    tm_tensor* out = tm_empty(a->shape, a->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (a->dtype == TM_I32 && b->dtype == TM_I32) {
        int32_t* ad = (int32_t*)a->data;
        int32_t* bd = (int32_t*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            od[i] = ((ad[i] != 0) != (bd[i] != 0)) ? 1 : 0;
        }
    }
    return out;
}

tm_tensor* tm_isfinite(tm_tensor* t) {
    if (!t) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, TM_I32);
    if (!out) return NULL;

    int32_t* od = (int32_t*)out->data;

    if (t->dtype == TM_F32) {
        float* td = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = tm_f32_class(&td[i]) == 0;
        }
    } else if (t->dtype == TM_F64) {
        double* td = (double*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = tm_f64_class(&td[i]) == 0;
        }
    }
    return out;
}

bool tm_equal(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return false;
    if (a->numel != b->numel) return false;
    if (a->dtype != b->dtype) return false;
    if (a->ndim != b->ndim) return false;

    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            if (ad[i] != bd[i]) return false;
        }
    } else if (a->dtype == TM_I32) {
        int32_t* ad = (int32_t*)a->data;
        int32_t* bd = (int32_t*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            if (ad[i] != bd[i]) return false;
        }
    }
    return true;
}

bool tm_allclose(tm_tensor* a, tm_tensor* b, double rtol, double atol) {
    if (!a || !b) return false;
    if (a->numel != b->numel) return false;
    if (a->dtype != b->dtype) return false;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            float diff = fabsf(ad[i] - bd[i]);
            float tol = (float)(atol + rtol * fabsf(bd[i]));
            if (diff > tol) return false;
        }
    } else if (a->dtype == TM_F64) {
        double* ad = (double*)a->data;
        double* bd = (double*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            double diff = fabs(ad[i] - bd[i]);
            double tol = atol + rtol * fabs(bd[i]);
            if (diff > tol) return false;
        }
    }
    return true;
}

/* ============================================================= */
/* Matrix Operations                                             */
/* ============================================================= */

/* Backward for matmul: grad_a = grad_out @ b.T, grad_b = a.T @ grad_out */
static void backward_matmul(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* a = gf->inputs[0];
    tm_tensor* b = gf->inputs[1];

    /* For 2D: a is [M, K], b is [K, N], out is [M, N] */
    if (a->ndim != 2 || b->ndim != 2) return;

    int64_t M = a->shape[0];
    int64_t K = a->shape[1];
    int64_t N = b->shape[1];

    float* og = (float*)out->grad->data;

    if (a->requires_grad) {
        if (!a->grad) a->grad = tm_zeros(a->shape, a->ndim, a->dtype);
        float* ag = (float*)a->grad->data;
        float* bd = (float*)b->data;

        /* grad_a = grad_out @ b.T: [M, N] @ [N, K] = [M, K] */
        for (int64_t m = 0; m < M; m++) {
            for (int64_t k = 0; k < K; k++) {
                float sum = 0;
                for (int64_t n = 0; n < N; n++) {
                    sum += og[m * N + n] * bd[k * N + n];
                }
                ag[m * K + k] += sum;
            }
        }
    }

    if (b->requires_grad) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = (float*)b->grad->data;
        float* ad = (float*)a->data;

        /* grad_b = a.T @ grad_out: [K, M] @ [M, N] = [K, N] */
        for (int64_t k = 0; k < K; k++) {
            for (int64_t n = 0; n < N; n++) {
                float sum = 0;
                for (int64_t m = 0; m < M; m++) {
                    sum += ad[m * K + k] * og[m * N + n];
                }
                bg[k * N + n] += sum;
            }
        }
    }
}

tm_tensor* tm_matmul(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->dtype != b->dtype) return NULL;

    /* Handle 2D case: [M, K] @ [K, N] -> [M, N] */
    if (a->ndim == 2 && b->ndim == 2) {
        if (a->shape[1] != b->shape[0]) return NULL;

        int64_t M = a->shape[0];
        int64_t K = a->shape[1];
        int64_t N = b->shape[1];

        int64_t out_shape[2] = {M, N};
        tm_tensor* out = tm_zeros(out_shape, 2, a->dtype);
        if (!out) return NULL;

        if (a->dtype == TM_F32) {
            float* ad = (float*)a->data;
            float* bd = (float*)b->data;
            float* od = (float*)out->data;

#ifdef USE_METAL
            if (fm_available() && M * K * N >= FM_GEMM_THRESHOLD) {
                fm_buffer *ba = fm_buffer_from_ptr(ad, M * K * sizeof(float));
                fm_buffer *bb = fm_buffer_from_ptr(bd, K * N * sizeof(float));
                fm_buffer *bo = fm_buffer_from_ptr(od, M * N * sizeof(float));
                fm_sgemm(ba, bb, bo, (int)M, (int)N, (int)K, 1.0f, 0.0f);
                /* Copy result back if buffer was a copy (non-page-aligned) */
                float *gpu_out = (float*)fm_buffer_data(bo);
                if (gpu_out != od) memcpy(od, gpu_out, M * N * sizeof(float));
                fm_buffer_free(ba);
                fm_buffer_free(bb);
                fm_buffer_free(bo);
            } else
#endif
#ifdef USE_CUDA
            if (fc_available() && M * K * N >= FC_GEMM_THRESHOLD) {
                fc_buffer *ba = fc_buffer_from_ptr(ad, M * K * sizeof(float));
                fc_buffer *bb = fc_buffer_from_ptr(bd, K * N * sizeof(float));
                fc_buffer *bo = fc_buffer_alloc(M * N * sizeof(float));
                fc_sgemm(ba, bb, bo, (int)M, (int)N, (int)K, 1.0f, 0.0f);
                fc_sync();
                fc_buffer_to_cpu(bo, od);
                fc_buffer_free(ba);
                fc_buffer_free(bb);
                fc_buffer_free(bo);
            } else
#endif
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)M, (int)N, (int)K,
                        1.0f, ad, (int)K, bd, (int)N, 0.0f, od, (int)N);
#else
            for (int64_t m = 0; m < M; m++) {
                for (int64_t n = 0; n < N; n++) {
                    float sum = 0;
                    for (int64_t k = 0; k < K; k++) {
                        sum += ad[m * K + k] * bd[k * N + n];
                    }
                    od[m * N + n] = sum;
                }
            }
#endif
        }

        if (should_track_grad(a, b)) {
            out->requires_grad = true;
            out->grad_fn = create_grad_fn(backward_matmul);
            out->grad_fn->inputs[0] = tm_retain(a);
            out->grad_fn->inputs[1] = tm_retain(b);
            out->grad_fn->num_inputs = 2;
        }

        return out;
    }

    /* Handle 1D @ 2D: [K] @ [K, N] -> [N] */
    if (a->ndim == 1 && b->ndim == 2) {
        int64_t new_shape[2] = {1, a->shape[0]};
        tm_tensor* a2d = tm_reshape(a, new_shape, 2);
        tm_tensor* out2d = tm_matmul(a2d, b);
        tm_release(a2d);
        if (!out2d) return NULL;

        int64_t out_shape[1] = {out2d->shape[1]};
        tm_tensor* out = tm_reshape(out2d, out_shape, 1);
        tm_release(out2d);
        return out;
    }

    return NULL;
}

tm_tensor* tm_bmm(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (a->ndim != 3 || b->ndim != 3) return NULL;
    if (a->dtype != b->dtype) return NULL;
    if (a->shape[0] != b->shape[0]) return NULL;  /* Batch size */
    if (a->shape[2] != b->shape[1]) return NULL;  /* K dimension */

    int64_t B = a->shape[0];
    int64_t M = a->shape[1];
    int64_t K = a->shape[2];
    int64_t N = b->shape[2];

    int64_t out_shape[3] = {B, M, N};
    tm_tensor* out = tm_zeros(out_shape, 3, a->dtype);
    if (!out) return NULL;

    if (a->dtype == TM_F32) {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;

        for (int64_t batch = 0; batch < B; batch++) {
            float* a_batch = ad + batch * M * K;
            float* b_batch = bd + batch * K * N;
            float* o_batch = od + batch * M * N;

#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        (int)M, (int)N, (int)K,
                        1.0f, a_batch, (int)K, b_batch, (int)N, 0.0f, o_batch, (int)N);
#else
            for (int64_t m = 0; m < M; m++) {
                for (int64_t n = 0; n < N; n++) {
                    float sum = 0;
                    for (int64_t k = 0; k < K; k++) {
                        sum += a_batch[m * K + k] * b_batch[k * N + n];
                    }
                    o_batch[m * N + n] = sum;
                }
            }
#endif
        }
    }

    return out;
}

/* ============================================================= */
/* Reduction Operations                                          */
/* ============================================================= */

/* Backward for sum: grad is broadcast back to input shape */
static void backward_sum(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* input = gf->inputs[0];

    if (input->requires_grad) {
        if (!input->grad) input->grad = tm_zeros(input->shape, input->ndim, input->dtype);
        float* ig = (float*)input->grad->data;
        float* og = (float*)out->grad->data;
        float grad_val = og[0];  /* Sum reduces to scalar, grad is scalar */

        for (int64_t i = 0; i < input->numel; i++) {
            ig[i] += grad_val;
        }
    }
}

tm_tensor* tm_sum(tm_tensor* t, int dim, bool keepdim) {
    if (!t) return NULL;

    /* Sum all elements */
    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_zeros(out_shape, 1, t->dtype);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            float sum = 0;
            for (int64_t i = 0; i < t->numel; i++) {
                sum += td[i];
            }
            ((float*)out->data)[0] = sum;
        }

        /* Set up autograd */
        if (t->requires_grad && grad_enabled) {
            out->requires_grad = true;
            out->grad_fn = create_grad_fn(backward_sum);
            out->grad_fn->inputs[0] = tm_retain(t);
            out->grad_fn->num_inputs = 1;
        }

        return out;
    }

    /* Sum along dimension */
    if (dim >= t->ndim) return NULL;

    int out_ndim = keepdim ? t->ndim : t->ndim - 1;
    int64_t out_shape[TM_MAX_DIMS];
    int j = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }

    tm_tensor* out = tm_zeros(out_shape, out_ndim, t->dtype);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* td = (float*)t->data;
        float* od = (float*)out->data;

        /* Compute strides for input */
        int64_t strides[TM_MAX_DIMS];
        strides[t->ndim - 1] = 1;
        for (int i = t->ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * t->shape[i + 1];
        }

        /* Iterate over output elements */
        int64_t out_idx[TM_MAX_DIMS] = {0};
        for (int64_t o = 0; o < out->numel; o++) {
            float sum = 0;

            /* Sum over the reduction dimension */
            for (int64_t k = 0; k < t->shape[dim]; k++) {
                /* Map output index to input index */
                int64_t in_linear = 0;
                int oi = 0;
                for (int i = 0; i < t->ndim; i++) {
                    if (i == dim) {
                        in_linear += k * strides[i];
                    } else {
                        in_linear += out_idx[oi] * strides[i];
                        oi++;
                    }
                }
                sum += td[in_linear];
            }
            od[o] = sum;

            /* Increment output index */
            for (int i = out_ndim - 1; i >= 0; i--) {
                out_idx[i]++;
                if (out_idx[i] < out_shape[i]) break;
                out_idx[i] = 0;
            }
        }
    }

    return out;
}

tm_tensor* tm_mean(tm_tensor* t, int dim, bool keepdim) {
    tm_tensor* s = tm_sum(t, dim, keepdim);
    if (!s) return NULL;

    int64_t count = (dim < 0) ? t->numel : t->shape[dim];
    tm_tensor* out = tm_mul_scalar(s, 1.0 / count);
    tm_release(s);
    return out;
}

tm_tensor* tm_max(tm_tensor* t, int dim, bool keepdim) {
    (void)keepdim;
    if (!t) return NULL;

    /* Max of all elements */
    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, t->dtype);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            float max_val = td[0];
            for (int64_t i = 1; i < t->numel; i++) {
                if (td[i] > max_val) max_val = td[i];
            }
            ((float*)out->data)[0] = max_val;
        }

        return out;
    }

    return NULL;  /* TODO: implement per-dimension max */
}

tm_tensor* tm_min(tm_tensor* t, int dim, bool keepdim) {
    (void)keepdim;
    if (!t) return NULL;

    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, t->dtype);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            float min_val = td[0];
            for (int64_t i = 1; i < t->numel; i++) {
                if (td[i] < min_val) min_val = td[i];
            }
            ((float*)out->data)[0] = min_val;
        }

        return out;
    }

    return NULL;
}

tm_tensor* tm_argmax(tm_tensor* t, int dim, bool keepdim) {
    if (!t) return NULL;

    /* Argmax of all elements */
    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, TM_I64);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            float max_val = td[0];
            int64_t max_idx = 0;
            for (int64_t i = 1; i < t->numel; i++) {
                if (td[i] > max_val) {
                    max_val = td[i];
                    max_idx = i;
                }
            }
            ((int64_t*)out->data)[0] = max_idx;
        }

        return out;
    }

    /* Argmax along dimension */
    if (dim >= t->ndim) return NULL;

    int out_ndim = keepdim ? t->ndim : t->ndim - 1;
    int64_t out_shape[TM_MAX_DIMS];
    int j = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }

    tm_tensor* out = tm_zeros(out_shape, out_ndim, TM_I64);
    if (!out) return NULL;

    /* Handle 2D case */
    if (t->ndim == 2 && t->dtype == TM_F32) {
        float* td = (float*)t->data;
        int64_t* od = (int64_t*)out->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];

        if (dim == 0) {
            /* Argmax along rows -> [cols] */
            for (int64_t c = 0; c < cols; c++) {
                float max_val = td[c];
                int64_t max_idx = 0;
                for (int64_t r = 1; r < rows; r++) {
                    if (td[r * cols + c] > max_val) {
                        max_val = td[r * cols + c];
                        max_idx = r;
                    }
                }
                od[c] = max_idx;
            }
        } else {
            /* Argmax along cols -> [rows] */
            for (int64_t r = 0; r < rows; r++) {
                float max_val = td[r * cols];
                int64_t max_idx = 0;
                for (int64_t c = 1; c < cols; c++) {
                    if (td[r * cols + c] > max_val) {
                        max_val = td[r * cols + c];
                        max_idx = c;
                    }
                }
                od[r] = max_idx;
            }
        }
    }

    return out;
}

tm_tensor* tm_argmin(tm_tensor* t, int dim, bool keepdim) {
    (void)keepdim;
    if (!t) return NULL;

    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, TM_I64);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            float min_val = td[0];
            int64_t min_idx = 0;
            for (int64_t i = 1; i < t->numel; i++) {
                if (td[i] < min_val) {
                    min_val = td[i];
                    min_idx = i;
                }
            }
            ((int64_t*)out->data)[0] = min_idx;
        }

        return out;
    }

    /* Similar implementation to argmax */
    return NULL;
}

tm_tensor* tm_logsumexp(tm_tensor* t, int dim, bool keepdim) {
    if (!t) return NULL;

    /* Logsumexp of all elements */
    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, t->dtype);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;

            /* Find max for numerical stability */
            float max_val = td[0];
            for (int64_t i = 1; i < t->numel; i++) {
                if (td[i] > max_val) max_val = td[i];
            }

            /* Compute sum of exp(x - max) */
            float sum = 0.0f;
            for (int64_t i = 0; i < t->numel; i++) {
                sum += expf(td[i] - max_val);
            }

            ((float*)out->data)[0] = max_val + logf(sum);
        }

        return out;
    }

    /* Logsumexp along dimension */
    if (dim >= t->ndim) return NULL;

    int out_ndim = keepdim ? t->ndim : t->ndim - 1;
    int64_t out_shape[TM_MAX_DIMS];
    int j = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }

    tm_tensor* out = tm_empty(out_shape, out_ndim, t->dtype);
    if (!out) return NULL;

    /* Handle 2D case */
    if (t->ndim == 2 && t->dtype == TM_F32) {
        float* td = (float*)t->data;
        float* od = (float*)out->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];

        if (dim == 0) {
            /* Logsumexp along rows -> [cols] */
            for (int64_t c = 0; c < cols; c++) {
                float max_val = td[c];
                for (int64_t r = 1; r < rows; r++) {
                    if (td[r * cols + c] > max_val) max_val = td[r * cols + c];
                }
                float sum = 0.0f;
                for (int64_t r = 0; r < rows; r++) {
                    sum += expf(td[r * cols + c] - max_val);
                }
                od[c] = max_val + logf(sum);
            }
        } else {
            /* Logsumexp along cols -> [rows] */
            for (int64_t r = 0; r < rows; r++) {
                float max_val = td[r * cols];
                for (int64_t c = 1; c < cols; c++) {
                    if (td[r * cols + c] > max_val) max_val = td[r * cols + c];
                }
                float sum = 0.0f;
                for (int64_t c = 0; c < cols; c++) {
                    sum += expf(td[r * cols + c] - max_val);
                }
                od[r] = max_val + logf(sum);
            }
        }
    }

    return out;
}

tm_tensor* tm_cumsum(tm_tensor* t, int dim) {
    if (!t) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;

        if (t->ndim == 1) {
            for (int64_t i = 1; i < t->numel; i++) {
                od[i] += od[i-1];
            }
        } else if (t->ndim == 2) {
            int64_t rows = t->shape[0];
            int64_t cols = t->shape[1];

            if (dim == 0) {
                for (int64_t c = 0; c < cols; c++) {
                    for (int64_t r = 1; r < rows; r++) {
                        od[r * cols + c] += od[(r-1) * cols + c];
                    }
                }
            } else {
                for (int64_t r = 0; r < rows; r++) {
                    for (int64_t c = 1; c < cols; c++) {
                        od[r * cols + c] += od[r * cols + c - 1];
                    }
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_var(tm_tensor* t, int dim, bool keepdim, bool unbiased) {
    if (!t) return NULL;

    /* Variance of all elements */
    if (dim < 0) {
        /* Calculate mean */
        float mean = 0.0f;
        float* td = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            mean += td[i];
        }
        mean /= t->numel;

        /* Calculate variance */
        float var = 0.0f;
        for (int64_t i = 0; i < t->numel; i++) {
            float diff = td[i] - mean;
            var += diff * diff;
        }
        int64_t divisor = unbiased ? (t->numel - 1) : t->numel;
        if (divisor <= 0) divisor = 1;
        var /= divisor;

        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_full(out_shape, 1, t->dtype, var);
        return out;
    }

    /* Variance along dimension */
    if (dim >= t->ndim) return NULL;

    int out_ndim = keepdim ? t->ndim : t->ndim - 1;
    int64_t out_shape[TM_MAX_DIMS];
    int j = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }

    tm_tensor* out = tm_empty(out_shape, out_ndim, t->dtype);
    if (!out) return NULL;

    /* Handle 2D case */
    if (t->ndim == 2 && t->dtype == TM_F32) {
        float* td = (float*)t->data;
        float* od = (float*)out->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];
        int64_t dim_size = t->shape[dim];
        int64_t divisor = unbiased ? (dim_size - 1) : dim_size;
        if (divisor <= 0) divisor = 1;

        if (dim == 0) {
            for (int64_t c = 0; c < cols; c++) {
                float mean = 0.0f;
                for (int64_t r = 0; r < rows; r++) {
                    mean += td[r * cols + c];
                }
                mean /= rows;

                float var = 0.0f;
                for (int64_t r = 0; r < rows; r++) {
                    float diff = td[r * cols + c] - mean;
                    var += diff * diff;
                }
                od[c] = var / divisor;
            }
        } else {
            for (int64_t r = 0; r < rows; r++) {
                float mean = 0.0f;
                for (int64_t c = 0; c < cols; c++) {
                    mean += td[r * cols + c];
                }
                mean /= cols;

                float var = 0.0f;
                for (int64_t c = 0; c < cols; c++) {
                    float diff = td[r * cols + c] - mean;
                    var += diff * diff;
                }
                od[r] = var / divisor;
            }
        }
    }

    return out;
}

tm_tensor* tm_std(tm_tensor* t, int dim, bool keepdim, bool unbiased) {
    tm_tensor* var = tm_var(t, dim, keepdim, unbiased);
    if (!var) return NULL;

    tm_tensor* out = tm_sqrt(var);
    tm_release(var);
    return out;
}

tm_tensor* tm_prod(tm_tensor* t, int dim, bool keepdim) {
    if (!t) return NULL;

    /* Product of all elements */
    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, t->dtype);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            float prod = 1.0f;
            for (int64_t i = 0; i < t->numel; i++) {
                prod *= td[i];
            }
            ((float*)out->data)[0] = prod;
        }

        return out;
    }

    /* Product along dimension */
    if (dim >= t->ndim) return NULL;

    int out_ndim = keepdim ? t->ndim : t->ndim - 1;
    int64_t out_shape[TM_MAX_DIMS];
    int j = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }

    tm_tensor* out = tm_ones(out_shape, out_ndim, t->dtype);
    if (!out) return NULL;

    if (t->ndim == 2 && t->dtype == TM_F32) {
        float* td = (float*)t->data;
        float* od = (float*)out->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];

        if (dim == 0) {
            for (int64_t c = 0; c < cols; c++) {
                float prod = 1.0f;
                for (int64_t r = 0; r < rows; r++) {
                    prod *= td[r * cols + c];
                }
                od[c] = prod;
            }
        } else {
            for (int64_t r = 0; r < rows; r++) {
                float prod = 1.0f;
                for (int64_t c = 0; c < cols; c++) {
                    prod *= td[r * cols + c];
                }
                od[r] = prod;
            }
        }
    }

    return out;
}

tm_tensor* tm_cumprod(tm_tensor* t, int dim) {
    if (!t) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;

        if (t->ndim == 1) {
            for (int64_t i = 1; i < t->numel; i++) {
                od[i] *= od[i-1];
            }
        } else if (t->ndim == 2) {
            int64_t rows = t->shape[0];
            int64_t cols = t->shape[1];

            if (dim == 0) {
                for (int64_t c = 0; c < cols; c++) {
                    for (int64_t r = 1; r < rows; r++) {
                        od[r * cols + c] *= od[(r-1) * cols + c];
                    }
                }
            } else {
                for (int64_t r = 0; r < rows; r++) {
                    for (int64_t c = 1; c < cols; c++) {
                        od[r * cols + c] *= od[r * cols + c - 1];
                    }
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_any(tm_tensor* t, int dim, bool keepdim) {
    if (!t) return NULL;

    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, TM_I32);
        if (!out) return NULL;

        int32_t* od = (int32_t*)out->data;
        od[0] = 0;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (td[i] != 0.0f) { od[0] = 1; break; }
            }
        } else if (t->dtype == TM_I32) {
            int32_t* td = (int32_t*)t->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (td[i]) { od[0] = 1; break; }
            }
        }
        return out;
    }

    /* Along dimension */
    if (dim >= t->ndim) return NULL;

    int out_ndim = keepdim ? t->ndim : t->ndim - 1;
    int64_t out_shape[TM_MAX_DIMS];
    int j = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }

    tm_tensor* out = tm_zeros(out_shape, out_ndim, TM_I32);
    if (!out) return NULL;

    /* Simplified 2D case */
    if (t->ndim == 2 && t->dtype == TM_F32) {
        float* td = (float*)t->data;
        int32_t* od = (int32_t*)out->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];

        if (dim == 0) {
            for (int64_t c = 0; c < cols; c++) {
                for (int64_t r = 0; r < rows; r++) {
                    if (td[r * cols + c] != 0.0f) { od[c] = 1; break; }
                }
            }
        } else {
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < cols; c++) {
                    if (td[r * cols + c] != 0.0f) { od[r] = 1; break; }
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_all(tm_tensor* t, int dim, bool keepdim) {
    if (!t) return NULL;

    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, TM_I32);
        if (!out) return NULL;

        int32_t* od = (int32_t*)out->data;
        od[0] = 1;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (td[i] == 0.0f) { od[0] = 0; break; }
            }
        } else if (t->dtype == TM_I32) {
            int32_t* td = (int32_t*)t->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (!td[i]) { od[0] = 0; break; }
            }
        }
        return out;
    }

    /* Along dimension */
    if (dim >= t->ndim) return NULL;

    int out_ndim = keepdim ? t->ndim : t->ndim - 1;
    int64_t out_shape[TM_MAX_DIMS];
    int j = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }

    tm_tensor* out = tm_ones(out_shape, out_ndim, TM_I32);
    if (!out) return NULL;

    if (t->ndim == 2 && t->dtype == TM_F32) {
        float* td = (float*)t->data;
        int32_t* od = (int32_t*)out->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];

        if (dim == 0) {
            for (int64_t c = 0; c < cols; c++) {
                for (int64_t r = 0; r < rows; r++) {
                    if (td[r * cols + c] == 0.0f) { od[c] = 0; break; }
                }
            }
        } else {
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < cols; c++) {
                    if (td[r * cols + c] == 0.0f) { od[r] = 0; break; }
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_norm(tm_tensor* t, double p, int dim, bool keepdim) {
    if (!t) return NULL;

    if (dim < 0) {
        /* Global norm */
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, t->dtype);
        if (!out) return NULL;

        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            float sum = 0.0f;
            if (p == 2.0) {
                for (int64_t i = 0; i < t->numel; i++) {
                    sum += td[i] * td[i];
                }
                ((float*)out->data)[0] = sqrtf(sum);
            } else if (p == 1.0) {
                for (int64_t i = 0; i < t->numel; i++) {
                    sum += fabsf(td[i]);
                }
                ((float*)out->data)[0] = sum;
            } else {
                for (int64_t i = 0; i < t->numel; i++) {
                    sum += powf(fabsf(td[i]), (float)p);
                }
                ((float*)out->data)[0] = powf(sum, 1.0f/(float)p);
            }
        }
        return out;
    }

    /* Along specific dimension */
    if (dim >= t->ndim) return NULL;

    /* Calculate output shape */
    int64_t out_shape[TM_MAX_DIMS];
    int out_ndim = keepdim ? t->ndim : t->ndim - 1;
    int j = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = t->shape[i];
        }
    }
    if (out_ndim == 0) out_ndim = 1;  /* At least 1D */

    tm_tensor* out = tm_zeros(out_shape, out_ndim, t->dtype);
    if (!out) return NULL;

    float* td = (float*)t->data;
    float* od = (float*)out->data;

    /* Calculate strides for iteration */
    int64_t outer_size = 1, inner_size = 1, dim_size = t->shape[dim];
    for (int i = 0; i < dim; i++) outer_size *= t->shape[i];
    for (int i = dim + 1; i < t->ndim; i++) inner_size *= t->shape[i];

    for (int64_t o = 0; o < outer_size; o++) {
        for (int64_t i = 0; i < inner_size; i++) {
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; d++) {
                int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                float val = td[idx];
                if (p == 2.0) {
                    sum += val * val;
                } else if (p == 1.0) {
                    sum += fabsf(val);
                } else {
                    sum += powf(fabsf(val), (float)p);
                }
            }
            int64_t out_idx = o * inner_size + i;
            if (p == 2.0) {
                od[out_idx] = sqrtf(sum);
            } else if (p == 1.0) {
                od[out_idx] = sum;
            } else {
                od[out_idx] = powf(sum, 1.0f / (float)p);
            }
        }
    }

    return out;
}

tm_tensor* tm_count_nonzero(tm_tensor* t, int dim) {
    if (!t) return NULL;

    if (dim < 0) {
        int64_t out_shape[1] = {1};
        tm_tensor* out = tm_empty(out_shape, 1, TM_I64);
        if (!out) return NULL;

        int64_t count = 0;
        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (td[i] != 0.0f) count++;
            }
        } else if (t->dtype == TM_I32) {
            int32_t* td = (int32_t*)t->data;
            for (int64_t i = 0; i < t->numel; i++) {
                if (td[i] != 0) count++;
            }
        }
        ((int64_t*)out->data)[0] = count;
        return out;
    }

    return NULL;
}

/* ============================================================= */
/* Type Conversion                                               */
/* ============================================================= */

tm_tensor* tm_to(tm_tensor* t, tm_dtype dtype) {
    if (!t) return NULL;
    if (t->dtype == dtype) return tm_clone(t);

    tm_tensor* out = tm_empty(t->shape, t->ndim, dtype);
    if (!out) return NULL;

    /* Handle all conversions */
    if (t->dtype == TM_F32) {
        float* td = (float*)t->data;
        if (dtype == TM_F64) {
            double* od = (double*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (double)td[i];
        } else if (dtype == TM_I32) {
            int32_t* od = (int32_t*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (int32_t)td[i];
        } else if (dtype == TM_I64) {
            int64_t* od = (int64_t*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (int64_t)td[i];
        }
    } else if (t->dtype == TM_F64) {
        double* td = (double*)t->data;
        if (dtype == TM_F32) {
            float* od = (float*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (float)td[i];
        } else if (dtype == TM_I32) {
            int32_t* od = (int32_t*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (int32_t)td[i];
        } else if (dtype == TM_I64) {
            int64_t* od = (int64_t*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (int64_t)td[i];
        }
    } else if (t->dtype == TM_I32) {
        int32_t* td = (int32_t*)t->data;
        if (dtype == TM_F32) {
            float* od = (float*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (float)td[i];
        } else if (dtype == TM_F64) {
            double* od = (double*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (double)td[i];
        } else if (dtype == TM_I64) {
            int64_t* od = (int64_t*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (int64_t)td[i];
        }
    } else if (t->dtype == TM_I64) {
        int64_t* td = (int64_t*)t->data;
        if (dtype == TM_F32) {
            float* od = (float*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (float)td[i];
        } else if (dtype == TM_F64) {
            double* od = (double*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (double)td[i];
        } else if (dtype == TM_I32) {
            int32_t* od = (int32_t*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (int32_t)td[i];
        }
    } else if (t->dtype == TM_F16) {
        uint16_t* td = (uint16_t*)t->data;
        if (dtype == TM_F32) {
            float* od = (float*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = half_to_float(td[i]);
        } else if (dtype == TM_F64) {
            double* od = (double*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (double)half_to_float(td[i]);
        } else if (dtype == TM_I32) {
            int32_t* od = (int32_t*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (int32_t)half_to_float(td[i]);
        } else if (dtype == TM_I64) {
            int64_t* od = (int64_t*)out->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = (int64_t)half_to_float(td[i]);
        }
    }

    /* Handle conversion TO F16 from other types */
    if (dtype == TM_F16) {
        uint16_t* od = (uint16_t*)out->data;
        if (t->dtype == TM_F32) {
            float* td = (float*)t->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = float_to_half(td[i]);
        } else if (t->dtype == TM_F64) {
            double* td = (double*)t->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = float_to_half((float)td[i]);
        } else if (t->dtype == TM_I32) {
            int32_t* td = (int32_t*)t->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = float_to_half((float)td[i]);
        } else if (t->dtype == TM_I64) {
            int64_t* td = (int64_t*)t->data;
            for (int64_t i = 0; i < t->numel; i++) od[i] = float_to_half((float)td[i]);
        }
    }

    return out;
}

tm_tensor* tm_float(tm_tensor* t) {
    return tm_to(t, TM_F32);
}

tm_tensor* tm_double(tm_tensor* t) {
    return tm_to(t, TM_F64);
}

tm_tensor* tm_int(tm_tensor* t) {
    return tm_to(t, TM_I32);
}

tm_tensor* tm_long(tm_tensor* t) {
    return tm_to(t, TM_I64);
}

/* Float16 conversion helpers */
static uint16_t float_to_half(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t x = v.u;
    uint16_t h;

    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    if (exp <= 0) {
        /* Subnormal or zero */
        if (exp < -10) {
            h = sign;  /* Too small, flush to zero */
        } else {
            mant = (mant | 0x800000) >> (1 - exp);
            h = sign | (mant >> 13);
        }
    } else if (exp >= 31) {
        /* Overflow to infinity */
        h = sign | 0x7C00;
    } else {
        h = sign | (exp << 10) | (mant >> 13);
    }
    return h;
}

static float half_to_float(uint16_t h) {
    union { float f; uint32_t u; } v;
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            v.u = sign;  /* Zero */
        } else {
            /* Subnormal */
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x400;
            v.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        v.u = sign | 0x7F800000 | (mant << 13);  /* Inf/NaN */
    } else {
        v.u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return v.f;
}

tm_tensor* tm_half(tm_tensor* t) {
    if (!t) return NULL;
    if (t->dtype == TM_F16) return tm_clone(t);

    tm_tensor* out = tm_empty(t->shape, t->ndim, TM_F16);
    if (!out) return NULL;

    uint16_t* od = (uint16_t*)out->data;

    if (t->dtype == TM_F32) {
        float* td = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = float_to_half(td[i]);
        }
    } else if (t->dtype == TM_F64) {
        double* td = (double*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = float_to_half((float)td[i]);
        }
    } else if (t->dtype == TM_I32) {
        int32_t* td = (int32_t*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = float_to_half((float)td[i]);
        }
    } else if (t->dtype == TM_I64) {
        int64_t* td = (int64_t*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            od[i] = float_to_half((float)td[i]);
        }
    }

    return out;
}

/* ============================================================= */
/* Neural Network Operations                                     */
/* ============================================================= */

tm_tensor* tm_softmax(tm_tensor* t, int dim) {
    if (!t) return NULL;

    /* Handle last dimension by default */
    if (dim < 0) dim = t->ndim - 1;
    if (dim >= t->ndim) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    if (t->dtype == TM_F32 && t->ndim == 2) {
        float* td = (float*)t->data;
        float* od = (float*)out->data;
        int64_t rows = t->shape[0];
        int64_t cols = t->shape[1];

        if (dim == 1) {
#ifdef USE_METAL
            if (fm_available() && rows * cols >= FM_ELEM_THRESHOLD) {
                fm_buffer *bx = fm_buffer_from_ptr(td, rows * cols * sizeof(float));
                fm_buffer *bo = fm_buffer_from_ptr(od, rows * cols * sizeof(float));
                fm_softmax(bx, bo, (int)rows, (int)cols);
                float *gpu_out = (float*)fm_buffer_data(bo);
                if (gpu_out != od) memcpy(od, gpu_out, rows * cols * sizeof(float));
                fm_buffer_free(bx);
                fm_buffer_free(bo);
            } else
#endif
#ifdef USE_CUDA
            if (fc_available() && rows * cols >= FC_ELEM_THRESHOLD) {
                fc_buffer *bx = fc_buffer_from_ptr(td, rows * cols * sizeof(float));
                fc_buffer *bo = fc_buffer_alloc(rows * cols * sizeof(float));
                fc_softmax(bx, bo, (int)rows, (int)cols);
                fc_sync();
                fc_buffer_to_cpu(bo, od);
                fc_buffer_free(bx);
                fc_buffer_free(bo);
            } else
#endif
            for (int64_t r = 0; r < rows; r++) {
                /* Find max for numerical stability */
                float max_val = td[r * cols];
                for (int64_t c = 1; c < cols; c++) {
                    if (td[r * cols + c] > max_val) {
                        max_val = td[r * cols + c];
                    }
                }

                /* Compute exp and sum */
                float sum = 0;
                for (int64_t c = 0; c < cols; c++) {
                    od[r * cols + c] = expf(td[r * cols + c] - max_val);
                    sum += od[r * cols + c];
                }

                /* Normalize */
                for (int64_t c = 0; c < cols; c++) {
                    od[r * cols + c] /= sum;
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_log_softmax(tm_tensor* t, int dim) {
    tm_tensor* sm = tm_softmax(t, dim);
    if (!sm) return NULL;
    tm_tensor* out = tm_log(sm);
    tm_release(sm);
    return out;
}

tm_tensor* tm_layer_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias, float eps) {
    if (!t) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    /* Normalize over last dimension */
    if (t->dtype == TM_F32 && t->ndim >= 1) {
        float* td = (float*)t->data;
        float* od = (float*)out->data;
        float* wd = weight ? (float*)weight->data : NULL;
        float* bd = bias ? (float*)bias->data : NULL;

        int64_t norm_size = t->shape[t->ndim - 1];
        int64_t batch_size = t->numel / norm_size;

        for (int64_t b = 0; b < batch_size; b++) {
            float* in_ptr = td + b * norm_size;
            float* out_ptr = od + b * norm_size;

            /* Compute mean */
            float mean = 0;
            for (int64_t i = 0; i < norm_size; i++) {
                mean += in_ptr[i];
            }
            mean /= norm_size;

            /* Compute variance */
            float var = 0;
            for (int64_t i = 0; i < norm_size; i++) {
                float diff = in_ptr[i] - mean;
                var += diff * diff;
            }
            var /= norm_size;

            /* Normalize */
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int64_t i = 0; i < norm_size; i++) {
                out_ptr[i] = (in_ptr[i] - mean) * inv_std;
                if (wd) out_ptr[i] *= wd[i];
                if (bd) out_ptr[i] += bd[i];
            }
        }
    }

    return out;
}

tm_tensor* tm_embedding(tm_tensor* weight, tm_tensor* indices) {
    if (!weight || !indices) return NULL;
    if (weight->ndim != 2) return NULL;

    int64_t vocab_size = weight->shape[0];
    int64_t embed_dim = weight->shape[1];

    /* Output shape is indices.shape + [embed_dim] */
    int out_ndim = indices->ndim + 1;
    int64_t out_shape[TM_MAX_DIMS];
    for (int i = 0; i < indices->ndim; i++) {
        out_shape[i] = indices->shape[i];
    }
    out_shape[indices->ndim] = embed_dim;

    tm_tensor* out = tm_empty(out_shape, out_ndim, weight->dtype);
    if (!out) return NULL;

    if (weight->dtype == TM_F32 && indices->dtype == TM_I32) {
        float* wd = (float*)weight->data;
        int32_t* id = (int32_t*)indices->data;
        float* od = (float*)out->data;

        for (int64_t i = 0; i < indices->numel; i++) {
            int32_t idx = id[i];
            if (idx >= 0 && idx < vocab_size) {
                memcpy(od + i * embed_dim, wd + idx * embed_dim,
                       embed_dim * sizeof(float));
            }
        }
    }

    return out;
}

tm_tensor* tm_dropout(tm_tensor* t, float p, bool training) {
    if (!t || !training || p <= 0) return tm_retain(t);
    if (p >= 1) return tm_zeros(t->shape, t->ndim, t->dtype);

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    float scale = 1.0f / (1.0f - p);

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        for (int64_t i = 0; i < out->numel; i++) {
            if (rand_uniform() < p) {
                od[i] = 0;
            } else {
                od[i] *= scale;
            }
        }
    }

    return out;
}

tm_tensor* tm_rms_norm(tm_tensor* t, tm_tensor* weight, float eps) {
    if (!t) return NULL;

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    /* Normalize over last dimension */
    if (t->dtype == TM_F32 && t->ndim >= 1) {
        float* td = (float*)t->data;
        float* od = (float*)out->data;
        float* wd = weight ? (float*)weight->data : NULL;

        int64_t norm_size = t->shape[t->ndim - 1];
        int64_t batch_size = t->numel / norm_size;

#ifdef USE_METAL
        if (fm_available() && wd && batch_size == 1 && norm_size >= FM_ELEM_THRESHOLD) {
            fm_buffer *bx = fm_buffer_from_ptr(td, norm_size * sizeof(float));
            fm_buffer *bw = fm_buffer_from_ptr(wd, norm_size * sizeof(float));
            fm_buffer *bo = fm_buffer_from_ptr(od, norm_size * sizeof(float));
            fm_rms_norm(bx, bw, bo, (int)norm_size, eps);
            float *gpu_out = (float*)fm_buffer_data(bo);
            if (gpu_out != od) memcpy(od, gpu_out, norm_size * sizeof(float));
            fm_buffer_free(bx);
            fm_buffer_free(bw);
            fm_buffer_free(bo);
        } else
#endif
#ifdef USE_CUDA
        if (fc_available() && wd && batch_size == 1 && norm_size >= FC_ELEM_THRESHOLD) {
            fc_buffer *bx = fc_buffer_from_ptr(td, norm_size * sizeof(float));
            fc_buffer *bw = fc_buffer_from_ptr(wd, norm_size * sizeof(float));
            fc_buffer *bo = fc_buffer_alloc(norm_size * sizeof(float));
            fc_rms_norm(bx, bw, bo, (int)norm_size, eps);
            fc_sync();
            fc_buffer_to_cpu(bo, od);
            fc_buffer_free(bx);
            fc_buffer_free(bw);
            fc_buffer_free(bo);
        } else
#endif
        for (int64_t b = 0; b < batch_size; b++) {
            float* in_ptr = td + b * norm_size;
            float* out_ptr = od + b * norm_size;

            /* Compute RMS: sqrt(mean(x^2)) */
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < norm_size; i++) {
                sum_sq += in_ptr[i] * in_ptr[i];
            }
            float rms = sqrtf(sum_sq / norm_size + eps);
            float inv_rms = 1.0f / rms;

            /* Normalize and apply weight */
            for (int64_t i = 0; i < norm_size; i++) {
                out_ptr[i] = in_ptr[i] * inv_rms;
                if (wd) out_ptr[i] *= wd[i];
            }
        }
    }

    return out;
}

tm_tensor* tm_linear(tm_tensor* x, tm_tensor* weight, tm_tensor* bias) {
    if (!x || !weight) return NULL;

    /* x: [..., in_features], weight: [out_features, in_features]
     * output: [..., out_features] = x @ weight.T + bias */

    /* Transpose weight */
    tm_tensor* wt = tm_transpose(weight, 0, 1);
    if (!wt) return NULL;

    /* Matmul x @ wt */
    tm_tensor* out = tm_matmul(x, wt);
    tm_release(wt);
    if (!out) return NULL;

    /* Add bias if present */
    if (bias) {
        tm_tensor* result = tm_add(out, bias);
        tm_release(out);
        return result;
    }

    return out;
}

tm_tensor* tm_triu(tm_tensor* t, int diagonal) {
    if (!t || t->ndim < 2) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    int64_t rows = t->shape[t->ndim - 2];
    int64_t cols = t->shape[t->ndim - 1];
    int64_t batch_size = t->numel / (rows * cols);

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        for (int64_t b = 0; b < batch_size; b++) {
            float* batch_ptr = od + b * rows * cols;
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < cols; c++) {
                    if (c < r + diagonal) {
                        batch_ptr[r * cols + c] = 0.0f;
                    }
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_tril(tm_tensor* t, int diagonal) {
    if (!t || t->ndim < 2) return NULL;

    tm_tensor* out = tm_clone(t);
    if (!out) return NULL;

    int64_t rows = t->shape[t->ndim - 2];
    int64_t cols = t->shape[t->ndim - 1];
    int64_t batch_size = t->numel / (rows * cols);

    if (t->dtype == TM_F32) {
        float* od = (float*)out->data;
        for (int64_t b = 0; b < batch_size; b++) {
            float* batch_ptr = od + b * rows * cols;
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t c = 0; c < cols; c++) {
                    if (c > r + diagonal) {
                        batch_ptr[r * cols + c] = 0.0f;
                    }
                }
            }
        }
    }

    return out;
}

/* ============================================================= */
/* Additional Neural Network Operations                          */
/* ============================================================= */

tm_tensor* tm_hardswish(tm_tensor* t) {
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        /* hardswish(x) = x * relu6(x + 3) / 6 */
        float relu6_val = x + 3.0f;
        if (relu6_val < 0.0f) relu6_val = 0.0f;
        if (relu6_val > 6.0f) relu6_val = 6.0f;
        o[i] = x * relu6_val / 6.0f;
    }

    return out;
}

tm_tensor* tm_hardtanh(tm_tensor* t, float min_val, float max_val) {
    if (!t) return NULL;
    return tm_clamp(t, min_val, max_val);
}

tm_tensor* tm_gelu_approximate(tm_tensor* t) {
    /* GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;
    const float sqrt_2_pi = 0.7978845608f;  /* sqrt(2/pi) */

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        float inner = sqrt_2_pi * (x + 0.044715f * x * x * x);
        o[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    return out;
}

tm_tensor* tm_softplus(tm_tensor* t, float beta, float threshold) {
    /* softplus(x) = log(1 + exp(beta * x)) / beta, or x if beta*x > threshold */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        float bx = beta * x;
        if (bx > threshold) {
            o[i] = x;
        } else {
            o[i] = logf(1.0f + expf(bx)) / beta;
        }
    }

    return out;
}

tm_tensor* tm_mish(tm_tensor* t) {
    /* mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))) */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        float sp = logf(1.0f + expf(x));  /* softplus */
        o[i] = x * tanhf(sp);
    }

    return out;
}

tm_tensor* tm_selu(tm_tensor* t) {
    /* selu(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
     * alpha = 1.6732632423543772848170429916717
     * scale = 1.0507009873554804934193349852946 */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        if (x > 0) {
            o[i] = scale * x;
        } else {
            o[i] = scale * alpha * (expf(x) - 1.0f);
        }
    }

    return out;
}

tm_tensor* tm_celu(tm_tensor* t, float alpha) {
    /* celu(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1)) */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        if (x > 0) {
            o[i] = x;
        } else {
            o[i] = alpha * (expf(x / alpha) - 1.0f);
        }
    }

    return out;
}

tm_tensor* tm_relu6(tm_tensor* t) {
    /* relu6(x) = min(max(0,x), 6) */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        if (x < 0.0f) x = 0.0f;
        if (x > 6.0f) x = 6.0f;
        o[i] = x;
    }

    return out;
}

tm_tensor* tm_glu(tm_tensor* t, int dim) {
    /* GLU: split input in half along dim, then a * sigmoid(b) */
    if (!t) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;
    if (t->shape[dim] % 2 != 0) return NULL;  /* Must be even */

    int64_t half = t->shape[dim] / 2;

    /* Create output shape */
    int64_t out_shape[TM_MAX_DIMS];
    for (int i = 0; i < t->ndim; i++) {
        out_shape[i] = t->shape[i];
    }
    out_shape[dim] = half;

    tm_tensor* out = tm_empty(out_shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    /* Calculate strides for manual indexing */
    int64_t outer_size = 1, inner_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= t->shape[i];
    for (int i = dim + 1; i < t->ndim; i++) inner_size *= t->shape[i];

    for (int64_t outer = 0; outer < outer_size; outer++) {
        for (int64_t h = 0; h < half; h++) {
            for (int64_t inner = 0; inner < inner_size; inner++) {
                int64_t a_idx = outer * t->shape[dim] * inner_size + h * inner_size + inner;
                int64_t b_idx = outer * t->shape[dim] * inner_size + (h + half) * inner_size + inner;
                int64_t o_idx = outer * half * inner_size + h * inner_size + inner;

                float a = in[a_idx];
                float b = in[b_idx];
                o[o_idx] = a * (1.0f / (1.0f + expf(-b)));  /* a * sigmoid(b) */
            }
        }
    }

    return out;
}

tm_tensor* tm_softsign(tm_tensor* t) {
    /* softsign(x) = x / (1 + |x|) */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        o[i] = x / (1.0f + fabsf(x));
    }

    return out;
}

tm_tensor* tm_tanhshrink(tm_tensor* t) {
    /* tanhshrink(x) = x - tanh(x) */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        o[i] = x - tanhf(x);
    }

    return out;
}

tm_tensor* tm_hardsigmoid(tm_tensor* t) {
    /* hardsigmoid(x) = 0 if x <= -3, 1 if x >= 3, else x/6 + 0.5 */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        if (x <= -3.0f) {
            o[i] = 0.0f;
        } else if (x >= 3.0f) {
            o[i] = 1.0f;
        } else {
            o[i] = x / 6.0f + 0.5f;
        }
    }

    return out;
}

tm_tensor* tm_threshold(tm_tensor* t, float threshold, float value) {
    /* threshold(x) = x if x > threshold, else value */
    if (!t) return NULL;
    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    for (int64_t i = 0; i < t->numel; i++) {
        float x = in[i];
        o[i] = (x > threshold) ? x : value;
    }

    return out;
}

tm_tensor* tm_pad(tm_tensor* t, int64_t* padding, int pad_len, float value) {
    /* Constant padding. padding is [left, right] for 1D, [left, right, top, bottom] for 2D, etc.
     * Padding is specified from last dimension backwards. */
    if (!t || !padding || pad_len <= 0) return NULL;
    if (pad_len % 2 != 0) return NULL;

    int num_dims_to_pad = pad_len / 2;
    if (num_dims_to_pad > t->ndim) return NULL;

    /* Calculate new shape */
    int64_t new_shape[TM_MAX_DIMS];
    for (int i = 0; i < t->ndim; i++) {
        new_shape[i] = t->shape[i];
    }

    /* Apply padding from last dimension backwards */
    for (int i = 0; i < num_dims_to_pad; i++) {
        int dim = t->ndim - 1 - i;
        int64_t pad_before = padding[2 * i];
        int64_t pad_after = padding[2 * i + 1];
        new_shape[dim] += pad_before + pad_after;
    }

    /* Create output filled with padding value */
    tm_tensor* out = tm_full(new_shape, t->ndim, t->dtype, value);
    if (!out) return NULL;

    /* Copy original data to correct position */
    if (t->ndim == 1) {
        int64_t pad_left = padding[0];
        float* in = (float*)t->data;
        float* o = (float*)out->data;
        for (int64_t i = 0; i < t->shape[0]; i++) {
            o[pad_left + i] = in[i];
        }
    } else if (t->ndim == 2) {
        int64_t pad_left = (pad_len >= 2) ? padding[0] : 0;
        int64_t pad_top = (pad_len >= 4) ? padding[2] : 0;
        float* in = (float*)t->data;
        float* o = (float*)out->data;
        for (int64_t r = 0; r < t->shape[0]; r++) {
            for (int64_t c = 0; c < t->shape[1]; c++) {
                o[(r + pad_top) * new_shape[1] + (c + pad_left)] = in[r * t->shape[1] + c];
            }
        }
    } else if (t->ndim == 3) {
        int64_t pad_left = (pad_len >= 2) ? padding[0] : 0;
        int64_t pad_top = (pad_len >= 4) ? padding[2] : 0;
        int64_t pad_front = (pad_len >= 6) ? padding[4] : 0;
        float* in = (float*)t->data;
        float* o = (float*)out->data;
        for (int64_t d = 0; d < t->shape[0]; d++) {
            for (int64_t r = 0; r < t->shape[1]; r++) {
                for (int64_t c = 0; c < t->shape[2]; c++) {
                    int64_t in_idx = d * t->shape[1] * t->shape[2] + r * t->shape[2] + c;
                    int64_t out_idx = (d + pad_front) * new_shape[1] * new_shape[2] +
                                      (r + pad_top) * new_shape[2] + (c + pad_left);
                    o[out_idx] = in[in_idx];
                }
            }
        }
    } else if (t->ndim == 4) {
        int64_t pad_left = (pad_len >= 2) ? padding[0] : 0;
        int64_t pad_top = (pad_len >= 4) ? padding[2] : 0;
        int64_t pad_front = (pad_len >= 6) ? padding[4] : 0;
        int64_t pad_batch = (pad_len >= 8) ? padding[6] : 0;
        float* in = (float*)t->data;
        float* o = (float*)out->data;
        for (int64_t b = 0; b < t->shape[0]; b++) {
            for (int64_t d = 0; d < t->shape[1]; d++) {
                for (int64_t r = 0; r < t->shape[2]; r++) {
                    for (int64_t c = 0; c < t->shape[3]; c++) {
                        int64_t in_idx = b * t->shape[1] * t->shape[2] * t->shape[3] +
                                         d * t->shape[2] * t->shape[3] +
                                         r * t->shape[3] + c;
                        int64_t out_idx = (b + pad_batch) * new_shape[1] * new_shape[2] * new_shape[3] +
                                          (d + pad_front) * new_shape[2] * new_shape[3] +
                                          (r + pad_top) * new_shape[3] + (c + pad_left);
                        o[out_idx] = in[in_idx];
                    }
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_normalize(tm_tensor* t, float p, int dim, float eps) {
    /* Lp normalization: v = v / max(||v||_p, eps) */
    if (!t) return NULL;
    if (dim < 0) dim += t->ndim;
    if (dim < 0 || dim >= t->ndim) return NULL;

    tm_tensor* norm = tm_norm(t, p, dim, true);
    if (!norm) return NULL;

    /* Clamp norm to eps minimum */
    float* norm_data = (float*)norm->data;
    for (int64_t i = 0; i < norm->numel; i++) {
        if (norm_data[i] < eps) norm_data[i] = eps;
    }

    /* Divide input by norm (with broadcasting) */
    tm_tensor* out = tm_div(t, norm);
    tm_free(norm);

    return out;
}

tm_tensor* tm_interpolate(tm_tensor* t, int64_t* size, int size_len, const char* mode) {
    /* Upsample/interpolate tensor. Currently supports nearest neighbor for 2D. */
    if (!t || !size || size_len <= 0) return NULL;

    /* For now, only support 4D tensors (batch, channels, height, width) */
    if (t->ndim != 4 || size_len != 2) return NULL;

    int64_t batch = t->shape[0];
    int64_t channels = t->shape[1];
    int64_t in_h = t->shape[2];
    int64_t in_w = t->shape[3];
    int64_t out_h = size[0];
    int64_t out_w = size[1];

    int64_t out_shape[4] = {batch, channels, out_h, out_w};
    tm_tensor* out = tm_empty(out_shape, 4, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;

    /* Nearest neighbor interpolation */
    if (!mode || strcmp(mode, "nearest") == 0) {
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t c = 0; c < channels; c++) {
                for (int64_t h = 0; h < out_h; h++) {
                    for (int64_t w = 0; w < out_w; w++) {
                        /* Map output coordinate to input */
                        int64_t src_h = (int64_t)(h * in_h / out_h);
                        int64_t src_w = (int64_t)(w * in_w / out_w);
                        if (src_h >= in_h) src_h = in_h - 1;
                        if (src_w >= in_w) src_w = in_w - 1;

                        int64_t in_idx = b * channels * in_h * in_w +
                                         c * in_h * in_w + src_h * in_w + src_w;
                        int64_t out_idx = b * channels * out_h * out_w +
                                          c * out_h * out_w + h * out_w + w;
                        o[out_idx] = in[in_idx];
                    }
                }
            }
        }
    } else if (strcmp(mode, "bilinear") == 0) {
        /* Bilinear interpolation */
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t c = 0; c < channels; c++) {
                for (int64_t h = 0; h < out_h; h++) {
                    for (int64_t w = 0; w < out_w; w++) {
                        float src_h = (float)h * (in_h - 1) / (out_h - 1);
                        float src_w = (float)w * (in_w - 1) / (out_w - 1);

                        int64_t h0 = (int64_t)src_h;
                        int64_t w0 = (int64_t)src_w;
                        int64_t h1 = h0 + 1;
                        int64_t w1 = w0 + 1;
                        if (h1 >= in_h) h1 = in_h - 1;
                        if (w1 >= in_w) w1 = in_w - 1;

                        float hf = src_h - h0;
                        float wf = src_w - w0;

                        int64_t base = b * channels * in_h * in_w + c * in_h * in_w;
                        float v00 = in[base + h0 * in_w + w0];
                        float v01 = in[base + h0 * in_w + w1];
                        float v10 = in[base + h1 * in_w + w0];
                        float v11 = in[base + h1 * in_w + w1];

                        float value = v00 * (1 - hf) * (1 - wf) +
                                      v01 * (1 - hf) * wf +
                                      v10 * hf * (1 - wf) +
                                      v11 * hf * wf;

                        int64_t out_idx = b * channels * out_h * out_w +
                                          c * out_h * out_w + h * out_w + w;
                        o[out_idx] = value;
                    }
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_batch_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias,
                         tm_tensor* running_mean, tm_tensor* running_var,
                         bool training, float momentum, float eps) {
    /* Batch norm: (x - mean) / sqrt(var + eps) * weight + bias */
    if (!t || t->ndim < 2) return NULL;

    int64_t batch = t->shape[0];
    int64_t channels = t->shape[1];
    int64_t spatial = 1;
    for (int i = 2; i < t->ndim; i++) {
        spatial *= t->shape[i];
    }

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;
    float* w = weight ? (float*)weight->data : NULL;
    float* b = bias ? (float*)bias->data : NULL;

    for (int64_t c = 0; c < channels; c++) {
        float mean, var;

        if (training || !running_mean || !running_var) {
            /* Compute batch statistics */
            mean = 0.0f;
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t s = 0; s < spatial; s++) {
                    mean += in[n * channels * spatial + c * spatial + s];
                }
            }
            mean /= (float)(batch * spatial);

            var = 0.0f;
            for (int64_t n = 0; n < batch; n++) {
                for (int64_t s = 0; s < spatial; s++) {
                    float diff = in[n * channels * spatial + c * spatial + s] - mean;
                    var += diff * diff;
                }
            }
            var /= (float)(batch * spatial);

            /* Update running stats if training */
            if (training && running_mean && running_var) {
                float* rm = (float*)running_mean->data;
                float* rv = (float*)running_var->data;
                rm[c] = (1.0f - momentum) * rm[c] + momentum * mean;
                rv[c] = (1.0f - momentum) * rv[c] + momentum * var;
            }
        } else {
            /* Use running statistics */
            mean = ((float*)running_mean->data)[c];
            var = ((float*)running_var->data)[c];
        }

        float inv_std = 1.0f / sqrtf(var + eps);
        float gamma = w ? w[c] : 1.0f;
        float beta_val = b ? b[c] : 0.0f;

        /* Normalize */
        for (int64_t n = 0; n < batch; n++) {
            for (int64_t s = 0; s < spatial; s++) {
                int64_t idx = n * channels * spatial + c * spatial + s;
                o[idx] = (in[idx] - mean) * inv_std * gamma + beta_val;
            }
        }
    }

    return out;
}

tm_tensor* tm_group_norm(tm_tensor* t, int num_groups, tm_tensor* weight,
                         tm_tensor* bias, float eps) {
    /* Group norm: normalize within groups of channels */
    if (!t || t->ndim < 2) return NULL;

    int64_t batch = t->shape[0];
    int64_t channels = t->shape[1];
    if (channels % num_groups != 0) return NULL;

    int64_t channels_per_group = channels / num_groups;
    int64_t spatial = 1;
    for (int i = 2; i < t->ndim; i++) {
        spatial *= t->shape[i];
    }

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;
    float* w = weight ? (float*)weight->data : NULL;
    float* b = bias ? (float*)bias->data : NULL;

    for (int64_t n = 0; n < batch; n++) {
        for (int g = 0; g < num_groups; g++) {
            /* Compute mean and var for this group */
            float mean = 0.0f;
            int64_t count = channels_per_group * spatial;

            for (int64_t c = 0; c < channels_per_group; c++) {
                int64_t ch = g * channels_per_group + c;
                for (int64_t s = 0; s < spatial; s++) {
                    mean += in[n * channels * spatial + ch * spatial + s];
                }
            }
            mean /= (float)count;

            float var = 0.0f;
            for (int64_t c = 0; c < channels_per_group; c++) {
                int64_t ch = g * channels_per_group + c;
                for (int64_t s = 0; s < spatial; s++) {
                    float diff = in[n * channels * spatial + ch * spatial + s] - mean;
                    var += diff * diff;
                }
            }
            var /= (float)count;

            float inv_std = 1.0f / sqrtf(var + eps);

            /* Normalize */
            for (int64_t c = 0; c < channels_per_group; c++) {
                int64_t ch = g * channels_per_group + c;
                float gamma = w ? w[ch] : 1.0f;
                float beta_val = b ? b[ch] : 0.0f;

                for (int64_t s = 0; s < spatial; s++) {
                    int64_t idx = n * channels * spatial + ch * spatial + s;
                    o[idx] = (in[idx] - mean) * inv_std * gamma + beta_val;
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_instance_norm(tm_tensor* t, tm_tensor* weight, tm_tensor* bias, float eps) {
    /* Instance norm: normalize per (batch, channel) independently */
    if (!t || t->ndim < 2) return NULL;

    int64_t batch = t->shape[0];
    int64_t channels = t->shape[1];
    int64_t spatial = 1;
    for (int i = 2; i < t->ndim; i++) {
        spatial *= t->shape[i];
    }

    tm_tensor* out = tm_empty(t->shape, t->ndim, t->dtype);
    if (!out) return NULL;

    float* in = (float*)t->data;
    float* o = (float*)out->data;
    float* w = weight ? (float*)weight->data : NULL;
    float* b = bias ? (float*)bias->data : NULL;

    for (int64_t n = 0; n < batch; n++) {
        for (int64_t c = 0; c < channels; c++) {
            /* Compute mean and var for this (batch, channel) */
            float mean = 0.0f;
            for (int64_t s = 0; s < spatial; s++) {
                mean += in[n * channels * spatial + c * spatial + s];
            }
            mean /= (float)spatial;

            float var = 0.0f;
            for (int64_t s = 0; s < spatial; s++) {
                float diff = in[n * channels * spatial + c * spatial + s] - mean;
                var += diff * diff;
            }
            var /= (float)spatial;

            float inv_std = 1.0f / sqrtf(var + eps);
            float gamma = w ? w[c] : 1.0f;
            float beta_val = b ? b[c] : 0.0f;

            /* Normalize */
            for (int64_t s = 0; s < spatial; s++) {
                int64_t idx = n * channels * spatial + c * spatial + s;
                o[idx] = (in[idx] - mean) * inv_std * gamma + beta_val;
            }
        }
    }

    return out;
}

tm_tensor* tm_scaled_dot_product_attention(tm_tensor* Q, tm_tensor* K, tm_tensor* V,
                                           tm_tensor* attn_mask, float dropout_p,
                                           bool is_causal) {
    /*
     * Scaled dot-product attention:
     * Q: [batch, heads, seq_q, head_dim]
     * K: [batch, heads, seq_k, head_dim]
     * V: [batch, heads, seq_k, head_dim]
     * attn_mask: optional [batch, heads, seq_q, seq_k] or broadcastable
     * Returns: [batch, heads, seq_q, head_dim]
     */
    if (!Q || !K || !V) return NULL;
    if (Q->ndim != 4 || K->ndim != 4 || V->ndim != 4) return NULL;

    int64_t batch = Q->shape[0];
    int64_t heads = Q->shape[1];
    int64_t seq_q = Q->shape[2];
    int64_t head_dim = Q->shape[3];
    int64_t seq_k = K->shape[2];

    /* Validate shapes */
    if (K->shape[0] != batch || K->shape[1] != heads || K->shape[3] != head_dim) return NULL;
    if (V->shape[0] != batch || V->shape[1] != heads || V->shape[2] != seq_k || V->shape[3] != head_dim) return NULL;

    float scale = 1.0f / sqrtf((float)head_dim);

    /* Allocate output */
    int64_t out_shape[4] = {batch, heads, seq_q, head_dim};
    tm_tensor* out = tm_zeros(out_shape, 4, Q->dtype);
    if (!out) return NULL;

    float* q_data = (float*)Q->data;
    float* k_data = (float*)K->data;
    float* v_data = (float*)V->data;
    float* o_data = (float*)out->data;
    float* mask_data = attn_mask ? (float*)attn_mask->data : NULL;

    /* For each batch and head */
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < heads; h++) {
            /* Compute attention scores: Q @ K^T * scale */
            float* scores = (float*)malloc(seq_q * seq_k * sizeof(float));
            if (!scores) {
                tm_release(out);
                return NULL;
            }

            /* Q @ K^T */
            for (int64_t i = 0; i < seq_q; i++) {
                for (int64_t j = 0; j < seq_k; j++) {
                    float sum = 0.0f;
                    for (int64_t d = 0; d < head_dim; d++) {
                        int64_t q_idx = b * heads * seq_q * head_dim + h * seq_q * head_dim + i * head_dim + d;
                        int64_t k_idx = b * heads * seq_k * head_dim + h * seq_k * head_dim + j * head_dim + d;
                        sum += q_data[q_idx] * k_data[k_idx];
                    }
                    scores[i * seq_k + j] = sum * scale;

                    /* Apply causal mask */
                    if (is_causal && j > i) {
                        scores[i * seq_k + j] = -INFINITY;
                    }

                    /* Apply attention mask */
                    if (mask_data) {
                        int64_t mask_idx = b * heads * seq_q * seq_k + h * seq_q * seq_k + i * seq_k + j;
                        scores[i * seq_k + j] += mask_data[mask_idx];
                    }
                }
            }

            /* Softmax over seq_k dimension */
            for (int64_t i = 0; i < seq_q; i++) {
                float max_val = scores[i * seq_k];
                for (int64_t j = 1; j < seq_k; j++) {
                    if (scores[i * seq_k + j] > max_val) {
                        max_val = scores[i * seq_k + j];
                    }
                }

                float sum_exp = 0.0f;
                for (int64_t j = 0; j < seq_k; j++) {
                    scores[i * seq_k + j] = expf(scores[i * seq_k + j] - max_val);
                    sum_exp += scores[i * seq_k + j];
                }

                for (int64_t j = 0; j < seq_k; j++) {
                    scores[i * seq_k + j] /= sum_exp;
                }
            }

            /* Apply dropout if needed (simplified: just scale) */
            if (dropout_p > 0.0f && dropout_p < 1.0f) {
                float scale_factor = 1.0f / (1.0f - dropout_p);
                for (int64_t i = 0; i < seq_q * seq_k; i++) {
                    if (rand_uniform() < dropout_p) {
                        scores[i] = 0.0f;
                    } else {
                        scores[i] *= scale_factor;
                    }
                }
            }

            /* Compute output: scores @ V */
            for (int64_t i = 0; i < seq_q; i++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int64_t j = 0; j < seq_k; j++) {
                        int64_t v_idx = b * heads * seq_k * head_dim + h * seq_k * head_dim + j * head_dim + d;
                        sum += scores[i * seq_k + j] * v_data[v_idx];
                    }
                    int64_t o_idx = b * heads * seq_q * head_dim + h * seq_q * head_dim + i * head_dim + d;
                    o_data[o_idx] = sum;
                }
            }

            free(scores);
        }
    }

    return out;
}

tm_tensor* tm_split_heads(tm_tensor* x, int num_heads) {
    /* x: [batch, seq, embed_dim] -> [batch, heads, seq, head_dim] */
    if (!x || x->ndim != 3) return NULL;

    int64_t batch = x->shape[0];
    int64_t seq = x->shape[1];
    int64_t embed_dim = x->shape[2];

    if (embed_dim % num_heads != 0) return NULL;
    int64_t head_dim = embed_dim / num_heads;

    int64_t out_shape[4] = {batch, num_heads, seq, head_dim};
    tm_tensor* out = tm_empty(out_shape, 4, x->dtype);
    if (!out) return NULL;

    float* in = (float*)x->data;
    float* o = (float*)out->data;

    /* Reshape and permute: [b, s, h, d] -> [b, h, s, d] */
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    int64_t in_idx = b * seq * embed_dim + s * embed_dim + h * head_dim + d;
                    int64_t out_idx = b * num_heads * seq * head_dim + h * seq * head_dim + s * head_dim + d;
                    o[out_idx] = in[in_idx];
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_merge_heads(tm_tensor* x) {
    /* x: [batch, heads, seq, head_dim] -> [batch, seq, embed_dim] */
    if (!x || x->ndim != 4) return NULL;

    int64_t batch = x->shape[0];
    int64_t heads = x->shape[1];
    int64_t seq = x->shape[2];
    int64_t head_dim = x->shape[3];
    int64_t embed_dim = heads * head_dim;

    int64_t out_shape[3] = {batch, seq, embed_dim};
    tm_tensor* out = tm_empty(out_shape, 3, x->dtype);
    if (!out) return NULL;

    float* in = (float*)x->data;
    float* o = (float*)out->data;

    /* Permute and reshape: [b, h, s, d] -> [b, s, h*d] */
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq; s++) {
            for (int64_t h = 0; h < heads; h++) {
                for (int64_t d = 0; d < head_dim; d++) {
                    int64_t in_idx = b * heads * seq * head_dim + h * seq * head_dim + s * head_dim + d;
                    int64_t out_idx = b * seq * embed_dim + s * embed_dim + h * head_dim + d;
                    o[out_idx] = in[in_idx];
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_conv1d(tm_tensor* input, tm_tensor* weight, tm_tensor* bias,
                     int stride, int padding, int dilation, int groups) {
    /* 1D Convolution
     * input: [batch, in_channels, length]
     * weight: [out_channels, in_channels/groups, kernel_size]
     * bias: [out_channels] or NULL
     * output: [batch, out_channels, output_length]
     */
    if (!input || !weight || input->ndim != 3 || weight->ndim != 3) return NULL;

    int64_t batch = input->shape[0];
    int64_t in_channels = input->shape[1];
    int64_t in_length = input->shape[2];
    int64_t out_channels = weight->shape[0];
    int64_t kernel_size = weight->shape[2];

    if (in_channels % groups != 0 || out_channels % groups != 0) return NULL;
    int64_t in_ch_per_group = in_channels / groups;
    int64_t out_ch_per_group = out_channels / groups;

    /* Calculate output length */
    int64_t out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    if (out_length <= 0) return NULL;

    int64_t out_shape[3] = {batch, out_channels, out_length};
    tm_tensor* out = tm_zeros(out_shape, 3, input->dtype);
    if (!out) return NULL;

    float* in_data = (float*)input->data;
    float* w_data = (float*)weight->data;
    float* b_data = bias ? (float*)bias->data : NULL;
    float* o_data = (float*)out->data;

    for (int64_t b = 0; b < batch; b++) {
        for (int g = 0; g < groups; g++) {
            for (int64_t oc = 0; oc < out_ch_per_group; oc++) {
                int64_t out_ch = g * out_ch_per_group + oc;

                for (int64_t o = 0; o < out_length; o++) {
                    float sum = 0.0f;

                    for (int64_t ic = 0; ic < in_ch_per_group; ic++) {
                        int64_t in_ch = g * in_ch_per_group + ic;

                        for (int64_t k = 0; k < kernel_size; k++) {
                            int64_t in_pos = o * stride - padding + k * dilation;
                            if (in_pos >= 0 && in_pos < in_length) {
                                int64_t in_idx = b * in_channels * in_length + in_ch * in_length + in_pos;
                                int64_t w_idx = out_ch * in_ch_per_group * kernel_size + ic * kernel_size + k;
                                sum += in_data[in_idx] * w_data[w_idx];
                            }
                        }
                    }

                    if (b_data) sum += b_data[out_ch];

                    int64_t out_idx = b * out_channels * out_length + out_ch * out_length + o;
                    o_data[out_idx] = sum;
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_conv2d(tm_tensor* input, tm_tensor* weight, tm_tensor* bias,
                     int stride_h, int stride_w, int padding_h, int padding_w,
                     int dilation_h, int dilation_w, int groups) {
    /* 2D Convolution
     * input: [batch, in_channels, height, width]
     * weight: [out_channels, in_channels/groups, kernel_h, kernel_w]
     * output: [batch, out_channels, out_height, out_width]
     */
    if (!input || !weight || input->ndim != 4 || weight->ndim != 4) return NULL;

    int64_t batch = input->shape[0];
    int64_t in_channels = input->shape[1];
    int64_t in_h = input->shape[2];
    int64_t in_w = input->shape[3];
    int64_t out_channels = weight->shape[0];
    int64_t kernel_h = weight->shape[2];
    int64_t kernel_w = weight->shape[3];

    if (in_channels % groups != 0 || out_channels % groups != 0) return NULL;
    int64_t in_ch_per_group = in_channels / groups;
    int64_t out_ch_per_group = out_channels / groups;

    int64_t out_h = (in_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (in_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    if (out_h <= 0 || out_w <= 0) return NULL;

    int64_t out_shape[4] = {batch, out_channels, out_h, out_w};
    tm_tensor* out = tm_zeros(out_shape, 4, input->dtype);
    if (!out) return NULL;

    float* in_data = (float*)input->data;
    float* w_data = (float*)weight->data;
    float* b_data = bias ? (float*)bias->data : NULL;
    float* o_data = (float*)out->data;

    for (int64_t b = 0; b < batch; b++) {
        for (int g = 0; g < groups; g++) {
            for (int64_t oc = 0; oc < out_ch_per_group; oc++) {
                int64_t out_ch = g * out_ch_per_group + oc;

                for (int64_t oh = 0; oh < out_h; oh++) {
                    for (int64_t ow = 0; ow < out_w; ow++) {
                        float sum = 0.0f;

                        for (int64_t ic = 0; ic < in_ch_per_group; ic++) {
                            int64_t in_ch = g * in_ch_per_group + ic;

                            for (int64_t kh = 0; kh < kernel_h; kh++) {
                                for (int64_t kw = 0; kw < kernel_w; kw++) {
                                    int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
                                    int64_t iw = ow * stride_w - padding_w + kw * dilation_w;

                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        int64_t in_idx = b * in_channels * in_h * in_w +
                                                         in_ch * in_h * in_w + ih * in_w + iw;
                                        int64_t w_idx = out_ch * in_ch_per_group * kernel_h * kernel_w +
                                                        ic * kernel_h * kernel_w + kh * kernel_w + kw;
                                        sum += in_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }

                        if (b_data) sum += b_data[out_ch];

                        int64_t out_idx = b * out_channels * out_h * out_w +
                                          out_ch * out_h * out_w + oh * out_w + ow;
                        o_data[out_idx] = sum;
                    }
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_avg_pool1d(tm_tensor* input, int kernel_size, int stride, int padding) {
    /* 1D Average Pooling */
    if (!input || input->ndim != 3) return NULL;

    int64_t batch = input->shape[0];
    int64_t channels = input->shape[1];
    int64_t in_length = input->shape[2];

    int64_t out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    if (out_length <= 0) return NULL;

    int64_t out_shape[3] = {batch, channels, out_length};
    tm_tensor* out = tm_empty(out_shape, 3, input->dtype);
    if (!out) return NULL;

    float* in_data = (float*)input->data;
    float* o_data = (float*)out->data;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            for (int64_t o = 0; o < out_length; o++) {
                float sum = 0.0f;
                int count = 0;

                for (int k = 0; k < kernel_size; k++) {
                    int64_t in_pos = o * stride - padding + k;
                    if (in_pos >= 0 && in_pos < in_length) {
                        sum += in_data[b * channels * in_length + c * in_length + in_pos];
                        count++;
                    }
                }

                o_data[b * channels * out_length + c * out_length + o] = sum / count;
            }
        }
    }

    return out;
}

tm_tensor* tm_max_pool1d(tm_tensor* input, int kernel_size, int stride, int padding) {
    /* 1D Max Pooling */
    if (!input || input->ndim != 3) return NULL;

    int64_t batch = input->shape[0];
    int64_t channels = input->shape[1];
    int64_t in_length = input->shape[2];

    int64_t out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    if (out_length <= 0) return NULL;

    int64_t out_shape[3] = {batch, channels, out_length};
    tm_tensor* out = tm_empty(out_shape, 3, input->dtype);
    if (!out) return NULL;

    float* in_data = (float*)input->data;
    float* o_data = (float*)out->data;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            for (int64_t o = 0; o < out_length; o++) {
                float max_val = -INFINITY;

                for (int k = 0; k < kernel_size; k++) {
                    int64_t in_pos = o * stride - padding + k;
                    if (in_pos >= 0 && in_pos < in_length) {
                        float val = in_data[b * channels * in_length + c * in_length + in_pos];
                        if (val > max_val) max_val = val;
                    }
                }

                o_data[b * channels * out_length + c * out_length + o] = max_val;
            }
        }
    }

    return out;
}

tm_tensor* tm_avg_pool2d(tm_tensor* input, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int padding_h, int padding_w) {
    /* 2D Average Pooling */
    if (!input || input->ndim != 4) return NULL;

    int64_t batch = input->shape[0];
    int64_t channels = input->shape[1];
    int64_t in_h = input->shape[2];
    int64_t in_w = input->shape[3];

    int64_t out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    if (out_h <= 0 || out_w <= 0) return NULL;

    int64_t out_shape[4] = {batch, channels, out_h, out_w};
    tm_tensor* out = tm_empty(out_shape, 4, input->dtype);
    if (!out) return NULL;

    float* in_data = (float*)input->data;
    float* o_data = (float*)out->data;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            for (int64_t oh = 0; oh < out_h; oh++) {
                for (int64_t ow = 0; ow < out_w; ow++) {
                    float sum = 0.0f;
                    int count = 0;

                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int64_t ih = oh * stride_h - padding_h + kh;
                            int64_t iw = ow * stride_w - padding_w + kw;

                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += in_data[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                                count++;
                            }
                        }
                    }

                    o_data[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = sum / count;
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_max_pool2d(tm_tensor* input, int kernel_h, int kernel_w,
                         int stride_h, int stride_w, int padding_h, int padding_w) {
    /* 2D Max Pooling */
    if (!input || input->ndim != 4) return NULL;

    int64_t batch = input->shape[0];
    int64_t channels = input->shape[1];
    int64_t in_h = input->shape[2];
    int64_t in_w = input->shape[3];

    int64_t out_h = (in_h + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_w = (in_w + 2 * padding_w - kernel_w) / stride_w + 1;
    if (out_h <= 0 || out_w <= 0) return NULL;

    int64_t out_shape[4] = {batch, channels, out_h, out_w};
    tm_tensor* out = tm_empty(out_shape, 4, input->dtype);
    if (!out) return NULL;

    float* in_data = (float*)input->data;
    float* o_data = (float*)out->data;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            for (int64_t oh = 0; oh < out_h; oh++) {
                for (int64_t ow = 0; ow < out_w; ow++) {
                    float max_val = -INFINITY;

                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int64_t ih = oh * stride_h - padding_h + kh;
                            int64_t iw = ow * stride_w - padding_w + kw;

                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                float val = in_data[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }

                    o_data[b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = max_val;
                }
            }
        }
    }

    return out;
}

tm_tensor* tm_adaptive_avg_pool1d(tm_tensor* input, int output_size) {
    /* Adaptive Average Pooling 1D */
    if (!input || input->ndim != 3) return NULL;

    int64_t batch = input->shape[0];
    int64_t channels = input->shape[1];
    int64_t in_length = input->shape[2];

    int64_t out_shape[3] = {batch, channels, output_size};
    tm_tensor* out = tm_empty(out_shape, 3, input->dtype);
    if (!out) return NULL;

    float* in_data = (float*)input->data;
    float* o_data = (float*)out->data;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            for (int o = 0; o < output_size; o++) {
                /* Calculate input indices for this output position */
                int64_t start = (o * in_length) / output_size;
                int64_t end = ((o + 1) * in_length) / output_size;
                if (end <= start) end = start + 1;

                float sum = 0.0f;
                for (int64_t i = start; i < end; i++) {
                    sum += in_data[b * channels * in_length + c * in_length + i];
                }

                o_data[b * channels * output_size + c * output_size + o] = sum / (end - start);
            }
        }
    }

    return out;
}

tm_tensor* tm_adaptive_avg_pool2d(tm_tensor* input, int output_h, int output_w) {
    /* Adaptive Average Pooling 2D */
    if (!input || input->ndim != 4) return NULL;

    int64_t batch = input->shape[0];
    int64_t channels = input->shape[1];
    int64_t in_h = input->shape[2];
    int64_t in_w = input->shape[3];

    int64_t out_shape[4] = {batch, channels, output_h, output_w};
    tm_tensor* out = tm_empty(out_shape, 4, input->dtype);
    if (!out) return NULL;

    float* in_data = (float*)input->data;
    float* o_data = (float*)out->data;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t c = 0; c < channels; c++) {
            for (int oh = 0; oh < output_h; oh++) {
                for (int ow = 0; ow < output_w; ow++) {
                    int64_t h_start = (oh * in_h) / output_h;
                    int64_t h_end = ((oh + 1) * in_h) / output_h;
                    int64_t w_start = (ow * in_w) / output_w;
                    int64_t w_end = ((ow + 1) * in_w) / output_w;

                    if (h_end <= h_start) h_end = h_start + 1;
                    if (w_end <= w_start) w_end = w_start + 1;

                    float sum = 0.0f;
                    int count = 0;
                    for (int64_t ih = h_start; ih < h_end; ih++) {
                        for (int64_t iw = w_start; iw < w_end; iw++) {
                            sum += in_data[b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw];
                            count++;
                        }
                    }

                    o_data[b * channels * output_h * output_w + c * output_h * output_w + oh * output_w + ow] = sum / count;
                }
            }
        }
    }

    return out;
}

/* ============================================================= */
/* Loss Functions                                                */
/* ============================================================= */

/* Backward for cross-entropy: grad = softmax(logits) - one_hot(targets) */
static void backward_cross_entropy(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* logits = gf->inputs[0];
    tm_tensor* targets = gf->inputs[1];  /* Class indices */

    if (!logits->requires_grad || !out->grad) return;

    if (!logits->grad) logits->grad = tm_zeros(logits->shape, logits->ndim, logits->dtype);

    int batch_size = (int)logits->shape[0];
    int num_classes = (int)logits->shape[1];

    float* lg = tm_data_f32(logits->grad);
    float* ld = tm_data_f32(logits);
    float* og = tm_data_f32(out->grad);  /* Scalar gradient */
    int* td = (int*)targets->data;

    float scale = og[0] / batch_size;

    /* Compute softmax and subtract 1 at target class */
    for (int b = 0; b < batch_size; b++) {
        /* Find max for numerical stability */
        float max_val = ld[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (ld[b * num_classes + c] > max_val) {
                max_val = ld[b * num_classes + c];
            }
        }

        /* Compute softmax */
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(ld[b * num_classes + c] - max_val);
        }

        for (int c = 0; c < num_classes; c++) {
            float softmax_val = expf(ld[b * num_classes + c] - max_val) / sum_exp;
            float grad_val = softmax_val;
            if (c == td[b]) {
                grad_val -= 1.0f;
            }
            lg[b * num_classes + c] += scale * grad_val;
        }
    }
}

tm_tensor* tm_cross_entropy(tm_tensor* logits, tm_tensor* targets) {
    if (!logits || !targets) return NULL;
    if (logits->ndim != 2) return NULL;  /* Expect [batch, classes] */
    if (targets->ndim != 1) return NULL;  /* Expect [batch] */
    if (logits->shape[0] != targets->shape[0]) return NULL;

    int batch_size = (int)logits->shape[0];
    int num_classes = (int)logits->shape[1];

    float* ld = tm_data_f32(logits);
    /* Support both int32 and int64 targets */
    int64_t* td64 = (targets->dtype == TM_I64) ? (int64_t*)targets->data : NULL;
    int32_t* td32 = (targets->dtype == TM_I32) ? (int32_t*)targets->data : NULL;

    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        /* Find max for numerical stability */
        float max_val = ld[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (ld[b * num_classes + c] > max_val) {
                max_val = ld[b * num_classes + c];
            }
        }

        /* Compute log_sum_exp */
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(ld[b * num_classes + c] - max_val);
        }
        float log_sum_exp = max_val + logf(sum_exp);

        /* Loss for this sample: log_sum_exp - logits[target] */
        int target_class = td64 ? (int)td64[b] : (td32 ? td32[b] : 0);
        if (target_class >= 0 && target_class < num_classes) {
            total_loss += log_sum_exp - ld[b * num_classes + target_class];
        }
    }

    /* Average over batch */
    float avg_loss = total_loss / batch_size;

    /* Create scalar output */
    int64_t scalar_shape[] = {1};
    tm_tensor* out = tm_full(scalar_shape, 1, TM_F32, avg_loss);

    if (logits->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_cross_entropy);
        out->grad_fn->inputs[0] = tm_retain(logits);
        out->grad_fn->inputs[1] = tm_retain(targets);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* Cross-entropy with soft targets (e.g., for knowledge distillation) */
tm_tensor* tm_cross_entropy_soft(tm_tensor* logits, tm_tensor* targets) {
    if (!logits || !targets) return NULL;
    if (logits->ndim != targets->ndim) return NULL;
    if (logits->numel != targets->numel) return NULL;

    /* Compute log_softmax of logits */
    tm_tensor* log_probs = tm_log_softmax(logits, -1);
    if (!log_probs) return NULL;

    /* Compute -sum(targets * log_probs) */
    float* lp = tm_data_f32(log_probs);
    float* td = tm_data_f32(targets);

    float total_loss = 0.0f;
    for (int64_t i = 0; i < logits->numel; i++) {
        total_loss -= td[i] * lp[i];
    }

    /* Average over batch (first dimension) */
    int batch_size = (int)logits->shape[0];
    float avg_loss = total_loss / batch_size;

    tm_release(log_probs);

    int64_t scalar_shape[] = {1};
    return tm_full(scalar_shape, 1, TM_F32, avg_loss);
}

/* Backward for MSE: grad = 2 * (pred - target) / n */
static void backward_mse(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* pred = gf->inputs[0];
    tm_tensor* target = gf->inputs[1];

    if (!pred->requires_grad || !out->grad) return;

    if (!pred->grad) pred->grad = tm_zeros(pred->shape, pred->ndim, pred->dtype);

    float* pg = tm_data_f32(pred->grad);
    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);
    float* og = tm_data_f32(out->grad);

    float scale = og[0] * 2.0f / pred->numel;

    for (int64_t i = 0; i < pred->numel; i++) {
        pg[i] += scale * (pd[i] - td[i]);
    }
}

tm_tensor* tm_mse_loss(tm_tensor* pred, tm_tensor* target) {
    if (!pred || !target) return NULL;
    if (pred->numel != target->numel) return NULL;

    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);

    float total_loss = 0.0f;
    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = pd[i] - td[i];
        total_loss += diff * diff;
    }

    float avg_loss = total_loss / pred->numel;

    int64_t scalar_shape[] = {1};
    tm_tensor* out = tm_full(scalar_shape, 1, TM_F32, avg_loss);

    if (pred->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_mse);
        out->grad_fn->inputs[0] = tm_retain(pred);
        out->grad_fn->inputs[1] = tm_retain(target);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* Backward for L1: grad = sign(pred - target) / n */
static void backward_l1(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* pred = gf->inputs[0];
    tm_tensor* target = gf->inputs[1];

    if (!pred->requires_grad || !out->grad) return;

    if (!pred->grad) pred->grad = tm_zeros(pred->shape, pred->ndim, pred->dtype);

    float* pg = tm_data_f32(pred->grad);
    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);
    float* og = tm_data_f32(out->grad);

    float scale = og[0] / pred->numel;

    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = pd[i] - td[i];
        float sign = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
        pg[i] += scale * sign;
    }
}

tm_tensor* tm_l1_loss(tm_tensor* pred, tm_tensor* target) {
    if (!pred || !target) return NULL;
    if (pred->numel != target->numel) return NULL;

    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);

    float total_loss = 0.0f;
    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = pd[i] - td[i];
        total_loss += (diff >= 0) ? diff : -diff;
    }

    float avg_loss = total_loss / pred->numel;

    int64_t scalar_shape[] = {1};
    tm_tensor* out = tm_full(scalar_shape, 1, TM_F32, avg_loss);

    if (pred->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_l1);
        out->grad_fn->inputs[0] = tm_retain(pred);
        out->grad_fn->inputs[1] = tm_retain(target);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* Backward for Smooth L1: grad = x if |x| < 1, else sign(x), scaled by 1/n */
static void backward_smooth_l1(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* pred = gf->inputs[0];
    tm_tensor* target = gf->inputs[1];
    float beta = *(float*)gf->ctx;

    if (!pred->requires_grad || !out->grad) return;

    if (!pred->grad) pred->grad = tm_zeros(pred->shape, pred->ndim, pred->dtype);

    float* pg = tm_data_f32(pred->grad);
    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);
    float* og = tm_data_f32(out->grad);

    float scale = og[0] / pred->numel;

    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = pd[i] - td[i];
        float abs_diff = (diff >= 0) ? diff : -diff;
        float grad;
        if (abs_diff < beta) {
            grad = diff / beta;
        } else {
            grad = (diff > 0) ? 1.0f : -1.0f;
        }
        pg[i] += scale * grad;
    }
}

tm_tensor* tm_smooth_l1_loss(tm_tensor* pred, tm_tensor* target, float beta) {
    if (!pred || !target) return NULL;
    if (pred->numel != target->numel) return NULL;
    if (beta <= 0) beta = 1.0f;

    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);

    float total_loss = 0.0f;
    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = pd[i] - td[i];
        float abs_diff = (diff >= 0) ? diff : -diff;
        if (abs_diff < beta) {
            total_loss += 0.5f * diff * diff / beta;
        } else {
            total_loss += abs_diff - 0.5f * beta;
        }
    }

    float avg_loss = total_loss / pred->numel;

    int64_t scalar_shape[] = {1};
    tm_tensor* out = tm_full(scalar_shape, 1, TM_F32, avg_loss);

    if (pred->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_smooth_l1);
        out->grad_fn->inputs[0] = tm_retain(pred);
        out->grad_fn->inputs[1] = tm_retain(target);
        out->grad_fn->num_inputs = 2;
        float* ctx = malloc(sizeof(float));
        *ctx = beta;
        out->grad_fn->ctx = ctx;
    }

    return out;
}

/* NLL Loss: assumes input is log probabilities, targets are class indices
 * Loss = -sum(log_probs[target]) / batch_size */
static void backward_nll(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* log_probs = gf->inputs[0];
    tm_tensor* targets = gf->inputs[1];

    if (!log_probs->requires_grad || !out->grad) return;

    if (!log_probs->grad) log_probs->grad = tm_zeros(log_probs->shape, log_probs->ndim, log_probs->dtype);

    float* lg = tm_data_f32(log_probs->grad);
    int* td = (int*)targets->data;
    float* og = tm_data_f32(out->grad);

    int batch_size = (int)log_probs->shape[0];
    int num_classes = (int)log_probs->shape[1];
    float scale = -og[0] / batch_size;

    for (int b = 0; b < batch_size; b++) {
        int target_class = td[b];
        if (target_class >= 0 && target_class < num_classes) {
            lg[b * num_classes + target_class] += scale;
        }
    }
}

tm_tensor* tm_nll_loss(tm_tensor* log_probs, tm_tensor* targets) {
    if (!log_probs || !targets) return NULL;
    if (log_probs->ndim != 2) return NULL;  /* Expect [batch, classes] */
    if (targets->ndim != 1) return NULL;    /* Expect [batch] */
    if (log_probs->shape[0] != targets->shape[0]) return NULL;

    int batch_size = (int)log_probs->shape[0];
    int num_classes = (int)log_probs->shape[1];

    float* ld = tm_data_f32(log_probs);
    /* Support both int32 and int64 targets */
    int64_t* td64 = (targets->dtype == TM_I64) ? (int64_t*)targets->data : NULL;
    int32_t* td32 = (targets->dtype == TM_I32) ? (int32_t*)targets->data : NULL;

    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int target_class = td64 ? (int)td64[b] : (td32 ? td32[b] : 0);
        if (target_class >= 0 && target_class < num_classes) {
            total_loss -= ld[b * num_classes + target_class];
        }
    }

    float avg_loss = total_loss / batch_size;

    int64_t scalar_shape[] = {1};
    tm_tensor* out = tm_full(scalar_shape, 1, TM_F32, avg_loss);

    if (log_probs->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_nll);
        out->grad_fn->inputs[0] = tm_retain(log_probs);
        out->grad_fn->inputs[1] = tm_retain(targets);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* KL Divergence: KL(target || input) = sum(target * (log(target) - input))
 * Input is log probabilities, target is probabilities */
static void backward_kl_div(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* input = gf->inputs[0];
    tm_tensor* target = gf->inputs[1];

    if (!input->requires_grad || !out->grad) return;

    if (!input->grad) input->grad = tm_zeros(input->shape, input->ndim, input->dtype);

    float* ig = tm_data_f32(input->grad);
    float* td = tm_data_f32(target);
    float* og = tm_data_f32(out->grad);

    float scale = -og[0] / input->shape[0]; /* Average over batch */

    for (int64_t i = 0; i < input->numel; i++) {
        ig[i] += scale * td[i];
    }
}

tm_tensor* tm_kl_div(tm_tensor* input, tm_tensor* target) {
    if (!input || !target) return NULL;
    if (input->numel != target->numel) return NULL;

    float* id = tm_data_f32(input);
    float* td = tm_data_f32(target);

    float total_loss = 0.0f;
    for (int64_t i = 0; i < input->numel; i++) {
        if (td[i] > 0) {
            total_loss += td[i] * (logf(td[i]) - id[i]);
        }
    }

    /* Average over batch (first dimension) */
    int batch_size = (int)input->shape[0];
    float avg_loss = total_loss / batch_size;

    int64_t scalar_shape[] = {1};
    tm_tensor* out = tm_full(scalar_shape, 1, TM_F32, avg_loss);

    if (input->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_kl_div);
        out->grad_fn->inputs[0] = tm_retain(input);
        out->grad_fn->inputs[1] = tm_retain(target);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* Binary Cross Entropy: -sum(target*log(pred) + (1-target)*log(1-pred)) / n */
static void backward_bce(tm_tensor* out) {
    tm_grad_fn* gf = out->grad_fn;
    tm_tensor* pred = gf->inputs[0];
    tm_tensor* target = gf->inputs[1];

    if (!pred->requires_grad || !out->grad) return;

    if (!pred->grad) pred->grad = tm_zeros(pred->shape, pred->ndim, pred->dtype);

    float* pg = tm_data_f32(pred->grad);
    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);
    float* og = tm_data_f32(out->grad);

    float scale = og[0] / pred->numel;
    float eps = 1e-7f;

    for (int64_t i = 0; i < pred->numel; i++) {
        float p = pd[i];
        /* Clamp for numerical stability */
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        /* grad = (p - t) / (p * (1 - p)) */
        float grad = (p - td[i]) / (p * (1.0f - p));
        pg[i] += scale * grad;
    }
}

tm_tensor* tm_binary_cross_entropy(tm_tensor* pred, tm_tensor* target) {
    if (!pred || !target) return NULL;
    if (pred->numel != target->numel) return NULL;

    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);

    float total_loss = 0.0f;
    float eps = 1e-7f;

    for (int64_t i = 0; i < pred->numel; i++) {
        float p = pd[i];
        /* Clamp for numerical stability */
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        total_loss -= td[i] * logf(p) + (1.0f - td[i]) * logf(1.0f - p);
    }

    float avg_loss = total_loss / pred->numel;

    int64_t scalar_shape[] = {1};
    tm_tensor* out = tm_full(scalar_shape, 1, TM_F32, avg_loss);

    if (pred->requires_grad) {
        out->requires_grad = true;
        out->grad_fn = create_grad_fn(backward_bce);
        out->grad_fn->inputs[0] = tm_retain(pred);
        out->grad_fn->inputs[1] = tm_retain(target);
        out->grad_fn->num_inputs = 2;
    }

    return out;
}

/* Binary Cross Entropy with Logits: applies sigmoid internally
 * More numerically stable than sigmoid + BCE */
tm_tensor* tm_binary_cross_entropy_with_logits(tm_tensor* logits, tm_tensor* target) {
    if (!logits || !target) return NULL;
    if (logits->numel != target->numel) return NULL;

    float* ld = tm_data_f32(logits);
    float* td = tm_data_f32(target);

    float total_loss = 0.0f;

    for (int64_t i = 0; i < logits->numel; i++) {
        float x = ld[i];
        float t = td[i];
        /* Numerically stable: max(x, 0) - x*t + log(1 + exp(-|x|)) */
        float max_val = (x > 0) ? x : 0;
        float abs_x = (x >= 0) ? x : -x;
        total_loss += max_val - x * t + logf(1.0f + expf(-abs_x));
    }

    float avg_loss = total_loss / logits->numel;

    int64_t scalar_shape[] = {1};
    return tm_full(scalar_shape, 1, TM_F32, avg_loss);
}

/* Huber Loss: smooth combination of L1 and L2 loss */
tm_tensor* tm_huber_loss(tm_tensor* pred, tm_tensor* target, float delta) {
    if (!pred || !target) return NULL;
    if (pred->numel != target->numel) return NULL;
    if (delta <= 0) delta = 1.0f;

    float* pd = tm_data_f32(pred);
    float* td = tm_data_f32(target);

    float total_loss = 0.0f;

    for (int64_t i = 0; i < pred->numel; i++) {
        float diff = pd[i] - td[i];
        float abs_diff = (diff >= 0) ? diff : -diff;
        if (abs_diff <= delta) {
            total_loss += 0.5f * diff * diff;
        } else {
            total_loss += delta * (abs_diff - 0.5f * delta);
        }
    }

    float avg_loss = total_loss / pred->numel;

    int64_t scalar_shape[] = {1};
    return tm_full(scalar_shape, 1, TM_F32, avg_loss);
}

/* Cosine Embedding Loss: measures if two inputs are similar or dissimilar
 * y=1: loss = 1 - cos(x1, x2)
 * y=-1: loss = max(0, cos(x1, x2) - margin) */
tm_tensor* tm_cosine_embedding_loss(tm_tensor* x1, tm_tensor* x2, tm_tensor* y, float margin) {
    if (!x1 || !x2 || !y) return NULL;
    if (x1->ndim != 2 || x2->ndim != 2 || y->ndim != 1) return NULL;
    if (x1->shape[0] != x2->shape[0] || x1->shape[0] != y->shape[0]) return NULL;
    if (x1->shape[1] != x2->shape[1]) return NULL;

    int batch = (int)x1->shape[0];
    int dim = (int)x1->shape[1];
    float* d1 = tm_data_f32(x1);
    float* d2 = tm_data_f32(x2);

    /* Support int64, int32, or float labels */
    int64_t* y64 = (y->dtype == TM_I64) ? (int64_t*)y->data : NULL;
    int32_t* y32 = (y->dtype == TM_I32) ? (int32_t*)y->data : NULL;
    float* yf = (y->dtype == TM_F32) ? (float*)y->data : NULL;

    float total_loss = 0.0f;
    float eps = 1e-8f;

    for (int b = 0; b < batch; b++) {
        /* Compute cosine similarity */
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (int i = 0; i < dim; i++) {
            float v1 = d1[b * dim + i];
            float v2 = d2[b * dim + i];
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        float cos_sim = dot / (sqrtf(norm1) * sqrtf(norm2) + eps);

        /* Get label value */
        int label = y64 ? (int)y64[b] : (y32 ? y32[b] : (yf ? (int)yf[b] : 0));

        if (label > 0) {
            /* Similar: loss = 1 - cos_sim */
            total_loss += 1.0f - cos_sim;
        } else {
            /* Dissimilar: loss = max(0, cos_sim - margin) */
            float loss = cos_sim - margin;
            if (loss > 0) total_loss += loss;
        }
    }

    float avg_loss = total_loss / batch;
    int64_t scalar_shape[] = {1};
    return tm_full(scalar_shape, 1, TM_F32, avg_loss);
}

/* Triplet Margin Loss: loss = max(0, d(anchor, positive) - d(anchor, negative) + margin) */
tm_tensor* tm_triplet_margin_loss(tm_tensor* anchor, tm_tensor* positive, tm_tensor* negative, float margin, float p) {
    if (!anchor || !positive || !negative) return NULL;
    if (anchor->ndim != 2 || positive->ndim != 2 || negative->ndim != 2) return NULL;
    if (anchor->shape[0] != positive->shape[0] || anchor->shape[0] != negative->shape[0]) return NULL;
    if (anchor->shape[1] != positive->shape[1] || anchor->shape[1] != negative->shape[1]) return NULL;

    int batch = (int)anchor->shape[0];
    int dim = (int)anchor->shape[1];
    float* ad = tm_data_f32(anchor);
    float* pd = tm_data_f32(positive);
    float* nd = tm_data_f32(negative);

    float total_loss = 0.0f;

    for (int b = 0; b < batch; b++) {
        /* Compute Lp distance to positive and negative */
        float dist_pos = 0.0f, dist_neg = 0.0f;
        for (int i = 0; i < dim; i++) {
            float dp = ad[b * dim + i] - pd[b * dim + i];
            float dn = ad[b * dim + i] - nd[b * dim + i];
            if (p == 2.0f) {
                dist_pos += dp * dp;
                dist_neg += dn * dn;
            } else {
                dist_pos += powf(fabsf(dp), p);
                dist_neg += powf(fabsf(dn), p);
            }
        }
        if (p == 2.0f) {
            dist_pos = sqrtf(dist_pos);
            dist_neg = sqrtf(dist_neg);
        } else {
            dist_pos = powf(dist_pos, 1.0f / p);
            dist_neg = powf(dist_neg, 1.0f / p);
        }

        float loss = dist_pos - dist_neg + margin;
        if (loss > 0) total_loss += loss;
    }

    float avg_loss = total_loss / batch;
    int64_t scalar_shape[] = {1};
    return tm_full(scalar_shape, 1, TM_F32, avg_loss);
}

/* Cross Entropy with Label Smoothing */
tm_tensor* tm_cross_entropy_label_smooth(tm_tensor* logits, tm_tensor* targets, float smoothing) {
    if (!logits || !targets) return NULL;
    if (logits->ndim != 2) return NULL;
    if (targets->ndim != 1) return NULL;
    if (logits->shape[0] != targets->shape[0]) return NULL;

    int batch_size = (int)logits->shape[0];
    int num_classes = (int)logits->shape[1];

    float* ld = tm_data_f32(logits);
    int* td = (int*)targets->data;

    float total_loss = 0.0f;
    float smooth_val = smoothing / num_classes;
    float confidence = 1.0f - smoothing;

    for (int b = 0; b < batch_size; b++) {
        /* Compute log_softmax */
        float max_val = ld[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (ld[b * num_classes + c] > max_val) {
                max_val = ld[b * num_classes + c];
            }
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(ld[b * num_classes + c] - max_val);
        }
        float log_sum_exp = max_val + logf(sum_exp);

        /* Compute smoothed cross entropy */
        float loss = 0.0f;
        int target_class = td[b];
        for (int c = 0; c < num_classes; c++) {
            float log_prob = ld[b * num_classes + c] - log_sum_exp;
            float target_prob = smooth_val;
            if (c == target_class) {
                target_prob += confidence;
            }
            loss -= target_prob * log_prob;
        }
        total_loss += loss;
    }

    float avg_loss = total_loss / batch_size;
    int64_t scalar_shape[] = {1};
    return tm_full(scalar_shape, 1, TM_F32, avg_loss);
}

/* ============================================================= */
/* Linear Algebra                                                */
/* ============================================================= */

/* Matrix inverse using Gauss-Jordan elimination */
tm_tensor* tm_inverse(tm_tensor* t) {
    if (!t || t->ndim != 2) return NULL;
    if (t->shape[0] != t->shape[1]) return NULL;  /* Must be square */

    int n = (int)t->shape[0];

    /* Create augmented matrix [A | I] */
    float* aug = (float*)malloc(n * 2 * n * sizeof(float));
    if (!aug) return NULL;

    float* td = tm_data_f32(t);

    /* Initialize augmented matrix */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * 2 * n + j] = td[i * n + j];
            aug[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    /* Gauss-Jordan elimination */
    for (int col = 0; col < n; col++) {
        /* Find pivot */
        int max_row = col;
        float max_val = fabsf(aug[col * 2 * n + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(aug[row * 2 * n + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        /* Check for singular matrix */
        if (max_val < 1e-10f) {
            free(aug);
            return NULL;  /* Singular matrix */
        }

        /* Swap rows */
        if (max_row != col) {
            for (int j = 0; j < 2 * n; j++) {
                float tmp = aug[col * 2 * n + j];
                aug[col * 2 * n + j] = aug[max_row * 2 * n + j];
                aug[max_row * 2 * n + j] = tmp;
            }
        }

        /* Scale pivot row */
        float pivot = aug[col * 2 * n + col];
        for (int j = 0; j < 2 * n; j++) {
            aug[col * 2 * n + j] /= pivot;
        }

        /* Eliminate column */
        for (int row = 0; row < n; row++) {
            if (row != col) {
                float factor = aug[row * 2 * n + col];
                for (int j = 0; j < 2 * n; j++) {
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                }
            }
        }
    }

    /* Extract inverse from augmented matrix */
    int64_t shape[] = {n, n};
    tm_tensor* out = tm_empty(shape, 2, TM_F32);
    if (!out) {
        free(aug);
        return NULL;
    }

    float* od = tm_data_f32(out);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            od[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    free(aug);
    return out;
}

/* Determinant using LU decomposition */
tm_tensor* tm_det(tm_tensor* t) {
    if (!t || t->ndim != 2) return NULL;
    if (t->shape[0] != t->shape[1]) return NULL;

    int n = (int)t->shape[0];

    /* Copy matrix for in-place LU decomposition */
    float* lu = (float*)malloc(n * n * sizeof(float));
    if (!lu) return NULL;

    float* td = tm_data_f32(t);
    memcpy(lu, td, n * n * sizeof(float));

    float det = 1.0f;
    int sign = 1;

    /* LU decomposition with partial pivoting */
    for (int col = 0; col < n; col++) {
        /* Find pivot */
        int max_row = col;
        float max_val = fabsf(lu[col * n + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(lu[row * n + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        if (max_val < 1e-10f) {
            free(lu);
            int64_t shape[] = {1};
            return tm_full(shape, 1, TM_F32, 0.0f);  /* Singular, det = 0 */
        }

        /* Swap rows */
        if (max_row != col) {
            sign = -sign;
            for (int j = 0; j < n; j++) {
                float tmp = lu[col * n + j];
                lu[col * n + j] = lu[max_row * n + j];
                lu[max_row * n + j] = tmp;
            }
        }

        det *= lu[col * n + col];

        /* Eliminate column */
        for (int row = col + 1; row < n; row++) {
            float factor = lu[row * n + col] / lu[col * n + col];
            for (int j = col; j < n; j++) {
                lu[row * n + j] -= factor * lu[col * n + j];
            }
        }
    }

    free(lu);

    int64_t shape[] = {1};
    return tm_full(shape, 1, TM_F32, sign * det);
}

/* SVD using power iteration (simplified - returns U, S, V where A = U * diag(S) * V^T)
 * This is a basic implementation; production would use LAPACK */
tm_tensor** tm_svd(tm_tensor* t, int* num_outputs) {
    if (!t || t->ndim != 2 || !num_outputs) return NULL;

    int m = (int)t->shape[0];
    int n = (int)t->shape[1];
    int k = (m < n) ? m : n;  /* Number of singular values */

    *num_outputs = 3;
    tm_tensor** result = (tm_tensor**)malloc(3 * sizeof(tm_tensor*));
    if (!result) return NULL;

    /* Allocate output tensors */
    int64_t u_shape[] = {m, k};
    int64_t s_shape[] = {k};
    int64_t v_shape[] = {n, k};

    result[0] = tm_zeros(u_shape, 2, TM_F32);  /* U */
    result[1] = tm_zeros(s_shape, 1, TM_F32);  /* S (singular values) */
    result[2] = tm_zeros(v_shape, 2, TM_F32);  /* V */

    if (!result[0] || !result[1] || !result[2]) {
        if (result[0]) tm_free(result[0]);
        if (result[1]) tm_free(result[1]);
        if (result[2]) tm_free(result[2]);
        free(result);
        return NULL;
    }

    float* A = tm_data_f32(t);
    float* U = tm_data_f32(result[0]);
    float* S = tm_data_f32(result[1]);
    float* V = tm_data_f32(result[2]);

    /* Copy A for deflation */
    float* Acopy = (float*)malloc(m * n * sizeof(float));
    if (!Acopy) {
        tm_free(result[0]);
        tm_free(result[1]);
        tm_free(result[2]);
        free(result);
        return NULL;
    }
    memcpy(Acopy, A, m * n * sizeof(float));

    /* Power iteration for each singular value */
    float* v = (float*)malloc(n * sizeof(float));
    float* u = (float*)malloc(m * sizeof(float));

    for (int i = 0; i < k; i++) {
        /* Initialize v randomly */
        for (int j = 0; j < n; j++) {
            v[j] = (float)rand() / RAND_MAX - 0.5f;
        }

        /* Power iteration */
        for (int iter = 0; iter < 100; iter++) {
            /* u = A * v */
            for (int row = 0; row < m; row++) {
                u[row] = 0;
                for (int col = 0; col < n; col++) {
                    u[row] += Acopy[row * n + col] * v[col];
                }
            }

            /* Normalize u */
            float norm_u = 0;
            for (int j = 0; j < m; j++) norm_u += u[j] * u[j];
            norm_u = sqrtf(norm_u);
            if (norm_u > 1e-10f) {
                for (int j = 0; j < m; j++) u[j] /= norm_u;
            }

            /* v = A^T * u */
            for (int col = 0; col < n; col++) {
                v[col] = 0;
                for (int row = 0; row < m; row++) {
                    v[col] += Acopy[row * n + col] * u[row];
                }
            }

            /* Normalize v */
            float norm_v = 0;
            for (int j = 0; j < n; j++) norm_v += v[j] * v[j];
            norm_v = sqrtf(norm_v);
            if (norm_v > 1e-10f) {
                for (int j = 0; j < n; j++) v[j] /= norm_v;
            }
        }

        /* Compute singular value: sigma = ||A * v|| */
        float sigma = 0;
        for (int row = 0; row < m; row++) {
            float val = 0;
            for (int col = 0; col < n; col++) {
                val += Acopy[row * n + col] * v[col];
            }
            sigma += val * val;
        }
        sigma = sqrtf(sigma);

        /* Store results */
        S[i] = sigma;
        for (int j = 0; j < m; j++) U[j * k + i] = u[j];
        for (int j = 0; j < n; j++) V[j * k + i] = v[j];

        /* Deflate: A = A - sigma * u * v^T */
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                Acopy[row * n + col] -= sigma * u[row] * v[col];
            }
        }
    }

    free(v);
    free(u);
    free(Acopy);

    return result;
}

/* Eigenvalues using QR iteration (simplified - for symmetric matrices)
 * Returns eigenvalues only; production would use LAPACK */
tm_tensor* tm_eig(tm_tensor* t) {
    if (!t || t->ndim != 2) return NULL;
    if (t->shape[0] != t->shape[1]) return NULL;

    int n = (int)t->shape[0];

    /* Copy matrix */
    float* A = (float*)malloc(n * n * sizeof(float));
    if (!A) return NULL;
    memcpy(A, tm_data_f32(t), n * n * sizeof(float));

    float* Q = (float*)malloc(n * n * sizeof(float));
    float* R = (float*)malloc(n * n * sizeof(float));
    if (!Q || !R) {
        free(A);
        if (Q) free(Q);
        if (R) free(R);
        return NULL;
    }

    /* QR iteration */
    for (int iter = 0; iter < 100; iter++) {
        /* QR decomposition using Gram-Schmidt */
        memset(Q, 0, n * n * sizeof(float));
        memset(R, 0, n * n * sizeof(float));

        for (int j = 0; j < n; j++) {
            /* Copy column j of A to Q */
            for (int i = 0; i < n; i++) {
                Q[i * n + j] = A[i * n + j];
            }

            /* Orthogonalize against previous columns */
            for (int k = 0; k < j; k++) {
                float dot = 0;
                for (int i = 0; i < n; i++) {
                    dot += Q[i * n + k] * A[i * n + j];
                }
                R[k * n + j] = dot;
                for (int i = 0; i < n; i++) {
                    Q[i * n + j] -= dot * Q[i * n + k];
                }
            }

            /* Normalize */
            float norm = 0;
            for (int i = 0; i < n; i++) {
                norm += Q[i * n + j] * Q[i * n + j];
            }
            norm = sqrtf(norm);
            R[j * n + j] = norm;
            if (norm > 1e-10f) {
                for (int i = 0; i < n; i++) {
                    Q[i * n + j] /= norm;
                }
            }
        }

        /* A = R * Q */
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = 0;
                for (int k = 0; k < n; k++) {
                    A[i * n + j] += R[i * n + k] * Q[k * n + j];
                }
            }
        }
    }

    /* Extract eigenvalues from diagonal */
    int64_t shape[] = {n};
    tm_tensor* eig = tm_empty(shape, 1, TM_F32);
    if (!eig) {
        free(A);
        free(Q);
        free(R);
        return NULL;
    }

    float* ed = tm_data_f32(eig);
    for (int i = 0; i < n; i++) {
        ed[i] = A[i * n + i];
    }

    free(A);
    free(Q);
    free(R);

    return eig;
}

/* ============================================================= */
/* Autograd                                                      */
/* ============================================================= */

void tm_backward(tm_tensor* loss) {
    if (!loss || !loss->requires_grad) return;

    /* Initialize gradient of loss to 1 */
    if (!loss->grad) {
        loss->grad = tm_ones(loss->shape, loss->ndim, loss->dtype);
    }

    /* Simple topological sort using DFS */
    /* For now, just call backward on this node */
    if (loss->grad_fn && loss->grad_fn->backward) {
        loss->grad_fn->backward(loss);

        /* Recursively backward through inputs */
        for (int i = 0; i < loss->grad_fn->num_inputs; i++) {
            tm_tensor* input = loss->grad_fn->inputs[i];
            if (input && input->requires_grad && input->grad_fn) {
                tm_backward(input);
            }
        }
    }
}

void tm_zero_grad(tm_tensor** params, int n) {
    for (int i = 0; i < n; i++) {
        if (params[i] && params[i]->grad) {
            memset(params[i]->grad->data, 0,
                   params[i]->grad->numel * tm_dtype_size(params[i]->grad->dtype));
        }
    }
}

/* Retain gradient for non-leaf tensors */
void tm_retain_grad(tm_tensor* t) {
    if (!t) return;
    /* Mark tensor to keep its gradient even if it's not a leaf */
    /* In our implementation, we always keep gradients, so this is a no-op
     * but we provide it for API compatibility */
    t->requires_grad = true;
}

/* Compute gradient of outputs w.r.t. inputs explicitly
 * Returns array of gradient tensors (one per input) */
tm_tensor** tm_grad(tm_tensor** outputs, int num_outputs, tm_tensor** inputs, int num_inputs,
                    tm_tensor** grad_outputs) {
    if (!outputs || !inputs || num_outputs <= 0 || num_inputs <= 0) return NULL;

    tm_tensor** grads = (tm_tensor**)calloc(num_inputs, sizeof(tm_tensor*));
    if (!grads) return NULL;

    /* Initialize gradients for outputs */
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i]) {
            if (grad_outputs && grad_outputs[i]) {
                outputs[i]->grad = tm_clone(grad_outputs[i]);
            } else {
                outputs[i]->grad = tm_ones(outputs[i]->shape, outputs[i]->ndim, outputs[i]->dtype);
            }
        }
    }

    /* Run backward on each output */
    for (int i = 0; i < num_outputs; i++) {
        if (outputs[i] && outputs[i]->requires_grad) {
            tm_backward(outputs[i]);
        }
    }

    /* Collect gradients for inputs */
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i] && inputs[i]->grad) {
            grads[i] = tm_clone(inputs[i]->grad);
        } else {
            grads[i] = NULL;
        }
    }

    return grads;
}

/* Gradient checkpointing: wrap a function to recompute during backward
 * This is a simplified version - full implementation would need function pointers
 * For now, we provide the API structure */
typedef struct {
    tm_tensor* (*forward_fn)(tm_tensor** inputs, int num_inputs, void* ctx);
    void* ctx;
    tm_tensor** saved_inputs;
    int num_inputs;
} tm_checkpoint_ctx;

/* Note: Full gradient checkpointing requires more infrastructure.
 * This provides the basic API; GPU testing needed for memory savings verification. */
tm_tensor* tm_checkpoint(tm_tensor* (*fn)(tm_tensor**, int, void*),
                         tm_tensor** inputs, int num_inputs, void* ctx) {
    if (!fn || !inputs || num_inputs <= 0) return NULL;

    /* For CPU, just run the function directly
     * Memory savings only significant on GPU where we'd clear activations */
    return fn(inputs, num_inputs, ctx);
}

/* ============================================================= */
/* In-place Operations                                           */
/* ============================================================= */

tm_tensor* tm_zeros_like(tm_tensor* t) {
    if (!t) return NULL;
    return tm_zeros(t->shape, t->ndim, t->dtype);
}

tm_tensor* tm_ones_like(tm_tensor* t) {
    if (!t) return NULL;
    return tm_ones(t->shape, t->ndim, t->dtype);
}

void tm_fill_(tm_tensor* t, float value) {
    if (!t) return;
    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        data[i] = value;
    }
}

/* Forward declaration for in-place autograd context */
typedef struct tm_inplace_ctx_early {
    tm_tensor* saved_input;
    tm_tensor* other;
    float scalar;
} tm_inplace_ctx_early;

/* Backward for in-place add: a += b
 * grad_a_old = grad_a_new, grad_b = grad_a_new */
static void backward_add_inplace(tm_tensor* a) {
    if (!a || !a->grad_fn || !a->grad) return;
    tm_inplace_ctx_early* ctx = (tm_inplace_ctx_early*)a->grad_fn->ctx;
    tm_tensor* b = ctx->other;

    if (b && b->requires_grad) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = (float*)b->grad->data;
        float* ag = (float*)a->grad->data;
        for (int64_t i = 0; i < b->numel; i++) {
            bg[i] += ag[i];
        }
    }
}

void tm_add_(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return;
    if (a->numel != b->numel) return;

    bool track = grad_enabled && (a->requires_grad || (b && b->requires_grad));

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int64_t i = 0; i < a->numel; i++) {
        a_data[i] += b_data[i];
    }

    if (track) {
        if (a->grad_fn) {
            if (a->grad_fn->ctx) {
                tm_inplace_ctx_early* old_ctx = (tm_inplace_ctx_early*)a->grad_fn->ctx;
                if (old_ctx->other) tm_free(old_ctx->other);
                free(old_ctx);
            }
            free(a->grad_fn);
        }
        a->grad_fn = create_grad_fn(backward_add_inplace);
        tm_inplace_ctx_early* ctx = (tm_inplace_ctx_early*)calloc(1, sizeof(tm_inplace_ctx_early));
        ctx->other = tm_retain(b);
        a->grad_fn->ctx = ctx;
        a->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place scale: a *= s (scalar)
 * grad_a_old = grad_a_new * s */
static void backward_scale_inplace(tm_tensor* t) {
    if (!t || !t->grad_fn || !t->grad || !t->requires_grad) return;
    tm_inplace_ctx_early* ctx = (tm_inplace_ctx_early*)t->grad_fn->ctx;
    float s = ctx->scalar;

    float* tg = (float*)t->grad->data;
    for (int64_t i = 0; i < t->numel; i++) {
        tg[i] *= s;
    }
}

void tm_scale_(tm_tensor* t, float s) {
    if (!t) return;

    bool track = grad_enabled && t->requires_grad;

    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        data[i] *= s;
    }

    if (track) {
        if (t->grad_fn) {
            if (t->grad_fn->ctx) free(t->grad_fn->ctx);
            free(t->grad_fn);
        }
        t->grad_fn = create_grad_fn(backward_scale_inplace);
        tm_inplace_ctx_early* ctx = (tm_inplace_ctx_early*)calloc(1, sizeof(tm_inplace_ctx_early));
        ctx->scalar = s;
        t->grad_fn->ctx = ctx;
        t->grad_fn->num_inputs = 0;
    }
}

void tm_zero_(tm_tensor* t) {
    if (!t || !t->data) return;
    memset(t->data, 0, t->numel * tm_dtype_size(t->dtype));
}

void tm_copy_(tm_tensor* dst, tm_tensor* src) {
    if (!dst || !src) return;
    if (dst->numel != src->numel) return;
    if (dst->dtype != src->dtype) return;
    memcpy(dst->data, src->data, dst->numel * tm_dtype_size(dst->dtype));
}

void tm_clamp_(tm_tensor* t, float min_val, float max_val) {
    if (!t) return;

    if (t->dtype == TM_F32) {
        float* data = (float*)t->data;
        for (int64_t i = 0; i < t->numel; i++) {
            if (data[i] < min_val) data[i] = min_val;
            if (data[i] > max_val) data[i] = max_val;
        }
    }
}

/* ============================================================= */
/* Utility                                                       */
/* ============================================================= */

void tm_print(tm_tensor* t) {
    if (!t) {
        printf("tensor(NULL)\n");
        return;
    }

    printf("tensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld%s", (long long)t->shape[i], i < t->ndim - 1 ? ", " : "");
    }
    printf("], dtype=%s, requires_grad=%s)\n",
           dtype_name(t->dtype),
           t->requires_grad ? "true" : "false");
}

void tm_print_data(tm_tensor* t) {
    if (!t) {
        printf("tensor(NULL)\n");
        return;
    }

    tm_print(t);

    if (t->dtype == TM_F32) {
        float* data = (float*)t->data;
        printf("[");
        for (int64_t i = 0; i < t->numel && i < 10; i++) {
            printf("%.4f%s", data[i], i < t->numel - 1 ? ", " : "");
        }
        if (t->numel > 10) printf(", ...");
        printf("]\n");
    }
}

/* ============================================================= */
/* Matrix Operations                                             */
/* ============================================================= */

tm_tensor* tm_mm(tm_tensor* a, tm_tensor* b) {
    /* Matrix-matrix multiplication for 2D tensors */
    if (!a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;

    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];

    int64_t shape[2] = {m, n};
    tm_tensor* result = tm_zeros(shape, 2, a->dtype);
    if (!result) return NULL;

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* r_data = (float*)result->data;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (int64_t kk = 0; kk < k; kk++) {
                sum += (double)a_data[i * k + kk] * (double)b_data[kk * n + j];
            }
            r_data[i * n + j] = (float)sum;
        }
    }

    return result;
}

tm_tensor* tm_addmm(tm_tensor* t, tm_tensor* a, tm_tensor* b, double beta, double alpha) {
    /* out = beta * t + alpha * (a @ b) */
    if (!t || !a || !b) return NULL;
    if (a->ndim != 2 || b->ndim != 2) return NULL;
    if (a->shape[1] != b->shape[0]) return NULL;

    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];

    /* t can be 1D [n] (bias) or 2D [m, n] */
    bool t_is_1d = (t->ndim == 1 && t->shape[0] == n);
    bool t_is_2d = (t->ndim == 2 && t->shape[0] == m && t->shape[1] == n);
    if (!t_is_1d && !t_is_2d) return NULL;

    int64_t shape[2] = {m, n};
    tm_tensor* result = tm_empty(shape, 2, a->dtype);
    if (!result) return NULL;

    float* t_data = (float*)t->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* r_data = (float*)result->data;

#ifdef USE_BLAS
    /* Fill result with beta * t (handling 1D broadcast) */
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float t_val = t_is_1d ? t_data[j] : t_data[i * n + j];
            r_data[i * n + j] = (float)(beta * t_val);
        }
    }
    /* result += alpha * (a @ b) */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                (float)alpha, a_data, (int)k, b_data, (int)n, 1.0f, r_data, (int)n);
#else
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (int64_t kk = 0; kk < k; kk++) {
                sum += (double)a_data[i * k + kk] * (double)b_data[kk * n + j];
            }
            /* Handle 1D bias (broadcast) vs 2D */
            float t_val = t_is_1d ? t_data[j] : t_data[i * n + j];
            r_data[i * n + j] = (float)(beta * t_val + alpha * sum);
        }
    }
#endif

    return result;
}

tm_tensor* tm_dot(tm_tensor* a, tm_tensor* b) {
    /* Dot product of two 1D tensors */
    if (!a || !b) return NULL;
    if (a->ndim != 1 || b->ndim != 1) return NULL;
    if (a->numel != b->numel) return NULL;

    int64_t shape[1] = {1};
    tm_tensor* result = tm_empty(shape, 0, a->dtype);  /* scalar */
    if (!result) return NULL;

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;

#ifdef USE_BLAS
    ((float*)result->data)[0] = cblas_sdot((int)a->numel, a_data, 1, b_data, 1);
#else
    double sum = 0.0;
    for (int64_t i = 0; i < a->numel; i++) {
        sum += (double)a_data[i] * (double)b_data[i];
    }
    ((float*)result->data)[0] = (float)sum;
#endif
    return result;
}

tm_tensor* tm_mv(tm_tensor* mat, tm_tensor* vec) {
    /* Matrix-vector multiplication */
    if (!mat || !vec) return NULL;
    if (mat->ndim != 2 || vec->ndim != 1) return NULL;
    if (mat->shape[1] != vec->numel) return NULL;

    int64_t m = mat->shape[0];
    int64_t n = mat->shape[1];

    int64_t shape[1] = {m};
    tm_tensor* result = tm_zeros(shape, 1, mat->dtype);
    if (!result) return NULL;

    float* mat_data = (float*)mat->data;
    float* vec_data = (float*)vec->data;
    float* r_data = (float*)result->data;

#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                (int)m, (int)n,
                1.0f, mat_data, (int)n, vec_data, 1, 0.0f, r_data, 1);
#else
    for (int64_t i = 0; i < m; i++) {
        double sum = 0.0;
        for (int64_t j = 0; j < n; j++) {
            sum += (double)mat_data[i * n + j] * (double)vec_data[j];
        }
        r_data[i] = (float)sum;
    }
#endif

    return result;
}

tm_tensor* tm_outer(tm_tensor* a, tm_tensor* b) {
    /* Outer product of two 1D tensors */
    if (!a || !b) return NULL;
    if (a->ndim != 1 || b->ndim != 1) return NULL;

    int64_t m = a->numel;
    int64_t n = b->numel;

    int64_t shape[2] = {m, n};
    tm_tensor* result = tm_empty(shape, 2, a->dtype);
    if (!result) return NULL;

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    float* r_data = (float*)result->data;

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            r_data[i * n + j] = a_data[i] * b_data[j];
        }
    }

    return result;
}

tm_tensor* tm_inner(tm_tensor* a, tm_tensor* b) {
    /* Inner product of two tensors (generalized dot product on last dimensions) */
    if (!a || !b) return NULL;

    /* For 1D tensors, this is the same as dot product */
    if (a->ndim == 1 && b->ndim == 1) {
        return tm_dot(a, b);
    }

    /* For higher dimensional tensors, contract the last dimensions */
    /* a: [..., n], b: [..., n] -> [...] */
    if (a->shape[a->ndim - 1] != b->shape[b->ndim - 1]) return NULL;

    int64_t n = a->shape[a->ndim - 1];

    /* Compute output shape (all dims except last from a, then all dims except last from b) */
    int out_ndim = a->ndim - 1 + b->ndim - 1;
    if (out_ndim > TM_MAX_DIMS) return NULL;

    int64_t out_shape[TM_MAX_DIMS];
    int idx = 0;
    for (int i = 0; i < a->ndim - 1; i++) {
        out_shape[idx++] = a->shape[i];
    }
    for (int i = 0; i < b->ndim - 1; i++) {
        out_shape[idx++] = b->shape[i];
    }

    if (out_ndim == 0) {
        /* Both are 1D, handled above, but just in case: scalar result */
        return tm_dot(a, b);
    }

    tm_tensor* result = tm_zeros(out_shape, out_ndim, a->dtype);
    if (!result) return NULL;

    /* For 2D case: inner(a, b) where a is [m, n], b is [k, n] -> [m, k] */
    if (a->ndim == 2 && b->ndim == 2) {
        int64_t m = a->shape[0];
        int64_t k = b->shape[0];
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* r_data = (float*)result->data;

        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < k; j++) {
                double sum = 0.0;
                for (int64_t l = 0; l < n; l++) {
                    sum += (double)a_data[i * n + l] * (double)b_data[j * n + l];
                }
                r_data[i * k + j] = (float)sum;
            }
        }
    }

    return result;
}

/* ============================================================= */
/* Remaining In-place Operations (with autograd support)         */
/* ============================================================= */

/* Context for in-place operations that need saved values */
typedef struct {
    tm_tensor* saved_input;   /* Original input values (for ops that need it) */
    tm_tensor* other;         /* Other operand for binary ops */
} tm_inplace_ctx;

/* Backward for in-place sub: a -= b means a_new = a_old - b
 * grad_a_old = grad_a_new, grad_b = -grad_a_new */
static void backward_sub_inplace(tm_tensor* a) {
    if (!a || !a->grad_fn || !a->grad) return;
    tm_inplace_ctx* ctx = (tm_inplace_ctx*)a->grad_fn->ctx;
    tm_tensor* b = ctx->other;

    if (b && b->requires_grad) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = (float*)b->grad->data;
        float* ag = (float*)a->grad->data;
        for (int64_t i = 0; i < b->numel; i++) {
            bg[i] -= ag[i];  /* grad_b = -grad_a */
        }
    }
}

void tm_sub_(tm_tensor* a, tm_tensor* b) {
    if (!a || !b || a->numel != b->numel) return;

    /* Setup autograd if needed */
    bool track = grad_enabled && (a->requires_grad || (b && b->requires_grad));

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int64_t i = 0; i < a->numel; i++) {
        a_data[i] -= b_data[i];
    }

    if (track) {
        /* Free old grad_fn if exists */
        if (a->grad_fn) {
            if (a->grad_fn->ctx) free(a->grad_fn->ctx);
            free(a->grad_fn);
        }
        a->grad_fn = create_grad_fn(backward_sub_inplace);
        tm_inplace_ctx* ctx = (tm_inplace_ctx*)calloc(1, sizeof(tm_inplace_ctx));
        ctx->other = tm_retain(b);
        a->grad_fn->ctx = ctx;
        a->grad_fn->num_inputs = 0;  /* We handle inputs via ctx */
    }
}

/* Backward for in-place mul: a *= b means a_new = a_old * b
 * grad_a_old = grad_a_new * b, grad_b = grad_a_new * a_old */
static void backward_mul_inplace(tm_tensor* a) {
    if (!a || !a->grad_fn || !a->grad) return;
    tm_inplace_ctx* ctx = (tm_inplace_ctx*)a->grad_fn->ctx;
    tm_tensor* saved = ctx->saved_input;  /* Original a values */
    tm_tensor* b = ctx->other;

    float* ag = (float*)a->grad->data;

    /* For a's gradient, we need to scale by b, but since this is in-place,
     * we're modifying a's gradient to be the gradient w.r.t. original a */
    if (a->requires_grad && b) {
        float* bd = (float*)b->data;
        /* grad_a_old = grad_a_new * b */
        for (int64_t i = 0; i < a->numel; i++) {
            ag[i] *= bd[i];
        }
    }

    /* For b's gradient */
    if (b && b->requires_grad && saved) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = (float*)b->grad->data;
        float* sd = (float*)saved->data;
        float* orig_ag = (float*)a->grad->data;
        /* grad_b = grad_a_new * a_old, but we already modified ag, so use saved */
        /* Actually we need the original gradient before we modified it */
        /* This is tricky - let's recompute: grad_b = (ag / b) * a_old = ag_orig * a_old */
        for (int64_t i = 0; i < b->numel; i++) {
            /* ag was multiplied by bd[i], so ag_orig = ag / bd[i] (if bd[i] != 0) */
            float bd_val = ((float*)b->data)[i];
            float ag_orig = (bd_val != 0.0f) ? orig_ag[i] / bd_val : 0.0f;
            bg[i] += ag_orig * sd[i];
        }
    }
}

void tm_mul_(tm_tensor* a, tm_tensor* b) {
    if (!a || !b || a->numel != b->numel) return;

    bool track = grad_enabled && (a->requires_grad || (b && b->requires_grad));
    tm_tensor* saved = NULL;

    /* Save original values if tracking grad and b needs gradient */
    if (track && b && b->requires_grad) {
        saved = tm_clone(a);
    }

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int64_t i = 0; i < a->numel; i++) {
        a_data[i] *= b_data[i];
    }

    if (track) {
        if (a->grad_fn) {
            if (a->grad_fn->ctx) {
                tm_inplace_ctx* old_ctx = (tm_inplace_ctx*)a->grad_fn->ctx;
                if (old_ctx->saved_input) tm_free(old_ctx->saved_input);
                if (old_ctx->other) tm_free(old_ctx->other);
                free(old_ctx);
            }
            free(a->grad_fn);
        }
        a->grad_fn = create_grad_fn(backward_mul_inplace);
        tm_inplace_ctx* ctx = (tm_inplace_ctx*)calloc(1, sizeof(tm_inplace_ctx));
        ctx->saved_input = saved;
        ctx->other = tm_retain(b);
        a->grad_fn->ctx = ctx;
        a->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place div: a /= b means a_new = a_old / b
 * grad_a_old = grad_a_new / b, grad_b = -grad_a_new * a_old / b^2 = -grad_a_new * a_new / b */
static void backward_div_inplace(tm_tensor* a) {
    if (!a || !a->grad_fn || !a->grad) return;
    tm_inplace_ctx* ctx = (tm_inplace_ctx*)a->grad_fn->ctx;
    tm_tensor* b = ctx->other;

    float* ag = (float*)a->grad->data;
    float* ad = (float*)a->data;  /* This is a_new = a_old / b */

    if (b && b->requires_grad) {
        if (!b->grad) b->grad = tm_zeros(b->shape, b->ndim, b->dtype);
        float* bg = (float*)b->grad->data;
        float* bd = (float*)b->data;
        /* grad_b = -grad_a_new * a_new / b */
        for (int64_t i = 0; i < b->numel; i++) {
            bg[i] -= ag[i] * ad[i] / bd[i];
        }
    }

    /* Modify a's gradient: grad_a_old = grad_a_new / b */
    if (a->requires_grad && b) {
        float* bd = (float*)b->data;
        for (int64_t i = 0; i < a->numel; i++) {
            ag[i] /= bd[i];
        }
    }
}

void tm_div_(tm_tensor* a, tm_tensor* b) {
    if (!a || !b || a->numel != b->numel) return;

    bool track = grad_enabled && (a->requires_grad || (b && b->requires_grad));

    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int64_t i = 0; i < a->numel; i++) {
        a_data[i] /= b_data[i];
    }

    if (track) {
        if (a->grad_fn) {
            if (a->grad_fn->ctx) {
                tm_inplace_ctx* old_ctx = (tm_inplace_ctx*)a->grad_fn->ctx;
                if (old_ctx->saved_input) tm_free(old_ctx->saved_input);
                if (old_ctx->other) tm_free(old_ctx->other);
                free(old_ctx);
            }
            free(a->grad_fn);
        }
        a->grad_fn = create_grad_fn(backward_div_inplace);
        tm_inplace_ctx* ctx = (tm_inplace_ctx*)calloc(1, sizeof(tm_inplace_ctx));
        ctx->other = tm_retain(b);
        a->grad_fn->ctx = ctx;
        a->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place exp: a = exp(a_old)
 * grad_a_old = grad_a_new * exp(a_old) = grad_a_new * a_new */
static void backward_exp_inplace(tm_tensor* t) {
    if (!t || !t->grad_fn || !t->grad || !t->requires_grad) return;

    float* tg = (float*)t->grad->data;
    float* td = (float*)t->data;  /* t_new = exp(t_old) */

    /* grad_t_old = grad_t_new * t_new */
    for (int64_t i = 0; i < t->numel; i++) {
        tg[i] *= td[i];
    }
}

void tm_exp_(tm_tensor* t) {
    if (!t) return;

    bool track = grad_enabled && t->requires_grad;

    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        data[i] = expf(data[i]);
    }

    if (track) {
        if (t->grad_fn) {
            if (t->grad_fn->ctx) free(t->grad_fn->ctx);
            free(t->grad_fn);
        }
        t->grad_fn = create_grad_fn(backward_exp_inplace);
        t->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place log: a = log(a_old)
 * grad_a_old = grad_a_new / a_old = grad_a_new / exp(a_new) */
static void backward_log_inplace(tm_tensor* t) {
    if (!t || !t->grad_fn || !t->grad || !t->requires_grad) return;
    tm_inplace_ctx* ctx = (tm_inplace_ctx*)t->grad_fn->ctx;
    tm_tensor* saved = ctx ? ctx->saved_input : NULL;

    if (!saved) return;

    float* tg = (float*)t->grad->data;
    float* sd = (float*)saved->data;  /* Original values */

    /* grad_t_old = grad_t_new / t_old */
    for (int64_t i = 0; i < t->numel; i++) {
        tg[i] /= sd[i];
    }
}

void tm_log_(tm_tensor* t) {
    if (!t) return;

    bool track = grad_enabled && t->requires_grad;
    tm_tensor* saved = NULL;

    if (track) {
        saved = tm_clone(t);  /* Save original for backward */
    }

    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        data[i] = logf(data[i]);
    }

    if (track) {
        if (t->grad_fn) {
            if (t->grad_fn->ctx) {
                tm_inplace_ctx* old_ctx = (tm_inplace_ctx*)t->grad_fn->ctx;
                if (old_ctx->saved_input) tm_free(old_ctx->saved_input);
                free(old_ctx);
            }
            free(t->grad_fn);
        }
        t->grad_fn = create_grad_fn(backward_log_inplace);
        tm_inplace_ctx* ctx = (tm_inplace_ctx*)calloc(1, sizeof(tm_inplace_ctx));
        ctx->saved_input = saved;
        t->grad_fn->ctx = ctx;
        t->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place sqrt: a = sqrt(a_old)
 * grad_a_old = grad_a_new / (2 * sqrt(a_old)) = grad_a_new / (2 * a_new) */
static void backward_sqrt_inplace(tm_tensor* t) {
    if (!t || !t->grad_fn || !t->grad || !t->requires_grad) return;

    float* tg = (float*)t->grad->data;
    float* td = (float*)t->data;  /* t_new = sqrt(t_old) */

    /* grad_t_old = grad_t_new / (2 * t_new) */
    for (int64_t i = 0; i < t->numel; i++) {
        tg[i] /= (2.0f * td[i]);
    }
}

void tm_sqrt_(tm_tensor* t) {
    if (!t) return;

    bool track = grad_enabled && t->requires_grad;

    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        data[i] = sqrtf(data[i]);
    }

    if (track) {
        if (t->grad_fn) {
            if (t->grad_fn->ctx) free(t->grad_fn->ctx);
            free(t->grad_fn);
        }
        t->grad_fn = create_grad_fn(backward_sqrt_inplace);
        t->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place abs: a = |a_old|
 * grad_a_old = grad_a_new * sign(a_old) */
static void backward_abs_inplace(tm_tensor* t) {
    if (!t || !t->grad_fn || !t->grad || !t->requires_grad) return;
    tm_inplace_ctx* ctx = (tm_inplace_ctx*)t->grad_fn->ctx;
    tm_tensor* saved = ctx ? ctx->saved_input : NULL;

    if (!saved) return;

    float* tg = (float*)t->grad->data;
    float* sd = (float*)saved->data;

    for (int64_t i = 0; i < t->numel; i++) {
        tg[i] *= (sd[i] >= 0.0f) ? 1.0f : -1.0f;
    }
}

void tm_abs_(tm_tensor* t) {
    if (!t) return;

    bool track = grad_enabled && t->requires_grad;
    tm_tensor* saved = NULL;

    if (track) {
        saved = tm_clone(t);
    }

    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        data[i] = fabsf(data[i]);
    }

    if (track) {
        if (t->grad_fn) {
            if (t->grad_fn->ctx) {
                tm_inplace_ctx* old_ctx = (tm_inplace_ctx*)t->grad_fn->ctx;
                if (old_ctx->saved_input) tm_free(old_ctx->saved_input);
                free(old_ctx);
            }
            free(t->grad_fn);
        }
        t->grad_fn = create_grad_fn(backward_abs_inplace);
        tm_inplace_ctx* ctx = (tm_inplace_ctx*)calloc(1, sizeof(tm_inplace_ctx));
        ctx->saved_input = saved;
        t->grad_fn->ctx = ctx;
        t->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place neg: a = -a_old
 * grad_a_old = -grad_a_new */
static void backward_neg_inplace(tm_tensor* t) {
    if (!t || !t->grad_fn || !t->grad || !t->requires_grad) return;

    float* tg = (float*)t->grad->data;
    for (int64_t i = 0; i < t->numel; i++) {
        tg[i] = -tg[i];
    }
}

void tm_neg_(tm_tensor* t) {
    if (!t) return;

    bool track = grad_enabled && t->requires_grad;

    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        data[i] = -data[i];
    }

    if (track) {
        if (t->grad_fn) {
            if (t->grad_fn->ctx) free(t->grad_fn->ctx);
            free(t->grad_fn);
        }
        t->grad_fn = create_grad_fn(backward_neg_inplace);
        t->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place relu: a = max(0, a_old)
 * grad_a_old = grad_a_new if a_old > 0 else 0 */
static void backward_relu_inplace(tm_tensor* t) {
    if (!t || !t->grad_fn || !t->grad || !t->requires_grad) return;

    float* tg = (float*)t->grad->data;
    float* td = (float*)t->data;

    /* If t_new > 0, then t_old > 0, so grad passes through */
    for (int64_t i = 0; i < t->numel; i++) {
        if (td[i] <= 0.0f) tg[i] = 0.0f;
    }
}

void tm_relu_(tm_tensor* t) {
    if (!t) return;

    bool track = grad_enabled && t->requires_grad;

    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        if (data[i] < 0) data[i] = 0;
    }

    if (track) {
        if (t->grad_fn) {
            if (t->grad_fn->ctx) free(t->grad_fn->ctx);
            free(t->grad_fn);
        }
        t->grad_fn = create_grad_fn(backward_relu_inplace);
        t->grad_fn->num_inputs = 0;
    }
}

/* Backward for in-place sigmoid: a = sigmoid(a_old)
 * grad_a_old = grad_a_new * sigmoid(a_old) * (1 - sigmoid(a_old))
 *            = grad_a_new * a_new * (1 - a_new) */
static void backward_sigmoid_inplace(tm_tensor* t) {
    if (!t || !t->grad_fn || !t->grad || !t->requires_grad) return;

    float* tg = (float*)t->grad->data;
    float* td = (float*)t->data;  /* t_new = sigmoid(t_old) */

    for (int64_t i = 0; i < t->numel; i++) {
        tg[i] *= td[i] * (1.0f - td[i]);
    }
}

void tm_sigmoid_(tm_tensor* t) {
    if (!t) return;

    bool track = grad_enabled && t->requires_grad;

    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->numel; i++) {
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }

    if (track) {
        if (t->grad_fn) {
            if (t->grad_fn->ctx) free(t->grad_fn->ctx);
            free(t->grad_fn);
        }
        t->grad_fn = create_grad_fn(backward_sigmoid_inplace);
        t->grad_fn->num_inputs = 0;
    }
}

/* ============================================================= */
/* Higher-Order Gradients                                        */
/* ============================================================= */

/* Compute Hessian matrix of a scalar loss w.r.t. parameters
 * Uses finite differences for numerical approximation
 * Returns: n x n Hessian matrix where n = total params */
tm_tensor* tm_hessian(tm_tensor** params, int num_params,
                      tm_tensor* (*loss_fn)(tm_tensor**, int, void*), void* ctx) {
    if (!params || num_params <= 0 || !loss_fn) return NULL;

    /* Count total parameters */
    int64_t total_params = 0;
    for (int i = 0; i < num_params; i++) {
        if (params[i]) total_params += params[i]->numel;
    }
    if (total_params == 0) return NULL;

    /* Create Hessian matrix */
    int64_t shape[2] = {total_params, total_params};
    tm_tensor* hessian = tm_zeros(shape, 2, TM_F32);
    if (!hessian) return NULL;

    float* h_data = (float*)hessian->data;
    float eps = 1e-4f;

    /* Compute Hessian via finite differences of gradients */
    /* H[i,j] = (grad_i(x + eps*e_j) - grad_i(x - eps*e_j)) / (2*eps) */

    int64_t col = 0;
    for (int p = 0; p < num_params; p++) {
        if (!params[p]) continue;
        float* p_data = (float*)params[p]->data;

        for (int64_t j = 0; j < params[p]->numel; j++) {
            float orig = p_data[j];

            /* Perturb +eps */
            p_data[j] = orig + eps;
            tm_tensor* loss_plus = loss_fn(params, num_params, ctx);
            if (!loss_plus) { tm_free(hessian); return NULL; }

            /* Zero grads and compute gradients */
            for (int k = 0; k < num_params; k++) {
                if (params[k] && params[k]->grad) {
                    tm_zero_(params[k]->grad);
                }
            }
            tm_backward(loss_plus);

            /* Store positive gradients */
            float* grad_plus = (float*)malloc(total_params * sizeof(float));
            int64_t idx = 0;
            for (int k = 0; k < num_params; k++) {
                if (params[k] && params[k]->grad) {
                    float* g = (float*)params[k]->grad->data;
                    for (int64_t l = 0; l < params[k]->numel; l++) {
                        grad_plus[idx++] = g[l];
                    }
                }
            }
            tm_free(loss_plus);

            /* Perturb -eps */
            p_data[j] = orig - eps;
            tm_tensor* loss_minus = loss_fn(params, num_params, ctx);
            if (!loss_minus) { free(grad_plus); tm_free(hessian); return NULL; }

            /* Zero grads and compute gradients */
            for (int k = 0; k < num_params; k++) {
                if (params[k] && params[k]->grad) {
                    tm_zero_(params[k]->grad);
                }
            }
            tm_backward(loss_minus);

            /* Compute Hessian column: (grad+ - grad-) / (2*eps) */
            idx = 0;
            for (int k = 0; k < num_params; k++) {
                if (params[k] && params[k]->grad) {
                    float* g = (float*)params[k]->grad->data;
                    for (int64_t l = 0; l < params[k]->numel; l++) {
                        float h_val = (grad_plus[idx] - g[l]) / (2.0f * eps);
                        h_data[idx * total_params + col] = h_val;
                        idx++;
                    }
                }
            }

            free(grad_plus);
            tm_free(loss_minus);
            p_data[j] = orig;  /* Restore */
            col++;
        }
    }

    return hessian;
}

/* Hessian-vector product: H @ v without computing full Hessian
 * More memory efficient for large models
 * Uses finite differences: Hv ≈ (grad(x + eps*v) - grad(x - eps*v)) / (2*eps) */
tm_tensor* tm_hvp(tm_tensor** params, int num_params, tm_tensor** vec, int num_vec,
                  tm_tensor* (*loss_fn)(tm_tensor**, int, void*), void* ctx) {
    if (!params || !vec || num_params <= 0 || !loss_fn) return NULL;
    if (num_params != num_vec) return NULL;

    /* Count total parameters */
    int64_t total_params = 0;
    for (int i = 0; i < num_params; i++) {
        if (params[i]) total_params += params[i]->numel;
    }
    if (total_params == 0) return NULL;

    float eps = 1e-4f;

    /* Save original parameters */
    float** orig = (float**)malloc(num_params * sizeof(float*));
    for (int i = 0; i < num_params; i++) {
        if (params[i]) {
            orig[i] = (float*)malloc(params[i]->numel * sizeof(float));
            memcpy(orig[i], params[i]->data, params[i]->numel * sizeof(float));
        } else {
            orig[i] = NULL;
        }
    }

    /* Perturb in direction +eps*v */
    for (int i = 0; i < num_params; i++) {
        if (params[i] && vec[i]) {
            float* pdata = (float*)params[i]->data;
            float* v = (float*)vec[i]->data;
            for (int64_t jj = 0; jj < params[i]->numel; jj++) {
                pdata[jj] = orig[i][jj] + eps * v[jj];
            }
        }
    }

    /* Compute gradient at x + eps*v */
    for (int k = 0; k < num_params; k++) {
        if (params[k] && params[k]->grad) tm_zero_(params[k]->grad);
    }
    tm_tensor* loss_plus = loss_fn(params, num_params, ctx);
    if (!loss_plus) {
        for (int i = 0; i < num_params; i++) free(orig[i]);
        free(orig);
        return NULL;
    }
    tm_backward(loss_plus);

    /* Store grad+ */
    float* grad_plus = (float*)malloc(total_params * sizeof(float));
    int64_t idx = 0;
    for (int i = 0; i < num_params; i++) {
        if (params[i] && params[i]->grad) {
            float* g = (float*)params[i]->grad->data;
            for (int64_t jj = 0; jj < params[i]->numel; jj++) {
                grad_plus[idx++] = g[jj];
            }
        }
    }
    tm_free(loss_plus);

    /* Perturb in direction -eps*v */
    for (int i = 0; i < num_params; i++) {
        if (params[i] && vec[i]) {
            float* pdata = (float*)params[i]->data;
            float* v = (float*)vec[i]->data;
            for (int64_t jj = 0; jj < params[i]->numel; jj++) {
                pdata[jj] = orig[i][jj] - eps * v[jj];
            }
        }
    }

    /* Compute gradient at x - eps*v */
    for (int k = 0; k < num_params; k++) {
        if (params[k] && params[k]->grad) tm_zero_(params[k]->grad);
    }
    tm_tensor* loss_minus = loss_fn(params, num_params, ctx);
    if (!loss_minus) {
        for (int i = 0; i < num_params; i++) free(orig[i]);
        free(orig);
        free(grad_plus);
        return NULL;
    }
    tm_backward(loss_minus);

    /* Compute Hv = (grad+ - grad-) / (2*eps) */
    int64_t hvp_shape[1] = {total_params};
    tm_tensor* hvp_result = tm_zeros(hvp_shape, 1, TM_F32);
    float* hvp_data = (float*)hvp_result->data;

    idx = 0;
    for (int i = 0; i < num_params; i++) {
        if (params[i] && params[i]->grad) {
            float* g = (float*)params[i]->grad->data;
            for (int64_t jj = 0; jj < params[i]->numel; jj++) {
                hvp_data[idx] = (grad_plus[idx] - g[jj]) / (2.0f * eps);
                idx++;
            }
        }
    }

    /* Restore original parameters */
    for (int i = 0; i < num_params; i++) {
        if (params[i] && orig[i]) {
            memcpy(params[i]->data, orig[i], params[i]->numel * sizeof(float));
        }
        free(orig[i]);
    }
    free(orig);
    free(grad_plus);
    tm_free(loss_minus);

    return hvp_result;
}

/* Jacobian-vector product (forward-mode AD)
 * Computes J @ v where J is the Jacobian of outputs w.r.t. inputs
 * Uses finite differences: Jv ≈ (f(x + eps*v) - f(x)) / eps */
tm_tensor** tm_jvp(tm_tensor* (*fn)(tm_tensor**, int, void*), tm_tensor** inputs, int num_inputs,
                   tm_tensor** tangents, void* ctx, int* num_outputs_out) {
    if (!fn || !inputs || !tangents || num_inputs <= 0) return NULL;

    float eps = 1e-4f;

    /* Compute f(x) */
    tm_tensor* out_base = fn(inputs, num_inputs, ctx);
    if (!out_base) return NULL;

    /* Save original inputs */
    float** orig = (float**)malloc(num_inputs * sizeof(float*));
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i]) {
            orig[i] = (float*)malloc(inputs[i]->numel * sizeof(float));
            memcpy(orig[i], inputs[i]->data, inputs[i]->numel * sizeof(float));
        } else {
            orig[i] = NULL;
        }
    }

    /* Perturb inputs by eps * tangent */
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i] && tangents[i]) {
            float* x = (float*)inputs[i]->data;
            float* t = (float*)tangents[i]->data;
            for (int64_t jj = 0; jj < inputs[i]->numel; jj++) {
                x[jj] = orig[i][jj] + eps * t[jj];
            }
        }
    }

    /* Compute f(x + eps*v) */
    tm_tensor* out_perturbed = fn(inputs, num_inputs, ctx);

    /* Restore inputs */
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i] && orig[i]) {
            memcpy(inputs[i]->data, orig[i], inputs[i]->numel * sizeof(float));
        }
        free(orig[i]);
    }
    free(orig);

    if (!out_perturbed) {
        tm_free(out_base);
        return NULL;
    }

    /* Compute JVP = (f(x + eps*v) - f(x)) / eps */
    tm_tensor** jvp_outputs = (tm_tensor**)malloc(sizeof(tm_tensor*));
    jvp_outputs[0] = tm_sub(out_perturbed, out_base);
    if (jvp_outputs[0]) {
        tm_scale_(jvp_outputs[0], 1.0f / eps);
    }

    if (num_outputs_out) *num_outputs_out = 1;

    tm_free(out_base);
    tm_free(out_perturbed);

    return jvp_outputs;
}

/* ============================================================= */
/* Custom Autograd Functions                                     */
/* ============================================================= */

/* Custom autograd function context for saving tensors */
struct tm_custom_ctx {
    tm_tensor** saved_tensors;
    int num_saved;
    void* user_data;
};

/* Create a new custom context */
tm_custom_ctx* tm_custom_ctx_new(void) {
    tm_custom_ctx* ctx_new = (tm_custom_ctx*)calloc(1, sizeof(tm_custom_ctx));
    return ctx_new;
}

/* Save tensor for backward */
void tm_save_for_backward(tm_custom_ctx* save_ctx, tm_tensor* t) {
    if (!save_ctx) return;
    save_ctx->num_saved++;
    save_ctx->saved_tensors = (tm_tensor**)realloc(save_ctx->saved_tensors,
                                               save_ctx->num_saved * sizeof(tm_tensor*));
    save_ctx->saved_tensors[save_ctx->num_saved - 1] = tm_retain(t);
}

/* Get saved tensors */
tm_tensor** tm_get_saved_tensors(tm_custom_ctx* get_ctx, int* num_saved) {
    if (!get_ctx) {
        if (num_saved) *num_saved = 0;
        return NULL;
    }
    if (num_saved) *num_saved = get_ctx->num_saved;
    return get_ctx->saved_tensors;
}

/* Set user data in context */
void tm_custom_ctx_set_data(tm_custom_ctx* set_ctx, void* user_data) {
    if (set_ctx) set_ctx->user_data = user_data;
}

/* Get user data from context */
void* tm_custom_ctx_get_data(tm_custom_ctx* get_ctx) {
    return get_ctx ? get_ctx->user_data : NULL;
}

/* Free custom context */
void tm_custom_ctx_free(tm_custom_ctx* free_ctx) {
    if (!free_ctx) return;
    for (int i = 0; i < free_ctx->num_saved; i++) {
        tm_free(free_ctx->saved_tensors[i]);
    }
    free(free_ctx->saved_tensors);
    free(free_ctx);
}

/* Custom autograd function definition */
struct tm_autograd_function {
    char* name;
    tm_tensor* (*forward)(tm_tensor** inputs, int num_inputs, tm_custom_ctx* fwd_ctx);
    tm_tensor** (*backward)(tm_tensor* grad_output, tm_custom_ctx* bwd_ctx, int* num_grads);
};

/* Registry for custom functions (simple array-based) */
#define TM_MAX_CUSTOM_FUNCTIONS 64
static tm_autograd_function* tm_custom_fn_registry[TM_MAX_CUSTOM_FUNCTIONS];
static int tm_num_custom_fns = 0;

/* Register a custom autograd function */
int tm_register_autograd_function(const char* name,
                                   tm_tensor* (*forward)(tm_tensor**, int, tm_custom_ctx*),
                                   tm_tensor** (*backward)(tm_tensor*, tm_custom_ctx*, int*)) {
    if (tm_num_custom_fns >= TM_MAX_CUSTOM_FUNCTIONS) return -1;
    if (!name || !forward || !backward) return -1;

    tm_autograd_function* fn = (tm_autograd_function*)malloc(sizeof(tm_autograd_function));
    fn->name = strdup(name);
    fn->forward = forward;
    fn->backward = backward;

    tm_custom_fn_registry[tm_num_custom_fns++] = fn;
    return tm_num_custom_fns - 1;  /* Return function ID */
}

/* Find a registered function by name */
static tm_autograd_function* tm_find_autograd_function(const char* name) {
    for (int i = 0; i < tm_num_custom_fns; i++) {
        if (tm_custom_fn_registry[i] && strcmp(tm_custom_fn_registry[i]->name, name) == 0) {
            return tm_custom_fn_registry[i];
        }
    }
    return NULL;
}

/* Context for custom function in grad_fn */
typedef struct {
    tm_autograd_function* fn;
    tm_custom_ctx* custom_ctx;
} tm_custom_grad_ctx;

/* Backward function for custom autograd */
static void custom_backward(tm_tensor* out) {
    if (!out || !out->grad_fn || !out->grad_fn->ctx) return;

    tm_custom_grad_ctx* grad_ctx = (tm_custom_grad_ctx*)out->grad_fn->ctx;
    if (!grad_ctx->fn || !grad_ctx->fn->backward) return;

    int num_grads = 0;
    tm_tensor** input_grads = grad_ctx->fn->backward(out->grad, grad_ctx->custom_ctx, &num_grads);

    /* Distribute gradients to inputs */
    for (int i = 0; i < num_grads && i < out->grad_fn->num_inputs; i++) {
        tm_tensor* input = out->grad_fn->inputs[i];
        if (input && input->requires_grad && input_grads && input_grads[i]) {
            if (!input->grad) {
                input->grad = tm_clone(input_grads[i]);
            } else {
                tm_add_(input->grad, input_grads[i]);
            }
            tm_free(input_grads[i]);
        }
    }
    free(input_grads);
}

/* Apply a custom autograd function by name */
tm_tensor* tm_apply_custom_function(const char* name, tm_tensor** inputs, int num_inputs) {
    if (!name || !inputs || num_inputs <= 0) return NULL;

    tm_autograd_function* fn = tm_find_autograd_function(name);
    if (!fn) return NULL;

    /* Create context for saving tensors */
    tm_custom_ctx* apply_ctx = tm_custom_ctx_new();

    /* Run forward */
    tm_tensor* output = fn->forward(inputs, num_inputs, apply_ctx);
    if (!output) {
        tm_custom_ctx_free(apply_ctx);
        return NULL;
    }

    /* Check if any input requires grad */
    bool any_requires_grad = false;
    for (int i = 0; i < num_inputs; i++) {
        if (inputs[i] && inputs[i]->requires_grad) {
            any_requires_grad = true;
            break;
        }
    }

    /* Setup autograd if needed */
    if (any_requires_grad) {
        output->requires_grad = true;
        output->grad_fn = (tm_grad_fn*)calloc(1, sizeof(tm_grad_fn));
        output->grad_fn->backward = custom_backward;
        output->grad_fn->num_inputs = num_inputs < 2 ? num_inputs : 2;  /* Max 2 in current struct */
        for (int i = 0; i < output->grad_fn->num_inputs; i++) {
            output->grad_fn->inputs[i] = inputs[i];
        }

        /* Store custom context */
        tm_custom_grad_ctx* grad_ctx = (tm_custom_grad_ctx*)malloc(sizeof(tm_custom_grad_ctx));
        grad_ctx->fn = fn;
        grad_ctx->custom_ctx = apply_ctx;
        output->grad_fn->ctx = grad_ctx;
    } else {
        tm_custom_ctx_free(apply_ctx);
    }

    return output;
}

/* Apply a custom autograd function by ID (faster than by name) */
tm_tensor* tm_apply_custom_function_id(int fn_id, tm_tensor** inputs, int num_inputs) {
    if (fn_id < 0 || fn_id >= tm_num_custom_fns) return NULL;
    tm_autograd_function* fn = tm_custom_fn_registry[fn_id];
    if (!fn) return NULL;
    return tm_apply_custom_function(fn->name, inputs, num_inputs);
}

/* Clean up custom function registry */
void tm_cleanup_autograd_registry(void) {
    for (int i = 0; i < tm_num_custom_fns; i++) {
        if (tm_custom_fn_registry[i]) {
            free(tm_custom_fn_registry[i]->name);
            free(tm_custom_fn_registry[i]);
            tm_custom_fn_registry[i] = NULL;
        }
    }
    tm_num_custom_fns = 0;
}

/* ============================================================= */
/* Sparse Tensor Operations                                      */
/* ============================================================= */

/* Check if tensor is sparse */
bool tm_is_sparse(tm_tensor* t) {
    return t ? t->is_sparse : false;
}

/* Get number of non-zero elements */
int64_t tm_sparse_nnz(tm_tensor* t) {
    if (!t || !t->is_sparse) return -1;
    return t->nnz;
}

/* Get sparse format */
tm_sparse_format tm_get_sparse_format(tm_tensor* t) {
    if (!t || !t->is_sparse) return TM_SPARSE_COO;  /* Default */
    return t->sparse_format;
}

/* Create sparse COO tensor from indices and values
 * indices: [2, nnz] tensor of (row, col) pairs
 * values: [nnz] tensor of non-zero values
 * shape: dense shape [rows, cols] */
tm_tensor* tm_sparse_coo(tm_tensor* indices, tm_tensor* values, int64_t* shape, int ndim) {
    if (!indices || !values || !shape || ndim < 2) return NULL;
    if (indices->ndim != 2 || indices->shape[0] != 2) return NULL;
    
    int64_t nnz = indices->shape[1];
    if (values->numel != nnz) return NULL;
    
    /* Create sparse tensor */
    tm_tensor* t = (tm_tensor*)calloc(1, sizeof(tm_tensor));
    if (!t) return NULL;
    
    t->is_sparse = true;
    t->sparse_format = TM_SPARSE_COO;
    t->nnz = nnz;
    t->ndim = ndim;
    t->dtype = values->dtype;
    t->refcount = 1;
    t->owns_data = true;
    
    /* Copy shape */
    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    
    /* Copy indices */
    t->indices = (int64_t*)malloc(2 * nnz * sizeof(int64_t));
    if (!t->indices) {
        free(t);
        return NULL;
    }
    
    /* Copy from indices tensor (assumes I64 or convert) */
    if (indices->dtype == TM_I64) {
        memcpy(t->indices, indices->data, 2 * nnz * sizeof(int64_t));
    } else if (indices->dtype == TM_I32) {
        int32_t* idx_data = (int32_t*)indices->data;
        for (int64_t i = 0; i < 2 * nnz; i++) {
            t->indices[i] = idx_data[i];
        }
    } else {
        /* Try to convert from float */
        float* idx_data = (float*)indices->data;
        for (int64_t i = 0; i < 2 * nnz; i++) {
            t->indices[i] = (int64_t)idx_data[i];
        }
    }
    
    /* Copy values */
    size_t val_size = nnz * tm_dtype_size(values->dtype);
    t->data = malloc(val_size);
    if (!t->data) {
        free(t->indices);
        free(t);
        return NULL;
    }
    memcpy(t->data, values->data, val_size);
    
    return t;
}

/* Create sparse CSR tensor from row_ptr, col_ind, and values */
tm_tensor* tm_sparse_csr(tm_tensor* row_ptr, tm_tensor* col_ind, tm_tensor* values,
                         int64_t* shape, int ndim) {
    if (!row_ptr || !col_ind || !values || !shape || ndim < 2) return NULL;
    
    int64_t rows = shape[0];
    int64_t nnz = values->numel;
    
    if (row_ptr->numel != rows + 1 || col_ind->numel != nnz) return NULL;
    
    /* Create sparse tensor */
    tm_tensor* t = (tm_tensor*)calloc(1, sizeof(tm_tensor));
    if (!t) return NULL;
    
    t->is_sparse = true;
    t->sparse_format = TM_SPARSE_CSR;
    t->nnz = nnz;
    t->ndim = ndim;
    t->dtype = values->dtype;
    t->refcount = 1;
    t->owns_data = true;
    
    /* Copy shape */
    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->numel *= shape[i];
    }
    
    /* Allocate and copy row_ptr */
    t->row_ptr = (int64_t*)malloc((rows + 1) * sizeof(int64_t));
    if (!t->row_ptr) {
        free(t);
        return NULL;
    }
    
    if (row_ptr->dtype == TM_I64) {
        memcpy(t->row_ptr, row_ptr->data, (rows + 1) * sizeof(int64_t));
    } else if (row_ptr->dtype == TM_I32) {
        int32_t* rp_data = (int32_t*)row_ptr->data;
        for (int64_t i = 0; i <= rows; i++) {
            t->row_ptr[i] = rp_data[i];
        }
    }
    
    /* Allocate and copy col_ind */
    t->col_ind = (int64_t*)malloc(nnz * sizeof(int64_t));
    if (!t->col_ind) {
        free(t->row_ptr);
        free(t);
        return NULL;
    }
    
    if (col_ind->dtype == TM_I64) {
        memcpy(t->col_ind, col_ind->data, nnz * sizeof(int64_t));
    } else if (col_ind->dtype == TM_I32) {
        int32_t* ci_data = (int32_t*)col_ind->data;
        for (int64_t i = 0; i < nnz; i++) {
            t->col_ind[i] = ci_data[i];
        }
    }
    
    /* Copy values */
    size_t val_size = nnz * tm_dtype_size(values->dtype);
    t->data = malloc(val_size);
    if (!t->data) {
        free(t->col_ind);
        free(t->row_ptr);
        free(t);
        return NULL;
    }
    memcpy(t->data, values->data, val_size);
    
    return t;
}

/* Convert dense tensor to sparse COO format */
tm_tensor* tm_to_sparse(tm_tensor* t) {
    if (!t || t->is_sparse) return NULL;
    if (t->ndim != 2) return NULL;  /* Only 2D for now */
    
    int64_t rows = t->shape[0];
    int64_t cols = t->shape[1];
    float* data = (float*)t->data;
    
    /* First pass: count non-zeros */
    int64_t nnz = 0;
    for (int64_t i = 0; i < t->numel; i++) {
        if (data[i] != 0.0f) nnz++;
    }
    
    if (nnz == 0) {
        /* Return empty sparse tensor */
        tm_tensor* sparse = (tm_tensor*)calloc(1, sizeof(tm_tensor));
        sparse->is_sparse = true;
        sparse->sparse_format = TM_SPARSE_COO;
        sparse->nnz = 0;
        sparse->ndim = 2;
        sparse->shape[0] = rows;
        sparse->shape[1] = cols;
        sparse->numel = rows * cols;
        sparse->dtype = t->dtype;
        sparse->refcount = 1;
        sparse->owns_data = true;
        sparse->indices = NULL;
        sparse->data = NULL;
        return sparse;
    }
    
    /* Allocate indices and values */
    int64_t* indices = (int64_t*)malloc(2 * nnz * sizeof(int64_t));
    float* values = (float*)malloc(nnz * sizeof(float));
    
    if (!indices || !values) {
        free(indices);
        free(values);
        return NULL;
    }
    
    /* Second pass: fill indices and values */
    int64_t idx = 0;
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            float val = data[r * cols + c];
            if (val != 0.0f) {
                indices[idx] = r;          /* Row index */
                indices[nnz + idx] = c;    /* Col index */
                values[idx] = val;
                idx++;
            }
        }
    }
    
    /* Create sparse tensor */
    tm_tensor* sparse = (tm_tensor*)calloc(1, sizeof(tm_tensor));
    sparse->is_sparse = true;
    sparse->sparse_format = TM_SPARSE_COO;
    sparse->nnz = nnz;
    sparse->ndim = 2;
    sparse->shape[0] = rows;
    sparse->shape[1] = cols;
    sparse->numel = rows * cols;
    sparse->dtype = t->dtype;
    sparse->refcount = 1;
    sparse->owns_data = true;
    sparse->indices = indices;
    sparse->data = values;
    
    return sparse;
}

/* Convert dense tensor to sparse CSR format */
tm_tensor* tm_to_sparse_csr(tm_tensor* t) {
    if (!t || t->is_sparse) return NULL;
    if (t->ndim != 2) return NULL;
    
    int64_t rows = t->shape[0];
    int64_t cols = t->shape[1];
    float* data = (float*)t->data;
    
    /* First pass: count non-zeros per row */
    int64_t nnz = 0;
    int64_t* row_counts = (int64_t*)calloc(rows, sizeof(int64_t));
    
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            if (data[r * cols + c] != 0.0f) {
                row_counts[r]++;
                nnz++;
            }
        }
    }
    
    /* Build row_ptr */
    int64_t* row_ptr = (int64_t*)malloc((rows + 1) * sizeof(int64_t));
    row_ptr[0] = 0;
    for (int64_t r = 0; r < rows; r++) {
        row_ptr[r + 1] = row_ptr[r] + row_counts[r];
    }
    free(row_counts);
    
    /* Allocate col_ind and values */
    int64_t* col_ind = (int64_t*)malloc(nnz * sizeof(int64_t));
    float* values = (float*)malloc(nnz * sizeof(float));
    
    if (!col_ind || !values) {
        free(row_ptr);
        free(col_ind);
        free(values);
        return NULL;
    }
    
    /* Fill col_ind and values */
    int64_t idx = 0;
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t c = 0; c < cols; c++) {
            float val = data[r * cols + c];
            if (val != 0.0f) {
                col_ind[idx] = c;
                values[idx] = val;
                idx++;
            }
        }
    }
    
    /* Create sparse tensor */
    tm_tensor* sparse = (tm_tensor*)calloc(1, sizeof(tm_tensor));
    sparse->is_sparse = true;
    sparse->sparse_format = TM_SPARSE_CSR;
    sparse->nnz = nnz;
    sparse->ndim = 2;
    sparse->shape[0] = rows;
    sparse->shape[1] = cols;
    sparse->numel = rows * cols;
    sparse->dtype = t->dtype;
    sparse->refcount = 1;
    sparse->owns_data = true;
    sparse->row_ptr = row_ptr;
    sparse->col_ind = col_ind;
    sparse->data = values;
    
    return sparse;
}

/* Convert sparse tensor to dense */
tm_tensor* tm_to_dense(tm_tensor* t) {
    if (!t || !t->is_sparse) return NULL;
    if (t->ndim != 2) return NULL;
    
    int64_t rows = t->shape[0];
    int64_t cols = t->shape[1];
    
    /* Create dense tensor */
    tm_tensor* dense = tm_zeros(t->shape, t->ndim, t->dtype);
    if (!dense) return NULL;
    
    float* dense_data = (float*)dense->data;
    float* sparse_vals = (float*)t->data;
    
    if (t->sparse_format == TM_SPARSE_COO) {
        /* COO format */
        for (int64_t i = 0; i < t->nnz; i++) {
            int64_t r = t->indices[i];
            int64_t c = t->indices[t->nnz + i];
            dense_data[r * cols + c] = sparse_vals[i];
        }
    } else if (t->sparse_format == TM_SPARSE_CSR) {
        /* CSR format */
        for (int64_t r = 0; r < rows; r++) {
            for (int64_t j = t->row_ptr[r]; j < t->row_ptr[r + 1]; j++) {
                int64_t c = t->col_ind[j];
                dense_data[r * cols + c] = sparse_vals[j];
            }
        }
    }
    
    return dense;
}

/* Convert COO to CSR */
tm_tensor* tm_sparse_coo_to_csr(tm_tensor* t) {
    if (!t || !t->is_sparse || t->sparse_format != TM_SPARSE_COO) return NULL;
    
    /* Convert to dense first, then to CSR (simple approach) */
    tm_tensor* dense = tm_to_dense(t);
    if (!dense) return NULL;
    
    tm_tensor* csr = tm_to_sparse_csr(dense);
    tm_free(dense);
    
    return csr;
}

/* Convert CSR to COO */
tm_tensor* tm_sparse_csr_to_coo(tm_tensor* t) {
    if (!t || !t->is_sparse || t->sparse_format != TM_SPARSE_CSR) return NULL;
    
    int64_t rows = t->shape[0];
    int64_t nnz = t->nnz;
    
    /* Allocate COO indices */
    int64_t* indices = (int64_t*)malloc(2 * nnz * sizeof(int64_t));
    float* values = (float*)malloc(nnz * sizeof(float));
    
    if (!indices || !values) {
        free(indices);
        free(values);
        return NULL;
    }
    
    /* Convert row_ptr to row indices */
    float* sparse_vals = (float*)t->data;
    int64_t idx = 0;
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t j = t->row_ptr[r]; j < t->row_ptr[r + 1]; j++) {
            indices[idx] = r;
            indices[nnz + idx] = t->col_ind[j];
            values[idx] = sparse_vals[j];
            idx++;
        }
    }
    
    /* Create COO tensor */
    tm_tensor* coo = (tm_tensor*)calloc(1, sizeof(tm_tensor));
    coo->is_sparse = true;
    coo->sparse_format = TM_SPARSE_COO;
    coo->nnz = nnz;
    coo->ndim = t->ndim;
    for (int i = 0; i < t->ndim; i++) {
        coo->shape[i] = t->shape[i];
    }
    coo->numel = t->numel;
    coo->dtype = t->dtype;
    coo->refcount = 1;
    coo->owns_data = true;
    coo->indices = indices;
    coo->data = values;
    
    return coo;
}

/* Sparse-dense matrix multiplication: sparse @ dense -> dense */
tm_tensor* tm_spmm(tm_tensor* sparse, tm_tensor* dense) {
    if (!sparse || !dense) return NULL;
    if (!sparse->is_sparse || dense->is_sparse) return NULL;
    if (sparse->ndim != 2 || dense->ndim != 2) return NULL;
    if (sparse->shape[1] != dense->shape[0]) return NULL;
    
    int64_t M = sparse->shape[0];
    int64_t N = dense->shape[1];
    
    /* Create output dense tensor */
    int64_t out_shape[2] = {M, N};
    tm_tensor* result = tm_zeros(out_shape, 2, TM_F32);
    if (!result) return NULL;
    
    float* out_data = (float*)result->data;
    float* dense_data = (float*)dense->data;
    float* sparse_vals = (float*)sparse->data;
    
    if (sparse->sparse_format == TM_SPARSE_COO) {
        /* COO SpMM */
        for (int64_t i = 0; i < sparse->nnz; i++) {
            int64_t r = sparse->indices[i];
            int64_t c = sparse->indices[sparse->nnz + i];
            float val = sparse_vals[i];
            
            for (int64_t n = 0; n < N; n++) {
                out_data[r * N + n] += val * dense_data[c * N + n];
            }
        }
    } else if (sparse->sparse_format == TM_SPARSE_CSR) {
        /* CSR SpMM (more efficient) */
        for (int64_t r = 0; r < M; r++) {
            for (int64_t j = sparse->row_ptr[r]; j < sparse->row_ptr[r + 1]; j++) {
                int64_t c = sparse->col_ind[j];
                float val = sparse_vals[j];
                
                for (int64_t n = 0; n < N; n++) {
                    out_data[r * N + n] += val * dense_data[c * N + n];
                }
            }
        }
    }
    
    return result;
}

/* Sparse matrix-vector multiplication: sparse @ vector -> vector */
tm_tensor* tm_spmv(tm_tensor* sparse, tm_tensor* vec) {
    if (!sparse || !vec) return NULL;
    if (!sparse->is_sparse || vec->is_sparse) return NULL;
    if (sparse->ndim != 2 || vec->ndim != 1) return NULL;
    if (sparse->shape[1] != vec->numel) return NULL;
    
    int64_t M = sparse->shape[0];
    
    /* Create output vector */
    int64_t out_shape[1] = {M};
    tm_tensor* result = tm_zeros(out_shape, 1, TM_F32);
    if (!result) return NULL;
    
    float* out_data = (float*)result->data;
    float* vec_data = (float*)vec->data;
    float* sparse_vals = (float*)sparse->data;
    
    if (sparse->sparse_format == TM_SPARSE_COO) {
        for (int64_t i = 0; i < sparse->nnz; i++) {
            int64_t r = sparse->indices[i];
            int64_t c = sparse->indices[sparse->nnz + i];
            out_data[r] += sparse_vals[i] * vec_data[c];
        }
    } else if (sparse->sparse_format == TM_SPARSE_CSR) {
        for (int64_t r = 0; r < M; r++) {
            for (int64_t j = sparse->row_ptr[r]; j < sparse->row_ptr[r + 1]; j++) {
                int64_t c = sparse->col_ind[j];
                out_data[r] += sparse_vals[j] * vec_data[c];
            }
        }
    }
    
    return result;
}

/* Sparse-sparse addition: sparse + sparse -> sparse (COO result) */
tm_tensor* tm_sparse_add(tm_tensor* a, tm_tensor* b) {
    if (!a || !b) return NULL;
    if (!a->is_sparse || !b->is_sparse) return NULL;
    if (a->ndim != b->ndim) return NULL;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }
    
    /* Simple approach: convert to dense, add, convert back */
    tm_tensor* da = tm_to_dense(a);
    tm_tensor* db = tm_to_dense(b);
    
    if (!da || !db) {
        tm_free(da);
        tm_free(db);
        return NULL;
    }
    
    tm_tensor* sum = tm_add(da, db);
    tm_free(da);
    tm_free(db);
    
    if (!sum) return NULL;
    
    tm_tensor* result = tm_to_sparse(sum);
    tm_free(sum);
    
    return result;
}

/* Sparse-scalar multiplication: sparse * scalar -> sparse */
tm_tensor* tm_sparse_mul_scalar(tm_tensor* t, double s) {
    if (!t || !t->is_sparse) return NULL;
    
    /* Clone the sparse tensor structure */
    tm_tensor* result = (tm_tensor*)calloc(1, sizeof(tm_tensor));
    if (!result) return NULL;
    
    result->is_sparse = true;
    result->sparse_format = t->sparse_format;
    result->nnz = t->nnz;
    result->ndim = t->ndim;
    result->dtype = t->dtype;
    result->refcount = 1;
    result->owns_data = true;
    result->numel = t->numel;
    
    for (int i = 0; i < t->ndim; i++) {
        result->shape[i] = t->shape[i];
    }
    
    /* Copy and scale values */
    float* t_vals = (float*)t->data;
    float* r_vals = (float*)malloc(t->nnz * sizeof(float));
    if (!r_vals) {
        free(result);
        return NULL;
    }
    
    for (int64_t i = 0; i < t->nnz; i++) {
        r_vals[i] = t_vals[i] * (float)s;
    }
    result->data = r_vals;
    
    /* Copy indices based on format */
    if (t->sparse_format == TM_SPARSE_COO) {
        result->indices = (int64_t*)malloc(2 * t->nnz * sizeof(int64_t));
        if (!result->indices) {
            free(r_vals);
            free(result);
            return NULL;
        }
        memcpy(result->indices, t->indices, 2 * t->nnz * sizeof(int64_t));
    } else if (t->sparse_format == TM_SPARSE_CSR) {
        result->row_ptr = (int64_t*)malloc((t->shape[0] + 1) * sizeof(int64_t));
        result->col_ind = (int64_t*)malloc(t->nnz * sizeof(int64_t));
        if (!result->row_ptr || !result->col_ind) {
            free(result->row_ptr);
            free(result->col_ind);
            free(r_vals);
            free(result);
            return NULL;
        }
        memcpy(result->row_ptr, t->row_ptr, (t->shape[0] + 1) * sizeof(int64_t));
        memcpy(result->col_ind, t->col_ind, t->nnz * sizeof(int64_t));
    }
    
    return result;
}

/* Get values tensor from sparse tensor */
tm_tensor* tm_sparse_values(tm_tensor* t) {
    if (!t || !t->is_sparse) return NULL;
    
    int64_t shape[1] = {t->nnz};
    tm_tensor* values = tm_empty(shape, 1, t->dtype);
    if (!values) return NULL;
    
    memcpy(values->data, t->data, t->nnz * tm_dtype_size(t->dtype));
    return values;
}

/* Get indices tensor from sparse COO tensor */
tm_tensor* tm_sparse_indices(tm_tensor* t) {
    if (!t || !t->is_sparse || t->sparse_format != TM_SPARSE_COO) return NULL;
    
    int64_t shape[2] = {2, t->nnz};
    tm_tensor* indices = tm_empty(shape, 2, TM_I64);
    if (!indices) return NULL;
    
    memcpy(indices->data, t->indices, 2 * t->nnz * sizeof(int64_t));
    return indices;
}

/* Coalesce sparse tensor (combine duplicate indices) */
tm_tensor* tm_sparse_coalesce(tm_tensor* t) {
    if (!t || !t->is_sparse) return NULL;
    if (t->sparse_format != TM_SPARSE_COO) return NULL;
    
    /* Simple approach: convert to dense and back (automatically coalesces) */
    tm_tensor* dense = tm_to_dense(t);
    if (!dense) return NULL;
    
    tm_tensor* result = tm_to_sparse(dense);
    tm_free(dense);
    
    return result;
}
