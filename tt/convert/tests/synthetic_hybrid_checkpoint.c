/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file synthetic_hybrid_checkpoint.c
 * @brief Implementation of the synthetic Qwen3.8-shaped test checkpoint.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "synthetic_hybrid_checkpoint.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ft_float_convert.h"
#include "ft_gguf_write.h"

#define H 256u   /* hidden */
#define QKV 384u /* GDN fused q/k/v */
#define Z 256u   /* GDN output gate / ssm_out input */
#define FF 512u  /* FFN */
#define VOCAB 320u
#define F32 SYNTHETIC_GGML_F32
#define Q8 SYNTHETIC_GGML_Q8_0

const synthetic_tensor_spec synthetic_hybrid_tensors[] = {
    {"token_embd.weight", 2, {H, VOCAB}, Q8},
    {"blk.0.attn_norm.weight", 1, {H, 1}, F32},
    {"blk.0.attn_qkv.weight", 2, {H, QKV}, Q8},
    {"blk.0.attn_gate.weight", 2, {H, Z}, Q8},
    {"blk.0.ssm_a", 1, {8, 1}, F32},
    {"blk.0.ssm_alpha.weight", 2, {H, 8}, F32},
    {"blk.0.ssm_beta.weight", 2, {H, 8}, F32},
    {"blk.0.ssm_conv1d.weight", 2, {4, QKV}, F32},
    {"blk.0.ssm_dt.bias", 1, {8, 1}, F32},
    {"blk.0.ssm_norm.weight", 1, {32, 1}, F32},
    {"blk.0.ssm_out.weight", 2, {Z, H}, Q8},
    {"blk.0.post_attention_norm.weight", 1, {H, 1}, F32},
    {"blk.0.ffn_gate.weight", 2, {H, FF}, Q8},
    {"blk.0.ffn_up.weight", 2, {H, FF}, Q8},
    {"blk.0.ffn_down.weight", 2, {FF, H}, Q8},
    {"blk.1.attn_norm.weight", 1, {H, 1}, F32},
    {"blk.1.attn_q.weight", 2, {H, 2u * H}, Q8},
    {"blk.1.attn_k.weight", 2, {H, 64}, Q8},
    {"blk.1.attn_v.weight", 2, {H, 64}, Q8},
    {"blk.1.attn_q_norm.weight", 1, {64, 1}, F32},
    {"blk.1.attn_output.weight", 2, {H, H}, F32},
    {"blk.1.ffn_gate.weight", 2, {H, FF}, Q8},
    {"blk.1.ffn_up.weight", 2, {H, FF}, Q8},
    {"blk.1.ffn_down.weight", 2, {FF, H}, Q8},
    {"blk.2.nextn.enorm.weight", 1, {H, 1}, F32},
    {"blk.2.nextn.hnorm.weight", 1, {H, 1}, F32},
    {"blk.2.nextn.eh_proj.weight", 2, {2u * H, H}, Q8},
    {"blk.2.ffn_up.weight", 2, {H, FF}, Q8},
    {"output_norm.weight", 1, {H, 1}, F32},
    {"output.weight", 2, {H, VOCAB}, Q8},
};

const uint32_t synthetic_hybrid_tensor_count =
    (uint32_t)(sizeof(synthetic_hybrid_tensors) / sizeof(synthetic_hybrid_tensors[0]));

/** xorshift64*: deterministic and good enough for test data. */
static double uniform_open01(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return ((double)((*state * 2685821657736338717ull) >> 11) + 0.5) / 9007199254740992.0;
}

static double standard_normal(uint64_t *state)
{
    double u = uniform_open01(state);
    double v = uniform_open01(state);

    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

float *synthetic_realistic_weights(uint32_t seed, uint64_t rows, uint64_t cols)
{
    float *values = malloc((size_t)(rows * cols) * sizeof(float));
    float *gain = malloc((size_t)cols * sizeof(float));
    uint64_t state = 0x9E3779B97F4A7C15ull ^ ((uint64_t)seed * 0xD1B54A32D192ED03ull);

    if (!values || !gain) {
        free(values);
        free(gain);
        return NULL;
    }
    for (uint64_t c = 0; c < cols; c++)
        gain[c] = uniform_open01(&state) < 0.02 ? 4.0f : 1.0f;
    for (uint64_t r = 0; r < rows; r++) {
        double row_scale = 0.02 * exp(0.3 * standard_normal(&state));

        for (uint64_t c = 0; c < cols; c++) {
            double draw = standard_normal(&state) * (uniform_open01(&state) < 0.005 ? 6.0 : 1.0);

            values[r * cols + c] = (float)(draw * row_scale * gain[c]);
        }
    }
    free(gain);
    return values;
}

/** fp32 -> fp16, round to nearest, for a positive normal-range Q8_0 scale. */
static uint16_t fp16_from_positive_float(float value)
{
    uint32_t bits;
    int32_t exponent;
    uint32_t mantissa;

    memcpy(&bits, &value, sizeof(bits));
    exponent = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    if (exponent < 1)
        return 0x0400u; /* smallest normal; the stored scale is the truth either way */
    if (exponent > 30)
        return 0x7bffu;
    mantissa = (bits & 0x7fffffu) + 0x1000u; /* round at bit 13 */
    if (mantissa & 0x800000u) {
        mantissa = 0;
        exponent++;
    }
    return (uint16_t)(((uint32_t)exponent << 10) | (mantissa >> 13));
}

unsigned char *synthetic_q8_0_encode(const float *values, uint64_t rows, uint64_t cols, uint64_t *bytes_out)
{
    uint64_t blocks = rows * cols / 32u;
    unsigned char *out = malloc((size_t)blocks * 34u);

    if (!out || cols % 32u != 0u) {
        free(out);
        return NULL;
    }
    for (uint64_t b = 0; b < blocks; b++) {
        const float *block = values + b * 32u;
        unsigned char *dst = out + b * 34u;
        float peak = 0.0f;
        uint16_t half;
        float scale;

        for (int i = 0; i < 32; i++)
            peak = fabsf(block[i]) > peak ? fabsf(block[i]) : peak;
        half = peak > 0.0f ? fp16_from_positive_float(peak / 127.0f) : 0u;
        scale = ft_float_from_fp16(half);
        memcpy(dst, &half, sizeof(half));
        for (int i = 0; i < 32; i++) {
            long q = scale > 0.0f ? lroundf(block[i] / scale) : 0;
            signed char byte = (signed char)(q > 127 ? 127 : (q < -127 ? -127 : q));

            memcpy(dst + 2 + i, &byte, 1);
        }
    }
    *bytes_out = blocks * 34u;
    return out;
}

/** Fill one write record, allocating its bytes into `owned`. */
static int build_tensor(const synthetic_tensor_spec *spec, uint32_t seed, ft_gguf_write_tensor *t, void **owned)
{
    uint64_t rows = spec->n_dims > 1u ? spec->dims[1] : 1u;
    uint64_t cols = spec->dims[0];
    float *values = synthetic_realistic_weights(seed, rows, cols);

    memset(t, 0, sizeof(*t));
    t->name = spec->name;
    t->n_dims = spec->n_dims;
    t->dims[0] = cols;
    t->dims[1] = spec->n_dims > 1u ? rows : 1u;
    t->dims[2] = 1u;
    t->dims[3] = 1u;
    t->ggml_type = spec->ggml_type;
    if (!values)
        return -1;
    if (spec->ggml_type == SYNTHETIC_GGML_Q8_0) {
        *owned = synthetic_q8_0_encode(values, rows, cols, &t->byte_count);
        free(values);
    } else {
        *owned = values;
        t->byte_count = rows * cols * sizeof(float);
    }
    t->data = *owned;
    return *owned ? 0 : -1;
}

int synthetic_hybrid_checkpoint_write(const char *path)
{
    ft_gguf_write_tensor tensors[sizeof(synthetic_hybrid_tensors) / sizeof(synthetic_hybrid_tensors[0])];
    void *owned[sizeof(synthetic_hybrid_tensors) / sizeof(synthetic_hybrid_tensors[0])];
    ft_gguf_kv_buffer kv;
    int rc = 0;

    memset(owned, 0, sizeof(owned));
    memset(&kv, 0, sizeof(kv));
    for (uint32_t i = 0; i < synthetic_hybrid_tensor_count && rc == 0; i++)
        rc = build_tensor(&synthetic_hybrid_tensors[i], 1000u + i, &tensors[i], &owned[i]);
    ft_gguf_kv_add_string(&kv, "general.architecture", "qwen35");
    ft_gguf_kv_add_u32(&kv, "qwen35.block_count", 3u);
    if (rc == 0 && !kv.failed)
        rc = ft_gguf_write_file(path, kv.bytes, kv.used, kv.count, tensors, synthetic_hybrid_tensor_count, 32u) ==
                     FT_GGUF_WRITE_OK
                 ? 0
                 : -1;
    else
        rc = -1;
    ft_gguf_kv_free(&kv);
    for (uint32_t i = 0; i < synthetic_hybrid_tensor_count; i++)
        free(owned[i]);
    return rc;
}
