/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_kvq4_scale.c
 * @brief Q4_0 KV append kernels vs a host reference, byte for byte, on one K80 die.
 *
 * Runs fc_attn_kv_append_q4, _q4_dev and _slots_q4 from the EMBEDDED PTX
 * (cuda/fc_cuda_ptx.h, what ships) and compares every written Q4_0 block with
 * a host quantizer: scale = fp16(amax / -8) (round to nearest even), codes =
 * clamp(rint(x / scale) + 8). Scale bytes and code bytes are counted apart.
 *
 * Guards the 2026-10 K80 bug: ptxas 11.4 (sm_37) compiled the scale's low-byte
 * store as F2I.U32.F16.TRUNC, so every device-written scale had low byte 0.
 * Inputs stay in the fp32 normal range (the tree builds with -ffast-math,
 * which flushes host subnormals).
 *
 * Usage: CUDA_VISIBLE_DEVICES=<die> ./cuda/tests/test_kvq4_scale   (exit 0 = PASS)
 *
 * @status Working
 * @issues None
 * @todo None
 */
#include "../fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KVQ4_BLOCK_ELEMS 32
#define KVQ4_BLOCK_BYTES 18
#define KVQ4_SINKMAX 8
#define KVQ4_WMAX 256

#define TEST_HEAD_DIM 256
#define TEST_N_HEAD_KV 2
#define TEST_MAX_SEQ 8
#define TEST_NPOS 6
#define TEST_NSLOTS 3
#define TEST_SINK 4
#define TEST_WIN 2

typedef struct {
    long scale_blocks;   /* blocks whose two scale bytes differ */
    long low_byte_zero;  /* ...of which the device low byte is 0 and the host's is not */
    long code_blocks;    /* blocks whose 16 code bytes differ */
    long blocks;         /* blocks compared */
} kvq4_compare_counts;

/* fp32 -> fp16 bits, round to nearest even (normal, subnormal, inf, NaN). */
static uint16_t host_float_to_half_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = (int32_t)((bits >> 23) & 0xFFu);
    uint32_t mantissa = bits & 0x7FFFFFu;

    if (exponent == 0xFF)
        return (uint16_t)(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    int32_t half_exponent = exponent - 127 + 15;
    if (half_exponent >= 31)
        return (uint16_t)(sign | 0x7C00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10)
            return (uint16_t)sign;
        mantissa |= 0x800000u;
        const int shift = 14 - half_exponent;
        uint32_t rounded = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rest > halfway || (rest == halfway && (rounded & 1u)))
            rounded++;
        return (uint16_t)(sign | rounded);
    }
    uint32_t result = ((uint32_t)half_exponent << 10) | (mantissa >> 13);
    const uint32_t rest = mantissa & 0x1FFFu;
    if (rest > 0x1000u || (rest == 0x1000u && (result & 1u)))
        result++;
    return (uint16_t)(sign | result);
}

static float host_half_bits_to_float(uint16_t half_bits)
{
    const uint32_t sign = (uint32_t)(half_bits & 0x8000u) << 16;
    const uint32_t exponent = (half_bits >> 10) & 0x1Fu;
    const uint32_t mantissa = half_bits & 0x3FFu;
    if (exponent == 0) {
        const float magnitude = ldexpf((float)mantissa, -24);
        return sign ? -magnitude : magnitude;
    }
    uint32_t bits = sign | ((exponent == 31 ? 0xFFu : exponent - 15 + 127) << 23) |
                    (mantissa << 13);
    float value;
    memcpy(&value, &bits, 4);
    return value;
}

/* Host Q4_0 quantizer, in the device's operation order (serial fmaxf fold,
 * d = amax / -8, id = 1 / d, code = rint(x * id) + 8 clamped to 0..15). */
static void host_quantize_block(const float *x32, uint8_t *out18)
{
    float amax = 0.0f;
    for (int j = 0; j < KVQ4_BLOCK_ELEMS; j++)
        amax = fmaxf(amax, fabsf(x32[j]));
    const float d = amax / -8.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint16_t d_bits = host_float_to_half_bits(d);
    out18[0] = (uint8_t)(d_bits & 0xFF);
    out18[1] = (uint8_t)(d_bits >> 8);
    for (int lane = 0; lane < KVQ4_BLOCK_ELEMS / 2; lane++) {
        int q_lo = (int)rintf(x32[lane] * id) + 8;
        int q_hi = (int)rintf(x32[lane + KVQ4_BLOCK_ELEMS / 2] * id) + 8;
        q_lo = q_lo < 0 ? 0 : (q_lo > 15 ? 15 : q_lo);
        q_hi = q_hi < 0 ? 0 : (q_hi > 15 ? 15 : q_hi);
        out18[2 + lane] = (uint8_t)(q_lo | (q_hi << 4));
    }
}

/* Compare one device row (head_dim / 32 blocks) with the host quantizer. */
static void compare_row(const float *src_row, const uint8_t *device_row,
                        kvq4_compare_counts *counts, int *printed)
{
    for (int b = 0; b < TEST_HEAD_DIM / KVQ4_BLOCK_ELEMS; b++) {
        uint8_t ref[KVQ4_BLOCK_BYTES];
        const uint8_t *dev = device_row + (size_t)b * KVQ4_BLOCK_BYTES;
        host_quantize_block(src_row + (size_t)b * KVQ4_BLOCK_ELEMS, ref);
        counts->blocks++;
        if (memcmp(dev, ref, 2) != 0) {
            counts->scale_blocks++;
            if (dev[0] == 0 && ref[0] != 0)
                counts->low_byte_zero++;
            if (*printed < 4) {
                const uint16_t dev_bits = (uint16_t)(dev[0] | (dev[1] << 8));
                const uint16_t ref_bits = (uint16_t)(ref[0] | (ref[1] << 8));
                printf("    scale: device 0x%04x (%.6g) host 0x%04x (%.6g)\n", dev_bits,
                       host_half_bits_to_float(dev_bits), ref_bits,
                       host_half_bits_to_float(ref_bits));
                (*printed)++;
            }
        }
        if (memcmp(dev + 2, ref + 2, KVQ4_BLOCK_BYTES - 2) != 0)
            counts->code_blocks++;
    }
}

/* Deterministic inputs: per-block magnitudes from 1e-3 to ~1e2, mixed signs,
 * one all-zero block per row (d = -0.0). */
static void fill_inputs(float *values, size_t n, uint32_t seed)
{
    uint32_t state = seed;
    for (size_t i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        const size_t block = i / KVQ4_BLOCK_ELEMS;
        const float magnitude = powf(10.0f, (float)(block % 6) - 3.0f) * (1.0f + (float)(block % 7));
        const float unit = (float)(state >> 8) / 16777216.0f * 2.0f - 1.0f;
        values[i] = (block % (TEST_HEAD_DIM / KVQ4_BLOCK_ELEMS) == 5) ? 0.0f : unit * magnitude;
    }
}

static int report(const char *name, const char *cache, const kvq4_compare_counts *c)
{
    const int pass = c->scale_blocks == 0 && c->code_blocks == 0;
    printf("  %-24s %s: %ld blocks, scale mismatches %ld (low byte 0: %ld), code mismatches %ld  %s\n",
           name, cache, c->blocks, c->scale_blocks, c->low_byte_zero, c->code_blocks,
           pass ? "OK" : "FAIL");
    return pass ? 0 : 1;
}

typedef struct {
    fc_cuda_ctx *ctx;
    fc_buffer *k_new, *v_new, *k_cache, *v_cache, *k_sink, *v_sink, *k_win, *v_win;
    size_t cache_bytes;
} kvq4_buffers;

static int buffers_create(kvq4_buffers *b, fc_cuda_ctx *ctx, const float *k_src,
                          const float *v_src, size_t src_floats, int nslots)
{
    const size_t row_bytes = (TEST_HEAD_DIM / KVQ4_BLOCK_ELEMS) * KVQ4_BLOCK_BYTES;
    const size_t shadow = (size_t)nslots * TEST_N_HEAD_KV * KVQ4_WMAX * TEST_HEAD_DIM * 2;
    b->ctx = ctx;
    b->cache_bytes = (size_t)nslots * TEST_N_HEAD_KV * TEST_MAX_SEQ * row_bytes;
    uint8_t *zero = calloc(1, shadow > b->cache_bytes ? shadow : b->cache_bytes);
    b->k_new = fc_buffer_from_ptr_ctx(ctx, k_src, src_floats * sizeof(float));
    b->v_new = fc_buffer_from_ptr_ctx(ctx, v_src, src_floats * sizeof(float));
    b->k_cache = fc_buffer_from_ptr_ctx(ctx, zero, b->cache_bytes);
    b->v_cache = fc_buffer_from_ptr_ctx(ctx, zero, b->cache_bytes);
    b->k_sink = fc_buffer_from_ptr_ctx(ctx, zero, shadow);
    b->v_sink = fc_buffer_from_ptr_ctx(ctx, zero, shadow);
    b->k_win = fc_buffer_from_ptr_ctx(ctx, zero, shadow);
    b->v_win = fc_buffer_from_ptr_ctx(ctx, zero, shadow);
    free(zero);
    return (b->k_new && b->v_new && b->k_cache && b->v_cache && b->k_sink && b->v_sink &&
            b->k_win && b->v_win) ? 0 : 1;
}

static void buffers_free(kvq4_buffers *b)
{
    fc_buffer *all[] = {b->k_new, b->v_new, b->k_cache, b->v_cache,
                        b->k_sink, b->v_sink, b->k_win, b->v_win};
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++)
        fc_buffer_free(all[i]);
}

/* Read both caches back and compare every row that `src_of_row` maps to an input
 * row (-1 = row not written). Rows are [slot][head][max_seq]. */
static int check_caches(const char *name, kvq4_buffers *b, const float *k_src,
                        const float *v_src, int nslots, int (*src_of_row)(int, int, int))
{
    const size_t row_bytes = (TEST_HEAD_DIM / KVQ4_BLOCK_ELEMS) * KVQ4_BLOCK_BYTES;
    uint8_t *k_dev = malloc(b->cache_bytes), *v_dev = malloc(b->cache_bytes);
    kvq4_compare_counts kc = {0}, vc = {0};
    int printed = 0, fail = 0;

    fc_sync_ctx(b->ctx);
    fc_buffer_to_cpu(b->k_cache, k_dev);
    fc_buffer_to_cpu(b->v_cache, v_dev);
    for (int u = 0; u < nslots; u++)
        for (int h = 0; h < TEST_N_HEAD_KV; h++)
            for (int p = 0; p < TEST_MAX_SEQ; p++) {
                const int src = src_of_row(u, h, p);
                if (src < 0)
                    continue;
                const size_t row = ((size_t)u * TEST_N_HEAD_KV + h) * TEST_MAX_SEQ + p;
                compare_row(k_src + (size_t)src * TEST_HEAD_DIM, k_dev + row * row_bytes, &kc,
                            &printed);
                compare_row(v_src + (size_t)src * TEST_HEAD_DIM, v_dev + row * row_bytes, &vc,
                            &printed);
            }
    fail |= report(name, "K", &kc);
    fail |= report(name, "V", &vc);
    if (kc.blocks == 0)
        fail = 1;
    free(k_dev);
    free(v_dev);
    return fail;
}

/* append_q4 (pos0 = 1, npos rows) and append_q4_dev (one row per launch from a
 * device pos) both write row p = 1 + t from input row (t, h). */
static int src_row_single(int u, int h, int p)
{
    (void)u;
    return (p >= 1 && p < 1 + TEST_NPOS) ? (p - 1) * TEST_N_HEAD_KV + h : -1;
}

/* append_slots_q4: slot u writes position u + 2 from input row (u, h). */
static int src_row_slots(int u, int h, int p)
{
    return p == u + 2 ? u * TEST_N_HEAD_KV + h : -1;
}

static int run_append_q4(fc_cuda_ctx *ctx, const float *k_src, const float *v_src)
{
    kvq4_buffers b;
    int fail = buffers_create(&b, ctx, k_src, v_src,
                              (size_t)TEST_NPOS * TEST_N_HEAD_KV * TEST_HEAD_DIM, 1);
    if (!fail) {
        fc_attn_kv_append_q4_ctx(ctx, b.k_new, b.v_new, b.k_cache, b.v_cache, b.k_sink,
                                 b.v_sink, b.k_win, b.v_win, TEST_N_HEAD_KV, TEST_HEAD_DIM,
                                 TEST_MAX_SEQ, 1, TEST_NPOS, TEST_SINK, TEST_WIN);
        fail = check_caches("fc_attn_kv_append_q4", &b, k_src, v_src, 1, src_row_single);
    }
    buffers_free(&b);
    return fail;
}

static int run_append_q4_dev(fc_cuda_ctx *ctx, const float *k_src, const float *v_src)
{
    const size_t row_floats = (size_t)TEST_N_HEAD_KV * TEST_HEAD_DIM;
    kvq4_buffers b;
    int fail = buffers_create(&b, ctx, k_src, v_src, TEST_NPOS * row_floats, 1);
    fc_buffer *pos = fc_buffer_alloc_ctx(ctx, sizeof(int));
    fc_buffer *k_one = fc_buffer_alloc_ctx(ctx, row_floats * sizeof(float));
    fc_buffer *v_one = fc_buffer_alloc_ctx(ctx, row_floats * sizeof(float));
    if (!pos || !k_one || !v_one)
        fail = 1;
    for (int t = 0; !fail && t < TEST_NPOS; t++) {
        const int p = 1 + t;
        fc_sync_ctx(ctx);
        fc_buffer_from_cpu(pos, &p);
        fc_buffer_from_cpu(k_one, k_src + (size_t)t * row_floats);
        fc_buffer_from_cpu(v_one, v_src + (size_t)t * row_floats);
        fc_attn_kv_append_q4_dev_ctx(ctx, k_one, v_one, b.k_cache, b.v_cache, b.k_sink,
                                     b.v_sink, b.k_win, b.v_win, TEST_N_HEAD_KV,
                                     TEST_HEAD_DIM, TEST_MAX_SEQ, pos, TEST_SINK, TEST_WIN);
    }
    if (!fail)
        fail = check_caches("fc_attn_kv_append_q4_dev", &b, k_src, v_src, 1, src_row_single);
    fc_buffer_free(pos);
    fc_buffer_free(k_one);
    fc_buffer_free(v_one);
    buffers_free(&b);
    return fail;
}

static int run_append_slots_q4(fc_cuda_ctx *ctx, const float *k_src, const float *v_src)
{
    const int slot_pos[TEST_NSLOTS] = {2, 3, 4};
    kvq4_buffers b;
    int fail = buffers_create(&b, ctx, k_src, v_src,
                              (size_t)TEST_NSLOTS * TEST_N_HEAD_KV * TEST_HEAD_DIM, TEST_NSLOTS);
    fc_buffer *pos = fc_buffer_from_ptr_ctx(ctx, slot_pos, sizeof slot_pos);
    if (!pos)
        fail = 1;
    if (!fail) {
        fc_attn_kv_append_slots_q4_ctx(ctx, b.k_new, b.v_new, b.k_cache, b.v_cache, b.k_sink,
                                       b.v_sink, b.k_win, b.v_win, TEST_N_HEAD_KV,
                                       TEST_HEAD_DIM, TEST_MAX_SEQ, pos, NULL, TEST_NSLOTS,
                                       TEST_SINK, TEST_WIN);
        fail = check_caches("fc_attn_kv_append_slots_q4", &b, k_src, v_src, TEST_NSLOTS,
                            src_row_slots);
    }
    fc_buffer_free(pos);
    buffers_free(&b);
    return fail;
}

int main(void)
{
    const size_t n = (size_t)TEST_NPOS * TEST_N_HEAD_KV * TEST_HEAD_DIM;
    float *k_src = malloc(n * sizeof(float)), *v_src = malloc(n * sizeof(float));
    int fail = 0;

    fill_inputs(k_src, n, 0x1234u);
    fill_inputs(v_src, n, 0xBEEFu);
    fc_cuda_ctx *ctx = fc_cuda_init_device(0);
    if (!ctx || !fc_attn_flash_q4_available_ctx(ctx) ||
        !fc_attn_decode_graph_available_ctx(ctx, 1) ||
        !fc_attn_flash_decode_slots_q4_available_ctx(ctx)) {
        printf("test_kvq4_scale: Q4 KV kernels not available on device 0 -- FAIL\n");
        return 1;
    }
    printf("test_kvq4_scale: Q4_0 KV append vs host reference (head_dim %d)\n", TEST_HEAD_DIM);
    fail |= run_append_q4(ctx, k_src, v_src);
    fail |= run_append_q4_dev(ctx, k_src, v_src);
    fail |= run_append_slots_q4(ctx, k_src, v_src);
    printf("test_kvq4_scale: %s\n", fail ? "FAIL" : "PASS");
    fc_cuda_ctx_free(ctx);
    free(k_src);
    free(v_src);
    return fail;
}
