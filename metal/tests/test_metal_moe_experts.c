/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file test_metal_moe_experts.c
 * @brief Absolute-correctness tests for the grouped MoE expert kernels: the
 *   Q3_K gate+up SwiGLU (slot-major and sort-prepass-driven) and the Q3_K /
 *   Q4_K down projection with its routed-weight sum.
 *
 *   Every assertion is against the CPU reference in this file, never against
 *   another GPU arm.  The reference dequantizes each super-block with a
 *   CLOSED-FORM per-weight index (and, for Q3_K, a per-scale bit-field unpack),
 *   while the kernels walk the block by half, shift level and 16-weight group —
 *   the same mapping derived two different ways, so a transcription slip in one
 *   cannot hide in the other.
 *
 *   The weights are REAL: blk.0.ffn_{gate,up}_exps (Q3_K), blk.5.ffn_down_exps
 *   (Q3_K) and blk.0.ffn_down_exps (Q4_K) are read out of the shipped GGUF at
 *   their true file offsets.  A synthetic block cannot catch a misread of the
 *   merged 3-D expert layout, which is the thing most likely to be wrong here.
 *
 *   Half the model's layers ship ffn_down_exps as Q4_K, not Q3_K — the plan's
 *   "all Q3_K in the MVP" is wrong on this point — so both decoders are tested.
 *
 * @status All tests pass on M4 / macOS 26.  Proven able to fail: a 0.01% scaled
 *   row sum, a swapped gate/up half assignment, an off-by-one super-block
 *   stride and a dropped routed weight were induced and all caught.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
/* The family header directly, not the fm_metal.h umbrella — see the note in
 * test_metal_moe.c. */
#include "metal/kernels/fm_kernel_moe.h"
#include "metal/tests/gguf_test_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * Fixtures
 * ================================================================ */

#define Q3_K_BLOCK_BYTES  110
#define Q4_K_BLOCK_BYTES  144
#define BLOCK_WEIGHTS     256

#define GGUF_TYPE_Q3_K  11
#define GGUF_TYPE_Q4_K  12

#define MOE_N_EMBD      2048
#define MOE_EXPERT_FF    512
#define MOE_EXPERT_USED    8

/* Experts lifted out of the real merged tensors, spread across the 256 so the
 * test exercises far-apart expert strides rather than the first few. */
#define MOE_TEST_EXPERTS  8
static const int kSourceExperts[MOE_TEST_EXPERTS] = { 0, 3, 17, 42, 91, 128, 200, 255 };

/* A byte offset that is a multiple of 32 (GGUF's tensor alignment) but not of
 * the 16384-byte page — the real placement a weight gets inside a region. */
#define WEIGHT_BIND_OFFSET  (1024u * 1024u + 32u)

static const char *kDefaultModelPath =
    "models/Nex-N2-mini/nex-agi_Nex-N2-mini-Q3_K_M.gguf";

static gguf_test_file g_model;
static int g_model_is_open = 0;

static uint32_t moe_rand_state = 0xC0FFEE11u;

static uint32_t moe_rand_u32(void)
{
    moe_rand_state ^= moe_rand_state << 13;
    moe_rand_state ^= moe_rand_state >> 17;
    moe_rand_state ^= moe_rand_state << 5;
    return moe_rand_state;
}

static float moe_rand_signed(float magnitude)
{
    float unit = (float)(moe_rand_u32() >> 8) / (float)(1u << 24);
    return (unit * 2.0f - 1.0f) * magnitude;
}

/* ================================================================
 * CPU dequant reference — closed form, from the GGUF block layouts
 * ================================================================ */

/* IEEE-754 binary16 -> binary32, including subnormals, Inf and NaN. */
static float half_bits_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* Subnormal: value = mant * 2^-24. Shifting left until bit 10 is
             * set leaves 1.f * 2^(-14 - shift), so the binary32 exponent field
             * is 127 - 14 - shift. Getting this off by one halves every
             * subnormal scale, and a real Q4_K `d` IS subnormal — block 1 of
             * blk.0.ffn_down_exps row 0 is 4.55e-05, below half's 6.10e-05
             * smallest normal. */
            int shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; shift++; }
            mant &= 0x03FFu;
            bits = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t read_u16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

/**
 * Dequantizes one 110-byte Q3_K super-block: hmask[32] @0, qs[64] @32,
 * scales[12] @96, d (fp16) @108.  Weight i takes its 2 low bits from
 * qs[(i/128)*32 + i%32] at shift 2*((i%128)/32), its high bit from hmask[i%32]
 * bit (i/128)*4 + (i%128)/32 — CLEARED means subtract 4 — and its scale from
 * sub-block i/16.  Six-bit scale j takes its low 4 bits from packed byte j%8
 * (low nibble for j < 8, high nibble after) and its high 2 bits from packed
 * byte 8 + j%4 at bit 2*(j/4).
 */
static void cpu_q3k_dequant_block(const uint8_t *blk, float *out)
{
    const uint8_t *hmask = blk;
    const uint8_t *qs = blk + 32;
    float d = half_bits_to_float(read_u16(blk + 108));

    const uint8_t *packed = blk + 96;
    uint8_t scales[16];
    for (int j = 0; j < 16; j++) {
        int low = (j < 8) ? (packed[j] & 0x0f) : (packed[j - 8] >> 4);
        int high = (packed[8 + j % 4] >> (2 * (j / 4))) & 3;
        scales[j] = (uint8_t)(low | (high << 4));
    }

    for (int i = 0; i < BLOCK_WEIGHTS; i++) {
        int qs_index = (i / 128) * 32 + (i % 32);
        int shift = 2 * ((i % 128) / 32);
        int high_bit = (i / 128) * 4 + (i % 128) / 32;
        int quant = (qs[qs_index] >> shift) & 3;
        int bias = (hmask[i % 32] & (1 << high_bit)) ? 0 : 4;
        out[i] = d * (float)((int)scales[i / 16] - 32) * (float)(quant - bias);
    }
}

/**
 * Dequantizes one 144-byte Q4_K super-block: d @0, dmin @2 (both fp16),
 * scales[12] @4, qs[128] @16.  Weight i sits in 64-weight group i/64: the
 * first 32 take the low nibble of qs[32*(i/64) + i%32] with sub-block scale
 * 2*(i/64), the second 32 the high nibble with sub-block 2*(i/64)+1.
 * value = d*scale*q - dmin*min.
 */
static void cpu_q4k_dequant_block(const uint8_t *blk, float *out)
{
    float d = half_bits_to_float(read_u16(blk));
    float dmin = half_bits_to_float(read_u16(blk + 2));
    const uint8_t *sr = blk + 4;
    const uint8_t *qs = blk + 16;

    for (int i = 0; i < BLOCK_WEIGHTS; i++) {
        int group = i / 64;
        int within = i % 64;
        int sub = 2 * group + (within >= 32 ? 1 : 0);
        uint8_t packed = qs[32 * group + (within % 32)];
        int quant = (within < 32) ? (packed & 0x0F) : (packed >> 4);
        int scale, minimum;
        if (sub < 4) {
            scale = sr[sub] & 63;
            minimum = sr[sub + 4] & 63;
        } else {
            scale = (sr[sub + 4] & 0x0F) | ((sr[sub - 4] >> 6) << 4);
            minimum = (sr[sub + 4] >> 4) | ((sr[sub] >> 6) << 4);
        }
        out[i] = d * (float)scale * (float)quant - dmin * (float)minimum;
    }
}

static float cpu_row_dot(const uint8_t *row, const float *x, int cols,
                         int block_bytes)
{
    float block[BLOCK_WEIGHTS];
    float acc = 0.0f;
    for (int b = 0; b < cols / BLOCK_WEIGHTS; b++) {
        if (block_bytes == Q3_K_BLOCK_BYTES)
            cpu_q3k_dequant_block(row + (size_t)b * block_bytes, block);
        else
            cpu_q4k_dequant_block(row + (size_t)b * block_bytes, block);
        for (int l = 0; l < BLOCK_WEIGHTS; l++)
            acc += block[l] * x[b * BLOCK_WEIGHTS + l];
    }
    return acc;
}

/* ================================================================
 * CPU MoE reference
 * ================================================================ */

static void cpu_gateup(const uint8_t *gate_all, const uint8_t *up_all,
                       const float *x, const int *top_idx, int n_requests,
                       int expert_used, int ff, int cols, int row_stride,
                       float *hid)
{
    size_t expert_stride = (size_t)ff * (size_t)row_stride;
    for (int r = 0; r < n_requests; r++) {
        int expert = top_idx[r];
        int token = r / expert_used;
        const float *xt = x + (size_t)token * cols;
        for (int row = 0; row < ff; row++) {
            size_t base = (size_t)expert * expert_stride + (size_t)row * row_stride;
            float g = cpu_row_dot(gate_all + base, xt, cols, Q3_K_BLOCK_BYTES);
            float u = cpu_row_dot(up_all + base, xt, cols, Q3_K_BLOCK_BYTES);
            hid[(size_t)r * ff + row] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

static void cpu_down(const uint8_t *down_all, const float *hid, const int *top_idx,
                     const float *top_wt, int n_tokens, int expert_used, int rows,
                     int ff, int row_stride, int block_bytes, float *out)
{
    size_t expert_stride = (size_t)rows * (size_t)row_stride;
    for (int t = 0; t < n_tokens; t++)
        for (int row = 0; row < rows; row++) {
            float acc = 0.0f;
            for (int s = 0; s < expert_used; s++) {
                int request = t * expert_used + s;
                int expert = top_idx[request];
                const uint8_t *w = down_all + (size_t)expert * expert_stride +
                                   (size_t)row * row_stride;
                acc += top_wt[request] *
                       cpu_row_dot(w, hid + (size_t)request * ff, ff, block_bytes);
            }
            out[(size_t)t * rows + row] = acc;
        }
}

/* ================================================================
 * Comparison
 *
 * The acceptance bar from the spec is max_abs_err / RMS(reference), which
 * stays meaningful where individual reference values pass through zero.
 * ================================================================ */

typedef struct error_report {
    double max_abs;
    double rms;
    double ratio;
    size_t worst_index;
} error_report;

static error_report compare_vectors(const float *got, const float *ref, size_t n)
{
    error_report report = {0.0, 0.0, 0.0, 0};
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs((double)got[i] - (double)ref[i]);
        if (diff > report.max_abs) {
            report.max_abs = diff;
            report.worst_index = i;
        }
        sum_sq += (double)ref[i] * (double)ref[i];
    }
    report.rms = sqrt(sum_sq / (double)n);
    report.ratio = (report.rms > 0.0) ? report.max_abs / report.rms : report.max_abs;
    return report;
}

/* ================================================================
 * Loading real expert weights
 * ================================================================ */

/* Copies MOE_TEST_EXPERTS real experts out of a merged 3-D expert tensor into a
 * contiguous table, so a routing over slots 0..7 addresses real bytes from
 * experts scattered across the full 256. */
static uint8_t *load_expert_table(const char *tensor_name, int expected_type,
                                  int expected_cols, int expected_rows,
                                  size_t *expert_stride_out)
{
    gguf_tensor_location location;
    if (gguf_test_find_tensor(&g_model, tensor_name, &location) != 0) {
        fprintf(stderr, "  missing tensor %s\n", tensor_name);
        return NULL;
    }
    if ((int)location.ggml_type != expected_type || location.n_dims != 3 ||
        (int)location.dims[0] != expected_cols ||
        (int)location.dims[1] != expected_rows) {
        fprintf(stderr, "  %s has an unexpected type/shape (%u, %llu x %llu x %llu)\n",
                tensor_name, location.ggml_type,
                (unsigned long long)location.dims[0],
                (unsigned long long)location.dims[1],
                (unsigned long long)location.dims[2]);
        return NULL;
    }

    int block_bytes = (expected_type == GGUF_TYPE_Q3_K) ? Q3_K_BLOCK_BYTES
                                                        : Q4_K_BLOCK_BYTES;
    size_t row_stride = (size_t)(expected_cols / BLOCK_WEIGHTS) * block_bytes;
    size_t expert_stride = (size_t)expected_rows * row_stride;
    uint8_t *table = (uint8_t *)malloc(expert_stride * MOE_TEST_EXPERTS);
    if (!table)
        return NULL;

    for (int i = 0; i < MOE_TEST_EXPERTS; i++) {
        uint64_t offset = location.file_offset +
                          (uint64_t)kSourceExperts[i] * expert_stride;
        if (gguf_test_read_at(&g_model, offset, table + (size_t)i * expert_stride,
                              expert_stride) != 0) {
            fprintf(stderr, "  short read of %s expert %d\n", tensor_name,
                    kSourceExperts[i]);
            free(table);
            return NULL;
        }
    }
    *expert_stride_out = expert_stride;
    return table;
}

/* Slot-major routing over the local expert table: every token uses all
 * MOE_TEST_EXPERTS slots, in a token-dependent rotation so no two tokens share
 * a slot->expert map. */
static void build_local_routing(int n_tokens, int *top_idx, float *top_wt)
{
    for (int t = 0; t < n_tokens; t++) {
        float sum = 0.0f;
        for (int s = 0; s < MOE_EXPERT_USED; s++) {
            top_idx[t * MOE_EXPERT_USED + s] =
                (s + t) % MOE_TEST_EXPERTS;
            float w = 0.1f + (float)(moe_rand_u32() % 1000u) / 1000.0f;
            top_wt[t * MOE_EXPERT_USED + s] = w;
            sum += w;
        }
        for (int s = 0; s < MOE_EXPERT_USED; s++)
            top_wt[t * MOE_EXPERT_USED + s] /= sum;
    }
}

/* ================================================================
 * Gate + up
 * ================================================================ */

/* Shared fixture for the gate+up tests: real weights, random activations, and
 * both the GPU and CPU results. `weight_bind_offset` places the weight table at
 * a non-page-aligned byte offset inside its buffer when nonzero. */
static int run_gateup_case(int n_tokens, int use_sorted, size_t weight_bind_offset,
                           error_report *report)
{
    const int cols = MOE_N_EMBD;
    const int ff = MOE_EXPERT_FF;
    const int n_requests = n_tokens * MOE_EXPERT_USED;
    const int row_stride = (cols / BLOCK_WEIGHTS) * Q3_K_BLOCK_BYTES;
    size_t gate_stride = 0, up_stride = 0;
    int rc = -1;

    uint8_t *gate_table = load_expert_table("blk.0.ffn_gate_exps.weight",
                                            GGUF_TYPE_Q3_K, cols, ff, &gate_stride);
    uint8_t *up_table = load_expert_table("blk.0.ffn_up_exps.weight",
                                          GGUF_TYPE_Q3_K, cols, ff, &up_stride);
    float *x = (float *)malloc((size_t)n_tokens * cols * sizeof(float));
    int *top_idx = (int *)malloc((size_t)n_requests * sizeof(int));
    float *top_wt = (float *)malloc((size_t)n_requests * sizeof(float));
    float *ref_hid = (float *)malloc((size_t)n_requests * ff * sizeof(float));
    fm_buffer *gate_buf = NULL, *up_buf = NULL, *x_buf = NULL, *hid_buf = NULL;
    fm_buffer *idx_buf = NULL, *st_buf = NULL, *ss_buf = NULL, *se_buf = NULL;
    fm_buffer *count_buf = NULL, *offset_buf = NULL;

    if (!gate_table || !up_table || !x || !top_idx || !top_wt || !ref_hid)
        goto cleanup;

    moe_rand_state = 0x1337BEEFu;
    for (int i = 0; i < n_tokens * cols; i++)
        x[i] = moe_rand_signed(1.0f);
    build_local_routing(n_tokens, top_idx, top_wt);

    gate_buf = fm_buffer_alloc(weight_bind_offset + gate_stride * MOE_TEST_EXPERTS);
    up_buf = fm_buffer_alloc(weight_bind_offset + up_stride * MOE_TEST_EXPERTS);
    x_buf = fm_buffer_alloc((size_t)n_tokens * cols * sizeof(float));
    idx_buf = fm_buffer_alloc((size_t)n_requests * sizeof(int));
    hid_buf = fm_buffer_alloc((size_t)n_requests * ff * sizeof(float));
    if (!gate_buf || !up_buf || !x_buf || !idx_buf || !hid_buf)
        goto cleanup;

    memcpy((uint8_t *)fm_buffer_data(gate_buf) + weight_bind_offset, gate_table,
           gate_stride * MOE_TEST_EXPERTS);
    memcpy((uint8_t *)fm_buffer_data(up_buf) + weight_bind_offset, up_table,
           up_stride * MOE_TEST_EXPERTS);
    memcpy(fm_buffer_data(x_buf), x, (size_t)n_tokens * cols * sizeof(float));
    memcpy(fm_buffer_data(idx_buf), top_idx, (size_t)n_requests * sizeof(int));
    memset(fm_buffer_data(hid_buf), 0x7F, (size_t)n_requests * ff * sizeof(float));

    if (use_sorted) {
        count_buf = fm_buffer_alloc(MOE_TEST_EXPERTS * sizeof(int));
        offset_buf = fm_buffer_alloc((MOE_TEST_EXPERTS + 1) * sizeof(int));
        st_buf = fm_buffer_alloc((size_t)n_requests * sizeof(int));
        ss_buf = fm_buffer_alloc((size_t)n_requests * sizeof(int));
        se_buf = fm_buffer_alloc((size_t)n_requests * sizeof(int));
        if (!count_buf || !offset_buf || !st_buf || !ss_buf || !se_buf)
            goto cleanup;
        fm_moe_expert_sort(idx_buf, count_buf, offset_buf, st_buf, ss_buf, se_buf,
                           n_tokens, MOE_TEST_EXPERTS, MOE_EXPERT_USED);
        fm_moe_gateup_q3k_sorted_at(fm_bufref_at(gate_buf, weight_bind_offset),
                                    fm_bufref_at(up_buf, weight_bind_offset),
                                    fm_bufref_at(x_buf, 0), fm_bufref_at(st_buf, 0),
                                    fm_bufref_at(ss_buf, 0), fm_bufref_at(se_buf, 0),
                                    fm_bufref_at(hid_buf, 0), n_requests,
                                    MOE_EXPERT_USED, ff, cols, row_stride);
    } else {
        fm_moe_gateup_q3k_at(fm_bufref_at(gate_buf, weight_bind_offset),
                             fm_bufref_at(up_buf, weight_bind_offset),
                             fm_bufref_at(x_buf, 0), fm_bufref_at(idx_buf, 0),
                             fm_bufref_at(hid_buf, 0), n_tokens, MOE_EXPERT_USED,
                             ff, cols, row_stride);
    }
    fm_sync();

    cpu_gateup(gate_table, up_table, x, top_idx, n_requests, MOE_EXPERT_USED, ff,
               cols, row_stride, ref_hid);
    *report = compare_vectors((const float *)fm_buffer_data(hid_buf), ref_hid,
                              (size_t)n_requests * ff);
    rc = 0;

cleanup:
    fm_buffer_free(gate_buf);
    fm_buffer_free(up_buf);
    fm_buffer_free(x_buf);
    fm_buffer_free(idx_buf);
    fm_buffer_free(hid_buf);
    fm_buffer_free(count_buf);
    fm_buffer_free(offset_buf);
    fm_buffer_free(st_buf);
    fm_buffer_free(ss_buf);
    fm_buffer_free(se_buf);
    free(gate_table);
    free(up_table);
    free(x);
    free(top_idx);
    free(top_wt);
    free(ref_hid);
    return rc;
}

static void test_gateup_q3k_matches_cpu_reference(void)
{
    error_report report;
    ASSERT_INT_EQ(0, run_gateup_case(1, 0, 0, &report));
    printf("    gate+up  max_abs=%.3e rms=%.3e ratio=%.2e (worst %zu)\n",
           report.max_abs, report.rms, report.ratio, report.worst_index);
    ASSERT_TRUE(report.rms > 0.0);
    ASSERT_TRUE(report.ratio < 5e-5);
}

static void test_gateup_q3k_batched_matches_cpu_reference(void)
{
    error_report report;
    ASSERT_INT_EQ(0, run_gateup_case(3, 0, 0, &report));
    printf("    gate+up x3 max_abs=%.3e rms=%.3e ratio=%.2e\n",
           report.max_abs, report.rms, report.ratio);
    ASSERT_TRUE(report.rms > 0.0);
    ASSERT_TRUE(report.ratio < 5e-5);
}

static void test_gateup_q3k_reads_weights_at_a_nonzero_offset(void)
{
    error_report report;
    ASSERT_INT_EQ(0, run_gateup_case(1, 0, WEIGHT_BIND_OFFSET, &report));
    printf("    gate+up @%u max_abs=%.3e ratio=%.2e\n", WEIGHT_BIND_OFFSET,
           report.max_abs, report.ratio);
    ASSERT_TRUE(report.rms > 0.0);
    ASSERT_TRUE(report.ratio < 5e-5);
}

static void test_gateup_q3k_sorted_matches_cpu_reference(void)
{
    error_report report;
    ASSERT_INT_EQ(0, run_gateup_case(3, 1, 0, &report));
    printf("    gate+up sorted max_abs=%.3e rms=%.3e ratio=%.2e\n",
           report.max_abs, report.rms, report.ratio);
    ASSERT_TRUE(report.rms > 0.0);
    ASSERT_TRUE(report.ratio < 5e-5);
}

/* ================================================================
 * Down projection
 * ================================================================ */

static int run_down_case(const char *tensor_name, int expected_type, int n_tokens,
                         error_report *report, float *garbage_survivor)
{
    const int rows = MOE_N_EMBD;
    const int ff = MOE_EXPERT_FF;
    const int n_requests = n_tokens * MOE_EXPERT_USED;
    const int block_bytes = (expected_type == GGUF_TYPE_Q3_K) ? Q3_K_BLOCK_BYTES
                                                              : Q4_K_BLOCK_BYTES;
    const int row_stride = (ff / BLOCK_WEIGHTS) * block_bytes;
    size_t expert_stride = 0;
    int rc = -1;

    uint8_t *down_table = load_expert_table(tensor_name, expected_type, ff, rows,
                                            &expert_stride);
    float *hid = (float *)malloc((size_t)n_requests * ff * sizeof(float));
    int *top_idx = (int *)malloc((size_t)n_requests * sizeof(int));
    float *top_wt = (float *)malloc((size_t)n_requests * sizeof(float));
    float *ref_out = (float *)malloc((size_t)n_tokens * rows * sizeof(float));
    fm_buffer *down_buf = NULL, *hid_buf = NULL, *idx_buf = NULL;
    fm_buffer *wt_buf = NULL, *out_buf = NULL;

    if (!down_table || !hid || !top_idx || !top_wt || !ref_out)
        goto cleanup;

    moe_rand_state = 0x5EED1234u;
    for (int i = 0; i < n_requests * ff; i++)
        hid[i] = moe_rand_signed(2.0f);
    build_local_routing(n_tokens, top_idx, top_wt);

    down_buf = fm_buffer_alloc(expert_stride * MOE_TEST_EXPERTS);
    hid_buf = fm_buffer_alloc((size_t)n_requests * ff * sizeof(float));
    idx_buf = fm_buffer_alloc((size_t)n_requests * sizeof(int));
    wt_buf = fm_buffer_alloc((size_t)n_requests * sizeof(float));
    out_buf = fm_buffer_alloc((size_t)n_tokens * rows * sizeof(float));
    if (!down_buf || !hid_buf || !idx_buf || !wt_buf || !out_buf)
        goto cleanup;

    memcpy(fm_buffer_data(down_buf), down_table, expert_stride * MOE_TEST_EXPERTS);
    memcpy(fm_buffer_data(hid_buf), hid, (size_t)n_requests * ff * sizeof(float));
    memcpy(fm_buffer_data(idx_buf), top_idx, (size_t)n_requests * sizeof(int));
    memcpy(fm_buffer_data(wt_buf), top_wt, (size_t)n_requests * sizeof(float));

    /* The kernel writes `out` rather than accumulating, so a pre-existing
     * value must not survive. */
    float *out_seed = (float *)fm_buffer_data(out_buf);
    for (int i = 0; i < n_tokens * rows; i++)
        out_seed[i] = 12345.0f;

    if (expected_type == GGUF_TYPE_Q3_K)
        fm_moe_down_q3k(down_buf, hid_buf, idx_buf, wt_buf, out_buf, n_tokens,
                        MOE_EXPERT_USED, rows, ff, row_stride);
    else
        fm_moe_down_q4k(down_buf, hid_buf, idx_buf, wt_buf, out_buf, n_tokens,
                        MOE_EXPERT_USED, rows, ff, row_stride);
    fm_sync();

    cpu_down(down_table, hid, top_idx, top_wt, n_tokens, MOE_EXPERT_USED, rows, ff,
             row_stride, block_bytes, ref_out);
    *report = compare_vectors((const float *)fm_buffer_data(out_buf), ref_out,
                              (size_t)n_tokens * rows);
    if (garbage_survivor)
        *garbage_survivor = ((const float *)fm_buffer_data(out_buf))[0];
    rc = 0;

cleanup:
    fm_buffer_free(down_buf);
    fm_buffer_free(hid_buf);
    fm_buffer_free(idx_buf);
    fm_buffer_free(wt_buf);
    fm_buffer_free(out_buf);
    free(down_table);
    free(hid);
    free(top_idx);
    free(top_wt);
    free(ref_out);
    return rc;
}

static void test_down_q3k_matches_cpu_reference(void)
{
    error_report report;
    float first = 0.0f;
    ASSERT_INT_EQ(0, run_down_case("blk.5.ffn_down_exps.weight", GGUF_TYPE_Q3_K, 1,
                                   &report, &first));
    printf("    down Q3_K max_abs=%.3e rms=%.3e ratio=%.2e (worst %zu)\n",
           report.max_abs, report.rms, report.ratio, report.worst_index);
    ASSERT_TRUE(report.rms > 0.0);
    /* Bounds the one reassociation in this family: the routed-weight sum is
     * split across lanes by (slot, half super-block) and reduced by simd_sum,
     * where the CPU finishes an expert dot before weighting it. */
    ASSERT_TRUE(report.ratio < 1e-5);
    ASSERT_TRUE(first != 12345.0f);
}

static void test_down_q3k_batched_matches_cpu_reference(void)
{
    error_report report;
    ASSERT_INT_EQ(0, run_down_case("blk.5.ffn_down_exps.weight", GGUF_TYPE_Q3_K, 3,
                                   &report, NULL));
    printf("    down Q3_K x3 max_abs=%.3e rms=%.3e ratio=%.2e\n",
           report.max_abs, report.rms, report.ratio);
    ASSERT_TRUE(report.rms > 0.0);
    ASSERT_TRUE(report.ratio < 1e-5);
}

static void test_down_q4k_matches_cpu_reference(void)
{
    error_report report;
    ASSERT_INT_EQ(0, run_down_case("blk.0.ffn_down_exps.weight", GGUF_TYPE_Q4_K, 1,
                                   &report, NULL));
    printf("    down Q4_K max_abs=%.3e rms=%.3e ratio=%.2e (worst %zu)\n",
           report.max_abs, report.rms, report.ratio, report.worst_index);
    ASSERT_TRUE(report.rms > 0.0);
    ASSERT_TRUE(report.ratio < 1e-5);
}

/* ================================================================
 * main
 * ================================================================ */

int main(int argc, char **argv)
{
    printf("\n  Metal MoE — Grouped Experts (real GGUF weights)\n");
    printf("  ==============================================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "  Metal unavailable — no device to test against.\n");
        return 77;
    }

    const char *model_path = (argc > 1) ? argv[1] : getenv("FOUNDRY_TEST_GGUF");
    if (!model_path) model_path = kDefaultModelPath;
    g_model_is_open = gguf_test_open(model_path, &g_model) == 0;
    if (!g_model_is_open) {
        fprintf(stderr, "  Model file missing — every test here is driven by real\n"
                        "  expert blocks and cannot run without it:\n    %s\n",
                model_path);
        fm_free();
        return 77;
    }
    printf("  model = %s\n\n", model_path);

    RUN_TEST(gateup_q3k_matches_cpu_reference);
    RUN_TEST(gateup_q3k_batched_matches_cpu_reference);
    RUN_TEST(gateup_q3k_reads_weights_at_a_nonzero_offset);
    RUN_TEST(gateup_q3k_sorted_matches_cpu_reference);
    RUN_TEST(down_q3k_matches_cpu_reference);
    RUN_TEST(down_q3k_batched_matches_cpu_reference);
    RUN_TEST(down_q4k_matches_cpu_reference);

    gguf_test_close(&g_model);
    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
