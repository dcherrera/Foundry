/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ft_gguf_dequant.c
 * @brief  Byte sizing, naming and scalar fp32 decoding of GGUF element types.
 *
 * Written from a behaviour specification; no ggml or llama.cpp source was consulted.
 *
 * One scalar code path on every target. The TT build compiles this with
 * -O3 -ffast-math, so the loop nests, the once-per-sub-block scale hoisting and
 * the one-product-per-statement shapes below are deliberate: they decide what
 * non-finite scales decode to under contraction. Do not restructure them.
 *
 * @status None
 * @issues F32/F16/BF16 byte sizes wrap in 64-bit arithmetic (kept as-is).
 * @todo   None
 */

#include "ft_gguf_dequant.h"

#include <string.h>

#include "ft_float_convert.h"
#include "ft_gguf_map.h"

/** fp16 stored little-endian at `bytes[offset]`, widened exactly. */
static float half_at(const unsigned char *bytes, size_t offset)
{
    uint16_t half = (uint16_t)(bytes[offset] | (bytes[offset + 1] << 8));
    return ft_float_from_fp16(half);
}

size_t ft_gguf_type_bytes(uint32_t ggml_type, uint64_t elements)
{
    uint64_t block_bytes = 0;

    switch (ggml_type) {
    case FT_GGML_TYPE_F32:
        return (size_t)(elements * 4u);
    case FT_GGML_TYPE_F16:
    case FT_GGML_TYPE_BF16:
        return (size_t)(elements * 2u);
    case FT_GGML_TYPE_Q8_0:
        if (elements % FT_GGML_Q8_0_BLOCK != 0)
            return 0;
        return (size_t)((elements / FT_GGML_Q8_0_BLOCK) * FT_GGML_Q8_0_BLOCK_BYTES);
    case FT_GGML_TYPE_Q2_K:
        block_bytes = FT_GGML_Q2_K_BLOCK_BYTES;
        break;
    case FT_GGML_TYPE_Q3_K:
        block_bytes = FT_GGML_Q3_K_BLOCK_BYTES;
        break;
    case FT_GGML_TYPE_Q4_K:
        block_bytes = FT_GGML_Q4_K_BLOCK_BYTES;
        break;
    case FT_GGML_TYPE_Q5_K:
        block_bytes = FT_GGML_Q5_K_BLOCK_BYTES;
        break;
    case FT_GGML_TYPE_Q6_K:
        block_bytes = FT_GGML_Q6_K_BLOCK_BYTES;
        break;
    default:
        return 0;
    }
    if (elements % FT_GGML_K_SUPERBLOCK != 0)
        return 0;
    return (size_t)((elements / FT_GGML_K_SUPERBLOCK) * block_bytes);
}

const char *ft_gguf_type_name(uint32_t ggml_type)
{
    switch (ggml_type) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 6: return "Q5_0";
    case 7: return "Q5_1";
    case 8: return "Q8_0";
    case 9: return "Q8_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 15: return "Q8_K";
    case 16: return "IQ2_XXS";
    case 17: return "IQ2_XS";
    case 18: return "IQ3_XXS";
    case 19: return "IQ1_S";
    case 20: return "IQ4_NL";
    case 21: return "IQ3_S";
    case 22: return "IQ2_S";
    case 23: return "IQ4_XS";
    case 24: return "I8";
    case 25: return "I16";
    case 26: return "I32";
    case 27: return "I64";
    case 28: return "F64";
    case 29: return "IQ1_M";
    case 30: return "BF16";
    case 34: return "TQ1_0";
    case 35: return "TQ2_0";
    case 39: return "MXFP4";
    case 40: return "NVFP4";
    case 41: return "Q1_0";
    default: return "?";
    }
}

int ft_gguf_type_supported(uint32_t ggml_type)
{
    switch (ggml_type) {
    case FT_GGML_TYPE_F32:
    case FT_GGML_TYPE_F16:
    case FT_GGML_TYPE_BF16:
    case FT_GGML_TYPE_Q8_0:
    case FT_GGML_TYPE_Q2_K:
    case FT_GGML_TYPE_Q3_K:
    case FT_GGML_TYPE_Q4_K:
    case FT_GGML_TYPE_Q5_K:
    case FT_GGML_TYPE_Q6_K:
        return 1;
    default:
        return 0;
    }
}

/** 34 bytes: fp16 scale, then 32 int8 quants. Quant on the left of the product. */
static void decode_q8_0_block(const unsigned char *block, float *out)
{
    uint16_t half;
    float scale;
    int i;

    memcpy(&half, block, sizeof(half));
    scale = ft_float_from_fp16(half);
    for (i = 0; i < FT_GGML_Q8_0_BLOCK; i++)
        out[i] = (float)(int8_t)block[2 + i] * scale;
}

/** 84 bytes: 16 packed scale/min nibbles, 64 bytes of 2-bit codes, d, dmin. */
static void decode_q2_k_block(const unsigned char *block, float *out)
{
    const unsigned char *scales = block;
    const unsigned char *codes = block + 16;
    float d = half_at(block, 80);
    float dmin = half_at(block, 82);
    size_t e = 0;
    int half;

    for (half = 0; half < 2; half++) {
        int shift;
        for (shift = 0; shift < 8; shift += 2) {
            int group;
            for (group = 0; group < 2; group++) {
                int k = 8 * half + 2 * (shift / 2) + group;
                float sub_scale = d * (float)(scales[k] & 15);
                float sub_min = dmin * (float)(scales[k] >> 4);
                int l;
                for (l = 0; l < 16; l++) {
                    int code = (codes[32 * half + 16 * group + l] >> shift) & 3;
                    float scaled = sub_scale * (float)code;
                    out[e] = scaled - sub_min;
                    e++;
                }
            }
        }
    }
}

/** Unpack Q3_K's twelve bytes of 6-bit scales into sixteen signed values. */
static void unpack_q3_k_scales(const unsigned char *packed, int *scales)
{
    int i;

    for (i = 0; i < 4; i++) {
        int hi = packed[8 + i];
        scales[i] = ((packed[i] & 15) | ((hi >> 0) & 3) << 4) - 32;
        scales[i + 4] = ((packed[i + 4] & 15) | ((hi >> 2) & 3) << 4) - 32;
        scales[i + 8] = ((packed[i] >> 4) | ((hi >> 4) & 3) << 4) - 32;
        scales[i + 12] = ((packed[i + 4] >> 4) | ((hi >> 6) & 3) << 4) - 32;
    }
}

/** 110 bytes: 32 bytes of high-bit masks, 64 bytes of 2-bit codes, 12 scale bytes, d. */
static void decode_q3_k_block(const unsigned char *block, float *out)
{
    const unsigned char *high_mask = block;
    const unsigned char *codes = block + 32;
    float d = half_at(block, 108);
    int scales[16];
    size_t e = 0;
    int half;

    unpack_q3_k_scales(block + 96, scales);
    for (half = 0; half < 2; half++) {
        int shift;
        for (shift = 0; shift < 8; shift += 2) {
            int chunk = 4 * half + shift / 2;
            int group;
            for (group = 0; group < 2; group++) {
                int k = 8 * half + 2 * (shift / 2) + group;
                float sub_scale = d * (float)scales[k];
                int l;
                for (l = 0; l < 16; l++) {
                    int low = (codes[32 * half + 16 * group + l] >> shift) & 3;
                    int offset = (high_mask[16 * group + l] & (1 << chunk)) ? 0 : 4;
                    out[e] = sub_scale * (float)(low - offset);
                    e++;
                }
            }
        }
    }
}

/** Unpack the twelve shared Q4_K/Q5_K bytes into eight 6-bit scales and mins. */
static void unpack_q45_k_scales(const unsigned char *packed, uint8_t *scales, uint8_t *mins)
{
    int j;

    for (j = 0; j < 4; j++) {
        scales[j] = (uint8_t)(packed[j] & 63);
        mins[j] = (uint8_t)(packed[j + 4] & 63);
    }
    for (j = 4; j < 8; j++) {
        scales[j] = (uint8_t)((packed[j + 4] & 15) | (packed[j - 4] >> 6) << 4);
        mins[j] = (uint8_t)((packed[j + 4] >> 4) | (packed[j] >> 6) << 4);
    }
}

/** 144 bytes: d, dmin, 12 scale bytes, 128 bytes of nibble pairs. */
static void decode_q4_k_block(const unsigned char *block, float *out)
{
    const unsigned char *codes = block + 16;
    float d = half_at(block, 0);
    float dmin = half_at(block, 2);
    uint8_t scales[8];
    uint8_t mins[8];
    size_t e = 0;
    int pair;

    unpack_q45_k_scales(block + 4, scales, mins);
    for (pair = 0; pair < 4; pair++) {
        int nibble;
        for (nibble = 0; nibble < 2; nibble++) {
            int k = 2 * pair + nibble;
            float sub_scale = d * (float)scales[k];
            float sub_min = dmin * (float)mins[k];
            int l;
            for (l = 0; l < 32; l++) {
                int c = codes[32 * pair + l];
                int code = nibble ? c >> 4 : c & 15;
                float scaled = sub_scale * (float)code;
                out[e] = scaled - sub_min;
                e++;
            }
        }
    }
}

/** 176 bytes: d, dmin, 12 scale bytes, 32 bytes of fifth bits, 128 bytes of nibbles. */
static void decode_q5_k_block(const unsigned char *block, float *out)
{
    const unsigned char *high_bits = block + 16;
    const unsigned char *codes = block + 48;
    float d = half_at(block, 0);
    float dmin = half_at(block, 2);
    uint8_t scales[8];
    uint8_t mins[8];
    size_t e = 0;
    int pair;

    unpack_q45_k_scales(block + 4, scales, mins);
    for (pair = 0; pair < 4; pair++) {
        int nibble;
        for (nibble = 0; nibble < 2; nibble++) {
            int k = 2 * pair + nibble;
            float sub_scale = d * (float)scales[k];
            float sub_min = dmin * (float)mins[k];
            int l;
            for (l = 0; l < 32; l++) {
                int c = codes[32 * pair + l];
                int code = nibble ? c >> 4 : c & 15;
                float scaled;
                if (high_bits[l] & (1 << k))
                    code += 16;
                scaled = sub_scale * (float)code;
                out[e] = scaled - sub_min;
                e++;
            }
        }
    }
}

/** 210 bytes: 128 bytes of low nibbles, 64 bytes of 2-bit highs, 16 int8 scales, d. */
static void decode_q6_k_block(const unsigned char *block, float *out)
{
    float d = half_at(block, 208);
    int half;

    for (half = 0; half < 2; half++) {
        const unsigned char *low = block + 64 * half;
        const unsigned char *high = block + 128 + 32 * half;
        const unsigned char *scales = block + 192 + 8 * half;
        float *base = out + 128 * half;
        int l;
        for (l = 0; l < 32; l++) {
            int t = l / 16;
            int q1 = ((low[l] & 15) | ((high[l] >> 0) & 3) << 4) - 32;
            int q2 = ((low[l + 32] & 15) | ((high[l] >> 2) & 3) << 4) - 32;
            int q3 = ((low[l] >> 4) | ((high[l] >> 4) & 3) << 4) - 32;
            int q4 = ((low[l + 32] >> 4) | ((high[l] >> 6) & 3) << 4) - 32;
            base[l] = (d * (float)(int8_t)scales[t]) * (float)q1;
            base[l + 32] = (d * (float)(int8_t)scales[t + 2]) * (float)q2;
            base[l + 64] = (d * (float)(int8_t)scales[t + 4]) * (float)q3;
            base[l + 96] = (d * (float)(int8_t)scales[t + 6]) * (float)q4;
        }
    }
}

typedef void (*superblock_decoder)(const unsigned char *block, float *out);

/** Decode `elements / 256` consecutive super-blocks, or refuse a partial one. */
static int decode_superblocks(superblock_decoder decode, size_t block_bytes,
                              const unsigned char *src, uint64_t elements, float *dst)
{
    uint64_t blocks;
    uint64_t b;

    if (elements % FT_GGML_K_SUPERBLOCK != 0)
        return -1;
    blocks = elements / FT_GGML_K_SUPERBLOCK;
    for (b = 0; b < blocks; b++)
        decode(src + b * block_bytes, dst + FT_GGML_K_SUPERBLOCK * b);
    return 0;
}

int ft_gguf_dequant_to_float(uint32_t ggml_type, const void *src, uint64_t elements, float *dst)
{
    const unsigned char *bytes = src;
    uint64_t i;

    if (src == NULL || dst == NULL)
        return -1;

    switch (ggml_type) {
    case FT_GGML_TYPE_F32:
        memcpy(dst, src, (size_t)(elements * 4u));
        return 0;
    case FT_GGML_TYPE_F16:
        for (i = 0; i < elements; i++) {
            uint16_t half;
            memcpy(&half, bytes + 2 * i, sizeof(half));
            dst[i] = ft_float_from_fp16(half);
        }
        return 0;
    case FT_GGML_TYPE_BF16:
        for (i = 0; i < elements; i++) {
            uint16_t bf16;
            memcpy(&bf16, bytes + 2 * i, sizeof(bf16));
            dst[i] = ft_float_from_bf16(bf16);
        }
        return 0;
    case FT_GGML_TYPE_Q8_0:
        if (elements % FT_GGML_Q8_0_BLOCK != 0)
            return -1;
        for (i = 0; i < elements / FT_GGML_Q8_0_BLOCK; i++)
            decode_q8_0_block(bytes + i * FT_GGML_Q8_0_BLOCK_BYTES, dst + FT_GGML_Q8_0_BLOCK * i);
        return 0;
    case FT_GGML_TYPE_Q2_K:
        return decode_superblocks(decode_q2_k_block, FT_GGML_Q2_K_BLOCK_BYTES, bytes, elements, dst);
    case FT_GGML_TYPE_Q3_K:
        return decode_superblocks(decode_q3_k_block, FT_GGML_Q3_K_BLOCK_BYTES, bytes, elements, dst);
    case FT_GGML_TYPE_Q4_K:
        return decode_superblocks(decode_q4_k_block, FT_GGML_Q4_K_BLOCK_BYTES, bytes, elements, dst);
    case FT_GGML_TYPE_Q5_K:
        return decode_superblocks(decode_q5_k_block, FT_GGML_Q5_K_BLOCK_BYTES, bytes, elements, dst);
    case FT_GGML_TYPE_Q6_K:
        return decode_superblocks(decode_q6_k_block, FT_GGML_Q6_K_BLOCK_BYTES, bytes, elements, dst);
    default:
        return -1;
    }
}
