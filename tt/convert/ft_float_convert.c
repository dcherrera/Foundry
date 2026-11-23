/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_float_convert.c
 * @brief Implementation of the fp32 / fp16 / bf16 conversions.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_float_convert.h"

#include <string.h>

static float float_from_bits(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t bits_from_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float ft_float_from_fp16(uint16_t half)
{
    uint32_t sign = (uint32_t)(half >> 15) << 31;
    uint32_t exp = (half >> 10) & 0x1fu;
    uint32_t frac = half & 0x3ffu;

    if (exp == 0) {
        if (frac == 0)
            return float_from_bits(sign);

        /* Denormal: renormalize by shifting the leading 1 into place. */
        exp = 1;
        while ((frac & 0x400u) == 0) {
            frac <<= 1;
            exp--;
        }
        frac &= 0x3ffu;
        return float_from_bits(sign | ((exp + 127u - 15u) << 23) | (frac << 13));
    }

    if (exp == 0x1fu)
        return float_from_bits(sign | 0x7f800000u | (frac << 13));

    return float_from_bits(sign | ((exp + 127u - 15u) << 23) | (frac << 13));
}

uint16_t ft_bf16_from_float(float value)
{
    uint32_t bits = bits_from_float(value);
    uint32_t rounded;

    /* NaN must stay NaN: rounding a NaN whose only set fraction bits are in
     * the discarded half would carry it into an infinity. */
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
        return (uint16_t)((bits >> 16) | 0x0040u);

    /* Round to nearest, ties to even, on the 16 bits being dropped. */
    rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

float ft_float_from_bf16(uint16_t bf16)
{
    return float_from_bits((uint32_t)bf16 << 16);
}

void ft_bf16_from_float_array(const float *src, uint16_t *dst, size_t count)
{
    if (!src || !dst)
        return;

    for (size_t i = 0; i < count; i++)
        dst[i] = ft_bf16_from_float(src[i]);
}

void ft_float_from_bf16_array(const uint16_t *src, float *dst, size_t count)
{
    if (!src || !dst)
        return;

    for (size_t i = 0; i < count; i++)
        dst[i] = ft_float_from_bf16(src[i]);
}
