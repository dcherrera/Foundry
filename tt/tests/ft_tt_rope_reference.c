/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_rope_reference.c
 * @brief Implementation of the RoPE reference and the RoPE device model. See
 *   ft_tt_rope_reference.h for why they are two functions and not one.
 *
 * Nothing here includes ft_tt_op_rope.h. That is the point: the reference is written from
 * the CUDA kernel and the CPU oracle, so a shared mistake between the driver and its own
 * check is impossible rather than unlikely.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_tt_rope_reference.h"

#include <math.h>

#include "../convert/ft_float_convert.h"

void ft_tt_rope_reference_row(float *row, uint32_t heads, uint32_t head_dim,
                              uint32_t rope_dim, uint32_t position, float theta_base)
{
    const uint32_t rot = (rope_dim > 0u && rope_dim <= head_dim) ? rope_dim : head_dim;
    const uint32_t half = rot / 2u;
    uint32_t head;

    for (head = 0; head < heads; head++) {
        float *values = row + (size_t)head * head_dim;
        uint32_t j;

        for (j = 0; j < half; j++) {
            /* fp32 powf/cosf/sinf, per j, exactly as rope.cu does it — so the reference has
             * the SAME numerical shape as the K80 path this op has to match, rather than a
             * better one that would make a real divergence look like a rounding. */
            const float frequency =
                1.0f / powf(theta_base, (float)(2u * j) / (float)rot);
            const float angle = (float)position * frequency;
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            const float low = values[j];
            const float high = values[j + half];

            values[j] = low * cosine - high * sine;
            values[j + half] = low * sine + high * cosine;
        }
    }
}

void ft_tt_rope_reference_apply(const float *in, float *out, uint32_t rows, uint32_t heads,
                                uint32_t head_dim, uint32_t rope_dim,
                                uint32_t block_first_position, float theta_base)
{
    const size_t width = (size_t)heads * head_dim;
    uint32_t row;

    for (row = 0; row < rows; row++) {
        size_t index;

        for (index = 0; index < width; index++)
            out[(size_t)row * width + index] = in[(size_t)row * width + index];
        ft_tt_rope_reference_row(out + (size_t)row * width, heads, head_dim, rope_dim,
                                 block_first_position + row, theta_base);
    }
}

void ft_tt_rope_model_apply(const uint16_t *in, const uint16_t *cos_table,
                            const uint16_t *sin_table, uint16_t *out, uint32_t rows,
                            uint32_t heads, uint32_t head_dim, uint32_t rot)
{
    const size_t width = (size_t)heads * head_dim;
    const uint32_t half = rot / 2u;
    uint32_t row;

    for (row = 0; row < rows; row++) {
        uint32_t head;

        for (head = 0; head < heads; head++) {
            const size_t base = (size_t)row * width + (size_t)head * head_dim;
            uint32_t column;

            for (column = 0; column < head_dim; column++) {
                if (column < rot) {
                    const uint32_t partner =
                        (column < half) ? (column + half) : (column - half);
                    const float value = ft_float_from_bf16(in[base + column]);
                    const float other = ft_float_from_bf16(in[base + partner]);
                    const float cosine =
                        ft_float_from_bf16(cos_table[(size_t)row * rot + column]);
                    const float sine =
                        ft_float_from_bf16(sin_table[(size_t)row * rot + column]);
                    /* Each product is packed to bf16 in its own staging circular buffer
                     * before the add, so the model rounds twice on the way in and once on
                     * the way out — exactly where resident_rope_compute.c packs. */
                    const float first = ft_float_from_bf16(ft_bf16_from_float(value * cosine));
                    const float second = ft_float_from_bf16(ft_bf16_from_float(other * sine));

                    out[base + column] = ft_bf16_from_float(first + second);
                } else {
                    out[base + column] = in[base + column];
                }
            }
        }
    }
}

void ft_tt_rope_reference_table(uint16_t *cos_table, uint16_t *sin_table,
                                uint32_t block_first_position, uint32_t rot,
                                double theta_base)
{
    const uint32_t half = rot / 2u;
    uint32_t row;

    for (row = 0; row < 32u; row++) {
        const uint32_t position = block_first_position + row;
        uint32_t column;

        for (column = 0; column < rot; column++) {
            const uint32_t j = (half == 0u) ? 0u : (column % half);
            const double frequency = pow(theta_base, -2.0 * (double)j / (double)rot);
            const double angle = (double)position * frequency;
            const double sine = sin(angle);

            cos_table[(size_t)row * rot + column] = ft_bf16_from_float((float)cos(angle));
            sin_table[(size_t)row * rot + column] =
                ft_bf16_from_float((float)((column < half) ? -sine : sine));
        }
    }
}
