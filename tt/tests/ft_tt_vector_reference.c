/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_vector_reference.c
 * @brief Implementation of the vector op family's references, tile-space model and
 *   metrics. See ft_tt_vector_reference.h for what the two levels are for.
 *
 * @status Green in test_vector_ops.c. ft_tt_vector_model_reduce_program() added 2026-08-15
 *   reproduces all four member models BIT-EXACTLY at an exact W, which is what licenses
 *   using it to score a padded one. The models predicted silicon to ~6 decimals again:
 *   model softmax Wt=48 0.999985077 against the card's 0.999985087, model RMS norm
 *   0.999996367 against 0.999996299, model layer norm 0.999993965 against 0.999993648.
 * @issues None
 * @todo None
 */

#include "ft_tt_vector_reference.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_l1_layout.h"

/** Elements in one 32x32 tile — the granularity every model here walks at. */
#define TILE_ELEMENTS (FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM)

/** One value through a circular buffer: fp32 out of DST, bf16 into L1, fp32 back out. */
static float through_bf16(float value)
{
    return ft_float_from_bf16(ft_bf16_from_float(value));
}

/* ================================================================
 * THE TWO NUMBER FORMATS A REDUCE PROGRAM CAN BE PLACED AT
 *
 * A CB table decides two SEPARATE roundings and the bf16 case hides the difference because
 * one of them is the identity there:
 *
 *   PACK    a value leaving DST for L1, and equally a host float rendered into the DRAM
 *           strip the reader pulls. bf16 rounds; Float32 is exact.
 *   UNPACK  a value leaving L1 for SrcA/SrcB. bf16 is exact — it is already bf16 — but
 *           **Float32 is NOT fp32 in the source registers: the FPU sees Tf32, 11 mantissa
 *           bits, and TRUNCATES rather than rounds.**
 *
 * So `pack` alone models the bf16 family (which is why every model below was written with
 * one function) and BOTH are needed to say what an all-Float32 table can actually buy. Every
 * read of a circular buffer goes through `load()` = unpack(pack(v)), which is exactly
 * `through_bf16` at bf16 — that identity is what makes this refactor bit-preserving for the
 * four members that were validated against silicon to six decimals.
 * ================================================================ */

typedef struct reduce_number_format {
    float (*pack)(float);   /**< DST -> L1, or host -> the DRAM strip */
    float (*unpack)(float); /**< L1 -> the source registers */
} reduce_number_format;

static float format_identity(float value)
{
    return value;
}

/**
 * Tf32 as the unpacker delivers it: **E8M10** — sign, the 8-bit exponent and TEN STORED
 * MANTISSA BITS (eleven of significand with the hidden one), the remaining THIRTEEN
 * discarded rather than rounded.
 *
 * The width is not inferred. `tensix_types.h` in the staged tt-metal tree spells it —
 * "Tf32 = 4, // E8M10   Stored in L1 in 32b container as : {1b sign, 8b exp, 10b man,
 * 13'b0}" — and the toolchain CONFIRMS the format reaches the source registers: a
 * `reduce_float32.cbtable` build emits `ft_cb_unpack_src_format = 0` (Float32) and
 * `ft_cb_unpack_dst_format = 4` (Tf32) into `ft_tensix_cb_formats.h`, against 5 and 5 (Float16_b) for
 * the bf16 table. That is the generated evidence that an all-Float32 program still does not
 * do fp32 arithmetic.
 *
 * Truncation, not rounding, is the load-bearing detail. It costs up to one ulp — between
 * 2^-11 and 2^-10, so ~4.9e-4 typical — and it is BIASED, every magnitude moving toward
 * zero, so the errors of a long accumulation add instead of cancelling. That is the same
 * signature measured on the GDN value path, where `card/want` sat at 0.9955-0.9977 and was
 * BELOW 1 on 768 of 768 head-layers.
 */
static float unpack_tf32(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    bits &= 0xFFFFE000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static const reduce_number_format g_format_float16b = { through_bf16, format_identity };
static const reduce_number_format g_format_float32 = { format_identity, unpack_tf32 };

/** THE COUNTERFACTUAL. Not a table anyone can write — the hardware has no fp32 unpack — and
 *  it exists only to ATTRIBUTE: whatever a Float32 program's residue is, running the same
 *  program with this format says how much of it was Tf32 and how much was everything else. */
static const reduce_number_format g_format_exact = { format_identity, format_identity };

/** One circular-buffer read: what the source registers see of a stored value. */
static float load(const reduce_number_format *format, float value)
{
    return format->unpack(format->pack(value));
}

static float sigmoid_of(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

/* ================================================================
 * References
 * ================================================================ */

void ft_tt_vector_reference_eltwise(ft_tt_op_vector_function function, const float *a,
                                    const float *b, float *out, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++) {
        const float x = a[index];
        const float y = b ? b[index] : 0.0f;

        switch (function) {
        case FT_TT_OP_VECTOR_FN_SILU:
            out[index] = x * sigmoid_of(x);
            break;
        case FT_TT_OP_VECTOR_FN_SIGMOID:
            out[index] = sigmoid_of(x);
            break;
        case FT_TT_OP_VECTOR_FN_ADD:
            out[index] = x + y;
            break;
        case FT_TT_OP_VECTOR_FN_MUL:
            out[index] = x * y;
            break;
        case FT_TT_OP_VECTOR_FN_SUB:
            out[index] = x - y;
            break;
        case FT_TT_OP_VECTOR_FN_SILU_MUL:
            out[index] = (x * sigmoid_of(x)) * y;
            break;
        default:
            out[index] = 0.0f;
            break;
        }
    }
}

void ft_tt_vector_reference_softmax(const float *x, float *out, uint32_t rows, uint32_t cols)
{
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        float *destination = out + (size_t)row * cols;
        float maximum = source[0];
        float total = 0.0f;
        uint32_t column;

        for (column = 1; column < cols; column++) {
            if (source[column] > maximum)
                maximum = source[column];
        }
        for (column = 0; column < cols; column++) {
            destination[column] = expf(source[column] - maximum);
            total += destination[column];
        }
        for (column = 0; column < cols; column++)
            destination[column] /= total;
    }
}

void ft_tt_vector_reference_rms_norm(const float *x, const float *gamma, float eps, float *out,
                                     uint32_t rows, uint32_t cols)
{
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        double sum = 0.0;
        double scale;
        uint32_t column;

        for (column = 0; column < cols; column++)
            sum += (double)source[column] * (double)source[column];
        scale = 1.0 / sqrt(sum / (double)cols + (double)eps);
        for (column = 0; column < cols; column++)
            out[(size_t)row * cols + column] =
                (float)((double)source[column] * scale) * gamma[column];
    }
}

void ft_tt_vector_reference_layer_norm(const float *x, const float *gamma, const float *beta,
                                       float eps, float *out, uint32_t rows, uint32_t cols)
{
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        double mean = 0.0;
        double variance = 0.0;
        double scale;
        uint32_t column;

        for (column = 0; column < cols; column++)
            mean += (double)source[column];
        mean /= (double)cols;
        for (column = 0; column < cols; column++) {
            const double centred = (double)source[column] - mean;

            variance += centred * centred;
        }
        variance /= (double)cols;
        scale = 1.0 / sqrt(variance + (double)eps);
        for (column = 0; column < cols; column++)
            out[(size_t)row * cols + column] =
                (float)(((double)source[column] - mean) * scale * (double)gamma[column] +
                        (double)beta[column]);
    }
}

void ft_tt_vector_reference_router_normalise(const float *x, float *out, uint32_t rows,
                                             uint32_t cols)
{
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        double total = 0.0;
        uint32_t column;

        for (column = 0; column < cols; column++)
            total += (double)source[column];
        for (column = 0; column < cols; column++)
            out[(size_t)row * cols + column] = (float)((double)source[column] / total);
    }
}

/* ================================================================
 * The tile-space model
 * ================================================================ */

void ft_tt_vector_model_eltwise(ft_tt_op_vector_function function, const float *a,
                                const float *b, float *out, size_t count)
{
    const size_t tiles = count / TILE_ELEMENTS;
    size_t tile;

    for (tile = 0; tile < tiles; tile++) {
        const size_t base = tile * TILE_ELEMENTS;
        size_t index;

        for (index = 0; index < TILE_ELEMENTS; index++) {
            /* The operands came off DRAM as bf16 tiles, so that is what the unpacker sees. */
            const float x = through_bf16(a[base + index]);
            const float y = b ? through_bf16(b[base + index]) : 0.0f;
            float result;

            switch (function) {
            case FT_TT_OP_VECTOR_FN_SILU:
                result = x * sigmoid_of(x);
                break;
            case FT_TT_OP_VECTOR_FN_SIGMOID:
                result = sigmoid_of(x);
                break;
            case FT_TT_OP_VECTOR_FN_ADD:
                result = x + y;
                break;
            case FT_TT_OP_VECTOR_FN_MUL:
                result = x * y;
                break;
            case FT_TT_OP_VECTOR_FN_SUB:
                result = x - y;
                break;
            case FT_TT_OP_VECTOR_FN_SILU_MUL:
                /* Two packs, not one: silu(a) goes through a scratch CB page before the
                 * multiply, so it is rounded to bf16 on the way. That extra rounding is the
                 * numerical price of the fusion, and it is why the fused arm scores
                 * slightly below a hypothetical single-DST one. */
                result = through_bf16(x * sigmoid_of(x)) * y;
                break;
            default:
                result = 0.0f;
                break;
            }
            out[base + index] = through_bf16(result);
        }
    }
}

void ft_tt_vector_model_softmax(const float *x, float *out, uint32_t rows, uint32_t cols)
{
    const uint32_t tile_cols = cols / FT_TT_OP_TILE_DIM;
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        float *destination = out + (size_t)row * cols;
        float maximum;
        float total = 0.0f;
        float reciprocal;
        uint32_t column;
        uint32_t wt;

        /* Pass 1: reduce_tile<MAX, REDUCE_ROW> accumulated across all tile_cols tiles into
         * one fp32 destination register, then packed to bf16 in cb_max. Walking the tiles
         * explicitly rather than the row is the point: a kernel that reduced over ONE tile
         * would agree with a row-wise model on tile_cols == 1 and disagree here. */
        maximum = through_bf16(source[0]);
        for (wt = 0; wt < tile_cols; wt++) {
            for (column = 0; column < FT_TT_OP_TILE_DIM; column++) {
                const float value = through_bf16(source[wt * FT_TT_OP_TILE_DIM + column]);

                if (value > maximum)
                    maximum = value;
            }
        }
        maximum = through_bf16(maximum);

        /* Pass 2: exp(x - max) in fp32, packed to bf16 in cb_exps.
         * Pass 3: the row sum accumulates the ROUNDED exponentials, in fp32, then one
         * reciprocal, packed to bf16 in cb_recip. */
        for (column = 0; column < cols; column++) {
            destination[column] = through_bf16(expf(through_bf16(source[column]) - maximum));
            total += destination[column];
        }
        reciprocal = through_bf16(1.0f / total);

        /* Pass 4: mul_tiles_bcast_cols, both operands bf16 out of L1, result packed bf16. */
        for (column = 0; column < cols; column++)
            destination[column] = through_bf16(destination[column] * reciprocal);
    }
}

/**
 * The row reduce as the kernel performs it: fp32 accumulation in DST across all tile_cols
 * tiles of bf16 operands, times the scaler tile, then eps as one more accumulated term.
 *
 * Walking the TILES rather than the row is the whole point — a kernel that reduced over one
 * tile agrees with a row-wise model at tile_cols == 1 and disagrees everywhere else.
 */
static float model_row_sum(const reduce_number_format *format, const float *source,
                           uint32_t tile_cols, float scaler, float eps_tile, bool add_eps,
                           float centre, bool square)
{
    const float unpacked_scaler = load(format, scaler);
    float total = 0.0f;
    uint32_t wt;

    for (wt = 0; wt < tile_cols; wt++) {
        uint32_t column;

        for (column = 0; column < FT_TT_OP_TILE_DIM; column++) {
            float value = load(format, source[wt * FT_TT_OP_TILE_DIM + column]) - centre;

            if (square)
                value = format->pack(value * value);
            total += load(format, value) * unpacked_scaler;
        }
    }
    if (add_eps)
        total += (float)FT_TT_OP_TILE_DIM * load(format, eps_tile) * unpacked_scaler;
    return total;
}

void ft_tt_vector_model_rms_norm(const float *x, const float *gamma, float eps, float *out,
                                 uint32_t rows, uint32_t cols)
{
    const uint32_t tile_cols = cols / FT_TT_OP_TILE_DIM;
    const float scaler = through_bf16(1.0f / (float)cols);
    const float eps_tile = through_bf16(eps * (float)cols / (float)FT_TT_OP_TILE_DIM);
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        float *destination = out + (size_t)row * cols;
        /* Pass 1: x*x packed to bf16 in cb_mid, then the row sum with eps, then rsqrt. */
        const float mean_square =
            model_row_sum(&g_format_float16b, source, tile_cols, scaler, eps_tile, true, 0.0f,
                          true);
        const float scale = through_bf16(1.0f / sqrtf(mean_square));
        uint32_t column;

        /* The chain: x * gamma packed to bf16, then * scale packed to bf16. */
        for (column = 0; column < cols; column++) {
            const float weighted =
                through_bf16(through_bf16(source[column]) * through_bf16(gamma[column]));

            destination[column] = through_bf16(weighted * scale);
        }
    }
}

void ft_tt_vector_model_layer_norm(const float *x, const float *gamma, const float *beta,
                                   float eps, float *out, uint32_t rows, uint32_t cols)
{
    const uint32_t tile_cols = cols / FT_TT_OP_TILE_DIM;
    const float scaler = through_bf16(1.0f / (float)cols);
    const float eps_tile = through_bf16(eps * (float)cols / (float)FT_TT_OP_TILE_DIM);
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        float *destination = out + (size_t)row * cols;
        /* Pass 1: the row mean, packed to bf16 in cb_col0. */
        const float mean =
            through_bf16(model_row_sum(&g_format_float16b, source, tile_cols, scaler, 0.0f,
                                       false, 0.0f, false));
        /* Pass 2: (x - mean)^2 packed to bf16 in cb_mid, the row sum with eps, then rsqrt. */
        const float variance =
            model_row_sum(&g_format_float16b, source, tile_cols, scaler, eps_tile, true, mean,
                          true);
        const float scale = through_bf16(1.0f / sqrtf(variance));
        uint32_t column;

        /* The chain, four packs: centre, gamma, scale, beta. */
        for (column = 0; column < cols; column++) {
            float value = through_bf16(through_bf16(source[column]) - mean);

            value = through_bf16(value * through_bf16(gamma[column]));
            value = through_bf16(value * scale);
            destination[column] = through_bf16(value + through_bf16(beta[column]));
        }
    }
}

void ft_tt_vector_model_router_normalise(const float *x, float *out, uint32_t rows, uint32_t cols)
{
    const uint32_t tile_cols = cols / FT_TT_OP_TILE_DIM;
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *source = x + (size_t)row * cols;
        float *destination = out + (size_t)row * cols;
        const float total = model_row_sum(&g_format_float16b, source, tile_cols, 1.0f, 0.0f,
                                          false, 0.0f, false);
        const float reciprocal = through_bf16(1.0f / total);
        uint32_t column;

        for (column = 0; column < cols; column++)
            destination[column] = through_bf16(through_bf16(source[column]) * reciprocal);
    }
}

/** One byte of a program word, the same way kernels/resident_reduce_protocol.h reads it. */
static uint32_t program_byte(uint32_t word, uint32_t index)
{
    return (word >> (index * 8u)) & 0xFFu;
}

/** The MAX pool over a whole row, tile by tile — the reduce accumulates across tiles. */
static float model_row_max(const reduce_number_format *format, const float *source,
                           uint32_t tile_cols)
{
    float maximum = load(format, source[0]);
    uint32_t wt;

    for (wt = 0; wt < tile_cols; wt++) {
        uint32_t column;

        for (column = 0; column < FT_TT_OP_TILE_DIM; column++) {
            const float value = load(format, source[wt * FT_TT_OP_TILE_DIM + column]);

            if (value > maximum)
                maximum = value;
        }
    }
    return format->pack(maximum);
}

/** Both public entry points; `format` is the whole of what the CB table decides. */
static void model_reduce_program(const reduce_number_format *format, uint32_t pass0_word,
                                 uint32_t pass1_word, uint32_t final_word, const float *x,
                                 const float *gamma, const float *beta, float scaler,
                                 float eps_tile, float *out, uint32_t rows, uint32_t cols)
{
    const uint32_t pass_word[2] = { pass0_word, pass1_word };
    const uint32_t pass_count = program_byte(final_word, FT_TT_OP_REDUCE_BYTE_PASS_COUNT);
    const uint32_t chain_source = program_byte(final_word, FT_TT_OP_REDUCE_BYTE_SOURCE);
    const uint32_t chain = program_byte(final_word, FT_TT_OP_REDUCE_BYTE_CHAIN);
    const uint32_t tile_cols = (cols == 0u) ? 0u : cols / FT_TT_OP_TILE_DIM;
    float *mid = NULL;
    bool runnable = pass_count >= 1u && pass_count <= 2u && tile_cols != 0u &&
                    (cols % FT_TT_OP_TILE_DIM) == 0u;
    uint32_t row;

    if (runnable) {
        mid = malloc((size_t)cols * sizeof(float));
        runnable = mid != NULL;
    }
    if (!runnable) {
        size_t index;

        /* A program this cannot run is NaN, never a plausible number: the score's
         * non-finite census names it, and a silent zero would read as an answer. */
        for (index = 0; index < (size_t)rows * (size_t)cols; index++)
            out[index] = (float)NAN;
        free(mid);
        return;
    }

    for (row = 0; row < rows; row++) {
        const float *x_row = x + (size_t)row * cols;
        float *out_row = out + (size_t)row * cols;
        float column[2] = { 0.0f, 0.0f };
        bool mid_built = false;
        const float *chain_stream;
        uint32_t index;
        uint32_t element;

        for (index = 0; index < pass_count; index++) {
            const uint32_t pool = program_byte(pass_word[index], FT_TT_OP_REDUCE_BYTE_POOL);
            const uint32_t map = program_byte(pass_word[index], FT_TT_OP_REDUCE_BYTE_MAP);
            const uint32_t verb = program_byte(pass_word[index], FT_TT_OP_REDUCE_BYTE_VERB);
            const uint32_t add_eps = program_byte(pass_word[index], FT_TT_OP_REDUCE_BYTE_ADD_EPS);
            const float *stream = x_row;
            float value;

            /* The MAP builds cb_mid out of cb_in, one pack per tile. */
            if (map != FT_TT_OP_REDUCE_MAP_NONE) {
                for (element = 0; element < cols; element++) {
                    const float source = load(format, x_row[element]);
                    const float centre = load(format, column[0]);

                    if (map == FT_TT_OP_REDUCE_MAP_SQUARE_SELF)
                        mid[element] = format->pack(source * source);
                    else if (map == FT_TT_OP_REDUCE_MAP_SUB_EXP)
                        mid[element] = format->pack(expf(source - centre));
                    else
                        mid[element] = format->pack((source - centre) * (source - centre));
                }
                stream = mid;
                mid_built = true;
            }

            if (pool == FT_TT_OP_REDUCE_POOL_MAX) {
                /* Upstream ignores the reduce factor on the MAX arm, so the scaler tile is
                 * not applied here even though the same tile is the SrcB operand. */
                value = model_row_max(format, stream, tile_cols);
            } else {
                value = model_row_sum(format, stream, tile_cols, scaler, eps_tile,
                                      add_eps != 0u, 0.0f, false);
            }

            if (verb == FT_TT_OP_REDUCE_VERB_RECIP)
                value = 1.0f / value;
            else if (verb == FT_TT_OP_REDUCE_VERB_RSQRT)
                value = 1.0f / sqrtf(value);
            column[index] = format->pack(value);
        }

        chain_stream = (chain_source == FT_TT_OP_REDUCE_SOURCE_MID) ? mid : x_row;
        if (chain_source == FT_TT_OP_REDUCE_SOURCE_MID && !mid_built) {
            for (element = 0; element < cols; element++)
                out_row[element] = (float)NAN;
            continue;
        }

        /* The final chain, lowest flag first: every step is one CB round trip, so the value
         * is UNPACKED into the source registers and PACKED back out once per flag. */
        for (element = 0; element < cols; element++) {
            float value = format->pack(chain_stream[element]);

            if ((chain & FT_TT_OP_REDUCE_CHAIN_CENTER) != 0u)
                value = format->pack(format->unpack(value) - load(format, column[0]));
            if ((chain & FT_TT_OP_REDUCE_CHAIN_GAMMA) != 0u)
                value = format->pack(format->unpack(value) * load(format, gamma[element]));
            if ((chain & FT_TT_OP_REDUCE_CHAIN_SCALE) != 0u)
                value = format->pack(format->unpack(value) *
                                     load(format, column[pass_count - 1u]));
            if ((chain & FT_TT_OP_REDUCE_CHAIN_BETA) != 0u)
                value = format->pack(format->unpack(value) + load(format, beta[element]));
            out_row[element] = value;
        }
    }
    free(mid);
}

void ft_tt_vector_model_reduce_program(uint32_t pass0_word, uint32_t pass1_word,
                                       uint32_t final_word, const float *x, const float *gamma,
                                       const float *beta, float scaler, float eps_tile,
                                       float *out, uint32_t rows, uint32_t cols)
{
    model_reduce_program(&g_format_float16b, pass0_word, pass1_word, final_word, x, gamma, beta,
                         scaler, eps_tile, out, rows, cols);
}

void ft_tt_vector_model_reduce_program_float32(uint32_t pass0_word, uint32_t pass1_word,
                                               uint32_t final_word, const float *x,
                                               const float *gamma, const float *beta,
                                               float scaler, float eps_tile, float *out,
                                               uint32_t rows, uint32_t cols)
{
    model_reduce_program(&g_format_float32, pass0_word, pass1_word, final_word, x, gamma, beta,
                         scaler, eps_tile, out, rows, cols);
}

void ft_tt_vector_model_reduce_program_exact(uint32_t pass0_word, uint32_t pass1_word,
                                             uint32_t final_word, const float *x,
                                             const float *gamma, const float *beta,
                                             float scaler, float eps_tile, float *out,
                                             uint32_t rows, uint32_t cols)
{
    model_reduce_program(&g_format_exact, pass0_word, pass1_word, final_word, x, gamma, beta,
                         scaler, eps_tile, out, rows, cols);
}

/* ================================================================
 * THE TWO GDN HEAD NORMS AT Float32, EVERY TRUNCATION STAGE UNDER A SWITCH
 *
 * See ft_tt_vector_reference.h for what each arm means. The invariant that makes any of it
 * worth reading: at `arms = 0` this reproduces `model_reduce_program(&g_format_float32, ...)`
 * BIT FOR BIT, and test_vector_ops.c checks exactly that before it reports an arm.
 * ================================================================ */

/** The mantissa this arm's unpack keeps — ten stored bits, or the nine-bit reading. */
static uint32_t head_norm_mask(uint32_t arms)
{
    return (arms & FT_TT_REDUCE_ARM_TF32_NINE_BITS) != 0u ? 0xFFFFC000u : 0xFFFFE000u;
}

/** Round to the NEAREST representable value under `mask`, ties to even — what an SFPU
 *  pre-round before the PACK gives, and the only way a value reaches the source registers
 *  without a downward bias. */
static float round_nearest_masked(float value, uint32_t mask)
{
    const uint32_t half = (~mask) >> 1; /* one below the mask's lowest kept bit */
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    if (((bits >> 23) & 0xFFu) == 0xFFu)
        return value; /* inf and NaN carry no mantissa to round */
    /* `half + 1` IS the mask's lowest kept bit, so the tie-to-even term needs no shift
     * count and no second spelling of the width. */
    bits += half + (((bits & (half + 1u)) != 0u) ? 1u : 0u);
    bits &= mask;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

float ft_tt_vector_sfpu_rsqrt(float x)
{
    uint32_t bits;
    float y;
    float xy;
    float c;
    float one_minus_xyy;

    /* Seed: y = as<float>(MAGIC - (bits(x) >> 1)), MAGIC = FT_SFPU_SQRT_SEED_MAGIC. */
    memcpy(&bits, &x, sizeof(bits));
    bits = 0x5F1110A0u - (bits >> 1);
    memcpy(&y, &bits, sizeof(y));

    /* One quadratic refinement of the reciprocal root, then one Newton step — the kernel's
     * two guards only cover x <= 0 and x = inf, which a sum of squares plus eps is not. */
    xy = x * y;
    c = (-y) * xy;
    y = y * (2.2825186f + c * (2.2533049f + c));

    xy = x * y;
    one_minus_xyy = (-y) * xy + 1.0f;
    return one_minus_xyy * (y * 0.5f) + y;
}

/** The head-norm arm's rsqrt, whichever one the flags select. */
static float head_norm_rsqrt(float value, uint32_t arms)
{
    float result = (arms & FT_TT_REDUCE_ARM_SFPU_RSQRT) != 0u ? ft_tt_vector_sfpu_rsqrt(value)
                                                              : 1.0f / sqrtf(value);

    if ((arms & FT_TT_REDUCE_ARM_RSQRT_PERTURB) != 0u)
        result *= 1.001f;
    return result;
}

/** One value into the source registers, unless this arm makes that stage exact. */
static float head_norm_unpack(float value, uint32_t exact_flag, uint32_t arms)
{
    uint32_t bits;

    if (exact_flag != 0u && (arms & exact_flag) != 0u)
        return value;
    memcpy(&bits, &value, sizeof(bits));
    bits &= head_norm_mask(arms);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/** A value the COMPUTE kernel packs out of DST. */
static float head_norm_store_kernel(float value, uint32_t arms)
{
    return (arms & FT_TT_REDUCE_ARM_ROUND_NEAREST) != 0u
               ? round_nearest_masked(value, head_norm_mask(arms))
               : value;
}

/** A value the HOST renders into the fp32 strip or the constants buffer. */
static float head_norm_store_host(float value, uint32_t arms)
{
    return (arms & FT_TT_REDUCE_ARM_HOST_ROUND) != 0u
               ? round_nearest_masked(value, head_norm_mask(arms))
               : value;
}

/** Is this the one program shape this arm models? Anything else is REFUSED, not approximated
 *  — a model that quietly runs a program it was not written for is a vacuous gate. */
static bool head_norm_program_is_modelled(uint32_t pass0_word, uint32_t final_word,
                                          const float *gamma, uint32_t arms)
{
    const uint32_t chain = program_byte(final_word, FT_TT_OP_REDUCE_BYTE_CHAIN);
    const bool wants_gamma = (chain & FT_TT_OP_REDUCE_CHAIN_GAMMA) != 0u &&
                             (arms & FT_TT_REDUCE_ARM_NO_GAMMA) == 0u;

    if (program_byte(final_word, FT_TT_OP_REDUCE_BYTE_PASS_COUNT) != 1u)
        return false;
    if (program_byte(final_word, FT_TT_OP_REDUCE_BYTE_SOURCE) != FT_TT_OP_REDUCE_SOURCE_IN)
        return false;
    if ((chain & ~(FT_TT_OP_REDUCE_CHAIN_GAMMA | FT_TT_OP_REDUCE_CHAIN_SCALE)) != 0u)
        return false;
    if ((chain & FT_TT_OP_REDUCE_CHAIN_SCALE) == 0u)
        return false;
    if (program_byte(pass0_word, FT_TT_OP_REDUCE_BYTE_POOL) != FT_TT_OP_REDUCE_POOL_SUM)
        return false;
    if (program_byte(pass0_word, FT_TT_OP_REDUCE_BYTE_MAP) != FT_TT_OP_REDUCE_MAP_SQUARE_SELF)
        return false;
    if (program_byte(pass0_word, FT_TT_OP_REDUCE_BYTE_VERB) != FT_TT_OP_REDUCE_VERB_RSQRT)
        return false;
    /* A gamma step with no gamma is a wrong answer with nothing to read. */
    return !wants_gamma || gamma != NULL;
}

void ft_tt_vector_model_head_norm_float32(uint32_t pass0_word, uint32_t pass1_word,
                                          uint32_t final_word, const float *x,
                                          const float *gamma, float scaler, float eps_tile,
                                          float *out, uint32_t rows, uint32_t cols,
                                          uint32_t arms)
{
    const uint32_t chain = program_byte(final_word, FT_TT_OP_REDUCE_BYTE_CHAIN);
    const bool uses_gamma = (chain & FT_TT_OP_REDUCE_CHAIN_GAMMA) != 0u &&
                            (arms & FT_TT_REDUCE_ARM_NO_GAMMA) == 0u;
    const uint32_t add_eps = program_byte(pass0_word, FT_TT_OP_REDUCE_BYTE_ADD_EPS);
    float *mid = NULL;
    uint32_t row;

    (void)pass1_word; /* one pass; the refusal above is what enforces that */

    if (cols != 0u && (cols % FT_TT_OP_TILE_DIM) == 0u &&
        head_norm_program_is_modelled(pass0_word, final_word, gamma, arms))
        mid = malloc((size_t)cols * sizeof(float));
    if (!mid) {
        size_t index;

        for (index = 0; index < (size_t)rows * (size_t)cols; index++)
            out[index] = (float)NAN;
        return;
    }

    for (row = 0; row < rows; row++) {
        const float *x_row = x + (size_t)row * cols;
        float *out_row = out + (size_t)row * cols;
        const float unpacked_scaler =
            head_norm_unpack(head_norm_store_host(scaler, arms), 0u, arms);
        float total = 0.0f;
        float column;
        uint32_t element;

        /* The MAP: cb_mid = x*x, one kernel pack per element. */
        for (element = 0; element < cols; element++) {
            const float value = head_norm_unpack(head_norm_store_host(x_row[element], arms),
                                                 FT_TT_REDUCE_ARM_EXACT_MAP, arms);

            mid[element] = head_norm_store_kernel(value * value, arms);
        }
        /* The SUM reduce, the scaler contributed per reduce position (spec §4). */
        for (element = 0; element < cols; element++)
            total += head_norm_unpack(mid[element], FT_TT_REDUCE_ARM_EXACT_MID, arms) *
                     unpacked_scaler;
        if (add_eps != 0u)
            total += (float)FT_TT_OP_TILE_DIM *
                     head_norm_unpack(head_norm_store_host(eps_tile, arms), 0u, arms) *
                     unpacked_scaler;
        column = head_norm_store_kernel(head_norm_rsqrt(total, arms), arms);

        for (element = 0; element < cols; element++) {
            const float value = head_norm_unpack(head_norm_store_host(x_row[element], arms),
                                                 FT_TT_REDUCE_ARM_EXACT_CHAIN_X, arms);
            const float scale = head_norm_unpack(column, FT_TT_REDUCE_ARM_EXACT_COLUMN, arms);
            float result;

            if (!uses_gamma) {
                result = head_norm_store_kernel(value * scale, arms);
            } else if ((arms & FT_TT_REDUCE_ARM_PREMULTIPLY) != 0u) {
                const float weight =
                    head_norm_unpack(head_norm_store_host(gamma[element], arms),
                                     FT_TT_REDUCE_ARM_EXACT_GAMMA, arms);
                const float staged = head_norm_store_kernel(weight * scale, arms);

                result = head_norm_store_kernel(
                    value * head_norm_unpack(staged, FT_TT_REDUCE_ARM_EXACT_STAGE, arms), arms);
            } else {
                const float weight =
                    head_norm_unpack(head_norm_store_host(gamma[element], arms),
                                     FT_TT_REDUCE_ARM_EXACT_GAMMA, arms);
                const float staged = head_norm_store_kernel(value * weight, arms);

                result = head_norm_store_kernel(
                    head_norm_unpack(staged, FT_TT_REDUCE_ARM_EXACT_STAGE, arms) * scale, arms);
            }
            out_row[element] = result;
        }
    }
    free(mid);
}

/* ================================================================
 * Metrics
 * ================================================================ */

double ft_tt_vector_pearson(const float *a, const float *b, size_t count)
{
    double mean_a = 0.0;
    double mean_b = 0.0;
    double covariance = 0.0;
    double variance_a = 0.0;
    double variance_b = 0.0;
    size_t index;

    if (count == 0u)
        return -2.0;
    for (index = 0; index < count; index++) {
        if (!isfinite(a[index]) || !isfinite(b[index]))
            return -2.0;
        mean_a += a[index];
        mean_b += b[index];
    }
    mean_a /= (double)count;
    mean_b /= (double)count;
    for (index = 0; index < count; index++) {
        const double da = a[index] - mean_a;
        const double db = b[index] - mean_b;

        covariance += da * db;
        variance_a += da * da;
        variance_b += db * db;
    }
    if (variance_a <= 0.0 || variance_b <= 0.0)
        return (variance_a == variance_b) ? 1.0 : 0.0;
    return covariance / sqrt(variance_a * variance_b);
}

double ft_tt_vector_max_abs_difference(const float *a, const float *b, size_t count)
{
    double worst = 0.0;
    size_t index;

    for (index = 0; index < count; index++) {
        const double difference = fabs((double)a[index] - (double)b[index]);

        if (difference > worst)
            worst = difference;
    }
    return worst;
}

size_t ft_tt_vector_count_non_finite(const float *expected, const float *actual, size_t count,
                                     char *out_detail, size_t detail_bytes)
{
    size_t expected_bad = 0;
    size_t actual_bad = 0;
    size_t first = count;
    size_t index;

    for (index = 0; index < count; index++) {
        const bool bad_expected = !isfinite(expected[index]);
        const bool bad_actual = !isfinite(actual[index]);

        if (bad_expected)
            expected_bad++;
        if (bad_actual)
            actual_bad++;
        if ((bad_expected || bad_actual) && first == count)
            first = index;
    }
    if (out_detail && detail_bytes > 0u) {
        if (first == count) {
            snprintf(out_detail, detail_bytes, "all %zu finite on both sides", count);
        } else {
            snprintf(out_detail, detail_bytes,
                     "reference %zu, device %zu of %zu; first at %zu (%g vs %g)", expected_bad,
                     actual_bad, count, first, (double)expected[first], (double)actual[first]);
        }
    }
    return expected_bad + actual_bad;
}

/* ================================================================
 * Test data
 * ================================================================ */

static uint64_t random_state = 0x9E3779B97F4A7C15ull;

void ft_tt_vector_random_seed(uint64_t seed)
{
    random_state = seed ? seed : 0x9E3779B97F4A7C15ull;
}

/** xorshift64*, then the top 24 bits into a float — so every value carries mantissa bits
 *  a bf16 rendering must round, which is what distinguishes two CB descriptor sets. */
static uint32_t next_random(void)
{
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (uint32_t)((random_state * 0x2545F4914F6CDD1Dull) >> 40);
}

float ft_tt_vector_random_value(void)
{
    return ft_tt_vector_random_value_scaled(1.0f);
}

float ft_tt_vector_random_value_scaled(float scale)
{
    const uint32_t bits = next_random();

    return (((float)bits / 8388608.0f) - 1.0f) * scale;
}
