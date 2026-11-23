/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_ppl_arm_gguf.c
 * @brief Build one arm of a perplexity A/B for a precision decision, by
 *   perturbing named tensors of a GGUF **in place** through a block-float
 *   round trip.
 *
 * WHY THIS EXISTS. Spec §6.4 assigns a precision to every tensor role, and the
 * quality evidence for those assignments is per-tensor PCC. PCC is the wrong
 * instrument for some of them: a correlation of 0.99997 says nothing about a
 * sampling distribution, which is what the embedding and the LM head actually
 * feed. Those decisions have to be settled on **perplexity**, and perplexity
 * needs an engine — llama.cpp, which cannot read BFP8. So the perturbation is
 * carried in a container llama.cpp can read:
 *
 *     Q8_0( BFP8_b( dequant_Q8_0(w) ) )
 *
 * written back into the same Q8_0 blocks, so the file layout is byte-identical
 * and only the values move. The result is a GGUF that runs anywhere the
 * original does. This is how `docs/tt_bringup_log.md` §13.5 demoted
 * `token_embd` and `output` from bf16 to BFP8 (+0.035% perplexity, -358.6 MiB).
 *
 * THE ARM IS CONSERVATIVE, DELIBERATELY. It carries a requantization the real
 * artifact does not — the trip back into Q8_0 — which can only ADD error. So
 * `ppl(test) == ppl(control)` is a sound GO, while a gap needs the `--identity`
 * control below to attribute. On MiniCPM5-1B that control came out at exactly
 * zero: identical printed perplexity at all 1067 chunks.
 *
 * RUN THREE ARMS, NOT TWO. A small perplexity delta looks the same as a
 * measurement that is not connected to anything, and this project has produced
 * six green-but-vacuous gates. So:
 *
 *   control     the untouched checkpoint
 *   --identity  requantize only, no block-float step. Isolates the method
 *   (default)   the format under test
 *   --bfp4      the same tensors at BFP4, ~16x the weight error. If this does
 *               not move perplexity far more, the harness is not measuring the
 *               tensors you think it is
 *
 * BLOCK ALIGNMENT. BFP8_b groups 16 consecutive elements along the fastest
 * axis. In a 32x32 tile a face row spans 16 consecutive columns of one matrix
 * row, and the block-float path is fed row-major-within-tile (§12.3 M12), so a
 * block never straddles a matrix row or a tile edge as long as the column count
 * divides by 32 — which is checked. That is why the round trip can be done flat
 * here and still be exactly what the converter produces.
 *
 * Usage — ALWAYS on a copy, it rewrites in place:
 *   cp model.gguf arm.gguf
 *   ./build/ft_tt_ppl_arm_gguf arm.gguf [--identity | --bfp4] [tensor ...]
 *   llama-perplexity -m arm.gguf -f corpus.txt -c 512
 *
 * Defaults to `token_embd.weight` and `output.weight` when no tensor is named.
 * Only Q8_0 tensors can be perturbed, since the container has to survive.
 *
 * @status Used for the §13.5 embedding/LM-head decision.
 * @issues None
 * @todo None
 */

/* pread/pwrite are POSIX.1-2001, not C99, so `-std=c99 -pedantic` hides them and
 * glibc compiles this to an implicit declaration -> -Werror. macOS declares them
 * unconditionally, which is why this built for as long as the tool only ever ran
 * on the Mac — the same shape of gap as MADV_DONTNEED in ft_gguf_map.c. */
#define _DEFAULT_SOURCE

#include "ft_bfp.h"
#include "ft_gguf_dequant.h"
#include "ft_gguf_map.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define Q8_0_BLOCK 32
#define Q8_0_BLOCK_BYTES 34

static uint32_t bits_of(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float float_of(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/** fp32 -> fp16 bits, round to nearest even, no denormal/overflow subtleties
 *  needed: Q8_0 scales here are ordinary normal magnitudes. */
static uint16_t fp16_from_float(float value)
{
    uint32_t bits = bits_of(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;
    uint32_t half;
    uint32_t round_bits;

    if (exponent <= 0)
        return (uint16_t)sign;
    if (exponent >= 31)
        return (uint16_t)(sign | 0x7C00u);

    half = sign | ((uint32_t)exponent << 10) | (mantissa >> 13);
    round_bits = mantissa & 0x1FFFu;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (half & 1u)))
        half++;
    return (uint16_t)half;
}

static float float_from_fp16(uint16_t half)
{
    uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1Fu;
    uint32_t mantissa = half & 0x3FFu;

    if (exponent == 0) {
        if (mantissa == 0)
            return float_of(sign);
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        exponent++;
        mantissa &= 0x3FFu;
    } else if (exponent == 31) {
        return float_of(sign | 0x7F800000u | (mantissa << 13));
    }
    return float_of(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

/** Which block-float format the arm applies. Set from the command line so the
 *  same tool can build the sensitivity control (BFP4, ~10x the weight error). */
static ft_bfp_format arm_format = FT_BFP_BFP8_B;

/** One block-float round trip over one 16-value block, in place. */
static void block_float_round_trip(float *values)
{
    uint32_t input_bits[FT_BFP_BLOCK_VALUES];
    uint8_t shared_exp;
    size_t i;

    for (i = 0; i < FT_BFP_BLOCK_VALUES; i++)
        input_bits[i] = bits_of(values[i]);

    shared_exp = ft_bfp_block_shared_exponent(arm_format, input_bits);
    for (i = 0; i < FT_BFP_BLOCK_VALUES; i++) {
        uint8_t datum = ft_bfp_convert_float_bits(arm_format, input_bits[i], shared_exp);
        values[i] = float_of(ft_bfp_convert_datum(arm_format, datum, shared_exp));
    }
}

/**
 * Quantize 32 floats into one 34-byte Q8_0 block: an fp16 scale (largest
 * magnitude / 127, rounded to half) and 32 int8 quants rounded with lrintf
 * against the scale as stored, so the block decodes the way it was encoded.
 */
static void requantize_q8_0_block(const float *values, unsigned char *out)
{
    float largest = 0.0f;
    float scale;
    float inverse;
    uint16_t half;
    int i;

    for (i = 0; i < Q8_0_BLOCK; i++) {
        float magnitude = fabsf(values[i]);
        if (magnitude > largest)
            largest = magnitude;
    }
    scale = largest / 127.0f;
    half = fp16_from_float(scale);
    memcpy(out, &half, sizeof(half));
    scale = float_from_fp16(half);
    inverse = scale != 0.0f ? 1.0f / scale : 0.0f;

    for (i = 0; i < Q8_0_BLOCK; i++) {
        float x = values[i] * inverse;
        int q = (int)lrintf(x);
        if (q > 127)
            q = 127;
        if (q < -128)
            q = -128;
        out[2 + i] = (unsigned char)(signed char)q;
    }
}

typedef struct {
    double sum_squared_error;
    double sum_squared_source;
    uint64_t count;
} error_accumulator;

static void accumulate(error_accumulator *accumulator, const float *source, const float *got,
                       size_t count)
{
    size_t i;
    for (i = 0; i < count; i++) {
        double difference = (double)got[i] - (double)source[i];
        accumulator->sum_squared_error += difference * difference;
        accumulator->sum_squared_source += (double)source[i] * (double)source[i];
    }
    accumulator->count += count;
}

static double relative_rms(const error_accumulator *accumulator)
{
    if (accumulator->sum_squared_source <= 0.0)
        return 0.0;
    return sqrt(accumulator->sum_squared_error / accumulator->sum_squared_source);
}

/**
 * Rewrite one tensor's Q8_0 payload in `fd` as Q8_0(BFP8(dequant(Q8_0))).
 *
 * @param identity  when non-zero, skip the BFP8 step entirely. The output must
 *                  then be byte-identical to the input, which is the control
 *                  that proves the requantization path is not itself the thing
 *                  moving the perplexity.
 */
static int perturb_tensor_in_place(int fd, const ft_gguf_map *map, const ft_gguf_tensor *tensor,
                          int identity, error_accumulator *bfp_only, error_accumulator *both,
                          uint64_t *bytes_changed)
{
    uint64_t file_offset = ft_gguf_map_tensor_file_offset(map, tensor);
    uint64_t columns = tensor->dims[0];
    uint64_t rows = tensor->dims[1];
    uint64_t blocks_per_row = columns / Q8_0_BLOCK;
    float *source = malloc(columns * sizeof(float));
    float *perturbed = malloc(columns * sizeof(float));
    float *reread = malloc(columns * sizeof(float));
    unsigned char *row_bytes = malloc(blocks_per_row * Q8_0_BLOCK_BYTES);
    uint64_t row;
    int rc = -1;

    if (columns % Q8_0_BLOCK != 0 || columns % FT_BFP_BLOCK_VALUES != 0) {
        fprintf(stderr, "column count %llu is not a whole number of blocks\n",
                (unsigned long long)columns);
        goto done;
    }
    if (!source || !perturbed || !reread || !row_bytes)
        goto done;

    for (row = 0; row < rows; row++) {
        uint64_t row_offset = file_offset + row * blocks_per_row * Q8_0_BLOCK_BYTES;
        uint64_t block;
        size_t i;

        if (pread(fd, row_bytes, blocks_per_row * Q8_0_BLOCK_BYTES, (off_t)row_offset) !=
            (ssize_t)(blocks_per_row * Q8_0_BLOCK_BYTES)) {
            perror("pread");
            goto done;
        }
        if (ft_gguf_dequant_to_float(FT_GGML_TYPE_Q8_0, row_bytes, columns, source) != 0)
            goto done;

        memcpy(perturbed, source, columns * sizeof(float));
        if (!identity) {
            for (i = 0; i + FT_BFP_BLOCK_VALUES <= columns; i += FT_BFP_BLOCK_VALUES)
                block_float_round_trip(perturbed + i);
        }
        accumulate(bfp_only, source, perturbed, (size_t)columns);

        for (block = 0; block < blocks_per_row; block++)
            requantize_q8_0_block(perturbed + block * Q8_0_BLOCK,
                                row_bytes + block * Q8_0_BLOCK_BYTES);

        if (ft_gguf_dequant_to_float(FT_GGML_TYPE_Q8_0, row_bytes, columns, reread) != 0)
            goto done;
        accumulate(both, source, reread, (size_t)columns);

        if (pwrite(fd, row_bytes, blocks_per_row * Q8_0_BLOCK_BYTES, (off_t)row_offset) !=
            (ssize_t)(blocks_per_row * Q8_0_BLOCK_BYTES)) {
            perror("pwrite");
            goto done;
        }
        *bytes_changed += blocks_per_row * Q8_0_BLOCK_BYTES;
    }
    rc = 0;

done:
    free(source);
    free(perturbed);
    free(reread);
    free(row_bytes);
    return rc;
}

int main(int argc, char **argv)
{
    static const char *DEFAULT_TENSORS[] = {"token_embd.weight", "output.weight"};
    ft_gguf_map *map = NULL;
    error_accumulator bfp_only = {0.0, 0.0, 0};
    error_accumulator both = {0.0, 0.0, 0};
    uint64_t bytes_changed = 0;
    int identity = 0;
    int fd;
    size_t t;

    const char **names = DEFAULT_TENSORS;
    size_t name_count = sizeof(DEFAULT_TENSORS) / sizeof(DEFAULT_TENSORS[0]);
    int first_name = 2;

    if (argc < 2) {
        fprintf(stderr, "usage: %s copy.gguf [--identity|--bfp4] [tensor ...]\n", argv[0]);
        fprintf(stderr, "  rewrites copy.gguf IN PLACE; give it a copy, never the original\n");
        return 2;
    }
    while (first_name < argc && argv[first_name][0] == '-') {
        if (strcmp(argv[first_name], "--identity") == 0)
            identity = 1;
        else if (strcmp(argv[first_name], "--bfp4") == 0)
            arm_format = FT_BFP_BFP4_B;
        else {
            fprintf(stderr, "unknown option '%s'\n", argv[first_name]);
            return 2;
        }
        first_name++;
    }
    if (first_name < argc) {
        names = (const char **)&argv[first_name];
        name_count = (size_t)(argc - first_name);
    }

    if (ft_gguf_map_open(argv[1], &map) != FT_GGUF_OK)
        return 1;

    for (t = 0; t < name_count; t++) {
        const ft_gguf_tensor *tensor = ft_gguf_map_find(map, names[t]);
        /* A tensor's offset is into the PART that holds it, which on a split
         * checkpoint is not the file named on the command line. Writing to
         * argv[1] at that offset would corrupt part 1 and leave the tensor
         * untouched — a perturbation experiment that perturbs the wrong thing
         * and still prints a number. */
        const char *part_path;

        if (tensor == NULL) {
            fprintf(stderr, "no tensor '%s'\n", names[t]);
            ft_gguf_map_close(map);
            return 1;
        }
        if (tensor->ggml_type != FT_GGML_TYPE_Q8_0) {
            fprintf(stderr, "'%s' is %s, expected Q8_0\n", names[t],
                    ft_gguf_type_name(tensor->ggml_type));
            ft_gguf_map_close(map);
            return 1;
        }
        part_path = ft_gguf_map_part_path(map, ft_gguf_map_tensor_part(map, tensor));
        if (part_path == NULL) {
            fprintf(stderr, "'%s' has no part file\n", names[t]);
            ft_gguf_map_close(map);
            return 1;
        }
        fd = open(part_path, O_RDWR);
        if (fd < 0) {
            perror("open for write");
            ft_gguf_map_close(map);
            return 1;
        }
        printf("%-20s %s offset %llu, %llu x %llu\n", names[t], part_path,
               (unsigned long long)ft_gguf_map_tensor_file_offset(map, tensor),
               (unsigned long long)tensor->dims[1], (unsigned long long)tensor->dims[0]);
        if (perturb_tensor_in_place(fd, map, tensor, identity, &bfp_only, &both, &bytes_changed) != 0) {
            close(fd);
            ft_gguf_map_close(map);
            return 1;
        }
        close(fd);
    }

    printf("\nmode: %s\n", identity ? "IDENTITY (requantization only)"
                              : (arm_format == FT_BFP_BFP4_B ? "BFP4_b then requantize"
                                                            : "BFP8_b then requantize"));
    printf("values touched          %llu\n", (unsigned long long)both.count);
    printf("bytes rewritten         %llu\n", (unsigned long long)bytes_changed);
    printf("relative RMS, BFP8 only %.8f\n", relative_rms(&bfp_only));
    printf("relative RMS, BFP8+Q8_0 %.8f\n", relative_rms(&both));
    if (relative_rms(&bfp_only) > 0.0)
        printf("contamination factor    %.3fx\n", relative_rms(&both) / relative_rms(&bfp_only));

    close(fd);
    ft_gguf_map_close(map);
    return 0;
}
