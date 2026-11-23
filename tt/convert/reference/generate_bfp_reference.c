/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file generate_bfp_reference.c
 * @brief Offline generator for ft_bfp_reference_vectors.h: four 32x32 input
 *   tiles chosen to hit every block-float edge case, packed and unpacked by
 *   foundry's ft_bfp.c, and written out ONLY after an independent reference in
 *   this file agrees with ft_bfp.c on every byte and word.
 *
 * WHY TWO IMPLEMENTATIONS. The header is the oracle test_bfp.c checks ft_bfp.c
 * against, so producing it from ft_bfp.c alone would be circular. The reference
 * below is written from the format definition (ft_bfp.h; tt-isa-documentation
 * FloatBitPatterns.md and Packers/FormatConversion.md) by a different method:
 *  - the shared exponent comes from frexp() / fpclassify(), not bit fields;
 *  - alignment is floor() of an exact ldexp() in double, and the datum
 *    rounding is nearbyint() in the default round-to-nearest-even mode, not
 *    an integer remainder comparison;
 *  - the tile walk (faces, blocks, BFP4 nibble order) is re-derived here;
 *  - a decoded datum in the normal range is checked against ldexpf() of its
 *    magnitude; out-of-range ones against the wrap rule ft_bfp.h documents.
 * Any disagreement prints the first difference and exits 1 without writing.
 *
 * Build and run with `make bfp-reference` in ../tests.
 *
 * @status Added 2026-10-08 (clean-room Phase 7): replaces the earlier generator,
 *   which spliced third-party source. At this regeneration the output was
 *   word-for-word identical to the vectors it replaced.
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_bfp.h"

#define TILE_VALUES 1024u
#define TILE_COUNT 4u
#define TOTAL_VALUES (TILE_VALUES * TILE_COUNT)
#define BLOCKS_PER_TILE 64u

/* ------------------------------------------------------------------------- */
/* Input tiles                                                               */
/* ------------------------------------------------------------------------- */

static float float_from_bits(uint32_t bits)
{
    float f;

    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint32_t bits_from_float(float f)
{
    uint32_t bits;

    memcpy(&bits, &f, sizeof(bits));
    return bits;
}

/** exp is the raw 8-bit exponent field (bias 127); frac the 23-bit fraction. */
static float make_float(uint32_t sign, uint32_t exp, uint32_t frac)
{
    return float_from_bits((sign << 31) | ((exp & 0xffu) << 23) | (frac & 0x7fffffu));
}

/** A block is one face row: row `row`, columns half*16 .. half*16+15. */
static void set_block(float *tile, uint32_t row, uint32_t half, const float *values)
{
    for (uint32_t j = 0; j < 16; j++)
        tile[row * 32u + half * 16u + j] = values[j];
}

/** Tile 0: pseudo-random normals over ~30 binades, both signs. */
static void build_random_tile(float *tile)
{
    uint32_t state = 0x5eed1234u;

    for (uint32_t i = 0; i < TILE_VALUES; i++) {
        uint32_t r;

        state = state * 1664525u + 1013904223u;
        r = state;
        tile[i] = make_float((r >> 31) & 1u, 112u + ((r >> 16) & 0x1fu), r & 0x7fffffu);
    }
}

/**
 * Tile 1: round-to-nearest-even ties. BFP8 keeps 7 bits of the 24-bit
 * significand, so the tie is a remainder of exactly 1<<16 with bit 17 as the
 * guard; BFP4 keeps 3, tie 1<<20.
 */
static void build_tie_tile(float *tile)
{
    float block[16];

    for (uint32_t b = 0; b < BLOCKS_PER_TILE; b++) {
        uint32_t exp = 120u + (b % 12u);

        for (uint32_t j = 0; j < 16; j++) {
            if (b < 16) /* BFP8 ties, no exponent spread, guard bit alternates */
                block[j] = make_float(j & 1u, exp, ((((b * 16u + j) & 0x3fu)) << 17) | 0x10000u);
            else if (b < 24) /* BFP8 ties after a 1-bit alignment shift */
                block[j] = make_float(j & 1u, exp, 0x20000u | (j & 1u));
            else if (b < 40) /* BFP4 ties, no spread */
                block[j] = make_float(j & 1u, exp, ((((b * 16u + j) & 3u)) << 21) | 0x100000u);
            else if (b < 48) /* BFP4 ties after a 1-bit shift */
                block[j] = make_float(j & 1u, exp, 0x200000u | ((j & 1u) << 1));
            else /* both tie families interleaved */
                block[j] = (j & 1u) ? make_float(1, exp, ((j & 3u) << 21) | 0x100000u)
                                    : make_float(0, exp, ((j & 7u) << 17) | 0x10000u);
        }
        if (b >= 16 && b < 48 && !(b >= 24 && b < 40))
            block[0] = make_float(0, exp + 1u, 0); /* one value a binade up sets the block exponent */
        set_block(tile, b / 2u, b % 2u, block);
    }
}

/** One value of tile 2: zeros, signed zeros, denormals, round-to-zero, saturation. */
static float zero_denormal_saturation_value(uint32_t kind, uint32_t exp, uint32_t j)
{
    switch (kind) {
    case 0: return 0.0f;
    case 1: return (j & 1u) ? -0.0f : 0.0f;
    case 2: return make_float(j & 1u, 0, 1u + j * 0x00100000u / 16u);
    case 3: return (j == 0) ? make_float(0, exp, 0) : make_float(1, 0, 0x7fffffu >> j);
    case 4: return (j == 0) ? make_float(0, exp + 20u, 0x123456u) : make_float(1, exp, 0x400000u);
    case 5: return make_float(j & 1u, exp, 0x7fffffu);
    case 6: return make_float(j & 1u, exp + (j & 1u), 0x7fffffu);
    default: return make_float(j & 1u, exp, 0x2aaaaau);
    }
}

static void build_zero_denormal_saturation_tile(float *tile)
{
    float block[16];

    for (uint32_t b = 0; b < BLOCKS_PER_TILE; b++) {
        for (uint32_t j = 0; j < 16; j++)
            block[j] = zero_denormal_saturation_value(b % 8u, 100u + (b % 40u), j);
        set_block(tile, b / 2u, b % 2u, block);
    }
}

/** One value of tile 3: exponent differences past 31, inf/NaN, the 5-bit clamp. */
static float exponent_straddle_value(uint32_t kind, uint32_t j)
{
    switch (kind) {
    case 0: return make_float(j & 1u, 150u - j, 0x400000u);
    case 1: return make_float(j & 1u, 160u - (j % 5u) * ((j % 5u) == 0 ? 0u : 30u) - (j % 5u), 0x555555u);
    case 2: return (j == 0) ? make_float(0, 200u, 0x123456u) : make_float(1, 160u, 0x654321u);
    case 3: return (j == 0) ? make_float(0, 250u, 0x000001u) : make_float(1, 150u, 0x7ffffeu);
    case 4:
        if (j == 0) return make_float(0, 255u, 0);
        if (j == 1) return make_float(1, 255u, 0);
        if (j == 2) return make_float(0, 255u, 0x400000u);
        return make_float(j & 1u, 254u - j, 0x321000u);
    case 5: return make_float(j & 1u, 144u + j, 0x0f0f0fu);
    case 6: return make_float(j & 1u, 100u + j, 0x0f0f0fu);
    default: return make_float(j & 1u, (j & 1u) ? 108u : 150u, 0x111111u);
    }
}

static void build_exponent_straddle_tile(float *tile)
{
    float block[16];

    for (uint32_t b = 0; b < BLOCKS_PER_TILE; b++) {
        for (uint32_t j = 0; j < 16; j++)
            block[j] = exponent_straddle_value(b % 8u, j);
        set_block(tile, b / 2u, b % 2u, block);
    }
}

/* ------------------------------------------------------------------------- */
/* Independent reference                                                     */
/* ------------------------------------------------------------------------- */

typedef struct {
    ft_bfp_format format;
    const char *array_name;
    uint32_t magnitude_bits;
    int five_bit_exponent;
} format_case;

static const format_case FORMAT_CASES[] = {
    {FT_BFP_BFP8_B, "ft_bfp_ref_bfp8b", 7, 0},
    {FT_BFP_BFP4_B, "ft_bfp_ref_bfp4b", 3, 0},
    {FT_BFP_BFP8_A, "ft_bfp_ref_bfp8_expa", 7, 1},
    {FT_BFP_BFP4_A, "ft_bfp_ref_bfp4_expa", 3, 1},
};

/** Biased (127) exponent of a value, 0 for zero/denormal, 255 for inf/NaN. */
static int reference_biased_exponent(float value)
{
    int e2;

    switch (fpclassify(value)) {
    case FP_ZERO:
    case FP_SUBNORMAL: return 0;
    case FP_INFINITE:
    case FP_NAN: return 255;
    default: (void)frexpf(value, &e2); return e2 + 126; /* frexp mantissa is in [0.5, 1) */
    }
}

/** The block's shared exponent in the format's own bias. */
static int reference_shared_exponent(const format_case *fc, const float *block)
{
    int largest = 0;

    for (int j = 0; j < 16; j++) {
        int e = reference_biased_exponent(block[j]);

        largest = e > largest ? e : largest;
    }
    if (!fc->five_bit_exponent)
        return largest;
    return largest <= 112 ? 0 : (largest - 112 > 31 ? 31 : largest - 112);
}

/** One datum, by exact double arithmetic and the C library's ties-to-even. */
static uint8_t reference_datum(const format_case *fc, float value, int shared)
{
    uint32_t bits = bits_from_float(value);
    uint32_t sign = bits >> 31;
    int exponent = reference_biased_exponent(value);
    int block_exponent = fc->five_bit_exponent ? shared + 112 : shared;
    double max_magnitude = ldexp(1.0, (int)fc->magnitude_bits) - 1.0;
    double significand, aligned, magnitude;

    if (exponent == 0)
        return 0;
    if (fc->five_bit_exponent && exponent < 112) {
        exponent = 112; /* raised to +-2^-15 */
        significand = ldexp(1.0, 23);
    } else if (exponent == 255) {
        significand = (double)((bits & 0x7fffffu) | 0x800000u);
    } else {
        int e2;

        significand = ldexp(frexp(fabs((double)value), &e2), 24); /* exact 24-bit integer */
    }
    if (exponent > block_exponent) {
        magnitude = max_magnitude;
    } else {
        aligned = floor(ldexp(significand, -(block_exponent - exponent)));
        magnitude = nearbyint(ldexp(aligned, -(int)(24u - fc->magnitude_bits)));
        magnitude = magnitude > max_magnitude ? max_magnitude : magnitude;
    }
    if (magnitude == 0.0)
        return 0;
    return (uint8_t)((sign << fc->magnitude_bits) | (uint32_t)magnitude);
}

/** Pack one row-major tile: 64 exponent bytes, then datums in face-row order. */
static void reference_pack_tile(const format_case *fc, const float *tile, unsigned char *out)
{
    unsigned char *datums = out + BLOCKS_PER_TILE;

    for (uint32_t b = 0; b < BLOCKS_PER_TILE; b++) {
        uint32_t face = b / 16u, face_row = b % 16u;
        const float *row = tile + ((face / 2u) * 16u + face_row) * 32u + (face % 2u) * 16u;
        int shared = reference_shared_exponent(fc, row);

        out[b] = (unsigned char)shared;
        for (uint32_t j = 0; j < 16; j++) {
            uint32_t index = b * 16u + j;
            uint8_t d = reference_datum(fc, row[j], shared);

            if (fc->magnitude_bits == 7)
                datums[index] = d;
            else if (index % 2u == 0)
                datums[index / 2u] = (unsigned char)d;
            else
                datums[index / 2u] = (unsigned char)(datums[index / 2u] | (d << 4));
        }
    }
}

/** Decode check: ldexpf for a normal result, the documented wrap rule otherwise. */
static uint32_t reference_decode(const format_case *fc, uint32_t datum, uint32_t shared)
{
    uint32_t wide = fc->magnitude_bits == 3 ? (datum & 0x0fu) << 4 : datum;
    uint32_t sign = wide >> 7, magnitude = wide & 0x7fu, leading = 0;
    int block_exponent = (int)shared + (fc->five_bit_exponent ? 112 : 0);
    int out_exponent;

    if (magnitude == 0)
        return sign << 31;
    while (!(magnitude & (0x40u >> leading)))
        leading++;
    out_exponent = block_exponent - (int)leading;
    if (out_exponent >= 1 && out_exponent <= 254) {
        float v = ldexpf((float)magnitude, block_exponent - 127 - 6);

        return bits_from_float(sign ? -v : v);
    }
    return (sign << 31) | ((uint32_t)out_exponent << 23) | (((magnitude << leading) & 0x3fu) << 17);
}

/* ------------------------------------------------------------------------- */
/* Output                                                                    */
/* ------------------------------------------------------------------------- */

static void print_words(FILE *f, const char *name, const uint32_t *values, size_t count, size_t per_line)
{
    fprintf(f, "static const uint32_t %s[%zu] = {\n", name, count);
    for (size_t i = 0; i < count; i++) {
        if (i % per_line == 0)
            fprintf(f, "   ");
        fprintf(f, " 0x%08xu,", values[i]);
        if (i % per_line == per_line - 1 || i + 1 == count)
            fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
}

static void fail(const char *what, size_t at, uint32_t ours, uint32_t reference)
{
    fprintf(stderr, "generate_bfp_reference: %s differs at %zu: ft_bfp 0x%08x, reference 0x%08x\n", what, at, ours,
            reference);
    exit(1);
}

/** Pack with ft_bfp.c and the reference; return ft_bfp's words if they agree. */
static uint32_t *pack_and_check(const format_case *fc, const float *tiles, size_t *words_out)
{
    size_t tile_bytes = ft_bfp_tile_bytes(fc->format);
    unsigned char *ours = calloc(TILE_COUNT, tile_bytes);
    unsigned char *reference = calloc(TILE_COUNT, tile_bytes);
    uint32_t *words = calloc(TILE_COUNT * tile_bytes / 4u, sizeof(uint32_t));

    if (!ours || !reference || !words || ft_bfp_pack_tiles(fc->format, tiles, TILE_COUNT, ours) != FT_BFP_OK)
        fail("pack setup", 0, 0, 0);
    for (uint32_t t = 0; t < TILE_COUNT; t++)
        reference_pack_tile(fc, tiles + t * TILE_VALUES, reference + t * tile_bytes);
    for (size_t i = 0; i < TILE_COUNT * tile_bytes; i++)
        if (ours[i] != reference[i])
            fail(fc->array_name, i, ours[i], reference[i]);
    memcpy(words, ours, TILE_COUNT * tile_bytes);
    *words_out = TILE_COUNT * tile_bytes / 4u;
    free(ours);
    free(reference);
    return words;
}

static const uint32_t UNPACK_SHARED_EXP_B[] = {0, 1, 7, 8, 64, 126, 127, 128, 200, 254, 255};
static const uint32_t UNPACK_SHARED_EXP_A[] = {0, 1, 7, 15, 16, 24, 31};

/** Decode every datum for every listed shared exponent, checked, into `out`. */
static size_t unpack_and_check(const format_case *fc, const uint32_t *exps, size_t exp_count, uint32_t *out)
{
    uint32_t datums = fc->magnitude_bits == 7 ? 256u : 16u;
    size_t n = 0;

    for (size_t e = 0; e < exp_count; e++) {
        for (uint32_t d = 0; d < datums; d++, n++) {
            out[n] = ft_bfp_convert_datum(fc->format, (uint8_t)d, exps[e]);
            if (out[n] != reference_decode(fc, d, exps[e]))
                fail("unpack", n, out[n], reference_decode(fc, d, exps[e]));
        }
    }
    return n;
}

static void write_header_top(FILE *f, const uint32_t *inputs)
{
    fprintf(f, "/* SPDX-License-Identifier: AGPL-3.0-or-later */\n"
               "/**\n"
               " * @file ft_bfp_reference_vectors.h\n"
               " * @brief GENERATED -- do not edit. Block-float reference vectors: four input\n"
               " *   tiles and their BFP8/BFP4 (exp_b and exp_a) packing and unpacking, produced\n"
               " *   by ft_bfp.c and written only after the independent reference in\n"
               " *   generate_bfp_reference.c agreed on every byte and word.\n"
               " *\n"
               " *   Regenerate with `make bfp-reference` in tt/convert/tests.\n"
               " *\n"
               " * @status None\n"
               " * @issues None\n"
               " * @todo None\n"
               " */\n\n"
               "#ifndef FT_BFP_REFERENCE_VECTORS_H\n#define FT_BFP_REFERENCE_VECTORS_H\n\n"
               "#include <stdint.h>\n\n"
               "#define FT_BFP_REF_TILE_COUNT %u\n#define FT_BFP_REF_TILE_VALUES %u\n\n",
            TILE_COUNT, TILE_VALUES);
    fprintf(f, "/* Input tiles, row-major 32x32, as raw fp32 bit patterns. */\n");
    print_words(f, "ft_bfp_ref_input_bits", inputs, TOTAL_VALUES, 8);
}

static void write_unpack_tables(FILE *f)
{
    static uint32_t table[11u * 256u];
    const char *names[] = {"ft_bfp_ref_unpack_bfp8b", "ft_bfp_ref_unpack_bfp4b", "ft_bfp_ref_unpack_bfp8_expa",
                           "ft_bfp_ref_unpack_bfp4_expa"};
    size_t exp_b_count = sizeof(UNPACK_SHARED_EXP_B) / sizeof(UNPACK_SHARED_EXP_B[0]);
    size_t exp_a_count = sizeof(UNPACK_SHARED_EXP_A) / sizeof(UNPACK_SHARED_EXP_A[0]);

    fprintf(f, "#define FT_BFP_REF_UNPACK_EXP_B_COUNT %zu\n#define FT_BFP_REF_UNPACK_EXP_A_COUNT %zu\n\n", exp_b_count,
            exp_a_count);
    print_words(f, "ft_bfp_ref_unpack_shared_exp_b", UNPACK_SHARED_EXP_B, exp_b_count, 12);
    print_words(f, "ft_bfp_ref_unpack_shared_exp_a", UNPACK_SHARED_EXP_A, exp_a_count, 12);
    fprintf(f, "/* Every datum 0..255 / 0..15 decoded, per shared exponent. */\n");
    for (int c = 0; c < 4; c++) {
        const uint32_t *exps = FORMAT_CASES[c].five_bit_exponent ? UNPACK_SHARED_EXP_A : UNPACK_SHARED_EXP_B;
        size_t count = FORMAT_CASES[c].five_bit_exponent ? exp_a_count : exp_b_count;

        print_words(f, names[c], table, unpack_and_check(&FORMAT_CASES[c], exps, count, table), 8);
    }
}

int main(int argc, char **argv)
{
    static float tiles[TOTAL_VALUES];
    static uint32_t inputs[TOTAL_VALUES];
    FILE *f;

    if (argc != 2) {
        fprintf(stderr, "usage: generate_bfp_reference <out.h>\n");
        return 2;
    }
    build_random_tile(tiles);
    build_tie_tile(tiles + TILE_VALUES);
    build_zero_denormal_saturation_tile(tiles + 2u * TILE_VALUES);
    build_exponent_straddle_tile(tiles + 3u * TILE_VALUES);
    for (uint32_t i = 0; i < TOTAL_VALUES; i++)
        inputs[i] = bits_from_float(tiles[i]);

    f = fopen(argv[1], "w");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    write_header_top(f, inputs);
    for (int c = 0; c < 4; c++) {
        size_t words;
        uint32_t *packed = pack_and_check(&FORMAT_CASES[c], tiles, &words);

        fprintf(f, "/* Packed output, %s, one tile after another. */\n", FORMAT_CASES[c].array_name);
        print_words(f, FORMAT_CASES[c].array_name, packed, words, 8);
        free(packed);
    }
    write_unpack_tables(f);
    fprintf(f, "#endif /* FT_BFP_REFERENCE_VECTORS_H */\n");
    return fclose(f) == 0 ? 0 : 1;
}
