/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file generate_kquant_reference.c
 * @brief ONE-TIME offline generator for ft_gguf_kquant_reference_vectors.h,
 *   and the cross-check that holds foundry's k-quant readers to ggml's.
 *
 * Foundry has no external dependencies, so it cannot link ggml at test time.
 * That leaves duplicated arithmetic: upstream ggml and foundry's
 * ft_gguf_dequant.c (and, when PANTHEON is defined, Pantheon's
 * src/formats/gguf_dequant.c as a third copy). This program is what stops that
 * duplication being a matter of hope:
 *
 *   1. It decodes every super-block it makes with ggml and with foundry (and
 *      Pantheon) and requires them to agree BIT-EXACTLY. ggml is linked as its
 *      own compiled shared library, so nothing here is a transcription of
 *      ggml — it IS ggml.
 *   2. It sweeps SWEEP_BLOCKS further synthetic super-blocks per type the same
 *      way, so a sub-block index error that shows on some blocks and not others
 *      is still caught.
 *   3. It freezes a subset — raw block bytes plus the fp32 words GGML produced
 *      from them — into a header the in-tree test suite checks against with no
 *      external dependency at all.
 *
 * Every block is SYNTHETIC and deterministic: one xorshift32 stream, seed
 * KQ_SEED, drives everything, in a fixed order. Per type, in this order:
 *
 *   - 2 blocks from ggml's own quantize_row_qN_K_ref over seeded weights
 *     (block 0 a smooth approximately-normal row, block 1 the same with spikes
 *     and a wide range), so the bytes are a real quantizer's output;
 *   - 2 blocks of pseudorandom bytes, which reach code points a quantizer
 *     never emits (every scale nibble, saturated quants);
 *   - 4 edge cases: all-zero; every byte 0xff with d (and dmin) at the largest
 *     finite fp16; pseudorandom codes with negative d (and dmin); pseudorandom
 *     codes with subnormal fp16 d (and dmin).
 *
 * The first four of each type are the ones the test runs as one contiguous
 * run (the stride check), so they are the four that genuinely differ.
 * Inf and NaN fp16 scales are not frozen: a NaN pins nothing (NaN != NaN).
 *
 * Build and run with generate_kquant_reference.sh.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_gguf_dequant.h"
#include "ft_gguf_map.h"

#ifdef PANTHEON
#include "gguf_dequant.h"
#endif

/* Upstream ggml, resolved from libggml-base at link time. The block pointer is
 * declared void * so this file needs none of ggml's headers. */
void dequantize_row_q2_K(const void *x, float *y, int64_t k);
void dequantize_row_q3_K(const void *x, float *y, int64_t k);
void dequantize_row_q4_K(const void *x, float *y, int64_t k);
void dequantize_row_q5_K(const void *x, float *y, int64_t k);
void dequantize_row_q6_K(const void *x, float *y, int64_t k);
void quantize_row_q2_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q3_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q4_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q5_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q6_K_ref(const float *x, void *y, int64_t k);

#define SUPERBLOCK FT_GGML_K_SUPERBLOCK
#define KQUANT_KINDS 5
#define VECTORS_PER_KIND 8
#define SWEEP_BLOCKS 4096 /* extra synthetic super-blocks per type, compared and not frozen */
#define KQ_SEED 0x5eed1234u
#define NO_OFFSET (-1)

typedef struct {
    uint32_t ggml_type;
    const char *name;
    size_t block_bytes;
    int d_offset;    /* byte offset of the fp16 super-block scale */
    int dmin_offset; /* byte offset of the fp16 super-block min, or NO_OFFSET */
    void (*ggml_decode)(const void *, float *, int64_t);
    void (*ggml_quantize)(const float *, void *, int64_t);
} kquant_kind;

static const kquant_kind KINDS[KQUANT_KINDS] = {
    {FT_GGML_TYPE_Q2_K, "Q2_K", FT_GGML_Q2_K_BLOCK_BYTES, 80, 82, dequantize_row_q2_K, quantize_row_q2_K_ref},
    {FT_GGML_TYPE_Q3_K, "Q3_K", FT_GGML_Q3_K_BLOCK_BYTES, 108, NO_OFFSET, dequantize_row_q3_K, quantize_row_q3_K_ref},
    {FT_GGML_TYPE_Q4_K, "Q4_K", FT_GGML_Q4_K_BLOCK_BYTES, 0, 2, dequantize_row_q4_K, quantize_row_q4_K_ref},
    {FT_GGML_TYPE_Q5_K, "Q5_K", FT_GGML_Q5_K_BLOCK_BYTES, 0, 2, dequantize_row_q5_K, quantize_row_q5_K_ref},
    {FT_GGML_TYPE_Q6_K, "Q6_K", FT_GGML_Q6_K_BLOCK_BYTES, 208, NO_OFFSET, dequantize_row_q6_K, quantize_row_q6_K_ref}
};

/* One frozen vector: where it came from, its bytes, and ggml's answer. */
typedef struct {
    uint32_t ggml_type;
    char origin[192];
    unsigned char bytes[FT_GGML_Q6_K_BLOCK_BYTES];
    size_t byte_count;
    uint32_t expected[SUPERBLOCK];
} reference_vector;

static reference_vector vectors[KQUANT_KINDS * VECTORS_PER_KIND];
static size_t vector_count;

static int failures;

/* Deterministic xorshift32: the single source of every synthetic byte. */
static uint32_t rng_state = KQ_SEED;

static uint32_t rng_next(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/** Uniform in [0, 1), exact in float (24 random bits). */
static float rng_unit(void)
{
    return (float)(rng_next() >> 8) * (1.0f / 16777216.0f);
}

/** Approximately normal, mean 0, unit variance: Irwin-Hall of 4 uniforms.
 *  Pure arithmetic, so the weights do not depend on the host's libm. */
static float rng_normal(void)
{
    return (rng_unit() + rng_unit() + rng_unit() + rng_unit() - 2.0f) * 1.7320508f;
}

static uint32_t bits_of(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void put_half(unsigned char *block, int offset, uint16_t half)
{
    if (offset == NO_OFFSET)
        return;
    block[offset] = (unsigned char)(half & 0xffu);
    block[offset + 1] = (unsigned char)(half >> 8);
}

/** Pantheon's reader for a type it has; *has_reader = 0 for one it does not. */
static void pantheon_decode(uint32_t ggml_type, const unsigned char *block, float *out, int *has_reader)
{
#ifdef PANTHEON
    *has_reader = 1;
    switch (ggml_type) {
    case FT_GGML_TYPE_Q3_K:
        pn_dequantize_block_q3_k(block, out);
        return;
    case FT_GGML_TYPE_Q4_K:
        pn_dequantize_block_q4_k(block, out);
        return;
    case FT_GGML_TYPE_Q5_K:
        pn_dequantize_block_q5_k(block, out);
        return;
    case FT_GGML_TYPE_Q6_K:
        pn_dequantize_block_q6_k(block, out);
        return;
    default:
        /* Q2_K: Pantheon names the type and sizes it but ships no decoder. */
        break;
    }
#else
    (void)ggml_type;
    (void)block;
    (void)out;
#endif
    *has_reader = 0;
}

/**
 * Decode one super-block two or three ways and require bit-exact agreement.
 * Returns the number of disagreements found.
 */
static int compare_one_superblock(const kquant_kind *kind, const unsigned char *block, const char *origin,
                                  float *ggml_out)
{
    float foundry_out[SUPERBLOCK];
    float pantheon_out[SUPERBLOCK];
    int has_pantheon = 0;
    int bad = 0;

    kind->ggml_decode(block, ggml_out, SUPERBLOCK);

    if (ft_gguf_dequant_to_float(kind->ggml_type, block, SUPERBLOCK, foundry_out) != 0) {
        fprintf(stderr, "FAIL %s %s: foundry refused a super-block\n", kind->name, origin);
        return 1;
    }
    pantheon_decode(kind->ggml_type, block, pantheon_out, &has_pantheon);

    for (int i = 0; i < SUPERBLOCK; i++) {
        if (bits_of(foundry_out[i]) != bits_of(ggml_out[i])) {
            if (bad < 4)
                fprintf(stderr, "FAIL %s %s [%d]: foundry %.9g (0x%08x) vs ggml %.9g (0x%08x)\n", kind->name,
                        origin, i, (double)foundry_out[i], bits_of(foundry_out[i]), (double)ggml_out[i],
                        bits_of(ggml_out[i]));
            bad++;
        }
        if (has_pantheon && bits_of(pantheon_out[i]) != bits_of(ggml_out[i])) {
            if (bad < 8)
                fprintf(stderr, "FAIL %s %s [%d]: PANTHEON %.9g vs ggml %.9g\n", kind->name, origin, i,
                        (double)pantheon_out[i], (double)ggml_out[i]);
            bad++;
        }
    }
    return bad;
}

static void take_vector(const kquant_kind *kind, const unsigned char *block, const char *origin)
{
    float ggml_out[SUPERBLOCK];
    reference_vector *slot;

    failures += compare_one_superblock(kind, block, origin, ggml_out);
    if (vector_count >= sizeof(vectors) / sizeof(vectors[0])) {
        fprintf(stderr, "FAIL: more vectors than slots\n");
        failures++;
        return;
    }
    slot = &vectors[vector_count++];
    slot->ggml_type = kind->ggml_type;
    snprintf(slot->origin, sizeof(slot->origin), "%s", origin);
    memcpy(slot->bytes, block, kind->block_bytes);
    slot->byte_count = kind->block_bytes;
    for (int i = 0; i < SUPERBLOCK; i++)
        slot->expected[i] = bits_of(ggml_out[i]);
}

/* ------------------------------------------------------------------------- */
/* Synthetic blocks                                                          */
/* ------------------------------------------------------------------------- */

/** Seeded weights for two super-blocks: [0,256) smooth, [256,512) spiky and wide. */
static void make_weights(float *weights)
{
    for (int i = 0; i < SUPERBLOCK; i++)
        weights[i] = 0.02f * rng_normal();
    for (int i = SUPERBLOCK; i < 2 * SUPERBLOCK; i++) {
        float w = 0.5f * rng_normal();

        if ((rng_next() & 31u) == 0u)
            w *= 16.0f;
        weights[i] = w;
    }
}

/**
 * Pseudorandom bytes with the fp16 super-block scales kept finite and ordinary:
 * a random fp16 is an inf or a NaN 1 time in 32, and a NaN reference vector pins
 * nothing while telling us nothing about the layout.
 */
static void make_random_block(const kquant_kind *kind, unsigned char *block)
{
    for (size_t i = 0; i < kind->block_bytes; i++)
        block[i] = (unsigned char)(rng_next() >> 17);
    block[kind->d_offset + 1] = (unsigned char)(block[kind->d_offset + 1] & 0x1Fu);
    if (kind->dmin_offset != NO_OFFSET)
        block[kind->dmin_offset + 1] = (unsigned char)(block[kind->dmin_offset + 1] & 0x1Fu);
}

static void take_quantized_vectors(const kquant_kind *kind)
{
    float weights[2 * SUPERBLOCK];
    unsigned char blocks[2 * FT_GGML_Q6_K_BLOCK_BYTES];
    char origin[192];

    make_weights(weights);
    kind->ggml_quantize(weights, blocks, 2 * SUPERBLOCK);
    for (int b = 0; b < 2; b++) {
        snprintf(origin, sizeof(origin), "synthetic: ggml quantize_row_%s_ref over seeded %s weights", kind->name,
                 b == 0 ? "smooth" : "spiky");
        take_vector(kind, blocks + (size_t)b * kind->block_bytes, origin);
    }
}

static void take_random_vectors(const kquant_kind *kind)
{
    for (int v = 0; v < 2; v++) {
        unsigned char block[FT_GGML_Q6_K_BLOCK_BYTES];
        char origin[192];

        make_random_block(kind, block);
        snprintf(origin, sizeof(origin), "pseudorandom bytes, xorshift32 seed 0x%08x, draw %d", KQ_SEED, v);
        take_vector(kind, block, origin);
    }
}

static void take_edge_vectors(const kquant_kind *kind)
{
    unsigned char block[FT_GGML_Q6_K_BLOCK_BYTES];

    memset(block, 0, sizeof(block));
    take_vector(kind, block, "edge: all-zero block");

    memset(block, 0xff, sizeof(block));
    put_half(block, kind->d_offset, 0x7bffu);
    put_half(block, kind->dmin_offset, 0x7bffu);
    take_vector(kind, block, "edge: every code byte 0xff, scales at the largest finite fp16 (65504)");

    make_random_block(kind, block);
    block[kind->d_offset + 1] = (unsigned char)(block[kind->d_offset + 1] | 0x80u);
    if (kind->dmin_offset != NO_OFFSET)
        block[kind->dmin_offset + 1] = (unsigned char)(block[kind->dmin_offset + 1] | 0x80u);
    take_vector(kind, block, "edge: pseudorandom codes, negative fp16 scales");

    make_random_block(kind, block);
    put_half(block, kind->d_offset, 0x0001u);
    put_half(block, kind->dmin_offset, 0x03ffu);
    take_vector(kind, block, "edge: pseudorandom codes, subnormal fp16 scales (0x0001 / 0x03ff)");
}

/**
 * SWEEP_BLOCKS further synthetic super-blocks per type — half quantizer output,
 * half pseudorandom bytes — compared ggml against foundry and not frozen.
 */
static void sweep_kind(const kquant_kind *kind)
{
    uint64_t bad = 0;
    float weights[2 * SUPERBLOCK];
    unsigned char blocks[2 * FT_GGML_Q6_K_BLOCK_BYTES];

    for (int s = 0; s < SWEEP_BLOCKS; s += 2) {
        float ggml_out[SUPERBLOCK];

        if ((s / 2) % 2 == 0) {
            make_weights(weights);
            kind->ggml_quantize(weights, blocks, 2 * SUPERBLOCK);
        } else {
            make_random_block(kind, blocks);
            make_random_block(kind, blocks + kind->block_bytes);
        }
        for (int b = 0; b < 2; b++)
            bad += (uint64_t)compare_one_superblock(kind, blocks + (size_t)b * kind->block_bytes, "sweep", ggml_out);
    }
    printf("sweep %s: %d synthetic super-blocks, %llu mismatched words\n", kind->name, SWEEP_BLOCKS,
           (unsigned long long)bad);
    if (bad)
        failures++;
}

/* ------------------------------------------------------------------------- */
/* Header emission                                                           */
/* ------------------------------------------------------------------------- */

static void emit_preamble(FILE *f)
{
    fprintf(f, "/* SPDX-License-Identifier: AGPL-3.0-or-later */\n");
    fprintf(f, "/* Copyright (C) 2026 David (Corey) Herrera */\n");
    fprintf(f, "/**\n");
    fprintf(f, " * @file ft_gguf_kquant_reference_vectors.h\n");
    fprintf(f, " * @brief GENERATED -- do not edit. K-quant super-blocks and the fp32 words\n");
    fprintf(f, " *   UPSTREAM GGML produced from them.\n");
    fprintf(f, " *\n");
    fprintf(f, " *   Every block is SYNTHETIC: generated from one xorshift32 stream, seed\n");
    fprintf(f, " *   0x%08x -- ggml's own quantize_row_qN_K_ref over seeded weights,\n", KQ_SEED);
    fprintf(f, " *   pseudorandom bytes, and edge cases (all-zero, saturated codes at the\n");
    fprintf(f, " *   largest finite fp16 scale, negative scales, subnormal scales). No block\n");
    fprintf(f, " *   comes from a model checkpoint.\n");
    fprintf(f, " *\n");
    fprintf(f, " *   Every `expected` word below came out of ggml's own compiled\n");
    fprintf(f, " *   dequantize_row_qN_K (libggml-base), not out of foundry and not out of a\n");
    fprintf(f, " *   transcription of ggml. test_convert.c requires ft_gguf_dequant.c to\n");
    fprintf(f, " *   reproduce all of them BIT-EXACTLY.\n");
    fprintf(f, " *\n");
    fprintf(f, " *   Regenerate, from tt/convert:\n");
    fprintf(f, " *     reference/generate_kquant_reference.sh <llama.cpp checkout>\n");
    fprintf(f, " *   (the checkout must have build/bin/libggml-base). Committed so the test\n");
    fprintf(f, " *   needs no ggml.\n");
    fprintf(f, " *\n");
    fprintf(f, " * @status None\n");
    fprintf(f, " * @issues None\n");
    fprintf(f, " * @todo None\n");
    fprintf(f, " */\n\n");
    fprintf(f, "#ifndef FT_GGUF_KQUANT_REFERENCE_VECTORS_H\n");
    fprintf(f, "#define FT_GGUF_KQUANT_REFERENCE_VECTORS_H\n\n");
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "#define FT_KQUANT_REF_COUNT %zu\n", vector_count);
    fprintf(f, "#define FT_KQUANT_REF_VALUES 256\n");
    fprintf(f, "#define FT_KQUANT_REF_MAX_BYTES %d\n\n", FT_GGML_Q6_K_BLOCK_BYTES);
    fprintf(f, "typedef struct {\n");
    fprintf(f, "    uint32_t ggml_type;\n");
    fprintf(f, "    const char *origin;\n");
    fprintf(f, "    uint32_t byte_count;\n");
    fprintf(f, "    const unsigned char *bytes;\n");
    fprintf(f, "    const uint32_t *expected_bits;\n");
    fprintf(f, "} ft_kquant_reference_vector;\n\n");
}

static void emit_header(const char *out_path)
{
    FILE *f = fopen(out_path, "w");

    if (!f) {
        fprintf(stderr, "cannot write %s\n", out_path);
        failures++;
        return;
    }
    emit_preamble(f);

    for (size_t v = 0; v < vector_count; v++) {
        fprintf(f, "static const unsigned char ft_kquant_ref_bytes_%zu[%zu] = {\n", v, vectors[v].byte_count);
        for (size_t i = 0; i < vectors[v].byte_count; i++)
            fprintf(f, "%s0x%02xu%s", i % 12 == 0 ? "    " : " ", vectors[v].bytes[i],
                    i + 1 == vectors[v].byte_count ? "\n" : (i % 12 == 11 ? ",\n" : ","));
        fprintf(f, "};\n\n");

        fprintf(f, "static const uint32_t ft_kquant_ref_expected_%zu[256] = {\n", v);
        for (int i = 0; i < SUPERBLOCK; i++)
            fprintf(f, "%s0x%08xu%s", i % 8 == 0 ? "    " : " ", vectors[v].expected[i],
                    i + 1 == SUPERBLOCK ? "\n" : (i % 8 == 7 ? ",\n" : ","));
        fprintf(f, "};\n\n");
    }

    fprintf(f, "static const ft_kquant_reference_vector ft_kquant_reference_vectors[FT_KQUANT_REF_COUNT] = {\n");
    for (size_t v = 0; v < vector_count; v++)
        fprintf(f, "    {%u, \"%s\", %u, ft_kquant_ref_bytes_%zu, ft_kquant_ref_expected_%zu},\n",
                vectors[v].ggml_type, vectors[v].origin, (unsigned)vectors[v].byte_count, v, v);
    fprintf(f, "};\n\n");
    fprintf(f, "#endif /* FT_GGUF_KQUANT_REFERENCE_VECTORS_H */\n");

    fclose(f);
    printf("wrote %s: %zu vectors\n", out_path, vector_count);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out.h>\n", argv[0]);
        return 2;
    }

    for (size_t k = 0; k < KQUANT_KINDS; k++) {
        take_quantized_vectors(&KINDS[k]);
        take_random_vectors(&KINDS[k]);
        take_edge_vectors(&KINDS[k]);
    }

    printf("frozen vectors:\n");
    for (size_t v = 0; v < vector_count; v++) {
        const char *name = "?";

        for (size_t k = 0; k < KQUANT_KINDS; k++) {
            if (KINDS[k].ggml_type == vectors[v].ggml_type)
                name = KINDS[k].name;
        }
        printf("  %-5s %s\n", name, vectors[v].origin);
    }
    printf("\n");

    for (size_t k = 0; k < KQUANT_KINDS; k++)
        sweep_kind(&KINDS[k]);

#ifdef PANTHEON
    printf("Pantheon's reader cross-checked (Q3_K..Q6_K)\n");
#else
    printf("Pantheon's reader not cross-checked (PANTHEON_DIR unset)\n");
#endif

    if (failures == 0)
        emit_header(argv[1]);
    printf("\n%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
