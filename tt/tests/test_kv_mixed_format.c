/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_kv_mixed_format.c
 * @brief Off-hardware gate for a MIXED KV cache, K and V in different formats (K Bfp8_b /
 *   V Bfp4_b and the like; ft_tt_op_memory.h §1.6): the K and V appends each carry their own
 *   region's format and destination strides from one bf16 source; attention places the K CB
 *   at K's tile and the V CB at V's (and the uniform `_kv` placement is the mixed one with the
 *   two equal); the encoder takes a K/V pair of different tiles and refuses a tile that is no
 *   cache format; and the four mixed programs' CB tables (attention_k8v4, _k16v4, _k4v16,
 *   _k16v8) differ from attention_bfp4_b.cbtable in exactly cb_k and cb_v, with the formats
 *   their names say.
 *
 * Build: make -f Makefile.kvmix && ./build-kvmix/test_kv_mixed_format
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_attention.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_op_memory.h"

#define CB_TABLE_DIR "../kernels/cb_tables/"
#define CB_TABLE_LINES 64
#define CB_TABLE_LINE_BYTES 160

/* Qwen3.8-27B's attention: 4 KV heads x 256, 24 Q heads (q_rows 6), ctx 30,720. */
#define Q38_KV_HEADS 4u
#define Q38_HEAD_DIM 256u
#define Q38_DIM_TILES 8u
#define Q38_Q_ROWS 6u
#define Q38_CTX 30720u
#define TEST_BLOCK_TILES 2u

static int g_checks;
static int g_failures;

#define CHECK(condition, ...)                                                                   \
    do {                                                                                        \
        g_checks++;                                                                             \
        if (!(condition)) {                                                                     \
            g_failures++;                                                                       \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                                \
            fprintf(stderr, __VA_ARGS__);                                                       \
            fputc('\n', stderr);                                                                \
        }                                                                                       \
    } while (0)

/** One mixed program: its name, the K and V CB formats and tiles its CB table must carry. */
typedef struct mixed_program {
    const char *name;
    const char *k_format;
    const char *v_format;
    uint32_t k_tile_bytes;
    uint32_t v_tile_bytes;
} mixed_program;

static const mixed_program k_mixed_programs[] = {
    { "attention_k8v4", "Bfp8_b", "Bfp4_b", FT_TT_OP_TILE_BYTES_BFP8_B,
      FT_TT_OP_TILE_BYTES_BFP4_B },
    { "attention_k16v4", "Float16_b", "Bfp4_b", FT_TT_OP_TILE_BYTES_BF16,
      FT_TT_OP_TILE_BYTES_BFP4_B },
    { "attention_k4v16", "Bfp4_b", "Float16_b", FT_TT_OP_TILE_BYTES_BFP4_B,
      FT_TT_OP_TILE_BYTES_BF16 },
    { "attention_k16v8", "Float16_b", "Bfp8_b", FT_TT_OP_TILE_BYTES_BF16,
      FT_TT_OP_TILE_BYTES_BFP8_B },
};
#define MIXED_PROGRAM_COUNT (sizeof(k_mixed_programs) / sizeof(k_mixed_programs[0]))

/** The cache format a tile size names (FT_TT_KV_FORMAT_*). */
static uint32_t kv_format_of_tile(uint32_t tile_bytes)
{
    return tile_bytes == FT_TT_OP_TILE_BYTES_BFP8_B   ? FT_TT_KV_FORMAT_BFP8
           : tile_bytes == FT_TT_OP_TILE_BYTES_BFP4_B ? FT_TT_KV_FORMAT_BFP4
                                                      : FT_TT_KV_FORMAT_BF16;
}

/** Encode one append of `token_count` rows at `position` from the bf16 strip into `layout`. */
static int encode_append(const ft_tt_kv_cache_layout *layout, uint32_t region_address,
                         uint32_t words[FT_TT_OP_MEMORY_WORDS])
{
    ft_tt_op_kv_append_request request;

    memset(&request, 0, sizeof(request));
    request.cache = layout;
    request.position = 70;
    request.token_count = 1;
    request.source_row0 = 6;
    request.source_tiles = Q38_KV_HEADS * Q38_DIM_TILES;
    request.source = (ft_tt_op_operand){ .x = 1, .y = 2, .address = 0x10000,
                                         .tile_bytes = FT_TT_OP_TILE_BYTES_BF16 };
    request.region = (ft_tt_op_operand){ .x = 3, .y = 4, .address = region_address,
                                         .tile_bytes = layout->tile_bytes };
    return ft_tt_op_kv_append_parameters(&request, words);
}

/**
 * The K and V appends of one token into a mixed cache: the same bf16 source, each descriptor
 * its own region's format flag and destination strides (the writer packs K and V to their own
 * tiles), and the two regions sized by their own tile.
 */
static void check_mixed_appends(const mixed_program *program)
{
    ft_tt_kv_cache_layout k_layout, v_layout;
    uint32_t k_words[FT_TT_OP_MEMORY_WORDS], v_words[FT_TT_OP_MEMORY_WORDS];
    int rc_k = ft_tt_kv_cache_layout_init(Q38_KV_HEADS, Q38_HEAD_DIM, Q38_CTX,
                                          program->k_tile_bytes, &k_layout);
    int rc_v = ft_tt_kv_cache_layout_init(Q38_KV_HEADS, Q38_HEAD_DIM, Q38_CTX,
                                          program->v_tile_bytes, &v_layout);

    CHECK(rc_k == FT_TT_OK && rc_v == FT_TT_OK, "%s: layouts rc %d / %d", program->name, rc_k,
          rc_v);
    if (rc_k != FT_TT_OK || rc_v != FT_TT_OK)
        return;
    CHECK(k_layout.region_bytes == 4ull * 960u * 8u * program->k_tile_bytes &&
              v_layout.region_bytes == 4ull * 960u * 8u * program->v_tile_bytes,
          "%s: region bytes K %llu V %llu", program->name,
          (unsigned long long)k_layout.region_bytes, (unsigned long long)v_layout.region_bytes);
    rc_k = encode_append(&k_layout, 0x200000, k_words);
    rc_v = encode_append(&v_layout, 0x900000, v_words);
    CHECK(rc_k == FT_TT_OK && rc_v == FT_TT_OK, "%s: append rc K %d V %d", program->name, rc_k,
          rc_v);
    if (rc_k != FT_TT_OK || rc_v != FT_TT_OK)
        return;
    CHECK(((k_words[FT_TT_OP_MEMORY_FLAGS] >> FT_TT_OP_MEMORY_KV_FORMAT_SHIFT) & 3u) ==
                  kv_format_of_tile(program->k_tile_bytes) &&
              ((v_words[FT_TT_OP_MEMORY_FLAGS] >> FT_TT_OP_MEMORY_KV_FORMAT_SHIFT) & 3u) ==
                  kv_format_of_tile(program->v_tile_bytes),
          "%s: format flags K %#x V %#x", program->name, k_words[FT_TT_OP_MEMORY_FLAGS],
          v_words[FT_TT_OP_MEMORY_FLAGS]);
    CHECK(k_words[FT_TT_OP_MEMORY_SRC_GROUP_STRIDE] == Q38_DIM_TILES * FT_TT_OP_TILE_BYTES_BF16 &&
              v_words[FT_TT_OP_MEMORY_SRC_GROUP_STRIDE] ==
                  Q38_DIM_TILES * FT_TT_OP_TILE_BYTES_BF16 &&
              k_words[FT_TT_OP_MEMORY_SRC_BASE] == v_words[FT_TT_OP_MEMORY_SRC_BASE],
          "%s: K and V must read the one bf16 strip", program->name);
    CHECK(k_words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE] == (uint32_t)k_layout.head_stride &&
              v_words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE] == (uint32_t)v_layout.head_stride,
          "%s: dst strides K %u (want %llu) V %u (want %llu)", program->name,
          k_words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE], (unsigned long long)k_layout.head_stride,
          v_words[FT_TT_OP_MEMORY_DST_GROUP_STRIDE], (unsigned long long)v_layout.head_stride);
}

static ft_tt_op_attention_residency q38_residency(void)
{
    ft_tt_op_attention_residency residency;

    memset(&residency, 0, sizeof(residency));
    residency.q_rows = Q38_Q_ROWS;
    residency.dim_tiles = Q38_DIM_TILES;
    residency.block_tiles = TEST_BLOCK_TILES;
    residency.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    return residency;
}

/**
 * The flash-decode placement: cb_k paged at K's tile, cb_v at V's, every other CB at the
 * residency's; the L1 estimate the sum of what is placed; and the uniform `_kv` placement the
 * mixed one with the two equal (so the bf16 / Bfp8 / Bfp4 programs place exactly as before).
 */
static void check_mixed_placement(const mixed_program *program)
{
    static uint8_t arena[256u * 1024u];
    const ft_tt_op_attention_residency residency = q38_residency();
    ft_tt_launch_cb_entry mixed[FT_TT_OP_ATTENTION_CB_COUNT];
    ft_tt_launch_cb_entry uniform[FT_TT_OP_ATTENTION_CB_COUNT];
    ft_tt_launch_cb_entry same[FT_TT_OP_ATTENTION_CB_COUNT];
    uint64_t mask = 0, uniform_mask = 0, same_mask = 0;
    int rc_same;
    size_t placed_bytes = 0;
    int rc = ft_tt_op_attention_place_cbs_mixed_kv(&residency, program->k_tile_bytes,
                                                   program->v_tile_bytes, mixed, arena,
                                                   sizeof(arena), &mask);

    CHECK(rc == FT_TT_OK, "%s: mixed placement rc %d", program->name, rc);
    if (rc != FT_TT_OK)
        return;
    for (unsigned i = 0; i < FT_TT_OP_ATTENTION_CB_COUNT; i++) {
        const uint32_t want = mixed[i].index == FT_TT_OP_ATTENTION_CB_K   ? program->k_tile_bytes
                              : mixed[i].index == FT_TT_OP_ATTENTION_CB_V ? program->v_tile_bytes
                                                                          : residency.tile_bytes;

        CHECK(mixed[i].page_size_bytes == want, "%s: CB %u paged %u, want %u", program->name,
              mixed[i].index, mixed[i].page_size_bytes, want);
        CHECK(mixed[i].size_bytes == mixed[i].page_size_bytes * mixed[i].page_count,
              "%s: CB %u size %u", program->name, mixed[i].index, mixed[i].size_bytes);
        placed_bytes += ((size_t)mixed[i].size_bytes + 63u) & ~(size_t)63u;
    }
    CHECK(ft_tt_op_attention_l1_bytes_mixed_kv(&residency, program->k_tile_bytes,
                                               program->v_tile_bytes) == placed_bytes,
          "%s: L1 estimate %zu vs placed %zu", program->name,
          ft_tt_op_attention_l1_bytes_mixed_kv(&residency, program->k_tile_bytes,
                                               program->v_tile_bytes),
          placed_bytes);

    rc = ft_tt_op_attention_place_cbs_kv(&residency, program->v_tile_bytes, uniform, arena,
                                         sizeof(arena), &uniform_mask);
    rc_same = ft_tt_op_attention_place_cbs_mixed_kv(&residency, program->v_tile_bytes,
                                                    program->v_tile_bytes, same, arena,
                                                    sizeof(arena), &same_mask);
    CHECK(rc == FT_TT_OK && rc_same == FT_TT_OK && uniform_mask == same_mask &&
              memcmp(uniform, same, sizeof(uniform)) == 0,
          "%s: the uniform placement must be the mixed one with K = V", program->name);
    CHECK(ft_tt_op_attention_l1_bytes_kv(&residency, program->v_tile_bytes) ==
              ft_tt_op_attention_l1_bytes_mixed_kv(&residency, program->v_tile_bytes,
                                                   program->v_tile_bytes),
          "%s: uniform L1 estimate", program->name);
}

static void fill_operand(ft_tt_op_operand *operand, uint16_t x, uint16_t y, uint32_t address,
                         uint32_t tile_bytes)
{
    operand->x = x;
    operand->y = y;
    operand->address = address;
    operand->tile_bytes = tile_bytes;
}

/** The encoder takes K and V of different cache tiles and refuses a tile no cache has. */
static void check_mixed_parameters(const mixed_program *program)
{
    ft_tt_op_attention_request request;
    uint32_t words[FT_TT_OP_ATTENTION_WORDS];
    int rc;

    memset(&request, 0, sizeof(request));
    request.q_rows = Q38_Q_ROWS;
    request.dim_tiles = Q38_DIM_TILES;
    request.block_tiles = TEST_BLOCK_TILES;
    request.kv_positions = 1000u;
    request.capacity_tiles = 960u;
    fill_operand(&request.query, 1u, 2u, 0x00010000u, FT_TT_OP_TILE_BYTES_BF16);
    fill_operand(&request.keys, 3u, 4u, 0x00020000u, program->k_tile_bytes);
    fill_operand(&request.values, 5u, 6u, 0x00030000u, program->v_tile_bytes);
    fill_operand(&request.constants, 7u, 8u, 0x00040000u, FT_TT_OP_TILE_BYTES_BF16);
    fill_operand(&request.output, 9u, 10u, 0x00050000u, FT_TT_OP_TILE_BYTES_BF16);
    rc = ft_tt_op_attention_parameters(&request, words);
    CHECK(rc == FT_TT_OK && words[FT_TT_OP_ATTENTION_K_BASE] == 0x00020000u &&
              words[FT_TT_OP_ATTENTION_V_BASE] == 0x00030000u,
          "%s: mixed K/V request rc %d", program->name, rc);
    request.values.tile_bytes = 1000u;
    CHECK(ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL,
          "%s: a V tile no cache has must be refused", program->name);
    request.values.tile_bytes = program->v_tile_bytes;
    request.keys.tile_bytes = 1000u;
    CHECK(ft_tt_op_attention_parameters(&request, words) == FT_TT_EINVAL,
          "%s: a K tile no cache has must be refused", program->name);
}

/** Read a CB table's non-comment, non-blank lines (comments stripped, spaces collapsed). */
static int read_cb_table(const char *name, char lines[CB_TABLE_LINES][CB_TABLE_LINE_BYTES])
{
    char path[256], raw[CB_TABLE_LINE_BYTES];
    FILE *file;
    int count = 0;

    snprintf(path, sizeof(path), CB_TABLE_DIR "%s.cbtable", name);
    file = fopen(path, "r");
    if (!file)
        return -1;
    while (count < CB_TABLE_LINES && fgets(raw, sizeof(raw), file)) {
        char *hash = strchr(raw, '#');
        char *out = lines[count];
        size_t used = 0;
        bool space = false;

        if (hash)
            *hash = '\0';
        for (const char *c = raw; *c; c++) {
            if (*c == ' ' || *c == '\t' || *c == '\n' || *c == '\r') {
                space = used > 0;
                continue;
            }
            if (space && used + 1 < CB_TABLE_LINE_BYTES)
                out[used++] = ' ';
            space = false;
            if (used + 1 < CB_TABLE_LINE_BYTES)
                out[used++] = *c;
        }
        out[used] = '\0';
        if (used > 0)
            count++;
    }
    fclose(file);
    return count;
}

/**
 * A mixed program's CB table is attention_bfp4_b.cbtable with cb_k and cb_v replaced by the
 * formats its name says, and nothing else changed (the program is that build plus defines).
 */
static void check_mixed_cb_table(const mixed_program *program)
{
    static char base[CB_TABLE_LINES][CB_TABLE_LINE_BYTES];
    static char mixed[CB_TABLE_LINES][CB_TABLE_LINE_BYTES];
    char want_k[64], want_v[64];
    const int base_count = read_cb_table("attention_bfp4_b", base);
    const int mixed_count = read_cb_table(program->name, mixed);
    int k_lines = 0, v_lines = 0;

    CHECK(base_count > 0 && mixed_count == base_count, "%s: %d table lines vs bfp4's %d",
          program->name, mixed_count, base_count);
    if (base_count <= 0 || mixed_count != base_count)
        return;
    snprintf(want_k, sizeof(want_k), "cb 1 %s 32 32 16 4", program->k_format);
    snprintf(want_v, sizeof(want_v), "cb 2 %s 32 32 16 4", program->v_format);
    for (int i = 0; i < base_count; i++) {
        if (strncmp(base[i], "cb 1 ", 5) == 0) {
            k_lines++;
            CHECK(strcmp(mixed[i], want_k) == 0, "%s: '%s', want '%s'", program->name, mixed[i],
                  want_k);
        } else if (strncmp(base[i], "cb 2 ", 5) == 0) {
            v_lines++;
            CHECK(strcmp(mixed[i], want_v) == 0, "%s: '%s', want '%s'", program->name, mixed[i],
                  want_v);
        } else {
            CHECK(strcmp(mixed[i], base[i]) == 0, "%s: line '%s' differs from bfp4's '%s'",
                  program->name, mixed[i], base[i]);
        }
    }
    CHECK(k_lines == 1 && v_lines == 1, "%s: cb_k %d / cb_v %d lines", program->name, k_lines,
          v_lines);
}

int main(void)
{
    printf("test_kv_mixed_format\n");
    for (size_t p = 0; p < MIXED_PROGRAM_COUNT; p++) {
        const int before = g_failures;

        check_mixed_appends(&k_mixed_programs[p]);
        check_mixed_placement(&k_mixed_programs[p]);
        check_mixed_parameters(&k_mixed_programs[p]);
        check_mixed_cb_table(&k_mixed_programs[p]);
        printf("  %-16s K %4u B / V %4u B tiles: appends, placement, parameters, CB table %s\n",
               k_mixed_programs[p].name, k_mixed_programs[p].k_tile_bytes,
               k_mixed_programs[p].v_tile_bytes, g_failures == before ? "ok" : "FAIL");
    }
    printf("%d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
