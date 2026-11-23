/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tile_row0.c
 * @brief Off-hardware acceptance for ft_tt_tile_row0.{h,c} — the row-0-only transfers
 *   that cut a matvec's readback 16x and its upload 32x.
 *
 * Needs no card. A simulated DRAM bank stands in for the NoC accessor and ENFORCES the
 * real alignment rules (64 B card->host, 16 B host->card), so a mutation that asks for a
 * 32-byte read is refused here exactly as Blackhole refuses it rather than passing
 * quietly.
 *
 * The gate is not "row 0 looks right". It is BYTE-IDENTICAL against the full-tile read
 * plus ft_tt_untilize_strip() — the path this replaces — on every shape the target model
 * uses, widest first (lm_head, Nt=4080, 130,560 outputs).
 *
 * Two mutations must be caught, and `run_tile_row0_red.sh` applies them to the SOURCE and
 * rebuilds whole-program:
 *   - the face stride as 256 instead of 512 (the element-index stride mistaken for the
 *     byte one — it reads the middle of face 0 and yields a plausible wrong answer);
 *   - the two chunks swapped (columns 16-31 written where 0-15 belong).
 *
 * It also carries the STRUCTURAL half of the two-accessor split: a register address and a
 * 4-byte length must both be refused by the DRAM transfer check, which is what makes it
 * impossible for run control to take the bulk path by accident.
 *
 * @status Written for the Phase 7 readback fixes.
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_noc.h"
#include "ft_tt_tile_row0.h"
#include "ft_tt_tilize.h"

static int g_failures;
static int g_checks;

static void check(int condition, const char *what)
{
    g_checks++;
    if (!condition) {
        g_failures++;
        printf("  FAIL: %s\n", what);
    }
}

/* ================================================================
 * A simulated DRAM bank behind an ft_tt_noc_access
 * ================================================================ */

typedef struct sim_bank {
    unsigned char *bytes;
    size_t size;
    uint64_t read_calls, write_calls;
    uint64_t read_bytes, write_bytes;
    int refusals;
} sim_bank;

/** Card->host: 64 B alignment on both address and length. A 32 B read is REFUSED on
 *  Blackhole, and that refusal is the whole reason the recoverable factor is 16 and not
 *  32 — so the simulation must refuse it too or the test proves nothing. */
static int sim_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst, size_t bytes)
{
    sim_bank *bank = (sim_bank *)user;

    (void)x;
    (void)y;
    if ((addr % 64u) != 0u || (bytes % 64u) != 0u) {
        bank->refusals++;
        return FT_TT_EINVAL;
    }
    if (addr + bytes > bank->size) {
        bank->refusals++;
        return FT_TT_EINVAL;
    }
    memcpy(dst, bank->bytes + addr, bytes);
    bank->read_calls++;
    bank->read_bytes += bytes;
    return FT_TT_OK;
}

/** Host->card: the looser 16 B DRAM write alignment. */
static int sim_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                     size_t bytes)
{
    sim_bank *bank = (sim_bank *)user;

    (void)x;
    (void)y;
    if ((addr % 16u) != 0u || (bytes % 16u) != 0u) {
        bank->refusals++;
        return FT_TT_EINVAL;
    }
    if (addr + bytes > bank->size) {
        bank->refusals++;
        return FT_TT_EINVAL;
    }
    memcpy(bank->bytes + addr, src, bytes);
    bank->write_calls++;
    bank->write_bytes += bytes;
    return FT_TT_OK;
}

static void sim_bank_reset(sim_bank *bank)
{
    bank->read_calls = bank->write_calls = 0;
    bank->read_bytes = bank->write_bytes = 0;
    bank->refusals = 0;
}

/* ================================================================
 * Shapes and data
 * ================================================================ */

/** Every distinct out_features MiniCPM5-1B asks a matvec for, widest first. */
static const uint32_t k_shapes[] = { 130560u, 4608u, 2048u, 1536u, 256u, 32u, 33u };

/** A deterministic, non-symmetric pattern. Symmetric data would let a swapped-chunk
 *  mutation pass, which is exactly the mutation this file exists to catch. */
static uint16_t pattern_at(uint32_t index)
{
    return (uint16_t)(0x3C00u + ((index * 7919u) & 0x03FFu) + ((index & 0x1Fu) << 5));
}

/* ================================================================
 * The round trip
 * ================================================================ */

/** Read row 0 two ways on one shape and require the bytes to be identical. */
static void check_read_matches_full_tile(sim_bank *bank, const ft_tt_noc_access *noc,
                                         uint32_t cols)
{
    uint32_t tile_count = ft_tt_tile_count_for(cols);
    size_t strip_bytes = (size_t)tile_count * FT_TT_TILE_ROW0_TILE_BYTES;
    uint16_t *source = malloc((size_t)cols * sizeof(*source));
    uint16_t *tiles = malloc(strip_bytes);
    uint16_t *full_read = malloc(strip_bytes);
    uint16_t *row_reference = calloc((size_t)tile_count * 32u, sizeof(*row_reference));
    uint16_t *row_fast = calloc((size_t)tile_count * 32u, sizeof(*row_fast));
    char label[128];
    uint32_t i;

    if (!source || !tiles || !full_read || !row_reference || !row_fast) {
        printf("  FAIL: allocation for cols=%u\n", cols);
        g_failures++;
        goto done;
    }
    for (i = 0; i < cols; i++)
        source[i] = pattern_at(i);

    /* Lay the strip out exactly as a matvec's writer leaves it: one valid row, 31 zero. */
    if (ft_tt_tilize_strip(source, sizeof(uint16_t), 1u, cols, (size_t)cols * sizeof(uint16_t),
                           tiles) != FT_TT_TILIZE_OK) {
        printf("  FAIL: tilize cols=%u\n", cols);
        g_failures++;
        goto done;
    }
    memcpy(bank->bytes, tiles, strip_bytes);

    /* Arm 1: what the code did before — the whole strip, then untilize. */
    sim_bank_reset(bank);
    check(noc->read(noc->user, 0, 0, 0, full_read, strip_bytes) == FT_TT_OK,
          "full-tile read accepted");
    {
        uint64_t full_bytes = bank->read_bytes;

        check(ft_tt_untilize_strip(full_read, sizeof(uint16_t), 1u, cols,
                                   (size_t)cols * sizeof(uint16_t),
                                   row_reference) == FT_TT_TILIZE_OK, "untilize");

        /* Arm 2: row 0 only. */
        sim_bank_reset(bank);
        check(ft_tt_tile_row0_read(noc, 0, 0, 0, tile_count, row_fast) == FT_TT_OK,
              "row-0 read accepted");
        check(bank->refusals == 0, "row-0 read is never refused by the 64 B alignment");

        snprintf(label, sizeof(label), "cols=%u row 0 is byte-identical to the full-tile path",
                 cols);
        check(memcmp(row_reference, row_fast, (size_t)cols * sizeof(uint16_t)) == 0, label);

        snprintf(label, sizeof(label), "cols=%u moves 128 B/tile against %u", cols,
                 FT_TT_TILE_ROW0_TILE_BYTES);
        check(bank->read_bytes == (uint64_t)tile_count * 128u, label);

        snprintf(label, sizeof(label), "cols=%u read reduction is exactly 16.00x", cols);
        check(full_bytes == bank->read_bytes * 16u, label);
    }

done:
    free(source);
    free(tiles);
    free(full_read);
    free(row_reference);
    free(row_fast);
}

/** Write row 0 into a zeroed strip and require the strip to equal a full tilize. */
static void check_write_matches_full_tile(sim_bank *bank, const ft_tt_noc_access *noc,
                                          uint32_t cols)
{
    uint32_t tile_count = ft_tt_tile_count_for(cols);
    size_t strip_bytes = (size_t)tile_count * FT_TT_TILE_ROW0_TILE_BYTES;
    uint16_t *source = calloc((size_t)tile_count * 32u, sizeof(*source));
    uint16_t *expected = malloc(strip_bytes);
    char label[128];
    uint32_t i;

    if (!source || !expected) {
        printf("  FAIL: allocation for cols=%u\n", cols);
        g_failures++;
        goto done;
    }
    for (i = 0; i < cols; i++)
        source[i] = pattern_at(i + 13u);

    if (ft_tt_tilize_strip(source, sizeof(uint16_t), 1u, tile_count * 32u,
                           (size_t)tile_count * 32u * sizeof(uint16_t),
                           expected) != FT_TT_TILIZE_OK) {
        printf("  FAIL: tilize cols=%u\n", cols);
        g_failures++;
        goto done;
    }

    /* Rows 1-31 staged zero ONCE, as the header requires; then row 0 only. */
    memset(bank->bytes, 0, strip_bytes);
    sim_bank_reset(bank);
    check(ft_tt_tile_row0_write(noc, 0, 0, 0, tile_count, source) == FT_TT_OK,
          "row-0 write accepted");
    check(bank->refusals == 0, "row-0 write is never refused by the 16 B alignment");

    snprintf(label, sizeof(label), "cols=%u staged strip is byte-identical to a full tilize",
             cols);
    check(memcmp(bank->bytes, expected, strip_bytes) == 0, label);

    snprintf(label, sizeof(label), "cols=%u write reduction is exactly 32.00x", cols);
    check(bank->write_bytes * 32u == (uint64_t)strip_bytes, label);

done:
    free(source);
    free(expected);
}

/* ================================================================
 * The offset function, directly
 * ================================================================ */

static void check_chunk_offsets(void)
{
    check(ft_tt_tile_row0_chunk_offset(0u, 0u) == 0u, "tile 0 chunk 0 is at byte 0");
    check(ft_tt_tile_row0_chunk_offset(0u, 1u) == 512u,
          "tile 0 chunk 1 is at byte 512 — one WHOLE face, not 256");
    check(ft_tt_tile_row0_chunk_offset(1u, 0u) == 2048u, "tile 1 chunk 0 is one tile on");
    check(ft_tt_tile_row0_chunk_offset(1u, 1u) == 2560u, "tile 1 chunk 1 is 2048 + 512");
    check(ft_tt_tile_row0_chunk_offset(4079u, 1u) == 4079u * 2048u + 512u,
          "the last lm_head tile's chunk 1");
    /* Every offset the reader issues must satisfy the 64 B DRAM read alignment, or the
     * card refuses it and the fix does not exist. */
    check((ft_tt_tile_row0_chunk_offset(0u, 1u) % 64u) == 0u, "chunk 1 is 64 B aligned");
    check((ft_tt_tile_row0_chunk_offset(7u, 1u) % 64u) == 0u, "tile 7 chunk 1 is 64 B aligned");
}

/* ================================================================
 * The two-accessor split, structurally
 * ================================================================ */

/**
 * The bulk DRAM path must be INCAPABLE of carrying run control.
 *
 * This is the failure mode of the second window that would not surface until something
 * hung: a register access silently taking the bulk path. It cannot, and these are the
 * reasons — the soft-reset register is outside the DRAM space and a go message is under
 * the 16 B unit. Both are checked against the real ft_tt_noc_check_transfer(), not a
 * re-derivation of its rules.
 */
static void check_registers_cannot_take_the_bulk_path(void)
{
    const uint64_t soft_reset = 0xFFB121B0ull;

    check(ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, false, soft_reset, 4u) != FT_TT_OK,
          "the soft-reset register is refused by the DRAM write check");
    check(ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, soft_reset, 4u) != FT_TT_OK,
          "the soft-reset register is refused by the DRAM read check");
    check(ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, false, 0u, 4u) != FT_TT_OK,
          "a 4-byte go message is refused by the DRAM write check");
    check(ft_tt_noc_check_transfer(FT_TT_MEM_SPACE_DRAM, true, 0u, 4u) != FT_TT_OK,
          "a 4-byte register read is refused by the DRAM read check");
    /* And the converse: the register accessor's width is not a knob that could be
     * "upgraded" to 16 B without changing what a 4-byte store means. */
    check(ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_WORD32) == 4u,
          "WORD32 is still one 32-bit access per word");
    check(ft_tt_noc_transfer_width_bytes(FT_TT_NOC_WIDTH_BLOCK16) == 16u,
          "BLOCK16 is still 16 B units");
    check(ft_tt_noc_alignment(FT_TT_MEM_SPACE_DRAM, true) == 64u,
          "DRAM reads still need 64 B — a 32 B row-0 read would be refused");
    check(ft_tt_noc_alignment(FT_TT_MEM_SPACE_DRAM, false) == 16u,
          "DRAM writes still take 16 B, which is what makes the write 32x and not 16x");
}

/* ================================================================
 * main
 * ================================================================ */

int main(void)
{
    sim_bank bank;
    ft_tt_noc_access noc;
    size_t widest = (size_t)ft_tt_tile_count_for(k_shapes[0]) * FT_TT_TILE_ROW0_TILE_BYTES;
    size_t shape;

    memset(&bank, 0, sizeof(bank));
    bank.size = widest;
    bank.bytes = calloc(1, widest);
    if (!bank.bytes) {
        printf("allocation failed\n");
        return 1;
    }
    noc.read = sim_read;
    noc.write = sim_write;
    noc.user = &bank;

    printf("== chunk offsets\n");
    check_chunk_offsets();

    printf("== the bulk path cannot carry run control\n");
    check_registers_cannot_take_the_bulk_path();

    printf("== read: row 0 only vs the full-tile path\n");
    for (shape = 0; shape < sizeof(k_shapes) / sizeof(k_shapes[0]); shape++)
        check_read_matches_full_tile(&bank, &noc, k_shapes[shape]);

    printf("== write: row 0 only vs a full tilize\n");
    for (shape = 0; shape < sizeof(k_shapes) / sizeof(k_shapes[0]); shape++)
        check_write_matches_full_tile(&bank, &noc, k_shapes[shape]);

    free(bank.bytes);
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
