/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_arc_telemetry.c
 * @brief Implementation of the ARC telemetry table walk (transport-agnostic), the grid
 *   built from it, and the kernel-driver transport over one TLB window.
 *
 * @status Complete; covered off-card by tests/test_arc_telemetry_walk.c and
 *   tests/test_runtime_seam.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_arc_telemetry.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ft_tt_coord.h"
#include "ft_tt_kmd.h"
#include "ft_tt_tlb.h"

/** The five grid tags, in the order the grid fields are filled. */
#define GRID_TAG_COUNT 5u

/** Bytes of the table header: version word + entry-count word. */
#define TABLE_HEADER_BYTES 8u

/* ------------------------------------------------------------------------- */
/* Bounds and bit helpers                                                    */
/* ------------------------------------------------------------------------- */

/** True when `[address, address + byte_count)` lies inside the ARC CSM window. Computed
 *  in 64 bits so neither end can wrap. */
static bool csm_contains_range(const ft_tt_arch_telemetry *telemetry, uint64_t address,
                               uint64_t byte_count)
{
    uint64_t csm_begin = telemetry->csm_base;
    uint64_t csm_end = csm_begin + telemetry->csm_size;

    return address >= csm_begin && address + byte_count <= csm_end;
}

static uint32_t popcount32(uint32_t value)
{
    uint32_t count = 0;

    while (value != 0u) {
        value &= value - 1u;
        count++;
    }
    return count;
}

/** Popcount of `value` restricted to its `bit_count` low bits (all 32 when >= 32). */
static uint32_t popcount_low_bits(uint32_t value, uint32_t bit_count)
{
    if (bit_count == 0u)
        return 0u;
    if (bit_count < 32u)
        value &= (1u << bit_count) - 1u;
    return popcount32(value);
}

/* ------------------------------------------------------------------------- */
/* Table location                                                            */
/* ------------------------------------------------------------------------- */

/** Where a published table lives: table base, values base and entry count, all
 *  already checked against the CSM. */
typedef struct telemetry_table_location {
    uint32_t table_address;
    uint32_t values_address;
    uint32_t entry_count;
} telemetry_table_location;

/** Read one scratch-register pointer. */
static int read_scratch_pointer(const ft_tt_arch_telemetry *telemetry,
                                ft_tt_arc_read32_fn read32, void *user, uint32_t index,
                                uint32_t *out_pointer)
{
    uint32_t address = telemetry->scratch_base + 4u * index;

    return read32(user, address, out_pointer);
}

/** Read both pointers and the entry count, refusing anything outside the CSM. */
static int locate_telemetry_table(const ft_tt_arch_telemetry *telemetry,
                                  ft_tt_arc_read32_fn read32, void *user,
                                  telemetry_table_location *out_location)
{
    telemetry_table_location location;
    int rc;

    rc = read_scratch_pointer(telemetry, read32, user, telemetry->table_pointer_scratch_index,
                              &location.table_address);
    if (rc != FT_TT_OK)
        return rc;
    rc = read_scratch_pointer(telemetry, read32, user, telemetry->values_pointer_scratch_index,
                              &location.values_address);
    if (rc != FT_TT_OK)
        return rc;

    if (location.table_address == 0u) {
        ft_tt_log_error("arc telemetry: ARC has not published a table yet (table pointer 0)");
        return FT_TT_EUNAVAIL;
    }
    if (!csm_contains_range(telemetry, location.table_address, TABLE_HEADER_BYTES)) {
        ft_tt_log_error("arc telemetry: table pointer 0x%08X is outside the CSM "
                        "[0x%08X, +0x%X)", location.table_address, telemetry->csm_base,
                        telemetry->csm_size);
        return FT_TT_EIO;
    }
    if (!csm_contains_range(telemetry, location.values_address, 4u)) {
        ft_tt_log_error("arc telemetry: values pointer 0x%08X is outside the CSM "
                        "[0x%08X, +0x%X)", location.values_address, telemetry->csm_base,
                        telemetry->csm_size);
        return FT_TT_EIO;
    }

    rc = read32(user, location.table_address + 4u, &location.entry_count);
    if (rc != FT_TT_OK)
        return rc;
    if (location.entry_count > FT_TT_ARC_TELEMETRY_MAX_ENTRIES) {
        ft_tt_log_error("arc telemetry: entry count %u exceeds %u: malformed table",
                        location.entry_count, FT_TT_ARC_TELEMETRY_MAX_ENTRIES);
        return FT_TT_EIO;
    }
    if (!csm_contains_range(telemetry, (uint64_t)location.table_address + TABLE_HEADER_BYTES,
                            4ull * location.entry_count)) {
        ft_tt_log_error("arc telemetry: %u entries at 0x%08X run past the CSM",
                        location.entry_count, location.table_address + TABLE_HEADER_BYTES);
        return FT_TT_EIO;
    }

    *out_location = location;
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* The walk                                                                  */
/* ------------------------------------------------------------------------- */

/** Read the value an entry points at into every still-unfound request for `tag`. */
static int resolve_entry_for_requests(const ft_tt_arch_telemetry *telemetry,
                                      ft_tt_arc_read32_fn read32, void *user,
                                      uint32_t values_address, uint32_t entry,
                                      const uint16_t *tags, unsigned count,
                                      uint32_t *out_values, bool *out_found,
                                      unsigned *found_total)
{
    uint16_t tag = (uint16_t)(entry & 0xFFFFu);
    uint32_t word_offset = entry >> 16;
    uint64_t value_address = (uint64_t)values_address + 4ull * word_offset;
    bool value_read = false;
    uint32_t value = 0u;
    unsigned i;
    int rc;

    for (i = 0; i < count; i++) {
        if (out_found[i] || tags[i] != tag)
            continue;
        if (!value_read) {
            if (!csm_contains_range(telemetry, value_address, 4u)) {
                ft_tt_log_error("arc telemetry: tag %u value at 0x%08llX (offset %u words) "
                                "is outside the CSM", (unsigned)tag,
                                (unsigned long long)value_address, word_offset);
                return FT_TT_EIO;
            }
            rc = read32(user, (uint32_t)value_address, &value);
            if (rc != FT_TT_OK)
                return rc;
            value_read = true;
        }
        out_values[i] = value;
        out_found[i] = true;
        (*found_total)++;
    }
    return FT_TT_OK;
}

int ft_tt_arc_telemetry_read_tags_via(const ft_tt_arch *arch, ft_tt_arc_read32_fn read32,
                                      void *user, const uint16_t *tags, unsigned count,
                                      uint32_t *out_values, bool *out_found)
{
    telemetry_table_location location;
    unsigned found_total = 0;
    uint32_t entry_index;
    int rc;

    if (!arch || !read32 || !tags || !out_values || !out_found)
        return FT_TT_EINVAL;
    memset(out_values, 0, (size_t)count * sizeof(*out_values));
    memset(out_found, 0, (size_t)count * sizeof(*out_found));

    rc = locate_telemetry_table(&arch->telemetry, read32, user, &location);
    if (rc != FT_TT_OK)
        return rc;

    for (entry_index = 0; entry_index < location.entry_count && found_total < count;
         entry_index++) {
        uint32_t entry;

        rc = read32(user, location.table_address + TABLE_HEADER_BYTES + 4u * entry_index,
                    &entry);
        if (rc != FT_TT_OK)
            return rc;
        rc = resolve_entry_for_requests(&arch->telemetry, read32, user,
                                        location.values_address, entry, tags, count,
                                        out_values, out_found, &found_total);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Grid                                                                      */
/* ------------------------------------------------------------------------- */

int ft_tt_arc_telemetry_read_grid_via(const ft_tt_arch *arch, ft_tt_arc_read32_fn read32,
                                      void *user, ft_tt_grid_info *out_grid)
{
    static const char *const tag_names[GRID_TAG_COUNT] = {
        "enabled_tensix_col", "enabled_eth", "enabled_gddr", "enabled_l2cpu",
        "noc_translation"
    };
    uint16_t tags[GRID_TAG_COUNT];
    uint32_t values[GRID_TAG_COUNT];
    bool found[GRID_TAG_COUNT];
    ft_tt_grid_info grid;
    unsigned i;
    int rc;

    if (!arch || !read32 || !out_grid)
        return FT_TT_EINVAL;
    tags[0] = arch->telemetry.tag_enabled_tensix_col;
    tags[1] = arch->telemetry.tag_enabled_eth;
    tags[2] = arch->telemetry.tag_enabled_gddr;
    tags[3] = arch->telemetry.tag_enabled_l2cpu;
    tags[4] = arch->telemetry.tag_noc_translation;

    rc = ft_tt_arc_telemetry_read_tags_via(arch, read32, user, tags, GRID_TAG_COUNT, values,
                                           found);
    if (rc != FT_TT_OK)
        return rc;
    for (i = 0; i < GRID_TAG_COUNT; i++) {
        if (!found[i]) {
            ft_tt_log_error("arc telemetry: grid tag %s (%u) is missing from the table",
                            tag_names[i], (unsigned)tags[i]);
            return FT_TT_EIO;
        }
    }

    memset(&grid, 0, sizeof(grid));
    grid.arch = arch;
    grid.tensix_enabled_col = values[0];
    grid.enabled_eth = values[1];
    grid.enabled_gddr = values[2];
    grid.enabled_l2cpu = values[3];
    grid.noc_translation_enabled = values[4];
    grid.tensix_column_count = popcount_low_bits(values[0], arch->tensix.column_count);
    grid.tensix_core_count = grid.tensix_column_count * arch->tensix.row_count;
    grid.dram_bank_count = popcount_low_bits(values[2], arch->dram.bank_count);
    grid.valid = true;

    *out_grid = grid;
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Kernel-driver transport                                                   */
/* ------------------------------------------------------------------------- */

/** State of the TLB-window read32: the window, the ARC tile, and the page it is aimed at. */
typedef struct tlb_arc_reader {
    ft_tt_ctx *ctx;
    ft_tt_tlb_window *window;
    uint64_t window_size;
    uint16_t arc_x;
    uint16_t arc_y;
    bool aimed;
    uint64_t aimed_page;
} tlb_arc_reader;

/** ft_tt_arc_read32_fn over a TLB window: re-aim only when the page changes. */
static int tlb_arc_read32(void *user, uint32_t address, uint32_t *out_value)
{
    tlb_arc_reader *reader = (tlb_arc_reader *)user;
    uint64_t page = (uint64_t)address & ~(reader->window_size - 1u);
    volatile uint8_t *aperture;
    int rc;

    if (!reader->aimed || reader->aimed_page != page) {
        rc = ft_tt_tlb_window_aim(reader->ctx, reader->window, reader->arc_x, reader->arc_y,
                                  page);
        if (rc != FT_TT_OK)
            return rc;
        reader->aimed = true;
        reader->aimed_page = page;
    }
    aperture = (volatile uint8_t *)ft_tt_tlb_window_aperture(reader->window);
    if (!aperture)
        return FT_TT_EIO;
    *out_value = *(volatile uint32_t *)(aperture + ((uint64_t)address - page));
    return FT_TT_OK;
}

/** Fill everything in the reader but the window. */
static int tlb_arc_reader_init(ft_tt_ctx *ctx, const ft_tt_arch *arch, tlb_arc_reader *reader)
{
    ft_tt_noc_coord arc;
    int rc;

    memset(reader, 0, sizeof(*reader));
    rc = ft_tt_coord_arc(arch, FT_TT_COORD_NOC0, &arc);
    if (rc != FT_TT_OK)
        return rc;
    if (arc.x > UINT16_MAX || arc.y > UINT16_MAX) {
        ft_tt_log_error("arc telemetry: ARC coordinate (%u, %u) does not fit a TLB aim",
                        arc.x, arc.y);
        return FT_TT_EIO;
    }
    reader->ctx = ctx;
    reader->arc_x = (uint16_t)arc.x;
    reader->arc_y = (uint16_t)arc.y;
    return FT_TT_OK;
}

int ft_tt_arc_telemetry_read_grid(ft_tt_ctx *ctx, ft_tt_grid_info *out_grid)
{
    const ft_tt_arch *arch;
    tlb_arc_reader reader;
    ft_tt_grid_info grid;
    int free_rc;
    int rc;

    if (!ctx || !out_grid || !ft_tt_ctx_available(ctx))
        return FT_TT_EINVAL;
    arch = ft_tt_ctx_arch(ctx);
    if (!arch)
        return FT_TT_EUNAVAIL;
    rc = tlb_arc_reader_init(ctx, arch, &reader);
    if (rc != FT_TT_OK)
        return rc;

    rc = ft_tt_tlb_window_alloc(ctx, FT_TT_KMD_WINDOW_SIZE_2M, FT_TT_KMD_CACHE_UC,
                                &reader.window);
    if (rc != FT_TT_OK)
        return rc;
    reader.window_size = ft_tt_tlb_window_size(reader.window);

    if (reader.window_size == 0u || arch->telemetry.csm_size > reader.window_size) {
        ft_tt_log_error("arc telemetry: CSM of 0x%X bytes exceeds the 0x%llX-byte window",
                        arch->telemetry.csm_size, (unsigned long long)reader.window_size);
        rc = FT_TT_EIO;
    } else {
        rc = ft_tt_arc_telemetry_read_grid_via(arch, tlb_arc_read32, &reader, &grid);
    }

    free_rc = ft_tt_tlb_window_free(ctx, reader.window);
    if (rc != FT_TT_OK)
        return rc;
    if (free_rc != FT_TT_OK)
        return free_rc;

    rc = ft_tt_grid_set(ctx, &grid);
    if (rc != FT_TT_OK)
        return rc;
    *out_grid = grid;
    return FT_TT_OK;
}
