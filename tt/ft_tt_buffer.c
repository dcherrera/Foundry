/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_buffer.c
 * @brief Implementation of the L1/DRAM bump allocator: arena setup from live
 *   telemetry, aligned carve-off, and a LIFO free that refuses rather than
 *   corrupts.
 *
 * @status Written on the Mac and unit-tested off-hardware (tests/test_noc.c).
 * @issues None
 * @todo None
 */

#include "ft_tt_buffer.h"

#include <string.h>

/* ------------------------------------------------------------------------- */
/* Arena helpers                                                             */
/* ------------------------------------------------------------------------- */

/** Round `value` up to a multiple of `alignment`, which is a power of two.
 *  Returns 0 on overflow so the caller can refuse. */
static uint64_t align_up(uint64_t value, uint32_t alignment)
{
    uint64_t mask = (uint64_t)alignment - 1u;

    if (value > UINT64_MAX - mask)
        return 0u;
    return (value + mask) & ~mask;
}

/** The arena for (space, bank), or NULL if the pair names no arena. */
static ft_tt_arena *arena_for(ft_tt_allocator *allocator, ft_tt_memory_space space,
                              uint32_t logical_bank)
{
    if (!allocator || !allocator->initialized)
        return NULL;

    switch (space) {
    case FT_TT_MEM_SPACE_L1:
        return logical_bank == 0u ? &allocator->l1 : NULL;
    case FT_TT_MEM_SPACE_DRAM:
        return logical_bank < allocator->dram_bank_count ? &allocator->dram[logical_bank] : NULL;
    default:
        return NULL;
    }
}

static const ft_tt_arena *arena_for_const(const ft_tt_allocator *allocator,
                                          ft_tt_memory_space space, uint32_t logical_bank)
{
    /* One implementation, and the const cast never escapes this function. */
    return arena_for((ft_tt_allocator *)(uintptr_t)allocator, space, logical_bank);
}

/* ------------------------------------------------------------------------- */
/* Lifecycle                                                                 */
/* ------------------------------------------------------------------------- */

int ft_tt_allocator_init(ft_tt_allocator *allocator, ft_tt_ctx *ctx, const ft_tt_grid_info *grid,
                         uint64_t l1_floor)
{
    uint64_t floor;
    uint32_t bank;

    if (!allocator || !grid)
        return FT_TT_EINVAL;

    /* The bank count is only knowable from ARC telemetry (spec §3.4). Refuse
     * rather than fall back to the nominal 8. */
    if (!grid->valid) {
        ft_tt_log_error("allocator: the grid has not been read from ARC telemetry");
        return FT_TT_EUNAVAIL;
    }
    /* Every bound below is the LIVE one or the selected part's; the only constant
     * involved is this backend's own array capacity. ft_tt_buffer.h's header note
     * says why those are kept as three separate numbers. */
    if (!grid->arch) {
        ft_tt_log_error("allocator: the grid names no architecture, so no DRAM geometry is "
                        "known — refusing rather than assuming a part");
        return FT_TT_EINVAL;
    }
    if (grid->dram_bank_count == 0u || grid->dram_bank_count > FT_TT_MAX_DRAM_BANKS) {
        ft_tt_log_error("allocator: %u DRAM banks is outside this backend's capacity of %u arenas",
                        grid->dram_bank_count, FT_TT_MAX_DRAM_BANKS);
        return FT_TT_EINVAL;
    }
    if (grid->dram_bank_count > grid->arch->dram.bank_count) {
        /* Telemetry claiming more banks than the part has means the mask was
         * misread. Allocating in a bank that is not there is a NoC write to
         * nowhere, which is not a failure this would notice later. */
        ft_tt_log_error("allocator: telemetry reports %u DRAM banks but %s has %u — refusing",
                        grid->dram_bank_count, grid->arch->name, grid->arch->dram.bank_count);
        return FT_TT_EINVAL;
    }

    floor = (l1_floor == 0u) ? (uint64_t)FT_TT_L1_USABLE_BASE : l1_floor;
    floor = align_up(floor, FT_TT_L1_ALIGNMENT);
    if (floor < (uint64_t)FT_TT_L1_USABLE_BASE || floor >= (uint64_t)FT_TT_MEM_L1_SIZE) {
        /* Below FT_TT_L1_USABLE_BASE is firmware, mailboxes and routing tables.
         * Handing that out does not fail a test, it wedges a core. */
        ft_tt_log_error("allocator: L1 floor 0x%llx is outside [0x%x, 0x%x)",
                        (unsigned long long)floor, FT_TT_L1_USABLE_BASE, FT_TT_MEM_L1_SIZE);
        return FT_TT_EINVAL;
    }

    memset(allocator, 0, sizeof(*allocator));
    allocator->ctx = ctx;
    allocator->dram_bank_count = grid->dram_bank_count;

    allocator->l1.floor = floor;
    /* FT_TT_MEM_L1_SIZE, never FT_TT_MEM_MAX_KERNEL_SIZE — the latter is 896
     * bytes larger than the L1 above MEM_MAP_END (ft_tt_memmap.h @issues). */
    allocator->l1.ceiling = (uint64_t)FT_TT_MEM_L1_SIZE;
    allocator->l1.cursor = floor;

    /* Bank size from the SELECTED part, not from the compiled-in memory map. The
     * two agree by construction — ft_tt_arch_check_headers() refuses a part where
     * they do not — so this is the same number today and one fewer Blackhole
     * constant on the path. */
    for (bank = 0; bank < allocator->dram_bank_count; bank++) {
        allocator->dram[bank].floor = 0u;
        allocator->dram[bank].ceiling = grid->arch->core.dram_bytes_per_bank;
        allocator->dram[bank].cursor = 0u;
    }

    allocator->initialized = true;
    return FT_TT_OK;
}

int ft_tt_allocator_reset(ft_tt_allocator *allocator)
{
    uint32_t bank;

    if (!allocator || !allocator->initialized)
        return FT_TT_EINVAL;

    if (allocator->ctx) {
        int64_t released = 0;

        for (bank = 0; bank < allocator->dram_bank_count; bank++)
            released += (int64_t)(allocator->dram[bank].cursor - allocator->dram[bank].floor);
        ft_tt_ctx_note_device_alloc(allocator->ctx, -released);
    }

    allocator->l1.cursor = allocator->l1.floor;
    allocator->l1.live_count = 0u;
    for (bank = 0; bank < allocator->dram_bank_count; bank++) {
        allocator->dram[bank].cursor = allocator->dram[bank].floor;
        allocator->dram[bank].live_count = 0u;
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Allocation                                                                */
/* ------------------------------------------------------------------------- */

int ft_tt_buffer_alloc(ft_tt_allocator *allocator, ft_tt_memory_space space, uint32_t logical_bank,
                       size_t bytes, ft_tt_buffer *out_buffer)
{
    ft_tt_arena *arena;
    uint32_t alignment;
    uint64_t reserved;
    uint64_t address;

    if (!allocator || !out_buffer || bytes == 0u)
        return FT_TT_EINVAL;

    arena = arena_for(allocator, space, logical_bank);
    if (!arena) {
        ft_tt_log_error("allocator: no %s arena for bank %u", ft_tt_memory_space_string(space),
                        logical_bank);
        return FT_TT_EINVAL;
    }

    /* Align to the stricter of the two directions so the buffer is legal for
     * both ft_tt_write_* and ft_tt_read_* (64 B in DRAM, 16 B in L1). */
    alignment = ft_tt_noc_alignment(space, true);
    if (alignment == 0u)
        return FT_TT_EINVAL;

    reserved = align_up((uint64_t)bytes, alignment);
    if (reserved == 0u)
        return FT_TT_EINVAL;

    address = arena->cursor;
    if (reserved > arena->ceiling - address) {
        ft_tt_log_error("allocator: %s bank %u cannot fit %zu bytes (%llu free of %llu)",
                        ft_tt_memory_space_string(space), logical_bank, bytes,
                        (unsigned long long)(arena->ceiling - address),
                        (unsigned long long)(arena->ceiling - arena->floor));
        return FT_TT_ENOMEM;
    }

    arena->cursor = address + reserved;
    arena->live_count++;

    if (space == FT_TT_MEM_SPACE_DRAM && allocator->ctx)
        ft_tt_ctx_note_device_alloc(allocator->ctx, (int64_t)reserved);

    out_buffer->space = space;
    out_buffer->logical_bank = (space == FT_TT_MEM_SPACE_DRAM) ? logical_bank : 0u;
    out_buffer->address = address;
    out_buffer->bytes = bytes;
    out_buffer->reserved_bytes = (size_t)reserved;
    return FT_TT_OK;
}

int ft_tt_buffer_free(ft_tt_allocator *allocator, const ft_tt_buffer *buffer)
{
    ft_tt_arena *arena;

    if (!allocator || !buffer)
        return FT_TT_EINVAL;

    arena = arena_for(allocator, buffer->space, buffer->logical_bank);
    if (!arena) {
        ft_tt_log_error("allocator: freeing a buffer in no arena (%s bank %u)",
                        ft_tt_memory_space_string(buffer->space), buffer->logical_bank);
        return FT_TT_EINVAL;
    }
    if (arena->live_count == 0u) {
        ft_tt_log_error("allocator: freeing from an empty %s arena",
                        ft_tt_memory_space_string(buffer->space));
        return FT_TT_EINVAL;
    }
    if (buffer->address + (uint64_t)buffer->reserved_bytes != arena->cursor) {
        ft_tt_log_error(
            "allocator: out-of-order free of %s 0x%llx+%zu; the cursor is at 0x%llx (bump "
            "allocators only retract their last step)",
            ft_tt_memory_space_string(buffer->space), (unsigned long long)buffer->address,
            buffer->reserved_bytes, (unsigned long long)arena->cursor);
        return FT_TT_EINVAL;
    }

    arena->cursor = buffer->address;
    arena->live_count--;

    if (buffer->space == FT_TT_MEM_SPACE_DRAM && allocator->ctx)
        ft_tt_ctx_note_device_alloc(allocator->ctx, -(int64_t)buffer->reserved_bytes);

    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Queries                                                                   */
/* ------------------------------------------------------------------------- */

size_t ft_tt_allocator_free_bytes(const ft_tt_allocator *allocator, ft_tt_memory_space space,
                                  uint32_t logical_bank)
{
    const ft_tt_arena *arena = arena_for_const(allocator, space, logical_bank);
    return arena ? (size_t)(arena->ceiling - arena->cursor) : 0u;
}

size_t ft_tt_allocator_used_bytes(const ft_tt_allocator *allocator, ft_tt_memory_space space,
                                  uint32_t logical_bank)
{
    const ft_tt_arena *arena = arena_for_const(allocator, space, logical_bank);
    return arena ? (size_t)(arena->cursor - arena->floor) : 0u;
}

uint32_t ft_tt_allocator_live_count(const ft_tt_allocator *allocator, ft_tt_memory_space space,
                                    uint32_t logical_bank)
{
    const ft_tt_arena *arena = arena_for_const(allocator, space, logical_bank);
    return arena ? arena->live_count : 0u;
}
