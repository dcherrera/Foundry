/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_buffer.h
 * @brief Device memory allocation for the Foundry Blackhole backend: hand out
 *   L1 and DRAM addresses that are legal to write, correctly aligned, and
 *   inside the space that is actually free.
 *
 * Module prefix: ft_tt_. Pure C99 + libc. Touches no device and issues no NoC
 * transaction — it is arithmetic over ft_tt_grid_info and ft_tt_memmap.h, so it
 * is fully testable with no card in the machine.
 *
 * ---------------------------------------------------------------------------
 * IT IS A BUMP ALLOCATOR, AND IT SAYS SO
 * ---------------------------------------------------------------------------
 * One cursor per arena, allocation walks it up, and ft_tt_buffer_free() only
 * accepts the MOST RECENT live allocation in that arena — anything else is
 * refused with FT_TT_EINVAL rather than silently leaking or corrupting the
 * cursor. Bulk teardown is ft_tt_allocator_reset(). The plan (Phase 1) calls a
 * bump allocator sufficient; a real free-list belongs with Phase 5's
 * core-group partitioning, where lifetimes stop being LIFO.
 *
 * ---------------------------------------------------------------------------
 * THE BOUNDS ARE REAL, NOT NOMINAL
 * ---------------------------------------------------------------------------
 *  - L1 is 1536 KiB, and the bottom 40,832 B of it is firmware text, the
 *    mailbox region and the fabric/routing tables (ft_tt_memmap.h). The floor
 *    is FT_TT_L1_USABLE_BASE, leaving 1,532,032 B.
 *  - **FT_TT_L1_USABLE_BASE IS NOT ENOUGH ONCE A CORE IS RUNNING.** 0x9F80 is
 *    exactly where the per-processor firmware `.data` staging regions START,
 *    so a default allocator will hand out the
 *    `.data` image of the firmware currently executing on the core. Any caller
 *    that has loaded images must pass `l1_floor` =
 *    ft_tt_image_placement_l1_floor() over the placements
 *    ft_tt_launch_load_image() reported.
 *  - **FT_TT_MEM_MAX_KERNEL_SIZE IS NOT A BOUND.** Upstream rounds it up to a
 *    whole 1497 KiB, which OVERSHOOTS the L1 that exists above MEM_MAP_END by
 *    896 bytes. Using it here would hand out 896 bytes of address space that is
 *    not there. FT_TT_L1_USABLE_SIZE is the bound.
 *  - DRAM is one arena per enabled bank, sized from the selected architecture's
 *    `dram_bytes_per_bank` (0xff000000 = 4,278,190,080 B on Blackhole). The bank
 *    COUNT comes from ARC telemetry and nowhere else — 8 on this card, and never
 *    assumed.
 *
 * ---------------------------------------------------------------------------
 * THE ARENA ARRAY IS A CAPACITY, AND THE BOUND IT CHECKS IS THE LIVE ONE
 * ---------------------------------------------------------------------------
 * `dram[]` is sized by FT_TT_MAX_DRAM_BANKS (ft_tt_arch.h), which is this
 * BACKEND's static capacity — not `FT_TT_NUM_DRAM_BANKS`, which is Blackhole's
 * bank count and a fact about one part. They are both 8 today and that is a
 * coincidence worth keeping separate: the allocator's array cannot be sized from
 * a table selected at runtime on a 7.7 GB host, so it needs a ceiling, and a
 * ceiling read off one architecture's constant is how a number quietly acquires
 * two meanings.
 *
 * Three checks, in the order they can catch something:
 *   1. `ft_tt_arch_check()` refuses an arch table declaring more banks than the
 *      capacity — a part we cannot hold is a named refusal at device open.
 *   2. `ft_tt_allocator_init()` refuses a live bank count above the capacity.
 *   3. ...and above the SELECTED ARCH's own bank count, because telemetry
 *      reporting more banks than the part has means the mask was misread, and
 *      handing out addresses in a bank that is not there is a NoC transaction
 *      to nowhere.
 * Every runtime bank index is then checked against the LIVE count, never the
 * constant.
 *
 * Addresses are aligned to the strictest rule the NoC helpers impose on that
 * space (ft_tt_noc_alignment): 16 B in L1, 64 B in DRAM. So any buffer this
 * hands out is legal for both ft_tt_write_* and ft_tt_read_*.
 *
 * @status Unit-tested off-hardware, and exercised on the card as the source of
 *   every address test_noc.c writes to. The
 *   `l1_floor` override now has a caller: ft_tt_image_placement_l1_floor()
 *   (ft_tt_launch.h) turns the placements of the loaded images into it, and
 *   tests/test_launch.c proves an allocation that would have landed on a
 *   loaded image is refused.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_BUFFER_H
#define FT_TT_BUFFER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_arch.h"
#include "ft_tt_device.h"
#include "ft_tt_memmap.h"
#include "ft_tt_noc.h"

/* ================================================================
 * Types
 * ================================================================ */

/**
 * One allocation. Plain data, copied by value.
 *
 * An L1 buffer is a CORE-RELATIVE address: the same offset exists on every
 * Tensix, and this allocator does not track which cores a caller actually used.
 * That mirrors how tt-metal places buffers across a core grid, and it is why
 * `logical_bank` is meaningless for L1.
 */
typedef struct ft_tt_buffer {
    ft_tt_memory_space space;
    uint32_t logical_bank;   /**< DRAM only: which enabled bank. 0 for L1 */
    uint64_t address;        /**< device address within that space */
    size_t bytes;            /**< what the caller asked for */
    size_t reserved_bytes;   /**< what the arena consumed, i.e. bytes rounded up */
} ft_tt_buffer;

/** One bump arena. Fields are public so the struct can be caller-owned; treat
 *  them as read-only and use the accessors. */
typedef struct ft_tt_arena {
    uint64_t floor;
    uint64_t ceiling; /**< exclusive */
    uint64_t cursor;
    uint32_t live_count;
} ft_tt_arena;

/**
 * L1 plus one arena per DRAM bank. Caller-owned storage — declaring it rather
 * than allocating it keeps the backend allocation-free on the host, which
 * matters on a 2-core / 7.7 GB box.
 */
typedef struct ft_tt_allocator {
    ft_tt_ctx *ctx; /**< optional; only for ft_tt_mem_info accounting */
    bool initialized;
    uint32_t dram_bank_count;
    ft_tt_arena l1;
    ft_tt_arena dram[FT_TT_MAX_DRAM_BANKS]; /**< capacity, not a bank count */
} ft_tt_allocator;

/* ================================================================
 * Lifecycle
 * ================================================================ */

/**
 * Set up the arenas from a grid read off ARC telemetry.
 *
 * @param allocator  caller-owned storage, zeroed and filled on success
 * @param ctx        may be NULL. When given, DRAM allocations are reported
 *                   through ft_tt_ctx_note_device_alloc() so ft_tt_mem_info()
 *                   answers a real free figure instead of the full 31.88 GiB
 * @param grid       topology; `valid` must be true, `arch` must be set, and
 *                   `dram_bank_count` must be 1..min(FT_TT_MAX_DRAM_BANKS,
 *                   grid->arch->dram.bank_count). The count is NEVER assumed
 * @param l1_floor   lowest L1 address to hand out, or 0 for
 *                   FT_TT_L1_USABLE_BASE. Rounded up to 16 B. Must be below
 *                   the top of L1 and at or above FT_TT_L1_USABLE_BASE —
 *                   allocating over the mailbox or firmware is a wedged core,
 *                   not a failed test
 * @return           FT_TT_OK; FT_TT_EINVAL on a bad argument or floor;
 *                   FT_TT_EUNAVAIL if the grid was never read
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_allocator_init(ft_tt_allocator *allocator, ft_tt_ctx *ctx, const ft_tt_grid_info *grid,
                         uint64_t l1_floor);

/**
 * Return every arena to its floor, dropping all outstanding buffers. Any
 * ft_tt_buffer a caller still holds is dangling afterwards.
 * @return FT_TT_OK, or FT_TT_EINVAL.
 */
int ft_tt_allocator_reset(ft_tt_allocator *allocator);

/* ================================================================
 * Allocation
 * ================================================================ */

/**
 * Carve `bytes` off the top of an arena.
 *
 * The returned address is aligned to ft_tt_noc_alignment() for the space (16 B
 * L1, 64 B DRAM), and `reserved_bytes` is `bytes` rounded up to the same, so a
 * caller may safely read back everything it wrote.
 *
 * @param allocator     an initialised allocator
 * @param space         FT_TT_MEM_SPACE_L1 or FT_TT_MEM_SPACE_DRAM
 * @param logical_bank  DRAM: 0..dram_bank_count-1. L1: must be 0
 * @param bytes         > 0
 * @param out_buffer    written only on success
 * @return              FT_TT_OK; FT_TT_EINVAL on a bad argument;
 *                      FT_TT_ENOMEM when the arena cannot fit it (the shortfall
 *                      is logged)
 *
 * Thread-safe: no (mutates the allocator)   Reentrant: no   Signal-safe: no
 */
int ft_tt_buffer_alloc(ft_tt_allocator *allocator, ft_tt_memory_space space, uint32_t logical_bank,
                       size_t bytes, ft_tt_buffer *out_buffer);

/**
 * Give a buffer back.
 *
 * **LIFO ONLY.** A bump allocator can only retract its last step, so this
 * succeeds exactly when `buffer` is the most recent live allocation in its
 * arena, and returns FT_TT_EINVAL otherwise with what it expected logged. It
 * never silently ignores a bad free — a leaked DRAM arena on a card with 201
 * TLB windows and no MMU is not something to discover later.
 *
 * @return FT_TT_OK; FT_TT_EINVAL on a bad argument, an empty arena, or an
 *         out-of-order free
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_buffer_free(ft_tt_allocator *allocator, const ft_tt_buffer *buffer);

/* ================================================================
 * Queries
 * ================================================================ */

/** Bytes still allocatable in one arena, or 0 for a bad argument. */
size_t ft_tt_allocator_free_bytes(const ft_tt_allocator *allocator, ft_tt_memory_space space,
                                  uint32_t logical_bank);

/** Bytes handed out and not yet freed in one arena, or 0 for a bad argument. */
size_t ft_tt_allocator_used_bytes(const ft_tt_allocator *allocator, ft_tt_memory_space space,
                                  uint32_t logical_bank);

/** Live allocations in one arena, or 0 for a bad argument. */
uint32_t ft_tt_allocator_live_count(const ft_tt_allocator *allocator, ft_tt_memory_space space,
                                    uint32_t logical_bank);

#endif /* FT_TT_BUFFER_H */
