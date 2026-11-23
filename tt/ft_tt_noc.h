/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_noc.h
 * @brief Named NoC read/write primitives for the Foundry Blackhole backend:
 *   move an arbitrary run of bytes between host memory and a Tensix's L1 or a
 *   DRAM bank, sliding one TLB window across the span as needed.
 *
 * Module prefix: ft_tt_ (foundry tenstorrent). Pure C99 + libc, plus SSE2
 * intrinsics on x86 (see the copy-shape note below). No tt-metal, no tt-umd.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS ADDS OVER THE WINDOW LAYER
 * ---------------------------------------------------------------------------
 * ft_tt_tlb.h owns ONE 2 MiB aperture aimed at ONE endpoint. Everything larger
 * than 2 MiB, or straddling a 2 MiB boundary, has to move the window mid-copy.
 * That is this file: a transfer of any length at any (aligned) address, with
 * the re-aiming hidden. With no BAR4 there are no 4 GiB windows (spec §2.3), so
 * crossing 32 GB of DRAM is ~16,000 window moves at a measured 1.23 µs each —
 * ~20 ms in total, which is not a bottleneck.
 *
 * ---------------------------------------------------------------------------
 * ONE WINDOW WALK IN THE BACKEND
 * ---------------------------------------------------------------------------
 * Phase 2's run control (ft_tt_launch.c) shipped its own 4-byte-granular copy
 * loop before this file existed. It cannot call ft_tt_write_l1() — it has to
 * reach a 4-byte register at 0xFFB121B0, outside every memory space, and write
 * sub-16 B mailbox fields, both of which the L1 helper correctly refuses. So
 * the seam is ft_tt_noc_transfer_raw(): the window walk and the copy, with a
 * caller-chosen store width and no space bounds. The named helpers below are
 * that function plus ft_tt_noc_check_transfer(); the launch path is that
 * function with FT_TT_NOC_WIDTH_WORD32. There is no second implementation.
 *
 * ---------------------------------------------------------------------------
 * COORDINATES ARE PARAMETERS. THEY ARE NEVER PROBED.
 * ---------------------------------------------------------------------------
 * Every entry point takes an ft_tt_noc_coord the CALLER derived from
 * ft_tt_coord.h against live ARC telemetry, in the TRANSLATED system — that is
 * what a TLB is aimed with (docs/tt_driver_notes.md §8, proven on this card by
 * a DRAM round-trip at translated (17,12), a coordinate that does not exist in
 * NoC0 space). Nothing here guesses, defaults, or scans. A wrong endpoint does
 * not fail a read, it hangs the chip, and only a physical power cycle recovers
 * it (spec §7.3). That has already happened here once.
 *
 * ---------------------------------------------------------------------------
 * THE COPY SHAPE, AND WHY IT IS NOT A memcpy
 * ---------------------------------------------------------------------------
 * The aperture is device MMIO. tt-umd moves 16 x 16 B per 256 B block because
 * that is what keeps the PCIe transaction count where it wants it — but their
 * device_memcpy.cpp is AVX2 and this host has NO AVX at all (spec §8.4), so
 * their binary SIGILLs here exactly like tt_umd did. We keep the 16 B unit and
 * the 256 B block and issue them as SSE2 `movdqu`, which is baseline x86-64.
 * On any non-x86 host (the Mac, for compile checks) it degrades to a 64-bit
 * store loop of the same shape.
 *
 * Measured through the raw window layer (spec §9.8): host->card 287 MB/s WC vs
 * 53 MB/s UC; card->host 2.3 MB/s in BOTH modes. So: WC for bulk, UC for
 * registers. The 125x read asymmetry is a property of non-posted reads with no
 * device->host DMA on Blackhole — it is a design input, not a tuning target.
 *
 * @status VERIFIED ON THE CARD 2026-08-11: L1
 *   round-trip on a derived Tensix, all 8 DRAM banks, a 5 MiB transfer across
 *   four windows, and alignment refusals that never aim the window. The copy
 *   shape was checked in the disassembly on the card host — no VEX instructions.
 *   INFERRED, not measured: the UC cache mode through these helpers (only WC
 *   was exercised), and the 64 B DRAM read alignment, which is transliterated
 *   from the arch constants and enforced conservatively rather than probed.
 *   Everything that touches the NoC is behind FT_TT_PHASE1B_NOC_ENABLED and
 *   returns FT_TT_EPERM in a default build.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_NOC_H
#define FT_TT_NOC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_kmd.h"
#include "ft_tt_memmap.h"
#include "ft_tt_noc_copy.h"
#include "ft_tt_tlb.h"

/* ================================================================
 * Copy shape
 * ================================================================
 *
 * FT_TT_NOC_COPY_UNIT_BYTES, FT_TT_NOC_COPY_BLOCK_BYTES,
 * ft_tt_noc_transfer_width and ft_tt_noc_transfer_width_bytes() now live in
 * ft_tt_noc_copy.h, included above, so that the userspace-driver transport
 * (ft_tt_udrv_noc.h) moves bytes through the SAME code without depending on
 * the tt-kmd layer this file is built on. Callers see no change.
 */

/* ================================================================
 * Memory spaces and their alignment rules
 * ================================================================ */

/** Which device memory a transfer targets. The two differ in alignment, in
 *  size, and in how a caller obtains a coordinate for them. */
typedef enum ft_tt_memory_space {
    FT_TT_MEM_SPACE_L1 = 0,  /**< one Tensix's 1536 KiB L1 */
    FT_TT_MEM_SPACE_DRAM = 1 /**< one GDDR bank, 0xff000000 bytes */
} ft_tt_memory_space;

/** "L1" / "DRAM". Never NULL. */
const char *ft_tt_memory_space_string(ft_tt_memory_space space);

/**
 * Required alignment, in bytes, of both the device address and the length of a
 * transfer.
 *
 * Blackhole's architectural alignments are L1 16 B and DRAM/PCIe 64 B
 * (ft_tt_memmap.h). The vendor's runtime applies the 64 B figure to the READ
 * direction and permits 16 B writes to DRAM, so that asymmetry is reproduced
 * here rather than flattened to the strictest value.
 *
 * @param space    L1 or DRAM
 * @param is_read  true for card->host
 * @return         16 or 64; 0 for an unknown space
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_noc_alignment(ft_tt_memory_space space, bool is_read);

/** Largest device address + 1 in this space: L1 1536 KiB, DRAM 0xff000000. */
uint64_t ft_tt_noc_space_size(ft_tt_memory_space space);

/**
 * Whether a transfer is legal before any device is touched: address and length
 * both aligned per ft_tt_noc_alignment(), and the whole span inside the space.
 *
 * This is a pure function so the rules can be proven off-hardware — the point
 * of separating it out. The helpers below all call it first and refuse without
 * issuing anything.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL (the reason is logged)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_noc_check_transfer(ft_tt_memory_space space, bool is_read, uint64_t address,
                             size_t bytes);

/**
 * Whether a DEVICE-TO-DEVICE transfer is legal — each end against its own space, AND the
 * two ends against EACH OTHER.
 *
 * **ALIGNMENT IS A CONGRUENCE, AND `ft_tt_noc_check_transfer()` cannot express it.** That
 * function scores one end at a time, so two addresses that each pass their own test can
 * still name a transfer the NoC will not do: what it cannot do is move bytes between two
 * addresses sitting at DIFFERENT offsets inside an alignment window. The vendor's check tests exactly
 * that and nothing weaker —
 * `if ((worker_addr & alignment_mask) != (noc_addr & alignment_mask))` — and it is watcher-only, so it does not exist in any image we run.
 *
 * The window is the STRICTER of the two ends' alignments, which is the same rule a derived
 * L1 base already broke once: a 16- and 32-byte-aligned circular buffer that was the LOCAL
 * end of a DRAM read needed 64, and every tile came back byte-exact up to element 992 of
 * 1024 and wrong in its last 64 bytes, with no refusal anywhere.
 *
 * @param local_space     space of the end this core owns
 * @param local_address   that end's address
 * @param remote_space    space of the other end
 * @param remote_address  that end's address
 * @param is_read         true when the transfer READS the remote end
 * @param bytes           length
 * @return                FT_TT_OK, or FT_TT_EINVAL (the reason is logged)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_noc_check_transfer_pair(ft_tt_memory_space local_space, uint64_t local_address,
                                  ft_tt_memory_space remote_space, uint64_t remote_address,
                                  bool is_read, size_t bytes);

/**
 * Whether a transfer is legal for `width` alone — address and length both a
 * multiple of ft_tt_noc_transfer_width_bytes(). This is everything
 * ft_tt_noc_transfer_raw() can check, because a raw transfer names no memory
 * space and so has no bounds to test against.
 *
 * Pure, so the rule is provable off-hardware.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL (the reason is logged)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_noc_check_raw_transfer(ft_tt_noc_transfer_width width, uint64_t address, size_t bytes);

/* ================================================================
 * The window walk, as arithmetic
 * ================================================================ */

/**
 * How much of a transfer fits in one window placement, and where.
 *
 * The walk is the part of this file that can silently corrupt a neighbouring
 * address rather than fail, so it is separated out as pure arithmetic that a
 * unit test on the Mac can prove wrong — a wrong endpoint hangs the chip
 * (spec §7.3), and a wrong OFFSET quietly writes somewhere else on the right
 * one. `ft_tt_noc_window_slice` and the pure form of the computation are in
 * ft_tt_noc_copy.h; this is that plus a log line and an FT_TT_* return.
 */

/**
 * First slice of a transfer of `bytes` at `address` through a `window_bytes`
 * aperture: the window base it needs, the offset inside that window, and how
 * many bytes fit before the next window boundary. Walk a whole transfer by
 * advancing address/host/remaining by `bytes` and calling again.
 *
 * @param window_bytes  aperture size; must be a non-zero power of two
 * @param out_slice     written only on success
 * @return              FT_TT_OK, or FT_TT_EINVAL for a NULL out or a window
 *                      size that is not a power of two
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_noc_window_slice_at(uint64_t address, size_t bytes, size_t window_bytes,
                              ft_tt_noc_window_slice *out_slice);

/* ================================================================
 * The data path
 * ================================================================ */

/**
 * One host<->card data path: a claimed TLB window plus the bookkeeping needed
 * to slide it. Caller-owned storage — declaring it rather than allocating it
 * keeps the layer allocation-free and its lifetime obvious. Treat the fields as
 * private; read them through the accessors.
 *
 * The path OWNS its window and is the only thing that aims it, which is what
 * makes the "already aimed where I need it" fast path sound.
 */
typedef struct ft_tt_noc_path {
    ft_tt_ctx *ctx;
    ft_tt_tlb_window *window;
    size_t window_bytes;
    uint64_t aimed_base;
    uint32_t aimed_x;
    uint32_t aimed_y;
    bool aimed;
    bool owns_window;       /**< false when the window came from the caller */
    uint64_t aim_count;     /**< windows moved since open — the re-windowing proof */
    uint64_t bytes_written; /**< host->card bytes through this path */
    uint64_t bytes_read;    /**< card->host bytes through this path */
    /**
     * PCIe ACCESSES, not bytes — and this link is priced in accesses.
     *
     * A 64-byte read at WORD32 is SIXTEEN non-posted round trips; the same 64 bytes at
     * BLOCK16 is four. The byte counters above cannot tell those apart, and more than one
     * recorded per-descriptor host cost on this backend has turned out to be a transaction
     * count seen through a division. Counted as `bytes / width` where the units are
     * actually issued, so it is the machine's own number and not a caller's model of it.
     */
    uint64_t write_transactions; /**< host->card store units issued through this path */
    uint64_t read_transactions;  /**< card->host load units issued through this path */
} ft_tt_noc_path;

/**
 * Claim a window and build a path over it.
 *
 * @param ctx    open device context
 * @param cache  FT_TT_KMD_CACHE_WC for bulk data (287 MB/s measured),
 *               FT_TT_KMD_CACHE_UC for registers (53 MB/s)
 * @param path   caller-owned storage, zeroed and filled on success
 * @return       FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM when all 201 windows are
 *               in use; FT_TT_EIO otherwise
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_noc_path_open(ft_tt_ctx *ctx, enum ft_tt_kmd_cache_mode cache, ft_tt_noc_path *path);

/**
 * Build a path over a window the CALLER already claimed, for a caller that has
 * its own reason to own the window's lifetime (ft_tt_launch's run control
 * does: the window outlives any one transfer and is freed with the session).
 *
 * The window is BORROWED. ft_tt_noc_path_close() will not free it, and the
 * caller must still call ft_tt_tlb_window_free().
 *
 * The path must be the ONLY thing that aims this window for as long as it is
 * attached — the "already aimed where I need it" fast path is unsound
 * otherwise, and unsound there means writing to a stale endpoint.
 *
 * @param window  a window from ft_tt_tlb_window_alloc(), of a power-of-two size
 * @param path    caller-owned storage, zeroed and filled on success
 * @return        FT_TT_OK; FT_TT_EINVAL; FT_TT_EIO for a window size that is
 *                not a power of two
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_noc_path_attach(ft_tt_ctx *ctx, ft_tt_tlb_window *window, ft_tt_noc_path *path);

/**
 * Blank the path, releasing the window only if this path claimed it (i.e. it
 * came from ft_tt_noc_path_open(), not ft_tt_noc_path_attach()). Safe on a
 * zeroed or already-closed path.
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_EIO (logged).
 *
 * Thread-safe: no (mutates the path)    Reentrant: no    Signal-safe: no
 */
int ft_tt_noc_path_close(ft_tt_noc_path *path);

/** Number of times this path has aimed its window. A transfer that stays inside
 *  one 2 MiB window at an endpoint the window already points at costs ZERO —
 *  which is how a test tells a re-windowing run from a lucky in-window one. */
uint64_t ft_tt_noc_path_aim_count(const ft_tt_noc_path *path);

/** Store units this path has issued host->card. 0 for a NULL path. */
uint64_t ft_tt_noc_path_write_transactions(const ft_tt_noc_path *path);

/** Load units this path has issued card->host — the number a read-bound arm is
 *  actually priced in. 0 for a NULL path. */
uint64_t ft_tt_noc_path_read_transactions(const ft_tt_noc_path *path);

/* ================================================================
 * The unchecked entry point the named helpers are built on
 * ================================================================ */

/**
 * Move `bytes` between host memory and (coord, address), sliding the window as
 * needed, with NO memory-space bounds check and a caller-chosen store width.
 *
 * **This is the low-level seam, not the API to reach for.** It exists because
 * two legitimate callers cannot use the named helpers below:
 *
 *  - the per-Tensix soft-reset register at 0xFFB121B0, which is neither in L1
 *    nor in a DRAM bank and must be touched 4 bytes at a time;
 *    and
 *  - the mailbox protocol, whose go message is 4 bytes — under the 16 B
 *    minimum ft_tt_write_l1() correctly enforces.
 *
 * Everything ELSE should go through ft_tt_{read,write}_{l1,dram}, which are
 * this function plus ft_tt_noc_check_transfer(). Skipping the space check means
 * an out-of-range address is not caught here; it becomes a NoC transaction.
 *
 * Still enforced, before anything reaches the device: a non-NULL path, window
 * and host buffer; `address` and `bytes` aligned to the width; and a coordinate
 * inside the TLB's 6-bit x_end/y_end fields, because a coordinate that does not
 * fit is silently truncated into a DIFFERENT live endpoint.
 *
 * @param path     an open or attached path
 * @param coord    TRANSLATED coordinate from ft_tt_coord.h. NEVER a guess
 * @param address  tile-local address, aligned to the width
 * @param host     source when writing, destination when reading
 * @param bytes    multiple of the width. Zero is a legal no-op
 * @param is_read  true for card->host
 * @param width    FT_TT_NOC_WIDTH_BLOCK16 for data, WORD32 for registers
 * @return         FT_TT_OK; FT_TT_EINVAL; FT_TT_EPERM when the Phase 1b fence
 *                 is closed; FT_TT_EIO
 *
 * Thread-safe: no (the path is stateful)    Reentrant: no    Signal-safe: no
 */
int ft_tt_noc_transfer_raw(ft_tt_noc_path *path, ft_tt_noc_coord coord, uint64_t address,
                           void *host, size_t bytes, bool is_read,
                           ft_tt_noc_transfer_width width);

/* ================================================================
 * Registers and mailbox fields — the WORD32 named pair
 * ================================================================ */

/**
 * Write `bytes` from host memory to (target, address) one 32-bit store per word.
 *
 * This is the third memory "space", and it is the one that has no bounds to check: a
 * per-Tensix soft-reset register at 0xFFB121B0 is in neither L1 nor a DRAM bank, and a
 * 4-byte mailbox field is under the 16 B minimum ft_tt_write_l1() correctly enforces.
 * Width is not a tuning choice here — a 16 B store over a 4-byte register writes three
 * neighbours with it.
 *
 * ft_tt_launch.h's run control is the caller. It used to reach
 * ft_tt_noc_transfer_raw() through a private helper of its own, which meant the WORD32
 * decision lived in a file that owns neither the width enum nor the copy; this pair is
 * that helper, folded back onto the module that owns both.
 *
 * @param path     an open or attached path
 * @param target   TRANSLATED coordinate from ft_tt_coord.h. NEVER a guess (spec §7.3)
 * @param address  tile-local address, 4-byte aligned
 * @param src      host source
 * @param bytes    multiple of 4. Zero is a legal no-op
 * @return         FT_TT_OK; FT_TT_EINVAL; FT_TT_EPERM when the Phase 1b fence is
 *                 closed; FT_TT_EIO
 *
 * Thread-safe: no (the path is stateful)    Reentrant: no    Signal-safe: no
 */
int ft_tt_write_register_words(ft_tt_noc_path *path, ft_tt_noc_coord target, uint64_t address,
                               const void *src, size_t bytes);

/**
 * Read `bytes` from (target, address) one 32-bit load per word. Same rules as
 * ft_tt_write_register_words().
 */
int ft_tt_read_register_words(ft_tt_noc_path *path, ft_tt_noc_coord target, uint64_t address,
                              void *dst, size_t bytes);

/* ================================================================
 * L1
 * ================================================================ */

/**
 * Write `bytes` from host memory into one Tensix's L1, sliding the window as
 * the span crosses 2 MiB boundaries. (L1 is 1536 KiB so at most one boundary is
 * in play, but the same code path serves DRAM where there are thousands.)
 *
 * @param path     an open path
 * @param core     TRANSLATED coordinate of the target Tensix, from
 *                 ft_tt_coord_tensix_at() / _by_index(). NEVER a guess
 * @param address  L1 byte address, 16 B aligned, with address + bytes <= 1536 KiB
 * @param src      host source
 * @param bytes    multiple of 16. Zero is a legal no-op
 * @return         FT_TT_OK; FT_TT_EINVAL on a bad argument, misalignment, an
 *                 out-of-L1 span, or a coordinate outside the 6-bit TLB fields;
 *                 FT_TT_EPERM when the Phase 1b fence is closed; FT_TT_EIO
 *
 * Thread-safe: no (the path is stateful)    Reentrant: no    Signal-safe: no
 */
int ft_tt_write_l1(ft_tt_noc_path *path, ft_tt_noc_coord core, uint64_t address, const void *src,
                   size_t bytes);

/**
 * Read `bytes` out of one Tensix's L1. Same rules as ft_tt_write_l1(); reads
 * are non-posted and run at ~2.3 MB/s in either cache mode, so this is for
 * verification and diagnostics, not for moving results in bulk.
 */
int ft_tt_read_l1(ft_tt_noc_path *path, ft_tt_noc_coord core, uint64_t address, void *dst,
                  size_t bytes);

/* ================================================================
 * DRAM
 * ================================================================ */

/**
 * Write `bytes` into one GDDR bank, sliding the window across as many 2 MiB
 * spans as the transfer covers.
 *
 * @param path       an open path
 * @param bank_port  TRANSLATED coordinate of a bank port, from
 *                   ft_tt_coord_dram_bank() / _dram_port(). On this card those
 *                   live at x 17-18, y 12-23 — coordinates absent from NoC0
 *                   space entirely, which is why passing a NoC0 coordinate here
 *                   is a silent, expensive mistake
 * @param address    byte offset within the bank, 16 B aligned
 * @param bytes      multiple of 16
 * @return           as ft_tt_write_l1()
 */
int ft_tt_write_dram(ft_tt_noc_path *path, ft_tt_noc_coord bank_port, uint64_t address,
                     const void *src, size_t bytes);

/**
 * Read `bytes` out of one GDDR bank. Address and length must be 64 B aligned —
 * the DRAM/PCIe read alignment, stricter than the 16 B the write direction
 * takes (ft_tt_noc_alignment()).
 */
int ft_tt_read_dram(ft_tt_noc_path *path, ft_tt_noc_coord bank_port, uint64_t address, void *dst,
                    size_t bytes);

#endif /* FT_TT_NOC_H */
