/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_noc_copy.h
 * @brief The PIO copy shape and the window-walk arithmetic, with no dependency
 *   on how the window was aimed — so the tt-kmd path (`ft_tt_noc.c`) and the
 *   userspace-driver path (`ft_tt_udrv_noc.c`) move bytes through exactly the
 *   same code instead of two implementations that drift.
 *
 *   Header-only and `static inline` on purpose. These are four short functions
 *   in the hot path, and a separate translation unit would have to be added to
 *   every Makefile that already links `ft_tt_noc.c` — sixteen of them — for no
 *   gain. Nothing here includes a project header, opens anything, or logs, so
 *   it can be compiled into a suite whose whole claim is that it cannot reach a
 *   device.
 *
 * ---------------------------------------------------------------------------
 * WHY THE COPY IS NOT A memcpy
 * ---------------------------------------------------------------------------
 * The aperture is device MMIO. tt-umd moves 16 x 16 B per 256 B block because
 * that is what keeps the PCIe transaction count where it wants it — but their
 * device_memcpy.cpp is AVX2 and the card host here has NO AVX at all, so their
 * binary SIGILLs on it exactly like tt_umd did. We keep the 16 B unit and the
 * 256 B block and issue them as SSE2 `movdqu`, which is baseline x86-64. On any
 * non-x86 host (the Mac, for compile checks) it degrades to a 64-bit store loop
 * of the same shape.
 *
 * ---------------------------------------------------------------------------
 * THE STORE FENCE IS A SEPARATE CALL, AND THAT IS DELIBERATE
 * ---------------------------------------------------------------------------
 * `ft_tt_noc_copy_slice()` ends in a COMPILER barrier, which stops the
 * optimiser hoisting, merging or eliding stores to a region it can see no other
 * reader of. It is NOT a CPU fence, and through a write-combined mapping those
 * are different things: WC stores sit in the core's write-combining buffers and
 * are flushed on their own schedule. `ft_tt_noc_copy_store_fence()` is the
 * `sfence` that retires them, and it is a separate call so that adding it to a
 * new path does not silently change the transaction shape of an old one.
 *
 * @status The copy shape is unchanged from ft_tt_noc.c, where it was verified
 *   on the card and the disassembly checked on the
 *   bank for VEX instructions. The extraction is mechanical; the store fence is
 *   the only new code and it is called by the userspace-driver write path only.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_NOC_COPY_H
#define FT_TT_NOC_COPY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

/* ================================================================
 * Copy shape
 * ================================================================ */

/** One store unit: 16 B, an SSE2 `movdqu`. tt-umd uses 32 B (AVX2), which the
 *  card host cannot execute. */
#define FT_TT_NOC_COPY_UNIT_BYTES 16u

/** 16 units per block, i.e. tt-umd's 256 B blocking, preserved so the PCIe
 *  transaction shape matches theirs even though the instruction does not. */
#define FT_TT_NOC_COPY_BLOCK_BYTES 256u

/**
 * The store width a transfer is issued in. Not a tuning knob — the two are for
 * different KINDS of device access and are not interchangeable:
 *
 *  - BLOCK16 is bulk data. 16 B units in 256 B blocks, the shape above.
 *  - WORD32 is one 32-bit store or load per word, which is the only legal way
 *    to touch a 4-byte device REGISTER (the per-Tensix soft-reset register at
 *    0xFFB121B0) or a sub-16 B mailbox field. A 16 B store over a 4 B register
 *    writes three neighbours with it.
 */
typedef enum ft_tt_noc_transfer_width {
    FT_TT_NOC_WIDTH_BLOCK16 = 0, /**< 16 B units, 256 B blocks — bulk data */
    FT_TT_NOC_WIDTH_WORD32 = 1   /**< one 32-bit access per word — registers, mailbox */
} ft_tt_noc_transfer_width;

/**
 * Bytes per access for a width: 16 for BLOCK16, 4 for WORD32, 0 for an unknown
 * width. Also the alignment both the device address and the length must satisfy.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
static inline uint32_t ft_tt_noc_transfer_width_bytes(ft_tt_noc_transfer_width width)
{
    switch (width) {
    case FT_TT_NOC_WIDTH_BLOCK16:
        return FT_TT_NOC_COPY_UNIT_BYTES;
    case FT_TT_NOC_WIDTH_WORD32:
        return 4u;
    default:
        return 0u;
    }
}

/* ================================================================
 * The window walk, as arithmetic
 * ================================================================ */

/**
 * How much of a transfer fits in one window placement, and where.
 *
 * The walk is the part of the transport that can silently corrupt a
 * neighbouring address rather than fail, so it is separated out as pure
 * arithmetic a unit test on the Mac can prove wrong — a wrong endpoint hangs
 * the chip, and a wrong OFFSET quietly writes somewhere else on the right one.
 */
typedef struct ft_tt_noc_window_slice {
    uint64_t window_base;    /**< what the window must be aimed at for this slice */
    size_t offset_in_window; /**< where the slice starts inside the aperture */
    size_t bytes;            /**< how much of the transfer this slice covers */
} ft_tt_noc_window_slice;

/**
 * First slice of a transfer of `bytes` at `address` through a `window_bytes`
 * aperture: the window base it needs, the offset inside that window, and how
 * many bytes fit before the next window boundary. Walk a whole transfer by
 * advancing address/host/remaining by `bytes` and calling again.
 *
 * The pure form: no logging, no project status codes, so it can live in a
 * dependency-free header. `ft_tt_noc_window_slice_at()` is this plus a log line
 * and an FT_TT_* return.
 *
 * @param window_bytes  aperture size; must be a non-zero power of two
 * @param out_slice     written only on success
 * @return              true on success; false for a NULL out or a window size
 *                      that is not a power of two
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
static inline bool ft_tt_noc_window_slice_compute(uint64_t address, size_t bytes,
                                                  size_t window_bytes,
                                                  ft_tt_noc_window_slice *out_slice)
{
    uint64_t window_base;
    size_t offset_in_window;
    size_t chunk;

    if (!out_slice)
        return false;
    if (window_bytes == 0u || (window_bytes & (window_bytes - 1u)) != 0u)
        return false;

    window_base = address & ~((uint64_t)window_bytes - 1u);
    offset_in_window = (size_t)(address - window_base);
    chunk = window_bytes - offset_in_window;
    if (chunk > bytes)
        chunk = bytes;

    out_slice->window_base = window_base;
    out_slice->offset_in_window = offset_in_window;
    out_slice->bytes = chunk;
    return true;
}

/* ================================================================
 * The PIO copy
 * ================================================================ */

/*
 * Both directions move FT_TT_NOC_COPY_UNIT_BYTES at a time, grouped into
 * FT_TT_NOC_COPY_BLOCK_BYTES blocks, with a compiler barrier between blocks.
 * The barrier is what stops the optimiser hoisting, merging or eliding stores
 * to a region it cannot see any other reader of — the aperture is MMIO, and
 * every store is a PCIe transaction whose ORDER and COUNT are the thing being
 * engineered here.
 */
#define FT_TT_NOC_COPY_BARRIER() __asm__ __volatile__("" : : : "memory")

/**
 * One 16 B unit. Identical in both directions: on x86 it is a `movdqu` load and
 * a `movdqu` store, and which side is the aperture only decides which of the
 * two is the PCIe transaction.
 */
static inline void ft_tt_noc_copy_unit(unsigned char *dst, const unsigned char *src)
{
#if defined(__SSE2__)
    _mm_storeu_si128((__m128i *)(void *)dst, _mm_loadu_si128((const __m128i *)(const void *)src));
#else
    /* Non-x86 (the Mac, for compile checks): same 16 B unit, two 64-bit moves.
     * The card host is always x86 with SSE2, so this path never runs on it. */
    uint64_t low;
    uint64_t high;

    memcpy(&low, src, sizeof(low));
    memcpy(&high, src + sizeof(low), sizeof(high));
    memcpy(dst, &low, sizeof(low));
    memcpy(dst + sizeof(low), &high, sizeof(high));
#endif
}

/**
 * Copy `bytes` between the aperture and host memory in tt-umd's 16 x 16 B per
 * 256 B shape. `bytes` must be a multiple of FT_TT_NOC_COPY_UNIT_BYTES.
 */
static inline void ft_tt_noc_copy_blocked(unsigned char *dst, const unsigned char *src,
                                          size_t bytes)
{
    const size_t units_per_block = FT_TT_NOC_COPY_BLOCK_BYTES / FT_TT_NOC_COPY_UNIT_BYTES;
    size_t done = 0;

    while (bytes - done >= FT_TT_NOC_COPY_BLOCK_BYTES) {
        size_t unit;

        for (unit = 0; unit < units_per_block; unit++) {
            size_t at = done + unit * FT_TT_NOC_COPY_UNIT_BYTES;

            ft_tt_noc_copy_unit(dst + at, src + at);
        }
        FT_TT_NOC_COPY_BARRIER();
        done += FT_TT_NOC_COPY_BLOCK_BYTES;
    }

    while (done < bytes) {
        ft_tt_noc_copy_unit(dst + done, src + done);
        done += FT_TT_NOC_COPY_UNIT_BYTES;
    }
    FT_TT_NOC_COPY_BARRIER();
}

/**
 * The WORD32 copy: exactly one 32-bit device access per word, which is what a
 * 4-byte register and a 4-byte mailbox field require. `volatile` here is not
 * belt and braces — it is what stops the compiler merging two adjacent word
 * stores into one wider one, which over a register file is a different
 * transaction entirely.
 *
 * The host side goes through memcpy per word because a caller's buffer (a
 * packed go message, say) need not be 4-byte aligned, and a misaligned
 * uint32_t load is undefined.
 */
static inline void ft_tt_noc_copy_words(volatile uint32_t *aperture_words,
                                        unsigned char *host_bytes, size_t word_count,
                                        bool to_device)
{
    size_t index;

    for (index = 0; index < word_count; index++) {
        uint32_t word;

        if (to_device) {
            memcpy(&word, host_bytes + index * 4u, sizeof(word));
            aperture_words[index] = word;
        } else {
            word = aperture_words[index];
            memcpy(host_bytes + index * 4u, &word, sizeof(word));
        }
    }
}

/** Move one slice between the aperture and host memory in the shape `width` names. */
static inline void ft_tt_noc_copy_slice(volatile unsigned char *aperture,
                                        unsigned char *host_bytes, size_t bytes, bool is_read,
                                        ft_tt_noc_transfer_width width)
{
    if (width == FT_TT_NOC_WIDTH_WORD32) {
        ft_tt_noc_copy_words((volatile uint32_t *)aperture, host_bytes, bytes / 4u, !is_read);
        return;
    }

    /* The aperture is cast to a plain pointer for the blocked copy; ordering and
     * non-elision come from the barrier in ft_tt_noc_copy_blocked(), not from
     * `volatile`, because the 16 B unit is one instruction and volatile would
     * force it back to scalar. */
    {
        unsigned char *window_bytes_ptr = (unsigned char *)(void *)aperture;

        if (is_read)
            ft_tt_noc_copy_blocked(host_bytes, window_bytes_ptr, bytes);
        else
            ft_tt_noc_copy_blocked(window_bytes_ptr, host_bytes, bytes);
    }
}

/**
 * Retire write-combining buffers, so a store issued through a WC mapping has
 * left the CPU.
 *
 * A compiler barrier does not do this and neither does `volatile`: through a
 * write-combined mapping the core is allowed to hold and merge stores in its WC
 * buffers indefinitely from the program's point of view. `sfence` is the
 * architectural retirement point on x86; elsewhere a full barrier stands in.
 *
 * Call it after the last store of a write and before anything that reads the
 * result back — including a read through a different mapping, which will
 * otherwise race the buffer rather than the card.
 *
 * MEASURED, AND SAID PLAINLY RATHER THAN OVERSOLD: on the card host a WC
 * store to the card lands WITHOUT this call. The mutation that removed every
 * fence from `test_udrv_card_noc.c` and the transport ran the whole on-card
 * suite green (2026-08-17). So the fence is not observably load-bearing on
 * THIS CPU with THESE access patterns. It is kept because it costs nothing and
 * because "the buffer happened to drain in time" is a timing property of one
 * host, not a guarantee — and the thing being built is meant to run on a
 * machine nobody here has.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
static inline void ft_tt_noc_copy_store_fence(void)
{
#if defined(__SSE2__)
    _mm_sfence();
#else
    __sync_synchronize();
#endif
    FT_TT_NOC_COPY_BARRIER();
}

#endif /* FT_TT_NOC_COPY_H */
