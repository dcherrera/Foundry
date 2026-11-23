/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_tlb_encode.h
 * @brief Encode a Blackhole 2 MiB TLB window's three configuration registers,
 *   and give the BAR0 byte offsets they are written to — the one and only
 *   thing tt-kmd's CONFIGURE_TLB ioctl does to hardware.
 *
 *   This is a PURE FUNCTION over a config struct. It performs no I/O, opens
 *   nothing, and has no dependency beyond <stdint.h>. Everything about it is
 *   provable off-hardware, which is why it is the first piece of the userspace
 *   driver to exist.
 *
 *   The register is 96 bits across three 32-bit words and the fields do not
 *   respect word boundaries — `y_start` (bits 61-66) straddles bit 64 by three
 *   bits. Getting that wrong produces a window aimed at a
 *   plausible-looking wrong endpoint, and a wrong endpoint HANGS THE NoC and
 *   costs a physical power cycle. Hence the reference-vector
 *   gate: `tests/reference/ft_tt_tlb2m_reference_vectors.h` is generated from
 *   the public field table (tt-isa-documentation, BlackholeA0/PCIExpressTile/
 *   HostToDeviceTLBs.md) by setting one bit at a time — a different method from
 *   this file's per-word shifts and masks.
 *
 * @status Gated against 96 generated vectors off-hardware, and — since
 *   2026-08-16 — against the REGISTERS TT-KMD ITSELF WROTE on the live p150a:
 *   `tests/test_udrv_card_tlb.c` aims a window through CONFIGURE_TLB and
 *   byte-compares the three words this encoder produces against the three the
 *   driver left in BAR0. The decoder below is what the read-only rung uses to
 *   confirm window 201 points where tt-kmd is demonstrably talking.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_UDRV_TLB_ENCODE_H
#define FT_TT_UDRV_TLB_ENCODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------------ *
 * Geometry — every constant here is tt-kmd's, named the same, so a re-vendor
 * is a diff. Source: the kernel driver's Blackhole TLB geometry.
 * ------------------------------------------------------------------------ */

/** Number of 2 MiB windows the part has. tt-kmd claims the last one
 *  (index 201, `KERNEL_TLB_INDEX`) for its own NoC reads; a userspace driver
 *  regains it, so all 202 are usable. */
#define FT_TT_UDRV_TLB_2M_WINDOW_COUNT 202u

/** log2 of the window size: a window's address field is `addr >> 21`. */
#define FT_TT_UDRV_TLB_2M_SHIFT 21u

/** 2 MiB. A window's aperture in BAR0 is `id * FT_TT_UDRV_TLB_2M_WINDOW_SIZE`. */
#define FT_TT_UDRV_TLB_2M_WINDOW_SIZE (1u << FT_TT_UDRV_TLB_2M_SHIFT)

/** Bytes per window configuration register triple. */
#define FT_TT_UDRV_TLB_REG_SIZE 12u

/** Byte offset of the TLB register file within BAR0. */
#define FT_TT_UDRV_TLB_REGS_START 0x1FC00000u

/** Extent of the TLB register file within BAR0. */
#define FT_TT_UDRV_TLB_REGS_LEN 0x00001000u

/** Windows below this index also carry a strided-multicast register, which
 *  CONFIGURE_TLB zeroes because the ioctl cannot express a strided pattern. */
#define FT_TT_UDRV_TLB_STRIDED_COUNT 32u

/** Bytes per strided register. */
#define FT_TT_UDRV_TLB_STRIDED_REG_SIZE 4u

/** Offset of the strided register block within the TLB register file. It is
 *  sized from the TOTAL window count (2 MiB + 4 GiB kinds = 202 + 8 = 210),
 *  NOT from the 2 MiB count — so it is 210 * 12 = 2520 = 0x9D8 even on a board
 *  like this one whose BAR4 is absent and which therefore has no 4 GiB
 *  windows at all. */
#define FT_TT_UDRV_TLB_STRIDED_REGS_OFFSET ((202u + 8u) * FT_TT_UDRV_TLB_REG_SIZE)

/** Ordering modes, `ordering` field. POSTED_STRICT is unicast-only. */
enum ft_tt_udrv_tlb_ordering {
    FT_TT_UDRV_TLB_ORDERING_RELAXED = 0,
    FT_TT_UDRV_TLB_ORDERING_STRICT = 1,
    FT_TT_UDRV_TLB_ORDERING_POSTED = 2,
    FT_TT_UDRV_TLB_ORDERING_POSTED_STRICT = 3
};

/**
 * Where a 2 MiB window points and how it behaves.
 *
 * Field widths are the REGISTER's, not this struct's: `x_end` and friends are
 * 6 bits, `noc` and `ordering` are 2, `static_vc_id` is 3. A value that does
 * not fit is a caller error — `ft_tt_udrv_tlb2m_config_check()` names it, and
 * `ft_tt_udrv_tlb2m_encode()` TRUNCATES it exactly as the kernel's bitfield
 * assignment does, so the two never disagree about what the hardware sees.
 */
typedef struct ft_tt_udrv_tlb2m_config {
    uint64_t addr;         /**< NoC byte address; must be 2 MiB aligned */
    uint8_t x_end;         /**< 6 bits — X for unicast, rectangle end for mcast */
    uint8_t y_end;         /**< 6 bits */
    uint8_t x_start;       /**< 6 bits — 0 for unicast */
    uint8_t y_start;       /**< 6 bits — 0 for unicast; STRADDLES word 2/3 */
    uint8_t noc;           /**< 2 bits — 0 or 1 */
    uint8_t multicast;     /**< 1 bit */
    uint8_t ordering;      /**< 2 bits — enum ft_tt_udrv_tlb_ordering */
    uint8_t linked;        /**< 1 bit */
    uint8_t use_static_vc; /**< 1 bit — CONFIGURE_TLB feeds its `static_vc` here */
    uint8_t stream_header; /**< 1 bit — the ioctl cannot set this; we can */
    uint8_t static_vc_id;  /**< 3 bits — likewise */
} ft_tt_udrv_tlb2m_config;

/**
 * Encode a config into the three 32-bit words, in the order they are written:
 * `out_words[0]` to `+0`, `[1]` to `+4`, `[2]` to `+8`.
 *
 * Out-of-range field values are truncated to their register widths, matching
 * the kernel's bitfield assignment bit for bit. Use
 * ft_tt_udrv_tlb2m_config_check() first if you want them refused instead.
 *
 * @param config     what to aim at; `addr` must be 2 MiB aligned
 * @param out_words  receives low32, mid32, high32
 * @return           FT_TT_UDRV_OK, or FT_TT_UDRV_EINVAL for a NULL argument or
 *                   a misaligned address (the same refusal the kernel makes)
 *
 * Thread-safe: yes (pure)   Reentrant: yes   Signal-safe: yes
 */
int ft_tt_udrv_tlb2m_encode(const ft_tt_udrv_tlb2m_config *config, uint32_t out_words[3]);

/**
 * Recover a config from the three 32-bit words — the inverse of
 * ft_tt_udrv_tlb2m_encode().
 *
 * This exists because the FIRST thing the userspace driver does on silicon is
 * READ a window it did not write: tt-kmd keeps window 201 for its own NoC
 * reads and aims it at the ARC tile, so decoding that window's registers and
 * finding (8,0) is a zero-risk proof that the BAR mapping, the register
 * offsets and the bit layout are all right.
 *
 * The 18 reserved bits above `static_vc` are NOT returned; a caller that cares
 * whether they were set must look at the raw words. Everything else is exact,
 * so `decode(encode(c)) == c` for any config that passes
 * ft_tt_udrv_tlb2m_config_check().
 *
 * @param words       low32, mid32, high32, in the order they appear in BAR0
 * @param out_config  receives the decoded fields; `addr` is the field shifted
 *                    back up, so it is always 2 MiB aligned
 * @return            FT_TT_UDRV_OK, or FT_TT_UDRV_EINVAL for a NULL argument
 *
 * Thread-safe: yes (pure)   Reentrant: yes   Signal-safe: yes
 */
int ft_tt_udrv_tlb2m_decode(const uint32_t words[3], ft_tt_udrv_tlb2m_config *out_config);

/**
 * Whether two configs describe the same aim, field by field.
 *
 * Not `memcmp`: the struct has padding a compiler may leave uninitialised, and
 * a gate that reads padding is a gate that fails at random. For comparing what
 * hardware actually holds, compare the ENCODED WORDS instead — that is a byte
 * comparison of the thing the register really is.
 *
 * @return true when every field matches.
 */
bool ft_tt_udrv_tlb2m_config_equal(const ft_tt_udrv_tlb2m_config *left,
                                   const ft_tt_udrv_tlb2m_config *right);

/**
 * Refuse a config whose fields do not fit their register widths, or whose
 * address is misaligned. Separate from the encoder on purpose: the encoder has
 * to be bit-compatible with the kernel (which truncates silently), while a
 * caller that would rather fail loudly needs somewhere to ask.
 *
 * @return FT_TT_UDRV_OK or FT_TT_UDRV_EINVAL.
 */
int ft_tt_udrv_tlb2m_config_check(const ft_tt_udrv_tlb2m_config *config);

/**
 * Fill a config the way CONFIGURE_TLB does for a plain unicast window: every
 * multicast/rectangle field zero, `x_end`/`y_end` the target, and
 * `use_static_vc` from the caller's `static_vc` flag. This is the exact field
 * mapping of blackhole_configure_tlb_2M(), so a window aimed through it is
 * indistinguishable from one aimed through the ioctl.
 *
 * @return FT_TT_UDRV_OK or FT_TT_UDRV_EINVAL.
 */
int ft_tt_udrv_tlb2m_config_unicast(ft_tt_udrv_tlb2m_config *out_config, uint8_t x, uint8_t y,
                                    uint64_t addr, uint8_t noc, uint8_t ordering);

/**
 * BAR0 byte offset of window `id`'s three configuration registers.
 *
 * @param id          window index, < FT_TT_UDRV_TLB_2M_WINDOW_COUNT
 * @param out_offset  receives FT_TT_UDRV_TLB_REGS_START + id * 12
 * @return            FT_TT_UDRV_OK or FT_TT_UDRV_EINVAL
 */
int ft_tt_udrv_tlb2m_register_offset(uint32_t id, uint32_t *out_offset);

/**
 * BAR0 byte offset of window `id`'s strided-multicast register, which
 * CONFIGURE_TLB zeroes after every aim. Only windows below
 * FT_TT_UDRV_TLB_STRIDED_COUNT have one; for the rest this returns
 * FT_TT_UDRV_ENOTSUP and there is nothing to write.
 *
 * A freshly claimed window is NOT blank (DEV_NOTES §1: 2 of 202 held live
 * config when the pool was last inspected), so this zeroing write is part of
 * aiming, not an optimisation.
 *
 * @return FT_TT_UDRV_OK, FT_TT_UDRV_EINVAL (bad id / NULL), or
 *         FT_TT_UDRV_ENOTSUP (id has no strided register).
 */
int ft_tt_udrv_tlb2m_strided_register_offset(uint32_t id, uint32_t *out_offset);

/**
 * BAR0 byte offset of window `id`'s 2 MiB aperture. This is also the `offset`
 * argument to mmap(2) on `/sys/.../resource0_wc`, because tt-kmd's own
 * map_tlb_window() is `io_remap_pfn_range(bar_start + id * 2 MiB)` with
 * pgprot_writecombine — the same page, the same caching, one fewer ioctl.
 *
 * @return FT_TT_UDRV_OK or FT_TT_UDRV_EINVAL
 */
int ft_tt_udrv_tlb2m_aperture_offset(uint32_t id, uint64_t *out_offset);

#endif /* FT_TT_UDRV_TLB_ENCODE_H */
