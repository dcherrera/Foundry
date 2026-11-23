/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_tlb_encode.c
 * @brief Implementation of the Blackhole 2 MiB TLB window register encoder.
 *
 * @status Off-hardware only.
 * @issues None
 * @todo None
 */

#include "ft_tt_udrv_tlb_encode.h"

#include <stdint.h>

#include "ft_tt_udrv.h"

/* ------------------------------------------------------------------------ *
 * The register layout.
 *
 * tt-kmd expresses this as a packed bitfield struct over a 12-byte union. Bitfields in a packed little-endian struct allocate
 * from the least significant bit upward and straddle byte and word boundaries
 * freely, so the layout below IS that struct, written as offsets.
 *
 * Widths are derived by accumulation so that inserting a field cannot leave a
 * stale offset behind. The literal PINS immediately after are deliberately NOT
 * derived: a derived pin re-derives whatever mistake the accumulation made and
 * agrees with itself (DEV_NOTES §3, "a struct assert that proves
 * SELF-CONSISTENCY does not prove the struct has not MOVED").
 * ------------------------------------------------------------------------ */

#define W_ADDRESS 43u
#define W_X_END 6u
#define W_Y_END 6u
#define W_X_START 6u
#define W_Y_START 6u
#define W_NOC 2u
#define W_MULTICAST 1u
#define W_ORDERING 2u
#define W_LINKED 1u
#define W_USE_STATIC_VC 1u
#define W_STREAM_HEADER 1u
#define W_STATIC_VC 3u

#define B_ADDRESS 0u
#define B_X_END (B_ADDRESS + W_ADDRESS)
#define B_Y_END (B_X_END + W_X_END)
#define B_X_START (B_Y_END + W_Y_END)
#define B_Y_START (B_X_START + W_X_START)
#define B_NOC (B_Y_START + W_Y_START)
#define B_MULTICAST (B_NOC + W_NOC)
#define B_ORDERING (B_MULTICAST + W_MULTICAST)
#define B_LINKED (B_ORDERING + W_ORDERING)
#define B_USE_STATIC_VC (B_LINKED + W_LINKED)
#define B_STREAM_HEADER (B_USE_STATIC_VC + W_USE_STATIC_VC)
#define B_STATIC_VC (B_STREAM_HEADER + W_STREAM_HEADER)

/* Literal pins. Read these off the TLB field list in ft_tt_udrv_tlb_encode.h by hand
 * and change them only when that list changes. */
FT_TT_UDRV_STATIC_ASSERT(B_ADDRESS == 0, address_at_bit_0);
FT_TT_UDRV_STATIC_ASSERT(B_X_END == 43, x_end_at_bit_43);
FT_TT_UDRV_STATIC_ASSERT(B_Y_END == 49, y_end_at_bit_49);
FT_TT_UDRV_STATIC_ASSERT(B_X_START == 55, x_start_at_bit_55);
FT_TT_UDRV_STATIC_ASSERT(B_Y_START == 61, y_start_at_bit_61);
FT_TT_UDRV_STATIC_ASSERT(B_NOC == 67, noc_at_bit_67);
FT_TT_UDRV_STATIC_ASSERT(B_MULTICAST == 69, multicast_at_bit_69);
FT_TT_UDRV_STATIC_ASSERT(B_ORDERING == 70, ordering_at_bit_70);
FT_TT_UDRV_STATIC_ASSERT(B_LINKED == 72, linked_at_bit_72);
FT_TT_UDRV_STATIC_ASSERT(B_USE_STATIC_VC == 73, use_static_vc_at_bit_73);
FT_TT_UDRV_STATIC_ASSERT(B_STREAM_HEADER == 74, stream_header_at_bit_74);
FT_TT_UDRV_STATIC_ASSERT(B_STATIC_VC == 75, static_vc_at_bit_75);
FT_TT_UDRV_STATIC_ASSERT(B_STATIC_VC + W_STATIC_VC == 78, eighteen_reserved_bits);
/* y_start MUST straddle the mid32/high32 boundary — that is the whole reason
 * this file has a reference-vector gate rather than a hand-written table. */
FT_TT_UDRV_STATIC_ASSERT(B_Y_START < 64 && (B_Y_START + W_Y_START) > 64, y_start_straddles);

/** The 2 MiB alignment the window's address field cannot express. */
#define TLB_2M_WINDOW_MASK ((uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE - 1u)

/**
 * Deposit `value` at bit `bit` of a 96-bit register held as a low 64 bits and
 * a high 32 bits, truncating to `width`.
 *
 * The straddling case is the whole point: a field that starts below bit 64 and
 * ends above it contributes its low part to `lo` (where the shift discards the
 * overflow for us) and its high part to `hi`.
 */
static void put_field(uint64_t *lo, uint64_t *hi, unsigned bit, unsigned width, uint64_t value)
{
    uint64_t mask = (width >= 64u) ? UINT64_MAX : ((UINT64_C(1) << width) - 1u);

    value &= mask;

    if (bit >= 64u) {
        *hi |= value << (bit - 64u);
        return;
    }

    *lo |= value << bit;

    if (bit + width > 64u)
        *hi |= value >> (64u - bit);
}

/**
 * Extract the `width`-bit field at bit `bit` from a 96-bit register held as a
 * low 64 bits and a high 32 bits.
 *
 * Written as its own traversal rather than as put_field() run backwards: the
 * whole value of the decoder is that it is a SECOND reading of the layout, so
 * a shift that is wrong in one direction has a chance of being caught rather
 * than cancelling out. The literal bit pins above are shared, deliberately —
 * they are pinned against the public field table (HostToDeviceTLBs.md), not
 * against this file.
 */
static uint64_t get_field(uint64_t lo, uint64_t hi, unsigned bit, unsigned width)
{
    uint64_t mask = (width >= 64u) ? UINT64_MAX : ((UINT64_C(1) << width) - 1u);
    uint64_t value;

    if (bit >= 64u)
        return (hi >> (bit - 64u)) & mask;

    value = lo >> bit;

    /* The straddle. `bit` cannot be 0 here — reaching this branch needs
     * bit + width > 64 with width <= 64 — so the shift is always < 64. */
    if (bit + width > 64u)
        value |= hi << (64u - bit);

    return value & mask;
}

/** True when `value` fits in `width` bits. */
static bool fits(uint64_t value, unsigned width)
{
    uint64_t mask = (width >= 64u) ? UINT64_MAX : ((UINT64_C(1) << width) - 1u);
    return (value & ~mask) == 0u;
}

int ft_tt_udrv_tlb2m_encode(const ft_tt_udrv_tlb2m_config *config, uint32_t out_words[3])
{
    uint64_t lo = 0;
    uint64_t hi = 0;

    if (!config || !out_words)
        return FT_TT_UDRV_EINVAL;

    /* The kernel's own refusal: a 2 MiB window cannot start off a 2 MiB
     * boundary, because the register holds addr >> 21. */
    if ((config->addr & TLB_2M_WINDOW_MASK) != 0u)
        return FT_TT_UDRV_EINVAL;

    put_field(&lo, &hi, B_ADDRESS, W_ADDRESS, config->addr >> FT_TT_UDRV_TLB_2M_SHIFT);
    put_field(&lo, &hi, B_X_END, W_X_END, config->x_end);
    put_field(&lo, &hi, B_Y_END, W_Y_END, config->y_end);
    put_field(&lo, &hi, B_X_START, W_X_START, config->x_start);
    put_field(&lo, &hi, B_Y_START, W_Y_START, config->y_start);
    put_field(&lo, &hi, B_NOC, W_NOC, config->noc);
    put_field(&lo, &hi, B_MULTICAST, W_MULTICAST, config->multicast);
    put_field(&lo, &hi, B_ORDERING, W_ORDERING, config->ordering);
    put_field(&lo, &hi, B_LINKED, W_LINKED, config->linked);
    put_field(&lo, &hi, B_USE_STATIC_VC, W_USE_STATIC_VC, config->use_static_vc);
    put_field(&lo, &hi, B_STREAM_HEADER, W_STREAM_HEADER, config->stream_header);
    put_field(&lo, &hi, B_STATIC_VC, W_STATIC_VC, config->static_vc_id);

    out_words[0] = (uint32_t)(lo & 0xFFFFFFFFu);
    out_words[1] = (uint32_t)(lo >> 32);
    out_words[2] = (uint32_t)(hi & 0xFFFFFFFFu);

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_tlb2m_decode(const uint32_t words[3], ft_tt_udrv_tlb2m_config *out_config)
{
    uint64_t lo;
    uint64_t hi;

    if (!words || !out_config)
        return FT_TT_UDRV_EINVAL;

    lo = (uint64_t)words[0] | ((uint64_t)words[1] << 32);
    hi = (uint64_t)words[2];

    out_config->addr = get_field(lo, hi, B_ADDRESS, W_ADDRESS) << FT_TT_UDRV_TLB_2M_SHIFT;
    out_config->x_end = (uint8_t)get_field(lo, hi, B_X_END, W_X_END);
    out_config->y_end = (uint8_t)get_field(lo, hi, B_Y_END, W_Y_END);
    out_config->x_start = (uint8_t)get_field(lo, hi, B_X_START, W_X_START);
    out_config->y_start = (uint8_t)get_field(lo, hi, B_Y_START, W_Y_START);
    out_config->noc = (uint8_t)get_field(lo, hi, B_NOC, W_NOC);
    out_config->multicast = (uint8_t)get_field(lo, hi, B_MULTICAST, W_MULTICAST);
    out_config->ordering = (uint8_t)get_field(lo, hi, B_ORDERING, W_ORDERING);
    out_config->linked = (uint8_t)get_field(lo, hi, B_LINKED, W_LINKED);
    out_config->use_static_vc = (uint8_t)get_field(lo, hi, B_USE_STATIC_VC, W_USE_STATIC_VC);
    out_config->stream_header = (uint8_t)get_field(lo, hi, B_STREAM_HEADER, W_STREAM_HEADER);
    out_config->static_vc_id = (uint8_t)get_field(lo, hi, B_STATIC_VC, W_STATIC_VC);

    return FT_TT_UDRV_OK;
}

bool ft_tt_udrv_tlb2m_config_equal(const ft_tt_udrv_tlb2m_config *left,
                                   const ft_tt_udrv_tlb2m_config *right)
{
    if (!left || !right)
        return false;

    return left->addr == right->addr && left->x_end == right->x_end &&
           left->y_end == right->y_end && left->x_start == right->x_start &&
           left->y_start == right->y_start && left->noc == right->noc &&
           left->multicast == right->multicast && left->ordering == right->ordering &&
           left->linked == right->linked && left->use_static_vc == right->use_static_vc &&
           left->stream_header == right->stream_header &&
           left->static_vc_id == right->static_vc_id;
}

int ft_tt_udrv_tlb2m_config_check(const ft_tt_udrv_tlb2m_config *config)
{
    if (!config)
        return FT_TT_UDRV_EINVAL;

    if ((config->addr & TLB_2M_WINDOW_MASK) != 0u)
        return FT_TT_UDRV_EINVAL;

    if (!fits(config->x_end, W_X_END) || !fits(config->y_end, W_Y_END))
        return FT_TT_UDRV_EINVAL;

    if (!fits(config->x_start, W_X_START) || !fits(config->y_start, W_Y_START))
        return FT_TT_UDRV_EINVAL;

    if (!fits(config->noc, W_NOC) || !fits(config->ordering, W_ORDERING))
        return FT_TT_UDRV_EINVAL;

    if (!fits(config->multicast, W_MULTICAST) || !fits(config->linked, W_LINKED))
        return FT_TT_UDRV_EINVAL;

    if (!fits(config->use_static_vc, W_USE_STATIC_VC) ||
        !fits(config->stream_header, W_STREAM_HEADER))
        return FT_TT_UDRV_EINVAL;

    if (!fits(config->static_vc_id, W_STATIC_VC))
        return FT_TT_UDRV_EINVAL;

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_tlb2m_config_unicast(ft_tt_udrv_tlb2m_config *out_config, uint8_t x, uint8_t y,
                                    uint64_t addr, uint8_t noc, uint8_t ordering)
{
    ft_tt_udrv_tlb2m_config config = {0};

    if (!out_config)
        return FT_TT_UDRV_EINVAL;

    config.addr = addr;
    config.x_end = x;
    config.y_end = y;
    config.noc = noc;
    config.ordering = ordering;

    if (ft_tt_udrv_tlb2m_config_check(&config) != FT_TT_UDRV_OK)
        return FT_TT_UDRV_EINVAL;

    *out_config = config;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_tlb2m_register_offset(uint32_t id, uint32_t *out_offset)
{
    if (!out_offset || id >= FT_TT_UDRV_TLB_2M_WINDOW_COUNT)
        return FT_TT_UDRV_EINVAL;

    *out_offset = FT_TT_UDRV_TLB_REGS_START + (id * FT_TT_UDRV_TLB_REG_SIZE);
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_tlb2m_strided_register_offset(uint32_t id, uint32_t *out_offset)
{
    if (!out_offset || id >= FT_TT_UDRV_TLB_2M_WINDOW_COUNT)
        return FT_TT_UDRV_EINVAL;

    if (id >= FT_TT_UDRV_TLB_STRIDED_COUNT)
        return FT_TT_UDRV_ENOTSUP;

    *out_offset = FT_TT_UDRV_TLB_REGS_START + FT_TT_UDRV_TLB_STRIDED_REGS_OFFSET +
                  (id * FT_TT_UDRV_TLB_STRIDED_REG_SIZE);
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_tlb2m_aperture_offset(uint32_t id, uint64_t *out_offset)
{
    if (!out_offset || id >= FT_TT_UDRV_TLB_2M_WINDOW_COUNT)
        return FT_TT_UDRV_EINVAL;

    *out_offset = (uint64_t)id * (uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE;
    return FT_TT_UDRV_OK;
}
