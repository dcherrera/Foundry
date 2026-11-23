/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_moe_route.h
 * @brief The MoE ROUTING step on fp32 BIT PATTERNS, integer-only: top-k of the router logits
 *   with the CPU's tie rule, the renormalised expert weights and the shared-expert gate
 *   (qwen36_blackhole_spec.md §4.3 step 2). The router writer (resident_matmul_writer.c built
 *   with -DFT_RTRIAD_ROUTER=1) runs it on the card; the host runs THIS header on the card's
 *   own logits, so the card's routing slot is checked bit for bit, not to a tolerance.
 *
 * SEMANTICS (Pantheon CPU qwen35moe_moe.c, the oracle):
 *   - order: a strictly greater logit ranks first; an EQUAL logit never displaces an earlier
 *     index, so ties go to the LOWEST index and ranks are in descending order (the CPU's
 *     selection sort over softmax probabilities, which are monotone in the logits).
 *   - weights: e_k = exp(l_k - l_max) over the selected k, w_k = e_k * (1 / sum e), the sum
 *     taken in rank order. Mathematically the CPU's softmax-then-renormalise; numerically
 *     ft_det_math.h's (exp to ~2^-29, one RNE), the same on the card and the host.
 *   - the shared-expert gate: sigmoid(l_shexp).
 *   - NaN logits are never selected.
 *
 * Red arms (bits of `red`): FTMR_RED_TIE_FLIP (an equal logit DOES displace: ties go to the
 * highest index), FTMR_RED_UNNORM (w_k = e_k), FTMR_RED_RANK_PERMUTE (ranks written in
 * reverse). Each must make the downstream answer DIFFER.
 */
#ifndef FT_MOE_ROUTE_H
#define FT_MOE_ROUTE_H

#include <stdint.h>

#include "ft_det_math.h"

#if defined(__riscv)
/**
 * libgcc's 64-bit logical shift, which ft_det_math.h's REFERENCE exp core needs on rv32 (its
 * header: a variable 64-bit shift, __lshrdi3 under gcc -Os) and which the Tensix kernel link
 * does not carry. Two 32-bit halves, no recursion, no libc.
 */
long long __lshrdi3(long long value, int shift);
/* `used` + `externally_visible`: the kernel link is LTO, and the call to this libcall is only
 * created at code generation, after LTO has already dropped an unreferenced definition. */
__attribute__((used, externally_visible, noinline)) long long __lshrdi3(long long value, int shift)
{
    const unsigned long long u = (unsigned long long)value;
    uint32_t low = (uint32_t)u, high = (uint32_t)(u >> 32);

    if (shift == 0)
        return value;
    if (shift >= 32) {
        low = shift >= 64 ? 0u : high >> (shift - 32);
        high = 0u;
    } else {
        low = (low >> shift) | (high << (32 - shift));
        high >>= shift;
    }
    return (long long)(((unsigned long long)high << 32) | low);
}
#endif

#define FTMR_MAX_K 8u
#define FTMR_RED_TIE_FLIP 1u
#define FTMR_RED_UNNORM 2u
#define FTMR_RED_RANK_PERMUTE 4u

typedef struct {
    uint32_t k;
    uint32_t count;
    uint32_t index[FTMR_MAX_K];
    uint32_t key[FTMR_MAX_K];
    uint32_t bits[FTMR_MAX_K];
} ftmr_top;

/** Monotone unsigned key of an fp32 bit pattern (-0 < +0 is harmless: a tie either way). */
static inline uint32_t ftmr_key(uint32_t bits)
{
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static inline void ftmr_init(ftmr_top *top, uint32_t k)
{
    top->k = k > FTMR_MAX_K ? FTMR_MAX_K : k;
    top->count = 0u;
}

/** Offer logit `bits` at expert `index`. Indices must be offered in ascending order. */
static inline void ftmr_offer(ftmr_top *top, uint32_t index, uint32_t bits, uint32_t red)
{
    const uint32_t key = ftmr_key(bits);
    uint32_t at, j;

    if ((bits & 0x7FFFFFFFu) > FT_DET_INF)
        return;
    /* First position this key beats; strict, so an equal key stays behind its elder. */
    for (at = 0u; at < top->count; at++)
        if ((red & FTMR_RED_TIE_FLIP) ? key >= top->key[at] : key > top->key[at])
            break;
    if (at >= top->k)
        return;
    j = top->count < top->k ? top->count : top->k - 1u;
    for (; j > at; j--) {
        top->index[j] = top->index[j - 1u];
        top->key[j] = top->key[j - 1u];
        top->bits[j] = top->bits[j - 1u];
    }
    top->index[at] = index;
    top->key[at] = key;
    top->bits[at] = bits;
    if (top->count < top->k)
        top->count++;
}

/** 1 / s for s in [1, 2^24]: a bit-trick seed and four Newton steps in det arithmetic. */
static inline uint32_t ftmr_reciprocal(uint32_t s)
{
    uint32_t r = 0x7EF311C7u - s;

    for (int step = 0; step < 4; step++) {
        const uint32_t e = ft_det_add(0x40000000u /* 2.0f */, ft_det_mul(s, r) ^ FT_DET_SIGN);

        r = ft_det_mul(r, e);
    }
    return r;
}

/**
 * The routing slot and weights for a filled top-k: `out_index[r]`, `out_weight[r]` (fp32 bits)
 * in rank order (reversed under FTMR_RED_RANK_PERMUTE).
 */
static inline void ftmr_weights(const ftmr_top *top, uint32_t red, uint32_t *out_index, uint32_t *out_weight)
{
    uint32_t e[FTMR_MAX_K];
    uint32_t sum = 0u, inverse;
    const uint32_t negative_max = top->bits[0] ^ FT_DET_SIGN;

    for (uint32_t r = 0u; r < top->count; r++) {
        e[r] = ft_det_exp(ft_det_add(top->bits[r], negative_max));
        sum = r == 0u ? e[r] : ft_det_add(sum, e[r]);
    }
    inverse = ftmr_reciprocal(sum);
    for (uint32_t r = 0u; r < top->count; r++) {
        const uint32_t to = (red & FTMR_RED_RANK_PERMUTE) ? top->count - 1u - r : r;

        out_index[to] = top->index[r];
        out_weight[to] = (red & FTMR_RED_UNNORM) ? e[r] : ft_det_mul(e[r], inverse);
    }
}

/** fp32 bits -> bf16 bits, round to nearest even (the strips' format). */
static inline uint32_t ftmr_bf16(uint32_t bits)
{
    return (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
}


/* ------------------------------------------------------------------------------------
 * ROWS MODE (qwen36 plan S1, batched prefill): a router descriptor over up to
 * FTMR_ROWS_MAX rows routes EVERY row, then takes the UNION of the experts any row chose —
 * ascending expert index, the list the expert residencies read — and writes, per list entry
 * and 32-row band, one bf16 SCALE tile whose row j is row j's weight for that expert (0 when
 * row j did not choose it). A row's contribution from an expert it did not choose is then an
 * exact zero, so its output bytes never depend on the other rows (batch invariance), and its
 * nonzero terms are summed in ascending expert order whatever the group.
 * ---------------------------------------------------------------------------------- */
#define FTMR_ROWS_MAX 128u
#define FTMR_EXPERTS_MAX 256u

/** One routed row: rank-ordered indices and weights (fp32 bits). 64 bytes. */
typedef struct {
    uint32_t index[FTMR_MAX_K];
    uint32_t weight[FTMR_MAX_K];
} ftmr_row;

/** Add a row's chosen experts to the union bitmap (FTMR_EXPERTS_MAX bits). */
static inline void ftmr_union_add(uint32_t *bitmap, const ftmr_row *row, uint32_t k)
{
    for (uint32_t r = 0u; r < k; r++)
        if (row->index[r] < FTMR_EXPERTS_MAX)
            bitmap[row->index[r] >> 5] |= 1u << (row->index[r] & 31u);
}

/** The union as an ascending list; returns its length. */
static inline uint32_t ftmr_union_list(const uint32_t *bitmap, uint32_t experts, uint32_t *list)
{
    uint32_t n = 0u;

    for (uint32_t e = 0u; e < experts && e < FTMR_EXPERTS_MAX; e++)
        if (bitmap[e >> 5] & (1u << (e & 31u)))
            list[n++] = e;
    return n;
}

/** Row `row`'s weight (fp32 bits) for `expert`, 0 if the row did not choose it. */
static inline uint32_t ftmr_row_weight(const ftmr_row *row, uint32_t k, uint32_t expert)
{
    for (uint32_t r = 0u; r < k; r++)
        if (row->index[r] == expert)
            return row->weight[r];
    return 0u;
}

/** Fill row `row` (0..31) of a bf16 32x32 tile in face order with `bf16` in all 32 columns:
 *  faces 0/1 (rows 0-15) or 2/3 (rows 16-31), 16 values each, 512 B per face. */
static inline void ftmr_tile_set_row(volatile uint32_t *tile, uint32_t row, uint32_t bf16)
{
    const uint32_t pair = (bf16 & 0xFFFFu) | (bf16 << 16);
    volatile uint32_t *left = tile + ((row >> 4) * 2u) * 128u + (row & 15u) * 8u;

    for (uint32_t w = 0u; w < 8u; w++) {
        left[w] = pair;
        left[128u + w] = pair;
    }
}

#endif /* FT_MOE_ROUTE_H */
