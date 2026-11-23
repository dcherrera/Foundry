/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_dram_shard.h
 * @brief WHERE each member of a concurrent group reads its own slice or copy of an operand
 *   from — which bank, which door, which base address — decided when the model LOADS, not
 *   when it was converted.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A LOADER'S DECISION AND NOT A CONVERTER'S
 * ---------------------------------------------------------------------------
 * Spreading an operand over BANKS is what lifts a reader group past one bank's ~64 GB/s
 * toward the card's 502 (ft_tt_dram_stripe.h has the ladder). But the shard map is a
 * function of N — how many members will read it — and **an artifact cannot be shipped once
 * per core count.** So the artifact stays CANONICAL and N-INDEPENDENT, and the split into
 * banks happens on the machine that is about to run it, against the group size that machine
 * chose.
 *
 * That is affordable for exactly one reason: a weight stored **N-MAJOR** has member c's
 * column slice as an ALREADY CONTIGUOUS RUN OF BYTES. Placing it in a chosen bank is a
 * `memcpy`, not a re-tiling — which is why this module's output names a `source_offset` into
 * the canonical bytes and a destination, and why it can be produced at load time at all.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT DOES NOT DO
 * ---------------------------------------------------------------------------
 * **It moves no bytes.** It is pure arithmetic against live telemetry, so it is testable off
 * hardware and the caller keeps ownership of the copy — which matters because the copy is a
 * WRITE, and writes on this part aggregate at ~120 GB/s rather than 500 (plan Phase 6.7).
 * Who pays that, and when, is the caller's problem and belongs in the caller's report.
 *
 * ---------------------------------------------------------------------------
 * THE MEASURED RULES IT RESPECTS, AND THE ONE IT CANNOT
 * ---------------------------------------------------------------------------
 *   BALANCE   the outstanding-read budget is per ENDPOINT and an unbalanced map measured 30%
 *             worse than either neighbouring balanced reader count. Banks are assigned
 *             round-robin, one operand offset from the next, so no member pulls two operands
 *             through one bank and every bank carries within one stream of every other.
 *             `busiest_bank_streams` / `quietest_bank_streams` report what came out.
 *   ROWS      two DRAM endpoints on one NoC y share ~60 GB/s. The doors are chosen by
 *             ft_tt_dram_stripe_choose_ports(), never by `port = member % 3`, which collides
 *             the moment the bank list holds a b and a b+4.
 *   PORTS     which is the same call: a bank's three doors are what separates ~45 GB/s from
 *             ~64.
 *
 * **Row collisions above 12 concurrent streams are STRUCTURAL** — 24 DRAM endpoints on 12
 * rows — so they are counted and reported, never refused. A colliding pair returns the right
 * bytes and is slower, and refusing a correct plan for a performance reason would be wrong.
 *
 * @status Off-hardware in tests/test_cooperative_matmul_plan.c; MEASURED on the card through
 *   tests/test_cooperative_matmul.c, which stages its bank-striped arms from these plans.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_DRAM_SHARD_H
#define FT_TT_DRAM_SHARD_H

#include <stdint.h>

#include "ft_tt_arch.h"
#include "ft_tt_dram_stripe.h"

/** The widest group this module will plan for. Matches the cooperative driver's own bound;
 *  a plan is caller storage, so this is a capacity and not a statement about the part.
 *  Raised 16 -> 32 with FT_TT_OP_COOP_MAX_MEMBERS on 2026-08-19. */
#define FT_TT_DRAM_SHARD_MAX_MEMBERS 32u

/** How many distinct operands one plan can place. A matmul places two (a weight and an
 *  activation); the bound exists so the plan is a plain struct with no allocation. */
#define FT_TT_DRAM_SHARD_MAX_OPERANDS 4u

/** How an operand is divided between the members that read it. */
typedef enum ft_tt_dram_shard_spread {
    /**
     * Member c owns a DISTINCT contiguous run of the canonical bytes — the N-major weight's
     * column slice. Total placed bytes equal the artifact's; nothing is duplicated.
     */
    FT_TT_DRAM_SHARD_PARTITIONED = 0,
    /**
     * Every member needs ALL of it, so each gets an identical copy and only the bank differs.
     * Affordable when the operand is small against the partitioned one — an activation is
     * 98 KiB against a weight's 14 MiB on `ffn_gate/up` — and priced in WRITES, not capacity.
     */
    FT_TT_DRAM_SHARD_REPLICATED = 1
} ft_tt_dram_shard_spread;

/** One operand to place, described in the canonical artifact's own terms. */
typedef struct ft_tt_dram_shard_operand {
    ft_tt_dram_shard_spread spread;
    /** Where inside a bank this operand's copies may start. Every bank uses the same base;
     *  members that land in one bank are stacked above it, `member_bytes` apart. Two operands
     *  sharing a base is refused, because their stacks would overlap. */
    uint32_t bank_base_address;
    /** Bytes of ONE member's slice or copy. Zero is refused. */
    uint32_t member_bytes;
} ft_tt_dram_shard_operand;

/** Where one member's piece of one operand goes, and which canonical bytes it is. */
typedef struct ft_tt_dram_shard_placement {
    /** The bank, the door, and the door's NoC row. */
    ft_tt_dram_endpoint endpoint;
    /** Base of this member's piece inside that bank. */
    uint32_t address;
    /** Byte offset into the CANONICAL artifact of the bytes that belong here. Always 0 for a
     *  replicated operand; `member * member_bytes` for a partitioned one. */
    uint32_t source_offset;
    /** How many, i.e. the operand's `member_bytes`. Carried per placement so a caller can
     *  drive the copy from the placement alone. */
    uint32_t bytes;
} ft_tt_dram_shard_placement;

/** A whole group's placement, plus what the measured rules say about it. */
typedef struct ft_tt_dram_shard_plan {
    uint32_t member_count;
    uint32_t operand_count;
    /** `[operand][member]`. Use ft_tt_dram_shard_placement_of() rather than indexing. */
    ft_tt_dram_shard_placement
        placement[FT_TT_DRAM_SHARD_MAX_OPERANDS][FT_TT_DRAM_SHARD_MAX_MEMBERS];
    /** Distinct (bank, port) endpoints the plan reads through. */
    uint32_t distinct_endpoints;
    /** Pairs of those endpoints sharing a NoC row, from ft_tt_dram_stripe_shared_rows().
     *  REPORTED, not refused — see the header block. */
    uint32_t row_collisions;
    /** Concurrent streams aimed at the busiest and the quietest bank. A balanced map differs
     *  by at most one; the gap is the BALANCE rule made visible, and it is what a caller
     *  should quote next to an aggregate GB/s. */
    uint32_t busiest_bank_streams;
    uint32_t quietest_bank_streams;
} ft_tt_dram_shard_plan;

/**
 * Plan where every member reads every operand from, for one group size.
 *
 * Deterministic: the same grid, operands and member count always give the same plan, which
 * is what lets a byte comparison against a single-core run be the correctness gate.
 *
 * @param grid           topology from ARC telemetry
 * @param operands       what to place, in the caller's own operand order. The order matters:
 *                       operand i is offset i * (banks / operand_count) banks from operand 0,
 *                       so that no member pulls two operands through one bank
 * @param operand_count  1 .. FT_TT_DRAM_SHARD_MAX_OPERANDS
 * @param member_count   1 .. FT_TT_DRAM_SHARD_MAX_MEMBERS
 * @param system         the space the endpoint coordinates are wanted in; usually
 *                       FT_TT_COORD_TRANSLATED
 * @param out            written only on success
 * @return FT_TT_OK; FT_TT_EINVAL on a NULL argument, a count out of range, a zero
 *         `member_bytes`, or two operands sharing a `bank_base_address`; FT_TT_ENOMEM if a
 *         bank cannot hold the members that landed in it; otherwise as
 *         ft_tt_dram_stripe_choose_ports(). Every reason is logged
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_dram_shard_plan_build(const ft_tt_grid_info *grid,
                                const ft_tt_dram_shard_operand *operands,
                                uint32_t operand_count, uint32_t member_count,
                                ft_tt_coord_system system, ft_tt_dram_shard_plan *out);

/**
 * One placement out of a plan, bounds-checked.
 *
 * @return the placement, or NULL if `plan` is NULL or either index is past the plan's counts
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
const ft_tt_dram_shard_placement *ft_tt_dram_shard_placement_of(
    const ft_tt_dram_shard_plan *plan, uint32_t operand, uint32_t member);

#endif /* FT_TT_DRAM_SHARD_H */
