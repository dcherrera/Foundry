/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_dram_stripe.h
 * @brief Which DOOR of a DRAM bank each concurrent reader uses — the port/row rotation that
 *   turns one saturated endpoint into several, with no data moved and no kernel change.
 *
 * ---------------------------------------------------------------------------
 * THE STRUCTURAL FACT THIS MODULE EXISTS FOR
 * ---------------------------------------------------------------------------
 * A Blackhole GDDR bank exposes THREE NoC ports, and they are **three doors into ONE memory,
 * not three memories** (ft_tt_coord.h; the west/east split is
 * ft_tt_arch_blackhole.c:93). The same byte at the same DRAM address is reachable through
 * port 0, 1 or 2. **So choosing a port moves no data.** It is purely a choice of endpoint
 * coordinate, which makes it the cheapest bandwidth lever in the backend: an addressing
 * change, host-side, that no kernel and no weight layout can see.
 *
 * ---------------------------------------------------------------------------
 * THE THREE CEILINGS, MEASURED 2026-08-14 (plan Phase 6.7)
 * ---------------------------------------------------------------------------
 *   ENDPOINT  ~45 GB/s   one (bank, port) pair, however many readers aim at it. NoC-side
 *   NoC ROW   ~60 GB/s   SHARED by every endpoint on one NoC y. Two endpoints in one COLUMN
 *                        do not interact at all; two on one ROW share this
 *   BANK      ~64 GB/s   the memory itself. Two ports of one bank already reach 58.5 of it,
 *                        so the third door buys little per bank and a lot per card
 *   CARD      502 GB/s   8 x 63.9, with no room left for a NoC term at the card level
 *
 * A reader set aimed at ONE port of a bank is therefore capped at ~45 GB/s while the bank
 * behind it can serve ~64. Rotating the port across the readers lifts the cap to the bank's
 * own limit. Doing the same across all 8 banks measured **216 -> 296.8 GB/s, +33%, for free**
 * — because a single port INDEX occupies only 4 of the 12 NoC rows (port p of bank b and of
 * bank b+4 share a y), and rotating spreads the traffic over distinct rows.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS MODULE IS NOT
 * ---------------------------------------------------------------------------
 * It does not stripe an operand across BANKS. That requires the bytes to actually live in
 * more than one bank, so it is a placement decision a LOADER makes — `ft_tt_dram_shard.h`,
 * which calls this module for the doors once it has decided the banks. The rotation here
 * raises a per-operand ceiling from one endpoint (~45) to one bank (~64); bank striping is
 * what raises it past that.
 *
 * ---------------------------------------------------------------------------
 * THE LADDER'S FIRST TWO RUNGS, MEASURED IN ISOLATION 2026-08-15
 * ---------------------------------------------------------------------------
 * N Tensix cores reading the SAME 96 KiB block from the SAME bank at the SAME address, the
 * arms differing only in which door each reader was handed (tests/test_multicast_bench.c):
 *
 *   N            1       2       3       4     aggregate
 *   one door   3092    4442    6662    8882 ns   44.3 GB/s from N=2 up
 *   rotated    3092    3625    4622    6191 ns   63.5-63.8 GB/s from N=3 up
 *
 * The un-rotated arm is pinned at the ENDPOINT ceiling however many cores it has; the rotated
 * arm reaches 99.4% of the BANK's own measured 63.9. N=1 is byte-identical in both, which is
 * the control that this is addressing and nothing else.
 *
 * @status Off-hardware in tests/test_cooperative_matmul_plan.c; MEASURED on the card through
 *   ft_tt_op_matmul_cooperative.h by tests/test_cooperative_matmul.c, and in isolation by
 *   tests/test_multicast_bench.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_DRAM_STRIPE_H
#define FT_TT_DRAM_STRIPE_H

#include <stdint.h>

#include "ft_tt_coord.h"

/** Storage capacity for the doors of one bank — NOT a statement about any part. Blackhole
 *  has 3 (ft_tt_arch_blackhole.c). ft_tt_dram_port_set_build() refuses a grid whose
 *  architecture declares more, naming both numbers, rather than overflowing the array. */
#define FT_TT_DRAM_STRIPE_MAX_PORTS 4u

/** One door into one bank. */
typedef struct ft_tt_dram_endpoint {
    uint32_t logical_bank;
    uint32_t port;
    /** In the coordinate system the set was built for. TRANSLATED is what a transfer is
     *  AIMED with; this is the value that reaches a runtime-arg block. */
    ft_tt_noc_coord coord;
    /**
     * NoC 0 y — the row whose ~60 GB/s two endpoints SHARE.
     *
     * Carried separately and always in NOC 0, because the row limit is a property of the
     * physical node and the translated grid is an alias with its own y. Reading a row off a
     * translated coordinate would be reading the alias, not the wire.
     */
    uint32_t noc_row;
} ft_tt_dram_endpoint;

/** Every door of one bank, in port order. */
typedef struct ft_tt_dram_port_set {
    uint32_t logical_bank;
    uint32_t port_count;
    ft_tt_dram_endpoint port[FT_TT_DRAM_STRIPE_MAX_PORTS];
} ft_tt_dram_port_set;

/**
 * Resolve every NoC port of one DRAM bank, from LIVE telemetry.
 *
 * Both coordinate systems are asked for per port: `coord` in `system` (what a transfer is
 * aimed with) and `noc_row` always in NOC 0 (what the row-sharing rule is expressed in).
 * Nothing here is probed or hardcoded — a wrong DRAM endpoint does not fail a read, it hangs
 * the chip.
 *
 * @param grid          topology from ARC telemetry
 * @param logical_bank  0 .. ft_tt_coord_dram_bank_count()-1
 * @param system        the space `coord` is wanted in; usually FT_TT_COORD_TRANSLATED
 * @param out           written only on success
 * @return FT_TT_OK; FT_TT_EINVAL on a NULL argument or a bank out of range;
 *         FT_TT_ENOMEM if the architecture declares more ports than this module can hold;
 *         otherwise as ft_tt_coord_dram_port(). Every reason is logged
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_dram_port_set_build(const ft_tt_grid_info *grid, uint32_t logical_bank,
                              ft_tt_coord_system system, ft_tt_dram_port_set *out);

/**
 * The door reader `reader_index` of a concurrent group should use: `reader_index % port_count`.
 *
 * Round-robin rather than anything cleverer, because the measured rule is BALANCE: the
 * outstanding-read budget is per endpoint, and an unbalanced map cost 30% and was worse than
 * either neighbouring balanced reader count (plan Phase 6.7). Modulo is the assignment that
 * differs by at most one reader between doors for every group size.
 *
 * @return the endpoint, or NULL if `set` is NULL or empty
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
const ft_tt_dram_endpoint *ft_tt_dram_port_set_reader_endpoint(const ft_tt_dram_port_set *set,
                                                               uint32_t reader_index);

/**
 * How many pairs of the given endpoints sit on ONE NoC row — the measured ~60 GB/s two
 * endpoints share where two in one column do not interact at all.
 *
 * DISTINCT endpoints only: the same (bank, port) appearing twice is one endpoint used by two
 * readers, which is the normal case and not a collision.
 *
 * This is a PERFORMANCE fact, not a correctness one — a colliding pair returns the right
 * bytes, slowly — so it is reported rather than refused, and the caller decides. Anything
 * that plans concurrent traffic should check it and say so in its own report.
 *
 * @param endpoints      the endpoints in concurrent use
 * @param count          how many
 * @param out_first_left  may be NULL; index of the first colliding pair's first member
 * @param out_first_right may be NULL; and its second
 * @return the number of colliding DISTINCT pairs; 0 on a NULL/empty array
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_dram_stripe_shared_rows(const ft_tt_dram_endpoint *endpoints, uint32_t count,
                                       uint32_t *out_first_left, uint32_t *out_first_right);

/**
 * Choose a DOOR for each reader whose BANK is already decided by where its bytes live.
 *
 * This is the assignment problem the two measured striping rules pose together, and neither
 * can be satisfied by a formula:
 *
 *   ROWS    two endpoints on one NoC y share ~60 GB/s, and port p of bank b and of bank b+4
 *           ARE one row — so `port = reader % 3` collides as soon as the bank list contains
 *           a b and a b+4, which any list of more than four banks does.
 *   BALANCE the outstanding-read budget is per endpoint, and an unbalanced map measured 30%
 *           worse than either neighbouring balanced core count.
 *
 * Greedy in reader order, deterministic: each reader takes the door of its own bank that adds
 * the least contention, and the cost order is the LADDER's, not the intuitive one — a fresh
 * door on a used ROW beats a used DOOR on a fresh row, because two streams through one
 * endpoint share ~45 GB/s where two endpoints of one row share ~60.
 *
 * Row collisions are therefore expected, not a failure: this part has 24 DRAM endpoints on
 * only 12 rows, so any assignment of more than 12 concurrent streams MUST double up. The
 * caller checks with ft_tt_dram_stripe_shared_rows() and says what it found.
 *
 * @param grid    topology from ARC telemetry
 * @param banks   `count` logical bank indices, one per reader, in reader order
 * @param count   readers; 0 is refused
 * @param system  the space `coord` is wanted in; usually FT_TT_COORD_TRANSLATED
 * @param out     `count` endpoints, written only on success
 * @return FT_TT_OK; FT_TT_EINVAL on a NULL argument or a zero count; otherwise as
 *         ft_tt_dram_port_set_build()
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_dram_stripe_choose_ports(const ft_tt_grid_info *grid, const uint32_t *banks,
                                   uint32_t count, ft_tt_coord_system system,
                                   ft_tt_dram_endpoint *out);

#endif /* FT_TT_DRAM_STRIPE_H */
