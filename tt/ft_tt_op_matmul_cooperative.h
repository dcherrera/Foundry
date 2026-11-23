/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_matmul_cooperative.h
 * @brief Host driver for a matmul that runs on N cores at once: who owns which output
 *   columns, where the shared A operand lands, and the runtime-arg blocks that say so.
 *
 *   The single-core driver is ft_tt_op_matmul.h and it stays exactly as it is. This is a
 *   SEPARATE entry point rather than a mode of that one, because everything here is about a
 *   SET of cores — a rectangle, a leader, a barrier, a per-core slice — and folding that into
 *   a function whose whole signature is `(noc, x, y, ...)` would mean rewriting the op every
 *   caller already uses in order to add one that nobody uses yet.
 *
 * ---------------------------------------------------------------------------
 * THE WORK SPLIT, AND WHY IT IS OVER N
 * ---------------------------------------------------------------------------
 * Core c of N owns `Nt / N` consecutive OUTPUT COLUMNS. It reads its own B panel from DRAM,
 * computes `Mt x Nt/N` output tiles, and writes them to its own output panel. Every core needs
 * the SAME A block at the same moment, and that is the only operand that is shared.
 *
 *   over N   every core needs the same A and a DISJOINT B. Exactly one operand is shared, and
 *            it is the one whose re-reading the plan has named as the blocker since Phase 4
 *            ("every A tile re-read Nt times"). This is the split.
 *   over K   every core would hold a PARTIAL sum of the same output tile. The compute kernel
 *            accumulates in one DST register across the whole K loop and has no way to emit or
 *            take in a partial, and there is no cross-core add. It is not expressible without
 *            changing the compute third.
 *   over M   every core would need a DIFFERENT A block, so the multicast would have nothing to
 *            carry — the split would share nothing and cost the same DRAM traffic as today.
 *
 * **The B side wants nothing.** Each core's B panel is disjoint, so there is no broadcast to
 * make and every byte a core reads it is the only consumer of. Splitting over N leaves TOTAL B
 * traffic unchanged — the same bytes, read by N cores instead of one — which is also why B is
 * what limits the split: at large N a core's B panel becomes small enough that the fixed cost
 * of a launch dominates the work.
 *
 * **Nt MUST divide by N.** A ragged split would give members different block COUNTS, and the
 * per-block rendezvous is only correct because every member walks the same number of blocks in
 * the same order. It is refused here, not rounded.
 *
 * ---------------------------------------------------------------------------
 * THE THREE MODES ARE THE MEASUREMENT, NOT THREE FEATURES
 * ---------------------------------------------------------------------------
 * A cooperative run that is faster than one core proves nothing on its own — N cores are
 * faster than one core for reasons that have nothing to do with sharing an operand. So the
 * driver expresses all three arms and the kernel is one binary playing all of them:
 *
 *   FT_TT_OP_MATMUL_COOPERATIVE  N cores, disjoint slices, A multicast from the leader
 *   FT_TT_OP_MATMUL_SPLIT        N cores, disjoint slices, every core reads A from DRAM
 *                                itself. COOPERATIVE minus COOPERATIVE-minus-SPLIT is what
 *                                the MULTICAST is worth; SPLIT against one core is what the
 *                                extra cores are worth
 *   FT_TT_OP_MATMUL_INDEPENDENT  N cores, each doing the WHOLE matmul into its own output.
 *                                Not a split at all: it is N concurrent streams, and its
 *                                scaling is the DRAM/interconnect ceiling rather than this
 *                                op's
 *
 * ---------------------------------------------------------------------------
 * WHERE THE OUTPUT GOES
 * ---------------------------------------------------------------------------
 * Member c writes `Mt * Nt/N` tiles consecutively to `output.address + c * Mt * (Nt/N) *
 * tile_bytes` — a COLUMN PANEL, which is what the unchanged writer third can express (it
 * streams tiles from one base). **At Mt = 1 the panels concatenate into exactly the
 * single-core output**, tile for tile and byte for byte, which is what makes the correctness
 * gate a byte comparison against the solo arm rather than a PCC. At Mt > 1 they do not: the
 * whole result is there, in N panels, and a consumer must be told the layout.
 *
 * ---------------------------------------------------------------------------
 * THE MEASURED RECOMMENDATION
 * ---------------------------------------------------------------------------
 * MEASURED 2026-08-13, MiniCPM5-1B shapes at 32 tokens, one DRAM bank per operand:
 *
 *   N        1      2      3      4      6      8      (ffn_gate/up, SPLIT, x vs one core)
 *   speedup  1.00   1.80   1.87   1.88   1.95   2.07
 *
 * **Use N = 2, and the rule is ONE CORE PER DRAM BANK THE OPERANDS OCCUPY.** A single core's
 * reader already runs at 39 GB/s against a measured ~42 GB/s per-bank read wall, so two cores
 * over two banks saturate everything the layout exposes and every core after that is idle
 * bandwidth-wise. Past N=2 the whole sweep buys 15% for four times the silicon.
 *
 * `attn_k/v` (Nt = 8) tops out at 1.24x at N=2 and goes BACKWARDS at N=8 (0.70x): at one
 * output tile per core the per-op fixed cost is the whole run. Leave it on one core, or two.
 *
 * ---------------------------------------------------------------------------
 * ...AND WHY THAT RECOMMENDATION HAS A PORT ROTATION UNDER IT NOW
 * ---------------------------------------------------------------------------
 * The wall above is not the BANK's, it is ONE DOOR of the bank. Every member read A through
 * port 0 of A's bank and B through port 0 of B's bank, because `ft_tt_coord_dram_bank()` IS
 * port 0 — so N members shared two endpoints at ~45 GB/s each, and 2 x 45 is exactly the 85-90
 * GB/s the N=8 sweep flattened at. The bank behind each door serves ~64.
 *
 * `activation_ports` / `weight_ports` give the driver ALL of a bank's doors, and member `c`
 * reads through door `c % port_count`. **The bytes do not move**: three ports of one bank are
 * three doors into one memory (ft_tt_dram_stripe.h), so the same address answers through any
 * of them and the reader kernel is untouched — the endpoint arrives in the same two runtime
 * args it always did. Leave both sets empty and every member uses the operand's own (x, y),
 * i.e. exactly today's behaviour.
 *
 * @status Off-hardware in tests/test_cooperative_matmul_plan.c; MEASURED on the card by
 *   tests/test_cooperative_matmul.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_MATMUL_COOPERATIVE_H
#define FT_TT_OP_MATMUL_COOPERATIVE_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_dram_stripe.h"
#include "ft_tt_group_shared_operand.h"
#include "ft_tt_launch.h"
#include "ft_tt_op_dispatch.h"
#include "ft_tt_op_l1_layout.h"
#include "kernels/cooperative_matmul_protocol.h"

/**
 * The control region: the status block, the two semaphores, the leader's valid word and the
 * stop flag, at the SAME address on every member.
 *
 * It sits in the gap between the end of the kernel-config region (0x14000 + 69 KiB = 0x25400)
 * and the circular-buffer arena (0x26000) — 3 KiB that the Phase 4 op map leaves unowned. That
 * is deliberate rather than convenient: putting it inside the arena would make it a function
 * of how many CBs an op placed, and putting it in the kernel-config region would put it inside
 * a 16-bit offset field that the launch message also has to reach.
 *
 * Every address is the same on every member because the hardware requires it for the arrival
 * semaphore (a multicast writes ONE address on all destinations) and because one runtime-arg
 * block then serves every member, so a per-member address cannot drift.
 */
#define FT_TT_OP_COOP_CONTROL_BASE \
    (FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_KERNEL_CONFIG_BYTES)

/** 16 bytes apart so no two control words share an L1 line. */
#define FT_TT_OP_COOP_STATUS_OFFSET 0x000u
#define FT_TT_OP_COOP_READY_SEMAPHORE_OFFSET 0x040u
#define FT_TT_OP_COOP_ARRIVAL_SEMAPHORE_OFFSET 0x050u
#define FT_TT_OP_COOP_VALID_WORD_OFFSET 0x060u
#define FT_TT_OP_COOP_STOP_FLAG_OFFSET 0x070u
#define FT_TT_OP_COOP_CONTROL_BYTES 0x080u

/** The widest group this driver will plan. The hardware bound is much lower and is a
 *  RECTANGLE bound, not a count: Blackhole's Tensix columns are NoC x 1..7 and 10..16 because
 *  x = 8 and 9 carry the ARC/L2CPU spine and the east DRAM column, so a multicast rectangle is
 *  at most SEVEN columns wide and may not straddle the gap. 16 is what four rows of four fit,
 *  which is the largest shape this project has evidence for.
 *
 *  RAISED TO 32 ON 2026-08-19 for the SPLIT arm, which multicasts nothing and is therefore
 *  not bound by any rectangle. The rectangle bound above still governs a COOPERATING group
 *  and is enforced where a rectangle is actually built (pn_tt_group_place_rectangle), not
 *  here — this number sizes the plan's arrays and nothing else. It was raised because it was
 *  the ceiling on the experiment: with the hot activation strips mirrored across banks the
 *  member count is no longer capped by one bank's ~59 GB/s, and "where does the split
 *  actually scale to" cannot be asked at a limit of 16. */
#define FT_TT_OP_COOP_MAX_MEMBERS 64u

/**
 * Where ONE member's own copy or slice of an operand lives, when that operand has been spread
 * over several DRAM banks instead of stored once in one.
 *
 * The two operands are spread differently and for different reasons, and the struct is the
 * same because the reader's view is the same — a bank, a door and a base:
 *
 *   WEIGHT       PARTITIONED. Member c owns a contiguous run of output columns and the weight
 *                is stored N-major, so its panel is already a contiguous run of BYTES. Moving
 *                that run into its own bank is a relocation, not a re-tiling, and the reader
 *                then walks a grid `column_count` wide starting at column 0.
 *   ACTIVATION   REPLICATED. Every member needs ALL of A, so the copies are identical and only
 *                the bank differs. It is affordable because A is Mt x Kt tiles against the
 *                weight's Kt x Nt — 98 KiB against 14 MiB on ffn_gate/up. In a serve the cost
 *                is not the capacity but the WRITES: N copies per token down the direction of
 *                the card that does ~120 GB/s aggregate, not 500.
 */
typedef struct ft_tt_op_dram_shard {
    ft_tt_dram_endpoint endpoint; /**< the bank, and which of its doors this member uses */
    uint32_t address;             /**< base of THIS member's copy or panel inside that bank */
} ft_tt_op_dram_shard;

/** How the work is spread. See the header block — these are the three arms of one
 *  measurement, not three features. */
typedef enum ft_tt_op_matmul_cooperative_mode {
    /** N cores, disjoint column slices, A multicast once per block from the leader. */
    FT_TT_OP_MATMUL_COOPERATIVE = 0,
    /** N cores, disjoint column slices, every core reads A from DRAM for itself. The control
     *  that separates "the multicast helped" from "there were more cores". */
    FT_TT_OP_MATMUL_SPLIT = 1,
    /** N cores, each computing the WHOLE matmul into its own output. N concurrent streams. */
    FT_TT_OP_MATMUL_INDEPENDENT = 2
} ft_tt_op_matmul_cooperative_mode;

/** One cooperative matmul: the same shape fields as ft_tt_op_matmul_request, plus the mode
 *  and the bound every device-side wait takes. */
typedef struct ft_tt_op_matmul_cooperative_request {
    uint32_t tile_rows;  /**< Mt */
    uint32_t tile_inner; /**< Kt */
    uint32_t tile_cols;  /**< Nt of the WHOLE matmul, not of one member */
    /** K-tiles per NoC barrier; shrunk to a divisor of tile_inner, exactly as the single-core
     *  driver does. */
    uint32_t prefetch_tiles;
    /** 0 = K-major weight tile grid (index kt*Nt + nt), 1 = N-major (nt*Kt + kt). A statement
     *  about DRAM, not a request: both index formulas are in range, so a wrong value is a
     *  wrong answer with no error anywhere. */
    uint32_t weight_n_major;
    ft_tt_op_operand activation; /**< shared by every member */
    ft_tt_op_operand weight;     /**< the WHOLE K x N grid; the column offset does the slicing */
    ft_tt_op_operand output;     /**< base of member 0's panel; the rest follow it */
    /**
     * The DOORS of the bank each read operand lives in — every NoC port of that one bank,
     * from ft_tt_dram_port_set_build(). Member `c` reads through port `c % port_count`.
     *
     * `port_count == 0` means "use the operand's own (x, y) on every member": one endpoint,
     * which is what every caller did before striping existed and what the recorded N=2..8
     * baseline was measured with. Setting it changes no bytes and no kernel — a port is a
     * door into the same memory, so this is addressing only.
     *
     * The OUTPUT is deliberately not rotated. Writes do not behave like reads on this part
     * (aggregate ~120 GB/s, ports barely help, and the row-sharing limit does not apply to
     * them, plan Phase 6.7), so rotating it would add a variable to the measurement without a
     * measured reason to expect a gain.
     */
    ft_tt_dram_port_set activation_ports;
    ft_tt_dram_port_set weight_ports;
    /**
     * Per-member placement, when the operand has been spread over BANKS and not just doors.
     * A count equal to `member_count` turns it on for that operand and OVERRIDES its port set;
     * 0 leaves it off. Anything else is refused rather than partly applied.
     *
     * This is the half that moves data, and it is what lifts the ceiling past ONE bank: the
     * port rotation above takes an operand from ~45 GB/s (one endpoint) to ~64 (one bank), and
     * only more banks go further. The card does 502.
     *
     * `weight_shards[c].address` is the base of member c's OWN panel, so the member reads a
     * grid `column_count` wide from column 0 — the full-grid column offset it would otherwise
     * carry is folded into the address by whoever laid the panels out.
     */
    ft_tt_op_dram_shard activation_shards[FT_TT_OP_COOP_MAX_MEMBERS];
    uint32_t activation_shard_count;
    ft_tt_op_dram_shard weight_shards[FT_TT_OP_COOP_MAX_MEMBERS];
    uint32_t weight_shard_count;
    ft_tt_op_matmul_cooperative_mode mode;
    /** Spins before a device-side wait gives up and drains. Bounded AND askable is rule 4 of
     *  the resident-kernel protocol; a wait that cannot expire is a wait only `assert_reset`
     *  can end, and `assert_reset` on a blocked kernel is how cores were wedged. */
    uint32_t spin_limit;
} ft_tt_op_matmul_cooperative_request;

/** What one member owns. */
typedef struct ft_tt_op_matmul_cooperative_member {
    /** FTCM_ROLE_*, from kernels/cooperative_matmul_protocol.h. */
    uint32_t role;
    uint32_t column_first; /**< first output tile column, in the FULL grid */
    uint32_t column_count; /**< how many; equal for every member by construction */
    uint32_t output_address; /**< DRAM base of this member's output panel */
    uint32_t output_tiles;   /**< tile_rows * column_count — what the writer streams */
    /** The DOOR this member reads each operand through, in the coordinate system the port
     *  sets were built in. Equal to the request's own operand coordinate when that operand
     *  was given no port set. These are the only two reader args striping changes. */
    uint16_t activation_x;
    uint16_t activation_y;
    uint16_t weight_x;
    uint16_t weight_y;
    /**
     * Where this member's operands START, and the weight GRID it walks.
     *
     * Equal to the request's own address, `column_first` and the full Nt — unless the operand
     * was sharded, in which case the member reads its own panel from column 0 of a grid
     * exactly `column_count` wide. `column_first` above keeps naming the FULL grid either way,
     * because that is what says which part of the answer this member owns; these three say how
     * to reach the bytes.
     */
    uint32_t activation_address;
    uint32_t weight_address;
    uint32_t weight_column_first;
    uint32_t weight_full_columns;
} ft_tt_op_matmul_cooperative_member;

/** A planned cooperative matmul: the slicing, the CB placement, and the shared-operand
 *  contract. Pure arithmetic — nothing here has touched the device. */
typedef struct ft_tt_op_matmul_cooperative_plan {
    uint32_t member_count;
    /** Index into the caller's coordinate array of the core that reads A and multicasts it.
     *  Meaningless outside FT_TT_OP_MATMUL_COOPERATIVE. */
    uint32_t leader;
    uint32_t depth;
    /** The rectangle, the layout and the counts. Only built in COOPERATIVE mode. */
    ft_tt_group_shared_operand shared;
    bool shared_valid;
    /** CB_IN0, CB_IN1, CB_OUT as placed — identical on every member, which is what makes one
     *  multicast destination address correct for all of them. */
    ft_tt_launch_cb_entry cb[3];
    uint64_t cb_mask;
    unsigned char cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];
    ft_tt_op_matmul_cooperative_member member[FT_TT_OP_COOP_MAX_MEMBERS];
    /** Distinct DRAM doors this plan's readers use, counted over the operands that were given
     *  port sets. 0 when neither was. */
    uint32_t distinct_endpoints;
    /**
     * Pairs of those doors that sit on ONE NoC row and therefore SHARE ~60 GB/s.
     *
     * REPORTED, NOT REFUSED. A colliding pair returns the right bytes, only slower, so
     * refusing would reject a legal plan for a performance reason; but it is also the single
     * mistake the striping rules name first, so it is logged when it happens and carried here
     * for a caller's own report to assert on. Zero is what a correct bank pairing gives.
     */
    uint32_t row_collisions;
} ft_tt_op_matmul_cooperative_plan;

/** L1 address of one control word on any member. */
static inline uint32_t ft_tt_op_coop_status_address(void)
{
    return FT_TT_OP_COOP_CONTROL_BASE + FT_TT_OP_COOP_STATUS_OFFSET;
}
static inline uint32_t ft_tt_op_coop_ready_semaphore_address(void)
{
    return FT_TT_OP_COOP_CONTROL_BASE + FT_TT_OP_COOP_READY_SEMAPHORE_OFFSET;
}
static inline uint32_t ft_tt_op_coop_arrival_semaphore_address(void)
{
    return FT_TT_OP_COOP_CONTROL_BASE + FT_TT_OP_COOP_ARRIVAL_SEMAPHORE_OFFSET;
}
static inline uint32_t ft_tt_op_coop_valid_word_address(void)
{
    return FT_TT_OP_COOP_CONTROL_BASE + FT_TT_OP_COOP_VALID_WORD_OFFSET;
}
static inline uint32_t ft_tt_op_coop_stop_flag_address(void)
{
    return FT_TT_OP_COOP_CONTROL_BASE + FT_TT_OP_COOP_STOP_FLAG_OFFSET;
}

/**
 * Plan a cooperative matmul over `member_count` cores, refusing every shape that cannot work.
 *
 * @param members       TRANSLATED coordinates, from ft_tt_coord.h against live telemetry and
 *                      forming a RECTANGLE — use ft_tt_group_rectangle_coordinates(), which is
 *                      the partitioner that refuses a run of consecutive indices that wraps a
 *                      row or crosses the non-Tensix column gap. Only read in COOPERATIVE mode
 * @param member_count  1..FT_TT_OP_COOP_MAX_MEMBERS
 * @param leader        index into `members` of the core that reads A and multicasts it
 * @param request       the shape and the mode
 * @param out           written only on success
 * @return FT_TT_OK;
 *         FT_TT_EINVAL for a NULL argument, a zero dimension, a member count out of range, a
 *         leader outside the group, or a `tile_cols` that `member_count` does not divide;
 *         FT_TT_ENOMEM if the circular buffers do not fit the arena;
 *         FT_TT_ERROR when the members are not a rectangle or the shared-operand layout is
 *         refused. Every reason is logged
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_op_matmul_cooperative_plan_build(const ft_tt_noc_coord *members, uint32_t member_count,
                                           uint32_t leader,
                                           const ft_tt_op_matmul_cooperative_request *request,
                                           ft_tt_op_matmul_cooperative_plan *out);

/**
 * The reader's FTCM_ARG_COUNT runtime args for one member.
 *
 * @param members   the same array the plan was built from; only used for the leader's
 *                  coordinate, and may be NULL outside COOPERATIVE mode
 * @param out       receives FTCM_ARG_COUNT words
 * @param capacity  how many words `out` has
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_ENOMEM
 */
int ft_tt_op_matmul_cooperative_reader_arguments(
    const ft_tt_op_matmul_cooperative_plan *plan,
    const ft_tt_op_matmul_cooperative_request *request, const ft_tt_noc_coord *members,
    uint32_t member, uint32_t *out, uint32_t capacity);

/**
 * The compute third's three args (Mt, Kt, this member's Nt) and the writer's four (tile count,
 * destination x, y, base). Neither kernel changes for a cooperative run; only these do.
 *
 * @return FT_TT_OK, FT_TT_EINVAL, or FT_TT_ENOMEM
 */
int ft_tt_op_matmul_cooperative_compute_arguments(
    const ft_tt_op_matmul_cooperative_plan *plan,
    const ft_tt_op_matmul_cooperative_request *request, uint32_t member, uint32_t *out,
    uint32_t capacity);

int ft_tt_op_matmul_cooperative_writer_arguments(
    const ft_tt_op_matmul_cooperative_plan *plan,
    const ft_tt_op_matmul_cooperative_request *request, uint32_t member, uint32_t *out,
    uint32_t capacity);

#endif /* FT_TT_OP_MATMUL_COOPERATIVE_H */
