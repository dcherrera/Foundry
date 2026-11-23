/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_matmul_moe_protocol.h
 * @brief The routed-expert (MoE) contract of the resident matmul triad: the BFP6_B weight
 *   expander and the INDIRECT expert operand (qwen36_blackhole_plan.md Phase 5.1 / 5.2,
 *   spec §4.1b / §4.3). Shared by the reader kernel and the host.
 *
 * TWO MECHANISMS, BOTH IN THE READER, BOTH COMPILED IN ONLY BY THE MoE BUILD.
 * The reader is built with `-DFT_RTRIAD_MOE=1` for the `rtriad_bfp6` program; every other
 * rtriad build compiles none of this, so their images do not move (the plan's "only the new
 * family moves").
 *
 *   1. EXPANDER. Weight tiles sit in DRAM as BFP6_B (832 B, foundry ft_bfp.h, the FOLD
 *      layout since plan S3: the host folds the artifact's stream tiles at upload); the
 *      compute thread unpacks BFP8_B (1088 B), which is what CB_IN1's pages are. Each block's
 *      `depth * columns` BFP6 tiles land in a STAGING half (the scratch, after the slot
 *      segment) and are expanded OUT OF PLACE into the block's CB_IN1 pages (eleven ALU ops
 *      a 16-datum group, exact: host reference ft_bfp6_expand_tiles()). Out of place, any
 *      split of a block's tiles between processors is safe.
 *      EXPANSION HELPERS (plan S3): with FT_RTRIAD_MOE_WORKERS = W > 1 the block's tiles are
 *      split over W processors (0 BRISC, 1 NCRISC, 2..4 TRISC0..2). BRISC posts each block
 *      as a JOB in the MAILBOX (FT_TT_MOE_MAILBOX_OFFSET from the queue control block, so
 *      every processor finds it without a CB record), expands its own share, waits for every
 *      helper's done word and only then pushes the pages. Every processor counts jobs the
 *      same way (one a block, from the descriptor words), so job ids need no handshake; the
 *      host zeroes the mailbox at every arm, so ids restart with the residency.

 *   2. INDIRECT EXPERT OPERAND. A descriptor whose flag word (parameter 13) has
 *      FT_TT_MOE_FLAG_INDIRECT set does not read B at `B base`: it reads `idx = slot[rank]`
 *      (u32) from the routing slot in DRAM and reads B at `B base + idx * stride`. `B base`
 *      is the stack's (expert 0's) address and stays CONSTANT token to token; the index is
 *      data. That is what keeps dispatch replay (Pantheon pn_tt_dispatch_hoist.c) and the
 *      parameter cache valid: no descriptor word moves when the routing does.
 *      An index >= the expert count is REFUSED (kErrMoeExpertIndex) before any B read: an
 *      unchecked index is a raw DRAM read anywhere in the bank. The refused descriptor is
 *      drained (pages pushed, nothing read) and never retired; the compute and writer still
 *      finish it on stale CB pages, so its output block holds garbage, and the residency
 *      exits with the error — the host must treat the whole token as failed.
 *
 * The slot, stride, count and an L1 scratch word are fixed for the residency, so they are
 * runtime args (the reader-only tail after the cooperative one; the arg block holds 32
 * words), not parameter words (the 16-word block is full).
 */
#ifndef FT_TT_RESIDENT_MATMUL_MOE_PROTOCOL_H
#define FT_TT_RESIDENT_MATMUL_MOE_PROTOCOL_H

#include <stdint.h>

/* ------------------------------------------------------------------------------------
 * The flag word (parameter kMatmulBNMajor = 13). Bits 0..19 are taken
 * (resident_queue_protocol.h); these follow.
 * ---------------------------------------------------------------------------------- */

/** B is a routed-expert stack: read expert `slot[rank]`, not expert 0. */
#define FT_TT_MOE_FLAG_INDIRECT (1u << 20)
/** The routing-slot entry this descriptor uses: row * 8 + rank for verify rows. 7 bits. */
#define FT_TT_MOE_FLAG_RANK_SHIFT 21u
#define FT_TT_MOE_FLAG_RANK_MASK 0x7Fu
/** Slot entries a descriptor can read: 128 through the rank field (decode / verify rows),
 *  up to 512 through the LIST mode (plan S1): the routing slot is 2 KiB. */
#define FT_TT_MOE_SLOT_ENTRIES 512u
#define FT_TT_MOE_SLOT_BYTES (FT_TT_MOE_SLOT_ENTRIES * 4u)
/**
 * K-CONCATENATED EXPERTS (the routed down projection + weighted sum in ONE matmul): K-block
 * b (of `depth` tiles, which must equal one expert's Kt) reads expert `slot[rank + b]`'s
 * N-major panel, i.e. B tile (column, kt) at `B base + slot[rank + kt / depth] * stride +
 * (column * depth + kt % depth) * tile`. A is the rank-ordered concatenation of the
 * scaled SwiGLU outputs, so DST accumulates sum_r (w_r h_r) . down_r in rank order. At most
 * FT_TT_MOE_K_EXPERTS_MAX blocks; every index is bound-checked before any B read.
 */
#define FT_TT_MOE_FLAG_K_EXPERTS (1u << 28)
/** K-blocks one descriptor may concatenate: 16 through the rank field, 256 in LIST mode. */
#define FT_TT_MOE_K_EXPERTS_MAX 16u

/* ------------------------------------------------------------------------------------
 * The reader-only runtime-arg tail, after resident_matmul_multicast_protocol.h's
 * (kArgCoopCount = 23). A residency that never runs an indirect descriptor plants zeros.
 * ---------------------------------------------------------------------------------- */

/** NoC x / y / address of the routing slot: FT_TT_MOE_SLOT_ENTRIES u32 expert indices,
 *  64 B aligned, written by the top-k op (or the host in the gate). */
enum { kArgMoeSlotX = 23 };
enum { kArgMoeSlotY = 24 };
enum { kArgMoeSlotAddress = 25 };
/** Bytes from one expert to the next in DRAM (ft_tt_fttw_expert_stride()): 64 B aligned. */
enum { kArgMoeExpertStride = 26 };
/** Experts in a stack. 0 refuses every indirect descriptor. */
enum { kArgMoeExpertCount = 27 };
/** A 64 B, 64 B-aligned L1 scratch on this core the slot segment is read into. */
enum { kArgMoeScratch = 28 };
enum { kArgMoeCount = 29 };

/** The bytes the reader reads from the slot: the aligned 64 B segment holding the entry. */
#define FT_TT_MOE_SLOT_SEGMENT_BYTES 64u

#ifndef FT_TT_MOE_LIST_MAX
#define FT_TT_MOE_LIST_MAX 256u
#endif
/** The reader's L1 scratch (runtime arg kArgMoeScratch): the slot segment, 64 B spare, then
 *  two staging halves of FT_TT_MOE_STAGE_TILES BFP6 tiles (a block is at most that many). */
#define FT_TT_MOE_STAGE_TILES 16u
#define FT_TT_MOE_STAGE_OFFSET 128u
#define FT_TT_MOE_STAGE_HALF_BYTES (FT_TT_MOE_STAGE_TILES * 832u)
/** LIST mode (plan S1): a bulk copy of slot entries 0..271 (the list and its count), then the
 *  descriptor's resolved per-entry B offsets (FT_TT_MOE_LIST_MAX words). */
#define FT_TT_MOE_SLOT_COPY_OFFSET (FT_TT_MOE_STAGE_OFFSET + 2u * FT_TT_MOE_STAGE_HALF_BYTES)
#define FT_TT_MOE_SLOT_COPY_BYTES 1088u
#define FT_TT_MOE_OFFSETS_OFFSET (FT_TT_MOE_SLOT_COPY_OFFSET + FT_TT_MOE_SLOT_COPY_BYTES)
#define FT_TT_MOE_SCRATCH_BYTES (FT_TT_MOE_OFFSETS_OFFSET + FT_TT_MOE_LIST_MAX * 4u)

/** The expansion MAILBOX, at this offset from the queue control block (past the 64-slot ring
 *  at 0x100..0x900, below the parameter slots at 0x9000). Words: job id (BRISC), source,
 *  target, tiles (BRISC), then one done word per helper (worker 1..4, each its own writer).
 *  Zeroed by the host before the residency boots. */
#define FT_TT_MOE_MAILBOX_OFFSET 0x1000u
#define FT_TT_MOE_MAILBOX_BYTES 64u
enum { kMoeJob = 0, kMoeJobSource = 1, kMoeJobTarget = 2, kMoeJobTiles = 3, kMoeDone1 = 4 };
/** Processors that can share a block's expansion. */
#define FT_TT_MOE_MAX_WORKERS 5u

/* ------------------------------------------------------------------------------------
 * The ROUTER program (`rtriad_router`: bf16 weights, fp32 output, writer built with
 * -DFT_RTRIAD_ROUTER=1). Its writer runs ft_moe_route.h over the row-0 logits of every
 * output tile it writes and, at the end of each descriptor, writes
 *   - the routing slot: entries 0..k-1 the expert indices (u32), entries 8..8+k-1 the
 *     weights (fp32 bits), one 64 B segment;
 *   - the SCALE strip (bf16, row 0 only): `expert_tiles` tiles of w_r for each rank r, then
 *     `shexp_tiles` tiles of sigmoid(l_shexp) — what the SwiGLU output is multiplied by
 *     before the down projections.
 * The writer-only runtime-arg tail, after the queue's ten:
 * ---------------------------------------------------------------------------------- */
enum { kArgRouteLogits = 10 };      /* routed experts N: logits 0..N-1 are candidates */
enum { kArgRouteShexpColumn = 11 }; /* output column of the shared-expert gate, or 0xFFFFFFFF */
enum { kArgRouteK = 12 };           /* experts selected, <= 8 */
enum { kArgRouteSlotX = 13 };
enum { kArgRouteSlotY = 14 };
enum { kArgRouteSlotAddress = 15 };
enum { kArgRouteScaleX = 16 };
enum { kArgRouteScaleY = 17 };
enum { kArgRouteScaleAddress = 18 };
enum { kArgRouteExpertTiles = 19 }; /* scale tiles per rank (512 / 32 = 16 on Qwen3.6) */
enum { kArgRouteShexpTiles = 20 };  /* scale tiles of the shared expert, 0 for none */
enum { kArgRouteScratch = 21 };     /* 128 B, 64 B-aligned L1 on this core */
enum { kArgRouteRed = 22 };         /* FTMR_RED_* bits; 0 ships */
enum { kArgRouteCount = 23 };
/** Slot entry of rank r's weight. */
#define FT_TT_MOE_SLOT_WEIGHT0 8u

/* ------------------------------------------------------------------------------------
 * BATCHED PREFILL (qwen36 plan S1): ROWS-mode routing and LIST-mode expert descriptors.
 *
 * ROUTER, flag word bit 29 (FT_TT_MOE_FLAG_ROUTE_ROWS): the descriptor's Mt (<= 4) tile rows
 * are routed row by row (ft_moe_route.h rows mode), and at its end the writer writes
 *   - the LIST into the routing slot: entries 0..n-1 the union of chosen experts in
 *     ascending order, entry FT_TT_MOE_LIST_COUNT_ENTRY = n;
 *   - for band b < Mt and list entry s < n, the E (= kArgRouteExpertTiles) scale tiles
 *     (b * 256 + s) * E + c, c < E, of the scale strip: row j = bf16 of row (32b + j)'s weight
 *     for list[s], 0 if that row did not choose it (one tile built, written E times, so the
 *     SwiGLU output strip and the scale strip are multiplied tile for tile);
 *   - the SHARED-EXPERT scale tiles of the band, FT_TT_MOE_SHEXP_SCALE_TILE0 + (a4 + b) * E + c:
 *     row j = bf16(sigmoid(l_shexp)) (descriptor word a4 = the group's first band in the pass).
 * Tiles of entries s >= n keep what they held (the host zeroes the strip once at arm, so
 * they only ever hold finite scales; the expert outputs they multiply are exact zeros).
 *
 * CONCAT (verify rows, plan S6): descriptor word a4 = first band | R << 16 with R in 1..32 and
 * Mt = 1 routes rows 0..R-1 and publishes, instead of the union, the CONCATENATION of each row's
 * own top-k in RANK order: entry r * k + i = row r's i-th expert (repeats allowed), n = R * k;
 * the scale tiles of entry r * k + i hold row r's weight for it in row r and 0 in every other
 * row; the shared-expert tiles as above. Each row's routed sum then runs over its own k experts
 * in the decode route's order with exact zeros between, so a verify row's MoE output is the
 * decode token's, bit for bit (decode publishes the same k in the same rank order).
 *
 * EXPERT READER, LIST mode (descriptor word a1, MoE build only, bit 31): B column j of the
 * descriptor is cell `s0 * E + col0 + j` of the (list entry, expert column) grid, i.e. column
 * (cell % E) of expert list[cell / E]; an entry >= n is an all-zero B column (nothing read,
 * the CB pages zero-filled). With FT_TT_MOE_FLAG_K_EXPERTS, K-block b reads list[s0 + b].
 * ---------------------------------------------------------------------------------- */
#define FT_TT_MOE_FLAG_ROUTE_ROWS (1u << 29)
#define FT_TT_MOE_LIST_COUNT_ENTRY 256u
/** Bands one rows-mode router descriptor (and one prefill MoE group) covers at most. */
#define FT_TT_MOE_ROWS_BANDS_MAX 4u
/** The scale strip: FT_TT_MOE_ROWS_BANDS_MAX x 256 list tiles, then one shared-expert tile a
 *  pass band (FT_TT_MOE_SHEXP_BANDS_MAX). */
/** Scale tiles a band: 256 list entries x E (16 on Qwen3.6) expert-ff tiles. */
#define FT_TT_MOE_EXPERT_TILES_MAX 16u
#define FT_TT_MOE_SCALE_LIST_TILES (FT_TT_MOE_LIST_MAX * FT_TT_MOE_EXPERT_TILES_MAX)
#define FT_TT_MOE_SHEXP_SCALE_TILE0 (FT_TT_MOE_ROWS_BANDS_MAX * FT_TT_MOE_SCALE_LIST_TILES)
#define FT_TT_MOE_SHEXP_BANDS_MAX 64u
#define FT_TT_MOE_SCALE_STRIP_TILES \
    (FT_TT_MOE_SHEXP_SCALE_TILE0 + FT_TT_MOE_SHEXP_BANDS_MAX * FT_TT_MOE_EXPERT_TILES_MAX)
/** The router's L1 scratch (runtime arg kArgRouteScratch): decode uses bytes 0..127; rows
 *  mode adds 32 top-k states, a routed-row table, the shared logits, the list and one tile. */
#define FT_TT_MOE_RROWS_TOPS_OFFSET 128u
#define FT_TT_MOE_RROWS_TABLE_OFFSET 4096u
#define FT_TT_MOE_RROWS_SHEXP_OFFSET (FT_TT_MOE_RROWS_TABLE_OFFSET + 128u * 64u)
#define FT_TT_MOE_RROWS_LIST_OFFSET (FT_TT_MOE_RROWS_SHEXP_OFFSET + 128u * 4u)
#define FT_TT_MOE_RROWS_TILE_OFFSET (FT_TT_MOE_RROWS_LIST_OFFSET + 1088u + 64u)
/** Per band: list position of each expert, and the (row, rank) pairs bucketed by position. */
#define FT_TT_MOE_RROWS_POS_OFFSET (FT_TT_MOE_RROWS_TILE_OFFSET + 2048u)
#define FT_TT_MOE_RROWS_FIRST_OFFSET (FT_TT_MOE_RROWS_POS_OFFSET + 512u)
#define FT_TT_MOE_RROWS_NEXT_OFFSET (FT_TT_MOE_RROWS_FIRST_OFFSET + 512u)
#define FT_TT_MOE_ROUTER_SCRATCH_BYTES (FT_TT_MOE_RROWS_NEXT_OFFSET + 512u)
/** CONCAT routing (verify rows): router descriptor word a4 = first band | rows << 16. */
#define FT_TT_MOE_CONCAT_ROWS_SHIFT 16u
#define FT_TT_MOE_CONCAT_ROWS_MAX 32u
static inline uint32_t ftmoe_route_a4(uint32_t first_band, uint32_t concat_rows)
{
    return (first_band & 0xFFFFu) | (concat_rows << FT_TT_MOE_CONCAT_ROWS_SHIFT);
}
/** LIST-mode word a1 (expert reader, MoE build). */
#define FT_TT_MOE_A1_LIST (1u << 31)
static inline uint32_t ftmoe_list_a1(uint32_t s0, uint32_t col0, uint32_t columns_per_expert)
{
    return FT_TT_MOE_A1_LIST | (s0 & 0xFFu) | ((col0 & 0xFFu) << 8) |
           ((columns_per_expert & 0xFFu) << 16);
}

/* ------------------------------------------------------------------------------------
 * Error code, continuing resident_queue_protocol.h's kErr list (which ends at 12).
 * ---------------------------------------------------------------------------------- */

/** An indirect descriptor named an expert index >= the stack's count, or the residency has
 *  no slot configured. Nothing was read. */
enum { kErrMoeExpertIndex = 13 };

/** The expert a descriptor's flag word names, or a negative value if it is not indirect. */
static inline int32_t ftmoe_rank(uint32_t flags)
{
    if ((flags & FT_TT_MOE_FLAG_INDIRECT) == 0u)
        return -1;
    return (int32_t)((flags >> FT_TT_MOE_FLAG_RANK_SHIFT) & FT_TT_MOE_FLAG_RANK_MASK);
}

/** The flag bits for an indirect descriptor reading slot entry `rank`. */
static inline uint32_t ftmoe_flags(uint32_t rank)
{
    return FT_TT_MOE_FLAG_INDIRECT | ((rank & FT_TT_MOE_FLAG_RANK_MASK) << FT_TT_MOE_FLAG_RANK_SHIFT);
}

#endif /* FT_TT_RESIDENT_MATMUL_MOE_PROTOCOL_H */
