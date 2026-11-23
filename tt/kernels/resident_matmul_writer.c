/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_matmul_writer.c
 * @brief Writer third of the RESIDENT matmul triad, and the processor that PUBLISHES
 * COMPLETION -- both per core (completed_count/completed_token) and per group.
 *
 * WHY THE WRITER OWNS COMPLETION. An op is not done when the reader has fetched its
 * operands or when the FPU has finished multiplying; it is done when the result has left
 * L1. Phase 5's rule is that completion is produced BY the work item, and in a pipeline
 * that means by the LAST stage of it. So this processor owns `completed_count` and
 * `completed_token`, the reader owns `tail`, and nothing else writes either.
 *
 * AND WHY IT FETCHES ON `head` BUT PUBLISHES ON `tail`. A descriptor with no writer-side
 * work -- WRITE_WORD, a semaphore post -- could otherwise be retired before the reader had
 * performed it, which is the "completion that is not evidence of work" this protocol
 * exists to prevent, one processor over. So the reader's half is a PRECONDITION of
 * publishing completion. It is deliberately NOT a precondition of starting work, and that
 * distinction cost a run: with the writer's CURSOR following `tail`, the output CB (two
 * pages) fills after two tiles, the compute thread blocks in `cb_reserve_back`, the reader
 * then blocks because nothing is draining the input CBs, and the reader can never reach
 * the end of the descriptor to bump `tail`. Three processors each waiting on the next.
 * The only reason that was a diagnosis instead of a dead core is that every one of those
 * waits is bounded and askable.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_OUT index   1: output tile size in bytes
 *   2: wait mode -- 0 upstream (the cost arm), 1 stoppable (ships), 2 bound-only (the
 *      red proof for the ask)
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors; the
 *   ROUTER build (-DFT_RTRIAD_ROUTER=1) also reads the tail 10..22 of
 *   resident_matmul_moe_protocol.h
 * Descriptor words: a0 the parameter block, a2 the DESTINATION ROW PITCH in tiles (0 means
 *   `tile_cols`, which is what every Mt = 1 descriptor computes anyway), a5 the group
 *   sequence. See resident_matmul_block_protocol.h for why the pitch is not a parameter word.
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=16,2048,1 NOC=1 KERNEL_TEXT_BASE=0x18000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_matmul_writer.c ncrisc /tmp/ttk_rtriad
 *
 * @status See docs/tt_bringup_log.md.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_matmul_block_protocol.h"
#include "resident_precondition_protocol.h"
#include "resident_queue_protocol.h"
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
#include "resident_matmul_moe_helpers.h"
#endif
/* The FUSED-SWIGLU build (program `rtriad_bfp8_swiglu`): a fused descriptor's output is fanned
 * out to the strip's mirrors through the resident replica table, as the vector writer does
 * (resident_output_replicas.h). Every line it adds is under this switch. */
#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
#include "resident_matmul_swiglu_protocol.h"
#include "resident_output_replicas.h"
#endif

/* L2 pull flags beside resident_queue_protocol.h's kPullRedSkip (kept here so only the
 * programs carrying this writer change). Host mirror: Pantheon pn_tt_matmul_pull.h. */
enum { kPullSerial = 0x40000000u };  /* bit 30 of a3: poll one member at a time (control) */
enum { kPullFlags = 0xC0000000u };   /* every a3 flag bit; the rest is the member count */
enum { kPullSweepOffset = 0xFB00 };  /* 16-byte landing per member (1..63), to 0xFEF0 */

#define CB_OUT ft_tensix_compile_time_arg(0)
#define OUT_TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)

/* =========================================================================
 * The ROUTER build (`KERNEL_DEFINES=-DFT_RTRIAD_ROUTER=1`, program `rtriad_router`, fp32
 * output tiles): every output tile's row-0 logits go through ft_moe_route.h, and the routing
 * slot + scale strip are written at the end of each descriptor
 * (resident_matmul_moe_protocol.h). Every other build compiles none of it.
 * ========================================================================= */
#if defined(FT_RTRIAD_ROUTER) && FT_RTRIAD_ROUTER
#include "ft_moe_route.h"
#include "resident_matmul_moe_protocol.h"

static ftmr_top ftrt_top;
static uint32_t ftrt_shexp_bits;

/** Row 0 of an fp32 output tile at L1 `page`, logical column base `column0`. */
static void ftrt_scan_tile(uint32_t page, uint32_t column0)
{
    const uint32_t logits = FTQ_RUNTIME_ARG(kArgRouteLogits);
    const uint32_t shexp = FTQ_RUNTIME_ARG(kArgRouteShexpColumn);
    const uint32_t red = FTQ_RUNTIME_ARG(kArgRouteRed);

    for (uint32_t j = 0u; j < 32u; j++) {
        /* fp32 face order: row 0 columns 0-15 open face 0, columns 16-31 open face 1 (1024 B on). */
        const uint32_t bits =
            *(volatile uint32_t *)(uintptr_t)(page + (j < 16u ? j * 4u : 1024u + (j - 16u) * 4u));
        const uint32_t column = column0 + j;

        if (column < logits)
            ftmr_offer(&ftrt_top, column, bits, red);
        else if (column == shexp)
            ftrt_shexp_bits = bits;
    }
}

/** Write `bytes` from L1 `source` to a DRAM address and wait for it. */
static void ftrt_write(uint32_t source, uint32_t x, uint32_t y, uint32_t address, uint32_t bytes)
{
    ft_tensix_noc_write(source, ft_tensix_noc_address(x, y, address), bytes);
    ft_tensix_noc_write_barrier();
}

/** Fill a 32 B pattern of 16 bf16 copies of fp32 `bits` and write it into row 0 of `tiles`
 *  bf16 tiles from tile `first` of the scale strip (faces 0 and 1, 512 B apart). */
static void ftrt_scale_tiles(volatile uint32_t *pattern, uint32_t first, uint32_t tiles, uint32_t bits)
{
    const uint32_t half = ftmr_bf16(bits);
    const uint32_t x = FTQ_RUNTIME_ARG(kArgRouteScaleX), y = FTQ_RUNTIME_ARG(kArgRouteScaleY);
    const uint32_t base = FTQ_RUNTIME_ARG(kArgRouteScaleAddress);

    for (uint32_t w = 0u; w < 8u; w++)
        pattern[w] = half | (half << 16);
    for (uint32_t t = first; t < first + tiles; t++) {
        ft_tensix_noc_write((uint32_t)(uintptr_t)pattern, ft_tensix_noc_address(x, y, base + t * 2048u), 32u);
        ft_tensix_noc_write((uint32_t)(uintptr_t)pattern, ft_tensix_noc_address(x, y, base + t * 2048u + 512u),
                            32u);
    }
    ft_tensix_noc_write_barrier();
}

/* ---- ROWS mode (plan S1; resident_matmul_moe_protocol.h "BATCHED PREFILL") ---- */
static uint32_t ftrt_a4;     /**< a4 low half: the group's first band in the pass */
static uint32_t ftrt_concat; /**< a4 high half: CONCAT rows (verify, plan S6); 0 = union */

static FT_BH_INLINE uint32_t ftrt_scratch(uint32_t offset)
{
    return FTQ_RUNTIME_ARG(kArgRouteScratch) + offset;
}

/** Offer every row of one fp32 output tile (tile row `band`, logical columns from
 *  `column0`) to the band's 32 top-k states; the first column tile starts them, the last
 *  finishes the band's rows into the routed-row table. */
static void ftrt_rows_scan(uint32_t page, uint32_t column0, uint32_t band, uint32_t tile_cols)
{
    /* CONCAT routes only its R verify rows: the offers are the router's cost (32 rows a band
     * measured 1.38 ms a descriptor against 74 us for the decode route's one row). */
    const uint32_t live = ftrt_concat != 0u ? ftrt_concat : 32u;
    ftmr_top *tops = (ftmr_top *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_TOPS_OFFSET);
    ftmr_row *table = (ftmr_row *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_TABLE_OFFSET);
    volatile uint32_t *shexp = (volatile uint32_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_SHEXP_OFFSET);
    const uint32_t logits = FTQ_RUNTIME_ARG(kArgRouteLogits);
    const uint32_t shexp_column = FTQ_RUNTIME_ARG(kArgRouteShexpColumn);
    const uint32_t red = FTQ_RUNTIME_ARG(kArgRouteRed);
    const uint32_t k = FTQ_RUNTIME_ARG(kArgRouteK);

    if (column0 == 0u)
        for (uint32_t j = 0u; j < live; j++)
            ftmr_init(&tops[j], k);
    for (uint32_t j = 0u; j < 32u; j++) {
        const uint32_t row_base = page + (j >> 4) * 2048u + (j & 15u) * 64u;

        if (j >= live) {
            /* Rows past the verify rows: only the shared-expert logit (its scale tile row). */
            if (shexp_column >= column0 && shexp_column < column0 + 32u)
                shexp[band * 32u + j] = *(volatile uint32_t *)(uintptr_t)(
                    row_base + ((shexp_column - column0) >> 4) * 1024u + ((shexp_column - column0) & 15u) * 4u);
            continue;
        }

        for (uint32_t c = 0u; c < 32u; c++) {
            const uint32_t bits = *(volatile uint32_t *)(uintptr_t)(row_base + (c >> 4) * 1024u + (c & 15u) * 4u);
            const uint32_t column = column0 + c;

            if (column < logits)
                ftmr_offer(&tops[j], column, bits, red);
            else if (column == shexp_column)
                shexp[band * 32u + j] = bits;
        }
    }
    if (column0 + 32u >= tile_cols * 32u)
        for (uint32_t j = 0u; j < live; j++)
            ftmr_weights(&tops[j], red, table[band * 32u + j].index, table[band * 32u + j].weight);
}

/** Write the scratch tile to scale-strip tiles first .. first + E - 1 and wait until the
 *  writes have left L1 (the tile buffer is then free to change). */
static void ftrt_put_tiles(uint32_t first)
{
    const uint32_t copies = FTQ_RUNTIME_ARG(kArgRouteExpertTiles);

    for (uint32_t c = 0u; c < copies; c++)
        ft_tensix_noc_write(ftrt_scratch(FT_TT_MOE_RROWS_TILE_OFFSET),
                            ft_tensix_noc_address(FTQ_RUNTIME_ARG(kArgRouteScaleX), FTQ_RUNTIME_ARG(kArgRouteScaleY),
                                                  FTQ_RUNTIME_ARG(kArgRouteScaleAddress) + (first + c) * 2048u),
                            2048u);
    ft_tensix_noc_writes_flushed_barrier();
}

/** The end of a ROWS-mode router descriptor over `bands` tile rows: the union list, the
 *  per-entry scale tiles and the shared-expert scale tiles. */
static void ftrt_rows_publish(uint32_t bands)
{
    const ftmr_row *table = (const ftmr_row *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_TABLE_OFFSET);
    volatile uint32_t *shexp = (volatile uint32_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_SHEXP_OFFSET);
    volatile uint32_t *list = (volatile uint32_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_LIST_OFFSET);
    volatile uint32_t *tile = (volatile uint32_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_TILE_OFFSET);
    volatile uint16_t *pos = (volatile uint16_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_POS_OFFSET);
    volatile uint16_t *first = (volatile uint16_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_FIRST_OFFSET);
    volatile uint16_t *next = (volatile uint16_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_NEXT_OFFSET);
    const uint32_t k = FTQ_RUNTIME_ARG(kArgRouteK) > FTMR_MAX_K ? FTMR_MAX_K : FTQ_RUNTIME_ARG(kArgRouteK);
    uint32_t bitmap[FTMR_EXPERTS_MAX / 32u];
    uint32_t n;

    /* Word stores, not an initializer: gcc turns that into a memset the kernel link lacks. */
    for (uint32_t w = 0u; w < FTMR_EXPERTS_MAX / 32u; w++)
        ((volatile uint32_t *)bitmap)[w] = 0u;

    for (uint32_t row = 0u; row < bands * 32u; row++)
        ftmr_union_add(bitmap, &table[row], k);
    /* The list is built in place in the scratch (the data mover's own stack is small). */
    n = ftmr_union_list(bitmap, FTQ_RUNTIME_ARG(kArgRouteLogits), (uint32_t *)(uintptr_t)list);
    for (uint32_t s = 0u; s < FT_TT_MOE_LIST_MAX; s++)
        pos[s] = 0xFFFFu;
    for (uint32_t s = n; s < 272u; s++)
        list[s] = 0u;
    list[FT_TT_MOE_LIST_COUNT_ENTRY] = n;
    for (uint32_t s = 0u; s < n; s++)
        pos[list[s]] = (uint16_t)s;
    ft_bh_l1_cache_invalidate();
    ftrt_write(ftrt_scratch(FT_TT_MOE_RROWS_LIST_OFFSET), FTQ_RUNTIME_ARG(kArgRouteSlotX),
               FTQ_RUNTIME_ARG(kArgRouteSlotY), FTQ_RUNTIME_ARG(kArgRouteSlotAddress), 1088u);
    for (uint32_t w = 0u; w < 512u; w++)
        tile[w] = 0u;
    for (uint32_t b = 0u; b < bands; b++) {
        /* Bucket the band's (row, rank) pairs by list position: first[s] -> next[...]. */
        for (uint32_t s = 0u; s < n; s++)
            first[s] = 0xFFFFu;
        for (uint32_t pair = 0u; pair < 32u * k; pair++) {
            const ftmr_row *row = &table[b * 32u + pair / k];
            const uint32_t expert = row->index[pair % k];
            const uint32_t s = expert < FTMR_EXPERTS_MAX ? pos[expert] : 0xFFFFu;

            if (s == 0xFFFFu)
                continue;
            next[pair] = first[s];
            first[s] = (uint16_t)pair;
        }
        for (uint32_t s = 0u; s < n; s++) {
            for (uint32_t pair = first[s]; pair != 0xFFFFu; pair = next[pair])
                ftmr_tile_set_row(tile, pair / k, ftmr_bf16(table[b * 32u + pair / k].weight[pair % k]));
            ft_bh_l1_cache_invalidate();
            ftrt_put_tiles(b * FT_TT_MOE_SCALE_LIST_TILES + s * FTQ_RUNTIME_ARG(kArgRouteExpertTiles));
            for (uint32_t pair = first[s]; pair != 0xFFFFu; pair = next[pair])
                ftmr_tile_set_row(tile, pair / k, 0u);
        }
        for (uint32_t j = 0u; j < 32u; j++)
            ftmr_tile_set_row(tile, j, ftmr_bf16(ft_det_sigmoid(shexp[b * 32u + j])));
        ft_bh_l1_cache_invalidate();
        ftrt_put_tiles(FT_TT_MOE_SHEXP_SCALE_TILE0 + (ftrt_a4 + b) * FTQ_RUNTIME_ARG(kArgRouteExpertTiles));
        for (uint32_t j = 0u; j < 32u; j++)
            ftmr_tile_set_row(tile, j, 0u);
    }
    ft_tensix_noc_write_barrier();
}

/** The end of a CONCAT router descriptor (one band, `rows` verify rows): entry r * k + i is
 *  row r's i-th ranked expert, its scale tiles carry row r's weight in row r only. */
static void ftrt_concat_publish(uint32_t rows)
{
    const ftmr_row *table = (const ftmr_row *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_TABLE_OFFSET);
    volatile uint32_t *shexp = (volatile uint32_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_SHEXP_OFFSET);
    volatile uint32_t *list = (volatile uint32_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_LIST_OFFSET);
    volatile uint32_t *tile = (volatile uint32_t *)(uintptr_t)ftrt_scratch(FT_TT_MOE_RROWS_TILE_OFFSET);
    const uint32_t k = FTQ_RUNTIME_ARG(kArgRouteK) > FTMR_MAX_K ? FTMR_MAX_K : FTQ_RUNTIME_ARG(kArgRouteK);
    const uint32_t n = rows * k;

    for (uint32_t e = 0u; e < n; e++)
        list[e] = table[e / k].index[e % k];
    for (uint32_t e = n; e < 272u; e++)
        list[e] = 0u;
    list[FT_TT_MOE_LIST_COUNT_ENTRY] = n;
    ft_bh_l1_cache_invalidate();
    ftrt_write(ftrt_scratch(FT_TT_MOE_RROWS_LIST_OFFSET), FTQ_RUNTIME_ARG(kArgRouteSlotX),
               FTQ_RUNTIME_ARG(kArgRouteSlotY), FTQ_RUNTIME_ARG(kArgRouteSlotAddress), 1088u);
    for (uint32_t w = 0u; w < 512u; w++)
        tile[w] = 0u;
    for (uint32_t e = 0u; e < n; e++) {
        ftmr_tile_set_row(tile, e / k, ftmr_bf16(table[e / k].weight[e % k]));
        ft_bh_l1_cache_invalidate();
        ftrt_put_tiles(e * FTQ_RUNTIME_ARG(kArgRouteExpertTiles));
        ftmr_tile_set_row(tile, e / k, 0u);
    }
    for (uint32_t j = 0u; j < 32u; j++)
        ftmr_tile_set_row(tile, j, ftmr_bf16(ft_det_sigmoid(shexp[j])));
    ft_bh_l1_cache_invalidate();
    ftrt_put_tiles(FT_TT_MOE_SHEXP_SCALE_TILE0 + ftrt_a4 * FTQ_RUNTIME_ARG(kArgRouteExpertTiles));
    ft_tensix_noc_write_barrier();
}

/** The end of a router descriptor: weights, slot, scale strip. */
static void ftrt_publish(void)
{
    const uint32_t scratch = FTQ_RUNTIME_ARG(kArgRouteScratch);
    volatile uint32_t *slot = (volatile uint32_t *)(uintptr_t)scratch;
    volatile uint32_t *pattern = (volatile uint32_t *)(uintptr_t)(scratch + 64u);
    const uint32_t expert_tiles = FTQ_RUNTIME_ARG(kArgRouteExpertTiles);
    const uint32_t shexp_tiles = FTQ_RUNTIME_ARG(kArgRouteShexpTiles);
    uint32_t index[FTMR_MAX_K], weight[FTMR_MAX_K];

    ftmr_weights(&ftrt_top, FTQ_RUNTIME_ARG(kArgRouteRed), index, weight);
    for (uint32_t r = 0u; r < 16u; r++)
        slot[r] = 0u;
    for (uint32_t r = 0u; r < ftrt_top.count; r++) {
        slot[r] = index[r];
        slot[FT_TT_MOE_SLOT_WEIGHT0 + r] = weight[r];
    }
    ftrt_write(scratch, FTQ_RUNTIME_ARG(kArgRouteSlotX), FTQ_RUNTIME_ARG(kArgRouteSlotY),
               FTQ_RUNTIME_ARG(kArgRouteSlotAddress), 64u);
    for (uint32_t r = 0u; r < ftrt_top.count && expert_tiles != 0u; r++)
        ftrt_scale_tiles(pattern, r * expert_tiles, expert_tiles, weight[r]);
    if (shexp_tiles != 0u)
        ftrt_scale_tiles(pattern, ftrt_top.count * expert_tiles, shexp_tiles, ft_det_sigmoid(ftrt_shexp_bits));
}
#endif
/**
 * The writer half of one matmul descriptor: drain Mt*Nt output tiles to the destination.
 *
 * vector_add_writer.c's loop, with a stoppable wait and the drain rule. The barrier is
 * still per tile, as it is there -- this kernel is about the dispatch model, and changing
 * the writer's barrier policy in the same round would put two variables in one measurement.
 *
 * THE TILES DO NOT ARRIVE IN DESTINATION ORDER ONCE THE ROW BLOCK IS MORE THAN ONE, so the
 * counter this loop used to be is gone. The compute thread packs a whole (row block x
 * column block) section before moving on, which is `MB` short runs of `NB` tiles inside one
 * output row-band -- so the writer re-walks the reader's own three loops and computes
 * `(m, n)` from them. `row_pitch` is the tile pitch of the DESTINATION grid, which is the
 * weight's FULL Nt and not this member's slice of it, and it arrives in descriptor word a2
 * because the sixteen parameter words are full and a seventeenth would move the shared slot
 * stride (resident_matmul_block_protocol.h).
 *
 * AT MB = 1 THIS IS THE OLD LOOP, TERM FOR TERM: the row loop runs once at `m = 0`, so the
 * pitch is multiplied by zero and cannot matter, the column blocks are walked in order and
 * `n` is the running counter the old code called `tile`.
 */
static bool write_one_matmul(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                             uint32_t row_pitch)
{
    const uint32_t tile_rows = parameters[kMatmulTileRows];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t block_word = parameters[kMatmulColumnBlock];
    const uint32_t column_block = ftmm_column_block(block_word);
    const uint32_t row_block = ftmm_row_block(block_word);
    const uint32_t destination_x = parameters[kMatmulCX];
    const uint32_t destination_y = parameters[kMatmulCY];
    const uint32_t destination_base = parameters[kMatmulCBase];
    /* 0 is what every descriptor written before the word existed says, and for those it is
     * also the only value that can be right: they are all Mt = 1. */
    const uint32_t pitch = row_pitch > 0u ? row_pitch : tile_cols;
    const uint32_t probe = parameters[kMatmulBNMajor];
#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
    /* ONCE PER DESCRIPTOR, as in resident_vector_writer.c, and only for a FUSED one: a plain
     * matmul on this program writes its one destination exactly as every other program does,
     * whatever the table holds. A miss (no table, no record) returns the primary alone. */
    uint32_t fan_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t fan_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t fan_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    const uint32_t fan =
        (block_word & FT_TT_MATMUL_SWIGLU_FLAG) != 0u
            ? ftr_resolve_destinations(
                  (const volatile uint32_t *)(uintptr_t)FT_TT_REPLICA_TABLE_ADDRESS,
                  destination_x, destination_y, destination_base, fan_x, fan_y, fan_base)
            : 0u;
#endif
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    /* The reader's block arithmetic, from the same words (the depth collapse included). */
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t requested_depth = parameters[kMatmulDepth];
    const uint32_t blocks_per_pass =
        (requested_depth > 0u && tile_inner % requested_depth == 0u) ? tile_inner / requested_depth
                                                                     : tile_inner;
#endif
#if defined(FT_RTRIAD_ROUTER) && FT_RTRIAD_ROUTER
    const bool rows_mode = (probe & FT_TT_MOE_FLAG_ROUTE_ROWS) != 0u;

    ftmr_init(&ftrt_top, FTQ_RUNTIME_ARG(kArgRouteK));
    ftrt_shexp_bits = 0u;
    if (rows_mode && (tile_rows > FT_TT_MOE_ROWS_BANDS_MAX || row_block != 1u || column_block != 1u ||
                      FTQ_RUNTIME_ARG(kArgRouteExpertTiles) > FT_TT_MOE_EXPERT_TILES_MAX ||
                      ftrt_a4 + tile_rows > FT_TT_MOE_SHEXP_BANDS_MAX ||
                      (ftrt_concat != 0u &&
                       (tile_rows != 1u || ftrt_concat > FT_TT_MOE_CONCAT_ROWS_MAX ||
                        ftrt_concat * FTQ_RUNTIME_ARG(kArgRouteK) > FT_TT_MOE_LIST_MAX)))) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
#endif

    for (uint32_t mt = 0; mt < tile_rows; mt += row_block) {
        const uint32_t rows = (tile_rows - mt) < row_block ? (tile_rows - mt) : row_block;

        for (uint32_t nt = 0; nt < tile_cols; nt += column_block) {
            const uint32_t columns =
                (tile_cols - nt) < column_block ? (tile_cols - nt) : column_block;
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
            /* EXPANSION HELPER (worker 1): this pass's blocks are expanded before its output
             * tiles can exist, so the writer helps with them first (resident_matmul_moe_helpers.h). */
            for (uint32_t b = 0; b < blocks_per_pass; b++)
                (void)ftmoe_help(ctx, 1u);
#endif

            for (uint32_t row = 0; row < rows; row++) {
                for (uint32_t column = 0; column < columns; column++) {
                    const uint32_t tile = (mt + row) * pitch + nt + column;
                    const bool have_tile =
                        ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_OUT, 1, WAIT_MODE);

#if defined(FT_RTRIAD_ROUTER) && FT_RTRIAD_ROUTER
                    /* Mt = 1 router: tile (0, n) carries logits n*32 .. n*32+31 in row 0;
                     * rows mode routes all 32 rows of every tile row. */
                    if (have_tile && rows_mode)
                        ftrt_rows_scan(ft_tensix_cb_read_pointer(CB_OUT), (nt + column) * 32u, mt + row,
                                       tile_cols);
                    else if (have_tile)
                        ftrt_scan_tile(ft_tensix_cb_read_pointer(CB_OUT), (nt + column) * 32u);
#endif
                    if (have_tile && (probe & kMatmulProbeNoWrite) == 0u) {
#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
                        /* R writes, ONE barrier below: the page may not be popped until every
                         * copy has left L1, a property of the set (resident_vector_writer.c). */
                        for (uint32_t copy = 0u; copy < fan; copy++)
                            ft_tensix_noc_write(
                                ft_tensix_cb_read_pointer(CB_OUT),
                                ft_tensix_noc_address(fan_x[copy], fan_y[copy],
                                                      fan_base[copy] + tile * OUT_TILE_BYTES),
                                OUT_TILE_BYTES);
                        if (fan == 0u)
#endif
                        ft_tensix_noc_write(
                            ft_tensix_cb_read_pointer(CB_OUT),
                            ft_tensix_noc_address(destination_x, destination_y,
                                                  destination_base + tile * OUT_TILE_BYTES),
                            OUT_TILE_BYTES);
                        /* kMatmulWriterFlush: the page may be popped once its write has LEFT
                         * L1; the ack the completion post needs is taken once, below. */
                        if ((probe & kMatmulWriterFlush) != 0u)
                            ft_tensix_noc_writes_flushed_barrier();
                        else
                            ft_tensix_noc_write_barrier();
                    }
                    ft_tensix_cb_pop_front(CB_OUT, 1);
                }
            }
        }
    }
    ft_tensix_noc_write_barrier();
#if defined(FT_RTRIAD_ROUTER) && FT_RTRIAD_ROUTER
    if (!ctx->aborting && rows_mode && ftrt_concat != 0u)
        ftrt_concat_publish(ftrt_concat);
    else if (!ctx->aborting && rows_mode)
        ftrt_rows_publish(tile_rows);
    else if (!ctx->aborting)
        ftrt_publish();
#endif
    return !ctx->aborting;
}

/**
 * L2 SWEEP: the same wait, every still-pending member read per sweep into its own landing
 * line and ONE read barrier per sweep. The one-at-a-time walk below pays a full read round
 * trip per member AFTER the last one finishes (members finish within a few us of each other,
 * so it usually walks most of the table); a sweep pays one. Same reads, same bound, same
 * abort: `spins` still counts reads against the spin limit.
 */
static bool pull_group_sweep(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *table,
                             uint32_t members, uint32_t sequence)
{
    const uint32_t control = (uint32_t)(uintptr_t)ctx->control;
    volatile FTQ_L1_PTR uint32_t *landing =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(control + kPullSweepOffset);
    uint32_t pending[2] = { 0u, 0u };
    uint32_t spins = 0u, sweeps = 0u, member;

    for (member = 1u; member < members; member++)
        pending[member >> 5] |= 1u << (member & 31u);
    for (;;) {
        for (member = 1u; member < members; member++) {
            const uint32_t xy = table[2u + member];

            if ((pending[member >> 5] & (1u << (member & 31u))) == 0u)
                continue;
            ft_tensix_noc_read(ft_tensix_noc_address(xy & 0xFFFFu, xy >> 16,
                                                     control + kFieldGroupDone * 4u),
                               (uint32_t)(uintptr_t)(landing + member * 4u), 16u);
            spins++;
        }
        ft_tensix_noc_read_barrier();
        ft_bh_l1_cache_invalidate();
        for (member = 1u; member < members; member++)
            if ((int32_t)(landing[member * 4u] - sequence) >= 0)
                pending[member >> 5] &= ~(1u << (member & 31u));
        if ((pending[0] | pending[1]) == 0u)
            return true;
        if ((++sweeps & 63u) == 0u && (ftq_stop_requested(ctx) || spins >= ctx->cb_spin_limit)) {
            ftq_abort(ctx, kErrSemWaitTimeout);
            return false;
        }
    }
}

/**
 * M2: wait until every member named by the pull table has finished group sequence `sequence`.
 * Reads only (16 B NoC read of the member's control bytes 0xE0..0xEF). Stoppable and bounded:
 * a stop request or the spin bound aborts the descriptor instead of hanging the core.
 * `serial` (a3 bit 30) keeps the one-member-at-a-time walk; otherwise pull_group_sweep.
 */
static bool pull_group(ftq_context *ctx, uint32_t members, uint32_t sequence, bool serial)
{
    const uint32_t control = (uint32_t)(uintptr_t)ctx->control;
    volatile FTQ_L1_PTR uint32_t *table =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(control + kPullTableOffset);
    volatile FTQ_L1_PTR uint32_t *scratch =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(control + kPullScratchOffset);
    uint32_t member;

    if (table[0] != kPullMagic || members > table[1] || members > kPullMaxMembers) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    if (!serial)
        return pull_group_sweep(ctx, table, members, sequence);
    for (member = 1u; member < members; member++) {
        const uint32_t xy = table[2u + member];
        uint32_t spins = 0u;

        for (;;) {
            ft_tensix_noc_read(ft_tensix_noc_address(xy & 0xFFFFu, xy >> 16,
                                                     control + kFieldGroupDone * 4u),
                               (uint32_t)(uintptr_t)scratch, 16u);
            ft_tensix_noc_read_barrier();
            ft_bh_l1_cache_invalidate();
            if ((int32_t)(scratch[0] - sequence) >= 0)
                break;
            if ((++spins & 1023u) == 0u &&
                (ftq_stop_requested(ctx) || spins >= ctx->cb_spin_limit)) {
                ftq_abort(ctx, kErrSemWaitTimeout);
                return false;
            }
        }
    }
    return true;
}

void kernel_main(void)
{
    uint32_t busy_started = 0u;
    ftq_context ctx;
    uint32_t completed;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    ftmoe_boot();
#endif

    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    // All five processors resume from completed_count, the ONE number that means "every
    // stage finished this descriptor". See resident_matmul_reader.c on why a relaunch is
    // only defined over a queue whose head, tail and completed_count agree.
    completed = ctx.control[kFieldCompletedCount];
    ctx.cursor = completed;
    ctx.control[12] = 0u;
    ctx.control[13] = 0u;
    ctx.control[14] = 0u;
    ctx.control[15] = 0u;
    ctx.control[61] = 0u;

    while (running) {
        // FETCH on `head`, so the writer drains the output CB while the reader is still
        // filling the input ones. Publishing waits on `tail`, below.
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);
        /* M1 CARD-BUSY CLOCK: this processor alone writes control words 12-15 (12 last end
         * stamp, 13 busy cycles >> 4, 14 idle cycles >> 4 between descriptors, 15 count).
         * Each delta is one descriptor or one gap, so the 32-bit wrap never bites. */
        {
            const uint32_t now = ft_bh_wall_clock_low();

            if (ctx.control[15] != 0u)
                ctx.control[14] += (now - ctx.control[12]) >> 4;
            busy_started = now;
        }

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t token = descriptor[kDescToken];
        const uint32_t group_sequence = descriptor[kDescA5];
        bool done = true;

        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpMatmul) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            /* a2 is the DESTINATION ROW PITCH in tiles; 0 means `tile_cols`. It is a
             * descriptor word rather than a parameter word because the parameter block is
             * full at sixteen and every resident family shares that slot stride. */
#if defined(FT_RTRIAD_ROUTER) && FT_RTRIAD_ROUTER
            ftrt_a4 = descriptor[kDescA4] & 0xFFFFu;
            ftrt_concat = descriptor[kDescA4] >> FT_TT_MOE_CONCAT_ROWS_SHIFT;
#endif
            done = write_one_matmul(&ctx, ftq_parameters(descriptor), descriptor[kDescA2]);
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpStop) {
            running = false;
        }

        if (!done) {
            running = false;
            continue;
        }

        // THE PRECONDITION, and the only place it belongs: the reader must have finished
        // this descriptor before its completion is published. For a matmul that is already
        // true by construction — the last output tile cannot exist until the last input pair
        // was read — so this costs nothing there. For an opcode the writer does not
        // participate in, it is the whole of the guarantee.
        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }

        ctx.cursor++;
        // Token before count: a host that sees the count is entitled to trust the token it
        // reads next. Both are plain stores from ONE in-order RISC-V, which is the only
        // ordering argument this protocol makes and the only one it needs -- every other
        // control word has a different single writer (ft_tt_queue.h, THE ORDERING
        // ASSUMPTION).
        {
            const uint32_t pull = descriptor[kDescA3];

            /* a4 = the sequence every pulled member must have reached: this descriptor's
             * own for a split projection, the chain's last split one for a solo projection
             * that follows it on member 0. */
            const uint32_t pull_started = ft_bh_wall_clock_low();

            if (opcode == kOpMatmul && (pull & ~kPullFlags) > 1u &&
                (pull & kPullRedSkip) == 0u &&
                !pull_group(&ctx, pull & ~kPullFlags, descriptor[kDescA4],
                            (pull & kPullSerial) != 0u)) {
                running = false;
                continue;
            }
            /* M8: word 61 = cycles >> 4 member 0 spent pulling (waiting on the others). */
            if (opcode == kOpMatmul && (pull & ~kPullFlags) > 1u)
                ctx.control[61] += (ft_bh_wall_clock_low() - pull_started) >> 4;
        }
        ctx.control[kFieldGroupDone] = group_sequence;
        {
            /* S2: the busy clock starts where the reader met this descriptor's precondition
             * (its stamp slot, resident_precondition_protocol.h), not at pickup: a descriptor
             * issued a window early would otherwise read busy for its whole wait, which goes
             * to the idle word instead. A stamp older than the pickup changes nothing. The
             * reader wrote it before `tail` passed this descriptor, and the slot is not reused
             * before descriptor index + 64, which no ring holds until this one is done. */
            const volatile FTQ_L1_PTR uint32_t *stamps =
                (const volatile FTQ_L1_PTR uint32_t *)(uintptr_t)((uint32_t)(uintptr_t)ctx.control +
                                                                  (uint32_t)kPreStampOffset);
            const uint32_t met = stamps[(ctx.cursor - 1u) & ((uint32_t)kPreStampSlots - 1u)];

            if ((int32_t)(met - busy_started) > 0) {
                ctx.control[14] += (met - busy_started) >> 4;
                busy_started = met;
            }
        }
        {
            const uint32_t now = ft_bh_wall_clock_low();

            ctx.control[13] += (now - busy_started) >> 4;
            ctx.control[12] = now;
            ctx.control[15] += 1u;
        }
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }

    // The writer does NOT touch kFieldError or kFieldState: the reader owns those, and one
    // writer per word is the whole ordering argument. This processor's own reason for
    // leaving is in its status slot, which is where a five-processor diagnosis belongs.
    ftq_status_set(&ctx, kStateExit);
}
