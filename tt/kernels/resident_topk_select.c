/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_topk_select.c
 * @brief The working third of the RESIDENT TOPK/SAMPLE program, and the processor that
 * PUBLISHES COMPLETION. It is the writer's slot in the triad and it does the selection
 * itself, which is the whole shape of this op: see resident_topk_protocol.h §2 for why
 * top-k is a scalar streaming selection and not a tile map.
 *
 * TWO OPCODES, ONE PROGRAM, AND THAT IS THE GROUPING RULE NOT A CONVENIENCE. A resident
 * binary is bound to a fixed per-processor slab, so only one program is resident on a
 * core at a time and swapping costs five kernel images -- ~0.19 ms over the x1 link
 * (docs/tt_bringup_log.md §20.4). A decode step must do TOPK and then SAMPLE on the SAME
 * core in sequence, so splitting them would put an image reload inside every token.
 *
 * WHAT LEAVES THE CARD: four bytes. TOPK leaves N pairs in L1 for SAMPLE to read; SAMPLE
 * writes ONE token id to the address the descriptor names -- normally the next step's
 * embedding input, so the loop closes here rather than at the host. That hand-off is a
 * 16-byte ALIGNED NoC write and not `noc_inline_dw_write`; the note in
 * resident_topk_protocol.h says what the inline call cost and why the API excludes the
 * destination this op actually uses.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_LOGITS index   1: tile size in bytes
 *   2: wait mode -- 0 upstream, 1 stoppable (ships), 2 bound-only
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors
 *
 * Build (NCRISC, NoC 1):
 *   CTARGS=0,2048,1 NOC=1 KERNEL_TEXT_BASE=0x18000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_topk_select.c ncrisc /tmp/ttk_topk
 *
 * @status GREEN ON SILICON (Tensix 51, 57, 59). text 9816 B against a 12 KiB slab,
 *         .data+.bss 128 B. Full MiniCPM5-1B row (130,560 logits) selected EXACTLY,
 *         2.590 ms/token. The token hand-off used `noc_inline_dw_write` until 2026-08-13
 *         and left the core unbootable every run while reporting 0 failures; it is now an
 *         aligned 16-byte `noc_async_write` and the suite's last stage re-boots the core
 *         it ran on so the same defect cannot pass again.
 * @issues The sampler's arithmetic is soft float (no F extension on this core, MEASURED).
 *         That is a cost on a set of at most 64 survivors, never on the 130,560-element
 *         scan, which is integer throughout.
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"
#include "resident_topk_protocol.h"

#define CB_LOGITS ft_tensix_compile_time_arg(0)
#define TILE_BYTES ft_tensix_compile_time_arg(1)
#define WAIT_MODE ft_tensix_compile_time_arg(2)
/** Elements of a tile row that live in one face (docs/tt_kernel_model.md §9). */
enum { kFaceWidth = 16 };
enum { kTileWidth = 32 };
/** Byte stride from face (r, 0) to face (r, 1) inside a bf16 tile: one 16x16 face. */
enum { kFaceBytes = 512 };

static uint32_t block_depth(uint32_t run, uint32_t requested)
{
    return (requested > 0u && run > 0u && run % requested == 0u) ? requested : 1u;
}

/**
 * Refuse a descriptor this program cannot execute, INSTEAD of blocking on it forever.
 *
 * The same four conditions the reader refuses (see its own note — the two circular-buffer
 * ones are not the documented "depth divides the run", and the difference OVERWROTE L1 on
 * this card), plus the one that is this processor's alone: **a zero select count would index
 * `entries[capacity - 1]` at capacity 0**, i.e. one word before the array. The host refuses
 * it too, and that is not a reason to leave it out here — a kernel that trusts the host to
 * have validated is a kernel whose safety lives in a different file from its unsafe
 * operation.
 *
 * Both processors compute the SAME predicate from the SAME words, so they refuse together
 * and stay in lockstep; a guard on one side only would leave the other pushing into a buffer
 * nobody drains.
 *
 * AND THE RESULT ADDRESS IS A WIRE-SUPPLIED POINTER THIS PROCESSOR WRITES THROUGH. A magic
 * word in the block is not a bounds check — it is read AFTER the dereference on the SAMPLE
 * side and it is not checked at all on the TOPK side, which WRITES the block. Upstream
 * refuses a NoC write below `MEM_MAP_READ_ONLY_END` for exactly this
 * (its NoC-address sanitizer): below that line sit the firmware
 * mailboxes, and overwriting them is how a core comes back unbootable at its NEXT boot while
 * the run that did it reports 0 failures. This is the file where that happened.
 */
static bool descriptor_is_executable(ftq_context *ctx, uint32_t tile_count, uint32_t depth,
                                     uint32_t select_count, uint32_t row, uint32_t result_address)
{
    if (tile_count == 0u || depth == 0u)
        ctx->error = kErrBadParameters;
    else if (!ft_tensix_cb_pop_grid_is_safe(CB_LOGITS, depth))
        ctx->error = kErrBadParameters;
    else if (select_count == 0u || select_count > kTopkMaxSelect || row >= kTileWidth)
        ctx->error = kErrBadParameters;
    else if (!ft_tensix_l1_block_is_writable(result_address, kTopkResultBytes))
        ctx->error = kErrBadParameters;
    else
        return true;
    return false;
}

/* The per-tile offer (two face rows, the reject path in registers) is ftk_offer_tile_row in
 * resident_topk_protocol.h, shared with the TRISC helpers. */

/** §6 tail stats: the host's threshold keys and the counts this descriptor accumulates. */
typedef struct ftk_tail {
    uint32_t keys[kTopkTailMaxBins];
    uint32_t counts[kTopkTailMaxBins + 1];
    uint32_t bins;
} ftk_tail;

/** Read and check the host's threshold keys; false (no counts) when they are unusable. */
static bool tail_load(ftk_tail *tail, volatile FTQ_L1_PTR uint32_t *result, uint32_t bins)
{
    if (bins == 0u || bins > (uint32_t)kTopkTailMaxBins)
        return false;
    for (uint32_t j = 0u; j < bins; j++) {
        tail->keys[j] = result[kTopkTailKeysOffset + j];
        if (tail->keys[j] > 0xFFFFu || (j > 0u && tail->keys[j] >= tail->keys[j - 1u]))
            return false;
    }
    for (uint32_t j = 0u; j <= bins; j++)
        tail->counts[j] = 0u;
    tail->bins = bins;
    return true;
}

/** Count the 16 logits of one face row into the tail bins (§6). */
static void count_face_row(ftk_tail *tail, const uint32_t* words)
{
    for (uint32_t pair = 0u; pair < kFaceWidth / 2u; pair++) {
        const uint32_t packed = words[pair];

        tail->counts[ftk_tail_bin(tail->keys, tail->bins, ftk_key_from_bf16(packed & 0xFFFFu))]++;
        tail->counts[ftk_tail_bin(tail->keys, tail->bins, ftk_key_from_bf16(packed >> 16))]++;
    }
}

/** count_face_row over one tile's row, both faces (ftk_offer_tile_row's addressing). */
static void count_tile_row(ftk_tail *tail, uint32_t tile_address, uint32_t row)
{
    const uint32_t face_row = row / kFaceWidth;
    const uint32_t within_face = row % kFaceWidth;
    const uint32_t low_offset = face_row * 2u * kFaceBytes + within_face * kFaceWidth * 2u;

    count_face_row(tail, (const uint32_t *)(uintptr_t)(tile_address + low_offset));
    count_face_row(tail, (const uint32_t *)(uintptr_t)(tile_address + low_offset + kFaceBytes));
}

/** Publish the counts (or a zero status when there are none) into the tail area. */
static void write_tail_counts(const ftk_tail *tail, bool counted,
                              volatile FTQ_L1_PTR uint32_t *result)
{
    volatile FTQ_L1_PTR uint32_t *area = result + kTopkTailCountsOffset;

    for (uint32_t j = 0u; counted && j <= tail->bins; j++)
        area[j] = tail->counts[j];
    area[kTopkTailStatusWord] = counted ? (uint32_t)kTopkTailStatusMagic : 0u;
}

/** Publish the selection into the L1 result block the parameter block names. */
static void write_result_block(const ftk_select *selection, volatile FTQ_L1_PTR uint32_t *result)
{
    volatile FTQ_L1_PTR uint32_t *pairs = result + kTopkResultHeaderWords;

    for (uint32_t entry = 0u; entry < selection->count; entry++) {
        pairs[entry * kTopkPairWords + kTopkPairIndex] = selection->entries[entry].index;
        pairs[entry * kTopkPairWords + kTopkPairValueBits] =
            ftk_fp32_bits_from_bf16(selection->entries[entry].bf16_bits);
    }
    result[kTopkResultTokenValid] = 0u;
    result[kTopkResultCount] = selection->count;
    // Magic LAST, so a host or a SAMPLE descriptor that sees the magic is entitled to the
    // count and the pairs beside it. One in-order RISC-V, the same claim the queue's
    // token-before-count makes.
    result[kTopkResultSequence] = result[kTopkResultSequence] + 1u;
    result[kTopkResultMagic] = kTopkResultMagicValue;
}

/** MULTI-ROW TOPK: one selection per row, in local data RAM (static, not the stack: four
 *  of them are ~3.1 KiB of NCRISC's 8 KiB). Row 0 of a single-row descriptor uses slot 0. */
static ftk_select g_row_selection[kTopkMaxRows];

/** Rows a descriptor selects over: the word, with 0 meaning one. */
static uint32_t descriptor_rows(volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t rows = parameters[kTopkRows];

    return rows == 0u ? 1u : rows;
}

/** The row span is checked HERE as well as by the host: row k writes the k-th result block,
 *  so a span the host did not mean is a write past the block the host placed. */
static bool rows_are_executable(ftq_context *ctx, uint32_t row, uint32_t rows, bool tail_asked,
                                uint32_t result_address)
{
    if (rows > (uint32_t)kTopkMaxRows || row + rows > kTileWidth || (rows > 1u && tail_asked) ||
        !ft_tensix_l1_block_is_writable(result_address, rows * kTopkResultBytes)) {
        ctx->error = kErrBadParameters;
        return false;
    }
    return true;
}

/** Whether a helper's done stamp has reached `stamp`: same descriptor, block at least as far.
 *  A helper posts its last block's stamp and then the final one straight after, so an exact
 *  match would miss the block stamp it overwrote (round 2026-10-22: NCRISC timed out on the
 *  last block while every helper sat at the final stamp). */
static bool stamp_reached(uint32_t done, uint32_t stamp)
{
    return (done >> 16) == (stamp >> 16) && (done & 0xFFFFu) >= (stamp & 0xFFFFu);
}

/** Every helper has posted `stamp` or later; stoppable and bounded like a CB wait. */
static bool helpers_reached(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *area, uint32_t stamp)
{
    for (uint32_t h = 0u; h < (uint32_t)kTopkHelpers; h++) {
        uint32_t spins = 0u;

        while (!stamp_reached(area[kTopkHelperDone + h], stamp)) {
            if (ftq_stop_requested(ctx) || ++spins >= ctx->cb_spin_limit) {
                ftq_abort(ctx, ftq_stop_requested(ctx) ? kErrNone : kErrCbWaitTimeout);
                return false;
            }
        }
    }
    return true;
}

/** Helper h's published selection for row k. */
static volatile FTQ_L1_PTR ftk_select *helper_selection(volatile FTQ_L1_PTR uint32_t *area,
                                                        uint32_t h, uint32_t k)
{
    return (volatile FTQ_L1_PTR ftk_select *)(uintptr_t)((uint32_t)(uintptr_t)area +
                                                          kTopkHelperSelectOffset +
                                                          (h * kTopkMaxRows + k) *
                                                              (uint32_t)sizeof(ftk_select));
}

/**
 * Merge NCRISC's selection for row k with the three helpers' into the result block: the
 * best head of the four sorted lists each step, key descending then index ascending — the
 * order a single pass over every tile produces, so the block is that pass's block.
 */
static void merge_and_write(const ftk_select *own, volatile FTQ_L1_PTR uint32_t *area, uint32_t k,
                            bool red_drop, volatile FTQ_L1_PTR uint32_t *result)
{
    volatile FTQ_L1_PTR uint32_t *pairs = result + kTopkResultHeaderWords;
    volatile FTQ_L1_PTR ftk_select *helper[kTopkHelpers];
    uint32_t head[1 + kTopkHelpers] = {0u, 0u, 0u, 0u};
    uint32_t count = 0u;

    for (uint32_t h = 0u; h < (uint32_t)kTopkHelpers; h++)
        helper[h] = helper_selection(area, h, k);
    while (count < own->capacity) {
        uint32_t best = 0xFFu, best_key = 0u, best_index = 0u, best_bits = 0u;

        for (uint32_t list = 0u; list <= (uint32_t)kTopkHelpers; list++) {
            uint32_t key, index, bits;

            if (list == 0u) {
                if (head[0] >= own->count)
                    continue;
                key = own->entries[head[0]].key;
                index = own->entries[head[0]].index;
                bits = own->entries[head[0]].bf16_bits;
            } else {
                volatile FTQ_L1_PTR ftk_select *other = helper[list - 1u];

                if ((red_drop && list == (uint32_t)kTopkHelpers) || head[list] >= other->count)
                    continue;
                key = other->entries[head[list]].key;
                index = other->entries[head[list]].index;
                bits = other->entries[head[list]].bf16_bits;
            }
            if (best == 0xFFu || key > best_key || (key == best_key && index < best_index)) {
                best = list;
                best_key = key;
                best_index = index;
                best_bits = bits;
            }
        }
        if (best == 0xFFu)
            break;
        head[best]++;
        pairs[count * kTopkPairWords + kTopkPairIndex] = best_index;
        pairs[count * kTopkPairWords + kTopkPairValueBits] = ftk_fp32_bits_from_bf16(best_bits);
        count++;
    }
    result[kTopkResultTokenValid] = 0u;
    result[kTopkResultCount] = count;
    result[kTopkResultSequence] = result[kTopkResultSequence] + 1u;
    result[kTopkResultMagic] = kTopkResultMagicValue;
}

/**
 * The selection half of one TOPK descriptor, over `rows` consecutive token rows.
 *
 * Every trip count comes from the parameter block and none from a wait, which is what makes
 * the drain safe (resident_queue_protocol.h rule 3): an abandoned wait still runs the loop
 * to completion and still pops every page, so the reader's push/pop counters stay balanced.
 * Each row sees the same tiles in the same order as a single-row descriptor for that row, so
 * its result block is that descriptor's block. With helpers, NCRISC scans tiles t % 4 == 0
 * and publishes each block before scanning it; it pops only after every helper is done.
 */
static bool select_one_logit_row(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t tile_count = parameters[kTopkTileCount];
    const uint32_t row = parameters[kTopkRow];
    const uint32_t rows = descriptor_rows(parameters);
    const uint32_t depth = block_depth(tile_count, parameters[kTopkDepth]);
    const bool drain_only = (parameters[kTopkFlags] & kTopkFlagDrainOnly) != 0u;
    const bool tail_asked = (parameters[kTopkFlags] & kTopkFlagTailStats) != 0u;
    const bool helpers = ftk_helpers_wanted(parameters);
    const uint32_t stride = helpers ? (uint32_t)kTopkHelpers + 1u : 1u;
    volatile FTQ_L1_PTR uint32_t *result =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(parameters[kTopkResultAddress]);
    volatile FTQ_L1_PTR uint32_t *area =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(parameters[kTopkHelperArea]);
    ftk_tail tail;
    bool counting;

    if (!descriptor_is_executable(ctx, tile_count, depth, parameters[kTopkSelectCount], row,
                                  parameters[kTopkResultAddress]) ||
        !rows_are_executable(ctx, row, rows, tail_asked, parameters[kTopkResultAddress]))
        return false;
    if (helpers && !ft_tensix_l1_block_is_writable(parameters[kTopkHelperArea], kTopkHelperBytes)) {
        ftq_abort(ctx, kErrBadParameters);
        return false;
    }
    for (uint32_t k = 0u; k < rows; k++)
        ftk_select_init(&g_row_selection[k], parameters[kTopkSelectCount]);
    counting = tail_asked && !drain_only && tail_load(&tail, result, parameters[kTopkTailBins]);
    if (helpers) {
        area[kTopkHelperTileBytes] = TILE_BYTES;
        area[kTopkHelperDepth] = depth;
    }

    for (uint32_t tile = 0u, block = 0u; tile < tile_count; tile += depth, block++) {
        const bool have = ftq_cb_wait_front_stoppable(ctx, (int32_t)CB_LOGITS,
                                                      (int32_t)depth, WAIT_MODE);
        const uint32_t page = ft_tensix_cb_read_pointer(CB_LOGITS);

        if (helpers) {
            area[kTopkHelperPage] = page;
            area[kTopkHelperStamp] = ftk_stamp(ctx->cursor, block + 1u);
        }
        if (have && !drain_only) {
            for (uint32_t within = 0u; within < depth; within++) {
                if (((tile + within) & (stride - 1u)) != 0u)
                    continue;
                for (uint32_t k = 0u; k < rows; k++)
                    ftk_offer_tile_row(&g_row_selection[k], page + within * TILE_BYTES, row + k,
                                       (tile + within) * kTileWidth);
                if (counting)
                    count_tile_row(&tail, page + within * TILE_BYTES, row);
            }
        }
        if (helpers)
            (void)helpers_reached(ctx, area, ftk_stamp(ctx->cursor, block + 1u));
        ft_tensix_cb_pop_front(CB_LOGITS, depth);
    }

    if (tail_asked)
        write_tail_counts(&tail, counting && !ctx->aborting, result);
    if (helpers && helpers_reached(ctx, area, ftk_stamp(ctx->cursor, kTopkStampFinal))) {
        for (uint32_t k = 0u; k < rows; k++)
            merge_and_write(&g_row_selection[k], area, k,
                            (parameters[kTopkFlags] & kTopkFlagRedDropHelper) != 0u,
                            result + k * (kTopkResultBytes / 4u));
        return !ctx->aborting;
    }
    for (uint32_t k = 0u; k < rows; k++) {
        if (drain_only)
            g_row_selection[k].count = 0u;
        write_result_block(&g_row_selection[k], result + k * (kTopkResultBytes / 4u));
    }
    return !ctx->aborting;
}

/* ================================================================
 * The sampler
 * ================================================================
 * Transliterated from Pantheon's `sparse_sample()` / `sparse_argmax()` (src/core/engine.c),
 * which is the path the K80 ALREADY takes whenever `fc_topk_logits` reduced the row -- the
 * host scatters k pairs into an all -INFINITY vocabulary and samples that. So this is not a
 * new approximation: it is the deployed semantics, and the reference the gate uses is that
 * same function.
 *
 * THE ORDER IS THE POINT, AND ONE ORDERING HERE WAS A REAL BUG ON THE K80. Greedy is a PURE
 * ARGMAX ON THE RAW VALUES, taken BEFORE any repetition penalty. Applying the penalty first
 * can demote the true argmax below a near-tied runner-up -- on Bonsai-27B that flipped ':'
 * (25.41) to ' given' (25.08) and lost the answer, with a forward pass that was faithful
 * throughout (engine.c's own note). The penalty belongs to the SAMPLING path only, and
 * `sample_greedy()` below is deliberately a separate function so the two orders cannot be
 * merged by accident.
 */

/** At most kTopkMaxSelect survivors, held on the stack. NCRISC's local memory budget is
 *  ~4 KiB (dev notes §5); three float arrays of 64 is 768 bytes. */
typedef struct ftk_survivors {
    uint32_t ids[kTopkMaxSelect];
    float values[kTopkMaxSelect];
    float probabilities[kTopkMaxSelect];
    float work[kTopkMaxSelect];
    uint32_t count;
} ftk_survivors;

/**
 * Gather the result block into ASCENDING INDEX order.
 *
 * Ascending is not cosmetic. The reference's cumulative draw walks the vocabulary in index
 * order and returns the first token whose running sum passes `r`; a different order gives a
 * different token for the same draw. The result block is stored descending by VALUE because
 * that is what `fc_topk_logits` returns, so the sort happens exactly once, here.
 */
static bool gather_survivors(volatile FTQ_L1_PTR uint32_t *result, ftk_survivors *survivors)
{
    volatile FTQ_L1_PTR uint32_t *pairs = result + kTopkResultHeaderWords;
    uint32_t count;

    if (result[kTopkResultMagic] != kTopkResultMagicValue)
        return false;
    count = result[kTopkResultCount];
    if (count == 0u || count > kTopkMaxSelect)
        return false;

    for (uint32_t entry = 0u; entry < count; entry++) {
        const uint32_t index = pairs[entry * kTopkPairWords + kTopkPairIndex];
        const uint32_t bits = pairs[entry * kTopkPairWords + kTopkPairValueBits];
        uint32_t position = entry;
        float value;

        __builtin_memcpy(&value, &bits, sizeof(value));
        while (position > 0u && survivors->ids[position - 1u] > index) {
            survivors->ids[position] = survivors->ids[position - 1u];
            survivors->values[position] = survivors->values[position - 1u];
            position--;
        }
        survivors->ids[position] = index;
        survivors->values[position] = value;
    }
    survivors->count = count;
    return true;
}

/** Greedy: strict `>` scanned in ascending index order, so the LOWEST id wins a tie --
 *  identical to the full path's `for (i=1..n_vocab) if (logits[i] > logits[best])`. */
static uint32_t sample_greedy(const ftk_survivors *survivors)
{
    uint32_t best = 0u;

    for (uint32_t entry = 1u; entry < survivors->count; entry++)
        if (survivors->values[entry] > survivors->values[best])
            best = entry;
    return survivors->ids[best];
}

/** The repetition penalty, over the recent ring, oldest entry first. Positive logits are
 *  divided and non-positive ones multiplied, per occurrence -- the reference applies it once
 *  per appearance in the window, not once per distinct token. */
static void apply_repeat_penalty(ftk_survivors *survivors, volatile FTQ_L1_PTR uint32_t *state,
                                 float penalty)
{
    volatile FTQ_L1_PTR uint32_t *recent = state + kSampleRecentOffsetWords;
    const uint32_t stored = state[kSampleStateRecentCount];
    const uint32_t window = state[kSampleStateRepeatWindow];
    const uint32_t cursor = state[kSampleStateRecentCursor];
    uint32_t take = (window < stored) ? window : stored;

    if (penalty == 1.0f || take == 0u)
        return;
    for (uint32_t back = take; back > 0u; back--) {
        const uint32_t slot = (cursor + kSampleRecentCapacity - back) % kSampleRecentCapacity;
        const uint32_t token = recent[slot];

        for (uint32_t entry = 0u; entry < survivors->count; entry++) {
            if (survivors->ids[entry] != token)
                continue;
            if (survivors->values[entry] > 0.0f)
                survivors->values[entry] = survivors->values[entry] / penalty;
            else
                survivors->values[entry] = survivors->values[entry] * penalty;
        }
    }
}

/** Descending insertion sort of `work[0..count)`. At most 64 elements, so the O(n^2) is
 *  4096 compares in the worst case and it keeps the tie order the reference's qsort has no
 *  opinion about (equal values are interchangeable in a threshold). */
static void sort_descending(float* work, uint32_t count)
{
    for (uint32_t entry = 1u; entry < count; entry++) {
        const float value = work[entry];
        uint32_t position = entry;

        while (position > 0u && work[position - 1u] < value) {
            work[position] = work[position - 1u];
            position--;
        }
        work[position] = value;
    }
}

/** Softmax over the survivors, then the two nucleus filters, then the draw. Statement for
 *  statement `sparse_sample()`; the deviations are named where they occur. */
static uint32_t sample_stochastic(ftk_survivors *survivors, volatile FTQ_L1_PTR uint32_t *state,
                                  float temperature, float top_p, uint32_t top_k)
{
    const float inverse_temperature = 1.0f / temperature;
    float maximum;
    float sum = 0.0f;
    float inverse_sum;
    uint32_t rng_state = state[kSampleStateRngState];
    float draw;
    float cumulative = 0.0f;

    for (uint32_t entry = 0u; entry < survivors->count; entry++)
        survivors->values[entry] = survivors->values[entry] * inverse_temperature;

    maximum = survivors->values[0];
    for (uint32_t entry = 1u; entry < survivors->count; entry++)
        if (survivors->values[entry] > maximum)
            maximum = survivors->values[entry];

    for (uint32_t entry = 0u; entry < survivors->count; entry++) {
        survivors->probabilities[entry] = ftk_exp_nonpositive(survivors->values[entry] - maximum);
        sum = sum + survivors->probabilities[entry];
    }
    inverse_sum = 1.0f / sum;
    for (uint32_t entry = 0u; entry < survivors->count; entry++)
        survivors->probabilities[entry] = survivors->probabilities[entry] * inverse_sum;

    if (top_k > 0u && top_k < survivors->count) {
        float threshold;

        for (uint32_t entry = 0u; entry < survivors->count; entry++)
            survivors->work[entry] = survivors->probabilities[entry];
        sort_descending(survivors->work, survivors->count);
        threshold = survivors->work[top_k - 1u];
        for (uint32_t entry = 0u; entry < survivors->count; entry++)
            if (survivors->probabilities[entry] < threshold)
                survivors->probabilities[entry] = 0.0f;
    }

    if (top_p < 1.0f) {
        uint32_t candidates = 0u;
        float mass = 0.0f;
        float threshold = 0.0f;

        for (uint32_t entry = 0u; entry < survivors->count; entry++)
            if (survivors->probabilities[entry] > 0.0f)
                survivors->work[candidates++] = survivors->probabilities[entry];
        sort_descending(survivors->work, candidates);
        for (uint32_t entry = 0u; entry < candidates; entry++) {
            if (!(mass < top_p))
                break;
            if (survivors->work[entry] <= 0.0f)
                break;
            mass = mass + survivors->work[entry];
            threshold = survivors->work[entry];
        }
        for (uint32_t entry = 0u; entry < survivors->count; entry++)
            if (survivors->probabilities[entry] < threshold)
                survivors->probabilities[entry] = 0.0f;
    }

    sum = 0.0f;
    for (uint32_t entry = 0u; entry < survivors->count; entry++)
        sum = sum + survivors->probabilities[entry];
    if (sum > 0.0f) {
        inverse_sum = 1.0f / sum;
        for (uint32_t entry = 0u; entry < survivors->count; entry++)
            survivors->probabilities[entry] = survivors->probabilities[entry] * inverse_sum;
    }

    draw = ftk_rng_next_unit_float(&rng_state);
    state[kSampleStateRngState] = rng_state;
    for (uint32_t entry = 0u; entry < survivors->count; entry++) {
        cumulative = cumulative + survivors->probabilities[entry];
        if (draw < cumulative)
            return survivors->ids[entry];
    }
    // Every probability underflowed. The reference's full-vocabulary scan falls off the end
    // and returns token 0; matching that is deliberate, and it is NOT "the lowest survivor".
    for (uint32_t entry = survivors->count; entry > 0u; entry--)
        if (survivors->probabilities[entry - 1u] > 0.0f)
            return survivors->ids[entry - 1u];
    return 0u;
}

/** Append the token to the recent ring and the emitted ring, and evaluate the stop
 *  condition -- the state a closed decode loop carries so the host is not in the path
 *  (resident_topk_protocol.h §5). */
static void record_token(volatile FTQ_L1_PTR uint32_t *state, uint32_t token)
{
    volatile FTQ_L1_PTR uint32_t *recent = state + kSampleRecentOffsetWords;
    volatile FTQ_L1_PTR uint32_t *emitted = state + kSampleEmittedOffsetWords;
    const uint32_t recent_cursor = state[kSampleStateRecentCursor];
    const uint32_t emitted_count = state[kSampleStateEmittedCount];
    const uint32_t eos_count = state[kSampleStateEosCount];
    const uint32_t budget = state[kSampleStateTokenBudget];
    uint32_t stopped = 0u;

    recent[recent_cursor % kSampleRecentCapacity] = token;
    state[kSampleStateRecentCursor] = (recent_cursor + 1u) % kSampleRecentCapacity;
    if (state[kSampleStateRecentCount] < kSampleRecentCapacity)
        state[kSampleStateRecentCount] = state[kSampleStateRecentCount] + 1u;

    emitted[emitted_count % kSampleEmittedCapacity] = token;
    state[kSampleStateEmittedCount] = emitted_count + 1u;
    state[kSampleStateLastToken] = token;

    for (uint32_t entry = 0u; entry < eos_count && entry < kSampleEosCapacity; entry++)
        if (state[kSampleEosOffsetWords + entry] == token)
            stopped = 1u;
    if (budget != 0u && emitted_count + 1u >= budget)
        stopped = 1u;
    if (stopped)
        state[kSampleStateStopped] = 1u;
}

/**
 * Refuse a token hand-off this processor cannot issue LEGALLY, instead of issuing it anyway.
 *
 * The transfer is 16 bytes and both of its ends must be 16-byte aligned — `ft_tt_noc_alignment()`
 * gives a DRAM write the L1 alignment, in address AND length, and the source end is inside
 * the result block. The check is repeated here rather than left to the host for the same
 * reason the reader's is: `ft_tt_noc_check_transfer()` runs on the host, and the transfer is
 * issued HERE. See resident_topk_protocol.h's hand-off note for what a sub-alignment write
 * cost.
 */
static bool token_handoff_is_executable(ftq_context *ctx, uint32_t result_address,
                                        uint32_t token_out_base)
{
    if (token_out_base == 0u)
        return true;
    if ((token_out_base % kTopkTokenOutAlignment) != 0u ||
        (result_address % kTopkTokenOutAlignment) != 0u) {
        ctx->error = kErrBadParameters;
        return false;
    }
    return true;
}

/** The sampler half of one SAMPLE descriptor. */
static bool sample_one_token(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters)
{
    const uint32_t result_address = parameters[kTopkResultAddress];
    const uint32_t token_out_base = parameters[kTopkTokenOutBase];
    volatile FTQ_L1_PTR uint32_t *result =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(result_address);
    volatile FTQ_L1_PTR uint32_t *state =
        (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(parameters[kTopkStateAddress]);
    ftk_survivors survivors;
    float temperature;
    float top_p;
    float penalty;
    uint32_t token;

    // Before anything mutates the sampler state: a descriptor that cannot be finished must
    // not be half-executed, and the hand-off is the last thing this function does.
    if (!token_handoff_is_executable(ctx, result_address, token_out_base))
        return false;
    // BOTH wire-supplied pointers are dereferenced AND written below — `result` by the token
    // slot and the valid flag, `state` by record_token(). The magic check underneath is read
    // THROUGH the pointer it is meant to validate, so it cannot be the bounds check; see
    // descriptor_is_executable() above for why the second comparison (the read-only end) is
    // the one that keeps the core bootable.
    if (!ft_tensix_l1_block_is_writable(result_address, kTopkResultBytes) ||
        !ft_tensix_l1_block_is_writable(parameters[kTopkStateAddress], kSampleStateBytes)) {
        ctx->error = kErrBadParameters;
        return false;
    }
    if (state[kSampleStateMagic] != kSampleStateMagicValue) {
        ctx->error = kErrBadParameters;
        return false;
    }
    if (!gather_survivors(result, &survivors)) {
        ctx->error = kErrBadParameters;
        return false;
    }

    __builtin_memcpy(&temperature, (const void*)&state[kSampleStateTemperatureBits],
                     sizeof(temperature));
    __builtin_memcpy(&top_p, (const void*)&state[kSampleStateTopPBits], sizeof(top_p));
    __builtin_memcpy(&penalty, (const void*)&state[kSampleStateRepeatPenaltyBits], sizeof(penalty));

    if (temperature <= 0.0f) {
        // Greedy: NO repetition penalty, argmax on the raw values. The ordering is the
        // whole point -- see the block comment above.
        token = sample_greedy(&survivors);
    } else {
        apply_repeat_penalty(&survivors, state, penalty);
        token = sample_stochastic(&survivors, state, temperature, top_p, state[kSampleStateTopK]);
    }

    record_token(state, token);
    result[kTopkResultToken] = token;
    result[kTopkTokenSlotWords + 1u] = 0u;
    result[kTopkTokenSlotWords + 2u] = 0u;
    result[kTopkTokenSlotWords + 3u] = 0u;
    result[kTopkResultTokenValid] = 1u;

    if (token_out_base != 0u) {
        // A LEGAL TRANSFER, not an inline write: `noc_inline_dw_write` excludes DRAM
        // destinations by its own contract, and on Blackhole it degrades into a 4-byte write
        // that is below this backend's own 16-byte write alignment. Four bytes of it landed
        // correctly and left the core unbootable — resident_topk_protocol.h's hand-off note
        // has the measurement. The slot the id sits in IS the source, so there is no second
        // copy of the value to disagree with the one the host reads.
        ft_tensix_noc_write(result_address + kTopkTokenSlotWords * 4u,
                            ft_tensix_noc_address(parameters[kTopkTokenOutX],
                                                  parameters[kTopkTokenOutY],
                                                  token_out_base),
                            kTopkTokenSlotBytes);
        ft_tensix_noc_write_barrier();
    }
    return true;
}

void kernel_main(void)
{
    ftq_context ctx;
    uint32_t completed;
    bool running = true;

    ftq_context_init(&ctx);
    ftq_status_set(&ctx, kStateBoot);

    if (!ftq_check_control(&ctx)) {
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    completed = ctx.control[kFieldCompletedCount];
    ctx.cursor = completed;

    while (running) {
        // FETCH on `head`, so this processor drains the logit CB while the reader is still
        // filling it. Publishing waits on `tail`, below.
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t token = descriptor[kDescToken];
        const uint32_t group_sequence = descriptor[kDescA5];
        bool done = true;

        ftq_status_opcode(&ctx, opcode);
        if (opcode == kOpTopk) {
            ftq_status_set(&ctx, kStateCbWaitFront);
            done = select_one_logit_row(&ctx, ftq_parameters(descriptor));
            ftq_status_set(&ctx, kStateExec);
        } else if (opcode == kOpSample) {
            done = sample_one_token(&ctx, ftq_parameters(descriptor));
        } else if (opcode == kOpStop) {
            running = false;
        }

        if (!done) {
            running = false;
            continue;
        }

        // THE PRECONDITION, and the only place it belongs: the reader must have finished
        // this descriptor before its completion is published.
        if (!ftq_wait_until_reaches(&ctx, kFieldTail, ctx.cursor + 1u)) {
            running = false;
            continue;
        }

        ctx.cursor++;
        // Token before count: a host that sees the count is entitled to trust the token it
        // reads next. Both are plain stores from ONE in-order RISC-V.
        ctx.control[kFieldCompletedToken] = token;
        ctx.control[kFieldCompletedCount] = ++completed;
        ftq_group_post(&ctx, group_sequence, token);
    }

    // This processor does NOT touch kFieldError or kFieldState: the reader owns those, and
    // one writer per word is the whole ordering argument.
    ftq_status_set(&ctx, kStateExit);
}
