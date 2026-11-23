/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_queue.h
 * @brief The Phase 5 work-queue protocol: an L1 descriptor ring a RESIDENT kernel polls,
 *   so the host enqueues work instead of launching it. This is what removes per-op
 *   dispatch cost (spec §7.2).
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS REPLACES, AND WHY IT IS NOT THE K80 DESIGN
 * ---------------------------------------------------------------------------
 * On the K80, `fc_fused_multi` and CUDA graph capture exist to AMORTISE a launch that
 * cannot be removed: the driver owns the dispatch path, so the only lever is to make
 * fewer, bigger launches. Here the launch is ours. A Tensix is an independently
 * programmable RISC-V, so a kernel can simply not return — it sits on the core reading
 * descriptors out of its own L1. The host's per-op work becomes "write 32 bytes and bump
 * a counter", with NO round trip: it never waits for the card between ops.
 *
 * The consequence for anyone writing ops after this: **do not fuse ops to save
 * dispatch.** There is no dispatch to save. Fuse only when fusion avoids a round trip
 * through L1 or DRAM. Reintroducing an `fc_fused_multi` here would be porting a
 * workaround for a problem this backend does not have.
 *
 * ---------------------------------------------------------------------------
 * THE THREE RULES THIS PROTOCOL IS SHAPED BY
 * ---------------------------------------------------------------------------
 *
 * 1. **Completion must be produced BY the work item.** `ft_tt_launch_wait_done_all()`
 *    once reported eight cores complete with one of them never sent a go message,
 *    because `RUN_MSG_DONE` is 0 and is exactly what a core's own firmware boot leaves
 *    in the mailbox (dev notes §3). So nothing here retires work on a run-message
 *    handshake. The kernel writes `completed_count` and `completed_token` AFTER
 *    executing a descriptor, and the token is a per-run nonce the host put in that
 *    descriptor. A core that never ran holds FT_TT_QUEUE_STATE_HOST_POISON, which is
 *    neither zero nor any token the host will ever issue, so "it did nothing" and "it
 *    finished" are different bit patterns rather than the same one.
 *
 * 2. **A resident kernel is a kernel that blocks, and that is how a core was wedged.**
 *    Killing a kernel blocked in a CB wait with `assert_reset` left Tensix 0 hanging in
 *    `tensix_sync()` for every later program until a host reboot (dev notes §1). This
 *    phase's steady state is kernels sitting in a wait, so the host retires one by
 *    ASKING: `stop_request`, or a FT_TT_QUEUE_OP_STOP sentinel in the stream. **Every
 *    blocking wait in the resident kernel polls `stop_request` in the same loop** — that
 *    is what makes a deadlocked group recoverable without touching reset.
 *    `assert_reset` on a blocked resident kernel is the last resort, never the exit path.
 *
 * 3. **Residency is still BOUNDED.** `idle_spin_limit` makes the kernel exit on its own
 *    if the host stops feeding it. A host that segfaults must not leave a core spinning
 *    forever on a machine where the recovery is a walk to the card. This is a safety net,
 *    not the shutdown protocol: a normal exit goes through rule 2.
 *
 * ---------------------------------------------------------------------------
 * THE LAYOUT, PER CORE
 * ---------------------------------------------------------------------------
 *   FT_TT_OP_QUEUE_BASE + 0x0000   control block, 64 B (ft_tt_queue_control_field)
 *                      + 0x0040   per-processor status, 5 x 32 B (ft_tt_queue_status_field)
 *                      + 0x0100   descriptor ring, capacity x 32 B (capacity <= 64: ends by 0x900)
 *                      + 0x0900   S2 wait sidecar, 16 B a ring slot, then the reader's
 *                                 precondition words, to 0xE20 (kernels/resident_precondition_
 *                                 protocol.h; the host zeroes it at every arm)
 *                      + 0x1000   MoE expansion mailbox (kernels/resident_matmul_moe_protocol.h)
 *                      + 0x2000   parameter-cache overflow, 448 x 64 B (written ONCE)
 *                      + 0x9000   matmul parameter blocks, 64 x 64 B (ROTATING, rewritten)
 *                      + 0xA000   parameter cache, 320 x 64 B (written ONCE, pointed at)
 *                      + 0xF000   group completion block, 4 KiB
 *
 * `head` and `tail` are MONOTONIC totals, not wrapped indices: occupancy is
 * `head - tail`, the ring is full at `capacity`, and "how many are done" is a
 * subtraction rather than a case analysis. `capacity` is a power of two so the kernel
 * indexes with a mask.
 *
 * ---------------------------------------------------------------------------
 * THE ORDERING ASSUMPTION, STATED FOR A TRIAD
 * ---------------------------------------------------------------------------
 * The BRISC-only kernel could make one argument for everything: "these are plain stores
 * from ONE in-order RISC-V, so token-before-count and head-after-descriptors hold." A
 * resident TRIAD is five processors, and that sentence does not survive the move. What
 * replaces it is a SINGLE-WRITER TABLE plus two ordering claims that are each still within
 * one processor:
 *
 * | word                       | the ONLY writer            | read by                |
 * |----------------------------|----------------------------|------------------------|
 * | magic, capacity            | host, before launch        | all five               |
 * | head                       | host                       | reader, compute        |
 * | stop_request               | host                       | all five               |
 * | descriptor ring, params    | host, before head          | reader, compute, writer|
 * | tail                       | READER (BRISC)             | host, writer           |
 * | completed_count, _token    | WRITER (NCRISC)            | host                   |
 * | state, error, last_opcode  | READER (BRISC)             | host                   |
 * | status[p].*                | processor p                | host                   |
 * | group slot[m].*            | member m's WRITER          | host, other groups     |
 *
 * No word has two writers, so there is no store-order relationship BETWEEN processors to
 * establish and no fence to place. The two orderings that remain are both intra-processor:
 *
 *  1. **token before count**, both stores by the writer. A host that reads a satisfied
 *     count is entitled to the token beside it.
 *  2. **descriptors before head**, both writes by the host, in that program order through
 *     the same TLB window.
 *
 * Three hardware facts make that enough on this part, and all three are worth writing down
 * because the next person will otherwise re-derive them:
 *
 *  - **The RISC-V L1 data cache is DISABLED.** Blackhole has a small write-through L1 cache
 *    with no range registers, and a reader of another processor's write would have to
 *    invalidate. `configure_l1_data_cache()` (firmware_common.h) selects
 *    `set_l1_data_cache<false>()` unless `ENABLE_L1_DATA_CACHE` is defined, our builds do
 *    not define it, and in fact no firmware source calls the function at all -- so the
 *    cache is left at its reset default, which upstream's own comment states is DISABLED.
 *    DOCUMENTED, not measured. It is also why the shared kernel header keeps
 *    `invalidate_l1_cache()` inside the transliterated `cb_reserve_back` loop: that call is
 *    the one place upstream does not rely on the default, and dropping it would be a
 *    rare-and-looks-like-hardware hang.
 *  - **Relaxed memory ordering is LEFT ON**, and what it relaxes is loads bypassing stores
 *    to separate addresses (`disable_relaxed_memory_ordering()` is compiled out without
 *    `DISABLE_RELAXED_MEMORY_ORDERING`, bit 0 of CSR 0x7c0). That is a load-vs-store
 *    relaxation; the two claims above are STORE-vs-STORE, which it does not touch.
 *  - **A NoC write lands as a unit.** The group post is two words in ONE transaction, so a
 *    host that reads a sequence number is entitled to the token beside it without any
 *    cross-core ordering claim at all.
 *
 * The failure this table is designed against is concrete: if the WRITER published
 * completion while the READER was still performing a WRITE_WORD descriptor, "completed"
 * would again stop being evidence of work. That is why the writer's cursor follows `tail`
 * and not `head` -- the reader's half becomes a precondition rather than a race.
 *
 * @status Green. Off-hardware in tests/test_queue.c against a simulated L1 and a software
 *   model of the resident kernel; on the card in tests/test_phase5_resident.c (BRISC-only)
 *   and tests/test_phase6_triad.c (the resident triad, real matmul).
 * @issues None
 * @todo None
 */

#ifndef FT_TT_QUEUE_H
#define FT_TT_QUEUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_l1_layout.h"

/* ================================================================
 * Wire format
 * ================================================================ */

/** Words of one descriptor. 32 bytes, so a batch of them is a contiguous host write and
 *  the ring never straddles the 16-byte L1 write alignment. */
#define FT_TT_QUEUE_DESCRIPTOR_WORDS 8u
#define FT_TT_QUEUE_DESCRIPTOR_BYTES (FT_TT_QUEUE_DESCRIPTOR_WORDS * 4u)

/** Byte offset of the ring from the control block. 0x100 leaves the 64-byte control
 *  block its own space with room to grow without moving the ring. */
#define FT_TT_QUEUE_RING_OFFSET 0x100u

/** Control-block words, as indices into the 64-byte block. The kernel and the host both
 *  index this array; it is the whole shared-state contract. */
typedef enum ft_tt_queue_control_field {
    FT_TT_QUEUE_FIELD_MAGIC = 0,           /**< host: FT_TT_QUEUE_MAGIC */
    FT_TT_QUEUE_FIELD_CAPACITY = 1,        /**< host: ring slots, a power of two */
    FT_TT_QUEUE_FIELD_HEAD = 2,            /**< host: total descriptors enqueued */
    FT_TT_QUEUE_FIELD_TAIL = 3,            /**< kernel: total descriptors taken */
    FT_TT_QUEUE_FIELD_STOP_REQUEST = 4,    /**< host: 1 = leave the loop and return */
    FT_TT_QUEUE_FIELD_COMPLETED_COUNT = 5, /**< kernel: total descriptors EXECUTED */
    FT_TT_QUEUE_FIELD_COMPLETED_TOKEN = 6, /**< kernel: token of the last one executed */
    FT_TT_QUEUE_FIELD_STATE = 7,           /**< kernel: ft_tt_queue_state */
    FT_TT_QUEUE_FIELD_LAST_OPCODE = 8,     /**< kernel: opcode of the last one taken */
    FT_TT_QUEUE_FIELD_ERROR = 9,           /**< kernel: ft_tt_queue_kernel_error */
    FT_TT_QUEUE_FIELD_IDLE_SPINS = 10,     /**< kernel: spins in the current idle stretch */
    FT_TT_QUEUE_FIELD_HEARTBEAT = 11,      /**< kernel: ++ every poll — liveness, not progress */
    FT_TT_QUEUE_FIELD_COUNT = 16
} ft_tt_queue_control_field;

/** Bytes of the control block. */
#define FT_TT_QUEUE_CONTROL_BYTES (FT_TT_QUEUE_FIELD_COUNT * 4u)

/**
 * The WINDOW ft_tt_queue_check_completion() actually reads: COMPLETED_COUNT (5) through
 * ERROR (9), which covers count, token, state and error and nothing else.
 *
 * It is five words rather than sixteen because a control-block read is HOST PIO over a
 * transaction-bound link: the register accessor is UC/WORD32 by design (4-byte width is a
 * correctness requirement for run control), so a 64-byte read is 16 transactions at ~3.3 us
 * each — ~53 us — and a caller that checks completion once per member per op pays that per
 * MEMBER. Narrowing it to the window the check names is the same four assertions for a third
 * of the transactions, and it is the only field span that stays contiguous while doing so.
 */
#define FT_TT_QUEUE_COMPLETION_FIRST_FIELD FT_TT_QUEUE_FIELD_COMPLETED_COUNT
#define FT_TT_QUEUE_COMPLETION_FIELD_COUNT 5u

/* ----------------------------------------------------------------
 * Per-processor status — "the group is stuck" is not a diagnosis
 * ---------------------------------------------------------------- */

/**
 * With five processors on one queue, the control block can say the core is blocked and
 * cannot say WHICH processor is blocked in WHICH wait — and those are different bugs with
 * different fixes. Each processor therefore keeps its own slot.
 *
 * Eight words (32 B) a slot, so no two processors share a 16-byte L1 line while they are
 * bumping heartbeats, and so there is room for the two-word payload the group post is
 * staged in.
 */
typedef enum ft_tt_queue_status_field {
    FT_TT_QUEUE_STATUS_STATE = 0,       /**< ft_tt_queue_state, this processor's */
    FT_TT_QUEUE_STATUS_CURSOR = 1,      /**< descriptors this processor has taken */
    FT_TT_QUEUE_STATUS_HEARTBEAT = 2,   /**< ++ every poll — liveness, not progress */
    FT_TT_QUEUE_STATUS_ERROR = 3,       /**< ft_tt_queue_kernel_error, this processor's */
    FT_TT_QUEUE_STATUS_LAST_OPCODE = 4,
    FT_TT_QUEUE_STATUS_SPINS = 5,
    FT_TT_QUEUE_STATUS_PAYLOAD = 6,     /**< 2 words; the group post is staged here */
    FT_TT_QUEUE_STATUS_WORDS = 8
} ft_tt_queue_status_field;

/** Byte offset of the status array from the control block, and one slot's stride. */
#define FT_TT_QUEUE_STATUS_OFFSET 0x40u
#define FT_TT_QUEUE_STATUS_BYTES  (FT_TT_QUEUE_STATUS_WORDS * 4u)

/** The status slot of one FT_TT_PROC_* index, given the control-block address. */
#define FT_TT_QUEUE_STATUS_ADDRESS(control_address, processor) \
    ((control_address) + FT_TT_QUEUE_STATUS_OFFSET + (uint32_t)(processor) * FT_TT_QUEUE_STATUS_BYTES)

/* The status array must end before the ring begins: 0x40 + 5 slots x 32 B = 0xE0 < 0x100.
 * Written with literals because the preprocessor cannot see the enumerators above, and
 * checked again as a static assert against them in ft_tt_queue.c. */
#if (0x40u + 5u * 32u) > 0x100u
#error "the per-processor status array overlaps the descriptor ring"
#endif

/* ----------------------------------------------------------------
 * The matmul parameter block
 * ---------------------------------------------------------------- */

/**
 * A 32-byte descriptor carries six free words and a matmul shape needs fourteen, so
 * FT_TT_QUEUE_OP_MATMUL carries a POINTER to a parameter block instead — which is exactly
 * the shape docs/tt_bringup_log.md §17.8 note 2 says to reach for. The blocks live in the
 * queue region so an op author never has to find L1 for them, and 64 of them is deeper
 * than any batch the ring is used at.
 */
typedef enum ft_tt_queue_matmul_field {
    FT_TT_QUEUE_MATMUL_TILE_ROWS = 0,  /**< Mt */
    FT_TT_QUEUE_MATMUL_TILE_INNER = 1, /**< Kt */
    FT_TT_QUEUE_MATMUL_TILE_COLS = 2,  /**< Nt */
    FT_TT_QUEUE_MATMUL_A_X = 3,
    FT_TT_QUEUE_MATMUL_A_Y = 4,
    FT_TT_QUEUE_MATMUL_A_BASE = 5,
    FT_TT_QUEUE_MATMUL_B_X = 6,
    FT_TT_QUEUE_MATMUL_B_Y = 7,
    FT_TT_QUEUE_MATMUL_B_BASE = 8,
    FT_TT_QUEUE_MATMUL_C_X = 9,
    FT_TT_QUEUE_MATMUL_C_Y = 10,
    FT_TT_QUEUE_MATMUL_C_BASE = 11,
    FT_TT_QUEUE_MATMUL_DEPTH = 12,     /**< prefetch depth in K-tiles; must divide Kt */
    FT_TT_QUEUE_MATMUL_B_N_MAJOR = 13, /**< weight tile-grid order; a STATEMENT about DRAM */
    /** Non-zero when this matmul's A block is MULTICAST by the group's leader instead of
     *  read from DRAM by every member. Per DESCRIPTOR, where the role is per RESIDENCY,
     *  and the kernel refuses a descriptor whose word disagrees with its role — a group
     *  that cooperates on some descriptors and not others drifts its CB_IN0 write pointers
     *  apart and every later multicast lands in the wrong half of the receivers' buffers.
     *  kernels/resident_matmul_multicast_protocol.h is the contract. */
    FT_TT_QUEUE_MATMUL_COOPERATIVE = 14,
    /** Output columns the reader supplies per pass, and DST tiles the compute thread holds
     *  at once. 0 and 1 are the unblocked shape. It is a PER-RESIDENCY CONSTANT: CB_IN1's
     *  chunk becomes `depth * NB` pages and `ft_tensix_cb_block_grid_is_safe()` refuses a
     *  chunk size that CHANGES through a buffer, so every descriptor of one residency must
     *  carry the same value and the host must size CB_IN1 to `depth * NB * 2`.
     *  kernels/resident_queue_protocol.h carries the derivation. */
    FT_TT_QUEUE_MATMUL_COLUMN_BLOCK = 15,
    FT_TT_QUEUE_MATMUL_WORDS = 16
} ft_tt_queue_matmul_field;

/**
 * Word FT_TT_QUEUE_MATMUL_B_N_MAJOR bit 30: this descriptor's weight tiles are Bfp4_b (576 B)
 * instead of the residency's Bfp8_b (1,088 B). Mirror of the kernel's kMatmulWeightBfp4
 * (kernels/resident_queue_protocol.h; pinned by a literal assert in ft_tt_queue.c and by
 * tests/test_queue.c against the kernel header's text).
 *
 * ONLY `rtriad_mixw` reads it. Every other resident build ignores bit 30 and reads the tiles
 * at its own size, and DRAM tiles carry no header: a wrong flag is a wrong answer with a
 * clean run. So the host is the guard — ft_tt_queue_matmul_check_weight() before every send.
 */
#define FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4 (1u << 30)
/** The two weight tile sizes a mixw residency takes (ft_tt_op_l1_layout.h spells both). */
#define FT_TT_QUEUE_MATMUL_WEIGHT_TILE_BYTES_BFP8 1088u
#define FT_TT_QUEUE_MATMUL_WEIGHT_TILE_BYTES_BFP4 576u

/**
 * The bit-30 value of a weight's flag word, from its tile size.
 *
 * @param tile_bytes  the weight's tile size in DRAM
 * @param out_flag    receives 0 for 1,088 (Bfp8_b) or FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4 for
 *                    576 (Bfp4_b); untouched on error
 * @return FT_TT_OK, or FT_TT_EINVAL for any other size or a NULL out_flag
 */
int ft_tt_queue_matmul_weight_flag(uint32_t tile_bytes, uint32_t *out_flag);

/**
 * The pre-send check of a matmul parameter block's weight format: refuse a flag word whose
 * bit 30 disagrees with the weight's tile size, before the card can read it wrong.
 *
 * @param tile_bytes       the weight's tile size in DRAM
 * @param flag_word        the block's FT_TT_QUEUE_MATMUL_B_N_MAJOR word as it will be sent
 * @param mixw_residency   true only when the core runs `rtriad_mixw`
 * @return FT_TT_OK; FT_TT_EINVAL when bit 30 is set on a 1,088 B weight or clear on a 576 B
 *         one, when the size is neither, when a non-mixw residency is given a 576 B
 *         weight or bit 30 (it would ignore the flag and read 576 B tiles as 1,088 B ones),
 *         or when a mixw residency is given bit 31 (its reader refuses it)
 */
int ft_tt_queue_matmul_check_weight(uint32_t tile_bytes, uint32_t flag_word, bool mixw_residency);

/** Words in ONE parameter block, whichever op family owns it. The matmul was first and
 *  named it; the vector family (ft_tt_op_vector.h) uses the same sixteen words with its
 *  own field meanings, so the SLOT MACHINERY is shared and only the interpretation is per
 *  family. Sixteen is not a budget anyone has come near — the widest block in use spells
 *  fourteen. */
#define FT_TT_QUEUE_PARAM_WORDS 16u
#define FT_TT_QUEUE_MATMUL_PARAM_OFFSET 0x9000u
#define FT_TT_QUEUE_MATMUL_PARAM_BYTES  (FT_TT_QUEUE_MATMUL_WORDS * 4u)
#define FT_TT_QUEUE_MATMUL_PARAM_SLOTS  64u

/** The `slot`-th parameter block, given the control-block address. */
#define FT_TT_QUEUE_MATMUL_PARAM_ADDRESS(control_address, slot) \
    ((control_address) + FT_TT_QUEUE_MATMUL_PARAM_OFFSET + (uint32_t)(slot) * FT_TT_QUEUE_MATMUL_PARAM_BYTES)

/** Family-neutral spelling of the same address; prefer this in new op drivers. */
#define FT_TT_QUEUE_PARAM_ADDRESS(control_address, slot) \
    FT_TT_QUEUE_MATMUL_PARAM_ADDRESS(control_address, slot)

/* ----------------------------------------------------------------
 * The parameter CACHE — blocks written once and pointed at many times
 * ---------------------------------------------------------------- */

/**
 * A SECOND parameter region, whose blocks are never rewritten.
 *
 * ---------------------------------------------------------------------------
 * WHY, AND WHY IT IS A DIFFERENT REGION RATHER THAN MORE SLOTS
 * ---------------------------------------------------------------------------
 * The rotating slots above are sized by the RING: a chained projection must not overwrite
 * parameters the card has not read, so the slot is the descriptor's own index modulo the
 * count and every block is rewritten every few descriptors. That is correct and it is why
 * the host writes 64 bytes per member per projection.
 *
 * But a matmul's sixteen words are a function of `(weight, input strip, output strip,
 * member)` and every one of those is a CONSTANT of a decode run: the weights are placed
 * once at load, the activation strips are resident at fixed addresses, and the depth, the
 * column block and the cooperative role are per-residency. So the same sixteen words are
 * written to the card ~24 times a token, once per layer, forever.
 *
 * A block in THIS region is written the first time its content is seen and then only
 * pointed at. It is never rewritten and never evicted, so no in-flight descriptor can have
 * its parameters changed under it — which is exactly the property the rotating slots buy
 * with their rewrite, obtained here by not rewriting at all. A caller that runs out of
 * cache slots falls back to the rotating ones and is merely as expensive as it was.
 *
 * **It is CONTENT-addressed, not weight-addressed.** A cache keyed on a weight would be
 * wrong the moment the same weight were issued against a different strip, or under a red
 * arm that perturbs one word, and it would be wrong silently. Keyed on the sixteen words
 * themselves, a hit means the block at that address holds exactly those words, which is the
 * whole correctness claim and it is checkable by memcmp.
 *
 * The region is `0xA000..0xF000` of the queue block — the gap between the rotating slots
 * (which end at 0xA000) and the group completion block (0xF000). 320 blocks of 64 bytes,
 * and Ornith's whole matmul weight set is 249.
 */
#define FT_TT_QUEUE_PARAM_CACHE_OFFSET 0xA000u
#define FT_TT_QUEUE_PARAM_CACHE_PRIMARY_SLOTS 320u
/**
 * THE OVERFLOW REGION (qwen36 plan S2): `0x2000..0x9000`, past the largest ring (64 slots end
 * at 0x900) and the MoE expansion mailbox (0x1000, resident_matmul_moe_protocol.h), up to
 * the rotating slots. 448 more blocks. Qwen3.6's 40 layers need 321 blocks on a decode
 * member; one past 320 fell back to the rotating slots, whose address moves with the
 * descriptor index, so no two tokens were one shape and dispatch replay was refused.
 * Slots 320.. live here; nothing else in the queue block is used by any family.
 */
#define FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_OFFSET 0x2000u
#define FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_SLOTS 448u
#define FT_TT_QUEUE_PARAM_CACHE_SLOTS                                         \
    (FT_TT_QUEUE_PARAM_CACHE_PRIMARY_SLOTS + FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_SLOTS)

/** The `slot`-th cached parameter block, given the control-block address. */
#define FT_TT_QUEUE_PARAM_CACHE_ADDRESS(control_address, slot)                \
    ((uint32_t)(slot) < FT_TT_QUEUE_PARAM_CACHE_PRIMARY_SLOTS                 \
         ? (control_address) + FT_TT_QUEUE_PARAM_CACHE_OFFSET +               \
               (uint32_t)(slot) * FT_TT_QUEUE_MATMUL_PARAM_BYTES              \
         : (control_address) + FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_OFFSET +      \
               ((uint32_t)(slot) - FT_TT_QUEUE_PARAM_CACHE_PRIMARY_SLOTS) *   \
                   FT_TT_QUEUE_MATMUL_PARAM_BYTES)

/* `ft_tt_queue_write_param_cache()` is declared beside the rotating-slot writer, below the
 * `ft_tt_queue` typedef. */

/* ----------------------------------------------------------------
 * The group completion block
 * ---------------------------------------------------------------- */

/**
 * "The group finished item N", which per-core `completed_count` cannot express.
 *
 * It is a SLOT ARRAY and not a shared counter, and that is the whole design. A counter the
 * members increment says "N increments arrived" — which one member incrementing N times
 * satisfies, so it retires work the other members never did. That is the same class of lie
 * as `ft_tt_launch_wait_done_all()` reporting eight cores complete with one never sent a go
 * message (dev notes §3), one level up again. A slot per member says "every member reached
 * item N", each slot has exactly one writer, and the sequence and its token go in ONE NoC
 * transaction so there is nothing to order.
 *
 * The block lives on the group's FIRST core, which makes it readable by another GROUP on
 * the device as well as by the host — the reason to put it on a core at all rather than
 * summing on the host.
 */
typedef enum ft_tt_queue_group_field {
    FT_TT_QUEUE_GROUP_MAGIC = 0,
    FT_TT_QUEUE_GROUP_MEMBERS = 1,
    FT_TT_QUEUE_GROUP_HEADER_WORDS = 16 /**< slots start here */
} ft_tt_queue_group_field;

/** Words of one member's slot: [sequence, token]. */
#define FT_TT_QUEUE_GROUP_SLOT_WORDS 2u

/** Written by the host; the check refuses a block that does not carry it, which is what
 *  turns "the wrong address" into a refusal instead of a plausible number. */
#define FT_TT_QUEUE_GROUP_MAGIC_VALUE 0x46545147u /* "FTQG" */

#define FT_TT_QUEUE_GROUP_OFFSET 0xF000u

/** Most members one group block holds, from the 4 KiB left above the offset. */
#define FT_TT_QUEUE_GROUP_MAX_MEMBERS 256u

/** What one member's slot currently says: the last sequence its WRITER published, and the
 *  descriptor token beside it. Written in ONE NoC transaction, so the pair is consistent. */
typedef struct ft_tt_queue_group_slot {
    uint32_t sequence;
    uint32_t token;
} ft_tt_queue_group_slot;

/** Words of the block `members` need: the header plus two words each. */
uint32_t ft_tt_queue_group_block_words(uint32_t members);

/**
 * Write the group completion block at `group_address` on the LEADER's core: the magic, the
 * member count, and every slot zeroed.
 *
 * The coordinates are taken directly rather than from an ft_tt_core_group, because a
 * dispatcher whose members are not a contiguous run of Tensix indices still has a leader.
 * ft_tt_group_completion_init() is this function plus the group's own bookkeeping.
 *
 * @param members  1..FT_TT_QUEUE_GROUP_MAX_MEMBERS
 * @return FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM for too many members
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_queue_group_init(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           uint32_t group_address, uint32_t members);

/**
 * Read the header AND every member's slot, and refuse a block that does not carry the
 * magic — which is what turns "the wrong address" into a refusal instead of a plausible
 * number. Use it once, when arming; the header is 64 bytes a poll loop must not pay for.
 */
int ft_tt_queue_group_read(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           uint32_t group_address, uint32_t members,
                           ft_tt_queue_group_slot *out_slots);

/**
 * THE HOT-PATH READ: the slots ALONE, in one device access, skipping the 64-byte header.
 *
 * This is the function that makes a group's completion cost CONSTANT in its member count
 * instead of linear. Polling N members individually costs N reads of another core's
 * control block; polling the block costs `ceil(8 * members / 16)` accesses of ONE core,
 * because reads on this link are transaction-count bound and the slots are contiguous
 * (DEV_NOTES.md §6: 8-byte loads read 2.3 MB/s, 16-byte loads 4.7).
 *
 * It deliberately does NOT check the magic: the caller has already done that once at
 * ft_tt_queue_group_init() time, and re-reading 16 header words per poll is exactly the
 * cost this exists to remove. A caller that has not initialised the block sees zeroed
 * sequences, i.e. a timeout, which is the same answer one word later.
 *
 * The length is rounded UP to the accessor's 16-byte unit — an odd member count is 8 bytes
 * short of it — so a BLOCK16 accessor accepts it. One slot past the group is read and
 * ignored; the block is 4 KiB and cannot be run off.
 *
 * @param out_slots  at least `members` entries
 * @return FT_TT_OK, FT_TT_EINVAL, or the accessor's failure
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_queue_group_read_slots(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                 uint32_t group_address, uint32_t members,
                                 ft_tt_queue_group_slot *out_slots);

/** Written by the host into FT_TT_QUEUE_FIELD_MAGIC. The kernel refuses a queue whose
 *  magic it does not recognise rather than interpreting whatever L1 held before. */
#define FT_TT_QUEUE_MAGIC 0x46545131u /* "FTQ1" */

/** What the kernel is doing. The host writes POISON before the kernel starts, so a state
 *  that is still POISON is positive evidence the kernel never ran — the distinction
 *  `RUN_MSG_DONE` cannot make (dev notes §3). */
typedef enum ft_tt_queue_state {
    FT_TT_QUEUE_STATE_HOST_POISON = 0x7B1B1B1Bu,
    FT_TT_QUEUE_STATE_BOOT = 0x51010001u,
    FT_TT_QUEUE_STATE_IDLE = 0x51010002u,
    FT_TT_QUEUE_STATE_EXEC = 0x51010003u,
    FT_TT_QUEUE_STATE_SEM_WAIT = 0x51010004u,
    FT_TT_QUEUE_STATE_EXIT = 0x51010005u,
    /** Blocked in a circular-buffer wait. Named separately from EXEC because "which wait"
     *  is the diagnosis a five-processor hang needs and EXEC cannot give. */
    FT_TT_QUEUE_STATE_CB_WAIT_FRONT = 0x51010006u,
    FT_TT_QUEUE_STATE_CB_RESERVE_BACK = 0x51010007u
} ft_tt_queue_state;

/** The bounded-residency default, for a caller with no reason to choose. ~400M spins is
 *  seconds, not milliseconds: it is there for a host that DIED, not for flow control. */
#define FT_TT_QUEUE_DEFAULT_IDLE_SPINS 400000000u

/** Why the kernel left its loop. OK means "asked to". */
typedef enum ft_tt_queue_kernel_error {
    FT_TT_QUEUE_KERR_NONE = 0,
    FT_TT_QUEUE_KERR_BAD_MAGIC = 1,
    FT_TT_QUEUE_KERR_BAD_OPCODE = 2,
    FT_TT_QUEUE_KERR_IDLE_TIMEOUT = 3,
    FT_TT_QUEUE_KERR_SEM_WAIT_TIMEOUT = 4,
    FT_TT_QUEUE_KERR_BAD_CAPACITY = 5,
    /** A circular-buffer wait exhausted its spin bound. The bound exists so that this is
     *  what a stuck triad produces instead of a core only a reboot can retire. */
    FT_TT_QUEUE_KERR_CB_WAIT_TIMEOUT = 6,
    FT_TT_QUEUE_KERR_BAD_PARAMETERS = 7,
    /* The COOPERATIVE matmul's five, mirroring kernels/resident_queue_protocol.h. Every
     * one means "this group is no longer in lockstep": the host's answer is to retire and
     * re-arm the whole group, never to retry the descriptor. */
    FT_TT_QUEUE_KERR_COOP_ROLE_MISMATCH = 8,
    FT_TT_QUEUE_KERR_COOP_WRITE_POINTER = 9,
    FT_TT_QUEUE_KERR_COOP_RENDEZVOUS = 10,
    FT_TT_QUEUE_KERR_COOP_ISSUE = 11,
    FT_TT_QUEUE_KERR_COOP_BAD_PARTITION = 12
} ft_tt_queue_kernel_error;

/**
 * What a descriptor asks for.
 *
 * Every opcode is either non-blocking or polls `stop_request` while it blocks; there is
 * no third kind, because a third kind is a core that only `assert_reset` can retire.
 */
typedef enum ft_tt_queue_opcode {
    /** Nothing. The dispatch-cost measurement: its wall time is the protocol's own. */
    FT_TT_QUEUE_OP_NOP = 0x01u,
    /** a0 = L1 address in THIS core, a1 = value. Verifiable work with no NoC involved. */
    FT_TT_QUEUE_OP_WRITE_WORD = 0x02u,
    /** a0,a1 = destination NoC x,y; a2 = destination address; a3 = source address in this
     *  core's L1; a4 = bytes. One `noc_async_write` plus a barrier. */
    FT_TT_QUEUE_OP_L1_COPY = 0x03u,
    /** a0,a1 = NoC x,y of the semaphore's core; a2 = its L1 address; a3 = increment.
     *  Atomic, so several groups may post to one semaphore. */
    FT_TT_QUEUE_OP_SEM_POST = 0x04u,
    /** a0 = semaphore address in THIS core's L1; a1 = value to wait for; a2 = spin bound,
     *  0 for the kernel's idle_spin_limit. BLOCKS — and polls `stop_request` each spin,
     *  which is what makes a cross-group deadlock recoverable by asking. */
    FT_TT_QUEUE_OP_SEM_WAIT = 0x05u,
    /** a0 = iterations of a volatile spin. Makes a group measurably busy. */
    FT_TT_QUEUE_OP_DELAY = 0x06u,
    /** Leave the loop. The ORDERED shutdown: everything queued before it runs first. */
    FT_TT_QUEUE_OP_STOP = 0x07u,
    /** A REAL op on the resident TRIAD: a0 = L1 address of a parameter block
     *  (FT_TT_QUEUE_MATMUL_PARAM_ADDRESS), a1 = a DEBUG count of trailing K-blocks the
     *  reader deliberately withholds so the compute thread is left blocked in the op's own
     *  `cb_wait_front` — 0 for every real request. Only the triad kernels implement it;
     *  the BRISC-only Phase 5 kernel refuses it as a bad opcode, which is correct. */
    FT_TT_QUEUE_OP_MATMUL = 0x08u,
    /** The VECTOR op family on the resident triad: a0 = L1 address of a parameter block
     *  (FT_TT_QUEUE_PARAM_ADDRESS), laid out by ft_tt_op_vector.h. ELTWISE covers SiLU,
     *  sigmoid, add, mul, sub and the SiLU-gated multiply in ONE binary — which function
     *  runs is a word in the block, not a different opcode, because a resident binary is
     *  bound to one slab and a core that has to do both halves of a SwiGLU cannot afford
     *  to reload five images between them. */
    FT_TT_QUEUE_OP_ELTWISE = 0x09u,
    /** Row-wise softmax over a [rows x W] activation, on its own resident triad.
     *  SUPERSEDED by FT_TT_QUEUE_OP_REDUCE, which answers it as one member of the reduce
     *  family; no resident program implements this value any more. */
    FT_TT_QUEUE_OP_SOFTMAX = 0x0Au,
    /** The REDUCE op family on the resident triad: a0 = L1 address of a parameter block
     *  (FT_TT_QUEUE_PARAM_ADDRESS), laid out by ft_tt_op_vector.h. Softmax, RMS norm, layer
     *  norm and the MoE router normalise are ONE binary — they are one skeleton (row reduce
     *  into DST column 0, an SFPU verb on that column, broadcast back with *_bcast_cols) and
     *  differ only in a pass count and two verbs, which are DESCRIPTOR FIELDS. A transformer
     *  block norms, softmaxes and norms again on one core; as separate binaries that is two
     *  ~0.19 ms image reloads per block. */
    FT_TT_QUEUE_OP_REDUCE = 0x0Bu,
    /** Gather rows of a ROW-MAJOR embedding table by token id into a face-ordered
     *  activation: a0 = L1 address of a parameter block (FT_TT_QUEUE_PARAM_ADDRESS), laid
     *  out by ft_tt_op_memory.h. */
    FT_TT_QUEUE_OP_EMBEDDING = 0x0Cu,
    /** Append this step's K or V into the KV cache at a given position. Same parameter
     *  block, same binary and same shape as EMBEDDING — both are "place rows into a
     *  face-ordered tile destination" — and the ONE resident program that answers them is
     *  the DEGENERATE triad: a reader and a writer, NO compute thread, enables DM0|DM1.
     *  **ft_tt_op_memory.h PART 1 is the KV memory layout contract flash-decode attention
     *  is written against.** */
    FT_TT_QUEUE_OP_KV_APPEND = 0x0Du,
    /** Partial NeoX rotary position embedding on the resident triad: a0 = L1 address of a
     *  parameter block (FT_TT_QUEUE_PARAM_ADDRESS), laid out by ft_tt_op_rope.h. Its own
     *  opcode and its own binary because its READER implements the rotation's PAIRING by
     *  addressing a second, permuted tile stream — no other reader here does that — and its
     *  sin/cos table is a fourth streamed operand. */
    FT_TT_QUEUE_OP_ROPE = 0x0Eu,
    /** Exact top-N over one vocabulary logit row: a0 = L1 address of a parameter block
     *  (FT_TT_QUEUE_PARAM_ADDRESS), laid out by ft_tt_op_sample.h. Writes N (index, value)
     *  pairs into an L1 result block on the SAME core; nothing crosses the host link. */
    FT_TT_QUEUE_OP_TOPK = 0x0Fu,
    /** Sample ONE token id from a result block FT_TT_QUEUE_OP_TOPK already produced, using
     *  the per-turn sampler state that lives beside it. Same parameter-block layout. The
     *  only thing this op ever hands back is a 4-byte token id, which is the whole reason
     *  decode can close on the card (dev notes §6: card->host is 2.3-4.7 MB/s). */
    FT_TT_QUEUE_OP_SAMPLE = 0x10u,
    /** Flash-decode attention over the KV cache for ONE GQA group at one decode position:
     *  a0 = L1 address of a parameter block (FT_TT_QUEUE_PARAM_ADDRESS), laid out by
     *  ft_tt_op_attention.h. Its own binary rather than a member of a family because it is
     *  the only op here that carries state ACROSS its input blocks — a running max, row sum
     *  and output accumulator — which is what an ONLINE softmax is, and what makes a 32k
     *  context a fixed L1 cost instead of an impossible one (docs/tt_bringup_log.md §22.3:
     *  a standalone softmax tops out at Wt = 325 and a 32k row is Wt = 1024). */
    FT_TT_QUEUE_OP_ATTENTION = 0x11u,
    /** The whole Gated-Delta-Net block for ONE decode token of ONE layer: a0 = L1 address
     *  of a parameter block (FT_TT_QUEUE_PARAM_ADDRESS), laid out by ft_tt_op_gdn.h.
     *
     *  ONE FUSED DESCRIPTOR PER LAYER, and that is the whole reason it is an opcode rather
     *  than a composition of ops that already exist. The block is conv1d+SiLU, a per-head
     *  L2 norm on q and k, a GQA repeat and a four-pass delta rule over 32 v-heads x 4
     *  tile-rows. Issued as separate descriptors that is ~3,800 dispatches a token, which
     *  costs more than the 0.225 s of host recurrence it exists to replace.
     *
     *  It is also the only op here that carries state across TOKENS — the conv window and
     *  a [128,128] fp32 recurrent state per head — a stronger version of what ATTENTION's
     *  comment above says about carrying state across input blocks. A wrong answer does
     *  not stay wrong for one token, it compounds, so the acceptance gate is a 48-token
     *  completion rather than a single step. */
    FT_TT_QUEUE_OP_GDN = 0x12u,
    /** The CHUNKWISE Gated-Delta-Net block for C POSITIONS of ONE layer: a0 = L1 address of
     *  a parameter block, laid out by ft_tt_op_gdnc.h.
     *
     *  A SEPARATE OPCODE AND A SEPARATE PROGRAM FROM FT_TT_QUEUE_OP_GDN, because the two
     *  cannot share a reader. The decode form is an elementwise multiply over a `[1, N]`
     *  ROW-0 STRIP; the chunk form is a BANDED MATMUL over a `[C, N]` strip whose row index
     *  IS the position, so the reader stages different shapes, the compute image needs the
     *  matmul replay buffer, and the writer rolls a window it has to take ROWS out of rather
     *  than whole tiles.
     *
     *  The two families share the WINDOW BYTES exactly -- `taps-1` tap-major row-0 planes --
     *  which is what makes a prompt that prefills in chunks and then decodes one token at a
     *  time one sequence with no transcription at the seam. */
    FT_TT_QUEUE_OP_GDN_CHUNK = 0x13u,
    /** STEP 3 (A): the on-card PULLER (kernels/resident_puller.c) replays recorded segments
     *  of a decode token's queue traffic; a0 -> {program, k, first, last segment, flags}. */
    FT_TT_QUEUE_OP_PULL = 0x14u,
    /** STEP 4 S2: the puller's verify row mover (kernels/resident_puller.c). */
    FT_TT_QUEUE_OP_ROWCOPY = 0x15u,
    /** PM 7 ITEM 5: the GDN gates (decay, beta) computed on the card from the alpha/beta
     *  projections with ft_det_math.h (kernels/resident_gdngate_protocol.h). */
    FT_TT_QUEUE_OP_GDN_GATE = 0x16u
} ft_tt_queue_opcode;

/** One work item, host-side. Mirrors the eight wire words exactly. */
typedef struct ft_tt_queue_descriptor {
    uint32_t opcode;
    uint32_t token; /**< echoed into FT_TT_QUEUE_FIELD_COMPLETED_TOKEN after execution */
    uint32_t a0;
    uint32_t a1;
    uint32_t a2;
    uint32_t a3;
    uint32_t a4;
    /** GROUP SEQUENCE NUMBER, or 0 for "post nothing". The kernel that publishes this
     *  descriptor's completion also writes {a5, token} into this core's slot of the group
     *  block, which is how a group expresses "every member finished item N". */
    uint32_t a5;
} ft_tt_queue_descriptor;

/* ================================================================
 * Host side
 * ================================================================ */

/** The host's view of one core's queue. Caller-owned, like everything else here. */
typedef struct ft_tt_queue {
    uint16_t x; /**< TRANSLATED coordinate, from ft_tt_coord — never a guess */
    uint16_t y;
    uint32_t control_address; /**< L1 address of the control block */
    uint32_t ring_address;    /**< L1 address of slot 0 */
    uint32_t capacity;        /**< ring slots, a power of two */
    uint32_t head;            /**< host's own copy; the device copy is written by _publish */
    uint32_t staged;          /**< descriptors written but not yet published to the device */
    uint32_t run_nonce;       /**< seeds every token, so a stale L1 word cannot match */
    uint32_t issued;          /**< tokens issued so far this run */
} ft_tt_queue;

/**
 * Bind a queue to a core and WRITE its initial control block: magic, capacity, head 0,
 * tail 0, stop 0, completed 0, and FT_TT_QUEUE_STATE_HOST_POISON into the state word.
 *
 * The poison is load-bearing. It is what makes "the kernel never ran" a different value
 * from "the kernel finished", instead of both being the zero a cleared mailbox holds.
 *
 * Call this BEFORE the kernel is launched — it is the kernel's first read.
 *
 * @param noc              device access
 * @param x, y             TRANSLATED coordinates from ft_tt_coord
 * @param control_address  4-byte aligned; FT_TT_OP_QUEUE_BASE is the normal answer
 * @param capacity         ring slots; must be a power of two, at least 2
 * @param run_nonce        any non-zero value unique to this run; ft_tt_queue_make_nonce()
 * @param out_queue        zeroed and filled on success
 * @return                 FT_TT_OK; FT_TT_EINVAL on a bad argument; the accessor's code
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_queue_init(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                     uint32_t control_address, uint32_t capacity, uint32_t run_nonce,
                     ft_tt_queue *out_queue);

/** A nonce for ft_tt_queue_init(), mixed from the clock and the pid. Never 0, and never
 *  FT_TT_QUEUE_STATE_HOST_POISON. */
uint32_t ft_tt_queue_make_nonce(void);

/**
 * The token ft_tt_queue_enqueue() will stamp on the `index`-th descriptor of this run.
 *
 * The caller needs this to check completion, and it is derived rather than remembered so
 * a test cannot accidentally check the token it was TOLD instead of the one that was
 * SENT.
 */
uint32_t ft_tt_queue_token_at(const ft_tt_queue *queue, uint32_t index);

/**
 * Write `count` descriptors into the ring and bump the device's head ONCE.
 *
 * This is the whole per-op cost of the dispatch model, and the reason it is a batch call
 * rather than a per-descriptor one: the descriptors are contiguous in the ring, so N of
 * them are ONE device write, and the head bump is one more. There is no completion poll
 * here and no wait — the host does not round-trip the card to dispatch work.
 *
 * A batch that would wrap the ring is split into two writes, which is a detail of this
 * function and not of the protocol.
 *
 * Each descriptor's `token` field is OVERWRITTEN with ft_tt_queue_token_at(), so a caller
 * cannot accidentally send a token that a previous run could also have produced.
 *
 * @param noc          device access
 * @param queue        updated in place
 * @param descriptors  `count` items
 * @param count        at most the free space, i.e. capacity - (head - tail_seen)
 * @return             FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if `count` exceeds capacity
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_queue_enqueue(const ft_tt_noc_access *noc, ft_tt_queue *queue,
                        ft_tt_queue_descriptor *descriptors, uint32_t count);

/**
 * The descriptor half of ft_tt_queue_enqueue(): write them into the ring, stamp their
 * tokens, and do NOT bump the device's head. The kernel cannot see them yet.
 *
 * Split out for two reasons. A caller building a batch bigger than the contiguous run
 * left in the ring can stage twice and publish once. And it is what makes the DEVICE-SIDE
 * op-to-op measurement possible: with the descriptors already in L1, the only host action
 * inside the timed window is ft_tt_queue_publish()'s single 4-byte write.
 *
 * @return FT_TT_OK; FT_TT_EINVAL; FT_TT_ENOMEM if `count` exceeds capacity
 */
int ft_tt_queue_stage(const ft_tt_noc_access *noc, ft_tt_queue *queue,
                      ft_tt_queue_descriptor *descriptors, uint32_t count);

/**
 * Bump the device's head to cover everything staged. ONE 4-byte write.
 *
 * Until this lands the kernel cannot see a half-written descriptor, which is the whole
 * ordering requirement of the protocol.
 */
int ft_tt_queue_publish(const ft_tt_noc_access *noc, ft_tt_queue *queue);

/**
 * Can this queue's PARAMETER blocks and ring slots be written through the BLOCK16 L1
 * accessor — i.e. is the `wide` issue shape available on it?
 *
 * The 64-byte parameter block (both regions) and the 32-byte ring slot are 16-byte aligned in
 * address and length whenever the control block is, so this is one question about placement.
 * The HEAD BUMP is four bytes at an arbitrary word and can never come through that accessor,
 * which is why `wide` is structurally two accessors and not one.
 *
 * Asked of `ft_tt_noc_check_transfer()` — the function the real accessor calls — so the
 * predicate and the refusal it predicts cannot drift.
 */
bool ft_tt_queue_issue_is_wide_addressable(const ft_tt_queue *queue);

/**
 * The raw head word ft_tt_queue_publish() is built on: ONE 4-byte write of `head`, with no
 * staging bookkeeping and no check that anything has been written into the ring.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS PUBLIC, AND WHAT IT IS FOR
 * ---------------------------------------------------------------------------
 * "Descriptors before head" is one of the two intra-processor orderings this whole protocol
 * rests on (see the ordering note at the top of this file). An ordering claim that cannot be
 * INVERTED cannot be tested: a caller that only ever has ft_tt_queue_enqueue() can show that
 * the right order works and can never show that the completion check would notice the wrong
 * one. This is the primitive that lets an arm publish a head for descriptors that have not
 * been written, so the four-agreement check in ft_tt_queue_check_completion() can be seen to
 * REFUSE — which is what makes a clean run on the real order evidence rather than an absence.
 *
 * It does not touch `queue->head` or `queue->staged`, so a caller that uses it must follow it
 * with the ordinary stage/publish pair to leave the bookkeeping consistent.
 *
 * **Not a fast path and not a shortcut.** Every ordinary caller wants ft_tt_queue_enqueue()
 * or the stage/publish pair, both of which own the ordering for you.
 *
 * @param noc    device access
 * @param queue  read only; the head value is the caller's
 * @param head   the monotonic total to publish
 * @return       FT_TT_OK; FT_TT_EINVAL
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_queue_write_head(const ft_tt_noc_access *noc, const ft_tt_queue *queue, uint32_t head);

/** The control block as the device currently holds it. Reads only, so it is safe against
 *  a core that is stuck — which is when it is wanted. */
int ft_tt_queue_read_control(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                             uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT]);

/** One control word, without reading the whole block. Used by the poll loops, where the
 *  read direction costs ~3.5 µs a go and reading 64 bytes to look at one word is waste. */
/**
 * Read only the fields ft_tt_queue_check_completion() reads, and zero the rest.
 *
 * A third of the PIO transactions of ft_tt_queue_read_control() for the same four
 * assertions. Use this on the hot path and the full read when reporting a failure, where
 * HEAD/TAIL/IDLE_SPINS are what a human needs and one extra read costs nothing.
 *
 * @param out_fields  FT_TT_QUEUE_FIELD_COUNT words; only the completion window is filled
 * @return FT_TT_OK, FT_TT_EINVAL, or the accessor's failure
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_queue_read_completion(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT]);

/**
 * The SAME four assertions in TWO device accesses instead of five, by asking a 16-byte-wide
 * accessor for a 16-byte-aligned span that contains the completion window.
 *
 * Reads on this link are transaction-count bound, not byte bound: 4-byte
 * loads read 2.3 MB/s and 16-byte loads 4.7. The completion window is words 5..9, which does
 * not start on a 16-byte boundary — so this reads words 4..11 (offset 16, 32 bytes), two
 * BLOCK16 transactions, and fills exactly those eight fields. Everything
 * ft_tt_queue_check_completion() names is inside it.
 *
 * @param noc  MUST be a BLOCK16 accessor over L1 (ft_tt_runtime.h `bulk_l1`). A WORD32
 *             accessor gives the right answer for eight transactions, which is the opposite
 *             of the point; an accessor bound to DRAM refuses the address outright.
 * @param out_fields  FT_TT_QUEUE_FIELD_COUNT words; only words 4..11 are filled
 * @return FT_TT_OK, FT_TT_EINVAL, or the accessor's failure
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_queue_read_completion_wide(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                     uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT]);

/**
 * THREE of the four assertions in ONE transaction — the completion LINE.
 *
 * ---------------------------------------------------------------------------
 * WHY A THIRD READ EXISTS, AND WHAT IT DELIBERATELY DROPS
 * ---------------------------------------------------------------------------
 * The wide read is two transactions because the four assertion words are 5, 6, 7 and 9, and
 * words 4..11 is the smallest 16-byte-aligned span that holds all four. Words 4..7 — bytes
 * 16..32 — is ONE 16-byte L1 line and holds `completed_count`, `completed_token` and
 * `state`. `error` (word 9) is in the NEXT line and is the only reason the wide read is
 * twice this one.
 *
 * A caller that also uses this AS ITS POLL therefore pays ONE transaction per member per
 * wait where the poll-then-check pair pays three, and that ratio is the whole reason this
 * function exists: the per-member host term is transaction-count bound and this is its
 * floor without a cross-core aggregation mechanism.
 *
 * **WHAT IS GIVEN UP, stated rather than hidden.** A kernel that REFUSES a descriptor does
 * not publish completion, so `completed_count` never reaches `expected`, so a refusal is
 * caught by the caller's timeout path — which reads `error` and names it. The only state in
 * which `error` is set AND completion is correct is a fault raised AFTER `completed_count`
 * was published, and the protocol has exactly one of those (`ftq_group_post()`'s refusal,
 * `resident_queue_protocol.h`). So this read is complete for every queue that does not arm
 * the group post, and a caller that does arm it must keep the wide read.
 *
 * It is also STRICTLY BETTER ORDERED than poll-then-check: the count and the token come out
 * of the same 16 bytes of the same transaction, so there is no window between reading the
 * count and reading the token the count entitled you to.
 *
 * @param noc  MUST be a BLOCK16 accessor over L1 (ft_tt_runtime.h `bulk_l1`).
 * @param out_fields  FT_TT_QUEUE_FIELD_COUNT words; only words 4..7 are filled, the rest
 *                    are ZEROED — so a caller that reaches for ERROR out of this buffer
 *                    reads KERR_NONE, which is why the check below refuses to look at it.
 * @return FT_TT_OK, FT_TT_EINVAL, or the accessor's failure
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_queue_read_completion_line(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                     uint32_t out_fields[FT_TT_QUEUE_FIELD_COUNT]);

/**
 * `ft_tt_queue_check_completion()` minus the `error` assertion, over a buffer that does not
 * carry `error`.
 *
 * It is a SEPARATE function rather than a flag on the existing one so that a caller cannot
 * hand a line read to the four-assertion check and have the zeroed `error` word silently
 * pass as KERR_NONE. That failure would be invisible — the buffer is memset — and this
 * project has already paid twice for a check that could not fail.
 *
 * @param out_reason  may be NULL; a static string naming the first disagreement
 * @return            FT_TT_OK, or FT_TT_ERROR with `out_reason` set
 */
int ft_tt_queue_check_completion_line(const ft_tt_queue *queue,
                                      const uint32_t fields[FT_TT_QUEUE_FIELD_COUNT],
                                      uint32_t expected, const char **out_reason);

/**
 * Can `ft_tt_queue_read_completion_line()` reach this queue AT ALL through a BLOCK16 L1
 * accessor? Asked BEFORE a caller commits to the fast path, not after.
 *
 * The line starts at `control_address + 16`, and a BLOCK16 L1 read is refused unless BOTH
 * the address and the length are 16-byte multiples. `ft_tt_queue_init()` only requires
 * 4-byte alignment, so a queue placed at a word-aligned address is legal, works through the
 * WORD32 accessor, and would make every fast read fail — which on the resident path arrives
 * as a completion that never completes, i.e. a retire-and-re-arm storm rather than a
 * diagnosable refusal. One boolean, asked once at arm time, is the difference.
 *
 * @return true when the line is addressable; false for a NULL queue or a misaligned one
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
bool ft_tt_queue_completion_line_is_addressable(const ft_tt_queue *queue);

int ft_tt_queue_read_field(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                           ft_tt_queue_control_field field, uint32_t *out_value);

/**
 * Ask the resident kernel to leave its loop, out of band.
 *
 * This is the recovery path, and it is a WRITE of one word — never a reset. It is seen by
 * the poll loop AND by every blocking opcode, so it retires a kernel that is deadlocked
 * on a semaphore as well as one that is merely idle.
 */
int ft_tt_queue_request_stop(const ft_tt_noc_access *noc, const ft_tt_queue *queue);

/** The value `stop_request` must hold to retire a resident kernel. */
#define FT_TT_QUEUE_STOP_REQUESTED 1u

/**
 * The L1 address of this queue's `stop_request` word — the one word worth handing to
 * tt-kmd's SET_NOC_CLEANUP so a process that dies abnormally still retires this core.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS WORD, DERIVED RATHER THAN ASSUMED
 * ---------------------------------------------------------------------------
 * tt-kmd performs exactly ONE 32-bit NoC write for us when our fd closes, for any reason
 * including `SIGKILL`, a segfault or the OOM killer. Which word buys the most is decidable,
 * because the candidate has to satisfy four things at once and only this one does:
 *
 *  1. **All five processors already read it, in every wait they can be in.** Rule 2 of this
 *     file's note: the reader's poll loop and EVERY blocking opcode test `stop_request` in
 *     the same loop as their own condition. No other control word is read by all five, and a
 *     word only one of them reads leaves the other four spinning.
 *  2. **Writing it IS the sanctioned retirement path**, not a new one — it is exactly what
 *     `ft_tt_queue_request_stop()` does. The driver then performs an operation the kernel is
 *     built to receive rather than an intervention invented for the death case. The
 *     alternative, `assert_reset`, is what has wedged cores here three times.
 *  3. **It is idempotent and harmless when nothing is running.** On a core whose kernel has
 *     already left it is a store to L1 nobody reads, and the next `ft_tt_queue_init()`
 *     clears it before the next launch. That matters because the driver fires the write on
 *     EVERY close, including clean ones.
 *  4. **It is ONE word.** `head`, the ring and the run nonce each need several, and a
 *     partial write of any of them is worse than no write at all.
 *
 * The honest half. The driver's write only ASKS: it cannot then wait for the five processors
 * to reach EXIT, and it cannot re-init the ring. So this converts "released after
 * `idle_spin_limit`" into "released immediately", and the next process still owes exactly
 * the boot-time care it always did.
 *
 * And it protects ONE core: tt-kmd keeps a single registration per fd (a later call replaces
 * an earlier one), so a process holding several residencies must choose one and say which.
 * The rest keep the bounded-residency backstop and nothing more.
 *
 * @return the address, or 0 when `queue` is NULL
 *
 * Thread-safe: yes (read-only)    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_queue_stop_request_address(const ft_tt_queue *queue);

/**
 * Poll until `completed_count` reaches `target`, or until `timeout_ms` elapses.
 *
 * `target` is a count of items the KERNEL executed. It cannot be satisfied by a core that
 * never ran, and it cannot be satisfied by the run-message handshake — which is the
 * entire point (rule 1 in the file note).
 *
 * @param out_completed  may be NULL; the last value read, on success or timeout
 * @return               FT_TT_OK; FT_TT_EIO on timeout; FT_TT_EINVAL for timeout_ms 0
 */
int ft_tt_queue_wait_completed(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                               uint32_t target, uint32_t timeout_ms, uint32_t *out_completed);

/**
 * Check that a finished run really was finished BY THE KERNEL.
 *
 * Four independent things have to agree, and each of them fails a different lie:
 *   - `state` is not HOST_POISON     — the kernel ran at all
 *   - `completed_count == expected`  — it executed every item
 *   - `completed_token == ft_tt_queue_token_at(queue, expected - 1)` — it executed THIS
 *     run's last item, not a previous run's residue
 *   - `error` is FT_TT_QUEUE_KERR_NONE
 *
 * @param out_reason  may be NULL; a static string naming the first disagreement
 * @return            FT_TT_OK, or FT_TT_ERROR with `out_reason` set
 */
int ft_tt_queue_check_completion(const ft_tt_queue *queue,
                                 const uint32_t fields[FT_TT_QUEUE_FIELD_COUNT],
                                 uint32_t expected, const char **out_reason);

/** "IDLE", "EXEC", ... for logs. Never NULL. */
const char *ft_tt_queue_state_string(uint32_t state);

/** "bad opcode", ... for logs. Never NULL. */
const char *ft_tt_queue_kernel_error_string(uint32_t error);

/** "SEM_WAIT", ... for logs. Never NULL. */
const char *ft_tt_queue_opcode_string(uint32_t opcode);

/* ================================================================
 * Per-processor status
 * ================================================================ */

/**
 * Read one processor's status slot.
 *
 * Reads only, so it is safe against a core that is stuck — which is when it is wanted.
 *
 * @param processor   FT_TT_PROC_* index
 * @param out_fields  FT_TT_QUEUE_STATUS_WORDS words
 */
int ft_tt_queue_read_status(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                            unsigned processor, uint32_t out_fields[FT_TT_QUEUE_STATUS_WORDS]);

/**
 * Has every processor in `processor_mask` reported FT_TT_QUEUE_STATE_EXIT?
 *
 * A triad is retired only when ALL FIVE have left their loops, and the control block cannot
 * say so: `state` there has one writer (the reader), so a reader that exited while the
 * compute thread was still spinning out a bound would read EXIT and be a lie. Relaunching
 * on that lie puts two kernels on one core.
 *
 * @param processor_mask  bit FT_TT_PROC_* per processor to require; 0 falls back to the
 *                        control block's own `state`, which is right for the BRISC-only
 *                        kernel and wrong for anything else
 * @param out_pending     may be NULL; receives the mask of processors NOT yet at EXIT
 * @return                FT_TT_OK if all have exited, FT_TT_EIO if any has not, or the
 *                        accessor's code
 */
int ft_tt_queue_all_processors_exited(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                      uint32_t processor_mask, uint32_t *out_pending);

/* ================================================================
 * The matmul parameter block
 * ================================================================ */

/**
 * Write one matmul parameter block into a core's L1 and return the address a descriptor's
 * `a0` should carry.
 *
 * The caller supplies the sixteen words; this is deliberately a thin call, because the
 * SHAPE is the op driver's business (ft_tt_op_matmul.h owns the orientation decision and
 * the depth-divides-Kt rule) and the PLACEMENT is the queue's.
 *
 * @param slot      0..FT_TT_QUEUE_MATMUL_PARAM_SLOTS-1
 * @param words     FT_TT_QUEUE_MATMUL_WORDS values
 * @param out_addr  may be NULL; the address to put in the descriptor's a0
 */
int ft_tt_queue_write_matmul_parameters(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                        uint32_t slot,
                                        const uint32_t words[FT_TT_QUEUE_MATMUL_WORDS],
                                        uint32_t *out_addr);

/**
 * The same call under a family-neutral name — sixteen words into slot `slot`, returning
 * the address a descriptor's `a0` should carry.
 *
 * The function never interpreted the words, so there was never anything matmul-specific
 * about it; only the name said otherwise. New op drivers call this one.
 */
int ft_tt_queue_write_op_parameters(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                    uint32_t slot, const uint32_t words[FT_TT_QUEUE_PARAM_WORDS],
                                    uint32_t *out_addr);

/**
 * Write one block of the PARAMETER CACHE — the region documented beside
 * FT_TT_QUEUE_PARAM_CACHE_OFFSET, whose blocks are pointed at and never rewritten.
 *
 * Identical to ft_tt_queue_write_op_parameters() except for the region, and a SEPARATE
 * function so that a caller cannot reach a never-rewritten block through the rotating-slot
 * API and quietly rewrite it under a descriptor the card has not read.
 *
 * @param slot      0..FT_TT_QUEUE_PARAM_CACHE_SLOTS-1
 * @param out_addr  may be NULL; the address a descriptor's `a0` must carry
 * @return FT_TT_OK, FT_TT_EINVAL, or the accessor's failure
 */
int ft_tt_queue_write_param_cache(const ft_tt_noc_access *noc, const ft_tt_queue *queue,
                                  uint32_t slot, const uint32_t words[FT_TT_QUEUE_PARAM_WORDS],
                                  uint32_t *out_addr);

/* ================================================================
 * Runtime args the resident kernels take
 * ================================================================ */

/** How many runtime args every resident kernel reads. ONE layout for all five processors —
 *  kernels/resident_queue_protocol.h says why. */
#define FT_TT_QUEUE_KERNEL_ARG_COUNT 10u

/**
 * What one PROCESSOR needs beyond the queue itself.
 *
 * Both spin limits are the bounded-residency rule, and they are separate because they bound
 * different things: `idle_spin_limit` is "the host stopped feeding me", `cb_wait_spin_limit`
 * is "a peer processor stopped feeding me". Neither may be 0 — an unbounded wait is the one
 * thing a resident kernel may not be configured into, and it is also what makes it safe to
 * run a build whose stop check is deliberately broken.
 */
typedef struct ft_tt_queue_kernel_binding {
    uint32_t idle_spin_limit;    /**< consecutive empty queue polls before returning */
    uint32_t cb_wait_spin_limit; /**< spins in ONE circular-buffer wait before abandoning */
    uint32_t status_address;     /**< this processor's slot; 0 = do not report status */
    uint16_t group_x;            /**< TRANSLATED coordinate of the group block's core */
    uint16_t group_y;
    uint32_t group_address;      /**< 0 = this core posts no group completion */
    uint32_t group_member;       /**< this core's slot in the group block */
} ft_tt_queue_kernel_binding;

/**
 * Fill one processor's runtime-arg block:
 *   0: control-block address   1: capacity mask (capacity - 1)
 *   2: ring address            3: idle spin limit
 *   4: status slot address     5: CB wait spin limit
 *   6: group NoC x             7: group NoC y
 *   8: group block address     9: group member slot
 *
 * @param binding   may be NULL for the defaults: no status, no group, and the CB bound
 *                  equal to the idle bound
 * @param out_args  FT_TT_QUEUE_KERNEL_ARG_COUNT words
 * @return          FT_TT_OK or FT_TT_EINVAL
 */
int ft_tt_queue_kernel_args(const ft_tt_queue *queue, const ft_tt_queue_kernel_binding *binding,
                            uint32_t out_args[FT_TT_QUEUE_KERNEL_ARG_COUNT]);

#endif /* FT_TT_QUEUE_H */
