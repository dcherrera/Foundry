/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_queue_protocol.h
 * @brief Device half of Foundry's resident command queue: the wire contract, the per-kernel
 *        queue context, and the stoppable, bounded waits every resident kernel uses.
 *
 * The host half is tt/ft_tt_queue.h; every number in the wire contract below must agree with
 * it. The header is shared by the resident data-movement kernels (BRISC/NCRISC) and the
 * resident compute kernels (TRISC, built once per thread), and it is a content input to all
 * of their images.
 *
 * Safety model:
 *   - Every control word has exactly one writer. Completion is published by the last
 *     processor of the pipeline.
 *   - Every blocking CB wait can be asked to stop (the host writes the stop word) and is
 *     bounded by a spin limit.
 *   - An abort does not return early from the descriptor. It sets `aborting`; every later
 *     wait in the same descriptor then returns false at once, while the kernel's loops still
 *     run their full trip counts so the peer threads' hardware handshakes stay balanced. An
 *     abandoned descriptor is never counted complete.
 *   - NoC barriers and the MATH/PACK/dvalid hardware stalls are deliberately not stoppable.
 *
 * WIRE CONTRACT FORM: off-hardware tests parse this file line by line, so every wire constant
 * is its own one-line anonymous enum `enum { name = value };`. Opcodes are the only
 * identifiers here that start with `kOp`. kMatmulColumnBlock, kMatmulWords and kDescA2 are
 * parsed as decimal.
 *
 * @status Complete for the data-movement and C99 compute arms.
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_QUEUE_PROTOCOL_H
#define RESIDENT_QUEUE_PROTOCOL_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#if defined(COMPILE_FOR_TRISC) && defined(__cplusplus)
#error "resident_queue_protocol.h: there is no C++ compute arm; build compute kernels as C"
#endif

#if !defined(COMPILE_FOR_TRISC)
#include "ft_tensix_dataflow.h"
#else
#include "ft_tensix_compute_cb.h"
#include "ft_tensix_kernel_args.h"
#endif

/* Linkage of every helper here (family protocol headers spell it identically). */
#ifndef FTQ_INLINE
#ifdef __cplusplus
#define FTQ_INLINE inline
#else
#define FTQ_INLINE static inline
#endif
#endif

/* The compiler's L1 pointer attribute, spelled locally so this header does not depend on the
 * register header for it. */
#ifndef FTQ_L1_PTR
#define FTQ_L1_PTR __attribute__((rvtt_l1_ptr))
#endif

#ifdef __cplusplus
#define FTQ_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define FTQ_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif

#define FTQ_RUNTIME_ARG(index) ft_tensix_runtime_arg(index)

/* ---- Control-block word indices (4-byte words from the control block base) ---- */
enum { kFieldMagic = 0 };
enum { kFieldCapacity = 1 };
enum { kFieldHead = 2 };            /* descriptors published by the host */
enum { kFieldTail = 3 };            /* descriptors taken by the reader */
enum { kFieldStopRequest = 4 };     /* host-written stop word */
enum { kFieldCompletedCount = 5 };  /* written by the last processor (writer) */
enum { kFieldCompletedToken = 6 };
enum { kFieldState = 7 };
enum { kFieldLastOpcode = 8 };
enum { kFieldError = 9 };
enum { kFieldIdleSpins = 10 };
enum { kFieldHeartbeat = 11 };
enum { kFieldGroupDone = 56 };      /* byte 0xE0: finished group sequence */
enum { kFieldIdleClock = 60 };      /* NCRISC's accumulated idle-for-work clock */

/* ---- Magic, states, errors ---- */
enum { kMagic = 0x46545131 };       /* "FTQ1" */
enum { kStatePoison = 0x7B1B1B1B }; /* host-written poison */
enum { kStateBoot = 0x51010001 };
enum { kStateIdle = 0x51010002 };
enum { kStateExec = 0x51010003 };
enum { kStateSemWait = 0x51010004 };
enum { kStateExit = 0x51010005 };
enum { kStateCbWaitFront = 0x51010006 };
enum { kStateCbReserveBack = 0x51010007 };
enum { kErrNone = 0 };
enum { kErrBadMagic = 1 };
enum { kErrBadOpcode = 2 };
enum { kErrIdleTimeout = 3 };
enum { kErrSemWaitTimeout = 4 };
enum { kErrBadCapacity = 5 };
enum { kErrCbWaitTimeout = 6 };
enum { kErrBadParameters = 7 };
enum { kErrCoopRoleMismatch = 8 };
enum { kErrCoopWritePointer = 9 };
enum { kErrCoopRendezvous = 10 };
enum { kErrCoopIssue = 11 };
enum { kErrCoopBadPartition = 12 };

/* ---- Opcodes ---- */
enum { kOpNop = 0x01 };
enum { kOpWriteWord = 0x02 };
enum { kOpL1Copy = 0x03 };
enum { kOpSemPost = 0x04 };
enum { kOpSemWait = 0x05 };
enum { kOpDelay = 0x06 };
enum { kOpStop = 0x07 };
enum { kOpMatmul = 0x08 };
enum { kOpEltwise = 0x09 };
enum { kOpSoftmax = 0x0A };
enum { kOpReduce = 0x0B };
enum { kOpEmbedding = 0x0C };
enum { kOpKvAppend = 0x0D };
enum { kOpRope = 0x0E };
enum { kOpTopk = 0x0F };
enum { kOpSample = 0x10 };
enum { kOpAttention = 0x11 };
enum { kOpGdn = 0x12 };

/* ---- Descriptor (one ring entry, 8 words) ---- */
enum { kDescriptorWords = 8 };
enum { kDescOpcode = 0 };
enum { kDescToken = 1 };
enum { kDescA0 = 2 };
enum { kDescA1 = 3 };
enum { kDescA2 = 4 };
enum { kDescA3 = 5 };
enum { kDescA4 = 6 };
enum { kDescA5 = 7 };               /* group sequence number; 0 = no group-completion post */

/* ---- Per-processor status slot (8 words = 32 bytes, so no two processors' heartbeats
 *      share a 16-byte L1 line) ---- */
enum { kStatusState = 0 };
enum { kStatusCursor = 1 };
enum { kStatusHeartbeat = 2 };
enum { kStatusError = 3 };
enum { kStatusLastOpcode = 4 };
enum { kStatusSpins = 5 };
enum { kStatusPayload = 6 };        /* 2 words: accumulated wait clock, wait count */

/* ---- Group completion block: [sequence, token] per member from byte 0x40 ---- */
enum { kGroupSlotOffsetWords = 16 };
enum { kGroupSlotWords = 2 };

/* ---- Pull completion (matmul writer), offsets from the control block ---- */
enum { kPullTableOffset = 0xF900 };   /* magic, count, then x | y << 16 per member */
enum { kPullScratchOffset = 0xFF00 }; /* 16-byte landing for the remote read */
enum { kPullMagic = 0x46545150 };     /* "FTQP" */
enum { kPullMaxMembers = 64 };
enum { kPullRedSkip = 0x80000000u };  /* bit 31 of a3: skip (unsigned enumerator) */

/* ---- Matmul parameter block (16 words at descriptor a0) ---- */
enum { kMatmulTileRows = 0 };
enum { kMatmulTileInner = 1 };
enum { kMatmulTileCols = 2 };
enum { kMatmulAX = 3 };
enum { kMatmulAY = 4 };
enum { kMatmulABase = 5 };
enum { kMatmulBX = 6 };
enum { kMatmulBY = 7 };
enum { kMatmulBBase = 8 };
enum { kMatmulCX = 9 };
enum { kMatmulCY = 10 };
enum { kMatmulCBase = 11 };
enum { kMatmulDepth = 12 };
enum { kMatmulBNMajor = 13 };       /* flag word: bit 0 N-major, 8 pipelined, 9 RED never-wait */
enum { kMatmulCooperative = 14 };
enum { kMatmulColumnBlock = 15 };
enum { kMatmulWords = 16 };
enum { kMatmulRow0 = 1u << 10 };
enum { kMatmulProbeNoRead = 1u << 12 };
enum { kMatmulProbeNoCompute = 1u << 13 };
enum { kMatmulProbeNoWrite = 1u << 14 };
enum { kMatmulWriterFlush = 1u << 15 };
enum { kMatmulAReuse = 1u << 18 };
enum { kMatmulAReuseRed = 1u << 19 };
/* Bits 20..29 are the MoE build's (resident_matmul_moe_protocol.h). Bit 30: the weight tile
 * is Bfp4_b (576 B); 0 = the CB table's cb 1 format. Read only by `rtriad_mixw`
 * (-DFT_RTRIAD_MIXW=1); every other build ignores it, so the host sets it only on a mixw
 * residency (ft_tt_queue_matmul_check_weight). Host mirror: FT_TT_QUEUE_MATMUL_FLAG_WEIGHT_BFP4.
 * Bit 31 is free; the mixw build refuses it. Declared only in that build, so every other
 * resident image preprocesses to the same tokens as before it existed. */
#if defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW
enum { kMatmulWeightBfp4 = 1u << 30 };
#endif

/* ---- Runtime-argument layout (same on all five processors) ---- */
enum { kArgControl = 0 };           /* control-block L1 address */
enum { kArgCapacityMask = 1 };      /* capacity - 1 */
enum { kArgRing = 2 };              /* descriptor ring L1 address */
enum { kArgIdleSpinLimit = 3 };
enum { kArgStatus = 4 };            /* this processor's status slot, 0 = none */
enum { kArgCbSpinLimit = 5 };
enum { kArgGroupX = 6 };
enum { kArgGroupY = 7 };
enum { kArgGroupAddress = 8 };      /* 0 = no group completion */
enum { kArgGroupMember = 9 };

/* ---- Poll masks for blocked waits ---- */
enum { kStopPollMask = 0 };         /* poll the stop word every blocked pass */
enum { kHeartbeatPollMask = 0xFF }; /* bump the heartbeat every 256 blocked passes */

/* ---- Wait modes (compile-time argument passed as a parameter). Mode 2 (bound only, stop
 *      ignored) needs no code of its own: only mode 1 polls the stop word. ---- */
enum { kWaitUpstream = 0 };         /* plain blocking wait, no escape, no bound */
enum { kWaitStoppable = 1 };        /* stop-aware and bounded */

FTQ_STATIC_ASSERT(kStatePoison != kStateExit, "poison must not read as a clean exit");
FTQ_STATIC_ASSERT(kStatePoison != 0, "poison must not read as zero");

/** One processor's view of the queue. */
typedef struct ftq_context {
    volatile FTQ_L1_PTR uint32_t *control;
    volatile FTQ_L1_PTR uint32_t *ring;
    volatile FTQ_L1_PTR uint32_t *status; /* may be null */
    uint32_t capacity_mask;
    uint32_t idle_spin_limit;
    uint32_t cb_spin_limit;
    uint32_t group_x;
    uint32_t group_y;
    uint32_t group_address;               /* 0 = no group completion */
    uint32_t group_member;
    uint32_t cursor;                      /* descriptors this processor has taken */
    uint32_t heartbeat;
    uint32_t error;
    bool aborting;
} ftq_context;

/** Fill `ctx` from this launch's runtime arguments. Statement order is kept as written. */
FTQ_INLINE void ftq_context_init(ftq_context *ctx)
{
    ctx->control = (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)FTQ_RUNTIME_ARG(kArgControl);
    ctx->capacity_mask = FTQ_RUNTIME_ARG(kArgCapacityMask);
    ctx->ring = (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)FTQ_RUNTIME_ARG(kArgRing);
    ctx->idle_spin_limit = FTQ_RUNTIME_ARG(kArgIdleSpinLimit);
    const uint32_t status_address = FTQ_RUNTIME_ARG(kArgStatus);
    ctx->status = status_address ? (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)status_address
                                 : (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)0;
    ctx->cb_spin_limit = FTQ_RUNTIME_ARG(kArgCbSpinLimit);
    if (ctx->status) {
        ctx->status[kStatusPayload] = 0u;
        ctx->status[kStatusPayload + 1] = 0u;
    }
#if defined(COMPILE_FOR_NCRISC)
    ctx->control[kFieldIdleClock] = 0u;
#endif
    ctx->group_x = FTQ_RUNTIME_ARG(kArgGroupX);
    ctx->group_y = FTQ_RUNTIME_ARG(kArgGroupY);
    ctx->group_address = FTQ_RUNTIME_ARG(kArgGroupAddress);
    ctx->group_member = FTQ_RUNTIME_ARG(kArgGroupMember);
    ctx->cursor = 0u;
    ctx->heartbeat = 0u;
    ctx->error = kErrNone;
    ctx->aborting = false;
}

/** Publish state, cursor, a fresh heartbeat and the error into the status slot. */
FTQ_INLINE void ftq_status_set(ftq_context *ctx, uint32_t state)
{
    if (ctx->status) {
        ctx->status[kStatusState] = state;
        ctx->status[kStatusCursor] = ctx->cursor;
        ctx->status[kStatusHeartbeat] = ++ctx->heartbeat;
        ctx->status[kStatusError] = ctx->error;
    }
}

FTQ_INLINE void ftq_status_opcode(const ftq_context *ctx, uint32_t opcode)
{
    if (ctx->status)
        ctx->status[kStatusLastOpcode] = opcode;
}

FTQ_INLINE bool ftq_stop_requested(const ftq_context *ctx)
{
    return ctx->control[kFieldStopRequest] != 0u;
}

/** Abandon the current descriptor: later waits return false at once. The first non-zero
 *  reason is kept. */
FTQ_INLINE void ftq_abort(ftq_context *ctx, uint32_t reason)
{
    ctx->aborting = true;
    if (reason != kErrNone && ctx->error == kErrNone)
        ctx->error = reason;
    if (ctx->status)
        ctx->status[kStatusError] = ctx->error;
}

/** Add one blocked wait (clock since `started`, in 16-cycle units) to the status payload. */
FTQ_INLINE void ftq_wait_account(ftq_context *ctx, uint32_t started)
{
    if (ctx->status) {
        ctx->status[kStatusPayload] += (ft_bh_wall_clock_low() - started) >> 4;
        ctx->status[kStatusPayload + 1] += 1u;
    }
}

/** Validate the control block and the launch parameters; sets ctx->error on failure. */
FTQ_INLINE bool ftq_check_control(ftq_context *ctx)
{
    if (ctx->control[kFieldMagic] != (uint32_t)kMagic) {
        ctx->error = kErrBadMagic;
        return false;
    }
    if (ctx->capacity_mask == 0u || ((ctx->capacity_mask + 1u) & ctx->capacity_mask) != 0u) {
        ctx->error = kErrBadCapacity;
        return false;
    }
    if (ctx->control[kFieldCapacity] != ctx->capacity_mask + 1u) {
        ctx->error = kErrBadCapacity;
        return false;
    }
    if (ctx->cb_spin_limit == 0u || ctx->idle_spin_limit == 0u) {
        ctx->error = kErrBadParameters;
        return false;
    }
    return true;
}

/** The ring entry at this processor's cursor. */
FTQ_INLINE volatile FTQ_L1_PTR uint32_t *ftq_descriptor(const ftq_context *ctx)
{
    return ctx->ring + (ctx->cursor & ctx->capacity_mask) * kDescriptorWords;
}

/** The parameter block a descriptor points at (word a0). */
FTQ_INLINE volatile FTQ_L1_PTR uint32_t *ftq_parameters(volatile FTQ_L1_PTR uint32_t *descriptor)
{
    return (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)descriptor[kDescA0];
}

/** Idle until control[barrier_field] differs from the cursor. False = leave the resident
 *  loop (stop requested, or the idle bound ran out with kErrIdleTimeout). */
FTQ_INLINE bool ftq_wait_for_work(ftq_context *ctx, uint32_t barrier_field)
{
    uint32_t idle_spins = 0u;
#if defined(COMPILE_FOR_NCRISC)
    const uint32_t idle_from = ft_bh_wall_clock_low();
#endif

    for (;;) {
        ctx->heartbeat++;
        if (ctx->status)
            ctx->status[kStatusHeartbeat] = ctx->heartbeat;
        if (ftq_stop_requested(ctx))
            return false;
        if (ctx->control[barrier_field] != ctx->cursor) {
#if defined(COMPILE_FOR_NCRISC)
            if (idle_spins != 0u)
                ctx->control[kFieldIdleClock] += (ft_bh_wall_clock_low() - idle_from) >> 4;
#endif
            return true;
        }
        if (ctx->status)
            ctx->status[kStatusState] = kStateIdle;
        if (++idle_spins >= ctx->idle_spin_limit) {
            ctx->error = kErrIdleTimeout;
            return false;
        }
    }
}

/** Wait until control[field] >= value; stoppable and bounded by the CB spin limit. */
FTQ_INLINE bool ftq_wait_until_reaches(ftq_context *ctx, uint32_t field, uint32_t value)
{
    uint32_t spins = 0u;

    while (ctx->control[field] < value) {
        ctx->heartbeat++;
        if (ctx->status)
            ctx->status[kStatusHeartbeat] = ctx->heartbeat;
        if (ftq_stop_requested(ctx)) {
            ftq_abort(ctx, kErrNone);
            return false;
        }
        if (++spins >= ctx->cb_spin_limit) {
            ftq_abort(ctx, kErrCbWaitTimeout);
            return false;
        }
    }
    return true;
}

/* What a blocked CB wait does after an unsatisfied test: start the wait clock on the first
 * pass, publish a heartbeat every 256 passes, and give up (aborting the descriptor) when the
 * stop word is set in stoppable mode or the spin bound runs out. Expects `ctx` and
 * `wait_mode` in scope; returns false from the enclosing function. */
#define FTQ_CB_WAIT_ESCAPE(spins, wait_started)                                             \
    do {                                                                                     \
        if ((spins) == 0u)                                                                   \
            (wait_started) = ft_bh_wall_clock_low();                                         \
        (spins)++;                                                                           \
        if (((spins) & (uint32_t)kHeartbeatPollMask) == 0u && ctx->status) {                 \
            ctx->heartbeat++;                                                                \
            ctx->status[kStatusHeartbeat] = ctx->heartbeat;                                  \
            ctx->status[kStatusSpins] = (spins);                                             \
        }                                                                                    \
        if (((spins) & (uint32_t)kStopPollMask) == 0u) {                                     \
            if (wait_mode == (uint32_t)kWaitStoppable && ftq_stop_requested(ctx)) {          \
                ftq_abort(ctx, kErrNone);                                                    \
                return false;                                                                \
            }                                                                                \
            if ((spins) >= ctx->cb_spin_limit) {                                             \
                ftq_abort(ctx, kErrCbWaitTimeout);                                           \
                return false;                                                                \
            }                                                                                \
        }                                                                                    \
    } while (0)

#if !defined(COMPILE_FOR_TRISC)

/** Data-movement consumer: wait until `num_pages` pages are readable in CB `operand`. */
FTQ_INLINE bool ftq_cb_wait_front_stoppable(ftq_context *ctx, int32_t operand, int32_t num_pages,
                                            uint32_t wait_mode)
{
    uint32_t pages_acked;
    uintptr_t pages_received_pointer;
    uint16_t pages_received;
    uint32_t spins = 0u;
    uint32_t wait_started = 0u;

    if (wait_mode == (uint32_t)kWaitUpstream) {
        ft_tensix_cb_wait_front(operand, num_pages);
        return true;
    }
    if (ctx->aborting)
        return false;
    pages_acked = ft_tensix_cb_pages_acked_pointer(operand)[0];
    pages_received_pointer = (uintptr_t)ft_tensix_cb_pages_received_pointer(operand);
    do {
        pages_received =
            ((uint16_t)ft_bh_reg_read(pages_received_pointer)) - (uint16_t)pages_acked;
        if (pages_received >= (uint16_t)num_pages)
            break;
        FTQ_CB_WAIT_ESCAPE(spins, wait_started);
    } while (true);
    if (spins != 0u)
        ftq_wait_account(ctx, wait_started);
    return true;
}

/** Data-movement producer: wait until `num_pages` pages are free in CB `operand`. */
FTQ_INLINE bool ftq_cb_reserve_back_stoppable(ftq_context *ctx, int32_t operand,
                                              int32_t num_pages, uint32_t wait_mode)
{
    uintptr_t pages_acked_pointer;
    uint32_t pages_received;
    int32_t free_space_pages;
    uint32_t spins = 0u;
    uint32_t wait_started = 0u;

    if (wait_mode == (uint32_t)kWaitUpstream) {
        ft_tensix_cb_reserve_back(operand, num_pages);
        return true;
    }
    if (ctx->aborting)
        return false;
    pages_acked_pointer = (uintptr_t)ft_tensix_cb_pages_acked_pointer(operand);
    pages_received = ft_tensix_cb_pages_received_pointer(operand)[0];
    do {
        uint16_t pages_acked;
        uint16_t free_space_pages_wrap;

        ft_bh_l1_cache_invalidate();
        pages_acked = (uint16_t)ft_bh_reg_read(pages_acked_pointer);
        free_space_pages_wrap = ft_tensix_cb((uint32_t)operand)->fifo_num_pages -
                                ((uint16_t)pages_received - pages_acked);
        free_space_pages = (int32_t)free_space_pages_wrap;
        if (free_space_pages >= num_pages)
            break;
        FTQ_CB_WAIT_ESCAPE(spins, wait_started);
    } while (true);
    if (spins != 0u)
        ftq_wait_account(ctx, wait_started);
    return true;
}

/** Write [sequence, token] into this member's slot of the group block on another core, as
 *  two inline word writes, token first. Called after completed_count is published; a refusal
 *  aborts with kErrBadParameters and is not rolled back. */
FTQ_INLINE void ftq_group_post(ftq_context *ctx, uint32_t sequence, uint32_t token)
{
    uint32_t slot_address;

    if (ctx->group_address == 0u || sequence == 0u)
        return;
    slot_address =
        ctx->group_address + (kGroupSlotOffsetWords + ctx->group_member * kGroupSlotWords) * 4u;
    if (ft_tensix_noc_inline_dword_write_l1(
            ft_tensix_noc_address(ctx->group_x, ctx->group_y, slot_address + 4u), token) !=
        FT_TENSIX_NOC_ISSUED) {
        ftq_abort(ctx, kErrBadParameters);
        return;
    }
    ft_tensix_noc_write_barrier();
    if (ft_tensix_noc_inline_dword_write_l1(
            ft_tensix_noc_address(ctx->group_x, ctx->group_y, slot_address), sequence) !=
        FT_TENSIX_NOC_ISSUED) {
        ftq_abort(ctx, kErrBadParameters);
        return;
    }
    ft_tensix_noc_write_barrier();
}

#else /* COMPILE_FOR_TRISC */

/** Unpack-thread consumer wait. The acked count is the thread's own 16-bit record field,
 *  not the stream register. */
FTQ_INLINE bool ftq_llk_wait_tiles_body(ftq_context *ctx, int32_t operand, int32_t num_pages,
                                        uint32_t wait_mode)
{
    uint32_t input = (uint32_t)operand;
    volatile FT_BH_REG_PTR uint32_t *tiles_received_pointer =
        ft_tensix_cb_pages_received_pointer(operand);
    uint16_t num_tiles_u = (uint16_t)num_pages;
    uint16_t tiles_received;
    uint16_t num_tiles_recv;
    uint32_t spins = 0u;
    uint32_t wait_started = 0u;

    do {
        tiles_received = (uint16_t)ft_bh_reg_read((uint32_t)(uintptr_t)tiles_received_pointer);
        num_tiles_recv =
            (uint16_t)(tiles_received - ft_tensix_cb(input)->tiles.counters.tiles_acked);
        if (num_tiles_recv >= num_tiles_u) {
            if (spins != 0u)
                ftq_wait_account(ctx, wait_started);
            return true;
        }
        FTQ_CB_WAIT_ESCAPE(spins, wait_started);
    } while (true);
}

/** Compute consumer wait; only the unpack thread waits, the others just honour aborting. */
FTQ_INLINE bool ftq_cb_wait_front_stoppable(ftq_context *ctx, int32_t operand, int32_t num_pages,
                                            uint32_t wait_mode)
{
    bool arrived = true;
    (void)operand;
    (void)num_pages;

    if (wait_mode == (uint32_t)kWaitUpstream) {
        FT_TENSIX_UNPACK(ft_tensix_compute_cb_wait_front((uint32_t)operand, (uint32_t)num_pages));
        return true;
    }
    if (ctx->aborting)
        return false;
    FT_TENSIX_UNPACK((arrived = ftq_llk_wait_tiles_body(ctx, operand, num_pages, wait_mode)));
    return arrived;
}

/** Pack-thread producer wait: free tiles = pages minus (received - acked), received being the
 *  thread's own record field, read once. */
FTQ_INLINE bool ftq_llk_wait_for_free_tiles_body(ftq_context *ctx, int32_t operand,
                                                 int32_t num_pages, uint32_t wait_mode)
{
    uint32_t output = (uint32_t)operand;
    volatile FT_BH_REG_PTR uint32_t *tiles_acked_pointer =
        ft_tensix_cb_pages_acked_pointer(operand);
    uint16_t tiles_received = ft_tensix_cb(output)->tiles.counters.tiles_received;
    int32_t free_tiles;
    uint32_t spins = 0u;
    uint32_t wait_started = 0u;

    do {
        uint16_t tiles_acked = (uint16_t)ft_bh_reg_read((uint32_t)(uintptr_t)tiles_acked_pointer);
        uint16_t free_tiles_wrap = (uint16_t)(ft_tensix_cb(output)->fifo_num_pages -
                                              (uint16_t)(tiles_received - tiles_acked));
        free_tiles = (int32_t)free_tiles_wrap;
        if (free_tiles >= (int32_t)num_pages) {
            if (spins != 0u)
                ftq_wait_account(ctx, wait_started);
            return true;
        }
        FTQ_CB_WAIT_ESCAPE(spins, wait_started);
    } while (true);
}

/** Compute producer wait; only the pack thread waits, the others just honour aborting. */
FTQ_INLINE bool ftq_cb_reserve_back_stoppable(ftq_context *ctx, int32_t operand,
                                              int32_t num_pages, uint32_t wait_mode)
{
    bool reserved = true;
    (void)operand;
    (void)num_pages;

    if (wait_mode == (uint32_t)kWaitUpstream) {
        FT_TENSIX_PACK(
            ft_tensix_compute_cb_reserve_back((uint32_t)operand, (uint32_t)num_pages));
        return true;
    }
    if (ctx->aborting)
        return false;
    FT_TENSIX_PACK(
        (reserved = ftq_llk_wait_for_free_tiles_body(ctx, operand, num_pages, wait_mode)));
    return reserved;
}

#endif /* COMPILE_FOR_TRISC */

#endif /* RESIDENT_QUEUE_PROTOCOL_H */
