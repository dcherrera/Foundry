/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_matmul_reader.c
 * @brief Reader third of the RESIDENT matmul triad -- the first Foundry kernel that is both
 * resident (it does not return between ops) and part of a reader/compute/writer triad
 * with real circular buffers. Phase 5's resident kernel was BRISC-only and CB-free
 * because `cb_wait_front` was the one wait the host could not break; this kernel and its
 * two peers are what closes that (resident_queue_protocol.h, rules 2-4).
 *
 * IT IS matmul_reader.c'S INNER LOOP INSIDE A QUEUE LOOP, PLUS A BLOCKED `nt`. The
 * one-transaction A block, the N-major B block and the depth-divides-Kt contract are all
 * exactly as that file has them and its header documents them; `cb_reserve_back` became
 * ftq_cb_reserve_back_stoppable(, WAIT_MODE) and the shapes arrive in a parameter block
 * instead of a runtime-arg block.
 *
 * THE ONE ARITHMETIC CHANGE IS `kMatmulColumnBlock`. At Mt = 1 the A offset is
 * `(mt*Kt + kt)` and does not depend on `nt`, so the unblocked loop re-reads the whole A
 * column ONCE PER OUTPUT COLUMN — measured at 15.871 of the 24.303 GB a token reads, 65.3%,
 * nearly all of it out of the ONE bank the activation strip lives in. With NB columns per
 * pass it is fetched Nt/NB times. B's bytes and B's transaction size are untouched: the
 * panel is N-major, so each column is still one read of `depth` contiguous tiles, and the
 * block lands COLUMN-MAJOR in L1 (column j at tile `j*depth`) which is what the compute
 * thread's SrcA stride is set for. NB = 0 or 1 is the unblocked shape, byte for byte.
 *
 * AND ITS MIRROR IS THE ROW BLOCK, IN THE HIGH HALF OF THE SAME WORD. `MB` A tile-rows are
 * held in L1 and B is streamed past ALL of them, so the WEIGHT column is fetched
 * `ceil(Mt/MB)` times instead of `Mt`. The two are exactly symmetric —
 * `total = Mt*Nt*Kt*(1/NB + 1/MB)` — and they SHARE the DST section, which is four tiles:
 * `MB * NB <= 4`, and the minimum of that sum is at MB = NB = 2, not at either extreme.
 * resident_matmul_block_protocol.h carries the derivation and the upstream citation. At
 * MB = 1 every loop below steps exactly as it did before the field existed.
 *
 * THIS PROCESSOR OWNS `tail`, AND NOTHING ELSE IN THE CONTROL BLOCK. `tail` advances
 * when the reader has finished its half of a descriptor and handed it to the writer. An
 * ABORTED descriptor advances nothing at all -- it was handed to nobody, and the next
 * launch re-attempts it.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: CB_IN0 index   1: CB_IN1 index
 *   2: in0 tile size in bytes   3: in1 tile size in bytes
 *   4: wait mode -- 0 upstream (the cost arm), 1 stoppable (ships), 2 bound-only (the
 *      red proof for the ask). resident_queue_protocol.h says why all three exist
 * Defines: `-DFT_RTRIAD_MOE=1` builds `rtriad_bfp6` — BFP6_B weight tiles expanded into the
 *   BFP8_B CB_IN1 pages (arg 3 stays 1088) and the indirect expert operand, pipelined path
 *   only; `-DFT_RTRIAD_MOE_RED_SHIFT=1` adds the off-by-one-bit expander (the red build).
 *   resident_matmul_moe_protocol.h is that contract, including its runtime-arg tail 23..28.
 *   `-DFT_RTRIAD_MIXW=1 -DFT_RTRIAD_MIXW_ALT_TILE_BYTES=576` builds `rtriad_mixw`: Bfp8_b or
 *   Bfp4_b weight tiles per descriptor (word 13 bit 30) into the same 1,088 B CB_IN1 pages.
 *   The two defines are exclusive (#error).
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors,
 *   plus the COOPERATIVE tail of resident_matmul_multicast_protocol.h (args 10..22), which
 *   ONLY this processor reads. `ftq_context_init()` is untouched, so the other four
 *   resident families' images are unaffected by that tail existing.
 *
 * AND IT IS ALSO THE COOPERATIVE READER. The shared A block is 65% of the bytes a token
 * reads, and every member of a split group reads it from the ONE bank holding that strip --
 * which is why four cores buy 3.44x and sixteen buy only 4.80x. With `role != SOLO` the
 * leader reads A once and MULTICASTS it as ceil(receivers / 4) packets; receivers read only
 * their own B panel. The mechanism is `cooperative_matmul_reader.c`'s, moved off a ONE-SHOT
 * program whose per-op image upload cost 218 us per member per op -- 147 ms a token at N=4
 * -- and onto the resident queue, where it costs a descriptor.
 *
 * Build (BRISC, NoC 0):
 *   CTARGS=0,1,2048,2048,1 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_matmul_reader.c brisc /tmp/ttk_rtriad
 *
 * @status See docs/tt_bringup_log.md. The cooperative path is new; the solo path is
 *   unchanged and is what a member_count == 1 residency runs.
 * @issues A cooperative descriptor arriving at a SOLO member -- or a solo one arriving at a
 *   cooperating member -- is REFUSED, and a refused descriptor strands the triad until the
 *   host re-arms it. That is the cheap outcome against a group whose CB_IN0 write pointers
 *   have drifted apart, which is silent and produces a wrong answer with every kernel
 *   completing and every status word clean.
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "ft_tensix_noc_multicast.h"
#include "resident_matmul_block_protocol.h"
#include "resident_matmul_multicast_protocol.h"
#include "resident_queue_protocol.h"
#include "resident_precondition.h"
/* The FUSED-SWIGLU build (program `rtriad_bfp8_swiglu`): a descriptor flagged in word 15 reads
 * the gate panel (B) and the up panel (a1/a2) of its columns, alternately per column block
 * (resident_matmul_swiglu_protocol.h). Every line it adds is under this switch. */
#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
#if (defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE) || (defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW)
#error "resident_matmul_reader.c: FT_RTRIAD_SWIGLU is its own program; build it without MOE/MIXW"
#endif
#include "resident_matmul_swiglu_protocol.h"
#endif
/* S3a, the RESIDUAL-ADD descriptor (resident_matmul_resadd_protocol.h): the fused program
 * only. Every line it adds is under this switch. */
#if defined(FT_RTRIAD_RESADD) && FT_RTRIAD_RESADD
#if !defined(FT_RTRIAD_SWIGLU) || !FT_RTRIAD_SWIGLU
#error "resident_matmul_reader.c: FT_RTRIAD_RESADD is built into the fused SwiGLU program only"
#endif
#include "resident_matmul_resadd_protocol.h"
#endif

#define CB_IN0 ft_tensix_compile_time_arg(0)
#define CB_IN1 ft_tensix_compile_time_arg(1)
#define IN0_TILE_BYTES ft_tensix_compile_time_arg(2)
#define IN1_TILE_BYTES ft_tensix_compile_time_arg(3)
#define WAIT_MODE ft_tensix_compile_time_arg(4)

/* =========================================================================
 * The MIXED-WEIGHT build (`KERNEL_DEFINES=-DFT_RTRIAD_MIXW=1
 * -DFT_RTRIAD_MIXW_ALT_TILE_BYTES=576`, program `rtriad_mixw`): one residency serving Bfp8_b
 * AND Bfp4_b weights, chosen per descriptor by word 13 bit 30 (kMatmulWeightBfp4). CB_IN1
 * keeps its 1,088 B pages for the whole residency; a Bfp4 descriptor's chunk holds its
 * `depth * NB` tiles at a 576 B stride in the front of the same `depth * NB` pages, so the
 * page counts pushed, the chunk grid and `fifo_page_size` never change.
 * Every B byte count and stride below goes through
 * B_TILE_BYTES, which every other build resolves to IN1_TILE_BYTES, token for token.
 * ========================================================================= */
#if defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
#error "resident_matmul_reader.c: FT_RTRIAD_MIXW and FT_RTRIAD_MOE are two programs (rtriad_mixw, rtriad_bfp6); build one"
#endif
#ifndef FT_RTRIAD_MIXW_ALT_TILE_BYTES
#error "resident_matmul_reader.c: the mixw build needs -DFT_RTRIAD_MIXW_ALT_TILE_BYTES=576 (the Bfp4_b weight tile)"
#endif
/** This descriptor's weight tile size in DRAM and its stride in CB_IN1: IN1_TILE_BYTES, or
 *  FT_RTRIAD_MIXW_ALT_TILE_BYTES when bit 30 is set. Set once a descriptor, before any read. */
static uint32_t ftmm_b_tile_bytes;
#define B_TILE_BYTES ftmm_b_tile_bytes

/**
 * Take this descriptor's weight format from word 13. Bit 31 is free and refused, as is an alt
 * tile larger than the CB_IN1 page it must fit inside. The compute thread switches SrcA from
 * the same bit; neither asks the other.
 */
static bool ftmm_select_weight_tile(ftq_context *ctx, uint32_t flags)
{
    if ((flags & 0x80000000u) != 0u || FT_RTRIAD_MIXW_ALT_TILE_BYTES > IN1_TILE_BYTES) {
        ctx->error = kErrBadParameters;
        return false;
    }
    ftmm_b_tile_bytes = (flags & (uint32_t)kMatmulWeightBfp4) != 0u
                            ? (uint32_t)FT_RTRIAD_MIXW_ALT_TILE_BYTES
                            : IN1_TILE_BYTES;
    return true;
}
#else
#define B_TILE_BYTES IN1_TILE_BYTES
#endif

/* =========================================================================
 * The MoE build (`KERNEL_DEFINES=-DFT_RTRIAD_MOE=1`, program `rtriad_bfp6`): BFP6_B weight
 * tiles in DRAM expanded to the BFP8_B pages CB_IN1 holds, and the indirect expert operand.
 * resident_matmul_moe_protocol.h is the contract. Every other build compiles none of it,
 * so B_DRAM_TILE_BYTES is B_TILE_BYTES (IN1_TILE_BYTES outside the mixw build) and no
 * instruction of theirs moves.
 * `-DFT_RTRIAD_MOE_RED_SHIFT=1` is the RED build: an expander off by one bit.
 * ========================================================================= */
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
#include "resident_matmul_moe_helpers.h"
#define B_DRAM_TILE_BYTES BFP6_TILE_BYTES
/** K-concatenated experts (FT_TT_MOE_FLAG_K_EXPERTS): the descriptor's per-block expert
 *  offsets, resolved once from the slot, in the scratch's offset table (LIST mode reaches 256
 *  blocks). 0 blocks = an ordinary descriptor. FTMM_ZERO_COLUMN = an all-zero B block. */
#define FTMM_ZERO_COLUMN 0xFFFFFFFFu
static uint32_t ftmm_k_blocks;
/** LIST mode (plan S1): descriptor column n is cell `ftmm_list_cell0 + n` of the (entry, expert
 *  column) grid; the offset table holds entries from `ftmm_list_s_base`. */
static bool ftmm_list;
static uint32_t ftmm_list_cell0, ftmm_list_columns, ftmm_list_s_base;
static FT_BH_INLINE volatile uint32_t *ftmm_offsets(void)
{
    return (volatile uint32_t *)(uintptr_t)(FTQ_RUNTIME_ARG(kArgMoeScratch) + FT_TT_MOE_OFFSETS_OFFSET);
}
/** Why this reader refused a descriptor, for the host's failure report: mailbox words 8..11 =
 *  reason, two values, the refused count (resident_matmul_moe_protocol.h mailbox). */
#define FTMM_MOE_WHY(ctx, reason, a, b)                                                          \
    do {                                                                                         \
        volatile FTQ_L1_PTR uint32_t *why_ = ftmoe_mailbox(ctx) + 8;                             \
        why_[0] = (reason);                                                                      \
        why_[1] = (a);                                                                           \
        why_[2] = (b);                                                                           \
        why_[3] = why_[3] + 1u;                                                                  \
    } while (0)
/** Where block B lands: the staging half (MoE) or the CB_IN1 pages (every other build). */
#define FTMM_B_LANDING(half, cb_pages) ftmm_stage(half)
/** Staging half `half` of the scratch: where a block's BFP6 tiles land before expansion. */
static FT_BH_INLINE uint32_t ftmm_stage(uint32_t half)
{
    return FTQ_RUNTIME_ARG(kArgMoeScratch) + FT_TT_MOE_STAGE_OFFSET + half * FT_TT_MOE_STAGE_HALF_BYTES;
}
#else
#define B_DRAM_TILE_BYTES B_TILE_BYTES
#define FTMM_B_LANDING(half, cb_pages) (cb_pages)
#define FTMM_MOE_WHY(ctx, reason, a, b) ((void)0)
#endif

/* =========================================================================
 * The COOPERATIVE tail: the shared A block, multicast once instead of read N times.
 *
 * All of it is gated on `role != SOLO`, which is a runtime arg fixed for the residency,
 * so a member_count == 1 residency executes the same instruction stream it always has
 * apart from one predictable branch per block. resident_matmul_multicast_protocol.h is
 * the contract and carries the reasoning; what is here is the loop body.
 * ========================================================================= */

/** Everything the cooperative roles share, read once at launch so a block indexes a
 *  struct instead of re-reading L1 through the runtime-arg accessor. `counter` is the
 *  member's own count of cooperative blocks and is what the rendezvous agrees on. */
typedef struct coop_context {
    uint32_t role;
    uint32_t leader_x;
    uint32_t leader_y;
    uint32_t ready_semaphore_address;
    uint32_t arrival_semaphore_address;
    uint32_t counter_word_address;
    uint32_t receiver_count;
    uint32_t group_count;
    uint32_t group[FT_TT_RESIDENT_COOP_MAX_GROUPS];
    uint32_t spin_limit;
    uint32_t counter;
    /** The queue's own stop word, reached as a plain pointer because that is what the
     *  multicast waits take. It is the SAME L1 word `ftq_stop_requested()` reads — rule 4
     *  is that every wait here is askable, and a rendezvous wait is no exception. */
    volatile const uint32_t *stop_flag;
} coop_context;

static void coop_read_arguments(coop_context *coop)
{
    uint32_t index;

    coop->stop_flag = (volatile const uint32_t *)(uintptr_t)(FTQ_RUNTIME_ARG(kArgControl) +
                                                             (uint32_t)kFieldStopRequest * 4u);
    coop->role = FTQ_RUNTIME_ARG(kArgCoopRole);
    coop->leader_x = FTQ_RUNTIME_ARG(kArgCoopLeaderX);
    coop->leader_y = FTQ_RUNTIME_ARG(kArgCoopLeaderY);
    coop->ready_semaphore_address = FTQ_RUNTIME_ARG(kArgCoopReadySemaphore);
    coop->arrival_semaphore_address = FTQ_RUNTIME_ARG(kArgCoopArrivalSemaphore);
    coop->counter_word_address = FTQ_RUNTIME_ARG(kArgCoopCounterWord);
    coop->receiver_count = FTQ_RUNTIME_ARG(kArgCoopReceiverCount);
    coop->group_count = FTQ_RUNTIME_ARG(kArgCoopGroupCount);
    for (index = 0u; index < FT_TT_RESIDENT_COOP_MAX_GROUPS; index++)
        coop->group[index] = FTQ_RUNTIME_ARG((int)(kArgCoopGroup0 + index));
    coop->spin_limit = FTQ_RUNTIME_ARG(kArgCoopSpinLimit);
    coop->counter = FT_TT_RESIDENT_COOP_FIRST_COUNTER;
}

/**
 * Refuse a cooperative CONFIGURATION at launch, before any descriptor runs.
 *
 * Checked here rather than trusted from the host for the reason DEV_NOTES.md §1 gives for
 * every other kernel-side guard: `ft_tt_noc_check_transfer()` runs host-side and cannot
 * see a kernel-issued transfer, and a multicast at a wrong coordinate is not a rejected
 * transaction — it is a write into a core another residency owns, or a hung NoC.
 */
static bool coop_configuration_is_legal(const coop_context *coop)
{
    uint32_t index;
    uint32_t destinations = 0u;

    if (coop->role == FT_TT_RESIDENT_COOP_ROLE_SOLO)
        return true;
    if (coop->role != FT_TT_RESIDENT_COOP_ROLE_LEADER &&
        coop->role != FT_TT_RESIDENT_COOP_ROLE_RECEIVER)
        return false;
    if (coop->ready_semaphore_address == 0u || coop->arrival_semaphore_address == 0u ||
        coop->counter_word_address == 0u || coop->spin_limit == 0u)
        return false;
    if (coop->receiver_count == 0u)
        /* The zero-length NoC transfer in multicast clothing. A group of one has no
         * receivers, and refusing is the only answer: "optimise it into a solo run" turns
         * a mis-planned group into a wrong answer that looks like a right one. */
        return false;
    if (coop->role != FT_TT_RESIDENT_COOP_ROLE_LEADER)
        return true;
    if (coop->group_count == 0u || coop->group_count > FT_TT_RESIDENT_COOP_MAX_GROUPS)
        return false;
    for (index = 0u; index < coop->group_count; index++) {
        const uint32_t packed = coop->group[index];
        const uint32_t count = ftcoop_rectangle_destinations(packed);

        if (count == 0u || count > FT_TT_RESIDENT_COOP_MAX_DESTINATIONS)
            return false;
        if (ftcoop_rectangle_x_low(packed) > ftcoop_rectangle_x_high(packed) ||
            ftcoop_rectangle_y_low(packed) > ftcoop_rectangle_y_high(packed))
            return false;
        destinations += count;
    }
    /* The barrier's acknowledgement accounting is this sum. Too many and the write
     * barrier never returns; too few and it returns while a payload is still landing in
     * L1 the kernel is about to reuse. */
    return destinations == coop->receiver_count;
}

/**
 * Where block `counter` of this residency MUST land in CB_IN0, from the wrap rule alone.
 *
 * A multicast writes ONE address into every destination, so the A block has to land at the
 * same L1 address on all N members. It does, because every member is given the same CB
 * placement and pushes the same pages in the same order — but "does" is a claim about host
 * code that no hardware would refuse if it stopped being true. The pointer resets by
 * subtraction on an EXACT hit of the limit and no other wrap is legal
 * (ft_tensix_circular_buffer.h), so the position is a pure function of how many blocks
 * this member has pushed. Everything is in the firmware's own pointer unit, whatever that
 * is: only the ratio is used.
 */
static uint32_t coop_expected_write_pointer(uint32_t cb_index, uint32_t chunk_pages,
                                            uint32_t counter)
{
    const ft_tensix_cb_interface *interface = ft_tensix_cb(cb_index);
    const uint32_t block = chunk_pages * interface->fifo_page_size;
    const uint32_t base = interface->fifo_limit - interface->fifo_size;

    if (block == 0u || interface->fifo_size < block)
        return 0u;
    return base + ((counter - FT_TT_RESIDENT_COOP_FIRST_COUNTER) % (interface->fifo_size / block)) *
                      block;
}

/**
 * LEADER: collect every receiver, fetch A, then hand it to the group as `group_count`
 * packets of at most four destinations each.
 *
 * The order of the last four steps is the protocol and none of it is rearrangeable —
 * ft_tensix_noc_multicast.h's header has the full account, including why the flush between
 * the payload and the semaphore is mandatory on Blackhole and nowhere else. ONE flush
 * covers every piece: it waits for requests SENT, and once every payload request has left,
 * ordering behind it on the shared static VC is the network's job.
 *
 * The payload's SOURCE is this core's own CB_IN0 block, so there is no staging buffer.
 * Reusing that block next round is safe for exactly one reason: the flush guarantees the
 * request has LEFT, which is what frees the source.
 */
static void coop_leader_block(ftq_context *ctx, coop_context *coop, uint32_t a_write,
                              uint32_t payload_bytes, bool skip_payload)
{
    volatile uint32_t *ready_semaphore =
        (volatile uint32_t *)(uintptr_t)coop->ready_semaphore_address;
    volatile uint32_t *counter_word = (volatile uint32_t *)(uintptr_t)coop->counter_word_address;
    uint32_t index;
    int code;

    code = ft_tensix_wait_semaphore_at_least(ready_semaphore, coop->receiver_count,
                                             coop->spin_limit, coop->stop_flag);
    if (code != FT_TENSIX_MULTICAST_OK) {
        /* A host that ASKED is not a fault, and reporting it as one would make every
         * retirement that lands mid-block look like a broken group. A bound that expired
         * is a different diagnosis and keeps its own code. */
        ftq_abort(ctx, code == FT_TENSIX_MULTICAST_ERROR_STOPPED ? kErrNone
                                                                 : kErrCoopRendezvous);
        return;
    }
    /* Safe here and only here: a receiver cannot increment again until it has been told
     * this round landed, which is the last thing this function does. */
    ft_tensix_set_local_semaphore(ready_semaphore, 0u);
    /* Staged BEFORE the payload so the value the semaphore multicast reads is already in
     * place, and re-staged every block because it is the block counter and not a constant
     * VALID — the receivers wait for the number THEY expect, so a group that has drifted
     * times out instead of computing from the wrong block. */
    ft_tensix_set_local_semaphore(counter_word, coop->counter);

    /* RED PROOF: skipping the payload leaves every receiver computing from the PREVIOUS
     * block while still being told this one arrived. The barrier accounting stays exact,
     * because the semaphore multicasts below advance it for themselves alone. */
    for (index = 0u; index < coop->group_count && !skip_payload; index++) {
        const uint32_t packed = coop->group[index];
        uint64_t destination;

        code = ft_tensix_multicast_address(ftcoop_rectangle_x_low(packed),
                                           ftcoop_rectangle_y_low(packed),
                                           ftcoop_rectangle_x_high(packed),
                                           ftcoop_rectangle_y_high(packed), a_write,
                                           &destination);
        if (code == FT_TENSIX_MULTICAST_OK)
            code = ft_tensix_noc_write_multicast(a_write, destination, payload_bytes,
                                                 ftcoop_rectangle_destinations(packed));
        if (code != FT_TENSIX_MULTICAST_OK) {
            ftq_abort(ctx, kErrCoopIssue);
            return;
        }
    }

    /* BLACKHOLE ONLY, and load-bearing: the payload multicasts use the WRITE command
     * buffer and the semaphore multicasts the WRITE_REG one; the two FIFOs are not
     * guaranteed to drain in issue order, so without this a receiver can be told the block
     * arrived before it did and read the PREVIOUS block's bytes with no error anywhere. */
    ft_tensix_noc_writes_flushed_barrier();

    for (index = 0u; index < coop->group_count; index++) {
        const uint32_t packed = coop->group[index];
        uint64_t destination;

        code = ft_tensix_multicast_address(ftcoop_rectangle_x_low(packed),
                                           ftcoop_rectangle_y_low(packed),
                                           ftcoop_rectangle_x_high(packed),
                                           ftcoop_rectangle_y_high(packed),
                                           coop->arrival_semaphore_address, &destination);
        if (code == FT_TENSIX_MULTICAST_OK)
            code = ft_tensix_noc_semaphore_set_multicast(
                coop->counter_word_address, destination,
                ftcoop_rectangle_destinations(packed));
        if (code != FT_TENSIX_MULTICAST_OK) {
            ftq_abort(ctx, kErrCoopIssue);
            return;
        }
    }
}

/**
 * RECEIVER: clear the arrival word, start its own B read, announce readiness, wait for A.
 *
 * The local clear comes FIRST. The leader cannot set the word until it has seen this
 * core's atomic, and this core issues that atomic after the clear — so an arrival can
 * never be erased by the clear that was meant to precede it.
 *
 * The B read is issued by the CALLER before this runs, so it overlaps the whole rendezvous:
 * this core's own DRAM traffic and the leader's multicast are then in flight together,
 * which is the only reason a receiver's block costs less than the sum of the two.
 */
static void coop_receiver_block(ftq_context *ctx, const coop_context *coop)
{
    volatile uint32_t *arrival_semaphore =
        (volatile uint32_t *)(uintptr_t)coop->arrival_semaphore_address;
    int code;

    ft_tensix_set_local_semaphore(arrival_semaphore, FT_TENSIX_SEMAPHORE_INVALID);
    ft_tensix_noc_semaphore_increment(
        ft_tensix_noc_address(coop->leader_x, coop->leader_y, coop->ready_semaphore_address), 1u);
    code = ft_tensix_wait_semaphore_equals(arrival_semaphore, coop->counter, coop->spin_limit,
                                           coop->stop_flag);
    if (code != FT_TENSIX_MULTICAST_OK)
        ftq_abort(ctx, code == FT_TENSIX_MULTICAST_ERROR_STOPPED ? kErrNone
                                                                 : kErrCoopRendezvous);
}

/**
 * Refuse a descriptor this program cannot execute, INSTEAD of writing outside the circular
 * buffers.
 *
 * THE BLOCK-PREFETCH INVARIANT IS NOT "DEPTH DIVIDES Kt" ON A RESIDENT OP. That is the
 * ONE-SHOT rule (docs/tt_kernel_model.md §12), where the FIFO write pointer starts at the
 * base. This kernel runs many descriptors, and `cb_push_back` wraps only on an EXACT hit of
 * the limit — so a descriptor that leaves the pointer part-way through the buffer makes the
 * next one's blocks step over the limit and walk up L1.
 *
 * It is reachable here with nothing exotic: at Mt=1, Nt=1, Kt=48, depth=48 the descriptor
 * pushes 48 pages into a 96-page (2*depth) buffer and leaves the pointer half way. The next
 * descriptor is then off the grid for good. `ft_tensix_cb_push_grid_is_safe()` carries the
 * derivation and the measured incident; resident_topk_reader.c refuses on exactly this and
 * this file had none of it.
 *
 * Both input CBs are tested, not just CB_IN0: the host places them independently and only
 * their page counts are supposed to match.
 */
static bool descriptor_is_executable(ftq_context *ctx, const coop_context *coop,
                                     bool cooperative, uint32_t tile_rows, uint32_t tile_cols,
                                     uint32_t tile_inner, uint32_t depth, uint32_t column_block,
                                     uint32_t row_block)
{
    if (tile_rows == 0u || tile_cols == 0u || tile_inner == 0u || depth == 0u ||
        column_block == 0u || row_block == 0u || IN0_TILE_BYTES == 0u || IN1_TILE_BYTES == 0u) {
        ctx->error = kErrBadParameters;
        FTMM_MOE_WHY(ctx, 1u, tile_rows, tile_cols);
    } else if (!ftmm_block_fits_dst(row_block, column_block)) {
        FTMM_MOE_WHY(ctx, 2u, row_block, column_block);
        /* MB * NB OUTPUT TILES ARE LIVE IN DST ACROSS THE WHOLE K LOOP, and DST has no
         * bounds check: a pass that asks for five would accumulate into the OTHER half of
         * the section, which the packer is reading, and every kernel would retire clean
         * with a wrong answer. Refused here for the same reason the host also refuses it —
         * the two failures cost differently, and only this one runs on silicon. */
        ctx->error = kErrBadParameters;
    } else if (!ft_tensix_cb_push_grid_is_safe(CB_IN0, depth * row_block) ||
             !ft_tensix_cb_push_grid_is_safe(CB_IN1, depth * column_block)) {
        FTMM_MOE_WHY(ctx, ft_tensix_cb_push_grid_is_safe(CB_IN0, depth * row_block) ? 4u : 3u,
                     ft_tensix_cb(CB_IN0)->fifo_write_pointer - (ft_tensix_cb(CB_IN0)->fifo_limit -
                                                                  ft_tensix_cb(CB_IN0)->fifo_size),
                     depth * row_block);
        /* THE BLOCKS ARE PART OF THE CHUNK SIZE, and the chunk size is what this guard is
         * about. CB_IN0 moves `depth * MB` pages per pass and CB_IN1 `depth * NB`, so a
         * residency whose CBs were sized for a different pair — or a host that changed
         * either mid-residency — is refused here rather than walking the write pointer off
         * the end of the buffer. */
        ctx->error = kErrBadParameters;
    } else if (row_block > 1u && cooperative)
        /* The rendezvous counts A BLOCKS and a row block changes what an A block IS: the
         * leader would multicast `depth * MB` tiles where the receivers expect `depth`.
         * Same reason as the column block below, one operand over. */
        ctx->error = kErrBadParameters;
    else if (column_block > 1u && cooperative)
        /* The rendezvous counts A BLOCKS, and blocking `nt` changes how many there are. Two
         * mechanisms that both move A's supply, agreed separately, is a wrong answer with
         * every status word clean; cooperation is disarmed on this model anyway (its
         * `Nt = 1` weights break the all-or-nothing rule), so this costs nothing. */
        ctx->error = kErrBadParameters;
    else if (cooperative != (coop->role != FT_TT_RESIDENT_COOP_ROLE_SOLO))
        /* COOPERATION IS ALL-OR-NOTHING FOR A RESIDENCY, and this is the line that makes
         * it so. A non-SOLO member that ran a solo descriptor would push CB_IN0 pages its
         * peers did not, and from then on every multicast would land in the wrong half of
         * their buffers with every kernel completing and every status word clean. Refusing
         * strands the triad and costs a re-arm, which is the cheap outcome. */
        ctx->error = kErrCoopRoleMismatch;
    else
        return true;
    return false;
}

/**
 * M4 PIPELINED READER (kMatmulBNMajor bit 8; N-major, solo, no starve only). Issues block k+1
 * into the CB's other half under its own NoC transaction id BEFORE waiting on block k's id,
 * so the DRAM never sees the per-block drain. CB_IN0/CB_IN1 hold exactly two chunks
 * (pn_tt_matmul_group.c: `depth * rows * 2`, `depth * block * 2`), so "the other half" is
 * the write pointer plus one chunk, wrapped. DRAM harness sweep M: 16 cores x 9 reads of
 * 4352 B = 407 GB/s with a barrier per block, 463 pipelined. Bit 9 = RED: never wait.
 */
static FT_BH_INLINE uint32_t ftmm_other_half(uint32_t cb, uint32_t address, uint32_t pages)
{
    uint32_t next = address + pages * ft_tensix_cb(cb)->fifo_page_size;

    if (next >= ft_tensix_cb(cb)->fifo_limit)
        next -= ft_tensix_cb(cb)->fifo_size;
    return next;
}

#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
/** B offset of descriptor column `n` at K-tile `kt` from the stack base, or FTMM_ZERO_COLUMN. */
static FT_BH_INLINE uint32_t ftmm_b_offset(uint32_t n, uint32_t kt, uint32_t tile_inner, uint32_t depth)
{
    if (ftmm_k_blocks != 0u) {
        const uint32_t offset = ftmm_offsets()[kt / depth];

        return offset == FTMM_ZERO_COLUMN ? offset : offset + (n * depth + kt % depth) * B_DRAM_TILE_BYTES;
    }
    if (ftmm_list) {
        const uint32_t cell = ftmm_list_cell0 + n;
        const uint32_t offset = ftmm_offsets()[cell / ftmm_list_columns - ftmm_list_s_base];

        return offset == FTMM_ZERO_COLUMN
                   ? offset
                   : offset + ((cell % ftmm_list_columns) * tile_inner + kt) * B_DRAM_TILE_BYTES;
    }
    return (n * tile_inner + kt) * B_DRAM_TILE_BYTES;
}
#endif

static void ftmm_issue_block(uint32_t mt, uint32_t rows, uint32_t nt, uint32_t columns,
                             uint32_t kt, uint32_t tile_inner, uint32_t depth,
                             uint32_t a_x, uint32_t a_y, uint32_t a_base, uint32_t b_x,
                             uint32_t b_y, uint32_t b_base, uint32_t a_write, uint32_t b_write,
                             bool a_half, bool red_face0, bool diag_face1_512, bool skip_a)
{
    /* BYTES A-REUSE: the CB_IN0 ring already holds this pass's A chunk (see
     * read_matmul_pipelined), so only B is fetched. */
    for (uint32_t row = 0; row < (skip_a ? 0u : rows); row++) {
        const uint32_t a_offset = ((mt + row) * tile_inner + kt) * IN0_TILE_BYTES;

        if (a_half) {
            /* M5: faces 0-1 (rows 0-15, bytes 0..1023) of each A tile — the half tile the
             * compute thread multiplies. RED (bit 11): face 1 never lands; must DIFFER. */
            for (uint32_t t = 0; t < depth; t++) {
                const uint32_t src = a_base + a_offset + t * IN0_TILE_BYTES;
                const uint32_t dst = a_write + (row * depth + t) * IN0_TILE_BYTES;

                ft_tensix_noc_read(ft_tensix_noc_address(a_x, a_y, src), dst,
                                   red_face0 ? 512u : 1024u);
            }
            (void)diag_face1_512;
            continue;
        }
        ft_tensix_noc_read(ft_tensix_noc_address(a_x, a_y, a_base + a_offset),
                           a_write + row * depth * IN0_TILE_BYTES, depth * IN0_TILE_BYTES);
    }
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    /* ONE offset a block, then a constant step a column: the offset arithmetic divides, and
     * these RISC-Vs pay for a division per column (S1 measured 0.309 -> 0.40 us a weight tile
     * with it inside this loop). A block's columns share one expert (K-experts: the block is
     * one expert's K range, so a column is `depth` tiles on; LIST: one column a block, refused
     * otherwise; plain: a column is `tile_inner` tiles on). */
    uint32_t b_next = ftmm_b_offset(nt, kt, tile_inner, depth);
    const uint32_t b_step = (ftmm_k_blocks != 0u ? depth : tile_inner) * B_DRAM_TILE_BYTES;

    if (b_next == FTMM_ZERO_COLUMN)
        columns = 0u; /* an entry past the list: zero-filled at expansion, nothing read */
#endif
    for (uint32_t column = 0; column < columns; column++) {
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
        const uint32_t b_offset = b_next;

        b_next += b_step;
#else
        const uint32_t b_offset = ((nt + column) * tile_inner + kt) * B_DRAM_TILE_BYTES;
#endif

#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
        /* `b_write` is the staging half here: the block lands raw, column after column. */
        ft_tensix_noc_read(ft_tensix_noc_address(b_x, b_y, b_base + b_offset),
                           b_write + column * depth * B_DRAM_TILE_BYTES, depth * B_DRAM_TILE_BYTES);
#else
        ft_tensix_noc_read(ft_tensix_noc_address(b_x, b_y, b_base + b_offset),
                           b_write + column * depth * B_TILE_BYTES, depth * B_DRAM_TILE_BYTES);
#endif
    }
}

#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
/**
 * Resolve an INDIRECT descriptor's B base: read the routing-slot segment holding entry
 * `rank`, bound-check the expert index, and add `idx * stride`. False refuses the
 * descriptor before any B byte is read (the caller drains it, then aborts).
 */
/** Slot entry `entry`, bound-checked; false if the residency has no slot or idx >= count. */
static bool ftmm_slot_expert(uint32_t entry, uint32_t *out_index)
{
    const uint32_t count = FTQ_RUNTIME_ARG(kArgMoeExpertCount);
    const uint32_t stride = FTQ_RUNTIME_ARG(kArgMoeExpertStride);
    const uint32_t scratch = FTQ_RUNTIME_ARG(kArgMoeScratch);
    const uint32_t segment = (entry * 4u) & ~(FT_TT_MOE_SLOT_SEGMENT_BYTES - 1u);

    if (count == 0u || scratch == 0u || (scratch & 63u) != 0u || (stride & 63u) != 0u ||
        entry >= FT_TT_MOE_SLOT_ENTRIES)
        return false;
    ft_tensix_noc_read(ft_tensix_noc_address(FTQ_RUNTIME_ARG(kArgMoeSlotX), FTQ_RUNTIME_ARG(kArgMoeSlotY),
                                             FTQ_RUNTIME_ARG(kArgMoeSlotAddress) + segment),
                       scratch, FT_TT_MOE_SLOT_SEGMENT_BYTES);
    ft_tensix_noc_read_barrier();
    *out_index = ((volatile uint32_t *)(uintptr_t)scratch)[(entry * 4u - segment) / 4u];
    return *out_index < count;
}

static bool ftmm_indirect_base(uint32_t flags, uint32_t tile_inner, uint32_t depth, uint32_t *b_base)
{
    const int32_t rank = ftmoe_rank(flags);
    const uint32_t stride = FTQ_RUNTIME_ARG(kArgMoeExpertStride);
    uint32_t index;

    ftmm_k_blocks = 0u;
    ftmm_list = false;
    if (rank < 0)
        return true;
    if ((flags & FT_TT_MOE_FLAG_K_EXPERTS) != 0u) {
        const uint32_t blocks = depth != 0u ? tile_inner / depth : 0u;

        if (blocks == 0u || blocks > FT_TT_MOE_K_EXPERTS_MAX)
            return false;
        for (uint32_t b = 0u; b < blocks; b++) {
            if (!ftmm_slot_expert((uint32_t)rank + b, &index))
                return false;
            ftmm_offsets()[b] = index * stride;
        }
        ftmm_k_blocks = blocks;
        return true;
    }
    if (!ftmm_slot_expert((uint32_t)rank, &index))
        return false;
    *b_base += index * stride;
    return true;
}
#endif

#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
/**
 * Resolve a LIST-mode descriptor (word a1, resident_matmul_moe_protocol.h "BATCHED PREFILL"):
 * copy slot entries 0..271 (the list and its count) once, bound-check every entry it uses,
 * and fill the offset table; entries >= n become FTMM_ZERO_COLUMN. False refuses it before
 * any B read (the caller drains it, then aborts).
 */
static bool ftmm_list_resolve(uint32_t a1, uint32_t flags, uint32_t tile_inner, uint32_t depth,
                              uint32_t tile_cols)
{
    const uint32_t count = FTQ_RUNTIME_ARG(kArgMoeExpertCount);
    const uint32_t stride = FTQ_RUNTIME_ARG(kArgMoeExpertStride);
    const uint32_t copy = FTQ_RUNTIME_ARG(kArgMoeScratch) + FT_TT_MOE_SLOT_COPY_OFFSET;
    volatile uint32_t *slot = (volatile uint32_t *)(uintptr_t)copy;
    volatile uint32_t *offsets = ftmm_offsets();
    const uint32_t s0 = a1 & 0xFFu, col0 = (a1 >> 8) & 0xFFu, columns = (a1 >> 16) & 0xFFu;
    uint32_t n, first, last;

    ftmm_k_blocks = 0u;
    ftmm_list = false;
    if (count == 0u || FTQ_RUNTIME_ARG(kArgMoeScratch) == 0u || (stride & 63u) != 0u)
        return false;
    ft_tensix_noc_read(ft_tensix_noc_address(FTQ_RUNTIME_ARG(kArgMoeSlotX), FTQ_RUNTIME_ARG(kArgMoeSlotY),
                                             FTQ_RUNTIME_ARG(kArgMoeSlotAddress)),
                       copy, FT_TT_MOE_SLOT_COPY_BYTES);
    ft_tensix_noc_read_barrier();
    n = slot[FT_TT_MOE_LIST_COUNT_ENTRY];
    if (n > FT_TT_MOE_LIST_MAX)
        return false;
    if ((flags & FT_TT_MOE_FLAG_K_EXPERTS) != 0u) {
        const uint32_t blocks = depth != 0u ? tile_inner / depth : 0u;

        if (blocks == 0u || s0 + blocks > FT_TT_MOE_LIST_MAX)
            return false;
        first = s0;
        last = s0 + blocks - 1u;
        ftmm_k_blocks = blocks;
    } else {
        if (columns == 0u || tile_cols == 0u)
            return false;
        ftmm_list_cell0 = s0 * columns + col0;
        ftmm_list_columns = columns;
        first = ftmm_list_cell0 / columns;
        last = (ftmm_list_cell0 + tile_cols - 1u) / columns;
        if (last >= FT_TT_MOE_LIST_MAX)
            return false;
        ftmm_list_s_base = first;
        ftmm_list = true;
    }
    for (uint32_t s = first; s <= last; s++) {
        if (s >= n) {
            offsets[s - first] = FTMM_ZERO_COLUMN;
            continue;
        }
        if (slot[s] >= count)
            return false;
        offsets[s - first] = slot[s] * stride;
    }
    return true;
}
#endif

/**
 * PHASE G0 WEIGHT STASH (descriptor a1 bit 31; bits 0..23 = L1 address >> 4): the member's
 * whole B slice (Nt_local x Kt tiles, N-major, contiguous) is read from DRAM ONCE into L1 at
 * the stash address, and every B read of every row block then comes from that L1 copy
 * through a LOCAL NoC read instead of DRAM. Same bytes, same order into CB_IN1: the compute
 * thread cannot tell. RED (a1 bit 30): the slice is loaded ONE TILE LATE (from b_base + one
 * tile) — every B tile shifts by one, and the answer must DIFFER. (Skipping the load is not a
 * red: L1 survives a re-boot, so a previous run's identical stash would still be there.)
 */
static bool ftmm_load_stash(ftq_context *ctx, uint32_t stash, uint32_t bytes, uint32_t b_x,
                            uint32_t b_y, uint32_t b_base, bool red_shift)
{
    if (ctx->aborting)
        return false;
    ft_tensix_noc_read(ft_tensix_noc_address(b_x, b_y, b_base + (red_shift ? B_TILE_BYTES : 0u)),
                       stash, bytes);
    ft_tensix_noc_read_barrier();
    return true;
}

static uint32_t ftmm_self_x, ftmm_self_y;

static bool read_matmul_pipelined(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                                  uint32_t depth, uint32_t row_block, uint32_t column_block,
                                  bool red_no_wait, uint32_t a1)
{
    const uint32_t tile_rows = parameters[kMatmulTileRows];
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t a_x = parameters[kMatmulAX];
    const uint32_t a_y = parameters[kMatmulAY];
    const uint32_t a_base = parameters[kMatmulABase];
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    /* The MoE build never stashes (the stash copies raw BFP6 bytes): a1 bit 31 is LIST mode. */
    const bool list_mode = (a1 & FT_TT_MOE_A1_LIST) != 0u;
    const bool stash = false;
#else
    const bool stash = (a1 & 0x80000000u) != 0u;
#endif
    const uint32_t b_x = stash ? ftmm_self_x : parameters[kMatmulBX];
    const uint32_t b_y = stash ? ftmm_self_y : parameters[kMatmulBY];
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    uint32_t b_base = stash ? (a1 & 0x00FFFFFFu) << 4 : parameters[kMatmulBBase];
#else
    const uint32_t b_base = stash ? (a1 & 0x00FFFFFFu) << 4 : parameters[kMatmulBBase];
#endif
    const uint32_t a_pages = depth * row_block;
    const uint32_t b_pages = depth * column_block;
    const uint32_t blocks_per_tile = tile_inner / depth;
    const uint32_t column_blocks = (tile_cols + column_block - 1u) / column_block;
    const uint32_t row_blocks = (tile_rows + row_block - 1u) / row_block;
    const uint32_t total = row_blocks * column_blocks * blocks_per_tile;
    uint32_t a_write, b_write, block;
    uint32_t trid = 1u;
    bool ok = true;
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    bool no_read = (parameters[kMatmulBNMajor] & kMatmulProbeNoRead) != 0u;
    bool refused = false, oversize = false;
#else
    const bool no_read = (parameters[kMatmulBNMajor] & kMatmulProbeNoRead) != 0u;
#endif
    /* The same condition the compute thread tests, from the same words. */
    const bool a_half = (parameters[kMatmulBNMajor] & kMatmulRow0) != 0u && tile_rows == 1u &&
                        row_block == 1u;
    const bool red_face0 = (parameters[kMatmulBNMajor] & (1u << 11)) != 0u;
    const bool diag_face1_512 = (parameters[kMatmulBNMajor] & (1u << 16)) != 0u;
    /* BYTES A-REUSE (word 13 bit 18): when CB_IN0's ring is EXACTLY Kt pages and MB = 1, the
     * chunk for (pass, kt) lands at the same L1 address on every pass over one A row, and
     * nothing but this reader writes the ring — so on every pass after a row's first (nt != 0)
     * pass 0's bytes are still there and the A read is skipped; the pages are pushed as
     * usual. A ring of any other size is refused here, not trusted. RED (bit 19): skip
     * regardless of the ring, so the compute sees another kt's A and the answer must DIFFER. */
    const uint32_t in0_ring_pages = ft_tensix_cb(CB_IN0)->fifo_size /
                                    ft_tensix_cb(CB_IN0)->fifo_page_size;
    const bool a_reuse = ((parameters[kMatmulBNMajor] & kMatmulAReuseRed) != 0u) ||
                         ((parameters[kMatmulBNMajor] & kMatmulAReuse) != 0u && row_block == 1u &&
                          in0_ring_pages == tile_inner);

#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    /* An indirect or LIST base is resolved here, once, before the first B read. */
    /* A refused index DRAINS the descriptor: the loop below runs as the no-read probe (every
     * page pushed, nothing read or expanded), so all three compute threads finish it together
     * on stale pages, and only then is it aborted (never retired). Returning before the first
     * push instead left UNPACK and MATH on different descriptors at the host's stop, and MATH
     * wedged the core's backend (card reset needed; 2026-10-28, test_q36_moe_matmul
     * moe_idx_bound). */
    if (!no_read && (list_mode ? !ftmm_list_resolve(a1, parameters[kMatmulBNMajor], tile_inner, depth,
                                                    tile_cols)
                               : !ftmm_indirect_base(parameters[kMatmulBNMajor], tile_inner, depth, &b_base))) {
        no_read = true;
        refused = true;
    }
    /* LIST mode reads one column a block, so a block is either all real or all zero. */
    if (list_mode && column_block != 1u && !refused) {
        no_read = true;
        refused = true;
        oversize = true;
    }
    /* A block larger than a staging half cannot land: drained and refused the same way. */
    if (b_pages > FT_TT_MOE_STAGE_TILES && !refused) {
        no_read = true;
        refused = true;
        oversize = true;
    }
#endif
    if (stash && !no_read &&
        !ftmm_load_stash(ctx, b_base, tile_cols * tile_inner * B_TILE_BYTES,
                         parameters[kMatmulBX], parameters[kMatmulBY], parameters[kMatmulBBase],
                         (a1 & 0x40000000u) != 0u))
        return false;
    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN0, (int32_t)a_pages, WAIT_MODE) ||
        !ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN1, (int32_t)b_pages, WAIT_MODE))
        return false;
    a_write = ft_tensix_cb_write_pointer(CB_IN0);
    b_write = ft_tensix_cb_write_pointer(CB_IN1);

    for (block = 0u; block < total; block++) {
        /* Block `block` -> (mt, nt, kt), the same order the unpipelined loop walks. */
        const uint32_t kt = (block % blocks_per_tile) * depth;
        const uint32_t pass = block / blocks_per_tile;
        const uint32_t mt = (pass / column_blocks) * row_block;
        const uint32_t nt = (pass % column_blocks) * column_block;
        const uint32_t rows = (tile_rows - mt) < row_block ? (tile_rows - mt) : row_block;
        const uint32_t columns = (tile_cols - nt) < column_block ? (tile_cols - nt) : column_block;

        if (block == 0u && !ctx->aborting && !no_read) {
            ft_tensix_noc_read_set_trid(trid);
            ftmm_issue_block(mt, rows, nt, columns, kt, tile_inner, depth, a_x, a_y, a_base,
                             b_x, b_y, b_base, a_write, FTMM_B_LANDING(0u, b_write), a_half,
                             red_face0, diag_face1_512, a_reuse && nt != 0u);
        }
        if (block + 1u < total) {
            const uint32_t next = block + 1u;
            const uint32_t nkt = (next % blocks_per_tile) * depth;
            const uint32_t npass = next / blocks_per_tile;
            const uint32_t nmt = (npass / column_blocks) * row_block;
            const uint32_t nnt = (npass % column_blocks) * column_block;
            const uint32_t nrows = (tile_rows - nmt) < row_block ? (tile_rows - nmt) : row_block;
            const uint32_t ncolumns =
                (tile_cols - nnt) < column_block ? (tile_cols - nnt) : column_block;

            /* BOTH halves free = the compute thread has popped block `block - 1`. */
            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN0, (int32_t)(2u * a_pages),
                                               WAIT_MODE) ||
                !ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN1, (int32_t)(2u * b_pages),
                                               WAIT_MODE))
                ok = false;
            if (ok && !ctx->aborting && !no_read) {
                ft_tensix_noc_read_set_trid(trid ^ 3u);
                ftmm_issue_block(nmt, nrows, nnt, ncolumns, nkt, tile_inner, depth, a_x, a_y,
                                 a_base, b_x, b_y, b_base,
                                 ftmm_other_half(CB_IN0, a_write, a_pages),
                                 FTMM_B_LANDING(next & 1u, ftmm_other_half(CB_IN1, b_write, b_pages)),
                                 a_half,
                                 red_face0, diag_face1_512, a_reuse && nnt != 0u);
            }
        }
        if (!red_no_wait)
            ft_tensix_noc_read_barrier_trid(trid);
        if (!ok) {
            ft_tensix_noc_read_barrier();
            ft_tensix_noc_read_set_trid(0u);
            return false;
        }
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
        /* EVERY block is a job (a drained one with no tiles), so the helpers' job count
         * stays the block count; a wait abandoned here leaves the descriptor aborting. */
        /* A block past the list (LIST mode) is zero-filled: source 0. */
        (void)ftmoe_expand_job(ctx,
                               ftmm_b_offset(nt, kt, tile_inner, depth) == FTMM_ZERO_COLUMN ? 0u
                                                                                       : ftmm_stage(block & 1u),
                               b_write, (!ctx->aborting && !no_read) ? columns * depth : 0u);
#endif
        ft_tensix_cb_push_back(CB_IN0, (int32_t)a_pages);
        ft_tensix_cb_push_back(CB_IN1, (int32_t)b_pages);
        a_write = ft_tensix_cb_write_pointer(CB_IN0);
        b_write = ft_tensix_cb_write_pointer(CB_IN1);
        trid ^= 3u;
    }
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_read_set_trid(0u);
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    if (refused) {
        FTMM_MOE_WHY(ctx, oversize ? 7u : 9u, b_pages, column_block);
        ftq_abort(ctx, oversize ? kErrBadParameters : kErrMoeExpertIndex);
    }
#endif
    return !ctx->aborting;
}

#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
/** Descriptor a2 of the descriptor being read: the up shard's bank coordinate. */
static uint32_t ftsw_reader_a2;

/**
 * The FUSED reader: read_matmul_pipelined() at Mt = MB = 1, no stash, over 2 * ceil(Nt/NB)
 * passes -- pass q is panel q & 1 (0 the gate shard in B, 1 the up shard at a1/a2) of column
 * block q >> 1, every K block of it. Same block issue, same transaction ids, same barriers,
 * same CB grid; only the pass -> (panel, column) map and the B triple per panel are new. A
 * copy rather than a shared helper, so the shipped pipelined reader is untouched.
 *
 * A-REUSE: the A ring holds the row after pass 0, so every later pass skips the A read
 * (the unfused rule, `nt != 0`, would re-read it for pass 1, the up panel of column 0).
 */
static bool read_swiglu_pipelined(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                                  uint32_t depth, uint32_t column_block, bool red_no_wait,
                                  uint32_t up_base, uint32_t up_xy)
{
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t a_x = parameters[kMatmulAX];
    const uint32_t a_y = parameters[kMatmulAY];
    const uint32_t a_base = parameters[kMatmulABase];
    const uint32_t b_x[2] = { parameters[kMatmulBX], ftsw_up_x(up_xy) };
    const uint32_t b_y[2] = { parameters[kMatmulBY], ftsw_up_y(up_xy) };
    const uint32_t b_base[2] = { parameters[kMatmulBBase], up_base };
    const uint32_t a_pages = depth;
    const uint32_t b_pages = depth * column_block;
    const uint32_t blocks_per_tile = tile_inner / depth;
    const uint32_t total = ftsw_passes(tile_cols, column_block) * blocks_per_tile;
    const bool no_read = (parameters[kMatmulBNMajor] & kMatmulProbeNoRead) != 0u;
    const bool a_half = (parameters[kMatmulBNMajor] & kMatmulRow0) != 0u;
    const bool red_face0 = (parameters[kMatmulBNMajor] & (1u << 11)) != 0u;
    const bool diag_face1_512 = (parameters[kMatmulBNMajor] & (1u << 16)) != 0u;
    const uint32_t in0_ring_pages = ft_tensix_cb(CB_IN0)->fifo_size /
                                    ft_tensix_cb(CB_IN0)->fifo_page_size;
    const bool a_reuse = ((parameters[kMatmulBNMajor] & kMatmulAReuseRed) != 0u) ||
                         ((parameters[kMatmulBNMajor] & kMatmulAReuse) != 0u &&
                          in0_ring_pages == tile_inner);
    uint32_t a_write, b_write, block;
    uint32_t trid = 1u;
    bool ok = true;

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN0, (int32_t)a_pages, WAIT_MODE) ||
        !ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN1, (int32_t)b_pages, WAIT_MODE))
        return false;
    a_write = ft_tensix_cb_write_pointer(CB_IN0);
    b_write = ft_tensix_cb_write_pointer(CB_IN1);

    for (block = 0u; block < total; block++) {
        const uint32_t kt = (block % blocks_per_tile) * depth;
        const uint32_t pass = block / blocks_per_tile;
        const uint32_t panel = ftsw_pass_panel(pass);

        if (block == 0u && !ctx->aborting && !no_read) {
            ft_tensix_noc_read_set_trid(trid);
            ftmm_issue_block(0u, 1u, ftsw_pass_column(pass, column_block),
                             ftsw_pass_columns(pass, tile_cols, column_block), kt, tile_inner,
                             depth, a_x, a_y, a_base, b_x[panel], b_y[panel], b_base[panel],
                             a_write, FTMM_B_LANDING(0u, b_write), a_half, red_face0,
                             diag_face1_512, a_reuse && pass != 0u);
        }
        if (block + 1u < total) {
            const uint32_t next = block + 1u;
            const uint32_t nkt = (next % blocks_per_tile) * depth;
            const uint32_t npass = next / blocks_per_tile;
            const uint32_t npanel = ftsw_pass_panel(npass);

            /* BOTH halves free = the compute thread has popped block `block - 1`. */
            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN0, (int32_t)(2u * a_pages),
                                               WAIT_MODE) ||
                !ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN1, (int32_t)(2u * b_pages),
                                               WAIT_MODE))
                ok = false;
            if (ok && !ctx->aborting && !no_read) {
                ft_tensix_noc_read_set_trid(trid ^ 3u);
                ftmm_issue_block(0u, 1u, ftsw_pass_column(npass, column_block),
                                 ftsw_pass_columns(npass, tile_cols, column_block), nkt,
                                 tile_inner, depth, a_x, a_y, a_base, b_x[npanel], b_y[npanel],
                                 b_base[npanel], ftmm_other_half(CB_IN0, a_write, a_pages),
                                 FTMM_B_LANDING(next & 1u,
                                                ftmm_other_half(CB_IN1, b_write, b_pages)),
                                 a_half, red_face0, diag_face1_512, a_reuse && npass != 0u);
            }
        }
        if (!red_no_wait)
            ft_tensix_noc_read_barrier_trid(trid);
        if (!ok) {
            ft_tensix_noc_read_barrier();
            ft_tensix_noc_read_set_trid(0u);
            return false;
        }
        ft_tensix_cb_push_back(CB_IN0, (int32_t)a_pages);
        ft_tensix_cb_push_back(CB_IN1, (int32_t)b_pages);
        a_write = ft_tensix_cb_write_pointer(CB_IN0);
        b_write = ft_tensix_cb_write_pointer(CB_IN1);
        trid ^= 3u;
    }
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_read_set_trid(0u);
    return !ctx->aborting;
}
#endif

#if defined(FT_RTRIAD_RESADD) && FT_RTRIAD_RESADD
/**
 * The RESADD reader (S3a): read_swiglu_pipelined()'s walk with ONE pass per column block (the
 * projection's own B), plus the residual. Once the first K block of column block c has landed,
 * the block's w residual tiles (columns c*NB .. c*NB+w-1 of the strip at a1/a2) are read into
 * the RESIDUAL CB and NB pages pushed (pad after a short block); the compute needs them only
 * after that block's K loop. One full barrier a column block (it also waits the prefetched next
 * K block, which was due anyway).
 *
 * A-REUSE: as unfused, every column block after the first skips the A read.
 */
static bool read_resadd_pipelined(ftq_context *ctx, volatile FTQ_L1_PTR uint32_t *parameters,
                                  uint32_t depth, uint32_t column_block, bool red_no_wait,
                                  uint32_t residual_base, uint32_t residual_xy)
{
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t a_x = parameters[kMatmulAX];
    const uint32_t a_y = parameters[kMatmulAY];
    const uint32_t a_base = parameters[kMatmulABase];
    const uint32_t b_x = parameters[kMatmulBX];
    const uint32_t b_y = parameters[kMatmulBY];
    const uint32_t b_base = parameters[kMatmulBBase];
    const uint32_t r_x = ftra_residual_x(residual_xy);
    const uint32_t r_y = ftra_residual_y(residual_xy);
    const uint32_t r_page = ft_tensix_cb(FT_TT_MATMUL_RESADD_CB_RESIDUAL)->fifo_page_size;
    const uint32_t r_pages = ftra_residual_block_pages(column_block);
    const uint32_t a_pages = depth;
    const uint32_t b_pages = depth * column_block;
    const uint32_t blocks_per_tile = tile_inner / depth;
    const uint32_t total = ftra_blocks(tile_cols, column_block) * blocks_per_tile;
    const bool no_read = (parameters[kMatmulBNMajor] & kMatmulProbeNoRead) != 0u;
    const bool a_half = (parameters[kMatmulBNMajor] & kMatmulRow0) != 0u;
    const bool red_face0 = (parameters[kMatmulBNMajor] & (1u << 11)) != 0u;
    const bool diag_face1_512 = (parameters[kMatmulBNMajor] & (1u << 16)) != 0u;
    const uint32_t in0_ring_pages = ft_tensix_cb(CB_IN0)->fifo_size /
                                    ft_tensix_cb(CB_IN0)->fifo_page_size;
    const bool a_reuse = ((parameters[kMatmulBNMajor] & kMatmulAReuseRed) != 0u) ||
                         ((parameters[kMatmulBNMajor] & kMatmulAReuse) != 0u &&
                          in0_ring_pages == tile_inner);
    uint32_t a_write, b_write, block;
    uint32_t trid = 1u;
    bool ok = true;

    if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN0, (int32_t)a_pages, WAIT_MODE) ||
        !ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN1, (int32_t)b_pages, WAIT_MODE))
        return false;
    a_write = ft_tensix_cb_write_pointer(CB_IN0);
    b_write = ft_tensix_cb_write_pointer(CB_IN1);

    for (block = 0u; block < total; block++) {
        const uint32_t kt = (block % blocks_per_tile) * depth;
        const uint32_t pass = block / blocks_per_tile;

        if (block == 0u && !ctx->aborting && !no_read) {
            ft_tensix_noc_read_set_trid(trid);
            ftmm_issue_block(0u, 1u, pass * column_block,
                             ftra_block_columns(pass, tile_cols, column_block), kt, tile_inner,
                             depth, a_x, a_y, a_base, b_x, b_y, b_base, a_write,
                             FTMM_B_LANDING(0u, b_write), a_half, red_face0, diag_face1_512,
                             a_reuse && pass != 0u);
        }
        if (block + 1u < total) {
            const uint32_t next = block + 1u;
            const uint32_t nkt = (next % blocks_per_tile) * depth;
            const uint32_t npass = next / blocks_per_tile;

            /* BOTH halves free = the compute thread has popped block `block - 1`. */
            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN0, (int32_t)(2u * a_pages),
                                               WAIT_MODE) ||
                !ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN1, (int32_t)(2u * b_pages),
                                               WAIT_MODE))
                ok = false;
            if (ok && !ctx->aborting && !no_read) {
                ft_tensix_noc_read_set_trid(trid ^ 3u);
                ftmm_issue_block(0u, 1u, npass * column_block,
                                 ftra_block_columns(npass, tile_cols, column_block), nkt,
                                 tile_inner, depth, a_x, a_y, a_base, b_x, b_y, b_base,
                                 ftmm_other_half(CB_IN0, a_write, a_pages),
                                 FTMM_B_LANDING(next & 1u,
                                                ftmm_other_half(CB_IN1, b_write, b_pages)),
                                 a_half, red_face0, diag_face1_512, a_reuse && npass != 0u);
            }
        }
        if (!red_no_wait)
            ft_tensix_noc_read_barrier_trid(trid);
        /* THE RESIDUAL of column block `pass`, once, at its first K block. */
        if (ok && kt == 0u) {
            if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)FT_TT_MATMUL_RESADD_CB_RESIDUAL,
                                               (int32_t)r_pages, WAIT_MODE))
                ok = false;
            if (ok && !ctx->aborting && !no_read) {
                const uint32_t r_write = ft_tensix_cb_write_pointer(FT_TT_MATMUL_RESADD_CB_RESIDUAL);
                const uint32_t first = pass * column_block;
                const uint32_t columns = ftra_block_columns(pass, tile_cols, column_block);

                ft_tensix_noc_read_set_trid(0u);
                for (uint32_t j = 0u; j < columns; j++)
                    ft_tensix_noc_read(
                        ft_tensix_noc_address(r_x, r_y,
                                              residual_base + ftra_residual_offset(first + j, r_page)),
                        r_write + j * r_page, r_page);
                ft_tensix_noc_read_barrier();
            }
            if (ok)
                ft_tensix_cb_push_back(FT_TT_MATMUL_RESADD_CB_RESIDUAL, (int32_t)r_pages);
        }
        if (!ok) {
            ft_tensix_noc_read_barrier();
            ft_tensix_noc_read_set_trid(0u);
            return false;
        }
        ft_tensix_cb_push_back(CB_IN0, (int32_t)a_pages);
        ft_tensix_cb_push_back(CB_IN1, (int32_t)b_pages);
        a_write = ft_tensix_cb_write_pointer(CB_IN0);
        b_write = ft_tensix_cb_write_pointer(CB_IN1);
        trid ^= 3u;
    }
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_read_set_trid(0u);
    return !ctx->aborting;
}
#endif

/**
 * The reader half of one matmul descriptor: matmul_reader.c's loop, with a stoppable
 * reserve and the drain rule.
 *
 * `starve_blocks` is a DEBUG field and the only thing here that is not in the Phase 4
 * kernel: it makes the reader skip that many trailing K-blocks of the LAST output tile, so
 * the compute thread is left genuinely blocked in the real `cb_wait_front` of the real op
 * rather than in a synthetic wait built to be easy to break. That distinction is the whole
 * point of the acceptance test -- a stop protocol proven against a wait nobody would ever
 * write is not proven.
 *
 * @return true if the whole descriptor was read; false if a wait was abandoned, in which
 *         case the loop still ran to completion (rule 3) and the caller must not retire it
 */
static bool read_one_matmul(ftq_context *ctx, coop_context *coop,
                            volatile FTQ_L1_PTR uint32_t *parameters, uint32_t starve_blocks)
{
    const uint32_t cooperative = parameters[kMatmulCooperative];
    const uint32_t tile_rows = parameters[kMatmulTileRows];
    const uint32_t tile_inner = parameters[kMatmulTileInner];
    const uint32_t tile_cols = parameters[kMatmulTileCols];
    const uint32_t a_x = parameters[kMatmulAX];
    const uint32_t a_y = parameters[kMatmulAY];
    const uint32_t a_base = parameters[kMatmulABase];
    const uint32_t b_x = parameters[kMatmulBX];
    const uint32_t b_y = parameters[kMatmulBY];
    const uint32_t b_base = parameters[kMatmulBBase];
    const uint32_t requested_depth = parameters[kMatmulDepth];
    const uint32_t b_n_major = parameters[kMatmulBNMajor] & 1u;
    const bool pipelined = (parameters[kMatmulBNMajor] & (1u << 8)) != 0u;
    const bool red_no_wait = (parameters[kMatmulBNMajor] & (1u << 9)) != 0u;
    const uint32_t block_word = parameters[kMatmulColumnBlock];

    // A depth that does not divide Kt would make a later block straddle the CB wrap, so the
    // kernel refuses to guess: anything the host did not make legal collapses to the
    // per-pair behaviour, which is always correct (matmul_reader.c).
    const uint32_t depth =
        (requested_depth > 0u && tile_inner % requested_depth == 0u) ? requested_depth : 1u;
    // 0 is what every descriptor written before the field existed says, and it means one --
    // on BOTH halves of the word (resident_matmul_block_protocol.h).
    const uint32_t column_block = ftmm_column_block(block_word);
    const uint32_t row_block = ftmm_row_block(block_word);
#if defined(FT_RTRIAD_MIXW) && FT_RTRIAD_MIXW
    if (!ftmm_select_weight_tile(ctx, parameters[kMatmulBNMajor]))
        return false;
#endif
    if (!descriptor_is_executable(ctx, coop, cooperative != 0u, tile_rows, tile_cols, tile_inner,
                                  depth, column_block, row_block))
        return false;

#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
    /* A FUSED descriptor: a1 is the up shard's base, not a starve count or a stash. Anything
     * but the one shape the fused reader serves is refused, never read as a plain matmul. */
    if ((block_word & FT_TT_MATMUL_SWIGLU_FLAG) != 0u) {
        /* The panel CB must hold whole 2*NB-page blocks: the compute's indexed unpacks
         * assume every block starts on that grid (protocol header, card fix 1). */
        if (!pipelined || !b_n_major || cooperative != 0u || tile_rows != 1u || row_block != 1u ||
            !ft_tensix_cb_push_grid_is_safe(FT_TT_MATMUL_SWIGLU_CB_PANEL,
                                            ftsw_panel_pages(column_block))) {
            ctx->error = kErrBadParameters;
            return false;
        }
        return read_swiglu_pipelined(ctx, parameters, depth, column_block, red_no_wait,
                                     starve_blocks, ftsw_reader_a2);
    }
#endif
#if defined(FT_RTRIAD_RESADD) && FT_RTRIAD_RESADD
    /* A RESADD descriptor: a1/a2 are the residual strip, not a starve count or a stash. The
     * panel CB must hold whole 2*NB blocks and the residual CB whole NB blocks (protocol). */
    if ((block_word & FT_TT_MATMUL_RESADD_FLAG) != 0u) {
        if (!pipelined || !b_n_major || cooperative != 0u || tile_rows != 1u || row_block != 1u ||
            (block_word & FT_TT_MATMUL_SWIGLU_FLAG) != 0u ||
            !ft_tensix_cb_push_grid_is_safe(FT_TT_MATMUL_SWIGLU_CB_PANEL,
                                            ftra_panel_block_pages(column_block)) ||
            !ft_tensix_cb_push_grid_is_safe(FT_TT_MATMUL_RESADD_CB_RESIDUAL,
                                            ftra_residual_block_pages(column_block))) {
            ctx->error = kErrBadParameters;
            return false;
        }
        return read_resadd_pipelined(ctx, parameters, depth, column_block, red_no_wait,
                                     starve_blocks, ftsw_reader_a2);
    }
#endif
    if (pipelined && b_n_major && cooperative == 0u &&
        (starve_blocks == 0u || (starve_blocks & 0x80000000u) != 0u))
        return read_matmul_pipelined(ctx, parameters, depth, row_block, column_block,
                                     red_no_wait, starve_blocks);
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    /* The MoE build expands and resolves indirect bases on the pipelined path only; the
     * unpipelined loop below reads B tiles as the CB's own format. Refused, not guessed. */
    ctx->error = kErrBadParameters;
    FTMM_MOE_WHY(ctx, 6u, parameters[kMatmulBNMajor], starve_blocks);
    return false;
#endif

    const uint32_t blocks_per_tile = tile_inner / depth;
    // The LAST block of either axis is short when the block does not divide the extent. The
    // CHUNK does not shorten with it — only the fill does — because a chunk size that
    // changes walks the write pointer off the grid the next descriptor needs
    // (ft_tensix_cb_block_grid_is_safe).
    const uint32_t column_blocks = (tile_cols + column_block - 1u) / column_block;
    const uint32_t row_blocks = (tile_rows + row_block - 1u) / row_block;
    const uint32_t passes = row_blocks * column_blocks;
    uint32_t block_index = 0u;
    const uint32_t starve_from = (starve_blocks == 0u || starve_blocks > blocks_per_tile)
                                     ? 0xFFFFFFFFu
                                     : passes * blocks_per_tile - starve_blocks;

    for (uint32_t mt = 0; mt < tile_rows; mt += row_block) {
        const uint32_t rows = (tile_rows - mt) < row_block ? (tile_rows - mt) : row_block;

        for (uint32_t nt = 0; nt < tile_cols; nt += column_block) {
            const uint32_t columns =
                (tile_cols - nt) < column_block ? (tile_cols - nt) : column_block;

            for (uint32_t kt = 0; kt < tile_inner; kt += depth, block_index++) {
                uint32_t a_write;
                uint32_t b_write;
                const bool starved = block_index >= starve_from;

                if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN0,
                                                   (int32_t)(depth * row_block), WAIT_MODE))
                    return false;
                if (!ftq_cb_reserve_back_stoppable(ctx, (int32_t)CB_IN1,
                                                   (int32_t)(depth * column_block), WAIT_MODE))
                    return false;
                a_write = ft_tensix_cb_write_pointer(CB_IN0);
                b_write = ft_tensix_cb_write_pointer(CB_IN1);

                // Rule 3: while draining, issue NOTHING. The pushes below still happen so
                // the page counters stay balanced, but a transaction issued during a drain
                // would still be in flight when the kernel returns.
                if (!ctx->aborting && !starved) {
                    const bool receiving = cooperative != 0u &&
                                           coop->role == FT_TT_RESIDENT_COOP_ROLE_RECEIVER;

                    if (cooperative != 0u &&
                        a_write != coop_expected_write_pointer(CB_IN0, depth * row_block,
                                                               coop->counter)) {
                        // The address the LEADER is about to multicast into is its own
                        // write pointer, so a member whose pointer has drifted would take
                        // the block in the wrong half of its buffer. Caught before
                        // anything is issued; see the protocol header on why this is the
                        // local half of the check and the counter is the other half.
                        ftq_abort(ctx, kErrCoopWritePointer);
                    } else {
                        // ONE A READ PER ROW FOR THE WHOLE COLUMN BLOCK, and that is the
                        // point of the column blocking: the offset never depended on `nt`,
                        // so at NB columns per pass the A column is fetched Nt/NB times
                        // instead of Nt.
                        //
                        // ROW-MAJOR IN L1: row i's `depth` tiles land at `i * depth`, which
                        // is the layout the compute thread's `tile_index_a` walks and the
                        // mirror of what the reader does with B. A row is one read of
                        // `depth` contiguous tiles, exactly as it was at MB = 1 — the row
                        // block adds reads, never shrinks one.
                        if (!receiving) {
                            for (uint32_t row = 0; row < rows; row++) {
                                const uint32_t a_offset =
                                    ((mt + row) * tile_inner + kt) * IN0_TILE_BYTES;

                                ft_tensix_noc_read(
                                    ft_tensix_noc_address(a_x, a_y, a_base + a_offset),
                                    a_write + row * depth * IN0_TILE_BYTES,
                                    depth * IN0_TILE_BYTES);
                            }
                        }
                        // COLUMN-MAJOR IN L1: column j's `depth` tiles land at `j * depth`.
                        // That keeps the N-major panel's per-column transaction whole — one
                        // read of `depth` contiguous tiles, the size every recorded number
                        // was taken at — and the compute thread's MOP walks the columns with
                        // an SrcA stride of `depth` tiles instead of one.
                        for (uint32_t column = 0; column < columns; column++) {
                            const uint32_t b_column = nt + column;
                            const uint32_t b_land = b_write + column * depth * B_TILE_BYTES;

                            if (b_n_major) {
                                const uint32_t b_offset =
                                    (b_column * tile_inner + kt) * B_TILE_BYTES;

                                ft_tensix_noc_read(
                                    ft_tensix_noc_address(b_x, b_y, b_base + b_offset), b_land,
                                    depth * B_TILE_BYTES);
                            } else {
                                for (uint32_t step = 0; step < depth; step++) {
                                    const uint32_t b_tile = (kt + step) * tile_cols + b_column;
                                    const uint32_t b_offset = b_tile * B_TILE_BYTES;

                                    ft_tensix_noc_read(
                                        ft_tensix_noc_address(b_x, b_y, b_base + b_offset),
                                        b_land + step * B_TILE_BYTES, B_TILE_BYTES);
                                }
                            }
                        }
                        if (receiving) {
                            // The B read is already ISSUED and not yet waited on, which is
                            // the whole point: this core's own DRAM traffic and the
                            // leader's multicast are in flight together. A receiver reads
                            // no A at all, which is why the row loop above is skipped
                            // wholesale rather than per row.
                            coop_receiver_block(ctx, coop);
                            // Taken whether or not the rendezvous succeeded — the B read
                            // was issued, and a transaction left outstanding is how L1 the
                            // next kernel is about to use gets written underneath it.
                            ft_tensix_noc_read_barrier();
                        } else {
                            // ONE barrier for the whole block: it is per-core, not
                            // per-transaction.
                            ft_tensix_noc_read_barrier();
                            if (cooperative != 0u)
                                coop_leader_block(
                                    ctx, coop, a_write, depth * row_block * IN0_TILE_BYTES,
                                    cooperative ==
                                        FT_TT_RESIDENT_COOP_DESCRIPTOR_RED_SKIP_PAYLOAD);
                        }
                    }
                }
                // Advanced on EVERY cooperative block, including one that aborted, so two
                // members that gave up at different points still agree on what block
                // number they are at. A launch resets it, and a launch is what the host
                // does after any refusal.
                if (cooperative != 0u)
                    coop->counter = ftcoop_next_counter(coop->counter);

                if (starved) {
                    // The starved block is never pushed, which is precisely what leaves the
                    // compute thread in cb_wait_front. Everything else about the descriptor
                    // is a normal, correct matmul.
                    return false;
                }
                // The FULL chunk on BOTH inputs, even on a short last block of either axis:
                // the pad pages carry whatever L1 held and the compute thread never names
                // them, but the grid closes.
                ft_tensix_cb_push_back(CB_IN0, depth * row_block);
                ft_tensix_cb_push_back(CB_IN1, depth * column_block);
            }
        }
    }
    if (cooperative != 0u) {
        /* NOTHING OUTSTANDING ACROSS A DESCRIPTOR BOUNDARY. The next descriptor may be a
         * different shape reusing the same circular buffers, and an abort leaves whatever
         * the failing block had already issued in flight. The leader's multicasts are
         * non-posted writes and the receivers' announcements are remote atomics, so both
         * counters are drained here — cheap, because in the normal case the rendezvous has
         * already forced them to have landed. */
        ft_tensix_noc_write_barrier();
        ft_tensix_noc_atomic_barrier();
    }
    return !ctx->aborting;
}

/**
 * The opcodes that need no circular buffer. Identical in meaning to resident_queue.c's,
 * and the reader is the processor that executes them because it is the one that owns
 * `tail` -- the writer's cursor follows `tail`, so a writer cannot retire one of these
 * before the reader has performed it.
 */
static bool execute_simple_opcode(ftq_context *ctx, uint32_t opcode, uint32_t a0, uint32_t a1,
                                  uint32_t a2,
                                  uint32_t a3, uint32_t a4, bool *keep_running)
{
    switch (opcode) {
    case kOpNop:
        return true;

    case kOpWriteWord:
        *(volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(a0) = a1;
        return true;

    case kOpL1Copy:
        /* Every word of this transfer arrived on the wire — see
         * ft_tensix_l1_copy_is_issuable(). Refuse, never issue-and-hope: a zero length is
         * the card-down shape and a low destination overwrites the other core's mailboxes. */
        if (!ft_tensix_l1_copy_is_issuable(a3, a2, a4)) {
            ctx->error = kErrBadParameters;
            *keep_running = false;
            return false;
        }
        ft_tensix_noc_write(a3, ft_tensix_noc_address(a0, a1, a2), a4);
        ft_tensix_noc_write_barrier();
        return true;

    case kOpSemPost:
        ft_tensix_noc_semaphore_increment(ft_tensix_noc_address(a0, a1, a2), a3);
        return true;

    case kOpSemWait: {
        volatile FTQ_L1_PTR uint32_t *semaphore = (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(a0);
        const uint32_t bound = (a2 != 0u) ? a2 : ctx->idle_spin_limit;
        uint32_t spins = 0u;

        ftq_status_set(ctx, kStateSemWait);
        while (*semaphore < a1) {
            ctx->heartbeat++;
            if (ctx->status)
                ctx->status[kStatusHeartbeat] = ctx->heartbeat;
            if (ftq_stop_requested(ctx)) {
                ftq_abort(ctx, kErrNone);
                *keep_running = false;
                return false;
            }
            if (++spins >= bound) {
                ftq_abort(ctx, kErrSemWaitTimeout);
                *keep_running = false;
                return false;
            }
        }
        ftq_status_set(ctx, kStateExec);
        return true;
    }

    case kOpDelay: {
        volatile uint32_t counter = 0;

        for (uint32_t iteration = 0; iteration < a0; iteration++)
            counter = counter + 1;
        return true;
    }

    case kOpStop:
        *keep_running = false;
        return true;

    default:
        ctx->error = kErrBadOpcode;
        *keep_running = false;
        return false;
    }
}

void kernel_main(void)
{
    ftq_context ctx;
    coop_context coop;
    bool running = true;

    ftq_context_init(&ctx);
    coop_read_arguments(&coop);
#if defined(FT_RTRIAD_MOE) && FT_RTRIAD_MOE
    ftmoe_boot();
#endif
    ftq_status_set(&ctx, kStateBoot);
    ctx.control[kFieldState] = kStateBoot;

    if (!ftq_check_control(&ctx)) {
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    // RESUME rather than restart, so a retired triad can be relaunched. All five processors
    // resume from `completed_count`, the ONE number that means "every stage finished this
    // descriptor" -- resuming from `tail` would let the reader and the writer restart at
    // different descriptors, because the writer legitimately trails the reader by the CB
    // pipeline's depth. A relaunch is therefore only defined over a queue whose head, tail
    // and completed_count agree: a fresh one, or one that was drained before it was
    // retired. Anything else is refused here rather than run.
    if (ctx.control[kFieldTail] != ctx.control[kFieldCompletedCount]) {
        ctx.error = kErrBadParameters;
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }
    /* THE COOPERATIVE CONFIGURATION IS REFUSED AT LAUNCH, not at the first descriptor.
     * A rectangle with a coordinate that does not fit the 6-bit node-id field does not
     * fail: it TRUNCATES onto a different live endpoint, and on this board a wrong
     * endpoint is a hung NoC and a walk to the machine. So the geometry is checked once,
     * here, before anything can be issued. */
    if (!coop_configuration_is_legal(&coop)) {
        ctx.error = kErrCoopBadPartition;
        ctx.control[kFieldError] = ctx.error;
        ctx.control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }
    ctx.cursor = ctx.control[kFieldCompletedCount];
    {
        /* G0 stash: this core's own NoC coordinate (upstream my_x/my_y from NOC_NODE_ID). */
        const uint32_t id = ft_bh_noc_cmd_read(NOC_INDEX, 0u,
                                                       FT_BH_NOC_REG_BASE + 0x44u);

        ftmm_self_x = id & FT_BH_NOC_NODE_ID_MASK;
        ftmm_self_y = (id >> FT_BH_NOC_NODE_ID_BITS) & FT_BH_NOC_NODE_ID_MASK;
    }

    while (running) {
        if (!ftq_wait_for_work(&ctx, kFieldHead))
            break;

        ctx.aborting = false;
        ftq_status_set(&ctx, kStateExec);
        /* S2: the producer this descriptor's sidecar entry names has completed, before any
         * input is read (resident_precondition.h). Refused or stopped: leave, unconsumed. */
        if (!ftq_precondition_wait(&ctx)) {
            running = false;
            continue;
        }

        volatile FTQ_L1_PTR uint32_t *descriptor = ftq_descriptor(&ctx);
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t a0 = descriptor[kDescA0];
        const uint32_t a1 = descriptor[kDescA1];
        const uint32_t a2 = descriptor[kDescA2];
        const uint32_t a3 = descriptor[kDescA3];
        const uint32_t a4 = descriptor[kDescA4];
        bool done;

        ctx.control[kFieldLastOpcode] = opcode;
        ftq_status_opcode(&ctx, opcode);

        if (opcode == kOpMatmul) {
            ftq_status_set(&ctx, kStateCbReserveBack);
#if defined(FT_RTRIAD_SWIGLU) && FT_RTRIAD_SWIGLU
            ftsw_reader_a2 = a2;
#endif
            done = read_one_matmul(&ctx, &coop,
                                   (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(a0), a1);
            ftq_status_set(&ctx, kStateExec);
            if (!done)
                running = false;
        } else {
            done = execute_simple_opcode(&ctx, opcode, a0, a1, a2, a3, a4, &running);
        }

        // Rule 5. An abandoned descriptor was handed to nobody, so it advances NOTHING and
        // the next launch re-attempts it. Only a descriptor the reader finished becomes
        // visible to the writer.
        if (done) {
            ctx.cursor++;
            ctx.control[kFieldTail] = ctx.cursor;
        }
    }

    // Nothing is outstanding here: the drain path issues no transactions, and a
    // cooperative descriptor drains its own multicasts and atomics before it returns. The
    // barriers are the cheap statement of that invariant rather than a fix for a case that
    // can happen. The ATOMIC one is here because a receiver's readiness announcement is a
    // remote atomic, and the stock BRISC firmware asserts every NoC counter is settled after this returns.
    ft_tensix_noc_read_barrier();
    ft_tensix_noc_write_barrier();
    ft_tensix_noc_atomic_barrier();
    ctx.control[kFieldError] = ctx.error;
    ctx.control[kFieldState] = kStateExit;
    ftq_status_set(&ctx, kStateExit);
}
