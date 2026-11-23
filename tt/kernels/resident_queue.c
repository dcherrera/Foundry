/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file resident_queue.c
 * @brief The Phase 5 resident kernel: it does not return after one op. It sits on BRISC
 * reading 32-byte descriptors out of a ring in its own L1, executing them, and
 * recording what it executed -- so the host enqueues work instead of launching it and
 * the per-op dispatch cost goes away (spec 7.2).
 *
 * THE PROTOCOL IS ft_tt_queue.h. Field indices, opcodes and state values are
 * transliterated from it below and the two must agree; tests/test_queue.c pins the
 * host half and tests/test_phase5_resident.c pins the pair on silicon.
 *
 * THREE THINGS ABOUT THIS KERNEL ARE SAFETY, NOT FEATURES:
 *
 * 1. COMPLETION IS PRODUCED BY THE WORK ITEM. completed_token is copied out of the
 *    descriptor and completed_count is incremented only AFTER the descriptor's work
 *    has been done. Neither can be satisfied by a core that merely booted, which is
 *    what RUN_MSG_DONE cannot say (dev notes 3: wait_done_all reported eight cores
 *    complete with one never sent a go).
 *
 * 2. EVERY BLOCKING WAIT POLLS stop_request. A resident kernel is a kernel that
 *    blocks, and killing a blocked kernel with assert_reset is exactly how Tensix 0
 *    was wedged on 2026-08-12 (dev notes 1). So the host retires this kernel by
 *    ASKING -- one word in L1 -- and the ask is visible from inside the semaphore wait
 *    as well as from the idle poll. There is no wait in this file that only a reset
 *    can break.
 *
 * 3. RESIDENCY IS BOUNDED. idle_spin_limit consecutive empty polls and the kernel
 *    returns on its own. That is for a host that DIED, not for flow control; a normal
 *    exit is stop_request or a STOP descriptor.
 *
 * THE WIRE CONTRACT NOW LIVES IN resident_queue_protocol.h, shared with the resident
 * matmul triad. It used to be copied into this file; two copies of a contract is how a
 * field index drifts, and the triad made it three.
 *
 *
 * C99 on Foundry's own `ft_tensix_dataflow.h` — no tt-metal include (plan Phase 6.6).
 * The wait mode, which used to be a `template <uint32_t kMode>` argument, is now an
 * ordinary parameter carrying the same compile-time value; `-Os` folds it identically.
 * Compile-time args:
 *   0: items per launch. 0 = RESIDENT, the Phase 5 model. 1 = ONE-SHOT, which is the
 *      Phase 4 slow-dispatch baseline built from this same source so that an A/B
 *      measures the DISPATCH and not two different implementations of the work.
 * Runtime args: the ten of resident_queue_protocol.h, identical on all five processors.
 *      This kernel uses the first four, plus the status slot and the group post; the CB
 *      bound and the group fields are simply 0 when the host has no use for them.
 *
 * The kernel RESUMES: tail and completed_count are read from the control block at
 * entry rather than started at zero, so one-shot mode is the resident loop stopped
 * after one item and a resident kernel can be retired and relaunched without the host
 * rebuilding its queue.
 *
 * Build (BRISC, NoC 0, relocated into the Phase 4 kernel-config slab):
 *   CTARGS=0 NOC=0 KERNEL_TEXT_BASE=0x15000 bash \
 *       foundry/tt/build_tensix_dataflow_kernel.sh \
 *       foundry/tt/kernels/resident_queue.c brisc /tmp/ttk_resident
 *   CTARGS=1 ... /tmp/ttk_oneshot          # the baseline arm
 *
 * @status RUN ON HARDWARE 2026-08-12 on the p150a, cores 71-74 and 76-79. 1316 B of
 *         text on BRISC. 1.36 us per descriptor of host dispatch, 0.39 us device-side.
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>

#include "ft_tensix_dataflow.h"
#include "resident_queue_protocol.h"

/** Items this launch executes before returning. 0 = RESIDENT, the Phase 5 model; 1 =
 *  ONE-SHOT, the Phase 4 slow-dispatch baseline built from this same source. A compile-time
 *  argument, so `-Os` folds the comparison away in both arms. */
#define ITEMS_PER_LAUNCH ft_tensix_compile_time_arg(0)

void kernel_main(void) {
    ftq_context ctx;

    ftq_context_init(&ctx);

    volatile FTQ_L1_PTR uint32_t *control = ctx.control;
    volatile FTQ_L1_PTR uint32_t *ring = ctx.ring;
    const uint32_t capacity_mask = ctx.capacity_mask;
    const uint32_t idle_spin_limit = ctx.idle_spin_limit;

    control[kFieldState] = kStateBoot;
    ftq_status_set(&ctx, kStateBoot);

    // Refuse a queue that was never set up rather than interpreting whatever L1 held
    // before. A resident kernel walking a garbage ring issues NoC transactions at
    // addresses nobody chose, and a wrong endpoint hangs the chip (spec 7.3).
    //
    // The CB bound is not checked here, unlike in the triad: this kernel is CB-FREE by
    // design and never reaches a wait that bound governs.
    if (control[kFieldMagic] != kMagic) {
        control[kFieldError] = kErrBadMagic;
        control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }
    if (capacity_mask == 0 || ((capacity_mask + 1u) & capacity_mask) != 0) {
        control[kFieldError] = kErrBadCapacity;
        control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }
    if (control[kFieldCapacity] != capacity_mask + 1u || idle_spin_limit == 0u) {
        control[kFieldError] = kErrBadCapacity;
        control[kFieldState] = kStateExit;
        ftq_status_set(&ctx, kStateExit);
        return;
    }

    // RESUME rather than restart. One-shot mode is this loop stopped after one item, so
    // the two arms of the dispatch A/B execute byte-identical per-descriptor code.
    uint32_t tail = control[kFieldTail];
    uint32_t completed = control[kFieldCompletedCount];
    uint32_t executed_this_launch = 0;
    uint32_t idle_spins = 0;
    uint32_t heartbeat = control[kFieldHeartbeat];
    uint32_t error = kErrNone;
    bool running = true;

    while (running) {
        control[kFieldHeartbeat] = ++heartbeat;

        if (control[kFieldStopRequest] != 0) {
            break;
        }

        if (control[kFieldHead] == tail) {
            control[kFieldState] = kStateIdle;
            control[kFieldIdleSpins] = ++idle_spins;
            if (idle_spins >= idle_spin_limit) {
                error = kErrIdleTimeout;
                break;
            }
            continue;
        }

        idle_spins = 0;
        control[kFieldState] = kStateExec;
        bool item_completed = true;

        volatile FTQ_L1_PTR uint32_t *descriptor = ring + (tail & capacity_mask) * kDescriptorWords;
        const uint32_t opcode = descriptor[kDescOpcode];
        const uint32_t token = descriptor[kDescToken];
        const uint32_t a0 = descriptor[kDescA0];
        const uint32_t a1 = descriptor[kDescA1];
        const uint32_t a2 = descriptor[kDescA2];
        const uint32_t a3 = descriptor[kDescA3];
        const uint32_t a4 = descriptor[kDescA4];

        control[kFieldLastOpcode] = opcode;

        switch (opcode) {
            case kOpNop:
                break;

            case kOpWriteWord:
                *(volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(a0) = a1;
                break;

            case kOpL1Copy:
                // Source, destination AND length all arrive on the wire, and this is the one
                // transfer site in the family with no compile-time-constant length behind it.
                // ft_tensix_l1_copy_is_issuable() carries the three refusals; a zero length is
                // the shape that most likely took the card down on 2026-08-12, and a
                // destination below the read-only end overwrites the far core's firmware
                // mailboxes and leaves it unbootable at its NEXT boot.
                if (!ft_tensix_l1_copy_is_issuable(a3, a2, a4)) {
                    error = kErrBadParameters;
                    item_completed = false;
                    running = false;
                    break;
                }
                ft_tensix_noc_write(a3, ft_tensix_noc_address(a0, a1, a2), a4);
                ft_tensix_noc_write_barrier();
                break;

            case kOpSemPost:
                ft_tensix_noc_semaphore_increment(ft_tensix_noc_address(a0, a1, a2), a3);
                break;

            case kOpSemWait: {
                // The blocking opcode, and the only place a group can deadlock. It polls
                // stop_request in the SAME loop as the semaphore, so the host's recovery
                // is a write rather than a reset (rule 2 at the top of this file), and it
                // carries its own spin bound on top of that.
                volatile FTQ_L1_PTR uint32_t *semaphore =
                    (volatile FTQ_L1_PTR uint32_t *)(uintptr_t)(a0);
                const uint32_t bound = (a2 != 0) ? a2 : idle_spin_limit;
                uint32_t spins = 0;
                control[kFieldState] = kStateSemWait;
                while (*semaphore < a1) {
                    control[kFieldHeartbeat] = ++heartbeat;
                    if (control[kFieldStopRequest] != 0) {
                        // Abandoned, not finished. Counting it would retire work that
                        // never happened -- the exact lie rule 1 forbids.
                        item_completed = false;
                        running = false;
                        break;
                    }
                    if (++spins >= bound) {
                        error = kErrSemWaitTimeout;
                        item_completed = false;
                        running = false;
                        break;
                    }
                }
                control[kFieldState] = kStateExec;
                break;
            }

            case kOpDelay: {
                volatile uint32_t counter = 0;
                for (uint32_t iteration = 0; iteration < a0; iteration++) {
                    counter = counter + 1;
                }
                break;
            }

            case kOpStop:
                running = false;
                break;

            default:
                error = kErrBadOpcode;
                item_completed = false;
                running = false;
                break;
        }

        // AN ABANDONED DESCRIPTOR ADVANCES NOTHING. It was completed by nobody and handed
        // to nobody, so neither `tail` nor `completed_count` moves and the next launch
        // re-attempts it. (This changed with the resident triad: there `tail` is the
        // handoff pointer the WRITER's cursor follows, so advancing it for an item the
        // reader abandoned would let the writer retire work that never ran. One meaning
        // for the word in both kernels is worth more than the older "taken" reading, which
        // nothing depended on.)
        if (item_completed) {
            tail++;
            control[kFieldTail] = tail;
            // Token before count: a host that sees the count is entitled to trust the
            // token it reads next. Both are plain stores from one in-order RISC-V.
            control[kFieldCompletedToken] = token;
            control[kFieldCompletedCount] = ++completed;
            ftq_group_post(&ctx, descriptor[kDescA5], token);
            executed_this_launch++;
            if (ITEMS_PER_LAUNCH != 0 && executed_this_launch >= ITEMS_PER_LAUNCH) {
                running = false;
            }
        }
    }

    control[kFieldError] = error;
    control[kFieldState] = kStateExit;
    ctx.error = error;
    ftq_status_set(&ctx, kStateExit);
}
