/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_puller_abi.h
 * @brief The puller program entries a host builder may emit beyond the A2/S2 set — shared by
 *        the kernel (resident_puller.c) and Pantheon's program builder (pn_tt_dispatch*.c), so a
 *        host that emits an entry and a kernel blob that runs it come from one definition.
 *
 * @status L10: FLUSH. S2 (on-card completion): PRECOND.
 * @issues None
 * @todo None
 */
#ifndef RESIDENT_PULLER_ABI_H
#define RESIDENT_PULLER_ABI_H

/** Program-entry type 9, one word: write every queued head now. The builder puts it before
 *  ring slots / parameter blocks it hoisted above an AWAIT (they then go out while the awaited
 *  core works). A puller built before it refuses the entry (error 2), never misreads it. */
#define FT_TT_PULLER_ENTRY_FLUSH 9u

/** Program-entry type 10, FT_TT_PULLER_PRECOND_WORDS words:
 *    10, target, index0, step, producer, count0, count_step
 *  Write the WAIT SIDECAR entry (kernels/resident_precondition_protocol.h) of descriptor
 *  index0 + step * k on queue `target` (bit 31 set: multicast group `target & 0xFFFF`, as a
 *  RING entry): {index, producer x | y << 8 | tag << 16, count0 + count_step * k, producer
 *  nonce}, at control + 0x900 + 16 * (index & the target's mask). A plain NoC write, landed by
 *  the barrier in front of the next head flush like a ring slot. The builder puts it after the
 *  descriptor's ring slot and before its head. A producer whose coordinate does not fit six
 *  bits is refused (error 2), never written. A puller built before it refuses the entry. */
#define FT_TT_PULLER_ENTRY_PRECOND 10u
#define FT_TT_PULLER_PRECOND_WORDS 7u

#endif /* RESIDENT_PULLER_ABI_H */
