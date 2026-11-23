/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file resident_matmul_multicast_protocol.h
 * @brief The host<->device wire contract for a RESIDENT matmul triad whose shared A block
 *   is MULTICAST once per block instead of read from DRAM by every member: the roles, the
 *   runtime-arg tail, the packed rectangle word, and the refusal codes.
 *
 *   **Included by BOTH sides verbatim**, like cooperative_matmul_protocol.h and for the
 *   same reason: every wire contract in this directory that was duplicated is one edit
 *   away from a kernel reading a different word than the host wrote. So this header
 *   contains NOTHING but preprocessor definitions, enums and pure inline arithmetic — no
 *   includes beyond <stdint.h>, no types that need one.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS IS FOR, IN ONE NUMBER
 * ---------------------------------------------------------------------------
 * At Mt = 1 the matmul reader re-reads the whole A column once per output column, so A is
 * **1.76 GB of the 2.70 GB a token reads — 65%** — and every member of a split group reads
 * it from the ONE DRAM bank holding that strip. That is why four cores buy 3.44x and four
 * times more cores buy only 1.4x more, and why N=16 is byte-exact and SLOWER end to end.
 * A multicast turns the A side from O(N) reads of that bank into O(1).
 *
 * The mechanism already existed as `cooperative_matmul_reader.c` and was unreachable: it
 * is a ONE-SHOT program, and running it per op would put the image upload back at 218 µs
 * per member per op — **147 ms a token at N=4** against a 0.109 s token. This header is
 * that rendezvous expressed as a tail on the RESIDENT queue's contract, so it costs a
 * descriptor rather than a program load.
 *
 * ---------------------------------------------------------------------------
 * ceil(RECEIVERS / 4) PACKETS, NOT ONE
 * ---------------------------------------------------------------------------
 * The standing hazard is **never more than four destinations in one multicast packet**.
 * Measured twice on silicon: a 5-destination broadcast leaves every receiver AND the
 * sender able to accept re-launches indefinitely, and then wedges a core at the NEXT
 * firmware boot (`0x00047000`, BRISC stalled inside `device_setup()`); N=8 issued as two
 * disjoint <=4-destination broadcasts through that same park-and-reboot shape came back
 * 20/20 PASS. The cause is believed to be our per-run teardown rather than the packet, and
 * the rule stands until a residency that is NEVER re-booted proves otherwise.
 *
 * So the leader carries a PARTITION of its receivers — up to
 * FT_TT_RESIDENT_COOP_MAX_GROUPS rectangles, each naming at most four destinations, built
 * host-side by ft_tt_group_broadcast_partition.h. At N=4 that is one packet; at N=8, two.
 *
 * The per-block order is one flush for the whole partition, not one per piece:
 *
 *   LEADER   wait ready == receivers -> ready = 0 -> stage the block counter ->
 *            read A and its own B from DRAM -> read barrier ->
 *            payload multicast to piece 0..G-1 -> WRITES-FLUSHED BARRIER ->
 *            counter multicast into every piece's arrival semaphore -> push both CBs
 *   RECEIVER arrival = INVALID -> issue its own B read -> increment the leader's ready ->
 *            wait arrival == this block's counter -> read barrier -> push both CBs
 *
 * The flush between the payload and the semaphore is **mandatory on Blackhole and only
 * Blackhole**: the two use different command-buffer FIFOs which are not guaranteed to
 * drain in issue order, so without it a receiver can be told the block arrived before it
 * did and read the PREVIOUS block's bytes with nothing reporting an error. ONE flush
 * covers all G payloads because it waits for requests SENT, and the pieces share one NoC
 * and one static VC — once every payload request has left, ordering behind it is the
 * network's job.
 *
 * ---------------------------------------------------------------------------
 * THE ARRIVAL WORD CARRIES THE BLOCK COUNTER, NOT A CONSTANT `VALID`
 * ---------------------------------------------------------------------------
 * This is the one deliberate divergence from upstream's protocol and from
 * cooperative_matmul_protocol.h, and it exists because a RESIDENT triad can drift in a way
 * a one-shot program cannot.
 *
 * A multicast writes ONE address into every destination, so the A block must land at the
 * same L1 address on every member — which it does only while every member has pushed the
 * same number of CB_IN0 pages since launch. In a one-shot program that is guaranteed by
 * construction. In a residency it is a property of the WHOLE DESCRIPTOR STREAM: one
 * descriptor executed by some members and not others, or split unevenly so members walk
 * different block counts, desynchronises the CB write pointers permanently and every later
 * multicast lands in the wrong half of the receivers' buffers.
 *
 * Two things stop that, and they are prevention and detection rather than one hopeful
 * check:
 *
 *  - **Prevention, host-side and structural:** cooperation is ALL-OR-NOTHING for a
 *    residency. A non-SOLO member that is handed a descriptor with the cooperate word
 *    clear refuses it (kErrCoopRoleMismatch) instead of quietly running it solo, so a
 *    group cannot drift one descriptor at a time. The host enables cooperation only when
 *    EVERY weight splits evenly across the group.
 *  - **Detection, device-side:** the leader multicasts its own cooperative block counter
 *    and each receiver waits for the value IT expects. A desynchronised group then times
 *    out on a bounded, askable wait — a refusal the host can see and re-arm from — instead
 *    of computing a wrong answer that every status word calls clean. The counter starts at
 *    1 and skips 0 on wrap, so it can never collide with FT_TENSIX_SEMAPHORE_INVALID.
 *
 * Each member additionally checks its own CB_IN0 write pointer against the position its
 * own counter says it must be at, which catches the local half of the same failure before
 * anything is issued.
 *
 * @status Wire contract for resident_matmul_reader.c's cooperative path.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_RESIDENT_MATMUL_MULTICAST_PROTOCOL_H
#define FT_TT_RESIDENT_MATMUL_MULTICAST_PROTOCOL_H

#include <stdint.h>

/* ------------------------------------------------------------------------------------
 * Roles. One binary plays all three; the role is a runtime arg, fixed for the residency.
 *
 * The values are cooperative_matmul_protocol.h's FTCM_ROLE_*, deliberately, so a reader
 * moving between the one-shot kernel and this one is never reading two spellings of the
 * same three states. tests/test_group_broadcast_partition.c does not assert this — the
 * two headers are independent contracts — but nothing gains from them disagreeing.
 * ---------------------------------------------------------------------------------- */

/** No cooperation: the descriptor's A block comes from DRAM, exactly as it always has.
 *  This is what a member_count == 1 residency runs and what every member ran before. */
#define FT_TT_RESIDENT_COOP_ROLE_SOLO 0u
/** Reads the shared A block from DRAM and multicasts it to every piece of the partition.
 *  Computes a slice of its own like any other member. */
#define FT_TT_RESIDENT_COOP_ROLE_LEADER 1u
/** Receives the shared A block; reads only its own B panel from DRAM. */
#define FT_TT_RESIDENT_COOP_ROLE_RECEIVER 2u

/* ------------------------------------------------------------------------------------
 * What the descriptor's cooperate word says.
 *
 * Three values rather than a flag, because the RED PROOF has to live on the shipping path.
 * A red build would prove its claim about a different program; a value in a word the real
 * dispatch already writes proves it about this one — the same argument PN_TT_RED_SPLIT and
 * PN_TT_RED_BAD_DESCRIPTOR are built on.
 * ---------------------------------------------------------------------------------- */

/** The A block comes from DRAM. What every descriptor written before this word existed
 *  says, and what a member_count == 1 residency runs. */
#define FT_TT_RESIDENT_COOP_DESCRIPTOR_SOLO 0u
/** The leader reads A once and multicasts it. */
#define FT_TT_RESIDENT_COOP_DESCRIPTOR_COOPERATE 1u
/**
 * RED PROOF ONLY: cooperate, but the leader SKIPS the payload multicast and still tells
 * the receivers the block arrived. Every receiver then computes from the PREVIOUS block's
 * bytes.
 *
 * It proves two things at once, which is why it is the one red arm worth having here:
 *  - the byte comparison can still go red on the cooperative path;
 *  - the multicast is the ONLY source of A on a receiver. If a receiver were somehow still
 *    reading A from DRAM — a leftover read, a mis-set role, a branch that did not fold —
 *    this arm would produce IDENTICAL output, and identical output is exactly what a
 *    vacuous gate looks like.
 */
#define FT_TT_RESIDENT_COOP_DESCRIPTOR_RED_SKIP_PAYLOAD 2u

/** Most pieces one leader's partition may hold. Four rectangles of four destinations
 *  reach sixteen members, which is FT_TT_OP_COOP_MAX_MEMBERS. */
#define FT_TT_RESIDENT_COOP_MAX_GROUPS 4u

/** The standing operational limit on destinations in ONE packet, restated device-side so
 *  the kernel refuses a host that ignored it rather than trusting the plan. */
#define FT_TT_RESIDENT_COOP_MAX_DESTINATIONS 4u

/* ------------------------------------------------------------------------------------
 * The runtime-arg TAIL, read ONLY by the matmul reader.
 *
 * Args 0..9 are resident_queue_protocol.h's, identical on all five processors, and
 * `ftq_context_init()` reads exactly those ten in a statement order that is load-bearing
 * for the Phase 6.6 acceptance diff. Nothing here is read there: the cooperative tail has
 * its own reader in the matmul kernel, so the other four resident families' images are
 * untouched by this file existing.
 *
 * The whole tail is fixed for the life of a residency — a group's roles, geometry and
 * semaphore addresses do not change between descriptors — so it belongs in the arg block
 * and not in the 16-word parameter block, which is per descriptor and already full.
 * ---------------------------------------------------------------------------------- */

enum { kArgCoopRole = 10 };
enum { kArgCoopLeaderX = 11 };
enum { kArgCoopLeaderY = 12 };
/** On the LEADER. Receivers increment it; the leader waits for it to reach the receiver
 *  count and then clears it. The same L1 address on every member, which is what lets one
 *  arg block serve all three roles. */
enum { kArgCoopReadySemaphore = 13 };
/** On EVERY member, the same address: the word the leader multicasts the block counter
 *  into. */
enum { kArgCoopArrivalSemaphore = 14 };
/** On the LEADER. The word it multicasts FROM, restaged with the block counter each
 *  block. */
enum { kArgCoopCounterWord = 15 };
/** Receivers the leader waits for, EXCLUDING itself. Equals the partition's destination
 *  total, and the host is required to have checked that they agree. */
enum { kArgCoopReceiverCount = 16 };
/** Pieces in the partition, 1..FT_TT_RESIDENT_COOP_MAX_GROUPS. */
enum { kArgCoopGroupCount = 17 };
/** The pieces, packed one word each — see ftcoop_pack_rectangle(). */
enum { kArgCoopGroup0 = 18 };
enum { kArgCoopGroup1 = 19 };
enum { kArgCoopGroup2 = 20 };
enum { kArgCoopGroup3 = 21 };
/** Spins one rendezvous wait takes before abandoning. Never 0: an unbounded wait is the
 *  one thing a resident kernel may not be configured into. */
enum { kArgCoopSpinLimit = 22 };
enum { kArgCoopCount = 23 };

/* ------------------------------------------------------------------------------------
 * The packed rectangle word
 *
 * Four 6-bit coordinates and a destination count in one 32-bit arg, because the arg block
 * is 32 words and five words per piece would not fit four pieces beside the ten the queue
 * already takes. Six bits is not a packing choice: it is the width of the NoC node-id
 * field, and a coordinate that does not fit does not fail — it TRUNCATES onto a different
 * live endpoint, which on this board is a hung NoC and a walk to the machine.
 * ---------------------------------------------------------------------------------- */

#define FT_TT_RESIDENT_COOP_COORDINATE_BITS 6u
#define FT_TT_RESIDENT_COOP_COORDINATE_MASK 0x3Fu
#define FT_TT_RESIDENT_COOP_COUNT_SHIFT 24u
#define FT_TT_RESIDENT_COOP_COUNT_MASK 0xFFu

/** Pack one piece. Returns 0 — never a legal piece, because a legal piece names at least
 *  one destination — when any field is out of range, so a caller that does not check gets
 *  a word the kernel refuses rather than a rectangle at the wrong place. */
static inline uint32_t ftcoop_pack_rectangle(uint32_t x_low, uint32_t y_low, uint32_t x_high,
                                             uint32_t y_high, uint32_t destination_count)
{
    if (x_low > x_high || y_low > y_high)
        return 0u;
    if (x_high > FT_TT_RESIDENT_COOP_COORDINATE_MASK ||
        y_high > FT_TT_RESIDENT_COOP_COORDINATE_MASK)
        return 0u;
    if (destination_count == 0u || destination_count > FT_TT_RESIDENT_COOP_MAX_DESTINATIONS)
        return 0u;
    return x_low | (y_low << FT_TT_RESIDENT_COOP_COORDINATE_BITS) |
           (x_high << (2u * FT_TT_RESIDENT_COOP_COORDINATE_BITS)) |
           (y_high << (3u * FT_TT_RESIDENT_COOP_COORDINATE_BITS)) |
           (destination_count << FT_TT_RESIDENT_COOP_COUNT_SHIFT);
}

static inline uint32_t ftcoop_rectangle_x_low(uint32_t packed)
{
    return packed & FT_TT_RESIDENT_COOP_COORDINATE_MASK;
}

static inline uint32_t ftcoop_rectangle_y_low(uint32_t packed)
{
    return (packed >> FT_TT_RESIDENT_COOP_COORDINATE_BITS) &
           FT_TT_RESIDENT_COOP_COORDINATE_MASK;
}

static inline uint32_t ftcoop_rectangle_x_high(uint32_t packed)
{
    return (packed >> (2u * FT_TT_RESIDENT_COOP_COORDINATE_BITS)) &
           FT_TT_RESIDENT_COOP_COORDINATE_MASK;
}

static inline uint32_t ftcoop_rectangle_y_high(uint32_t packed)
{
    return (packed >> (3u * FT_TT_RESIDENT_COOP_COORDINATE_BITS)) &
           FT_TT_RESIDENT_COOP_COORDINATE_MASK;
}

static inline uint32_t ftcoop_rectangle_destinations(uint32_t packed)
{
    return (packed >> FT_TT_RESIDENT_COOP_COUNT_SHIFT) & FT_TT_RESIDENT_COOP_COUNT_MASK;
}

/** The FIRST cooperative block counter. Never FT_TENSIX_SEMAPHORE_INVALID (0), because the
 *  receiver clears its arrival word to that value before announcing readiness and a
 *  counter that could equal it would let a stale clear read as an arrival. */
#define FT_TT_RESIDENT_COOP_FIRST_COUNTER 1u

/** The next counter, skipping 0 on wrap for the reason above. */
static inline uint32_t ftcoop_next_counter(uint32_t counter)
{
    const uint32_t next = counter + 1u;

    return next == 0u ? FT_TT_RESIDENT_COOP_FIRST_COUNTER : next;
}

#endif /* FT_TT_RESIDENT_MATMUL_MULTICAST_PROTOCOL_H */
