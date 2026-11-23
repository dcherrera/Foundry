/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file cooperative_matmul_protocol.h
 * @brief The host<->device wire contract for the COOPERATIVE matmul reader: the three roles,
 *   the runtime-arg layout, the status block, and the refusal codes.
 *
 *   **Included by BOTH sides on purpose.** Every other wire contract in this directory that
 *   was duplicated (multicast_bench_reader.c's arg map, copied by hand into
 *   tests/test_multicast_bench.c with "change both or neither" in a comment) is one edit away
 *   from a kernel reading a different word than the host wrote — and a runtime-arg block that
 *   did not say what its reader thought has already stalled a core here
 *   (docs/tt_launch_notes.md §3). This header therefore contains NOTHING but preprocessor
 *   definitions: no includes, no types, no code. That is what lets a C99 host program and a
 *   Tensix kernel share it verbatim.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE COOPERATION IS
 * ---------------------------------------------------------------------------
 * N cores compute ONE matmul C[Mt x Nt] = A[Mt x Kt] * B[Kt x Nt] by splitting the OUTPUT
 * COLUMNS: core c owns `Nt / N` consecutive columns, reads its own B panel from DRAM, and
 * writes its own output panel. Every core needs the SAME A block at the same time, and that
 * is the one thing multicast is for.
 *
 * The split is over N and not over K or M for one reason each:
 *   - over K every core would produce a PARTIAL sum of the same output tile, needing a
 *     reduction the compute kernel cannot express (it accumulates in one DST register across
 *     the whole K loop) and a cross-core add that does not exist;
 *   - over M every core would need a different A block, which is precisely the operand the
 *     multicast makes free — the split would share nothing;
 *   - over N every core needs the SAME A and a DISJOINT B, so exactly one operand is shared
 *     and it is the one whose re-reading the plan has named as the blocker since Phase 4.
 *
 * The B side wants nothing: each core's B panel is disjoint, so there is nothing to broadcast
 * and every byte a core reads it is the only consumer of. Total DRAM traffic for B is
 * unchanged by the split — it is the same bytes, read by N cores instead of one.
 *
 * ---------------------------------------------------------------------------
 * THE PER-BLOCK PROTOCOL
 * ---------------------------------------------------------------------------
 * The reader's loop nest is matmul_reader.c's, UNCHANGED: for each output tile (mt, nt), for
 * each block of `depth` K-tiles. The block index therefore runs
 * `((mt * Nt_local + nt) * Kt/depth + kb)` on every core, and because A does not depend on
 * `nt`, **every core needs the same A block at the same block index**. That is what makes a
 * lockstep rendezvous correct rather than merely convenient, and it is why the loop structure
 * had to stay untouched: hoisting A out of the `nt` loop would need the compute kernel to stop
 * popping it, and the compute kernel is not ours to change.
 *
 *   LEADER, per block:  reserve both CBs -> wait ready == destination_count -> ready = 0 ->
 *                       read A from DRAM into its own CB_IN0 block + its own B -> read barrier
 *                       -> multicast the A block -> writes-flushed barrier (BLACKHOLE, and
 *                       mandatory) -> multicast VALID into every receiver's arrival semaphore
 *                       -> push both CBs
 *   RECEIVER, per block: reserve both CBs -> arrival = INVALID -> issue its own B read ->
 *                       increment the leader's ready semaphore -> wait arrival == VALID ->
 *                       read barrier -> push both CBs
 *   SOLO, per block:     matmul_reader.c exactly. No semaphores, no rectangle, no multicast.
 *
 * The receiver clears its arrival word BEFORE announcing readiness, never after: the leader
 * cannot set the word until it has seen the announcement, so a VALID can never be erased by
 * the clear that was meant to precede it.
 *
 * **The multicast destination is the CB's own write pointer, and that is checked, not
 * assumed.** A multicast writes ONE address on every destination, so the A block must land at
 * the same L1 address on all N cores. It does, because every core is given the same CB
 * placement and pushes the same number of pages in the same order — but "does" is a claim
 * about host code and CB arithmetic that no hardware would refuse if it stopped being true.
 * So the host passes the CB_IN0 base and every core, every block, checks
 * `write_pointer == base + (block & 1) * depth * tile_bytes` and REFUSES on a mismatch. The
 * `& 1` is the whole of the CB wrap rule for a 2*depth-page buffer pushed `depth` at a time
 * (ft_tensix_circular_buffer.h): the pointer is at the base or one block up, never anywhere
 * else, and a value that is anywhere else means the invariant has already broken.
 *
 * ---------------------------------------------------------------------------
 * A FAILURE DRAINS, IT DOES NOT RETURN
 * ---------------------------------------------------------------------------
 * Every wait here is bounded and askable, so any of them can give up. A reader that gave up
 * and RETURNED would leave the compute thread in `cb_wait_front` and the writer in its own —
 * and a kernel blocked in a CB wait is the shape that wedged Tensix 0 and cost core 62
 * (DEV_NOTES.md §1). So an abort sets an error, stops ISSUING anything, and runs the loop to
 * the end still pushing pages, exactly as resident_matmul_reader.c's drain rule requires. The
 * answer is garbage and the status block says so; the core stays bootable.
 *
 * @status MEASURED on the card 2026-08-13. All four arms run byte-identical to the
 *   single-core result; both refusals below were driven red on silicon.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_COOPERATIVE_MATMUL_PROTOCOL_H
#define FT_TT_COOPERATIVE_MATMUL_PROTOCOL_H

/* ------------------------------------------------------------------------------------
 * Roles. One binary plays all three; the role is runtime arg 0.
 *
 * One binary rather than three is the MEASUREMENT's method, not a convenience: the whole
 * question is what cooperation costs against one core doing the same work, and two binaries
 * would put every codegen difference — inlining, text placement, register allocation — inside
 * the number. It is the same rule `weight_n_major` follows in matmul_reader.c and the role
 * word follows in multicast_bench_reader.c (docs/tt_kernel_model.md §14).
 * ---------------------------------------------------------------------------------- */

/** No cooperation at all: matmul_reader.c's loop, byte for byte. The BASELINE arm. */
#define FTCM_ROLE_SOLO 0u
/** Reads the shared A block from DRAM and multicasts it. Also computes a slice of its own. */
#define FTCM_ROLE_LEADER 1u
/** Receives the shared A block; reads only its own B panel from DRAM. */
#define FTCM_ROLE_RECEIVER 2u

/* ------------------------------------------------------------------------------------
 * Runtime arguments — ONE layout for all three roles.
 *
 * One layout rather than three, because three arg maps is three things to keep in step with
 * one host. A role that does not use a field ignores it; nothing is overloaded.
 * ---------------------------------------------------------------------------------- */

#define FTCM_ARG_ROLE 0u
#define FTCM_ARG_TILE_ROWS 1u          /**< Mt */
#define FTCM_ARG_TILE_INNER 2u         /**< Kt */
#define FTCM_ARG_TILE_COLS 3u          /**< Nt_local: the columns THIS core owns */
#define FTCM_ARG_A_X 4u
#define FTCM_ARG_A_Y 5u
#define FTCM_ARG_A_BASE 6u             /**< DRAM base of A tile 0. The SAME on every core */
#define FTCM_ARG_B_X 7u
#define FTCM_ARG_B_Y 8u
#define FTCM_ARG_B_BASE 9u             /**< DRAM base of the WHOLE B grid, not this core's panel */
#define FTCM_ARG_B_COLUMN_FIRST 10u    /**< this core's first column in the FULL grid */
#define FTCM_ARG_B_FULL_COLS 11u       /**< Nt of the full grid; the K-major row stride */
#define FTCM_ARG_DEPTH 12u             /**< prefetch depth in K-tiles */
#define FTCM_ARG_B_N_MAJOR 13u         /**< 0 = tile index kt*Nt + nt, 1 = nt*Kt + kt */
#define FTCM_ARG_CB_IN0_BASE 14u       /**< where CB_IN0's FIFO starts; the invariant checked */
#define FTCM_ARG_RECTANGLE_X_LOW 15u
#define FTCM_ARG_RECTANGLE_Y_LOW 16u
#define FTCM_ARG_RECTANGLE_X_HIGH 17u
#define FTCM_ARG_RECTANGLE_Y_HIGH 18u
#define FTCM_ARG_DESTINATION_COUNT 19u /**< receivers, EXCLUDING the leader */
#define FTCM_ARG_LEADER_X 20u
#define FTCM_ARG_LEADER_Y 21u
#define FTCM_ARG_READY_SEMAPHORE 22u   /**< on the LEADER; receivers increment it */
#define FTCM_ARG_ARRIVAL_SEMAPHORE 23u /**< on EVERY member, the same address */
#define FTCM_ARG_VALID_WORD 24u        /**< on the LEADER; the word it multicasts FROM */
#define FTCM_ARG_STATUS 25u
#define FTCM_ARG_STOP_FLAG 26u
#define FTCM_ARG_SPIN_LIMIT 27u
#define FTCM_ARG_COUNT 28u

/* ------------------------------------------------------------------------------------
 * The status block. Written ONCE, after the loop, with the magic LAST.
 *
 * Magic last so a host reading a torn block sees "never ran" rather than a half-filled
 * result — the same ordering rule the resident queue's token-before-count uses. The host
 * POISONS the block before launch, so "never ran" and "finished" are different bit patterns
 * rather than the same zero.
 * ---------------------------------------------------------------------------------- */

/** "FTCM". */
#define FTCM_STATUS_MAGIC 0x4654434Du

#define FTCM_STATUS_MAGIC_WORD 0u
#define FTCM_STATUS_ROLE 1u
#define FTCM_STATUS_BLOCKS_COMPLETED 2u
#define FTCM_STATUS_BLOCKS_EXPECTED 3u
#define FTCM_STATUS_ERROR 4u
#define FTCM_STATUS_LAST_ISSUE_CODE 5u
#define FTCM_STATUS_CLOCK_AT_ENTRY 6u
#define FTCM_STATUS_CLOCK_AFTER_FIRST_BLOCK 7u
#define FTCM_STATUS_CLOCK_AT_EXIT 8u
/** The write pointer that failed the invariant, so a refusal names the value it saw. */
#define FTCM_STATUS_OBSERVED_WRITE_POINTER 9u
#define FTCM_STATUS_WORDS 10u

/** What the host writes into every status word before a launch. Different from every value
 *  the kernel writes and from cleared L1. */
#define FTCM_STATUS_POISON 0x7C2C2C2Cu

/* ------------------------------------------------------------------------------------
 * Refusal codes. 0 and -1..-6 are kernels/ft_tensix_noc_multicast.h's, reproduced by value
 * so this header stays include-free; the two must agree and
 * tests/test_cooperative_matmul.c asserts they do.
 * ---------------------------------------------------------------------------------- */

#define FTCM_OK 0
#define FTCM_ERROR_ZERO_LENGTH (-1)
#define FTCM_ERROR_ZERO_DESTINATIONS (-2)
#define FTCM_ERROR_ALIGNMENT (-3)
#define FTCM_ERROR_RECTANGLE (-4)
#define FTCM_ERROR_WAIT_TIMEOUT (-5)
#define FTCM_ERROR_STOPPED (-6)
/** The CB_IN0 write pointer was not where the block index says it must be, so the address
 *  the leader would have multicast into is not the address the receivers reserved. */
#define FTCM_ERROR_WRITE_POINTER (-10)
/** A shape, a depth or a tile size the kernel cannot execute. Refused before any issue. */
#define FTCM_ERROR_BAD_PARAMETERS (-11)
/** An unknown role word. A refusal, never a no-op. */
#define FTCM_ERROR_BAD_ROLE (-12)

#endif /* FT_TT_COOPERATIVE_MATMUL_PROTOCOL_H */
