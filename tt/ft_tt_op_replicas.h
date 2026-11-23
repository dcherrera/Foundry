/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_replicas.h
 * @brief The HOST half of the resident replica table — the map that tells a writer kernel
 *        which of its output strips are mirrored into other DRAM banks, and where.
 *
 * ---------------------------------------------------------------------------
 * WHAT PROBLEM THIS SOLVES, IN ONE PARAGRAPH
 * ---------------------------------------------------------------------------
 * The Blackhole's 510.8 GB/s is EIGHT banks of ~59 GB/s. An activation strip is allocated
 * in one bank, and every member of a column-split matmul reads the WHOLE strip — so a
 * 16-member projection puts sixteen readers on one bank. Measured on Ornith during decode:
 * bank 2 at 43.1 GB/s (73% of its own ceiling) with the other seven under 5, and the card
 * at 16% of peak. That is why matmul card work is flat from 4 members to 16 and why the
 * shipped setting is four cores on a 140-Tensix part.
 *
 * The fix is to give the strip more than one bank. The DESCRIPTOR side is free — dispatch
 * already writes a separate parameter block per member, so member `m` naming replica
 * `m % replicas` is an index, not a protocol change. The cost lands on the PRODUCER, and
 * this file is how the producer is told.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS A RESIDENT TABLE AND NOT DESCRIPTOR WORDS
 * ---------------------------------------------------------------------------
 * `FT_TT_QUEUE_PARAM_WORDS` is 16 for every family and a seventeenth changes the shared
 * slot stride. The REDUCE family already spends words 13/14/15 on its program, and reduce
 * is exactly the producer that must fan out (`normed` is a reduce output). So there is no
 * free word to put a destination list in, for the one family that needs it most.
 *
 * The table is therefore keyed on the destination the descriptor ALREADY carries. The
 * writer looks up `(kVecCX, kVecCY, kVecCBase)`; a hit names the extra destinations, a
 * miss is the shipped path word for word. An empty table misses everything, which is what
 * makes "off" the default and the null arm the same object.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS FILE REFUSES, AND WHY EACH REFUSAL IS A WRONG ANSWER OTHERWISE
 * ---------------------------------------------------------------------------
 *  - a set with fewer than two destinations — it would cost a table entry and mean nothing
 *  - two destinations that are the same (x, y, base) — the same bytes written twice, which
 *    reads as a working replica map and delivers one bank
 *  - two SETS with the same primary — the writer takes the first hit, so the second set
 *    would be silently dead
 *  - more sets or destinations than the wire format holds
 *
 * None of these can be seen on the card: a bad map produces a strip that is stale in one
 * bank, and a matmul member reading it returns a plausible wrong answer.
 *
 * @status Written 2026-08-19 for the A-bank spread. Pinned off-hardware by
 *         tests/test_replica_table.c, which exercises the KERNEL's own lookup.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_OP_REPLICAS_H
#define FT_TT_OP_REPLICAS_H

#include <stdint.h>

#include "ft_tt_launch.h"
#include "kernels/resident_output_replicas.h"

/** One place a tile is written: a DRAM bank's NoC coordinate and a byte offset in it. */
typedef struct ft_tt_op_replica_destination {
    uint16_t x, y;
    uint32_t base;
} ft_tt_op_replica_destination;

/** One strip and its mirrors. `destination[0]` is the PRIMARY and is the lookup key — the
 *  address every other consumer of the strip already uses, which is what keeps replication
 *  invisible to everything but the matmul's A side. */
typedef struct ft_tt_op_replica_set {
    uint32_t count;
    /** 0: the exact key. Else a RANGE record covering [primary, primary + range_bytes),
     *  a multiple of FT_TT_REPLICA_RANGE_UNIT (kernels/resident_output_replicas.h). */
    uint32_t range_bytes;
    ft_tt_op_replica_destination destination[FT_TT_REPLICA_MAX_DESTINATIONS];
} ft_tt_op_replica_set;

/** Everything one residency core needs to know. Zeroed is a valid, empty table. */
typedef struct ft_tt_op_replica_table {
    uint32_t set_count;
    ft_tt_op_replica_set set[FT_TT_REPLICA_MAX_SETS];
} ft_tt_op_replica_table;

/**
 * Encode a table into the words the kernel reads.
 *
 * Pure: no device access, no allocation — which is what lets the wire format be tested
 * where a wrong word costs a second instead of a card session.
 *
 * A table with `set_count == 0` still encodes, with the magic present and the count zero.
 * That is deliberately not the same as an unwritten region: it says "this core has been
 * told, and the answer is nothing", which is the state a run with replication disabled
 * should be able to assert.
 *
 * @param table      the map; every field read
 * @param out_words  FT_TT_REPLICA_TABLE_WORDS values, fully written
 * @return FT_TT_OK, or FT_TT_EINVAL with the reason on stderr
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (it may print)
 */
int ft_tt_op_replica_table_encode(const ft_tt_op_replica_table *table,
                                  uint32_t out_words[FT_TT_REPLICA_TABLE_WORDS]);

/**
 * Encode and write the table into one core's L1, at FT_TT_REPLICA_TABLE_ADDRESS.
 *
 * ONE NoC write of the whole block. The region is inside the kernel-config region every
 * program already owns, so this needs no arena, no circular buffer and no change to the
 * L1 map — see kernels/resident_output_replicas.h for where the gap is.
 *
 * MUST be written before the first descriptor that could hit it, and it is safe to write
 * again between descriptors: the writer reads it once per descriptor and never caches it.
 *
 * @return FT_TT_OK, or the NoC error, or FT_TT_EINVAL
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_op_replica_table_write(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                 const ft_tt_op_replica_table *table);

/**
 * Read one core's table back and compare it, word for word, against what `table` encodes.
 *
 * THIS IS THE ONLY THING THAT CAN CATCH A HALF-WRITTEN MAP. A replica whose bytes were
 * never written is a strip that is stale in one bank, and the consumer that reads it
 * returns a plausible wrong answer — there is no refusal anywhere in the path. So the map
 * is verified by reading it back rather than by trusting the write's return code.
 *
 * @return FT_TT_OK when they match, FT_TT_ERROR with the first differing word on stderr,
 *         or the NoC error
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_op_replica_table_verify(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  const ft_tt_op_replica_table *table);

#endif /* FT_TT_OP_REPLICAS_H */
