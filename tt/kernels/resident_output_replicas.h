// SPDX-License-Identifier: AGPL-3.0-or-later
//
// resident_output_replicas.h
//   The RESIDENT REPLICA TABLE: a small, host-written block of L1 that tells a writer
//   kernel "this output strip is mirrored into these other DRAM banks as well".
//
//   Included by BOTH sides — kernels/resident_vector_writer.c reads it out of L1 and
//   tt/ft_tt_op_replicas.c fills it in — so there is one layout and not two that must
//   agree. Same discipline as kernels/resident_gdn_protocol.h.
//
// ===========================================================================
// WHY A RESIDENT TABLE RATHER THAN DESCRIPTOR WORDS
// ===========================================================================
// A parameter block is SIXTEEN words for every family (FT_TT_QUEUE_PARAM_WORDS) and a
// seventeenth changes the shared slot stride and breaks on the card. The REDUCE family
// already spends words 13, 14 and 15 on its program (FT_TT_OP_REDUCE_PASS0/PASS1/FINAL),
// so a reduce descriptor has NO free word at all — and the reduce writer is exactly the
// producer that has to fan out, because `normed` is a reduce output.
//
// So the fan-out is not addressed per descriptor. It is keyed on the DESTINATION the
// descriptor already carries: the writer looks its (x, y, base) triple up in this table,
// and on a hit writes the same tile to the extra destinations the table names. A miss is
// the shipped behaviour, word for word, and an EMPTY table is a miss for every descriptor.
// That is what makes "replication off" the default and the null arm at the same time.
//
// The keying is EXACT on all three words. A partial key — bank alone, or base alone —
// would fan a descriptor out that happens to share a bank with a replicated strip, and
// that is a wrong answer with nothing to see it.
//
// ===========================================================================
// WHY THE PRODUCER PAYS AND NOT THE CONSUMER
// ===========================================================================
// The Blackhole has 8 DRAM banks of ~59 GB/s each. An activation strip lives in ONE of
// them, and every member of a split matmul reads the WHOLE of it — so a 16-member matmul
// puts 16 readers on one bank and gets 59 GB/s out of a 511 GB/s card. Measured on Ornith
// during decode: b2 43.1 GB/s while six other banks sat under 5.
//
// Replicating the strip lets member `m` read replica `m % replicas`, which is R banks
// instead of one. The bytes it costs are WRITES, which are the cheap direction on this
// card (the recorded 58x write/read asymmetry), and they are card-to-card: the host never
// sees them.
//
// ===========================================================================
// WHERE IT LIVES
// ===========================================================================
// In the kernel-config region's own free gap: the runtime-arg blocks end at
// FT_TT_OP_RTA_BASE + 5 * FT_TT_OP_RTA_STRIDE = base + 0x680 and BRISC's text slab starts
// at base + 0x1000, so 0x980 bytes are unused there. This table takes 0x200 of them,
// leaving the rest. It is INSIDE the region every program already owns, so it needs no
// arena, no CB and no change to the L1 map's boundaries.
//
// It is written by the host once, after the strips are placed, and never again — a
// replica set is a property of the model's allocation, not of a descriptor.
//
#ifndef FOUNDRY_RESIDENT_OUTPUT_REPLICAS_H
#define FOUNDRY_RESIDENT_OUTPUT_REPLICAS_H

#include <stdint.h>

#ifndef FTR_INLINE
#ifdef __cplusplus
#define FTR_INLINE inline
#else
#define FTR_INLINE static inline
#endif
#endif

/** Absolute L1 address of the table. In the gap between the runtime-arg blocks
 *  (FT_TT_OP_KERNEL_CONFIG_BASE + 0x680) and BRISC's text slab (+ 0x1000).
 *  ft_tt_op_replicas.c static-asserts that against ft_tt_op_l1_layout.h. */
#define FT_TT_REPLICA_TABLE_ADDRESS 0x00014800u

/** Bytes reserved. Two header words plus FT_TT_REPLICA_MAX_SETS records, rounded up. */
#define FT_TT_REPLICA_TABLE_BYTES 0x400u /* 8 sets = 808 B; ends at 0x14C00, below DM0 text at 0x15000 */

/** Not 0 and not 0xFFFFFFFF, so neither a zeroed nor an erased L1 region reads as a table.
 *  A writer that finds anything else treats the table as absent, which is the shipped
 *  behaviour — the failure mode of a mis-written table is "no fan-out", never a stray
 *  NoC write to whatever the words happened to hold. */
#define FT_TT_REPLICA_TABLE_MAGIC 0x52504C43u /* "RPLC" */

/** Destinations one strip may have, PRIMARY INCLUDED. Eight is the DRAM bank count, which
 *  is the most a replica map can usefully reach. */
#define FT_TT_REPLICA_MAX_DESTINATIONS 8u

/** Replicated strips one residency may know about. EIGHT since 2026-10 (was 4): a batched
 *  prefill at two bands keys a set per (band, slice) — `normed` x 2 bands + `ffn_swiglu` x 2
 *  eltwise slices x 2 bands = 6 — and at 4 the host collapsed the eltwise split to fit
 *  (-10%). The search is linear over at most 8 records. */
#define FT_TT_REPLICA_MAX_SETS 8u

/** word 0 magic, word 1 set count, then FT_TT_REPLICA_MAX_SETS fixed-stride records. */
#define FT_TT_REPLICA_HEADER_WORDS 2u

/** Inside one record: the destination count, then that many (x, y, base) triples, the
 *  first of which is the PRIMARY and is the key. Fixed stride so a record's address is
 *  arithmetic and the kernel never walks a variable-length list.
 *
 *  RANGE RECORDS (2026-10-27): the count word's bits 8..31 are the record's RANGE in
 *  FT_TT_REPLICA_RANGE_UNIT units; 0 is the exact key above, word for word. A range record
 *  matches any descriptor whose (x, y) is the primary's and whose base lies in
 *  [primary, primary + range), and every destination gets the SAME offset from its own
 *  base. So one record covers a strip that is written band by band, slice by slice, at
 *  any span (the prefill out-projection inputs), where an exact key needs a set each. */
#define FT_TT_REPLICA_COUNT_MASK  0xFFu
#define FT_TT_REPLICA_RANGE_SHIFT 8u
#define FT_TT_REPLICA_RANGE_UNIT  2048u
#define FT_TT_REPLICA_RECORD_WORDS (1u + 3u * FT_TT_REPLICA_MAX_DESTINATIONS)

/** Words the whole table occupies. */
#define FT_TT_REPLICA_TABLE_WORDS \
    (FT_TT_REPLICA_HEADER_WORDS + FT_TT_REPLICA_MAX_SETS * FT_TT_REPLICA_RECORD_WORDS)

/** Word offset of record `set` within the table. */
FTR_INLINE uint32_t ftr_record_offset(uint32_t set)
{
    return FT_TT_REPLICA_HEADER_WORDS + set * FT_TT_REPLICA_RECORD_WORDS;
}

/**
 * Resolve one descriptor's output into the full destination list.
 *
 * `table` is the L1 words at FT_TT_REPLICA_TABLE_ADDRESS on the kernel side and an
 * ordinary array on the host side; the lookup is the same either way, which is what lets
 * tests/test_replica_table.c exercise the kernel's own logic off-hardware.
 *
 * ALWAYS writes at least the primary and ALWAYS returns at least 1, so a caller has no
 * "no table" branch to get wrong.
 *
 * @return how many destinations were written, 1..FT_TT_REPLICA_MAX_DESTINATIONS
 */
FTR_INLINE uint32_t ftr_resolve_destinations(const volatile uint32_t *table, uint32_t x,
                                             uint32_t y, uint32_t base, uint32_t *out_x,
                                             uint32_t *out_y, uint32_t *out_base)
{
    uint32_t sets, set;

    out_x[0] = x;
    out_y[0] = y;
    out_base[0] = base;
    if (!table || table[0] != FT_TT_REPLICA_TABLE_MAGIC)
        return 1u;
    sets = table[1];
    if (sets > FT_TT_REPLICA_MAX_SETS)
        return 1u;

    for (set = 0u; set < sets; set++) {
        const uint32_t record = ftr_record_offset(set);
        const uint32_t word = table[record];
        const uint32_t count = word & FT_TT_REPLICA_COUNT_MASK;
        const uint32_t range_units = word >> FT_TT_REPLICA_RANGE_SHIFT;
        uint32_t index, offset;

        if (count < 2u || count > FT_TT_REPLICA_MAX_DESTINATIONS)
            continue;
        /* THE KEY IS ALL THREE WORDS. A record whose primary is not exactly this
         * descriptor's destination is a different strip, whatever it shares. A RANGE
         * record keys (x, y) exactly and the base by the range (unsigned: below the
         * primary wraps past it and misses). */
        if (table[record + 1u] != x || table[record + 2u] != y)
            continue;
        offset = base - table[record + 3u];
        if (range_units == 0u ? offset != 0u : offset >= range_units * FT_TT_REPLICA_RANGE_UNIT)
            continue;
        for (index = 1u; index < count; index++) {
            out_x[index] = table[record + 1u + index * 3u];
            out_y[index] = table[record + 2u + index * 3u];
            out_base[index] = table[record + 3u + index * 3u] + offset;
        }
        return count;
    }
    return 1u;
}

#endif /* FOUNDRY_RESIDENT_OUTPUT_REPLICAS_H */
