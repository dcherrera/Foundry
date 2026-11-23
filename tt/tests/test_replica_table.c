/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_replica_table.c
 * @brief The resident replica table, off hardware — the wire format, the refusals, and the
 *        KERNEL'S OWN LOOKUP run against the host's own encoding.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS SUITE IS FOR
 * ---------------------------------------------------------------------------
 * A replica map has no failure mode the card can report. If a destination is wrong, the
 * writer writes a tile somewhere harmless and one bank's copy of the strip goes stale; the
 * matmul member that reads that bank then returns a plausible wrong answer with every
 * kernel completing, every residency healthy and no error word set. So the map is checked
 * where checking is cheap.
 *
 * THE LOOKUP UNDER TEST IS THE KERNEL'S. `ftr_resolve_destinations()` is compiled into
 * `resident_vector_writer.c` from the same header this file includes, so a change to the
 * kernel's search changes this suite's answers. That is the point — the alternative is a
 * host model of the kernel, which is a second source of truth and has cost this project a
 * round already.
 *
 * ---------------------------------------------------------------------------
 * THE RED ARMS, AND WHAT EACH WOULD HAVE COST ON SILICON
 * ---------------------------------------------------------------------------
 *  1. NO MAGIC — a region nobody wrote must resolve to one destination, not to whatever the
 *     L1 held. Red: the lookup reads the count anyway and fans a tile out to garbage NoC
 *     coordinates.
 *  2. A PARTIAL KEY — matching on the bank alone. Red: any descriptor whose output shares a
 *     bank with a replicated strip is fanned out over the replicated strip's addresses,
 *     which corrupts an unrelated buffer.
 *  3. A DUPLICATE DESTINATION — encode must refuse. Red: the map looks armed, the writes
 *     land twice in one bank, and the whole mechanism delivers one bank's bandwidth while
 *     reporting R.
 *
 *   make -f Makefile.replicatable run
 *
 * @status Written 2026-08-19 with the A-bank spread.
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <string.h>

#include "ft_tt_op_replicas.h"
#include "ft_tt_test_report.h"

/** A table with one four-way set, so every field has a distinct value. */
static void build_reference_table(ft_tt_op_replica_table *table)
{
    uint32_t index;

    memset(table, 0, sizeof(*table));
    table->set_count = 1u;
    table->set[0].count = 4u;
    for (index = 0u; index < 4u; index++) {
        table->set[0].destination[index].x = (uint16_t)(10u + index);
        table->set[0].destination[index].y = (uint16_t)(20u + index);
        table->set[0].destination[index].base = 0x100000u * (index + 1u);
    }
}

static void check_encode_round_trip(void)
{
    uint32_t words[FT_TT_REPLICA_TABLE_WORDS];
    uint32_t out_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    ft_tt_op_replica_table table;
    uint32_t found, index;
    bool all = true;

    ft_tt_test_step("encode, then resolve with the KERNEL's own lookup");
    build_reference_table(&table);
    ft_tt_test_check_fmt("a four-way set encodes", ft_tt_op_replica_table_encode(&table, words) == 0,
                         "%u words", (unsigned)FT_TT_REPLICA_TABLE_WORDS);
    ft_tt_test_check_fmt("word 0 is the magic", words[0] == FT_TT_REPLICA_TABLE_MAGIC,
                         "0x%08x", words[0]);
    ft_tt_test_check_fmt("word 1 is the set count", words[1] == 1u, "%u", words[1]);

    found = ftr_resolve_destinations(words, table.set[0].destination[0].x,
                                     table.set[0].destination[0].y,
                                     table.set[0].destination[0].base, out_x, out_y, out_base);
    ft_tt_test_check_fmt("the primary resolves to all four destinations", found == 4u, "%u",
                         found);
    for (index = 0u; index < found && index < 4u; index++)
        all = all && out_x[index] == table.set[0].destination[index].x &&
              out_y[index] == table.set[0].destination[index].y &&
              out_base[index] == table.set[0].destination[index].base;
    ft_tt_test_check("every destination comes back in order, x, y and base", all, "all four");

    /* A DESTINATION THAT IS NOT THE PRIMARY IS NOT A KEY. Otherwise a replica's own address
     * would fan out again — the same tile written R times to R^2 places, and the second
     * fan-out reads a record whose primary is somebody else's mirror. */
    found = ftr_resolve_destinations(words, table.set[0].destination[1].x,
                                     table.set[0].destination[1].y,
                                     table.set[0].destination[1].base, out_x, out_y, out_base);
    ft_tt_test_check_fmt("a MIRROR's own address is not a key", found == 1u, "%u", found);
}

static void check_miss_is_the_shipped_path(void)
{
    uint32_t words[FT_TT_REPLICA_TABLE_WORDS];
    uint32_t out_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    ft_tt_op_replica_table table;
    uint32_t found;

    ft_tt_test_step("a miss is exactly one destination — the shipped path");
    build_reference_table(&table);
    (void)ft_tt_op_replica_table_encode(&table, words);

    found = ftr_resolve_destinations(words, 99u, 99u, 0x777000u, out_x, out_y, out_base);
    ft_tt_test_check_fmt("an unrelated destination resolves to itself alone", found == 1u,
                         "%u", found);
    ft_tt_test_check_fmt("and it is unchanged",
                         out_x[0] == 99u && out_y[0] == 99u && out_base[0] == 0x777000u,
                         "(%u,%u)+0x%x", out_x[0], out_y[0], out_base[0]);

    /* RED ARM 2, as a positive check: the key is ALL THREE WORDS. Same bank, different
     * offset must MISS. A bank-only key would fan this out over the replicated strip's
     * addresses and corrupt a buffer that has nothing to do with the map. */
    found = ftr_resolve_destinations(words, table.set[0].destination[0].x,
                                     table.set[0].destination[0].y,
                                     table.set[0].destination[0].base + 0x1000u, out_x, out_y,
                                     out_base);
    ft_tt_test_check_fmt("the same bank at a different offset MISSES", found == 1u, "%u", found);
}

static void check_empty_and_absent(void)
{
    uint32_t words[FT_TT_REPLICA_TABLE_WORDS];
    uint32_t garbage[FT_TT_REPLICA_TABLE_WORDS];
    uint32_t out_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    ft_tt_op_replica_table empty;
    uint32_t index;

    ft_tt_test_step("the OFF state — an empty table and an unwritten region");
    memset(&empty, 0, sizeof(empty));
    ft_tt_test_check("an empty table encodes", ft_tt_op_replica_table_encode(&empty, words) == 0,
                     "set_count 0");
    ft_tt_test_check_fmt("and every destination misses it",
                         ftr_resolve_destinations(words, 3u, 4u, 0x5000u, out_x, out_y,
                                                  out_base) == 1u,
                         "1 destination");

    /* RED ARM 1, AND IT HAD TO BE SHARPENED TO DISCRIMINATE. The first version of this arm
     * filled the region with 0xDEADBEEF and passed with the magic check DELETED — the set
     * count read 0xDEADBEEF, which the range guard rejects anyway, so it tested the guard
     * and not the magic. The region that actually needs the magic is one holding a
     * STRUCTURALLY VALID table nobody meant to arm: the residue of an earlier run, or the
     * same L1 words another program used for something else. Here it is, built by hand, and
     * removing the magic check makes this check fail. */
    for (index = 0u; index < FT_TT_REPLICA_TABLE_WORDS; index++)
        garbage[index] = 0u;
    garbage[1] = 1u;                             /* one set */
    garbage[ftr_record_offset(0u)] = 2u;         /* of two destinations */
    garbage[ftr_record_offset(0u) + 1u] = 3u;    /* whose primary is exactly the query */
    garbage[ftr_record_offset(0u) + 2u] = 4u;
    garbage[ftr_record_offset(0u) + 3u] = 0x5000u;
    garbage[ftr_record_offset(0u) + 4u] = 31u;   /* and whose mirror is not DRAM at all */
    garbage[ftr_record_offset(0u) + 5u] = 31u;
    garbage[ftr_record_offset(0u) + 6u] = 0x9000u;
    ft_tt_test_check_fmt("a VALID-LOOKING region with no magic resolves to ONE destination",
                         ftr_resolve_destinations(garbage, 3u, 4u, 0x5000u, out_x, out_y,
                                                  out_base) == 1u,
                         "magic 0x%08x is not 0x%08x", garbage[0], FT_TT_REPLICA_TABLE_MAGIC);
    ft_tt_test_check_fmt("...and the SAME words WITH the magic would have fanned out",
                         (garbage[0] = FT_TT_REPLICA_TABLE_MAGIC,
                          ftr_resolve_destinations(garbage, 3u, 4u, 0x5000u, out_x, out_y,
                                                   out_base)) == 2u,
                         "so the magic is what refused it");
    memset(garbage, 0, sizeof(garbage));
    ft_tt_test_check("a ZEROED region resolves to ONE destination",
                     ftr_resolve_destinations(garbage, 3u, 4u, 0x5000u, out_x, out_y,
                                              out_base) == 1u,
                     "no magic");
    ft_tt_test_check("a NULL table resolves to ONE destination",
                     ftr_resolve_destinations(NULL, 3u, 4u, 0x5000u, out_x, out_y,
                                              out_base) == 1u,
                     "no table at all");
}

static void check_refusals(void)
{
    uint32_t words[FT_TT_REPLICA_TABLE_WORDS];
    ft_tt_op_replica_table table;

    ft_tt_test_step("the four refusals, each of which is a silent wrong answer otherwise");

    build_reference_table(&table);
    table.set[0].count = 1u;
    ft_tt_test_check("a one-destination set is refused",
                     ft_tt_op_replica_table_encode(&table, words) != 0, "count 1");

    build_reference_table(&table);
    table.set[0].count = FT_TT_REPLICA_MAX_DESTINATIONS + 1u;
    ft_tt_test_check("more destinations than the format holds is refused",
                     ft_tt_op_replica_table_encode(&table, words) != 0, "count 9");

    /* RED ARM 3. This is the bank-cursor bug: two "replicas" that are one bank. */
    build_reference_table(&table);
    table.set[0].destination[2] = table.set[0].destination[1];
    ft_tt_test_check("a DUPLICATE destination is refused",
                     ft_tt_op_replica_table_encode(&table, words) != 0, "two of one bank");

    build_reference_table(&table);
    table.set_count = 2u;
    table.set[1] = table.set[0];
    ft_tt_test_check("two sets with the same PRIMARY are refused",
                     ft_tt_op_replica_table_encode(&table, words) != 0, "a dead second set");

    build_reference_table(&table);
    table.set_count = FT_TT_REPLICA_MAX_SETS + 1u;
    ft_tt_test_check("more sets than the format holds is refused",
                     ft_tt_op_replica_table_encode(&table, words) != 0, "set_count 5");
}

/**
 * Every set the format holds, all at the widest destination count, so the record stride is
 * exercised at its extreme rather than at one.
 */
static void check_full_table(void)
{
    uint32_t words[FT_TT_REPLICA_TABLE_WORDS];
    uint32_t out_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    ft_tt_op_replica_table table;
    uint32_t set, index;
    bool all = true;

    ft_tt_test_step("a FULL table — every set, every destination");
    memset(&table, 0, sizeof(table));
    table.set_count = FT_TT_REPLICA_MAX_SETS;
    for (set = 0u; set < FT_TT_REPLICA_MAX_SETS; set++) {
        table.set[set].count = FT_TT_REPLICA_MAX_DESTINATIONS;
        for (index = 0u; index < FT_TT_REPLICA_MAX_DESTINATIONS; index++) {
            table.set[set].destination[index].x = (uint16_t)(1u + index);
            table.set[set].destination[index].y = (uint16_t)(2u + index);
            table.set[set].destination[index].base = 0x10000u * (set + 1u) + 0x100u * index;
        }
    }
    ft_tt_test_check("it encodes", ft_tt_op_replica_table_encode(&table, words) == 0,
                     "4 sets x 8");

    for (set = 0u; set < FT_TT_REPLICA_MAX_SETS; set++) {
        uint32_t found = ftr_resolve_destinations(words, table.set[set].destination[0].x,
                                                  table.set[set].destination[0].y,
                                                  table.set[set].destination[0].base, out_x,
                                                  out_y, out_base);

        all = all && found == FT_TT_REPLICA_MAX_DESTINATIONS;
        for (index = 0u; index < FT_TT_REPLICA_MAX_DESTINATIONS && found == 8u; index++)
            all = all && out_base[index] == table.set[set].destination[index].base;
    }
    ft_tt_test_check("every set resolves to its own eight", all, "4 of 4");
    ft_tt_test_check_fmt("the last set's record ends inside the reservation",
                         (ftr_record_offset(FT_TT_REPLICA_MAX_SETS - 1u) +
                          FT_TT_REPLICA_RECORD_WORDS) * 4u <= FT_TT_REPLICA_TABLE_BYTES,
                         "%u B of %u", (unsigned)(FT_TT_REPLICA_TABLE_WORDS * 4u),
                         (unsigned)FT_TT_REPLICA_TABLE_BYTES);
}

/** RANGE RECORDS: one record covers a strip written band by band and slice by slice. */
static void check_range_records(void)
{
    uint32_t words[FT_TT_REPLICA_TABLE_WORDS];
    uint32_t out_x[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t out_base[FT_TT_REPLICA_MAX_DESTINATIONS];
    const uint32_t range = 64u * 128u * FT_TT_REPLICA_RANGE_UNIT; /* 64 bands x 128 tiles */
    ft_tt_op_replica_table table;
    uint32_t found, index, offset;
    bool all = true;

    ft_tt_test_step("range records: any base inside [primary, primary + range), same offset");
    build_reference_table(&table);
    table.set[0].range_bytes = range;
    ft_tt_test_check("a range record encodes", ft_tt_op_replica_table_encode(&table, words) == 0,
                     "16 MiB");
    ft_tt_test_check_fmt("the count word carries the range above the count",
                         words[ftr_record_offset(0u)] ==
                             (4u | (range / FT_TT_REPLICA_RANGE_UNIT) << FT_TT_REPLICA_RANGE_SHIFT),
                         "0x%08x", words[ftr_record_offset(0u)]);
    const uint32_t offsets[] = { 0u, 0x800u, 37u * 128u * 2048u + 14u * 2048u, range - 2048u };
    for (offset = 0u; offset < sizeof(offsets) / sizeof(offsets[0]); offset++) {
        found = ftr_resolve_destinations(words, table.set[0].destination[0].x,
                                         table.set[0].destination[0].y,
                                         table.set[0].destination[0].base + offsets[offset],
                                         out_x, out_y, out_base);
        all = all && found == 4u;
        for (index = 0u; index < found && index < 4u; index++)
            all = all && out_x[index] == table.set[0].destination[index].x &&
                  out_y[index] == table.set[0].destination[index].y &&
                  out_base[index] == table.set[0].destination[index].base + offsets[offset];
    }
    ft_tt_test_check("bases inside the range fan out at the SAME offset in every mirror", all,
                     "4 offsets x 4 destinations");
    found = ftr_resolve_destinations(words, table.set[0].destination[0].x,
                                     table.set[0].destination[0].y,
                                     table.set[0].destination[0].base + range, out_x, out_y,
                                     out_base);
    ft_tt_test_check_fmt("the first byte past the range MISSES", found == 1u, "%u", found);
    found = ftr_resolve_destinations(words, table.set[0].destination[0].x,
                                     table.set[0].destination[0].y,
                                     table.set[0].destination[0].base - 2048u, out_x, out_y,
                                     out_base);
    ft_tt_test_check_fmt("a base below the primary MISSES", found == 1u, "%u", found);
    found = ftr_resolve_destinations(words, table.set[0].destination[0].x + 1u,
                                     table.set[0].destination[0].y,
                                     table.set[0].destination[0].base + 0x800u, out_x, out_y,
                                     out_base);
    ft_tt_test_check_fmt("another bank inside the range MISSES", found == 1u, "%u", found);

    table.set[0].range_bytes = range + 100u;
    ft_tt_test_check("a range that is not a multiple of the unit is REFUSED",
                     ft_tt_op_replica_table_encode(&table, words) != 0, "range + 100 B");
    table.set[0].range_bytes = range;
    table.set_count = 2u;
    table.set[1] = table.set[0];
    table.set[1].range_bytes = 0u;
    for (index = 0u; index < 4u; index++)
        table.set[1].destination[index].base += 0x1000u + index * 0x10u;
    ft_tt_test_check("an exact record INSIDE another's range is REFUSED (it could never hit)",
                     ft_tt_op_replica_table_encode(&table, words) != 0, "overlap");
}

int main(void)
{
    printf("=== the resident replica table, off hardware\n");
    check_encode_round_trip();
    check_miss_is_the_shipped_path();
    check_empty_and_absent();
    check_refusals();
    check_full_table();
    check_range_records();
    return ft_tt_test_finish();
}
