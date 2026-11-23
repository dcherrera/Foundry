/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_replicas.c
 * @brief Implementation of the resident replica table's host half.
 *
 * @status Written 2026-08-19.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_replicas.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_mailbox.h"
#include "ft_tt_op_l1_layout.h"

/* The table lives in the kernel-config region's own free gap. These are the two facts that
 * make that true, and they are asserted rather than described because the L1 map is edited
 * for other reasons and nothing else would notice. */
FT_TT_STATIC_ASSERT(FT_TT_REPLICA_TABLE_ADDRESS >=
                        FT_TT_OP_RTA_BASE + 5u * FT_TT_OP_RTA_STRIDE,
                    replica_table_is_past_the_runtime_arg_blocks);
FT_TT_STATIC_ASSERT(FT_TT_REPLICA_TABLE_ADDRESS + FT_TT_REPLICA_TABLE_BYTES <=
                        FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_TEXT_OFFSET_DM0,
                    replica_table_is_below_the_first_kernel_text_slab);
FT_TT_STATIC_ASSERT(FT_TT_REPLICA_TABLE_WORDS * 4u <= FT_TT_REPLICA_TABLE_BYTES,
                    replica_table_words_fit_the_reservation);

/** Two destinations are the same PLACE, which is the one thing a replica must never be. */
static bool replica_destination_equal(const ft_tt_op_replica_destination *a,
                                      const ft_tt_op_replica_destination *b)
{
    return a->x == b->x && a->y == b->y && a->base == b->base;
}

int ft_tt_op_replica_table_encode(const ft_tt_op_replica_table *table,
                                  uint32_t out_words[FT_TT_REPLICA_TABLE_WORDS])
{
    uint32_t set;

    if (!table || !out_words)
        return FT_TT_EINVAL;
    if (table->set_count > FT_TT_REPLICA_MAX_SETS) {
        fprintf(stderr, "ft_tt: %u replica sets is more than the %u the wire format holds\n",
                table->set_count, FT_TT_REPLICA_MAX_SETS);
        return FT_TT_EINVAL;
    }

    memset(out_words, 0, FT_TT_REPLICA_TABLE_WORDS * sizeof(out_words[0]));
    out_words[0] = FT_TT_REPLICA_TABLE_MAGIC;
    out_words[1] = table->set_count;

    for (set = 0u; set < table->set_count; set++) {
        const ft_tt_op_replica_set *entry = &table->set[set];
        const uint32_t record = ftr_record_offset(set);
        uint32_t index, other;

        if (entry->count < 2u || entry->count > FT_TT_REPLICA_MAX_DESTINATIONS) {
            fprintf(stderr, "ft_tt: replica set %u has %u destinations; 2..%u is the range "
                            "(one destination is not a replica, it is the strip)\n",
                    set, entry->count, FT_TT_REPLICA_MAX_DESTINATIONS);
            return FT_TT_EINVAL;
        }
        /* A DUPLICATE DESTINATION READS AS A WORKING MAP AND DELIVERS ONE BANK. It is the
         * exact failure a bank-cursor bug produces, and nothing downstream can see it: the
         * bytes are written twice, both readers get the right answer, and the bandwidth
         * the whole mechanism exists for is not there. */
        for (index = 0u; index < entry->count; index++)
            for (other = index + 1u; other < entry->count; other++)
                if (replica_destination_equal(&entry->destination[index],
                                              &entry->destination[other])) {
                    fprintf(stderr, "ft_tt: replica set %u names (%u,%u)+0x%x twice — that "
                                    "is one bank wearing two names\n",
                            set, entry->destination[index].x, entry->destination[index].y,
                            entry->destination[index].base);
                    return FT_TT_EINVAL;
                }
        /* Two sets with the same PRIMARY: the writer takes the first hit, so the second is
         * silently dead. Caught here because "the map has a dead entry" is invisible. */
        for (other = 0u; other < set; other++)
            if (replica_destination_equal(&entry->destination[0],
                                          &table->set[other].destination[0])) {
                fprintf(stderr, "ft_tt: replica sets %u and %u have the same primary — the "
                                "second could never be found\n", other, set);
                return FT_TT_EINVAL;
            }

        if (entry->range_bytes % FT_TT_REPLICA_RANGE_UNIT != 0u ||
            entry->range_bytes / FT_TT_REPLICA_RANGE_UNIT >
                (0xFFFFFFFFu >> FT_TT_REPLICA_RANGE_SHIFT)) {
            fprintf(stderr, "ft_tt: replica set %u range %u B is not a multiple of %u or is "
                            "too long for the record\n", set, entry->range_bytes,
                    FT_TT_REPLICA_RANGE_UNIT);
            return FT_TT_EINVAL;
        }
        /* OVERLAPPING RANGES: the writer takes the first hit, so the second record's
         * overlap is dead — the same failure as a repeated primary. */
        for (other = 0u; other < set; other++) {
            const ft_tt_op_replica_set *prior = &table->set[other];
            const uint32_t a0 = entry->destination[0].base, b0 = prior->destination[0].base;
            const uint32_t a1 = a0 + (entry->range_bytes ? entry->range_bytes : 1u);
            const uint32_t b1 = b0 + (prior->range_bytes ? prior->range_bytes : 1u);

            if (entry->destination[0].x == prior->destination[0].x &&
                entry->destination[0].y == prior->destination[0].y && a0 < b1 && b0 < a1) {
                fprintf(stderr, "ft_tt: replica sets %u and %u overlap — the second could "
                                "never be found there\n", other, set);
                return FT_TT_EINVAL;
            }
        }
        out_words[record] = entry->count |
                            (entry->range_bytes / FT_TT_REPLICA_RANGE_UNIT)
                                << FT_TT_REPLICA_RANGE_SHIFT;
        for (index = 0u; index < entry->count; index++) {
            out_words[record + 1u + index * 3u] = entry->destination[index].x;
            out_words[record + 2u + index * 3u] = entry->destination[index].y;
            out_words[record + 3u + index * 3u] = entry->destination[index].base;
        }
    }
    return FT_TT_OK;
}

int ft_tt_op_replica_table_write(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                 const ft_tt_op_replica_table *table)
{
    uint32_t words[FT_TT_REPLICA_TABLE_WORDS];
    int rc;

    if (!noc || !noc->write)
        return FT_TT_EINVAL;
    rc = ft_tt_op_replica_table_encode(table, words);
    if (rc != FT_TT_OK)
        return rc;
    return noc->write(noc->user, x, y, FT_TT_REPLICA_TABLE_ADDRESS, words, sizeof(words));
}

int ft_tt_op_replica_table_verify(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  const ft_tt_op_replica_table *table)
{
    uint32_t want[FT_TT_REPLICA_TABLE_WORDS];
    uint32_t got[FT_TT_REPLICA_TABLE_WORDS];
    uint32_t index;
    int rc;

    if (!noc || !noc->read)
        return FT_TT_EINVAL;
    rc = ft_tt_op_replica_table_encode(table, want);
    if (rc != FT_TT_OK)
        return rc;
    rc = noc->read(noc->user, x, y, FT_TT_REPLICA_TABLE_ADDRESS, got, sizeof(got));
    if (rc != FT_TT_OK)
        return rc;
    for (index = 0u; index < FT_TT_REPLICA_TABLE_WORDS; index++)
        if (got[index] != want[index]) {
            fprintf(stderr, "ft_tt: the replica table on (%u,%u) differs at word %u: "
                            "0x%08x, wanted 0x%08x\n", x, y, index, got[index], want[index]);
            return FT_TT_ERROR;
        }
    return FT_TT_OK;
}
