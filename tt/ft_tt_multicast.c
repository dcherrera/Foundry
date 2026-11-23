/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_multicast.c
 * @brief Implementation of ft_tt_multicast.h — see that header for why each refusal
 *   exists. Nothing here touches the device.
 */

#include "ft_tt_multicast.h"

#include "ft_tt_device.h"

/* The encoding, transliterated from
 * the NoC multicast-address layout (the measured encoding in test_multicast_rectangle.c).
 * Kept as file-private constants rather than reaching into the kernel header, because
 * this file must build on the Mac where no tt-metal and no Tensix headers exist. */
#define FT_TT_MULTICAST_LOCAL_BITS 36u
#define FT_TT_MULTICAST_NODE_ID_BITS 6u

/** L1 is 1536 KiB; ft_tt_noc.h's ft_tt_noc_space_size() says the same thing for the
 *  transfer path. Named here so this file has no link dependency on the NoC module. */
#define FT_TT_MULTICAST_L1_BYTES (1536u * 1024u)

/* ------------------------------------------------------------------------- */

static bool coordinate_fits(ft_tt_noc_coord coord)
{
    return coord.x <= FT_TT_MULTICAST_MAX_COORDINATE &&
           coord.y <= FT_TT_MULTICAST_MAX_COORDINATE;
}

static void fill_counts(ft_tt_multicast_rectangle *rectangle, ft_tt_noc_coord sender)
{
    rectangle->node_count = (rectangle->x_high - rectangle->x_low + 1u) *
                            (rectangle->y_high - rectangle->y_low + 1u);
    rectangle->sender_inside = ft_tt_multicast_rectangle_contains(rectangle, sender);
    rectangle->destination_count =
        rectangle->sender_inside ? rectangle->node_count - 1u : rectangle->node_count;
}

/* ------------------------------------------------------------------------- */

bool ft_tt_multicast_rectangle_contains(const ft_tt_multicast_rectangle *rectangle,
                                        ft_tt_noc_coord coord)
{
    if (rectangle == NULL)
        return false;
    return coord.x >= rectangle->x_low && coord.x <= rectangle->x_high &&
           coord.y >= rectangle->y_low && coord.y <= rectangle->y_high;
}

int ft_tt_multicast_check_rectangle(const ft_tt_multicast_rectangle *rectangle)
{
    uint32_t nodes;

    if (rectangle == NULL) {
        ft_tt_log_error("multicast: NULL rectangle");
        return FT_TT_EINVAL;
    }
    if (rectangle->x_low > rectangle->x_high || rectangle->y_low > rectangle->y_high) {
        ft_tt_log_error("multicast: inverted rectangle (%u,%u)..(%u,%u) — the hardware does "
                        "not validate the corners, it broadcasts to whatever they describe",
                        rectangle->x_low, rectangle->y_low, rectangle->x_high,
                        rectangle->y_high);
        return FT_TT_EINVAL;
    }
    if (rectangle->x_high > FT_TT_MULTICAST_MAX_COORDINATE ||
        rectangle->y_high > FT_TT_MULTICAST_MAX_COORDINATE) {
        ft_tt_log_error("multicast: corner (%u,%u) is above the 6-bit node-id field; it "
                        "would truncate onto a DIFFERENT live endpoint",
                        rectangle->x_high, rectangle->y_high);
        return FT_TT_EINVAL;
    }

    nodes = (rectangle->x_high - rectangle->x_low + 1u) *
            (rectangle->y_high - rectangle->y_low + 1u);
    if (rectangle->node_count != nodes) {
        ft_tt_log_error("multicast: node_count %u does not match the rectangle's %u nodes",
                        rectangle->node_count, nodes);
        return FT_TT_EINVAL;
    }
    if (rectangle->destination_count !=
        (rectangle->sender_inside ? nodes - 1u : nodes)) {
        ft_tt_log_error("multicast: destination_count %u disagrees with %u nodes and a "
                        "sender that is %s the rectangle — the write barrier's accounting "
                        "is this number",
                        rectangle->destination_count, nodes,
                        rectangle->sender_inside ? "inside" : "outside");
        return FT_TT_EINVAL;
    }
    if (rectangle->destination_count == 0u) {
        ft_tt_log_error("multicast: the rectangle leaves ZERO destinations. A multicast to "
                        "nobody is the zero-length NoC transfer in different clothes and it "
                        "is refused, not optimised away (DEV_NOTES.md §1)");
        return FT_TT_ERROR;
    }
    return FT_TT_OK;
}

int ft_tt_multicast_rectangle_from_corners(uint32_t x_low, uint32_t y_low, uint32_t x_high,
                                           uint32_t y_high, ft_tt_noc_coord sender,
                                           ft_tt_multicast_rectangle *out)
{
    ft_tt_multicast_rectangle rectangle;

    if (out == NULL) {
        ft_tt_log_error("multicast: NULL out");
        return FT_TT_EINVAL;
    }
    rectangle.x_low = x_low;
    rectangle.y_low = y_low;
    rectangle.x_high = x_high;
    rectangle.y_high = y_high;
    rectangle.node_count = 0u;
    rectangle.destination_count = 0u;
    rectangle.sender_inside = false;

    /* fill_counts() would compute a nonsense span from an inverted or oversized pair, so
     * the corners are checked BEFORE it runs. `out` is left at the zeroed counts on a
     * refusal, which ft_tt_multicast_check_rectangle() also refuses, so a caller that
     * ignores the return code still cannot launch with it. */
    if (x_low > x_high || y_low > y_high ||
        x_high > FT_TT_MULTICAST_MAX_COORDINATE || y_high > FT_TT_MULTICAST_MAX_COORDINATE) {
        *out = rectangle;
        return ft_tt_multicast_check_rectangle(&rectangle);
    }
    fill_counts(&rectangle, sender);
    *out = rectangle;
    return ft_tt_multicast_check_rectangle(&rectangle);
}

int ft_tt_multicast_rectangle_from_cores(const ft_tt_noc_coord *members, uint32_t count,
                                         ft_tt_noc_coord sender,
                                         ft_tt_multicast_rectangle *out)
{
    ft_tt_multicast_rectangle rectangle;
    uint32_t index;
    uint32_t other;

    if (members == NULL || out == NULL || count == 0u) {
        ft_tt_log_error("multicast: NULL members/out or a zero member count");
        return FT_TT_EINVAL;
    }
    if (!coordinate_fits(sender)) {
        ft_tt_log_error("multicast: sender (%u,%u) is above the 6-bit node-id field",
                        sender.x, sender.y);
        return FT_TT_EINVAL;
    }

    rectangle.x_low = members[0].x;
    rectangle.x_high = members[0].x;
    rectangle.y_low = members[0].y;
    rectangle.y_high = members[0].y;
    for (index = 0u; index < count; index++) {
        if (!coordinate_fits(members[index])) {
            ft_tt_log_error("multicast: member %u at (%u,%u) is above the 6-bit node-id "
                            "field; it would truncate onto a DIFFERENT live endpoint",
                            index, members[index].x, members[index].y);
            return FT_TT_EINVAL;
        }
        /* Distinctness is checked pairwise rather than assumed, because the fill test
         * below is a COUNT and a duplicated member would let a non-rectangle pass it.
         * This is the same "check the N units against EACH OTHER" rule that caught the
         * multicore seed collision. */
        for (other = 0u; other < index; other++) {
            if (members[other].x == members[index].x && members[other].y == members[index].y) {
                ft_tt_log_error("multicast: members %u and %u are both (%u,%u); a duplicate "
                                "would let a set that is NOT a rectangle pass the fill check",
                                other, index, members[index].x, members[index].y);
                return FT_TT_EINVAL;
            }
        }
        if (members[index].x < rectangle.x_low)
            rectangle.x_low = members[index].x;
        if (members[index].x > rectangle.x_high)
            rectangle.x_high = members[index].x;
        if (members[index].y < rectangle.y_low)
            rectangle.y_low = members[index].y;
        if (members[index].y > rectangle.y_high)
            rectangle.y_high = members[index].y;
    }

    fill_counts(&rectangle, sender);

    /* The fill test. Distinct members whose cardinality equals the bounding box's IS the
     * box, by pigeonhole — so this needs no grid and cannot be fooled by a harvested
     * column, which simply makes the box bigger than the set. */
    if (count != rectangle.node_count) {
        ft_tt_log_error("multicast: %u cores do not FILL their bounding box (%u,%u)..(%u,%u), "
                        "which holds %u nodes. A multicast writes to every node in the box, "
                        "so this would write into %u core(s) that are not in the set",
                        count, rectangle.x_low, rectangle.y_low, rectangle.x_high,
                        rectangle.y_high, rectangle.node_count, rectangle.node_count - count);
        return FT_TT_ERROR;
    }

    *out = rectangle;
    return ft_tt_multicast_check_rectangle(&rectangle);
}

int ft_tt_multicast_check_payload(uint32_t l1_address, size_t bytes)
{
    if (bytes == 0u) {
        ft_tt_log_error("multicast: a zero-length payload is refused. A zero-length NoC "
                        "transfer is what took this card to `Mem- BusMaster-` on 2026-08-12 "
                        "(DEV_NOTES.md §1)");
        return FT_TT_EINVAL;
    }
    if ((l1_address % FT_TT_MULTICAST_ALIGNMENT_BYTES) != 0u) {
        ft_tt_log_error("multicast: L1 address 0x%08x is not %u-byte aligned", l1_address,
                        FT_TT_MULTICAST_ALIGNMENT_BYTES);
        return FT_TT_EINVAL;
    }
    if ((bytes % FT_TT_MULTICAST_ALIGNMENT_BYTES) != 0u) {
        ft_tt_log_error("multicast: length %zu is not a multiple of %u", bytes,
                        FT_TT_MULTICAST_ALIGNMENT_BYTES);
        return FT_TT_EINVAL;
    }
    if ((uint64_t)l1_address + (uint64_t)bytes > (uint64_t)FT_TT_MULTICAST_L1_BYTES) {
        ft_tt_log_error("multicast: 0x%08x + %zu leaves L1 (%u bytes)", l1_address, bytes,
                        FT_TT_MULTICAST_L1_BYTES);
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

int ft_tt_multicast_encode_address(const ft_tt_multicast_rectangle *rectangle,
                                   uint32_t noc_index, uint32_t l1_address,
                                   uint64_t *out_address)
{
    uint32_t start_x;
    uint32_t start_y;
    uint32_t end_x;
    uint32_t end_y;
    int rc;

    if (out_address == NULL) {
        ft_tt_log_error("multicast: NULL out_address");
        return FT_TT_EINVAL;
    }
    if (noc_index > 1u) {
        ft_tt_log_error("multicast: NoC index %u; this part has two", noc_index);
        return FT_TT_EINVAL;
    }
    rc = ft_tt_multicast_check_rectangle(rectangle);
    if (rc != FT_TT_OK)
        return rc;

    /* NoC 0 takes the LOW corner as "start"; NoC 1 broadcasts the other way and takes them
     * reversed. */
    if (noc_index == 0u) {
        start_x = rectangle->x_low;
        start_y = rectangle->y_low;
        end_x = rectangle->x_high;
        end_y = rectangle->y_high;
    } else {
        start_x = rectangle->x_high;
        start_y = rectangle->y_high;
        end_x = rectangle->x_low;
        end_y = rectangle->y_low;
    }

    *out_address =
        ((uint64_t)start_x << (FT_TT_MULTICAST_LOCAL_BITS + 2u * FT_TT_MULTICAST_NODE_ID_BITS)) |
        ((uint64_t)start_y << (FT_TT_MULTICAST_LOCAL_BITS + 3u * FT_TT_MULTICAST_NODE_ID_BITS)) |
        ((uint64_t)end_x << FT_TT_MULTICAST_LOCAL_BITS) |
        ((uint64_t)end_y << (FT_TT_MULTICAST_LOCAL_BITS + FT_TT_MULTICAST_NODE_ID_BITS)) |
        (uint64_t)l1_address;
    return FT_TT_OK;
}
