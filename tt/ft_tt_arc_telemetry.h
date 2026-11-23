/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_arc_telemetry.h
 * @brief Reader for the telemetry table the ARC firmware publishes: look up tags by id,
 *   and build the card's grid topology (ft_tt_grid_info) from the five grid tags.
 *
 * THE FORMAT (all words little-endian, 32-bit; addresses are ARC-tile-local)
 *   - Scratch register k sits at `arch->telemetry.scratch_base + 4*k`. The register at
 *     `table_pointer_scratch_index` holds the TABLE address, the one at
 *     `values_pointer_scratch_index` the VALUES address. A table address of 0 means
 *     the firmware has not published yet.
 *   - TABLE: word 0 is a format version (not interpreted), word 1 the entry count N
 *     (at most FT_TT_ARC_TELEMETRY_MAX_ENTRIES), then N entry words. Each entry packs a
 *     tag in its low 16 bits and, in its high 16 bits, a WORD offset into the values.
 *   - A tag's value is the word at `VALUES + 4*offset`. When a tag appears more than
 *     once, the first entry in table order is the one used.
 *   - Every address is checked against the ARC CSM window
 *     `[csm_base, csm_base + csm_size)` before it is read: the table header, the values
 *     base, the whole entry array and each value word. Nothing outside it is followed.
 *
 * TRANSPORTS
 *   The walk is written against one callback, `ft_tt_arc_read32_fn`, so the kernel-driver
 *   transport (a TLB window here) and the userspace driver (ft_tt_runtime_udrv.c) share
 *   it, and a host test can drive it through a fake ARC with no card.
 *
 * Return codes are the FT_TT_* values of ft_tt_device.h. Refusals are logged through
 * ft_tt_log_error().
 *
 * @status Complete; covered off-card by tests/test_arc_telemetry_walk.c and
 *   tests/test_runtime_seam.c.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_ARC_TELEMETRY_H
#define FT_TT_ARC_TELEMETRY_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_arch.h"
#include "ft_tt_device.h"

/** Largest entry count a well-formed table can carry: one entry per 16-bit tag. */
#define FT_TT_ARC_TELEMETRY_MAX_ENTRIES (1u << 16)

/**
 * Read one 32-bit word at an ARC-tile-local address.
 *
 * @param user       the transport's own state, passed through unchanged
 * @param address    tile-local byte address on the ARC tile
 * @param out_value  receives the word on success
 * @return FT_TT_OK, or an FT_TT_* error that the walk propagates unchanged
 */
typedef int (*ft_tt_arc_read32_fn)(void *user, uint32_t address, uint32_t *out_value);

/**
 * Look up `count` tags in the published table.
 *
 * Every out value and found flag is zeroed first. The table is walked in order; for each
 * requested tag not yet found, the first matching entry's value is bounds-checked and
 * read. A tag the table does not carry is left found=false with value 0 — not an error.
 *
 * @param arch        architecture whose telemetry layout to use
 * @param read32      transport
 * @param user        passed to `read32`
 * @param tags        `count` tag ids (duplicates allowed; each gets the same value)
 * @param count       number of tags
 * @param out_values  `count` words
 * @param out_found   `count` flags
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL argument; FT_TT_EUNAVAIL when no table is
 *         published yet; FT_TT_EIO for a pointer, entry array or value outside the CSM or
 *         an entry count above FT_TT_ARC_TELEMETRY_MAX_ENTRIES; any `read32` error as is.
 *
 * Thread-safe: as the transport    Reentrant: as the transport    Signal-safe: no
 */
int ft_tt_arc_telemetry_read_tags_via(const ft_tt_arch *arch, ft_tt_arc_read32_fn read32,
                                      void *user, const uint16_t *tags, unsigned count,
                                      uint32_t *out_values, bool *out_found);

/**
 * Build the grid topology from the five grid tags of `arch->telemetry` (enabled Tensix
 * columns, enabled ETH, enabled GDDR, enabled L2CPU, NoC translation).
 *
 * The derived counts mask before counting: `tensix_column_count` is the popcount of the
 * column mask restricted to the part's `tensix.column_count` low bits,
 * `tensix_core_count` that times `tensix.row_count`, and `dram_bank_count` the popcount
 * of the GDDR mask restricted to `dram.bank_count` low bits. `valid` is set and `arch`
 * recorded. `*out_grid` is written only on success.
 *
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL argument; FT_TT_EIO when any of the five
 *         tags is missing (logged, naming it) or for any table refusal of
 *         ft_tt_arc_telemetry_read_tags_via(); FT_TT_EUNAVAIL with no table yet; any
 *         `read32` error as is.
 */
int ft_tt_arc_telemetry_read_grid_via(const ft_tt_arch *arch, ft_tt_arc_read32_fn read32,
                                      void *user, ft_tt_grid_info *out_grid);

/**
 * The kernel-driver transport: read the grid through one 2 MiB uncached TLB window aimed
 * at the ARC tile (NoC0), re-aimed only when a read leaves the current window-aligned
 * page. On success the grid is installed with ft_tt_grid_set() and copied to
 * `*out_grid`. The window is always freed before returning.
 *
 * With the NoC fence closed (default build) aiming fails with FT_TT_EPERM, which is
 * returned as is.
 *
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL argument or an unavailable context;
 *         FT_TT_EUNAVAIL when the context has no architecture or no table is published;
 *         FT_TT_EIO when the CSM is larger than the window, or for any table refusal;
 *         window allocation / aim errors and ft_tt_grid_set() errors as is.
 */
int ft_tt_arc_telemetry_read_grid(ft_tt_ctx *ctx, ft_tt_grid_info *out_grid);

#endif /* FT_TT_ARC_TELEMETRY_H */
