/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file dispatch_gate.c
 * @brief PHASE D's gate: a ONE-SHOT dispatcher on BRISC that issues a recorded list of resident
 *        descriptors into ANOTHER core's queue over the NoC and awaits each one there — the
 *        on-card half of the host's issue + await, so its per-descriptor cost can be priced
 *        against the host round trip before the full dispatcher is built.
 *
 * @status Gate kernel. Driven by tests/test_phase5_resident.c stage 7.
 * @issues None
 * @todo None
 *
 * WHAT IT REPRODUCES, AND THE TWO RULES IT MUST KEEP (tt_ornith_optimized_spec.md PHASE D):
 *   per entry — write the 32 B descriptor into the target's ring slot (token pre-stamped by
 *   the host, which owns the target's run nonce), barrier, THEN bump the target's HEAD with one
 *   4 B inline write (descriptors before head), barrier, then poll the target's words 4..7
 *   (stop, completed count, completed token, state) with a 16 B NoC READ until the count
 *   reaches the entry, and check the token and the state the host would check.
 *   It NEVER posts into a triad writer's path (the `ftq_group_post` wedge suspect); its only
 *   writes are the ring slot and the head word of the ONE target the table names.
 *
 * THE TABLE (runtime arg 0 = its L1 address, 64-byte aligned), words:
 *   0 magic 0x47505344 "DSPG"    1 entries          2 target NoC x    3 target NoC y
 *   4 target control address     5 target ring      6 capacity mask   7 first queue index
 *   8 poll spin bound            16..19 poll buffer (16 B, kernel-owned)
 *   24 state (kernel: 1 running, 2 done, 3 failed)   25 entries completed   26 error
 *   27 last count seen           28 last token seen  29 last state seen
 *   32.. entries, 16 words each: 0..7 the descriptor (token at word 1), 8 expected token.
 * Errors: 1 bad magic, 2 poll bound exhausted, 3 token mismatch, 4 target state poison/error.
 */

#include <stdbool.h>
#include <stdint.h>

#include "../fw/ft_bh_regs.h"
#include "ft_tensix_dataflow.h"
#include "ft_tensix_noc.h"

#define DSPG_MAGIC          0x47505344u
#define DSPG_ENTRY_WORDS    16u
#define DSPG_FIRST_ENTRY    32u
#define QUEUE_HEAD_WORD     2u
#define QUEUE_WINDOW_OFFSET 16u /* words 4..7 */
#define QUEUE_STATE_POISON  0x7B1B1B1Bu

static void dispatch_fail(volatile uint32_t *table, uint32_t error, uint32_t done)
{
    table[25] = done;
    table[26] = error;
    table[24] = 3u;
}

void kernel_main(void)
{
    volatile uint32_t *table = (volatile uint32_t *)(uintptr_t)ft_tensix_runtime_arg(0);
    volatile uint32_t *poll;
    uint32_t entries, x, y, control, ring, mask, first, bound, entry;

    table[24] = 1u;
    if (table[0] != DSPG_MAGIC) {
        dispatch_fail(table, 1u, 0u);
        return;
    }
    entries = table[1];
    x = table[2];
    y = table[3];
    control = table[4];
    ring = table[5];
    mask = table[6];
    first = table[7];
    bound = table[8];
    poll = table + 16;

    for (entry = 0; entry < entries; entry++) {
        const uint32_t index = first + entry;
        volatile uint32_t *record = table + DSPG_FIRST_ENTRY + entry * DSPG_ENTRY_WORDS;
        uint32_t spins = 0u;
        uint32_t count, token, state;

        /* 1. THE DESCRIPTOR, then a barrier: it must have landed before the head says so. */
        ft_tensix_noc_write((uint32_t)(uintptr_t)record,
                            ft_tensix_noc_address(x, y, ring + (index & mask) * 32u), 32u);
        ft_tensix_noc_write_barrier();
        /* 2. THE HEAD — one 4 B inline write, the host's own publish. */
        if (ft_tensix_noc_inline_dword_write_l1(
                ft_tensix_noc_address(x, y, control + QUEUE_HEAD_WORD * 4u), index + 1u) !=
            FT_TENSIX_NOC_ISSUED) {
            dispatch_fail(table, 4u, entry);
            return;
        }
        ft_tensix_noc_write_barrier();
        /* 3. THE AWAIT — a 16 B read of the target's completion window, as the host's `line`. */
        for (;;) {
            ft_tensix_noc_read(ft_tensix_noc_address(x, y, control + QUEUE_WINDOW_OFFSET),
                               (uint32_t)(uintptr_t)poll, 16u);
            ft_tensix_noc_read_barrier();
            ft_bh_l1_cache_invalidate();
            count = poll[1];
            if (count >= index + 1u)
                break;
            if (++spins >= bound) {
                table[27] = count;
                table[29] = poll[3];
                dispatch_fail(table, 2u, entry);
                return;
            }
        }
        token = poll[2];
        state = poll[3];
        table[27] = count;
        table[28] = token;
        table[29] = state;
        if (state == QUEUE_STATE_POISON) {
            dispatch_fail(table, 4u, entry);
            return;
        }
        if (token != record[8]) {
            dispatch_fail(table, 3u, entry);
            return;
        }
    }
    table[25] = entries;
    table[26] = 0u;
    table[24] = 2u;
}
