/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_runtime_udrv.h
 * @brief THE SEAM. The userspace-driver half of `ft_tt_runtime` — everything a
 *   serving process needs from a card that has NO VENDOR DRIVER on the machine:
 *   find it in sysfs, bring the ARC up, read the grid, claim two windows, and
 *   hand back the SAME three `ft_tt_noc_access` accessors the tt-kmd path hands
 *   back.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS FILE IS THE WHOLE CHANGE, AND NOTHING ABOVE IT MOVES
 * ---------------------------------------------------------------------------
 * The backend above `ft_tt_runtime` — launch, queue, ops, residencies,
 * `ft_tt_core` — is written against `ft_tt_launch.h`'s `ft_tt_noc_access`
 * vtable and against `ft_tt_grid_info`. Neither mentions a device handle. So
 * the ONLY thing tt-kmd was supplying that could not be supplied otherwise was
 * the way those two get built, and that is exactly what this file replaces.
 *
 * There is deliberately no second copy of anything:
 *   - the PIO copy shape is `ft_tt_noc_copy.h`, shared with `ft_tt_noc.c`;
 *   - the memory-space bounds and alignment rules are
 *     `ft_tt_noc_check_transfer()`, called here, so `bulk_dram` refuses a
 *     register address on BOTH transports for the same reason and with the same
 *     message;
 *   - the telemetry table walk is `ft_tt_arc_telemetry_read_grid_via()`, shared;
 *   - the coordinates are `ft_tt_coord`, shared.
 * If a file ever has to be duplicated to make this work, the cut is in the
 * wrong place.
 *
 * ---------------------------------------------------------------------------
 * IT REFUSES. IT NEVER FALLS BACK.
 * ---------------------------------------------------------------------------
 * Every failure here returns a negative code and leaves nothing open. There is
 * no "try the other transport" path in this file or in its caller, because this
 * project has shipped a silent fallback twice — the one-shot dispatch fallback
 * and the export 404 — and both times the symptom was a correct-looking answer
 * produced by the wrong thing. A serve asked for `FT_TT_DRIVER=udrv` on a
 * machine with no usable card must stop, not answer on the CPU.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS LOST RELATIVE TO THE tt-kmd PATH — priced, not hidden
 * ---------------------------------------------------------------------------
 *  - **No `ft_tt_ctx`.** `ft_tt_runtime.ctx` is NULL on this path, so anything
 *    reached through it is unavailable: the DRAM allocation tally (accounting
 *    only — `ft_tt_allocator_init()` takes a NULL ctx) and `SET_NOC_CLEANUP`,
 *    the death-time stop write. That second one is a real loss and is stated at
 *    `ft_tt_udrv.h` §5(d); the residency idle-spin bound is what remains.
 *  - **No second reader of a window's registers.** Every aim is verified by
 *    reading the registers back through the mapping that wrote them, which is a
 *    real uncached PCIe read and catches a dropped word, a torn write or a
 *    register that does not hold — but not a mapping that consistently points
 *    at the wrong page. `ft_tt_udrv_noc.h` prices this exactly.
 *  - **Window arbitration is a flock, and the flock cannot see tt-kmd's
 *    bitmap.** While the module is loaded, a process using `ALLOCATE_TLB` and a
 *    process using this one can pick the same window and neither is told. The
 *    mitigation here is to claim from the TOP of the window range downward,
 *    because the ioctl allocates from the bottom, and to reserve the window the
 *    driver keeps for itself. It is a mitigation, not a lock. On the machine
 *    this exists for there is no ioctl and the flock is the whole arbitrator,
 *    which is the condition the rehearsal proved.
 *
 * @status **THE PRODUCT RUNS ON THE CARD OVER THIS PATH WITH `tenstorrent`
 *   UNLOADED (2026-08-17).** A real `pantheon serve` — MiniCPM5-1B,
 *   `PN_TT_SPARSE_LOGITS=1`, `PN_TT_MATMUL_CORES=4`, host attention, programs
 *   from the embedded blob — answered a chat with `[admit] path=gpu` and
 *   **6,760 matmuls on the card**, resident triad up on four cores, clean
 *   shutdown. The same binary over tt-kmd returns BYTE-IDENTICAL text, which is
 *   exactly why the text is not the gate.
 *
 *   | arm | tt-kmd | path | RUN TOTAL | s/token |
 *   |---|---|---|---|---|
 *   | kmd (control) | loaded | gpu | 6760 | 0.106 |
 *   | udrv | loaded | gpu | 6760 | 0.113 |
 *   | udrv | UNLOADED | gpu | 6760 | 0.114 |
 *   | kmd | UNLOADED | **cpu** | — | — |
 *
 *   **THE 6.6% IS THE AIM READBACK AND THAT IS MEASURED.** All of the gap is in
 *   `issue` (0.331 s / 49.0 µs per op against 0.104 / 15.3); `wait`, which is
 *   the card's own work, is identical to 0.5%, and the aim counts agree to 10
 *   out of 68,071. Deleting the readback from `ft_tt_udrv_noc_window_aim()` in
 *   a scratch build put `issue` back to 0.100 s and the token to **0.105 s — at
 *   or below the kmd control**. So the transport costs nothing and the three
 *   uncached PCIe register reads that verify each aim cost **3.3 µs per aim**,
 *   1,702 aims a token. They stay: with no vendor driver there is no second
 *   reader, and that readback is the only thing that makes a wrong aim
 *   detectable instead of silent.
 *
 *   Everything under this file is on silicon with the module removed
 *   (`ft_tt_udrv.h` @status): the ARC answers in A3, the bring-up sequence is
 *   accepted, the telemetry walk is the only reader left, the flock is the only
 *   arbitrator, and a 64 KiB L1 round trip lands.
 * @issues
 *   - The bring-up sends `POWER_SETTING` asking for the maximum AI clock and
 *     nothing lowers it at close, so a card this path owned stays at its top
 *     clock until something asks otherwise. Visible, harmless, and tracked in
 *     `ft_tt_udrv.h`'s @todo.
 *   - MRRS is not set here. It lives in DEVCTL and is rewritten by the reset
 *     path's config restore; a machine where tt-kmd has NEVER run has nobody
 *     setting it. Same @todo.
 * @todo None
 */

#ifndef FT_TT_RUNTIME_UDRV_H
#define FT_TT_RUNTIME_UDRV_H

#include <stdbool.h>
#include <stddef.h>

#include "ft_tt_arch.h"
#include "ft_tt_coord.h"
#include "ft_tt_launch.h"

/** An open card over the userspace driver. Opaque: it owns mappings, claims and
 *  paths whose lifetimes the accessors below depend on, and a caller has no
 *  business reaching into any of them. Heap-allocated so the accessors' `user`
 *  pointers stay valid however the caller moves its own runtime struct. */
typedef struct ft_tt_runtime_udrv ft_tt_runtime_udrv;

/**
 * Open device `device_index` with NO vendor driver, bring it up, and build the
 * three accessors a serving process runs on.
 *
 * The sequence, and every step is one the rehearsal ran with `tenstorrent`
 * removed:
 *   1. find the card in `/sys/bus/pci/devices` by vendor and device id, and
 *      select the architecture table from those ids alone;
 *   2. map BAR0's TLB register file, writable;
 *   3. claim two 2 MiB windows through `ft_tt_udrv_claim` — one UC for run
 *      control and the ARC, one WC for bulk data;
 *   4. bind an ARC transport to the UC path and run
 *      `ft_tt_udrv_arc_probe_hardware()`: `ASIC_STATE0`, `SET_WDT_TIMEOUT`,
 *      `POWER_SETTING`. This is what `tt_cdev_open()` was doing for us;
 *   5. walk the ARC telemetry table over that same path for the grid, and
 *      refuse a topology `ft_tt_coord` cannot map — a coordinate we cannot
 *      derive is a coordinate we must not probe (spec §7.3);
 *   6. fill the vtables.
 *
 * @param device_index  which Blackhole, in sysfs enumeration order
 * @param out_udrv      receives the handle; release with
 *                      ft_tt_runtime_udrv_close()
 * @param out_grid      the live topology
 * @param out_noc       UC / WORD32 — registers, mailbox, images. NEVER bulk
 * @param out_bulk_dram BLOCK16 over the WC path, DRAM addresses only
 * @param out_bulk_l1   BLOCK16 over the WC path, L1 addresses only
 * @return FT_TT_OK, or a negative FT_TT_* code with the reason logged. On any
 *         failure nothing is left open and no output is written.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_runtime_udrv_open(int device_index, ft_tt_runtime_udrv **out_udrv,
                            ft_tt_grid_info *out_grid, ft_tt_noc_access *out_noc,
                            ft_tt_noc_access *out_bulk_dram, ft_tt_noc_access *out_bulk_l1);

/** Release every window claim, unmap every aperture and free the handle. NULL
 *  is a no-op. The card is left in A0 at whatever clock the bring-up asked for
 *  — see @issues. */
void ft_tt_runtime_udrv_close(ft_tt_runtime_udrv *udrv);

/** MEASURE: read ARC telemetry tags (reads only, no ARC message) — see
 *  ft_tt_arc_telemetry_read_tags_via(). */
int ft_tt_runtime_udrv_read_telemetry(ft_tt_runtime_udrv *udrv, const uint16_t *tags,
                                      unsigned count, uint32_t *out_values, bool *out_found);

/** How many reader banks a runtime may hold (one per reader thread). */
#define FT_TT_UDRV_READER_BANKS_MAX 4u

/**
 * PHASE W: DRAM accessors over READER bank `index` — extra WC window banks
 * (`FT_TT_UDRV_READER_BANKS`, default 2, of `FT_TT_UDRV_READER_WINDOWS` windows each,
 * default 2, 0 = none), each for exactly ONE thread other than the one driving the control
 * and bulk banks. The prefill's alpha/beta readback runs there so it stops serialising with
 * descriptor issue on the main thread.
 *
 * @return FT_TT_OK; FT_TT_EUNAVAIL when that bank was not opened; FT_TT_EINVAL
 *
 * Thread-safe: the returned accessor may be used concurrently with the runtime's other
 *              accessors, by ONE thread at a time.
 */
int ft_tt_runtime_udrv_reader_access(ft_tt_runtime_udrv *udrv, uint32_t index,
                                     ft_tt_noc_access *out_dram);

/** The same reader bank, L1 addresses (BLOCK16): the reader polls a queue's completion word
 *  there before reading the strip that queue's descriptor writes (PHASE W wavefront). */
int ft_tt_runtime_udrv_reader_l1_access(ft_tt_runtime_udrv *udrv, uint32_t index,
                                        ft_tt_noc_access *out_l1);

/** The card's PCI address, e.g. `0000:12:00.0`. Never NULL for an open handle;
 *  the empty string for NULL. This is what `ft_tt_device_get_info()` answers on
 *  the tt-kmd path, read from sysfs instead of from an ioctl. */
const char *ft_tt_runtime_udrv_pci_address(const ft_tt_runtime_udrv *udrv);

/**
 * What each of a runtime's two windows carried.
 *
 * Declared HERE, in the udrv header, only because `ft_tt_runtime.h` includes
 * this file and the reverse would be a cycle. It describes both transports.
 *
 * **`aims` is the number that separates them.** Both paths write the same three
 * registers to move a window; the tt-kmd path then returns from an ioctl, while
 * this one READS ALL FOUR REGISTERS BACK and refuses the aperture on any
 * disagreement — a syscall can report an error and three stores through an mmap
 * cannot. Those readbacks are uncached PCIe reads on a Gen2 x1 link, so the aim
 * count is what turns a per-aim cost into a per-token one, and without it a
 * difference between the transports can only be guessed at.
 */
typedef struct ft_tt_runtime_window_stats {
    uint64_t register_read;    /**< bytes card->host through the UC window */
    uint64_t register_written; /**< bytes host->card through the UC window */
    uint64_t register_aims;    /**< times the UC window was re-aimed */
    uint64_t bulk_read;
    uint64_t bulk_written;
    uint64_t bulk_aims;
    uint64_t aim_failures; /**< aims whose readback disagreed; udrv only, must be 0 */

    /** WHAT THOSE AIMS COST, over BOTH windows, and the readback's share.
     *
     *  The aim count above turns a per-aim cost into a per-token one only if the per-aim cost
     *  is known, and it never was — it was quoted at 1.23 us from the tt-kmd ioctl path, which
     *  is a different mechanism. These are the udrv path's own clocks. ZERO on tt-kmd, where
     *  the aim is an ioctl and there is no readback to separate. */
    double aim_seconds;
    double aim_verify_seconds;

    /** THE WINDOW BANK, AND WHY EVERY ONE OF THESE IS ON THE REPORT LINE.
     *
     *  Each kind of window is a SET now, and the aim count above only falls if the set is
     *  actually absorbing the endpoint rotation. `hits`/`misses` is that mechanism measured
     *  directly rather than inferred from the aim count, and `evictions` separates "the set is
     *  too small" from "the endpoint had never been seen". `misaims_injected` is the red arm's
     *  own counter — an arm that never fired must read as a zero and not as a silence — and
     *  `audit_failures` must be zero on every arm that is not the red one. ZERO on tt-kmd. */
    uint32_t register_windows; /**< windows the UC bank actually claimed */
    uint32_t bulk_windows;     /**< windows the WC bank actually claimed */
    /* PER BANK, NOT POOLED. The two streams are different mechanisms and pooling
     * them hides the one that matters: the WC bank's weight upload walks
     * thousands of consecutive slices inside one 2 MiB window, so it reads ~99%
     * hit however badly the dispatch path is thrashing. */
    uint64_t register_hits;
    uint64_t register_misses;
    uint64_t register_evictions;
    uint64_t bulk_hits;
    uint64_t bulk_misses;
    uint64_t bulk_evictions;
    uint64_t misaims_injected;   /**< FT_TT_UDRV_RED=misaim's own count */
    uint64_t audit_failures;     /**< FT_TT_UDRV_AIM_AUDIT refusals; must be 0 unless armed */
    uint32_t register_endpoints; /**< distinct (x,y) seen; 0 without the histogram */
    uint32_t bulk_endpoints;
} ft_tt_runtime_window_stats;

/** Fill `out` from an open udrv handle. Zeroed for NULL. */
int ft_tt_runtime_udrv_window_stats_read(const ft_tt_runtime_udrv *udrv,
                                    ft_tt_runtime_window_stats *out);

#endif /* FT_TT_RUNTIME_UDRV_H */
