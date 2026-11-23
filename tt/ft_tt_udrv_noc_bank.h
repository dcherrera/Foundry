/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_noc_bank.h
 * @brief A SET of 2 MiB TLB windows driven as one accessor, so a dispatch path
 *   that talks to many cores stops re-pointing one window at each of them in
 *   turn.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS: THE AIM IS A PER-DESCRIPTOR COST, AND IT IS THE LARGEST ONE
 * ---------------------------------------------------------------------------
 * `ft_tt_udrv_noc_path` owns ONE window and already skips the re-aim when the
 * window is where the caller wants it (`aim_if_needed()`). That skip is a cache
 * of ONE ENTRY, and the dispatch path's access stream is not one endpoint — it
 * is a rotation over the resident cores, so the one entry misses on nearly
 * every access.
 *
 * Measured on silicon 2026-08-26 (spec, "the issue side is not accesses, it is
 * the TLB aim"): an aim is **8.92 us**, 98.0% of it four NON-POSTED register
 * readbacks at 2.19 us each, and the decode token pays **4,333 of them =
 * 0.0387 s = 25.1% of the wall**. All 25 PCIe accesses of a descriptor's issue
 * together cost 0.79 us. The cost is the aim, not the traffic.
 *
 * A bank is the same cache with N entries, N being the number of windows it
 * claimed. **202 windows exist on this part.** Two were being used.
 *
 * ---------------------------------------------------------------------------
 * THE CACHE IS THE WINDOWS THEMSELVES — THERE IS NO SECOND COPY OF THE STATE
 * ---------------------------------------------------------------------------
 * A lookup scans the paths' OWN `aim`/`aimed` fields, which is what
 * `ft_tt_udrv_noc_transfer()` maintains. There is deliberately no side table of
 * "what I think window k holds": a side table can disagree with the register,
 * and a cache that disagrees with the hardware aims at the wrong endpoint
 * SILENTLY. Anything that aims a path in this bank — including the ARC
 * transport, which is handed path 0 — updates the very field the lookup reads.
 *
 * ---------------------------------------------------------------------------
 * AND A WRONG WINDOW IS A WRONG ADDRESS, SO THERE IS AN AUDIT AND A RED ARM
 * ---------------------------------------------------------------------------
 * The failure this module can introduce is routing a transfer to a window
 * pointed at a DIFFERENT core. That does not fail; it reads and writes the
 * wrong endpoint. So:
 *
 *   - `FT_TT_UDRV_AIM_AUDIT=1` re-reads the selected window's register triple
 *     from the card and decodes it before a HIT is used. It is the same four
 *     non-posted reads the aim pays, which is why it is a check and not the
 *     shipped path — but it is an INDEPENDENT reading of where the window
 *     points, not the bank's own belief.
 *   - `FT_TT_UDRV_RED=misaim` falsifies a path's RECORDED aim so the re-aim is
 *     skipped and the read is dereferenced through a window pointed at another
 *     endpoint. It injects a STATE the whole transfer sees, not a race.
 *     **IT NEEDS A SECOND KEY AND IT IS NOT SAFE ON A LIVE MODEL** — it reads
 *     from whatever another window holds, which includes the ARC tile, and on
 *     2026-08-26 one such read did not complete, wedged the NoC and cost the
 *     host a physical power cycle. Drive it through
 *     `tests/test_udrv_card_bank.c`, which uses two DRAM banks that always
 *     decode and compares the bytes.
 *
 * @status On silicon 2026-08-26. See the plan item "THE NEXT RUNG IS THE AIM".
 * @issues
 *   - The audit reads back through the mapping that wrote the window, the same
 *     limit `ft_tt_udrv_noc.h` states for the aim's own verification: it cannot
 *     catch a mapping that consistently points at the wrong physical page.
 *   - A bank that could not claim every window it asked for runs with fewer
 *     cache entries. It says so loudly at open and reports the count; it never
 *     shares a window with another claimant.
 * @todo None
 */

#ifndef FT_TT_UDRV_NOC_BANK_H
#define FT_TT_UDRV_NOC_BANK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_udrv_claim.h"
#include "ft_tt_udrv_noc.h"

/** Lowest window index a bank will claim, and it is NOT an arbitrary margin.
 *
 *  WINDOWS BELOW `FT_TT_UDRV_TLB_STRIDED_COUNT` (32) CARRY A STRIDED-MULTICAST
 *  REGISTER, AND THAT MAKES THEM A DIFFERENT AND MORE EXPENSIVE KIND OF WINDOW.
 *  `ft_tt_udrv_noc_window_aim_timed()` guards both the strided CLEAR and the
 *  strided CHECK on `id < 32`, so on a window below the line every aim pays a
 *  fourth posted store and — at the shipped `full` verification — a FOURTH
 *  NON-POSTED READBACK, which is a whole PCIe round trip on the term this bank
 *  exists to remove. And a stale multicast pattern on one of those windows is
 *  this project's most reliable way to wedge the NoC, so it is also the one
 *  window kind whose aim has a fabric failure mode.
 *
 *  There is no shortage that would justify reaching below it: indices 32..201
 *  are 170 windows, 169 once the vendor driver's is skipped, against two banks
 *  whose combined ceiling is 168. **A bank that wants more says so and runs
 *  SHORT rather than quietly buying windows that cost more per aim than the aim
 *  they save.** */
#define FT_TT_UDRV_NOC_BANK_MIN_WINDOW_INDEX FT_TT_UDRV_TLB_STRIDED_COUNT

/** Most windows one bank will ever hold.
 *
 *  EIGHTY-FOUR, AND THE BOUND IS THE POOL ABOVE THE STRIDED LINE. Indices
 *  32..201 are 170; the vendor driver keeps 201; two banks at 84 claim 168 of
 *  the remaining 169. So both banks can be satisfied in full, every window is
 *  one of the cheap kind, and one is left over.
 *
 *  It was 32, and 32 was sized on a working set of 26 endpoints measured at
 *  EIGHT matmul members. The member count is a knob, the endpoint set grows
 *  with it, and at sixteen members the program touches more cores than a
 *  32-window bank can hold — which turns the bank from a cache into an eviction
 *  machine: the bulk hit rate falls 99.74% / 99.50% / 98.20% at N = 8 / 12 / 16
 *  while the aims a token go 829.2 / 1,939.9 / 7,566.1.
 *
 *  THIS IS A CEILING AND NOT A SIZE. What a bank actually claims is
 *  `FT_TT_UDRV_WINDOWS`, and its default is set from the reuse-distance
 *  measurement below, not from this number. Raising the ceiling costs nothing
 *  but `sizeof(ft_tt_udrv_noc_bank)`; claiming a window costs one file
 *  descriptor and one 2 MiB mapping of BAR0. */
#define FT_TT_UDRV_NOC_BANK_MAX_WINDOWS 84u

/** How deep the LRU reuse-distance instrument simulates.
 *
 *  A BANK IS AN LRU CACHE, SO ITS HIT RATE AT EVERY SIZE IS ONE MEASUREMENT.
 *  For LRU, a reference hits a cache of `W` entries exactly when its reuse
 *  distance — how many DISTINCT other keys were touched since this key last
 *  was — is below `W`. So a histogram of reuse distances gives the hit rate at
 *  every bank size at once, and the bank size stops being a guess.
 *
 *  Deeper than the window ceiling on purpose: a distance that lands past the
 *  ceiling has to be VISIBLE as "no bank this part can hold would catch it"
 *  rather than folded into the cold count. */
#define FT_TT_UDRV_NOC_BANK_REUSE_MAX 192u

/** Distinct (x, y) endpoints the histogram will name before it starts counting
 *  overflow. Sized well above the card's Tensix column count so a long tail is
 *  VISIBLE as overflow rather than silently truncated.
 *
 *  IT IS KEYED ON (x, y) AND NOT ON (x, y, base), WHICH IS A MEASURED
 *  CORRECTION. A (x, y, base) table is FLOODED by the weight upload: 9.5 GB of
 *  DRAM through 2 MiB windows is ~4,750 distinct bases across four endpoints,
 *  and since the upload happens before the first token it fills any table and
 *  pushes every L1 endpoint of the dispatch path into overflow. Keyed on the
 *  core, DRAM is four rows however many bases it walks.
 *
 *  IT WAS 64 AND 64 WAS NOT ENOUGH — measured 2026-09-03. "Sized well above the
 *  card's Tensix column count" is the wrong yardstick: the bulk bank's set is
 *  cores PLUS DRAM ports, and both grow with the matmul member count. At eight
 *  members it is 49 rows, at twelve 61, and at SIXTEEN it hit the 64-row ceiling
 *  and reported 7,219 selections past the table — so the one figure the round
 *  existed to produce read as ">= 64" instead of as a number. 160 is past the
 *  140 Tensix the part has, which is the bound the table can never need to
 *  exceed for cores, plus the DRAM and ARC rows. */
#define FT_TT_UDRV_NOC_BANK_HISTOGRAM_MAX 160u

/** Distinct 2 MiB bases one row can count.
 *
 *  PAST THIS THE ROW STOPS COUNTING AND SAYS SO. The first version kept
 *  incrementing once the list was full, which turns a DISTINCT-VALUE counter
 *  into a SELECTION counter the moment the set outgrows its list: a DRAM
 *  endpoint reported 21,790 "distinct bases" for 21,854 selections of a 32 GiB
 *  card. A cardinality that cannot be counted has to read as "at least N",
 *  never as a number. */
#define FT_TT_UDRV_NOC_BANK_BASES_PER_ROW 8u

/** One row of the endpoint histogram: which core, how often, and how much of
 *  its address space the window had to walk. */
typedef struct ft_tt_udrv_noc_bank_endpoint {
    uint16_t x;
    uint16_t y;
    uint64_t selections; /**< transfers routed to this core */
    uint64_t aims;       /**< of those, how many had to re-point a window */
    uint32_t base_count;  /**< distinct 2 MiB bases seen, up to the cap */
    bool base_uncounted;  /**< true once the set outgrew the list: read base_count as ">=" */
    uint64_t base[FT_TT_UDRV_NOC_BANK_BASES_PER_ROW];
} ft_tt_udrv_noc_bank_endpoint;

/** One routing key: the triple the window lookup actually compares.
 *
 *  IT IS (x, y, base) AND NOT (x, y). `path_holds()` compares the 2 MiB base
 *  as well as the core, so two bases on one core are two windows' worth of
 *  working set. The endpoint histogram above is keyed on the CORE — which is
 *  the right key for "how many cores does the dispatch path talk to" and the
 *  WRONG key for "how many windows would hold it". */
typedef struct ft_tt_udrv_noc_bank_key {
    uint16_t x;
    uint16_t y;
    uint64_t base;
} ft_tt_udrv_noc_bank_key;

/** Which deliberate defect is armed. Off unless FT_TT_UDRV_RED names it. */
typedef enum ft_tt_udrv_noc_bank_red {
    FT_TT_UDRV_NOC_BANK_RED_NONE = 0,
    FT_TT_UDRV_NOC_BANK_RED_MISAIM = 1
} ft_tt_udrv_noc_bank_red;

/**
 * A set of claimed windows, their apertures, and the routing over them.
 *
 * Caller-owned storage. Treat the fields as private except through the
 * accessors below; `path[0]` is public only because the ARC transport needs a
 * single path and must share this bank's window rather than claim a third.
 */
typedef struct ft_tt_udrv_noc_bank {
    ft_tt_udrv_noc_path path[FT_TT_UDRV_NOC_BANK_MAX_WINDOWS];
    ft_tt_udrv_claim *claim[FT_TT_UDRV_NOC_BANK_MAX_WINDOWS];
    uint64_t last_used[FT_TT_UDRV_NOC_BANK_MAX_WINDOWS]; /**< LRU stamp */
    uint32_t window_count;                               /**< paths actually open */
    uint32_t windows_requested;                          /**< what was asked for */
    uint64_t clock;                                      /**< LRU stamp source */
    const char *name;                                    /**< "register" / "bulk", for logs */

    uint64_t selections;       /**< transfers routed, one per window slice */
    uint64_t hits;             /**< routed to a window already pointed there */
    uint64_t misses;           /**< routed to a window that had to be re-aimed */
    uint64_t evictions;        /**< misses that displaced a live aim */
    uint64_t misaims_injected; /**< the red arm's own count; 0 means it never fired */
    uint64_t audit_failures;   /**< audits that caught a wrong endpoint */

    bool histogram_enabled;
    bool audit_enabled;
    bool reuse_enabled;
    ft_tt_udrv_noc_bank_red red;

    ft_tt_udrv_noc_bank_endpoint histogram[FT_TT_UDRV_NOC_BANK_HISTOGRAM_MAX];
    uint32_t histogram_used;
    uint64_t histogram_overflow; /**< selections past the table's capacity */

    /** The reuse-distance instrument. `reuse_stack` is an LRU stack of routing
     *  keys, MRU first; `reuse_hist[d]` counts references found at depth `d`,
     *  so a bank of `W` windows would have hit exactly `sum(reuse_hist[0..W))`
     *  of them. `reuse_cold` is everything else — first touches and distances
     *  past the simulated depth, which are the references no bank catches. */
    ft_tt_udrv_noc_bank_key reuse_stack[FT_TT_UDRV_NOC_BANK_REUSE_MAX];
    uint32_t reuse_depth;
    uint64_t reuse_hist[FT_TT_UDRV_NOC_BANK_REUSE_MAX];
    uint64_t reuse_cold;
    uint64_t reuse_references;
} ft_tt_udrv_noc_bank;

/**
 * Claim up to `windows` windows, map each one's aperture, and build the bank.
 *
 * Claims walk DOWN from the top of the pool, skipping `reserved_window`, for
 * the reason `ft_tt_runtime_udrv.c` documents: `ALLOCATE_TLB` allocates from
 * the bottom, and while tt-kmd is loaded a bottom-up claim here collides with
 * it on the first window every time. `reserved_window` is the index the vendor
 * driver keeps for itself (`KERNEL_TLB_INDEX`, 201) and is NEVER handed out —
 * claiming it out from under a loaded tt-kmd corrupts its telemetry reads.
 *
 * A short claim is NOT silently accepted as success: the achieved count is
 * logged at error level with the shortfall named, and the bank runs with fewer
 * cache entries. Zero windows is a hard failure.
 *
 * @param device          an open device
 * @param ns              the "tlb2m" claim namespace for this card
 * @param reserved_window index never to hand out
 * @param write_combine   true for the bulk path, false for registers
 * @param windows         how many to ask for, 1..FT_TT_UDRV_NOC_BANK_MAX_WINDOWS
 * @param name            "register" or "bulk"; used in log lines only
 * @param out_bank        zeroed and filled; close with ft_tt_udrv_noc_bank_close()
 * @return                FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL; FT_TT_UDRV_EAGAIN when
 *                        no window at all could be claimed; FT_TT_UDRV_EIO
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_noc_bank_open(ft_tt_udrv_noc_device *device,
                             const ft_tt_udrv_claim_namespace *ns, uint32_t reserved_window,
                             bool write_combine, uint32_t windows, const char *name,
                             ft_tt_udrv_noc_bank *out_bank);

/** Unmap every aperture and release every claim. Safe on a zeroed bank. */
int ft_tt_udrv_noc_bank_close(ft_tt_udrv_noc_bank *bank);

/**
 * Move `bytes` between host memory and (x, y, address) through whichever of the
 * bank's windows is already pointed there, aiming the least recently used one
 * when none is.
 *
 * Semantics are otherwise exactly `ft_tt_udrv_noc_transfer()`'s, which is what
 * this calls once the window is chosen — the copy shape, the fence and the
 * bounds are not re-derived here.
 *
 * @return FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL; FT_TT_UDRV_ENOTSUP; FT_TT_UDRV_EIO
 *         when an aim did not verify or the audit caught a wrong endpoint (in
 *         both cases nothing was dereferenced)
 */
int ft_tt_udrv_noc_bank_transfer(ft_tt_udrv_noc_bank *bank, uint16_t x, uint16_t y,
                                 uint64_t address, void *host, size_t bytes, bool is_read,
                                 ft_tt_noc_transfer_width width);

/** `ft_tt_noc_access`-shaped adapters whose `user` is an
 *  `ft_tt_udrv_noc_bank *`. WORD32, matching the single-path pair. */
int ft_tt_udrv_noc_bank_access_write(void *user, uint16_t x, uint16_t y, uint64_t address,
                                     const void *src, size_t bytes);
int ft_tt_udrv_noc_bank_access_read(void *user, uint16_t x, uint16_t y, uint64_t address,
                                    void *dst, size_t bytes);

/** Totals over every window in the bank. */
uint64_t ft_tt_udrv_noc_bank_aim_count(const ft_tt_udrv_noc_bank *bank);
uint64_t ft_tt_udrv_noc_bank_aim_failures(const ft_tt_udrv_noc_bank *bank);
double ft_tt_udrv_noc_bank_aim_seconds(const ft_tt_udrv_noc_bank *bank);
double ft_tt_udrv_noc_bank_aim_verify_seconds(const ft_tt_udrv_noc_bank *bank);
uint64_t ft_tt_udrv_noc_bank_bytes_read(const ft_tt_udrv_noc_bank *bank);
uint64_t ft_tt_udrv_noc_bank_bytes_written(const ft_tt_udrv_noc_bank *bank);

/** How many distinct (x, y) endpoints the histogram saw. Zero when the
 *  histogram was never enabled. */
uint32_t ft_tt_udrv_noc_bank_endpoint_count(const ft_tt_udrv_noc_bank *bank);

/**
 * Log the endpoint histogram at error level, rows sorted by selection count.
 *
 * A no-op unless the bank was opened with `FT_TT_UDRV_AIM_HISTOGRAM` set — the
 * recording costs a linear scan per transfer and has no business on the shipped
 * path. Prints the row count, the overflow count and the concentration, which
 * is the number that decides whether a window per endpoint is a cache with a
 * 100% hit rate or an eviction-policy problem.
 */
void ft_tt_udrv_noc_bank_histogram_log(const ft_tt_udrv_noc_bank *bank);

/**
 * Log the reuse-distance ladder: what hit rate a bank of each size WOULD have
 * had on the access stream this run actually produced.
 *
 * A no-op unless the bank was opened with `FT_TT_UDRV_AIM_REUSE` set. It costs
 * a stack search and a move-to-front per routing decision, so a run carrying it
 * is a run measuring the PATTERN and not the wall — its own timings are not
 * comparable to an arm without it, and the line says so.
 *
 * IT PRINTS ITS OWN FALSIFIER. The last column reprints the ladder's entry at
 * this bank's ACTUAL window count beside the hit rate the bank measured for
 * itself. Those two are the same number computed two ways — an LRU cache of `W`
 * entries hits exactly the references whose reuse distance is below `W` — so a
 * disagreement means the simulation does not describe the cache and the ladder
 * must not be used to size anything. An instrument that cannot be caught being
 * wrong is this program's most expensive defect.
 */
void ft_tt_udrv_noc_bank_reuse_log(const ft_tt_udrv_noc_bank *bank);

/** References this bank's reuse instrument would have served from a bank of
 *  `windows` entries. Zero when the instrument was never enabled. */
uint64_t ft_tt_udrv_noc_bank_reuse_hits_at(const ft_tt_udrv_noc_bank *bank, uint32_t windows);

/** Parse `FT_TT_UDRV_WINDOWS` once. Clamped to
 *  1..FT_TT_UDRV_NOC_BANK_MAX_WINDOWS; an unparseable value is refused loudly
 *  and the default is used. Exposed so the caller can echo what it parsed —
 *  a knob that reads as armed and is not is this project's most expensive
 *  instrument defect. */
uint32_t ft_tt_udrv_noc_bank_windows_selected(void);

#endif /* FT_TT_UDRV_NOC_BANK_H */
