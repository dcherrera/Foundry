/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_noc_bank.c
 * @brief Implementation of the multi-window accessor (see the header for the
 *   whole argument, including the audit and the mis-aim red arm).
 *
 *   Everything below the window CHOICE is unchanged: once a window is chosen
 *   this calls `ft_tt_udrv_noc_transfer()` on it, so the slice arithmetic, the
 *   copy shape, the store fence and the aim verification are the ones already
 *   on silicon. This file only decides WHICH window.
 *
 * ---------------------------------------------------------------------------
 * WHY THE LOOKUP COMPARES THREE INTEGERS AND NOT THE WHOLE CONFIG
 * ---------------------------------------------------------------------------
 * An aim is `template + endpoint`. The bank stamps EVERY path with the same
 * template at open and nothing else in the tree changes a bank path's template,
 * so two paths' aims can differ only in the endpoint triple. Comparing
 * (x_end, y_end, addr) is therefore exactly equivalent to comparing the full
 * config, at a fraction of the cost — and the cost matters, because this scan
 * runs on every window slice of every descriptor.
 *
 * ---------------------------------------------------------------------------
 * THIS LAYER RETURNS CODES; ITS INSTRUMENTS PRINT
 * ---------------------------------------------------------------------------
 * The rest of the userspace driver logs nothing and leaves the message to its
 * caller, which is why it has no dependency on `ft_tt_device.h`. That is kept.
 * The two things that DO print here are instruments, not error paths: the
 * histogram dump, and the audit's refusal — which has to name the two endpoints
 * it disagrees about at the moment it sees them, because by the time a code has
 * travelled up through three layers the coordinates are gone.
 *
 * @status On silicon 2026-08-26.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_udrv_noc_bank.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_udrv.h"

/** Windows a bank asks for when `FT_TT_UDRV_WINDOWS` says nothing.
 *
 *  THIRTY-TWO, ON A MEASURED WORKING SET — AND THE WORKING SET IS A FUNCTION OF
 *  A KNOB. The dispatch path aims at exactly 26 distinct (x, y) on both banks at
 *  EIGHT matmul members, and the ladder saturates there: aims a token run
 *  4,329.5 / 3,449.1 / 1,220.7 / 262.2 / 130.8 / 111.8 at N = 1 / 8 / 16 / 24 /
 *  28 / 32, a cliff exactly where the bank grows past the working set, because
 *  LRU over a CYCLIC access pattern is worst-case until it fits.
 *
 *  **THE MEMBER COUNT MOVES THE SET, AND THE 32 WAS SIZED AT ONE POINT OF IT.**
 *  Raising the matmul group to twelve and sixteen members adds cores the
 *  dispatch path talks to every descriptor, and the bulk bank's hit rate falls
 *  99.74% / 99.50% / 98.20% at N = 8 / 12 / 16 while the aims a token go
 *  829.2 / 1,939.9 / 7,566.1. A cache that is one entry short of a cyclic
 *  working set misses on nearly every reference, which is why the last step is
 *  4x and not 20%.
 *
 *  **SIZE IT FROM `FT_TT_UDRV_AIM_REUSE`, NOT FROM A CORE COUNT.** The routing
 *  key is (x, y, base), not (x, y): one core reached at two 2 MiB bases is two
 *  entries. The reuse-distance ladder that instrument prints is the hit rate
 *  this bank would have had at every size, measured on the stream the run
 *  actually produced, and it is what this default is set from.
 *
 *  **`FT_TT_UDRV_WINDOWS=1` is the control**: one window per kind is the
 *  single-window transport this replaced, in the same binary with the same
 *  counters, which is why the ladder is a ladder and not a comparison of two
 *  builds. */
/*  SIXTY-FOUR SINCE 2026-10, flipped with the 16-member matmul placement: at
 *  N = 16 the working set is 41 register / 65 bulk keys and the knee is 40; 64
 *  is the knee plus margin and overflows only at N ~ 40. */
#define BANK_WINDOWS_DEFAULT 64u

/* ------------------------------------------------------------------------- */
/* Knobs                                                                     */
/* ------------------------------------------------------------------------- */

uint32_t ft_tt_udrv_noc_bank_windows_selected(void)
{
    static long cached = -1;

    if (cached < 0) {
        const char *value = getenv("FT_TT_UDRV_WINDOWS");
        char *end = NULL;
        long parsed;

        cached = (long)BANK_WINDOWS_DEFAULT;
        if (value && *value) {
            parsed = strtol(value, &end, 10);
            if (end && *end == '\0' && parsed >= 1 &&
                parsed <= (long)FT_TT_UDRV_NOC_BANK_MAX_WINDOWS)
                cached = parsed;
            else
                fprintf(stderr,
                        "ft_tt_udrv_noc_bank: FT_TT_UDRV_WINDOWS=%s is not 1..%u — using %u\n",
                        value, FT_TT_UDRV_NOC_BANK_MAX_WINDOWS, BANK_WINDOWS_DEFAULT);
        }
    }
    return (uint32_t)cached;
}

/** Whether an environment flag is set to something other than "0". */
static bool env_flag(const char *name)
{
    const char *value = getenv(name);

    return value && *value && strcmp(value, "0") != 0;
}

/** The second key `misaim` needs. See env_red(). */
#define BANK_RED_CONFIRM "i-know-this-can-wedge-the-card"

/**
 * Which deliberate defect is armed.
 *
 * ---------------------------------------------------------------------------
 * `misaim` TAKES TWO KEYS, AND THAT IS NOT CEREMONY — IT COST A CARD
 * ---------------------------------------------------------------------------
 * The arm sends a READ to whatever endpoint another window happens to hold, and
 * on a live model those windows include the ARC tile at (8, 0), whose address
 * map is not a Tensix's. **On 2026-08-26 a read aimed there did not complete,
 * the NoC wedged, the next process could not bring the ARC up, and the host
 * went off the network and needed a physical power cycle.** "Reads only" makes
 * the arm safe for OTHER PEOPLE'S DATA; it does not make it safe for the fabric.
 *
 * So: arm it through `tests/test_udrv_card_bank.c`, which drives two DRAM banks
 * that always decode and compares the bytes, and NOT against a live model. The
 * confirmation string exists so nobody reaches this by adding one word to an
 * env line.
 */
static ft_tt_udrv_noc_bank_red env_red(void)
{
    const char *value = getenv("FT_TT_UDRV_RED");
    const char *confirm;

    if (!value || !*value)
        return FT_TT_UDRV_NOC_BANK_RED_NONE;
    if (strcmp(value, "misaim") != 0) {
        fprintf(stderr, "ft_tt_udrv_noc_bank: FT_TT_UDRV_RED=%s names no arm here — none armed\n",
                value);
        return FT_TT_UDRV_NOC_BANK_RED_NONE;
    }
    confirm = getenv("FT_TT_UDRV_RED_CONFIRM");
    if (!confirm || strcmp(confirm, BANK_RED_CONFIRM) != 0) {
        fprintf(stderr,
                "ft_tt_udrv_noc_bank: FT_TT_UDRV_RED=misaim NOT ARMED — it also needs "
                "FT_TT_UDRV_RED_CONFIRM=%s. It reads from whatever endpoint another window "
                "holds, which on a live model includes the ARC tile; one such read did not "
                "complete on 2026-08-26 and the box needed a power cycle. Use "
                "tests/test_udrv_card_bank.c instead\n",
                BANK_RED_CONFIRM);
        return FT_TT_UDRV_NOC_BANK_RED_NONE;
    }
    fprintf(stderr,
            "ft_tt_udrv_noc_bank: FT_TT_UDRV_RED=misaim IS ARMED. Every read that hits may be "
            "served by a window pointed at another endpoint. THIS CAN WEDGE THE NoC\n");
    return FT_TT_UDRV_NOC_BANK_RED_MISAIM;
}

/* ------------------------------------------------------------------------- */
/* Open and close                                                            */
/* ------------------------------------------------------------------------- */

/**
 * Claim one window, walking DOWN from `*cursor` and skipping the index another
 * driver owns. Same policy and same reasoning as the single-window claim in
 * ft_tt_runtime_udrv.c, kept here so a bank is self-contained.
 *
 * THE CURSOR IS WHAT MAKES A LARGE BANK CHEAP TO OPEN. Without it every claim
 * restarts at the top of the pool and re-tries — open, flock, fail, close — on
 * every window THIS BANK ALREADY HOLDS, so opening W windows costs O(W^2)
 * syscalls and a second bank pays the first bank's windows too. Two banks of 96
 * is ~18,000 futile open/flock pairs; with the cursor it is one attempt per
 * index. Passing a window we skipped can only mean losing one that a DIFFERENT
 * process released mid-open, which costs a cache entry and never correctness.
 *
 * The walk STOPS at `FT_TT_UDRV_NOC_BANK_MIN_WINDOW_INDEX` — see the header for
 * why a window below the strided line is a more EXPENSIVE window and not merely
 * a lower-numbered one, and why running SHORT is the right answer to wanting
 * one.
 */
static int bank_claim_one(const ft_tt_udrv_claim_namespace *ns, uint32_t reserved_window,
                          const char *reason, uint32_t *cursor, ft_tt_udrv_claim **out_claim)
{
    int last_other = FT_TT_UDRV_OK;

    while (*cursor > FT_TT_UDRV_NOC_BANK_MIN_WINDOW_INDEX) {
        uint32_t index;
        int rc;

        (*cursor)--;
        index = *cursor;
        if (index == reserved_window)
            continue;
        rc = ft_tt_udrv_claim_acquire(ns, index, reason, out_claim);
        if (rc == FT_TT_UDRV_OK)
            return FT_TT_UDRV_OK;
        if (rc != FT_TT_UDRV_EAGAIN)
            last_other = rc;
    }
    return last_other != FT_TT_UDRV_OK ? last_other : FT_TT_UDRV_EAGAIN;
}

int ft_tt_udrv_noc_bank_open(ft_tt_udrv_noc_device *device,
                             const ft_tt_udrv_claim_namespace *ns, uint32_t reserved_window,
                             bool write_combine, uint32_t windows, const char *name,
                             ft_tt_udrv_noc_bank *out_bank)
{
    uint32_t cursor;
    uint32_t opened;

    if (!device || !device->registers || !ns || !out_bank || !name)
        return FT_TT_UDRV_EINVAL;
    if (windows == 0u || windows > FT_TT_UDRV_NOC_BANK_MAX_WINDOWS)
        return FT_TT_UDRV_EINVAL;

    memset(out_bank, 0, sizeof(*out_bank));
    out_bank->name = name;
    out_bank->windows_requested = windows;
    out_bank->histogram_enabled = env_flag("FT_TT_UDRV_AIM_HISTOGRAM");
    out_bank->audit_enabled = env_flag("FT_TT_UDRV_AIM_AUDIT");
    out_bank->reuse_enabled = env_flag("FT_TT_UDRV_AIM_REUSE");
    out_bank->red = env_red();

    cursor = ns->count;
    for (opened = 0u; opened < windows; opened++) {
        uint32_t id;
        int rc = bank_claim_one(ns, reserved_window, name, &cursor, &out_bank->claim[opened]);

        if (rc != FT_TT_UDRV_OK)
            break;
        id = ft_tt_udrv_claim_index(out_bank->claim[opened]);
        rc = ft_tt_udrv_noc_path_open(device, id, write_combine, &out_bank->path[opened]);
        if (rc != FT_TT_UDRV_OK) {
            (void)ft_tt_udrv_claim_release(out_bank->claim[opened]);
            out_bank->claim[opened] = NULL;
            break;
        }
        /* THE TEMPLATE IS THE BANK'S INVARIANT, NOT A DEFAULT IT INHERITS. The
         * three-integer lookup below is only equivalent to a full config compare
         * because every path here carries the same non-endpoint fields. Stamped
         * explicitly so the invariant is code rather than an assumption about
         * what path_open() memset. */
        (void)ft_tt_udrv_noc_path_set_aim_template(&out_bank->path[opened], 0u, 0u, 0u, 0u, 0u,
                                                   0u);
        out_bank->window_count = opened + 1u;
    }

    if (out_bank->window_count == 0u) {
        fprintf(stderr,
                "ft_tt_udrv_noc_bank: %s bank could claim NO window of %u — refusing\n", name,
                ns->count);
        return FT_TT_UDRV_EAGAIN;
    }
    if (out_bank->window_count < windows)
        fprintf(stderr,
                "ft_tt_udrv_noc_bank: %s bank asked for %u windows and got %u — running SHORT. "
                "Every aim that misses this smaller set costs 8.9 us; nothing is shared. ANY "
                "FIGURE SCORED AGAINST %u IS VOID — the bank that ran is %u\n",
                name, windows, out_bank->window_count, windows, out_bank->window_count);
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_noc_bank_close(ft_tt_udrv_noc_bank *bank)
{
    uint32_t i;
    int rc = FT_TT_UDRV_OK;

    if (!bank)
        return FT_TT_UDRV_EINVAL;
    for (i = 0u; i < bank->window_count; i++) {
        int one = ft_tt_udrv_noc_path_close(&bank->path[i]);

        if (one != FT_TT_UDRV_OK)
            rc = one;
    }
    for (i = 0u; i < FT_TT_UDRV_NOC_BANK_MAX_WINDOWS; i++) {
        if (bank->claim[i])
            (void)ft_tt_udrv_claim_release(bank->claim[i]);
        bank->claim[i] = NULL;
    }
    memset(bank, 0, sizeof(*bank));
    return rc;
}

/* ------------------------------------------------------------------------- */
/* The histogram                                                             */
/* ------------------------------------------------------------------------- */

static void histogram_record(ft_tt_udrv_noc_bank *bank, uint16_t x, uint16_t y, uint64_t base,
                             bool was_hit)
{
    ft_tt_udrv_noc_bank_endpoint *row = NULL;
    uint32_t i;

    for (i = 0u; i < bank->histogram_used; i++)
        if (bank->histogram[i].x == x && bank->histogram[i].y == y) {
            row = &bank->histogram[i];
            break;
        }
    if (!row) {
        if (bank->histogram_used >= FT_TT_UDRV_NOC_BANK_HISTOGRAM_MAX) {
            bank->histogram_overflow++;
            return;
        }
        row = &bank->histogram[bank->histogram_used++];
        row->x = x;
        row->y = y;
    }
    row->selections++;
    if (!was_hit)
        row->aims++;
    if (row->base_uncounted)
        return;
    for (i = 0u; i < row->base_count; i++)
        if (row->base[i] == base)
            return;
    /* THE LIST IS THE COUNTER. Incrementing past it would count SELECTIONS and
     * print them as a cardinality, which is exactly the defect this replaced. */
    if (row->base_count >= FT_TT_UDRV_NOC_BANK_BASES_PER_ROW) {
        row->base_uncounted = true;
        return;
    }
    row->base[row->base_count++] = base;
}

uint32_t ft_tt_udrv_noc_bank_endpoint_count(const ft_tt_udrv_noc_bank *bank)
{
    return bank ? bank->histogram_used : 0u;
}

void ft_tt_udrv_noc_bank_histogram_log(const ft_tt_udrv_noc_bank *bank)
{
    uint32_t order[FT_TT_UDRV_NOC_BANK_HISTOGRAM_MAX];
    uint64_t total = 0u;
    uint64_t running = 0u;
    uint32_t i;

    if (!bank || !bank->histogram_enabled)
        return;

    for (i = 0u; i < bank->histogram_used; i++) {
        order[i] = i;
        total += bank->histogram[i].selections;
    }
    for (i = 0u; i + 1u < bank->histogram_used; i++) {
        uint32_t j;

        for (j = i + 1u; j < bank->histogram_used; j++)
            if (bank->histogram[order[j]].selections > bank->histogram[order[i]].selections) {
                uint32_t swap = order[i];

                order[i] = order[j];
                order[j] = swap;
            }
    }
    fprintf(stderr,
            "ft_tt_udrv_noc_bank: %s HISTOGRAM — %u distinct (x,y), %llu selections, %llu past "
            "the table, %u windows\n",
            bank->name, bank->histogram_used, (unsigned long long)total,
            (unsigned long long)bank->histogram_overflow, bank->window_count);
    for (i = 0u; i < bank->histogram_used; i++) {
        const ft_tt_udrv_noc_bank_endpoint *row = &bank->histogram[order[i]];

        running += row->selections;
        fprintf(stderr,
                "ft_tt_udrv_noc_bank: %s   #%-3u (%2u,%2u) — %10llu selections (%5.2f%%, "
                "cumulative %6.2f%%), %llu aims, %s%u bases from 0x%09llx\n",
                bank->name, i, row->x, row->y, (unsigned long long)row->selections,
                total ? 100.0 * (double)row->selections / (double)total : 0.0,
                total ? 100.0 * (double)running / (double)total : 0.0,
                (unsigned long long)row->aims, row->base_uncounted ? ">=" : "",
                row->base_count, (unsigned long long)row->base[0]);
    }
}

/* ------------------------------------------------------------------------- */
/* The reuse-distance ladder — what EVERY bank size would have done           */
/* ------------------------------------------------------------------------- */

/**
 * Record one routing decision against an LRU stack of routing keys.
 *
 * ---------------------------------------------------------------------------
 * WHY A REUSE DISTANCE IS THE WHOLE LADDER
 * ---------------------------------------------------------------------------
 * An LRU cache of `W` entries holds exactly the `W` most recently used distinct
 * keys. So a reference hits it if and only if fewer than `W` DISTINCT other keys
 * were touched since this key last was — its reuse distance. Bucket every
 * reference by that distance once and the hit rate at every size falls out of
 * one run, instead of one run per rung. The recorded window ladder cost six
 * model loads to produce six points; this produces every point up to
 * `FT_TT_UDRV_NOC_BANK_REUSE_MAX` from one.
 *
 * ---------------------------------------------------------------------------
 * THE KEY IS THE ROUTING KEY, WHICH IS NOT THE ENDPOINT
 * ---------------------------------------------------------------------------
 * `path_holds()` compares (x, y, base). The endpoint histogram beside this is
 * keyed on (x, y) and deliberately so — keyed on the triple it is FLOODED by
 * the weight upload's thousands of DRAM bases. Both keys are right for their own
 * question and neither answers the other's: a core reached at two bases is ONE
 * endpoint and TWO windows.
 *
 * IT IS O(depth) A REFERENCE. That is why it is off by default and why a run
 * carrying it is measuring the pattern, not the wall.
 */
static void reuse_record(ft_tt_udrv_noc_bank *bank, uint16_t x, uint16_t y, uint64_t base)
{
    uint32_t found = FT_TT_UDRV_NOC_BANK_REUSE_MAX;
    uint32_t i;

    bank->reuse_references++;
    for (i = 0u; i < bank->reuse_depth; i++)
        if (bank->reuse_stack[i].x == x && bank->reuse_stack[i].y == y &&
            bank->reuse_stack[i].base == base) {
            found = i;
            break;
        }

    if (found < FT_TT_UDRV_NOC_BANK_REUSE_MAX)
        bank->reuse_hist[found]++;
    else
        bank->reuse_cold++;

    /* Move to front. A miss pushes and drops the tail, which is exactly what an
     * LRU cache of REUSE_MAX entries does; anything that fell off the tail is
     * already counted as cold and cannot be double counted. */
    if (found == FT_TT_UDRV_NOC_BANK_REUSE_MAX) {
        if (bank->reuse_depth < FT_TT_UDRV_NOC_BANK_REUSE_MAX)
            bank->reuse_depth++;
        found = bank->reuse_depth - 1u;
    }
    for (i = found; i > 0u; i--)
        bank->reuse_stack[i] = bank->reuse_stack[i - 1u];
    bank->reuse_stack[0].x = x;
    bank->reuse_stack[0].y = y;
    bank->reuse_stack[0].base = base;
}

uint64_t ft_tt_udrv_noc_bank_reuse_hits_at(const ft_tt_udrv_noc_bank *bank, uint32_t windows)
{
    uint64_t total = 0u;
    uint32_t i;

    if (!bank || !bank->reuse_enabled)
        return 0u;
    if (windows > FT_TT_UDRV_NOC_BANK_REUSE_MAX)
        windows = FT_TT_UDRV_NOC_BANK_REUSE_MAX;
    for (i = 0u; i < windows; i++)
        total += bank->reuse_hist[i];
    return total;
}

void ft_tt_udrv_noc_bank_reuse_log(const ft_tt_udrv_noc_bank *bank)
{
    static const uint32_t rungs[] = {1u,  2u,  4u,  8u,  12u, 16u, 20u, 24u, 28u,
                                     32u, 40u, 48u, 56u, 64u, 80u, 96u, 128u, 192u};
    uint64_t predicted;
    uint64_t measured;
    double predicted_rate;
    double measured_rate;
    size_t r;

    if (!bank || !bank->reuse_enabled)
        return;
    if (bank->reuse_references == 0u) {
        fprintf(stderr,
                "ft_tt_udrv_noc_bank: %s REUSE — the instrument was ARMED and saw ZERO "
                "references. It measured nothing; do not read a ladder off it\n",
                bank->name);
        return;
    }

    fprintf(stderr,
            "ft_tt_udrv_noc_bank: %s REUSE — %llu routing decisions over (x,y,base) keys, "
            "%llu of them cold or past a %u-deep stack. THIS RUN'S WALL IS NOT COMPARABLE: the "
            "instrument costs a stack search per decision\n",
            bank->name, (unsigned long long)bank->reuse_references,
            (unsigned long long)bank->reuse_cold, FT_TT_UDRV_NOC_BANK_REUSE_MAX);
    for (r = 0u; r < sizeof(rungs) / sizeof(rungs[0]); r++) {
        uint64_t hits = ft_tt_udrv_noc_bank_reuse_hits_at(bank, rungs[r]);

        fprintf(stderr,
                "ft_tt_udrv_noc_bank: %s   windows %3u -> %10llu hits, %10llu aims, %7.4f%% hit\n",
                bank->name, rungs[r], (unsigned long long)hits,
                (unsigned long long)(bank->reuse_references - hits),
                100.0 * (double)hits / (double)bank->reuse_references);
    }

    /* THE FALSIFIER, PRINTED BESIDE THE LADDER. The ladder's entry at the bank's
     * OWN size and the bank's OWN hit counter are the same quantity computed two
     * ways. They agree or the simulation does not describe this cache. */
    predicted = ft_tt_udrv_noc_bank_reuse_hits_at(bank, bank->window_count);
    measured = bank->hits;
    predicted_rate = 100.0 * (double)predicted / (double)bank->reuse_references;
    measured_rate = bank->selections
                        ? 100.0 * (double)measured / (double)bank->selections
                        : 0.0;
    fprintf(stderr,
            "ft_tt_udrv_noc_bank: %s   SELF-CHECK at the live size %u — ladder says %llu hits "
            "of %llu (%.4f%%), the bank counted %llu of %llu (%.4f%%). %s\n",
            bank->name, bank->window_count, (unsigned long long)predicted,
            (unsigned long long)bank->reuse_references, predicted_rate,
            (unsigned long long)measured, (unsigned long long)bank->selections, measured_rate,
            (predicted == measured && bank->reuse_references == bank->selections)
                ? "IDENTICAL — the ladder describes this cache"
                : "THEY DISAGREE — THE LADDER IS VOID, do not size a bank from it");
}

/* ------------------------------------------------------------------------- */
/* Routing                                                                   */
/* ------------------------------------------------------------------------- */

/** True when this path's window is already pointed at (x, y, base). See the
 *  file header for why three integers are the whole comparison. */
static bool path_holds(const ft_tt_udrv_noc_path *path, uint16_t x, uint16_t y, uint64_t base)
{
    return path->aimed && path->aim.x_end == (uint8_t)x && path->aim.y_end == (uint8_t)y &&
           path->aim.addr == base;
}

/** The window to displace on a miss: an unused one if there is one, otherwise
 *  the least recently used. */
static uint32_t bank_choose_victim(const ft_tt_udrv_noc_bank *bank)
{
    uint32_t best = 0u;
    uint32_t i;

    for (i = 0u; i < bank->window_count; i++)
        if (!bank->path[i].aimed)
            return i;
    for (i = 1u; i < bank->window_count; i++)
        if (bank->last_used[i] < bank->last_used[best])
            best = i;
    return best;
}

/**
 * Deliberately make one window LIE about where it points.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS FALSIFIES THE PATH'S BELIEF AND NOT JUST THE ROUTING
 * ---------------------------------------------------------------------------
 * The first version of this arm simply returned a window aimed somewhere else,
 * and it was VACUOUS — measured on silicon 2026-08-26, 176,406 "injections" and
 * a completion byte-identical to its own control. The reason is that
 * `ft_tt_udrv_noc_transfer()` calls `aim_if_needed()`, which compares the path's
 * recorded aim against the endpoint it was given and RE-AIMS. The arm handed the
 * transfer a wrong window and the transfer quietly corrected it; all that moved
 * was the cache statistics.
 *
 * **The real failure mode of a bank is a path whose RECORDED aim disagrees with
 * its register**, because that is the one state in which `aim_if_needed()` skips
 * and the aperture is dereferenced while pointed at another core. So the arm
 * writes the wanted endpoint into `path->aim` WITHOUT touching the hardware:
 * the belief now says (x, y, base), the register still holds the old endpoint,
 * the re-aim is skipped, and the read lands on the wrong core. That is a STATE
 * the whole transfer sees, not a race it might.
 *
 * READS ONLY, and the belief is restored the moment the read is done
 * (`ft_tt_udrv_noc_bank_transfer()` invalidates the path). A poisoned belief
 * left in place would eventually carry a WRITE into a live core's L1 and corrupt
 * a resident kernel, and a red arm on a card nobody can power-cycle has to be
 * survivable.
 *
 * @return the index to use instead, or `chosen` when no other window is aimed
 *   anywhere different — in which case nothing was injected and the counter does
 *   not move, so an arm that could not fire reads as a zero.
 */
static uint32_t bank_inject_misaim(ft_tt_udrv_noc_bank *bank, uint32_t chosen, uint16_t x,
                                   uint16_t y, uint64_t base)
{
    uint32_t i;

    for (i = 0u; i < bank->window_count; i++) {
        if (i == chosen || !bank->path[i].aimed)
            continue;
        if (path_holds(&bank->path[i], x, y, base))
            continue;
        bank->path[i].aim.x_end = (uint8_t)x;
        bank->path[i].aim.y_end = (uint8_t)y;
        bank->path[i].aim.addr = base;
        bank->misaims_injected++;
        return i;
    }
    return chosen;
}

/**
 * Read the chosen window's register triple back from the card and refuse if it
 * does not decode to the endpoint this transfer asked for.
 *
 * This is four non-posted reads — the very cost the bank exists to remove — so
 * it is an audit and never the shipped path. What it buys is an INDEPENDENT
 * reading: the routing above believes `path->aim`, and this believes the
 * register.
 */
static int bank_audit(ft_tt_udrv_noc_bank *bank, uint32_t index, uint16_t x, uint16_t y,
                      uint64_t base)
{
    ft_tt_udrv_tlb2m_config held;
    uint32_t words[3] = {0u, 0u, 0u};
    int rc = ft_tt_udrv_noc_window_read(bank->path[index].device, bank->path[index].window_id,
                                        words);

    if (rc != FT_TT_UDRV_OK)
        return rc;
    rc = ft_tt_udrv_tlb2m_decode(words, &held);
    if (rc != FT_TT_UDRV_OK)
        return rc;
    if (held.x_end == (uint8_t)x && held.y_end == (uint8_t)y && held.addr == base)
        return FT_TT_UDRV_OK;

    bank->audit_failures++;
    fprintf(stderr,
            "ft_tt_udrv_noc_bank: %s AUDIT REFUSED — window %u holds (%u,%u) base 0x%09llx and "
            "this transfer wants (%u,%u) base 0x%09llx. A wrong window is a wrong ADDRESS, so "
            "nothing was dereferenced\n",
            bank->name, bank->path[index].window_id, held.x_end, held.y_end,
            (unsigned long long)held.addr, x, y, (unsigned long long)base);
    return FT_TT_UDRV_EIO;
}

/**
 * Pick the window for (x, y, base), recording the hit or miss.
 *
 * On a miss the victim is returned WITHOUT being aimed — the aim happens inside
 * `ft_tt_udrv_noc_transfer()`, which is the one place in the tree that aims and
 * verifies, and duplicating it here is how the two would drift.
 */
static int bank_select(ft_tt_udrv_noc_bank *bank, uint16_t x, uint16_t y, uint64_t base,
                       bool is_read, uint32_t *out_index, bool *out_poisoned)
{
    uint32_t index = bank->window_count;
    bool hit = false;
    uint32_t i;

    *out_poisoned = false;

    for (i = 0u; i < bank->window_count; i++)
        if (path_holds(&bank->path[i], x, y, base)) {
            index = i;
            hit = true;
            break;
        }
    if (!hit) {
        index = bank_choose_victim(bank);
        if (bank->path[index].aimed)
            bank->evictions++;
        bank->misses++;
    } else {
        bank->hits++;
    }
    bank->selections++;
    bank->last_used[index] = ++bank->clock;
    if (bank->histogram_enabled)
        histogram_record(bank, x, y, base, hit);
    if (bank->reuse_enabled)
        reuse_record(bank, x, y, base);
    if (hit && bank->red == FT_TT_UDRV_NOC_BANK_RED_MISAIM && is_read) {
        uint32_t poisoned = bank_inject_misaim(bank, index, x, y, base);

        *out_poisoned = poisoned != index;
        index = poisoned;
    }
    if (hit && bank->audit_enabled) {
        int rc = bank_audit(bank, index, x, y, base);

        if (rc != FT_TT_UDRV_OK) {
            /* The transfer will not run, so nothing downstream restores a
             * poisoned belief. Undo it here or the arm outlives its own read. */
            if (*out_poisoned)
                bank->path[index].aimed = false;
            return rc;
        }
    }
    *out_index = index;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_noc_bank_transfer(ft_tt_udrv_noc_bank *bank, uint16_t x, uint16_t y,
                                 uint64_t address, void *host, size_t bytes, bool is_read,
                                 ft_tt_noc_transfer_width width)
{
    unsigned char *host_bytes = (unsigned char *)host;

    if (!bank || bank->window_count == 0u)
        return FT_TT_UDRV_EINVAL;
    if (!host && bytes != 0u)
        return FT_TT_UDRV_EINVAL;

    while (bytes > 0u) {
        ft_tt_noc_window_slice slice;
        uint32_t index = 0u;
        bool poisoned = false;
        int rc;

        if (!ft_tt_noc_window_slice_compute(address, bytes, bank->path[0].window_bytes, &slice))
            return FT_TT_UDRV_EINVAL;
        rc = bank_select(bank, x, y, slice.window_base, is_read, &index, &poisoned);
        if (rc != FT_TT_UDRV_OK)
            return rc;
        /* The chosen path re-derives this same slice and finds its window either
         * already there (a hit costs no aim) or in need of one. Everything about
         * the move itself is that function's, unchanged. */
        rc = ft_tt_udrv_noc_transfer(&bank->path[index], x, y, address, host_bytes, slice.bytes,
                                     is_read, width);
        /* A red arm's poisoned belief lives for exactly one read. Left in place
         * it would eventually carry a WRITE to the wrong core. */
        if (poisoned)
            bank->path[index].aimed = false;
        if (rc != FT_TT_UDRV_OK)
            return rc;

        address += slice.bytes;
        host_bytes += slice.bytes;
        bytes -= slice.bytes;
    }
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_noc_bank_access_write(void *user, uint16_t x, uint16_t y, uint64_t address,
                                     const void *src, size_t bytes)
{
    /* Const is dropped so one walk serves both directions; with is_read false
     * nothing is ever written through this pointer. */
    return ft_tt_udrv_noc_bank_transfer((ft_tt_udrv_noc_bank *)user, x, y, address,
                                        (void *)(uintptr_t)src, bytes, false,
                                        FT_TT_NOC_WIDTH_WORD32);
}

int ft_tt_udrv_noc_bank_access_read(void *user, uint16_t x, uint16_t y, uint64_t address,
                                    void *dst, size_t bytes)
{
    return ft_tt_udrv_noc_bank_transfer((ft_tt_udrv_noc_bank *)user, x, y, address, dst, bytes,
                                        true, FT_TT_NOC_WIDTH_WORD32);
}

/* ------------------------------------------------------------------------- */
/* Totals                                                                    */
/* ------------------------------------------------------------------------- */

#define BANK_SUM(field, type)                                                                  \
    type total = (type)0;                                                                      \
    uint32_t i;                                                                                \
    if (!bank)                                                                                 \
        return (type)0;                                                                        \
    for (i = 0u; i < bank->window_count; i++)                                                  \
        total += field;                                                                        \
    return total

uint64_t ft_tt_udrv_noc_bank_aim_count(const ft_tt_udrv_noc_bank *bank)
{
    BANK_SUM(ft_tt_udrv_noc_path_aim_count(&bank->path[i]), uint64_t);
}

uint64_t ft_tt_udrv_noc_bank_aim_failures(const ft_tt_udrv_noc_bank *bank)
{
    BANK_SUM(ft_tt_udrv_noc_path_aim_failures(&bank->path[i]), uint64_t);
}

double ft_tt_udrv_noc_bank_aim_seconds(const ft_tt_udrv_noc_bank *bank)
{
    BANK_SUM(ft_tt_udrv_noc_path_aim_seconds(&bank->path[i]), double);
}

double ft_tt_udrv_noc_bank_aim_verify_seconds(const ft_tt_udrv_noc_bank *bank)
{
    BANK_SUM(ft_tt_udrv_noc_path_aim_verify_seconds(&bank->path[i]), double);
}

uint64_t ft_tt_udrv_noc_bank_bytes_read(const ft_tt_udrv_noc_bank *bank)
{
    BANK_SUM(bank->path[i].bytes_read, uint64_t);
}

uint64_t ft_tt_udrv_noc_bank_bytes_written(const ft_tt_udrv_noc_bank *bank)
{
    BANK_SUM(bank->path[i].bytes_written, uint64_t);
}
