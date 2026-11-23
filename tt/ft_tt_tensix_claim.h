/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_tensix_claim.h
 * @brief Per-CORE ownership of Tensix cores, so ordinary correctness work stops
 *   serialising behind the whole-card lock. A job names the cores it wants, gets
 *   them or gets told who has them, and the kernel takes them back if it dies.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS — the card has 140 cores and one lock
 * ---------------------------------------------------------------------------
 * `tools/ft_tt_card_lock.h` is a WHOLE-CARD, single-holder lock. It is correct
 * for what it was written for (a reset returns every Tensix to power-on default,
 * so it must exclude everyone), but almost everything this project runs is
 * correctness work on a handful of cores. Serialising a four-core suite behind a
 * card-wide lock costs 136 idle cores per round.
 *
 * This module is the per-core half of that split. It is a thin, opinionated
 * layer over `ft_tt_udrv_claim.h`, which already supplies the mechanism: one
 * `flock(2)` per resource in a per-card, per-class namespace, so THE CLAIM IS
 * THE LOCK — there is no window in which two processes both believe they own a
 * core, and the kernel releases every claim a process holds the instant it dies,
 * including under `SIGKILL`. Read that header for why flock rather than a shared
 * bitmap or a broker.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS CARD-WIDE AND WHAT IS PER-CORE
 * ---------------------------------------------------------------------------
 * The split, with the reason each operation lands where it does. Getting this
 * wrong means one job silently corrupting another's cores, so the test is not
 * "how big does it feel" but **what can this operation disturb that it does not
 * own**.
 *
 * CARD-WIDE (`ft_tt_card_lock.h`, exclusive):
 *   - `recover --rung 3` / `--rung 4` — `RESET_DEVICE ASIC_RESET` / `ASIC_DMC_RESET`.
 *     Every Tensix returns to power-on default. It also unbinds tt-kmd, removes
 *     `/dev/tenstorrent/0`, invalidates every TLB window and BAR mapping in every
 *     process on the box, and drops the PCIe link to Gen1. Not a
 *     core operation at any granularity.
 *   - `recover --rung 5` — PCIe secondary bus reset. Everything above, plus it
 *     reaches every other device behind the same bridge.
 *   - A FIRMWARE RELOAD (SPI flash). Not implemented here; it is card state and
 *     the pin at 19.2.0 is a card-wide decision.
 *   - An AI-CLOCK / POWER-STATE REQUEST (`tools/ft_tt_power_request.c`). The
 *     clock tree is chip-wide — measured at +38 W for the clock against +4-7 W
 *     for four busy Tensix — so one job asking for max moves every
 *     other job's timing. This one is currently UNLOCKED and is flagged, not
 *     fixed, here: it needs the card lock, and wiring it is a separate change.
 *
 * PER-CORE (this module):
 *   - `probe --write`. It boots firmware and runs a vector_add triad on ONE named
 *     core: that core's L1, that core's mailbox, that core's soft-reset register.
 *     Nothing it writes is addressed outside the core.
 *   - `recover --rung 0` (deassert the RISC-Vs), `--rung 1` (reload firmware and
 *     boot), `--rung 2` (poke what the core waits on). All three are one core's
 *     registers and L1, which is exactly why `ft_tt_card_lock.h` already says
 *     rungs 0-2 must not take the card lock.
 *   - Every on-card test harness that names its cores through
 *     `FT_TT_TENSIX_INDEX` (one core) or `FT_TT_TENSIX_INDICES` (a set).
 *
 * NEITHER (no lock at all):
 *   - `status`, `topology`. They only read. A NoC read of a wedged core is
 *     indistinguishable from a read of a healthy one and changes neither.
 *
 * ---------------------------------------------------------------------------
 * THREE BLAST RADII THAT ARE NOT SETTLED — say so rather than guess
 * ---------------------------------------------------------------------------
 * 1. **MULTICAST.** A multicast rectangle physically reaches cores the issuing
 *    job does not own, and more than four destinations wedges reliably
 *    (fingerprint `0x00047000`). Claiming every core in the rectangle is
 *    necessary and is what a caller should do — but it is NOT provably
 *    sufficient, because the failure is in shared NoC fabric rather than in the
 *    destination core, and the recovery for it is whole-card. A job doing wide
 *    multicast should take the card lock until someone bounds this.
 * 2. **DRAM.** A core claim says nothing about a DRAM address range. Two jobs on
 *    disjoint cores can still write the same arena, and the bank-striped matmul's
 *    own history is exactly that failure (two members sharing a base address,
 *    every status word clean, only a byte comparison saw it — DEV_NOTES §3).
 *    A `dram` namespace does not exist yet; DRAM is UNCLAIMED.
 * 3. **CONCURRENT FIRMWARE BOOTS.** At 12-way and above, independent host
 *    processes booting firmware at once start returning `device I/O failed` and
 *    leave cores mid-handshake. Per-core claims make that
 *    concurrency reachable; they do not make it safe. This is a real ceiling on
 *    how many claim-holders may boot at once, and it is a property of the boot
 *    path, not of this module.
 *
 * ---------------------------------------------------------------------------
 * A RELEASED CLAIM IS NOT A RETIRED RESIDENCY
 * ---------------------------------------------------------------------------
 * This is the whole reason `retire` exists. The kernel gives the LOCK back on
 * abnormal death; it cannot give the CORE back. A `SIGKILL`ed process was
 * measured holding a core with all five processors still running for over 40
 * seconds, so the resident kernel's bounded-idle backstop is far weaker than it
 * reads, and a harness that calls `_Exit()` is enough to leave a core busy with
 * no host attached.
 *
 * So the design is RECOVERY-ON-ACQUIRE: the next claimant retires whatever it
 * finds before using it. `retire` is a CALLBACK rather than something this module
 * does itself, for two reasons that are not stylistic:
 *   - This module must stay pure filesystem coordination so it builds and is
 *     gated with no card and no `PHASE1B=1`. A retirement issues NoC writes.
 *   - There is no safe GENERIC retirement. Retiring a resident kernel is done by
 *     ASKING it — `ft_tt_queue_request_stop()` then `ft_tt_queue_init()` for a
 *     fresh nonce — which needs the op family's queue. The obvious generic
 *     alternative, asserting reset, is the one thing DEV_NOTES §1 forbids
 *     outright: parking a resident triad breaks the core's NEXT boot even after
 *     a textbook-clean shutdown. A caller that owns a queue can supply a real
 *     retirement; a caller that does not should supply one that INSPECTS and
 *     REFUSES a core that is not idle, which is strictly better than proceeding.
 *
 * @status Off-hardware complete. Touches no device: filesystem coordination
 *   only, plus a callback the caller supplies. Gated by tests/test_tensix_claim.c.
 * @issues
 *   - flock is ADVISORY, inherited from ft_tt_udrv_claim. A process that does not
 *     use this module can drive any core it likes. This protects cooperating
 *     jobs, which is all the card lock protected either.
 * @todo None
 */

#ifndef FT_TT_TENSIX_CLAIM_H
#define FT_TT_TENSIX_CLAIM_H

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "ft_tt_udrv_claim.h"

/**
 * Exclusive upper bound on a claimable core index.
 *
 * This is a FILESYSTEM NAMESPACE bound, not a part size, and the distinction
 * matters: DEV_NOTES §4 records that "the SoC descriptor's 140 cores" is a
 * property of the board and that nothing may be sized from it. Nothing here is.
 * The bound that decides whether a core EXISTS stays where it always was, in
 * `ft_tt_coord_tensix_by_index()` against the live core count; this one only
 * decides how many lock files a namespace may contain, and is deliberately
 * larger than any p150 so it can never be the thing that refuses a real core.
 */
#define FT_TT_TENSIX_CLAIM_INDEX_COUNT 256u

/** Most cores one request may name. */
#define FT_TT_TENSIX_CLAIM_MAX_CORES 256u

/** Resource class, i.e. the namespace directory. Cores and TLB windows are
 *  different resources, so core 7 and window 7 never collide. */
#define FT_TT_TENSIX_CLAIM_NAMESPACE "tensix"

/** Card identity when the caller does not supply one. Overridden by
 *  FT_TT_TENSIX_CLAIM_CARD_ENVIRONMENT; a PCI BDF is the right value once the
 *  device is open, and this default exists so a caller that has not opened one
 *  yet still coordinates with everyone else on a single-card box. */
#define FT_TT_TENSIX_CLAIM_CARD_DEFAULT "card0"

/** Environment override for the card identity, e.g. "0000:12:00.0". */
#define FT_TT_TENSIX_CLAIM_CARD_ENVIRONMENT "FT_TT_CARD"

/** Environment coordination root. Exists so a test drives the whole module
 *  inside a scratch directory and never touches the live one. */
#define FT_TT_TENSIX_CLAIM_ROOT_ENVIRONMENT "FT_TT_CLAIM_ROOT"

/**
 * Environment reserved set: cores this process must never claim, as a
 * comma-separated list of indices and ranges ("0-9,14,15,28,42,70").
 *
 * Two different things need it. Cores belonging to another op family (attention,
 * norms, sampling) are a POLICY exclusion — nobody holds them right now, but a
 * sweep must not wander into them. And a core a live serve is using is excluded
 * by the serve's own claim, which is the load-bearing mechanism; the reserved set
 * is the belt to that braces, for the window before a serve claims or for a serve
 * that predates this module.
 */
#define FT_TT_TENSIX_CLAIM_RESERVED_ENVIRONMENT "FT_TT_TENSIX_RESERVED"

/**
 * Recovery-on-acquire hook. Called once per core, in ascending index order,
 * AFTER every core in the request is held — so a retirement can never race a
 * second claimant for the same core.
 *
 * @param context      whatever the caller put in the request
 * @param core_index   the core to retire
 * @return             0 to accept the core, non-zero to refuse it. A refusal
 *                     releases the WHOLE set: a job that cannot get one of its
 *                     cores into a known state must not run on the others.
 */
typedef int (*ft_tt_tensix_claim_retire_fn)(void *context, uint32_t core_index);

/** What a request wants. Zero-initialise and fill in what you need. */
typedef struct ft_tt_tensix_claim_request {
    const char *root;         /**< coordination root; NULL reads the environment */
    const char *card;         /**< card identity; NULL reads the environment */
    const uint32_t *indices;  /**< the cores wanted */
    size_t count;             /**< how many */
    const char *reason;       /**< short purpose, recorded so a refusal can name it */
    const uint32_t *reserved; /**< explicit reserved set; NULL reads the environment */
    size_t reserved_count;    /**< how many; ignored when `reserved` is NULL */
    ft_tt_tensix_claim_retire_fn retire; /**< recovery-on-acquire; may be NULL */
    void *retire_context;
} ft_tt_tensix_claim_request;

/** Why a request was refused. Every one of these prints differently, because a
 *  reader at 2am needs to know which it was. */
typedef enum ft_tt_tensix_claim_outcome {
    FT_TT_TENSIX_CLAIM_GRANTED = 0, /**< nothing was refused */
    FT_TT_TENSIX_CLAIM_HELD,        /**< another live process holds the core */
    FT_TT_TENSIX_CLAIM_RESERVED,    /**< excluded by the reserved set */
    FT_TT_TENSIX_CLAIM_DUPLICATE,   /**< the request named one core twice */
    FT_TT_TENSIX_CLAIM_UNRETIRED,   /**< claimed, but recovery-on-acquire refused it */
    FT_TT_TENSIX_CLAIM_FAILED       /**< bad argument, or the filesystem failed */
} ft_tt_tensix_claim_outcome;

/** The refusal, filled in enough to name a human. */
typedef struct ft_tt_tensix_claim_refusal {
    ft_tt_tensix_claim_outcome outcome;
    uint32_t index;                            /**< the core that was refused */
    pid_t pid;                                 /**< holder, 0 when unknown */
    char reason[FT_TT_UDRV_CLAIM_REASON_MAX];  /**< holder's reason, "" when unknown */
    int detail;                                /**< underlying FT_TT_UDRV_* code */
} ft_tt_tensix_claim_refusal;

/** A held set of core claims. Opaque; release with ft_tt_tensix_claim_release(). */
typedef struct ft_tt_tensix_claim_set ft_tt_tensix_claim_set;

/**
 * Claim every core in `request`, all or nothing.
 *
 * Never blocks — every acquisition is `LOCK_NB`. A blocked caller is worse than a
 * refused one here: the holder may be a multi-hour serve, and a timeout tells the
 * next person nothing while a named pid tells them everything.
 *
 * ALL OR NOTHING is the property that matters and it is not decoration. Cores are
 * taken in the order given; the first refusal releases everything already taken
 * before returning, so no caller is ever left holding half a set it cannot use
 * and nobody else can claim.
 *
 * @param request      what to claim; `indices` and `count` are required
 * @param out_set      receives the set on success; untouched on refusal
 * @param out_refusal  zeroed and filled on every call, so it is safe to print
 *                     unconditionally. May be NULL
 * @return             FT_TT_UDRV_OK; FT_TT_UDRV_EAGAIN when a core is held or
 *                     reserved or a retirement refused; FT_TT_UDRV_EINVAL on a
 *                     bad argument or a duplicated index; FT_TT_UDRV_EIO;
 *                     FT_TT_UDRV_ENOMEM
 *
 * Thread-safe: yes across processes   Reentrant: no (allocates)   Signal-safe: no
 */
int ft_tt_tensix_claim_acquire(const ft_tt_tensix_claim_request *request,
                               ft_tt_tensix_claim_set **out_set,
                               ft_tt_tensix_claim_refusal *out_refusal);

/** Release every claim in the set and free it. NULL is a no-op. */
void ft_tt_tensix_claim_release(ft_tt_tensix_claim_set *set);

/** How many cores the set holds; 0 for NULL. */
size_t ft_tt_tensix_claim_count(const ft_tt_tensix_claim_set *set);

/** The `slot`-th core index held, or UINT32_MAX when out of range. */
uint32_t ft_tt_tensix_claim_index_at(const ft_tt_tensix_claim_set *set, size_t slot);

/**
 * Print the refusal to stderr, naming the holder.
 *
 * "core 71 is held by pid 12345" is useful; a timeout is not. This is the whole
 * point of the record `ft_tt_udrv_claim` writes next to each lock.
 */
void ft_tt_tensix_claim_refusal_print(const ft_tt_tensix_claim_refusal *refusal);

/**
 * Parse a core list: comma-separated indices and inclusive ranges, e.g.
 * "1,3,17-19,139". Whitespace is not accepted, deliberately — an environment
 * variable with a space in it is more likely a quoting mistake than an intent.
 *
 * @param text        the list; NULL or empty yields zero indices and FT_TT_UDRV_OK
 * @param out         receives the indices, in the order written
 * @param capacity    how many `out` holds
 * @param out_count   receives how many were written
 * @return            FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL on malformed text, an
 *                    index at or above FT_TT_TENSIX_CLAIM_INDEX_COUNT, an
 *                    inverted range, or more indices than `capacity`
 */
int ft_tt_tensix_claim_parse_indices(const char *text, uint32_t *out, size_t capacity,
                                     size_t *out_count);

/**
 * The coordination root and card identity a claim would use for `root` / `card`.
 *
 * Exported so that anything INSPECTING the namespace — a whole-card operation
 * taking a census of who it is about to destroy — resolves it the same way a
 * claimant does. A census that looked in a different directory than the
 * claimants would report "nobody is using the card" and be believed, which is
 * the most expensive possible wrong answer here.
 *
 * @return `root` / `card` when non-empty; otherwise the environment; otherwise
 *         NULL for the root (meaning ft_tt_udrv_claim's own default) and
 *         FT_TT_TENSIX_CLAIM_CARD_DEFAULT for the card.
 */
const char *ft_tt_tensix_claim_resolve_root(const char *root);
const char *ft_tt_tensix_claim_resolve_card(const char *card);

/**
 * Read FT_TT_TENSIX_CLAIM_RESERVED_ENVIRONMENT into `out`.
 *
 * Unset is an EMPTY reserved set, not a default list of cores. A constant
 * default would be a bound nobody chose, and this project has already paid for
 * one of those (DEV_NOTES §4: a constant that records a fact no code consults
 * reads like a guard).
 */
int ft_tt_tensix_claim_reserved_from_environment(uint32_t *out, size_t capacity,
                                                 size_t *out_count);

#endif /* FT_TT_TENSIX_CLAIM_H */
