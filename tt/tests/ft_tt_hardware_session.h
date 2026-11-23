/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_hardware_session.h
 * @brief The setup every on-card Phase 2 test repeats: open the device, read the grid
 *   from ARC telemetry, derive a real Tensix coordinate, claim a TLB window, and build
 *   an ft_tt_noc_access over it. Plus the two whole-core operations that follow —
 *   booting Tenstorrent's firmware on the five RISC-Vs, and parking them in reset.
 *   Which core and which firmware build come from the environment
 *   (FT_TT_TENSIX_INDEX, FT_TT_FIRMWARE_DIR) so two workers can share the card.
 *
 * This is TEST scaffolding, not backend API. It lives here because the checklist in
 * docs/tt_launch_notes.md §6 is nine separate programs against one core, and repeating
 * the opening 60 lines nine times is how one of them ends up hardcoding a coordinate.
 * Nothing here guesses: the coordinate comes from ft_tt_coord against live telemetry
 * every time (spec §7.3 — a wrong NoC coordinate hangs the chip).
 *
 * ---------------------------------------------------------------------------
 * IT IS THE PRODUCT'S TRANSPORT SEAM, NOT A SECOND ONE
 * ---------------------------------------------------------------------------
 * `ft_tt_hardware_session_open()` opens through `ft_tt_runtime_open_with_driver()`,
 * so `FT_TT_DRIVER=kmd|udrv` selects the transport for every on-card suite exactly
 * as it does for a serve. Until 2026-08-18 this file called `ft_tt_init_device()`
 * directly — i.e. every one of the ~30 on-card suites was hard-wired to tt-kmd, and
 * the card could have moved to a board with no vendor module where NOT ONE OF THEM
 * would run. The engine had a seam; the instruments did not.
 *
 * The default stays `kmd` and an unrecognised value is a REFUSAL, not a fallback —
 * `ft_tt_runtime_driver_select()` owns both rules and this file does not restate
 * them.
 *
 * Two fields the tt-kmd path has and the userspace one does not — `runtime.ctx`
 * and `runtime.binding` — are reachable through `session->runtime`, deliberately
 * NOT mirrored here. A harness that needs them is a harness that needs tt-kmd, and
 * it should say so at the point it reaches for one (`session->runtime.ctx == NULL`
 * is that test) rather than dereference a session field that quietly became NULL.
 *
 * @status Runs on the p150a test card. Every function is used by the on-card
 *   test_phase2_*.c programs. Transport selection added 2026-08-18 and exercised
 *   over `udrv` with `tenstorrent` unloaded.
 * @issues
 *   - The firmware ELFs are read from a program directory on disk rather than embedded:
 *     the tests take whatever build_tensix_program.sh produced. The shipping backend
 *     embeds them (ft_tt_programs.h).
 * @todo None
 */

#ifndef FT_TT_HARDWARE_SESSION_H
#define FT_TT_HARDWARE_SESSION_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_kernel_blob.h"
#include "ft_tt_launch.h"
#include "ft_tt_runtime.h"
#include "ft_tt_tensix_claim.h"
#include "ft_tt_tlb.h"

/** Environment variable naming the directory where build_tensix_program.sh left
 *  fw_{brisc,ncrisc,trisc0,trisc1,trisc2}.elf (a program directory such as
 *  tt/programs/<op>). REQUIRED: there is no default, because a kernel is linked at its own
 *  firmware's `__fw_export_text_end`, so mixing a kernel from one build with firmware from
 *  another is not a configuration, it is a wedged core. */
#define FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT "FT_TT_FIRMWARE_DIR"

/** How long to give the five RISC-Vs to boot. Measured at well under 100 ms. */
#define FT_TT_FIRMWARE_BOOT_TIMEOUT_MS 2000u

/** Environment variable naming which enabled Tensix a single-core test drives. Unset
 *  means index 0, which is what every Phase 2 test used before this existed. */
#define FT_TT_TENSIX_INDEX_ENVIRONMENT "FT_TT_TENSIX_INDEX"

/** How many separate core claims one session may hold. More than one because a
 *  harness can learn about a second core only after resolving the first — the
 *  cross-core test reads its peer index after the compute index — and forcing
 *  those into one call would mean restructuring the resolution order of every
 *  such test for no safety gained. Each call is its own all-or-nothing set. */
#define FT_TT_HARDWARE_SESSION_MAX_CLAIMS 4u

/** An open card on whichever transport `FT_TT_DRIVER` names, its accessors, and
 *  the CORES this process owns for the duration.
 *
 *  `noc` and `grid` are copies of the runtime's, and copying them is safe for the
 *  reason that matters: an accessor vtable and a grid description are both written
 *  once at open and never again, and `noc.user` points into `runtime`, which lives
 *  inside this struct. Anything that MOVES — the byte counters, the aim counts —
 *  is deliberately not copied; read it from `runtime` so a measurement can never
 *  be taken from a frozen snapshot.
 *
 *  `cores` is the per-core half of the coordination split (ft_tt_tensix_claim.h):
 *  the whole-card lock in tools/ft_tt_card_lock.h stays for operations that reset
 *  the card, and everything a single-core or small-group harness does is claimed
 *  core by core instead, so two suites on disjoint cores run at the same time.
 *  Entries are NULL until something claims; all are released by
 *  ft_tt_hardware_session_close(). A claim is a flock under /tmp and needs NO
 *  device handle on either transport. */
typedef struct ft_tt_hardware_session {
    ft_tt_runtime runtime;
    ft_tt_noc_access noc;
    ft_tt_grid_info grid;
    ft_tt_tensix_claim_set *cores[FT_TT_HARDWARE_SESSION_MAX_CLAIMS];
} ft_tt_hardware_session;

/**
 * The tt-kmd context behind this session, or NULL on the userspace driver.
 *
 * For the two harnesses that legitimately want the vendor handle — allocating
 * DRAM through `ft_tt_allocator_init()`, opening a second `ft_tt_noc_path` — so
 * that "this arm needs tt-kmd" is a readable test with a message, rather than a
 * NULL dereference on a field that used to always be set.
 */
ft_tt_ctx *ft_tt_hardware_session_kmd_context(const ft_tt_hardware_session *session);

/** The five firmware images, read from disk once and reused across cores. */
typedef struct ft_tt_firmware_set {
    unsigned char *elf_data[FT_TT_PROC_COUNT];
    size_t elf_size[FT_TT_PROC_COUNT];
} ft_tt_firmware_set;

/**
 * Open device 0 on the transport `FT_TT_DRIVER` selects, validate its grid, claim
 * the windows and wire up the accessors.
 *
 * @param out_session  zeroed and filled on success. The accessors point into it, so
 *                     it must not be copied by value after this returns
 * @return             FT_TT_OK, or a negative FT_TT_* code with the reason logged.
 *                     FT_TT_EINVAL when `FT_TT_DRIVER` names no transport — a typo
 *                     is refused rather than defaulted, because a whole suite
 *                     attributed to the wrong transport is worse than no suite
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_hardware_session_open(ft_tt_hardware_session *out_session);

/** Free the window and close the device. Safe on a NULL or half-opened session. */
void ft_tt_hardware_session_close(ft_tt_hardware_session *session);

/**
 * NoC coordinate of the `index`-th enabled Tensix, in the TRANSLATED system that a TLB
 * window must be aimed with.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL / FT_TT_EUNAVAIL from the coordinate module
 */
int ft_tt_hardware_session_tensix(const ft_tt_hardware_session *session, uint32_t index,
                                  uint16_t *out_x, uint16_t *out_y);

/**
 * Which enabled Tensix this process was told to drive, from
 * FT_TT_TENSIX_INDEX_ENVIRONMENT. Unset is index 0.
 *
 * This exists so two people — or two agents — can run the single-core suite at once
 * without editing sources. It resolves an INDEX, not a coordinate: the coordinate still
 * comes from ft_tt_coord against live telemetry, so a bad value here is a refusal, never
 * a NoC transaction at a guessed endpoint (spec §7.3).
 *
 * @param out_index  written only on success
 * @return           FT_TT_OK; FT_TT_EINVAL for a NULL out, a non-numeric value or
 *                   trailing junk. The RANGE is checked against the live core count when
 *                   the index is resolved, not here — a constant bound would be wrong on
 *                   a 120-core part
 *
 * Thread-safe: no (getenv)    Reentrant: no    Signal-safe: no
 */
int ft_tt_tensix_index_from_environment(uint32_t *out_index);

/**
 * CLAIM the cores this process is about to drive, exclusively against every other
 * process using ft_tt_tensix_claim.h.
 *
 * Refuses rather than blocking, and prints WHO holds a core it cannot get — a pid
 * and the reason that process registered. A timeout would tell the next person
 * nothing; a name tells them everything.
 *
 * All or nothing: a refusal releases anything already taken, so a harness never
 * runs on half its cores.
 *
 * @param retire  RECOVERY-ON-ACQUIRE, run once per core after the whole set is
 *                held. A released claim is NOT a retired residency: the kernel
 *                gives the lock back when a holder dies and cannot give the CORE
 *                back — a SIGKILLed process was measured leaving a core with all
 *                five processors running for over 40 s. Pass a function that
 *                retires (a queue's `ft_tt_queue_request_stop()` + re-init) or
 *                that INSPECTS and refuses a core that is not idle. Passing NULL
 *                means "this harness accepts whatever state it finds", which is
 *                only defensible for a program that boots firmware from scratch
 *                anyway. **Never retire by parking**: asserting reset on a
 *                resident triad breaks the core's NEXT boot.
 * @return  FT_TT_OK; FT_TT_EPERM when a core is held, reserved or unretirable
 *          (already explained on stderr); FT_TT_EINVAL; FT_TT_EIO
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_hardware_session_claim_cores(ft_tt_hardware_session *session, const uint32_t *indices,
                                       size_t count, const char *reason,
                                       ft_tt_tensix_claim_retire_fn retire,
                                       void *retire_context);

/**
 * ft_tt_tensix_index_from_environment() plus ft_tt_hardware_session_tensix(), printing
 * the resolved index and TRANSLATED coordinate — which every one of these tests wants in
 * its log, because "which core did this run on" is the first question a failure raises.
 *
 * It also CLAIMS that core, because a harness that names its core should own it.
 * The session is no longer `const` for exactly that reason. A harness that drives
 * MORE cores than the one this resolves must claim the rest itself with
 * ft_tt_hardware_session_claim_cores() — this call claims one core, and one only.
 *
 * @param out_index  may be NULL
 * @return           FT_TT_OK, or the first failing step's code (logged); FT_TT_EPERM
 *                   when the core is held by another process
 */
int ft_tt_hardware_session_tensix_from_environment(ft_tt_hardware_session *session,
                                                   uint16_t *out_x, uint16_t *out_y,
                                                   uint32_t *out_index);

/** The firmware/program directory this process should read, from
 *  FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT, else $HOME/ttprog. NULL (with the reason logged)
 *  only when both are unset; ft_tt_firmware_set_load() refuses a NULL directory with
 *  FT_TT_EINVAL. */
const char *ft_tt_firmware_directory(void);

/**
 * Read the five firmware ELFs out of `directory` (pass ft_tt_firmware_directory()).
 *
 * @return FT_TT_OK, FT_TT_EIO if a file is missing or short, FT_TT_ENOMEM.
 *         Ownership passes to the caller: release with ft_tt_firmware_set_release().
 */
int ft_tt_firmware_set_load(const char *directory, ft_tt_firmware_set *out_set);

/** Free every image. Safe on NULL and on a partially loaded set. */
void ft_tt_firmware_set_release(ft_tt_firmware_set *set);

/**
 * Bring one Tensix from unknown to "firmware running, waiting for a launch message":
 * assert reset on the five RISC-Vs, clear the mailbox, load the five firmware images,
 * prime the boot handshake with RUN_MSG_INIT, deassert BRISC only, and wait for BRISC to
 * write RUN_MSG_DONE.
 *
 * This is docs/tt_launch_notes.md §6 steps 3 and 4 as one call. Every ordering
 * constraint in it is load-bearing and documented there; do not reorder.
 *
 * @param placements  indexed by FT_TT_PROC_*, receives where each firmware image landed;
 *                    may be NULL. This is what ft_tt_image_placement_l1_floor() consumes,
 *                    and it is the FIRMWARE placements that matter — firmware `.data`
 *                    staging starts exactly at FT_TT_L1_USABLE_BASE, while kernel images
 *                    sit below it and cannot reach the arena at all (ft_tt_launch.h)
 * @return FT_TT_OK, or the first failing step's code (logged)
 */
int ft_tt_core_boot_firmware(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                             const ft_tt_firmware_set *firmware,
                             ft_tt_image_placement placements[FT_TT_PROC_COUNT]);

/** Write every image of `program` into `(x, y)`'s L1, reporting where each landed.
 *  `placements` is indexed by FT_TT_PROC_* and zeroed for processors the program
 *  leaves idle; it may be NULL. */
int ft_tt_core_load_program(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            const ft_tt_kernel_program *program,
                            ft_tt_image_placement placements[FT_TT_PROC_COUNT]);

/**
 * Put every RISC-V of one Tensix back in reset. Call it on EVERY exit path, including
 * failures — a core left running is a core that can still issue NoC transactions.
 *
 * The five RISC-Vs only. The matrix backend's reset bits are in the same register and
 * must NOT be touched (ft_tt_launch.h).
 */
int ft_tt_core_park(const ft_tt_noc_access *noc, uint16_t x, uint16_t y);

/**
 * Wait until TRISC0 reports RUN_SYNC_MSG_DONE.
 *
 * BRISC asks TRISC0 to re-zero all 64 circular-buffer counter registers after every run
 * (BRISC's post-run firmware step), and it does that AFTER writing RUN_MSG_DONE is
 * scheduled — so a host that stages CB counters between runs must wait for this or its
 * writes get zeroed underneath it.
 *
 * @return FT_TT_OK, FT_TT_EIO on timeout
 */
int ft_tt_core_wait_sync_registers_ready(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                         uint32_t timeout_ms);

#endif /* FT_TT_HARDWARE_SESSION_H */
