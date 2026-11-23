/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_runtime.h
 * @brief The host runtime a SERVING process needs on top of the backend: open the card and
 *   hold one NoC window, read firmware and kernel images off disk, and bring a Tensix from
 *   unknown to "firmware running, waiting for a launch message".
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS — it is a PROMOTION, not a new layer
 * ---------------------------------------------------------------------------
 * Every one of these steps already existed, in `tests/ft_tt_hardware_session.{h,c}` and
 * `tests/ft_tt_program_directory.{h,c}`, both of which say in their own headers that they
 * are TEST SCAFFOLDING and not backend API. That was right while the only caller was a
 * checklist. Phase 7 makes Pantheon a caller, and a serving process cannot include a test
 * directory — so the shipping half moves here and the test scaffolding stays where it is,
 * unchanged, so no on-card suite has to be re-proven to land this.
 *
 * The overlap is deliberate and bounded: this file owns the four operations a serve needs
 * and nothing else. It has no environment reading, no logging conventions of its own, and
 * no core-selection policy — those are the caller's, because a serve picks cores from a
 * placement and a test picks them from `FT_TT_TENSIX_INDEX`.
 *
 * ---------------------------------------------------------------------------
 * THE TWO RULES A CALLER MUST NOT BREAK
 * ---------------------------------------------------------------------------
 *  1. **A kernel and its firmware are ONE build output.** A kernel's text is linked at its
 *     own firmware's `__fw_export_text_end`, so mixing a kernel from one directory with
 *     firmware from another is not a configuration, it is a wedged core. Load both from
 *     the SAME directory — `ft_tt_runtime_firmware_load()` and
 *     `ft_tt_runtime_program_load()` take one each precisely so the pairing is visible at
 *     the call site.
 *  2. **Never park a resident triad, and never park on a failure path.** There is no park
 *     function here on purpose. Retirement is `ft_tt_watchdog_shutdown()`, which asks the
 *     kernel to leave and returns the core to its firmware's GO wait — where a healthy
 *     idle core sits. Parking a five-processor resident program breaks its NEXT boot, twice
 *     measured.
 *
 * ---------------------------------------------------------------------------
 * THIS IS THE TRANSPORT SEAM
 * ---------------------------------------------------------------------------
 * `ft_tt_runtime_open()` is the ONE place the backend decides how it reaches the card.
 * It owns the device open, the windows and the accessors, and `ft_tt_launch.h` already
 * defines a vtable both transports satisfy — so selecting tt-kmd or foundry's own
 * userspace driver is a branch here and NOTHING above this file changes. See
 * `ft_tt_runtime_udrv.h` for the second branch and for what it costs.
 *
 * @status Written for Phase 7. Compiles into a Pantheon `USE_TT=1` build. The second
 *   (WC) window is on the card: it took weight residency 28.7 s -> 2.6 s and a token's
 *   readback 27.0 s -> 6.6 s with no other change. The `FT_TT_DRIVER=udrv` branch was
 *   added 2026-08-17 and is what lets a serve run on a machine with no vendor module.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_RUNTIME_H
#define FT_TT_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_kernel_blob.h"
#include "ft_tt_launch.h"
#include "ft_tt_runtime_udrv.h"
#include "ft_tt_tlb.h"

/** The sliding aperture size a runtime claims. One 2 MiB window is all the NoC layer needs;
 *  it re-aims transparently as an address moves (ft_tt_noc.h), and the aim costs 1.23 µs. */
#define FT_TT_RUNTIME_WINDOW_BYTES (2u * 1024u * 1024u)

/** How long to give the five RISC-Vs to boot. Measured at well under 100 ms. */
#define FT_TT_RUNTIME_BOOT_TIMEOUT_MS 2000u

/** Longest program or source name held; the strings are only for logs. */
#define FT_TT_RUNTIME_NAME_BYTES 96u

/** Environment variable that selects the transport. `kmd` (the default) opens
 *  `/dev/tenstorrent/<n>`; `udrv` uses foundry's own userspace driver and needs
 *  no vendor module. Anything else is a refusal, not a fallback. */
#define FT_TT_RUNTIME_DRIVER_ENVIRONMENT "FT_TT_DRIVER"

/**
 * Which transport a runtime is built on.
 *
 * ---------------------------------------------------------------------------
 * THE SELECTION IS OPT-IN AND THE FAILURE IS A REFUSAL
 * ---------------------------------------------------------------------------
 * The default stays `kmd` until the card moves, so nothing changes for anyone
 * who does not ask. But an EXPLICIT `FT_TT_DRIVER=udrv` that cannot be honoured
 * must stop the process, not quietly serve on the CPU:
 * `ft_tt_runtime_driver_is_explicit()` is what lets a caller tell "there is no
 * card on this machine" (fine, stay CPU-resident) from "I was told to use the
 * card and could not" (fatal). This project has shipped a silent fallback twice
 * and both times it produced a plausible answer from the wrong thing.
 */
typedef enum ft_tt_runtime_driver {
    FT_TT_RUNTIME_DRIVER_KMD = 0, /**< tt-kmd's chardev — the default */
    FT_TT_RUNTIME_DRIVER_UDRV = 1 /**< foundry's own userspace driver */
} ft_tt_runtime_driver;

/**
 * An open card, TWO claimed windows, an accessor over each, and the live grid.
 *
 * ---------------------------------------------------------------------------
 * WHY TWO WINDOWS — and why the second one is not an optimisation of the first
 * ---------------------------------------------------------------------------
 * `noc` is the RUN-CONTROL accessor: a UC window driven at FT_TT_NOC_WIDTH_WORD32,
 * one 32-bit access per word. That width is a CORRECTNESS REQUIREMENT, not a knob —
 * the per-Tensix soft-reset register at 0xFFB121B0 is four bytes wide and so is a go
 * message, and a 16 B store over either takes its neighbours with it (ft_tt_noc.h).
 * Nothing may move off it.
 *
 * `bulk_dram` is the DATA accessor: a WC window driven through ft_tt_read_dram() /
 * ft_tt_write_dram(), i.e. FT_TT_NOC_WIDTH_BLOCK16 with the memory-space bounds and
 * alignment checks those helpers carry. Card->host is transaction-count bound, so
 * moving an activation strip 4 bytes at a time costs 4x what 16 does — measured as
 * 71.1% of a whole token.
 *
 * The separation is STRUCTURAL, not a convention: `bulk_dram` refuses a register
 * address (0xFFB121B0 is outside the 0xff000000 DRAM space) and refuses a 4-byte
 * length (under the 16 B unit), so a register access cannot silently take the bulk
 * path. That is the failure mode worth designing out — it would not surface until
 * something hung.
 */
typedef struct ft_tt_runtime {
    ft_tt_ctx *ctx;
    ft_tt_tlb_window *window;
    ft_tt_noc_window_binding binding;
    ft_tt_noc_access noc; /**< UC / WORD32 — registers, mailbox, images. NEVER bulk data */
    ft_tt_noc_path bulk_path;    /**< WC window, owned by this path */
    ft_tt_noc_access bulk_dram;  /**< BLOCK16 over bulk_path; DRAM addresses only */
    /**
     * BLOCK16 over the same WC path, for TENSIX L1 — the third accessor, and it exists for
     * exactly one reason: reads on this link are TRANSACTION-COUNT bound, and `noc` moves
     * four bytes per transaction. Anything that reads more than a word out of a core's L1
     * on a hot path pays 4x through `noc` and has no business there.
     *
     * The group completion block is the case that forced it: asking N members "are you
     * done" is `2N` words of one core's L1, which is `2N` PIO transactions at ~3.3 us
     * through `noc` and `ceil(8N/16)` at ~3.4 through this — a 4x shallower slope in the
     * member count, on the same bytes, with no kernel involved.
     *
     * It is NOT a general replacement for `noc`. L1 alignment is 16 bytes in address AND
     * length, so a 4-byte control word cannot come through it; that separation is what
     * keeps a register or a go message off the bulk path.
     */
    ft_tt_noc_access bulk_l1;
    ft_tt_grid_info grid;

    /**
     * WHICH TRANSPORT THIS RUNTIME IS ON, and the handle for the one that has
     * no `ft_tt_ctx`.
     *
     * On the udrv path `ctx`, `window` and `binding` are all zero and `udrv` is
     * non-NULL; on the kmd path it is the other way round. Everything ABOVE
     * this struct — launch, queue, ops, residencies, ft_tt_core — reads only
     * `noc`, `bulk_dram`, `bulk_l1` and `grid`, and therefore does not change
     * at all. The two callers that legitimately want the tt-kmd handle
     * (`SET_NOC_CLEANUP` and the DRAM allocation tally) already guard on NULL,
     * which is the honest answer: those are tt-kmd features and the userspace
     * driver does not have them.
     */
    ft_tt_runtime_driver driver;
    ft_tt_runtime_udrv *udrv;
} ft_tt_runtime;

/**
 * The transport `FT_TT_DRIVER` asks for, defaulting to `kmd`.
 *
 * @param out_explicit  may be NULL; set true when the variable was present and
 *                      named a transport, so a caller can make a failure fatal
 * @return FT_TT_OK, or FT_TT_EINVAL for a value that is neither `kmd` nor
 *         `udrv` — an unrecognised transport is REFUSED rather than defaulted,
 *         because a typo that silently selects tt-kmd is a whole round measured
 *         on the wrong path
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (getenv)
 */
int ft_tt_runtime_driver_select(ft_tt_runtime_driver *out_driver, bool *out_explicit);

/** Human-readable name for a transport. Never NULL. */
const char *ft_tt_runtime_driver_string(ft_tt_runtime_driver driver);

/**
 * The card's PCI address, e.g. `0000:12:00.0`, on either transport.
 *
 * @return FT_TT_OK; FT_TT_EINVAL on a bad argument; FT_TT_EUNAVAIL when the
 *         open transport cannot answer
 */
int ft_tt_runtime_pci_address(const ft_tt_runtime *runtime, char *out, size_t out_bytes);

/**
 * What each of this runtime's two windows carried, on EITHER transport.
 *
 * The instrument this replaces read the tt-kmd binding directly and therefore
 * answered ZERO on the udrv path — a counter whose whole job is to catch bulk
 * data on the register window, reporting that no bytes moved at all on a
 * transport that had just served a token. A broken instrument is worse than no
 * instrument (`harnesses here measure nothing`), so it goes through the seam
 * like everything else.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL with `out` zeroed
 */
int ft_tt_runtime_window_stats_read(const ft_tt_runtime *runtime, ft_tt_runtime_window_stats *out);

/**
 * Total DRAM this card carries, derived from the LIVE grid: the telemetry bank
 * count times the selected architecture's bank size.
 *
 * Transport-agnostic on purpose — this is what a serve budgets its models
 * against, and `ft_tt_mem_info()` can only answer it through an `ft_tt_ctx`.
 *
 * @return bytes, or 0 when the grid has not been read
 */
uint64_t ft_tt_runtime_dram_bytes(const ft_tt_runtime *runtime);

/**
 * Open device `device_index` on the transport `FT_TT_DRIVER` selects, read the grid from
 * ARC telemetry, refuse a topology ft_tt_coord cannot map, claim BOTH windows and wire up
 * both accessors.
 *
 * On `kmd` that is `/dev/tenstorrent/<device_index>` and two `ALLOCATE_TLB` windows; on
 * `udrv` it is a sysfs BAR mapping and two flock'd windows, with our own ARC bring-up in
 * place of the chardev open's. **The two produce the same three accessors and the same
 * grid, and everything above this function is identical on both.**
 *
 * The grid check is not a formality: a coordinate we cannot derive is a coordinate we must
 * not probe, and a wrong NoC endpoint hangs the chip (spec §7.3).
 *
 * @param out_runtime  zeroed and filled on success; both accessors point into it, so it
 *                     must not be copied by value after this returns
 * @return             FT_TT_OK, or a negative FT_TT_* code with the reason logged
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_runtime_open(int device_index, ft_tt_runtime *out_runtime);

/**
 * Open on a NAMED transport, ignoring `FT_TT_DRIVER`.
 *
 * For the one caller that knows which transport it needs from its own command
 * line rather than from the environment: `ft_tt_core recover --driver udrv` has
 * to read a card's state on a machine where the chardev does not exist, and
 * making it export an environment variable to say so would put the decision two
 * places instead of one.
 *
 * @return FT_TT_OK, or a negative FT_TT_* code. There is no fallback to the
 *         other transport, ever.
 */
int ft_tt_runtime_open_with_driver(int device_index, ft_tt_runtime_driver driver,
                                   ft_tt_runtime *out_runtime);

/** Free the window and close the device. Safe on NULL and on a half-opened runtime. */
void ft_tt_runtime_close(ft_tt_runtime *runtime);

/** Where a set of images came from. Carried on the images themselves so a log or an
 *  assertion can state it, rather than inferring it from which call was made. */
typedef enum ft_tt_program_source {
    FT_TT_PROGRAM_SOURCE_NONE = 0,      /**< nothing loaded */
    FT_TT_PROGRAM_SOURCE_EMBEDDED = 1,  /**< out of ft_tt_programs.h — the default */
    FT_TT_PROGRAM_SOURCE_DIRECTORY = 2  /**< off disk, because a caller asked explicitly */
} ft_tt_program_source;

/**
 * Five ELF images, plus the program they form.
 *
 * ONE struct serves firmware and kernels alike because the difference is only which file
 * names are read and which `ft_tt_image_kind` the loader is told — the images themselves
 * are the same shape. `program` is meaningful only after a program load; a firmware load
 * fills `elf_data`/`elf_size` and leaves it empty.
 *
 * `elf_data` is what a caller reads and is `const`, because the bytes may be a `static
 * const` array inside the binary. `owned_data` is the release side: non-NULL exactly when
 * the buffer was malloc'd from a file, NULL for an embedded image. Two arrays rather than
 * one flag so that freeing embedded memory is not a mistake the code CAN make.
 */
typedef struct ft_tt_runtime_images {
    ft_tt_kernel_program program;
    ft_tt_kernel_image images[FT_TT_PROC_COUNT];
    const unsigned char *elf_data[FT_TT_PROC_COUNT];
    size_t elf_size[FT_TT_PROC_COUNT];
    unsigned char *owned_data[FT_TT_PROC_COUNT];
    char source_names[FT_TT_PROC_COUNT][FT_TT_RUNTIME_NAME_BYTES];
    char name[FT_TT_RUNTIME_NAME_BYTES];
    ft_tt_program_source source;
} ft_tt_runtime_images;

/**
 * Read `fw_{brisc,ncrisc,trisc0,trisc1,trisc2}.elf` from `directory`.
 *
 * All five, always: `ft_tt_runtime_boot_core()` loads every processor's firmware, and a
 * missing one is a core that boots into whatever L1 held.
 *
 * @return FT_TT_OK; FT_TT_EIO if a file is missing or short; FT_TT_ENOMEM; FT_TT_EINVAL
 *
 * Ownership: release with ft_tt_runtime_images_release().
 */
int ft_tt_runtime_firmware_load(const char *directory, ft_tt_runtime_images *out_images);

/**
 * Read one kernel ELF per processor named in `processor_mask` from `directory`.
 *
 * File names are the ones build_tensix_program.sh writes: brisc.elf, ncrisc.elf,
 * trisc0.elf, trisc1.elf, trisc2.elf. A `.src` sidecar, if present, supplies the source
 * name for logs; its absence is not an error. The resulting program's enables mask is
 * exactly `processor_mask`, so a two-processor program (the memory family is DM0|DM1, with
 * no compute thread) is expressed by the mask and not by a missing file.
 *
 * @return FT_TT_OK; FT_TT_EINVAL on an empty mask; FT_TT_EIO; FT_TT_ENOMEM
 */
int ft_tt_runtime_program_load(const char *directory, uint32_t processor_mask,
                               ft_tt_runtime_images *out_images);

/** Free every image. Safe on NULL and on a partially loaded set. */
void ft_tt_runtime_images_release(ft_tt_runtime_images *images);

/**
 * Bring one Tensix from unknown to "firmware running, waiting for a launch message".
 *
 * Six steps and every ordering constraint in them is load-bearing:
 *   1. assert reset on the five RISC-Vs — and NOTHING ELSE. The same register carries the
 *      matrix backend's bits, and asserting those permanently corrupts that core's compute
 *      (ft_tt_launch.h; it cost 14 cores once);
 *   2. clear the mailbox;
 *   3. load all five firmware images;
 *   4. prime the boot handshake with RUN_MSG_INIT — `RUN_MSG_DONE` is 0, so without this
 *      step 6 passes whether or not firmware is running;
 *   5. deassert BRISC ONLY — its firmware releases the other four itself, and releasing
 *      them from the host races it;
 *   6. wait for BRISC to report done, bounded.
 *
 * @param x, y  TRANSLATED coordinates from ft_tt_coord — never a guess (spec §7.3)
 * @return      FT_TT_OK, or the first failing step's code
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_runtime_boot_core(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            const ft_tt_runtime_images *firmware);

/** Where one boot's host time went (seconds), for the role-switch floor (P-boot). */
typedef struct ft_tt_runtime_boot_timing {
    double reset_seconds;     /**< steps 1-2: assert reset, clear the mailbox */
    double images_seconds;    /**< step 3: the five firmware images written over the NoC */
    double handshake_seconds; /**< steps 4-6: prime, release BRISC, wait for firmware ready */
    uint64_t image_bytes;     /**< bytes of ELF handed to the image loader */
} ft_tt_runtime_boot_timing;

/**
 * ft_tt_runtime_boot_core() that ADDS each step's host time to `timing` (NULL: no timing).
 * Same steps, same order, same return codes; the clock reads are host-only.
 */
int ft_tt_runtime_boot_core_timed(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  const ft_tt_runtime_images *firmware,
                                  ft_tt_runtime_boot_timing *timing);

/**
 * The kernel ELF file name `build_tensix_program.sh` writes for one processor, e.g.
 * "trisc0.elf". NULL for an out-of-range index.
 *
 * Exposed so that a caller reaching for a SIDECAR of that image -- its `.prov` record, say --
 * spells the name from the one table that produced it rather than a second copy.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
const char *ft_tt_runtime_kernel_image_name(unsigned processor);

#endif /* FT_TT_RUNTIME_H */
