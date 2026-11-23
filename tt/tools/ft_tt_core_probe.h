/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core_probe.h
 * @brief The health test a status read cannot be: boot firmware on one Tensix, run a real
 *   vector-add triad on it, and compare the result byte-for-byte against a C reference.
 *
 * **Why this exists.** A wedged Tensix answers every NoC read, so anything built only on
 * reads calls it healthy — that failure has happened on this project more than once (dev
 * notes §3). The discriminator is arithmetic: the triad exercises the five RISC-Vs, the
 * unpacker/FPU/packer backend and the NoC, and the two failures this card has produced —
 * a core that will not boot firmware, and a core whose backend was taken through reset —
 * both show up here as a timeout or a wrong tile.
 *
 * **What it writes.** Firmware images, kernel images, CB config, runtime args, two source
 * tiles and a wiped destination. All of it to the target core's own L1, and the kernels
 * are self-addressed, so the blast radius is one core.
 *
 * **What it refuses.** A core that is running a kernel. Booting is `assert_reset` first,
 * and asserting reset on a blocked kernel is exactly how a core gets wedged (dev notes
 * §1) — so the caller must pass a status read taken first, and a core with a live kernel
 * or a ticking resident queue is refused rather than probed.
 *
 * **What it leaves behind.** Firmware running and idle, waiting for a launch message —
 * NOT parked. Parking a core is what wedged two of the three cores this tool was written
 * for; `ft_tt_core_park()` is deliberately never called here, on any path.
 *
 * @status Runs on the p150a test card, and is shown to SEPARATE: in one invocation
 *   over cores 60-69 it failed the three known-wedged cores and passed all seven controls
 *   4/4 byte-exact.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_CORE_PROBE_H
#define FT_TT_CORE_PROBE_H

#include <stdbool.h>
#include <stdint.h>

#include "ft_tt_core_status.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_program_directory.h"

/** Where the probe's tiles and CB config land, and the reference it checks against. */
#define FT_TT_CORE_PROBE_TILE_COUNT 2u

/** Default repeats. One run can pass on a core that then fails; a handful is cheap. */
#define FT_TT_CORE_PROBE_DEFAULT_REPEATS 4u

/** The firmware and kernel images every probed core is given, read from ONE directory.
 *  A kernel is linked against its own firmware's exported symbols, so firmware from one
 *  build and a kernel from another is a wedged core, not a configuration. */
typedef struct ft_tt_core_probe_context {
    ft_tt_firmware_set firmware;
    ft_tt_program_directory program;
} ft_tt_core_probe_context;

/** What one core's probe established. */
typedef struct ft_tt_core_probe_result {
    uint32_t index;
    uint16_t x;
    uint16_t y;
    bool refused;              /**< the core was running a kernel; nothing was written */
    bool firmware_booted;
    bool program_loaded;
    bool completed;            /**< the triad reached RUN_MSG_DONE inside the timeout */
    unsigned repeats_requested;
    unsigned repeats_passed;
    unsigned first_bad_tile;   /**< FT_TT_CORE_PROBE_TILE_COUNT when nothing was wrong */
    unsigned first_bad_element;
    int failure_code;
    const char *failure_stage; /**< NULL when the probe passed */
} ft_tt_core_probe_result;

/** True only if every requested repeat came back byte-exact. */
bool ft_tt_core_probe_passed(const ft_tt_core_probe_result *result);

/**
 * Read the five firmware ELFs and the five kernel ELFs out of one build directory.
 *
 * @param directory  a build_tensix_program.sh output directory holding the vector_add
 *                   triad — brisc.elf is the reader, ncrisc.elf the writer, trisc*.elf
 *                   the compute thread, plus fw_*.elf
 * @param out        zeroed and filled; release with ft_tt_core_probe_context_close()
 * @return           FT_TT_OK, FT_TT_EIO if an ELF is missing, FT_TT_ENOMEM
 */
int ft_tt_core_probe_context_open(const char *directory, ft_tt_core_probe_context *out);

/** Free both image sets. Safe on NULL and on a partially opened context. */
void ft_tt_core_probe_context_close(ft_tt_core_probe_context *context);

/**
 * Probe one core: boot, load, run, compare, repeat.
 *
 * @param noc      device access
 * @param grid     live grid — the coordinate is resolved here, never passed in
 * @param context  images to use
 * @param status   a status read taken IMMEDIATELY BEFORE. A core with a running kernel
 *                 or a ticking resident queue is refused; see the file header
 * @param force    override the running-kernel refusal. Only defensible on a core ALREADY
 *                 known wedged, where the "kernel" the refusal protects is one that can
 *                 never finish and the state it protects is already lost
 * @param repeats  runs to demand, each with the destination wiped first
 * @param out      zeroed and filled
 * @return         FT_TT_OK when the probe ran to a verdict (pass or fail);
 *                 FT_TT_EPERM when it was refused; a negative code if it could not start
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_core_probe_run(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                         ft_tt_core_probe_context *context, const ft_tt_core_status *status,
                         unsigned repeats, bool force, ft_tt_core_probe_result *out);

/** One line: core, pass/fail, and the stage that failed. */
void ft_tt_core_probe_print_line(const ft_tt_core_probe_result *result);

#endif /* FT_TT_CORE_PROBE_H */
