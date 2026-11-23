/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_core.c
 * @brief `ft_tt_core` — target one Tensix, a list, a range or all of them: report what
 *   state they are in, prove whether they still compute, and climb a recovery ladder on
 *   the ones that do not.
 *
 * ---------------------------------------------------------------------------
 * READ-ONLY BY DEFAULT
 * ---------------------------------------------------------------------------
 * `status` and `topology` write nothing. `probe` and `recover` write to the card and are
 * refused without `--write`. Rung 5 resets a PCIe bus and needs its own second flag on
 * top. A tool that can wedge a core by accident is worse than no tool.
 *
 * ---------------------------------------------------------------------------
 * WHY A READ IS NOT A HEALTH CHECK
 * ---------------------------------------------------------------------------
 * A wedged Tensix answers NoC reads exactly like a healthy one. `status` therefore
 * reports what the registers SAY and names its verdicts so that none of them can be read
 * as "this core works" — a core with all five RISC-Vs held comes back "held in reset",
 * which is what a cold core and a wedged core both look like. `probe` is the
 * discriminator: it boots firmware and runs a real vector-add triad through the
 * unpacker, FPU and packer, and compares the result byte-for-byte.
 *
 * ---------------------------------------------------------------------------
 * USAGE
 * ---------------------------------------------------------------------------
 *   ft_tt_core status   <targets> [--detail]
 *   ft_tt_core probe    <targets> --write [--program DIR] [--repeats N] [--force]
 *   ft_tt_core recover  <targets> --rung N --write [--then-probe] [--program DIR] [--force]
 *   ft_tt_core topology [PCI-ADDRESS]
 *
 *   <targets>  60 | 60,62,65 | 60-69 | 60-62,65 | all
 *
 *   Rungs:  0 deassert the RISC-Vs      (one core, one register write)
 *           1 reload firmware and boot  (one core)
 *           2 poke what the core waits on (one core)
 *           3 ASIC reset                  (whole card)
 *           4 RESET_DEVICE ASIC_DMC_RESET (whole card)
 *           5 PCIe secondary bus reset    (whole card; needs
 *             --allow-pcie-secondary-bus-reset as well as --write)
 *
 * ---------------------------------------------------------------------------
 * THE RUNG IS THE BLAST RADIUS; `--driver` IS THE MECHANISM
 * ---------------------------------------------------------------------------
 * `--driver kmd` (the default) reaches rung 3 through tt-kmd's `RESET_DEVICE` ioctl and
 * completes it with a PCI remove + bridge rescan. `--driver udrv` does the SAME RUNG with no
 * vendor driver at all: three PCI config-space writes, our own config snapshot restored in
 * place, and a link retrain (`ft_tt_udrv_reset.h`).
 *
 * The ladder is NOT extended with new numbers for this. A rung number says what the operation
 * can destroy — that is what a caller has to reason about, and it is identical either way.
 * Numbering the userspace ASIC reset "6" would put it above the SBR, implying it is more
 * drastic than a bus reset, which is false. So the vocabulary fits, and `--driver` is
 * orthogonal to it. `--driver udrv` is only meaningful at rung 3: rung 4 needs two ARC
 * messages and there is no userspace ARC transport yet, and rung 5 is already pure sysfs.
 *
 * ---------------------------------------------------------------------------
 * WHY THE CARD RESET IS THE FLOOR — settled 2026-08-12, do not re-open
 * ---------------------------------------------------------------------------
 * Recovering ONE wedged core costs the whole card, and that is a property of the silicon,
 * not a missing feature here. Four things were checked, so nobody has to check them again:
 *
 * 1. Every wedge seen has been in the Tensix BACKEND — MATH stalled in the instruction
 *    FIFO, dvalid never satisfied, a boot handshake that never finishes. None of it is
 *    executing anything that could poll a flag, so rungs 0-2 are not incomplete, they are
 *    structurally out of reach (`docs/tt_bringup_log.md` §21.2).
 * 2. **The per-tile reset register is a QUASAR feature.** `SMN_SLV_TILE_RESET`
 *    (`TENSIX_RESET_N`, `NOC_RESET_N`, `OVERLAY_RESET_N`, `D2D_RESET_N`) exists only under
 *    `hw/inc/internal/tt-2xx/quasar/`. Blackhole is tt-1xx and has no `TILE_RESET` at all.
 * 3. The Tensix backend soft-reset bits are not a substitute. Firmware boot never programs
 *    the backend datapath — it sets reset PCs, and the stock sync init zeroes only
 *    CB tile counters — so the 14 cores corrupted that day already got a full boot AND a
 *    complete kernel LLK re-init, and still returned garbage until the machine was reset.
 * 4. PERST# alone does not do it: §21.4 measured a PCIe secondary bus reset leaving wedged
 *    cores byte-identical. Only `RESET_DEVICE ASIC_RESET` has ever cleared one, and the ARC
 *    exposes no per-tile granularity.
 *
 * So: check `status --detail` first, in case a wedge is ever NOT backend-resident and a
 * per-core rung can reach it. Otherwise rung 3 is the answer, and the advisory lock
 * (`ft_tt_card_lock.h`) is what makes it safe — it does not make the reset less
 * destructive, it makes it something other users see coming and refuse. Revisit on Quasar.
 *
 * @status Runs on the p150a test card. `probe all` is 140/140 on this card
 *   (2026-08-17). Replaces tests/tt_resident_recover.c, which was written for the core-60
 *   wedge, never ran, and did not build against the current ft_tt_launch.h.
 *   `--driver udrv` is on silicon as of 2026-08-17: it recovered core 101 from a deliberate
 *   0x00047000 wedge in place, no PCI remove and no rescan, and retrained the link from the
 *   2.5 GT/s the reset left behind.
 *
 *   **AND THE WHOLE TOOL NOW RUNS WITH NO VENDOR MODULE, which is the board it was built
 *   for (2026-08-17).** `--driver` selects the TRANSPORT for every command — status, probe
 *   and all six rungs — and not just the mechanism for a whole-card rung, and `udrv` means
 *   udrv even when a module happens to be loaded, so it can be compared against the oracle
 *   on the card host that has one. With `tenstorrent` unloaded: `status all` read 140 cores,
 *   `probe all --write` was **140 passed / 0 failed / 0 refused**, and
 *   `recover 71 --rung 3 --write` did the ASIC reset and read both states. The same probe
 *   over `--driver kmd` with no module exits 1, which is what makes the green mean
 *   something.
 *
 *   Three things were in the way and none of them was the driver. `--driver udrv` opened
 *   the chardev to print the before-state; it checked the root-only PCI remove/rescan
 *   permission that the in-place config restore never uses; and `probe` and rungs 0-2 took
 *   their per-core claim through the tt-kmd session's wrapper, when a claim is a flock
 *   under /tmp that has never needed a device handle.
 * @issues
 *   - `topology` and `--driver udrv` read and write PCI config space through sysfs, which is
 *     root-only, so both need sudo. The card lock and the per-core claim namespace live under
 *     /tmp and a sudo run leaves them ROOT-OWNED, which then refuses every non-root card job
 *     with FT_TT_EIO ("device I/O failed") — a message that reads like a card fault and is
 *     not one. Fix the ownership, do not diagnose the card.
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ft_tt_card_lock.h"
#include "ft_tt_core_probe.h"
#include "ft_tt_core_readiness.h"
#include "ft_tt_tensix_claim.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_claim.h"
#include "ft_tt_udrv_reset.h"
#include "ft_tt_core_recover.h"
#include "ft_tt_core_status.h"
#include "ft_tt_core_status_print.h"
#include "ft_tt_device_reset.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_runtime.h"

/** More than any p150 has cores; the real bound is the LIVE core count, checked below. */
#define TOOL_MAX_TARGETS 256u

#define TOOL_CHARDEV_PATH "/dev/tenstorrent/0"

/** Where sysfs publishes PCI devices. Both the card and its bridge are read from here. */
#define TOOL_PCI_DEVICES_DIR "/sys/bus/pci/devices"

/** Which mechanism a whole-card rung uses. The rung says WHAT is destroyed; this says HOW. */
typedef enum tool_reset_driver {
    TOOL_RESET_DRIVER_KMD = 0, /**< tt-kmd's RESET_DEVICE ioctl, then remove + rescan */
    TOOL_RESET_DRIVER_UDRV = 1 /**< three config writes, our own snapshot restored in place */
} tool_reset_driver;

typedef enum tool_command {
    TOOL_COMMAND_STATUS,
    TOOL_COMMAND_READINESS,
    TOOL_COMMAND_PROBE,
    TOOL_COMMAND_RECOVER,
    TOOL_COMMAND_TOPOLOGY,
    TOOL_COMMAND_CLAIM
} tool_command;

typedef struct tool_options {
    tool_command command;
    const char *target_text;
    bool allow_write;
    bool allow_secondary_bus_reset;
    bool detail;
    bool then_probe;
    bool force;
    unsigned rung;
    bool rung_given;
    tool_reset_driver reset_driver;
    bool udrv_reenumerate; /**< --driver udrv only: use remove + rescan, not in-place restore */
    unsigned repeats;
    const char *program_directory;
    const char *save_directory;
    /* Card-lock coordination (ft_tt_card_lock.h) -- only rungs 3-5 and `claim` use these. */
    const char *card_lock_reason;
    bool override_card_lock;
    /* Per-core coordination (ft_tt_tensix_claim.h) -- probe and rungs 0-2 use this. */
    bool override_core_claims;
    unsigned hold_seconds; /**< `claim` only; 0 = hold until Ctrl-C */
} tool_options;

typedef struct tool_targets {
    uint32_t index[TOOL_MAX_TARGETS];
    unsigned count;
    bool all;
} tool_targets;

static void print_usage(const char *program_name)
{
    fprintf(stderr,
            "usage:\n"
            "  %s status   <targets> [--detail]\n"
            "  %s readiness <targets>   may firmware be booted onto these cores RIGHT\n"
            "                           NOW? Read-only. The same reading Pantheon's\n"
            "                           arming path refuses on\n"
            "  %s probe    <targets> --write [--program DIR] [--repeats N] [--force]\n"
            "  %s recover  <targets> --rung N --write [--then-probe] [--program DIR]\n"
            "               [--driver kmd|udrv] [--udrv-reenumerate]\n"
            "               [--allow-pcie-secondary-bus-reset] [--save-dir DIR]\n"
            "               [--reason TEXT] [--override-card-lock]\n"
            "  %s topology [PCI-ADDRESS]   (no address = this card; an address makes the\n"
            "                               rung-5 refusal demonstrable on another device)\n"
            "  %s claim    --reason TEXT [--hold-seconds N]\n"
            "               register that this card is in use; holds a shared lock until\n"
            "               Ctrl-C (or N seconds), so `recover --rung 3/4/5` refuses and\n"
            "               names you instead of resetting the card out from under you\n"
            "\n"
            "  <targets>  60 | 60,62,65 | 60-69 | 60-62,65 | all\n"
            "  --force    probe or reload a core whose kernel is still 'running' — only\n"
            "             for a core already known wedged\n"
            "  --driver   which TRANSPORT reaches the card, and how a whole-card rung is\n"
            "             performed. It applies to EVERY command: status, probe, and all\n"
            "             six rungs. kmd (default) = tt-kmd's\n"
            "             RESET_DEVICE ioctl then a PCI remove + rescan. udrv = NO VENDOR\n"
            "             DRIVER: three PCI config writes, our own config snapshot restored\n"
            "             in place, and a link retrain. Rung 3 only. The rung number is the\n"
            "             blast radius and does not change with the mechanism. udrv works\n"
            "             with NO tt-kmd loaded: the before/after state is then read over\n"
            "             foundry's own driver, and the root-only PCI remove/rescan check\n"
            "             is skipped because the in-place restore does not use it.\n"
            "  --udrv-reenumerate  with --driver udrv, finish with remove + rescan instead\n"
            "             of the in-place config restore — the fallback, for comparison\n"
            "  rungs      0 deassert RISC-Vs · 1 reload firmware · 2 poke the wait\n"
            "             3 ASIC reset · 4 RESET_DEVICE ASIC_DMC_RESET\n"
            "             5 PCIe secondary bus reset (also needs "
            "--allow-pcie-secondary-bus-reset)\n"
            "             — rungs 3-5 reset the WHOLE CARD and take the card lock first;\n"
            "               --override-card-lock proceeds anyway and says loudly who it\n"
            "               overrode. Rungs 0-2 and probe touch ONE CORE EACH: they take a\n"
            "               per-core claim instead (ft_tt_tensix_claim.h) and refuse, naming\n"
            "               the holder, rather than driving someone else's core.\n"
            "  --override-core-claims  probe/rungs 0-2 proceed on a core another process\n"
            "               holds. It is a claim, not a fence — say why out loud first.\n",
            program_name, program_name, program_name, program_name, program_name,
            program_name);
}

/** Add one index, refusing a list longer than the array. */
static int add_target(tool_targets *targets, unsigned long value)
{
    if (targets->count >= TOOL_MAX_TARGETS) {
        fprintf(stderr, "too many targets (max %u)\n", TOOL_MAX_TARGETS);
        return FT_TT_EINVAL;
    }
    targets->index[targets->count++] = (uint32_t)value;
    return FT_TT_OK;
}

/** Parse "all", "60", "60,62", "60-69" or any comma-separated mix of those. */
static int parse_targets(const char *text, tool_targets *out)
{
    char buffer[512];
    char *saved = NULL;
    char *token;

    memset(out, 0, sizeof(*out));
    if (!text)
        return FT_TT_EINVAL;
    if (strcmp(text, "all") == 0) {
        out->all = true;
        return FT_TT_OK;
    }
    if (strlen(text) >= sizeof(buffer))
        return FT_TT_EINVAL;
    snprintf(buffer, sizeof(buffer), "%s", text);

    for (token = strtok_r(buffer, ",", &saved); token; token = strtok_r(NULL, ",", &saved)) {
        char *dash = strchr(token, '-');
        unsigned long first;
        unsigned long last;
        unsigned long value;

        if (dash) {
            *dash = '\0';
            first = strtoul(token, NULL, 10);
            last = strtoul(dash + 1, NULL, 10);
            if (last < first)
                return FT_TT_EINVAL;
            for (value = first; value <= last; value++)
                if (add_target(out, value) != FT_TT_OK)
                    return FT_TT_EINVAL;
        } else {
            if (token[0] == '\0')
                return FT_TT_EINVAL;
            if (add_target(out, strtoul(token, NULL, 10)) != FT_TT_OK)
                return FT_TT_EINVAL;
        }
    }
    return out->count == 0u ? FT_TT_EINVAL : FT_TT_OK;
}

/** Expand "all" and check every index against the LIVE core count, never a constant. */
static int resolve_targets(const ft_tt_grid_info *grid, tool_targets *targets)
{
    uint32_t live = ft_tt_coord_tensix_core_count(grid);
    unsigned entry;

    if (live == 0u) {
        fprintf(stderr, "the grid reports no enabled Tensix cores\n");
        return FT_TT_EIO;
    }
    if (targets->all) {
        uint32_t index;

        targets->count = 0;
        for (index = 0; index < live && targets->count < TOOL_MAX_TARGETS; index++)
            targets->index[targets->count++] = index;
        return FT_TT_OK;
    }
    for (entry = 0; entry < targets->count; entry++) {
        if (targets->index[entry] >= live) {
            fprintf(stderr, "Tensix %u is out of range: this card has %u enabled cores\n",
                    targets->index[entry], live);
            return FT_TT_EINVAL;
        }
    }
    return FT_TT_OK;
}

static int parse_options(int argc, char **argv, tool_options *out)
{
    int argument;
    /* Pre-scanned, not tracked through the loop, so the acknowledgement works whichever
     * side of --program it is written on. Order-dependent safety is not safety. */
    bool foreign_program_acknowledged = false;

    memset(out, 0, sizeof(*out));
    out->repeats = FT_TT_CORE_PROBE_DEFAULT_REPEATS;
    out->program_directory = ft_tt_firmware_directory();
    if (argc < 2)
        return FT_TT_EINVAL;

    for (argument = 2; argument < argc; argument++) {
        if (strcmp(argv[argument], "--i-know-this-must-be-a-vector-add-triad") == 0)
            foreign_program_acknowledged = true;
    }

    /* THE ENVIRONMENT REACHES THE SAME CODE PATH AS --program, AND GATING ONLY THE FLAG
     * WAS AN INCOMPLETE FIX. ft_tt_firmware_directory() returns
     * FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT when it is set, so an exported
     * FT_TT_FIRMWARE_DIR silently pointed probe at another op's program directory --
     * exactly the mixing the flag was gated to prevent. On 2026-08-13 a bisect script
     * exported it for its own builds, the shell then ran probe, and it wedged core 26 and
     * reported a false "boots but computes garbage". A safety check that covers one of two
     * routes to the same call is not a safety check. Other suites legitimately use this
     * variable, so the refusal belongs here in the tool rather than in the shared helper. */
    if (getenv(FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT) != NULL &&
        getenv(FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT)[0] != '\0' &&
        !foreign_program_acknowledged) {
        fprintf(stderr,
                "refused: %s is set in the environment (%s).\n"
                "  probe drives its OWN vector_add triad; pointing it at another op's\n"
                "  directory mixes those kernels with vector_add's runtime args and WEDGES\n"
                "  THE CORE -- and a wedged core needs a whole-card reset to recover.\n"
                "  Unset it, or pass --i-know-this-must-be-a-vector-add-triad if that\n"
                "  directory really is another build of the same triad.\n",
                FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT,
                getenv(FT_TT_FIRMWARE_DIRECTORY_ENVIRONMENT));
        return FT_TT_EINVAL;
    }

    for (argument = 2; argument < argc; argument++) {
        if (strcmp(argv[argument], "--i-know-this-must-be-a-vector-add-triad") == 0)
            foreign_program_acknowledged = true;
    }

    if (strcmp(argv[1], "status") == 0)
        out->command = TOOL_COMMAND_STATUS;
    else if (strcmp(argv[1], "readiness") == 0)
        out->command = TOOL_COMMAND_READINESS;
    else if (strcmp(argv[1], "probe") == 0)
        out->command = TOOL_COMMAND_PROBE;
    else if (strcmp(argv[1], "recover") == 0)
        out->command = TOOL_COMMAND_RECOVER;
    else if (strcmp(argv[1], "topology") == 0)
        out->command = TOOL_COMMAND_TOPOLOGY;
    else if (strcmp(argv[1], "claim") == 0)
        out->command = TOOL_COMMAND_CLAIM;
    else
        return FT_TT_EINVAL;

    for (argument = 2; argument < argc; argument++) {
        const char *option = argv[argument];
        bool has_value = argument + 1 < argc;

        if (option[0] != '-') {
            if (out->target_text)
                return FT_TT_EINVAL;
            out->target_text = option;
        } else if (strcmp(option, "--write") == 0) {
            out->allow_write = true;
        } else if (strcmp(option, "--allow-pcie-secondary-bus-reset") == 0) {
            out->allow_secondary_bus_reset = true;
        } else if (strcmp(option, "--detail") == 0) {
            out->detail = true;
        } else if (strcmp(option, "--force") == 0) {
            out->force = true;
        } else if (strcmp(option, "--then-probe") == 0) {
            out->then_probe = true;
        } else if (strcmp(option, "--rung") == 0 && has_value) {
            out->rung = (unsigned)strtoul(argv[++argument], NULL, 10);
            out->rung_given = true;
        } else if (strcmp(option, "--repeats") == 0 && has_value) {
            out->repeats = (unsigned)strtoul(argv[++argument], NULL, 10);
        } else if (strcmp(option, "--program") == 0 && has_value) {
            /* WEDGES CORES IF MISUSED, so it is gated. probe drives ITS OWN vector_add
             * triad — its runtime args, CB configuration and processor mask (0x1f, all
             * five). Pointing it at some other op's program directory boots that
             * directory's firmware and loads that directory's kernels, then drives them
             * with vector_add's expectations. The boot handshake never finishes and the
             * core is wedged until a card reset. It cost Tensix 30 and 31 on 2026-08-12.
             *
             * A file-existence check cannot catch this: every program directory holds the
             * same file names. The only valid --program is a DIFFERENT BUILD OF THE SAME
             * vector_add triad, which the tool cannot verify, so the caller asserts it. */
            if (!foreign_program_acknowledged) {
                fprintf(stderr,
                        "--program needs --i-know-this-must-be-a-vector-add-triad.\n"
                        "  probe drives its own vector_add triad; pointing it at another\n"
                        "  op's directory mixes those kernels with vector_add's runtime\n"
                        "  args and WEDGES THE CORE (cost Tensix 30 and 31, 2026-08-12).\n"
                        "  The only valid use is another build of the SAME triad.\n"
                        "  To health-check a core, run probe with no --program.\n");
                return FT_TT_EINVAL;
            }
            out->program_directory = argv[++argument];
        } else if (strcmp(option, "--i-know-this-must-be-a-vector-add-triad") == 0) {
            foreign_program_acknowledged = true;
        } else if (strcmp(option, "--save-dir") == 0 && has_value) {
            out->save_directory = argv[++argument];
        } else if (strcmp(option, "--reason") == 0 && has_value) {
            out->card_lock_reason = argv[++argument];
        } else if (strcmp(option, "--override-card-lock") == 0) {
            out->override_card_lock = true;
        } else if (strcmp(option, "--override-core-claims") == 0) {
            out->override_core_claims = true;
        } else if (strcmp(option, "--driver") == 0 && has_value) {
            const char *name = argv[++argument];

            if (strcmp(name, "kmd") == 0) {
                out->reset_driver = TOOL_RESET_DRIVER_KMD;
            } else if (strcmp(name, "udrv") == 0) {
                out->reset_driver = TOOL_RESET_DRIVER_UDRV;
            } else {
                fprintf(stderr, "--driver must be kmd or udrv\n");
                return FT_TT_EINVAL;
            }
        } else if (strcmp(option, "--udrv-reenumerate") == 0) {
            out->udrv_reenumerate = true;
        } else if (strcmp(option, "--hold-seconds") == 0 && has_value) {
            out->hold_seconds = (unsigned)strtoul(argv[++argument], NULL, 10);
        } else {
            fprintf(stderr, "unknown option: %s\n", option);
            return FT_TT_EINVAL;
        }
    }
    if (out->command != TOOL_COMMAND_TOPOLOGY && out->command != TOOL_COMMAND_CLAIM &&
        !out->target_text)
        return FT_TT_EINVAL;
    if (out->repeats == 0u)
        return FT_TT_EINVAL;
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* readiness                                                                 */
/* ------------------------------------------------------------------------- */

/**
 * May firmware be booted onto these cores RIGHT NOW?
 *
 * READ-ONLY, like `status`, and for the same reason: a NoC read of a wedged core is
 * indistinguishable from a read of a healthy one and changes neither. So it goes through the
 * transport seam rather than the tt-kmd session and needs no claim.
 *
 * This is the SAME reading Pantheon's arming path refuses on (`pn_tt_core_boot_guard.h`), so
 * running it here answers "why did that serve refuse core 90" without loading anything, and
 * — the reason it was added — it can be pointed at a core carrying a REAL leftover to see
 * whether the detector sees it, with nothing written to the card either way.
 */
static int run_readiness(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                         const tool_targets *targets)
{
    static const char *const processor_name[FT_TT_PROC_COUNT] = {"BRISC", "NCRISC", "TRISC0",
                                                                 "TRISC1", "TRISC2"};
    unsigned not_ready = 0;
    unsigned entry;

    for (entry = 0; entry < targets->count; entry++) {
        ft_tt_core_readiness readiness;
        unsigned processor;

        if (ft_tt_core_readiness_read(noc, grid, targets->index[entry], &readiness) !=
            FT_TT_OK) {
            fprintf(stderr, "Tensix %u does not resolve to a coordinate\n",
                    targets->index[entry]);
            continue;
        }
        printf("Tensix %u — translated NoC (%u, %u)\n", readiness.status.index,
               readiness.status.x, readiness.status.y);
        printf("  %-10s %s\n", readiness.ready ? "READY" : "NOT READY",
               ft_tt_core_readiness_string(readiness.state));
        printf("  verdict '%s'  reset 0x%08x  instrn_buf 0x%08x  go 0x%02x\n",
               ft_tt_core_verdict_string(readiness.status.verdict),
               readiness.status.state.reset_register,
               readiness.status.state.instrn_buf_status,
               readiness.status.state.go_messages[0].signal);
        /* The CONTROL heartbeat and the PER-PROCESSOR ones printed side by side, because
         * the whole reason this command exists rather than `status` alone is that they
         * disagree in exactly the failure being guarded against: the reader has exited, so
         * the control block is frozen while a TRISC is still spinning. */
        printf("  resident queue %s, control heartbeat %s\n",
               readiness.status.queue_present ? "present" : "absent",
               readiness.status.queue_heartbeat_ticking ? "TICKING" : "frozen");
        for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
            if (!readiness.processor[processor].slot_read)
                continue;
            printf("    %-7s state %s  heartbeat %u  %s\n", processor_name[processor],
                   ft_tt_queue_state_string(readiness.processor[processor].state),
                   readiness.processor[processor].heartbeat,
                   readiness.processor[processor].ticking ? "TICKING" : "frozen");
        }
        if (!readiness.ready)
            printf("  waiting %s help\n", readiness.waiting_can_help ? "CAN" : "cannot");
        if (!readiness.ready)
            not_ready++;
    }
    printf("\n%u core%s, %u not ready\n", targets->count, targets->count == 1u ? "" : "s",
           not_ready);
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* status                                                                    */
/* ------------------------------------------------------------------------- */

/** Every target's state, over whichever transport opened the card.
 *
 *  Accessor + grid, not a session: `status` is READ-ONLY — a NoC read of a wedged core is
 *  indistinguishable from a read of a healthy one and changes neither — so it is the one
 *  command that can run on a board with no vendor module and no claim, and
 *  `instrn_buf_status` (0x66 wedged / 0x77 healthy) read this way is the WHOLE health
 *  instrument there. */
static int run_status(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                      const tool_targets *targets, bool detail)
{
    unsigned verdict_total[FT_TT_CORE_VERDICT_LAST + 1u];
    unsigned entry;
    unsigned verdict;

    memset(verdict_total, 0, sizeof(verdict_total));
    if (!detail)
        ft_tt_core_status_print_header();

    for (entry = 0; entry < targets->count; entry++) {
        ft_tt_core_status status;

        if (ft_tt_core_status_read(noc, grid, targets->index[entry], &status) != FT_TT_OK) {
            fprintf(stderr, "Tensix %u does not resolve to a coordinate\n", targets->index[entry]);
            continue;
        }
        if (detail)
            ft_tt_core_status_print_detail(&status);
        else
            ft_tt_core_status_print_line(&status);
        verdict_total[status.verdict]++;
    }

    printf("\n%u core%s:", targets->count, targets->count == 1u ? "" : "s");
    for (verdict = 0; verdict <= (unsigned)FT_TT_CORE_VERDICT_LAST; verdict++)
        if (verdict_total[verdict] != 0u)
            printf("  %u %s", verdict_total[verdict],
                   ft_tt_core_verdict_string((ft_tt_core_verdict)verdict));
    printf("\n");
    printf("(a read cannot tell a healthy parked core from a wedged one — that is what "
           "`probe` is for)\n");
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* claim                                                                     */
/* ------------------------------------------------------------------------- */

/** set by the signal handler, read by the wait loop; nothing else touches it, so a plain
 *  sig_atomic_t is enough without needing signal-safe cleanup inside the handler itself. */
static volatile sig_atomic_t g_claim_stop_requested = 0;

static void handle_claim_stop_signal(int signal_number)
{
    (void)signal_number;
    g_claim_stop_requested = 1;
}

/** Register a shared claim and hold it until Ctrl-C (or --hold-seconds runs out). Touches
 *  no card state -- this only ever writes to /tmp -- so it needs no --write and cannot be
 *  refused by the read-only-by-default rule. */
static int run_claim(const tool_options *options)
{
    struct sigaction action;
    const char *reason = options->card_lock_reason ? options->card_lock_reason
                                                    : "(no reason given)";
    int fd = -1;
    unsigned waited_seconds = 0;
    int rc = ft_tt_card_lock_claim_acquire(options->card_lock_reason, &fd);

    if (rc != FT_TT_OK) {
        fprintf(stderr, "claim: could not register (rc=%d)\n", rc);
        return rc;
    }
    printf("claim registered: pid %d, reason \"%s\"\n", (int)getpid(), reason);
    if (options->hold_seconds != 0u)
        printf("  held for up to %u second%s, or Ctrl-C to release early\n",
               options->hold_seconds, options->hold_seconds == 1u ? "" : "s");
    else
        printf("  held until Ctrl-C -- a card-level reset (rung 3/4/5) will refuse and "
               "name this claim while it runs\n");
    fflush(stdout);

    memset(&action, 0, sizeof(action));
    action.sa_handler = handle_claim_stop_signal;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);

    while (!g_claim_stop_requested) {
        if (options->hold_seconds != 0u && waited_seconds >= options->hold_seconds)
            break;
        sleep(1u);
        waited_seconds++;
    }

    ft_tt_card_lock_release(fd);
    printf("claim released\n");
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* probe                                                                     */
/* ------------------------------------------------------------------------- */

static int run_probe(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                     const tool_targets *targets,
                     const tool_options *options)
{
    ft_tt_core_probe_context context;
    unsigned passed = 0;
    unsigned failed = 0;
    unsigned refused = 0;
    unsigned entry;
    int rc;

    rc = ft_tt_core_probe_context_open(options->program_directory, &context);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "cannot read firmware and kernels from %s (rc=%d)\n",
                options->program_directory, rc);
        return rc;
    }
    printf("probe: firmware and vector_add triad from %s, %u repeat%s per core\n",
           options->program_directory, options->repeats, options->repeats == 1u ? "" : "s");

    for (entry = 0; entry < targets->count; entry++) {
        ft_tt_core_probe_result result;
        ft_tt_core_status status;

        if (ft_tt_core_status_read(noc, grid, targets->index[entry], &status) != FT_TT_OK) {
            fprintf(stderr, "Tensix %u does not resolve to a coordinate\n", targets->index[entry]);
            continue;
        }
        (void)ft_tt_core_probe_run(noc, grid, &context, &status,
                                   options->repeats, options->force, &result);
        ft_tt_core_probe_print_line(&result);
        fflush(stdout);
        if (result.refused)
            refused++;
        else if (ft_tt_core_probe_passed(&result))
            passed++;
        else
            failed++;
    }
    ft_tt_core_probe_context_close(&context);

    printf("\nprobe: %u passed, %u failed, %u refused, of %u core%s\n", passed, failed, refused,
           targets->count, targets->count == 1u ? "" : "s");
    return failed == 0u ? FT_TT_OK : FT_TT_ERROR;
}

/* ------------------------------------------------------------------------- */
/* recover                                                                   */
/* ------------------------------------------------------------------------- */

/** One core through one of the core-level rungs, with a before and after status. */
static void recover_one_core(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                             uint32_t index,
                             const tool_options *options, const ft_tt_firmware_set *firmware)
{
    ft_tt_core_recovery_report report;
    ft_tt_core_status before;
    ft_tt_core_status after;

    if (ft_tt_core_status_read(noc, grid, index, &before) != FT_TT_OK) {
        fprintf(stderr, "Tensix %u does not resolve to a coordinate\n", index);
        return;
    }
    printf("--- before ---\n");
    ft_tt_core_status_print_detail(&before);

    switch ((ft_tt_core_recovery_rung)options->rung) {
    case FT_TT_RECOVERY_RUNG_DEASSERT_RISCS:
        (void)ft_tt_core_recover_deassert_riscs(noc, &before, &report);
        break;
    case FT_TT_RECOVERY_RUNG_RELOAD_FIRMWARE:
        (void)ft_tt_core_recover_reload_firmware(noc, &before, firmware,
                                                 options->force, &report);
        break;
    case FT_TT_RECOVERY_RUNG_POKE_WAIT:
    default:
        (void)ft_tt_core_recover_poke_wait(noc, &before, &report);
        break;
    }
    printf("--- %s ---\n", ft_tt_core_recovery_rung_string((ft_tt_core_recovery_rung)options->rung));
    ft_tt_core_recovery_print(&report);

    printf("--- after ---\n");
    if (ft_tt_core_status_read(noc, grid, index, &after) == FT_TT_OK)
        ft_tt_core_status_print_detail(&after);
    fflush(stdout);
}

/** Probe every target and print the result, used by `--then-probe`.
 *
 *  Takes the ACCESSOR and the GRID rather than a session, because a card-level
 *  rung may be driven on a machine where the vendor chardev does not exist and
 *  the accessor came from foundry's own driver instead. Nothing in here cares
 *  which — that is the whole point of the seam. */
static void probe_targets(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                          const tool_targets *targets, const tool_options *options)
{
    ft_tt_core_probe_context context;
    unsigned entry;

    if (ft_tt_core_probe_context_open(options->program_directory, &context) != FT_TT_OK) {
        fprintf(stderr, "--then-probe: cannot read images from %s\n", options->program_directory);
        return;
    }
    printf("\n--- probe after the rung ---\n");
    for (entry = 0; entry < targets->count; entry++) {
        ft_tt_core_probe_result result;
        ft_tt_core_status status;

        if (ft_tt_core_status_read(noc, grid, targets->index[entry], &status) != FT_TT_OK)
            continue;
        (void)ft_tt_core_probe_run(noc, grid, &context, &status, options->repeats,
                                   options->force, &result);
        ft_tt_core_probe_print_line(&result);
        fflush(stdout);
    }
    ft_tt_core_probe_context_close(&context);
}

/** Rungs 0-2: one core at a time, over the NoC. */
static int run_core_recovery(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                             const tool_targets *targets, const tool_options *options)
{
    ft_tt_firmware_set firmware;
    bool firmware_loaded = false;
    unsigned entry;

    if (options->rung == FT_TT_RECOVERY_RUNG_RELOAD_FIRMWARE) {
        if (ft_tt_firmware_set_load(options->program_directory, &firmware) != FT_TT_OK) {
            fprintf(stderr, "rung 1 needs firmware images; %s did not provide them\n",
                    options->program_directory);
            return FT_TT_EIO;
        }
        firmware_loaded = true;
    }
    for (entry = 0; entry < targets->count; entry++)
        recover_one_core(noc, grid, targets->index[entry], options,
                         firmware_loaded ? &firmware : NULL);
    if (firmware_loaded)
        ft_tt_firmware_set_release(&firmware);
    if (options->then_probe)
        probe_targets(noc, grid, targets, options);
    return FT_TT_OK;
}

/** The state of every target, printed before a whole-card rung erases it.
 *
 *  Accessor + grid, not a session, for the same reason probe_targets() is: with
 *  tt-kmd unloaded `instrn_buf_status` read over a udrv NoC path is the WHOLE
 *  health instrument (0x66 wedged / 0x77 healthy), and it is carried in exactly
 *  this struct. */
static void print_targets_status(const ft_tt_noc_access *noc, const ft_tt_grid_info *grid,
                                 const tool_targets *targets, const char *title)
{
    unsigned entry;

    printf("--- %s ---\n", title);
    for (entry = 0; entry < targets->count; entry++) {
        ft_tt_core_status status;

        if (ft_tt_core_status_read(noc, grid, targets->index[entry], &status) == FT_TT_OK)
            ft_tt_core_status_print_detail(&status);
    }
    fflush(stdout);
}

/* ------------------------------------------------------------------------- */
/* main                                                                      */
/* ------------------------------------------------------------------------- */

/**
 * The card as a whole-card rung sees it: EITHER a tt-kmd hardware session OR,
 * when there is no vendor module, an ft_tt_runtime over foundry's own driver.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS, AND WHY IT IS NOT A FORK
 * ---------------------------------------------------------------------------
 * `--driver udrv` was built for a board with NO tt-kmd, and until 2026-08-17 it
 * opened the chardev anyway — to print the before-state — so it could not run on
 * the one board it exists for. The reset library itself was always fine; the
 * tool around it was not.
 *
 * Everything a card-level rung does with the card is `ft_tt_core_status_read()`
 * and `ft_tt_core_probe_run()`, and both take an `ft_tt_noc_access` and a
 * `ft_tt_grid_info`. So this struct holds whichever thing produced those two
 * and hands out pointers to them; NOTHING below it is duplicated or branched.
 * With the module gone `instrn_buf_status` (0x66 wedged / 0x77 healthy) read
 * over a udrv path is the whole health instrument, and it arrives through
 * exactly this accessor.
 */
typedef struct tool_card {
    ft_tt_hardware_session kmd;
    bool kmd_open;
    ft_tt_runtime udrv;
    bool udrv_open;
    const ft_tt_noc_access *noc;  /**< NULL when nothing is open */
    const ft_tt_grid_info *grid;
    char address[FT_TT_PCI_ADDRESS_MAX]; /**< the claim namespace's card identity */
    /** Per-core claims this invocation holds. Taken through ft_tt_tensix_claim
     *  DIRECTLY rather than through the tt-kmd session's wrapper: a claim is a
     *  flock under /tmp and has never needed a device handle, and routing it
     *  through the session is what tied `probe` and rungs 0-2 to a chardev that
     *  the target board will not have. */
    ft_tt_tensix_claim_set *cores;
} tool_card;

/** Open, resolve targets, and hand back the card's PCI address for the device rungs. */
static int open_session(ft_tt_hardware_session *session, tool_targets *targets,
                        char *out_card_address, size_t card_address_bytes)
{
    int rc = ft_tt_hardware_session_open(session);

    if (rc != FT_TT_OK)
        return rc;
    rc = resolve_targets(&session->grid, targets);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(session);
        return rc;
    }
    /* Through the runtime, which answers on either transport — the session no
     * longer necessarily holds a tt-kmd context, and the address is what the
     * per-core claim namespace is named after. */
    if (out_card_address)
        (void)ft_tt_runtime_pci_address(&session->runtime, out_card_address, card_address_bytes);
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* per-core claims (ft_tt_tensix_claim.h)                                    */
/* ------------------------------------------------------------------------- */

/**
 * Claim every target before touching any of them.
 *
 * `probe` and rungs 0-2 write to ONE CORE EACH -- that core's L1, mailbox and
 * soft-reset register -- so they belong on a per-core claim, not the whole-card
 * lock. The card lock stays where it was, on the rungs that reset the card.
 *
 * Straight to `ft_tt_tensix_claim`, NOT through the tt-kmd session's wrapper. A
 * claim has always been a flock under /tmp and has never needed a device handle;
 * going through the session is the only reason `probe` and rungs 0-2 required a
 * chardev, which is precisely what the board this driver exists for will not
 * have.
 *
 * @return FT_TT_OK (also under an override, where nothing was claimed), or the
 *         refusal code with nothing written to the card
 */
static int claim_targets_or_explain(tool_card *card, const tool_targets *targets,
                                    const tool_options *options, const char *what)
{
    const char *reason = options->card_lock_reason ? options->card_lock_reason : what;
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    int rc;

    memset(&request, 0, sizeof(request));
    /* NULL card reads the environment; ours is the address the open resolved, so
     * two invocations on one box cannot land in different namespaces. */
    request.card = card->address[0] ? card->address : NULL;
    request.indices = targets->index;
    request.count = targets->count;
    request.reason = reason;

    rc = ft_tt_tensix_claim_acquire(&request, &card->cores, &refusal);
    if (rc == FT_TT_UDRV_OK)
        return FT_TT_OK;
    ft_tt_tensix_claim_refusal_print(&refusal);
    card->cores = NULL;
    /* FT_TT_UDRV_EAGAIN and FT_TT_EPERM are the same number by design
     * (ft_tt_udrv.h), but the caller's table is ft_tt_device.h's, so name it. */
    rc = (rc == FT_TT_UDRV_EAGAIN) ? FT_TT_EPERM : rc;

    /* The refusal itself was already printed, naming the holder's pid and reason
     * (ft_tt_tensix_claim_refusal_print). All this adds is what to do next. */
    if (!options->override_core_claims) {
        fprintf(stderr,
                "  %s writes to each core it targets, so it will not drive a core somebody\n"
                "  else owns. Nothing has been written. Re-run with --override-core-claims\n"
                "  only if you know that process is finished with it.\n",
                what);
        return rc;
    }
    fprintf(stderr,
            "--override-core-claims: PROCEEDING on a core another process holds. If it is "
            "still driving that core, both of you will get wrong answers.\n");
    return FT_TT_OK;
}

/**
 * Name every core-claim holder a whole-card reset is about to destroy.
 *
 * The card lock only ever saw processes that registered a SHARED CLAIM through
 * `ft_tt_core claim`. Now that ordinary work holds per-core claims instead, a
 * reset that consulted only the card lock would report "nobody is using the
 * card" while wiping eight live suites. The census closes that: the whole-card
 * operation reads the per-core namespace too.
 *
 * @return how many cores are held, or 0 when the namespace cannot be read
 */
static unsigned report_core_claim_holders(const char *card_address)
{
    ft_tt_udrv_claim_namespace ns;
    ft_tt_udrv_claim_info *info;
    uint32_t held = 0u;
    uint32_t index;

    /* Resolved through the claim module, not with NULL and a literal, so this
     * census cannot end up reading a different directory than the claimants
     * write to -- that mismatch would report "nobody is using the card" and be
     * believed. */
    if (ft_tt_udrv_claim_namespace_init(&ns, ft_tt_tensix_claim_resolve_root(NULL),
                                        ft_tt_tensix_claim_resolve_card(card_address),
                                        FT_TT_TENSIX_CLAIM_NAMESPACE,
                                        FT_TT_TENSIX_CLAIM_INDEX_COUNT) != FT_TT_UDRV_OK)
        return 0u;

    info = calloc((size_t)ns.count, sizeof(*info));
    if (!info)
        return 0u;

    if (ft_tt_udrv_claim_scan(&ns, info, (size_t)ns.count, &held) != FT_TT_UDRV_OK || held == 0u) {
        free(info);
        return 0u;
    }

    fprintf(stderr, "  %u Tensix core%s are claimed by live processes:\n", held,
            held == 1u ? "" : "s");
    for (index = 0u; index < ns.count; index++) {
        if (!info[index].held)
            continue;
        fprintf(stderr, "    core %-3u  pid %-7ld  %s\n", index, (long)info[index].pid,
                info[index].reason[0] != '\0' ? info[index].reason : "(no reason recorded)");
    }
    free(info);
    return (unsigned)held;
}

/** Take the exclusive card lock, or explain the refusal / print the override banner.
 *  @return FT_TT_OK with `*out_fd` holding the lock (may be -1 under a granted override --
 *          nothing was acquired, so there is nothing to release), or a refusal code with
 *          `*out_fd` at -1 and nothing written to the card. */
static int acquire_card_lock_or_explain(const tool_options *options, int *out_fd)
{
    ft_tt_card_lock_status lock_status;
    int rc = ft_tt_card_lock_exclusive_try(options->card_lock_reason, out_fd, &lock_status);

    if (rc == FT_TT_OK)
        return FT_TT_OK;

    /* CONTENDED and BROKEN are different failures and must not share a message. EPERM
     * means somebody holds it and lock_status names them; anything else means the lock
     * could not be consulted AT ALL, and claiming contention there sends the reader
     * hunting a holder that does not exist. MEASURED on the card host: fs.protected_regular=2
     * stopped root open()ing the lock, and this branch reported a phantom holder. */
    if (rc != FT_TT_EPERM) {
        fprintf(stderr,
                "refused: rung %u resets the whole card, and the card lock could NOT BE "
                "CONSULTED (%s).\n"
                "  This is NOT contention. The lock file could not be opened, so whether\n"
                "  anyone else is using the card is UNKNOWN -- refusing rather than\n"
                "  guessing. Nothing has been reset. Re-run with --override-card-lock only\n"
                "  if you are certain the card is yours alone.\n",
                options->rung, ft_tt_error_string(rc));
        if (!options->override_card_lock)
            return rc;
    } else if (!options->override_card_lock) {
        fprintf(stderr,
                "refused: rung %u resets the whole card, and another process already "
                "holds the card lock:\n",
                options->rung);
        ft_tt_card_lock_status_print(&lock_status);
        fprintf(stderr,
                "  Wait for it to finish or coordinate with whoever it is (`ft_tt_core "
                "claim` names them here). If you are certain it is safe to proceed "
                "anyway, re-run with --override-card-lock. Nothing has been reset.\n");
        return FT_TT_EPERM;
    }

    fprintf(stderr,
            "--override-card-lock: PROCEEDING WITHOUT THE LOCK. Rung %u resets the whole "
            "card and will destroy the in-flight work of:\n",
            options->rung);
    ft_tt_card_lock_status_print(&lock_status);
    *out_fd = -1; /* nothing was acquired under an override; nothing to release later */
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* rung 3 with NO VENDOR DRIVER (ft_tt_udrv_reset.h)                          */
/* ------------------------------------------------------------------------- */

/** "/sys/bus/pci/devices/0000:12:00.0" from "0000:12:00.0". */
static int pci_device_dir(const char *bdf, char *out, size_t out_bytes)
{
    int written = snprintf(out, out_bytes, "%s/%s", TOOL_PCI_DEVICES_DIR, bdf);

    return (written > 0 && (size_t)written < out_bytes) ? FT_TT_OK : FT_TT_EINVAL;
}

/** One line naming a device's negotiated link, or why it could not be read. */
static void print_link_status(const char *label, const char *device_dir)
{
    ft_tt_udrv_link_status status;

    if (ft_tt_udrv_link_status_read(device_dir, &status) != FT_TT_UDRV_OK) {
        printf("  %-8s link: could not be read\n", label);
        return;
    }
    printf("  %-8s link: %s x%u%s\n", label, ft_tt_udrv_link_speed_string(status.speed_code),
           status.width, status.training ? " (TRAINING)" : "");
}

/**
 * The link comes back at Gen1 after every card-level reset, and every host-involved timing
 * then silently halves. Retrain it on the BRIDGE, and only when the topology
 * check says the bridge has exactly one child and that child is this card — retraining a
 * bridge with K80s under it is not ours to do.
 */
static void retrain_link_if_downgraded(const char *card_dir, const char *card_bdf)
{
    ft_tt_udrv_link_retrain_report retrain;
    ft_tt_udrv_link_status status;
    ft_tt_pcie_topology topology;
    char bridge_dir[FT_TT_UDRV_RESET_PATH_MAX];

    if (ft_tt_udrv_link_status_read(card_dir, &status) != FT_TT_UDRV_OK) {
        printf("link: the card's LNKSTA could not be read; NOT retraining\n");
        return;
    }
    if (status.speed_code >= FT_TT_UDRV_LINK_SPEED_5GT) {
        printf("link: already %s x%u — no retrain needed\n",
               ft_tt_udrv_link_speed_string(status.speed_code), status.width);
        return;
    }
    if (ft_tt_pcie_topology_resolve(card_bdf, &topology) != FT_TT_OK || !topology.safe) {
        printf("link: at %s but the bridge topology is not exclusively this card (%s) — "
               "NOT retraining\n",
               ft_tt_udrv_link_speed_string(status.speed_code),
               topology.refusal ? topology.refusal : "unresolved");
        return;
    }
    if (pci_device_dir(topology.bridge, bridge_dir, sizeof(bridge_dir)) != FT_TT_OK)
        return;

    printf("link: %s — retraining on bridge %s\n",
           ft_tt_udrv_link_speed_string(status.speed_code), topology.bridge);
    (void)ft_tt_udrv_link_retrain(bridge_dir, 2000u, &retrain);
    ft_tt_udrv_link_retrain_print(&retrain);
    print_link_status("card", card_dir);
    print_link_status("bridge", bridge_dir);
}

/**
 * Rung 3 through the userspace driver: snapshot config, three config writes, wait for the
 * card to clear its own reset marker, write config back, retrain the link.
 *
 * The caller has already taken the card lock, censused the per-core claims, printed the
 * before state and CLOSED the session — this function must not be reached with an aperture
 * mapped, which is the discipline that replaces `tenstorrent_vma_zap`.
 *
 * @param card_bdf  "0000:12:00.0"
 * @param options   `udrv_reenumerate` selects the remove + rescan fallback instead
 * @return          FT_TT_OK once the card is back and holds its config again
 */
static int run_udrv_asic_reset(const char *card_bdf, const tool_options *options)
{
    ft_tt_udrv_config_restore_report restore;
    ft_tt_udrv_config_snapshot snapshot;
    ft_tt_udrv_reset_wait_report wait;
    ft_tt_udrv_reset_report reset;
    char card_dir[FT_TT_UDRV_RESET_PATH_MAX];
    int rc = pci_device_dir(card_bdf, card_dir, sizeof(card_dir));

    if (rc != FT_TT_OK)
        return rc;

    printf("\n--- rung 3, NO VENDOR DRIVER: %s ---\n", card_dir);
    print_link_status("card", card_dir);

    /* The snapshot is taken BEFORE the marker is set, so the saved command register has the
     * marker clear and restoring it cannot re-arm what the card just finished clearing. */
    rc = ft_tt_udrv_config_snapshot_take(card_dir, &snapshot);
    if (rc != FT_TT_UDRV_OK) {
        fprintf(stderr,
                "refused: could not snapshot PCI config space (%d). Without it an in-place\n"
                "  ASIC reset leaves the BARs and the command register at zero and the card\n"
                "  undecodable. Nothing has been reset. Re-run under sudo if %s/config is\n"
                "  not writable.\n",
                rc, card_dir);
        return FT_TT_EPERM;
    }
    printf("config snapshot: vendor 0x%04x device 0x%04x, PCI Express capability at 0x%02x "
           "version %u, BAR0 0x%08x, command 0x%04x\n",
           snapshot.vendor_id, snapshot.device_id, snapshot.pcie_capability_offset,
           snapshot.pcie_capability_version, snapshot.header[4],
           (unsigned)(snapshot.header[1] & 0xFFFFu));

    rc = ft_tt_udrv_reset_asic(card_dir, &reset);
    if (rc != FT_TT_UDRV_OK) {
        ft_tt_udrv_reset_print(&reset, NULL);
        fprintf(stderr, "the ASIC reset did not complete its three writes\n");
        return FT_TT_EIO;
    }

    /* WAIT FOR THE MARKER, then restore. Restoring while the marker is still set is the bug
     * that made tt-kmd's in-place path look impossible (ft_tt_udrv_reset.h @status). */
    rc = ft_tt_udrv_reset_wait_ready(card_dir, snapshot.vendor_id, 15000u, &wait);
    ft_tt_udrv_reset_print(&reset, &wait);
    if (rc != FT_TT_UDRV_OK) {
        fprintf(stderr,
                "the card did not come back within 15 s. It is reset and NOT restored;\n"
                "  `echo 1 > %s/remove` then `echo 1 > /sys/bus/pci/rescan` is the recovery.\n",
                card_dir);
        return FT_TT_EIO;
    }

    if (options->udrv_reenumerate) {
        ft_tt_pcie_reenumeration_report reenumeration;

        printf("--udrv-reenumerate: finishing with remove + rescan instead of the in-place "
               "restore\n");
        rc = ft_tt_pcie_reenumerate(card_bdf, &reenumeration);
        ft_tt_pcie_reenumeration_print(&reenumeration);
        if (rc != FT_TT_OK)
            return rc;
    } else {
        rc = ft_tt_udrv_config_snapshot_restore(card_dir, &snapshot, &restore);
        ft_tt_udrv_config_restore_print(&restore);
        if (rc != FT_TT_UDRV_OK) {
            fprintf(stderr, "the config restore did not land; the card is reset and "
                            "undecodable. Recover with remove + rescan.\n");
            return FT_TT_EIO;
        }
    }

    retrain_link_if_downgraded(card_dir, card_bdf);
    return FT_TT_OK;
}

static void close_card(tool_card *card);

/**
 * Open the card over the transport `--driver` names — and ONLY that one.
 *
 * `--driver kmd` (the default) is the vendor chardev, exactly as before.
 * `--driver udrv` is foundry's own driver and NEVER the chardev, whether or not
 * a module happens to be loaded. Two reasons, and the second is the important
 * one:
 *
 *   - a flag that silently uses the other transport when the better one is
 *     available is untestable on the machine that HAS the better one, and this
 *     bank is exactly that machine — tt-kmd is the oracle and is deliberately
 *     kept installed. `--driver udrv` has to mean udrv here or it can never be
 *     compared against anything;
 *   - a preference is a fallback wearing a different word, and a fallback that
 *     produces a plausible answer from the wrong path is the defect this project
 *     has shipped twice.
 *
 * What `--driver udrv` therefore does NOT get is `probe --write`, which needs
 * the tt-kmd backend. That is a named limit, not a surprise: `probe` keeps the
 * session (see main()).
 *
 * @param out_card  zeroed and filled; `noc`/`grid` point into it, so it must
 *                  not be copied by value afterwards
 */
static int open_card(tool_card *card, tool_targets *targets, const tool_options *options,
                     char *out_address, size_t address_bytes)
{
    int rc;

    memset(card, 0, sizeof(*card));

    if (options->reset_driver != TOOL_RESET_DRIVER_UDRV) {
        rc = open_session(&card->kmd, targets, card->address, sizeof(card->address));
        if (rc != FT_TT_OK) {
            fprintf(stderr, "the vendor chardev did not open (rc=%d). --driver udrv reads "
                            "the card without one.\n", rc);
            return rc;
        }
        card->kmd_open = true;
        card->noc = &card->kmd.noc;
        card->grid = &card->kmd.grid;
        if (out_address)
            (void)snprintf(out_address, address_bytes, "%s", card->address);
        return FT_TT_OK;
    }

    rc = ft_tt_runtime_open_with_driver(0, FT_TT_RUNTIME_DRIVER_UDRV, &card->udrv);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "--driver udrv: the userspace driver could not open the card "
                        "(rc=%d). REFUSING — it will not quietly use the chardev instead. "
                        "Nothing has been done to the card.\n", rc);
        return rc;
    }
    card->udrv_open = true;
    card->noc = &card->udrv.noc;
    card->grid = &card->udrv.grid;
    rc = resolve_targets(card->grid, targets);
    if (rc != FT_TT_OK) {
        close_card(card);
        return rc;
    }
    (void)ft_tt_runtime_pci_address(&card->udrv, card->address, sizeof(card->address));
    if (out_address)
        (void)snprintf(out_address, address_bytes, "%s", card->address);
    printf("card read over foundry's own driver (no vendor ioctl in this path)\n");
    return FT_TT_OK;
}

/** Close whatever is open and blank the pointers.
 *
 *  Called BEFORE every reset, not only at the end: a reset invalidates every
 *  mapping the card had, and on the udrv path nothing zaps our VMAs for us
 *  (ft_tt_udrv.h §5(b) — the replacement is this discipline, not a mechanism). */
static void close_card(tool_card *card)
{
    if (!card)
        return;
    ft_tt_tensix_claim_release(card->cores);
    card->cores = NULL;
    if (card->kmd_open)
        ft_tt_hardware_session_close(&card->kmd);
    if (card->udrv_open)
        ft_tt_runtime_close(&card->udrv);
    card->kmd_open = false;
    card->udrv_open = false;
    card->noc = NULL;
    card->grid = NULL;
}

/** Rungs 3-5, which reset the whole card: capture, close, reset, reopen, report.
 *
 * Takes the exclusive card lock FIRST, before anything else -- the same property as the
 * PCIe reenumeration pre-flight below: acquire and check before anything destructive, and
 * release on every exit path via `cleanup`, including every failure path. A lock leaked by
 * a failed reset would be worse than no lock at all. */
static int run_device_recovery(tool_targets *targets, const tool_options *options)
{
    char card_address[FT_TT_PCI_ADDRESS_MAX] = {0};
    /* The userspace ASIC reset restores config space IN PLACE and needs no PCI
     * remove and no bus-wide rescan — that was measured, and `remove` + `rescan`
     * is kept only as the explicit `--udrv-reenumerate` fallback. So the
     * root-only reenumeration pre-flight applies to the kmd path and to that
     * fallback, and NOT to the default udrv path. Demanding it there would
     * refuse the one machine this driver exists for, for a permission it never
     * uses. */
    bool needs_reenumeration = options->reset_driver != TOOL_RESET_DRIVER_UDRV ||
                               options->udrv_reenumerate;
    tool_card card;
    int lock_fd = -1;
    int rc = acquire_card_lock_or_explain(options, &lock_fd);

    memset(&card, 0, sizeof(card));
    if (rc != FT_TT_OK)
        return rc;

    rc = open_card(&card, targets, options, card_address, sizeof(card_address));
    if (rc != FT_TT_OK)
        goto cleanup;
    printf("card is at PCI %s\n", card_address);

    /* SECOND PRE-FLIGHT: the card lock above only sees processes that registered
     * a shared claim. Ordinary work now holds PER-CORE claims, and a reset wipes
     * those too, so consult that namespace before anything destructive. */
    {
        unsigned core_holders = report_core_claim_holders(card_address);

        if (core_holders != 0u && !options->override_card_lock) {
            fprintf(stderr,
                    "refused: rung %u resets the whole card, and %u core%s are claimed by "
                    "live processes (listed above).\n"
                    "  A card reset returns every Tensix to power-on default, so it destroys "
                    "their work as surely as it would a shared claim's.\n"
                    "  Wait for them, or re-run with --override-card-lock. Nothing has been "
                    "reset.\n",
                    options->rung, core_holders, core_holders == 1u ? " is" : "s are");
            rc = FT_TT_EPERM;
            goto cleanup;
        }
        if (core_holders != 0u)
            fprintf(stderr, "--override-card-lock: those %u core claim%s will be destroyed.\n",
                    core_holders, core_holders == 1u ? "" : "s");
    }

    /* PRE-FLIGHT, BEFORE ANYTHING DESTRUCTIVE. Every card-level rung leaves the card
     * reset-but-not-re-enumerated (§21.3): memory decode off, BARs disabled, tt-kmd
     * unbound, /dev/tenstorrent gone. Only the PCI remove + rescan finishes the job, and
     * those sysfs nodes are root-only. Discovering that AFTER the reset leaves the card
     * dead until someone runs the rescan by hand -- which is exactly what happened on
     * 2026-08-12 (errno 13 at "removing the card from the PCI bus", card down until it was
     * re-enumerated manually). So refuse up front rather than half-finish. */
    rc = needs_reenumeration ? ft_tt_pcie_reenumerate_is_permitted(card_address) : FT_TT_OK;
    if (rc != FT_TT_OK) {
        close_card(&card);
        fprintf(stderr,
                "refused: rung %u resets the whole card, and completing it needs write\n"
                "  access to /sys/bus/pci/devices/%s/remove and /sys/bus/pci/rescan,\n"
                "  which this process does not have. Without them the card would be left\n"
                "  reset-but-not-re-enumerated: BARs disabled, tt-kmd unbound, no chardev.\n"
                "  Re-run under sudo. Nothing has been reset.\n",
                options->rung, card_address);
        rc = FT_TT_EPERM;
        goto cleanup;
    }

    print_targets_status(card.noc, card.grid, targets,
                         "before (captured now: a successful reset erases it)");
    /* Every handle must be closed: the reset paths zap VMAs and reclaim TLB windows, so a
     * live session's mapped windows are invalid the moment they return. On the udrv path
     * nothing zaps them for us, which makes closing here the mechanism rather than a tidy-up. */
    close_card(&card);

    if (options->reset_driver == TOOL_RESET_DRIVER_UDRV) {
        rc = run_udrv_asic_reset(card_address, options);
        if (rc != FT_TT_OK)
            goto cleanup;
    } else if (options->rung == 5u) {
        ft_tt_pcie_sbr_report report;

        rc = ft_tt_pcie_secondary_bus_reset(card_address, options->save_directory, &report);
        ft_tt_pcie_sbr_print(&report);
        if (rc == FT_TT_EPERM) {
            fprintf(stderr, "refused: %s\n",
                    report.topology.refusal ? report.topology.refusal : "topology check");
            goto cleanup;
        }
    } else {
        ft_tt_device_reset_report report;
        ft_tt_pcie_reenumeration_report reenumeration;
        ft_tt_device_reset_flag flag = options->rung == 3u ? FT_TT_RESET_DEVICE_ASIC_RESET
                                                           : FT_TT_RESET_DEVICE_ASIC_DMC_RESET;

        rc = ft_tt_device_reset_issue(TOOL_CHARDEV_PATH, flag, &report);
        ft_tt_device_reset_print(&report);
        if (rc != FT_TT_OK)
            goto cleanup;
        /* The driver's in-place completion does not finish the job on this host: the card
         * comes back with memory decode disabled and an ARC that answers 0xffffffff. See
         * ft_tt_pcie_reenumerate()'s header — this is measured, not defensive. */
        rc = ft_tt_pcie_reenumerate(card_address, &reenumeration);
        ft_tt_pcie_reenumeration_print(&reenumeration);
        if (rc != FT_TT_OK)
            goto cleanup;
    }

    rc = open_card(&card, targets, options, NULL, 0);
    if (rc != FT_TT_OK) {
        fprintf(stderr, "the card did not reopen after the reset (rc=%d)\n", rc);
        goto cleanup;
    }
    print_targets_status(card.noc, card.grid, targets, "after");
    if (options->then_probe)
        probe_targets(card.noc, card.grid, targets, options);
    rc = FT_TT_OK;

cleanup:
    close_card(&card);
    ft_tt_card_lock_release(lock_fd);
    return rc;
}

/** The refusals that keep a read-only tool read-only. */
static int check_permissions(const tool_options *options)
{
    if (options->command == TOOL_COMMAND_PROBE && !options->allow_write) {
        fprintf(stderr, "probe writes firmware, kernels and tiles to the card: pass --write\n");
        return FT_TT_EPERM;
    }
    if (options->command != TOOL_COMMAND_RECOVER)
        return FT_TT_OK;
    if (!options->rung_given || options->rung > 5u) {
        fprintf(stderr, "recover needs --rung 0..5\n");
        return FT_TT_EINVAL;
    }
    if (!options->allow_write) {
        fprintf(stderr, "recover writes to the card: pass --write\n");
        return FT_TT_EPERM;
    }
    if (options->rung == 5u && !options->allow_secondary_bus_reset) {
        fprintf(stderr, "rung 5 resets a PCIe bus: pass --allow-pcie-secondary-bus-reset "
                        "as well as --write\n");
        return FT_TT_EPERM;
    }
    /* `--driver` selects the TRANSPORT for every command, and additionally the MECHANISM
     * for a whole-card rung. Only rung 3 has a userspace mechanism, so those are the only
     * two rungs it can be refused for — and the refusal names why rather than silently
     * doing the kmd thing. Rung 4 is two ARC messages and there is no userspace ARC
     * transport wired into the reset path yet; rung 5 is already pure sysfs and has no
     * vendor-driver dependency to remove.
     *
     * `status`, `probe` and rungs 0-2 are NOT restricted: they reach the card only through
     * the ft_tt_noc_access vtable and a per-core flock, and both exist on either transport.
     * That is what makes the whole tool usable on a board with no vendor module. */
    if (options->command == TOOL_COMMAND_RECOVER &&
        options->reset_driver == TOOL_RESET_DRIVER_UDRV && options->rung >= 4u) {
        fprintf(stderr,
                "--driver udrv has no mechanism for rung %u. It is %s.\n", options->rung,
                options->rung == 4u
                    ? "ASIC+DMC, which needs two ARC messages and there is no userspace ARC "
                      "transport wired into the reset path yet"
                    : "already pure sysfs and uses no vendor driver");
        return FT_TT_EINVAL;
    }
    if (options->udrv_reenumerate && options->reset_driver != TOOL_RESET_DRIVER_UDRV) {
        fprintf(stderr, "--udrv-reenumerate only means something with --driver udrv\n");
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

int main(int argc, char **argv)
{
    tool_card card;
    tool_options options;
    tool_targets targets;
    char card_address[FT_TT_PCI_ADDRESS_MAX] = {0};
    int rc;

    setvbuf(stdout, NULL, _IOLBF, 0);

    if (parse_options(argc, argv, &options) != FT_TT_OK) {
        print_usage(argv[0]);
        return 2;
    }
    if (check_permissions(&options) != FT_TT_OK)
        return 2;

    if (options.command == TOOL_COMMAND_CLAIM)
        return run_claim(&options) == FT_TT_OK ? 0 : 1;

    if (options.command == TOOL_COMMAND_TOPOLOGY) {
        ft_tt_pcie_topology topology;

        /* An optional PCI address makes the REFUSAL demonstrable: point it at a device
         * that is not the card — a K80 behind a bridge with two children — and watch it
         * refuse. Read-only, and it does not change what rung 5 targets: that address is
         * always derived from the open device, never taken from a command line. */
        if (options.target_text) {
            snprintf(card_address, sizeof(card_address), "%s", options.target_text);
        } else {
            memset(&targets, 0, sizeof(targets));
            targets.all = true;
            rc = open_card(&card, &targets, &options, card_address, sizeof(card_address));
            if (rc != FT_TT_OK)
                return 1;
            close_card(&card);
        }
        if (ft_tt_pcie_topology_resolve(card_address, &topology) != FT_TT_OK)
            fprintf(stderr, "could not read the PCIe topology for %s\n", card_address);
        ft_tt_pcie_topology_print(&topology);

        /* The negotiated link, at BOTH ends. A reset leaves it at Gen1 and every
         * host-involved number then silently halves, so it belongs in the read-only
         * command that gets run at the start and end of a round. */
        {
            char device_dir[FT_TT_UDRV_RESET_PATH_MAX];

            if (pci_device_dir(topology.card[0] ? topology.card : card_address, device_dir,
                               sizeof(device_dir)) == FT_TT_OK)
                print_link_status("card", device_dir);
            if (topology.bridge[0] &&
                pci_device_dir(topology.bridge, device_dir, sizeof(device_dir)) == FT_TT_OK)
                print_link_status("bridge", device_dir);
        }
        return topology.safe ? 0 : 1;
    }

    if (parse_targets(options.target_text, &targets) != FT_TT_OK) {
        fprintf(stderr, "cannot parse targets: %s\n", options.target_text);
        return 2;
    }

    if (options.command == TOOL_COMMAND_RECOVER && options.rung >= 3u)
        return run_device_recovery(&targets, &options) == FT_TT_OK ? 0 : 1;

    /* STATUS goes through the transport seam and NOT through the tt-kmd session, because it
     * is the one command that has to work on a board with no vendor module: it reads only,
     * so it needs no claim, and `instrn_buf_status` read this way is the whole health
     * instrument once the module is gone. PROBE and rungs 0-2 WRITE to each core they
     * target, so they keep the session that carries the per-core claim. */
    if (options.command == TOOL_COMMAND_STATUS) {
        rc = open_card(&card, &targets, &options, NULL, 0);
        if (rc != FT_TT_OK)
            return 1;
        rc = run_status(card.noc, card.grid, &targets, options.detail);
        close_card(&card);
        return rc == FT_TT_OK ? 0 : 1;
    }
    /* READINESS travels with STATUS for the same reason: reads only, so no claim and no
     * session, and it must work on a board with no vendor module. */
    if (options.command == TOOL_COMMAND_READINESS) {
        rc = open_card(&card, &targets, &options, NULL, 0);
        if (rc != FT_TT_OK)
            return 1;
        rc = run_readiness(card.noc, card.grid, &targets);
        close_card(&card);
        return rc == FT_TT_OK ? 0 : 1;
    }

    rc = open_card(&card, &targets, &options, NULL, 0);
    if (rc != FT_TT_OK)
        return 1;

    switch (options.command) {
    case TOOL_COMMAND_PROBE:
        rc = claim_targets_or_explain(&card, &targets, &options, "probe");
        if (rc == FT_TT_OK)
            rc = run_probe(card.noc, card.grid, &targets, &options);
        break;
    case TOOL_COMMAND_STATUS:
    case TOOL_COMMAND_READINESS:
    case TOOL_COMMAND_RECOVER:
    default:
        rc = claim_targets_or_explain(&card, &targets, &options, "recover");
        if (rc == FT_TT_OK)
            rc = run_core_recovery(card.noc, card.grid, &targets, &options);
        break;
    }
    close_card(&card);
    return rc == FT_TT_OK ? 0 : 1;
}
