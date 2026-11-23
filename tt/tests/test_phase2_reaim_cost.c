/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_reaim_cost.c — what the launch path's window-aim skip is worth.
 *
 * Plan Phase 2 follow-up: "measure what the launch path's re-aim skip is worth. It used
 * to call CONFIGURE_TLB on every 4-byte access; now the shared ft_tt_noc path skips the
 * aim when the window is already on that (x, y, base). Launch cost was 16 µs with the
 * old behaviour — re-measure."
 *
 * The two arms differ by ONE line. `ft_tt_noc_path` remembers the (x, y, base) its
 * window is aimed at and skips the ioctl when the next access matches; clearing that flag
 * before every access reproduces the old behaviour EXACTLY, through the same code, with
 * no second implementation to be wrong. Nothing else changes: same firmware, same kernel,
 * same launch message, same order, same process.
 *
 *   1  per-access cost — N reads of the soft-reset register, one 4-byte access each.
 *      This isolates CONFIGURE_TLB from everything else and is what the launch delta
 *      has to be divided by to mean anything.
 *   2  per-launch cost — N launches of the do-nothing kernel: write the launch message,
 *      send the go message, poll for DONE. Step 9's measurement, unchanged, so the
 *      forced arm is directly comparable to the 16 µs the plan recorded.
 *
 * **The arm that must be able to fail is the forced one.** If poking the flag did not
 * actually force a re-aim, both arms would run identical code and report the same number
 * — a green run that measures nothing. So the aim COUNTER
 * (ft_tt_noc_path_aim_count()) is checked, not just the clock: the forced arm must aim
 * once per access and the skipping arm must aim far less often than that. A test that
 * only printed two timings would be exactly the "harness that measures nothing" this
 * repo keeps catching.
 *
 * Usage:  test_phase2_reaim_cost
 *   Firmware comes from FT_TT_FIRMWARE_DIR and the kernel is `ft_tt_kernel_noop` out of
 *   the committed ft_tt_kernels.h — deliberately the SAME pair step 9 launched a
 *   thousand times, because the number being re-measured is step 9's.
 *
 * Build:
 *   make -f Makefile.phase2_followups PHASE1B=1 build-phase2b/test_phase2_reaim_cost
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_hardware_session.h"
#include "ft_tt_kernels.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

#include <time.h>

#define RUN_TIMEOUT_MS 5000u

/** Register reads per arm of measurement 1. Large enough that the clock's resolution is
 *  irrelevant, small enough that the slow arm still finishes in a couple of seconds. */
#define REGISTER_READ_COUNT 20000u

/** Launches per arm of measurement 2 — step 9's count, so the numbers are comparable. */
#define LAUNCH_COUNT 1000u

/** Launches per block when the two arms are INTERLEAVED. The card host also runs eight K80s and
 *  other agents, and the host is two 1.8 GHz cores: measuring one arm to completion and
 *  then the other put a several-microsecond drift straight into the difference, and once
 *  produced a NEGATIVE cost for CONFIGURE_TLB. Alternating in short blocks makes both
 *  arms see the same load. */
#define LAUNCH_BLOCK 25u

/** Same idea for the register reads. */
#define READ_BLOCK 500u

/** Discarded iterations before each timed run, so the first-touch cost of a cold window
 *  or a cold page does not land in the average. */
#define WARMUP_COUNT 200u

static ft_tt_phase2_l1_layout g_layout;

/**
 * An accessor that wraps another one, counts accesses, and optionally invalidates the
 * path's "already aimed" memo first.
 *
 * Reaching into ft_tt_noc_path is deliberate and is confined to this file: it is the
 * only way to run the OLD behaviour through the CURRENT code. Re-aiming by bouncing
 * between two endpoints would double the transaction count and measure something else.
 */
typedef struct reaim_probe {
    const ft_tt_noc_access *inner;
    ft_tt_noc_window_binding *binding;
    bool force_reaim;
    uint64_t accesses;
} reaim_probe;

static void probe_before_access(reaim_probe *probe)
{
    probe->accesses++;
    if (probe->force_reaim)
        probe->binding->path.aimed = false;
}

static int probe_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                       size_t bytes)
{
    reaim_probe *probe = (reaim_probe *)user;

    probe_before_access(probe);
    return probe->inner->write(probe->inner->user, x, y, addr, src, bytes);
}

static int probe_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst, size_t bytes)
{
    reaim_probe *probe = (reaim_probe *)user;

    probe_before_access(probe);
    return probe->inner->read(probe->inner->user, x, y, addr, dst, bytes);
}

static void probe_init(reaim_probe *probe, ft_tt_hardware_session *session, bool force_reaim,
                       ft_tt_noc_access *out_access)
{
    memset(probe, 0, sizeof(*probe));
    probe->inner = &session->noc;
    probe->binding = &session->runtime.binding;
    probe->force_reaim = force_reaim;
    out_access->write = probe_write;
    out_access->read = probe_read;
    out_access->user = probe;
}

static uint64_t monotonic_nanoseconds(void)
{
    struct timespec now;

    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000uLL + (uint64_t)now.tv_nsec;
}

/** One arm's numbers. Kept together so the two arms are printed with the same fields. */
typedef struct arm_result {
    uint64_t elapsed_ns;
    uint64_t accesses;
    uint64_t aims;
    unsigned completed;
} arm_result;

static double nanoseconds_each(const arm_result *result, unsigned operations)
{
    return operations ? (double)result->elapsed_ns / (double)operations : 0.0;
}

/* ------------------------------------------------------------------------- */
/* 1 — per-access cost                                                       */
/* ------------------------------------------------------------------------- */

/** Time one block of register reads through `probe`, accumulating into `result`. */
static int timed_read_block(reaim_probe *probe, const ft_tt_noc_access *access,
                            ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                            unsigned count, arm_result *result)
{
    uint64_t aims_before = ft_tt_noc_path_aim_count(&session->runtime.binding.path);
    uint64_t accesses_before = probe->accesses;
    uint64_t started = monotonic_nanoseconds();
    unsigned iteration;

    for (iteration = 0; iteration < count; iteration++) {
        uint32_t value = 0;
        int rc = ft_tt_launch_read_reset(access, x, y, &value);

        if (rc != FT_TT_OK)
            return rc;
        result->completed++;
    }
    result->elapsed_ns += monotonic_nanoseconds() - started;
    result->aims += ft_tt_noc_path_aim_count(&session->runtime.binding.path) - aims_before;
    result->accesses += probe->accesses - accesses_before;
    return FT_TT_OK;
}

/** Both arms, interleaved, for the same reason measure_launches() interleaves. */
static int measure_register_reads(ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                                  arm_result *out_skipping, arm_result *out_forced)
{
    ft_tt_noc_access skipping_access;
    ft_tt_noc_access forced_access;
    reaim_probe skipping_probe;
    reaim_probe forced_probe;
    unsigned block;
    unsigned iteration;

    probe_init(&skipping_probe, session, false, &skipping_access);
    probe_init(&forced_probe, session, true, &forced_access);
    memset(out_skipping, 0, sizeof(*out_skipping));
    memset(out_forced, 0, sizeof(*out_forced));

    for (iteration = 0; iteration < WARMUP_COUNT; iteration++) {
        uint32_t value = 0;
        int rc = ft_tt_launch_read_reset(&skipping_access, x, y, &value);

        if (rc != FT_TT_OK)
            return rc;
    }

    for (block = 0; block * READ_BLOCK < REGISTER_READ_COUNT; block++) {
        int rc = timed_read_block(&skipping_probe, &skipping_access, session, x, y, READ_BLOCK,
                                  out_skipping);

        if (rc == FT_TT_OK)
            rc = timed_read_block(&forced_probe, &forced_access, session, x, y, READ_BLOCK,
                                  out_forced);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* 2 — per-launch cost                                                       */
/* ------------------------------------------------------------------------- */

/** One launch: stage the message, kick, poll for DONE. Step 9's loop body exactly. */
static int one_launch(const ft_tt_noc_access *access, uint16_t x, uint16_t y,
                      const ft_tt_launch_config *config)
{
    int rc = ft_tt_launch_write_launch_message(access, x, y, config);

    if (rc == FT_TT_OK)
        rc = ft_tt_launch_send_go(access, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_wait_done(access, x, y, RUN_TIMEOUT_MS, NULL);
    return rc;
}

/** Time one block of launches through `probe`, accumulating into `result`. */
static int timed_block(reaim_probe *probe, const ft_tt_noc_access *access,
                       ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                       const ft_tt_launch_config *config, unsigned count, arm_result *result)
{
    uint64_t aims_before = ft_tt_noc_path_aim_count(&session->runtime.binding.path);
    uint64_t accesses_before = probe->accesses;
    uint64_t started = monotonic_nanoseconds();
    unsigned iteration;

    for (iteration = 0; iteration < count; iteration++) {
        int rc = one_launch(access, x, y, config);

        if (rc != FT_TT_OK)
            return rc;
        result->completed++;
    }
    result->elapsed_ns += monotonic_nanoseconds() - started;
    result->aims += ft_tt_noc_path_aim_count(&session->runtime.binding.path) - aims_before;
    result->accesses += probe->accesses - accesses_before;
    return FT_TT_OK;
}

/**
 * Both arms, interleaved in blocks, so a host that gets busier halfway through the
 * measurement slows BOTH of them rather than only the one that ran second.
 */
static int measure_launches(ft_tt_hardware_session *session, uint16_t x, uint16_t y,
                            const ft_tt_launch_config *config, arm_result *out_skipping,
                            arm_result *out_forced)
{
    ft_tt_noc_access skipping_access;
    ft_tt_noc_access forced_access;
    reaim_probe skipping_probe;
    reaim_probe forced_probe;
    unsigned block;
    unsigned iteration;

    probe_init(&skipping_probe, session, false, &skipping_access);
    probe_init(&forced_probe, session, true, &forced_access);
    memset(out_skipping, 0, sizeof(*out_skipping));
    memset(out_forced, 0, sizeof(*out_forced));

    for (iteration = 0; iteration < WARMUP_COUNT / 10u; iteration++) {
        int rc = one_launch(&skipping_access, x, y, config);

        if (rc != FT_TT_OK)
            return rc;
    }

    for (block = 0; block * LAUNCH_BLOCK < LAUNCH_COUNT; block++) {
        int rc = timed_block(&skipping_probe, &skipping_access, session, x, y, config,
                             LAUNCH_BLOCK, out_skipping);

        if (rc == FT_TT_OK)
            rc = timed_block(&forced_probe, &forced_access, session, x, y, config, LAUNCH_BLOCK,
                             out_forced);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */

static void report_arm(const char *what, const arm_result *result, unsigned operations,
                       const char *unit)
{
    printf("        %-22s %8.2f µs/%s   %llu accesses, %llu window aims (%.2f per access)\n", what,
           nanoseconds_each(result, operations) / 1000.0, unit,
           (unsigned long long)result->accesses, (unsigned long long)result->aims,
           result->accesses ? (double)result->aims / (double)result->accesses : 0.0);
}

int main(void)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_image_placement firmware_placements[FT_TT_PROC_COUNT];
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_launch_config config;
    arm_result skipping = {0, 0, 0, 0};
    arm_result forced = {0, 0, 0, 0};
    unsigned processor;
    uint16_t x = 0;
    uint16_t y = 0;
    int rc;

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;
    rc = ft_tt_hardware_session_tensix_from_environment(&session, &x, &y, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    ft_tt_test_step("setup — boot the firmware from %s and load the do-nothing kernel",
                    ft_tt_firmware_directory());
    rc = ft_tt_core_boot_firmware(&session.noc, x, y, &firmware, firmware_placements);
    if (rc == FT_TT_OK)
        rc = ft_tt_phase2_l1_layout_derive(firmware_placements, FT_TT_PROC_COUNT, &g_layout);
    ft_tt_test_check_fmt("firmware booted and the L1 map derived", rc == FT_TT_OK,
                         "base 0x%05x, rc=%d (%s)", g_layout.kernel_config_base, rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    rc = ft_tt_core_load_program(&session.noc, x, y, &ft_tt_kernel_noop, placements);
    ft_tt_test_check_fmt("program loaded", rc == FT_TT_OK, "%u images",
                         ft_tt_kernel_noop.image_count);
    if (rc != FT_TT_OK)
        goto done;

    memset(&config, 0, sizeof(config));
    config.kernel_config_base = g_layout.kernel_config_base;
    config.enables = ft_tt_kernel_program_enables(&ft_tt_kernel_noop);
    config.brisc_noc_id = 0u;
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        uint32_t zero = 0;

        if (placements[processor].bytes_written == 0u)
            continue;
        config.kernel_text_address[processor] = placements[processor].entry_address;
        config.kernel_text_bytes[processor] = placements[processor].text_bytes;
        /* A VALID runtime-arg block for every enabled processor, even though noop reads
         * none. The sentinel is not safer than a real one: a TRISC that finds whatever
         * the previous program left at its block has stalled a core here before (dev
         * notes §5), and this program runs after 20,000 unrelated device accesses. */
        config.runtime_args_address[processor] =
            ft_tt_phase2_runtime_args_address(&g_layout, processor);
        rc = ft_tt_launch_plant_runtime_args(&session.noc, x, y,
                                             config.runtime_args_address[processor], &zero, 1u);
        if (rc != FT_TT_OK)
            break;
    }
    ft_tt_test_check_fmt("a valid runtime-arg block planted for every enabled processor",
                         rc == FT_TT_OK, "rc=%d (%s)", rc, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    /* --- 2: per-launch ---------------------------------------------------- */
    ft_tt_test_step("2 — %u launches of the do-nothing kernel per arm, interleaved in blocks "
                    "of %u", LAUNCH_COUNT, LAUNCH_BLOCK);
    rc = measure_launches(&session, x, y, &config, &skipping, &forced);
    ft_tt_test_check_fmt("both arms completed every launch",
                         rc == FT_TT_OK && skipping.completed == LAUNCH_COUNT &&
                             forced.completed == LAUNCH_COUNT,
                         "%u and %u of %u, interleaved %u at a time", skipping.completed,
                         forced.completed, LAUNCH_COUNT, LAUNCH_BLOCK);
    if (rc != FT_TT_OK)
        goto done;

    report_arm("aim skipped (today)", &skipping, LAUNCH_COUNT, "launch");
    report_arm("aim forced (before)", &forced, LAUNCH_COUNT, "launch");

    ft_tt_test_check_fmt("the forced arm really re-aimed on every access",
                         forced.aims == forced.accesses, "%llu aims for %llu accesses",
                         (unsigned long long)forced.aims, (unsigned long long)forced.accesses);
    ft_tt_test_check_fmt("skipping the aim made a launch cheaper",
                         skipping.elapsed_ns < forced.elapsed_ns, "%.2f µs vs %.2f µs per launch",
                         nanoseconds_each(&skipping, LAUNCH_COUNT) / 1000.0,
                         nanoseconds_each(&forced, LAUNCH_COUNT) / 1000.0);
    printf("        the skip is worth %.2f µs per launch (%.2f accesses per launch, "
           "%.2f of them aimed before)\n",
           (nanoseconds_each(&forced, LAUNCH_COUNT) -
            nanoseconds_each(&skipping, LAUNCH_COUNT)) / 1000.0,
           (double)forced.accesses / (double)LAUNCH_COUNT,
           (double)forced.aims / (double)LAUNCH_COUNT);
    /* --- 1: per-access ---------------------------------------------------- */
    ft_tt_test_step("1 — %u reads of the soft-reset register per arm, one 4-byte access "
                    "each, interleaved in blocks of %u", REGISTER_READ_COUNT, READ_BLOCK);
    rc = measure_register_reads(&session, x, y, &skipping, &forced);
    ft_tt_test_check_fmt("both arms completed", rc == FT_TT_OK,
                         "rc=%d (%s), interleaved %u reads at a time", rc,
                         ft_tt_error_string(rc), READ_BLOCK);
    if (rc != FT_TT_OK)
        goto done;

    report_arm("aim skipped (today)", &skipping, REGISTER_READ_COUNT, "read");
    report_arm("aim forced (before)", &forced, REGISTER_READ_COUNT, "read");

    /* These two are the reason the timings mean anything. */
    ft_tt_test_check_fmt("the forced arm really re-aimed on every access",
                         forced.aims == forced.accesses, "%llu aims for %llu accesses",
                         (unsigned long long)forced.aims, (unsigned long long)forced.accesses);
    ft_tt_test_check_fmt("the skipping arm re-aimed far less often",
                         skipping.aims * 10u < skipping.accesses, "%llu aims for %llu accesses",
                         (unsigned long long)skipping.aims,
                         (unsigned long long)skipping.accesses);
    printf("        one CONFIGURE_TLB costs %.2f µs (the difference of the two arms, "
           "%.2f aims apart per access)\n",
           (nanoseconds_each(&forced, REGISTER_READ_COUNT) -
            nanoseconds_each(&skipping, REGISTER_READ_COUNT)) / 1000.0,
           (double)(forced.aims - skipping.aims) / (double)REGISTER_READ_COUNT);


done:
    (void)ft_tt_core_park(&session.noc, x, y);
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
