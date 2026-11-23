/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_launch.c — checklist steps 7 to 10 from docs/tt_launch_notes.md §6.
 *
 * This is where our binary first RUNS on the card.
 *
 *   7  A do-nothing compute kernel. Nothing about circular buffers, destination
 *      registers or data movement is involved, so a hang here is the launch message or
 *      the mailbox handshake and nothing else.
 *   8  A deliberate timeout, twice over, because the notes' literal form cannot work:
 *      noop finishes in microseconds, so `timeout_ms = 1` on it is a race, not a proof.
 *        8a — go signal planted on a core held in RESET. Nothing can ever answer, so
 *             FT_TT_EIO plus the dump is the only possible outcome. Deterministic.
 *        8b — the same thing against a kernel that is genuinely still running
 *             (delay_compute.c), then a second wait that collects the completion.
 *             That second wait is the part that matters: it shows a timeout is a
 *             report, not damage.
 *   9  A thousand consecutive launches. A protocol that leaks state fails here and
 *      nowhere else.
 *  10  Runtime args. Each TRISC gets its own (address, value) pair and writes the value
 *      where it was told, which proves the per-processor rta_offset encoding AND is the
 *      first evidence that OUR kernel text executed at all.
 *
 * Every wait has a bound. Every exit path parks the core in reset.
 *
 * Build:
 *   cc -std=c99 -O1 -Wall -Wextra -Werror -pedantic -I.. -DFT_TT_PHASE1B_NOC_ENABLED \
 *      ../ft_tt_kmd.c ../ft_tt_device.c ../ft_tt_tlb.c ../ft_tt_arc_telemetry.c \
 *      ../ft_tt_coord.c ../ft_tt_elf.c ../ft_tt_kernel_blob.c ../ft_tt_launch.c \
 *      ft_tt_test_report.c ft_tt_hardware_session.c test_phase2_launch.c \
 *      -o build/test_phase2_launch
 */

#define _POSIX_C_SOURCE 200809L

#include "ft_tt_hardware_session.h"
#include "ft_tt_kernels.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <string.h>

#include <time.h>

/** Generous bound for a kernel we expect to finish immediately. */
#define QUICK_RUN_TIMEOUT_MS 5000u

/** Bound for the deliberately slow kernel, well above its measured runtime. */
#define SLOW_RUN_TIMEOUT_MS 60000u

/** Step 9's repeat count — the plan's acceptance criterion. */
#define REPEAT_LAUNCH_COUNT 1000u

/** How many of those repeats are re-run with a kernel whose effect the host can see. */
#define REPEAT_EVIDENCE_COUNT 100u

/** How long step 8's timeout is asked to wait on a core that can never answer. */
#define DEAD_CORE_TIMEOUT_MS 50u

/** Iterations for the first, calibrating run of delay_compute. */
#define DELAY_CALIBRATION_ITERATIONS 1000000u

/** What step 8b wants the slow kernel to take, so `timeout_ms = 1` cannot race it. */
#define DELAY_TARGET_MILLISECONDS 2000u

/** The L1 map, derived from where the firmware images landed rather than chosen. A
 *  file-scope object because every helper below needs it and threading one more
 *  parameter through all of them would say less than this comment does. Written once,
 *  in main(), before anything reads it. */
static ft_tt_phase2_l1_layout g_layout;

/**
 * Boot the firmware and DERIVE the L1 map from where its images landed.
 *
 * The map's base used to be the literal 0x14000. It is now the higher of "the top of the
 * five firmware images the loader just placed" and "the end of the regions the stock memory map
 * reserves above MEM_MAP_END", which is exactly what ft_tt_image_placement_l1_floor()
 * was written to compute and had no caller for.
 */
static int boot_and_derive_layout(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                  const ft_tt_firmware_set *firmware)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    int rc = ft_tt_core_boot_firmware(noc, x, y, firmware, placements);

    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_phase2_l1_layout_derive(placements, FT_TT_PROC_COUNT, &g_layout);
}

static uint64_t monotonic_milliseconds(void)
{
    struct timespec now;

    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)(now.tv_nsec / 1000000L);
}

/**
 * Fill in a launch config for a compute-only program from where its images landed.
 * Runtime-arg addresses are left at 0 (the sentinel is written by the encoder) unless
 * `with_runtime_args` is set.
 */
static void build_launch_config(const ft_tt_kernel_program *program,
                                const ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                                bool with_runtime_args, ft_tt_launch_config *out_config)
{
    unsigned processor;

    memset(out_config, 0, sizeof(*out_config));
    out_config->kernel_config_base = g_layout.kernel_config_base;
    out_config->enables = ft_tt_kernel_program_enables(program);
    out_config->brisc_noc_id = 0u;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (placements[processor].bytes_written == 0u)
            continue;
        out_config->kernel_text_address[processor] = placements[processor].entry_address;
        out_config->kernel_text_bytes[processor] = placements[processor].text_bytes;
        if (with_runtime_args)
            out_config->runtime_args_address[processor] = ft_tt_phase2_runtime_args_address(&g_layout, processor);
    }
}

/** Stage the launch message, kick, and wait. Returns wait_done's code. */
static int launch_and_wait(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                           const ft_tt_launch_config *config, uint32_t timeout_ms,
                           uint64_t *out_elapsed_ms)
{
    uint64_t started;
    int rc = ft_tt_launch_write_launch_message(noc, x, y, config);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;

    started = monotonic_milliseconds();
    rc = ft_tt_launch_wait_done(noc, x, y, timeout_ms, NULL);
    if (out_elapsed_ms)
        *out_elapsed_ms = monotonic_milliseconds() - started;
    return rc;
}

/* ------------------------------------------------------------------------- */
/* Step 7                                                                    */
/* ------------------------------------------------------------------------- */

static void run_step_7_do_nothing_kernel(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                         ft_tt_launch_config *out_noop_config)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    uint64_t elapsed = 0;
    int rc;

    ft_tt_test_step("step 7 — run a do-nothing compute kernel");

    rc = ft_tt_core_load_program(noc, x, y, &ft_tt_kernel_noop, placements);
    ft_tt_test_check_fmt("noop images loaded", rc == FT_TT_OK, "%u images, rc=%d",
                         ft_tt_kernel_noop.image_count, rc);
    if (rc != FT_TT_OK)
        return;

    build_launch_config(&ft_tt_kernel_noop, placements, false, out_noop_config);
    ft_tt_test_check_fmt("enables covers MATH0/1/2 only", out_noop_config->enables == 0x1cu,
                         "0x%02x", out_noop_config->enables);

    rc = launch_and_wait(noc, x, y, out_noop_config, QUICK_RUN_TIMEOUT_MS, &elapsed);
    ft_tt_test_check_fmt("kernel completed", rc == FT_TT_OK, "rc=%d (%s) in %llu ms", rc,
                         ft_tt_error_string(rc), (unsigned long long)elapsed);
}

/* ------------------------------------------------------------------------- */
/* Step 8                                                                    */
/* ------------------------------------------------------------------------- */

/** 8a: a core in reset can never write DONE, so the timeout is the only outcome. */
static void run_step_8a_dead_core_timeout(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    ft_tt_launch_state state;
    int rc;

    ft_tt_test_step("step 8a — deliberate timeout against a core held in reset "
                    "(the dump below is expected output, not a failure)");

    rc = ft_tt_core_park(noc, x, y);
    ft_tt_test_check_fmt("all five RISC-Vs back in reset", rc == FT_TT_OK, "rc=%d", rc);
    rc = ft_tt_launch_send_go(noc, x, y);
    ft_tt_test_check_fmt("go signal planted with nothing running", rc == FT_TT_OK, "rc=%d", rc);

    memset(&state, 0, sizeof(state));
    rc = ft_tt_launch_wait_done(noc, x, y, DEAD_CORE_TIMEOUT_MS, &state);
    ft_tt_test_check_fmt("wait_done reported FT_TT_EIO", rc == FT_TT_EIO, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    ft_tt_test_check_fmt("the dump captured a core still holding GO",
                         state.go_messages[0].signal == FT_TT_RUN_MSG_GO, "go[0].signal=0x%02x",
                         state.go_messages[0].signal);
    ft_tt_test_check_fmt("the dump saw all five held in reset",
                         (state.reset_register & FT_TT_SOFT_RESET_ALL_RISCS) ==
                             FT_TT_SOFT_RESET_ALL_RISCS,
                         "soft reset 0x%08x", state.reset_register);

    /* Refuse to wait forever — the guard that makes this whole phase survivable. */
    rc = ft_tt_launch_wait_done(noc, x, y, 0u, NULL);
    ft_tt_test_check_fmt("wait_done refuses timeout_ms == 0", rc == FT_TT_EINVAL, "rc=%d (%s)",
                         rc, ft_tt_error_string(rc));
}

/** 8b: the same timeout against a kernel that really is still running. */
static void run_step_8b_running_kernel_timeout(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_launch_config config;
    uint32_t arguments[1];
    uint64_t calibration_ms = 0;
    uint64_t elapsed = 0;
    uint32_t iterations;
    unsigned processor;
    int rc;

    ft_tt_test_step("step 8b — deliberate timeout against a kernel that is still running");

    rc = ft_tt_core_load_program(noc, x, y, &ft_tt_kernel_delay, placements);
    ft_tt_test_check_fmt("delay images loaded", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    build_launch_config(&ft_tt_kernel_delay, placements, true, &config);

    /* Calibrate: nothing here knows the Tensix clock, so measure one short run and
     * scale it rather than hardcoding an iteration count that ages badly. */
    arguments[0] = DELAY_CALIBRATION_ITERATIONS;
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (config.runtime_args_address[processor] == 0u)
            continue;
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config.runtime_args_address[processor],
                                             arguments, 1u);
        if (rc != FT_TT_OK)
            break;
    }
    ft_tt_test_check_fmt("calibration args planted", rc == FT_TT_OK, "rc=%d", rc);
    rc = launch_and_wait(noc, x, y, &config, SLOW_RUN_TIMEOUT_MS, &calibration_ms);
    ft_tt_test_check_fmt("calibration run completed", rc == FT_TT_OK, "%u iterations in %llu ms",
                         DELAY_CALIBRATION_ITERATIONS, (unsigned long long)calibration_ms);
    if (rc != FT_TT_OK)
        return;

    if (calibration_ms == 0u)
        calibration_ms = 1u;
    iterations = (uint32_t)((uint64_t)DELAY_CALIBRATION_ITERATIONS * DELAY_TARGET_MILLISECONDS /
                            calibration_ms);
    arguments[0] = iterations;
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (config.runtime_args_address[processor] == 0u)
            continue;
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config.runtime_args_address[processor],
                                             arguments, 1u);
        if (rc != FT_TT_OK)
            break;
    }
    ft_tt_test_check_fmt("slow-run args planted", rc == FT_TT_OK, "%u iterations, target %u ms",
                         iterations, DELAY_TARGET_MILLISECONDS);
    if (rc != FT_TT_OK)
        return;

    ft_tt_test_step("step 8b — the timeout (the dump below is expected output)");
    rc = launch_and_wait(noc, x, y, &config, 1u, &elapsed);
    ft_tt_test_check_fmt("wait_done(timeout_ms=1) reported FT_TT_EIO", rc == FT_TT_EIO,
                         "rc=%d (%s) after %llu ms", rc, ft_tt_error_string(rc),
                         (unsigned long long)elapsed);

    /* The core was never touched by the timeout: pick the completion up. */
    rc = ft_tt_launch_wait_done(noc, x, y, SLOW_RUN_TIMEOUT_MS, NULL);
    ft_tt_test_check_fmt("a second wait collected the completion", rc == FT_TT_OK, "rc=%d (%s)",
                         rc, ft_tt_error_string(rc));
}

/* ------------------------------------------------------------------------- */
/* Step 9                                                                    */
/* ------------------------------------------------------------------------- */

static void run_step_9_repeat(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_launch_config config;
    uint64_t started;
    uint64_t total_ms;
    uint32_t completed = 0;
    int last_rc = FT_TT_OK;
    int rc;

    ft_tt_test_step("step 9 — %u consecutive launches of the do-nothing kernel",
                    REPEAT_LAUNCH_COUNT);

    rc = ft_tt_core_load_program(noc, x, y, &ft_tt_kernel_noop, placements);
    ft_tt_test_check_fmt("noop images reloaded", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    build_launch_config(&ft_tt_kernel_noop, placements, false, &config);

    started = monotonic_milliseconds();
    while (completed < REPEAT_LAUNCH_COUNT) {
        last_rc = launch_and_wait(noc, x, y, &config, QUICK_RUN_TIMEOUT_MS, NULL);
        if (last_rc != FT_TT_OK)
            break;
        completed++;
    }
    total_ms = monotonic_milliseconds() - started;

    ft_tt_test_check_fmt("every launch completed", completed == REPEAT_LAUNCH_COUNT,
                         "%u/%u in %llu ms (%.2f ms each), last rc=%d", completed,
                         REPEAT_LAUNCH_COUNT, (unsigned long long)total_ms,
                         completed ? (double)total_ms / (double)completed : 0.0, last_rc);
}

/**
 * The loop above is fast enough to deserve suspicion: a completion poll that returned
 * OK without the kernel ever running would look exactly like this. So repeat it with a
 * kernel that leaves evidence, checking a FRESH value every iteration.
 */
static void run_step_9_repeat_with_evidence(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_launch_config config;
    uint32_t iteration;
    uint32_t verified = 0;
    int rc;

    ft_tt_test_step("step 9 — %u more launches with a kernel that leaves evidence, "
                    "a fresh value checked each time", REPEAT_EVIDENCE_COUNT);

    rc = ft_tt_core_load_program(noc, x, y, &ft_tt_kernel_runtime_arg_probe, placements);
    ft_tt_test_check_fmt("runtime_arg_probe images loaded", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    build_launch_config(&ft_tt_kernel_runtime_arg_probe, placements, true, &config);

    /* EVERY enabled TRISC needs its own valid arg block, not just the one being
     * checked. The compute kernel runs on all three, and a TRISC left pointing at a
     * stale block (or at the 0xFFFF sentinel) dereferences whatever that word happens
     * to hold and STALLS the core — observed here on the first attempt, when TRISC1 was
     * still holding step 8b's iteration count and stored to address 222222222. */
    for (iteration = 0; iteration < REPEAT_EVIDENCE_COUNT; iteration++) {
        unsigned processor;
        bool matched = true;

        for (processor = 0; processor < FT_TT_PROC_COUNT && rc == FT_TT_OK; processor++) {
            uint32_t arguments[2];
            uint32_t zero = 0;

            if (config.runtime_args_address[processor] == 0u)
                continue;
            arguments[0] = g_layout.probe_address + processor * FT_TT_PHASE2_PROBE_STRIDE;
            arguments[1] = 0xE0000000u | (iteration << 8) | processor;
            rc = noc->write(noc->user, x, y, arguments[0], &zero, sizeof(zero));
            if (rc == FT_TT_OK)
                rc = ft_tt_launch_plant_runtime_args(
                    noc, x, y, config.runtime_args_address[processor], arguments, 2u);
        }
        if (rc == FT_TT_OK)
            rc = launch_and_wait(noc, x, y, &config, QUICK_RUN_TIMEOUT_MS, NULL);
        if (rc != FT_TT_OK)
            break;

        for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
            uint32_t address = g_layout.probe_address + processor * FT_TT_PHASE2_PROBE_STRIDE;
            uint32_t expected = 0xE0000000u | (iteration << 8) | processor;
            uint32_t readback = 0xFFFFFFFFu;

            if (config.runtime_args_address[processor] == 0u)
                continue;
            rc = noc->read(noc->user, x, y, address, &readback, sizeof(readback));
            if (rc != FT_TT_OK || readback != expected)
                matched = false;
        }
        if (!matched)
            break;
        verified++;
    }
    ft_tt_test_check_fmt("every launch left the value it was told to write",
                         verified == REPEAT_EVIDENCE_COUNT, "%u/%u verified, last rc=%d", verified,
                         REPEAT_EVIDENCE_COUNT, rc);
}

/** The negative control for step 9: with the core in reset the very same loop body must
 *  FAIL. A green loop nobody has watched go red is not evidence. */
static void run_step_9_negative_control(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    int rc;

    ft_tt_test_step("step 9 — negative control: the same launch against a core in reset "
                    "(the dump below is expected output)");

    rc = ft_tt_core_park(noc, x, y);
    ft_tt_test_check_fmt("core parked in reset", rc == FT_TT_OK, "rc=%d", rc);
    rc = ft_tt_launch_send_go(noc, x, y);
    if (rc == FT_TT_OK)
        rc = ft_tt_launch_wait_done(noc, x, y, DEAD_CORE_TIMEOUT_MS, NULL);
    ft_tt_test_check_fmt("the loop body reports failure when nothing is running",
                         rc == FT_TT_EIO, "rc=%d (%s)", rc, ft_tt_error_string(rc));
}

/* ------------------------------------------------------------------------- */
/* Step 10                                                                   */
/* ------------------------------------------------------------------------- */

static void run_step_10_runtime_args(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    static const uint32_t k_planted_values[FT_TT_PROC_COUNT] = {0u, 0u, 0xA1B2C3D4u, 0x5E6F7089u,
                                                                0x0F1E2D3Cu};
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_launch_config config;
    uint32_t zeros[FT_TT_PROC_COUNT] = {0};
    uint32_t readback = 0;
    unsigned processor;
    int rc;

    ft_tt_test_step("step 10 — runtime args, one distinct (address, value) pair per TRISC");

    rc = ft_tt_core_load_program(noc, x, y, &ft_tt_kernel_runtime_arg_probe, placements);
    ft_tt_test_check_fmt("runtime_arg_probe images loaded", rc == FT_TT_OK, "rc=%d", rc);
    if (rc != FT_TT_OK)
        return;
    build_launch_config(&ft_tt_kernel_runtime_arg_probe, placements, true, &config);

    /* Wipe the landing zone: a stale value from an earlier run would pass this test
     * without the kernel executing at all. */
    rc = noc->write(noc->user, x, y, g_layout.probe_address, zeros, sizeof(zeros));
    ft_tt_test_check_fmt("probe landing zone zeroed", rc == FT_TT_OK, "%u words at 0x%05x",
                         (unsigned)FT_TT_PROC_COUNT, g_layout.probe_address);

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        uint32_t arguments[2];

        if (config.runtime_args_address[processor] == 0u)
            continue;
        arguments[0] = g_layout.probe_address + processor * FT_TT_PHASE2_PROBE_STRIDE;
        arguments[1] = k_planted_values[processor];
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config.runtime_args_address[processor],
                                             arguments, 2u);
        ft_tt_test_check_fmt("args planted", rc == FT_TT_OK, "%-6s rta block 0x%05x -> write "
                             "0x%08x to 0x%05x", ft_tt_processor_name(processor),
                             config.runtime_args_address[processor], arguments[1], arguments[0]);
    }

    rc = launch_and_wait(noc, x, y, &config, QUICK_RUN_TIMEOUT_MS, NULL);
    ft_tt_test_check_fmt("probe kernel completed", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        return;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        uint32_t address = g_layout.probe_address + processor * FT_TT_PHASE2_PROBE_STRIDE;

        if (config.runtime_args_address[processor] == 0u)
            continue;
        rc = noc->read(noc->user, x, y, address, &readback, sizeof(readback));
        ft_tt_test_check_fmt("the value the host planted came back",
                             rc == FT_TT_OK && readback == k_planted_values[processor],
                             "%-6s 0x%05x reads 0x%08x, expected 0x%08x",
                             ft_tt_processor_name(processor), address, readback,
                             k_planted_values[processor]);
    }
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_launch_config noop_config;
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

    ft_tt_test_step("setup — boot the firmware on all five RISC-Vs");
    rc = boot_and_derive_layout(&session.noc, x, y, &firmware);
    ft_tt_test_check_fmt("firmware booted", rc == FT_TT_OK, "rc=%d (%s)", rc,
                         ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    memset(&noop_config, 0, sizeof(noop_config));
    run_step_7_do_nothing_kernel(&session.noc, x, y, &noop_config);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_step_8a_dead_core_timeout(&session.noc, x, y);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    /* 8a left the core in reset on purpose; bring it back up. */
    rc = boot_and_derive_layout(&session.noc, x, y, &firmware);
    ft_tt_test_check_fmt("firmware re-booted after the dead-core timeout", rc == FT_TT_OK,
                         "rc=%d (%s)", rc, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    run_step_8b_running_kernel_timeout(&session.noc, x, y);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_step_9_repeat(&session.noc, x, y);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_step_9_repeat_with_evidence(&session.noc, x, y);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    run_step_9_negative_control(&session.noc, x, y);
    if (ft_tt_test_failure_count() != 0)
        goto done;

    rc = boot_and_derive_layout(&session.noc, x, y, &firmware);
    ft_tt_test_check_fmt("firmware re-booted after the negative control", rc == FT_TT_OK,
                         "rc=%d (%s)", rc, ft_tt_error_string(rc));
    if (rc != FT_TT_OK)
        goto done;

    run_step_10_runtime_args(&session.noc, x, y);

done:
    (void)ft_tt_core_park(&session.noc, x, y);
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
