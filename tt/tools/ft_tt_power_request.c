/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_power_request.c
 * @brief Ask tt-kmd for a non-minimum AI clock, hold the request open, and sample the
 *   telemetry row that says whether the firmware honoured it.
 *
 * @status New 2026-08-15. Written to answer one question: every Foundry process opens the
 *   chardev with `extra_flags = 0`, which tt-kmd reads as a LEGACY client and initialises to
 *   `FT_TT_KMD_POWER_FLAGS_ALL & ~FT_TT_KMD_POWER_FLAG_MAX_AI_CLOCK` — bit 0 of the firmware's
 *   power word is documented `1=Max AI Clock, 0=Min AI Clock`. So we have been asking for the
 *   MINIMUM AI clock on every open the backend has ever done, which is the standing candidate
 *   explanation for `DEV_NOTES.md`'s "Busy AICLK 800 MHz, unchanged under load".
 * @issues None
 * @todo None
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A SEPARATE HOLDER PROCESS AND NOT A CHANGE TO ft_tt_device_open()
 * ---------------------------------------------------------------------------
 * tt-kmd aggregates the power request across EVERY open fd with a bitwise OR, and re-runs that aggregation on every open, every close and every
 * SET_POWER_STATE. So a second process that merely holds an fd asking for bit 0 raises the
 * clock for the whole card, including for a workload process that knows nothing about it,
 * and closing that fd lowers it again.
 *
 * That makes the experiment a ONE-VARIABLE A/B on an unmodified workload binary: run the
 * benchmark with this holder absent, run the identical binary with it present. Changing the
 * backend's own open() instead would have rebuilt every arm and made "the clock" and "the
 * build" move together — and DEV_NOTES §3 is a catalogue of what that costs.
 *
 * ---------------------------------------------------------------------------
 * WHY O_APPEND
 * ---------------------------------------------------------------------------
 * the driver's open(): `bool power_aware = file->f_flags & O_APPEND`. A power-aware open starts
 * at flags 0 and triggers NO aggregation, so opening this tool changes nothing until the
 * ioctl lands — the request is exactly one ioctl, at a timestamp we print. A legacy open
 * would first push `0x7FFE` in on its own, which is the state we are trying to measure
 * against.
 *
 * The request itself is `FT_TT_KMD_POWER_FLAGS_ALL` (0x7FFF) — bit-for-bit what tt-kmd's own legacy
 * default sends, plus bit 0. That is deliberately the smallest possible difference from the
 * state the card has always run in: nothing else about the power word moves.
 */

/* `-std=c99 -pedantic` hides O_CLOEXEC, sigaction, clock_gettime and nanosleep behind a
 * feature-test macro on glibc while macOS declares them unconditionally, which is the
 * cross-machine trap DEV_NOTES §5 records for tt/convert. Declared before any include. */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "ft_tt_kmd_abi.h"

#define POWER_CHARDEV_PATH_FORMAT "/dev/tenstorrent/%d"
#define POWER_SYSFS_CLASS_FORMAT "/sys/class/tenstorrent/tenstorrent!%d"
#define POWER_PATH_MAX 512u

/** One telemetry row. Every field is a live sysfs read; see the sampling note below. */
typedef struct {
    double seconds;
    long aiclk_mhz;
    long axiclk_mhz;
    long arcclk_mhz;
    long heartbeat;
    long temp_millidegrees;
    long vcore_millivolts;
    long current_milliamps;
    long power_microwatts;
    long fan_rpm;
} power_telemetry_row;

static volatile sig_atomic_t stop_requested = 0;

static void on_stop_signal(int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

static double monotonic_seconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

/**
 * Read one integer out of a sysfs file.
 *
 * @return the value, or -1 when the file is absent or unreadable. -1 is reported as-is
 *         rather than substituted, because a silently-zero telemetry row is exactly the
 *         "the experiment is broken" case DEV_NOTES warns about.
 */
static long read_sysfs_long(const char *path)
{
    FILE *file = fopen(path, "r");
    long value = -1;

    if (!file)
        return -1;
    if (fscanf(file, "%ld", &value) != 1)
        value = -1;
    fclose(file);
    return value;
}

static void build_path(char *out, size_t out_bytes, const char *directory, const char *leaf)
{
    int written = snprintf(out, out_bytes, "%s/%s", directory, leaf);

    if (written < 0 || (size_t)written >= out_bytes)
        out[0] = '\0';
}

/** Locate the hwmon directory the driver registered for this device. */
static int find_hwmon_directory(const char *class_directory, char *out, size_t out_bytes)
{
    char pattern[POWER_PATH_MAX];
    char candidate[POWER_PATH_MAX];
    int index;

    build_path(pattern, sizeof(pattern), class_directory, "device/hwmon");
    for (index = 0; index < 64; index++) {
        int written = snprintf(candidate, sizeof(candidate), "%s/hwmon%d/name", pattern, index);

        if (written < 0 || (size_t)written >= sizeof(candidate))
            return -1;
        if (access(candidate, R_OK) == 0) {
            written = snprintf(out, out_bytes, "%s/hwmon%d", pattern, index);
            return (written < 0 || (size_t)written >= out_bytes) ? -1 : 0;
        }
    }
    return -1;
}

/**
 * Sample the whole row at once.
 *
 * Every field is read from sysfs, and tt-kmd services each read with a LIVE telemetry
 * fetch. The row is sampled together deliberately: DEV_NOTES records that a clock reading
 * without a load you can SEE is not a measurement, and board power in the same row is what
 * makes "800 MHz under load" distinguishable from "800 MHz while idle".
 */
static void sample_telemetry(const char *class_directory, const char *hwmon_directory,
                             double start_seconds, power_telemetry_row *row)
{
    char path[POWER_PATH_MAX];

    memset(row, 0, sizeof(*row));
    row->seconds = monotonic_seconds() - start_seconds;

    build_path(path, sizeof(path), class_directory, "tt_aiclk");
    row->aiclk_mhz = read_sysfs_long(path);
    build_path(path, sizeof(path), class_directory, "tt_axiclk");
    row->axiclk_mhz = read_sysfs_long(path);
    build_path(path, sizeof(path), class_directory, "tt_arcclk");
    row->arcclk_mhz = read_sysfs_long(path);
    build_path(path, sizeof(path), class_directory, "tt_heartbeat");
    row->heartbeat = read_sysfs_long(path);

    if (hwmon_directory[0] == '\0') {
        row->temp_millidegrees = -1;
        row->vcore_millivolts = -1;
        row->current_milliamps = -1;
        row->power_microwatts = -1;
        row->fan_rpm = -1;
        return;
    }
    build_path(path, sizeof(path), hwmon_directory, "temp1_input");
    row->temp_millidegrees = read_sysfs_long(path);
    build_path(path, sizeof(path), hwmon_directory, "in0_input");
    row->vcore_millivolts = read_sysfs_long(path);
    build_path(path, sizeof(path), hwmon_directory, "curr1_input");
    row->current_milliamps = read_sysfs_long(path);
    build_path(path, sizeof(path), hwmon_directory, "power1_input");
    row->power_microwatts = read_sysfs_long(path);
    build_path(path, sizeof(path), hwmon_directory, "fan1_input");
    row->fan_rpm = read_sysfs_long(path);
}

static void print_row(FILE *out, const power_telemetry_row *row)
{
    fprintf(out, "%.3f,%ld,%ld,%ld,%ld,%.3f,%ld,%.3f,%.3f,%ld\n", row->seconds, row->aiclk_mhz,
            row->axiclk_mhz, row->arcclk_mhz, row->heartbeat, (double)row->temp_millidegrees / 1000.0,
            row->vcore_millivolts, (double)row->current_milliamps / 1000.0,
            (double)row->power_microwatts / 1e6, row->fan_rpm);
    fflush(out);
}

/** Issue the request. `power_flags` is the whole word this fd contributes to the OR. */
static int request_power_flags(int fd, uint16_t power_flags)
{
    struct ft_tt_kmd_power_state_request request;

    memset(&request, 0, sizeof(request));
    request.argsz = (uint32_t)sizeof(request);
    request.flags = 0;
    request.reserved = 0;
    /* 15 valid flag bits and 0 valid settings: the aggregator's "unspecified" mask is then
     * empty (`~((1u << 15) - 1) & 0x7FFF == 0`), so the word this fd contributes is exactly
     * `power_flags` and nothing is inherited by default. */
    request.validity = (uint8_t)FT_TT_KMD_POWER_VALIDITY(15, 0);
    request.flags_word = power_flags;

    if (ioctl(fd, (unsigned long)FT_TT_KMD_IOCTL_SET_POWER_STATE, &request) != 0)
        return -errno;
    return 0;
}

static void print_usage(const char *program_name)
{
    fprintf(stderr,
            "usage: %s [--device N] [--max-ai-clk|--min-ai-clk] [--seconds S] [--interval-ms M]\n"
            "\n"
            "  --max-ai-clk    request power_flags 0x7FFF (every flag, INCLUDING MAX_AI_CLK)\n"
            "  --min-ai-clk    request power_flags 0x7FFE (the legacy default; a null arm)\n"
            "  --seconds S     hold for S seconds, then release (0 = until SIGINT/SIGTERM)\n"
            "  --interval-ms M sample the telemetry row every M ms (default 200)\n"
            "\n"
            "Prints a CSV header, then one row per sample:\n"
            "  seconds,aiclk_mhz,axiclk_mhz,arcclk_mhz,heartbeat,temp_c,vcore_mv,current_a,power_w,fan_rpm\n",
            program_name);
}

int main(int argc, char **argv)
{
    char chardev_path[POWER_PATH_MAX];
    char class_directory[POWER_PATH_MAX];
    char hwmon_directory[POWER_PATH_MAX] = "";
    power_telemetry_row row;
    struct sigaction action;
    double start_seconds;
    double hold_seconds = 0.0;
    long interval_ms = 200;
    uint16_t power_flags = FT_TT_KMD_POWER_FLAGS_ALL;
    int device_index = 0;
    int fd = -1;
    int rc;
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_index = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-ai-clk") == 0) {
            power_flags = FT_TT_KMD_POWER_FLAGS_ALL;
        } else if (strcmp(argv[i], "--min-ai-clk") == 0) {
            power_flags =
                (uint16_t)(FT_TT_KMD_POWER_FLAGS_ALL & ~FT_TT_KMD_POWER_FLAG_MAX_AI_CLOCK);
        } else if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
            hold_seconds = atof(argv[++i]);
        } else if (strcmp(argv[i], "--interval-ms") == 0 && i + 1 < argc) {
            interval_ms = atol(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }
    if (interval_ms < 1)
        interval_ms = 1;

    if (snprintf(chardev_path, sizeof(chardev_path), POWER_CHARDEV_PATH_FORMAT, device_index) < 0 ||
        snprintf(class_directory, sizeof(class_directory), POWER_SYSFS_CLASS_FORMAT, device_index) < 0) {
        fprintf(stderr, "ft_tt_power_request: device index %d does not fit a path\n", device_index);
        return 1;
    }
    if (find_hwmon_directory(class_directory, hwmon_directory, sizeof(hwmon_directory)) != 0)
        fprintf(stderr, "ft_tt_power_request: WARNING no hwmon under %s — power and temperature "
                        "columns will read -1, and a clock reading alone is not a measurement\n",
                class_directory);

    memset(&action, 0, sizeof(action));
    action.sa_handler = on_stop_signal;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);

    start_seconds = monotonic_seconds();

    /* BEFORE: sampled with no fd of ours open at all, so this row is the card's state
     * under whatever every other client is already asking for. */
    sample_telemetry(class_directory, hwmon_directory, start_seconds, &row);
    printf("# ft_tt_power_request device=%d requesting power_flags=0x%04X (MAX_AI_CLK=%s)\n",
           device_index, power_flags,
           (power_flags & FT_TT_KMD_POWER_FLAG_MAX_AI_CLOCK) ? "1" : "0");
    printf("seconds,aiclk_mhz,axiclk_mhz,arcclk_mhz,heartbeat,temp_c,vcore_mv,current_a,power_w,fan_rpm\n");
    printf("# before open:\n");
    print_row(stdout, &row);

    fd = open(chardev_path, O_RDWR | O_CLOEXEC | O_APPEND);
    if (fd == -1) {
        fprintf(stderr, "ft_tt_power_request: cannot open %s: %s\n", chardev_path, strerror(errno));
        return 1;
    }

    rc = request_power_flags(fd, power_flags);
    if (rc != 0) {
        fprintf(stderr, "ft_tt_power_request: SET_POWER_STATE refused: %s\n", strerror(-rc));
        close(fd);
        return 1;
    }
    printf("# request accepted, holding fd open\n");

    while (!stop_requested) {
        struct timespec sleep_request;

        sample_telemetry(class_directory, hwmon_directory, start_seconds, &row);
        print_row(stdout, &row);
        if (hold_seconds > 0.0 && row.seconds >= hold_seconds)
            break;
        sleep_request.tv_sec = (time_t)(interval_ms / 1000);
        sleep_request.tv_nsec = (long)((interval_ms % 1000) * 1000000L);
        nanosleep(&sleep_request, NULL);
    }

    /* Closing is the whole release path: tt-kmd removes this fd from the aggregation and
     * recomputes, so the card returns to whatever the remaining clients ask for. */
    close(fd);
    printf("# released\n");
    sample_telemetry(class_directory, hwmon_directory, start_seconds, &row);
    print_row(stdout, &row);
    return 0;
}
