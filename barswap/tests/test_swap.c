/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#define _GNU_SOURCE

/*
 * test_swap.c — End-to-end die swap test for BAR Swap v2.0
 *
 * Cycles through all K80 dies, acquiring each into the MMIO slot,
 * loading the nvidia driver, verifying via nvidia-smi, unloading,
 * and releasing.
 *
 * Key technique: after acquiring a die, we identify which PCI BDF was
 * programmed by reading BAR 0 from sysfs (the acquired die has BAR 0
 * at 0xF3000000, all others are 0). We set driver_override="none" on
 * all OTHER GK210 devices so nvidia only probes the acquired die.
 *
 * Must run as root on the GPU server with bar_swap.ko loaded and
 * nvidia blacklisted at boot.
 *
 * Usage: sudo ./test_swap [die_id]
 *   No args:   test all dies
 *   With arg:  test only that die
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <time.h>

#include "../bar_swap_ko/bar_swap.h"

#define COLOR_GREEN  "\033[92m"
#define COLOR_RED    "\033[31m"
#define COLOR_YELLOW "\033[93m"
#define COLOR_RESET  "\033[0m"

#define MAX_DIES 16

/* BDF strings for all GK210 devices found in sysfs */
static char gk210_bdf[MAX_DIES][16];
static int n_gk210;

/*
 * Discover GK210 device BDFs from sysfs.
 */
static int discover_gk210s(void)
{
    FILE *fp;
    char path[512], buf[16];
    DIR *dir;
    struct dirent *ent;
    char *bdfs[MAX_DIES];
    int count = 0;

    dir = opendir("/sys/bus/pci/devices");
    if (!dir) return 0;

    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;

        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", ent->d_name);
        fp = fopen(path, "r");
        if (!fp) continue;
        if (!fgets(buf, sizeof(buf), fp)) { fclose(fp); continue; }
        fclose(fp);
        if (strncmp(buf, "0x10de", 6) != 0) continue;

        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", ent->d_name);
        fp = fopen(path, "r");
        if (!fp) continue;
        if (!fgets(buf, sizeof(buf), fp)) { fclose(fp); continue; }
        fclose(fp);
        if (strncmp(buf, "0x102d", 6) != 0) continue;

        if (count < MAX_DIES) {
            bdfs[count] = strdup(ent->d_name);
            count++;
        }
    }
    closedir(dir);

    /* Sort by BDF name */
    for (int i = 0; i < count - 1; i++)
        for (int j = i + 1; j < count; j++)
            if (strcmp(bdfs[i], bdfs[j]) > 0) {
                char *tmp = bdfs[i]; bdfs[i] = bdfs[j]; bdfs[j] = tmp;
            }

    for (int i = 0; i < count; i++) {
        snprintf(gk210_bdf[i], sizeof(gk210_bdf[i]), "%s", bdfs[i]);
        free(bdfs[i]);
    }
    n_gk210 = count;
    return count;
}

/*
 * Read BAR 0 value from sysfs for a PCI device.
 * Returns the resource start address, or 0 if unreadable/unmapped.
 */
static uint64_t read_bar0(const char *bdf)
{
    char path[512], buf[256];
    uint64_t start = 0;

    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", bdf);
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;

    /* First line of 'resource' file is BAR 0: start end flags */
    if (fgets(buf, sizeof(buf), fp))
        start = strtoull(buf, NULL, 16);
    fclose(fp);
    return start;
}

/*
 * Find which GK210 BDF has BAR 0 at the expected address (0xF3000000).
 * Returns the index into gk210_bdf[], or -1 if not found.
 */
static int find_acquired_bdf(uint64_t expected_bar0)
{
    for (int i = 0; i < n_gk210; i++) {
        if (read_bar0(gk210_bdf[i]) == expected_bar0)
            return i;
    }
    return -1;
}

static void set_driver_override(const char *bdf, const char *value)
{
    char path[512];
    int fd;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/driver_override", bdf);
    fd = open(path, O_WRONLY);
    if (fd < 0) return;
    if (*value == '\0') {
        /* Write newline to clear override — 0-byte write is ignored by VFS */
        write(fd, "\n", 1);
    } else {
        write(fd, value, strlen(value));
    }
    close(fd);
}

/*
 * Block all GK210 devices except the one at bdf_index.
 */
static void block_all_except(int bdf_index)
{
    for (int i = 0; i < n_gk210; i++) {
        if (i == bdf_index)
            set_driver_override(gk210_bdf[i], "");  /* clear — allow nvidia */
        else
            set_driver_override(gk210_bdf[i], "none");  /* block */
    }
}

static void unblock_all(void)
{
    for (int i = 0; i < n_gk210; i++)
        set_driver_override(gk210_bdf[i], "");
}

static int load_nvidia(void)
{
    int ret;
    ret = system("modprobe --ignore-install nvidia 2>&1");
    if (ret != 0) return ret;
    ret = system("modprobe --ignore-install nvidia_modeset 2>&1");
    if (ret != 0) return ret;
    ret = system("modprobe --ignore-install nvidia_drm 2>&1");
    if (ret != 0) return ret;
    ret = system("modprobe --ignore-install nvidia_uvm 2>&1");
    return ret;
}

static int unload_nvidia(void)
{
    return system("rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null");
}

/*
 * Verify GPU via /proc/driver/nvidia/gpus/<BDF>/information.
 * Much safer than nvidia-smi — reads from the driver's cached data
 * instead of querying GPU hardware (temperature, power, etc.) which
 * can hang or crash on dynamically swapped K80 dies.
 */
static int verify_gpu_proc(const char *bdf, char *output, size_t output_size)
{
    char path[512], buf[256];
    FILE *fp;
    size_t total = 0;

    snprintf(path, sizeof(path), "/proc/driver/nvidia/gpus/%s/information", bdf);
    fp = fopen(path, "r");
    if (!fp) return -1;

    while (total < output_size - 1 && fgets(buf, sizeof(buf), fp)) {
        size_t len = strlen(buf);
        if (total + len < output_size - 1) {
            memcpy(output + total, buf, len);
            total += len;
        }
    }
    output[total] = '\0';
    fclose(fp);
    return 0;
}

static double time_ms(struct timespec *start, struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

static int test_die(int fd, int die_id)
{
    struct bs_acquire_req acq;
    struct bs_release_req rel;
    struct timespec t_start, t_acquired, t_nvidia, t_smi, t_unload, t_released;
    char smi_output[4096];
    int ret, smi_ret = -1;

    printf("\n" COLOR_YELLOW "=== Die %d ===" COLOR_RESET "\n", die_id);

    /* Acquire */
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    memset(&acq, 0, sizeof(acq));
    acq.die_id = die_id;
    if (ioctl(fd, BAR_SWAP_ACQUIRE, &acq) < 0) {
        perror("  ACQUIRE ioctl");
        printf(COLOR_RED "  FAIL: could not acquire die %d" COLOR_RESET "\n", die_id);
        return -1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t_acquired);

    /* Find which BDF got the BARs */
    int bdf_idx = find_acquired_bdf(acq.bar0_addr);
    if (bdf_idx < 0) {
        printf(COLOR_RED "  FAIL: could not find acquired BDF in sysfs" COLOR_RESET "\n");
        goto release;
    }

    printf("  acquired: %s BAR0=0x%llx BAR1=0x%llx BAR3=0x%llx\n",
           gk210_bdf[bdf_idx],
           (unsigned long long)acq.bar0_addr,
           (unsigned long long)acq.bar1_addr,
           (unsigned long long)acq.bar3_addr);

    /* Block all other GK210s, allow only the acquired one */
    printf("  blocking %d other GK210s via driver_override...\n", n_gk210 - 1);
    block_all_except(bdf_idx);

    /* Load nvidia */
    ret = load_nvidia();
    clock_gettime(CLOCK_MONOTONIC, &t_nvidia);
    if (ret != 0) {
        printf(COLOR_RED "  FAIL: nvidia modprobe returned %d" COLOR_RESET "\n", ret);
        unload_nvidia();
        goto release;
    }
    printf("  nvidia loaded\n");

    /* Verify GPU via /proc instead of nvidia-smi (which can hang/crash) */
    smi_ret = verify_gpu_proc(gk210_bdf[bdf_idx], smi_output, sizeof(smi_output));
    clock_gettime(CLOCK_MONOTONIC, &t_smi);

    if (smi_ret != 0) {
        printf(COLOR_RED "  FAIL: /proc/driver/nvidia/gpus not found" COLOR_RESET "\n");
        unload_nvidia();
        goto release;
    }

    if (strstr(smi_output, "Tesla K80") || strstr(smi_output, "GK210")) {
        printf(COLOR_GREEN "  PASS: nvidia sees GPU" COLOR_RESET "\n");
        printf("  %s", smi_output);
    } else {
        printf(COLOR_RED "  FAIL: unexpected GPU info" COLOR_RESET "\n");
        printf("  %s", smi_output);
        smi_ret = -1;
        unload_nvidia();
        goto release;
    }

    /* Unload nvidia */
    ret = unload_nvidia();
    clock_gettime(CLOCK_MONOTONIC, &t_unload);
    if (ret != 0)
        printf(COLOR_YELLOW "  WARN: nvidia unload returned %d" COLOR_RESET "\n", ret);
    else
        printf("  nvidia unloaded\n");

release:
    unblock_all();

    memset(&rel, 0, sizeof(rel));
    rel.die_id = die_id;
    ioctl(fd, BAR_SWAP_RELEASE, &rel);

    clock_gettime(CLOCK_MONOTONIC, &t_released);
    printf("  released\n");

    printf("  timing: acquire=%.0fms nvidia=%.0fms smi=%.0fms unload=%.0fms release=%.0fms total=%.0fms\n",
           time_ms(&t_start, &t_acquired),
           time_ms(&t_acquired, &t_nvidia),
           time_ms(&t_nvidia, &t_smi),
           time_ms(&t_smi, &t_unload),
           time_ms(&t_unload, &t_released),
           time_ms(&t_start, &t_released));

    return (smi_ret == 0) ? 0 : -1;
}

int main(int argc, char **argv)
{
    struct bs_query_info info;
    int fd, i, passed = 0, failed = 0;
    int single_die = -1;

    if (argc > 1)
        single_die = atoi(argv[1]);

    if (getuid() != 0) {
        fprintf(stderr, "error: must run as root (sudo)\n");
        return 1;
    }

    discover_gk210s();
    unload_nvidia();

    fd = open("/dev/bar_swap", O_RDWR);
    if (fd < 0) {
        perror("open /dev/bar_swap");
        fprintf(stderr, "error: is bar_swap.ko loaded?\n");
        return 1;
    }

    memset(&info, 0, sizeof(info));
    if (ioctl(fd, BAR_SWAP_QUERY, &info) < 0) {
        perror("QUERY ioctl");
        close(fd);
        return 1;
    }

    printf("\n  BAR Swap v2.0 — Die Swap Test\n");
    printf("  %d dies (kernel), %d GK210s (sysfs)\n", info.n_dies, n_gk210);
    for (i = 0; i < n_gk210; i++)
        printf("    %s\n", gk210_bdf[i]);

    if (n_gk210 < (int)info.n_dies) {
        printf("\n  " COLOR_YELLOW "WARNING" COLOR_RESET
               ": sysfs has fewer GK210s than kernel (%d < %d)\n"
               "  some dies may need a reboot to recover\n",
               n_gk210, info.n_dies);
    }

    if (single_die >= 0) {
        if (single_die >= (int)info.n_dies) {
            fprintf(stderr, "error: die %d out of range (0..%d)\n",
                    single_die, info.n_dies - 1);
            close(fd);
            return 1;
        }
        if (test_die(fd, single_die) == 0)
            passed++;
        else
            failed++;
    } else {
        for (i = 0; i < (int)info.n_dies; i++) {
            if (test_die(fd, i) == 0)
                passed++;
            else
                failed++;
        }
    }

    close(fd);

    printf("\n  " COLOR_YELLOW "Results:" COLOR_RESET " %d dies tested, "
           COLOR_GREEN "%d passed" COLOR_RESET ", "
           COLOR_RED "%d failed" COLOR_RESET "\n\n",
           passed + failed, passed, failed);

    return failed > 0 ? 1 : 0;
}
