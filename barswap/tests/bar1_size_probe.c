/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * bar1_size_probe.c — Probe GK210 BAR1 minimum size
 *
 * Acquires a die via bar_swap, disables decode, writes 0xFFFFFFFF to BAR1
 * config registers, reads back the size mask, restores original values.
 * This reveals which bits are hardwired (minimum BAR size).
 *
 * Must run as root. bar_swap.ko must be loaded. nvidia must NOT be loaded.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/ioctl.h>

#include "../bar_swap_ko/bar_swap.h"

/* Find GK210 sysfs path by scanning PCI devices */
static int find_gk210_bdf(int die_id, char *bdf_out, size_t bdf_sz) {
    /* Just use setpci on known GK210 device IDs */
    /* For now, scan /sys/bus/pci/devices/ for vendor=10de device=102d */
    DIR *dir = opendir("/sys/bus/pci/devices");
    if (!dir) return -1;

    int found = 0;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;

        char path[512];
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", ent->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        unsigned vendor;
        fscanf(f, "%x", &vendor);
        fclose(f);
        if (vendor != 0x10de) continue;

        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", ent->d_name);
        f = fopen(path, "r");
        if (!f) continue;
        unsigned device;
        fscanf(f, "%x", &device);
        fclose(f);
        if (device != 0x102d) continue;

        if (found == die_id) {
            snprintf(bdf_out, bdf_sz, "%s", ent->d_name);
            closedir(dir);
            return 0;
        }
        found++;
    }
    closedir(dir);
    return -1;
}

static int read_pci_config(const char *bdf, int offset, uint32_t *val) {
    char path[512];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/config", bdf);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return -1; }
    if (lseek(fd, offset, SEEK_SET) < 0) { perror("lseek"); close(fd); return -1; }
    if (read(fd, val, 4) != 4) { perror("read"); close(fd); return -1; }
    close(fd);
    return 0;
}

static int write_pci_config(const char *bdf, int offset, uint32_t val) {
    char path[512];
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/config", bdf);
    int fd = open(path, O_RDWR);
    if (fd < 0) { perror(path); return -1; }
    if (lseek(fd, offset, SEEK_SET) < 0) { perror("lseek"); close(fd); return -1; }
    if (write(fd, &val, 4) != 4) { perror("write"); close(fd); return -1; }
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    int die_id = 0;
    if (argc > 1) die_id = atoi(argv[1]);

    printf("=== GK210 BAR1 Size Probe ===\n\n");

    /* Open bar_swap */
    int bs_fd = open("/dev/bar_swap", O_RDWR);
    if (bs_fd < 0) { perror("/dev/bar_swap"); return 1; }

    /* Query */
    struct bs_query_info qi;
    if (ioctl(bs_fd, BAR_SWAP_QUERY, &qi) < 0) { perror("QUERY"); close(bs_fd); return 1; }
    printf("Dies: %u, Active: %s\n", qi.n_dies, qi.active_die == 0xFFFFFFFF ? "none" : "occupied");

    if ((unsigned)die_id >= qi.n_dies) {
        printf("Die %d out of range (max %u)\n", die_id, qi.n_dies - 1);
        close(bs_fd);
        return 1;
    }

    /* Find GK210 BDF */
    char bdf[64];
    if (find_gk210_bdf(die_id, bdf, sizeof(bdf)) < 0) {
        printf("Could not find GK210 die %d in sysfs\n", die_id);
        close(bs_fd);
        return 1;
    }
    printf("Die %d: %s\n\n", die_id, bdf);

    /* Acquire die */
    struct bs_acquire_req aq = { .die_id = die_id, .flags = 0 };
    if (ioctl(bs_fd, BAR_SWAP_ACQUIRE, &aq) < 0) { perror("ACQUIRE"); close(bs_fd); return 1; }
    printf("Die %d acquired. BAR0=0x%llx BAR1=0x%llx BAR3=0x%llx\n\n",
           die_id, aq.bar0_addr, aq.bar1_addr, aq.bar3_addr);

    /*
     * BAR1 is a 64-bit BAR at PCI config offset 0x14 (BAR2 in PCI terms,
     * since BAR0 is at 0x10 and is 64-bit, taking offsets 0x10+0x14).
     *
     * Actually for GK210:
     *   BAR0 = offset 0x10 (32-bit, 16 MB)
     *   BAR1 = offset 0x14 + 0x18 (64-bit, 16 GB)
     *   BAR3 = offset 0x1C + 0x20 (64-bit, 32 MB)
     *
     * Let's just read all 6 BAR registers first.
     */
    printf("Current BAR config registers:\n");
    uint32_t bars[6];
    for (int i = 0; i < 6; i++) {
        if (read_pci_config(bdf, 0x10 + i * 4, &bars[i]) < 0) {
            printf("  Failed to read BAR%d\n", i);
            goto release;
        }
        printf("  BAR%d (0x%02X): 0x%08X\n", i, 0x10 + i * 4, bars[i]);
    }

    /* Identify BAR1 (the 16 GB VRAM BAR) by checking which is 64-bit prefetchable */
    printf("\nBAR type analysis:\n");
    for (int i = 0; i < 6; i++) {
        uint32_t v = bars[i];
        if (v & 1) {
            printf("  BAR%d: I/O\n", i);
        } else {
            int type = (v >> 1) & 3;  /* 0=32bit, 2=64bit */
            int pref = (v >> 3) & 1;
            printf("  BAR%d: Memory, %s, %s\n", i,
                   type == 2 ? "64-bit" : "32-bit",
                   pref ? "prefetchable" : "non-prefetchable");
            if (type == 2) {
                printf("         (high dword in BAR%d: 0x%08X)\n", i + 1, bars[i + 1]);
                i++; /* skip high dword */
            }
        }
    }

    /*
     * Now probe BAR1 size.
     * For GK210: BAR0 at 0x10 is 32-bit non-pref (registers).
     *            BAR1 at 0x14+0x18 is 64-bit pref (VRAM).
     *
     * To probe size:
     *   1. Disable memory decode (clear CMD bit 1)
     *   2. Save BAR1 low + high
     *   3. Write 0xFFFFFFFF to BAR1 low, read back
     *   4. Write 0xFFFFFFFF to BAR1 high, read back
     *   5. Restore original values
     *   6. Re-enable memory decode
     */

    /* BAR1 is at offset 0x14 (low) and 0x18 (high) based on GK210 layout */
    int bar1_lo_off = -1, bar1_hi_off = -1;

    /* Find the 64-bit prefetchable BAR */
    for (int i = 0; i < 6; i++) {
        uint32_t v = bars[i];
        if (!(v & 1) && ((v >> 1) & 3) == 2 && ((v >> 3) & 1)) {
            /* 64-bit prefetchable memory BAR */
            bar1_lo_off = 0x10 + i * 4;
            bar1_hi_off = 0x10 + (i + 1) * 4;
            printf("\nBAR1 (VRAM) found at config offset 0x%02X/0x%02X\n", bar1_lo_off, bar1_hi_off);
            break;
        }
    }

    if (bar1_lo_off < 0) {
        printf("\nERROR: Could not find 64-bit prefetchable BAR\n");
        goto release;
    }

    /* Save originals */
    uint32_t orig_lo = bars[(bar1_lo_off - 0x10) / 4];
    uint32_t orig_hi = bars[(bar1_hi_off - 0x10) / 4];
    printf("Original BAR1: low=0x%08X high=0x%08X\n", orig_lo, orig_hi);
    printf("Current BAR1 address: 0x%llX\n",
           ((uint64_t)orig_hi << 32) | (orig_lo & 0xFFFFFFF0));

    /* Disable decode */
    uint32_t cmd;
    read_pci_config(bdf, 0x04, &cmd);
    uint32_t cmd_disabled = cmd & ~0x06; /* clear MEM + BusMaster */
    write_pci_config(bdf, 0x04, cmd_disabled);
    printf("\nDecode disabled (CMD: 0x%04X -> 0x%04X)\n", cmd, cmd_disabled);

    /* Write all-ones to BAR1 low */
    write_pci_config(bdf, bar1_lo_off, 0xFFFFFFFF);
    uint32_t mask_lo;
    read_pci_config(bdf, bar1_lo_off, &mask_lo);

    /* Write all-ones to BAR1 high */
    write_pci_config(bdf, bar1_hi_off, 0xFFFFFFFF);
    uint32_t mask_hi;
    read_pci_config(bdf, bar1_hi_off, &mask_hi);

    printf("\n=== SIZE MASK RESULTS ===\n");
    printf("BAR1 low  mask: 0x%08X\n", mask_lo);
    printf("BAR1 high mask: 0x%08X\n", mask_hi);

    /* Decode the size */
    /* For a memory BAR, bits [3:0] are type/pref flags, rest is size mask */
    uint64_t full_mask = ((uint64_t)mask_hi << 32) | (mask_lo & 0xFFFFFFF0);
    uint64_t size = ~full_mask + 1;  /* Two's complement gives size */

    printf("\nFull mask (excluding type bits): 0x%016llX\n", (unsigned long long)full_mask);
    printf("Decoded BAR1 size: 0x%llX (%llu GB)\n",
           (unsigned long long)size, (unsigned long long)(size >> 30));

    if (mask_lo == 0xFFFFFFFF && mask_hi == 0xFFFFFFFF) {
        printf("\nWARNING: BAR1 returned all-ones — may not be accessible\n");
    }

    /* Check if smaller sizes are possible */
    printf("\n=== ANALYSIS ===\n");
    if (size == 0x400000000ULL) {
        printf("BAR1 is hardwired to 16 GB. Cannot resize.\n");
        printf("Multi-slot via BAR1 resizing is NOT possible.\n");
        printf("Fallback: NIC relocation to free 0x400000000 slot.\n");
    } else if (size < 0x400000000ULL) {
        printf("BAR1 minimum size: %llu GB — SMALLER THAN 16 GB!\n",
               (unsigned long long)(size >> 30));
        printf("Multi-slot via BAR1 resizing IS possible!\n");
        printf("Could fit %llu dies in the 48 GB MMIO window.\n",
               48ULL / (size >> 30));
    } else {
        printf("BAR1 size %llu GB — larger than expected\n",
               (unsigned long long)(size >> 30));
    }

    /* Restore original BAR1 values */
    write_pci_config(bdf, bar1_lo_off, orig_lo);
    write_pci_config(bdf, bar1_hi_off, orig_hi);

    /* Re-enable decode */
    write_pci_config(bdf, 0x04, cmd);
    printf("\nBAR1 restored, decode re-enabled.\n");

release:
    /* Release die */
    {
        struct bs_release_req rr = { .die_id = die_id, .flags = 0 };
        ioctl(bs_fd, BAR_SWAP_RELEASE, &rr);
        printf("Die %d released.\n", die_id);
    }
    close(bs_fd);
    return 0;
}
