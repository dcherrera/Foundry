/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * mmio_probe.c — Quick MMIO accessibility test for a single die.
 *
 * Acquires a die via bar_swap ioctl, then reads NV_PMC_BOOT_0 (offset 0x0)
 * from BAR 0 via /dev/mem. If we get 0xFFFFFFFF, the die is unreachable.
 * A GK210 should return something like 0x0E0000A1.
 *
 * Usage: sudo ./mmio_probe <die_id>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "../bar_swap_ko/bar_swap.h"

int main(int argc, char **argv)
{
    int die_id, fd, mem_fd;
    struct bs_acquire_req acq;
    struct bs_release_req rel;
    void *mapped;
    volatile uint32_t *reg;
    uint32_t boot0, pmc_id;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <die_id>\n", argv[0]);
        return 1;
    }

    die_id = atoi(argv[1]);

    fd = open("/dev/bar_swap", O_RDWR);
    if (fd < 0) {
        perror("open /dev/bar_swap");
        return 1;
    }

    /* Acquire the die */
    memset(&acq, 0, sizeof(acq));
    acq.die_id = die_id;
    if (ioctl(fd, BAR_SWAP_ACQUIRE, &acq) < 0) {
        perror("ACQUIRE ioctl");
        close(fd);
        return 1;
    }

    printf("Die %d acquired: BAR0=0x%llx BAR1=0x%llx BAR3=0x%llx\n",
           die_id,
           (unsigned long long)acq.bar0_addr,
           (unsigned long long)acq.bar1_addr,
           (unsigned long long)acq.bar3_addr);

    /* Verify BARs via config space (setpci) while die is acquired */
    printf("\nConfig space verification:\n");
    {
        char cmd[256], line[128];
        FILE *pp;

        /* Read BAR 0 (reg 0x10), BAR 1 lo (0x14), BAR 1 hi (0x18) */
        snprintf(cmd, sizeof(cmd),
                 "setpci -s 07:00.0 0x10.L 0x14.L 0x18.L 0x1C.L 0x20.L 0x04.W");
        pp = popen(cmd, "r");
        if (pp) {
            int reg_idx = 0;
            const char *names[] = {"BAR0", "BAR1_lo", "BAR1_hi", "BAR3_lo", "BAR3_hi", "CMD"};
            while (fgets(line, sizeof(line), pp) && reg_idx < 6) {
                line[strcspn(line, "\n")] = '\0';
                printf("  %s = 0x%s\n", names[reg_idx], line);
                reg_idx++;
            }
            pclose(pp);
        }

        /* Also check downstream bridge 06:08.0 memory windows */
        printf("\nBridge 06:08.0 windows:\n");
        snprintf(cmd, sizeof(cmd),
                 "setpci -s 06:08.0 0x20.W 0x22.W 0x24.W 0x26.W 0x28.L 0x2C.L 0x04.W");
        pp = popen(cmd, "r");
        if (pp) {
            int reg_idx = 0;
            const char *names[] = {"MEM_BASE", "MEM_LIMIT", "PREF_BASE", "PREF_LIMIT",
                                   "PREF_BASE_HI", "PREF_LIMIT_HI", "CMD"};
            while (fgets(line, sizeof(line), pp) && reg_idx < 7) {
                line[strcspn(line, "\n")] = '\0';
                printf("  %s = 0x%s\n", names[reg_idx], line);
                reg_idx++;
            }
            pclose(pp);
        }
    }

    /* Map BAR 0 via /dev/mem (first 4 KB page) */
    mem_fd = open("/dev/mem", O_RDONLY | O_SYNC);
    if (mem_fd < 0) {
        perror("open /dev/mem");
        goto release;
    }

    mapped = mmap(NULL, 4096, PROT_READ, MAP_SHARED, mem_fd, acq.bar0_addr);
    if (mapped == MAP_FAILED) {
        perror("mmap BAR 0");
        close(mem_fd);
        goto release;
    }

    reg = (volatile uint32_t *)mapped;

    /* NV_PMC_BOOT_0 at offset 0x0 — GPU identification register */
    boot0 = reg[0];

    /* NV_PMC_BOOT_42 at offset 0x0 is actually PMC_BOOT_0.
     * Also read offset 0x4 (PMC_INTR) for another data point */
    pmc_id = reg[1];

    printf("BAR0[0x00] (NV_PMC_BOOT_0) = 0x%08x\n", boot0);
    printf("BAR0[0x04]                  = 0x%08x\n", pmc_id);

    if (boot0 == 0xFFFFFFFF) {
        printf("\n** DIE IS UNREACHABLE (fallen off bus) **\n");
    } else {
        printf("\nDie responds! Implementation ID: 0x%02x, Chip: 0x%03x\n",
               (boot0 >> 20) & 0xFF, (boot0 >> 0) & 0xFFF);
    }

    /* Read a few more offsets for diagnostics */
    printf("\nFirst 16 registers:\n");
    for (int i = 0; i < 16; i++)
        printf("  [0x%02x] = 0x%08x\n", i * 4, reg[i]);

    munmap(mapped, 4096);
    close(mem_fd);

release:
    memset(&rel, 0, sizeof(rel));
    rel.die_id = die_id;
    ioctl(fd, BAR_SWAP_RELEASE, &rel);
    close(fd);

    printf("\nDie %d released.\n", die_id);
    return (boot0 == 0xFFFFFFFF) ? 1 : 0;
}
