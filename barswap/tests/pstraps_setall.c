/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pstraps_setall.c
 * @brief Shrink BAR1 to 256 MB on GK210 dies via the BOOT-assigned BAR0 (no barswap).
 *
 *   Confirmed: at boot each die's BAR0 (16 MB)
 *   IS assigned; only the 16 GB BAR1 (and 32 MB BAR3) fail for lack of a
 *   prefetchable window. So we can reach PSTRAPS (BAR0+0x101000) on every die
 *   WITHOUT barswap: enable the device's memory decode, mmap its BAR0, and write
 *   the BAR1-size field [17:14] = 2 (256 MB). After this the dies ADVERTISE
 *   256 MB; a subsequent `remove`+`rescan` (+ pci=realloc) should let the kernel
 *   ALLOCATE the small BARs — sixteen of them fit trivially.
 *
 *   The strap is LEFT SET (not restored) so the rescan sees the small size.
 *
 *   Usage (root, NO barswap loaded):  pstraps_setall [all | <die-index>]
 *   Prints each die's BAR0, PSTRAPS before/after, and the advertised BAR1 size
 *   before/after (config write-1s probe). _Exit()s.
 *
 * @status small-BAR §15.6 — the all-16 strap-set, no-barswap path.
 * @issues Does NOT trigger rescan (done from shell so we can watch lspci).
 * @todo None
 */

#define _GNU_SOURCE

#include "barswap/gpu_owner_util.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PSTRAPS_OFFSET   0x101000u
#define PAGE_MASK        (~0xfffULL)
#define BAR1_FIELD_SHIFT 14
#define BAR1_FIELD_MASK  0xfu
#define BAR1_VAL_256MB   2u
#define CFG_COMMAND      0x04
#define CFG_MEM_DECODE   0x02
#define CFG_BAR1_LO      0x14
#define CFG_BAR1_HI      0x18

static int cfg_rw32(const char *bdf, unsigned off, uint32_t *val, int write)
{
    char path[256];
    int fd, ok;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/config", bdf);
    fd = open(path, write ? O_WRONLY : O_RDONLY);
    if (fd < 0)
        return -1;
    ok = (write ? pwrite(fd, val, 4, off) : pread(fd, val, 4, off)) == 4 ? 0 : -1;
    close(fd);
    return ok;
}

static void enable_mem_decode(const char *bdf)
{
    uint32_t cmd = 0;
    if (cfg_rw32(bdf, CFG_COMMAND, &cmd, 0) == 0 && !(cmd & CFG_MEM_DECODE)) {
        cmd |= CFG_MEM_DECODE;
        cfg_rw32(bdf, CFG_COMMAND, &cmd, 1);
    }
}

static uint64_t probe_bar1_advertised(const char *bdf)
{
    uint32_t lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0, ones = 0xffffffffu;
    uint64_t mask;
    if (cfg_rw32(bdf, CFG_BAR1_LO, &lo0, 0) || cfg_rw32(bdf, CFG_BAR1_HI, &hi0, 0))
        return 0;
    cfg_rw32(bdf, CFG_BAR1_LO, &ones, 1);
    cfg_rw32(bdf, CFG_BAR1_HI, &ones, 1);
    cfg_rw32(bdf, CFG_BAR1_LO, &lo1, 0);
    cfg_rw32(bdf, CFG_BAR1_HI, &hi1, 0);
    cfg_rw32(bdf, CFG_BAR1_LO, &lo0, 1);   /* restore */
    cfg_rw32(bdf, CFG_BAR1_HI, &hi0, 1);
    mask = ((uint64_t)hi1 << 32) | (uint64_t)(lo1 & ~0xfu);
    return mask ? (~mask) + 1ULL : 0;
}

static int set_die(const char *bdf)
{
    uint64_t bar0, target, page, off, before, after;
    int fd;
    volatile uint32_t *map;
    uint32_t v, vnew;

    enable_mem_decode(bdf);
    bar0 = gou_read_bar0(bdf);
    if (!bar0) {
        printf("  %s: BAR0 unassigned — skip\n", bdf);
        return -1;
    }
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        printf("  %s: open /dev/mem failed\n", bdf);
        return -1;
    }
    target = bar0 + PSTRAPS_OFFSET;
    page = target & PAGE_MASK;
    off = target - page;
    map = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)page);
    if (map == MAP_FAILED) {
        printf("  %s: mmap failed\n", bdf);
        close(fd);
        return -1;
    }
    v = map[off / 4];
    before = probe_bar1_advertised(bdf);
    vnew = (v & ~(BAR1_FIELD_MASK << BAR1_FIELD_SHIFT)) | (BAR1_VAL_256MB << BAR1_FIELD_SHIFT);
    map[off / 4] = vnew;
    (void)map[off / 4];
    after = probe_bar1_advertised(bdf);
    printf("  %s  BAR0=0x%09llx  PSTRAPS 0x%08x->0x%08x  BAR1 adv %llu MB -> %llu MB%s\n",
           bdf, (unsigned long long)bar0, v, map[off / 4],
           (unsigned long long)(before >> 20), (unsigned long long)(after >> 20),
           after == 0x10000000ULL ? "  [256 MB OK]" : "");
    munmap((void *)map, 0x1000);
    close(fd);
    return 0;
}

int main(int argc, char **argv)
{
    int n, i, one = -1;

    setvbuf(stdout, NULL, _IONBF, 0);
    n = gou_discover_gk210s();
    printf("discovered %d GK210 dies\n", n);
    if (n <= 0)
        return 1;

    if (argc > 1 && strcmp(argv[1], "all") != 0)
        one = atoi(argv[1]);

    if (one >= 0 && one < n) {
        set_die(gou_gk210_bdf(one));
    } else {
        for (i = 0; i < n; i++)
            set_die(gou_gk210_bdf(i));
    }
    _Exit(0);
}
