/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pstraps_probe.c
 * @brief Read / override the BAR1-size strap on a K80 die (PSTRAPS 0x101000).
 *
 *   The K80's 16 GB BAR1 is a strap-selected size, not silicon.
 *   CONFIRMED on GK210: the BAR1-size field
 *   is PSTRAPS bits [17:14] (BAR0+0x101000), value 8 = 16 GB, and override-enable
 *   (bit 31) is already set. This tool acquires a die via barswap (maps its
 *   BAR0), and:
 *     - default:  READ + decode PSTRAPS, and probe the size the BAR1 config
 *                 register currently advertises (write-1s probe, restored).
 *     - "set":    additionally write the field to 256 MB (value 2), re-probe the
 *                 advertised BAR1 size, then RESTORE the strap + BAR1 config
 *                 exactly as found. Fully reversible — no remove/rescan.
 *
 *   The question "set" answers: does a runtime strap write change the size the
 *   device advertises (→ a remove+rescan would then allocate a small BAR), or is
 *   the size latched at reset/enumeration (→ need a pre-enumeration quirk / VBIOS
 *   reflash)?  §15.6.
 *
 *   Usage (root, on a K80 host):  pstraps_probe [die] [set]
 *   Leaves the die mapped, strap restored, and _Exit()s (no teardown, §11.1).
 *
 * @status small-BAR §15.6 — read confirmed; "set" = reversible advertise-size test.
 * @issues remove+rescan (full allocation test) is a deliberate later step.
 * @todo
 *   - [ ] §15.6: if advertised size changes, do remove+rescan to confirm the
 *         kernel ALLOCATES the 256 MB BAR1 (the barswap-obsoleting result).
 */

#define _GNU_SOURCE

#include "barswap/barswap.h"
#include "barswap/gpu_owner_util.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define PSTRAPS_OFFSET  0x101000u
#define PAGE_MASK       (~0xfffULL)
#define BAR1_FIELD_SHIFT 14          /* PSTRAPS bits [17:14] = BAR1 size */
#define BAR1_FIELD_MASK  0xfu
#define BAR1_VAL_16GB    8u
#define BAR1_VAL_256MB   2u
#define CFG_BAR1_LO      0x14        /* BAR1 is a 64-bit BAR: 0x14 (lo) + 0x18 (hi) */
#define CFG_BAR1_HI      0x18

static const char *ladder[] = {
    "64 MB", "128 MB", "256 MB", "512 MB", "1 GB", "2 GB",
    "4 GB", "8 GB", "16 GB", "32 GB", "64 GB"
};

static const char *ladder_name(unsigned v)
{
    return v < (sizeof(ladder) / sizeof(ladder[0])) ? ladder[v] : "(out of ladder)";
}

/* /sys/bus/pci/devices/<bdf>/config 32-bit accessors. */
static int cfg_read32(const char *bdf, unsigned off, uint32_t *out)
{
    char path[256];
    int fd, ok;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/config", bdf);
    fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;
    ok = (pread(fd, out, 4, off) == 4) ? 0 : -1;
    close(fd);
    return ok;
}

static int cfg_write32(const char *bdf, unsigned off, uint32_t val)
{
    char path[256];
    int fd, ok;
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/config", bdf);
    fd = open(path, O_WRONLY);
    if (fd < 0)
        return -1;
    ok = (pwrite(fd, &val, 4, off) == 4) ? 0 : -1;
    close(fd);
    return ok;
}

/* Write-1s probe of the 64-bit BAR1: returns the size the device advertises,
 * restoring the original BAR1 config values. 0 on error. */
static uint64_t probe_bar1_advertised(const char *bdf)
{
    uint32_t lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;
    uint64_t mask;
    if (cfg_read32(bdf, CFG_BAR1_LO, &lo0) || cfg_read32(bdf, CFG_BAR1_HI, &hi0))
        return 0;
    cfg_write32(bdf, CFG_BAR1_LO, 0xffffffffu);
    cfg_write32(bdf, CFG_BAR1_HI, 0xffffffffu);
    cfg_read32(bdf, CFG_BAR1_LO, &lo1);
    cfg_read32(bdf, CFG_BAR1_HI, &hi1);
    cfg_write32(bdf, CFG_BAR1_LO, lo0);   /* restore */
    cfg_write32(bdf, CFG_BAR1_HI, hi0);
    mask = ((uint64_t)hi1 << 32) | (uint64_t)(lo1 & ~0xfu);
    if (mask == 0)
        return 0;
    return (~mask) + 1ULL;
}

int main(int argc, char **argv)
{
    int die = argc > 1 ? atoi(argv[1]) : 0;
    int do_keep = (argc > 2 && strcmp(argv[2], "setkeep") == 0);
    int do_set = do_keep || (argc > 2 && strcmp(argv[2], "set") == 0);
    bs_context *bs = NULL;
    bs_gpu *gpu = NULL;
    int idx = -1, fd = -1;
    const char *bdf = "?";
    uint64_t bar0 = 0, target = 0, page = 0, off = 0, adv = 0;
    volatile uint32_t *map = NULL;
    uint32_t v = 0, vnew = 0;

    setvbuf(stdout, NULL, _IONBF, 0);

    bs = bs_init();
    if (!bs) { printf("[SKIP] no /dev/bar_swap — not on a K80 host\n"); return 0; }
    gou_discover_gk210s();
    gpu = bs_acquire(bs, die);
    if (!gpu) { printf("bs_acquire(%d) failed\n", die); return 1; }
    idx = gou_find_acquired_bdf();
    if (idx >= 0)
        bdf = gou_gk210_bdf(idx);
    bar0 = gou_read_bar0(bdf);
    printf("die %d  bdf %s  BAR0=0x%llx  BAR1=0x%llx (programmed size %llu GB)\n",
           die, bdf, (unsigned long long)bar0, (unsigned long long)gpu->bar1_addr,
           (unsigned long long)(gpu->bar1_size >> 30));
    if (!bar0) { printf("BAR0 unreadable\n"); _Exit(1); }

    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("open /dev/mem"); _Exit(1); }
    target = bar0 + PSTRAPS_OFFSET;
    page = target & PAGE_MASK;
    off = target - page;
    map = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)page);
    if (map == MAP_FAILED) { perror("mmap /dev/mem"); close(fd); _Exit(1); }

    v = map[off / 4];
    printf("PSTRAPS = 0x%08x  (override-enable bit31=%u; [17:14]=%u -> %s)\n",
           v, (v >> 31) & 1u, (v >> BAR1_FIELD_SHIFT) & BAR1_FIELD_MASK,
           ladder_name((v >> BAR1_FIELD_SHIFT) & BAR1_FIELD_MASK));

    adv = probe_bar1_advertised(bdf);
    printf("BAR1 advertised size (config write-1s probe) = %llu MB %s\n",
           (unsigned long long)(adv >> 20),
           adv == 0x400000000ULL ? "(16 GB)" : adv == 0x10000000ULL ? "(256 MB)" : "");

    if (do_set) {
        printf("\n--- SET: writing BAR1 field [17:14] = 2 (256 MB), keeping bit31 ---\n");
        vnew = (v & ~(BAR1_FIELD_MASK << BAR1_FIELD_SHIFT))
             | (BAR1_VAL_256MB << BAR1_FIELD_SHIFT);
        map[off / 4] = vnew;
        (void)map[off / 4];                    /* read-back barrier */
        printf("PSTRAPS now = 0x%08x  ([17:14]=%u -> %s)\n", map[off / 4],
               (map[off / 4] >> BAR1_FIELD_SHIFT) & BAR1_FIELD_MASK,
               ladder_name((map[off / 4] >> BAR1_FIELD_SHIFT) & BAR1_FIELD_MASK));
        adv = probe_bar1_advertised(bdf);
        printf(">>> BAR1 advertised size AFTER strap write = %llu MB %s <<<\n",
               (unsigned long long)(adv >> 20),
               adv == 0x400000000ULL ? "(STILL 16 GB — latched, strap ignored at runtime)"
               : adv == 0x10000000ULL ? "(256 MB — RUNTIME STRAP WORKS!)" : "(unexpected)");
        if (!do_keep) {
            map[off / 4] = v;                  /* restore strap exactly as found */
            (void)map[off / 4];
            printf("PSTRAPS restored to 0x%08x\n", map[off / 4]);
        } else {
            printf("PSTRAPS LEFT at 0x%08x (kept for rescan)\n", map[off / 4]);
        }
    }

    munmap((void *)map, 0x1000);
    close(fd);
    _Exit(0);
}
