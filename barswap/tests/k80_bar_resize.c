/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file k80_bar_resize.c
 * @brief Shrink every K80/GK210 BAR1 16 GB -> 256 MB and re-enumerate (all 16).
 *
 *   The one-command runtime automation: for
 *   each GK210, write the BAR1-size strap (PSTRAPS BAR0+0x101000 [15:14]=2 +
 *   override bit 31) via its boot-assigned BAR0, then `remove`+`rescan` so the
 *   kernel allocates the small BAR1 (16 GB fits nowhere; 256 MB does).
 *
 *   v2 — reach the "broken bridge" dies: when a die's 16 GB BAR1 failed at boot,
 *   the kernel disables decode up its bridge chain, so a plain BAR0 write never
 *   reaches the chip and the strap silently no-ops (this stranded 2/16). Fix:
 *   before strapping each die, walk its PCI ancestor chain and enable memory
 *   decode (command reg bit 1) on every bridge + the die, so BAR0 routes and the
 *   strap takes. Then all 16 advertise 256 MB and the rescan mounts them.
 *
 *   Usage (root):  k80_bar_resize
 *
 * @status Automation — runtime resize, bridge-chain decode-enable for all 16.
 * @issues The truly clean every-boot version is a built-in pci_fixup_early
 *   (loadable modules load after PCI enum); this runtime tool is the practical
 *   equivalent on stock kernels.
 * @todo None
 */

#define _GNU_SOURCE

#include "barswap/gpu_owner_util.h"

#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define PSTRAPS_OFFSET   0x101000u
#define PAGE_MASK        (~0xfffULL)
#define BAR1_FIELD_SHIFT 14
#define BAR1_FIELD_MASK  0xfu
#define BAR1_VAL_256MB   2u
#define BAR1_256MB       0x10000000ULL
#define CFG_COMMAND      0x04
#define CFG_MEM_DECODE   0x02

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

static void enable_decode(const char *bdf)
{
	uint32_t cmd = 0;
	if (cfg_rw32(bdf, CFG_COMMAND, &cmd, 0) == 0 && !(cmd & CFG_MEM_DECODE)) {
		cmd |= CFG_MEM_DECODE;
		cfg_rw32(bdf, CFG_COMMAND, &cmd, 1);
	}
}

static int is_bdf(const char *s)
{
	return strlen(s) == 12 && s[4] == ':' && s[7] == ':' && s[10] == '.';
}

/* Walk the die's PCI ancestor chain (root port -> PLX bridges -> die) and enable
 * memory decode on every node, so the die's BAR0 actually routes. */
static void enable_chain_decode(const char *die_bdf)
{
	char path[256], real[PATH_MAX], *tok;
	snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s", die_bdf);
	if (!realpath(path, real))
		return;
	for (tok = strtok(real, "/"); tok; tok = strtok(NULL, "/"))
		if (is_bdf(tok))
			enable_decode(tok);
}

static uint64_t resource_size(const char *bdf, int idx)
{
	char path[256], line[128];
	uint64_t start = 0, end = 0, size = 0;
	int i;
	FILE *fp;
	snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/resource", bdf);
	fp = fopen(path, "r");
	if (!fp)
		return 0;
	for (i = 0; i <= idx; i++) {
		if (!fgets(line, sizeof(line), fp))
			break;
		if (i == idx && sscanf(line, "%llx %llx", (unsigned long long *)&start,
				       (unsigned long long *)&end) == 2 && end > start)
			size = end - start + 1;
	}
	fclose(fp);
	return size;
}

static int shrink_die(const char *bdf)
{
	uint64_t bar0, target, page, off;
	int fd, rc = -1;
	volatile uint32_t *map;
	uint32_t v;

	enable_chain_decode(bdf);                    /* v2: route BAR0 even on broken bridges */
	bar0 = gou_read_bar0(bdf);
	if (!bar0)
		return -1;
	fd = open("/dev/mem", O_RDWR | O_SYNC);
	if (fd < 0)
		return -1;
	target = bar0 + PSTRAPS_OFFSET;
	page = target & PAGE_MASK;
	off = target - page;
	map = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)page);
	if (map != MAP_FAILED) {
		v = map[off / 4];
		/* sanity: a routed PSTRAPS has bit31 (override) set on these cards and a
		 * sane field; all-ones means the read didn't route. */
		if (v != 0xffffffffu) {
			map[off / 4] = (v & ~(BAR1_FIELD_MASK << BAR1_FIELD_SHIFT))
				     | (BAR1_VAL_256MB << BAR1_FIELD_SHIFT);
			(void)map[off / 4];
			rc = (map[off / 4] != 0xffffffffu) ? 0 : -1;
		}
		munmap((void *)map, 0x1000);
	}
	close(fd);
	return rc;
}

int main(void)
{
	int n, i, shrunk = 0, mounted = 0;
	struct timespec ts = { .tv_sec = 3, .tv_nsec = 0 };

	setvbuf(stdout, NULL, _IONBF, 0);
	n = gou_discover_gk210s();
	printf("k80_bar_resize: %d GK210 dies\n", n);
	if (n <= 0)
		return 1;

	for (i = 0; i < n; i++) {
		if (shrink_die(gou_gk210_bdf(i)) == 0)
			shrunk++;
		else
			printf("  %s: strap write did not route (still unreachable)\n",
			       gou_gk210_bdf(i));
	}
	printf("strapped 256 MB (verified routed): %d / %d\n", shrunk, n);

	for (i = 0; i < n; i++) {
		char p[256];
		snprintf(p, sizeof(p), "/sys/bus/pci/devices/%s/remove", gou_gk210_bdf(i));
		gou_sysfs_write(p, "1\n");
	}
	gou_sysfs_write("/sys/bus/pci/rescan", "1\n");
	nanosleep(&ts, NULL);

	gou_discover_gk210s();
	n = gou_num_gk210();
	for (i = 0; i < n; i++)
		if (resource_size(gou_gk210_bdf(i), 1) == BAR1_256MB)
			mounted++;
	printf(">>> mounted 256 MB BAR1: %d / %d <<<\n", mounted, n);
	return 0;
}
