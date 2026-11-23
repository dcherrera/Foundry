/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_pci_sysfs.c
 * @brief Gate for ft_tt_udrv_pci_sysfs.c, driven against a SYNTHETIC sysfs
 *   tree built in a scratch directory.
 *
 *   A fixture is what makes this testable at all: the cases that matter are
 *   the ones a real card cannot produce on demand — an absent BAR, a
 *   non-prefetchable BAR, a one-byte BAR, a truncated `resource` file, a
 *   device whose ids do not parse. The real card contributes exactly one of
 *   them (BAR4 absent) and only by accident of this board's 36-bit host.
 *
 *   The reference numbers in stage 1 are the REAL card's, read from
 *   /sys/bus/pci/devices/0000:12:00.0 on the card host, so the parser is held to
 *   what the machine actually prints and not to what this file imagines.
 *
 *   WHAT THIS CANNOT PROVE, stated so it is never assumed: that the numbers
 *   describe hardware, that a BAR's sysfs file can be opened as root or under
 *   a udev rule, that the PCI core's mmap path behaves like a regular file's,
 *   or that a write-combined store reaches the card. Stage 7 maps a REAL file
 *   through map_bar, which covers the path building, the offset arithmetic and
 *   the protection — everything except the fact that the far end is silicon.
 *   `tests/test_udrv_card_tlb.c` is where that is settled.
 *
 * @status Off-hardware. Touches nothing but a scratch directory.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ft_tt_test_report.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_pci_sysfs.h"

/** Size of the fixture's real, mappable `resource0`. Sixteen pages. */
#define FIXTURE_RESOURCE_BYTES 65536u

/** The real card's BAR table, copied from `cat resource` on the card host. */
#define REAL_BAR0_LINE "0x00000005e0000000 0x00000005ffffffff 0x000000000014220c\n"
#define REAL_BAR2_LINE "0x0000000600000000 0x00000006000fffff 0x000000000014220c\n"
#define ABSENT_LINE "0x0000000000000000 0x0000000000000000 0x0000000000000000\n"

/** BAR2 as a NON-prefetchable memory BAR: the same line without bit 13. */
#define NONPREFETCH_BAR2_LINE "0x0000000600000000 0x00000006000fffff 0x000000000014020c\n"

/** A one-byte memory BAR: end == start != 0. Length must be 1, not 0. */
#define ONE_BYTE_LINE "0x0000000700000000 0x0000000700000000 0x0000000000000200\n"

static char scratch_root[256];

/** mkdir, aborting the run if it fails — a broken fixture is not a result. */
static void make_directory(const char *path)
{
    if (mkdir(path, 0755) != 0) {
        (void)fprintf(stderr, "fixture: mkdir %s failed\n", path);
        exit(2);
    }
}

/** Write one small file into a fixture device directory. */
static void write_file(const char *device_dir, const char *name, const char *contents)
{
    char path[512];
    FILE *file;

    (void)snprintf(path, sizeof(path), "%s/%s", device_dir, name);
    file = fopen(path, "w");
    if (!file) {
        (void)fprintf(stderr, "fixture: fopen %s failed\n", path);
        exit(2);
    }
    (void)fputs(contents, file);
    (void)fclose(file);
}

/** Build one fixture device. `resource_body` may be NULL to omit the file. */
static void make_device(const char *address, const char *vendor, const char *device,
                        const char *resource_body)
{
    char dir[512];

    (void)snprintf(dir, sizeof(dir), "%s/%s", scratch_root, address);
    make_directory(dir);

    write_file(dir, "vendor", vendor);
    write_file(dir, "device", device);
    write_file(dir, "subsystem_vendor", "0x1e52\n");
    write_file(dir, "subsystem_device", "0x0040\n");
    write_file(dir, "class", "0x120000\n");
    write_file(dir, "revision", "0x00\n");

    if (resource_body)
        write_file(dir, "resource", resource_body);
}

/**
 * Give a fixture device a real 64 KiB `resource0` full of a pattern whose
 * value depends on the byte's OFFSET — so an implementation that maps the
 * wrong page reads different bytes rather than plausible ones.
 *
 * The `>> 8` term is not decoration and it is not arbitrary. The first
 * version of this was `(i * 7 + 11) & 0xFF`, which repeats every 256 bytes;
 * every page therefore held identical bytes, and a mutation that mapped
 * offset 0 instead of the offset it was given PASSED. That is precisely the
 * failure mode this whole phase is about — a wrong aim is a wrong ADDRESS,
 * not an error — reproduced in a fixture. The pattern must be aperiodic
 * across pages or the offset is not being tested at all.
 */
static void make_mappable_resource(const char *address)
{
    unsigned char page[FIXTURE_RESOURCE_BYTES];
    char path[512];
    size_t index;
    FILE *file;

    for (index = 0u; index < sizeof(page); index++)
        page[index] = (unsigned char)((index * 7u + 11u + (index >> 8)) & 0xFFu);

    (void)snprintf(path, sizeof(path), "%s/%s/resource0", scratch_root, address);
    file = fopen(path, "wb");
    if (!file) {
        (void)fprintf(stderr, "fixture: fopen %s failed\n", path);
        exit(2);
    }
    if (fwrite(page, 1u, sizeof(page), file) != sizeof(page)) {
        (void)fprintf(stderr, "fixture: short write to %s\n", path);
        exit(2);
    }
    (void)fclose(file);
}

static void build_fixture(void)
{
    (void)snprintf(scratch_root, sizeof(scratch_root), "/tmp/ft_tt_udrv_pci_fixture.%ld",
                   (long)getpid());
    make_directory(scratch_root);

    /* The real card: prefetchable 512 MiB BAR0, 1 MiB BAR2, no BAR4. */
    make_device("0000:12:00.0", "0x1e52\n", "0xb140\n",
                REAL_BAR0_LINE ABSENT_LINE REAL_BAR2_LINE ABSENT_LINE ABSENT_LINE ABSENT_LINE);

    /* A second Tenstorrent card at a LOWER address, to prove ordering. */
    make_device("0000:03:00.0", "0x1e52\n", "0xb140\n",
                REAL_BAR0_LINE ABSENT_LINE REAL_BAR2_LINE ABSENT_LINE ABSENT_LINE ABSENT_LINE);

    /* Right vendor, wrong part — must not be handed to a Blackhole caller. */
    make_device("0000:04:00.0", "0x1e52\n", "0x401e\n",
                REAL_BAR0_LINE ABSENT_LINE REAL_BAR2_LINE ABSENT_LINE ABSENT_LINE ABSENT_LINE);

    /* Somebody else's device entirely. */
    make_device("0000:05:00.0", "0x10de\n", "0x102d\n",
                REAL_BAR0_LINE ABSENT_LINE REAL_BAR2_LINE ABSENT_LINE ABSENT_LINE ABSENT_LINE);

    /* Non-prefetchable BAR2 and a one-byte BAR3, neither of which the real
     * card can be made to produce. */
    make_device("0000:06:00.0", "0x1e52\n", "0xb140\n",
                REAL_BAR0_LINE ABSENT_LINE NONPREFETCH_BAR2_LINE ONE_BYTE_LINE ABSENT_LINE
                    ABSENT_LINE);

    /* Malformed: only four BAR lines. */
    make_device("0000:07:00.0", "0x1e52\n", "0xb140\n",
                REAL_BAR0_LINE ABSENT_LINE REAL_BAR2_LINE ABSENT_LINE);

    /* Malformed: `resource` missing entirely. */
    make_device("0000:08:00.0", "0x1e52\n", "0xb140\n", NULL);

    /* Malformed: vendor is not a number. */
    make_device("0000:09:00.0", "banana\n", "0xb140\n",
                REAL_BAR0_LINE ABSENT_LINE REAL_BAR2_LINE ABSENT_LINE ABSENT_LINE ABSENT_LINE);

    /* A device with a REAL, mappable `resource0`, so stage 7 can exercise
     * map_bar's plumbing. Its vendor is deliberately NOT Tenstorrent's — every
     * enumeration count in stage 5 would otherwise move, and a fixture that
     * makes an unrelated stage change is a fixture people stop trusting. */
    make_device("0000:0a:00.0", "0x1e53\n", "0xb140\n",
                REAL_BAR0_LINE ABSENT_LINE REAL_BAR2_LINE ABSENT_LINE ABSENT_LINE ABSENT_LINE);
    make_mappable_resource("0000:0a:00.0");
}

static void remove_fixture(void)
{
    char command[512];

    /* The fixture is ours, freshly made under a pid-qualified name. */
    (void)snprintf(command, sizeof(command), "rm -rf '%s'", scratch_root);
    if (system(command) != 0)
        (void)fprintf(stderr, "fixture: cleanup of %s failed\n", scratch_root);
}

static void device_path(const char *address, char *out_path, size_t path_bytes)
{
    (void)snprintf(out_path, path_bytes, "%s/%s", scratch_root, address);
}

static void stage_the_real_card(void)
{
    ft_tt_udrv_pci_device device;
    char path[512];

    ft_tt_test_step("Stage 1 — the real card's own sysfs numbers parse correctly");

    device_path("0000:12:00.0", path, sizeof(path));
    ft_tt_test_check("0000:12:00.0 reads",
                     ft_tt_udrv_pci_read_device(path, &device) == FT_TT_UDRV_OK, "");

    ft_tt_test_check_fmt("vendor is Tenstorrent",
                         device.vendor == FT_TT_UDRV_PCI_VENDOR_TENSTORRENT, "0x%04x",
                         device.vendor);
    ft_tt_test_check_fmt("device is Blackhole", device.device == FT_TT_UDRV_PCI_DEVICE_BLACKHOLE,
                         "0x%04x", device.device);
    ft_tt_test_check_fmt("address is the basename", strcmp(device.address, "0000:12:00.0") == 0,
                         "%s", device.address);

    ft_tt_test_check_fmt("BAR0 is 512 MiB", device.bar[0].length == (512ull << 20), "%llu bytes",
                         (unsigned long long)device.bar[0].length);
    ft_tt_test_check_fmt("BAR0 starts at 0x5e0000000", device.bar[0].start == 0x5e0000000ull,
                         "0x%llx", (unsigned long long)device.bar[0].start);
    ft_tt_test_check("BAR0 is prefetchable — this is what makes resource0_wc exist",
                     device.bar[0].prefetchable, "flags bit 13");
    ft_tt_test_check("BAR0 is a 64-bit memory BAR",
                     device.bar[0].memory && device.bar[0].is_64bit, "flags bits 9 and 20");

    ft_tt_test_check_fmt("BAR1 is absent (the upper half of the 64-bit BAR0)",
                         !device.bar[1].present && device.bar[1].length == 0u, "len=%llu",
                         (unsigned long long)device.bar[1].length);

    ft_tt_test_check_fmt("BAR2 is 1 MiB", device.bar[2].length == (1ull << 20), "%llu bytes",
                         (unsigned long long)device.bar[2].length);

    ft_tt_test_check("BAR4 is absent — this board has no 4 GiB TLB windows",
                     !device.bar[4].present, "36-bit host, pcie_bar4_size 0");
}

static void stage_length_arithmetic(void)
{
    ft_tt_udrv_pci_device device;
    char path[512];

    ft_tt_test_step("Stage 2 — length follows the KERNEL's rule, not a plausible one");

    device_path("0000:06:00.0", path, sizeof(path));
    ft_tt_test_check("0000:06:00.0 reads",
                     ft_tt_udrv_pci_read_device(path, &device) == FT_TT_UDRV_OK, "");

    /* pci_resource_len() is "0 when end is 0", NOT "0 when end <= start". A
     * one-byte BAR has end == start != 0 and a length of 1; a rule written as
     * `end > start` calls it absent and silently loses the aperture. */
    ft_tt_test_check_fmt("a one-byte BAR has length 1, not 0", device.bar[3].length == 1u,
                         "%llu", (unsigned long long)device.bar[3].length);
    ft_tt_test_check("a one-byte BAR is present", device.bar[3].present, "len 1");

    ft_tt_test_check("a non-prefetchable BAR2 is reported as such",
                     !device.bar[2].prefetchable, "flags bit 13 clear");
    ft_tt_test_check_fmt("...and still has its full length", device.bar[2].length == (1ull << 20),
                         "%llu", (unsigned long long)device.bar[2].length);
}

static void stage_resource_paths(void)
{
    ft_tt_udrv_pci_device real;
    ft_tt_udrv_pci_device odd;
    char path[512];
    char built[FT_TT_UDRV_PCI_PATH_MAX];
    char expected[FT_TT_UDRV_PCI_PATH_MAX];

    ft_tt_test_step("Stage 3 — resource paths, and the _wc file that only sometimes exists");

    device_path("0000:12:00.0", path, sizeof(path));
    (void)ft_tt_udrv_pci_read_device(path, &real);
    device_path("0000:06:00.0", path, sizeof(path));
    (void)ft_tt_udrv_pci_read_device(path, &odd);

    (void)snprintf(expected, sizeof(expected), "%s/0000:12:00.0/resource0_wc", scratch_root);
    ft_tt_test_check_fmt("BAR0 write-combined path",
                         ft_tt_udrv_pci_resource_path(&real, 0u, true, built, sizeof(built)) ==
                                 FT_TT_UDRV_OK &&
                             strcmp(built, expected) == 0,
                         "%s", built);

    (void)snprintf(expected, sizeof(expected), "%s/0000:12:00.0/resource2", scratch_root);
    ft_tt_test_check_fmt("BAR2 uncached path",
                         ft_tt_udrv_pci_resource_path(&real, 2u, false, built, sizeof(built)) ==
                                 FT_TT_UDRV_OK &&
                             strcmp(built, expected) == 0,
                         "%s", built);

    ft_tt_test_check("a _wc path on a NON-prefetchable BAR is refused",
                     ft_tt_udrv_pci_resource_path(&odd, 2u, true, built, sizeof(built)) ==
                         FT_TT_UDRV_EINVAL,
                     "the PCI core never created that file");

    ft_tt_test_check("a path for an ABSENT BAR is refused",
                     ft_tt_udrv_pci_resource_path(&real, 4u, false, built, sizeof(built)) ==
                         FT_TT_UDRV_EINVAL,
                     "BAR4 does not exist here");

    ft_tt_test_check("a path for BAR 6 is refused",
                     ft_tt_udrv_pci_resource_path(&real, 6u, false, built, sizeof(built)) ==
                         FT_TT_UDRV_EINVAL,
                     "only 6 BARs");

    ft_tt_test_check("a too-small output buffer is refused",
                     ft_tt_udrv_pci_resource_path(&real, 0u, true, built, 8u) == FT_TT_UDRV_EINVAL,
                     "would truncate");
}

static void stage_malformed_inputs(void)
{
    ft_tt_udrv_pci_device device;
    char path[512];

    ft_tt_test_step("Stage 4 — malformed sysfs is refused, not silently zeroed");

    device_path("0000:07:00.0", path, sizeof(path));
    ft_tt_test_check("a `resource` with only 4 BAR lines is refused",
                     ft_tt_udrv_pci_read_device(path, &device) == FT_TT_UDRV_ENODEV,
                     "a short file would read as two absent BARs");

    device_path("0000:08:00.0", path, sizeof(path));
    ft_tt_test_check("a missing `resource` file is refused",
                     ft_tt_udrv_pci_read_device(path, &device) == FT_TT_UDRV_ENODEV, "");

    device_path("0000:09:00.0", path, sizeof(path));
    ft_tt_test_check("an unparseable `vendor` is refused",
                     ft_tt_udrv_pci_read_device(path, &device) == FT_TT_UDRV_ENODEV,
                     "\"banana\"");

    device_path("0000:99:00.0", path, sizeof(path));
    ft_tt_test_check("a device directory that does not exist is refused",
                     ft_tt_udrv_pci_read_device(path, &device) == FT_TT_UDRV_ENODEV, "");

    ft_tt_test_check("a NULL directory is refused",
                     ft_tt_udrv_pci_read_device(NULL, &device) == FT_TT_UDRV_EINVAL, "");

    ft_tt_test_check("a NULL output is refused",
                     ft_tt_udrv_pci_read_device("/tmp", NULL) == FT_TT_UDRV_EINVAL, "");
}

static void stage_enumeration(void)
{
    ft_tt_udrv_pci_device found[4];
    size_t count = 0u;

    ft_tt_test_step("Stage 5 — enumeration matches on ids and orders by address");

    ft_tt_test_check("scanning the fixture succeeds",
                     ft_tt_udrv_pci_find_devices(scratch_root, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT,
                                                 FT_TT_UDRV_PCI_DEVICE_BLACKHOLE, found, 4u,
                                                 &count) == FT_TT_UDRV_OK,
                     "");

    /* Three well-formed Blackholes: 03, 06, 12. The Wormhole at 04, the NVIDIA
     * device at 05, and the three malformed ones must all be skipped. */
    ft_tt_test_check_fmt("exactly 3 Blackholes matched", count == 3u, "%zu", count);

    if (count >= 3u) {
        ft_tt_test_check_fmt("device 0 is the lowest address",
                             strcmp(found[0].address, "0000:03:00.0") == 0, "%s",
                             found[0].address);
        ft_tt_test_check_fmt("device 1 is next", strcmp(found[1].address, "0000:06:00.0") == 0,
                             "%s", found[1].address);
        ft_tt_test_check_fmt("device 2 is the highest",
                             strcmp(found[2].address, "0000:12:00.0") == 0, "%s",
                             found[2].address);
    }

    count = 0u;
    (void)ft_tt_udrv_pci_find_devices(scratch_root, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT, 0xFFFFu,
                                      found, 4u, &count);
    ft_tt_test_check_fmt("\"any device id\" also matches the Wormhole", count == 4u, "%zu",
                         count);

    count = 0u;
    (void)ft_tt_udrv_pci_find_devices(scratch_root, 0x1234u, 0xFFFFu, found, 4u, &count);
    ft_tt_test_check_fmt("an unknown vendor matches nothing", count == 0u, "%zu", count);

    count = 99u;
    ft_tt_test_check("a directory that does not exist is refused",
                     ft_tt_udrv_pci_find_devices("/tmp/ft_tt_udrv_no_such_dir", 0x1e52u, 0xFFFFu,
                                                 found, 4u, &count) == FT_TT_UDRV_ENODEV,
                     "");

    count = 0u;
    (void)ft_tt_udrv_pci_find_devices(scratch_root, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT,
                                      FT_TT_UDRV_PCI_DEVICE_BLACKHOLE, found, 1u, &count);
    ft_tt_test_check_fmt("an undersized buffer still reports the true count", count == 3u, "%zu",
                         count);
}

static void stage_map_refusals(void)
{
    ft_tt_udrv_pci_device real;
    ft_tt_udrv_pci_device odd;
    void *mapping = (void *)0x1;
    char path[512];
    long page = sysconf(_SC_PAGESIZE);

    ft_tt_test_step("Stage 6 — map_bar refuses every bad range BEFORE it opens anything");

    device_path("0000:12:00.0", path, sizeof(path));
    (void)ft_tt_udrv_pci_read_device(path, &real);
    device_path("0000:06:00.0", path, sizeof(path));
    (void)ft_tt_udrv_pci_read_device(path, &odd);

    ft_tt_test_check("an absent BAR is refused",
                     ft_tt_udrv_pci_map_bar(&real, 4u, false, false, 0u, 4096u, &mapping) ==
                         FT_TT_UDRV_EINVAL,
                     "BAR4 does not exist here");
    ft_tt_test_check("...and the caller's pointer is cleared, not left stale", mapping == NULL,
                     "NULL");

    ft_tt_test_check("BAR 6 is refused",
                     ft_tt_udrv_pci_map_bar(&real, 6u, false, false, 0u, 4096u, &mapping) ==
                         FT_TT_UDRV_EINVAL,
                     "only 6 BARs");

    ft_tt_test_check("a zero length is refused",
                     ft_tt_udrv_pci_map_bar(&real, 0u, false, false, 0u, 0u, &mapping) ==
                         FT_TT_UDRV_EINVAL,
                     "mmap would fail anyway; refuse where the reason is visible");

    ft_tt_test_check("a NULL output is refused",
                     ft_tt_udrv_pci_map_bar(&real, 0u, false, false, 0u, 4096u, NULL) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    /* A wrong aim is a wrong ADDRESS, not an error — so the range arithmetic
     * is checked against the length the KERNEL published, and it is checked
     * here rather than trusted. */
    ft_tt_test_check("an offset past the end of the BAR is refused",
                     ft_tt_udrv_pci_map_bar(&real, 0u, false, false, 512ull << 20,
                                            (size_t)page, &mapping) == FT_TT_UDRV_EINVAL,
                     "BAR0 is exactly 512 MiB, so 512 MiB is one past the end");

    ft_tt_test_check("a length that runs off the end of the BAR is refused",
                     ft_tt_udrv_pci_map_bar(&real, 0u, false, false,
                                            (512ull << 20) - (uint64_t)page, (size_t)page * 2u,
                                            &mapping) == FT_TT_UDRV_EINVAL,
                     "last page + one more");

    ft_tt_test_check("a misaligned offset is refused",
                     ft_tt_udrv_pci_map_bar(&real, 0u, false, false, 0x1FC00004ull, 4096u,
                                            &mapping) == FT_TT_UDRV_EINVAL,
                     "mmap needs a page-aligned offset; the PCI core would map a whole page");

    ft_tt_test_check("a write-combined map of a NON-prefetchable BAR is refused",
                     ft_tt_udrv_pci_map_bar(&odd, 2u, true, false, 0u, 4096u, &mapping) ==
                         FT_TT_UDRV_EINVAL,
                     "the PCI core never created resource2_wc for it");

    /* The range check must not be what refuses a LEGAL range. This one is
     * legal and fails at open(), which is the fixture having no resource0 —
     * i.e. the two refusals are distinguishable. */
    ft_tt_test_check("a legal range on a BAR with no resource file fails at open, not at the "
                     "range check",
                     ft_tt_udrv_pci_map_bar(&real, 2u, false, false, 0u, (size_t)page,
                                            &mapping) == FT_TT_UDRV_EIO,
                     "EIO, not EINVAL");

    ft_tt_test_check("unmapping NULL is a no-op success",
                     ft_tt_udrv_pci_unmap_bar(NULL, 4096u) == FT_TT_UDRV_OK, "");
    ft_tt_test_check("unmapping with a zero length is refused",
                     ft_tt_udrv_pci_unmap_bar((void *)&real, 0u) == FT_TT_UDRV_EINVAL, "");

    /* The alignment rule is the HOST's page size, which is 4096 on the card host
     * and 16384 on the Mac this is written on. Nothing here may assume 4096. */
    ft_tt_test_check_fmt("the host page size is a power of two", page > 0 && (page & (page - 1)) == 0,
                         "%ld", page);
}

/**
 * Stage 7 — map a REAL file through map_bar.
 *
 * The fixture device at 0000:0a:00.0 has a genuine 64 KiB `resource0` full of
 * a known pattern. mmap does not care that it is a regular file rather than a
 * BAR, so this exercises the whole of map_bar's plumbing — path building,
 * open, offset arithmetic, prot — off-hardware, and it is the ONLY part of
 * that plumbing a fixture can reach.
 *
 * The child-process arm is the point of the `writable` parameter: the
 * read-only rung's whole safety claim is that a stray store CANNOT reach a
 * register, and "we were careful" is not that claim. A fork that stores into
 * a PROT_READ mapping must die on a signal.
 */
static void stage_map_a_real_file(void)
{
    ft_tt_udrv_pci_device device;
    volatile unsigned char *bytes;
    void *mapping = NULL;
    char path[512];
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    pid_t child;
    int status = 0;

    ft_tt_test_step("Stage 7 — map_bar really maps, and a read-only map really is read-only");

    device_path("0000:0a:00.0", path, sizeof(path));
    ft_tt_test_check("the file-backed fixture device reads",
                     ft_tt_udrv_pci_read_device(path, &device) == FT_TT_UDRV_OK, "");

    ft_tt_test_check("map_bar maps page 1 of it read-only",
                     ft_tt_udrv_pci_map_bar(&device, 0u, false, false, (uint64_t)page, page,
                                            &mapping) == FT_TT_UDRV_OK &&
                         mapping != NULL,
                     "");

    if (!mapping)
        return;

    bytes = mapping;

    /* The fixture writes byte i of the file as (i * 7 + 11) & 0xFF, so an
     * offset the implementation got wrong lands on a different value rather
     * than on a plausible one. */
    ft_tt_test_check_fmt("the first mapped byte is page 1's, not page 0's",
                         bytes[0] == (unsigned char)((page * 7u + 11u + (page >> 8)) & 0xFFu), "0x%02x",
                         bytes[0]);
    ft_tt_test_check_fmt("the last mapped byte is page 1's last",
                         bytes[page - 1u] ==
                             (unsigned char)(((2u * page - 1u) * 7u + 11u + ((2u * page - 1u) >> 8)) & 0xFFu),
                         "0x%02x", bytes[page - 1u]);

    child = fork();
    if (child == 0) {
        bytes[0] = 0xFFu;
        _exit(0);
    }

    (void)waitpid(child, &status, 0);
    ft_tt_test_check_fmt("a store into a read-only mapping kills the process",
                         WIFSIGNALED(status) && (WTERMSIG(status) == SIGSEGV ||
                                                 WTERMSIG(status) == SIGBUS),
                         "signalled=%d signal=%d", WIFSIGNALED(status),
                         WIFSIGNALED(status) ? WTERMSIG(status) : 0);

    ft_tt_test_check("unmapping succeeds", ft_tt_udrv_pci_unmap_bar(mapping, page) ==
                                               FT_TT_UDRV_OK,
                     "");
}

int main(void)
{
    build_fixture();

    stage_the_real_card();
    stage_length_arithmetic();
    stage_resource_paths();
    stage_malformed_inputs();
    stage_enumeration();
    stage_map_refusals();
    stage_map_a_real_file();

    remove_fixture();
    return ft_tt_test_finish();
}
