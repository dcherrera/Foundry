/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_kmd.c
 * @brief Off-device unit test of the kernel-driver boundary (ft_tt_kmd.c): every entry point
 *   is driven against fake open/ioctl/mmap/munmap/close, and every request is compared byte
 *   for byte against an image built from the ABI's documented OFFSETS — never from the
 *   library's own structs, so a wrong field offset in ft_tt_kmd_abi.h is a failure here.
 *
 *   Built twice by Makefile.kmd: once fenced (default build: aiming must be -EPERM and issue
 *   nothing) and once with FT_TT_PHASE1B_NOC_ENABLED (aiming issues CONFIGURE_WINDOW).
 *
 * @status Covers every entry point in ft_tt_kmd.h, success and failure paths.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <sys/mman.h>
#include <sys/types.h>

#include "ft_tt_kmd.h"
#include "ft_tt_test_report.h"
#include "ft_tt_kmd_fake_syscalls.h"

/* ------------------------------------------------------------------------- */
/* The ABI as the spec writes it: literal numbers, independent of the library */
/* ------------------------------------------------------------------------- */

#define ABI_REQUEST(n) (0xFA00ul | (unsigned long)(n))
#define REQ_DEVICE_INFO   ABI_REQUEST(0)
#define REQ_QUERY_MAPPINGS ABI_REQUEST(2)
#define REQ_DRIVER_INFO   ABI_REQUEST(5)
#define REQ_ALLOCATE      ABI_REQUEST(11)
#define REQ_FREE          ABI_REQUEST(12)
#define REQ_CONFIGURE     ABI_REQUEST(13)
#define REQ_NOC_CLEANUP   ABI_REQUEST(14)
#define REQ_POWER_STATE   ABI_REQUEST(15)

#define FAKE_FD 7
#define MAX_CALLS 16
#define MAX_ARG_BYTES 256
#define MIB (1ull << 20)
#define GIB (1ull << 30)

/* Argument size of each request, from the spec. */
static size_t request_length(unsigned long request)
{
    switch (request) {
    case REQ_DEVICE_INFO:    return 24;
    case REQ_QUERY_MAPPINGS: return 8 + 6 * 24;
    case REQ_DRIVER_INFO:    return 16;
    case REQ_ALLOCATE:       return 48;
    case REQ_FREE:           return 8;
    case REQ_CONFIGURE:      return 48;
    case REQ_NOC_CLEANUP:    return 32;
    case REQ_POWER_STATE:    return 40;
    default:                 return 0;
    }
}

/* ------------------------------------------------------------------------- */
/* Fake syscalls                                                             */
/* ------------------------------------------------------------------------- */

typedef enum { CALL_OPEN, CALL_CLOSE, CALL_IOCTL, CALL_MMAP, CALL_MUNMAP } fake_call_kind;

typedef struct {
    fake_call_kind kind;
    int fd;
    char path[64];
    int flags;
    unsigned long request;
    unsigned char bytes[MAX_ARG_BYTES];
    void *address;
    size_t length;
    int protection;
    int map_flags;
    off_t offset;
} fake_call;

/** What call N does: fail with fail_errno, or copy `reply` over the ioctl argument. */
typedef struct {
    int fail_errno;
    unsigned char reply[MAX_ARG_BYTES];
    size_t reply_length;
} fake_step;

static fake_call calls[MAX_CALLS];
static fake_step script[MAX_CALLS];
static int call_count;
static unsigned char fake_aperture[64];

static void fake_reset(void)
{
    memset(calls, 0, sizeof(calls));
    memset(script, 0, sizeof(script));
    call_count = 0;
}

/* Record the next call; NULL once the log is full (the test then fails on the count). */
static fake_call *fake_record(fake_call_kind kind)
{
    fake_call *call;

    if (call_count >= MAX_CALLS)
        return NULL;
    call = &calls[call_count];
    call->kind = kind;
    return call;
}

/* Apply the scripted outcome of the call just recorded. Returns false when it fails. */
static bool fake_outcome(void)
{
    int fail = script[call_count].fail_errno;

    call_count++;
    if (fail) {
        errno = fail;
        return false;
    }
    return true;
}

int test_kmd_fake_open(const char *path, int flags)
{
    fake_call *call = fake_record(CALL_OPEN);

    if (!call)
        return -1;
    snprintf(call->path, sizeof(call->path), "%s", path);
    call->flags = flags;
    return fake_outcome() ? FAKE_FD : -1;
}

int test_kmd_fake_close(int fd)
{
    fake_call *call = fake_record(CALL_CLOSE);

    if (!call)
        return -1;
    call->fd = fd;
    return fake_outcome() ? 0 : -1;
}

int test_kmd_fake_ioctl(int fd, unsigned long request, void *argument)
{
    fake_call *call = fake_record(CALL_IOCTL);
    const fake_step *step = &script[call_count];
    size_t length = request_length(request);

    if (!call)
        return -1;
    call->fd = fd;
    call->request = request;
    if (argument && length)
        memcpy(call->bytes, argument, length);
    if (!fake_outcome())
        return -1;
    if (argument && step->reply_length)
        memcpy(argument, step->reply, step->reply_length);
    return 0;
}

void *test_kmd_fake_mmap(void *address, size_t length, int protection, int flags, int fd,
                         off_t offset)
{
    fake_call *call = fake_record(CALL_MMAP);

    if (!call)
        return MAP_FAILED;
    call->address = address;
    call->length = length;
    call->protection = protection;
    call->map_flags = flags;
    call->fd = fd;
    call->offset = offset;
    return fake_outcome() ? (void *)fake_aperture : MAP_FAILED;
}

int test_kmd_fake_munmap(void *address, size_t length)
{
    fake_call *call = fake_record(CALL_MUNMAP);

    if (!call)
        return -1;
    call->address = address;
    call->length = length;
    return fake_outcome() ? 0 : -1;
}

/* ------------------------------------------------------------------------- */
/* Byte images                                                               */
/* ------------------------------------------------------------------------- */

static void put_le(unsigned char *image, size_t offset, unsigned width, uint64_t value)
{
    unsigned index;

    for (index = 0; index < width; index++)
        image[offset + index] = (unsigned char)(value >> (8u * index));
}

/** Check call `index` is an ioctl of `request` on FAKE_FD whose argument equals `expected`. */
static void check_ioctl(int index, unsigned long request, const unsigned char *expected,
                        const char *what)
{
    const fake_call *call = &calls[index];
    size_t length = request_length(request);
    size_t offset;

    if (index >= call_count || call->kind != CALL_IOCTL) {
        ft_tt_test_check(what, false, "no ioctl at this position in the call log");
        return;
    }
    ft_tt_test_check_fmt(what, call->request == request && call->fd == FAKE_FD,
                         "request 0x%lx (want 0x%lx) fd %d", call->request, request, call->fd);
    for (offset = 0; offset < length; offset++) {
        if (call->bytes[offset] != expected[offset])
            break;
    }
    ft_tt_test_check_fmt("  argument bytes match the ABI offsets", offset == length,
                         offset == length ? "%zu bytes" : "first mismatch at byte %zu",
                         offset == length ? length : offset);
}

static void check_call_count(int want, const char *what)
{
    ft_tt_test_check_fmt(what, call_count == want, "%d syscalls (want %d)", call_count, want);
}

/* ------------------------------------------------------------------------- */
/* Fixtures                                                                  */
/* ------------------------------------------------------------------------- */

/* A QUERY_MAPPINGS reply: six records at 8 + 24k, with an unknown id in slot 4. */
static void script_mappings(fake_step *step, uint64_t bar0_len, uint64_t bar4_len)
{
    static const uint32_t ids[6] = { 1, 2, 3, 4, 9, 6 };
    size_t record;

    memset(step, 0, sizeof(*step));
    step->reply_length = 8 + 6 * 24;
    put_le(step->reply, 0, 4, 6);
    for (record = 0; record < 6; record++) {
        size_t base = 8 + record * 24;

        put_le(step->reply, base + 0, 4, ids[record]);
        put_le(step->reply, base + 8, 8, (uint64_t)(record + 1) << 36);
        put_le(step->reply, base + 16, 8, 0x5555u); /* overwritten for UC ids below */
    }
    put_le(step->reply, 8 + 0 * 24 + 16, 8, bar0_len);
    put_le(step->reply, 8 + 2 * 24 + 16, 8, 1 * MIB);
    if (bar4_len) {
        put_le(step->reply, 8 + 4 * 24 + 0, 4, 5); /* make slot 4 resource2 UC */
        put_le(step->reply, 8 + 4 * 24 + 16, 8, bar4_len);
    }
}

/* Open a device with the given BAR lengths; the call log is reset afterwards. */
static ft_tt_kmd_device *open_fixture(uint64_t bar0_len, uint64_t bar4_len)
{
    ft_tt_kmd_device *device = NULL;

    fake_reset();
    script_mappings(&script[1], bar0_len, bar4_len);
    if (ft_tt_kmd_open("/dev/tenstorrent/0", &device, 0) != 0)
        ft_tt_test_check("fixture open", false, "open failed");
    fake_reset();
    return device;
}

static void close_fixture(ft_tt_kmd_device *device)
{
    fake_reset();
    (void)ft_tt_kmd_close(device);
    fake_reset();
}

/* ------------------------------------------------------------------------- */
/* Open / close                                                              */
/* ------------------------------------------------------------------------- */

static void test_open_success(void)
{
    ft_tt_kmd_device *device = NULL;
    ft_tt_kmd_resources resources;
    unsigned char expected[MAX_ARG_BYTES] = { 0 };
    int rc;

    ft_tt_test_step("open: one open, one QUERY_MAPPINGS asking for six records");
    fake_reset();
    script_mappings(&script[1], 512 * MIB, 0);
    rc = ft_tt_kmd_open("/dev/tenstorrent/3", &device, O_APPEND);
    ft_tt_test_check_fmt("open succeeds", rc == 0 && device != NULL, "rc %d", rc);
    check_call_count(2, "exactly two syscalls");
    ft_tt_test_check_fmt("open path and flags", calls[0].kind == CALL_OPEN &&
                         strcmp(calls[0].path, "/dev/tenstorrent/3") == 0 &&
                         calls[0].flags == (O_RDWR | O_CLOEXEC | O_APPEND),
                         "%s flags 0x%x", calls[0].path, (unsigned)calls[0].flags);
    put_le(expected, 0, 4, 6);
    check_ioctl(1, REQ_QUERY_MAPPINGS, expected, "QUERY_MAPPINGS request");
    ft_tt_test_check_fmt("fd is the opened one", ft_tt_kmd_fd(device) == FAKE_FD, "%d",
                         ft_tt_kmd_fd(device));

    memset(&resources, 0xAB, sizeof(resources));
    rc = ft_tt_kmd_get_resources(device, &resources);
    ft_tt_test_check("resources are readable", rc == 0, "");
    ft_tt_test_check_fmt("BAR0 from ids 1/2", resources.bar0_uc_offset == (1ull << 36) &&
                         resources.bar0_wc_offset == (2ull << 36) &&
                         resources.bar0_len == 512 * MIB,
                         "uc 0x%llx wc 0x%llx len 0x%llx",
                         (unsigned long long)resources.bar0_uc_offset,
                         (unsigned long long)resources.bar0_wc_offset,
                         (unsigned long long)resources.bar0_len);
    ft_tt_test_check_fmt("BAR2 from ids 3/4 (length from the UC record only)",
                         resources.bar2_uc_offset == (3ull << 36) &&
                         resources.bar2_wc_offset == (4ull << 36) && resources.bar2_len == MIB,
                         "len 0x%llx", (unsigned long long)resources.bar2_len);
    ft_tt_test_check_fmt("unknown id ignored; BAR4 has only its WC offset",
                         resources.bar4_uc_offset == 0 && resources.bar4_len == 0 &&
                         resources.bar4_wc_offset == (6ull << 36),
                         "uc 0x%llx len 0x%llx", (unsigned long long)resources.bar4_uc_offset,
                         (unsigned long long)resources.bar4_len);
    close_fixture(device);
}

static void test_open_failures(void)
{
    ft_tt_kmd_device *device = NULL;
    int rc;

    ft_tt_test_step("open: argument checks and failure paths");
    fake_reset();
    rc = ft_tt_kmd_open(NULL, &device, 0);
    ft_tt_test_check_fmt("NULL path is -EINVAL", rc == -EINVAL, "rc %d", rc);
    rc = ft_tt_kmd_open("/dev/tenstorrent/0", NULL, 0);
    ft_tt_test_check_fmt("NULL out is -EINVAL", rc == -EINVAL, "rc %d", rc);
    check_call_count(0, "no syscalls for bad arguments");

    fake_reset();
    script[0].fail_errno = ENOENT;
    rc = ft_tt_kmd_open("/dev/tenstorrent/0", &device, 0);
    ft_tt_test_check_fmt("failed open is -errno", rc == -ENOENT && device == NULL, "rc %d", rc);
    check_call_count(1, "nothing after a failed open");

    fake_reset();
    script[1].fail_errno = ENOTTY;
    rc = ft_tt_kmd_open("/dev/tenstorrent/0", &device, 0);
    ft_tt_test_check_fmt("failed QUERY_MAPPINGS is its -errno", rc == -ENOTTY && device == NULL,
                         "rc %d", rc);
    check_call_count(3, "open, ioctl, close");
    ft_tt_test_check("the fd is closed", calls[2].kind == CALL_CLOSE && calls[2].fd == FAKE_FD,
                     "");
}

static void test_close(void)
{
    ft_tt_kmd_device *device;
    int rc;

    ft_tt_test_step("close");
    fake_reset();
    rc = ft_tt_kmd_close(NULL);
    ft_tt_test_check_fmt("NULL is 0 and touches nothing", rc == 0 && call_count == 0, "rc %d",
                         rc);

    device = open_fixture(512 * MIB, 0);
    rc = ft_tt_kmd_close(device);
    ft_tt_test_check_fmt("close closes the fd", rc == 0 && call_count == 1 &&
                         calls[0].kind == CALL_CLOSE && calls[0].fd == FAKE_FD, "rc %d", rc);

    device = open_fixture(512 * MIB, 0);
    script[0].fail_errno = EBADF;
    rc = ft_tt_kmd_close(device);
    ft_tt_test_check_fmt("a failing close is -errno", rc == -EBADF, "rc %d", rc);
    ft_tt_test_check_fmt("ft_tt_kmd_fd(NULL) is -1", ft_tt_kmd_fd(NULL) == -1, "%d",
                         ft_tt_kmd_fd(NULL));
}

/* ------------------------------------------------------------------------- */
/* Attributes                                                                */
/* ------------------------------------------------------------------------- */

/* One GET_DEVICE_INFO against a device with the given BARs, replying `device_id`. */
static int attrs_for(uint64_t bar0_len, uint64_t bar4_len, uint16_t device_id,
                     ft_tt_kmd_device_attrs *out)
{
    ft_tt_kmd_device *device = open_fixture(bar0_len, bar4_len);
    int rc;

    script[0].reply_length = 24;
    put_le(script[0].reply, 4, 4, 20);
    put_le(script[0].reply, 8, 2, 0x1E52);
    put_le(script[0].reply, 10, 2, device_id);
    put_le(script[0].reply, 12, 2, 0x1E53);
    put_le(script[0].reply, 14, 2, 0x0040);
    put_le(script[0].reply, 16, 2, (0x12u << 8) | (3u << 3) | 5u);
    put_le(script[0].reply, 18, 2, 32);
    put_le(script[0].reply, 20, 2, 0x0007);
    memset(out, 0xEE, sizeof(*out));
    rc = ft_tt_kmd_get_attrs(device, out);
    {
        unsigned char expected[MAX_ARG_BYTES] = { 0 };

        put_le(expected, 0, 4, 20);
        check_ioctl(0, REQ_DEVICE_INFO, expected, "GET_DEVICE_INFO request");
        check_call_count(1, "one syscall");
    }
    close_fixture(device);
    return rc;
}

static void test_attrs(void)
{
    ft_tt_kmd_device_attrs attrs;
    int rc;

    ft_tt_test_step("attrs: one GET_DEVICE_INFO, decoded");
    rc = attrs_for(512 * MIB, 0, 0xB140, &attrs);
    ft_tt_test_check_fmt("ids and arch", rc == 0 && attrs.pci_vendor_id == 0x1E52 &&
                         attrs.pci_device_id == 0xB140 &&
                         attrs.pci_subsystem_vendor_id == 0x1E53 &&
                         attrs.pci_subsystem_id == 0x40 &&
                         attrs.chip_arch == FT_TT_KMD_ARCH_BLACKHOLE,
                         "rc %d arch %llu", rc, (unsigned long long)attrs.chip_arch);
    ft_tt_test_check_fmt("bdf and domain decoded", attrs.pci_bus == 0x12 &&
                         attrs.pci_device == 3 && attrs.pci_function == 5 &&
                         attrs.pci_domain == 7, "%llx:%llx.%llx domain %llx",
                         (unsigned long long)attrs.pci_bus, (unsigned long long)attrs.pci_device,
                         (unsigned long long)attrs.pci_function,
                         (unsigned long long)attrs.pci_domain);
    ft_tt_test_check_fmt("Blackhole, 512 MiB BAR0, no BAR4: 202 x 2M (capped), 0 x 4G",
                         attrs.window_count_2m == 202 && attrs.window_count_4g == 0 &&
                         attrs.window_count_1m == 0 && attrs.window_count_16m == 0,
                         "2m %llu 4g %llu", (unsigned long long)attrs.window_count_2m,
                         (unsigned long long)attrs.window_count_4g);

    rc = attrs_for(100 * MIB, 8 * GIB, 0xB140, &attrs);
    ft_tt_test_check_fmt("100 MiB BAR0 and 8 GiB BAR4: 50 x 2M, 2 x 4G",
                         rc == 0 && attrs.window_count_2m == 50 && attrs.window_count_4g == 2,
                         "2m %llu 4g %llu", (unsigned long long)attrs.window_count_2m,
                         (unsigned long long)attrs.window_count_4g);
    rc = attrs_for(512 * MIB, 8 * GIB, 0x401E, &attrs);
    ft_tt_test_check_fmt("Wormhole: arch 1, no window counts", rc == 0 &&
                         attrs.chip_arch == FT_TT_KMD_ARCH_WORMHOLE &&
                         attrs.window_count_2m == 0 && attrs.window_count_4g == 0,
                         "arch %llu", (unsigned long long)attrs.chip_arch);
    rc = attrs_for(512 * MIB, 0, 0xFACA, &attrs);
    ft_tt_test_check_fmt("anything else: arch 0", rc == 0 &&
                         attrs.chip_arch == FT_TT_KMD_ARCH_UNKNOWN && attrs.window_count_2m == 0,
                         "arch %llu", (unsigned long long)attrs.chip_arch);

    {
        ft_tt_kmd_device *device = open_fixture(512 * MIB, 0);

        script[0].fail_errno = EFAULT;
        rc = ft_tt_kmd_get_attrs(device, &attrs);
        ft_tt_test_check_fmt("a failing ioctl is -errno", rc == -EFAULT, "rc %d", rc);
        ft_tt_test_check_fmt("NULL out is -EINVAL",
                             ft_tt_kmd_get_attrs(device, NULL) == -EINVAL &&
                             ft_tt_kmd_get_resources(device, NULL) == -EINVAL &&
                             ft_tt_kmd_get_attrs(NULL, &attrs) == -EINVAL, "%s", "");
        close_fixture(device);
    }
}

/* ------------------------------------------------------------------------- */
/* Power state and close-time NoC write                                      */
/* ------------------------------------------------------------------------- */

static void test_power_state(void)
{
    ft_tt_kmd_device *device = open_fixture(512 * MIB, 0);
    unsigned char expected[MAX_ARG_BYTES] = { 0 };
    int rc;

    ft_tt_test_step("power state");
    put_le(expected, 0, 4, 40);
    put_le(expected, 9, 1, 0x0F);
    put_le(expected, 10, 2, 0x7FFF);
    rc = ft_tt_kmd_request_max_ai_clock(device, true);
    ft_tt_test_check_fmt("max succeeds", rc == 0, "rc %d", rc);
    check_ioctl(0, REQ_POWER_STATE, expected, "SET_POWER_STATE, max: word 0x7FFF");

    fake_reset();
    put_le(expected, 10, 2, 0x7FFE);
    rc = ft_tt_kmd_request_max_ai_clock(device, false);
    check_ioctl(0, REQ_POWER_STATE, expected, "SET_POWER_STATE, min: word 0x7FFE");

    fake_reset();
    script[0].fail_errno = EINVAL;
    rc = ft_tt_kmd_request_max_ai_clock(device, true);
    ft_tt_test_check_fmt("a refusal is -errno", rc == -EINVAL && call_count == 1, "rc %d", rc);
    ft_tt_test_check("NULL device is -EINVAL",
                     ft_tt_kmd_request_max_ai_clock(NULL, true) == -EINVAL, "");
    close_fixture(device);
}

static void test_noc_cleanup(void)
{
    ft_tt_kmd_device *device = open_fixture(512 * MIB, 0);
    unsigned char expected[MAX_ARG_BYTES] = { 0 };
    int rc;

    ft_tt_test_step("close-time NoC write");
    put_le(expected, 0, 4, 32);
    put_le(expected, 8, 1, 1);
    put_le(expected, 9, 1, 7);
    put_le(expected, 10, 1, 9);
    put_le(expected, 11, 1, 1);
    put_le(expected, 16, 8, 0x123456789Aull);
    put_le(expected, 24, 8, 0xDEADBEEFCAFEull);
    rc = ft_tt_kmd_set_noc_cleanup(device, 1, 7, 9, 0x123456789Aull, 0xDEADBEEFCAFEull);
    ft_tt_test_check_fmt("set succeeds", rc == 0, "rc %d", rc);
    check_ioctl(0, REQ_NOC_CLEANUP, expected, "NOC_CLEANUP, set");

    fake_reset();
    memset(expected, 0, sizeof(expected));
    put_le(expected, 0, 4, 32);
    rc = ft_tt_kmd_clear_noc_cleanup(device);
    ft_tt_test_check_fmt("clear succeeds", rc == 0, "rc %d", rc);
    check_ioctl(0, REQ_NOC_CLEANUP, expected, "NOC_CLEANUP, clear: enabled 0, all else 0");

    fake_reset();
    script[0].fail_errno = ENOTTY;
    rc = ft_tt_kmd_set_noc_cleanup(device, 0, 1, 2, 3, 4);
    ft_tt_test_check_fmt("a refusal is -errno", rc == -ENOTTY, "rc %d", rc);
    ft_tt_test_check("NULL device is -EINVAL",
                     ft_tt_kmd_set_noc_cleanup(NULL, 0, 0, 0, 0, 0) == -EINVAL &&
                     ft_tt_kmd_clear_noc_cleanup(NULL) == -EINVAL, "");
    close_fixture(device);
}

/* ------------------------------------------------------------------------- */
/* Driver attributes                                                         */
/* ------------------------------------------------------------------------- */

static void test_driver_attr(void)
{
    ft_tt_kmd_device *device;
    fake_step reply;
    unsigned char expected[MAX_ARG_BYTES] = { 0 };
    uint64_t value = 99;
    int rc;

    ft_tt_test_step("driver attributes");
    fake_reset();
    ft_tt_test_check("NULL out is -EINVAL",
                     ft_tt_kmd_driver_attr(NULL, FT_TT_KMD_DRIVER_API_VERSION, NULL) == -EINVAL,
                     "");
    rc = ft_tt_kmd_driver_attr(NULL, FT_TT_KMD_DRIVER_API_VERSION, &value);
    ft_tt_test_check_fmt("API version without a device is 2", rc == 0 && value == 2,
                         "rc %d value %llu", rc, (unsigned long long)value);
    value = 99;
    rc = ft_tt_kmd_driver_attr(NULL, FT_TT_KMD_DRIVER_VERSION_MINOR, &value);
    ft_tt_test_check_fmt("semver without a device is -ENODEV, value 0",
                         rc == -ENODEV && value == 0, "rc %d value %llu", rc,
                         (unsigned long long)value);
    check_call_count(0, "no syscalls without a device");

    memset(&reply, 0, sizeof(reply));
    reply.reply_length = 16;
    put_le(reply.reply, 4, 4, 12);
    put_le(reply.reply, 8, 4, 99);
    put_le(reply.reply, 12, 1, 1);
    put_le(reply.reply, 13, 1, 5);
    put_le(reply.reply, 14, 1, 7);
    device = open_fixture(512 * MIB, 0);
    script[0] = reply;
    script[1] = reply;
    rc = ft_tt_kmd_driver_attr(device, FT_TT_KMD_DRIVER_VERSION_MINOR, &value);
    put_le(expected, 0, 4, 12);
    check_ioctl(0, REQ_DRIVER_INFO, expected, "GET_DRIVER_INFO request");
    ft_tt_test_check_fmt("minor is the reply's byte 13", rc == 0 && value == 5, "value %llu",
                         (unsigned long long)value);
    rc = ft_tt_kmd_driver_attr(device, FT_TT_KMD_DRIVER_API_VERSION, &value);
    ft_tt_test_check_fmt("API version is the compiled-in 2, not the reply's 99",
                         rc == 0 && value == 2 && call_count == 2, "value %llu",
                         (unsigned long long)value);

    fake_reset();
    script[0] = reply;
    script[1] = reply;
    rc = ft_tt_kmd_driver_attr(device, FT_TT_KMD_DRIVER_VERSION_MAJOR, &value);
    ft_tt_test_check_fmt("major is byte 12", rc == 0 && value == 1, "%llu",
                         (unsigned long long)value);
    rc = ft_tt_kmd_driver_attr(device, FT_TT_KMD_DRIVER_VERSION_PATCH, &value);
    ft_tt_test_check_fmt("patch is byte 14", rc == 0 && value == 7, "%llu",
                         (unsigned long long)value);
    rc = ft_tt_kmd_driver_attr(device, (enum ft_tt_kmd_driver_attr)4, &value);
    ft_tt_test_check_fmt("an unknown attribute is -EINVAL", rc == -EINVAL, "rc %d", rc);

    fake_reset();
    script[0].fail_errno = EIO;
    rc = ft_tt_kmd_driver_attr(device, FT_TT_KMD_DRIVER_API_VERSION, &value);
    ft_tt_test_check_fmt("a failing ioctl is -errno", rc == -EIO, "rc %d", rc);
    close_fixture(device);
}

/* ------------------------------------------------------------------------- */
/* Windows                                                                   */
/* ------------------------------------------------------------------------- */

/* Script an ALLOCATE_WINDOW reply at step `index`: id 42, distinct UC and WC offsets. */
static void script_allocate(int index)
{
    script[index].reply_length = 48;
    put_le(script[index].reply, 16, 4, 42);
    put_le(script[index].reply, 24, 8, 0x1000000000ull);
    put_le(script[index].reply, 32, 8, 0x2000000000ull);
}

static void test_window_alloc(void)
{
    ft_tt_kmd_device *device = open_fixture(512 * MIB, 0);
    ft_tt_kmd_window *window = NULL;
    unsigned char expected[MAX_ARG_BYTES] = { 0 };
    void *aperture = NULL;
    uint32_t id = 0;
    size_t size = 0;
    int rc;

    ft_tt_test_step("window alloc: ALLOCATE_WINDOW then mmap");
    script_allocate(0);
    rc = ft_tt_kmd_window_alloc(device, 2 * MIB, FT_TT_KMD_CACHE_WC, PROT_READ, &window);
    ft_tt_test_check_fmt("alloc succeeds", rc == 0 && window != NULL, "rc %d", rc);
    put_le(expected, 0, 8, 2 * MIB);
    check_ioctl(0, REQ_ALLOCATE, expected, "ALLOCATE_WINDOW request");
    ft_tt_test_check_fmt("mmap of the WC offset", call_count == 2 && calls[1].kind == CALL_MMAP &&
                         calls[1].address == NULL && calls[1].length == 2 * MIB &&
                         calls[1].protection == PROT_READ && calls[1].map_flags == MAP_SHARED &&
                         calls[1].fd == FAKE_FD && calls[1].offset == (off_t)0x2000000000ll,
                         "offset 0x%llx", (unsigned long long)calls[1].offset);
    ft_tt_test_check_fmt("accessors", ft_tt_kmd_window_id(window, &id) == 0 && id == 42 &&
                         ft_tt_kmd_window_size(window, &size) == 0 && size == 2 * MIB &&
                         ft_tt_kmd_window_aperture(window, &aperture) == 0 &&
                         aperture == (void *)fake_aperture, "id %u size %zu", id, size);

    fake_reset();
    rc = ft_tt_kmd_window_free(device, window);
    memset(expected, 0, sizeof(expected));
    put_le(expected, 0, 4, 42);
    ft_tt_test_check_fmt("free: munmap FIRST", rc == 0 && call_count == 2 &&
                         calls[0].kind == CALL_MUNMAP && calls[0].address == fake_aperture &&
                         calls[0].length == 2 * MIB, "rc %d", rc);
    check_ioctl(1, REQ_FREE, expected, "then FREE_WINDOW with the id");

    fake_reset();
    script_allocate(0);
    rc = ft_tt_kmd_window_alloc(device, 2 * MIB, FT_TT_KMD_CACHE_UC, PROT_NONE, &window);
    ft_tt_test_check_fmt("UC maps the UC offset", rc == 0 &&
                         calls[1].offset == (off_t)0x1000000000ll &&
                         calls[1].protection == PROT_NONE, "offset 0x%llx",
                         (unsigned long long)calls[1].offset);
    fake_reset();
    script[0].fail_errno = EFAULT;
    script[1].fail_errno = EINVAL;
    rc = ft_tt_kmd_window_free(device, window);
    ft_tt_test_check_fmt("free continues past a failed munmap; last failure wins",
                         rc == -EINVAL && call_count == 2, "rc %d", rc);
    close_fixture(device);
}

static void test_window_alloc_failures(void)
{
    ft_tt_kmd_device *device = open_fixture(512 * MIB, 0);
    ft_tt_kmd_window *window = NULL;
    unsigned char expected[MAX_ARG_BYTES] = { 0 };
    int rc;

    ft_tt_test_step("window alloc: failure paths");
    ft_tt_test_check("NULL device or out is -EINVAL",
                     ft_tt_kmd_window_alloc(NULL, 2 * MIB, FT_TT_KMD_CACHE_UC, 0, &window) ==
                     -EINVAL && ft_tt_kmd_window_alloc(device, 2 * MIB, FT_TT_KMD_CACHE_UC, 0,
                                                       NULL) == -EINVAL && call_count == 0, "");
    script[0].fail_errno = ENOMEM;
    rc = ft_tt_kmd_window_alloc(device, 4 * GIB, FT_TT_KMD_CACHE_WC, 0, &window);
    ft_tt_test_check_fmt("a refused ALLOCATE is -errno and maps nothing",
                         rc == -ENOMEM && call_count == 1, "rc %d", rc);

    fake_reset();
    script_allocate(0);
    script[1].fail_errno = ENODEV;
    rc = ft_tt_kmd_window_alloc(device, 2 * MIB, FT_TT_KMD_CACHE_WC, 0, &window);
    put_le(expected, 0, 4, 42);
    ft_tt_test_check_fmt("a failed mmap is its -errno", rc == -ENODEV && call_count == 3,
                         "rc %d calls %d", rc, call_count);
    check_ioctl(2, REQ_FREE, expected, "and the claim is released");

    fake_reset();
    script_allocate(0);
    script[1].fail_errno = ENODEV;
    script[2].fail_errno = EBADF;
    rc = ft_tt_kmd_window_alloc(device, 2 * MIB, FT_TT_KMD_CACHE_WC, 0, &window);
    ft_tt_test_check_fmt("a failed release after a failed mmap reports the release",
                         rc == -EBADF, "rc %d", rc);
    ft_tt_test_check("window accessors and free refuse NULL",
                     ft_tt_kmd_window_id(NULL, NULL) == -EINVAL &&
                     ft_tt_kmd_window_size(NULL, NULL) == -EINVAL &&
                     ft_tt_kmd_window_aperture(NULL, NULL) == -EINVAL &&
                     ft_tt_kmd_window_free(device, NULL) == -EINVAL, "");
    close_fixture(device);
}

#ifdef FT_TT_PHASE1B_NOC_ENABLED

static void test_window_aim(void)
{
    ft_tt_kmd_device *device = open_fixture(512 * MIB, 0);
    ft_tt_kmd_window *window = NULL;
    ft_tt_kmd_window_target target;
    unsigned char expected[MAX_ARG_BYTES] = { 0 };
    int rc;

    ft_tt_test_step("window aim (FT_TT_PHASE1B_NOC_ENABLED): CONFIGURE_WINDOW");
    script_allocate(0);
    (void)ft_tt_kmd_window_alloc(device, 2 * MIB, FT_TT_KMD_CACHE_UC, 0, &window);
    fake_reset();

    memset(&target, 0, sizeof(target));
    target.addr = 6 * MIB;
    target.x_end = 0x0102;
    target.y_end = 0x0304;
    target.x_start = 0x0506;
    target.y_start = 0x0708;
    target.noc = 1;
    target.mcast = 1;
    target.ordering = FT_TT_KMD_ORDERING_POSTED_STRICT;
    target.static_vc = 1;
    rc = ft_tt_kmd_window_aim(device, window, &target);
    put_le(expected, 0, 4, 42);
    put_le(expected, 8, 8, 6 * MIB);
    put_le(expected, 16, 2, 0x0102);
    put_le(expected, 18, 2, 0x0304);
    put_le(expected, 20, 2, 0x0506);
    put_le(expected, 22, 2, 0x0708);
    put_le(expected, 24, 1, 1);
    put_le(expected, 25, 1, 1);
    put_le(expected, 26, 1, 3);
    put_le(expected, 28, 1, 1);
    ft_tt_test_check_fmt("aim succeeds", rc == 0, "rc %d", rc);
    check_ioctl(0, REQ_CONFIGURE, expected, "CONFIGURE_WINDOW, every field");

    fake_reset();
    rc = ft_tt_kmd_window_aim_unicast(device, window, 9, 11, 2 * MIB);
    memset(expected, 0, sizeof(expected));
    put_le(expected, 0, 4, 42);
    put_le(expected, 8, 8, 2 * MIB);
    put_le(expected, 16, 2, 9);
    put_le(expected, 18, 2, 11);
    check_ioctl(0, REQ_CONFIGURE, expected, "CONFIGURE_WINDOW, unicast");

    fake_reset();
    rc = ft_tt_kmd_window_aim_unicast(device, window, 9, 11, 2 * MIB + 4);
    ft_tt_test_check_fmt("a misaligned address is -EINVAL and issues nothing",
                         rc == -EINVAL && call_count == 0, "rc %d", rc);
    ft_tt_test_check("NULL arguments are -EINVAL",
                     ft_tt_kmd_window_aim(NULL, window, &target) == -EINVAL &&
                     ft_tt_kmd_window_aim(device, NULL, &target) == -EINVAL &&
                     ft_tt_kmd_window_aim(device, window, NULL) == -EINVAL, "");
    script[0].fail_errno = EPERM;
    rc = ft_tt_kmd_window_aim(device, window, &target);
    ft_tt_test_check_fmt("a refused CONFIGURE is -errno", rc == -EPERM && call_count == 1,
                         "rc %d", rc);
    fake_reset();
    (void)ft_tt_kmd_window_free(device, window);
    close_fixture(device);
}

#else /* !FT_TT_PHASE1B_NOC_ENABLED */

static void test_window_aim(void)
{
    ft_tt_kmd_device *device = open_fixture(512 * MIB, 0);
    ft_tt_kmd_window *window = NULL;
    ft_tt_kmd_window_target target;
    int rc;

    ft_tt_test_step("window aim (default build): fenced");
    script_allocate(0);
    (void)ft_tt_kmd_window_alloc(device, 2 * MIB, FT_TT_KMD_CACHE_UC, 0, &window);
    fake_reset();
    memset(&target, 0, sizeof(target));
    rc = ft_tt_kmd_window_aim(device, window, &target);
    ft_tt_test_check_fmt("aim is -EPERM", rc == -EPERM, "rc %d", rc);
    rc = ft_tt_kmd_window_aim_unicast(device, window, 1, 2, 0);
    ft_tt_test_check_fmt("unicast aim is -EPERM", rc == -EPERM, "rc %d", rc);
    ft_tt_test_check_fmt("even NULL arguments get -EPERM",
                         ft_tt_kmd_window_aim(NULL, NULL, NULL) == -EPERM, "%s", "");
    check_call_count(0, "no syscall at all");
    (void)ft_tt_kmd_window_free(device, window);
    close_fixture(device);
}

#endif /* FT_TT_PHASE1B_NOC_ENABLED */

int main(void)
{
    test_open_success();
    test_open_failures();
    test_close();
    test_attrs();
    test_power_state();
    test_noc_cleanup();
    test_driver_attr();
    test_window_alloc();
    test_window_alloc_failures();
    test_window_aim();
    return ft_tt_test_finish();
}
