/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_device.c
 * @brief Phase 1a acceptance test for the Foundry Blackhole backend: device
 *   open/close, enumeration, BAR mappings, derived TLB counts, TLB window
 *   lifecycle, and the Phase 1a NoC fence.
 *
 * READ-ONLY AGAINST THE DEVICE. The only ioctls this issues are
 * GET_DEVICE_INFO, GET_DRIVER_INFO, QUERY_MAPPINGS, ALLOCATE_TLB and FREE_TLB;
 * the only device memory it touches is the TLB configuration register file in
 * BAR0, which it reads. It never calls CONFIGURE_TLB, never reads or writes
 * through a TLB window, and never resets, pins or powers anything.
 *
 * @status Runs green on the p150a at 12:00.0. Every case has been
 *   observed RED by mutating the code it covers — see docs/tt_bringup_log.md
 *   for the mutation table and the failure counts.
 * @issues None
 * @todo None
 *
 * Expected values and where they come from:
 *   spec §2.1  — p150a, BAR0 512 MiB, BAR2 1 MiB, no BAR4, PCI 12:00.0
 *   kernel driver — the fixed mmap offsets
 *   kernel driver — 202 windows, index 201 is the kernel's,
 *                                      4 GiB count derived from BAR4
 *   (see BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md)
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_device.h"
#include "ft_tt_tlb.h"

/* ------------------------------------------------------------------------- */
/* Expected values                                                           */
/* ------------------------------------------------------------------------- */

#define EXPECTED_PCI_VENDOR_ID 0x1e52u
#define EXPECTED_PCI_DEVICE_ID 0xb140u /* Blackhole */
#define EXPECTED_PCI_BUS       0x12u   /* 12:00.0 */

#define EXPECTED_BAR0_LEN (512u * 1024u * 1024u)
#define EXPECTED_BAR2_LEN (1u * 1024u * 1024u)
#define EXPECTED_BAR4_LEN 0u /* spec §2.3 — flashed out, and that is correct */

#define EXPECTED_BAR0_UC_OFFSET (UINT64_C(0) << 36)
#define EXPECTED_BAR0_WC_OFFSET (UINT64_C(1) << 36)
#define EXPECTED_BAR2_UC_OFFSET (UINT64_C(2) << 36)
#define EXPECTED_BAR2_WC_OFFSET (UINT64_C(3) << 36)

#define EXPECTED_TLB_2M_TOTAL     202u
#define EXPECTED_TLB_2M_AVAILABLE 201u
#define EXPECTED_TLB_4G_TOTAL     0u

#define MAX_WINDOWS_TRACKED 256

/* ------------------------------------------------------------------------- */
/* Tiny harness                                                              */
/* ------------------------------------------------------------------------- */

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got == want) {
        printf("  ok    %s = %" PRIu64 "\n", what, got);
    } else {
        checks_failed++;
        printf("  FAIL  %s = %" PRIu64 " (expected %" PRIu64 ")\n", what, got, want);
    }
}

/* ------------------------------------------------------------------------- */
/* Cases                                                                     */
/* ------------------------------------------------------------------------- */

/* The API version is answerable without a device, which is how a caller checks
 * ABI compatibility before opening anything. */
static void test_driver_api_version_without_device(void)
{
    uint64_t version = 0;
    uint64_t major = 0;

    printf("driver attributes without a device\n");
    check(ft_tt_kmd_driver_attr(NULL, FT_TT_KMD_DRIVER_API_VERSION, &version) == 0,
          "API version is answerable with a NULL device");
    check_u64(version, 2, "ioctl API version");
    check(ft_tt_kmd_driver_attr(NULL, FT_TT_KMD_DRIVER_VERSION_MAJOR, &major) != 0,
          "driver semver is refused without a device");
}

static void test_device_identity(const ft_tt_ctx *ctx)
{
    ft_tt_device_info info;

    printf("device identity\n");
    memset(&info, 0, sizeof(info));
    check(ft_tt_device_get_info(ctx, &info) == FT_TT_OK, "ft_tt_device_get_info succeeds");
    check_u64(info.pci_vendor_id, EXPECTED_PCI_VENDOR_ID, "PCI vendor id");
    check_u64(info.pci_device_id, EXPECTED_PCI_DEVICE_ID, "PCI device id");
    check_u64(info.chip_arch, (uint64_t)FT_TT_KMD_ARCH_BLACKHOLE, "chip arch (2 = Blackhole)");
    check_u64(info.pci_bus, EXPECTED_PCI_BUS, "PCI bus");
    check_u64(info.pci_device, 0, "PCI device");
    check_u64(info.pci_function, 0, "PCI function");
    check_u64(info.driver_api_version, 2, "driver ioctl API version");
    check(info.driver_version_major != 0 || info.driver_version_minor != 0,
          "driver semver is populated");
    printf("        driver %u.%u.%u\n", info.driver_version_major, info.driver_version_minor,
           info.driver_version_patch);
}

/* QUERY_MAPPINGS is the source of truth for BAR lengths, and therefore for the
 * TLB counts. The absence of BAR4 must show up as a length of 0, not as a
 * missing-field error. */
static void test_bar_mappings(const ft_tt_ctx *ctx)
{
    ft_tt_kmd_resources res;

    printf("BAR mappings\n");
    memset(&res, 0, sizeof(res));
    check(ft_tt_device_get_resources(ctx, &res) == FT_TT_OK, "QUERY_MAPPINGS cache is readable");
    check_u64(res.bar0_len, EXPECTED_BAR0_LEN, "BAR0 length");
    check_u64(res.bar2_len, EXPECTED_BAR2_LEN, "BAR2 length");
    check_u64(res.bar4_len, EXPECTED_BAR4_LEN, "BAR4 length");
    check_u64(res.bar0_uc_offset, EXPECTED_BAR0_UC_OFFSET, "BAR0 UC mmap offset");
    check_u64(res.bar0_wc_offset, EXPECTED_BAR0_WC_OFFSET, "BAR0 WC mmap offset");
    check_u64(res.bar2_uc_offset, EXPECTED_BAR2_UC_OFFSET, "BAR2 UC mmap offset");
    check_u64(res.bar2_wc_offset, EXPECTED_BAR2_WC_OFFSET, "BAR2 WC mmap offset");
    check(res.bar4_uc_offset == 0 && res.bar4_wc_offset == 0,
          "BAR4 has no mapping entries at all");
}

/* Window counts derived from BAR lengths (ft_tt_kmd.h): upstream would report eight 4 GiB
 * windows on this board. There are none, because there is no BAR4. */
static void test_derived_tlb_counts(const ft_tt_ctx *ctx)
{
    ft_tt_tlb_counts counts;

    printf("TLB counts derived from BAR lengths\n");
    memset(&counts, 0, sizeof(counts));
    check(ft_tt_tlb_counts_get(ctx, &counts) == FT_TT_OK, "counts are readable");
    check_u64(counts.window_2m_total, EXPECTED_TLB_2M_TOTAL, "2 MiB windows on the board");
    check_u64(counts.window_2m_available, EXPECTED_TLB_2M_AVAILABLE,
              "2 MiB windows left after the driver's own");
    check_u64(counts.window_4g_total, EXPECTED_TLB_4G_TOTAL,
              "4 GiB windows (upstream's table would say 8)");
}

/* The TLB register file is BAR0 configuration space, not a NoC aperture:
 * reading it issues no NoC transaction. Reading it is also the only way to
 * prove the mapping landed where we think it did. */
static void test_tlb_register_file(const ft_tt_ctx *ctx)
{
    const volatile uint32_t *regs = ft_tt_tlb_registers(ctx);
    uint32_t sink = 0;
    size_t word;

    printf("TLB register file\n");
    check(regs != NULL, "BAR0 + 0x1FC00000 is mapped");
    check_u64((uint64_t)ft_tt_tlb_registers_size(ctx), FT_TT_TLB_REGS_LEN, "mapped bytes");

    if (!regs)
        return;

    /* Every window's registers must be inside the mapping. */
    check((size_t)FT_TT_TLB_TOTAL_WINDOW_COUNT * FT_TT_TLB_REG_STRIDE <=
              ft_tt_tlb_registers_size(ctx),
          "all per-window registers fit in the mapping");
    check((size_t)FT_TT_TLB_STRIDED_REGS_OFFSET +
                  (size_t)FT_TT_TLB_STRIDED_WINDOW_COUNT * 4u <=
              ft_tt_tlb_registers_size(ctx),
          "the strided-multicast register bank fits too");

    for (word = 0; word < 8; word++)
        sink |= regs[word];
    check(1, "register file is readable without faulting");
    printf("        first 8 words OR to 0x%08" PRIx32 "\n", sink);
}

static void test_window_lifecycle(ft_tt_ctx *ctx)
{
    ft_tt_tlb_window *windows[4] = {NULL, NULL, NULL, NULL};
    size_t i;
    size_t j;
    int all_distinct = 1;
    int all_in_range = 1;

    printf("TLB window lifecycle\n");
    check_u64(ft_tt_ctx_live_window_count(ctx), 0, "live windows before allocating");

    for (i = 0; i < 4; i++) {
        int rc = ft_tt_tlb_window_alloc(ctx, (size_t)FT_TT_KMD_WINDOW_SIZE_2M,
                                        FT_TT_KMD_CACHE_WC, &windows[i]);
        if (rc != FT_TT_OK) {
            printf("  FAIL  allocating window %zu: %s\n", i, ft_tt_error_string(rc));
            checks_run++;
            checks_failed++;
            break;
        }
    }

    check(windows[3] != NULL, "four 2 MiB windows allocate");
    check_u64(ft_tt_ctx_live_window_count(ctx), 4, "live window tally");

    for (i = 0; i < 4 && windows[i]; i++) {
        uint32_t id = ft_tt_tlb_window_id(windows[i]);

        if (id >= EXPECTED_TLB_2M_TOTAL || id == FT_TT_TLB_KERNEL_WINDOW_INDEX)
            all_in_range = 0;
        for (j = 0; j < i; j++) {
            if (ft_tt_tlb_window_id(windows[j]) == id)
                all_distinct = 0;
        }
        if (ft_tt_tlb_window_register_offset(windows[i]) != (size_t)id * FT_TT_TLB_REG_STRIDE)
            all_in_range = 0;
    }

    check(all_in_range,
          "ids are in [0, 202), never the kernel's 201, and map to id * 12 register bytes");
    check(all_distinct, "ids are distinct");
    check(ft_tt_tlb_window_size(windows[0]) == (size_t)FT_TT_KMD_WINDOW_SIZE_2M,
          "window size is 2 MiB");
    check(ft_tt_tlb_window_aperture(windows[0]) != NULL, "the aperture is mapped");
    check(!ft_tt_tlb_window_is_aimed(windows[0]), "a fresh window is aimed nowhere");

    for (i = 0; i < 4; i++) {
        if (windows[i])
            check(ft_tt_tlb_window_free(ctx, windows[i]) == FT_TT_OK, "window frees");
    }
    check_u64(ft_tt_ctx_live_window_count(ctx), 0, "live windows after freeing");
}

/* Direct proof that the derived 4 GiB count matches the driver: with no BAR4
 * the kernel finds zero windows of that size and rejects the request. A caller trusting upstream's table would call this
 * eight times and fail eight times. */
static void test_4g_window_is_refused(ft_tt_ctx *ctx)
{
    ft_tt_tlb_window *window = NULL;
    int rc;

    printf("4 GiB window request\n");
    rc = ft_tt_tlb_window_alloc(ctx, (size_t)FT_TT_KMD_WINDOW_SIZE_4G, FT_TT_KMD_CACHE_WC,
                                &window);
    check(rc == FT_TT_EINVAL, "the driver refuses a 4 GiB window with -EINVAL");
    check(window == NULL, "no handle is produced");
    if (window)
        (void)ft_tt_tlb_window_free(ctx, window);
}

/* Exhaust the pool. This is the only way to observe that 201, not 202, windows
 * are actually obtainable — the kernel claims index 201 at init. Nothing else
 * on the card host holds windows on this card. */
static void test_window_pool_exhaustion(ft_tt_ctx *ctx)
{
    static ft_tt_tlb_window *windows[MAX_WINDOWS_TRACKED];
    uint32_t obtained = 0;
    uint32_t kernel_window_seen = 0;
    size_t i;

    printf("TLB window pool exhaustion\n");
    memset(windows, 0, sizeof(windows));

    while (obtained < MAX_WINDOWS_TRACKED) {
        ft_tt_tlb_window *window = NULL;

        if (ft_tt_tlb_window_alloc(ctx, (size_t)FT_TT_KMD_WINDOW_SIZE_2M, FT_TT_KMD_CACHE_WC,
                                   &window) != FT_TT_OK)
            break;
        if (ft_tt_tlb_window_id(window) == FT_TT_TLB_KERNEL_WINDOW_INDEX)
            kernel_window_seen++;
        windows[obtained++] = window;
    }

    check_u64(obtained, EXPECTED_TLB_2M_AVAILABLE, "windows obtainable before -ENOMEM");
    check_u64(kernel_window_seen, 0, "the driver never hands out its own window");

    for (i = 0; i < obtained; i++)
        (void)ft_tt_tlb_window_free(ctx, windows[i]);

    check_u64(ft_tt_ctx_live_window_count(ctx), 0, "every window is returned");
}

/* Capacity is a telemetry question, and telemetry is Phase 1b. The contract is
 * that this REFUSES rather than reporting the nominal 8 banks as a fact. */
static void test_mem_info_is_honest(const ft_tt_ctx *ctx)
{
    ft_tt_grid_info grid;
    size_t free_bytes = 12345;
    size_t total_bytes = 12345;

    printf("capacity reporting\n");
    memset(&grid, 0, sizeof(grid));
    check(ft_tt_grid_get(ctx, &grid) == FT_TT_OK, "grid is readable");
    check(!grid.valid, "grid is marked invalid until ARC telemetry is read");
    check(ft_tt_mem_info(ctx, &free_bytes, &total_bytes) == FT_TT_EUNAVAIL,
          "ft_tt_mem_info refuses rather than guessing");
    check(free_bytes == 12345 && total_bytes == 12345,
          "it writes nothing when it cannot answer");
    check_u64(ft_tt_ctx_alloc_bytes(ctx), 0, "no device memory is allocated in Phase 1a");
}

/* The Phase 1a fence (ft_tt_kmd.h, "THE NOC FENCE"). Aiming a window is the operation
 * that hung this card; in a default build it must be mechanically unreachable,
 * not merely unused. */
static void test_noc_fence_is_closed(ft_tt_ctx *ctx)
{
    ft_tt_tlb_window *window = NULL;
    ft_tt_grid_info grid;
    int rc;

    printf("Phase 1a NoC fence\n");
    memset(&grid, 0, sizeof(grid));

    check(ft_tt_tlb_window_alloc(ctx, (size_t)FT_TT_KMD_WINDOW_SIZE_2M, FT_TT_KMD_CACHE_UC,
                                 &window) == FT_TT_OK,
          "a window allocates");
    if (!window)
        return;

    rc = ft_tt_tlb_window_aim(ctx, window, FT_TT_ARC_NOC0_X, FT_TT_ARC_NOC0_Y, 0x80000000u);
    check(rc == FT_TT_EPERM, "aiming a window is refused (CONFIGURE_TLB never issued)");
    check(!ft_tt_tlb_window_is_aimed(window), "the window is still aimed nowhere");
    check_u64(ft_tt_tlb_window_base_address(window), 0, "no base address was recorded");

    check(ft_tt_tlb_window_free(ctx, window) == FT_TT_OK, "the window frees");

    check(ft_tt_arc_telemetry_read_grid(ctx, &grid) == FT_TT_EPERM,
          "the ARC telemetry reader stops at the fence");
    check(!grid.valid, "no grid is produced");
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    ft_tt_ctx *ctx = NULL;

    printf("ft_tt Phase 1a device test — READ-ONLY against the card\n\n");

    test_driver_api_version_without_device();

    ctx = ft_tt_init_device(0);
    checks_run++;
    if (!ctx) {
        checks_failed++;
        printf("  FAIL  ft_tt_init_device(0)\n");
        printf("\n%d/%d checks failed\n", checks_failed, checks_run);
        return 1;
    }
    printf("  ok    ft_tt_init_device(0)\n");

    check(ft_tt_ctx_available(ctx), "context reports available");
    check_u64((uint64_t)ft_tt_ctx_device_index(ctx), 0, "device index");
    check(ft_tt_ctx_fd(ctx) >= 0, "device fd is open");

    test_device_identity(ctx);
    test_bar_mappings(ctx);
    test_derived_tlb_counts(ctx);
    test_tlb_register_file(ctx);
    test_window_lifecycle(ctx);
    test_4g_window_is_refused(ctx);
    test_window_pool_exhaustion(ctx);
    test_mem_info_is_honest(ctx);
    test_noc_fence_is_closed(ctx);

    ft_tt_ctx_free(ctx);

    printf("\n%d/%d checks failed\n", checks_failed, checks_run);
    return checks_failed == 0 ? 0 : 1;
}
