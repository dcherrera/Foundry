/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_card_arc.c
 * @brief ON-SILICON bring-up for the userspace driver's ARC PROBE SEQUENCE —
 *   driver step 6, and the last thing between here and unloading tt-kmd.
 *
 *   The ARC message encoders have existed since 2026-08-15 with no transport.
 *   `ft_tt_udrv_noc.[ch]` gave them one on 2026-08-17. This file sends real
 *   messages with it and asks whether the FIRMWARE answered — which is a
 *   different question from whether our ring arithmetic is self-consistent, and
 *   the two are easy to confuse.
 *
 *   WHAT MAKES EACH ANSWER INDEPENDENT OF THE MECHANISM THAT PRODUCED IT:
 *
 *     the queue address   RESET_SCRATCH(11) is confirmed by making TT-KMD send
 *                         a message and watching the request write pointer OF
 *                         THE QUEUE WE LOCATED advance by exactly one. If our
 *                         address were not the driver's, nothing would move.
 *     the doorbell        0x800B0000 is confirmed by a transport whose trigger
 *                         does nothing: the request is pushed identically and
 *                         no response ever arrives. The same arm is the proof
 *                         that a response is FIRMWARE action — the response
 *                         write pointer is the ARC's and nothing on the host
 *                         can move it.
 *     the messages        the same TEST message is sent through TT-KMD's
 *                         SMC_MSG ioctl and through ours, and the two responses
 *                         are compared word for word.
 *     POWER_SETTING       by the AI CLOCK, read on an instrument that is not
 *                         the message ring — tt-kmd's telemetry sysfs, and our
 *                         own NoC read of the ARC telemetry table — with Vcore,
 *                         board power, current, temperature and the firmware
 *                         heartbeat in the same row, so a dead row is visible
 *                         (DEV_NOTES: a clock reading is not a measurement
 *                         without a load you can see).
 *     the watchdog        an idle hold of 3x the timeout doing NOTHING, then
 *                         the link speed at both ends (a card reset drops it to
 *                         Gen1), the firmware heartbeat (a reset restarts it),
 *                         and the clock we asked for (a reset loses the
 *                         request). Three independent ways for a reset to show.
 *
 *   WHY TT-KMD'S POWER REQUEST IS ADVERSARIAL HERE AND THAT IS THE POINT. A
 *   legacy chardev open — which this file performs, because the oracle needs
 *   it — makes the driver request `FT_TT_KMD_POWER_FLAGS_ALL & ~MAX_AI_CLOCK`, the
 *   MINIMUM AI clock. So throughout the power arms the vendor driver is
 *   actively asking for 800 MHz and our message has to win. It does, and it is
 *   the reason the crossed control (ours asking for tt-kmd's own 0x7FFE) is
 *   worth running: it shows the clock tracks the CONTENT of our message rather
 *   than the act of sending one.
 *
 * @status Runs on the p150a at 0000:12:00.0, with tt-kmd
 *   loaded. Needs root (the BAR sysfs files are 0600 root:root).
 * @issues
 *   - The `load` arm boots firmware on one core, which is the operation this
 *     project's park/boot defect keys on. It is opt-in for that reason.
 * @todo None
 */

#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include "ft_tt_arch.h"
#include "ft_tt_coord.h"
#include "ft_tt_core_probe.h"
#include "ft_tt_core_status.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_kmd_abi.h"
#include "ft_tt_kmd_abi.h"
#include "ft_tt_test_report.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_arc_msg.h"
#include "ft_tt_udrv_arc_probe.h"
#include "ft_tt_udrv_noc.h"
#include "ft_tt_udrv_pci_sysfs.h"
#include "ft_tt_udrv_reset.h"
#include "ft_tt_udrv_tlb_encode.h"

#define CHARDEV_PATH "/dev/tenstorrent/0"
#define SYSFS_CLASS "/sys/class/tenstorrent/tenstorrent!0"
#define BRIDGE_DIR "/sys/bus/pci/devices/0000:00:1c.6"

/** Telemetry tag ids. A firmware contract, not a chip constant (the kernel
 *  driver's telemetry tag list); listed here because this file reads the table itself. */
#define TAG_VCORE 6u
#define TAG_POWER 7u
#define TAG_CURRENT 8u
#define TAG_ASIC_TEMP 11u
#define TAG_AICLK 14u
#define TAG_HEARTBEAT 32u

/** Where the ARC publishes the telemetry table and its values. */
#define ARC_TELEMETRY_TABLE_PTR FT_TT_UDRV_ARC_RESET_SCRATCH(13)
#define ARC_TELEMETRY_VALUES_PTR FT_TT_UDRV_ARC_RESET_SCRATCH(12)

/** The AI clocks this part runs at, from DEV_NOTES. Used only to LABEL a
 *  reading; every verdict below compares two readings taken in this session. */
#define AICLK_MIN_MHZ 800u
#define AICLK_MAX_MHZ 1350u

/** The idle hold: 3x tt-kmd's default watchdog, plus a margin so a card that
 *  resets exactly at the boundary is still caught. */
#define WATCHDOG_MS FT_TT_UDRV_ARC_WATCHDOG_DEFAULT_MS
#define IDLE_HOLD_SECONDS 35u

/** The `load` arm. FOUR cores, and the length comes from REPEATS rather than
 *  from more cores or more probe calls, because `ft_tt_core_probe_run()` boots
 *  the five RISC-Vs ONCE and then re-launches: 1500 repeats is seconds of real
 *  Tensix work for exactly one boot per core. Re-booting a core is this
 *  project's standing wedge trigger, so the load is made long by working the
 *  core harder and not by booting it more. */
#define LOAD_CORE_COUNT 4u
#define LOAD_REPEATS 1500u
static const uint32_t load_cores[LOAD_CORE_COUNT] = {0u, 1u, 2u, 3u};

/* ------------------------------------------------------------------------ *
 * Session state
 * ------------------------------------------------------------------------ */

static ft_tt_udrv_pci_device card;
static ft_tt_udrv_noc_device udrv;
static bool udrv_open;

static int chardev = -1;
static const volatile uint32_t *driver_registers; /**< tt-kmd's own BAR0 mapping */

static uint32_t claimed[2];
static size_t claimed_count;

static ft_tt_udrv_noc_path arc_path;
static bool arc_path_open;

static ft_tt_udrv_arc_noc_context arc_context;
static ft_tt_udrv_arc_transport arc_transport;
static ft_tt_udrv_arc_queue arc_queue;
static bool arc_queue_located;

static uint16_t arc_x, arc_y;

/** The `load` arm's tt-kmd-side session. Opened in rung 1 when it is wanted, so
 *  that every chardev this run will ever open is already open before rung 6
 *  sends a POWER_SETTING — an open recomputes tt-kmd's aggregate and would
 *  overwrite ours. */
static ft_tt_hardware_session session;
static bool session_open;
static ft_tt_core_probe_context probe_context;
static bool probe_context_open;
static bool want_load;

/* ------------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------------ */

static double seconds_now(void)
{
    struct timespec now;

    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void sleep_seconds(unsigned seconds)
{
    struct timespec request;

    request.tv_sec = (time_t)seconds;
    request.tv_nsec = 0;
    (void)nanosleep(&request, NULL);
}

/** Read one decimal integer out of a sysfs file. `-1` means unreadable. */
static long read_sysfs_long(const char *path)
{
    char buffer[64];
    FILE *file = fopen(path, "r");
    long value = -1;

    if (!file)
        return -1;

    if (fgets(buffer, sizeof(buffer), file))
        value = strtol(buffer, NULL, 10);

    (void)fclose(file);
    return value;
}

static long read_class_attribute(const char *name)
{
    char path[256];

    if (snprintf(path, sizeof(path), "%s/%s", SYSFS_CLASS, name) >= (int)sizeof(path))
        return -1;
    return read_sysfs_long(path);
}

/** hwmon numbering is discovered once; the directory name is not a constant. */
static long read_hwmon_attribute(const char *name)
{
    char path[320];
    static char hwmon_dir[256];
    long value;
    unsigned index;

    if (hwmon_dir[0] == '\0') {
        for (index = 0u; index < 32u; index++) {
            (void)snprintf(hwmon_dir, sizeof(hwmon_dir), "%s/device/hwmon/hwmon%u", SYSFS_CLASS,
                           index);
            if (snprintf(path, sizeof(path), "%s/name", hwmon_dir) < (int)sizeof(path) &&
                access(path, R_OK) == 0)
                break;
            hwmon_dir[0] = '\0';
        }
        if (hwmon_dir[0] == '\0')
            return -1;
    }

    if (snprintf(path, sizeof(path), "%s/%s", hwmon_dir, name) >= (int)sizeof(path))
        return -1;
    value = read_sysfs_long(path);
    return value;
}

/* ------------------------------------------------------------------------ *
 * The telemetry row — read TWICE, through two unrelated paths
 * ------------------------------------------------------------------------ */

/**
 * One sample of everything that can show whether the card acted on a message.
 *
 * `*_sysfs` comes from tt-kmd, which reads the ARC telemetry table through its
 * OWN private TLB window on every access — a different
 * file, a different window, a different address space from ours.
 * `*_ours` comes from this process walking the same table over the userspace
 * NoC path. Both are recorded because the first disappears with the module and
 * the second has to be trustworthy afterwards; agreement here is what earns it.
 */
typedef struct telemetry_row {
    const char *label;
    bool sysfs_ok;
    long aiclk_sysfs;
    long heartbeat_sysfs;
    long temp_millidegrees;
    long vcore_millivolts;
    long current_milliamps;
    long power_microwatts;

    bool ours_ok;
    uint32_t aiclk_ours;
    uint32_t heartbeat_ours;
    uint32_t vcore_ours;
    uint32_t power_ours;
    uint32_t current_ours;
    uint32_t temp_ours;
} telemetry_row;

/**
 * Walk the ARC telemetry table over OUR path and read one tag.
 *
 * This is `blackhole_populate_telemetry_cache()` reduced to a single lookup: the
 * table pointer and the values pointer come from the reset scratch file, every
 * pointer is bounds-checked against the CSM before it is dereferenced, and the
 * entry count is bounded by the tag space.
 */
static int read_tag_over_our_path(uint16_t tag, uint32_t *out_value)
{
    uint32_t table_pointer = 0u;
    uint32_t values_pointer = 0u;
    uint32_t entries = 0u;
    uint32_t index;
    int rc;

    if (!out_value)
        return FT_TT_UDRV_EINVAL;

    rc = ft_tt_udrv_arc_noc_read32(&arc_context, ARC_TELEMETRY_TABLE_PTR, &table_pointer);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = ft_tt_udrv_arc_noc_read32(&arc_context, ARC_TELEMETRY_VALUES_PTR, &values_pointer);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    if (table_pointer < FT_TT_UDRV_ARC_CSM_BASE ||
        table_pointer + 8u > FT_TT_UDRV_ARC_CSM_BASE + FT_TT_UDRV_ARC_CSM_SIZE ||
        values_pointer < FT_TT_UDRV_ARC_CSM_BASE ||
        values_pointer + 4u > FT_TT_UDRV_ARC_CSM_BASE + FT_TT_UDRV_ARC_CSM_SIZE)
        return FT_TT_UDRV_ENODEV;

    rc = ft_tt_udrv_arc_noc_read32(&arc_context, table_pointer + 4u, &entries);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    if (entries == 0u || entries > (1u << 16))
        return FT_TT_UDRV_EIO;

    for (index = 0u; index < entries; index++) {
        uint32_t entry = 0u;
        uint32_t address;

        rc = ft_tt_udrv_arc_noc_read32(&arc_context, table_pointer + 8u + (index * 4u), &entry);
        if (rc != FT_TT_UDRV_OK)
            return rc;

        if ((uint16_t)(entry & 0xFFFFu) != tag)
            continue;

        address = values_pointer + (((entry >> 16) & 0xFFFFu) * 4u);
        if (address < FT_TT_UDRV_ARC_CSM_BASE ||
            address + 4u > FT_TT_UDRV_ARC_CSM_BASE + FT_TT_UDRV_ARC_CSM_SIZE)
            return FT_TT_UDRV_EIO;

        return ft_tt_udrv_arc_noc_read32(&arc_context, address, out_value);
    }

    return FT_TT_UDRV_ENODEV;
}

static void sample_telemetry(const char *label, telemetry_row *out_row)
{
    memset(out_row, 0, sizeof(*out_row));
    out_row->label = label;

    out_row->aiclk_sysfs = read_class_attribute("tt_aiclk");
    out_row->heartbeat_sysfs = read_class_attribute("tt_heartbeat");
    out_row->temp_millidegrees = read_hwmon_attribute("temp1_input");
    out_row->vcore_millivolts = read_hwmon_attribute("in0_input");
    out_row->current_milliamps = read_hwmon_attribute("curr1_input");
    out_row->power_microwatts = read_hwmon_attribute("power1_input");
    out_row->sysfs_ok = (out_row->aiclk_sysfs >= 0) && (out_row->heartbeat_sysfs >= 0);

    if (arc_path_open) {
        out_row->ours_ok = read_tag_over_our_path(TAG_AICLK, &out_row->aiclk_ours) ==
                               FT_TT_UDRV_OK &&
                           read_tag_over_our_path(TAG_HEARTBEAT, &out_row->heartbeat_ours) ==
                               FT_TT_UDRV_OK &&
                           read_tag_over_our_path(TAG_VCORE, &out_row->vcore_ours) ==
                               FT_TT_UDRV_OK &&
                           read_tag_over_our_path(TAG_POWER, &out_row->power_ours) ==
                               FT_TT_UDRV_OK &&
                           read_tag_over_our_path(TAG_CURRENT, &out_row->current_ours) ==
                               FT_TT_UDRV_OK &&
                           read_tag_over_our_path(TAG_ASIC_TEMP, &out_row->temp_ours) ==
                               FT_TT_UDRV_OK;
    }
}

static void print_telemetry_row(const telemetry_row *row)
{
    printf("    %-26s aiclk %4ld MHz | vcore %4ld mV | %6.2f W | %5.2f A | %5.1f C | "
           "heartbeat %ld\n",
           row->label, row->aiclk_sysfs, row->vcore_millivolts,
           (double)row->power_microwatts / 1e6, (double)row->current_milliamps / 1e3,
           (double)row->temp_millidegrees / 1e3, row->heartbeat_sysfs);
    if (row->ours_ok)
        printf("    %-26s aiclk %4u MHz | vcore %4u mV | (our own NoC read of the same table)\n",
               "  ^ without tt-kmd:", row->aiclk_ours, row->vcore_ours);
    else
        printf("    %-26s our own telemetry read FAILED\n", "  ^ without tt-kmd:");
}

/* ------------------------------------------------------------------------ *
 * tt-kmd, as arbitrator and as oracle
 * ------------------------------------------------------------------------ */

static int allocate_window(uint32_t *out_id)
{
    struct ft_tt_kmd_allocate_window_request request;

    memset(&request, 0, sizeof(request));
    request.in.size = (uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE;

    if (ioctl(chardev, FT_TT_KMD_IOCTL_ALLOCATE_WINDOW, &request) != 0)
        return FT_TT_UDRV_EIO;

    *out_id = request.out.id;
    return FT_TT_UDRV_OK;
}

static bool query_driver_bar0_mapping(uint64_t *out_base)
{
    struct {
        struct ft_tt_kmd_query_mappings_in in;
        struct ft_tt_kmd_mapping mappings[8];
    } query;
    size_t index;

    memset(&query, 0, sizeof(query));
    query.in.count = 8u;

    if (ioctl(chardev, FT_TT_KMD_IOCTL_QUERY_MAPPINGS, &query) != 0)
        return false;

    for (index = 0u; index < 8u; index++) {
        if (query.mappings[index].id != FT_TT_KMD_MAPPING_RESOURCE0_UC)
            continue;
        *out_base = query.mappings[index].base;
        return true;
    }
    return false;
}

/**
 * Make TT-KMD send an ARC message of our choosing and hand back the firmware's
 * response. POST then POLL; the driver does not interpret the status word, so a
 * nonzero one still comes back to us and can be compared.
 *
 * @return FT_TT_UDRV_OK, or FT_TT_UDRV_EIO with errno left set
 */
static int send_via_driver(const ft_tt_udrv_arc_msg *request, ft_tt_udrv_arc_msg *out_response)
{
    struct ft_tt_kmd_smc_message message;
    unsigned attempt;

    memset(&message, 0, sizeof(message));
    message.argsz = (uint32_t)sizeof(message);
    message.flags = FT_TT_KMD_SMC_MESSAGE_POST;
    message.words[0] = request->header;
    memcpy(&message.words[1], request->payload, sizeof(request->payload));

    if (ioctl(chardev, FT_TT_KMD_IOCTL_SMC_MESSAGE, &message) != 0)
        return FT_TT_UDRV_EIO;

    for (attempt = 0u; attempt < 2000u; attempt++) {
        memset(&message, 0, sizeof(message));
        message.argsz = (uint32_t)sizeof(message);
        message.flags = FT_TT_KMD_SMC_MESSAGE_POLL;

        if (ioctl(chardev, FT_TT_KMD_IOCTL_SMC_MESSAGE, &message) == 0) {
            out_response->header = message.words[0];
            memcpy(out_response->payload, &message.words[1], sizeof(out_response->payload));
            return FT_TT_UDRV_OK;
        }
        if (errno != EAGAIN)
            return FT_TT_UDRV_EIO;

        usleep(500u);
    }
    return FT_TT_UDRV_EIO;
}

/* ------------------------------------------------------------------------ *
 * Rung 1 — open everything, and derive the ARC coordinate
 * ------------------------------------------------------------------------ */

/**
 * Open the tt-kmd-side session the `load` arm needs, ASKING FOR THE MINIMUM AI
 * CLOCK on the way in.
 *
 * That is not a detail. It makes the vendor path adversarial for the whole rest
 * of the run: two legacy fds and an explicit SET_POWER_STATE are all requesting
 * 800 MHz, and the only thing asking for 1350 is our own ARC message. A load arm
 * that let foundry ask for max would have proved nothing about our message.
 */
static bool open_load_session(void)
{
    uint16_t x = 0u, y = 0u;
    int rc;

    (void)setenv("FT_TT_AI_CLK", "min", 1);

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the load session opens", false, "rc=%d", rc);
        return false;
    }
    session_open = true;

    /* This whole suite measures OUR ARC path against tt-kmd's, so the session that
     * drives the Tensix load must be the vendor one. The session honours
     * FT_TT_DRIVER now; an environment that redirected it would quietly replace the
     * oracle with the thing under test. */
    if (!ft_tt_hardware_session_kmd_context(&session)) {
        ft_tt_test_check("the load session is on tt-kmd — FT_TT_DRIVER must not redirect it",
                         false, "this suite's oracle is the vendor driver");
        return false;
    }

    rc = ft_tt_hardware_session_claim_cores(&session, load_cores, LOAD_CORE_COUNT,
                                            "udrv ARC probe: load arm", NULL, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the load arm claims its cores", false, "rc=%d", rc);
        return false;
    }

    rc = ft_tt_core_probe_context_open(ft_tt_firmware_directory(), &probe_context);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the probe images load", false, "%s rc=%d", ft_tt_firmware_directory(),
                             rc);
        return false;
    }
    probe_context_open = true;

    (void)ft_tt_hardware_session_tensix(&session, load_cores[0], &x, &y);
    ft_tt_test_check_fmt("the load arm holds its cores, and foundry's own open asked for the "
                         "MINIMUM AI clock — so nothing but our ARC message is asking for max",
                         true, "%u cores %u..%u, first at (%u, %u), images from %s",
                         LOAD_CORE_COUNT, load_cores[0], load_cores[LOAD_CORE_COUNT - 1u], x, y,
                         ft_tt_firmware_directory());
    return true;
}

static bool stage_open_the_card(bool writing)
{
    size_t count = 0u;
    uint64_t driver_base = 0u;
    void *driver_mapping;
    const ft_tt_arch *arch = ft_tt_arch_blackhole();
    ft_tt_noc_coord coord;
    uint32_t index;
    bool device_id_known = false;

    ft_tt_test_step("Rung 1 — the card, our own BAR0 mapping, and tt-kmd as arbitrator + oracle");

    ft_tt_test_check_fmt("running as root (the BAR files are 0600 root:root)", geteuid() == 0u,
                         "euid=%u", (unsigned)geteuid());

    if (ft_tt_udrv_pci_find_devices(NULL, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT,
                                    FT_TT_UDRV_PCI_DEVICE_BLACKHOLE, &card, 1u,
                                    &count) != FT_TT_UDRV_OK ||
        count == 0u) {
        ft_tt_test_check("a Blackhole is present in sysfs", false, "none found");
        return false;
    }
    ft_tt_test_check_fmt("a Blackhole is present in sysfs", true, "%s", card.address);

    /* The ARC coordinate is DERIVED, never probed and never a literal here: the
     * arch table is selected by the PCI device id sysfs reports, and
     * ft_tt_coord_arc() answers from it. */
    for (index = 0u; index < arch->pci_device_id_count; index++)
        if (arch->pci_device_ids[index] == card.device)
            device_id_known = true;
    ft_tt_test_check_fmt("sysfs's PCI device id is one the Blackhole arch table names",
                         device_id_known, "0x%04x", card.device);
    if (!device_id_known)
        return false;

    if (ft_tt_coord_arc(arch, FT_TT_COORD_TRANSLATED, &coord) != FT_TT_OK) {
        ft_tt_test_check("ft_tt_coord_arc() answers", false, "");
        return false;
    }
    arc_x = (uint16_t)coord.x;
    arc_y = (uint16_t)coord.y;
    ft_tt_test_check_fmt("the ARC tile is DERIVED from the arch table, not written down here",
                         arc_x == FT_TT_UDRV_ARC_X && arc_y == FT_TT_UDRV_ARC_Y, "(%u, %u)", arc_x,
                         arc_y);

    if (ft_tt_udrv_noc_device_open(&card, writing, &udrv) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("BAR0's TLB register file maps", false, "errno %d (%s)", errno,
                             strerror(errno));
        return false;
    }
    udrv_open = true;
    ft_tt_test_check_fmt("BAR0's TLB register file maps, uncached", true, "resource0 + 0x%08X, %s",
                         FT_TT_UDRV_TLB_REGS_START, writing ? "PROT_READ|PROT_WRITE" : "PROT_READ");

    chardev = open(CHARDEV_PATH, O_RDWR | O_CLOEXEC);
    if (chardev < 0) {
        ft_tt_test_check_fmt("tt-kmd's chardev opens — it is the ORACLE and the way back", false,
                             "%s errno %d", CHARDEV_PATH, errno);
        return false;
    }
    ft_tt_test_check_fmt("tt-kmd's chardev opens — a LEGACY open, so the driver is now asking "
                         "the firmware for the MINIMUM AI clock",
                         true, "fd=%d", chardev);

    if (!query_driver_bar0_mapping(&driver_base)) {
        ft_tt_test_check("QUERY_MAPPINGS reports BAR0's UC mapping", false, "");
        return false;
    }
    driver_mapping = mmap(NULL, FT_TT_UDRV_TLB_REGS_LEN, PROT_READ, MAP_SHARED, chardev,
                          (off_t)(driver_base + FT_TT_UDRV_TLB_REGS_START));
    if (driver_mapping == MAP_FAILED) {
        ft_tt_test_check_fmt("tt-kmd's own BAR0 mapping — the second reader", false, "errno %d",
                             errno);
        return false;
    }
    driver_registers = (const volatile uint32_t *)driver_mapping;
    ft_tt_test_check("tt-kmd's own BAR0 mapping — the second reader", true, "");

    for (index = 0u; index < 2u; index++) {
        if (allocate_window(&claimed[claimed_count]) != FT_TT_UDRV_OK) {
            ft_tt_test_check_fmt("ALLOCATE_TLB — arbitration while the module is still here",
                                 false, "after %zu", claimed_count);
            return false;
        }
        claimed_count++;
    }
    ft_tt_test_check_fmt("ALLOCATE_TLB — two windows, arbitrated by tt-kmd", true, "%u and %u",
                         claimed[0], claimed[1]);

    /* Every chardev this run will open has to be open BEFORE rung 6 sends a
     * POWER_SETTING, because an open makes tt-kmd recompute its aggregate and
     * push it to the firmware, which would silently overwrite ours. */
    if (want_load && !open_load_session())
        return false;
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung 2 — a path aimed at the ARC, and the queue behind RESET_SCRATCH(11)
 * ------------------------------------------------------------------------ */

/** Aim confirmation through tt-kmd's mapping, before anything is dereferenced. */
static bool confirm_aim_through_driver(uint32_t id, const ft_tt_udrv_tlb2m_config *config)
{
    uint32_t intended[3] = {0u, 0u, 0u};
    uint32_t through_driver[3];
    size_t base = (size_t)id * (FT_TT_UDRV_TLB_REG_SIZE / 4u);

    (void)ft_tt_udrv_tlb2m_encode(config, intended);
    through_driver[0] = driver_registers[base + 0u];
    through_driver[1] = driver_registers[base + 1u];
    through_driver[2] = driver_registers[base + 2u];

    return memcmp(intended, through_driver, sizeof(intended)) == 0;
}

static bool stage_locate_the_queue(void)
{
    ft_tt_udrv_arc_locate_report report;
    ft_tt_udrv_tlb2m_config aim;
    uint32_t neighbour_10 = 0u, neighbour_12 = 0u, neighbour_13 = 0u;
    uint64_t ring_bytes;
    int rc;

    ft_tt_test_step("Rung 2 — a UNCACHED path on our own window, and the message queue that "
                    "RESET_SCRATCH(11) points at");

    if (ft_tt_udrv_noc_path_open(&udrv, claimed[0], false, &arc_path) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("a path opens on window %u, UNCACHED", false, "%u", claimed[0]);
        return false;
    }
    arc_path_open = true;
    ft_tt_test_check_fmt("a path opens on our window, UNCACHED — write combining would let a "
                         "ring pointer overtake the payload it publishes",
                         true, "window %u", claimed[0]);

    /* The first transfer aims the window; confirm that aim through the reader
     * that is not ours before a single ARC register is touched. */
    memset(&aim, 0, sizeof(aim));
    aim.addr = (uint64_t)FT_TT_UDRV_ARC_RESET_SCRATCH(0) &
               ~((uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE - 1u);
    aim.x_end = (uint8_t)arc_x;
    aim.y_end = (uint8_t)arc_y;

    rc = ft_tt_udrv_arc_transport_init(&arc_context, &arc_path, arc_x, arc_y, &arc_transport);
    ft_tt_test_check_fmt("a transport binds to the path", rc == FT_TT_UDRV_OK, "rc=%d", rc);
    if (rc != FT_TT_UDRV_OK)
        return false;

    rc = ft_tt_udrv_arc_noc_read32(&arc_context, FT_TT_UDRV_ARC_BOOT_STATUS, &neighbour_10);
    if (rc != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("the ARC answers a NoC read at all", false, "rc=%d", rc);
        return false;
    }
    ft_tt_test_check_fmt("our aim at the ARC agrees with TT-KMD's mapping of the same registers",
                         confirm_aim_through_driver(claimed[0], &aim), "window %u", claimed[0]);

    rc = ft_tt_udrv_arc_locate_queue(&arc_context, FT_TT_UDRV_ARC_READY_TIMEOUT_MS, &report,
                                     &arc_queue);
    ft_tt_test_check_fmt("ARC_BOOT_STATUS says the firmware will accept queue operations",
                         (report.boot_status & FT_TT_UDRV_ARC_BOOT_STATUS_READY_FOR_MSG) != 0u,
                         "0x%08x after %u poll%s", report.boot_status, report.boot_status_polls,
                         report.boot_status_polls == 1u ? "" : "s");

    ft_tt_test_check_fmt("RESET_SCRATCH(11) holds a CSM address — THE FIRST OF THE TWO ADDRESSES "
                         "THIS ROUND EXISTS TO CONFIRM",
                         report.qcb_pointer >= FT_TT_UDRV_ARC_CSM_BASE &&
                             report.qcb_pointer <
                                 FT_TT_UDRV_ARC_CSM_BASE + FT_TT_UDRV_ARC_CSM_SIZE,
                         "0x%08X = 0x%08X", FT_TT_UDRV_ARC_MSG_QCB_PTR, report.qcb_pointer);

    ft_tt_test_check_fmt("the queue descriptor builds and fits inside the CSM", rc == FT_TT_UDRV_OK,
                         "rc=%d", rc);
    if (rc != FT_TT_UDRV_OK)
        return false;
    arc_queue_located = true;

    ring_bytes = (uint64_t)FT_TT_UDRV_ARC_QUEUE_HEADER_BYTES +
                 (2u * (uint64_t)arc_queue.num_entries * FT_TT_UDRV_ARC_MSG_BYTES);
    printf("    QCB 0x%08X -> base 0x%08X, %u entries (%llu bytes), requests 0x%08X, "
           "responses 0x%08X\n",
           report.qcb_pointer, arc_queue.base, arc_queue.num_entries,
           (unsigned long long)ring_bytes, arc_queue.request_base, arc_queue.response_base);

    /* "Any scratch register would have worked" is the way this could be vacuous.
     * Read the three neighbours: they hold different values, and only 11's
     * produces a queue the firmware answers on (proved in rung 3). */
    (void)ft_tt_udrv_arc_noc_read32(&arc_context, FT_TT_UDRV_ARC_RESET_SCRATCH(10), &neighbour_10);
    (void)ft_tt_udrv_arc_noc_read32(&arc_context, ARC_TELEMETRY_VALUES_PTR, &neighbour_12);
    (void)ft_tt_udrv_arc_noc_read32(&arc_context, ARC_TELEMETRY_TABLE_PTR, &neighbour_13);
    ft_tt_test_check_fmt("scratch 11 is not interchangeable with its neighbours",
                         report.qcb_pointer != neighbour_10 && report.qcb_pointer != neighbour_12 &&
                             report.qcb_pointer != neighbour_13,
                         "10=0x%08x 11=0x%08x 12=0x%08x 13=0x%08x", neighbour_10,
                         report.qcb_pointer, neighbour_12, neighbour_13);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung 3 — the queue we located IS the queue tt-kmd uses
 * ------------------------------------------------------------------------ */

static bool stage_queue_is_the_drivers(void)
{
    ft_tt_udrv_arc_msg request, response;
    uint32_t wptr_before = 0u, wptr_after = 0u;
    uint32_t decoy_before = 0u, decoy_after = 0u;
    uint32_t expected = 0u;
    int rc;

    ft_tt_test_step("Rung 3 — RESET_SCRATCH(11) LOCATES TT-KMD'S OWN QUEUE, proved by making the "
                    "driver send a message and watching OUR pointer move");

    (void)ft_tt_udrv_arc_noc_read32(&arc_context, arc_queue.req_wptr_addr, &wptr_before);
    /* Queue header word 2 (base + 0x08) is not one of the four pointers. If it
     * moved too, "the pointer advanced" would mean "everything changed". */
    (void)ft_tt_udrv_arc_noc_read32(&arc_context, arc_queue.base + 0x08u, &decoy_before);

    rc = ft_tt_udrv_arc_msg_test(&request);
    if (rc != FT_TT_UDRV_OK)
        return false;

    memset(&response, 0, sizeof(response));
    rc = send_via_driver(&request, &response);
    ft_tt_test_check_fmt("TT-KMD sends a TEST message on our behalf (SMC_MSG POST + POLL)",
                         rc == FT_TT_UDRV_OK, "rc=%d errno=%d", rc, errno);
    if (rc != FT_TT_UDRV_OK)
        return false;
    printf("    tt-kmd's response: header 0x%08x payload %08x %08x %08x %08x %08x %08x %08x\n",
           response.header, response.payload[0], response.payload[1], response.payload[2],
           response.payload[3], response.payload[4], response.payload[5], response.payload[6]);

    (void)ft_tt_udrv_arc_noc_read32(&arc_context, arc_queue.req_wptr_addr, &wptr_after);
    (void)ft_tt_udrv_arc_noc_read32(&arc_context, arc_queue.base + 0x08u, &decoy_after);

    (void)ft_tt_udrv_arc_ring_advance(wptr_before, arc_queue.num_entries, &expected);
    ft_tt_test_check_fmt("the request write pointer OF THE QUEUE WE LOCATED advanced by exactly "
                         "one — so our address is the driver's address",
                         wptr_after == expected, "%u -> %u (expected %u), 2N=%u", wptr_before,
                         wptr_after, expected, 2u * arc_queue.num_entries);

    ft_tt_test_check_fmt("and a header word that is NOT a pointer did not move — the advance is "
                         "at the offset we claim, not everywhere",
                         decoy_before == decoy_after, "base+0x08: 0x%08x -> 0x%08x", decoy_before,
                         decoy_after);
    return wptr_after == expected;
}

/* ------------------------------------------------------------------------ *
 * Rung 4 — the doorbell, and what makes a response FIRMWARE evidence
 * ------------------------------------------------------------------------ */

static int trigger_that_does_nothing(void *context)
{
    (void)context;
    return 0;
}

static bool stage_doorbell_is_load_bearing(void)
{
    ft_tt_udrv_arc_transport deaf = arc_transport;
    ft_tt_udrv_arc_msg message;
    ft_tt_udrv_arc_exchange exchange;
    uint64_t triggers_before;
    uint32_t res_wptr_before = 0u, res_wptr_after = 0u;
    int rc;

    ft_tt_test_step("Rung 4 — 0x800B0000 IS THE DOORBELL, and a response is the ARC's action and "
                    "not our own bookkeeping");

    deaf.trigger = trigger_that_does_nothing;
    triggers_before = arc_context.triggers;

    (void)ft_tt_udrv_arc_noc_read32(&arc_context, arc_queue.res_wptr_addr, &res_wptr_before);

    /* THE RED ARM. Identical push, identical poll loop, one thing removed: the
     * write to ARC_MSI_FIFO. The budget is the FULL one the green arms use —
     * a short one would only show the firmware is slower than the wait, and the
     * green arms answer in under 0.1 ms, so this waits ~10,000x that. */
    (void)ft_tt_udrv_arc_msg_test(&message);
    rc = ft_tt_udrv_arc_exchange_send(&deaf, &arc_queue, &message, 0u, &exchange);
    ft_tt_test_check_fmt("WITHOUT the doorbell write, the request is pushed and NO response ever "
                         "arrives",
                         rc == FT_TT_UDRV_EAGAIN, "rc=%d after %.3f s", rc,
                         exchange.elapsed_seconds);
    ft_tt_test_check_fmt("and that arm really did skip the doorbell", arc_context.triggers ==
                                                                         triggers_before,
                         "%llu triggers", (unsigned long long)arc_context.triggers);

    (void)ft_tt_udrv_arc_noc_read32(&arc_context, arc_queue.res_wptr_addr, &res_wptr_after);
    ft_tt_test_check_fmt("the RESPONSE write pointer — which only the ARC can move — did not "
                         "move either",
                         res_wptr_before == res_wptr_after, "%u -> %u", res_wptr_before,
                         res_wptr_after);

    /* Now ring it. The message from the red arm is still sitting in the request
     * ring, so this both proves the doorbell released it and leaves the ring
     * clean for everything that follows. */
    rc = arc_transport.trigger(arc_transport.context);
    ft_tt_test_check_fmt("the doorbell write itself lands", rc == FT_TT_UDRV_OK, "0x%08X, rc=%d",
                         FT_TT_UDRV_ARC_MSI_FIFO, rc);

    {
        unsigned poll;
        ft_tt_udrv_arc_msg drained;

        for (poll = 0u; poll < 4000u; poll++) {
            if (ft_tt_udrv_arc_try_pop(&arc_transport, &arc_queue, &drained) == FT_TT_UDRV_OK)
                break;
            usleep(FT_TT_UDRV_ARC_POLL_WAIT_US);
        }
        ft_tt_test_check_fmt("ringing the doorbell releases the message that was already queued — "
                             "so the doorbell is what the firmware waits on",
                             poll < 4000u, "after %u poll%s, response header 0x%08x", poll,
                             poll == 1u ? "" : "s", drained.header);
        (void)ft_tt_udrv_arc_noc_read32(&arc_context, arc_queue.res_wptr_addr, &res_wptr_after);
        ft_tt_test_check_fmt("and now the ARC's own response pointer HAS moved", res_wptr_after !=
                                                                                    res_wptr_before,
                             "%u -> %u", res_wptr_before, res_wptr_after);
        return poll < 4000u;
    }
}

/* ------------------------------------------------------------------------ *
 * Rung 5 — our messages, and tt-kmd's answer to the same one
 * ------------------------------------------------------------------------ */

static void print_exchange(const char *name, const ft_tt_udrv_arc_exchange *exchange)
{
    printf("    %-18s request 0x%08x -> response 0x%08x  %08x %08x %08x %08x %08x %08x %08x  "
           "(%.3f ms)\n",
           name, exchange->request_header, exchange->response_header, exchange->response_payload[0],
           exchange->response_payload[1], exchange->response_payload[2],
           exchange->response_payload[3], exchange->response_payload[4],
           exchange->response_payload[5], exchange->response_payload[6],
           exchange->elapsed_seconds * 1e3);
}

static bool stage_our_messages(void)
{
    ft_tt_udrv_arc_msg message, theirs;
    ft_tt_udrv_arc_exchange ours;
    int rc;

    ft_tt_test_step("Rung 5 — ASIC_STATE0 and SET_WDT_TIMEOUT, the whole hardware footprint of "
                    "blackhole_init_hardware(), sent by us");

    /* TEST first, because it is the one message tt-kmd will also send for us
     * and therefore the only one that can be compared rather than believed. */
    (void)ft_tt_udrv_arc_msg_test(&message);
    rc = ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &message, 0u, &ours);
    ft_tt_test_check_fmt("TEST (0x90) through OUR transport is answered with status 0",
                         rc == FT_TT_UDRV_OK, "rc=%d, header 0x%08x", rc, ours.response_header);
    print_exchange("TEST, ours", &ours);

    memset(&theirs, 0, sizeof(theirs));
    (void)ft_tt_udrv_arc_msg_test(&message);
    if (send_via_driver(&message, &theirs) == FT_TT_UDRV_OK) {
        /* SEVEN of the eight words must match exactly. The eighth, payload[1],
         * is a FIRMWARE-SIDE COUNTER of TEST messages — it read 0x72 for the
         * driver's message in rung 3, 0x73 for the one the doorbell arm
         * released, 0x74 for ours here — so demanding equality would be
         * demanding the firmware not count. Demanding it advance by exactly one
         * is the stronger statement anyway: it says both paths feed the SAME
         * queue and the firmware served them in the order they were pushed. */
        bool fixed_words_match = theirs.header == ours.response_header &&
                                 theirs.payload[0] == ours.response_payload[0] &&
                                 memcmp(&theirs.payload[2], &ours.response_payload[2],
                                        5u * sizeof(uint32_t)) == 0;

        ft_tt_test_check_fmt("BYTE COMPARISON: seven of the eight words tt-kmd gets back are the "
                             "words we got back",
                             fixed_words_match, "theirs 0x%08x/%08x, ours 0x%08x/%08x",
                             theirs.header, theirs.payload[0], ours.response_header,
                             ours.response_payload[0]);
        ft_tt_test_check_fmt("and the eighth — the firmware's own TEST counter — advanced by "
                             "exactly one from ours to theirs, so both paths feed ONE queue",
                             theirs.payload[1] == ours.response_payload[1] + 1u,
                             "ours 0x%02x -> theirs 0x%02x", ours.response_payload[1],
                             theirs.payload[1]);
    } else {
        ft_tt_test_check("tt-kmd sends the comparison message", false, "SMC_MSG failed");
    }

    (void)ft_tt_udrv_arc_msg_asic_state0(&message);
    rc = ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &message, 0u, &ours);
    ft_tt_test_check_fmt("ASIC_STATE0 (0xA0) — the power-up handshake — answered with status 0",
                         rc == FT_TT_UDRV_OK, "rc=%d, header 0x%08x", rc, ours.response_header);
    print_exchange("ASIC_STATE0", &ours);

    (void)ft_tt_udrv_arc_msg_set_wdt_timeout(WATCHDOG_MS, &message);
    rc = ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &message, 0u, &ours);
    ft_tt_test_check_fmt("SET_WDT_TIMEOUT (0xC1) at 10 s — tt-kmd's own default — answered with "
                         "status 0, so this firmware DOES implement it",
                         rc == FT_TT_UDRV_OK, "rc=%d, header 0x%08x", rc, ours.response_header);
    print_exchange("SET_WDT_TIMEOUT", &ours);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung 6 — POWER_SETTING, measured on an instrument that is not the ring
 * ------------------------------------------------------------------------ */

/** Send one POWER_SETTING and sample the row after it has had time to land. */
static bool request_clock(bool max_ai_clock, const char *label, telemetry_row *out_row)
{
    ft_tt_udrv_arc_msg message;
    ft_tt_udrv_arc_exchange exchange;
    int rc;

    (void)ft_tt_udrv_arc_power_request_build(max_ai_clock, &message);
    rc = ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &message, 0u, &exchange);
    ft_tt_test_check_fmt("POWER_SETTING (0x21) accepted", rc == FT_TT_UDRV_OK,
                         "%s: request 0x%08x -> response 0x%08x", label, exchange.request_header,
                         exchange.response_header);

    /* The firmware ramps; sample after it has settled rather than racing it. */
    sleep_seconds(2u);
    sample_telemetry(label, out_row);
    print_telemetry_row(out_row);
    return rc == FT_TT_UDRV_OK;
}

static bool stage_power_setting(void)
{
    telemetry_row baseline, at_max, at_min, at_max_again;
    bool ok = true;

    ft_tt_test_step("Rung 6 — POWER_SETTING, and the AI CLOCK as the independent witness. "
                    "tt-kmd is asking for the MINIMUM throughout; our message has to win");

    sample_telemetry("baseline (tt-kmd's 0x7FFE)", &baseline);
    print_telemetry_row(&baseline);
    ft_tt_test_check_fmt("the telemetry row is LIVE, not a cache", baseline.sysfs_ok &&
                                                                      baseline.heartbeat_sysfs > 0,
                         "heartbeat %ld", baseline.heartbeat_sysfs);
    ft_tt_test_check_fmt("our own NoC read of the ARC telemetry table agrees with tt-kmd's",
                         baseline.ours_ok &&
                             baseline.aiclk_ours == (uint32_t)baseline.aiclk_sysfs,
                         "ours %u MHz, tt-kmd %ld MHz", baseline.aiclk_ours, baseline.aiclk_sysfs);
    ft_tt_test_check_fmt("baseline is the MINIMUM clock, because the driver's legacy open asked "
                         "for it",
                         baseline.aiclk_sysfs == (long)AICLK_MIN_MHZ, "%ld MHz",
                         baseline.aiclk_sysfs);

    ok &= request_clock(true, "ours: MAX (0x7FFF)", &at_max);
    ft_tt_test_check_fmt("THE FIRMWARE ACTED ON OUR MESSAGE — the AI clock rose", at_max.aiclk_sysfs >
                                                                                     baseline.aiclk_sysfs,
                         "%ld -> %ld MHz", baseline.aiclk_sysfs, at_max.aiclk_sysfs);
    ft_tt_test_check_fmt("and it rose to this part's maximum", at_max.aiclk_sysfs ==
                                                                   (long)AICLK_MAX_MHZ,
                         "%ld MHz", at_max.aiclk_sysfs);
    ft_tt_test_check_fmt("Vcore moved with it — a second, unrelated sensor", at_max.vcore_millivolts >
                                                                                baseline.vcore_millivolts,
                         "%ld -> %ld mV", baseline.vcore_millivolts, at_max.vcore_millivolts);
    ft_tt_test_check_fmt("board power moved with it — the clock tree is a chip-wide load you can "
                         "SEE",
                         at_max.power_microwatts > baseline.power_microwatts, "%.2f -> %.2f W",
                         (double)baseline.power_microwatts / 1e6,
                         (double)at_max.power_microwatts / 1e6);
    ft_tt_test_check_fmt("our own telemetry read sees the same clock", at_max.ours_ok &&
                                                                          at_max.aiclk_ours ==
                                                                              (uint32_t)at_max.aiclk_sysfs,
                         "ours %u MHz", at_max.aiclk_ours);

    /* THE CROSSED CONTROL. Same transport, same ring, same doorbell — one bit
     * of the header different. If the clock tracked "we sent a message" rather
     * than "we sent THIS message", this arm would stay at 1350. */
    ok &= request_clock(false, "ours: MIN (0x7FFE)", &at_min);
    ft_tt_test_check_fmt("CROSSED CONTROL: asking for tt-kmd's own 0x7FFE puts it back down, so "
                         "the clock tracks the CONTENT of the message",
                         at_min.aiclk_sysfs == baseline.aiclk_sysfs, "%ld MHz", at_min.aiclk_sysfs);
    ft_tt_test_check_fmt("and Vcore came back down with it", at_min.vcore_millivolts <
                                                                 at_max.vcore_millivolts,
                         "%ld -> %ld mV", at_max.vcore_millivolts, at_min.vcore_millivolts);

    ok &= request_clock(true, "ours: MAX again", &at_max_again);
    ft_tt_test_check_fmt("and it is repeatable, not a one-way latch", at_max_again.aiclk_sysfs ==
                                                                         at_max.aiclk_sysfs,
                         "%ld MHz", at_max_again.aiclk_sysfs);

    ft_tt_test_check_fmt("the heartbeat advanced across the whole arm — the instrument row was "
                         "alive the entire time",
                         at_max_again.heartbeat_sysfs > baseline.heartbeat_sysfs, "%ld -> %ld",
                         baseline.heartbeat_sysfs, at_max_again.heartbeat_sysfs);
    return ok;
}

/* ------------------------------------------------------------------------ *
 * Rung 6b — the same reading under a REAL Tensix load
 * ------------------------------------------------------------------------ */

/**
 * A sampler in a SEPARATE PROCESS, so the row is read WHILE the card is busy and
 * not microseconds after it went idle again.
 *
 * It touches no card resource: sysfs telemetry only, which tt-kmd serves out of
 * its own private window. It inherits the chardev fd, which is a duplicate of
 * the same open file, so its exit cannot make tt-kmd recompute the power
 * aggregation — that only happens on the LAST close, and the parent still holds
 * one. It is killed by the parent, so its own exit path does nothing.
 */
static pid_t start_sampler(const char *output_path)
{
    pid_t child = fork();
    FILE *output;
    unsigned sample;

    if (child != 0)
        return child;

    output = fopen(output_path, "w");
    if (!output)
        _exit(1);

    for (sample = 0u; sample < 100000u; sample++) {
        struct timespec pause;

        (void)fprintf(output, "%ld %ld %ld %ld\n", read_class_attribute("tt_aiclk"),
                      read_hwmon_attribute("power1_input"), read_hwmon_attribute("curr1_input"),
                      read_hwmon_attribute("in0_input"));
        (void)fflush(output);
        pause.tv_sec = 0;
        pause.tv_nsec = 100000000L; /* 100 ms */
        (void)nanosleep(&pause, NULL);
    }
    (void)fclose(output);
    _exit(0);
}

/** Aggregate what the sampler saw. Returns the number of samples. */
static unsigned summarise_samples(const char *path, long *out_aiclk_min, long *out_aiclk_max,
                                  long *out_power_max, long *out_current_max)
{
    FILE *input = fopen(path, "r");
    long aiclk, power, current, vcore;
    unsigned count = 0u;

    *out_aiclk_min = -1;
    *out_aiclk_max = -1;
    *out_power_max = -1;
    *out_current_max = -1;

    if (!input)
        return 0u;

    while (fscanf(input, "%ld %ld %ld %ld", &aiclk, &power, &current, &vcore) == 4) {
        if (*out_aiclk_min < 0 || aiclk < *out_aiclk_min)
            *out_aiclk_min = aiclk;
        if (aiclk > *out_aiclk_max)
            *out_aiclk_max = aiclk;
        if (power > *out_power_max)
            *out_power_max = power;
        if (current > *out_current_max)
            *out_current_max = current;
        count++;
    }
    (void)fclose(input);
    return count;
}

/**
 * Boot five RISC-Vs on each claimed core, run the vector_add triad, compare the
 * result byte for byte, and report what the clock read WHILE that happened.
 *
 * Two things come out of it that the idle A/B cannot give. The clock reading is
 * taken under work rather than at rest — DEV_NOTES is explicit that those two
 * print the same number for different reasons. And the card is shown to still
 * COMPUTE CORRECTLY at a clock nothing but our own ARC message asked for.
 */
static bool run_load_arm(bool max_ai_clock, const char *label)
{
    char samples_path[128];
    telemetry_row row;
    pid_t sampler;
    unsigned passed = 0u;
    unsigned attempted = 0u;
    unsigned index;
    unsigned sample_count;
    long aiclk_min, aiclk_max, power_max, current_max;
    double started;

    if (!request_clock(max_ai_clock, label, &row))
        return false;

    (void)snprintf(samples_path, sizeof(samples_path), "/tmp/ft_tt_arc_load_%s.txt",
                   max_ai_clock ? "max" : "min");
    sampler = start_sampler(samples_path);
    started = seconds_now();

    for (index = 0u; index < LOAD_CORE_COUNT; index++) {
        ft_tt_core_status status;
        ft_tt_core_probe_result result;

        memset(&status, 0, sizeof(status));
        memset(&result, 0, sizeof(result));

        if (ft_tt_core_status_read(&session.noc, &session.grid, load_cores[index], &status) !=
            FT_TT_OK)
            continue;

        attempted++;
        (void)ft_tt_core_probe_run(&session.noc, &session.grid, &probe_context, &status,
                                   LOAD_REPEATS, false, &result);
        ft_tt_core_probe_print_line(&result);
        if (ft_tt_core_probe_passed(&result))
            passed++;
    }

    if (sampler > 0) {
        (void)kill(sampler, SIGTERM);
        (void)waitpid(sampler, NULL, 0);
    }

    sample_count = summarise_samples(samples_path, &aiclk_min, &aiclk_max, &power_max,
                                     &current_max);
    printf("    %s: %u cores in %.1f s, %u telemetry samples DURING the work — "
           "aiclk %ld..%ld MHz, peak %.2f W / %.2f A\n",
           label, attempted, seconds_now() - started, sample_count, aiclk_min, aiclk_max,
           (double)power_max / 1e6, (double)current_max / 1e3);

    ft_tt_test_check_fmt("every probed core computed a byte-exact result at this clock",
                         passed == attempted && attempted > 0u, "%u of %u", passed, attempted);
    ft_tt_test_check_fmt("the sampler really ran while the card was working", sample_count > 0u,
                         "%u samples", sample_count);
    ft_tt_test_check_fmt("and the clock held at the value our message asked for throughout",
                         sample_count > 0u && aiclk_min == aiclk_max &&
                             aiclk_min == (long)(max_ai_clock ? AICLK_MAX_MHZ : AICLK_MIN_MHZ),
                         "%ld..%ld MHz", aiclk_min, aiclk_max);
    return passed == attempted && attempted > 0u;
}

static bool stage_load(void)
{
    bool at_max, at_min;

    ft_tt_test_step("Rung 6b — THE SAME READING UNDER A REAL TENSIX LOAD. Five RISC-Vs booted per "
                    "core, a vector_add triad run and byte-compared, sampled by a second process "
                    "WHILE it happens");

    if (!session_open || !probe_context_open) {
        ft_tt_test_check("the load session is open", false, "not requested or failed to open");
        return false;
    }

    /* Both arms run whatever the first one concludes — the crossed control is
     * the point, and skipping it on a failure would hide the one result that
     * says the clock tracks our message and not the act of sending one. */
    at_max = run_load_arm(true, "under load: MAX (0x7FFF)");
    at_min = run_load_arm(false, "under load: MIN (0x7FFE)");
    return at_max && at_min;
}

/* ------------------------------------------------------------------------ *
 * Rung 7 — the watchdog: hold the device open, doing NOTHING, for 3x
 * ------------------------------------------------------------------------ */

static bool stage_idle_hold(void)
{
    ft_tt_udrv_link_status link_before, link_after;
    telemetry_row before, after;
    ft_tt_udrv_arc_msg message;
    ft_tt_udrv_arc_exchange exchange;
    ft_tt_udrv_arc_locate_report relocated;
    ft_tt_udrv_arc_queue queue_after;
    double started;
    int rc;

    ft_tt_test_step("Rung 7 — THE WATCHDOG. Arm it at 10 s, then do NOTHING for %u s (3.5x) and "
                    "ask whether the card reset underneath us",
                    IDLE_HOLD_SECONDS);

    (void)ft_tt_udrv_arc_msg_set_wdt_timeout(WATCHDOG_MS, &message);
    rc = ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &message, 0u, &exchange);
    ft_tt_test_check_fmt("the watchdog is armed EXPLICITLY, not inherited", rc == FT_TT_UDRV_OK,
                         "%u ms, response 0x%08x", WATCHDOG_MS, exchange.response_header);

    /* Hold at the MAX clock, which is a state NOTHING on this machine but our own
     * ARC message produces: tt-kmd is asking for the minimum from two open fds.
     * So "the clock is still 1350 afterwards" is a much stronger statement about
     * a reset not having happened than "it is still 800" would be. */
    (void)ft_tt_udrv_arc_power_request_build(true, &message);
    (void)ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &message, 0u, &exchange);
    sleep_seconds(2u);

    memset(&link_before, 0, sizeof(link_before));
    (void)ft_tt_udrv_link_status_read(card.sysfs_path, &link_before);
    sample_telemetry("before the hold", &before);
    print_telemetry_row(&before);
    printf("    link before: %s x%u\n", ft_tt_udrv_link_speed_string(link_before.speed_code),
           link_before.width);

    /* NOTHING happens here. No NoC access, no ioctl, no message. That is the
     * experiment: tt-kmd never pets this timer either, and if the timer watched
     * host liveness the card would reset in this window. */
    started = seconds_now();
    fflush(stdout);
    sleep_seconds(IDLE_HOLD_SECONDS);
    printf("    held idle for %.1f s with no host traffic of any kind\n", seconds_now() - started);

    memset(&link_after, 0, sizeof(link_after));
    (void)ft_tt_udrv_link_status_read(card.sysfs_path, &link_after);
    ft_tt_test_check_fmt("the LINK did not retrain — a card reset drops it to Gen1 and this is "
                         "still where it was",
                         link_after.speed_code == link_before.speed_code &&
                             link_after.width == link_before.width,
                         "%s x%u -> %s x%u", ft_tt_udrv_link_speed_string(link_before.speed_code),
                         link_before.width, ft_tt_udrv_link_speed_string(link_after.speed_code),
                         link_after.width);

    sample_telemetry("after the hold", &after);
    print_telemetry_row(&after);

    ft_tt_test_check_fmt("the firmware HEARTBEAT advanced — it did not restart, so the firmware "
                         "did not reboot",
                         after.heartbeat_sysfs > before.heartbeat_sysfs, "%ld -> %ld (+%ld)",
                         before.heartbeat_sysfs, after.heartbeat_sysfs,
                         after.heartbeat_sysfs - before.heartbeat_sysfs);

    /* The monotonicity predicate must be able to say no. Fed the same pair
     * backwards it has to fail, or it is not a check. */
    ft_tt_test_check("...and that predicate is not vacuous: the same pair in the other order is "
                     "REJECTED",
                     !(before.heartbeat_sysfs > after.heartbeat_sysfs), "");

    ft_tt_test_check_fmt("the AI clock we asked for is STILL in force — a reset would have lost "
                         "the request",
                         after.aiclk_sysfs == before.aiclk_sysfs, "%ld -> %ld MHz",
                         before.aiclk_sysfs, after.aiclk_sysfs);

    rc = ft_tt_udrv_arc_locate_queue(&arc_context, FT_TT_UDRV_ARC_READY_TIMEOUT_MS, &relocated,
                                     &queue_after);
    ft_tt_test_check_fmt("the message queue is still where it was", rc == FT_TT_UDRV_OK &&
                                                                        queue_after.base ==
                                                                            arc_queue.base,
                         "0x%08x", relocated.qcb_pointer);

    (void)ft_tt_udrv_arc_msg_test(&message);
    rc = ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &message, 0u, &exchange);
    ft_tt_test_check_fmt("and the ARC still answers a message — the card is alive and was NOT "
                         "reset by its own watchdog",
                         rc == FT_TT_UDRV_OK, "rc=%d, header 0x%08x", rc, exchange.response_header);
    return rc == FT_TT_UDRV_OK;
}

/* ------------------------------------------------------------------------ *
 * Release
 * ------------------------------------------------------------------------ */

static void stage_release(void)
{
    ft_tt_udrv_arc_msg message;
    ft_tt_udrv_arc_exchange exchange;
    ft_tt_udrv_link_status link;
    telemetry_row final_row;
    size_t index;

    ft_tt_test_step("Release — hand the card back in the state tt-kmd believes it is in");

    if (arc_queue_located) {
        /* Our last word to the firmware is tt-kmd's own open-time request, so
         * the driver's model of the card and the card agree again. Its last
         * close will drop it the rest of the way. */
        (void)ft_tt_udrv_arc_power_request_build(false, &message);
        (void)ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &message, 0u, &exchange);
        ft_tt_test_check_fmt("POWER_SETTING restored to tt-kmd's 0x7FFE", exchange.rc ==
                                                                             FT_TT_UDRV_OK,
                             "response 0x%08x", exchange.response_header);
        sleep_seconds(2u); /* let the firmware ramp down before the row is read */
    }

    if (probe_context_open) {
        ft_tt_core_probe_context_close(&probe_context);
        probe_context_open = false;
    }
    if (session_open) {
        ft_tt_hardware_session_close(&session);
        session_open = false;
    }

    sample_telemetry("released", &final_row);
    print_telemetry_row(&final_row);

    if (arc_path_open) {
        printf("    path: %llu aims, %llu aim failures, %llu reads, %llu writes, %llu triggers\n",
               (unsigned long long)ft_tt_udrv_noc_path_aim_count(&arc_path),
               (unsigned long long)ft_tt_udrv_noc_path_aim_failures(&arc_path),
               (unsigned long long)arc_context.reads, (unsigned long long)arc_context.writes,
               (unsigned long long)arc_context.triggers);
        ft_tt_test_check_fmt("no aim ever failed its readback", ft_tt_udrv_noc_path_aim_failures(
                                                                    &arc_path) == 0u,
                             "%llu",
                             (unsigned long long)ft_tt_udrv_noc_path_aim_failures(&arc_path));
        (void)ft_tt_udrv_noc_path_close(&arc_path);
        arc_path_open = false;
    }

    for (index = 0u; index < claimed_count; index++) {
        struct ft_tt_kmd_free_window_request request;

        memset(&request, 0, sizeof(request));
        request.in.id = claimed[index];
        (void)ioctl(chardev, FT_TT_KMD_IOCTL_FREE_WINDOW, &request);
    }
    claimed_count = 0u;

    if (driver_registers) {
        (void)munmap((void *)(uintptr_t)driver_registers, FT_TT_UDRV_TLB_REGS_LEN);
        driver_registers = NULL;
    }
    if (chardev >= 0) {
        (void)close(chardev);
        chardev = -1;
    }
    if (udrv_open) {
        (void)ft_tt_udrv_noc_device_close(&udrv);
        udrv_open = false;
    }

    memset(&link, 0, sizeof(link));
    if (ft_tt_udrv_link_status_read(card.sysfs_path, &link) == FT_TT_UDRV_OK)
        printf("    card link at release: %s x%u\n", ft_tt_udrv_link_speed_string(link.speed_code),
               link.width);
    memset(&link, 0, sizeof(link));
    if (ft_tt_udrv_link_status_read(BRIDGE_DIR, &link) == FT_TT_UDRV_OK)
        printf("    bridge link at release: %s x%u\n", ft_tt_udrv_link_speed_string(link.speed_code),
               link.width);
}

/* ------------------------------------------------------------------------ *
 * main
 * ------------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    bool writing = false;
    bool hold = false;
    int index;

    for (index = 1; index < argc; index++) {
        if (strcmp(argv[index], "write") == 0)
            writing = true;
        else if (strcmp(argv[index], "hold") == 0)
            hold = true;
        else if (strcmp(argv[index], "load") == 0)
            want_load = true;
    }

    printf("ARC PROBE SEQUENCE — driver step 6, on the p150a with tt-kmd as the oracle\n");
    printf("mode: %s%s%s\n\n",
           writing ? "WRITE (messages will be sent)" : "READ-ONLY (nothing sent)",
           want_load ? " + Tensix load" : "", hold ? " + idle hold" : "");

    if (!stage_open_the_card(writing))
        goto finish;

    if (!writing) {
        ft_tt_test_step("Stopping here — `write` is required before anything is sent to the ARC");
        goto finish;
    }

    if (!stage_locate_the_queue())
        goto finish;
    if (!stage_queue_is_the_drivers())
        goto finish;
    if (!stage_doorbell_is_load_bearing())
        goto finish;
    if (!stage_our_messages())
        goto finish;
    (void)stage_power_setting();
    if (want_load)
        (void)stage_load();
    if (hold)
        (void)stage_idle_hold();

finish:
    stage_release();
    return ft_tt_test_finish();
}
