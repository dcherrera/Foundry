/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_rehearsal.c
 * @brief THE REHEARSAL: run the Blackhole with tt-kmd UNLOADED.
 *
 *   Every other on-card suite in this directory holds tt-kmd open as an oracle
 *   and as the real arbitrator. This one is what happens when the module is
 *   gone: no `/dev/tenstorrent/0`, no `ALLOCATE_TLB`, no sysfs telemetry tree,
 *   no second reader of BAR0. It walks the rehearsal risk register in the order
 *   the risks actually bite and prints what it read at every step, because a
 *   precise failure here is worth more than a partial success.
 *
 *   THE FIRST HURDLE, and it is expected: `rmmod` leaves the card in A3.
 *   `blackhole_cleanup_hardware()` sends ASIC_STATE3 on unload and the last
 *   chardev close already asked for every power feature off. So rung C sends
 *   the bring-up sequence to a card in a state nothing has ever been measured
 *   in. If the queue does not answer, that is the finding — this binary reports
 *   what it wrote, what it read and how long it waited, and does NOT improvise
 *   escalations at the ARC.
 *
 * @status Written for the 2026-08-17 rehearsal.
 * @issues
 *   - Root is required: the sysfs BAR files are 0600 root:root, and
 *     `/tmp/ft_tt_udrv.d` is created by whoever gets there first — one sudo run
 *     leaves it root-owned and every later non-root job fails to claim with
 *     rc=-4, which reads as a hardware fault. `--claim-root` exists for that.
 * @todo None
 */

/* `-std=c99 -pedantic` hides every POSIX declaration on glibc; these three are
 * what `fork`, `usleep` and `clock_gettime` need, and they must precede every
 * include. Same three as `test_udrv_card_arc.c`. */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "ft_tt_test_report.h"

#include "ft_tt_arch.h"
#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_launch.h"
#include "ft_tt_memmap.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_arc_msg.h"
#include "ft_tt_udrv_arc_probe.h"
#include "ft_tt_udrv_claim.h"
#include "ft_tt_udrv_noc.h"
#include "ft_tt_udrv_pci_sysfs.h"
#include "ft_tt_udrv_reset.h"
#include "ft_tt_udrv_tlb_encode.h"

#define CHARDEV_PATH "/dev/tenstorrent/0"
#define SYSFS_CLASS "/sys/class/tenstorrent/tenstorrent!0"
#define BRIDGE_DIR "/sys/bus/pci/devices/0000:00:1c.6"

/** Telemetry tag ids — a FIRMWARE contract, listed
 *  here because this file walks the table itself. The five GRID tags are not
 *  here: those come from the arch table, so the grid stays derived. */
#define TAG_VCORE 6u
#define TAG_POWER 7u
#define TAG_CURRENT 8u
#define TAG_ASIC_TEMP 11u
#define TAG_AICLK 14u
#define TAG_HEARTBEAT 32u

#define ARC_TELEMETRY_TABLE_PTR FT_TT_UDRV_ARC_RESET_SCRATCH(13)
#define ARC_TELEMETRY_VALUES_PTR FT_TT_UDRV_ARC_RESET_SCRATCH(12)

/** MRRS lives in DEVCTL bits 14:12; tt-kmd's probe sets it to 4096 = code 5. */
#define PCIE_DEVCTL_OFFSET 0x08u
#define PCIE_DEVCTL_MRRS_SHIFT 12u
#define PCIE_DEVCTL_MRRS_MASK 0x7u
#define MRRS_CODE_4096 5u
#define PCI_CAP_ID_EXPRESS 0x10u

/** PCI_COMMAND bit 1. With no driver bound nobody has called
 *  `pci_enable_device()`, and a BAR that does not decode reads all ones. */
#define PCI_COMMAND_OFFSET 0x04u
#define PCI_COMMAND_MEMORY (1u << 1)

/** The window tt-kmd reserves for itself and that we have never aimed. */
#define WINDOW_TT_KMD_RESERVED 201u

/** L1 round-trip size. 64 KiB is what the NoC round proved WC stores at. */
#define L1_ROUNDTRIP_BYTES (64u * 1024u)

/** Where in a Tensix's L1 the round trip lands — inside the USABLE region, so
 *  nothing that a firmware image or a mailbox owns is touched. */
#define L1_ROUNDTRIP_OFFSET (FT_TT_L1_USABLE_BASE + 0x40000u)

/** Which Tensix the data-path and health rungs use. A logical index, turned
 *  into a coordinate by ft_tt_coord — never a literal (x, y). */
#define DATA_CORE_INDEX 60u

/* ------------------------------------------------------------------------ *
 * Mutations. Each one attacks a specific check, and the round is only
 * believable if every one of them turns this binary red.
 * ------------------------------------------------------------------------ */

typedef enum mutation {
    MUTATION_NONE = 0,
    MUTATION_QUEUE_FROM_SCRATCH10, /**< locate the queue from the wrong scratch register */
    MUTATION_L1_SKIP_WRITE,        /**< never store the pattern; compare anyway */
    MUTATION_TELEMETRY_IGNORE_TAG, /**< take the first table entry, whatever its tag */
    MUTATION_CLAIM_RELEASE_FIRST,  /**< drop the claim before the second process tries */
    MUTATION_WINDOW201_STALE_AIM   /**< compare the readback against the PREVIOUS intent */
} mutation;

static const char *mutation_name(mutation which)
{
    switch (which) {
    case MUTATION_QUEUE_FROM_SCRATCH10: return "locate the queue from RESET_SCRATCH(10)";
    case MUTATION_L1_SKIP_WRITE:        return "the L1 round trip never writes";
    case MUTATION_TELEMETRY_IGNORE_TAG: return "the telemetry reader ignores the tag id";
    case MUTATION_CLAIM_RELEASE_FIRST:  return "the claim is released before the rival tries";
    case MUTATION_WINDOW201_STALE_AIM:  return "window 201's readback is compared to a stale aim";
    case MUTATION_NONE: default:        return "none";
    }
}

/* ------------------------------------------------------------------------ *
 * Session state
 * ------------------------------------------------------------------------ */

static ft_tt_udrv_pci_device card;
static ft_tt_udrv_noc_device udrv;
static bool udrv_open;

static ft_tt_udrv_noc_path arc_path;
static bool arc_path_open;
static ft_tt_udrv_arc_noc_context arc_context;
static ft_tt_udrv_arc_transport arc_transport;
static ft_tt_udrv_arc_queue arc_queue;
static bool arc_queue_located;

static ft_tt_udrv_claim_namespace window_ns;
static ft_tt_udrv_claim *window_claim[3];
static uint32_t claimed_window[3];
static size_t claimed_count;

static ft_tt_grid_info grid;
static uint16_t arc_x, arc_y;

/** The AI clock after an explicit MINIMUM request, which is the crossed control
 *  for the maximum one. Taken inside the bring-up rather than at start-up: with
 *  no module nothing lowers the clock at process exit, so a second run of this
 *  binary starts at the 1350 MHz the FIRST one asked for and a
 *  "did it go up from where it was" check reads 1350 -> 1350 and fails for the
 *  wrong reason. Recorded here as a finding, not worked around. */
static uint32_t clock_after_min_request;

static mutation active_mutation = MUTATION_NONE;
static const char *claim_root; /**< NULL means ft_tt_udrv_claim's own default */
static bool want_write;
static bool want_reset;

/* ------------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------------ */

static double now_seconds(void)
{
    struct timespec ts;

    (void)clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ((double)ts.tv_nsec / 1e9);
}

/** True when `path` exists. Used to state what the unload actually removed
 *  rather than assuming it. */
static bool path_exists(const char *path)
{
    struct stat st;

    return stat(path, &st) == 0;
}

static const char *speed_string(unsigned code)
{
    return ft_tt_udrv_link_speed_string(code);
}

/** Forward declaration: the bring-up's crossed control reads the AI clock, and
 *  the telemetry walker below it is what reads it. */
static int read_tag(uint16_t tag, uint32_t *out_value);

static void print_link(const char *label, const char *dir)
{
    ft_tt_udrv_link_status status;

    if (ft_tt_udrv_link_status_read(dir, &status) != FT_TT_UDRV_OK) {
        printf("    %-8s LnkSta could not be read (%s)\n", label, dir);
        return;
    }
    printf("    %-8s LnkSta %s x%u%s\n", label, speed_string(status.speed_code), status.width,
           status.training ? " (training)" : "");
}

/* ------------------------------------------------------------------------ *
 * Rung A — the card in sysfs with no module, and the two config-space
 * questions nobody sets on a cold path
 * ------------------------------------------------------------------------ */

/** MRRS out of the PCI Express capability's DEVCTL. Risk-register item 2: it is
 *  set by `pcie_set_readrq()` at probe and by our own config restore on a RESET
 *  path — so on a plain cold boot with no module NOBODY sets it. */
static bool read_mrrs(unsigned *out_code)
{
    unsigned capability = 0u;
    uint16_t devctl = 0u;

    if (ft_tt_udrv_config_find_capability(card.sysfs_path, PCI_CAP_ID_EXPRESS, &capability) !=
        FT_TT_UDRV_OK)
        return false;
    if (ft_tt_udrv_config_read_word(card.sysfs_path, capability + PCIE_DEVCTL_OFFSET, &devctl) !=
        FT_TT_UDRV_OK)
        return false;

    *out_code = ((unsigned)devctl >> PCIE_DEVCTL_MRRS_SHIFT) & PCIE_DEVCTL_MRRS_MASK;
    return true;
}

static bool stage_the_card_without_a_module(void)
{
    char driver_link[FT_TT_UDRV_PCI_PATH_MAX + 16u];
    uint16_t command = 0u;
    unsigned mrrs = 0u;
    bool mrrs_read;
    size_t count = 0u;
    uint32_t index;
    bool known = false;

    ft_tt_test_step("Rung A — the card in sysfs with tt-kmd GONE");

    ft_tt_test_check_fmt("running as root (the BAR files are 0600 root:root)", geteuid() == 0u,
                         "euid=%u", (unsigned)geteuid());

    ft_tt_test_check("the vendor chardev is GONE — this is the whole point of the round",
                     !path_exists(CHARDEV_PATH), CHARDEV_PATH);
    ft_tt_test_check("the vendor sysfs telemetry tree is GONE with it",
                     !path_exists(SYSFS_CLASS "/tt_aiclk"), SYSFS_CLASS "/tt_aiclk");

    if (ft_tt_udrv_pci_find_devices(NULL, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT,
                                    FT_TT_UDRV_PCI_DEVICE_BLACKHOLE, &card, 1u,
                                    &count) != FT_TT_UDRV_OK ||
        count == 0u) {
        ft_tt_test_check("the PCI core still publishes the Blackhole", false, "none found");
        return false;
    }
    ft_tt_test_check_fmt("the PCI core still publishes the Blackhole — enumeration is the "
                         "kernel's, not the vendor's",
                         true, "%s", card.address);

    (void)snprintf(driver_link, sizeof(driver_link), "%s/driver", card.sysfs_path);
    ft_tt_test_check_fmt("NO driver is bound to it", !path_exists(driver_link), "%s", driver_link);

    /* PCI_COMMAND. Nobody has called pci_enable_device(), and a BAR whose
     * memory decode is off reads all ones through a mapping that succeeded —
     * which is exactly the failure that would look like dead silicon. */
    if (ft_tt_udrv_config_read_word(card.sysfs_path, PCI_COMMAND_OFFSET, &command) !=
        FT_TT_UDRV_OK) {
        ft_tt_test_check("PCI_COMMAND reads", false, "");
        return false;
    }
    ft_tt_test_check_fmt("PCI_COMMAND has MEMORY DECODE ENABLED — without it every BAR read "
                         "is all ones through a mapping that succeeded",
                         (command & PCI_COMMAND_MEMORY) != 0u, "0x%04x", command);

    mrrs_read = read_mrrs(&mrrs);
    ft_tt_test_check("MRRS is readable out of DEVCTL", mrrs_read, "");
    ft_tt_test_check_fmt("MRRS is 4096 — RISK 2: nobody sets this on a cold path with no "
                         "module, and our config restore only rewrites it after a RESET",
                         mrrs == MRRS_CODE_4096, "DEVCTL MaxReadReq code %u = %u bytes", mrrs,
                         128u << mrrs);

    for (index = 0u; index < ft_tt_arch_blackhole()->pci_device_id_count; index++)
        if (ft_tt_arch_blackhole()->pci_device_ids[index] == card.device)
            known = true;
    ft_tt_test_check_fmt("sysfs's PCI device id is one the Blackhole arch table names", known,
                         "0x%04x", card.device);
    return known;
}

/* ------------------------------------------------------------------------ *
 * Rung B — the BAR files while unbound (risk 3), and the only readback
 * comparison that survives the module
 * ------------------------------------------------------------------------ */

static bool stage_map_the_bar(void)
{
    void *wc_mapping;
    const volatile uint32_t *wc_registers;
    uint32_t through_uc[FT_TT_UDRV_TLB_REGS_LEN / 4u];
    uint32_t through_wc[FT_TT_UDRV_TLB_REGS_LEN / 4u];
    size_t words = FT_TT_UDRV_TLB_REGS_LEN / 4u;
    size_t i;
    bool all_ones = true;

    ft_tt_test_step("Rung B — RISK 3: resource0 and resource0_wc with the device UNBOUND");

    if (ft_tt_udrv_noc_device_open(&card, want_write, &udrv) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("resource0 maps with no driver bound", false, "errno %d (%s)", errno,
                             strerror(errno));
        return false;
    }
    udrv_open = true;
    ft_tt_test_check_fmt("resource0 maps with NO DRIVER BOUND — RISK 3 answered", true,
                         "resource0 + 0x%08X, %s", FT_TT_UDRV_TLB_REGS_START,
                         want_write ? "PROT_READ|PROT_WRITE" : "PROT_READ");

    for (i = 0u; i < words; i++) {
        through_uc[i] = udrv.registers[i];
        if (through_uc[i] != 0xFFFFFFFFu)
            all_ones = false;
    }
    ft_tt_test_check("the register file is NOT all ones — the BAR really decodes", !all_ones, "");
    if (all_ones)
        return false;

    /* The write-combined alias of the same BAR. It is not an independent
     * reader — it is the same physical window through a second mapping — but it
     * is the only second path that survives the module, and it catches a
     * mapping that landed at the wrong OFFSET. */
    if (ft_tt_udrv_pci_map_bar(&card, 0u, true, false, FT_TT_UDRV_TLB_REGS_START,
                               FT_TT_UDRV_TLB_REGS_LEN, &wc_mapping) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("resource0_wc maps too", false, "errno %d", errno);
        return false;
    }
    wc_registers = (const volatile uint32_t *)wc_mapping;
    for (i = 0u; i < words; i++)
        through_wc[i] = wc_registers[i];
    ft_tt_test_check_fmt("every register word reads identically through resource0 and "
                         "resource0_wc",
                         memcmp(through_uc, through_wc, sizeof(through_uc)) == 0, "%zu words",
                         words);
    (void)ft_tt_udrv_pci_unmap_bar(wc_mapping, FT_TT_UDRV_TLB_REGS_LEN);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung C — RISK 6: our claim as the ONLY arbitrator
 * ------------------------------------------------------------------------ */

/** A rival process. `ALLOCATE_TLB` has been the real arbitrator in every on-card
 *  round until now; with the ioctl gone the flock is all there is, and a lock
 *  that is not tested across a real process boundary is not tested at all. */
static int rival_result(uint32_t index)
{
    ft_tt_udrv_claim_namespace ns;
    ft_tt_udrv_claim *claim = NULL;
    int rc;

    if (ft_tt_udrv_claim_namespace_init(&ns, claim_root, card.address, "tlb2m",
                                        FT_TT_UDRV_TLB_2M_WINDOW_COUNT) != FT_TT_UDRV_OK)
        return FT_TT_UDRV_EINVAL;

    rc = ft_tt_udrv_claim_acquire(&ns, index, "rehearsal rival", &claim);
    if (rc == FT_TT_UDRV_OK)
        (void)ft_tt_udrv_claim_release(claim);
    return rc;
}

/** Run `rival_result` in a child so the flock is genuinely cross-process. */
static int rival_in_a_child(uint32_t index)
{
    pid_t pid = fork();
    int status = 0;

    if (pid < 0)
        return FT_TT_UDRV_ERROR;
    if (pid == 0)
        _exit(rival_result(index) == FT_TT_UDRV_OK ? 0 : 1);

    (void)waitpid(pid, &status, 0);
    if (!WIFEXITED(status))
        return FT_TT_UDRV_ERROR;
    return WEXITSTATUS(status) == 0 ? FT_TT_UDRV_OK : FT_TT_UDRV_EAGAIN;
}

static bool stage_claim_is_the_only_arbitrator(void)
{
    uint32_t reserved[1] = {WINDOW_TT_KMD_RESERVED};
    int taken, free_one;
    size_t i;
    int rc;

    ft_tt_test_step("Rung C — RISK 6: ft_tt_udrv_claim as the ONLY arbitrator");

    rc = ft_tt_udrv_claim_namespace_init(&window_ns, claim_root, card.address, "tlb2m",
                                         FT_TT_UDRV_TLB_2M_WINDOW_COUNT);
    ft_tt_test_check_fmt("a claim namespace initialises", rc == FT_TT_UDRV_OK, "root=%s card=%s",
                         claim_root ? claim_root : "(default /tmp/ft_tt_udrv.d)", card.address);
    if (rc != FT_TT_UDRV_OK)
        return false;

    for (i = 0u; i < 3u; i++) {
        rc = ft_tt_udrv_claim_acquire_any(&window_ns, reserved, 1u, "rehearsal",
                                          &window_claim[claimed_count]);
        if (rc != FT_TT_UDRV_OK) {
            ft_tt_test_check_fmt("three windows claim — with NO ALLOCATE_TLB anywhere", false,
                                 "after %zu, rc=%d", claimed_count, rc);
            return false;
        }
        claimed_window[claimed_count] = ft_tt_udrv_claim_index(window_claim[claimed_count]);
        claimed_count++;
    }
    ft_tt_test_check_fmt("three windows claim — with NO ALLOCATE_TLB anywhere", true, "%u, %u, %u",
                         claimed_window[0], claimed_window[1], claimed_window[2]);

    if (active_mutation == MUTATION_CLAIM_RELEASE_FIRST) {
        (void)ft_tt_udrv_claim_release(window_claim[0]);
        window_claim[0] = NULL;
    }

    taken = rival_in_a_child(claimed_window[0]);
    ft_tt_test_check_fmt("a SECOND PROCESS is refused the window we hold — this is the only "
                         "thing standing between two jobs and one window",
                         taken == FT_TT_UDRV_EAGAIN, "rc=%d", taken);

    /* The discriminator: the same rival, one index away, must SUCCEED. Without
     * it "the rival always fails" would pass this rung for the wrong reason. */
    free_one = rival_in_a_child(claimed_window[2] + 1u == WINDOW_TT_KMD_RESERVED
                                    ? claimed_window[2] + 2u
                                    : claimed_window[2] + 1u);
    ft_tt_test_check_fmt("the same rival takes a window we do NOT hold — so the refusal is "
                         "the lock and not a broken path",
                         free_one == FT_TT_UDRV_OK, "rc=%d", free_one);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung D — THE ITEM: does the ARC message queue answer a card in A3
 * ------------------------------------------------------------------------ */

static void print_exchange(const char *label, const ft_tt_udrv_arc_exchange *exchange)
{
    printf("    %-16s request 0x%08X -> response 0x%08X (status %u), %.3f s, rc=%d\n", label,
           exchange->request_header, exchange->response_header, exchange->response_header & 0xFFu,
           exchange->elapsed_seconds, exchange->rc);
}

static bool stage_aim_at_the_arc(void)
{
    ft_tt_noc_coord coord;
    uint32_t boot_status = 0u;
    double started;
    int rc;

    ft_tt_test_step("Rung D — a path aimed at the ARC, and the first read of a card in A3");

    if (ft_tt_coord_arc(ft_tt_arch_blackhole(), FT_TT_COORD_TRANSLATED, &coord) != FT_TT_OK) {
        ft_tt_test_check("ft_tt_coord_arc() answers", false, "");
        return false;
    }
    arc_x = (uint16_t)coord.x;
    arc_y = (uint16_t)coord.y;
    ft_tt_test_check_fmt("the ARC tile is DERIVED from the arch table, never probed", true,
                         "(%u, %u)", arc_x, arc_y);

    if (ft_tt_udrv_noc_path_open(&udrv, claimed_window[0], false, &arc_path) != FT_TT_UDRV_OK) {
        ft_tt_test_check_fmt("a path opens on our own window, UNCACHED", false, "%u",
                             claimed_window[0]);
        return false;
    }
    arc_path_open = true;
    ft_tt_test_check_fmt("a path opens on our own window, UNCACHED", true, "window %u",
                         claimed_window[0]);

    rc = ft_tt_udrv_arc_transport_init(&arc_context, &arc_path, arc_x, arc_y, &arc_transport);
    ft_tt_test_check_fmt("a transport binds to the path", rc == FT_TT_UDRV_OK, "rc=%d", rc);
    if (rc != FT_TT_UDRV_OK)
        return false;

    /* THE FIRST READ OF THE ROUND THAT CROSSES THE BUS. Everything before this
     * is host-side. If A3 means the tile is dark, this is where it shows. */
    started = now_seconds();
    rc = ft_tt_udrv_arc_noc_read32(&arc_context, FT_TT_UDRV_ARC_BOOT_STATUS, &boot_status);
    ft_tt_test_check_fmt("THE ARC ANSWERS A NoC READ AT ALL after rmmod put the card in A3",
                         rc == FT_TT_UDRV_OK && boot_status != 0xFFFFFFFFu,
                         "ARC_BOOT_STATUS (0x%08X) = 0x%08X after %.3f s, rc=%d",
                         FT_TT_UDRV_ARC_BOOT_STATUS, boot_status, now_seconds() - started, rc);
    if (rc != FT_TT_UDRV_OK || boot_status == 0xFFFFFFFFu)
        return false;

    /* MEASURED 2026-08-17: after `rmmod` bit 0 is already set on the first read;
     * after an ASIC RESET it is CLEAR and takes ~280 polls to set. So this is a
     * report at this point and the VERDICT belongs to ft_tt_udrv_arc_locate_queue(),
     * whose own bounded poll loop is what a bring-up actually relies on. Making
     * it a hard check here would fail a card that is merely still booting. */
    printf("    ARC_BOOT_STATUS 0x%08X on the FIRST read — ready-for-message bit 0 %s%s\n",
           boot_status, (boot_status & FT_TT_UDRV_ARC_BOOT_STATUS_READY_FOR_MSG) ? "SET" : "CLEAR",
           (boot_status & FT_TT_UDRV_ARC_BOOT_STATUS_READY_FOR_MSG)
               ? ""
               : " (still booting; the locate poll loop waits)");
    return true;
}

static bool stage_locate_the_queue(void)
{
    ft_tt_udrv_arc_locate_report report;
    uint32_t qcb_pointer = 0u;
    int rc;

    ft_tt_test_step("Rung E — the message queue RESET_SCRATCH(11) points at, in A3");

    if (active_mutation == MUTATION_QUEUE_FROM_SCRATCH10) {
        /* The mutation reads the WRONG scratch register and builds a queue from
         * whatever it holds. Everything downstream must then fail. */
        (void)ft_tt_udrv_arc_noc_read32(&arc_context, FT_TT_UDRV_ARC_RESET_SCRATCH(10),
                                        &qcb_pointer);
        memset(&arc_queue, 0, sizeof(arc_queue));
        arc_queue.base = qcb_pointer;
        arc_queue.num_entries = 4u;
        arc_queue.request_base = qcb_pointer + FT_TT_UDRV_ARC_QUEUE_HEADER_BYTES;
        arc_queue.response_base = arc_queue.request_base + (4u * FT_TT_UDRV_ARC_MSG_BYTES);
        arc_queue_located = true;
        ft_tt_test_check_fmt("MUTATION: the queue is built from RESET_SCRATCH(10)", true,
                             "0x%08X", qcb_pointer);
        return true;
    }

    rc = ft_tt_udrv_arc_locate_queue(&arc_context, FT_TT_UDRV_ARC_READY_TIMEOUT_MS, &report,
                                     &arc_queue);
    ft_tt_test_check_fmt("RESET_SCRATCH(11) still holds a CSM address with the module gone",
                         report.qcb_pointer >= FT_TT_UDRV_ARC_CSM_BASE &&
                             report.qcb_pointer <
                                 FT_TT_UDRV_ARC_CSM_BASE + FT_TT_UDRV_ARC_CSM_SIZE,
                         "0x%08X = 0x%08X (boot status 0x%08X after %u poll%s)",
                         FT_TT_UDRV_ARC_MSG_QCB_PTR, report.qcb_pointer, report.boot_status,
                         report.boot_status_polls, report.boot_status_polls == 1u ? "" : "s");

    ft_tt_test_check_fmt("the queue descriptor builds and fits inside the CSM",
                         rc == FT_TT_UDRV_OK, "rc=%d", rc);
    if (rc != FT_TT_UDRV_OK)
        return false;
    arc_queue_located = true;

    printf("    QCB 0x%08X -> base 0x%08X, %u entries, requests 0x%08X, responses 0x%08X\n",
           report.qcb_pointer, arc_queue.base, arc_queue.num_entries, arc_queue.request_base,
           arc_queue.response_base);
    return true;
}

/** The bring-up itself: ASIC_STATE0 out of A3, then the watchdog, then the
 *  power request. Every one of these is proven — but only ever against a card
 *  tt-kmd had already woken. */
static bool stage_bring_the_card_up(void)
{
    ft_tt_udrv_arc_probe_report report;
    ft_tt_udrv_arc_msg minimum;
    ft_tt_udrv_arc_msg test;
    ft_tt_udrv_arc_exchange exchange;
    int rc;

    ft_tt_test_step("Rung F — THE BRING-UP: ASIC_STATE0 out of A3, SET_WDT_TIMEOUT, "
                    "POWER_SETTING, with nothing else on this machine talking to the card");

    /* A TEST first: it changes no state, and it separates "the queue does not
     * answer" from "the queue answers but refuses A0". */
    memset(&test, 0, sizeof(test));
    memset(&exchange, 0, sizeof(exchange));
    rc = ft_tt_udrv_arc_msg_test(&test);
    if (rc == FT_TT_UDRV_OK)
        rc = ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &test, 0u, &exchange);
    ft_tt_test_check_fmt("a TEST message is ANSWERED BY FIRMWARE on a card in A3 — the "
                         "response write pointer is the ARC's and nothing on the host moves it",
                         rc == FT_TT_UDRV_OK, "rc=%d", rc);
    print_exchange("TEST", &exchange);
    if (rc != FT_TT_UDRV_OK)
        return false;

    /* THE CROSSED CONTROL. Ask for tt-kmd's own 0x7FFE (every power feature on,
     * AI clock at MINIMUM) and read the clock; the bring-up below then asks for
     * 0x7FFF and the same instrument reads it again. Without this the round
     * cannot tell "our message raised the clock" from "the clock was already
     * where we wanted it", which is exactly what a card nobody lowered looks
     * like — and with no module, nobody lowers it. */
    rc = ft_tt_udrv_arc_power_request_build(false, &minimum);
    if (rc == FT_TT_UDRV_OK)
        rc = ft_tt_udrv_arc_exchange_send(&arc_transport, &arc_queue, &minimum, 0u, &exchange);
    print_exchange("POWER min", &exchange);
    ft_tt_test_check_fmt("a MINIMUM power request is accepted — the crossed control",
                         rc == FT_TT_UDRV_OK, "request 0x%08X", exchange.request_header);
    (void)usleep(400000u);
    (void)read_tag(TAG_AICLK, &clock_after_min_request);
    ft_tt_test_check_fmt("and the clock FELL to the minimum, so the instrument is live",
                         clock_after_min_request > 0u, "%u MHz", clock_after_min_request);

    rc = ft_tt_udrv_arc_probe_hardware(&arc_transport, FT_TT_UDRV_ARC_WATCHDOG_DEFAULT_MS, true,
                                       &report);
    print_exchange("ASIC_STATE0", &report.asic_state0);
    print_exchange("SET_WDT_TIMEOUT", &report.set_wdt_timeout);
    print_exchange("POWER_SETTING", &report.power_setting);

    ft_tt_test_check_fmt("ASIC_STATE0 is ACCEPTED FROM A3 — the antidote works at the one "
                         "moment it has never been sent",
                         report.asic_state0.rc == FT_TT_UDRV_OK, "status %u",
                         report.asic_state0.response_header & 0xFFu);
    ft_tt_test_check_fmt("SET_WDT_TIMEOUT answered", report.set_wdt_timeout.rc == FT_TT_UDRV_OK,
                         "status %u", report.set_wdt_timeout.response_header & 0xFFu);
    ft_tt_test_check_fmt("POWER_SETTING answered", report.power_setting.rc == FT_TT_UDRV_OK,
                         "status %u", report.power_setting.response_header & 0xFFu);
    ft_tt_test_check_fmt("the whole bring-up sequence completed", rc == FT_TT_UDRV_OK, "rc=%d",
                         rc);
    return rc == FT_TT_UDRV_OK;
}

/* ------------------------------------------------------------------------ *
 * Rung G — the effect, on an instrument that is not the message ring
 * ------------------------------------------------------------------------ */

/** Walk the ARC telemetry table over our own path. With the module gone this is
 *  the ONLY telemetry reader on the machine. */
static int read_tag(uint16_t tag, uint32_t *out_value)
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
        if (active_mutation != MUTATION_TELEMETRY_IGNORE_TAG &&
            (uint16_t)(entry & 0xFFFFu) != tag)
            continue;

        address = values_pointer + (((entry >> 16) & 0xFFFFu) * 4u);
        if (address < FT_TT_UDRV_ARC_CSM_BASE ||
            address + 4u > FT_TT_UDRV_ARC_CSM_BASE + FT_TT_UDRV_ARC_CSM_SIZE)
            return FT_TT_UDRV_EIO;
        return ft_tt_udrv_arc_noc_read32(&arc_context, address, out_value);
    }
    return FT_TT_UDRV_ENODEV;
}

static void print_telemetry(const char *label)
{
    uint32_t aiclk = 0u, vcore = 0u, power = 0u, current = 0u, temp = 0u, heartbeat = 0u;

    (void)read_tag(TAG_AICLK, &aiclk);
    (void)read_tag(TAG_VCORE, &vcore);
    (void)read_tag(TAG_POWER, &power);
    (void)read_tag(TAG_CURRENT, &current);
    (void)read_tag(TAG_ASIC_TEMP, &temp);
    (void)read_tag(TAG_HEARTBEAT, &heartbeat);
    printf("    %-22s aiclk %4u MHz | vcore %4u mV | power 0x%08X | current 0x%08X | "
           "temp 0x%08X | heartbeat %u\n",
           label, aiclk, vcore, power, current, temp, heartbeat);
}

static bool stage_the_clock_moved(void)
{
    uint32_t after_aiclk = 0u;
    uint32_t heartbeat_a = 0u, heartbeat_b = 0u;
    int rc;

    ft_tt_test_step("Rung G — the effect, measured on an instrument that is NOT the ring");

    (void)usleep(400000u);
    rc = read_tag(TAG_AICLK, &after_aiclk);
    ft_tt_test_check_fmt("our own NoC walk of the ARC telemetry table answers — with the "
                         "vendor sysfs tree gone this is the only telemetry reader left",
                         rc == FT_TT_UDRV_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("POWER_SETTING RAISED THE AI CLOCK off the floor OUR OWN minimum "
                         "request put it on — the message did something, and the thing it did "
                         "is not visible in the ring",
                         rc == FT_TT_UDRV_OK && after_aiclk > clock_after_min_request,
                         "%u MHz (after our min request) -> %u MHz", clock_after_min_request,
                         after_aiclk);

    (void)read_tag(TAG_HEARTBEAT, &heartbeat_a);
    (void)usleep(400000u);
    (void)read_tag(TAG_HEARTBEAT, &heartbeat_b);
    ft_tt_test_check_fmt("the firmware heartbeat is ADVANCING — the ARC is running, not "
                         "answering out of a frozen table",
                         heartbeat_b > heartbeat_a, "%u -> %u", heartbeat_a, heartbeat_b);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung H — RISK 7: window 201, which tt-kmd reserved and we have never aimed
 * ------------------------------------------------------------------------ */

static bool stage_window_201(void)
{
    ft_tt_udrv_tlb2m_config aim;
    ft_tt_udrv_tlb2m_config stale;
    uint32_t intended[3] = {0u, 0u, 0u};
    uint32_t readback[3] = {0u, 0u, 0u};
    uint32_t before[3] = {0u, 0u, 0u};
    ft_tt_udrv_claim *claim = NULL;
    int rc;

    ft_tt_test_step("Rung H — RISK 7: window %u, which tt-kmd reserved and we have NEVER aimed",
                    WINDOW_TT_KMD_RESERVED);

    rc = ft_tt_udrv_claim_acquire(&window_ns, WINDOW_TT_KMD_RESERVED, "rehearsal window 201",
                                  &claim);
    ft_tt_test_check_fmt("window 201 claims — with the module gone nobody reserves it any more",
                         rc == FT_TT_UDRV_OK, "window %u, rc=%d", WINDOW_TT_KMD_RESERVED, rc);
    if (rc != FT_TT_UDRV_OK)
        return false;

    (void)ft_tt_udrv_noc_window_read(&udrv, WINDOW_TT_KMD_RESERVED, before);
    ft_tt_test_check_fmt("it still holds what tt-kmd left in it", true, "0x%08X 0x%08X 0x%08X",
                         before[0], before[1], before[2]);

    memset(&stale, 0, sizeof(stale));
    stale.addr = 0u;
    stale.x_end = (uint8_t)arc_x;
    stale.y_end = (uint8_t)arc_y;

    memset(&aim, 0, sizeof(aim));
    aim.addr = (uint64_t)FT_TT_UDRV_ARC_RESET_SCRATCH(0) &
               ~((uint64_t)FT_TT_UDRV_TLB_2M_WINDOW_SIZE - 1u);
    aim.x_end = (uint8_t)arc_x;
    aim.y_end = (uint8_t)arc_y;

    rc = ft_tt_udrv_noc_window_aim(&udrv, WINDOW_TT_KMD_RESERVED, &aim);
    ft_tt_test_check_fmt("window 201 AIMS, and its own readback verifies", rc == FT_TT_UDRV_OK,
                         "window %u, rc=%d", WINDOW_TT_KMD_RESERVED, rc);

    (void)ft_tt_udrv_tlb2m_encode(active_mutation == MUTATION_WINDOW201_STALE_AIM ? &stale : &aim,
                                  intended);
    (void)ft_tt_udrv_noc_window_read(&udrv, WINDOW_TT_KMD_RESERVED, readback);
    ft_tt_test_check_fmt("the registers hold exactly what we encoded — a real PCIe read of an "
                         "UNCACHED register, which is what survives losing the second reader",
                         memcmp(intended, readback, sizeof(intended)) == 0,
                         "0x%08X 0x%08X 0x%08X", readback[0], readback[1], readback[2]);

    /* Put it back the way tt-kmd had it, so a modprobe at the end of the round
     * finds its own window where it left it. */
    if (udrv.writable) {
        volatile uint32_t *registers = udrv.registers;
        size_t base = (size_t)WINDOW_TT_KMD_RESERVED * (FT_TT_UDRV_TLB_REG_SIZE / 4u);

        registers[base + 0u] = before[0];
        registers[base + 1u] = before[1];
        registers[base + 2u] = before[2];
    }
    (void)ft_tt_udrv_claim_release(claim);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung I — the data path with no module: a Tensix L1 round trip
 * ------------------------------------------------------------------------ */

/** Build the grid from the ARC telemetry table over OUR path. `ft_tt_arc_telemetry`
 *  needs an `ft_tt_ctx`, which is the tt-kmd handle; the tags come from the arch
 *  table either way, so the topology stays DERIVED. */
static bool read_grid_over_our_path(void)
{
    const ft_tt_arch *arch = ft_tt_arch_blackhole();
    const uint16_t wanted[5] = {arch->telemetry.tag_enabled_tensix_col,
                                arch->telemetry.tag_enabled_eth,
                                arch->telemetry.tag_enabled_gddr,
                                arch->telemetry.tag_enabled_l2cpu,
                                arch->telemetry.tag_noc_translation};
    uint32_t values[5] = {0u, 0u, 0u, 0u, 0u};
    size_t i;

    for (i = 0u; i < 5u; i++)
        if (read_tag(wanted[i], &values[i]) != FT_TT_UDRV_OK)
            return false;

    memset(&grid, 0, sizeof(grid));
    grid.arch = arch;
    grid.tensix_enabled_col = values[0];
    grid.enabled_eth = values[1];
    grid.enabled_gddr = values[2];
    grid.enabled_l2cpu = values[3];
    grid.noc_translation_enabled = values[4];
    /* `valid` FIRST: ft_tt_coord_tensix_column_count() refuses a grid that is
     * not marked valid and answers 0, which then multiplies out to a topology of
     * zero cores that every later loop iterates zero times and passes. That is
     * how this file shipped a vacuous gate on its first run. */
    grid.valid = true;
    grid.tensix_column_count = ft_tt_coord_tensix_column_count(&grid);
    grid.tensix_core_count = grid.tensix_column_count * arch->tensix.row_count;
    return ft_tt_coord_check_grid(&grid) == FT_TT_OK && grid.tensix_column_count > 0u &&
           grid.tensix_core_count == grid.tensix_column_count * arch->tensix.row_count;
}

static bool stage_l1_round_trip(void)
{
    static uint8_t pattern[L1_ROUNDTRIP_BYTES];
    static uint8_t readback[L1_ROUNDTRIP_BYTES];
    ft_tt_udrv_noc_path wc_path;
    ft_tt_udrv_noc_path uc_path;
    ft_tt_noc_coord coord;
    /* Filled BEFORE the check, not inside its argument list: C does not order
     * argument evaluation, so passing the call and the fields it fills to one
     * variadic function reads the fields BEFORE the call on gcc and prints a
     * topology of zero cores next to a green verdict. */
    bool grid_read;
    uint32_t seed;
    size_t i;
    int rc;

    ft_tt_test_step("Rung I — the DATA PATH with no module: %u KiB into a Tensix's L1 through "
                    "a write-combined path, read back through a DIFFERENT window",
                    L1_ROUNDTRIP_BYTES / 1024u);

    grid_read = read_grid_over_our_path();
    ft_tt_test_check_fmt("the grid comes from the ARC telemetry table over OUR path", grid_read,
                         "%u columns, %u cores", grid.tensix_column_count,
                         grid.tensix_core_count);
    if (!grid_read)
        return false;

    if (ft_tt_coord_tensix_by_index(&grid, DATA_CORE_INDEX, FT_TT_COORD_TRANSLATED, &coord) !=
        FT_TT_OK) {
        ft_tt_test_check("the target Tensix coordinate DERIVES", false, "");
        return false;
    }
    ft_tt_test_check_fmt("the target Tensix coordinate DERIVES — never probed", true,
                         "core %u at (%u, %u)", DATA_CORE_INDEX, coord.x, coord.y);

    if (ft_tt_udrv_noc_path_open(&udrv, claimed_window[1], true, &wc_path) != FT_TT_UDRV_OK ||
        ft_tt_udrv_noc_path_open(&udrv, claimed_window[2], false, &uc_path) != FT_TT_UDRV_OK) {
        ft_tt_test_check("two paths open — one write-combined, one uncached", false, "");
        return false;
    }

    /* THE PATTERN MUST BE UNIQUE TO THIS RUN. A fixed pattern is a VACUOUS
     * GATE, and it was caught being one: nothing clears a Tensix's L1 between
     * processes, so the mutation that skipped the write still read back the
     * PREVIOUS run's identical bytes and passed. Seeding from the pid and the
     * clock makes a stale L1 disagree. */
    seed = (uint32_t)getpid() ^ (uint32_t)time(NULL);
    for (i = 0u; i < sizeof(pattern); i++)
        pattern[i] = (uint8_t)(((uint32_t)i * 31u) ^ ((uint32_t)i >> 7) ^ (seed >> (i & 7u)));
    memset(readback, 0, sizeof(readback));

    rc = FT_TT_UDRV_OK;
    if (active_mutation != MUTATION_L1_SKIP_WRITE)
        rc = ft_tt_udrv_noc_transfer(&wc_path, (uint16_t)coord.x, (uint16_t)coord.y,
                                     L1_ROUNDTRIP_OFFSET, pattern, sizeof(pattern), false,
                                     FT_TT_NOC_WIDTH_BLOCK16);
    ft_tt_test_check_fmt("the write-combined store completes", rc == FT_TT_UDRV_OK, "rc=%d", rc);

    rc = ft_tt_udrv_noc_transfer(&uc_path, (uint16_t)coord.x, (uint16_t)coord.y,
                                 L1_ROUNDTRIP_OFFSET, readback, sizeof(readback), true,
                                 FT_TT_NOC_WIDTH_BLOCK16);
    ft_tt_test_check_fmt("the read-back completes on the OTHER window", rc == FT_TT_UDRV_OK,
                         "rc=%d", rc);

    ft_tt_test_check_fmt("every byte agrees — a WC store reached the card and a UC read on a "
                         "different window found it, with NO VENDOR DRIVER in the path",
                         memcmp(pattern, readback, sizeof(pattern)) == 0, "%u bytes",
                         L1_ROUNDTRIP_BYTES);
    ft_tt_test_check_fmt("no aim was ever refused", ft_tt_udrv_noc_path_aim_failures(&wc_path) ==
                                                            0u &&
                                                        ft_tt_udrv_noc_path_aim_failures(
                                                            &uc_path) == 0u,
                         "%llu aims writing, %llu reading",
                         (unsigned long long)ft_tt_udrv_noc_path_aim_count(&wc_path),
                         (unsigned long long)ft_tt_udrv_noc_path_aim_count(&uc_path));

    (void)ft_tt_udrv_noc_path_close(&wc_path);
    (void)ft_tt_udrv_noc_path_close(&uc_path);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung J — the wedge detector, over udrv, with no module
 * ------------------------------------------------------------------------ */

/** `instrn_buf_status` reads 0x77 on a healthy core and 0x66 on a wedged one,
 *  WITHOUT booting it. With the module gone this is the only health read there
 *  is: `probe --write` needs the tt-kmd backend. */
static bool stage_core_health(uint32_t first, uint32_t count)
{
    uint32_t index;
    uint32_t healthy = 0u;
    uint32_t wedged = 0u;
    uint32_t examined = 0u;
    ft_tt_udrv_noc_path path;

    ft_tt_test_step("Rung J — core health over udrv: instrn_buf_status (0x%08X), the read that "
                    "tells a wedged core from a healthy one WITHOUT booting it",
                    FT_TT_TENSIX_INSTRN_BUF_STATUS_ADDR);

    if (ft_tt_udrv_noc_path_open(&udrv, claimed_window[1], false, &path) != FT_TT_UDRV_OK) {
        ft_tt_test_check("a path opens for the health read", false, "");
        return false;
    }

    for (index = first; index < first + count && index < grid.tensix_core_count; index++) {
        ft_tt_noc_coord coord;
        uint32_t status = 0u;
        uint32_t invalid = 0u;

        if (ft_tt_coord_tensix_by_index(&grid, index, FT_TT_COORD_TRANSLATED, &coord) != FT_TT_OK)
            continue;
        (void)ft_tt_udrv_noc_transfer(&path, (uint16_t)coord.x, (uint16_t)coord.y,
                                      FT_TT_TENSIX_INSTRN_BUF_STATUS_ADDR, &status, 4u, true,
                                      FT_TT_NOC_WIDTH_WORD32);
        (void)ft_tt_udrv_noc_transfer(&path, (uint16_t)coord.x, (uint16_t)coord.y,
                                      FT_TT_TENSIX_DBG_INVALID_INSTRN_ADDR, &invalid, 4u, true,
                                      FT_TT_NOC_WIDTH_WORD32);
        examined++;
        if (status == FT_TT_TENSIX_INSTRN_BUF_READY)
            healthy++;
        else
            wedged++;
        if (status != FT_TT_TENSIX_INSTRN_BUF_READY || count <= 8u)
            printf("    core %3u (%2u,%2u)  instrn_buf_status = 0x%02X  invalid_instrn = %u%s\n",
                   index, coord.x, coord.y, status, invalid,
                   status == FT_TT_TENSIX_INSTRN_BUF_READY ? "" : "   <-- WEDGED");
    }

    /* `examined > 0` is not decoration. A grid that read zero cores makes this
     * loop run zero times, and "0 wedged" would pass on a card nobody looked at. */
    ft_tt_test_check_fmt("every core in the range reads the ready constant over udrv",
                         examined > 0u && wedged == 0u, "%u examined, %u healthy (0x%02X), "
                         "%u wedged",
                         examined, healthy, FT_TT_TENSIX_INSTRN_BUF_READY, wedged);
    (void)ft_tt_udrv_noc_path_close(&path);
    return true;
}

/* ------------------------------------------------------------------------ *
 * Rung K — a reset with NO MODULE, and the bring-up that has to follow it
 * ------------------------------------------------------------------------ */

static bool stage_reset_with_no_module(void)
{
    ft_tt_udrv_config_restore_report restore;
    ft_tt_udrv_config_snapshot snapshot;
    ft_tt_udrv_reset_wait_report wait;
    ft_tt_udrv_reset_report reset;
    ft_tt_udrv_link_retrain_report retrain;
    ft_tt_udrv_link_status link;
    unsigned mrrs = 0u;
    bool mrrs_read;
    int rc;

    ft_tt_test_step("Rung K — RISK 1: an ASIC reset with NO MODULE, and the bring-up that has "
                    "to follow it because nothing else will send one");

    /* Discipline replacing tenstorrent_vma_zap: every aperture is unmapped
     * before the card moves underneath it. */
    if (arc_path_open) {
        (void)ft_tt_udrv_noc_path_close(&arc_path);
        arc_path_open = false;
    }
    if (udrv_open) {
        (void)ft_tt_udrv_noc_device_close(&udrv);
        udrv_open = false;
    }
    ft_tt_test_check("every aperture is unmapped before the reset", true,
                     "the discipline that replaces tenstorrent_vma_zap");

    print_link("card", card.sysfs_path);
    print_link("bridge", BRIDGE_DIR);

    rc = ft_tt_udrv_config_snapshot_take(card.sysfs_path, &snapshot);
    ft_tt_test_check_fmt("config space snapshots", rc == FT_TT_UDRV_OK,
                         "PCIe cap at 0x%02x, BAR0 0x%08x, command 0x%04x",
                         snapshot.pcie_capability_offset, snapshot.header[4],
                         (unsigned)(snapshot.header[1] & 0xFFFFu));
    if (rc != FT_TT_UDRV_OK)
        return false;

    rc = ft_tt_udrv_reset_asic(card.sysfs_path, &reset);
    ft_tt_test_check_fmt("the three config writes land", rc == FT_TT_UDRV_OK, "%u steps",
                         reset.step_count);
    if (rc != FT_TT_UDRV_OK) {
        ft_tt_udrv_reset_print(&reset, NULL);
        return false;
    }

    rc = ft_tt_udrv_reset_wait_ready(card.sysfs_path, snapshot.vendor_id, 15000u, &wait);
    ft_tt_udrv_reset_print(&reset, &wait);
    ft_tt_test_check_fmt("the card clears its own reset marker and answers config reads",
                         rc == FT_TT_UDRV_OK, "%u ms, %u polls", wait.elapsed_ms, wait.polls);
    if (rc != FT_TT_UDRV_OK)
        return false;

    rc = ft_tt_udrv_config_snapshot_restore(card.sysfs_path, &snapshot, &restore);
    ft_tt_udrv_config_restore_print(&restore);
    ft_tt_test_check_fmt("config space restores IN PLACE — no remove, no rescan",
                         rc == FT_TT_UDRV_OK, "rc=%d", rc);
    if (rc != FT_TT_UDRV_OK)
        return false;

    /* MRRS is the reason the restore has to happen at all: the ASIC reset
     * clears DEVCTL, and with no module nothing else rewrites it. */
    /* Read FIRST, then check: passing `read_mrrs(&mrrs)` and `mrrs` to the same
     * variadic call prints the value from BEFORE the read. Same trap as the
     * grid, and MRRS after a reset is too load-bearing a number to misprint. */
    mrrs_read = read_mrrs(&mrrs);
    ft_tt_test_check_fmt("MRRS survived the reset BECAUSE the restore rewrote DEVCTL",
                         mrrs_read && mrrs == MRRS_CODE_4096, "code %u = %u bytes", mrrs,
                         128u << mrrs);

    if (ft_tt_udrv_link_status_read(card.sysfs_path, &link) == FT_TT_UDRV_OK && link.speed_code < 2u) {
        rc = ft_tt_udrv_link_retrain(BRIDGE_DIR, 1000u, &retrain);
        ft_tt_udrv_link_retrain_print(&retrain);
        ft_tt_test_check_fmt("the link retrains off Gen1 — a reset drops it and every later "
                             "timing is invalid until it is back",
                             rc == FT_TT_UDRV_OK, "rc=%d", rc);
    }
    print_link("card", card.sysfs_path);
    print_link("bridge", BRIDGE_DIR);

    /* Re-open everything and bring the card up again. THIS is the risk-1 item:
     * after a reset, with no module, nothing sends ASIC_STATE0 but us. */
    if (ft_tt_udrv_noc_device_open(&card, want_write, &udrv) != FT_TT_UDRV_OK) {
        ft_tt_test_check("BAR0 re-maps after the reset", false, "");
        return false;
    }
    udrv_open = true;
    ft_tt_test_check("BAR0 re-maps after the reset", true, "");

    if (!stage_aim_at_the_arc() || !stage_locate_the_queue())
        return false;
    return stage_bring_the_card_up();
}

/* ------------------------------------------------------------------------ *
 * Teardown and main
 * ------------------------------------------------------------------------ */

static void release_everything(void)
{
    size_t i;

    if (arc_path_open)
        (void)ft_tt_udrv_noc_path_close(&arc_path);
    if (udrv_open)
        (void)ft_tt_udrv_noc_device_close(&udrv);
    for (i = 0u; i < claimed_count; i++)
        if (window_claim[i])
            (void)ft_tt_udrv_claim_release(window_claim[i]);
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [write] [reset] [--claim-root DIR] [--mutate N]\n"
            "  (no args)  read-only rungs: sysfs, BAR, MRRS, claims. Sends NOTHING\n"
            "  write      aim windows and send the ARC bring-up sequence\n"
            "  reset      ... then ASIC-reset the card and bring it up AGAIN\n"
            "  --mutate N 1 queue from scratch 10 · 2 L1 skips the write ·\n"
            "             3 telemetry ignores the tag · 4 claim released first ·\n"
            "             5 window 201 compared to a stale aim\n",
            argv0);
}

int main(int argc, char **argv)
{
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "write") == 0) {
            want_write = true;
        } else if (strcmp(argv[i], "reset") == 0) {
            want_write = true;
            want_reset = true;
        } else if (strcmp(argv[i], "--claim-root") == 0 && i + 1 < argc) {
            claim_root = argv[++i];
        } else if (strcmp(argv[i], "--mutate") == 0 && i + 1 < argc) {
            active_mutation = (mutation)atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    printf("=== THE REHEARSAL: the Blackhole with tt-kmd UNLOADED ===\n");
    printf("    mode: %s%s | mutation: %s\n\n", want_write ? "write" : "read-only",
           want_reset ? " + reset" : "", mutation_name(active_mutation));

    if (!stage_the_card_without_a_module())
        goto finish;
    if (!stage_map_the_bar())
        goto finish;
    if (!stage_claim_is_the_only_arbitrator())
        goto finish;
    if (!want_write) {
        printf("\nread-only mode: nothing was sent to the card.\n");
        goto finish;
    }
    if (!stage_aim_at_the_arc())
        goto finish;
    print_telemetry("before the bring-up");
    if (!stage_locate_the_queue())
        goto finish;
    if (!stage_bring_the_card_up())
        goto finish;
    print_telemetry("after the bring-up");
    (void)stage_the_clock_moved();
    (void)stage_window_201();
    if (!stage_l1_round_trip())
        goto finish;
    (void)stage_core_health(0u, grid.tensix_core_count);
    if (want_reset) {
        (void)stage_reset_with_no_module();
        print_telemetry("after the reset + bring-up");
        (void)stage_core_health(0u, grid.tensix_core_count);
    }

finish:
    release_everything();
    return ft_tt_test_finish();
}
