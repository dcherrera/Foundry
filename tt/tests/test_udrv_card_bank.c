/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_card_bank.c
 * @brief ON THE CARD: prove that a window bank routes to the right endpoint, that a window
 *   pointed at the WRONG one is silent corruption, and that the audit catches it.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS AND WHY THE MODEL RUN COULD NOT SETTLE IT
 * ---------------------------------------------------------------------------
 * `ft_tt_udrv_noc_bank.c` replaced a one-entry window cache with an N-entry one, and the
 * failure it can introduce is the one this whole transport is built around: **a wrong window
 * is a wrong ADDRESS, not an error.** The mis-aim red arm was run against a full decode and
 * the answer was unreadable — the injection lands during ARC bring-up or core arming, the
 * card refuses for its own reasons, the run falls back to the host forward, and the host
 * forward produces the RIGHT text. A red arm whose consequence is "the run declined to use
 * the card" cannot distinguish a detector that works from one that never looked.
 *
 * So the proof is here instead, on two endpoints nobody else is using, with the bytes read
 * back and compared:
 *
 *   GREEN   write P to endpoint A and Q to endpoint B, read A, get P.
 *   RED     poison the bank's belief so a window aimed at B claims to be aimed at A, read A,
 *           **get Q** — the read went to B and nothing said so.
 *   CAUGHT  the same poisoned belief with `FT_TT_UDRV_AIM_AUDIT=1` REFUSES with EIO and
 *           names both endpoints, and nothing is dereferenced.
 *
 * IT USES DRAM, NOT L1, AND THAT IS DELIBERATE. Two DRAM banks on an idle card are two
 * endpoints this process can write without touching a Tensix that might be running something.
 * The mis-aimed read then lands in another DRAM bank, which is inert.
 *
 * @status None
 * @issues
 *   - Needs root: the sysfs resource0 file is 0600 root:root.
 *   - `write` is required for anything that aims a window, same rule as the sibling suites.
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_coord.h"
#include "ft_tt_hardware_session.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_claim.h"
#include "ft_tt_udrv_noc_bank.h"
#include "ft_tt_udrv_pci_sysfs.h"

/** Windows the bank under test claims. Two is the minimum that can hold two endpoints at
 *  once, which is the minimum that lets the red arm have somewhere wrong to point. */
#define BANK_WINDOWS 4u

/** Where in each DRAM bank the patterns go. High enough to be nowhere near anything a
 *  loader would put at the bottom, and 2 MiB aligned so one window covers it. */
#define TEST_DRAM_OFFSET 0x08000000ull

/** Bytes of pattern. Small: this is a correctness proof, not a bandwidth one. */
#define TEST_BYTES 256u

static unsigned checks;
static unsigned failures;

static void check(bool ok, const char *what)
{
    checks++;
    if (!ok)
        failures++;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static void fill(unsigned char *buffer, size_t bytes, unsigned salt)
{
    size_t i;

    for (i = 0; i < bytes; i++)
        buffer[i] = (unsigned char)((i * 31u + salt * 131u) & 0xFFu);
}

/**
 * One arm: open a bank with the given environment, write both patterns, read endpoint A back.
 *
 * The bank is opened FRESH each time because the red and audit flags are read at open. The
 * writes always go through the honest path — the red arm is read-only by construction, which
 * is what keeps a stray store out of somebody else's memory.
 */
static int run_arm(ft_tt_udrv_noc_device *device, const ft_tt_udrv_claim_namespace *ns,
                   uint32_t reserved, uint16_t ax, uint16_t ay, uint16_t bx, uint16_t by,
                   const unsigned char *pattern_a, const unsigned char *pattern_b,
                   unsigned char *out_read, uint64_t *out_misaims)
{
    ft_tt_udrv_noc_bank bank;
    int rc = ft_tt_udrv_noc_bank_open(device, ns, reserved, false, BANK_WINDOWS, "test", &bank);

    *out_misaims = 0u;
    memset(out_read, 0, TEST_BYTES);
    if (rc != FT_TT_UDRV_OK)
        return rc;

    rc = ft_tt_udrv_noc_bank_transfer(&bank, ax, ay, TEST_DRAM_OFFSET,
                                      (void *)(uintptr_t)pattern_a, TEST_BYTES, false,
                                      FT_TT_NOC_WIDTH_WORD32);
    if (rc == FT_TT_UDRV_OK)
        rc = ft_tt_udrv_noc_bank_transfer(&bank, bx, by, TEST_DRAM_OFFSET,
                                          (void *)(uintptr_t)pattern_b, TEST_BYTES, false,
                                          FT_TT_NOC_WIDTH_WORD32);
    /* THE READ IS THE MEASUREMENT. Both windows are now aimed — A on one, B on another — so a
     * red arm has somewhere wrong to send this and the bank has a hit to serve it from. */
    if (rc == FT_TT_UDRV_OK)
        rc = ft_tt_udrv_noc_bank_transfer(&bank, ax, ay, TEST_DRAM_OFFSET, out_read, TEST_BYTES,
                                          true, FT_TT_NOC_WIDTH_WORD32);
    *out_misaims = bank.misaims_injected;
    (void)ft_tt_udrv_noc_bank_close(&bank);
    return rc;
}

/**
 * THE `aimlow` ARM: one write through a bank whose every aim is perturbed in register word 0.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS A SEPARATE MODE AND NOT A FIFTH STAGE
 * ---------------------------------------------------------------------------
 * Both `FT_TT_UDRV_AIM_VERIFY` and `FT_TT_UDRV_AIM_RED` are parsed ONCE into a static, which
 * is deliberate — they are on the hottest path in the driver. So `full` and `word` cannot
 * both be exercised inside one process, and the two halves of this proof are two
 * invocations of this binary with two environments. The runner prints both verdicts side by
 * side; each alone says nothing.
 *
 *   FT_TT_UDRV_AIM_VERIFY=full  the readback disagrees with what was asked for and the aim
 *                               REFUSES with EIO. Nothing is dereferenced.
 *   FT_TT_UDRV_AIM_VERIFY=word  the readback AGREES, because word 1 does not carry a single
 *                               significant address bit (proven off hardware in
 *                               test_udrv_tlb_encode.c), and the write lands one 2 MiB page
 *                               away — on DRAM, which is why this is DRAM and not L1.
 *
 * @return 0 when the verdict matched what the caller declared it expected.
 */
static int run_aimlow(ft_tt_udrv_noc_device *device, const ft_tt_udrv_claim_namespace *ns,
                      uint32_t reserved, uint16_t ax, uint16_t ay,
                      const unsigned char *pattern_a, bool arm, bool expect_refusal)
{
    ft_tt_udrv_noc_bank bank;
    const char *mode = getenv("FT_TT_UDRV_AIM_VERIFY");
    uint64_t aim_failures;
    bool armed;
    int rc = ft_tt_udrv_noc_bank_open(device, ns, reserved, false, BANK_WINDOWS, "test", &bank);

    if (rc != FT_TT_UDRV_OK) {
        printf("  [FAIL] the bank would not open (rc=%d)\n", rc);
        return 1;
    }
    /* THE ARM GOES LIVE HERE AND NOWHERE EARLIER. The card is already open, so the bring-up's
     * own aims were honest; this perturbs exactly the transfer under test. An arm that could
     * not fire returns false, so it reads as a NUMBER and not as an absence. */
    armed = ft_tt_udrv_noc_aim_red_arm(arm);
    printf("  the word-0 arm is %s\n", armed ? "ARMED" : "not armed");
    if (arm != armed) {
        check(false, "the arm did not reach the state this invocation asked for");
        (void)ft_tt_udrv_noc_bank_close(&bank);
        return 1;
    }

    rc = ft_tt_udrv_noc_bank_transfer(&bank, ax, ay, TEST_DRAM_OFFSET,
                                      (void *)(uintptr_t)pattern_a, TEST_BYTES, false,
                                      FT_TT_NOC_WIDTH_WORD32);
    aim_failures = ft_tt_udrv_noc_bank_aim_failures(&bank);
    (void)ft_tt_udrv_noc_aim_red_arm(false);
    (void)ft_tt_udrv_noc_bank_close(&bank);

    printf("  FT_TT_UDRV_AIM_VERIFY=%s, word 0 %s: rc=%d, %llu aim failures\n",
           mode ? mode : "full(default)", armed ? "PERTURBED" : "untouched", rc,
           (unsigned long long)aim_failures);
    if (expect_refusal) {
        check(rc == FT_TT_UDRV_EIO, "the aim was REFUSED — a wrong address is caught here");
        check(aim_failures > 0u, "and the refusal is counted, not merely returned");
    } else {
        check(rc == FT_TT_UDRV_OK, "the aim was ACCEPTED");
        check(aim_failures == 0u, "nothing was counted as a failure");
    }
    return failures == 0u ? 0 : 1;
}

int main(int argc, char **argv)
{
    ft_tt_hardware_session session;
    ft_tt_udrv_pci_device found[4];
    ft_tt_udrv_noc_device device;
    ft_tt_udrv_claim_namespace ns;
    ft_tt_noc_coord bank_a, bank_b;
    unsigned char pattern_a[TEST_BYTES], pattern_b[TEST_BYTES], readback[TEST_BYTES];
    uint64_t misaims = 0u;
    size_t count = 0;
    uint32_t reserved;
    int rc;

    if (argc < 2 || strcmp(argv[1], "write") != 0) {
        printf("usage: %s write [clean-accept|aimlow-refuse|aimlow-accept]\n"
               "  (this aims windows and writes DRAM; root required)\n"
               "  the aimlow modes need FT_TT_UDRV_AIM_RED=aimlow plus its confirmation and\n"
               "  a FT_TT_UDRV_AIM_VERIFY setting; run them once each and read the pair\n",
               argv[0]);
        return 2;
    }

    /* The session is only used for the GRID — it names the DRAM endpoints. The bank under
     * test claims its own windows through the same flock namespace the runtime uses. */
    if (ft_tt_hardware_session_open(&session) != FT_TT_OK) {
        printf("cannot open the card through the product seam — set FT_TT_DRIVER=udrv\n");
        return 1;
    }
    if (ft_tt_coord_dram_bank(&session.grid, 0u, FT_TT_COORD_TRANSLATED, &bank_a) != FT_TT_OK ||
        ft_tt_coord_dram_bank(&session.grid, 1u, FT_TT_COORD_TRANSLATED, &bank_b) != FT_TT_OK) {
        printf("cannot derive two DRAM endpoints from the live grid\n");
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    printf("endpoint A = DRAM bank 0 at (%u,%u), endpoint B = DRAM bank 1 at (%u,%u)\n",
           (unsigned)bank_a.x, (unsigned)bank_a.y, (unsigned)bank_b.x, (unsigned)bank_b.y);

    if (ft_tt_udrv_pci_find_devices(NULL, FT_TT_UDRV_PCI_VENDOR_TENSTORRENT, 0xFFFFu, found, 4u,
                                    &count) != FT_TT_UDRV_OK ||
        count == 0u) {
        printf("no Tenstorrent device in sysfs\n");
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    if (ft_tt_udrv_noc_device_open(&found[0], true, &device) != FT_TT_UDRV_OK) {
        printf("cannot map BAR0 writable — this needs root\n");
        ft_tt_hardware_session_close(&session);
        return 1;
    }
    /* A SECOND namespace, so this test's windows can never collide with the ones the open
     * session above is already holding through the runtime. */
    if (ft_tt_udrv_claim_namespace_init(&ns, NULL, found[0].address, "tlb2m",
                                        FT_TT_UDRV_TLB_2M_WINDOW_COUNT) != FT_TT_UDRV_OK) {
        printf("cannot build the claim namespace\n");
        return 1;
    }
    reserved = 201u;

    fill(pattern_a, TEST_BYTES, 1u);
    fill(pattern_b, TEST_BYTES, 2u);

    /* THE `aimlow` MODE RUNS ALONE, because the verify knob is cached in a static and the
     * proof needs two settings of it. `refuse` and `accept` declare what this invocation
     * expects, so an invocation whose environment did not arm reads as a FAIL and not as a
     * silent pass. */
    if (argc >= 3 && (strcmp(argv[2], "aimlow-refuse") == 0 ||
                      strcmp(argv[2], "aimlow-accept") == 0 ||
                      strcmp(argv[2], "clean-accept") == 0)) {
        bool arm = strncmp(argv[2], "aimlow", 6) == 0;
        int one = run_aimlow(&device, &ns, reserved, (uint16_t)bank_a.x, (uint16_t)bank_a.y,
                             pattern_a, arm, strcmp(argv[2], "aimlow-refuse") == 0);

        (void)ft_tt_udrv_noc_device_close(&device);
        ft_tt_hardware_session_close(&session);
        printf("\n%u checks / %u failed\n", checks, failures);
        return one;
    }

    printf("\n1. GREEN — the bank routes to the endpoint it was asked for\n");
    unsetenv("FT_TT_UDRV_RED");
    unsetenv("FT_TT_UDRV_AIM_AUDIT");
    rc = run_arm(&device, &ns, reserved, (uint16_t)bank_a.x, (uint16_t)bank_a.y,
                 (uint16_t)bank_b.x, (uint16_t)bank_b.y, pattern_a, pattern_b, readback,
                 &misaims);
    check(rc == FT_TT_UDRV_OK, "the clean arm completed");
    check(memcmp(readback, pattern_a, TEST_BYTES) == 0, "reading A returned A's bytes");
    check(memcmp(readback, pattern_b, TEST_BYTES) != 0, "and they are NOT B's bytes");
    check(misaims == 0u, "nothing was injected");

    printf("\n2. GREEN — the audit refuses nothing on a correct bank\n");
    setenv("FT_TT_UDRV_AIM_AUDIT", "1", 1);
    rc = run_arm(&device, &ns, reserved, (uint16_t)bank_a.x, (uint16_t)bank_a.y,
                 (uint16_t)bank_b.x, (uint16_t)bank_b.y, pattern_a, pattern_b, readback,
                 &misaims);
    check(rc == FT_TT_UDRV_OK, "the audited arm completed");
    check(memcmp(readback, pattern_a, TEST_BYTES) == 0, "reading A still returned A's bytes");

    printf("\n3. RED — a window that LIES about where it points is silent corruption\n");
    unsetenv("FT_TT_UDRV_AIM_AUDIT");
    setenv("FT_TT_UDRV_RED", "misaim", 1);
    setenv("FT_TT_UDRV_RED_CONFIRM", "i-know-this-can-wedge-the-card", 1);
    rc = run_arm(&device, &ns, reserved, (uint16_t)bank_a.x, (uint16_t)bank_a.y,
                 (uint16_t)bank_b.x, (uint16_t)bank_b.y, pattern_a, pattern_b, readback,
                 &misaims);
    check(rc == FT_TT_UDRV_OK, "the arm returned SUCCESS — a wrong address is not an error");
    check(misaims > 0u, "the arm actually fired (a red arm that could not fire is a zero)");
    check(memcmp(readback, pattern_b, TEST_BYTES) == 0,
          "reading A returned B'S BYTES — the read went to the wrong endpoint");
    check(memcmp(readback, pattern_a, TEST_BYTES) != 0, "and NOT A's bytes");

    printf("\n4. CAUGHT — the same lie with the audit watching\n");
    setenv("FT_TT_UDRV_AIM_AUDIT", "1", 1);
    rc = run_arm(&device, &ns, reserved, (uint16_t)bank_a.x, (uint16_t)bank_a.y,
                 (uint16_t)bank_b.x, (uint16_t)bank_b.y, pattern_a, pattern_b, readback,
                 &misaims);
    check(rc == FT_TT_UDRV_EIO, "the transfer was REFUSED with EIO");
    check(misaims > 0u, "the arm fired here too");
    check(memcmp(readback, pattern_b, TEST_BYTES) != 0,
          "and nothing was dereferenced — the buffer never got B's bytes");

    unsetenv("FT_TT_UDRV_RED");
    unsetenv("FT_TT_UDRV_RED_CONFIRM");
    unsetenv("FT_TT_UDRV_AIM_AUDIT");
    (void)ft_tt_udrv_noc_device_close(&device);
    ft_tt_hardware_session_close(&session);

    printf("\n%u checks / %u failed\n", checks, failures);
    return failures == 0u ? 0 : 1;
}
