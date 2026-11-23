/* SPDX-License-Identifier: GPL-2.0-only OR AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * slot2_test.c — Test mapping a second GK210 die at 0x400000000
 *
 * Programs die 2 (Card B, 07:00.0) at the second MMIO slot while
 * bar_swap holds die 0 (Card A, 03:00.0) at 0xC00000000.
 *
 * Both dies are on separate root ports with independent bridge chains,
 * so they shouldn't conflict.
 *
 * Usage:
 *   # First: acquire die 0 via bar_swap (from userspace)
 *   # Then:
 *   insmod slot2_test.ko die_id=2
 *   # Check dmesg for results
 *   rmmod slot2_test
 *
 * The module programs BARs and bridge windows, does a quick MMIO probe
 * to verify the die responds, then cleans up on rmmod.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/init.h>
#include <linux/io.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("David (Corey) Herrera");
MODULE_DESCRIPTION("Test second MMIO slot at 0x400000000");

#define GK210_VENDOR 0x10de
#define GK210_DEVICE 0x102d
#define LOG "slot2_test: "

/* Slot 2 addresses (0x400000000) */
#define SLOT2_BAR1_ADDR  0x400000000ULL  /* 16 GB, pref */
#define SLOT2_BAR0_ADDR  0xF6000000ULL   /* 16 MB, non-pref (after bar_swap's F3-F5) */
#define SLOT2_BAR3_ADDR  0xF7000000ULL   /* 32 MB, pref */

#define BAR0_SIZE  0x01000000ULL   /* 16 MB */
#define BAR1_SIZE  0x400000000ULL  /* 16 GB */
#define BAR3_SIZE  0x02000000ULL   /* 32 MB */

/* Bridge windows for slot 2 */
#define S2_PREF_BASE   0x400000000ULL
#define S2_PREF_LIMIT  0x7FFFFFFFFULL
#define S2_MEM_BASE    0xF6000000ULL
#define S2_MEM_LIMIT   0xF7FFFFFFULL

static int die_id = 2;
module_param(die_id, int, 0444);
MODULE_PARM_DESC(die_id, "Die to map at slot 2 (default: 2)");

#define MAX_CHAIN 4

static struct pci_dev *target_dev;
static struct pci_dev *bridge_chain[MAX_CHAIN];
static int n_bridges;
static u32 saved_bars[6];
static u16 saved_cmd;
static struct {
    u16 pref_base, pref_limit, mem_base, mem_limit, cmd;
    u32 pref_base_upper, pref_limit_upper;
} saved_bridge[MAX_CHAIN];

/* Sibling die's downstream bridge (need to disable to prevent PLX routing conflict) */
static struct pci_dev *sibling_downstream;
static u16 sibling_saved_cmd;
static u16 sibling_saved_mem_base, sibling_saved_mem_limit;
static u16 sibling_saved_pref_base, sibling_saved_pref_limit;
static u32 sibling_saved_pref_base_upper, sibling_saved_pref_limit_upper;

static void save_bridge(struct pci_dev *br, int idx)
{
    pci_read_config_word(br, PCI_PREF_MEMORY_BASE, &saved_bridge[idx].pref_base);
    pci_read_config_word(br, PCI_PREF_MEMORY_LIMIT, &saved_bridge[idx].pref_limit);
    pci_read_config_dword(br, PCI_PREF_BASE_UPPER32, &saved_bridge[idx].pref_base_upper);
    pci_read_config_dword(br, PCI_PREF_LIMIT_UPPER32, &saved_bridge[idx].pref_limit_upper);
    pci_read_config_word(br, PCI_MEMORY_BASE, &saved_bridge[idx].mem_base);
    pci_read_config_word(br, PCI_MEMORY_LIMIT, &saved_bridge[idx].mem_limit);
    pci_read_config_word(br, PCI_COMMAND, &saved_bridge[idx].cmd);
}

static void restore_bridge(struct pci_dev *br, int idx)
{
    u16 cmd = saved_bridge[idx].cmd;
    pci_write_config_word(br, PCI_COMMAND, cmd & ~(PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER));
    pci_write_config_word(br, PCI_PREF_MEMORY_BASE, saved_bridge[idx].pref_base);
    pci_write_config_word(br, PCI_PREF_MEMORY_LIMIT, saved_bridge[idx].pref_limit);
    pci_write_config_dword(br, PCI_PREF_BASE_UPPER32, saved_bridge[idx].pref_base_upper);
    pci_write_config_dword(br, PCI_PREF_LIMIT_UPPER32, saved_bridge[idx].pref_limit_upper);
    pci_write_config_word(br, PCI_MEMORY_BASE, saved_bridge[idx].mem_base);
    pci_write_config_word(br, PCI_MEMORY_LIMIT, saved_bridge[idx].mem_limit);
    pci_write_config_word(br, PCI_COMMAND, cmd);
}

static void program_bridge_windows(struct pci_dev *br)
{
    u16 cmd;

    /* Disable decode first */
    pci_read_config_word(br, PCI_COMMAND, &cmd);
    pci_write_config_word(br, PCI_COMMAND, cmd & ~(PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER | PCI_COMMAND_IO));

    /* Prefetchable window */
    pci_write_config_word(br, PCI_PREF_MEMORY_BASE,
                          (u16)((S2_PREF_BASE >> 16) & 0xFFF0) | 0x1);  /* 64-bit */
    pci_write_config_word(br, PCI_PREF_MEMORY_LIMIT,
                          (u16)((S2_PREF_LIMIT >> 16) & 0xFFF0) | 0x1);
    pci_write_config_dword(br, PCI_PREF_BASE_UPPER32, (u32)(S2_PREF_BASE >> 32));
    pci_write_config_dword(br, PCI_PREF_LIMIT_UPPER32, (u32)(S2_PREF_LIMIT >> 32));

    /* Non-prefetchable window */
    pci_write_config_word(br, PCI_MEMORY_BASE, (u16)((S2_MEM_BASE >> 16) & 0xFFF0));
    pci_write_config_word(br, PCI_MEMORY_LIMIT, (u16)((S2_MEM_LIMIT >> 16) & 0xFFF0));
}

static void enable_decode(struct pci_dev *dev)
{
    u16 cmd;
    pci_read_config_word(dev, PCI_COMMAND, &cmd);
    cmd |= PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER | PCI_COMMAND_IO;
    pci_write_config_word(dev, PCI_COMMAND, cmd);
}

static void disable_decode(struct pci_dev *dev)
{
    u16 cmd;
    pci_read_config_word(dev, PCI_COMMAND, &cmd);
    cmd &= ~(PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER | PCI_COMMAND_IO);
    pci_write_config_word(dev, PCI_COMMAND, cmd);
}

static int __init slot2_test_init(void)
{
    struct pci_dev *dev = NULL;
    struct pci_dev *br;
    void __iomem *bar0_map;
    u32 boot_0, pmc;
    int found = 0, i;

    pr_info(LOG "=== Slot 2 Test: mapping die %d at 0x%llX ===\n",
            die_id, SLOT2_BAR1_ADDR);

    /* Find the target die */
    while ((dev = pci_get_device(GK210_VENDOR, GK210_DEVICE, dev)) != NULL) {
        if (found == die_id) {
            target_dev = pci_dev_get(dev);
            break;
        }
        found++;
    }

    if (!target_dev) {
        pr_err(LOG "Die %d not found (only %d dies)\n", die_id, found);
        return -ENODEV;
    }
    pr_info(LOG "Target: %s\n", pci_name(target_dev));

    /* Walk bridge chain upward */
    n_bridges = 0;
    br = target_dev->bus->self;
    while (br && n_bridges < MAX_CHAIN) {
        bridge_chain[n_bridges] = pci_dev_get(br);
        save_bridge(br, n_bridges);
        pr_info(LOG "Bridge[%d]: %s\n", n_bridges, pci_name(br));
        n_bridges++;
        if (br->bus && br->bus->self)
            br = br->bus->self;
        else
            break;
    }

    /* Save die BAR state */
    for (i = 0; i < 6; i++)
        pci_read_config_dword(target_dev, PCI_BASE_ADDRESS_0 + i*4, &saved_bars[i]);
    pci_read_config_word(target_dev, PCI_COMMAND, &saved_cmd);

    /* Find and disable sibling downstream bridge (PLX routing conflict fix) */
    {
        int sib_id = (die_id % 2 == 0) ? die_id + 1 : die_id - 1;
        struct pci_dev *sib_dev = NULL;
        int sib_found = 0;

        while ((sib_dev = pci_get_device(GK210_VENDOR, GK210_DEVICE, sib_dev)) != NULL) {
            if (sib_found == sib_id) {
                /* Get sibling's downstream bridge (chain[0]) */
                if (sib_dev->bus && sib_dev->bus->self) {
                    sibling_downstream = pci_dev_get(sib_dev->bus->self);
                    pr_info(LOG "Sibling downstream bridge: %s\n",
                            pci_name(sibling_downstream));

                    /* Save sibling state */
                    pci_read_config_word(sibling_downstream, PCI_COMMAND, &sibling_saved_cmd);
                    pci_read_config_word(sibling_downstream, PCI_MEMORY_BASE, &sibling_saved_mem_base);
                    pci_read_config_word(sibling_downstream, PCI_MEMORY_LIMIT, &sibling_saved_mem_limit);
                    pci_read_config_word(sibling_downstream, PCI_PREF_MEMORY_BASE, &sibling_saved_pref_base);
                    pci_read_config_word(sibling_downstream, PCI_PREF_MEMORY_LIMIT, &sibling_saved_pref_limit);
                    pci_read_config_dword(sibling_downstream, PCI_PREF_BASE_UPPER32, &sibling_saved_pref_base_upper);
                    pci_read_config_dword(sibling_downstream, PCI_PREF_LIMIT_UPPER32, &sibling_saved_pref_limit_upper);

                    /* Disable decode and zero windows */
                    disable_decode(sibling_downstream);
                    pci_write_config_word(sibling_downstream, PCI_MEMORY_BASE, 0xFFFF);
                    pci_write_config_word(sibling_downstream, PCI_MEMORY_LIMIT, 0x0000);
                    pci_write_config_word(sibling_downstream, PCI_PREF_MEMORY_BASE, 0xFFFF);
                    pci_write_config_word(sibling_downstream, PCI_PREF_MEMORY_LIMIT, 0x0000);
                    pci_write_config_dword(sibling_downstream, PCI_PREF_BASE_UPPER32, 0);
                    pci_write_config_dword(sibling_downstream, PCI_PREF_LIMIT_UPPER32, 0);
                    pr_info(LOG "Sibling disabled\n");
                }
                pci_dev_put(sib_dev);
                break;
            }
            sib_found++;
        }
    }

    /* Disable decode on target chain (top-down) */
    pr_info(LOG "Disabling decode on bridge chain...\n");
    for (i = n_bridges - 1; i >= 0; i--)
        disable_decode(bridge_chain[i]);
    disable_decode(target_dev);

    /* Program bridge windows (top-down: root port first) */
    pr_info(LOG "Programming bridge windows...\n");
    for (i = n_bridges - 1; i >= 0; i--) {
        program_bridge_windows(bridge_chain[i]);
        pr_info(LOG "  %s: pref=0x%llX-0x%llX mem=0x%llX-0x%llX\n",
                pci_name(bridge_chain[i]),
                S2_PREF_BASE, S2_PREF_LIMIT,
                S2_MEM_BASE, S2_MEM_LIMIT);
    }

    /* Program die BARs */
    pr_info(LOG "Programming die BARs...\n");

    /* BAR0: 32-bit non-pref at 0xF6000000 */
    pci_write_config_dword(target_dev, PCI_BASE_ADDRESS_0, (u32)SLOT2_BAR0_ADDR);
    target_dev->resource[0].start = SLOT2_BAR0_ADDR;
    target_dev->resource[0].end = SLOT2_BAR0_ADDR + BAR0_SIZE - 1;
    target_dev->resource[0].flags = IORESOURCE_MEM;

    /* BAR1: 64-bit pref at 0x400000000 (low dword at 0x14, high at 0x18) */
    pci_write_config_dword(target_dev, PCI_BASE_ADDRESS_1,
                           (u32)(SLOT2_BAR1_ADDR) | PCI_BASE_ADDRESS_MEM_PREFETCH | PCI_BASE_ADDRESS_MEM_TYPE_64);
    pci_write_config_dword(target_dev, PCI_BASE_ADDRESS_2, (u32)(SLOT2_BAR1_ADDR >> 32));
    target_dev->resource[1].start = SLOT2_BAR1_ADDR;
    target_dev->resource[1].end = SLOT2_BAR1_ADDR + BAR1_SIZE - 1;
    target_dev->resource[1].flags = IORESOURCE_MEM | IORESOURCE_PREFETCH | IORESOURCE_MEM_64;

    /* BAR3: 64-bit pref at 0xF7000000 (low dword at 0x1C, high at 0x20) */
    pci_write_config_dword(target_dev, PCI_BASE_ADDRESS_3,
                           (u32)(SLOT2_BAR3_ADDR) | PCI_BASE_ADDRESS_MEM_PREFETCH | PCI_BASE_ADDRESS_MEM_TYPE_64);
    pci_write_config_dword(target_dev, PCI_BASE_ADDRESS_4, (u32)(SLOT2_BAR3_ADDR >> 32));
    target_dev->resource[3].start = SLOT2_BAR3_ADDR;
    target_dev->resource[3].end = SLOT2_BAR3_ADDR + BAR3_SIZE - 1;
    target_dev->resource[3].flags = IORESOURCE_MEM | IORESOURCE_PREFETCH | IORESOURCE_MEM_64;

    pr_info(LOG "  BAR0=0x%llX BAR1=0x%llX BAR3=0x%llX\n",
            SLOT2_BAR0_ADDR, SLOT2_BAR1_ADDR, SLOT2_BAR3_ADDR);

    /* Enable decode (top-down) */
    pr_info(LOG "Enabling decode...\n");
    for (i = n_bridges - 1; i >= 0; i--)
        enable_decode(bridge_chain[i]);
    enable_decode(target_dev);

    /* Set power state to D0 */
    pci_set_power_state(target_dev, PCI_D0);
    pci_save_state(target_dev);

    /* Verify: try to read BAR0 registers */
    pr_info(LOG "Probing BAR0 at 0x%llX...\n", SLOT2_BAR0_ADDR);
    bar0_map = ioremap(SLOT2_BAR0_ADDR, BAR0_SIZE);
    if (!bar0_map) {
        pr_err(LOG "ioremap BAR0 FAILED\n");
        goto done;
    }

    boot_0 = ioread32(bar0_map + 0x0);     /* NV_PMC_BOOT_0 — GPU ID register */
    pmc = ioread32(bar0_map + 0x200);       /* NV_PMC_ENABLE */

    pr_info(LOG "  BOOT_0 (0x000): 0x%08X\n", boot_0);
    pr_info(LOG "  PMC_EN (0x200): 0x%08X\n", pmc);

    if (boot_0 == 0xFFFFFFFF) {
        pr_err(LOG "  >>> BAR0 READS ALL-F — die not responding at 0x400! <<<\n");
    } else if (boot_0 == 0x00000000) {
        pr_warn(LOG "  >>> BAR0 reads zero — die may need SBR first <<<\n");
    } else {
        pr_info(LOG "  >>> DIE RESPONDS AT 0x400000000! <<<\n");
        pr_info(LOG "  GPU ID: GK%X (revision 0x%X)\n",
                (boot_0 >> 20) & 0x1FF, boot_0 & 0xFF);
    }

    iounmap(bar0_map);

done:
    pr_info(LOG "Module loaded. rmmod to clean up.\n");
    return 0;
}

static void __exit slot2_test_exit(void)
{
    int i;

    pr_info(LOG "Cleaning up...\n");

    /* Disable decode */
    if (target_dev)
        disable_decode(target_dev);
    for (i = 0; i < n_bridges; i++)
        disable_decode(bridge_chain[i]);

    /* Restore die BARs */
    if (target_dev) {
        for (i = 0; i < 6; i++)
            pci_write_config_dword(target_dev, PCI_BASE_ADDRESS_0 + i*4, saved_bars[i]);
        pci_write_config_word(target_dev, PCI_COMMAND, saved_cmd);
        pci_dev_put(target_dev);
    }

    /* Restore bridge chain */
    for (i = 0; i < n_bridges; i++) {
        restore_bridge(bridge_chain[i], i);
        pci_dev_put(bridge_chain[i]);
    }

    /* Restore sibling */
    if (sibling_downstream) {
        pci_write_config_word(sibling_downstream, PCI_MEMORY_BASE, sibling_saved_mem_base);
        pci_write_config_word(sibling_downstream, PCI_MEMORY_LIMIT, sibling_saved_mem_limit);
        pci_write_config_word(sibling_downstream, PCI_PREF_MEMORY_BASE, sibling_saved_pref_base);
        pci_write_config_word(sibling_downstream, PCI_PREF_MEMORY_LIMIT, sibling_saved_pref_limit);
        pci_write_config_dword(sibling_downstream, PCI_PREF_BASE_UPPER32, sibling_saved_pref_base_upper);
        pci_write_config_dword(sibling_downstream, PCI_PREF_LIMIT_UPPER32, sibling_saved_pref_limit_upper);
        pci_write_config_word(sibling_downstream, PCI_COMMAND, sibling_saved_cmd);
        pci_dev_put(sibling_downstream);
    }

    pr_info(LOG "Cleaned up. Die %d released.\n", die_id);
}

module_init(slot2_test_init);
module_exit(slot2_test_exit);
