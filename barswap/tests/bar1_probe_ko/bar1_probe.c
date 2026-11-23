/* SPDX-License-Identifier: GPL-2.0-only OR AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * bar1_probe.c — Kernel module to probe GK210 BAR1 minimum size
 *
 * Uses pci_read/write_config_dword() which works even when ECAM/CF8
 * doesn't reach the device (bar_swap holds the pci_dev references).
 *
 * Usage:
 *   insmod bar1_probe.ko
 *   dmesg | grep bar1_probe
 *   rmmod bar1_probe
 *
 * Prerequisites: bar_swap.ko loaded, a die must be acquired first.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/init.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("David (Corey) Herrera");
MODULE_DESCRIPTION("GK210 BAR1 size probe");

#define GK210_VENDOR 0x10de
#define GK210_DEVICE 0x102d
#define LOG "bar1_probe: "

static int __init bar1_probe_init(void)
{
    struct pci_dev *dev = NULL;
    int die_idx = 0;

    pr_info(LOG "=== GK210 BAR1 Size Probe ===\n");

    /* Find all GK210 dies */
    while ((dev = pci_get_device(GK210_VENDOR, GK210_DEVICE, dev)) != NULL) {
        u32 bar_vals[6];
        u16 cmd_orig, cmd_disabled;
        u32 lo_orig, hi_orig, lo_mask, hi_mask;
        u64 full_mask, bar1_size;
        int i;

        pr_info(LOG "Die %d: %s\n", die_idx, pci_name(dev));

        /* Read current BARs */
        pr_info(LOG "  Current BARs:\n");
        for (i = 0; i < 6; i++) {
            pci_read_config_dword(dev, PCI_BASE_ADDRESS_0 + i*4, &bar_vals[i]);
            pr_info(LOG "    BAR%d (0x%02X): 0x%08X\n", i, 0x10 + i*4, bar_vals[i]);
        }

        /* Read CMD */
        pci_read_config_word(dev, PCI_COMMAND, &cmd_orig);
        pr_info(LOG "  CMD: 0x%04X\n", cmd_orig);

        /* Identify BAR layout:
         * GK210: BAR0 = 32-bit non-pref (16 MB registers)
         *        BAR1+BAR2 = 64-bit pref (VRAM)
         *        BAR3+BAR4 = 64-bit pref (NV control)
         *        BAR5 = I/O (typically)
         */

        /* Disable memory decode */
        cmd_disabled = cmd_orig & ~(PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER | PCI_COMMAND_IO);
        pci_write_config_word(dev, PCI_COMMAND, cmd_disabled);
        pr_info(LOG "  Decode disabled (0x%04X -> 0x%04X)\n", cmd_orig, cmd_disabled);

        /* Probe each BAR size */
        pr_info(LOG "  BAR size masks:\n");
        for (i = 0; i < 6; i++) {
            u32 orig, mask;
            int off = PCI_BASE_ADDRESS_0 + i * 4;

            pci_read_config_dword(dev, off, &orig);
            pci_write_config_dword(dev, off, 0xFFFFFFFF);
            pci_read_config_dword(dev, off, &mask);
            pci_write_config_dword(dev, off, orig);

            pr_info(LOG "    BAR%d: orig=0x%08X mask=0x%08X\n", i, orig, mask);
        }

        /* Now specifically probe BAR1 (64-bit): offsets 0x14 + 0x18 */
        pci_read_config_dword(dev, 0x14, &lo_orig);
        pci_read_config_dword(dev, 0x18, &hi_orig);

        pci_write_config_dword(dev, 0x14, 0xFFFFFFFF);
        pci_write_config_dword(dev, 0x18, 0xFFFFFFFF);
        pci_read_config_dword(dev, 0x14, &lo_mask);
        pci_read_config_dword(dev, 0x18, &hi_mask);
        pci_write_config_dword(dev, 0x14, lo_orig);
        pci_write_config_dword(dev, 0x18, hi_orig);

        pr_info(LOG "  BAR1 (64-bit VRAM) probe:\n");
        pr_info(LOG "    Low  orig=0x%08X mask=0x%08X\n", lo_orig, lo_mask);
        pr_info(LOG "    High orig=0x%08X mask=0x%08X\n", hi_orig, hi_mask);

        full_mask = ((u64)hi_mask << 32) | ((u64)(lo_mask & 0xFFFFFFF0));
        bar1_size = ~full_mask + 1;

        pr_info(LOG "    Full mask: 0x%016llX\n", full_mask);
        pr_info(LOG "    BAR1 size: 0x%llX (%llu GB)\n",
                bar1_size, bar1_size >> 30);

        if (bar1_size == 0x400000000ULL) {
            pr_info(LOG "    >>> HARDWIRED TO 16 GB — cannot resize <<<\n");
        } else if (bar1_size > 0 && bar1_size < 0x400000000ULL) {
            pr_info(LOG "    >>> MINIMUM SIZE = %llu GB — RESIZABLE! <<<\n",
                    bar1_size >> 30);
        } else {
            pr_info(LOG "    >>> UNEXPECTED SIZE — investigate <<<\n");
        }

        /* Also probe BAR0 (32-bit) and BAR3 (64-bit) for reference */
        {
            u32 b0_orig, b0_mask;
            pci_read_config_dword(dev, 0x10, &b0_orig);
            pci_write_config_dword(dev, 0x10, 0xFFFFFFFF);
            pci_read_config_dword(dev, 0x10, &b0_mask);
            pci_write_config_dword(dev, 0x10, b0_orig);

            u32 b0_size_mask = b0_mask & 0xFFFFFFF0;
            u32 b0_size = ~b0_size_mask + 1;
            pr_info(LOG "  BAR0 size: %u MB (mask=0x%08X)\n", b0_size >> 20, b0_mask);
        }

        {
            u32 b3lo_orig, b3hi_orig, b3lo_mask, b3hi_mask;
            pci_read_config_dword(dev, 0x1C, &b3lo_orig);
            pci_read_config_dword(dev, 0x20, &b3hi_orig);
            pci_write_config_dword(dev, 0x1C, 0xFFFFFFFF);
            pci_write_config_dword(dev, 0x20, 0xFFFFFFFF);
            pci_read_config_dword(dev, 0x1C, &b3lo_mask);
            pci_read_config_dword(dev, 0x20, &b3hi_mask);
            pci_write_config_dword(dev, 0x1C, b3lo_orig);
            pci_write_config_dword(dev, 0x20, b3hi_orig);

            u64 b3_full = ((u64)b3hi_mask << 32) | ((u64)(b3lo_mask & 0xFFFFFFF0));
            u64 b3_size = ~b3_full + 1;
            pr_info(LOG "  BAR3 size: %llu MB (mask_lo=0x%08X mask_hi=0x%08X)\n",
                    b3_size >> 20, b3lo_mask, b3hi_mask);
        }

        /* Restore CMD */
        pci_write_config_word(dev, PCI_COMMAND, cmd_orig);
        pr_info(LOG "  CMD restored to 0x%04X\n\n", cmd_orig);

        die_idx++;
    }

    if (die_idx == 0)
        pr_info(LOG "No GK210 devices found\n");
    else
        pr_info(LOG "Probed %d dies total\n", die_idx);

    /* Module doesn't need to stay loaded */
    return -ENODEV;  /* Force unload after printing results */
}

static void __exit bar1_probe_exit(void)
{
    pr_info(LOG "unloaded\n");
}

module_init(bar1_probe_init);
module_exit(bar1_probe_exit);
