// SPDX-License-Identifier: GPL-2.0-only OR AGPL-3.0-or-later
// Copyright (C) 2026 David (Corey) Herrera
// The BAR1 resource-shrink fixup is modelled on quirk_s3_64M in the Linux
// kernel (drivers/pci/quirks.c, GPL-2.0, https://www.kernel.org/), modified.
// See NOTICE.
/*
 * k80-strap-quirk.c — make all 16 GK210 dies of an 8x K80 machine map at 256 MB at
 * boot on the BTC-S37, fully in-kernel, with zero userspace. Two cooperating parts:
 *
 *  1. quirk_gk210_shrink_bar1 (DECLARE_PCI_FIXUP_HEADER)
 *     Shrinks the kernel's *view* of each die's BAR1 to 256 MB BEFORE bus
 *     resource assignment. Default BAR1 = 16 GB; 16 dies x 16 GB = 256 GB can't
 *     fit under this board's 36-bit (64 GB) ceiling, so the kernel's allocator
 *     fails to place it and DISABLES the bridge prefetchable window — which
 *     strands the last card ("card 8": dies 20/21 on root port 1c.7) and makes
 *     its dies stop decoding entirely. By pre-shrinking the resource to 256 MB
 *     the allocator never attempts the 16 GB placement, never disables any
 *     bridge, and assigns all 16 dies a 256 MB BAR1 with every bridge window
 *     alive. Modeled on quirk_s3_64M (drivers/pci/quirks.c).
 *
 *     Must be a HEADER fixup, NOT EARLY: pci_fixup_early runs before
 *     pci_read_bases() (BAR0/resource[] not populated -> the EARLY version
 *     reported "BAR0 unassigned, skipping"). pci_fixup_header runs after sizing
 *     but before assignment (verified against drivers/pci/probe.c).
 *
 *  2. k80_strap_silicon (late_initcall)
 *     The HEADER fixup only changes the kernel's VIEW; the silicon still decodes
 *     a 16 GB aperture. Because the two dies on a card share one bridge window,
 *     an unstrapped 16 GB die would alias over its sibling's region. So after PCI
 *     resources are assigned (when BAR0 is reachable), write each die's PSTRAPS
 *     (BAR0 + 0x101000, field [17:14] = 2 = 256 MB, override bit 31) so the
 *     silicon decode matches its 256 MB assignment. The HEADER fixup is what
 *     keeps every bridge alive so this can reach all 16 — card 8 included.
 *
 * Result: drop this kernel in, reboot, all 16 dies map at 256 MB. No barswap, no
 * runtime tool, no VBIOS flash, no userspace. (PSTRAPS mechanism proven on these
 * exact dies at runtime before being baked in.)
 */
#include <linux/pci.h>
#include <linux/io.h>
#include <linux/init.h>

#define GK210_DEVICE_ID    0x102d
#define GK210_BAR1_256MB   0x10000000UL		/* 256 MB */
#define GK210_PSTRAPS      0x101000u
#define GK210_BAR1_SHIFT   14
#define GK210_BAR1_MASK    (0xfu << GK210_BAR1_SHIFT)
#define GK210_BAR1_VAL_256 (2u << GK210_BAR1_SHIFT)	/* ladder: 2 = 256 MB */
#define GK210_OVERRIDE_EN  (1u << 31)

/* Part 1: shrink the kernel's BAR1 resource before assignment. */
static void quirk_gk210_shrink_bar1(struct pci_dev *dev)
{
	struct resource *r = &dev->resource[1];		/* BAR1: 64-bit prefetchable */

	if (!(r->flags & IORESOURCE_MEM))
		return;
	if (resource_size(r) <= GK210_BAR1_256MB)
		return;					/* already small (e.g. a rescan) */

	pci_info(dev, "k80_strap: BAR1 %lluMB -> 256MB (resource shrink before assignment)\n",
		 (unsigned long long)(resource_size(r) >> 20));

	r->flags |= IORESOURCE_UNSET;
	r->start = 0;
	r->end = GK210_BAR1_256MB - 1;
}
DECLARE_PCI_FIXUP_HEADER(PCI_VENDOR_ID_NVIDIA, GK210_DEVICE_ID,
			quirk_gk210_shrink_bar1);

/* Part 2: after assignment, strap the silicon BAR1 to 256 MB so the hardware
 * decode matches the kernel's 256 MB assignment (no intra-card aliasing). */
static int __init k80_strap_silicon(void)
{
	struct pci_dev *dev = NULL;

	while ((dev = pci_get_device(PCI_VENDOR_ID_NVIDIA, GK210_DEVICE_ID, dev))) {
		resource_size_t bar0 = pci_resource_start(dev, 0);
		void __iomem *base;
		u16 cmd;
		u32 v;

		if (!bar0)
			continue;

		/* BAR0 must decode for the PSTRAPS write to land. */
		pci_read_config_word(dev, PCI_COMMAND, &cmd);
		if (!(cmd & PCI_COMMAND_MEMORY))
			pci_write_config_word(dev, PCI_COMMAND,
					      cmd | PCI_COMMAND_MEMORY);

		base = ioremap(bar0 + GK210_PSTRAPS, 8);
		if (!base)
			continue;

		v = readl(base);
		if (v != 0xffffffffu) {		/* reachable */
			v = (v & ~GK210_BAR1_MASK) | GK210_BAR1_VAL_256 | GK210_OVERRIDE_EN;
			writel(v, base);
			(void)readl(base);	/* post-write read barrier */
			pci_info(dev, "k80_strap: silicon BAR1 -> 256MB (PSTRAPS=0x%08x)\n", v);
		} else {
			pci_info(dev, "k80_strap: silicon unreachable, skipping\n");
		}
		iounmap(base);
	}
	return 0;
}
late_initcall(k80_strap_silicon);
