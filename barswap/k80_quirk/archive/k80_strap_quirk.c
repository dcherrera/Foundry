// SPDX-License-Identifier: GPL-2.0-only OR AGPL-3.0-or-later
// Copyright (C) 2026 David (Corey) Herrera
// Research/reference code, not used in production builds (superseded by
// ../k80-strap-quirk.c). Portions modelled on quirk_s3_64M in the Linux kernel
// (drivers/pci/quirks.c, GPL-2.0, https://www.kernel.org/), modified.
// See NOTICE.
/*
 * k80_strap_quirk.c — strap every K80/GK210 BAR1 to 256 MB BEFORE the kernel
 * sizes its BARs, via a PCI early-fixup. This is the clean, every-boot version
 * of the runtime PSTRAPS resize: because the
 * strap is written in pci_fixup_early — which runs immediately before
 * pci_read_bases() probes the BAR sizes — the kernel sees 256 MB from the start,
 * so it allocates BAR1 small AND sizes the PLX bridge windows correctly. That is
 * what the post-boot userspace resize can't do (it can't resize a bridge window
 * after the fact), and it's why ~2 dies were left unmountable.
 *
 * The strap field on GK210 is PSTRAPS (BAR0 + 0x101000) bits [15:14] (value 2 =
 * 256 MB) with override-enable bit 31. We read BAR0 from config space (it is
 * assigned by boot before fixups run), ioremap PSTRAPS, and write the field.
 *
 * Usage: insmod, then re-enumerate the GPU cards so the fixup fires:
 *   for each GPU PLX upstream bridge:  echo 1 > .../remove ; echo 1 > /sys/bus/pci/rescan
 * (or build into initramfs for boot-time; loadable fixups apply on rescans.)
 */

#include <linux/io.h>
#include <linux/module.h>
#include <linux/pci.h>

#define GK210_VENDOR   0x10de
#define GK210_DEVICE   0x102d
#define PSTRAPS_OFFSET 0x101000u
#define BAR1_SHIFT     14
#define BAR1_MASK      0xfu
#define BAR1_VAL_256MB 2u
#define OVERRIDE_EN    (1u << 31)

static void k80_strap_fixup(struct pci_dev *dev)
{
	u32 bar0 = 0, v;
	void __iomem *base;

	pci_read_config_dword(dev, PCI_BASE_ADDRESS_0, &bar0);
	bar0 &= PCI_BASE_ADDRESS_MEM_MASK;          /* strip the low flag bits */
	if (!bar0) {
		dev_info(&dev->dev, "k80_strap: BAR0 unassigned, skipping\n");
		return;
	}
	base = ioremap(bar0 + PSTRAPS_OFFSET, 8);
	if (!base) {
		dev_info(&dev->dev, "k80_strap: ioremap(0x%x) failed\n",
			 bar0 + PSTRAPS_OFFSET);
		return;
	}
	v = readl(base);
	v = (v & ~(BAR1_MASK << BAR1_SHIFT)) | (BAR1_VAL_256MB << BAR1_SHIFT) | OVERRIDE_EN;
	writel(v, base);
	(void)readl(base);                          /* post-write read barrier */
	iounmap(base);
	dev_info(&dev->dev,
		 "k80_strap: BAR1 -> 256MB (PSTRAPS=0x%08x via BAR0 0x%x)\n",
		 v, bar0);
}
DECLARE_PCI_FIXUP_EARLY(GK210_VENDOR, GK210_DEVICE, k80_strap_fixup);

static int __init k80q_init(void)
{
	pr_info("k80_strap_quirk: loaded (pci_fixup_early for %04x:%04x)\n",
		GK210_VENDOR, GK210_DEVICE);
	return 0;
}

static void __exit k80q_exit(void)
{
	pr_info("k80_strap_quirk: unloaded\n");
}

module_init(k80q_init);
module_exit(k80q_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Strap K80/GK210 BAR1 to 256MB in pci_fixup_early (before BAR sizing)");
MODULE_AUTHOR("Nex-N2");
