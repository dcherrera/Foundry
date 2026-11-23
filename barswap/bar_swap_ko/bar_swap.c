/* SPDX-License-Identifier: GPL-2.0-only OR AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * bar_swap.c — BAR Swap kernel module v2.0 for NVIDIA K80 GPUs
 *
 * Dynamic die multiplexer: any of the 6 K80 dies can be swapped into the
 * single working MMIO slot (0xC00000000) on demand. One die active at a time,
 * VRAM persists across swaps.
 *
 * Only 0xC00000000 works for BAR 1 on this Ivy Bridge system — 0x400 has a
 * NIC conflict and 0x800 causes nvidia RmInitAdapter timeout. The 36-bit
 * address space only has three 16-GB-aligned slots, two of which are broken.
 *
 * Operation:
 *   1. Discover all GK210 dies, build per-die bridge chains
 *   2. Release ALL existing GPU resources, disable ALL dies
 *   3. Register /dev/bar_swap — slot starts empty
 *   4. Userspace issues ACQUIRE(die_id) to program a die into the slot
 *   5. Userspace issues RELEASE(die_id) to vacate the slot
 *   6. On unload, restore all original bridge/BAR config space
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/delay.h>
#include <linux/mutex.h>

#include "bar_swap.h"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("David (Corey) Herrera");
MODULE_DESCRIPTION("BAR Swap v2 — dynamic die multiplexer for NVIDIA K80 GPUs");
MODULE_VERSION("2.0");

#define BS_LOG "bar_swap: "

/* NVIDIA GK210 (K80 die) */
#define GK210_VENDOR  0x10de
#define GK210_DEVICE  0x102d

/*
 * Address assignments — single MMIO slot
 *
 * Only one 16-GB-aligned address works: 0xC00000000.
 * Non-prefetchable BARs go at 0xF3-0xF5.
 */
#define SLOT_BAR1_ADDR   0xC00000000ULL   /* 16 GB, pref */
#define SLOT_BAR0_ADDR   0xF3000000ULL    /* 16 MB */
#define SLOT_BAR3_ADDR   0xF4000000ULL    /* 32 MB */

/* BAR sizes */
#define BAR0_SIZE  0x01000000ULL   /* 16 MB */
#define BAR1_SIZE  0x400000000ULL  /* 16 GB */
#define BAR3_SIZE  0x02000000ULL   /* 32 MB */

/* Bridge windows — same for all dies (single slot) */
#define BR_PREF_BASE   0xC00000000ULL
#define BR_PREF_LIMIT  0xFFFFFFFFFULL
#define BR_MEM_BASE    0xF3000000ULL
#define BR_MEM_LIMIT   0xF5FFFFFFULL

/* Max bridges per die chain (downstream, PLX upstream, root port, +1 spare) */
#define MAX_CHAIN  4

/* ---- State structures ---- */

struct bs_bridge_saved {
    u32 pref_base_upper;
    u32 pref_limit_upper;
    u16 pref_base;
    u16 pref_limit;
    u16 mem_base;
    u16 mem_limit;
    u16 cmd;
};

struct bs_die_state {
    struct pci_dev *dev;
    int die_id;
    int card_id;                                /* die_id / 2 */
    int slot;                                   /* 0 = active, -1 = unmapped */
    bool sbr_done;                              /* SBR issued at least once? */
    u32 saved_bars[6];                          /* original BAR config space */
    u16 saved_cmd;                              /* original Command register */
    int n_bridges;                              /* typically 3 */
    struct pci_dev *bridge_chain[MAX_CHAIN];    /* [0]=downstream, [1]=PLX up, [2]=root port */
    struct bs_bridge_saved bridge_saved[MAX_CHAIN]; /* original state for restore */
};

/* Module globals */
static struct bs_die_state dies[BS_MAX_DIES];
static int n_dies;
static int active_die = -1;                    /* -1 = slot empty */
static DEFINE_MUTEX(slot_mutex);

/* ---- PCI config space helpers ---- */

static void bs_save_die_bars(struct bs_die_state *die)
{
    int i;
    for (i = 0; i < 6; i++)
        pci_read_config_dword(die->dev, PCI_BASE_ADDRESS_0 + 4 * i, &die->saved_bars[i]);
    pci_read_config_word(die->dev, PCI_COMMAND, &die->saved_cmd);
}

static void bs_save_bridge(struct pci_dev *br, struct bs_bridge_saved *s)
{
    pci_read_config_word(br, PCI_PREF_MEMORY_BASE, &s->pref_base);
    pci_read_config_word(br, PCI_PREF_MEMORY_LIMIT, &s->pref_limit);
    pci_read_config_dword(br, PCI_PREF_BASE_UPPER32, &s->pref_base_upper);
    pci_read_config_dword(br, PCI_PREF_LIMIT_UPPER32, &s->pref_limit_upper);
    pci_read_config_word(br, PCI_MEMORY_BASE, &s->mem_base);
    pci_read_config_word(br, PCI_MEMORY_LIMIT, &s->mem_limit);
    pci_read_config_word(br, PCI_COMMAND, &s->cmd);
}

static void bs_restore_bridge(struct pci_dev *br, struct bs_bridge_saved *s)
{
    pci_write_config_word(br, PCI_COMMAND,
                          s->cmd & ~(PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER));
    pci_write_config_word(br, PCI_PREF_MEMORY_BASE, s->pref_base);
    pci_write_config_word(br, PCI_PREF_MEMORY_LIMIT, s->pref_limit);
    pci_write_config_dword(br, PCI_PREF_BASE_UPPER32, s->pref_base_upper);
    pci_write_config_dword(br, PCI_PREF_LIMIT_UPPER32, s->pref_limit_upper);
    pci_write_config_word(br, PCI_MEMORY_BASE, s->mem_base);
    pci_write_config_word(br, PCI_MEMORY_LIMIT, s->mem_limit);
    pci_write_config_word(br, PCI_COMMAND, s->cmd);
}

static void bs_restore_die_bars(struct bs_die_state *die)
{
    int i;
    pci_write_config_word(die->dev, PCI_COMMAND,
                          die->saved_cmd & ~(PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER));
    for (i = 0; i < 6; i++)
        pci_write_config_dword(die->dev, PCI_BASE_ADDRESS_0 + 4 * i, die->saved_bars[i]);
    pci_write_config_word(die->dev, PCI_COMMAND, die->saved_cmd);
}

static void bs_disable_decode(struct pci_dev *dev)
{
    u16 cmd;
    pci_read_config_word(dev, PCI_COMMAND, &cmd);
    cmd &= ~(PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER);
    pci_write_config_word(dev, PCI_COMMAND, cmd);
}

static void bs_enable_decode(struct pci_dev *dev)
{
    u16 cmd;
    pci_read_config_word(dev, PCI_COMMAND, &cmd);
    cmd |= PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER;
    pci_write_config_word(dev, PCI_COMMAND, cmd);
}

/* ---- Secondary Bus Reset ---- */

static void bs_secondary_bus_reset(struct pci_dev *bridge)
{
    u16 bridge_ctrl;

    pci_read_config_word(bridge, PCI_BRIDGE_CONTROL, &bridge_ctrl);
    pci_write_config_word(bridge, PCI_BRIDGE_CONTROL,
                          bridge_ctrl | PCI_BRIDGE_CTL_BUS_RESET);
    msleep(100);
    pci_write_config_word(bridge, PCI_BRIDGE_CONTROL, bridge_ctrl);
    msleep(1000);

    pr_info(BS_LOG "  SBR on bridge %s complete\n", pci_name(bridge));
}

/* ---- Bridge window programming ---- */

static void bs_program_bridge_pref(struct pci_dev *br, u64 base, u64 limit)
{
    u16 base_lo  = (u16)((base >> 16) & 0xFFF0) | 0x1;
    u16 limit_lo = (u16)((limit >> 16) & 0xFFF0) | 0x1;
    u32 base_hi  = (u32)(base >> 32);
    u32 limit_hi = (u32)(limit >> 32);

    pci_write_config_word(br, PCI_PREF_MEMORY_BASE, base_lo);
    pci_write_config_word(br, PCI_PREF_MEMORY_LIMIT, limit_lo);
    pci_write_config_dword(br, PCI_PREF_BASE_UPPER32, base_hi);
    pci_write_config_dword(br, PCI_PREF_LIMIT_UPPER32, limit_hi);

    pr_info(BS_LOG "  bridge %s pref: 0x%llx-0x%llx\n",
            pci_name(br), base, limit);
}

static void bs_program_bridge_mem(struct pci_dev *br, u32 base, u32 limit)
{
    u16 base_reg  = (u16)((base >> 16) & 0xFFF0);
    u16 limit_reg = (u16)((limit >> 16) & 0xFFF0);

    pci_write_config_word(br, PCI_MEMORY_BASE, base_reg);
    pci_write_config_word(br, PCI_MEMORY_LIMIT, limit_reg);

    pr_info(BS_LOG "  bridge %s mem:  0x%x-0x%x\n",
            pci_name(br), base, limit);
}

/* ---- BAR programming ---- */

static void bs_write_bar64(struct pci_dev *dev, int bar_idx, u64 addr)
{
    u32 lo = (u32)(addr & 0xFFFFFFFF);
    u32 hi = (u32)(addr >> 32);
    pci_write_config_dword(dev, PCI_BASE_ADDRESS_0 + 4 * bar_idx, lo);
    pci_write_config_dword(dev, PCI_BASE_ADDRESS_0 + 4 * (bar_idx + 1), hi);
}

static void bs_write_bar32(struct pci_dev *dev, int bar_idx, u32 addr)
{
    pci_write_config_dword(dev, PCI_BASE_ADDRESS_0 + 4 * bar_idx, addr);
}

static void bs_set_resource(struct pci_dev *dev, int bar_idx,
                            u64 start, u64 size, unsigned long flags)
{
    struct resource *res = &dev->resource[bar_idx];
    res->start = start;
    res->end   = start + size - 1;
    res->flags = flags;
}

static void bs_release_bar(struct pci_dev *dev, int bar_idx)
{
    struct resource *res = &dev->resource[bar_idx];
    if (res->parent)
        pci_release_resource(dev, bar_idx);
}

static void bs_set_bridge_resource(struct pci_dev *br, int win_idx,
                                   u64 start, u64 end, unsigned long flags)
{
    int res_idx = PCI_BRIDGE_RESOURCES + win_idx;
    struct resource *res = &br->resource[res_idx];
    res->start = start;
    res->end   = end;
    res->flags = flags;
}

static void bs_release_bridge_resource(struct pci_dev *br, int win_idx)
{
    int res_idx = PCI_BRIDGE_RESOURCES + win_idx;
    struct resource *res = &br->resource[res_idx];
    if (res->parent)
        pci_release_resource(br, res_idx);
}

/* ---- Zero out a die's BAR config space ---- */

static void bs_zero_die_bars(struct pci_dev *dev)
{
    int j;

    bs_disable_decode(dev);

    for (j = 0; j < 6; j++)
        pci_write_config_dword(dev, PCI_BASE_ADDRESS_0 + 4 * j, 0);

    for (j = 0; j < 6; j++) {
        struct resource *res = &dev->resource[j];
        res->start = 0;
        res->end   = 0;
        res->flags = 0;
    }
}

/* ---- Discovery ---- */

/*
 * Discover all GK210 dies and build per-die bridge chains.
 * Walk upward from each die: die->bus->self -> bus->self -> bus->self
 * Stores the full chain per die. No target/other distinction — all dies equal.
 */
static int bs_discover(void)
{
    struct pci_dev *dev = NULL;
    int i, j;

    n_dies = 0;

    /* Find all GK210 dies */
    while ((dev = pci_get_device(GK210_VENDOR, GK210_DEVICE, dev)) != NULL) {
        struct bs_die_state *d;
        struct pci_dev *br;

        if (n_dies >= BS_MAX_DIES) {
            pr_warn(BS_LOG "more than %d dies found, ignoring extras\n", BS_MAX_DIES);
            break;
        }

        d = &dies[n_dies];
        d->dev = pci_dev_get(dev);
        d->die_id = n_dies;
        d->card_id = n_dies / 2;
        d->slot = -1;
        d->sbr_done = false;
        d->n_bridges = 0;
        bs_save_die_bars(d);

        /* Prevent runtime PM from putting dies into D3cold.
         * PCH PCIe ports aggressively sleep devices after decode is
         * disabled, and SBR on a D3cold device crashes the system. */
        pci_d3cold_disable(dev);

        /* Walk upward to build bridge chain */
        br = dev->bus->self;
        while (br && d->n_bridges < MAX_CHAIN) {
            j = d->n_bridges;
            d->bridge_chain[j] = pci_dev_get(br);
            bs_save_bridge(br, &d->bridge_saved[j]);
            d->n_bridges++;

            if (br->bus && br->bus->self)
                br = br->bus->self;
            else
                break;
        }

        pr_info(BS_LOG "found die %d (card %d): %s (bus %02x), %d bridges\n",
                n_dies, d->card_id, pci_name(dev), dev->bus->number, d->n_bridges);
        for (i = 0; i < d->n_bridges; i++)
            pr_info(BS_LOG "  chain[%d]: %s\n", i, pci_name(d->bridge_chain[i]));

        n_dies++;
    }

    pr_info(BS_LOG "discovered %d dies across %d cards\n",
            n_dies, n_dies > 0 ? (dies[n_dies - 1].card_id + 1) : 0);

    return n_dies;
}

/* ---- Release all GPU-related iomem entries ---- */

static void bs_release_all_gpu_resources(void)
{
    int i, j;

    pr_info(BS_LOG "releasing all existing GPU resources...\n");

    /* Release all die BAR resources */
    for (i = 0; i < n_dies; i++)
        for (j = 0; j < 6; j++)
            bs_release_bar(dies[i].dev, j);

    /* Release all bridge window resources (dedup by checking parent) */
    for (i = 0; i < n_dies; i++) {
        struct bs_die_state *d = &dies[i];
        for (j = 0; j < d->n_bridges; j++) {
            struct pci_dev *br = d->bridge_chain[j];
            bs_release_bridge_resource(br, 0);  /* I/O */
            bs_release_bridge_resource(br, 1);  /* mem */
            bs_release_bridge_resource(br, 2);  /* pref */
        }
    }
}

/* ---- Disable all dies and zero all bridge windows ---- */

static void bs_zero_bridge_windows(struct pci_dev *br)
{
    /* Zero non-pref memory window */
    pci_write_config_word(br, PCI_MEMORY_BASE, 0xFFFF);
    pci_write_config_word(br, PCI_MEMORY_LIMIT, 0x0000);

    /* Zero prefetchable memory window (set base > limit to disable) */
    pci_write_config_word(br, PCI_PREF_MEMORY_BASE, 0xFFFF);
    pci_write_config_word(br, PCI_PREF_MEMORY_LIMIT, 0x0000);
    pci_write_config_dword(br, PCI_PREF_BASE_UPPER32, 0xFFFFFFFF);
    pci_write_config_dword(br, PCI_PREF_LIMIT_UPPER32, 0x00000000);
}

static void bs_disable_all_dies(void)
{
    int i;

    pr_info(BS_LOG "disabling all dies...\n");
    for (i = 0; i < n_dies; i++) {
        pr_info(BS_LOG "  disabling die %d: %s\n", i, pci_name(dies[i].dev));
        bs_zero_die_bars(dies[i].dev);
    }
}

/* ---- Acquire / Release core logic ---- */

/*
 * Release the currently active die from the MMIO slot.
 * Caller must hold slot_mutex.
 */
static void bs_do_release_locked(void)
{
    struct bs_die_state *d;
    int i;

    if (active_die < 0 || active_die >= n_dies)
        return;

    d = &dies[active_die];

    pr_info(BS_LOG "releasing die %d (%s)...\n", active_die, pci_name(d->dev));

    /* Disable decode on die */
    bs_disable_decode(d->dev);

    /* Disable decode on bridges (bottom-up) */
    for (i = 0; i < d->n_bridges; i++)
        bs_disable_decode(d->bridge_chain[i]);

    /* Release die BAR resources */
    bs_release_bar(d->dev, 0);
    bs_release_bar(d->dev, 1);
    bs_release_bar(d->dev, 3);

    /* Release bridge window resources (bottom-up) */
    for (i = 0; i < d->n_bridges; i++) {
        bs_release_bridge_resource(d->bridge_chain[i], 1);  /* mem */
        bs_release_bridge_resource(d->bridge_chain[i], 2);  /* pref */
    }

    /* Zero BAR config space */
    bs_zero_die_bars(d->dev);

    d->slot = -1;
    active_die = -1;

    pr_info(BS_LOG "slot now empty\n");
}

/*
 * Acquire a die into the MMIO slot.
 * Caller must hold slot_mutex.
 * Returns 0 on success, negative errno on failure.
 */
static int bs_do_acquire_locked(int die_id, u32 flags,
                                u64 *out_bar0, u64 *out_bar1,
                                u64 *out_bar1_size, u64 *out_bar3)
{
    struct bs_die_state *d;
    int i, ret;

    if (die_id < 0 || die_id >= n_dies)
        return -EINVAL;

    d = &dies[die_id];

    /* Same die already active — return addresses immediately */
    if (active_die == die_id) {
        pr_info(BS_LOG "die %d already active, returning addresses\n", die_id);
        goto out_addrs;
    }

    /* Different die active — release it first */
    if (active_die >= 0) {
        pr_info(BS_LOG "implicit release of die %d for die %d\n",
                active_die, die_id);
        bs_do_release_locked();
    }

    pr_info(BS_LOG "acquiring die %d (%s)...\n", die_id, pci_name(d->dev));

    /* SBR on downstream bridge (first acquire only, or if forced) */
    if (d->n_bridges > 0 && (!d->sbr_done || (flags & BS_ACQUIRE_FORCE_SBR))) {
        pr_info(BS_LOG "issuing SBR on downstream bridge %s...\n",
                pci_name(d->bridge_chain[0]));
        bs_secondary_bus_reset(d->bridge_chain[0]);
        d->sbr_done = true;
    }

    /* Disable decode on target chain */
    bs_disable_decode(d->dev);
    for (i = 0; i < d->n_bridges; i++)
        bs_disable_decode(d->bridge_chain[i]);

    /* Disable sibling downstream bridge to prevent PLX routing conflicts.
     * BIOS may have assigned overlapping memory windows to sibling PLX
     * downstream ports. If the sibling has MEM decode enabled with a window
     * that overlaps our target (0xF3000000), the PLX can't route and MMIO
     * returns all-F's. Only touch the sibling's downstream bridge (chain[0]),
     * not shared bridges. */
    {
        int sibling_id = (die_id % 2 == 0) ? die_id + 1 : die_id - 1;
        if (sibling_id >= 0 && sibling_id < n_dies &&
            dies[sibling_id].n_bridges > 0) {
            struct pci_dev *sib_br = dies[sibling_id].bridge_chain[0];
            bs_disable_decode(sib_br);
            bs_zero_bridge_windows(sib_br);
            pr_info(BS_LOG "cleared sibling downstream %s\n",
                    pci_name(sib_br));
        }
    }

    /* Program bridge windows top-down (root port -> PLX up -> downstream)
     * All bridges get the same window: 0xC00-0xFFF pref + 0xF3-0xF5 mem */
    pr_info(BS_LOG "programming bridges top-down...\n");
    for (i = d->n_bridges - 1; i >= 0; i--) {
        struct pci_dev *br = d->bridge_chain[i];

        bs_program_bridge_pref(br, BR_PREF_BASE, BR_PREF_LIMIT);
        bs_program_bridge_mem(br, (u32)BR_MEM_BASE, (u32)BR_MEM_LIMIT);
        bs_set_bridge_resource(br, 2, BR_PREF_BASE, BR_PREF_LIMIT,
                               IORESOURCE_MEM | IORESOURCE_PREFETCH | IORESOURCE_MEM_64);
        bs_set_bridge_resource(br, 1, BR_MEM_BASE, BR_MEM_LIMIT,
                               IORESOURCE_MEM);
    }

    /* Program die BARs */
    pr_info(BS_LOG "programming die BARs...\n");

    bs_write_bar32(d->dev, 0, (u32)SLOT_BAR0_ADDR);
    bs_set_resource(d->dev, 0, SLOT_BAR0_ADDR, BAR0_SIZE, IORESOURCE_MEM);

    bs_write_bar64(d->dev, 1, SLOT_BAR1_ADDR);
    bs_set_resource(d->dev, 1, SLOT_BAR1_ADDR, BAR1_SIZE,
                    IORESOURCE_MEM | IORESOURCE_PREFETCH | IORESOURCE_MEM_64);
    bs_set_resource(d->dev, 2, 0, 0, 0);  /* BAR 2 is upper half of BAR 1 */

    bs_write_bar64(d->dev, 3, SLOT_BAR3_ADDR);
    bs_set_resource(d->dev, 3, SLOT_BAR3_ADDR, BAR3_SIZE, IORESOURCE_MEM);
    bs_set_resource(d->dev, 4, 0, 0, 0);  /* BAR 4 is upper half of BAR 3 */

    pr_info(BS_LOG "  die %s: BAR0=0x%llx BAR1=0x%llx BAR3=0x%llx\n",
            pci_name(d->dev), SLOT_BAR0_ADDR, SLOT_BAR1_ADDR, SLOT_BAR3_ADDR);

    /* Claim resources top-down (root port -> PLX up -> downstream -> die) */
    pr_info(BS_LOG "claiming resources...\n");
    for (i = d->n_bridges - 1; i >= 0; i--) {
        struct pci_dev *br = d->bridge_chain[i];

        ret = pci_claim_resource(br, PCI_BRIDGE_RESOURCES + 2);
        if (ret) pr_warn(BS_LOG "claim bridge %s pref: %d\n", pci_name(br), ret);
        ret = pci_claim_resource(br, PCI_BRIDGE_RESOURCES + 1);
        if (ret) pr_warn(BS_LOG "claim bridge %s mem: %d\n", pci_name(br), ret);
    }

    ret = pci_claim_resource(d->dev, 0);
    if (ret) pr_warn(BS_LOG "claim die%d BAR0: %d\n", die_id, ret);
    ret = pci_claim_resource(d->dev, 1);
    if (ret) pr_warn(BS_LOG "claim die%d BAR1: %d\n", die_id, ret);
    ret = pci_claim_resource(d->dev, 3);
    if (ret) pr_warn(BS_LOG "claim die%d BAR3: %d\n", die_id, ret);

    /* Enable decode top-down (root port -> PLX up -> downstream -> die) */
    pr_info(BS_LOG "enabling decode...\n");
    for (i = d->n_bridges - 1; i >= 0; i--)
        bs_enable_decode(d->bridge_chain[i]);
    bs_enable_decode(d->dev);

    /*
     * Fix PM state: after SBR the hardware is in D0, but the kernel may
     * track it as D3cold. Now that bridges and BARs are programmed and
     * decode is enabled, the device is accessible — safe to update state.
     * This prevents nvidia's pci_enable_device() from attempting a
     * D3cold -> D0 platform transition that crashes the system.
     */
    d->dev->current_state = PCI_D0;
    pci_save_state(d->dev);

    /* Mark active */
    d->slot = 0;
    active_die = die_id;

    pr_info(BS_LOG "die %d acquired at slot 0\n", die_id);

out_addrs:
    *out_bar0 = SLOT_BAR0_ADDR;
    *out_bar1 = SLOT_BAR1_ADDR;
    *out_bar1_size = BAR1_SIZE;
    *out_bar3 = SLOT_BAR3_ADDR;
    return 0;
}

/* ---- Misc device (ioctl interface) ---- */

static long bs_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch (cmd) {
    case BAR_SWAP_QUERY: {
        struct bs_query_info info;
        int i;

        memset(&info, 0, sizeof(info));
        info.n_dies = n_dies;
        info.n_slots = 1;
        info.active_die = (active_die >= 0) ? (u32)active_die : 0xFFFFFFFF;

        for (i = 0; i < n_dies && i < BS_MAX_DIES; i++)
            info.mapped[i] = (dies[i].slot >= 0) ? 1 : 0;

        if (copy_to_user((void __user *)arg, &info, sizeof(info)))
            return -EFAULT;
        return 0;
    }

    case BAR_SWAP_ACQUIRE: {
        struct bs_acquire_req req;
        int ret;

        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;

        mutex_lock(&slot_mutex);
        ret = bs_do_acquire_locked(req.die_id, req.flags,
                                   &req.bar0_addr, &req.bar1_addr,
                                   &req.bar1_size, &req.bar3_addr);
        mutex_unlock(&slot_mutex);

        if (ret)
            return ret;

        if (copy_to_user((void __user *)arg, &req, sizeof(req)))
            return -EFAULT;
        return 0;
    }

    case BAR_SWAP_RELEASE: {
        struct bs_release_req req;

        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;

        mutex_lock(&slot_mutex);
        if (active_die != (int)req.die_id) {
            mutex_unlock(&slot_mutex);
            return (active_die < 0) ? 0 : -EINVAL;  /* not active = no-op */
        }
        bs_do_release_locked();
        mutex_unlock(&slot_mutex);
        return 0;
    }

    default:
        return -ENOTTY;
    }
}

static const struct file_operations bs_fops = {
    .owner          = THIS_MODULE,
    .unlocked_ioctl = bs_ioctl,
};

static struct miscdevice bs_misc = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = "bar_swap",
    .fops  = &bs_fops,
};

/* ---- Module init/exit ---- */

static int __init bar_swap_init(void)
{
    int ret;

    pr_info(BS_LOG "loading v2.0...\n");
    active_die = -1;

    ret = bs_discover();
    if (ret == 0) {
        pr_warn(BS_LOG "no GK210 dies found\n");
        goto register_dev;
    }

    /* Release ALL existing GPU resources and disable ALL dies.
     * Slot starts empty — userspace must ACQUIRE to use a die. */
    bs_release_all_gpu_resources();
    bs_disable_all_dies();

register_dev:
    ret = misc_register(&bs_misc);
    if (ret) {
        pr_err(BS_LOG "misc_register failed: %d\n", ret);
        return ret;
    }

    pr_info(BS_LOG "registered /dev/bar_swap — %d dies, slot empty\n", n_dies);
    return 0;
}

static void __exit bar_swap_exit(void)
{
    int i, j;

    pr_info(BS_LOG "unloading...\n");
    misc_deregister(&bs_misc);

    /* Release active die if any */
    mutex_lock(&slot_mutex);
    if (active_die >= 0)
        bs_do_release_locked();
    mutex_unlock(&slot_mutex);

    /* Restore all original bridge and BAR config space */
    pr_info(BS_LOG "restoring original state...\n");
    for (i = 0; i < n_dies; i++) {
        struct bs_die_state *d = &dies[i];

        /* Restore bridges (top-down = reverse of chain order) */
        for (j = d->n_bridges - 1; j >= 0; j--)
            bs_restore_bridge(d->bridge_chain[j], &d->bridge_saved[j]);

        /* Restore die BARs */
        bs_restore_die_bars(d);
    }

    /* Drop pci_dev references */
    for (i = 0; i < n_dies; i++) {
        struct bs_die_state *d = &dies[i];
        if (d->dev)
            pci_dev_put(d->dev);
        for (j = 0; j < d->n_bridges; j++)
            if (d->bridge_chain[j])
                pci_dev_put(d->bridge_chain[j]);
    }

    pr_info(BS_LOG "unloaded\n");
}

module_init(bar_swap_init);
module_exit(bar_swap_exit);
