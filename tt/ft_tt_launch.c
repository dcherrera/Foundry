/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_launch.c
 * @brief Implementation of Tensix run control: image placement, launch-message
 *   encoding, reset release, multi-core placement groups, and a completion poll that
 *   always terminates.
 *
 *   No window walk and no width choice live here. Device access is
 *   ft_tt_{write,read}_register_words(), through a path attached to the caller's
 *   window — see ft_tt_launch_noc_over_window().
 *
 * @status Unit-tested against a simulated L1 (tests/test_launch.c) and run on the card
 *   through all 13 steps of docs/tt_launch_notes.md §6.
 * @issues None
 * @todo None
 */

/* clock_gettime and CLOCK_MONOTONIC are POSIX.1-2008; -std=c99 turns off glibc's
 * implicit feature-test macros, so ask for them explicitly (same as ft_tt_tlb.c). */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_launch.h"

#include <string.h>

#include <time.h>

/* The kernel-config map. Read, never retyped: this header is the ONE source of the
 * relocation map (build_tensix_program.sh parses the same macros out of it), and the
 * placement rule below has to agree with the linker exactly. */
#include "ft_tt_op_l1_layout.h"

/* ------------------------------------------------------------------------- */
/* Per-processor geography                                                   */
/* ------------------------------------------------------------------------- */

/** Indexed by FT_TT_PROC_*: B, NC, T0, T1, T2 — the launch message's array order
 *  (ft_tt_mailbox.h, fw/ft_*_fw.c). */
typedef struct processor_geography {
    const char *name;
    uint32_t reset_bit;
    uint32_t firmware_base;
    uint32_t firmware_size;
    uint32_t local_init_scratch;
} processor_geography;

static const processor_geography k_processors[FT_TT_PROC_COUNT] = {
    {"BRISC", FT_TT_SOFT_RESET_BRISC, FT_TT_MEM_BRISC_FIRMWARE_BASE, FT_TT_MEM_BRISC_FIRMWARE_SIZE,
     FT_TT_MEM_BRISC_INIT_LOCAL_L1_BASE_SCRATCH},
    {"NCRISC", FT_TT_SOFT_RESET_NCRISC, FT_TT_MEM_NCRISC_FIRMWARE_BASE, FT_TT_MEM_NCRISC_FIRMWARE_SIZE,
     FT_TT_MEM_NCRISC_INIT_LOCAL_L1_BASE_SCRATCH},
    {"TRISC0", FT_TT_SOFT_RESET_TRISC0, FT_TT_MEM_TRISC0_FIRMWARE_BASE, FT_TT_MEM_TRISC0_FIRMWARE_SIZE,
     FT_TT_MEM_TRISC0_INIT_LOCAL_L1_BASE_SCRATCH},
    {"TRISC1", FT_TT_SOFT_RESET_TRISC1, FT_TT_MEM_TRISC1_FIRMWARE_BASE, FT_TT_MEM_TRISC1_FIRMWARE_SIZE,
     FT_TT_MEM_TRISC1_INIT_LOCAL_L1_BASE_SCRATCH},
    {"TRISC2", FT_TT_SOFT_RESET_TRISC2, FT_TT_MEM_TRISC2_FIRMWARE_BASE, FT_TT_MEM_TRISC2_FIRMWARE_SIZE,
     FT_TT_MEM_TRISC2_INIT_LOCAL_L1_BASE_SCRATCH},
};

static const processor_geography *geography_for(unsigned processor)
{
    return processor < FT_TT_PROC_COUNT ? &k_processors[processor] : NULL;
}

const char *ft_tt_processor_name(unsigned processor)
{
    const processor_geography *geo = geography_for(processor);
    return geo ? geo->name : "?";
}

uint32_t ft_tt_processor_reset_bit(unsigned processor)
{
    const processor_geography *geo = geography_for(processor);
    return geo ? geo->reset_bit : 0u;
}

uint32_t ft_tt_processor_firmware_base(unsigned processor)
{
    const processor_geography *geo = geography_for(processor);
    return geo ? geo->firmware_base : 0u;
}

uint32_t ft_tt_processor_firmware_size(unsigned processor)
{
    const processor_geography *geo = geography_for(processor);
    return geo ? geo->firmware_size : 0u;
}

uint32_t ft_tt_processor_local_init_scratch(unsigned processor)
{
    const processor_geography *geo = geography_for(processor);
    return geo ? geo->local_init_scratch : 0u;
}

/* ------------------------------------------------------------------------- */
/* Device access helpers                                                     */
/* ------------------------------------------------------------------------- */

static int noc_write(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t addr,
                     const void *src, size_t bytes)
{
    if (!noc || !noc->write)
        return FT_TT_EINVAL;
    return noc->write(noc->user, x, y, addr, src, bytes);
}

static int noc_read(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t addr, void *dst,
                    size_t bytes)
{
    if (!noc || !noc->read)
        return FT_TT_EINVAL;
    return noc->read(noc->user, x, y, addr, dst, bytes);
}

static int noc_write_u32(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t addr,
                         uint32_t value)
{
    return noc_write(noc, x, y, addr, &value, sizeof(value));
}

static int noc_read_u32(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t addr,
                        uint32_t *out_value)
{
    return noc_read(noc, x, y, addr, out_value, sizeof(*out_value));
}

/* ------------------------------------------------------------------------- */
/* NoC access over a TLB window                                              */
/* ------------------------------------------------------------------------- */

/**
 * The ft_tt_noc_access vtable over an ft_tt_noc path, and nothing else.
 *
 * Every byte moves through ft_tt_write_register_words() / ft_tt_read_register_words():
 * the window walk, the re-aiming, the copy AND the choice of 32-bit granularity all
 * live in ft_tt_noc.c. That width is not an optimisation choice — the soft-reset
 * register at 0xFFB121B0 is four bytes wide and so is a go message, and a 16 B store
 * over either takes its neighbours with it — which is exactly why it belongs beside the
 * width enum rather than in a private helper here.
 */
static int window_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                        size_t bytes)
{
    ft_tt_noc_window_binding *binding = (ft_tt_noc_window_binding *)user;
    ft_tt_noc_coord coord;

    if (!binding || !src)
        return FT_TT_EINVAL;

    coord.x = x;
    coord.y = y;
    return ft_tt_write_register_words(&binding->path, coord, addr, src, bytes);
}

static int window_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst, size_t bytes)
{
    ft_tt_noc_window_binding *binding = (ft_tt_noc_window_binding *)user;
    ft_tt_noc_coord coord;

    if (!binding || !dst)
        return FT_TT_EINVAL;

    coord.x = x;
    coord.y = y;
    return ft_tt_read_register_words(&binding->path, coord, addr, dst, bytes);
}

int ft_tt_launch_noc_over_window(ft_tt_ctx *ctx, ft_tt_tlb_window *window,
                                 ft_tt_noc_window_binding *binding, ft_tt_noc_access *out_access)
{
    int rc;

    if (!ctx || !window || !binding || !out_access)
        return FT_TT_EINVAL;

    /* Attach, not open: the window is the caller's and outlives the accessor. */
    rc = ft_tt_noc_path_attach(ctx, window, &binding->path);
    if (rc != FT_TT_OK)
        return rc;

    out_access->write = window_write;
    out_access->read = window_read;
    out_access->user = binding;
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* Reset control                                                             */
/* ------------------------------------------------------------------------- */

int ft_tt_launch_read_reset(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t *out_value)
{
    if (!out_value)
        return FT_TT_EINVAL;
    return noc_read_u32(noc, x, y, FT_TT_TENSIX_SOFT_RESET_ADDR, out_value);
}

int ft_tt_launch_assert_reset(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t risc_mask)
{
    uint32_t current = 0;
    int rc = ft_tt_launch_read_reset(noc, x, y, &current);

    if (rc != FT_TT_OK)
        return rc;
    return noc_write_u32(noc, x, y, FT_TT_TENSIX_SOFT_RESET_ADDR, current | risc_mask);
}

int ft_tt_launch_deassert_reset(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                uint32_t risc_mask, bool staggered)
{
    uint32_t current = 0;
    uint32_t updated;
    int rc = ft_tt_launch_read_reset(noc, x, y, &current);

    if (rc != FT_TT_OK)
        return rc;

    updated = current & ~risc_mask;
    if (staggered)
        updated |= FT_TT_SOFT_RESET_STAGGERED_START;
    return noc_write_u32(noc, x, y, FT_TT_TENSIX_SOFT_RESET_ADDR, updated);
}

/* ------------------------------------------------------------------------- */
/* Image loading                                                             */
/* ------------------------------------------------------------------------- */

/**
 * Where the host writes one segment.
 *
 * Firmware `.data` is the special case: its execution address is in the processor's
 * private local memory, its ELF load address is not where the firmware looks for it,
 * and the firmware's `do_crt1()` copies it from a fixed per-processor scratch region
 * instead (fw/ft_*_fw.c initialise_local_memory()).
 */
static uint32_t destination_of(const ft_tt_elf_segment *segment, ft_tt_image_kind kind,
                               const processor_geography *geo)
{
    bool is_local_memory =
        (segment->exec_address & FT_TT_MEM_LOCAL_BASE) == FT_TT_MEM_LOCAL_BASE;

    if (kind == FT_TT_IMAGE_FIRMWARE && is_local_memory)
        return (segment->exec_address & ~(uint32_t)FT_TT_MEM_LOCAL_BASE) + geo->local_init_scratch;
    return segment->load_address;
}

/**
 * One L1 span a kernel image is allowed to occupy, named so a refusal can say which
 * span it measured against.
 */
typedef struct kernel_text_region {
    const char *name;
    uint32_t base; /**< inclusive */
    uint32_t end;  /**< exclusive; equal to `base` means "this processor has no such region" */
} kernel_text_region;

/** Index into the array kernel_text_regions() fills. */
enum {
    KERNEL_REGION_FIRMWARE_HOLE = 0,
    KERNEL_REGION_CONFIG_SLAB = 1,
    KERNEL_REGION_COUNT = 2
};

/**
 * The only two places a Tensix kernel is ever linked, in the order a refusal names them.
 * The ELF chooses between them by where its text is; nothing here is a caller's opinion.
 *
 *  - FIRMWARE HOLE — the default link. fw/ft_tensix_kernel.ld puts `.text` at `__fw_export_text_end`,
 *    which is in the TAIL of this processor's own firmware hole, and the linker's only
 *    bound is MEM_*_KERNEL_SIZE (1497 KiB, i.e. all of L1). What lies above the hole is
 *    the NEXT processor's firmware. Measured 2026-08-11: TRISC0's `vector_add` clears
 *    TRISC1's firmware base by 68 bytes, so the hole end is a live bound.
 *  - KERNEL-CONFIG SLAB — a RELOCATE_KERNEL_TEXT=1 link
 *    (`--defsym=__fw_export_text_end=<slab>`): every kernel binary at a fixed slab in the
 *    kernel-config region (ft_tt_op_l1_layout.h). This is not a preference — the firmware tails are
 *    1296-1952 B, `matmul_compute` misses TRISC0's by 44 B and `rms_norm_compute` misses
 *    all three (ft_tt_op_l1_layout.h has the measured table), so without it the op
 *    library stops at `vector_add`.
 */
static void kernel_text_regions(unsigned processor, const processor_geography *geo,
                                kernel_text_region regions[KERNEL_REGION_COUNT])
{
    uint32_t slab_base = ft_tt_op_text_base(processor);
    uint32_t slab_bytes = ft_tt_op_text_slab_bytes(processor);

    regions[KERNEL_REGION_FIRMWARE_HOLE].name = "firmware hole";
    regions[KERNEL_REGION_FIRMWARE_HOLE].base = geo->firmware_base;
    regions[KERNEL_REGION_FIRMWARE_HOLE].end = geo->firmware_base + geo->firmware_size;

    /* A slab is legal only while it is inside the kernel-config region: outside it a
     * segment would land in the CB arena, or lower down on the mailbox. The map is a
     * compile-time constant, so this can only fire if someone edits the header — which
     * is exactly when it is wanted. */
    regions[KERNEL_REGION_CONFIG_SLAB].name = "kernel-config slab";
    regions[KERNEL_REGION_CONFIG_SLAB].base = slab_base;
    regions[KERNEL_REGION_CONFIG_SLAB].end = slab_base + slab_bytes;
    if (slab_base < FT_TT_OP_KERNEL_CONFIG_BASE ||
        slab_base + slab_bytes >
            FT_TT_OP_KERNEL_CONFIG_BASE + FT_TT_OP_KERNEL_CONFIG_BYTES)
        regions[KERNEL_REGION_CONFIG_SLAB].end = regions[KERNEL_REGION_CONFIG_SLAB].base;
}

/** The region holding `address`, or NULL. An empty region holds nothing. */
static const kernel_text_region *region_holding(const kernel_text_region *regions,
                                                uint32_t address)
{
    unsigned index;

    for (index = 0; index < (unsigned)KERNEL_REGION_COUNT; index++) {
        if (address >= regions[index].base && address < regions[index].end)
            return &regions[index];
    }
    return NULL;
}

/**
 * THE placement rule. Refuse anything that would land outside L1, overrun a firmware
 * hole, or escape a kernel-config slab.
 *
 * Firmware: text exactly at its own base, and no larger than its hole.
 *
 * Kernel: **entirely inside its own firmware hole, OR entirely inside its own
 * kernel-config slab — never straddling.** The region is picked by where the ELF put
 * its TEXT, because the ELF is the only party that cannot be wrong about how it was
 * linked; every other segment must then agree with that choice, which is what makes a
 * straddle a refusal instead of a half-write.
 *
 * This is the single rule Phase 4 had to keep in two files (`validate_placement()` here
 * and `validate_slab()` in ft_tt_op_dispatch.c). Per region it is as strict as before or
 * stricter — the hole branch gains a LOWER bound it never had — and it protects the same
 * invariant on both branches: a loaded kernel can reach neither a firmware image, nor
 * another processor's slab, nor the CB arena.
 */
static int validate_placement(const ft_tt_elf_image *image, ft_tt_image_kind kind,
                              const processor_geography *geo, unsigned processor)
{
    const ft_tt_elf_segment *text = ft_tt_elf_text_segment(image);
    kernel_text_region regions[KERNEL_REGION_COUNT];
    const kernel_text_region *region = NULL;
    unsigned index;

    if (kind == FT_TT_IMAGE_FIRMWARE) {
        if (text->exec_address != geo->firmware_base) {
            ft_tt_log_error("launch: %s firmware text is at 0x%x, expected 0x%x", geo->name,
                            text->exec_address, geo->firmware_base);
            return FT_TT_EINVAL;
        }
        if (text->memory_bytes > geo->firmware_size) {
            ft_tt_log_error("launch: %s firmware text is %u bytes, hole is %u", geo->name,
                            text->memory_bytes, geo->firmware_size);
            return FT_TT_EINVAL;
        }
    } else {
        kernel_text_regions(processor, geo, regions);
        region = region_holding(regions, text->exec_address);
        if (!region) {
            ft_tt_log_error("launch: %s kernel text at 0x%x is in neither legal region — its "
                            "firmware hole [0x%x,0x%x) nor its kernel-config slab [0x%x,0x%x)",
                            geo->name, text->exec_address,
                            regions[KERNEL_REGION_FIRMWARE_HOLE].base,
                            regions[KERNEL_REGION_FIRMWARE_HOLE].end,
                            regions[KERNEL_REGION_CONFIG_SLAB].base,
                            regions[KERNEL_REGION_CONFIG_SLAB].end);
            return FT_TT_EINVAL;
        }
        if (region == &regions[KERNEL_REGION_FIRMWARE_HOLE] &&
            text->exec_address == geo->firmware_base) {
            ft_tt_log_error("launch: %s kernel text at 0x%x overlaps its firmware at 0x%x",
                            geo->name, text->exec_address, geo->firmware_base);
            return FT_TT_EINVAL;
        }
        /* A relocated image starts AT its slab. Anything else means the ELF was not built
         * with RELOCATE_KERNEL_TEXT, or this map and build_tensix_program.sh's have
         * drifted apart — which loads fine and leaves the launch message pointing at
         * empty L1. */
        if (region == &regions[KERNEL_REGION_CONFIG_SLAB] &&
            image->entry_address != region->base) {
            ft_tt_log_error("launch: %s kernel entry is 0x%x, its slab starts at 0x%x — the ELF "
                            "was not built with RELOCATE_KERNEL_TEXT, or the two maps have "
                            "drifted apart",
                            geo->name, image->entry_address, region->base);
            return FT_TT_EINVAL;
        }
    }

    for (index = 0; index < image->segment_count; index++) {
        uint32_t destination = destination_of(&image->segments[index], kind, geo);
        uint32_t end = destination + image->segments[index].file_bytes;

        if (end < destination || end > FT_TT_MEM_L1_SIZE) {
            ft_tt_log_error("launch: %s segment %u would land at [0x%x,0x%x), outside L1",
                            geo->name, index, destination, end);
            return FT_TT_EINVAL;
        }
        if (image->segments[index].file_bytes == 0u)
            continue;
        if (region && (destination < region->base || end > region->end)) {
            ft_tt_log_error("launch: %s kernel segment %u lands at [0x%x,0x%x), outside the %s "
                            "[0x%x,0x%x) its text is in — it would overwrite what lives there",
                            geo->name, index, destination, end, region->name, region->base,
                            region->end);
            return FT_TT_EINVAL;
        }
    }
    return FT_TT_OK;
}

int ft_tt_launch_load_image(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, unsigned processor,
                            ft_tt_image_kind kind, const void *elf_data, size_t elf_size,
                            ft_tt_image_placement *out_place)
{
    const processor_geography *geo = geography_for(processor);
    ft_tt_elf_image image;
    ft_tt_image_placement placement;
    unsigned index;
    int rc;

    if (!noc || !geo || !elf_data)
        return FT_TT_EINVAL;

    rc = ft_tt_elf_parse(elf_data, elf_size, &image);
    if (rc != FT_TT_OK)
        return rc;

    rc = validate_placement(&image, kind, geo, processor);
    if (rc != FT_TT_OK)
        return rc;

    memset(&placement, 0, sizeof(placement));
    placement.entry_address = image.entry_address;
    placement.text_address = image.segments[0].exec_address;
    placement.text_bytes = image.segments[0].memory_bytes;
    placement.lowest_l1_address = UINT32_MAX;

    for (index = 0; index < image.segment_count; index++) {
        const ft_tt_elf_segment *segment = &image.segments[index];
        uint32_t destination = destination_of(segment, kind, geo);

        if (segment->file_bytes == 0u)
            continue;
        rc = noc_write(noc, x, y, destination, segment->file_data, segment->file_bytes);
        if (rc != FT_TT_OK) {
            ft_tt_log_error("launch: writing %s segment %u to 0x%x failed: %s", geo->name, index,
                            destination, ft_tt_error_string(rc));
            return rc;
        }
        if (destination < placement.lowest_l1_address)
            placement.lowest_l1_address = destination;
        if (destination + segment->file_bytes > placement.highest_l1_address)
            placement.highest_l1_address = destination + segment->file_bytes;
        placement.bytes_written += segment->file_bytes;
    }

    if (out_place)
        *out_place = placement;
    return FT_TT_OK;
}

uint32_t ft_tt_image_placement_l1_floor(const ft_tt_image_placement *placements, unsigned count)
{
    uint32_t floor = FT_TT_L1_USABLE_BASE;
    unsigned index;

    if (!placements && count > 0u)
        return 0u;

    for (index = 0; index < count; index++) {
        /* A placement nothing was written into is an idle processor, not an image
         * at address 0. */
        if (placements[index].bytes_written == 0u)
            continue;
        if (placements[index].highest_l1_address > floor)
            floor = placements[index].highest_l1_address;
    }

    /* Round up: the allocator would round anyway, and a floor that splits a 16 B
     * unit would put the first buffer's first bytes over the image's last. */
    floor = (floor + (FT_TT_L1_ALIGNMENT - 1u)) & ~(uint32_t)(FT_TT_L1_ALIGNMENT - 1u);
    return floor;
}

/* ------------------------------------------------------------------------- */
/* Launch-message encoding                                                   */
/* ------------------------------------------------------------------------- */

/** Convert an absolute L1 address into a 16-bit offset from the config base, or
 *  report that it does not fit. 0xFFFF is reserved as the "no args" sentinel. */
static int encode_short_offset(uint32_t address, uint32_t base, const char *what, uint16_t *out)
{
    uint32_t offset;

    if (address < base) {
        ft_tt_log_error("launch: %s at 0x%x is below kernel_config_base 0x%x", what, address, base);
        return FT_TT_EINVAL;
    }
    offset = address - base;
    if (offset >= FT_TT_RTA_NO_ARGS_SENTINEL) {
        ft_tt_log_error("launch: %s is 0x%x past kernel_config_base, over the 16-bit field", what,
                        offset);
        return FT_TT_EINVAL;
    }
    *out = (uint16_t)offset;
    return FT_TT_OK;
}

int ft_tt_launch_encode_launch_message(const ft_tt_launch_config *config, void *buffer,
                                       size_t buffer_bytes)
{
    ft_tt_kernel_config_msg *message = (ft_tt_kernel_config_msg *)buffer;
    unsigned processor;
    int rc;

    if (!config || !buffer)
        return FT_TT_EINVAL;
    if (buffer_bytes < sizeof(*message)) {
        ft_tt_log_error("launch: launch-message buffer is %zu bytes, need %zu", buffer_bytes,
                        sizeof(*message));
        return FT_TT_EINVAL;
    }
    if ((config->kernel_config_base & (FT_TT_MAX_NOC_WRITE_ALIGNMENT - 1u)) != 0u) {
        ft_tt_log_error("launch: kernel_config_base 0x%x is not %u-byte aligned",
                        config->kernel_config_base, FT_TT_MAX_NOC_WRITE_ALIGNMENT);
        return FT_TT_EINVAL;
    }
    if (config->enables == 0u) {
        ft_tt_log_error("launch: `enables` is 0, so the go message would run nothing");
        return FT_TT_EINVAL;
    }

    memset(message, 0, sizeof(*message));
    message->kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX] = config->kernel_config_base;
    message->mode = FT_TT_DISPATCH_MODE_HOST;
    message->enables = config->enables;
    message->brisc_noc_id = config->brisc_noc_id;
    message->local_cb_mask = config->local_cb_mask;
    /* "No remote circular buffers": Foundry's firmware accepts only this value and halts
     * on any other (fw/ft_*_fw.c run_launch()). */
    message->min_remote_cb_start_index = (uint8_t)FT_TT_NUM_CIRCULAR_BUFFERS;

    /* Packed struct: never take the address of a member, assign through a local. */
    if (config->local_cb_config_address != 0u) {
        uint16_t offset = 0;

        rc = encode_short_offset(config->local_cb_config_address, config->kernel_config_base,
                                 "local CB config", &offset);
        if (rc != FT_TT_OK)
            return rc;
        message->local_cb_offset = offset;
    }

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        bool enabled = (config->enables & (1u << processor)) != 0u;

        message->rta_offset[processor].rta_offset = FT_TT_RTA_NO_ARGS_SENTINEL;
        message->rta_offset[processor].crta_offset = FT_TT_RTA_NO_ARGS_SENTINEL;

        if (enabled && config->kernel_text_address[processor] == 0u) {
            ft_tt_log_error("launch: %s is enabled but has no kernel text address",
                            ft_tt_processor_name(processor));
            return FT_TT_EINVAL;
        }
        if (config->kernel_text_address[processor] != 0u) {
            /* Unsigned wraparound is the encoding, not an accident — see the header. */
            message->kernel_text_offset[processor] =
                config->kernel_text_address[processor] - config->kernel_config_base;
            message->kernel_text_size[processor] = config->kernel_text_bytes[processor];
        }
        if (config->runtime_args_address[processor] != 0u) {
            uint16_t offset = 0;

            rc = encode_short_offset(config->runtime_args_address[processor],
                                     config->kernel_config_base, ft_tt_processor_name(processor),
                                     &offset);
            if (rc != FT_TT_OK)
                return rc;
            message->rta_offset[processor].rta_offset = offset;
        }
    }
    return FT_TT_OK;
}

int ft_tt_launch_encode_cb_config(const ft_tt_launch_cb_entry *entries, unsigned entry_count,
                                  void *buffer, size_t buffer_bytes, uint64_t *out_mask)
{
    const size_t required =
        (size_t)FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * sizeof(uint32_t);
    uint32_t *words = (uint32_t *)buffer;
    uint64_t mask = 0;
    unsigned index;

    if (!buffer || (!entries && entry_count > 0u) || !out_mask)
        return FT_TT_EINVAL;
    if (buffer_bytes < required) {
        ft_tt_log_error("launch: CB config buffer is %zu bytes, need %zu", buffer_bytes, required);
        return FT_TT_EINVAL;
    }

    memset(buffer, 0, required);
    for (index = 0; index < entry_count; index++) {
        const ft_tt_launch_cb_entry *entry = &entries[index];
        size_t base_word;

        if (entry->index >= FT_TT_NUM_CIRCULAR_BUFFERS) {
            ft_tt_log_error("launch: CB index %u is out of range (max %u)", entry->index,
                            FT_TT_NUM_CIRCULAR_BUFFERS - 1u);
            return FT_TT_EINVAL;
        }
        base_word = (size_t)entry->index * FT_TT_WORDS_PER_LOCAL_CB_CONFIG;
        words[base_word + 0u] = entry->address;
        words[base_word + 1u] = entry->size_bytes;
        words[base_word + 2u] = entry->page_count;
        words[base_word + 3u] = entry->page_size_bytes;
        mask |= (uint64_t)1u << entry->index;
    }

    *out_mask = mask;
    return FT_TT_OK;
}

/* ------------------------------------------------------------------------- */
/* core_info — the endpoint tables NoC sanitize resolves against              */
/* ------------------------------------------------------------------------- */

/** Upper bound of the PCIe address window recorded in core_info (historical, see
 *  ft_tt_launch.h). Foundry never targets PCIe from a kernel. */
#define FT_TT_PCIE_ADDRESS_UPPER_BOUND 0x13FFFFFFFFFFFFFFuLL

const char *ft_tt_sanitize_return_code_string(uint16_t return_code)
{
    switch (return_code) {
    case FT_TT_SANITIZE_NO_REPORT:                      return "no report";
    case FT_TT_SANITIZE_OK:                             return "OK";
    case FT_TT_SANITIZE_NOC_ADDR_UNDERFLOW:             return "address below the space";
    case FT_TT_SANITIZE_NOC_ADDR_OVERFLOW:              return "address past the end of the space";
    case FT_TT_SANITIZE_NOC_ADDR_ZERO_LENGTH:           return "zero length";
    case FT_TT_SANITIZE_NOC_TARGET_INVALID_XY:          return "endpoint is not an addressable core";
    case FT_TT_SANITIZE_NOC_MULTICAST_NON_WORKER:       return "multicast to a non-worker";
    case FT_TT_SANITIZE_NOC_MULTICAST_INVALID_RANGE:    return "multicast range invalid";
    case FT_TT_SANITIZE_NOC_ALIGNMENT:                  return "misaligned";
    case FT_TT_SANITIZE_NOC_MIXED_VIRTUAL_AND_PHYSICAL: return "mixed virtual and physical";
    case FT_TT_SANITIZE_INLINE_WRITE_DRAM_UNSUPPORTED:  return "inline write to DRAM";
    case FT_TT_SANITIZE_NOC_ADDR_MAILBOX:               return "write into the read-only region";
    case FT_TT_SANITIZE_NOC_LINKED_TRANSACTION:         return "linked-transaction violation";
    case FT_TT_SANITIZE_L1_ADDR_OVERFLOW:               return "local L1 address past the end";
    case FT_TT_SANITIZE_ETH_SRC_L1_ADDR_OVERFLOW:       return "eth source past the end";
    case FT_TT_SANITIZE_ETH_DEST_L1_ADDR_OVERFLOW:      return "eth destination past the end";
    case FT_TT_SANITIZE_CB_OUT_OF_BOUNDS:               return "outside the circular buffer";
    case FT_TT_SANITIZE_NOC_INVALID_TXN_ID:             return "invalid transaction id";
    default:                                            return "unrecognised";
    }
}

int ft_tt_launch_core_info_from_grid(const ft_tt_grid_info *grid, uint32_t logical_x,
                                     uint32_t logical_y, uint32_t l1_unreserved_start,
                                     ft_tt_launch_non_worker_core *storage, unsigned storage_count,
                                     ft_tt_launch_core_info_config *out_config)
{
    uint32_t bank_count;
    uint32_t ports_per_bank;
    uint32_t noc_size_x = 0;
    uint32_t noc_size_y = 0;
    uint32_t bank;
    unsigned used = 0;
    int check_rc;

    if (!grid || !storage || !out_config)
        return FT_TT_EINVAL;
    if (logical_x > 0xFFu || logical_y > 0xFFu) {
        ft_tt_log_error("core_info: logical (%u, %u) does not fit the byte-wide fields",
                        logical_x, logical_y);
        return FT_TT_EINVAL;
    }

    /* This is the one place that builds the device-visible view of the grid, so
     * it is where the kernels' compiled-in board shape is checked against the
     * live one. On a card with a different core or bank count the images'
     * addressing describes a grid that is not there, and nothing on the device
     * would complain (ft_tt_coord.h). */
    check_rc = ft_tt_coord_check_kernel_build_assumptions(grid);
    if (check_rc != FT_TT_OK)
        return check_rc;

    check_rc = ft_tt_coord_noc_size(grid, &noc_size_x, &noc_size_y);
    if (check_rc != FT_TT_OK)
        return check_rc;

    memset(out_config, 0, sizeof(*out_config));

    bank_count = ft_tt_coord_dram_bank_count(grid);
    ports_per_bank = ft_tt_coord_dram_ports_per_bank(grid);
    for (bank = 0; bank < bank_count; bank++) {
        uint32_t port;

        for (port = 0; port < ports_per_bank; port++) {
            ft_tt_noc_coord coord;
            int rc;

            if (used >= storage_count) {
                ft_tt_log_error("core_info: %u slots cannot hold %u DRAM banks x %u ports",
                                storage_count, bank_count, ports_per_bank);
                return FT_TT_ENOMEM;
            }
            rc = ft_tt_coord_dram_port(grid, bank, port, FT_TT_COORD_TRANSLATED, &coord);
            if (rc != FT_TT_OK)
                return rc;
            storage[used].x = (uint16_t)coord.x;
            storage[used].y = (uint16_t)coord.y;
            storage[used].type = FT_TT_ADDRESSABLE_CORE_DRAM;
            used++;
        }
    }

    /* All four are LIVE, from the grid's architecture and its telemetry masks.
     * A zeroed noc_size_x is not "empty", it is wrong — it underflows
     * virtual_end_x to 0xFFFFFFFF and makes every coordinate classify as
     * TENSIX (dev notes §1) — and a constant one is wrong on any part but this
     * one. */
    out_config->noc_size_x = (uint8_t)noc_size_x;
    out_config->noc_size_y = (uint8_t)noc_size_y;
    out_config->worker_grid_size_x = (uint8_t)ft_tt_coord_tensix_column_count(grid);
    out_config->worker_grid_size_y = (uint8_t)ft_tt_coord_tensix_row_count(grid);
    out_config->absolute_logical_x = (uint8_t)logical_x;
    out_config->absolute_logical_y = (uint8_t)logical_y;
    out_config->l1_unreserved_start = l1_unreserved_start;
    /* A bank's addresses start at 0 and the window covers every bank end to end. */
    out_config->noc_dram_addr_base = 0u;
    out_config->noc_dram_addr_end =
        (uint64_t)bank_count * ft_tt_noc_space_size(FT_TT_MEM_SPACE_DRAM);
    out_config->noc_pcie_addr_base = 0u;
    out_config->noc_pcie_addr_end = FT_TT_PCIE_ADDRESS_UPPER_BOUND;
    out_config->virtual_non_worker_cores = storage;
    out_config->virtual_non_worker_count = used;
    return FT_TT_OK;
}

/** Stamp every slot of an addressable-core table with the sentinel. Zeroing is the bug
 *  this function exists to prevent: a zeroed slot is (0, 0) of type TENSIX — a real
 *  coordinate with a real type — and matches in get_core_type()'s first loop. */
static void fill_addressable_core_table(ft_tt_addressable_core *table, unsigned count)
{
    unsigned index;

    for (index = 0; index < count; index++) {
        table[index].x = FT_TT_CORE_COORD_INVALID;
        table[index].y = FT_TT_CORE_COORD_INVALID;
        table[index].type = (uint8_t)FT_TT_ADDRESSABLE_CORE_UNKNOWN;
    }
}

int ft_tt_launch_encode_core_info(const ft_tt_launch_core_info_config *config, void *buffer,
                                  size_t buffer_bytes)
{
    ft_tt_core_info_msg message;
    unsigned index;

    if (!config || !buffer)
        return FT_TT_EINVAL;
    if (buffer_bytes < sizeof(message)) {
        ft_tt_log_error("core_info: buffer is %zu bytes, need %zu", buffer_bytes, sizeof(message));
        return FT_TT_EINVAL;
    }
    if (config->virtual_non_worker_count > FT_TT_MAX_VIRTUAL_NON_WORKER_CORES) {
        ft_tt_log_error("core_info: %u virtual non-worker cores exceeds the ABI's %u slots",
                        config->virtual_non_worker_count, FT_TT_MAX_VIRTUAL_NON_WORKER_CORES);
        return FT_TT_EINVAL;
    }
    if (config->virtual_non_worker_count != 0u && !config->virtual_non_worker_cores)
        return FT_TT_EINVAL;

    memset(&message, 0, sizeof(message));
    fill_addressable_core_table(message.non_worker_cores, FT_TT_MAX_PHYSICAL_NON_WORKER_CORES);
    fill_addressable_core_table(message.virtual_non_worker_cores,
                                FT_TT_MAX_VIRTUAL_NON_WORKER_CORES);
    for (index = 0; index < FT_TT_MAX_HARVESTED_ON_AXIS; index++) {
        message.harvested_coords[index] = FT_TT_CORE_COORD_INVALID;
        message.virtual_harvested_coords[index] = FT_TT_CORE_COORD_INVALID;
    }

    for (index = 0; index < config->virtual_non_worker_count; index++) {
        const ft_tt_launch_non_worker_core *core = &config->virtual_non_worker_cores[index];

        if (core->x >= FT_TT_CORE_COORD_INVALID || core->y >= FT_TT_CORE_COORD_INVALID) {
            ft_tt_log_error("core_info: entry %u at (%u, %u) does not fit a byte below the 0xFF "
                            "sentinel", index, (unsigned)core->x, (unsigned)core->y);
            return FT_TT_EINVAL;
        }
        message.virtual_non_worker_cores[index].x = (uint8_t)core->x;
        message.virtual_non_worker_cores[index].y = (uint8_t)core->y;
        message.virtual_non_worker_cores[index].type = (uint8_t)core->type;
    }

    message.noc_pcie_addr_base = config->noc_pcie_addr_base;
    message.noc_pcie_addr_end = config->noc_pcie_addr_end;
    message.noc_dram_addr_base = config->noc_dram_addr_base;
    message.noc_dram_addr_end = config->noc_dram_addr_end;
    message.noc_size_x = config->noc_size_x;
    message.noc_size_y = config->noc_size_y;
    message.worker_grid_size_x = config->worker_grid_size_x;
    message.worker_grid_size_y = config->worker_grid_size_y;
    message.absolute_logical_x = config->absolute_logical_x;
    message.absolute_logical_y = config->absolute_logical_y;
    message.l1_unreserved_start = config->l1_unreserved_start;
    message.core_magic_number = FT_TT_CORE_MAGIC_WORKER;

    memcpy(buffer, &message, sizeof(message));
    return FT_TT_OK;
}

int ft_tt_launch_prime_watcher(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    unsigned index;
    int rc = noc_write_u32(noc, x, y, FT_TT_MB_ADDR(watcher.enable), FT_TT_WATCHER_ENABLED);

    if (rc != FT_TT_OK)
        return rc;
    for (index = 0; index < FT_TT_MAX_NUM_NOCS_PER_CORE; index++) {
        uint16_t ok = FT_TT_SANITIZE_OK;
        uint32_t address = FT_TT_MB_WATCHER_SANITIZE_ADDR(index) +
                           (uint32_t)offsetof(ft_tt_debug_sanitize_addr_msg, return_code);

        /* return_code is 16 bits and this layer writes 32, so read-modify-write the word
         * it shares with which_risc rather than taking its neighbour with it. */
        uint32_t word = 0;

        rc = noc_read_u32(noc, x, y, address & ~3u, &word);
        if (rc != FT_TT_OK)
            return rc;
        if ((address & 3u) == 0u)
            word = (word & 0xFFFF0000u) | ok;
        else
            word = (word & 0x0000FFFFu) | ((uint32_t)ok << 16);
        rc = noc_write_u32(noc, x, y, address & ~3u, word);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

int ft_tt_launch_write_core_info(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                 const ft_tt_launch_core_info_config *config)
{
    ft_tt_core_info_msg message;
    int rc = ft_tt_launch_encode_core_info(config, &message, sizeof(message));

    if (rc != FT_TT_OK)
        return rc;
    return noc_write(noc, x, y, FT_TT_MB_CORE_INFO_ADDR, &message, sizeof(message));
}

/* ------------------------------------------------------------------------- */
/* Staging and kick                                                          */
/* ------------------------------------------------------------------------- */

int ft_tt_launch_clear_mailbox(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    static const uint32_t zeros[128] = {0};
    uint32_t address = FT_TT_MEM_MAILBOX_BASE;
    uint32_t end = FT_TT_MEM_MAILBOX_END;

    /* The region base is 0x60 and its length is not a multiple of the chunk, so walk
     * it explicitly rather than assuming either. */
    while (address < end) {
        size_t chunk = sizeof(zeros);
        int rc;

        if (chunk > (size_t)(end - address))
            chunk = (size_t)(end - address) & ~(size_t)3u;
        if (chunk == 0u)
            break;
        rc = noc_write(noc, x, y, address, zeros, chunk);
        if (rc != FT_TT_OK)
            return rc;
        address += (uint32_t)chunk;
    }
    return FT_TT_OK;
}

/** Read-modify-write one byte of a go message without disturbing its neighbours. */
static int write_go_signal(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint8_t signal)
{
    ft_tt_go_msg message;
    int rc = noc_read(noc, x, y, FT_TT_MB_GO_SLOT_ADDR(0), &message, sizeof(message));

    if (rc != FT_TT_OK)
        return rc;
    message.signal = signal;
    return noc_write(noc, x, y, FT_TT_MB_GO_SLOT_ADDR(0), &message, sizeof(message));
}

int ft_tt_launch_prime_firmware_handshake(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    return write_go_signal(noc, x, y, FT_TT_RUN_MSG_INIT);
}

int ft_tt_launch_send_go(const ft_tt_noc_access *noc, uint16_t x, uint16_t y)
{
    return write_go_signal(noc, x, y, FT_TT_RUN_MSG_GO);
}

int ft_tt_launch_plant_runtime_args(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                    uint32_t l1_address, const uint32_t *args, unsigned count)
{
    if (!args || count == 0u)
        return FT_TT_EINVAL;
    if ((l1_address & 3u) != 0u) {
        ft_tt_log_error("launch: runtime-arg address 0x%x is not 4-byte aligned", l1_address);
        return FT_TT_EINVAL;
    }
    return noc_write(noc, x, y, l1_address, args, (size_t)count * sizeof(uint32_t));
}

int ft_tt_launch_write_launch_message(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                      const ft_tt_launch_config *config)
{
    ft_tt_kernel_config_msg message;
    int rc = ft_tt_launch_encode_launch_message(config, &message, sizeof(message));

    if (rc != FT_TT_OK)
        return rc;
    return noc_write(noc, x, y, FT_TT_MB_LAUNCH_SLOT_ADDR(0), &message, sizeof(message));
}

/* ------------------------------------------------------------------------- */
/* Multi-core placement                                                      */
/* ------------------------------------------------------------------------- */

int ft_tt_launch_core_set_init(ft_tt_launch_core_set *set, ft_tt_launch_core_placement *storage,
                               unsigned capacity)
{
    if (!set || !storage || capacity == 0u)
        return FT_TT_EINVAL;

    set->cores = storage;
    set->capacity = capacity;
    set->count = 0u;
    return FT_TT_OK;
}

unsigned ft_tt_launch_core_set_capacity(const ft_tt_launch_core_set *set)
{
    return (set && set->cores) ? set->capacity : 0u;
}

void ft_tt_launch_core_set_clear(ft_tt_launch_core_set *set)
{
    if (set)
        set->count = 0u;
}

unsigned ft_tt_launch_core_set_count(const ft_tt_launch_core_set *set)
{
    return set ? set->count : 0u;
}

const ft_tt_launch_core_placement *ft_tt_launch_core_set_at(const ft_tt_launch_core_set *set,
                                                            unsigned index)
{
    if (!set || index >= set->count)
        return NULL;
    return &set->cores[index];
}

/** Encode a config into a throwaway buffer purely to find out whether it is legal. */
static int validate_launch_config(const ft_tt_launch_config *config)
{
    ft_tt_kernel_config_msg probe;

    return ft_tt_launch_encode_launch_message(config, &probe, sizeof(probe));
}

int ft_tt_launch_core_set_add(ft_tt_launch_core_set *set, uint16_t x, uint16_t y,
                              const ft_tt_launch_config *config)
{
    unsigned index;
    int rc;

    if (!set)
        return FT_TT_EINVAL;
    /* An unbound set has no storage. Refusing here is what keeps a forgotten
     * ft_tt_launch_core_set_init() a returned error instead of a write through NULL. */
    if (!set->cores || set->capacity == 0u) {
        ft_tt_log_error("launch: this placement group was never given storage — call "
                        "ft_tt_launch_core_set_init() with an array sized from the LIVE core "
                        "count");
        return FT_TT_EINVAL;
    }
    if (set->count >= set->capacity) {
        ft_tt_log_error("launch: this placement group holds at most %u cores", set->capacity);
        return FT_TT_ENOMEM;
    }

    for (index = 0; index < set->count; index++) {
        if (set->cores[index].x == x && set->cores[index].y == y) {
            ft_tt_log_error("launch: (%u, %u) is already in this placement group at slot %u; the "
                            "second launch message would overwrite the first",
                            (unsigned)x, (unsigned)y, index);
            return FT_TT_EINVAL;
        }
    }

    /* Encode now so a group cannot be half-staged before a bad field surfaces. */
    if (config) {
        rc = validate_launch_config(config);
        if (rc != FT_TT_OK)
            return rc;
    }

    set->cores[set->count].x = x;
    set->cores[set->count].y = y;
    if (config)
        set->cores[set->count].config = *config;
    else
        memset(&set->cores[set->count].config, 0, sizeof(set->cores[set->count].config));
    set->count++;
    return FT_TT_OK;
}

int ft_tt_launch_core_set_configure(ft_tt_launch_core_set *set, unsigned index,
                                    const ft_tt_launch_config *config)
{
    int rc;

    if (!set || !config || index >= set->count)
        return FT_TT_EINVAL;

    rc = validate_launch_config(config);
    if (rc != FT_TT_OK)
        return rc;

    set->cores[index].config = *config;
    return FT_TT_OK;
}

int ft_tt_launch_core_set_add_tensix(ft_tt_launch_core_set *set, const ft_tt_grid_info *grid,
                                     uint32_t tensix_index, const ft_tt_launch_config *config)
{
    ft_tt_noc_coord core;
    ft_tt_core_type type;
    int rc;

    if (!set || !grid)
        return FT_TT_EINVAL;

    rc = ft_tt_coord_tensix_by_index(grid, tensix_index, FT_TT_COORD_TRANSLATED, &core);
    if (rc != FT_TT_OK)
        return rc;

    /* Derived is not the same as verified. Classifying the coordinate before it can
     * reach a launch is the whole safety margin here (spec §7.3). */
    type = ft_tt_coord_classify(grid, FT_TT_COORD_TRANSLATED, core.x, core.y);
    if (type != FT_TT_CORE_TENSIX) {
        ft_tt_log_error("launch: Tensix index %u derives to translated (%u, %u), which classifies "
                        "as %s — refusing to launch on it",
                        tensix_index, core.x, core.y, ft_tt_core_type_string(type));
        return FT_TT_EINVAL;
    }
    if (core.x > UINT16_MAX || core.y > UINT16_MAX)
        return FT_TT_EINVAL;

    return ft_tt_launch_core_set_add(set, (uint16_t)core.x, (uint16_t)core.y, config);
}

/** Run `operation` over every core, stopping at the first failure. */
static int for_each_core(const ft_tt_noc_access *noc, const ft_tt_launch_core_set *set,
                         int (*operation)(const ft_tt_noc_access *, const ft_tt_launch_core_placement *,
                                          uint32_t),
                         uint32_t argument, unsigned *out_failed_index)
{
    unsigned index;

    if (!noc || !set)
        return FT_TT_EINVAL;

    for (index = 0; index < set->count; index++) {
        int rc = operation(noc, &set->cores[index], argument);

        if (rc != FT_TT_OK) {
            if (out_failed_index)
                *out_failed_index = index;
            return rc;
        }
    }
    return FT_TT_OK;
}

static int write_one_launch_message(const ft_tt_noc_access *noc,
                                    const ft_tt_launch_core_placement *core, uint32_t unused)
{
    (void)unused;
    return ft_tt_launch_write_launch_message(noc, core->x, core->y, &core->config);
}

static int send_one_go(const ft_tt_noc_access *noc, const ft_tt_launch_core_placement *core,
                       uint32_t unused)
{
    (void)unused;
    return ft_tt_launch_send_go(noc, core->x, core->y);
}

static int wait_one_done(const ft_tt_noc_access *noc, const ft_tt_launch_core_placement *core,
                         uint32_t timeout_ms)
{
    return ft_tt_launch_wait_done(noc, core->x, core->y, timeout_ms, NULL);
}

int ft_tt_launch_write_launch_messages(const ft_tt_noc_access *noc,
                                       const ft_tt_launch_core_set *set,
                                       unsigned *out_failed_index)
{
    return for_each_core(noc, set, write_one_launch_message, 0u, out_failed_index);
}

int ft_tt_launch_send_go_all(const ft_tt_noc_access *noc, const ft_tt_launch_core_set *set,
                             unsigned *out_failed_index)
{
    return for_each_core(noc, set, send_one_go, 0u, out_failed_index);
}

int ft_tt_launch_wait_done_all(const ft_tt_noc_access *noc, const ft_tt_launch_core_set *set,
                               uint32_t timeout_ms, unsigned *out_failed_index)
{
    return for_each_core(noc, set, wait_one_done, timeout_ms, out_failed_index);
}

int ft_tt_launch_park_all(const ft_tt_noc_access *noc, const ft_tt_launch_core_set *set)
{
    unsigned index;
    int first_error = FT_TT_OK;

    if (!noc || !set)
        return FT_TT_EINVAL;

    /* Deliberately not for_each_core(): this is the cleanup path, and one core that
     * cannot be reached must not leave the rest of the group running. */
    for (index = 0; index < set->count; index++) {
        int rc = ft_tt_launch_assert_reset(noc, set->cores[index].x, set->cores[index].y,
                                           FT_TT_SOFT_RESET_ALL_RISCS);

        if (rc != FT_TT_OK && first_error == FT_TT_OK)
            first_error = rc;
    }
    return first_error;
}

/* ------------------------------------------------------------------------- */
/* Completion                                                                */
/* ------------------------------------------------------------------------- */

static uint64_t monotonic_milliseconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * 1000u + (uint64_t)(now.tv_nsec / 1000000L);
}

int ft_tt_launch_read_state(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            ft_tt_launch_state *out_state)
{
    uint32_t slot;
    int rc;

    if (!out_state)
        return FT_TT_EINVAL;
    memset(out_state, 0, sizeof(*out_state));

    rc = ft_tt_launch_read_reset(noc, x, y, &out_state->reset_register);
    if (rc != FT_TT_OK)
        return rc;
    /* Same 4 KiB debug block as the reset register, so this costs nothing to reach — and it
     * is the one thing a read can say that a mailbox read cannot (ft_tt_launch.h).
     * BEST EFFORT on purpose: this is a diagnostic snapshot, and giving up on the whole
     * mailbox because a debug register did not answer would lose the half that matters. */
    if (noc_read_u32(noc, x, y, FT_TT_TENSIX_INSTRN_BUF_STATUS_ADDR,
                     &out_state->instrn_buf_status) != FT_TT_OK)
        out_state->instrn_buf_status = FT_TT_TENSIX_INSTRN_BUF_UNREAD;
    if (noc_read_u32(noc, x, y, FT_TT_TENSIX_DBG_INVALID_INSTRN_ADDR,
                     &out_state->invalid_instrn) != FT_TT_OK)
        out_state->invalid_instrn = FT_TT_TENSIX_INSTRN_BUF_UNREAD;

    rc = noc_read(noc, x, y, FT_TT_MB_SUBORDINATE_SYNC_ADDR, out_state->subordinate_sync,
                  sizeof(out_state->subordinate_sync));
    if (rc != FT_TT_OK)
        return rc;
    rc = noc_read_u32(noc, x, y, FT_TT_MB_LAUNCH_RD_PTR_ADDR, &out_state->launch_msg_rd_ptr);
    if (rc != FT_TT_OK)
        return rc;
    rc = noc_read_u32(noc, x, y, FT_TT_MB_GO_MESSAGE_INDEX_ADDR, &out_state->go_message_index);
    if (rc != FT_TT_OK)
        return rc;
    rc = noc_read(noc, x, y, FT_TT_MB_GO_MESSAGES_ADDR, out_state->go_messages,
                  sizeof(out_state->go_messages));
    if (rc != FT_TT_OK)
        return rc;

    /* The read pointer comes from the device and may be garbage on a wedged core, so
     * clamp it rather than reading an arbitrary L1 address. */
    slot = out_state->launch_msg_rd_ptr % FT_TT_LAUNCH_MSG_BUFFER_NUM_ENTRIES;
    rc = noc_read(noc, x, y, FT_TT_MB_LAUNCH_SLOT_ADDR(slot), &out_state->kernel_config,
                  sizeof(out_state->kernel_config));
    if (rc != FT_TT_OK)
        return rc;

    return noc_read(noc, x, y, FT_TT_MB_WATCHER_ADDR, &out_state->watcher,
                    sizeof(out_state->watcher));
}

/** Four waypoint characters, or "----" when the firmware was built without watcher
 *  support and the bytes are all zero. */
static void format_waypoint(const uint8_t *bytes, char out[5])
{
    unsigned index;

    for (index = 0; index < 4u; index++)
        out[index] = (bytes[index] >= 0x20u && bytes[index] < 0x7Fu) ? (char)bytes[index] : '-';
    out[4] = '\0';
}

void ft_tt_launch_dump_state(const ft_tt_launch_state *state, uint16_t x, uint16_t y)
{
    unsigned processor;

    if (!state)
        return;

    ft_tt_log_error("--- Tensix (%u, %u) run state ---", (unsigned)x, (unsigned)y);
    ft_tt_log_error("  soft reset 0x%08x: BRISC=%s NCRISC=%s TRISC0=%s TRISC1=%s TRISC2=%s",
                    state->reset_register,
                    (state->reset_register & FT_TT_SOFT_RESET_BRISC) ? "held" : "running",
                    (state->reset_register & FT_TT_SOFT_RESET_NCRISC) ? "held" : "running",
                    (state->reset_register & FT_TT_SOFT_RESET_TRISC0) ? "held" : "running",
                    (state->reset_register & FT_TT_SOFT_RESET_TRISC1) ? "held" : "running",
                    (state->reset_register & FT_TT_SOFT_RESET_TRISC2) ? "held" : "running");
    /* subordinate_sync is [NCRISC, TRISC0, TRISC1, TRISC2] — NCRISC first (ft_tt_mailbox.h,
     * fw/ft_brisc_fw.c). 0x00 = done, 0x80 = go, 0x40 = init, 1 = load, 3 = reset streams. */
    ft_tt_log_error("  subordinate sync NCRISC=0x%02x TRISC0=0x%02x TRISC1=0x%02x TRISC2=0x%02x",
                    state->subordinate_sync[0], state->subordinate_sync[1],
                    state->subordinate_sync[2], state->subordinate_sync[3]);
    ft_tt_log_error("  go[0].signal=0x%02x  go_message_index=%u  launch_msg_rd_ptr=%u",
                    state->go_messages[0].signal, state->go_message_index,
                    state->launch_msg_rd_ptr);
    ft_tt_log_error("  instrn_buf_status=0x%08x (%s)  invalid_instrn=0x%08x",
                    state->instrn_buf_status,
                    state->instrn_buf_status == FT_TT_TENSIX_INSTRN_BUF_READY
                        ? "ready — a boot can push into the Tensix FIFO"
                        : (state->instrn_buf_status == FT_TT_TENSIX_INSTRN_BUF_UNREAD
                               ? "unread"
                               : "NOT READY — BRISC is stalled pushing into the Tensix FIFO"),
                    state->invalid_instrn);
    ft_tt_log_error("  launch: enables=0x%08x mode=%u config_base=0x%08x cb_offset=0x%04x",
                    state->kernel_config.enables, state->kernel_config.mode,
                    state->kernel_config.kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX],
                    state->kernel_config.local_cb_offset);

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        char waypoint[5];

        format_waypoint(state->watcher.debug_waypoint[processor].waypoint, waypoint);
        ft_tt_log_error("  %-6s text_offset=0x%08x size=%u waypoint=%s",
                        ft_tt_processor_name(processor),
                        state->kernel_config.kernel_text_offset[processor],
                        state->kernel_config.kernel_text_size[processor], waypoint);
    }
    ft_tt_log_error("  (waypoints read '----' unless the firmware was built with "
                    "WATCHER_ENABLED)");

    /* The sanitize slots are the reason a hang can be a report. Print them only when one
     * is populated: without WATCHER_ENABLED / with WATCHER_DISABLE_NOC_SANITIZE the
     * firmware never writes here, and two lines of zeros every timeout is noise. */
    for (processor = 0; processor < FT_TT_MAX_NUM_NOCS_PER_CORE; processor++) {
        const ft_tt_debug_sanitize_addr_msg *report = &state->watcher.sanitize[processor];

        if (report->return_code == FT_TT_SANITIZE_NO_REPORT)
            continue;
        ft_tt_log_error("  NoC %u SANITIZE: %s (code %u) — %s %s at noc_addr 0x%016llx, "
                        "local L1 0x%08x, %u bytes, risc %u%s", processor,
                        ft_tt_sanitize_return_code_string(report->return_code),
                        report->return_code, report->is_write ? "write" : "read",
                        report->is_target ? "target" : "local buffer",
                        (unsigned long long)report->noc_addr, report->l1_addr, report->len,
                        report->which_risc, report->is_multicast ? ", multicast" : "");
        ft_tt_log_error("    the transaction was NOT issued — sanitize ran before the "
                        "command register write");
    }
}

/**
 * Poll one 4-byte go message until it reads `target`, bounded by a wall-clock
 * timeout. Shared by the firmware-boot handshake and the kernel-completion wait,
 * which differ only in the signal they start from.
 */
static int poll_go_signal(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint8_t pending,
                          uint32_t timeout_ms, const char *what, ft_tt_launch_state *out_state)
{
    uint64_t deadline;

    if (timeout_ms == 0u) {
        ft_tt_log_error("launch: %s was asked to wait forever; refusing (a hang here costs a "
                        "power cycle)", what);
        return FT_TT_EINVAL;
    }

    deadline = monotonic_milliseconds() + timeout_ms;
    for (;;) {
        ft_tt_go_msg message;
        int rc = noc_read(noc, x, y, FT_TT_MB_GO_SLOT_ADDR(0), &message, sizeof(message));

        if (rc != FT_TT_OK)
            return rc;
        if (message.signal == FT_TT_RUN_MSG_DONE)
            return FT_TT_OK;
        if (message.signal != pending) {
            ft_tt_log_error("launch: %s read go signal 0x%02x, expected 0x%02x or done (0x%02x) "
                            "— the mailbox has been overwritten",
                            what, message.signal, pending, FT_TT_RUN_MSG_DONE);
            return FT_TT_ERROR;
        }
        if (monotonic_milliseconds() >= deadline) {
            ft_tt_launch_state state;

            ft_tt_log_error("launch: %s timed out after %u ms on Tensix (%u, %u)", what, timeout_ms,
                            (unsigned)x, (unsigned)y);
            if (ft_tt_launch_read_state(noc, x, y, &state) == FT_TT_OK) {
                ft_tt_launch_dump_state(&state, x, y);
                if (out_state)
                    *out_state = state;
            } else {
                ft_tt_log_error("launch: could not read the core's state either");
            }
            return FT_TT_EIO;
        }
    }
}

int ft_tt_launch_wait_firmware_ready(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                     uint32_t timeout_ms)
{
    return poll_go_signal(noc, x, y, FT_TT_RUN_MSG_INIT, timeout_ms, "firmware boot", NULL);
}

int ft_tt_launch_wait_done(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t timeout_ms,
                           ft_tt_launch_state *out_state)
{
    return poll_go_signal(noc, x, y, FT_TT_RUN_MSG_GO, timeout_ms, "kernel completion", out_state);
}
