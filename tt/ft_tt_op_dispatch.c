/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_op_dispatch.c
 * @brief Implementation of Phase 4 op dispatch: circular-buffer placement, runtime args
 *   and the kick. Image loading is ft_tt_launch.c's — see the header for why it moved.
 *
 * @status Loads and runs the relocated matmul and rms_norm programs on the card.
 * @issues None
 * @todo None
 */

#include "ft_tt_op_dispatch.h"

#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"

int ft_tt_op_image_load(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, unsigned processor,
                        const void *elf_data, size_t elf_size, ft_tt_image_placement *out_place)
{
    return ft_tt_launch_load_image(noc, x, y, processor, FT_TT_IMAGE_KERNEL, elf_data, elf_size,
                                   out_place);
}

int ft_tt_op_program_load(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                          const ft_tt_kernel_program *program,
                          ft_tt_image_placement placements[FT_TT_PROC_COUNT],
                          ft_tt_launch_config *out_config)
{
    ft_tt_image_placement local[FT_TT_PROC_COUNT];
    unsigned index;
    int rc;

    if (!noc || !program || !out_config)
        return FT_TT_EINVAL;

    memset(local, 0, sizeof(local));
    memset(out_config, 0, sizeof(*out_config));

    for (index = 0; index < program->image_count; index++) {
        const ft_tt_kernel_image *image = &program->images[index];

        if (image->processor >= FT_TT_PROC_COUNT)
            return FT_TT_EINVAL;
        rc = ft_tt_op_image_load(noc, x, y, image->processor, image->elf_data, image->elf_size,
                                 &local[image->processor]);
        if (rc != FT_TT_OK)
            return rc;
    }

    out_config->kernel_config_base = FT_TT_OP_KERNEL_CONFIG_BASE;
    out_config->enables = ft_tt_kernel_program_enables(program);

    for (index = 0; index < FT_TT_PROC_COUNT; index++) {
        if (local[index].bytes_written == 0u)
            continue;
        out_config->kernel_text_address[index] = local[index].entry_address;
        out_config->kernel_text_bytes[index] = local[index].text_bytes;
        /* Every ENABLED processor needs a real runtime-arg block even when it reads none:
         * a stale block or the 0xFFFF sentinel resolves to an address the kernel then
         * dereferences, which stalled a core once already. */
        out_config->runtime_args_address[index] = FT_TT_OP_RTA_ADDRESS(index);
    }

    if (placements)
        memcpy(placements, local, sizeof(local));
    return FT_TT_OK;
}

/* ================================================================
 * Operands and block depth
 * ================================================================ */

uint32_t ft_tt_op_block_depth(uint32_t run, uint32_t requested)
{
    uint32_t depth;

    if (run == 0u)
        return 1u;
    depth = (requested == 0u) ? 1u : requested;
    if (depth > run)
        depth = run;
    while (depth > 1u && run % depth != 0u)
        depth--;
    return depth;
}

/* ================================================================
 * Circular buffers
 * ================================================================ */

int ft_tt_op_cb_place(const ft_tt_op_cb_request *requests, unsigned count,
                      ft_tt_launch_cb_entry *out_entries, void *buffer, size_t buffer_bytes,
                      uint64_t *out_mask)
{
    uint32_t cursor = FT_TT_OP_CB_ARENA_BASE;
    unsigned index;

    if (!requests || !out_entries || !buffer || !out_mask || count == 0u ||
        count > FT_TT_OP_MAX_CB)
        return FT_TT_EINVAL;

    for (index = 0; index < count; index++) {
        const ft_tt_op_cb_request *request = &requests[index];
        uint32_t bytes;

        if (request->page_bytes == 0u || request->page_count == 0u)
            return FT_TT_EINVAL;
        bytes = request->page_bytes * request->page_count;
        if (bytes / request->page_bytes != request->page_count)
            return FT_TT_EINVAL;
        if (FT_TT_OP_CB_ARENA_END - cursor < bytes) {
            ft_tt_log_error("op cb: CB %u needs %u bytes and the arena has %u left",
                            request->index, bytes, FT_TT_OP_CB_ARENA_END - cursor);
            return FT_TT_ENOMEM;
        }

        out_entries[index].index = request->index;
        out_entries[index].address = cursor;
        out_entries[index].size_bytes = bytes;
        out_entries[index].page_size_bytes = request->page_bytes;
        out_entries[index].page_count = request->page_count;

        cursor = (cursor + bytes + 63u) & ~63u;
    }

    return ft_tt_launch_encode_cb_config(out_entries, count, buffer, buffer_bytes, out_mask);
}

const ft_tt_launch_cb_entry *ft_tt_op_cb_find(const ft_tt_launch_cb_entry *entries,
                                              unsigned count, unsigned index)
{
    unsigned position;

    if (!entries)
        return NULL;
    for (position = 0; position < count; position++) {
        if (entries[position].index == index)
            return &entries[position];
    }
    return NULL;
}

/* ================================================================
 * Runtime args and the kick
 * ================================================================ */

/** The args for `processor`, or NULL when the op named none. */
static const ft_tt_op_args *args_for(const ft_tt_op_args *args, unsigned arg_count,
                                     unsigned processor)
{
    unsigned index;

    for (index = 0; index < arg_count; index++) {
        if (args[index].processor == processor)
            return &args[index];
    }
    return NULL;
}

int ft_tt_op_stage(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                   const ft_tt_launch_config *config, const void *cb_buffer, size_t cb_bytes,
                   const ft_tt_op_args *args, unsigned arg_count)
{
    static const uint32_t no_args[1] = {0u};
    unsigned processor;
    int rc;

    if (!noc || !config || !cb_buffer)
        return FT_TT_EINVAL;

    rc = noc->write(noc->user, x, y, (uint64_t)config->local_cb_config_address, cb_buffer,
                    cb_bytes);
    if (rc != FT_TT_OK)
        return rc;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        const ft_tt_op_args *planned;

        if ((config->enables & (1u << processor)) == 0u)
            continue;
        planned = args_for(args, arg_count, processor);
        rc = ft_tt_launch_plant_runtime_args(noc, x, y, config->runtime_args_address[processor],
                                             planned ? planned->values : no_args,
                                             planned ? planned->count : 1u);
        if (rc != FT_TT_OK)
            return rc;
    }

    return ft_tt_launch_write_launch_message(noc, x, y, config);
}

int ft_tt_op_launch(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                    const ft_tt_launch_config *config, const void *cb_buffer, size_t cb_bytes,
                    const ft_tt_op_args *args, unsigned arg_count, uint32_t timeout_ms,
                    ft_tt_launch_state *out_state)
{
    int rc = ft_tt_op_stage(noc, x, y, config, cb_buffer, cb_bytes, args, arg_count);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_launch_send_go(noc, x, y);
    if (rc != FT_TT_OK)
        return rc;
    return ft_tt_launch_wait_done(noc, x, y, timeout_ms, out_state);
}
