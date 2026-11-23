/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_kernel_blob.c
 * @brief Implementation of the embedded-kernel-blob accessors. Two lookups over a
 *   small table; no allocation, no device contact.
 *
 * @status Compile-checked and unit-tested off-device.
 * @issues None
 * @todo None
 */

#include "ft_tt_kernel_blob.h"

#include <stddef.h>

const ft_tt_kernel_image *ft_tt_kernel_image_for_processor(const ft_tt_kernel_program *program,
                                                           unsigned processor)
{
    unsigned index;

    if (!program || !program->images)
        return NULL;

    for (index = 0; index < program->image_count; index++) {
        if (program->images[index].processor == processor)
            return &program->images[index];
    }
    return NULL;
}

unsigned ft_tt_kernel_program_enables(const ft_tt_kernel_program *program)
{
    unsigned mask = 0;
    unsigned index;

    if (!program || !program->images)
        return 0;

    for (index = 0; index < program->image_count; index++)
        mask |= 1u << program->images[index].processor;

    return mask;
}
