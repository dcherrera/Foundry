/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_metal_internal.h
 * @brief Contract between the fm_metal core (device, pipeline cache, command
 *   buffer, dispatch) and the per-family kernel units under metal/kernels/.
 *   Objective-C only — never include this from a C translation unit or from
 *   the public headers; C callers use metal/fm_metal.h.
 *
 * @status Complete for the kernels that exist; extend it only when a kernel
 *   family genuinely needs something the core already owns.
 * @issues None
 * @todo None
 */

#ifndef FOUNDRY_METAL_INTERNAL_H
#define FOUNDRY_METAL_INTERNAL_H

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/fm_metal.h"

/* ================================================================
 * fm_buffer internals
 *
 * The kernel units need the MTLBuffer to bind it, so the struct is
 * shared here rather than duplicated.  Everything else about a buffer
 * goes through the public fm_buffer_* API.
 * ================================================================ */

struct fm_buffer {
    id<MTLBuffer> mtl;     /* Metal buffer object */
    size_t        size;    /* Size in bytes */
    bool          owns;    /* True if we allocated the underlying memory */
};

/* ================================================================
 * Device access
 *
 * Accessors, not extern globals: the singleton stays private to
 * fm_metal.m so a kernel unit cannot reassign it, and the reserved
 * leading-underscore names never leave that file.
 * ================================================================ */

/* The shared MTLDevice, or nil before fm_init(). */
id<MTLDevice> fm_metal_device(void);

/* True between a successful fm_init() and fm_free(). */
bool fm_metal_is_initialized(void);

/* ================================================================
 * Pipeline cache
 * ================================================================ */

/* Pipeline for `name` specialized with `constants` (NULL = unspecialized,
 * which still goes through constantValues: so shader defaults apply).
 * The cache owns the result; nil after logging if it cannot be built. */
id<MTLComputePipelineState> fm_get_pipeline_specialized(NSString *name,
                                                        const fm_fc_values *constants);

/* fm_get_pipeline_specialized(name, NULL). */
id<MTLComputePipelineState> fm_get_pipeline(NSString *name);

/* ================================================================
 * Command buffer and submission
 *
 * Every op encodes into fm_get_command_buffer() and finishes with
 * fm_submit_if_not_batching(); the batching mode lives entirely in
 * fm_metal.m so no kernel unit has to reason about it.
 * ================================================================ */

/* The command buffer to encode into (explicit batch, or the persistent one). */
id<MTLCommandBuffer> fm_get_command_buffer(void);

/* Commits and waits in immediate mode; accumulates in either batch mode. */
void fm_submit_if_not_batching(void);

/* ================================================================
 * Dispatch helpers
 *
 * `constants` here are ordinary runtime buffer arguments bound after the
 * buffers — unrelated to the function constants in `fc`, which are bound
 * when the pipeline is built.
 * ================================================================ */

/* Largest number of buffer bindings any kernel here takes. */
#define FM_MAX_BINDINGS 12

/* 1-D dispatch over `total_threads`, operands bound at explicit offsets. */
void fm_dispatch_1d_refs(NSString *kernel_name, const fm_fc_values *fc,
                         const fm_bufref *refs, int n_refs,
                         uint32_t *constants, int n_constants,
                         int total_threads);

/* 1-D dispatch of a specialized pipeline over whole buffers (offset 0). */
void fm_dispatch_1d_specialized(NSString *kernel_name, const fm_fc_values *fc,
                                fm_buffer **buffers, int n_buffers,
                                uint32_t *constants, int n_constants,
                                int total_threads);

/* fm_dispatch_1d_specialized with no function constants. */
void fm_dispatch_1d(NSString *kernel_name,
                    fm_buffer **buffers, int n_buffers,
                    uint32_t *constants, int n_constants,
                    int total_threads);

/* Threadgroup dispatch binding each operand at its own byte offset — the
 * path a weight living inside a shared mmap region takes. */
void fm_dispatch_threadgroups_refs(NSString *kernel_name, const fm_fc_values *fc,
                                   const fm_bufref *refs, int n_refs,
                                   uint32_t *constants, int n_constants,
                                   int num_groups, int threads_per_group);

/* Threadgroup dispatch over whole buffers (offset 0), unspecialized. */
void fm_dispatch_threadgroups(NSString *kernel_name,
                              fm_buffer **buffers, int n_buffers,
                              uint32_t *constants, int n_constants,
                              int num_groups, int threads_per_group);

/* Threadgroup dispatch with a trailing float argument (eps, x_scale, ...). */
void fm_dispatch_with_float_constant(NSString *kernel_name,
                                     fm_buffer **buffers, int n_buffers,
                                     uint32_t *uint_consts, int n_uint,
                                     float float_const,
                                     int num_groups, int threads_per_group);

#endif /* FOUNDRY_METAL_INTERNAL_H */
