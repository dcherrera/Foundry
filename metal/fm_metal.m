/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file fm_metal.m
 * @brief Core of the Metal backend: the MTLDevice + MTLCommandQueue + shader
 *   library singleton, the function-constant pipeline cache, buffers and
 *   no-copy regions, command batching, and the dispatch helpers every kernel
 *   family encodes through. Unified memory (StorageModeShared) gives zero-copy
 *   on Apple Silicon.
 *
 *   The kernels themselves do NOT live here. Each family owns one unit under
 *   metal/kernels/ holding both its host dispatch and its MSL source fragment;
 *   fm_kernel_library_source() concatenates the fragments into the single
 *   source compiled once at fm_init(). See metal/kernels/fm_kernel_registry.h.
 *
 * @status Buffers, no-copy mmap regions with offset binding, custom kernel
 *   dispatch and function-constant specialization all implemented and tested.
 * @issues None
 * @todo None
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal/fm_metal.h"
#include "metal/fm_metal_internal.h"
#include "metal/kernels/fm_kernel_registry.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  /* getpagesize */

/* ================================================================
 * Global singleton state
 * ================================================================ */

static id<MTLDevice>       _fm_device       = nil;
static id<MTLCommandQueue> _fm_queue        = nil;
static id<MTLLibrary>      _fm_library      = nil;
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>> *_fm_pipelines = nil;
static bool                _fm_initialized  = false;
static long                _fm_page_size    = 0;

/* Batching state */
static id<MTLCommandBuffer>         _fm_batch_cmdbuf  = nil;
static bool                         _fm_batching      = false;

/* MLX-style persistent command buffer: accumulates GPU work until fm_sync() */
static id<MTLCommandBuffer>         _fm_persistent_cmdbuf = nil;
static int                          _fm_pending_ops       = 0;

/* Auto-batch mode: when true, dispatches accumulate until fm_sync().
 * When false (default), each dispatch commits and waits immediately. */
static bool                         _fm_auto_batch        = false;

/* ================================================================
 * Lifecycle
 * ================================================================ */

int fm_init(void) {
    if (_fm_initialized) return 0;

    @autoreleasepool {
        _fm_device = MTLCreateSystemDefaultDevice();
        if (!_fm_device) return -1;

        _fm_queue = [_fm_device newCommandQueue];
        if (!_fm_queue) {
            _fm_device = nil;
            return -1;
        }

        _fm_pipelines = [NSMutableDictionary new];
        _fm_page_size = getpagesize();

        /* Compile shader library */
        NSString *source = fm_kernel_library_source();
        if (source && [source length] > 0) {
            NSError *error = nil;
            MTLCompileOptions *opts = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                opts.mathMode = MTLMathModeFast;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }
            _fm_library = [_fm_device newLibraryWithSource:source
                                                  options:opts
                                                    error:&error];
            if (error && !_fm_library) {
                /* Non-fatal by design — MPS ops still work without custom kernels —
                 * but it must be LOUD. Every family's MSL is one translation unit, so a
                 * syntax error in any one of them takes out every custom kernel in the
                 * backend at once: they silently become no-ops and output buffers keep
                 * whatever they held. That reads as "my kernel is wrong", not "the
                 * library didn't build", and it makes GPU-vs-GPU comparisons pass by
                 * agreeing on all-zeros. Callers gate on fm_shaders_available(). */
                NSLog(@"[fm] ***** SHADER COMPILATION FAILED — ALL CUSTOM KERNELS ARE "
                      @"DISABLED AND WILL SILENTLY DO NOTHING *****");
                NSLog(@"[fm] %@", error);
                fprintf(stderr,
                        "[fm] ***** SHADER COMPILATION FAILED *****\n"
                        "[fm] Every custom Metal kernel is now a no-op; results will be\n"
                        "[fm] whatever the output buffers already contained. Fix the MSL\n"
                        "[fm] before trusting ANY measurement or test result.\n");
            }
        }

        _fm_initialized = true;
    }
    return 0;
}

void fm_free(void) {
    if (!_fm_initialized) return;

    /* Flush any pending GPU work */
    fm_sync();

    @autoreleasepool {
        _fm_persistent_cmdbuf = nil;
        _fm_pending_ops = 0;
        _fm_auto_batch = false;
        _fm_batch_cmdbuf = nil;
        _fm_batching = false;
        _fm_pipelines = nil;
        _fm_library = nil;
        _fm_queue = nil;
        _fm_device = nil;
        _fm_initialized = false;
    }
}

bool fm_available(void) {
    return _fm_initialized;
}

bool fm_shaders_available(void) {
    return _fm_initialized && _fm_library != nil;
}

/* ================================================================
 * Device access for the kernel units (metal/fm_metal_internal.h)
 * ================================================================ */

id<MTLDevice> fm_metal_device(void) {
    return _fm_device;
}

bool fm_metal_is_initialized(void) {
    return _fm_initialized;
}

/* ================================================================
 * Function-constant value sets
 *
 * Pure C, no Metal types: this is the value type callers build on the
 * stack and hand to the pipeline cache.  Entries stay sorted by index so
 * the cache key does not depend on the order they were set in.
 * ================================================================ */

void fm_fc_reset(fm_fc_values *values) {
    if (!values) return;
    memset(values, 0, sizeof(*values));
}

/* Inserts or replaces the entry for `index`, keeping the array sorted.
 * Returns 0 on success, -1 if the set is full or the arguments are bad. */
static int fm_fc_set_value(fm_fc_values *values, unsigned index,
                           fm_fc_type type, int32_t value) {
    if (!values || index > UINT16_MAX) return -1;
    if (values->count < 0 || values->count > FM_FC_MAX_VALUES) return -1;

    int slot = 0;
    while (slot < values->count && values->index[slot] < index)
        slot++;

    /* Same index set twice: overwrite in place rather than duplicating,
     * so the key stays canonical and Metal never sees a repeated index. */
    if (slot < values->count && values->index[slot] == index) {
        values->type[slot]  = (uint8_t)type;
        values->value[slot] = value;
        return 0;
    }

    if (values->count == FM_FC_MAX_VALUES) return -1;

    for (int i = values->count; i > slot; i--) {
        values->index[i] = values->index[i - 1];
        values->type[i]  = values->type[i - 1];
        values->value[i] = values->value[i - 1];
    }
    values->index[slot] = (uint16_t)index;
    values->type[slot]  = (uint8_t)type;
    values->value[slot] = value;
    values->count++;
    return 0;
}

int fm_fc_set_int(fm_fc_values *values, unsigned index, int32_t value) {
    return fm_fc_set_value(values, index, FM_FC_TYPE_INT, value);
}

int fm_fc_set_short(fm_fc_values *values, unsigned index, int16_t value) {
    return fm_fc_set_value(values, index, FM_FC_TYPE_SHORT, (int32_t)value);
}

int fm_fc_set_bool(fm_fc_values *values, unsigned index, bool value) {
    return fm_fc_set_value(values, index, FM_FC_TYPE_BOOL, value ? 1 : 0);
}

/* ================================================================
 * Pipeline caching (specialized by function constants)
 * ================================================================ */

/* Deterministic cache key: the kernel name for an unspecialized build,
 * otherwise the name plus every (index, type, value) triple in index
 * order.  Equal constant sets always produce an equal key, so identical
 * requests hit the same cached pipeline object. */
static NSString *fm_pipeline_cache_key(NSString *name, const fm_fc_values *constants) {
    if (!constants || constants->count == 0) return name;

    NSMutableString *key = [NSMutableString stringWithString:name];
    for (int i = 0; i < constants->count; i++) {
        [key appendFormat:@"|%u:%u=%d",
            (unsigned)constants->index[i],
            (unsigned)constants->type[i],
            (int)constants->value[i]];
    }
    return key;
}

/* Translates a constant set into the MTLFunctionConstantValues the
 * shader compiler wants.  The C type handed to Metal must match the
 * declared MSL type exactly, hence the per-tag narrowing.  A NULL or
 * empty set yields an empty object, which leaves every constant
 * undefined so the shader's is_function_constant_defined defaults win. */
static MTLFunctionConstantValues *fm_make_constant_values(const fm_fc_values *constants) {
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    if (!constants) return cv;

    for (int i = 0; i < constants->count; i++) {
        NSUInteger idx = constants->index[i];
        switch ((fm_fc_type)constants->type[i]) {
        case FM_FC_TYPE_BOOL: {
            bool v = constants->value[i] != 0;
            [cv setConstantValue:&v type:MTLDataTypeBool atIndex:idx];
            break;
        }
        case FM_FC_TYPE_SHORT: {
            int16_t v = (int16_t)constants->value[i];
            [cv setConstantValue:&v type:MTLDataTypeShort atIndex:idx];
            break;
        }
        case FM_FC_TYPE_INT: {
            int32_t v = constants->value[i];
            [cv setConstantValue:&v type:MTLDataTypeInt atIndex:idx];
            break;
        }
        case FM_FC_TYPE_NONE:
        default:
            NSLog(@"[fm] Function constant at index %lu has no type, skipped",
                  (unsigned long)idx);
            break;
        }
    }
    return cv;
}

/* Builds the MTLFunction for `name`, specialized with `constants`.
 *
 * This always goes through the constantValues API, even for an empty
 * set: Metal refuses to build a pipeline from a plain
 * newFunctionWithName: result if the kernel declares any function
 * constant at all, and an empty set is exactly how a kernel is asked to
 * fall back to its is_function_constant_defined defaults.
 *
 * Returns nil (after logging) if the kernel is missing or a constant
 * does not match the shader's declaration. */
static id<MTLFunction> fm_new_function(NSString *name, const fm_fc_values *constants) {
    NSError *error = nil;
    id<MTLFunction> fn = [_fm_library newFunctionWithName:name
                                           constantValues:fm_make_constant_values(constants)
                                                    error:&error];
    if (!fn)
        NSLog(@"[fm] Kernel '%@' unavailable: %@", name, error);
    return fn;
}

id<MTLComputePipelineState> fm_get_pipeline_specialized(NSString *name,
                                                        const fm_fc_values *constants) {
    if (!_fm_library || !name) return nil;

    NSString *key = fm_pipeline_cache_key(name, constants);
    id<MTLComputePipelineState> ps = _fm_pipelines[key];
    if (ps) return ps;

    @autoreleasepool {
        id<MTLFunction> fn = fm_new_function(name, constants);
        if (!fn) return nil;

        NSError *error = nil;
        ps = [_fm_device newComputePipelineStateWithFunction:fn error:&error];
        if (error || !ps) {
            NSLog(@"[fm] Pipeline creation failed for '%@': %@", key, error);
            return nil;
        }
        _fm_pipelines[key] = ps;
    }
    return ps;
}

id<MTLComputePipelineState> fm_get_pipeline(NSString *name) {
    return fm_get_pipeline_specialized(name, NULL);
}

const void *fm_pipeline_for_constants(const char *kernel_name,
                                      const fm_fc_values *constants) {
    if (!_fm_initialized || !kernel_name) return NULL;

    NSString *name = [NSString stringWithUTF8String:kernel_name];
    /* __bridge, not __bridge_retained: the cache keeps the pipeline
     * alive and the caller must not release it. */
    return (__bridge const void *)fm_get_pipeline_specialized(name, constants);
}

/* ================================================================
 * Buffer management
 * ================================================================ */

fm_buffer* fm_buffer_from_ptr(void *ptr, size_t bytes) {
    if (!_fm_initialized || !ptr || bytes == 0) return NULL;

    fm_buffer *buf = (fm_buffer*)calloc(1, sizeof(fm_buffer));
    if (!buf) return NULL;

    @autoreleasepool {
        /* Check if pointer is page-aligned for zero-copy */
        bool page_aligned = ((uintptr_t)ptr % _fm_page_size) == 0;
        /* Also need page-aligned size for newBufferWithBytesNoCopy */
        size_t aligned_size = (bytes + _fm_page_size - 1) & ~(_fm_page_size - 1);

        if (page_aligned) {
            /* Zero-copy: wrap existing pointer */
            buf->mtl = [_fm_device newBufferWithBytesNoCopy:ptr
                                                    length:aligned_size
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
            if (!buf->mtl) {
                /* Fallback: copy into new buffer */
                buf->mtl = [_fm_device newBufferWithBytes:ptr
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
                buf->owns = true;
            } else {
                buf->owns = false;
            }
        } else {
            /* Not page-aligned: must copy */
            buf->mtl = [_fm_device newBufferWithBytes:ptr
                                              length:bytes
                                             options:MTLResourceStorageModeShared];
            buf->owns = true;
        }

        if (!buf->mtl) {
            free(buf);
            return NULL;
        }
        buf->size = bytes;
    }
    return buf;
}

fm_buffer* fm_buffer_alloc(size_t bytes) {
    if (!_fm_initialized || bytes == 0) return NULL;

    fm_buffer *buf = (fm_buffer*)calloc(1, sizeof(fm_buffer));
    if (!buf) return NULL;

    @autoreleasepool {
        buf->mtl = [_fm_device newBufferWithLength:bytes
                                           options:MTLResourceStorageModeShared];
        if (!buf->mtl) {
            free(buf);
            return NULL;
        }
        buf->size = bytes;
        buf->owns = true;
    }
    return buf;
}

void fm_buffer_free(fm_buffer *buf) {
    if (!buf) return;
    /* MTLBuffer is ARC-managed; setting to nil releases it */
    buf->mtl = nil;
    free(buf);
}

void* fm_buffer_data(fm_buffer *buf) {
    if (!buf || !buf->mtl) return NULL;
    return [buf->mtl contents];
}

size_t fm_buffer_size(fm_buffer *buf) {
    if (!buf) return 0;
    return buf->size;
}

/* ================================================================
 * No-copy regions and offset binding
 * ================================================================ */

fm_bufref fm_bufref_at(fm_buffer *buffer, size_t offset) {
    fm_bufref ref = { buffer, offset };
    return ref;
}

size_t fm_buffer_offset_alignment(void) {
    return FM_BUFFER_OFFSET_ALIGNMENT;
}

size_t fm_buffer_max_length(void) {
    if (!_fm_initialized) return 0;
    return (size_t)[_fm_device maxBufferLength];
}

static size_t fm_page_round_up(size_t bytes) {
    size_t page = (size_t)_fm_page_size;
    return (bytes + page - 1) & ~(page - 1);
}

fm_buffer *fm_buffer_map_no_copy(void *mapped_base, size_t mapped_bytes) {
    if (!_fm_initialized || !mapped_base || mapped_bytes == 0) return NULL;

    if (((uintptr_t)mapped_base % (uintptr_t)_fm_page_size) != 0) {
        NSLog(@"[fm] map_no_copy: base %p is not page-aligned — refusing "
              @"(a copy here would duplicate the mapping)", mapped_base);
        return NULL;
    }

    /* Safe to round up: mmap always maps whole pages, so the tail of the
     * final page belongs to this mapping even past mapped_bytes. */
    size_t length = fm_page_round_up(mapped_bytes);
    size_t max_length = (size_t)[_fm_device maxBufferLength];
    if (length > max_length) {
        NSLog(@"[fm] map_no_copy: %zu bytes exceeds maxBufferLength %zu — "
              @"use fm_region to cover it with segments", length, max_length);
        return NULL;
    }

    fm_buffer *buf = (fm_buffer *)calloc(1, sizeof(*buf));
    if (!buf) return NULL;

    buf->mtl = [_fm_device newBufferWithBytesNoCopy:mapped_base
                                             length:length
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
    if (!buf->mtl) {
        NSLog(@"[fm] map_no_copy: Metal refused a %zu byte mapping at %p",
              length, mapped_base);
        free(buf);
        return NULL;
    }

    /* Paranoia, cheap: the whole point is that this aliases the mapping. */
    if ([buf->mtl contents] != mapped_base) {
        NSLog(@"[fm] map_no_copy: Metal returned a copy, not an alias — refusing");
        buf->mtl = nil;
        free(buf);
        return NULL;
    }

    buf->size = mapped_bytes;
    buf->owns = false;
    return buf;
}

struct fm_region {
    void       *base;
    size_t      bytes;         /* logical length the caller asked for */
    size_t      segment_span;  /* bytes covered by one segment */
    size_t      segment_step;  /* distance between segment starts */
    size_t      max_span;      /* largest bindable range */
    int         count;
    fm_buffer **segments;
};

#define FM_REGION_DEFAULT_MAX_SPAN  (1024u * 1024u * 1024u)

/* Test override for the device's maxBufferLength; 0 = use the real one. */
static size_t _fm_region_span_override = 0;

void fm_region_force_segment_span(size_t bytes) {
    _fm_region_span_override = bytes;
}

/* The largest buffer fm_region_open may create. */
static size_t fm_region_max_segment_length(void) {
    if (_fm_region_span_override) return _fm_region_span_override;
    return (size_t)[_fm_device maxBufferLength];
}

/* Chooses segment geometry for a region that does not fit in one buffer.
 * Segments overlap by max_span so no range of that size can straddle a
 * boundary.  Returns -1 if no positive step is possible. */
static int fm_region_plan(fm_region *region, size_t max_length) {
    size_t page = (size_t)_fm_page_size;
    size_t span = (max_length / page) * page;
    size_t overlap = fm_page_round_up(region->max_span);

    if (overlap >= span) {
        NSLog(@"[fm] region_open: max_span %zu leaves no room inside a %zu byte "
              @"segment", region->max_span, span);
        return -1;
    }

    region->segment_span = span;
    region->segment_step = span - overlap;
    /* Segments start at 0, step, 2*step, ... until one reaches the end. */
    size_t remaining = region->bytes - span;
    region->count = 1 + (int)((remaining + region->segment_step - 1) / region->segment_step);
    return 0;
}

fm_region *fm_region_open(void *mapped_base, size_t mapped_bytes,
                          size_t max_span_bytes) {
    if (!_fm_initialized || !mapped_base || mapped_bytes == 0) return NULL;

    if (((uintptr_t)mapped_base % (uintptr_t)_fm_page_size) != 0) {
        NSLog(@"[fm] region_open: base %p is not page-aligned", mapped_base);
        return NULL;
    }

    fm_region *region = (fm_region *)calloc(1, sizeof(*region));
    if (!region) return NULL;

    region->base     = mapped_base;
    region->bytes    = mapped_bytes;
    region->max_span = max_span_bytes ? max_span_bytes : FM_REGION_DEFAULT_MAX_SPAN;

    size_t max_length = fm_region_max_segment_length();
    if (fm_page_round_up(mapped_bytes) <= max_length) {
        region->segment_span = mapped_bytes;
        region->segment_step = mapped_bytes;
        region->count        = 1;
    } else if (fm_region_plan(region, max_length) != 0) {
        free(region);
        return NULL;
    }

    region->segments = (fm_buffer **)calloc((size_t)region->count, sizeof(*region->segments));
    if (!region->segments) {
        free(region);
        return NULL;
    }

    for (int i = 0; i < region->count; i++) {
        size_t start = (size_t)i * region->segment_step;
        size_t len   = region->bytes - start;
        if (len > region->segment_span) len = region->segment_span;

        region->segments[i] = fm_buffer_map_no_copy((char *)mapped_base + start, len);
        if (!region->segments[i]) {
            NSLog(@"[fm] region_open: segment %d/%d (%zu bytes at +%zu) failed",
                  i + 1, region->count, len, start);
            fm_region_close(region);
            return NULL;
        }
    }
    return region;
}

void fm_region_close(fm_region *region) {
    if (!region) return;
    if (region->segments) {
        for (int i = 0; i < region->count; i++)
            fm_buffer_free(region->segments[i]);
        free(region->segments);
    }
    free(region);
}

int fm_region_segment_count(const fm_region *region) {
    return region ? region->count : 0;
}

int fm_region_bind(const fm_region *region, size_t offset, size_t bytes,
                   fm_bufref *out) {
    if (!region || !out || bytes == 0) return -1;

    if (offset > region->bytes || bytes > region->bytes - offset) {
        NSLog(@"[fm] region_bind: range [%zu, %zu) runs past the %zu byte region",
              offset, offset + bytes, region->bytes);
        return -1;
    }
    if ((offset % FM_BUFFER_OFFSET_ALIGNMENT) != 0) {
        NSLog(@"[fm] region_bind: offset %zu is not a multiple of %d",
              offset, FM_BUFFER_OFFSET_ALIGNMENT);
        return -1;
    }
    if (bytes > region->max_span) {
        NSLog(@"[fm] region_bind: %zu bytes exceeds the region's max span %zu — "
              @"reopen the region with a larger max_span_bytes",
              bytes, region->max_span);
        return -1;
    }

    int index = (int)(offset / region->segment_step);
    if (index >= region->count) index = region->count - 1;
    size_t start = (size_t)index * region->segment_step;

    /* Guaranteed by the overlap, but a silent mis-bind reads whatever
     * happens to follow the segment, so check rather than trust. */
    size_t within = offset - start;
    if (within + bytes > fm_buffer_size(region->segments[index])) {
        NSLog(@"[fm] region_bind: range [%zu, %zu) straddles segment %d",
              offset, offset + bytes, index);
        return -1;
    }

    out->buffer = region->segments[index];
    out->offset = within;
    return 0;
}

/* ================================================================
 * Helper: persistent command buffer (MLX-style auto-batching)
 *
 * Instead of creating+committing a command buffer per dispatch,
 * we keep a persistent one that accumulates work. The caller
 * must call fm_sync() when it needs GPU results on the CPU.
 * This follows MLX's pattern of batching 20-50 ops per buffer.
 * ================================================================ */

id<MTLCommandBuffer> fm_get_command_buffer(void) {
    if (_fm_batching && _fm_batch_cmdbuf) {
        return _fm_batch_cmdbuf;
    }
    /* Auto-batching: reuse persistent command buffer */
    if (!_fm_persistent_cmdbuf) {
        _fm_persistent_cmdbuf = [_fm_queue commandBuffer];
    }
    return _fm_persistent_cmdbuf;
}

void fm_submit_if_not_batching(void) {
    if (_fm_batching) return;
    if (_fm_auto_batch) {
        /* Auto-batch mode: accumulate, let fm_sync() handle commit */
        _fm_pending_ops++;
    } else {
        /* Immediate mode (default): commit and wait now */
        if (_fm_persistent_cmdbuf) {
            [_fm_persistent_cmdbuf commit];
            [_fm_persistent_cmdbuf waitUntilCompleted];
            _fm_persistent_cmdbuf = nil;
            _fm_pending_ops = 0;
        }
    }
}

/* ================================================================
 * Custom compute kernel dispatch helpers
 * ================================================================ */

/* Starts an encoder with the pipeline, the buffer references (each bound
 * at its own byte offset) and the inline uint constants already set.
 * Returns nil if the pipeline is unavailable. */
static id<MTLComputeCommandEncoder> fm_begin_dispatch(id<MTLCommandBuffer> cmdbuf,
                                                      id<MTLComputePipelineState> ps,
                                                      const fm_bufref *refs, int n_refs,
                                                      const uint32_t *constants,
                                                      int n_constants) {
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    [enc setComputePipelineState:ps];
    for (int i = 0; i < n_refs; i++) {
        [enc setBuffer:refs[i].buffer->mtl offset:refs[i].offset atIndex:i];
    }
    for (int i = 0; i < n_constants; i++) {
        [enc setBytes:&constants[i] length:sizeof(uint32_t) atIndex:n_refs + i];
    }
    return enc;
}

/* Wraps plain buffers as references at offset 0, for the many callers
 * that bind whole buffers. */
static int fm_refs_from_buffers(fm_bufref *refs, fm_buffer **buffers, int n_buffers) {
    if (n_buffers > FM_MAX_BINDINGS) {
        NSLog(@"[fm] dispatch: %d buffers exceeds FM_MAX_BINDINGS %d",
              n_buffers, FM_MAX_BINDINGS);
        return -1;
    }
    for (int i = 0; i < n_buffers; i++)
        refs[i] = fm_bufref_at(buffers[i], 0);
    return 0;
}

/* 1-D dispatch of a pipeline specialized by `fc` (NULL = unspecialized).
 * `constants` are ordinary runtime buffer arguments bound after the
 * buffers, not function constants — the two mechanisms are unrelated. */
void fm_dispatch_1d_refs(NSString *kernel_name, const fm_fc_values *fc,
                         const fm_bufref *refs, int n_refs,
                         uint32_t *constants, int n_constants,
                         int total_threads) {
    if (!_fm_initialized) return;

    id<MTLComputePipelineState> ps = fm_get_pipeline_specialized(kernel_name, fc);
    if (!ps) return;

    id<MTLCommandBuffer> cmdbuf = fm_get_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        fm_begin_dispatch(cmdbuf, ps, refs, n_refs, constants, n_constants);

    NSUInteger tpg = MIN((NSUInteger)total_threads, ps.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(total_threads, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];

    fm_submit_if_not_batching();
}

void fm_dispatch_1d_specialized(NSString *kernel_name, const fm_fc_values *fc,
                                fm_buffer **buffers, int n_buffers,
                                uint32_t *constants, int n_constants,
                                int total_threads) {
    fm_bufref refs[FM_MAX_BINDINGS];
    if (fm_refs_from_buffers(refs, buffers, n_buffers) != 0) return;
    fm_dispatch_1d_refs(kernel_name, fc, refs, n_buffers,
                        constants, n_constants, total_threads);
}

void fm_dispatch_1d(NSString *kernel_name,
                    fm_buffer **buffers, int n_buffers,
                    uint32_t *constants, int n_constants,
                    int total_threads) {
    fm_dispatch_1d_specialized(kernel_name, NULL, buffers, n_buffers,
                               constants, n_constants, total_threads);
}

/* Threadgroup dispatch binding each operand at its own byte offset —
 * the path a weight living inside a shared mmap region takes. */
void fm_dispatch_threadgroups_refs(NSString *kernel_name, const fm_fc_values *fc,
                                   const fm_bufref *refs, int n_refs,
                                   uint32_t *constants, int n_constants,
                                   int num_groups, int threads_per_group) {
    if (!_fm_initialized) return;

    id<MTLComputePipelineState> ps = fm_get_pipeline_specialized(kernel_name, fc);
    if (!ps) return;

    id<MTLCommandBuffer> cmdbuf = fm_get_command_buffer();
    id<MTLComputeCommandEncoder> enc =
        fm_begin_dispatch(cmdbuf, ps, refs, n_refs, constants, n_constants);

    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
    [enc endEncoding];

    fm_submit_if_not_batching();
}

void fm_dispatch_threadgroups(NSString *kernel_name,
                              fm_buffer **buffers, int n_buffers,
                              uint32_t *constants, int n_constants,
                              int num_groups, int threads_per_group) {
    fm_bufref refs[FM_MAX_BINDINGS];
    if (fm_refs_from_buffers(refs, buffers, n_buffers) != 0) return;
    fm_dispatch_threadgroups_refs(kernel_name, NULL, refs, n_buffers,
                                  constants, n_constants,
                                  num_groups, threads_per_group);
}

/* Helper for dispatching with a float constant (e.g., eps, x_scale) */
void fm_dispatch_with_float_constant(NSString *kernel_name,
                                     fm_buffer **buffers, int n_buffers,
                                     uint32_t *uint_consts, int n_uint,
                                     float float_const,
                                     int num_groups, int threads_per_group) {
    if (!_fm_initialized) return;

    id<MTLComputePipelineState> ps = fm_get_pipeline(kernel_name);
    if (!ps) return;

    id<MTLCommandBuffer> cmdbuf = fm_get_command_buffer();
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    [enc setComputePipelineState:ps];
    for (int i = 0; i < n_buffers; i++) {
        [enc setBuffer:buffers[i]->mtl offset:0 atIndex:i];
    }
    int idx = n_buffers;
    for (int i = 0; i < n_uint; i++) {
        [enc setBytes:&uint_consts[i] length:sizeof(uint32_t) atIndex:idx++];
    }
    [enc setBytes:&float_const length:sizeof(float) atIndex:idx];

    MTLSize groups = MTLSizeMake(num_groups, 1, 1);
    MTLSize tpg = MTLSizeMake(threads_per_group, 1, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
    [enc endEncoding];

    fm_submit_if_not_batching();
}

/* ================================================================
 * Synchronization & Command batching
 * ================================================================ */

void fm_sync(void) {
    if (!_fm_initialized) return;

    /* Flush persistent command buffer if it has pending work */
    if (_fm_persistent_cmdbuf && _fm_pending_ops > 0) {
        [_fm_persistent_cmdbuf commit];
        [_fm_persistent_cmdbuf waitUntilCompleted];
        _fm_persistent_cmdbuf = nil;
        _fm_pending_ops = 0;
    }
}

void fm_set_auto_batch(bool enable) {
    if (!_fm_initialized) return;
    if (!enable && _fm_auto_batch) {
        /* Switching off: flush any pending work */
        fm_sync();
    }
    _fm_auto_batch = enable;
}

int fm_batch_begin(void) {
    if (!_fm_initialized || _fm_batching) return -1;

    /* Flush any pending auto-batched work first */
    fm_sync();

    _fm_batch_cmdbuf = [_fm_queue commandBuffer];
    if (!_fm_batch_cmdbuf) return -1;
    _fm_batching = true;
    return 0;
}

void fm_batch_end(void) {
    if (!_fm_batching || !_fm_batch_cmdbuf) return;

    [_fm_batch_cmdbuf commit];
    [_fm_batch_cmdbuf waitUntilCompleted];
    _fm_batch_cmdbuf = nil;
    _fm_batching = false;
}

bool fm_batch_active(void) {
    return _fm_batching;
}
