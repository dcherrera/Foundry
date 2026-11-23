/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file test_metal_region.c
 * @brief Tests no-copy mmap regions and offset binding: that wrapping a
 *   mapping really aliases it instead of copying, that a range resolves to
 *   the right (segment, offset), and that a real Q4_K matvec reading a
 *   weight at a nonzero offset produces exactly what it produces at zero.
 *
 * The offsets used are multiples of 32 but NOT of the 16 KiB page, which
 * is the real GGUF case: tensor data starts at an arbitrary file offset
 * and tensors are only 32-byte aligned.
 *
 * Exits 77 when no Metal device is present so GPU-less CI skips cleanly.
 *
 * @status All tests pass on M4 / macOS 26; each was proven able to fail.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
#include "metal/fm_metal.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

/* A byte offset that is a multiple of 32 (GGUF's tensor alignment) but
 * deliberately not a multiple of the 16384-byte page. */
#define WEIGHT_OFFSET  (1024u * 1024u + 32u)

#define Q4K_BLOCK_BYTES  144u

#define SCRATCH_DIR \
    "metal/tests/"   /* run from the repo root; test_* is gitignored */

static const char *kScratchPath  = SCRATCH_DIR "test_region_map.bin";
static const char *kScratchPathB = SCRATCH_DIR "test_region_map_b.bin";

/* ================================================================
 * Mapping helpers
 * ================================================================ */

/* Creates a sparse file of `bytes` and maps it read/write, private.
 * Returns the mapping base, or NULL. */
static void *map_file(const char *path, size_t bytes, int *fd_out)
{
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return NULL;
    if (ftruncate(fd, (off_t)bytes) != 0) { close(fd); return NULL; }

    void *base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return NULL; }

    *fd_out = fd;
    return base;
}

static void unmap_file(const char *path, void *base, size_t bytes, int fd)
{
    munmap(base, bytes);
    close(fd);
    unlink(path);
}

static void *map_scratch(size_t bytes, int *fd_out)
{
    return map_file(kScratchPath, bytes, fd_out);
}

static void unmap_scratch(void *base, size_t bytes, int fd)
{
    unmap_file(kScratchPath, base, bytes, fd);
}

/* ================================================================
 * Q4_K test weight
 * ================================================================ */

static uint16_t f32_to_f16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Fills `rows` structurally valid Q4_K super-blocks (one per row, so
 * cols = 256).  Values are small and bounded so the matvec output is
 * finite and clearly nonzero.  `seed` selects a distinct weight — tests
 * rely on two different seeds producing visibly different outputs, so
 * that a binding which ignores its offset cannot pass. */
static void fill_q4k_rows(uint8_t *dst, int rows, uint32_t seed)
{
    uint32_t rng = 0x9e3779b9u + seed * 0x85ebca6bu;
    for (int r = 0; r < rows; r++) {
        uint8_t *blk = dst + (size_t)r * Q4K_BLOCK_BYTES;
        uint16_t d    = f32_to_f16(0.01f + 0.001f * (float)((r + seed) % 7));
        uint16_t dmin = f32_to_f16(0.005f);
        memcpy(blk + 0, &d, 2);
        memcpy(blk + 2, &dmin, 2);
        for (int i = 0; i < 12; i++)
            blk[4 + i] = (uint8_t)(8 + ((r * 3 + i * 5 + (int)seed * 11) % 40));
        for (int i = 0; i < 128; i++) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            blk[16 + i] = (uint8_t)(rng & 0xFFu);
        }
    }
}

/* Fraction-free "these are clearly different vectors" check. */
static int vectors_differ(const float *a, const float *b, int n)
{
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > 1e-3f) return 1;
    return 0;
}

/* Runs the Q4_K matvec with the weight at `weight_offset` inside `region`
 * and writes `rows` floats into out_host.  Returns 0 on success. */
static int matvec_from_region(fm_region *region, size_t weight_offset,
                              int rows, int cols, fm_buffer *x, fm_buffer *out,
                              float *out_host)
{
    size_t row_stride = (size_t)(cols / 256) * Q4K_BLOCK_BYTES;

    fm_bufref w;
    if (fm_region_bind(region, weight_offset, (size_t)rows * row_stride, &w) != 0)
        return -1;

    memset(fm_buffer_data(out), 0, (size_t)rows * sizeof(float));
    fm_matvec_q4_k_at(w, fm_bufref_at(x, 0), fm_bufref_at(out, 0),
                      rows, cols, (int)row_stride);
    fm_sync();
    memcpy(out_host, fm_buffer_data(out), (size_t)rows * sizeof(float));
    return 0;
}

/* ================================================================
 * No-copy wrapping
 * ================================================================ */

static void test_map_no_copy_aliases_the_mapping(void)
{
    size_t bytes = 4u * 1024 * 1024;
    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    fm_buffer *buf = fm_buffer_map_no_copy(base, bytes);
    ASSERT_NOT_NULL(buf);

    /* The whole point: the buffer must BE the mapping, not a duplicate. */
    ASSERT_TRUE(fm_buffer_data(buf) == base);

    /* And it must stay aliased — a host write is visible through it. */
    ((uint8_t *)base)[12345] = 0xA7;
    ASSERT_INT_EQ(0xA7, ((uint8_t *)fm_buffer_data(buf))[12345]);

    fm_buffer_free(buf);
    unmap_scratch(base, bytes, fd);
}

static void test_map_no_copy_refuses_unaligned_base(void)
{
    size_t bytes = 1024u * 1024;
    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    /* 32-byte aligned but not page-aligned: fm_buffer_from_ptr would
     * silently copy here, which is the bug.  This must refuse instead. */
    fm_buffer *buf = fm_buffer_map_no_copy((char *)base + 32, bytes - 32);
    ASSERT_NULL(buf);

    unmap_scratch(base, bytes, fd);
}

static void test_map_no_copy_refuses_oversize_length(void)
{
    size_t bytes = 1024u * 1024;
    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    size_t max_length = fm_buffer_max_length();
    ASSERT_TRUE(max_length > 0);

    /* Rejected on the length check before Metal is asked, so this never
     * trips the validation layer's hard assert. */
    ASSERT_NULL(fm_buffer_map_no_copy(base, max_length + 1));

    unmap_scratch(base, bytes, fd);
}

/* ================================================================
 * Region geometry and binding
 * ================================================================ */

static void test_small_mapping_is_one_segment(void)
{
    size_t bytes = 8u * 1024 * 1024;
    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    fm_region *region = fm_region_open(base, bytes, 0);
    ASSERT_NOT_NULL(region);
    ASSERT_INT_EQ(1, fm_region_segment_count(region));

    fm_region_close(region);
    unmap_scratch(base, bytes, fd);
}

static void test_bind_resolves_to_the_host_address(void)
{
    size_t bytes = 8u * 1024 * 1024;
    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    fm_region *region = fm_region_open(base, bytes, 0);
    ASSERT_NOT_NULL(region);

    size_t offsets[] = { 0, 32, 4096, WEIGHT_OFFSET, bytes - 1024 };
    for (unsigned i = 0; i < sizeof(offsets) / sizeof(offsets[0]); i++) {
        fm_bufref ref;
        ASSERT_INT_EQ(0, fm_region_bind(region, offsets[i], 512, &ref));
        const char *bound = (const char *)fm_buffer_data(ref.buffer) + ref.offset;
        ASSERT_TRUE(bound == (const char *)base + offsets[i]);
    }

    fm_region_close(region);
    unmap_scratch(base, bytes, fd);
}

static void test_bind_rejects_bad_ranges(void)
{
    size_t bytes = 8u * 1024 * 1024;
    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    fm_region *region = fm_region_open(base, bytes, 64 * 1024);
    ASSERT_NOT_NULL(region);

    fm_bufref ref;
    ASSERT_INT_EQ(-1, fm_region_bind(region, bytes - 16, 512, &ref));   /* past the end */
    ASSERT_INT_EQ(-1, fm_region_bind(region, bytes + 4096, 16, &ref));  /* wholly outside */
    ASSERT_INT_EQ(-1, fm_region_bind(region, 8, 512, &ref));            /* misaligned offset */
    ASSERT_INT_EQ(-1, fm_region_bind(region, 4096, 128 * 1024, &ref));  /* over max span */
    ASSERT_INT_EQ(-1, fm_region_bind(region, 4096, 0, &ref));           /* empty range */

    fm_region_close(region);
    unmap_scratch(base, bytes, fd);
}

/* ================================================================
 * A real kernel reading at a nonzero offset
 * ================================================================ */

/* Places weight `seed` at `offset` in a fresh second mapping and runs the
 * matvec from its offset 0 — the independent reference an offset-bound
 * run must reproduce.  Returns 0 on success. */
static int reference_at_offset_zero(int rows, int cols, uint32_t seed,
                                    fm_buffer *x, fm_buffer *out, float *result)
{
    size_t weight_bytes = (size_t)rows * Q4K_BLOCK_BYTES;
    size_t bytes = 1024u * 1024;
    int fd = -1;
    void *base = map_file(kScratchPathB, bytes, &fd);
    if (!base) return -1;

    fill_q4k_rows((uint8_t *)base, rows, seed);

    fm_region *region = fm_region_open(base, bytes, 0);
    int rc = region ? matvec_from_region(region, 0, rows, cols, x, out, result) : -1;
    (void)weight_bytes;

    fm_region_close(region);
    unmap_file(kScratchPathB, base, bytes, fd);
    return rc;
}

static void test_matvec_at_nonzero_offset_matches_offset_zero(void)
{
    int rows = 64, cols = 256;
    size_t bytes = 8u * 1024 * 1024;

    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    /* DIFFERENT weights at offset 0 and at the 32-aligned, non-page-aligned
     * offset.  If the binding ignored its offset both runs would read the
     * decoy at 0 and agree, so the weights must not match. */
    fill_q4k_rows((uint8_t *)base, rows, 1);                        /* decoy at 0 */
    fill_q4k_rows((uint8_t *)base + WEIGHT_OFFSET, rows, 2);        /* real weight */
    ASSERT_TRUE(WEIGHT_OFFSET % 32 == 0);
    ASSERT_TRUE(WEIGHT_OFFSET % 16384 != 0);

    fm_region *region = fm_region_open(base, bytes, 0);
    ASSERT_NOT_NULL(region);

    fm_buffer *x = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int i = 0; i < cols; i++) xh[i] = 0.5f - (float)(i % 11) * 0.1f;

    float decoy[64], at_offset[64], reference[64];
    ASSERT_INT_EQ(0, matvec_from_region(region, 0, rows, cols, x, out, decoy));
    ASSERT_INT_EQ(0, matvec_from_region(region, WEIGHT_OFFSET, rows, cols, x, out, at_offset));

    /* The kernel actually ran, and it did not just re-read offset 0. */
    float magnitude = 0.0f;
    for (int r = 0; r < rows; r++) magnitude += fabsf(at_offset[r]);
    ASSERT_TRUE(magnitude > 1e-3f);
    for (int r = 0; r < rows; r++) ASSERT_TRUE(isfinite(at_offset[r]));
    ASSERT_TRUE(vectors_differ(decoy, at_offset, rows));

    /* And it read the right weight: same bytes at offset 0 of another
     * region give exactly the same numbers. */
    ASSERT_INT_EQ(0, reference_at_offset_zero(rows, cols, 2, x, out, reference));
    for (int r = 0; r < rows; r++)
        ASSERT_FLT_NEAR(reference[r], at_offset[r], 0.0f);

    fm_buffer_free(x);
    fm_buffer_free(out);
    fm_region_close(region);
    unmap_scratch(base, bytes, fd);
}

static void test_matvec_at_region_end_is_not_truncated(void)
{
    int rows = 64, cols = 256;
    size_t weight_bytes = (size_t)rows * Q4K_BLOCK_BYTES;
    size_t bytes = 8u * 1024 * 1024;

    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    /* Last weight-sized range in the region, kept 32-byte aligned, and a
     * different decoy weight at 0. */
    size_t tail_offset = (bytes - weight_bytes) & ~(size_t)31;
    fill_q4k_rows((uint8_t *)base, rows, 3);
    fill_q4k_rows((uint8_t *)base + tail_offset, rows, 4);

    fm_region *region = fm_region_open(base, bytes, 0);
    ASSERT_NOT_NULL(region);

    fm_buffer *x = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int i = 0; i < cols; i++) xh[i] = 0.25f + (float)(i % 5) * 0.05f;

    float decoy[64], at_tail[64], reference[64];
    ASSERT_INT_EQ(0, matvec_from_region(region, 0, rows, cols, x, out, decoy));
    ASSERT_INT_EQ(0, matvec_from_region(region, tail_offset, rows, cols, x, out, at_tail));

    ASSERT_TRUE(vectors_differ(decoy, at_tail, rows));
    ASSERT_INT_EQ(0, reference_at_offset_zero(rows, cols, 4, x, out, reference));

    /* The very last row must be right too — that is where a truncated
     * buffer length would show up. */
    for (int r = 0; r < rows; r++)
        ASSERT_FLT_NEAR(reference[r], at_tail[r], 0.0f);
    ASSERT_TRUE(fabsf(at_tail[rows - 1]) > 1e-6f);

    fm_buffer_free(x);
    fm_buffer_free(out);
    fm_region_close(region);
    unmap_scratch(base, bytes, fd);
}

/* ================================================================
 * Multi-segment regions
 * ================================================================ */

static void test_oversized_mapping_splits_into_segments(void)
{
    size_t bytes = 8u * 1024 * 1024;
    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    /* Pretend one buffer can only cover 2 MiB. */
    fm_region_force_segment_span(2u * 1024 * 1024);
    fm_region *region = fm_region_open(base, bytes, 256 * 1024);
    fm_region_force_segment_span(0);
    ASSERT_NOT_NULL(region);
    ASSERT_TRUE(fm_region_segment_count(region) > 1);

    /* Every 32-aligned range in the region must resolve to the right host
     * bytes, including ranges that cross where a naive split would cut. */
    for (size_t offset = 0; offset + 4096 <= bytes; offset += 64 * 1024 + 32) {
        fm_bufref ref;
        ASSERT_INT_EQ(0, fm_region_bind(region, offset, 4096, &ref));
        const char *bound = (const char *)fm_buffer_data(ref.buffer) + ref.offset;
        ASSERT_TRUE(bound == (const char *)base + offset);
        ASSERT_TRUE(ref.offset + 4096 <= fm_buffer_size(ref.buffer));
    }

    fm_region_close(region);
    unmap_scratch(base, bytes, fd);
}

static void test_matvec_through_a_later_segment(void)
{
    int rows = 64, cols = 256;
    size_t weight_bytes = (size_t)rows * Q4K_BLOCK_BYTES;
    size_t bytes = 8u * 1024 * 1024;
    /* Well past the first 2 MiB segment, 32-aligned, not page-aligned. */
    size_t far_offset = 6u * 1024 * 1024 + 32u;

    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);

    /* Different weight at 0 so a binding that ignores its offset, or one
     * that always picks segment 0, reads visibly wrong numbers. */
    fill_q4k_rows((uint8_t *)base, rows, 5);
    fill_q4k_rows((uint8_t *)base + far_offset, rows, 6);
    (void)weight_bytes;

    fm_region_force_segment_span(2u * 1024 * 1024);
    fm_region *region = fm_region_open(base, bytes, 256 * 1024);
    fm_region_force_segment_span(0);
    ASSERT_NOT_NULL(region);
    ASSERT_TRUE(fm_region_segment_count(region) > 1);

    fm_buffer *x = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *out = fm_buffer_alloc((size_t)rows * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    float *xh = (float *)fm_buffer_data(x);
    for (int i = 0; i < cols; i++) xh[i] = 0.3f - (float)(i % 9) * 0.07f;

    float decoy[64], at_far[64], reference[64];
    ASSERT_INT_EQ(0, matvec_from_region(region, 0, rows, cols, x, out, decoy));
    ASSERT_INT_EQ(0, matvec_from_region(region, far_offset, rows, cols, x, out, at_far));

    float magnitude = 0.0f;
    for (int r = 0; r < rows; r++) magnitude += fabsf(at_far[r]);
    ASSERT_TRUE(magnitude > 1e-3f);
    ASSERT_TRUE(vectors_differ(decoy, at_far, rows));

    ASSERT_INT_EQ(0, reference_at_offset_zero(rows, cols, 6, x, out, reference));
    for (int r = 0; r < rows; r++)
        ASSERT_FLT_NEAR(reference[r], at_far[r], 0.0f);

    fm_buffer_free(x);
    fm_buffer_free(out);
    fm_region_close(region);
    unmap_scratch(base, bytes, fd);
}

/* The real thing: a mapping larger than the device's maxBufferLength,
 * which is exactly the 14.89 GiB weight case that started this. */
static void test_mapping_larger_than_one_buffer(void)
{
    size_t max_length = fm_buffer_max_length();
    ASSERT_TRUE(max_length > 0);

    size_t bytes = max_length + 1024u * 1024 * 1024;   /* ~1 GiB over the limit */
    int fd = -1;
    void *base = map_scratch(bytes, &fd);
    ASSERT_NOT_NULL(base);   /* sparse file: costs address space, not disk */

    fm_region *region = fm_region_open(base, bytes, 512u * 1024 * 1024);
    ASSERT_NOT_NULL(region);
    ASSERT_TRUE(fm_region_segment_count(region) > 1);

    /* A weight near the very end must still bind, and the resolved
     * address must be exact — this is where a 32-bit offset or a
     * truncated segment length would show. */
    size_t tail = (bytes - 64 * 1024) & ~(size_t)31;
    fm_bufref ref;
    ASSERT_INT_EQ(0, fm_region_bind(region, tail, 64 * 1024, &ref));
    ASSERT_TRUE((const char *)fm_buffer_data(ref.buffer) + ref.offset
                == (const char *)base + tail);

    /* And it must genuinely be the mapping, not a copy. */
    ((uint8_t *)base)[tail] = 0x5C;
    ASSERT_INT_EQ(0x5C, ((const uint8_t *)fm_buffer_data(ref.buffer))[ref.offset]);

    fm_region_close(region);
    unmap_scratch(base, bytes, fd);
}

/* ================================================================
 * main
 * ================================================================ */

int main(void)
{
    printf("\n  Metal No-Copy Region + Offset Binding Tests\n");
    printf("  ==========================================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "Metal unavailable\n");
        return 77;
    }

    printf("  device maxBufferLength   = %zu bytes (%.2f GiB)\n",
           fm_buffer_max_length(), fm_buffer_max_length() / 1073741824.0);
    printf("  bind offset alignment    = %zu bytes\n\n",
           fm_buffer_offset_alignment());

    RUN_TEST(map_no_copy_aliases_the_mapping);
    RUN_TEST(map_no_copy_refuses_unaligned_base);
    RUN_TEST(map_no_copy_refuses_oversize_length);

    RUN_TEST(small_mapping_is_one_segment);
    RUN_TEST(bind_resolves_to_the_host_address);
    RUN_TEST(bind_rejects_bad_ranges);

    RUN_TEST(matvec_at_nonzero_offset_matches_offset_zero);
    RUN_TEST(matvec_at_region_end_is_not_truncated);

    RUN_TEST(oversized_mapping_splits_into_segments);
    RUN_TEST(matvec_through_a_later_segment);
    RUN_TEST(mapping_larger_than_one_buffer);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
