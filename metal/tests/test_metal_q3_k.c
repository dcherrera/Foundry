/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/**
 * @file test_metal_q3_k.c
 * @brief Absolute-correctness tests for the Metal Q3_K matvec kernels.
 *
 *   The bar here is ABSOLUTE, never GPU-against-GPU.  Every test compares the
 *   kernel either to the CPU dequant+dot reference written below — decoded
 *   straight from the GGUF Q3_K block spec, walking the block in its storage order
 *   rather than the kernel's inverted-index order, so the two derive the same
 *   mapping by different routes — or to the Pantheon golden vectors, which
 *   come from a third independent implementation.  A GPU-vs-GPU test proves
 *   offset invariance and nothing else: a uniformly scaled row sum cancels in
 *   both arms and passes.  The offset test here is therefore ADDITIONAL to an
 *   absolute one, not a substitute for it.
 *
 *   Inputs are real Q3_K super-blocks read out of the shipped model file, not
 *   only synthetic ones, because a synthetic block cannot catch a misread of a
 *   real tensor's layout or placement.
 *
 * @status All tests pass on M4 / macOS 26.  Each was proven able to fail
 *   against a deliberately broken kernel (scaled row sum, wrong scale decode,
 *   off-by-one block stride) before being trusted.
 * @issues None
 * @todo None
 */

#include "lib/test.h"
#include "metal/fm_metal.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* ================================================================
 * Fixtures
 * ================================================================ */

#define Q3K_BLOCK_BYTES    110
#define Q3K_BLOCK_WEIGHTS  256

/* ggml tensor type id for Q3_K, as written in a GGUF tensor info record. */
#define GGUF_TENSOR_TYPE_Q3_K  11

/* A byte offset that is a multiple of 32 (GGUF's tensor alignment) but
 * deliberately not a multiple of the 16384-byte page — the real GGUF case. */
#define SCRATCH_WEIGHT_OFFSET  (1024u * 1024u + 32u)

#define SCRATCH_DIR \
    "metal/tests/"   /* run from the repo root; test_* is gitignored */

static const char *kScratchPath = SCRATCH_DIR "test_q3k_region_map.bin";

static const char *kDefaultGoldenDir =
    "../Pantheon/tests/golden";
static const char *kDefaultModelPath =
    "models/Nex-N2-mini/nex-agi_Nex-N2-mini-Q3_K_M.gguf";

/* The golden matvec fixture's shape, from tests/golden/README.md. */
#define GOLDEN_ROWS  24
#define GOLDEN_COLS  1024

/* Rows of the real model tensor to exercise.  Enough to cover many blocks
 * and many threadgroups without making the CPU reference slow. */
#define MODEL_ROWS  24

static const char *golden_dir(void)
{
    const char *env = getenv("PN_GOLDEN_DIR");
    return (env && *env) ? env : kDefaultGoldenDir;
}

static const char *model_path(void)
{
    const char *env = getenv("NEX_N2_GGUF");
    return (env && *env) ? env : kDefaultModelPath;
}

/* ================================================================
 * CPU reference — decoded from the GGUF Q3_K block spec
 *
 *   hmask[32]  @   0   high bit of each of the 256 quants
 *   qs[64]     @  32   low 2 bits of each quant, 4 per byte
 *   scales[12] @  96   16 six-bit sub-block scales, interleaved packing
 *   d (fp16)   @ 108   super-block scale
 * ================================================================ */

/* IEEE-754 binary16 -> binary32, including subnormals, Inf and NaN. */
static float half_bits_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* Subnormal: value = mant * 2^-24. Shifting left until bit 10 is set
             * leaves 1.f * 2^(-14 - shift), so the binary32 exponent field is
             * 127 - 14 - shift. Off by one here HALVES every subnormal scale,
             * and real Q4_K/Q3_K `d` values do go subnormal (4.55e-05 is below
             * half's 6.10e-05 smallest normal). A wrong oracle fails a correct
             * kernel. */
            int shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; shift++; }
            mant &= 0x03FFu;
            bits = sign | ((uint32_t)(127 - 14 - shift) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Unpack the 16 six-bit sub-block scales, already biased by -32.  ggml packs
 * the low 4 bits of scales 0..7 into the low nibbles of scales_raw[0..7], the
 * low 4 bits of scales 8..15 into their high nibbles, and the top 2 bits of
 * all sixteen into scales_raw[8..11], two bits per scale. */
static void q3k_unpack_subblock_scales(const uint8_t *scales_raw, int out[16])
{
    for (int i = 0; i < 4; i++) {
        uint8_t hi = scales_raw[8 + i];
        out[i]      = (int)((scales_raw[i]     & 0x0F) | (((hi >> 0) & 3) << 4)) - 32;
        out[i + 4]  = (int)((scales_raw[i + 4] & 0x0F) | (((hi >> 2) & 3) << 4)) - 32;
        out[i + 8]  = (int)((scales_raw[i]     >> 4)   | (((hi >> 4) & 3) << 4)) - 32;
        out[i + 12] = (int)((scales_raw[i + 4] >> 4)   | (((hi >> 6) & 3) << 4)) - 32;
    }
}

/*
 * Dequantize one 110-byte Q3_K super-block into 256 floats.
 *
 * Walks the block the way ggml writes it — two halves of 128 weights, four
 * 2-bit shift levels per half, two 16-weight groups per level, the high-mask
 * bit advancing once per 32 weights — rather than the closed-form index the
 * kernel uses.  A mistake in either derivation shows up as a mismatch.
 *
 * @param block  Pointer to the 110-byte super-block
 * @param out    Receives 256 dequantized weights
 */
static void q3k_dequant_block_reference(const uint8_t *block, float *out)
{
    const uint8_t *hmask = block;
    const uint8_t *qs    = block + 32;
    uint16_t d_bits;
    memcpy(&d_bits, block + 108, sizeof(d_bits));
    const float d = half_bits_to_float(d_bits);

    int scales[16];
    q3k_unpack_subblock_scales(block + 96, scales);

    int idx = 0, is = 0;
    uint8_t mask_bit = 1;
    for (int half = 0; half < 2; half++) {
        const uint8_t *q = qs + half * 32;
        for (int shift = 0; shift < 8; shift += 2) {
            for (int group = 0; group < 32; group += 16) {
                float dl = d * (float)scales[is++];
                for (int l = 0; l < 16; l++) {
                    int qv = (q[group + l] >> shift) & 3;
                    int hb = (hmask[group + l] & mask_bit) ? 0 : 4;
                    out[idx++] = dl * (float)(qv - hb);
                }
            }
            mask_bit = (uint8_t)(mask_bit << 1);
        }
    }
}

/*
 * out[r] = dequant(w[r]) . x, in fp32, block by block.
 *
 * @param w           Q3_K bytes, [rows, cols], rows strided by row_stride
 * @param x           [cols] activations
 * @param out         [rows] result
 * @param row_stride  Bytes per weight row, (cols/256)*110
 */
static void q3k_matvec_reference(const uint8_t *w, const float *x, float *out,
                                 int rows, int cols, int row_stride)
{
    const int n_blocks = cols / Q3K_BLOCK_WEIGHTS;
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            float dq[Q3K_BLOCK_WEIGHTS];
            q3k_dequant_block_reference(w + (size_t)r * (size_t)row_stride
                                          + (size_t)b * Q3K_BLOCK_BYTES, dq);
            const float *xb = x + b * Q3K_BLOCK_WEIGHTS;
            for (int i = 0; i < Q3K_BLOCK_WEIGHTS; i++)
                acc += dq[i] * xb[i];
        }
        out[r] = acc;
    }
}

/* ================================================================
 * Error metric
 *
 * The spec's acceptance bar for an exact kernel is max_abs_err / RMS, so
 * that is what these report — a scalar big enough to see a scaled row sum
 * and small enough that fp32 reassociation noise stays well under it.
 * ================================================================ */

static float max_abs_error_over_rms(const float *got, const float *want, int count)
{
    double sumsq = 0.0;
    float worst = 0.0f;
    for (int i = 0; i < count; i++) {
        double diff = (double)got[i] - (double)want[i];
        if (fabs(diff) > worst) worst = (float)fabs(diff);
        sumsq += (double)want[i] * (double)want[i];
    }
    double rms = sqrt(sumsq / (count > 0 ? count : 1));
    /* An all-zero reference has no scale to divide by: any difference at all
     * is a total failure, so report a value no tolerance will accept.
     * (A literal, not INFINITY — the build enables -ffast-math.) */
    if (rms < 1e-30) return worst > 0.0f ? 1e30f : 0.0f;
    return (float)(worst / rms);
}

/* ================================================================
 * Golden vector loading (Pantheon tests/golden, PNGOLDEN1 format:
 * one ASCII header line, then count raw little-endian values)
 * ================================================================ */

/* Reads a golden file's payload into a fresh buffer.  Caller frees.
 * Returns NULL if the file is missing or shorter than expected. */
static void *load_golden_payload(const char *name, size_t expect_bytes)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", golden_dir(), name);

    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    /* Skip the single ASCII header line. */
    int c;
    while ((c = fgetc(f)) != EOF && c != '\n') { }
    if (c == EOF) { fclose(f); return NULL; }

    void *buf = malloc(expect_bytes);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, expect_bytes, f);
    fclose(f);
    if (got != expect_bytes) { free(buf); return NULL; }
    return buf;
}

/* Bytes in the golden weight matrix — also the bound on how many rows a test
 * may reinterpret it as at a narrower column count. */
#define GOLDEN_WEIGHT_BYTES \
    ((size_t)GOLDEN_ROWS * (GOLDEN_COLS / 256) * Q3K_BLOCK_BYTES)

/* Loads the golden Q3_K weight matrix (24 rows x 1024 cols). Caller frees. */
static uint8_t *load_golden_weight(void)
{
    return load_golden_payload("q3k_matvec_weight.bin", GOLDEN_WEIGHT_BYTES);
}

/* ================================================================
 * Minimal GGUF locator — enough to find a real Q3_K tensor
 * ================================================================ */

typedef struct {
    uint64_t file_offset;   /* absolute byte offset of the tensor's data */
    uint32_t cols;          /* ne[0] — weights per row */
    uint64_t rows;          /* product of ne[1..] */
} gguf_q3k_location;

static int gguf_read_exact(FILE *f, void *dst, size_t bytes)
{
    return fread(dst, 1, bytes, f) == bytes ? 0 : -1;
}

static int gguf_skip_value(FILE *f, uint32_t type);

static int gguf_skip_string(FILE *f)
{
    uint64_t len;
    if (gguf_read_exact(f, &len, 8) != 0) return -1;
    return fseeko(f, (off_t)len, SEEK_CUR);
}

/* Byte width of a GGUF scalar value type, or -1 for string/array. */
static int gguf_scalar_width(uint32_t type)
{
    static const int width[13] = { 1, 1, 2, 2, 4, 4, 4, 1, -1, -1, 8, 8, 8 };
    return type < 13 ? width[type] : -1;
}

static int gguf_skip_value(FILE *f, uint32_t type)
{
    if (type == 8) return gguf_skip_string(f);
    if (type == 9) {
        uint32_t elem_type;
        uint64_t count;
        if (gguf_read_exact(f, &elem_type, 4) != 0) return -1;
        if (gguf_read_exact(f, &count, 8) != 0) return -1;
        int w = gguf_scalar_width(elem_type);
        if (w > 0)   /* one seek for a whole scalar array — vocabs are huge */
            return fseeko(f, (off_t)(count * (uint64_t)w), SEEK_CUR);
        for (uint64_t i = 0; i < count; i++)
            if (gguf_skip_value(f, elem_type) != 0) return -1;
        return 0;
    }
    int w = gguf_scalar_width(type);
    if (w < 0) return -1;
    return fseeko(f, w, SEEK_CUR);
}

/* Skips the header and metadata, leaving the stream on the first tensor info
 * record.  Writes the tensor count and the data-section alignment. */
static int gguf_seek_to_tensor_infos(FILE *f, uint64_t *tensor_count,
                                     uint64_t *alignment)
{
    char magic[4];
    uint32_t version;
    uint64_t kv_count;

    if (gguf_read_exact(f, magic, 4) != 0 || memcmp(magic, "GGUF", 4) != 0) return -1;
    if (gguf_read_exact(f, &version, 4) != 0 || version < 2) return -1;
    if (gguf_read_exact(f, tensor_count, 8) != 0) return -1;
    if (gguf_read_exact(f, &kv_count, 8) != 0) return -1;

    *alignment = 32;   /* GGUF's default when general.alignment is absent */
    for (uint64_t i = 0; i < kv_count; i++) {
        uint64_t key_len;
        char key[256];
        if (gguf_read_exact(f, &key_len, 8) != 0) return -1;
        size_t copy = key_len < sizeof(key) - 1 ? (size_t)key_len : sizeof(key) - 1;
        if (gguf_read_exact(f, key, copy) != 0) return -1;
        key[copy] = '\0';
        if (fseeko(f, (off_t)(key_len - copy), SEEK_CUR) != 0) return -1;

        uint32_t type;
        if (gguf_read_exact(f, &type, 4) != 0) return -1;
        if (strcmp(key, "general.alignment") == 0 && type == 4) {
            uint32_t a;
            if (gguf_read_exact(f, &a, 4) != 0) return -1;
            *alignment = a;
        } else if (gguf_skip_value(f, type) != 0) {
            return -1;
        }
    }
    return 0;
}

/*
 * Finds the first Q3_K tensor whose data does NOT start on a page boundary —
 * the case every pn_qweight in a real GGUF is in, and the one a kernel that
 * binds at offset 0 cannot read.
 *
 * @param path  GGUF file to scan
 * @param out   Receives the tensor's absolute file offset and shape
 * @return      0 on success, -1 if the file is unreadable or has no such tensor
 */
static int gguf_find_unaligned_q3k_tensor(const char *path, gguf_q3k_location *out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint64_t tensor_count = 0, alignment = 32;
    if (gguf_seek_to_tensor_infos(f, &tensor_count, &alignment) != 0) {
        fclose(f);
        return -1;
    }

    /* The data-section start is only known after the last info record, so
     * collect a few candidates and pick an unaligned one afterwards. */
    enum { MAX_CANDIDATES = 8 };
    gguf_q3k_location candidate[MAX_CANDIDATES];
    int n_candidates = 0;
    uint64_t dims[16];
    for (uint64_t t = 0; t < tensor_count; t++) {
        uint32_t n_dims, type;
        uint64_t offset;
        if (gguf_skip_string(f) != 0) break;                    /* tensor name */
        if (gguf_read_exact(f, &n_dims, 4) != 0 || n_dims > 16) break;
        if (gguf_read_exact(f, dims, (size_t)n_dims * 8) != 0) break;
        if (gguf_read_exact(f, &type, 4) != 0) break;
        if (gguf_read_exact(f, &offset, 8) != 0) break;
        if (type != GGUF_TENSOR_TYPE_Q3_K || n_candidates == MAX_CANDIDATES) continue;

        uint64_t rows = 1;
        for (uint32_t d = 1; d < n_dims; d++) rows *= dims[d];
        candidate[n_candidates].cols = (uint32_t)dims[0];
        candidate[n_candidates].rows = rows;
        candidate[n_candidates].file_offset = offset;   /* rebased below */
        n_candidates++;
    }

    off_t infos_end = ftello(f);
    fclose(f);
    if (n_candidates == 0 || infos_end < 0) return -1;

    uint64_t data_start = ((uint64_t)infos_end + alignment - 1) / alignment * alignment;
    size_t page = (size_t)getpagesize();
    for (int i = 0; i < n_candidates; i++) {
        uint64_t off = candidate[i].file_offset + data_start;
        if (off % page == 0) continue;      /* want the unaligned case */
        *out = candidate[i];
        out->file_offset = off;
        return 0;
    }
    return -1;
}

/* ================================================================
 * Shared helpers
 * ================================================================ */

/* Deterministic activations — xorshift32 so a failure is reproducible. */
static void fill_pseudorandom(float *dst, int count, uint32_t seed)
{
    uint32_t s = seed ? seed : 1u;
    for (int i = 0; i < count; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        dst[i] = (float)((int32_t)(s >> 8) - 8388608) / 8388608.0f;
    }
}

/* Runs fm_matvec_q3k over plain buffers and returns the GPU output buffer's
 * contents.  Caller owns nothing — the buffers are freed here. */
static void run_matvec_on_bytes(const uint8_t *w_bytes, const float *x,
                                float *out, int rows, int cols, int row_stride)
{
    fm_buffer *w = fm_buffer_alloc((size_t)rows * (size_t)row_stride);
    fm_buffer *bx = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *bo = fm_buffer_alloc((size_t)rows * sizeof(float));
    if (!w || !bx || !bo) { if (w) fm_buffer_free(w); if (bx) fm_buffer_free(bx);
                            if (bo) fm_buffer_free(bo); return; }

    memcpy(fm_buffer_data(w), w_bytes, (size_t)rows * (size_t)row_stride);
    memcpy(fm_buffer_data(bx), x, (size_t)cols * sizeof(float));
    memset(fm_buffer_data(bo), 0, (size_t)rows * sizeof(float));

    fm_matvec_q3k(w, bx, bo, rows, cols, row_stride);
    fm_sync();
    memcpy(out, fm_buffer_data(bo), (size_t)rows * sizeof(float));

    fm_buffer_free(w);
    fm_buffer_free(bx);
    fm_buffer_free(bo);
}

/* Creates a sparse scratch file and maps it read/write, private. */
static void *map_scratch(size_t bytes, int *fd_out)
{
    int fd = open(kScratchPath, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return NULL;
    if (ftruncate(fd, (off_t)bytes) != 0) { close(fd); return NULL; }
    void *base = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { close(fd); return NULL; }
    *fd_out = fd;
    return base;
}

static void unmap_scratch(void *base, size_t bytes, int fd)
{
    munmap(base, bytes);
    close(fd);
    unlink(kScratchPath);
}

/* The real model's first unaligned Q3_K tensor, mapped through a page-aligned
 * window so the tensor keeps its true (32-aligned, not page-aligned) offset. */
typedef struct {
    void    *window;        /* mmap base, page-aligned */
    size_t   window_bytes;
    size_t   tensor_offset; /* tensor's byte offset inside the window */
    int      fd;
    int      cols;
    int      row_stride;
} model_tensor_window;

static int open_model_tensor_window(model_tensor_window *w, int rows)
{
    gguf_q3k_location loc;
    memset(w, 0, sizeof(*w));
    if (gguf_find_unaligned_q3k_tensor(model_path(), &loc) != 0) return -1;
    if (loc.cols % Q3K_BLOCK_WEIGHTS != 0 || loc.rows < (uint64_t)rows) return -1;

    int fd = open(model_path(), O_RDONLY);
    if (fd < 0) return -1;

    size_t page = (size_t)getpagesize();
    uint64_t window_start = loc.file_offset & ~(uint64_t)(page - 1);
    size_t inside = (size_t)(loc.file_offset - window_start);
    int row_stride = (int)(loc.cols / Q3K_BLOCK_WEIGHTS) * Q3K_BLOCK_BYTES;
    size_t span = inside + (size_t)rows * (size_t)row_stride;
    size_t bytes = (span + page - 1) / page * page;

    void *base = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, (off_t)window_start);
    if (base == MAP_FAILED) { close(fd); return -1; }

    w->window = base;
    w->window_bytes = bytes;
    w->tensor_offset = inside;
    w->fd = fd;
    w->cols = (int)loc.cols;
    w->row_stride = row_stride;
    return 0;
}

static void close_model_tensor_window(model_tensor_window *w)
{
    if (w->window) munmap(w->window, w->window_bytes);
    if (w->fd >= 0) close(w->fd);
    w->window = NULL;
    w->fd = -1;
}

/* ================================================================
 * Tests — dequant
 * ================================================================ */

/* Anchors this file's CPU reference to Pantheon's independent CPU dequant.
 * If this fails, every other absolute test in the file is suspect. */
static void test_reference_dequant_matches_the_golden_block(void)
{
    uint8_t *weight = load_golden_weight();
    float *golden = load_golden_payload("q3k_dequant_block0.bin", 256 * sizeof(float));
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(golden);

    float mine[256];
    q3k_dequant_block_reference(weight, mine);

    float err = max_abs_error_over_rms(mine, golden, 256);
    printf("      dequant block0 max_abs/RMS = %.3e\n", (double)err);
    ASSERT_TRUE(err < 1e-6f);

    free(weight);
    free(golden);
}

/* ================================================================
 * Tests — matvec, absolute
 * ================================================================ */

static void test_matvec_matches_the_golden_vectors(void)
{
    const int row_stride = (GOLDEN_COLS / 256) * Q3K_BLOCK_BYTES;
    uint8_t *weight = load_golden_weight();
    float *x = load_golden_payload("q3k_matvec_input.bin", GOLDEN_COLS * sizeof(float));
    float *golden = load_golden_payload("q3k_matvec_output.bin", GOLDEN_ROWS * sizeof(float));
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(golden);

    float gpu[GOLDEN_ROWS];
    memset(gpu, 0, sizeof(gpu));
    run_matvec_on_bytes(weight, x, gpu, GOLDEN_ROWS, GOLDEN_COLS, row_stride);

    float err = max_abs_error_over_rms(gpu, golden, GOLDEN_ROWS);
    printf("      golden matvec max_abs/RMS  = %.3e  (gpu[0]=%.6f golden[0]=%.6f)\n",
           (double)err, (double)gpu[0], (double)golden[0]);
    ASSERT_TRUE(err < 1e-5f);

    free(weight);
    free(x);
    free(golden);
}

static void test_matvec_matches_the_cpu_reference_on_real_model_blocks(void)
{
    model_tensor_window win;
    ASSERT_INT_EQ(0, open_model_tensor_window(&win, MODEL_ROWS));

    const uint8_t *w = (const uint8_t *)win.window + win.tensor_offset;
    float *x = malloc((size_t)win.cols * sizeof(float));
    float *want = malloc(MODEL_ROWS * sizeof(float));
    float *got = malloc(MODEL_ROWS * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(want);
    ASSERT_NOT_NULL(got);

    fill_pseudorandom(x, win.cols, 0x51EDU);
    q3k_matvec_reference(w, x, want, MODEL_ROWS, win.cols, win.row_stride);
    memset(got, 0, MODEL_ROWS * sizeof(float));
    run_matvec_on_bytes(w, x, got, MODEL_ROWS, win.cols, win.row_stride);

    float err = max_abs_error_over_rms(got, want, MODEL_ROWS);
    printf("      real blocks max_abs/RMS    = %.3e  (cols=%d stride=%d)\n",
           (double)err, win.cols, win.row_stride);
    ASSERT_TRUE(err < 1e-5f);

    free(x); free(want); free(got);
    close_model_tensor_window(&win);
}

/* ================================================================
 * Tests — matvec at a nonzero region offset
 *
 * Additional to the absolute tests above, never a replacement: both arms
 * here are still checked against a CPU reference, so a uniform scale
 * cannot cancel out the way it does in a GPU-vs-GPU comparison.
 * ================================================================ */

static void test_matvec_at_nonzero_region_offset_is_still_absolutely_right(void)
{
    const int row_stride = (GOLDEN_COLS / 256) * Q3K_BLOCK_BYTES;
    const size_t weight_bytes = (size_t)GOLDEN_ROWS * row_stride;
    uint8_t *weight = load_golden_weight();
    float *x = load_golden_payload("q3k_matvec_input.bin", GOLDEN_COLS * sizeof(float));
    float *golden = load_golden_payload("q3k_matvec_output.bin", GOLDEN_ROWS * sizeof(float));
    ASSERT_NOT_NULL(weight);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(golden);

    const size_t map_bytes = 4u * 1024u * 1024u;
    int fd = -1;
    void *base = map_scratch(map_bytes, &fd);
    ASSERT_NOT_NULL(base);
    memcpy((uint8_t *)base + SCRATCH_WEIGHT_OFFSET, weight, weight_bytes);

    ASSERT_INT_EQ(0, SCRATCH_WEIGHT_OFFSET % 32);
    ASSERT_INT_NE(0, SCRATCH_WEIGHT_OFFSET % 16384);

    fm_region *region = fm_region_open(base, map_bytes, 0);
    ASSERT_NOT_NULL(region);
    fm_bufref wref;
    ASSERT_INT_EQ(0, fm_region_bind(region, SCRATCH_WEIGHT_OFFSET, weight_bytes, &wref));

    fm_buffer *bx = fm_buffer_alloc(GOLDEN_COLS * sizeof(float));
    fm_buffer *bo = fm_buffer_alloc(GOLDEN_ROWS * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bo);
    memcpy(fm_buffer_data(bx), x, GOLDEN_COLS * sizeof(float));
    memset(fm_buffer_data(bo), 0, GOLDEN_ROWS * sizeof(float));

    fm_matvec_q3k_at(wref, fm_bufref_at(bx, 0), fm_bufref_at(bo, 0),
                     GOLDEN_ROWS, GOLDEN_COLS, row_stride);
    fm_sync();

    float err = max_abs_error_over_rms((const float *)fm_buffer_data(bo),
                                       golden, GOLDEN_ROWS);
    printf("      offset %u max_abs/RMS      = %.3e\n",
           (unsigned)SCRATCH_WEIGHT_OFFSET, (double)err);
    ASSERT_TRUE(err < 1e-5f);

    fm_buffer_free(bx);
    fm_buffer_free(bo);
    fm_region_close(region);
    unmap_scratch(base, map_bytes, fd);
    free(weight); free(x); free(golden);
}

static void test_matvec_reads_a_real_tensor_through_a_region(void)
{
    model_tensor_window win;
    ASSERT_INT_EQ(0, open_model_tensor_window(&win, MODEL_ROWS));

    /* This is the whole point of fm_bufref: a real GGUF tensor is 32-byte
     * aligned and never page-aligned, so offset-0 binding cannot reach it. */
    ASSERT_INT_EQ(0, win.tensor_offset % 32);
    ASSERT_INT_NE(0, win.tensor_offset % (size_t)getpagesize());

    const size_t weight_bytes = (size_t)MODEL_ROWS * (size_t)win.row_stride;
    fm_region *region = fm_region_open(win.window, win.window_bytes, 0);
    ASSERT_NOT_NULL(region);
    fm_bufref wref;
    ASSERT_INT_EQ(0, fm_region_bind(region, win.tensor_offset, weight_bytes, &wref));

    float *x = malloc((size_t)win.cols * sizeof(float));
    float *want = malloc(MODEL_ROWS * sizeof(float));
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(want);
    fill_pseudorandom(x, win.cols, 0xC0FFEEU);
    q3k_matvec_reference((const uint8_t *)win.window + win.tensor_offset,
                         x, want, MODEL_ROWS, win.cols, win.row_stride);

    fm_buffer *bx = fm_buffer_alloc((size_t)win.cols * sizeof(float));
    fm_buffer *bo = fm_buffer_alloc(MODEL_ROWS * sizeof(float));
    ASSERT_NOT_NULL(bx);
    ASSERT_NOT_NULL(bo);
    memcpy(fm_buffer_data(bx), x, (size_t)win.cols * sizeof(float));
    memset(fm_buffer_data(bo), 0, MODEL_ROWS * sizeof(float));

    fm_matvec_q3k_at(wref, fm_bufref_at(bx, 0), fm_bufref_at(bo, 0),
                     MODEL_ROWS, win.cols, win.row_stride);
    fm_sync();

    float err = max_abs_error_over_rms((const float *)fm_buffer_data(bo),
                                       want, MODEL_ROWS);
    printf("      model tensor @+%zu max_abs/RMS = %.3e\n",
           win.tensor_offset, (double)err);
    ASSERT_TRUE(err < 1e-5f);

    fm_buffer_free(bx);
    fm_buffer_free(bo);
    fm_region_close(region);
    free(x); free(want);
    close_model_tensor_window(&win);
}

/* ================================================================
 * Tests — the ragged row tail
 * ================================================================ */

/* Floats of sentinel written past the output rows, to catch a store from a
 * lane whose row index ran off the end of the grid's last threadgroup. */
#define TAIL_GUARD_FLOATS  16
#define TAIL_GUARD_VALUE   (-12345.0f)

/*
 * Runs fm_matvec_q3k over an output buffer padded with a sentinel.
 *
 * @param weight        Q3_K bytes, at least rows * row_stride of them
 * @param rows, cols    Shape to dispatch; cols must be a multiple of 256
 * @param row_stride    Bytes per weight row, (cols/256)*110
 * @param tail_intact   Set true only if every sentinel past `rows` survived
 * @return              max_abs/RMS over the valid rows, or -1 if a buffer
 *                      could not be allocated
 */
static float matvec_error_with_tail_guard(const uint8_t *weight, int rows,
                                          int cols, int row_stride,
                                          bool *tail_intact)
{
    float *x = malloc((size_t)cols * sizeof(float));
    float *want = malloc((size_t)rows * sizeof(float));
    fm_buffer *bw = fm_buffer_alloc((size_t)rows * (size_t)row_stride);
    fm_buffer *bx = fm_buffer_alloc((size_t)cols * sizeof(float));
    fm_buffer *bo = fm_buffer_alloc((size_t)(rows + TAIL_GUARD_FLOATS) * sizeof(float));
    float err = -1.0f;

    *tail_intact = false;
    if (!x || !want || !bw || !bx || !bo) goto cleanup;

    fill_pseudorandom(x, cols, 0x51EDU + (uint32_t)rows);
    q3k_matvec_reference(weight, x, want, rows, cols, row_stride);

    memcpy(fm_buffer_data(bw), weight, (size_t)rows * (size_t)row_stride);
    memcpy(fm_buffer_data(bx), x, (size_t)cols * sizeof(float));
    float *o = (float *)fm_buffer_data(bo);
    for (int i = 0; i < rows + TAIL_GUARD_FLOATS; i++)
        o[i] = TAIL_GUARD_VALUE;

    fm_matvec_q3k(bw, bx, bo, rows, cols, row_stride);
    fm_sync();

    o = (float *)fm_buffer_data(bo);
    *tail_intact = true;
    for (int i = rows; i < rows + TAIL_GUARD_FLOATS; i++)
        if (o[i] != TAIL_GUARD_VALUE) *tail_intact = false;
    err = max_abs_error_over_rms(o, want, rows);

cleanup:
    if (bw) fm_buffer_free(bw);
    if (bx) fm_buffer_free(bx);
    if (bo) fm_buffer_free(bo);
    free(x);
    free(want);
    return err;
}

/*
 * A threadgroup covers 32/(cols/256) output rows, so the grid is rounded up and
 * the last group carries row indices past the end.  Those lanes must contribute
 * nothing to a live row's reduction and store nothing — a hazard that did not
 * exist while every threadgroup owned exactly one row.
 *
 * Swept over both column counts the golden weight can be reinterpreted at:
 * cols=512 splits a group 16 ways (2 lanes per row, the MoE expert shape) and
 * cols=1024 splits it 8 ways, with row counts chosen to leave every kind of
 * remainder including a single row and one short of a full group.
 */
static void test_ragged_row_tail_is_exact_and_writes_nothing_past_rows(void)
{
    static const int shape[][2] = {
        { 512, 1 }, { 512, 15 }, { 512, 17 }, { 512, 31 }, { 512, 33 },
        { 1024, 1 }, { 1024, 7 }, { 1024, 9 }, { 1024, 23 },
    };
    uint8_t *weight = load_golden_weight();
    ASSERT_NOT_NULL(weight);

    for (size_t i = 0; i < sizeof(shape) / sizeof(shape[0]); i++) {
        int cols = shape[i][0];
        int rows = shape[i][1];
        int row_stride = (cols / 256) * Q3K_BLOCK_BYTES;
        ASSERT_TRUE((size_t)rows * (size_t)row_stride <= GOLDEN_WEIGHT_BYTES);

        bool tail_intact = false;
        float err = matvec_error_with_tail_guard(weight, rows, cols, row_stride,
                                                 &tail_intact);
        printf("      tail cols=%-4d rows=%-3d max_abs/RMS = %.3e\n",
               cols, rows, (double)err);
        ASSERT_TRUE(tail_intact);
        ASSERT_TRUE(err >= 0.0f && err < 1e-5f);
    }
    free(weight);
}

/* ================================================================
 * Tests — batched _nx
 * ================================================================ */

/*
 * Runs fm_matvec_q3k_nx over `n` columns and returns the WORST per-column
 * max_abs/RMS against the CPU reference, so a tile-indexing bug cannot hide
 * behind a column that happens to land right.
 *
 * @return  Worst per-column error, or -1 if a buffer could not be allocated
 */
static float nx_worst_column_error(const uint8_t *weight, int rows, int cols,
                                   int row_stride, int n)
{
    const size_t weight_bytes = (size_t)rows * (size_t)row_stride;
    float *x = malloc((size_t)n * (size_t)cols * sizeof(float));
    float *want = malloc((size_t)n * (size_t)rows * sizeof(float));
    fm_buffer *bw = fm_buffer_alloc(weight_bytes);
    fm_buffer *bx = fm_buffer_alloc((size_t)n * (size_t)cols * sizeof(float));
    fm_buffer *bo = fm_buffer_alloc((size_t)n * (size_t)rows * sizeof(float));
    float worst = -1.0f;
    if (!x || !want || !bw || !bx || !bo) goto cleanup;

    fill_pseudorandom(x, n * cols, 0x9E37U + (uint32_t)n);
    for (int t = 0; t < n; t++)
        q3k_matvec_reference(weight, x + (size_t)t * cols, want + (size_t)t * rows,
                             rows, cols, row_stride);

    memcpy(fm_buffer_data(bw), weight, weight_bytes);
    memcpy(fm_buffer_data(bx), x, (size_t)n * (size_t)cols * sizeof(float));
    memset(fm_buffer_data(bo), 0, (size_t)n * (size_t)rows * sizeof(float));

    fm_matvec_q3k_nx(bw, bx, bo, rows, cols, row_stride, n);
    fm_sync();

    const float *got = (const float *)fm_buffer_data(bo);
    worst = 0.0f;
    for (int t = 0; t < n; t++) {
        float e = max_abs_error_over_rms(got + (size_t)t * rows,
                                         want + (size_t)t * rows, rows);
        if (e > worst) worst = e;
    }
    printf("      nx n=%-3d worst column max_abs/RMS = %.3e\n", n, (double)worst);

cleanup:
    if (bw) fm_buffer_free(bw);
    if (bx) fm_buffer_free(bx);
    if (bo) fm_buffer_free(bo);
    free(x);
    free(want);
    return worst;
}

static void test_nx_single_column_matches_the_reference(void)
{
    const int row_stride = (GOLDEN_COLS / 256) * Q3K_BLOCK_BYTES;
    uint8_t *weight = load_golden_weight();
    ASSERT_NOT_NULL(weight);
    float err = nx_worst_column_error(weight, GOLDEN_ROWS, GOLDEN_COLS, row_stride, 1);
    free(weight);
    ASSERT_TRUE(err >= 0.0f && err < 1e-5f);
}

static void test_nx_full_tile_matches_the_reference(void)
{
    const int row_stride = (GOLDEN_COLS / 256) * Q3K_BLOCK_BYTES;
    uint8_t *weight = load_golden_weight();
    ASSERT_NOT_NULL(weight);
    float err = nx_worst_column_error(weight, GOLDEN_ROWS, GOLDEN_COLS, row_stride,
                                      FM_Q3_K_NX_MAX_COLS);
    free(weight);
    ASSERT_TRUE(err >= 0.0f && err < 1e-5f);
}

/* n = 19 is two full 8-wide tiles plus a 3-wide remainder tile, so it
 * exercises both specialized pipelines and the tile->column arithmetic. */
static void test_nx_tiles_plus_remainder_match_the_reference(void)
{
    const int row_stride = (GOLDEN_COLS / 256) * Q3K_BLOCK_BYTES;
    uint8_t *weight = load_golden_weight();
    ASSERT_NOT_NULL(weight);
    float err = nx_worst_column_error(weight, GOLDEN_ROWS, GOLDEN_COLS, row_stride, 19);
    free(weight);
    ASSERT_TRUE(err >= 0.0f && err < 1e-5f);
}

static void test_nx_on_real_model_blocks_matches_the_reference(void)
{
    model_tensor_window win;
    ASSERT_INT_EQ(0, open_model_tensor_window(&win, MODEL_ROWS));
    float err = nx_worst_column_error((const uint8_t *)win.window + win.tensor_offset,
                                      MODEL_ROWS, win.cols, win.row_stride, 11);
    close_model_tensor_window(&win);
    ASSERT_TRUE(err >= 0.0f && err < 1e-5f);
}

/* ================================================================
 * main
 * ================================================================ */

static int fixtures_present(void)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/q3k_matvec_weight.bin", golden_dir());
    if (access(path, R_OK) != 0) {
        fprintf(stderr, "golden vectors not found at %s (set PN_GOLDEN_DIR)\n", golden_dir());
        return 0;
    }
    if (access(model_path(), R_OK) != 0) {
        fprintf(stderr, "model not found at %s (set NEX_N2_GGUF)\n", model_path());
        return 0;
    }
    return 1;
}

int main(void)
{
    printf("\n  Metal Q3_K Matvec Tests (absolute correctness)\n");
    printf("  =============================================\n\n");

    if (fm_init() != 0 || !fm_available()) {
        fprintf(stderr, "Metal unavailable\n");
        return 77;
    }
    if (!fixtures_present()) {
        fm_free();
        return 77;
    }

    printf("  golden dir = %s\n", golden_dir());
    printf("  model      = %s\n\n", model_path());

    RUN_TEST(reference_dequant_matches_the_golden_block);

    RUN_TEST(matvec_matches_the_golden_vectors);
    RUN_TEST(matvec_matches_the_cpu_reference_on_real_model_blocks);

    RUN_TEST(matvec_at_nonzero_region_offset_is_still_absolutely_right);
    RUN_TEST(matvec_reads_a_real_tensor_through_a_region);
    RUN_TEST(ragged_row_tail_is_exact_and_writes_nothing_past_rows);

    RUN_TEST(nx_single_column_matches_the_reference);
    RUN_TEST(nx_full_tile_matches_the_reference);
    RUN_TEST(nx_tiles_plus_remainder_match_the_reference);
    RUN_TEST(nx_on_real_model_blocks_matches_the_reference);

    fm_free();

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
