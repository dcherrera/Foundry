/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_matmul_block.c
 * @brief The BLOCKED resident matmul, checked off hardware: the reader's DRAM reads and L1
 *   layout, the compute thread's indexed unpacks, the pairing of A and B tiles that reaches
 *   each DST tile, and the destination the writer places that tile at.
 *
 * WHAT THIS SUITE IS FOR. `kMatmulColumnBlock` carries TWO blocks now. `NB` moves the reader
 * from one output column per pass to NB, which is what stops the A column being re-read once
 * per column. `MB` (the high half of the same word) holds MB A tile-rows and streams B past
 * all of them, which is what stops the WEIGHT column being re-read once per row. Nothing
 * about either is visible on the card: a wrong L1 stride pairs the right A tile with the
 * wrong weight tile, a wrong destination index writes a correct tile to the wrong address,
 * every kernel completes, every status word is clean, and the answer is silently wrong. So
 * the arithmetic is modelled here and checked against the DRAM tile identities, before a
 * card is involved.
 *
 * THE MODEL IS THE THREE KERNELS' OWN ARITHMETIC, transliterated:
 *   - resident_matmul_reader.c: for each pass it reserves `depth * MB` pages of CB_IN0 and
 *     `depth * NB` of CB_IN1, reads `rows` A blocks ROW-MAJOR and `columns` B blocks
 *     COLUMN-MAJOR, and pushes the FULL chunk on both. The pointers wrap on an EXACT hit of
 *     the limit and nowhere else.
 *   - resident_matmul_compute.c: for each k-step and each row it names A tile
 *     `row*depth + step` and B tile `step`, and the unpacker's MOP then walks `columns` SrcA
 *     tiles at a stride of `depth` tiles — which is what
 *     `ft_tensix_unpack_matmul_set_srca_tile_stride()` programs. DST base is `row*columns`.
 *   - resident_matmul_writer.c: it re-walks those loops and places output tile `(m, n)` at
 *     `m * pitch + n`, where the pitch is descriptor word a2 (0 meaning `tile_cols`).
 *
 * THE THREE CHECKS THAT MATTER:
 *   1. the (A tile, B tile) PAIRS reaching output tile `(m, n)` are exactly
 *      `{ (m*Kt + k, tile(n, k)) : k < Kt }` — that IS the matmul, stated as a multiset;
 *   2. the DESTINATION the writer computes for that tile is `m * pitch + n`;
 *   3. the DRAM tiles read obey `A = Mt*ceil(Nt/NB)*Kt` and `B = ceil(Mt/MB)*Nt*Kt`, whose
 *      sum is `Mt*Nt*Kt*(1/NB + 1/MB)` — the law the whole MB lever rests on. **In TILES
 *      that is symmetric and its optimum is MB = NB = 2. In BYTES it is not**, because NB
 *      divides the 2048 B activation tile and MB divides the 1088 B Bfp8_b weight tile: the
 *      byte optimum is MB=2/NB=2 by **2.0%** at Bfp8_b, MB=1/NB=4 at Bfp4_b, and MB=2/NB=2
 *      by 25% only at bf16 weights. Both laws are checked, because checking only the first
 *      is how "the row block is worth 1.25x" survives contact with a Q8_0 model.
 *
 * NO CARD. It opens no device and reads no artifact, so it runs on the Mac where the code is
 * written. It reads `kernels/resident_queue_protocol.h` through FT_TT_KERNEL_DIR to pin the
 * parameter-word contract; running it without that is a FAILURE, not a skip.
 *
 * @status Complete for the shapes Ornith dispatches (Mt = 1, N-major and K-major panels, NB
 *   in {1, 2, 4}, and the ragged Nt = 1 weight that 48 of Ornith's 249 projections are) and
 *   for every (MB, NB) the DST section admits.
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "resident_matmul_block_protocol.h"

/** Ornith's activation tile (bf16 32x32) and weight tile (Bfp8_b). */
#define A_TILE_BYTES 2048u
#define B_TILE_BYTES 1088u

/** The DST section at `dst_accum_mode 1`, SyncHalf: `(FT_TENSIX_DEST_REGISTER_HALF_SIZE >> 1)` rows
 *  over `1 << 6` rows a tile = 4. */
#define DST_SECTION_TILES FT_TT_MATMUL_DST_SECTION_TILES

#define MAX_TILES 4096u

/** One circular buffer as the firmware keeps it: a base, a limit and a page size, with the
 *  exact-hit wrap that `ft_tensix_cb_push_back()` and `..._pop_front()` perform. */
typedef struct model_cb {
    uint32_t page_bytes;
    uint32_t page_count;
    uint32_t base;
    uint32_t limit; /* exclusive here; the kernel's `fifo_limit` is the same address */
    uint32_t write_pointer;
    uint32_t read_pointer;
    /** Which DRAM tile each page currently holds, indexed by page. -1 is never written. */
    int32_t page_tile[MAX_TILES];
} model_cb;

/** Everything one descriptor says, plus the residency constants the CBs were sized for. */
typedef struct model_shape {
    const char *name;
    uint32_t tile_rows;  /* Mt */
    uint32_t tile_inner; /* Kt */
    uint32_t tile_cols;  /* Nt */
    uint32_t depth;
    uint32_t column_block; /* NB */
    uint32_t row_block;    /* MB */
    int b_n_major;
    /** The DST section this shape's KERNEL was built for, in tiles. 0 means this build's
     *  own `FT_TT_MATMUL_DST_SECTION_TILES`, which is the SyncHalf four. */
    uint32_t section_tiles;
} model_shape;

/** The section a shape is checked against: its own, or this build's if it names none. */
static uint32_t shape_section(const model_shape *shape)
{
    return shape->section_tiles ? shape->section_tiles : DST_SECTION_TILES;
}

/** What the compute thread issued: one (A tile, B tile) pair into one DST tile. */
typedef struct model_pair {
    uint32_t row;    /* the OUTPUT row this DST tile becomes */
    uint32_t column; /* the OUTPUT column this DST tile becomes */
    uint32_t dst;
    int32_t a_tile;
    int32_t b_tile;
} model_pair;

/** One tile leaving CB_OUT: which output element it is, and where the writer put it. */
typedef struct model_out {
    uint32_t row;
    uint32_t column;
    uint32_t destination; /* the tile index the writer computed */
} model_out;

typedef struct model_result {
    uint64_t a_tiles_read;
    uint64_t b_tiles_read;
    uint32_t max_dst;
    int off_grid;       /* a pointer left the chunk grid */
    int out_of_bounds;  /* a pointer or an indexed read left the buffer */
    int unwritten;      /* the compute named a page the reader never filled */
    uint64_t pair_count;
    model_pair *pairs;
    uint64_t pair_capacity;
    /* CB_OUT: what the writer did with each tile, in the order the packer pushed them. */
    model_out *out;
    uint64_t out_count;
} model_result;

/** The mutations this suite proves red with. The shipped values are all zero. */
typedef struct model_mutation {
    uint32_t stride_tiles;  /* the MOP's SrcA step; shipped is `depth` */
    int b_k_major;          /* lay B out K-major in L1 instead of column-major */
    int a_row_shared;       /* every output row unpacks A tile `step` — i.e. row 0's */
    int writer_counter;     /* the writer places tiles with a running counter, not (m, n) */
} model_mutation;

/* ------------------------------------------------------------------------------------
 * The kernels' pointer arithmetic
 * ---------------------------------------------------------------------------------- */

static void model_cb_init(model_cb *cb, uint32_t page_bytes, uint32_t page_count, uint32_t base)
{
    uint32_t page;

    cb->page_bytes = page_bytes;
    cb->page_count = page_count;
    cb->base = base;
    cb->limit = base + page_bytes * page_count;
    cb->write_pointer = base;
    cb->read_pointer = base;
    for (page = 0; page < MAX_TILES; page++)
        cb->page_tile[page] = -1;
}

/** `ft_tensix_cb_push_back()`: advance, then wrap on an EXACT hit of the limit. */
static void model_cb_push(model_cb *cb, uint32_t pages)
{
    cb->write_pointer += pages * cb->page_bytes;
    if (cb->write_pointer == cb->limit)
        cb->write_pointer = cb->base;
}

/** `ft_tensix_compute_cb_pop_front()`: the same rule on the read side. */
static void model_cb_pop(model_cb *cb, uint32_t pages)
{
    cb->read_pointer += pages * cb->page_bytes;
    if (cb->read_pointer >= cb->limit)
        cb->read_pointer -= cb->page_bytes * cb->page_count;
}

/** `ft_tensix_cb_block_grid_is_safe()`, the producer's form. */
static int model_cb_on_grid(const model_cb *cb, uint32_t chunk_pages, uint32_t pointer)
{
    if (chunk_pages == 0u || chunk_pages > cb->page_count || cb->page_count % chunk_pages != 0u)
        return 0;
    if (pointer < cb->base || pointer >= cb->limit)
        return 0;
    return ((pointer - cb->base) % (chunk_pages * cb->page_bytes)) == 0;
}

/** Which page of the buffer an L1 address names, or -1 if it is outside. */
static int32_t model_cb_page_of(const model_cb *cb, uint32_t address)
{
    if (address < cb->base || address >= cb->limit)
        return -1;
    if ((address - cb->base) % cb->page_bytes != 0u)
        return -1;
    return (int32_t)((address - cb->base) / cb->page_bytes);
}

/* ------------------------------------------------------------------------------------
 * The three kernels, run in lockstep one pass at a time
 * ---------------------------------------------------------------------------------- */

/** The DRAM tile identity of weight tile (column, k), as the reader computes its offset. */
static int32_t model_b_tile(const model_shape *shape, uint32_t column, uint32_t k)
{
    return shape->b_n_major ? (int32_t)(column * shape->tile_inner + k)
                            : (int32_t)(k * shape->tile_cols + column);
}

/**
 * Run one descriptor through all three thirds of the triad.
 *
 * LOCKSTEP IS A SIMPLIFICATION AND IT IS THE CONSERVATIVE ONE. On the card the reader may be
 * a whole chunk ahead; here it fills a pass and the compute thread drains it immediately.
 * That cannot hide a layout error — the addresses are the same either way — and it lets the
 * model check that the page the compute NAMES is the page the reader WROTE, which is the
 * property a stride error breaks.
 */
static void model_run(const model_shape *shape, const model_mutation *mutate, model_result *out)
{
    model_cb in0;
    model_cb in1;
    const uint32_t depth = shape->depth;
    const uint32_t block = shape->column_block;
    const uint32_t rblock = shape->row_block;
    const uint32_t chunk = depth * block;
    const uint32_t a_chunk = depth * rblock;
    const uint32_t stride_tiles = mutate->stride_tiles ? mutate->stride_tiles : depth;
    /* The host sends the weight's FULL Nt; a single member's descriptor carries its own
     * slice in `tile_cols`. Modelled as the whole tensor, which is the un-sharded case. */
    const uint32_t pitch = shape->tile_cols;
    uint32_t mt;
    uint32_t nt;
    uint32_t kt;
    uint32_t step;
    uint32_t column;
    uint32_t row;
    uint32_t counter = 0u;

    model_cb_init(&in0, A_TILE_BYTES, a_chunk * 2u, 0x10000u);
    model_cb_init(&in1, B_TILE_BYTES, chunk * 2u, 0x40000u);
    out->a_tiles_read = 0;
    out->b_tiles_read = 0;
    out->max_dst = 0;
    out->off_grid = 0;
    out->out_of_bounds = 0;
    out->unwritten = 0;
    out->pair_count = 0;
    out->out_count = 0;

    for (mt = 0; mt < shape->tile_rows; mt += rblock) {
        const uint32_t rows =
            (shape->tile_rows - mt) < rblock ? (shape->tile_rows - mt) : rblock;

        for (nt = 0; nt < shape->tile_cols; nt += block) {
            const uint32_t columns =
                (shape->tile_cols - nt) < block ? (shape->tile_cols - nt) : block;

            for (kt = 0; kt < shape->tile_inner; kt += depth) {
                uint32_t a_write;
                uint32_t b_write;

                /* --- the reader --- */
                if (!model_cb_on_grid(&in0, a_chunk, in0.write_pointer) ||
                    !model_cb_on_grid(&in1, chunk, in1.write_pointer))
                    out->off_grid = 1;
                a_write = in0.write_pointer;
                b_write = in1.write_pointer;
                if (a_write + a_chunk * A_TILE_BYTES > in0.limit ||
                    b_write + chunk * B_TILE_BYTES > in1.limit)
                    out->out_of_bounds = 1;

                for (row = 0; row < rows; row++) {
                    for (step = 0; step < depth; step++) {
                        /* ROW-MAJOR IN L1: row i's `depth` tiles at `i * depth`. */
                        const uint32_t slot = row * depth + step;
                        int32_t page = model_cb_page_of(&in0, a_write + slot * A_TILE_BYTES);

                        if (page < 0) {
                            out->out_of_bounds = 1;
                            continue;
                        }
                        in0.page_tile[page] =
                            (int32_t)((mt + row) * shape->tile_inner + kt + step);
                        out->a_tiles_read++;
                    }
                }
                for (column = 0; column < columns; column++) {
                    for (step = 0; step < depth; step++) {
                        /* COLUMN-MAJOR IN L1: column j's `depth` tiles at `j * depth`, which
                         * is what keeps the N-major panel's per-column read whole. The
                         * mutation lays them K-major instead. */
                        const uint32_t slot = mutate->b_k_major ? step * columns + column
                                                                : column * depth + step;
                        int32_t page = model_cb_page_of(&in1, b_write + slot * B_TILE_BYTES);

                        if (page < 0) {
                            out->out_of_bounds = 1;
                            continue;
                        }
                        in1.page_tile[page] = model_b_tile(shape, nt + column, kt + step);
                        out->b_tiles_read++;
                    }
                }
                model_cb_push(&in0, a_chunk);
                model_cb_push(&in1, chunk);

                /* --- the compute thread --- */
                for (step = 0; step < depth; step++) {
                    for (row = 0; row < rows; row++) {
                        /* `tile_index_a` is `row * depth + step`; the mutation shares row
                         * 0's activation across every output row. */
                        const uint32_t a_index = mutate->a_row_shared ? step : row * depth + step;
                        const uint32_t a_address = in0.read_pointer + a_index * A_TILE_BYTES;
                        const uint32_t b_base = in1.read_pointer + step * B_TILE_BYTES;

                        for (column = 0; column < columns; column++) {
                            /* The MOP bumps SrcA by the tile-size GPR once per iteration. */
                            const uint32_t b_address =
                                b_base + column * stride_tiles * B_TILE_BYTES;
                            const uint32_t dst = row * columns + column;
                            int32_t a_page = model_cb_page_of(&in0, a_address);
                            int32_t b_page = model_cb_page_of(&in1, b_address);
                            model_pair *pair;

                            if (dst >= shape_section(shape))
                                out->max_dst = shape_section(shape) + 1u;
                            else if (dst + 1u > out->max_dst)
                                out->max_dst = dst + 1u;
                            if (a_page < 0 || b_page < 0) {
                                out->out_of_bounds = 1;
                                continue;
                            }
                            if (in0.page_tile[a_page] < 0 || in1.page_tile[b_page] < 0) {
                                out->unwritten = 1;
                                continue;
                            }
                            if (out->pair_count >= out->pair_capacity)
                                continue;
                            pair = &out->pairs[out->pair_count++];
                            pair->row = mt + row;
                            pair->column = nt + column;
                            pair->dst = dst;
                            pair->a_tile = in0.page_tile[a_page];
                            pair->b_tile = in1.page_tile[b_page];
                        }
                    }
                }
                model_cb_pop(&in0, a_chunk);
                model_cb_pop(&in1, chunk);
            }

            /* --- the packer, then the writer: one tile at a time, ROW-MAJOR within the
             *     block because that is the order the DST indices were assigned in. --- */
            for (row = 0; row < rows; row++) {
                for (column = 0; column < columns; column++) {
                    model_out *entry = &out->out[out->out_count++];

                    entry->row = mt + row;
                    entry->column = nt + column;
                    entry->destination = mutate->writer_counter
                                             ? counter
                                             : (mt + row) * pitch + nt + column;
                    counter++;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------------------
 * The checks
 * ---------------------------------------------------------------------------------- */

/**
 * Every pair that reached output tile `(m, c)` must be `(m*Kt + k, tile(c, k))` for a
 * distinct k, and there must be exactly Kt of them. THIS IS THE MATMUL, and it is what a
 * stride error, an L1 layout error, a DST-index error or a shared activation row all break.
 */
static int pairs_are_the_matmul(const model_shape *shape, const model_result *result)
{
    uint32_t column;
    uint32_t m;
    unsigned char *seen = calloc(shape->tile_inner, 1u);
    int ok = 1;

    if (!seen)
        return 0;
    for (m = 0; m < shape->tile_rows && ok; m++) {
        for (column = 0; column < shape->tile_cols && ok; column++) {
            uint64_t index;
            uint32_t found = 0;

            memset(seen, 0, shape->tile_inner);
            for (index = 0; index < result->pair_count; index++) {
                const model_pair *pair = &result->pairs[index];
                uint32_t k;

                if (pair->column != column || pair->row != m)
                    continue;
                if (pair->a_tile < 0 ||
                    (uint32_t)pair->a_tile < m * shape->tile_inner ||
                    (uint32_t)pair->a_tile >= (m + 1u) * shape->tile_inner) {
                    ok = 0;
                    break;
                }
                k = (uint32_t)pair->a_tile - m * shape->tile_inner;
                if (seen[k] || pair->b_tile != model_b_tile(shape, column, k)) {
                    ok = 0;
                    break;
                }
                seen[k] = 1u;
                found++;
            }
            if (found != shape->tile_inner)
                ok = 0;
        }
    }
    free(seen);
    return ok;
}

/** Output tile `(m, n)` must leave for destination `m * Nt + n`, exactly once. */
static int writer_places_every_tile(const model_shape *shape, const model_result *result)
{
    const uint64_t total = (uint64_t)shape->tile_rows * shape->tile_cols;
    unsigned char *seen;
    uint64_t index;
    int ok = 1;

    if (result->out_count != total)
        return 0;
    seen = calloc((size_t)total, 1u);
    if (!seen)
        return 0;
    for (index = 0; index < result->out_count; index++) {
        const model_out *entry = &result->out[index];
        const uint64_t want = (uint64_t)entry->row * shape->tile_cols + entry->column;

        if (entry->destination != want || want >= total || seen[want]) {
            ok = 0;
            break;
        }
        seen[want] = 1u;
    }
    free(seen);
    return ok;
}

/** The A tiles a descriptor costs: one column per ROW per column BLOCK. Unmoved by MB. */
static uint64_t expected_a_tiles(const model_shape *shape)
{
    const uint64_t passes = (shape->tile_cols + shape->column_block - 1u) / shape->column_block;

    return (uint64_t)shape->tile_rows * passes * shape->tile_inner;
}

/** The B tiles a descriptor costs: one column per COLUMN per row BLOCK. Divided by MB. */
static uint64_t expected_b_tiles(const model_shape *shape)
{
    const uint64_t passes = (shape->tile_rows + shape->row_block - 1u) / shape->row_block;

    return passes * shape->tile_cols * shape->tile_inner;
}

static const model_mutation kShipped = { 0u, 0, 0, 0 };

static void check_shape(const model_shape *shape, model_result *result)
{
    uint64_t index;
    int order_ok = 1;

    model_run(shape, &kShipped, result);

    ft_tt_test_step("shape %s — Mt %u Kt %u Nt %u depth %u MBxNB %ux%u %s, DST %u",
                    shape->name, shape->tile_rows, shape->tile_inner, shape->tile_cols,
                    shape->depth, shape->row_block, shape->column_block,
                    shape->b_n_major ? "N-major" : "K-major", shape_section(shape));
    ft_tt_test_check_fmt("  the A x B pairs reaching every output TILE ARE the matmul",
                         pairs_are_the_matmul(shape, result),
                         "%s: %llu pairs over %u x %u tiles", shape->name,
                         (unsigned long long)result->pair_count, shape->tile_rows,
                         shape->tile_cols);
    ft_tt_test_check_fmt("  A tiles read = Mt * ceil(Nt/NB) * Kt, UNMOVED by MB",
                         result->a_tiles_read == expected_a_tiles(shape),
                         "%s: %llu, want %llu", shape->name,
                         (unsigned long long)result->a_tiles_read,
                         (unsigned long long)expected_a_tiles(shape));
    ft_tt_test_check_fmt("  B tiles read = ceil(Mt/MB) * Nt * Kt, DIVIDED by MB",
                         result->b_tiles_read == expected_b_tiles(shape),
                         "%s: %llu, want %llu (MB=%u divides the %llu an unblocked reader "
                         "costs)", shape->name, (unsigned long long)result->b_tiles_read,
                         (unsigned long long)expected_b_tiles(shape), shape->row_block,
                         (unsigned long long)((uint64_t)shape->tile_rows * shape->tile_cols *
                                              shape->tile_inner));
    ft_tt_test_check_fmt("  every CB pointer stays on its chunk grid", !result->off_grid, "%s",
                         shape->name);
    ft_tt_test_check_fmt("  no read or write leaves a circular buffer", !result->out_of_bounds,
                         "%s", shape->name);
    ft_tt_test_check_fmt("  the compute thread names no page the reader did not fill",
                         !result->unwritten, "%s", shape->name);
    ft_tt_test_check_fmt("  DST tiles held at once are MB*NB and fit the section",
                         result->max_dst <= shape_section(shape) &&
                             result->max_dst <= shape->row_block * shape->column_block,
                         "%s: %u of %u", shape->name, result->max_dst, shape_section(shape));
    /* THE REASON NO DST WRITE-ADDRESS SITE CHANGES AT SyncFull. Every one of them computes
     * `(tile_index << DstTileSizeLog2) + (dest_offset_id ? HALF : 0)`, and SyncFull pins the
     * id at 0, so the highest address a section of `section_tiles` reaches is
     * `(section_tiles - 1) << 6`. At eight tiles that is 448, one tile inside
     * `DEST_REGISTER_FULL_SIZE >> 1 = 512`. The formula fits the bigger section exactly. */
    ft_tt_test_check_fmt("  the DST write address of the last tile is inside the section",
                         result->max_dst == 0u ||
                             ((result->max_dst - 1u) << 6) + 64u <= (shape_section(shape) << 6),
                         "%s: tile %u lands at row %u of %u", shape->name,
                         result->max_dst ? result->max_dst - 1u : 0u,
                         result->max_dst ? (result->max_dst - 1u) << 6 : 0u,
                         shape_section(shape) << 6);
    ft_tt_test_check_fmt("  the writer places every output tile at m*Nt + n, exactly once",
                         writer_places_every_tile(shape, result), "%s: %llu tiles",
                         shape->name, (unsigned long long)result->out_count);

    /* At MB = 1 the destination sequence is the RUNNING COUNTER the pre-MB writer used, and
     * that identity is what makes a decode `cmp` evidence about this change. */
    if (shape->row_block == 1u) {
        for (index = 0; index < result->out_count; index++)
            if (result->out[index].destination != (uint32_t)index)
                order_ok = 0;
        ft_tt_test_check_fmt("  at MB=1 the destinations ARE the old running counter",
                             order_ok, "%s: %llu tiles", shape->name,
                             (unsigned long long)result->out_count);
    }
}

/**
 * The mutations that must be caught, run on a shape where both blocks are real.
 *
 * A suite that has never been seen to fail is not evidence (best-practices §9), and none of
 * these are synthetic: an SrcA stride of one tile is what the hardware does if
 * `ft_tensix_unpack_matmul_set_srca_tile_stride()` is not called; a K-major L1 layout is the
 * other design the reader could have used; a shared activation index is what the compute
 * thread does if the row loop is added without moving `tile_index_a`; and a running-counter
 * writer is the loop this file's writer USED TO BE, which is correct at MB=1 and silently
 * transposes the output block at MB=2.
 */
static void check_red_arms(model_result *result)
{
    const model_shape wide = { "red-NB", 1u, 32u, 8u, 8u, 4u, 1u, 1, 0u };
    const model_shape square = { "red-MBxNB", 4u, 32u, 8u, 8u, 2u, 2u, 1, 0u };
    model_mutation mutate;

    ft_tt_test_step("RED — the mutations that must NOT pass");

    mutate = kShipped;
    mutate.stride_tiles = 1u;
    model_run(&wide, &mutate, result);
    ft_tt_test_check_fmt("an SrcA stride of ONE tile mis-pairs the column block",
                         !pairs_are_the_matmul(&wide, result),
                         "the un-strided MOP reads tiles 0..NB-1 of column 0's block");

    mutate = kShipped;
    mutate.b_k_major = 1;
    model_run(&wide, &mutate, result);
    ft_tt_test_check_fmt("a K-major L1 layout under the shipped stride mis-pairs the block",
                         !pairs_are_the_matmul(&wide, result),
                         "the reader's slot and the MOP's stride must agree");

    mutate = kShipped;
    mutate.a_row_shared = 1;
    model_run(&square, &mutate, result);
    ft_tt_test_check_fmt("a compute thread that does NOT advance tile_index_a by row*depth "
                         "feeds every output row row 0's activation",
                         !pairs_are_the_matmul(&square, result),
                         "MB is a loop AND an index; the loop alone is a wrong answer");

    mutate = kShipped;
    mutate.writer_counter = 1;
    model_run(&square, &mutate, result);
    ft_tt_test_check_fmt("a running-counter writer places the MB>1 block at the wrong "
                         "destination",
                         !writer_places_every_tile(&square, result),
                         "correct at MB=1, silently transposed at MB=2");

    /* THE FAILURE THE `PN_TT_MATMUL_DST_TILES` INTERLOCK EXISTS FOR: a host that believes
     * the section is eight tiles driving a kernel built at four. Nothing refuses on the
     * card -- DST has no bounds check and tiles 4..7 are the OTHER half, which the packer is
     * reading -- so the model is the only place it can be seen. It is the same shape twice,
     * differing ONLY in the section it is checked against. */
    {
        model_shape too_wide = { "red-section", 1u, 32u, 32u, 8u, 8u, 1u, 1, 4u };

        model_run(&too_wide, &kShipped, result);
        ft_tt_test_check_fmt("NB=8 against a FOUR-tile section writes outside it",
                             result->max_dst > 4u, "highest DST tile reached is %u of 4",
                             result->max_dst);
        too_wide.section_tiles = 8u;
        model_run(&too_wide, &kShipped, result);
        ft_tt_test_check_fmt("  CONTROL: the same NB=8 against EIGHT tiles is inside it",
                             result->max_dst == 8u, "highest DST tile reached is %u of 8",
                             result->max_dst);
    }

    /* THE CONTROL FOR THE TWO ABOVE. Both mutations are run on a shape the SHIPPED model
     * passes, so a red that fired because the shape itself was broken is excluded. */
    model_run(&square, &kShipped, result);
    ft_tt_test_check_fmt("CONTROL: the same MB=2 NB=2 shape PASSES unmutated",
                         pairs_are_the_matmul(&square, result) &&
                             writer_places_every_tile(&square, result),
                         "the reds above are about the mutation, not the shape");
}

/* ------------------------------------------------------------------------------------
 * The byte law, and where its optimum is
 * ---------------------------------------------------------------------------------- */

/**
 * THE LAW IN TILES: `total = Mt*Nt*Kt*(1/NB + 1/MB)`, symmetric in the two blocks, and the
 * DST section caps `MB*NB` at four. In TILES the optimum is therefore MB = NB = 2.
 *
 * THE LAW IN BYTES IS NOT SYMMETRIC AND THAT IS THE RESULT:
 * `bytes = Mt*T*(a/NB + b/MB)`, because NB divides the operand whose tile is `a` (the bf16
 * ACTIVATION, 2048 B) and MB divides the one whose tile is `b` (the WEIGHT — 1088 B at
 * Bfp8_b, 576 at Bfp4_b). So the blocking budget belongs to whichever operand has the
 * bigger tile, and on a Q8_0 model that is the ACTIVATION. `check_the_byte_law` asks the
 * model both questions, because answering only the first is how "MB = NB = 2 is worth 1.25x"
 * survives — it is true in tiles and false in bytes at every weight format Ornith uses.
 */
static void check_the_tile_law(model_result *result)
{
    static const struct { uint32_t rows; uint32_t columns; } pairs[] = {
        { 1u, 1u }, { 1u, 2u }, { 1u, 4u }, { 2u, 1u }, { 2u, 2u }, { 4u, 1u },
    };
    /* Divisible on both axes so the closed form is exact rather than a ceiling. */
    model_shape shape = { "law", 4u, 32u, 8u, 8u, 1u, 1u, 1, 0u };
    uint64_t best = 0;
    uint32_t best_rows = 0;
    uint32_t best_columns = 0;
    unsigned index;

    ft_tt_test_step("the TILE law, and the (MB, NB) the DST section makes optimal IN TILES");
    for (index = 0; index < sizeof(pairs) / sizeof(pairs[0]); index++) {
        const uint64_t unblocked =
            (uint64_t)shape.tile_rows * shape.tile_cols * shape.tile_inner;
        uint64_t total;

        shape.row_block = pairs[index].rows;
        shape.column_block = pairs[index].columns;
        ft_tt_test_check_fmt("the pair fits the DST section",
                             ftmm_block_fits_dst(shape.row_block, shape.column_block) != 0,
                             "MB=%u NB=%u: %u tiles of %u", shape.row_block, shape.column_block,
                             shape.row_block * shape.column_block, DST_SECTION_TILES);
        model_run(&shape, &kShipped, result);
        total = result->a_tiles_read + result->b_tiles_read;
        ft_tt_test_check_fmt("  it reads Mt*Nt*Kt*(1/NB + 1/MB) tiles",
                             total * shape.row_block * shape.column_block ==
                                 unblocked * (shape.row_block + shape.column_block),
                             "MB=%u NB=%u: %llu tiles against the law's %llu", shape.row_block,
                             shape.column_block, (unsigned long long)total,
                             (unsigned long long)(unblocked *
                                                  (shape.row_block + shape.column_block) /
                                                  (shape.row_block * shape.column_block)));
        if (best == 0 || total < best) {
            best = total;
            best_rows = shape.row_block;
            best_columns = shape.column_block;
        }
    }
    ft_tt_test_check_fmt("the DST-4 optimum IN TILES is MB = NB = 2, not either extreme",
                         best_rows == 2u && best_columns == 2u, "best is MB=%u NB=%u",
                         best_rows, best_columns);

    shape.row_block = 4u;
    shape.column_block = 1u;
    model_run(&shape, &kShipped, result);
    {
        const uint64_t rows4 = result->a_tiles_read + result->b_tiles_read;

        shape.row_block = 1u;
        shape.column_block = 4u;
        model_run(&shape, &kShipped, result);
        ft_tt_test_check_fmt("MB=4/NB=1 reads EXACTLY as many tiles as the shipped MB=1/NB=4",
                             rows4 == result->a_tiles_read + result->b_tiles_read,
                             "%llu against %llu — 128 positions a weight sweep buys nothing "
                             "when A pays it all back",
                             (unsigned long long)rows4,
                             (unsigned long long)(result->a_tiles_read + result->b_tiles_read));
    }

    ft_tt_test_check_fmt("MB*NB = 5 is REFUSED, which is what the kernel guard asks",
                         ftmm_block_fits_dst(5u, 1u) == 0 && ftmm_block_fits_dst(2u, 3u) == 0 &&
                             ftmm_block_fits_dst(0u, 4u) == 0,
                         "the section is %u tiles", DST_SECTION_TILES);
}

/** `Mt * T * (a/NB + b/MB)` in BYTES a weight tile, from the model's own tile counts. */
static uint64_t law_bytes(const model_result *result, uint32_t weight_tile)
{
    return result->a_tiles_read * A_TILE_BYTES + result->b_tiles_read * weight_tile;
}

/**
 * THE BYTE LAW, WHICH IS NOT THE TILE LAW, AND THE REASON THE ROW BLOCK IS WORTH 2%.
 *
 * NB divides the operand whose tile is the bf16 ACTIVATION (2048 B) and MB divides the one
 * whose tile is the WEIGHT (1088 B at Bfp8_b, 576 at Bfp4_b, 2048 at bf16). So the two
 * blocks are symmetric in tiles and NOT in bytes, and the budget belongs to whichever
 * operand has the bigger tile.
 *
 * THE LAW IS VALIDATED AGAINST SILICON WITH NO FREE PARAMETER and that check is here:
 * at Mt=1/MB=1/NB=4 with Bfp8_b it says `A/B = a/(4b) = 2048/4352 = 0.4706`, and Ornith's
 * measured decode split is A 3.978 GB / B 8.432 GB = 0.4718. Nothing was fitted.
 */
static void check_the_byte_law(model_result *result)
{
    static const struct { const char *name; uint32_t tile; uint32_t rows; uint32_t columns; }
        formats[] = {
            /* the shipped Ornith weight (Q8_0 -> Bfp8_b) and the two others in the tree */
            { "Bfp8_b", 1088u, 2u, 2u },
            { "Bfp4_b", 576u, 1u, 4u },
            { "bf16", 2048u, 2u, 2u },
        };
    static const struct { uint32_t rows; uint32_t columns; } pairs[] = {
        { 1u, 1u }, { 1u, 2u }, { 1u, 4u }, { 2u, 1u }, { 2u, 2u }, { 4u, 1u },
    };
    model_shape shape = { "law", 4u, 32u, 8u, 8u, 1u, 1u, 1, 0u };
    unsigned format;
    unsigned index;

    ft_tt_test_step("the BYTE law, and why the optimum moves with the WEIGHT FORMAT");
    for (format = 0; format < sizeof(formats) / sizeof(formats[0]); format++) {
        uint64_t best = 0;
        uint32_t best_rows = 0;
        uint32_t best_columns = 0;

        for (index = 0; index < sizeof(pairs) / sizeof(pairs[0]); index++) {
            uint64_t bytes;

            shape.row_block = pairs[index].rows;
            shape.column_block = pairs[index].columns;
            model_run(&shape, &kShipped, result);
            bytes = law_bytes(result, formats[format].tile);
            if (best == 0 || bytes < best) {
                best = bytes;
                best_rows = shape.row_block;
                best_columns = shape.column_block;
            }
        }
        ft_tt_test_check_fmt("  the byte optimum at this weight format",
                             best_rows == formats[format].rows &&
                                 best_columns == formats[format].columns,
                             "%s (%u B/tile): best MB=%u NB=%u, want MB=%u NB=%u",
                             formats[format].name, formats[format].tile, best_rows,
                             best_columns, formats[format].rows, formats[format].columns);
    }

    /* THE NUMBER THE WHOLE RUNG TURNS ON. At Ornith's Q8_0 weights the row block moves the
     * matmul's DRAM bytes by 2%, not by 1.25x -- the 1.25x is the TILE answer and it holds
     * only when the two tiles are the same size, which is the bf16 row above. */
    {
        uint64_t shipped, square;

        shape.row_block = 1u;
        shape.column_block = 4u;
        model_run(&shape, &kShipped, result);
        shipped = law_bytes(result, 1088u);
        shape.row_block = 2u;
        shape.column_block = 2u;
        model_run(&shape, &kShipped, result);
        square = law_bytes(result, 1088u);
        ft_tt_test_check_fmt("at Bfp8_b, MB=2/NB=2 beats the shipped MB=1/NB=4 by 1.5-2.5%%",
                             square < shipped && square * 1000u > shipped * 970u,
                             "%llu B against %llu B — %.2f%%",
                             (unsigned long long)square, (unsigned long long)shipped,
                             100.0 * (1.0 - (double)square / (double)shipped));

        /* `a/(4b)` with no fitted constant, against Ornith's own measured 0.4718. */
        shape.row_block = 1u;
        shape.column_block = 4u;
        model_run(&shape, &kShipped, result);
        {
            const double ratio = (double)(result->a_tiles_read * A_TILE_BYTES) /
                                 (double)(result->b_tiles_read * 1088u);

            ft_tt_test_check_fmt("the law's A/B at the shipped shape reproduces the measured "
                                 "0.4718 with NO fitted constant",
                                 ratio > 0.469 && ratio < 0.473, "law %.4f, silicon 0.4718",
                                 ratio);
        }
    }
}

/* ------------------------------------------------------------------------------------
 * The DST section: its two sizes, which of them this build is, and what the bigger one
 * actually reaches at the Mt every descriptor carries
 * ---------------------------------------------------------------------------------- */

/**
 * THE SECTION IS FIXED BY THE HARDWARE AND IT HAS EXACTLY TWO VALUES AT `dst_accum_mode 1`.
 * The section is
 * `(FT_TENSIX_DEST_REGISTER_{HALF,FULL}_SIZE >> 1) >> 6` for 32x32 tiles, i.e.
 * `(512 >> 1) >> 6 = 4` at SyncHalf and `(1024 >> 1) >> 6 = 8` at SyncFull. Sixteen would
 * need `dst_accum_mode 0`, which is worth 13-35x accuracy and is not on the table.
 *
 * AND THE WRITE-ADDRESS FORMULA FITS THE BIGGER SECTION EXACTLY, which is the reason twelve
 * DST address sites across seven headers need no edit: they compute
 * `(tile_index << 6) + (dest_offset_id ? HALF : 0)`, SyncFull pins the id at 0, and tile 7
 * lands at row 448 -- the last 64-row tile inside `DEST_REGISTER_FULL_SIZE >> 1 = 512`.
 */
static void check_the_dst_section(void)
{
    static const struct { uint32_t rows; uint32_t columns; int half; int full; } lattice[] = {
        { 1u, 1u, 1, 1 }, { 1u, 4u, 1, 1 }, { 2u, 2u, 1, 1 }, { 4u, 1u, 1, 1 },
        { 1u, 8u, 0, 1 }, { 2u, 4u, 0, 1 }, { 4u, 2u, 0, 1 }, { 8u, 1u, 0, 1 },
        { 3u, 3u, 0, 0 }, { 2u, 8u, 0, 0 }, { 5u, 1u, 0, 1 }, { 1u, 9u, 0, 0 },
        { 0u, 4u, 0, 0 }, { 4u, 0u, 0, 0 },
    };
    const uint32_t half = FT_TT_MATMUL_DST_SECTION_TILES_HALF;
    const uint32_t full = FT_TT_MATMUL_DST_SECTION_TILES_FULL;
    unsigned index;
    int agrees = 1;

    ft_tt_test_step("the DST section: SyncHalf 4, SyncFull 8, and what each admits");
    ft_tt_test_check_fmt("the two sections ARE upstream's get_dest_max_tiles at accum mode",
                         half == ((512u >> 1) >> 6) && full == ((1024u >> 1) >> 6),
                         "half %u, full %u", half, full);
    ft_tt_test_check_fmt("ftmm_dst_section_tiles() answers for a kernel that is not this one",
                         ftmm_dst_section_tiles(0) == half && ftmm_dst_section_tiles(1) == full,
                         "SyncHalf %u, SyncFull %u", ftmm_dst_section_tiles(0),
                         ftmm_dst_section_tiles(1));

    for (index = 0; index < sizeof(lattice) / sizeof(lattice[0]); index++) {
        const uint32_t rows = lattice[index].rows;
        const uint32_t columns = lattice[index].columns;

        ft_tt_test_check_fmt("  the pair is admitted by the right sections",
                             ftmm_block_fits_section(rows, columns, half) ==
                                     lattice[index].half &&
                                 ftmm_block_fits_section(rows, columns, full) ==
                                     lattice[index].full,
                             "MB=%u NB=%u: four %d eight %d, want %d and %d", rows, columns,
                             ftmm_block_fits_section(rows, columns, half),
                             ftmm_block_fits_section(rows, columns, full),
                             lattice[index].half, lattice[index].full);
        if (ftmm_block_fits_dst(rows, columns) !=
            ftmm_block_fits_section(rows, columns, FT_TT_MATMUL_DST_SECTION_TILES))
            agrees = 0;
    }
    /* THE ONE THAT MATTERS FOR THE INTERLOCK: four pairs fit at eight and NOT at four, so a
     * host that believed the wrong section would write past it with nothing refusing. */
    ft_tt_test_check_fmt("four pairs SEPARATE the two sections, so the size is not decorative",
                         ftmm_block_fits_section(1u, 8u, full) &&
                             !ftmm_block_fits_section(1u, 8u, half) &&
                             ftmm_block_fits_section(2u, 4u, full) &&
                             !ftmm_block_fits_section(2u, 4u, half),
                         "(1,8) and (2,4) fit 8 and not 4");
    ft_tt_test_check_fmt("ftmm_block_fits_dst() asks about THIS BUILD's own section",
                         agrees, "FT_TT_MATMUL_DST_SECTION_TILES = %u",
                         FT_TT_MATMUL_DST_SECTION_TILES);
    ft_tt_test_check_fmt("this build is SyncHalf, which is what the shipped blob carries",
                         FT_TT_MATMUL_DST_SYNC == FT_TT_MATMUL_DST_SYNC_HALF &&
                             FT_TT_MATMUL_DST_SECTION_TILES == half,
                         "mode %d, section %u", (int)FT_TT_MATMUL_DST_SYNC,
                         FT_TT_MATMUL_DST_SECTION_TILES);

    /* The write-address formula, which is why no address site is edited -- and why NINE
     * tiles would not merely be refused, it would not fit. */
    ft_tt_test_check_fmt("the LAST SyncFull tile's 64 DST rows end exactly at FULL_SIZE >> 1",
                         ((full - 1u) << 6) + 64u == (1024u >> 1) &&
                             (full << 6) + 64u > (1024u >> 1),
                         "tile %u ends at row %u of %u; tile %u would end at %u", full - 1u,
                         ((full - 1u) << 6) + 64u, 1024u >> 1, full, (full << 6) + 64u);
    ft_tt_test_check_fmt("and the last SyncHalf tile ends exactly at HALF_SIZE >> 1",
                         ((half - 1u) << 6) + 64u == (512u >> 1),
                         "tile %u ends at row %u of %u", half - 1u, ((half - 1u) << 6) + 64u,
                         512u >> 1);
}

/**
 * WHAT THE BIGGER SECTION IS WORTH, AND THE CLAUSE THAT DECIDES IT: `Mt`.
 *
 * The DST-8 lattice's Bfp8_b optimum is MB=2/NB=4 at `512 + 544 = 1056` bytes a unit,
 * 1.52x the shipped 1600. **That point needs `Mt >= 2` and this tree has none** --
 * `pn_tt_matmul_group.c` writes `FT_TT_QUEUE_MATMUL_TILE_ROWS` from a literal 1 on the
 * decode path and on the batched-prefill path alike, because `rows` is capped at 32
 * positions, which is one tile-row. At `Mt = 1` the row loop runs once whatever MB says, so
 * MB is INERT and the reachable DST-8 point is MB=1/NB=8 at `256 + 1088 = 1344` -- 1.19x.
 *
 * The inertness is not asserted, it is MEASURED off the same model that produces the tile
 * counts, with an `Mt = 2` control beside it so the check is able to fail.
 */
static void check_what_the_bigger_section_reaches(model_result *result)
{
    static const struct { uint32_t rows; uint32_t columns; } lattice[] = {
        { 1u, 1u }, { 1u, 2u }, { 1u, 4u }, { 1u, 8u },
        { 2u, 1u }, { 2u, 2u }, { 2u, 4u },
        { 4u, 1u }, { 4u, 2u }, { 8u, 1u },
    };
    model_shape shape = { "dst8", 8u, 32u, 8u, 8u, 1u, 1u, 1, 8u };
    uint64_t best = 0;
    uint32_t best_rows = 0;
    uint32_t best_columns = 0;
    uint64_t best_at_mt1 = 0;
    uint32_t best_mt1_columns = 0;
    uint64_t shipped;
    uint64_t rows1_columns8;
    uint64_t rows2_columns4;
    unsigned index;

    ft_tt_test_step("the DST-8 lattice in BYTES, and which of it `Mt = 1` can reach");
    for (index = 0; index < sizeof(lattice) / sizeof(lattice[0]); index++) {
        uint64_t bytes;

        shape.row_block = lattice[index].rows;
        shape.column_block = lattice[index].columns;
        ft_tt_test_check_fmt("  the pair fits the eight-tile section",
                             ftmm_block_fits_section(shape.row_block, shape.column_block,
                                                     FT_TT_MATMUL_DST_SECTION_TILES_FULL) != 0,
                             "MB=%u NB=%u: %u tiles", shape.row_block, shape.column_block,
                             shape.row_block * shape.column_block);
        model_run(&shape, &kShipped, result);
        bytes = law_bytes(result, 1088u);
        if (best == 0 || bytes < best) {
            best = bytes;
            best_rows = shape.row_block;
            best_columns = shape.column_block;
        }
        if (shape.row_block == 1u && (best_at_mt1 == 0 || bytes < best_at_mt1)) {
            best_at_mt1 = bytes;
            best_mt1_columns = shape.column_block;
        }
    }
    ft_tt_test_check_fmt("the DST-8 Bfp8_b optimum is MB=2/NB=4 — and it needs Mt >= 2",
                         best_rows == 2u && best_columns == 4u, "best is MB=%u NB=%u",
                         best_rows, best_columns);
    ft_tt_test_check_fmt("restricted to MB=1, which Mt=1 forces, the optimum is NB=8",
                         best_mt1_columns == 8u, "best MB=1 column block is NB=%u",
                         best_mt1_columns);

    /* The two multiples, per unit of `T`, from the model's own counts rather than by hand. */
    shape.row_block = 1u;
    shape.column_block = 4u;
    model_run(&shape, &kShipped, result);
    shipped = law_bytes(result, 1088u);
    shape.column_block = 8u;
    model_run(&shape, &kShipped, result);
    rows1_columns8 = law_bytes(result, 1088u);
    shape.row_block = 2u;
    shape.column_block = 4u;
    model_run(&shape, &kShipped, result);
    rows2_columns4 = law_bytes(result, 1088u);

    ft_tt_test_check_fmt("MB=1/NB=8 is 1.18-1.20x the shipped MB=1/NB=4 — the REACHABLE win",
                         rows1_columns8 * 120u > shipped * 100u &&
                             rows1_columns8 * 118u < shipped * 100u,
                         "%llu B against %llu B — %.4fx", (unsigned long long)rows1_columns8,
                         (unsigned long long)shipped,
                         (double)shipped / (double)rows1_columns8);
    ft_tt_test_check_fmt("MB=2/NB=4 is 1.50-1.53x it — the figure the spec carried",
                         rows2_columns4 * 153u > shipped * 100u &&
                             rows2_columns4 * 150u < shipped * 100u,
                         "%llu B against %llu B — %.4fx", (unsigned long long)rows2_columns4,
                         (unsigned long long)shipped,
                         (double)shipped / (double)rows2_columns4);

    /* THE CLAUSE. At Mt = 1 the row block moves NOTHING -- not a tile, not a byte -- so the
     * 1.52x is gated on a host shape change and not on the DST section. The Mt = 2 arm
     * beside it is the control that makes this check able to fail. */
    {
        model_shape mt1 = { "Mt=1 MB=1", 1u, 128u, 64u, 8u, 4u, 1u, 1, 8u };
        uint64_t a_at_mb1, b_at_mb1, a_at_mb2, b_at_mb2;

        model_run(&mt1, &kShipped, result);
        a_at_mb1 = result->a_tiles_read;
        b_at_mb1 = result->b_tiles_read;
        mt1.row_block = 2u;
        model_run(&mt1, &kShipped, result);
        a_at_mb2 = result->a_tiles_read;
        b_at_mb2 = result->b_tiles_read;
        ft_tt_test_check_fmt("at Mt=1 the ROW BLOCK IS INERT — same A tiles, same B tiles",
                             a_at_mb1 == a_at_mb2 && b_at_mb1 == b_at_mb2,
                             "A %llu -> %llu, B %llu -> %llu",
                             (unsigned long long)a_at_mb1, (unsigned long long)a_at_mb2,
                             (unsigned long long)b_at_mb1, (unsigned long long)b_at_mb2);

        mt1.tile_rows = 2u;
        mt1.row_block = 1u;
        model_run(&mt1, &kShipped, result);
        b_at_mb1 = result->b_tiles_read;
        mt1.row_block = 2u;
        model_run(&mt1, &kShipped, result);
        b_at_mb2 = result->b_tiles_read;
        ft_tt_test_check_fmt("  and at Mt=2 it HALVES B — the control that makes that a result",
                             b_at_mb2 * 2u == b_at_mb1, "B %llu -> %llu",
                             (unsigned long long)b_at_mb1, (unsigned long long)b_at_mb2);
    }
}

/* ------------------------------------------------------------------------------------
 * The wire contract: the parameter word, parsed out of the kernel's own header
 * ---------------------------------------------------------------------------------- */

/** Read `enum { name = N };` out of resident_queue_protocol.h, or -1. */
static long kernel_enum_value(const char *name)
{
    const char *directory = getenv("FT_TT_KERNEL_DIR");
    char path[512];
    char line[512];
    FILE *file;
    long value = -1;

    snprintf(path, sizeof(path), "%s/resident_queue_protocol.h",
             directory ? directory : "../kernels");
    file = fopen(path, "r");
    if (!file)
        return -1;
    while (fgets(line, sizeof(line), file)) {
        char found[128];
        long parsed;

        if (sscanf(line, " enum { %127[A-Za-z0-9_] = %ld }", found, &parsed) == 2 &&
            strcmp(found, name) == 0) {
            value = parsed;
            break;
        }
    }
    fclose(file);
    return value;
}

static void check_wire_contract(void)
{
    const long block_word = kernel_enum_value("kMatmulColumnBlock");
    const long words = kernel_enum_value("kMatmulWords");
    const long a2 = kernel_enum_value("kDescA2");
    uint32_t block;

    ft_tt_test_step("the parameter word, PARSED out of the kernel's own header");
    ft_tt_test_check_fmt("kMatmulColumnBlock == FT_TT_QUEUE_MATMUL_COLUMN_BLOCK",
                         block_word == (long)FT_TT_QUEUE_MATMUL_COLUMN_BLOCK,
                         "kernel %ld, host %d", block_word,
                         (int)FT_TT_QUEUE_MATMUL_COLUMN_BLOCK);
    ft_tt_test_check_fmt("the block still fits sixteen words — the row block bought NO word",
                         words == (long)FT_TT_QUEUE_MATMUL_WORDS && words == 16,
                         "kernel %ld, host %d", words, (int)FT_TT_QUEUE_MATMUL_WORDS);
    ft_tt_test_check_fmt("FT_TT_QUEUE_PARAM_WORDS is still 16 for every family",
                         FT_TT_QUEUE_PARAM_WORDS == 16u, "%u", FT_TT_QUEUE_PARAM_WORDS);
    ft_tt_test_check_fmt("kDescA2 exists and is the pitch's home", a2 == 4, "kernel %ld", a2);

    ft_tt_test_step("the packed encoding, and its zero");
    for (block = 1u; block <= 4u; block++)
        ft_tt_test_check_fmt("a pre-encoding word still decodes to (MB=1, NB=itself)",
                             ftmm_column_block(block) == block && ftmm_row_block(block) == 1u,
                             "word %u -> NB %u, MB %u", block, ftmm_column_block(block),
                             ftmm_row_block(block));
    ft_tt_test_check_fmt("a zero word is (MB=1, NB=1)",
                         ftmm_column_block(0u) == 1u && ftmm_row_block(0u) == 1u, "NB %u, MB %u",
                         ftmm_column_block(0u), ftmm_row_block(0u));
    ft_tt_test_check_fmt("ftmm_block_word(1, NB) is BIT-IDENTICAL to the pre-encoding word",
                         ftmm_block_word(1u, 4u) == 4u && ftmm_block_word(1u, 1u) == 1u,
                         "%u and %u", ftmm_block_word(1u, 4u), ftmm_block_word(1u, 1u));
    ft_tt_test_check_fmt("ftmm_block_word(2, 2) round-trips",
                         ftmm_row_block(ftmm_block_word(2u, 2u)) == 2u &&
                             ftmm_column_block(ftmm_block_word(2u, 2u)) == 2u,
                         "word 0x%x", ftmm_block_word(2u, 2u));
}

int main(void)
{
    static const model_shape shapes[] = {
        /* name                        Mt  Kt    Nt   depth NB  MB  n-major DST */
        { "unblocked control",          1u, 32u,  16u, 8u,  1u, 1u, 1, 0u },
        { "qkv-shaped, NB=4",           1u, 128u, 64u, 8u,  4u, 1u, 1, 0u },
        { "ffn-shaped, NB=4",           1u, 384u, 96u, 8u,  4u, 1u, 1, 0u },
        { "NB=2",                       1u, 64u,  12u, 8u,  2u, 1u, 1, 0u },
        { "ragged: Nt=6 under NB=4",    1u, 32u,  6u,  8u,  4u, 1u, 1, 0u },
        { "Nt=1, the ssm_alpha case",   1u, 32u,  1u,  8u,  4u, 1u, 1, 0u },
        { "K-major panel, NB=4",        1u, 64u,  16u, 8u,  4u, 1u, 0, 0u },
        { "depth 32, NB=4",             1u, 64u,  16u, 32u, 4u, 1u, 1, 0u },
        { "depth 1, NB=4",              1u, 16u,  8u,  1u,  4u, 1u, 1, 0u },
        /* The row block. Mt = 2 is 64 prompt positions, Mt = 4 is 128. */
        { "MB=2 NB=2, 64 positions",    2u, 128u, 64u, 8u,  2u, 2u, 1, 0u },
        { "MB=4 NB=1, 128 positions",   4u, 128u, 64u, 8u,  1u, 4u, 1, 0u },
        { "MB=2 NB=1",                  2u, 64u,  16u, 8u,  1u, 2u, 1, 0u },
        { "MB=2 NB=2, Nt=1 weight",     2u, 32u,  1u,  8u,  2u, 2u, 1, 0u },
        { "MB=2 NB=2, ragged Mt=3",     3u, 32u,  8u,  8u,  2u, 2u, 1, 0u },
        { "MB=4 NB=1, ragged Mt=6",     6u, 32u,  8u,  8u,  1u, 4u, 1, 0u },
        { "MB=2 NB=2, ragged Nt=7",     4u, 32u,  7u,  8u,  2u, 2u, 1, 0u },
        { "MB=2 NB=2, K-major panel",   2u, 64u,  16u, 8u,  2u, 2u, 0, 0u },
        { "MB=2 NB=2, depth 1",         2u, 16u,  8u,  1u,  2u, 2u, 1, 0u },
        { "MB=1 NB=4 at Mt=4 (today)",  4u, 128u, 64u, 8u,  4u, 1u, 1, 0u },
        /* DstSync::SyncFull: eight tiles. At Mt = 1 -- which is every descriptor the tree
         * writes -- the whole section belongs to NB, because MB has no second tile-row to
         * divide. The MB=2/NB=4 rows below are what a host that batched more than 32
         * positions would reach; they are checked so the arithmetic is on record, not
         * because anything dispatches them. */
        { "SyncFull: Mt=1 NB=8",        1u, 128u, 64u, 8u,  8u, 1u, 1, 8u },
        { "SyncFull: Mt=1 NB=8, Nt=1",  1u, 32u,  1u,  8u,  8u, 1u, 1, 8u },
        { "SyncFull: Mt=1 NB=8, ragged",1u, 64u,  20u, 8u,  8u, 1u, 1, 8u },
        { "SyncFull: Mt=1 NB=8 K-major",1u, 64u,  16u, 8u,  8u, 1u, 0, 8u },
        { "SyncFull: MB=2 NB=4",        2u, 128u, 64u, 8u,  4u, 2u, 1, 8u },
        { "SyncFull: MB=2 NB=4 ragged", 3u, 64u,  22u, 8u,  4u, 2u, 1, 8u },
        { "SyncFull: MB=4 NB=2",        4u, 64u,  16u, 8u,  2u, 4u, 1, 8u },
    };
    model_result result;
    unsigned index;

    result.pair_capacity = 1u << 22;
    result.pairs = malloc((size_t)result.pair_capacity * sizeof(*result.pairs));
    result.out = malloc((size_t)(256u << 10) * sizeof(*result.out));
    if (!result.pairs || !result.out) {
        fprintf(stderr, "test_matmul_block: out of memory\n");
        return 1;
    }

    check_wire_contract();
    ft_tt_test_step("the blocked reader, compute thread and writer, shape by shape");
    for (index = 0; index < sizeof(shapes) / sizeof(shapes[0]); index++)
        check_shape(&shapes[index], &result);
    check_the_tile_law(&result);
    check_the_byte_law(&result);
    check_the_dst_section();
    check_what_the_bigger_section_reaches(&result);
    check_red_arms(&result);

    free(result.pairs);
    free(result.out);
    return ft_tt_test_finish();
}
