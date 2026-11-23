/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file test_reduce_schedule.c
 * @brief Off-hardware acceptance of the REDUCE family's P3 schedule word: the host encoding
 *        against the kernel's own enums, the chain-chunk refusal, half-read eligibility per
 *        member, the P3 placement, and the streamed reader's issue/retire policy run against
 *        a modelled compute thread for deadlock and overwrite.
 *
 * The device header is #included, so every check is against the SAME inline functions the
 * reader and the compute thread compile — there is no transliteration to drift.
 *
 * @status added with P3 (streamed norm reads, chunked final chain).
 * @issues None
 * @todo None
 */

#include <stdint.h>
#include <string.h>

#include "ft_tt_launch.h"
#include "ft_tt_op_vector.h"
/* The SFPI L1 pointer attribute is the device compiler's; a host reads the header without it. */
#define FTQ_L1_PTR
#include "kernels/resident_reduce_protocol.h"

#include "ft_tt_test_report.h"

static uint8_t g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

static void test_constants_agree(void)
{
    ft_tt_test_step("schedule word: host macros against the kernel enums");
    ft_tt_test_check("word", FT_TT_OP_REDUCE_SCHEDULE == (uint32_t)kRedSchedule, "");
    ft_tt_test_check("chunk mask", FT_TT_OP_REDUCE_SCHED_CHUNK_MASK == (uint32_t)kRedSchedChunkMask,
                     "");
    ft_tt_test_check("stream bit", FT_TT_OP_REDUCE_SCHED_STREAM == (uint32_t)kRedSchedStream, "");
    ft_tt_test_check("half-read bit",
                     FT_TT_OP_REDUCE_SCHED_HALF_READ == (uint32_t)kRedSchedHalfRead, "");
    ft_tt_test_check("RED face-0 bit",
                     FT_TT_OP_REDUCE_SCHED_RED_FACE0 == (uint32_t)kRedSchedRedFace0, "");
    ft_tt_test_check("chunk max is the mask",
                     FT_TT_OP_REDUCE_CHAIN_CHUNK_MAX == (uint32_t)kRedSchedChunkMask, "");
    /* The reader reads the scratch CBs' page counts by these fixed indices. */
    ft_tt_test_check("scratch CB 0", FT_TT_OP_REDUCE_CB_STAGE0 == (uint32_t)kRedCbStage0, "");
    ft_tt_test_check("scratch CB 1", FT_TT_OP_REDUCE_CB_STAGE1 == (uint32_t)kRedCbStage1, "");
    ft_tt_test_check("the schedule rides the unused FUNCTION slot",
                     FT_TT_OP_REDUCE_SCHEDULE == (uint32_t)FT_TT_OP_VECTOR_FUNCTION, "");
}

static ft_tt_op_reduce_request rms_request(void)
{
    ft_tt_op_reduce_request request;

    memset(&request, 0, sizeof(request));
    request.function = FT_TT_OP_REDUCE_FN_RMS_NORM;
    request.row_blocks = 1u;
    request.tile_cols = 160u;
    request.prefetch_tiles = 16u;
    request.input.x = 1u;
    request.input.address = 0x1000u;
    request.input.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request.constants = request.input;
    request.output = request.input;
    return request;
}

static void test_encoding(void)
{
    ft_tt_op_reduce_request request = rms_request();
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    int rc;

    ft_tt_test_step("encoding: word 0 is the schedule, 0 when nothing is asked");
    rc = ft_tt_op_reduce_parameters(&request, words);
    ft_tt_test_check_fmt("pre-P3 request encodes", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("pre-P3 request leaves word 0 zero", words[0] == 0u, "0x%X", words[0]);

    request.chain_chunk = 16u;
    request.stream_input = true;
    request.half_read = true;
    rc = ft_tt_op_reduce_parameters(&request, words);
    ft_tt_test_check_fmt("P3 request encodes", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("chunk 16 + stream + half = 0x310", words[0] == 0x310u, "0x%X",
                         words[0]);
    ft_tt_test_check_fmt("depth word untouched by the schedule", words[3] == 16u, "%u", words[3]);

    request.chain_chunk = FT_TT_OP_REDUCE_CHAIN_CHUNK_MAX + 1u;
    rc = ft_tt_op_reduce_parameters(&request, words);
    ft_tt_test_check_fmt("a chunk past the byte is refused", rc == FT_TT_EINVAL, "rc=%d", rc);
}

static void test_chain_chunk(void)
{
    ft_tt_test_step("chain chunk: the reader's wedge guard");
    ft_tt_test_check_fmt("0 is per tile", ftr_chain_chunk(0u, 1u) == 1u, "%u",
                         ftr_chain_chunk(0u, 1u));
    ft_tt_test_check_fmt("1 is per tile", ftr_chain_chunk(1u, 1u) == 1u, "%u",
                         ftr_chain_chunk(1u, 1u));
    ft_tt_test_check_fmt("16 into 16 pages", ftr_chain_chunk(16u, 16u) == 16u, "%u",
                         ftr_chain_chunk(16u, 16u));
    ft_tt_test_check_fmt("17 into 16 pages is refused", ftr_chain_chunk(17u, 16u) == 0u, "%u",
                         ftr_chain_chunk(17u, 16u));
    ft_tt_test_check_fmt("16 into the pre-P3 one page is refused",
                         ftr_chain_chunk(16u, 1u) == 0u, "%u", ftr_chain_chunk(16u, 1u));
    ft_tt_test_check_fmt("the other schedule bits are not the chunk",
                         ftr_chain_chunk(0x710u, 16u) == 16u, "%u", ftr_chain_chunk(0x710u, 16u));
}

/** The decoded program of `function`, through the host encoder and the kernel decoder. */
static ftr_program decoded(ft_tt_op_reduce_function function)
{
    uint32_t words[16];

    memset(words, 0, sizeof(words));
    (void)ft_tt_op_reduce_program(function, &words[kRedPass0], &words[kRedPass1],
                                  &words[kRedFinal]);
    return ftr_decode(words);
}

static void test_half_read(void)
{
    static const struct {
        ft_tt_op_reduce_function function;
        bool norm_shaped;
    } members[] = {
        { FT_TT_OP_REDUCE_FN_RMS_NORM, true },
        { FT_TT_OP_REDUCE_FN_L2_NORM, true },
        { FT_TT_OP_REDUCE_FN_SOFTMAX, false },
        { FT_TT_OP_REDUCE_FN_LAYER_NORM, false },
        { FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE, false },
    };
    const uint32_t half_depth = 16u | (1u << 23);
    size_t index;

    ft_tt_test_step("half reads: only where faces 2-3 of input and gamma are never unpacked");
    for (index = 0; index < sizeof(members) / sizeof(members[0]); index++) {
        const ftr_program program = decoded(members[index].function);
        const char *name = ft_tt_op_reduce_function_name(members[index].function);

        ft_tt_test_check_fmt(name, ftr_program_is_norm_shaped(&program) ==
                                       members[index].norm_shaped,
                             "norm-shaped %d, expected %d",
                             (int)ftr_program_is_norm_shaped(&program),
                             (int)members[index].norm_shaped);
        ft_tt_test_check_fmt(name, ftr_half_read(kRedSchedHalfRead, half_depth, &program) ==
                                       members[index].norm_shaped,
                             "half read %d", (int)ftr_half_read(kRedSchedHalfRead, half_depth,
                                                                &program));
    }
    {
        const ftr_program rms = decoded(FT_TT_OP_REDUCE_FN_RMS_NORM);

        ft_tt_test_check("no half read without the half-tile bit",
                         !ftr_half_read(kRedSchedHalfRead, 16u, &rms), "");
        ft_tt_test_check("no half read unless asked", !ftr_half_read(0u, half_depth, &rms), "");
    }
    ft_tt_test_check_fmt("bf16 whole", ftr_tile_read_bytes(2048u, false, 0u) == 2048u, "%u",
                         ftr_tile_read_bytes(2048u, false, 0u));
    ft_tt_test_check_fmt("bf16 faces 0-1", ftr_tile_read_bytes(2048u, true, 0u) == 1024u, "%u",
                         ftr_tile_read_bytes(2048u, true, 0u));
    ft_tt_test_check_fmt("fp32 faces 0-1", ftr_tile_read_bytes(4096u, true, 0u) == 2048u, "%u",
                         ftr_tile_read_bytes(4096u, true, 0u));
    ft_tt_test_check_fmt("RED: face 0 only",
                         ftr_tile_read_bytes(2048u, true, kRedSchedRedFace0) == 512u, "%u",
                         ftr_tile_read_bytes(2048u, true, kRedSchedRedFace0));
    ft_tt_test_check_fmt("RED is inert on a whole read",
                         ftr_tile_read_bytes(2048u, false, kRedSchedRedFace0) == 2048u, "%u",
                         ftr_tile_read_bytes(2048u, false, kRedSchedRedFace0));
}

/** The page count `place_cbs` gave CB `cb`, or 0. */
static uint32_t placed_pages(const ft_tt_launch_cb_entry *entries, unsigned cb)
{
    size_t index;

    for (index = 0; index < FT_TT_OP_REDUCE_CB_COUNT; index++) {
        if (entries[index].index == cb)
            return entries[index].page_count;
    }
    return 0u;
}

static void test_placement(void)
{
    static const ft_tt_op_reduce_function rms[] = { FT_TT_OP_REDUCE_FN_RMS_NORM };
    static const ft_tt_op_reduce_function head[] = { FT_TT_OP_REDUCE_FN_RMS_NORM,
                                                     FT_TT_OP_REDUCE_FN_L2_NORM };
    ft_tt_launch_cb_entry entries[FT_TT_OP_REDUCE_CB_COUNT];
    ft_tt_op_reduce_residency residency;
    uint64_t mask = 0u;
    int rc;

    ft_tt_test_step("placement: P3 scratch pages and CB_IN rows, 0 = the pre-P3 extents");
    rc = ft_tt_op_reduce_residency_for(rms, 1u, 160u, FT_TT_OP_TILE_BYTES_BF16, &residency);
    ft_tt_test_check_fmt("Qwen3.8 RMS residency (Wt 160, bf16)", rc == FT_TT_OK, "rc=%d", rc);
    rc = ft_tt_op_reduce_place_cbs(&residency, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("pre-P3 placement fits", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("pre-P3 scratch is one page", placed_pages(entries, 8u) == 1u &&
                                                           placed_pages(entries, 9u) == 1u,
                         "%u/%u", placed_pages(entries, 8u), placed_pages(entries, 9u));
    ft_tt_test_check_fmt("pre-P3 CB_IN is one row", placed_pages(entries, 0u) == 160u, "%u",
                         placed_pages(entries, 0u));
    ft_tt_test_check_fmt("pre-P3 cb_out is double-buffered", placed_pages(entries, 16u) == 2u,
                         "%u", placed_pages(entries, 16u));

    residency.stage_pages = 16u;
    rc = ft_tt_op_reduce_place_cbs(&residency, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("K=16 scratch fits at Wt 160", rc == FT_TT_OK, "rc=%d", rc);
    ft_tt_test_check_fmt("both scratch CBs are 16 pages", placed_pages(entries, 8u) == 16u &&
                                                              placed_pages(entries, 9u) == 16u,
                         "%u/%u", placed_pages(entries, 8u), placed_pages(entries, 9u));
    ft_tt_test_check_fmt("cb_out holds a run", placed_pages(entries, 16u) == 16u, "%u",
                         placed_pages(entries, 16u));
    ft_tt_test_check_fmt("the ceiling still admits Wt 160",
                         ft_tt_op_reduce_max_tile_cols(&residency) >= 160u, "%u",
                         ft_tt_op_reduce_max_tile_cols(&residency));

    residency.stage_pages = FT_TT_OP_REDUCE_CHAIN_CHUNK_MAX + 1u;
    ft_tt_test_check_fmt("a scratch past the chunk byte has no ceiling",
                         ft_tt_op_reduce_max_tile_cols(&residency) == 0u, "%u",
                         ft_tt_op_reduce_max_tile_cols(&residency));

    rc = ft_tt_op_reduce_residency_for(head, 2u, 8u, FT_TT_OP_TILE_BYTES_FP32, &residency);
    residency.input_rows = 4u;
    residency.stage_pages = 8u;
    if (rc == FT_TT_OK)
        rc = ft_tt_op_reduce_place_cbs(&residency, entries, g_cb_buffer, sizeof(g_cb_buffer),
                                       &mask);
    ft_tt_test_check_fmt("head-norm residency (Wt 8, fp32, 4 rows, K 8) fits", rc == FT_TT_OK,
                         "rc=%d", rc);
    ft_tt_test_check_fmt("CB_IN is four rows", placed_pages(entries, 0u) == 32u, "%u",
                         placed_pages(entries, 0u));
    ft_tt_test_check_fmt("CB_MID is still one row", placed_pages(entries, 4u) == 8u, "%u",
                         placed_pages(entries, 4u));
}

/* ================================================================
 * The streamed reader against a modelled compute thread
 * ================================================================ */

/** One run: the reader follows ftr_stream_action(); the consumer pops a row once all of it
 *  is pushed, at moments a seeded generator picks. A blocking reserve that no consumer step
 *  can satisfy is a deadlock; an issue onto unpopped pages is an overwrite. */
typedef struct stream_model {
    uint32_t pages;     /* CB_IN */
    uint32_t tile_cols; /* a row */
    uint32_t depth;     /* a block */
    uint32_t rows;
    uint32_t pushed, acked; /* pages */
    uint32_t seed;
} stream_model;

static uint32_t model_random(stream_model *model)
{
    model->seed = model->seed * 1103515245u + 12345u;
    return model->seed >> 16;
}

/** The compute thread pops a row when the whole row is pushed. */
static bool model_consume(stream_model *model)
{
    if (model->pushed - model->acked < model->tile_cols)
        return false;
    model->acked += model->tile_cols;
    return true;
}

/** @return 0 clean, 1 deadlock, 2 overwrite, 3 out of order / runaway */
static int model_run(stream_model *model, uint32_t *out_max_flight)
{
    const uint32_t total = model->rows * (model->tile_cols / model->depth);
    uint32_t issued = 0u, retired = 0u, steps = 0u;

    *out_max_flight = 0u;
    while (steps++ < 1000000u) {
        const uint32_t free_pages = model->pages - (model->pushed - model->acked);
        const uint32_t action =
            ftr_stream_action(issued, retired, total, free_pages, model->depth);

        if ((model_random(model) & 3u) == 0u)
            (void)model_consume(model);
        if (action == (uint32_t)kRedStreamDone)
            return issued == total && model->pushed == total * model->depth ? 0 : 3;
        if (action == (uint32_t)kRedStreamIssue) {
            /* Block `issued` lands on pages [issued*depth, +depth) of the stream; page p is
             * reusable once p - pages < acked. */
            if ((issued + 1u) * model->depth > model->acked + model->pages)
                return 2;
            issued++;
            if (issued - retired > *out_max_flight)
                *out_max_flight = issued - retired;
        } else if (action == (uint32_t)kRedStreamRetire) {
            model->pushed += model->depth;
            retired++;
        } else {
            while (model->pages - (model->pushed - model->acked) < model->depth) {
                if (!model_consume(model))
                    return 1;
            }
        }
    }
    return 3;
}

static void test_stream_model(void)
{
    static const struct {
        const char *name;
        uint32_t pages, tile_cols, depth, rows, flight;
    } shapes[] = {
        { "Qwen3.8 RMS: Wt 160, depth 16, one row", 160u, 160u, 16u, 1u, 4u },
        { "head norm: Wt 8, CB 4 rows, 12 heads", 32u, 8u, 8u, 12u, 4u },
        { "GDN head norm: Wt 4, CB 4 rows, 13 heads", 16u, 4u, 4u, 13u, 4u },
        { "CB of one row, several rows", 16u, 16u, 4u, 5u, 4u },
        { "depth 1", 8u, 8u, 1u, 3u, 4u },
        { "block = row = CB", 8u, 8u, 8u, 6u, 1u },
    };
    size_t index;

    ft_tt_test_step("streamed reader: no deadlock, no overwrite, bounded flight");
    for (index = 0; index < sizeof(shapes) / sizeof(shapes[0]); index++) {
        unsigned bad = 0u, worst_flight = 0u;
        uint32_t seed;

        for (seed = 1u; seed <= 200u; seed++) {
            stream_model model;
            uint32_t flight = 0u;

            memset(&model, 0, sizeof(model));
            model.pages = shapes[index].pages;
            model.tile_cols = shapes[index].tile_cols;
            model.depth = shapes[index].depth;
            model.rows = shapes[index].rows;
            model.seed = seed;
            if (model_run(&model, &flight) != 0)
                bad++;
            if (flight > worst_flight)
                worst_flight = flight;
        }
        ft_tt_test_check_fmt(shapes[index].name, bad == 0u && worst_flight == shapes[index].flight,
                             "%u of 200 interleavings failed, max %u in flight (want %u)", bad,
                             worst_flight, shapes[index].flight);
    }
}

int main(void)
{
    ft_tt_test_step("REDUCE P3 schedule — off-hardware acceptance");
    test_constants_agree();
    test_encoding();
    test_chain_chunk();
    test_half_read();
    test_placement();
    test_stream_model();
    return ft_tt_test_finish();
}
