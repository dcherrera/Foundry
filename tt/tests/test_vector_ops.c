/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_vector_ops.c
 * @brief OFF-HARDWARE gate for the vector op family — the elementwise binary and the
 *   REDUCE binary (softmax, RMS norm, layer norm, router normalise). Runs
 *   without a card, and is what has to be green before the hardware pass is worth a core.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS CAN PROVE, AND WHAT IT CANNOT
 * ---------------------------------------------------------------------------
 * It CAN prove:
 *   - the host and the KERNEL agree on every wire constant, by parsing the device headers
 *     rather than by a comment claiming they match (dev notes §4: "the confident sentence
 *     in a doc is the one to check, because nobody re-derives it");
 *   - the parameter block puts every field in the word the kernel reads;
 *   - the circular buffers fit the arena, do not overlap, and the documented Wt ceiling is
 *     the real one rather than a rounded-down guess;
 *   - the ARRANGEMENT OF PASSES computes the op, and clears a PCC ≥ 0.99 gate with bf16 at
 *     every circular-buffer boundary — which is a prediction of the card's score;
 *   - the max-subtraction pass is load-bearing, by running a row whose range makes exp
 *     overflow without it.
 *
 * It CANNOT prove the kernel implements the model. Nothing off-hardware can. dev notes §3:
 * "off-hardware suites cannot cover the copy path" — and here they cannot cover the SFPU,
 * the packer, the face order or the CB credit protocol either. A green run here means the
 * plan is sound, never that the op works.
 *
 * ---------------------------------------------------------------------------
 * THE DATA
 * ---------------------------------------------------------------------------
 * Full-mantissa random, never small integers. A bf16-exact array is bit-identical to its
 * own bf16 rendering, so it cannot tell two CB descriptor sets apart and made two
 * deliberately-different builds byte-equal once (dev notes §3).
 *
 * Environment:
 *   FT_TT_KERNEL_DIR  where the device headers live (default ../kernels)
 *
 * @status Green: 198 checks, 0 failures, on the Mac (clang). Seven mutations were run
 *   against the original 87 and six went red — 1, 6, 1, 1, 1 and 2 failures. The seventh
 *   (the row max reduced over ONE tile instead of Wt) came back GREEN and that is what
 *   added test_softmax_max_is_over_the_whole_row(); with it, the same mutation fails.
 *   The 2026-08-15 ragged-W stages carry five more red proofs, each on a `make red` clean
 *   rebuild: the pad value forced to 0 (4 failures), the scaler built at the PADDED width
 *   (2 — and PCC stayed 0.999995, so max-abs is what caught it), ONE pad lane out of 31
 *   written wrong (5), the layer-norm refusal removed (2), and the layer-norm stage's own
 *   subject removed by making W exact so there are no pad lanes at all (7).
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../convert/ft_float_convert.h"
#include "../ft_tt_op_dispatch.h"
#include "../ft_tt_op_l1_layout.h"
#include "../ft_tt_op_matmul.h"
#include "../ft_tt_op_vector.h"
#include "../ft_tt_queue.h"
#include "../ft_tt_tilize.h"
#include "ft_tt_test_report.h"
#include "ft_tt_vector_reference.h"

#define PCC_ACCEPT 0.99

/** Where the device headers live, so the wire-constant pin reads the real file. */
#define KERNEL_DIRECTORY_ENVIRONMENT "FT_TT_KERNEL_DIR"
#define KERNEL_DIRECTORY_DEFAULT "../kernels"

/** Longest path this assembles. */
#define PATH_MAX_BYTES 512u

static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

/* ================================================================
 * Reading a constant out of a device header
 * ================================================================ */

/**
 * The value of `enum { <name> = <value> };` in `path`.
 *
 * The device headers are C and C++ at once (plan Phase 6.6), so the wire constants are
 * declared one per line as an anonymous `enum` rather than as `constexpr uint32_t` --
 * a C translation unit cannot say `constexpr`. The FORM is what changed; the intent is
 * unchanged and is the whole point of this function: the value is read out of the real
 * device header rather than retyped here, because a duplicated contract with nothing
 * enforcing agreement is a contract that will disagree. A missing constant is a FAILURE
 * and never a default -- a default would make this check pass against a header that no
 * longer declares the field.
 *
 * Parsed out of the real header rather than retyped here, which is the same discipline
 * build_tensix_program.sh applies to the kernel-text relocation map: a duplicated contract
 * with nothing enforcing agreement is a contract that will disagree. A missing constant is
 * a FAILURE and never a default — a default would make this check pass against a header
 * that no longer declares the field.
 *
 * @return 0 on success, -1 if the file or the constant is missing
 */
static int read_device_constant(const char *path, const char *name, uint32_t *out_value)
{
    char line[512];
    FILE *file = fopen(path, "r");
    int found = -1;

    if (!file)
        return -1;
    while (fgets(line, sizeof(line), file)) {
        char parsed_name[128];
        char parsed_value[64];

        if (sscanf(line, " enum { %127[A-Za-z0-9_] = %63[^}] };", parsed_name,
                   parsed_value) != 2)
            continue;
        if (strcmp(parsed_name, name) != 0)
            continue;
        *out_value = (uint32_t)strtoul(parsed_value, NULL, 0);
        found = 0;
        break;
    }
    fclose(file);
    return found;
}

/** `directory`/`file`, for the two device headers this pins against. */
static void device_header_path(char *out, size_t out_bytes, const char *file)
{
    const char *directory = getenv(KERNEL_DIRECTORY_ENVIRONMENT);

    if (!directory || directory[0] == '\0')
        directory = KERNEL_DIRECTORY_DEFAULT;
    snprintf(out, out_bytes, "%s/%s", directory, file);
}

/** One host constant against its device twin, by name. */
static void check_device_constant(const char *path, const char *name, uint32_t host_value)
{
    uint32_t device_value = 0;

    if (read_device_constant(path, name, &device_value) != 0) {
        ft_tt_test_check_fmt(name, false, "not declared in %s", path);
        return;
    }
    ft_tt_test_check_fmt(name, device_value == host_value, "host %u, kernel %u", host_value,
                         device_value);
}

/**
 * Every word index, every function value and both opcodes, host against kernel.
 *
 * This is the check that would have caught a field moving under an op author, and it is
 * cheap enough that there is no reason for any future op family not to have one.
 */
static void test_wire_constants_agree(void)
{
    char vector_path[PATH_MAX_BYTES];
    char queue_path[PATH_MAX_BYTES];
    char reduce_path[PATH_MAX_BYTES];

    ft_tt_test_step("wire constants: the host enum against the kernel's own header");
    device_header_path(vector_path, sizeof(vector_path), "resident_vector_protocol.h");
    device_header_path(queue_path, sizeof(queue_path), "resident_queue_protocol.h");
    device_header_path(reduce_path, sizeof(reduce_path), "resident_reduce_protocol.h");

    check_device_constant(vector_path, "kVecFunction", FT_TT_OP_VECTOR_FUNCTION);
    check_device_constant(vector_path, "kVecTileCount", FT_TT_OP_VECTOR_TILE_COUNT);
    check_device_constant(vector_path, "kVecTileCols", FT_TT_OP_VECTOR_TILE_COLS);
    check_device_constant(vector_path, "kVecDepth", FT_TT_OP_VECTOR_DEPTH);
    check_device_constant(vector_path, "kVecAX", FT_TT_OP_VECTOR_A_X);
    check_device_constant(vector_path, "kVecAY", FT_TT_OP_VECTOR_A_Y);
    check_device_constant(vector_path, "kVecABase", FT_TT_OP_VECTOR_A_BASE);
    check_device_constant(vector_path, "kVecBX", FT_TT_OP_VECTOR_B_X);
    check_device_constant(vector_path, "kVecBY", FT_TT_OP_VECTOR_B_Y);
    check_device_constant(vector_path, "kVecBBase", FT_TT_OP_VECTOR_B_BASE);
    check_device_constant(vector_path, "kVecCX", FT_TT_OP_VECTOR_C_X);
    check_device_constant(vector_path, "kVecCY", FT_TT_OP_VECTOR_C_Y);
    check_device_constant(vector_path, "kVecCBase", FT_TT_OP_VECTOR_C_BASE);
    check_device_constant(vector_path, "kVecWords", FT_TT_OP_VECTOR_WORDS);

    check_device_constant(vector_path, "kVecFnSilu", FT_TT_OP_VECTOR_FN_SILU);
    check_device_constant(vector_path, "kVecFnSigmoid", FT_TT_OP_VECTOR_FN_SIGMOID);
    check_device_constant(vector_path, "kVecFnAdd", FT_TT_OP_VECTOR_FN_ADD);
    check_device_constant(vector_path, "kVecFnMul", FT_TT_OP_VECTOR_FN_MUL);
    check_device_constant(vector_path, "kVecFnSub", FT_TT_OP_VECTOR_FN_SUB);
    check_device_constant(vector_path, "kVecFnSiluMul", FT_TT_OP_VECTOR_FN_SILU_MUL);

    check_device_constant(queue_path, "kOpEltwise", FT_TT_QUEUE_OP_ELTWISE);
    check_device_constant(queue_path, "kOpSoftmax", FT_TT_QUEUE_OP_SOFTMAX);
    check_device_constant(queue_path, "kOpReduce", FT_TT_QUEUE_OP_REDUCE);

    check_device_constant(reduce_path, "kRedPass0", FT_TT_OP_REDUCE_PASS0);
    check_device_constant(reduce_path, "kRedPass1", FT_TT_OP_REDUCE_PASS1);
    check_device_constant(reduce_path, "kRedFinal", FT_TT_OP_REDUCE_FINAL);
    check_device_constant(reduce_path, "kRedBytePool", FT_TT_OP_REDUCE_BYTE_POOL);
    check_device_constant(reduce_path, "kRedByteMap", FT_TT_OP_REDUCE_BYTE_MAP);
    check_device_constant(reduce_path, "kRedByteVerb", FT_TT_OP_REDUCE_BYTE_VERB);
    check_device_constant(reduce_path, "kRedByteAddEps", FT_TT_OP_REDUCE_BYTE_ADD_EPS);
    check_device_constant(reduce_path, "kRedBytePassCount", FT_TT_OP_REDUCE_BYTE_PASS_COUNT);
    check_device_constant(reduce_path, "kRedByteSource", FT_TT_OP_REDUCE_BYTE_SOURCE);
    check_device_constant(reduce_path, "kRedByteChain", FT_TT_OP_REDUCE_BYTE_CHAIN);
    check_device_constant(reduce_path, "kRedPoolSum", FT_TT_OP_REDUCE_POOL_SUM);
    check_device_constant(reduce_path, "kRedPoolMax", FT_TT_OP_REDUCE_POOL_MAX);
    check_device_constant(reduce_path, "kRedMapNone", FT_TT_OP_REDUCE_MAP_NONE);
    check_device_constant(reduce_path, "kRedMapSquareSelf", FT_TT_OP_REDUCE_MAP_SQUARE_SELF);
    check_device_constant(reduce_path, "kRedMapSubExp", FT_TT_OP_REDUCE_MAP_SUB_EXP);
    check_device_constant(reduce_path, "kRedMapSubSquare", FT_TT_OP_REDUCE_MAP_SUB_SQUARE);
    check_device_constant(reduce_path, "kRedVerbNone", FT_TT_OP_REDUCE_VERB_NONE);
    check_device_constant(reduce_path, "kRedVerbRecip", FT_TT_OP_REDUCE_VERB_RECIP);
    check_device_constant(reduce_path, "kRedVerbRsqrt", FT_TT_OP_REDUCE_VERB_RSQRT);
    check_device_constant(reduce_path, "kRedSourceIn", FT_TT_OP_REDUCE_SOURCE_IN);
    check_device_constant(reduce_path, "kRedSourceMid", FT_TT_OP_REDUCE_SOURCE_MID);
    check_device_constant(reduce_path, "kRedChainCenter", FT_TT_OP_REDUCE_CHAIN_CENTER);
    check_device_constant(reduce_path, "kRedChainGamma", FT_TT_OP_REDUCE_CHAIN_GAMMA);
    check_device_constant(reduce_path, "kRedChainScale", FT_TT_OP_REDUCE_CHAIN_SCALE);
    check_device_constant(reduce_path, "kRedChainBeta", FT_TT_OP_REDUCE_CHAIN_BETA);
    check_device_constant(reduce_path, "kRedConstScalerTile", FT_TT_OP_REDUCE_CONST_SCALER_TILE);
    check_device_constant(reduce_path, "kRedConstEpsTile", FT_TT_OP_REDUCE_CONST_EPS_TILE);
    check_device_constant(reduce_path, "kRedConstStreamTile", FT_TT_OP_REDUCE_CONST_STREAM_TILE);

    /* The parameter block and the queue's slot machinery are the same sixteen words; if
     * they ever diverge, a block written by the host is read past its end by the kernel. */
    ft_tt_test_check_fmt("the vector block fits a queue parameter slot",
                         (uint32_t)FT_TT_OP_VECTOR_WORDS == FT_TT_QUEUE_PARAM_WORDS,
                         "vector %u, queue slot %u", (uint32_t)FT_TT_OP_VECTOR_WORDS,
                         FT_TT_QUEUE_PARAM_WORDS);
}

/* ================================================================
 * Arity and block depth
 * ================================================================ */

/**
 * The KERNEL's block-depth rule, transliterated from ftv_block_depth() in
 * resident_vector_protocol.h.
 *
 * The two rules are deliberately different: the host SHRINKS a request to the largest
 * divisor, the kernel ACCEPTS a divisor or falls back to 1. That is not a disagreement, it
 * is a host that computes and a kernel that refuses to guess — but it is only safe if the
 * host's answer always survives the kernel's check, which is what the sweep below asserts.
 */
static uint32_t kernel_block_depth(uint32_t run, uint32_t requested)
{
    return (requested > 0u && run > 0u && run % requested == 0u) ? requested : 1u;
}

static void test_block_depth(void)
{
    static const uint32_t runs[] = { 1u, 2u, 3u, 8u, 12u, 48u, 96u, 144u, 1000u };
    unsigned run_index;
    unsigned mismatches = 0;
    unsigned non_divisors = 0;

    ft_tt_test_step("block depth: the host's answer must survive the kernel's check");

    for (run_index = 0; run_index < sizeof(runs) / sizeof(runs[0]); run_index++) {
        const uint32_t run = runs[run_index];
        uint32_t requested;

        for (requested = 0; requested <= run + 4u; requested++) {
            const uint32_t host = ft_tt_op_block_depth(run, requested);

            if (host == 0u || run % host != 0u)
                non_divisors++;
            if (kernel_block_depth(run, host) != host)
                mismatches++;
        }
    }
    ft_tt_test_check_fmt("every host depth divides its run", non_divisors == 0, "%u violations",
                         non_divisors);
    ft_tt_test_check_fmt("the kernel accepts every host depth", mismatches == 0, "%u rejected",
                         mismatches);

    ft_tt_test_check_fmt("depth 0 becomes 1", ft_tt_op_block_depth(48u, 0u) == 1u, "%u",
                         ft_tt_op_block_depth(48u, 0u));
    ft_tt_test_check_fmt("depth past the run is clamped to the run",
                         ft_tt_op_block_depth(48u, 4096u) == 48u, "%u",
                         ft_tt_op_block_depth(48u, 4096u));
    ft_tt_test_check_fmt("a non-divisor shrinks rather than clamps",
                         ft_tt_op_block_depth(48u, 32u) == 24u, "%u",
                         ft_tt_op_block_depth(48u, 32u));
    ft_tt_test_check_fmt("a prime run collapses to 1", ft_tt_op_block_depth(97u, 48u) == 1u, "%u",
                         ft_tt_op_block_depth(97u, 48u));
    /* The matmul's named entry point is now this function; keeping the two answers equal is
     * what makes collapsing them a refactor rather than a behaviour change. */
    ft_tt_test_check_fmt("ft_tt_op_matmul_prefetch_depth still agrees",
                         ft_tt_op_matmul_prefetch_depth(144u, 64u) ==
                             ft_tt_op_block_depth(144u, 64u),
                         "%u vs %u", ft_tt_op_matmul_prefetch_depth(144u, 64u),
                         ft_tt_op_block_depth(144u, 64u));
}

static void test_arity(void)
{
    ft_tt_test_step("arity: derived from the function, sent nowhere");
    ft_tt_test_check("SILU is unary",
                     !ft_tt_op_vector_function_is_binary(FT_TT_OP_VECTOR_FN_SILU), "");
    ft_tt_test_check("SIGMOID is unary",
                     !ft_tt_op_vector_function_is_binary(FT_TT_OP_VECTOR_FN_SIGMOID), "");
    ft_tt_test_check("ADD is binary",
                     ft_tt_op_vector_function_is_binary(FT_TT_OP_VECTOR_FN_ADD), "");
    ft_tt_test_check("MUL is binary",
                     ft_tt_op_vector_function_is_binary(FT_TT_OP_VECTOR_FN_MUL), "");
    ft_tt_test_check("SUB is binary",
                     ft_tt_op_vector_function_is_binary(FT_TT_OP_VECTOR_FN_SUB), "");
    ft_tt_test_check("SILU_MUL is binary",
                     ft_tt_op_vector_function_is_binary(FT_TT_OP_VECTOR_FN_SILU_MUL), "");
}

/* ================================================================
 * Parameter blocks
 * ================================================================ */

static ft_tt_op_operand make_operand(uint16_t x, uint16_t y, uint32_t address)
{
    ft_tt_op_operand operand;

    operand.x = x;
    operand.y = y;
    operand.address = address;
    operand.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    return operand;
}

static void test_eltwise_parameters(void)
{
    ft_tt_op_eltwise_request request;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    int rc;

    ft_tt_test_step("elementwise parameter block");

    memset(&request, 0, sizeof(request));
    request.function = FT_TT_OP_VECTOR_FN_MUL;
    request.tile_count = 48u;
    request.prefetch_tiles = 32u; /* not a divisor of 48 — must shrink to 24, not clamp */
    request.input_a = make_operand(3u, 4u, 0x100000u);
    request.input_b = make_operand(5u, 6u, 0x200000u);
    request.output = make_operand(7u, 8u, 0x300000u);

    rc = ft_tt_op_eltwise_parameters(&request, words);
    ft_tt_test_check_fmt("a well-formed request encodes", rc == FT_TT_OK, "rc %d", rc);
    ft_tt_test_check_fmt("function lands in word 0",
                         words[FT_TT_OP_VECTOR_FUNCTION] == FT_TT_OP_VECTOR_FN_MUL, "%u",
                         words[FT_TT_OP_VECTOR_FUNCTION]);
    ft_tt_test_check_fmt("tile count lands in word 1", words[FT_TT_OP_VECTOR_TILE_COUNT] == 48u,
                         "%u", words[FT_TT_OP_VECTOR_TILE_COUNT]);
    ft_tt_test_check_fmt("tile_cols is 0 for elementwise", words[FT_TT_OP_VECTOR_TILE_COLS] == 0u,
                         "%u", words[FT_TT_OP_VECTOR_TILE_COLS]);
    ft_tt_test_check_fmt("depth is shrunk to a divisor", words[FT_TT_OP_VECTOR_DEPTH] == 24u, "%u",
                         words[FT_TT_OP_VECTOR_DEPTH]);
    request.row0 = 7u; /* any nonzero */
    rc = ft_tt_op_eltwise_parameters(&request, words);
    ft_tt_test_check_fmt("row0 sets exactly the word-2 row-0 flag",
                         rc == FT_TT_OK && words[FT_TT_OP_VECTOR_TILE_COLS] ==
                                               FT_TT_OP_ELTWISE_FLAG_ROW0,
                         "rc %d word2 %u", rc, words[FT_TT_OP_VECTOR_TILE_COLS]);
    request.row0 = 0u;
    rc = ft_tt_op_eltwise_parameters(&request, words);
    ft_tt_test_check_fmt("A operand round-trips",
                         words[FT_TT_OP_VECTOR_A_X] == 3u && words[FT_TT_OP_VECTOR_A_Y] == 4u &&
                             words[FT_TT_OP_VECTOR_A_BASE] == 0x100000u,
                         "%u,%u,0x%X", words[FT_TT_OP_VECTOR_A_X], words[FT_TT_OP_VECTOR_A_Y],
                         words[FT_TT_OP_VECTOR_A_BASE]);
    ft_tt_test_check_fmt("B operand round-trips",
                         words[FT_TT_OP_VECTOR_B_X] == 5u && words[FT_TT_OP_VECTOR_B_Y] == 6u &&
                             words[FT_TT_OP_VECTOR_B_BASE] == 0x200000u,
                         "%u,%u,0x%X", words[FT_TT_OP_VECTOR_B_X], words[FT_TT_OP_VECTOR_B_Y],
                         words[FT_TT_OP_VECTOR_B_BASE]);
    ft_tt_test_check_fmt("C operand round-trips",
                         words[FT_TT_OP_VECTOR_C_X] == 7u && words[FT_TT_OP_VECTOR_C_Y] == 8u &&
                             words[FT_TT_OP_VECTOR_C_BASE] == 0x300000u,
                         "%u,%u,0x%X", words[FT_TT_OP_VECTOR_C_X], words[FT_TT_OP_VECTOR_C_Y],
                         words[FT_TT_OP_VECTOR_C_BASE]);
    ft_tt_test_check_fmt("the reserved tail is zeroed",
                         words[13] == 0u && words[14] == 0u && words[15] == 0u, "%u,%u,%u",
                         words[13], words[14], words[15]);

    /* A unary request must encode without a B operand at all — that is the whole point of
     * the kernel deriving arity. */
    request.function = FT_TT_OP_VECTOR_FN_SILU;
    memset(&request.input_b, 0, sizeof(request.input_b));
    rc = ft_tt_op_eltwise_parameters(&request, words);
    ft_tt_test_check_fmt("a unary request needs no B operand", rc == FT_TT_OK, "rc %d", rc);

    /* …and a BINARY one must be refused without it, because the reader would otherwise
     * fetch tiles from address 0 and produce a plausible wrong answer. */
    request.function = FT_TT_OP_VECTOR_FN_ADD;
    rc = ft_tt_op_eltwise_parameters(&request, words);
    ft_tt_test_check_fmt("a binary request without B is refused", rc == FT_TT_EINVAL, "rc %d", rc);

    request.function = (ft_tt_op_vector_function)99;
    request.input_b = make_operand(5u, 6u, 0x200000u);
    rc = ft_tt_op_eltwise_parameters(&request, words);
    ft_tt_test_check_fmt("an unknown function is refused", rc == FT_TT_EINVAL, "rc %d", rc);

    request.function = FT_TT_OP_VECTOR_FN_ADD;
    request.tile_count = 0u;
    rc = ft_tt_op_eltwise_parameters(&request, words);
    ft_tt_test_check_fmt("a zero tile count is refused", rc == FT_TT_EINVAL, "rc %d", rc);
}

/**
 * The four members' PROGRAM WORDS, pinned as LITERALS.
 *
 * Deliberately not derived. A pin computed from the same table the encoder reads re-derives
 * any mistake in that table and passes — which is exactly how eleven self-consistent asserts
 * on ft_tt_mailbox.h missed a 16-byte insertion (dev notes §3). These are the hex words that
 * have driven silicon; changing one is a rebuild of the reduce kernel and a re-run of the
 * on-card acceptance, not a header edit.
 *
 * Read a word as four bytes, low to high:
 *   a pass  = pool, map, verb, add_eps
 *   final   = pass count, chain source, chain flags, 0
 */
static void test_reduce_programs(void)
{
    static const struct {
        ft_tt_op_reduce_function function;
        uint32_t pass0;
        uint32_t pass1;
        uint32_t final_word;
    } expected[] = {
        /* softmax:    MAX/none/none/no   then SUM/sub-exp/recip/no ; scale the MID stream */
        { FT_TT_OP_REDUCE_FN_SOFTMAX, 0x00000002u, 0x00010201u, 0x00040102u },
        /* rms_norm:   SUM/x*x/rsqrt/eps                            ; gamma then scale on IN */
        { FT_TT_OP_REDUCE_FN_RMS_NORM, 0x01020101u, 0x00000000u, 0x00060001u },
        /* layer_norm: SUM/none/none/no   then SUM/sub-square/rsqrt/eps ; centre+gamma+scale+beta */
        { FT_TT_OP_REDUCE_FN_LAYER_NORM, 0x00000001u, 0x01020301u, 0x000F0002u },
        /* router:     SUM/none/recip/no                            ; scale on IN */
        { FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE, 0x00010001u, 0x00000000u, 0x00040001u },
        /* l2_norm:    SUM/x*x/rsqrt/eps                            ; scale on IN, no gamma.
         * RMS norm's PASS word beside the router's FINAL word — which is the whole of the
         * claim that a fifth member is a table row: not one new bit reaches the wire. */
        { FT_TT_OP_REDUCE_FN_L2_NORM, 0x01020101u, 0x00000000u, 0x00040001u }
    };
    unsigned index;

    ft_tt_test_step("the reduce family's five programs, pinned as literals");

    for (index = 0; index < sizeof(expected) / sizeof(expected[0]); index++) {
        uint32_t pass0 = 0u;
        uint32_t pass1 = 0u;
        uint32_t final_word = 0u;
        const int rc = ft_tt_op_reduce_program(expected[index].function, &pass0, &pass1,
                                               &final_word);
        char label[96];

        snprintf(label, sizeof(label), "%s compiles to its pinned program",
                 ft_tt_op_reduce_function_name(expected[index].function));
        ft_tt_test_check_fmt(label,
                             rc == FT_TT_OK && pass0 == expected[index].pass0 &&
                                 pass1 == expected[index].pass1 &&
                                 final_word == expected[index].final_word,
                             "rc %d, got 0x%08X 0x%08X 0x%08X, want 0x%08X 0x%08X 0x%08X", rc,
                             pass0, pass1, final_word, expected[index].pass0,
                             expected[index].pass1, expected[index].final_word);
    }

    {
        uint32_t pass0 = 0u;
        uint32_t pass1 = 0u;
        uint32_t final_word = 0u;
        const int rc = ft_tt_op_reduce_program((ft_tt_op_reduce_function)99, &pass0, &pass1,
                                               &final_word);

        ft_tt_test_check_fmt("an unknown member is refused rather than encoded",
                             rc == FT_TT_EINVAL, "rc %d", rc);
    }

    /* The derived facts, checked against the definitions rather than restated: gamma and
     * beta are what the CHAIN says, eps is what a PASS says, and the pass count is what the
     * final word's low byte says. Two spellings of one fact is the shape of bug this file
     * exists to catch. */
    ft_tt_test_check("only the two norms use gamma",
                     !ft_tt_op_reduce_uses_gamma(FT_TT_OP_REDUCE_FN_SOFTMAX) &&
                         ft_tt_op_reduce_uses_gamma(FT_TT_OP_REDUCE_FN_RMS_NORM) &&
                         ft_tt_op_reduce_uses_gamma(FT_TT_OP_REDUCE_FN_LAYER_NORM) &&
                         !ft_tt_op_reduce_uses_gamma(FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE), "");
    ft_tt_test_check("only layer norm uses beta",
                     ft_tt_op_reduce_uses_beta(FT_TT_OP_REDUCE_FN_LAYER_NORM) &&
                         !ft_tt_op_reduce_uses_beta(FT_TT_OP_REDUCE_FN_RMS_NORM), "");
    ft_tt_test_check("only the two norms use eps",
                     ft_tt_op_reduce_uses_eps(FT_TT_OP_REDUCE_FN_RMS_NORM) &&
                         ft_tt_op_reduce_uses_eps(FT_TT_OP_REDUCE_FN_LAYER_NORM) &&
                         !ft_tt_op_reduce_uses_eps(FT_TT_OP_REDUCE_FN_SOFTMAX) &&
                         !ft_tt_op_reduce_uses_eps(FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE), "");
    /* The scaler is 1/W exactly when a pass wants a MEAN. Softmax's and the router's reduces
     * are a max and a plain sum, so theirs is 1.0 — one tile, two meanings, one slot. */
    ft_tt_test_check("only the two norms take a 1/W scaler",
                     ft_tt_op_reduce_scaler_is_mean(FT_TT_OP_REDUCE_FN_RMS_NORM) &&
                         ft_tt_op_reduce_scaler_is_mean(FT_TT_OP_REDUCE_FN_LAYER_NORM) &&
                         !ft_tt_op_reduce_scaler_is_mean(FT_TT_OP_REDUCE_FN_SOFTMAX) &&
                         !ft_tt_op_reduce_scaler_is_mean(FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE),
                     "");
    ft_tt_test_check("softmax and layer norm are two passes, the others one",
                     ft_tt_op_reduce_pass_count(FT_TT_OP_REDUCE_FN_SOFTMAX) == 2u &&
                         ft_tt_op_reduce_pass_count(FT_TT_OP_REDUCE_FN_LAYER_NORM) == 2u &&
                         ft_tt_op_reduce_pass_count(FT_TT_OP_REDUCE_FN_RMS_NORM) == 1u &&
                         ft_tt_op_reduce_pass_count(FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE) == 1u,
                     "");
    /* L2 norm, stated where its neighbours are, because the two cells it differs from RMS
     * norm in are exactly the two a copy-paste of that row would keep. */
    ft_tt_test_check("L2 norm takes no gamma, no beta, one pass, eps, and a 1.0 scaler",
                     !ft_tt_op_reduce_uses_gamma(FT_TT_OP_REDUCE_FN_L2_NORM) &&
                         !ft_tt_op_reduce_uses_beta(FT_TT_OP_REDUCE_FN_L2_NORM) &&
                         ft_tt_op_reduce_uses_eps(FT_TT_OP_REDUCE_FN_L2_NORM) &&
                         !ft_tt_op_reduce_scaler_is_mean(FT_TT_OP_REDUCE_FN_L2_NORM) &&
                         ft_tt_op_reduce_pass_count(FT_TT_OP_REDUCE_FN_L2_NORM) == 1u,
                     "");
}

/** `pn_qwen35moe_l2_norm()` in Pantheon, spelled here so the two cannot drift: a double
 *  sum of squares, and a CLAMP at eps rather than an epsilon under the root. */
static void oracle_l2_norm_rows(const float *x, float eps, float *out, uint32_t rows,
                                uint32_t cols)
{
    uint32_t row;

    for (row = 0; row < rows; row++) {
        const float *xr = x + (size_t)row * cols;
        float *outr = out + (size_t)row * cols;
        double sumsq = 0.0;
        float inv;
        uint32_t col;

        for (col = 0; col < cols; col++)
            sumsq += (double)xr[col] * (double)xr[col];
        inv = 1.0f / fmaxf((float)sqrt(sumsq), eps);
        for (col = 0; col < cols; col++)
            outr[col] = xr[col] * inv;
    }
}

/** Max-norm relative error of one row, folded into a running worst. The score
 *  `PN_TT_GDN_NORMCHECK` prints on the card, spelled here so the two are comparable. */
static double row_max_norm_relative(const float *want, const float *got, uint32_t cols,
                                    double worst)
{
    double max_abs = 0.0;
    double max_want = 0.0;
    uint32_t col;

    for (col = 0; col < cols; col++) {
        const double diff = fabs((double)got[col] - (double)want[col]);
        const double magnitude = fabs((double)want[col]);

        if (diff > max_abs)
            max_abs = diff;
        if (magnitude > max_want)
            max_want = magnitude;
    }
    if (max_want > 0.0 && max_abs / max_want > worst)
        worst = max_abs / max_want;
    return worst;
}

/**
 * THE L2 MEMBER AGAINST THE ORACLE IT EXISTS TO REPRODUCE — and against the RMS norm it is
 * one table row away from.
 *
 * The shape is the GDN's, not the model's: 16 rows of 128, which is `[k_heads, S]` at
 * Ornith's shape and the ONE reduce descriptor the q/k head norm compiles to. The red arm
 * runs the same data through RMS_NORM's program and requires the score to COLLAPSE: an L2
 * norm and an RMS norm over the same row differ by exactly sqrt(W), so a test that cannot
 * separate them is measuring nothing.
 */
static void test_l2_norm_against_the_oracle(void)
{
    const uint32_t rows = 16u;
    const uint32_t cols = 128u;
    const size_t count = (size_t)rows * cols;
    const float eps = 1e-6f;
    float *x = malloc(count * sizeof(float));
    float *want = malloc(count * sizeof(float));
    float *got = malloc(count * sizeof(float));
    uint32_t pass0 = 0u;
    uint32_t pass1 = 0u;
    uint32_t final_word = 0u;
    size_t element;

    ft_tt_test_step("the L2 member reproduces pn_qwen35moe_l2_norm at the GDN's head shape");
    if (!x || !want || !got) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }
    ft_tt_vector_random_seed(0x1220E7ULL);
    for (element = 0; element < count; element++)
        x[element] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value()));
    oracle_l2_norm_rows(x, eps, want, rows, cols);

    /* eps*eps, because the oracle clamps the MAGNITUDE at eps and this folds into the
     * SQUARE. eps_tile is that divided by 32 — the value build_constants writes. */
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_L2_NORM, &pass0, &pass1, &final_word);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x, NULL, NULL, 1.0f,
                                      ft_float_from_bf16(ft_bf16_from_float(
                                          eps * eps / (float)FT_TT_OP_TILE_DIM)),
                                      got, rows, cols);
    {
        const double pcc = ft_tt_vector_pearson(want, got, count);
        const double max_abs = ft_tt_vector_max_abs_difference(want, got, count);

        ft_tt_test_check_fmt("PCC against the fp64 oracle is at least 0.9999",
                             pcc >= 0.9999, "PCC %.9f, max abs %.6g", pcc, max_abs);
        /* bf16 through the circular buffers is ~2^-8 a value, and the modelled norm scales
         * a whole row by one number, so the bar is the FORMAT and not the arithmetic. */
        ft_tt_test_check_fmt("max abs is inside the bf16 budget", max_abs < 0.01,
                             "max abs %.6g", max_abs);
    }

    /* RED ARM: RMS norm's program on the same data. Same pass, same verb — only the scaler
     * and the gamma step differ, and that is a factor of sqrt(128) = 11.3 on every element. */
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_RMS_NORM, &pass0, &pass1, &final_word);
    {
        float *gamma = malloc(cols * sizeof(float));
        double max_abs;
        uint32_t index;

        if (!gamma) {
            ft_tt_test_check("red-arm allocation", false, "out of memory");
            goto done;
        }
        for (index = 0; index < cols; index++)
            gamma[index] = 1.0f;
        ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x, gamma, NULL,
                                          1.0f / (float)cols,
                                          ft_float_from_bf16(ft_bf16_from_float(
                                              eps * (float)cols / (float)FT_TT_OP_TILE_DIM)),
                                          got, rows, cols);
        max_abs = ft_tt_vector_max_abs_difference(want, got, count);
        ft_tt_test_check_fmt("RED: the RMS member fails the same bar by sqrt(W)",
                             max_abs > 0.1, "max abs %.6g (want > 0.1)", max_abs);
        free(gamma);
    }

    /* THE OPERAND'S FORMAT, PRICED SEPARATELY FROM THE REDUCE'S — because on silicon the two
     * are quoted as one number and the lever is not the same.
     *
     * Everything above is bf16-EXACT input: `x` was rounded before the oracle saw it, so the
     * only error measured is the reduce's own. A real operand is not bf16-exact — the GDN's
     * `conv_out` and `y` are fp32 — and a bf16 activation strip rounds it on the way in. This
     * arm gives the ORACLE the fp32 values and the CARD MODEL the bf16 rendering of them,
     * which is exactly the comparison `PN_TT_GDN_NORMCHECK` makes on the card. Scored the
     * same way the probe scores: max-norm relative, PER ROW. */
    {
        float *fp32 = malloc(count * sizeof(float));
        double worst_reduce = 0.0;
        double worst_storage = 0.0;
        uint32_t row;

        if (!fp32) {
            ft_tt_test_check("format-floor allocation", false, "out of memory");
            goto done;
        }
        ft_tt_vector_random_seed(0x1220E8ULL);
        for (element = 0; element < count; element++)
            fp32[element] = ft_tt_vector_random_value();
        oracle_l2_norm_rows(fp32, eps, want, rows, cols);
        for (element = 0; element < count; element++)
            x[element] = ft_float_from_bf16(ft_bf16_from_float(fp32[element]));

        (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_L2_NORM, &pass0, &pass1, &final_word);
        ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x, NULL, NULL, 1.0f,
                                          ft_float_from_bf16(ft_bf16_from_float(
                                              eps * eps / (float)FT_TT_OP_TILE_DIM)),
                                          got, rows, cols);
        for (row = 0; row < rows; row++)
            worst_storage = row_max_norm_relative(want + (size_t)row * cols,
                                                  got + (size_t)row * cols, cols,
                                                  worst_storage);
        /* And the same reduce over the bf16-exact operand, which is the control: whatever is
         * left there is the reduce, and the difference between the two is the FORMAT. */
        oracle_l2_norm_rows(x, eps, want, rows, cols);
        for (row = 0; row < rows; row++)
            worst_reduce = row_max_norm_relative(want + (size_t)row * cols,
                                                 got + (size_t)row * cols, cols, worst_reduce);
        /* AND THE SPLIT IS NOT WHERE IT LOOKED. The operand's own bf16 rendering is the
         * SMALLER term: the REDUCE's format is what puts a narrow norm past the gate, so an
         * fp32 activation strip in front of a bf16 reduce would buy almost nothing.
         * Measured on silicon 2026-08-19 by PN_TT_GDN_NORMCHECK on Ornith's real conv_out:
         * q/k L2 worst 8.599e-03, gated worst 1.283e-02 — this model lands within 20% of
         * the first, which is what licenses using it to price the alternative. */
        ft_tt_test_check_fmt("the bf16 REDUCE alone is past the 1e-3 gate", worst_reduce > 1e-3,
                             "%.3e per-row max-norm relative", worst_reduce);
        ft_tt_test_check_fmt("and the operand's own bf16 storage adds LESS than the reduce",
                             worst_storage - worst_reduce < worst_reduce,
                             "storage %.3e, reduce %.3e — the CB's format is the lever, not "
                             "the strip's", worst_storage, worst_reduce);
        free(fp32);
    }

done:
    free(x);
    free(want);
    free(got);
}

/**
 * THE Float32 `reduce` PROGRAM AGAINST THE bf16 ONE, ON THE GDN HEAD SHAPE, OFF-HARDWARE.
 *
 * The question this answers before any card time: a Float32 CB table costs a doubled page
 * and a second program in the blob, so what does it BUY, and is what it buys enough?
 *
 * Both models run the SAME program words over the SAME data against the SAME fp64 oracle.
 * The only difference is the CB table, which is two roundings (ft_tt_vector_reference.c):
 * bf16 rounds on the way into L1 and is exact on the way out; Float32 is exact on the way
 * in and TRUNCATES TO Tf32 — 11 mantissa bits, toward zero — on the way out. That second
 * one is a floor no table can move, so the honest question is not "does Float32 help" but
 * "does Tf32 clear 1e-3 at this shape".
 *
 * Scored per row, max-norm relative, which is exactly what `PN_TT_GDN_NORMCHECK` prints per
 * head on the card. 16 rows of 128 is `[k_heads, S]` at Ornith's shape.
 */
static void test_float32_reduce_at_the_head_shape(void)
{
    const uint32_t rows = 16u;
    const uint32_t cols = 128u;
    const size_t count = (size_t)rows * cols;
    const float eps = 1e-6f;
    float *fp32 = malloc(count * sizeof(float));
    float *stored_bf16 = malloc(count * sizeof(float));
    float *gamma = malloc(cols * sizeof(float));
    float *want = malloc(count * sizeof(float));
    float *got = malloc(count * sizeof(float));
    double l2_bf16 = 0.0;
    double l2_float32 = 0.0;
    double rms_bf16 = 0.0;
    double rms_float32 = 0.0;
    uint32_t pass0 = 0u;
    uint32_t pass1 = 0u;
    uint32_t final_word = 0u;
    uint32_t row;
    size_t element;

    ft_tt_test_step("the Float32 reduce program against the bf16 one, at the GDN head shape");
    if (!fp32 || !stored_bf16 || !gamma || !want || !got) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }
    ft_tt_vector_random_seed(0x1220E8ULL);
    for (element = 0; element < count; element++) {
        fp32[element] = ft_tt_vector_random_value();
        stored_bf16[element] = ft_float_from_bf16(ft_bf16_from_float(fp32[element]));
    }
    for (element = 0; element < cols; element++)
        gamma[element] = 0.5f + ft_tt_vector_random_value() * 0.25f;

    /* ---- the q/k arm: L2 norm, no gamma, scaler 1.0, eps as eps*eps/32 ---- */
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_L2_NORM, &pass0, &pass1, &final_word);
    oracle_l2_norm_rows(fp32, eps, want, rows, cols);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, stored_bf16, NULL, NULL, 1.0f,
                                      ft_float_from_bf16(ft_bf16_from_float(
                                          eps * eps / (float)FT_TT_OP_TILE_DIM)),
                                      got, rows, cols);
    for (row = 0; row < rows; row++)
        l2_bf16 = row_max_norm_relative(want + (size_t)row * cols, got + (size_t)row * cols,
                                        cols, l2_bf16);
    /* The Float32 strip stores the fp32 values as given — that is the whole point of it — so
     * the operand this model reduces is `fp32` and not the bf16 rendering of it. */
    ft_tt_vector_model_reduce_program_float32(pass0, pass1, final_word, fp32, NULL, NULL, 1.0f,
                                              eps * eps / (float)FT_TT_OP_TILE_DIM, got, rows,
                                              cols);
    for (row = 0; row < rows; row++)
        l2_float32 = row_max_norm_relative(want + (size_t)row * cols, got + (size_t)row * cols,
                                           cols, l2_float32);

    /* ---- the gated arm's reduce half: RMS norm with a real gamma, scaler 1/W ---- */
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_RMS_NORM, &pass0, &pass1, &final_word);
    ft_tt_vector_reference_rms_norm(fp32, gamma, eps, want, rows, cols);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, stored_bf16, gamma, NULL,
                                      1.0f / (float)cols,
                                      ft_float_from_bf16(ft_bf16_from_float(
                                          eps * (float)cols / (float)FT_TT_OP_TILE_DIM)),
                                      got, rows, cols);
    for (row = 0; row < rows; row++)
        rms_bf16 = row_max_norm_relative(want + (size_t)row * cols, got + (size_t)row * cols,
                                         cols, rms_bf16);
    ft_tt_vector_model_reduce_program_float32(pass0, pass1, final_word, fp32, gamma, NULL,
                                              1.0f / (float)cols,
                                              eps * (float)cols / (float)FT_TT_OP_TILE_DIM,
                                              got, rows, cols);
    for (row = 0; row < rows; row++)
        rms_float32 = row_max_norm_relative(want + (size_t)row * cols, got + (size_t)row * cols,
                                            cols, rms_float32);

    ft_tt_test_check_fmt("the bf16 program misses the 1e-3 gate on BOTH members",
                         l2_bf16 > 1e-3 && rms_bf16 > 1e-3,
                         "L2 %.3e, RMS %.3e per-row max-norm relative", l2_bf16, rms_bf16);
    ft_tt_test_check_fmt("the Float32 program is at least 4x better on both",
                         l2_float32 * 4.0 < l2_bf16 && rms_float32 * 4.0 < rms_bf16,
                         "L2 %.3e -> %.3e (%.1fx), RMS %.3e -> %.3e (%.1fx)", l2_bf16,
                         l2_float32, l2_bf16 / l2_float32, rms_bf16, rms_float32,
                         rms_bf16 / rms_float32);
    /* AND THE TWO MEMBERS LAND ON OPPOSITE SIDES OF THE GATE, which is the result. The L2
     * member's chain is ONE step; the RMS member's is TWO (gamma, then scale) and every step
     * is a CB round trip, so it pays two more Tf32 unpacks on the value path. */
    ft_tt_test_check_fmt("the L2 member CLEARS 1e-3 at Float32", l2_float32 < 1e-3,
                         "%.3e against 1e-3", l2_float32);
    ft_tt_test_check_fmt("the RMS member does NOT, and it is inside the Tf32 budget of 2e-3",
                         rms_float32 > 1e-3 && rms_float32 < 2e-3,
                         "%.3e — over the gate and under twice it", rms_float32);

    /* THE ATTRIBUTION, and it is the whole finding: run the SAME program with an fp32 unpack
     * the hardware does not have. If the residue collapses, every bit of what is left at
     * Float32 is Tf32 TRUNCATION and no CB table can touch it — the next lever is the chain's
     * step count, i.e. kernel source, which this round does not open. */
    {
        double l2_exact = 0.0;
        double rms_exact = 0.0;

        ft_tt_vector_model_reduce_program_exact(pass0, pass1, final_word, fp32, gamma, NULL,
                                                1.0f / (float)cols,
                                                eps * (float)cols / (float)FT_TT_OP_TILE_DIM,
                                                got, rows, cols);
        for (row = 0; row < rows; row++)
            rms_exact = row_max_norm_relative(want + (size_t)row * cols,
                                              got + (size_t)row * cols, cols, rms_exact);
        (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_L2_NORM, &pass0, &pass1, &final_word);
        oracle_l2_norm_rows(fp32, eps, want, rows, cols);
        ft_tt_vector_model_reduce_program_exact(pass0, pass1, final_word, fp32, NULL, NULL, 1.0f,
                                                eps * eps / (float)FT_TT_OP_TILE_DIM, got, rows,
                                                cols);
        for (row = 0; row < rows; row++)
            l2_exact = row_max_norm_relative(want + (size_t)row * cols,
                                             got + (size_t)row * cols, cols, l2_exact);
        ft_tt_test_check_fmt("an fp32 unpack would clear the gate by 100x on BOTH — so the "
                             "whole Float32 residue is Tf32 TRUNCATION, not the pass structure",
                             l2_exact * 100.0 < l2_float32 && rms_exact * 100.0 < rms_float32,
                             "counterfactual L2 %.3e RMS %.3e against Tf32's %.3e and %.3e",
                             l2_exact, rms_exact, l2_float32, rms_float32);
    }

done:
    free(fp32);
    free(stored_bf16);
    free(gamma);
    free(want);
    free(got);
}

/* ---------------------------------------------------------------------------
 * THE THREE WALLS UNDER THE Float32 HEAD NORMS, PRICED APART
 *
 * On silicon the two GDN members score 1.737e-03 (L2) and 3.165e-03 (RMS) against a 1e-3
 * gate, and the model above says 8.406e-04 and 1.566e-03 — the card is 2.07x and 2.02x
 * ABOVE it, the SAME factor on both members. Three things were named for that gap and none
 * had been measured apart:
 *
 *   1. the SFPU's rsqrt polynomial, which the model does not have (it uses libm);
 *   2. the RMS member's chain STEP COUNT, two steps against the L2 member's one;
 *   3. the sample size — the card takes the worst of 48 layers x 32 heads, the model of 16
 *      rows in one draw.
 *
 * Each is an arm here. The rsqrt one is the important one to run FIRST, because if it is
 * real then the chain-step work cannot reach the gate on its own and is not worth grinding.
 * --------------------------------------------------------------------------- */
static void test_the_three_walls_priced_apart(void)
{
    const uint32_t rows = 1536u; /* 48 SSM layers x 32 heads — what the card's worst is over */
    const uint32_t cols = 128u;
    const size_t count = (size_t)rows * cols;
    const float eps = 1e-6f;
    float *x = malloc(count * sizeof(float));
    float *gamma = malloc(cols * sizeof(float));
    /* The "gamma folded into a downstream weight" arm's oracle is the same RMS norm with
     * gamma = 1, which is a gamma vector rather than a second reference function. */
    float *unit_gamma = malloc(cols * sizeof(float));
    float *want = malloc(cols * sizeof(float));
    float *got = malloc(cols * sizeof(float));
    float *control = malloc((size_t)16u * cols * sizeof(float));
    uint32_t l2_pass0 = 0u;
    uint32_t l2_pass1 = 0u;
    uint32_t l2_final = 0u;
    uint32_t rms_pass0 = 0u;
    uint32_t rms_pass1 = 0u;
    uint32_t rms_final = 0u;
    size_t element;

    /* Every arm is scored the same way: worst per-row max-norm relative over `rows`, which is
     * exactly `normcheck_score()` per head on the card. */
    struct arm_result {
        double l2;
        double rms;
        double rms_no_gamma;
    };

    ft_tt_test_step("the three walls under the Float32 head norms, priced apart");
    if (!x || !gamma || !unit_gamma || !want || !got || !control) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }
    ft_tt_vector_random_seed(0x1220E8ULL);
    for (element = 0; element < count; element++)
        x[element] = ft_tt_vector_random_value();
    for (element = 0; element < cols; element++) {
        gamma[element] = 0.5f + ft_tt_vector_random_value() * 0.25f;
        unit_gamma[element] = 1.0f;
    }

    /* THE STREAM, PINNED AS LITERALS. The card-side probe that settles the model-vs-silicon
     * gap (`PN_TT_RED=head_norm_synthetic`, Pantheon `qwen35moe_gdn_normcheck_tt.c`) cannot
     * link this file, so it spells the same xorshift64* itself and checks it against exactly
     * these eight words. Two spellings of one stream is a thing that must agree; this is what
     * makes a disagreement a refusal instead of an incomparable number. */
    {
        static const uint32_t pinned[8] = { 0xBF508D2Au, 0x3EF4CC04u, 0x3EE46400u, 0xBDF7F450u,
                                            0xBE494EE8u, 0xBF4FF826u, 0x3E88AB9Cu,
                                            0xBF4CD48Cu };
        bool matches = true;
        size_t index;

        for (index = 0; index < 8u; index++) {
            uint32_t bits;

            memcpy(&bits, &x[index], sizeof(bits));
            if (bits != pinned[index])
                matches = false;
        }
        ft_tt_test_check("the synthetic stream at seed 0x1220E8 is pinned as literals", matches,
                         "the card-side probe checks the same eight words");
    }
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_L2_NORM, &l2_pass0, &l2_pass1, &l2_final);
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_RMS_NORM, &rms_pass0, &rms_pass1,
                                  &rms_final);

/* One arm, both members plus the gamma-free RMS the "fold gamma downstream" item would run. */
#define ARM(name, arms, row_count)                                                             \
    do {                                                                                       \
        uint32_t row;                                                                          \
        (name).l2 = 0.0;                                                                       \
        (name).rms = 0.0;                                                                      \
        (name).rms_no_gamma = 0.0;                                                             \
        for (row = 0; row < (row_count); row++) {                                              \
            const float *x_row = x + (size_t)row * cols;                                       \
            oracle_l2_norm_rows(x_row, eps, want, 1u, cols);                                   \
            ft_tt_vector_model_head_norm_float32(                                              \
                l2_pass0, l2_pass1, l2_final, x_row, NULL, 1.0f,                               \
                eps *eps / (float)FT_TT_OP_TILE_DIM, got, 1u, cols, (arms));                   \
            (name).l2 = row_max_norm_relative(want, got, cols, (name).l2);                     \
            ft_tt_vector_reference_rms_norm(x_row, gamma, eps, want, 1u, cols);                \
            ft_tt_vector_model_head_norm_float32(                                              \
                rms_pass0, rms_pass1, rms_final, x_row, gamma, 1.0f / (float)cols,             \
                eps * (float)cols / (float)FT_TT_OP_TILE_DIM, got, 1u, cols, (arms));          \
            (name).rms = row_max_norm_relative(want, got, cols, (name).rms);                   \
            ft_tt_vector_model_head_norm_float32(                                              \
                rms_pass0, rms_pass1, rms_final, x_row, NULL, 1.0f / (float)cols,              \
                eps * (float)cols / (float)FT_TT_OP_TILE_DIM, got, 1u, cols,                   \
                (arms) | FT_TT_REDUCE_ARM_NO_GAMMA);                                           \
            ft_tt_vector_reference_rms_norm(x_row, unit_gamma, eps, want, 1u, cols);           \
            (name).rms_no_gamma =                                                              \
                row_max_norm_relative(want, got, cols, (name).rms_no_gamma);                   \
        }                                                                                      \
    } while (0)

    /* THE CONTROL THAT LICENSES EVERY ARM: at arms = 0 the switchable model must be the
     * shipped Float32 model BIT FOR BIT. Without this the arms below measure a second
     * program that merely resembles the first. */
    {
        bool identical = true;
        bool armed_differs = false;
        uint32_t row;

        ft_tt_vector_model_reduce_program_float32(rms_pass0, rms_pass1, rms_final, x, gamma,
                                                  NULL, 1.0f / (float)cols,
                                                  eps * (float)cols / (float)FT_TT_OP_TILE_DIM,
                                                  control, 16u, cols);
        for (row = 0; row < 16u; row++) {
            const size_t offset = (size_t)row * cols;

            ft_tt_vector_model_head_norm_float32(
                rms_pass0, rms_pass1, rms_final, x + offset, gamma, 1.0f / (float)cols,
                eps * (float)cols / (float)FT_TT_OP_TILE_DIM, got, 1u, cols, 0u);
            if (memcmp(got, control + offset, (size_t)cols * sizeof(float)) != 0)
                identical = false;
            /* The same comparison with ONE arm on must DIFFER, or the check above is a
             * memcmp of two buffers that were never going to disagree. */
            ft_tt_vector_model_head_norm_float32(
                rms_pass0, rms_pass1, rms_final, x + offset, gamma, 1.0f / (float)cols,
                eps * (float)cols / (float)FT_TT_OP_TILE_DIM, got, 1u, cols,
                FT_TT_REDUCE_ARM_ROUND_NEAREST);
            if (memcmp(got, control + offset, (size_t)cols * sizeof(float)) != 0)
                armed_differs = true;
        }
        ft_tt_test_check("the switchable model IS the shipped Float32 model at arms = 0",
                         identical, "bit-for-bit over 16 rows of the RMS member");
        ft_tt_test_check("RED: one arm on and the same comparison DIFFERS", armed_differs,
                         "FT_TT_REDUCE_ARM_ROUND_NEAREST against the same 16 rows");
    }

    /* A program this arm does not model must be REFUSED, not approximated — softmax through
     * it is all-NaN, which is the red proof that the shape guard is not decoration. */
    {
        uint32_t soft0 = 0u;
        uint32_t soft1 = 0u;
        uint32_t soft_final = 0u;
        bool all_nan = true;

        (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_SOFTMAX, &soft0, &soft1, &soft_final);
        ft_tt_vector_model_head_norm_float32(soft0, soft1, soft_final, x, NULL, 1.0f, 0.0f, got,
                                             1u, cols, 0u);
        for (element = 0; element < cols; element++) {
            if (!isnan(got[element]))
                all_nan = false;
        }
        ft_tt_test_check_fmt(
            "RED: softmax through the head-norm arm is refused, not approximated", all_nan,
            "all %u outputs NaN", cols);
    }

    {
        struct arm_result shipped;
        struct arm_result sfpu;
        struct arm_result perturbed;
        struct arm_result premultiply;
        struct arm_result nearest;
        struct arm_result nearest_host;
        struct arm_result small;

        ARM(shipped, 0u, rows);
        ARM(small, 0u, 16u);
        ARM(sfpu, FT_TT_REDUCE_ARM_SFPU_RSQRT, rows);
        ARM(perturbed, FT_TT_REDUCE_ARM_RSQRT_PERTURB, rows);
        ARM(premultiply, FT_TT_REDUCE_ARM_PREMULTIPLY, rows);
        ARM(nearest, FT_TT_REDUCE_ARM_ROUND_NEAREST, rows);
        ARM(nearest_host,
            FT_TT_REDUCE_ARM_ROUND_NEAREST | FT_TT_REDUCE_ARM_HOST_ROUND, rows);

        /* WALL 1: the SFPU rsqrt. Its own accuracy first, so the null below is a number and
         * not only an absence — swept against fp64 across the binades a sum of 128 squares
         * plus eps can land in. */
        {
            double worst = 0.0;
            int exponent;

            for (exponent = -20; exponent <= 20; exponent++) {
                uint32_t step;

                for (step = 0; step < 4096u; step++) {
                    const float value =
                        ldexpf(1.0f + (float)step / 4096.0f, exponent);
                    const double reference = 1.0 / sqrt((double)value);
                    const double relative =
                        fabs((double)ft_tt_vector_sfpu_rsqrt(value) - reference) / reference;

                    if (relative > worst)
                        worst = relative;
                }
            }
            ft_tt_test_check_fmt("the SFPU rsqrt polynomial is 4 orders BELOW the residue it "
                                 "was nominated for",
                                 worst < 1e-6, "%.3e worst relative over 2^-20..2^21, against "
                                               "a residue of 8.4e-04",
                                 worst);
        }
        /* And in the program, where it is a NULL — the perturbed arm below is what says the
         * score could have seen it. */
        ft_tt_test_check_fmt(
            "the SFPU rsqrt polynomial moves NEITHER member — the 2x candidate is refuted",
            fabs(sfpu.l2 - shipped.l2) < shipped.l2 * 0.01 &&
                fabs(sfpu.rms - shipped.rms) < shipped.rms * 0.01,
            "libm L2 %.3e RMS %.3e -> SFPU %.3e / %.3e", shipped.l2, shipped.rms, sfpu.l2,
            sfpu.rms);
        /* THE RED ARM, and its DIRECTION is a second result. A +1e-3 rsqrt moves the L2
         * member UP by 1.7x and the RMS member DOWN by 1.5x — because the RMS residue is a
         * downward-biased scale error that an upward perturbation partly cancels. That is the
         * same signature the GDN value path carries (`card/want` 0.9955-0.9977, below 1 on
         * 768 of 768 head-layers), measured here on the reduce instead of asserted. So the
         * check is that a 1e-3 rsqrt MOVES the score, not that it raises it. */
        ft_tt_test_check_fmt(
            "RED: an rsqrt off by 1e-3 relative moves BOTH members by >30% — so the null above "
            "means something",
            fabs(perturbed.l2 - shipped.l2) > shipped.l2 * 0.3 &&
                fabs(perturbed.rms - shipped.rms) > shipped.rms * 0.3,
            "L2 %.3e -> %.3e (up), RMS %.3e -> %.3e (DOWN — the residue is biased low)",
            shipped.l2, perturbed.l2, shipped.rms, perturbed.rms);

        /* WALL 2: the chain step count. Reordering it is a null — two FPU multiplies through
         * circular buffers cost four Tf32 unpacks whichever operand pair goes first — and the
         * whole of the RMS member's excess is gamma plus the scratch round trip. */
        ft_tt_test_check_fmt(
            "reordering the chain is a NULL: the unpack count, not the order, is what costs",
            fabs(premultiply.rms - shipped.rms) < shipped.rms * 0.15,
            "as shipped %.3e, premultiplied %.3e", shipped.rms, premultiply.rms);
        ft_tt_test_check_fmt(
            "and DROPPING gamma is the whole of the RMS member's excess over L2",
            shipped.rms_no_gamma < shipped.rms * 0.5 &&
                shipped.rms_no_gamma < shipped.l2 * 1.2,
            "RMS %.3e -> %.3e with a one-step chain, against the L2 member's %.3e",
            shipped.rms, shipped.rms_no_gamma, shipped.l2);
        /* AND THAT IS THE RESULT THAT PRICES THE WHOLE ITEM: the L2 member's chain is ALREADY
         * one step and it is 1.737e-03 on the card. A perfect chain-step fix lands the RMS
         * member on the L2 member's number, which is itself 1.7x past the gate. */
        ft_tt_test_check_fmt(
            "so a PERFECT chain-step fix lands RMS on L2's figure — which still misses 1e-3 "
            "on silicon",
            shipped.rms_no_gamma > 1e-3 * 0.5,
            "modelled one-step RMS %.3e; the card's one-step member measures 1.737e-03",
            shipped.rms_no_gamma);

        /* WALL 3: the sample size. It is real and it is small — and it is on the RMS member
         * only, because the L2 member's residue is a common-mode scale error that every row
         * carries equally. */
        ft_tt_test_check_fmt("the card's 48x32 sample costs the RMS member a little and the "
                             "L2 member nothing",
                             small.l2 == shipped.l2 && shipped.rms > small.rms,
                             "16 rows: L2 %.3e RMS %.3e; 1536 rows: L2 %.3e RMS %.3e", small.l2,
                             small.rms, shipped.l2, shipped.rms);

        /* The remaining kernel-source lever, priced: an SFPU pre-round to the NEAREST Tf32
         * before every pack. Worth ~1.3x on RMS, and it makes the L2 member WORSE, because
         * the truncation biases of the map and the chain partly CANCEL. */
        ft_tt_test_check_fmt(
            "rounding to nearest before the pack is worth ~1.3x on RMS and is NOT free on L2",
            nearest.rms < shipped.rms && nearest.l2 > shipped.l2 * 0.9,
            "RMS %.3e -> %.3e, L2 %.3e -> %.3e (host pre-round too: %.3e / %.3e)", shipped.rms,
            nearest.rms, shipped.l2, nearest.l2, nearest_host.l2, nearest_host.rms);

        /* WHAT IS LEFT OF THE 2x, AND IT IS ONE MANTISSA BIT. "E8M10" is written down two
         * ways — ten stored plus a hidden one, or ten of significand of which nine are
         * stored — and the difference is exactly a factor of two on a truncating unpack. The
         * nine-bit reading reproduces the card's q/k L2 figure to four significant figures on
         * uniform data, where the ten-bit reading is 2.07x under it. It is a HYPOTHESIS with
         * one strong prediction behind it; `PN_TT_RED=head_norm_synthetic` settles it on
         * silicon by removing the other difference, the data. */
        {
            struct arm_result nine;
            struct arm_result nine_best;

            ARM(nine, FT_TT_REDUCE_ARM_TF32_NINE_BITS, rows);
            ft_tt_test_check_fmt("a NINE-bit Tf32 unpack reproduces the card's L2 member, and "
                                 "the ten-bit one is 2.07x under it",
                                 nine.l2 > 1.6e-3 && nine.l2 < 1.9e-3,
                                 "ten bits %.3e, nine bits %.3e, silicon 1.737e-03 (RMS %.3e "
                                 "/ %.3e against silicon's 3.165e-03)",
                                 shipped.l2, nine.l2, shipped.rms, nine.rms);
            /* AND THE CONSEQUENCE, which is why the bit matters at all: at nine bits every
             * lever this family has, composed, still misses the gate. */
            {
                struct arm_result nine_kernel_rne;

                ARM(nine_best,
                    FT_TT_REDUCE_ARM_TF32_NINE_BITS | FT_TT_REDUCE_ARM_ROUND_NEAREST |
                        FT_TT_REDUCE_ARM_HOST_ROUND,
                    rows);
                ARM(nine_kernel_rne,
                    FT_TT_REDUCE_ARM_TF32_NINE_BITS | FT_TT_REDUCE_ARM_ROUND_NEAREST, rows);
                ft_tt_test_check_fmt(
                    "at nine bits EVERY lever composed still misses 1e-3 — the gate is not "
                    "reachable in this family at Wt = 4",
                    nine_best.l2 > 1e-3 && nine_best.rms_no_gamma > 1e-3 &&
                        nine_kernel_rne.l2 > 1e-3 && nine_kernel_rne.rms_no_gamma > 1e-3,
                    "one-step chain, round-to-nearest on the kernel packs: L2 %.3e RMS %.3e; "
                    "and with the host operands too: L2 %.3e RMS %.3e",
                    nine_kernel_rne.l2, nine_kernel_rne.rms_no_gamma, nine_best.l2,
                    nine_best.rms_no_gamma);
            }
        }
    }

    /* The per-stage attribution, REPORTED rather than gated: which unpack spends the budget.
     * Removing a stage can make the score WORSE, which is the evidence that the truncation
     * biases partly cancel and that "fewer stages" is not automatically "less error". */
    {
        static const struct {
            const char *name;
            uint32_t arm;
        } stages[] = {
            { "the MAP's x unpack", FT_TT_REDUCE_ARM_EXACT_MAP },
            { "the reduce's mid unpack", FT_TT_REDUCE_ARM_EXACT_MID },
            { "the chain's x unpack", FT_TT_REDUCE_ARM_EXACT_CHAIN_X },
            { "the rsqrt column's unpack", FT_TT_REDUCE_ARM_EXACT_COLUMN },
            { "gamma's unpack", FT_TT_REDUCE_ARM_EXACT_GAMMA },
            { "the chain's scratch round trip", FT_TT_REDUCE_ARM_EXACT_STAGE },
        };
        size_t index;

        for (index = 0; index < sizeof(stages) / sizeof(stages[0]); index++) {
            struct arm_result without;
            char label[160];

            ARM(without, stages[index].arm, rows);
            snprintf(label, sizeof(label), "REPORTED: with %s made exact", stages[index].name);
            ft_tt_test_check_fmt(label, true, "L2 %.3e, RMS %.3e", without.l2, without.rms);
        }
    }
#undef ARM

done:
    free(x);
    free(gamma);
    free(unit_gamma);
    free(want);
    free(got);
    free(control);
}

/* ---------------------------------------------------------------------------
 * `gated` IS BOUND BY THE bf16 BRIDGE, NOT BY THE REDUCE
 *
 * The GDN's output gated norm is TWO descriptors on TWO residencies at TWO formats:
 * `FT_TT_OP_REDUCE_FN_RMS_NORM` on the Float32 `reduce_fp32` core, and then
 * `pn_tt_silu_mul(z, normed)` on the bf16 ELEMENTWISE residency. So the fp32 answer is
 * RENDERED INTO A bf16 STRIP before the gate ever sees it.
 *
 * On silicon the Float32 program took the reduce alone from 8.599e-03 to 3.165e-03 (2.7x)
 * and moved `gated` only 1.283e-02 -> 1.023e-02 (1.25x). That is the shape of a term the
 * reduce does not own, and this prices it: the same two steps with the bridge and without.
 * --------------------------------------------------------------------------- */
static void test_gated_is_bound_by_the_bf16_bridge(void)
{
    /* FT_TT_OP_VECTOR_FN_SILU_MUL models a whole tile at a time, so the block is 1024
     * elements = 8 heads of 128. */
    const uint32_t cols = 128u;
    const uint32_t rows = 128u;
    const uint32_t block_rows = 8u;
    const size_t count = (size_t)rows * cols;
    const float eps = 1e-6f;
    float *y = malloc(count * sizeof(float));
    float *z = malloc(count * sizeof(float));
    float *gamma = malloc(cols * sizeof(float));
    float *normed = malloc((size_t)block_rows * cols * sizeof(float));
    float *want = malloc((size_t)block_rows * cols * sizeof(float));
    float *bridged = malloc((size_t)block_rows * cols * sizeof(float));
    double reduce_alone = 0.0;
    double gated_bridged = 0.0;
    double gated_no_bridge = 0.0;
    uint32_t pass0 = 0u;
    uint32_t pass1 = 0u;
    uint32_t final_word = 0u;
    uint32_t block;
    size_t element;

    ft_tt_test_step("`gated`: the fp32 reduce against the bf16 bridge it crosses");
    if (!y || !z || !gamma || !normed || !want || !bridged) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }
    ft_tt_vector_random_seed(0x1220E8ULL);
    for (element = 0; element < count; element++)
        y[element] = ft_tt_vector_random_value();
    for (element = 0; element < count; element++)
        z[element] = ft_tt_vector_random_value();
    for (element = 0; element < cols; element++)
        gamma[element] = 0.5f + ft_tt_vector_random_value() * 0.25f;
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_RMS_NORM, &pass0, &pass1, &final_word);

    for (block = 0; block < rows / block_rows; block++) {
        const size_t base = (size_t)block * block_rows * cols;
        uint32_t row;

        /* Step one: the Float32 reduce, exactly as it ships. */
        ft_tt_vector_model_head_norm_float32(pass0, pass1, final_word, y + base, gamma,
                                             1.0f / (float)cols,
                                             eps * (float)cols / (float)FT_TT_OP_TILE_DIM,
                                             normed, block_rows, cols, 0u);
        ft_tt_vector_reference_rms_norm(y + base, gamma, eps, want, block_rows, cols);
        for (row = 0; row < block_rows; row++)
            reduce_alone = row_max_norm_relative(want + (size_t)row * cols,
                                                 normed + (size_t)row * cols, cols,
                                                 reduce_alone);

        /* Step two AS SHIPPED: that answer rendered into a bf16 strip and gated on the bf16
         * ELEMENTWISE residency. `pn_tt_silu_mul(z, normed)` is silu(a) * b with a = z. */
        ft_tt_vector_model_eltwise(FT_TT_OP_VECTOR_FN_SILU_MUL, z + base, normed, bridged,
                                   (size_t)block_rows * cols);
        for (element = 0; element < (size_t)block_rows * cols; element++)
            want[element] = (float)((double)want[element] *
                                    ((double)z[base + element] /
                                     (1.0 + exp(-(double)z[base + element]))));
        for (row = 0; row < block_rows; row++)
            gated_bridged = row_max_norm_relative(want + (size_t)row * cols,
                                                  bridged + (size_t)row * cols, cols,
                                                  gated_bridged);

        /* THE COUNTERFACTUAL: the same gate with no render in front of it. Not a thing the
         * card can do today — that is the qkv/z strip bridge — and it exists to attribute. */
        for (element = 0; element < (size_t)block_rows * cols; element++) {
            const float gate = z[base + element] / (1.0f + expf(-z[base + element]));

            bridged[element] = normed[element] * gate;
        }
        for (row = 0; row < block_rows; row++)
            gated_no_bridge = row_max_norm_relative(want + (size_t)row * cols,
                                                    bridged + (size_t)row * cols, cols,
                                                    gated_no_bridge);
    }

    ft_tt_test_check_fmt("the bf16 gate costs MORE than the whole Float32 reduce does",
                         gated_bridged > reduce_alone * 2.0,
                         "reduce alone %.3e, then gated %.3e", reduce_alone, gated_bridged);
    ft_tt_test_check_fmt("and removing the render puts `gated` back at the reduce's own figure",
                         gated_no_bridge < gated_bridged * 0.5,
                         "with the bf16 bridge %.3e, without it %.3e — a %.1fx term the "
                         "reduce does not own",
                         gated_bridged, gated_no_bridge, gated_bridged / gated_no_bridge);
    ft_tt_test_check_fmt("so `gated` cannot reach 1e-3 while it crosses into bf16, whatever "
                         "the reduce does",
                         gated_bridged > 1e-3, "%.3e against a 1e-3 gate", gated_bridged);

done:
    free(y);
    free(z);
    free(gamma);
    free(normed);
    free(want);
    free(bridged);
}

static void test_reduce_parameters(void)
{
    ft_tt_op_reduce_request request;
    ft_tt_op_reduce_residency residency;
    ft_tt_op_reduce_function softmax = FT_TT_OP_REDUCE_FN_SOFTMAX;
    uint32_t words[FT_TT_OP_VECTOR_WORDS];
    uint32_t pass0 = 0u;
    uint32_t pass1 = 0u;
    uint32_t final_word = 0u;
    uint32_t ceiling;
    int rc;

    ft_tt_test_step("reduce parameter block");

    memset(&request, 0, sizeof(request));
    request.function = FT_TT_OP_REDUCE_FN_SOFTMAX;
    request.row_blocks = 2u;
    request.tile_cols = 48u;
    request.prefetch_tiles = 32u;
    request.input = make_operand(3u, 4u, 0x100000u);
    request.constants = make_operand(5u, 6u, 0x200000u);
    request.output = make_operand(7u, 8u, 0x300000u);

    rc = ft_tt_op_reduce_parameters(&request, words);
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_SOFTMAX, &pass0, &pass1, &final_word);
    ft_tt_test_check_fmt("a well-formed request encodes", rc == FT_TT_OK, "rc %d", rc);
    ft_tt_test_check_fmt("tile count is row_blocks * tile_cols",
                         words[FT_TT_OP_VECTOR_TILE_COUNT] == 96u, "%u",
                         words[FT_TT_OP_VECTOR_TILE_COUNT]);
    ft_tt_test_check_fmt("tile_cols lands in word 2", words[FT_TT_OP_VECTOR_TILE_COLS] == 48u,
                         "%u", words[FT_TT_OP_VECTOR_TILE_COLS]);
    /* The run a reduce block may not span is Wt, not the total: a block that crossed a row
     * boundary would hand the compute thread half of two different rows. */
    ft_tt_test_check_fmt("depth divides Wt, not the total",
                         words[FT_TT_OP_VECTOR_DEPTH] == 24u, "%u", words[FT_TT_OP_VECTOR_DEPTH]);
    ft_tt_test_check_fmt("the program lands in words 13, 14 and 15",
                         words[FT_TT_OP_REDUCE_PASS0] == pass0 &&
                             words[FT_TT_OP_REDUCE_PASS1] == pass1 &&
                             words[FT_TT_OP_REDUCE_FINAL] == final_word,
                         "0x%08X 0x%08X 0x%08X", words[FT_TT_OP_REDUCE_PASS0],
                         words[FT_TT_OP_REDUCE_PASS1], words[FT_TT_OP_REDUCE_FINAL]);
    /* The one word the family must NOT use, because it means something else to the
     * elementwise binary that shares this block layout. */
    ft_tt_test_check_fmt("word 0 stays 0, so it cannot be read as an eltwise function",
                         words[FT_TT_OP_VECTOR_FUNCTION] == 0u, "%u",
                         words[FT_TT_OP_VECTOR_FUNCTION]);

    memset(&residency, 0, sizeof(residency));
    residency.max_tile_cols = 1u;
    residency.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    ceiling = ft_tt_op_reduce_max_tile_cols(&residency);
    ft_tt_test_check_fmt("a softmax residency's Wt ceiling is the arena's", ceiling >= 300u,
                         "Wt <= %u, i.e. W <= %u", ceiling, ceiling * FT_TT_OP_TILE_DIM);

    request.tile_cols = ceiling;
    rc = ft_tt_op_reduce_parameters(&request, words);
    ft_tt_test_check_fmt("the ceiling itself is accepted", rc == FT_TT_OK, "rc %d at Wt %u", rc,
                         ceiling);
    request.tile_cols = ceiling + 1u;
    rc = ft_tt_op_reduce_parameters(&request, words);
    ft_tt_test_check_fmt("one past the ceiling is refused", rc == FT_TT_EINVAL, "rc %d at Wt %u",
                         rc, ceiling + 1u);

    /* 32k of attention context is Wt = 1024, and it does not fit. This assertion is the
     * capacity argument for flash-decode attention, written down where it can go red. */
    ft_tt_test_check_fmt("a 32k attention row does NOT fit a standalone softmax",
                         ceiling < 1024u, "Wt ceiling %u vs 1024 needed", ceiling);

    /* A residency that also carries gamma and beta holds FOUR rows, not two, so its ceiling
     * is half. Naming both is what stops "the softmax number" being quoted for a norm. */
    {
        static const ft_tt_op_reduce_function all[] = {
            FT_TT_OP_REDUCE_FN_SOFTMAX, FT_TT_OP_REDUCE_FN_RMS_NORM,
            FT_TT_OP_REDUCE_FN_LAYER_NORM, FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE
        };
        ft_tt_op_reduce_residency full;
        uint32_t full_ceiling;

        rc = ft_tt_op_reduce_residency_for(all, sizeof(all) / sizeof(all[0]), 48u,
                                           FT_TT_OP_TILE_BYTES_BF16, &full);
        full_ceiling = ft_tt_op_reduce_max_tile_cols(&full);
        ft_tt_test_check_fmt("a residency for all four carries gamma and beta",
                             rc == FT_TT_OK && full.carries_gamma && full.carries_beta, "rc %d",
                             rc);
        ft_tt_test_check_fmt("and its ceiling is about half the softmax-only one",
                             full_ceiling * 2u <= ceiling + 4u && full_ceiling * 2u + 8u >= ceiling,
                             "four rows Wt <= %u against two rows Wt <= %u", full_ceiling,
                             ceiling);
        rc = ft_tt_op_reduce_residency_for(all, sizeof(all) / sizeof(all[0]), full_ceiling + 1u,
                                           FT_TT_OP_TILE_BYTES_BF16, &full);
        ft_tt_test_check_fmt("one past THAT ceiling is refused at residency time",
                             rc == FT_TT_ENOMEM, "rc %d", rc);
    }

    request.tile_cols = 48u;
    request.row_blocks = 0u;
    rc = ft_tt_op_reduce_parameters(&request, words);
    ft_tt_test_check_fmt("a zero row-block count is refused", rc == FT_TT_EINVAL, "rc %d", rc);
    request.row_blocks = 2u;
    request.function = (ft_tt_op_reduce_function)0;
    rc = ft_tt_op_reduce_parameters(&request, words);
    ft_tt_test_check_fmt("an unknown member is refused", rc == FT_TT_EINVAL, "rc %d", rc);

    (void)softmax;
}

/* ================================================================
 * Circular-buffer placement
 * ================================================================ */

/** Every placed CB must lie in the arena, be 64-byte aligned and overlap nothing. */
static void check_placement(const char *label, const ft_tt_launch_cb_entry *entries,
                            unsigned count)
{
    unsigned index;
    unsigned misaligned = 0;
    unsigned outside = 0;
    unsigned overlapping = 0;

    for (index = 0; index < count; index++) {
        const uint32_t base = entries[index].address;
        const uint32_t bytes = entries[index].size_bytes;
        unsigned other;

        if ((base & 63u) != 0u)
            misaligned++;
        if (base < FT_TT_OP_CB_ARENA_BASE || base + bytes > FT_TT_OP_CB_ARENA_END)
            outside++;
        for (other = 0; other < index; other++) {
            const uint32_t other_base = entries[other].address;
            const uint32_t other_bytes =
                entries[other].size_bytes;

            if (base < other_base + other_bytes && other_base < base + bytes)
                overlapping++;
        }
    }
    ft_tt_test_check_fmt(label, misaligned == 0 && outside == 0 && overlapping == 0,
                         "%u misaligned, %u outside the arena, %u overlapping", misaligned,
                         outside, overlapping);
}

static void test_eltwise_placement(void)
{
    ft_tt_op_eltwise_request request;
    ft_tt_launch_cb_entry entries[FT_TT_OP_ELTWISE_CB_COUNT];
    uint64_t mask = 0;
    int rc;

    ft_tt_test_step("elementwise circular buffers");

    memset(&request, 0, sizeof(request));
    request.function = FT_TT_OP_VECTOR_FN_SILU_MUL;
    request.tile_count = 96u;
    request.prefetch_tiles = 48u;
    request.input_a = make_operand(3u, 4u, 0x100000u);
    request.input_b = make_operand(5u, 6u, 0x200000u);
    request.output = make_operand(7u, 8u, 0x300000u);

    rc = ft_tt_op_eltwise_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("four CBs place", rc == FT_TT_OK, "rc %d", rc);
    check_placement("they are aligned, inside the arena and disjoint", entries,
                    FT_TT_OP_ELTWISE_CB_COUNT);
    ft_tt_test_check_fmt("both inputs hold two blocks",
                         entries[0].page_count == 96u && entries[1].page_count == 96u, "%u, %u",
                         entries[0].page_count, entries[1].page_count);
    ft_tt_test_check_fmt("the mask names exactly the four CBs used",
                         mask == ((1ull << FT_TT_OP_ELTWISE_CB_IN0) |
                                  (1ull << FT_TT_OP_ELTWISE_CB_IN1) |
                                  (1ull << FT_TT_OP_ELTWISE_CB_SCRATCH) |
                                  (1ull << FT_TT_OP_ELTWISE_CB_OUT)),
                         "0x%llX", (unsigned long long)mask);

    /* A unary function still places cb_in1, so the KERNEL alone decides arity and the host
     * never has a table that can disagree with it. */
    request.function = FT_TT_OP_VECTOR_FN_SILU;
    memset(&request.input_b, 0, sizeof(request.input_b));
    rc = ft_tt_op_eltwise_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("a unary request still places cb_in1",
                         rc == FT_TT_OK && entries[1].page_size_bytes == FT_TT_OP_TILE_BYTES_BF16,
                         "rc %d, page %u", rc, entries[1].page_size_bytes);

    /* Past the arena it must REFUSE, not truncate: a truncated CB is a kernel writing over
     * the queue region at the top of L1. */
    request.function = FT_TT_OP_VECTOR_FN_ADD;
    request.input_b = make_operand(5u, 6u, 0x200000u);
    request.tile_count = 4096u;
    request.prefetch_tiles = 4096u;
    rc = ft_tt_op_eltwise_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("a depth past the arena is refused", rc == FT_TT_ENOMEM, "rc %d", rc);
}

static void test_reduce_placement(void)
{
    static const ft_tt_op_reduce_function all[] = {
        FT_TT_OP_REDUCE_FN_SOFTMAX, FT_TT_OP_REDUCE_FN_RMS_NORM,
        FT_TT_OP_REDUCE_FN_LAYER_NORM, FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE
    };
    ft_tt_op_reduce_residency residency;
    ft_tt_launch_cb_entry entries[FT_TT_OP_REDUCE_CB_COUNT];
    uint64_t mask = 0;
    uint32_t ceiling;
    int rc;

    ft_tt_test_step("reduce circular buffers");

    rc = ft_tt_op_reduce_residency_for(all, sizeof(all) / sizeof(all[0]), 48u,
                                       FT_TT_OP_TILE_BYTES_BF16, &residency);
    if (rc == FT_TT_OK)
        rc = ft_tt_op_reduce_place_cbs(&residency, entries, g_cb_buffer, sizeof(g_cb_buffer),
                                       &mask);
    ft_tt_test_check_fmt("eleven CBs place", rc == FT_TT_OK, "rc %d", rc);
    check_placement("they are aligned, inside the arena and disjoint", entries,
                    FT_TT_OP_REDUCE_CB_COUNT);
    ft_tt_test_check_fmt("cb_in, cb_mid, gamma and beta are each a whole row deep",
                         entries[0].page_count == 48u && entries[4].page_count == 48u &&
                             entries[6].page_count == 48u && entries[7].page_count == 48u,
                         "%u, %u, %u, %u", entries[0].page_count, entries[4].page_count,
                         entries[6].page_count, entries[7].page_count);

    /* A softmax-only residency must still PLACE gamma and beta, at one page. An unplaced CB
     * a descriptor then asks for is a NoC write to an address nobody chose (spec §7.3); a
     * one-page CB is a cb_reserve_back that blocks and is retired by its bound. */
    {
        ft_tt_op_reduce_function softmax = FT_TT_OP_REDUCE_FN_SOFTMAX;

        rc = ft_tt_op_reduce_residency_for(&softmax, 1u, 48u, FT_TT_OP_TILE_BYTES_BF16,
                                           &residency);
        if (rc == FT_TT_OK)
            rc = ft_tt_op_reduce_place_cbs(&residency, entries, g_cb_buffer, sizeof(g_cb_buffer),
                                           &mask);
        ft_tt_test_check_fmt("a softmax-only residency still places gamma and beta, at one page",
                             rc == FT_TT_OK && entries[6].page_count == 1u &&
                                 entries[7].page_count == 1u,
                             "rc %d, gamma %u pages, beta %u pages", rc, entries[6].page_count,
                             entries[7].page_count);
        ft_tt_test_check_fmt("the mask names all eleven",
                             mask == ((1ull << FT_TT_OP_REDUCE_CB_IN) |
                                      (1ull << FT_TT_OP_REDUCE_CB_SCALER) |
                                      (1ull << FT_TT_OP_REDUCE_CB_COL0) |
                                      (1ull << FT_TT_OP_REDUCE_CB_COL1) |
                                      (1ull << FT_TT_OP_REDUCE_CB_MID) |
                                      (1ull << FT_TT_OP_REDUCE_CB_EPS) |
                                      (1ull << FT_TT_OP_REDUCE_CB_GAMMA) |
                                      (1ull << FT_TT_OP_REDUCE_CB_BETA) |
                                      (1ull << FT_TT_OP_REDUCE_CB_STAGE0) |
                                      (1ull << FT_TT_OP_REDUCE_CB_STAGE1) |
                                      (1ull << FT_TT_OP_REDUCE_CB_OUT)),
                             "0x%llX", (unsigned long long)mask);
    }

    /* The documented ceiling has to be the REAL one. A ceiling that is merely conservative is
     * a number nobody can plan against, and one that is optimistic is an ENOMEM on the card. */
    ceiling = ft_tt_op_reduce_max_tile_cols(&residency);
    residency.max_tile_cols = ceiling;
    rc = ft_tt_op_reduce_place_cbs(&residency, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("the ceiling places", rc == FT_TT_OK, "rc %d at Wt %u", rc, ceiling);
    residency.max_tile_cols = ceiling + 1u;
    rc = ft_tt_op_reduce_place_cbs(&residency, entries, g_cb_buffer, sizeof(g_cb_buffer), &mask);
    ft_tt_test_check_fmt("one past the ceiling does not place", rc == FT_TT_ENOMEM,
                         "rc %d at Wt %u", rc, ceiling + 1u);
}

/* ================================================================
 * The constants buffer
 * ================================================================ */

/** Untilize one constant tile and return its first element, plus a count of the elements
 *  that disagree with it — a constant tile that is not constant is the interesting failure. */
static float constant_tile_value(const unsigned char *tile, unsigned *out_wrong)
{
    uint16_t plain[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
    float first;
    size_t index;

    *out_wrong = 0u;
    if (ft_tt_untilize_strip(tile, sizeof(uint16_t), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                             FT_TT_OP_TILE_DIM * sizeof(uint16_t), plain) != 0) {
        *out_wrong = 1u;
        return 0.0f;
    }
    first = ft_float_from_bf16(plain[0]);
    for (index = 0; index < FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM; index++) {
        if (ft_float_from_bf16(plain[index]) != first)
            (*out_wrong)++;
    }
    return first;
}

/** `constant_tile_value()` for a Float32 constants buffer: 4-byte elements, and the value is
 *  compared EXACTLY because an fp32 tile has nothing to round it. */
static float constant_tile_value_float32(const unsigned char *tile, unsigned *out_wrong)
{
    float plain[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
    float first;
    size_t index;

    *out_wrong = 0u;
    if (ft_tt_untilize_strip(tile, sizeof(float), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                             FT_TT_OP_TILE_DIM * sizeof(float), plain) != 0) {
        *out_wrong = 1u;
        return 0.0f;
    }
    first = plain[0];
    for (index = 0; index < FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM; index++) {
        if (plain[index] != first)
            (*out_wrong)++;
    }
    return first;
}

/**
 * THE SAME CONSTANTS BUFFER AT Float32, WHICH IS THE ONE HOST PIECE `reduce_float32.cbtable`
 * NEEDS AND THE ONE THAT CANNOT FAIL LOUDLY.
 *
 * A constants buffer built at 2048 under a program placed at 4096 is not refused by anything
 * on the card: the reader issues 4096-byte reads, there is no descriptor header on a
 * data-movement core, and the result is a scaler tile made of half a tile and whatever
 * followed it — a smoothly wrong norm. So the page size is a parameter of the builder and
 * this is what pins it.
 *
 * The GDN head width is 128, which is what the two per-head norms run at.
 */
static void test_reduce_constants_float32(void)
{
    const uint32_t width = 128u;
    const uint32_t tile_cols = width / FT_TT_OP_TILE_DIM;
    const float eps = 1e-6f;
    const size_t bf16_bytes =
        ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_RMS_NORM, tile_cols,
                                        FT_TT_OP_TILE_BYTES_BF16);
    const size_t bytes = ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_RMS_NORM, tile_cols,
                                                         FT_TT_OP_TILE_BYTES_FP32);
    float *gamma = malloc(width * sizeof(float));
    unsigned char *buffer = malloc(bytes);
    unsigned wrong = 0u;
    uint32_t index;
    int rc;

    ft_tt_test_step("the constants buffer at Float32 — the host half of reduce_float32");
    if (!gamma || !buffer) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }
    ft_tt_vector_random_seed(0x1220E9ULL);
    for (index = 0; index < width; index++)
        gamma[index] = 0.5f + ft_tt_vector_random_value() * 0.25f;

    ft_tt_test_check_fmt("an fp32 constants buffer is exactly twice a bf16 one",
                         bytes == 2u * bf16_bytes &&
                             bytes == (size_t)(2u + tile_cols) * FT_TT_OP_TILE_BYTES_FP32,
                         "%zu vs %zu bytes at Wt=%u", bytes, bf16_bytes, tile_cols);

    rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_RMS_NORM, width, eps, gamma, NULL,
                                         FT_TT_OP_TILE_BYTES_FP32, buffer, bytes);
    ft_tt_test_check_fmt("it builds at Float32", rc == FT_TT_OK, "rc %d", rc);
    if (rc != FT_TT_OK)
        goto done;

    /* EXACT equality, not a bf16 round trip — that IS the difference this table buys, and a
     * builder that still went through bf16 would fail here and pass every bf16 check. */
    ft_tt_test_check_fmt("slot 0 is 1/W to the bit, with no bf16 rounding in it",
                         constant_tile_value_float32(buffer, &wrong) == 1.0f / (float)width &&
                             wrong == 0u,
                         "%.9g against %.9g, %u elements disagreeing",
                         (double)constant_tile_value_float32(buffer, &wrong),
                         (double)(1.0f / (float)width), wrong);
    ft_tt_test_check_fmt("slot 1 is eps*W/32 to the bit",
                         constant_tile_value_float32(buffer + FT_TT_OP_TILE_BYTES_FP32,
                                                     &wrong) ==
                                 eps * (float)width / (float)FT_TT_OP_TILE_DIM &&
                             wrong == 0u,
                         "%.9g",
                         (double)constant_tile_value_float32(buffer + FT_TT_OP_TILE_BYTES_FP32,
                                                             &wrong));

    /* gamma, ROW-REPLICATED at the fp32 stride. A builder that kept the bf16 tile stride
     * would put tile 1 half a tile early and this is what sees it. */
    {
        float plain[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
        unsigned bad = 0u;
        uint32_t tile;

        for (tile = 0; tile < tile_cols; tile++) {
            const unsigned char *source =
                buffer + (size_t)(2u + tile) * FT_TT_OP_TILE_BYTES_FP32;
            uint32_t row;
            uint32_t column;

            (void)ft_tt_untilize_strip(source, sizeof(float), FT_TT_OP_TILE_DIM,
                                       FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM * sizeof(float),
                                       plain);
            for (row = 0; row < FT_TT_OP_TILE_DIM; row++) {
                for (column = 0; column < FT_TT_OP_TILE_DIM; column++) {
                    if (plain[row * FT_TT_OP_TILE_DIM + column] !=
                        gamma[tile * FT_TT_OP_TILE_DIM + column])
                        bad++;
                }
            }
        }
        ft_tt_test_check_fmt("gamma is row-replicated at the fp32 tile stride, bit-exact",
                             bad == 0u, "%u of %u elements wrong", bad,
                             tile_cols * FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM);
    }

    /* The L2 member, which is the q/k arm and the one whose eps FOLLOWS THE SCALER: it sums
     * rather than means, so the eps tile is eps/32 and not eps*W/32. */
    rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_L2_NORM, width, eps * eps, NULL,
                                         NULL, FT_TT_OP_TILE_BYTES_FP32, buffer, bytes);
    ft_tt_test_check_fmt("the L2 member builds at Float32 with scaler 1.0 and eps/32",
                         rc == FT_TT_OK &&
                             constant_tile_value_float32(buffer, &wrong) == 1.0f &&
                             constant_tile_value_float32(buffer + FT_TT_OP_TILE_BYTES_FP32,
                                                         &wrong) ==
                                 eps * eps / (float)FT_TT_OP_TILE_DIM,
                         "rc %d, scaler %.9g, eps tile %.9g", rc,
                         (double)constant_tile_value_float32(buffer, &wrong),
                         (double)constant_tile_value_float32(buffer + FT_TT_OP_TILE_BYTES_FP32,
                                                             &wrong));

    /* THE TWO REFUSALS THAT MATTER, because both are the silent-wrong-answer shape. */
    rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_RMS_NORM, width, eps, gamma, NULL,
                                         FT_TT_OP_TILE_BYTES_FP32, buffer, bf16_bytes);
    ft_tt_test_check_fmt("a buffer SIZED at bf16 and BUILT at fp32 is refused, not overrun",
                         rc == FT_TT_EINVAL, "rc %d", rc);
    rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_RMS_NORM, width, eps, gamma, NULL,
                                         1024u, buffer, bytes);
    ft_tt_test_check_fmt("a page size that is neither bf16 nor fp32 is refused",
                         rc == FT_TT_EINVAL, "rc %d", rc);
    ft_tt_test_check_fmt("and so is sizing one",
                         ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_RMS_NORM, tile_cols,
                                                         0u) == 0u,
                         "0 bytes at page size 0, Wt %u", tile_cols);

done:
    free(gamma);
    free(buffer);
}

static void test_reduce_constants(void)
{
    const uint32_t width = 1536u;
    const uint32_t tile_cols = width / FT_TT_OP_TILE_DIM;
    const float eps = 1e-5f;
    float *gamma = malloc(width * sizeof(float));
    float *beta = malloc(width * sizeof(float));
    unsigned char *buffer;
    size_t bytes;
    unsigned wrong = 0u;
    uint32_t index;
    int rc;

    ft_tt_test_step("the reduce constants buffer");

    bytes = ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_LAYER_NORM, tile_cols,
                                            FT_TT_OP_TILE_BYTES_BF16);
    buffer = malloc(bytes);
    if (!gamma || !beta || !buffer) {
        ft_tt_test_check("host buffers allocate", false, "out of memory");
        goto done;
    }
    for (index = 0; index < width; index++) {
        gamma[index] = 0.5f + (float)index * 0.001f;
        beta[index] = -0.25f + (float)index * 0.0005f;
    }

    ft_tt_test_check_fmt("layer norm asks for 2 + 2*Wt tiles",
                         bytes == (size_t)(2u + 2u * tile_cols) * FT_TT_OP_TILE_BYTES_BF16,
                         "%zu bytes for Wt %u", bytes, tile_cols);
    ft_tt_test_check_fmt("softmax asks for the two constant tiles and no stream",
                         ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_SOFTMAX, tile_cols,
                                                         FT_TT_OP_TILE_BYTES_BF16) ==
                             2u * FT_TT_OP_TILE_BYTES_BF16,
                         "%zu",
                         ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_SOFTMAX, tile_cols,
                                                         FT_TT_OP_TILE_BYTES_BF16));

    rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_LAYER_NORM, width, eps, gamma, beta,
                                         FT_TT_OP_TILE_BYTES_BF16, buffer, bytes);
    ft_tt_test_check_fmt("it builds", rc == FT_TT_OK, "rc %d", rc);

    /* Slot 0: the scaler. 1/W for a member that wants a MEAN. */
    ft_tt_test_check_fmt("slot 0 is the 1/W scaler",
                         constant_tile_value(buffer, &wrong) ==
                                 ft_float_from_bf16(ft_bf16_from_float(1.0f / (float)width)) &&
                             wrong == 0u,
                         "%g with %u elements disagreeing",
                         (double)constant_tile_value(buffer, &wrong), wrong);

    /* Slot 1: eps*W/32, which is the whole trick — each row of that tile has 32 columns, so
     * 32 * (eps*W/32) * (1/W) = eps, and eps costs one extra reduce_tile instead of a
     * broadcast add and a CB round trip. */
    ft_tt_test_check_fmt("slot 1 is eps*W/32, so one reduce_tile adds exactly eps",
                         constant_tile_value(buffer + FT_TT_OP_TILE_BYTES_BF16, &wrong) ==
                                 ft_float_from_bf16(ft_bf16_from_float(
                                     eps * (float)width / (float)FT_TT_OP_TILE_DIM)) &&
                             wrong == 0u,
                         "%g",
                         (double)constant_tile_value(buffer + FT_TT_OP_TILE_BYTES_BF16, &wrong));

    /* The gamma stream: each tile's every ROW must repeat that tile's 32 gamma values, which
     * is what turns the multiply into a plain mul_tiles rather than a broadcast. */
    {
        uint16_t plain[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
        unsigned bad_columns = 0u;
        unsigned bad_rows = 0u;
        uint32_t tile;

        for (tile = 0; tile < tile_cols; tile++) {
            const unsigned char *source =
                buffer + (size_t)(2u + tile) * FT_TT_OP_TILE_BYTES_BF16;
            uint32_t row;
            uint32_t column;

            (void)ft_tt_untilize_strip(source, sizeof(uint16_t), FT_TT_OP_TILE_DIM,
                                       FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM * sizeof(uint16_t),
                                       plain);
            for (column = 0; column < FT_TT_OP_TILE_DIM; column++) {
                const float want =
                    ft_float_from_bf16(ft_bf16_from_float(gamma[tile * FT_TT_OP_TILE_DIM +
                                                                column]));

                if (ft_float_from_bf16(plain[column]) != want)
                    bad_columns++;
                for (row = 1; row < FT_TT_OP_TILE_DIM; row++) {
                    if (ft_float_from_bf16(plain[row * FT_TT_OP_TILE_DIM + column]) != want)
                        bad_rows++;
                }
            }
        }
        ft_tt_test_check_fmt("gamma expands to full tiles, row-replicated",
                             bad_columns == 0u && bad_rows == 0u,
                             "%u wrong columns, %u wrong replications", bad_columns, bad_rows);
    }

    rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_LAYER_NORM, width, eps, gamma, NULL,
                                         FT_TT_OP_TILE_BYTES_BF16, buffer, bytes);
    ft_tt_test_check_fmt("layer norm without a beta is refused, not silently zeroed",
                         rc == FT_TT_EINVAL, "rc %d", rc);
    rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_LAYER_NORM, width, eps, gamma, beta,
                                         FT_TT_OP_TILE_BYTES_BF16, buffer, 8u);
    ft_tt_test_check_fmt("a short buffer is refused", rc == FT_TT_EINVAL, "rc %d", rc);

done:
    free(gamma);
    free(beta);
    free(buffer);
}

/* ================================================================
 * The model against the reference — the PCC prediction
 * ================================================================ */

/** Score one array pair the way the hardware pass will, and print both metrics. */
static void score(const char *label, const float *expected, const float *actual, size_t count)
{
    char census[192];
    const size_t non_finite =
        ft_tt_vector_count_non_finite(expected, actual, count, census, sizeof(census));
    double pcc;
    double worst;

    if (non_finite != 0u) {
        ft_tt_test_check_fmt(label, false, "NON-FINITE: %s", census);
        return;
    }
    pcc = ft_tt_vector_pearson(expected, actual, count);
    worst = ft_tt_vector_max_abs_difference(expected, actual, count);
    ft_tt_test_check_fmt(label, pcc >= PCC_ACCEPT, "PCC %.9f, max abs %.6g", pcc, worst);
}

/**
 * THE EPS TILE FOLLOWS THE SCALER, and a member that SUMS needs a different value from one
 * that MEANS.
 *
 * The eps trick is one more accumulated row reduce, so the pass contributes
 * `32 * tile_value * scaler`. At scaler 1/W that wants `eps*W/32`; at scaler 1.0 it wants
 * `eps/32`. The unconditional `eps*W/32` that shipped before L2_NORM existed is wrong by a
 * factor of W for any summing member — 128x at the GDN's head width — and nothing on the
 * card can see it, because an eps that is too large is a smoothly wrong norm.
 */
static void test_eps_tile_follows_the_scaler(void)
{
    const uint32_t width = 128u;
    const uint32_t tile_cols = width / FT_TT_OP_TILE_DIM;
    const float eps = 1e-5f;
    unsigned char *buffer = NULL;
    size_t bytes;
    unsigned wrong = 0u;
    int rc;

    ft_tt_test_step("the eps tile is scaled by the member's own reduce scaler");

    bytes = ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_L2_NORM, tile_cols,
                                            FT_TT_OP_TILE_BYTES_BF16);
    buffer = malloc(bytes);
    if (!buffer) {
        ft_tt_test_check("host buffer allocates", false, "out of memory");
        return;
    }

    ft_tt_test_check_fmt("L2 norm asks for the two constant tiles and no stream",
                         bytes == 2u * FT_TT_OP_TILE_BYTES_BF16, "%zu bytes", bytes);

    rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_L2_NORM, width, eps, NULL, NULL,
                                         FT_TT_OP_TILE_BYTES_BF16, buffer, bytes);
    ft_tt_test_check_fmt("it builds with no gamma and no beta", rc == FT_TT_OK, "rc %d", rc);

    ft_tt_test_check_fmt("slot 0 is 1.0, not 1/W — the reduce is a SUM",
                         constant_tile_value(buffer, &wrong) == 1.0f && wrong == 0u,
                         "%g with %u elements disagreeing",
                         (double)constant_tile_value(buffer, &wrong), wrong);
    ft_tt_test_check_fmt("slot 1 is eps/32, not eps*W/32",
                         constant_tile_value(buffer + FT_TT_OP_TILE_BYTES_BF16, &wrong) ==
                                 ft_float_from_bf16(ft_bf16_from_float(
                                     eps / (float)FT_TT_OP_TILE_DIM)) &&
                             wrong == 0u,
                         "%g with %u elements disagreeing",
                         (double)constant_tile_value(buffer + FT_TT_OP_TILE_BYTES_BF16, &wrong),
                         wrong);

    /* The two members that already shipped are UNMOVED: their scaler is a mean, so the
     * value is the one they have always carried. */
    {
        float gamma[128];
        unsigned char *rms;
        uint32_t index;

        for (index = 0; index < width; index++)
            gamma[index] = 0.5f + (float)index * 0.001f;
        bytes = ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_RMS_NORM, tile_cols,
                                                FT_TT_OP_TILE_BYTES_BF16);
        rms = malloc(bytes);
        if (!rms) {
            ft_tt_test_check("host buffer allocates", false, "out of memory");
            free(buffer);
            return;
        }
        rc = ft_tt_op_reduce_build_constants(FT_TT_OP_REDUCE_FN_RMS_NORM, width, eps, gamma,
                                             NULL, FT_TT_OP_TILE_BYTES_BF16, rms, bytes);
        ft_tt_test_check_fmt("RMS norm's eps tile is still eps*W/32",
                             rc == FT_TT_OK &&
                                 constant_tile_value(rms + FT_TT_OP_TILE_BYTES_BF16, &wrong) ==
                                     ft_float_from_bf16(ft_bf16_from_float(
                                         eps * (float)width / (float)FT_TT_OP_TILE_DIM)),
                             "rc %d, %g", rc,
                             (double)constant_tile_value(rms + FT_TT_OP_TILE_BYTES_BF16,
                                                         &wrong));
        free(rms);
    }
    free(buffer);
}

static void test_eltwise_model(void)
{
    static const ft_tt_op_vector_function functions[] = {
        FT_TT_OP_VECTOR_FN_SILU, FT_TT_OP_VECTOR_FN_SIGMOID, FT_TT_OP_VECTOR_FN_ADD,
        FT_TT_OP_VECTOR_FN_MUL,  FT_TT_OP_VECTOR_FN_SUB,     FT_TT_OP_VECTOR_FN_SILU_MUL
    };
    const size_t count = 48u * FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM;
    float *a = malloc(count * sizeof(float));
    float *b = malloc(count * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    unsigned index;

    ft_tt_test_step("elementwise: the tile-space model against the fp32 reference");
    if (!a || !b || !expected || !actual) {
        ft_tt_test_check("allocation", false, "out of memory");
        free(a);
        free(b);
        free(expected);
        free(actual);
        return;
    }

    for (index = 0; index < sizeof(functions) / sizeof(functions[0]); index++) {
        const ft_tt_op_vector_function function = functions[index];
        const bool binary = ft_tt_op_vector_function_is_binary(function);
        size_t element;
        char label[96];

        ft_tt_vector_random_seed(0x51EDULL + index);
        for (element = 0; element < count; element++) {
            a[element] = ft_tt_vector_random_value();
            b[element] = ft_tt_vector_random_value();
        }
        /* The reference consumes the values the CARD will see — bf16-rounded — so the score
         * is about the OP and not about a rounding the kernel is not responsible for. */
        for (element = 0; element < count; element++) {
            a[element] = ft_float_from_bf16(ft_bf16_from_float(a[element]));
            b[element] = ft_float_from_bf16(ft_bf16_from_float(b[element]));
        }

        ft_tt_vector_reference_eltwise(function, a, binary ? b : NULL, expected, count);
        ft_tt_vector_model_eltwise(function, a, binary ? b : NULL, actual, count);
        snprintf(label, sizeof(label), "%s over %u tiles",
                 ft_tt_op_vector_function_name(function), 48u);
        score(label, expected, actual, count);
    }

    free(a);
    free(b);
    free(expected);
    free(actual);
}

static void test_softmax_model(void)
{
    static const uint32_t widths[] = { 32u, 64u, 1536u, 4096u };
    const uint32_t rows = 64u;
    unsigned index;

    ft_tt_test_step("softmax: the tile-space model against the fp32 reference");

    for (index = 0; index < sizeof(widths) / sizeof(widths[0]); index++) {
        const uint32_t cols = widths[index];
        const size_t count = (size_t)rows * cols;
        float *x = malloc(count * sizeof(float));
        float *expected = malloc(count * sizeof(float));
        float *actual = malloc(count * sizeof(float));
        size_t element;
        char label[96];

        if (!x || !expected || !actual) {
            ft_tt_test_check("allocation", false, "out of memory");
            free(x);
            free(expected);
            free(actual);
            return;
        }
        ft_tt_vector_random_seed(0x50F7ULL + index);
        for (element = 0; element < count; element++)
            x[element] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value()));

        ft_tt_vector_reference_softmax(x, expected, rows, cols);
        ft_tt_vector_model_softmax(x, actual, rows, cols);
        snprintf(label, sizeof(label), "%u x %u (Wt = %u)", rows, cols,
                 cols / FT_TT_OP_TILE_DIM);
        score(label, expected, actual, count);

        free(x);
        free(expected);
        free(actual);
    }
}

/**
 * The max-subtraction pass, proved load-bearing by the only data that can prove it.
 *
 * At the [-1, 1) range every other test uses, a softmax with NO max subtraction scores
 * identically — exp(1) is 2.7 and nothing overflows. So a PCC gate on random small
 * operands cannot see the difference, and a kernel that dropped the pass would ship green.
 * At a range where exp overflows, the same arrangement is the difference between a result
 * and a page of infinities.
 */
static void test_softmax_dynamic_range(void)
{
    const uint32_t rows = 32u;
    const uint32_t cols = 128u;
    const size_t count = (size_t)rows * cols;
    float *x = malloc(count * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    size_t element;
    unsigned naive_non_finite = 0;

    ft_tt_test_step("softmax: the range where the max pass stops being cosmetic");
    if (!x || !expected || !actual) {
        ft_tt_test_check("allocation", false, "out of memory");
        free(x);
        free(expected);
        free(actual);
        return;
    }

    ft_tt_vector_random_seed(0xBADFULL);
    for (element = 0; element < count; element++)
        x[element] = ft_float_from_bf16(ft_bf16_from_float(
            ft_tt_vector_random_value_scaled(120.0f)));

    /* What a max-free implementation would produce, computed here rather than assumed. */
    for (element = 0; element < count; element++) {
        if (!isfinite(expf(x[element])))
            naive_non_finite++;
    }
    ft_tt_test_check_fmt("exp(x) alone overflows on this data", naive_non_finite > 0u,
                         "%u of %zu elements", naive_non_finite, count);

    ft_tt_vector_reference_softmax(x, expected, rows, cols);
    ft_tt_vector_model_softmax(x, actual, rows, cols);
    score("with the max pass, 32 x 128 at range +/-120", expected, actual, count);

    free(x);
    free(expected);
    free(actual);
}

/**
 * The max must be taken over the WHOLE ROW, not over the first tile — and this is the only
 * test in the file that can tell.
 *
 * FOUND BY A MUTATION THAT CAME BACK GREEN. Reducing the model's max over one tile instead
 * of all Wt fails nothing above, and the reason is mathematics rather than a weak
 * assertion: softmax(x - c) = softmax(x) for ANY constant c, so subtracting the wrong
 * maximum changes nothing at all as long as no exponential overflows. Every PCC check in
 * this file is therefore structurally blind to a partial-row max, and so is the on-card
 * acceptance, and so would a reviewer be.
 *
 * What is not blind to it is a row whose largest element is NOT in the first tile and is
 * more than ~88 above it — then a partial max leaves exp() overflowing and the answer is a
 * page of infinities instead of a slightly different constant. That is this test: a plain
 * [-1, 1) row with ONE outlier of +200 planted in the LAST tile.
 *
 * The general lesson is dev notes §3's, one op further on: a check with a mathematical
 * invariance in it is a check that cannot see whatever the invariance absorbs.
 */
static void test_softmax_max_is_over_the_whole_row(void)
{
    const uint32_t rows = 32u;
    const uint32_t cols = 256u; /* Wt = 8, so "the first tile" is an eighth of the row */
    const size_t count = (size_t)rows * cols;
    float *x = malloc(count * sizeof(float));
    float *expected = malloc(count * sizeof(float));
    float *actual = malloc(count * sizeof(float));
    size_t element;
    uint32_t row;

    ft_tt_test_step("softmax: the max is over the WHOLE row, not the first tile");
    if (!x || !expected || !actual) {
        ft_tt_test_check("allocation", false, "out of memory");
        free(x);
        free(expected);
        free(actual);
        return;
    }

    ft_tt_vector_random_seed(0x0177ULL);
    for (element = 0; element < count; element++)
        x[element] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value()));
    /* One outlier per row, in the LAST tile. 200 is well past the ~88 where exp overflows
     * fp32, so a max taken over any earlier tile cannot rescue it. */
    for (row = 0; row < rows; row++)
        x[(size_t)row * cols + (cols - 3u)] = 200.0f;

    ft_tt_vector_reference_softmax(x, expected, rows, cols);
    ft_tt_vector_model_softmax(x, actual, rows, cols);
    score("an outlier of +200 in the last tile, 32 x 256", expected, actual, count);

    free(x);
    free(expected);
    free(actual);
}


/**
 * The three new members' tile-space models against their fp32 references.
 *
 * Same purpose as the softmax model test: predict the on-card PCC before spending a core on
 * it. §22 recorded that the model called silicon to ~6 decimals for the two ops it had, so
 * a prediction here is worth making and worth checking against the hardware pass afterwards.
 */
static void test_reduce_models(void)
{
    static const uint32_t widths[] = { 32u, 1536u };
    const uint32_t rows = 64u;
    const float eps = 1e-5f;
    unsigned index;

    ft_tt_test_step("RMS norm, layer norm and the router: model against reference");

    for (index = 0; index < sizeof(widths) / sizeof(widths[0]); index++) {
        const uint32_t cols = widths[index];
        const size_t count = (size_t)rows * cols;
        float *x = malloc(count * sizeof(float));
        float *gamma = malloc(cols * sizeof(float));
        float *beta = malloc(cols * sizeof(float));
        float *expected = malloc(count * sizeof(float));
        float *actual = malloc(count * sizeof(float));
        size_t element;
        char label[96];

        if (!x || !gamma || !beta || !expected || !actual) {
            ft_tt_test_check("allocation", false, "out of memory");
            goto next;
        }
        ft_tt_vector_random_seed(0x4EDCE0ULL + index);
        for (element = 0; element < count; element++)
            x[element] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value()));
        for (element = 0; element < cols; element++) {
            gamma[element] =
                ft_float_from_bf16(ft_bf16_from_float(0.5f + ft_tt_vector_random_value() * 0.25f));
            beta[element] =
                ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value() * 0.1f));
        }

        ft_tt_vector_reference_rms_norm(x, gamma, eps, expected, rows, cols);
        ft_tt_vector_model_rms_norm(x, gamma, eps, actual, rows, cols);
        snprintf(label, sizeof(label), "RMS norm %u x %u (Wt = %u)", rows, cols,
                 cols / FT_TT_OP_TILE_DIM);
        score(label, expected, actual, count);

        ft_tt_vector_reference_layer_norm(x, gamma, beta, eps, expected, rows, cols);
        ft_tt_vector_model_layer_norm(x, gamma, beta, eps, actual, rows, cols);
        snprintf(label, sizeof(label), "layer norm %u x %u (Wt = %u)", rows, cols,
                 cols / FT_TT_OP_TILE_DIM);
        score(label, expected, actual, count);

        /* Positive gates: x/sum(x) on signed noise divides by something near zero, which is
         * a property of the TEST DATA and not of the op. */
        for (element = 0; element < count; element++)
            x[element] = ft_float_from_bf16(ft_bf16_from_float(0.05f + fabsf(x[element])));
        ft_tt_vector_reference_router_normalise(x, expected, rows, cols);
        ft_tt_vector_model_router_normalise(x, actual, rows, cols);
        snprintf(label, sizeof(label), "router normalise %u x %u (Wt = %u)", rows, cols,
                 cols / FT_TT_OP_TILE_DIM);
        score(label, expected, actual, count);

    next:
        free(x);
        free(gamma);
        free(beta);
        free(expected);
        free(actual);
    }
}

/**
 * PCC IS BLIND TO A GLOBAL SCALE, AND THREE OF THE FOUR MEMBERS ARE SCALE-SHAPED.
 *
 * This is §22.2's lesson in its second form. Softmax hides a wrong row maximum behind
 * shift-invariance; RMS norm, layer norm and the router hide a wrong SCALE behind the fact
 * that a correlation is invariant under multiplying one side by a constant. A reduce that
 * summed the row twice, a scaler tile off by two, an rsqrt where a reciprocal belongs — all
 * of them are exactly that, and every PCC gate in this file and in the on-card acceptance
 * would pass them.
 *
 * So this test does the mutation itself, in the model, and asserts BOTH things: PCC does not
 * move, and max-abs does. That is the evidence for scoring the family on a relative max-abs
 * budget as well as on PCC, and it is what the on-card kernel mutation then confirms.
 */
static void test_pcc_is_blind_to_scale(void)
{
    const uint32_t rows = 32u;
    const uint32_t cols = 1536u;
    const size_t count = (size_t)rows * cols;
    float *x = malloc(count * sizeof(float));
    float *gamma = malloc(cols * sizeof(float));
    float *reference = malloc(count * sizeof(float));
    float *scaled = malloc(count * sizeof(float));
    size_t element;
    double pcc;
    double worst;
    double peak = 0.0;

    ft_tt_test_step("PCC cannot see a global scale — which is why max-abs is a CHECK here");
    if (!x || !gamma || !reference || !scaled) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }

    ft_tt_vector_random_seed(0x5CA1EULL);
    for (element = 0; element < count; element++)
        x[element] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value()));
    for (element = 0; element < cols; element++)
        gamma[element] = 1.0f;

    ft_tt_vector_reference_rms_norm(x, gamma, 1e-5f, reference, rows, cols);
    /* A reduce that accumulated each tile TWICE gives a mean square 2x too large, hence an
     * output 1/sqrt(2) too small. That is a real mutation of the kernel's inner loop, and
     * this is what it does to the answer. */
    for (element = 0; element < count; element++)
        scaled[element] = reference[element] * (float)(1.0 / sqrt(2.0));
    for (element = 0; element < count; element++) {
        if (fabs((double)reference[element]) > peak)
            peak = fabs((double)reference[element]);
    }

    pcc = ft_tt_vector_pearson(reference, scaled, count);
    worst = ft_tt_vector_max_abs_difference(reference, scaled, count);
    ft_tt_test_check_fmt("a 29% scale error keeps PCC at 1.0 — the gate does not see it",
                         pcc >= 0.999999, "PCC %.9f", pcc);
    ft_tt_test_check_fmt("and max-abs sees it at 100x the 5% budget",
                         worst > 0.05 * peak, "max abs %.6g against a %.6g budget", worst,
                         0.05 * peak);

done:
    free(x);
    free(gamma);
    free(reference);
    free(scaled);
}

/* ================================================================
 * A RAGGED W, SERVED BY PADDING — and the one member it cannot serve
 * ================================================================
 *
 * The reduce triad has no mask tile: it reduces over `tile_cols * 32` columns
 * unconditionally, and a ragged W has no wire representation at all. What closes that for
 * three of the four members is a caller that PADS — pad value from the member's own passes,
 * DIVISOR from the real W — and what these checks are for is that the padded pipeline lands
 * on the true-W reference, not merely near it.
 *
 * THE MODEL IS NOT TOLD HOW THE PADDING WAS DONE. It is handed the padded activation, the
 * scaler and eps values UNTILIZED OUT OF the constants buffer the host built, the gamma row
 * read back the same way, and the program words — i.e. only what the card would get. A model
 * that recomputed 1/W itself would agree with a wrong builder and prove nothing (dev notes
 * §3).
 *
 * AND LAYER NORM IS MEASURED, NOT ASSERTED. Its second pass squares (x - row mean), so a
 * padded lane contributes (pad - mean)^2 and vanishes only at pad == the row's own mean. The
 * last stage sweeps constant pads across rows with deliberately DIFFERENT means and shows
 * every one of them wrong — which is why the builder refuses it instead of shipping it with
 * a caveat.
 */

/** Widest padded row any stage below builds, for the fixed-size scratch the sweep uses. */
#define RAGGED_MAX_PADDED_COLS 1536u

/** Read a gamma/beta stream back out of a constants buffer, the way the card will index it:
 *  row 0 of each 32x32 tile, `tile_cols` tiles. Reading the BYTES rather than the caller's
 *  array is what makes the padded tail an observation instead of an assumption. */
static void read_broadcast_row(const unsigned char *tiles, uint32_t tile_cols, float *out)
{
    uint32_t tile;

    for (tile = 0; tile < tile_cols; tile++) {
        uint16_t plain[FT_TT_OP_TILE_DIM * FT_TT_OP_TILE_DIM];
        uint32_t column;

        (void)ft_tt_untilize_strip(tiles + (size_t)tile * FT_TT_OP_TILE_BYTES_BF16,
                                   sizeof(uint16_t), FT_TT_OP_TILE_DIM, FT_TT_OP_TILE_DIM,
                                   FT_TT_OP_TILE_DIM * sizeof(uint16_t), plain);
        for (column = 0; column < FT_TT_OP_TILE_DIM; column++)
            out[tile * FT_TT_OP_TILE_DIM + column] = ft_float_from_bf16(plain[column]);
    }
}

/** Test data shaped for the member: the router divides by a row sum, so it needs positive
 *  gates; softmax gets an ALL-NEGATIVE row, which is the arm a zero pad gets wrong twice
 *  over (a max clamped at 0, and a denominator with exp(0 - max) in it). */
static void fill_member_input(ft_tt_op_reduce_function function, float *out, size_t count)
{
    size_t index;

    for (index = 0; index < count; index++) {
        const float value = ft_tt_vector_random_value();

        if (function == FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE)
            out[index] = ft_float_from_bf16(ft_bf16_from_float(0.05f + fabsf(value)));
        else if (function == FT_TT_OP_REDUCE_FN_SOFTMAX)
            out[index] = ft_float_from_bf16(ft_bf16_from_float(value - 4.0f));
        else
            out[index] = ft_float_from_bf16(ft_bf16_from_float(value));
    }
}

static void reference_for(ft_tt_op_reduce_function function, const float *x, const float *gamma,
                          float eps, float *out, uint32_t rows, uint32_t cols)
{
    if (function == FT_TT_OP_REDUCE_FN_SOFTMAX)
        ft_tt_vector_reference_softmax(x, out, rows, cols);
    else if (function == FT_TT_OP_REDUCE_FN_RMS_NORM)
        ft_tt_vector_reference_rms_norm(x, gamma, eps, out, rows, cols);
    else
        ft_tt_vector_reference_router_normalise(x, out, rows, cols);
}

/**
 * One member at one W through the padding caller, scored against the true-W reference, with
 * an exact-W control at the PADDED width as the rounding floor.
 *
 * The control is what turns the verdict into a comparison instead of a magic constant: a
 * padded lane contributes the pool's identity, so padding must cost nothing beyond the bf16
 * rounding the same shape already pays.
 *
 * @return false only if the run could not be made (allocation); the checks report themselves
 */
static bool score_padded_member(ft_tt_op_reduce_function function, uint32_t width, float eps,
                                unsigned seed)
{
    const uint32_t tile_cols = ft_tt_op_reduce_padded_tile_cols(width);
    const uint32_t padded = tile_cols * FT_TT_OP_TILE_DIM;
    const uint32_t rows = 32u;
    const size_t padded_count = (size_t)rows * padded;
    float *x = malloc((size_t)rows * width * sizeof(float));
    float *x_padded = malloc(padded_count * sizeof(float));
    float *gamma = malloc(padded * sizeof(float));
    float *gamma_wire = malloc(padded * sizeof(float));
    float *expected = malloc(padded_count * sizeof(float));
    float *modelled = malloc(padded_count * sizeof(float));
    float *sliced = malloc(padded_count * sizeof(float));
    unsigned char *constants = NULL;
    size_t constants_bytes =
        ft_tt_op_reduce_constants_bytes(function, tile_cols, FT_TT_OP_TILE_BYTES_BF16);
    double ragged_worst = 0.0;
    double control_worst = 0.0;
    double ragged_pcc = 0.0;
    uint32_t pass0 = 0u;
    uint32_t pass1 = 0u;
    uint32_t final_word = 0u;
    float scaler = 0.0f;
    float eps_tile = 0.0f;
    unsigned wrong = 0u;
    uint32_t row;
    size_t element;
    char label[128];
    bool made = false;
    int rc;

    constants = malloc(constants_bytes);
    if (!x || !x_padded || !gamma || !gamma_wire || !expected || !modelled || !sliced ||
        !constants)
        goto done;

    ft_tt_vector_random_seed(0x5AD9E0ULL + seed);
    fill_member_input(function, x, (size_t)rows * width);
    for (element = 0; element < padded; element++)
        gamma[element] =
            ft_float_from_bf16(ft_bf16_from_float(0.5f + ft_tt_vector_random_value() * 0.25f));

    /* ---- the ragged arm: everything the card sees comes from the caller ---- */
    rc = ft_tt_op_reduce_pad_rows(function, x, rows, width, x_padded, padded_count);
    snprintf(label, sizeof(label), "%s W=%u: the row pads to Wt=%u",
             ft_tt_op_reduce_function_name(function), width, tile_cols);
    ft_tt_test_check_fmt(label, rc == FT_TT_OK, "rc %d", rc);
    rc = ft_tt_op_reduce_build_constants_ragged(function, width, eps, gamma, NULL,
                                                FT_TT_OP_TILE_BYTES_BF16, constants,
                                                constants_bytes);
    snprintf(label, sizeof(label), "%s W=%u: the constants build at the real W",
             ft_tt_op_reduce_function_name(function), width);
    ft_tt_test_check_fmt(label, rc == FT_TT_OK, "rc %d", rc);
    if (rc != FT_TT_OK)
        goto done;

    scaler = constant_tile_value(constants, &wrong);
    eps_tile = constant_tile_value(constants + FT_TT_OP_TILE_BYTES_BF16, &wrong);
    if (ft_tt_op_reduce_uses_gamma(function))
        read_broadcast_row(constants + 2u * FT_TT_OP_TILE_BYTES_BF16, tile_cols, gamma_wire);
    else
        memset(gamma_wire, 0, padded * sizeof(float));

    (void)ft_tt_op_reduce_program(function, &pass0, &pass1, &final_word);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x_padded, gamma_wire, NULL,
                                      scaler, eps_tile, modelled, rows, padded);

    /* Only the real columns are the answer; the padded tail is scratch by construction. */
    for (row = 0; row < rows; row++)
        memcpy(sliced + (size_t)row * width, modelled + (size_t)row * padded,
               width * sizeof(float));
    reference_for(function, x, gamma, eps, expected, rows, width);
    ragged_worst = ft_tt_vector_max_abs_difference(expected, sliced, (size_t)rows * width);
    ragged_pcc = ft_tt_vector_pearson(expected, sliced, (size_t)rows * width);

    /* ---- the control: the same shape at an exact W, which needs no padding at all ---- */
    fill_member_input(function, x_padded, padded_count);
    rc = ft_tt_op_reduce_build_constants(function, padded, eps, gamma, NULL,
                                         FT_TT_OP_TILE_BYTES_BF16, constants, constants_bytes);
    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt("the exact-width control builds", false, "rc %d", rc);
        goto done;
    }
    scaler = constant_tile_value(constants, &wrong);
    eps_tile = constant_tile_value(constants + FT_TT_OP_TILE_BYTES_BF16, &wrong);
    if (ft_tt_op_reduce_uses_gamma(function))
        read_broadcast_row(constants + 2u * FT_TT_OP_TILE_BYTES_BF16, tile_cols, gamma_wire);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x_padded, gamma_wire, NULL,
                                      scaler, eps_tile, modelled, rows, padded);
    reference_for(function, x_padded, gamma, eps, expected, rows, padded);
    control_worst = ft_tt_vector_max_abs_difference(expected, modelled, padded_count);

    snprintf(label, sizeof(label), "%s W=%u: padding costs no more than the bf16 floor",
             ft_tt_op_reduce_function_name(function), width);
    ft_tt_test_check_fmt(label,
                         ragged_pcc >= PCC_ACCEPT &&
                             ragged_worst <= 4.0 * control_worst + 1e-9,
                         "max abs %.6g against an exact-W floor of %.6g, PCC %.9f",
                         ragged_worst, control_worst, ragged_pcc);
    made = true;

done:
    if (!made && !constants)
        ft_tt_test_check("allocation", false, "out of memory");
    free(x);
    free(x_padded);
    free(gamma);
    free(gamma_wire);
    free(expected);
    free(modelled);
    free(sliced);
    free(constants);
    return made;
}

/** The emulator against the four models it has to agree with, at an exact W. A model driven
 *  by the program words is only a fair instrument if it reproduces the hand-written ones
 *  where both apply — and it has to be BIT-exact, because both apply the same rounding at
 *  the same places and anything else would be an unexplained difference. */
static void test_program_model_matches_the_member_models(void)
{
    const uint32_t rows = 32u;
    const uint32_t cols = 1536u;
    const uint32_t tile_cols = cols / FT_TT_OP_TILE_DIM;
    const size_t count = (size_t)rows * cols;
    const float eps = 1e-5f;
    float *x = malloc(count * sizeof(float));
    float *gamma = malloc(cols * sizeof(float));
    float *beta = malloc(cols * sizeof(float));
    float *from_member = malloc(count * sizeof(float));
    float *from_program = malloc(count * sizeof(float));
    const float scaler_mean = ft_float_from_bf16(ft_bf16_from_float(1.0f / (float)cols));
    const float eps_tile = ft_float_from_bf16(
        ft_bf16_from_float(eps * (float)cols / (float)FT_TT_OP_TILE_DIM));
    uint32_t pass0 = 0u;
    uint32_t pass1 = 0u;
    uint32_t final_word = 0u;
    size_t element;

    (void)tile_cols;
    ft_tt_test_step("the program-driven model reproduces the four member models exactly");
    if (!x || !gamma || !beta || !from_member || !from_program) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }
    ft_tt_vector_random_seed(0x9C0DEFULL);
    for (element = 0; element < count; element++)
        x[element] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value()));
    for (element = 0; element < cols; element++) {
        gamma[element] =
            ft_float_from_bf16(ft_bf16_from_float(0.5f + ft_tt_vector_random_value() * 0.25f));
        beta[element] = ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value() * 0.1f));
    }

    ft_tt_vector_model_rms_norm(x, gamma, eps, from_member, rows, cols);
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_RMS_NORM, &pass0, &pass1, &final_word);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x, gamma, beta, scaler_mean,
                                      eps_tile, from_program, rows, cols);
    ft_tt_test_check_fmt("RMS norm", ft_tt_vector_max_abs_difference(from_member, from_program,
                                                                    count) == 0.0,
                         "max abs %.6g",
                         ft_tt_vector_max_abs_difference(from_member, from_program, count));

    ft_tt_vector_model_layer_norm(x, gamma, beta, eps, from_member, rows, cols);
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_LAYER_NORM, &pass0, &pass1, &final_word);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x, gamma, beta, scaler_mean,
                                      eps_tile, from_program, rows, cols);
    ft_tt_test_check_fmt("layer norm", ft_tt_vector_max_abs_difference(from_member, from_program,
                                                                      count) == 0.0,
                         "max abs %.6g",
                         ft_tt_vector_max_abs_difference(from_member, from_program, count));

    ft_tt_vector_model_softmax(x, from_member, rows, cols);
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_SOFTMAX, &pass0, &pass1, &final_word);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x, gamma, beta, 1.0f, 0.0f,
                                      from_program, rows, cols);
    ft_tt_test_check_fmt("softmax",
                         ft_tt_vector_max_abs_difference(from_member, from_program, count) == 0.0,
                         "max abs %.6g",
                         ft_tt_vector_max_abs_difference(from_member, from_program, count));

    for (element = 0; element < count; element++)
        x[element] = ft_float_from_bf16(ft_bf16_from_float(0.05f + fabsf(x[element])));
    ft_tt_vector_model_router_normalise(x, from_member, rows, cols);
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE, &pass0, &pass1,
                                  &final_word);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x, gamma, beta, 1.0f, 0.0f,
                                      from_program, rows, cols);
    ft_tt_test_check_fmt("router normalise",
                         ft_tt_vector_max_abs_difference(from_member, from_program, count) == 0.0,
                         "max abs %.6g",
                         ft_tt_vector_max_abs_difference(from_member, from_program, count));

done:
    free(x);
    free(gamma);
    free(beta);
    free(from_member);
    free(from_program);
}

/** The derivation itself: which member takes which pad, and what ceil(W/32) is. */
static void test_ragged_padding_contract(void)
{
    float value = 0.0f;
    int rc;

    ft_tt_test_step("a ragged W: the pad value each member's own passes ask for");

    ft_tt_test_check_fmt("Wt is ceil(W/32)",
                         ft_tt_op_reduce_padded_tile_cols(1500u) == 47u &&
                             ft_tt_op_reduce_padded_tile_cols(1536u) == 48u &&
                             ft_tt_op_reduce_padded_tile_cols(1u) == 1u &&
                             ft_tt_op_reduce_padded_tile_cols(0u) == 0u,
                         "1500 -> %u, 1536 -> %u", ft_tt_op_reduce_padded_tile_cols(1500u),
                         ft_tt_op_reduce_padded_tile_cols(1536u));

    /* Asserted as the PROPERTY the pad has to have — far below any activation, and finite so
     * the subtraction cannot overflow — rather than against the macro, which would make the
     * check agree with whatever the macro says (dev notes §3). */
    rc = ft_tt_op_reduce_pad_value(FT_TT_OP_REDUCE_FN_SOFTMAX, &value);
    ft_tt_test_check_fmt("softmax pads far below any activation, not with 0 — its sum would "
                         "pick up exp(0 - max) per pad lane",
                         rc == FT_TT_OK && value <= -1.0e20f && value > -HUGE_VALF, "rc %d, %g",
                         rc, (double)value);

    rc = ft_tt_op_reduce_pad_value(FT_TT_OP_REDUCE_FN_RMS_NORM, &value);
    ft_tt_test_check_fmt("RMS norm pads with 0 — a zero lane squares to zero",
                         rc == FT_TT_OK && value == 0.0f, "rc %d, %g", rc, (double)value);

    rc = ft_tt_op_reduce_pad_value(FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE, &value);
    ft_tt_test_check_fmt("the router pads with 0 — a zero lane adds nothing to the row sum",
                         rc == FT_TT_OK && value == 0.0f, "rc %d, %g", rc, (double)value);

    rc = ft_tt_op_reduce_pad_value(FT_TT_OP_REDUCE_FN_LAYER_NORM, &value);
    ft_tt_test_check_fmt("layer norm has NO constant pad value, and says so",
                         rc == FT_TT_EINVAL &&
                             !ft_tt_op_reduce_padding_is_exact(FT_TT_OP_REDUCE_FN_LAYER_NORM),
                         "rc %d", rc);
    ft_tt_test_check("the other three are exact under padding",
                     ft_tt_op_reduce_padding_is_exact(FT_TT_OP_REDUCE_FN_SOFTMAX) &&
                         ft_tt_op_reduce_padding_is_exact(FT_TT_OP_REDUCE_FN_RMS_NORM) &&
                         ft_tt_op_reduce_padding_is_exact(FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE),
                     NULL);

    {
        /* Six tiles: scaler, eps, and TWO tiles each of gamma and beta at Wt = 2. Sizing it
         * to what layer norm actually asks for is what keeps the refusals below about the
         * ragged W rather than about a short buffer — two grounds for one failure is a check
         * that tests whichever is cheaper (dev notes §3). */
        unsigned char scratch[6u * FT_TT_OP_TILE_BYTES_BF16];
        static float gamma[64];
        unsigned index;

        for (index = 0; index < 64u; index++)
            gamma[index] = 1.0f;
        rc = ft_tt_op_reduce_build_constants_ragged(FT_TT_OP_REDUCE_FN_LAYER_NORM, 33u, 1e-5f,
                                                    gamma, gamma, FT_TT_OP_TILE_BYTES_BF16,
                                                    scratch, sizeof(scratch));
        ft_tt_test_check_fmt("a RAGGED layer norm is refused rather than served wrong",
                             rc == FT_TT_EINVAL, "rc %d", rc);
        rc = ft_tt_op_reduce_build_constants_ragged(FT_TT_OP_REDUCE_FN_LAYER_NORM, 64u, 1e-5f,
                                                    gamma, gamma, FT_TT_OP_TILE_BYTES_BF16,
                                                    scratch, sizeof(scratch));
        ft_tt_test_check_fmt("an EXACT layer norm still builds — the refusal is about "
                             "raggedness, not the member",
                             rc == FT_TT_OK, "rc %d", rc);
    }
}

/** The divisor, checked as bytes: the scaler and eps tiles must carry the REAL W. This is
 *  the mutation PCC cannot see, so it is asserted directly and not only through a score. */
static void test_ragged_divisor_is_the_real_width(void)
{
    const uint32_t width = 1057u;
    const uint32_t tile_cols = ft_tt_op_reduce_padded_tile_cols(width);
    const uint32_t padded = tile_cols * FT_TT_OP_TILE_DIM;
    const float eps = 1e-5f;
    const float want_scaler = ft_float_from_bf16(ft_bf16_from_float(1.0f / (float)width));
    const float padded_scaler = ft_float_from_bf16(ft_bf16_from_float(1.0f / (float)padded));
    const float want_eps =
        ft_float_from_bf16(ft_bf16_from_float(eps * (float)width / (float)FT_TT_OP_TILE_DIM));
    const float padded_eps = ft_float_from_bf16(
        ft_bf16_from_float(eps * (float)padded / (float)FT_TT_OP_TILE_DIM));
    size_t bytes = ft_tt_op_reduce_constants_bytes(FT_TT_OP_REDUCE_FN_RMS_NORM, tile_cols,
                                                   FT_TT_OP_TILE_BYTES_BF16);
    unsigned char *constants = malloc(bytes);
    float *gamma = malloc(width * sizeof(float));
    unsigned wrong = 0u;
    uint32_t index;
    int rc;

    ft_tt_test_step("a ragged W: the reduce divisor stays the REAL W");
    if (!constants || !gamma) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }
    for (index = 0; index < width; index++)
        gamma[index] = 1.0f;

    /* Non-vacuity FIRST: at this W the two candidate divisors are different bf16 values, so
     * the checks below can tell them apart. At a W where they round together they could
     * not, and a green would mean nothing. */
    ft_tt_test_check_fmt("1/W and 1/padded-W are distinct bf16 values at this W",
                         want_scaler != padded_scaler && want_eps != padded_eps,
                         "%g vs %g", (double)want_scaler, (double)padded_scaler);

    rc = ft_tt_op_reduce_build_constants_ragged(FT_TT_OP_REDUCE_FN_RMS_NORM, width, eps, gamma,
                                                NULL, FT_TT_OP_TILE_BYTES_BF16, constants,
                                                bytes);
    ft_tt_test_check_fmt("it builds at W=1057 into Wt=34", rc == FT_TT_OK && tile_cols == 34u,
                         "rc %d, Wt %u", rc, tile_cols);
    if (rc != FT_TT_OK)
        goto done;

    ft_tt_test_check_fmt("the scaler tile is 1/1057, not 1/1088",
                         constant_tile_value(constants, &wrong) == want_scaler && wrong == 0u,
                         "%g", (double)constant_tile_value(constants, &wrong));
    ft_tt_test_check_fmt("the eps tile is eps*1057/32, not eps*1088/32",
                         constant_tile_value(constants + FT_TT_OP_TILE_BYTES_BF16, &wrong) ==
                                 want_eps &&
                             wrong == 0u,
                         "%g",
                         (double)constant_tile_value(constants + FT_TT_OP_TILE_BYTES_BF16,
                                                     &wrong));

    /* The gamma tail is the other thing a padded width invites: reading 31 floats past the
     * caller's array. It has to be a written zero, not whatever was next in memory. */
    {
        float *wire = malloc(padded * sizeof(float));
        unsigned bad = 0u;

        if (wire) {
            read_broadcast_row(constants + 2u * FT_TT_OP_TILE_BYTES_BF16, tile_cols, wire);
            for (index = 0; index < padded; index++) {
                const float want = (index < width) ? 1.0f : 0.0f;

                if (wire[index] != want)
                    bad++;
            }
            ft_tt_test_check_fmt("gamma's padded tail is a written zero, not read past the "
                                 "caller's array",
                                 bad == 0u, "%u of %u columns wrong", bad, padded);
            free(wire);
        } else {
            ft_tt_test_check("allocation", false, "out of memory");
        }
    }

done:
    free(constants);
    free(gamma);
}

/** The three paddable members, end to end, at three ragged widths. */
static void test_ragged_padded_pipeline(void)
{
    static const uint32_t widths[] = { 33u, 1057u, 1500u };
    static const ft_tt_op_reduce_function members[] = { FT_TT_OP_REDUCE_FN_SOFTMAX,
                                                        FT_TT_OP_REDUCE_FN_RMS_NORM,
                                                        FT_TT_OP_REDUCE_FN_ROUTER_NORMALISE };
    unsigned member;
    unsigned index;

    ft_tt_test_step("a ragged W: the padded pipeline against the true-W reference");
    for (member = 0; member < sizeof(members) / sizeof(members[0]); member++) {
        for (index = 0; index < sizeof(widths) / sizeof(widths[0]); index++)
            (void)score_padded_member(members[member], widths[index], 1e-5f,
                                      member * 8u + index);
    }
}

/**
 * WHY LAYER NORM IS REFUSED, MEASURED.
 *
 * Its variance pass reduces (x - row mean)^2, so a padded lane contributes (pad - mean)^2 —
 * zero only where the pad equals THAT ROW's mean. The rows here are given deliberately
 * different means (row r is offset by r), which is the input that makes the claim
 * discriminating: on zero-centred data the same error is ~1e-5 relative and would let a
 * broken padding through, and that is exactly the "vacuous because the input never
 * exercises it" trap in dev notes §3. Both regimes are reported.
 */
static void test_layer_norm_cannot_be_padded(void)
{
    static const float pads[] = { 0.0f, 8.0f, 16.0f, 31.0f, -1.0e30f };
    const uint32_t width = 1057u;
    const uint32_t tile_cols = ft_tt_op_reduce_padded_tile_cols(width);
    const uint32_t padded = tile_cols * FT_TT_OP_TILE_DIM;
    const uint32_t rows = 32u;
    const float eps = 1e-5f;
    const float scaler = ft_float_from_bf16(ft_bf16_from_float(1.0f / (float)width));
    const float eps_tile =
        ft_float_from_bf16(ft_bf16_from_float(eps * (float)width / (float)FT_TT_OP_TILE_DIM));
    float *x = malloc((size_t)rows * width * sizeof(float));
    float *x_padded = malloc((size_t)rows * padded * sizeof(float));
    float *gamma = malloc(padded * sizeof(float));
    float *beta = malloc(padded * sizeof(float));
    float *expected = malloc((size_t)rows * width * sizeof(float));
    float *modelled = malloc((size_t)rows * padded * sizeof(float));
    /* padded, not width: the exact-W control at the end scores a whole padded row through
     * this same buffer. */
    float *sliced = malloc((size_t)rows * padded * sizeof(float));
    uint32_t pass0 = 0u;
    uint32_t pass1 = 0u;
    uint32_t final_word = 0u;
    double best = 0.0;
    double floor_worst = 0.0;
    double broken = 0.0;
    unsigned failures_seen = 0u;
    unsigned arm;
    uint32_t row;
    uint32_t column;

    ft_tt_test_step("layer norm under padding: measured wrong for EVERY constant pad");
    if (!x || !x_padded || !gamma || !beta || !expected || !modelled || !sliced) {
        ft_tt_test_check("allocation", false, "out of memory");
        goto done;
    }
    ft_tt_vector_random_seed(0x1A4E70ULL);
    for (column = 0; column < padded; column++) {
        gamma[column] = 1.0f;
        beta[column] = 0.0f;
    }
    (void)ft_tt_op_reduce_program(FT_TT_OP_REDUCE_FN_LAYER_NORM, &pass0, &pass1, &final_word);

    /* THE FLOOR FIRST, AND IT IS NOT A GUESS. The same member, the same generator, the same
     * model, at an EXACT W where no padding happens at all — so the bar every arm below has
     * to clear is four times what this arrangement's bf16 rounding already costs, rather
     * than a constant somebody picked. On this deliberately offset data the floor is large
     * (the CENTER step subtracts two numbers near 31 in bf16), which is exactly why a fixed
     * threshold would have been wrong. */
    for (row = 0; row < rows; row++) {
        for (column = 0; column < padded; column++)
            x_padded[(size_t)row * padded + column] = ft_float_from_bf16(
                ft_bf16_from_float((float)row + ft_tt_vector_random_value()));
    }
    ft_tt_vector_reference_layer_norm(x_padded, gamma, beta, eps, modelled, rows, padded);
    ft_tt_vector_model_reduce_program(
        pass0, pass1, final_word, x_padded, gamma, beta,
        ft_float_from_bf16(ft_bf16_from_float(1.0f / (float)padded)),
        ft_float_from_bf16(ft_bf16_from_float(eps * (float)padded / (float)FT_TT_OP_TILE_DIM)),
        sliced, rows, padded);
    floor_worst = ft_tt_vector_max_abs_difference(modelled, sliced, (size_t)rows * padded);
    broken = 4.0 * floor_worst;
    ft_tt_test_check_fmt("CONTROL: at an EXACT W the same layer norm needs no padding and "
                         "lands on the reference — this is the floor the arms below clear",
                         floor_worst > 0.0 && floor_worst < 0.25, "max abs %.6g, so the bar "
                         "is %.6g", floor_worst, broken);

    for (column = 0; column < padded; column++)
        gamma[column] = (column < width) ? 1.0f : 0.0f;
    /* Row r is centred on r, so no single constant can sit near every row's mean. */
    for (row = 0; row < rows; row++) {
        for (column = 0; column < width; column++)
            x[(size_t)row * width + column] = ft_float_from_bf16(
                ft_bf16_from_float((float)row + ft_tt_vector_random_value()));
    }
    ft_tt_vector_reference_layer_norm(x, gamma, beta, eps, expected, rows, width);

    for (arm = 0; arm < sizeof(pads) / sizeof(pads[0]); arm++) {
        double worst;

        for (row = 0; row < rows; row++) {
            for (column = 0; column < padded; column++)
                x_padded[(size_t)row * padded + column] =
                    (column < width) ? x[(size_t)row * width + column] : pads[arm];
        }
        ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x_padded, gamma, beta,
                                          scaler, eps_tile, modelled, rows, padded);
        for (row = 0; row < rows; row++)
            memcpy(sliced + (size_t)row * width, modelled + (size_t)row * padded,
                   width * sizeof(float));
        worst = ft_tt_vector_max_abs_difference(expected, sliced, (size_t)rows * width);
        if (arm == 0u || worst < best)
            best = worst;
        if (worst > broken)
            failures_seen++;
        ft_tt_test_check_fmt("a constant pad does not save layer norm", worst > broken,
                             "pad %g: max abs %.6g against a floor of %.6g", (double)pads[arm],
                             worst, floor_worst);
    }
    ft_tt_test_check_fmt("the BEST constant pad is still wrong",
                         failures_seen == sizeof(pads) / sizeof(pads[0]) && best > broken,
                         "best of %u pads: max abs %.6g, %.1fx the floor", failures_seen, best,
                         best / floor_worst);

    /* AND A PER-ROW PAD DOES NOT RESCUE IT EITHER, which is the stronger form of the finding
     * and was not obvious until it was run. The row's own mean makes pass 1's (pad - mean)^2
     * vanish — and CORRUPTS pass 0, which is a plain SUM over the padded row with a 1/W_true
     * scaler, so the "mean" it produces is m*(padded/W). The two passes want different
     * padding of the SAME buffer. Nothing a caller writes into the input satisfies both. */
    {
        double worst;

        for (row = 0; row < rows; row++) {
            double mean = 0.0;

            for (column = 0; column < width; column++)
                mean += (double)x[(size_t)row * width + column];
            mean /= (double)width;
            for (column = 0; column < padded; column++)
                x_padded[(size_t)row * padded + column] =
                    (column < width) ? x[(size_t)row * width + column] : (float)mean;
        }
        ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x_padded, gamma, beta,
                                          scaler, eps_tile, modelled, rows, padded);
        for (row = 0; row < rows; row++)
            memcpy(sliced + (size_t)row * width, modelled + (size_t)row * padded,
                   width * sizeof(float));
        worst = ft_tt_vector_max_abs_difference(expected, sliced, (size_t)rows * width);
        ft_tt_test_check_fmt("nor does a PER-ROW pad — pass 0 wants 0 and pass 1 wants the "
                             "row mean, and there is one buffer",
                             worst > broken, "row-mean pad: max abs %.6g, %.1fx the floor",
                             worst, worst / floor_worst);
    }

    /* The honest nuance, reported rather than asserted: on a zero-centred row the same hole
     * is invisible, which is why "it scored fine on our data" is not evidence here. */
    for (row = 0; row < rows; row++) {
        for (column = 0; column < width; column++)
            x[(size_t)row * width + column] =
                ft_float_from_bf16(ft_bf16_from_float(ft_tt_vector_random_value()));
        for (column = 0; column < padded; column++)
            x_padded[(size_t)row * padded + column] =
                (column < width) ? x[(size_t)row * width + column] : 0.0f;
    }
    ft_tt_vector_reference_layer_norm(x, gamma, beta, eps, expected, rows, width);
    ft_tt_vector_model_reduce_program(pass0, pass1, final_word, x_padded, gamma, beta, scaler,
                                      eps_tile, modelled, rows, padded);
    for (row = 0; row < rows; row++)
        memcpy(sliced + (size_t)row * width, modelled + (size_t)row * padded,
               width * sizeof(float));
    ft_tt_test_check_fmt("REPORTED, not a gate: on zero-centred rows the same hole hides",
                         true, "zero pad, zero-mean data: max abs %.6g",
                         ft_tt_vector_max_abs_difference(expected, sliced, (size_t)rows * width));

done:
    free(x);
    free(x_padded);
    free(gamma);
    free(beta);
    free(expected);
    free(modelled);
    free(sliced);
}

/* ================================================================ */

int main(void)
{
    ft_tt_test_step("Phase 6 vector ops — OFF-HARDWARE gate (no card touched)");

    test_wire_constants_agree();
    test_arity();
    test_block_depth();
    test_eltwise_parameters();
    test_reduce_programs();
    test_reduce_parameters();
    test_eltwise_placement();
    test_reduce_placement();
    test_reduce_constants();
    test_reduce_constants_float32();
    test_eps_tile_follows_the_scaler();
    test_l2_norm_against_the_oracle();
    test_float32_reduce_at_the_head_shape();
    test_the_three_walls_priced_apart();
    test_gated_is_bound_by_the_bf16_bridge();
    test_eltwise_model();
    test_softmax_model();
    test_reduce_models();
    test_pcc_is_blind_to_scale();
    test_softmax_dynamic_range();
    test_softmax_max_is_over_the_whole_row();
    test_program_model_matches_the_member_models();
    test_ragged_padding_contract();
    test_ragged_divisor_is_the_real_width();
    test_ragged_padded_pipeline();
    test_layer_norm_cannot_be_padded();

    return ft_tt_test_finish();
}
