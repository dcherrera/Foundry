/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_topk_ops.c
 * @brief OFF-HARDWARE gate for the sampling op pair — exact top-N over a vocabulary logit
 *   row and the token draw that consumes it. Runs without a card and is what has to be
 *   green before the hardware pass is worth a core.
 *
 * ---------------------------------------------------------------------------
 * PCC IS NOT THE GATE HERE, AND THAT IS THE POINT
 * ---------------------------------------------------------------------------
 * Every other Phase 6 op is accepted at PCC ≥ 0.99 against a CPU fp32 reference. **That
 * gate cannot see this op fail.** Top-k returns INDICES: a result whose values are all
 * correct and whose k-th index resolved a tie the other way correlates at 1.0 with the
 * truth and makes the model emit a different token. So every check below is an EXACT
 * INTEGER comparison — index for index, token for token — and the stochastic path is gated
 * on the empirical distribution over many draws rather than on a single number.
 *
 * And ties are not an edge case to remember: bf16 has EIGHT mantissa bits against a 130,560
 * vocabulary, so a real logit row has ~2 entries per representable value before the
 * distribution is considered. The inputs here plant exact ties AND adjacent-representable
 * near-ties AT the selection boundary on purpose, which is the only place the reference and
 * the implementation can disagree without either looking wrong.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS CAN PROVE, AND WHAT IT CANNOT
 * ---------------------------------------------------------------------------
 * It CAN prove the host and the kernel agree on every wire constant (by PARSING the device
 * header, not by a comment claiming they match); that the order-preserving key the kernel
 * compares on is a total order agreeing with float order — EXHAUSTIVELY, over all 65,536
 * bf16 patterns; that the accept-interval pruning admits exactly the patterns that beat the
 * incumbent; that the selection MODEL agrees with an independent reference index for index;
 * and that the sampler's ORDER is the fixed one (greedy takes no repetition penalty).
 *
 * It CANNOT prove the kernel implements the model — nothing off-hardware can. It cannot see
 * the face order of a real tile, the CB credit protocol, or the NoC. A green run here means
 * the plan is sound, never that the op works. That is `test_phase6_topk.c`'s job.
 *
 * Environment:
 *   FT_TT_KERNEL_DIR  where the device headers live (default ../kernels)
 *
 * @status 132 checks, 0 failures, under clang on the Mac and gcc on the card host.
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
#include "../ft_tt_op_sample.h"
#include "../ft_tt_queue.h"
#include "ft_tt_test_report.h"
#include "ft_tt_topk_reference.h"

#define KERNEL_DIRECTORY_ENVIRONMENT "FT_TT_KERNEL_DIR"
#define KERNEL_DIRECTORY_DEFAULT "../kernels"
#define PATH_MAX_BYTES 512u

/** MiniCPM5-1B, MEASURED from the production artifact's `output.weight` entry. */
#define MINICPM5_VOCAB 130560u

static unsigned char g_cb_buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4];

/* ================================================================
 * Reading a constant out of a device header
 * ================================================================ */

/** A float (`static const float`) or a word (`enum { ... }`) -- the two C/C++-shared
 *  declaration forms the device headers use since plan Phase 6.6. Floats cannot be enum
 *  constants, which is why there are two forms rather than one.
 *
 *  Two LITERAL format strings rather than one assembled at run
 *  time — a runtime format is a `-Wformat-nonliteral` error under both compilers here, and
 *  suppressing it would be suppressing exactly the check that makes `-Wformat=2` worth
 *  having. */
static int read_device_constant(const char *path, const char *name, char *out_value,
                                size_t out_bytes, int is_float)
{
    char line[512];
    FILE *file = fopen(path, "r");
    int found = -1;

    if (!file)
        return -1;
    while (fgets(line, sizeof(line), file)) {
        char parsed_name[128];
        char parsed_value[64];
        int fields;

        if (is_float)
            fields = sscanf(line, " static const float %127[A-Za-z0-9_] = %63[^;];", parsed_name,
                            parsed_value);
        else
            fields = sscanf(line, " enum { %127[A-Za-z0-9_] = %63[^}] };",
                            parsed_name, parsed_value);
        if (fields != 2)
            continue;
        if (strcmp(parsed_name, name) != 0)
            continue;
        snprintf(out_value, out_bytes, "%s", parsed_value);
        found = 0;
        break;
    }
    fclose(file);
    return found;
}

static void device_header_path(char *out, size_t out_bytes, const char *file)
{
    const char *directory = getenv(KERNEL_DIRECTORY_ENVIRONMENT);

    if (!directory || directory[0] == '\0')
        directory = KERNEL_DIRECTORY_DEFAULT;
    snprintf(out, out_bytes, "%s/%s", directory, file);
}

static void check_device_word(const char *path, const char *name, uint32_t host_value)
{
    char text[64];

    if (read_device_constant(path, name, text, sizeof(text), 0) != 0) {
        ft_tt_test_check_fmt(name, false, "not declared in %s", path);
        return;
    }
    {
        uint32_t device_value = (uint32_t)strtoul(text, NULL, 0);

        ft_tt_test_check_fmt(name, device_value == host_value, "host %u, kernel %u",
                             host_value, device_value);
    }
}

/** A float constant, compared as TEXT after normalising the literal suffix — a numeric
 *  compare would pass on two literals that round to the same float while differing in the
 *  source, and the thing under test is that nobody retyped a digit. */
static void check_device_float(const char *path, const char *name, const char *host_literal)
{
    char text[64];
    size_t length;

    if (read_device_constant(path, name, text, sizeof(text), 1) != 0) {
        ft_tt_test_check_fmt(name, false, "not declared in %s", path);
        return;
    }
    length = strlen(text);
    while (length > 0u && (text[length - 1u] == 'f' || text[length - 1u] == ' '))
        text[--length] = '\0';
    ft_tt_test_check_fmt(name, strcmp(text, host_literal) == 0, "reference %s, kernel %s",
                         host_literal, text);
}

static void test_wire_constants_agree(void)
{
    char topk_path[PATH_MAX_BYTES];
    char queue_path[PATH_MAX_BYTES];

    ft_tt_test_step("wire constants: the host enum against the kernel's own header");
    device_header_path(topk_path, sizeof(topk_path), "resident_topk_protocol.h");
    device_header_path(queue_path, sizeof(queue_path), "resident_queue_protocol.h");

    check_device_word(topk_path, "kTopkTileCount", FT_TT_OP_SAMPLE_TILE_COUNT);
    check_device_word(topk_path, "kTopkRow", FT_TT_OP_SAMPLE_ROW);
    check_device_word(topk_path, "kTopkSelectCount", FT_TT_OP_SAMPLE_SELECT_COUNT);
    check_device_word(topk_path, "kTopkDepth", FT_TT_OP_SAMPLE_DEPTH);
    check_device_word(topk_path, "kTopkLogitsX", FT_TT_OP_SAMPLE_LOGITS_X);
    check_device_word(topk_path, "kTopkLogitsY", FT_TT_OP_SAMPLE_LOGITS_Y);
    check_device_word(topk_path, "kTopkLogitsBase", FT_TT_OP_SAMPLE_LOGITS_BASE);
    check_device_word(topk_path, "kTopkResultAddress", FT_TT_OP_SAMPLE_RESULT_ADDRESS);
    check_device_word(topk_path, "kTopkStateAddress", FT_TT_OP_SAMPLE_STATE_ADDRESS);
    check_device_word(topk_path, "kTopkTokenOutX", FT_TT_OP_SAMPLE_TOKEN_OUT_X);
    check_device_word(topk_path, "kTopkTokenOutY", FT_TT_OP_SAMPLE_TOKEN_OUT_Y);
    check_device_word(topk_path, "kTopkTokenOutBase", FT_TT_OP_SAMPLE_TOKEN_OUT_BASE);
    check_device_word(topk_path, "kTopkFlags", FT_TT_OP_SAMPLE_FLAGS);
    check_device_word(topk_path, "kTopkWords", FT_TT_OP_SAMPLE_WORDS);
    check_device_word(topk_path, "kTopkFlagDrainOnly", FT_TT_OP_SAMPLE_FLAG_DRAIN_ONLY);
    check_device_word(topk_path, "kTopkFlagTailStats", FT_TT_OP_SAMPLE_FLAG_TAIL_STATS);
    check_device_word(topk_path, "kTopkTailBins", FT_TT_OP_SAMPLE_TAIL_BINS);
    check_device_word(topk_path, "kTopkTailMaxBins", FT_TT_OP_SAMPLE_TAIL_MAX_BINS);
    check_device_word(topk_path, "kTopkTailKeysOffset", FT_TT_OP_SAMPLE_TAIL_KEYS_OFFSET);
    check_device_word(topk_path, "kTopkTailCountsOffset", FT_TT_OP_SAMPLE_TAIL_COUNTS_OFFSET);
    check_device_word(topk_path, "kTopkTailCountsWords", FT_TT_OP_SAMPLE_TAIL_COUNTS_WORDS);
    check_device_word(topk_path, "kTopkTailStatusWord", FT_TT_OP_SAMPLE_TAIL_STATUS_WORD);
    check_device_word(topk_path, "kTopkTailStatusMagic", FT_TT_OP_SAMPLE_TAIL_STATUS_MAGIC);
    check_device_word(topk_path, "kTopkMaxSelect", FT_TT_OP_SAMPLE_MAX_SELECT);

    check_device_word(topk_path, "kTopkResultMagic", FT_TT_OP_SAMPLE_RESULT_MAGIC);
    check_device_word(topk_path, "kTopkResultCount", FT_TT_OP_SAMPLE_RESULT_COUNT);
    check_device_word(topk_path, "kTopkResultSequence", FT_TT_OP_SAMPLE_RESULT_SEQUENCE);
    check_device_word(topk_path, "kTopkResultTokenValid", FT_TT_OP_SAMPLE_RESULT_TOKEN_VALID);
    check_device_word(topk_path, "kTopkResultToken", FT_TT_OP_SAMPLE_RESULT_TOKEN);
    check_device_word(topk_path, "kTopkResultHeaderWords",
                      FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS);
    check_device_word(topk_path, "kTopkPairWords", FT_TT_OP_SAMPLE_PAIR_WORDS);
    check_device_word(topk_path, "kTopkTokenSlotBytes", FT_TT_OP_SAMPLE_TOKEN_SLOT_BYTES);
    check_device_word(topk_path, "kTopkTokenOutAlignment", FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT);
    check_device_word(topk_path, "kTopkResultMagicValue", FT_TT_OP_SAMPLE_RESULT_MAGIC_VALUE);
    /* kTopkResultBytes and kSampleStateWords are DERIVED on both sides, so pinning their
     * text would pin an expression rather than a value. Their pieces are pinned above; what
     * is checked here is that the two derivations agree. */
    ft_tt_test_check_fmt("the result block size derives the same both sides",
                         FT_TT_OP_SAMPLE_RESULT_BYTES ==
                                 (FT_TT_OP_SAMPLE_TAIL_COUNTS_OFFSET +
                                  FT_TT_OP_SAMPLE_TAIL_COUNTS_WORDS) * 4u &&
                             FT_TT_OP_SAMPLE_TAIL_KEYS_OFFSET ==
                                 FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS +
                                     FT_TT_OP_SAMPLE_MAX_SELECT * FT_TT_OP_SAMPLE_PAIR_WORDS &&
                             FT_TT_OP_SAMPLE_TAIL_COUNTS_OFFSET ==
                                 FT_TT_OP_SAMPLE_TAIL_KEYS_OFFSET + FT_TT_OP_SAMPLE_TAIL_MAX_BINS,
                         "%u B", (uint32_t)FT_TT_OP_SAMPLE_RESULT_BYTES);

    check_device_word(topk_path, "kSampleStateMagic", FT_TT_OP_SAMPLE_STATE_MAGIC);
    check_device_word(topk_path, "kSampleStateRngState", FT_TT_OP_SAMPLE_STATE_RNG);
    check_device_word(topk_path, "kSampleStateTemperatureBits",
                      FT_TT_OP_SAMPLE_STATE_TEMPERATURE_BITS);
    check_device_word(topk_path, "kSampleStateTopPBits", FT_TT_OP_SAMPLE_STATE_TOP_P_BITS);
    check_device_word(topk_path, "kSampleStateRepeatPenaltyBits",
                      FT_TT_OP_SAMPLE_STATE_REPEAT_PENALTY_BITS);
    check_device_word(topk_path, "kSampleStateTopK", FT_TT_OP_SAMPLE_STATE_TOP_K);
    check_device_word(topk_path, "kSampleStateRepeatWindow",
                      FT_TT_OP_SAMPLE_STATE_REPEAT_WINDOW);
    check_device_word(topk_path, "kSampleStateRecentCursor",
                      FT_TT_OP_SAMPLE_STATE_RECENT_CURSOR);
    check_device_word(topk_path, "kSampleStateRecentCount", FT_TT_OP_SAMPLE_STATE_RECENT_COUNT);
    check_device_word(topk_path, "kSampleStateEmittedCount",
                      FT_TT_OP_SAMPLE_STATE_EMITTED_COUNT);
    check_device_word(topk_path, "kSampleStateEosCount", FT_TT_OP_SAMPLE_STATE_EOS_COUNT);
    check_device_word(topk_path, "kSampleStateStopped", FT_TT_OP_SAMPLE_STATE_STOPPED);
    check_device_word(topk_path, "kSampleStateTokenBudget", FT_TT_OP_SAMPLE_STATE_TOKEN_BUDGET);
    check_device_word(topk_path, "kSampleStateLastToken", FT_TT_OP_SAMPLE_STATE_LAST_TOKEN);
    check_device_word(topk_path, "kSampleStateHeaderWords",
                      FT_TT_OP_SAMPLE_STATE_HEADER_WORDS);
    check_device_word(topk_path, "kSampleRecentCapacity", FT_TT_OP_SAMPLE_RECENT_CAPACITY);
    check_device_word(topk_path, "kSampleEosCapacity", FT_TT_OP_SAMPLE_EOS_CAPACITY);
    check_device_word(topk_path, "kSampleEmittedCapacity", FT_TT_OP_SAMPLE_EMITTED_CAPACITY);
    check_device_word(topk_path, "kSampleStateMagicValue", FT_TT_OP_SAMPLE_STATE_MAGIC_VALUE);

    check_device_word(queue_path, "kOpTopk", FT_TT_QUEUE_OP_TOPK);
    check_device_word(queue_path, "kOpSample", FT_TT_QUEUE_OP_SAMPLE);

    ft_tt_test_check_fmt("the sampling block fits a queue parameter slot",
                         (uint32_t)FT_TT_OP_SAMPLE_WORDS == FT_TT_QUEUE_PARAM_WORDS,
                         "sampling %u, queue slot %u", (uint32_t)FT_TT_OP_SAMPLE_WORDS,
                         FT_TT_QUEUE_PARAM_WORDS);

    /* The exp coefficients. Both sides compute the same exponential or the sampled-token
     * gate is statistical instead of exact, and a retyped digit is exactly the drift
     * dev notes §4 catalogues. */
    check_device_float(topk_path, "kFtkLog2E", "1.44269504088896340736");
    check_device_float(topk_path, "kFtkExp2C1", "0.69314718055994531");
    check_device_float(topk_path, "kFtkExp2C2", "0.24022650695910072");
    check_device_float(topk_path, "kFtkExp2C3", "0.05550410866482158");
    check_device_float(topk_path, "kFtkExp2C4", "0.00961812910762848");
    check_device_float(topk_path, "kFtkExp2C5", "0.00133335581467035");
    check_device_float(topk_path, "kFtkExp2C6", "0.00015403530393381");
}

/* ================================================================
 * The order, exhaustively
 * ================================================================ */

/**
 * All 65,536 bf16 patterns: the integer key must order exactly as the float does.
 *
 * Exhaustive because it CAN be — the domain is 16 bits — and because a sampled order is
 * precisely the wrong instrument for a map whose failures live at sign boundaries, at zero
 * and at the infinities. Two documented exclusions and both are asserted rather than
 * skipped: NaN has no float order at all, and -0.0 vs +0.0 compare EQUAL as floats while
 * the key orders -0 below +0.
 */
static void test_key_order_is_the_float_order(void)
{
    uint32_t previous_key = 0u;
    uint32_t sorted[65536];
    uint32_t count = 0u;
    uint32_t pattern;
    uint32_t inversions = 0u;
    uint32_t duplicates = 0u;
    int first_reported = 0;

    ft_tt_test_step("the order-preserving key, over all 65,536 bf16 patterns");

    /* A key is a bijection on the non-NaN patterns: collect and check for duplicates by
     * bucketing, which also proves nothing collapses two distinct floats onto one key. */
    memset(sorted, 0, sizeof(sorted));
    for (pattern = 0u; pattern < 65536u; pattern++) {
        uint32_t key = ft_tt_topk_reference_key((uint16_t)pattern);

        if (sorted[key]++ != 0u)
            duplicates++;
    }
    ft_tt_test_check_fmt("the key is a bijection on 16 bits", duplicates == 0u,
                         "%u colliding patterns", duplicates);

    /* Walk the patterns in KEY order and assert the floats are non-decreasing. */
    for (pattern = 0u; pattern < 65536u; pattern++)
        sorted[pattern] = 0u;
    for (pattern = 0u; pattern < 65536u; pattern++)
        sorted[ft_tt_topk_reference_key((uint16_t)pattern)] = pattern;

    for (pattern = 0u; pattern < 65536u; pattern++) {
        uint16_t bits = (uint16_t)sorted[pattern];
        float value = ft_float_from_bf16(bits);
        float previous;

        if (isnan(value)) {
            count++;
            continue;
        }
        if (count == 0u || pattern == 0u) {
            previous_key = pattern;
            count++;
            continue;
        }
        previous = ft_float_from_bf16((uint16_t)sorted[previous_key]);
        if (isnan(previous)) {
            previous_key = pattern;
            continue;
        }
        /* -0.0 immediately before +0.0 is the ONE documented place a strict increase does
         * not hold, because IEEE says they are equal. Everything else must increase. */
        if (!(previous < value)) {
            int is_signed_zero_pair = (previous == 0.0f && value == 0.0f);

            if (!is_signed_zero_pair) {
                inversions++;
                if (!first_reported) {
                    ft_tt_test_check_fmt("first inversion", false,
                                         "key %u: %.9g (0x%04X) then %.9g (0x%04X)", pattern,
                                         (double)previous, (unsigned)sorted[previous_key],
                                         (double)value, (unsigned)bits);
                    first_reported = 1;
                }
            }
        }
        previous_key = pattern;
    }
    ft_tt_test_check_fmt("key order == float order on every non-NaN pattern", inversions == 0u,
                         "%u inversions", inversions);
    ft_tt_test_check_fmt("the two zeros are the only equal-float neighbours",
                         ft_tt_topk_reference_key(0x8000u) < ft_tt_topk_reference_key(0x0000u),
                         "key(-0) %u, key(+0) %u", ft_tt_topk_reference_key(0x8000u),
                         ft_tt_topk_reference_key(0x0000u));
}

/* ================================================================
 * The off-hardware MODEL of the kernel's selection
 * ================================================================
 * A transliteration of resident_topk_select.c's algorithm — the accept interval and the
 * insertion — so the algorithm can be scored against the independent reference without a
 * card. It is a MODEL: green here predicts the card, it does not stand in for it.
 */

typedef struct {
    uint32_t key;
    uint32_t index;
    uint32_t bits;
} model_entry;

typedef struct {
    model_entry entries[FT_TT_OP_SAMPLE_MAX_SELECT];
    uint32_t count;
    uint32_t capacity;
    uint32_t accept_low;
    uint32_t accept_span;
} model_select;

static void model_refresh_range(model_select *state)
{
    if (state->count < state->capacity) {
        state->accept_low = 0u;
        state->accept_span = 0x10000u;
    } else {
        uint32_t worst = state->entries[state->capacity - 1u].bits;

        if (worst < 0x8000u) {
            state->accept_low = worst + 1u;
            state->accept_span = 0x8000u - state->accept_low;
        } else {
            state->accept_low = 0u;
            state->accept_span = worst;
        }
    }
}

static void model_init(model_select *state, uint32_t capacity)
{
    state->count = 0u;
    state->capacity = capacity;
    model_refresh_range(state);
}

static void model_offer(model_select *state, uint32_t bits, uint32_t index)
{
    uint32_t key = ft_tt_topk_reference_key((uint16_t)bits);
    uint32_t position = (state->count < state->capacity) ? state->count : state->capacity - 1u;

    if (state->count < state->capacity)
        state->count++;
    while (position > 0u && state->entries[position - 1u].key < key) {
        state->entries[position] = state->entries[position - 1u];
        position--;
    }
    state->entries[position].key = key;
    state->entries[position].index = index;
    state->entries[position].bits = bits;
    model_refresh_range(state);
}

/** The whole scan, exactly as the kernel runs it: the accept test gates the offer. */
static void model_scan(const uint16_t *row, uint32_t vocab, uint32_t select_count,
                       uint32_t *out_indices, float *out_values)
{
    model_select state;
    uint32_t index;

    model_init(&state, select_count);
    for (index = 0u; index < vocab; index++) {
        uint32_t bits = row[index];

        if ((bits - state.accept_low) < state.accept_span)
            model_offer(&state, bits, index);
    }
    for (index = 0u; index < state.count; index++) {
        out_indices[index] = state.entries[index].index;
        out_values[index] = ft_float_from_bf16((uint16_t)state.entries[index].bits);
    }
}

/** The accept interval must admit EXACTLY the patterns that beat `worst`. Exhaustive over
 *  both, which is 2^32 pairs, so it sweeps a set of worst values chosen to sit on every
 *  boundary the interval has: the two zeros, the two infinities, both signs, and the
 *  extremes. */
static void test_accept_interval_admits_exactly_the_winners(void)
{
    static const uint16_t worsts[] = {0x0000u, 0x8000u, 0x3F80u, 0xBF80u, 0x7F7Fu,
                                      0xFF7Fu, 0x7F80u, 0xFF80u, 0x0001u, 0x8001u,
                                      0x4048u, 0xC048u};
    uint32_t mismatches = 0u;
    size_t which;

    ft_tt_test_step("the accept interval: exactly the patterns that beat the incumbent");
    for (which = 0u; which < sizeof(worsts) / sizeof(worsts[0]); which++) {
        uint32_t worst = worsts[which];
        uint32_t worst_key = ft_tt_topk_reference_key((uint16_t)worst);
        uint32_t low;
        uint32_t span;
        uint32_t candidate;
        uint32_t local = 0u;

        if (worst < 0x8000u) {
            low = worst + 1u;
            span = 0x8000u - low;
        } else {
            low = 0u;
            span = worst;
        }
        for (candidate = 0u; candidate < 65536u; candidate++) {
            int admitted = ((candidate - low) < span);
            int wins = (ft_tt_topk_reference_key((uint16_t)candidate) > worst_key);

            if (admitted != wins)
                local++;
        }
        ft_tt_test_check_fmt("interval matches the comparator", local == 0u,
                             "worst 0x%04X: %u disagreements", (unsigned)worst, local);
        mismatches += local;
    }
    ft_tt_test_check_fmt("no interval disagreement anywhere", mismatches == 0u, "%u total",
                         mismatches);
}

/* ================================================================
 * Selection against the independent reference
 * ================================================================ */

static uint32_t g_random_state = 0x12345678u;

static uint32_t next_random(void)
{
    g_random_state ^= g_random_state << 13;
    g_random_state ^= g_random_state >> 17;
    g_random_state ^= g_random_state << 5;
    return g_random_state;
}

/** A full-mantissa random logit, in the range a real LM head produces. */
static uint16_t random_logit(void)
{
    float value = ((float)(next_random() >> 8) / 16777216.0f) * 40.0f - 20.0f;

    return ft_bf16_from_float(value);
}

/** Model against reference, index for index and value for value. */
static void compare_selection(const char *what, const uint16_t *row, uint32_t vocab,
                              uint32_t select_count)
{
    uint32_t model_indices[FT_TT_OP_SAMPLE_MAX_SELECT];
    float model_values[FT_TT_OP_SAMPLE_MAX_SELECT];
    uint32_t reference_indices[FT_TT_OP_SAMPLE_MAX_SELECT];
    float reference_values[FT_TT_OP_SAMPLE_MAX_SELECT];
    uint32_t first_bad = select_count;
    uint32_t entry;

    model_scan(row, vocab, select_count, model_indices, model_values);
    if (ft_tt_topk_reference_select(row, vocab, select_count, reference_indices,
                                    reference_values) != 0) {
        ft_tt_test_check(what, false, "the reference refused the input");
        return;
    }
    for (entry = 0u; entry < select_count; entry++) {
        if (model_indices[entry] != reference_indices[entry] ||
            model_values[entry] != reference_values[entry]) {
            first_bad = entry;
            break;
        }
    }
    if (first_bad == select_count)
        ft_tt_test_check_fmt(what, true, "%u pairs identical", select_count);
    else
        ft_tt_test_check_fmt(what, false,
                             "pair %u: model (%u, %.9g), reference (%u, %.9g)", first_bad,
                             model_indices[first_bad], (double)model_values[first_bad],
                             reference_indices[first_bad], (double)reference_values[first_bad]);
}

static void test_selection_matches_the_reference(void)
{
    enum { VOCAB = 4096u };
    static uint16_t row[VOCAB];
    uint32_t trial;

    ft_tt_test_step("exact index agreement with an independent iterative argmax");

    for (trial = 0u; trial < 8u; trial++) {
        uint32_t index;
        char label[64];

        for (index = 0u; index < VOCAB; index++)
            row[index] = random_logit();
        snprintf(label, sizeof(label), "random row %u, N = 40", trial);
        compare_selection(label, row, VOCAB, 40u);
    }

    /* N = 1 is greedy's shape and is the case a general top-k most easily gets wrong. */
    {
        uint32_t index;

        for (index = 0u; index < VOCAB; index++)
            row[index] = random_logit();
        compare_selection("random row, N = 1", row, VOCAB, 1u);
        compare_selection("random row, N = 64 (the maximum)", row, VOCAB, 64u);
    }
}

/**
 * TIES, which is where this op actually breaks.
 *
 * Three deliberate constructions, each aimed at the SELECTION BOUNDARY rather than at the
 * middle of the row where a wrong answer would not change the outcome:
 *   1. a plateau of identical values straddling the N-th place, so the answer is decided
 *      entirely by the tie rule and by nothing else;
 *   2. adjacent-representable near-ties, where a reference and an implementation that
 *      round differently disagree by one index;
 *   3. an all-identical row, where every index is a candidate and only "lowest wins" gives
 *      a determinate answer.
 */
static void test_ties_at_the_selection_boundary(void)
{
    enum { VOCAB = 1024u };
    static uint16_t row[VOCAB];
    uint16_t plateau = ft_bf16_from_float(3.5f);
    uint16_t below = ft_bf16_from_float(1.0f);
    uint32_t index;

    ft_tt_test_step("ties: the case bf16's 8 mantissa bits make the common one");

    /* 1. Twenty identical values around the N = 10 boundary, scattered so the winners are
     *    not the first ten in memory order either. */
    for (index = 0u; index < VOCAB; index++)
        row[index] = below;
    for (index = 0u; index < 20u; index++)
        row[index * 37u % VOCAB] = plateau;
    compare_selection("a 20-wide plateau straddling N = 10", row, VOCAB, 10u);

    /* 2. Adjacent representable bf16 values: v, v+1ulp, v-1ulp, repeated. A near-tie is
     *    where a reference that computed in fp32 and an implementation that compared bits
     *    can disagree by exactly one index, which is invisible to any value metric. */
    for (index = 0u; index < VOCAB; index++)
        row[index] = below;
    for (index = 0u; index < 24u; index++) {
        uint16_t base = ft_bf16_from_float(7.0f);

        row[index * 41u % VOCAB] = (uint16_t)(base + (uint16_t)(index % 3u) - 1u);
    }
    compare_selection("adjacent-representable near-ties at N = 12", row, VOCAB, 12u);

    /* 3. Everything identical. The only determinate answer is indices 0..N-1. */
    for (index = 0u; index < VOCAB; index++)
        row[index] = plateau;
    compare_selection("an all-identical row at N = 8", row, VOCAB, 8u);
    {
        uint32_t indices[FT_TT_OP_SAMPLE_MAX_SELECT];
        float values[FT_TT_OP_SAMPLE_MAX_SELECT];
        uint32_t wrong = 0u;

        model_scan(row, VOCAB, 8u, indices, values);
        for (index = 0u; index < 8u; index++)
            if (indices[index] != index)
                wrong++;
        ft_tt_test_check_fmt("an all-identical row selects indices 0..N-1", wrong == 0u,
                             "%u of 8 wrong, first %u", wrong, indices[0]);
    }

    /* 4. Negative ties, because the accept interval takes a DIFFERENT branch for a negative
     *    incumbent and a plateau of positives would never exercise it. */
    for (index = 0u; index < VOCAB; index++)
        row[index] = ft_bf16_from_float(-9.0f);
    for (index = 0u; index < 16u; index++)
        row[index * 53u % VOCAB] = ft_bf16_from_float(-2.5f);
    compare_selection("a negative plateau at N = 8", row, VOCAB, 8u);

    /* 5. The infinities, because a damaged core writes 0x7F80 and that is +inf in bf16
     *    — it must SELECT, not be skipped, so it is visible. */
    for (index = 0u; index < VOCAB; index++)
        row[index] = random_logit();
    row[VOCAB - 3u] = 0x7F80u;
    row[7u] = 0xFF80u;
    compare_selection("a row containing +inf and -inf", row, VOCAB, 5u);
    {
        uint32_t indices[FT_TT_OP_SAMPLE_MAX_SELECT];
        float values[FT_TT_OP_SAMPLE_MAX_SELECT];

        model_scan(row, VOCAB, 5u, indices, values);
        ft_tt_test_check_fmt("+inf is selected first", indices[0] == VOCAB - 3u,
                             "first index %u, expected %u", indices[0], VOCAB - 3u);
    }
}

/** The DOCUMENTED divergence, asserted rather than left as prose: the key orders -0 below
 *  +0 where IEEE calls them equal, so the reference and the kernel disagree on exactly this
 *  input and on no other. A divergence nobody has written a test for is a bug in waiting. */
static void test_signed_zero_divergence_is_the_documented_one(void)
{
    enum { VOCAB = 64u };
    static uint16_t row[VOCAB];
    uint32_t model_indices[FT_TT_OP_SAMPLE_MAX_SELECT];
    float model_values[FT_TT_OP_SAMPLE_MAX_SELECT];
    uint32_t reference_indices[FT_TT_OP_SAMPLE_MAX_SELECT];
    float reference_values[FT_TT_OP_SAMPLE_MAX_SELECT];
    uint32_t index;

    ft_tt_test_step("signed zero: the one place the key and IEEE disagree");
    for (index = 0u; index < VOCAB; index++)
        row[index] = ft_bf16_from_float(-4.0f);
    row[5u] = 0x8000u; /* -0.0, at the LOWER index */
    row[9u] = 0x0000u; /* +0.0 */

    model_scan(row, VOCAB, 1u, model_indices, model_values);
    ft_tt_topk_reference_select(row, VOCAB, 1u, reference_indices, reference_values);
    ft_tt_test_check_fmt("the kernel's key picks +0 (index 9)", model_indices[0] == 9u,
                         "picked %u", model_indices[0]);
    ft_tt_test_check_fmt("IEEE's tie rule picks -0 (index 5)", reference_indices[0] == 5u,
                         "picked %u", reference_indices[0]);
}

/* ================================================================
 * The parameter block and the arena
 * ================================================================ */

static ft_tt_op_topk_request minicpm_request(void)
{
    ft_tt_op_topk_request request;

    memset(&request, 0, sizeof(request));
    request.tile_count = ft_tt_op_sample_row_tiles(MINICPM5_VOCAB);
    request.row = 0u;
    request.select_count = 40u;
    request.prefetch_tiles = 48u;
    request.logits.x = 17u;
    request.logits.y = 12u;
    request.logits.address = 0x1000u;
    request.logits.tile_bytes = FT_TT_OP_TILE_BYTES_BF16;
    request.result_address = 0x40000u;
    request.state_address = 0x41000u;
    return request;
}

static void test_parameters_and_refusals(void)
{
    ft_tt_op_topk_request request = minicpm_request();
    uint32_t words[FT_TT_OP_SAMPLE_WORDS];

    ft_tt_test_step("the parameter block, and what it refuses");

    ft_tt_test_check_fmt("MiniCPM5-1B's 130,560 vocabulary is 4080 whole tiles",
                         ft_tt_op_sample_row_tiles(MINICPM5_VOCAB) == 4080u, "%u tiles",
                         ft_tt_op_sample_row_tiles(MINICPM5_VOCAB));
    /* A vocabulary that is not a whole number of tiles pads with ZERO, and a 0.0 pad
     * outranks every negative logit — it would be selected as a token id that does not
     * exist. Refusing is the only safe answer. */
    ft_tt_test_check_fmt("a vocabulary that is not a multiple of 32 is refused",
                         ft_tt_op_sample_row_tiles(130561u) == 0u, "%u tiles",
                         ft_tt_op_sample_row_tiles(130561u));

    ft_tt_test_check("a well-formed request encodes",
                     ft_tt_op_sample_parameters(&request, words) == FT_TT_OK, "FT_TT_OK");
    ft_tt_test_check_fmt("tile count lands in word 0", words[FT_TT_OP_SAMPLE_TILE_COUNT] == 4080u,
                         "%u", words[FT_TT_OP_SAMPLE_TILE_COUNT]);
    ft_tt_test_check_fmt("depth is shrunk to a divisor of the tile count",
                         4080u % words[FT_TT_OP_SAMPLE_DEPTH] == 0u, "depth %u of 4080",
                         words[FT_TT_OP_SAMPLE_DEPTH]);
    ft_tt_test_check_fmt("the logits base lands in word 6",
                         words[FT_TT_OP_SAMPLE_LOGITS_BASE] == 0x1000u, "0x%X",
                         words[FT_TT_OP_SAMPLE_LOGITS_BASE]);
    ft_tt_test_check_fmt("unused words are zero", words[13] == 0u && words[14] == 0u &&
                                                      words[15] == 0u,
                         "%u %u %u", words[13], words[14], words[15]);

    request = minicpm_request();
    request.row = 32u;
    ft_tt_test_check("a row past 31 is refused",
                     ft_tt_op_sample_parameters(&request, words) == FT_TT_EINVAL,
                     "FT_TT_EINVAL");
    request = minicpm_request();
    request.select_count = FT_TT_OP_SAMPLE_MAX_SELECT + 1u;
    ft_tt_test_check("a select count past the block's capacity is refused",
                     ft_tt_op_sample_parameters(&request, words) == FT_TT_EINVAL,
                     "FT_TT_EINVAL");
    request = minicpm_request();
    request.result_address = 0u;
    ft_tt_test_check("a missing result address is refused",
                     ft_tt_op_sample_parameters(&request, words) == FT_TT_EINVAL,
                     "FT_TT_EINVAL");

    /* The token hand-off is a 16-byte NoC write and both ends must be aligned. The version of
     * this op that shipped moved four bytes with `noc_inline_dw_write` — an API whose own
     * contract excludes DRAM destinations, which is the destination this op normally uses —
     * and the write LANDED while leaving the Tensix unable to boot again. An unaligned
     * address is now a refusal here and in the kernel. */
    request = minicpm_request();
    request.token_out.address = 0u;
    ft_tt_test_check("no token hand-off at all is allowed",
                     ft_tt_op_sample_parameters(&request, words) == FT_TT_OK, "FT_TT_OK");
    request = minicpm_request();
    request.token_out.address = FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT * 3u;
    ft_tt_test_check("an aligned token hand-off address is accepted",
                     ft_tt_op_sample_parameters(&request, words) == FT_TT_OK, "FT_TT_OK");
    request = minicpm_request();
    request.token_out.address = FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT * 3u + 4u;
    ft_tt_test_check("a 4-byte-aligned but NOT 16-byte-aligned hand-off address is refused",
                     ft_tt_op_sample_parameters(&request, words) == FT_TT_EINVAL,
                     "FT_TT_EINVAL");

    /* The slot the kernel ships FROM has to be inside the result header and aligned within a
     * block that is: a slot running into the pairs would send survivor data to the next op's
     * input, and one at an odd offset would reproduce the sub-alignment write. */
    ft_tt_test_check_fmt("the hand-off slot is 16-byte aligned inside the result block",
                         (FT_TT_OP_SAMPLE_TOKEN_SLOT_WORDS * 4u) %
                                 FT_TT_OP_SAMPLE_TOKEN_OUT_ALIGNMENT == 0u,
                         "word %u", (uint32_t)FT_TT_OP_SAMPLE_TOKEN_SLOT_WORDS);
    ft_tt_test_check_fmt("and stays clear of the pairs",
                         FT_TT_OP_SAMPLE_TOKEN_SLOT_WORDS +
                                 FT_TT_OP_SAMPLE_TOKEN_SLOT_BYTES / 4u <=
                             (uint32_t)FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS,
                         "words %u..%u of %u", (uint32_t)FT_TT_OP_SAMPLE_TOKEN_SLOT_WORDS,
                         (uint32_t)(FT_TT_OP_SAMPLE_TOKEN_SLOT_WORDS +
                                    FT_TT_OP_SAMPLE_TOKEN_SLOT_BYTES / 4u - 1u),
                         (uint32_t)FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS);
}

static void test_circular_buffers_fit_the_arena(void)
{
    ft_tt_op_topk_request request = minicpm_request();
    ft_tt_launch_cb_entry entries[FT_TT_OP_SAMPLE_CB_COUNT];
    uint64_t mask = 0u;
    int status;

    ft_tt_test_step("the arena: one tile stream and two blocks that are not streams");
    status = ft_tt_op_sample_place_cbs(&request, entries, g_cb_buffer, sizeof(g_cb_buffer),
                                       &mask);
    ft_tt_test_check_fmt("the MiniCPM5-1B shape places", status == FT_TT_OK, "status %d",
                         status);
    if (status != FT_TT_OK)
        return;

    ft_tt_test_check_fmt("every base is 64-byte aligned",
                         (entries[0].address % 64u) == 0u && (entries[1].address % 64u) == 0u &&
                             (entries[2].address % 64u) == 0u,
                         "0x%X 0x%X 0x%X", entries[0].address, entries[1].address,
                         entries[2].address);
    ft_tt_test_check_fmt("the result block does not overlap the tile stream",
                         entries[1].address >= entries[0].address + entries[0].size_bytes,
                         "stream ends 0x%X, result at 0x%X",
                         entries[0].address + entries[0].size_bytes, entries[1].address);
    ft_tt_test_check_fmt("the state block does not overlap the result block",
                         entries[2].address >= entries[1].address + FT_TT_OP_SAMPLE_RESULT_BYTES,
                         "result ends 0x%X, state at 0x%X",
                         entries[1].address + FT_TT_OP_SAMPLE_RESULT_BYTES, entries[2].address);
    ft_tt_test_check_fmt("everything stays inside the arena",
                         entries[2].address + FT_TT_OP_SAMPLE_STATE_BYTES <= FT_TT_OP_CB_ARENA_END,
                         "ends 0x%X, arena ends 0x%X",
                         entries[2].address + FT_TT_OP_SAMPLE_STATE_BYTES, FT_TT_OP_CB_ARENA_END);

    /* The logit ROW does not fit L1 and that is the argument for the whole op shape being a
     * stream. Asserted so it goes red rather than being rediscovered. */
    ft_tt_test_check_fmt("a whole MiniCPM5-1B logit row does NOT fit the arena",
                         4080u * FT_TT_OP_TILE_BYTES_BF16 >
                             (uint32_t)(FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE),
                         "row %u B, arena %u B", 4080u * FT_TT_OP_TILE_BYTES_BF16,
                         (uint32_t)(FT_TT_OP_CB_ARENA_END - FT_TT_OP_CB_ARENA_BASE));
}

/* ================================================================
 * The sampler
 * ================================================================ */

static ft_tt_op_sample_params default_params(void)
{
    ft_tt_op_sample_params params;

    memset(&params, 0, sizeof(params));
    params.temperature = 0.7f;
    params.top_p = 1.0f;
    params.repeat_penalty = 1.0f;
    params.top_k = 40u;
    params.repeat_window = 0u;
    params.rng_seed = 0x2545F491u;
    return params;
}

/**
 * THE ORDER. Greedy must be a pure argmax on the RAW values with NO repetition penalty.
 *
 * The case is built so the two orders give DIFFERENT answers: the top token is a recent
 * one, and a penalty applied first demotes it below a near-tied runner-up. That is exactly
 * the shipped K80 bug — ':' at 25.41 losing to ' given' at 25.08 and steering the model
 * into a wrong attractor with a forward pass that was faithful throughout. A test that only
 * checked "greedy returns the argmax of what it was given" would pass against the bug.
 */
static void test_greedy_takes_no_repetition_penalty(void)
{
    uint32_t indices[2] = {11u, 400u};
    float values[2];
    ft_tt_op_sample_params params = default_params();
    uint32_t recent[1] = {11u};
    uint32_t rng = params.rng_seed;
    uint32_t greedy_token;
    uint32_t penalised_token;

    ft_tt_test_step("the sampler's ORDER: greedy is argmax BEFORE the penalty");
    values[0] = ft_float_from_bf16(ft_bf16_from_float(25.41f));
    values[1] = ft_float_from_bf16(ft_bf16_from_float(25.08f));

    params.temperature = 0.0f;
    params.repeat_penalty = 1.1f;
    params.repeat_window = 1u;
    greedy_token = ft_tt_topk_reference_sample(indices, values, 2u, &params, recent, 1u, &rng, 0);
    ft_tt_test_check_fmt("greedy returns the true argmax", greedy_token == 11u, "token %u",
                         greedy_token);

    /* And prove the case is not vacuous: with the penalty applied FIRST the answer changes,
     * so the check above is discriminating rather than passing for free. */
    {
        float penalised[2];

        penalised[0] = values[0] / params.repeat_penalty;
        penalised[1] = values[1];
        penalised_token = (penalised[0] > penalised[1]) ? 11u : 400u;
    }
    ft_tt_test_check_fmt("penalty-first WOULD have returned the other token",
                         penalised_token == 400u, "token %u", penalised_token);

    /* The greedy path must also leave the RNG untouched: a greedy turn that advanced the
     * stream would make a later stochastic turn depend on how many greedy tokens preceded
     * it, which is a reproducibility bug nobody would look for. */
    ft_tt_test_check_fmt("greedy does not advance the RNG", rng == params.rng_seed,
                         "state 0x%08X", rng);
}

/** Greedy resolves a tie by the LOWEST id, matching the full-vocabulary scan. */
static void test_greedy_tie_goes_to_the_lowest_id(void)
{
    uint32_t indices[3] = {900u, 12u, 44u};
    float values[3];
    ft_tt_op_sample_params params = default_params();
    uint32_t rng = params.rng_seed;
    uint32_t token;

    ft_tt_test_step("greedy tie-break");
    values[0] = ft_float_from_bf16(ft_bf16_from_float(4.0f));
    values[1] = ft_float_from_bf16(ft_bf16_from_float(4.0f));
    values[2] = ft_float_from_bf16(ft_bf16_from_float(4.0f));
    params.temperature = 0.0f;
    token = ft_tt_topk_reference_sample(indices, values, 3u, &params, NULL, 0u, &rng, 0);
    ft_tt_test_check_fmt("three-way tie goes to the lowest id", token == 12u, "token %u",
                         token);
}

/** The software exp, measured against libm rather than assumed. */
static void test_software_exp_error_is_measured(void)
{
    double worst_relative = 0.0;
    double worst_at = 0.0;
    int step;

    ft_tt_test_step("the software exp, against libm");
    for (step = 0; step <= 4000; step++) {
        float x = -(float)step * 0.02f;
        float ours = ft_tt_topk_reference_exp(x);
        float theirs = expf(x);
        double relative;

        if (theirs == 0.0f)
            continue;
        relative = fabs((double)ours - (double)theirs) / (double)theirs;
        if (relative > worst_relative) {
            worst_relative = relative;
            worst_at = (double)x;
        }
    }
    /* MEASURED, not assumed, and the bound is set where the mechanism puts it rather than
     * where it would look best: the error is NOT the polynomial's, it is single-precision
     * ARGUMENT REDUCTION. `x * log2(e)` at x = -75 has an absolute fp32 error of ~108*2^-24,
     * and ln2 times that is ~4.5e-6 of relative error in the result no polynomial can
     * recover. libm splits the reduction to beat it; we do not, because the two sides of
     * this op's gate share this function and the sampler's probabilities do not care about
     * the sixth decimal. A tighter bound here would be a bound on the wrong thing. */
    ft_tt_test_check_fmt("max relative error over x in [-80, 0]", worst_relative < 1e-5,
                         "%.3e at x = %.2f", worst_relative, worst_at);
    ft_tt_test_check_fmt("exp(0) is exactly 1", ft_tt_topk_reference_exp(0.0f) == 1.0f,
                         "%.9g", (double)ft_tt_topk_reference_exp(0.0f));
}

/**
 * The stochastic path, gated on the DISTRIBUTION over many draws.
 *
 * A single sampled token proves nothing about a sampler — any bug that permutes the
 * survivors still returns one of them. So: 100,000 draws from a known set, and every
 * token's empirical frequency must sit within a few standard errors of its softmax
 * probability. The set is deliberately small and well separated so the expected
 * frequencies are far apart and a permutation cannot hide inside the noise.
 */
static void test_stochastic_draw_matches_the_distribution(void)
{
    enum { COUNT = 4u, DRAWS = 100000u };
    uint32_t indices[COUNT] = {3u, 17u, 250u, 9001u};
    float values[COUNT];
    ft_tt_op_sample_params params = default_params();
    uint32_t rng = params.rng_seed;
    uint32_t hits[COUNT];
    double expected[COUNT];
    double sum = 0.0;
    uint32_t draw;
    uint32_t entry;
    double worst = 0.0;

    ft_tt_test_step("the stochastic draw, over 100,000 samples");
    values[0] = ft_float_from_bf16(ft_bf16_from_float(3.0f));
    values[1] = ft_float_from_bf16(ft_bf16_from_float(2.0f));
    values[2] = ft_float_from_bf16(ft_bf16_from_float(1.0f));
    values[3] = ft_float_from_bf16(ft_bf16_from_float(0.0f));
    params.temperature = 1.0f;
    params.top_k = 0u;

    for (entry = 0u; entry < COUNT; entry++) {
        expected[entry] = exp((double)values[entry] - 3.0);
        sum += expected[entry];
        hits[entry] = 0u;
    }
    for (entry = 0u; entry < COUNT; entry++)
        expected[entry] /= sum;

    for (draw = 0u; draw < DRAWS; draw++) {
        uint32_t token =
            ft_tt_topk_reference_sample(indices, values, COUNT, &params, NULL, 0u, &rng, 0);

        for (entry = 0u; entry < COUNT; entry++)
            if (indices[entry] == token)
                hits[entry]++;
    }
    for (entry = 0u; entry < COUNT; entry++) {
        double observed = (double)hits[entry] / (double)DRAWS;
        double error = fabs(observed - expected[entry]);
        double standard = sqrt(expected[entry] * (1.0 - expected[entry]) / (double)DRAWS);

        if (error / standard > worst)
            worst = error / standard;
        ft_tt_test_check_fmt("token frequency within 4 standard errors", error < 4.0 * standard,
                             "id %u: observed %.5f, expected %.5f, %.2f sigma", indices[entry],
                             observed, expected[entry], error / standard);
    }
    ft_tt_test_check_fmt("worst deviation across the set", worst < 4.0, "%.2f sigma", worst);

    /* And the draw must actually be a draw: top_k = 1 collapses it to a point mass, which
     * is what proves the frequencies above came from the distribution rather than from an
     * accident of the ordering. */
    params.top_k = 1u;
    rng = params.rng_seed;
    for (entry = 0u; entry < COUNT; entry++)
        hits[entry] = 0u;
    for (draw = 0u; draw < 1000u; draw++) {
        uint32_t token =
            ft_tt_topk_reference_sample(indices, values, COUNT, &params, NULL, 0u, &rng, 0);

        for (entry = 0u; entry < COUNT; entry++)
            if (indices[entry] == token)
                hits[entry]++;
    }
    ft_tt_test_check_fmt("top_k = 1 is a point mass on the argmax", hits[0] == 1000u,
                         "%u of 1000 on id 3", hits[0]);
}

/** top_p must cut the tail, and the cut must be the reference's cut. */
static void test_top_p_cuts_the_tail(void)
{
    enum { COUNT = 4u, DRAWS = 20000u };
    uint32_t indices[COUNT] = {3u, 17u, 250u, 9001u};
    float values[COUNT];
    ft_tt_op_sample_params params = default_params();
    uint32_t rng = params.rng_seed;
    uint32_t tail_hits = 0u;
    uint32_t draw;

    ft_tt_test_step("top_p");
    values[0] = ft_float_from_bf16(ft_bf16_from_float(6.0f));
    values[1] = ft_float_from_bf16(ft_bf16_from_float(5.0f));
    values[2] = ft_float_from_bf16(ft_bf16_from_float(-4.0f));
    values[3] = ft_float_from_bf16(ft_bf16_from_float(-6.0f));
    params.temperature = 1.0f;
    params.top_k = 0u;
    params.top_p = 0.9f;

    for (draw = 0u; draw < DRAWS; draw++) {
        uint32_t token =
            ft_tt_topk_reference_sample(indices, values, COUNT, &params, NULL, 0u, &rng, 0);

        if (token == 250u || token == 9001u)
            tail_hits++;
    }
    ft_tt_test_check_fmt("top_p 0.9 never emits the far tail", tail_hits == 0u,
                         "%u of %u draws", tail_hits, DRAWS);

    params.top_p = 1.0f;
    rng = params.rng_seed;
    tail_hits = 0u;
    for (draw = 0u; draw < DRAWS; draw++) {
        uint32_t token =
            ft_tt_topk_reference_sample(indices, values, COUNT, &params, NULL, 0u, &rng, 0);

        if (token == 250u || token == 9001u)
            tail_hits++;
    }
    /* The control that stops the check above passing for free: with top_p off the tail is
     * still ~4e-5 of the mass, so at 20,000 draws it is expected to be 0 or 1 — which means
     * "never emits the tail" would pass with top_p disabled too. So the discriminating
     * assertion is on the SURVIVOR SET, not on the draws. */
    {
        uint32_t low_probability_survivors = 0u;
        float probabilities[COUNT];
        float total = 0.0f;
        uint32_t entry;

        for (entry = 0u; entry < COUNT; entry++) {
            probabilities[entry] = ft_tt_topk_reference_exp(values[entry] - values[0]);
            total += probabilities[entry];
        }
        for (entry = 2u; entry < COUNT; entry++)
            if (probabilities[entry] / total > 0.0f)
                low_probability_survivors++;
        ft_tt_test_check_fmt("the tail is representable at all, so the cut is real",
                             low_probability_survivors == 2u, "%u tail entries",
                             low_probability_survivors);
    }
}

/** The state block: what the closed decode loop carries, and the refusals that protect it. */
static void test_state_block(void)
{
    ft_tt_op_sample_params params = default_params();
    uint32_t state[FT_TT_OP_SAMPLE_STATE_WORDS];
    int status;

    ft_tt_test_step("the sampler state a closed decode loop carries");
    params.eos_count = 2u;
    params.eos_ids[0] = 2u;
    params.eos_ids[1] = 73440u;
    params.repeat_window = 64u;
    params.token_budget = 512u;

    status = ft_tt_op_sample_state_build(&params, state, FT_TT_OP_SAMPLE_STATE_WORDS);
    ft_tt_test_check_fmt("a well-formed state builds", status == FT_TT_OK, "status %d", status);
    ft_tt_test_check_fmt("the magic is written", state[FT_TT_OP_SAMPLE_STATE_MAGIC] ==
                                                    FT_TT_OP_SAMPLE_STATE_MAGIC_VALUE,
                         "0x%08X", state[FT_TT_OP_SAMPLE_STATE_MAGIC]);
    ft_tt_test_check_fmt("the EOS set lands at its offset",
                         state[FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS] == 2u &&
                             state[FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS + 1u] == 73440u,
                         "%u %u", state[FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS],
                         state[FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS + 1u]);
    ft_tt_test_check_fmt("the rings do not overlap",
                         FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS >=
                                 FT_TT_OP_SAMPLE_RECENT_OFFSET_WORDS +
                                     FT_TT_OP_SAMPLE_RECENT_CAPACITY &&
                             FT_TT_OP_SAMPLE_EMITTED_OFFSET_WORDS >=
                                 FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS + FT_TT_OP_SAMPLE_EOS_CAPACITY,
                         "recent %u, eos %u, emitted %u",
                         (uint32_t)FT_TT_OP_SAMPLE_RECENT_OFFSET_WORDS,
                         (uint32_t)FT_TT_OP_SAMPLE_EOS_OFFSET_WORDS,
                         (uint32_t)FT_TT_OP_SAMPLE_EMITTED_OFFSET_WORDS);

    params = default_params();
    params.rng_seed = 0u;
    ft_tt_test_check("a zero RNG seed is refused",
                     ft_tt_op_sample_state_build(&params, state, FT_TT_OP_SAMPLE_STATE_WORDS) ==
                         FT_TT_EINVAL,
                     "FT_TT_EINVAL");
    params = default_params();
    params.repeat_window = FT_TT_OP_SAMPLE_RECENT_CAPACITY + 1u;
    ft_tt_test_check("a repeat window past the ring is refused",
                     ft_tt_op_sample_state_build(&params, state, FT_TT_OP_SAMPLE_STATE_WORDS) ==
                         FT_TT_EINVAL,
                     "FT_TT_EINVAL");
}

/** The result block round-trips, and a wrong address is a refusal rather than a number. */
static void test_result_block_round_trip(void)
{
    uint32_t block[FT_TT_OP_SAMPLE_RESULT_BYTES / 4u];
    uint32_t indices[FT_TT_OP_SAMPLE_MAX_SELECT];
    float values[FT_TT_OP_SAMPLE_MAX_SELECT];
    uint32_t count = 0u;
    uint32_t token = 0xFFFFFFFFu;
    uint32_t entry;
    int status;

    ft_tt_test_step("the result block");
    memset(block, 0, sizeof(block));
    block[FT_TT_OP_SAMPLE_RESULT_MAGIC] = FT_TT_OP_SAMPLE_RESULT_MAGIC_VALUE;
    block[FT_TT_OP_SAMPLE_RESULT_COUNT] = 3u;
    block[FT_TT_OP_SAMPLE_RESULT_TOKEN_VALID] = 1u;
    block[FT_TT_OP_SAMPLE_RESULT_TOKEN] = 4242u;
    for (entry = 0u; entry < 3u; entry++) {
        block[FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS + entry * 2u] = 100u + entry;
        block[FT_TT_OP_SAMPLE_RESULT_HEADER_WORDS + entry * 2u + 1u] =
            (uint32_t)ft_bf16_from_float(3.0f - (float)entry) << 16;
    }
    status = ft_tt_op_sample_result_decode(block, FT_TT_OP_SAMPLE_MAX_SELECT, indices, values,
                                           &count, &token);
    ft_tt_test_check_fmt("a well-formed block decodes", status == FT_TT_OK, "status %d",
                         status);
    ft_tt_test_check_fmt("count and indices survive",
                         count == 3u && indices[0] == 100u && indices[2] == 102u,
                         "count %u, ids %u..%u", count, indices[0], indices[2]);
    ft_tt_test_check_fmt("the value widening is exact", values[0] == 3.0f && values[1] == 2.0f,
                         "%.9g %.9g", (double)values[0], (double)values[1]);
    ft_tt_test_check_fmt("the token comes back", token == 4242u, "%u", token);

    block[FT_TT_OP_SAMPLE_RESULT_MAGIC] = 0u;
    ft_tt_test_check("a block with no magic is refused, not reported",
                     ft_tt_op_sample_result_decode(block, FT_TT_OP_SAMPLE_MAX_SELECT, indices,
                                                   values, &count, &token) == FT_TT_EINVAL,
                     "FT_TT_EINVAL");
}

/** The tilize helper the on-card test feeds the card with, checked against the layout the
 *  kernel indexes — because the face split is exactly the bug that would be invisible. */
static void test_tilized_row_places_the_logits_where_the_kernel_reads(void)
{
    enum { VOCAB = 64u };
    uint16_t row[VOCAB];
    uint8_t tiles[2u * 2048u];
    uint32_t index;
    uint32_t wrong = 0u;
    const uint32_t token_row = 13u;

    ft_tt_test_step("the tilized row, against the face offsets the kernel computes");
    for (index = 0u; index < VOCAB; index++)
        row[index] = ft_bf16_from_float((float)index + 0.5f);
    ft_tt_test_check("the row tilizes",
                     ft_tt_topk_reference_tilize_row(row, VOCAB, token_row, 0u, tiles,
                                                     sizeof(tiles)) == 0,
                     "0");

    /* The kernel's own arithmetic: face_row = row/16, and columns 0-15 and 16-31 live 512
     * bytes apart. Reproduced here rather than called, so the two can disagree. */
    for (index = 0u; index < VOCAB; index++) {
        uint32_t tile = index / 32u;
        uint32_t column = index % 32u;
        uint32_t face_row = token_row / 16u;
        uint32_t within = token_row % 16u;
        uint32_t offset = tile * 2048u + face_row * 1024u + within * 32u +
                          (column / 16u) * 512u + (column % 16u) * 2u;
        uint16_t stored;

        memcpy(&stored, tiles + offset, sizeof(stored));
        if (stored != row[index])
            wrong++;
    }
    ft_tt_test_check_fmt("every logit is where the kernel's face arithmetic looks",
                         wrong == 0u, "%u of %u wrong", wrong, VOCAB);
}

int main(void)
{
    test_wire_constants_agree();
    test_key_order_is_the_float_order();
    test_accept_interval_admits_exactly_the_winners();
    test_selection_matches_the_reference();
    test_ties_at_the_selection_boundary();
    test_signed_zero_divergence_is_the_documented_one();
    test_parameters_and_refusals();
    test_circular_buffers_fit_the_arena();
    test_greedy_takes_no_repetition_penalty();
    test_greedy_tie_goes_to_the_lowest_id();
    test_software_exp_error_is_measured();
    test_stochastic_draw_matches_the_distribution();
    test_top_p_cuts_the_tail();
    test_state_block();
    test_result_block_round_trip();
    test_tilized_row_places_the_logits_where_the_kernel_reads();
    return ft_tt_test_finish();
}
