/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_norm_role.c
 * @brief The PERSISTENT NORM ROLE off the card (P0.1,
 *   P0.2):
 *   1. the role skeleton's wire: per-member tables check, a corrupted / foreign table and an
 *      out-of-step descriptor are refused, the schedule walks token after token;
 *   2. exchange targets: the members as rectangles with exact receiver counts;
 *   3. the CB placement fits the arena at Qwen3.8's shape (T 160, 8 members);
 *   4. a host model of the exchange across members under random interleavings: every reduce
 *      sees exactly its own step's T tiles — and the RED model that drops the group-completion
 *      dependency is caught overwriting a slice under a slower member;
 *   5. teardown: the exchange wait leaves on the exit word (RED=ignore_exit builds the wait
 *      that ignores it, and the host check must catch it);
 *   5b. flags: each member's stage word shares its destination word's offset within 16 bytes
 *      (a NoC write lands only then), and the writer stages there;
 *   6. the compute verbs, from the SOURCES: map / reduce / chain are resident_reduce_compute.c's
 *      and ADD / copy are resident_eltwise_compute.c's, call for call;
 *   7. the CB tables agree on format, fidelity, approximation and DST mode;
 *   8. numerics: a host model of the role's distributed reduce equals the one-core reduce bit
 *      for bit, and RED=order (slice 0 summed last) differs;
 *   9. exchange levers (spec §11.11): a slice goes as whole-page runs (one write per rectangle),
 *      and neither the exchange nor the output has a barrier inside a loop — RED=per_tile_barrier
 *      reads the frozen P0 writer (reference/norm_role_writer_p0_per_tile_barrier.c) and fails.
 *  10. exchange modes (spec §11.13): a host model of every member's sends in each mode lands
 *      every receiver's faces 0-1 exactly once (faces 2-3 untouched under faces), faces halves
 *      the link bytes, stagger splits the first bursts 4 / 4, undefined mode bits are refused,
 *      and the reader keeps per-sender arrival clocks — RED=no_stagger and RED=face0_only fail.
 *
 *   make -f Makefile.norm_role run                    must pass
 *   make -f Makefile.norm_role RED=ignore_exit run    must FAIL (check 5)
 *   make -f Makefile.norm_role RED=order run          must FAIL (check 8)
 *   make -f Makefile.norm_role RED=no_dependency run  must FAIL (check 4)
 *   make -f Makefile.norm_role RED=shared_stage run   must FAIL (check 5b)
 *   make -f Makefile.norm_role RED=per_tile_barrier run  must FAIL (check 9)
 *   make -f Makefile.norm_role RED=no_stagger run     must FAIL (check 10)
 *   make -f Makefile.norm_role RED=face0_only run     must FAIL (check 10)
 *   FT_TT_KERNEL_DIR=<dir> reads the kernel sources from a mutated copy.
 *
 * @status P0.
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_op_norm_role.h"
#include "ft_tt_test_report.h"

#define QWEN38_TILE_COLS 160u
#define QWEN38_STEPS 129u
#define QWEN38_MEMBERS 8u

/* ===== Fixtures ========================================================================= */

/** Qwen3.8's role on grid cores 0-3 and 7-10: (1..4, 2) and (10..13, 2). */
static void qwen38_schedule(ft_tt_op_norm_role_schedule *s)
{
    static const uint32_t xs[QWEN38_MEMBERS] = { 1, 2, 3, 4, 10, 11, 12, 13 };

    memset(s, 0, sizeof(*s));
    s->members = QWEN38_MEMBERS;
    for (uint32_t m = 0; m < QWEN38_MEMBERS; m++) {
        s->member_x[m] = xs[m];
        s->member_y[m] = 2u;
    }
    s->tile_cols = QWEN38_TILE_COLS;
    s->steps = QWEN38_STEPS;
    for (uint32_t k = 0; k < QWEN38_STEPS; k++) {
        ft_tt_op_norm_role_step *step = &s->step[k];

        step->kind = k == 0u ? kNrKindLoad : kNrKindAdd;
        step->input.x = 0u;
        step->input.y = 1u;
        step->input.base = k == 0u ? 0x100000u : 0x200000u;
        step->constants.x = 0u;
        step->constants.y = (k % 7u) + 1u;
        step->constants.base = 0x300000u + k * 0x60000u;
        step->output.x = 0u;
        step->output.y = 2u;
        step->output.base = 0x400000u;
    }
}

/* ===== 1. Wire =========================================================================== */

static void check_wire(void)
{
    static ft_tt_op_norm_role_schedule s;
    uint32_t table[FT_TT_OP_NORM_ROLE_TABLE_WORDS];
    uint32_t params[kNrParamWords];
    uint32_t used = 0u, refused = 0u, walked = 0u;

    ft_tt_test_step("1. role skeleton wire");
    qwen38_schedule(&s);
    for (uint32_t m = 0; m < QWEN38_MEMBERS; m++) {
        const int rc = ft_tt_op_norm_role_table(&s, m, table, &used);

        if (rc != FT_TT_OK || ftnr_table_check(table, m) != kNrOk)
            refused++;
    }
    ft_tt_test_check_fmt("every member's table checks", refused == 0u, "%u refused, %u words",
                         refused, used);
    (void)ft_tt_op_norm_role_table(&s, 3u, table, &used);
    ft_tt_test_check("a table read as another member is refused",
                     ftnr_table_check(table, 4u) == kNrBadTable, "member 3's table as 4");
    table[kNrHeadWords + 50u * kNrEntryWords + kNrEntryOutBase] ^= 0x40u;
    ft_tt_test_check("one flipped entry bit is refused", ftnr_table_check(table, 3u) == kNrBadTable,
                     "step 50 output base");
    (void)ft_tt_op_norm_role_table(&s, 3u, table, &used);

    /* Three tokens of descriptors, in order: every one accepted; the walk wraps. */
    for (uint32_t n = 1; n <= 3u * QWEN38_STEPS; n++) {
        (void)ft_tt_op_norm_role_parameters(&s, (n - 1u) % QWEN38_STEPS, params);
        walked += ftnr_descriptor_check(params, table, n) == kNrOk;
    }
    ft_tt_test_check_fmt("three tokens of steps walk in order", walked == 3u * QWEN38_STEPS,
                         "%u of %u accepted", walked, 3u * QWEN38_STEPS);
    (void)ft_tt_op_norm_role_parameters(&s, 5u, params);
    ft_tt_test_check("a step out of order is refused",
                     ftnr_descriptor_check(params, table, 7u) == kNrBadStep, "step 5 at n=7");
    params[kNrParamKind] = kNrKindLoad;
    ft_tt_test_check("a kind that disagrees with the table is refused",
                     ftnr_descriptor_check(params, table, 6u) == kNrBadKind, "LOAD at step 5");
    s.step[5].kind = kNrKindLoad;
    ft_tt_test_check("a LOAD past step 0 is not a table", ft_tt_op_norm_role_table(&s, 0u, table,
                     &used) == FT_TT_EINVAL, "step 5 LOAD");
}

/* ===== 2. Targets ======================================================================== */

static void check_targets(void)
{
    static ft_tt_op_norm_role_schedule s;
    uint32_t words[2u * kNrMaxTargets];
    uint32_t count = 0u;

    ft_tt_test_step("2. exchange targets");
    qwen38_schedule(&s);
    (void)ft_tt_op_norm_role_targets(&s, 1u, words, &count);
    ft_tt_test_check_fmt("two rows of four: two rectangles", count == 2u &&
                         words[0] == (1u | 2u << 8 | 4u << 16 | 2u << 24) &&
                         (words[1] & 0xFFu) == 3u && (words[3] & 0xFFu) == 4u,
                         "count %u, receivers %u / %u", count, words[1] & 0xFFu, words[3] & 0xFFu);
    for (uint32_t m = 0; m < 8u; m++) {
        s.member_x[m] = 1u + m % 4u;
        s.member_y[m] = 2u + m / 4u;
    }
    (void)ft_tt_op_norm_role_targets(&s, 0u, words, &count);
    ft_tt_test_check_fmt("a 4x2 block is one rectangle of 7 receivers",
                         count == 1u && (words[1] & 0xFFu) == 7u, "count %u, receivers %u",
                         count, words[1] & 0xFFu);
    for (uint32_t m = 0; m < 8u; m++) {
        s.member_x[m] = 1u + 2u * m;
        s.member_y[m] = 2u;
    }
    ft_tt_test_check("eight scattered cores (8 rectangles) are refused",
                     ft_tt_op_norm_role_targets(&s, 0u, words, &count) == FT_TT_EINVAL, "x odd");
}

/* ===== 3. Placement ====================================================================== */

static void check_placement(void)
{
    ft_tt_launch_cb_entry entries[FT_TT_OP_NORM_ROLE_CB_COUNT];
    unsigned char buffer[64u * 64u];
    uint64_t mask = 0u;
    uint32_t top = 0u;
    int rc;

    ft_tt_test_step("3. CB placement");
    rc = ft_tt_op_norm_role_place_cbs(QWEN38_TILE_COLS, QWEN38_TILE_COLS / QWEN38_MEMBERS, entries,
                                      buffer, sizeof(buffer), &mask);
    for (uint32_t i = 0; rc == FT_TT_OK && i < FT_TT_OP_NORM_ROLE_CB_COUNT; i++)
        if (entries[i].address + entries[i].size_bytes > top)
            top = entries[i].address + entries[i].size_bytes;
    ft_tt_test_check_fmt("T 160 over 8 members fits the arena", rc == FT_TT_OK && top <=
                         FT_TT_OP_CB_ARENA_END, "rc %d, top 0x%X of 0x%X (%u KB used)", rc, top,
                         FT_TT_OP_CB_ARENA_END, (top - FT_TT_OP_CB_ARENA_BASE) / 1024u);
    ft_tt_test_check("SqAll holds the whole row", rc == FT_TT_OK &&
                     ft_tt_op_norm_role_cb_address(entries, kNrCbSqAll) != 0u, "cb 7 placed");
    ft_tt_test_check("16 members (past kNrMaxMembers) are refused",
                     ft_tt_op_norm_role_place_cbs(160u, 10u, entries, buffer, sizeof(buffer),
                                                  &mask) == FT_TT_EINVAL, "S 10");
}

/* ===== 4. Exchange model ================================================================= */

/* One member's progress through step n: 0 waiting for the precondition, 1 sent its slice,
 * 2 reduced, 3 published. Every member's SqAll slot j carries the step its last write was. */
typedef struct model_member {
    uint32_t step;
    uint32_t phase;
    uint32_t slot[QWEN38_MEMBERS];
} model_member;

static uint32_t model_rng = 12345u;

static uint32_t model_next(void)
{
    model_rng = model_rng * 1103515245u + 12345u;
    return model_rng >> 8;
}

/** The precondition of member `m`'s next step: the consumer of the previous step's output ran,
 *  i.e. every member published it (member 0's group pull). RED=no_dependency: only `m`'s own. */
static bool model_precondition(const model_member *members, uint32_t m)
{
#if defined(NORM_ROLE_TEST_RED_NO_DEPENDENCY)
    return members[m].phase == 3u;
#else
    for (uint32_t j = 0; j < QWEN38_MEMBERS; j++)
        if (members[j].step < members[m].step ||
            (members[j].step == members[m].step && members[j].phase != 3u))
            return false;
    return true;
#endif
}

/** One random action. Returns false when a reduce read a slot of another step. */
static bool model_tick(model_member *members, uint32_t steps)
{
    const uint32_t m = model_next() % QWEN38_MEMBERS;
    model_member *me = &members[m];

    if (me->step >= steps)
        return true;
    if (me->phase == 3u) {
        if (me->step + 1u < steps && model_precondition(members, m)) {
            me->step++;
            me->phase = 0u;
        } else if (me->step + 1u >= steps) {
            me->step = steps;
        }
        return true;
    }
    if (me->phase == 0u) {
        for (uint32_t j = 0; j < QWEN38_MEMBERS; j++)
            members[j].slot[m] = me->step;
        me->phase = 1u;
        return true;
    }
    if (me->phase == 1u) {
        for (uint32_t j = 0; j < QWEN38_MEMBERS; j++)
            if (me->slot[j] != me->step)
                return true; /* a flag not yet at this step: wait */
        me->phase = 2u;
        return true;
    }
    /* phase 2: the chain and the publish; the slots must still be this step's (no member
     * overwrote one while this member's reduce was reading them). */
    for (uint32_t j = 0; j < QWEN38_MEMBERS; j++)
        if (me->slot[j] != me->step)
            return false;
    me->phase = 3u;
    return true;
}

static void check_exchange_model(void)
{
    uint32_t bad = 0u, runs = 2000u;

    ft_tt_test_step("4. exchange model (random interleavings)");
    for (uint32_t r = 0; r < runs; r++) {
        model_member members[QWEN38_MEMBERS];
        uint32_t ticks = 0u;
        bool done = false;

        memset(members, 0, sizeof(members));
        for (uint32_t m = 0; m < QWEN38_MEMBERS; m++)
            for (uint32_t j = 0; j < QWEN38_MEMBERS; j++)
                members[m].slot[j] = UINT32_MAX;
        while (!done && ticks < 200000u) {
            if (!model_tick(members, 24u)) {
                bad++;
                break;
            }
            done = true;
            for (uint32_t m = 0; m < QWEN38_MEMBERS; m++)
                done = done && members[m].step >= 24u;
            ticks++;
        }
    }
    ft_tt_test_check_fmt("no reduce ever reads another step's slice", bad == 0u,
                         "%u of %u runs overwrote a slice", bad, runs);
}

/* ===== 5. Teardown ======================================================================= */

static void check_teardown(void)
{
    const uint32_t bound = 1u << 20;
    uint32_t spins = 0u, action = kNrWaitSpin;

    ft_tt_test_step("5. teardown: the exit word ends an exchange wait");
    /* A member that never sends (flag stuck one step behind); the host sets the exit word
     * after 5000 polls. The wait must leave with kNrWaitStop within 1024 polls of it. */
    while (spins < bound) {
        action = ftnr_exchange_wait_action(41u, 42u, spins, spins >= 5000u, bound);
        if (action != kNrWaitSpin)
            break;
        spins++;
    }
    ft_tt_test_check_fmt("the wait leaves on the exit word", action == kNrWaitStop &&
                         spins < 5000u + 1024u, "action %u after %u polls", action, spins);
    ft_tt_test_check("a landed flag is ready at once",
                     ftnr_exchange_wait_action(42u, 42u, 0u, true, bound) == kNrWaitReady,
                     "flag 42 at n=42");
    ft_tt_test_check("a flag one step ahead is not this step's",
                     ftnr_exchange_wait_action(43u, 42u, 1u, false, bound) == kNrWaitSpin,
                     "flag 43 at n=42");
}

/* ===== 5b. Flag alignment ================================================================ */

static char *read_source(const char *name);

static void check_flag_alignment(void)
{
    uint32_t bad = 0u;
    char *writer = read_source("resident_norm_role_writer.c");

    ft_tt_test_step("5b. flags: every stage word shares its destination's offset within 16 B");
    /* Card, P0 block 2: one shared stage word at +32 reached only members 0 and 4 (destination
     * offsets 0 and 16); the other six flags never landed and every reader timed out. */
    for (uint32_t m = 0; m < (uint32_t)kNrMaxMembers; m++) {
        const uint32_t stage = ftnr_flag_stage_word(m);

        if ((4u * stage) % 16u != (4u * m) % 16u || stage < (uint32_t)kNrMaxMembers ||
            stage >= (uint32_t)kNrFlagsBytes / 4u)
            bad++;
    }
    ft_tt_test_check_fmt("stage word 8 + member: same 16-byte offset, inside the page, past the "
                         "member words", bad == 0u, "%u of %u members misaligned", bad,
                         (uint32_t)kNrMaxMembers);
    ft_tt_test_check("the writer multicasts from ftnr_flag_stage_word(member)",
                     writer && strstr(writer, "flags_base + 4u * ftnr_flag_stage_word(member)") &&
                         strstr(writer, "flags[ftnr_flag_stage_word(member)] = sequence;"),
                     "resident_norm_role_writer.c send_exchange");
    free(writer);
}

/* ===== 6. Verbs from the sources ========================================================= */

#define LLK_MAX 64
#define LLK_TEXT 320

typedef struct llk_list {
    char call[LLK_MAX][LLK_TEXT];
    int count;
} llk_list;

static const char *kernel_dir(void)
{
    const char *directory = getenv("FT_TT_KERNEL_DIR");

    return directory && *directory ? directory : "../kernels";
}

static char *read_file(const char *path)
{
    FILE *file = fopen(path, "rb");
    long size;
    char *text;

    if (!file)
        return NULL;
    fseek(file, 0, SEEK_END);
    size = ftell(file);
    fseek(file, 0, SEEK_SET);
    text = malloc((size_t)size + 1u);
    if (text && fread(text, 1, (size_t)size, file) != (size_t)size) {
        free(text);
        text = NULL;
    }
    if (text)
        text[size] = '\0';
    fclose(file);
    return text;
}

static char *read_source(const char *name)
{
    char path[512];

    snprintf(path, sizeof(path), "%s/%s", kernel_dir(), name);
    return read_file(path);
}

/** The body of function `name`: the first `name(` whose parameter list is followed by `{`. */
static const char *function_body(const char *text, const char *name, const char **end)
{
    const size_t length = strlen(name);

    for (const char *at = strstr(text, name); at; at = strstr(at + 1, name)) {
        const char *p = at + length;
        int depth = 0;

        if (*p != '(' || (at > text && (at[-1] == '_' || (at[-1] >= 'a' && at[-1] <= 'z'))))
            continue;
        for (; *p; p++) {
            depth += (*p == '(') - (*p == ')');
            if (depth == 0)
                break;
        }
        while (*p == ')' || *p == ' ' || *p == '\n')
            p++;
        if (*p != '{')
            continue;
        depth = 0;
        for (const char *q = p; *q; q++) {
            depth += (*q == '{') - (*q == '}');
            if (depth == 0) {
                *end = q;
                return p;
            }
        }
        return NULL;
    }
    return NULL;
}

/** Replace whole-word `from` with `to` in `text` (in place, bounded). */
static void replace_word(char *text, size_t size, const char *from, const char *to)
{
    char out[LLK_TEXT * 2];
    size_t n = 0u, from_length = strlen(from);

    for (const char *p = text; *p && n + 1u < sizeof(out);) {
        const bool start = p == text || !(p[-1] == '_' || (p[-1] >= 'a' && p[-1] <= 'z') ||
                                          (p[-1] >= 'A' && p[-1] <= 'Z') ||
                                          (p[-1] >= '0' && p[-1] <= '9'));
        const char after = p[from_length];

        if (start && strncmp(p, from, from_length) == 0 &&
            !(after == '_' || (after >= 'a' && after <= 'z') || (after >= 'A' && after <= 'Z') ||
              (after >= '0' && after <= '9'))) {
            n += (size_t)snprintf(out + n, sizeof(out) - n, "%s", to);
            p += from_length;
        } else {
            out[n++] = *p++;
        }
    }
    out[n < sizeof(out) ? n : sizeof(out) - 1u] = '\0';
    snprintf(text, size, "%s", out);
}

/**
 * Every ft_tensix_* call (name and argument text, whitespace removed) and every commit, in
 * source order, between `from` and `to`; `map` is pairs of words to canonicalise.
 */
static void collect_calls(const char *from, const char *to, const char *const *map, llk_list *out)
{
    for (const char *p = from; p && p < to && out->count < LLK_MAX;) {
        const char *ft = strstr(p, "ft_tensix_");
        const char *commit = strstr(p, "commit_dst_to(");
        const char *at = ft && (!commit || ft < commit) ? ft : commit;
        char *slot;
        int depth = 0;
        size_t n = 0u;

        if (!at || at >= to)
            break;
        if (at == commit && at > from && at[-1] == '_')
            at -= 3; /* nr_commit_dst_to */
        slot = out->call[out->count];
        for (const char *q = at; *q && n + 1u < LLK_TEXT; q++) {
            if (*q == ' ' || *q == '\n')
                continue;
            slot[n++] = *q;
            depth += (*q == '(') - (*q == ')');
            if (depth == 0 && *q == ')')
                break;
            p = q;
        }
        slot[n] = '\0';
        p = at + 1;
        while (*p && *p != ';')
            p++;
        if (strncmp(slot, "ft_tensix_compute_cb_", 21) == 0 ||
            strncmp(slot, "ft_tensix_cb", 12) == 0)
            continue; /* CB bookkeeping is scheduling, compared by the host models */
        replace_word(slot, LLK_TEXT, "nr_commit_dst_to", "commit_dst_to");
        for (int i = 0; map && map[i]; i += 2)
            replace_word(slot, LLK_TEXT, map[i], map[i + 1]);
        out->count++;
    }
}

static bool calls_equal(const llk_list *a, const llk_list *b, char *why, size_t size)
{
    if (a->count != b->count) {
        snprintf(why, size, "%d calls vs %d", a->count, b->count);
        return false;
    }
    for (int i = 0; i < a->count; i++)
        if (strcmp(a->call[i], b->call[i]) != 0) {
            snprintf(why, size, "call %d: %s | %s", i, a->call[i], b->call[i]);
            return false;
        }
    snprintf(why, size, "%d calls identical", a->count);
    return true;
}

/** The calls of a whole function (or of the segment between two markers inside it). */
static void calls_of(const char *text, const char *function, const char *start_marker,
                     const char *end_marker, const char *const *map, llk_list *out)
{
    const char *end = NULL;
    const char *body = function_body(text, function, &end);
    const char *from = body, *to = end;

    memset(out, 0, sizeof(*out));
    if (!body)
        return;
    if (start_marker && (from = strstr(body, start_marker)) == NULL)
        return;
    if (end_marker) {
        const char *stop = strstr(from, end_marker);

        to = stop && stop < end ? stop : end;
    }
    collect_calls(from, to, map, out);
}

static void compare(const char *what, const llk_list *a, const llk_list *b)
{
    char why[LLK_TEXT * 3];
    const bool same = a->count > 0 && calls_equal(a, b, why, sizeof(why));

    ft_tt_test_check(what, same, a->count > 0 ? why : "no calls found");
}

static void check_verbs(void)
{
    static const char *const reduce_map[] = {
        "FTR_CB_IN", "X", "FTR_CB_MID", "MID", "source", "SQ", "column", "COL",
        "FTR_CB_SCALER", "SCALER", "FTR_CB_EPS", "EPS", "ftr_nf_rows", "HALF_ROWS",
        "ftr_nf_total", "HALF_FACES", "FTR_CB_GAMMA", "GAMMA", NULL
    };
    static const char *const role_map[] = {
        "NR_CB_H", "X", "NR_CB_SQ", "MID", "NR_CB_SQALL", "SQ", "NR_CB_COL", "COL",
        "NR_CB_SCALER", "SCALER", "NR_CB_EPS", "EPS", "NR_HALF_FACE_ROWS", "HALF_ROWS",
        "NR_HALF_FACES", "HALF_FACES", "NR_CB_GAMMA", "GAMMA", NULL
    };
    static const char *const chain_gamma_map[] = {
        "source", "X", "source_index", "t", "wt", "t", "scale_column", "COL", "FTR_CB_GAMMA",
        "GAMMA", "ftr_nf_rows", "HALF_ROWS", "ftr_nf_total", "HALF_FACES", NULL
    };
    static const char *const chain_scale_map[] = {
        "source", "STAGE", "source_index", "0", "scale_column", "COL", "ftr_nf_rows",
        "HALF_ROWS", "ftr_nf_total", "HALF_FACES", NULL
    };
    static const char *const chain_role_map[] = {
        "NR_CB_H", "X", "NR_CB_STAGE", "STAGE", "NR_CB_COL", "COL", "NR_CB_GAMMA", "GAMMA",
        "NR_CB_OUT", "OUT", "NR_HALF_FACE_ROWS", "HALF_ROWS", "NR_HALF_FACES", "HALF_FACES", NULL
    };
    static const char *const eltwise_map[] = { "binary_op_word", "ADDWORD", "a_index", "0",
                                               "dst_register", "DST_REGISTER", NULL };
    static const char *const role_add_map[] = { "NR_ELTWISE_ADD_WORD", "ADDWORD", "a_index", "0",
                                                NULL };
    char *reduce = read_source("resident_reduce_compute.c");
    char *eltwise = read_source("resident_eltwise_compute.c");
    char *role = read_source("resident_norm_role_compute.c");
    llk_list a, b;

    ft_tt_test_step("6. compute verbs vs the shipped sources (%s)", kernel_dir());
    if (!reduce || !eltwise || !role) {
        ft_tt_test_check("kernel sources readable", false, kernel_dir());
        free(reduce);
        free(eltwise);
        free(role);
        return;
    }
    {
        llk_list commit;

        calls_of(reduce, "map_row", "mul_tiles_init(cb_in, cb_in)", "} else {", reduce_map, &a);
        calls_of(reduce, "map_row", "commit_dst_to(ctx, FTR_CB_MID)", NULL, reduce_map, &commit);
        for (int i = 0; i < commit.count && a.count < LLK_MAX; i++)
            memcpy(a.call[a.count++], commit.call[i], LLK_TEXT);
    }
    calls_of(role, "nr_map_square", NULL, NULL, role_map, &b);
    compare("x^2 map = map_row(kRedMapSquareSelf), half tile", &a, &b);

    {
        llk_list verb;

        calls_of(reduce, "reduce_row_sum", NULL, "apply_verb", reduce_map, &a);
        calls_of(reduce, "apply_verb", "kRedVerbRsqrt", NULL, reduce_map, &verb);
        for (int i = 0; i < verb.count && a.count < LLK_MAX; i++)
            memcpy(a.call[a.count++], verb.call[i], LLK_TEXT);
        {
            llk_list tail;

            calls_of(reduce, "reduce_row_sum", "commit_dst_to(ctx, column)", NULL, reduce_map,
                     &tail);
            for (int i = 0; i < tail.count && a.count < LLK_MAX; i++)
                memcpy(a.call[a.count++], tail.call[i], LLK_TEXT);
        }
        calls_of(role, "nr_reduce_sum", NULL, NULL, role_map, &b);
        compare("reduce = reduce_row_sum(SUM, + eps) then rsqrt, commit, uninit", &a, &b);
    }
    {
        llk_list init, tile, scale_init, scale_tile;

        calls_of(reduce, "chain_step_init", "kRedChainGamma", "} else if (step == (uint32_t)kRedChainScale",
                 chain_gamma_map, &init);
        calls_of(reduce, "chain_step_tile", "kRedChainGamma", "} else if (step == (uint32_t)kRedChainScale",
                 chain_gamma_map, &tile);
        calls_of(reduce, "chain_step_init", "kRedChainScale", "} else {", chain_scale_map,
                 &scale_init);
        calls_of(reduce, "chain_step_tile", "kRedChainScale", "} else {", chain_scale_map,
                 &scale_tile);
        memset(&a, 0, sizeof(a));
        for (int i = 0; i < init.count; i++)
            memcpy(a.call[a.count++], init.call[i], LLK_TEXT);
        {
            llk_list wait;

            calls_of(reduce, "chain_step_tile", NULL, "if (step", chain_gamma_map, &wait);
            for (int i = 0; i < wait.count; i++)
                memcpy(a.call[a.count++], wait.call[i], LLK_TEXT);
            for (int i = 0; i < tile.count; i++)
                memcpy(a.call[a.count++], tile.call[i], LLK_TEXT);
            snprintf(a.call[a.count++], LLK_TEXT, "commit_dst_to(ctx,STAGE)");
            for (int i = 0; i < scale_init.count; i++)
                memcpy(a.call[a.count++], scale_init.call[i], LLK_TEXT);
            for (int i = 0; i < wait.count; i++)
                memcpy(a.call[a.count++], wait.call[i], LLK_TEXT);
            for (int i = 0; i < scale_tile.count; i++)
                memcpy(a.call[a.count++], scale_tile.call[i], LLK_TEXT);
            snprintf(a.call[a.count++], LLK_TEXT, "commit_dst_to(ctx,OUT)");
        }
        calls_of(role, "nr_chain", NULL, NULL, chain_role_map, &b);
        compare("chain = chain_step gamma then scale, per tile", &a, &b);
    }
    calls_of(eltwise, "eltwise_addsub_init", NULL, NULL, eltwise_map, &a);
    calls_of(role, "nr_add_init", NULL, NULL, role_add_map, &b);
    compare("ADD init = eltwise_addsub_init", &a, &b);
    calls_of(eltwise, "eltwise_addsub_apply", NULL, NULL, eltwise_map, &a);
    calls_of(role, "nr_add_apply", NULL, NULL, role_add_map, &b);
    compare("ADD apply = eltwise_addsub_apply (residual as A)", &a, &b);
    ft_tt_test_check("ADD call site: residual (H) is operand A, the projection B",
                     strstr(role, "nr_add_apply(NR_CB_H, NR_CB_IN, i)") != NULL,
                     "pn_tt_add(hidden, project)");
    calls_of(eltwise, "eltwise_copy_init", NULL, NULL, NULL, &a);
    calls_of(role, "nr_copy_init", NULL, NULL, NULL, &b);
    compare("LOAD copy init = eltwise_copy_init", &a, &b);
    calls_of(eltwise, "eltwise_copy_apply", NULL, NULL, eltwise_map, &a);
    calls_of(role, "nr_copy_apply", NULL, NULL, NULL, &b);
    compare("LOAD copy apply = eltwise_copy_apply", &a, &b);
    {
        const char *e = strstr(eltwise, "#define ELTWISE_ADD_WORD");
        const char *r = strstr(role, "#define NR_ELTWISE_ADD_WORD");
        char ew[160] = "", rw[160] = "";

        if (e)
            sscanf(strstr(e, "FT_BH_ENC_ELWADD"), "%159[^\n]", ew);
        if (r)
            sscanf(strstr(r, "FT_BH_ENC_ELWADD"), "%159[^\n]", rw);
        ft_tt_test_check_fmt("the ADD word is the eltwise program's", e && r && strcmp(ew, rw) == 0,
                             "%s | %s", ew, rw);
    }
    calls_of(reduce, "commit_dst_to", NULL, NULL, NULL, &a);
    calls_of(role, "nr_commit_dst_to", NULL, NULL, NULL, &b);
    compare("commit = the reduce program's commit_dst_to", &a, &b);
    free(reduce);
    free(eltwise);
    free(role);
}

/* ===== 7. CB tables ====================================================================== */

static bool cbtable_read(const char *name, int *fidelity, int *approx, int *accum, char rows[32][64])
{
    char path[512], line[256];
    FILE *file;

    snprintf(path, sizeof(path), "%s/cb_tables/%s", kernel_dir(), name);
    file = fopen(path, "r");
    if (!file)
        return false;
    memset(rows, 0, 32u * 64u);
    while (fgets(line, sizeof(line), file)) {
        int index;
        char rest[64];

        if (sscanf(line, "math_fidelity %d", fidelity) == 1 || sscanf(line, "approx %d", approx) == 1 ||
            sscanf(line, "dst_accum_mode %d", accum) == 1)
            continue;
        if (sscanf(line, "cb %d %63[^\n]", &index, rest) == 2 && index >= 0 && index < 32)
            snprintf(rows[index], 64, "%s", rest);
    }
    fclose(file);
    return true;
}

static void check_cbtables(void)
{
    static const int compute_cbs[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 16 };
    char reduce[32][64], eltwise[32][64], role[32][64];
    int f[3] = { -1, -1, -1 }, ap[3] = { -1, -1, -1 }, ac[3] = { -1, -1, -1 };
    bool ok, same = true;

    ft_tt_test_step("7. CB tables");
    ok = cbtable_read("reduce_float16b.cbtable", &f[0], &ap[0], &ac[0], reduce) &&
         cbtable_read("eltwise_float16b.cbtable", &f[1], &ap[1], &ac[1], eltwise) &&
         cbtable_read("norm_role_float16b.cbtable", &f[2], &ap[2], &ac[2], role);
    ft_tt_test_check_fmt("fidelity / approx / DST mode equal", ok && f[0] == f[2] && f[1] == f[2] &&
                         ap[0] == ap[2] && ap[1] == ap[2] && ac[0] == ac[2] && ac[1] == ac[2],
                         "fidelity %d %d %d, approx %d %d %d, accum %d %d %d", f[0], f[1], f[2],
                         ap[0], ap[1], ap[2], ac[0], ac[1], ac[2]);
    for (size_t i = 0; ok && i < sizeof(compute_cbs) / sizeof(compute_cbs[0]); i++)
        same = same && strcmp(role[compute_cbs[i]], reduce[0]) == 0 &&
               strcmp(role[compute_cbs[i]], eltwise[0]) == 0;
    ft_tt_test_check_fmt("every compute CB is the shipped tile", ok && same, "'%s'",
                         ok ? reduce[0] : "?");
}

/* ===== 8. Numerics model ================================================================= */

static float bf16_round(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, 4);
    bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    memcpy(&value, &bits, 4);
    return value;
}

/** Row sum of a tile row (32 values), sequential fp32 — the model of one tile's reduce. */
static float tile_row_sum(const float *row)
{
    float sum = 0.0f;

    for (uint32_t c = 0; c < 32u; c++)
        sum += row[c];
    return sum;
}

static void check_numerics(void)
{
    static float h[QWEN38_TILE_COLS * 32u], o[QWEN38_TILE_COLS * 32u];
    static float square[QWEN38_TILE_COLS * 32u], sqall[QWEN38_MEMBERS][QWEN38_TILE_COLS * 32u];
    const uint32_t slice = QWEN38_TILE_COLS / QWEN38_MEMBERS;
    float reference = 0.0f, member_sum[QWEN38_MEMBERS];
    uint32_t agree = 0u;

    ft_tt_test_step("8. numerics model");
    for (uint32_t i = 0; i < QWEN38_TILE_COLS * 32u; i++) {
        h[i] = bf16_round(((float)(model_next() % 20001u) - 10000.0f) / 2500.0f);
        o[i] = bf16_round(((float)(model_next() % 20001u) - 10000.0f) / 7000.0f);
    }
    /* One core, today: ADD then x^2, both packed bf16, then the sum over tiles in order. */
    for (uint32_t i = 0; i < QWEN38_TILE_COLS * 32u; i++) {
        const float next = bf16_round(h[i] + o[i]);

        square[i] = bf16_round(next * next);
    }
    for (uint32_t t = 0; t < QWEN38_TILE_COLS; t++)
        reference += tile_row_sum(&square[t * 32u]);
    /* The role: member m squares its slice, every member receives every slice, each reduces. */
    for (uint32_t m = 0; m < QWEN38_MEMBERS; m++)
        for (uint32_t t = m * slice; t < (m + 1u) * slice; t++)
            for (uint32_t c = 0; c < 32u; c++) {
                const float next = bf16_round(h[t * 32u + c] + o[t * 32u + c]);

                for (uint32_t r = 0; r < QWEN38_MEMBERS; r++)
                    sqall[r][t * 32u + c] = bf16_round(next * next);
            }
    for (uint32_t m = 0; m < QWEN38_MEMBERS; m++) {
        member_sum[m] = 0.0f;
        for (uint32_t j = 0; j < QWEN38_MEMBERS; j++) {
#if defined(NORM_ROLE_TEST_RED_ORDER)
            const uint32_t which = (j + 1u) % QWEN38_MEMBERS; /* slice 0 summed last */
#else
            const uint32_t which = j;
#endif
            for (uint32_t t = which * slice; t < (which + 1u) * slice; t++)
                member_sum[m] += tile_row_sum(&sqall[m][t * 32u]);
        }
        agree += memcmp(&member_sum[m], &reference, 4) == 0;
    }
    ft_tt_test_check_fmt("every member's sum is the one-core sum, bit for bit",
                         agree == QWEN38_MEMBERS, "%u of %u agree (ref %.9g, member 0 %.9g)",
                         agree, QWEN38_MEMBERS, (double)reference, (double)member_sum[0]);
}

/* ===== 9. Exchange levers (spec §11.11) =================================================== */

/** Barrier calls inside a `for` / `while` loop of [from, to): the P0 pattern, one round trip a
 *  tile. A loop body is its braced block, or its one statement. */
static uint32_t barriers_in_loops(const char *from, const char *to)
{
    static const char *const loops[] = { "for (", "while (" };
    uint32_t found = 0u;

    for (int k = 0; k < 2; k++) {
        for (const char *at = strstr(from, loops[k]); at && at < to; at = strstr(at + 1, loops[k])) {
            const char *p = at + strlen(loops[k]) - 1u, *end;
            int depth = 0;

            for (; p < to; p++) {
                depth += (*p == '(') - (*p == ')');
                if (depth == 0)
                    break;
            }
            for (p++; p < to && (*p == ' ' || *p == '\n'); p++)
                ;
            if (p < to && *p == '{') {
                depth = 0;
                for (end = p; end < to; end++) {
                    depth += (*end == '{') - (*end == '}');
                    if (depth == 0)
                        break;
                }
            } else {
                end = p;
                while (end < to && *end != ';')
                    end++;
            }
            for (const char *b = strstr(p, "_barrier("); b && b < end; b = strstr(b + 1, "_barrier("))
                found++;
        }
    }
    return found;
}

/** Does [from, to) contain `needle`? */
static bool contains(const char *from, const char *to, const char *needle)
{
    const char *at = strstr(from, needle);

    return at && at < to;
}

/** The writer source check 9 reads: the kernel, or under RED=per_tile_barrier the frozen P0
 *  writer, which must fail. */
static char *exchange_writer_source(void)
{
#if defined(NORM_ROLE_TEST_RED_PER_TILE_BARRIER)
    return read_file("reference/norm_role_writer_p0_per_tile_barrier.c");
#else
    return read_source("resident_norm_role_writer.c");
#endif
}

static void check_exchange_levers(void)
{
    const uint32_t slice = QWEN38_TILE_COLS / QWEN38_MEMBERS;
    uint32_t first[kNrMaxExchangeRuns], tile[kNrMaxExchangeRuns], pages[kNrMaxExchangeRuns];
    uint32_t runs, covered = 0u, bad = 0u;
    char *writer = exchange_writer_source();
    const char *body, *end = NULL;

    ft_tt_test_step("9. exchange levers: one write per rectangle, one barrier per slice");
    runs = ftnr_exchange_runs(0u, slice, slice, first, tile, pages);
    ft_tt_test_check_fmt("a slice from the ring base is ONE run of S whole pages", runs == 1u &&
                         first[0] == 0u && tile[0] == 0u && pages[0] == slice,
                         "%u runs, first %u pages %u", runs, first[0], pages[0]);
    for (uint32_t read = 0u; read < slice; read++) {
        runs = ftnr_exchange_runs(read, slice, slice, first, tile, pages);
        covered = 0u;
        for (uint32_t r = 0u; r < runs; r++) {
            for (uint32_t i = 0u; i < pages[r]; i++)
                bad += ftnr_ring_page(read, tile[r] + i, slice) != first[r] + i;
            bad += tile[r] != covered;
            covered += pages[r];
        }
        bad += covered != slice || runs != (read == 0u ? 1u : 2u);
    }
    ft_tt_test_check_fmt("every read page: runs cover tiles 0..S-1 in order, ring-contiguous",
                         bad == 0u, "%u mismatches over %u read pages", bad, slice);
    ft_tt_test_check("a slice past the ring, an empty slice and a page past the ring are refused",
                     ftnr_exchange_runs(0u, slice, slice + 1u, first, tile, pages) == 0u &&
                         ftnr_exchange_runs(0u, slice, 0u, first, tile, pages) == 0u &&
                         ftnr_exchange_runs(slice, slice, slice, first, tile, pages) == 0u,
                     "S+1 / 0 / read page S");
    body = writer ? function_body(writer, "send_exchange", &end) : NULL;
    ft_tt_test_check_fmt("send_exchange: no barrier inside a loop, runs from ftnr_exchange_runs",
                         body && barriers_in_loops(body, end) == 0u &&
                             contains(body, end, "ftnr_exchange_runs(") &&
                             !contains(body, end, "kNrExchangeBytes"),
                         "%u barriers in loops", body ? barriers_in_loops(body, end) : 99u);
    body = writer ? function_body(writer, "send_run", &end) : NULL;
    ft_tt_test_check_fmt("send_run: no barrier inside a loop", body &&
                         barriers_in_loops(body, end) == 0u, "%u barriers in loops",
                         body ? barriers_in_loops(body, end) : 99u);
    body = writer ? function_body(writer, "write_output", &end) : NULL;
    ft_tt_test_check_fmt("write_output: no barrier inside a loop", body &&
                         barriers_in_loops(body, end) == 0u, "%u barriers in loops",
                         body ? barriers_in_loops(body, end) : 99u);
    free(writer);
}

/* ===== 10. Exchange modes (spec §11.13): faces 0-1, stagger, arrival clocks ============== */

/** Writes a byte of every receiver's SqAll took, one byte a byte (8 x T pages x 2 KB). */
static uint8_t landed[QWEN38_MEMBERS][QWEN38_TILE_COLS * FT_TT_OP_TILE_BYTES_BF16];

/** The member at (x, y), or QWEN38_MEMBERS. */
static uint32_t member_of(const ft_tt_op_norm_role_schedule *s, uint32_t x, uint32_t y)
{
    for (uint32_t m = 0; m < s->members; m++)
        if (s->member_x[m] == x && s->member_y[m] == y)
            return m;
    return QWEN38_MEMBERS;
}

/** Host model of one step's exchange under `mode`, as send_run issues it (the same planner
 *  calls, read from each member's own table): every receiver's landed bytes, the bytes that
 *  cross links (multicast receivers, the self copy excluded) and the targets sent first. */
static bool model_exchange(uint32_t mode, uint32_t *link_bytes, uint32_t first_target[kNrMaxTargets])
{
    static ft_tt_op_norm_role_schedule s;
    uint32_t table[FT_TT_OP_NORM_ROLE_TABLE_WORDS], used = 0u;
    const uint32_t tile = FT_TT_OP_TILE_BYTES_BF16, slice = QWEN38_TILE_COLS / QWEN38_MEMBERS;

    qwen38_schedule(&s);
    s.exchange = mode;
    memset(landed, 0, sizeof(landed));
    memset(first_target, 0, kNrMaxTargets * sizeof(uint32_t));
    *link_bytes = 0u;
    for (uint32_t m = 0; m < QWEN38_MEMBERS; m++) {
        uint32_t first[kNrMaxExchangeRuns], at[kNrMaxExchangeRuns], pages[kNrMaxExchangeRuns];
        uint32_t runs, targets, writes, bytes;

        if (ft_tt_op_norm_role_table(&s, m, table, &used) != FT_TT_OK ||
            ftnr_table_check(table, m) != kNrOk || ftnr_exchange_mode(table) != mode)
            return false;
        targets = table[kNrHeadTargets];
        first_target[ftnr_exchange_target(0u, m, targets, ftnr_exchange_mode(table))]++;
        runs = ftnr_exchange_runs(0u, slice, slice, first, at, pages);
        for (uint32_t r = 0; r < runs; r++) {
            const uint32_t landing = ftnr_sqall_offset(m, at[r], slice, tile);

            writes = ftnr_exchange_writes(pages[r], ftnr_exchange_mode(table));
            bytes = ftnr_exchange_write_bytes(pages[r], tile, ftnr_exchange_mode(table));
            for (uint32_t k = 0; k < targets; k++) {
                const uint32_t t = ftnr_exchange_target(k, m, targets, ftnr_exchange_mode(table));
                const uint32_t rect = table[kNrHeadTarget0 + 2u * t];

                for (uint32_t y = (rect >> 8) & 0xFFu; y <= rect >> 24; y++)
                    for (uint32_t x = rect & 0xFFu; x <= ((rect >> 16) & 0xFFu); x++) {
                        const uint32_t to = member_of(&s, x, y);

                        if (to == QWEN38_MEMBERS || to == m) /* non-loopback multicast */
                            continue;
                        for (uint32_t w = 0; w < writes; w++)
                            for (uint32_t b = 0; b < bytes; b++)
                                landed[to][landing + w * tile + b]++;
                        *link_bytes += writes * bytes;
                    }
            }
            for (uint32_t b = 0; b < pages[r] * tile; b++) /* the self copy, whole run */
                landed[m][landing + b]++;
        }
    }
    return true;
}

static void check_exchange_modes(void)
{
    static ft_tt_op_norm_role_schedule s;
    uint32_t table[FT_TT_OP_NORM_ROLE_TABLE_WORDS], used = 0u, link[4] = { 0u }, firsts[4][kNrMaxTargets];
    const uint32_t tile = FT_TT_OP_TILE_BYTES_BF16, slice = QWEN38_TILE_COLS / QWEN38_MEMBERS;
    char *reader = read_source("resident_norm_role_reader.c");
    uint32_t perm_bad = 0u;

    ft_tt_test_step("10. exchange modes: faces 0-1, stagger, arrival clocks");
    qwen38_schedule(&s);
    s.exchange = 4u;
    ft_tt_test_check("an undefined mode bit is not a table",
                     ft_tt_op_norm_role_table(&s, 0u, table, &used) == FT_TT_EINVAL, "mode 4");
    s.exchange = 0u;
    (void)ft_tt_op_norm_role_table(&s, 2u, table, &used);
    table[kNrHeadMembers] |= 4u << 16;
    table[kNrHeadChecksum] = ftnr_checksum(table, s.steps);
    ft_tt_test_check("a table with an undefined mode bit is refused on the card",
                     ftnr_table_check(table, 2u) == kNrBadShape, "mode 4, checksum right");
    for (uint32_t mode = 0; mode < 4u; mode++)
        for (uint32_t m = 0; m < QWEN38_MEMBERS; m++)
            for (uint32_t targets = 1u; targets <= kNrMaxTargets; targets++) {
                uint32_t hit = 0u;

                for (uint32_t k = 0; k < targets; k++)
                    hit |= 1u << ftnr_exchange_target(k, m, targets, mode);
                perm_bad += hit != (1u << targets) - 1u;
            }
    ft_tt_test_check_fmt("every member visits every target once, in every mode", perm_bad == 0u,
                         "%u orders not a permutation", perm_bad);

    for (uint32_t mode = 0; mode < 4u; mode++) {
        uint32_t twice = 0u, missing = 0u, upper = 0u;

        if (!model_exchange(mode, &link[mode], firsts[mode])) {
            ft_tt_test_check_fmt("model the exchange", false, "mode %u refused", mode);
            continue;
        }
        for (uint32_t r = 0; r < QWEN38_MEMBERS; r++)
            for (uint32_t page = 0; page < QWEN38_TILE_COLS; page++)
                for (uint32_t b = 0; b < tile; b++) {
                    const uint8_t n = landed[r][page * tile + b];
                    const bool own = page / slice == r;

                    twice += n > 1u;
                    if (b < (uint32_t)kNrExchangeBytes)
                        missing += n == 0u; /* faces 0-1: what the reduce's row 0 reads */
                    else if ((mode & (uint32_t)kNrExchangeFaces) && !own)
                        upper += n != 0u;   /* faces 2-3 keep the boot zeros */
                    else
                        missing += n == 0u;
                }
        ft_tt_test_check_fmt("every receiver's faces 0-1 land exactly once from their owner",
                             twice == 0u && missing == 0u && upper == 0u,
                             "mode %u: %u bytes twice, %u missing, %u faces-2-3 written", mode,
                             twice, missing, upper);
    }
    ft_tt_test_check_fmt("faces 0-1 halve the bytes on links (whole 2.29 MB a step)",
                         link[1] * 2u == link[0] && link[3] == link[1] && link[2] == link[0] &&
                             link[0] == QWEN38_MEMBERS * (QWEN38_MEMBERS - 1u) * slice * tile,
                         "whole %u, faces %u, stagger %u, both %u B", link[0], link[1], link[2],
                         link[3]);
    ft_tt_test_check_fmt("plain: all 8 senders open on target 0; stagger: 4 and 4",
                         firsts[0][0] == QWEN38_MEMBERS && firsts[3][0] == QWEN38_MEMBERS / 2u &&
                             firsts[3][1] == QWEN38_MEMBERS / 2u,
                         "plain %u/%u, stagger %u/%u", firsts[0][0], firsts[0][1], firsts[3][0],
                         firsts[3][1]);
    ft_tt_test_check("arrival clocks sit past the stage words, inside the flags page",
                     (uint32_t)kNrFlagArrival >= (uint32_t)kNrFlagStage + (uint32_t)kNrMaxMembers &&
                         (uint32_t)kNrFlagArrival + (uint32_t)kNrMaxMembers <=
                             (uint32_t)kNrFlagsBytes / 4u,
                     "words 16..23 of 32");
    ft_tt_test_check("step clocks: the reader's (24, 25) and the writer's (28, 29) on their own 16-byte "
                     "lines, inside the page",
                     (uint32_t)kNrFlagAllSeen / 4u == (uint32_t)kNrFlagSqAllWait / 4u &&
                         (uint32_t)kNrFlagOutFirst / 4u == (uint32_t)kNrFlagOutDone / 4u &&
                         (uint32_t)kNrFlagAllSeen / 4u != (uint32_t)kNrFlagOutFirst / 4u &&
                         (uint32_t)kNrFlagAllSeen >= (uint32_t)kNrFlagArrival + (uint32_t)kNrMaxMembers &&
                         (uint32_t)kNrFlagOutDone < (uint32_t)kNrFlagsBytes / 4u,
                     "24-25 / 28-29 of 32");
    ft_tt_test_check("the reader times every flag's first sighting on every poll",
                     reader && strstr(reader, "seen = note_arrivals(ctx, role, members, seen);") &&
                         strstr(reader, "role->flags[kNrFlagArrival + k] +="),
                     "resident_norm_role_reader.c push_exchange");
    free(reader);
}

int main(void)
{
    check_wire();
    check_targets();
    check_placement();
    check_exchange_model();
    check_teardown();
    check_flag_alignment();
    check_verbs();
    check_cbtables();
    check_numerics();
    check_exchange_levers();
    check_exchange_modes();
    return ft_tt_test_finish();
}
