/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_swiglu_fusion.c
 * @brief The fused-SwiGLU matmul (program `rtriad_bfp8_swiglu`), off hardware.
 *
 *   1. LLK SEQUENCE. The SILU_MUL half of the fused compute
 *      (kernels/resident_matmul_swiglu_compute.h, ftsw_silu_gated_multiply) is extracted from
 *      its SOURCE and compared, verb by verb, thread by thread, with the format / fidelity /
 *      geometry tokens of every argument, against eltwise SILU_MUL's own
 *      (kernels/resident_eltwise_compute.c, apply_silu_gated_multiply) — helpers expanded on
 *      both sides. The only accepted difference is the DST commit going through
 *      resident_matmul_dst_sync.h. The CB tables are checked too: every CB both sequences
 *      touch is Float16_b 32x32, and fidelity / approx / DST mode agree with eltwise's and
 *      matmul_bfp8_b's. Built-in negative controls: a dropped staging pack, a swapped
 *      init/apply and a changed fidelity must each be reported.
 *   2. NUMERICS AND ORDER. A host model of the fused descriptor — the protocol header's own
 *      pass -> (panel, column) map, the panel CB as a RING whose pointers persist across
 *      descriptors and whose indexed unpacks do not wrap (card fix 1: NB 8, 3 descriptors a
 *      member), the writer's tile order — against
 *      gate -> bf16, up -> bf16, silu -> bf16, MUL -> bf16 over whole strips, at every member
 *      count / ragged width / column block swept. Must be bit-identical; the no-stage RED must
 *      differ by bytes on the same data.
 *   3. FAN-OUT. The writer's replica lookup (ftr_resolve_destinations, the kernel's own) over
 *      a range record the way Pantheon builds it: every member's slice of every band lands in
 *      every mirror exactly once, nothing outside the strip is written, a descriptor outside
 *      the strip is not fanned out.
 *
 *   RED BUILDS (each must FAIL): `make -f Makefile.swiglu RED=no_stage run` (the model skips
 *   the bf16 staging), `RED=mirror_skip` (the table drops one mirror), `RED=panel_wrap` (the
 *   first, card-failed panel layout: w pages a pass, up at w + j). A red SOURCE run:
 *   FT_TT_KERNEL_DIR=<dir with a mutated resident_matmul_swiglu_compute.h> (see the report).
 *
 * @status Added 2026-11-17.
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_op_replicas.h"
#include "ft_tt_test_report.h"
#include "kernels/resident_matmul_swiglu_protocol.h"

/* ===================================================================================
 * 1. LLK sequence, from the sources
 * =================================================================================== */

#define LLK_MAX_CALLS 96
#define LLK_TEXT 192

typedef struct llk_call {
    char thread;              /* U, M, P, or * (outside a thread wrapper) */
    char name[64];
    char args[LLK_TEXT];      /* normalised: format/fidelity/geometry tokens kept */
} llk_call;

typedef struct llk_sequence {
    llk_call call[LLK_MAX_CALLS];
    int count;
} llk_sequence;

static const char *kernel_dir(void)
{
    const char *directory = getenv("FT_TT_KERNEL_DIR");

    return directory && *directory ? directory : "../kernels";
}

static char *read_text(const char *name)
{
    char path[512];
    FILE *file;
    long size;
    char *text;

    snprintf(path, sizeof(path), "%s/%s", kernel_dir(), name);
    file = fopen(path, "rb");
    if (!file)
        return NULL;
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    text = malloc((size_t)size + 1u);
    if (text && fread(text, 1u, (size_t)size, file) != (size_t)size) {
        free(text);
        text = NULL;
    }
    if (text)
        text[size] = '\0';
    fclose(file);
    return text;
}

static bool is_ident_start(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

static bool is_ident(char c)
{
    return is_ident_start(c) || (c >= '0' && c <= '9');
}

/** The matching close of the bracket at `open`, or NULL. */
static const char *match_close(const char *open, char o, char c)
{
    int depth = 0;
    const char *p;

    for (p = open; *p; p++) {
        if (*p == o)
            depth++;
        else if (*p == c && --depth == 0)
            return p;
    }
    return NULL;
}

/** The body `{...}` of `static ... name(...)` defined in `text`, or NULL. */
static const char *function_body(const char *text, const char *name, const char **end)
{
    const size_t length = strlen(name);
    const char *p = text;

    while ((p = strstr(p, name)) != NULL) {
        const char *line = p;
        const char *close;
        const char *q;

        while (line > text && line[-1] != '\n')
            line--;
        if ((p > text && is_ident(p[-1])) || p[length] != '(' || strncmp(line, "static", 6) != 0) {
            p += length;
            continue;
        }
        close = match_close(p + length, '(', ')');
        if (!close)
            return NULL;
        for (q = close + 1; *q == ' ' || *q == '\n' || *q == '\t'; q++) {
        }
        if (*q == '{') {
            *end = match_close(q, '{', '}');
            return *end ? q : NULL;
        }
        p = close;
    }
    return NULL;
}

/** Arguments with every token but FT_* / ft_cb_* / ft_tensix_* names and operators made `_`. */
static void normalise_args(const char *from, const char *to, char *out, size_t size)
{
    size_t n = 0u;
    const char *p = from;

    while (p < to && n + 1u < size) {
        if (is_ident_start(*p) || (*p >= '0' && *p <= '9')) {
            const char *start = p;
            const size_t keep_room = size - n - 1u;

            while (p < to && is_ident(*p))
                p++;
            if ((strncmp(start, "FT_", 3) == 0 || strncmp(start, "ft_cb_", 6) == 0 ||
                 strncmp(start, "ft_tensix_", 10) == 0) &&
                (size_t)(p - start) <= keep_room) {
                memcpy(out + n, start, (size_t)(p - start));
                n += (size_t)(p - start);
            } else {
                out[n++] = '_';
            }
            continue;
        }
        if (*p != ' ' && *p != '\n' && *p != '\t')
            out[n++] = *p;
        p++;
    }
    out[n] = '\0';
}

/** The DST commit through resident_matmul_dst_sync.h is the one accepted substitution. */
static const char *normalise_name(const char *name)
{
    if (strcmp(name, "ftmm_math_dest_section_done") == 0)
        return "ft_tensix_math_dest_section_done";
    if (strcmp(name, "ftmm_pack_dest_section_done") == 0)
        return "ft_tensix_pack_dest_section_done";
    return name;
}

static bool in_list(const char *name, const char *const *list, int count)
{
    int i;

    for (i = 0; i < count; i++)
        if (strcmp(name, list[i]) == 0)
            return true;
    return false;
}

/** Record the LLK calls of [from, to) in order, expanding `helpers` from `text`. */
static void extract_calls(const char *text, const char *from, const char *to,
                          const char *const *helpers, int helper_count, char thread,
                          llk_sequence *out, int depth)
{
    const char *p = from;

    while (p < to) {
        const char *start;
        const char *q;
        const char *close;
        char name[64];
        size_t length;

        if (!is_ident_start(*p) || (p > from && is_ident(p[-1]))) {
            p++;
            continue;
        }
        start = p;
        while (p < to && is_ident(*p))
            p++;
        length = (size_t)(p - start);
        for (q = p; q < to && (*q == ' ' || *q == '\n'); q++) {
        }
        if (q >= to || *q != '(' || length >= sizeof(name))
            continue;
        memcpy(name, start, length);
        name[length] = '\0';
        close = match_close(q, '(', ')');
        if (!close || close > to)
            return;
        if (strcmp(name, "FT_TENSIX_UNPACK") == 0 || strcmp(name, "FT_TENSIX_MATH") == 0 ||
            strcmp(name, "FT_TENSIX_PACK") == 0) {
            extract_calls(text, q + 1, close, helpers, helper_count, name[10], out, depth);
            p = close + 1;
            continue;
        }
        if (in_list(name, helpers, helper_count) && depth < 4) {
            const char *end = NULL;
            const char *body = function_body(text, name, &end);

            if (body)
                extract_calls(text, body + 1, end, helpers, helper_count, thread, out,
                              depth + 1);
            p = close + 1;
            continue;
        }
        if ((strncmp(name, "ft_tensix_", 10) == 0 || strncmp(name, "ftmm_", 5) == 0 ||
             strncmp(name, "ftq_cb_", 7) == 0) &&
            out->count < LLK_MAX_CALLS) {
            llk_call *call = &out->call[out->count++];

            call->thread = thread;
            snprintf(call->name, sizeof(call->name), "%s", normalise_name(name));
            normalise_args(q + 1, close, call->args, sizeof(call->args));
            p = close + 1;
        }
    }
}

static bool extract_function(const char *text, const char *name, const char *const *helpers,
                             int helper_count, llk_sequence *out)
{
    const char *end = NULL;
    const char *body = text ? function_body(text, name, &end) : NULL;

    memset(out, 0, sizeof(*out));
    if (!body)
        return false;
    extract_calls(text, body + 1, end, helpers, helper_count, '*', out, 0);
    return out->count > 0;
}

/** -1 when equal, else the index of the first difference. */
static int sequence_mismatch(const llk_sequence *a, const llk_sequence *b)
{
    int i;
    const int n = a->count < b->count ? a->count : b->count;

    for (i = 0; i < n; i++)
        if (a->call[i].thread != b->call[i].thread || strcmp(a->call[i].name, b->call[i].name) != 0 ||
            strcmp(a->call[i].args, b->call[i].args) != 0)
            return i;
    return a->count == b->count ? -1 : n;
}

static int find_call(const llk_sequence *s, const char *name, int from)
{
    int i;

    for (i = from; i < s->count; i++)
        if (strcmp(s->call[i].name, name) == 0)
            return i;
    return -1;
}

/** A CB table line `cb <index> <rest>` for `index`, or "" if absent. */
static void cbtable_line(const char *text, unsigned index, char *out, size_t size)
{
    char want[16];
    const char *p = text;

    out[0] = '\0';
    snprintf(want, sizeof(want), "cb %2u ", index);
    while (p && *p) {
        const char *eol = strchr(p, '\n');
        const size_t length = eol ? (size_t)(eol - p) : strlen(p);

        if (strncmp(p, want, strlen(want)) == 0 && length < size) {
            memcpy(out, p + strlen(want), length - strlen(want));
            out[length - strlen(want)] = '\0';
            return;
        }
        p = eol ? eol + 1 : NULL;
    }
}

/** The value word of `key` in a CB table, or -1. */
static long cbtable_setting(const char *text, const char *key)
{
    const char *p = text;
    const size_t length = strlen(key);

    while (p && *p) {
        if (strncmp(p, key, length) == 0 && p[length] == ' ')
            return strtol(p + length, NULL, 10);
        p = strchr(p, '\n');
        p = p ? p + 1 : NULL;
    }
    return -1;
}

static void test_cb_tables(void)
{
    char *eltwise = read_text("cb_tables/eltwise_float16b.cbtable");
    char *matmul = read_text("cb_tables/matmul_bfp8_b.cbtable");
    char *fused = read_text("cb_tables/matmul_bfp8_b_swiglu.cbtable");
    static const char *const settings[3] = { "math_fidelity", "approx", "dst_accum_mode" };
    static const unsigned bf16_cbs[3] = { FT_TT_MATMUL_SWIGLU_CB_STAGE, FT_TT_MATMUL_SWIGLU_CB_PANEL,
                                          16u };
    static const unsigned same_cbs[3] = { 0u, 1u, 16u };
    char a[96], b[96];
    int i;

    ft_tt_test_step("CB tables: formats, fidelity, DST mode");
    ft_tt_test_check("the three tables read", eltwise && matmul && fused, kernel_dir());
    if (!eltwise || !matmul || !fused)
        goto done;
    for (i = 0; i < 3; i++) {
        const long e = cbtable_setting(eltwise, settings[i]);
        const long m = cbtable_setting(matmul, settings[i]);
        const long f = cbtable_setting(fused, settings[i]);

        ft_tt_test_check_fmt(settings[i], e >= 0 && e == m && m == f,
                             "eltwise %ld, matmul_bfp8_b %ld, swiglu %ld", e, m, f);
    }
    for (i = 0; i < 3; i++) {
        cbtable_line(fused, bf16_cbs[i], a, sizeof(a));
        cbtable_line(eltwise, 2u, b, sizeof(b));
        ft_tt_test_check_fmt("a SILU_MUL CB is eltwise's bf16 32x32 tile", a[0] && strcmp(a, b) == 0,
                             "swiglu cb %u '%s', eltwise cb 2 '%s'", bf16_cbs[i], a, b);
    }
    for (i = 0; i < 3; i++) {
        cbtable_line(fused, same_cbs[i], a, sizeof(a));
        cbtable_line(matmul, same_cbs[i], b, sizeof(b));
        ft_tt_test_check_fmt("a matmul CB is matmul_bfp8_b's", a[0] && strcmp(a, b) == 0,
                             "cb %u: swiglu '%s', matmul_bfp8_b '%s'", same_cbs[i], a, b);
    }
done:
    free(eltwise);
    free(matmul);
    free(fused);
}

static void test_llk_sequence(void)
{
    static const char *const eltwise_helpers[5] = { "eltwise_copy_init", "eltwise_copy_apply",
                                                    "eltwise_mul_init", "eltwise_mul_apply",
                                                    "commit_dst_to" };
    static const char *const fused_helpers[5] = { "ftsw_copy_init", "ftsw_copy_apply",
                                                  "ftsw_mul_init", "ftsw_mul_apply",
                                                  "ftsw_commit_dst_to" };
    char *eltwise_text = read_text("resident_eltwise_compute.c");
    char *fused_text = read_text("resident_matmul_swiglu_compute.h");
    llk_sequence eltwise, fused, mutated;
    int at, pack;

    ft_tt_test_step("LLK sequence: fused SILU_MUL against eltwise SILU_MUL, from the sources");
    ft_tt_test_check("eltwise apply_silu_gated_multiply extracted",
                     extract_function(eltwise_text, "apply_silu_gated_multiply", eltwise_helpers, 5,
                                      &eltwise),
                     kernel_dir());
    ft_tt_test_check("fused ftsw_silu_gated_multiply extracted",
                     extract_function(fused_text, "ftsw_silu_gated_multiply", fused_helpers, 5,
                                      &fused),
                     kernel_dir());
    at = sequence_mismatch(&eltwise, &fused);
    ft_tt_test_check_fmt("same verbs, threads and format/fidelity tokens, in order", at < 0,
                         "%d calls each; first difference at %d: eltwise %c %s(%s) / fused %c %s(%s)",
                         eltwise.count, at, at >= 0 && at < eltwise.count ? eltwise.call[at].thread : '-',
                         at >= 0 && at < eltwise.count ? eltwise.call[at].name : "-",
                         at >= 0 && at < eltwise.count ? eltwise.call[at].args : "-",
                         at >= 0 && at < fused.count ? fused.call[at].thread : '-',
                         at >= 0 && at < fused.count ? fused.call[at].name : "-",
                         at >= 0 && at < fused.count ? fused.call[at].args : "-");
    pack = find_call(&fused, "ft_tensix_pack_tile", 0);
    ft_tt_test_check_fmt("two bf16 pack points (the stage, then the output)",
                         pack >= 0 && find_call(&fused, "ft_tensix_pack_tile", pack + 1) > pack,
                         "first pack at call %d", pack);
    at = find_call(&fused, "ft_tensix_math_eltwise_binary_mul_init", 0);
    ft_tt_test_check_fmt("the multiply runs at the table's fidelity",
                         at >= 0 && strstr(fused.call[at].args, "FT_CB_MATH_FIDELITY") != NULL,
                         "%s", at >= 0 ? fused.call[at].args : "(no multiply init)");
    at = find_call(&fused, "ft_tensix_math_silu_tile_row0", 0);
    ft_tt_test_check_fmt("silu is eltwise's own SFPU silu (row 0 and full)",
                         at >= 0 && find_call(&fused, "ft_tensix_math_silu_tile", at + 1) > at,
                         "row0 at call %d", at);

    ft_tt_test_step("LLK sequence: negative controls (each must be reported)");
    mutated = fused;
    if (pack >= 0) {
        memmove(&mutated.call[pack], &mutated.call[pack + 1],
                (size_t)(mutated.count - pack - 1) * sizeof(mutated.call[0]));
        mutated.count--;
    }
    ft_tt_test_check("a dropped staging pack is seen", sequence_mismatch(&eltwise, &mutated) >= 0,
                     "");
    mutated = fused;
    at = find_call(&mutated, "ft_tensix_math_eltwise_binary_mul_init", 0);
    if (at >= 0 && at + 1 < mutated.count) {
        const llk_call keep = mutated.call[at];

        mutated.call[at] = mutated.call[at + 1];
        mutated.call[at + 1] = keep;
    }
    ft_tt_test_check("a swapped init/apply is seen", sequence_mismatch(&eltwise, &mutated) >= 0, "");
    mutated = fused;
    if (at >= 0) {
        char *fidelity = strstr(mutated.call[at].args, "FT_CB_MATH_FIDELITY");

        if (fidelity)
            memcpy(fidelity, "FT_CB_MATH_FIDELITX", 19u);
    }
    ft_tt_test_check("a changed fidelity is seen", sequence_mismatch(&eltwise, &mutated) >= 0, "");
    free(eltwise_text);
    free(fused_text);
}

/** The fused compute (resident_matmul_compute.c) uses the layout the model below checks:
 *  NB-page passes with pad pages, up at NB + j, whole 2*NB-page blocks waited and popped. */
static void test_panel_layout_source(void)
{
    char *text = read_text("resident_matmul_compute.c");
    const char *end = NULL;
    const char *pass = text ? function_body(text, "ftsw_matmul_pass", &end) : NULL;
    const char *pass_end = end;
    const char *block = text ? function_body(text, "compute_one_swiglu", &end) : NULL;
    char *pass_text = NULL, *block_text = NULL;

    ft_tt_test_step("Panel layout in the fused compute's source (card fix 1)");
    if (pass && pass_end) {
        pass_text = calloc((size_t)(pass_end - pass) + 1u, 1u);
        if (pass_text)
            memcpy(pass_text, pass, (size_t)(pass_end - pass));
    }
    if (block && end) {
        block_text = calloc((size_t)(end - block) + 1u, 1u);
        if (block_text)
            memcpy(block_text, block, (size_t)(end - block));
    }
    ft_tt_test_check("ftsw_matmul_pass and compute_one_swiglu found", pass_text && block_text,
                     kernel_dir());
    ft_tt_test_check("a short pass pushes pad pages up to ftsw_panel_pass_pages(column_block)",
                     pass_text && strstr(pass_text, "ftsw_panel_pass_pages(column_block)"), "");
    ft_tt_test_check("the up tile is ftsw_panel_up_index(j, column_block)",
                     block_text && strstr(block_text, "ftsw_panel_up_index(j, column_block)") &&
                         !strstr(block_text, "ftsw_panel_up_index(j, columns)"),
                     "");
    ft_tt_test_check("whole blocks are waited and popped: ftsw_panel_pages(column_block)",
                     block_text && strstr(block_text, "ftsw_panel_pages(column_block), wait_mode") &&
                         strstr(block_text, "pop_front(cb_panel, ftsw_panel_pages(column_block))"),
                     "");
    free(pass_text);
    free(block_text);
    free(text);
}

/* ===================================================================================
 * 2. Numerics and order: the fused descriptor against the unfused ops
 * =================================================================================== */

#define TILE_W 32u /* columns of a tile row (decode: row 0 is the live one) */
#define SWIGLU_DESCRIPTORS 3u /* descriptors per member, back to back on one panel ring */
/* The panel layout the kernel uses: pages a pass occupies, and the up tile's index. RED
 * `panel_wrap` is the first (card-failed) layout: a pass took w pages, up at w + j. */
#ifdef SWIGLU_TEST_RED_PANEL_WRAP
#define SWIGLU_PASS_PAGES(w, nb) (w)
#define SWIGLU_UP_INDEX(j, w, nb) ((w) + (j))
#else
#define SWIGLU_PASS_PAGES(w, nb) ftsw_panel_pass_pages(nb)
#define SWIGLU_UP_INDEX(j, w, nb) ftsw_panel_up_index((j), (nb))
#endif

static uint16_t bf16_round(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000u) == 0x7F800000u)
        return (uint16_t)(bits >> 16);
    bits += 0x7FFFu + ((bits >> 16) & 1u); /* round to nearest even, as the packer */
    return (uint16_t)(bits >> 16);
}

static float bf16_value(uint16_t half)
{
    const uint32_t bits = (uint32_t)half << 16;
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/** The SFPU silu stand-in (both paths call the same one; the kernels share the SFPU code). */
static float silu_model(float x)
{
    return x / (1.0f + expf(-x));
}

/** One column's matmul in fp32 (the shared FPU stand-in), packed bf16. */
static uint16_t matmul_column(const float *a, const float *w_column, uint32_t k)
{
    float sum = 0.0f;
    uint32_t i;

    for (i = 0; i < k; i++)
        sum += a[i] * w_column[i];
    return bf16_round(sum);
}

static uint32_t g_rng = 0x9E3779B9u;

static float random_value(void)
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float)((int32_t)(g_rng >> 8) - (1 << 23)) / (float)(1 << 21);
}

/** Pantheon's split of `nt` tile columns over `members`: widths within one, running first. */
static void shard_map(uint32_t nt, uint32_t members, uint32_t *cols, uint32_t *first)
{
    uint32_t m, running = 0u;

    for (m = 0; m < members; m++) {
        cols[m] = nt / members + (m < nt % members ? 1u : 0u);
        first[m] = running;
        running += cols[m];
    }
}

/** silu(g) staged bf16 (or, RED, not), times u, packed bf16. */
static uint16_t silu_mul(uint16_t g, uint16_t u, bool stage)
{
    const float s = silu_model(bf16_value(g));
    const float staged = stage ? bf16_value(bf16_round(s)) : s;

    return bf16_round(staged * bf16_value(u));
}

/* THE PANEL CB AS THE CARD HAS IT: a ring of `pages` tile slots whose read and write
 * pointers PERSIST from one descriptor to the next (a member runs every layer's FFN), and an
 * indexed unpack (ft_tensix_unpack_a / _ab: read pointer + page * index) that does NOT wrap — an
 * index past the ring's end reads whatever L1 follows the CB. Modelled as POISON. */
#define PANEL_MAX_PAGES 16u
#define PANEL_POISON 0x7FC1u

typedef struct panel_ring {
    uint16_t slot[PANEL_MAX_PAGES][TILE_W];
    uint32_t pages, read, write; /* pointers in pages, wrapped on push / pop as the CB does */
} panel_ring;

static void panel_push(panel_ring *ring, const uint16_t *tile)
{
    if (tile)
        memcpy(ring->slot[ring->write], tile, sizeof(ring->slot[0]));
    ring->write = (ring->write + 1u) % ring->pages;
}

static const uint16_t *panel_at(const panel_ring *ring, uint32_t index)
{
    static uint16_t poison[TILE_W];
    uint32_t lane;

    if (ring->read + index < ring->pages)
        return ring->slot[ring->read + index];
    for (lane = 0; lane < TILE_W; lane++)
        poison[lane] = PANEL_POISON;
    return poison;
}

/**
 * One member's FUSED descriptor, walked the way the kernels walk it: the reader's passes
 * (protocol header), each pass's tiles packed into the panel ring (pad pages included), the
 * SILU_MUL half on the ring's (gate, up) indices, the block popped, the writer placing tile
 * nt + j at the member's first column.
 */
static void fused_member(const float *a, uint32_t k, const float *wg, const float *wu,
                         uint32_t first, uint32_t cols, uint32_t nb, bool stage,
                         panel_ring *ring, uint16_t *out)
{
    const uint32_t passes = ftsw_passes(cols, nb);
    uint16_t tile[TILE_W];
    uint32_t pass, column, j, lane;

    for (pass = 0; pass < passes; pass++) {
        const float *panel = ftsw_pass_panel(pass) ? wu : wg;
        const uint32_t nt = ftsw_pass_column(pass, nb);
        const uint32_t w = ftsw_pass_columns(pass, cols, nb);

        for (column = 0; column < w; column++) {
            for (lane = 0; lane < TILE_W; lane++)
                tile[lane] = matmul_column(
                    a, panel + ((size_t)(first + nt + column) * TILE_W + lane) * k, k);
            panel_push(ring, tile);
        }
        for (column = w; column < SWIGLU_PASS_PAGES(w, nb); column++)
            panel_push(ring, NULL);
        if (!ftsw_pass_panel(pass))
            continue;
        for (j = 0; j < w; j++) {
            const uint16_t *g = panel_at(ring, ftsw_panel_gate_index(j));
            const uint16_t *u = panel_at(ring, SWIGLU_UP_INDEX(j, w, nb));

            for (lane = 0; lane < TILE_W; lane++)
                out[(size_t)(first + nt + j) * TILE_W + lane] =
                    g[lane] == PANEL_POISON || u[lane] == PANEL_POISON
                        ? PANEL_POISON
                        : silu_mul(g[lane], u[lane], stage);
        }
        ring->read = (ring->read + 2u * SWIGLU_PASS_PAGES(w, nb)) % ring->pages;
    }
}

/** 0 equal, else the count of differing outputs. Runs fused (stage as given) vs unfused. */
static uint32_t fused_vs_unfused(uint32_t nt, uint32_t members, uint32_t nb, uint32_t kt,
                                 bool stage)
{
    const uint32_t k = kt * 32u, n = nt * TILE_W;
    float *a = malloc(k * sizeof(*a));
    float *wg = malloc((size_t)n * k * sizeof(*wg));
    float *wu = malloc((size_t)n * k * sizeof(*wu));
    uint16_t *ref = calloc(n, sizeof(*ref));
    uint16_t *got = calloc(n, sizeof(*got));
    uint32_t cols[16], first[16], m, i, differ = 0u;

    if (!a || !wg || !wu || !ref || !got || members > 16u) {
        differ = 0xFFFFFFFFu;
        goto done;
    }
    for (i = 0; i < k; i++)
        a[i] = random_value();
    for (i = 0; i < n * k; i++) {
        wg[i] = random_value() / 8.0f;
        wu[i] = random_value() / 8.0f;
    }
    /* UNFUSED: ffn_gate and ffn_up whole (bf16), then SILU_MUL over the strip. */
    for (i = 0; i < n; i++)
        ref[i] = silu_mul(matmul_column(a, wg + (size_t)i * k, k),
                          matmul_column(a, wu + (size_t)i * k, k), true);
    shard_map(nt, members, cols, first);
    /* SEVERAL DESCRIPTORS per member on one panel ring: the ring pointers carry over, as they
     * do from layer to layer on the card. */
    for (m = 0; m < members; m++) {
        panel_ring *ring = calloc(1, sizeof(*ring));
        uint32_t descriptor;

        if (!ring) {
            differ = 0xFFFFFFFFu;
            goto done;
        }
        ring->pages = ftsw_panel_pages(nb);
        for (descriptor = 0; cols[m] && descriptor < SWIGLU_DESCRIPTORS; descriptor++) {
            fused_member(a, k, wg, wu, first[m], cols[m], nb, stage, ring, got);
            for (i = first[m] * TILE_W; i < (first[m] + cols[m]) * TILE_W; i++)
                differ += ref[i] != got[i] ? 1u : 0u;
        }
        free(ring);
    }
done:
    free(a);
    free(wg);
    free(wu);
    free(ref);
    free(got);
    return differ;
}

static void test_numerics(void)
{
#ifdef SWIGLU_TEST_RED_NO_STAGE
    const bool stage = false; /* RED BUILD: the model skips the bf16 staging; must FAIL */
#else
    const bool stage = true;
#endif
    static const uint32_t member_counts[3] = { 1u, 3u, 16u };
    uint32_t mc, nt, nb, cases = 0u, failed = 0u, last_differ = 0u;

    ft_tt_test_step("Numerics: fused (pass order, panel FIFO, writer order) vs unfused ops");
    for (mc = 0; mc < 3u; mc++)
        for (nt = 1u; nt <= 9u; nt++)
            for (nb = 1u; nb <= 8u; nb++) {
                const uint32_t differ = fused_vs_unfused(nt, member_counts[mc], nb, 2u, stage);

                cases++;
                if (differ != 0u) {
                    failed++;
                    last_differ = differ;
                }
            }
    ft_tt_test_check_fmt("bit-identical at members 1/3/16, Nt 1..9, NB 1..8, 3 descriptors a member", failed == 0u,
                         "%u of %u cases differ (last: %u outputs)", failed, cases, last_differ);
    last_differ = fused_vs_unfused(544u, 16u, 4u, 2u, stage);
    ft_tt_test_check_fmt("bit-identical at Qwen3.8's shape, NB 4 (16 members x 34, 3 descriptors)",
                         last_differ == 0u, "%u of %u outputs differ", last_differ,
                         544u * TILE_W * SWIGLU_DESCRIPTORS);
    last_differ = fused_vs_unfused(544u, 16u, 8u, 2u, stage);
    ft_tt_test_check_fmt("bit-identical at Qwen3.8's card shape, NB 8 SyncFull (the served one)",
                         last_differ == 0u, "%u of %u outputs differ", last_differ,
                         544u * TILE_W * SWIGLU_DESCRIPTORS);
    last_differ = fused_vs_unfused(544u, 16u, 4u, 2u, false);
    ft_tt_test_check_fmt("the no-stage RED differs by bytes on the same data", last_differ > 0u,
                         "%u of %u outputs differ", last_differ, 544u * TILE_W * SWIGLU_DESCRIPTORS);
}

/* ===================================================================================
 * 3. Fan-out: the writer's lookup over the members' range record
 * =================================================================================== */

#define MIRRORS 8u
#define STRIP_NT 544u
#define STRIP_BANDS 2u
#define STRIP_BYTES (STRIP_BANDS * STRIP_NT * 2048u)

static void test_fan_out(void)
{
    ft_tt_op_replica_table table;
    uint32_t words[FT_TT_REPLICA_TABLE_WORDS];
    static uint8_t landed[MIRRORS][STRIP_BANDS * STRIP_NT];
    uint32_t cols[16], first[16], band, m, t, d, i;
    uint32_t misses = 0u, strays = 0u, wrong_count = 0u, holes = 0u, doubles = 0u;
    uint32_t x[FT_TT_REPLICA_MAX_DESTINATIONS], y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t base[FT_TT_REPLICA_MAX_DESTINATIONS];
    int rc;

    ft_tt_test_step("Fan-out: every member's slice of every band in every mirror, once");
    memset(&table, 0, sizeof(table));
    memset(landed, 0, sizeof(landed));
    table.set_count = 1u;
    /* As Pantheon's replica_map_fused_table(): keyed at band 0, over every band. */
    table.set[0].count = MIRRORS;
    table.set[0].range_bytes = STRIP_BYTES;
    for (d = 0; d < MIRRORS; d++) {
        table.set[0].destination[d].x = (uint16_t)(30u + d);
        table.set[0].destination[d].y = 1u;
        table.set[0].destination[d].base = 0x01000000u * (d + 1u) + 0x2000u * d;
    }
#ifdef SWIGLU_TEST_RED_MIRROR_SKIP
    table.set[0].count = MIRRORS - 1u; /* RED BUILD: one mirror never written; must FAIL */
#endif
    rc = ft_tt_op_replica_table_encode(&table, words);
    ft_tt_test_check_fmt("the range record encodes", rc == 0, "rc %d", rc);
    shard_map(STRIP_NT, 16u, cols, first);
    for (band = 0; band < STRIP_BANDS; band++)
        for (m = 0; m < 16u; m++) {
            /* The member's descriptor: C = the strip's seeked band + its first column. */
            const uint32_t c_base = table.set[0].destination[0].base +
                                    band * STRIP_NT * 2048u + first[m] * 2048u;
            const uint32_t fan = ftr_resolve_destinations(words, 30u, 1u, c_base, x, y, base);

            if (fan != MIRRORS)
                wrong_count++;
            for (t = 0; t < cols[m]; t++)
                for (d = 0; d < fan; d++) {
                    uint32_t mirror = MIRRORS, tile;

                    for (i = 0; i < MIRRORS; i++)
                        if (x[d] == table.set[0].destination[i].x && y[d] == 1u)
                            mirror = i;
                    if (mirror == MIRRORS ||
                        base[d] + t * 2048u < table.set[0].destination[mirror].base) {
                        strays++;
                        continue;
                    }
                    tile = (base[d] + t * 2048u - table.set[0].destination[mirror].base) / 2048u;
                    if (tile >= STRIP_BANDS * STRIP_NT) {
                        strays++;
                        continue;
                    }
                    landed[mirror][tile]++;
                }
        }
    for (d = 0; d < MIRRORS; d++)
        for (t = 0; t < STRIP_BANDS * STRIP_NT; t++) {
            holes += landed[d][t] == 0u ? 1u : 0u;
            doubles += landed[d][t] > 1u ? 1u : 0u;
        }
    ft_tt_test_check_fmt("every member's descriptor resolves to all 8 copies", wrong_count == 0u,
                         "%u of %u descriptors resolved short", wrong_count, 16u * STRIP_BANDS);
    ft_tt_test_check_fmt("every tile of every band lands in every mirror", holes == 0u,
                         "%u (mirror, tile) slots never written", holes);
    ft_tt_test_check_fmt("no tile lands twice, none outside the strip", doubles == 0u && strays == 0u,
                         "%u doubled, %u stray", doubles, strays);
    misses += ftr_resolve_destinations(words, 30u, 1u, table.set[0].destination[0].base + STRIP_BYTES,
                                       x, y, base) != 1u ? 1u : 0u;
    misses += ftr_resolve_destinations(words, 30u, 1u, table.set[0].destination[0].base - 2048u,
                                       x, y, base) != 1u ? 1u : 0u;
    misses += ftr_resolve_destinations(words, 31u, 1u, table.set[0].destination[0].base, x, y,
                                       base) != 1u ? 1u : 0u;
    ft_tt_test_check_fmt("a destination outside the strip (past, before, other bank) is not fanned",
                         misses == 0u, "%u of 3 fanned out", misses);
}

int main(void)
{
    test_cb_tables();
    test_llk_sequence();
    test_panel_layout_source();
    test_numerics();
    test_fan_out();
    return ft_tt_test_finish();
}
