/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_resadd_fusion.c
 * @brief S3a, the residual ADD folded into the out-proj / down members (program
 *   rtriad_bfp8_swiglu built with -DFT_RTRIAD_RESADD=1), off the card:
 *   1. the word-15 bits are disjoint from the block word and the SwiGLU bits;
 *   2. the CB table: the residual CB is eltwise ADD's bf16 tile, fidelity/approx/DST mode equal;
 *   3. the ADD verbs, from the sources: ftra_addsub_init / ftra_addsub_apply / the commit are
 *      eltwise_addsub_init / eltwise_addsub_apply / commit_dst_to call for call, the ELWADD
 *      word is eltwise's verbatim, the residual is operand A (eltwise's cb_in0 = hidden), and
 *      the per-tile order (acquire, add, commit) is eltwise's;
 *   4. the compute (resident_matmul_compute.c) pads the panel to the whole 2*NB block, waits
 *      and pops what the protocol says, switches SrcA before the add and restores after;
 *   5. a host model of the PANEL and RESIDUAL CB rings across mixed SwiGLU / RESADD
 *      descriptors (pointers persist, indexed unpacks never wrap): no index past either CB.
 *   6. numerics: the fused tile, staged at the protocol's rounding points, equals the unfused
 *      matmul -> bf16 -> eltwise ADD -> bf16 for the served shapes (the RED skips the add).
 *
 *   RED=panel_nb (make -f Makefile.resadd RED=panel_nb run): the model gives a RESADD block NB
 *   panel pages instead of the protocol's 2*NB; check 5 must fail.
 *   FT_TT_KERNEL_DIR=<dir> reads the kernel sources from a mutated copy.
 *
 * @status Added with S3a (q38dec-p5-s3).
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_test_report.h"
#include "kernels/resident_matmul_block_protocol.h"
#include "kernels/resident_matmul_resadd_protocol.h"
#include "kernels/resident_matmul_swiglu_protocol.h"

/* ===== Source helpers (test_swiglu_fusion.c's, trimmed) ================================== */

#define LLK_MAX_CALLS 64
#define LLK_TEXT 192

typedef struct llk_call {
    char thread;
    char name[64];
    char args[LLK_TEXT];
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
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
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

/** The body `{...}` of `static ... name(...)` in `text`, or NULL. */
static const char *function_body(const char *text, const char *name, const char **end)
{
    const size_t length = strlen(name);
    const char *p = text;

    while (p && (p = strstr(p, name)) != NULL) {
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

/** Arguments with every token but FT_* / ft_cb_* / ft_tensix_* names made `_`. */
static void normalise_args(const char *from, const char *to, char *out, size_t size)
{
    size_t n = 0u;
    const char *p = from;

    while (p < to && n + 1u < size) {
        if (is_ident_start(*p) || (*p >= '0' && *p <= '9')) {
            const char *start = p;

            while (p < to && is_ident(*p))
                p++;
            if ((strncmp(start, "FT_", 3) == 0 || strncmp(start, "ft_cb_", 6) == 0 ||
                 strncmp(start, "ft_tensix_", 10) == 0) &&
                (size_t)(p - start) <= size - n - 1u) {
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

/** The LLK calls of [from, to) in order, through the thread wrappers. */
static void extract_calls(const char *from, const char *to, char thread, llk_sequence *out)
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
            extract_calls(q + 1, close, name[10], out);
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

static bool extract_function(const char *text, const char *name, llk_sequence *out)
{
    const char *end = NULL;
    const char *body = text ? function_body(text, name, &end) : NULL;

    memset(out, 0, sizeof(*out));
    if (!body)
        return false;
    extract_calls(body + 1, end, '*', out);
    return out->count > 0;
}

static int sequence_mismatch(const llk_sequence *a, const llk_sequence *b)
{
    int i;
    const int n = a->count < b->count ? a->count : b->count;

    for (i = 0; i < n; i++)
        if (a->call[i].thread != b->call[i].thread ||
            strcmp(a->call[i].name, b->call[i].name) != 0 ||
            strcmp(a->call[i].args, b->call[i].args) != 0)
            return i;
    return a->count == b->count ? -1 : n;
}

/** The body text of `name` as a NUL-terminated copy with whitespace removed, or NULL. */
static char *squeezed_body(const char *text, const char *name)
{
    const char *end = NULL;
    const char *body = text ? function_body(text, name, &end) : NULL;
    char *out;
    size_t n = 0u;
    const char *p;

    if (!body)
        return NULL;
    out = malloc((size_t)(end - body) + 2u);
    if (!out)
        return NULL;
    for (p = body; p <= end; p++)
        if (*p != ' ' && *p != '\n' && *p != '\t')
            out[n++] = *p;
    out[n] = '\0';
    return out;
}

/** Positions of `needles` in `hay`, each after the previous; true when all are found. */
static bool in_order(const char *hay, const char *const *needles, int count)
{
    const char *at = hay;
    int i;

    for (i = 0; at && i < count; i++) {
        at = strstr(at, needles[i]);
        if (at)
            at += strlen(needles[i]);
    }
    return at != NULL;
}

/** The value of `#define name` as one squeezed line, or "". */
static void define_value(const char *text, const char *name, char *out, size_t size)
{
    char want[96];
    const char *p;
    size_t n = 0u;

    out[0] = '\0';
    snprintf(want, sizeof(want), "#define %s", name);
    p = text ? strstr(text, want) : NULL;
    if (!p)
        return;
    for (p += strlen(want); *p && n + 1u < size; p++) {
        if (*p == '\\' && p[1] == '\n') {
            p++;
            continue;
        }
        if (*p == '\n')
            break;
        if (*p != ' ' && *p != '\t')
            out[n++] = *p;
    }
    out[n] = '\0';
}

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

/* ===== 1. Words =========================================================================== */

static void test_words(void)
{
    const uint32_t block_word = ftmm_block_word(1u, 8u);

    ft_tt_test_step("Words: the RESADD bits beside the block word and the SwiGLU bits");
    ft_tt_test_check_fmt("disjoint from the block word's 24 bits",
                         (FT_TT_MATMUL_RESADD_BITS & 0x00FFFFFFu) == 0u, "bits 0x%08x",
                         FT_TT_MATMUL_RESADD_BITS);
    ft_tt_test_check_fmt("disjoint from the SwiGLU bits",
                         (FT_TT_MATMUL_RESADD_BITS & FT_TT_MATMUL_SWIGLU_BITS) == 0u,
                         "resadd 0x%08x swiglu 0x%08x", FT_TT_MATMUL_RESADD_BITS,
                         FT_TT_MATMUL_SWIGLU_BITS);
    ft_tt_test_check_fmt("the block decode ignores them",
                         ftmm_column_block(block_word | FT_TT_MATMUL_RESADD_BITS) == 8u &&
                             ftmm_row_block(block_word | FT_TT_MATMUL_RESADD_BITS) == 1u,
                         "NB %u MB %u", ftmm_column_block(block_word | FT_TT_MATMUL_RESADD_BITS),
                         ftmm_row_block(block_word | FT_TT_MATMUL_RESADD_BITS));
    ft_tt_test_check_fmt("a2 round-trips a bank coordinate",
                         ftra_residual_x(ftra_residual_xy(17u, 12u)) == 17u &&
                             ftra_residual_y(ftra_residual_xy(17u, 12u)) == 12u,
                         "0x%08x", ftra_residual_xy(17u, 12u));
    ft_tt_test_check_fmt("the residual CB is not a CB the matmul or SwiGLU names",
                         FT_TT_MATMUL_RESADD_CB_RESIDUAL != 0u &&
                             FT_TT_MATMUL_RESADD_CB_RESIDUAL != 1u &&
                             FT_TT_MATMUL_RESADD_CB_RESIDUAL != 16u &&
                             FT_TT_MATMUL_RESADD_CB_RESIDUAL != FT_TT_MATMUL_SWIGLU_CB_STAGE &&
                             FT_TT_MATMUL_RESADD_CB_RESIDUAL != FT_TT_MATMUL_SWIGLU_CB_PANEL,
                         "cb %u", FT_TT_MATMUL_RESADD_CB_RESIDUAL);
}

/* ===== 2. CB table ======================================================================== */

static void test_cb_table(void)
{
    char *eltwise = read_text("cb_tables/eltwise_float16b.cbtable");
    char *fused = read_text("cb_tables/matmul_bfp8_b_swiglu.cbtable");
    static const char *const settings[3] = { "math_fidelity", "approx", "dst_accum_mode" };
    char a[96], b[96];
    int i;

    ft_tt_test_step("CB table: the residual CB and the unit settings are eltwise ADD's");
    ft_tt_test_check("both tables read", eltwise && fused, kernel_dir());
    if (eltwise && fused) {
        for (i = 0; i < 3; i++) {
            const long e = cbtable_setting(eltwise, settings[i]);
            const long f = cbtable_setting(fused, settings[i]);

            ft_tt_test_check_fmt(settings[i], e >= 0 && e == f, "eltwise %ld, fused %ld", e, f);
        }
        cbtable_line(fused, FT_TT_MATMUL_RESADD_CB_RESIDUAL, a, sizeof(a));
        cbtable_line(eltwise, 0u, b, sizeof(b));
        ft_tt_test_check_fmt("the residual CB is eltwise cb_in0's tile", a[0] && strcmp(a, b) == 0,
                             "fused cb %u '%s', eltwise cb 0 '%s'", FT_TT_MATMUL_RESADD_CB_RESIDUAL,
                             a, b);
        cbtable_line(fused, FT_TT_MATMUL_SWIGLU_CB_PANEL, a, sizeof(a));
        cbtable_line(eltwise, 1u, b, sizeof(b));
        ft_tt_test_check_fmt("the panel CB is eltwise cb_in1's tile", a[0] && strcmp(a, b) == 0,
                             "fused cb %u '%s', eltwise cb 1 '%s'", FT_TT_MATMUL_SWIGLU_CB_PANEL,
                             a, b);
    }
    free(eltwise);
    free(fused);
}

/* ===== 3. The ADD verbs ==================================================================== */

static void check_same(const char *what, const char *eltwise_text, const char *eltwise_name,
                       const char *fused_text, const char *fused_name)
{
    llk_sequence e, f;
    const bool got = extract_function(eltwise_text, eltwise_name, &e) &&
                     extract_function(fused_text, fused_name, &f);
    const int at = got ? sequence_mismatch(&e, &f) : 0;

    ft_tt_test_check_fmt(what, got && at < 0,
                         "%s %d calls / %s %d calls; first difference at %d: %s(%s) / %s(%s)",
                         eltwise_name, got ? e.count : -1, fused_name, got ? f.count : -1, at,
                         got && at >= 0 && at < e.count ? e.call[at].name : "-",
                         got && at >= 0 && at < e.count ? e.call[at].args : "-",
                         got && at >= 0 && at < f.count ? f.call[at].name : "-",
                         got && at >= 0 && at < f.count ? f.call[at].args : "-");
}

static void test_add_verbs(void)
{
    char *eltwise = read_text("resident_eltwise_compute.c");
    char *resadd = read_text("resident_matmul_resadd_compute.h");
    char *swiglu = read_text("resident_matmul_swiglu_compute.h");
    char word_e[192], word_f[192];
    char *body;

    ft_tt_test_step("ADD verbs: the fused add against eltwise ADD, from the sources");
    ft_tt_test_check("the three sources read", eltwise && resadd && swiglu, kernel_dir());
    if (!eltwise || !resadd || !swiglu)
        goto done;
    check_same("init: ftra_addsub_init is eltwise_addsub_init", eltwise, "eltwise_addsub_init",
               resadd, "ftra_addsub_init");
    check_same("apply: ftra_addsub_apply is eltwise_addsub_apply", eltwise, "eltwise_addsub_apply",
               resadd, "ftra_addsub_apply");
    check_same("commit: ftsw_commit_dst_to is commit_dst_to", eltwise, "commit_dst_to", swiglu,
               "ftsw_commit_dst_to");
    define_value(eltwise, "ELTWISE_ADD_WORD", word_e, sizeof(word_e));
    define_value(resadd, "FTRA_ELTWISE_ADD_WORD", word_f, sizeof(word_f));
    ft_tt_test_check_fmt("the ELWADD word is eltwise's verbatim",
                         word_e[0] && strcmp(word_e, word_f) == 0, "eltwise '%s' fused '%s'",
                         word_e, word_f);
    /* Eltwise: the ADD init names cb_in0 and the apply reads (cb_in0, cb_in1); per tile it
     * acquires DST, applies, commits to cb_out. pn_tt_add(hidden, project) loads hidden into
     * cb_in0, so the residual is operand A. */
    body = squeezed_body(eltwise, "configure_for_function");
    ft_tt_test_check("eltwise ADD init names cb_in0",
                     body && strstr(body, "casekVecFnAdd:"
                                          "eltwise_addsub_init(ELTWISE_ADD_WORD,cb_in0);"),
                     "");
    free(body);
    body = squeezed_body(eltwise, "apply_to_dst");
    ft_tt_test_check("eltwise ADD apply reads (cb_in0, cb_in1)",
                     body && strstr(body, "casekVecFnAdd:eltwise_addsub_apply(cb_in0,cb_in1);"),
                     "");
    free(body);
    body = squeezed_body(eltwise, "compute_one_eltwise");
    {
        static const char *const order[3] = { "ft_tensix_math_wait_for_dest_available()",
                                              "apply_to_dst(function,row0);",
                                              "commit_dst_to(ctx,cb_out);" };

        ft_tt_test_check("eltwise per tile: acquire, apply, commit",
                         body && in_order(body, order, 3), "");
    }
    free(body);
    body = squeezed_body(resadd, "ftra_add_block");
    {
        static const char *const order[4] = {
            "ftra_addsub_init(FTRA_ELTWISE_ADD_WORD,cb_residual);",
            "ft_tensix_math_wait_for_dest_available()",
            "ftra_addsub_apply(cb_residual,cb_panel,j,j);",
            "ftsw_commit_dst_to(ctx,cb_out,wait_mode);",
        };

        ft_tt_test_check("fused: init on the residual, then per tile acquire, add (residual A, "
                         "panel B), commit",
                         body && in_order(body, order, 4), "");
        ft_tt_test_check("the cb_panel/cb_residual names are the protocol's",
                         body &&
                             strstr(body, "constuint32_tcb_panel=FT_TT_MATMUL_SWIGLU_CB_PANEL;") &&
                             strstr(body, "constuint32_tcb_residual="
                                          "FT_TT_MATMUL_RESADD_CB_RESIDUAL;"),
                         "");
    }
    free(body);
done:
    free(eltwise);
    free(resadd);
    free(swiglu);
}

/* ===== 4. The compute's layout ============================================================= */

static void test_compute_source(void)
{
    char *text = read_text("resident_matmul_compute.c");
    char *body = squeezed_body(text, "compute_one_resadd");
    static const char *const order[8] = {
        "ftsw_matmul_pass(",
        "page<ftra_panel_block_pages(column_block);",
        "ftq_cb_wait_front_stoppable(ctx,(int32_t)cb_panel,"
        "(int32_t)ftra_panel_block_pages(column_block),",
        "ftq_cb_wait_front_stoppable(ctx,(int32_t)cb_residual,"
        "(int32_t)ftra_residual_block_pages(column_block),",
        "ftsw_unpack_srca_format(",
        "ftra_add_block(ctx,columns,cb_out,red_skip,wait_mode);",
        "ft_tensix_compute_cb_pop_front(cb_panel,ftra_panel_block_pages(column_block))",
        "ftsw_matmul_restore(cb_in0,cb_in1,depth,column_block);",
    };

    ft_tt_test_step("Compute: the RESADD block layout in resident_matmul_compute.c");
    ft_tt_test_check("compute_one_resadd found", body != NULL, kernel_dir());
    ft_tt_test_check("pass, pad to 2*NB, wait both, SrcA bf16, add, pop, restore — in order",
                     body && in_order(body, order, 8), "");
    ft_tt_test_check("the residual CB pops NB a block",
                     body && strstr(body, "ft_tensix_compute_cb_pop_front(cb_residual,"
                                          "ftra_residual_block_pages(column_block))"),
                     "");
    free(body);
    free(text);
}

/* ===== 5. CB rings across mixed descriptors =============================================== */

#ifdef RESADD_TEST_RED_PANEL_NB
#define MODEL_RESADD_PANEL_PAGES(nb) (nb)
#else
#define MODEL_RESADD_PANEL_PAGES(nb) ftra_panel_block_pages(nb)
#endif

typedef struct ring {
    uint32_t pages;   /* CB size */
    uint32_t read;    /* read pointer, pages from the CB base */
} ring;

/** An indexed unpack at `index` from the read pointer: true if inside the CB (no wrap). */
static bool ring_index_ok(const ring *r, uint32_t index)
{
    return r->read + index < r->pages;
}

static void ring_pop(ring *r, uint32_t pages)
{
    r->read = (r->read + pages) % r->pages;
}

/** One descriptor on the two rings; returns the indexed reads that ran past a CB's end. */
static uint32_t model_descriptor(ring *panel, ring *residual, bool resadd, uint32_t nt,
                                 uint32_t nb)
{
    uint32_t bad = 0u;
    uint32_t block, j;

    if (resadd) {
        for (block = 0; block < ftra_blocks(nt, nb); block++) {
            const uint32_t w = ftra_block_columns(block, nt, nb);

            for (j = 0; j < w; j++) {
                bad += ring_index_ok(residual, j) ? 0u : 1u;
                bad += ring_index_ok(panel, j) ? 0u : 1u;
            }
            ring_pop(panel, MODEL_RESADD_PANEL_PAGES(nb));
            ring_pop(residual, ftra_residual_block_pages(nb));
        }
    } else {
        for (block = 0; block < ftsw_passes(nt, nb); block += 2u) {
            const uint32_t w = ftsw_pass_columns(block, nt, nb);

            for (j = 0; j < w; j++) {
                bad += ring_index_ok(panel, ftsw_panel_gate_index(j)) ? 0u : 1u;
                bad += ring_index_ok(panel, ftsw_panel_up_index(j, nb)) ? 0u : 1u;
            }
            ring_pop(panel, ftsw_panel_pages(nb));
        }
    }
    return bad;
}

static void test_rings(void)
{
    static const uint32_t nbs[4] = { 1u, 2u, 4u, 8u };
    uint32_t rng = 0x2545F491u;
    uint32_t cases = 0u, failing = 0u, bad_total = 0u;
    uint32_t n, sequence, step;

    ft_tt_test_step("CB rings: mixed SwiGLU / RESADD descriptors, pointers kept, no wrap");
    for (n = 0; n < 4; n++) {
        const uint32_t nb = nbs[n];

        for (sequence = 0; sequence < 64u; sequence++) {
            ring panel = { ftsw_panel_pages(nb), 0u };
            ring residual = { ftra_residual_cb_pages(nb), 0u };
            uint32_t bad = 0u;

            for (step = 0; step < 24u; step++) {
                bool resadd;
                uint32_t nt;

                rng = rng * 1664525u + 1013904223u;
                resadd = ((rng >> 16) & 1u) != 0u;
                nt = 1u + ((rng >> 20) % 40u);
                bad += model_descriptor(&panel, &residual, resadd, nt, nb);
            }
            cases++;
            failing += bad ? 1u : 0u;
            bad_total += bad;
        }
    }
    ft_tt_test_check_fmt("no indexed read past either CB (NB 1/2/4/8, 256 random sequences)",
                         failing == 0u, "%u of %u sequences, %u reads past the end", failing, cases,
                         bad_total);
    {
        /* The served Qwen3.8 shapes at NB 8: out-proj Nt 10 (160/16), down Nt 10, then the
         * FFN's SwiGLU Nt 34, a layer after a layer. */
        ring panel = { ftsw_panel_pages(8u), 0u };
        ring residual = { ftra_residual_cb_pages(8u), 0u };
        uint32_t bad = 0u;
        uint32_t layer;

        for (layer = 0; layer < 8u; layer++) {
            bad += model_descriptor(&panel, &residual, true, 10u, 8u);
            bad += model_descriptor(&panel, &residual, false, 34u, 8u);
            bad += model_descriptor(&panel, &residual, true, 10u, 8u);
        }
        ft_tt_test_check_fmt("served shapes (Nt 10 RESADD, Nt 34 SwiGLU, NB 8), 8 layers",
                             bad == 0u, "%u reads past the end", bad);
    }
}

/* ===== 6. Numerics ========================================================================== */

static uint16_t bf16_round(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000u) == 0x7F800000u)
        return (uint16_t)(bits >> 16);
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float bf16_value(uint16_t half)
{
    const uint32_t bits = (uint32_t)half << 16;
    float value;

    memcpy(&value, &bits, sizeof(value));
    return value;
}

/** The unfused chain: matmul packs bf16 into `project`; eltwise ADD(hidden, project) packs bf16. */
static uint16_t unfused(float dot, uint16_t hidden)
{
    const uint16_t project = bf16_round(dot);

    return bf16_round(bf16_value(hidden) + bf16_value(project));
}

/** The fused tile as the protocol stages it: the panel holds bf16(dot); ADD(residual, panel). */
static uint16_t fused(float dot, uint16_t residual, bool red_skip)
{
    const uint16_t panel = bf16_round(dot);

    return red_skip ? panel : bf16_round(bf16_value(residual) + bf16_value(panel));
}

static void test_numerics(void)
{
    uint32_t rng = 0x9E3779B9u;
    uint32_t differ = 0u, red_differ = 0u, i;
    const uint32_t count = 5120u * 4u;

    ft_tt_test_step("Numerics: fused = matmul -> bf16 -> eltwise ADD -> bf16 (5,120-wide rows)");
    for (i = 0; i < count; i++) {
        float dot, hidden;
        uint16_t h;

        rng = rng * 1664525u + 1013904223u;
        dot = ((float)(rng >> 8) / 16777216.0f - 0.5f) * 8.0f;
        rng = rng * 1664525u + 1013904223u;
        hidden = ((float)(rng >> 8) / 16777216.0f - 0.5f) * 64.0f;
        h = bf16_round(hidden);
        differ += unfused(dot, h) != fused(dot, h, false) ? 1u : 0u;
        red_differ += unfused(dot, h) != fused(dot, h, true) ? 1u : 0u;
    }
    ft_tt_test_check_fmt("bit-identical", differ == 0u, "%u of %u differ", differ, count);
    ft_tt_test_check_fmt("the RED (add skipped) differs", red_differ > count / 2u,
                         "%u of %u differ", red_differ, count);
}

int main(void)
{
    test_words();
    test_cb_table();
    test_add_verbs();
    test_compute_source();
    test_rings();
    test_numerics();
    return ft_tt_test_finish();
}
