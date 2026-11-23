/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_main.c
 * @brief  `ingot` CLI: exact compression of safetensors and GGUF model files.
 *   ingot encode [opts] SRC [DST]     DST defaults to SRC.ingot; SRC "-" reads stdin
 *   ingot decode [opts] SRC [DST]     byte-identical rebuild; DST defaults to SRC minus .ingot
 *   ingot verify [opts] SRC [ORIG]    decode without writing; every hash + the sha256 (+ ORIG)
 *   ingot info SRC | st-ls HEADER | st-wrap NAME DTYPE AxB NBYTES
 *   opts: -j N, --base PATH (repeatable), --base-sha256 HEX, --check-base, --self-check, --measure,
 *         --policy auto|plain|delta, --no-predict, --max-version 3|4, --refs PATH (repeatable),
 *         --pool-layers LIST, --candidates N, --tensor-log FILE, -q
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ingot/cli/ingot_cli.h"
#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/container/ingot_stream.h"
#include "ingot/container/ingot_writer.h"
#include "ingot/core/ingot_error.h"

enum { MAX_BASES = 64, MAX_ARGS = 4, COMPARE_CHUNK = 1 << 20, MAX_POOL_LAYERS = 8 };

typedef struct {
    int threads;
    const char *bases[MAX_BASES];
    size_t base_count;
    const char *refs[MAX_BASES];
    size_t ref_count;
    int pool_layers[MAX_POOL_LAYERS];
    int pool_layer_count; /* -1: none given (defaults) */
    int max_version;
    int candidates;
    long twin_offset;
    int no_self;
    long search_layers;
    uint8_t base_sha256[32];
    int has_base_sha256;
    int check_base;
    int self_check;
    int measure;
    int no_predict;
    const char *tensor_log;
    int quiet;
    ingot_policy policy;
    const char *args[MAX_ARGS];
    int arg_count;
} cli_options;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int parse_hex32(const char *s, uint8_t out[32]) {
    if (strlen(s) != 64)
        return -1;
    for (int i = 0; i < 64; i++) {
        char ch = s[i];
        int v = ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : -1;
        if (v < 0)
            return -1;
        out[i / 2] = (uint8_t)(i % 2 ? (out[i / 2] | v) : v << 4);
    }
    return 0;
}

/* "1,16" -> pool layer offsets (each 1..1024; "0" or "" gives none). */
static int parse_layers(const char *s, cli_options *o) {
    o->pool_layer_count = 0;
    while (*s) {
        char *end = NULL;
        long v = strtol(s, &end, 10);
        if (end == s || v < 0 || v > 1024 || o->pool_layer_count == MAX_POOL_LAYERS)
            return -1;
        if (v > 0)
            o->pool_layers[o->pool_layer_count++] = (int)v;
        s = *end == ',' ? end + 1 : end;
        if (*end && *end != ',')
            return -1;
    }
    return 0;
}

static int parse_options(int argc, char **argv, cli_options *o) {
    *o = (cli_options){.threads = 4, .pool_layer_count = -1};
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        int has_value = i + 1 < argc;
        if (!strcmp(a, "-j") && has_value) {
            o->threads = atoi(argv[++i]);
        } else if (!strcmp(a, "--base") && has_value && o->base_count < MAX_BASES) {
            o->bases[o->base_count++] = argv[++i];
        } else if (!strcmp(a, "--refs") && has_value && o->ref_count < MAX_BASES) {
            o->refs[o->ref_count++] = argv[++i];
        } else if (!strcmp(a, "--pool-layers") && has_value) {
            if (parse_layers(argv[++i], o))
                return -1;
        } else if (!strcmp(a, "--max-version") && has_value) {
            o->max_version = atoi(argv[++i]);
            if (o->max_version != 3 && o->max_version != 4)
                return -1;
        } else if (!strcmp(a, "--search-layers") && has_value) {
            o->search_layers = atol(argv[++i]);
        } else if (!strcmp(a, "--no-self-partners")) {
            o->no_self = 1;
        } else if (!strcmp(a, "--twin-offset") && has_value) {
            o->twin_offset = atol(argv[++i]);
            if (o->twin_offset < 0)
                return -1;
        } else if (!strcmp(a, "--candidates") && has_value) {
            o->candidates = atoi(argv[++i]);
            if (o->candidates < 1 || o->candidates > 64)
                return -1;
        } else if (!strcmp(a, "--base-sha256") && has_value) {
            if (parse_hex32(argv[++i], o->base_sha256))
                return -1;
            o->has_base_sha256 = 1;
        } else if (!strcmp(a, "--check-base")) {
            o->check_base = 1;
        } else if (!strcmp(a, "--self-check")) {
            o->self_check = 1;
        } else if (!strcmp(a, "--measure")) {
            o->measure = 1;
        } else if (!strcmp(a, "--no-predict")) {
            o->no_predict = 1;
        } else if (!strcmp(a, "--tensor-log") && has_value) {
            o->tensor_log = argv[++i];
        } else if (!strcmp(a, "-q")) {
            o->quiet = 1;
        } else if (!strcmp(a, "--policy") && has_value) {
            const char *p = argv[++i];
            o->policy = !strcmp(p, "plain") ? INGOT_POLICY_PLAIN
                        : !strcmp(p, "delta") ? INGOT_POLICY_DELTA : INGOT_POLICY_AUTO;
        } else if ((a[0] == '-' && a[1]) || o->arg_count == MAX_ARGS) {
            return -1;
        } else {
            o->args[o->arg_count++] = a;
        }
    }
    return o->threads > 0 ? 0 : -1;
}

static int open_refs(const cli_options *o, ingot_base_set **refs) {
    *refs = NULL;
    if (!o->ref_count)
        return 0;
    int rc = ingot_base_set_open(o->refs, o->ref_count, o->threads, refs);
    if (rc)
        fprintf(stderr, "ingot: cannot open refs: %s\n", ingot_error_name(rc));
    return rc;
}

static int open_bases(const cli_options *o, ingot_base_set **base) {
    *base = NULL;
    if (!o->base_count)
        return 0;
    int rc = ingot_base_set_open(o->bases, o->base_count, o->threads, base);
    if (rc)
        fprintf(stderr, "ingot: cannot open base: %s\n", ingot_error_name(rc));
    return rc;
}

/* Default output name: SRC + ".ingot" (encode) or SRC minus ".ingot" (decode). NULL if SRC
 * does not end in ".ingot" when stripping, or on allocation failure. Caller frees. */
static char *default_output(const char *src, int strip) {
    size_t n = strlen(src), m = strlen(INGOT_SUFFIX);
    if (strip && (n <= m || strcmp(src + n - m, INGOT_SUFFIX)))
        return NULL;
    char *out = malloc(n + m + 1);
    if (!out)
        return NULL;
    memcpy(out, src, n);
    if (strip)
        out[n - m] = '\0';
    else
        memcpy(out + n, INGOT_SUFFIX, m + 1);
    return out;
}

/* Resolve DST: the given one, or the default, which must not exist yet (never overwrite the
 * original by accident). Returns 0 and sets *dst (and *owned when allocated), else 2. */
static int output_path(const cli_options *o, const char *cmd, int strip, const char **dst,
                       char **owned) {
    *owned = NULL;
    if (o->arg_count == 2) {
        *dst = o->args[1];
        return 0;
    }
    *owned = strcmp(o->args[0], "-") ? default_output(o->args[0], strip) : NULL;
    if (!*owned) {
        fprintf(stderr, "ingot %s: %s: cannot derive an output name (needs a %s suffix, not "
                        "stdin); give DST\n", cmd, o->args[0], strip ? INGOT_SUFFIX : "file");
        return 2;
    }
    if (access(*owned, F_OK) == 0) {
        fprintf(stderr, "ingot %s: %s exists; give DST to overwrite\n", cmd, *owned);
        free(*owned);
        *owned = NULL;
        return 2;
    }
    *dst = *owned;
    return 0;
}

static void print_encode_stats(const char *src, const ingot_encode_stats *st, double dt) {
    char hex[65];
    ingot_cli_hex(st->source_sha256, hex);
    double per = st->elements ? 8.0 / (double)st->elements : 0.0;
    printf("encoded\t%s\tsource_bytes %" PRIu64 "\toutput_bytes %" PRIu64 "\tratio %.4f"
           "\tbits_per_element %.4f\tv2_output_bytes %" PRIu64 "\tv2_bits_per_element %.4f"
           "\tsha256 %s\tseconds %.1f\n",
           src, st->source_bytes, st->output_bytes,
           st->source_bytes ? (double)st->output_bytes / (double)st->source_bytes : 0.0,
           per * (double)st->output_bytes, st->v2_output_bytes,
           per * (double)st->v2_output_bytes, hex, dt);
    for (int m = 0; m < INGOT_MODE_COUNT; m++)
        if (st->tensors[m])
            printf("mode\t%s\ttensors %u\telements %" PRIu64 "\tpayload_bytes %" PRIu64
                   "\tbits_per_element %.4f\n", ingot_cli_mode_name((ingot_mode)m), st->tensors[m],
                   st->coded_elements[m],
                   st->payload_bytes[m],
                   st->coded_elements[m] ? 8.0 * (double)st->payload_bytes[m] /
                                               (double)st->coded_elements[m] : 0.0);
}

static int cmd_encode(const cli_options *o) {
    ingot_base_set *base = NULL, *refs = NULL;
    const char *dst = NULL;
    char *owned = NULL;
    if (o->arg_count < 1 || o->arg_count > 2)
        return 2;
    if (strcmp(o->args[0], "-") && ingot_reader_probe(o->args[0], NULL) == 1) {
        fprintf(stderr, "ingot encode: %s is already an .ingot file\n", o->args[0]);
        return 2;
    }
    if ((!o->measure && output_path(o, "encode", 0, &dst, &owned)) || open_bases(o, &base) ||
        open_refs(o, &refs)) {
        ingot_base_set_close(base);
        free(owned);
        return 2;
    }
    ingot_encode_options opt = {.threads = o->threads, .base = base, .policy = o->policy,
                                .base_sha256 = o->has_base_sha256 ? o->base_sha256 : NULL,
                                .self_check = o->self_check,
                                .no_predict = o->no_predict,
                                .max_version = o->max_version,
                                .refs = refs,
                                .pool_layers = o->pool_layer_count >= 0 ? o->pool_layers : NULL,
                                .pool_layer_count = o->pool_layer_count,
                                .match_candidates = o->candidates,
                                .match_twin_offset = o->twin_offset,
                                .match_no_self = o->no_self,
                                .match_search_layers = o->search_layers,
                                .exact_v2 = o->measure || o->tensor_log != NULL,
                                .progress = o->quiet ? NULL : ingot_cli_progress,
                                .progress_ctx = (void *)"encode"};
    FILE *log = o->tensor_log ? fopen(o->tensor_log, "w") : NULL;
    if (o->tensor_log && !log) {
        fprintf(stderr, "ingot encode: cannot write %s\n", o->tensor_log);
        ingot_base_set_close(base);
        ingot_base_set_close(refs);
        free(owned);
        return 2;
    }
    if (log) {
        fprintf(log, "#name\tdtype\tshape\tmode\tpayload_bytes\tv2_bytes\tprefix_bytes\n");
        opt.entry_done = ingot_cli_log_entry;
        opt.entry_ctx = log;
    }
    ingot_encode_stats st;
    double t0 = now_seconds();
    int rc = ingot_encode_file(o->args[0], dst, &opt, &st);
    if (log && fclose(log) && !rc)
        rc = -INGOT_EIO;
    ingot_cli_progress_end();
    if (rc)
        ingot_cli_error("encode", o->args[0], rc);
    else
        print_encode_stats(o->args[0], &st, now_seconds() - t0);
    ingot_base_set_close(base);
    ingot_base_set_close(refs);
    free(owned);
    return rc ? 1 : 0;
}

/* Version 1 files keep their directory at the end: rebuild by random access. */
static int decode_v1(const cli_options *o, const char *dst, ingot_base_set *base) {
    ingot_reader *r = NULL;
    int rc = ingot_reader_open(o->args[0], &r);
    if (!rc)
        rc = ingot_reader_rebuild(r, dst, base, o->threads);
    ingot_reader_close(r);
    return rc;
}

static int stream_to(const cli_options *o, ingot_base_set *base, const char *dst,
                     ingot_write_fn write, void *ctx, ingot_stream_result *res) {
    ingot_input *in = NULL;
    ingot_base_set *refs = NULL;
    if (open_refs(o, &refs))
        return -INGOT_EBASE;
    ingot_stream_options opt = {.threads = o->threads, .base = base, .refs = refs,
                                .check_base_sha256 = o->check_base,
                                .progress = o->quiet ? NULL : ingot_cli_progress,
                                .progress_ctx = (void *)(dst ? "decode" : "verify")};
    int rc = ingot_input_open_path(o->args[0], &in);
    if (!rc)
        rc = dst ? ingot_stream_decode_to_file(in, dst, &opt, res)
                 : ingot_stream_decode(in, write, ctx, &opt, res);
    ingot_cli_progress_end();
    ingot_input_close(in);
    ingot_base_set_close(refs);
    return rc;
}

static int cmd_decode(const cli_options *o) {
    ingot_base_set *base = NULL;
    const char *dst = NULL;
    char *owned = NULL;
    if (o->arg_count < 1 || o->arg_count > 2 || output_path(o, "decode", 1, &dst, &owned))
        return 2;
    if (open_bases(o, &base)) {
        free(owned);
        return 2;
    }
    uint32_t version = 0;
    ingot_stream_result res = {0};
    int rc = 0;
    if (strcmp(o->args[0], "-") && ingot_reader_probe(o->args[0], &version) == 1 && version == 1)
        rc = decode_v1(o, dst, base);
    else
        rc = stream_to(o, base, dst, NULL, NULL, &res);
    char hex[65];
    ingot_cli_hex(res.sha256, hex);
    if (rc)
        ingot_cli_error("decode", o->args[0], rc);
    else if (version != 1)
        printf("decoded\t%s\tbytes %" PRIu64 "\tsha256 %s\n", dst, res.source_bytes, hex);
    ingot_base_set_close(base);
    free(owned);
    return rc ? 1 : 0;
}

/* Verify sink: compare the rebuilt bytes with ORIG (if given) as they stream past. */
typedef struct {
    ingot_input *orig;
    uint8_t *chunk;
} compare_sink;

static int compare_write(void *ctx, const void *buf, size_t n) {
    compare_sink *c = ctx;
    const uint8_t *p = buf;
    while (c->orig && n) {
        size_t step = n < COMPARE_CHUNK ? n : COMPARE_CHUNK;
        int rc = ingot_input_read(c->orig, c->chunk, step);
        if (rc == -INGOT_ETRUNCATED || (!rc && memcmp(c->chunk, p, step)))
            return -INGOT_ECORRUPT;
        if (rc)
            return rc;
        p += step;
        n -= step;
    }
    return 0;
}

/* Version 1: decode every entry by random access (+ compare with ORIG). */
static int verify_v1(const cli_options *o, ingot_base_set *base, ingot_input *orig) {
    ingot_reader *r = NULL;
    int rc = ingot_reader_open(o->args[0], &r);
    uint8_t *buf = NULL, *ref = NULL;
    for (size_t i = 0; !rc && i < ingot_reader_count(r); i++) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        free(buf);
        free(ref);
        buf = malloc((size_t)(e->source_size ? e->source_size : 1));
        ref = malloc((size_t)(e->source_size ? e->source_size : 1));
        rc = (buf && ref) ? ingot_reader_decode(r, i, base, o->threads, buf) : -INGOT_ENOMEM;
        if (!rc && orig)
            rc = ingot_input_read_at(orig, e->source_offset, ref, (size_t)e->source_size);
        if (!rc && orig && memcmp(buf, ref, (size_t)e->source_size))
            rc = -INGOT_ECORRUPT;
    }
    free(buf);
    free(ref);
    ingot_reader_close(r);
    return rc;
}

static int cmd_verify(const cli_options *o) {
    ingot_base_set *base = NULL;
    if (o->arg_count < 1 || o->arg_count > 2 || open_bases(o, &base))
        return 2;
    compare_sink sink = {NULL, malloc(COMPARE_CHUNK)};
    int rc = sink.chunk ? 0 : -INGOT_ENOMEM;
    if (!rc && o->arg_count == 2)
        rc = ingot_input_open_path(o->args[1], &sink.orig);
    uint32_t version = 0;
    ingot_stream_result res = {0};
    double t0 = now_seconds();
    if (!rc && strcmp(o->args[0], "-") && ingot_reader_probe(o->args[0], &version) == 1 &&
        version == 1)
        rc = verify_v1(o, base, sink.orig);
    else if (!rc)
        rc = stream_to(o, base, NULL, compare_write, &sink, &res);
    uint8_t extra = 0;
    if (!rc && sink.orig && version != 1 && ingot_input_read(sink.orig, &extra, 1) == 0)
        rc = -INGOT_ECORRUPT; /* ORIG is longer than what the file rebuilds */
    double dt = now_seconds() - t0;
    char hex[65];
    ingot_cli_hex(res.sha256, hex);
    if (rc)
        ingot_cli_error("verify", o->args[0], rc);
    else
        printf("verify\t%s\tOK\tbytes %" PRIu64 "\tsha256 %s\tcompared_to_source %s"
               "\tseconds %.1f\tmb_per_s %.1f\n", o->args[0], res.source_bytes,
               version == 1 ? "(v1)" : hex, sink.orig ? "yes" : "no", dt,
               dt > 0 ? 1e-6 * (double)res.source_bytes / dt : 0.0);
    ingot_input_close(sink.orig);
    free(sink.chunk);
    ingot_base_set_close(base);
    return rc ? 1 : 0;
}

static void usage(void) {
    fprintf(stderr,
            "usage: ingot encode [opts] SRC [DST]   SRC: .safetensors or .gguf, or - (stdin)\n"
            "       ingot decode [opts] SRC [DST]   rebuild the exact original file\n"
            "       ingot verify [opts] SRC [ORIG]  decode without writing; check every hash,\n"
            "                                       the sha256, and ORIG byte for byte if given\n"
            "       ingot info SRC                  header and per-tensor sizes\n"
            "opts:  -j N                 threads (default 4)\n"
            "       --base PATH          base model for delta files (file, dir, .ingot; repeat)\n"
            "       --base-sha256 HEX    encode: the base file's known sha256 (skips hashing)\n"
            "       --check-base         decode/verify: also check the base file's sha256\n"
            "       --self-check         encode: decode every coded tensor back and compare\n"
            "       --measure            encode: sizes only, no output file (with --self-check:\n"
            "                            a full round-trip check that needs no disk)\n"
            "       --policy auto|plain|delta   encode mode policy (default auto)\n"
            "       --no-predict         encode: no cross-tensor prediction (a version 2 file)\n"
            "       --max-version 3|4    encode: newest format (default 4: match mode)\n"
            "       --refs PATH          earlier files of the model (match sources outside this\n"
            "                            file; encode and decode; repeat)\n"
            "       --pool-layers LIST   encode: match pool layer offsets (default 1,16)\n"
            "       --candidates N       encode: searched partner candidates per unit (32)\n"
            "       --twin-offset N      encode: also try units u -+ N (grown models; Qwen3.5-27B\n"
            "                            lineage: 12288)\n"
            "       --search-layers N    encode: sketch-search only pool layers up to offset N\n"
            "                            (farther ones: same-index and twin candidates only)\n"
            "       --no-self-partners   encode: no partners from the tensor's own rows (every\n"
            "                            tensor decodes in parallel; slightly larger)\n"
            "       --tensor-log FILE    encode: one line per entry (mode, payload, v2 bytes)\n"
            "       -q                   no progress line\n"
            "Encode writes SRC.ingot and decode strips .ingot unless DST is given; an existing\n"
            "default output is never overwritten. Files are recognised by magic, not by name.\n");
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "info"))
        return ingot_cli_info(argv[2]);
    if (argc == 3 && !strcmp(argv[1], "st-ls"))
        return ingot_cli_st_ls(argv[2]);
    if (argc == 6 && !strcmp(argv[1], "st-wrap"))
        return ingot_cli_st_wrap(argv[2], argv[3], argv[4], argv[5]);
    cli_options o;
    if (argc < 2 || parse_options(argc, argv, &o)) {
        usage();
        return 2;
    }
    if (!strcmp(argv[1], "encode"))
        return cmd_encode(&o);
    if (!strcmp(argv[1], "decode"))
        return cmd_decode(&o);
    if (!strcmp(argv[1], "verify"))
        return cmd_verify(&o);
    usage();
    return 2;
}
