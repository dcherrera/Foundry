/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_cli_info.c
 * @brief  `ingot info`, error messages, progress, and the converter helpers st-ls / st-wrap.
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_cli.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "ingot/container/ingot_layout.h"
#include "ingot/container/ingot_reader.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_input.h"

const char *ingot_cli_mode_name(ingot_mode m) {
    static const char *const NAMES[] = {"raw",         "plain",       "delta",  "quant",
                                        "quant-delta", "block-delta", "predict"};
    return (unsigned)m < sizeof(NAMES) / sizeof(NAMES[0]) ? NAMES[m] : "?";
}

void ingot_cli_log_entry(void *ctx, const ingot_entry *e, uint64_t v2_bytes,
                         uint64_t prefix_bytes) {
    char shape[8 * 21 + 1] = "";
    size_t at = 0;
    for (int d = 0; d < e->ndim && at < sizeof(shape); d++)
        at += (size_t)snprintf(shape + at, sizeof(shape) - at, d ? "x%" PRIu64 : "%" PRIu64,
                               e->shape[d]);
    fprintf((FILE *)ctx, "%s\t%s\t%s\t%s\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\n",
            e->name[0] ? e->name : "(filler)", ingot_dtype_name(e->dtype), shape,
            ingot_cli_mode_name(e->mode), e->payload_size, v2_bytes, prefix_bytes);
}

static const char *kind_name(ingot_source_kind k) {
    return k == INGOT_SOURCE_GGUF ? "gguf" : "safetensors";
}

void ingot_cli_hex(const uint8_t digest[32], char out[65]) {
    static const char HEX[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[2 * i] = HEX[digest[i] >> 4];
        out[2 * i + 1] = HEX[digest[i] & 15];
    }
    out[64] = '\0';
}

/* The base a delta file needs, from its header (when the file can be opened). */
static void print_base_hint(const char *cmd, const char *path) {
    ingot_reader *r = NULL;
    if (ingot_reader_open(path, &r))
        return;
    const ingot_file_header *h = ingot_reader_header(r);
    char hex[65];
    if (ingot_header_has_base(h)) {
        ingot_cli_hex(h->base_sha256, hex);
        fprintf(stderr, "ingot %s: this file needs its base model: --base FILE with sha256 %s"
                        " (%" PRIu64 " bytes)\n", cmd, hex, h->base_size);
    } else {
        fprintf(stderr, "ingot %s: this file needs the base model it was coded against (--base)\n",
                cmd);
    }
    ingot_reader_close(r);
}

void ingot_cli_error(const char *cmd, const char *path, int rc) {
    uint32_t version = 0;
    int probe = strcmp(path, "-") ? ingot_reader_probe(path, &version) : -1;
    if (rc == -INGOT_EUNSUPPORTED && probe == 1 &&
        (version < INGOT_VERSION_MIN || version > INGOT_VERSION))
        fprintf(stderr, "ingot %s: %s: unsupported .ingot format version %" PRIu32
                        " (this build reads %d..%d)\n", cmd, path, version, INGOT_VERSION_MIN,
                INGOT_VERSION);
    else if (rc == -INGOT_EFORMAT && probe == 0)
        fprintf(stderr, "ingot %s: %s: not an .ingot file\n", cmd, path);
    else if (rc == -INGOT_ETRUNCATED)
        fprintf(stderr, "ingot %s: %s: input ended early (truncated or incomplete download)\n",
                cmd, path);
    else if (rc == -INGOT_ECORRUPT)
        fprintf(stderr, "ingot %s: %s: data check failed (the file is corrupt)\n", cmd, path);
    else
        fprintf(stderr, "ingot %s: %s: %s\n", cmd, path, ingot_error_name(rc));
    if (rc == -INGOT_EBASE && probe == 1)
        print_base_hint(cmd, path);
}

static int progress_drawn;

void ingot_cli_progress(void *ctx, uint64_t done, uint64_t total) {
    static struct timespec last;
    struct timespec now;
    if (!isatty(STDERR_FILENO))
        return;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double dt = (double)(now.tv_sec - last.tv_sec) + 1e-9 * (double)(now.tv_nsec - last.tv_nsec);
    if (dt < 0.25 && done != total)
        return;
    last = now;
    if (total)
        fprintf(stderr, "\r%s: %.2f / %.2f GB (%.0f%%)  ", (const char *)ctx, 1e-9 * (double)done,
                1e-9 * (double)total, 100.0 * (double)done / (double)total);
    else
        fprintf(stderr, "\r%s: %.2f GB  ", (const char *)ctx, 1e-9 * (double)done);
    progress_drawn = 1;
}

void ingot_cli_progress_end(void) {
    if (progress_drawn)
        fputc('\n', stderr);
    progress_drawn = 0;
}

static void print_header(const ingot_reader *r) {
    const ingot_file_header *h = ingot_reader_header(r);
    char hex[65];
    printf("#format\tversion %" PRIu32 "\tsource %s\tsource_bytes %" PRIu64 "\n", h->version,
           h->version >= 2 ? kind_name(h->kind) : "safetensors", h->source_size);
    if (h->version < 2)
        return;
    ingot_cli_hex(h->source_sha256, hex);
    printf("#source_sha256\t%s\n", hex);
    if (ingot_header_has_base(h)) {
        ingot_cli_hex(h->base_sha256, hex);
        printf("#base_sha256\t%s\tbase_bytes %" PRIu64 "\n", hex, h->base_size);
    }
}

int ingot_cli_info(const char *path) {
    ingot_reader *r = NULL;
    int rc = ingot_reader_open(path, &r);
    if (rc) {
        ingot_cli_error("info", path, rc);
        return 1;
    }
    print_header(r);
    printf("#tensor\tdtype\tmode\telements\tsource_bytes\tpayload_bytes\tbits_per_element\n");
    uint64_t elements = 0;
    for (size_t i = 0; i < ingot_reader_count(r); i++) {
        const ingot_entry *e = ingot_reader_entry(r, i);
        uint64_t n = 1;
        for (int d = 0; d < e->ndim; d++)
            n *= e->shape[d];
        if (e->name[0])
            elements += n;
        printf("%s\t%s\t%s\t%" PRIu64 "\t%" PRIu64 "\t%" PRIu64 "\t%.4f\n",
               e->name[0] ? e->name : "(filler)", ingot_dtype_name(e->dtype),
               ingot_cli_mode_name(e->mode),
               n, e->source_size, e->payload_size,
               n ? 8.0 * (double)e->payload_size / (double)n : 0.0);
    }
    printf("#file\t%" PRIu64 " bytes\t%" PRIu64 " elements\t%.4f bits/element\t%.4f of source\n",
           ingot_reader_file_size(r), elements,
           elements ? 8.0 * (double)ingot_reader_file_size(r) / (double)elements : 0.0,
           (double)ingot_reader_file_size(r) / (double)ingot_reader_source_size(r));
    ingot_reader_close(r);
    return 0;
}

int ingot_cli_st_ls(const char *path) {
    ingot_input *in = NULL;
    ingot_layout *l = NULL;
    int rc = ingot_input_open_path(path, &in);
    if (!rc)
        rc = ingot_layout_read(in, &l);
    if (rc) {
        fprintf(stderr, "ingot st-ls: %s: %s\n", path, ingot_error_name(rc));
        ingot_input_close(in);
        return 1;
    }
    for (size_t i = 0; i < ingot_layout_count(l); i++) {
        const ingot_layout_tensor *t = ingot_layout_tensor_at(l, i);
        printf("%s\t%s\t", t->name, ingot_dtype_name(t->dtype));
        for (int d = 0; d < t->ndim; d++)
            printf(d ? "x%" PRIu64 : "%" PRIu64, t->shape[d]);
        printf("\t%" PRIu64 "\t%" PRIu64 "\n", t->offset, t->offset + t->size);
    }
    ingot_layout_free(l);
    ingot_input_close(in);
    return 0;
}

int ingot_cli_st_wrap(const char *name, const char *dtype, const char *shape,
                      const char *nbytes) {
    char dims[256] = {0};
    size_t used = 0;
    for (const char *p = shape; *p && used + 2 < sizeof(dims); p++)
        dims[used++] = (*p == 'x') ? ',' : *p;
    char json[4096];
    int n = snprintf(json, sizeof(json),
                     "{\"%s\":{\"dtype\":\"%s\",\"shape\":[%s],\"data_offsets\":[0,%s]}}", name,
                     dtype, dims, nbytes);
    if (n < 0 || (size_t)n >= sizeof(json)) {
        fprintf(stderr, "ingot st-wrap: header too long\n");
        return 1;
    }
    size_t padded = ((size_t)n + 7) & ~(size_t)7;
    unsigned char len[8];
    for (int i = 0; i < 8; i++)
        len[i] = (unsigned char)((uint64_t)padded >> (8 * i));
    fwrite(len, 1, 8, stdout);
    fwrite(json, 1, (size_t)n, stdout);
    for (size_t i = (size_t)n; i < padded; i++)
        fputc(' ', stdout);
    return ferror(stdout) ? 1 : 0;
}
