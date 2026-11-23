/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_ingot_cuda_file.c
 * @brief  GPU decode of a whole .ingot file on one die, checked against the source model:
 *         every BF16 entry of mode 1, 6 or 7 is decoded on the GPU in payload order (match
 *         sources = entries decoded before it, or external refs read from the model) and
 *         compared byte for byte with the source tensor and its directory hash; its matvec is
 *         compared with a double-precision dot of the source. Prints per-tensor decode time.
 *
 * Usage (bank, one die):
 *   CUDA_VISIBLE_DEVICES=<die> ingot/build/test_ingot_cuda_file ingot/build/ingot_cuda.ptx \
 *       FILE.ingot MODEL_DIR [matvec]
 * MODEL_DIR holds the source safetensors (the file's own shard and the earlier shards its
 * external refs name). Device memory: every BF16 2-D tensor of the shard decoded, plus images.
 */
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ingot/codec/ingot_tensor_codec.h"
#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_dirtail.h"
#include "ingot/container/ingot_header.h"
#include "ingot/core/ingot_bf16.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_hash.h"
#include "ingot/cuda/ingot_cuda_reference.h"
#include "ingot/cuda/ingot_cuda_runtime.h"

/** The mapped file, its directory and tail. */
typedef struct {
    const uint8_t *map;
    size_t size;
    ingot_file_header h;
    ingot_entry *e;
    ingot_dir_tail tail;
} ingot_file;

static int file_open(const char *path, ingot_file *f) {
    memset(f, 0, sizeof(*f));
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st))
        return -INGOT_EIO;
    f->size = (size_t)st.st_size;
    void *m = mmap(NULL, f->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED)
        return -INGOT_EIO;
    f->map = m;
    uint64_t used = 0;
    int rc = ingot_header_parse(f->map, f->size, &f->h);
    if (!rc)
        rc = ingot_directory_parse_used(f->map + f->h.dir_offset, f->h.dir_size,
                                        f->h.entry_count, &f->e, &used);
    if (!rc && f->h.version >= 4)
        rc = ingot_dir_tail_parse(f->map + f->h.dir_offset + used,
                                  (size_t)(f->h.dir_size - used), f->e, f->h.entry_count,
                                  &f->tail);
    return rc;
}

/** Running totals. */
typedef struct {
    uint64_t weights[8];
    double store_ms[8], matvec_ms[8];
    uint64_t table_bytes, side_bytes, payload_bytes;
    int mismatches, tensors;
} totals;

/** Device copies: decoded entries (sources) and external refs. */
typedef struct {
    uint64_t *entry;
    uint64_t *ref;
} device_tensors;

static int is_bf16_2d(const ingot_entry *e) { return e->dtype == INGOT_DTYPE_BF16 && e->ndim == 2; }

/* Upload a source tensor (by name, from the model) to a new device buffer. */
static int upload_named(ingot_cuda_runtime *rt, ingot_base_set *model, const char *name,
                        uint64_t bytes, uint64_t *dev) {
    ingot_base_tensor t;
    const uint8_t *data = NULL;
    int rc = ingot_base_set_find(model, name, &t);
    if (!rc && t.size != bytes)
        rc = -INGOT_EFORMAT;
    if (!rc)
        rc = ingot_base_set_load(model, &t, &data);
    if (!rc)
        rc = ingot_cuda_alloc(rt, bytes, dev);
    if (!rc)
        rc = ingot_cuda_upload(rt, *dev, data, bytes);
    return rc;
}

/* Bind every non-self source: an earlier entry's device copy or an external ref. */
static int bind_file_sources(ingot_cuda_runtime *rt, const ingot_file *f, ingot_base_set *model,
                             device_tensors *dt, const ingot_cuda_image *img, ingot_cuda_params *p) {
    for (uint32_t s = 0; s < p->source_count; s++) {
        if (img->shape[s].self)
            continue;
        uint32_t ref = img->shape[s].ref;
        const ingot_entry *se;
        uint64_t dev;
        if (ref < f->h.entry_count) {
            se = &f->e[ref];
            dev = dt->entry[ref];
        } else {
            uint32_t r = ref - f->h.entry_count;
            se = &f->tail.refs[r];
            if (!dt->ref[r] && upload_named(rt, model, se->name, se->source_size, &dt->ref[r]))
                return -INGOT_EBASE;
            dev = dt->ref[r];
        }
        if (!dev || ingot_cuda_bind_source(img, p, s, dev, se->shape[0], se->shape[1]))
            return -INGOT_EFORMAT;
    }
    return 0;
}

/* Worst relative error of y against a double dot of the source rows. */
static double matvec_error(const uint16_t *w, uint64_t rows, uint64_t cols, const float *x,
                           const float *y) {
    double worst = 0;
    for (uint64_t r = 0; r < rows; r++) {
        double want = 0, mag = 0;
        for (uint64_t c = 0; c < cols; c++) {
            double t = (double)ingot_bf16_to_f32(w[r * cols + c]) * x[c];
            want += t;
            mag += fabs(t);
        }
        double err = fabs(y[r] - want) / (mag + 1e-30);
        worst = err > worst ? err : worst;
    }
    return worst;
}

/* Matvec of a decoded entry: time, and the error against the source. */
static int entry_matvec(ingot_cuda_runtime *rt, const ingot_cuda_image *img, ingot_cuda_params *p,
                        const uint16_t *want, float *ms, double *err) {
    uint64_t x = 0, y = 0, part = 0;
    float *hx = malloc(p->cols * sizeof(float)), *hy = malloc(p->rows * sizeof(float));
    for (uint64_t c = 0; c < p->cols; c++)
        hx[c] = (float)((c * 2654435761u % 2001) / 1000.0 - 1.0);
    int rc = ingot_cuda_alloc(rt, p->cols * sizeof(float), &x);
    if (!rc)
        rc = ingot_cuda_upload(rt, x, hx, p->cols * sizeof(float));
    if (!rc)
        rc = ingot_cuda_alloc(rt, p->rows * sizeof(float), &y);
    if (!rc)
        rc = ingot_cuda_alloc(rt, 2 * p->block_count * sizeof(float), &part);
    p->flags = INGOT_CUDA_MATVEC;
    p->x = x;
    p->y = y;
    p->partials = part;
    if (!rc)
        rc = ingot_cuda_launch(rt, img, p, ms);
    if (!rc)
        rc = ingot_cuda_download(rt, hy, y, p->rows * sizeof(float));
    if (!rc)
        *err = matvec_error(want, p->rows, p->cols, hx, hy);
    ingot_cuda_free(rt, x);
    ingot_cuda_free(rt, y);
    ingot_cuda_free(rt, part);
    free(hx);
    free(hy);
    return rc;
}

/* Decode one coded entry on the GPU into *dev and check it against the source. */
static int entry_decode(ingot_cuda_runtime *rt, const ingot_file *f, ingot_base_set *model,
                        device_tensors *dt, size_t i, int matvec, totals *tot) {
    const ingot_entry *e = &f->e[i];
    ingot_cuda_image img;
    int rc = ingot_cuda_image_build(e, f->map + e->payload_offset, e->payload_size, &img);
    if (rc)
        return rc;
    uint64_t image = 0, status = 0, out = 0;
    ingot_cuda_params p;
    rc = ingot_cuda_image_upload(rt, &img, &image);
    ingot_cuda_image_params(&img, image, &p);
    if (!rc)
        rc = ingot_cuda_alloc(rt, e->source_size, &out);
    if (!rc)
        rc = ingot_cuda_alloc(rt, p.block_count * 4, &status);
    if (!rc)
        rc = bind_file_sources(rt, f, model, dt, &img, &p);
    ingot_cuda_bind_self(&p, out);
    p.flags = INGOT_CUDA_STORE;
    p.out = out;
    p.status = status;
    float store_ms = 0, mv_ms = 0;
    if (!rc)
        rc = ingot_cuda_launch(rt, &img, &p, &store_ms);
    uint8_t *got = malloc(e->source_size);
    uint32_t *st = calloc(p.block_count, 4);
    if (!rc)
        rc = ingot_cuda_download(rt, got, out, e->source_size);
    if (!rc)
        rc = ingot_cuda_download(rt, st, status, p.block_count * 4);
    int bad = 0;
    for (uint64_t b = 0; !rc && b < p.block_count; b++)
        bad += st[b] != 0;
    int same = !rc && !bad && ingot_hash64(got, e->source_size) == e->hash;
    ingot_base_tensor t;
    const uint8_t *want = NULL;
    if (!rc && !ingot_base_set_find(model, e->name, &t) && !ingot_base_set_load(model, &t, &want))
        same = same && t.size == e->source_size && !memcmp(got, want, e->source_size);
    double err = -1;
    if (!rc && matvec && same && p.cols % ingot_cuda_step_weights(p.kind) == 0)
        rc = entry_matvec(rt, &img, &p, (const uint16_t *)got, &mv_ms, &err);
    tot->mismatches += !same;
    tot->tensors++;
    tot->weights[e->mode] += e->source_size / 2;
    tot->store_ms[e->mode] += store_ms;
    tot->matvec_ms[e->mode] += mv_ms;
    tot->table_bytes += img.table_bytes;
    tot->side_bytes += img.side_bytes;
    tot->payload_bytes += e->payload_size;
    printf("%-60s m%d %7.0f Mw %s blocks-bad %d  store %8.2f ms (%5.2f Gw/s)  matvec %8.2f ms  "
           "err %.1e  tables %llu KB\n",
           e->name, e->mode, e->source_size / 2e6, same ? "ok  " : "BAD ", bad, store_ms,
           e->source_size / 2e6 / (store_ms + 1e-9), mv_ms, err,
           (unsigned long long)(img.table_bytes >> 10));
    free(got);
    free(st);
    ingot_cuda_free(rt, image);
    ingot_cuda_free(rt, status);
    if (is_bf16_2d(e) && !rc)
        dt->entry[i] = out;
    else
        ingot_cuda_free(rt, out);
    ingot_cuda_image_free(&img);
    return rc;
}

static void print_totals(const totals *t) {
    printf("tensors %d, mismatches %d, payload %.3f GB, device tables %.1f MB, side %.1f MB\n",
           t->tensors, t->mismatches, t->payload_bytes / 1e9, t->table_bytes / 1e6,
           t->side_bytes / 1e6);
    for (int m = 1; m < 8; m++)
        if (t->weights[m])
            printf("  mode %d: %.3f G weights, store %.1f ms (%.2f Gw/s), matvec %.1f ms\n", m,
                   t->weights[m] / 1e9, t->store_ms[m], t->weights[m] / 1e6 / t->store_ms[m],
                   t->matvec_ms[m]);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s KERNELS.ptx FILE.ingot MODEL_DIR [matvec]\n", argv[0]);
        return 2;
    }
    ingot_file f;
    ingot_base_set *model = NULL;
    ingot_cuda_runtime *rt = NULL;
    const char *dirs[1] = {argv[3]};
    if (file_open(argv[2], &f) || ingot_base_set_open(dirs, 1, 1, &model) ||
        ingot_cuda_runtime_open(0, argv[1], &rt)) {
        fprintf(stderr, "open failed\n");
        return 2;
    }
    device_tensors dt = {calloc(f.h.entry_count, 8), calloc(f.tail.ref_count + 1, 8)};
    totals tot = {0};
    int failures = 0;
    for (size_t i = 0; i < f.h.entry_count; i++) {
        const ingot_entry *e = &f.e[i];
        int coded = e->dtype == INGOT_DTYPE_BF16 && (e->mode == INGOT_MODE_PLAIN ||
                    e->mode == INGOT_MODE_PREDICT || e->mode == INGOT_MODE_MATCH);
        int rc = 0;
        if (coded)
            rc = entry_decode(rt, &f, model, &dt, i, argc > 4, &tot);
        else if (is_bf16_2d(e) && e->name[0])
            rc = upload_named(rt, model, e->name, e->source_size, &dt.entry[i]);
        if (rc)
            printf("%s: rc %d\n", e->name, rc);
        failures += rc != 0;
    }
    print_totals(&tot);
    failures += tot.mismatches;
    printf("%s: %d failure(s)\n", failures ? "FAILED" : "PASSED", failures);
    ingot_cuda_runtime_close(rt);
    ingot_base_set_close(model);
    return failures ? 1 : 0;
}
