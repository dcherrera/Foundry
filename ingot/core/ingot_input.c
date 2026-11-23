/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_input.c
 * @brief  Byte sources: file, pipe, callback (see header).
 *
 * @status Active
 * @issues None
 * @todo   None
 */
#include "ingot_input.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ingot/core/ingot_error.h"
#include "tt/convert/ft_sha256.h"

/* macOS read(2) rejects counts above INT_MAX; skips go through a small scratch buffer. */
enum { MAX_READ = 1 << 30, SKIP_CHUNK = 1 << 16 };

struct ingot_input {
    int fd;
    int owned;
    int seekable;
    uint64_t size;
    ingot_read_fn fn;
    void *ctx;
    uint64_t pos;
    int hashing;
    ft_sha256_ctx sha;
};

static int input_new(ingot_input **out) {
    *out = calloc(1, sizeof(**out));
    if (!*out)
        return -INGOT_ENOMEM;
    (*out)->fd = -1;
    return 0;
}

int ingot_input_open_fd(int fd, int owned, ingot_input **out) {
    if (fd < 0 || !out)
        return -INGOT_EINVAL;
    int rc = input_new(out);
    if (rc)
        return rc;
    struct stat st;
    (*out)->fd = fd;
    (*out)->owned = owned;
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) {
        (*out)->seekable = 1;
        (*out)->size = (uint64_t)st.st_size;
    }
    return 0;
}

int ingot_input_open_path(const char *path, ingot_input **out) {
    if (!path || !out)
        return -INGOT_EINVAL;
    if (!strcmp(path, "-"))
        return ingot_input_open_fd(STDIN_FILENO, 0, out);
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -INGOT_EIO;
    int rc = ingot_input_open_fd(fd, 1, out);
    if (rc)
        close(fd);
    return rc;
}

int ingot_input_open_callback(ingot_read_fn fn, void *ctx, ingot_input **out) {
    if (!fn || !out)
        return -INGOT_EINVAL;
    int rc = input_new(out);
    if (rc)
        return rc;
    (*out)->fn = fn;
    (*out)->ctx = ctx;
    return 0;
}

void ingot_input_close(ingot_input *in) {
    if (!in)
        return;
    if (in->owned && in->fd >= 0)
        close(in->fd);
    free(in);
}

int ingot_input_seekable(const ingot_input *in) { return in ? in->seekable : 0; }

uint64_t ingot_input_size(const ingot_input *in) { return in ? in->size : 0; }

uint64_t ingot_input_pos(const ingot_input *in) { return in ? in->pos : 0; }

/* One read of up to n bytes from the source; *got = 0 at the end. */
static int read_some(ingot_input *in, void *buf, size_t n, size_t *got) {
    if (in->fn)
        return in->fn(in->ctx, buf, n, got);
    for (;;) {
        ssize_t r = read(in->fd, buf, n < MAX_READ ? n : MAX_READ);
        if (r >= 0) {
            *got = (size_t)r;
            return 0;
        }
        if (errno != EINTR)
            return -INGOT_EIO;
    }
}

int ingot_input_read(ingot_input *in, void *buf, size_t n) {
    if (!in || (!buf && n))
        return -INGOT_EINVAL;
    uint8_t *p = buf;
    while (n) {
        size_t got = 0;
        int rc = read_some(in, p, n, &got);
        if (rc)
            return rc;
        if (!got)
            return -INGOT_ETRUNCATED;
        if (in->hashing)
            ft_sha256_update(&in->sha, p, got);
        p += got;
        n -= got;
        in->pos += got;
    }
    return 0;
}

int ingot_input_skip(ingot_input *in, uint64_t n) {
    if (!in)
        return -INGOT_EINVAL;
    if (in->seekable && !in->hashing && !in->fn) {
        if (n > in->size - (in->pos < in->size ? in->pos : in->size))
            return -INGOT_ETRUNCATED;
        if (lseek(in->fd, (off_t)n, SEEK_CUR) < 0)
            return -INGOT_EIO;
        in->pos += n;
        return 0;
    }
    uint8_t scratch[SKIP_CHUNK];
    while (n) {
        size_t step = n < sizeof(scratch) ? (size_t)n : sizeof(scratch);
        int rc = ingot_input_read(in, scratch, step);
        if (rc)
            return rc;
        n -= step;
    }
    return 0;
}

int ingot_input_read_at(ingot_input *in, uint64_t at, void *buf, size_t n) {
    if (!in || (!buf && n))
        return -INGOT_EINVAL;
    if (!in->seekable) {
        if (at < in->pos)
            return -INGOT_EUNSUPPORTED;
        int rc = ingot_input_skip(in, at - in->pos);
        return rc ? rc : ingot_input_read(in, buf, n);
    }
    uint8_t *p = buf;
    while (n) {
        ssize_t r = pread(in->fd, p, n < MAX_READ ? n : MAX_READ, (off_t)at);
        if (r < 0 && errno == EINTR)
            continue;
        if (r < 0)
            return -INGOT_EIO;
        if (r == 0)
            return -INGOT_ETRUNCATED;
        p += r;
        n -= (size_t)r;
        at += (uint64_t)r;
    }
    return 0;
}

int ingot_input_hash_start(ingot_input *in) {
    if (!in || in->pos)
        return -INGOT_EINVAL;
    ft_sha256_init(&in->sha);
    in->hashing = 1;
    return 0;
}

int ingot_input_hash_finish(ingot_input *in, uint8_t digest[32]) {
    if (!in || !in->hashing || !digest)
        return -INGOT_EINVAL;
    uint8_t scratch[SKIP_CHUNK];
    for (;;) {
        size_t got = 0;
        int rc = read_some(in, scratch, sizeof(scratch), &got);
        if (rc)
            return rc;
        if (!got)
            break;
        ft_sha256_update(&in->sha, scratch, got);
        in->pos += got;
    }
    ft_sha256_final(&in->sha, digest);
    in->hashing = 0;
    return 0;
}
