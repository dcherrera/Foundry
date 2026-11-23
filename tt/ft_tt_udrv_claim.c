/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_claim.c
 * @brief Implementation of flock-based multi-tenant resource claims (see
 *   ft_tt_udrv_claim.h for the design and the rejected alternatives).
 *
 * @status Off-hardware complete; this module never touches the card.
 * @issues None
 * @todo None
 */

/* flock(2) is BSD, not POSIX, so a bare _POSIX_C_SOURCE hides it on both
 * platforms this code is built on. Same shape as the _DEFAULT_SOURCE note in
 * DEV_NOTES §5 for MADV_DONTNEED: macOS declares BSD interfaces under
 * _DARWIN_C_SOURCE, glibc under _DEFAULT_SOURCE (which implies POSIX 2008). */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _DEFAULT_SOURCE
#endif

#include "ft_tt_udrv_claim.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "ft_tt_udrv.h"

/** Single-host coordination: a pid means nothing on another machine, and /tmp
 *  being cleared at reboot is desirable because a reboot invalidates every pid
 *  a record could name. Same reasoning as tools/ft_tt_card_lock.c. */
#define FT_TT_UDRV_CLAIM_DEFAULT_ROOT "/tmp/ft_tt_udrv.d"

/** Room for "<root>/<card>/<name>/<index>.claim" with margin, sized so the
 *  compiler can prove -Wformat-truncation is unreachable rather than sized to
 *  the tightest bound that compiles today. */
#define CLAIM_PATH_MAX 512u

struct ft_tt_udrv_claim {
    ft_tt_udrv_claim_namespace ns;
    uint32_t index;
    int fd; /**< holds the flock; closing it releases the claim */
};

/** True when `text` is a usable single path component. */
static bool is_path_component(const char *text, size_t capacity)
{
    size_t length;

    if (!text)
        return false;

    length = strlen(text);
    if (length == 0u || length >= capacity)
        return false;

    if (strchr(text, '/') != NULL)
        return false;

    /* "." and ".." would escape or alias the namespace directory. */
    if (strcmp(text, ".") == 0 || strcmp(text, "..") == 0)
        return false;

    return true;
}

/** Bounded copy that always terminates and cannot trip -Wformat-truncation. */
static void copy_bounded(char *dest, size_t dest_bytes, const char *src)
{
    size_t length;

    if (dest_bytes == 0u)
        return;

    length = strlen(src);
    if (length >= dest_bytes)
        length = dest_bytes - 1u;

    memcpy(dest, src, length);
    dest[length] = '\0';
}

/**
 * mkdir the path, treating "already exists" as success, and make sure the mode
 * really IS 0777 rather than whatever the umask allowed.
 *
 * THE UMASK IS THE TRAP, AND IT IS NOT HYPOTHETICAL. `mkdir(0777)` under root's
 * usual umask 022 produces a **0755 root-owned** directory, and every later
 * non-root job then fails to claim with `rc=-4` — which reads like a hardware
 * fault, and has cost this project time. Whoever creates the tree first decides
 * everyone else's access, and on this box that is usually a `sudo` bring-up.
 *
 * The chmod is deliberately NOT conditional on having just created the directory:
 * a tree left too narrow by an EARLIER root run is exactly the case that needs
 * fixing, and a chmod attempted by a non-owner simply fails and is ignored. These
 * are /tmp lock files and not secrets — the claim IS the lock, so a user who
 * cannot open the file cannot take part in the arbitration at all.
 */
static int ensure_directory(const char *path)
{
    if (mkdir(path, 0777) != 0 && errno != EEXIST)
        return FT_TT_UDRV_EIO;

    (void)chmod(path, 0777);
    return FT_TT_UDRV_OK;
}

/** Widen a claim file to 0666 whatever the umask did to it, for the same reason
 *  ensure_directory() widens the tree. Ignored when we do not own it. */
static void widen_claim_file(int fd)
{
    (void)fchmod(fd, 0666);
}

/** Build "<root>/<card>/<name>" and create every level of it. */
static int ensure_namespace_directory(const ft_tt_udrv_claim_namespace *ns, char *out_path,
                                      size_t out_bytes)
{
    char level[CLAIM_PATH_MAX];
    int written;

    written = snprintf(level, sizeof(level), "%s", ns->root);
    if (written < 0 || (size_t)written >= sizeof(level))
        return FT_TT_UDRV_EINVAL;
    if (ensure_directory(level) != FT_TT_UDRV_OK)
        return FT_TT_UDRV_EIO;

    written = snprintf(level, sizeof(level), "%s/%s", ns->root, ns->card);
    if (written < 0 || (size_t)written >= sizeof(level))
        return FT_TT_UDRV_EINVAL;
    if (ensure_directory(level) != FT_TT_UDRV_OK)
        return FT_TT_UDRV_EIO;

    written = snprintf(level, sizeof(level), "%s/%s/%s", ns->root, ns->card, ns->name);
    if (written < 0 || (size_t)written >= sizeof(level))
        return FT_TT_UDRV_EINVAL;
    if (ensure_directory(level) != FT_TT_UDRV_OK)
        return FT_TT_UDRV_EIO;

    copy_bounded(out_path, out_bytes, level);
    return FT_TT_UDRV_OK;
}

/** Path of one index's lock file, and of its record. */
static int claim_paths(const ft_tt_udrv_claim_namespace *ns, uint32_t index, char *out_lock,
                       size_t lock_bytes, char *out_record, size_t record_bytes)
{
    char directory[CLAIM_PATH_MAX];
    int written;
    int rc;

    rc = ensure_namespace_directory(ns, directory, sizeof(directory));
    if (rc != FT_TT_UDRV_OK)
        return rc;

    written = snprintf(out_lock, lock_bytes, "%s/%u.lock", directory, index);
    if (written < 0 || (size_t)written >= lock_bytes)
        return FT_TT_UDRV_EINVAL;

    written = snprintf(out_record, record_bytes, "%s/%u.claim", directory, index);
    if (written < 0 || (size_t)written >= record_bytes)
        return FT_TT_UDRV_EINVAL;

    return FT_TT_UDRV_OK;
}

/**
 * Write the advisory record naming this holder.
 *
 * Deliberately AFTER the flock and deliberately not load-bearing: the lock is
 * the claim. A failure here is reported but does not fail the acquisition,
 * because refusing a resource we already own would be a worse outcome than an
 * unnamed holder.
 */
static void write_record(const char *record_path, const char *reason)
{
    FILE *file = fopen(record_path, "w");

    if (!file)
        return;

    (void)fprintf(file, "pid %ld\nreason %s\nacquired %lld\n", (long)getpid(),
                  reason ? reason : "", (long long)time(NULL));
    (void)fchmod(fileno(file), 0666);
    (void)fclose(file);
}

/** Read pid and reason out of a record. Missing or malformed reads as pid 0. */
static void read_record(const char *record_path, pid_t *out_pid, char *out_reason,
                        size_t reason_bytes)
{
    char line[CLAIM_PATH_MAX];
    FILE *file;

    *out_pid = 0;
    if (reason_bytes > 0u)
        out_reason[0] = '\0';

    file = fopen(record_path, "r");
    if (!file)
        return;

    while (fgets(line, (int)sizeof(line), file) != NULL) {
        size_t length = strlen(line);

        while (length > 0u && (line[length - 1u] == '\n' || line[length - 1u] == '\r'))
            line[--length] = '\0';

        if (strncmp(line, "pid ", 4) == 0) {
            *out_pid = (pid_t)strtol(line + 4, NULL, 10);
        } else if (strncmp(line, "reason ", 7) == 0) {
            copy_bounded(out_reason, reason_bytes, line + 7);
        }
    }

    (void)fclose(file);
}

int ft_tt_udrv_claim_namespace_init(ft_tt_udrv_claim_namespace *out_namespace, const char *root,
                                    const char *card, const char *name, uint32_t count)
{
    if (!out_namespace || count == 0u)
        return FT_TT_UDRV_EINVAL;

    if (!is_path_component(card, FT_TT_UDRV_CLAIM_NAMESPACE_MAX))
        return FT_TT_UDRV_EINVAL;

    if (!is_path_component(name, FT_TT_UDRV_CLAIM_NAMESPACE_MAX))
        return FT_TT_UDRV_EINVAL;

    memset(out_namespace, 0, sizeof(*out_namespace));
    copy_bounded(out_namespace->root, sizeof(out_namespace->root),
                 root ? root : FT_TT_UDRV_CLAIM_DEFAULT_ROOT);
    copy_bounded(out_namespace->card, sizeof(out_namespace->card), card);
    copy_bounded(out_namespace->name, sizeof(out_namespace->name), name);
    out_namespace->count = count;

    if (out_namespace->root[0] == '\0')
        return FT_TT_UDRV_EINVAL;

    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_claim_acquire(const ft_tt_udrv_claim_namespace *ns, uint32_t index,
                             const char *reason, ft_tt_udrv_claim **out_claim)
{
    char lock_path[CLAIM_PATH_MAX];
    char record_path[CLAIM_PATH_MAX];
    ft_tt_udrv_claim *claim = NULL;
    int fd = -1;
    int rc;

    if (!ns || !out_claim || index >= ns->count)
        return FT_TT_UDRV_EINVAL;

    if (reason && strlen(reason) >= FT_TT_UDRV_CLAIM_REASON_MAX)
        return FT_TT_UDRV_EINVAL;

    rc = claim_paths(ns, index, lock_path, sizeof(lock_path), record_path, sizeof(record_path));
    if (rc != FT_TT_UDRV_OK)
        return rc;

    /* O_CLOEXEC: a claim that survived exec() would be held by a process that
     * does not know it holds it, and released at a moment nobody chose. */
    fd = open(lock_path, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (fd < 0)
        return FT_TT_UDRV_EIO;
    widen_claim_file(fd);

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        rc = (errno == EWOULDBLOCK || errno == EINTR) ? FT_TT_UDRV_EAGAIN : FT_TT_UDRV_EIO;
        (void)close(fd);
        return rc;
    }

    claim = calloc(1u, sizeof(*claim));
    if (!claim) {
        (void)close(fd); /* releases the flock */
        return FT_TT_UDRV_ENOMEM;
    }

    claim->ns = *ns;
    claim->index = index;
    claim->fd = fd;

    write_record(record_path, reason);

    *out_claim = claim;
    return FT_TT_UDRV_OK;
}

int ft_tt_udrv_claim_acquire_any(const ft_tt_udrv_claim_namespace *ns, const uint32_t *reserved,
                                 size_t reserved_count, const char *reason,
                                 ft_tt_udrv_claim **out_claim)
{
    uint32_t index;

    if (!ns || !out_claim)
        return FT_TT_UDRV_EINVAL;

    if (reserved_count > 0u && !reserved)
        return FT_TT_UDRV_EINVAL;

    for (index = 0u; index < ns->count; index++) {
        bool is_reserved = false;
        size_t i;
        int rc;

        for (i = 0u; i < reserved_count; i++) {
            if (reserved[i] == index) {
                is_reserved = true;
                break;
            }
        }

        if (is_reserved)
            continue;

        rc = ft_tt_udrv_claim_acquire(ns, index, reason, out_claim);
        if (rc == FT_TT_UDRV_OK)
            return FT_TT_UDRV_OK;

        /* EAGAIN means somebody else holds it; anything else is a real error
         * and must not be hidden by trying the next index. */
        if (rc != FT_TT_UDRV_EAGAIN)
            return rc;
    }

    return FT_TT_UDRV_EAGAIN;
}

uint32_t ft_tt_udrv_claim_index(const ft_tt_udrv_claim *claim)
{
    return claim ? claim->index : UINT32_MAX;
}

int ft_tt_udrv_claim_release(ft_tt_udrv_claim *claim)
{
    char lock_path[CLAIM_PATH_MAX];
    char record_path[CLAIM_PATH_MAX];
    int rc = FT_TT_UDRV_OK;

    if (!claim)
        return FT_TT_UDRV_OK;

    if (claim_paths(&claim->ns, claim->index, lock_path, sizeof(lock_path), record_path,
                    sizeof(record_path)) == FT_TT_UDRV_OK) {
        if (unlink(record_path) != 0 && errno != ENOENT)
            rc = FT_TT_UDRV_EIO;
    }

    /* Closing is what releases the lock; do it last so the record is gone
     * before the resource can be re-claimed by anyone else. */
    if (close(claim->fd) != 0)
        rc = FT_TT_UDRV_EIO;

    free(claim);
    return rc;
}

int ft_tt_udrv_claim_scan(const ft_tt_udrv_claim_namespace *ns, ft_tt_udrv_claim_info *out_info,
                          size_t info_cap, uint32_t *out_held)
{
    uint32_t index;
    uint32_t held = 0u;

    if (!ns || !out_info || info_cap < (size_t)ns->count)
        return FT_TT_UDRV_EINVAL;

    for (index = 0u; index < ns->count; index++) {
        char lock_path[CLAIM_PATH_MAX];
        char record_path[CLAIM_PATH_MAX];
        ft_tt_udrv_claim_info *info = &out_info[index];
        int fd;
        int rc;

        memset(info, 0, sizeof(*info));
        info->index = index;

        rc = claim_paths(ns, index, lock_path, sizeof(lock_path), record_path,
                         sizeof(record_path));
        if (rc != FT_TT_UDRV_OK)
            return rc;

        read_record(record_path, &info->pid, info->reason, sizeof(info->reason));
        info->pid_alive = (info->pid > 0) && (kill(info->pid, 0) == 0 || errno == EPERM);

        fd = open(lock_path, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        if (fd < 0)
            return FT_TT_UDRV_EIO;
        widen_claim_file(fd);

        /* The lock is the truth. A record can lie in both directions: orphaned
         * after a crash, or absent because the holder died before writing it. */
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) {
            info->held = false;
            (void)flock(fd, LOCK_UN);
        } else {
            info->held = true;
            held++;
        }

        (void)close(fd);

        /* Sweep the orphan: the lock is free and the record names a pid that
         * is gone, so the record is stale and would otherwise mislead the next
         * person to read a report. */
        if (!info->held && info->pid > 0 && !info->pid_alive) {
            (void)unlink(record_path);
            info->pid = 0;
            info->reason[0] = '\0';
        }
    }

    if (out_held)
        *out_held = held;

    return FT_TT_UDRV_OK;
}
