/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_card_lock.c
 * @brief Implementation of the advisory card-level lock (see ft_tt_card_lock.h).
 *
 * @status Runs on the p150a test card. Red/green evidence for the refusal,
 *   acquisition, staleness and override paths is in the recovery notes; a real card-level
 *   reset was never run against this build (concurrent agents were live on the card).
 * @issues None
 * @todo None
 */

/* flock(2) is BSD, not POSIX, so the bare _POSIX_C_SOURCE below hides it -- on
 * macOS outright, and on glibc by clearing __USE_MISC. Same guard and same
 * reasoning as ft_tt_udrv_claim.c, which is this module's sibling: macOS
 * declares BSD interfaces under _DARWIN_C_SOURCE, glibc under _DEFAULT_SOURCE.
 * Without this the file cannot even be syntax-checked off the card host, which is
 * where its off-hardware review has to happen. */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _DEFAULT_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L

#include "ft_tt_card_lock.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "ft_tt_device.h"

/** Single-host coordination only -- see the header for why /tmp is the right place. World
 *  writable (0666 / 0777) on purpose: any user's agent on this box must be able to
 *  register a claim or take the lock, not just whoever created the file first. */
/* Both live inside a directory we create ourselves, and that placement is load-bearing.
 * `/tmp` is sticky and world-writable, and a host with `fs.protected_regular=2` (the card
 * host has it) then refuses to let ROOT open() a regular file in it owned by another user.
 * A card-level reset must run under sudo, so the lock was unopenable by exactly the caller
 * that needs it -- MEASURED on the card host: as a user the flock succeeds, as root the open
 * fails EACCES. protected_regular only applies to sticky world-writable directories, so
 * putting the files one level down in a directory of our own making sidesteps it. */
#define FT_TT_CARD_LOCK_DIR "/tmp/ft_tt_core_card.d"
#define FT_TT_CARD_LOCK_PATH FT_TT_CARD_LOCK_DIR "/lock"
#define FT_TT_CARD_LOCK_CLAIMS_DIR FT_TT_CARD_LOCK_DIR "/claims"

/* Generous enough that GCC's -Wformat-truncation can prove a "<dir>/<name>" join never
 * truncates even at NAME_MAX (255): the claims directory path is ~30 bytes, so 512 leaves
 * a wide margin instead of the tightest bound that would compile today and warn again the
 * next time a path constant grows. */
#define FT_TT_CARD_LOCK_PATH_MAX 512u
#define FT_TT_CARD_LOCK_RECORD_MAX (FT_TT_CARD_LOCK_REASON_MAX + 128u)

/** Bounded copy that cannot trip -Wformat-truncation the way `snprintf(d, n, "%s", s)`
 *  can when the compiler cannot bound `s`: truncates silently, always NUL-terminates,
 *  never touches `dest` if `dest_bytes` is 0. */
static void copy_bounded(char *dest, size_t dest_bytes, const char *src)
{
    size_t length = strlen(src);

    if (dest_bytes == 0u)
        return;
    if (length >= dest_bytes)
        length = dest_bytes - 1u;
    memcpy(dest, src, length);
    dest[length] = '\0';
}

/** This process's own claim-record path: "<claims dir>/<pid>.claim". Deterministic from
 *  getpid(), so release needs no state carried over from acquire. */
static void own_record_path(char *out, size_t out_bytes)
{
    snprintf(out, out_bytes, "%s/%d.claim", FT_TT_CARD_LOCK_CLAIMS_DIR, (int)getpid());
}

/** mkdir the claims directory if it is not already there. EEXIST is success. */
/** The parent both the lock file and the claims directory live in.
 *
 *  Every entry point must call this before touching either, and it must be idempotent:
 *  whoever runs first creates it, and everyone after has to be able to write in it. Mode
 *  0777 explicitly (mkdir honours umask) and chmod'd after, because the first caller may
 *  be root arriving via sudo for a card-level rung, and a plain user must still be able to place
 *  a claim afterwards. Deliberately NOT sticky -- see FT_TT_CARD_LOCK_DIR. */
static int ensure_lock_directory(void)
{
    if (mkdir(FT_TT_CARD_LOCK_DIR, 0777) != 0 && errno != EEXIST)
        return FT_TT_EIO;
    (void)chmod(FT_TT_CARD_LOCK_DIR, 0777);
    return FT_TT_OK;
}

static int ensure_claims_directory(void)
{
    /* The parent first. Creating the child alone fails ENOENT when nobody has taken the
     * lock yet -- which is precisely the common case for `claim`, and it is how the first
     * version of this broke: `claim: could not register (rc=-4)` on a clean host. */
    if (ensure_lock_directory() != FT_TT_OK)
        return FT_TT_EIO;
    if (mkdir(FT_TT_CARD_LOCK_CLAIMS_DIR, 0777) != 0 && errno != EEXIST)
        return FT_TT_EIO;
    /* Same reason as the parent: root may have created it, a user must still write here. */
    (void)chmod(FT_TT_CARD_LOCK_CLAIMS_DIR, 0777);
    return FT_TT_OK;
}

/** Open (creating if needed) the single lock file every holder flock()s. */
static int open_lock_file(void)
{
    int fd;

    if (ensure_lock_directory() != FT_TT_OK)
        return -1;

    fd = open(FT_TT_CARD_LOCK_PATH, O_CREAT | O_RDWR | O_CLOEXEC, 0666);
    if (fd < 0)
        return -1;
    /* open() honours umask, so a restrictive one leaves this owner-only even though 0666
     * was asked for. Best-effort widen it back; harmless if it fails (e.g. another user
     * already owns the file, so we cannot chmod it -- the lock still works, just possibly
     * not across users). */
    (void)fchmod(fd, 0666);
    return fd;
}

/** Write this process's own claim record: pid, exclusive/shared, reason, acquired-at. */
static int write_own_record(bool exclusive, const char *reason)
{
    char path[FT_TT_CARD_LOCK_PATH_MAX];
    char sanitized_reason[FT_TT_CARD_LOCK_REASON_MAX];
    char body[FT_TT_CARD_LOCK_RECORD_MAX];
    size_t char_index;
    size_t body_length;
    int fd;
    ssize_t written;

    if (ensure_claims_directory() != FT_TT_OK)
        return FT_TT_EIO;

    /* Keep the record line-based: a reason containing a newline would otherwise corrupt
     * the next reader's line-by-line parse. */
    if (!reason)
        reason = "(no reason given)";
    copy_bounded(sanitized_reason, sizeof(sanitized_reason), reason);
    for (char_index = 0; sanitized_reason[char_index] != '\0'; char_index++)
        if (sanitized_reason[char_index] == '\n')
            sanitized_reason[char_index] = ' ';

    own_record_path(path, sizeof(path));
    fd = open(path, O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0)
        return FT_TT_EIO;

    snprintf(body, sizeof(body), "pid=%d\nexclusive=%d\nreason=%s\nacquired=%ld\n",
             (int)getpid(), exclusive ? 1 : 0, sanitized_reason, (long)time(NULL));
    body_length = strlen(body);
    written = write(fd, body, body_length);
    close(fd);
    return written == (ssize_t)body_length ? FT_TT_OK : FT_TT_EIO;
}

/** Remove this process's own claim record. Not fatal if it never existed -- release must
 *  be safe to call even when acquire failed partway through. */
static void remove_own_record(void)
{
    char path[FT_TT_CARD_LOCK_PATH_MAX];

    own_record_path(path, sizeof(path));
    (void)unlink(path);
}

/** True if `pid` still names a live process. ESRCH is dead; EPERM (owned by a different
 *  user) is treated as alive since a permission failure does not prove the process is
 *  gone. */
static bool pid_is_alive(pid_t pid)
{
    if (pid <= 0)
        return false;
    if (kill(pid, 0) == 0)
        return true;
    return errno == EPERM;
}

/** Parse one claim-record file into `holder`. Missing fields are left zeroed. */
static bool parse_record(const char *path, ft_tt_card_lock_holder *holder)
{
    char line[256];
    FILE *file = fopen(path, "r");

    if (!file)
        return false;
    memset(holder, 0, sizeof(*holder));
    while (fgets(line, sizeof(line), file)) {
        size_t length = strlen(line);

        if (length > 0 && line[length - 1] == '\n')
            line[length - 1] = '\0';
        if (strncmp(line, "pid=", 4) == 0)
            holder->pid = (pid_t)strtol(line + 4, NULL, 10);
        else if (strncmp(line, "exclusive=", 10) == 0)
            holder->exclusive = strtol(line + 10, NULL, 10) != 0;
        else if (strncmp(line, "reason=", 7) == 0)
            copy_bounded(holder->reason, sizeof(holder->reason), line + 7);
        else if (strncmp(line, "acquired=", 9) == 0)
            holder->acquired_at = (time_t)strtol(line + 9, NULL, 10);
    }
    fclose(file);
    return true;
}

/** True if `name` is "<digits>.claim" -- what own_record_path() writes. */
static bool looks_like_claim_file(const char *name)
{
    size_t length = strlen(name);
    static const char suffix[] = ".claim";
    size_t suffix_length = sizeof(suffix) - 1u;

    if (length <= suffix_length)
        return false;
    return strcmp(name + (length - suffix_length), suffix) == 0;
}

/** Scan the claims directory: collect live holders into `status`, delete dead-pid records.
 *  Succeeds with an empty status if the directory does not exist yet -- that is the normal
 *  "nobody has ever claimed this card" state, not an error. */
static int scan_claims(ft_tt_card_lock_status *status)
{
    DIR *directory;
    struct dirent *entry;

    memset(status, 0, sizeof(*status));
    directory = opendir(FT_TT_CARD_LOCK_CLAIMS_DIR);
    if (!directory)
        return errno == ENOENT ? FT_TT_OK : FT_TT_EIO;

    while ((entry = readdir(directory)) != NULL) {
        char path[FT_TT_CARD_LOCK_PATH_MAX];
        ft_tt_card_lock_holder holder;

        if (!looks_like_claim_file(entry->d_name))
            continue;
        snprintf(path, sizeof(path), "%s/%s", FT_TT_CARD_LOCK_CLAIMS_DIR, entry->d_name);
        if (!parse_record(path, &holder))
            continue;

        if (!pid_is_alive(holder.pid)) {
            (void)unlink(path);
            status->stale_cleaned++;
            continue;
        }
        if (status->holder_count < FT_TT_CARD_LOCK_MAX_HOLDERS)
            status->holders[status->holder_count] = holder;
        status->holder_count++;
    }
    closedir(directory);
    return FT_TT_OK;
}

int ft_tt_card_lock_claim_acquire(const char *reason, int *out_fd)
{
    int fd;

    if (!out_fd)
        return FT_TT_EINVAL;
    *out_fd = -1;

    if (ensure_claims_directory() != FT_TT_OK)
        return FT_TT_EIO;
    fd = open_lock_file();
    if (fd < 0)
        return FT_TT_EIO;

    /* LOCK_SH blocks only behind an exclusive holder, and any number of shared claims
     * coexist -- this call is allowed to block, unlike the exclusive path below. */
    if (flock(fd, LOCK_SH) != 0) {
        close(fd);
        return FT_TT_EIO;
    }
    if (write_own_record(false, reason) != FT_TT_OK) {
        (void)flock(fd, LOCK_UN);
        close(fd);
        return FT_TT_EIO;
    }
    *out_fd = fd;
    return FT_TT_OK;
}

int ft_tt_card_lock_exclusive_try(const char *reason, int *out_fd,
                                  ft_tt_card_lock_status *out_status)
{
    ft_tt_card_lock_status local_status;
    int fd;

    if (!out_fd)
        return FT_TT_EINVAL;
    *out_fd = -1;
    /* BEFORE any early return. A caller that prints the status on a non-OK result would
     * otherwise print this frame's stack -- which is exactly what happened on the card host:
     * the open failed, out_status was never written, and the tool announced it had
     * "cleared 2477232864 stale lock records" and advised waiting for a holder that did
     * not exist. An out-parameter must be defined on EVERY path that returns. */
    if (out_status)
        memset(out_status, 0, sizeof(*out_status));

    fd = open_lock_file();
    if (fd < 0)
        return FT_TT_EIO;

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        /* Contended. Report who -- this scan is also what cleans up any stale (dead-pid)
         * records, whether or not the caller wants to see the result. */
        (void)scan_claims(out_status ? out_status : &local_status);
        return FT_TT_EPERM;
    }
    if (write_own_record(true, reason) != FT_TT_OK) {
        (void)flock(fd, LOCK_UN);
        close(fd);
        return FT_TT_EIO;
    }
    if (out_status)
        memset(out_status, 0, sizeof(*out_status));
    *out_fd = fd;
    return FT_TT_OK;
}

int ft_tt_card_lock_inspect(ft_tt_card_lock_status *out_status)
{
    if (!out_status)
        return FT_TT_EINVAL;
    return scan_claims(out_status);
}

void ft_tt_card_lock_release(int fd)
{
    if (fd < 0)
        return;
    remove_own_record();
    (void)flock(fd, LOCK_UN);
    close(fd);
}

/** "3s" / "4m12s" / "1h03m" -- short, since the refusal message is already busy. */
static void format_duration(time_t seconds, char *out, size_t out_bytes)
{
    long total = (long)seconds;
    long hours;
    long minutes;
    long secs;

    if (total < 0)
        total = 0;
    hours = total / 3600;
    minutes = (total % 3600) / 60;
    secs = total % 60;
    if (hours > 0)
        snprintf(out, out_bytes, "%ldh%02ldm", hours, minutes);
    else if (minutes > 0)
        snprintf(out, out_bytes, "%ldm%02lds", minutes, secs);
    else
        snprintf(out, out_bytes, "%lds", secs);
}

void ft_tt_card_lock_status_print(const ft_tt_card_lock_status *status)
{
    unsigned holder_index;
    time_t now = time(NULL);

    if (!status)
        return;
    if (status->stale_cleaned != 0u)
        printf("  (cleared %u stale lock record%s left by a dead process)\n",
               status->stale_cleaned, status->stale_cleaned == 1u ? "" : "s");
    if (status->holder_count == 0u) {
        printf("  nobody currently holds the card lock\n");
        return;
    }
    for (holder_index = 0; holder_index < status->holder_count &&
                            holder_index < FT_TT_CARD_LOCK_MAX_HOLDERS; holder_index++) {
        char duration[32];
        const ft_tt_card_lock_holder *holder = &status->holders[holder_index];

        format_duration(now - holder->acquired_at, duration, sizeof(duration));
        printf("  pid %d (%s), held %s: %s\n", (int)holder->pid,
               holder->exclusive ? "card-level operation" : "claim", duration,
               holder->reason[0] ? holder->reason : "(no reason given)");
    }
    if (status->holder_count > FT_TT_CARD_LOCK_MAX_HOLDERS)
        printf("  ... and %u more\n", status->holder_count - FT_TT_CARD_LOCK_MAX_HOLDERS);
}
