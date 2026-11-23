/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tensix_claim.c
 * @brief Gate for ft_tt_tensix_claim.c — the per-CORE ownership that lets
 *   correctness work run in parallel instead of serialising behind the
 *   whole-card lock.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS SUITE IS BUILT TO CATCH, AND HOW IT AVOIDS BEING VACUOUS
 * ---------------------------------------------------------------------------
 * Two rules shape every stage here.
 *
 * **The lock is checked directly, never through the claim layer.** A claim
 * verified by asking `ft_tt_tensix_claim_acquire()` again proves only that the
 * module agrees with itself; a build whose acquire is a no-op passes that. So
 * `lock_file_is_free()` opens `<root>/<card>/tensix/<index>.lock` and takes the
 * `flock` itself. That is the kernel's state, and nothing in the module can make
 * it lie. Same reasoning as test_udrv_claim.c's forged-record stage.
 *
 * **The decisive case is two processes racing for one core**, and it is run in
 * two shapes because they catch different bugs. Stage 4 is DETERMINISTIC — a
 * helper holds the middle core of a three-core request, so the refusal and the
 * roll-back happen at a known point and the "nothing was half-claimed" assertion
 * is not a timing accident. Stage 5 is a REAL RACE — four forked children fire at
 * one core off a shared starting gun, and exactly one may win.
 *
 * @status Off-hardware. Filesystem and fork only; no device is opened, no NoC
 *   transaction is possible from anything this links.
 * @issues None
 * @todo None
 */

/* flock(2) is BSD, not POSIX, so a bare _POSIX_C_SOURCE hides it on both
 * platforms this code is built on (DEV_NOTES §5, the _DEFAULT_SOURCE note). */
#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _DEFAULT_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ft_tt_tensix_claim.h"
#include "ft_tt_udrv_claim.h"
#include "ft_tt_test_report.h"
#include "ft_tt_udrv.h"

#define TEST_CARD "0000:12:00.0"
#define RACE_CHILD_COUNT 4u
#define RACE_CORE 9u

static char scratch_root[160];

static void build_scratch_root(void)
{
    (void)snprintf(scratch_root, sizeof(scratch_root), "/tmp/ft_tt_tensix_claim_test.%ld",
                   (long)getpid());
}

static void remove_scratch_root(void)
{
    char command[256];

    (void)snprintf(command, sizeof(command), "rm -rf '%s'", scratch_root);
    if (system(command) != 0)
        (void)fprintf(stderr, "fixture: cleanup of %s failed\n", scratch_root);
}

/**
 * THE INDEPENDENT INSTRUMENT: is the kernel's lock on this core free?
 *
 * Deliberately does not call into ft_tt_tensix_claim or ft_tt_udrv_claim at all.
 * It opens the lock file by path and takes the flock, so a module that forgets to
 * lock, forgets to release, or leaks a descriptor cannot hide from it.
 */
static bool lock_file_is_free(uint32_t index)
{
    char path[256];
    int fd;
    bool free_now;

    (void)snprintf(path, sizeof(path), "%s/%s/%s/%u.lock", scratch_root, TEST_CARD,
                   FT_TT_TENSIX_CLAIM_NAMESPACE, index);

    fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0666);
    if (fd < 0)
        return true; /* no file at all means nobody ever claimed it */

    free_now = (flock(fd, LOCK_EX | LOCK_NB) == 0);
    if (free_now)
        (void)flock(fd, LOCK_UN);
    (void)close(fd);
    return free_now;
}

/** Non-NULL with a count of zero: an EXPLICITLY empty reserved set. NULL means
 *  "read the environment", and a suite whose result depends on the shell it was
 *  launched from is a suite that lies. */
static const uint32_t k_no_reserved[1] = {0u};

/** A request against the scratch root with no reserved set and no retirement. */
static void build_request(ft_tt_tensix_claim_request *request, const uint32_t *indices,
                          size_t count, const char *reason)
{
    memset(request, 0, sizeof(*request));
    request->root = scratch_root;
    request->card = TEST_CARD;
    request->indices = indices;
    request->count = count;
    request->reason = reason;
    request->reserved = k_no_reserved;
    request->reserved_count = 0u;
}

static void stage_parse_index_lists(void)
{
    uint32_t out[16];
    size_t count = 0u;

    ft_tt_test_step("Stage 1 — the core list parser, which decides what a sweep may name");

    ft_tt_test_check_fmt("a single index parses",
                         ft_tt_tensix_claim_parse_indices("71", out, 16u, &count) ==
                                 FT_TT_UDRV_OK &&
                             count == 1u && out[0] == 71u,
                         "count=%zu", count);

    ft_tt_test_check_fmt("a comma list keeps its order",
                         ft_tt_tensix_claim_parse_indices("3,1,17", out, 16u, &count) ==
                                 FT_TT_UDRV_OK &&
                             count == 3u && out[0] == 3u && out[1] == 1u && out[2] == 17u,
                         "count=%zu", count);

    ft_tt_test_check_fmt("an inclusive range expands",
                         ft_tt_tensix_claim_parse_indices("30-33", out, 16u, &count) ==
                                 FT_TT_UDRV_OK &&
                             count == 4u && out[0] == 30u && out[3] == 33u,
                         "count=%zu", count);

    ft_tt_test_check_fmt("ranges and indices mix",
                         ft_tt_tensix_claim_parse_indices("1,4-6,9", out, 16u, &count) ==
                                 FT_TT_UDRV_OK &&
                             count == 5u && out[1] == 4u && out[3] == 6u && out[4] == 9u,
                         "count=%zu", count);

    ft_tt_test_check("an empty string is zero indices, not an error",
                     ft_tt_tensix_claim_parse_indices("", out, 16u, &count) == FT_TT_UDRV_OK &&
                         count == 0u,
                     "unset means no reserved cores");

    ft_tt_test_check("a NULL string is zero indices",
                     ft_tt_tensix_claim_parse_indices(NULL, out, 16u, &count) == FT_TT_UDRV_OK &&
                         count == 0u,
                     "");

    ft_tt_test_check("an inverted range is refused",
                     ft_tt_tensix_claim_parse_indices("9-3", out, 16u, &count) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    ft_tt_test_check("trailing junk is refused",
                     ft_tt_tensix_claim_parse_indices("1,2x", out, 16u, &count) ==
                         FT_TT_UDRV_EINVAL,
                     "a typo must not silently claim fewer cores");

    ft_tt_test_check("a space is refused",
                     ft_tt_tensix_claim_parse_indices("1, 2", out, 16u, &count) ==
                         FT_TT_UDRV_EINVAL,
                     "more likely a quoting mistake than an intent");

    ft_tt_test_check("an index at the namespace bound is refused",
                     ft_tt_tensix_claim_parse_indices("256", out, 16u, &count) ==
                         FT_TT_UDRV_EINVAL,
                     "");

    ft_tt_test_check("overflowing the caller's array is refused, not truncated",
                     ft_tt_tensix_claim_parse_indices("0-99", out, 16u, &count) ==
                         FT_TT_UDRV_EINVAL,
                     "a truncated set is a set that claims the wrong cores");
}

static void stage_set_ownership(void)
{
    static const uint32_t wanted[] = {40u, 41u, 42u};
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;

    ft_tt_test_step("Stage 2 — a set is claimed, and the KERNEL agrees it is claimed");

    ft_tt_test_check("all three cores are free before we start",
                     lock_file_is_free(40u) && lock_file_is_free(41u) && lock_file_is_free(42u),
                     "checked by flock, not by asking the module");

    build_request(&request, wanted, 3u, "stage 2 set");
    ft_tt_test_check("the set is granted",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_OK, "");
    ft_tt_test_check_fmt("it holds three cores", ft_tt_tensix_claim_count(set) == 3u, "%zu",
                         ft_tt_tensix_claim_count(set));
    ft_tt_test_check_fmt("...in the order asked for",
                         ft_tt_tensix_claim_index_at(set, 0u) == 40u &&
                             ft_tt_tensix_claim_index_at(set, 2u) == 42u,
                         "%u..%u", ft_tt_tensix_claim_index_at(set, 0u),
                         ft_tt_tensix_claim_index_at(set, 2u));

    ft_tt_test_check("the kernel says all three locks are taken",
                     !lock_file_is_free(40u) && !lock_file_is_free(41u) && !lock_file_is_free(42u),
                     "flock refused on each");
    ft_tt_test_check("a core NOT in the set is untouched", lock_file_is_free(43u), "");

    ft_tt_tensix_claim_release(set);
    ft_tt_test_check("releasing gives all three back",
                     lock_file_is_free(40u) && lock_file_is_free(41u) && lock_file_is_free(42u),
                     "flock granted on each");

    ft_tt_tensix_claim_release(NULL);
    ft_tt_test_check("releasing NULL is a no-op", true, "returned");
}

static void stage_bad_requests(void)
{
    static const uint32_t twice[] = {7u, 12u, 7u};
    static const uint32_t out_of_range[] = {FT_TT_TENSIX_CLAIM_INDEX_COUNT};
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;

    ft_tt_test_step("Stage 3 — malformed requests are refused BEFORE anything is locked");

    build_request(&request, twice, 3u, "duplicate");
    ft_tt_test_check("a core named twice is refused",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_EINVAL,
                     "it would self-deadlock and report itself as the holder");
    ft_tt_test_check_fmt("...and the refusal names the duplicate",
                         refusal.outcome == FT_TT_TENSIX_CLAIM_DUPLICATE && refusal.index == 7u,
                         "index %u", refusal.index);
    ft_tt_test_check("...and nothing was locked", lock_file_is_free(7u) && lock_file_is_free(12u),
                     "");

    build_request(&request, out_of_range, 1u, "oob");
    ft_tt_test_check("an index at the namespace bound is refused",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_EINVAL,
                     "");

    build_request(&request, twice, 0u, "empty");
    ft_tt_test_check("an empty request is refused",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_EINVAL,
                     "claiming nothing is not a claim");
}

/** The other job: claim `index`, say so down the pipe, then wait to be killed
 *  WITHOUT releasing anything. Never returns. */
static void holder_child(uint32_t index, const char *reason, int ready_pipe[2])
{
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;
    uint32_t held_index = index;
    char signal_byte = 1;

    (void)close(ready_pipe[0]);
    build_request(&request, &held_index, 1u, reason);

    if (ft_tt_tensix_claim_acquire(&request, &set, &refusal) != FT_TT_UDRV_OK)
        _exit(3);
    if (write(ready_pipe[1], &signal_byte, 1u) != 1)
        _exit(4);
    for (;;)
        pause();
}

/**
 * Fork the holder above.
 *
 * A pipe handshake rather than a poll loop: a parent that polls by ATTEMPTING the
 * claim can win the race itself, and the stage then goes red for a reason that
 * has nothing to do with the mechanism (test_udrv_claim.c learned this one).
 *
 * @return the child pid, or -1
 */
static pid_t fork_holder(uint32_t index, const char *reason, int ready_pipe[2])
{
    pid_t child;

    if (pipe(ready_pipe) != 0)
        return -1;

    child = fork();
    if (child == 0)
        holder_child(index, reason, ready_pipe);

    (void)close(ready_pipe[1]);
    return child;
}

static void stage_no_half_claim(void)
{
    static const uint32_t wanted[] = {20u, 21u, 22u};
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;
    int ready_pipe[2];
    char ready = 0;
    pid_t holder;
    int status = 0;

    ft_tt_test_step("Stage 4 — THE DECISIVE ONE: a refused set leaves NOTHING half-claimed");

    holder = fork_holder(21u, "the other job", ready_pipe);
    if (holder < 0) {
        ft_tt_test_check("fork succeeded", false, "fork failed");
        return;
    }
    ft_tt_test_check("the other process holds core 21",
                     read(ready_pipe[0], &ready, 1u) == 1 && ready == 1, "handshake");
    (void)close(ready_pipe[0]);
    ft_tt_test_check("...and the kernel agrees", !lock_file_is_free(21u), "flock refused");

    /* 20 is taken first, 21 refuses, 22 is never reached. The roll-back is what
     * must give 20 back — and it is checked by flock, not by re-asking. */
    build_request(&request, wanted, 3u, "the sweep that loses");
    ft_tt_test_check("the whole set is refused",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_EAGAIN,
                     "one core of three was taken");
    ft_tt_test_check_fmt("the refusal names the core it lost on",
                         refusal.outcome == FT_TT_TENSIX_CLAIM_HELD && refusal.index == 21u,
                         "outcome=%d index=%u", (int)refusal.outcome, refusal.index);
    ft_tt_test_check_fmt("...and names the process holding it", refusal.pid == holder,
                         "pid %ld vs holder %ld", (long)refusal.pid, (long)holder);
    ft_tt_test_check_fmt("...and the reason that process registered",
                         strcmp(refusal.reason, "the other job") == 0, "%s", refusal.reason);

    ft_tt_test_check("core 20 — claimed then rolled back — is FREE", lock_file_is_free(20u),
                     "the kernel's answer, not the module's");
    ft_tt_test_check("core 22 — never reached — is FREE", lock_file_is_free(22u), "");

    (void)kill(holder, SIGKILL);
    (void)waitpid(holder, &status, 0);

    ft_tt_test_check("SIGKILL alone returns core 21", lock_file_is_free(21u),
                     "no cleanup path of ours ran");
    ft_tt_test_check("...so the same set now succeeds",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_OK, "");
    ft_tt_tensix_claim_release(set);
}

/** One racer: wait for the gun, claim RACE_CORE, report the outcome down a pipe. */
static void race_child(int gate_read, int report_write)
{
    static uint32_t index = RACE_CORE;
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;
    char gun = 0;
    char line[64];
    int rc;
    int written;

    if (read(gate_read, &gun, 1u) != 1)
        _exit(9);

    build_request(&request, &index, 1u, "racer");
    rc = ft_tt_tensix_claim_acquire(&request, &set, &refusal);

    written = snprintf(line, sizeof(line), "%ld %d %ld\n", (long)getpid(),
                       rc == FT_TT_UDRV_OK ? 0 : (int)refusal.outcome, (long)refusal.pid);
    if (written > 0) {
        /* glibc marks write() warn_unused_result, and a (void) cast does NOT silence that
         * under gcc -Werror=unused-result (it does under clang, which is why this built on
         * the Mac and broke Makefile.udrv on the card host). This is a child reporting its verdict
         * down a pipe on a path where there is nothing useful to do about a short write. */
        ssize_t reported = write(report_write, line, (size_t)written);
        (void)reported;
    }

    /* A winner must hold the claim long enough for every loser to be refused,
     * otherwise the race is decided by scheduling rather than by flock. */
    if (rc == FT_TT_UDRV_OK) {
        sleep(1u);
        ft_tt_tensix_claim_release(set);
    }
    _exit(0);
}

static void stage_real_race(void)
{
    int gate[2];
    int report[2];
    pid_t children[RACE_CHILD_COUNT];
    char gun[RACE_CHILD_COUNT];
    char buffer[512];
    ssize_t got;
    size_t total = 0u;
    unsigned granted = 0u;
    unsigned held = 0u;
    unsigned other = 0u;
    unsigned named_the_winner = 0u;
    long winner_pid = 0;
    unsigned child;
    char *cursor;

    ft_tt_test_step("Stage 5 — a REAL race: %u processes, one core, one winner",
                    RACE_CHILD_COUNT);

    if (pipe(gate) != 0 || pipe(report) != 0) {
        ft_tt_test_check("pipes created", false, "pipe failed");
        return;
    }

    for (child = 0u; child < RACE_CHILD_COUNT; child++) {
        children[child] = fork();
        if (children[child] == 0) {
            (void)close(gate[1]);
            (void)close(report[0]);
            race_child(gate[0], report[1]);
        }
    }
    (void)close(gate[0]);
    (void)close(report[1]);

    memset(gun, 1, sizeof(gun));
    ft_tt_test_check("the starting gun fires for all of them at once",
                     write(gate[1], gun, sizeof(gun)) == (ssize_t)sizeof(gun), "one write");
    (void)close(gate[1]);

    while (total < sizeof(buffer) - 1u &&
           (got = read(report[0], buffer + total, sizeof(buffer) - 1u - total)) > 0)
        total += (size_t)got;
    buffer[total] = '\0';
    (void)close(report[0]);

    for (child = 0u; child < RACE_CHILD_COUNT; child++) {
        int status = 0;

        (void)waitpid(children[child], &status, 0);
    }

    /* First pass finds the winner, second pass checks every loser named it. */
    for (cursor = buffer; *cursor != '\0';) {
        long reporter = 0;
        int outcome = -1;
        long holder = 0;

        if (sscanf(cursor, "%ld %d %ld", &reporter, &outcome, &holder) == 3) {
            if (outcome == (int)FT_TT_TENSIX_CLAIM_GRANTED) {
                granted++;
                winner_pid = reporter;
            } else if (outcome == (int)FT_TT_TENSIX_CLAIM_HELD) {
                held++;
            } else {
                other++;
            }
        }
        cursor = strchr(cursor, '\n');
        if (!cursor)
            break;
        cursor++;
    }

    for (cursor = buffer; *cursor != '\0';) {
        long reporter = 0;
        int outcome = -1;
        long holder = 0;

        if (sscanf(cursor, "%ld %d %ld", &reporter, &outcome, &holder) == 3 &&
            outcome == (int)FT_TT_TENSIX_CLAIM_HELD && holder == winner_pid)
            named_the_winner++;
        cursor = strchr(cursor, '\n');
        if (!cursor)
            break;
        cursor++;
    }

    ft_tt_test_check_fmt("exactly one process won", granted == 1u, "%u granted", granted);
    ft_tt_test_check_fmt("every other process was told the core is HELD",
                         held == RACE_CHILD_COUNT - 1u && other == 0u, "%u held, %u other", held,
                         other);
    ft_tt_test_check_fmt("...and every refusal named the winner by pid",
                         named_the_winner == RACE_CHILD_COUNT - 1u, "%u of %u named pid %ld",
                         named_the_winner, RACE_CHILD_COUNT - 1u, winner_pid);
    ft_tt_test_check("the core is free once the winner exits", lock_file_is_free(RACE_CORE), "");
}

static void stage_reserved_set(void)
{
    static const uint32_t wanted[] = {30u, 31u};
    static const uint32_t reserved[] = {31u};
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;
    size_t parsed_count = 0u;
    uint32_t parsed[32];

    ft_tt_test_step("Stage 6 — reserved cores: another family's, and a live serve's");

    build_request(&request, wanted, 2u, "sweep into reserved");
    request.reserved = reserved;
    request.reserved_count = 1u;

    ft_tt_test_check("a request naming a reserved core is refused",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_EAGAIN,
                     "");
    ft_tt_test_check_fmt("...and says RESERVED, not HELD",
                         refusal.outcome == FT_TT_TENSIX_CLAIM_RESERVED && refusal.index == 31u,
                         "outcome=%d index=%u", (int)refusal.outcome, refusal.index);
    /* The point of checking 30: reserved is evaluated for the WHOLE request
     * before any lock is taken, so even the core that WOULD have been fine is
     * never touched. */
    ft_tt_test_check("...and core 30, which comes first, was never locked",
                     lock_file_is_free(30u), "reserved is checked before anything is claimed");

    request.reserved_count = 0u;
    ft_tt_test_check("with nothing reserved the same request succeeds",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_OK, "");
    ft_tt_tensix_claim_release(set);

    (void)setenv(FT_TT_TENSIX_CLAIM_RESERVED_ENVIRONMENT, "0-9,14,15,28", 1);
    ft_tt_test_check_fmt("the environment reserved set parses",
                         ft_tt_tensix_claim_reserved_from_environment(parsed, 32u,
                                                                      &parsed_count) ==
                                 FT_TT_UDRV_OK &&
                             parsed_count == 13u,
                         "%zu cores", parsed_count);

    {
        static const uint32_t into_environment_reserved[] = {14u};
        ft_tt_tensix_claim_request env_request;

        build_request(&env_request, into_environment_reserved, 1u, "env reserved");
        env_request.reserved = NULL; /* NULL is what reads the environment */
        ft_tt_test_check("a NULL reserved set reads the environment and refuses",
                         ft_tt_tensix_claim_acquire(&env_request, &set, &refusal) ==
                             FT_TT_UDRV_EAGAIN,
                         "");
        ft_tt_test_check_fmt("...as RESERVED", refusal.outcome == FT_TT_TENSIX_CLAIM_RESERVED,
                             "outcome=%d", (int)refusal.outcome);
    }
    (void)unsetenv(FT_TT_TENSIX_CLAIM_RESERVED_ENVIRONMENT);
}

/* Recovery-on-acquire bookkeeping for stage 7. */
static uint32_t g_retired[8];
static size_t g_retired_count;
static uint32_t g_retire_fails_on = UINT32_MAX;

static int record_retirement(void *context, uint32_t core_index)
{
    (void)context;
    if (g_retired_count < 8u)
        g_retired[g_retired_count++] = core_index;
    return (core_index == g_retire_fails_on) ? -7 : 0;
}

static void stage_recovery_on_acquire(void)
{
    static const uint32_t wanted[] = {50u, 51u, 52u};
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;

    ft_tt_test_step("Stage 7 — recovery-on-acquire: a released claim is not a retired core");

    g_retired_count = 0u;
    g_retire_fails_on = UINT32_MAX;
    build_request(&request, wanted, 3u, "retire ok");
    request.retire = record_retirement;

    ft_tt_test_check("the set is granted",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_OK, "");
    ft_tt_test_check_fmt("every claimed core was offered for retirement, in order",
                         g_retired_count == 3u && g_retired[0] == 50u && g_retired[1] == 51u &&
                             g_retired[2] == 52u,
                         "%zu retirements", g_retired_count);
    ft_tt_tensix_claim_release(set);

    /* A retirement that refuses must release the WHOLE set — a job that cannot
     * get one core into a known state must not run on the others. */
    g_retired_count = 0u;
    g_retire_fails_on = 51u;
    ft_tt_test_check("a set whose middle core cannot be retired is refused",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_EAGAIN,
                     "");
    ft_tt_test_check_fmt("...reported as UNRETIRED, naming that core",
                         refusal.outcome == FT_TT_TENSIX_CLAIM_UNRETIRED && refusal.index == 51u,
                         "outcome=%d index=%u", (int)refusal.outcome, refusal.index);
    ft_tt_test_check("...and ALL THREE locks are back, not just the failing one",
                     lock_file_is_free(50u) && lock_file_is_free(51u) && lock_file_is_free(52u),
                     "checked by flock");
    ft_tt_test_check_fmt("...and retirement stopped at the refusal", g_retired_count == 2u,
                         "%zu retirements", g_retired_count);
    g_retire_fails_on = UINT32_MAX;
}

/**
 * The hazard this project actually has: a harness that `_Exit()`s.
 *
 * `_Exit` runs no atexit handler and no destructor, so nothing of ours releases
 * anything — and the claim comes back anyway, because the kernel closes the
 * descriptor. That is the property flock was chosen for. It is also exactly why
 * `retire` has to exist: the LOCK came back and the CORE did not.
 */
static void stage_exit_without_release(void)
{
    static uint32_t index = 60u;
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;
    pid_t child;
    int status = 0;

    ft_tt_test_step("Stage 8 — a harness that _Exit()s: the LOCK returns, the CORE does not");

    child = fork();
    if (child == 0) {
        ft_tt_tensix_claim_request child_request;
        ft_tt_tensix_claim_refusal child_refusal;
        ft_tt_tensix_claim_set *child_set = NULL;

        build_request(&child_request, &index, 1u, "harness that _Exit()s");
        if (ft_tt_tensix_claim_acquire(&child_request, &child_set, &child_refusal) !=
            FT_TT_UDRV_OK)
            _exit(3);
        _exit(0); /* no release, no cleanup, no unwind */
    }
    if (child < 0) {
        ft_tt_test_check("fork succeeded", false, "fork failed");
        return;
    }
    (void)waitpid(child, &status, 0);
    ft_tt_test_check_fmt("the child claimed and then _Exit()ed",
                         WIFEXITED(status) && WEXITSTATUS(status) == 0, "status %d", status);

    ft_tt_test_check("the lock came back with no cleanup code of ours running",
                     lock_file_is_free(60u), "the kernel closed the descriptor");

    g_retired_count = 0u;
    g_retire_fails_on = UINT32_MAX;
    build_request(&request, &index, 1u, "the next claimant");
    request.retire = record_retirement;
    ft_tt_test_check("the next claimant gets the core",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_OK, "");
    ft_tt_test_check_fmt("...and is made to retire it before using it — that is the whole point",
                         g_retired_count == 1u && g_retired[0] == 60u, "%zu retirements",
                         g_retired_count);
    ft_tt_tensix_claim_release(set);
}

/**
 * A whole-card operation must be able to SEE per-core claims before it resets.
 *
 * The card lock only ever saw processes that registered a shared claim, so once
 * ordinary work moved to per-core claims a reset could report "nobody is using
 * the card" while wiping eight live suites. `ft_tt_core` closes that with a
 * census over this namespace — and the census is only true if it resolves the
 * root and card exactly as a claimant does, which is what this stage pins.
 */
static void stage_card_wide_census_sees_core_claims(void)
{
    static const uint32_t wanted[] = {33u};
    ft_tt_tensix_claim_request request;
    ft_tt_tensix_claim_refusal refusal;
    ft_tt_tensix_claim_set *set = NULL;
    ft_tt_udrv_claim_namespace census;
    ft_tt_udrv_claim_info *info;
    uint32_t held = 0u;

    ft_tt_test_step("Stage 9 — a card-wide operation can see, and name, per-core claims");

    (void)setenv(FT_TT_TENSIX_CLAIM_ROOT_ENVIRONMENT, scratch_root, 1);
    /* NULL-guarded: an unset root is a LEGAL answer from this resolver (it means
     * "ft_tt_udrv_claim's own default"), so a mutation that stops reading the
     * environment must go red here rather than segfault a stage later. */
    {
        const char *resolved = ft_tt_tensix_claim_resolve_root(NULL);

        ft_tt_test_check_fmt("the root resolver honours the environment",
                             resolved != NULL && strcmp(resolved, scratch_root) == 0, "%s",
                             resolved ? resolved : "(null)");
        if (!resolved) {
            (void)unsetenv(FT_TT_TENSIX_CLAIM_ROOT_ENVIRONMENT);
            return; /* the census below would coordinate outside the scratch root */
        }
    }
    ft_tt_test_check("an explicit root still wins over the environment",
                     strcmp(ft_tt_tensix_claim_resolve_root("/explicit"), "/explicit") == 0, "");
    ft_tt_test_check("the card resolver falls back to the single-card default",
                     strcmp(ft_tt_tensix_claim_resolve_card(NULL),
                            FT_TT_TENSIX_CLAIM_CARD_DEFAULT) == 0,
                     "");

    memset(&request, 0, sizeof(request));
    request.root = NULL; /* NULL is what makes the resolver load-bearing here */
    request.card = TEST_CARD;
    request.indices = wanted;
    request.count = 1u;
    request.reason = "a live suite";
    request.reserved = k_no_reserved;

    ft_tt_test_check("a core is claimed through the resolved root",
                     ft_tt_tensix_claim_acquire(&request, &set, &refusal) == FT_TT_UDRV_OK, "");

    /* The census, built the way ft_tt_core builds it — a SEPARATE namespace
     * object over the lower layer, not a second call into ft_tt_tensix_claim. */
    (void)ft_tt_udrv_claim_namespace_init(&census, ft_tt_tensix_claim_resolve_root(NULL),
                                          ft_tt_tensix_claim_resolve_card(TEST_CARD),
                                          FT_TT_TENSIX_CLAIM_NAMESPACE,
                                          FT_TT_TENSIX_CLAIM_INDEX_COUNT);
    info = calloc((size_t)census.count, sizeof(*info));
    if (!info) {
        ft_tt_test_check("census buffer allocates", false, "out of memory");
        ft_tt_tensix_claim_release(set);
        return;
    }
    (void)ft_tt_udrv_claim_scan(&census, info, (size_t)census.count, &held);

    ft_tt_test_check_fmt("the census sees exactly the one claimed core", held == 1u, "%u held",
                         held);
    ft_tt_test_check_fmt("...at the right index, naming this process",
                         info[33].held && info[33].pid == getpid(), "index 33 pid %ld",
                         (long)info[33].pid);
    ft_tt_test_check_fmt("...with the reason it registered",
                         strcmp(info[33].reason, "a live suite") == 0, "%s", info[33].reason);

    free(info);
    ft_tt_tensix_claim_release(set);
    (void)unsetenv(FT_TT_TENSIX_CLAIM_ROOT_ENVIRONMENT);
}

int main(void)
{
    build_scratch_root();
    /* Nothing here may inherit the operator's reserved set: a suite whose result
     * depends on the shell it was launched from is a suite that lies. */
    (void)unsetenv(FT_TT_TENSIX_CLAIM_RESERVED_ENVIRONMENT);

    stage_parse_index_lists();
    stage_set_ownership();
    stage_bad_requests();
    stage_no_half_claim();
    stage_real_race();
    stage_reserved_set();
    stage_recovery_on_acquire();
    stage_exit_without_release();
    stage_card_wide_census_sees_core_claims();

    remove_scratch_root();
    return ft_tt_test_finish();
}
