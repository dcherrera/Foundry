/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_artifact_cache.c
 * @brief Implementation of the convert-on-load artifact cache.
 *
 * @status None
 * @issues None
 * @todo None
 */

#define _DEFAULT_SOURCE

#include "ft_tt_artifact_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------------- */
/* Paths                                                                     */
/* ------------------------------------------------------------------------- */

/** The suffix that names a rung in an artifact filename. */
static const char *cache_rung_suffix(ft_tt_ledger_mode mode)
{
    const char *name = ft_tt_ledger_mode_name(mode);

    return name ? name : "unknown";
}

/** The last path separator, or NULL. Only '/' — this tree is POSIX everywhere. */
static const char *cache_basename(const char *path)
{
    const char *slash = strrchr(path, '/');

    return slash ? slash + 1 : path;
}

int ft_tt_artifact_cache_path(const char *gguf_path, const char *cache_dir,
                              ft_tt_ledger_mode mode, char *out, size_t out_bytes)
{
    const char *stem_start;
    size_t stem_length;
    int written;

    if (!gguf_path || !out || out_bytes == 0u)
        return FT_TT_ARTIFACT_CACHE_EINVAL;

    stem_start = (cache_dir && *cache_dir) ? cache_basename(gguf_path) : gguf_path;
    stem_length = strlen(stem_start);
    if (stem_length > 5u && strcmp(stem_start + stem_length - 5u, ".gguf") == 0)
        stem_length -= 5u;

    if (cache_dir && *cache_dir)
        written = snprintf(out, out_bytes, "%s/%.*s.%s.fttw", cache_dir, (int)stem_length,
                           stem_start, cache_rung_suffix(mode));
    else
        written = snprintf(out, out_bytes, "%.*s.%s.fttw", (int)stem_length, stem_start,
                           cache_rung_suffix(mode));

    if (written < 0 || (size_t)written >= out_bytes)
        return FT_TT_ARTIFACT_CACHE_EINVAL;
    return FT_TT_ARTIFACT_CACHE_OK;
}

/** Total bytes of the checkpoint, every part of a split set included. */
static uint64_t cache_source_bytes(const ft_gguf_map *map)
{
    uint32_t parts = ft_gguf_map_part_count(map);
    uint64_t total = 0;
    uint32_t i;

    for (i = 0; i < parts; i++) {
        const char *path = ft_gguf_map_part_path(map, i);
        struct stat info;

        if (path && stat(path, &info) == 0)
            total += (uint64_t)info.st_size;
    }
    return total;
}

/* ------------------------------------------------------------------------- */
/* Rotary pairing                                                            */
/* ------------------------------------------------------------------------- */

/** Does the checkpoint declare all four keys the split-half permutation needs? */
static int cache_declares_rope_geometry(const ft_gguf_map *map, const char *architecture)
{
    static const char *const SUFFIXES[] = {"attention.head_count", "attention.head_count_kv",
                                           "attention.key_length", "rope.dimension_count"};
    size_t i;

    for (i = 0; i < sizeof(SUFFIXES) / sizeof(SUFFIXES[0]); i++) {
        char key[128];
        uint64_t value;

        if (snprintf(key, sizeof(key), "%s.%s", architecture, SUFFIXES[i]) >= (int)sizeof(key))
            return 0;
        if (ft_gguf_map_metadata_u64(map, key, &value) != FT_GGUF_OK)
            return 0;
    }
    return 1;
}

ft_tt_rope_pairing ft_tt_artifact_cache_rope_pairing(const ft_gguf_map *map,
                                                     ft_tt_model_family family,
                                                     const char *architecture,
                                                     const char **out_reason)
{
    const char *reason;
    ft_tt_rope_pairing pairing;

    if (family == FT_TT_FAMILY_QWEN) {
        pairing = FT_TT_ROPE_PAIRING_SOURCE;
        reason = "this family already stores q/k in split-half order";
    } else if (map && architecture && *architecture &&
               cache_declares_rope_geometry(map, architecture)) {
        pairing = FT_TT_ROPE_PAIRING_SPLIT_HALF;
        reason = "a llama-family checkpoint stores q/k interleaved, which costs 22% "
                 "perplexity on chat-templated prompts";
    } else {
        /* Announced, not silent. The alternative is refusing a checkpoint for lacking
         * metadata the user has never heard of, which is exactly the "your file is the
         * wrong format" answer this whole phase exists to remove. */
        pairing = FT_TT_ROPE_PAIRING_SOURCE;
        reason = "the checkpoint does not declare its head geometry, so the split-half "
                 "permutation is undefined for it";
    }
    if (out_reason)
        *out_reason = reason;
    return pairing;
}

/* ------------------------------------------------------------------------- */
/* The built-in progress report                                              */
/* ------------------------------------------------------------------------- */

/**
 * A first load of a large model is a PROGRESS BAR, NOT A QUESTION.
 *
 * On a terminal that is a redrawn line; in a log it is one line per decile, because a log
 * full of carriage returns is a log nobody reads. Both carry the same three numbers, so a
 * run recorded from a pipe is as auditable as one watched live.
 */
typedef struct {
    uint64_t total_source_bytes;
    uint64_t done_source_bytes;
    uint32_t tensors_done;
    uint32_t tensors_total;
    int last_decile;
    int interactive;
    double started_at;
    double last_drawn_at;
    const char *model_label;
} cache_progress_state;

static double cache_monotonic_seconds(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static void cache_progress_draw(cache_progress_state *state, int final)
{
    double elapsed = cache_monotonic_seconds() - state->started_at;
    double fraction = state->total_source_bytes > 0u
                          ? (double)state->done_source_bytes / (double)state->total_source_bytes
                          : 0.0;
    double rate = elapsed > 0.01 ? (double)state->done_source_bytes / elapsed : 0.0;
    double remaining = rate > 1.0 ? ((double)state->total_source_bytes -
                                     (double)state->done_source_bytes) / rate
                                  : 0.0;

    fprintf(stderr, "%s%s: preparing for the card  %3d%%  %.2f/%.2f GiB  %.0f MiB/s  eta %.0fs%s",
            state->interactive ? "\r" : "", state->model_label, (int)(fraction * 100.0 + 0.5),
            (double)state->done_source_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)state->total_source_bytes / (1024.0 * 1024.0 * 1024.0),
            rate / (1024.0 * 1024.0), remaining,
            (state->interactive && !final) ? "" : "\n");
    fflush(stderr);
}

static int cache_progress_report(const ft_tt_convert_tensor_report *report, void *user)
{
    cache_progress_state *state = (cache_progress_state *)user;
    double now;

    if (!state || !report)
        return 0;

    state->done_source_bytes += report->source_bytes;
    state->tensors_done++;

    if (state->interactive) {
        /* Redrawing per tensor is 300+ writes for a small model and thousands for a large
         * one; ten a second is smooth and costs nothing. */
        now = cache_monotonic_seconds();
        if (now - state->last_drawn_at < 0.1 && state->tensors_done != state->tensors_total)
            return 0;
        state->last_drawn_at = now;
        cache_progress_draw(state, state->tensors_done == state->tensors_total);
        return 0;
    }

    {
        int decile = state->total_source_bytes > 0u
                         ? (int)((state->done_source_bytes * 10u) / state->total_source_bytes)
                         : 10;

        if (decile > state->last_decile) {
            state->last_decile = decile;
            cache_progress_draw(state, decile >= 10);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------- */
/* The cache                                                                 */
/* ------------------------------------------------------------------------- */

/** Fill the record a fresh conversion of THIS checkpoint at THESE settings would carry. */
static void cache_expected_provenance(ft_tt_artifact_provenance *want, const ft_gguf_map *map,
                                      uint64_t source_bytes, ft_tt_ledger_mode mode,
                                      ft_tt_rope_pairing rope_pairing)
{
    ft_tt_convert_options options;

    /* The options the cache's own conversion runs with (cache_convert): the version and the
     * entry count follow from them, and a fresh conversion must match what is asked here. */
    memset(&options, 0, sizeof(options));
    options.mode = mode;
    options.rope_pairing = rope_pairing;
    memset(want, 0, sizeof(*want));
    want->source_bytes = source_bytes;
    want->source_digest = ft_tt_artifact_source_digest(map);
    want->provenance_version = FT_TT_PROVENANCE_VERSION;
    want->artifact_version = ft_tt_convert_artifact_version_for(map, &options);
    want->ledger_mode = (uint32_t)mode;
    want->rope_pairing = (uint32_t)rope_pairing;
    want->artifact_flags =
        (rope_pairing == FT_TT_ROPE_PAIRING_SPLIT_HALF ||
         rope_pairing == FT_TT_ROPE_PAIRING_SPLIT_HALF_REVERSED)
            ? FT_TT_ARTIFACT_FLAG_ROPE_SPLIT_HALF
            : 0u;
    if (rope_pairing == FT_TT_ROPE_PAIRING_SPLIT_HALF_REVERSED ||
        rope_pairing == FT_TT_ROPE_PAIRING_INTERLEAVED_REVERSED)
        want->artifact_flags |= FT_TT_ARTIFACT_FLAG_ROPE_PAIRS_REVERSED;
    /* A tied checkpoint's artifact carries one more tensor than the GGUF: the head the
     * converter writes from token_embd. An artifact cached before that has one tensor too
     * few, so this count is also what retires it. A trailing MTP block is NOT in the trunk
     * artifact (FT_TT_CONVERT_SCOPE_TRUNK), so it is not counted either. */
    want->tensor_count = ft_tt_convert_tensor_count_for(map, &options);
    want->behaviour_stamp = ft_tt_artifact_behaviour_stamp();
    want->magic = FT_TT_PROVENANCE_MAGIC;
}

/**
 * Is the artifact at `path` the one this checkpoint needs?
 *
 * Every answer is derived from the artifact's OWN BYTES — its trailer, its FTTW header and
 * its length. Nothing here trusts the filename, the mtime, or the fact that a file exists.
 *
 * @return 1 when it is usable; 0 with `note` naming what disagreed
 */
static int cache_artifact_is_current(const char *path, const ft_tt_artifact_provenance *want,
                                     ft_tt_artifact_provenance *got, char *note,
                                     size_t note_bytes)
{
    int rc = ft_tt_artifact_provenance_read(path, got);

    if (rc == FT_TT_PROVENANCE_EIO) {
        snprintf(note, note_bytes, "no cached artifact");
        return 0;
    }
    if (rc == FT_TT_PROVENANCE_EABSENT) {
        snprintf(note, note_bytes,
                 "the cached artifact carries no provenance trailer — it predates the cache "
                 "or was produced by hand");
        return 0;
    }
    if (rc != FT_TT_PROVENANCE_OK) {
        snprintf(note, note_bytes, "the cached artifact's provenance trailer is malformed");
        return 0;
    }
    if (ft_tt_artifact_provenance_verify_file(path, got, note, note_bytes) != FT_TT_PROVENANCE_OK)
        return 0;
    if (ft_tt_artifact_provenance_compare(want, got, note, note_bytes) != FT_TT_PROVENANCE_OK)
        return 0;

    note[0] = '\0';
    return 1;
}

/** Convert into a temp file beside the target and rename it into place. */
static int cache_produce(const char *gguf_path, const char *artifact_path,
                         const ft_tt_artifact_cache_request *request, ft_tt_ledger_mode mode,
                         ft_tt_rope_pairing rope_pairing, const ft_gguf_map *map,
                         uint64_t source_bytes, char *reason, size_t reason_bytes)
{
    char temp_path[FT_TT_ARTIFACT_CACHE_PATH_BYTES];
    cache_progress_state progress_state;
    ft_tt_convert_options options;
    int written;
    int rc;

    written = snprintf(temp_path, sizeof(temp_path), "%s.tmp.%ld", artifact_path,
                       (long)getpid());
    if (written < 0 || (size_t)written >= sizeof(temp_path)) {
        snprintf(reason, reason_bytes, "the artifact path is too long to write beside");
        return FT_TT_ARTIFACT_CACHE_EINVAL;
    }

    memset(&options, 0, sizeof(options));
    options.mode = mode;
    options.rope_pairing = rope_pairing;

    memset(&progress_state, 0, sizeof(progress_state));
    if (request->progress) {
        options.progress = request->progress;
        options.progress_user = request->progress_user;
    } else if (request->progress_style == FT_TT_CACHE_PROGRESS_STDERR) {
        progress_state.total_source_bytes = source_bytes;
        progress_state.tensors_total = ft_gguf_map_tensor_count(map) +
                                       (ft_tt_convert_tied_head_source(map) ? 1u : 0u);
        progress_state.interactive = isatty(2) ? 1 : 0;
        progress_state.started_at = cache_monotonic_seconds();
        progress_state.model_label = cache_basename(gguf_path);
        progress_state.last_decile = -1;
        options.progress = cache_progress_report;
        options.progress_user = &progress_state;
    }

    rc = ft_tt_convert_gguf(gguf_path, temp_path, &options);
    if (rc != FT_TT_CONVERT_OK) {
        remove(temp_path);
        snprintf(reason, reason_bytes,
                 "converting %.400s at %s failed (ft_tt_convert_gguf rc %d)", gguf_path,
                 ft_tt_ledger_mode_name(mode), rc);
        return FT_TT_ARTIFACT_CACHE_ECONVERT;
    }

    rc = ft_tt_artifact_provenance_append(temp_path, source_bytes,
                                          ft_tt_artifact_source_digest(map), (uint32_t)mode,
                                          (uint32_t)rope_pairing);
    if (rc != FT_TT_PROVENANCE_OK) {
        remove(temp_path);
        snprintf(reason, reason_bytes,
                 "the artifact converted but its provenance trailer could not be written; "
                 "an artifact with no trailer would be reconverted on every launch, so it "
                 "was discarded rather than kept");
        return FT_TT_ARTIFACT_CACHE_EIO;
    }

    /* rename() is the atomic step: until it runs there is no artifact at the cache path at
     * all, so a converter killed mid-write can never leave a plausible short one. */
    if (rename(temp_path, artifact_path) != 0) {
        remove(temp_path);
        snprintf(reason, reason_bytes, "could not move the converted artifact into %.400s",
                 artifact_path);
        return FT_TT_ARTIFACT_CACHE_EIO;
    }
    return FT_TT_ARTIFACT_CACHE_OK;
}

int ft_tt_artifact_cache_ensure(const ft_tt_artifact_cache_request *request,
                                ft_tt_artifact_cache_result *result)
{
    ft_tt_artifact_provenance want;
    ft_tt_rope_pairing rope_pairing;
    ft_tt_model_family family;
    ft_tt_ledger_mode mode;
    ft_gguf_map *map = NULL;
    char architecture[64];
    int rc;

    if (!request || !result || !request->gguf_path)
        return FT_TT_ARTIFACT_CACHE_EINVAL;

    memset(result, 0, sizeof(*result));

    if (ft_gguf_map_open(request->gguf_path, &map) != FT_GGUF_OK || !map) {
        snprintf(result->reason, sizeof(result->reason),
                 "%.400s could not be read as a GGUF checkpoint", request->gguf_path);
        return FT_TT_ARTIFACT_CACHE_ESOURCE;
    }

    architecture[0] = '\0';
    ft_gguf_map_metadata_string(map, "general.architecture", architecture, sizeof(architecture));
    family = ft_tt_ledger_family_from_architecture(architecture);
    result->source_bytes = cache_source_bytes(map);

    if (request->rope_pairing >= 0) {
        rope_pairing = (ft_tt_rope_pairing)request->rope_pairing;
        result->rope_reason = "forced by the caller";
    } else {
        rope_pairing = ft_tt_artifact_cache_rope_pairing(map, family, architecture,
                                                         &result->rope_reason);
    }
    result->rope_pairing = rope_pairing;

    rc = ft_tt_capacity_select(map, family, &request->budget, &result->selection);
    if (rc == FT_TT_CAPACITY_EUNCLASSIFIED) {
        snprintf(result->reason, sizeof(result->reason), "%s", result->selection.reason);
        ft_gguf_map_close(map);
        return FT_TT_ARTIFACT_CACHE_EUNCLASSIFIED;
    }
    if (rc == FT_TT_CAPACITY_ENOFIT) {
        snprintf(result->reason, sizeof(result->reason), "%s", result->selection.reason);
        ft_gguf_map_close(map);
        return FT_TT_ARTIFACT_CACHE_ENOFIT;
    }
    if (rc != FT_TT_CAPACITY_OK) {
        snprintf(result->reason, sizeof(result->reason),
                 "the capacity planner refused this checkpoint (rc %d)", rc);
        ft_gguf_map_close(map);
        return FT_TT_ARTIFACT_CACHE_ERROR;
    }

    /* Forcing a rung is an instrument. It still goes through the fit check above, so a
     * forced rung that does not fit is refused with the same arithmetic as an automatic
     * one — an override may pick a rung, it may not overrule the card. */
    mode = request->forced_mode >= 0 ? (ft_tt_ledger_mode)request->forced_mode
                                     : result->selection.mode;
    if (request->forced_mode >= 0) {
        uint32_t i;

        for (i = 0; i < result->selection.rung_count; i++) {
            if (result->selection.rungs[i].mode != mode)
                continue;
            if (result->selection.rungs[i].artifact_bytes + result->selection.reserve_bytes >
                request->budget.card_bytes) {
                snprintf(result->reason, sizeof(result->reason),
                         "%s was forced but does not fit: %.2f GiB of weights plus %.2f GiB "
                         "reserved against a %.3f GiB card",
                         ft_tt_ledger_mode_name(mode),
                         (double)result->selection.rungs[i].artifact_bytes /
                             (1024.0 * 1024.0 * 1024.0),
                         (double)result->selection.reserve_bytes / (1024.0 * 1024.0 * 1024.0),
                         (double)request->budget.card_bytes / (1024.0 * 1024.0 * 1024.0));
                ft_gguf_map_close(map);
                return FT_TT_ARTIFACT_CACHE_ENOFIT;
            }
            result->selection.chosen_index = i;
            result->selection.mode = mode;
            break;
        }
    }

    if (ft_tt_artifact_cache_path(request->gguf_path, request->cache_dir, mode,
                                  result->artifact_path,
                                  sizeof(result->artifact_path)) != FT_TT_ARTIFACT_CACHE_OK) {
        snprintf(result->reason, sizeof(result->reason),
                 "the cached artifact's path would be longer than %u bytes",
                 (unsigned)FT_TT_ARTIFACT_CACHE_PATH_BYTES);
        ft_gguf_map_close(map);
        return FT_TT_ARTIFACT_CACHE_EINVAL;
    }

    cache_expected_provenance(&want, map, result->source_bytes, mode, rope_pairing);

    {
        struct stat existing;

        /* Whether a file was THERE is a different fact from whether it was USABLE, and the
         * caller needs both: "reconverting because the cached artifact is stale" and "there
         * was nothing to reconvert from" are different sentences to print. */
        result->had_artifact = stat(result->artifact_path, &existing) == 0 ? 1 : 0;
    }

    if (cache_artifact_is_current(result->artifact_path, &want, &result->provenance,
                                  result->note, sizeof(result->note))) {
        ft_gguf_map_close(map);
        return FT_TT_ARTIFACT_CACHE_OK;
    }

    rc = cache_produce(request->gguf_path, result->artifact_path, request, mode, rope_pairing,
                       map, result->source_bytes, result->reason, sizeof(result->reason));
    if (rc != FT_TT_ARTIFACT_CACHE_OK) {
        ft_gguf_map_close(map);
        return rc;
    }
    result->produced = 1;

    /* Read the trailer BACK off the file we just wrote and check it against `want` — the
     * same check a cache hit passes. A conversion whose output would not satisfy the next
     * launch is a cache that reconverts forever, and it is better to learn that now, from
     * the bytes, than to wonder later why nothing is ever cached. */
    {
        char recheck_note[FT_TT_ARTIFACT_CACHE_NOTE_BYTES];

        if (!cache_artifact_is_current(result->artifact_path, &want, &result->provenance,
                                       recheck_note, sizeof(recheck_note))) {
            snprintf(result->reason, sizeof(result->reason),
                     "the artifact was converted but does not validate against the checkpoint "
                     "it came from: %.300s", recheck_note);
            ft_gguf_map_close(map);
            return FT_TT_ARTIFACT_CACHE_ECONVERT;
        }
    }

    ft_gguf_map_close(map);
    return FT_TT_ARTIFACT_CACHE_OK;
}
