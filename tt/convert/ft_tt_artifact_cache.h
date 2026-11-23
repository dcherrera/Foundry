/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_artifact_cache.h
 * @brief GGUF in, a card-ready FTTW artifact out — produced once, cached beside the
 *   checkpoint, and never asked about again.
 *
 * ---------------------------------------------------------------------------
 * THE REQUIREMENT THIS EXISTS FOR
 * ---------------------------------------------------------------------------
 * A user downloads a GGUF and our software makes it run. They never see the words
 * "Bfp8_b" or "FTTW", never pick a rung, and are never told their file is the wrong
 * format.
 *
 * What cannot be removed: the card cannot consume GGUF layout. Tensix eats 32x32 tiles in
 * four 16x16 faces and the matmul kernels read a block-float the unpacker decodes
 * natively, so SOME conversion always happens. This makes it invisible and cached — a
 * shader cache, not a decision. The Tensix program blob is the precedent: produce once,
 * cache, use thereafter.
 *
 * ---------------------------------------------------------------------------
 * THE THREE ANSWERS IT CAN GIVE, AND WHY NONE OF THEM IS A SILENT FALLBACK
 * ---------------------------------------------------------------------------
 *  - **hit** — a cached artifact whose own trailer matches this checkpoint and these
 *    settings. Nothing is converted and nothing is uploaded twice.
 *  - **produced** — no artifact, or one whose trailer disagreed. The disagreement is
 *    NAMED in `note` before the reconversion, because "reconverting" with no reason is how
 *    a cache that never hits looks exactly like a cache that works.
 *  - **refused** — the model does not fit at any rung the card can load, or the converter
 *    cannot classify it. Both carry the arithmetic. A refusal here means the caller stays
 *    on the CPU or stops; it must never mean "convert it anyway and hope".
 *
 * ---------------------------------------------------------------------------
 * WRITES ARE ATOMIC, AND THAT IS LOAD-BEARING
 * ---------------------------------------------------------------------------
 * Conversion writes `<artifact>.tmp.<pid>` and rename()s it into place. A converter killed
 * halfway leaves a temp file, never a short artifact at the cache path — so the next
 * launch finds "no artifact" and converts, rather than finding a plausible one and serving
 * a truncated model.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_TT_ARTIFACT_CACHE_H
#define FT_TT_ARTIFACT_CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_artifact_provenance.h"
#include "ft_tt_capacity_plan.h"
#include "ft_tt_weight_convert.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Longest artifact path this library will build. */
#define FT_TT_ARTIFACT_CACHE_PATH_BYTES 1024u

/** Room for one explanatory sentence. */
#define FT_TT_ARTIFACT_CACHE_NOTE_BYTES 512u

/** Status codes. Each one means a different thing for the caller to do. */
enum {
    FT_TT_ARTIFACT_CACHE_OK = 0,
    FT_TT_ARTIFACT_CACHE_ERROR = -1,
    FT_TT_ARTIFACT_CACHE_EINVAL = -3,
    FT_TT_ARTIFACT_CACHE_EIO = -4,
    /** The checkpoint could not be opened or read as GGUF. */
    FT_TT_ARTIFACT_CACHE_ESOURCE = -5,
    /** No rung the card can load leaves room for this model. `reason` has the arithmetic. */
    FT_TT_ARTIFACT_CACHE_ENOFIT = -11,
    /** The converter would refuse this checkpoint: unclassified 2-D tensors. */
    FT_TT_ARTIFACT_CACHE_EUNCLASSIFIED = -9,
    /** A conversion was needed, was attempted, and failed. */
    FT_TT_ARTIFACT_CACHE_ECONVERT = -14
};

/** How the conversion should report itself while it runs. */
typedef enum {
    FT_TT_CACHE_PROGRESS_SILENT = 0,
    /** A percent/rate/ETA line on stderr: a live bar on a terminal, a decile line in a log. */
    FT_TT_CACHE_PROGRESS_STDERR = 1
} ft_tt_cache_progress_style;

/** What to convert, where to keep it, and what it has to fit inside. */
typedef struct {
    const char *gguf_path;   /**< the checkpoint; any part of a split set */
    const char *cache_dir;   /**< NULL or "" keeps the artifact beside the checkpoint */
    ft_tt_capacity_budget budget;
    /** An ft_tt_rope_pairing to force, or a negative value to resolve it from the
     *  checkpoint. The product path leaves it negative: which pairing is CORRECT is a
     *  property of the checkpoint, not a preference, and asking the user would be asking
     *  them a question about a defect they have never heard of. */
    int rope_pairing;
    /** An ft_tt_ledger_mode to force, or a negative value to select automatically. Forcing
     *  is an instrument (and how a bfp4 arm is measured on a model that fits at bfp8); the
     *  product path leaves it negative and never asks the user. */
    int forced_mode;
    ft_tt_cache_progress_style progress_style;
    ft_tt_convert_progress_fn progress;      /**< overrides `progress_style` when set */
    void *progress_user;
} ft_tt_artifact_cache_request;

/** What happened, in enough detail to print one honest line about it. */
typedef struct {
    char artifact_path[FT_TT_ARTIFACT_CACHE_PATH_BYTES];
    ft_tt_capacity_selection selection;
    ft_tt_artifact_provenance provenance;
    uint64_t source_bytes;
    ft_tt_rope_pairing rope_pairing; /**< the pairing that was resolved and used */
    const char *rope_reason;         /**< a static string: why that pairing */
    int produced;      /**< 1 = converted during this call, 0 = the cache answered */
    int had_artifact;  /**< 1 = a file was already at the cache path when this call started */
    char note[FT_TT_ARTIFACT_CACHE_NOTE_BYTES];   /**< why the cached artifact was rejected;
                                                   *   meaningful only when `had_artifact` */
    char reason[FT_TT_ARTIFACT_CACHE_NOTE_BYTES]; /**< the refusal, on a non-OK return */
} ft_tt_artifact_cache_result;

/**
 * The path a checkpoint's artifact is cached at: the checkpoint's name with `.gguf`
 * replaced by `.<rung>.fttw`.
 *
 * The rung is IN THE NAME so two rungs of one checkpoint can coexist — which is what makes
 * a bfp4-vs-bfp8 comparison one command instead of a file shuffle — and so a human reading
 * a directory can see what each artifact is without a tool.
 *
 * @param gguf_path  the checkpoint
 * @param cache_dir  NULL/"" for beside the checkpoint, else a directory
 * @param mode       the rung
 * @param out        destination
 * @param out_bytes  its size
 * @return           FT_TT_ARTIFACT_CACHE_OK, or EINVAL when the path does not fit
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_artifact_cache_path(const char *gguf_path, const char *cache_dir,
                              ft_tt_ledger_mode mode, char *out, size_t out_bytes);

/**
 * The rotary pairing a checkpoint needs, decided from the checkpoint.
 *
 * Phase 7 established that every GGUF `llama`-family checkpoint stores q/k INTERLEAVED and
 * that converting it that way costs +22.1% perplexity on chat-templated prompts, which is
 * every request a serve receives. Split-half is the fix — but it is WRONG for the `qwen`
 * family, which already stores split-half, and it is UNDEFINED for a checkpoint that does
 * not declare its head geometry. So the answer is a property of the file.
 *
 * @param map           an open GGUF map
 * @param family        from ft_tt_ledger_family_from_architecture()
 * @param architecture  the `general.architecture` string, for the metadata key prefix
 * @param out_reason    receives a static string naming why; may be NULL
 * @return              the pairing to convert with
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
ft_tt_rope_pairing ft_tt_artifact_cache_rope_pairing(const ft_gguf_map *map,
                                                     ft_tt_model_family family,
                                                     const char *architecture,
                                                     const char **out_reason);

/**
 * Make sure a card-ready artifact for this checkpoint exists, and say which one it is.
 *
 * Opens the checkpoint, chooses the rung, checks the cache by the cached artifact's own
 * bytes, converts if it must, and returns the path to serve. The caller never chooses a
 * format and never learns one exists unless it prints `result.selection`.
 *
 * @param request  what to convert and what it must fit inside
 * @param result   zeroed then filled; `reason` is set on every non-OK return
 * @return         FT_TT_ARTIFACT_CACHE_OK and a usable `result.artifact_path`, or one of
 *                 the status codes above
 *
 * Thread-safe: no (writes files)    Reentrant: no    Signal-safe: no
 */
int ft_tt_artifact_cache_ensure(const ft_tt_artifact_cache_request *request,
                                ft_tt_artifact_cache_result *result);

#ifdef __cplusplus
}
#endif

#endif /* FT_TT_ARTIFACT_CACHE_H */
