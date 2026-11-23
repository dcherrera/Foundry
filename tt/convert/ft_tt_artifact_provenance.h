/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_artifact_provenance.h
 * @brief The record a cached FTTW artifact carries about the checkpoint and the settings
 *   it was produced from — appended to the artifact, so it is the artifact's OWN bytes.
 *
 * ---------------------------------------------------------------------------
 * WHY THE ARTIFACT CARRIES THIS AND NOT A SIDECAR
 * ---------------------------------------------------------------------------
 * A cache that decides freshness from a file NAME, an mtime, or a `.json` beside the file
 * is deciding it from something the artifact does not know about itself. This project has
 * twice recorded a green measurement taken on a stale artifact, and the reason both times
 * was the same: a stale v1 artifact is byte-for-byte the SAME SIZE as the v2 that should
 * have replaced it, so nothing short of reading its content could tell them apart.
 *
 * So the record goes INSIDE the file, after the directory, with its magic as the LAST FOUR
 * BYTES — findable by one seek to the end, and impossible to have without the bytes.
 *
 * ---------------------------------------------------------------------------
 * WHY IT IS APPENDED RATHER THAN PUT IN THE HEADER
 * ---------------------------------------------------------------------------
 * `ft_tt_artifact_header` is 40 bytes of on-disk contract that a shipping reader
 * (`ft_tt_fttw_open`) and every existing artifact already agree on. Extending it would
 * make every artifact ever written unreadable by the new reader or the reverse. The
 * trailer sits past `directory_offset + tensor_count * FT_TT_ARTIFACT_ENTRY_BYTES`, which
 * no reader walks — so an artifact with a trailer is still an ordinary v2 artifact, and an
 * artifact WITHOUT one is simply not cache-managed and gets reconverted. Neither direction
 * is a format break.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS CATCHES, AND WHAT IT DOES NOT — STATED, NOT IMPLIED
 * ---------------------------------------------------------------------------
 * CATCHES: a different checkpoint (the directory digest), the same checkpoint re-quantized
 * (digest again, since types and shapes are in it), a truncated or grown source, a
 * different rung, a different rotary pairing, an artifact written by an older artifact
 * version, a torn or truncated artifact, and a trailer edited by hand.
 *
 * DOES NOT CATCH: a converter change that alters VALUES without altering any shape, format
 * or flag. Nothing derived from the source can see that. `FT_TT_CONVERT_BEHAVIOUR_STAMP`
 * is the manual half of the answer and MUST be bumped by whoever changes what the packer
 * writes. It is the same contract a shader cache runs on, and the same failure mode.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_TT_ARTIFACT_PROVENANCE_H
#define FT_TT_ARTIFACT_PROVENANCE_H

#include <stddef.h>
#include <stdint.h>

#include "ft_gguf_map.h"

#ifdef __cplusplus
extern "C" {
#endif

/** 'FTTP', little-endian, and the LAST four bytes of a cache-managed artifact. */
#define FT_TT_PROVENANCE_MAGIC 0x50545446u

/** Layout version of the trailer itself. */
#define FT_TT_PROVENANCE_VERSION 1u

/**
 * The converter's BEHAVIOUR stamp.
 *
 * Bump this whenever a change makes the converter write different BYTES for the same input
 * at the same rung — a packing fix, a tilizer change, a rounding change. Shapes, formats
 * and flags are checked structurally and do not need it; values are not, and this is the
 * only thing standing between a value-level fix and every cached artifact on disk still
 * being the old one.
 */
#define FT_TT_CONVERT_BEHAVIOUR_STAMP "ft_tt_convert/2026-08-18/bfp-pack-1"

/** Status codes. */
enum {
    FT_TT_PROVENANCE_OK = 0,
    FT_TT_PROVENANCE_ERROR = -1,
    FT_TT_PROVENANCE_EINVAL = -3,
    FT_TT_PROVENANCE_EIO = -4,
    /** The file ends without a trailer: not a cache-managed artifact. */
    FT_TT_PROVENANCE_EABSENT = -12,
    /** A trailer is present and describes something other than what was asked for. */
    FT_TT_PROVENANCE_ESTALE = -13
};

/**
 * The trailer. Fixed width, naturally aligned, no padding — checked by a static assertion
 * in the implementation rather than assumed.
 */
typedef struct {
    uint64_t source_bytes;        /**< total bytes of the source checkpoint, all parts */
    uint64_t source_digest;       /**< FNV-1a-64 over the source's tensor directory */
    uint64_t payload_bytes;       /**< the artifact header's `total_bytes` at write time */
    uint32_t provenance_version;  /**< FT_TT_PROVENANCE_VERSION */
    uint32_t artifact_version;    /**< the FTTW version the writer wrote */
    uint32_t ledger_mode;         /**< ft_tt_ledger_mode */
    uint32_t rope_pairing;        /**< ft_tt_rope_pairing */
    uint32_t artifact_flags;      /**< the artifact header's `flags` */
    uint32_t tensor_count;        /**< the artifact header's `tensor_count` */
    uint32_t behaviour_stamp;     /**< FNV-1a-32 of FT_TT_CONVERT_BEHAVIOUR_STAMP */
    uint32_t trailer_bytes;       /**< sizeof(ft_tt_artifact_provenance) */
    uint32_t magic;               /**< FT_TT_PROVENANCE_MAGIC; the file's last four bytes */
    uint32_t reserved;            /**< zero; keeps the struct 8-byte aligned with no padding */
} ft_tt_artifact_provenance;

/**
 * Digest a checkpoint's tensor directory: every tensor's name, dimension count, dimensions
 * and ggml type, in directory order, plus the part count. (Not the byte count — the map
 * derives that from the dimensions and the type, so it carries nothing the rest does not.)
 *
 * No tensor byte is read, so this costs milliseconds on a 50 GiB split checkpoint — which
 * is what makes it usable on every serve start instead of only at conversion time.
 *
 * @param map  an open GGUF map
 * @return     the digest, or 0 when `map` is NULL (0 is also a legal digest value; callers
 *             compare digests, they never test one for truth)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (touches the mmap)
 */
uint64_t ft_tt_artifact_source_digest(const ft_gguf_map *map);

/** FNV-1a-32 of FT_TT_CONVERT_BEHAVIOUR_STAMP, computed once per call. */
uint32_t ft_tt_artifact_behaviour_stamp(void);

/**
 * Append a trailer to an artifact that has just been written.
 *
 * Reads the artifact's own header back to fill `payload_bytes`, `artifact_version`,
 * `artifact_flags` and `tensor_count`, so the trailer describes THE FILE rather than the
 * caller's intention about it.
 *
 * @param artifact_path  a complete FTTW artifact
 * @param source_bytes   total bytes of the source checkpoint
 * @param source_digest  from ft_tt_artifact_source_digest()
 * @param ledger_mode    the rung it was converted at
 * @param rope_pairing   the rotary pairing it was converted with
 * @return               FT_TT_PROVENANCE_OK, or EIO/EINVAL
 *
 * Thread-safe: no (writes a file)    Reentrant: no    Signal-safe: no
 */
int ft_tt_artifact_provenance_append(const char *artifact_path, uint64_t source_bytes,
                                     uint64_t source_digest, uint32_t ledger_mode,
                                     uint32_t rope_pairing);

/**
 * Read the trailer from the end of an artifact.
 *
 * @param artifact_path  path to read
 * @param out            zeroed, then filled on success
 * @return               FT_TT_PROVENANCE_OK;
 *                       FT_TT_PROVENANCE_EABSENT when the file is too short or the last
 *                       four bytes are not the magic;
 *                       FT_TT_PROVENANCE_ESTALE when the magic is there but the trailer
 *                       does not describe itself consistently (size or version);
 *                       FT_TT_PROVENANCE_EIO on a read failure
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_artifact_provenance_read(const char *artifact_path, ft_tt_artifact_provenance *out);

/**
 * Compare what an artifact says about itself against what the caller needs, and NAME the
 * first field that disagrees.
 *
 * The name matters as much as the verdict: "stale" with no field is the shape of message
 * that has sent someone to fix the wrong thing here before.
 *
 * @param want       the record a fresh conversion would produce
 * @param got        the record read from the artifact
 * @param why        receives the disagreeing field and both values; "" on a match
 * @param why_bytes  size of `why`
 * @return           FT_TT_PROVENANCE_OK on a full match, FT_TT_PROVENANCE_ESTALE otherwise
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_artifact_provenance_compare(const ft_tt_artifact_provenance *want,
                                      const ft_tt_artifact_provenance *got, char *why,
                                      size_t why_bytes);

/**
 * Check that the artifact agrees with its OWN trailer — length, header version, flags and
 * tensor count.
 *
 * This is the half of the check that needs no source checkpoint: it catches a truncated
 * write, an artifact something appended to, and a trailer moved onto a different file.
 *
 * @param artifact_path  path to check
 * @param record         the trailer already read from that file
 * @param why            receives what disagreed; "" on a match
 * @param why_bytes      size of `why`
 * @return               FT_TT_PROVENANCE_OK, FT_TT_PROVENANCE_ESTALE, or EIO/EINVAL
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_artifact_provenance_verify_file(const char *artifact_path,
                                          const ft_tt_artifact_provenance *record, char *why,
                                          size_t why_bytes);

#ifdef __cplusplus
}
#endif

#endif /* FT_TT_ARTIFACT_PROVENANCE_H */
