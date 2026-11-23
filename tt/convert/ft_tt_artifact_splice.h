/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_artifact_splice.h
 * @brief Splice precomputed tensor payloads (MXTL tile files) into a copy of an FTTW
 *   artifact, without the source checkpoint, streaming, never artifact-resident.
 *
 * @status None
 * @issues None
 * @todo None
 */

/*
 * ---------------------------------------------------------------------------
 * WHAT IT IS FOR
 * ---------------------------------------------------------------------------
 * A tier artifact whose FFN matrices are GPTQ-rounded Bfp4_b and whose every other tensor is
 * the Bfp8_b rung's, when the Q8_0 checkpoint the rung was converted from is not at hand.
 * GPTQ writes the tiles (ft_tt_tile_file.h); this puts them in.
 *
 * ---------------------------------------------------------------------------
 * WHAT THE OUTPUT IS — THE CONVERTER'S BYTES, PLUS ONE RECORD
 * ---------------------------------------------------------------------------
 * Bytes [0, header.total_bytes) are laid out exactly as ft_tt_convert_gguf() lays out an
 * artifact: the 40-byte header, zeros to `data_offset`, every entry's payload in DIRECTORY
 * order at the next 64-byte boundary with zero padding, then the directory. A replaced entry
 * takes the tile file's payload and format; its offset, and every later entry's, is
 * recomputed, as are `directory_offset` and `total_bytes`. Nothing else in the header or a
 * directory entry changes: the grid, layout, role and dims of a tensor do not depend on its
 * block-float format. So splicing the converter's own Bfp4_b encoding of tensors X, Y into
 * the `bfp8` rung gives the bytes `--format-map X,Y=bfp4_b` gives, which is the gate.
 *
 * After the directory comes the SPLICE RECORD, the artifact's own account of what was put
 * in, with its magic as the last four bytes (the provenance trailer's convention):
 *
 *   per replaced entry, in directory order (FT_TT_SPLICE_ENTRY_BYTES = 176):
 *     char name[128] (NUL-padded) | u32 format | u32 layout | u64 n_bytes | u8 sha256[32]
 *   footer (FT_TT_SPLICE_FOOTER_BYTES = 128):
 *     u8 base_provenance[64]       the source's FTTP trailer verbatim; zeros if it had none
 *     u8 base_directory_sha256[32] sha256 of the source's directory bytes
 *     u64 base_total_bytes         the source header's total_bytes
 *     u32 entry_count | u32 flags (bit 0: base_provenance present) |
 *     u32 record_bytes | u32 version = 1 | u32 reserved = 0 | u32 magic 'FTTS'
 *
 * ---------------------------------------------------------------------------
 * WHY THE OUTPUT CARRIES NO PROVENANCE TRAILER, AND WHY IT STILL LOADS
 * ---------------------------------------------------------------------------
 * Every FTTW reader bounds the directory and ignores bytes past it: ft_tt_fttw_open()
 * (the card loader), Pantheon's FTTW overlay, ft_tt_fttw_list. So the record costs a reader
 * nothing. The provenance trailer (ft_tt_artifact_provenance.h) is deliberately NOT carried
 * forward: it certifies "this is rung R of checkpoint C", and a spliced artifact is not a
 * rung — the cache must never adopt it as one. Without a trailer the cache reads it as "not
 * cache-managed" (ft_tt_artifact_provenance_read() -> EABSENT), which is the same thing it
 * says of a `--format-map` artifact. The base's trailer survives inside the record, so where
 * the base came from is not lost. A source that already carries a splice record is refused:
 * splice from the unspliced base, so one record describes every replaced tensor.
 */

#ifndef FT_TT_ARTIFACT_SPLICE_H
#define FT_TT_ARTIFACT_SPLICE_H

#include <stddef.h>
#include <stdint.h>

#include "ft_sha256.h"
#include "ft_tt_weight_convert.h"

#ifdef __cplusplus
extern "C" {
#endif

/** 'FTTS', little-endian: the last four bytes of a spliced artifact. */
#define FT_TT_SPLICE_MAGIC 0x53545446u

/** Layout version of the splice record. */
#define FT_TT_SPLICE_VERSION 1u

/** On-disk bytes of one record entry and of the footer. */
#define FT_TT_SPLICE_ENTRY_BYTES 176u
#define FT_TT_SPLICE_FOOTER_BYTES 128u

/** Footer `flags`: the source carried a provenance trailer, kept in `base_provenance`. */
#define FT_TT_SPLICE_FLAG_BASE_PROVENANCE 0x1u

/** Room for a refusal: it names the tile file, the tensor and both sides of the mismatch. */
#define FT_TT_SPLICE_ERROR_BYTES 768u

/** Status codes. Every refusal leaves no output file. */
enum {
    FT_TT_SPLICE_OK = 0,
    FT_TT_SPLICE_ERROR = -1,
    FT_TT_SPLICE_ENOMEM = -2,
    FT_TT_SPLICE_EINVAL = -3,
    FT_TT_SPLICE_EIO = -4,
    FT_TT_SPLICE_ESOURCE = -5,    /**< the source is not a well-formed, unspliced FTTW artifact */
    FT_TT_SPLICE_ETILEFILE = -6,  /**< a tile file's reader refused it (structure, sha256, names) */
    FT_TT_SPLICE_ENAME = -7,      /**< a tile file names a tensor the source does not have */
    FT_TT_SPLICE_EROLE = -8,      /**< a tensor ft_tt_ledger_map_refusal() keeps off block float */
    FT_TT_SPLICE_EFORMAT = -9,    /**< a format other than the map formats, bfp4_b and bfp8_b */
    FT_TT_SPLICE_ELAYOUT = -10,   /**< a layout other than the source entry's */
    FT_TT_SPLICE_ESIZE = -11,     /**< n_bytes != the converter's size for that shape and format */
    FT_TT_SPLICE_EDUPLICATE = -12, /**< one tensor in two tile files */
    FT_TT_SPLICE_ESAMEPATH = -13, /**< the output is the source or a tile file */
    FT_TT_SPLICE_EABSENT = -14    /**< (record read) the artifact carries no splice record */
};

/** One record entry, as read back. */
typedef struct {
    char name[FT_TT_ARTIFACT_NAME_BYTES];
    uint32_t format;
    uint32_t layout;
    uint64_t n_bytes;
    unsigned char sha256[FT_SHA256_BYTES];
} ft_tt_splice_record_entry;

/** A splice record. Zero it before reading; free with ft_tt_splice_record_free(). */
typedef struct {
    ft_tt_splice_record_entry *entries;
    uint32_t entry_count;
    uint32_t flags;
    unsigned char base_provenance[64];
    unsigned char base_directory_sha256[FT_SHA256_BYTES];
    uint64_t base_total_bytes;
} ft_tt_splice_record;

/**
 * Write `output_path`: `source_path` with every entry named in the tile files replaced.
 *
 * Refused before any output exists (see the status codes): a malformed or already-spliced
 * source, a malformed tile file, a name the source lacks, a role a format map may not move,
 * a format other than bfp4_b/bfp8_b, a layout other than the source entry's, a payload size
 * other than the converter's for that shape, format and layout, a tensor in two tile files,
 * and an output that is the source or a tile file (same path, or same inode). The output is
 * written to `<output_path>.tmp.<pid>` and renamed into place, so a failure mid-write leaves
 * no file at `output_path`.
 *
 * Memory: one copy buffer and the directory. The source and the payloads are streamed.
 *
 * @param tile_paths   MXTL files, at least one
 * @param error        receives the refusal; may be NULL
 * @return FT_TT_SPLICE_OK, or a negative status
 *
 * Thread-safe: no (writes a file)    Reentrant: no    Signal-safe: no
 */
int ft_tt_artifact_splice(const char *source_path, const char *const *tile_paths, uint32_t tile_path_count,
                          const char *output_path, char *error, size_t error_bytes);

/**
 * Write an MXTL file holding the named entries of an artifact, each with its own format,
 * layout and payload bytes — the payloads the converter wrote. The inverse of a splice, for
 * building tile files from a converter run (and for the equivalence gate).
 *
 * @return FT_TT_SPLICE_OK; ENAME for a name the artifact lacks; ESOURCE for a malformed
 *         artifact; ETILEFILE/EIO when the tile file cannot be written
 *
 * Thread-safe: no (writes a file)    Reentrant: no    Signal-safe: no
 */
int ft_tt_artifact_extract_tiles(const char *artifact_path, const char *const *names, uint32_t name_count,
                                 const char *tile_path, char *error, size_t error_bytes);

/**
 * Read a spliced artifact's record.
 *
 * @param record  zeroed, then filled; entries owned by the caller via ft_tt_splice_record_free()
 * @return FT_TT_SPLICE_OK; EABSENT when the file does not end in 'FTTS'; ESOURCE when the
 *         record is inconsistent with itself or with the artifact's header; EIO; ENOMEM
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_splice_record_read(const char *artifact_path, ft_tt_splice_record *record);

/** Free a record's entries and zero it. Safe on a zeroed record. */
void ft_tt_splice_record_free(ft_tt_splice_record *record);

#ifdef __cplusplus
}
#endif

#endif /* FT_TT_ARTIFACT_SPLICE_H */
