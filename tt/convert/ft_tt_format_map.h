/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_format_map.h
 * @brief A per-tensor FORMAT MAP: named matmul tensors moved between Bfp8_b and Bfp4_b
 *   on top of a ledger rung. Parser, lookup and the caution a CLI must print.
 *
 * ---------------------------------------------------------------------------
 * THE FILE FORMAT (shared with Pantheon's PN_MIXW_BFP4 knob)
 * ---------------------------------------------------------------------------
 * One entry per line:
 *
 *     <gguf tensor name> <fttw format name>
 *
 * The format is `bfp4_b` or `bfp8_b`, spelled as ft_tt_ledger_format_name() prints it.
 * `#` starts a comment that runs to the end of the line; blank lines are ignored; fields
 * are separated by spaces or tabs. Anything else is a parse error naming the line: a
 * missing or extra field, any other format, a name of FT_TT_FORMAT_MAP_NAME_BYTES or more,
 * and a name listed twice (two lines for one tensor is a map nobody can read at a glance).
 *
 * Names are GGUF SOURCE names, whatever the artifact calls the tensor: an MTP artifact's
 * `mtp.eh_proj_e` / `mtp.eh_proj_h` both follow `blk.L.nextn.eh_proj.weight`. A tied
 * checkpoint's head is written from `token_embd`, which a map may not name, so it cannot be
 * mapped.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS FILE DOES NOT DO
 * ---------------------------------------------------------------------------
 * It parses text. Whether a name is a tensor of the checkpoint, and whether that tensor may
 * be mapped (ft_tt_ledger_map_refusal()), is the converter's check, against the GGUF, before
 * it creates its output. The converter reads only the data structure below, so it links
 * nothing it did not link before; a reader of the map (the CLI, Pantheon) links this file.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_TT_FORMAT_MAP_H
#define FT_TT_FORMAT_MAP_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ft_tt_precision_ledger.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Name field width; matches the GGUF and FTTW name limits. */
#define FT_TT_FORMAT_MAP_NAME_BYTES 128

/** Room for where a map came from, as its messages name it. */
#define FT_TT_FORMAT_MAP_SOURCE_BYTES 256

/** Room for a parse error, which names the source and the line. */
#define FT_TT_FORMAT_MAP_ERROR_BYTES 512

/** Status codes. */
enum {
    FT_TT_FORMAT_MAP_OK = 0,
    FT_TT_FORMAT_MAP_ENOMEM = -2,
    FT_TT_FORMAT_MAP_EINVAL = -3,
    FT_TT_FORMAT_MAP_EIO = -4,
    FT_TT_FORMAT_MAP_EPARSE = -5 /**< the text is not a format map; `error` says where */
};

/** One line of a map. */
typedef struct {
    char name[FT_TT_FORMAT_MAP_NAME_BYTES];
    ft_tt_tensor_format format; /**< FT_TT_FORMAT_BFP8_B or FT_TT_FORMAT_BFP4_B */
    uint32_t line;              /**< 1-based, for messages */
} ft_tt_format_map_entry;

/** A parsed map. Zero it before the first parse; free with ft_tt_format_map_free(). */
typedef struct {
    ft_tt_format_map_entry *entries;
    uint32_t count;
    uint32_t capacity;
    char source[FT_TT_FORMAT_MAP_SOURCE_BYTES]; /**< file path, or the label given to parse */
} ft_tt_format_map;

/**
 * Parse map text into `map`, replacing what it held.
 *
 * @param text         the text; need not be NUL-terminated
 * @param length       bytes of `text`
 * @param source       label for messages (a path); may be NULL
 * @param map          receives the entries; left empty on failure
 * @param error        receives "<source>:<line>: <what>" on EPARSE; may be NULL
 * @param error_bytes  size of `error`
 * @return FT_TT_FORMAT_MAP_OK, EPARSE, ENOMEM or EINVAL
 *
 * Thread-safe: yes (no shared state)    Reentrant: yes    Signal-safe: no
 */
int ft_tt_format_map_parse(const char *text, size_t length, const char *source, ft_tt_format_map *map, char *error,
                           size_t error_bytes);

/**
 * Read and parse a map file. As ft_tt_format_map_parse(), with `source` = `path`, plus EIO
 * (named in `error`) when the file cannot be read.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_format_map_load(const char *path, ft_tt_format_map *map, char *error, size_t error_bytes);

/** Free the entries and zero `map`. Safe on a zeroed map. */
void ft_tt_format_map_free(ft_tt_format_map *map);

/** How many entries carry `format`. */
uint32_t ft_tt_format_map_count(const ft_tt_format_map *map, ft_tt_tensor_format format);

/**
 * The sentence a CLI MUST print when it converts with a map: it names the map, its counts
 * and the rung under it, and says the artifact is not that rung. Same contract as
 * ft_tt_ledger_mode_caution(): the cost travels with the choice.
 *
 * @return the length snprintf() reports
 */
int ft_tt_format_map_caution(const ft_tt_format_map *map, ft_tt_ledger_mode mode, char *out, size_t out_bytes);

/**
 * The entry naming `name`, or NULL. Inline so the converter can look a name up without
 * linking this file (see the header).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
static inline const ft_tt_format_map_entry *ft_tt_format_map_find(const ft_tt_format_map *map, const char *name)
{
    if (!map || !name)
        return NULL;
    for (uint32_t i = 0; i < map->count; i++)
        if (strcmp(map->entries[i].name, name) == 0)
            return &map->entries[i];
    return NULL;
}

#ifdef __cplusplus
}
#endif

#endif /* FT_TT_FORMAT_MAP_H */
