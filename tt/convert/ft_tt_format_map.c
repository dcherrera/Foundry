/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_format_map.c
 * @brief Implementation of the format-map parser, its counts and its caution.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "ft_tt_format_map.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/** A map file larger than this is not a hand- or tool-written map of one model. */
#define FORMAT_MAP_MAX_FILE_BYTES (16u * 1024u * 1024u)

/** Most fields a line is scanned for; a third one is already the error. */
#define FORMAT_MAP_LINE_FIELDS 3

typedef struct {
    const char *start;
    size_t length;
} text_field;

static void format_map_error(char *error, size_t error_bytes, const char *source, uint32_t line, const char *what,
                             ...) __attribute__((format(printf, 5, 6)));

static void format_map_error(char *error, size_t error_bytes, const char *source, uint32_t line, const char *what,
                             ...)
{
    va_list args;
    int used;

    if (!error || error_bytes == 0)
        return;
    used = line ? snprintf(error, error_bytes, "%s:%u: ", source, line) : snprintf(error, error_bytes, "%s: ", source);
    if (used < 0 || (size_t)used >= error_bytes)
        return;
    va_start(args, what);
    vsnprintf(error + used, error_bytes - (size_t)used, what, args);
    va_end(args);
}

static int is_field_space(char c)
{
    return c == ' ' || c == '\t' || c == '\r';
}

/** Split one line (comment already cut) into up to FORMAT_MAP_LINE_FIELDS fields; returns how many. */
static int split_line_fields(const char *line, size_t length, text_field *fields)
{
    size_t at = 0;
    int count = 0;

    while (at < length) {
        while (at < length && is_field_space(line[at]))
            at++;
        if (at == length)
            break;
        if (count == FORMAT_MAP_LINE_FIELDS)
            return count + 1;
        fields[count].start = line + at;
        while (at < length && !is_field_space(line[at]))
            at++;
        fields[count].length = (size_t)(line + at - fields[count].start);
        count++;
    }
    return count;
}

static int map_reserve_entry(ft_tt_format_map *map)
{
    ft_tt_format_map_entry *grown;
    uint32_t capacity;

    if (map->count < map->capacity)
        return FT_TT_FORMAT_MAP_OK;
    capacity = map->capacity ? map->capacity * 2u : 64u;
    grown = realloc(map->entries, (size_t)capacity * sizeof(*grown));
    if (!grown)
        return FT_TT_FORMAT_MAP_ENOMEM;
    map->entries = grown;
    map->capacity = capacity;
    return FT_TT_FORMAT_MAP_OK;
}

/** Parse one line's fields into a new entry, or say why the line is not one. */
static int parse_map_line(ft_tt_format_map *map, const text_field *fields, int field_count, uint32_t line,
                          char *error, size_t error_bytes)
{
    char format_word[16];
    ft_tt_format_map_entry *entry;
    int format;

    if (field_count != 2) {
        format_map_error(error, error_bytes, map->source, line,
                         "want '<gguf tensor name> <bfp4_b|bfp8_b>', got %d field%s", field_count,
                         field_count == 1 ? "" : "s");
        return FT_TT_FORMAT_MAP_EPARSE;
    }
    if (fields[0].length >= FT_TT_FORMAT_MAP_NAME_BYTES) {
        format_map_error(error, error_bytes, map->source, line, "tensor name is %zu bytes, the limit is %d",
                         fields[0].length, FT_TT_FORMAT_MAP_NAME_BYTES - 1);
        return FT_TT_FORMAT_MAP_EPARSE;
    }
    snprintf(format_word, sizeof(format_word), "%.*s", (int)fields[1].length, fields[1].start);
    format = fields[1].length < sizeof(format_word) ? ft_tt_ledger_format_from_name(format_word) : -1;
    if (format != FT_TT_FORMAT_BFP4_B && format != FT_TT_FORMAT_BFP8_B) {
        format_map_error(error, error_bytes, map->source, line, "format '%.*s' is not bfp4_b or bfp8_b",
                         (int)fields[1].length, fields[1].start);
        return FT_TT_FORMAT_MAP_EPARSE;
    }
    if (map_reserve_entry(map) != FT_TT_FORMAT_MAP_OK)
        return FT_TT_FORMAT_MAP_ENOMEM;
    entry = &map->entries[map->count];
    snprintf(entry->name, sizeof(entry->name), "%.*s", (int)fields[0].length, fields[0].start);
    entry->format = (ft_tt_tensor_format)format;
    entry->line = line;
    {
        const ft_tt_format_map_entry *earlier = ft_tt_format_map_find(map, entry->name);

        if (earlier) {
            format_map_error(error, error_bytes, map->source, line, "'%s' is already mapped on line %u",
                             entry->name, earlier->line);
            return FT_TT_FORMAT_MAP_EPARSE;
        }
    }
    map->count++;
    return FT_TT_FORMAT_MAP_OK;
}

int ft_tt_format_map_parse(const char *text, size_t length, const char *source, ft_tt_format_map *map, char *error,
                           size_t error_bytes)
{
    size_t at = 0;
    uint32_t line = 0;

    if (!map || (!text && length))
        return FT_TT_FORMAT_MAP_EINVAL;
    ft_tt_format_map_free(map);
    snprintf(map->source, sizeof(map->source), "%s", source ? source : "format map");

    while (at < length) {
        const char *start = text + at;
        const char *end = memchr(start, '\n', length - at);
        size_t line_length = end ? (size_t)(end - start) : length - at;
        const char *comment = memchr(start, '#', line_length);
        text_field fields[FORMAT_MAP_LINE_FIELDS];
        int field_count;
        int rc;

        at += line_length + (end ? 1u : 0u);
        line++;
        if (comment)
            line_length = (size_t)(comment - start);
        field_count = split_line_fields(start, line_length, fields);
        if (field_count == 0)
            continue;
        rc = parse_map_line(map, fields, field_count, line, error, error_bytes);
        if (rc != FT_TT_FORMAT_MAP_OK) {
            char kept[FT_TT_FORMAT_MAP_SOURCE_BYTES];

            memcpy(kept, map->source, sizeof(kept));
            ft_tt_format_map_free(map);
            memcpy(map->source, kept, sizeof(kept));
            return rc;
        }
    }
    return FT_TT_FORMAT_MAP_OK;
}

int ft_tt_format_map_load(const char *path, ft_tt_format_map *map, char *error, size_t error_bytes)
{
    FILE *file;
    char *text;
    size_t length;
    int rc;

    if (!path || !map)
        return FT_TT_FORMAT_MAP_EINVAL;
    file = fopen(path, "rb");
    if (!file) {
        format_map_error(error, error_bytes, path, 0, "cannot open: %s", strerror(errno));
        return FT_TT_FORMAT_MAP_EIO;
    }
    text = malloc(FORMAT_MAP_MAX_FILE_BYTES + 1u);
    if (!text) {
        fclose(file);
        return FT_TT_FORMAT_MAP_ENOMEM;
    }
    length = fread(text, 1, FORMAT_MAP_MAX_FILE_BYTES + 1u, file);
    rc = ferror(file) ? FT_TT_FORMAT_MAP_EIO : FT_TT_FORMAT_MAP_OK;
    fclose(file);
    if (rc == FT_TT_FORMAT_MAP_OK && length > FORMAT_MAP_MAX_FILE_BYTES)
        rc = FT_TT_FORMAT_MAP_EIO;
    if (rc != FT_TT_FORMAT_MAP_OK)
        format_map_error(error, error_bytes, path, 0, "unreadable, or larger than %u bytes",
                         FORMAT_MAP_MAX_FILE_BYTES);
    else
        rc = ft_tt_format_map_parse(text, length, path, map, error, error_bytes);
    free(text);
    return rc;
}

void ft_tt_format_map_free(ft_tt_format_map *map)
{
    if (!map)
        return;
    free(map->entries);
    memset(map, 0, sizeof(*map));
}

uint32_t ft_tt_format_map_count(const ft_tt_format_map *map, ft_tt_tensor_format format)
{
    uint32_t count = 0;

    for (uint32_t i = 0; map && i < map->count; i++)
        count += map->entries[i].format == format ? 1u : 0u;
    return count;
}

int ft_tt_format_map_caution(const ft_tt_format_map *map, ft_tt_ledger_mode mode, char *out, size_t out_bytes)
{
    const char *rung = ft_tt_ledger_mode_name(mode);

    if (!map || !out || out_bytes == 0)
        return -1;
    return snprintf(out, out_bytes,
                    "FORMAT MAP '%s': %u tensor%s at bfp4_b and %u at bfp8_b on top of the '%s' rung. "
                    "This artifact is NOT the '%s' rung: name it by its map, and confirm what it holds "
                    "with ft_tt_fttw_list. An artifact whose dense matmuls mix Bfp8_b and Bfp4_b runs "
                    "one-shot, not resident, until a dual-format resident matmul exists.",
                    map->source, ft_tt_format_map_count(map, FT_TT_FORMAT_BFP4_B),
                    ft_tt_format_map_count(map, FT_TT_FORMAT_BFP4_B) == 1u ? "" : "s",
                    ft_tt_format_map_count(map, FT_TT_FORMAT_BFP8_B), rung, rung);
}
