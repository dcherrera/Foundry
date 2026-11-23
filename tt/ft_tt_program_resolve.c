/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_program_resolve.c
 * @brief Implementation of name-based program resolution (see ft_tt_program_resolve.h).
 *
 *   Its own translation unit, and not part of `ft_tt_runtime.c`, for a reason that is
 *   about the build and not about tidiness: this is the only file that pulls
 *   `ft_tt_program_blob.c` (and through it the whole embedded blob) into a link. A caller
 *   that has not adopted name-based resolution yet links exactly what it linked before.
 *
 * @status Implemented; gated off-device by tests/test_program_blob.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_program_resolve.h"

#include <stdio.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_program_blob.h"

/** Longest `<root>/<name>` this builds. The same bound ft_tt_runtime.c uses. */
#define FT_TT_PROGRAM_RESOLVE_PATH_BYTES 512u

const char *ft_tt_program_source_string(ft_tt_program_source source)
{
    switch (source) {
    case FT_TT_PROGRAM_SOURCE_EMBEDDED:  return "embedded";
    case FT_TT_PROGRAM_SOURCE_DIRECTORY: return "directory";
    case FT_TT_PROGRAM_SOURCE_NONE:      return "none";
    default:                             return "none";
    }
}

/* ================================================================
 * The directory arm
 * ================================================================ */

/** Build `<root>/<name>`, refusing a truncation rather than reading a shortened path. */
static int join_program_path(const char *root, const char *name, char *out, size_t out_bytes)
{
    int written = snprintf(out, out_bytes, "%s/%s", root, name);

    if (written < 0 || (size_t)written >= out_bytes) {
        ft_tt_log_error("tt program: '%s/%s' does not fit in %lu bytes", root, name,
                        (unsigned long)out_bytes);
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

/**
 * The explicit-directory arm of both entry points.
 *
 * NOTHING here falls back to the embedded blob. The caller named a directory; if that
 * directory cannot serve the request, the answer is the reason and not a quiet substitute
 * that makes the next measurement a measurement of something else.
 */
static int open_from_directory(const char *program_name, const char *override_root,
                               uint32_t processor_mask, int want_firmware,
                               ft_tt_runtime_images *out_images)
{
    char path[FT_TT_PROGRAM_RESOLVE_PATH_BYTES];
    int rc = join_program_path(override_root, program_name, path, sizeof(path));

    if (rc != FT_TT_OK)
        return rc;
    ft_tt_log_error("tt program: '%s' %s OVERRIDDEN to %s (embedded blob not used)",
                    program_name, want_firmware ? "firmware" : "kernels", path);

    rc = want_firmware ? ft_tt_runtime_firmware_load(path, out_images)
                       : ft_tt_runtime_program_load(path, processor_mask, out_images);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("tt program: REFUSED — %s was named as an override and cannot be "
                        "read (%s). This does NOT fall back to the embedded program: a run "
                        "that silently used images the operator did not choose is the "
                        "defect this refusal replaces.", path, ft_tt_error_string(rc));
        return rc;
    }
    (void)snprintf(out_images->name, sizeof(out_images->name), "%s", program_name);
    out_images->program.name = out_images->name;
    return FT_TT_OK;
}

/* ================================================================
 * The embedded arm
 * ================================================================ */

/** Find the entry and prove its bytes before any of them are handed out. */
static int find_verified_entry(const char *program_name,
                               const ft_tt_program_blob_entry **out_entry)
{
    const ft_tt_program_blob_entry *entry = NULL;
    int rc = ft_tt_program_blob_find(program_name, &entry);

    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_program_blob_verify(entry);
    if (rc != FT_TT_OK) {
        ft_tt_log_error("tt program: REFUSED — embedded program '%s' failed its integrity "
                        "check. The binary itself is damaged; rebuilding it is the fix.",
                        program_name);
        return rc;
    }
    *out_entry = entry;
    return FT_TT_OK;
}

/** Point one slot of `out_images` at an embedded image. Nothing is copied and nothing is
 *  owned, which is why `owned_data` stays NULL. */
static void adopt_embedded_image(ft_tt_runtime_images *out_images, unsigned processor,
                                 const ft_tt_program_blob_image *image)
{
    out_images->elf_data[processor] = image->elf_data;
    out_images->elf_size[processor] = image->elf_size;
    out_images->owned_data[processor] = NULL;
    (void)snprintf(out_images->source_names[processor],
                   sizeof(out_images->source_names[processor]), "%s",
                   image->source_name ? image->source_name : "embedded");
}

static void finish_embedded(ft_tt_runtime_images *out_images,
                            const ft_tt_program_blob_entry *entry)
{
    (void)snprintf(out_images->name, sizeof(out_images->name), "%s", entry->name);
    out_images->source = FT_TT_PROGRAM_SOURCE_EMBEDDED;
    out_images->program.name = out_images->name;
    out_images->program.toolchain = entry->toolchain;
    out_images->program.compile_time_args = entry->build_recipe;
}

static int open_embedded_firmware(const char *program_name, ft_tt_runtime_images *out_images)
{
    const ft_tt_program_blob_entry *entry = NULL;
    unsigned processor;
    int rc = find_verified_entry(program_name, &entry);

    if (rc != FT_TT_OK)
        return rc;
    memset(out_images, 0, sizeof(*out_images));
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++)
        adopt_embedded_image(out_images, processor, &entry->firmware->images[processor]);
    finish_embedded(out_images, entry);
    return FT_TT_OK;
}

static int open_embedded_kernels(const char *program_name, uint32_t processor_mask,
                                 ft_tt_runtime_images *out_images)
{
    const ft_tt_program_blob_entry *entry = NULL;
    unsigned processor;
    unsigned count = 0;
    int rc = find_verified_entry(program_name, &entry);

    if (rc != FT_TT_OK)
        return rc;
    memset(out_images, 0, sizeof(*out_images));

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        const ft_tt_program_blob_image *image = NULL;

        if ((processor_mask & (1u << processor)) == 0u)
            continue;
        rc = ft_tt_program_blob_kernel_image(entry, processor, &image);
        if (rc != FT_TT_OK) {
            ft_tt_log_error("tt program: REFUSED — '%s' was asked to enable processor %u and "
                            "carries no kernel for it. A dropped image is a core the launch "
                            "message starts with nothing to run.", program_name, processor);
            memset(out_images, 0, sizeof(*out_images));
            return rc;
        }
        adopt_embedded_image(out_images, processor, image);
        out_images->images[count].source_name = out_images->source_names[processor];
        out_images->images[count].processor = processor;
        out_images->images[count].elf_data = image->elf_data;
        out_images->images[count].elf_size = image->elf_size;
        count++;
    }
    finish_embedded(out_images, entry);
    out_images->program.images = out_images->images;
    out_images->program.image_count = count;
    return FT_TT_OK;
}

/* ================================================================
 * Entry points
 * ================================================================ */

int ft_tt_program_open_firmware(const char *program_name, const char *override_root,
                                ft_tt_runtime_images *out_images)
{
    if (!program_name || !*program_name || !out_images)
        return FT_TT_EINVAL;
    if (override_root && *override_root)
        return open_from_directory(program_name, override_root, 0u, 1, out_images);
    return open_embedded_firmware(program_name, out_images);
}

int ft_tt_program_open_kernels(const char *program_name, const char *override_root,
                               uint32_t processor_mask, ft_tt_runtime_images *out_images)
{
    if (!program_name || !*program_name || !out_images || processor_mask == 0u ||
        processor_mask >= (1u << FT_TT_PROC_COUNT))
        return FT_TT_EINVAL;
    if (override_root && *override_root)
        return open_from_directory(program_name, override_root, processor_mask, 0, out_images);
    return open_embedded_kernels(program_name, processor_mask, out_images);
}

/* ================================================================
 * The recorded build recipe — what the ARMED image says about itself
 * ================================================================
 *
 * `ft_tt_program_blob.h` says of the `provenance` field that "NOTHING AT RUN TIME READS
 * THIS", and the reason it gave is still right about the thing it was about: a binary has no
 * path to a source tree, so a STALENESS check here could only ever pass. This is a different
 * question. It does not ask whether the image matches sources that are not present; it asks
 * what the image was BUILT WITH, which the record carries and which nothing else in the
 * binary can say.
 *
 * IT EXISTS BECAUSE ONE HOST ASSUMPTION HAS NO OTHER EVIDENCE. The matmul's DST section is a
 * compile-time property of the armed kernel — four tiles at `DstSync::SyncHalf`, eight at
 * SyncFull — and being wrong in the permissive direction is SILENT: a pass writes over the
 * half the packer is reading and every kernel retires clean. The interlock that guarded it
 * keyed on "an override directory is set", which was sound only while the embedded blob was
 * SyncHalf. A SyncFull blob REVERSES that argument, and there is no version of "which
 * directory did you name" that survives the reversal. The recorded `defines=` does, because
 * it moves with the image instead of with the operator.
 */

/** The `recipe` line of a `.prov` record, or NULL. Records are small and line-oriented. */
static const char *provenance_recipe_line(const char *record)
{
    const char *line = record;

    while (line && *line) {
        if (strncmp(line, "recipe ", 7) == 0)
            return line + 7;
        line = strchr(line, '\n');
        if (line)
            line++;
    }
    return NULL;
}

/**
 * Copy the `defines=` field out of a recipe line.
 *
 * `defines` is the LAST key `format_recipe()` writes and the only one whose value may
 * contain spaces, so it is read to end-of-line rather than as a whitespace token. An absent
 * key is not the same as an empty one: `format_recipe()` writes every key even when unset,
 * so a record with no `defines=` at all predates the field and cannot answer the question —
 * which is a REFUSAL here, not an empty string that would read as "not SyncFull".
 */
static int recipe_defines(const char *recipe, char *out, size_t out_len)
{
    const char *field = strstr(recipe, "defines=");
    const char *end;
    size_t length;

    if (!field)
        return FT_TT_ENOTFOUND;
    field += 8;
    end = strchr(field, '\n');
    length = end ? (size_t)(end - field) : strlen(field);
    if (length + 1u > out_len)
        return FT_TT_EINVAL;
    memcpy(out, field, length);
    out[length] = '\0';
    return FT_TT_OK;
}

int ft_tt_program_recorded_defines(const char *program_name, const char *override_root,
                                   unsigned processor, char *out, size_t out_len)
{
    const ft_tt_program_blob_entry *entry = NULL;
    const ft_tt_program_blob_image *image = NULL;
    const char *recipe;
    int rc;

    if (!program_name || !*program_name || !out || out_len == 0u ||
        processor >= FT_TT_PROC_COUNT)
        return FT_TT_EINVAL;
    out[0] = '\0';
    if (override_root && *override_root) {
        char path[FT_TT_PROGRAM_RESOLVE_PATH_BYTES];
        char record[4096];
        size_t read_bytes;
        FILE *file;

        const char *image_name = ft_tt_runtime_kernel_image_name(processor);

        if (!image_name)
            return FT_TT_EINVAL;
        if (snprintf(path, sizeof(path), "%s/%s/%s.prov", override_root, program_name,
                     image_name) >= (int)sizeof(path))
            return FT_TT_EINVAL;
        file = fopen(path, "rb");
        if (!file) {
            ft_tt_log_error("tt program: '%s' has no provenance record at %s — a directory "
                            "that does not record how it was built cannot answer what it was "
                            "built WITH.", program_name, path);
            return FT_TT_ENOTFOUND;
        }
        read_bytes = fread(record, 1u, sizeof(record) - 1u, file);
        fclose(file);
        record[read_bytes] = '\0';
        recipe = provenance_recipe_line(record);
        if (!recipe)
            return FT_TT_ENOTFOUND;
        return recipe_defines(recipe, out, out_len);
    }

    rc = ft_tt_program_blob_find(program_name, &entry);
    if (rc != FT_TT_OK)
        return rc;
    rc = ft_tt_program_blob_kernel_image(entry, processor, &image);
    if (rc != FT_TT_OK)
        return rc;
    if (!image->provenance)
        return FT_TT_ENOTFOUND;
    recipe = provenance_recipe_line(image->provenance);
    if (!recipe)
        return FT_TT_ENOTFOUND;
    return recipe_defines(recipe, out, out_len);
}

/**
 * Does `c` separate two `-D` tokens in a recorded `defines` value?
 *
 * The build records the list COMMA-joined (`-DA=1,-DB=1`; the rebuild scripts undo it with
 * `tr ',' ' '`), and a hand-written override record may use spaces, so both count. A
 * separator set that misses the comma reads every multi-define record as one long token and
 * answers "not present" — for the DST interlock, a SyncFull program refused as SyncHalf.
 */
static int defines_separator(char c)
{
    return c == ' ' || c == '\t' || c == ',';
}

int ft_tt_program_defines_contain(const char *defines, const char *wanted)
{
    const char *at;
    size_t length;

    if (!defines || !wanted || !*wanted)
        return 0;
    length = strlen(wanted);
    for (at = strstr(defines, wanted); at; at = strstr(at + 1, wanted)) {
        const char after = at[length];
        const char before = (at == defines) ? ' ' : at[-1];

        /* WHOLE TOKEN ONLY. `-DFT_TT_MATMUL_DST_SYNC=1` must not be found inside
         * `-DFT_TT_MATMUL_DST_SYNC=11`, and a substring match is exactly how a check like
         * this reads green on something it does not mean. */
        if (defines_separator(before) && (after == '\0' || defines_separator(after)))
            return 1;
    }
    return 0;
}
