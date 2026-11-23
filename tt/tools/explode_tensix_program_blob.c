/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file explode_tensix_program_blob.c
 * @brief Write the EMBEDDED program blob back out as program DIRECTORIES — the exact
 *   shape `emit_tensix_program_header.py` consumes — so a machine with no
 *   `tt/programs/` can still regenerate the blob after one kernel changes.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS
 * ---------------------------------------------------------------------------
 * `ft_tt_programs.h` is generated FROM directories of ELFs, and those directories live on
 * whichever machine last ran the builders. `tt/programs/` is not in the tree. So on any
 * other machine — including the second motherboard the card now lives in — changing one
 * kernel meant rebuilding all nineteen programs from scratch, or not changing it.
 *
 * The blob already holds everything a directory holds: five kernel ELFs, the five firmware
 * ELFs, each image's `.src` source name, the toolchain version and the build recipe. This
 * writes them back out. The round trip is then:
 *
 *   explode -> rebuild the ONE image that changed -> emit_tensix_program_header.py
 *
 * ---------------------------------------------------------------------------
 * THE CHECK THAT MAKES IT TRUSTWORTHY, AND IT IS NOT THIS PROGRAM'S OWN
 * ---------------------------------------------------------------------------
 * Explode, re-emit with NOTHING changed, and diff against the header you started from. A
 * faithful round trip is byte-identical. That is the discriminator, and it is worth having
 * because the failure this tool could produce — a program silently emitted from a
 * mis-assembled directory — is exactly the "stale artifact baked into a binary" the
 * emitter's own refusals exist to prevent, one level further back.
 *
 * `--verify` additionally runs `ft_tt_program_blob_verify()` on every entry before writing
 * anything, so a blob whose hex emission lost a byte is refused rather than exploded.
 *
 * Usage:
 *   explode_tensix_program_blob <output-directory> [--verify] [--recipes <file>]
 *
 * `--recipes` writes one `<name>\t<toolchain>\t<recipe>` line per program, because those
 * two strings are arguments to the emitter and are not recoverable from the ELFs.
 *
 * @status Written for the resident cooperative matmul round, where one kernel image
 *   changed on a machine with no program directories.
 * @issues None
 * @todo None
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "ft_tt_device.h"
#include "ft_tt_memmap.h"
#include "ft_tt_program_blob.h"

/** Indexed by FT_TT_PROC_*, and it must stay in the emitter's order — the emitter reads
 *  these file names back and infers the processor from the position. */
static const char *const kKernelFileNames[FT_TT_PROC_COUNT] = { "brisc.elf", "ncrisc.elf",
                                                                "trisc0.elf", "trisc1.elf",
                                                                "trisc2.elf" };
static const char *const kFirmwareFileNames[FT_TT_PROC_COUNT] = {
    "fw_brisc.elf", "fw_ncrisc.elf", "fw_trisc0.elf", "fw_trisc1.elf", "fw_trisc2.elf"
};

static int write_whole_file(const char *path, const void *data, size_t bytes)
{
    FILE *handle = fopen(path, "wb");
    size_t written;

    if (handle == NULL) {
        fprintf(stderr, "explode: could not create %s: %s\n", path, strerror(errno));
        return FT_TT_ERROR;
    }
    written = fwrite(data, 1u, bytes, handle);
    if (fclose(handle) != 0 || written != bytes) {
        fprintf(stderr, "explode: short write to %s (%zu of %zu bytes)\n", path, written,
                bytes);
        return FT_TT_ERROR;
    }
    return FT_TT_OK;
}

/** The `.src` sidecar the emitter reads to name an image's kernel source. Firmware has
 *  none — it is not built from a kernel source — and the emitter's own reader falls back
 *  to the file name for exactly that case, so writing one here would be a lie. */
static int write_sidecar(const char *directory, const char *file_name, const char *source_name)
{
    char path[1024];

    if (source_name == NULL || source_name[0] == '\0')
        return FT_TT_OK;
    /* An image whose recorded source is just its own file name had no sidecar when it was
     * embedded, and inventing one now would make a later staleness check compare an ELF
     * against a source that never built it. */
    if (strcmp(source_name, file_name) == 0)
        return FT_TT_OK;
    if (snprintf(path, sizeof(path), "%s/%s.src", directory, file_name) >= (int)sizeof(path))
        return FT_TT_ERROR;
    return write_whole_file(path, source_name, strlen(source_name));
}

/** The `<image>.prov` content record beside the ELF.
 *
 *  WITHOUT THIS, AN EXPLODE LAUNDERS THE ONE CHECK THAT SURVIVES IT. The `.src` sidecar's
 *  mtime is reset by definition here — every file is written fresh — which is how 21
 *  directories reported `current` with two of them provably stale. The provenance record
 *  is content, so writing it back out unchanged carries the truth across the round trip
 *  rather than erasing it. Firmware carries NULL and gets no file. */
static int write_provenance(const char *directory, const char *file_name,
                            const char *provenance)
{
    char path[1024];

    if (provenance == NULL || provenance[0] == '\0')
        return FT_TT_OK;
    if (snprintf(path, sizeof(path), "%s/%s.prov", directory, file_name) >= (int)sizeof(path))
        return FT_TT_ERROR;
    return write_whole_file(path, provenance, strlen(provenance));
}

static int write_image(const char *directory, const char *file_name,
                       const ft_tt_program_blob_image *image, bool with_sidecar)
{
    char path[1024];
    int rc;

    if (image == NULL || image->elf_data == NULL || image->elf_size == 0u)
        return FT_TT_OK;
    if (snprintf(path, sizeof(path), "%s/%s", directory, file_name) >= (int)sizeof(path))
        return FT_TT_ERROR;
    rc = write_whole_file(path, image->elf_data, image->elf_size);
    if (rc != FT_TT_OK || !with_sidecar)
        return rc;
    rc = write_sidecar(directory, file_name, image->source_name);
    if (rc != FT_TT_OK)
        return rc;
    return write_provenance(directory, file_name, image->provenance);
}

static int explode_one(const char *root, const ft_tt_program_blob_entry *entry)
{
    char directory[1024];
    unsigned index;
    int rc;

    if (snprintf(directory, sizeof(directory), "%s/%s", root, entry->name) >=
        (int)sizeof(directory))
        return FT_TT_ERROR;
    if (mkdir(directory, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "explode: could not create %s: %s\n", directory, strerror(errno));
        return FT_TT_ERROR;
    }

    for (index = 0u; index < entry->image_count; index++) {
        const ft_tt_program_blob_image *image = &entry->images[index];

        if (image->processor >= FT_TT_PROC_COUNT) {
            fprintf(stderr, "explode: %s image %u names processor %u\n", entry->name, index,
                    image->processor);
            return FT_TT_ERROR;
        }
        rc = write_image(directory, kKernelFileNames[image->processor], image, true);
        if (rc != FT_TT_OK)
            return rc;
    }
    if (entry->firmware == NULL) {
        /* Firmware is all five images or none: a program whose set is incomplete boots a
         * core into whatever L1 held, and the emitter refuses such a directory. Refuse to
         * WRITE one rather than leave it for the emitter to find. */
        fprintf(stderr, "explode: %s carries no firmware set\n", entry->name);
        return FT_TT_ERROR;
    }
    for (index = 0u; index < FT_TT_PROC_COUNT; index++) {
        rc = write_image(directory, kFirmwareFileNames[index], &entry->firmware->images[index],
                         false);
        if (rc != FT_TT_OK)
            return rc;
    }
    printf("  %-20s %u kernel image(s), firmware set %s, toolchain %s\n", entry->name,
           entry->image_count, entry->firmware->set_id, entry->toolchain);
    return FT_TT_OK;
}

int main(int argc, char **argv)
{
    const char *root = NULL;
    const char *recipes_path = NULL;
    bool verify = false;
    unsigned count;
    unsigned index;
    int argument;

    for (argument = 1; argument < argc; argument++) {
        if (strcmp(argv[argument], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[argument], "--recipes") == 0 && argument + 1 < argc) {
            recipes_path = argv[++argument];
        } else if (argv[argument][0] == '-') {
            fprintf(stderr, "usage: %s <output-directory> [--verify] [--recipes <file>]\n",
                    argv[0]);
            return 2;
        } else if (root == NULL) {
            root = argv[argument];
        } else {
            fprintf(stderr, "usage: %s <output-directory> [--verify] [--recipes <file>]\n",
                    argv[0]);
            return 2;
        }
    }
    if (root == NULL) {
        fprintf(stderr, "usage: %s <output-directory> [--verify] [--recipes <file>]\n",
                argv[0]);
        return 2;
    }

    count = ft_tt_program_blob_count();
    /* ZERO IS A MEANINGFUL ANSWER and is not "nothing to do": it means this binary was
     * built against a header that was never regenerated, and exploding it would produce an
     * empty tree that a later emit would happily turn into an empty blob. */
    if (count == 0u) {
        fprintf(stderr, "explode: this binary's program blob is EMPTY (%s) — there is "
                        "nothing to write, and an empty tree is how an empty blob gets "
                        "regenerated by accident\n",
                ft_tt_program_blob_generated_header());
        return 1;
    }
    if (mkdir(root, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "explode: could not create %s: %s\n", root, strerror(errno));
        return 1;
    }

    printf("exploding %u programs from %s into %s\n", count,
           ft_tt_program_blob_generated_header(), root);
    if (verify)
        for (index = 0u; index < count; index++) {
            const ft_tt_program_blob_entry *entry = ft_tt_program_blob_entry_at(index);

            if (entry == NULL || ft_tt_program_blob_verify(entry) != FT_TT_OK) {
                fprintf(stderr, "explode: entry %u failed its integrity check; NOTHING was "
                                "written\n", index);
                return 1;
            }
        }

    for (index = 0u; index < count; index++) {
        const ft_tt_program_blob_entry *entry = ft_tt_program_blob_entry_at(index);

        if (entry == NULL || explode_one(root, entry) != FT_TT_OK)
            return 1;
    }

    if (recipes_path != NULL) {
        FILE *handle = fopen(recipes_path, "w");

        if (handle == NULL) {
            fprintf(stderr, "explode: could not create %s: %s\n", recipes_path,
                    strerror(errno));
            return 1;
        }
        for (index = 0u; index < count; index++) {
            const ft_tt_program_blob_entry *entry = ft_tt_program_blob_entry_at(index);

            fprintf(handle, "%s\t%s\t%s\n", entry->name, entry->toolchain,
                    entry->build_recipe);
        }
        if (fclose(handle) != 0) {
            fprintf(stderr, "explode: could not finish %s\n", recipes_path);
            return 1;
        }
        printf("recipes written to %s\n", recipes_path);
    }
    return 0;
}
