/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file verify_tensix_program_blob.c
 * @brief Compare the EMBEDDED program blob against the program directories on disk, byte
 *   for byte, and say which of the two is wrong when they differ.
 *
 *   The generator produced `ft_tt_programs.h` from those directories. This tool reads the
 *   directories again, with plain `fread`, and `memcmp`s them against the arrays the
 *   compiler put in this binary. Nothing here shares an implementation with the emitter, so
 *   "the blob is the directories" is a comparison and not a restatement of how the blob was
 *   built.
 *
 *   THIS IS THE TOOL THAT MAKES A REAL REGENERATION CHECKABLE. The blob can only be
 *   regenerated on the card host, where SFPI and the real program directories live. After that
 *   regeneration, this is what proves the committed header carries those exact bytes:
 *
 *       make -f tools/Makefile verify-programs
 *       ./tools/build/verify_tensix_program_blob ../programs
 *
 *   It also prints what the blob CONTAINS, because a comparison of an empty blob against a
 *   directory tree is trivially green and would otherwise read as a pass.
 *
 * @status Written on the Mac; exercised through tests/Makefile.programblob's fixtures. Has
 *   never been run against real SFPI program directories — that needs a bank session.
 * @issues None
 * @todo None
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_memmap.h"
#include "ft_tt_program_blob.h"

#define MAX_IMAGE_BYTES (4u * 1024u * 1024u)

static const char *const k_kernel_file_names[FT_TT_PROC_COUNT] = {
    "brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf"
};
static const char *const k_firmware_file_names[FT_TT_PROC_COUNT] = {
    "fw_brisc.elf", "fw_ncrisc.elf", "fw_trisc0.elf", "fw_trisc1.elf", "fw_trisc2.elf"
};

/** Read a whole file. NULL on any failure, which is always a finding and never a skip. */
static unsigned char *read_whole_file(const char *path, size_t *out_size)
{
    FILE *file = fopen(path, "rb");
    unsigned char *buffer;
    long size;

    *out_size = 0;
    if (!file)
        return NULL;
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) <= 0 ||
        (size_t)size > MAX_IMAGE_BYTES || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    buffer = malloc((size_t)size);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *out_size = (size_t)size;
    return buffer;
}

/** @return 0 when the file equals the embedded image, 1 otherwise (and it says why). */
static int compare_one(const char *root, const char *program, const char *file_name,
                       const ft_tt_program_blob_image *image)
{
    char path[1024];
    unsigned char *disk;
    size_t size = 0;

    if (snprintf(path, sizeof(path), "%s/%s/%s", root, program, file_name) < 0)
        return 1;
    disk = read_whole_file(path, &size);
    if (!disk) {
        printf("    %-16s MISSING or unreadable at %s\n", file_name, path);
        return 1;
    }
    if (size != image->elf_size) {
        printf("    %-16s DIFFERS: embedded %lu B, on disk %lu B\n", file_name,
               (unsigned long)image->elf_size, (unsigned long)size);
        free(disk);
        return 1;
    }
    if (memcmp(disk, image->elf_data, size) != 0) {
        printf("    %-16s DIFFERS: %lu bytes, embedded digest 0x%016llx, file digest "
               "0x%016llx\n", file_name, (unsigned long)size,
               (unsigned long long)image->elf_digest,
               (unsigned long long)ft_tt_program_blob_digest(disk, size));
        free(disk);
        return 1;
    }
    printf("    %-16s matches (%lu B)\n", file_name, (unsigned long)size);
    free(disk);
    return 0;
}

static int verify_one_program(const char *root, const ft_tt_program_blob_entry *entry)
{
    unsigned index;
    int differences = 0;

    printf("  %s  (toolchain %s, %u kernel image(s), firmware set %s)\n", entry->name,
           entry->toolchain, entry->image_count, entry->firmware->set_id);
    if (ft_tt_program_blob_verify(entry) != FT_TT_OK) {
        printf("    the EMBEDDED copy fails its own integrity check — this binary is "
               "damaged, independently of what is on disk\n");
        differences++;
    }
    for (index = 0; index < entry->image_count; index++) {
        const ft_tt_program_blob_image *image = &entry->images[index];

        differences += compare_one(root, entry->name, k_kernel_file_names[image->processor],
                                   image);
    }
    for (index = 0; index < FT_TT_PROC_COUNT; index++)
        differences += compare_one(root, entry->name, k_firmware_file_names[index],
                                   &entry->firmware->images[index]);
    return differences;
}

int main(int argc, char **argv)
{
    unsigned count = ft_tt_program_blob_count();
    unsigned index;
    int differences = 0;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <programs-root>\n\n"
                        "Compares every program embedded in this binary against\n"
                        "<programs-root>/<name>/ on disk, byte for byte.\n", argv[0]);
        return 2;
    }

    /* Say what is in the blob BEFORE comparing anything. An empty blob compares clean
     * against every directory tree in existence, and "no differences" would then be a
     * statement about nothing. */
    printf("blob: %s, %u program(s)\n", ft_tt_program_blob_generated_header(), count);
    if (count == 0u) {
        printf("REFUSED: this binary carries no embedded programs, so there is nothing to "
               "compare. Regenerate with tools/regen_tensix_program_header.sh on the bank.\n");
        return 1;
    }
    printf("against: %s\n\n", argv[1]);

    for (index = 0; index < count; index++)
        differences += verify_one_program(argv[1], ft_tt_program_blob_entry_at(index));

    printf("\n%d difference(s)\n", differences);
    return differences == 0 ? 0 : 1;
}
