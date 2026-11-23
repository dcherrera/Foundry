/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_program_directory.c
 * @brief Implementation of the on-disk program loader for the Phase 2 follow-up tests.
 *
 * @status Runs on the p150a test card.
 * @issues None
 * @todo None
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "ft_tt_program_directory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_device.h"

/** Kernel ELF names build_tensix_program.sh writes, indexed by FT_TT_PROC_*. */
static const char *const k_kernel_file_names[FT_TT_PROC_COUNT] = {
    "brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf"
};

/** Read a whole file into a fresh buffer. */
static int read_whole_file(const char *path, unsigned char **out_data, size_t *out_size)
{
    FILE *file = fopen(path, "rb");
    unsigned char *buffer = NULL;
    long size;

    if (!file) {
        ft_tt_log_error("program dir: cannot open %s", path);
        return FT_TT_EIO;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return FT_TT_EIO;
    }
    buffer = malloc((size_t)size);
    if (!buffer) {
        fclose(file);
        return FT_TT_ENOMEM;
    }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        ft_tt_log_error("program dir: short read on %s", path);
        free(buffer);
        fclose(file);
        return FT_TT_EIO;
    }
    fclose(file);
    *out_data = buffer;
    *out_size = (size_t)size;
    return FT_TT_OK;
}

/** Copy the `.src` sidecar's first line into `out`, or the ELF name if there is none. */
static void read_source_name(const char *directory, unsigned processor, char *out, size_t out_bytes)
{
    char path[512];
    FILE *file;
    size_t length;

    snprintf(out, out_bytes, "%s", k_kernel_file_names[processor]);
    if (snprintf(path, sizeof(path), "%s/%s.src", directory,
                 k_kernel_file_names[processor]) >= (int)sizeof(path))
        return;
    file = fopen(path, "r");
    if (!file)
        return;
    if (fgets(out, (int)out_bytes, file)) {
        length = strlen(out);
        while (length > 0u && (out[length - 1u] == '\n' || out[length - 1u] == '\r'))
            out[--length] = '\0';
    }
    fclose(file);
}

int ft_tt_program_directory_load(const char *directory, uint32_t processor_mask,
                                 ft_tt_program_directory *out_loaded)
{
    unsigned processor;
    unsigned image_count = 0;
    const char *last_separator;

    if (!directory || !out_loaded || processor_mask == 0u ||
        processor_mask >= (1u << FT_TT_PROC_COUNT))
        return FT_TT_EINVAL;
    memset(out_loaded, 0, sizeof(*out_loaded));

    last_separator = strrchr(directory, '/');
    snprintf(out_loaded->name, sizeof(out_loaded->name), "%s",
             last_separator ? last_separator + 1 : directory);

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        char path[512];
        int rc;

        if ((processor_mask & (1u << processor)) == 0u)
            continue;
        if (snprintf(path, sizeof(path), "%s/%s", directory,
                     k_kernel_file_names[processor]) >= (int)sizeof(path)) {
            ft_tt_program_directory_release(out_loaded);
            return FT_TT_EINVAL;
        }
        rc = read_whole_file(path, &out_loaded->elf_data[processor],
                             &out_loaded->images[image_count].elf_size);
        if (rc != FT_TT_OK) {
            ft_tt_program_directory_release(out_loaded);
            return rc;
        }
        read_source_name(directory, processor, out_loaded->source_names[processor],
                         sizeof(out_loaded->source_names[processor]));

        out_loaded->images[image_count].processor = processor;
        out_loaded->images[image_count].elf_data = out_loaded->elf_data[processor];
        out_loaded->images[image_count].source_name = out_loaded->source_names[processor];
        image_count++;
    }

    out_loaded->program.name = out_loaded->name;
    out_loaded->program.toolchain = "on-disk build";
    out_loaded->program.compile_time_args = "see the build log beside the directory";
    out_loaded->program.images = out_loaded->images;
    out_loaded->program.image_count = image_count;
    return FT_TT_OK;
}

void ft_tt_program_directory_release(ft_tt_program_directory *loaded)
{
    unsigned processor;

    if (!loaded)
        return;
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        free(loaded->elf_data[processor]);
        loaded->elf_data[processor] = NULL;
    }
    memset(&loaded->program, 0, sizeof(loaded->program));
}
