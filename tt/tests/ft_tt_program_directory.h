/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_program_directory.h
 * @brief Build an ft_tt_kernel_program out of the ELFs build_tensix_program.sh leaves in
 *   a directory, instead of out of the committed ft_tt_kernels.h.
 *
 * The embedded header is the shipping path and stays that way. It cannot serve the
 * Phase 2 follow-ups, for one reason: a kernel's text is linked at its own firmware's
 * `__fw_export_text_end`, so a kernel and the firmware it will run under are ONE build
 * output. Any build that changes firmware size — WATCHER=1 is exactly that — invalidates
 * every kernel in the committed header. Reading both halves out of the same directory
 * keeps them together and keeps experimental toolchain variants out of the tree.
 *
 * Pair with ft_tt_firmware_set_load(directory, ...) from ft_tt_hardware_session.h, and
 * pass BOTH the same directory. Mixing directories is the failure this file exists to
 * prevent.
 *
 * @status Used by test_phase2_cb_formats.c and
 *   test_phase2_dram.c. TEST scaffolding, not backend API.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_PROGRAM_DIRECTORY_H
#define FT_TT_PROGRAM_DIRECTORY_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_kernel_blob.h"
#include "ft_tt_memmap.h"

/** Longest program or source name this holds; the strings are only for logs. */
#define FT_TT_PROGRAM_DIRECTORY_NAME_MAX 96u

/** A program read off disk. `program` is what the launch path consumes; everything else
 *  is the storage it points into, so the struct must outlive every use of it. */
typedef struct ft_tt_program_directory {
    ft_tt_kernel_program program;
    ft_tt_kernel_image images[FT_TT_PROC_COUNT];
    unsigned char *elf_data[FT_TT_PROC_COUNT];
    char source_names[FT_TT_PROC_COUNT][FT_TT_PROGRAM_DIRECTORY_NAME_MAX];
    char name[FT_TT_PROGRAM_DIRECTORY_NAME_MAX];
} ft_tt_program_directory;

/**
 * Read one ELF per processor named in `processor_mask` from `directory`.
 *
 * File names are the ones build_tensix_program.sh writes: brisc.elf, ncrisc.elf,
 * trisc0.elf, trisc1.elf, trisc2.elf. A `.src` sidecar, if present, supplies the source
 * name for logs; its absence is not an error.
 *
 * @param directory       a build_tensix_program.sh output directory
 * @param processor_mask  bit FT_TT_PROC_* set for each image to load; the resulting
 *                        program's enables mask is exactly this
 * @param out_loaded      zeroed and filled on success
 * @return                FT_TT_OK; FT_TT_EINVAL on a bad argument or an empty mask;
 *                        FT_TT_EIO if a named ELF is missing or unreadable;
 *                        FT_TT_ENOMEM
 *
 * Ownership: the caller releases with ft_tt_program_directory_release().
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_program_directory_load(const char *directory, uint32_t processor_mask,
                                 ft_tt_program_directory *out_loaded);

/** Free every image. Safe on NULL and on a partially loaded program. */
void ft_tt_program_directory_release(ft_tt_program_directory *loaded);

#endif /* FT_TT_PROGRAM_DIRECTORY_H */
