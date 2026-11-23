/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_kernel_blob.h
 * @brief The format of the embedded device-kernel blobs, and the accessors over them.
 *
 *   This is the TT side of Foundry's existing CUDA pattern: the `.cu` sources under
 *   `cuda_kernels/` are compiled offline by nvcc into `fc_cuda_ptx.h`, an embedded
 *   blob the runtime hands to `cuModuleLoadData`. Here, the `.c` sources under
 *   `foundry/tt/kernels/` are compiled offline by SFPI into TRISC and dataflow ELFs,
 *   and `tools/emit_tensix_kernel_header.py` turns
 *   those ELFs into `ft_tt_kernels.h` — a header of `static const unsigned char`
 *   arrays plus the descriptor tables declared below. Nothing at run time reads a
 *   file, and no compiler is needed on the card host.
 *
 *   THE BLOB IS THE WHOLE ELF, not an extracted `.text`. The loader needs the entry
 *   point and both load addresses (`docs/tt_launch_notes.md` §2), and an ELF already
 *   carries them; stripping it down would mean re-encoding the same facts in a weaker
 *   format. Parsing costs one pass over ~2 KB.
 *
 * Generated-header contract — `ft_tt_kernels.h` defines, for each program:
 *
 *   static const unsigned char ft_tt_kernel_<program>_<proc>_elf[] = { ... };
 *   static const ft_tt_kernel_image ft_tt_kernel_<program>_images[] = { ... };
 *   static const ft_tt_kernel_program ft_tt_kernel_<program> = { ... };
 *
 *   plus `ft_tt_kernel_programs[]` / `ft_tt_kernel_program_count` over all of them.
 *   Everything is `static const`, so a translation unit that includes the header but
 *   uses one program does not carry the rest into the binary.
 *
 * @status Format defined and compile-checked. `ft_tt_kernels.h` is committed like
 *   `fc_cuda_ptx.h`, because SFPI exists only on the card host. Last regenerated 2026-10-27
 *   by `tools/regen_tensix_kernel_header.sh` from the current Foundry sources (SFPI 7.68.0). `tools/dump_tensix_kernel_header.c` round-trips it through
 *   `ft_tt_elf_parse` and its output diffs clean against readelf.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_KERNEL_BLOB_H
#define FT_TT_KERNEL_BLOB_H

#include <stddef.h>

/** One processor's ELF within a program. `processor` is an FT_TT_PROC_* index from
 *  ft_tt_memmap.h, which is also the index into the launch message's
 *  `kernel_text_offset[]` and the bit position in its `enables` mask. */
typedef struct ft_tt_kernel_image {
    const char *source_name;       /**< the .c this was built from, for logs */
    unsigned processor;            /**< FT_TT_PROC_DM0 .. FT_TT_PROC_MATH2 */
    const unsigned char *elf_data; /**< the complete ELF file */
    size_t elf_size;
} ft_tt_kernel_image;

/** One op: the images that make up its reader/compute/writer triad. */
typedef struct ft_tt_kernel_program {
    const char *name;
    const char *toolchain;               /**< SFPI version the blobs were built with */
    const char *compile_time_args;       /**< the CTARGS the blobs were built with */
    const ft_tt_kernel_image *images;
    unsigned image_count;
} ft_tt_kernel_program;

/**
 * Find a program's image for one processor.
 *
 * @param program    the program to search
 * @param processor  an FT_TT_PROC_* index
 * @return           the image, or NULL when this program leaves that processor idle
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
const ft_tt_kernel_image *ft_tt_kernel_image_for_processor(const ft_tt_kernel_program *program,
                                                           unsigned processor);

/**
 * The `enables` bitmask a program implies: bit `processor` set for every image it
 * carries. This is exactly what the launch message's `enables` field wants
 * (consumed by BRISC firmware when it releases the TRISCs).
 *
 * @return the mask, or 0 for a NULL or empty program
 */
unsigned ft_tt_kernel_program_enables(const ft_tt_kernel_program *program);

#endif /* FT_TT_KERNEL_BLOB_H */
