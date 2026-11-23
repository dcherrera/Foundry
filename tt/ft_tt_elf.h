/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_elf.h
 * @brief Parse an in-memory ELF32 little-endian RISC-V executable (a Tensix kernel or
 *   firmware image) into the list of segments a loader writes into L1.
 *
 * The parser copies nothing: segment file_data points into the caller's blob, which must
 * outlive the image. Anything the loader could not place safely is refused with
 * FT_TT_EINVAL (ft_tt_device.h) and a reason logged through ft_tt_log_error():
 *
 *   - not ELF / not ELFCLASS32 / not little-endian / not ET_EXEC / not EM_RISCV;
 *   - a missing, misaligned or out-of-bounds program-header table;
 *   - a PT_LOAD whose file offset, virtual or physical address is not 4-byte aligned,
 *     whose file size is not a multiple of 4, or whose file bytes leave the blob;
 *   - more than FT_TT_ELF_MAX_SEGMENTS PT_LOADs;
 *   - an entry point that is not the start of the first PT_LOAD (the text);
 *   - a `.segments` section (written by fw/ft_tensix_*.ld) that disagrees with the
 *     PT_LOADs: a different execution address, or a memory size over the region limit.
 *
 * @status Clean-room rewrite. Oracle: tests/test_launch.c (`make check-launch`).
 * @issues None
 * @todo None
 */

#ifndef FT_TT_ELF_H
#define FT_TT_ELF_H

#include <stddef.h>
#include <stdint.h>

/** Most PT_LOAD segments an image may carry (text, data, and headroom). */
#define FT_TT_ELF_MAX_SEGMENTS 4u

/** One PT_LOAD segment. */
typedef struct ft_tt_elf_segment {
    uint32_t exec_address;    /**< p_vaddr: where the core sees it when running */
    uint32_t load_address;    /**< p_paddr: where its initial image is staged */
    const uint8_t *file_data; /**< p_filesz bytes inside the caller's blob */
    uint32_t file_bytes;      /**< p_filesz */
    uint32_t memory_bytes;    /**< p_memsz (file bytes plus zero-filled tail) */
} ft_tt_elf_segment;

/** A parsed image: PT_LOADs in program-header order, text first. */
typedef struct ft_tt_elf_image {
    uint32_t entry_address;                              /**< e_entry == segments[0].exec_address */
    ft_tt_elf_segment segments[FT_TT_ELF_MAX_SEGMENTS];
    unsigned segment_count;                              /**< 1..FT_TT_ELF_MAX_SEGMENTS */
    uint32_t loaded_bytes;                               /**< sum of file_bytes over segments */
} ft_tt_elf_image;

/**
 * Parse and validate `blob`.
 *
 * @param blob       ELF file contents.
 * @param blob_size  Bytes in `blob`.
 * @param out_image  Filled on success; unspecified on failure.
 * @return FT_TT_OK, or FT_TT_EINVAL (reason logged).
 */
int ft_tt_elf_parse(const void *blob, size_t blob_size, ft_tt_elf_image *out_image);

/**
 * The text segment (the first PT_LOAD), or NULL for a NULL or empty image.
 */
const ft_tt_elf_segment *ft_tt_elf_text_segment(const ft_tt_elf_image *image);

/**
 * The span of load addresses a load writes: `*out_low` is the lowest load address of a
 * segment with file bytes, `*out_high` the highest load_address + file_bytes. With no
 * file bytes at all, both are the text segment's load address.
 *
 * @return FT_TT_OK, or FT_TT_EINVAL on NULL arguments or an empty image.
 */
int ft_tt_elf_load_extent(const ft_tt_elf_image *image, uint32_t *out_low, uint32_t *out_high);

#endif
