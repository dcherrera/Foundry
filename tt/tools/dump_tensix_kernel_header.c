/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file dump_tensix_kernel_header.c
 * @brief Round-trip check for the generated ft_tt_kernels.h: parse every embedded image
 *   with ft_tt_elf_parse() and print what the parser found, in a format that can be
 *   diffed byte-for-byte against readelf's view of the same ELFs.
 *
 *   The point is that the comparison is against the TOOLCHAIN, not against ourselves.
 *   tools/readelf_tensix_kernel_dump.sh emits the identical format straight out of
 *   riscv-tt-elf-readelf; `diff` between the two is the whole test. A parser bug and a
 *   generator bug that happened to agree would still show up, because readelf agrees
 *   with neither.
 *
 *   Build (from foundry/tt/tools):
 *     cc -std=c99 -Wall -Wextra -Werror -pedantic -I.. -o /tmp/dump_kernels \
 *        dump_tensix_kernel_header.c ../ft_tt_elf.c ../ft_tt_kernel_blob.c \
 *        ../ft_tt_device.c ../ft_tt_kmd.c ../ft_tt_tlb.c ../ft_tt_arch.c \
 *        ../ft_tt_arch_blackhole.c ../ft_tt_coord.c ../ft_tt_arc_telemetry.c
 *
 *   Lines beginning with '#' are metadata that only the header knows (program name,
 *   toolchain, source .cpp); strip them before diffing.
 *
 * @status Run on the card host 2026-10-27 against the regenerated ft_tt_kernels.h: all 14
 *   images' entry and PT_LOAD records match riscv-tt-elf-readelf exactly.
 * @issues None
 * @todo None
 */

#include <stdio.h>

#include "ft_tt_device.h"
#include "ft_tt_elf.h"
#include "ft_tt_kernel_blob.h"
#include "ft_tt_kernels.h"
#include "ft_tt_memmap.h"

static const char *processor_name(unsigned processor)
{
    switch (processor) {
    case FT_TT_PROC_DM0:   return "DM0";
    case FT_TT_PROC_DM1:   return "DM1";
    case FT_TT_PROC_MATH0: return "MATH0";
    case FT_TT_PROC_MATH1: return "MATH1";
    case FT_TT_PROC_MATH2: return "MATH2";
    default:               return "UNKNOWN";
    }
}

/** Parse one embedded image and print its toolchain-comparable description.
 *  @return 0 on success, 1 if the parser rejected the blob. */
static int dump_image(const ft_tt_kernel_image *image)
{
    const char *name = processor_name(image->processor);
    ft_tt_elf_image parsed = {0};
    uint32_t low = 0;
    uint32_t high = 0;
    unsigned index;

    printf("# image %s source %s elf_bytes %zu\n", name, image->source_name, image->elf_size);

    if (ft_tt_elf_parse(image->elf_data, image->elf_size, &parsed) != FT_TT_OK) {
        printf("%s PARSE_FAILED\n", name);
        return 1;
    }
    if (ft_tt_elf_load_extent(&parsed, &low, &high) != FT_TT_OK) {
        printf("%s EXTENT_FAILED\n", name);
        return 1;
    }

    printf("%s entry=0x%08x\n", name, parsed.entry_address);
    for (index = 0; index < parsed.segment_count; index++) {
        const ft_tt_elf_segment *segment = &parsed.segments[index];
        printf("%s seg%u exec=0x%08x load=0x%08x file=%u mem=%u\n", name, index,
               segment->exec_address, segment->load_address, segment->file_bytes,
               segment->memory_bytes);
    }
    printf("# %s loaded_bytes %u l1_extent 0x%08x..0x%08x\n", name, parsed.loaded_bytes, low,
           high);
    return 0;
}

int main(void)
{
    unsigned failures = 0;
    unsigned program_index;

    for (program_index = 0; program_index < FT_TT_KERNEL_PROGRAM_COUNT; program_index++) {
        const ft_tt_kernel_program *program = ft_tt_kernel_programs[program_index];
        unsigned image_index;

        printf("# program %s toolchain %s ctargs %s images %u enables 0x%x\n", program->name,
               program->toolchain, program->compile_time_args, program->image_count,
               ft_tt_kernel_program_enables(program));

        for (image_index = 0; image_index < program->image_count; image_index++)
            failures += (unsigned)dump_image(&program->images[image_index]);
    }

    if (failures) {
        fprintf(stderr, "dump_tensix_kernel_header: %u image(s) failed to parse\n", failures);
        return 1;
    }
    return 0;
}
