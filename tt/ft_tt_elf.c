/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_elf.c
 * @brief Implementation of the Tensix ELF image parser (ft_tt_elf.h).
 *
 * Field offsets are those of the generic ELF32 format (System V ABI, ELF header /
 * program header / section header). Every multi-byte field is read byte by byte as
 * little-endian, so the blob needs no alignment and the host's byte order is irrelevant.
 * Every range check is done in 64-bit arithmetic so a hostile offset cannot wrap.
 *
 * @status Clean-room rewrite. Green against tests/test_launch.c.
 * @issues None
 * @todo None
 */

#include "ft_tt_elf.h"

#include <string.h>

#include "ft_tt_device.h"

/* ELF32 constants (generic ELF specification). */
#define ELF_HEADER_BYTES         52u
#define ELF_PROGRAM_HEADER_BYTES 32u
#define ELF_SECTION_HEADER_BYTES 40u
#define ELF_CLASS_32             1u
#define ELF_DATA_LITTLE_ENDIAN   1u
#define ELF_TYPE_EXECUTABLE      2u
#define ELF_MACHINE_RISCV        243u
#define ELF_PROGRAM_LOAD         1u
#define ELF_SECTION_NO_BITS      8u

/** Every Tensix load is done in 32-bit words. */
#define WORD_BYTES 4u

/** One `.segments` record from fw/ft_tensix_*.ld: three 32-bit words per segment. */
#define SEGMENT_RECORD_BYTES 12u

/** ELF header fields this parser uses. */
typedef struct elf_header_fields {
    uint32_t entry;
    uint32_t program_header_offset;
    uint32_t section_header_offset;
    uint16_t program_header_size;
    uint16_t program_header_count;
    uint16_t section_header_size;
    uint16_t section_header_count;
    uint16_t section_name_table_index;
} elf_header_fields;

/** Where a section's bytes are, and how many. */
typedef struct elf_section_span {
    uint32_t offset;
    uint32_t size;
} elf_section_span;

static uint16_t read_u16(const uint8_t *bytes, size_t offset)
{
    return (uint16_t)(bytes[offset] | (bytes[offset + 1u] << 8));
}

static uint32_t read_u32(const uint8_t *bytes, size_t offset)
{
    return (uint32_t)bytes[offset] | ((uint32_t)bytes[offset + 1u] << 8) | ((uint32_t)bytes[offset + 2u] << 16) |
           ((uint32_t)bytes[offset + 3u] << 24);
}

/** Whether [offset, offset + length) lies inside a blob of `blob_size` bytes. */
static int span_is_inside(uint64_t offset, uint64_t length, size_t blob_size)
{
    return offset <= (uint64_t)blob_size && length <= (uint64_t)blob_size - offset;
}

/** Identification, type and machine, then the fields the rest of the parse needs. */
static int read_elf_header(const uint8_t *bytes, size_t blob_size, elf_header_fields *header)
{
    if (blob_size < ELF_HEADER_BYTES) {
        ft_tt_log_error("elf: blob is %zu bytes, shorter than an ELF32 header", blob_size);
        return FT_TT_EINVAL;
    }
    if (memcmp(bytes, "\x7f" "ELF", 4u) != 0) {
        ft_tt_log_error("elf: bad magic");
        return FT_TT_EINVAL;
    }
    if (bytes[4] != ELF_CLASS_32) {
        ft_tt_log_error("elf: class %u is not ELFCLASS32", bytes[4]);
        return FT_TT_EINVAL;
    }
    if (bytes[5] != ELF_DATA_LITTLE_ENDIAN) {
        ft_tt_log_error("elf: data encoding %u is not little-endian", bytes[5]);
        return FT_TT_EINVAL;
    }
    if (read_u16(bytes, 16u) != ELF_TYPE_EXECUTABLE) {
        ft_tt_log_error("elf: type %u is not ET_EXEC", read_u16(bytes, 16u));
        return FT_TT_EINVAL;
    }
    if (read_u16(bytes, 18u) != ELF_MACHINE_RISCV) {
        ft_tt_log_error("elf: machine %u is not EM_RISCV", read_u16(bytes, 18u));
        return FT_TT_EINVAL;
    }
    header->entry = read_u32(bytes, 24u);
    header->program_header_offset = read_u32(bytes, 28u);
    header->section_header_offset = read_u32(bytes, 32u);
    header->program_header_size = read_u16(bytes, 42u);
    header->program_header_count = read_u16(bytes, 44u);
    header->section_header_size = read_u16(bytes, 46u);
    header->section_header_count = read_u16(bytes, 48u);
    header->section_name_table_index = read_u16(bytes, 50u);
    return FT_TT_OK;
}

static int program_header_table_is_valid(const elf_header_fields *header, size_t blob_size)
{
    if (header->program_header_offset == 0u || header->program_header_count == 0u) {
        ft_tt_log_error("elf: no program headers");
        return 0;
    }
    if (header->program_header_size != ELF_PROGRAM_HEADER_BYTES) {
        ft_tt_log_error("elf: program header entry size %u, expected %u", header->program_header_size,
                        ELF_PROGRAM_HEADER_BYTES);
        return 0;
    }
    if (header->program_header_offset % WORD_BYTES != 0u) {
        ft_tt_log_error("elf: program header table at 0x%x is not word aligned", header->program_header_offset);
        return 0;
    }
    if (!span_is_inside(header->program_header_offset,
                        (uint64_t)header->program_header_count * ELF_PROGRAM_HEADER_BYTES, blob_size)) {
        ft_tt_log_error("elf: program header table runs past the end of the blob");
        return 0;
    }
    return 1;
}

/** Validate one PT_LOAD and record it as the next segment of the image. */
static int add_load_segment(const uint8_t *bytes, size_t blob_size, size_t header_at, ft_tt_elf_image *image)
{
    uint32_t file_offset = read_u32(bytes, header_at + 4u);
    uint32_t exec_address = read_u32(bytes, header_at + 8u);
    uint32_t load_address = read_u32(bytes, header_at + 12u);
    uint32_t file_bytes = read_u32(bytes, header_at + 16u);
    uint32_t memory_bytes = read_u32(bytes, header_at + 20u);
    unsigned index = image->segment_count;
    ft_tt_elf_segment *segment;

    if (index >= FT_TT_ELF_MAX_SEGMENTS) {
        ft_tt_log_error("elf: more than %u loadable segments", FT_TT_ELF_MAX_SEGMENTS);
        return FT_TT_EINVAL;
    }
    if (file_offset % WORD_BYTES != 0u || exec_address % WORD_BYTES != 0u || load_address % WORD_BYTES != 0u) {
        ft_tt_log_error("elf: segment %u is not word aligned (offset 0x%x, vaddr 0x%x, paddr 0x%x)", index,
                        file_offset, exec_address, load_address);
        return FT_TT_EINVAL;
    }
    if (file_bytes % WORD_BYTES != 0u) {
        ft_tt_log_error("elf: segment %u file size %u is not a multiple of 4", index, file_bytes);
        return FT_TT_EINVAL;
    }
    if (!span_is_inside(file_offset, file_bytes, blob_size)) {
        ft_tt_log_error("elf: segment %u contents [0x%x, +%u) run past the %zu-byte blob", index, file_offset,
                        file_bytes, blob_size);
        return FT_TT_EINVAL;
    }
    if (file_bytes > memory_bytes || (uint64_t)load_address + file_bytes > UINT32_MAX) {
        ft_tt_log_error("elf: segment %u sizes are inconsistent (file %u, memory %u, load 0x%x)", index,
                        file_bytes, memory_bytes, load_address);
        return FT_TT_EINVAL;
    }

    segment = &image->segments[index];
    segment->exec_address = exec_address;
    segment->load_address = load_address;
    segment->file_data = bytes + file_offset;
    segment->file_bytes = file_bytes;
    /* The host transfers and clears whole 32-bit words, so a segment's memory size is
     * counted in whole words: rounded UP, never down (a partial last word is still
     * written). Real images are already word-sized (fw/ft_tensix_*.ld ALIGN(4)). */
    segment->memory_bytes = (memory_bytes + 3u) & ~3u;
    image->segment_count = index + 1u;
    image->loaded_bytes += file_bytes;
    return FT_TT_OK;
}

static int collect_load_segments(const uint8_t *bytes, size_t blob_size, const elf_header_fields *header,
                                 ft_tt_elf_image *image)
{
    uint32_t index;

    for (index = 0; index < header->program_header_count; index++) {
        size_t header_at = (size_t)header->program_header_offset + (size_t)index * ELF_PROGRAM_HEADER_BYTES;

        if (read_u32(bytes, header_at) != ELF_PROGRAM_LOAD)
            continue;
        if (add_load_segment(bytes, blob_size, header_at, image) != FT_TT_OK)
            return FT_TT_EINVAL;
    }
    if (image->segment_count == 0u) {
        ft_tt_log_error("elf: no PT_LOAD segments");
        return FT_TT_EINVAL;
    }
    if (header->entry != image->segments[0].exec_address) {
        ft_tt_log_error("elf: entry 0x%x is not the start of the text segment (0x%x)", header->entry,
                        image->segments[0].exec_address);
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

/** Bounds of section `index`'s bytes. The section table itself is already validated. */
static elf_section_span section_span(const uint8_t *bytes, const elf_header_fields *header, uint32_t index)
{
    size_t at = (size_t)header->section_header_offset + (size_t)index * ELF_SECTION_HEADER_BYTES;
    elf_section_span span;

    span.offset = read_u32(bytes, at + 16u);
    span.size = read_u32(bytes, at + 20u);
    if (read_u32(bytes, at + 4u) == ELF_SECTION_NO_BITS)
        span.size = 0u; /* occupies no file bytes */
    return span;
}

/**
 * Find the `.segments` section.
 *
 * @return 1 and `*found` when present, 0 when the ELF has no such section (or no section
 *         table at all), -1 when the section table is malformed.
 */
static int find_segments_section(const uint8_t *bytes, size_t blob_size, const elf_header_fields *header,
                                 elf_section_span *found)
{
    static const char wanted_name[] = ".segments";
    elf_section_span names;
    uint32_t index;

    if (header->section_header_offset == 0u || header->section_header_count == 0u)
        return 0;
    if (header->section_header_size != ELF_SECTION_HEADER_BYTES ||
        !span_is_inside(header->section_header_offset,
                        (uint64_t)header->section_header_count * ELF_SECTION_HEADER_BYTES, blob_size) ||
        header->section_name_table_index >= header->section_header_count) {
        ft_tt_log_error("elf: section header table is malformed or out of bounds");
        return -1;
    }
    names = section_span(bytes, header, header->section_name_table_index);
    if (!span_is_inside(names.offset, names.size, blob_size)) {
        ft_tt_log_error("elf: section name table runs past the end of the blob");
        return -1;
    }
    for (index = 0; index < header->section_header_count; index++) {
        size_t at = (size_t)header->section_header_offset + (size_t)index * ELF_SECTION_HEADER_BYTES;
        uint32_t name_offset = read_u32(bytes, at);

        if (name_offset >= names.size || names.size - name_offset < sizeof(wanted_name))
            continue;
        if (memcmp(bytes + names.offset + name_offset, wanted_name, sizeof(wanted_name)) != 0)
            continue;
        *found = section_span(bytes, header, index);
        if (!span_is_inside(found->offset, found->size, blob_size)) {
            ft_tt_log_error("elf: .segments runs past the end of the blob");
            return -1;
        }
        return 1;
    }
    return 0;
}

/**
 * Hold each PT_LOAD to the region the linker script recorded for it.
 *
 * The linker scripts write, per output region in segment order: the execution address
 * (twice; the second copy is ADDR() again, not the load address) and the region's size
 * limit. A PT_LOAD that starts elsewhere than its record says, or whose memory image is
 * larger than the limit, did not come from the layout the loader expects.
 */
static int check_against_segment_records(const uint8_t *bytes, elf_section_span records,
                                         const ft_tt_elf_image *image)
{
    uint32_t record_count = records.size / SEGMENT_RECORD_BYTES;
    unsigned index;

    if (records.size % SEGMENT_RECORD_BYTES != 0u) {
        ft_tt_log_error("elf: .segments is %u bytes, not a whole number of records", records.size);
        return FT_TT_EINVAL;
    }
    if (image->segment_count > record_count && record_count != 0u) {
        ft_tt_log_error("elf: %u loadable segments but .segments describes only %u", image->segment_count,
                        record_count);
        return FT_TT_EINVAL;
    }
    for (index = 0; index < image->segment_count && index < record_count; index++) {
        size_t at = (size_t)records.offset + (size_t)index * SEGMENT_RECORD_BYTES;
        const ft_tt_elf_segment *segment = &image->segments[index];
        uint32_t recorded_exec = read_u32(bytes, at);
        uint32_t recorded_second = read_u32(bytes, at + 4u);
        uint32_t limit = read_u32(bytes, at + 8u);

        if (recorded_exec != segment->exec_address || recorded_second != segment->exec_address) {
            ft_tt_log_error("elf: segment %u is at 0x%x but .segments records 0x%x/0x%x", index,
                            segment->exec_address, recorded_exec, recorded_second);
            return FT_TT_EINVAL;
        }
        if (segment->memory_bytes > limit) {
            ft_tt_log_error("elf: segment %u is %u bytes, over its %u-byte region", index, segment->memory_bytes,
                            limit);
            return FT_TT_EINVAL;
        }
    }
    return FT_TT_OK;
}

int ft_tt_elf_parse(const void *blob, size_t blob_size, ft_tt_elf_image *out_image)
{
    const uint8_t *bytes = blob;
    elf_header_fields header;
    elf_section_span records;
    int found;

    if (!blob || !out_image) {
        ft_tt_log_error("elf: NULL %s", blob ? "output image" : "blob");
        return FT_TT_EINVAL;
    }
    memset(out_image, 0, sizeof(*out_image));

    if (read_elf_header(bytes, blob_size, &header) != FT_TT_OK)
        return FT_TT_EINVAL;
    if (!program_header_table_is_valid(&header, blob_size))
        return FT_TT_EINVAL;
    if (collect_load_segments(bytes, blob_size, &header, out_image) != FT_TT_OK)
        return FT_TT_EINVAL;

    found = find_segments_section(bytes, blob_size, &header, &records);
    if (found < 0)
        return FT_TT_EINVAL;
    if (found > 0 && check_against_segment_records(bytes, records, out_image) != FT_TT_OK)
        return FT_TT_EINVAL;

    out_image->entry_address = header.entry;
    return FT_TT_OK;
}

const ft_tt_elf_segment *ft_tt_elf_text_segment(const ft_tt_elf_image *image)
{
    if (!image || image->segment_count == 0u)
        return NULL;
    return &image->segments[0];
}

int ft_tt_elf_load_extent(const ft_tt_elf_image *image, uint32_t *out_low, uint32_t *out_high)
{
    uint32_t low = UINT32_MAX;
    uint32_t high = 0u;
    unsigned index;

    if (!image || !out_low || !out_high || image->segment_count == 0u ||
        image->segment_count > FT_TT_ELF_MAX_SEGMENTS)
        return FT_TT_EINVAL;

    for (index = 0; index < image->segment_count; index++) {
        const ft_tt_elf_segment *segment = &image->segments[index];
        /* ft_tt_elf_parse() refused any segment whose end would not fit in 32 bits. */
        uint32_t end = segment->load_address + segment->file_bytes;

        if (segment->file_bytes == 0u)
            continue;
        if (segment->load_address < low)
            low = segment->load_address;
        if (end > high)
            high = end;
    }
    if (high == 0u) {
        low = image->segments[0].load_address;
        high = low;
    }
    *out_low = low;
    *out_high = high;
    return FT_TT_OK;
}
