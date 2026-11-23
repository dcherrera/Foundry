/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file generate_tlb2m_reference.c
 * @brief Offline generator for ft_tt_tlb2m_reference_vectors.h: the expected CONFIGURE_TLB
 *   words for Blackhole 2 MiB TLB windows, computed from the PUBLIC field table.
 *
 * SOURCE OF TRUTH. tt-isa-documentation, BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md,
 * "Configuration": the three 32-bit words `low32, mid32, high32` concatenate into 96 bits,
 * and the 2 MiB column gives each field's first bit and width. The table below is that
 * column. Where the encoder's field names differ from the doc:
 *   - `noc` is 2 bits at 67: the doc's `noc_sel` (67) plus the reserved bit 68;
 *   - `use_static_vc` is the doc's `static_vc` (73);
 *   - `stream_header` is the doc's reserved bit 74;
 *   - `static_vc` is 3 bits at 75: the doc's `static_vc_buddy` (75) and `static_vc_class` (76-77).
 *
 * WHY THIS IS AN INDEPENDENT ORACLE. ft_tt_udrv_tlb_encode.c packs with per-word shifts and
 * masks. This program sets the 96-bit register ONE BIT AT A TIME from (first bit, width), so
 * the word-straddling `y_start` (bits 61-66) needs no special case here, and a shift or mask
 * mistake in the encoder cannot be repeated by construction.
 *
 * The cases and the LCG sweep are unchanged from the generator this replaces (2026-10-08), and
 * its output was required to reproduce the previously committed vector data exactly.
 *
 * @status Clean-room rewrite 2026-10-08 (no driver source is read or compiled).
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

typedef struct tlb2m_field_vector {
    const char *name;
    uint64_t addr;
    uint8_t x_end, y_end, x_start, y_start;
    uint8_t noc, multicast, ordering, linked, use_static_vc, stream_header, static_vc;
} tlb2m_field_vector;

/** Window size: 2 MiB, so the stored local_offset is the address's bits 21 and up. */
#define TLB2M_WINDOW_SHIFT 21u

/** Set `width` bits of `value` into the 96-bit register starting at `first_bit`. */
static void insert_field_bitwise(uint32_t words[3], unsigned first_bit, unsigned width,
                                 uint64_t value)
{
    unsigned bit;

    for (bit = 0; bit < width; bit++) {
        unsigned position = first_bit + bit;

        if ((value >> bit) & 1u)
            words[position / 32u] |= (uint32_t)1u << (position % 32u);
    }
}

/** The doc's 2 MiB field table applied to one vector. */
static void encode_from_field_table(const tlb2m_field_vector *vector, uint32_t words[3])
{
    words[0] = words[1] = words[2] = 0u;
    insert_field_bitwise(words, 0u, 43u, vector->addr >> TLB2M_WINDOW_SHIFT); /* local_offset */
    insert_field_bitwise(words, 43u, 6u, vector->x_end);
    insert_field_bitwise(words, 49u, 6u, vector->y_end);
    insert_field_bitwise(words, 55u, 6u, vector->x_start);
    insert_field_bitwise(words, 61u, 6u, vector->y_start);
    insert_field_bitwise(words, 67u, 2u, vector->noc);
    insert_field_bitwise(words, 69u, 1u, vector->multicast);
    insert_field_bitwise(words, 70u, 2u, vector->ordering);
    insert_field_bitwise(words, 72u, 1u, vector->linked);
    insert_field_bitwise(words, 73u, 1u, vector->use_static_vc);
    insert_field_bitwise(words, 74u, 1u, vector->stream_header);
    insert_field_bitwise(words, 75u, 3u, vector->static_vc);
}

/* Boundary cases, chosen against the field list: every field alone at its maximum, the
 * word-straddling y_start swept across the boundary, the address at bit 21 and bit 63, and
 * the all-ones register. */
static const tlb2m_field_vector k_boundary_vectors[] = {
    { "all_zero",            0,                       0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "addr_one_window",     1ull << 21,              0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "addr_bit_31",         1ull << 31,              0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "addr_bit_32",         1ull << 32,              0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "addr_bit_52",         1ull << 52,              0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "addr_bit_63",         1ull << 63,              0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "addr_all_ones",       0xFFFFFFFFFFE00000ull,   0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "x_end_max",           0,                      63,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "y_end_max",           0,                       0, 63,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "x_start_max",         0,                       0,  0, 63,  0,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_max",         0,                       0,  0,  0, 63,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_1",           0,                       0,  0,  0,  1,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_2",           0,                       0,  0,  0,  2,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_4",           0,                       0,  0,  0,  4,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_7",           0,                       0,  0,  0,  7,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_8",           0,                       0,  0,  0,  8,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_15",          0,                       0,  0,  0, 15,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_16",          0,                       0,  0,  0, 16,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_31",          0,                       0,  0,  0, 31,  0, 0, 0, 0, 0, 0, 0 },
    { "y_start_32",          0,                       0,  0,  0, 32,  0, 0, 0, 0, 0, 0, 0 },
    { "noc_1",               0,                       0,  0,  0,  0,  1, 0, 0, 0, 0, 0, 0 },
    { "noc_max",             0,                       0,  0,  0,  0,  3, 0, 0, 0, 0, 0, 0 },
    { "multicast",           0,                       0,  0,  0,  0,  0, 1, 0, 0, 0, 0, 0 },
    { "ordering_strict",     0,                       0,  0,  0,  0,  0, 0, 1, 0, 0, 0, 0 },
    { "ordering_max",        0,                       0,  0,  0,  0,  0, 0, 3, 0, 0, 0, 0 },
    { "linked",              0,                       0,  0,  0,  0,  0, 0, 0, 1, 0, 0, 0 },
    { "use_static_vc",       0,                       0,  0,  0,  0,  0, 0, 0, 0, 1, 0, 0 },
    { "stream_header",       0,                       0,  0,  0,  0,  0, 0, 0, 0, 0, 1, 0 },
    { "static_vc_max",       0,                       0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 7 },
    { "arc_tile_unicast",    0x80030400ull & ~0x1FFFFFull, 8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
    { "tensix_unicast",      0x0ull,                  1, 16,  0,  0,  0, 0, 1, 0, 0, 0, 0 },
    { "dram_translated",     0x400000ull,            17, 12,  0,  0,  0, 0, 1, 0, 0, 0, 0 },
    { "multicast_rect",      0x200000ull,            10,  9,  1,  2,  0, 1, 0, 0, 0, 0, 0 },
    { "everything_max",      0xFFFFFFFFFFE00000ull,  63, 63, 63, 63,  3, 1, 3, 1, 1, 1, 7 },
};

#define SWEEP_VECTOR_COUNT 62u

/* Deterministic LCG: the boundary cases prove the edges, the sweep catches a field swapped
 * with a neighbour in a combination nobody wrote down. */
static uint64_t sweep_lcg_state = 0x2545F4914F6CDD1Dull;

static uint64_t sweep_lcg_next(void)
{
    sweep_lcg_state = (sweep_lcg_state * 6364136223846793005ull) + 1442695040888963407ull;
    return sweep_lcg_state >> 17;
}

static void print_vector_row(const tlb2m_field_vector *vector)
{
    uint32_t words[3];

    encode_from_field_table(vector, words);
    printf("    { \"%s\", UINT64_C(0x%016" PRIx64 "), %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u,"
           " { 0x%08" PRIx32 "u, 0x%08" PRIx32 "u, 0x%08" PRIx32 "u } },\n",
           vector->name, vector->addr, vector->x_end, vector->y_end, vector->x_start,
           vector->y_start, vector->noc, vector->multicast, vector->ordering, vector->linked,
           vector->use_static_vc, vector->stream_header, vector->static_vc, words[0], words[1],
           words[2]);
}

int main(void)
{
    char sweep_names[SWEEP_VECTOR_COUNT][24];
    size_t boundary_count = sizeof(k_boundary_vectors) / sizeof(k_boundary_vectors[0]);
    size_t index;

    printf("#define FT_TT_TLB2M_REF_VECTOR_COUNT %zu\n\n", boundary_count + SWEEP_VECTOR_COUNT);
    printf("static const ft_tt_tlb2m_reference_vector ft_tt_tlb2m_reference_vectors"
           "[FT_TT_TLB2M_REF_VECTOR_COUNT] = {\n");

    for (index = 0; index < boundary_count; index++)
        print_vector_row(&k_boundary_vectors[index]);

    for (index = 0; index < SWEEP_VECTOR_COUNT; index++) {
        tlb2m_field_vector vector;
        uint64_t random = sweep_lcg_next();

        snprintf(sweep_names[index], sizeof(sweep_names[0]), "sweep_%02zu", index);
        vector.name = sweep_names[index];
        vector.addr = (sweep_lcg_next() << 21) & 0xFFFFFFFFFFE00000ull;
        vector.x_end = (uint8_t)(random & 0x3Fu);
        vector.y_end = (uint8_t)((random >> 6) & 0x3Fu);
        vector.x_start = (uint8_t)((random >> 12) & 0x3Fu);
        vector.y_start = (uint8_t)((random >> 18) & 0x3Fu);
        vector.noc = (uint8_t)((random >> 24) & 0x3u);
        vector.multicast = (uint8_t)((random >> 26) & 0x1u);
        vector.ordering = (uint8_t)((random >> 27) & 0x3u);
        vector.linked = (uint8_t)((random >> 29) & 0x1u);
        vector.use_static_vc = (uint8_t)((random >> 30) & 0x1u);
        vector.stream_header = (uint8_t)((random >> 31) & 0x1u);
        vector.static_vc = (uint8_t)((random >> 32) & 0x7u);
        print_vector_row(&vector);
    }

    printf("};\n");
    return 0;
}
