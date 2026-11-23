/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_udrv_tlb_encode.c
 * @brief Gate for ft_tt_udrv_tlb_encode.c — the CONFIGURE_TLB register words.
 *
 *   The primary gate is 96 reference vectors whose expected words are computed
 *   bit by bit from the public field table (tests/reference/
 *   generate_tlb2m_reference.c). Nothing in this file transcribes the bit
 *   layout, and the generator does not share the encoder's method.
 *
 *   The register OFFSETS are pinned literally, on purpose. `0x1FC009D8` for
 *   window 0's strided register is 210 * 12 past the register file base — the
 *   TOTAL window count including the eight 4 GiB windows this board does not
 *   have. Sizing it from 202 gives 0x1FC00978 and sizing it from 214 gives
 *   0x1FC00A08; both look right and both would zero the wrong register.
 *
 * @status Off-hardware. 100% of what it covers is arithmetic; nothing here
 *   touches a device.
 * @issues None
 * @todo None
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_test_report.h"
#include "ft_tt_udrv.h"
#include "ft_tt_udrv_tlb_encode.h"
#include "reference/ft_tt_tlb2m_reference_vectors.h"

/** Turn a reference vector's fields into the encoder's config struct. */
static void config_from_vector(const ft_tt_tlb2m_reference_vector *vector,
                               ft_tt_udrv_tlb2m_config *out_config)
{
    memset(out_config, 0, sizeof(*out_config));
    out_config->addr = vector->addr;
    out_config->x_end = vector->x_end;
    out_config->y_end = vector->y_end;
    out_config->x_start = vector->x_start;
    out_config->y_start = vector->y_start;
    out_config->noc = vector->noc;
    out_config->multicast = vector->multicast;
    out_config->ordering = vector->ordering;
    out_config->linked = vector->linked;
    out_config->use_static_vc = vector->use_static_vc;
    out_config->stream_header = vector->stream_header;
    out_config->static_vc_id = vector->static_vc;
}

static void stage_reference_vectors(void)
{
    size_t index;

    ft_tt_test_step("Stage 1 — %d reference vectors from the public field table",
                    FT_TT_TLB2M_REF_VECTOR_COUNT);

    for (index = 0u; index < FT_TT_TLB2M_REF_VECTOR_COUNT; index++) {
        const ft_tt_tlb2m_reference_vector *vector = &ft_tt_tlb2m_reference_vectors[index];
        ft_tt_udrv_tlb2m_config config;
        uint32_t words[3] = {0u, 0u, 0u};
        char what[96];
        int rc;

        config_from_vector(vector, &config);
        rc = ft_tt_udrv_tlb2m_encode(&config, words);

        (void)snprintf(what, sizeof(what), "vector %02zu %s encodes", index, vector->name);

        if (rc != FT_TT_UDRV_OK) {
            ft_tt_test_check_fmt(what, false, "encode returned %d", rc);
            continue;
        }

        ft_tt_test_check_fmt(what,
                             words[0] == vector->words[0] && words[1] == vector->words[1] &&
                                 words[2] == vector->words[2],
                             "got %08x %08x %08x, reference says %08x %08x %08x", words[0], words[1],
                             words[2], vector->words[0], vector->words[1], vector->words[2]);
    }
}

static void stage_straddle_is_real(void)
{
    ft_tt_udrv_tlb2m_config config = {0};
    uint32_t low[3] = {0u, 0u, 0u};
    uint32_t high[3] = {0u, 0u, 0u};

    ft_tt_test_step("Stage 2 — y_start really does straddle the mid32/high32 boundary");

    /* If y_start did not straddle, one of these two would leave the other
     * word untouched. This is the single property the whole file exists for,
     * so it is asserted directly rather than left implicit in the vectors. */
    config.y_start = 7u; /* bits 61..63 of the low word */
    (void)ft_tt_udrv_tlb2m_encode(&config, low);

    config.y_start = 56u; /* bits 64..66, i.e. entirely in the high word */
    (void)ft_tt_udrv_tlb2m_encode(&config, high);

    ft_tt_test_check_fmt("y_start=7 lands only in mid32", low[1] != 0u && low[2] == 0u,
                         "mid32=%08x high32=%08x", low[1], low[2]);
    ft_tt_test_check_fmt("y_start=56 lands only in high32", high[1] == 0u && high[2] != 0u,
                         "mid32=%08x high32=%08x", high[1], high[2]);

    config.y_start = 63u;
    (void)ft_tt_udrv_tlb2m_encode(&config, low);
    ft_tt_test_check_fmt("y_start=63 lands in BOTH words", low[1] != 0u && low[2] != 0u,
                         "mid32=%08x high32=%08x", low[1], low[2]);
}

static void stage_alignment_refusal(void)
{
    ft_tt_udrv_tlb2m_config config = {0};
    uint32_t words[3] = {0u, 0u, 0u};

    ft_tt_test_step("Stage 3 — the kernel's own refusals");

    config.addr = (1ull << 21) + 4u;
    ft_tt_test_check_fmt("a misaligned address is refused",
                         ft_tt_udrv_tlb2m_encode(&config, words) == FT_TT_UDRV_EINVAL,
                         "addr=0x%llx", (unsigned long long)config.addr);

    config.addr = 1ull << 20;
    ft_tt_test_check_fmt("a 1 MiB address is refused (windows are 2 MiB)",
                         ft_tt_udrv_tlb2m_encode(&config, words) == FT_TT_UDRV_EINVAL, "addr=1MiB");

    config.addr = 1ull << 21;
    ft_tt_test_check("a 2 MiB address is accepted",
                     ft_tt_udrv_tlb2m_encode(&config, words) == FT_TT_UDRV_OK, "addr=2MiB");

    ft_tt_test_check("a NULL config is refused",
                     ft_tt_udrv_tlb2m_encode(NULL, words) == FT_TT_UDRV_EINVAL, "NULL");

    ft_tt_test_check("a NULL output is refused",
                     ft_tt_udrv_tlb2m_encode(&config, NULL) == FT_TT_UDRV_EINVAL, "NULL");
}

static void stage_truncation_matches_the_kernel(void)
{
    ft_tt_udrv_tlb2m_config wide = {0};
    ft_tt_udrv_tlb2m_config narrow = {0};
    uint32_t wide_words[3] = {0u, 0u, 0u};
    uint32_t narrow_words[3] = {0u, 0u, 0u};

    ft_tt_test_step("Stage 4 — out-of-range fields truncate exactly as a bitfield does");

    /* The kernel assigns straight into a 6-bit bitfield, so 64 becomes 0 and
     * 65 becomes 1 with no diagnostic. An encoder that masked differently —
     * or clamped — would aim a window somewhere else entirely. */
    wide.x_end = 64u;
    narrow.x_end = 0u;
    (void)ft_tt_udrv_tlb2m_encode(&wide, wide_words);
    (void)ft_tt_udrv_tlb2m_encode(&narrow, narrow_words);
    ft_tt_test_check_fmt("x_end=64 encodes as x_end=0",
                         memcmp(wide_words, narrow_words, sizeof(wide_words)) == 0,
                         "%08x %08x %08x", wide_words[0], wide_words[1], wide_words[2]);

    memset(&wide, 0, sizeof(wide));
    memset(&narrow, 0, sizeof(narrow));
    wide.y_start = 70u;
    narrow.y_start = 6u;
    (void)ft_tt_udrv_tlb2m_encode(&wide, wide_words);
    (void)ft_tt_udrv_tlb2m_encode(&narrow, narrow_words);
    ft_tt_test_check_fmt("y_start=70 encodes as y_start=6",
                         memcmp(wide_words, narrow_words, sizeof(wide_words)) == 0,
                         "%08x %08x %08x", wide_words[0], wide_words[1], wide_words[2]);

    /* ...and the separate validator is what a caller uses to be told. */
    memset(&wide, 0, sizeof(wide));
    wide.x_end = 64u;
    ft_tt_test_check("config_check refuses x_end=64",
                     ft_tt_udrv_tlb2m_config_check(&wide) == FT_TT_UDRV_EINVAL, "64 > 6 bits");

    memset(&wide, 0, sizeof(wide));
    wide.x_end = 63u;
    ft_tt_test_check("config_check accepts x_end=63",
                     ft_tt_udrv_tlb2m_config_check(&wide) == FT_TT_UDRV_OK, "63 fits 6 bits");

    memset(&wide, 0, sizeof(wide));
    wide.static_vc_id = 8u;
    ft_tt_test_check("config_check refuses static_vc_id=8",
                     ft_tt_udrv_tlb2m_config_check(&wide) == FT_TT_UDRV_EINVAL, "8 > 3 bits");

    memset(&wide, 0, sizeof(wide));
    wide.ordering = 4u;
    ft_tt_test_check("config_check refuses ordering=4",
                     ft_tt_udrv_tlb2m_config_check(&wide) == FT_TT_UDRV_EINVAL, "4 > 2 bits");
}

static void stage_register_offsets(void)
{
    uint32_t offset = 0u;
    uint64_t aperture = 0u;

    ft_tt_test_step("Stage 5 — register and aperture offsets, pinned literally");

    ft_tt_test_check_fmt("window 0 registers at 0x1FC00000",
                         ft_tt_udrv_tlb2m_register_offset(0u, &offset) == FT_TT_UDRV_OK &&
                             offset == 0x1FC00000u,
                         "0x%08x", offset);

    ft_tt_test_check_fmt("window 1 registers at 0x1FC0000C (stride 12, not 16)",
                         ft_tt_udrv_tlb2m_register_offset(1u, &offset) == FT_TT_UDRV_OK &&
                             offset == 0x1FC0000Cu,
                         "0x%08x", offset);

    ft_tt_test_check_fmt("window 201 (tt-kmd's own) registers at 0x1FC0096C",
                         ft_tt_udrv_tlb2m_register_offset(201u, &offset) == FT_TT_UDRV_OK &&
                             offset == 0x1FC0096Cu,
                         "0x%08x", offset);

    ft_tt_test_check("window 202 is refused",
                     ft_tt_udrv_tlb2m_register_offset(202u, &offset) == FT_TT_UDRV_EINVAL,
                     "only 202 windows exist");

    /* THE PIN THAT MATTERS. 210 * 12 = 2520 = 0x9D8. Sizing this from the 2 MiB
     * count alone gives 0x978; a common mis-derivation gives 0xA08. */
    ft_tt_test_check_fmt("window 0 strided register at 0x1FC009D8",
                         ft_tt_udrv_tlb2m_strided_register_offset(0u, &offset) ==
                                 FT_TT_UDRV_OK &&
                             offset == 0x1FC009D8u,
                         "0x%08x", offset);

    ft_tt_test_check_fmt("window 31 strided register at 0x1FC00A54",
                         ft_tt_udrv_tlb2m_strided_register_offset(31u, &offset) ==
                                 FT_TT_UDRV_OK &&
                             offset == 0x1FC00A54u,
                         "0x%08x", offset);

    ft_tt_test_check("window 32 has NO strided register",
                     ft_tt_udrv_tlb2m_strided_register_offset(32u, &offset) == FT_TT_UDRV_ENOTSUP,
                     "only the first 32 windows do");

    ft_tt_test_check("window 201 has NO strided register",
                     ft_tt_udrv_tlb2m_strided_register_offset(201u, &offset) ==
                         FT_TT_UDRV_ENOTSUP,
                     "201 >= 32");

    ft_tt_test_check_fmt("the strided block stays inside the 4 KiB register file",
                         0x9D8u + (32u * 4u) <= FT_TT_UDRV_TLB_REGS_LEN, "%u <= %u",
                         0x9D8u + 128u, FT_TT_UDRV_TLB_REGS_LEN);

    ft_tt_test_check_fmt("window 0 aperture at BAR0 + 0",
                         ft_tt_udrv_tlb2m_aperture_offset(0u, &aperture) == FT_TT_UDRV_OK &&
                             aperture == 0u,
                         "0x%llx", (unsigned long long)aperture);

    ft_tt_test_check_fmt("window 201 aperture at BAR0 + 0x19200000",
                         ft_tt_udrv_tlb2m_aperture_offset(201u, &aperture) == FT_TT_UDRV_OK &&
                             aperture == 0x19200000ull,
                         "0x%llx", (unsigned long long)aperture);

    /* Every window's aperture must fit the 512 MiB BAR0 this board has. */
    ft_tt_test_check_fmt("the last window's aperture ends inside a 512 MiB BAR0",
                         0x19200000ull + FT_TT_UDRV_TLB_2M_WINDOW_SIZE <= (512ull << 20),
                         "0x%llx <= 0x%llx",
                         0x19200000ull + FT_TT_UDRV_TLB_2M_WINDOW_SIZE, 512ull << 20);
}

static void stage_unicast_helper(void)
{
    ft_tt_udrv_tlb2m_config config;
    uint32_t helper_words[3] = {0u, 0u, 0u};
    uint32_t manual_words[3] = {0u, 0u, 0u};
    ft_tt_udrv_tlb2m_config manual = {0};

    ft_tt_test_step("Stage 6 — the unicast helper maps fields the way CONFIGURE_TLB does");

    ft_tt_test_check("unicast(8, 0, ARC scratch page, noc 0, strict) builds",
                     ft_tt_udrv_tlb2m_config_unicast(&config, 8u, 0u, 0x80000000ull, 0u,
                                                     FT_TT_UDRV_TLB_ORDERING_STRICT) ==
                         FT_TT_UDRV_OK,
                     "");

    manual.addr = 0x80000000ull;
    manual.x_end = 8u;
    manual.y_end = 0u;
    manual.ordering = FT_TT_UDRV_TLB_ORDERING_STRICT;

    (void)ft_tt_udrv_tlb2m_encode(&config, helper_words);
    (void)ft_tt_udrv_tlb2m_encode(&manual, manual_words);

    ft_tt_test_check_fmt("it equals the hand-built unicast config",
                         memcmp(helper_words, manual_words, sizeof(helper_words)) == 0,
                         "%08x %08x %08x", helper_words[0], helper_words[1], helper_words[2]);

    ft_tt_test_check_fmt("it leaves every multicast field at zero",
                         config.x_start == 0u && config.y_start == 0u && config.multicast == 0u,
                         "x_start=%u y_start=%u mcast=%u", config.x_start, config.y_start,
                         config.multicast);

    ft_tt_test_check("a misaligned unicast address is refused",
                     ft_tt_udrv_tlb2m_config_unicast(&config, 8u, 0u, 0x1000ull, 0u, 0u) ==
                         FT_TT_UDRV_EINVAL,
                     "0x1000 is not 2 MiB aligned");

    ft_tt_test_check("an out-of-range coordinate is refused",
                     ft_tt_udrv_tlb2m_config_unicast(&config, 64u, 0u, 0ull, 0u, 0u) ==
                         FT_TT_UDRV_EINVAL,
                     "x=64 does not fit 6 bits");
}

/**
 * The decoder, gated against the SAME reference vectors as the encoder.
 *
 * Deliberately NOT a round-trip through our own encoder. A decoder checked
 * only by `decode(encode(c)) == c` agrees with whatever the encoder does,
 * including agreeing with it about a field in the wrong place — the exact
 * self-consistency trap DEV_NOTES §3 catalogues. Here the INPUT is the
 * driver's own words and the EXPECTED OUTPUT is the driver's own fields, so
 * the encoder is not in the loop at all.
 */
static void stage_decode_against_the_kernels_words(void)
{
    ft_tt_udrv_tlb2m_config sink = {0};
    size_t index;

    ft_tt_test_step("Stage 7 — the decoder recovers every reference vector's fields from its words");

    for (index = 0u; index < FT_TT_TLB2M_REF_VECTOR_COUNT; index++) {
        const ft_tt_tlb2m_reference_vector *vector = &ft_tt_tlb2m_reference_vectors[index];
        ft_tt_udrv_tlb2m_config expected;
        ft_tt_udrv_tlb2m_config decoded;

        config_from_vector(vector, &expected);
        memset(&decoded, 0xA5, sizeof(decoded));

        if (ft_tt_udrv_tlb2m_decode(vector->words, &decoded) != FT_TT_UDRV_OK) {
            ft_tt_test_check_fmt("decode succeeds", false, "%s", vector->name);
            continue;
        }

        ft_tt_test_check_fmt(
            vector->name, ft_tt_udrv_tlb2m_config_equal(&expected, &decoded),
            "addr 0x%llx x_end %u y_end %u x_start %u y_start %u noc %u mcast %u ord %u "
            "linked %u vc? %u sh %u vc %u",
            (unsigned long long)decoded.addr, decoded.x_end, decoded.y_end, decoded.x_start,
            decoded.y_start, decoded.noc, decoded.multicast, decoded.ordering, decoded.linked,
            decoded.use_static_vc, decoded.stream_header, decoded.static_vc_id);
    }

    ft_tt_test_check("a NULL word array is refused",
                     ft_tt_udrv_tlb2m_decode(NULL, &sink) == FT_TT_UDRV_EINVAL, "");
    ft_tt_test_check("a NULL output is refused",
                     ft_tt_udrv_tlb2m_decode(ft_tt_tlb2m_reference_vectors[0].words, NULL) ==
                         FT_TT_UDRV_EINVAL,
                     "");
}

/**
 * The comparator itself. A gate built on `config_equal` is worthless if
 * `config_equal` returns true for two configs that differ, so every field gets
 * perturbed by one step and the comparator must notice.
 */
static void stage_the_comparator_can_say_no(void)
{
    static const char *const names[] = {"addr",          "x_end",         "y_end",
                                        "x_start",       "y_start",       "noc",
                                        "multicast",     "ordering",      "linked",
                                        "use_static_vc", "stream_header", "static_vc_id"};
    ft_tt_udrv_tlb2m_config base = {0};
    size_t field;

    ft_tt_test_step("Stage 8 — config_equal notices a difference in EVERY field");

    base.addr = 0x200000ull;
    base.x_end = 8u;
    base.y_end = 3u;
    base.x_start = 1u;
    base.y_start = 2u;
    base.noc = 1u;
    base.multicast = 1u;
    base.ordering = 1u;
    base.linked = 1u;
    base.use_static_vc = 1u;
    base.stream_header = 1u;
    base.static_vc_id = 3u;

    ft_tt_test_check("a config equals itself", ft_tt_udrv_tlb2m_config_equal(&base, &base), "");

    for (field = 0u; field < sizeof(names) / sizeof(names[0]); field++) {
        ft_tt_udrv_tlb2m_config other = base;
        uint8_t *by_index[] = {NULL,
                               &other.x_end,
                               &other.y_end,
                               &other.x_start,
                               &other.y_start,
                               &other.noc,
                               &other.multicast,
                               &other.ordering,
                               &other.linked,
                               &other.use_static_vc,
                               &other.stream_header,
                               &other.static_vc_id};

        if (field == 0u)
            other.addr += (1ull << FT_TT_UDRV_TLB_2M_SHIFT);
        else
            *by_index[field] = (uint8_t)(*by_index[field] + 1u);

        ft_tt_test_check_fmt("a one-step change is noticed",
                             !ft_tt_udrv_tlb2m_config_equal(&base, &other), "%s", names[field]);
    }

    ft_tt_test_check("NULL never compares equal",
                     !ft_tt_udrv_tlb2m_config_equal(&base, NULL) &&
                         !ft_tt_udrv_tlb2m_config_equal(NULL, &base),
                     "");
}

/**
 * WHAT `FT_TT_UDRV_AIM_VERIFY=word` CAN AND CANNOT SEE, PROVEN ON THE ENCODING.
 *
 * The `word` verification reads register word 1 and nothing else, and it is on record as
 * giving up "the low address bits". THE BIT LAYOUT SAYS OTHERWISE, and this stage is the
 * proof rather than the assertion: the address field is `addr >> 21` at bit 0, 43 bits
 * wide, so **word 0 IS address bits 21..52** and word 1 carries address bits 53..63 —
 * which are zero for every endpoint on this part. `word` reads back the CORE and not one
 * significant address bit.
 *
 * Two endpoints one 2 MiB page apart on the SAME core must therefore encode to an IDENTICAL
 * word 1 and a different word 0. If that ever stops being true this stage fails, and the
 * safety argument against shipping `word` has to be re-derived instead of re-asserted.
 */
static void stage_word_verify_cannot_see_the_address(void)
{
    ft_tt_udrv_tlb2m_config here;
    ft_tt_udrv_tlb2m_config one_page_on;
    ft_tt_udrv_tlb2m_config decoded;
    ft_tt_udrv_tlb2m_config high;
    uint32_t words_here[3];
    uint32_t words_on[3];
    uint32_t highest[3];

    memset(&here, 0, sizeof(here));
    /* The run-control block the dispatch path aims at, on a Tensix. Its neighbour one page
     * up is what a corrupted word 0 selects. */
    here.addr = 0x0ffa00000ull;
    here.x_end = 18u;
    here.y_end = 11u;
    one_page_on = here;
    one_page_on.addr = here.addr + (1ull << FT_TT_UDRV_TLB_2M_SHIFT);

    ft_tt_test_check("both aims encode",
                     ft_tt_udrv_tlb2m_encode(&here, words_here) == FT_TT_UDRV_OK &&
                         ft_tt_udrv_tlb2m_encode(&one_page_on, words_on) == FT_TT_UDRV_OK,
                     "");
    ft_tt_test_check("they differ in word 0", words_here[0] != words_on[0],
                     "the 2 MiB base lives in word 0");
    /* THE WHOLE POINT. */
    ft_tt_test_check("and word 1 is IDENTICAL", words_here[1] == words_on[1],
                     "a word-1 readback cannot tell these two apart");
    ft_tt_test_check("word 2 is identical too", words_here[2] == words_on[2], "");
    ft_tt_test_check("the endpoint really is in word 1",
                     ft_tt_udrv_tlb2m_decode(words_here, &decoded) == FT_TT_UDRV_OK &&
                         decoded.x_end == 18u && decoded.y_end == 11u,
                     "x_end at bit 43 and y_end at bit 49 both land in word 1");

    /* AND THE ELEVEN ADDRESS BITS WORD 1 DOES CARRY ARE ZERO ON THIS PART. They are
     * non-zero only past 2^53 = 8 PiB; the card is 32 GiB of DRAM and the highest L1 base
     * the dispatch path aims at is 0x0ffa00000. Encode an 8 TiB address — far past
     * anything reachable — and word 1's address bits are still zero. */
    memset(&high, 0, sizeof(high));
    high.addr = 0x7FFFFE00000ull & ~((1ull << FT_TT_UDRV_TLB_2M_SHIFT) - 1ull);
    ft_tt_test_check("an 8 TiB address encodes",
                     ft_tt_udrv_tlb2m_encode(&high, highest) == FT_TT_UDRV_OK, "");
    ft_tt_test_check("and word 1's address bits are STILL zero", (highest[1] & 0x7FFu) == 0u,
                     "address bits 53..63 are unreachable on a 32 GiB part");
}

int main(void)
{
    stage_reference_vectors();
    stage_word_verify_cannot_see_the_address();
    stage_straddle_is_real();
    stage_alignment_refusal();
    stage_truncation_matches_the_kernel();
    stage_register_offsets();
    stage_unicast_helper();
    stage_decode_against_the_kernels_words();
    stage_the_comparator_can_say_no();

    return ft_tt_test_finish();
}
