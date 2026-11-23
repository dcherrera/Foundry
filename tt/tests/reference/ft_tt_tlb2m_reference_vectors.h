/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_tlb2m_reference_vectors.h
 * @brief GENERATED — do not edit. Expected CONFIGURE_TLB register words for Blackhole 2 MiB
 *   TLB windows, computed bit by bit from the public field table
 *   (tt-isa-documentation, BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md) by
 *   tests/reference/generate_tlb2m_reference.c — not by the encoder's own method.
 *
 *   Regenerate with:
 *     tests/reference/generate_tlb2m_reference.sh \
 *         > tests/reference/ft_tt_tlb2m_reference_vectors.h
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef FT_TT_TLB2M_REFERENCE_VECTORS_H
#define FT_TT_TLB2M_REFERENCE_VECTORS_H

#include <stdint.h>

/** One case: the config fields, and the three words the field table produces from them. */
typedef struct ft_tt_tlb2m_reference_vector {
    const char *name;
    uint64_t addr;
    uint8_t x_end;
    uint8_t y_end;
    uint8_t x_start;
    uint8_t y_start;
    uint8_t noc;
    uint8_t multicast;
    uint8_t ordering;
    uint8_t linked;
    uint8_t use_static_vc;
    uint8_t stream_header;
    uint8_t static_vc;
    uint32_t words[3];
} ft_tt_tlb2m_reference_vector;

#define FT_TT_TLB2M_REF_VECTOR_COUNT 96

static const ft_tt_tlb2m_reference_vector ft_tt_tlb2m_reference_vectors[FT_TT_TLB2M_REF_VECTOR_COUNT] = {
    { "all_zero", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000000u } },
    { "addr_one_window", UINT64_C(0x0000000000200000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0x00000001u, 0x00000000u, 0x00000000u } },
    { "addr_bit_31", UINT64_C(0x0000000080000000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0x00000400u, 0x00000000u, 0x00000000u } },
    { "addr_bit_32", UINT64_C(0x0000000100000000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0x00000800u, 0x00000000u, 0x00000000u } },
    { "addr_bit_52", UINT64_C(0x0010000000000000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0x80000000u, 0x00000000u, 0x00000000u } },
    { "addr_bit_63", UINT64_C(0x8000000000000000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x00000400u, 0x00000000u } },
    { "addr_all_ones", UINT64_C(0xffffffffffe00000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0xffffffffu, 0x000007ffu, 0x00000000u } },
    { "x_end_max", UINT64_C(0x0000000000000000), 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x0001f800u, 0x00000000u } },
    { "y_end_max", UINT64_C(0x0000000000000000), 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x007e0000u, 0x00000000u } },
    { "x_start_max", UINT64_C(0x0000000000000000), 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x1f800000u, 0x00000000u } },
    { "y_start_max", UINT64_C(0x0000000000000000), 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0xe0000000u, 0x00000007u } },
    { "y_start_1", UINT64_C(0x0000000000000000), 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x20000000u, 0x00000000u } },
    { "y_start_2", UINT64_C(0x0000000000000000), 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x40000000u, 0x00000000u } },
    { "y_start_4", UINT64_C(0x0000000000000000), 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x80000000u, 0x00000000u } },
    { "y_start_7", UINT64_C(0x0000000000000000), 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0xe0000000u, 0x00000000u } },
    { "y_start_8", UINT64_C(0x0000000000000000), 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000001u } },
    { "y_start_15", UINT64_C(0x0000000000000000), 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0xe0000000u, 0x00000001u } },
    { "y_start_16", UINT64_C(0x0000000000000000), 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000002u } },
    { "y_start_31", UINT64_C(0x0000000000000000), 0, 0, 0, 31, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0xe0000000u, 0x00000003u } },
    { "y_start_32", UINT64_C(0x0000000000000000), 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000004u } },
    { "noc_1", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000008u } },
    { "noc_max", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000018u } },
    { "multicast", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000020u } },
    { "ordering_strict", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000040u } },
    { "ordering_max", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x000000c0u } },
    { "linked", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, { 0x00000000u, 0x00000000u, 0x00000100u } },
    { "use_static_vc", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, { 0x00000000u, 0x00000000u, 0x00000200u } },
    { "stream_header", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, { 0x00000000u, 0x00000000u, 0x00000400u } },
    { "static_vc_max", UINT64_C(0x0000000000000000), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, { 0x00000000u, 0x00000000u, 0x00003800u } },
    { "arc_tile_unicast", UINT64_C(0x0000000080000000), 8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, { 0x00000400u, 0x00004000u, 0x00000040u } },
    { "tensix_unicast", UINT64_C(0x0000000000000000), 1, 16, 0, 0, 0, 0, 1, 0, 0, 0, 0, { 0x00000000u, 0x00200800u, 0x00000040u } },
    { "dram_translated", UINT64_C(0x0000000000400000), 17, 12, 0, 0, 0, 0, 1, 0, 0, 0, 0, { 0x00000002u, 0x00188800u, 0x00000040u } },
    { "multicast_rect", UINT64_C(0x0000000000200000), 10, 9, 1, 2, 0, 1, 0, 0, 0, 0, 0, { 0x00000001u, 0x40925000u, 0x00000020u } },
    { "everything_max", UINT64_C(0xffffffffffe00000), 63, 63, 63, 63, 3, 1, 3, 1, 1, 1, 7, { 0xffffffffu, 0xffffffffu, 0x00003fffu } },
    { "sweep_00", UINT64_C(0x765a806006e00000), 14, 58, 25, 49, 2, 1, 1, 0, 1, 0, 6, { 0xd4030037u, 0x2cf473b2u, 0x00003276u } },
    { "sweep_01", UINT64_C(0xc3a48ce926000000), 47, 38, 28, 55, 3, 1, 2, 1, 0, 0, 4, { 0x24674930u, 0xee4d7e1du, 0x000021beu } },
    { "sweep_02", UINT64_C(0x9f049c4376800000), 17, 17, 14, 13, 0, 1, 3, 1, 0, 1, 4, { 0x24e21bb4u, 0xa7228cf8u, 0x000025e1u } },
    { "sweep_03", UINT64_C(0x02b58fbe69a00000), 19, 41, 27, 39, 3, 1, 1, 1, 0, 0, 6, { 0xac7df34du, 0xedd29815u, 0x0000317cu } },
    { "sweep_04", UINT64_C(0xd0c6af181ca00000), 59, 39, 40, 47, 3, 0, 3, 1, 0, 1, 2, { 0x3578c0e5u, 0xf44fde86u, 0x000015ddu } },
    { "sweep_05", UINT64_C(0xe153296db1e00000), 28, 37, 6, 10, 0, 1, 1, 0, 0, 1, 0, { 0x994b6d8fu, 0x434ae70au, 0x00000461u } },
    { "sweep_06", UINT64_C(0x6a596461c4000000), 21, 47, 42, 8, 1, 1, 1, 1, 1, 1, 7, { 0xcb230e20u, 0x155eab52u, 0x00003f69u } },
    { "sweep_07", UINT64_C(0x18cf09b18e800000), 27, 35, 5, 11, 1, 0, 2, 0, 0, 0, 0, { 0x784d8c74u, 0x62c6d8c6u, 0x00000089u } },
    { "sweep_08", UINT64_C(0x5c5dc35d8f400000), 15, 60, 5, 47, 1, 0, 0, 1, 0, 1, 4, { 0xee1aec7au, 0xe2f87ae2u, 0x0000250du } },
    { "sweep_09", UINT64_C(0x009a34cb28000000), 62, 3, 2, 31, 0, 0, 2, 0, 0, 0, 2, { 0xd1a65940u, 0xe107f004u, 0x00001083u } },
    { "sweep_10", UINT64_C(0x26e708ab2a600000), 23, 51, 14, 27, 2, 1, 0, 1, 0, 1, 0, { 0x38455953u, 0x6766b937u, 0x00000533u } },
    { "sweep_11", UINT64_C(0x8408c7363b400000), 58, 50, 28, 39, 1, 1, 0, 0, 1, 0, 3, { 0x4639b1dau, 0xee65d420u, 0x00001a2cu } },
    { "sweep_12", UINT64_C(0x79ee8e54a2000000), 22, 44, 0, 43, 3, 1, 0, 1, 1, 0, 6, { 0x7472a510u, 0x6058b3cfu, 0x0000333du } },
    { "sweep_13", UINT64_C(0xb8545e5459a00000), 24, 52, 34, 50, 0, 0, 1, 0, 0, 0, 7, { 0xa2f2a2cdu, 0x5168c5c2u, 0x00003846u } },
    { "sweep_14", UINT64_C(0x29a9edd71ce00000), 8, 51, 26, 34, 3, 1, 2, 1, 0, 1, 5, { 0x4f6eb8e7u, 0x4d66414du, 0x00002dbcu } },
    { "sweep_15", UINT64_C(0x5c1b41fbae000000), 31, 52, 36, 41, 0, 1, 3, 1, 1, 0, 4, { 0xda0fdd70u, 0x3268fae0u, 0x000023e5u } },
    { "sweep_16", UINT64_C(0x1e612d72ba800000), 56, 15, 58, 32, 0, 0, 1, 1, 1, 0, 0, { 0x096b95d4u, 0x1d1fc0f3u, 0x00000344u } },
    { "sweep_17", UINT64_C(0x21ea011984000000), 27, 1, 61, 51, 1, 1, 1, 0, 1, 0, 5, { 0x5008cc20u, 0x7e82d90fu, 0x00002a6eu } },
    { "sweep_18", UINT64_C(0x845a81ebd5400000), 21, 61, 14, 16, 1, 0, 3, 0, 1, 1, 3, { 0xd40f5eaau, 0x077aac22u, 0x00001ecau } },
    { "sweep_19", UINT64_C(0x8874a61cd5400000), 8, 24, 26, 51, 2, 1, 0, 1, 1, 0, 5, { 0xa530e6aau, 0x6d304443u, 0x00002b36u } },
    { "sweep_20", UINT64_C(0xb2e79c0be8800000), 51, 53, 34, 11, 2, 0, 0, 1, 0, 1, 0, { 0x3ce05f44u, 0x716b9d97u, 0x00000511u } },
    { "sweep_21", UINT64_C(0xa3793aa118c00000), 30, 43, 17, 46, 0, 1, 0, 0, 1, 0, 6, { 0xc9d508c6u, 0xc8d6f51bu, 0x00003225u } },
    { "sweep_22", UINT64_C(0x154481945b400000), 33, 62, 36, 14, 2, 1, 3, 0, 0, 1, 7, { 0x240ca2dau, 0xd27d08aau, 0x00003cf1u } },
    { "sweep_23", UINT64_C(0xc61888d597a00000), 58, 13, 63, 39, 1, 0, 2, 1, 0, 0, 2, { 0xc446acbdu, 0xff9bd630u, 0x0000118cu } },
    { "sweep_24", UINT64_C(0xf09eb7f96e000000), 35, 25, 4, 12, 1, 1, 0, 0, 0, 1, 1, { 0xf5bfcb70u, 0x82331f84u, 0x00000c29u } },
    { "sweep_25", UINT64_C(0xf515a86185800000), 38, 37, 45, 53, 3, 0, 1, 1, 0, 0, 6, { 0xad430c2cu, 0xb6cb37a8u, 0x0000315eu } },
    { "sweep_26", UINT64_C(0x4224701f8d200000), 13, 6, 32, 0, 0, 1, 3, 0, 0, 0, 1, { 0x2380fc69u, 0x100c6a11u, 0x000008e0u } },
    { "sweep_27", UINT64_C(0x7bc6f0091c400000), 5, 19, 0, 53, 1, 0, 2, 1, 0, 1, 3, { 0x378048e2u, 0xa0262bdeu, 0x00001d8eu } },
    { "sweep_28", UINT64_C(0xa274117947200000), 60, 55, 43, 26, 3, 0, 2, 1, 1, 1, 2, { 0xa08bca39u, 0x55efe513u, 0x0000179bu } },
    { "sweep_29", UINT64_C(0xf5d1a11bfc800000), 29, 50, 10, 21, 3, 1, 2, 0, 0, 0, 4, { 0x8d08dfe4u, 0xa564efaeu, 0x000020bau } },
    { "sweep_30", UINT64_C(0x0dccdc4827200000), 50, 43, 35, 14, 3, 1, 2, 1, 0, 0, 6, { 0x66e24139u, 0xd1d7906eu, 0x000031b9u } },
    { "sweep_31", UINT64_C(0x60a791b912200000), 2, 31, 6, 9, 3, 0, 1, 1, 0, 0, 7, { 0x3c8dc891u, 0x233e1305u, 0x00003959u } },
    { "sweep_32", UINT64_C(0x8bcf8266b2a00000), 12, 4, 1, 0, 2, 0, 3, 0, 1, 0, 5, { 0x7c133595u, 0x0088645eu, 0x00002ad0u } },
    { "sweep_33", UINT64_C(0xf86603383c200000), 28, 33, 4, 31, 3, 1, 1, 0, 1, 1, 2, { 0x3019c1e1u, 0xe242e7c3u, 0x0000167bu } },
    { "sweep_34", UINT64_C(0x7461efe5cc000000), 34, 19, 32, 19, 1, 1, 2, 1, 1, 1, 7, { 0x0f7f2e60u, 0x702713a3u, 0x00003faau } },
    { "sweep_35", UINT64_C(0xb385b68df8e00000), 21, 48, 1, 14, 2, 1, 0, 1, 0, 1, 7, { 0x2db46fc7u, 0xc0e0ad9cu, 0x00003d31u } },
    { "sweep_36", UINT64_C(0xc892acebb9000000), 60, 60, 28, 0, 0, 1, 0, 0, 0, 1, 6, { 0x95675dc8u, 0x0e79e644u, 0x00003420u } },
    { "sweep_37", UINT64_C(0x520d754156c00000), 35, 43, 45, 44, 0, 1, 1, 0, 0, 1, 4, { 0x6baa0ab6u, 0x96d71a90u, 0x00002465u } },
    { "sweep_38", UINT64_C(0xadf72ca70c800000), 44, 19, 39, 38, 1, 0, 0, 0, 1, 1, 6, { 0xb9653864u, 0xd3a7656fu, 0x0000360cu } },
    { "sweep_39", UINT64_C(0xaadd2e9068000000), 45, 57, 62, 37, 1, 0, 3, 1, 1, 1, 5, { 0xe9748340u, 0xbf736d56u, 0x00002fccu } },
    { "sweep_40", UINT64_C(0x870c25cbb5600000), 13, 33, 46, 10, 1, 0, 1, 0, 1, 1, 0, { 0x612e5dabu, 0x57426c38u, 0x00000649u } },
    { "sweep_41", UINT64_C(0xe69439a37b400000), 29, 48, 25, 63, 3, 0, 1, 0, 1, 1, 6, { 0xa1cd1bdau, 0xece0ef34u, 0x0000365fu } },
    { "sweep_42", UINT64_C(0xefd91bf470000000), 23, 57, 52, 40, 3, 0, 2, 1, 0, 1, 7, { 0xc8dfa380u, 0x1a72bf7eu, 0x00003d9du } },
    { "sweep_43", UINT64_C(0x5fa14b3357a00000), 1, 47, 23, 19, 3, 0, 2, 0, 0, 1, 6, { 0x0a599abdu, 0x6bde0afdu, 0x0000349au } },
    { "sweep_44", UINT64_C(0x74103266dd000000), 28, 5, 62, 34, 1, 1, 2, 0, 0, 0, 6, { 0x819336e8u, 0x5f0ae3a0u, 0x000030acu } },
    { "sweep_45", UINT64_C(0x7ac67acabcc00000), 30, 19, 40, 30, 1, 0, 1, 1, 0, 1, 2, { 0x33d655e6u, 0xd426f3d6u, 0x0000154bu } },
    { "sweep_46", UINT64_C(0x305b1c1c7ba00000), 35, 45, 0, 34, 3, 1, 2, 0, 0, 0, 0, { 0xd8e0e3ddu, 0x405b1982u, 0x000000bcu } },
    { "sweep_47", UINT64_C(0xeed3d5cf67c00000), 19, 50, 13, 26, 1, 1, 1, 1, 0, 0, 1, { 0x9eae7b3eu, 0x46e49f76u, 0x0000096bu } },
    { "sweep_48", UINT64_C(0x0c15a696c7600000), 32, 8, 31, 43, 0, 1, 2, 1, 1, 0, 7, { 0xad34b63bu, 0x6f910060u, 0x00003ba5u } },
    { "sweep_49", UINT64_C(0xe45c38c2daa00000), 38, 56, 34, 54, 0, 1, 2, 0, 1, 1, 3, { 0xe1c616d5u, 0xd1713722u, 0x00001ea6u } },
    { "sweep_50", UINT64_C(0x203058e8bb200000), 13, 58, 36, 35, 1, 0, 0, 0, 1, 0, 3, { 0x82c745d9u, 0x72746901u, 0x00001a0cu } },
    { "sweep_51", UINT64_C(0x0bc37daf29000000), 46, 32, 31, 18, 0, 0, 1, 1, 1, 0, 1, { 0x1bed7948u, 0x4fc1705eu, 0x00000b42u } },
    { "sweep_52", UINT64_C(0xfb8a9283f6600000), 40, 40, 32, 20, 0, 1, 1, 0, 1, 1, 5, { 0x54941fb3u, 0x905147dcu, 0x00002e62u } },
    { "sweep_53", UINT64_C(0xf053c57808600000), 29, 33, 61, 33, 2, 1, 1, 0, 1, 1, 6, { 0x9e2bc043u, 0x3ec2ef82u, 0x00003674u } },
    { "sweep_54", UINT64_C(0x31b661bbc8600000), 9, 47, 43, 13, 1, 0, 2, 0, 0, 1, 7, { 0xb30dde43u, 0xb5de498du, 0x00003c89u } },
    { "sweep_55", UINT64_C(0x132f3ccc64000000), 41, 63, 39, 10, 2, 0, 1, 0, 1, 0, 2, { 0x79e66320u, 0x53ff4899u, 0x00001251u } },
    { "sweep_56", UINT64_C(0x082214945de00000), 31, 56, 11, 36, 0, 0, 1, 0, 1, 1, 7, { 0x10a4a2efu, 0x85f0f841u, 0x00003e44u } },
    { "sweep_57", UINT64_C(0x946a00b8b5c00000), 23, 42, 9, 51, 2, 0, 1, 1, 0, 0, 1, { 0x5005c5aeu, 0x64d4bca3u, 0x00000956u } },
    { "sweep_58", UINT64_C(0xf6b7b35853800000), 8, 39, 19, 60, 3, 1, 2, 1, 1, 1, 0, { 0xbd9ac29cu, 0x89ce47b5u, 0x000007bfu } },
    { "sweep_59", UINT64_C(0xa5a6b65e61c00000), 35, 9, 25, 1, 0, 1, 3, 0, 1, 0, 4, { 0x35b2f30eu, 0x2c931d2du, 0x000022e0u } },
    { "sweep_60", UINT64_C(0xa232c792ec200000), 10, 52, 1, 48, 0, 1, 0, 1, 0, 0, 2, { 0x963c9761u, 0x00e85511u, 0x00001126u } },
    { "sweep_61", UINT64_C(0xce1a491856e00000), 15, 39, 43, 54, 2, 0, 3, 0, 0, 1, 6, { 0xd248c2b7u, 0xd5ce7e70u, 0x000034d6u } },
};

#endif /* FT_TT_TLB2M_REFERENCE_VECTORS_H */
