#!/usr/bin/env bash
#
# generate_tlb2m_reference.sh — offline generator for ft_tt_tlb2m_reference_vectors.h,
# the oracle for ft_tt_udrv_tlb_encode.c.
#
# The words come from generate_tlb2m_reference.c, which applies the PUBLIC 2 MiB TLB field
# table (tt-isa-documentation, BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md) one bit at a
# time — a different method from the encoder's per-word shifts and masks. No driver source is
# read or compiled. The generated header is committed; the binary goes to ../build-tlbref/
# (gitignored).
#
# Usage:
#   tests/reference/generate_tlb2m_reference.sh > tests/reference/ft_tt_tlb2m_reference_vectors.h
#
# Status: clean-room rewrite 2026-10-08; reproduces the previously committed vector data exactly.
# Issues: None
# Todo:   None

set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD="$HERE/../build-tlbref"
CC_BIN="${CC:-cc}"

mkdir -p "$BUILD"
"$CC_BIN" -std=c99 -O2 -Wall -Wextra -Werror -pedantic \
    -o "$BUILD/generate_tlb2m_reference" "$HERE/generate_tlb2m_reference.c"

cat <<'HEADER_EOF'
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

HEADER_EOF

"$BUILD/generate_tlb2m_reference"

cat <<'FOOTER_EOF'

#endif /* FT_TT_TLB2M_REFERENCE_VECTORS_H */
FOOTER_EOF
