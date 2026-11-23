/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_program_blob.c
 * @brief Implementation of the embedded-program-blob accessors (see ft_tt_program_blob.h).
 *
 *   This translation unit IS the registry: it is the one place `ft_tt_programs.h` is
 *   included, so every embedded byte in the binary is reachable from these four functions
 *   and nothing else. Keeping the generated header out of every other file is what makes
 *   the blob's cost one link decision instead of a property of whoever happened to
 *   `#include` it.
 *
 *   `FT_TT_PROGRAMS_HEADER` exists so a TEST can substitute a fixture blob. It is not a
 *   configuration knob for a serve — a build that overrides it is a build carrying
 *   somebody's fixtures, and `ft_tt_program_blob_generated_header()` reports which header
 *   was compiled in so a log can say so.
 *
 * @status Implemented and gated by tests/test_program_blob.c off-device. The blob it reads
 *   in a normal build is REAL as of 2026-08-17 — 19 programs, see ft_tt_programs.h.
 * @issues None
 * @todo None
 */

#include "ft_tt_program_blob.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_elf.h"

/** The generated blob. Overridable ONLY so a test can compile against a fixture. */
#ifndef FT_TT_PROGRAMS_HEADER
#define FT_TT_PROGRAMS_HEADER "ft_tt_programs.h"
#endif
#include FT_TT_PROGRAMS_HEADER

/** Longest list of names one refusal prints before it gives up and says how many more. */
#define FT_TT_PROGRAM_BLOB_NAMES_IN_REFUSAL 16u

/* ================================================================
 * The digest
 *
 * FNV-1a-64. Chosen because it is six lines in C and six lines in Python, so the two
 * implementations can be read against each other by eye — the point of the digest is to
 * catch a hex emission that lost or reordered bytes, and a cryptographic hash would only
 * make the two sides harder to compare while catching the same class.
 * ================================================================ */

uint64_t ft_tt_program_blob_digest(const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    uint64_t hash = 0xcbf29ce484222325ull;
    size_t index;

    if (!data)
        return 0u;
    for (index = 0; index < size; index++) {
        hash ^= (uint64_t)bytes[index];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

/* ================================================================
 * Lookup
 * ================================================================ */

unsigned ft_tt_program_blob_count(void)
{
    return FT_TT_PROGRAM_BLOB_TABLE_COUNT;
}

/* FT_TT_PROGRAM_BLOB_TABLE_COUNT is an unsigned literal from the generated header, and in the
 * EMPTY placeholder state it is 0u — so `index < COUNT` and `index >= COUNT` are tautologies
 * that gcc rejects under -Wextra -Werror=type-limits. clang does not raise them, which is why
 * the empty state compiled on the Mac and REFUSED TO BUILD on the card host, blocking the one tool
 * that verifies a real regeneration against real directories. A widening cast does not help:
 * gcc still sees the literal, and `#if` is not available either because the generated header
 * defines the count as an expression in the populated case, not a preprocessor constant. So
 * every site reads it into a local first: that is one value of one type, and the comparison
 * stops being a comparison against a literal. */
const ft_tt_program_blob_entry *ft_tt_program_blob_entry_at(unsigned index)
{
    const unsigned count = FT_TT_PROGRAM_BLOB_TABLE_COUNT;

    if (index >= count)
        return NULL;
    return ft_tt_program_blob_table[index];
}

const char *ft_tt_program_blob_generated_header(void)
{
    return FT_TT_PROGRAMS_HEADER;
}

/** Say what IS embedded, so a refusal is actionable rather than just negative. */
static void log_available_programs(void)
{
    unsigned index;

    const unsigned count = FT_TT_PROGRAM_BLOB_TABLE_COUNT;

    if (count == 0u) {
        ft_tt_log_error("tt program blob: this binary carries NO embedded programs — %s was "
                        "never regenerated from real program directories "
                        "(tools/regen_tensix_program_header.sh, on the bank)",
                        FT_TT_PROGRAMS_HEADER);
        return;
    }
    for (index = 0; index < count; index++) {
        if (index >= FT_TT_PROGRAM_BLOB_NAMES_IN_REFUSAL) {
            ft_tt_log_error("tt program blob:   ... and %u more",
                            FT_TT_PROGRAM_BLOB_TABLE_COUNT - index);
            return;
        }
        ft_tt_log_error("tt program blob:   embedded: %s", ft_tt_program_blob_table[index]->name);
    }
}

int ft_tt_program_blob_find(const char *name, const ft_tt_program_blob_entry **out_entry)
{
    const unsigned count = FT_TT_PROGRAM_BLOB_TABLE_COUNT;
    unsigned index;

    if (!name || !*name || !out_entry)
        return FT_TT_EINVAL;

    for (index = 0; index < count; index++) {
        if (strcmp(ft_tt_program_blob_table[index]->name, name) == 0) {
            *out_entry = ft_tt_program_blob_table[index];
            return FT_TT_OK;
        }
    }
    ft_tt_log_error("tt program blob: no embedded program named '%s'", name);
    log_available_programs();
    return FT_TT_ENOTFOUND;
}

int ft_tt_program_blob_kernel_image(const ft_tt_program_blob_entry *entry, unsigned processor,
                                    const ft_tt_program_blob_image **out_image)
{
    unsigned index;

    if (!entry || !out_image || processor >= FT_TT_PROC_COUNT)
        return FT_TT_EINVAL;

    for (index = 0; index < entry->image_count; index++) {
        if (entry->images[index].processor == processor) {
            *out_image = &entry->images[index];
            return FT_TT_OK;
        }
    }
    ft_tt_log_error("tt program blob: program '%s' carries no kernel for processor %u "
                    "(it enables 0x%02x)",
                    entry->name, processor, (unsigned)ft_tt_program_blob_enables(entry));
    return FT_TT_ENOTFOUND;
}

uint32_t ft_tt_program_blob_enables(const ft_tt_program_blob_entry *entry)
{
    uint32_t mask = 0u;
    unsigned index;

    if (!entry || !entry->images)
        return 0u;
    for (index = 0; index < entry->image_count; index++)
        mask |= 1u << entry->images[index].processor;
    return mask;
}

/* ================================================================
 * Integrity
 * ================================================================ */

/** One image: the digest the generator took of the FILE, then the ELF reader. */
static int verify_one_image(const char *program, const char *role,
                            const ft_tt_program_blob_image *image)
{
    ft_tt_elf_image parsed;
    uint64_t recomputed;

    if (!image->elf_data || image->elf_size == 0u) {
        ft_tt_log_error("tt program blob: %s/%s (proc %u) is EMPTY", program, role,
                        image->processor);
        return FT_TT_EINVAL;
    }
    recomputed = ft_tt_program_blob_digest(image->elf_data, image->elf_size);
    if (recomputed != image->elf_digest) {
        ft_tt_log_error("tt program blob: %s/%s (proc %u, %s, %lu B) is CORRUPT — the "
                        "embedded bytes digest 0x%016llx, the generator recorded 0x%016llx "
                        "for the file it read",
                        program, role, image->processor,
                        image->source_name ? image->source_name : "?",
                        (unsigned long)image->elf_size, (unsigned long long)recomputed,
                        (unsigned long long)image->elf_digest);
        return FT_TT_EINVAL;
    }
    if (ft_tt_elf_parse(image->elf_data, image->elf_size, &parsed) != FT_TT_OK) {
        ft_tt_log_error("tt program blob: %s/%s (proc %u) digests correctly but is not a "
                        "loadable Tensix ELF", program, role, image->processor);
        return FT_TT_EINVAL;
    }
    return FT_TT_OK;
}

int ft_tt_program_blob_verify(const ft_tt_program_blob_entry *entry)
{
    unsigned index;
    int rc;

    if (!entry || !entry->name || !entry->images || entry->image_count == 0u ||
        entry->image_count > FT_TT_PROC_COUNT || !entry->firmware)
        return FT_TT_EINVAL;

    for (index = 0; index < FT_TT_PROC_COUNT; index++) {
        rc = verify_one_image(entry->name, "firmware", &entry->firmware->images[index]);
        if (rc != FT_TT_OK)
            return rc;
    }
    for (index = 0; index < entry->image_count; index++) {
        rc = verify_one_image(entry->name, "kernel", &entry->images[index]);
        if (rc != FT_TT_OK)
            return rc;
    }
    return FT_TT_OK;
}

int ft_tt_program_blob_firmware_matches(const ft_tt_program_blob_entry *a,
                                        const ft_tt_program_blob_entry *b)
{
    unsigned index;

    if (!a || !b || !a->firmware || !b->firmware)
        return 0;
    for (index = 0; index < FT_TT_PROC_COUNT; index++) {
        const ft_tt_program_blob_image *left = &a->firmware->images[index];
        const ft_tt_program_blob_image *right = &b->firmware->images[index];

        if (left->elf_size != right->elf_size || !left->elf_data || !right->elf_data)
            return 0;
        if (memcmp(left->elf_data, right->elf_data, left->elf_size) != 0)
            return 0;
    }
    return 1;
}
