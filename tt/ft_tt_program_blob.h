/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_program_blob.h
 * @brief The format of the embedded Tensix PROGRAM blob, and the accessors over it.
 *
 *   `ft_tt_kernel_blob.h` embeds single kernels so a Foundry build needs no toolchain.
 *   This is the same arrangement one level up: a PROGRAM is a whole
 *   `build_tensix_program.sh` output directory — five kernel ELFs (`brisc`, `ncrisc`,
 *   `trisc0/1/2`) plus the five `fw_*.elf` firmware images they were linked against — and
 *   this header is how a binary carries those directories instead of reading them off
 *   somebody's disk at start-up.
 *
 * ---------------------------------------------------------------------------
 * WHY THE BLOB CARRIES THE FIRMWARE TOO — the question, and the answer
 * ---------------------------------------------------------------------------
 * A kernel's `.text` is linked at its own firmware's `__fw_export_text_end`, so a kernel
 * and the firmware it runs under are ONE build output; mixing two directories is a wedged
 * core, not a configuration (`ft_tt_runtime.h`, rule 1). That makes "does the blob carry
 * the firmware" a real question rather than an obvious yes, and it is answered YES for
 * two reasons that point the same way:
 *
 *   1. **A serve boots firmware from the program directory.** `ft_tt_runtime_boot_core()`
 *      takes a firmware set, and Pantheon reads it out of the same directory the kernels
 *      come from. Embed only the kernels and the hardcoded path survives for the firmware
 *      — Phase 12's first item would be half done and the machine would still be one
 *      machine.
 *   2. **Embedding one half and reading the other off disk makes the pairing a runtime
 *      coincidence.** With both halves in one blob entry the pairing is structural: there
 *      is no call that can name a program and get somebody else's firmware, because the
 *      firmware is reached THROUGH the program.
 *
 * The obvious objection is size — every program directory carries its own copy of the
 * same five firmware images — and the generator answers it by CONTENT-DEDUPLICATING.
 * Programs built from one tree share one `ft_tt_program_blob_firmware`, stored once, and
 * two programs' firmware being identical is then a pointer comparison. The bytes are still
 * there, so it stays checkable BY BYTES and never only by the shared identity.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS FORMAT DOES NOT DO
 * ---------------------------------------------------------------------------
 * It does not check the SOURCES. A `.src` sidecar naming a kernel that has since changed
 * or vanished is refused by the GENERATOR, at emit time, where the source tree exists —
 * `tools/check_tensix_program_sources.sh` is the check and the generator runs it rather
 * than reimplementing it. A binary has no path to a source tree, so a stale-source check
 * inside these accessors would be a check that can only ever pass.
 *
 * The blob DOES carry each kernel image's `provenance` record, which is the same fact
 * stated by CONTENT rather than by mtime, so that a tree with no program directories can
 * still hold the shipping blob against its own sources
 * (`tools/check_tensix_program_provenance.py --header`). That check runs beside the
 * source tree, never inside these accessors, for the same reason.
 *
 * What the accessors DO check is that the bytes are the bytes: every image carries an
 * FNV-1a-64 digest that the GENERATOR computed over the file on disk, and
 * `ft_tt_program_blob_verify()` recomputes it over the embedded array. A hex emission that
 * drops, reorders or truncates a byte fails that comparison, because the two sides never
 * saw the same data — one read the file, the other reads the header.
 *
 * @status Format defined, implemented, and gated off-device by tests/test_program_blob.c
 *   against locally-constructed fixtures. `ft_tt_programs.h` is REAL as of 2026-08-17
 *   (df85613): 19 programs, firmware included, byte-verified against the on-disk
 *   directories. Pantheon serves from it.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_PROGRAM_BLOB_H
#define FT_TT_PROGRAM_BLOB_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_memmap.h"

/** One embedded ELF, plus what proves it arrived intact and what it was built from. */
typedef struct ft_tt_program_blob_image {
    const char *source_name;       /**< the kernel .c its `.src` sidecar named */
    unsigned processor;            /**< FT_TT_PROC_DM0 .. FT_TT_PROC_MATH2 */
    const unsigned char *elf_data; /**< the complete ELF file */
    size_t elf_size;
    uint64_t elf_digest;           /**< FNV-1a-64, computed by the generator over the FILE */
    /**
     * The image's `.prov` record: the recipe that reached code generation, and a content
     * hash of every in-tree file that compile opened
     * (foundry/tt/tools/tensix_image_provenance.py).
     *
     * NULL for FIRMWARE: the firmware build (tt/fw/) writes no `.prov` record. Its
     * sources are in this tree; the clean-room byte gate covers them, not this field.
     *
     * NOTHING AT RUN TIME READS THIS. A binary has no path to a source tree, so a check
     * here could only ever pass. It rides in the blob so that
     * `tools/check_tensix_program_provenance.py --header` can hold the SHIPPING blob
     * against the sources beside it on any machine — no program directories, no toolchain,
     * no card — and so `explode_tensix_program_blob` can write it back out and keep the
     * explode / rebuild-one / re-emit round trip checkable instead of laundering it.
     */
    const char *provenance;
} ft_tt_program_blob_image;

/**
 * The five firmware images one build produced, indexed by FT_TT_PROC_*.
 *
 * Deduplicated across programs: `set_id` is the generator's digest of all five, and two
 * programs from one tree point at the SAME object. Never treat that shared pointer as the
 * proof — `ft_tt_program_blob_firmware_matches()` compares bytes.
 */
typedef struct ft_tt_program_blob_firmware {
    const char *set_id;
    ft_tt_program_blob_image images[FT_TT_PROC_COUNT];
} ft_tt_program_blob_firmware;

/** One embedded program directory. */
typedef struct ft_tt_program_blob_entry {
    const char *name;         /**< the directory's basename, e.g. "rtriad_bfp8" */
    const char *toolchain;    /**< SFPI version the images were built with */
    const char *build_recipe; /**< the build_tensix_program.sh invocation, for logs */
    const ft_tt_program_blob_firmware *firmware;
    const ft_tt_program_blob_image *images; /**< kernel images, one per processor present */
    unsigned image_count;
} ft_tt_program_blob_entry;

/**
 * How many programs this binary carries.
 *
 * ZERO IS A MEANINGFUL ANSWER AND IS NOT THE SAME AS "your name is not here": it means
 * `ft_tt_programs.h` was never regenerated from real program directories. Callers that
 * report a lookup failure should say which of the two it was — the difference is
 * "rebuild the blob" against "you asked for a program nobody built".
 */
unsigned ft_tt_program_blob_count(void);

/** The `index`-th program, or NULL when `index` is out of range. Enumeration order is the
 *  order the generator was given, which is sorted by name. */
const ft_tt_program_blob_entry *ft_tt_program_blob_entry_at(unsigned index);

/** Which generated header this binary compiled in. Normally `"ft_tt_programs.h"`; a build
 *  that overrode `FT_TT_PROGRAMS_HEADER` is carrying a TEST FIXTURE and this is how a log
 *  can say so instead of quietly presenting fixtures as the shipping blob. */
const char *ft_tt_program_blob_generated_header(void);

/**
 * Find a program by name.
 *
 * REFUSES rather than returning an empty program: `*out_entry` is left untouched on
 * failure, so a caller that ignores the return code gets whatever it initialised the
 * pointer to and not a zero-image program that loads cleanly and computes nothing.
 *
 * @param name       the directory basename the generator recorded
 * @param out_entry  set only on success
 * @return           FT_TT_OK; FT_TT_EINVAL for a NULL argument; FT_TT_ENOTFOUND when the
 *                   blob holds no such name (the reason, including the empty-blob case,
 *                   is logged with the names that ARE present)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_program_blob_find(const char *name, const ft_tt_program_blob_entry **out_entry);

/**
 * One program's kernel image for one processor.
 *
 * @return FT_TT_OK; FT_TT_EINVAL on a bad argument; FT_TT_ENOTFOUND when this program leaves
 *         that processor idle. A program that carries no image for a processor the caller
 *         is about to enable is a refusal, not an empty load.
 */
int ft_tt_program_blob_kernel_image(const ft_tt_program_blob_entry *entry, unsigned processor,
                                    const ft_tt_program_blob_image **out_image);

/** The `enables` bitmask a program implies: bit `processor` set for every kernel image it
 *  carries. 0 for NULL. */
uint32_t ft_tt_program_blob_enables(const ft_tt_program_blob_entry *entry);

/**
 * Recompute every image's digest and re-parse every image as an ELF.
 *
 * This is the integrity gate, and it is deliberately NOT the generator's own arithmetic
 * replayed: the digests were computed over the FILES ON DISK by the Python emitter and are
 * recomputed here over the EMBEDDED ARRAYS by C, so a hex emission that lost, duplicated
 * or reordered a byte fails. The ELF re-parse is `ft_tt_elf_parse()`, an independently
 * written reader, and catches a blob that is intact-but-not-loadable.
 *
 * Costs one pass over the program's bytes. Call it once at load, not per dispatch.
 *
 * @return FT_TT_OK; FT_TT_EINVAL with the failing image named
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
int ft_tt_program_blob_verify(const ft_tt_program_blob_entry *entry);

/** Do two programs carry byte-identical firmware? Compares BYTES, never the deduplicated
 *  pointer, because the pointer is the generator's claim and the bytes are the fact.
 *  @return 1 when every one of the five images matches, 0 otherwise (and 0 for NULL) */
int ft_tt_program_blob_firmware_matches(const ft_tt_program_blob_entry *a,
                                        const ft_tt_program_blob_entry *b);

/** FNV-1a-64 over `size` bytes. Exposed so a test can compute the expected digest of a
 *  file it read itself, instead of trusting the one in the header. */
uint64_t ft_tt_program_blob_digest(const void *data, size_t size);

#endif /* FT_TT_PROGRAM_BLOB_H */
