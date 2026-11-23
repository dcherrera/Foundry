/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_program_resolve.h
 * @brief Resolve a Tensix program by NAME into the images a serve loads — embedded by
 *   default, off disk only when a caller explicitly asks.
 *
 *   `ft_tt_runtime_firmware_load()` / `ft_tt_runtime_program_load()` take a DIRECTORY and
 *   are unchanged; they are still the right call for a test driving a directory it just
 *   built. These two take a NAME, and a name is the thing a shipping binary can hold. That
 *   is the whole of Phase 12's first item: a program root is a property of one machine, a
 *   program name is a property of the model.
 *
 * ---------------------------------------------------------------------------
 * THE OVERRIDE IS EXPLICIT IN BOTH DIRECTIONS
 * ---------------------------------------------------------------------------
 * `override_root` NULL or empty means EMBEDDED. Non-empty means
 * `<override_root>/<program_name>` off disk — and that is announced on stderr, because the
 * failure this whole item exists to kill is a run that silently used something other than
 * what the operator believed.
 *
 * **An override that cannot be satisfied is a REFUSAL, never a fall back to embedded.** A
 * developer who points at a directory to test a program they just built, and gets the
 * embedded one because they typed the path wrong, has measured the wrong binary and been
 * told nothing. That is the `rtriad_bfp8` incident with the arrow reversed — the same
 * 1.085x nobody would have noticed.
 *
 * ---------------------------------------------------------------------------
 * THE FIRMWARE COMES FROM THE PROGRAM, NOT FROM A SECOND DECISION
 * ---------------------------------------------------------------------------
 * A kernel's text is linked at its own firmware's `__fw_export_text_end`, so the two are
 * one build output. `ft_tt_program_open_firmware()` therefore takes the same program name
 * and the same override as `ft_tt_program_open_kernels()`, and on the embedded path it
 * reaches the firmware THROUGH the program entry — there is no call that can pair one
 * program's kernels with another's firmware.
 *
 * @status Implemented, gated off-device by tests/test_program_blob.c, and IN USE — Pantheon's
 *   TT backend resolves every program through these two calls (pn_tt_device.c,
 *   pn_tt_residency.c, pn_tt_oncard.c). No program path remains in that backend.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_PROGRAM_RESOLVE_H
#define FT_TT_PROGRAM_RESOLVE_H

#include <stdint.h>

#include "ft_tt_runtime.h"

/**
 * The five `fw_*.elf` images `ft_tt_runtime_boot_core()` wants, for one program.
 *
 * @param program_name   the directory basename, e.g. "rtriad_bfp8"
 * @param override_root  NULL or "" for the embedded blob; otherwise a directory whose
 *                       `<override_root>/<program_name>` is read, announced, and REFUSED
 *                       rather than fallen back on if it cannot be read
 * @param out_images     zeroed and filled on success; `source` records which arm ran
 * @return               FT_TT_OK; FT_TT_ENOTFOUND when the blob holds no such program;
 *                       FT_TT_EINVAL when an embedded image fails its integrity check;
 *                       FT_TT_EIO / FT_TT_ENOMEM from the directory arm
 *
 * Ownership: release with `ft_tt_runtime_images_release()` on BOTH arms. It frees only
 * what it owns, so releasing an embedded set is correct and costs nothing.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs)
 */
int ft_tt_program_open_firmware(const char *program_name, const char *override_root,
                                ft_tt_runtime_images *out_images);

/**
 * The kernel images for every processor in `processor_mask`, for one program.
 *
 * A mask naming a processor the program carries no image for is a REFUSAL. That is not
 * pedantry: `enables` is derived from the mask, so a silently-dropped image is a core the
 * launch message starts and that has nothing to run.
 *
 * @param processor_mask  bit FT_TT_PROC_* per image wanted; 0 is invalid
 * @return                as above, plus FT_TT_ENOTFOUND when the mask asks for a processor
 *                        this program leaves idle
 */
int ft_tt_program_open_kernels(const char *program_name, const char *override_root,
                               uint32_t processor_mask, ft_tt_runtime_images *out_images);

/** "embedded", "directory" or "none". Never NULL. */
const char *ft_tt_program_source_string(ft_tt_program_source source);

/**
 * The `-D`s one image of one program was BUILT WITH, out of its own provenance record.
 *
 * ---------------------------------------------------------------------------
 * WHY A RUNTIME READ OF PROVENANCE IS NOT THE CHECK THAT COULD ONLY PASS
 * ---------------------------------------------------------------------------
 * `ft_tt_program_blob.h` says of the `provenance` field that "NOTHING AT RUN TIME READS
 * THIS", and it is right about the question it was answering: a binary has no path to a
 * source tree, so a STALENESS check inside these accessors could only ever pass. This asks
 * something else — not "does this image match sources that are absent" but "what was this
 * image built WITH", which the record carries and which nothing else in the binary can say.
 *
 * IT EXISTS BECAUSE ONE HOST ASSUMPTION HAS NO OTHER EVIDENCE. The matmul's DST section is a
 * compile-time property of the armed kernel — four tiles at `DstSync::SyncHalf`, eight at
 * SyncFull — and being wrong in the permissive direction is SILENT: the pass writes over the
 * half the packer is reading and every kernel retires clean. The interlock that guarded it
 * keyed on "an override directory is set", which was sound only while the embedded blob was
 * SyncHalf; a SyncFull blob REVERSES the argument and makes a SyncHalf override the unsafe
 * case. No version of "which directory did you name" survives that reversal. The recorded
 * `defines=` does, because it travels with the IMAGE instead of with the operator.
 *
 * AN ABSENT `defines=` IS A REFUSAL, NOT AN EMPTY STRING. `format_recipe()` writes every
 * recipe key even when unset, so a record with no `defines=` at all predates the field and
 * cannot answer the question — reporting it as "" would read as "definitely not SyncFull",
 * which is the permissive direction of exactly the mistake this call exists to stop.
 *
 * @param program_name   the directory basename, e.g. "rtriad_bfp8"
 * @param override_root  NULL or "" for the embedded blob; otherwise
 *                       `<root>/<program>/<image>.prov` is read
 * @param processor      FT_TT_PROC_*; the sync mode lives in the COMPUTE images
 * @param out            NUL-terminated on success, "" on every failure
 * @return FT_TT_OK; FT_TT_EINVAL for a bad argument or an `out` too small;
 *         FT_TT_ENOTFOUND when the program, the image or the record's `defines=` is absent
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
int ft_tt_program_recorded_defines(const char *program_name, const char *override_root,
                                   unsigned processor, char *out, size_t out_len);

/**
 * Does a recorded `defines` string contain `wanted` AS A WHOLE TOKEN?
 *
 * Whole token and not a substring, because `-DFT_TT_MATMUL_DST_SYNC=1` is a prefix of
 * `-DFT_TT_MATMUL_DST_SYNC=11` and a substring match is exactly how a check like this reads
 * green on something it does not mean. Tokens are separated by commas (how the build
 * records the list: `-DA=1,-DB=1`), spaces or tabs.
 *
 * @return 1 when present, 0 otherwise and for any NULL
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_program_defines_contain(const char *defines, const char *wanted);

#endif /* FT_TT_PROGRAM_RESOLVE_H */
