/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_program_blob.c
 * @brief Off-device gate for the embedded Tensix PROGRAM blob: the format, the lookup, the
 *   integrity checks, the firmware dedup, and the embedded-vs-override resolution.
 *
 * ---------------------------------------------------------------------------
 * THE CHECK THAT MATTERS IS THE BYTE COMPARISON, AND IT IS NOT A SELF-CHECK
 * ---------------------------------------------------------------------------
 * The generator emitted this blob from files on disk. This test opens THE SAME FILES with
 * fread() and memcmp()s them against the embedded arrays. Neither side learned anything
 * from the other: the Python wrote hex, the C reads a file. That is what makes "the
 * embedded bytes are the on-disk bytes" a measurement rather than a restatement of how the
 * header was built.
 *
 * The digests are the second, independent line: the generator computed them over the FILE,
 * `ft_tt_program_blob_verify()` recomputes them over the EMBEDDED ARRAY, and this test
 * computes them a third time from bytes it read itself.
 *
 * ---------------------------------------------------------------------------
 * THE HARNESS PROVES IT IS LOOKING AT FIXTURES BEFORE IT REPORTS ANYTHING
 * ---------------------------------------------------------------------------
 * `ft_tt_program_blob_generated_header()` must name the fixture header. Compiled against
 * the shipping (empty) blob every one of these checks would be over zero programs, and
 * "no program failed" would be true and worthless. Four vacuous gates have been caught in
 * this project; this is the guard against a fifth.
 *
 * Run: ./build-programblob/test_program_blob <fixture-root>
 * The fixture root is what tests/make_program_blob_fixtures.py wrote.
 *
 * @status Green on the Mac against fixtures; every check in it has been seen RED under the
 *   mutation battery in tests/run_program_blob_gates.sh. Never run against real SFPI
 *   program directories — they are on the card host.
 * @issues None
 * @todo None
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ft_tt_device.h"
#include "ft_tt_memmap.h"
#include "ft_tt_program_blob.h"
#include "ft_tt_program_resolve.h"
#include "ft_tt_test_report.h"

/** The generated header this test insists it was compiled against. */
#define EXPECTED_FIXTURE_HEADER "ft_tt_programs_fixture.h"

#define ALL_PROCESSORS 0x1fu
#define DM0_DM1        0x03u
#define MAX_FILE_BYTES (1u << 20)

static const char *const k_kernel_file_names[FT_TT_PROC_COUNT] = {
    "brisc.elf", "ncrisc.elf", "trisc0.elf", "trisc1.elf", "trisc2.elf"
};
static const char *const k_firmware_file_names[FT_TT_PROC_COUNT] = {
    "fw_brisc.elf", "fw_ncrisc.elf", "fw_trisc0.elf", "fw_trisc1.elf", "fw_trisc2.elf"
};

static char g_programs_root[512];

/* ================================================================
 * Reading the fixture files, independently of everything under test
 * ================================================================ */

/** Slurp one fixture file. Returns NULL and reports on any failure — a file this test
 *  cannot read must FAIL a check, never quietly skip one. */
static unsigned char *read_fixture_file(const char *program, const char *file_name,
                                        size_t *out_size)
{
    char path[768];
    unsigned char *buffer;
    FILE *file;
    long size;

    *out_size = 0;
    if (snprintf(path, sizeof(path), "%s/%s/%s", g_programs_root, program, file_name) < 0)
        return NULL;
    file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "cannot open fixture %s\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) <= 0 ||
        (size_t)size > MAX_FILE_BYTES || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    buffer = malloc((size_t)size);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *out_size = (size_t)size;
    return buffer;
}

/** One embedded image against the file it came from: length, bytes, and the digest
 *  recomputed from the FILE rather than from the array. */
static void check_image_against_file(const char *program, const char *file_name,
                                     const ft_tt_program_blob_image *image)
{
    size_t size = 0;
    unsigned char *disk = read_fixture_file(program, file_name, &size);
    char label[192];

    (void)snprintf(label, sizeof(label), "%s/%s embedded == on disk", program, file_name);
    if (!disk) {
        ft_tt_test_check(label, false, "the fixture file could not be read at all");
        return;
    }
    if (size != image->elf_size) {
        ft_tt_test_check_fmt(label, false, "embedded %lu B, file %lu B",
                             (unsigned long)image->elf_size, (unsigned long)size);
    } else if (memcmp(disk, image->elf_data, size) != 0) {
        ft_tt_test_check_fmt(label, false, "%lu bytes, and they differ", (unsigned long)size);
    } else if (ft_tt_program_blob_digest(disk, size) != image->elf_digest) {
        ft_tt_test_check_fmt(label, false, "bytes equal but the recorded digest is wrong "
                                           "(file digests 0x%016llx, header says 0x%016llx)",
                             (unsigned long long)ft_tt_program_blob_digest(disk, size),
                             (unsigned long long)image->elf_digest);
    } else {
        ft_tt_test_check_fmt(label, true, "%lu bytes identical, digest 0x%016llx",
                             (unsigned long)size, (unsigned long long)image->elf_digest);
    }
    free(disk);
}

/* ================================================================
 * The checks
 * ================================================================ */

static void check_harness_is_looking_at_fixtures(void)
{
    const char *header = ft_tt_program_blob_generated_header();

    ft_tt_test_step("the harness is measuring the fixture blob, not the shipping one");
    ft_tt_test_check_fmt("compiled against " EXPECTED_FIXTURE_HEADER,
                         strcmp(header, EXPECTED_FIXTURE_HEADER) == 0, "header is %s", header);
    ft_tt_test_check_fmt("the blob holds the three fixture programs",
                         ft_tt_program_blob_count() == 3u, "count is %u",
                         ft_tt_program_blob_count());
}

static void check_lookup_refuses(void)
{
    const ft_tt_program_blob_entry *sentinel = (const ft_tt_program_blob_entry *)0x1;
    const ft_tt_program_blob_entry *entry = NULL;
    int rc;

    ft_tt_test_step("a name that is not embedded is REFUSED, not answered with nothing");
    rc = ft_tt_program_blob_find("prog_nope", &sentinel);
    ft_tt_test_check_fmt("find(\"prog_nope\") refuses", rc == FT_TT_ENOTFOUND, "rc %d (%s)",
                         rc, ft_tt_error_string(rc));
    ft_tt_test_check("a refused find leaves the caller's pointer alone",
                     sentinel == (const ft_tt_program_blob_entry *)0x1,
                     "the sentinel survived");
    ft_tt_test_check("find(NULL) refuses",
                     ft_tt_program_blob_find(NULL, &entry) == FT_TT_EINVAL, "EINVAL");
    ft_tt_test_check("find(\"\") refuses",
                     ft_tt_program_blob_find("", &entry) == FT_TT_EINVAL, "EINVAL");
    ft_tt_test_check("entry_at() past the end is NULL",
                     ft_tt_program_blob_entry_at(ft_tt_program_blob_count()) == NULL, "NULL");
    ft_tt_test_check("find(\"prog_alpha\") succeeds",
                     ft_tt_program_blob_find("prog_alpha", &entry) == FT_TT_OK && entry &&
                         strcmp(entry->name, "prog_alpha") == 0,
                     "found");
}

/** Every image of one program against the directory it was generated from. */
static void check_program_against_disk(const char *name)
{
    const ft_tt_program_blob_entry *entry = NULL;
    unsigned index;

    ft_tt_test_step("%s: embedded bytes against the files on disk", name);
    if (ft_tt_program_blob_find(name, &entry) != FT_TT_OK) {
        ft_tt_test_check_fmt("the program is embedded", false, "%s is not in the blob", name);
        return;
    }
    ft_tt_test_check_fmt("verify() accepts it", ft_tt_program_blob_verify(entry) == FT_TT_OK,
                         "%u kernel image(s) + 5 firmware", entry->image_count);
    for (index = 0; index < entry->image_count; index++) {
        const ft_tt_program_blob_image *image = &entry->images[index];

        check_image_against_file(name, k_kernel_file_names[image->processor], image);
    }
    for (index = 0; index < FT_TT_PROC_COUNT; index++)
        check_image_against_file(name, k_firmware_file_names[index],
                                 &entry->firmware->images[index]);
}

static void check_firmware_dedup(void)
{
    const ft_tt_program_blob_entry *alpha = NULL;
    const ft_tt_program_blob_entry *beta = NULL;
    const ft_tt_program_blob_entry *gamma = NULL;

    ft_tt_test_step("firmware dedup shares what is identical and NOTHING else");
    if (ft_tt_program_blob_find("prog_alpha", &alpha) != FT_TT_OK ||
        ft_tt_program_blob_find("prog_beta", &beta) != FT_TT_OK ||
        ft_tt_program_blob_find("prog_gamma", &gamma) != FT_TT_OK) {
        ft_tt_test_check("all three fixture programs are present", false, "one is missing");
        return;
    }
    ft_tt_test_check("alpha and beta were built against ONE firmware set",
                     ft_tt_program_blob_firmware_matches(alpha, beta) == 1,
                     "bytes identical across all five");
    ft_tt_test_check("alpha and beta SHARE one stored copy",
                     alpha->firmware == beta->firmware, "same object");
    /* The mutation that matters: a dedup that merged everything would pass the line above
     * and fail this one. */
    ft_tt_test_check("alpha and gamma were NOT merged",
                     ft_tt_program_blob_firmware_matches(alpha, gamma) == 0 &&
                         alpha->firmware != gamma->firmware,
                     "different sets, different objects");
}

static void check_processor_masks(void)
{
    const ft_tt_program_blob_entry *alpha = NULL;
    const ft_tt_program_blob_entry *beta = NULL;
    const ft_tt_program_blob_image *image = NULL;

    ft_tt_test_step("a program is asked only for processors it carries");
    if (ft_tt_program_blob_find("prog_alpha", &alpha) != FT_TT_OK ||
        ft_tt_program_blob_find("prog_beta", &beta) != FT_TT_OK)
        return;
    ft_tt_test_check_fmt("alpha enables all five", ft_tt_program_blob_enables(alpha) == 0x1fu,
                         "0x%02x", (unsigned)ft_tt_program_blob_enables(alpha));
    ft_tt_test_check_fmt("beta enables DM0|DM1 only",
                         ft_tt_program_blob_enables(beta) == 0x03u, "0x%02x",
                         (unsigned)ft_tt_program_blob_enables(beta));
    ft_tt_test_check("beta refuses MATH0",
                     ft_tt_program_blob_kernel_image(beta, FT_TT_PROC_MATH0, &image) ==
                         FT_TT_ENOTFOUND,
                     "ENOTFOUND, not an empty image");
    ft_tt_test_check("alpha hands back MATH2",
                     ft_tt_program_blob_kernel_image(alpha, FT_TT_PROC_MATH2, &image) ==
                             FT_TT_OK &&
                         image && image->processor == FT_TT_PROC_MATH2,
                     "found");
    ft_tt_test_check("an out-of-range processor is refused",
                     ft_tt_program_blob_kernel_image(alpha, FT_TT_PROC_COUNT, &image) ==
                         FT_TT_EINVAL,
                     "EINVAL");
}

/** Compare a loaded image set against the fixture files, whichever arm produced it. */
static void check_loaded_set_against_disk(const char *label, const char *program,
                                          const ft_tt_runtime_images *images,
                                          const char *const *file_names, uint32_t mask)
{
    unsigned processor;
    unsigned compared = 0;
    int equal = 1;

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        size_t size = 0;
        unsigned char *disk;

        if ((mask & (1u << processor)) == 0u)
            continue;
        disk = read_fixture_file(program, file_names[processor], &size);
        if (!disk || size == 0u || size != images->elf_size[processor] ||
            !images->elf_data[processor] ||
            memcmp(disk, images->elf_data[processor], size) != 0)
            equal = 0;
        else
            compared++;
        free(disk);
    }
    ft_tt_test_check_fmt(label, equal && compared != 0, "%u image(s) compared byte for byte",
                         compared);
}

static void check_embedded_resolution(void)
{
    ft_tt_runtime_images images;
    unsigned processor;
    int nothing_owned = 1;

    ft_tt_test_step("resolving by NAME with no override reads the embedded blob");
    memset(&images, 0, sizeof(images));
    ft_tt_test_check_fmt("open_kernels(prog_alpha, NULL) succeeds",
                         ft_tt_program_open_kernels("prog_alpha", NULL, ALL_PROCESSORS,
                                                    &images) == FT_TT_OK,
                         "source=%s", ft_tt_program_source_string(images.source));
    ft_tt_test_check("it says it came from the blob",
                     images.source == FT_TT_PROGRAM_SOURCE_EMBEDDED, "embedded");
    ft_tt_test_check_fmt("the program carries five images", images.program.image_count == 5u,
                         "%u", images.program.image_count);
    check_loaded_set_against_disk("embedded kernels == the fixture directory", "prog_alpha",
                                  &images, k_kernel_file_names, ALL_PROCESSORS);
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        if (images.owned_data[processor])
            nothing_owned = 0;
    }
    ft_tt_test_check("an embedded load owns no heap memory", nothing_owned,
                     "owned_data is NULL for all five");
    ft_tt_runtime_images_release(&images);
    ft_tt_runtime_images_release(&images); /* releasing twice must stay safe */
    ft_tt_test_check("release() clears it and is safe twice",
                     images.elf_data[0] == NULL && images.program.image_count == 0u,
                     "cleared");

    memset(&images, 0, sizeof(images));
    ft_tt_test_check("open_firmware(prog_alpha, NULL) succeeds",
                     ft_tt_program_open_firmware("prog_alpha", NULL, &images) == FT_TT_OK,
                     "five firmware images");
    check_loaded_set_against_disk("embedded firmware == the fixture directory", "prog_alpha",
                                  &images, k_firmware_file_names, ALL_PROCESSORS);
    ft_tt_runtime_images_release(&images);
}

static void check_embedded_refusals(void)
{
    ft_tt_runtime_images images;

    ft_tt_test_step("the embedded arm refuses rather than loading something else");
    memset(&images, 0, sizeof(images));
    ft_tt_test_check("a mask naming a processor beta lacks is refused",
                     ft_tt_program_open_kernels("prog_beta", NULL, ALL_PROCESSORS, &images) !=
                         FT_TT_OK,
                     "refused, not silently 2 of 5");
    ft_tt_test_check("and nothing was handed back", images.program.image_count == 0u &&
                                                        images.elf_data[0] == NULL,
                     "empty");
    ft_tt_test_check("beta's own mask still works",
                     ft_tt_program_open_kernels("prog_beta", NULL, DM0_DM1, &images) ==
                         FT_TT_OK,
                     "DM0|DM1 loaded");
    ft_tt_runtime_images_release(&images);
    memset(&images, 0, sizeof(images));
    ft_tt_test_check("an unknown name is refused",
                     ft_tt_program_open_kernels("prog_nope", NULL, ALL_PROCESSORS, &images) ==
                         FT_TT_ENOTFOUND,
                     "ENOTFOUND");
    ft_tt_test_check("an empty mask is refused",
                     ft_tt_program_open_kernels("prog_alpha", NULL, 0u, &images) ==
                         FT_TT_EINVAL,
                     "EINVAL");
}

static void check_override(void)
{
    ft_tt_runtime_images images;

    ft_tt_test_step("the filesystem override is taken when asked, and NEVER falls back");
    memset(&images, 0, sizeof(images));
    ft_tt_test_check_fmt("open_kernels with an override root succeeds",
                         ft_tt_program_open_kernels("prog_alpha", g_programs_root,
                                                    ALL_PROCESSORS, &images) == FT_TT_OK,
                         "root=%s", g_programs_root);
    ft_tt_test_check("it says it came from a directory",
                     images.source == FT_TT_PROGRAM_SOURCE_DIRECTORY, "directory");
    ft_tt_test_check("an override load owns its buffers",
                     images.owned_data[FT_TT_PROC_DM0] != NULL, "owned_data is set");
    check_loaded_set_against_disk("override kernels == the fixture directory", "prog_alpha",
                                  &images, k_kernel_file_names, ALL_PROCESSORS);
    ft_tt_runtime_images_release(&images);

    memset(&images, 0, sizeof(images));
    ft_tt_test_check("an override root that cannot be read is REFUSED",
                     ft_tt_program_open_kernels("prog_alpha", "/nonexistent/program/root",
                                                ALL_PROCESSORS, &images) != FT_TT_OK,
                     "refused");
    /* The whole point: a bad override must not quietly become the embedded program, or a
     * developer testing a freshly built directory measures the blob and is told nothing. */
    ft_tt_test_check("and it did NOT silently fall back to the embedded program",
                     images.source != FT_TT_PROGRAM_SOURCE_EMBEDDED &&
                         images.elf_data[FT_TT_PROC_DM0] == NULL,
                     "nothing loaded");
    memset(&images, 0, sizeof(images));
    ft_tt_test_check("open_firmware with a bad override is refused too",
                     ft_tt_program_open_firmware("prog_alpha", "/nonexistent/program/root",
                                                 &images) != FT_TT_OK &&
                         images.source != FT_TT_PROGRAM_SOURCE_EMBEDDED,
                     "refused");
}

static void check_digest_discriminates(void)
{
    static const unsigned char left[] = {'a', 'b', 'c'};
    static const unsigned char right[] = {'a', 'b', 'd'};

    ft_tt_test_step("the digest can tell two things apart at all");
    ft_tt_test_check("one changed byte changes the digest",
                     ft_tt_program_blob_digest(left, sizeof(left)) !=
                         ft_tt_program_blob_digest(right, sizeof(right)),
                     "differs");
    ft_tt_test_check("a shorter run of the same bytes changes it",
                     ft_tt_program_blob_digest(left, sizeof(left)) !=
                         ft_tt_program_blob_digest(left, sizeof(left) - 1u),
                     "differs");
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <fixture-root>\n", argv[0]);
        return 2;
    }
    if (snprintf(g_programs_root, sizeof(g_programs_root), "%s/programs", argv[1]) < 0)
        return 2;

    check_harness_is_looking_at_fixtures();
    check_lookup_refuses();
    check_program_against_disk("prog_alpha");
    check_program_against_disk("prog_beta");
    check_program_against_disk("prog_gamma");
    check_firmware_dedup();
    check_processor_masks();
    check_embedded_resolution();
    check_embedded_refusals();
    check_override();
    check_digest_discriminates();
    return ft_tt_test_finish();
}
