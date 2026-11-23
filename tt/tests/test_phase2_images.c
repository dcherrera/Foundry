/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * test_phase2_images.c — checklist steps 5 and 6 from docs/tt_launch_notes.md §6.
 *
 * Both steps are PURE READS of a core held in reset. Nothing here starts a processor,
 * so it cannot hang anything; it exists to check the one thing no later step can
 * diagnose after the fact.
 *
 * Step 5 — read the loaded images back. The firmware-vs-kernel `.data` split is the
 * silent failure mode of this phase: a firmware image's `.data` must be staged at
 * MEM_*_INIT_LOCAL_L1_BASE_SCRATCH (do_crt1 reads it from there), while a KERNEL's
 * `.data` goes at the ELF's own p_paddr (its `_start` reads __kernel_data_lma). Put one
 * in the other's place and the core boots with garbage globals and reports success.
 * The expected destination is re-derived here from the rule rather than shared with
 * ft_tt_launch.c, so a wrong rule cannot agree with itself.
 *
 * Step 6 — record where the kernels actually landed. docs/tt_launch_notes.md §2 INFERS
 * that a kernel's text runs past its processor's firmware hole into the region the L1
 * table calls free. This measures it, and checks nothing reached the address the tests
 * are about to allocate buffers at.
 *
 * Build:
 *   cc -std=c99 -O1 -Wall -Wextra -Werror -DFT_TT_PHASE1B_NOC_ENABLED -I.. \
 *      ../ft_tt_kmd.c ../ft_tt_device.c ../ft_tt_tlb.c ../ft_tt_arc_telemetry.c \
 *      ../ft_tt_coord.c ../ft_tt_elf.c ../ft_tt_kernel_blob.c ../ft_tt_launch.c \
 *      ft_tt_test_report.c ft_tt_hardware_session.c test_phase2_images.c \
 *      -o build/test_phase2_images
 */

#include "ft_tt_hardware_session.h"
#include "ft_tt_kernels.h"
#include "ft_tt_phase2_l1_layout.h"
#include "ft_tt_test_report.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** Biggest segment we are willing to read back in one buffer. BRISC's firmware text is
 *  the largest image in play at 8704 bytes of hole. */
#define READBACK_BUFFER_BYTES (16u * 1024u)

/**
 * Where the host should have written one segment — re-derived from the rule in
 * docs/tt_launch_notes.md §2 rather than borrowed from ft_tt_launch.c's static helper.
 * That independence is the point of this test.
 */
static uint32_t expected_destination(const ft_tt_elf_segment *segment, ft_tt_image_kind kind,
                                     unsigned processor)
{
    bool executes_from_local_memory =
        (segment->exec_address & FT_TT_MEM_LOCAL_BASE) == FT_TT_MEM_LOCAL_BASE;

    if (kind == FT_TT_IMAGE_FIRMWARE && executes_from_local_memory)
        return (segment->exec_address & ~(uint32_t)FT_TT_MEM_LOCAL_BASE) +
               ft_tt_processor_local_init_scratch(processor);
    return segment->load_address;
}

/** Read one image's segments back out of L1 and compare them byte for byte. */
static void verify_image_in_l1(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                               unsigned processor, ft_tt_image_kind kind, const char *label,
                               const void *elf_data, size_t elf_size)
{
    static uint8_t readback[READBACK_BUFFER_BYTES];
    ft_tt_elf_image image;
    unsigned index;
    int rc = ft_tt_elf_parse(elf_data, elf_size, &image);

    if (rc != FT_TT_OK) {
        ft_tt_test_check_fmt(label, false, "ELF parse failed: %s", ft_tt_error_string(rc));
        return;
    }

    for (index = 0; index < image.segment_count; index++) {
        const ft_tt_elf_segment *segment = &image.segments[index];
        uint32_t destination = expected_destination(segment, kind, processor);
        const char *segment_name = (index == 0u) ? "text" : "data";

        if (segment->file_bytes == 0u)
            continue;
        if (segment->file_bytes > sizeof(readback)) {
            ft_tt_test_check_fmt(label, false, "%s segment is %u bytes, buffer is %zu",
                                 segment_name, segment->file_bytes, sizeof(readback));
            continue;
        }
        rc = noc->read(noc->user, x, y, destination, readback, segment->file_bytes);
        if (rc != FT_TT_OK) {
            ft_tt_test_check_fmt(label, false, "read back of %s at 0x%x failed: %s", segment_name,
                                 destination, ft_tt_error_string(rc));
            continue;
        }
        ft_tt_test_check_fmt(label,
                             memcmp(readback, segment->file_data, segment->file_bytes) == 0,
                             "%s %u B at 0x%05x (vma 0x%08x)", segment_name, segment->file_bytes,
                             destination, segment->exec_address);
    }
}

/** Step 6: where each kernel image landed, and whether it stayed clear of the floor. */
static void report_placement(unsigned processor, const ft_tt_image_placement *place,
                             const ft_tt_phase2_l1_layout *layout)
{
    uint32_t firmware_base = ft_tt_processor_firmware_base(processor);
    uint32_t firmware_end = firmware_base + ft_tt_processor_firmware_size(processor);
    bool inside_hole = place->highest_l1_address <= firmware_end;

    if (place->bytes_written == 0u)
        return;

    ft_tt_test_check_fmt("kernel image is below the allocation floor",
                         place->highest_l1_address <= layout->allocation_floor,
                         "%-6s [0x%05x,0x%05x) floor 0x%05x", ft_tt_processor_name(processor),
                         place->lowest_l1_address, place->highest_l1_address,
                         layout->allocation_floor);
    /* Not a pass/fail: the notes INFER that a kernel's text runs past its firmware
     * hole, and this is the measurement of that inference either way. */
    printf("        %-6s entry 0x%05x, text %u B at 0x%05x, %u B written, %s its %u B "
           "firmware hole [0x%05x,0x%05x)\n",
           ft_tt_processor_name(processor), place->entry_address, place->text_bytes,
           place->text_address, place->bytes_written, inside_hole ? "inside" : "PAST",
           ft_tt_processor_firmware_size(processor), firmware_base, firmware_end);
}

int main(void)
{
    ft_tt_hardware_session session;
    ft_tt_firmware_set firmware;
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_image_placement firmware_placements[FT_TT_PROC_COUNT];
    ft_tt_phase2_l1_layout layout;
    uint16_t x = 0;
    uint16_t y = 0;
    unsigned processor;
    unsigned index;
    int rc;

    rc = ft_tt_hardware_session_open(&session);
    if (rc != FT_TT_OK)
        return 1;

    rc = ft_tt_hardware_session_tensix_from_environment(&session, &x, &y, NULL);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    rc = ft_tt_firmware_set_load(ft_tt_firmware_directory(), &firmware);
    if (rc != FT_TT_OK) {
        ft_tt_hardware_session_close(&session);
        return 1;
    }

    /* Held in reset for the whole test: nothing below starts a processor. */
    ft_tt_test_step("setup — hold all five RISC-Vs in reset, clear the mailbox, load images");
    rc = ft_tt_core_park(&session.noc, x, y);
    ft_tt_test_check_fmt("reset asserted on all five", rc == FT_TT_OK, "rc=%d", rc);
    rc = ft_tt_launch_clear_mailbox(&session.noc, x, y);
    ft_tt_test_check_fmt("mailbox cleared", rc == FT_TT_OK, "rc=%d", rc);

    for (processor = 0; processor < FT_TT_PROC_COUNT && ft_tt_test_failure_count() == 0;
         processor++) {
        rc = ft_tt_launch_load_image(&session.noc, x, y, processor, FT_TT_IMAGE_FIRMWARE,
                                     firmware.elf_data[processor], firmware.elf_size[processor],
                                     &firmware_placements[processor]);
        ft_tt_test_check_fmt("firmware image written", rc == FT_TT_OK, "%-6s %zu B",
                             ft_tt_processor_name(processor), firmware.elf_size[processor]);
    }

    rc = ft_tt_core_load_program(&session.noc, x, y, &ft_tt_kernel_vector_add, placements);
    ft_tt_test_check_fmt("vector_add program written", rc == FT_TT_OK, "%u images, rc=%d",
                         ft_tt_kernel_vector_add.image_count, rc);

    /* The allocation floor is DERIVED from the firmware placements above, not chosen. */
    rc = ft_tt_phase2_l1_layout_derive(firmware_placements, FT_TT_PROC_COUNT, &layout);
    ft_tt_test_check_fmt("L1 map derived from where the firmware landed", rc == FT_TT_OK,
                         "base 0x%05x (reserved-region end 0x%05x)", layout.kernel_config_base,
                         (unsigned)FT_TT_PHASE2_RESERVED_L1_END);

    if (ft_tt_test_failure_count() != 0)
        goto done;

    /* --- step 5: read every loaded segment back and compare ------------------ */
    ft_tt_test_step("step 5 — read the loaded images back out of L1");
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        char label[64];

        (void)snprintf(label, sizeof(label), "firmware %s matches in L1",
                       ft_tt_processor_name(processor));
        verify_image_in_l1(&session.noc, x, y, processor, FT_TT_IMAGE_FIRMWARE, label,
                           firmware.elf_data[processor], firmware.elf_size[processor]);
    }
    for (index = 0; index < ft_tt_kernel_vector_add.image_count; index++) {
        const ft_tt_kernel_image *image = &ft_tt_kernel_vector_add.images[index];
        char label[64];

        (void)snprintf(label, sizeof(label), "kernel %s matches in L1",
                       ft_tt_processor_name(image->processor));
        verify_image_in_l1(&session.noc, x, y, image->processor, FT_TT_IMAGE_KERNEL, label,
                           image->elf_data, image->elf_size);
    }

    /* --- step 6: where the kernels landed ----------------------------------- */
    ft_tt_test_step("step 6 — record where the kernel images landed");
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++)
        report_placement(processor, &placements[processor], &layout);

done:
    (void)ft_tt_core_park(&session.noc, x, y);
    ft_tt_firmware_set_release(&firmware);
    ft_tt_hardware_session_close(&session);
    return ft_tt_test_finish();
}
