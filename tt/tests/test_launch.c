/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_launch.c
 * @brief Phase 2 off-device tests: ELF parsing and segment placement, the embedded
 *   blob format, launch-message and CB encoding, runtime-arg planting, reset
 *   read-modify-write, and the mailbox completion poll including its timeout.
 *
 * NO DEVICE IS TOUCHED. Every case runs against a simulated Tensix — 1.5 MiB of host
 * memory standing in for L1 plus one word standing in for the soft-reset register —
 * behind the same ft_tt_noc_access vtable the real TLB path implements. That is the
 * whole point of that indirection: this file exercises the exact bytes the card would
 * have received, on a machine where a mistake costs a failed assert instead of a
 * power cycle.
 *
 * What this CANNOT tell you: whether the transliterated mailbox layout matches the
 * firmware binary, or whether a real SFPI-built ELF parses. Both need the card and
 * its host. docs/tt_launch_notes.md §6 lists them in the order to check them.
 *
 * @status Green on macOS/arm64 under
 *   `cc -std=c99 -Wall -Wextra -Werror -pedantic -Wshadow -Wconversion`.
 *   Every case has been observed RED — the mutation table is docs/tt_launch_notes.md §8.
 * @issues None
 * @todo None
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ft_tt_arc_telemetry.h"
#include "ft_tt_arch.h"
#include "ft_tt_buffer.h"
#include "ft_tt_elf.h"
#include "ft_tt_kernel_blob.h"
#include "ft_tt_launch.h"
#include "ft_tt_mailbox.h"
#include "ft_tt_op_l1_layout.h"
#include "ft_tt_phase2_l1_layout.h"

/* ------------------------------------------------------------------------- */
/* Tiny harness                                                              */
/* ------------------------------------------------------------------------- */

static int checks_run;
static int checks_failed;

static void check(int condition, const char *what)
{
    checks_run++;
    if (condition) {
        printf("  ok    %s\n", what);
    } else {
        checks_failed++;
        printf("  FAIL  %s\n", what);
    }
}

static void check_u64(uint64_t got, uint64_t want, const char *what)
{
    checks_run++;
    if (got == want) {
        printf("  ok    %s = 0x%" PRIx64 "\n", what, got);
    } else {
        checks_failed++;
        printf("  FAIL  %s = 0x%" PRIx64 ", expected 0x%" PRIx64 "\n", what, got, want);
    }
}

/* ------------------------------------------------------------------------- */
/* Simulated Tensix                                                          */
/* ------------------------------------------------------------------------- */

/**
 * Stands in for one core. `go_reads_before_done` scripts the completion handshake:
 * after that many reads of the go message, the simulated firmware writes DONE,
 * exactly as the BRISC firmware does. UINT32_MAX means "never finishes", which is
 * how the timeout path is exercised.
 */
typedef struct simulated_tensix {
    uint8_t l1[FT_TT_MEM_L1_SIZE];
    uint32_t soft_reset;
    /* The two Tensix debug registers a snapshot reads alongside the soft reset. Modelled
     * here because a fake that answered FT_TT_EINVAL for them would let a snapshot which
     * SILENTLY DROPPED them look identical to one that reads them. */
    uint32_t instrn_buf_status;
    uint32_t invalid_instrn;
    int refuse_debug_registers; /* model a device that will not answer them */
    uint32_t go_reads;
    uint32_t go_reads_before_done;
    uint32_t write_calls;
    int fail_after_writes; /* -1 = never fail */
} simulated_tensix;

static simulated_tensix simulated_core;

static int simulated_write(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                           size_t bytes)
{
    simulated_tensix *core = (simulated_tensix *)user;

    (void)x;
    (void)y;
    core->write_calls++;
    if (core->fail_after_writes >= 0 && (int)core->write_calls > core->fail_after_writes)
        return FT_TT_EIO;
    if (addr == FT_TT_TENSIX_SOFT_RESET_ADDR && bytes == 4u) {
        memcpy(&core->soft_reset, src, 4u);
        return FT_TT_OK;
    }
    if (addr + bytes > FT_TT_MEM_L1_SIZE)
        return FT_TT_EINVAL;
    memcpy(core->l1 + addr, src, bytes);
    return FT_TT_OK;
}

static int simulated_read(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst, size_t bytes)
{
    simulated_tensix *core = (simulated_tensix *)user;

    (void)x;
    (void)y;
    if (addr == FT_TT_TENSIX_SOFT_RESET_ADDR && bytes == 4u) {
        memcpy(dst, &core->soft_reset, 4u);
        return FT_TT_OK;
    }
    if (addr == FT_TT_TENSIX_INSTRN_BUF_STATUS_ADDR && bytes == 4u) {
        if (core->refuse_debug_registers)
            return FT_TT_EIO;
        memcpy(dst, &core->instrn_buf_status, 4u);
        return FT_TT_OK;
    }
    if (addr == FT_TT_TENSIX_DBG_INVALID_INSTRN_ADDR && bytes == 4u) {
        if (core->refuse_debug_registers)
            return FT_TT_EIO;
        memcpy(dst, &core->invalid_instrn, 4u);
        return FT_TT_OK;
    }
    if (addr + bytes > FT_TT_MEM_L1_SIZE)
        return FT_TT_EINVAL;

    /* The scripted half of the fake firmware: reading the go message eventually
     * returns DONE. */
    if (addr == FT_TT_MB_GO_SLOT_ADDR(0) && bytes == sizeof(ft_tt_go_msg)) {
        core->go_reads++;
        if (core->go_reads >= core->go_reads_before_done)
            core->l1[FT_TT_MB_GO_SLOT_ADDR(0) + offsetof(ft_tt_go_msg, signal)] =
                (uint8_t)FT_TT_RUN_MSG_DONE;
    }
    memcpy(dst, core->l1 + addr, bytes);
    return FT_TT_OK;
}

static ft_tt_noc_access simulated_access(void)
{
    ft_tt_noc_access access;

    access.write = simulated_write;
    access.read = simulated_read;
    access.user = &simulated_core;
    return access;
}

static void reset_simulated_core(void)
{
    memset(&simulated_core, 0, sizeof(simulated_core));
    simulated_core.go_reads_before_done = UINT32_MAX;
    simulated_core.fail_after_writes = -1;
    /* A healthy core: 140 of 140 read this on silicon, in every state observed. */
    simulated_core.instrn_buf_status = FT_TT_TENSIX_INSTRN_BUF_READY;
    simulated_core.invalid_instrn = 0u;
}

static uint32_t simulated_l1_u32(uint32_t address)
{
    uint32_t value;

    memcpy(&value, simulated_core.l1 + address, sizeof(value));
    return value;
}

/* ------------------------------------------------------------------------- */
/* Synthetic ELF builder                                                     */
/* ------------------------------------------------------------------------- */

/*
 * Lays out an ELF32 little-endian RISC-V executable shaped like the ones
 * build_tensix_kernel.sh produces: a text PT_LOAD whose VMA equals its LMA, an
 * optional data PT_LOAD whose VMA is in local memory and whose LMA is in L1, and a
 * non-allocatable `.segments` note carrying the per-region size limits.
 */

#define SYNTHETIC_ELF_CAPACITY 4096u

typedef struct synthetic_elf_spec {
    uint32_t entry;
    uint32_t text_vaddr;
    uint32_t text_bytes;
    uint32_t text_limit;
    int with_data;
    uint32_t data_vaddr;
    uint32_t data_lma;
    uint32_t data_bytes;
    uint32_t data_limit;
    int with_segment_note;
    uint16_t elf_type;
    uint16_t machine;
    uint32_t text_filesz_override; /* 0 = use text_bytes */
} synthetic_elf_spec;

typedef struct synthetic_elf {
    uint8_t bytes[SYNTHETIC_ELF_CAPACITY];
    size_t size;
} synthetic_elf;

static void put_u16(uint8_t *base, size_t offset, uint16_t value)
{
    base[offset] = (uint8_t)(value & 0xFFu);
    base[offset + 1u] = (uint8_t)(value >> 8);
}

static void put_u32(uint8_t *base, size_t offset, uint32_t value)
{
    base[offset] = (uint8_t)(value & 0xFFu);
    base[offset + 1u] = (uint8_t)((value >> 8) & 0xFFu);
    base[offset + 2u] = (uint8_t)((value >> 16) & 0xFFu);
    base[offset + 3u] = (uint8_t)((value >> 24) & 0xFFu);
}

static void write_program_header(uint8_t *base, size_t at, uint32_t offset, uint32_t vaddr,
                                 uint32_t paddr, uint32_t filesz, uint32_t memsz)
{
    put_u32(base, at + 0u, 1u); /* PT_LOAD */
    put_u32(base, at + 4u, offset);
    put_u32(base, at + 8u, vaddr);
    put_u32(base, at + 12u, paddr);
    put_u32(base, at + 16u, filesz);
    put_u32(base, at + 20u, memsz);
    put_u32(base, at + 24u, 5u); /* PF_R | PF_X */
    put_u32(base, at + 28u, 4u);
}

static void write_section_header(uint8_t *base, size_t at, uint32_t name, uint32_t type,
                                 uint32_t offset, uint32_t size)
{
    memset(base + at, 0, 40u);
    put_u32(base, at + 0u, name);
    put_u32(base, at + 4u, type);
    put_u32(base, at + 16u, offset);
    put_u32(base, at + 20u, size);
}

/** Fixed layout so every offset below is a constant, not a running total. */
#define ELF_PHDR_OFFSET   64u
#define ELF_TEXT_OFFSET   256u
#define ELF_DATA_OFFSET   1024u
#define ELF_NOTE_OFFSET   1536u
#define ELF_STRTAB_OFFSET 1664u
#define ELF_SHDR_OFFSET   1792u

static void build_synthetic_elf(const synthetic_elf_spec *spec, synthetic_elf *out)
{
    static const char string_table[] = "\0.segments\0.shstrtab";
    uint32_t note_words[6];
    uint32_t note_bytes = 0;
    uint32_t text_filesz = spec->text_filesz_override ? spec->text_filesz_override : spec->text_bytes;
    uint16_t phnum = spec->with_data ? 2u : 1u;
    uint32_t index;

    memset(out, 0, sizeof(*out));
    memcpy(out->bytes, "\x7f" "ELF", 4u);
    out->bytes[4] = 1u; /* ELFCLASS32 */
    out->bytes[5] = 1u; /* ELFDATA2LSB */
    out->bytes[6] = 1u; /* EV_CURRENT */
    put_u16(out->bytes, 16u, spec->elf_type);
    put_u16(out->bytes, 18u, spec->machine);
    put_u32(out->bytes, 20u, 1u);
    put_u32(out->bytes, 24u, spec->entry);
    put_u32(out->bytes, 28u, ELF_PHDR_OFFSET);
    put_u32(out->bytes, 32u, ELF_SHDR_OFFSET);
    put_u16(out->bytes, 40u, 52u); /* e_ehsize */
    put_u16(out->bytes, 42u, 32u); /* e_phentsize */
    put_u16(out->bytes, 44u, phnum);
    put_u16(out->bytes, 46u, 40u); /* e_shentsize */
    put_u16(out->bytes, 48u, 3u);  /* null, .segments, .shstrtab */
    put_u16(out->bytes, 50u, 2u);  /* e_shstrndx */

    write_program_header(out->bytes, ELF_PHDR_OFFSET, ELF_TEXT_OFFSET, spec->text_vaddr,
                         spec->text_vaddr, text_filesz, spec->text_bytes);
    if (spec->with_data)
        write_program_header(out->bytes, ELF_PHDR_OFFSET + 32u, ELF_DATA_OFFSET, spec->data_vaddr,
                             spec->data_lma, spec->data_bytes, spec->data_bytes);

    /* Recognisable contents so a misplaced write is obvious in the dump. */
    for (index = 0; index < spec->text_bytes && index < 256u; index++)
        out->bytes[ELF_TEXT_OFFSET + index] = (uint8_t)(0xA0u + (index & 0x0Fu));
    if (spec->with_data)
        for (index = 0; index < spec->data_bytes && index < 256u; index++)
            out->bytes[ELF_DATA_OFFSET + index] = (uint8_t)(0xD0u + (index & 0x0Fu));

    if (spec->with_segment_note) {
        note_words[0] = spec->text_vaddr;
        note_words[1] = spec->text_vaddr;
        note_words[2] = spec->text_limit;
        note_bytes = 12u;
        if (spec->with_data) {
            note_words[3] = spec->data_vaddr;
            note_words[4] = spec->data_vaddr;
            note_words[5] = spec->data_limit;
            note_bytes = 24u;
        }
        for (index = 0; index < note_bytes / 4u; index++)
            put_u32(out->bytes, ELF_NOTE_OFFSET + index * 4u, note_words[index]);
    }

    memcpy(out->bytes + ELF_STRTAB_OFFSET, string_table, sizeof(string_table));

    write_section_header(out->bytes, ELF_SHDR_OFFSET, 0u, 0u, 0u, 0u);
    write_section_header(out->bytes, ELF_SHDR_OFFSET + 40u, 1u /* ".segments" */, 1u /* PROGBITS */,
                         ELF_NOTE_OFFSET, note_bytes);
    write_section_header(out->bytes, ELF_SHDR_OFFSET + 80u, 11u /* ".shstrtab" */, 3u /* STRTAB */,
                         ELF_STRTAB_OFFSET, (uint32_t)sizeof(string_table));

    out->size = ELF_SHDR_OFFSET + 120u;
}

/** A plausible TRISC0 kernel: text just above the firmware hole, data staged in L1. */
static void build_kernel_elf(synthetic_elf *out, uint32_t text_vaddr, uint32_t text_bytes)
{
    synthetic_elf_spec spec;

    memset(&spec, 0, sizeof(spec));
    spec.entry = text_vaddr;
    spec.text_vaddr = text_vaddr;
    spec.text_bytes = text_bytes;
    spec.text_limit = FT_TT_MEM_MAX_KERNEL_SIZE;
    spec.with_data = 1;
    spec.data_vaddr = FT_TT_MEM_LOCAL_BASE + 0x400u;
    spec.data_lma = text_vaddr + text_bytes;
    spec.data_bytes = 64u;
    spec.data_limit = FT_TT_MEM_TRISC_LOCAL_SIZE;
    spec.with_segment_note = 1;
    spec.elf_type = 2u;
    spec.machine = 243u;
    build_synthetic_elf(&spec, out);
}

/** A plausible TRISC0 firmware: text AT the firmware base, data image relocated. */
static void build_firmware_elf(synthetic_elf *out, unsigned processor, uint32_t text_bytes)
{
    synthetic_elf_spec spec;

    memset(&spec, 0, sizeof(spec));
    spec.entry = ft_tt_processor_firmware_base(processor);
    spec.text_vaddr = spec.entry;
    spec.text_bytes = text_bytes;
    spec.text_limit = ft_tt_processor_firmware_size(processor);
    spec.with_data = 1;
    spec.data_vaddr = FT_TT_MEM_LOCAL_BASE + 0x100u;
    spec.data_lma = spec.entry + text_bytes; /* deliberately NOT where it belongs */
    spec.data_bytes = 32u;
    spec.data_limit = FT_TT_MEM_TRISC_LOCAL_SIZE;
    spec.with_segment_note = 1;
    spec.elf_type = 2u;
    spec.machine = 243u;
    build_synthetic_elf(&spec, out);
}

/* ------------------------------------------------------------------------- */
/* 1. Mailbox layout                                                         */
/* ------------------------------------------------------------------------- */

/*
 * These are a regression lock on ft_tt_mailbox.h, not an independent derivation: they
 * pin the offsets that were computed from upstream's field order so a later edit to
 * the struct cannot move them silently. The values themselves are only proven correct
 * by the read-back in docs/tt_launch_notes.md §6 step 4.
 */
/** The p150a as ARC telemetry reports it: 14 Tensix columns, 8 DRAM banks, nothing
 *  harvested. */
static ft_tt_grid_info p150a_grid(void)
{
    ft_tt_grid_info grid;

    memset(&grid, 0, sizeof(grid));
    grid.valid = true;
    /* Off-hardware nothing selects an architecture, so the fixture names the one it
     * is a fixture for. Every geometry constant ft_tt_coord uses arrives this way. */
    grid.arch = ft_tt_arch_blackhole();
    /* The card reports translation on, and DRAM's translated column pair only exists
     * when it is: without this the coordinate module correctly answers in NoC0. */
    grid.noc_translation_enabled = 1u;
    grid.tensix_enabled_col = 16383u;
    grid.tensix_column_count = 14u;
    grid.tensix_core_count = 140u;
    grid.enabled_gddr = 255u;
    grid.dram_bank_count = 8u;
    return grid;
}

static void test_mailbox_layout(void)
{
    printf("\nmailbox layout (locked against ft_tt_mailbox.h)\n");
    check_u64(sizeof(ft_tt_kernel_config_msg), 144u, "sizeof(kernel_config_msg)");
    check_u64(sizeof(ft_tt_launch_msg), 144u, "sizeof(launch_msg)");
    check_u64(sizeof(ft_tt_go_msg), 4u, "sizeof(go_msg)");
    check_u64(FT_TT_MB_SUBORDINATE_SYNC_ADDR, 0x68u, "subordinate_sync L1 address");
    check_u64(FT_TT_MB_LAUNCH_RD_PTR_ADDR, 0x6Cu, "launch_msg_rd_ptr L1 address");
    check_u64(FT_TT_MB_LAUNCH_ADDR, 0x70u, "launch[0] L1 address");
    check_u64(FT_TT_MB_GO_MESSAGES_ADDR, 0x4F0u, "go_messages[0] L1 address");
    check_u64(FT_TT_MB_GO_MESSAGE_INDEX_ADDR, 0x520u, "go_message_index L1 address");
    check_u64(offsetof(ft_tt_kernel_config_msg, enables), 120u, "kernel_config.enables offset");
    check_u64(offsetof(ft_tt_kernel_config_msg, kernel_text_offset), 52u,
              "kernel_config.kernel_text_offset offset");
    check_u64(offsetof(ft_tt_kernel_config_msg, preload), 143u,
              "kernel_config.preload offset (must be last)");
    check(FT_TT_RUN_MSG_DONE == 0u, "RUN_MSG_DONE is zero, which is why the handshake needs priming");
    check(FT_TT_SANITIZE_OK != 0u,
          "DebugSanitizeOK is NOT zero, which is why the watcher slots need priming too");

    /* core_info sits BEHIND watcher and dprint_buf, so these three numbers are one
     * chain: a wrong watcher size moves core_info and the firmware then resolves NoC
     * endpoints out of someone else's bytes. */
    check_u64(sizeof(ft_tt_watcher_msg), 288u, "sizeof(watcher_msg)");
    check_u64(sizeof(ft_tt_device_print_memory_layout), 1020u, "sizeof(dprint_buf)");
    check_u64(sizeof(ft_tt_core_info_msg), 248u, "sizeof(core_info_msg)");
    check_u64(FT_TT_MB_WATCHER_ADDR, 0x530u, "watcher L1 address");
    check_u64(FT_TT_MB_CORE_INFO_ADDR, 0xA50u, "core_info L1 address");
    check_u64(FT_TT_MB_WATCHER_SANITIZE_ADDR(0), 0x548u, "watcher.sanitize[0] L1 address");
    check_u64(FT_TT_MB_WATCHER_SANITIZE_ADDR(1), 0x560u, "watcher.sanitize[1] L1 address");
    check_u64(offsetof(ft_tt_core_info_msg, virtual_non_worker_cores), 137u,
              "core_info.virtual_non_worker_cores offset");
    check_u64(offsetof(ft_tt_core_info_msg, noc_size_x), 228u, "core_info.noc_size_x offset");
    check_u64(offsetof(ft_tt_core_info_msg, core_magic_number), 240u,
              "core_info.core_magic_number offset");
}

/* ------------------------------------------------------------------------- */
/* 1b. core_info encoding                                                     */
/* ------------------------------------------------------------------------- */

/**
 * The sentinel is the whole test. Zero-filled tables are not "empty" to the firmware:
 * an all-zero addressable_core reads as (0, 0) of type TENSIX, which is a real
 * coordinate with a real type, and get_core_type() matches it.
 */
static void test_core_info_encoding(void)
{
    const ft_tt_grid_info grid = p150a_grid();
    ft_tt_launch_non_worker_core storage[FT_TT_LAUNCH_MAX_NON_WORKER_CORES];
    ft_tt_launch_non_worker_core too_many[FT_TT_LAUNCH_MAX_NON_WORKER_CORES];
    ft_tt_launch_core_info_config config;
    ft_tt_core_info_msg message;
    unsigned index;
    unsigned dram_entries = 0;
    bool every_unused_slot_sentinel = true;

    printf("\ncore_info: the endpoint tables NoC sanitize resolves against\n");

    check(ft_tt_launch_core_info_from_grid(&grid, 0u, 0u, 0x14000u, storage,
                                           FT_TT_LAUNCH_MAX_NON_WORKER_CORES,
                                           &config) == FT_TT_OK,
          "a config builds from the p150a grid");
    check_u64(config.virtual_non_worker_count, 24u,
              "8 enabled DRAM banks x 3 NoC ports are registered");
    check_u64(config.noc_size_x, 17u, "noc_size_x");
    check_u64(config.noc_size_y, 12u, "noc_size_y");
    check_u64(config.worker_grid_size_x, 14u, "worker_grid_size_x");
    check_u64(config.noc_dram_addr_base, 0u, "noc_dram_addr_base");
    check_u64(config.noc_dram_addr_end, 8uLL * 0xff000000uLL, "noc_dram_addr_end (8 banks)");
    check(storage[0].x == 17u && storage[0].y == 12u,
          "the first entry is DRAM bank 0 port 0 at TRANSLATED (17, 12)");
    check(storage[0].type == FT_TT_ADDRESSABLE_CORE_DRAM, "typed DRAM, not TENSIX");

    check(ft_tt_launch_encode_core_info(&config, &message, sizeof(message)) == FT_TT_OK,
          "it encodes");
    check_u64(message.core_magic_number, FT_TT_CORE_MAGIC_WORKER, "core_magic_number is WORKER");

    for (index = 0; index < FT_TT_MAX_PHYSICAL_NON_WORKER_CORES; index++) {
        if (message.non_worker_cores[index].x != FT_TT_CORE_COORD_INVALID ||
            message.non_worker_cores[index].y != FT_TT_CORE_COORD_INVALID ||
            message.non_worker_cores[index].type != FT_TT_ADDRESSABLE_CORE_UNKNOWN)
            every_unused_slot_sentinel = false;
    }
    check(every_unused_slot_sentinel,
          "every PHYSICAL slot is the 0xFF/UNKNOWN sentinel — Blackhole skips that table");

    for (index = 0; index < FT_TT_MAX_VIRTUAL_NON_WORKER_CORES; index++) {
        if (message.virtual_non_worker_cores[index].type == FT_TT_ADDRESSABLE_CORE_DRAM) {
            dram_entries++;
        } else if (message.virtual_non_worker_cores[index].x != FT_TT_CORE_COORD_INVALID ||
                   message.virtual_non_worker_cores[index].y != FT_TT_CORE_COORD_INVALID ||
                   message.virtual_non_worker_cores[index].type !=
                       FT_TT_ADDRESSABLE_CORE_UNKNOWN) {
            every_unused_slot_sentinel = false;
        }
    }
    check_u64(dram_entries, 24u, "24 DRAM ports in the virtual table");
    check(every_unused_slot_sentinel,
          "and the 5 unused virtual slots are the sentinel, NOT zero (which would read as "
          "Tensix (0, 0))");
    check(message.harvested_coords[0] == FT_TT_CORE_COORD_INVALID &&
              message.virtual_harvested_coords[1] == FT_TT_CORE_COORD_INVALID,
          "an unharvested part gets the sentinel in both harvested arrays");

    /* Refusals. */
    check(ft_tt_launch_encode_core_info(&config, &message, sizeof(message) - 1u) == FT_TT_EINVAL,
          "a short buffer is refused");
    check(ft_tt_launch_encode_core_info(NULL, &message, sizeof(message)) == FT_TT_EINVAL,
          "a NULL config is refused");
    for (index = 0; index < FT_TT_LAUNCH_MAX_NON_WORKER_CORES; index++) {
        too_many[index].x = 1u;
        too_many[index].y = 2u;
        too_many[index].type = FT_TT_ADDRESSABLE_CORE_DRAM;
    }
    config.virtual_non_worker_cores = too_many;
    config.virtual_non_worker_count = FT_TT_MAX_VIRTUAL_NON_WORKER_CORES + 1u;
    check(ft_tt_launch_encode_core_info(&config, &message, sizeof(message)) == FT_TT_EINVAL,
          "more entries than the ABI's 29 virtual slots is refused");
    config.virtual_non_worker_count = 1u;
    too_many[0].x = FT_TT_CORE_COORD_INVALID;
    check(ft_tt_launch_encode_core_info(&config, &message, sizeof(message)) == FT_TT_EINVAL,
          "a coordinate that collides with the 0xFF sentinel is refused");

    check(ft_tt_launch_core_info_from_grid(&grid, 0u, 0u, 0x14000u, storage, 4u, &config) ==
              FT_TT_ENOMEM,
          "too little storage for the DRAM ports is refused, not truncated");
}

/* ------------------------------------------------------------------------- */
/* 1b2. Watcher priming                                                       */
/* ------------------------------------------------------------------------- */

/**
 * The firmware writes a sanitize report only into a slot that already reads
 * DebugSanitizeOK. A cleared mailbox reads 0, so an unprimed core stops on a violation
 * and records nothing — which is exactly what happened on the card before this existed.
 */
static void test_watcher_priming(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_debug_sanitize_addr_msg report;
    ft_tt_watcher_msg watcher;
    uint32_t enable = 0;

    printf("\nwatcher priming: DebugSanitizeOK is 2, and a cleared slot reads 0\n");
    reset_simulated_core();

    check(ft_tt_launch_clear_mailbox(&noc, 1u, 2u) == FT_TT_OK, "the mailbox is cleared");
    check(noc.read(noc.user, 1u, 2u, FT_TT_MB_WATCHER_SANITIZE_ADDR(0), &report,
                   sizeof(report)) == FT_TT_OK,
          "slot 0 is readable");
    check_u64(report.return_code, 0u, "and a CLEARED slot reads 0, not DebugSanitizeOK");

    check(ft_tt_launch_prime_watcher(&noc, 1u, 2u) == FT_TT_OK, "priming succeeds");
    check(noc.read(noc.user, 1u, 2u, FT_TT_MB_ADDR(watcher), &watcher, sizeof(watcher)) ==
              FT_TT_OK,
          "the whole watcher block is readable");
    check_u64(watcher.sanitize[0].return_code, FT_TT_SANITIZE_OK, "slot 0 now reads OK");
    check_u64(watcher.sanitize[1].return_code, FT_TT_SANITIZE_OK, "slot 1 now reads OK");
    check(noc.read(noc.user, 1u, 2u, FT_TT_MB_ADDR(watcher.enable), &enable, sizeof(enable)) ==
              FT_TT_OK,
          "enable is readable");
    check_u64(enable, FT_TT_WATCHER_ENABLED, "and watcher.enable says enabled");

    /* return_code shares its 32-bit word with which_risc; priming must not take it. */
    check_u64(watcher.sanitize[0].which_risc, 0u, "which_risc, its word neighbour, is untouched");
    check_u64(watcher.sanitize[0].noc_addr, 0u, "and so is the rest of the slot");
    check(ft_tt_launch_prime_watcher(NULL, 1u, 2u) == FT_TT_EINVAL, "a NULL accessor is refused");
}

/* ------------------------------------------------------------------------- */
/* 1c. The Phase 2 test L1 map, derived rather than chosen                     */
/* ------------------------------------------------------------------------- */

static void test_phase2_layout_derivation(void)
{
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_phase2_l1_layout layout;
    unsigned processor;

    printf("\nPhase 2 L1 map: derived from the firmware placements\n");

    memset(placements, 0, sizeof(placements));
    check(ft_tt_phase2_l1_layout_derive(NULL, 0u, &layout) == FT_TT_OK,
          "with no placements at all it still derives");
    check_u64(layout.kernel_config_base,
              (FT_TT_PHASE2_RESERVED_L1_END + FT_TT_PHASE2_LAYOUT_ALIGNMENT - 1u) &
                  ~(uint32_t)(FT_TT_PHASE2_LAYOUT_ALIGNMENT - 1u),
              "and the base is the end of the reserved regions, not FT_TT_L1_USABLE_BASE");
    check(layout.kernel_config_base > FT_TT_L1_USABLE_BASE,
          "which is ABOVE the allocator floor — the .data staging and bank tables live "
          "between them");

    /* A firmware image that stops below the reserved end must not lower the floor. */
    placements[FT_TT_PROC_DM0].bytes_written = 256u;
    placements[FT_TT_PROC_DM0].lowest_l1_address = FT_TT_L1_USABLE_BASE;
    placements[FT_TT_PROC_DM0].highest_l1_address = FT_TT_L1_USABLE_BASE + 256u;
    check(ft_tt_phase2_l1_layout_derive(placements, FT_TT_PROC_COUNT, &layout) == FT_TT_OK,
          "a small firmware image derives");
    check(layout.kernel_config_base >= FT_TT_PHASE2_RESERVED_L1_END,
          "and does NOT pull the base below the reserved end");

    /* One that reaches past it must raise the floor above itself. */
    placements[FT_TT_PROC_DM0].highest_l1_address = FT_TT_PHASE2_RESERVED_L1_END + 0x1000u + 4u;
    check(ft_tt_phase2_l1_layout_derive(placements, FT_TT_PROC_COUNT, &layout) == FT_TT_OK,
          "a firmware image reaching past the reserved end derives");
    check(layout.kernel_config_base >= placements[FT_TT_PROC_DM0].highest_l1_address,
          "and the base moves above it");
    check((layout.kernel_config_base % FT_TT_PHASE2_LAYOUT_ALIGNMENT) == 0u,
          "the base is 64-byte aligned — the DRAM alignment, not L1's 16, because a CB is "
          "the local end of a DRAM read");
    check(FT_TT_PHASE2_LAYOUT_ALIGNMENT == 64u,
          "and 64 is what that alignment is on this part");
    check((FT_TT_PHASE2_CB_IN0_OFFSET % FT_TT_PHASE2_LAYOUT_ALIGNMENT) == 0u &&
              (FT_TT_PHASE2_RTA_OFFSET % FT_TT_PHASE2_LAYOUT_ALIGNMENT) == 0u &&
              (FT_TT_PHASE2_DESTINATION_OFFSET % FT_TT_PHASE2_LAYOUT_ALIGNMENT) == 0u,
          "every offset is a multiple of it, so aligning the base aligns the whole map");

    check_u64(layout.cb_config_address, layout.kernel_config_base, "cb_config is offset 0");
    check(layout.destination_address - layout.kernel_config_base ==
              FT_TT_PHASE2_DESTINATION_OFFSET,
          "and every other field is a fixed offset from the base");
    check(layout.cb_config_address - layout.kernel_config_base <= 0xFFFFu &&
              ft_tt_phase2_runtime_args_address(&layout, FT_TT_PROC_COUNT - 1u) -
                      layout.kernel_config_base <= 0xFFFFu,
          "the CB and runtime-arg offsets still fit their 16-bit wire fields");

    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        check(ft_tt_phase2_runtime_args_address(&layout, processor) ==
                  layout.runtime_args_base + processor * FT_TT_PHASE2_RTA_STRIDE,
              "runtime-arg block address is base + processor * stride");
    }
    check(ft_tt_phase2_runtime_args_address(&layout, FT_TT_PROC_COUNT) == 0u,
          "and an out-of-range processor gets 0, the encoder's no-args value");

    /* A base that cannot hold the map is refused rather than wrapping. */
    placements[FT_TT_PROC_DM0].highest_l1_address = FT_TT_MEM_L1_SIZE - 16u;
    check(ft_tt_phase2_l1_layout_derive(placements, FT_TT_PROC_COUNT, &layout) == FT_TT_EINVAL,
          "a base with no room left for the map is refused");
}

/* ------------------------------------------------------------------------- */
/* 2. ELF parsing                                                            */
/* ------------------------------------------------------------------------- */

static void test_elf_parses_a_kernel(void)
{
    synthetic_elf elf;
    ft_tt_elf_image image;
    uint32_t low = 0, high = 0;

    printf("\nELF parsing: a well-formed kernel image\n");
    build_kernel_elf(&elf, 0x7000u, 512u);

    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_OK, "parses");
    check_u64(image.entry_address, 0x7000u, "entry address");
    check_u64(image.segment_count, 2u, "loadable segment count");
    check_u64(image.segments[0].exec_address, 0x7000u, "text execution address");
    check_u64(image.segments[0].load_address, 0x7000u, "text load address (VMA == LMA)");
    check_u64(image.segments[0].file_bytes, 512u, "text file bytes");
    check_u64(image.segments[1].exec_address, FT_TT_MEM_LOCAL_BASE + 0x400u,
              "data executes from local memory");
    check_u64(image.segments[1].load_address, 0x7000u + 512u, "data is staged in L1 after text");
    check(ft_tt_elf_text_segment(&image) == &image.segments[0], "text segment accessor");

    check(ft_tt_elf_load_extent(&image, &low, &high) == FT_TT_OK, "load extent computes");
    check_u64(low, 0x7000u, "load extent low");
    check_u64(high, 0x7000u + 512u + 64u, "load extent high");
    check_u64(image.loaded_bytes, 512u + 64u, "bytes a load would transfer");
}

static void test_elf_rejections(void)
{
    synthetic_elf elf;
    ft_tt_elf_image image;
    synthetic_elf_spec spec;

    printf("\nELF parsing: malformed images are refused\n");

    build_kernel_elf(&elf, 0x7000u, 512u);
    elf.bytes[1] = 'X';
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL, "bad magic");

    build_kernel_elf(&elf, 0x7000u, 512u);
    elf.bytes[4] = 2u;
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL, "ELF64");

    build_kernel_elf(&elf, 0x7000u, 512u);
    elf.bytes[5] = 2u;
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL, "big-endian");

    memset(&spec, 0, sizeof(spec));
    build_kernel_elf(&elf, 0x7000u, 512u);
    put_u16(elf.bytes, 16u, 1u); /* ET_REL */
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL, "not ET_EXEC");

    build_kernel_elf(&elf, 0x7000u, 512u);
    put_u16(elf.bytes, 18u, 62u); /* EM_X86_64 */
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL, "not EM_RISCV");

    build_kernel_elf(&elf, 0x7000u, 512u);
    put_u32(elf.bytes, 24u, 0x7004u);
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL,
          "entry is not the start of the first segment");

    build_kernel_elf(&elf, 0x7002u, 512u); /* misaligned vaddr */
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL, "misaligned segment");

    build_kernel_elf(&elf, 0x7000u, 512u);
    put_u32(elf.bytes, ELF_PHDR_OFFSET + 16u, 510u); /* p_filesz not a word multiple */
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL,
          "file size is not a multiple of 4");

    build_kernel_elf(&elf, 0x7000u, 512u);
    put_u32(elf.bytes, ELF_PHDR_OFFSET + 16u, 0x10000u); /* p_filesz past the blob */
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL, "contents past end of blob");

    build_kernel_elf(&elf, 0x7000u, 512u);
    put_u32(elf.bytes, 28u, 0u); /* no program headers */
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL, "no program headers");

    check(ft_tt_elf_parse(NULL, 0u, &image) == FT_TT_EINVAL, "NULL blob");
    check(ft_tt_elf_parse(elf.bytes, 8u, &image) == FT_TT_EINVAL, "blob shorter than a header");
}

static void test_elf_segment_limits(void)
{
    synthetic_elf elf;
    ft_tt_elf_image image;
    synthetic_elf_spec spec;

    printf("\nELF parsing: the linker's region limits are enforced\n");

    memset(&spec, 0, sizeof(spec));
    spec.entry = 0x7000u;
    spec.text_vaddr = 0x7000u;
    spec.text_bytes = 512u;
    spec.text_limit = 256u; /* the kernel is twice its region */
    spec.with_segment_note = 1;
    spec.elf_type = 2u;
    spec.machine = 243u;
    build_synthetic_elf(&spec, &elf);
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_EINVAL,
          "text over its region limit is refused");

    spec.text_limit = 512u; /* exactly its region */
    build_synthetic_elf(&spec, &elf);
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_OK,
          "text exactly at its region limit is accepted");

    spec.with_segment_note = 0; /* no note: nothing to check against */
    build_synthetic_elf(&spec, &elf);
    check(ft_tt_elf_parse(elf.bytes, elf.size, &image) == FT_TT_OK,
          "an image with no .segments note still parses");
}

/* ------------------------------------------------------------------------- */
/* 3. Embedded blob format                                                   */
/* ------------------------------------------------------------------------- */

static void test_kernel_blob_format(void)
{
    static const unsigned char fake_trisc0[] = {1, 2, 3, 4};
    static const unsigned char fake_trisc1[] = {5, 6, 7, 8, 9};
    static const ft_tt_kernel_image images[] = {
        {"vector_add_compute.c", FT_TT_PROC_MATH0, fake_trisc0, sizeof(fake_trisc0)},
        {"vector_add_compute.c", FT_TT_PROC_MATH1, fake_trisc1, sizeof(fake_trisc1)},
    };
    static const ft_tt_kernel_program program = {"vector_add", "7.68.0", "0,1,16", images, 2u};

    printf("\nembedded blob format\n");
    check(ft_tt_kernel_image_for_processor(&program, FT_TT_PROC_MATH0) == &images[0],
          "lookup finds MATH0");
    check(ft_tt_kernel_image_for_processor(&program, FT_TT_PROC_MATH1) == &images[1],
          "lookup finds MATH1");
    check(ft_tt_kernel_image_for_processor(&program, FT_TT_PROC_MATH2) == NULL,
          "lookup returns NULL for an absent processor");
    check(ft_tt_kernel_image_for_processor(NULL, FT_TT_PROC_MATH0) == NULL, "lookup tolerates NULL");
    check_u64(ft_tt_kernel_program_enables(&program),
              (1u << FT_TT_PROC_MATH0) | (1u << FT_TT_PROC_MATH1), "derived enables mask");
    check_u64(ft_tt_kernel_program_enables(NULL), 0u, "enables of NULL is 0");
    check_u64(images[1].elf_size, 5u, "blob size travels with the blob");
}

/* ------------------------------------------------------------------------- */
/* 4. Segment placement into L1                                              */
/* ------------------------------------------------------------------------- */

static void test_kernel_image_placement(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_image_placement placement;
    synthetic_elf elf;
    const uint32_t text_address = FT_TT_MEM_TRISC0_FIRMWARE_BASE + 0x400u;

    printf("\nplacement: a kernel image lands at its ELF addresses\n");
    reset_simulated_core();
    build_kernel_elf(&elf, text_address, 512u);

    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, &placement) == FT_TT_OK,
          "kernel image loads");
    check_u64(placement.entry_address, text_address, "placement reports the entry point");
    check_u64(placement.bytes_written, 512u + 64u, "text and data were both written");
    check_u64(simulated_core.l1[text_address], 0xA0u, "text landed at its execution address");
    check_u64(simulated_core.l1[text_address + 512u], 0xD0u,
              "kernel data landed at its ELF load address");
    check(simulated_core.l1[ft_tt_processor_local_init_scratch(FT_TT_PROC_MATH0)] == 0u,
          "kernel data did NOT go to the firmware init scratch");
}

static void test_firmware_image_placement(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_image_placement placement;
    synthetic_elf elf;
    const uint32_t firmware_base = ft_tt_processor_firmware_base(FT_TT_PROC_MATH0);
    const uint32_t scratch = ft_tt_processor_local_init_scratch(FT_TT_PROC_MATH0);

    printf("\nplacement: firmware data is relocated to the init scratch\n");
    reset_simulated_core();
    build_firmware_elf(&elf, FT_TT_PROC_MATH0, 256u);

    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_FIRMWARE, elf.bytes,
                                  elf.size, &placement) == FT_TT_OK,
          "firmware image loads");
    check_u64(simulated_core.l1[firmware_base], 0xA0u, "text landed at the firmware base");
    /* data_vaddr was MEM_LOCAL_BASE + 0x100, so it must land at scratch + 0x100 —
     * NOT at the ELF's own load address, which the firmware never reads. */
    check_u64(simulated_core.l1[scratch + 0x100u], 0xD0u,
              "firmware data landed at init scratch + its local-memory offset");
    check(simulated_core.l1[firmware_base + 256u] == 0u,
          "firmware data did NOT go to its ELF load address");
}

static void test_placement_refusals(void)
{
    ft_tt_noc_access noc = simulated_access();
    synthetic_elf elf;

    printf("\nplacement: bad placements are refused before any byte is written\n");
    reset_simulated_core();

    build_firmware_elf(&elf, FT_TT_PROC_MATH1, 256u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_FIRMWARE, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "firmware for the wrong processor is refused");

    build_firmware_elf(&elf, FT_TT_PROC_MATH0, 4096u); /* hole is 2560 bytes */
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_FIRMWARE, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "firmware larger than its hole is refused");

    build_kernel_elf(&elf, FT_TT_MEM_TRISC0_FIRMWARE_BASE - 0x100u, 256u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "kernel text below its firmware is refused");

    build_kernel_elf(&elf, FT_TT_MEM_L1_SIZE - 128u, 256u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "a segment running off the end of L1 is refused");

    /* Our kernels are not execute-in-place, so their text sits in the TAIL of their own
     * processor's firmware hole and the linker's only bound is MEM_*_KERNEL_SIZE — all
     * of L1. One byte past the hole is TRISC1's firmware. The measured margin on the
     * real vector_add build is 68 bytes (docs/tt_bringup_log.md, step 6). */
    build_kernel_elf(&elf, FT_TT_MEM_TRISC0_FIRMWARE_BASE + 0x900u, 512u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "a kernel that overruns its firmware hole into the next processor is refused");

    check(simulated_core.write_calls == 0u, "no write reached the device on any refusal");

    check(ft_tt_launch_load_image(&noc, 1, 2, 99u, FT_TT_IMAGE_KERNEL, elf.bytes, elf.size, NULL) ==
              FT_TT_EINVAL,
          "an out-of-range processor index is refused");
}

/**
 * The OTHER half of the merged placement rule: a RELOCATE_KERNEL_TEXT=1 image, linked
 * into this processor's slab of the KERNEL_CONFIG region rather than the tail of its
 * firmware hole. Same entry point, same kind, no flag — the ELF's text address is what
 * selects the region.
 */
static void test_relocated_kernel_placement(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_image_placement placement;
    synthetic_elf elf;
    const uint32_t slab = ft_tt_op_text_base(FT_TT_PROC_MATH0);

    printf("\nplacement: a relocated kernel lands in its kernel-config slab\n");
    reset_simulated_core();
    check(slab >= FT_TT_OP_KERNEL_CONFIG_BASE, "MATH0's slab is inside the kernel-config region");
    build_kernel_elf(&elf, slab, 512u);

    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, &placement) == FT_TT_OK,
          "a relocated kernel loads through the SAME entry point as an in-hole one");
    check_u64(placement.entry_address, slab, "placement reports the slab base as the entry");
    check_u64(simulated_core.l1[slab], 0xA0u, "text landed at the slab base");
    check_u64(simulated_core.l1[slab + 512u], 0xD0u, "its data image followed the text");
}

/**
 * The asymmetric slab map (2026-11-19): the shipped map is legal, and a map that overlaps,
 * runs past the region, sits on the reserved head or is misaligned is refused. Then the
 * grown slabs are usable to their new end: an image in BRISC's 16 KiB slab past the old
 * 12 KiB bound loads, and one running into NCRISC's slab does not.
 */
static void test_text_slab_map(void)
{
    ft_tt_noc_access noc = simulated_access();
    uint32_t offsets[FT_TT_PROC_COUNT];
    uint32_t bytes[FT_TT_PROC_COUNT];
    synthetic_elf_spec spec;
    synthetic_elf elf;
    const uint32_t dm0 = ft_tt_op_text_base(FT_TT_PROC_DM0);
    const uint32_t dm0_end = dm0 + ft_tt_op_text_slab_bytes(FT_TT_PROC_DM0);
    unsigned processor;

    printf("\nslab map: asymmetric slabs are legal, broken maps are refused\n");
    for (processor = 0; processor < FT_TT_PROC_COUNT; processor++) {
        offsets[processor] = ft_tt_op_text_base(processor) - FT_TT_OP_KERNEL_CONFIG_BASE;
        bytes[processor] = ft_tt_op_text_slab_bytes(processor);
    }
    check(ft_tt_op_text_slab_map_valid(offsets, bytes), "the shipped map is legal");
    check_u64(offsets[FT_TT_PROC_MATH2] + bytes[FT_TT_PROC_MATH2], FT_TT_OP_KERNEL_CONFIG_BYTES,
              "the slabs fill the region to its end");
    check(bytes[FT_TT_PROC_DM0] >= 13512u && bytes[FT_TT_PROC_DM1] >= 10008u,
          "BRISC holds the stamped gdn reader, NCRISC the stamped topk writer");

    bytes[FT_TT_PROC_DM0] += 0x10u;
    check(!ft_tt_op_text_slab_map_valid(offsets, bytes), "DM0 overlapping DM1 is refused");
    bytes[FT_TT_PROC_DM0] -= 0x10u;
    bytes[FT_TT_PROC_MATH2] += 0x10u;
    check(!ft_tt_op_text_slab_map_valid(offsets, bytes), "MATH2 past the region end is refused");
    bytes[FT_TT_PROC_MATH2] -= 0x10u;
    offsets[FT_TT_PROC_DM0] = FT_TT_OP_TEXT_REGION_FIRST_OFFSET - 0x10u;
    check(!ft_tt_op_text_slab_map_valid(offsets, bytes), "DM0 on the replica table is refused");
    offsets[FT_TT_PROC_DM0] = FT_TT_OP_TEXT_OFFSET_DM0;
    offsets[FT_TT_PROC_MATH1] += 0x4u;
    check(!ft_tt_op_text_slab_map_valid(offsets, bytes), "a misaligned slab is refused");
    offsets[FT_TT_PROC_MATH1] -= 0x4u;
    bytes[FT_TT_PROC_DM1] = 0u;
    check(!ft_tt_op_text_slab_map_valid(offsets, bytes), "an empty slab is refused");

    reset_simulated_core();
    memset(&spec, 0, sizeof(spec));
    spec.entry = dm0;
    spec.text_vaddr = dm0;
    spec.text_bytes = 256u;
    spec.text_limit = FT_TT_MEM_MAX_KERNEL_SIZE;
    spec.with_data = 1;
    spec.data_vaddr = FT_TT_MEM_LOCAL_BASE + 0x400u;
    /* The load image at the slab's last 64 B: inside NCRISC's OLD slab (0x18000), so this
     * passes only under the grown map. */
    spec.data_lma = dm0_end - 64u;
    spec.data_bytes = 64u;
    spec.data_limit = FT_TT_MEM_BRISC_LOCAL_SIZE;
    spec.with_segment_note = 1;
    spec.elf_type = 2u;
    spec.machine = 243u;
    build_synthetic_elf(&spec, &elf);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_DM0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_OK,
          "a BRISC image ending at its 16 KiB slab end loads");
    spec.data_lma = dm0_end - 32u;
    build_synthetic_elf(&spec, &elf);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_DM0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "the same image running into NCRISC's slab is refused");
}

/**
 * "Never straddling", proven from both sides of both regions, plus the neighbour's slab.
 *
 * These are the cases the merged rule exists for. Each one is a placement that ONE of the
 * two old rules would have let through — an image in the slab was refused outright by
 * ft_tt_launch.c and accepted blind by ft_tt_op_dispatch.c, and vice versa — so a check
 * that only asserts "a good image loads" cannot see the merge at all.
 */
static void test_merged_region_refusals(void)
{
    ft_tt_noc_access noc = simulated_access();
    const uint32_t slab = ft_tt_op_text_base(FT_TT_PROC_MATH0);
    const uint32_t slab_end = slab + ft_tt_op_text_slab_bytes(FT_TT_PROC_MATH0);
    const uint32_t hole = FT_TT_MEM_TRISC0_FIRMWARE_BASE;
    synthetic_elf_spec spec;
    synthetic_elf elf;

    printf("\nplacement: a kernel is wholly in ONE region or it is refused\n");
    reset_simulated_core();

    /* Between the two regions: above the firmware hole, below the slab. This is the
     * address a default-linked kernel reaches by being too big, and it is where the next
     * processor's firmware lives. */
    build_kernel_elf(&elf, FT_TT_MEM_MAP_END, 256u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "text between the two regions is refused");

    /* A neighbour's slab is not this processor's. Loading MATH0's image as MATH1 is the
     * cheapest way to make that mistake and it must not load. */
    build_kernel_elf(&elf, slab, 256u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH1, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "an image in ANOTHER processor's slab is refused");

    /* Inside its own slab but not AT the base. Either the ELF was not built with
     * RELOCATE_KERNEL_TEXT, or this map and build_tensix_program.sh's have drifted apart:
     * it would load fine and leave the launch message pointing at empty L1. */
    build_kernel_elf(&elf, slab + 0x400u, 256u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "an image inside its slab but not at the slab BASE is refused");

    /* Straddling the top of the slab: text at the slab base, .data image running off the
     * end into the next slab. Refusing this is what stops one op overwriting another. */
    memset(&spec, 0, sizeof(spec));
    spec.entry = slab;
    spec.text_vaddr = slab;
    spec.text_bytes = 256u;
    spec.text_limit = FT_TT_MEM_MAX_KERNEL_SIZE;
    spec.with_data = 1;
    spec.data_vaddr = FT_TT_MEM_LOCAL_BASE + 0x400u;
    spec.data_lma = slab_end - 32u;
    spec.data_bytes = 64u;
    spec.data_limit = FT_TT_MEM_TRISC_LOCAL_SIZE;
    spec.with_segment_note = 1;
    spec.elf_type = 2u;
    spec.machine = 243u;
    build_synthetic_elf(&spec, &elf);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "a segment straddling the TOP of the slab is refused");

    /* ...and the same image ending exactly AT the boundary is legal, so the refusal above
     * is a bound and not an off-by-one that refuses everything near the edge. */
    spec.data_lma = slab_end - 64u;
    build_synthetic_elf(&spec, &elf);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_OK,
          "a segment ending exactly at the slab end is accepted");

    /* Straddling the BOTTOM of the firmware hole. This bound is new: the old in-hole rule
     * checked only that a segment ended before the hole did, so a .data image staged
     * BELOW the hole — on this processor's own firmware, or on the mailbox under it —
     * was accepted. */
    reset_simulated_core();
    spec.entry = hole + 0x400u;
    spec.text_vaddr = hole + 0x400u;
    spec.data_lma = hole - 64u;
    build_synthetic_elf(&spec, &elf);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "a segment straddling the BOTTOM of the firmware hole is refused");

    /* Text exactly at the firmware base IS the firmware, not a kernel. */
    build_kernel_elf(&elf, hole, 256u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, NULL) == FT_TT_EINVAL,
          "kernel text exactly at the firmware base is refused");

    check(simulated_core.write_calls == 0u,
          "no write reached the device on any of these refusals");
}

/* ------------------------------------------------------------------------- */
/* 5. Launch-message encoding                                                */
/* ------------------------------------------------------------------------- */

static ft_tt_launch_config baseline_config(void)
{
    ft_tt_launch_config config;

    memset(&config, 0, sizeof(config));
    config.kernel_config_base = 0x10000u;
    config.enables = (1u << FT_TT_PROC_MATH0) | (1u << FT_TT_PROC_MATH1) | (1u << FT_TT_PROC_MATH2);
    config.kernel_text_address[FT_TT_PROC_MATH0] = 0x6A00u;
    config.kernel_text_address[FT_TT_PROC_MATH1] = 0x7400u;
    config.kernel_text_address[FT_TT_PROC_MATH2] = 0x7E00u;
    config.kernel_text_bytes[FT_TT_PROC_MATH0] = 1228u;
    config.local_cb_config_address = 0x10100u;
    config.local_cb_mask = 0x10003u;
    config.runtime_args_address[FT_TT_PROC_MATH0] = 0x10800u;
    config.brisc_noc_id = 0u;
    return config;
}

static void test_launch_message_encoding(void)
{
    ft_tt_launch_config config = baseline_config();
    ft_tt_kernel_config_msg message;

    printf("\nlaunch message: encoding\n");
    check(ft_tt_launch_encode_launch_message(&config, &message, sizeof(message)) == FT_TT_OK,
          "encodes");
    check_u64(message.kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX], 0x10000u,
              "kernel_config_base[TENSIX]");
    check_u64(message.mode, FT_TT_DISPATCH_MODE_HOST, "dispatch mode is HOST");
    check_u64(message.enables, config.enables, "enables passes through");
    check_u64(message.local_cb_offset, 0x100u, "local CB offset is base-relative");
    check_u64(message.local_cb_mask, 0x10003u, "local CB mask passes through");
    check_u64(message.min_remote_cb_start_index, FT_TT_NUM_CIRCULAR_BUFFERS,
              "remote CBs are switched off");
    check_u64(message.rta_offset[FT_TT_PROC_MATH0].rta_offset, 0x800u,
              "runtime-arg offset is base-relative");
    check_u64(message.rta_offset[FT_TT_PROC_MATH1].rta_offset, FT_TT_RTA_NO_ARGS_SENTINEL,
              "a processor with no args gets the sentinel, not 0");
    check_u64(message.rta_offset[FT_TT_PROC_MATH0].crta_offset, FT_TT_RTA_NO_ARGS_SENTINEL,
              "common runtime args are always the sentinel");
    check_u64(message.kernel_text_size[FT_TT_PROC_MATH0], 1228u, "kernel text size");
    check_u64(message.preload, 0u, "preload stays 0 for host dispatch");

    /* The whole point of the wraparound encoding: base + offset must reconstruct the
     * text address in uint32 arithmetic, exactly as the TRISC firmware computes it. */
    printf("\nlaunch message: the non-XIP text-offset encoding round-trips\n");
    check_u64((uint32_t)(message.kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX] +
                         message.kernel_text_offset[FT_TT_PROC_MATH0]),
              0x6A00u, "MATH0: base + offset reconstructs the entry point");
    check_u64((uint32_t)(message.kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX] +
                         message.kernel_text_offset[FT_TT_PROC_MATH2]),
              0x7E00u, "MATH2: base + offset reconstructs the entry point");
    check(message.kernel_text_offset[FT_TT_PROC_MATH0] > 0x80000000u,
          "the offset really is a wrapped negative, not a coincidence");
}

static void test_launch_message_refusals(void)
{
    ft_tt_launch_config config;
    ft_tt_kernel_config_msg message;

    printf("\nlaunch message: bad configurations are refused\n");

    config = baseline_config();
    config.kernel_config_base = 0x10008u;
    check(ft_tt_launch_encode_launch_message(&config, &message, sizeof(message)) == FT_TT_EINVAL,
          "a base that is not 16-byte aligned");

    config = baseline_config();
    config.enables = 0u;
    check(ft_tt_launch_encode_launch_message(&config, &message, sizeof(message)) == FT_TT_EINVAL,
          "enables == 0, which would run nothing");

    config = baseline_config();
    config.kernel_text_address[FT_TT_PROC_MATH1] = 0u;
    check(ft_tt_launch_encode_launch_message(&config, &message, sizeof(message)) == FT_TT_EINVAL,
          "a processor enabled with no text address");

    config = baseline_config();
    config.runtime_args_address[FT_TT_PROC_MATH0] = config.kernel_config_base + 0xFFFFu;
    check(ft_tt_launch_encode_launch_message(&config, &message, sizeof(message)) == FT_TT_EINVAL,
          "runtime args beyond the 16-bit offset field");

    config = baseline_config();
    config.runtime_args_address[FT_TT_PROC_MATH0] = config.kernel_config_base - 16u;
    check(ft_tt_launch_encode_launch_message(&config, &message, sizeof(message)) == FT_TT_EINVAL,
          "runtime args below the config base");

    config = baseline_config();
    check(ft_tt_launch_encode_launch_message(&config, &message, sizeof(message) - 1u) ==
              FT_TT_EINVAL,
          "a buffer too small for the message");
    check(ft_tt_launch_encode_launch_message(NULL, &message, sizeof(message)) == FT_TT_EINVAL,
          "a NULL config");
}

/* ------------------------------------------------------------------------- */
/* 6. CB config and runtime args                                             */
/* ------------------------------------------------------------------------- */

static void test_cb_config_encoding(void)
{
    const ft_tt_launch_cb_entry entries[] = {
        {0u, 0x20000u, 4096u, 2048u, 2u},
        {1u, 0x21000u, 4096u, 2048u, 2u},
        {16u, 0x22000u, 4096u, 2048u, 2u},
    };
    uint32_t buffer[FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG];
    ft_tt_launch_cb_entry bad;
    uint64_t mask = 0;

    printf("\ncircular-buffer config encoding\n");
    memset(buffer, 0xEE, sizeof(buffer));
    check(ft_tt_launch_encode_cb_config(entries, 3u, buffer, sizeof(buffer), &mask) == FT_TT_OK,
          "encodes");
    check_u64(mask, (1u << 0) | (1u << 1) | (1u << 16), "mask has a bit per configured CB");
    check_u64(buffer[0 * 4 + 0], 0x20000u, "CB0 word 0 is the address");
    check_u64(buffer[0 * 4 + 1], 4096u, "CB0 word 1 is the size in bytes");
    check_u64(buffer[0 * 4 + 2], 2u, "CB0 word 2 is the page count");
    check_u64(buffer[0 * 4 + 3], 2048u, "CB0 word 3 is the page size in bytes");
    check_u64(buffer[16 * 4 + 0], 0x22000u, "CB16 is written at its own stride");
    check_u64(buffer[2 * 4 + 0], 0u, "an unconfigured CB is zeroed, not left as garbage");

    bad = entries[0];
    bad.index = FT_TT_NUM_CIRCULAR_BUFFERS;
    check(ft_tt_launch_encode_cb_config(&bad, 1u, buffer, sizeof(buffer), &mask) == FT_TT_EINVAL,
          "a CB index past the end is refused");
    check(ft_tt_launch_encode_cb_config(entries, 3u, buffer, 16u, &mask) == FT_TT_EINVAL,
          "a short buffer is refused");
}

static void test_runtime_argument_planting(void)
{
    ft_tt_noc_access noc = simulated_access();
    const uint32_t args[] = {7u, 0xDEADBEEFu, 3u};
    const uint32_t address = 0x10800u;

    printf("\nruntime arguments\n");
    reset_simulated_core();
    check(ft_tt_launch_plant_runtime_args(&noc, 1, 2, address, args, 3u) == FT_TT_OK, "plants");
    check_u64(simulated_l1_u32(address), 7u, "arg 0");
    check_u64(simulated_l1_u32(address + 4u), 0xDEADBEEFu, "arg 1");
    check_u64(simulated_l1_u32(address + 8u), 3u, "arg 2");
    check_u64(simulated_l1_u32(address + 12u), 0u, "nothing was written past the last arg");

    check(ft_tt_launch_plant_runtime_args(&noc, 1, 2, address + 2u, args, 3u) == FT_TT_EINVAL,
          "a misaligned address is refused");
    check(ft_tt_launch_plant_runtime_args(&noc, 1, 2, address, args, 0u) == FT_TT_EINVAL,
          "an empty arg list is refused");
    check(ft_tt_launch_plant_runtime_args(&noc, 1, 2, address, NULL, 3u) == FT_TT_EINVAL,
          "a NULL arg list is refused");
}

/* ------------------------------------------------------------------------- */
/* 7. Reset control                                                          */
/* ------------------------------------------------------------------------- */

static void test_reset_control(void)
{
    ft_tt_noc_access noc = simulated_access();
    uint32_t value = 0;

    printf("\nreset control (read-modify-write at 0xFFB121B0)\n");
    reset_simulated_core();
    simulated_core.soft_reset = 0u;

    check(ft_tt_launch_assert_reset(&noc, 1, 2, FT_TT_SOFT_RESET_ALL_RISCS) == FT_TT_OK,
          "assert all");
    check_u64(simulated_core.soft_reset, FT_TT_SOFT_RESET_ALL_RISCS, "all five bits are set");

    check(ft_tt_launch_deassert_reset(&noc, 1, 2, FT_TT_SOFT_RESET_BRISC, false) == FT_TT_OK,
          "deassert BRISC only");
    check_u64(simulated_core.soft_reset,
              FT_TT_SOFT_RESET_ALL_RISCS & ~(uint32_t)FT_TT_SOFT_RESET_BRISC,
              "BRISC released, the other four still held");

    check(ft_tt_launch_read_reset(&noc, 1, 2, &value) == FT_TT_OK, "read back");
    check_u64(value, simulated_core.soft_reset, "read matches");

    /* Unrelated bits in the register must survive a read-modify-write. */
    simulated_core.soft_reset = 0x00000055u;
    check(ft_tt_launch_assert_reset(&noc, 1, 2, FT_TT_SOFT_RESET_TRISC1) == FT_TT_OK,
          "assert one bit with foreign bits set");
    check_u64(simulated_core.soft_reset, 0x00000055u | FT_TT_SOFT_RESET_TRISC1,
              "foreign bits survived");

    simulated_core.soft_reset = FT_TT_SOFT_RESET_ALL_RISCS;
    check(ft_tt_launch_deassert_reset(&noc, 1, 2, FT_TT_SOFT_RESET_BRISC, true) == FT_TT_OK,
          "deassert with staggered start");
    check((simulated_core.soft_reset & FT_TT_SOFT_RESET_STAGGERED_START) != 0u,
          "staggered-start bit 31 was set");
}

/* ------------------------------------------------------------------------- */
/* 8. Mailbox protocol                                                       */
/* ------------------------------------------------------------------------- */

static void test_mailbox_clear(void)
{
    ft_tt_noc_access noc = simulated_access();

    printf("\nmailbox: clearing\n");
    reset_simulated_core();
    memset(simulated_core.l1, 0xAB, sizeof(simulated_core.l1));

    check(ft_tt_launch_clear_mailbox(&noc, 1, 2) == FT_TT_OK, "clears");
    check_u64(simulated_core.l1[FT_TT_MEM_MAILBOX_BASE], 0u, "first byte of the region");
    check_u64(simulated_core.l1[FT_TT_MEM_MAILBOX_END - 1u], 0u, "last byte of the region");
    check_u64(simulated_core.l1[FT_TT_MEM_MAILBOX_BASE - 1u], 0xABu,
              "the byte below the region was not touched");
    check_u64(simulated_core.l1[FT_TT_MEM_MAILBOX_END], 0xABu,
              "the byte above the region was not touched");
}

static void test_firmware_handshake_is_not_vacuous(void)
{
    ft_tt_noc_access noc = simulated_access();
    const uint32_t signal_address = FT_TT_MB_GO_SLOT_ADDR(0) + (uint32_t)offsetof(ft_tt_go_msg, signal);

    printf("\nmailbox: the firmware handshake has to be primed\n");
    reset_simulated_core();
    check(ft_tt_launch_clear_mailbox(&noc, 1, 2) == FT_TT_OK, "mailbox cleared");
    check_u64(simulated_core.l1[signal_address], FT_TT_RUN_MSG_DONE,
              "a cleared mailbox already reads DONE");

    check(ft_tt_launch_prime_firmware_handshake(&noc, 1, 2) == FT_TT_OK, "prime writes INIT");
    check_u64(simulated_core.l1[signal_address], FT_TT_RUN_MSG_INIT, "go signal is now INIT");
    check_u64(simulated_core.l1[FT_TT_MB_GO_SLOT_ADDR(0)], 0u,
              "priming touched only the signal byte");

    /* With the handshake primed, a firmware that never boots must time out rather
     * than report success. */
    simulated_core.go_reads_before_done = UINT32_MAX;
    check(ft_tt_launch_wait_firmware_ready(&noc, 1, 2, 20u) == FT_TT_EIO,
          "an unprimed-looking core times out instead of passing");

    simulated_core.go_reads = 0u;
    simulated_core.go_reads_before_done = 3u;
    simulated_core.l1[signal_address] = (uint8_t)FT_TT_RUN_MSG_INIT;
    check(ft_tt_launch_wait_firmware_ready(&noc, 1, 2, 2000u) == FT_TT_OK,
          "a booting firmware is detected");
}

static void test_launch_message_write_and_go(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_launch_config config = baseline_config();
    ft_tt_kernel_config_msg from_l1;
    const uint32_t signal_address = FT_TT_MB_GO_SLOT_ADDR(0) + (uint32_t)offsetof(ft_tt_go_msg, signal);

    printf("\nmailbox: launch message and go\n");
    reset_simulated_core();

    check(ft_tt_launch_write_launch_message(&noc, 1, 2, &config) == FT_TT_OK, "writes slot 0");
    memcpy(&from_l1, simulated_core.l1 + FT_TT_MB_LAUNCH_SLOT_ADDR(0), sizeof(from_l1));
    check_u64(from_l1.enables, config.enables, "enables round-tripped through L1");
    check_u64(from_l1.kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX], 0x10000u,
              "config base round-tripped through L1");
    check_u64(simulated_core.l1[FT_TT_MB_LAUNCH_SLOT_ADDR(1)], 0u, "slot 1 was left alone");

    check(ft_tt_launch_send_go(&noc, 1, 2) == FT_TT_OK, "go is sent");
    check_u64(simulated_core.l1[signal_address], FT_TT_RUN_MSG_GO, "go signal is GO");

    config.kernel_config_base = 0x10008u;
    check(ft_tt_launch_write_launch_message(&noc, 1, 2, &config) == FT_TT_EINVAL,
          "an invalid config is refused before it reaches L1");
}

static void test_completion_poll(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_launch_state state;
    const uint32_t signal_address = FT_TT_MB_GO_SLOT_ADDR(0) + (uint32_t)offsetof(ft_tt_go_msg, signal);

    printf("\nmailbox: completion poll\n");
    reset_simulated_core();
    simulated_core.l1[signal_address] = (uint8_t)FT_TT_RUN_MSG_GO;
    simulated_core.go_reads_before_done = 5u;
    check(ft_tt_launch_wait_done(&noc, 1, 2, 2000u, NULL) == FT_TT_OK,
          "returns OK when the firmware writes DONE");
    check(simulated_core.go_reads >= 5u, "it really polled rather than returning at once");

    printf("\nmailbox: completion poll TIMES OUT rather than spinning forever\n");
    reset_simulated_core();
    simulated_core.l1[signal_address] = (uint8_t)FT_TT_RUN_MSG_GO;
    simulated_core.go_reads_before_done = UINT32_MAX;
    memset(&state, 0xCD, sizeof(state));
    check(ft_tt_launch_wait_done(&noc, 1, 2, 30u, &state) == FT_TT_EIO,
          "returns FT_TT_EIO on timeout");
    check_u64(state.go_messages[0].signal, FT_TT_RUN_MSG_GO,
              "the timeout snapshot captured the stuck go signal");
    check(state.reset_register == simulated_core.soft_reset,
          "the timeout snapshot captured the reset register");

    printf("\nmailbox: a corrupted go signal is reported, not waited on\n");
    reset_simulated_core();
    simulated_core.l1[signal_address] = 0x33u;
    simulated_core.go_reads_before_done = UINT32_MAX;
    check(ft_tt_launch_wait_done(&noc, 1, 2, 2000u, NULL) == FT_TT_ERROR,
          "an unexpected signal value returns FT_TT_ERROR immediately");

    printf("\nmailbox: an unbounded wait is refused\n");
    reset_simulated_core();
    simulated_core.l1[signal_address] = (uint8_t)FT_TT_RUN_MSG_GO;
    check(ft_tt_launch_wait_done(&noc, 1, 2, 0u, NULL) == FT_TT_EINVAL,
          "timeout_ms == 0 is refused (a hang here costs a power cycle)");
}

static void test_state_snapshot_and_dump(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_launch_config config = baseline_config();
    ft_tt_launch_state state;

    printf("\nmailbox: state snapshot\n");
    reset_simulated_core();
    simulated_core.soft_reset = FT_TT_SOFT_RESET_ALL_RISCS & ~(uint32_t)FT_TT_SOFT_RESET_BRISC;
    check(ft_tt_launch_write_launch_message(&noc, 1, 2, &config) == FT_TT_OK, "launch message set");
    simulated_core.l1[FT_TT_MB_SUBORDINATE_SYNC_ADDR + 1u] = (uint8_t)FT_TT_RUN_SYNC_MSG_GO;
    memcpy(simulated_core.l1 + FT_TT_MB_WATCHER_ADDR +
               offsetof(ft_tt_watcher_msg, debug_waypoint[FT_TT_PROC_MATH0]),
           "GW  ", 4u);

    /* The wedge shape this snapshot exists to describe: BRISC out of reset with the four
     * subordinates held, and the Tensix instruction FIFO not accepting a write. MEASURED on
     * silicon 2026-08-17 — 140 healthy cores read 0x77 with invalid_instrn 0, the core the
     * 1x5 multicast relaunch arm wedged read exactly these two values. */
    simulated_core.instrn_buf_status = 0x66u;
    simulated_core.invalid_instrn = 1u;

    check(ft_tt_launch_read_state(&noc, 1, 2, &state) == FT_TT_OK, "reads");
    check_u64(state.reset_register, simulated_core.soft_reset, "reset register");
    check_u64(state.instrn_buf_status, 0x66u, "the Tensix instruction FIFO status is carried");
    check(state.instrn_buf_status != FT_TT_TENSIX_INSTRN_BUF_READY,
          "and 0x66 is NOT the ready value, so the wedge shape is distinguishable");
    check_u64(state.invalid_instrn, 1u, "and so is the invalid-instruction flag");
    check_u64(state.subordinate_sync[1], FT_TT_RUN_SYNC_MSG_GO, "TRISC0 sync byte");
    check_u64(state.kernel_config.enables, config.enables, "launch slot the read pointer selects");
    check(memcmp(state.watcher.debug_waypoint[FT_TT_PROC_MATH0].waypoint, "GW  ", 4u) == 0,
          "waypoint bytes");

    /* A garbage read pointer must not make the snapshot read a wild address. */
    simulated_core.l1[FT_TT_MB_LAUNCH_RD_PTR_ADDR] = 0xFFu;
    simulated_core.l1[FT_TT_MB_LAUNCH_RD_PTR_ADDR + 1u] = 0xFFu;
    simulated_core.l1[FT_TT_MB_LAUNCH_RD_PTR_ADDR + 2u] = 0xFFu;
    simulated_core.l1[FT_TT_MB_LAUNCH_RD_PTR_ADDR + 3u] = 0xFFu;
    check(ft_tt_launch_read_state(&noc, 1, 2, &state) == FT_TT_OK,
          "a garbage read pointer is clamped, not followed");

    /* A debug register that does not answer must not cost the MAILBOX snapshot, which is
     * the half a wedged core is being read for. It reports UNREAD instead. */
    simulated_core.refuse_debug_registers = 1;
    check(ft_tt_launch_read_state(&noc, 1, 2, &state) == FT_TT_OK,
          "a debug register that refuses does not abort the snapshot");
    check_u64(state.instrn_buf_status, FT_TT_TENSIX_INSTRN_BUF_UNREAD,
              "and it reports UNREAD rather than a value it did not read");
    check_u64(state.subordinate_sync[1], FT_TT_RUN_SYNC_MSG_GO,
              "while the mailbox half is still there");
    simulated_core.refuse_debug_registers = 0;

    printf("  (dump follows on stderr; it is inspected by eye, not asserted)\n");
    ft_tt_launch_dump_state(&state, 1, 2);
}

static void test_access_errors_propagate(void)
{
    ft_tt_noc_access noc = simulated_access();
    ft_tt_launch_config config = baseline_config();
    ft_tt_noc_access broken;

    printf("\nerror transparency: device failures are not swallowed\n");
    reset_simulated_core();
    simulated_core.fail_after_writes = 0;
    check(ft_tt_launch_write_launch_message(&noc, 1, 2, &config) == FT_TT_EIO,
          "a failing write surfaces as FT_TT_EIO");

    broken.write = NULL;
    broken.read = NULL;
    broken.user = NULL;
    check(ft_tt_launch_clear_mailbox(&broken, 1, 2) == FT_TT_EINVAL,
          "an accessor with no callbacks is refused");
    check(ft_tt_launch_read_state(&broken, 1, 2, NULL) == FT_TT_EINVAL, "a NULL out-state");
}

/* ------------------------------------------------------------------------- */
/* 9. Multi-core placement                                                   */
/* ------------------------------------------------------------------------- */

/** The topology this card reports, built by hand so coordinate derivation can be
 *  exercised with no device present. Same values as tests/test_noc.c. */
/** This test's own buffer, not a property of any part. The backend no longer carries a
 *  core count in a header (Phase 5): a set is bound to CALLER storage whose capacity comes
 *  from the live grid, so the number below only has to be at least as large as the
 *  synthetic grids these cases build. */
#define TEST_CORE_STORAGE_CAPACITY 256u
static ft_tt_launch_core_placement test_core_storage[TEST_CORE_STORAGE_CAPACITY];

static void test_core_set_construction(void)
{
    static ft_tt_launch_core_set set;
    ft_tt_launch_config config = baseline_config();
    ft_tt_launch_config broken = baseline_config();
    const ft_tt_grid_info grid = p150a_grid();
    const uint32_t live_cores = ft_tt_coord_tensix_core_count(&grid);
    unsigned index;

    printf("\nplacement group: construction\n");
    check(ft_tt_launch_core_set_add(&set, 1u, 2u, &config) == FT_TT_EINVAL,
          "a set that was never given storage refuses an add instead of writing through NULL");
    check_u64(ft_tt_launch_core_set_capacity(&set), 0u, "and reports no capacity");
    check(ft_tt_launch_core_set_init(&set, NULL, 4u) == FT_TT_EINVAL, "NULL storage is refused");
    check(ft_tt_launch_core_set_init(&set, test_core_storage, 0u) == FT_TT_EINVAL,
          "and so is capacity 0");

    check(ft_tt_launch_core_set_init(&set, test_core_storage, TEST_CORE_STORAGE_CAPACITY) ==
              FT_TT_OK,
          "storage binds");
    check_u64(ft_tt_launch_core_set_count(&set), 0u, "a fresh set is empty");
    check(ft_tt_launch_core_set_at(&set, 0u) == NULL, "and has no entry 0");

    check(ft_tt_launch_core_set_add(&set, 1u, 2u, &config) == FT_TT_OK, "first core added");
    check(ft_tt_launch_core_set_add(&set, 5u, 9u, &config) == FT_TT_OK, "second core added");
    check_u64(ft_tt_launch_core_set_count(&set), 2u, "count");
    check(ft_tt_launch_core_set_at(&set, 1u)->x == 5u && ft_tt_launch_core_set_at(&set, 1u)->y == 9u,
          "entry 1 is the second core's coordinate");
    check(ft_tt_launch_core_set_at(&set, 2u) == NULL, "one past the end is NULL");

    /* Two entries for one Tensix means the second launch message silently
     * overwrites the first and only one program ever runs. */
    check(ft_tt_launch_core_set_add(&set, 1u, 2u, &config) == FT_TT_EINVAL,
          "a duplicate coordinate is refused");
    check_u64(ft_tt_launch_core_set_count(&set), 2u, "and the refusal did not grow the set");

    broken.enables = 0u;
    check(ft_tt_launch_core_set_add(&set, 7u, 7u, &broken) == FT_TT_EINVAL,
          "a config the encoder rejects is refused at ADD time, not mid-stage");
    check(ft_tt_launch_core_set_add(NULL, 1u, 1u, &config) == FT_TT_EINVAL, "a NULL set");

    /* Coordinate first, config second: a program's text addresses are not known
     * until its images have been loaded, and loading needs the coordinate. */
    check(ft_tt_launch_core_set_add(&set, 9u, 9u, NULL) == FT_TT_OK,
          "a coordinate can be added with no config yet");
    check_u64(ft_tt_launch_core_set_at(&set, 2u)->config.enables, 0u,
              "and it comes back with nothing enabled");
    check(ft_tt_launch_core_set_configure(&set, 2u, &config) == FT_TT_OK,
          "configuring it once the images are placed works");
    check_u64(ft_tt_launch_core_set_at(&set, 2u)->config.enables, config.enables,
              "and the config took");
    check(ft_tt_launch_core_set_configure(&set, 3u, &config) == FT_TT_EINVAL,
          "configuring an entry past the end is refused");
    check(ft_tt_launch_core_set_configure(&set, 0u, &broken) == FT_TT_EINVAL,
          "and so is a config the encoder rejects");

    /* The capacity is whatever the CALLER sized it from, and here that is the live core
     * count of the synthetic grid — so a constant creeping back into ft_tt_launch.h goes
     * red here rather than passing on a part that happens to have 140 cores. */
    check(ft_tt_launch_core_set_init(&set, test_core_storage, live_cores) == FT_TT_OK,
          "a set sized from the LIVE core count");
    check_u64(ft_tt_launch_core_set_capacity(&set), live_cores, "reports that capacity");
    for (index = 0; index < TEST_CORE_STORAGE_CAPACITY; index++) {
        if (ft_tt_launch_core_set_add(&set, (uint16_t)(index % 100u), (uint16_t)(index / 100u),
                                      &config) != FT_TT_OK)
            break;
    }
    check_u64(index, live_cores, "exactly the live core count fits, no more and no fewer");
    check(ft_tt_launch_core_set_add(&set, 200u, 200u, &config) == FT_TT_ENOMEM,
          "one more than the caller sized for is refused");
}

static void test_core_set_derives_and_classifies(void)
{
    static ft_tt_launch_core_set set;
    const ft_tt_grid_info grid = p150a_grid();
    ft_tt_launch_config config = baseline_config();
    const ft_tt_launch_core_placement *first;

    printf("\nplacement group: coordinates come from the grid and are classified\n");
    (void)ft_tt_launch_core_set_init(&set, test_core_storage, TEST_CORE_STORAGE_CAPACITY);

    check(ft_tt_launch_core_set_add_tensix(&set, &grid, 0u, &config) == FT_TT_OK,
          "Tensix 0 derives and classifies");
    first = ft_tt_launch_core_set_at(&set, 0u);
    check(first != NULL && first->x == 1u && first->y == 2u,
          "and it is translated (1, 2), the first Tensix on this part");
    check(ft_tt_launch_core_set_add_tensix(&set, &grid, 139u, &config) == FT_TT_OK,
          "so does the last one, index 139");
    check(ft_tt_launch_core_set_add_tensix(&set, &grid, 140u, &config) != FT_TT_OK,
          "an index past the enabled count is refused, not clamped");
    check(ft_tt_launch_core_set_add_tensix(&set, NULL, 3u, &config) == FT_TT_EINVAL, "a NULL grid");
    check_u64(ft_tt_launch_core_set_count(&set), 2u, "only the two good ones are in the set");

    /* An unconfigured entry must not reach the device as a launch of nothing. */
    check(ft_tt_launch_core_set_add_tensix(&set, &grid, 5u, NULL) == FT_TT_OK,
          "a coordinate-only entry is allowed");
    {
        ft_tt_noc_access noc = simulated_access();
        unsigned failed = UINT32_MAX;

        reset_simulated_core();
        check(ft_tt_launch_write_launch_messages(&noc, &set, &failed) == FT_TT_EINVAL,
              "but staging the group refuses it rather than launching nothing");
        check_u64(failed, 2u, "naming the entry that was never configured");
    }
}

static void test_core_set_stage_kick_wait_park(void)
{
    static ft_tt_launch_core_set set;
    ft_tt_noc_access noc = simulated_access();
    ft_tt_launch_config config = baseline_config();
    const uint32_t signal_address =
        FT_TT_MB_GO_SLOT_ADDR(0) + (uint32_t)offsetof(ft_tt_go_msg, signal);
    unsigned failed = UINT32_MAX;

    printf("\nplacement group: stage every core, then kick, then wait\n");
    (void)ft_tt_launch_core_set_init(&set, test_core_storage, TEST_CORE_STORAGE_CAPACITY);
    (void)ft_tt_launch_core_set_add(&set, 1u, 2u, &config);
    (void)ft_tt_launch_core_set_add(&set, 3u, 2u, &config);
    (void)ft_tt_launch_core_set_add(&set, 5u, 2u, &config);

    reset_simulated_core();
    check(ft_tt_launch_write_launch_messages(&noc, &set, &failed) == FT_TT_OK,
          "all three launch messages written");
    check_u64(simulated_core.write_calls, 3u, "one write per core, and no more");

    check(ft_tt_launch_send_go_all(&noc, &set, &failed) == FT_TT_OK, "all three kicked");
    check_u64(simulated_core.l1[signal_address], FT_TT_RUN_MSG_GO, "the go signal is GO");

    simulated_core.go_reads = 0u;
    simulated_core.go_reads_before_done = 2u;
    check(ft_tt_launch_wait_done_all(&noc, &set, 2000u, &failed) == FT_TT_OK,
          "all three completed");

    check(ft_tt_launch_park_all(&noc, &set) == FT_TT_OK, "all three parked");
    check_u64(simulated_core.soft_reset, FT_TT_SOFT_RESET_ALL_RISCS,
              "and every RISC-V is back in reset");

    printf("\nplacement group: failures name the core they happened on\n");
    reset_simulated_core();
    simulated_core.fail_after_writes = 1; /* core 0's message lands, core 1's does not */
    failed = UINT32_MAX;
    check(ft_tt_launch_write_launch_messages(&noc, &set, &failed) == FT_TT_EIO,
          "a failing write stops the group");
    check_u64(failed, 1u, "and reports WHICH core failed");

    reset_simulated_core();
    simulated_core.l1[signal_address] = (uint8_t)FT_TT_RUN_MSG_GO;
    simulated_core.go_reads_before_done = UINT32_MAX;
    failed = UINT32_MAX;
    check(ft_tt_launch_wait_done_all(&noc, &set, 30u, &failed) == FT_TT_EIO,
          "a core that never finishes times out");
    check_u64(failed, 0u, "on the first core");
    check(ft_tt_launch_wait_done_all(&noc, &set, 0u, NULL) == FT_TT_EINVAL,
          "an unbounded group wait is refused, exactly like the single-core one");

    /* Cleanup must not stop at the first unreachable core, or one bad core leaves
     * the rest of the group running and still able to issue NoC transactions. */
    printf("\nplacement group: park_all tries every core even when one fails\n");
    reset_simulated_core();
    simulated_core.fail_after_writes = 0;
    check(ft_tt_launch_park_all(&noc, &set) == FT_TT_EIO, "the failure is reported");
    check_u64(simulated_core.write_calls, 3u, "but all three cores were tried");

    check(ft_tt_launch_park_all(&noc, NULL) == FT_TT_EINVAL, "a NULL set is refused");
    check(ft_tt_launch_send_go_all(NULL, &set, NULL) == FT_TT_EINVAL, "a NULL accessor is refused");
}

/* ------------------------------------------------------------------------- */
/* 10. The allocator floor above the loaded images                           */
/* ------------------------------------------------------------------------- */

static void test_l1_floor_above_loaded_images(void)
{
    ft_tt_noc_access noc = simulated_access();
    const ft_tt_grid_info grid = p150a_grid();
    ft_tt_image_placement placements[FT_TT_PROC_COUNT];
    ft_tt_allocator allocator;
    ft_tt_buffer buffer;
    synthetic_elf elf;
    uint32_t floor;

    printf("\nallocator floor: it must clear the images that are loaded\n");
    memset(placements, 0, sizeof(placements));

    check_u64(ft_tt_image_placement_l1_floor(NULL, 0u), FT_TT_L1_USABLE_BASE,
              "with nothing loaded the floor is the usable base");
    check_u64(ft_tt_image_placement_l1_floor(placements, FT_TT_PROC_COUNT), FT_TT_L1_USABLE_BASE,
              "and placements nothing was written into are ignored, not read as address 0");
    check_u64(ft_tt_image_placement_l1_floor(NULL, 3u), 0u, "a NULL array with a count is refused");

    /* A kernel image is the no-op case: its text lives in the tail of its own
     * firmware hole, below the usable base, so it cannot reach the arena. */
    reset_simulated_core();
    build_kernel_elf(&elf, FT_TT_MEM_TRISC0_FIRMWARE_BASE + 0x400u, 512u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_MATH0, FT_TT_IMAGE_KERNEL, elf.bytes,
                                  elf.size, &placements[FT_TT_PROC_MATH0]) == FT_TT_OK,
          "a kernel image loads");
    check_u64(ft_tt_image_placement_l1_floor(placements, FT_TT_PROC_COUNT), FT_TT_L1_USABLE_BASE,
              "a kernel below the usable base does not raise the floor");

    /* Firmware is the case that bites: BRISC's .data is staged AT 0x9F80, which is
     * exactly where a default allocator starts handing out addresses. */
    memset(placements, 0, sizeof(placements));
    reset_simulated_core();
    build_firmware_elf(&elf, FT_TT_PROC_DM0, 256u);
    check(ft_tt_launch_load_image(&noc, 1, 2, FT_TT_PROC_DM0, FT_TT_IMAGE_FIRMWARE, elf.bytes,
                                  elf.size, &placements[FT_TT_PROC_DM0]) == FT_TT_OK,
          "BRISC firmware loads");
    check(placements[FT_TT_PROC_DM0].highest_l1_address > FT_TT_L1_USABLE_BASE,
          "its .data image reaches ABOVE the allocator's own floor");

    floor = ft_tt_image_placement_l1_floor(placements, FT_TT_PROC_COUNT);
    check_u64(floor, placements[FT_TT_PROC_DM0].highest_l1_address,
              "so the derived floor is the top of that image");
    check((floor % FT_TT_L1_ALIGNMENT) == 0u, "and it is 16-byte aligned");

    /* The collision, shown rather than asserted: a DEFAULT allocator hands out an
     * address inside the firmware image that is about to run. */
    check(ft_tt_allocator_init(&allocator, NULL, &grid, 0u) == FT_TT_OK, "default allocator");
    check(ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, 64u, &buffer) == FT_TT_OK,
          "it allocates");
    check(buffer.address < (uint64_t)placements[FT_TT_PROC_DM0].highest_l1_address,
          "and the address it hands out lands INSIDE the loaded firmware image");

    check(ft_tt_allocator_init(&allocator, NULL, &grid, floor) == FT_TT_OK,
          "an allocator built with the derived floor");
    check(ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, 64u, &buffer) == FT_TT_OK,
          "still allocates");
    check(buffer.address >= (uint64_t)placements[FT_TT_PROC_DM0].highest_l1_address,
          "but now the first address is above every loaded image");

    (void)ft_tt_allocator_reset(&allocator);
    check(ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u, FT_TT_L1_USABLE_SIZE, &buffer) ==
              FT_TT_ENOMEM,
          "and the span that would have covered the image is now REFUSED");
    check(ft_tt_buffer_alloc(&allocator, FT_TT_MEM_SPACE_L1, 0u,
                             FT_TT_L1_USABLE_SIZE - (size_t)(floor - FT_TT_L1_USABLE_BASE),
                             &buffer) == FT_TT_OK,
          "while everything from just above the images to the top of L1 still fits");
}

/* ------------------------------------------------------------------------- */

int main(void)
{
    printf("Foundry Blackhole backend — Phase 2 launch tests (no device)\n");

    test_mailbox_layout();
    test_core_info_encoding();
    test_watcher_priming();
    test_phase2_layout_derivation();
    test_elf_parses_a_kernel();
    test_elf_rejections();
    test_elf_segment_limits();
    test_kernel_blob_format();
    test_kernel_image_placement();
    test_firmware_image_placement();
    test_placement_refusals();
    test_relocated_kernel_placement();
    test_text_slab_map();
    test_merged_region_refusals();
    test_launch_message_encoding();
    test_launch_message_refusals();
    test_cb_config_encoding();
    test_runtime_argument_planting();
    test_reset_control();
    test_mailbox_clear();
    test_firmware_handshake_is_not_vacuous();
    test_launch_message_write_and_go();
    test_completion_poll();
    test_state_snapshot_and_dump();
    test_access_errors_propagate();
    test_core_set_construction();
    test_core_set_derives_and_classifies();
    test_core_set_stage_kick_wait_park();
    test_l1_floor_above_loaded_images();

    printf("\n%d/%d checks failed\n", checks_failed, checks_run);
    return checks_failed == 0 ? 0 : 1;
}
