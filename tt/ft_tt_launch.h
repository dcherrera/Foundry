/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_launch.h
 * @brief Foundry's slow-dispatch run control for one Tensix: put binaries in L1,
 *   stage the launch message, take the RISC-Vs out of reset, kick, and find out
 *   whether it finished — with a timeout, because a poll that never returns here
 *   costs a physical power cycle (spec §7.3).
 *
 * Module prefix: ft_tt_launch_. Pure C99, libc only. Every device touch goes through
 * an ft_tt_noc_access vtable, so the whole layer runs against a plain array of host
 * memory in tests and against a TLB window on the card, with the same code.
 *
 * ## What actually runs on the core
 *
 * Foundry does not boot the Tensix bare. It loads Foundry's per-processor FIRMWARE
 * (tt/fw/ft_*_fw.c, built by `build_tensix_kernel.sh` alongside the kernel) and then
 * hands it work through the mailbox protocol that firmware and this file share
 * (ft_tt_mailbox.h). The five-line summary:
 *
 *   1. host asserts reset on all five RISC-Vs
 *   2. host writes firmware + kernel images into L1, clears the mailbox region
 *   3. host DEASSERTS BRISC ONLY — BRISC firmware releases the other four itself
 *      (fw/ft_brisc_fw.c boot(): it sets the subordinates' reset PCs and clears
 *      RISCV_DEBUG_REG_SOFT_RESET_0, tt-isa-documentation TensixTile/SoftReset.md)
 *   4. host writes the launch message, then the go message
 *   5. BRISC starts the TRISCs, they run the kernel, BRISC writes DONE back
 *
 * The full sequence, with the L1 map and the first-hardware-run order, is in
 * docs/tt_launch_notes.md. Read it before running any of this against the card.
 *
 * @status VERIFIED ON THE CARD. 2026-08-11: all 13 steps of docs/tt_launch_notes.md §6,
 *   including a byte-exact vector add as a full triad repeated 1000 times and eight
 *   discontiguous cores staged and kicked together. 2026-08-12: Tensix→OTHER-Tensix
 *   transactions in both directions, and `core_info` + `_prime_watcher()` bringing up the
 *   firmware's NoC sanitize — a bad endpoint is now a reported, stopped core instead of a
 *   hung NoC — on the firmware of that time; Foundry's
 *   own firmware (tt/fw/, 2026-10-08) has no watcher, see the core_info section. The
 *   mailbox protocol is ft_tt_mailbox.h, served by tt/fw/. Everything that touches the NoC is
 *   behind FT_TT_PHASE1B_NOC_ENABLED and is a no-op returning FT_TT_EPERM in a default
 *   build.
 * @issues
 *   - Kernel ELFs from build_tensix_kernel.sh are NOT execute-in-place: their text
 *     sits at a fixed L1 address. `kernel_text_offset` therefore encodes
 *     `entry - kernel_config_base` in unsigned wraparound arithmetic. See
 *     ft_tt_launch_encode_launch_message().
 *   - A kernel killed mid-flight can wedge a Tensix past every reset this file can issue.
 *     It cost Tensix 0 on 2026-08-12 and there is still no host-side recovery.
 * @todo None
 */

#ifndef FT_TT_LAUNCH_H
#define FT_TT_LAUNCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_coord.h"
#include "ft_tt_device.h"
#include "ft_tt_elf.h"
#include "ft_tt_mailbox.h"
#include "ft_tt_memmap.h"
#include "ft_tt_noc.h"
#include "ft_tt_tlb.h"

/* ================================================================
 * Core reset control register
 * ================================================================ */

/** Per-Tensix soft-reset register, addressed over the NoC like any other location:
 *  RISCV_DEBUG_REG_SOFT_RESET_0, tt-isa-documentation TensixTile/SoftReset.md. There is no reset ioctl:
 *  release is a read-modify-write here. */
#define FT_TT_TENSIX_SOFT_RESET_ADDR 0xFFB121B0u

/* Bit per processor, SoftReset.md. Bit 31 has no hardware effect there (bits >= 24 do
 * nothing); it is kept as a software marker for a staggered grid start. */
#define FT_TT_SOFT_RESET_BRISC   (1u << 11)
#define FT_TT_SOFT_RESET_TRISC0  (1u << 12)
#define FT_TT_SOFT_RESET_TRISC1  (1u << 13)
#define FT_TT_SOFT_RESET_TRISC2  (1u << 14)
#define FT_TT_SOFT_RESET_NCRISC  (1u << 18)
#define FT_TT_SOFT_RESET_STAGGERED_START (1u << 31)

#define FT_TT_SOFT_RESET_ALL_RISCS                                            \
    (FT_TT_SOFT_RESET_BRISC | FT_TT_SOFT_RESET_NCRISC | FT_TT_SOFT_RESET_TRISC0 | \
     FT_TT_SOFT_RESET_TRISC1 | FT_TT_SOFT_RESET_TRISC2)

/* The SAME register also resets the Tensix BACKEND — the matrix engine the three TRISCs
 * drive (SoftReset.md bits 0-10 and 15-17).
 *
 * ############################################################################
 * # DO NOT ASSERT THESE. Doing so CORRUPTS THE CORE'S COMPUTE, PERSISTENTLY. #
 * ############################################################################
 *
 * They are recorded because knowing they exist is worth having and because the next
 * person will otherwise reach for them exactly as this file did on 2026-08-12 —
 * reasoning, wrongly, that a "park" which leaves the matrix engine running is
 * indefensible. Measured on the card, on two virgin cores, with nothing else on the
 * card:
 *
 *   ops PASS (PCC 0.999978292) -> boot asserting these bits and run a vector add
 *   -> the vector add is WRONG (45-111 of 1024 elements) -> ops on that core now
 *   returns PCC -nan FOREVER, for every later program, from any tree.
 *
 * Remove the assert and the same core goes ops PASS -> vector add byte-exact 1024/1024
 * -> ops PASS again. The RISC-V reset alone (FT_TT_SOFT_RESET_ALL_RISCS) is correct and
 * is what the boot uses; the backend is brought up by the firmware's own initialisation,
 * and taking it through reset underneath that leaves it configured wrong.
 *
 * There is no FT_TT_SOFT_RESET_WHOLE_TENSIX, deliberately: a convenience macro for
 * "everything" is exactly the shape of the mistake.
 */
#define FT_TT_SOFT_RESET_UNPACKER 0x00000003u /* bits 0-1, two unpackers */
#define FT_TT_SOFT_RESET_PACKER   0x0000003Cu /* bits 2-5, four packers */
#define FT_TT_SOFT_RESET_MOVER    (1u << 6)
#define FT_TT_SOFT_RESET_SEARCH   (1u << 7) /* SoftReset.md: a third unpacker reset bit */
#define FT_TT_SOFT_RESET_GLUE     (1u << 8) /* SoftReset.md: TDMA-RISC and glue */
#define FT_TT_SOFT_RESET_THCON    (1u << 9)
#define FT_TT_SOFT_RESET_FPU      (1u << 10)
#define FT_TT_SOFT_RESET_SRCA_REG (1u << 15)
#define FT_TT_SOFT_RESET_SRCB_REG (1u << 16)
#define FT_TT_SOFT_RESET_DEST_REG (1u << 17)

/* ================================================================
 * Tensix instruction FIFO — READ ONLY, and the one thing a read CAN say
 * ================================================================
 *
 * Two more registers in the SAME 4 KiB debug block as the soft reset above
 * (`RISCV_DEBUG_REGS_START_ADDR` 0xFFB12000; tt-isa-documentation
 * TensixTile/BabyRISCV/PushTensixInstruction.md, "Debug registers"), so the
 * window that already reaches 0xFFB121B0 reaches these with no new plumbing.
 *
 * **Why they matter.** Writes to `INSTRN_BUF_BASE` (0xFFE40000) are appended to the Tensix
 * instruction FIFO, and a full FIFO stalls the pushing RISC-V until there is space
 * (PushTensixInstruction.md). BRISC firmware's boot pushes ZEROACC, SFPENCC, a NOP,
 * SFPLOADI, SFPCONFIG and four SEMINITs into **thread 0's** FIFO *before* it releases the
 * subordinates (fw/ft_brisc_fw.c initialise_tensix() then boot()). So a FIFO that cannot accept
 * a write stalls BRISC inside a single store, before the subordinates are ever released —
 * which is exactly the `0x00047000` fingerprint (BRISC deasserted by us, NCRISC/TRISC0..2
 * still held, go signal still INIT).
 *
 * `FT_TT_TENSIX_INSTRN_BUF_READY` (0x77) is the status value measured on an idle,
 * ready core of this card; RISCV_DEBUG_REG_INSTRN_BUF_STATUS's
 * fields are described in PushTensixInstruction.md.
 *
 * **READ these. Do not write `INSTRN_BUF_CTRL0/1`** — that pair is an override that pushes
 * instructions into the backend from outside the RISC-Vs, i.e. the same family as the
 * banned backend reset bits above, and nothing here needs it.
 */
#define FT_TT_TENSIX_INSTRN_BUF_STATUS_ADDR 0xFFB120A8u
#define FT_TT_TENSIX_INSTRN_BUF_READY 0x77u
/** Last instruction the backend rejected as invalid (a debug register in the same block;
 *  TensixTile/TileControlDebugStatus.md). */
#define FT_TT_TENSIX_DBG_INVALID_INSTRN_ADDR 0xFFB12094u
/** What a snapshot reports when the register did not answer. A snapshot that ABORTED
 *  because a debug register did not answer would be a worse diagnostic than one that says
 *  so, and the mailbox half is the half a wedged core is being read for. */
#define FT_TT_TENSIX_INSTRN_BUF_UNREAD 0xFFFFFFFFu

/* ================================================================
 * Device access
 * ================================================================ */

/**
 * How this layer reaches device memory. One indirection, deliberately: it is what
 * lets every byte of the protocol be exercised against a simulated L1 on the Mac
 * before any of it is pointed at the card.
 *
 * `addr` is a tile-local address (L1 from 0, or a register such as the soft-reset
 * one at 0xFFB121B0). Implementations must accept any 4-byte-aligned address and
 * length; they are not required to accept anything finer.
 *
 * Both callbacks return FT_TT_OK or a negative FT_TT_* code.
 */
typedef struct ft_tt_noc_access {
    int (*write)(void *user, uint16_t x, uint16_t y, uint64_t addr, const void *src, size_t bytes);
    int (*read)(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst, size_t bytes);
    void *user;
} ft_tt_noc_access;

/** Caller-owned storage tying an accessor to an ft_tt_noc path over a borrowed
 *  window. Declaring it rather than allocating it keeps the accessor allocation-free
 *  and its lifetime obvious: it lives exactly as long as the caller's stack frame.
 *
 *  Treat the path as private. It is here rather than hidden because the whole backend
 *  is allocation-free on the host, and because it is what makes this layer's device
 *  access and ft_tt_write_l1()'s the same code. */
typedef struct ft_tt_noc_window_binding {
    ft_tt_noc_path path;
} ft_tt_noc_window_binding;

/**
 * Build a NoC accessor that moves bytes through one claimed TLB window, re-aiming it
 * as the address moves (docs/tt_driver_notes.md §4 — a re-aim is three MMIO stores,
 * not an ioctl, though this first version goes through CONFIGURE_TLB for safety).
 *
 * The bytes move through ft_tt_noc_transfer_raw() at FT_TT_NOC_WIDTH_WORD32. This
 * layer cannot use ft_tt_write_l1(): it has to reach a 4-byte register at
 * 0xFFB121B0, which is in no memory space, and write 4-byte mailbox fields, which
 * are under the 16 B minimum that helper correctly enforces.
 *
 * **PHASE 1B.** In a default build every call through the returned accessor returns
 * FT_TT_EPERM without reaching the device, because aiming a window is fenced at the
 * kernel boundary (ft_tt_kmd.h, "THE NOC FENCE"). That fence is the reason this file can be
 * written and tested without risking the card.
 *
 * The window and the binding are borrowed, not owned: both must outlive the
 * accessor, and the caller still frees the window with ft_tt_tlb_window_free().
 *
 * @param ctx         open device context
 * @param window      a 2 MiB window claimed with ft_tt_tlb_window_alloc()
 * @param binding     caller-owned storage, must outlive `out_access`
 * @param out_access  receives the accessor
 * @return            FT_TT_OK or FT_TT_EINVAL
 *
 * Thread-safe: no (the window is stateful)   Reentrant: no   Signal-safe: no
 */
int ft_tt_launch_noc_over_window(ft_tt_ctx *ctx, ft_tt_tlb_window *window,
                                 ft_tt_noc_window_binding *binding, ft_tt_noc_access *out_access);

/* ================================================================
 * Per-processor L1 geography
 * ================================================================ */

/** "BRISC", "NCRISC", "TRISC0"... for logs. Never NULL. */
const char *ft_tt_processor_name(unsigned processor);

/** Soft-reset bit for an FT_TT_PROC_* index, or 0 if the index is out of range. */
uint32_t ft_tt_processor_reset_bit(unsigned processor);

/** L1 address of this processor's firmware text hole (FT_TT_MEM_*_FIRMWARE_BASE),
 *  or 0 if the index is out of range. */
uint32_t ft_tt_processor_firmware_base(unsigned processor);

/** Size of that hole in bytes, or 0. Firmware text over this will not fit. */
uint32_t ft_tt_processor_firmware_size(unsigned processor);

/** L1 address where this processor's firmware `.data` image is staged, which its
 *  `do_crt1()` copies into private local memory at boot
 *  (FT_TT_MEM_*_INIT_LOCAL_L1_BASE_SCRATCH; fw/ft_*_fw.c initialise_local_memory()).
 *  0 if the index is out of range. */
uint32_t ft_tt_processor_local_init_scratch(unsigned processor);

/* ================================================================
 * Image loading
 * ================================================================ */

/**
 * Which boot contract an ELF was linked under. This picks where its `.data` image is
 * staged, and getting it wrong puts the image somewhere nothing reads — a core that
 * boots into garbage globals with no error anywhere.
 */
typedef enum ft_tt_image_kind {
    /** Firmware. `.data` has a local-memory execution address and its ELF load
     *  address is meaningless: the firmware reads its image from
     *  FT_TT_MEM_*_INIT_LOCAL_L1_BASE_SCRATCH (fw/ft_*_fw.c initialise_local_memory()). */
    FT_TT_IMAGE_FIRMWARE = 0,
    /** Kernel, linked either way. Its `.data` load address IS meaningful — the kernel's
     *  own `_start` copies from the linker symbol `__kernel_data_lma` (fw/ft_*_entry.c)
     *  — so the ELF's p_paddr is used verbatim.
     *
     *  There is deliberately no separate "relocated kernel" kind. The ELF's own text
     *  address says which of the two legal regions it was linked into, and the ELF is
     *  the only party that cannot be wrong about that; a caller-supplied flag can. */
    FT_TT_IMAGE_KERNEL = 1
} ft_tt_image_kind;

/** Where an image ended up, for logging and for filling in the launch message. */
typedef struct ft_tt_image_placement {
    uint32_t entry_address; /**< what the firmware jumps to */
    uint32_t text_address;
    uint32_t text_bytes;
    uint32_t lowest_l1_address;  /**< inclusive */
    uint32_t highest_l1_address; /**< exclusive */
    uint32_t bytes_written;
} ft_tt_image_placement;

/**
 * Parse one ELF and write its loadable segments into the target core's L1.
 *
 * Refuses before writing anything if: the ELF is malformed; a firmware image's text
 * is not at this processor's firmware base or overflows its hole; any segment would
 * land outside L1. Refusing is the point — a bad placement is not a failed test on
 * this board, it is a wedged core.
 *
 * **The kernel rule, in one sentence: entirely inside its own firmware hole, OR
 * entirely inside its own kernel-config slab, never straddling.** Those are the only
 * two places a Tensix kernel is linked — the default link puts `.text` at
 * `__fw_export_text_end` in the tail of this processor's firmware hole, and a
 * RELOCATE_KERNEL_TEXT=1 link puts it in this processor's 12 KiB slab of the
 * KERNEL_CONFIG region, which is the only way an op bigger than `vector_add` fits
 * (ft_tt_op_l1_layout.h). The region is chosen by the ELF's own text address; every
 * other segment must then land inside the SAME one. So a loaded kernel can reach
 * neither a firmware image, nor another processor's slab, nor the CB arena.
 *
 * @param noc         device access
 * @param x, y        NoC coordinates of the target Tensix. Take them from ARC
 *                    telemetry or the coordinate module — NEVER a guess (spec §7.3)
 * @param processor   FT_TT_PROC_* index this image belongs to
 * @param kind        firmware or kernel; see ft_tt_image_kind
 * @param elf_data    the ELF blob (typically from ft_tt_kernels.h)
 * @param elf_size    its length
 * @param out_place   may be NULL
 * @return            FT_TT_OK, FT_TT_EINVAL, or whatever the accessor returned
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_launch_load_image(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, unsigned processor,
                            ft_tt_image_kind kind, const void *elf_data, size_t elf_size,
                            ft_tt_image_placement *out_place);

/**
 * The L1 floor to hand ft_tt_allocator_init() so no buffer it returns can land on a
 * loaded image: the highest address any of `placements` reached, rounded up to
 * FT_TT_L1_ALIGNMENT, and never below FT_TT_L1_USABLE_BASE.
 *
 * **This is not a formality.** The allocator's own floor is FT_TT_L1_USABLE_BASE
 * (0x9F80), which is exactly where the per-processor firmware `.data` staging regions
 * begin — so a default allocator will hand out the BRISC
 * `.data` image of the firmware currently running on the core. The Phase 2 tests dodge
 * that by hand, with a layout header that starts everything at 0x14000; this is the
 * number that makes the dodge unnecessary.
 *
 * Kernel images are a different case and come out as a no-op: they land in the tail of
 * their own processor's firmware hole, BELOW the usable base, so they cannot collide
 * with the arena and the clamp returns FT_TT_L1_USABLE_BASE.
 *
 * Entries with `bytes_written == 0` are skipped — that is what an unwritten
 * ft_tt_image_placement looks like, i.e. a processor this program leaves idle.
 *
 * @param placements  array indexed however the caller likes; NULL is allowed with
 *                    count 0
 * @param count       number of entries
 * @return            an L1 address suitable for ft_tt_allocator_init(), or 0 for a bad
 *                    argument
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t ft_tt_image_placement_l1_floor(const ft_tt_image_placement *placements, unsigned count);

/* ================================================================
 * Reset control
 * ================================================================ */

/** Read the raw soft-reset register of one Tensix. */
int ft_tt_launch_read_reset(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t *out_value);

/**
 * Hold `risc_mask` in reset, leaving the other bits as they are (OR in).
 */
int ft_tt_launch_assert_reset(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t risc_mask);

/**
 * Release `risc_mask` from reset, leaving the other bits as they are (AND out).
 *
 * For a Tensix worker pass FT_TT_SOFT_RESET_BRISC and nothing else: BRISC firmware
 * releases NCRISC and the TRISCs itself once it is running, and releasing them from
 * the host races that (fw/ft_brisc_fw.c boot()).
 *
 * @param staggered  set bit 31 so a whole grid starts skewed rather than at once.
 *                   Meaningless for a single core; pass false.
 */
int ft_tt_launch_deassert_reset(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                uint32_t risc_mask, bool staggered);

/* ================================================================
 * Launch message
 * ================================================================ */

/** One circular buffer for the host to describe to the firmware. Addresses and sizes
 *  are in BYTES; the device applies its own 16-byte-word shift (fw/ft_trisc_fw.c shifts
 *  by 4 for the compute threads; BRISC and NCRISC keep bytes). */
typedef struct ft_tt_launch_cb_entry {
    unsigned index;           /**< CB id, 0..FT_TT_NUM_CIRCULAR_BUFFERS-1 */
    uint32_t address;         /**< L1 byte address of the FIFO */
    uint32_t size_bytes;      /**< total FIFO size */
    uint32_t page_size_bytes; /**< one tile, for tile CBs */
    uint32_t page_count;      /**< number of pages the FIFO holds */
} ft_tt_launch_cb_entry;

/** Everything the host has to tell the firmware about one program on one core.
 *  Addresses here are ABSOLUTE L1 addresses; the encoder converts them to the
 *  base-relative offsets the wire format wants and refuses anything that will not
 *  fit its field. */
typedef struct ft_tt_launch_config {
    /** L1 address the whole message is relative to. Must be 16-byte aligned and at or
     *  above FT_TT_L1_USABLE_BASE. */
    uint32_t kernel_config_base;
    /** Per-processor kernel entry address, 0 for a processor this program leaves
     *  idle. Indexed by FT_TT_PROC_*. */
    uint32_t kernel_text_address[FT_TT_PROC_COUNT];
    /** Per-processor kernel text size in bytes. */
    uint32_t kernel_text_bytes[FT_TT_PROC_COUNT];
    /** Per-processor runtime-arg block address, 0 for none. Must be within 64 KiB
     *  above kernel_config_base — the wire field is 16 bits. */
    uint32_t runtime_args_address[FT_TT_PROC_COUNT];
    /** Bit `p` set enables processor `p`. Usually ft_tt_kernel_program_enables(). */
    uint32_t enables;
    /** L1 address of the local-CB config array, 0 for none. Same 64 KiB constraint. */
    uint32_t local_cb_config_address;
    /** Bit `i` set means CB `i` is configured. */
    uint64_t local_cb_mask;
    /** Which NoC BRISC's dataflow kernel uses, 0 or 1. */
    uint8_t brisc_noc_id;
} ft_tt_launch_config;

/**
 * Serialise a launch message into `buffer`, which must be at least
 * sizeof(ft_tt_kernel_config_msg) bytes.
 *
 * Nothing is written to the device and no state is touched, so this is the piece
 * that can be fully verified off-hardware — which is why it is a separate function
 * from ft_tt_launch_write_launch_message().
 *
 * Two encodings worth knowing about:
 *
 *  - `kernel_text_offset[p]` is `kernel_text_address[p] - kernel_config_base` in
 *    32-bit UNSIGNED WRAPAROUND. Our kernels are not execute-in-place: their text is
 *    linked at a fixed low L1 address (just above the firmware) while
 *    kernel_config_base sits in high, usable L1, so the difference is "negative".
 *    The firmware computes `kernel_config_base + kernel_text_offset` in uint32
 *    (fw/ft_*_fw.c run_launch()), which wraps back to exactly the right address.
 *    Well-defined, and the only encoding that works without execute-in-place kernels.
 *  - A processor with no runtime args gets FT_TT_RTA_NO_ARGS_SENTINEL rather than 0,
 *    which marks "no args" for the kernel's argument reader (ft_tt_mailbox.h).
 *
 * @return FT_TT_OK; FT_TT_EINVAL for a misaligned base, an out-of-range 16-bit
 *         offset, an `enables` bit with no text address, or too small a buffer
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_launch_encode_launch_message(const ft_tt_launch_config *config, void *buffer,
                                       size_t buffer_bytes);

/**
 * Serialise CB descriptors into the four-word-per-CB array the firmware's
 * set_up_local_circular_buffers() reads (fw/ft_*_fw.c):
 * word 0 address, word 1 size, word 2 page count, word 3 page size.
 *
 * The buffer is fully zeroed first, so unconfigured CBs read as zero.
 *
 * @param buffer_bytes  must be at least
 *                      FT_TT_NUM_CIRCULAR_BUFFERS * FT_TT_WORDS_PER_LOCAL_CB_CONFIG * 4
 * @param out_mask      receives the bitmask to put in ft_tt_launch_config.local_cb_mask
 * @return              FT_TT_OK, or FT_TT_EINVAL for a bad index or a short buffer
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_launch_encode_cb_config(const ft_tt_launch_cb_entry *entries, unsigned entry_count,
                                  void *buffer, size_t buffer_bytes, uint64_t *out_mask);

/* ================================================================
 * core_info — what the STAGED firmware's NoC sanitize resolved endpoints against
 * ================================================================
 *
 * HISTORICAL as of 2026-10-08. The firmware this was written for could, in WATCHER builds,
 * bounds-check every NoC transaction against these tables. Foundry's own firmware (tt/fw/)
 * and C kernels have no watcher and read none of this, so writing it is harmless and
 * reading it back reports nothing. Dead: remove it together with a card round (it changes
 * what the host writes to L1). */

/**
 * One non-Tensix endpoint the firmware should recognise, in the TRANSLATED coordinate
 * system — the one that goes on the wire (tt-isa-documentation NoC/Coordinates.md).
 */
typedef struct ft_tt_launch_non_worker_core {
    uint16_t x;
    uint16_t y;
    ft_tt_addressable_core_type type;
} ft_tt_launch_non_worker_core;

/**
 * Everything the host has to tell the firmware about the CHIP, as opposed to the
 * program. Written once per core, after the mailbox is cleared and before any kernel
 * runs.
 *
 * This existed for one feature of the staged firmware: **NoC sanitize** (WATCHER builds
 * only). Every data-movement read or write first resolved its endpoint through these
 * tables and bounds-checked the address; a violation was recorded in
 * `watcher.sanitize[noc]` and the core spun — **before** the transaction was issued.
 * That was the whole point: it turned a
 * bad endpoint from "hangs the NoC, costs a physical power cycle" into "one core stopped,
 * with the address written down".
 *
 * Leave this unwritten and the feature is worse than useless: zeroed tables make
 * `noc_size_x` 0, `virtual_end_x` underflows to 0xFFFFFFFF, and every coordinate
 * classifies as TENSIX — so a DRAM address is bounds-checked against L1 and a correct
 * program is stopped. Hence the seven `WATCHER_DISABLE_*` flags the build scripts pass
 * by default.
 */
typedef struct ft_tt_launch_core_info_config {
    uint8_t noc_size_x;         /**< NoC coordinate space, not core count */
    uint8_t noc_size_y;
    uint8_t worker_grid_size_x; /**< enabled Tensix columns */
    uint8_t worker_grid_size_y; /**< enabled Tensix rows */
    uint8_t absolute_logical_x; /**< THIS core's logical coordinate */
    uint8_t absolute_logical_y;
    uint32_t l1_unreserved_start;
    uint64_t noc_dram_addr_base;
    uint64_t noc_dram_addr_end;
    uint64_t noc_pcie_addr_base;
    uint64_t noc_pcie_addr_end;
    /** Borrowed, and may be NULL with count 0. Every slot NOT covered by this list is
     *  stamped with FT_TT_CORE_COORD_INVALID / UNKNOWN, never zeroed. */
    const ft_tt_launch_non_worker_core *virtual_non_worker_cores;
    unsigned virtual_non_worker_count;
} ft_tt_launch_core_info_config;

/**
 * Build a config for one Tensix from a live grid: every enabled DRAM bank's NoC ports in
 * TRANSLATED form, the NoC and worker grid sizes, and the DRAM and PCIe address windows
 * the firmware bounds-checks against.
 *
 * Coordinates come from ft_tt_coord against `grid`, never a literal — a wrong entry here
 * does not fail a check, it teaches the checker to approve a bad endpoint. So do the grid
 * sizes and the ports-per-bank count: they are read from the grid's architecture table,
 * not from a Blackhole macro, because a zeroed `noc_size_x` is not "empty" but WRONG (it
 * underflows `virtual_end_x` and makes every coordinate classify as TENSIX — dev notes §1)
 * and a constant one is wrong on any other part.
 *
 * **This is also where the kernels' compiled-in board shape meets the live one.** It calls
 * ft_tt_coord_check_kernel_build_assumptions() first and returns FT_TT_EIO if the card's
 * core or bank count is not the one `-DNUM_L1_BANKS` / `-DNUM_DRAM_BANKS` were built with.
 * This function is the single place that builds the device-visible view of the grid, which
 * is why the check lives here rather than at every call site.
 *
 * Ethernet is deliberately NOT registered: 8 banks x 3 ports already fills 24 of the 29
 * virtual slots the ABI provides, and Foundry never targets an ETH core. The consequence
 * is the safe direction — an ETH target would be REFUSED as an unknown endpoint, not
 * allowed. PCIe is left out for the same reason plus a harder one: ft_tt_coord offers no
 * TRANSLATED PCIe coordinate, because which of the two endpoints is live comes from a
 * telemetry tag ft_tt_grid_info does not carry.
 *
 * @param grid            from ft_tt_arc_telemetry_read_grid(); never a constructed one
 * @param logical_x, logical_y  THIS core's logical coordinate, for `absolute_logical_*`
 * @param l1_unreserved_start   first L1 address the host allocates from
 * @param storage         caller-owned array the config's table points into; must outlive
 *                        the config. FT_TT_LAUNCH_MAX_NON_WORKER_CORES is enough
 * @param storage_count   entries in `storage`
 * @param out_config      written only on success
 * @return                FT_TT_OK; FT_TT_EINVAL for a NULL argument; FT_TT_ENOMEM when
 *                        `storage` is too small; FT_TT_EIO when the live grid is not the
 *                        one the kernels were compiled for; or the coordinate module's code
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (logs on failure)
 */
#define FT_TT_LAUNCH_MAX_NON_WORKER_CORES FT_TT_MAX_VIRTUAL_NON_WORKER_CORES

int ft_tt_launch_core_info_from_grid(const ft_tt_grid_info *grid, uint32_t logical_x,
                                     uint32_t logical_y, uint32_t l1_unreserved_start,
                                     ft_tt_launch_non_worker_core *storage, unsigned storage_count,
                                     ft_tt_launch_core_info_config *out_config);

/**
 * Serialise a core_info message into `buffer`, which must be at least
 * sizeof(ft_tt_core_info_msg) bytes.
 *
 * Both endpoint tables are filled with the FT_TT_CORE_COORD_INVALID / UNKNOWN sentinel
 * first, then the caller's entries are written over the virtual one. The physical table
 * is left entirely sentinel: with coordinate translation on, only the translated table
 * is meaningful.
 *
 * Nothing is written to the device, so this is the piece that is fully verifiable
 * off-hardware.
 *
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL argument, too small a buffer, more entries
 *         than the ABI's 29 virtual slots, or a coordinate that does not fit a byte
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_launch_encode_core_info(const ft_tt_launch_core_info_config *config, void *buffer,
                                  size_t buffer_bytes);

/**
 * Encode and write core_info into the mailbox.
 *
 * ORDERING: after ft_tt_launch_clear_mailbox(), which zeroes the whole region and would
 * otherwise wipe this, and before the first go message. The firmware never writes this
 * field, so between those two points is the only requirement.
 */
int ft_tt_launch_write_core_info(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                 const ft_tt_launch_core_info_config *config);

/**
 * Make the watcher's report slots usable: `watcher.enable` = WatcherEnabled, and both
 * `watcher.sanitize[]` return codes = FT_TT_SANITIZE_OK.
 *
 * **Without this a NoC-sanitize violation stops the core and tells you nothing.** The
 * staged firmware only wrote a report into a slot that already read OK — the guard that
 * keeps the FIRST violation instead of the last — and ft_tt_launch_clear_mailbox() leaves
 * those slots at 0. Measured on the card 2026-08-12: a deliberate violation hung BRISC at
 * waypoint `NAWW` with `sanitize[0]` entirely zero. Foundry's firmware has no watcher. 
 *
 * ORDERING: after ft_tt_launch_clear_mailbox(), like core_info. Harmless on firmware
 * built without watcher — nothing reads the fields.
 */
int ft_tt_launch_prime_watcher(const ft_tt_noc_access *noc, uint16_t x, uint16_t y);

/** "OK", "noc addr overflow", ... for a sanitize return code (ft_tt_mailbox.h). Never NULL. */
const char *ft_tt_sanitize_return_code_string(uint16_t return_code);

/* ================================================================
 * Staging and kick
 * ================================================================ */

/**
 * Zero the whole mailbox region [FT_TT_MEM_MAILBOX_BASE, FT_TT_MEM_MAILBOX_END).
 *
 * Do this while every RISC-V is in reset and BEFORE loading firmware. It is what
 * makes `go_message_index`, `launch_msg_rd_ptr` and the subordinate-sync bytes start
 * from a known value instead of whatever the last program left there — and the
 * firmware indexes arrays with two of those.
 */
int ft_tt_launch_clear_mailbox(const ft_tt_noc_access *noc, uint16_t x, uint16_t y);

/**
 * Write RUN_MSG_INIT into go_messages[0] so the firmware's boot handshake is
 * observable.
 *
 * Call this after ft_tt_launch_clear_mailbox() and before deasserting BRISC.
 * Without it the handshake is vacuous and silently so: a cleared mailbox already
 * reads RUN_MSG_DONE, because DONE is 0 (ft_tt_mailbox.h). Waiting for DONE on a zeroed
 * mailbox succeeds instantly whether or not any firmware is running.
 */
int ft_tt_launch_prime_firmware_handshake(const ft_tt_noc_access *noc, uint16_t x, uint16_t y);

/**
 * Write `count` 32-bit runtime args at `l1_address`. Args are plain 4-byte words the
 * kernel reads through `rta_l1_base` (kernels/ft_tensix_kernel_args.h); there is no ioctl
 * and no descriptor. `l1_address` must be 4-byte aligned.
 */
int ft_tt_launch_plant_runtime_args(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                    uint32_t l1_address, const uint32_t *args, unsigned count);

/** Encode and write the launch message into mailbox launch slot 0. */
int ft_tt_launch_write_launch_message(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                      const ft_tt_launch_config *config);

/**
 * Wait for the firmware to finish booting: BRISC writes RUN_MSG_DONE to
 * go_messages[0] once it and all four subordinates have initialised (fw/ft_brisc_fw.c boot()).
 * Call this after deasserting BRISC and before the first launch message.
 */
int ft_tt_launch_wait_firmware_ready(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                                     uint32_t timeout_ms);

/** Write RUN_MSG_GO into go_messages[0]. The launch message must already be there. */
int ft_tt_launch_send_go(const ft_tt_noc_access *noc, uint16_t x, uint16_t y);

/* ================================================================
 * Multi-core placement
 * ================================================================ */

/** One core in a placement group: where it is, and what it was told to run. */
typedef struct ft_tt_launch_core_placement {
    uint16_t x;
    uint16_t y;
    ft_tt_launch_config config;
} ft_tt_launch_core_placement;

/**
 * A group of cores running the same program with per-core configuration.
 *
 * **The capacity is the CALLER'S array, not a constant in this header.** It used to be
 * `FT_TT_LAUNCH_MAX_CORES 140` — every Tensix on an unharvested p150a — and that is the
 * shape of mistake Phase 5's derive-from-live rule exists to stop: TT's 140→120
 * reduction is a compatibility measure rather than yield binning, so 120-core p150s are
 * a shipping configuration running this same software (dev notes §5). A group sized from
 * 140 does not fail there; it reserves storage for cores that are not present, and
 * anything that then loops to the capacity addresses them. Size it from
 * `ft_tt_coord_tensix_core_count(grid)` and the question cannot arise.
 *
 * Still allocation-free: the storage is the caller's, declared or carved out of one
 * array shared by several groups (ft_tt_group.h does exactly that). A set that was never
 * given storage has capacity 0 and refuses every add, rather than writing through NULL.
 *
 * Fields are public so the struct can be declared, not allocated. Read them through
 * ft_tt_launch_core_set_at() and _count().
 */
typedef struct ft_tt_launch_core_set {
    ft_tt_launch_core_placement *cores; /**< borrowed; must outlive the set */
    unsigned capacity;
    unsigned count;
} ft_tt_launch_core_set;

/**
 * Bind caller storage to a set and empty it.
 *
 * @param storage   `capacity` placements, owned by the caller
 * @param capacity  at least 1
 * @return          FT_TT_OK, or FT_TT_EINVAL for a NULL argument or capacity 0
 */
int ft_tt_launch_core_set_init(ft_tt_launch_core_set *set, ft_tt_launch_core_placement *storage,
                               unsigned capacity);

/** How many cores the set could hold, or 0 for NULL / unbound. */
unsigned ft_tt_launch_core_set_capacity(const ft_tt_launch_core_set *set);

/** Empty the set, keeping its storage. Safe on NULL. */
void ft_tt_launch_core_set_clear(ft_tt_launch_core_set *set);

/** How many cores are in the set, or 0 for NULL. */
unsigned ft_tt_launch_core_set_count(const ft_tt_launch_core_set *set);

/** The `index`-th placement, or NULL if there is no such entry. */
const ft_tt_launch_core_placement *ft_tt_launch_core_set_at(const ft_tt_launch_core_set *set,
                                                            unsigned index);

/**
 * Add one core at an already-derived TRANSLATED coordinate.
 *
 * A non-NULL config is validated here, by encoding it, so a group cannot be
 * half-staged before a bad field is discovered — the same "refuse before touching the
 * device" rule the rest of this file follows.
 *
 * `config` may be NULL, which is the normal case: a program's per-processor text
 * addresses are not known until its images have been LOADED, and loading needs the
 * coordinate. So build the group first, stage against it, then fill each entry in with
 * ft_tt_launch_core_set_configure(). An entry left unconfigured has `enables` 0 and is
 * refused by ft_tt_launch_write_launch_messages() rather than launching nothing.
 *
 * A coordinate already in the set is refused: two entries for one Tensix means the
 * second launch message overwrites the first, and the core runs whichever staged
 * last while the caller believes both ran.
 *
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL set, a duplicate coordinate, or a
 *         config ft_tt_launch_encode_launch_message() rejects; FT_TT_ENOMEM when the
 *         set is full
 *
 * Thread-safe: no (mutates the set)    Reentrant: no    Signal-safe: no
 */
int ft_tt_launch_core_set_add(ft_tt_launch_core_set *set, uint16_t x, uint16_t y,
                              const ft_tt_launch_config *config);

/**
 * Give the `index`-th entry its config, once its images have been placed. The
 * coordinate is fixed at add time — that is the half a mistake in cannot be undone —
 * and the config is the half that is not knowable until then.
 *
 * @return FT_TT_OK; FT_TT_EINVAL for a NULL argument, an index past the end, or a
 *         config the encoder rejects
 */
int ft_tt_launch_core_set_configure(ft_tt_launch_core_set *set, unsigned index,
                                    const ft_tt_launch_config *config);

/**
 * Add the `tensix_index`-th enabled Tensix of a live grid, deriving its TRANSLATED
 * coordinate and CLASSIFYING it before it can be used.
 *
 * Prefer this to ft_tt_launch_core_set_add(): it is the only entry point where a
 * coordinate that is not a Tensix cannot get into a launch. A wrong NoC endpoint does
 * not fail a write, it hangs the chip, and only a physical power cycle recovers it
 * (spec §7.3).
 *
 * @param grid    from ft_tt_arc_telemetry_read_grid(); never a constructed one
 * @param config  may be NULL — see ft_tt_launch_core_set_add()
 * @return        FT_TT_OK; FT_TT_EINVAL as above, or when the derived coordinate does
 *                not classify as a Tensix; FT_TT_ENOMEM; or the coordinate module's code
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_launch_core_set_add_tensix(ft_tt_launch_core_set *set, const ft_tt_grid_info *grid,
                                     uint32_t tensix_index, const ft_tt_launch_config *config);

/**
 * Write every core's launch message.
 *
 * STAGE THE WHOLE GROUP BEFORE KICKING ANY OF IT. A loop of
 * write-message-then-go over N cores is N sequential single-core runs wearing a
 * loop; the point of a group is that they overlap.
 *
 * @param out_failed_index  may be NULL; set to the core that failed
 * @return                  FT_TT_OK, or the first failure's code. Stops there —
 *                          the cores already staged are left staged
 */
int ft_tt_launch_write_launch_messages(const ft_tt_noc_access *noc,
                                       const ft_tt_launch_core_set *set,
                                       unsigned *out_failed_index);

/** Send RUN_MSG_GO to every core in the set, in order. Same failure semantics as
 *  ft_tt_launch_write_launch_messages(). */
int ft_tt_launch_send_go_all(const ft_tt_noc_access *noc, const ft_tt_launch_core_set *set,
                             unsigned *out_failed_index);

/**
 * Wait for every core to report RUN_MSG_DONE. `timeout_ms` bounds EACH core, not the
 * group, so the worst case is count * timeout_ms — and, as with the single-core wait,
 * 0 is refused rather than treated as "forever".
 *
 * A core that times out is dumped by ft_tt_launch_wait_done() before this returns.
 */
int ft_tt_launch_wait_done_all(const ft_tt_noc_access *noc, const ft_tt_launch_core_set *set,
                               uint32_t timeout_ms, unsigned *out_failed_index);

/**
 * Put every RISC-V of every core in the set back in reset.
 *
 * This is the cleanup call, so unlike the others it does NOT stop at the first
 * failure — a core left running is a core that can still issue NoC transactions, and
 * one unreachable core must not strand the rest. The first error is returned after
 * every core has been tried.
 */
int ft_tt_launch_park_all(const ft_tt_noc_access *noc, const ft_tt_launch_core_set *set);

/* ================================================================
 * Completion, and what to do when it does not come
 * ================================================================ */

/** Everything worth knowing about a core that did not finish. Read-only. */
typedef struct ft_tt_launch_state {
    uint32_t reset_register;
    uint8_t subordinate_sync[FT_TT_SUBORDINATE_MAP_SIZE];
    uint32_t launch_msg_rd_ptr;
    uint32_t go_message_index;
    ft_tt_go_msg go_messages[FT_TT_GO_MESSAGE_NUM_ENTRIES];
    ft_tt_kernel_config_msg kernel_config; /**< the slot launch_msg_rd_ptr selects */
    ft_tt_watcher_msg watcher;
    /** 0xFFB120A8. `FT_TT_TENSIX_INSTRN_BUF_READY` (0x77) is the only value at which the
     *  next firmware boot's `device_setup()` can push an instruction without stalling. */
    uint32_t instrn_buf_status;
    /** 0xFFB12094, for context only — an invalid instruction is a different fault. */
    uint32_t invalid_instrn;
} ft_tt_launch_state;

/** Snapshot a core's run state. Issues only reads, so it is safe on a core that is
 *  stuck — which is exactly when it is wanted. */
int ft_tt_launch_read_state(const ft_tt_noc_access *noc, uint16_t x, uint16_t y,
                            ft_tt_launch_state *out_state);

/**
 * Print a snapshot to stderr through ft_tt_log_error(), decoded: which RISC-Vs are
 * held in reset, what each subordinate's sync byte says, the go signal, and the
 * four-character waypoint each processor last recorded.
 *
 * Waypoints were written only by the staged firmware's WATCHER builds; Foundry's firmware
 * writes none, so the waypoint columns read "----" and the sync bytes are the diagnostic.
 */
void ft_tt_launch_dump_state(const ft_tt_launch_state *state, uint16_t x, uint16_t y);

/**
 * Poll go_messages[0] until the firmware writes RUN_MSG_DONE, or until `timeout_ms`
 * elapses.
 *
 * **This function is the reason Phase 2 cannot hang the machine.** There is no
 * untimed path: a kernel that never returns costs `timeout_ms` and a dump, not a
 * walk to the card. Pass a real bound; 0 is rejected.
 *
 * On timeout it snapshots the core, dumps it (see ft_tt_launch_dump_state) and
 * returns FT_TT_EIO. The core is left exactly as it was — recovery is the caller's
 * decision, because asserting reset under a stuck NoC transaction is itself a risk.
 *
 * A go signal that is neither GO nor DONE means something overwrote the mailbox;
 * that is reported as FT_TT_ERROR with the value logged.
 *
 * @param out_state  may be NULL; filled on timeout when provided
 * @return           FT_TT_OK; FT_TT_EIO on timeout; FT_TT_ERROR on a corrupt signal
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int ft_tt_launch_wait_done(const ft_tt_noc_access *noc, uint16_t x, uint16_t y, uint32_t timeout_ms,
                           ft_tt_launch_state *out_state);

#endif /* FT_TT_LAUNCH_H */
