/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_device.h
 * @brief Foundry's Tenstorrent Blackhole device surface: open/close a card,
 *   report what it is, expose the TLB register file, and answer capacity
 *   questions. Mirrors the shape of `foundry/cuda/fc_cuda.h` so a Foundry
 *   caller sees the same lifecycle on either backend.
 *
 * Module prefix: ft_tt_ (foundry tenstorrent). Pure C99, libc only — no C++,
 * no tt-metal, no tt-umd, nothing to dlopen. The kernel boundary is the
 * `ft_tt_kmd.h`; everything here is Foundry code (spec §3.4).
 *
 * @status Phase 1a. Device open/close, enumeration, driver version, BAR
 *   enumeration and the TLB register-file mapping are implemented and have run
 *   against the p150a at 12:00.0. NO NoC access of any kind exists in this
 *   phase: nothing configures a TLB window and nothing reads or writes device
 *   memory. `ft_tt_mem_info` therefore reports FT_TT_EUNAVAIL until Phase 1b
 *   fills the grid from ARC telemetry.
 * @issues None
 * @todo None
 *
 * Sources for the constants below (TLB facts: BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md;
 * the rest as noted per constant):
 *   SRC5 = foundry_blackhole_backend_spec.md
 */

#ifndef FT_TT_DEVICE_H
#define FT_TT_DEVICE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_arch.h"
#include "ft_tt_kmd.h"

/* ================================================================
 * Status codes
 * ================================================================ */

enum {
    FT_TT_OK = 0,
    FT_TT_ERROR = -1,     /**< unclassified failure */
    FT_TT_ENOMEM = -2,    /**< host allocation failed */
    FT_TT_EINVAL = -3,    /**< bad argument */
    FT_TT_EIO = -4,       /**< ioctl or mmap failed; errno was logged */
    FT_TT_ENODEV = -5,    /**< no such device, or not a Blackhole */
    FT_TT_EPERM = -6,     /**< blocked by the Phase 1a NoC fence */
    FT_TT_EUNAVAIL = -7,  /**< the answer is not known yet in this phase */
    FT_TT_ENOTFOUND = -8  /**< the named thing is not here — a program name the embedded
                           *   blob does not carry. Distinct from ENODEV on purpose: the
                           *   card is fine, the ARTIFACT does not contain what was asked
                           *   for, and the two want different answers from a caller */
};

/** Human-readable name for an FT_TT_* code. Never NULL. */
const char *ft_tt_error_string(int code);

/** Log to stderr with the `[ft_tt]` tag. Used across the backend so a failure
 *  can be reconstructed from the log alone (no debugger on the card host). */
void ft_tt_log_error(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

/**
 * Whether this process wants the MAXIMUM AI clock — true unless
 * `FT_TT_AI_CLK=min`.
 *
 * ONE definition of what that variable means, because there are now two ways to
 * ask the firmware for it: `SET_POWER_STATE` through tt-kmd, and our own
 * `POWER_SETTING` message on the userspace-driver path. A second `getenv` in
 * the second place is how two arms of an A/B end up at different clocks with
 * nothing in the log saying so — and every figure recorded before 2026-08-15
 * was taken at 800 MHz for exactly that class of reason.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no (getenv)
 */
bool ft_tt_ai_clock_wants_maximum(void);

/* ================================================================
 * Host-side device map — the parts of BAR0 Foundry uses
 * ================================================================ */

/** TLB configuration register file, at this offset into BAR0 (BlackholeA0/PCIExpressTile/HostToDeviceTLBs.md).
 *  Writing three 32-bit words here re-aims a window with no ioctl — the
 *  load-bearing performance fact of the whole backend (docs/tt_driver_notes.md
 *  §4). Phase 1a only READS it. */
#define FT_TT_TLB_REGS_OFFSET 0x1FC00000u
#define FT_TT_TLB_REGS_LEN    0x00001000u /* covers every TLB register */

/** Bytes of register per window, same for 2 MiB and 4 GiB kinds. */
#define FT_TT_TLB_REG_STRIDE 12u

/** Architectural window counts. The *usable* counts come from the device at
 *  runtime (ft_tt_tlb_counts) — these are for cross-checking, not for sizing. */
#define FT_TT_TLB_2M_WINDOW_COUNT 202u /* per HostToDeviceTLBs.md */
#define FT_TT_TLB_4G_WINDOW_COUNT 8u   /* none exist without BAR4 */
#define FT_TT_TLB_TOTAL_WINDOW_COUNT (FT_TT_TLB_2M_WINDOW_COUNT + FT_TT_TLB_4G_WINDOW_COUNT)

/** The driver claims the topmost 2 MiB window for its own NoC access at init
 *  (the driver's `KERNEL_TLB_INDEX`, set at driver init), leaving 201
 *  for userspace. It is never handed out by ALLOCATE_TLB. */
#define FT_TT_TLB_KERNEL_WINDOW_INDEX (FT_TT_TLB_2M_WINDOW_COUNT - 1u)

/** The first 32 windows also carry a strided-multicast register, in a second
 *  bank after all the per-window registers. */
#define FT_TT_TLB_STRIDED_WINDOW_COUNT 32u
#define FT_TT_TLB_STRIDED_REGS_OFFSET (FT_TT_TLB_TOTAL_WINDOW_COUNT * FT_TT_TLB_REG_STRIDE)

/** Tensix rows per column, from tt-kmd's view of the part (SRC5 §2.1).
 *
 *  **This is a CROSS-CHECK, not a source.** The live row count comes from the
 *  selected architecture table (`ft_tt_coord_tensix_row_count()`), and
 *  `ft_tt_arch_check_headers()` refuses to run if this macro disagrees with it.
 *  Never size a group or a loop from this — see the derive-from-live rule in
 *  the plan's Phase 5. */
#define FT_TT_TENSIX_ROWS_PER_COLUMN 10u

/** ARC tile, NoC0 coordinates. The kernel driver and the vendor host library both place the ARC tile at (8, 0). NOT a probed value — the kernel reads
 *  telemetry from exactly this endpoint at every probe.
 *
 *  Deliberately kept as a SECOND copy of the arch table's `fixed.arc`, sourced
 *  from tt-kmd rather than tt-umd. `ft_tt_arch_check_headers()` compares the
 *  two, so the duplication is an enforced agreement between two independent
 *  upstream files rather than a drift hazard. */
#define FT_TT_ARC_NOC0_X 8u
#define FT_TT_ARC_NOC0_Y 0u

/* ================================================================
 * Types
 * ================================================================ */

/** Per-card context: one open `/dev/tenstorrent/N`, its cached attributes, and
 *  the mapped TLB register file. Opaque — see ft_tt_device.c. One instance per
 *  card, mirroring one fc_cuda_ctx per K80 die. */
typedef struct ft_tt_ctx ft_tt_ctx;

/** What the card is. Filled from GET_DEVICE_INFO + GET_DRIVER_INFO. */
typedef struct ft_tt_device_info {
    uint64_t chip_arch;      /**< enum ft_tt_kmd_arch */
    uint16_t pci_vendor_id;
    uint16_t pci_device_id;
    uint16_t pci_subsystem_vendor_id;
    uint16_t pci_subsystem_id;
    uint16_t pci_domain;
    uint8_t pci_bus;
    uint8_t pci_device;
    uint8_t pci_function;
    uint8_t driver_version_major;
    uint8_t driver_version_minor;
    uint8_t driver_version_patch;
    uint32_t driver_api_version;
} ft_tt_device_info;

/** TLB windows this card actually has, derived from live BAR lengths. */
typedef struct ft_tt_tlb_counts {
    uint32_t window_2m_total;     /**< as the driver counts them */
    uint32_t window_2m_available; /**< minus the driver's own window */
    uint32_t window_4g_total;     /**< 0 on a board with no BAR4 */
} ft_tt_tlb_counts;

/**
 * Grid topology, the only safe source of which is ARC telemetry (spec §3.4;
 * GET_HARVESTING is a no-op and probing hangs the chip). `valid` is false
 * until ft_tt_arc_telemetry_read_grid() has succeeded, which cannot happen in
 * Phase 1a — see ft_tt_arc_telemetry.h.
 *
 * `arch` is what makes the telemetry masks MEAN anything: the same
 * `tensix_enabled_col` word denotes different columns on different parts, and
 * on a row-harvesting part it denotes rows. Carrying it in the grid rather
 * than passing it beside every call is what keeps `ft_tt_coord`'s signatures
 * unchanged and makes "a grid with no architecture" a refusal instead of a
 * default. It is installed by ft_tt_grid_set(); a NULL is refused by
 * ft_tt_coord_check_grid().
 */
typedef struct ft_tt_grid_info {
    bool valid;
    const ft_tt_arch *arch;
    uint32_t tensix_enabled_col; /**< bitmask, one bit per Tensix column */
    uint32_t enabled_eth;
    uint32_t enabled_gddr;
    uint32_t enabled_l2cpu;
    uint32_t noc_translation_enabled;
    uint32_t tensix_column_count; /**< popcount(tensix_enabled_col) */
    uint32_t tensix_core_count;   /**< columns x FT_TT_TENSIX_ROWS_PER_COLUMN */
    uint32_t dram_bank_count;     /**< popcount(enabled_gddr) */
} ft_tt_grid_info;

/* ================================================================
 * Lifecycle
 * ================================================================ */

/**
 * Open `/dev/tenstorrent/<device_index>`, enumerate it, and map the TLB
 * register file. Refuses anything that is not a Blackhole.
 *
 * Nothing on the device is written and no NoC transaction is issued. The BAR0
 * NoC window aperture is deliberately NOT mapped — see ft_tt_tlb_registers().
 *
 * @param device_index  0 for the first card
 * @return              owned context, or NULL (reason logged). Free with
 *                      ft_tt_ctx_free().
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
ft_tt_ctx *ft_tt_init_device(int device_index);

/**
 * Unmap the register file and close the device. Safe on NULL. Any TLB windows
 * still allocated through this context are released by the driver when the fd
 * closes, but callers should free them explicitly (ft_tt_tlb_window_free).
 *
 * Thread-safe: no (frees `ctx`)    Reentrant: no    Signal-safe: no
 */
void ft_tt_ctx_free(ft_tt_ctx *ctx);

/** True if `ctx` is non-NULL and fully initialized. */
bool ft_tt_ctx_available(const ft_tt_ctx *ctx);

/** The `/dev/tenstorrent/N` index this context is bound to, or -1. */
int ft_tt_ctx_device_index(const ft_tt_ctx *ctx);

/** The device file descriptor, owned by `ctx`. -1 if unavailable. For the TLB
 *  layer's mmap calls; not for general use. */
int ft_tt_ctx_fd(const ft_tt_ctx *ctx);

/** The kernel-boundary handle, owned by `ctx`. NULL if unavailable. */
ft_tt_kmd_device *ft_tt_ctx_kmd_device(const ft_tt_ctx *ctx);

/**
 * The architecture table selected for this card at open time.
 *
 * Never NULL on an open context: `ft_tt_init_device()` refuses a card whose
 * architecture has no verified table, so a context that exists has one. This is
 * the handle every layer above should ask for geometry instead of naming a
 * Blackhole constant.
 *
 * @return the table, or NULL if `ctx` is NULL.
 */
const ft_tt_arch *ft_tt_ctx_arch(const ft_tt_ctx *ctx);

/* ================================================================
 * Enumeration
 * ================================================================ */

/**
 * Copy the cached device identity. No ioctl is issued.
 * @return FT_TT_OK, or FT_TT_EINVAL.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_device_get_info(const ft_tt_ctx *ctx, ft_tt_device_info *out_info);

/**
 * Copy the cached BAR resources (mmap offsets and lengths). A length of 0 means
 * the BAR is absent, which is the expected state of BAR4 here (spec §2.3).
 * @return FT_TT_OK, or FT_TT_EINVAL.
 */
int ft_tt_device_get_resources(const ft_tt_ctx *ctx, ft_tt_kmd_resources *out_resources);

/**
 * Copy the TLB window counts derived from those BAR lengths.
 * @return FT_TT_OK, or FT_TT_EINVAL.
 */
int ft_tt_tlb_counts_get(const ft_tt_ctx *ctx, ft_tt_tlb_counts *out_counts);

/* ================================================================
 * TLB register file
 * ================================================================ */

/**
 * Pointer to the mapped TLB configuration register file (BAR0 + 0x1FC00000),
 * the mechanism Phase 1b uses to slide a window with three stores instead of an
 * ioctl. Window `id`'s registers start at word index `id * 12 / 4`.
 *
 * These are NoC2AXI configuration registers in BAR0, not a NoC aperture:
 * reading them issues no NoC transaction. In Phase 1a the mapping is READ-ONLY
 * (PROT_READ); building with FT_TT_PHASE1B_NOC_ENABLED makes it writable.
 *
 * The rest of BAR0 is intentionally left unmapped. Its low 202 x 2 MiB is the
 * TLB window aperture, and a stray store there is a NoC transaction to whatever
 * that window happens to be aimed at — the failure mode that costs a power
 * cycle. Phase 1b maps each window individually via ALLOCATE_TLB, which is all
 * it needs, so the whole-BAR mapping in docs/tt_driver_notes.md §7 step 3 is
 * not required and is not done.
 *
 * @return the mapping, or NULL if unavailable. Valid until ft_tt_ctx_free().
 */
const volatile uint32_t *ft_tt_tlb_registers(const ft_tt_ctx *ctx);

/** Bytes mapped at ft_tt_tlb_registers(), or 0. */
size_t ft_tt_tlb_registers_size(const ft_tt_ctx *ctx);

/* ================================================================
 * Capacity
 * ================================================================ */

/**
 * Copy the grid topology. `valid` is false until ARC telemetry has been read.
 * @return FT_TT_OK, or FT_TT_EINVAL.
 */
int ft_tt_grid_get(const ft_tt_ctx *ctx, ft_tt_grid_info *out_grid);

/** Install a grid read by the telemetry module. Recomputes the derived counts.
 *  @return FT_TT_OK, or FT_TT_EINVAL. */
int ft_tt_grid_set(ft_tt_ctx *ctx, const ft_tt_grid_info *grid);

/**
 * Device DRAM free/total in bytes — the TT analogue of fc_mem_info_ctx().
 * Total is `dram_bank_count x FT_TT_MEM_DRAM_SIZE`; free is total minus what
 * this context has allocated and not freed.
 *
 * Bank count is only knowable from ARC telemetry, so in Phase 1a this returns
 * **FT_TT_EUNAVAIL and writes nothing**. It does not guess, and it does not
 * report the nominal 8 banks as if it had measured them.
 *
 * @param free_bytes   may be NULL
 * @param total_bytes  may be NULL
 * @return             FT_TT_OK, FT_TT_EINVAL, or FT_TT_EUNAVAIL
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int ft_tt_mem_info(const ft_tt_ctx *ctx, size_t *free_bytes, size_t *total_bytes);

/** Bytes of device memory this context has allocated and not yet freed. Moved
 *  by ft_tt_ctx_note_device_alloc(), which the buffer allocator calls. */
size_t ft_tt_ctx_alloc_bytes(const ft_tt_ctx *ctx);

/**
 * Adjust the device-memory tally that ft_tt_mem_info() subtracts from total.
 * Called by ft_tt_buffer.c on every DRAM allocation and free; not for general
 * use. A negative delta larger than the tally is clamped to zero and logged —
 * an underflow means a double free, which is worth seeing.
 *
 * @param delta_bytes  positive on allocate, negative on free
 */
void ft_tt_ctx_note_device_alloc(ft_tt_ctx *ctx, int64_t delta_bytes);

/** Live TLB windows currently held by this context. */
uint32_t ft_tt_ctx_live_window_count(const ft_tt_ctx *ctx);

/** Adjust the live-window tally. Called by the TLB layer; not for general use.
 *  @param delta  +1 on allocate, -1 on free */
void ft_tt_ctx_note_window(ft_tt_ctx *ctx, int delta);

#endif /* FT_TT_DEVICE_H */
