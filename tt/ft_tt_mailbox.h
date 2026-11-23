/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_mailbox.h
 * @brief The mailbox at FT_TT_MEM_MAILBOX_BASE: how Foundry's host (ft_tt_launch.c) and
 *   Foundry's firmware (tt/fw/) hand a core its launch message, start it and learn it is done.
 *
 * THIS IS FOUNDRY'S PROTOCOL, NOT A HARDWARE CONTRACT. Both ends are Foundry code, and the
 * silicon reads none of it. Every offset, size and value is the one the shipped images and
 * the host were built with, so a rewrite of this file changes no byte of either; the static
 * assertions at the end pin each offset a consumer relies on.
 *
 * Layout rules:
 *   - fixed-width types only, so the riscv32 firmware and the 64-bit host agree;
 *   - every gap is an explicit `reserved_*` array, so no offset depends on implicit padding;
 *   - the launch message is packed: the firmware reads it with byte-safe loads, and its images
 *     were built that way;
 *   - spans neither side uses are reserved bytes, kept so every used field stays put.
 *
 * Handshake in brief (fw/ft_brisc_fw.c, ft_tt_launch.c):
 *   boot   host writes go signal INIT; BRISC firmware boots, writes DONE;
 *   launch host writes launch[launch_msg_rd_ptr], then go signal GO;
 *          BRISC drives each subordinate through its sync byte (GO, then DONE back);
 *          BRISC writes go signal DONE.
 *
 * @status Clean-room rewrite (plan 4.3), from Foundry's own consumers and the measured layout
 *   of the previous build.
 * @issues The debug block, the core-information block and the reserved print span are written
 *   or cleared by the host but read by no Foundry firmware. Dead: remove with a card round.
 * @todo None
 */

#ifndef FT_TT_MAILBOX_H
#define FT_TT_MAILBOX_H

#include <stddef.h>
#include <stdint.h>

#include "ft_tt_memmap.h"

/* ================================================================
 * Shape constants
 * ================================================================ */

/** Circular buffers per core: one bit each in the 64-bit local_cb_mask. Kept a bare literal:
 *  tools/emit_tensix_cb_descriptors.py reads it from this file with a regular expression. */
#define FT_TT_NUM_CIRCULAR_BUFFERS 64u

/** 32-bit words per circular buffer in the CB configuration array the launch message points
 *  at: address, size, page count, page size (all in bytes). */
#define FT_TT_WORDS_PER_LOCAL_CB_CONFIG 4u

/** Launch-message slots and go-message slots in the mailbox. */
#define FT_TT_LAUNCH_MSG_BUFFER_NUM_ENTRIES 8u
#define FT_TT_GO_MESSAGE_NUM_ENTRIES        9u

/** Per-core-type arrays in the launch message; Tensix is index 0. */
#define FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX 0u
#define FT_TT_PROGRAMMABLE_CORE_TYPE_COUNT  4u

/** Per-processor arrays in the launch message are FT_TT_PROC_COUNT long. */
#define FT_TT_MAX_PROCESSORS_PER_CORE_TYPE 5u

/** Shape values ft_tt_arch_check_headers() holds to the arch table. */
#define FT_TT_MAX_NUM_KERNELS                        3u
#define FT_TT_MAX_PROCESSORS_FOR_THREADING_VARIABLES 0u

/** Sync bytes, one per subordinate: NCRISC, TRISC0, TRISC1, TRISC2. */
#define FT_TT_SUBORDINATE_MAP_SIZE 4u

/** NoCs per core; one debug report slot each. */
#define FT_TT_MAX_NUM_NOCS_PER_CORE 2u

/** Required alignment of kernel_config_base. */
#define FT_TT_MAX_NOC_WRITE_ALIGNMENT 16u

/* ================================================================
 * Values
 * ================================================================ */

/** Go-signal values (ft_tt_go_msg.signal). DONE is 0, the same as a cleared mailbox, which is
 *  why the host primes the boot handshake with INIT. */
#define FT_TT_RUN_MSG_INIT 0x40u
#define FT_TT_RUN_MSG_GO   0x80u
#define FT_TT_RUN_MSG_DONE 0x00u

/** Subordinate sync-byte values, and the whole 4-byte sync word in two states. */
#define FT_TT_RUN_SYNC_MSG_GO                   0x80u
#define FT_TT_RUN_SYNC_MSG_DONE                 0x00u
#define FT_TT_RUN_SYNC_MSG_ALL_INIT             0x40404040u
#define FT_TT_RUN_SYNC_MSG_ALL_SUBORDINATES_DONE 0x00000000u

/** The only launch mode Foundry's firmware serves; any other value halts BRISC. */
#define FT_TT_DISPATCH_MODE_HOST 1u

/** rta_offset value meaning "this processor has no runtime arguments". */
#define FT_TT_RTA_NO_ARGS_SENTINEL 0xFFFFu

/** Value the host writes to watcher.enable when priming the debug block. */
#define FT_TT_WATCHER_ENABLED 3u

/** Codes the debug block's NoC report slots carry in return_code. NO_REPORT is a cleared
 *  slot; the host primes each slot to OK. */
#define FT_TT_SANITIZE_NO_REPORT                      0u
#define FT_TT_SANITIZE_OK                             2u
#define FT_TT_SANITIZE_NOC_ADDR_UNDERFLOW             3u
#define FT_TT_SANITIZE_NOC_ADDR_OVERFLOW              4u
#define FT_TT_SANITIZE_NOC_ADDR_ZERO_LENGTH           5u
#define FT_TT_SANITIZE_NOC_TARGET_INVALID_XY          6u
#define FT_TT_SANITIZE_NOC_MULTICAST_NON_WORKER       7u
#define FT_TT_SANITIZE_NOC_MULTICAST_INVALID_RANGE    8u
#define FT_TT_SANITIZE_NOC_ALIGNMENT                  9u
#define FT_TT_SANITIZE_NOC_MIXED_VIRTUAL_AND_PHYSICAL 10u
#define FT_TT_SANITIZE_INLINE_WRITE_DRAM_UNSUPPORTED  11u
#define FT_TT_SANITIZE_NOC_ADDR_MAILBOX               12u
#define FT_TT_SANITIZE_NOC_LINKED_TRANSACTION         13u
#define FT_TT_SANITIZE_L1_ADDR_OVERFLOW               14u
#define FT_TT_SANITIZE_ETH_SRC_L1_ADDR_OVERFLOW       15u
#define FT_TT_SANITIZE_ETH_DEST_L1_ADDR_OVERFLOW      16u
#define FT_TT_SANITIZE_CB_OUT_OF_BOUNDS               17u
#define FT_TT_SANITIZE_NOC_INVALID_TXN_ID             18u

/** Core-information block: table sizes, the unused-slot sentinel, and the tag that marks a
 *  worker core. A zeroed table entry would read as a real (0, 0) endpoint, so unused slots
 *  carry CORE_COORD_INVALID coordinates and the UNKNOWN type instead. */
#define FT_TT_MAX_PHYSICAL_NON_WORKER_CORES 35u
#define FT_TT_MAX_VIRTUAL_NON_WORKER_CORES  29u
#define FT_TT_MAX_HARVESTED_ON_AXIS         2u
#define FT_TT_CORE_COORD_INVALID            0xFFu
#define FT_TT_CORE_MAGIC_WORKER             0x50EC09A3u

/** Endpoint type in the core-information block (0 is a Tensix core). */
typedef enum ft_tt_addressable_core_type {
    FT_TT_ADDRESSABLE_CORE_DRAM = 3,
    FT_TT_ADDRESSABLE_CORE_UNKNOWN = 6
} ft_tt_addressable_core_type;

/* ================================================================
 * The launch message
 * ================================================================ */

/** A processor's runtime-argument offsets from kernel_config_base, in bytes. */
typedef struct ft_tt_rta_offset {
    uint16_t rta_offset;
    uint16_t crta_offset; /**< always FT_TT_RTA_NO_ARGS_SENTINEL from Foundry's host */
} ft_tt_rta_offset;

/** One launch: where the kernels and their configuration sit in L1, and who runs. Offsets are
 *  relative to kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_TENSIX]. */
typedef struct __attribute__((packed)) ft_tt_kernel_config_msg {
    uint32_t kernel_config_base[FT_TT_PROGRAMMABLE_CORE_TYPE_COUNT];        /* 0 */
    uint8_t reserved_0[8];                                                   /* 16 */
    uint16_t local_cb_offset;                                                /* 24 */
    uint8_t reserved_1[2];                                                   /* 26 */
    ft_tt_rta_offset rta_offset[FT_TT_MAX_PROCESSORS_PER_CORE_TYPE];         /* 28 */
    uint8_t mode;                                                            /* 48 */
    uint8_t reserved_2[3];                                                   /* 49 */
    uint32_t kernel_text_offset[FT_TT_MAX_PROCESSORS_PER_CORE_TYPE];         /* 52 */
    uint32_t kernel_text_size[FT_TT_MAX_PROCESSORS_PER_CORE_TYPE];           /* 72 */
    uint8_t reserved_3[12];                                                  /* 92 */
    uint64_t local_cb_mask;                                                  /* 104 */
    uint8_t brisc_noc_id;                                                    /* 112 */
    uint8_t brisc_noc_mode;                                                  /* 113 */
    uint8_t min_remote_cb_start_index; /**< FT_TT_NUM_CIRCULAR_BUFFERS: none */ /* 114 */
    uint8_t reserved_4[5];                                                   /* 115 */
    uint32_t enables; /**< bit FT_TT_PROC_* per processor that runs */      /* 120 */
    uint8_t reserved_5[19];                                                  /* 124 */
    uint8_t preload; /**< last byte; always 0 from Foundry's host */         /* 143 */
} ft_tt_kernel_config_msg;

/** One launch-message slot. */
typedef struct ft_tt_launch_msg {
    ft_tt_kernel_config_msg kernel_config;
} ft_tt_launch_msg;

/** One go-message slot. Only the last byte is used. */
typedef struct ft_tt_go_msg {
    uint8_t reserved[3];
    uint8_t signal; /**< FT_TT_RUN_MSG_* */
} ft_tt_go_msg;

/* ================================================================
 * The debug block (host-primed; no Foundry firmware reads it)
 * ================================================================ */

/** Four ASCII bytes per processor. */
typedef struct ft_tt_debug_waypoint_msg {
    uint8_t waypoint[4];
} ft_tt_debug_waypoint_msg;

/** One NoC report slot. */
typedef struct ft_tt_debug_sanitize_addr_msg {
    uint64_t noc_addr;
    uint32_t l1_addr;
    uint32_t len;
    uint16_t which_risc;
    uint16_t return_code; /**< FT_TT_SANITIZE_* */
    uint8_t is_multicast;
    uint8_t is_write;
    uint8_t is_target;
    uint8_t reserved;
} ft_tt_debug_sanitize_addr_msg;

typedef struct ft_tt_watcher_msg {
    uint32_t enable; /**< FT_TT_WATCHER_ENABLED once primed */
    ft_tt_debug_waypoint_msg debug_waypoint[FT_TT_MAX_PROCESSORS_PER_CORE_TYPE];
    ft_tt_debug_sanitize_addr_msg sanitize[FT_TT_MAX_NUM_NOCS_PER_CORE];
    uint8_t reserved[216];
} ft_tt_watcher_msg;

/** A reserved span the host clears with the mailbox and nothing else touches. */
typedef struct ft_tt_device_print_memory_layout {
    uint32_t reserved[255];
} ft_tt_device_print_memory_layout;

/* ================================================================
 * The core-information block (host-written; no Foundry firmware reads it)
 * ================================================================ */

/** One endpoint: coordinates and an ft_tt_addressable_core_type, one byte each. */
typedef struct ft_tt_addressable_core {
    uint8_t x;
    uint8_t y;
    uint8_t type;
} ft_tt_addressable_core;

typedef struct ft_tt_core_info_msg {
    uint64_t noc_pcie_addr_base;
    uint64_t noc_pcie_addr_end;
    uint64_t noc_dram_addr_base;
    uint64_t noc_dram_addr_end;
    ft_tt_addressable_core non_worker_cores[FT_TT_MAX_PHYSICAL_NON_WORKER_CORES];
    ft_tt_addressable_core virtual_non_worker_cores[FT_TT_MAX_VIRTUAL_NON_WORKER_CORES];
    uint8_t harvested_coords[FT_TT_MAX_HARVESTED_ON_AXIS];
    uint8_t virtual_harvested_coords[FT_TT_MAX_HARVESTED_ON_AXIS];
    uint8_t noc_size_x;
    uint8_t noc_size_y;
    uint8_t worker_grid_size_x;
    uint8_t worker_grid_size_y;
    uint8_t absolute_logical_x;
    uint8_t absolute_logical_y;
    uint8_t reserved_0[2];
    uint32_t l1_unreserved_start;
    uint32_t core_magic_number; /**< FT_TT_CORE_MAGIC_WORKER */
    uint8_t reserved_1[4];
} ft_tt_core_info_msg;

/* ================================================================
 * The mailbox
 * ================================================================ */

typedef struct ft_tt_mailboxes {
    uint8_t reserved_0[8];                                                    /* 0 */
    uint8_t subordinate_sync[FT_TT_SUBORDINATE_MAP_SIZE];                     /* 8 */
    uint32_t launch_msg_rd_ptr;                                               /* 12 */
    ft_tt_launch_msg launch[FT_TT_LAUNCH_MSG_BUFFER_NUM_ENTRIES];             /* 16 */
    ft_tt_go_msg go_messages[FT_TT_GO_MESSAGE_NUM_ENTRIES];                   /* 1168 */
    uint8_t reserved_1[12];                                                   /* 1204 */
    uint32_t go_message_index;                                                /* 1216 */
    uint8_t reserved_2[12];                                                   /* 1220 */
    ft_tt_watcher_msg watcher;                                                /* 1232 */
    ft_tt_device_print_memory_layout print_reserved;                          /* 1520 */
    uint8_t reserved_3[4];                                                    /* 2540 */
    ft_tt_core_info_msg core_info;                                            /* 2544 */
    uint8_t reserved_4[8];                                                    /* 2792 */
} ft_tt_mailboxes;

/* ================================================================
 * Mailbox addresses (L1, as uint32_t)
 * ================================================================ */

/** L1 address of `field` (a member path inside ft_tt_mailboxes, e.g. watcher.enable). */
#define FT_TT_MB_ADDR(field) \
    ((uint32_t)(FT_TT_MEM_MAILBOX_BASE + offsetof(ft_tt_mailboxes, field)))

#define FT_TT_MB_SUBORDINATE_SYNC_ADDR FT_TT_MB_ADDR(subordinate_sync)
#define FT_TT_MB_LAUNCH_RD_PTR_ADDR    FT_TT_MB_ADDR(launch_msg_rd_ptr)
#define FT_TT_MB_LAUNCH_ADDR           FT_TT_MB_ADDR(launch)
#define FT_TT_MB_GO_MESSAGES_ADDR      FT_TT_MB_ADDR(go_messages)
#define FT_TT_MB_GO_MESSAGE_INDEX_ADDR FT_TT_MB_ADDR(go_message_index)
#define FT_TT_MB_WATCHER_ADDR          FT_TT_MB_ADDR(watcher)
#define FT_TT_MB_CORE_INFO_ADDR        FT_TT_MB_ADDR(core_info)

/** Report slot of NoC `noc`. */
#define FT_TT_MB_WATCHER_SANITIZE_ADDR(noc) \
    ((uint32_t)(FT_TT_MB_ADDR(watcher.sanitize) +  \
                (uint32_t)(noc) * (uint32_t)sizeof(ft_tt_debug_sanitize_addr_msg)))

/** Launch-message slot `slot`. */
#define FT_TT_MB_LAUNCH_SLOT_ADDR(slot) \
    ((uint32_t)(FT_TT_MB_LAUNCH_ADDR + (uint32_t)(slot) * (uint32_t)sizeof(ft_tt_launch_msg)))

/** Go-message slot `slot`. */
#define FT_TT_MB_GO_SLOT_ADDR(slot) \
    ((uint32_t)(FT_TT_MB_GO_MESSAGES_ADDR + (uint32_t)(slot) * (uint32_t)sizeof(ft_tt_go_msg)))

/* ================================================================
 * The layout both ends depend on
 * ================================================================ */

FT_TT_STATIC_ASSERT(FT_TT_PROC_COUNT == FT_TT_MAX_PROCESSORS_PER_CORE_TYPE,
                    mailbox_one_slot_per_processor);
FT_TT_STATIC_ASSERT(FT_TT_NUM_CIRCULAR_BUFFERS <= 64u, mailbox_cb_count_fits_mask);
FT_TT_STATIC_ASSERT(FT_TT_PROC_COUNT == FT_TT_SUBORDINATE_MAP_SIZE + 1u, mailbox_four_subordinates);

FT_TT_STATIC_ASSERT(sizeof(ft_tt_rta_offset) == 4u, mailbox_rta_offset_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_kernel_config_msg) == 144u, mailbox_kernel_config_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_launch_msg) == 144u, mailbox_launch_slot_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_go_msg) == 4u, mailbox_go_slot_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_debug_sanitize_addr_msg) == 24u, mailbox_report_slot_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_watcher_msg) == 288u, mailbox_debug_block_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_device_print_memory_layout) == 1020u, mailbox_print_span_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_addressable_core) == 3u, mailbox_endpoint_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_core_info_msg) == 248u, mailbox_core_info_size);
FT_TT_STATIC_ASSERT(sizeof(ft_tt_mailboxes) == 2800u, mailbox_size);
FT_TT_STATIC_ASSERT(FT_TT_MEM_MAILBOX_BASE + sizeof(ft_tt_mailboxes) <= FT_TT_MEM_MAILBOX_END,
                    mailbox_fits_its_region);

FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, local_cb_offset) == 24u, kcm_local_cb_offset);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, rta_offset) == 28u, kcm_rta_offset);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, mode) == 48u, kcm_mode);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, kernel_text_offset) == 52u, kcm_text_offset);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, kernel_text_size) == 72u, kcm_text_size);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, local_cb_mask) == 104u, kcm_local_cb_mask);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, brisc_noc_id) == 112u, kcm_brisc_noc_id);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, brisc_noc_mode) == 113u, kcm_brisc_noc_mode);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, min_remote_cb_start_index) == 114u,
                    kcm_min_remote_cb_start_index);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, enables) == 120u, kcm_enables);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_kernel_config_msg, preload) == 143u, kcm_preload);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_go_msg, signal) == 3u, go_signal);

FT_TT_STATIC_ASSERT(offsetof(ft_tt_debug_sanitize_addr_msg, l1_addr) == 8u, report_l1_addr);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_debug_sanitize_addr_msg, which_risc) == 16u, report_which_risc);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_debug_sanitize_addr_msg, return_code) == 18u,
                    report_return_code);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_debug_sanitize_addr_msg, is_target) == 22u, report_is_target);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_watcher_msg, debug_waypoint) == 4u, debug_waypoints);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_watcher_msg, sanitize) == 24u, debug_reports);

FT_TT_STATIC_ASSERT(offsetof(ft_tt_core_info_msg, non_worker_cores) == 32u, info_physical_table);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_core_info_msg, virtual_non_worker_cores) == 137u,
                    info_virtual_table);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_core_info_msg, harvested_coords) == 224u, info_harvested);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_core_info_msg, noc_size_x) == 228u, info_noc_size_x);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_core_info_msg, absolute_logical_y) == 233u, info_logical_y);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_core_info_msg, l1_unreserved_start) == 236u,
                    info_l1_unreserved_start);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_core_info_msg, core_magic_number) == 240u, info_magic);

FT_TT_STATIC_ASSERT(offsetof(ft_tt_mailboxes, subordinate_sync) == 8u, mb_subordinate_sync);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_mailboxes, launch_msg_rd_ptr) == 12u, mb_launch_rd_ptr);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_mailboxes, launch) == 16u, mb_launch);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_mailboxes, go_messages) == 1168u, mb_go_messages);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_mailboxes, go_message_index) == 1216u, mb_go_message_index);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_mailboxes, watcher) == 1232u, mb_debug_block);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_mailboxes, print_reserved) == 1520u, mb_print_span);
FT_TT_STATIC_ASSERT(offsetof(ft_tt_mailboxes, core_info) == 2544u, mb_core_info);

#endif /* FT_TT_MAILBOX_H */
