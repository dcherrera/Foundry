/* SPDX-License-Identifier: AGPL-3.0-or-later */
/**
 * @file ft_tt_kmd_abi.h
 * @brief The Tenstorrent kernel driver's ioctl ABI on /dev/tenstorrent/N, as Foundry names:
 *   request codes, mapping ids, power-word bits and the exact byte layout of every argument
 *   struct Foundry passes. Numbers only; no behaviour. ft_tt_kmd.c is the one consumer that
 *   issues these on the backend path; card tests that drive the driver directly also use them.
 *
 * Little-endian, natural alignment. Every struct is the WHOLE argument the ioctl pointer
 * refers to, and every size and load-bearing offset is checked at compile time below, so a
 * layout slip is a build error rather than a wrong request on the wire.
 *
 * @status Covers the eight commands Foundry issues: device info, query mappings, driver info,
 *   allocate / free / configure window, close-time NoC write, power state.
 * @issues None
 * @todo None
 */

#ifndef FT_TT_KMD_ABI_H
#define FT_TT_KMD_ABI_H

#include <stddef.h>
#include <stdint.h>

/* ================================================================
 * Request codes: (0xFA << 8) | n — Linux _IO(0xFA, n), no size or direction bits
 * ================================================================ */

#define FT_TT_KMD_IOCTL_MAGIC 0xFAu
#define FT_TT_KMD_IOCTL_CODE(command) ((unsigned long)((FT_TT_KMD_IOCTL_MAGIC << 8) | (command)))

#define FT_TT_KMD_IOCTL_GET_DEVICE_INFO  FT_TT_KMD_IOCTL_CODE(0u)
#define FT_TT_KMD_IOCTL_QUERY_MAPPINGS   FT_TT_KMD_IOCTL_CODE(2u)
#define FT_TT_KMD_IOCTL_GET_DRIVER_INFO  FT_TT_KMD_IOCTL_CODE(5u)
#define FT_TT_KMD_IOCTL_ALLOCATE_WINDOW  FT_TT_KMD_IOCTL_CODE(11u)
#define FT_TT_KMD_IOCTL_FREE_WINDOW      FT_TT_KMD_IOCTL_CODE(12u)
#define FT_TT_KMD_IOCTL_CONFIGURE_WINDOW FT_TT_KMD_IOCTL_CODE(13u)
/** Register a NoC write the driver performs when the fd is closed ("noc cleanup"). */
#define FT_TT_KMD_IOCTL_NOC_CLEANUP      FT_TT_KMD_IOCTL_CODE(14u)
#define FT_TT_KMD_IOCTL_SET_POWER_STATE  FT_TT_KMD_IOCTL_CODE(15u)
/* Used only by tools and card tests, each with its own argument layout or the one below. */
#define FT_TT_KMD_IOCTL_ALLOCATE_DMA_BUF FT_TT_KMD_IOCTL_CODE(3u)
#define FT_TT_KMD_IOCTL_RESET_DEVICE     FT_TT_KMD_IOCTL_CODE(6u)
#define FT_TT_KMD_IOCTL_SMC_MESSAGE      FT_TT_KMD_IOCTL_CODE(17u)

/** The ioctl API version this code speaks. */
#define FT_TT_KMD_ABI_VERSION 2u

/* ================================================================
 * Identity
 * ================================================================ */

#define FT_TT_KMD_PCI_DEVICE_ID_BLACKHOLE 0xB140u
#define FT_TT_KMD_PCI_DEVICE_ID_WORMHOLE  0x401Eu

/* ================================================================
 * Mapping ids (QUERY_MAPPINGS). resource0 = BAR0, resource1 = BAR2, resource2 = BAR4.
 * ================================================================ */

#define FT_TT_KMD_MAPPING_UNUSED       0u
#define FT_TT_KMD_MAPPING_RESOURCE0_UC 1u
#define FT_TT_KMD_MAPPING_RESOURCE0_WC 2u
#define FT_TT_KMD_MAPPING_RESOURCE1_UC 3u
#define FT_TT_KMD_MAPPING_RESOURCE1_WC 4u
#define FT_TT_KMD_MAPPING_RESOURCE2_UC 5u
#define FT_TT_KMD_MAPPING_RESOURCE2_WC 6u

/** How many mapping records Foundry asks for: one UC and one WC per resource. */
#define FT_TT_KMD_MAPPING_QUERY_COUNT 6u

/* ================================================================
 * Power word (SET_POWER_STATE)
 * ================================================================ */

#define FT_TT_KMD_POWER_FLAG_MAX_AI_CLOCK   0x0001u /**< 1 = max AI clock, 0 = min */
#define FT_TT_KMD_POWER_FLAG_MEMORY_PHY_AWAKE 0x0002u
#define FT_TT_KMD_POWER_FLAG_TENSIX_ENABLE  0x0004u
#define FT_TT_KMD_POWER_FLAG_L2CPU_ENABLE   0x0008u
/** Every flag bit the word carries. */
#define FT_TT_KMD_POWER_FLAGS_ALL           0x7FFFu

/** The validity byte: low nibble = number of valid flag bits, high nibble = number of valid
 *  settings. */
#define FT_TT_KMD_POWER_VALIDITY(flag_count, setting_count) \
    ((((unsigned)(setting_count) & 0xFu) << 4) | ((unsigned)(flag_count) & 0xFu))

#define FT_TT_KMD_POWER_SETTING_COUNT 14u

/* ================================================================
 * Argument structs
 * ================================================================ */

/** GET_DEVICE_INFO (0). The caller sets in.out_size to sizeof(out). */
struct ft_tt_kmd_device_info_request {
    struct {
        uint32_t out_size;
    } in;
    struct {
        uint32_t out_size;
        uint16_t vendor_id;
        uint16_t device_id;
        uint16_t subsystem_vendor_id;
        uint16_t subsystem_id;
        uint16_t bdf;            /**< bits 0-2 function, 3-7 device, 8-15 bus */
        uint16_t max_dma_log2;
        uint16_t pci_domain;
        uint16_t reserved;
    } out;
};

/** QUERY_MAPPINGS (2) header. `count` records of ft_tt_kmd_mapping follow it directly. */
struct ft_tt_kmd_query_mappings_in {
    uint32_t count;
    uint32_t reserved;
};

/** One QUERY_MAPPINGS record. */
struct ft_tt_kmd_mapping {
    uint32_t id;          /**< FT_TT_KMD_MAPPING_* */
    uint32_t reserved;
    uint64_t base;        /**< an mmap offset on the chardev */
    uint64_t size;        /**< bytes */
};

/** QUERY_MAPPINGS as Foundry issues it: the header plus six records. */
struct ft_tt_kmd_query_mappings_request {
    struct ft_tt_kmd_query_mappings_in in;
    struct ft_tt_kmd_mapping mappings[FT_TT_KMD_MAPPING_QUERY_COUNT];
};

/** GET_DRIVER_INFO (5). The caller sets in.out_size to sizeof(out). */
struct ft_tt_kmd_driver_info_request {
    struct {
        uint32_t out_size;
    } in;
    struct {
        uint32_t out_size;
        uint32_t api_version;
        uint8_t major;
        uint8_t minor;
        uint8_t patch;
        uint8_t reserved;
    } out;
};

/** ALLOCATE_WINDOW (11). */
struct ft_tt_kmd_allocate_window_request {
    struct {
        uint64_t size;
        uint64_t reserved;
    } in;
    struct {
        uint32_t id;
        uint32_t reserved0;
        uint64_t mmap_offset_uc;
        uint64_t mmap_offset_wc;
        uint64_t reserved1;
    } out;
};

/** FREE_WINDOW (12). */
struct ft_tt_kmd_free_window_request {
    struct {
        uint32_t id;
        uint32_t reserved;
    } in;
};

/** Where a window points: the 32-byte target inside CONFIGURE_WINDOW. */
struct ft_tt_kmd_window_config {
    uint64_t addr;
    uint16_t x_end;
    uint16_t y_end;
    uint16_t x_start;
    uint16_t y_start;
    uint8_t noc;
    uint8_t mcast;
    uint8_t ordering;
    uint8_t linked;
    uint8_t static_vc;
    uint8_t reserved0[3];
    uint32_t reserved1[2];
};

/** CONFIGURE_WINDOW (13). */
struct ft_tt_kmd_configure_window_request {
    struct {
        uint32_t id;
        uint32_t reserved;
        struct ft_tt_kmd_window_config config;
    } in;
    struct {
        uint64_t reserved;
    } out;
};

/** NOC_CLEANUP (14): the write the driver performs at close. argsz = 32, flags = 0. */
struct ft_tt_kmd_noc_cleanup_request {
    uint32_t argsz;
    uint32_t flags;
    uint8_t enabled;
    uint8_t x;
    uint8_t y;
    uint8_t noc;
    uint32_t reserved;
    uint64_t addr;
    uint64_t data;
};

/** SET_POWER_STATE (15). argsz = 40, flags = 0. */
struct ft_tt_kmd_power_state_request {
    uint32_t argsz;
    uint32_t flags;
    uint8_t reserved;
    uint8_t validity;     /**< FT_TT_KMD_POWER_VALIDITY(flags, settings) */
    uint16_t flags_word;  /**< FT_TT_KMD_POWER_FLAG_* */
    uint16_t settings[FT_TT_KMD_POWER_SETTING_COUNT];
};

/** Driver-relayed ARC message (cmd 17), 48 bytes. Card tests use it as an independent path. */
#define FT_TT_KMD_SMC_MESSAGE_POST    (1u << 0)
#define FT_TT_KMD_SMC_MESSAGE_POLL    (1u << 1)
#define FT_TT_KMD_SMC_MESSAGE_ABANDON (1u << 2)
#define FT_TT_KMD_SMC_MESSAGE_WORDS   8u

struct ft_tt_kmd_smc_message {
    uint32_t argsz;       /**< 48 */
    uint32_t flags;       /**< FT_TT_KMD_SMC_MESSAGE_* */
    uint32_t queue;       /**< must be 0 */
    uint32_t reserved;    /**< must be 0 */
    uint32_t words[FT_TT_KMD_SMC_MESSAGE_WORDS]; /**< request on POST, reply on a good POLL */
};

/* ================================================================
 * Layout checks (C99 has no _Static_assert: a negative array size is the error)
 * ================================================================ */

#define FT_TT_KMD_ABI_CHECK(name, condition) \
    typedef char ft_tt_kmd_abi_check_##name[(condition) ? 1 : -1]

FT_TT_KMD_ABI_CHECK(device_info_size, sizeof(struct ft_tt_kmd_device_info_request) == 24);
FT_TT_KMD_ABI_CHECK(device_info_out_at_4,
                    offsetof(struct ft_tt_kmd_device_info_request, out) == 4);
FT_TT_KMD_ABI_CHECK(device_info_out_size,
                    sizeof(((struct ft_tt_kmd_device_info_request *)0)->out) == 20);
FT_TT_KMD_ABI_CHECK(query_in_size, sizeof(struct ft_tt_kmd_query_mappings_in) == 8);
FT_TT_KMD_ABI_CHECK(mapping_size, sizeof(struct ft_tt_kmd_mapping) == 24);
FT_TT_KMD_ABI_CHECK(mapping_base_at_8, offsetof(struct ft_tt_kmd_mapping, base) == 8);
FT_TT_KMD_ABI_CHECK(mapping_size_at_16, offsetof(struct ft_tt_kmd_mapping, size) == 16);
FT_TT_KMD_ABI_CHECK(query_request_size,
                    sizeof(struct ft_tt_kmd_query_mappings_request) == 8 + 6 * 24);
FT_TT_KMD_ABI_CHECK(driver_info_size, sizeof(struct ft_tt_kmd_driver_info_request) == 16);
FT_TT_KMD_ABI_CHECK(driver_info_out_size,
                    sizeof(((struct ft_tt_kmd_driver_info_request *)0)->out) == 12);
FT_TT_KMD_ABI_CHECK(allocate_size, sizeof(struct ft_tt_kmd_allocate_window_request) == 48);
FT_TT_KMD_ABI_CHECK(allocate_out_at_16,
                    offsetof(struct ft_tt_kmd_allocate_window_request, out) == 16);
FT_TT_KMD_ABI_CHECK(free_size, sizeof(struct ft_tt_kmd_free_window_request) == 8);
FT_TT_KMD_ABI_CHECK(window_config_size, sizeof(struct ft_tt_kmd_window_config) == 32);
FT_TT_KMD_ABI_CHECK(window_config_noc_at_16,
                    offsetof(struct ft_tt_kmd_window_config, noc) == 16);
FT_TT_KMD_ABI_CHECK(configure_size, sizeof(struct ft_tt_kmd_configure_window_request) == 48);
FT_TT_KMD_ABI_CHECK(configure_config_at_8,
                    offsetof(struct ft_tt_kmd_configure_window_request, in.config) == 8);
FT_TT_KMD_ABI_CHECK(configure_out_at_40,
                    offsetof(struct ft_tt_kmd_configure_window_request, out) == 40);
FT_TT_KMD_ABI_CHECK(noc_cleanup_size, sizeof(struct ft_tt_kmd_noc_cleanup_request) == 32);
FT_TT_KMD_ABI_CHECK(noc_cleanup_addr_at_16,
                    offsetof(struct ft_tt_kmd_noc_cleanup_request, addr) == 16);
FT_TT_KMD_ABI_CHECK(power_state_size, sizeof(struct ft_tt_kmd_power_state_request) == 40);
FT_TT_KMD_ABI_CHECK(power_state_settings_at_12,
                    offsetof(struct ft_tt_kmd_power_state_request, settings) == 12);
FT_TT_KMD_ABI_CHECK(smc_message_size, sizeof(struct ft_tt_kmd_smc_message) == 48);
FT_TT_KMD_ABI_CHECK(smc_message_words_at_16, offsetof(struct ft_tt_kmd_smc_message, words) == 16);

#endif /* FT_TT_KMD_ABI_H */
