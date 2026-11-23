/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv_noc.h
 * @brief NoC transport for the USERSPACE DRIVER: move an arbitrary run of bytes
 *   between host memory and a NoC endpoint through a 2 MiB window WE aim, with
 *   no tt-kmd, no ioctl and no character device.
 *
 *   This is `ft_tt_noc.c`'s window walk ported onto our own register writes.
 *   The copy shape itself is not ported — it is SHARED, from
 *   `ft_tt_noc_copy.h`, so the two transports issue byte-for-byte the same
 *   stores and a measurement taken through one is a measurement of the other.
 *   What changes is only how the window gets aimed:
 *
 *     tt-kmd path   ALLOCATE_TLB -> CONFIGURE_TLB ioctl -> mmap the chardev
 *     this path     flock a window -> three MMIO stores -> mmap resource0[_wc]
 *
 * ---------------------------------------------------------------------------
 * A WRONG AIM IS A WRONG ADDRESS, NOT AN ERROR — SO EVERY AIM IS VERIFIED
 * ---------------------------------------------------------------------------
 * A window pointed somewhere nobody chose does not fail. It reads and writes a
 * different endpoint, silently, and on this project that has cost a power
 * cycle. tt-kmd's CONFIGURE_TLB is a syscall that can return an error; three
 * stores through an mmap cannot. So `ft_tt_udrv_noc_window_aim()` does not
 * return once the stores are issued: it READS THE REGISTER TRIPLE BACK and
 * refuses, with FT_TT_UDRV_EIO and nothing dereferenced, unless all three words
 * are bit-identical to the intent. `ft_tt_udrv_noc_transfer()` will not touch
 * an aperture whose aim did not verify.
 *
 * **The honest limit of that check, stated so nobody over-reads it.** The
 * readback goes through the same mapping that wrote it, which is the
 * "instrumenting a mechanism with itself" hazard. It IS a real device access —
 * the register file is mapped uncached through `resource0`, so the read is a
 * PCIe read of the card, not of a cache — so it catches a dropped word, a torn
 * write, a register that does not hold the value, and a strided register that
 * did not clear. What it CANNOT catch is a mapping that consistently points at
 * the wrong physical page, because such a mapping would agree with itself.
 * That residual is closed on the current bank, and only there, by
 * `tests/test_udrv_card_noc.c` reading the same registers back through tt-kmd's
 * BAR0 mapping — a different file, vma and kernel path. After the move to a
 * board with no tt-kmd there is no second reader, which is why the comparison
 * has to be made while the module is still installed.
 *
 * ---------------------------------------------------------------------------
 * COORDINATES ARE PARAMETERS. THEY ARE NEVER PROBED.
 * ---------------------------------------------------------------------------
 * Same rule as ft_tt_noc.h and for the same reason. Every entry point takes a
 * TRANSLATED coordinate the caller derived; nothing here guesses, defaults or
 * scans. Coordinates above 63 are REFUSED rather than truncated, because the
 * register's 6-bit fields would silently fold 64 onto 0 — a different, live
 * endpoint.
 *
 * ---------------------------------------------------------------------------
 * WRITE-COMBINED vs UNCACHED, AND THE FENCE
 * ---------------------------------------------------------------------------
 * A path is opened over one or the other. WC is the bulk upload path; UC is for
 * registers and for anything whose store ORDER matters, because write combining
 * is free to merge and reorder stores within a buffer. The write path calls
 * `ft_tt_noc_copy_store_fence()` after the last store of every slice, which is
 * an `sfence`: without it a WC store is still in the core's write-combining
 * buffer and a readback races the buffer rather than the card. The tt-kmd path
 * has only ever had the compiler barrier — see this file's implementation notes.
 *
 * @status THE TRANSPORT IS ON SILICON (2026-08-17, p150a at 0000:12:00.0 with
 *   tt-kmd loaded as the oracle). `tests/test_udrv_card_noc.c`, 62 checks, 0
 *   failed, and 6 mutations all red:
 *     - a window we aimed with our own register writes returns byte-identical
 *       bytes to one tt-kmd aimed, and the discrimination arm (their window
 *       re-aimed one page away) fails as it must;
 *     - **A WRITE-COMBINED STORE REACHES THE CARD.** Proven twice. At register
 *       level: three TLB configurations stored through `resource0_wc` and read
 *       back through tt-kmd's own BAR0 mapping, with no NoC transaction
 *       involved at all. And through an APERTURE: 64 KiB into a claimed
 *       Tensix's L1, read back byte-exact by `ft_tt_read_l1()` over tt-kmd's
 *       window. **296.8 MB/s WC against 109.0 MB/s UC**, same 64 KiB, same
 *       core, same copy shape;
 *     - the walk moves the window mid-transfer on a DRAM span that straddles a
 *       2 MiB boundary, and the "already aimed" skip costs zero aims;
 *     - all 32 combinations of `stream_header` x `static_vc` x `use_static_vc`
 *       are RETAINED by the hardware — they are not reserved bits that read
 *       back zero — and no two of them produce the same register words.
 * @issues
 *   - The aim verification reads back through the mapping that wrote it. See
 *     the note above for exactly what that does and does not prove.
 *   - There is no arbitration in this file. A caller must claim its window
 *     through `ft_tt_udrv_claim.h` first; nothing here stops two processes
 *     aiming the same one.
 *   - `static_vc` 1-7 are proven HELD by the register and NOT proven to ROUTE.
 *     Nothing in tt-kmd sets a non-zero static VC, so there is no oracle for
 *     one, and a VC reserved for another traffic class is a fabric hazard whose
 *     cost is a power cycle. Only VC 0 has a functional arm.
 * @todo None
 */

#ifndef FT_TT_UDRV_NOC_H
#define FT_TT_UDRV_NOC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ft_tt_noc_copy.h"
#include "ft_tt_udrv_pci_sysfs.h"
#include "ft_tt_udrv_tlb_encode.h"

/* ================================================================
 * The device
 * ================================================================ */

/**
 * An open card, for NoC purposes: the sysfs description plus a mapping of
 * BAR0's TLB register file.
 *
 * The register mapping is UNCACHED, through `resourceN` and never
 * `resourceN_wc`. That is what tt-kmd's own `pci_iomap_range()` gives it, and write-combining a register write is how a store
 * arrives late, out of order, or merged with its neighbour.
 *
 * Caller-owned storage; treat the fields as private.
 */
typedef struct ft_tt_udrv_noc_device {
    ft_tt_udrv_pci_device pci;
    volatile uint32_t *registers; /**< BAR0 + FT_TT_UDRV_TLB_REGS_START, uncached */
    bool writable;                /**< false for a read-only bring-up rung */
} ft_tt_udrv_noc_device;

/**
 * Map BAR0's TLB register file for `pci`.
 *
 * @param pci        a device from ft_tt_udrv_pci_find_devices()
 * @param writable   PROT_WRITE as well as PROT_READ. Pass false unless the
 *                   caller's whole purpose is to aim windows: a read-only
 *                   mapping makes a stray store a SIGSEGV on the host instead
 *                   of a wrong aim on the card
 * @param out_device zeroed and filled on success
 * @return           FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL; FT_TT_UDRV_EIO when the
 *                   mapping fails (errno is left set; EACCES means "not root
 *                   and no udev rule")
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ft_tt_udrv_noc_device_open(const ft_tt_udrv_pci_device *pci, bool writable,
                               ft_tt_udrv_noc_device *out_device);

/** Unmap the register file. Safe on a zeroed or already-closed device. */
int ft_tt_udrv_noc_device_close(ft_tt_udrv_noc_device *device);

/**
 * Read window `id`'s three configuration registers.
 *
 * @return FT_TT_UDRV_OK, or FT_TT_UDRV_EINVAL for a NULL argument, an unopened
 *         device or an id past FT_TT_UDRV_TLB_2M_WINDOW_COUNT
 */
int ft_tt_udrv_noc_window_read(const ft_tt_udrv_noc_device *device, uint32_t id,
                               uint32_t out_words[3]);

/**
 * Aim window `id` — the whole of what CONFIGURE_TLB does to hardware, plus the
 * verification the ioctl gets for free by being a syscall.
 *
 * Four stores in tt-kmd's own order (low, mid, high, then the strided register
 * zeroed for id < 32), then all four read back and compared bit for bit.
 *
 * @param device  an open, WRITABLE device
 * @param id      window index
 * @param config  where to aim it; `addr` must be 2 MiB aligned and every field
 *                must fit its register width (this refuses rather than
 *                truncating — see ft_tt_udrv_tlb2m_config_check())
 * @return        FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL for a bad argument or a
 *                config that does not fit; FT_TT_UDRV_ENOTSUP for a read-only
 *                device; FT_TT_UDRV_EIO when the readback disagrees, in which
 *                case the aperture MUST NOT be touched
 *
 * Thread-safe: no (the register file is shared)   Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_noc_window_aim(ft_tt_udrv_noc_device *device, uint32_t id,
                              const ft_tt_udrv_tlb2m_config *config);

/** The same aim, reporting how long its READBACK took. Same stores, same checks, same
 *  refusals — the only difference is that the caller can price the two halves apart. */
int ft_tt_udrv_noc_window_aim_timed(ft_tt_udrv_noc_device *device, uint32_t id,
                                    const ft_tt_udrv_tlb2m_config *config,
                                    double *out_verify_seconds);

/* ================================================================
 * The data path
 * ================================================================ */

/**
 * One host<->card data path: a window this process owns, its aperture, and the
 * bookkeeping needed to slide it.
 *
 * The path OWNS its window for as long as it is open and is the only thing that
 * aims it, which is what makes the "already aimed where I need it" fast path
 * sound. Aiming the same window from anywhere else while a path is attached is
 * a wrong-endpoint bug, not a race to be handled.
 */
typedef struct ft_tt_udrv_noc_path {
    ft_tt_udrv_noc_device *device;
    uint32_t window_id;
    void *aperture;    /**< mmap of resource0[_wc] at id * 2 MiB */
    size_t window_bytes;
    bool write_combined;

    /** The fields every aim carries that are not the endpoint: ordering, NoC,
     *  linked, and the two the ioctl cannot express. Set with
     *  ft_tt_udrv_noc_path_set_aim_template(); defaults match what foundry's
     *  tt-kmd path asks for (all zero — relaxed, NoC 0), so a register written
     *  by either transport is byte-identical. */
    ft_tt_udrv_tlb2m_config aim_template;

    ft_tt_udrv_tlb2m_config aim; /**< what the window currently holds */
    bool aimed;

    uint64_t aim_count;      /**< windows moved since open — the re-windowing proof */
    uint64_t aim_failures;   /**< aims whose readback disagreed */

    /** WHAT AN AIM COSTS, AND WHICH HALF OF IT.
     *
     *  On the dispatch path a window is re-aimed ONCE PER DESCRIPTOR — every descriptor goes
     *  to a different core than the last one did — so the aim is not setup, it is a
     *  per-descriptor term of the token. Its two halves are different kinds of access and
     *  cannot be assumed equal: the four register stores are POSTED and retire at
     *  store-buffer speed, while the four readbacks are NON-POSTED and each is a PCIe round
     *  trip. WHICH HALF DOMINATES DECIDES THE FIX — fewer readbacks, or more windows so
     *  there is nothing to re-aim. */
    double aim_seconds;        /**< the whole aim, stores and readback */
    double aim_verify_seconds; /**< the readback alone */
    uint64_t bytes_written;  /**< host->card bytes through this path */
    uint64_t bytes_read;     /**< card->host bytes through this path */
} ft_tt_udrv_noc_path;

/**
 * Map window `id`'s aperture and build a path over it.
 *
 * The window is NOT claimed here — claim it through `ft_tt_udrv_claim.h` first.
 * This maps BAR0 at `id * 2 MiB`, which is bit for bit what tt-kmd's
 * `map_tlb_window()` does, with the ioctl removed.
 *
 * @param device         an open device; must be writable to write through the path
 * @param id             window index, < FT_TT_UDRV_TLB_2M_WINDOW_COUNT
 * @param write_combine  true for the bulk data path, false for registers
 * @param out_path       caller-owned storage, zeroed and filled on success
 * @return               FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL; FT_TT_UDRV_EIO
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int ft_tt_udrv_noc_path_open(ft_tt_udrv_noc_device *device, uint32_t id, bool write_combine,
                             ft_tt_udrv_noc_path *out_path);

/** Seconds spent aiming this path's window, and the readback's share of it. */
double ft_tt_udrv_noc_path_aim_seconds(const ft_tt_udrv_noc_path *path);
double ft_tt_udrv_noc_path_aim_verify_seconds(const ft_tt_udrv_noc_path *path);

/**
 * Arm or disarm the word-0 aim perturbation. **TEST HARNESSES ONLY.**
 *
 * ---------------------------------------------------------------------------
 * WHAT IT IS FOR: PRICING WHAT `FT_TT_UDRV_AIM_VERIFY=word` STOPPED CHECKING
 * ---------------------------------------------------------------------------
 * While armed, every aim writes its three config words and then corrupts word 0
 * by one 2 MiB page — moving the window to a DIFFERENT PAGE OF THE SAME CORE and
 * leaving `x_end`/`y_end` alone. `full` reads word 0 back and REFUSES with EIO;
 * `word` reads only word 1 and cannot see it. Running the two side by side is
 * the only way to show that the dropped check fires.
 *
 * ---------------------------------------------------------------------------
 * IT TAKES TWO KEYS AND A CALL, AND THE CALL IS THE TIMING
 * ---------------------------------------------------------------------------
 * Arming requires `FT_TT_UDRV_AIM_RED=aimlow` AND
 * `FT_TT_UDRV_AIM_RED_CONFIRM=i-know-this-aims-at-the-wrong-page`; without both
 * this refuses loudly and returns false, so an environment line on a model run
 * arms nothing. The call exists because the FIRST aims of any process are the
 * card bring-up's: an arm that is live before the card is open kills the process
 * before it reaches its own check, which is how the first version of this
 * measured nothing. Open the card honestly, arm, do ONE transfer, disarm.
 *
 * **DRAM ONLY.** One page off a DRAM base is inert; one page off an L1 base is an
 * address no tile decodes, and such a read wedged the NoC and cost a physical
 * power cycle on 2026-08-26.
 *
 * @param on  true to arm, false to disarm
 * @return    whether the arm is now live — an arm that could not fire reads as
 *            false rather than as an absence
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
bool ft_tt_udrv_noc_aim_red_arm(bool on);

/** Unmap the aperture and blank the path. Safe on a zeroed or closed path. */
int ft_tt_udrv_noc_path_close(ft_tt_udrv_noc_path *path);

/**
 * Set the non-endpoint half of every aim this path makes: ordering, NoC,
 * `linked`, and the two fields CONFIGURE_TLB has no way to express —
 * `stream_header` and the 3-bit `static_vc`. Changing the template invalidates
 * the current aim, so the next transfer re-aims.
 *
 * @return FT_TT_UDRV_OK, or FT_TT_UDRV_EINVAL if a field does not fit its
 *         register width
 */
int ft_tt_udrv_noc_path_set_aim_template(ft_tt_udrv_noc_path *path, uint8_t noc, uint8_t ordering,
                                         uint8_t linked, uint8_t use_static_vc,
                                         uint8_t static_vc_id, uint8_t stream_header);

/** Number of times this path has aimed its window. A transfer that stays inside
 *  one 2 MiB window at an endpoint the window already points at costs ZERO —
 *  which is how a test tells a re-windowing run from a lucky in-window one. */
uint64_t ft_tt_udrv_noc_path_aim_count(const ft_tt_udrv_noc_path *path);

/** Aims whose readback disagreed with the intent. Must be zero; a non-zero
 *  value means an aperture was refused, not that a wrong address was used. */
uint64_t ft_tt_udrv_noc_path_aim_failures(const ft_tt_udrv_noc_path *path);

/**
 * Move `bytes` between host memory and (x, y, address), sliding the window as
 * needed, with a caller-chosen store width and NO memory-space bounds check.
 *
 * This is the same seam `ft_tt_noc_transfer_raw()` is, and for the same two
 * callers: a 4-byte device register outside every memory space, and a sub-16 B
 * mailbox field. Everything before the device is still enforced — a non-NULL
 * path and aperture, `address` and `bytes` aligned to the width, a coordinate
 * inside the register's 6-bit fields, and an aim that verified.
 *
 * @param path     an open path
 * @param x, y     TRANSLATED coordinate. NEVER a guess
 * @param address  tile-local address, aligned to the width
 * @param host     source when writing, destination when reading
 * @param bytes    multiple of the width. Zero is a legal no-op
 * @param is_read  true for card->host
 * @param width    FT_TT_NOC_WIDTH_BLOCK16 for data, WORD32 for registers
 * @return         FT_TT_UDRV_OK; FT_TT_UDRV_EINVAL; FT_TT_UDRV_ENOTSUP for a
 *                 write through a read-only device; FT_TT_UDRV_EIO when an aim
 *                 did not verify (nothing was dereferenced)
 *
 * Thread-safe: no (the path is stateful)   Reentrant: no   Signal-safe: no
 */
int ft_tt_udrv_noc_transfer(ft_tt_udrv_noc_path *path, uint16_t x, uint16_t y, uint64_t address,
                            void *host, size_t bytes, bool is_read,
                            ft_tt_noc_transfer_width width);

/* ================================================================
 * ft_tt_noc_access-shaped adapters
 * ================================================================ */

/**
 * `ft_tt_launch.h`'s `ft_tt_noc_access` vtable is
 * `int (*)(void *user, uint16_t x, uint16_t y, uint64_t addr, ..., size_t bytes)`.
 * These two have exactly that signature with `user` = an
 * `ft_tt_udrv_noc_path *`, so the whole upper backend — run control, the op
 * layer, everything built on that vtable — runs over this transport by filling
 * the struct with these, and this header still does not include the tt-kmd
 * layer it replaces.
 *
 * WORD32 width, matching what `ft_tt_launch_noc_over_window()` uses, because
 * the vtable's contract is "any 4-byte-aligned address and length".
 */
int ft_tt_udrv_noc_access_write(void *user, uint16_t x, uint16_t y, uint64_t address,
                                const void *src, size_t bytes);

/** Read half of the vtable pair. See ft_tt_udrv_noc_access_write(). */
int ft_tt_udrv_noc_access_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                               size_t bytes);

#endif /* FT_TT_UDRV_NOC_H */
