/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ft_tt_udrv.h
 * @brief Umbrella header for the FOUNDRY USERSPACE DRIVER for Blackhole — the
 *   layer that replaces tt-kmd so a user can plug the card in and run Pantheon
 *   with no kernel module, no DKMS, no kernel headers and no module signing.
 *
 *   This header owns two things and no code: the shared status codes for the
 *   `ft_tt_udrv_*` modules, and the written record of WHAT tt-kmd does that we
 *   must replicate, and WHAT we lose by not having it.
 *
 * @status **THE PRODUCT NOW RUNS ON THIS LAYER WITH NO VENDOR DRIVER
 *   (2026-08-17).** The seam is `ft_tt_runtime_open()` selecting a transport
 *   from `FT_TT_DRIVER` (default `kmd`); the branch is
 *   `ft_tt_runtime_udrv.[ch]`. A real `pantheon serve` with `tenstorrent`
 *   unloaded answered a chat with `[admit] path=gpu` and 6,760 matmuls on the
 *   card. `ft_tt_core status all --driver udrv` read 140 cores and
 *   `recover --rung 3 --driver udrv --write` did the whole ASIC reset — both
 *   with no chardev on the machine. See `ft_tt_runtime_udrv.h` @status for the
 *   table and for the measured price of the aim readback.
 *
 *   THE TLB PATH IS ON SILICON (2026-08-16, p150a at 0000:12:00.0, with
 *   tt-kmd loaded as the oracle). `tests/test_udrv_card_tlb.c` runs four rungs
 *   and all of them are green:
 *     1. sysfs discovery + `mmap` of BAR0's register file, read-only
 *     2. window 201 — tt-kmd's own — decodes to ARC (8,0), NoC 0, strict,
 *        aimed at the CSM page: `0x00000080 0x00004000 0x00000040`
 *     3. 64 configurations aimed through CONFIGURE_TLB on two windows, one
 *        below and one above the strided threshold, byte-identical to what
 *        our encoder produces; coordinate truncation matches the kernel's
 *     4. our own register writes, read back through TT-KMD's BAR0 mapping,
 *        and a window we aimed ourselves returns the same bytes as one the
 *        driver aimed. `0x9D8` is confirmed on silicon and `0xA08` refuted.
 *   THE RESET PATH IS ALSO ON SILICON (2026-08-17). `ft_tt_udrv_reset.[ch]`
 *   recovered a deliberately wedged core with three PCI config writes, its own
 *   config snapshot restored in place, and a bridge link retrain — no vendor
 *   ioctl, no PCI remove, no rescan.
 *   AND SO IS THE NoC TRANSPORT (2026-08-17). `ft_tt_udrv_noc.[ch]` moves bytes
 *   through windows we aim, with no ioctl in the data path. The load-bearing
 *   new fact: **a WRITE-COMBINED STORE THROUGH `resource0_wc` REACHES THE
 *   CARD** — proven at register level (read back through tt-kmd's BAR0
 *   mapping) and through an aperture into a Tensix's L1 (read back by
 *   `ft_tt_read_l1()` over tt-kmd's own window), at 296.8 MB/s WC against
 *   109.0 MB/s UC. `stream_header` and the 3-bit `static_vc`, which
 *   CONFIGURE_TLB cannot express, are swept: the hardware RETAINS all 32
 *   combinations.
 *
 *   AND THE ARC PROBE SEQUENCE IS ON SILICON (2026-08-17), which was the last
 *   step. `ft_tt_udrv_arc_probe.[ch]` binds the message encoders to the NoC
 *   transport and sends real messages; `tests/test_udrv_card_arc.c` is the
 *   round: 62 checks green, 7 mutations all red. What it settled:
 *     - **BOTH UNPROVEN ADDRESSES ARE CONFIRMED.** `RESET_SCRATCH(11)`
 *       (0x8003042C) held 0x1002DD68 -> queue base 0x10032DF0, 4 entries — and
 *       it is TT-KMD'S OWN queue, proved by making the driver send a message
 *       through SMC_MSG and watching the request write pointer OF THE QUEUE WE
 *       LOCATED advance by exactly one while a header word that is not a
 *       pointer stayed put. The doorbell at `ARC_MSI_FIFO` 0x800B0000 is proved
 *       LOAD-BEARING by a transport whose trigger does nothing: same push, same
 *       poll loop, and no response in 1.5 s; ringing it then released the
 *       message that was already queued.
 *     - ASIC_STATE0, SET_WDT_TIMEOUT (10 s) and POWER_SETTING all answered with
 *       firmware status 0. A TEST message sent through OUR path and through
 *       tt-kmd's SMC_MSG returns the same seven fixed words, and the eighth —
 *       the firmware's own TEST counter — advances by exactly one between them,
 *       so both paths feed ONE queue.
 *     - **POWER_SETTING WORKS AND IT IS MEASURABLE ON AN INSTRUMENT THAT IS NOT
 *       THE RING.** With tt-kmd asking for the MINIMUM from two open fds, our
 *       0x7FFF request took the AI clock 800 -> 1350 MHz, Vcore 740 -> 862 mV
 *       and board power 35 -> 69 W; the crossed control (our own message asking
 *       for tt-kmd's 0x7FFE) put all three back; repeating the max request
 *       raised them again. Under a real Tensix load — four cores, 1500
 *       byte-exact vector_add runs each, 18.4 s, 184 telemetry samples taken by
 *       a second process WHILE it ran — the clock held at 1350 with a 75 W /
 *       88 A peak, against 800 MHz and 39 W for the same work in the min arm.
 *     - Our own NoC walk of the ARC telemetry table agrees with tt-kmd's sysfs
 *       on every row, so the instrument survives the module.
 *   THE REHEARSAL IS RUN (2026-08-17) AND THE CARD WORKS WITH tt-kmd REMOVED.
 *   `tests/test_udrv_rehearsal.c` + `Makefile.udrvrehearsal`: 43 checks
 *   green in the bring-up arm and 64 in the reset arm, 5 mutations all red, card released `probe all` 140/140 and 5 GT/s x1 both
 *   ends. Its Makefile links NO `ft_tt_kmd.c`, NO `ft_tt_tlb.c`, NO
 *   `ft_tt_noc.c` and NO `ft_tt_device.c`, and `nm -u` on the binary finds no
 *   `ioctl` symbol at all. What it settled, each against the module being gone:
 *     - **THE ARC ANSWERS IN A3, AND THAT WAS THE ROUND'S BIGGEST UNKNOWN.**
 *       First NoC read after `rmmod`: ARC_BOOT_STATUS = 0x00000005, ready bit
 *       already SET, 0.000 s. RESET_SCRATCH(11) still 0x1002DD68 -> queue
 *       0x10032DF0, 4 entries. TEST, ASIC_STATE0, SET_WDT_TIMEOUT and
 *       POWER_SETTING all answered firmware status 0.
 *     - **POWER_SETTING MEASURED THROUGH A CROSSED CONTROL, not a bare "did it
 *       go up".** With no module nothing lowers the clock at process exit, so a
 *       second run starts at 1350 and a naive check reads 1350 -> 1350. The
 *       bring-up now asks for tt-kmd's own 0x7FFE first (clock falls to 800),
 *       then 0x7FFF (rises to 1350), read through our own NoC walk of the ARC
 *       telemetry table — the only telemetry reader left once the sysfs tree
 *       goes.
 *     - **RESET WITH NO MODULE WORKS, and it surfaced one new fact:** straight
 *       after an ASIC reset ARC_BOOT_STATUS reads 0x00000002 — ready bit CLEAR
 *       — and takes ~280 polls to set, where after `rmmod` it is set on the
 *       first read. `ft_tt_udrv_arc_locate_queue()`'s bounded poll loop is what
 *       covers that; a bring-up that read the bit once would fail on the reset
 *       path only. Marker cleared in 119-120 ms, config restored in place (23
 *       steps, 6 written, 0 mismatches), link retrained 2.5 -> 5 GT/s in 10 ms,
 *       MRRS back to 4096, heartbeat restarted (172040 -> 5).
 *     - **RECOVERY OF A WEDGED CORE WITH NO VENDOR DRIVER.** Core 100 wedged
 *       deliberately by the 1x5 multicast relaunch arm (soft reset 0x00047000,
 *       instrn_buf_status 0x66, invalid_instrn 1); read as 0x66 over udrv with
 *       the module gone, recovered by the udrv ASIC reset, read back 0x77, and
 *       confirmed by `ft_tt_core probe 100 --write` PASS 4/4 once tt-kmd was
 *       reloaded. **`instrn_buf_status` over a udrv NoC path is the whole
 *       health instrument on a board with no module** — `probe --write` needs
 *       the tt-kmd backend.
 *     - MRRS reads 4096 straight after `rmmod`: `pci_disable_device()` does not
 *       clear DEVCTL. PCI_COMMAND reads 0x0102 — MEMORY DECODE STILL ON — even
 *       though `enable` reads 0. Had it been off, every BAR read would have
 *       been all ones through a mapping that succeeded.
 *     - `resource0` maps with no driver bound; `vfio-pci` was never needed.
 *       All 1024 register words identical through `resource0` and
 *       `resource0_wc`. Window 201 aimed and read back byte-exact, then
 *       restored. Three windows claimed with the flock as the ONLY arbitrator,
 *       a forked rival refused the held one and granted a free one.
 *     - **600 s IDLE HOLD WITH THE MODULE REMOVED and no host access of any
 *       kind: heartbeat 781 -> 6789 continuous, clock still 1350, Vcore still
 *       860 mV, ARC still ready.** 60 watchdog timeouts, on the exact condition
 *       the target board will be in. Still not an afternoon.
 * @issues
 *   - ~~THE PRODUCT CANNOT USE THIS LAYER~~ — **THE SEAM IS WRITTEN,
 *     2026-08-17.** `ft_tt_runtime_open()` selects a transport from
 *     `FT_TT_DRIVER` (default `kmd`) and the udrv branch lives in
 *     `ft_tt_runtime_udrv.[ch]`: sysfs discovery, BAR mapping, two flock'd
 *     windows, our own ARC bring-up, the shared telemetry walk, and the same
 *     three `ft_tt_noc_access` accessors the tt-kmd branch produces. Nothing
 *     above `ft_tt_runtime` changed, and nothing was duplicated — the copy
 *     shape, the memory-space bounds, the telemetry walk and the coordinates
 *     are all shared. A `FT_TT_DRIVER` that cannot be honoured REFUSES: a
 *     serve stops rather than answering on the CPU, because the completion
 *     text is byte-identical either way. `ft_tt_core recover --driver udrv`
 *     is fixed the same way and no longer opens the chardev to read the
 *     before-state.
 *   - RESET IS DONE AND EXERCISED (2026-08-17), and it was the load-bearing
 *     loss. `ft_tt_udrv_reset.[ch]` does the ASIC reset as three PCI config
 *     writes, snapshots and restores config space IN PLACE, and retrains the
 *     link. It recovered a DELIBERATELY WEDGED core — 101 at 0x00047000,
 *     `probe --write` FAIL before and PASS 4/4 after — with no PCI remove and
 *     no rescan. Entry point: `ft_tt_core recover --rung 3 --driver udrv
 *     --write`. §5(a) below is updated and §5(e)'s discriminating experiment
 *     is answered.
 *   - **`rmmod tenstorrent` LEAVES THE CARD IN A3 — AND THE QUEUE ANSWERS
 *     ANYWAY (MEASURED 2026-08-17). This entry is kept as the record of what
 *     was feared; it is no longer an open question.** The paragraph below is
 *     the reasoning that made it the round's first move, and it was right to
 *     be: the answer just went the good way.**
 *     `blackhole_cleanup_hardware()` sends ASIC_STATE3 and
 *     FW_LOG RELEASE on unload, and the last chardev close has already sent
 *     POWER_SETTING with every flag off. So the first card this driver ever
 *     meets alone is in a state nothing here has been measured in. The antidote
 *     is exactly `ft_tt_udrv_arc_probe_hardware()` and every message in it is
 *     proven — but only ever sent to a card already in A0. **Whether the message
 *     queue answers at all in A3 is the open question**, and the fallback is one
 *     `modprobe` away.
 *   - We do not send ASIC_STATE3 ourselves. `ft_tt_udrv_arc_msg_asic_state3()`
 *     exists and is deliberately not sequenced: a power-down at the wrong moment
 *     is card-wide, and this module is what a bring-up runs. A card we owned is
 *     therefore left in A0, which is an asymmetry with the driver, not a leak.
 *   - /dev/mem is NOT a usable second reader of the BAR on this box: a `pread`
 *     at the BAR's physical address returns EFAULT even as root. The second
 *     reader is tt-kmd's own BAR0 mapping, which will not exist on a machine
 *     with no tt-kmd — so after the move there is no independent readback
 *     path at all, and the register comparison becomes self-instrumenting.
 *     WHAT SURVIVES THE MOVE, and it is more than it sounds: a bad aim is still
 *     detectable WITHOUT a second reader, because the window's registers are
 *     mapped UNCACHED and reading them back is a real PCIe read of the card.
 *     `ft_tt_udrv_noc_window_aim()` does that on every aim and refuses the
 *     aperture on any disagreement. What is lost is only the ability to catch a
 *     mapping that consistently points at the WRONG PAGE — a failure that would
 *     agree with itself. That specific residual is what the on-card comparison
 *     against tt-kmd closes, and it has now been made (2026-08-17: all 606
 *     configuration words identical through both mappings, and our own aim
 *     confirmed through tt-kmd's before any aperture was dereferenced).
 * @todo
 *   - [x] NoC transport. DONE 2026-08-17, `ft_tt_udrv_noc.[ch]`. The copy shape
 *         is SHARED with ft_tt_noc.c through ft_tt_noc_copy.h rather than
 *         ported, so there is one PIO copy in the backend and a measurement
 *         through either transport is a measurement of both.
 *   - [x] The probe sequence: ASIC_STATE0, SET_WDT_TIMEOUT, POWER_SETTING. DONE
 *         2026-08-17, `ft_tt_udrv_arc_probe.[ch]`. MRRS stayed off the list: it
 *         lives in DEVCTL and the config restore rewrites it.
 *   - [x] Decide whether the ARC watchdog needs a host heartbeat. ANSWERED — see
 *         §3. No heartbeat. Armed at 10 s and held idle for 35 s with no host
 *         traffic of any kind; link unchanged at 5 GT/s, heartbeat advanced 350
 *         with no restart, the AI clock ONLY our message could have set was
 *         still in force, and the ARC still answered.
 *   - [x] Recovery, exercised on a deliberately wedged core. DONE 2026-08-17.
 *   - [x] Bring the card up after a reset with NO tt-kmd present. **DONE
 *         2026-08-17 — the rehearsal ran exactly this and it works.** See the
 *         @status block; the one thing it taught is that the ARC is NOT ready
 *         for messages on the first read after a reset.
 *   - [x] **Give the product a udrv seam.** DONE 2026-08-17,
 *         `ft_tt_runtime_udrv.[ch]` + the `FT_TT_DRIVER` branch in
 *         `ft_tt_runtime_open()`.
 *   - [ ] Send `POWER_SETTING` 0x7FFE (or ASIC_STATE3) on release, so a card we
 *         owned is not left at 1350 MHz forever. Visible today: a second run of
 *         the rehearsal starts on a card the first one left at max.
 *   - [x] **Set MRRS ourselves at bring-up. DONE 2026-08-18**,
 *         `ft_tt_udrv_config_max_read_request_set()` called from
 *         `runtime_udrv_open_locked()` before the BAR is mapped — the order
 *         tt-kmd's own probe uses. **Proven on a COLD path, not a reset path:**
 *         `tests/run_udrv_mrrs_cold_gate.sh` seeds a WRONG MRRS from pid+clock
 *         (this bank always reads 4096, so an unseeded check is vacuous), opens
 *         over udrv with no reset anywhere in the path, and the card comes back
 *         at 4096 with the bring-up reporting `WRITTEN by us` rather than
 *         `already correct`. Discriminator: the same command over `kmd` leaves
 *         the seeded value alone, so 4096 is not something that happens anyway.
 *         Two mutations, both red — the DEVCTL write deleted (the readback
 *         refuses and says so), and the bring-up not calling it at all.
 *         **HONEST SCOPE: this is probe parity, not a throughput lever.** MRRS
 *         bounds reads the DEVICE issues upstream and our data path never has
 *         the card read host memory (§4).
 *         **AND IT NEEDS ROOT.** A config-space write is privileged, and an
 *         unprivileged READ of `config` is truncated by the kernel to the
 *         64-byte standard header — so a non-root process cannot even walk to
 *         the PCI Express capability, let alone write DEVCTL. No udev rule
 *         changes that. The bring-up is therefore BEST EFFORT and says loudly
 *         what it could not do; `tools/60-foundry-blackhole.rules` carries a
 *         `setpci CAP_EXP+8.w=5000:7000` so udev does it once at device-add
 *         time and a non-root run inherits a correct card.
 *   - [x] **The udev rule, exercised at last (2026-08-18).**
 *         `tools/60-foundry-blackhole.rules` + `tests/run_udev_nonroot_gate.sh`.
 *         With the rule installed, `tenstorrent` UNLOADED and no sudo anywhere
 *         in the command, an ordinary user in the `tenstorrent` group ran
 *         `ft_tt_core status` over udrv, `probe 0 --write` (which BOOTS the five
 *         RISC-Vs and byte-compares a `vector_add`: PASS 4/4) and the whole
 *         `test_phase2_vector_add` suite (18 checks, 0 failed). Red arm: the
 *         BARs put back to 0600 make the same command exit 1 with "this needs
 *         root or a udev rule", and re-triggering udev makes it pass again. A
 *         user OUTSIDE the group is still refused — the rule grants, it does not
 *         open.
 *   - **THE /tmp OWNERSHIP TRAP IS CLOSED, and it had a second half nobody had
 *     named.** `mkdir(0777)` under root's umask 022 leaves a **0755 root-owned**
 *     `/tmp/ft_tt_udrv.d`, and every later non-root job then fails to claim.
 *     `ensure_directory()` now chmods unconditionally and the claim files are
 *     `fchmod`ed to 0666. The second half: the failure was REPORTED as "every
 *     one of the 202 windows is claimed by a live process" on a completely idle
 *     card, because the claim loop treated any non-OK as contention. It now
 *     names the real cause.
 *   - [ ] (superseded, kept for the record) Bring the card up after a reset with NO tt-kmd present. HALF CLOSED:
 *         we can now send ASIC_STATE0 and POWER_SETTING ourselves, which is what
 *         the chardev open was doing for us, and both were proved on a live
 *         card. What is still untested is that sequence run AFTER an actual
 *         reset with the module absent — i.e. that the ARC comes back ready for
 *         messages on its own. Nothing here is a NEW unknown; it is the same
 *         sequence at a moment nobody has stood in yet.
 *
 * ===========================================================================
 * §1. WHAT THE SIX IOCTLS ACTUALLY DO
 * ===========================================================================
 * Foundry uses exactly six of tt-kmd's ioctls. Three of them never touch the
 * card, two are pure arbitration, and one programs hardware.
 *
 *   GET_DEVICE_INFO                  constants + PCI config values. All of it
 *                                    is in sysfs (vendor/device/subsystem_*).
 *   GET_DRIVER_INFO                  the driver's own version. Moot.
 *   QUERY_MAPPINGS                   pci_resource_len() on BARs 0/2/4 plus
 *                                    mmap cookies the driver invented. Derived
 *                                    from `/sys/.../resource` instead.
 *   ALLOCATE_TLB                     a bit in a per-device bitmap. NOTHING is
 *   FREE_TLB                         written to the card — but this is NOT
 *                                    therefore trivial. It is the arbitration
 *                                    that stops two processes claiming the
 *                                    same one of the 202 windows, and the card
 *                                    is meant to host several jobs at once.
 *                                    Replaced by `ft_tt_udrv_claim.[ch]`,
 *                                    which is the one module here that is
 *                                    finished, because it touches no hardware.
 *   CONFIGURE_TLB                      THE ONLY ONE THAT PROGRAMS HARDWARE:
 *                                    three iowrite32 to BAR0 + 0x1FC00000 +
 *                                    id*12, plus one zeroing write to the
 *                                    strided register for id < 32.
 *
 * ===========================================================================
 * §2. WHAT PROBE DOES TO HARDWARE — the replacement obligation
 * ===========================================================================
 * `blackhole_init_hardware()` is four steps:
 *
 *   1. pcie_set_readrq(pdev, 4096) — MRRS. A PCIe capability config write
 *      (DEVCTL bits 14:12), reachable through `/sys/.../config`.
 *   2. ARC message ASIC_STATE0 (0xA0), no payload — the power-up handshake.
 *   3. ARC message SET_WDT_TIMEOUT (0xC1), payload[0] = auto_reset_timeout
 *      seconds * 1000 (module default 10 s => 10000).
 *   4. fw_log_setup() — hands a coherent host DMA buffer to the ARC for
 *      firmware logging. See §4: we do not need it.
 *
 * And open()/close() are NOT neutral. `tt_cdev_open()` sets
 * a legacy fd's power state to `FT_TT_KMD_POWER_FLAGS_ALL & ~FT_TT_KMD_POWER_FLAG_MAX_AI_CLOCK`
 * with validity FT_TT_KMD_POWER_VALIDITY(15, 0), then aggregates over all open fds
 * and sends ARC POWER_SETTING (0x21). With exactly one legacy fd open that is
 *
 *      header = 0x21 | (0x0F << 8) | (0x7FFE << 16) = 0x7FFE0F21, payload 0
 *
 * and on LAST CLOSE the aggregation loop runs over an empty list, so
 *
 *      header = 0x21 | (0x0F << 8) | (0x0000 << 16) = 0x00000F21, payload 0
 *
 * i.e. every power feature off and AICLK at minimum. `blackhole_class` does
 * not set `defer_idle_powerdown`, so that lands SYNCHRONOUSLY at last close —
 * there is no grace window on this part.
 *
 * Two things follow that are easy to miss. The tt-kmd default deliberately
 * CLEARS MAX_AI_CLK ("legacy client: default to AICLK=Low"), so every number
 * this project has ever measured was taken with the low-clock request in
 * force. And the powerdown at last close is why a card that has been idle
 * behaves differently from one that has not.
 *
 * **ALL FOUR OF THESE ARE NOW OURS TO SEND, AND THREE ARE PROVEN ON SILICON
 * (2026-08-17, `ft_tt_udrv_arc_probe.[ch]`):** ASIC_STATE0, SET_WDT_TIMEOUT and
 * POWER_SETTING were sent through our own transport and answered with firmware
 * status 0, and POWER_SETTING's effect was measured on the AI clock. MRRS needs
 * no message at all — it is DEVCTL, which the reset round's config restore
 * rewrites. `fw_log_setup()` is not implemented by decision (§4).
 *
 * We ask for `FT_TT_KMD_POWER_FLAGS_ALL` (0x7FFF), i.e. tt-kmd's own legacy word plus
 * bit 0, which is what every other part of this backend already asks tt-kmd for
 * through SET_POWER_STATE. `ft_tt_udrv_arc_power_request_build(false, ...)`
 * reproduces the driver's 0x7FFE exactly, and is what we send on release so the
 * card and the driver's model of it agree again.
 *
 * ===========================================================================
 * §3. THE ARC WATCHDOG — an open question, deliberately not assumed
 * ===========================================================================
 * SET_WDT_TIMEOUT is sent once at probe and NOTHING in tt-kmd ever pets it:
 * `git grep` over the whole module finds ARC_MSG_TYPE_SET_WDT_TIMEOUT in
 * exactly one place, blackhole_init_hardware(). There is no timer, no work
 * item, and no ioctl that re-arms it. Two readings are consistent with that:
 *
 *   (a) the timer is firmware-side and watches the FIRMWARE, not the host, so
 *       a host that never speaks again is fine; or
 *   (b) the timer watches host liveness and 10 s is simply longer than any
 *       gap tt-kmd's own traffic leaves.
 *
 * **ANSWERED ON SILICON, 2026-08-17: (a). THERE IS NO HOST HEARTBEAT
 * OBLIGATION.** The timer is firmware-side and watches the FIRMWARE. The arm:
 * SET_WDT_TIMEOUT sent explicitly at 10000 ms and answered with status 0 (so
 * this firmware does implement it and did not ignore the message), then the
 * device held open doing NOTHING for 35 s — 3.5x the timeout, no NoC access, no
 * ioctl, no message. A reset in that window had three independent ways to show
 * and none of them did:
 *
 *   the LINK      a card reset drops it to Gen1 (measured repeatedly, §5(a)).
 *                 It read 5 GT/s x1 before and after.
 *   the HEARTBEAT `tt_heartbeat` is a firmware counter; a reboot restarts it.
 *                 It advanced by 350 with no discontinuity.
 *   the AI CLOCK  the hold was deliberately taken at 1350 MHz, a state NOTHING
 *                 on that machine but our own POWER_SETTING produces — tt-kmd
 *                 was asking for the minimum from two open fds. A reset loses
 *                 the request. It was still 1350 afterwards.
 *
 * And the queue was still at the same address and still answered a TEST.
 *
 * The driver still sends SET_WDT_TIMEOUT EXPLICITLY rather than inheriting
 * whatever the previous owner left, because a previous owner's value is not
 * knowable and the failure mode of the wrong one is the card resetting
 * underneath us.
 *
 * WHAT THIS ARM DOES NOT PROVE, stated because the rehearsal has to carry it:
 * 35 s is 3.5 timeouts, not an afternoon. It rules out "the watchdog fires on
 * host silence at the timeout"; it does not rule out a much longer firmware
 * timer that nothing here would have tripped. The counter-evidence is that an
 * idle card with tt-kmd loaded and no client open goes hours without ARC
 * traffic and does not reset, which is the state this bank sits in every day.
 *
 * ===========================================================================
 * §4. FIRMWARE LOGGING — we do not need it, and here is why
 * ===========================================================================
 * `fw_log_setup()` allocates a coherent host buffer and hands its DMA address
 * to the ARC (FW_LOG 0xC7 / SETUP). It is the ONLY thing in our whole path
 * that would require the card to read host memory, and it is the ONLY thing
 * that would require an interrupt (fw_log_interrupt() is the MSI handler).
 *
 * Dropping it costs a diagnostic channel and nothing else: it is best-effort
 * in the driver too ("Best-effort: start firmware log forwarding"), a failure
 * is a dev_warn, and it feeds a debugfs/parser path Foundry has never read.
 * Keeping it would drag in dma_alloc_coherent-equivalent host memory, a
 * physical address, and MSI plumbing — i.e. it would single-handedly
 * reintroduce every hard part of a userspace driver that this card otherwise
 * does not have. DECISION: not implemented, not planned.
 *
 * ===========================================================================
 * §5. WHAT WE LOSE — and whether it is replaceable
 * ===========================================================================
 * Measured against tt-kmd 2.11.1-pre as installed on the card host.
 *
 * (a) RESET_DEVICE. FOUR of the five destructive flags are replaceable and one
 *     is trivially so; this is the loss weighted highest and it turns out to be
 *     the smallest.
 *
 *       RESET_PCIE_LINK (flag 1) is `pcie_hot_reset_and_restore_state()`, a
 *       secondary-bus reset on the upstream bridge — exactly what
 *       `/sys/bus/pci/devices/0000:12:00.0/reset` does, and that file's
 *       `reset_method` on this box reads `bus`. REPLACEABLE, already present.
 *
 *       CONFIG_WRITE (flag 2) is `pcie_timer_interrupt()` = two
 *       PCI CONFIG dword writes: 0x934 <- 1, then 0x930 <- 0x11. Config space
 *       on this box is 4096 bytes, so both offsets are reachable through
 *       `pwrite()` on `/sys/.../config`. REPLACEABLE.
 *
 *       ASIC_RESET (flag 4) — the rung 3 that this project's recovery ladder
 *       actually depends on — is `set_reset_marker()` + `pcie_timer_interrupt()`
 *       (in the driver), i.e. set PCI_COMMAND.PARITY (config 0x04 bit 6) and
 *       then the same two dword writes. THREE CONFIG WRITES. REPLACEABLE.
 *
 *       ASIC_DMC_RESET (flag 5) is two ARC messages — TEST (0x90) to prove the
 *       queue is alive, then TRIGGER_RESET (0x56) with payload[0] = 3
 *       (in the driver). REPLACEABLE by ft_tt_udrv_arc_msg.
 *
 *       What is NOT free is the bookkeeping tt-kmd does AROUND the reset:
 *       bump_reset_gen, vma_zap, reclaim TLBs, reclaim iATUs, and the
 *       save/restore of PCI state (`pci_save_state`/`pci_restore_state`).
 *       Config-space save/restore has no sysfs equivalent, so we take our own
 *       snapshot and write it back. **BUILT AND PROVEN ON SILICON 2026-08-17**
 *       in `ft_tt_udrv_reset.[ch]`: the ASIC reset recovered a deliberately
 *       wedged core in place, no PCI remove and no rescan.
 *
 *       `remove` + `rescan` was the alternative and it is RULED OUT as the
 *       primary path, not merely unchosen. A rescan re-runs BAR ASSIGNMENT, and
 *       this card only re-enumerates at all because its flash set
 *       `pcie_bar4_size = 0`; with the stock 32 GiB BAR4 request a 36-bit host
 *       cannot place the BAR and the device never comes back
 *       (`docs/p150a-flashing.md`). An in-place restore writes back the
 *       addresses the PCI core already chose and cannot fail that way. `remove`
 *       is also unscopeable and needs the bus-wide `/sys/bus/pci/rescan`. It is
 *       kept as an explicit fallback (`--udrv-reenumerate`), not deleted.
 *
 *       And rung 3's remove + rescan was never a hardware requirement: tt-kmd's
 *       POST_RESET clears `needs_hw_init` before deciding whether it can
 *       restore, so a POST_RESET issued before the card
 *       clears the reset marker loses the restore permanently. We wait for the
 *       marker first. RESIDUAL EXPOSURE: bringing the card up after a reset
 *       with NO tt-kmd loaded is still untested — the reset needs no ARC
 *       message, but nothing then sends ASIC_STATE0 or POWER_SETTING.
 *
 * (b) tenstorrent_vma_zap. Revokes every userspace BAR mapping
 *     across a reset so a stale pointer cannot touch a device that has moved
 *     underneath it. NOT REPLACEABLE as a mechanism — one process cannot
 *     unmap another's pages. It is also LARGELY UNNEEDED here: its purpose is
 *     to protect processes that did not ask for the reset, and in the
 *     userspace-driver design the process asking for the reset is the only
 *     process holding a mapping. The replacement is a discipline, not a
 *     mechanism: munmap our own apertures before resetting and re-map after,
 *     and hold the card lock (`tools/ft_tt_card_lock.c`) across the whole
 *     sequence. EXPOSURE: a second process that mapped the BAR behind our back
 *     keeps a live mapping across a reset. That is a real hole and it is the
 *     honest cost of leaving the kernel; it is bounded by the fact that the
 *     BAR files are root-owned 0600.
 *
 * (c) The sysfs/hwmon telemetry tree (`/sys/.../tenstorrent/`, hwmon
 *     asic_temp/vcore/current/power/fan_rpm). REPLACEABLE and already
 *     replaced: `ft_tt_arc_telemetry.c` reads the same telemetry table over
 *     the NoC from tile (8,0), which is where tt-kmd reads it from too.
 *     What is lost is the ability for a THIRD PARTY —
 *     tt-smi, a monitoring agent, `sensors` — to read temperature without
 *     going through Foundry. EXPOSURE: operational visibility, not
 *     correctness. Note tt-smi is already dead on this host (AVX SIGILL,
 *     DEV_NOTES §2), so nothing on this machine actually loses a reader.
 *
 * (d) SET_NOC_CLEANUP's crash-time write. Registers ONE
 *     32-bit NoC write that the kernel performs when the fd closes for any
 *     reason, including SIGKILL. NOT REPLACEABLE — the whole value of it is
 *     that it survives the death of the process, which is precisely what a
 *     userspace driver cannot promise. EXPOSURE: this is the "an abort must
 *     DRAIN" hazard with the last safety net removed. Note carefully that
 *     Foundry does not use it today (no caller anywhere in `tt/`), so nothing
 *     REGRESSES — but the option to fix `SIGTERM does not retire card
 *     residencies` (plan Phase 7) by registering a stop-flag write with the
 *     kernel goes away, and that was the cheap fix. The remaining answers are
 *     a signal handler (which the plan already rejects as unsafe from a
 *     handler), the residency idle-spin bound (which already exists and is
 *     what makes an idle stop safe), or a supervisor process. RECOMMENDATION:
 *     treat the idle-spin bound as the actual guarantee and say so, rather
 *     than adopting the ioctl now and losing it later.
 *
 * (e) RECOVERY GRANULARITY — a design question, not just a replacement
 *     obligation. tt-kmd exposes reset as a fixed menu (link / config-write /
 *     ASIC / ASIC+DMC) and every entry on it resets the WHOLE CARD. On a
 *     140-core part that is meant to host several jobs at once (see
 *     `ft_tt_udrv_claim.h`), the blast radius IS the problem: one job's wedged
 *     core costs every other job its work. Foundry's rungs 0-2 are the
 *     targeted, per-core attempts and they are recorded as FAILING on the
 *     `0x00047000` fingerprint, with only rung 3 — the card-wide ASIC reset —
 *     working.
 *
 *     That record is evidence the targeted path does not work TODAY. It is not
 *     evidence it cannot. Three explanations are separable, and the tree
 *     already contains enough to rank them:
 *
 *       (i)   The stuck state is in the NoC FABRIC, so no per-core action can
 *             clear it. LARGELY REFUTED ALREADY: a core wedged in the
 *             `tensix_sync()` family "still answer[s] NoC reads",
 *             and the whole 15-core incident recovered core by
 *             core once the cause was removed. A fabric that had lost the core
 *             would not answer.
 *       (ii)  The rungs are simply not doing the right thing and nobody dug,
 *             because rung 3 always worked.
 *       (iii) The rungs address the WRONG UNIT. This is the reading the code
 *             already states: rung 0 deasserts RISC-Vs, rung 1 reboots
 *             firmware, rung 2 writes the value a PROCESSOR is waiting for —
 *             all three assume something is executing that can observe a host
 *             write. The recorded wedges are in the Tensix BACKEND (the matrix
 *             engine), which executes nothing (`ft_tt_core_recover.h`, @status).
 *
 *     MY READING: (iii), with (ii) as a subset — the rungs are not wrong so
 *     much as aimed at the processors when the stuck unit is the backend. And
 *     the uncomfortable corollary is that the ONE per-core register that would
 *     reach the backend is `0xFFB121B0`'s matrix-engine bits, which are on
 *     this project's permanent do-not-touch list: asserting them silently and
 *     permanently corrupts that core's compute and cost 14 cores once.
 *     **So per-core recovery from this fingerprint is blocked
 *     by a hazard, not by unfinished work**, and a userspace driver does not
 *     change that — it only removes the ioctl wrapper around the card-wide
 *     reset that does work.
 *
 *     THAT EXPERIMENT IS RUN AND ANSWERED, 2026-08-17. Core 101 was wedged
 *     deliberately at 0x00047000; it still answered NoC reads and all five of
 *     its neighbours probed PASS, so the fabric was intact and (i) is dead. An
 *     ASIC reset with NO PCI remove and NO rescan then recovered it — `probe
 *     --write` FAIL before, PASS 4/4 after. **So the stuck state is ASIC
 *     configuration cleared at reset, and re-enumeration was never doing the
 *     work.** That leaves (iii) and keeps a narrower per-core equivalent
 *     PLAUSIBLE — but the only per-core register that reaches the backend is
 *     still `0xFFB121B0`'s matrix-engine bits, permanently banned, so this is a
 *     direction for Quasar and not a plan for Blackhole.
 *
 * ===========================================================================
 * §6. WHAT "NO KERNEL MODULE" DOES AND DOES NOT MEAN
 * ===========================================================================
 * It means no VENDOR module: no tt-kmd, no DKMS build, no kernel headers on
 * the user's machine, no signing for Secure Boot, no version skew with the
 * running kernel. It does NOT mean no kernel code — the PCI core still
 * enumerates the device and sysfs still publishes it. The device must be
 * unbound from any driver (or bound to vfio-pci), and the sysfs resource
 * files are root-owned, so this needs root or a udev rule. Both are packaging,
 * not engineering.
 *
 * **IT ALSO DOES NOT MEAN NO ROOT AT ALL, and the line is exactly where PCI
 * CONFIG SPACE begins (measured 2026-08-18).** With
 * `tools/60-foundry-blackhole.rules` installed, an ordinary user in the
 * `tenstorrent` group does the whole DATA path with no privilege: map both BAR
 * mappings, claim windows, aim them, move bytes, boot firmware on a Tensix, read
 * the ARC telemetry table, take the per-core flock claims. What that user CANNOT
 * do is anything in config space — MRRS and the ASIC reset (and therefore
 * `ft_tt_core recover`) — because the kernel truncates an unprivileged read of
 * `config` to the 64-byte standard header and refuses the writes outright. So on
 * the target board: **serving is unprivileged, recovery is privileged**, and the
 * udev rule's `setpci` line is what keeps the one privileged bring-up step off
 * the run-time path.
 */

#ifndef FT_TT_UDRV_H
#define FT_TT_UDRV_H

/** C99 has no `_Static_assert`; this is the classic negative-array-size form,
 *  the same one ft_tt_kmd_abi.h uses. Restated rather than included so this
 *  module carries no dependency on the layer it replaces. */
#define FT_TT_UDRV_STATIC_ASSERT(cond, tag) \
    typedef char ft_tt_udrv_static_assert_##tag[(cond) ? 1 : -1]

/**
 * Status codes for the userspace-driver layer. The numeric values match
 * `ft_tt_device.h`'s enum deliberately, so a caller that already maps those to
 * messages needs no second table — but this module does NOT include that
 * header, because the whole point is that it does not depend on the tt-kmd
 * path it replaces.
 */
enum {
    FT_TT_UDRV_OK = 0,
    FT_TT_UDRV_ERROR = -1,   /**< unclassified failure */
    FT_TT_UDRV_ENOMEM = -2,  /**< host allocation failed */
    FT_TT_UDRV_EINVAL = -3,  /**< bad argument */
    FT_TT_UDRV_EIO = -4,     /**< a read/write/mmap failed; errno was logged */
    FT_TT_UDRV_ENODEV = -5,  /**< no such device, or not the part we expect */
    FT_TT_UDRV_EAGAIN = -6,  /**< ring full, or no response ready yet */
    FT_TT_UDRV_ENOTSUP = -7  /**< a declared seam that is not implemented yet */
};

#endif /* FT_TT_UDRV_H */
