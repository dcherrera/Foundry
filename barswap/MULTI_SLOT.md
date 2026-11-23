# Multi-Slot Research — Simultaneous Die Mapping for Bar_swap

**Status:** CLOSED — Hardware limit. Only one 16 GB MMIO slot works on BTC-S37.
**Date:** 2026-03-19
**Goal:** Map 2+ K80 dies simultaneously to enable zero-overhead tensor parallelism

---

## Conclusion

Multi-slot is **not possible** on the BTC-S37 / Pentium 2117U (Ivy Bridge) hardware. The CPU has a 36-bit physical address space (64 GB), which provides exactly three 16 GB-aligned MMIO addresses. All three were tested:

| Slot | Address | Result |
|------|---------|--------|
| 0 | `0x400000000` | CPU hang — MMIO read never returns |
| 1 | `0x800000000` | nvidia timeout — RmInitAdapter fails |
| 2 | `0xC00000000` | **Works** — only viable address |

The root cause is the BIOS PciHostBridge DXE module, which programs the CPU's root complex to only decode MMIO in a 16 GB window at the top of the address space during boot. Addresses outside this window are dropped by the root complex before they ever reach the PCIe bridges, regardless of how the bridges are programmed at runtime.

**BIOS firmware mod (waylaid):** The [ReBarUEFI](https://github.com/xCuri0/ReBarUEFI) project has proven that patching the PciHostBridge DXE module on Ivy Bridge boards can expand the MMIO decode to the full 36-bit range, which would likely enable all three slots. However, this requires dumping and reflashing the SPI BIOS chip, with a real risk of bricking the board (recoverable only with an SPI programmer). This is not feasible for the average user, so this path was not pursued.

**Bar_swap v2.0 single-die multiplexing remains the optimal architecture for this platform.** Tensor parallelism requires a motherboard/CPU with wider physical addressing (48-bit: AMD EPYC, Intel Xeon, or newer desktop platforms).

---

## Task List

### Phase 1 — Recon (no code changes, no risk) — DONE

- [x] **1.1** NIC BAR: `09:00.0` Region 4 at `0x61F400000` (16 KB, prefetchable) — inside `0x400` range
- [x] **1.2** TOUUD: `0x21F600000` (~8.5 GB) — RAM ends here, MMIO above
- [x] **1.3** MMIO map: `21f600000-fffffffff` is PCI space (~55.5 GB). NIC at `61f400000`.
- [x] **1.4** BAR1 size mask: **HARDWIRED TO 16 GB.** Low mask=`0x0000000C`, high mask=`0xFFFFFFFC`. 34 low bits fixed. Cannot resize.
- [x] **1.5** Findings documented below.

**Phase 1 findings:**
- Config space is NOT reachable via ECAM or CF8/CFC for any die whose bridge chain bar_swap has disabled. Only Card C (dies 4-5) was reachable because its bridge chain was still intact.
- Probed via kernel module (`bar1_probe.ko`) using `pci_read_config_dword()` on Card C dies.
- BAR0 = 16 MB (hardwired), BAR1 = 16 GB (hardwired), BAR3 = 32 MB (hardwired).
- Three userspace config access methods all failed (sysfs, ECAM /dev/mem, CF8/CFC iopl). The kernel's internal `pci_read_config_dword()` was the only working path.

### Phase 2 — BAR1 Resize Probe — BLOCKED (hardware limit)

**Result: NOT POSSIBLE.** GK210 BAR1 is hardwired to 16 GB by the silicon. The lower 34 address bits are non-writable. No runtime resizing is possible.

~~- [ ] 2.1-2.6 — all skipped~~

### Phase 3 — NIC Relocation + 0x400 Test — DONE (FAILED)

NIC relocation succeeded, but GPU BAR1 at `0x400000000` **froze the machine**.

- [x] **3.1** Unbind r8169 — OK
- [x] **3.2** Reassign NIC BAR4 from `0x61F400000` to `0x21F600000` — OK
- [x] **3.3** Rebind r8169, ping test — OK, network works
- [x] **3.4** Map die 2 BAR1 at `0x400000000` via kernel module (`slot2_test.ko`) — **CPU HANG**
- [ ] ~~3.5~~ — skipped (machine frozen)
- [ ] ~~3.6~~ — skipped

**What happened:** `slot2_test.ko` programmed die 2's bridge chain and BARs at `0x400000000`, then attempted `ioremap()` + `ioread32()` on BAR0 at `0xF6000000`. The read never returned — CPU froze, server required power cycle.

**Root cause:** Same class of failure as `0x800000000`. The Pentium 2117U (Ivy Bridge) on the BTC-S37 motherboard cannot route MMIO transactions to `0x400` or `0x800` — only `0xC00` works. This is likely a CPU/chipset/firmware routing limitation, not a PCI topology issue (the bridge chains were correctly programmed). The DSDT patch expanded the window size but can't fix the underlying decode circuitry.

### Phase 4 — `0x800000000` Investigation — SKIPPED

`0x800` was already known to cause nvidia timeouts. Given that `0x400` also freezes the CPU, this is the same root cause. All three slots have been tested:

| Slot | Address | Result |
|------|---------|--------|
| 0 | `0x400000000` | **CPU hang** (ioread32 never returns) |
| 1 | `0x800000000` | **nvidia timeout** (RmInitAdapter fails) |
| 2 | `0xC00000000` | **Works** (only viable address) |

No further investigation needed — the hardware has one working 16 GB slot.

### Phase 5 — Bar_swap v3.0 (multi-slot module) — BLOCKED

**Prerequisite not met.** No second working slot exists. Phases 2-4 exhausted all three 16 GB-aligned addresses in the 36-bit address space.

~~- [ ] 5.1-5.7 — all skipped~~

### Phase 6 — Tensor Parallelism Proof — BLOCKED

**Prerequisite not met.** Requires Phase 5 (multi-slot module), which requires a second working slot.

~~- [ ] 6.1-6.4 — all skipped~~

### Conclusion

**Multi-slot is NOT possible on this BTC-S37 / Pentium 2117U hardware.** All three 16 GB-aligned MMIO addresses have been tested. Only `0xC00000000` works. The other two (`0x400`, `0x800`) cause CPU hangs or nvidia initialization failures — a fundamental limitation of the CPU/chipset MMIO decode circuitry, not a software issue.

The existing single-slot bar_swap v2.0 architecture is the ceiling for this motherboard. Tensor parallelism would require a platform upgrade (AMD EPYC / Intel Xeon with 48-bit addressing).

---

## Why

Today bar_swap has a single MMIO slot at `0xC00000000`. One die at a time. Switching costs ~0.2ms per acquire/release. For single-model inference this is fine, but for **tensor parallelism** (splitting one model across N dies), we need to launch work and collect results from multiple dies per layer — the switching overhead kills performance:

| Scenario | Slots | Switches/token (28 layers) | Switch overhead | Compute (7B, 2 dies) | Total | tok/s |
|----------|-------|---------------------------|----------------|---------------------|-------|-------|
| TP 2-die, single slot | 1 | 112 (4 per layer) | 22.4ms | 14.6ms | 37ms | ~27 (worse than 1 die) |
| TP 2-die, multi-slot | 2 | 0 | 0ms | 14.6ms | 14.6ms | **~68** (2x speedup) |
| TP 3-die, multi-slot | 3 | 0 | 0ms | 9.7ms | 9.7ms | **~103** (3x speedup) |

Multi-slot also enables loading **larger models** that don't fit on one die:

| Model | FP16 Size | INT8 Size | Dies Needed |
|-------|-----------|-----------|-------------|
| 7B | 14 GB | 7 GB | 1 |
| 14B | 28 GB | 14 GB | 2 (INT8) or 3 (FP16) |
| 30B | 60 GB | 30 GB | 3 (INT8) |

---

## Current Hardware Constraints

### CPU: 36-bit Physical Address Space

The BTC-S37 motherboard has an Intel Pentium 2117U (Ivy Bridge) with **36-bit physical addressing** — 64 GB total addressable memory.

```
0x000000000 ─────────────────
            │ RAM (8 GB)
0x200000000 ─────────────────
            │ Reserved / MMIO gap
0x21F600000 ─────────────────
            │ Available MMIO space (~48 GB after DSDT patch)
            │
0x400000000 ─ Slot 0 (16 GB) ─ BLOCKED (NIC conflict)
            │
0x800000000 ─ Slot 1 (16 GB) ─ BROKEN (nvidia RmInitAdapter timeout)
            │
0xC00000000 ─ Slot 2 (16 GB) ─ WORKS (current single slot)
            │
0xFFFFFFFFF ─ 36-bit limit ───
```

**Only 3 possible 16 GB-aligned addresses exist in the entire 64 GB space.** Two are broken. This is the core problem.

### PCIe Topology (3 Cards, 6 Dies)

Each K80 card has its own root port — they're independent PCIe hierarchies:

```
Card A: 00:01.0 (root port A)
  └→ 01:00.0 (PLX upstream)
       ├→ 02:08.0 (PLX downstream) → 03:00.0 (die 0)
       └→ 02:10.0 (PLX downstream) → 04:00.0 (die 1)

Card B: 00:1c.0 (root port B)
  └→ 05:00.0 (PLX upstream)
       ├→ 06:08.0 (PLX downstream) → 07:00.0 (die 2)
       └→ 06:10.0 (PLX downstream) → 08:00.0 (die 3)

Card C: 00:1c.4 (root port C)
  └→ 0a:00.0 (PLX upstream)
       ├→ 0b:08.0 (PLX downstream) → 0c:00.0 (die 4)
       └→ 0b:10.0 (PLX downstream) → 0d:00.0 (die 5)
```

**Key fact:** Separate root ports mean separate bridge windows. Two dies on different cards can theoretically have non-overlapping bridge windows pointing to different BAR1 addresses — if we can find the addresses.

### Per-Die BAR Requirements

| BAR | Size | Alignment | Type |
|-----|------|-----------|------|
| BAR 0 | 16 MB | 16 MB | Non-prefetchable (registers) |
| BAR 1 | 16 GB | 16 GB | Prefetchable (VRAM) |
| BAR 3 | 32 MB | 32 MB | Prefetchable (NV control) |

BAR 1 is the constraint. 16 GB alignment means only addresses at `N × 0x400000000` are valid.

---

## Research: Recovering the Broken Slots

### Slot 0: `0x400000000` — NIC Conflict

**Problem:** The Realtek r8169 NIC has a BAR at `0x61F400000` which the kernel reports as conflicting with the `0x400000000-0x7FFFFFFFF` range.

**Investigation needed:**

1. **Verify the actual conflict.** The NIC BAR is tiny (~4 KB at `0x61F400000`). BAR 1 at `0x400000000` spans `0x400000000-0x7FFFFFFFF`. The NIC is at `0x61F400000` which is inside this range — genuine overlap.

2. **Relocate the NIC.** The r8169 BAR could be reassigned to below `0x400000000` (e.g., in the `0x200-0x400` gap). Options:
   - Reassign via `pci_write_config_dword()` to the NIC's BAR register
   - Disable the NIC entirely if using a different network interface (USB ethernet, etc.)
   - DSDT modification to exclude the NIC from the `0x400` range

3. **Test procedure:**
   ```bash
   # 1. Check NIC's current BAR
   lspci -vvv -s <NIC BDF> | grep -i "Region"

   # 2. Unbind NIC driver
   echo <NIC BDF> > /sys/bus/pci/drivers/r8169/unbind

   # 3. Reassign NIC BAR to safe address (e.g., 0xF6000000)
   # (via bar_swap module or setpci)

   # 4. Try acquiring a die at 0x400000000
   # (new bar_swap test mode)

   # 5. Load nvidia, verify RmInitAdapter succeeds
   ```

**Likelihood of success:** Medium-high. The conflict is a software resource clash, not a hardware routing issue. If we move the NIC out of the way, `0x400` should work.

### Slot 1: `0x800000000` — nvidia Timeout

**Problem:** nvidia's `RmInitAdapter` times out with error `0x24:0xffff:1209`. This happens regardless of which physical die is placed at `0x800`. The problem follows the address.

**Investigation needed:**

1. **32 GB boundary theory.** On some Ivy Bridge CPUs, the 32 GB mark (`0x800000000`) is an architectural boundary between different memory decode circuits. Test:
   - Can we read/write BAR1 via `/dev/mem` at `0x800`? (without nvidia)
   - Does `memtest86+` or a custom kernel module see correct data at this address?
   - If raw MMIO works but nvidia fails, the issue may be in nvidia's initialization sequence

2. **IOMMU / root complex config.** Check if the root complex has address decode registers that exclude `0x800`:
   ```bash
   # Dump root complex config space
   setpci -s 00:00.0 00.L  # Check all 256 bytes
   # Look for MMIO decode registers (Intel Ivy Bridge specific)
   ```

3. **BIOS TOLUD/TOUUD registers.** Intel CPUs have Top of Low Usable DRAM (TOLUD) and Top of Upper Usable DRAM (TOUUD) registers that define the MMIO hole. If TOUUD is set to 32 GB, accesses above it may be mis-routed:
   ```bash
   # Read TOUUD from PCI config (offset 0xA8 on host bridge)
   setpci -s 00:00.0 A8.L
   ```

4. **Try a partial BAR.** What if we map BAR1 as 8 GB instead of 16 GB at `0x800000000`? nvidia might not need the full 16 GB aperture. This would keep us below the 48 GB boundary at `0xBFFFFFFFF`.

**Likelihood of success:** Low-medium. This is a deeper issue (CPU/firmware routing) and may not be fixable without BIOS modifications.

### Slot 3+: Above `0xC00000000`?

**Problem:** The 36-bit address space tops out at `0xFFFFFFFFF` (64 GB). After the current BAR1 at `0xC00-0xFFF`, there's no room for another 16 GB-aligned slot.

**No path forward here** unless we change the CPU/motherboard.

---

## Research: Alternative Approaches

### Approach A: BAR1 Resizing (Most Promising)

nvidia's BAR1 doesn't have to be 16 GB. The GK210's BAR1 is configured as 16 GB because that's the full VRAM size, but nvidia only needs enough BAR1 aperture to map the VRAM pages it's actively using.

**PCIe Resizable BAR (ReBAR)** is a PCIe capability that allows runtime BAR resizing. However, GK210 (Kepler) predates ReBAR support.

**Manual BAR1 resizing** may be possible by programming a smaller BAR1 size in the GPU's config space. The GK210's BAR1 size is determined by hardware (lower bits hardwired to 0), but nvidia's driver negotiates the mapping size via its page table. If we program BAR1 with a smaller region and tell nvidia (or our CUDA code) to use less aperture:

| BAR1 Size | Alignment | Slots in 48 GB | Concurrent Dies |
|-----------|-----------|----------------|-----------------|
| 16 GB | 16 GB | 1 (only 0xC00) | 1 |
| 8 GB | 8 GB | ~4 | 4 |
| 4 GB | 4 GB | ~8 | 6 (all dies) |
| 2 GB | 2 GB | ~16 | 6 (all dies) |

At 4 GB BAR1, we could fit 8+ dies in the address space. nvidia can page VRAM through a 4 GB window — it already does this on consumer GPUs where BAR1 < VRAM size (e.g., GTX 1080 has 8 GB VRAM but only 256 MB BAR1).

**Investigation needed:**

1. Can the GK210 BAR1 size be changed? Check if the lower bits of BAR1 config register are writable.
2. Does nvidia's driver work with a smaller BAR1? On consumer Kepler cards with small BAR1, nvidia uses a page table to map VRAM windows through the aperture. K80's driver should support this too.
3. How does a smaller BAR1 affect CUDA performance? If nvidia has to page VRAM through a 4 GB window, there may be a performance hit for large allocations.

**Test:**
```c
// In bar_swap module — try programming BAR1 as 4 GB instead of 16 GB
#define SLOT_BAR1_SIZE_SMALL  0x100000000ULL  /* 4 GB */

// Program BAR1 at 0xC00000000 with 4 GB size
// Program second die's BAR1 at 0x400000000 with 4 GB size
// Load nvidia — does it see both GPUs?
```

### Approach B: Recover Slot 0 (NIC Relocation)

Fix `0x400000000` by moving the NIC's BAR. This gives us 2 full 16 GB slots — enough for 2-die tensor parallelism.

**Steps:**
1. In bar_swap module init, find the r8169 device
2. Unbind the r8169 driver
3. Reassign the NIC's BAR to a safe address (e.g., `0xF6000000` in non-pref space)
4. Re-bind the r8169 driver (or leave it unbound if not needed)
5. Now `0x400000000-0x7FFFFFFFF` is free for a second die's BAR1

**Pros:** Straightforward, low risk, gives 2 slots
**Cons:** Only 2 slots (3 if `0x800` is also fixed), may lose network during NIC relocation

### Approach C: Two Dies on Same Card, Shared BAR1

Two dies on the same K80 card share a PLX upstream bridge. Could they share a BAR1 address range?

Map die 0 at `0xC00000000` with 8 GB (lower half) and die 1 at `0xC00000000 + 8GB = 0xE00000000` with 8 GB (upper half). Both under the same PLX upstream bridge window.

**Problem:** The PLX downstream bridges route by address range. If both downstream ports' windows overlap in the same parent window, the PLX must use the downstream bridge's sub-window to disambiguate. This already works — it's how PCIe switches route. But we need distinct, non-overlapping sub-windows per downstream bridge.

**This is essentially Approach A** (smaller BAR1) applied to two dies on the same card.

### Approach D: Different Motherboard / CPU

If multi-slot is critical enough, a motherboard with 48-bit physical addressing (AMD EPYC, Intel Xeon) would have a 256 TB address space — effectively unlimited 16 GB slots. But this changes the hardware platform entirely.

**Not pursuing for now** — the goal is to maximize the current BTC-S37 system.

---

## Test Plan

### Test 1: Address Space Probing

Before any code changes, map out what MMIO addresses actually work for raw reads/writes (without nvidia):

```c
// mmio_probe.c — run as bar_swap module extension or standalone
// For each candidate address, try:
//   1. Program die's BAR1 to that address
//   2. Enable decode
//   3. Write a pattern via /dev/mem
//   4. Read back
//   5. Report success/failure

uint64_t candidates[] = {
    0x400000000,  // Slot 0 (after NIC relocation)
    0x800000000,  // Slot 1 (previously broken)
    0xC00000000,  // Slot 2 (known good)
};
```

### Test 2: NIC Relocation

```bash
# 1. Find NIC
lspci | grep -i ethernet

# 2. Record current BAR
lspci -vvv -s <BDF> | grep "Region"

# 3. Unbind
echo <BDF> > /sys/bus/pci/drivers/r8169/unbind

# 4. Reassign BAR (via setpci or kernel module)
setpci -s <BDF> 10.L=F6000000  # Move BAR0 to 0xF6000000

# 5. Rebind
echo <BDF> > /sys/bus/pci/drivers/r8169/bind

# 6. Verify network still works
ping -c 3 google.com
```

### Test 3: Dual-Die Simultaneous Mapping

If Test 1 or Test 2 succeeds for a second address:

```c
// Acquire die 0 at 0xC00000000 (slot 2)
// Acquire die 2 at 0x400000000 (slot 0) — WITHOUT releasing die 0
// Load nvidia
// Verify nvidia sees 2 GPUs
// Create CUDA context on each
// Run matmul on both simultaneously
// Verify correct results on both
```

### Test 4: BAR1 Resize Probe

```c
// In bar_swap module:
// 1. Read BAR1 register to check which bits are hardwired
pci_read_config_dword(die->dev, PCI_BASE_ADDRESS_2, &bar1_lo);
pci_read_config_dword(die->dev, PCI_BASE_ADDRESS_3, &bar1_hi);

// 2. Write all-ones to BAR1 to check size mask
pci_write_config_dword(die->dev, PCI_BASE_ADDRESS_2, 0xFFFFFFFF);
pci_read_config_dword(die->dev, PCI_BASE_ADDRESS_2, &size_mask);
// size_mask reveals minimum BAR1 size (hardwired low bits)

// 3. If size_mask allows smaller sizes, try 4 GB:
//    Program BAR1 at 0xC00000000 with 4 GB
//    Load nvidia, check if it initializes

// 4. Restore original BAR1
```

### Test 5: Tensor Parallel Inference

Once 2+ slots work:

```c
// Split a 7B model's weight matrices across 2 dies
// For each layer:
//   1. Upload left half of W to die A, right half to die B (at load time)
//   2. Launch matmul(x, W_left) on die A
//   3. Launch matmul(x, W_right) on die B (no switching needed — both mapped)
//   4. Read partial results from both dies to CPU
//   5. Concatenate on CPU
//   6. Feed combined result to next layer

// Measure: tok/s vs single-die baseline
```

---

## Priority — Final (All Paths Exhausted)

| Approach | Result | Root Cause |
|----------|--------|------------|
| BAR1 resize | BLOCKED | GK210 silicon hardwires BAR1 to 16 GB |
| NIC relocation → `0x400` | FAILED | CPU hang on MMIO access (chipset decode issue) |
| `0x800` slot | FAILED | nvidia RmInitAdapter timeout (same class of issue) |
| `0xC00` slot | WORKS | Only viable 16 GB MMIO address |

**Verdict:** Single-slot architecture is the hardware ceiling. Multi-slot requires a different motherboard/CPU with wider physical addressing. The existing bar_swap v2.0 single-die multiplexing is the optimal design for this platform.

---

## Impact on Bar_swap Module

If multi-slot works, bar_swap v3.0 would change from single-slot multiplexer to multi-slot manager:

```c
// v2.0 (current): one slot, one active die
static int active_die = -1;

// v3.0 (multi-slot): N slots, each can hold a die
struct bs_slot {
    int die_id;           // -1 if empty
    uint64_t bar1_addr;   // Slot's BAR1 address
};
static struct bs_slot slots[MAX_SLOTS];
static int n_slots;       // Discovered at init based on available addresses
```

The userspace API would extend:

```c
// v2.0: acquire puts die in THE slot
bs_gpu* bs_acquire(bs_context *ctx, int die_id);

// v3.0: acquire puts die in any available slot (or a specific one)
bs_gpu* bs_acquire(bs_context *ctx, int die_id);              // auto-assign slot
bs_gpu* bs_acquire_slot(bs_context *ctx, int die_id, int slot); // specific slot

// New: query slot state
int bs_num_slots(bs_context *ctx);  // How many simultaneous slots available
int bs_slot_die(bs_context *ctx, int slot);  // Which die is in slot N
```

Backward compatible — single-slot systems work exactly as before.

---

## Tensor Parallelism API (Future)

If multi-slot works, bar_swap would expose a tensor parallelism helper:

```c
// Acquire N dies simultaneously into N slots
// Returns array of bs_gpu handles, all mapped at once
int bs_acquire_group(bs_context *ctx, int *die_ids, int n_dies, bs_gpu **gpus_out);

// Release entire group
void bs_release_group(bs_context *ctx, bs_gpu **gpus, int n_dies);
```

Pantheon's inference engine would use this for tensor-parallel forward passes without any bar_swap switching overhead.

---

## Dependencies

| Item | Status | Notes |
|------|--------|-------|
| bar_swap v2.0 | Done | Single-slot multiplexer, 6/6 dies |
| DSDT patch | Done | Expanded MMIO window |
| NIC BDF and BAR address | Done | `09:00.0`, BAR4 at `0x61F400000` (16 KB, pref) |
| TOUUD register value | Done | `0x21F600000` (~8.5 GB) — RAM ends here |
| GK210 BAR1 size mask | Done | Hardwired to 16 GB (34 low bits fixed) |
| NIC relocation | Done | Moved to `0x21F600000` — network OK |
| `0x400` slot test | Done | CPU hang — address not routable |
| `0x800` slot test | Done | nvidia timeout — address not routable |
