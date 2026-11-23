# BAR Swap — Dynamic Die Multiplexer for NVIDIA K80 GPUs

> Kernel module + userspace library that time-multiplexes 6 K80 dies through a single working MMIO slot (0xC00000000) via dynamic acquire/release.

---

## Overview

The GPU server has 3x NVIDIA K80 cards (6 GK210 dies, 72 GB VRAM total). Linux's PCI allocator fails to assign 16 GB prefetchable BARs because the BIOS doesn't provision enough MMIO space behind the nested PLX bridge hierarchy. BAR Swap fixes this by directly programming PCI config space and kernel resource structs on demand.

**v2.0** upgrades from static single-die assignment to a dynamic multiplexer — any die can be swapped into the MMIO slot at runtime. One die active at a time, VRAM persists across swaps.

## Problem

K80 cards use a PLX PCIe switch that creates a 3-level bridge hierarchy:

```
Root Port (00:01.0)
  └── PLX Upstream (01:00.0)
        ├── PLX Downstream (02:08.0) → GK210 Die 0 (03:00.0)
        └── PLX Downstream (02:10.0) → GK210 Die 1 (04:00.0)
```

Each GK210 die needs:
- **BAR 0**: 16 MB non-prefetchable (registers)
- **BAR 1**: 16 GB prefetchable (VRAM)
- **BAR 3**: 32 MB prefetchable (NV control)

The kernel's bridge window allocator sees insufficient MMIO space behind the root port and marks all BARs as `[virtual]` (unassigned). The nvidia driver then fails to load.

### Address Constraints

BAR 1 (16 GB) requires 16 GB alignment. Available aligned addresses in the DSDT window:

| Address | Status | Notes |
|---------|--------|-------|
| `0x400000000` | Blocked | NIC (r8169) at `0x61F400000` sits inside this range |
| `0x800000000` | Broken | nvidia `RmInitAdapter` times out — CPU/firmware routing issue |
| `0xC00000000` | Works | **Only viable slot** for BAR 1 on this Ivy Bridge system |

Since only one address works, we time-multiplex all 6 dies through it.

## Solution — Dynamic Multiplexer (v2.0)

### Architecture

```
                    ┌─────────────────────────────┐
                    │     MMIO Slot (0xC00)        │
                    │  BAR0=0xF3  BAR1=0xC00       │
                    │  BAR3=0xF4                    │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         Card A (0,1)    Card B (2,3)     Card C (4,5)
         ┌───┴───┐       ┌───┴───┐       ┌───┴───┐
        Die 0  Die 1    Die 2  Die 3    Die 4  Die 5
```

At any time, **one die** occupies the slot. Userspace calls `bs_acquire(die_id)` to program a die in, `bs_release()` to vacate. Acquiring a new die while one is active performs an implicit release.

### Kernel Module Operation

1. **Discovery** — find all GK210 dies, walk bridge hierarchy upward from each to build per-die bridge chains
2. **Init** — release ALL existing GPU resources, disable ALL dies, register `/dev/bar_swap`. Slot starts **empty**
3. **Acquire(die_id)** — under mutex:
   - If different die active, release it (disable decode, release resources, zero BARs)
   - SBR on target die's downstream bridge (first acquire only)
   - Disable decode on target chain
   - Program bridge windows top-down (root port → PLX up → downstream)
   - Program die BARs (BAR0=0xF3, BAR1=0xC00, BAR3=0xF4)
   - Claim resources top-down into iomem tree
   - Enable decode top-down
4. **Release(die_id)** — under mutex:
   - Disable decode on die + bridges
   - Release claimed resources
   - Zero BAR config space
5. **Exit** — release active die, restore all original bridge/BAR state, drop references

### Why Direct Manipulation

- `pci_update_resource()` is not exported to modules — we replicate it with `pci_write_config_dword()`
- `pci_claim_resource()` and `pci_release_resource()` are exported in kernel 6.8
- nvidia reads BAR addresses from `pci_dev->resource[]` structs (not config space)
- nvidia calls `request_mem_region()` which checks the iomem tree — so claiming is required

## Address Layout (v2.0)

### Single MMIO Slot — All Dies Share These Addresses

| Address | Size | Assignment |
|---------|------|------------|
| `0xC00000000` | 16 GB | BAR 1 (VRAM) — prefetchable |
| `0xF3000000` | 16 MB | BAR 0 (registers) — non-prefetchable |
| `0xF4000000` | 32 MB | BAR 3 (NV control) — non-prefetchable |

### Bridge Windows

All bridges in the active die's chain get the same windows:

| Window | Range |
|--------|-------|
| Prefetchable | `0xC00000000 – 0xFFFFFFFFF` |
| Non-prefetchable | `0xF3000000 – 0xF5FFFFFF` |

## Files

```
foundry/barswap/
├── bar_swap_ko/
│   ├── bar_swap.h      # Shared ioctl definitions (kernel + userspace)
│   ├── bar_swap.c      # Kernel module v2.0
│   └── Makefile         # Kbuild
├── barswap.h            # Userspace library API
├── barswap.c            # Userspace library (ioctl + VRAM mmap)
└── tests/
    ├── test_barswap.c   # Unit tests (14 tests)
    ├── vram_persist.c   # VRAM data persistence test (Architecture A)
    └── compute_persist.c # GPU compute persistence test (parallel MoE)
```

## Kernel Module API

The module registers `/dev/bar_swap` as a misc device with three ioctls:

| Ioctl | Direction | Description |
|-------|-----------|-------------|
| `BAR_SWAP_QUERY` | Read | Returns die count, slot count, per-die mapped status, active die |
| `BAR_SWAP_ACQUIRE` | Read/Write | Programs a die into the MMIO slot, returns BAR addresses |
| `BAR_SWAP_RELEASE` | Write | Releases a die from the MMIO slot |

### Ioctl Structs

```c
struct bs_query_info {
    __u32 n_dies;           /* total GK210 dies discovered */
    __u32 n_slots;          /* always 1 */
    __u32 mapped[16];       /* mapped[i] = 1 if die i is in slot */
    __u32 active_die;       /* die in slot, or 0xFFFFFFFF if empty */
};

struct bs_acquire_req {
    __u32 die_id;           /* in: which die to acquire */
    __u32 flags;            /* in: BS_ACQUIRE_FORCE_SBR */
    __u64 bar0_addr;        /* out: BAR 0 physical address */
    __u64 bar1_addr;        /* out: BAR 1 (VRAM) physical address */
    __u64 bar1_size;        /* out: BAR 1 size in bytes */
    __u64 bar3_addr;        /* out: BAR 3 physical address */
};

struct bs_release_req {
    __u32 die_id;           /* in: which die to release */
    __u32 flags;            /* reserved, must be 0 */
};
```

## Userspace Library API

```c
/* Context management */
bs_context* bs_init(void);                       /* open device, cache state */
void        bs_free(bs_context *ctx);            /* close device */

/* Query state */
uint32_t    bs_num_dies(const bs_context *ctx);  /* total GPU dies */
uint32_t    bs_num_slots(const bs_context *ctx); /* MMIO slots (always 1) */
int         bs_is_mapped(const bs_context *ctx, int die_id);  /* 1 if active */
int         bs_active_die(bs_context *ctx);      /* active die or -1 */
int         bs_refresh(bs_context *ctx);         /* re-query kernel */

/* Acquire/release */
bs_gpu*     bs_acquire(bs_context *ctx, int die_id);  /* program die, mmap VRAM */
void        bs_release(bs_context *ctx, bs_gpu *gpu); /* unmap, release die */
```

### bs_gpu Handle

```c
typedef struct bs_gpu {
    int die_id;             /* which die this handle refers to */
    uint64_t bar1_addr;     /* VRAM physical address */
    uint64_t bar1_size;     /* VRAM size (16 GB) */
    void *vram;             /* mmap'd VRAM pointer (NULL if mmap failed) */
    size_t vram_mapped_size;
    int mem_fd;             /* /dev/mem fd */
} bs_gpu;
```

### Usage Example

```c
bs_context *ctx = bs_init();

/* Acquire die 0, use its VRAM */
bs_gpu *gpu = bs_acquire(ctx, 0);
if (gpu && gpu->vram) {
    memset(gpu->vram, 0, 4096);  /* zero first page */
}
bs_release(ctx, gpu);

/* Switch to die 3 on a different card */
gpu = bs_acquire(ctx, 3);
/* ... use VRAM ... */
bs_release(ctx, gpu);

bs_free(ctx);
```

## Build & Test

```bash
# Kernel module (on server only)
cd foundry/barswap/bar_swap_ko && make clean && make

# Userspace library (anywhere)
cd foundry && make barswap

# Run tests (14 tests — gracefully skip device-dependent tests on dev machine)
cd foundry && make test-barswap
```

## Deployment

```bash
# 1. rsync to server
rsync -avz foundry/ user@k80-host:foundry/

# 2. Build kernel module
ssh user@k80-host 'cd foundry/barswap/bar_swap_ko && make clean && make'

# 3. Unload nvidia + old module
ssh user@k80-host 'sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia bar_swap 2>/dev/null'

# 4. Load new module
ssh user@k80-host 'sudo insmod foundry/barswap/bar_swap_ko/bar_swap.ko'
ssh user@k80-host 'dmesg | tail -30'
# Should show: 6 dies discovered, slot empty

# 5. Build + run userspace tests
ssh user@k80-host 'cd foundry && make test-barswap'

# 6. Manual: acquire a die, load nvidia, verify
ssh user@k80-host '
  cd foundry && make test-barswap   # acquire/release all 6 dies
  # Then from another program: acquire die 1 → modprobe nvidia → nvidia-smi
'
```

## Module Lifecycle

- **Load**: discovers all 6 dies with per-die bridge chains, releases all GPU resources, disables all dies, registers `/dev/bar_swap` with slot empty
- **Runtime**: userspace issues ACQUIRE/RELEASE ioctls to swap dies in and out of the single MMIO slot
- **Unload**: releases active die (if any), restores all original BAR/bridge config space values per-die, drops PCI references

The module saves all original config space values at discovery time and restores them on `rmmod`, making it safe to load/unload without rebooting.

## Prerequisites

- DSDT patched with expanded MMIO window (via `dsdt_override.cpio` in GRUB)
- GRUB: `pci=hpmemsize=0 pci=hpmmioprefsize=0` (prevents allocator from trying)
- nvidia driver **fully blacklisted** — all 4 modules need both `blacklist` and `install ... /bin/false` in `/etc/modprobe.d/blacklist-nvidia.conf`, with `update-initramfs -u` applied:
  ```
  blacklist nvidia
  blacklist nvidia_drm
  blacklist nvidia_modeset
  blacklist nvidia_uvm
  install nvidia /bin/false
  install nvidia_drm /bin/false
  install nvidia_modeset /bin/false
  install nvidia_uvm /bin/false
  ```
- Kernel headers installed for module build (`linux-headers-$(uname -r)`)

## Tests

14 tests total (6 Phase 1 + 8 Phase 2). Device-dependent tests gracefully skip when `/dev/bar_swap` is not present.

| Test | Phase | What it verifies |
|------|-------|-----------------|
| `init_free` | 1 | Context lifecycle, NULL-safe free |
| `null_safety` | 1 | All query functions return safe defaults for NULL ctx |
| `is_mapped_bounds` | 1 | Out-of-range die_id returns 0 |
| `num_dies` | 1 | At least 2 dies discovered |
| `num_slots` | 1 | Exactly 1 slot |
| `slot_starts_empty` | 1 | No dies mapped after module load |
| `acquire_null_ctx` | 2 | Acquire with NULL context returns NULL |
| `acquire_invalid_die` | 2 | Acquire with -1 or 99 returns NULL |
| `release_null` | 2 | Release with NULL arguments is safe |
| `acquire_basic` | 2 | Acquire die 0, verify addresses, release, verify empty |
| `acquire_all_dies` | 2 | Acquire/release each die in sequence |
| `active_die_tracking` | 2 | active_die reports correct state through acquire/release |
| `implicit_release` | 2 | Acquiring die B while A active works correctly |
| `vram_roundtrip` | 2 | Write pattern to VRAM via mmap, read back |

## Persistence Tests

Beyond the 14 unit tests, two standalone tests verify critical hardware properties:

### VRAM Persistence (`vram_persist`)

Tests whether GPU VRAM data survives across nvidia unload/reload cycles. Result: **PASS** — Architecture A confirmed. CUDA contexts and allocated buffers persist across bar_swap release/acquire without destroying the CUDA context or unloading nvidia.

```bash
make vram-persist
sudo ./barswap/tests/vram_persist [die_id]
```

### Compute Persistence (`compute_persist`)

Tests whether in-flight GPU computation continues executing after bar_swap release unmaps the BARs. Launches a 16.5s kernel (1M elements × 10M iterations), releases the die immediately (without syncing), waits 49s, re-acquires, and checks results.

```bash
make compute-persist
sudo ./barswap/tests/compute_persist [die_id]
```

**Result: PASS** — All 1,048,576 elements correct (100%). `cuCtxSynchronize` returned in 0.02ms after re-acquire (work already completed). The GPU's streaming multiprocessors continue executing queued kernels even when BARs are unmapped — the die is still powered and clocked independently of the MMIO mapping.

**Implication:** Enables true parallel multi-die computation. Launch async work on die A, release, launch on die B, release — both dies compute simultaneously. Re-acquire each to collect results. This makes top-2 MoE routing cost the same wall-clock time as top-1.

**Prerequisites:**
- `bar_swap.ko` loaded
- Run as root (test loads nvidia internally)
- nvidia must NOT be pre-loaded (test acquires die first, then loads nvidia + nvidia_uvm)

**nvidia loading note:** Each module must be loaded with separate `modprobe --ignore-install` calls. Loading `nvidia_uvm` is required for `cuInit` to succeed (error 999 without it).

## Current Status

- **v2.0** — dynamic die multiplexer, 6 dies time-multiplexed through single slot
- **v1.5** — 1 die static assignment (die 1 hardcoded), GPU validation passed

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-19 | Compute persistence confirmed — GPU kernels continue after release, enables parallel MoE |
| 2026-03-16 | v2.0 — dynamic acquire/release, per-die bridge chains, mutex, 14 tests |
| 2026-03-16 | v1.5 deployed — die 0 fully disabled (prevents nvidia lockup), GPU validation passed |
| 2026-03-16 | v1.4 deployed — 1 GPU die working, nvidia blacklist + load order documented |
| 2026-03-16 | v1.2 fixed non-pref addresses (0xE0 → 0xF0, avoiding Intel iGPU) |
| 2026-03-15 | v1.0 initial deploy — claims failed, identified address conflicts |
| 2026-03-15 | Initial version — Card 1 (2 dies) support |
