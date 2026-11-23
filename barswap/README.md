# Barswap — Canonical Quick Reference

**This is the authoritative current-state doc.** The other files in this folder are
deep-dives and history; where they show interim die counts ("1 of 2 dies", "6/6"),
those are *historical* — the current state is below. Read this first.

> One sentence: the BTC-S37 board can map **only one 16 GB GPU die into CPU
> address space at a time**, so all 16 K80 dies are **time-multiplexed** through a
> single MMIO slot via `bar_swap.ko` — acquire a die, use it, release, repeat.

## Current hardware reality (verified 2026-03-25)

| Fact | Value |
|---|---|
| Board / CPU | BTC-S37, Intel Pentium 2117U (Ivy Bridge, **36-bit phys = 64 GB**), 8 GB DDR3 |
| GPUs | 8× K80 = **16 dies, 192 GB VRAM** — **16/16 tested and passing** |
| Working MMIO slot | **exactly one**, at `0xC00000000` (16 GB) |
| Simultaneous dies | **1** (`bs_num_slots()` == 1). The other two 16 GB-aligned addresses (`0x400000000`, `0x800000000`) hang/timeout — see `MULTI_SLOT.md` |
| Multi-slot | **hardware-impossible** on this board without a risky BIOS reflash (closed; `MULTI_SLOT.md`) |
| Swap cost | ~0.2–1.2 ms BAR remap (nvidia stays loaded, context persists); ~2.5 s only if you reload nvidia |
| VRAM + context | **persist across release/acquire** — don't destroy the context on swap (Architecture A, confirmed) |
| **In-flight compute** | **continues executing on-die while the die is released** — the BAR is only a CPU access window, not needed during kernel execution (**`compute_persist` PASS 2026-06-26**) |
| Driver / toolkit | nvidia **470** (Kepler→Turing); CUDA **11.4**; K80 = **sm_37** |

**Architectural consequence — parallel compute behind a serialized control plane.**
The single MMIO slot multiplexes only the **control/IO** path (launch kernels, copy
inputs/results in and out). It is **not** a compute bottleneck: once work is launched
on a die, that die keeps computing on its local VRAM after it's released from the slot
(BAR = doorway, not the room). So all 16 dies can compute **in parallel**;
the slot is time-multiplexed (~0.2–1.2 ms/visit) just to dispatch work and collect
finished results. Effective aggregate throughput ≈ N × single-die, minus control-plane
serialization — bounded by *(slot-time per die-visit) ÷ (on-die compute-time per step)*,
which is small when compute ≫ launch/copy (true on the slow K80).

> **Load-bearing assumption — CONFIRMED (2026-06-26, `compute_persist` die 1).**
> Calibrated a 16.2 s kernel, launched it async, **released the die from the slot at
> 128 ms** (GPU unmapped from MMIO), waited **48.7 s with the die released**, then
> re-acquired: `cuCtxSynchronize` returned in 0.02 ms ("work already done") with
> **1,048,576 / 1,048,576 elements correct, 0 zeros**. In-flight GPU compute continues
> after BAR unmap — the parallel-compute model holds. Switch overhead ~0.2 ms (launch
> only); "top-2 cost = top-1 cost (parallel, not sequential)."

## Build recipe (CUDA on this box)

The default toolchain does **not** work out of the box; these are required:

- Use the **separately-installed CUDA 11.4** at `/usr/local/cuda-11.4`.
  **Do NOT** use the apt `nvidia-cuda-toolkit` package — it installs CUDA 12, which
  **dropped sm_37** (K80). 11.4 is the last toolkit supporting Kepler.
- nvcc host compiler must be **gcc-11** (`-ccbin gcc-11`). Ubuntu 24.04's default is
  gcc-13; nvcc 11.4 refuses any gcc > 11 (`unsupported GNU version`).
- glibc 2.39 needs the **`-isystem nvcc_shim`** include shim (a patched `sys/cdefs.h`),
  or nvcc dies on `error: attribute "__malloc__" does not take arguments` from
  `/usr/include/stdio.h`.

All three are already wired into `cuda_kernels/Makefile`. To (re)compile the kernels
to PTX and regenerate the embedded `cuda/fc_cuda_ptx.h`:

```bash
cd foundry/cuda_kernels && make      # nvcc -ptx -arch=sm_37 -ccbin gcc-11 -isystem nvcc_shim
```

This needs only the host CPU (no GPU, no driver). The generated `fc_cuda_ptx.h` is
committed so foundry/Pantheon build without nvcc on dev machines.

## Manual bring-up (today's procedure)

nvidia is **blacklisted on purpose** (`/etc/modprobe.d/blacklist-nvidia.conf` with
`install … /bin/false`) so it never auto-loads and corrupts GPU state. Bring it up by
hand, **in this order**:

```bash
# 1. Load the barswap module — registers /dev/bar_swap, slot starts empty
sudo insmod ~/foundry/barswap/bar_swap_ko/bar_swap.ko
sudo dmesg | grep bar_swap            # expect 16 dies listed, slot empty

# 2. Acquire ONE die into the 0xC00000000 slot (userspace lib: bs_acquire(die_id))
#    e.g. via a small program linking barswap.c, or a barswap CLI

# 3. Load the nvidia driver (it's blacklisted, so --ignore-install)
sudo modprobe --ignore-install nvidia nvidia_modeset nvidia_drm nvidia_uvm

# 4. Verify — read CACHED driver data, never probe hardware
cat /proc/driver/nvidia/gpus/*/information
```

### Hazards (this box is remote/headless — a wrong move means a physical trip)

- **Never `nvidia-smi` on a swapped die.** It probes hardware (temp/power/ECC) and can
  hang or crash the die. Use `/proc/driver/nvidia/gpus/<BDF>/information` instead.
- **Never automate PCI manipulation at boot.** A systemd service doing setpci/rescan/
  modprobe froze the box and required a physical power cycle + recovery mode. Barswap
  bring-up must be **on-demand from userspace**, never a boot unit. (The intended home
  for this is *inside Pantheon*, acquiring dies as needed.)

### Why nvidia is blacklisted — and the cleaner replacement

**Why (root cause):** at boot the 16 dies are in a broken-BAR state (single-slot
constraint — most have no valid CPU-routable BAR). If nvidia auto-loads it **binds all
16 dies** and probes them (`RmInitAdapter`); accessing the broken BARs / config space
soft-locks the kernel (`pci_conf1_read` never returns), and even when it survives,
nvidia stays bound and **corrupts state when bar_swap later disables a die's decode**.
The blacklist (`blacklist` + `install … /bin/false` + `update-initramfs -u`) is the
blunt fix: stop nvidia loading at all, load it by hand after a die is acquired.

**Why it's hacky:** it's system-wide — it kills nvidia for the *whole box*, so you can't
run a normal nvidia GPU alongside the K80s; and it needs `/etc/modprobe.d` edits +
initramfs regen that don't survive cleanly as a shipped product.

**Ship-grade replacement (planned): make `bar_swap` *bind* the dies (vfio-pci / pci-stub
pattern).** Today `bar_swap` claims die *resources* (`pci_claim_resource`) but is **not a
registered `pci_driver`** — so nothing stops nvidia binding them; hence the blacklist.
Register `bar_swap` as a `pci_driver` matching the GK210 device ID so the kernel **binds
all 16 dies to bar_swap at probe** → they're invisible to nvidia → no probe, no panic, no
corruption — *without* a global blacklist. On `acquire`, hand the one slot die to nvidia
(unbind → nvidia binds it); the other 15 stay parked with bar_swap. A box could then run
a normal nvidia GPU *and* the K80s at once. **Zero-code de-risk first:** put the GK210
ID in `pci-stub.ids=`/`vfio-pci.ids=` on the kernel cmdline and check nvidia boots clean,
before writing the bind path into `bar_swap`.
- **One die *mapped* at a time** (slot occupancy, not compute — released dies keep
  computing). Acquiring a new die implicitly releases the active one.
- **Don't destroy the CUDA context on swap** — it survives release/acquire; tearing it
  down forces a full ~2.5 s reload.
- **CUDA atexit teardown hangs on this rig (confirmed 2026-06-26).** When a process with
  a CUDA context exits normally, the atexit context-destroy wedges a thread against the
  bar_swap'd die — the process becomes a `Zl` zombie and nvidia ends up half-unloaded
  (`NVRM: kernel module is unloaded` while `lsmod` still lists it). These zombies can't
  be killed and accumulate until the box needs a power cycle (it also makes clean
  shutdown crawl). **Workaround for short-lived processes (tests, one-shots): call
  `_Exit()` instead of `return`/`exit()`** — skip the userspace teardown and let the
  kernel reclaim the context on process death (clean exit, no zombie; verified with the
  kernel-parity harness). **Proper fix (TODO, product-critical — every model process
  exit hits this):** an ordered context-destroy + `bs_release` before exit, and/or the
  device-binding driver work above. A long-lived Pantheon server avoids it by not
  exiting; it still needs a clean teardown path for shutdown.

## barswap userspace API (`barswap.h`)

```c
bs_context* bs_init(void);                       // open /dev/bar_swap
uint32_t    bs_num_dies(ctx);                    // 16 on this rig
uint32_t    bs_num_slots(ctx);                   // 1 (v2 — hardware limit)
bs_gpu*     bs_acquire(ctx, int die_id);         // program die into the slot, mmap VRAM
void        bs_release(ctx, bs_gpu*);            // vacate the slot
int         bs_active_die(ctx);                  // die in the slot, or -1
int         bs_is_mapped(ctx, int die_id);
int         bs_refresh(ctx);
void        bs_free(ctx);
```

Kernel module ioctls on `/dev/bar_swap`: `BAR_SWAP_ACQUIRE` / `…RELEASE` / `…QUERY`.

## Field service — identifying a physical card (future feature, not built)

This rig has **no BMC/IPMI** and the K80 bracket LED isn't software-programmable, so
there's no turnkey "locator LED." Two practical ways to find a specific/failing card,
worth building into the barswap-in-Pantheon tooling later:

1. **BDF → physical map (deterministic).** Every die has a fixed PCI address (die 0 =
   `03:00.0`, die 1 = `04:00.0`, Card B = `07/08:00.0`, … — see `CARDS.md`).
   A failing die's BDF → its card → its riser. **One-time win: physically label each
   riser with its BDF at install** so a bad die maps to a card by eye.
2. **`identify <die>` by workload (lights up the card).** Acquire *only* the suspect die
   and run a continuous busy kernel for ~30 s — its **activity LED flickers** while the
   others stay dark, so you can physically spot it. barswap's single-die acquire makes
   this clean. This is the small operability feature to add (a `pantheon --barswap
   identify <die>` mode).

Also useful and already possible: a *failed* K80 often shows **amber** on the bracket LED
(passive), and barswap can **disable/skip a die by BDF** in software to route around a
flaky card until it's pulled.

## Where to read more

- **`MULTI_SLOT.md`** — *why* only one slot works (Ivy Bridge 36-bit addressing; closed).
- **`CARDS.md`** — GPU compatibility (which cards need barswap; T4/P4 don't) + current rig.
- **`memoryTricks.md`** — barswap-as-scheduler notes for multi-model hosting.
