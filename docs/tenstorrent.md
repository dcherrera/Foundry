# tenstorrent

Tenstorrent Blackhole compute backend. Runs ops on a p150a's Tensix cores through Foundry's
own firmware, kernels and driver, rewritten from public documentation and measured hardware
behaviour.

**Headers:** `tt/ft_tt_device.h` (device), `tt/ft_tt_noc.h` (data path),
`tt/ft_tt_launch.h` (load/run), `tt/ft_tt_queue.h` (resident dispatch),
`tt/ft_tt_op_*.h` (one per op family), `tt/ft_tt_fttw.h` (weights)
**Source:** `tt/*.c` (host, C99) + `tt/kernels/*.c` (Tensix, C99, compiled ahead of time by SFPI)
**Status:** 8 op families green on silicon (2026-08-12; the per-op figures below are **one core of 140**). Since then: multi-core matmul and Gated DeltaNet are built, and [Pantheon](https://git.teamide.dev/David/Pantheon.git) runs a 9B `qwen35` model end-to-end on the card (see [`tt/README.md`](../tt/README.md))

Folder guide, layout and the build/test commands are in
[`../tt/README.md`](../tt/README.md). This page is the deeper technical companion —
what the layers are, why the shapes are what they are, and where the numbers came from.

---

## Overview

`cuda/` and `metal/` accelerate an existing CPU op by handing a buffer to a vendor runtime.
There is no vendor runtime here. The backend is the whole stack:

| Layer | Files | What it owns |
|-------|-------|--------------|
| device | `ft_tt_device.c`, `ft_tt_tlb.c`, `ft_tt_arch_blackhole.c` | open `/dev/tenstorrent/N`, BAR mapping, TLB windows, arch tables |
| coordinates | `ft_tt_coord.c`, `ft_tt_arc_telemetry.c` | logical → NOC0 → translated, from ARC telemetry |
| data path | `ft_tt_noc.c`, `ft_tt_buffer.c` | L1/DRAM reads and writes through an aimed TLB window |
| load & run | `ft_tt_elf.c`, `ft_tt_kernel_blob.c`, `ft_tt_launch.c` | place kernel text, prime mailboxes, go, wait |
| dispatch | `ft_tt_queue.c`, `ft_tt_op_dispatch.c`, `ft_tt_group.c` | resident descriptor ring (fast) and one-shot launch (slow) |
| ops | `ft_tt_op_*.c` | per-family host drivers: CBs, runtime args, parameters |
| weights | `ft_tt_fttw.c`, `ft_tt_tilize.c`, `convert/` | GGUF → tiled on-card artifact, and uploading it |
| operations | `tools/ft_tt_core.c`, `ft_tt_watchdog.c` | per-core status, probe, recovery ladder |

The dependency boundary: **the SFPI compiler only** (plus `python3` for the
`tt/tools/emit_tensix_*.py` generators), to rebuild the device images (they are embedded, so a
Foundry build needs nothing). The backend was rewritten from public documentation and measured
hardware behaviour; no tt-metal, TT-NN, tt-umd or tt-llk source is compiled, linked or vendored,
and there is no C++ anywhere. The in-tree userspace driver needs no kernel module;
tt-kmd's ioctl ABI is an optional path. See `tt/README.md` "Dependencies".

---

## The execution model, and what it costs

### Reader / compute / writer

A Tensix has five RISC-V cores. One op is five programs: reader on BRISC (NoC 0), compute
compiled three times for TRISC0/1/2, writer on NCRISC (NoC 1). Splitting the data movement
across two NoCs is what lets the reader run ahead of the math.

The compute unit is a **32x32 tile** stored as four 16x16 faces. Everything downstream follows
from that: the weight converter emits faces in face order, block-float formats share one
exponent along a face row (so a transpose after packing is not free), and `DST` — the
MATH↔PACK register file — holds 16 tiles.

### Circular buffers are a compile-time contract

Data formats, tile geometry and math fidelity are baked into the kernel image via
`chlkc_descriptors.h`, generated from a `.cbtable`. The host writes a matching CB config into
L1 at launch. **Nothing checks that these agree at runtime** — a mismatch is a wrong answer,
not an error. That is why every format decision is recorded in the `.cbtable` rather than
being implied by a call site.

`math_fidelity 4` (HiFi4) is a recorded decision, not a default: a fidelity sweep moved
accuracy but moved wall time by 0.6%, because the op is data-movement bound at every setting.

### Resident kernels, not per-op dispatch

The slow path (`ft_tt_op_dispatch.h`) is one host round trip per op — describe CBs, plant args,
kick, wait. The fast path (`ft_tt_queue.h`) leaves a kernel resident on the core polling a
32-byte descriptor ring in its own L1; the host writes a descriptor and bumps a counter.

Three rules shape it, and each was paid for:

1. **Completion is produced by the work item.** The kernel writes `completed_count` and a
   per-run nonce *after* executing. `RUN_MSG_DONE` is 0, identical to a freshly booted mailbox,
   so "wait for DONE" would pass on a core that never ran.
2. **Retire by asking.** Every blocking wait polls `stop_request` in the same loop, and
   `FT_TT_QUEUE_OP_STOP` is an in-stream sentinel. `assert_reset` on a kernel blocked in a CB
   wait is what wedged cores.
3. **Residency is bounded.** `idle_spin_limit` makes the kernel exit on its own if the host
   dies.

Measured cost: **1.363 µs** per op host-side at batch 32, **0.391 µs** device-side op-to-op,
against upstream's `bh_p150` CI gate of **2.08 µs**. The direct consequence, stated in the
header: *do not fuse ops to save dispatch — there is no dispatch to save.*

---

## Performance, and what each number is

Every figure is **one core**, HiFi4, at this card's 800 MHz.

| Measurement | Value | Kind |
|---|---|---|
| matmul, `ffn_down`, prefetch depth 48 | 468.8 GFLOP/s = 28.6% of one Tensix | measured |
| matmul, `ffn_gate/up`, Mt=32 | 631.0 GFLOP/s = 38.5% | measured |
| matmul through the resident triad, real op | 595.2 GFLOP/s = 36.30% | measured |
| one Tensix at HiFi4, 800 MHz | 1.64 TFLOP/s | **derived** (2048 FLOP/cycle x 800 MHz) |
| AICLK under load | 800 MHz, every sample | measured (473 busy samples) |
| dispatch, host-side, batch 32 | 1.363 µs | measured |
| dispatch, device-side op-to-op | 0.391 µs | measured |
| attention, 8k KV | 0.840 ms at `block_tiles=16`/descriptor → 40.3 ms/token → 24.8 tok/s | measured / composed |
| attention L1 footprint | 413,696 B, flat in context | measured |
| top-k, 130,560 vocab | 2.590 ms/token vs 55.6 ms to host = 21x | measured |
| binary switch | 0.484 ms, ~93% of it the loader | measured |
| whole forward pass, one core | ~111.9 ms = 8.94 decode tok/s | **composed**, not run |
| core health | 140/140 | measured |

The distinction matters here and is enforced: 1.64 TFLOP/s is arithmetic from the vendor's
matrix-engine spec, not a benchmark, so every "% of one Tensix" inherits that. The 111.9 ms
forward pass is 3.745 ms/layer x 24 layers plus a 22.0 ms lm_head with the dispatch floor
subtracted, composed from per-op costs; it predates the end-to-end forward Pantheon now runs.

### Where the speed came from, and where it stopped

Phase 4 started at 81.3 GFLOP/s. The 5.9x to 468.8 was memory-identical and
accuracy-identical, and it was **not** bandwidth: deeper reader prefetch was worth 5.5x on its
own, and cutting transaction *count* another 1.28x. At 39.4 GB/s the card was never near a
bandwidth wall. The residual ~103 ns/tile-pair floor beneath that is unidentified.

Two corollaries that generalise:

- **Byte reduction does not reliably convert to speed.** Float16_b (2048 B/tile) → Bfp8_b
  (1088 B) bought 13%; Bfp8_b → Bfp4_b (576 B) bought nothing.
- **`DST_ACCUM_MODE=1` bought 13-35x accuracy for zero wall clock** (max-abs on `ffn_down`
  2.187 → 0.063 at 473 GFLOP/s either way). A precision knob you did not set is still a
  precision knob.

### Correctness gating

PCC alone is not a gate here — it caught **none** of four real kernel mutations on the reduce
family, and a softmax mutation came back green because `softmax(x-c) == softmax(x)`. Op checks
report **max-abs error alongside PCC**, top-k is gated on **exact index agreement**, and every
suite has a matching red table produced by mutating the kernel until it fails.

---

## Weights: the FTTW artifact

`convert/` streams a GGUF checkpoint into a tiled Blackhole artifact (magic `FTTW`). The unit
of work is a 32-row strip, so peak RSS is set by strip buffers, not model size: a full model
converts in 55.5 s at 210 MiB peak RSS on a 2-core x86-64 host.

The artifact is written in the card's own format, face-ordered and 64-byte aligned, so
`ft_tt_fttw_upload()` is a byte copy — the loader does not transpose and cannot usefully.
Two 2-D layouts exist as a per-tensor *format* field, not a hint:

- `FT_TT_LAYOUT_SOURCE` — GGUF order, row-major tile grid. What an embedding table needs,
  because it is indexed by row.
- `FT_TT_LAYOUT_MATMUL_NMAJOR` — transposed to K x N, tile grid stored column-major. Makes a
  reader's block of K-tiles one NoC transaction instead of `depth` strided ones.

The transpose has to happen in the converter because a Bfp8_b/Bfp4_b tile shares one exponent
across a face row; transposing after packing means unpacking and repacking. 1-D tensors stay
untiled fp32 — deliberate, on the hardware's broadcast rule.

Per-tensor precision comes from a ledger (`convert/ft_tt_precision_ledger.h`) and is gated by
`ft_tt_pcc_gate` against the fp32 source: BFP8 worst 0.999967 over 219 tensors, BFP4 worst
0.991455.

---

## Limits

- **Per-op figures are one core of 140.** Multi-core matmul exists (split groups; see `tt/README.md`). In the single-core op there is no reuse: every A tile is re-read Nt
  times and every B tile Mt times, so scaling needs an in0 broadcast and core groups, not a
  fan-out of the current op.
- **Aggregate DRAM read bandwidth is **>= 537 GB/s sustained across 14 concurrent cores and NOT saturated** — the ceiling is still unknown, and every core in that test read the same ~17 MB working set, which is the friendliest possible pattern.** Only achieved reader rates (28.4 →
  39.4 GB/s) and the finding that the binding constraint was transaction count.
- **Card→host DMA does not exist on Blackhole.** Host→card is 287 MB/s write-combined on this
  host's x1 link; reads back are 2.3-4.7 MB/s. Anything that ships tensors to the host for a
  decision is a design error — that is why top-k runs on the card.
- **MoE is not built** (the SSM family's Gated DeltaNet is), and no fusion was built: measured operand
  placement showed the round trip a fusion would remove is 346 ns/tile of *barrier*, not DRAM.
- **Not wired into `tensor/`.** There is no `tm_device` dispatch to Blackhole; `tt/` is driven
  directly. Unlike `metal/`, it has no integration points inside `tensor.c`.

---

## Operating rules

- **Never derive a NoC coordinate by probing.** A wrong endpoint does not fail a read, it hangs
  the chip and costs a physical power cycle. Coordinates come from ARC telemetry.
- **Never park a resident triad**, and never `assert_reset` a kernel blocked in a wait.
- Wedged cores are recoverable without rebooting the host: `ft_tt_core recover` climbs a
  documented ladder, rungs 3+ resetting the whole card under an advisory lock.
- Off-hardware test binaries are built *without* `-DFT_TT_PHASE1B_NOC_ENABLED` so they cannot
  reach the card even if they are wrong. On-card builds require `PHASE1B=1` explicitly.

---

## See Also

- [`../tt/README.md`](../tt/README.md) — the folder: layout, op set, build and test commands
- [metal.md](metal.md) · [bar-swap.md](bar-swap.md) — the other backends
