# tt/

Tenstorrent Blackhole GPU backend. Host-side C plus hand-written Tensix kernels, driven
straight through the kernel driver.

**Card:** Tenstorrent Blackhole **p150a**, 140 Tensix cores, firmware pinned at 19.2.0.
**Status:** 8 op families green on silicon (2026-08-12), all measured on **one core of 140**.
Device kernels are **C-only** as of 2026-08-13 — Phase 6.6 converted all 36, and the last C++
in the tree (a compute arm inside `resident_queue_protocol.h`) went with them.
**Read first:** "Build and test" below — on-card runs can wedge cores.
Comments citing `DEV_NOTES` or `docs/tt_*.md` refer to pre-release notes not in this repo
(see `CONTRIBUTING.md`, Code Style).

---

## Dependencies

**SFPI compiler only** (`/opt/tenstorrent/sfpi`: GCC 15.1 for `riscv-tt-elf`, a tool like GCC),
plus `python3` for the `tools/emit_tensix_*.py` generators, and only to rebuild Tensix programs. The built images are embedded in `ft_tt_programs.h`, so a
Foundry build needs no toolchain — the same arrangement as `cuda/`, which embeds PTX.

- **Provenance.** The Tenstorrent backend was rewritten from public documentation and measured
  hardware behaviour. No tt-metal, TT-NN, tt-umd or tt-llk library is linked or vendored, and
  the current build compiles no Tenstorrent source. The one exception, listed in `../NOTICE`,
  is the measurement test `tests/test_dma_vs_pio.c` (tt-umd, Apache-2.0). The clean-room record
  is `../docs/cleanroom/tt/README.md`. Firmware (`fw/`), startup, linker scripts, hardware headers, kernel library and
  host driver are Foundry's own, rewritten in 2026-10 from the public `tt-isa-documentation`
  and our own measurements; every image is linked `-nostdlib`.
- **No C++ anywhere, host or device.** Host and kernels are C99.
- **Driver:** the in-tree userspace driver (`ft_tt_udrv_*`, PCI sysfs) drives the card with
  no kernel module. `ft_tt_kmd.c` can use tt-kmd's ioctl ABI instead (numbers only,
  `ft_tt_kmd_abi.h`); tt-kmd is optional, not a dependency.

## Layout

| Path | What |
|------|------|
| `ft_tt_*.{c,h}` | Host backend: device open, TLBs, NoC, launch, queue, ops, telemetry |
| `ft_tt_op_*.{c,h}` | One host driver per op family (matmul, vector, memory, rope, sample, attention, rms_norm, dispatch) |
| `kernels/*.c` | Tensix device kernels, named `<op>_{reader,compute,writer}.c` |
| `kernels/ft_tensix_*.h` | The C99 device library the kernels are written against — CB primitives, NoC, the unpack/math/pack sequences, the SFPU |
| `kernels/*_protocol.h` | Host↔device wire contracts — the two sides must agree word for word |
| `kernels/cb_tables/*.cbtable` | Circular-buffer descriptor tables (compile-time format contract) |
| `convert/` | FTTW weight converter: GGUF in, tiled on-card artifact out |
| `tests/` | 48 Makefiles (plus `convert/tests/Makefile`); off-hardware gates and on-card acceptance are separate build trees |
| `tools/` | `ft_tt_core` (per-core status / probe / recovery), the kernel + CB emitters, and the gates below |
| `build_tensix_*.sh` | The kernel build scripts (SFPI) |

`.cbtable` files are the compile-time half of the CB contract — data format, tile geometry,
math fidelity — turned into `ft_tensix_cb_formats.h` by `tools/emit_tensix_cb_formats.py`.
**There is no runtime check:** what the host writes into the L1 CB config must match what was
baked in, or the kernel reads the right bytes as the wrong format.

### The gates a kernel change must pass

Instruction equality is the main one, but it is **not sufficient** — two of the three findings
that nearly shipped in Phase 6.6 were invisible to it.

| gate | what it catches | how |
|---|---|---|
| instruction-stream diff | a different Tensix instruction, store or immediate | `tools/diff_tensix_kernel_instructions.sh` — identical stream, plus a **named** cause for any RISC-V difference |
| **text size vs the slab** | **a kernel that will not load at all** | `size` against `FT_TT_OP_TEXT_SLAB_BYTES_<proc>` (11-16 KiB per processor, `ft_tt_op_l1_layout.h`). The first C build of `resident_attention_compute` was **836 B over** — and the instruction diff was perfectly happy, because C inlined what C++ outlined |

Two more, for the artifacts rather than the code: `make -f tools/Makefile regen-kernels` rebuilds
the embedded ELFs in `ft_tt_kernels.h` (and **refuses** if the built firmware differs from
`FT_TT_FIRMWARE_DIR` — kernels link at their firmware's `__fw_export_text_end`, so the two must
move together), and `make -f tests/Makefile.vector check-programs` reads the `.src` sidecars to
refuse any on-card program directory built from a source that has since changed or vanished.
Both exist because a stale artifact shipped twice in one day.

## Execution model — it is not `cuda/`

Three differences that change how you write an op:

1. **Cores are programmed independently.** There is no grid launch. Each Tensix is loaded,
   started, and waited on by name; `ft_tt_group.c` batches that, it does not hide it.
2. **A `__global__` becomes a reader/compute/writer triad.** Five RISC-Vs per core: reader on
   BRISC (NoC 0), compute on TRISC0/1/2, writer on NCRISC (NoC 1) — the two data-movement
   kernels sit on different NoCs so they do not contend. The unit of data is a **32x32 tile**
   (four 16x16 faces), not a scalar or a warp lane.
3. **Ops are resident, not dispatched.** `ft_tt_queue.h` is a descriptor ring in the core's own
   L1; a resident kernel polls it and executes 32-byte descriptors. The host enqueues work and
   bumps a counter — no per-op launch. The consequence is stated in the header: *do not fuse
   ops to save dispatch, there is no dispatch to save.*

Retirement is by asking, never by reset: every blocking wait polls `stop_request` in the same
loop, and `FT_TT_QUEUE_OP_STOP` is the in-stream sentinel. `assert_reset` on a blocked resident
kernel is how cores were wedged.

## What works, with numbers

All figures are **one core**, at HiFi4 and this card's 800 MHz (measured busy).

| Op family | State |
|---|---|
| matmul (bf16 / Bfp8_b / Bfp4_b) | green, the load-bearing op |
| elementwise (SiLU, sigmoid, add, mul, sub, SiLU-mul) | one kernel, runtime-selected |
| reduce (softmax, RMS norm, layer norm, router normalise) | one residency, four ops |
| embedding, KV-cache append | green |
| RoPE (NeoX), flash-decode attention | green |
| on-card top-k + sampling | green, gated on exact index agreement |
| Gated DeltaNet (Qwen3.5 linear layers: conv1d+SiLU, delta rule) | built (`ft_tt_op_gdn`, `ft_tt_op_gdnc`); runs on the card through Pantheon's Ornith forward |
| MoE, fusions | **not built** |

- **matmul** 468.8 GFLOP/s (`ffn_down`, prefetch depth 48) to **631.0 GFLOP/s** (`ffn_gate/up`,
  Mt=32) = **28.6% to 38.5% of one Tensix**. The 1.64 TFLOP/s denominator is *derived*
  (2048 FLOP/cycle at HiFi4 x 800 MHz), not measured; the 800 MHz busy clock is measured.
  Through the resident triad on a real op: 595.2 GFLOP/s (36.30%).
- **dispatch** 1.363 µs per op host-side at batch 32, **0.391 µs** device-side op-to-op.
  Upstream's `bh_p150` CI gates op-to-op at 2.08 µs.
- **attention** (flash-decode) 0.840 ms at `block_tiles=16`/descriptor at 8k KV = 40.3 ms/token = **24.8 tok/s**.
  L1 is flat in context at 413,696 B. Correctness verified kv=1..2048, PCC 0.99998-1.00000 —
  the timing point is beyond the verified range.
- **top-k** 2.590 ms/token over a 130,560 vocabulary, against 55.6 ms to ship that row to the
  host: 21x, and context-independent.
- **binary switch** 0.484 ms, of which ~93% is the loader, not the x1 link.
- **cores** `ft_tt_core probe all` is 140/140.

## Honest limits

- **The per-op figures above are one core of 140.** Multi-core matmul exists — split groups
  (`ft_tt_group.c`) and a cooperative matmul that multicasts the shared A block
  (`ft_tt_op_matmul_cooperative.c`, `ft_tt_op_matmul_resident_multicast.c`). Scaling is
  far from linear: an N=8 split of `ffn_gate/up` measured 2.07x over one core (2026-08-13,
  operands in two DRAM banks).
- **One forward pass on one core is ~111.9 ms = 8.94 decode tok/s.** That is *composed* from
  measured per-op costs (3.745 ms/layer x 24 layers + 22.0 ms lm_head, dispatch floor
  subtracted). Separately, Pantheon (a separate engine built on this backend) runs a 9B
  `qwen35` model (Ornith) end-to-end on the card, with greedy output byte-identical to its
  CPU oracle (Pantheon `39d0512`, 2026-08-18).
- **The card's aggregate DRAM read bandwidth is >= 537 GB/s sustained across 14 concurrent cores and NOT saturated** — the ceiling is still unknown, and every core in that test read the same ~17 MB working set, which is the friendliest possible pattern. The reader reaches
  39.4 GB/s and the measured floor was transaction *count*, not bandwidth; the residual
  ~103 ns/tile-pair floor under that is unidentified.
- Card→host DMA does not exist on Blackhole; host→card is 287 MB/s (write-combined) on this
  host's x1 link.

## Build and test

Kernel builds need **SFPI**, so kernel compilation runs on a host with SFPI installed, and every
on-card suite runs on the host with the card. Off-hardware suites build anywhere.

`make USE_TT=1 tt` from Foundry's root builds the host backend objects (no SFPI
needed — see above). Suites still have no root-Makefile target; each has its own
Makefile, run from `tt/tests/`:

```bash
# off-hardware (no card, safe while someone else owns the silicon)
make -f Makefile.coord check          # coordinates, arch tables, harvesting
make -f Makefile.noc   check
make -f Makefile.queue check          # the resident-queue protocol, host side
make -f Makefile.vector all           # -> build-vector/test_vector_ops

# on-card (the host with the card). PHASE1B=1 opens the NoC fence and is mandatory.
make -f Makefile.ops    PHASE1B=1 all
make -f Makefile.phase5 PHASE1B=1 all
make -f Makefile.phase6 PHASE1B=1 all
make -f Makefile.vector PHASE1B=1 card

# the kernel gates (no card needed; needs SFPI)
make -f tools/Makefile regen-kernels              # rebuild the embedded ELFs in ft_tt_kernels.h
make -f tests/Makefile.vector check-programs      # refuse stale on-card program directories

# per-core status / recovery tool
cd ../tools && make PHASE1B=1 && ./build/ft_tt_core status all

# weight converter (no card, no tt-metal)
cd ../convert/tests && make check
```

`make red` on any of these means clean-then-build-then-run. It exists because Apple ships GNU
make 3.81, whose one-second mtime granularity silently skips a rebuild of a source edited in
the same second as its object — a mutation loop on incremental make can be testing the
previous binary.

On-card runs are detached and polled, never held over an SSH session, and never launched
against cores another job owns.

## See also

- [`../docs/tenstorrent.md`](../docs/tenstorrent.md) — the backend page
