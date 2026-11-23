# Tenstorrent backend clean-room record (2026-10-08)

Provenance record for the rewrite that removed Tenstorrent-derived code (tt-metal, tt-llk,
tt-umd) from `tt/`. The per-phase records (specs, briefs, rules, rename logs, measurements,
gate scripts) and the phase commits are in the project's pre-release history; they are not part
of this public tree.

## Process

- **Roles.** A *spec writer* (the only party allowed to read the old TT-derived file) wrote a
  behaviour spec per file. A separate *implementer* deleted the old file and wrote a new one
  at the same path from the spec. Implementers did not commit; a reviewer reviewed and committed.
- **Hard rules for implementers:**
  - never open, grep, diff or `git show` the old file or its history, or any other checkout;
  - never read tt-metal, tt-llk, tt-umd, tt-kmd or sfpi sources;
  - comments in their own words, no upstream citations, no "transliterated from".
- **Inputs allowed:** the spec (or, in 6b, a comment-free API declaration file); the public
  `tt-isa-documentation`; Foundry's own clean headers and the callers' usage; values measured
  from Foundry's own built images; the test suites as a black-box oracle.
- **What specs could not contain:** no line or comment of the old code, no tt-metal / tt-llk
  file or function names beyond Foundry identifiers. Existing Foundry identifiers were kept so
  callers compile unchanged.
- **Gates.**
  - Header and kernel rewrites (Phase 4, Phase 7 link names): every one of the 351 program
    images rebuilds LOAD-identical to the reference (`tt/tools/rebuild_all_tensix_programs.sh`,
    `tt/tools/compare_tensix_load_images.py`).
  - Firmware replacements (Phases 2, 3, 5, 6) change the images by design, so they were gated on
    the card instead: every core probed, CLI outputs against recorded md5s, the serve reference
    set, and a red build that must fail.
  - Host-side rewrites (6b): the existing test suites, each with a red, plus a differential
    harness (old vs new on identical inputs).

## Phases

All on 2026-10-08.

| Phase | Replaced |
|---|---|
| 1 | Inventory: every TT source and header each program build compiled (shim toolchain) |
| 2 | Startup, libc substitutes, kernel entries → `fw/ft_crt0.S`, `fw/ft_libc.c`, `fw/ft_trisc_entry.c`, `fw/ft_dataflow_entry.c` |
| 3 | `main.ld` → `fw/ft_tensix_{firmware,kernel}.ld` from `tools/emit_tensix_linker_script.py` |
| 4.1 | Register and cfg-constant headers → `fw/ft_bh_regs.h` |
| 4.2 | Instruction encoders → `fw/ft_bh_isa.h` |
| 4.3 | `ft_tt_mailbox.h`, `ft_tt_memmap.h` |
| 4.4 | `kernels/ft_tensix_{circular_buffer,dataflow,kernel_args,noc,noc_multicast,transfer_guard}.h`, `kernels/resident_queue_protocol.h` |
| 4.5a | `kernels/ft_tensix_compute{,_broadcast,_cb,_math,_pack,_unpack}.h` |
| 4.5b | `kernels/ft_tensix_{compute_matmul,compute_reduce,compute_sfpu,math_eltwise_sfpu,sfpu,sfpu_math}.h` |
| 4.6 | CB descriptor generator → `tools/emit_tensix_cb_formats.py` |
| 4.7 | `kernels/resident_{attention,eltwise,rope,gdn,gdnc}_compute.c`, `mop_expander_probe_compute.c`, `resident_matmul_dst_sync.h`, `resident_vector_writer_{flushed,blocked}.c` |
| 5 | `trisc.cc`, `ncrisc.cc` → `fw/ft_trisc_fw.c`, `fw/ft_ncrisc_fw.c` |
| 6 | `brisc.cc`, `noc.c` → `fw/ft_brisc_fw.c`, `fw/ft_noc.{c,h}` |
| 6b | Host: `ft_tt_tilize`, `ft_tt_elf`, `convert/ft_bfp`, `ft_tt_arch*`, `ft_tt_coord`, vendored tt-umd lib → `ft_tt_kmd*`, `ft_tt_arc_telemetry`, runtime archive |
| 7 | Regen without the staged tree; firmware link names; k-quant decoders |

Follow-up: `tt/ft_tt_kernels.h` (four small test programs) was not part of the rebuild and
still held images built in 2026-08 with tt-metal firmware. It was regenerated on 2026-10-27
from the current sources with `tt/tools/regen_tensix_kernel_header.sh`.

Not covered: `tt/tests/test_dma_vs_pio.c` (a measurement test transliterated from tt-umd);
it keeps its Apache-2.0 attribution in `NOTICE`.
