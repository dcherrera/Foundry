/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_attn_vstage.cu
 * @brief V-path variants of the register-K prefill flash-attention kernel for
 *   the K80/GK210 PV-loop microbenchmark. Includes attention.cu VERBATIM so the
 *   shipped fc_attn_flash_prefill_q4_kr is a byte-identical control symbol in
 *   the same module.
 *
 * WHAT WAS COUNTED FIRST, and why these forms exist (SASS of the shipped
 * kernel, cuobjdump 11.4, sm_37, foundry 0c1ed6e):
 *
 *   The register-K change deleted the shared K tile and never examined V. At the
 *   shipped shape (head_dim 256, kv_groups 8, TILE_KEYS 16, win 128, q4 body)
 *   the PV loop is 97 EXECUTED instructions per key, x16 keys per tile, x8
 *   warps = 62.5 % of the kernel's executed warp-issue slots per tile. Of those
 *   97:
 *     29  64-bit global ADDRESS arithmetic for one V byte triple, of which 15
 *         are loop-INVARIANT (they depend only on tid and kvh) and are
 *         recomputed on all 16 iterations
 *     12  the three LD.E.U8 + fp16 assemble + dequant of ONE element
 *     46  the actual PV work (8 predicated LDS of p + 8 FFMA + predicates)
 *     ~10 loop control and path select
 *   Useful fp work is the 8 FFMA: 8.2 %.
 *
 *   Loads in flight (plain global LD; this kernel has ZERO LDG, so TEXDEPBAR
 *   does not appear and the measure is "LDs issued before the first consumer"):
 *   V q4 path 3, V fp16 paths 1, K q4 path 3. So the register-K win did NOT
 *   come from memory-level parallelism — K and V sit at the same 3 — it came
 *   from deleting a shared round trip (1.194x) and buying residency (1.269x).
 *
 *   NEITHER of those two mechanisms is available on V:
 *     - V has no shared round trip to delete. It is already read straight from
 *       global into a register.
 *     - There is no residency to buy. 72 registers x 256 threads gives
 *       floor(131072/(72*256)) = 7 blocks/SMX and 80 registers gives 6, so the
 *       kernel sits EXACTLY on the cliff with ZERO register headroom. Any form
 *       that costs one register bucket costs a resident block (0.857x) and must
 *       earn 1.17x before it breaks even.
 *
 *   What IS available is the mirror image of the K finding. K had a shared tile
 *   with 1:1 reuse (one writer, the same thread the only reader) and removing it
 *   paid. V has 32:1 reuse and NO tile: a V row is 144 B, and the block issues
 *   256 threads x 3 = 768 byte-load instructions to read it, because all 32
 *   threads of a Q4_0 block load the same 2-byte fp16 scale and each nibble byte
 *   serves 2 threads. The staging pattern (none) does not match the consuming
 *   pattern (block-wide). And the shared memory to fix it is already free:
 *   fc_attn_flash_kreg_shared_bytes' own comment records that 7 is the REGISTER
 *   cap, so 9 312 -> 11 616 B still leaves 7 blocks/SMX.
 *
 * THE ARMS, and what each isolates:
 *
 *   COPY  this file's copy of the kernel at the shipped PV loop, verbatim. Its
 *         SASS must be identical to fc_attn_flash_prefill_q4_kr's and its ratio
 *         must be 1.00x. If it is not, the harness is not compiling the form the
 *         shipped source would (the failure that mislabelled an arm in the
 *         matvec gate) and every other number here is void.
 *   VIND  the 15 loop-invariant index terms hoisted to the tile preheader and
 *         the q4 byte pointer strength-reduced to an induction variable. V stays
 *         in GLOBAL. Loads in flight UNCHANGED at 3. This is the pure
 *         instruction-reduction arm.
 *   VSH   VIND's exact instruction structure with one thing changed: the tile's
 *         tk contiguous V rows are staged into shared as RAW BYTES and the
 *         dequant reads shared instead of global. Staging is placed in the
 *         softmax shadow (the softmax is `if (tid < G)`, so 248 of 256 threads
 *         are idle there) and consumes the __syncthreads() that already
 *         separates the softmax from the PV loop, so it adds NO barrier.
 *
 *   VIND vs COPY is therefore a clean single-variable dial on instruction
 *   reduction, and VSH vs VIND a clean single-variable dial on the global ->
 *   shared move, because their PV loops are structurally identical apart from
 *   the address space of three byte loads.
 *
 * WHAT WAS COUNTED AFTER BUILDING THEM (SASS, compile-only, no die):
 *
 *   COPY is SASS-IDENTICAL to fc_attn_flash_prefill_q4_kr, instruction for
 *   instruction, 2 310 = 2 310 after normalising branch targets. So the template
 *   above IS the shipped kernel and every _Pragma landed exactly as the shipped
 *   #pragma. (nvcc emits one "this pragma must immediately precede a statement"
 *   warning attributed to the LAST expansion; this identity is the proof it is
 *   inert.)
 *
 *   PV body per key, q4 path, j != 0:  kr 97 -> VIND 74 (-23.7 %) -> VSH 71
 *   (-26.8 %). VSH also drops the global loads on that path to ZERO: the fp16
 *   scale becomes ONE LDS.U16 instead of two LD.E.U8 plus a PRMT, because in
 *   shared ptxas CAN prove the 2-byte alignment it cannot prove for
 *   v_head + vp*row_bytes with a runtime row_bytes. The whole 64-bit address
 *   chain collapses to IMAD/IMUL32I/IMAD (18*(8j + tid/32) + base), 3
 *   instructions of 32-bit shared addressing.
 *
 *   Whole-kernel counted multiplier, executed warp-issue slots per tile per
 *   block (K+QK+reduce 897 x8 warps, softmax 277 x1 warp, PV x16 x8):
 *   kr 19 869, VIND 16 925 (m = 1.174x), VSH 16 749 (m = 1.186x, including its
 *   26-instruction staging x8).
 *
 *   RESIDUAL, counted and NOT fixed: ptxas schedules the staging copy as
 *   LD -> STS -> LD -> STS -> LD -> STS, i.e. ONE global load in flight, not
 *   three, because holding w0/w1/w2 live would cost registers at the 72 cliff.
 *   It still cuts exposed global V latency per tile from ~16 batches to 3 serial
 *   loads (~5x), and it sits in the softmax shadow. Named and priced, not
 *   attempted: any fix for it must not cost a register.
 *
 * BIT-EXACTNESS. Every arm except the poison is bit-exact BY CONSTRUCTION:
 *   - the address set is unchanged. VIND's induction pointer reaches
 *     v_head + (j0+j)*row_bytes + (tid/32)*18, which is exactly what
 *     fc_kvq4_dequant_elem(v_head + vp*row_bytes, tid) computes. VSH stages
 *     those same bytes and reads them back at the same offsets.
 *   - the dequant is fc_kvq4_dequant_elem's own expression, in its own order:
 *     raw = bp[0] | (bp[1] << 8); d = __half2float(raw);
 *     q = (bp[2 + byte] >> shift) & 0xF; return (float)(q - 8) * d.
 *     One FMUL, same operands, same rounding.
 *   - the g loop is copied character for character, INCLUDING the rescale fused
 *     at j == 0 that ptxas contracts to acc = fma(acc, corr, p0*v0). Hoisting it
 *     is algebraically identical and moves 148 099 of 524 288 outputs by 1 ulp
 *     (measured, prefill_attention_register_ktile.md). It is not touched here.
 *   - no float changes residence. VSH moves quantised BYTES into shared;
 *     integers have no rounding and no contraction, so the __fmul_rn/__fadd_rn
 *     contraction barriers the SSM rewrite needed do not apply.
 *   - the accumulation order over j and over g is unchanged in all arms.
 *
 * THE PREPARED GATE — PRE-REGISTERED, AND NOT RUN. One die of one board only,
 * sibling verified IDLE in the trace, pinned with -ac 2505,666 (NEVER -rac,
 * which CAPS a K80 at 562), restored under trap ... EXIT. ~40 MB of VRAM.
 *
 *   cd foundry/cuda_kernels/tests
 *   /usr/local/cuda-11.4/bin/nvcc -arch=sm_37 -ccbin gcc-11 \
 *       -Wno-deprecated-gpu-targets -Wno-deprecated-declarations \
 *       -isystem ../nvcc_shim -I.. -O3 -Xptxas -v \
 *       -o /tmp/bench_attn_vstage bench_attn_vstage.cu -lm
 *   sudo nvidia-smi -i <die> -ac 2505,666
 *   nvidia-smi -i <die> --query-gpu=clocks.applications.graphics --format=csv
 *   for i in $(seq 1 10); do CUDA_VISIBLE_DEVICES=<die> \
 *       /tmp/bench_attn_vstage --self; done          # the NOISE FLOOR, first
 *   CUDA_VISIBLE_DEVICES=<die> /tmp/bench_attn_vstage --poison
 *   CUDA_VISIBLE_DEVICES=<die> /tmp/bench_attn_vstage --form 2   # COPY  control
 *   CUDA_VISIBLE_DEVICES=<die> /tmp/bench_attn_vstage --form 3   # VIND  dial 1
 *   CUDA_VISIBLE_DEVICES=<die> /tmp/bench_attn_vstage --form 4   # VSH   the lever
 *   CUDA_VISIBLE_DEVICES=<die> /tmp/bench_attn_vstage --sweep
 *   CUDA_VISIBLE_DEVICES=<die> /tmp/bench_attn_vstage --sweep --depth 8192
 *   CUDA_VISIBLE_DEVICES=<die> /tmp/bench_attn_vstage --sweep --depth 32000
 *   sudo nvidia-smi -i <die> -ac 2505,875                        # RESTORE
 *
 * Every arm is timed A...B...A inside the process and the ratio uses the mean of
 * the two A passes, so the -0.49 % second-arm warming bias MEASURED on the
 * matvec gate's control cancels by design instead of being corrected afterwards.
 * The A-to-A drift is printed per arm.
 *
 * DECISION RULES, written before any arm runs:
 *
 *   --self x10      MUST be EXACT on all 262 144 outputs, ratio spread <= 1 %.
 *                   Expect ~0.03 % (lever B's gate measured 0.030 % on this box
 *                   at this pin). FAIL => THE GATE IS BROKEN and every other
 *                   verdict here is void.
 *   --poison        MUST DIFFER, and on a LARGE fraction. EXACT => the gate is
 *                   broken: the comparison is not comparing bits and the
 *                   by-construction claim is unproven. A handful (<< 1 %) is
 *                   WORSE than a failure — it means the poison is not reaching
 *                   most keys, i.e. the arm ran the wrong kernel. (Unlike the
 *                   matvec/MoE poisons, no host replica of the expected fraction
 *                   exists for this kernel: a 14 080-key online softmax
 *                   renormalises by l, so the fraction is NOT predicted here and
 *                   must not be quoted as if it were. The criterion is "large",
 *                   with the direction pre-registered, not a number.)
 *   resources line  MUST read REG 72 and blocks/SMX 7 on ALL FIVE arms. Anything
 *                   else => THE GATE IS BROKEN: the binary compiled a different
 *                   form than the source states. Registers are this lever's
 *                   whole gate — 72 -> 80 is 7 -> 6 blocks = 0.857x, which eats
 *                   the entire counted 1.186x.
 *   --form 2 COPY   MUST be bit-EXACT and ratio 1.000x +/- the --self spread. It
 *                   is SASS-identical to the shipped kernel, so a ratio outside
 *                   the floor means the TIMING is not measuring the launch, not
 *                   that the copy is slower.
 *   --form 3 VIND   bit-EXACT required. Counted 1.174x. Predicted 1.09-1.17x.
 *                   This is the instruction-reduction dial: at or below the
 *                   --self floor it says instruction count does NOT convert on
 *                   this kernel at 7 blocks/SMX, which is a real closure and
 *                   would retire the whole PV-loop surface.
 *   --form 4 VSH    bit-EXACT required. Counted 1.186x. Predicted 1.10-1.25x —
 *                   the top of the band is above the counted cut because the
 *                   3 -> 0 global loads on the modal V path is a latency
 *                   conversion the instruction count cannot price. NOT EXACT =>
 *                   STOP: the claim is bit-exactness by construction, so a
 *                   DIFFER is a real bug, not a tolerance question.
 *   VSH minus VIND  the single-variable dial on global -> shared. No gap => the
 *                   V loads were already hidden behind 7 resident blocks and the
 *                   whole win is instruction count; ship VIND instead, which
 *                   needs no shared and no host change.
 *   --depth sweep   the ratio MUST be flat in depth. attn_flash's SHARE grows
 *                   (17.87 -> 46.08 %) but the per-tile structure does not, so a
 *                   kernel ratio that moves with depth falsifies the model
 *                   regardless of how fast the arm is.
 *
 * WHAT MEANS THE GATE IS BROKEN RATHER THAN THE LEVER: --self DIFFER; --poison
 * EXACT or << 1 %; the resources line disagreeing with the counted table; COPY
 * outside the --self floor; or a --sweep ratio disagreeing with the same arm's
 * own --form N run by more than the --self spread.
 *
 * IF IT PASSES — the in-model arm, NOT run here. A ship is a NEW kernel symbol
 * (fc_attn_flash_prefill_q4_kvs) plus a host guard and fallback, exactly how
 * _kr itself landed — NEVER a flag on the existing kernel, because VSH's shared
 * request grows and a host built one way against a prebuilt PTX built the other
 * under-allocates shared. Then: full-logit FNV-1a cross-BUILD
 * (PN_PREFILL_LOGITS_HASH=1) at >= 512 positions with a poisoned sibling binary
 * to prove the digest can fail; and note that a 128-position gate on this kernel
 * is VACUOUS because QWEN35_KV_Q4_WIN defaults to 128, so every key comes from
 * the fp16 shadow and the q4 path never runs. The bucket that must move is
 * attn_flash and nothing else; proj, out, attn_mv, recur_seq, moe gateup, moe
 * down and router_shexp are the common-mode controls.
 *
 * @status Microbenchmark only. NOTHING here is reachable from Pantheon or from
 *   fc_cuda.c: tests/ is not in cuda_kernels/Makefile's KERNELS list, so no
 *   symbol in this file can enter cuda/fc_cuda_ptx.h. No shipped kernel, no
 *   launcher and no default is modified. Compiled, NOT RUN.
 * @issues
 *   - Registers are the whole gate and they are the OFFLINE ptxas allocation.
 *     Foundry ships PTX and loads it with cuModuleLoadData with no JIT options,
 *     so the DRIVER's ptxas allocates on the shipping path. cuFuncGetAttribute
 *     on the loaded function is the only authority for a ship decision; this
 *     harness reports cudaFuncGetAttributes, which is the offline number.
 *   - VSH assumes tk contiguous V rows exist at v_head + j0*row_bytes. They do
 *     (j0 + tk <= seq_len <= max_seq), and rows shadowed by the fp16 sink or
 *     window are staged but never read, because the per-j path select is
 *     unchanged.
 *   - VSH's shared request grows by FC_ATTN_TILE_KEYS * row_bytes (2 304 B at
 *     the shipped shape). A ship would have to grow
 *     fc_attn_flash_kreg_shared_bytes by the same amount under a NEW kernel
 *     symbol, never by flagging the existing one, or a host built one way
 *     against a PTX built the other under-allocates shared.
 * @todo None
 */

#include "../attention.cu"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* saber-orch / nex-n2-mini's full-attention shape at the shipped prefill chunk.
 * The host guard on fc_attn_flash_prefill_q4_kr admits head_dim <= 256 AND
 * head_dim >= 256, so head_dim is provably 256 on every real launch. */
#define XV_HEAD_DIM   256
#define XV_KV_GROUPS  8
#define XV_N_HEAD_KV  2
#define XV_NPOS       128                  /* QWEN35_BS_PREFILL_CHUNK default */
#define XV_DEPTH      14080                /* the 14 080-token attribution point */
#define XV_SINK       4                    /* QWEN35_KV_Q4_SINK default */
#define XV_WIN        128                  /* QWEN35_KV_Q4_WIN default */
#define XV_ROW_BYTES  ((XV_HEAD_DIM / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES)   /* 144 */

/* Bytes of shared the staged V tile needs, and its word count for the copy.
 * 16 * 144 = 2 304 B = 576 words; 576 = 2*256 + 64, so ceil(576/256) = 3
 * compile-time unrolled rounds, which is what puts three loads in flight. */
#define XV_VTILE_BYTES (FC_ATTN_TILE_KEYS * XV_ROW_BYTES)
#define XV_VTILE_WORDS (XV_VTILE_BYTES / 4)
#define XV_VTILE_ROUNDS ((XV_VTILE_WORDS + XV_HEAD_DIM - 1) / XV_HEAD_DIM)

/* The staging stride is the compile-time XV_HEAD_DIM rather than blockDim.x,
 * because a compile-time stride is what lets the copy unroll and put all its
 * loads in flight. Legal only because a block has head_dim threads AND the host
 * guard on this kernel admits head_dim <= 256 and head_dim >= 256 — so
 * head_dim == 256 exactly on every launch. Assert it rather than assume it. */
#if XV_HEAD_DIM != FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES
#error "XV_HEAD_DIM must equal FC_ATTN_TILE_KEYS * FC_ATTN_DOT_LANES (256)"
#endif
#if XV_ROW_BYTES % 4 != 0
#error "XV_ROW_BYTES must be a multiple of 4 for the word-wise V tile copy"
#endif

/* 1 + 2^-23, ~1.2e-07 relative. The same poison whose host replica predicted
 * the matvec gate's GPU differ-count to 0.4 % (tests/calib_nx_stage_poison.c).
 * Applied to vv so it is INSIDE tolerance and defeats any tolerance-based
 * check, which is the point of a negative control. */
#define XV_POISON 1.00000011920928955078125f

/* ------------------------------------------------------------ PV forms ---- */
/*
 * Each form is the body between the softmax barrier and the tile barrier. They
 * are macros rather than functions because acc[] is a per-thread register array
 * and the g loop must stay in the caller's unrolled scope.
 *
 * XV_PV_PRE  runs once per tile, in the softmax shadow (before the barrier that
 *            already separates the softmax from the PV loop).
 * XV_PV_BODY runs the j loop.
 */

/* The g loop, character for character from attention.cu, in all four forms.
 * The rescale stays fused at j == 0. VV is the dequantised V element. */
#define XV_PV_GLOOP(VV)                                                        \
            const float *pj_s = p_s + (size_t)j * G;                           \
            _Pragma("unroll")                                                  \
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {                     \
                if (g < G) {                                                   \
                    if (j == 0)                                                \
                        acc[g] *= corr_s[g];                                   \
                    acc[g] += pj_s[g] * (VV);                                  \
                }                                                              \
            }

/* --- form FLAT: the shipped loop, verbatim ------------------------------- */
#define XV_PRE_FLAT   /* nothing */
#define XV_BODY_FLAT                                                           \
        for (int j = 0; j < tk; j++) {                                         \
            int vp = j0 + j;                                                   \
            float vv;                                                          \
            if (vp < sink)                                                     \
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);   \
            else if (win > 0 && vp > cur_pos - win && vp > win_live_lo)        \
                vv = __half2float(                                             \
                    v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]); \
            else                                                               \
                vv = fc_kvq4_dequant_elem(v_head + (size_t)vp * row_bytes,     \
                                          tid);                                \
            XV_PV_GLOOP(vv)                                                    \
        }

/* --- form POISON: FLAT plus exactly one perturbation on vv ---------------- */
#define XV_PRE_POISON /* nothing */
#define XV_BODY_POISON                                                         \
        for (int j = 0; j < tk; j++) {                                         \
            int vp = j0 + j;                                                   \
            float vv;                                                          \
            if (vp < sink)                                                     \
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);   \
            else if (win > 0 && vp > cur_pos - win && vp > win_live_lo)        \
                vv = __half2float(                                             \
                    v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]); \
            else                                                               \
                vv = fc_kvq4_dequant_elem(v_head + (size_t)vp * row_bytes,     \
                                          tid);                                \
            vv = __fmul_rn(vv, XV_POISON);                                     \
            XV_PV_GLOOP(vv)                                                    \
        }

/* --- form VIND: invariants hoisted, q4 byte pointer strength-reduced ------
 * vbyte/vshift/vq0 depend only on tid; vq advances by row_bytes per j, which is
 * exactly fc_kvq4_dequant_elem's bp for vp = j0 + j. The fp16 arms are left
 * alone: they are the rare path at depth (only the last ceil(win/16) = 8 tiles
 * of ~880 read the window, and only tile 0 reads the sink), and keeping their
 * addressing inline is what stops two more 64-bit pointers going live across
 * the loop and costing the resident block. */
#define XV_PRE_VIND                                                            \
        const int vlocal = tid & (FC_KVQ4_BLK - 1);                            \
        const int vbyte = 2 + (vlocal < 16 ? vlocal : vlocal - 16);            \
        const int vshift = vlocal < 16 ? 0 : 4;                                \
        const unsigned char *vq = v_head + (size_t)j0 * row_bytes +            \
                                  (size_t)(tid / FC_KVQ4_BLK) *               \
                                      FC_KVQ4_BLK_BYTES;
#define XV_BODY_VIND                                                           \
        for (int j = 0; j < tk; j++, vq += row_bytes) {                        \
            int vp = j0 + j;                                                   \
            float vv;                                                          \
            if (vp < sink)                                                     \
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);   \
            else if (win > 0 && vp > cur_pos - win && vp > win_live_lo)        \
                vv = __half2float(                                             \
                    v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]); \
            else {                                                             \
                unsigned short raw =                                           \
                    (unsigned short)(vq[0] | (vq[1] << 8));                    \
                float d = __half2float(                                        \
                    *reinterpret_cast<const __half *>(&raw));                  \
                int qq = (vq[vbyte] >> vshift) & 0xF;                          \
                vv = (float)(qq - 8) * d;                                      \
            }                                                                  \
            XV_PV_GLOOP(vv)                                                    \
        }

/* --- form VSH: VIND, with the tile's V bytes staged in shared -------------
 * The three staging rounds are written OUT, not looped under a #pragma unroll.
 * Two reasons, and the second is the load-bearing one:
 *   1. XV_VTILE_ROUNDS is a compile-time 3, so a loop buys nothing.
 *   2. A _Pragma inside a macro that is itself passed as a macro ARGUMENT is
 *      exactly the construction that silently dropped an unroll in the matvec
 *      gate and mislabelled an arm. nvcc caught it here too — "this pragma must
 *      immediately precede a statement" — and a dropped unroll would leave this
 *      copy runtime-bounded and rolled at 1-2 loads in flight, i.e. it would
 *      compile the pathology instead of the fix. Straight-line code cannot fail
 *      that way.
 * The three loads are mutually independent and precede their first consumer, so
 * they are 3 in flight, and the copy is placed in the softmax shadow (the
 * softmax is `if (tid < G)`, leaving 248 of 256 threads idle) and covered by the
 * __syncthreads() that already separates the softmax from the PV loop. */
#define XV_PRE_VSH                                                             \
        const int vlocal = tid & (FC_KVQ4_BLK - 1);                            \
        const int vbyte = 2 + (vlocal < 16 ? vlocal : vlocal - 16);            \
        const int vshift = vlocal < 16 ? 0 : 4;                                \
        const int vqoff = (int)(tid / FC_KVQ4_BLK) * FC_KVQ4_BLK_BYTES;        \
        {                                                                      \
            const unsigned int *vsrc = (const unsigned int *)(                 \
                v_head + (size_t)j0 * row_bytes);                              \
            const int vwords = (tk * row_bytes) >> 2;                          \
            /* Every round is guarded, including the first: a partial LAST tile
             * has tk as low as 1, i.e. vwords 36, and an unguarded vsrc[tid]
             * would read up to 1 020 B past the tile — off the end of v_cache
             * for the last kv head. */                                        \
            const unsigned int w0 = (tid < vwords) ? vsrc[tid] : 0u;           \
            const unsigned int w1 = (tid + XV_HEAD_DIM < vwords)               \
                                        ? vsrc[tid + XV_HEAD_DIM] : 0u;        \
            const unsigned int w2 = (tid + 2 * XV_HEAD_DIM < vwords)           \
                                        ? vsrc[tid + 2 * XV_HEAD_DIM] : 0u;    \
            if (tid < vwords)                                                  \
                v_sh[tid] = w0;                                                \
            if (tid + XV_HEAD_DIM < vwords)                                    \
                v_sh[tid + XV_HEAD_DIM] = w1;                                  \
            if (tid + 2 * XV_HEAD_DIM < vwords)                                \
                v_sh[tid + 2 * XV_HEAD_DIM] = w2;                              \
        }
#define XV_BODY_VSH                                                            \
        for (int j = 0; j < tk; j++) {                                         \
            int vp = j0 + j;                                                   \
            float vv;                                                          \
            if (vp < sink)                                                     \
                vv = __half2float(v_sink_head[(size_t)vp * head_dim + tid]);   \
            else if (win > 0 && vp > cur_pos - win && vp > win_live_lo)        \
                vv = __half2float(                                             \
                    v_win_head[(size_t)(vp % FC_KVQ4_WMAX) * head_dim + tid]); \
            else {                                                             \
                const unsigned char *vq = (const unsigned char *)v_sh +        \
                                          j * row_bytes + vqoff;               \
                unsigned short raw =                                           \
                    (unsigned short)(vq[0] | (vq[1] << 8));                    \
                float d = __half2float(                                        \
                    *reinterpret_cast<const __half *>(&raw));                  \
                int qq = (vq[vbyte] >> vshift) & 0xF;                          \
                vv = (float)(qq - 8) * d;                                      \
            }                                                                  \
            XV_PV_GLOOP(vv)                                                    \
        }

/* --------------------------------------------------------- the kernel ---- */
/*
 * A copy of fc_attn_flash_prefill_q4_kr with the PV loop swapped for one of the
 * forms above. Everything else — the q group stage, the register K load, the
 * i-outer QK dot, the shfl fold, the online softmax and all three barriers — is
 * character for character the shipped kernel, so the COPY arm's SASS is the
 * check that this template did not drift.
 *
 * VDECL is empty for every form except VSH, which declares the staged V tile at
 * the end of the shared layout. It is a macro parameter rather than an
 * unconditional declaration so the COPY arm has not one token of extra source
 * over the shipped kernel — a dead pointer would almost certainly be eliminated,
 * but "almost certainly" is not the standard for a control arm.
 */
#define XV_DEFINE_KERNEL(NAME, VDECL, PRE, BODY)                               \
__global__ void NAME(const float *q, const unsigned char *k_cache,             \
                     const unsigned char *v_cache, const __half *k_sink,       \
                     const __half *v_sink, const __half *k_win,                \
                     const __half *v_win, float *out, int kv_groups,           \
                     int head_dim, int max_seq, int pos0, int sink, int win,   \
                     int attn_sink, int attn_win, float scale) {               \
    const int t = blockIdx.x;                                                  \
    const int kvh = blockIdx.y;                                                \
    const int tid = threadIdx.x;                                              \
    const int G = kv_groups;                                                   \
    const int seq_len = pos0 + t + 1;                                          \
    const int cur_pos = seq_len - 1;                                           \
    const int row_bytes = fc_kvq4_row_bytes(head_dim);                         \
    const int win_live_lo = (cur_pos - t + (int)gridDim.x - 1) - FC_KVQ4_WMAX; \
    extern __shared__ float shk[];                                             \
    float *q_s = shk;                                                          \
    float *score_s = q_s + (size_t)G * head_dim;                               \
    float *p_s = score_s + G * FC_ATTN_TILE_KEYS;                              \
    float *m_s = p_s + G * FC_ATTN_TILE_KEYS;                                  \
    float *l_s = m_s + G;                                                      \
    float *corr_s = l_s + G;                                                   \
    VDECL                                                                      \
    const size_t n_head = (size_t)gridDim.y * G;                               \
    const float *q_base =                                                      \
        q + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;                 \
    for (int g = 0; g < G; g++)                                                \
        q_s[(size_t)g * head_dim + tid] = q_base[(size_t)g * head_dim + tid];  \
    if (tid < G) {                                                             \
        m_s[tid] = -__int_as_float(0x7f800000);                                \
        l_s[tid] = 0.0f;                                                       \
    }                                                                          \
    __syncthreads();                                                           \
    float acc[FC_ATTN_MAX_GROUPS];                                             \
    _Pragma("unroll")                                                          \
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)                               \
        acc[g] = 0.0f;                                                         \
    const unsigned char *k_head =                                              \
        k_cache + (size_t)kvh * max_seq * row_bytes;                           \
    const unsigned char *v_head =                                              \
        v_cache + (size_t)kvh * max_seq * row_bytes;                           \
    const __half *k_win_head = k_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;  \
    const __half *v_win_head = v_win + (size_t)kvh * FC_KVQ4_WMAX * head_dim;  \
    const __half *k_sink_head =                                                \
        k_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;                     \
    const __half *v_sink_head =                                                \
        v_sink + (size_t)kvh * FC_KVQ4_SINKMAX * head_dim;                     \
    const int ky = tid / FC_ATTN_DOT_LANES;                                    \
    const int lane = tid % FC_ATTN_DOT_LANES;                                  \
    const int chunk = head_dim / FC_ATTN_DOT_LANES;                            \
    for (int j0 = 0; j0 < seq_len; j0 += FC_ATTN_TILE_KEYS) {                  \
        int tk = seq_len - j0;                                                  \
        if (tk > FC_ATTN_TILE_KEYS) tk = FC_ATTN_TILE_KEYS;                    \
        if (!fc_attn_tile_in_window(j0, tk, cur_pos, attn_sink, attn_win))     \
            continue;                                                          \
        float kreg[FC_ATTN_KREG_MAX_CHUNK];                                    \
        if (ky < tk) {                                                          \
            const int kp = j0 + ky;                                            \
            if (kp < sink) {                                                    \
                const __half *s = k_sink_head + (size_t)kp * head_dim;         \
                _Pragma("unroll")                                              \
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)               \
                    if (i < chunk)                                             \
                        kreg[i] = __half2float(                                \
                            s[lane + i * FC_ATTN_DOT_LANES]);                  \
            } else if (win > 0 && kp > cur_pos - win && kp > win_live_lo) {    \
                const __half *s = k_win_head +                                 \
                    (size_t)(kp % FC_KVQ4_WMAX) * head_dim;                    \
                _Pragma("unroll")                                              \
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++)               \
                    if (i < chunk)                                             \
                        kreg[i] = __half2float(                                \
                            s[lane + i * FC_ATTN_DOT_LANES]);                  \
            } else {                                                            \
                fc_kvq4_dequant_lane_chunk(k_head + (size_t)kp * row_bytes,    \
                                           lane, chunk, kreg);                 \
            }                                                                  \
        }                                                                      \
        {                                                                      \
            float part[FC_ATTN_MAX_GROUPS];                                    \
            _Pragma("unroll")                                                  \
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)                       \
                part[g] = 0.0f;                                                \
            if (ky < tk) {                                                      \
                _Pragma("unroll")                                              \
                for (int i = 0; i < FC_ATTN_KREG_MAX_CHUNK; i++) {             \
                    if (i < chunk) {                                            \
                        const int e = lane + i * FC_ATTN_DOT_LANES;            \
                        const float kve = kreg[i];                             \
                        _Pragma("unroll")                                      \
                        for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)           \
                            if (g < G)                                          \
                                part[g] += q_s[(size_t)g * head_dim + e] * kve;\
                    }                                                          \
                }                                                              \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++) {                     \
                if (g < G) {                                                    \
                    float p = part[g];                                         \
                    for (int off = FC_ATTN_DOT_LANES / 2; off > 0; off >>= 1)  \
                        p += __shfl_down_sync(0xffffffffu, p, (unsigned)off,   \
                                              FC_ATTN_DOT_LANES);              \
                    if (lane == 0 && ky < tk)                                   \
                        score_s[g * FC_ATTN_TILE_KEYS + ky] = p * scale;       \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        if (tid < G) {                                                          \
            const float *sc = score_s + tid * FC_ATTN_TILE_KEYS;               \
            float m_new = m_s[tid];                                            \
            for (int j = 0; j < tk; j++)                                       \
                m_new = fmaxf(m_new, sc[j]);                                   \
            float corr = expf(m_s[tid] - m_new);                               \
            float sum = 0.0f;                                                  \
            for (int j = 0; j < tk; j++) {                                     \
                float pj = expf(sc[j] - m_new);                                \
                p_s[(size_t)j * G + tid] = pj;                                 \
                sum += pj;                                                     \
            }                                                                  \
            l_s[tid] = l_s[tid] * corr + sum;                                  \
            m_s[tid] = m_new;                                                  \
            corr_s[tid] = corr;                                                \
        }                                                                      \
        PRE                                                                    \
        __syncthreads();                                                       \
        BODY                                                                   \
        __syncthreads();                                                      \
    }                                                                          \
    float *out_base =                                                          \
        out + ((size_t)t * n_head + (size_t)kvh * G) * head_dim;               \
    _Pragma("unroll")                                                          \
    for (int g = 0; g < FC_ATTN_MAX_GROUPS; g++)                               \
        if (g < G)                                                              \
            out_base[(size_t)g * head_dim + tid] = acc[g] / l_s[g];            \
}

#define XV_VDECL_NONE /* nothing */
#define XV_VDECL_VSH  unsigned int *v_sh = (unsigned int *)(corr_s + G);

XV_DEFINE_KERNEL(bench_attn_copy,   XV_VDECL_NONE, XV_PRE_FLAT,   XV_BODY_FLAT)
XV_DEFINE_KERNEL(bench_attn_poison, XV_VDECL_NONE, XV_PRE_POISON, XV_BODY_POISON)
XV_DEFINE_KERNEL(bench_attn_vind,   XV_VDECL_NONE, XV_PRE_VIND,   XV_BODY_VIND)
XV_DEFINE_KERNEL(bench_attn_vsh,    XV_VDECL_VSH,  XV_PRE_VSH,    XV_BODY_VSH)

/* ---------------------------------------------------------------- host ---- */

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t rc_ = (call);                                              \
        if (rc_ != cudaSuccess) {                                              \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,     \
                    cudaGetErrorString(rc_));                                  \
            return 1;                                                          \
        }                                                                      \
    } while (0)

typedef void (*xv_kernel_fn)(const float *, const unsigned char *,
                             const unsigned char *, const __half *,
                             const __half *, const __half *, const __half *,
                             float *, int, int, int, int, int, int, int, int,
                             float);

struct xv_arm {
    const char *name;
    xv_kernel_fn fn;
    int extra_shared;      /* XV_VTILE_BYTES for VSH, 0 otherwise */
    int reg_expected;      /* from cuobjdump -res-usage on this file's cubin */
    int blocks_expected;
};

/* REG / blocks-per-SMX are COUNTED from cuobjdump -res-usage on a cubin built
 * from this file (nvcc/ptxas 11.4.152, sm_37, -O3):
 *
 *   arm     REG  STACK  LOCAL  blocks/SMX  PV body/key  V global LD  in flight
 *   kr       72    0      0        7           97            3           3
 *   POISON   72    0      0        7           96            3           3
 *   COPY     72    0      0        7           97            3           3
 *   VIND     72    0      0        7           74            3           3
 *   VSH      72    0      0        7           71            0        n/a (shared)
 *
 * The POISON arm is structurally near-identical to the baseline, which is not
 * something to take on trust — the MoE staging gate's poison changed nvcc's
 * unroll decision and had to be demoted to a bit-only control. Counted here:
 * the static PV loop body goes 160 -> 161, exactly +1 instruction, and its q4
 * dequant arm is 41 = 41. So it is a usable timing sanity check as well as a bit
 * control, though its RATIO is still not a lever result.
 *
 * ALL FOUR ARMS HOLD 72 REGISTERS. That was this lever's entire risk — the
 * kernel sits exactly on the cliff, floor(131072/(72*256)) = 7 and 80 registers
 * gives 6 — and it is counted clear. VSH additionally costs 2 304 B of shared
 * (9 312 -> 11 616), which is still 7 because 7 is the REGISTER cap.
 *
 * If this binary reports different numbers it did not compile the form it claims
 * to, and no ratio below is trustworthy. */
static const struct xv_arm xv_arms[] = {
    { "kr(self)", fc_attn_flash_prefill_q4_kr, 0,             72, 7 },
    { "POISON",   bench_attn_poison,           0,             72, 7 },
    { "COPY",     bench_attn_copy,             0,             72, 7 },
    { "VIND",     bench_attn_vind,             0,             72, 7 },
    { "VSH",      bench_attn_vsh,              XV_VTILE_BYTES, 72, 7 },
};
#define XV_N_ARMS ((int)(sizeof(xv_arms) / sizeof(xv_arms[0])))
#define XV_ARM_SELF   0
#define XV_ARM_POISON 1

static unsigned xv_lcg(unsigned *st) {
    *st = *st * 1664525u + 1013904223u;
    return *st;
}

static void xv_fill_f32(float *p, size_t n, unsigned seed, float lo, float hi) {
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        float u = (float)(xv_lcg(&st) >> 8) / (float)(1u << 24);
        p[i] = lo + u * (hi - lo);
    }
}

/* fp16 shadow rows, written as raw halves so the host needs no fp16 support. */
static void xv_fill_f16(unsigned short *p, size_t n, unsigned seed) {
    unsigned st = seed | 1u;
    for (size_t i = 0; i < n; i++) {
        /* sign 0, exponent 0x0e..0x11 (magnitudes ~0.25..2), random mantissa:
         * finite, non-denormal, and small enough that a 14 080-term online
         * softmax cannot reach inf and let two arms differ for the wrong
         * reason. */
        unsigned e = 0x0eu + (xv_lcg(&st) & 3u);
        unsigned m = xv_lcg(&st) & 0x3ffu;
        unsigned s = (xv_lcg(&st) >> 3) & 1u;
        p[i] = (unsigned short)((s << 15) | (e << 10) | m);
    }
}

/* Synthetic Q4_0 rows, 18 B per 32-element block: fp16 scale then 16 nibble
 * pairs. Scale kept in the same magnitude band as the fp16 shadow. */
static void xv_fill_q4_rows(unsigned char *buf, size_t nrows, int row_bytes,
                            unsigned seed) {
    unsigned st = seed | 1u;
    for (size_t r = 0; r < nrows; r++) {
        unsigned char *row = buf + r * (size_t)row_bytes;
        for (int b = 0; b < row_bytes / FC_KVQ4_BLK_BYTES; b++) {
            unsigned char *blk = row + b * FC_KVQ4_BLK_BYTES;
            unsigned e = 0x0cu + (xv_lcg(&st) & 3u);
            unsigned m = xv_lcg(&st) & 0x3ffu;
            unsigned short d = (unsigned short)((1u << 15) | (e << 10) | m);
            blk[0] = (unsigned char)(d & 0xff);
            blk[1] = (unsigned char)(d >> 8);
            for (int i = 0; i < 16; i++)
                blk[2 + i] = (unsigned char)(xv_lcg(&st) >> 24);
        }
    }
}

int main(int argc, char **argv) {
    int form = XV_ARM_SELF, sweep = 0, reps = 3, depth = XV_DEPTH;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--self"))
            form = XV_ARM_SELF;
        else if (!strcmp(argv[i], "--poison"))
            form = XV_ARM_POISON;
        else if (!strcmp(argv[i], "--sweep"))
            sweep = 1;
        else if (!strcmp(argv[i], "--form") && i + 1 < argc)
            form = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth") && i + 1 < argc)
            depth = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--self|--poison|--form N|--sweep] "
                            "[--reps N] [--depth N]\n", argv[0]);
            return 2;
        }
    }
    if (form < 0 || form >= XV_N_ARMS) {
        fprintf(stderr, "--form must be 0..%d\n", XV_N_ARMS - 1);
        return 2;
    }
    if (depth < XV_NPOS) {
        fprintf(stderr, "--depth must be >= %d\n", XV_NPOS);
        return 2;
    }

    const int head_dim = XV_HEAD_DIM, G = XV_KV_GROUPS;
    const int n_head_kv = XV_N_HEAD_KV, npos = XV_NPOS;
    const int row_bytes = XV_ROW_BYTES;
    const int max_seq = depth;
    const int pos0 = depth - npos;
    const int n_head = n_head_kv * G;
    const float scale = 1.0f / sqrtf((float)head_dim);

    const size_t n_q = (size_t)npos * n_head * head_dim;
    const size_t n_out = n_q;
    const size_t cache_bytes = (size_t)n_head_kv * max_seq * row_bytes;
    const size_t n_win = (size_t)n_head_kv * FC_KVQ4_WMAX * head_dim;
    const size_t n_snk = (size_t)n_head_kv * FC_KVQ4_SINKMAX * head_dim;

    float *h_q = (float *)malloc(n_q * sizeof(float));
    float *h_oa = (float *)malloc(n_out * sizeof(float));
    float *h_ob = (float *)malloc(n_out * sizeof(float));
    unsigned char *h_kv = (unsigned char *)malloc(cache_bytes);
    unsigned short *h_w16 = (unsigned short *)malloc(n_win * 2);
    unsigned short *h_s16 = (unsigned short *)malloc(n_snk * 2);
    if (!h_q || !h_oa || !h_ob || !h_kv || !h_w16 || !h_s16) {
        fprintf(stderr, "host alloc failed\n");
        return 1;
    }
    xv_fill_f32(h_q, n_q, 0x51ceu, -1.0f, 1.0f);

    float *d_q, *d_out;
    unsigned char *d_k, *d_v;
    unsigned short *d_kw, *d_vw, *d_ks, *d_vs;
    CHECK(cudaMalloc(&d_q, n_q * sizeof(float)));
    CHECK(cudaMalloc(&d_out, n_out * sizeof(float)));
    CHECK(cudaMalloc(&d_k, cache_bytes));
    CHECK(cudaMalloc(&d_v, cache_bytes));
    CHECK(cudaMalloc(&d_kw, n_win * 2));
    CHECK(cudaMalloc(&d_vw, n_win * 2));
    CHECK(cudaMalloc(&d_ks, n_snk * 2));
    CHECK(cudaMalloc(&d_vs, n_snk * 2));
    CHECK(cudaMemcpy(d_q, h_q, n_q * sizeof(float), cudaMemcpyHostToDevice));
    xv_fill_q4_rows(h_kv, cache_bytes / row_bytes, row_bytes, 0x1111u);
    CHECK(cudaMemcpy(d_k, h_kv, cache_bytes, cudaMemcpyHostToDevice));
    xv_fill_q4_rows(h_kv, cache_bytes / row_bytes, row_bytes, 0x2222u);
    CHECK(cudaMemcpy(d_v, h_kv, cache_bytes, cudaMemcpyHostToDevice));
    xv_fill_f16(h_w16, n_win, 0x3333u);
    CHECK(cudaMemcpy(d_kw, h_w16, n_win * 2, cudaMemcpyHostToDevice));
    xv_fill_f16(h_w16, n_win, 0x4444u);
    CHECK(cudaMemcpy(d_vw, h_w16, n_win * 2, cudaMemcpyHostToDevice));
    xv_fill_f16(h_s16, n_snk, 0x5555u);
    CHECK(cudaMemcpy(d_ks, h_s16, n_snk * 2, cudaMemcpyHostToDevice));
    xv_fill_f16(h_s16, n_snk, 0x6666u);
    CHECK(cudaMemcpy(d_vs, h_s16, n_snk * 2, cudaMemcpyHostToDevice));

    const unsigned base_shared =
        (unsigned)((G * head_dim + 2 * G * FC_ATTN_TILE_KEYS + 3 * G) *
                   sizeof(float));

    printf("shape        head_dim=%d kv_groups=%d n_head_kv=%d npos=%d "
           "depth=%d pos0=%d row_bytes=%d\n", head_dim, G, n_head_kv, npos,
           depth, pos0, row_bytes);
    printf("kv path      sink=%d win=%d  => tiles/block ~%d, of which %d read "
           "the fp16 window (%.2f %%)\n", XV_SINK, XV_WIN,
           (depth + FC_ATTN_TILE_KEYS - 1) / FC_ATTN_TILE_KEYS,
           XV_WIN / FC_ATTN_TILE_KEYS,
           100.0 * (XV_WIN / FC_ATTN_TILE_KEYS) /
               (double)((depth + FC_ATTN_TILE_KEYS - 1) / FC_ATTN_TILE_KEYS));
    printf("grid         (%d,%d) = %d blocks against 13 SMX\n", npos,
           n_head_kv, npos * n_head_kv);
    printf("shared       base %u B, VSH adds %d B -> %u B "
           "(114688/7 = 16384 B is the budget at 7 blocks/SMX)\n",
           base_shared, XV_VTILE_BYTES, base_shared + XV_VTILE_BYTES);
    printf("resources    (OFFLINE ptxas in this binary — the shipping path JITs "
           "from PTX; cuFuncGetAttribute on the LOADED function is the "
           "authority for a ship decision)\n");

    int mismatch = 0;
    for (int a = 0; a < XV_N_ARMS; a++) {
        cudaFuncAttributes at;
        int occ = 0;
        CHECK(cudaFuncGetAttributes(&at, (const void *)xv_arms[a].fn));
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, (const void *)xv_arms[a].fn, head_dim,
            base_shared + xv_arms[a].extra_shared));
        int bad = (at.numRegs != xv_arms[a].reg_expected ||
                   occ != xv_arms[a].blocks_expected);
        mismatch |= bad;
        printf("  %-9s REG %3d (expect %3d)  frame %4zu B  dyn shared %5u B  "
               "blocks/SMX %d (expect %d)%s\n",
               xv_arms[a].name, at.numRegs, xv_arms[a].reg_expected,
               (size_t)at.localSizeBytes,
               base_shared + xv_arms[a].extra_shared, occ,
               xv_arms[a].blocks_expected, bad ? "   <== MISMATCH" : "");
    }
    if (mismatch)
        printf("WARNING      an arm's REG/occupancy does not match the counted "
               "table. This binary may have compiled a different form than the "
               "source states (a _Pragma dropped inside a macro ARGUMENT does "
               "exactly this). Do NOT trust a ratio until the arm is "
               "re-identified from SASS. Registers are this lever's whole gate: "
               "72 -> 80 is 7 -> 6 blocks/SMX = 0.857x.\n");

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    const int first = sweep ? XV_ARM_POISON + 1 : form;
    const int last = sweep ? XV_N_ARMS - 1 : form;
    printf("\n%-9s %10s %10s %10s %8s   %s\n", "arm", "A1 ms", "B ms", "A2 ms",
           "ratio", "out bits");

    for (int a = first; a <= last; a++) {
        double tm[3] = { 0.0, 0.0, 0.0 };
        size_t diff = 0;
        double dmax = 0.0, drelmax = 0.0;
        for (int pass = 0; pass < 3; pass++) {
            xv_kernel_fn fn = (pass == 1) ? xv_arms[a].fn
                                          : fc_attn_flash_prefill_q4_kr;
            unsigned shb = base_shared +
                           ((pass == 1) ? xv_arms[a].extra_shared : 0);
            /* out poisoned to 0xCD first so an element no arm writes cannot
             * masquerade as a match. Exactly one launch for correctness. */
            CHECK(cudaMemset(d_out, 0xCD, n_out * sizeof(float)));
            fn<<<dim3(npos, n_head_kv), head_dim, shb>>>(
                d_q, d_k, d_v, (const __half *)d_ks, (const __half *)d_vs,
                (const __half *)d_kw, (const __half *)d_vw, d_out, G, head_dim,
                max_seq, pos0, XV_SINK, XV_WIN, 0, 0, scale);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy((pass == 1) ? h_ob : h_oa, d_out,
                             n_out * sizeof(float), cudaMemcpyDeviceToHost));
            for (int w = 0; w < 1; w++)
                fn<<<dim3(npos, n_head_kv), head_dim, shb>>>(
                    d_q, d_k, d_v, (const __half *)d_ks, (const __half *)d_vs,
                    (const __half *)d_kw, (const __half *)d_vw, d_out, G,
                    head_dim, max_seq, pos0, XV_SINK, XV_WIN, 0, 0, scale);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < reps; it++)
                fn<<<dim3(npos, n_head_kv), head_dim, shb>>>(
                    d_q, d_k, d_v, (const __half *)d_ks, (const __half *)d_vs,
                    (const __half *)d_kw, (const __half *)d_vw, d_out, G,
                    head_dim, max_seq, pos0, XV_SINK, XV_WIN, 0, 0, scale);
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());
            float ms = 0.0f;
            CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            tm[pass] = (double)ms / (double)reps;
        }
        for (size_t i = 0; i < n_out; i++) {
            if (memcmp(&h_oa[i], &h_ob[i], sizeof(float)) != 0) {
                diff++;
                double d = fabs((double)h_oa[i] - (double)h_ob[i]);
                double r = h_oa[i] != 0.0f ? d / fabs((double)h_oa[i]) : d;
                if (d > dmax) dmax = d;
                if (r > drelmax) drelmax = r;
            }
        }
        double abase = 0.5 * (tm[0] + tm[2]);
        printf("%-9s %10.4f %10.4f %10.4f %8.4fx  %s (%zu/%zu differ, max |d| "
               "%.3e, max rel %.3e)   A-drift %.3f %%\n",
               xv_arms[a].name, tm[0], tm[1], tm[2],
               tm[1] > 0.0 ? abase / tm[1] : 0.0,
               diff ? "DIFFER" : "EXACT", diff, n_out, dmax, drelmax,
               tm[0] > 0.0 ? 100.0 * (tm[2] - tm[0]) / tm[0] : 0.0);

        if (a == XV_ARM_SELF && diff)
            printf("VERDICT      BROKEN — --self must be EXACT; every other "
                   "verdict from this harness is void\n");
        else if (a == XV_ARM_POISON && !diff)
            printf("VERDICT      BROKEN — --poison must be DIFFER; the bit "
                   "comparison is not comparing bits\n");
        else if (a == XV_ARM_SELF)
            printf("VERDICT      control OK; the ratio above is this process's "
                   "noise floor (run 10 processes for the real one)\n");
        else if (a == XV_ARM_POISON)
            printf("VERDICT      negative control fires\n");
        else
            printf("VERDICT      %s vs shipped fc_attn_flash_prefill_q4_kr: "
                   "%s\n", xv_arms[a].name,
                   diff ? "NOT bit-exact — STOP, the construction argument is "
                          "wrong and there is a real bug"
                        : "bit-exact");
    }

    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_k));
    CHECK(cudaFree(d_v));
    CHECK(cudaFree(d_kw));
    CHECK(cudaFree(d_vw));
    CHECK(cudaFree(d_ks));
    CHECK(cudaFree(d_vs));
    free(h_q);
    free(h_oa);
    free(h_ob);
    free(h_kv);
    free(h_w16);
    free(h_s16);
    return 0;
}
