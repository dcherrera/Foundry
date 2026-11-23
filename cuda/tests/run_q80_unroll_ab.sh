#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# run_q80_unroll_ab.sh — Q8_0 group-loop unroll A/B on ONE die (bank, CUDA 11.4).
#
# Builds matvec_kquants.ptx at BASE_REF (default origin/main) and at this tree once per
# variant (extra nvcc flags, e.g. -DFC_Q80_G4_UNROLL=1), offline-ptxas cubins (+ ptxas -v),
# then on the die: the driver-JIT cubin of every build (the SASS production runs), the
# null A/B base vs base, and base vs each variant with bench_matvec_q80_ab on the live
# TP2 per-die shapes (Q80_SHAPES=tp2), n = 1/2/3/4/5/8/16, memcmp on every shape.
# Then test_matvec_q80_parity: base vs the CPU replica, each variant vs the replica and base
# (a variant built with -DFC_Q80_G4_RED must FAIL both the parity test and the A/B memcmp).
#
#   [VARIANTS="u2: tile2:-DFC_Q80_TILE_UNROLL=2"]   name:flags, ',' separates flags
#   [BASE_REF=origin/main] [ROUNDS=25]
#   setsid nohup bash cuda/tests/run_q80_unroll_ab.sh <out dir> <die 4-11 | build> > log 2>&1 < /dev/null &
# Writes <out>/DONE (or BUILT for "build", which touches no GPU) or <out>/FAILED with the step.
#
# @status complete  @issues none  @todo none
set -uo pipefail
OUT=${1:?out dir}
DIE=${2:?die 4-11 or build}
FOUNDRY=$(cd "$(dirname "$0")/../.." && pwd)
CUDA=/usr/local/cuda-11.4
BASE_REF=${BASE_REF:-origin/main}
VARIANTS=${VARIANTS:-u2:}
ROUNDS=${ROUNDS:-25}
NVFLAGS=(-ptx -arch=sm_37 -ccbin gcc-11 -Wno-deprecated-gpu-targets
         -isystem "$FOUNDRY/cuda_kernels/nvcc_shim" -I "$FOUNDRY/cuda_kernels")
mkdir -p "$OUT"
fail() { echo "FAILED: $1" > "$OUT/FAILED"; exit 1; }
case "$DIE" in 4|5|6|7|8|9|10|11|build) ;; *) fail "die must be 4-11 or build (got $DIE)" ;; esac

names=base
git -C "$FOUNDRY" show "$BASE_REF:cuda_kernels/matvec_kquants.cu" > "$OUT/base.cu" || fail base-source
"$CUDA/bin/nvcc" "${NVFLAGS[@]}" -o "$OUT/base.ptx" "$OUT/base.cu" || fail nvcc-base
for v in $VARIANTS; do
    name=${v%%:*} flags=${v#*:}
    # shellcheck disable=SC2086  # flags is a word list on purpose
    "$CUDA/bin/nvcc" "${NVFLAGS[@]}" ${flags//,/ } -o "$OUT/$name.ptx" \
        "$FOUNDRY/cuda_kernels/matvec_kquants.cu" || fail "nvcc-$name"
    echo "$name: ${flags//,/ }" >> "$OUT/variants.txt"
    names="$names $name"
done
for name in $names; do
    "$CUDA/bin/ptxas" -arch=sm_37 -v -o "$OUT/$name.offline.cubin" "$OUT/$name.ptx" \
        > "$OUT/$name.ptxas.txt" 2>&1 || fail "ptxas-$name"
done
gcc -O2 -I"$CUDA/include" -o "$OUT/bench_matvec_q80_ab" "$FOUNDRY/cuda/tests/bench_matvec_q80_ab.c" \
    -lcuda -lm || fail gcc-bench
gcc -O2 -ffp-contract=off -I"$CUDA/include" -o "$OUT/test_matvec_q80_parity" \
    "$FOUNDRY/cuda/tests/test_matvec_q80_parity.c" -lcuda -lm || fail gcc-parity
if [ "$DIE" = build ]; then
    echo BUILT > "$OUT/BUILT"
    exit 0
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DIE
[ -z "$(nvidia-smi -i "$DIE" --query-compute-apps=pid --format=csv,noheader)" ] || fail "die $DIE busy"
smi() { nvidia-smi -i "$DIE" --query-gpu=index,pci.bus_id,clocks.sm,power.draw,temperature.gpu,memory.used \
            --format=csv > "$OUT/smi_$1.csv"; }
smi before
for name in $names; do
    "$OUT/bench_matvec_q80_ab" --dump-cubin "$OUT/$name.ptx" "$OUT/$name.jit.cubin" || fail "jit-$name"
done
Q80_SHAPES=tp2 "$OUT/bench_matvec_q80_ab" "$OUT/base.ptx" "$OUT/base.ptx" "$ROUNDS" > "$OUT/ab_null.txt" 2>&1 ||
    fail null-ab
"$OUT/test_matvec_q80_parity" "$OUT/base.ptx" > "$OUT/parity_base.txt" 2>&1
echo "parity base rc=$?" >> "$OUT/parity_rc.txt"
for name in ${names#base}; do
    "$OUT/test_matvec_q80_parity" "$OUT/$name.ptx" "$OUT/base.ptx" > "$OUT/parity_$name.txt" 2>&1
    echo "parity $name rc=$?" >> "$OUT/parity_rc.txt"
done
for name in ${names#base}; do
    Q80_SHAPES=tp2 "$OUT/bench_matvec_q80_ab" "$OUT/base.ptx" "$OUT/$name.ptx" "$ROUNDS" \
        > "$OUT/ab_$name.txt" 2>&1
    echo "ab $name rc=$?" >> "$OUT/ab_rc.txt"
done
smi after
echo DONE > "$OUT/DONE"
