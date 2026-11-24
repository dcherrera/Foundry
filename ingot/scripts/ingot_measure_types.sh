#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# ingot_measure_types.sh — per tensor type totals from the --tensor-log files of an
# ingot_measure.sh run: tensors, predicted count, weights, v3 and v2 bits per weight of that
# type, saved bits per weight of the type and of the whole model.
#   ingot/scripts/ingot_measure_types.sh <out-dir>
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: ingot_measure_types.sh <out-dir>" >&2; exit 2; }
cat "$1"/*.tensors.tsv | awk -F'\t' '
    /^#/ || $1 == "(filler)" {next}
    {
        n = split($3, d, "x"); w = 1; for (i = 1; i <= n; i++) w *= d[i]
        t = $1; sub(/^(model\.)?(language_model\.)?(mtp\.)?layers\.[0-9]+\./, "", t)
        sub(/^model\.(language_model\.)?/, "", t)
        if ($1 ~ /^mtp\./) t = "mtp." t
        T[t]++; W[t] += w; P[t] += $5; V[t] += $6; if ($4 == "predict" || $4 == "match") K[t]++
        all += w
    }
    END {
        printf "#type\ttensors\tpredicted\tweights\tv3_bpw\tv2_bpw\tsaved_bpw\tsaved_model_bpw\n"
        for (t in T)
            printf "%s\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.5f\n", t, T[t], K[t], W[t],
                   8 * P[t] / W[t], 8 * V[t] / W[t], 8 * (V[t] - P[t]) / W[t],
                   8 * (V[t] - P[t]) / all
    }' | sort -t$'\t' -k8,8gr
