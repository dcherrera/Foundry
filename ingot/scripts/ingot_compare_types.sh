#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# ingot_compare_types.sh — per tensor type, two runs' --tensor-log files of the same model side by
# side (e.g. version 3 vs version 4): weights, bits per weight of each, the change per weight of
# the type and per weight of the whole model. Tensors present in only one run are skipped.
#   ingot/scripts/ingot_compare_types.sh <old-dir> <new-dir>
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: ingot_compare_types.sh <old-dir> <new-dir>" >&2; exit 2; }
{ for f in "$1"/*.tensors.tsv; do sed 's/^/0\t/' "$f"; done
  for f in "$2"/*.tensors.tsv; do sed 's/^/1\t/' "$f"; done; } | awk -F'\t' '
    $2 ~ /^#/ || $2 == "" {next}
    {
        n = split($4, d, "x"); w = 1; for (i = 1; i <= n; i++) w *= d[i]
        key = $2; B[$1, key] = $6; S[key] = w; seen[$1, key] = 1
    }
    END {
        for (k in S) {
            if (!seen[0, k] || !seen[1, k]) continue
            t = k; sub(/^(model\.)?(language_model\.)?(mtp\.)?layers\.[0-9]+\./, "", t)
            sub(/^model\.(language_model\.)?/, "", t)
            if (k ~ /^mtp\./) t = "mtp." t
            W[t] += S[k]; O[t] += B[0, k]; N[t] += B[1, k]; all += S[k]
        }
        printf "#type\tweights\told_bpw\tnew_bpw\tdelta_bpw\tdelta_model_bpw\n"
        for (t in W)
            printf "%s\t%d\t%.4f\t%.4f\t%+.4f\t%+.5f\n", t, W[t], 8 * O[t] / W[t], 8 * N[t] / W[t],
                   8 * (N[t] - O[t]) / W[t], 8 * (N[t] - O[t]) / all
    }' | sort -t$'\t' -k6,6g
