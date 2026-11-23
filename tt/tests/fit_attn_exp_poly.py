#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
fit_attn_exp_poly.py
    Derive the polynomial of the attention fast exp (kernels/ft_tensix_sfpu_exp_attention.h):
    p(f) = c0 + c1*f + c2*f^2 ~ e^f on |f| <= ln2/2, minimising the max RELATIVE error
    |p(f) * e^-f - 1|, with c1 and c2 fp16-representable (the compiler then loads each with
    ONE short SFPLOADI) and c0 a full fp32 (it lives in a programmable constant register).

      1. Lawson's iteratively reweighted least squares: the real-valued minimax (c0, c1, c2);
      2. the fp16 grid around (c1, c2), each pair with its best c0 (a ternary search: the max
         error is convex in c0), c0 rounded to fp32; the pair with the smallest max error wins.

    Pinning c0 = 1 (so exp(0) = 1 exactly) costs 2.0e-3 = 2^-8.96 here, over the 2^-9 bound,
    which is why c0 is free; degree 3 with c0 = 1 is ~1e-4 for one more MAD.

    Usage: python3 tt/tests/fit_attn_exp_poly.py   (numpy only; prints the constants)
    The kernel's constants were taken from this output; test_attn_fast_exp.c checks them
    through the SFPU arithmetic over the whole input range.

    Status: added with ATTN_FAST_EXP (Qwen3.8 prefill attention fast-math).
    Issues: None
    Todo:   None
"""

import numpy as np

H = np.log(2.0) / 2.0 * (1.0 + 1e-4)  # the reduced range, with a little slop
F = np.linspace(-H, H, 20001)
E = np.exp(-F)


def max_rel(c0, c1, c2):
    return float(np.max(np.abs((c0 + F * (c1 + F * c2)) * E - 1.0)))


def lawson(iterations=4000):
    basis = np.stack([E, F * E, F * F * E], axis=1)
    target = np.ones_like(F)
    w = np.full(F.size, 1.0 / F.size)
    c = np.zeros(3)
    for _ in range(iterations):
        sw = np.sqrt(w)
        c, *_ = np.linalg.lstsq(basis * sw[:, None], target * sw, rcond=None)
        w = w * (np.abs(basis @ c - target) + 1e-300)
        w /= w.sum()
    return c


def best_c0(c1, c2):
    lo, hi = 0.998, 1.002
    for _ in range(80):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if max_rel(m1, c1, c2) < max_rel(m2, c1, c2):
            hi = m2
        else:
            lo = m1
    return float(np.float32((lo + hi) / 2))


def fp16_neighbours(x, span):
    bits = int(np.float16(x).view(np.uint16))
    return [float(np.uint16(bits + k).view(np.float16)) for k in range(-span, span + 1)]


def main():
    c0, c1, c2 = lawson()
    print(f"real minimax: c0={c0:.9f} c1={c1:.9f} c2={c2:.9f} max rel {max_rel(c0, c1, c2):.4e}")
    best = None
    for h1 in fp16_neighbours(c1, 12):
        for h2 in fp16_neighbours(c2, 12):
            z = best_c0(h1, h2)
            e = max_rel(z, h1, h2)
            if best is None or e < best[0]:
                best = (e, z, h1, h2)
    e, z, h1, h2 = best
    zbits = int(np.float32(z).view(np.uint32))
    print(f"kernel:       c0={z:.9g} (fp32 0x{zbits:08X}) c1={h1:.10g} (fp16 "
          f"0x{int(np.float16(h1).view(np.uint16)):04X}) c2={h2:.12g} (fp16 "
          f"0x{int(np.float16(h2).view(np.uint16)):04X}) max rel {e:.4e} (2^{np.log2(e):.2f})")


if __name__ == "__main__":
    main()
