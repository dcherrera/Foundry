# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
Comparison tests for matrix operations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from compare_utils import ComparisonTest, TestRunner


class TestMatmul(ComparisonTest):
    def __init__(self):
        super().__init__("matmul", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(64, 128).astype(np.float32)
        b = np.random.randn(128, 64).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.matmul(a, b)


class TestMatmulSquare(ComparisonTest):
    def __init__(self):
        super().__init__("matmul_square", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64, 64).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.matmul(a, b)

    def foundry_operation(self):
        return "matmul"


class TestBmm(ComparisonTest):
    def __init__(self):
        super().__init__("bmm", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(8, 32, 64).astype(np.float32)
        b = np.random.randn(8, 64, 32).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.bmm(a, b)


class TestMm(ComparisonTest):
    def __init__(self):
        super().__init__("mm", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(128, 256).astype(np.float32)
        b = np.random.randn(256, 128).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.mm(a, b)


class TestMv(ComparisonTest):
    def __init__(self):
        super().__init__("mv", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        mat = np.random.randn(128, 64).astype(np.float32)
        vec = np.random.randn(64).astype(np.float32)
        return (mat, vec)

    def pytorch_forward(self, mat, vec):
        return torch.mv(mat, vec)


class TestDot(ComparisonTest):
    def __init__(self):
        super().__init__("dot", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(1000).astype(np.float32)
        b = np.random.randn(1000).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.dot(a, b)


class TestOuter(ComparisonTest):
    def __init__(self):
        super().__init__("outer", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(100).astype(np.float32)
        b = np.random.randn(100).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.outer(a, b)


class TestLinear(ComparisonTest):
    def __init__(self):
        super().__init__("linear", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        x = np.random.randn(32, 128).astype(np.float32)
        weight = np.random.randn(64, 128).astype(np.float32)
        bias = np.random.randn(64).astype(np.float32)
        return (x, weight, bias)

    def pytorch_forward(self, x, weight, bias):
        return F.linear(x, weight, bias)


class TestLinearNoBias(ComparisonTest):
    def __init__(self):
        super().__init__("linear_no_bias", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        x = np.random.randn(32, 128).astype(np.float32)
        weight = np.random.randn(64, 128).astype(np.float32)
        return (x, weight)

    def pytorch_forward(self, x, weight):
        return F.linear(x, weight, None)

    def foundry_operation(self):
        return "linear"


class TestAddmm(ComparisonTest):
    def __init__(self):
        super().__init__("addmm", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        bias = np.random.randn(64).astype(np.float32)
        mat1 = np.random.randn(32, 128).astype(np.float32)
        mat2 = np.random.randn(128, 64).astype(np.float32)
        return (bias, mat1, mat2)

    def pytorch_forward(self, bias, mat1, mat2):
        return torch.addmm(bias, mat1, mat2)


class TestTranspose(ComparisonTest):
    def __init__(self):
        super().__init__("transpose", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(64, 128).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.transpose(x, 0, 1)

    def foundry_params(self):
        return {"dim0": 0, "dim1": 1}


class TestPermute(ComparisonTest):
    def __init__(self):
        super().__init__("permute", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(8, 16, 32, 64).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.permute(x, (0, 2, 1, 3))

    def foundry_params(self):
        return {"dims": "0,2,1,3"}


def get_all_tests():
    """Return all matrix operation tests."""
    return [
        TestMatmul(),
        TestMatmulSquare(),
        TestBmm(),
        TestMm(),
        TestMv(),
        TestDot(),
        TestOuter(),
        TestLinear(),
        TestLinearNoBias(),
        TestAddmm(),
        TestTranspose(),
        TestPermute(),
    ]


if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv

    runner = TestRunner()
    for test in get_all_tests():
        runner.add_test(test)

    runner.run_all(debug=debug)
    runner.save_results()
