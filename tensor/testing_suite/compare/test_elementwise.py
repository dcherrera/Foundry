# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
Comparison tests for element-wise operations.
"""

import numpy as np
import torch
from compare_utils import ComparisonTest, TestRunner


class TestAdd(ComparisonTest):
    def __init__(self):
        super().__init__("add", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(100, 100).astype(np.float32)
        b = np.random.randn(100, 100).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.add(a, b)


class TestSub(ComparisonTest):
    def __init__(self):
        super().__init__("sub", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(100, 100).astype(np.float32)
        b = np.random.randn(100, 100).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.sub(a, b)


class TestMul(ComparisonTest):
    def __init__(self):
        super().__init__("mul", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(100, 100).astype(np.float32)
        b = np.random.randn(100, 100).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.mul(a, b)


class TestDiv(ComparisonTest):
    def __init__(self):
        super().__init__("div", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(100, 100).astype(np.float32)
        # Avoid division by zero - ensure |b| >= 0.1
        b = np.random.randn(100, 100).astype(np.float32)
        b = np.where(np.abs(b) < 0.1, np.sign(b) * 0.1 + (np.sign(b) == 0) * 0.1, b)
        b = np.where(b == 0, 0.1, b)  # Extra safety for any remaining zeros
        return (a, b.astype(np.float32))

    def pytorch_forward(self, a, b):
        return torch.div(a, b)


class TestNeg(ComparisonTest):
    def __init__(self):
        super().__init__("neg", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.neg(x)


class TestAbs(ComparisonTest):
    def __init__(self):
        super().__init__("abs", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 10,)

    def pytorch_forward(self, x):
        return torch.abs(x)


class TestExp(ComparisonTest):
    def __init__(self):
        super().__init__("exp", tolerance=2e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Keep values small to avoid overflow
        return (np.random.randn(1000).astype(np.float32) * 2,)

    def pytorch_forward(self, x):
        return torch.exp(x)


class TestLog(ComparisonTest):
    def __init__(self):
        super().__init__("log", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Only positive values for log
        return (np.abs(np.random.randn(1000).astype(np.float32)) + 0.01,)

    def pytorch_forward(self, x):
        return torch.log(x)


class TestSqrt(ComparisonTest):
    def __init__(self):
        super().__init__("sqrt", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Only positive values for sqrt
        return (np.abs(np.random.randn(1000).astype(np.float32)) + 0.01,)

    def pytorch_forward(self, x):
        return torch.sqrt(x)


class TestRsqrt(ComparisonTest):
    def __init__(self):
        super().__init__("rsqrt", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Only positive values for rsqrt
        return (np.abs(np.random.randn(1000).astype(np.float32)) + 0.1,)

    def pytorch_forward(self, x):
        return torch.rsqrt(x)


class TestPow(ComparisonTest):
    def __init__(self):
        super().__init__("pow", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Positive base for fractional exponents
        base = np.abs(np.random.randn(1000).astype(np.float32)) + 0.1
        return (base,)

    def pytorch_forward(self, x):
        return torch.pow(x, 2.0)

    def foundry_params(self):
        return {"exponent": 2.0}


class TestSin(ComparisonTest):
    def __init__(self):
        super().__init__("sin", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3.14,)

    def pytorch_forward(self, x):
        return torch.sin(x)


class TestCos(ComparisonTest):
    def __init__(self):
        super().__init__("cos", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3.14,)

    def pytorch_forward(self, x):
        return torch.cos(x)


class TestTan(ComparisonTest):
    def __init__(self):
        super().__init__("tan", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Avoid values near pi/2 where tan explodes
        x = np.random.randn(1000).astype(np.float32) * 0.5
        return (x,)

    def pytorch_forward(self, x):
        return torch.tan(x)


class TestSinh(ComparisonTest):
    def __init__(self):
        super().__init__("sinh", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 2,)

    def pytorch_forward(self, x):
        return torch.sinh(x)


class TestCosh(ComparisonTest):
    def __init__(self):
        super().__init__("cosh", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 2,)

    def pytorch_forward(self, x):
        return torch.cosh(x)


class TestAsin(ComparisonTest):
    def __init__(self):
        super().__init__("asin", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # asin domain is [-1, 1]
        return (np.random.uniform(-0.99, 0.99, 1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.asin(x)


class TestAcos(ComparisonTest):
    def __init__(self):
        super().__init__("acos", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # acos domain is [-1, 1]
        return (np.random.uniform(-0.99, 0.99, 1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.acos(x)


class TestAtan(ComparisonTest):
    def __init__(self):
        super().__init__("atan", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 5,)

    def pytorch_forward(self, x):
        return torch.atan(x)


class TestFloor(ComparisonTest):
    def __init__(self):
        super().__init__("floor", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 10,)

    def pytorch_forward(self, x):
        return torch.floor(x)


class TestCeil(ComparisonTest):
    def __init__(self):
        super().__init__("ceil", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 10,)

    def pytorch_forward(self, x):
        return torch.ceil(x)


class TestRound(ComparisonTest):
    def __init__(self):
        super().__init__("round", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 10,)

    def pytorch_forward(self, x):
        return torch.round(x)


class TestClamp(ComparisonTest):
    def __init__(self):
        super().__init__("clamp", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 5,)

    def pytorch_forward(self, x):
        return torch.clamp(x, min=-2.0, max=2.0)

    def foundry_params(self):
        return {"min": -2.0, "max": 2.0}


class TestMaximum(ComparisonTest):
    def __init__(self):
        super().__init__("maximum", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(1000).astype(np.float32)
        b = np.random.randn(1000).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.maximum(a, b)


class TestMinimum(ComparisonTest):
    def __init__(self):
        super().__init__("minimum", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        a = np.random.randn(1000).astype(np.float32)
        b = np.random.randn(1000).astype(np.float32)
        return (a, b)

    def pytorch_forward(self, a, b):
        return torch.minimum(a, b)


def get_all_tests():
    """Return all element-wise tests."""
    return [
        TestAdd(),
        TestSub(),
        TestMul(),
        TestDiv(),
        TestNeg(),
        TestAbs(),
        TestExp(),
        TestLog(),
        TestSqrt(),
        TestRsqrt(),
        TestPow(),
        TestSin(),
        TestCos(),
        TestTan(),
        TestSinh(),
        TestCosh(),
        TestAsin(),
        TestAcos(),
        TestAtan(),
        TestFloor(),
        TestCeil(),
        TestRound(),
        TestClamp(),
        TestMaximum(),
        TestMinimum(),
    ]


if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv

    runner = TestRunner()
    for test in get_all_tests():
        runner.add_test(test)

    runner.run_all(debug=debug)
    runner.save_results()
