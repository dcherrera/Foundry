# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
Comparison tests for activation functions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from compare_utils import ComparisonTest, TestRunner


class TestReLU(ComparisonTest):
    def __init__(self):
        super().__init__("relu", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return F.relu(x)


class TestSigmoid(ComparisonTest):
    def __init__(self):
        super().__init__("sigmoid", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 5,)

    def pytorch_forward(self, x):
        return torch.sigmoid(x)


class TestTanh(ComparisonTest):
    def __init__(self):
        super().__init__("tanh", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return torch.tanh(x)


class TestGELU(ComparisonTest):
    def __init__(self):
        # GELU uses different approximations - allow larger tolerance
        super().__init__("gelu", tolerance=1e-3)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return F.gelu(x)


class TestSiLU(ComparisonTest):
    def __init__(self):
        super().__init__("silu", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return F.silu(x)


class TestMish(ComparisonTest):
    def __init__(self):
        super().__init__("mish", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return F.mish(x)


class TestSELU(ComparisonTest):
    def __init__(self):
        super().__init__("selu", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return F.selu(x)


class TestCELU(ComparisonTest):
    def __init__(self):
        super().__init__("celu", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return F.celu(x, alpha=1.0)

    def foundry_params(self):
        return {"alpha": 1.0}


class TestELU(ComparisonTest):
    def __init__(self):
        super().__init__("elu", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return F.elu(x, alpha=1.0)

    def foundry_params(self):
        return {"alpha": 1.0}


class TestReLU6(ComparisonTest):
    def __init__(self):
        super().__init__("relu6", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 10,)

    def pytorch_forward(self, x):
        return F.relu6(x)


class TestLeakyReLU(ComparisonTest):
    def __init__(self):
        super().__init__("leaky_relu", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 5,)

    def pytorch_forward(self, x):
        return F.leaky_relu(x, negative_slope=0.01)

    def foundry_params(self):
        return {"slope": 0.01}


class TestSoftmax(ComparisonTest):
    def __init__(self):
        super().__init__("softmax", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(32, 100).astype(np.float32) * 2,)

    def pytorch_forward(self, x):
        return F.softmax(x, dim=-1)

    def foundry_params(self):
        return {"dim": -1}


class TestLogSoftmax(ComparisonTest):
    def __init__(self):
        super().__init__("log_softmax", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(32, 100).astype(np.float32) * 2,)

    def pytorch_forward(self, x):
        return F.log_softmax(x, dim=-1)

    def foundry_params(self):
        return {"dim": -1}


class TestHardsigmoid(ComparisonTest):
    def __init__(self):
        super().__init__("hardsigmoid", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 5,)

    def pytorch_forward(self, x):
        return F.hardsigmoid(x)


class TestHardswish(ComparisonTest):
    def __init__(self):
        super().__init__("hardswish", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 5,)

    def pytorch_forward(self, x):
        return F.hardswish(x)


class TestHardtanh(ComparisonTest):
    def __init__(self):
        super().__init__("hardtanh", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return F.hardtanh(x, min_val=-1.0, max_val=1.0)

    def foundry_params(self):
        return {"min": -1.0, "max": 1.0}


class TestSoftsign(ComparisonTest):
    def __init__(self):
        super().__init__("softsign", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 5,)

    def pytorch_forward(self, x):
        return F.softsign(x)


class TestTanhshrink(ComparisonTest):
    def __init__(self):
        super().__init__("tanhshrink", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 3,)

    def pytorch_forward(self, x):
        return F.tanhshrink(x)


class TestSoftplus(ComparisonTest):
    def __init__(self):
        super().__init__("softplus", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32) * 5,)

    def pytorch_forward(self, x):
        return F.softplus(x, beta=1.0, threshold=20.0)

    def foundry_params(self):
        return {"beta": 1.0, "threshold": 20.0}


def get_all_tests():
    """Return all activation tests."""
    return [
        TestReLU(),
        TestSigmoid(),
        TestTanh(),
        TestGELU(),
        TestSiLU(),
        TestMish(),
        TestSELU(),
        TestCELU(),
        TestELU(),
        TestReLU6(),
        TestLeakyReLU(),
        TestSoftmax(),
        TestLogSoftmax(),
        TestHardsigmoid(),
        TestHardswish(),
        TestHardtanh(),
        TestSoftsign(),
        TestTanhshrink(),
        TestSoftplus(),
    ]


if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv

    runner = TestRunner()
    for test in get_all_tests():
        runner.add_test(test)

    runner.run_all(debug=debug)
    runner.save_results()
