# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
Comparison tests for reduction operations.
"""

import numpy as np
import torch
from compare_utils import ComparisonTest, TestRunner


class TestSum(ComparisonTest):
    def __init__(self):
        super().__init__("sum", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(100, 100).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.sum(x)


class TestSumDim(ComparisonTest):
    def __init__(self):
        super().__init__("sum_dim", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(32, 64, 128).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.sum(x, dim=1)

    def foundry_operation(self):
        return "sum"

    def foundry_params(self):
        return {"dim": 1}


class TestMean(ComparisonTest):
    def __init__(self):
        super().__init__("mean", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(100, 100).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.mean(x)


class TestMeanDim(ComparisonTest):
    def __init__(self):
        super().__init__("mean_dim", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(32, 64, 128).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.mean(x, dim=1)

    def foundry_operation(self):
        return "mean"

    def foundry_params(self):
        return {"dim": 1}


class TestMax(ComparisonTest):
    def __init__(self):
        super().__init__("max", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.max(x)


class TestMin(ComparisonTest):
    def __init__(self):
        super().__init__("min", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.min(x)


class TestArgmax(ComparisonTest):
    def __init__(self):
        super().__init__("argmax", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(100, 100).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.argmax(x)


class TestArgmaxDim(ComparisonTest):
    def __init__(self):
        super().__init__("argmax_dim", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(32, 64).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.argmax(x, dim=1)

    def foundry_operation(self):
        return "argmax"

    def foundry_params(self):
        return {"dim": 1}


class TestArgmin(ComparisonTest):
    def __init__(self):
        super().__init__("argmin", tolerance=0.0)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(100, 100).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.argmin(x)


class TestStd(ComparisonTest):
    def __init__(self):
        super().__init__("std", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.std(x, unbiased=True)


class TestVar(ComparisonTest):
    def __init__(self):
        super().__init__("var", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.var(x, unbiased=True)


class TestProd(ComparisonTest):
    def __init__(self):
        super().__init__("prod", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Use small positive values to avoid overflow
        return (np.random.uniform(0.9, 1.1, 100).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.prod(x)


class TestNorm(ComparisonTest):
    def __init__(self):
        super().__init__("norm", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.norm(x, p=2)

    def foundry_params(self):
        return {"p": 2}


class TestNormL1(ComparisonTest):
    def __init__(self):
        super().__init__("norm_l1", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(1000).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.norm(x, p=1)

    def foundry_operation(self):
        return "norm"

    def foundry_params(self):
        return {"p": 1}


class TestLogsumexp(ComparisonTest):
    def __init__(self):
        super().__init__("logsumexp", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(32, 64).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.logsumexp(x, dim=1)

    def foundry_params(self):
        return {"dim": 1}


class TestCumsum(ComparisonTest):
    def __init__(self):
        super().__init__("cumsum", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        return (np.random.randn(100).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.cumsum(x, dim=0)

    def foundry_params(self):
        return {"dim": 0}


class TestCumprod(ComparisonTest):
    def __init__(self):
        super().__init__("cumprod", tolerance=1e-4)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Use values close to 1 to avoid overflow/underflow
        return (np.random.uniform(0.95, 1.05, 50).astype(np.float32),)

    def pytorch_forward(self, x):
        return torch.cumprod(x, dim=0)

    def foundry_params(self):
        return {"dim": 0}


def get_all_tests():
    """Return all reduction tests."""
    return [
        TestSum(),
        TestSumDim(),
        TestMean(),
        TestMeanDim(),
        TestMax(),
        TestMin(),
        TestArgmax(),
        TestArgmaxDim(),
        TestArgmin(),
        TestStd(),
        TestVar(),
        TestProd(),
        TestNorm(),
        TestNormL1(),
        TestLogsumexp(),
        TestCumsum(),
        TestCumprod(),
    ]


if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv

    runner = TestRunner()
    for test in get_all_tests():
        runner.add_test(test)

    runner.run_all(debug=debug)
    runner.save_results()
