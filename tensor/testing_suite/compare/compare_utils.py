# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
Comparison utilities for PyTorch vs Foundry testing.

This module provides the infrastructure for comparing PyTorch and Foundry outputs.
"""

import numpy as np
import subprocess
import struct
import os
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Callable
import sys

# Paths
SUITE_DIR = Path(__file__).parent.parent
DATA_DIR = SUITE_DIR / "data"
RESULTS_DIR = SUITE_DIR / "results"
COMPARE_HELPER = SUITE_DIR.parent.parent / "bin" / "compare_helper"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Data type mappings
DTYPE_MAP = {
    'float32': 0,
    'float64': 1,
    'int32': 2,
    'int64': 3,
}

NUMPY_DTYPE_MAP = {
    0: np.float32,
    1: np.float64,
    2: np.int32,
    3: np.int64,
}


@dataclass
class CompareResult:
    """Result of a comparison test."""
    name: str
    passed: bool
    max_diff: float
    mean_diff: float
    tolerance: float
    input_shape: List[int]
    time_pytorch_ms: float
    time_foundry_ms: float
    error_message: Optional[str] = None


def write_tensor(filepath: Path, data: np.ndarray):
    """
    Write a tensor to binary file in Foundry format.

    Format:
        - ndim: int32 (4 bytes)
        - dtype: int32 (4 bytes)
        - reserved: int64 (8 bytes)
        - shape: ndim * int64
        - data: numel * dtype
    """
    ndim = len(data.shape)
    dtype_code = DTYPE_MAP.get(str(data.dtype), 0)

    with open(filepath, 'wb') as f:
        # Header
        f.write(struct.pack('<i', ndim))
        f.write(struct.pack('<i', dtype_code))
        f.write(struct.pack('<q', 0))  # reserved

        # Shape
        for dim in data.shape:
            f.write(struct.pack('<q', dim))

        # Data (ensure contiguous)
        f.write(np.ascontiguousarray(data).tobytes())


def read_tensor(filepath: Path) -> np.ndarray:
    """
    Read a tensor from binary file in Foundry format.
    """
    with open(filepath, 'rb') as f:
        # Header
        ndim = struct.unpack('<i', f.read(4))[0]
        dtype_code = struct.unpack('<i', f.read(4))[0]
        _ = struct.unpack('<q', f.read(8))[0]  # reserved

        # Shape
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('<q', f.read(8))[0])

        # Data
        dtype = NUMPY_DTYPE_MAP.get(dtype_code, np.float32)
        numel = int(np.prod(shape)) if shape else 1
        data = np.frombuffer(f.read(numel * np.dtype(dtype).itemsize), dtype=dtype)

        return data.reshape(shape) if shape else data


def run_foundry(operation: str, inputs: List[str], output: str,
                params: Optional[dict] = None) -> Tuple[bool, float]:
    """
    Run a Foundry operation via the comparison helper.

    Args:
        operation: Name of the operation (e.g., "mish", "matmul")
        inputs: List of input file paths
        output: Output file path
        params: Optional parameters (e.g., {"alpha": 1.0})

    Returns:
        Tuple of (success, time_ms)
    """
    if not COMPARE_HELPER.exists():
        raise FileNotFoundError(
            f"Compare helper not found at {COMPARE_HELPER}. "
            "Run 'make compare-helper' first."
        )

    cmd = [str(COMPARE_HELPER), operation, output] + inputs

    if params:
        for key, value in params.items():
            cmd.extend([f"--{key}", str(value)])

    import time
    start = time.perf_counter()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        elapsed = (time.perf_counter() - start) * 1000

        if result.returncode != 0:
            print(f"Foundry error: {result.stderr}", file=sys.stderr)
            return False, elapsed

        return True, elapsed

    except subprocess.TimeoutExpired:
        return False, 30000.0
    except Exception as e:
        print(f"Error running Foundry: {e}", file=sys.stderr)
        return False, 0.0


def compare_tensors(expected: np.ndarray, actual: np.ndarray,
                    tolerance: float) -> Tuple[bool, float, float]:
    """
    Compare two tensors for numerical equivalence.

    Returns:
        Tuple of (passed, max_diff, mean_diff)
    """
    # Squeeze both to handle scalar shape differences: () vs (1,)
    expected = np.squeeze(expected)
    actual = np.squeeze(actual)

    if expected.shape != actual.shape:
        return False, float('inf'), float('inf')

    diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    passed = max_diff <= tolerance

    return passed, max_diff, mean_diff


class ComparisonTest:
    """Base class for comparison tests."""

    def __init__(self, name: str, tolerance: float = 1e-5):
        self.name = name
        self.tolerance = tolerance
        self.results: List[CompareResult] = []

    def generate_input(self, seed: int = 42) -> Tuple[np.ndarray, ...]:
        """Generate input tensors. Override in subclass."""
        raise NotImplementedError

    def pytorch_forward(self, *inputs) -> np.ndarray:
        """Run PyTorch forward. Override in subclass."""
        raise NotImplementedError

    def foundry_operation(self) -> str:
        """Return the Foundry operation name."""
        return self.name

    def foundry_params(self) -> Optional[dict]:
        """Return any extra parameters for Foundry."""
        return None

    def run(self, seed: int = 42, debug: bool = False) -> CompareResult:
        """Run the comparison test."""
        import time

        try:
            # Generate inputs
            inputs = self.generate_input(seed)

            # Run PyTorch
            import torch
            torch_inputs = [torch.from_numpy(x) for x in inputs]

            start = time.perf_counter()
            expected = self.pytorch_forward(*torch_inputs)
            if isinstance(expected, torch.Tensor):
                expected = expected.detach().numpy()
            pytorch_time = (time.perf_counter() - start) * 1000

            # Write inputs to files
            input_files = []
            for i, inp in enumerate(inputs):
                filepath = DATA_DIR / f"{self.name}_input_{i}.bin"
                write_tensor(filepath, inp)
                input_files.append(str(filepath))

            # Run Foundry
            output_file = DATA_DIR / f"{self.name}_output.bin"
            success, foundry_time = run_foundry(
                self.foundry_operation(),
                input_files,
                str(output_file),
                self.foundry_params()
            )

            if not success:
                return CompareResult(
                    name=self.name,
                    passed=False,
                    max_diff=float('inf'),
                    mean_diff=float('inf'),
                    tolerance=self.tolerance,
                    input_shape=list(inputs[0].shape),
                    time_pytorch_ms=pytorch_time,
                    time_foundry_ms=foundry_time,
                    error_message="Foundry execution failed"
                )

            # Read Foundry output
            actual = read_tensor(output_file)

            # Compare
            passed, max_diff, mean_diff = compare_tensors(
                expected, actual, self.tolerance
            )

            # Debug output
            if debug:
                write_tensor(DATA_DIR / f"{self.name}_expected.bin", expected)
                with open(DATA_DIR / f"{self.name}_diff.txt", 'w') as f:
                    f.write(f"Max diff: {max_diff}\n")
                    f.write(f"Mean diff: {mean_diff}\n")
                    f.write(f"Tolerance: {self.tolerance}\n")
                    f.write(f"Passed: {passed}\n")
                    if not passed:
                        diff = np.abs(expected - actual)
                        idx = np.unravel_index(np.argmax(diff), diff.shape)
                        f.write(f"Max diff at index {idx}\n")
                        f.write(f"Expected: {expected[idx]}\n")
                        f.write(f"Actual: {actual[idx]}\n")

            result = CompareResult(
                name=self.name,
                passed=passed,
                max_diff=max_diff,
                mean_diff=mean_diff,
                tolerance=self.tolerance,
                input_shape=list(inputs[0].shape),
                time_pytorch_ms=pytorch_time,
                time_foundry_ms=foundry_time,
                error_message=None if passed else f"max_diff {max_diff} > tolerance {self.tolerance}"
            )

            self.results.append(result)
            return result

        except Exception as e:
            return CompareResult(
                name=self.name,
                passed=False,
                max_diff=float('inf'),
                mean_diff=float('inf'),
                tolerance=self.tolerance,
                input_shape=[],
                time_pytorch_ms=0.0,
                time_foundry_ms=0.0,
                error_message=str(e)
            )


class TestRunner:
    """Run multiple comparison tests and report results."""

    def __init__(self):
        self.tests: List[ComparisonTest] = []
        self.results: List[CompareResult] = []

    def add_test(self, test: ComparisonTest):
        self.tests.append(test)

    def run_all(self, debug: bool = False) -> List[CompareResult]:
        """Run all tests and return results."""
        self.results = []

        print("=" * 80)
        print("FOUNDRY vs PYTORCH COMPARISON TESTS")
        print("=" * 80)
        print()

        for test in self.tests:
            result = test.run(debug=debug)
            self.results.append(result)

            status = "\033[92m[PASS]\033[0m" if result.passed else "\033[91m[FAIL]\033[0m"
            dots = "." * (40 - len(test.name))
            print(f"  {status} {test.name} {dots} max_diff={result.max_diff:.6f}, mean_diff={result.mean_diff:.6f}")

            if not result.passed and result.error_message:
                print(f"         {result.error_message}")

        print()
        print("=" * 80)
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        print(f"SUMMARY: {passed}/{len(self.results)} passed, {failed} failed")
        print("=" * 80)

        return self.results

    def save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = RESULTS_DIR / f"{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {filepath}")


def print_summary(results: List[CompareResult]):
    """Print a summary of test results."""
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total:  {len(results)}")
    print(f"Passed: {len(passed)} ({100*len(passed)/len(results):.1f}%)")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  - {r.name}: max_diff={r.max_diff:.6e} (tolerance={r.tolerance:.0e})")
            if r.error_message:
                print(f"    {r.error_message}")
