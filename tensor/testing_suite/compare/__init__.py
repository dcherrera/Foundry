# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
Comparison test modules for Foundry vs PyTorch verification.
"""

from .compare_utils import (
    ComparisonTest,
    TestRunner,
    CompareResult,
    write_tensor,
    read_tensor,
    run_foundry,
    compare_tensors,
    print_summary,
)

__all__ = [
    'ComparisonTest',
    'TestRunner',
    'CompareResult',
    'write_tensor',
    'read_tensor',
    'run_foundry',
    'compare_tensors',
    'print_summary',
]
