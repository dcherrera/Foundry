#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
Run all Foundry vs PyTorch comparison tests.

Usage:
    python run_all.py                   # Run all tests
    python run_all.py --category activations  # Run specific category
    python run_all.py --debug           # Save debug files
    python run_all.py --quick           # Run subset of fast tests
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add compare directory to path
sys.path.insert(0, str(Path(__file__).parent / "compare"))

from compare_utils import TestRunner, print_summary


def get_all_tests(category=None):
    """Get all tests, optionally filtered by category."""
    tests = []

    categories = {
        'activations': 'test_activations',
        'elementwise': 'test_elementwise',
        'matmul': 'test_matmul',
        'loss': 'test_loss',
        'reductions': 'test_reductions',
    }

    if category:
        if category not in categories:
            print(f"Unknown category: {category}")
            print(f"Available categories: {', '.join(categories.keys())}")
            sys.exit(1)
        categories = {category: categories[category]}

    for cat_name, module_name in categories.items():
        try:
            module = __import__(module_name)
            cat_tests = module.get_all_tests()
            tests.extend(cat_tests)
            print(f"Loaded {len(cat_tests)} tests from {cat_name}")
        except ImportError as e:
            print(f"Warning: Could not load {module_name}: {e}")
        except AttributeError as e:
            print(f"Warning: {module_name} has no get_all_tests(): {e}")

    return tests


def get_quick_tests():
    """Get a subset of fast tests for quick verification."""
    tests = []

    # Just a few tests from each category
    try:
        from test_activations import TestReLU, TestSigmoid, TestSoftmax
        tests.extend([TestReLU(), TestSigmoid(), TestSoftmax()])
    except ImportError as e:
        print(f"Warning: could not load quick tests: {e}")

    try:
        from test_elementwise import TestAdd, TestMul, TestExp
        tests.extend([TestAdd(), TestMul(), TestExp()])
    except ImportError as e:
        print(f"Warning: could not load quick tests: {e}")

    try:
        from test_matmul import TestMatmul, TestLinear
        tests.extend([TestMatmul(), TestLinear()])
    except ImportError as e:
        print(f"Warning: could not load quick tests: {e}")

    try:
        from test_loss import TestMSELoss, TestCrossEntropyLoss
        tests.extend([TestMSELoss(), TestCrossEntropyLoss()])
    except ImportError as e:
        print(f"Warning: could not load quick tests: {e}")

    try:
        from test_reductions import TestSum, TestMean
        tests.extend([TestSum(), TestMean()])
    except ImportError as e:
        print(f"Warning: could not load quick tests: {e}")

    return tests


def main():
    parser = argparse.ArgumentParser(
        description='Run Foundry vs PyTorch comparison tests'
    )
    parser.add_argument(
        '--category', '-c',
        help='Run only tests from this category'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Save debug files for failed tests'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick subset of tests'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("FOUNDRY vs PYTORCH COMPARISON TEST SUITE")
    print("=" * 80)
    print()
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check for compare_helper
    suite_dir = Path(__file__).parent
    helper_path = suite_dir.parent.parent / "bin" / "compare_helper"
    if not helper_path.exists():
        print("ERROR: compare_helper not found!")
        print(f"Expected at: {helper_path}")
        print()
        print("Build it with:")
        print("  make compare-helper")
        print()
        sys.exit(1)

    print(f"Compare helper: {helper_path}")
    print()

    # Get tests
    if args.quick:
        print("Running QUICK test suite...")
        tests = get_quick_tests()
    else:
        tests = get_all_tests(args.category)

    if not tests:
        print("No tests found!")
        sys.exit(1)

    print(f"Total tests to run: {len(tests)}")
    print()

    # Run tests
    runner = TestRunner()
    for test in tests:
        runner.add_test(test)

    results = runner.run_all(debug=args.debug)

    # Summary
    print()
    print_summary(results)

    # Save results
    runner.save_results()

    # Exit code
    failed = sum(1 for r in results if not r.passed)
    if failed > 0:
        print(f"\n{failed} test(s) failed.")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
