# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""
Comparison tests for loss functions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from compare_utils import ComparisonTest, TestRunner


class TestMSELoss(ComparisonTest):
    def __init__(self):
        super().__init__("mse_loss", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        pred = np.random.randn(32, 10).astype(np.float32)
        target = np.random.randn(32, 10).astype(np.float32)
        return (pred, target)

    def pytorch_forward(self, pred, target):
        return F.mse_loss(pred, target, reduction='mean')

    def foundry_params(self):
        return {"reduction": "mean"}


class TestL1Loss(ComparisonTest):
    def __init__(self):
        super().__init__("l1_loss", tolerance=1e-6)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        pred = np.random.randn(32, 10).astype(np.float32)
        target = np.random.randn(32, 10).astype(np.float32)
        return (pred, target)

    def pytorch_forward(self, pred, target):
        return F.l1_loss(pred, target, reduction='mean')

    def foundry_params(self):
        return {"reduction": "mean"}


class TestSmoothL1Loss(ComparisonTest):
    def __init__(self):
        super().__init__("smooth_l1_loss", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        pred = np.random.randn(32, 10).astype(np.float32)
        target = np.random.randn(32, 10).astype(np.float32)
        return (pred, target)

    def pytorch_forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, reduction='mean', beta=1.0)

    def foundry_params(self):
        return {"reduction": "mean", "beta": 1.0}


class TestCrossEntropyLoss(ComparisonTest):
    def __init__(self):
        super().__init__("cross_entropy", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Logits (not probabilities)
        logits = np.random.randn(32, 10).astype(np.float32)
        # Class indices
        targets = np.random.randint(0, 10, 32).astype(np.int64)
        return (logits, targets)

    def pytorch_forward(self, logits, targets):
        return F.cross_entropy(logits, targets, reduction='mean')

    def foundry_params(self):
        return {"reduction": "mean"}


class TestNLLLoss(ComparisonTest):
    def __init__(self):
        super().__init__("nll_loss", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Log probabilities
        logits = np.random.randn(32, 10).astype(np.float32)
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        # Class indices
        targets = np.random.randint(0, 10, 32).astype(np.int64)
        return (log_probs.astype(np.float32), targets)

    def pytorch_forward(self, log_probs, targets):
        return F.nll_loss(log_probs, targets, reduction='mean')

    def foundry_params(self):
        return {"reduction": "mean"}


class TestBinaryCrossEntropyWithLogits(ComparisonTest):
    def __init__(self):
        super().__init__("binary_cross_entropy_with_logits", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        logits = np.random.randn(32, 10).astype(np.float32)
        # Binary targets
        targets = np.random.randint(0, 2, (32, 10)).astype(np.float32)
        return (logits, targets)

    def pytorch_forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')

    def foundry_params(self):
        return {"reduction": "mean"}


class TestKLDivLoss(ComparisonTest):
    def __init__(self):
        super().__init__("kl_div", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        # Log probabilities for input
        logits = np.random.randn(32, 10).astype(np.float32)
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        # Probabilities for target
        target_logits = np.random.randn(32, 10).astype(np.float32)
        target_probs = np.exp(target_logits) / np.sum(np.exp(target_logits), axis=1, keepdims=True)
        return (log_probs.astype(np.float32), target_probs.astype(np.float32))

    def pytorch_forward(self, log_probs, target):
        return F.kl_div(log_probs, target, reduction='batchmean')

    def foundry_params(self):
        return {"reduction": "batchmean"}


class TestHuberLoss(ComparisonTest):
    def __init__(self):
        super().__init__("huber_loss", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        pred = np.random.randn(32, 10).astype(np.float32)
        target = np.random.randn(32, 10).astype(np.float32)
        return (pred, target)

    def pytorch_forward(self, pred, target):
        return F.huber_loss(pred, target, reduction='mean', delta=1.0)

    def foundry_params(self):
        return {"reduction": "mean", "delta": 1.0}


class TestCosineEmbeddingLoss(ComparisonTest):
    def __init__(self):
        super().__init__("cosine_embedding_loss", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        x1 = np.random.randn(32, 64).astype(np.float32)
        x2 = np.random.randn(32, 64).astype(np.float32)
        # Labels: 1 for similar, -1 for dissimilar
        y = np.random.choice([-1, 1], 32).astype(np.int64)
        return (x1, x2, y)

    def pytorch_forward(self, x1, x2, y):
        return F.cosine_embedding_loss(x1, x2, torch.tensor(y), reduction='mean')

    def foundry_params(self):
        return {"reduction": "mean", "margin": 0.0}


class TestTripletMarginLoss(ComparisonTest):
    def __init__(self):
        super().__init__("triplet_margin_loss", tolerance=1e-5)

    def generate_input(self, seed=42):
        np.random.seed(seed)
        anchor = np.random.randn(32, 64).astype(np.float32)
        positive = anchor + np.random.randn(32, 64).astype(np.float32) * 0.1
        negative = np.random.randn(32, 64).astype(np.float32)
        return (anchor, positive.astype(np.float32), negative)

    def pytorch_forward(self, anchor, positive, negative):
        return F.triplet_margin_loss(anchor, positive, negative, margin=1.0, reduction='mean')

    def foundry_params(self):
        return {"reduction": "mean", "margin": 1.0}


def get_all_tests():
    """Return all loss function tests."""
    return [
        TestMSELoss(),
        TestL1Loss(),
        TestSmoothL1Loss(),
        TestCrossEntropyLoss(),
        TestNLLLoss(),
        TestBinaryCrossEntropyWithLogits(),
        TestKLDivLoss(),
        TestHuberLoss(),
        TestCosineEmbeddingLoss(),
        TestTripletMarginLoss(),
    ]


if __name__ == "__main__":
    import sys
    debug = "--debug" in sys.argv

    runner = TestRunner()
    for test in get_all_tests():
        runner.add_test(test)

    runner.run_all(debug=debug)
    runner.save_results()
