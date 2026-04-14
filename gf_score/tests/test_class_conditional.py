"""
Tests for Class-Conditional GREAT Score
========================================
Validates per-class score computation, decomposition consistency,
and sample complexity formulas.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gf_score.core.class_conditional_great import (
    ClassConditionalGREAT,
    compute_sample_complexity_bound,
)
from gf_score.config import SQRT_PI_OVER_2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """GREAT Score engine with default settings (T=1, sigmoid, cpu)."""
    return ClassConditionalGREAT(
        num_classes=10,
        temperature=1.0,
        use_sigmoid=True,
        batch_size=64,
        device="cpu",
    )


@pytest.fixture
def synthetic_logits():
    """
    Create synthetic logits that give predictable GREAT Scores.
    
    Setup: 100 samples, 10 classes, with class labels [0..9] repeated 10 times.
    Logits are designed so the true class has the highest value.
    """
    np.random.seed(42)
    n_samples = 100
    n_classes = 10
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)

    # Create logits where true class is dominant
    logits = np.random.randn(n_samples, n_classes).astype(np.float32) * 0.5
    for i in range(n_samples):
        logits[i, labels[i]] += 3.0  # Make true class dominant

    return logits, labels


@pytest.fixture
def uniform_logits():
    """Logits where all classes have equal confidence (GREAT Score = 0)."""
    n_samples = 50
    n_classes = 10
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    return logits, labels


@pytest.fixture
def perfect_logits():
    """Logits where model is perfectly confident in the correct class."""
    n_samples = 50
    n_classes = 10
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    logits = np.zeros((n_samples, n_classes), dtype=np.float32) - 10.0
    for i in range(n_samples):
        logits[i, labels[i]] = 10.0  # Very high confidence in true class
    return logits, labels


# ---------------------------------------------------------------------------
# Local Score Tests
# ---------------------------------------------------------------------------

class TestLocalScores:
    def test_scores_are_non_negative(self, engine, synthetic_logits):
        """GREAT local scores must be >= 0 (from max(gap, 0) formulation)."""
        logits, labels = synthetic_logits
        scores = engine.compute_local_scores(logits, labels)
        assert np.all(scores >= 0), "Local scores must be non-negative"

    def test_scores_bounded_above(self, engine, synthetic_logits):
        """For sigmoid outputs in [0,1], gap is at most 1, so score <= sqrt(pi/2)."""
        logits, labels = synthetic_logits
        scores = engine.compute_local_scores(logits, labels)
        assert np.all(scores <= SQRT_PI_OVER_2 + 1e-6), \
            f"Scores exceed sqrt(pi/2)={SQRT_PI_OVER_2:.4f}"

    def test_uniform_logits_give_zero_scores(self, engine, uniform_logits):
        """Uniform logits → sigmoid(0)=0.5 for all → gap=0 → score=0."""
        logits, labels = uniform_logits
        scores = engine.compute_local_scores(logits, labels)
        assert np.allclose(scores, 0.0, atol=1e-6), \
            "Uniform logits should give zero GREAT Scores"

    def test_confident_predictions_give_high_scores(self, engine, perfect_logits):
        """Very confident predictions should yield high GREAT Scores."""
        logits, labels = perfect_logits
        scores = engine.compute_local_scores(logits, labels)
        assert np.all(scores > 0.5), "Very confident predictions should have high scores"

    def test_wrong_predictions_give_zero_scores(self, engine):
        """If model predicts wrong class, score should be 0."""
        logits = np.array([[-5.0, 10.0, -5.0] + [-5.0] * 7], dtype=np.float32)
        labels = np.array([0])  # True class is 0 but model predicts class 1
        scores = engine.compute_local_scores(logits, labels)
        assert scores[0] == 0.0, "Wrong prediction should give score = 0"


# ---------------------------------------------------------------------------
# Per-Class Score Tests
# ---------------------------------------------------------------------------

class TestPerClassScores:
    def test_all_classes_present(self, engine, synthetic_logits):
        """Each class should have a score entry."""
        logits, labels = synthetic_logits
        per_class = engine.compute_per_class_scores(logits, labels)
        assert len(per_class) == 10, "Should have exactly 10 class entries"
        for k in range(10):
            assert k in per_class, f"Missing class {k}"

    def test_counts_sum_to_total(self, engine, synthetic_logits):
        """Sum of per-class sample counts should equal total samples."""
        logits, labels = synthetic_logits
        per_class = engine.compute_per_class_scores(logits, labels)
        total = sum(pc["count"] for pc in per_class.values())
        assert total == len(labels), "Per-class counts must sum to total"

    def test_per_class_scores_non_negative(self, engine, synthetic_logits):
        """Per-class scores should be non-negative."""
        logits, labels = synthetic_logits
        per_class = engine.compute_per_class_scores(logits, labels)
        for k, pc in per_class.items():
            assert pc["score"] >= 0, f"Class {k} score must be non-negative"

    def test_accuracy_in_valid_range(self, engine, synthetic_logits):
        """Per-class accuracy should be in [0, 1]."""
        logits, labels = synthetic_logits
        per_class = engine.compute_per_class_scores(logits, labels)
        for k, pc in per_class.items():
            assert 0 <= pc["accuracy"] <= 1.0, f"Class {k} accuracy out of range"


# ---------------------------------------------------------------------------
# Decomposition Consistency Tests
# ---------------------------------------------------------------------------

class TestDecompositionConsistency:
    def test_aggregate_matches_direct(self, engine, synthetic_logits):
        """
        CRITICAL TEST: The per-class decomposition must recover the
        original aggregate GREAT Score exactly:
            sum_k (n_k/N) * Omega_hat_k = Omega_hat(f)
        """
        logits, labels = synthetic_logits
        per_class = engine.compute_per_class_scores(logits, labels)
        agg_decomposed = engine.compute_aggregate_score(per_class)
        agg_direct = engine.compute_aggregate_from_local(logits, labels)

        assert abs(agg_decomposed - agg_direct) < 1e-10, (
            f"Decomposition inconsistency: "
            f"decomposed={agg_decomposed:.10f}, direct={agg_direct:.10f}"
        )

    def test_aggregate_matches_direct_uniform(self, engine, uniform_logits):
        logits, labels = uniform_logits
        per_class = engine.compute_per_class_scores(logits, labels)
        agg_decomposed = engine.compute_aggregate_score(per_class)
        agg_direct = engine.compute_aggregate_from_local(logits, labels)
        assert abs(agg_decomposed - agg_direct) < 1e-10

    def test_aggregate_matches_direct_perfect(self, engine, perfect_logits):
        logits, labels = perfect_logits
        per_class = engine.compute_per_class_scores(logits, labels)
        agg_decomposed = engine.compute_aggregate_score(per_class)
        agg_direct = engine.compute_aggregate_from_local(logits, labels)
        assert abs(agg_decomposed - agg_direct) < 1e-10

    def test_temperature_does_not_break_decomposition(self, synthetic_logits):
        """Decomposition should hold for any temperature value."""
        logits, labels = synthetic_logits
        for temp in [0.1, 0.5, 1.0, 2.0, 5.0]:
            engine = ClassConditionalGREAT(
                num_classes=10, temperature=temp, use_sigmoid=True, device="cpu"
            )
            pc = engine.compute_per_class_scores(logits, labels)
            agg_d = engine.compute_aggregate_score(pc)
            agg_r = engine.compute_aggregate_from_local(logits, labels)
            assert abs(agg_d - agg_r) < 1e-10, f"Decomposition fails at T={temp}"


# ---------------------------------------------------------------------------
# Sample Complexity Tests
# ---------------------------------------------------------------------------

class TestSampleComplexity:
    def test_basic_bound(self):
        """Check sample complexity for standard parameters."""
        n = compute_sample_complexity_bound(epsilon=0.05, delta=0.05, num_classes=10)
        assert n > 0, "Sample complexity must be positive"
        assert n >= 100, "Must require at least ~100 samples for eps=0.05"

    def test_smaller_epsilon_needs_more_samples(self):
        """Tighter tolerance requires more samples."""
        n1 = compute_sample_complexity_bound(epsilon=0.1, delta=0.05, num_classes=10)
        n2 = compute_sample_complexity_bound(epsilon=0.05, delta=0.05, num_classes=10)
        assert n2 > n1, "Smaller epsilon should require more samples"

    def test_more_classes_needs_more_samples(self):
        """Union bound correction: more classes = more samples needed."""
        n1 = compute_sample_complexity_bound(epsilon=0.05, delta=0.05, num_classes=2)
        n2 = compute_sample_complexity_bound(epsilon=0.05, delta=0.05, num_classes=100)
        assert n2 > n1, "More classes should require more samples (union bound)"

    def test_smaller_delta_needs_more_samples(self):
        """Higher confidence requires more samples."""
        n1 = compute_sample_complexity_bound(epsilon=0.05, delta=0.1, num_classes=10)
        n2 = compute_sample_complexity_bound(epsilon=0.05, delta=0.01, num_classes=10)
        assert n2 > n1, "Smaller delta should require more samples"


# ---------------------------------------------------------------------------
# Robustness Profile Tests
# ---------------------------------------------------------------------------

class TestRobustnessProfile:
    def test_profile_contains_required_keys(self, engine, synthetic_logits):
        logits, labels = synthetic_logits
        pc = engine.compute_per_class_scores(logits, labels)
        profile = engine.get_robustness_profile(pc)
        required = [
            "per_class_scores", "aggregate_score", "worst_class",
            "worst_score", "best_class", "best_score", "score_range",
        ]
        for key in required:
            assert key in profile, f"Profile missing key: {key}"

    def test_worst_leq_best(self, engine, synthetic_logits):
        logits, labels = synthetic_logits
        pc = engine.compute_per_class_scores(logits, labels)
        profile = engine.get_robustness_profile(pc)
        assert profile["worst_score"] <= profile["best_score"]

    def test_score_range_consistent(self, engine, synthetic_logits):
        logits, labels = synthetic_logits
        pc = engine.compute_per_class_scores(logits, labels)
        profile = engine.get_robustness_profile(pc)
        expected_range = profile["best_score"] - profile["worst_score"]
        assert abs(profile["score_range"] - expected_range) < 1e-10
