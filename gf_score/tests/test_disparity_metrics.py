"""
Tests for Disparity Metrics
============================
Validates RDI, NRGC, WCR, FP-GREAT computations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gf_score.core.disparity_metrics import (
    robustness_disparity_index,
    normalized_robustness_gini,
    worst_case_robustness,
    fairness_penalized_great,
    pairwise_disparity_matrix,
    class_vulnerability_ranking,
    compute_all_metrics,
    rdi_concentration_bound,
)
from gf_score.config import SQRT_PI_OVER_2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_scores():
    """Per-class scores that are all equal (perfect fairness)."""
    return {k: {"score": 0.3, "std": 0.01, "count": 100, "accuracy": 0.9,
                "local_scores": np.array([])} for k in range(10)}


@pytest.fixture
def disparate_scores():
    """Per-class scores with large disparity."""
    scores = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    return {k: {"score": scores[k], "std": 0.01, "count": 100, "accuracy": 0.7 + k * 0.02,
                "local_scores": np.array([])} for k in range(10)}


@pytest.fixture
def binary_scores():
    """Only two distinct score values (half high, half low)."""
    return {k: {"score": 0.8 if k < 5 else 0.2, "std": 0.01, "count": 100,
                "accuracy": 0.9, "local_scores": np.array([])} for k in range(10)}


@pytest.fixture
def single_outlier():
    """One class is much worse than all others."""
    return {k: {"score": 0.5 if k > 0 else 0.01, "std": 0.01, "count": 100,
                "accuracy": 0.9, "local_scores": np.array([])} for k in range(10)}


# ---------------------------------------------------------------------------
# RDI Tests
# ---------------------------------------------------------------------------

class TestRDI:
    def test_rdi_zero_for_uniform(self, uniform_scores):
        """RDI = 0 when all classes have equal robustness."""
        rdi = robustness_disparity_index(uniform_scores)
        assert abs(rdi) < 1e-10, f"RDI should be 0 for uniform scores, got {rdi}"

    def test_rdi_positive_for_non_uniform(self, disparate_scores):
        """RDI > 0 when classes have different robustness."""
        rdi = robustness_disparity_index(disparate_scores)
        assert rdi > 0, "RDI should be positive for non-uniform scores"

    def test_rdi_equals_max_minus_min(self, disparate_scores):
        """RDI = max - min by definition."""
        rdi = robustness_disparity_index(disparate_scores)
        scores = [disparate_scores[k]["score"] for k in range(10)]
        expected = max(scores) - min(scores)
        assert abs(rdi - expected) < 1e-10, f"RDI={rdi} != max-min={expected}"

    def test_rdi_bounded_by_sqrt_pi_over_2(self, disparate_scores):
        """RDI <= sqrt(pi/2) for valid GREAT Scores."""
        rdi = robustness_disparity_index(disparate_scores)
        assert rdi <= SQRT_PI_OVER_2 + 1e-6


# ---------------------------------------------------------------------------
# NRGC Tests
# ---------------------------------------------------------------------------

class TestNRGC:
    def test_nrgc_zero_for_uniform(self, uniform_scores):
        """Gini = 0 for perfect equality."""
        nrgc = normalized_robustness_gini(uniform_scores)
        assert abs(nrgc) < 1e-10, f"NRGC should be 0 for uniform scores, got {nrgc}"

    def test_nrgc_in_unit_interval(self, disparate_scores):
        """NRGC should be in [0, 1]."""
        nrgc = normalized_robustness_gini(disparate_scores)
        assert 0 <= nrgc <= 1.0 + 1e-6, f"NRGC={nrgc} out of [0,1]"

    def test_nrgc_increases_with_inequality(self, uniform_scores, disparate_scores):
        """More unequal distribution should have higher Gini."""
        nrgc_uniform = normalized_robustness_gini(uniform_scores)
        nrgc_disparate = normalized_robustness_gini(disparate_scores)
        assert nrgc_disparate > nrgc_uniform

    def test_nrgc_binary_case(self, binary_scores):
        """Known Gini for binary distribution."""
        nrgc = normalized_robustness_gini(binary_scores)
        # For scores [0.8]*5 + [0.2]*5, mean=0.5
        # Gini = sum|si-sj| / (2*K^2*mean)
        # Each same-group pair: diff=0, each cross-group pair: diff=0.6
        # Total cross-group pairs = 5*5*2 = 50, each diff=0.6 => total=30
        # Gini = 30 / (2*100*0.5) = 0.3
        assert abs(nrgc - 0.3) < 1e-6, f"Binary NRGC should be 0.3, got {nrgc}"


# ---------------------------------------------------------------------------
# WCR Tests
# ---------------------------------------------------------------------------

class TestWCR:
    def test_wcr_equals_min(self, disparate_scores):
        """WCR should equal the minimum per-class score."""
        wcr, idx = worst_case_robustness(disparate_scores)
        scores = [disparate_scores[k]["score"] for k in range(10)]
        assert abs(wcr - min(scores)) < 1e-10

    def test_wcr_returns_correct_class(self, single_outlier):
        """WCR should identify the correct worst class."""
        wcr, idx = worst_case_robustness(single_outlier)
        assert idx == 0, f"Worst class should be 0, got {idx}"
        assert abs(wcr - 0.01) < 1e-10

    def test_wcr_equals_score_for_uniform(self, uniform_scores):
        """When all classes are equal, WCR = common score."""
        wcr, idx = worst_case_robustness(uniform_scores)
        assert abs(wcr - 0.3) < 1e-10


# ---------------------------------------------------------------------------
# FP-GREAT Tests
# ---------------------------------------------------------------------------

class TestFPGREAT:
    def test_fp_great_equals_mean_when_lambda_zero(self, disparate_scores):
        """FP-GREAT(lambda=0) = mean GREAT Score."""
        fp = fairness_penalized_great(disparate_scores, lambda_val=0.0)
        scores = [disparate_scores[k]["score"] for k in range(10)]
        expected = np.mean(scores)
        assert abs(fp - expected) < 1e-10

    def test_fp_great_decreases_with_lambda(self, disparate_scores):
        """Higher lambda should reduce FP-GREAT (more penalty)."""
        fp_0 = fairness_penalized_great(disparate_scores, lambda_val=0.0)
        fp_05 = fairness_penalized_great(disparate_scores, lambda_val=0.5)
        fp_1 = fairness_penalized_great(disparate_scores, lambda_val=1.0)
        assert fp_0 >= fp_05 >= fp_1

    def test_fp_great_equals_mean_for_uniform(self, uniform_scores):
        """When RDI=0, FP-GREAT = mean for any lambda."""
        for lam in [0.0, 0.5, 1.0]:
            fp = fairness_penalized_great(uniform_scores, lambda_val=lam)
            assert abs(fp - 0.3) < 1e-10, f"FP-GREAT should be 0.3 at lambda={lam}"


# ---------------------------------------------------------------------------
# Pairwise Disparity Matrix Tests
# ---------------------------------------------------------------------------

class TestPairwiseMatrix:
    def test_matrix_is_symmetric(self, disparate_scores):
        D = pairwise_disparity_matrix(disparate_scores)
        assert np.allclose(D, D.T), "Disparity matrix must be symmetric"

    def test_diagonal_is_zero(self, disparate_scores):
        D = pairwise_disparity_matrix(disparate_scores)
        assert np.allclose(np.diag(D), 0), "Diagonal must be zero"

    def test_matrix_is_non_negative(self, disparate_scores):
        D = pairwise_disparity_matrix(disparate_scores)
        assert np.all(D >= -1e-10), "Disparity matrix must be non-negative"

    def test_max_element_equals_rdi(self, disparate_scores):
        """Maximum element of the disparity matrix should equal RDI."""
        D = pairwise_disparity_matrix(disparate_scores)
        rdi = robustness_disparity_index(disparate_scores)
        assert abs(D.max() - rdi) < 1e-10


# ---------------------------------------------------------------------------
# Vulnerability Ranking Tests
# ---------------------------------------------------------------------------

class TestVulnerabilityRanking:
    def test_ranking_sorted_ascending(self, disparate_scores):
        ranking = class_vulnerability_ranking(disparate_scores)
        scores = [s for _, s in ranking]
        assert scores == sorted(scores), "Ranking should be sorted ascending"

    def test_ranking_has_all_classes(self, disparate_scores):
        ranking = class_vulnerability_ranking(disparate_scores)
        assert len(ranking) == 10


# ---------------------------------------------------------------------------
# Concentration Bound Tests
# ---------------------------------------------------------------------------

class TestConcentrationBound:
    def test_basic_bound(self):
        n, tol = rdi_concentration_bound(epsilon=0.05, delta=0.05, num_classes=10)
        assert n > 0
        assert abs(tol - 0.10) < 1e-10, "RDI tolerance should be 2*epsilon"

    def test_more_classes_larger_sample(self):
        n1, _ = rdi_concentration_bound(epsilon=0.05, delta=0.05, num_classes=2)
        n2, _ = rdi_concentration_bound(epsilon=0.05, delta=0.05, num_classes=100)
        assert n2 > n1


# ---------------------------------------------------------------------------
# compute_all_metrics Integration Tests
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_returns_all_keys(self, disparate_scores):
        metrics = compute_all_metrics(disparate_scores)
        required_keys = ["rdi", "nrgc", "wcr", "fp_great", "vulnerability_ranking",
                         "disparity_matrix", "mean_score", "std_score"]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_internal_consistency(self, disparate_scores):
        """All metrics from compute_all_metrics should match individual calls."""
        metrics = compute_all_metrics(disparate_scores)
        assert abs(metrics["rdi"] - robustness_disparity_index(disparate_scores)) < 1e-10
        assert abs(metrics["nrgc"] - normalized_robustness_gini(disparate_scores)) < 1e-10
        wcr, _ = worst_case_robustness(disparate_scores)
        assert abs(metrics["wcr"] - wcr) < 1e-10
