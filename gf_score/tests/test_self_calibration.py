"""
Tests for Self-Calibration
============================
Validates attack-free calibration methods.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gf_score.core.self_calibration import SelfCalibrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calibrator():
    return SelfCalibrator(num_classes=10, use_sigmoid=True, device="cpu")


@pytest.fixture
def synthetic_model_data():
    """
    Create synthetic logits for 3 models with known class structure.
    
    Model A: Uniformly strong across all classes
    Model B: Strong on some classes, weak on others
    Model C: Overall weak
    """
    np.random.seed(42)
    n_per_class = 50
    n_classes = 10
    n_total = n_per_class * n_classes
    labels = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)

    def make_logits(class_strengths):
        logits = np.random.randn(n_total, n_classes).astype(np.float32) * 0.3
        for i in range(n_total):
            c = labels[i]
            logits[i, c] += class_strengths[c]
        return logits

    # Model A: all classes have strength 3.0
    logits_a = make_logits([3.0] * 10)

    # Model B: first 5 classes strong, last 5 weak
    logits_b = make_logits([4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Model C: all classes weak
    logits_c = make_logits([1.5] * 10)

    all_logits = {"model_a": logits_a, "model_b": logits_b, "model_c": logits_c}
    all_labels = {"model_a": labels, "model_b": labels, "model_c": labels}

    return all_logits, all_labels


# ---------------------------------------------------------------------------
# Accuracy Correlation Calibration Tests
# ---------------------------------------------------------------------------

class TestAccuracyCorrelation:
    def test_returns_valid_temperature(self, calibrator, synthetic_model_data):
        """Best temperature should be positive."""
        all_logits, all_labels = synthetic_model_data
        result = calibrator.calibrate_accuracy_correlation(
            all_logits, all_labels,
            temp_min=0.1, temp_max=3.0, temp_step=0.2,
            fine_step=0.05,
        )
        assert result["best_temperature"] > 0, "Temperature must be positive"

    def test_returns_valid_correlation(self, calibrator, synthetic_model_data):
        """Best correlation should be in [-1, 1]."""
        all_logits, all_labels = synthetic_model_data
        result = calibrator.calibrate_accuracy_correlation(
            all_logits, all_labels,
            temp_min=0.5, temp_max=2.0, temp_step=0.3,
            fine_step=0.1,
        )
        assert -1.0 <= result["best_correlation"] <= 1.0

    def test_has_per_model_results(self, calibrator, synthetic_model_data):
        """Should produce results for each model."""
        all_logits, all_labels = synthetic_model_data
        result = calibrator.calibrate_accuracy_correlation(
            all_logits, all_labels,
            temp_min=0.5, temp_max=2.0, temp_step=0.5,
            fine_step=0.1,
        )
        assert "per_model_results" in result
        assert len(result["per_model_results"]) == 3

    def test_search_history_recorded(self, calibrator, synthetic_model_data):
        """Search history should be non-empty."""
        all_logits, all_labels = synthetic_model_data
        result = calibrator.calibrate_accuracy_correlation(
            all_logits, all_labels,
            temp_min=0.5, temp_max=2.0, temp_step=0.5,
            fine_step=0.1,
        )
        assert len(result["search_history"]) > 0

    def test_method_identified(self, calibrator, synthetic_model_data):
        all_logits, all_labels = synthetic_model_data
        result = calibrator.calibrate_accuracy_correlation(
            all_logits, all_labels,
            temp_min=0.5, temp_max=2.0, temp_step=0.5, fine_step=0.1,
        )
        assert result["method"] == "accuracy_correlation"


# ---------------------------------------------------------------------------
# Ranking Stability Calibration Tests
# ---------------------------------------------------------------------------

class TestRankingStability:
    def test_returns_valid_temperature(self, calibrator, synthetic_model_data):
        all_logits, all_labels = synthetic_model_data
        result = calibrator.calibrate_ranking_stability(
            all_logits, all_labels,
            temp_min=0.5, temp_max=2.0, temp_step=0.3,
        )
        assert result["best_temperature"] > 0

    def test_stability_in_valid_range(self, calibrator, synthetic_model_data):
        all_logits, all_labels = synthetic_model_data
        result = calibrator.calibrate_ranking_stability(
            all_logits, all_labels,
            temp_min=0.5, temp_max=2.0, temp_step=0.3,
        )
        assert -1.0 <= result["best_stability"] <= 1.0

    def test_method_identified(self, calibrator, synthetic_model_data):
        all_logits, all_labels = synthetic_model_data
        result = calibrator.calibrate_ranking_stability(
            all_logits, all_labels,
            temp_min=0.5, temp_max=2.0, temp_step=0.5,
        )
        assert result["method"] == "ranking_stability"


# ---------------------------------------------------------------------------
# JSON Serialization Tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_make_serializable_numpy(self, calibrator):
        """numpy types should be converted to Python types."""
        obj = {
            "float": np.float64(1.5),
            "int": np.int64(42),
            "array": np.array([1, 2, 3]),
            "nested": {"val": np.float32(0.5)},
        }
        result = calibrator._make_serializable(obj)
        assert isinstance(result["float"], float)
        assert isinstance(result["int"], int)
        assert isinstance(result["array"], list)
        assert isinstance(result["nested"]["val"], float)
