"""
Attack-Free Self-Calibration
============================
Calibrates GREAT Score temperature parameter WITHOUT requiring
adversarial attack results, addressing Shortcoming 3 of the original paper.

Two approaches:
1. Accuracy Correlation: Optimize T to maximize Spearman correlation
   between per-class GREAT Scores and per-class clean accuracies.
2. Ranking Stability: Optimize T for maximum ranking stability under
   small temperature perturbations.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from gf_score.config import (
    CALIBRATION_TEMP_MIN,
    CALIBRATION_TEMP_MAX,
    CALIBRATION_TEMP_STEP,
    CALIBRATION_TEMP_FINE_STEP,
    CIFAR10_NUM_CLASSES,
    RESULTS_DIR,
)
from gf_score.core.class_conditional_great import ClassConditionalGREAT

logger = logging.getLogger("gf_score.core.self_calibration")


class SelfCalibrator:
    """
    Attack-free temperature calibration for GREAT Score.

    Replaces the CW-attack-based calibration in Section 3.5 of the
    original paper with a fully attack-free approach.
    """

    def __init__(
        self,
        num_classes: int = CIFAR10_NUM_CLASSES,
        use_sigmoid: bool = True,
        device: str = "cuda",
    ):
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid
        self.device = device

    def calibrate_accuracy_correlation(
        self,
        all_logits: Dict[str, np.ndarray],
        all_labels: Dict[str, np.ndarray],
        temp_min: float = CALIBRATION_TEMP_MIN,
        temp_max: float = CALIBRATION_TEMP_MAX,
        temp_step: float = CALIBRATION_TEMP_STEP,
        fine_step: float = CALIBRATION_TEMP_FINE_STEP,
    ) -> dict:
        """
        Calibrate temperature via per-class accuracy correlation.

        Method: For each model, find T that maximizes Spearman correlation
        between per-class GREAT Scores and per-class clean accuracies.

        Two-phase grid search:
        1. Coarse search over [temp_min, temp_max] with step temp_step
        2. Fine search around the best coarse T with step fine_step

        Args:
            all_logits: dict mapping model_name -> (N, K) logits array.
            all_labels: dict mapping model_name -> (N,) labels array.
            temp_min, temp_max, temp_step: coarse grid parameters.
            fine_step: fine grid step size.

        Returns:
            dict with {
                'best_temperature': float,
                'best_correlation': float,
                'per_model_results': dict,
                'search_history': list of (temp, correlation) pairs,
                'method': str,
            }
        """
        logger.info("Running self-calibration via accuracy correlation...")
        logger.info(f"  Temperature range: [{temp_min}, {temp_max}], "
                    f"coarse step: {temp_step}, fine step: {fine_step}")

        model_names = sorted(all_logits.keys())
        num_models = len(model_names)

        # Phase 1: Coarse grid search
        coarse_temps = np.arange(temp_min, temp_max + temp_step, temp_step)
        search_history = []
        best_coarse_temp = 1.0
        best_coarse_corr = -1.0

        logger.info(f"  Phase 1: Coarse search over {len(coarse_temps)} temperatures...")
        for temp in coarse_temps:
            avg_corr = self._evaluate_temperature_accuracy(
                all_logits, all_labels, model_names, temp
            )
            search_history.append((float(temp), float(avg_corr)))

            if avg_corr > best_coarse_corr:
                best_coarse_corr = avg_corr
                best_coarse_temp = temp

        logger.info(f"  Phase 1 result: T={best_coarse_temp:.4f}, "
                    f"corr={best_coarse_corr:.4f}")

        # Phase 2: Fine grid search around best coarse temperature
        fine_min = max(temp_min, best_coarse_temp - 5 * temp_step)
        fine_max = min(temp_max, best_coarse_temp + 5 * temp_step)
        fine_temps = np.arange(fine_min, fine_max + fine_step, fine_step)

        best_fine_temp = best_coarse_temp
        best_fine_corr = best_coarse_corr

        logger.info(f"  Phase 2: Fine search over {len(fine_temps)} temperatures "
                    f"in [{fine_min:.4f}, {fine_max:.4f}]...")
        for temp in fine_temps:
            avg_corr = self._evaluate_temperature_accuracy(
                all_logits, all_labels, model_names, temp
            )
            search_history.append((float(temp), float(avg_corr)))

            if avg_corr > best_fine_corr:
                best_fine_corr = avg_corr
                best_fine_temp = temp

        logger.info(f"  Phase 2 result: T={best_fine_temp:.4f}, "
                    f"corr={best_fine_corr:.4f}")

        # Compute per-model detailed results at best temperature
        per_model_results = {}
        for model_name in model_names:
            logits = all_logits[model_name]
            labels = all_labels[model_name]
            great_engine = ClassConditionalGREAT(
                num_classes=self.num_classes,
                temperature=best_fine_temp,
                use_sigmoid=self.use_sigmoid,
                device="cpu",
            )
            per_class = great_engine.compute_per_class_scores(logits, labels)
            class_scores = [per_class[k]["score"] for k in sorted(per_class.keys())]
            class_accs = [per_class[k]["accuracy"] for k in sorted(per_class.keys())]

            if len(set(class_scores)) > 1 and len(set(class_accs)) > 1:
                corr, pval = scipy_stats.spearmanr(class_scores, class_accs)
            else:
                corr, pval = 0.0, 1.0

            per_model_results[model_name] = {
                "per_class_scores": class_scores,
                "per_class_accuracies": class_accs,
                "spearman_corr": float(corr),
                "p_value": float(pval),
            }

        result = {
            "best_temperature": float(best_fine_temp),
            "best_correlation": float(best_fine_corr),
            "per_model_results": per_model_results,
            "search_history": search_history,
            "method": "accuracy_correlation",
        }

        return result

    def calibrate_ranking_stability(
        self,
        all_logits: Dict[str, np.ndarray],
        all_labels: Dict[str, np.ndarray],
        temp_min: float = CALIBRATION_TEMP_MIN,
        temp_max: float = CALIBRATION_TEMP_MAX,
        temp_step: float = CALIBRATION_TEMP_STEP,
        perturbation_delta: float = 0.05,
    ) -> dict:
        """
        Calibrate temperature via ranking stability.

        Method: Find T where per-class score rankings are most stable
        under small perturbations delta around T.

        T* = argmax_T min_{T' in [T-delta, T+delta]}
                rho(rank(Omega_k(f; T)), rank(Omega_k(f; T')))

        Args:
            all_logits: dict mapping model_name -> (N, K) logits array.
            all_labels: dict mapping model_name -> (N,) labels array.
            temp_min, temp_max, temp_step: grid search parameters.
            perturbation_delta: size of temperature perturbation window.

        Returns:
            dict with calibration results.
        """
        logger.info("Running self-calibration via ranking stability...")
        logger.info(f"  Perturbation delta: {perturbation_delta}")

        model_names = sorted(all_logits.keys())
        temperatures = np.arange(temp_min, temp_max + temp_step, temp_step)

        best_temp = 1.0
        best_stability = -1.0
        search_history = []

        for temp in temperatures:
            stability = self._evaluate_temperature_stability(
                all_logits, all_labels, model_names, temp, perturbation_delta
            )
            search_history.append((float(temp), float(stability)))

            if stability > best_stability:
                best_stability = stability
                best_temp = temp

        logger.info(f"  Result: T={best_temp:.4f}, stability={best_stability:.4f}")

        return {
            "best_temperature": float(best_temp),
            "best_stability": float(best_stability),
            "search_history": search_history,
            "method": "ranking_stability",
            "perturbation_delta": perturbation_delta,
        }

    def _evaluate_temperature_accuracy(
        self,
        all_logits: Dict[str, np.ndarray],
        all_labels: Dict[str, np.ndarray],
        model_names: List[str],
        temperature: float,
    ) -> float:
        """
        Evaluate a temperature by computing average per-model Spearman
        correlation between per-class GREAT Scores and per-class accuracies.
        """
        correlations = []
        for model_name in model_names:
            logits = all_logits[model_name]
            labels = all_labels[model_name]

            great_engine = ClassConditionalGREAT(
                num_classes=self.num_classes,
                temperature=temperature,
                use_sigmoid=self.use_sigmoid,
                device="cpu",
            )
            per_class = great_engine.compute_per_class_scores(logits, labels)
            class_scores = [per_class[k]["score"] for k in sorted(per_class.keys())]
            class_accs = [per_class[k]["accuracy"] for k in sorted(per_class.keys())]

            if len(set(class_scores)) > 1 and len(set(class_accs)) > 1:
                corr, _ = scipy_stats.spearmanr(class_scores, class_accs)
                correlations.append(corr)

        if not correlations:
            return 0.0
        return float(np.mean(correlations))

    def _evaluate_temperature_stability(
        self,
        all_logits: Dict[str, np.ndarray],
        all_labels: Dict[str, np.ndarray],
        model_names: List[str],
        temperature: float,
        delta: float,
    ) -> float:
        """
        Evaluate ranking stability at a given temperature.

        Computes min Spearman correlation between rankings at T and at
        T +/- delta, averaged over all models.
        """
        perturbed_temps = [
            max(0.001, temperature - delta),
            temperature + delta,
        ]

        min_correlations = []
        for model_name in model_names:
            logits = all_logits[model_name]
            labels = all_labels[model_name]

            # Scores at T
            engine_t = ClassConditionalGREAT(
                num_classes=self.num_classes,
                temperature=temperature,
                use_sigmoid=self.use_sigmoid,
                device="cpu",
            )
            pc_t = engine_t.compute_per_class_scores(logits, labels)
            scores_t = [pc_t[k]["score"] for k in sorted(pc_t.keys())]

            # Scores at perturbed temperatures
            model_corrs = []
            for pt in perturbed_temps:
                engine_pt = ClassConditionalGREAT(
                    num_classes=self.num_classes,
                    temperature=pt,
                    use_sigmoid=self.use_sigmoid,
                    device="cpu",
                )
                pc_pt = engine_pt.compute_per_class_scores(logits, labels)
                scores_pt = [pc_pt[k]["score"] for k in sorted(pc_pt.keys())]

                if len(set(scores_t)) > 1 and len(set(scores_pt)) > 1:
                    corr, _ = scipy_stats.spearmanr(scores_t, scores_pt)
                    model_corrs.append(corr)

            if model_corrs:
                min_correlations.append(min(model_corrs))

        if not min_correlations:
            return 0.0
        return float(np.mean(min_correlations))

    def save_results(self, results: dict, filename: str = "self_calibration_results.json"):
        """Save calibration results to JSON."""
        save_path = RESULTS_DIR / filename
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Make numpy arrays JSON-serializable
        serializable = self._make_serializable(results)
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Saved calibration results to {save_path}")

    def _make_serializable(self, obj):
        """Recursively convert numpy types to Python types for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
