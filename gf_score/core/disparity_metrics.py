"""
Robustness Disparity Metrics
=============================
Novel fairness-aware metrics for quantifying class-level robustness inequality.

Metrics implemented:
- RDI (Robustness Disparity Index): max_k - min_k per-class score
- NRGC (Normalized Robustness Gini Coefficient): Gini index over per-class scores  
- WCR (Worst-Case Class Robustness): min_k per-class score
- FP-GREAT (Fairness-Penalized GREAT): Omega_bar - lambda * RDI

Grounded in established fairness theory:
- RDI adapts the Max Group Disparity principle
- NRGC adapts the Gini Index from welfare economics (Gini, 1912)
- WCR embodies the Rawlsian Maximin principle (Rawls, 1971)
- FP-GREAT adapts Inequality-Adjusted Welfare (cf. UNDP IHDI)
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from gf_score.config import DEFAULT_FP_LAMBDA

logger = logging.getLogger("gf_score.core.disparity_metrics")


def _extract_scores(per_class_scores: Dict[int, dict]) -> np.ndarray:
    """
    Extract score values from per-class scores dict.

    Args:
        per_class_scores: dict mapping class_idx -> {'score': float, ...}

    Returns:
        scores: 1D numpy array of per-class GREAT scores, sorted by class index.
    """
    sorted_keys = sorted(per_class_scores.keys())
    scores = np.array([per_class_scores[k]["score"] for k in sorted_keys])
    return scores


def robustness_disparity_index(per_class_scores: Dict[int, dict]) -> float:
    """
    Robustness Disparity Index (RDI).

    RDI(f) = max_k Omega_hat_k(f) - min_k Omega_hat_k(f)

    Properties:
    - Non-negative: RDI >= 0
    - Zero iff uniform: RDI = 0  <=>  all classes have equal robustness
    - Bounded: RDI <= sqrt(pi/2) ≈ 1.253 (classifier outputs bounded in [0,1])

    Adapts the Max Group Disparity principle from fairness auditing.

    Args:
        per_class_scores: dict from ClassConditionalGREAT.compute_per_class_scores()

    Returns:
        RDI value (float >= 0).
    """
    scores = _extract_scores(per_class_scores)
    if len(scores) == 0:
        return 0.0
    return float(np.max(scores) - np.min(scores))


def normalized_robustness_gini(per_class_scores: Dict[int, dict]) -> float:
    """
    Normalized Robustness Gini Coefficient (NRGC).

    NRGC(f) = sum_i sum_j |Omega_i - Omega_j| / (2 * K^2 * Omega_bar)

    where Omega_bar = mean per-class robustness.

    Properties:
    - NRGC in [0, 1]  (0 = perfect equality, approaching 1 = maximal inequality)
    - More informative than RDI for K > 2 (captures full distribution shape)
    - Returns 0.0 if mean robustness is 0 (undefined case, treated as equal)

    Adapts the Gini Index from welfare economics (Corrado Gini, 1912).

    Args:
        per_class_scores: dict from ClassConditionalGREAT.compute_per_class_scores()

    Returns:
        NRGC value (float in [0, 1]).
    """
    scores = _extract_scores(per_class_scores)
    K = len(scores)

    if K <= 1:
        return 0.0

    mean_score = np.mean(scores)
    if mean_score <= 0:
        return 0.0

    # Compute pairwise absolute differences
    abs_diffs = 0.0
    for i in range(K):
        for j in range(K):
            abs_diffs += abs(scores[i] - scores[j])

    gini = abs_diffs / (2.0 * K * K * mean_score)
    return float(gini)


def worst_case_robustness(per_class_scores: Dict[int, dict]) -> Tuple[float, int]:
    """
    Worst-Case Class Robustness (WCR).

    WCR(f) = min_k Omega_hat_k(f)

    Operational meaning: the certified L2 perturbation level guaranteed for
    EVERY class. Embodies the Rawlsian Maximin principle — a model passes
    a fairness audit only if WCR >= threshold.

    Args:
        per_class_scores: dict from ClassConditionalGREAT.compute_per_class_scores()

    Returns:
        Tuple of (wcr_score, worst_class_idx).
    """
    scores = _extract_scores(per_class_scores)
    if len(scores) == 0:
        return 0.0, -1

    worst_idx = int(np.argmin(scores))
    sorted_keys = sorted(per_class_scores.keys())
    return float(scores[worst_idx]), sorted_keys[worst_idx]


def fairness_penalized_great(
    per_class_scores: Dict[int, dict],
    lambda_val: float = DEFAULT_FP_LAMBDA,
) -> float:
    """
    Fairness-Penalized GREAT Score (FP-GREAT).

    FP-GREAT(f; lambda) = Omega_bar(f) - lambda * RDI(f)

    Ranks models by BOTH aggregate robustness and fairness.
    - lambda=0: reduces to mean GREAT Score (no fairness penalty)
    - lambda=1: heavily penalizes models with high disparity

    Adapts Inequality-Adjusted Welfare (cf. UNDP IHDI).

    Args:
        per_class_scores: dict from ClassConditionalGREAT.compute_per_class_scores()
        lambda_val: fairness penalty weight in [0, 1].

    Returns:
        FP-GREAT score (float).
    """
    scores = _extract_scores(per_class_scores)
    if len(scores) == 0:
        return 0.0

    omega_bar = float(np.mean(scores))
    rdi = robustness_disparity_index(per_class_scores)
    return omega_bar - lambda_val * rdi


def pairwise_disparity_matrix(per_class_scores: Dict[int, dict]) -> np.ndarray:
    """
    Compute the pairwise robustness disparity matrix.

    D[i, j] = |Omega_hat_i - Omega_hat_j|

    Useful for identifying which pairs of classes have the largest
    robustness gap.

    Args:
        per_class_scores: dict from ClassConditionalGREAT.compute_per_class_scores()

    Returns:
        D: (K, K) symmetric matrix of pairwise absolute differences.
    """
    scores = _extract_scores(per_class_scores)
    K = len(scores)
    D = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            D[i, j] = abs(scores[i] - scores[j])
    return D


def class_vulnerability_ranking(
    per_class_scores: Dict[int, dict],
    class_names: List[str] = None,
) -> List[Tuple[str, float]]:
    """
    Rank classes from most vulnerable (lowest score) to most robust.

    Args:
        per_class_scores: dict from ClassConditionalGREAT.compute_per_class_scores()
        class_names: optional list of class name strings.

    Returns:
        List of (class_name, score) tuples, sorted ascending by score.
    """
    sorted_keys = sorted(per_class_scores.keys())
    items = []
    for k in sorted_keys:
        if class_names and k < len(class_names):
            name = class_names[k]
        else:
            name = f"class_{k}"
        items.append((name, per_class_scores[k]["score"]))

    items.sort(key=lambda x: x[1])
    return items


def compute_all_metrics(
    per_class_scores: Dict[int, dict],
    lambda_val: float = DEFAULT_FP_LAMBDA,
    class_names: List[str] = None,
) -> dict:
    """
    Compute all disparity metrics in one call.

    Returns:
        dict with all metric values and supporting information.
    """
    scores = _extract_scores(per_class_scores)
    rdi = robustness_disparity_index(per_class_scores)
    nrgc = normalized_robustness_gini(per_class_scores)
    wcr_score, wcr_class = worst_case_robustness(per_class_scores)
    fp_great = fairness_penalized_great(per_class_scores, lambda_val)
    vuln_ranking = class_vulnerability_ranking(per_class_scores, class_names)
    disparity_matrix = pairwise_disparity_matrix(per_class_scores)

    # Best class info
    best_idx = int(np.argmax(scores))
    sorted_keys = sorted(per_class_scores.keys())
    best_class = sorted_keys[best_idx]

    if class_names:
        wcr_class_name = class_names[wcr_class] if wcr_class < len(class_names) else f"class_{wcr_class}"
        best_class_name = class_names[best_class] if best_class < len(class_names) else f"class_{best_class}"
    else:
        wcr_class_name = f"class_{wcr_class}"
        best_class_name = f"class_{best_class}"

    return {
        "rdi": rdi,
        "nrgc": nrgc,
        "wcr": wcr_score,
        "wcr_class_idx": wcr_class,
        "wcr_class_name": wcr_class_name,
        "fp_great": fp_great,
        "fp_lambda": lambda_val,
        "best_score": float(np.max(scores)),
        "best_class_idx": best_class,
        "best_class_name": best_class_name,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "vulnerability_ranking": vuln_ranking,
        "disparity_matrix": disparity_matrix.tolist(),
    }


def rdi_concentration_bound(
    epsilon: float,
    delta: float,
    num_classes: int,
) -> Tuple[int, float]:
    """
    Compute sample size for RDI concentration (Proposition 2).

    Under the conditions of Proposition 1, with probability >= 1 - delta:
        |RDI_S(f) - RDI(f)| <= 2 * epsilon

    Args:
        epsilon: Per-class estimation tolerance.
        delta: Maximum failure probability.
        num_classes: Number of classes K.

    Returns:
        Tuple of (min_samples_per_class, rdi_tolerance).
        rdi_tolerance = 2 * epsilon (the bound on |RDI_S - RDI|).
    """
    import math
    e = math.e
    n_k = math.ceil(32 * e * math.log(2 * num_classes / delta) / (epsilon ** 2))
    return n_k, 2 * epsilon
