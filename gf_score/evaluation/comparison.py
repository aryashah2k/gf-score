"""
Comparison with Original GREAT Score
=====================================
Compares GF-Score results with the original GREAT Score paper values
to validate consistency and demonstrate novel insights.

Usage:
    python -m gf_score.evaluation.comparison
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gf_score.config import (
    CIFAR10_CLASSES,
    MODEL_SHORT_NAMES,
    ROBUSTBENCH_ACCURACY,
    ORIGINAL_GREAT_SCORES,
    ORIGINAL_GREAT_SCORES_TEST,
    CW_DISTORTION,
    CALIBRATED_GREAT_SCORES,
    PAPER_CORRELATIONS,
    RESULTS_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    get_dataset_config,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("gf_score.evaluation.comparison")


def load_results(dataset: str = "cifar10", results_path: Path = None) -> dict:
    """Load the full results from the evaluation pipeline."""
    if results_path is None:
        suffix = f"_{dataset}" if dataset != "cifar10" else ""
        results_path = RESULTS_DIR / f"full_results{suffix}.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}. "
            f"Run the evaluation pipeline first: python -m gf_score.evaluation.run_evaluation --dataset {dataset}"
        )
    with open(results_path, "r") as f:
        return json.load(f)


def compare_with_paper(results: dict, dataset: str = "cifar10"):
    """
    Compare our results with the original paper's values.

    Generates:
    1. Decomposition consistency check
    2. Rank correlation comparison
    3. Identification of cases where per-class analysis reveals new insights
    """
    ds_cfg = get_dataset_config(dataset)
    ds_short_names = ds_cfg["short_names"]
    ds_rb_accuracy = ds_cfg["robustbench_accuracy"]
    model_results = results["model_results"]
    comparison = {}

    print("\n" + "=" * 100)
    print("COMPARISON WITH ORIGINAL GREAT SCORE PAPER")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Decomposition consistency verification
    # ------------------------------------------------------------------
    print("\n1. DECOMPOSITION CONSISTENCY")
    print("-" * 60)
    print(f"{'Model':>25s} | {'Decomposed':>10s} | {'Direct':>10s} | {'Error':>12s} | {'Status':>8s}")
    print("-" * 75)

    all_consistent = True
    for model_name, r in model_results.items():
        decomposed = r["aggregate_great_score"]
        direct = r["aggregate_great_score_direct"]
        error = abs(decomposed - direct)
        status = "✓ PASS" if error < 1e-8 else "✗ FAIL"
        if error >= 1e-8:
            all_consistent = False

        short = ds_short_names.get(model_name, model_name[:25])
        print(f"{short:>25s} | {decomposed:>10.6f} | {direct:>10.6f} | {error:>12.2e} | {status}")

    print(f"\nOverall: {'ALL PASSED ✓' if all_consistent else 'SOME FAILED ✗'}")
    comparison["decomposition_consistent"] = all_consistent

    # ------------------------------------------------------------------
    # 2. Rank correlation comparison
    # ------------------------------------------------------------------
    print("\n2. RANK CORRELATIONS")
    print("-" * 60)

    model_list = list(model_results.keys())
    our_scores = [model_results[m]["aggregate_great_score"] for m in model_list]
    our_cal_scores = [model_results[m].get("calibrated_great_score", 0) for m in model_list]
    rb_accs = [ds_rb_accuracy.get(m, 0) for m in model_list]
    paper_scores = [ORIGINAL_GREAT_SCORES_TEST.get(m, 0) for m in model_list]

    # Our uncalibrated vs RobustBench
    corr_ours, _ = scipy_stats.spearmanr(our_scores, rb_accs)
    # Paper's uncalibrated vs RobustBench (CIFAR-10 only)
    corr_paper = PAPER_CORRELATIONS.get("uncalibrated_vs_robustbench", 0) if dataset == "cifar10" else None
    # Our calibrated vs RobustBench
    corr_cal, _ = scipy_stats.spearmanr(our_cal_scores, rb_accs)
    corr_paper_cal = PAPER_CORRELATIONS.get("calibrated_vs_robustbench", 0) if dataset == "cifar10" else None

    # Our scores vs Paper's test scores (CIFAR-10 only)
    if dataset == "cifar10" and any(s > 0 for s in paper_scores):
        corr_vs_paper, _ = scipy_stats.spearmanr(our_scores, paper_scores)
    else:
        corr_vs_paper = None

    print(f"{'Metric':>45s} | {'Ours':>8s} | {'Paper':>8s}")
    print("-" * 70)
    print(f"{'Uncalibrated GREAT vs RobustBench':>45s} | {corr_ours:>8.4f} | {(f'{corr_paper:.4f}' if corr_paper else 'N/A'):>8s}")
    print(f"{'Calibrated GREAT vs RobustBench':>45s} | {corr_cal:>8.4f} | {(f'{corr_paper_cal:.4f}' if corr_paper_cal else 'N/A'):>8s}")
    if corr_vs_paper is not None:
        print(f"{'Our scores vs Paper test scores':>45s} | {corr_vs_paper:>8.4f} | {'N/A':>8s}")

    comparison["rank_correlations"] = {
        "ours_uncal_vs_rb": float(corr_ours),
        "paper_uncal_vs_rb": float(corr_paper) if corr_paper else None,
        "ours_cal_vs_rb": float(corr_cal),
        "paper_cal_vs_rb": float(corr_paper_cal) if corr_paper_cal else None,
        "ours_vs_paper_test": float(corr_vs_paper) if corr_vs_paper else None,
    }

    # ------------------------------------------------------------------
    # 3. Novel insights from per-class analysis
    # ------------------------------------------------------------------
    print("\n3. NOVEL INSIGHTS FROM PER-CLASS ANALYSIS")
    print("-" * 60)

    # Find models with similar aggregate scores but different RDI
    scores_with_rdi = []
    for model_name, r in model_results.items():
        scores_with_rdi.append({
            "model": ds_short_names.get(model_name, model_name),
            "aggregate": r["aggregate_great_score"],
            "rdi": r["rdi"],
            "wcr": r["wcr"],
            "wcr_class": r["wcr_class"],
            "nrgc": r["nrgc"],
        })

    # Sort by aggregate score
    scores_with_rdi.sort(key=lambda x: x["aggregate"], reverse=True)

    print(f"\n{'Model':>25s} | {'Aggregate':>10s} | {'RDI':>8s} | {'NRGC':>8s} | {'WCR':>8s} | {'Worst Class':>12s}")
    print("-" * 85)
    for item in scores_with_rdi:
        print(f"{item['model']:>25s} | {item['aggregate']:>10.4f} | "
              f"{item['rdi']:>8.4f} | {item['nrgc']:>8.4f} | "
              f"{item['wcr']:>8.4f} | {item['wcr_class']:>12s}")

    # Identify interesting cases: similar aggregate but different RDI
    print("\n  Interesting Findings:")
    for i in range(len(scores_with_rdi)):
        for j in range(i + 1, len(scores_with_rdi)):
            a, b = scores_with_rdi[i], scores_with_rdi[j]
            agg_diff = abs(a["aggregate"] - b["aggregate"])
            rdi_diff = abs(a["rdi"] - b["rdi"])
            if agg_diff < 0.05 and rdi_diff > 0.1:
                print(f"  → {a['model']} and {b['model']} have similar "
                      f"aggregate scores ({a['aggregate']:.4f} vs {b['aggregate']:.4f}) "
                      f"but different RDI ({a['rdi']:.4f} vs {b['rdi']:.4f})")

    # Most and least fair models
    most_fair = min(scores_with_rdi, key=lambda x: x["rdi"])
    least_fair = max(scores_with_rdi, key=lambda x: x["rdi"])
    print(f"\n  Most equitable model:  {most_fair['model']} (RDI={most_fair['rdi']:.4f})")
    print(f"  Least equitable model: {least_fair['model']} (RDI={least_fair['rdi']:.4f})")

    # Most common worst class
    worst_classes = [r["wcr_class"] for r in model_results.values()]
    from collections import Counter
    worst_counter = Counter(worst_classes)
    print(f"\n  Most common worst class: {worst_counter.most_common(1)[0]}")

    comparison["novel_insights"] = {
        "most_fair_model": most_fair["model"],
        "most_fair_rdi": most_fair["rdi"],
        "least_fair_model": least_fair["model"],
        "least_fair_rdi": least_fair["rdi"],
        "worst_class_distribution": dict(worst_counter),
    }

    # Save comparison results
    comparison_path = RESULTS_DIR / f"comparison_results{'_' + dataset if dataset != 'cifar10' else ''}.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved comparison results to {comparison_path}")

    print("\n" + "=" * 100)
    return comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare GF-Score with paper values")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "imagenet"])
    args = parser.parse_args()
    results = load_results(args.dataset)
    compare_with_paper(results, args.dataset)


if __name__ == "__main__":
    main()
