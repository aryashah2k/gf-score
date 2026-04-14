"""
GF-Score Evaluation Pipeline
=============================
Main evaluation script that runs the complete GF-Score analysis
on RobustBench models for CIFAR-10 (L2) or ImageNet (Linf).

Pipeline:
1. Download/load data
2. For each model: compute logits (with checkpoint/resume)
3. Compute per-class GREAT Scores
4. Compute disparity metrics (RDI, NRGC, WCR, FP-GREAT)
5. Run self-calibration
6. Validate decomposition consistency
7. Save all results

Usage:
    python -m gf_score.evaluation.run_evaluation [--dataset cifar10] [--quick_test]
    python -m gf_score.evaluation.run_evaluation --dataset imagenet [--quick_test]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scipy_stats

# Add parent to path for direct module execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gf_score.config import (
    CIFAR10_CLASSES,
    CIFAR10_NUM_CLASSES,
    ORIGINAL_GREAT_SCORES,
    ORIGINAL_GREAT_SCORES_TEST,
    CW_DISTORTION,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FP_LAMBDA,
    RESULTS_DIR,
    LOGITS_DIR,
    SCORES_DIR,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    get_dataset_config,
)
from gf_score.core.class_conditional_great import (
    ClassConditionalGREAT,
    compute_sample_complexity_bound,
)
from gf_score.core.disparity_metrics import (
    compute_all_metrics,
    robustness_disparity_index,
    normalized_robustness_gini,
    worst_case_robustness,
    fairness_penalized_great,
)
from gf_score.core.self_calibration import SelfCalibrator
from gf_score.data.download_data import (
    download_cifar10,
    download_imagenet,
    load_gan_generated,
    get_class_conditional_data,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("gf_score.evaluation")


def load_model_safe(model_name: str, dataset: str = "cifar10", threat_model: str = "L2"):
    """Load a RobustBench model with error handling.

    Patches torch.load to use weights_only=False for compatibility with
    PyTorch 2.6+ (which defaults to weights_only=True). RobustBench
    checkpoints from Google Drive are trusted sources.

    Args:
        model_name: RobustBench model identifier
        dataset: 'cifar10' or 'imagenet'
        threat_model: 'L2' or 'Linf'
    """
    try:
        from robustbench.utils import load_model
        import functools

        # Monkey-patch torch.load to force weights_only=False
        # Required for PyTorch >= 2.6 where the default changed
        _original_torch_load = torch.load

        @functools.wraps(_original_torch_load)
        def _patched_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load
        try:
            logger.info(f"Loading model: {model_name} (dataset={dataset}, threat={threat_model})")
            model = load_model(
                model_name=model_name,
                dataset=dataset,
                threat_model=threat_model,
            )
        finally:
            # Restore original torch.load
            torch.load = _original_torch_load

        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def run_evaluation(args):
    """Main evaluation pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ======================================================================
    # Dataset configuration dispatch
    # ======================================================================
    ds_cfg = get_dataset_config(args.dataset)
    ds_classes = ds_cfg["classes"]            # list or None (ImageNet)
    ds_num_classes = ds_cfg["num_classes"]
    ds_models = ds_cfg["models"]
    ds_short_names = ds_cfg["short_names"]
    ds_rb_accuracy = ds_cfg["robustbench_accuracy"]
    ds_threat_model = ds_cfg["threat_model"]
    ds_rb_dataset = ds_cfg["dataset_robustbench"]
    ds_use_sigmoid = ds_cfg["use_sigmoid"]
    dataset_tag = args.dataset.lower()  # for filenames

    logger.info(f"Dataset: {dataset_tag} | Threat model: {ds_threat_model} | "
                f"Classes: {ds_num_classes} | Sigmoid: {ds_use_sigmoid}")

    # ======================================================================
    # Step 1: Load data
    # ======================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    if args.gan_path:
        data_info = load_gan_generated(args.gan_path)
        data_source = "gan_generated"
    elif dataset_tag == "imagenet":
        data_info = download_imagenet()
        data_source = "imagenet_val"
        # Dynamically set class names from data
        if ds_classes is None:
            ds_classes = data_info.get("class_names", [f"class_{i}" for i in range(ds_num_classes)])
    else:
        data_info = download_cifar10()
        data_source = "cifar10_test"

    images = data_info["images"]
    labels = data_info["labels"]

    # Subsample if requested
    max_per_class = args.num_samples if args.num_samples else None
    class_data = get_class_conditional_data(
        images, labels, ds_num_classes, max_per_class=max_per_class, seed=42
    )

    # Rebuild flat arrays from class-conditional data for consistency
    all_images = []
    all_labels = []
    for k in range(ds_num_classes):
        all_images.append(class_data[k]["images"])
        all_labels.append(class_data[k]["labels"])
    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    total_samples = len(images)
    per_class_counts = {k: class_data[k]["count"] for k in range(ds_num_classes)}
    logger.info(f"Total samples: {total_samples}, Classes: {ds_num_classes}")

    # Sample complexity check (Proposition 1)
    min_samples = compute_sample_complexity_bound(epsilon=0.05, delta=0.05, num_classes=ds_num_classes)
    min_per_class = min(per_class_counts.values())
    logger.info(
        f"Proposition 1 sample complexity (eps=0.05, delta=0.05, K={ds_num_classes}): "
        f"n_k >= {min_samples} | Actual min n_k = {min_per_class}"
    )
    if min_per_class < min_samples:
        logger.warning(
            f"Per-class sample count ({min_per_class}) is below the "
            f"theoretical minimum ({min_samples}). Consider increasing --num_samples."
        )

    # ======================================================================
    # Step 2: Model selection
    # ======================================================================
    if args.quick_test:
        model_list = ds_models[:2]
        logger.info(f"Quick test mode: using {len(model_list)} models")
    elif args.models:
        model_list = args.models
        logger.info(f"Custom model list: {model_list}")
    else:
        model_list = ds_models
        logger.info(f"Full evaluation: {len(model_list)} models")

    # ======================================================================
    # Step 3: Compute logits for all models (with checkpoints)
    # ======================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Computing model logits")
    logger.info("=" * 60)

    great_engine = ClassConditionalGREAT(
        num_classes=ds_num_classes,
        temperature=1.0,
        use_sigmoid=ds_use_sigmoid,
        batch_size=args.batch_size,
        device=device,
    )

    all_logits = {}
    for model_name in model_list:
        # Check for existing checkpoint
        checkpoint_path = LOGITS_DIR / f"{model_name}_logits.npz"
        if checkpoint_path.exists():
            data = np.load(checkpoint_path)
            if int(data["num_samples"]) == total_samples:
                logger.info(f"  [CACHED] {model_name}")
                all_logits[model_name] = data["logits"]
                continue

        # Load model and compute logits
        model = load_model_safe(model_name, dataset=ds_rb_dataset, threat_model=ds_threat_model)
        model.eval()
        model.to(device)

        logits = great_engine.compute_logits(model, images, model_name=model_name)
        all_logits[model_name] = logits

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ======================================================================
    # Step 4: Compute per-class GREAT Scores and disparity metrics
    # ======================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Computing per-class GREAT Scores and disparity metrics")
    logger.info("=" * 60)

    all_results = {}
    for model_name in model_list:
        logits = all_logits[model_name]

        # Per-class GREAT Scores
        per_class = great_engine.compute_per_class_scores(logits, labels)
        great_engine.save_scores(per_class, model_name)

        # Aggregate GREAT Score (via decomposition)
        agg_decomposed = great_engine.compute_aggregate_score(per_class)
        # Aggregate GREAT Score (direct computation for validation)
        agg_direct = great_engine.compute_aggregate_from_local(logits, labels)

        # Validate decomposition consistency
        decomp_error = abs(agg_decomposed - agg_direct)
        if decomp_error > 1e-10:
            logger.warning(
                f"  {model_name}: decomposition error = {decomp_error:.2e} "
                f"(decomposed={agg_decomposed:.6f}, direct={agg_direct:.6f})"
            )
        else:
            logger.info(f"  {model_name}: decomposition VERIFIED (error={decomp_error:.2e})")

        # Robustness profile
        class_names_for_profile = ds_classes if ds_classes else [f"class_{k}" for k in range(ds_num_classes)]
        profile = great_engine.get_robustness_profile(per_class, class_names_for_profile)

        # Disparity metrics
        metrics = compute_all_metrics(per_class, lambda_val=DEFAULT_FP_LAMBDA, class_names=class_names_for_profile)

        short_name = ds_short_names.get(model_name, model_name)
        logger.info(
            f"  {short_name:>20s} | "
            f"GREAT={agg_decomposed:.4f} | "
            f"RDI={metrics['rdi']:.4f} | "
            f"NRGC={metrics['nrgc']:.4f} | "
            f"WCR={metrics['wcr']:.4f} ({metrics['wcr_class_name']}) | "
            f"FP-GREAT={metrics['fp_great']:.4f}"
        )

        all_results[model_name] = {
            "model_name": model_name,
            "short_name": short_name,
            "aggregate_great_score": agg_decomposed,
            "aggregate_great_score_direct": agg_direct,
            "decomposition_error": decomp_error,
            "per_class_scores": {
                class_names_for_profile[k]: per_class[k]["score"] for k in range(ds_num_classes)
            },
            "per_class_std": {
                class_names_for_profile[k]: per_class[k]["std"] for k in range(ds_num_classes)
            },
            "per_class_accuracy": {
                class_names_for_profile[k]: per_class[k]["accuracy"] for k in range(ds_num_classes)
            },
            "per_class_counts": {
                class_names_for_profile[k]: per_class[k]["count"] for k in range(ds_num_classes)
            },
            "rdi": metrics["rdi"],
            "nrgc": metrics["nrgc"],
            "wcr": metrics["wcr"],
            "wcr_class": metrics["wcr_class_name"],
            "fp_great": metrics["fp_great"],
            "fp_lambda": metrics["fp_lambda"],
            "best_score": metrics["best_score"],
            "best_class": metrics["best_class_name"],
            "vulnerability_ranking": metrics["vulnerability_ranking"],
            "robustbench_accuracy": ds_rb_accuracy.get(model_name, None),
            "original_great_score_gan": ORIGINAL_GREAT_SCORES.get(model_name, None),
            "original_great_score_test": ORIGINAL_GREAT_SCORES_TEST.get(model_name, None),
            "cw_distortion": CW_DISTORTION.get(model_name, None),
        }

    # ======================================================================
    # Step 5: Self-Calibration
    # ======================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Self-calibration")
    logger.info("=" * 60)

    all_labels_dict = {m: labels for m in model_list}
    calibrator = SelfCalibrator(
        num_classes=ds_num_classes,
        use_sigmoid=ds_use_sigmoid,
        device=device,
    )

    # Method 1: Accuracy correlation
    acc_cal_results = calibrator.calibrate_accuracy_correlation(
        all_logits, all_labels_dict,
        temp_min=0.1, temp_max=3.0, temp_step=0.05, fine_step=0.005,
    )
    calibrator.save_results(acc_cal_results, f"self_calibration_accuracy_{dataset_tag}.json")

    logger.info(f"  Accuracy calibration: T*={acc_cal_results['best_temperature']:.4f}, "
                f"corr={acc_cal_results['best_correlation']:.4f}")

    # Method 2: Ranking stability
    stab_cal_results = calibrator.calibrate_ranking_stability(
        all_logits, all_labels_dict,
        temp_min=0.1, temp_max=3.0, temp_step=0.05,
    )
    calibrator.save_results(stab_cal_results, f"self_calibration_stability_{dataset_tag}.json")

    logger.info(f"  Stability calibration: T*={stab_cal_results['best_temperature']:.4f}, "
                f"stability={stab_cal_results['best_stability']:.4f}")

    # Compute calibrated per-class scores with best temperature
    best_temp = acc_cal_results["best_temperature"]
    logger.info(f"\n  Computing calibrated scores with T={best_temp:.4f}...")

    calibrated_engine = ClassConditionalGREAT(
        num_classes=ds_num_classes,
        temperature=best_temp,
        use_sigmoid=ds_use_sigmoid,
        batch_size=args.batch_size,
        device="cpu",
    )

    for model_name in model_list:
        logits = all_logits[model_name]
        cal_pc = calibrated_engine.compute_per_class_scores(logits, labels)
        cal_agg = calibrated_engine.compute_aggregate_score(cal_pc)
        cal_metrics = compute_all_metrics(cal_pc, lambda_val=DEFAULT_FP_LAMBDA, class_names=class_names_for_profile)

        all_results[model_name]["calibrated_temperature"] = best_temp
        all_results[model_name]["calibrated_great_score"] = cal_agg
        all_results[model_name]["calibrated_per_class_scores"] = {
            class_names_for_profile[k]: cal_pc[k]["score"] for k in range(ds_num_classes)
        }
        all_results[model_name]["calibrated_rdi"] = cal_metrics["rdi"]
        all_results[model_name]["calibrated_nrgc"] = cal_metrics["nrgc"]
        all_results[model_name]["calibrated_wcr"] = cal_metrics["wcr"]
        all_results[model_name]["calibrated_fp_great"] = cal_metrics["fp_great"]

    # ======================================================================
    # Step 6: Rank correlations
    # ======================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Computing rank correlations")
    logger.info("=" * 60)

    our_scores = [all_results[m]["aggregate_great_score"] for m in model_list]
    our_cal_scores = [all_results[m]["calibrated_great_score"] for m in model_list]
    rb_accs = [ds_rb_accuracy[m] for m in model_list]
    our_rdi = [all_results[m]["rdi"] for m in model_list]
    our_wcr = [all_results[m]["wcr"] for m in model_list]
    our_fp = [all_results[m]["fp_great"] for m in model_list]

    correlations = {}

    # GREAT Score vs RobustBench
    corr, pval = scipy_stats.spearmanr(our_scores, rb_accs)
    correlations["uncalibrated_vs_robustbench"] = {"correlation": corr, "p_value": pval}
    logger.info(f"  Uncalibrated GREAT vs RobustBench: rho={corr:.4f} (p={pval:.4f})")

    # Calibrated GREAT Score vs RobustBench
    corr, pval = scipy_stats.spearmanr(our_cal_scores, rb_accs)
    correlations["calibrated_vs_robustbench"] = {"correlation": corr, "p_value": pval}
    logger.info(f"  Calibrated GREAT vs RobustBench:   rho={corr:.4f} (p={pval:.4f})")

    # WCR vs RobustBench
    corr, pval = scipy_stats.spearmanr(our_wcr, rb_accs)
    correlations["wcr_vs_robustbench"] = {"correlation": corr, "p_value": pval}
    logger.info(f"  WCR vs RobustBench:                rho={corr:.4f} (p={pval:.4f})")

    # FP-GREAT vs RobustBench
    corr, pval = scipy_stats.spearmanr(our_fp, rb_accs)
    correlations["fp_great_vs_robustbench"] = {"correlation": corr, "p_value": pval}
    logger.info(f"  FP-GREAT vs RobustBench:           rho={corr:.4f} (p={pval:.4f})")

    # RDI vs RobustBench (negative expected: more robust models may have lower disparity)
    corr, pval = scipy_stats.spearmanr(our_rdi, rb_accs)
    correlations["rdi_vs_robustbench"] = {"correlation": corr, "p_value": pval}
    logger.info(f"  RDI vs RobustBench:                rho={corr:.4f} (p={pval:.4f})")

    # ======================================================================
    # Step 7: Save all results
    # ======================================================================
    logger.info("=" * 60)
    logger.info("STEP 6: Saving results")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Full results JSON
    full_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_tag,
            "data_source": data_source,
            "num_classes": ds_num_classes,
            "class_names": class_names_for_profile if ds_num_classes <= 100 else None,
            "threat_model": ds_threat_model,
            "use_sigmoid": ds_use_sigmoid,
            "total_samples": total_samples,
            "per_class_counts": per_class_counts,
            "num_models": len(model_list),
            "device": device,
            "calibration_temperature": best_temp,
        },
        "model_results": all_results,
        "rank_correlations": correlations,
        "calibration": {
            "accuracy_correlation": {
                "best_temperature": acc_cal_results["best_temperature"],
                "best_correlation": acc_cal_results["best_correlation"],
            },
            "ranking_stability": {
                "best_temperature": stab_cal_results["best_temperature"],
                "best_stability": stab_cal_results["best_stability"],
            },
        },
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    full_results = convert_numpy(full_results)

    # Use dataset-specific filenames
    results_suffix = f"_{dataset_tag}" if dataset_tag != "cifar10" else ""
    results_path = RESULTS_DIR / f"full_results{results_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"  Saved full results to {results_path}")

    # Summary CSV table
    rows = []
    for model_name in model_list:
        r = all_results[model_name]
        rows.append({
            "Model": ds_short_names.get(model_name, model_name),
            "RobustBench_Acc": r.get("robustbench_accuracy", ""),
            "GREAT_Score": round(r["aggregate_great_score"], 4),
            "Calibrated_GREAT": round(r["calibrated_great_score"], 4),
            "RDI": round(r["rdi"], 4),
            "NRGC": round(r["nrgc"], 4),
            "WCR": round(r["wcr"], 4),
            "WCR_Class": r["wcr_class"],
            "FP_GREAT": round(r["fp_great"], 4),
            "Best_Class": r["best_class"],
            "Best_Score": round(r["best_score"], 4),
        })

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"summary_table{results_suffix}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved summary CSV to {csv_path}")

    # Per-class scores CSV (skip for 1000-class datasets — too large for CSV)
    if ds_num_classes <= 100:
        pc_rows = []
        for model_name in model_list:
            r = all_results[model_name]
            row = {"Model": ds_short_names.get(model_name, model_name)}
            for cls_name in class_names_for_profile:
                row[f"GREAT_{cls_name}"] = round(r["per_class_scores"].get(cls_name, 0), 4)
            row["GREAT_aggregate"] = round(r["aggregate_great_score"], 4)
            pc_rows.append(row)

        pc_df = pd.DataFrame(pc_rows)
        pc_csv_path = RESULTS_DIR / f"per_class_scores{results_suffix}.csv"
        pc_df.to_csv(pc_csv_path, index=False)
        logger.info(f"  Saved per-class scores CSV to {pc_csv_path}")
    else:
        logger.info(f"  Skipping per-class CSV (too many classes: {ds_num_classes})")

    # ======================================================================
    # Step 8: Print summary
    # ======================================================================
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    print("\n" + "=" * 100)
    print("GF-SCORE EVALUATION SUMMARY")
    print("=" * 100)
    print(f"\nData: {data_source} | Samples: {total_samples} | "
          f"Models: {len(model_list)} | Calibration T*: {best_temp:.4f}")
    print()
    print(df.to_string(index=False))
    print()
    print("Rank Correlations:")
    for key, val in correlations.items():
        print(f"  {key:40s}: rho={val['correlation']:.4f}, p={val['p_value']:.4f}")
    print()
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="GF-Score Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "imagenet"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Maximum samples per class (None = all available).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--gan_path", type=str, default=None,
        help="Path to GAN-generated .npz file.",
    )
    parser.add_argument(
        "--quick_test", action="store_true",
        help="Run on first 2 models only for testing.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Specific model names to evaluate.",
    )
    args = parser.parse_args()

    start_time = time.time()
    run_evaluation(args)
    elapsed = time.time() - start_time
    logger.info(f"Total evaluation time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
