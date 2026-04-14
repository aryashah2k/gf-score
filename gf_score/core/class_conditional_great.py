"""
Class-Conditional GREAT Score Computation
==========================================
Implements per-class GREAT Score decomposition with certified guarantees.

The key formula (from Theorem 1 of the original paper):
    g(x) = sqrt(pi/2) * max{f_c(x) - max_{k!=c} f_k(x), 0}

Our contribution: decompose E[g(G(z))] into per-class expectations:
    Omega_hat_k(f) = E_{z | y=k}[g(G(z|k))]

By linearity of expectation:
    Omega_hat(f) = sum_k P(Y=k) * Omega_hat_k(f)
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gf_score.config import (
    SQRT_PI_OVER_2,
    CIFAR10_NUM_CLASSES,
    CIFAR10_CLASSES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    LOGITS_DIR,
    SCORES_DIR,
)

logger = logging.getLogger("gf_score.core.class_conditional")


class ClassConditionalGREAT:
    """
    Computes class-conditional GREAT Scores with certified guarantees.

    Supports:
    - Per-class robustness score decomposition
    - Aggregate score (recovers original GREAT Score)
    - Worst-case class robustness (WCR)
    - Batch GPU inference with checkpoint/resume
    """

    def __init__(
        self,
        num_classes: int = CIFAR10_NUM_CLASSES,
        temperature: float = DEFAULT_TEMPERATURE,
        use_sigmoid: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            num_classes: Number of output classes.
            temperature: Temperature for sigmoid/softmax scaling.
            use_sigmoid: If True, use sigmoid (CIFAR-10). If False, use softmax (ImageNet).
            batch_size: Batch size for GPU inference.
            device: Compute device ('cuda' or 'cpu').
        """
        self.num_classes = num_classes
        self.temperature = temperature
        self.use_sigmoid = use_sigmoid
        self.batch_size = batch_size
        self.device = device

    def compute_logits(
        self,
        model: torch.nn.Module,
        images: np.ndarray,
        model_name: str = "model",
    ) -> np.ndarray:
        """
        Run model forward pass to get logits, with checkpoint support.

        Args:
            model: PyTorch model (already on device and in eval mode).
            images: (N, C, H, W) float32 array in [0, 1].
            model_name: Name for checkpoint file.

        Returns:
            logits: (N, K) float32 array of raw model logits.
        """
        # Check for existing checkpoint
        checkpoint_path = LOGITS_DIR / f"{model_name}_logits.npz"
        if checkpoint_path.exists():
            logger.info(f"Loading cached logits for {model_name} from {checkpoint_path}")
            data = np.load(checkpoint_path)
            cached_logits = data["logits"]
            cached_n = data["num_samples"]
            if int(cached_n) == len(images):
                return cached_logits
            else:
                logger.warning(
                    f"Cached logits have {cached_n} samples but {len(images)} requested. "
                    f"Recomputing."
                )

        logger.info(f"Computing logits for {model_name} ({len(images)} samples)...")
        model.eval()
        all_logits = []

        num_batches = math.ceil(len(images) / self.batch_size)
        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc=f"  {model_name}",
                leave=False,
            ):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(images))
                batch = torch.from_numpy(images[start:end]).float().to(self.device)
                logits = model(batch)
                all_logits.append(logits.cpu().numpy())

        logits_array = np.concatenate(all_logits, axis=0)  # (N, K)

        # Save checkpoint
        LOGITS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint_path,
            logits=logits_array,
            num_samples=len(images),
        )
        logger.info(f"Saved logits checkpoint to {checkpoint_path}")

        return logits_array

    def compute_local_scores(
        self, logits: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-sample local GREAT Score.

        For each sample i with true label c_i:
            p = sigmoid(logits_i / T) or softmax(logits_i / T)
            gap = p[c_i] - max_{k != c_i} p[k]
            score_i = sqrt(pi/2) * max(gap, 0)

        Args:
            logits: (N, K) raw model logits.
            labels: (N,) true class labels.

        Returns:
            scores: (N,) per-sample GREAT Scores.
        """
        logits_tensor = torch.from_numpy(logits).float()

        # Apply temperature scaling + activation
        scaled_logits = logits_tensor / self.temperature
        if self.use_sigmoid:
            probs = torch.sigmoid(scaled_logits)
        else:
            probs = F.softmax(scaled_logits, dim=1)

        probs = probs.numpy()
        labels = labels.astype(np.int64)
        n_samples = len(labels)
        scores = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            true_class = labels[i]
            p_true = probs[i, true_class]

            # Get max probability among non-true classes
            mask = np.ones(self.num_classes, dtype=bool)
            mask[true_class] = False
            p_top2 = probs[i, mask].max()

            gap = p_true - p_top2
            if gap > 0:
                scores[i] = SQRT_PI_OVER_2 * gap
            else:
                scores[i] = 0.0

        return scores

    def compute_per_class_scores(
        self, logits: np.ndarray, labels: np.ndarray
    ) -> Dict[int, dict]:
        """
        Compute per-class GREAT Scores (our novel contribution).

        Omega_hat_k(f) = mean of local scores for class k.

        Args:
            logits: (N, K) raw model logits.
            labels: (N,) true class labels.

        Returns:
            per_class: dict mapping class_idx -> {
                'score': float (per-class GREAT Score),
                'std': float (standard deviation),
                'count': int (number of samples),
                'local_scores': np.ndarray (per-sample scores for this class),
                'accuracy': float (clean accuracy on this class),
            }
        """
        # Compute all local scores
        local_scores = self.compute_local_scores(logits, labels)

        # Compute clean predictions for accuracy
        logits_tensor = torch.from_numpy(logits).float()
        predictions = logits_tensor.argmax(dim=1).numpy()
        labels_np = labels.astype(np.int64)

        per_class = {}
        for k in range(self.num_classes):
            mask = labels_np == k
            class_scores = local_scores[mask]
            class_preds = predictions[mask]
            class_labels = labels_np[mask]

            if len(class_scores) == 0:
                per_class[k] = {
                    "score": 0.0,
                    "std": 0.0,
                    "count": 0,
                    "local_scores": np.array([]),
                    "accuracy": 0.0,
                }
                continue

            per_class[k] = {
                "score": float(np.mean(class_scores)),
                "std": float(np.std(class_scores, ddof=1)) if len(class_scores) > 1 else 0.0,
                "count": int(len(class_scores)),
                "local_scores": class_scores,
                "accuracy": float(np.mean(class_preds == class_labels)),
            }

        return per_class

    def compute_aggregate_score(
        self, per_class_scores: Dict[int, dict]
    ) -> float:
        """
        Compute the aggregate (original) GREAT Score from per-class scores.

        By linearity of expectation:
            Omega_hat(f) = sum_k (n_k / N) * Omega_hat_k(f)

        This should match the original GREAT Score computation.
        """
        total_count = sum(pc["count"] for pc in per_class_scores.values())
        if total_count == 0:
            return 0.0

        weighted_sum = sum(
            pc["score"] * pc["count"] for pc in per_class_scores.values()
        )
        return weighted_sum / total_count

    def compute_aggregate_from_local(
        self, logits: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Compute the aggregate GREAT Score directly (for validation).

        This bypasses per-class decomposition to verify consistency.
        """
        local_scores = self.compute_local_scores(logits, labels)
        return float(np.mean(local_scores))

    def get_robustness_profile(
        self, per_class_scores: Dict[int, dict], class_names: List[str] = None
    ) -> dict:
        """
        Build a comprehensive robustness profile for a model.

        Returns:
            profile: dict with {
                'per_class_scores': {class_name: score},
                'per_class_std': {class_name: std},
                'per_class_accuracy': {class_name: accuracy},
                'aggregate_score': float,
                'worst_class': str,
                'worst_score': float,
                'best_class': str,
                'best_score': float,
                'score_range': float,
                'num_classes': int,
                'total_samples': int,
            }
        """
        if class_names is None:
            class_names = CIFAR10_CLASSES

        scores_dict = {}
        std_dict = {}
        acc_dict = {}
        for k, pc in per_class_scores.items():
            name = class_names[k] if k < len(class_names) else f"class_{k}"
            scores_dict[name] = pc["score"]
            std_dict[name] = pc["std"]
            acc_dict[name] = pc["accuracy"]

        aggregate = self.compute_aggregate_score(per_class_scores)
        score_values = [pc["score"] for pc in per_class_scores.values() if pc["count"] > 0]

        worst_idx = min(per_class_scores, key=lambda k: per_class_scores[k]["score"])
        best_idx = max(per_class_scores, key=lambda k: per_class_scores[k]["score"])
        worst_name = class_names[worst_idx] if worst_idx < len(class_names) else f"class_{worst_idx}"
        best_name = class_names[best_idx] if best_idx < len(class_names) else f"class_{best_idx}"

        return {
            "per_class_scores": scores_dict,
            "per_class_std": std_dict,
            "per_class_accuracy": acc_dict,
            "aggregate_score": aggregate,
            "worst_class": worst_name,
            "worst_score": per_class_scores[worst_idx]["score"],
            "best_class": best_name,
            "best_score": per_class_scores[best_idx]["score"],
            "score_range": max(score_values) - min(score_values) if score_values else 0.0,
            "num_classes": self.num_classes,
            "total_samples": sum(pc["count"] for pc in per_class_scores.values()),
        }

    def save_scores(
        self, per_class_scores: Dict[int, dict], model_name: str, output_dir: Path = SCORES_DIR
    ):
        """Save per-class scores to JSON checkpoint (excluding numpy arrays)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{model_name}_per_class_scores.json"

        serializable = {}
        for k, pc in per_class_scores.items():
            serializable[str(k)] = {
                "score": pc["score"],
                "std": pc["std"],
                "count": pc["count"],
                "accuracy": pc["accuracy"],
            }

        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Saved per-class scores to {save_path}")

    def load_scores(self, model_name: str, output_dir: Path = SCORES_DIR) -> Optional[Dict[int, dict]]:
        """Load per-class scores from JSON checkpoint."""
        save_path = output_dir / f"{model_name}_per_class_scores.json"
        if not save_path.exists():
            return None

        with open(save_path, "r") as f:
            data = json.load(f)

        per_class = {}
        for k_str, pc in data.items():
            per_class[int(k_str)] = {
                "score": pc["score"],
                "std": pc["std"],
                "count": pc["count"],
                "accuracy": pc["accuracy"],
                "local_scores": np.array([]),  # Not stored in checkpoint
            }
        logger.info(f"Loaded per-class scores from {save_path}")
        return per_class


def compute_sample_complexity_bound(
    epsilon: float,
    delta: float,
    num_classes: int = 1,
) -> int:
    """
    Compute minimum sample size per class from Theorem 2 with union-bound
    correction (Proposition 1).

    For joint guarantee across K classes:
        n_k >= 32 * e * log(2K / delta) / epsilon^2

    Args:
        epsilon: Tolerance on score estimation.
        delta: Maximum failure probability.
        num_classes: Number of classes (K) for union bound.

    Returns:
        Minimum number of samples per class.
    """
    e = math.e
    n = math.ceil(32 * e * math.log(2 * num_classes / delta) / (epsilon ** 2))
    return n
