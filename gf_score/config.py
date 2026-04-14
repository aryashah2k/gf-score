"""
GF-Score Configuration
======================
Central configuration file containing all constants, model lists,
reference values from the original GREAT Score paper (NeurIPS 2024),
and path definitions.
"""

import os
import math
from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
GF_SCORE_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CIFAR10_DIR = DATA_DIR / "cifar10"
IMAGENET_DIR = DATA_DIR / "imagenet"
GAN_DATA_DIR = DATA_DIR / "gan_generated"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOGITS_DIR = CHECKPOINT_DIR / "logits"
SCORES_DIR = CHECKPOINT_DIR / "scores"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Create directories
for d in [DATA_DIR, CIFAR10_DIR, IMAGENET_DIR, GAN_DATA_DIR, OUTPUT_DIR,
          CHECKPOINT_DIR, LOGITS_DIR, SCORES_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Mathematical Constants
# =============================================================================
SQRT_PI_OVER_2 = math.sqrt(math.pi / 2)

# =============================================================================
# Dataset Configuration
# =============================================================================
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CIFAR10_NUM_CLASSES = 10

# Default evaluation parameters
DEFAULT_NUM_SAMPLES = 500      # per class (paper uses 500 total, we use 500/class)
DEFAULT_BATCH_SIZE = 128
DEFAULT_TEMPERATURE = 1.0

# =============================================================================
# CIFAR-10 L2 RobustBench Models (17 models from Table 2 of the paper)
# =============================================================================
CIFAR10_L2_MODELS = [
    "Augustin2020Adversarial_34_10_extra",
    "Augustin2020Adversarial_34_10",
    "Augustin2020Adversarial",
    "Ding2020MMA",
    "Engstrom2019Robustness",
    "Gowal2020Uncovering",
    "Gowal2020Uncovering_extra",
    "Rade2021Helper_R18_ddpm",
    "Rebuffi2021Fixing_28_10_cutmix_ddpm",
    "Rebuffi2021Fixing_70_16_cutmix_ddpm",
    "Rebuffi2021Fixing_70_16_cutmix_extra",
    "Rebuffi2021Fixing_R18_cutmix_ddpm",
    "Rice2020Overfitting",
    "Rony2019Decoupling",
    "Sehwag2021Proxy",
    "Sehwag2021Proxy_R18",
    "Wu2020Adversarial",
]

# Short display names for plots
MODEL_SHORT_NAMES = {
    "Augustin2020Adversarial_34_10_extra": "Augustin_WRN_extra",
    "Augustin2020Adversarial_34_10": "Augustin_WRN",
    "Augustin2020Adversarial": "Augustin2020",
    "Ding2020MMA": "Ding_MMA",
    "Engstrom2019Robustness": "Engstrom2019",
    "Gowal2020Uncovering": "Gowal2020",
    "Gowal2020Uncovering_extra": "Gowal_extra",
    "Rade2021Helper_R18_ddpm": "Rade_R18",
    "Rebuffi2021Fixing_28_10_cutmix_ddpm": "Rebuffi_28_ddpm",
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": "Rebuffi_70_ddpm",
    "Rebuffi2021Fixing_70_16_cutmix_extra": "Rebuffi_extra",
    "Rebuffi2021Fixing_R18_cutmix_ddpm": "Rebuffi_R18",
    "Rice2020Overfitting": "Rice2020",
    "Rony2019Decoupling": "Rony2019",
    "Sehwag2021Proxy": "Sehwag_Proxy",
    "Sehwag2021Proxy_R18": "Sehwag_R18",
    "Wu2020Adversarial": "Wu2020",
}

# =============================================================================
# Reference Values from GREAT Score Paper (Table 2)
# =============================================================================
# RobustBench L2 robust accuracy (%) at eps=0.5
ROBUSTBENCH_ACCURACY = {
    "Augustin2020Adversarial_34_10_extra": 78.79,
    "Augustin2020Adversarial_34_10": 76.25,
    "Augustin2020Adversarial": 72.91,
    "Ding2020MMA": 66.09,
    "Engstrom2019Robustness": 69.24,
    "Gowal2020Uncovering": 74.50,
    "Gowal2020Uncovering_extra": 80.53,
    "Rade2021Helper_R18_ddpm": 76.15,
    "Rebuffi2021Fixing_28_10_cutmix_ddpm": 78.80,
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": 80.42,
    "Rebuffi2021Fixing_70_16_cutmix_extra": 82.32,
    "Rebuffi2021Fixing_R18_cutmix_ddpm": 75.86,
    "Rice2020Overfitting": 67.68,
    "Rony2019Decoupling": 66.44,
    "Sehwag2021Proxy": 77.24,
    "Sehwag2021Proxy_R18": 74.41,
    "Wu2020Adversarial": 73.66,
}

# Original uncalibrated GREAT Score values (Table 2, GAN-generated samples)
ORIGINAL_GREAT_SCORES = {
    "Augustin2020Adversarial_34_10_extra": 0.525,
    "Augustin2020Adversarial_34_10": 0.583,
    "Augustin2020Adversarial": 0.569,
    "Ding2020MMA": 0.112,
    "Engstrom2019Robustness": 0.160,
    "Gowal2020Uncovering": 0.124,
    "Gowal2020Uncovering_extra": 0.534,
    "Rade2021Helper_R18_ddpm": 0.413,
    "Rebuffi2021Fixing_28_10_cutmix_ddpm": 0.424,
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": 0.451,
    "Rebuffi2021Fixing_70_16_cutmix_extra": 0.507,
    "Rebuffi2021Fixing_R18_cutmix_ddpm": 0.369,
    "Rice2020Overfitting": 0.152,
    "Rony2019Decoupling": 0.275,
    "Sehwag2021Proxy": 0.227,
    "Sehwag2021Proxy_R18": 0.236,
    "Wu2020Adversarial": 0.128,
}

# Original GREAT Score on test samples (Appendix, Table 9)
ORIGINAL_GREAT_SCORES_TEST = {
    "Augustin2020Adversarial_34_10_extra": 0.525,
    "Augustin2020Adversarial_34_10": 0.489,
    "Augustin2020Adversarial": 0.493,
    "Ding2020MMA": 0.080,
    "Engstrom2019Robustness": 0.127,
    "Gowal2020Uncovering": 0.109,
    "Gowal2020Uncovering_extra": 0.481,
    "Rade2021Helper_R18_ddpm": 0.331,
    "Rebuffi2021Fixing_28_10_cutmix_ddpm": 0.344,
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": 0.377,
    "Rebuffi2021Fixing_70_16_cutmix_extra": 0.465,
    "Rebuffi2021Fixing_R18_cutmix_ddpm": 0.297,
    "Rice2020Overfitting": 0.120,
    "Rony2019Decoupling": 0.221,
    "Sehwag2021Proxy": 0.227,
    "Sehwag2021Proxy_R18": 0.176,
    "Wu2020Adversarial": 0.106,
}

# CW attack average distortion (Table 2)
CW_DISTORTION = {
    "Augustin2020Adversarial_34_10_extra": 1.340,
    "Augustin2020Adversarial_34_10": 1.332,
    "Augustin2020Adversarial": 1.285,
    "Ding2020MMA": 1.095,
    "Engstrom2019Robustness": 1.084,
    "Gowal2020Uncovering": 1.253,
    "Gowal2020Uncovering_extra": 1.324,
    "Rade2021Helper_R18_ddpm": 1.486,
    "Rebuffi2021Fixing_28_10_cutmix_ddpm": 1.796,
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": 1.943,
    "Rebuffi2021Fixing_70_16_cutmix_extra": 1.859,
    "Rebuffi2021Fixing_R18_cutmix_ddpm": 1.413,
    "Rice2020Overfitting": 1.097,
    "Rony2019Decoupling": 1.165,
    "Sehwag2021Proxy": 1.392,
    "Sehwag2021Proxy_R18": 1.343,
    "Wu2020Adversarial": 1.369,
}

# Calibrated GREAT Score values (Table 2)
CALIBRATED_GREAT_SCORES = {
    "Augustin2020Adversarial_34_10_extra": 1.206,
    "Augustin2020Adversarial_34_10": 1.206,
    "Augustin2020Adversarial": 1.199,
    "Ding2020MMA": 0.909,
    "Engstrom2019Robustness": 1.020,
    "Gowal2020Uncovering": 1.116,
    "Gowal2020Uncovering_extra": 1.213,
    "Rade2021Helper_R18_ddpm": 1.200,
    "Rebuffi2021Fixing_28_10_cutmix_ddpm": 1.214,
    "Rebuffi2021Fixing_70_16_cutmix_ddpm": 1.208,
    "Rebuffi2021Fixing_70_16_cutmix_extra": 1.216,
    "Rebuffi2021Fixing_R18_cutmix_ddpm": 1.210,
    "Rice2020Overfitting": 1.040,
    "Rony2019Decoupling": 1.101,
    "Sehwag2021Proxy": 1.143,
    "Sehwag2021Proxy_R18": 1.135,
    "Wu2020Adversarial": 1.110,
}

# Rank correlations from the paper (Table 2)
PAPER_CORRELATIONS = {
    "uncalibrated_vs_robustbench": 0.6618,
    "uncalibrated_vs_autoattack": 0.3690,
    "calibrated_vs_robustbench": 0.8971,
    "calibrated_vs_autoattack": 0.6941,
    "robustbench_vs_autoattack": 0.7296,
}

# =============================================================================
# Self-Calibration Parameters
# =============================================================================
CALIBRATION_TEMP_MIN = 0.01
CALIBRATION_TEMP_MAX = 5.0
CALIBRATION_TEMP_STEP = 0.01
CALIBRATION_TEMP_FINE_STEP = 0.001

# FP-GREAT default lambda
DEFAULT_FP_LAMBDA = 0.5

# =============================================================================
# Visualization Settings
# =============================================================================
FIGURE_DPI = 300
FIGURE_FONT_FAMILY = "serif"
FIGURE_FORMAT = ["png", "pdf"]

# Color palette for per-class visualizations
CLASS_COLORS = {
    "airplane": "#E74C3C",
    "automobile": "#3498DB",
    "bird": "#2ECC71",
    "cat": "#F39C12",
    "deer": "#9B59B6",
    "dog": "#1ABC9C",
    "frog": "#E67E22",
    "horse": "#34495E",
    "ship": "#16A085",
    "truck": "#C0392B",
}

# Model ranking color maps
HEATMAP_CMAP = "RdYlGn"
DIVERGING_CMAP = "RdBu_r"

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# ImageNet L∞ RobustBench Models (5 models from Table 3 of the paper)
# =============================================================================
IMAGENET_LINF_MODELS = [
    "Salman2020Do_50_2",
    "Salman2020Do_R50",
    "Engstrom2019Robustness",
    "Wong2020Fast",
    "Salman2020Do_R18",
]

IMAGENET_NUM_CLASSES = 1000

IMAGENET_MODEL_SHORT_NAMES = {
    "Salman2020Do_50_2": "Salman_WRN50-2",
    "Salman2020Do_R50": "Salman_R50",
    "Engstrom2019Robustness": "Engstrom2019",
    "Wong2020Fast": "Wong2020",
    "Salman2020Do_R18": "Salman_R18",
}

# RobustBench ImageNet Linf robust accuracy (%) at eps=4/255
IMAGENET_ROBUSTBENCH_ACCURACY = {
    "Salman2020Do_50_2": 38.14,
    "Salman2020Do_R50": 34.96,
    "Engstrom2019Robustness": 29.22,
    "Wong2020Fast": 26.24,
    "Salman2020Do_R18": 25.32,
}

# AutoAttack accuracy on GAN-generated samples (from paper main.py lines 343-348)
IMAGENET_AUTOATTACK_ACCURACY = {
    "Salman2020Do_50_2": 30.40,
    "Salman2020Do_R50": 25.80,
    "Engstrom2019Robustness": 30.60,
    "Wong2020Fast": 19.20,
    "Salman2020Do_R18": 19.60,
}

# Original uncalibrated GREAT Score values for ImageNet (Table 3 of the paper)
IMAGENET_ORIGINAL_GREAT_SCORES = {
    "Salman2020Do_50_2": 0.504,
    "Salman2020Do_R50": 0.443,
    "Engstrom2019Robustness": 0.449,
    "Wong2020Fast": 0.273,
    "Salman2020Do_R18": 0.275,
}

# Rank correlations from the paper for ImageNet (Section 4.3)
# Note: calibration was not performed for ImageNet in the original paper
IMAGENET_PAPER_CORRELATIONS = {
    "uncalibrated_vs_robustbench": 0.800,
    "uncalibrated_vs_autoattack": 0.600,
}


# =============================================================================
# Dataset Configuration Dispatcher
# =============================================================================
def get_dataset_config(dataset_name: str) -> dict:
    """
    Return all dataset-specific constants for the given dataset name.

    Args:
        dataset_name: One of 'cifar10' or 'imagenet'.

    Returns:
        dict with keys:
            - classes: list of class name strings (or None for very large sets)
            - num_classes: int
            - models: list of model name strings
            - short_names: dict mapping full -> short model names
            - robustbench_accuracy: dict mapping model -> accuracy
            - threat_model: str ('L2' or 'Linf')
            - dataset_robustbench: str (for robustbench.utils.load_model)
            - data_dir: Path
            - use_sigmoid: bool (activation function choice from paper)
    """
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return {
            "classes": CIFAR10_CLASSES,
            "num_classes": CIFAR10_NUM_CLASSES,
            "models": CIFAR10_L2_MODELS,
            "short_names": MODEL_SHORT_NAMES,
            "robustbench_accuracy": ROBUSTBENCH_ACCURACY,
            "threat_model": "L2",
            "dataset_robustbench": "cifar10",
            "data_dir": CIFAR10_DIR,
            "use_sigmoid": True,
        }
    elif dataset_name == "imagenet":
        return {
            "classes": None,  # 1000 classes — loaded dynamically from data
            "num_classes": IMAGENET_NUM_CLASSES,
            "models": IMAGENET_LINF_MODELS,
            "short_names": IMAGENET_MODEL_SHORT_NAMES,
            "robustbench_accuracy": IMAGENET_ROBUSTBENCH_ACCURACY,
            "threat_model": "Linf",
            "dataset_robustbench": "imagenet",
            "data_dir": IMAGENET_DIR,
            "use_sigmoid": False,  # Paper uses softmax for ImageNet
        }
    else:
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Supported: 'cifar10', 'imagenet'"
        )
