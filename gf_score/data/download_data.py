"""
Data Download and Preparation
=============================
Downloads CIFAR-10 test data via torchvision and prepares 
class-conditional data for GF-Score evaluation.
Also supports loading pre-generated GAN samples from .npz files.

Usage:
    python -m gf_score.data.download_data [--gan_path PATH]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent to path for direct module execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from gf_score.config import (
    CIFAR10_DIR, IMAGENET_DIR, GAN_DATA_DIR, DATA_DIR, CIFAR10_CLASSES,
    CIFAR10_NUM_CLASSES, IMAGENET_NUM_CLASSES, LOG_FORMAT, LOG_DATE_FORMAT
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("gf_score.data")


def download_cifar10(data_dir: Path = CIFAR10_DIR) -> dict:
    """
    Download the CIFAR-10 test set via torchvision.

    Returns:
        dict with keys:
            - 'images': np.ndarray of shape (N, 3, 32, 32), float32 in [0, 1]
            - 'labels': np.ndarray of shape (N,), int64
            - 'num_samples': int
            - 'num_classes': int
            - 'per_class_counts': dict mapping class_idx -> count
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = data_dir / "cifar10_test_prepared.npz"

    # Check if already prepared
    if prepared_path.exists():
        logger.info(f"Loading prepared CIFAR-10 test data from {prepared_path}")
        data = np.load(prepared_path)
        images = data["images"]
        labels = data["labels"]
        per_class_counts = {}
        for k in range(CIFAR10_NUM_CLASSES):
            per_class_counts[k] = int(np.sum(labels == k))
        logger.info(
            f"Loaded {len(images)} images, "
            f"{CIFAR10_NUM_CLASSES} classes, "
            f"per-class counts: {per_class_counts}"
        )
        return {
            "images": images,
            "labels": labels,
            "num_samples": len(images),
            "num_classes": CIFAR10_NUM_CLASSES,
            "per_class_counts": per_class_counts,
        }

    # Download via torchvision
    logger.info("Downloading CIFAR-10 test set via torchvision...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    # Extract all images and labels
    all_images = []
    all_labels = []
    logger.info(f"Processing {len(dataset)} images...")
    for i in tqdm(range(len(dataset)), desc="Preparing CIFAR-10"):
        img, label = dataset[i]
        all_images.append(img.numpy())
        all_labels.append(label)

    images = np.array(all_images, dtype=np.float32)  # (N, 3, 32, 32)
    labels = np.array(all_labels, dtype=np.int64)     # (N,)

    # Save prepared data
    np.savez_compressed(prepared_path, images=images, labels=labels)
    logger.info(f"Saved prepared data to {prepared_path}")

    # Compute per-class counts
    per_class_counts = {}
    for k in range(CIFAR10_NUM_CLASSES):
        per_class_counts[k] = int(np.sum(labels == k))

    logger.info(
        f"Downloaded {len(images)} images, "
        f"{CIFAR10_NUM_CLASSES} classes"
    )
    for k, name in enumerate(CIFAR10_CLASSES):
        logger.info(f"  Class {k} ({name}): {per_class_counts[k]} samples")

    return {
        "images": images,
        "labels": labels,
        "num_samples": len(images),
        "num_classes": CIFAR10_NUM_CLASSES,
        "per_class_counts": per_class_counts,
    }


def download_imagenet(data_dir: Path = IMAGENET_DIR, max_samples: int = None) -> dict:
    """
    Load the ImageNet (ILSVRC2012) validation set from pre-extracted class folders.

    Expects the validation images to be organized in:
        data_dir/val/<wnid>/<image>.JPEG

    Use scripts/prepare_imagenet.py to extract and organize the raw tar files.

    Args:
        data_dir: Root ImageNet directory containing 'val/' subfolder.
        max_samples: If set, limit total samples (for quick testing).

    Returns:
        dict with keys:
            - 'images': np.ndarray of shape (N, 3, 224, 224), float32 in [0, 1]
            - 'labels': np.ndarray of shape (N,), int64
            - 'num_samples': int
            - 'num_classes': int
            - 'per_class_counts': dict mapping class_idx -> count
            - 'class_names': list of synset WNIDs (str)
    """
    val_dir = data_dir / "val"
    prepared_path = data_dir / "imagenet_val_prepared.npz"
    meta_path = data_dir / "imagenet_val_meta.json"

    # Check if already prepared
    if prepared_path.exists() and meta_path.exists():
        logger.info(f"Loading prepared ImageNet validation data from {prepared_path}")
        data = np.load(prepared_path)
        images = data["images"]
        labels = data["labels"]

        import json as _json
        with open(meta_path, "r") as f:
            meta = _json.load(f)
        class_names = meta["class_names"]
        num_classes = meta["num_classes"]

        per_class_counts = {}
        for k in range(num_classes):
            per_class_counts[k] = int(np.sum(labels == k))

        if max_samples and len(images) > max_samples:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(images), max_samples, replace=False)
            indices.sort()
            images = images[indices]
            labels = labels[indices]
            per_class_counts = {}
            for k in range(num_classes):
                per_class_counts[k] = int(np.sum(labels == k))

        logger.info(
            f"Loaded {len(images)} images, "
            f"{num_classes} classes"
        )
        return {
            "images": images,
            "labels": labels,
            "num_samples": len(images),
            "num_classes": num_classes,
            "per_class_counts": per_class_counts,
            "class_names": class_names,
        }

    # Load from pre-extracted class folders
    if not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet validation directory not found: {val_dir}\n"
            f"Run: python scripts/prepare_imagenet.py"
        )

    logger.info(f"Loading ImageNet validation set from {val_dir}")
    logger.info("This will take a few minutes on first run (resizing to 224x224)...")

    from PIL import Image

    # Discover class folders (sorted for reproducible label assignment)
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    num_classes = len(class_dirs)
    class_names = [d.name for d in class_dirs]

    logger.info(f"Found {num_classes} class directories")

    all_images = []
    all_labels = []

    for k, class_dir in enumerate(class_dirs):
        image_files = sorted(class_dir.glob("*.JPEG"))
        if k % 100 == 0:
            logger.info(f"  Loading class {k}/{num_classes} ({class_dir.name}): {len(image_files)} images")

        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                # Resize to 224x224 (standard ImageNet input)
                img = img.resize((224, 224), Image.BILINEAR)
                # Convert to numpy: (H, W, C) -> (C, H, W), float32 [0, 1]
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))  # CHW
                all_images.append(arr)
                all_labels.append(k)
            except Exception as e:
                logger.warning(f"  Skipping {img_path}: {e}")

    images = np.array(all_images, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    logger.info(f"Loaded {len(images)} images total, saving cache...")

    # Save prepared data
    np.savez_compressed(prepared_path, images=images, labels=labels)

    import json as _json
    meta = {"class_names": class_names, "num_classes": num_classes}
    with open(meta_path, "w") as f:
        _json.dump(meta, f)

    logger.info(f"Saved prepared data to {prepared_path}")

    per_class_counts = {}
    for k in range(num_classes):
        per_class_counts[k] = int(np.sum(labels == k))

    if max_samples and len(images) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(images), max_samples, replace=False)
        indices.sort()
        images = images[indices]
        labels = labels[indices]
        per_class_counts = {}
        for k in range(num_classes):
            per_class_counts[k] = int(np.sum(labels == k))

    return {
        "images": images,
        "labels": labels,
        "num_samples": len(images),
        "num_classes": num_classes,
        "per_class_counts": per_class_counts,
        "class_names": class_names,
    }


def load_gan_generated(npz_path: str) -> dict:
    """
    Load pre-generated GAN samples from a .npz file.

    Expected npz format (matching original GREAT Score codebase):
        - 'x': images array of shape (N, C, H, W) or (N, H, W, C)
        - 'y': labels array of shape (N,)

    Returns:
        dict with same structure as download_cifar10() output.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"GAN data file not found: {npz_path}")

    logger.info(f"Loading GAN-generated samples from {npz_path}")
    data = np.load(str(npz_path))

    images = data["x"].astype(np.float32)
    labels = data["y"].astype(np.int64)

    # Handle channel-last format: (N, H, W, C) -> (N, C, H, W)
    if images.ndim == 4 and images.shape[-1] in (1, 3):
        logger.info("Converting from channel-last to channel-first format")
        images = np.transpose(images, (0, 3, 1, 2))

    # Normalize to [0, 1] if values are in [0, 255]
    if images.max() > 1.0:
        logger.info("Normalizing pixel values from [0, 255] to [0, 1]")
        images = images / 255.0

    num_classes = int(labels.max()) + 1
    per_class_counts = {}
    for k in range(num_classes):
        per_class_counts[k] = int(np.sum(labels == k))

    logger.info(
        f"Loaded {len(images)} GAN-generated images, "
        f"{num_classes} classes"
    )
    for k, count in per_class_counts.items():
        logger.info(f"  Class {k}: {count} samples")

    return {
        "images": images,
        "labels": labels,
        "num_samples": len(images),
        "num_classes": num_classes,
        "per_class_counts": per_class_counts,
    }


def get_class_conditional_data(
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int = CIFAR10_NUM_CLASSES,
    max_per_class: int = None,
    seed: int = 42,
) -> dict:
    """
    Organize data into class-conditional groups.

    Args:
        images: (N, C, H, W) array
        labels: (N,) array
        num_classes: number of classes
        max_per_class: maximum samples per class (None for all)
        seed: random seed for reproducible subsampling

    Returns:
        dict mapping class_idx -> {
            'images': np.ndarray of shape (n_k, C, H, W),
            'labels': np.ndarray of shape (n_k,),
            'count': int,
        }
    """
    rng = np.random.RandomState(seed)
    class_data = {}

    for k in range(num_classes):
        mask = labels == k
        class_images = images[mask]
        class_labels = labels[mask]

        if max_per_class is not None and len(class_images) > max_per_class:
            indices = rng.choice(len(class_images), max_per_class, replace=False)
            indices.sort()
            class_images = class_images[indices]
            class_labels = class_labels[indices]

        class_data[k] = {
            "images": class_images,
            "labels": class_labels,
            "count": len(class_images),
        }

    return class_data


def save_data_summary(data_info: dict, output_path: Path):
    """Save a JSON summary of the loaded data for reproducibility."""
    summary = {
        "num_samples": int(data_info["num_samples"]),
        "num_classes": int(data_info["num_classes"]),
        "per_class_counts": {
            str(k): int(v) for k, v in data_info["per_class_counts"].items()
        },
        "image_shape": list(data_info["images"].shape),
        "pixel_range": [
            float(data_info["images"].min()),
            float(data_info["images"].max()),
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved data summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare data for GF-Score evaluation"
    )
    parser.add_argument(
        "--gan_path",
        type=str,
        default=None,
        help="Path to pre-generated GAN samples (.npz file). "
             "If not provided, downloads CIFAR-10 test set.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum number of samples per class (default: all available)",
    )
    args = parser.parse_args()

    if args.gan_path:
        data_info = load_gan_generated(args.gan_path)
        source = "gan"
    else:
        data_info = download_cifar10()
        source = "cifar10_test"

    # Organize into class-conditional groups
    class_data = get_class_conditional_data(
        data_info["images"],
        data_info["labels"],
        data_info["num_classes"],
        max_per_class=args.max_per_class,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Source:           {source}")
    print(f"Total samples:    {data_info['num_samples']}")
    print(f"Num classes:      {data_info['num_classes']}")
    print(f"Image shape:      {data_info['images'].shape[1:]}")
    print(f"Pixel range:      [{data_info['images'].min():.3f}, {data_info['images'].max():.3f}]")
    print("\nPer-class sample counts:")
    for k in range(data_info["num_classes"]):
        class_name = CIFAR10_CLASSES[k] if k < len(CIFAR10_CLASSES) else f"class_{k}"
        original_count = data_info["per_class_counts"].get(k, 0)
        used_count = class_data[k]["count"]
        print(f"  {k:2d} ({class_name:>10s}): {used_count:5d} / {original_count:5d}")
    print("=" * 60)

    # Save summary
    summary_path = DATA_DIR / f"{source}_data_summary.json"
    save_data_summary(data_info, summary_path)

    print(f"\n✓ Data ready for evaluation. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
