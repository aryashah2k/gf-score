"""
ImageNet Validation Set Preparation
====================================
Extracts ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz,
organizes validation images into class-specific subdirectories,
and prepares a cached .npz file for fast loading.

Usage:
    python scripts/prepare_imagenet.py
"""

import os
import sys
import tarfile
import shutil
import scipy.io
from pathlib import Path
from collections import defaultdict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
IMAGENET_DIR = DATA_DIR / "imagenet"
VAL_DIR = IMAGENET_DIR / "val"

VAL_TAR = DATA_DIR / "ILSVRC2012_img_val.tar"
DEVKIT_TAR = DATA_DIR / "ILSVRC2012_devkit_t12.tar.gz"


def extract_devkit():
    """Extract the devkit and parse validation ground truth labels."""
    print("=" * 60)
    print("STEP 1: Extracting devkit for label mappings")
    print("=" * 60)

    devkit_dir = IMAGENET_DIR / "devkit"
    if not devkit_dir.exists():
        print(f"  Extracting {DEVKIT_TAR.name}...")
        with tarfile.open(str(DEVKIT_TAR), "r:gz") as tar:
            tar.extractall(str(IMAGENET_DIR))
        print(f"  Extracted to {devkit_dir}")
    else:
        print(f"  Devkit already extracted at {devkit_dir}")

    # The devkit contains:
    # - ILSVRC2012_devkit_t12/data/meta.mat  (synset metadata)
    # - ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
    devkit_data = IMAGENET_DIR / "ILSVRC2012_devkit_t12" / "data"

    # Read validation ground truth (1-indexed class IDs)
    gt_path = devkit_data / "ILSVRC2012_validation_ground_truth.txt"
    print(f"  Reading ground truth from {gt_path}")
    with open(gt_path, "r") as f:
        val_labels = [int(line.strip()) for line in f.readlines()]
    print(f"  Found {len(val_labels)} validation labels")

    # Read meta.mat for synset -> ILSVRC2012_ID mapping
    meta_path = devkit_data / "meta.mat"
    print(f"  Reading synset metadata from {meta_path}")
    meta = scipy.io.loadmat(str(meta_path), squeeze_me=True)
    synsets = meta["synsets"]

    # Build mapping: ILSVRC2012_ID (1-indexed) -> synset WNID (e.g., "n01440764")
    id_to_wnid = {}
    for entry in synsets:
        ilsvrc_id = int(entry[0])
        wnid = str(entry[1])
        if ilsvrc_id <= 1000:  # Only validation classes
            id_to_wnid[ilsvrc_id] = wnid

    print(f"  Mapped {len(id_to_wnid)} class IDs to synset WNIDs")
    return val_labels, id_to_wnid


def extract_and_organize_val(val_labels, id_to_wnid):
    """Extract validation images and organize into class folders."""
    print("\n" + "=" * 60)
    print("STEP 2: Extracting and organizing validation images")
    print("=" * 60)

    # Check if already organized
    if VAL_DIR.exists():
        subdirs = [d for d in VAL_DIR.iterdir() if d.is_dir()]
        if len(subdirs) >= 1000:
            total_images = sum(len(list(d.glob("*.JPEG"))) for d in subdirs[:5])
            if total_images > 0:
                print(f"  Validation set already organized ({len(subdirs)} class folders)")
                return

    VAL_DIR.mkdir(parents=True, exist_ok=True)

    # Create class subdirectories
    for ilsvrc_id, wnid in id_to_wnid.items():
        (VAL_DIR / wnid).mkdir(exist_ok=True)

    # Extract tar directly into organized structure
    print(f"  Extracting {VAL_TAR.name} ({VAL_TAR.stat().st_size / 1e9:.1f} GB)...")
    print("  This may take a few minutes...")

    with tarfile.open(str(VAL_TAR), "r:") as tar:
        members = tar.getmembers()
        print(f"  Found {len(members)} files in archive")

        # Sort members by name to match ground truth order
        # Validation images are named ILSVRC2012_val_00000001.JPEG through _00050000.JPEG
        image_members = sorted(
            [m for m in members if m.name.endswith(".JPEG")],
            key=lambda m: m.name
        )
        print(f"  Organizing {len(image_members)} images into class folders...")

        for idx, member in enumerate(image_members):
            if idx % 5000 == 0:
                print(f"    Progress: {idx}/{len(image_members)} ({100*idx/len(image_members):.0f}%)")

            # Get label for this image (1-indexed in ground truth)
            label_id = val_labels[idx]
            wnid = id_to_wnid[label_id]

            # Extract to correct class directory
            member_data = tar.extractfile(member)
            if member_data is None:
                continue

            dest_path = VAL_DIR / wnid / os.path.basename(member.name)
            with open(dest_path, "wb") as f:
                f.write(member_data.read())

    # Verify
    class_counts = {}
    for wnid_dir in sorted(VAL_DIR.iterdir()):
        if wnid_dir.is_dir():
            count = len(list(wnid_dir.glob("*.JPEG")))
            class_counts[wnid_dir.name] = count

    total = sum(class_counts.values())
    print(f"\n  Extraction complete!")
    print(f"  Total classes: {len(class_counts)}")
    print(f"  Total images:  {total}")
    print(f"  Images/class:  {total // max(len(class_counts), 1)}")

    return class_counts


def verify_extraction():
    """Quick verification of the organized dataset."""
    print("\n" + "=" * 60)
    print("STEP 3: Verification")
    print("=" * 60)

    if not VAL_DIR.exists():
        print("  ERROR: Validation directory not found!")
        return False

    class_dirs = sorted([d for d in VAL_DIR.iterdir() if d.is_dir()])
    print(f"  Class directories: {len(class_dirs)}")

    if len(class_dirs) != 1000:
        print(f"  WARNING: Expected 1000 classes, got {len(class_dirs)}")

    # Check a few classes
    total = 0
    empty_classes = 0
    for d in class_dirs:
        count = len(list(d.glob("*.JPEG")))
        total += count
        if count == 0:
            empty_classes += 1

    print(f"  Total images: {total}")
    print(f"  Empty classes: {empty_classes}")
    print(f"  Expected: 50,000 images across 1000 classes")

    if total == 50000 and len(class_dirs) == 1000 and empty_classes == 0:
        print("\n  ✓ VERIFICATION PASSED")
        return True
    else:
        print("\n  ✗ VERIFICATION ISSUES DETECTED")
        return False


def main():
    print("ImageNet (ILSVRC2012) Validation Set Preparation")
    print("=" * 60)

    # Check files exist
    if not VAL_TAR.exists():
        print(f"ERROR: {VAL_TAR} not found!")
        print("Please download ILSVRC2012_img_val.tar from image-net.org")
        sys.exit(1)
    if not DEVKIT_TAR.exists():
        print(f"ERROR: {DEVKIT_TAR} not found!")
        print("Please download ILSVRC2012_devkit_t12.tar.gz from image-net.org")
        sys.exit(1)

    print(f"  Val tar:  {VAL_TAR} ({VAL_TAR.stat().st_size / 1e9:.1f} GB)")
    print(f"  Devkit:   {DEVKIT_TAR} ({DEVKIT_TAR.stat().st_size / 1e6:.1f} MB)")
    print()

    # Step 1: Extract devkit
    val_labels, id_to_wnid = extract_devkit()

    # Step 2: Extract and organize images
    extract_and_organize_val(val_labels, id_to_wnid)

    # Step 3: Verify
    success = verify_extraction()

    if success:
        print(f"\n✓ ImageNet validation set ready at: {VAL_DIR}")
        print(f"  You can now run: python -m gf_score.evaluation.run_evaluation --dataset imagenet")
    else:
        print(f"\n⚠ There may be issues with the extraction. Please check manually.")


if __name__ == "__main__":
    main()
