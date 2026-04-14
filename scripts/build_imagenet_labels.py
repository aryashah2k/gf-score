"""
Build a WNID-to-human-readable-name mapping for ImageNet classes.

Reads the devkit meta.mat and produces a JSON file mapping synset WNIDs
(e.g., 'n01756291') to their human-readable names (e.g., 'king snake').

Usage:
    python scripts/build_imagenet_labels.py

Output:
    data/imagenet/wnid_to_name.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def main():
    devkit_dir = ROOT / "data" / "imagenet" / "ILSVRC2012_devkit_t12"
    meta_path = devkit_dir / "data" / "meta.mat"
    output_path = ROOT / "data" / "imagenet" / "wnid_to_name.json"

    if not meta_path.exists():
        print(f"ERROR: meta.mat not found at {meta_path}")
        print("Make sure you have run: python scripts/prepare_imagenet.py")
        sys.exit(1)

    import scipy.io
    meta = scipy.io.loadmat(str(meta_path))
    synsets = meta["synsets"]

    wnid_to_name = {}
    for i in range(synsets.shape[0]):
        entry = synsets[i][0]
        wnid = str(entry[1][0])
        # The 'words' field may contain multiple comma-separated names;
        # take the first one for brevity.
        words = str(entry[2][0])
        short_name = words.split(",")[0].strip()
        wnid_to_name[wnid] = short_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(wnid_to_name, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(wnid_to_name)} WNID-to-name mappings to {output_path}")

    # Show a few examples
    val_dir = ROOT / "data" / "imagenet" / "val"
    if val_dir.exists():
        class_dirs = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])[:10]
        print("\nSample mappings (first 10 val classes):")
        for wnid in class_dirs:
            name = wnid_to_name.get(wnid, "???")
            print(f"  {wnid} -> {name}")


if __name__ == "__main__":
    main()
