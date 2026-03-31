"""
Step 1: Data curation for West African cacao disease classifier.

Datasets used:
- KaraAgroAI: 17,703 images, sub-Saharan Africa context (PRIMARY)
  https://github.com/KaraAgroAI/cocoa-disease-dataset
- Sykes Ecuador: 7,220 images (SUPPLEMENTARY — visual diversity only,
  Latin American disease classes are EXCLUDED)

Output: data/curated/ with train/val/test splits
  data/curated/train/{healthy, black_pod, cssvd, pod_borer, mirid}/
  data/curated/val/{...}/
  data/curated/test/{...}/

Classes (5 total):
  0 - healthy
  1 - black_pod      (Phytophthora palmivora/megakarya)
  2 - cssvd          (Cocoa Swollen Shoot Virus Disease)
  3 - pod_borer      (Conopomorpha cramerella)
  4 - mirid          (Sahlbergella singularis / Distantiella theobroma)

NOTE: CSSVD symptoms are subtle (leaf deformation, stem swelling).
      If CSSVD images in KaraAgroAI are <500, drop the class and train
      on 4 classes (healthy + black_pod + pod_borer + mirid).
"""

import os
import shutil
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_ROOT = Path("data")
RAW_KARAAGRO = DATA_ROOT / "raw" / "karaagro"
RAW_ECUADOR = DATA_ROOT / "raw" / "sykes_ecuador"
CURATED = DATA_ROOT / "curated"

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# West African target classes only.
# Keys = folder names in source dataset (adjust to match actual dataset layout).
# Values = our canonical class names.
KARAAGRO_CLASS_MAP = {
    "healthy": "healthy",
    "black_pod": "black_pod",
    "cssvd": "cssvd",
    "pod_borer": "pod_borer",
    "mirid": "mirid",
    # Add other mappings here if KaraAgroAI uses different folder names.
}

# Ecuador dataset: only use these classes for transfer learning diversity.
# EXCLUDE monilia, witches_broom, and any other Latin American classes.
ECUADOR_ALLOWED_CLASSES = {
    "healthy": "healthy",
    "black_pod": "black_pod",  # present in Ecuador too, safe to include
}

MIN_IMAGES_PER_CLASS = 200  # below this, warn and consider dropping the class
CSSVD_MIN_IMAGES = 500       # CSSVD specifically — drop if below this threshold

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ── Utilities ──────────────────────────────────────────────────────────────────

def collect_images(source_dir: Path, class_map: dict) -> dict[str, list[Path]]:
    """Walk source_dir, return {class_name: [image_paths]} using class_map."""
    result: dict[str, list[Path]] = {}
    if not source_dir.exists():
        print(f"  [WARN] Directory not found: {source_dir}")
        return result

    for folder, class_name in class_map.items():
        folder_path = source_dir / folder
        if not folder_path.exists():
            print(f"  [WARN] Class folder not found: {folder_path}")
            continue
        images = [
            p for p in folder_path.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        result.setdefault(class_name, []).extend(images)

    return result


def verify_image(path: Path) -> bool:
    """Return True if image is readable and non-corrupt."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def check_class_coverage(images_by_class: dict[str, list[Path]]) -> bool:
    """Print class counts and return False if any required class is missing or low."""
    required = {"healthy", "black_pod", "cssvd", "pod_borer", "mirid"}
    ok = True

    print("\n── Class coverage check ──────────────────────────")
    for cls in sorted(required | set(images_by_class.keys())):
        count = len(images_by_class.get(cls, []))
        status = "OK" if count >= MIN_IMAGES_PER_CLASS else "LOW"
        if cls == "cssvd" and count < CSSVD_MIN_IMAGES:
            status = "CRITICAL — consider dropping CSSVD class"
            ok = False
        if cls not in images_by_class:
            status = "MISSING"
            ok = False
        print(f"  {cls:<15} {count:>5} images  [{status}]")

    if not ok:
        print(
            "\n  [ACTION] If CSSVD count < 500, set INCLUDE_CSSVD = False "
            "and retrain with 4 classes."
        )
    print("──────────────────────────────────────────────────\n")
    return ok


def split_and_copy(images_by_class: dict[str, list[Path]], output_dir: Path):
    """Split each class into train/val/test and copy images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        for cls in images_by_class:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls, paths in images_by_class.items():
        random.seed(RANDOM_SEED)
        random.shuffle(paths)

        # Stratified split
        n = len(paths)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])

        train_paths = paths[:n_train]
        val_paths = paths[n_train : n_train + n_val]
        test_paths = paths[n_train + n_val :]

        print(f"  {cls}: {len(train_paths)} train / {len(val_paths)} val / {len(test_paths)} test")

        for split_name, split_paths in [
            ("train", train_paths),
            ("val", val_paths),
            ("test", test_paths),
        ]:
            dest_dir = output_dir / split_name / cls
            for i, src in enumerate(tqdm(split_paths, desc=f"{cls}/{split_name}", leave=False)):
                ext = src.suffix.lower()
                dest = dest_dir / f"{cls}_{i:05d}{ext}"
                shutil.copy2(src, dest)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Step 1: Data curation ===\n")

    # 1. Collect images from KaraAgroAI (primary)
    print("Collecting KaraAgroAI images...")
    karaagro_images = collect_images(RAW_KARAAGRO, KARAAGRO_CLASS_MAP)
    karaagro_total = sum(len(v) for v in karaagro_images.values())
    print(f"  Found {karaagro_total} images across {len(karaagro_images)} classes")

    # 2. Collect allowed images from Ecuador (supplementary)
    print("\nCollecting Sykes Ecuador images (West African classes only)...")
    ecuador_images = collect_images(RAW_ECUADOR, ECUADOR_ALLOWED_CLASSES)
    ecuador_total = sum(len(v) for v in ecuador_images.values())
    print(f"  Found {ecuador_total} images (filtered, Latin American classes excluded)")

    # 3. Merge datasets
    print("\nMerging datasets...")
    merged: dict[str, list[Path]] = {}
    for cls, paths in karaagro_images.items():
        merged[cls] = list(paths)
    for cls, paths in ecuador_images.items():
        merged.setdefault(cls, []).extend(paths)

    # 4. Filter corrupt images
    print("\nVerifying image integrity (this may take a few minutes)...")
    clean: dict[str, list[Path]] = {}
    corrupt_count = 0
    for cls, paths in merged.items():
        valid = []
        for p in tqdm(paths, desc=cls, leave=False):
            if verify_image(p):
                valid.append(p)
            else:
                corrupt_count += 1
        clean[cls] = valid
    print(f"  Removed {corrupt_count} corrupt images")

    # 5. Check coverage — CRITICAL step before training
    coverage_ok = check_class_coverage(clean)
    if not coverage_ok:
        print("[WARN] Coverage issues detected — review before proceeding to training.")
        print("       See CSSVD note in module docstring.\n")

    # 6. Split and copy to curated/
    print(f"Writing curated dataset to {CURATED}/ ...")
    split_and_copy(clean, CURATED)

    # 7. Summary
    total = sum(len(v) for v in clean.values())
    print(f"\n=== Done. {total} images written to {CURATED}/ ===")
    print("Next: run 02_train.py")


if __name__ == "__main__":
    main()
