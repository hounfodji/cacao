"""
Step 3: Stratified train/val/test split by (dominant_class, source).

Stratifying by (class, source) prevents the test set from being 100% one
geography — critical because our West Africa eval subset depends on
karaagro/amini/bpr_podborer being represented in test.

Output:
    ml/data/unified/splits.json    {train: [...], val: [...], test: [...]}
    ml/data/unified/data.yaml      ultralytics-style dataset config
    Also writes train.txt / val.txt / test.txt with absolute image paths.

Run:
    python ml/03_split.py
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("ml/data/unified")
IMG_DIR = OUT / "images"
LBL_DIR = OUT / "labels"
SIDECAR = OUT / "sidecar.jsonl"

SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train, val, test
CLASS_NAMES = ["healthy", "cssvd", "anthracnose", "black_pod"]


def load_sidecar() -> dict:
    by_name = {}
    with open(SIDECAR) as f:
        for line in f:
            d = json.loads(line)
            by_name[d["image"]] = d
    return by_name


def dominant_class(label_path: Path) -> int:
    """Most-frequent class id in the label file (proxy for image-level class)."""
    counts = Counter()
    if not label_path.exists():
        return -1
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            counts[int(parts[0])] += 1
    if not counts:
        return -1
    return counts.most_common(1)[0][0]


def stratified_split(items_by_stratum: dict, ratios: tuple, rng: random.Random):
    """items_by_stratum: {stratum_key: [names]} → (train, val, test) lists."""
    train, val, test = [], [], []
    for stratum, names in items_by_stratum.items():
        names = list(names)
        rng.shuffle(names)
        n = len(names)
        if n == 0:
            continue
        if n < 3:
            # Too few to split — put all in train, warn
            print(f"  WARN: stratum {stratum} has only {n} — all to train")
            train.extend(names)
            continue
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_val = max(1, n - n_train - n_test)
        train.extend(names[:n_train])
        val.extend(names[n_train:n_train + n_val])
        test.extend(names[n_train + n_val:n_train + n_val + n_test])
    return train, val, test


def main():
    sidecar = load_sidecar()
    images = sorted(p.name for p in IMG_DIR.iterdir())
    print(f"Splitting {len(images)} images...")

    # Build strata: (dominant_class, source)
    strata = defaultdict(list)
    for name in images:
        cls = dominant_class(LBL_DIR / (Path(name).stem + ".txt"))
        if cls < 0:
            continue
        src = sidecar.get(name, {}).get("source", "unknown")
        strata[(cls, src)].append(name)

    print(f"  {len(strata)} strata")
    rng = random.Random(SEED)
    train, val, test = stratified_split(strata, RATIOS, rng)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    # Per-class breakdown for sanity
    def class_counts(split):
        c = Counter()
        for name in split:
            cls = dominant_class(LBL_DIR / (Path(name).stem + ".txt"))
            c[CLASS_NAMES[cls]] += 1
        return dict(c)

    print(f"  train classes: {class_counts(train)}")
    print(f"  val   classes: {class_counts(val)}")
    print(f"  test  classes: {class_counts(test)}")

    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    (OUT / "splits.json").write_text(json.dumps(splits, indent=2))

    # Write split list files for ultralytics (absolute paths)
    abs_img = IMG_DIR.resolve()
    for split_name, names in splits.items():
        (OUT / f"{split_name}.txt").write_text(
            "\n".join(str(abs_img / n) for n in names) + "\n"
        )

    # data.yaml for ultralytics
    yaml = (
        f"path: {OUT.resolve()}\n"
        f"train: train.txt\n"
        f"val: val.txt\n"
        f"test: test.txt\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (OUT / "data.yaml").write_text(yaml)
    print(f"\nWrote splits.json, train/val/test.txt, data.yaml")


if __name__ == "__main__":
    main()
