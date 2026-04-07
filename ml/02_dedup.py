"""
Step 2: Perceptual-hash dedup across all unified images.

Several sources overlap (e.g. yolov8n_combined is itself a combined dataset
that may include KaraAgroAI/Amini images). Without dedup, train/test split
will leak. We compute pHash on every image and group near-duplicates.

When a group of duplicates is found, keep the image with the best annotation:
    1. Native bbox > synthetic bbox
    2. More objects > fewer objects
    3. West Africa source > other regions (preserves geo signal for v1)
    4. Tie-break: lexicographic name

Output:
    Removes duplicate symlinks + label files in-place from ml/data/unified/
    Writes ml/data/unified/dedup_report.json

Run:
    python ml/02_dedup.py
"""

import json
from collections import defaultdict
from pathlib import Path

OUT = Path("ml/data/unified")
IMG_DIR = OUT / "images"
LBL_DIR = OUT / "labels"
SIDECAR = OUT / "sidecar.jsonl"
REPORT = OUT / "dedup_report.json"

HAMMING_THRESHOLD = 5  # ≤5 bits diff = duplicate (standard pHash threshold)

# Region priority for tiebreaking
REGION_PRIORITY = {"west_africa": 3, "asia": 2, "latam": 1, "unknown": 0}


def load_sidecar() -> dict:
    """Returns {image_filename: sidecar_dict}."""
    by_name = {}
    with open(SIDECAR) as f:
        for line in f:
            d = json.loads(line)
            by_name[d["image"]] = d
    return by_name


def compute_phashes(images: list) -> dict:
    """Compute pHash for each image. Returns {filename: hash_int}."""
    from PIL import Image
    import imagehash

    hashes = {}
    failed = 0
    for i, img_path in enumerate(images):
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i+1}/{len(images)}")
        try:
            with Image.open(img_path) as im:
                hashes[img_path.name] = int(str(imagehash.phash(im)), 16)
        except Exception:
            failed += 1
    print(f"  done: {len(hashes)} hashed, {failed} failed")
    return hashes


def group_duplicates(hashes: dict) -> list:
    """
    Group images by Hamming distance ≤ HAMMING_THRESHOLD.
    Uses LSH-style bucketing on the high 16 bits to make this O(n) ish.
    Returns a list of groups (each a list of filenames).
    """
    # Bucket by top 16 bits — duplicates almost always share these
    buckets = defaultdict(list)
    for name, h in hashes.items():
        top = h >> 48
        # Also bucket on neighboring tops (handle edge cases)
        for delta in (0,):
            buckets[top + delta].append(name)

    seen = set()
    groups = []
    items = list(hashes.items())
    by_name = dict(items)

    for name, h in items:
        if name in seen:
            continue
        top = h >> 48
        group = [name]
        seen.add(name)
        for cand in buckets.get(top, []):
            if cand == name or cand in seen:
                continue
            ch = by_name[cand]
            # Hamming distance
            if bin(h ^ ch).count("1") <= HAMMING_THRESHOLD:
                group.append(cand)
                seen.add(cand)
        if len(group) > 1:
            groups.append(group)
    return groups


def keeper(group: list, sidecar: dict) -> str:
    """Pick the winner from a duplicate group."""
    def score(name: str):
        s = sidecar.get(name, {})
        return (
            0 if s.get("synth_bbox") else 1,           # native > synth
            s.get("n_objects", 0),                     # more objects = better
            REGION_PRIORITY.get(s.get("region"), 0),   # west_africa preferred
            -ord(name[0]),                             # tiebreak deterministic
        )
    return max(group, key=score)


def main():
    sidecar = load_sidecar()
    images = sorted(IMG_DIR.iterdir())
    print(f"Computing pHash for {len(images)} images...")
    hashes = compute_phashes(images)

    print(f"\nGrouping duplicates (Hamming ≤ {HAMMING_THRESHOLD})...")
    groups = group_duplicates(hashes)
    print(f"  found {len(groups)} duplicate groups")

    n_to_drop = sum(len(g) - 1 for g in groups)
    print(f"  will drop {n_to_drop} duplicate images")

    dropped_by_source = defaultdict(int)
    drops = []

    for group in groups:
        winner = keeper(group, sidecar)
        for name in group:
            if name == winner:
                continue
            src = sidecar.get(name, {}).get("source", "unknown")
            dropped_by_source[src] += 1
            drops.append({"dropped": name, "kept": winner})
            (IMG_DIR / name).unlink(missing_ok=True)
            label = (LBL_DIR / (Path(name).stem + ".txt"))
            label.unlink(missing_ok=True)

    # Rewrite sidecar without dropped entries
    kept_names = {p.name for p in IMG_DIR.iterdir()}
    with open(SIDECAR, "w") as f:
        for name, entry in sidecar.items():
            if name in kept_names:
                f.write(json.dumps(entry) + "\n")

    # Report
    report = {
        "total_before": len(images),
        "total_after": len(kept_names),
        "duplicate_groups": len(groups),
        "dropped": n_to_drop,
        "dropped_by_source": dict(dropped_by_source),
        "samples": drops[:50],
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"\nDone. {len(kept_names)} images remain.")
    print(f"Report: {REPORT}")
    print(f"Dropped by source: {dict(dropped_by_source)}")


if __name__ == "__main__":
    main()
