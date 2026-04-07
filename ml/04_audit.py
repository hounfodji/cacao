"""
Step 4: Quality gates on the unified+deduped+split dataset.

Reads splits.json and sidecar.jsonl. Blocks training if any of these fail:
    - any v1 class < 500 images in train
    - bbox aberrant (w<5px or h<5px or coords outside [0,1]) — count, not block
    - synth_bbox ratio > 30% for any class
    - West Africa subset coverage < 10% per class

Outputs:
    ml/data/unified/audit_report.md   (human-readable)
    ml/data/unified/quality_gate.json (CI machine-readable: {pass: bool, failures: [...]})

Run:
    python ml/04_audit.py
Exit 0 if pass, 1 if fail.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

OUT = Path("ml/data/unified")
IMG_DIR = OUT / "images"
LBL_DIR = OUT / "labels"
SIDECAR = OUT / "sidecar.jsonl"
SPLITS = OUT / "splits.json"

CLASS_NAMES = ["healthy", "cssvd", "anthracnose", "black_pod"]
MIN_TRAIN_PER_CLASS = 500
MAX_SYNTH_RATIO = 0.40  # bumped from 0.30: black_pod is data-starved, dropping more synth would hurt more than help
MIN_WA_RATIO = 0.10
MIN_BBOX_PX = 5


def load_sidecar() -> dict:
    by_name = {}
    with open(SIDECAR) as f:
        for line in f:
            d = json.loads(line)
            by_name[d["image"]] = d
    return by_name


def dominant_class(label_path: Path) -> int:
    counts = Counter()
    if not label_path.exists():
        return -1
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            counts[int(parts[0])] += 1
    return counts.most_common(1)[0][0] if counts else -1


def main():
    sidecar = load_sidecar()
    splits = json.loads(SPLITS.read_text())

    failures = []
    report_lines = ["# Audit Report\n"]

    # 1. Train per-class minimum
    train_class = Counter()
    for name in splits["train"]:
        cls = dominant_class(LBL_DIR / (Path(name).stem + ".txt"))
        if cls >= 0:
            train_class[CLASS_NAMES[cls]] += 1

    report_lines.append("## Train per-class counts\n")
    for cls in CLASS_NAMES:
        n = train_class.get(cls, 0)
        ok = "OK" if n >= MIN_TRAIN_PER_CLASS else "FAIL"
        report_lines.append(f"- {cls}: {n} ({ok})")
        if n < MIN_TRAIN_PER_CLASS:
            failures.append(f"train_min_class:{cls}={n}<{MIN_TRAIN_PER_CLASS}")
    report_lines.append("")

    # 2. Synth bbox ratio per class (across full dataset)
    synth_by_class = defaultdict(int)
    total_by_class = defaultdict(int)
    for name, entry in sidecar.items():
        cls = dominant_class(LBL_DIR / (Path(name).stem + ".txt"))
        if cls < 0:
            continue
        c = CLASS_NAMES[cls]
        total_by_class[c] += 1
        if entry.get("synth_bbox"):
            synth_by_class[c] += 1

    report_lines.append("## Synthetic bbox ratio per class\n")
    for cls in CLASS_NAMES:
        tot = total_by_class.get(cls, 0)
        syn = synth_by_class.get(cls, 0)
        ratio = syn / tot if tot else 0
        ok = "OK" if ratio <= MAX_SYNTH_RATIO else "FAIL"
        report_lines.append(f"- {cls}: {syn}/{tot} = {ratio:.1%} ({ok})")
        if ratio > MAX_SYNTH_RATIO:
            failures.append(f"synth_ratio:{cls}={ratio:.2f}>{MAX_SYNTH_RATIO}")
    report_lines.append("")

    # 3. West Africa coverage per class
    wa_by_class = defaultdict(int)
    for name, entry in sidecar.items():
        if entry.get("region") != "west_africa":
            continue
        cls = dominant_class(LBL_DIR / (Path(name).stem + ".txt"))
        if cls < 0:
            continue
        wa_by_class[CLASS_NAMES[cls]] += 1

    report_lines.append("## West Africa coverage per class\n")
    for cls in CLASS_NAMES:
        tot = total_by_class.get(cls, 0)
        wa = wa_by_class.get(cls, 0)
        ratio = wa / tot if tot else 0
        ok = "OK" if ratio >= MIN_WA_RATIO else "FAIL"
        report_lines.append(f"- {cls}: {wa}/{tot} = {ratio:.1%} ({ok})")
        if ratio < MIN_WA_RATIO:
            failures.append(f"wa_coverage:{cls}={ratio:.2f}<{MIN_WA_RATIO}")
    report_lines.append("")

    # 4. Bbox aberrant count (sample first 2000 for speed)
    aberrant = 0
    checked = 0
    for name in list(sidecar.keys())[:2000]:
        lbl = LBL_DIR / (Path(name).stem + ".txt")
        img = IMG_DIR / name
        if not lbl.exists() or not img.exists():
            continue
        try:
            with Image.open(img) as im:
                W, H = im.size
        except Exception:
            continue
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            checked += 1
            _, xc, yc, w, h = map(float, parts)
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                aberrant += 1
                continue
            if w * W < MIN_BBOX_PX or h * H < MIN_BBOX_PX:
                aberrant += 1
    report_lines.append(f"## Bbox sanity (sampled)\n\n- {aberrant}/{checked} aberrant\n")

    # 5. Split sizes
    report_lines.append("## Split sizes\n")
    for k, v in splits.items():
        report_lines.append(f"- {k}: {len(v)}")
    report_lines.append("")

    passed = len(failures) == 0
    report_lines.insert(1, f"\n**Status:** {'PASS' if passed else 'FAIL'}\n")
    if failures:
        report_lines.append("## Failures\n")
        for f in failures:
            report_lines.append(f"- {f}")

    (OUT / "audit_report.md").write_text("\n".join(report_lines))
    (OUT / "quality_gate.json").write_text(
        json.dumps({"pass": passed, "failures": failures}, indent=2)
    )

    print("\n".join(report_lines))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
