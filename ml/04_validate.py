"""
Step 7: Validation beyond global mAP.

Computes mAP@0.5 on three slices of the test set, using ultralytics' built-in
validator + custom test.txt files:

    1. Global test set
    2. West Africa subset (karaagro + amini + bpr_podborer) — THE product number
    3. Native bbox only (excludes sykes/davao/yolov4 synth bboxes) — sanity check

Run:
    python ml/04_validate.py
"""

import json
from pathlib import Path

from ultralytics import YOLO

OUT = Path("ml/data/unified")
SIDECAR = OUT / "sidecar.jsonl"
SPLITS = OUT / "splits.json"
WEIGHTS = Path("runs/detect/ml/runs/v1_yolov8n/weights/best.pt")
DATA_TEMPLATE = OUT / "data.yaml"

WEST_AFRICA_SOURCES = {"karaagro", "amini", "bpr_podborer"}

GATES = {
    "global_map50": 0.65,
    "wa_map50": 0.60,
    "per_class_map50": 0.55,
}


def load_sidecar() -> dict:
    by_name = {}
    with open(SIDECAR) as f:
        for line in f:
            d = json.loads(line)
            by_name[d["image"]] = d
    return by_name


def write_subset_yaml(name: str, image_names: list) -> Path:
    """Build a temp data.yaml + .txt that points only at the subset images."""
    abs_img = (OUT / "images").resolve()
    subset_txt = OUT / f"_subset_{name}.txt"
    subset_txt.write_text("\n".join(str(abs_img / n) for n in image_names) + "\n")

    yaml_path = OUT / f"_subset_{name}.yaml"
    new_lines = []
    for line in DATA_TEMPLATE.read_text().splitlines():
        if line.startswith(("test:", "val:")):
            continue
        new_lines.append(line)
    new_lines.append(f"val: _subset_{name}.txt")
    new_lines.append(f"test: _subset_{name}.txt")
    yaml_path.write_text("\n".join(new_lines) + "\n")
    return yaml_path


def run_val(yaml_path: Path, name: str) -> dict:
    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=str(yaml_path),
        split="val",
        project="runs/detect/ml/runs",
        name=f"v1_validate_{name}",
        exist_ok=True,
        verbose=False,
    )
    out = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50": {},
    }
    if hasattr(metrics.box, "ap50"):
        names = list(metrics.names.values())
        for i, ap in enumerate(metrics.box.ap50):
            if i < len(names):
                out["per_class_map50"][names[i]] = float(ap)
    return out


def main():
    assert WEIGHTS.exists(), f"missing {WEIGHTS} — train first"
    sidecar = load_sidecar()
    splits = json.loads(SPLITS.read_text())
    test = splits["test"]
    print(f"Test set: {len(test)} images")

    wa = [n for n in test if sidecar.get(n, {}).get("source") in WEST_AFRICA_SOURCES]
    native = [n for n in test if not sidecar.get(n, {}).get("synth_bbox")]
    print(f"  West Africa subset: {len(wa)}")
    print(f"  Native bbox subset: {len(native)}")

    slices = {
        "global": write_subset_yaml("global", test),
        "wa": write_subset_yaml("wa", wa),
        "native": write_subset_yaml("native", native),
    }

    results = {}
    for name, yaml_path in slices.items():
        print(f"\n=== Evaluating {name} ===")
        results[name] = run_val(yaml_path, name)
        print(f"  mAP@0.5: {results[name]['map50']:.4f}")
        print(f"  mAP@0.5:0.95: {results[name]['map50_95']:.4f}")
        for cls, ap in results[name]["per_class_map50"].items():
            print(f"    {cls}: {ap:.4f}")

    failures = []
    g = results["global"]["map50"]
    if g < GATES["global_map50"]:
        failures.append(f"global_map50={g:.3f}<{GATES['global_map50']}")
    w = results["wa"]["map50"]
    if w < GATES["wa_map50"]:
        failures.append(f"wa_map50={w:.3f}<{GATES['wa_map50']}")
    for cls, ap in results["global"]["per_class_map50"].items():
        if ap < GATES["per_class_map50"]:
            failures.append(f"{cls}_map50={ap:.3f}<{GATES['per_class_map50']}")

    print("\n=== Acceptance gates ===")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASS — all gates met")

    report = {"results": results, "gates": GATES, "failures": failures, "pass": not failures}
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nReport: {OUT / 'validation_report.json'}")

    for f in OUT.glob("_subset_*"):
        f.unlink()


if __name__ == "__main__":
    main()
