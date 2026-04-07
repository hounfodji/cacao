"""
Step 1: Unify all raw datasets into a single YOLO-format detection dataset.

For each source, convert native annotations (Supervisely / YOLO / COCO / folder-class)
into YOLO format with v1 class indices, and drop any out-of-scope classes.

v1 classes:
    0 = healthy
    1 = cssvd
    2 = anthracnose
    3 = black_pod

Folder-class sources (Sykes, Davao, YOLOv4) get a synthetic full-image bbox
(0.5 0.5 1.0 1.0). These are tagged in the sidecar so eval can filter them out.

Output:
    ml/data/unified/
        images/<source>__<original_name>.jpg   (symlink to raw)
        labels/<source>__<original_name>.txt   (YOLO format)
        sidecar.jsonl                          (per-image metadata)
        data.yaml                              (ultralytics dataset config)
        unify_report.md                        (human summary)

Run:
    python ml/01_unify.py
"""

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

RAW = Path("ml/data/raw")
OUT = Path("ml/data/unified")
IMG_OUT = OUT / "images"
LBL_OUT = OUT / "labels"
SIDECAR = OUT / "sidecar.jsonl"

V1_CLASSES = ["healthy", "cssvd", "anthracnose", "black_pod"]
HEALTHY, CSSVD, ANTHRACNOSE, BLACK_POD = 0, 1, 2, 3

# Region tags (West Africa = priority for product eval)
WEST_AFRICA = {"karaagro", "amini", "bpr_podborer"}  # Ghana / multi-Africa / Cameroon
LATAM = {"yolov8n_combined", "cocoa_localization_peru", "yolov4", "sykes"}
ASIA = {"davao"}

# Synthetic full-image bbox in YOLO normalized format
SYNTH_BBOX = (0.5, 0.5, 1.0, 1.0)

# ── Helpers ───────────────────────────────────────────────────────────────────


class Unifier:
    def __init__(self):
        self.stats = defaultdict(lambda: Counter())  # source -> Counter(class -> count)
        self.skipped = defaultdict(Counter)          # source -> Counter(reason -> count)
        self.sidecar_lines = []
        self._used_names = set()                     # disambiguate stem collisions

    def emit(self, source: str, src_img: Path, bboxes: list, synth: bool = False):
        """
        bboxes: list of (cls_idx, xc, yc, w, h) — all normalized [0,1].
        Writes symlink + label file + sidecar entry.
        """
        if not src_img.exists():
            self.skipped[source]["missing_image"] += 1
            return
        if not bboxes:
            self.skipped[source]["no_valid_bbox"] += 1
            return

        # Validate bboxes
        valid = []
        for cls, xc, yc, w, h in bboxes:
            if not (0 < w <= 1 and 0 < h <= 1 and 0 <= xc <= 1 and 0 <= yc <= 1):
                continue
            if w < 0.005 or h < 0.005:  # < 5px on a 1000px image
                continue
            valid.append((cls, xc, yc, w, h))
        if not valid:
            self.skipped[source]["invalid_bbox"] += 1
            return

        # Sanitize stem (no spaces, no special chars)
        stem = src_img.stem.replace(" ", "_").replace("__", "_")
        base_name = f"{source}__{stem}"
        # Disambiguate collisions (e.g. amini same ID in train + val)
        candidate = base_name
        i = 1
        while candidate in self._used_names:
            candidate = f"{base_name}_{i}"
            i += 1
        self._used_names.add(candidate)
        out_name = f"{candidate}{src_img.suffix.lower()}"
        img_out = IMG_OUT / out_name
        lbl_out = LBL_OUT / (candidate + ".txt")
        img_out.symlink_to(src_img.resolve())

        with open(lbl_out, "w") as f:
            for cls, xc, yc, w, h in valid:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                self.stats[source][V1_CLASSES[cls]] += 1

        region = (
            "west_africa" if source in WEST_AFRICA
            else "latam" if source in LATAM
            else "asia" if source in ASIA
            else "unknown"
        )
        self.sidecar_lines.append({
            "image": out_name,
            "source": source,
            "src_path": str(src_img),
            "n_objects": len(valid),
            "synth_bbox": synth,
            "region": region,
        })

    # ── Source converters ────────────────────────────────────────────────────

    def unify_karaagro(self):
        source = "karaagro"
        ann_dir = RAW / "karaagro/karaagro-ai-cocoa/ds/ann"
        img_dir = RAW / "karaagro/karaagro-ai-cocoa/ds/img"

        cls_map = {
            "cocoa-swollen-shoot-virus-leaf": CSSVD,
            "cocoa-swollen-shoot-virus-pod": CSSVD,
            "cocoa-swollen-shoot-virus-stem": CSSVD,
            "anthracnose-leaf": ANTHRACNOSE,
            "anthracnose pod": ANTHRACNOSE,
            "healthy-cocoa": HEALTHY,
            "healthy-cocoa-leaf": HEALTHY,
            "healthy-cocoa-pod": HEALTHY,
        }

        for ann_file in ann_dir.glob("*.json"):
            try:
                d = json.loads(ann_file.read_text())
            except json.JSONDecodeError:
                self.skipped[source]["bad_json"] += 1
                continue

            W, H = d["size"]["width"], d["size"]["height"]
            if W <= 0 or H <= 0:
                self.skipped[source]["bad_size"] += 1
                continue

            bboxes = []
            for obj in d.get("objects", []):
                if obj.get("geometryType") != "rectangle":
                    continue
                cls_title = obj.get("classTitle")
                if cls_title not in cls_map:
                    continue
                pts = obj.get("points", {}).get("exterior", [])
                if len(pts) < 2:
                    continue
                (x1, y1), (x2, y2) = pts[0], pts[1]
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                xc = (x1 + x2) / 2 / W
                yc = (y1 + y2) / 2 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                bboxes.append((cls_map[cls_title], xc, yc, w, h))

            # KaraAgroAI ann file name = image name minus .json
            img_name = ann_file.stem
            src_img = img_dir / img_name
            if not src_img.exists():
                # Try common extensions
                for ext in [".jpg", ".jpeg", ".JPG", ".png"]:
                    if (img_dir / f"{ann_file.stem}{ext}").exists():
                        src_img = img_dir / f"{ann_file.stem}{ext}"
                        break

            self.emit(source, src_img, bboxes)

    def unify_yolov8n_combined(self):
        source = "yolov8n_combined"
        base = RAW / "yolov8n_combined/combined_dataset"
        # classes.txt: 0 Anthracnose, 1 CSSVD, 2 Phytophthora, 3 Monilia, 4 Healthy
        cls_map = {0: ANTHRACNOSE, 1: CSSVD, 2: BLACK_POD, 4: HEALTHY}  # 3 (Monilia) dropped

        for split in ["train", "val"]:  # test has no labels
            img_dir = base / "images" / split
            lbl_dir = base / "labels" / split
            if not lbl_dir.exists():
                continue
            for lbl_file in lbl_dir.glob("*.txt"):
                bboxes = []
                for line in lbl_file.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    if cls not in cls_map:
                        continue
                    xc, yc, w, h = map(float, parts[1:])
                    bboxes.append((cls_map[cls], xc, yc, w, h))
                # find matching image
                src_img = None
                for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
                    candidate = img_dir / f"{lbl_file.stem}{ext}"
                    if candidate.exists():
                        src_img = candidate
                        break
                if src_img is None:
                    self.skipped[source]["missing_image"] += 1
                    continue
                self.emit(source, src_img, bboxes)

    def unify_amini(self):
        source = "amini"
        csv_path = RAW / "amini/Train.csv"
        img_root = RAW / "amini"
        cls_map = {"healthy": HEALTHY, "cssvd": CSSVD, "anthracnose": ANTHRACNOSE}

        # Group by image
        from PIL import Image
        per_image = defaultdict(list)  # image_path -> list of (cls, xmin, ymin, xmax, ymax)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls_name = row["class"].strip().lower()
                if cls_name not in cls_map:
                    continue
                try:
                    xmin = float(row["xmin"])
                    ymin = float(row["ymin"])
                    xmax = float(row["xmax"])
                    ymax = float(row["ymax"])
                except (ValueError, KeyError):
                    continue
                if xmax <= xmin or ymax <= ymin:
                    continue
                per_image[row["ImagePath"]].append(
                    (cls_map[cls_name], xmin, ymin, xmax, ymax)
                )

        for img_rel, raw_boxes in per_image.items():
            src_img = img_root / img_rel
            if not src_img.exists():
                self.skipped[source]["missing_image"] += 1
                continue
            try:
                with Image.open(src_img) as im:
                    W, H = im.size
            except Exception:
                self.skipped[source]["unreadable_image"] += 1
                continue
            bboxes = []
            for cls, x1, y1, x2, y2 in raw_boxes:
                xc = (x1 + x2) / 2 / W
                yc = (y1 + y2) / 2 / H
                w = (x2 - x1) / W
                h = (y2 - y1) / H
                bboxes.append((cls, xc, yc, w, h))
            self.emit(source, src_img, bboxes)

    def unify_peru(self):
        source = "cocoa_localization_peru"
        base = RAW / "cocoa_localization_peru/cocoa_diseases"
        # YOLO class indices in this dataset: 0=Health, 4=phytophthora palmivora
        cls_map = {0: HEALTHY, 4: BLACK_POD}

        img_dir = base / "images"
        lbl_dir = base / "labels"
        for lbl_file in lbl_dir.glob("*.txt"):
            bboxes = []
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                if cls not in cls_map:
                    continue
                xc, yc, w, h = map(float, parts[1:])
                bboxes.append((cls_map[cls], xc, yc, w, h))
            src_img = None
            for ext in [".jpg", ".JPG", ".jpeg", ".png"]:
                candidate = img_dir / f"{lbl_file.stem}{ext}"
                if candidate.exists():
                    src_img = candidate
                    break
            if src_img is None:
                self.skipped[source]["missing_image"] += 1
                continue
            self.emit(source, src_img, bboxes)

    def unify_bpr_podborer(self):
        source = "bpr_podborer"
        # COCO categories: 1=Cocoa (drop), 2=Pod Borer (drop v2), 3=Black Pod Rot
        cls_map = {3: BLACK_POD}

        for split in ["train", "val"]:
            coco_path = RAW / f"bpr_podborer/{split}/COCO_{split}.json"
            if not coco_path.exists():
                continue
            d = json.loads(coco_path.read_text())
            img_dir = RAW / f"bpr_podborer/{split}/images"

            img_by_id = {im["id"]: im for im in d.get("images", [])}
            ann_by_img = defaultdict(list)
            for ann in d.get("annotations", []):
                cat = ann["category_id"]
                if cat not in cls_map:
                    continue
                ann_by_img[ann["image_id"]].append(ann)

            for img_id, anns in ann_by_img.items():
                im_info = img_by_id.get(img_id)
                if not im_info:
                    continue
                W = im_info["width"]
                H = im_info["height"]
                src_img = img_dir / im_info["file_name"]
                if not src_img.exists():
                    # try without leading paths
                    src_img = img_dir / Path(im_info["file_name"]).name
                bboxes = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]  # COCO: xywh top-left
                    xc = (x + w / 2) / W
                    yc = (y + h / 2) / H
                    bboxes.append((cls_map[ann["category_id"]], xc, yc, w / W, h / H))
                self.emit(source, src_img, bboxes)

    def unify_folder_source(
        self, source: str, base: Path, folder_to_class: dict
    ):
        """For folder-per-class datasets, emit synthetic full-image bboxes."""
        for folder, cls in folder_to_class.items():
            d = base / folder
            if not d.exists():
                continue
            for img in d.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                self.emit(source, img, [(cls, *SYNTH_BBOX)], synth=True)

    def unify_sykes(self):
        # Use train + val + test (we'll re-split later)
        for split in ["train", "val", "test"]:
            base = RAW / f"sykes/cocoa_disease_images/{split}"
            self.unify_folder_source(
                "sykes", base,
                {"Healthy": HEALTHY, "BPR": BLACK_POD},
            )

    def unify_davao(self):
        base = RAW / "davao/cacao_diseases/cacao_photos"
        self.unify_folder_source(
            "davao", base,
            {"healthy": HEALTHY, "black_pod_rot": BLACK_POD},
        )

    def unify_yolov4(self):
        base = RAW / "yolov4/Enfermedades Cacao"
        self.unify_folder_source(
            "yolov4", base,
            {"Sana": HEALTHY, "Fito": BLACK_POD},
        )

    # ── Outputs ──────────────────────────────────────────────────────────────

    def write_outputs(self):
        # sidecar
        with open(SIDECAR, "w") as f:
            for line in self.sidecar_lines:
                f.write(json.dumps(line) + "\n")

        # data.yaml (ultralytics)
        yaml = OUT / "data.yaml"
        # Note: train/val/test paths are filled in by 03_split.py
        yaml.write_text(
            "# Generated by 01_unify.py\n"
            "# Splits added by 03_split.py\n"
            f"path: {OUT.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            f"nc: {len(V1_CLASSES)}\n"
            f"names: {V1_CLASSES}\n"
        )

        # report
        report = OUT / "unify_report.md"
        lines = ["# Unify Report\n"]
        lines.append(f"Total images emitted: {len(self.sidecar_lines)}\n")
        lines.append("## Per-source per-class object counts\n")
        lines.append("| source | healthy | cssvd | anthracnose | black_pod | total |")
        lines.append("|---|---|---|---|---|---|")
        totals = Counter()
        for src in sorted(self.stats):
            row = self.stats[src]
            t = sum(row.values())
            totals.update(row)
            lines.append(
                f"| {src} | {row['healthy']} | {row['cssvd']} | "
                f"{row['anthracnose']} | {row['black_pod']} | {t} |"
            )
        t = sum(totals.values())
        lines.append(
            f"| **TOTAL** | **{totals['healthy']}** | **{totals['cssvd']}** | "
            f"**{totals['anthracnose']}** | **{totals['black_pod']}** | **{t}** |"
        )

        lines.append("\n## Skipped (per source)\n")
        for src in sorted(self.skipped):
            for reason, n in self.skipped[src].most_common():
                lines.append(f"- {src}: {reason} = {n}")

        lines.append("\n## Synthetic bbox ratio (folder-class sources)\n")
        synth_count = sum(1 for s in self.sidecar_lines if s["synth_bbox"])
        lines.append(
            f"Synthetic bbox images: {synth_count} / {len(self.sidecar_lines)} "
            f"({100*synth_count/max(1,len(self.sidecar_lines)):.1f}%)\n"
        )

        report.write_text("\n".join(lines))
        print(f"\nReport written: {report}")
        print((OUT / "unify_report.md").read_text())


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    if OUT.exists():
        print(f"Cleaning {OUT}/ ...")
        shutil.rmtree(OUT)
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    LBL_OUT.mkdir(parents=True, exist_ok=True)

    u = Unifier()

    sources = [
        ("karaagro", u.unify_karaagro),
        ("yolov8n_combined", u.unify_yolov8n_combined),
        ("amini", u.unify_amini),
        ("cocoa_localization_peru", u.unify_peru),
        ("bpr_podborer", u.unify_bpr_podborer),
        ("sykes", u.unify_sykes),
        ("davao", u.unify_davao),
        ("yolov4", u.unify_yolov4),
    ]
    for name, fn in sources:
        print(f"\n=== Unifying {name} ===")
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()

    u.write_outputs()


if __name__ == "__main__":
    main()
