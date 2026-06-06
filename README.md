# Cacao — Offline Cacao Disease Detection

An offline-first, on-device computer-vision pipeline that detects cacao pod and leaf diseases from a phone photo, built for West African smallholder farmers on low-end Android hardware.

## Overview

No production-quality, offline cacao disease app exists for West Africa — the only Play Store option (DR CACAO) targets Latin American diseases and requires internet. This project closes that gap with a full training-to-device pipeline: harmonize eight heterogeneous public datasets into one detection corpus, train a compact YOLOv8n detector, export it to TensorFlow Lite (FP16), and run it fully offline inside a Flutter Android app that draws bounding boxes and shows agronomic advice in French. Target hardware is deliberately constrained: 2–3 GB RAM, no GPU/NPU guarantee, Android 7+.

The detector covers four classes relevant to the region: `healthy`, `cssvd` (Cocoa Swollen Shoot Virus Disease), `anthracnose`, and `black_pod` (Phytophthora pod rot).

## Technical approach

This is the core of the project. Everything below the model and the datasets — the data unification, deduplication, stratified splitting, quality gates, regional evaluation, and the entire Dart inference/postprocessing stack — was built for this repo. The model architecture (Ultralytics YOLOv8n), the TFLite/XNNPACK runtimes, and the source datasets are third-party.

### Data pipeline (`ml/`)

A numbered, reproducible pipeline turns raw public datasets into one YOLO-format detection set:

1. **`00_download_datasets.py`** — automates download of KaraAgroAI (Harvard Dataverse, ~17.7k images, West Africa) and the Sykes Ecuador set (OSF, BPR + Healthy folders only). The other six sources below are obtained manually.
2. **`01_unify.py`** — the heart of the data work. Ingests **eight** sources in four different annotation formats (Supervisely JSON, YOLO `.txt`, COCO JSON, and folder-per-class) and remaps every label into a single 4-class taxonomy, dropping out-of-scope classes (e.g. Monilia / Witches' Broom, which do not occur in West Africa). Sources: `karaagro`, `yolov8n_combined`, `amini`, `cocoa_localization_peru`, `bpr_podborer`, `sykes`, `davao`, `yolov4`. Folder-class sources (no boxes) get a synthetic full-image bbox, **tagged in a sidecar** so evaluation can exclude them. Each image carries a region tag (`west_africa` / `latam` / `asia`) used downstream.
3. **`02_dedup.py`** — perceptual-hash (pHash) dedup with LSH-style bucketing and a Hamming-distance ≤ 5 threshold. Because several sources re-package each other, this prevents train/test leakage. Tie-breaking keeps the best-annotated copy (native bbox > synthetic, more objects, West-Africa origin preferred).
4. **`03_split.py`** — stratified 70/15/15 split by `(dominant_class, source)` so the test set is never dominated by a single geography, preserving a usable West-Africa evaluation slice (seed 42).
5. **`04_audit.py`** — pre-training quality gates that exit non-zero on failure: ≥ 500 train images per class, synthetic-bbox ratio ≤ 40% per class, West-Africa coverage ≥ 10% per class, plus a bbox-sanity sample. Emits `audit_report.md` + `quality_gate.json`.

### Training & export

- **`02_train.py`** — fine-tunes **YOLOv8n** (Ultralytics, COCO-pretrained `yolov8n.pt`) for 100 epochs at 640×640, batch 16, `patience=15`, seed 42, with augmentation (mosaic 0.5, mixup 0.1, HSV jitter, ±10° rotation, translate/scale). This is full-model transfer learning, not adapter/LoRA fine-tuning.
- **`03_export_tflite.py`** — exports the best checkpoint to **TFLite FP16** at 640×640 (via the ONNX → `onnx2tf` path defined in `requirements-export.txt`) for offline ARM inference. The exported model is committed at `app/assets/models/best_float16.tflite` (6.2 MB).

### Regional evaluation (`04_validate.py`)

Validation goes beyond a single global mAP. It computes mAP@0.5 on three test slices — **global**, **West-Africa subset** (`karaagro` + `amini` + `bpr_podborer`, the product-relevant number), and **native-bbox only** (excludes synthetic full-image boxes, as a sanity check) — and checks them against acceptance gates (global ≥ 0.65, West-Africa ≥ 0.60, per-class ≥ 0.55), writing `validation_report.json`.

### On-device inference (`app/`)

A Flutter Android app runs the model entirely offline. The inference stack is hand-written in Dart and unit-tested:

- **`yolo_inference.dart`** — loads the FP16 model via `tflite_flutter` with the **XNNPACK CPU delegate** (no NNAPI/GPU — chosen for reliability on old SoCs). Letterboxes the photo into 640×640 with gray padding, normalizes to NHWC float32 `[0,1]`, and auto-detects the output tensor layout (`(1, 8, 8400)` vs `(1, 8400, 8)`). Confidence threshold 0.25, IoU 0.45.
- **`coordinate_mapping.dart`** — pure functions mapping between model-640, source-photo, and display-canvas spaces (the one place an xy inversion could silently break the app — covered by tests).
- **`nms.dart`** — pure-Dart per-class non-max suppression.
- **UI** — `camera_screen` → capture → infer → `result_screen` draws boxes (`bbox_overlay.dart`) and shows per-disease French advice from `advice/advice_fr.json`.

## Results / status

**Work in progress.** A model has been trained and exported, and the FP16 TFLite artifact is committed in the repo.

- **Reported metrics** (from the training run, recorded in `app/README.md`): mAP@0.5 ≈ **0.803** global, **0.751** on the West-Africa subset — both above the project's acceptance gates. These numbers are **not reproducible from the repository alone**: the datasets, training `runs/`, `.pt` weights, and `validation_report.json` are gitignored, so the trained `best_float16.tflite` is the only model artifact checked in.
- **On-device validation is the open blocking gate.** Per `TODOS.md`, the end-to-end bounding-box sanity check on a real Android device (verifying boxes land on the pod, not in a random corner) has **not yet been completed**. On-device inference latency on target hardware has not been measured in this repo.
- The Dart inference logic (coordinate mapping, NMS, YOLO output decoding) is covered by unit tests in `app/test/` that run without a device.

## Tech stack

- **ML:** Python, Ultralytics YOLOv8, PyTorch, TensorFlow Lite, `onnx2tf`, Pillow, `imagehash`, scikit-learn
- **App:** Flutter / Dart, `tflite_flutter` (XNNPACK delegate), `camera`, `image`
- **Model:** YOLOv8n, fine-tuned, exported to TFLite FP16 (640×640, 4 classes)

## Repository structure

```
ml/                       Data -> training -> export -> validation pipeline
  00_download_datasets.py   Download KaraAgroAI + Sykes (other sources manual)
  01_unify.py               Merge 8 sources into one 4-class YOLO dataset
  02_dedup.py               pHash near-duplicate removal (anti-leakage)
  03_split.py               Stratified 70/15/15 split by (class, source)
  04_audit.py               Pre-training quality gates
  02_train.py               Fine-tune YOLOv8n
  03_export_tflite.py       Export to TFLite FP16
  04_validate.py            Regional mAP slices + acceptance gates
app/                      Flutter Android app (offline inference)
  lib/services/             yolo_inference, coordinate_mapping, nms
  lib/screens/, widgets/    camera, result, bbox overlay
  assets/models/            best_float16.tflite (committed, 6.2 MB)
  test/                     Dart unit tests (no device needed)
cacao_research.md         Landscape analysis & technical rationale
datasets.md               Dataset notes and references
TODOS.md                  Roadmap and open gates
```

> Note: the `ml/` script step numbers are not strictly sequential (two `02_`/`03_` files exist). Run them in the documented pipeline order: unify → dedup → split → audit → train → export → validate.

## Setup & usage

### ML pipeline

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Acquire data (KaraAgroAI + Sykes automated; other 6 sources manual)
python 00_download_datasets.py

# 2. Build the unified dataset
python 01_unify.py
python 02_dedup.py
python 03_split.py
python 04_audit.py        # exits non-zero if quality gates fail

# 3. Train and export
python 02_train.py
sudo apt install cmake && pip install -r requirements-export.txt
python 03_export_tflite.py

# 4. Evaluate
python 04_validate.py
```

### Android app

```bash
cd app
# Generate the native Android scaffold (keeps existing lib/ and assets/)
flutter create --platforms=android --org com.cacaoai --project-name cacao .
# Set minSdkVersion 24 in android/app/build.gradle (tflite_flutter + Android 7+)

flutter pub get
flutter test                 # runs the coordinate-mapping / NMS / decode tests
flutter run --release        # deploy to a USB-connected device (--release is required for valid perf)
```

## Limitations / roadmap

- **Not field-validated.** On-device bbox correctness and inference latency on real low-end hardware are unverified (blocking gate in `TODOS.md`).
- **Reproducibility.** Training data and run artifacts are not committed; only the exported TFLite model is. Reproducing the reported metrics requires re-acquiring all eight datasets, six of them manually.
- **Agronomic advice is a stub** (`advice_fr.json`) and must be reviewed by a West-African cacao agronomist (CNRA/CRIG) before any real deployment.
- **French only, Android only, no history** in v1 by design; localization, on-device history, and an inference timeout are planned (see `TODOS.md`).
- **Class scope** is four diseases; `pod_borer`, `mirid`, and `monilia` are deferred pending data.
