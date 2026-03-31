"""
Step 4: Validate TFLite model against all acceptance criteria.

MUST PASS before building the Flutter app:
  [CRITICAL] Confidence fallback: model output <0.60 → "Unable to diagnose"
  [REQUIRED] Test set accuracy ≥ 0.80
  [REQUIRED] CSSVD recall ≥ 0.70
  [REQUIRED] TFLite file size ≤ 5 MB
  [REQUIRED] Inference latency ≤ 100 ms (median over 50 runs)
  [REQUIRED] "Healthy" class precision ≥ 0.90

Input:  models/cacao_classifier.tflite
        models/labels.txt
        data/curated/test/
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
TFLITE_PATH = MODELS_DIR / "cacao_classifier.tflite"
LABELS_PATH = MODELS_DIR / "labels.txt"
TEST_DIR = Path("data/curated/test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 1  # single-image inference (matches mobile usage)

CONFIDENCE_THRESHOLD = 0.60

THRESHOLDS = {
    "accuracy": 0.80,
    "cssvd_recall": 0.70,
    "healthy_precision": 0.90,
    "model_size_mb": 5.0,
    "latency_median_ms": 100.0,
}

# ── Inference helpers ──────────────────────────────────────────────────────────

def load_interpreter(path: Path) -> tf.lite.Interpreter:
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def preprocess(image_path: Path) -> np.ndarray:
    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)


def run_inference(interp: tf.lite.Interpreter, image: np.ndarray) -> np.ndarray:
    input_idx = interp.get_input_details()[0]["index"]
    output_idx = interp.get_output_details()[0]["index"]
    interp.set_tensor(input_idx, image)
    interp.invoke()
    return interp.get_tensor(output_idx)[0]  # shape: (NUM_CLASSES,)


def apply_confidence_threshold(probs: np.ndarray, threshold: float) -> int | None:
    """Return class index if max confidence >= threshold, else None (low confidence)."""
    max_conf = probs.max()
    if max_conf < threshold:
        return None  # → "Unable to diagnose — consult an agronomist"
    return int(probs.argmax())


# ── Validation ─────────────────────────────────────────────────────────────────

def validate():
    print("=== Step 4: Validation ===\n")
    results = {}
    all_passed = True

    def check(name: str, value: float, threshold: float, op: str = ">=") -> bool:
        passed = (value >= threshold) if op == ">=" else (value <= threshold)
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}: {value:.4f}  ({op} {threshold})")
        return passed

    # ── 1. File size ─────────────────────────────────────────────────────────
    size_mb = TFLITE_PATH.stat().st_size / (1024 * 1024)
    passed = check("Model size (MB)", size_mb, THRESHOLDS["model_size_mb"], "<=")
    all_passed &= passed

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print()
    if not TFLITE_PATH.exists():
        raise FileNotFoundError(f"TFLite model not found: {TFLITE_PATH}")
    interp = load_interpreter(TFLITE_PATH)
    class_names = LABELS_PATH.read_text().strip().splitlines()
    print(f"Loaded model. Classes: {class_names}")

    # ── 3. Latency benchmark (50 runs on a blank image) ───────────────────────
    print("\n── Latency benchmark ─────────────────────────────")
    dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
    latencies_ms = []
    for _ in range(50):
        start = time.perf_counter()
        run_inference(interp, dummy)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    print(f"  Median: {median_ms:.1f} ms  |  P95: {p95_ms:.1f} ms")
    passed = check(
        "Inference latency median (ms)",
        median_ms,
        THRESHOLDS["latency_median_ms"],
        "<=",
    )
    all_passed &= passed
    print("  NOTE: This benchmark runs on the dev machine. Field devices will be slower.")
    print("  Run on target hardware (Snapdragon 450) before shipping.\n")

    # ── 4. Accuracy on test set ───────────────────────────────────────────────
    print("── Test set evaluation ───────────────────────────")
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
    y_true, y_pred_raw, y_pred_thresholded = [], [], []
    low_confidence_count = 0

    for class_idx, class_name in enumerate(class_names):
        class_dir = TEST_DIR / class_name
        if not class_dir.exists():
            print(f"  [WARN] Test class dir not found: {class_dir}")
            continue
        image_paths = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        for path in image_paths:
            try:
                img = preprocess(path)
                probs = run_inference(interp, img)
                raw_pred = int(probs.argmax())
                thresholded_pred = apply_confidence_threshold(probs, CONFIDENCE_THRESHOLD)

                y_true.append(class_idx)
                y_pred_raw.append(raw_pred)

                if thresholded_pred is None:
                    low_confidence_count += 1
                    # For accuracy metrics, count low-confidence as incorrect
                    y_pred_thresholded.append(-1)
                else:
                    y_pred_thresholded.append(thresholded_pred)
            except Exception as e:
                print(f"  [SKIP] {path.name}: {e}")

    total = len(y_true)
    print(f"  Evaluated {total} test images")
    print(f"  Low-confidence predictions (<{CONFIDENCE_THRESHOLD}): {low_confidence_count} ({100*low_confidence_count/total:.1f}%)")

    # Accuracy (raw, no threshold)
    acc_raw = accuracy_score(y_true, y_pred_raw)
    passed = check("Test accuracy (raw)", acc_raw, THRESHOLDS["accuracy"])
    all_passed &= passed

    # Per-class metrics
    print("\n── Per-class report (raw predictions) ─────────────")
    print(classification_report(y_true, y_pred_raw, target_names=class_names, zero_division=0))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_raw, labels=list(range(len(class_names))), zero_division=0
    )
    class_metrics = dict(zip(class_names, zip(precision, recall, f1)))

    # CSSVD recall
    print("── Critical thresholds ───────────────────────────")
    if "cssvd" in class_metrics:
        cssvd_recall = class_metrics["cssvd"][1]
        passed = check("CSSVD recall", cssvd_recall, THRESHOLDS["cssvd_recall"])
        all_passed &= passed
        if cssvd_recall < THRESHOLDS["cssvd_recall"]:
            print(
                "  [ACTION] CSSVD recall too low. Options:\n"
                "    1. Collect more CSSVD images\n"
                "    2. Adjust class weights during training (class_weight param)\n"
                "    3. Drop CSSVD from v1, retrain as 4-class model"
            )
    else:
        print("  [SKIP] CSSVD class not in labels — 4-class model")

    # Healthy precision
    if "healthy" in class_metrics:
        healthy_prec = class_metrics["healthy"][0]
        passed = check("Healthy precision", healthy_prec, THRESHOLDS["healthy_precision"])
        all_passed &= passed
        if healthy_prec < THRESHOLDS["healthy_precision"]:
            print(
                "  [ACTION] Healthy precision too low — too many false positives "
                "(healthy pods wrongly classified as diseased). Check training data balance."
            )

    # ── 5. [CRITICAL] Confidence fallback unit test ────────────────────────────
    print("\n── [CRITICAL] Confidence fallback test ───────────")
    # Simulate model returning low-confidence output
    low_conf_probs = np.array([0.25, 0.20, 0.20, 0.18, 0.17], dtype=np.float32)
    result = apply_confidence_threshold(low_conf_probs, CONFIDENCE_THRESHOLD)
    assert result is None, (
        f"CRITICAL: confidence fallback failed — returned {result} instead of None "
        f"for probs {low_conf_probs} (max={low_conf_probs.max():.2f} < {CONFIDENCE_THRESHOLD})"
    )
    print(f"  [OK] Low-confidence input (max={low_conf_probs.max():.2f}) → None (fallback)")

    # Simulate high-confidence output
    high_conf_probs = np.array([0.05, 0.80, 0.05, 0.05, 0.05], dtype=np.float32)
    result = apply_confidence_threshold(high_conf_probs, CONFIDENCE_THRESHOLD)
    assert result == 1, (
        f"CRITICAL: confidence fallback failed — returned {result} instead of 1 "
        f"for probs {high_conf_probs}"
    )
    print(f"  [OK] High-confidence input (max={high_conf_probs.max():.2f}) → class 1 (black_pod)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    if all_passed:
        print("=== ALL CHECKS PASSED — ready for Flutter integration ===")
        print(f"  Copy {TFLITE_PATH} → app/assets/ml/cacao_classifier.tflite")
    else:
        print("=== SOME CHECKS FAILED — do not proceed to Flutter integration ===")
        print("  Review the [FAIL] items above before building the app.")
    print("=" * 50)


if __name__ == "__main__":
    validate()
