"""
Step 3: Convert trained Keras model to TFLite (FP16 quantization).

Input:  models/cacao_classifier.keras
Output: models/cacao_classifier.tflite   (target: <5MB)
        models/labels.txt                (one class name per line)

The exported model is copied to ../app/assets/ if the Flutter app directory exists.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "cacao_classifier.keras"
TFLITE_PATH = MODELS_DIR / "cacao_classifier.tflite"
LABELS_PATH = MODELS_DIR / "labels.txt"

# Flutter app assets directory (relative to ml/ — adjust if needed)
FLUTTER_ASSETS = Path("../app/assets/ml")

CLASS_NAMES = ["healthy", "black_pod", "cssvd", "pod_borer", "mirid"]

# Budget: must stay under this
MAX_MODEL_SIZE_MB = 5.0

# ── Export ─────────────────────────────────────────────────────────────────────

def export():
    print("=== Step 3: TFLite export ===\n")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 02_train.py first."
        )

    model = keras.models.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # FP16 quantization: reduces size ~50% vs float32, no accuracy loss vs full quant
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    print("Converting to TFLite FP16...")
    tflite_model = converter.convert()

    # Write TFLite file
    TFLITE_PATH.write_bytes(tflite_model)

    size_mb = TFLITE_PATH.stat().st_size / (1024 * 1024)
    print(f"TFLite model: {size_mb:.2f} MB  (budget: <{MAX_MODEL_SIZE_MB} MB)")

    if size_mb > MAX_MODEL_SIZE_MB:
        print(
            f"  [FAIL] Model exceeds {MAX_MODEL_SIZE_MB} MB budget.\n"
            "  Options:\n"
            "  1. Apply INT8 quantization (converter.target_spec.supported_ops = "
            "[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]) — needs representative_dataset\n"
            "  2. Reduce NUM_CLASSES (drop CSSVD if data is insufficient)\n"
            "  3. Use MobileNetV3-Small with alpha=0.75"
        )
        raise SystemExit(1)
    else:
        print(f"  [OK] Size within budget")

    # Write labels file
    LABELS_PATH.write_text("\n".join(CLASS_NAMES) + "\n")
    print(f"Labels written to {LABELS_PATH}")

    # Quick correctness check: run one inference on a black image
    print("\nRunning sanity inference on blank image...")
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]  # [1, 224, 224, 3]
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    assert output.shape == (1, len(CLASS_NAMES)), (
        f"Unexpected output shape: {output.shape}"
    )
    assert abs(output.sum() - 1.0) < 1e-3, "Output probabilities don't sum to 1"
    print("  [OK] Sanity inference passed")

    # Copy to Flutter assets if the app directory exists
    if FLUTTER_ASSETS.parent.exists():
        FLUTTER_ASSETS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TFLITE_PATH, FLUTTER_ASSETS / "cacao_classifier.tflite")
        shutil.copy2(LABELS_PATH, FLUTTER_ASSETS / "labels.txt")
        print(f"\nCopied model + labels to {FLUTTER_ASSETS}/")
    else:
        print(
            f"\n[INFO] Flutter app not found at {FLUTTER_ASSETS.parent} — "
            "copy manually when the app directory is created:\n"
            f"  cp {TFLITE_PATH} <app>/assets/ml/\n"
            f"  cp {LABELS_PATH} <app>/assets/ml/"
        )

    print(f"\n=== Export complete ===")
    print(f"  TFLite: {TFLITE_PATH}  ({size_mb:.2f} MB)")
    print(f"  Labels: {LABELS_PATH}")
    print("Next: run 04_validate.py")


if __name__ == "__main__":
    export()
