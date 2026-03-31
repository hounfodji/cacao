"""
Step 2: Train MobileNetV3-Small classifier on West African cacao diseases.

Input:  data/curated/{train,val,test}/{healthy,black_pod,cssvd,pod_borer,mirid}/
Output: models/cacao_classifier.keras  (best checkpoint by val_accuracy)
        models/training_history.json

Architecture:
  MobileNetV3-Small (ImageNet weights, frozen base)
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(NUM_CLASSES, activation='softmax')

Training strategy:
  Phase 1 (10 epochs): frozen base, train head only (fast convergence)
  Phase 2 (20 epochs): unfreeze last 30 layers, fine-tune at low LR

Target: val_accuracy ≥ 0.80, CSSVD recall ≥ 0.70
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────

CURATED = Path("data/curated")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Phase 1: train head only
PHASE1_EPOCHS = 10
PHASE1_LR = 1e-3

# Phase 2: fine-tune
PHASE2_EPOCHS = 30
PHASE2_LR = 1e-5
UNFREEZE_LAYERS = 30  # last N layers of base model

CLASS_NAMES = ["healthy", "black_pod", "cssvd", "pod_borer", "mirid"]
NUM_CLASSES = len(CLASS_NAMES)

# ── Data pipeline ──────────────────────────────────────────────────────────────

def make_dataset(split: str, augment: bool = False) -> tf.data.Dataset:
    split_dir = CURATED / split
    ds = keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=42,
    )

    # Normalize to [0, 1] — MobileNetV3 preprocess_input expects this
    normalization = layers.Rescaling(1.0 / 255)

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="augmentation")

    def preprocess(images, labels):
        images = normalization(images)
        if augment:
            images = augmentation(images, training=True)
        return images, labels

    return ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model() -> keras.Model:
    base = keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,  # we handle normalization in the pipeline
    )
    base.trainable = False  # frozen for phase 1

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="cacao_classifier")


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    print("=== Step 2: Training ===\n")

    train_ds = make_dataset("train", augment=True)
    val_ds = make_dataset("val", augment=False)

    model = build_model()
    model.summary()

    checkpoint_path = str(MODELS_DIR / "cacao_classifier.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    # Phase 1: train classification head only
    print("\n── Phase 1: training head (base frozen) ──────────")
    model.compile(
        optimizer=keras.optimizers.Adam(PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks,
    )

    # Phase 2: unfreeze last N layers of base model
    print(f"\n── Phase 2: fine-tuning last {UNFREEZE_LAYERS} layers ──────────")
    base_model = model.get_layer("mobilenet_v3_small")
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks,
    )

    # Save training history
    combined_history = {
        "phase1": {k: [float(v) for v in vs] for k, vs in history1.history.items()},
        "phase2": {k: [float(v) for v in vs] for k, vs in history2.history.items()},
    }
    with open(MODELS_DIR / "training_history.json", "w") as f:
        json.dump(combined_history, f, indent=2)

    # Evaluate on val set
    print("\n── Validation set evaluation ──────────────────────")
    best_model = keras.models.load_model(checkpoint_path)
    val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)
    print(f"  val_accuracy: {val_acc:.4f}  (target: ≥0.80)")

    if val_acc < 0.80:
        print("  [WARN] val_accuracy below 0.80 — check class balance and training data")
    else:
        print("  [OK] Target met")

    # Per-class report on val set
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = best_model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\n── Per-class classification report ────────────────")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # CSSVD recall check
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    cssvd_recall = report.get("cssvd", {}).get("recall", 0.0)
    print(f"  CSSVD recall: {cssvd_recall:.4f}  (target: ≥0.70)")
    if cssvd_recall < 0.70:
        print(
            "  [WARN] CSSVD recall below threshold — "
            "consider dropping CSSVD class or collecting more data"
        )

    print(f"\n=== Model saved to {checkpoint_path} ===")
    print("Next: run 03_export_tflite.py")


if __name__ == "__main__":
    train()
