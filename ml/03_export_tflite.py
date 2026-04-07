"""
Step 6: Export trained YOLOv8n to TFLite (FP16) for Android offline inference.

Run:
    python ml/03_export_tflite.py
"""

from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path("runs/detect/ml/runs/v1_yolov8n/weights/best.pt")


def main():
    assert WEIGHTS.exists(), f"missing {WEIGHTS} — train first via ml/02_train.py"
    model = YOLO(str(WEIGHTS))
    model.export(format="tflite", half=True, imgsz=640)
    print(f"Exported. Look in {WEIGHTS.parent}/ for best_saved_model/best_float16.tflite")


if __name__ == "__main__":
    main()
