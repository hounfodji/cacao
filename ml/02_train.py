"""
Step 5: Train YOLOv8n on the unified dataset.

Uses ultralytics (official YOLOv8). Reads ml/data/unified/data.yaml produced
by 03_split.py. Outputs weights to ml/runs/v1_yolov8n/weights/best.pt.

Run:
    python ml/02_train.py
"""

from pathlib import Path

from ultralytics import YOLO

DATA = Path("ml/data/unified/data.yaml")
PROJECT = "ml/runs"
NAME = "v1_yolov8n"


def main():
    model = YOLO("yolov8n.pt")  # COCO-pretrained
    model.train(
        data=str(DATA),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        augment=True,
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        project=PROJECT,
        name=NAME,
        seed=42,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
