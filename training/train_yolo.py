"""
=========================================================
PCB Defect Detection

File : train_yolo.py

Purpose
-------
Train YOLO11 for PCB Defect Detection

Input
-----
PCB_Combined Dataset

Output
------
Best YOLO Model
Training Metrics
Loss Curves
Evaluation Results

Author : Jasmeen
=========================================================
"""

from pathlib import Path
import torch
from ultralytics import YOLO

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset" / "PCB_Combined" / "data.yaml"

MODEL = "yolo11s.pt"

OUTPUT = ROOT / "runs"

# ==========================================================
# Training Configuration
# ==========================================================

EPOCHS = 100

IMAGE_SIZE = 640

BATCH_SIZE = 16

WORKERS = 4

PATIENCE = 20

PROJECT = str(OUTPUT)

RUN_NAME = "pcb_detector"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==========================================================


def print_configuration():

    print("=" * 60)
    print("PCB DEFECT DETECTOR TRAINING")
    print("=" * 60)

    print(f"Dataset   : {DATASET}")
    print(f"Model     : {MODEL}")
    print(f"Epochs    : {EPOCHS}")
    print(f"ImageSize : {IMAGE_SIZE}")
    print(f"BatchSize : {BATCH_SIZE}")
    print(f"Workers   : {WORKERS}")
    print(f"Device    : {DEVICE}")

    print("=" * 60)


# ==========================================================


def train():

    model = YOLO(MODEL)

    model.train(

        data=str(DATASET),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        device=DEVICE,

        project=PROJECT,

        name=RUN_NAME,

        patience=PATIENCE,

        pretrained=True,

        optimizer="auto",

        lr0=0.01,

        cos_lr=True,

        cache=True,

        amp=True,

        plots=True,

        save=True,

        save_period=10,

        val=True,

        verbose=True

    )

# ==========================================================


def validate():

    print("\nRunning Validation...\n")

    model = YOLO(
        OUTPUT /
        RUN_NAME /
        "weights" /
        "best.pt"
    )

    metrics = model.val()

    print("\nValidation Complete\n")

    print(metrics)

# ==========================================================


def main():

    print_configuration()

    train()

    validate()

    print("\nTraining Finished Successfully.")

# ==========================================================

if __name__ == "__main__":

    main()