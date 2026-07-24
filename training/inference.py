"""
=========================================================
PCB Defect Detection - Inference

Input
-----
PCB Image

Output
------
1. Annotated Image
2. Detection JSON

Author : Jasmeen
=========================================================
"""

import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "runs" / "pcb_detector-3" / "weights" / "best.pt"

TEST_DIR = ROOT / "test_images"

OUTPUT_IMAGE_DIR = ROOT / "outputs" / "images"
OUTPUT_JSON_DIR = ROOT / "outputs" / "json"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Load Model
# =========================================================

print("=" * 60)
print("Loading Model...")
print(MODEL_PATH)
print("=" * 60)

model = YOLO(str(MODEL_PATH))

# =========================================================
# Supported Formats
# =========================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

image_files = sorted(
    [
        file
        for file in TEST_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
)

print(f"\nFound {len(image_files)} image(s)\n")

# =========================================================
# Inference
# =========================================================

for image_path in image_files:

    print("=" * 60)
    print(f"Processing : {image_path.name}")
    print("=" * 60)

    results = model.predict(
        source=str(image_path),
        conf=0.50,
        iou=0.40,
        save=False,
        verbose=False,
    )

    result = results[0]

    annotated = result.plot()

    output_image = OUTPUT_IMAGE_DIR / image_path.name

    cv2.imwrite(str(output_image), annotated)

    detections = []

    if len(result.boxes) > 0:

        print("\nDetected Defects\n")

        for box in result.boxes:

            cls_id = int(box.cls.item())

            class_name = model.names[cls_id]

            confidence = float(box.conf.item())

            x1, y1, x2, y2 = map(float, box.xyxy[0])

            print(f"{class_name:20s} {confidence:.3f}")

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

    else:

        print("\nNo defects detected.")

    json_data = {
        "image": image_path.name,
        "total_defects": len(detections),
        "detections": detections,
    }

    json_path = OUTPUT_JSON_DIR / f"{image_path.stem}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    print(f"\nImage Saved : {output_image}")
    print(f"JSON Saved  : {json_path}")

print("\n" + "=" * 60)
print("Inference Completed Successfully")
print("=" * 60)