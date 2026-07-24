"""
=========================================================
Pipeline Integration Test

Purpose
-------
Test complete PCB inspection pipeline.

Workflow
--------
Image
    ↓
YOLO Detection
    ↓
Crop Defects
    ↓
Prompt Builder
    ↓
Gemini
    ↓
Final JSON

Author : Jasmeen
=========================================================
"""

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT))
from services.pipeline import PCBInspectionPipeline





# =========================================================
# Test Image
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = ROOT / "test_images" / "test_img.jpg"


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PCB INSPECTION PIPELINE TEST")
    print("=" * 60)

    pipeline = PCBInspectionPipeline()

    result = pipeline.inspect(IMAGE_PATH)

    print("\nPipeline Output\n")

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    print("\n")

    print("=" * 60)
    print("Pipeline Test Completed Successfully")
    print("=" * 60)