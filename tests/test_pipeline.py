"""
=========================================================
Pipeline Manual/Live Smoke Test

Purpose
-------
End-to-end sanity check against REAL resources: a trained
model (models/best.pt or runs/*/weights/best.pt), a live
GEMINI_API_KEY, and an image at test_images/test_img.jpg. Not
part of the automated unit-test suite (no assertions, needs
live network + weights) — run it directly:

    python tests/test_pipeline.py

For the automated, dependency-light suite that runs in CI
without any of the above, see the other tests/test_*.py files
(pytest ignores/skips this one automatically when ultralytics
isn't installed).

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

import pytest

# Lets `pytest` skip this file cleanly (instead of erroring) in any
# environment that only has the lightweight unit-test dependencies
# installed, while `python tests/test_pipeline.py` still runs it
# directly for a real end-to-end check.
pytest.importorskip("ultralytics")

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