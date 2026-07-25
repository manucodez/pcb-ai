"""
=========================================================
PCB Inspection Pipeline

Purpose
-------
Complete PCB defect inspection pipeline.

Workflow
--------
PCB Image
    ↓
YOLO Detection
    ↓
Crop Defects
    ↓
AI Explanation
    ↓
Final JSON

Author : Manjeet
=========================================================
"""

import time

from training.inference import PCBDetector
from services.explanation_engine import ExplanationEngine


class PCBInspectionPipeline:

    def __init__(self):

        self.detector = PCBDetector()

        self.engine = ExplanationEngine()

    # =====================================================
    # Inspect PCB
    # =====================================================

    def inspect(self, image_path, confidence=0.50, iou=0.40, imgsz=1280, augment=False):

        start_time = time.time()

        # -------------------------------------------------
        # YOLO Detection
        # -------------------------------------------------

        detections = self.detector.predict(
            image_path,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            augment=augment,
        )

        # -------------------------------------------------
        # AI Explanation
        # -------------------------------------------------

        explanations = self.engine.generate(
            image_path=image_path,
            detections=detections,
        ) if detections else []

        result = {

            "image": image_path.name,

            "total_defects": len(explanations),

            "detections": explanations,

            "scan_settings": {

                "confidence": confidence,
                "iou": iou,
                "imgsz": imgsz,
                "augment": augment,

            },

            "pipeline_time_sec": round(

                time.time() - start_time,

                2

            )

        }

        print()

        print("=" * 60)

        print(
            f"Pipeline completed in "
            f"{result['pipeline_time_sec']} sec"
        )

        print("=" * 60)

        return result
