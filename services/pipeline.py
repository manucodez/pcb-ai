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

Author : Jasmeen
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

    def inspect(self, image_path):

        start_time = time.time()

        # -------------------------------------------------
        # YOLO Detection
        # -------------------------------------------------

        detections = self.detector.predict(image_path)

        # -------------------------------------------------
        # AI Explanation
        # -------------------------------------------------

        explanations = self.engine.generate(

            image_path=image_path,

            detections=detections

        )

        result = {

            "image": image_path.name,

            "total_defects": len(explanations),

            "detections": explanations,

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