"""
=========================================================
PCB Inspection Pipeline

Purpose
-------
Complete PCB defect inspection pipeline.

Workflow
--------
PCB Image
    -> YOLO Detection (single-pass or tiled)
    -> Crop Defects
    -> AI Explanation (with rule-based fallback)
    -> Final JSON

Author : Manjeet
=========================================================
"""

from __future__ import annotations

import time
from pathlib import Path

from training.inference import PCBDetector
from services.explanation_engine import ExplanationEngine


class PCBInspectionPipeline:

    def __init__(self):
        self.detector = PCBDetector()
        self.engine = ExplanationEngine()

    # =====================================================
    # Inspect PCB
    # =====================================================

    def inspect(
        self,
        image_path,
        confidence: float = 0.50,
        iou: float = 0.40,
        imgsz: int = 1280,
        augment: bool = False,
        tiled: bool = False,
        class_conf_overrides: dict[str, float] | None = None,
    ) -> dict:

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        start_time = time.time()

        # -------------------------------------------------
        # YOLO Detection
        # -------------------------------------------------

        if tiled:
            detections = self.detector.predict_tiled(
                image_path,
                confidence=confidence,
                iou=iou,
                imgsz=imgsz,
                class_conf_overrides=class_conf_overrides,
            )
        else:
            detections = self.detector.predict(
                image_path,
                confidence=confidence,
                iou=iou,
                imgsz=imgsz,
                augment=augment,
                class_conf_overrides=class_conf_overrides,
            )

        # -------------------------------------------------
        # AI Explanation (falls back to standard guidance
        # per-defect rather than raising — see explanation_engine)
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
                "tiled": tiled,
            },
            "pipeline_time_sec": round(time.time() - start_time, 2),
        }

        print()
        print("=" * 60)
        print(f"Pipeline completed in {result['pipeline_time_sec']} sec")
        print("=" * 60)

        return result
