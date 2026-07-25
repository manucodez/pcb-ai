"""
=========================================================
PCB Defect Detection

Purpose
-------
YOLO PCB Defect Detector

Input
-----
PCB Image

Output
------
Detection JSON
Annotated Image (Optional)

Author : Jasmeen
=========================================================
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

# =========================================================
# Single source of truth for the model path.
# app.py imports DEFAULT_MODEL_PATH from here instead of
# hardcoding its own copy, so the two can never drift apart.
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = ROOT_DIR / "runs" / "pcb_detector-3" / "weights" / "best.pt"


class PCBDetector:

    def __init__(self, model_path=None):

        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def predict(
        self,
        image_path,
        confidence=0.50,
        iou=0.40,
        imgsz=1280,
        augment=False,
        save_image=False,
        output_dir=None,
    ):
        """
        imgsz
            Inference resolution. PCB defects (mouse bites, spurs,
            pin-holes) are tiny relative to the full board, so running
            inference at a higher resolution than the model was trained
            at (commonly 640) generally improves recall on small defects
            without any retraining. 1280 is a solid default; push to
            1536+ for very high-resolution boards.

        augment
            Enables Ultralytics' built-in test-time augmentation
            (multi-scale + flips, results merged via NMS). Improves
            recall and confidence calibration at the cost of ~2-3x
            inference time. Good for a "thorough scan" mode, not
            necessary for quick checks.
        """

        image_path = Path(image_path)

        results = self.model.predict(
            source=str(image_path),
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            augment=augment,
            save=False,
            verbose=False,
        )

        result = results[0]

        detections = []

        if len(result.boxes) > 0:

            for box in result.boxes:

                cls = int(box.cls.item())

                class_name = self.model.names[cls]

                score = float(box.conf.item())

                x1, y1, x2, y2 = map(
                    float,
                    box.xyxy[0]
                )

                detections.append(
                    {
                        "class": class_name,
                        "confidence": round(score, 3),
                        "bbox":
                        {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        }
                    }
                )

        if save_image:

            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided."
                )

            output_dir = Path(output_dir)

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            annotated = result.plot()

            cv2.imwrite(
                str(output_dir / image_path.name),
                annotated
            )

        return detections
