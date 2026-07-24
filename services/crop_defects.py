"""
=========================================================
PCB Defect Cropper

Purpose
-------
Extract defect regions detected by YOLO.

Input
-----
Original PCB Image
Detection JSON

Output
------
List of Cropped Defect Images

Author : Jasmeen
=========================================================
"""

from pathlib import Path
import cv2


class DefectCropper:

    def __init__(self, padding=10):
        self.padding = padding

    def crop_defects(self, image_path, detections):

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        h, w = image.shape[:2]

        crops = []

        for idx, defect in enumerate(detections):

            box = defect["bbox"]

            x1 = max(0, int(box["x1"]) - self.padding)
            y1 = max(0, int(box["y1"]) - self.padding)

            x2 = min(w, int(box["x2"]) + self.padding)
            y2 = min(h, int(box["y2"]) + self.padding)

            crop = image[y1:y2, x1:x2]

            crops.append(
                {
                    "id": idx + 1,
                    "class": defect["class"],
                    "confidence": defect["confidence"],
                    "bbox": box,
                    "image": crop,
                }
            )

        return crops