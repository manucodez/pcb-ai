"""
=========================================================
PCB Defect Cropper

Purpose
-------
Extract defect regions detected by YOLO, and enhance them
for both AI analysis and human viewing.

Why enhance
-----------
Defect boxes are often tiny (a mouse bite or spur can be
20-40px). A raw crop that size is nearly unreadable once
displayed or examined — and a small, low-detail crop gives
Gemini less to work with too. enhance_clarity() boosts local
contrast, upscales small crops with high-quality
interpolation, then sharpens to counter the softness that
upscaling introduces — before the image ever reaches Gemini
or the UI.

Input
-----
Original PCB Image
Detection JSON

Output
------
List of Cropped, Enhanced Defect Images

Author : Manjeet
=========================================================
"""

from pathlib import Path
import cv2


def enhance_clarity(image, target_min_side=320, max_upscale=6.0, clahe_clip=2.0):
    """
    image           BGR numpy crop.
    target_min_side Upscale so the shorter side reaches at least this
                     many pixels (skipped if already larger).
    max_upscale      Cap on the upscale factor, so a 10px sliver
                     doesn't get blown up into a blurry mess.
    clahe_clip       Contrast-limiting factor for local contrast boost.
    """

    if image is None or image.size == 0:
        return image

    # ---------------------------------------------------
    # 1. Local contrast boost (CLAHE on the L channel).
    #    Helps thin copper defects (spurs, mouse bites) stand
    #    out from the surrounding solder mask / substrate.
    # ---------------------------------------------------

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)

    image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # ---------------------------------------------------
    # 2. Upscale small crops so detail is actually visible.
    # ---------------------------------------------------

    h, w = image.shape[:2]
    short_side = min(h, w)

    if short_side and short_side < target_min_side:
        scale = min(target_min_side / short_side, max_upscale)
        image = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    # ---------------------------------------------------
    # 3. Unsharp mask — counters the softness that
    #    upscaling (and phone-camera compression) introduces.
    # ---------------------------------------------------

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    return image


class DefectCropper:

    def __init__(self, padding=15, enhance=True):
        self.padding = padding
        self.enhance = enhance

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

            if self.enhance:
                crop = enhance_clarity(crop)

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
