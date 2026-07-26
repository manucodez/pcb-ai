"""
=========================================================
PCB Defect Detection

Purpose
-------
YOLO PCB Defect Detector, plus an optional tiled (sliding-
window) inference mode for maximum recall on high-resolution
board photos.

Input
-----
PCB Image

Output
------
Detection JSON
Annotated Image (Optional)

Author : Manjeet
=========================================================
"""

from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

from training.tiling import generate_tiles, merge_detections_nms, offset_detections
from training.model_paths import (  # noqa: F401 — re-exported for existing callers/tests
    ROOT_DIR,
    RUNS_DIR,
    MODELS_DIR,
    PROMOTED_MODEL_PATH,
    resolve_model_path,
    get_default_model_path,
)

# See training/model_paths.py for the resolution order and the bug
# this fixes (a hardcoded run-folder name that broke on retraining).
DEFAULT_MODEL_PATH = get_default_model_path()


class PCBDetector:

    def __init__(self, model_path=None):

        self.model_path = resolve_model_path(model_path)
        self.model = YOLO(str(self.model_path))

    # =====================================================
    # Single-pass inference
    # =====================================================

    def predict(
        self,
        image_path,
        confidence=0.50,
        iou=0.40,
        imgsz=1280,
        augment=False,
        save_image=False,
        output_dir=None,
        class_conf_overrides: dict[str, float] | None = None,
    ):
        """
        imgsz
            Inference resolution. PCB defects (mouse bites, spurs,
            pin-holes) are tiny relative to the full board, so running
            inference at a higher resolution than the model was trained
            at (commonly 640) generally improves recall on small defects
            without any retraining. 1280 is a solid default; push to
            1536+ for very high-resolution boards, or use predict_tiled()
            for maximum recall on very large images.

        augment
            Enables Ultralytics' built-in test-time augmentation
            (multi-scale + flips, results merged via NMS). Improves
            recall and confidence calibration at the cost of ~2-3x
            inference time. Good for a "thorough scan" mode, not
            necessary for quick checks.

        class_conf_overrides
            Optional {class_name: min_confidence} map for per-class
            thresholds — e.g. keep "short"/"open_circuit" at a low
            threshold (false negatives are expensive) while raising
            the bar for classes more prone to false positives. Any
            class not listed uses `confidence`.
        """

        image_path = Path(image_path)
        results = self._run_model(
            source=str(image_path),
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            augment=augment,
            class_conf_overrides=class_conf_overrides,
        )
        detections = self._extract_detections(results[0], confidence, class_conf_overrides)

        if save_image:
            if output_dir is None:
                raise ValueError("output_dir must be provided.")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            annotated = results[0].plot()
            cv2.imwrite(str(output_dir / image_path.name), annotated)

        return detections

    # =====================================================
    # Tiled (sliding-window) inference — maximum recall
    # =====================================================

    def predict_tiled(
        self,
        image_path,
        confidence=0.50,
        iou=0.40,
        imgsz=1280,
        tile_size=960,
        overlap=0.2,
        merge_iou=0.5,
        class_conf_overrides: dict[str, float] | None = None,
    ):
        """Runs a whole-image pass PLUS inference on overlapping
        tiles of the original (full-resolution) image, then merges
        everything with class-aware NMS.

        Why both a whole-image pass and tiles: tiles alone can split
        a larger defect awkwardly across a tile boundary; the
        whole-image pass alone loses tiny defects to downscaling.
        Combining the two and de-duplicating with NMS recovers both.

        This is meaningfully slower than predict() — proportional to
        tile count — so it's intended as an explicit "maximum recall"
        mode, not the default scan.
        """

        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        h, w = image.shape[:2]

        # Tiling an image no bigger than one tile can't recover any
        # extra detail and only adds inference calls — skip straight
        # to a normal pass.
        if max(h, w) <= tile_size:
            return self.predict(
                image_path, confidence=confidence, iou=iou, imgsz=imgsz,
                class_conf_overrides=class_conf_overrides,
            )

        all_detections = self.predict(
            image_path, confidence=confidence, iou=iou, imgsz=imgsz,
            class_conf_overrides=class_conf_overrides,
        )

        tiles = generate_tiles(w, h, tile_size=tile_size, overlap=overlap)
        for (x0, y0, x1, y1) in tiles:
            crop = image[y0:y1, x0:x1]
            results = self._run_model(
                source=crop,
                confidence=confidence,
                iou=iou,
                imgsz=imgsz,
                augment=False,
                class_conf_overrides=class_conf_overrides,
            )
            tile_detections = self._extract_detections(results[0], confidence, class_conf_overrides)
            all_detections.extend(offset_detections(tile_detections, dx=x0, dy=y0))

        merged = merge_detections_nms(all_detections, iou_threshold=merge_iou)
        merged.sort(key=lambda d: d["confidence"], reverse=True)
        return merged

    # =====================================================
    # Shared helpers
    # =====================================================

    def _run_model(self, source, confidence, iou, imgsz, augment, class_conf_overrides):
        # When per-class overrides are in play, some classes need a
        # lower bar than the global `confidence` — run the model at
        # the lowest threshold actually needed so those boxes survive
        # YOLO's own internal filtering, then apply the real per-class
        # cutoff ourselves in _extract_detections.
        run_conf = confidence
        if class_conf_overrides:
            run_conf = min([confidence, *class_conf_overrides.values()])

        return self.model.predict(
            source=source,
            conf=run_conf,
            iou=iou,
            imgsz=imgsz,
            augment=augment,
            save=False,
            verbose=False,
        )

    def _extract_detections(self, result, confidence, class_conf_overrides):
        detections = []
        if len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            cls = int(box.cls.item())
            class_name = self.model.names[cls]
            score = float(box.conf.item())

            threshold = confidence
            if class_conf_overrides:
                threshold = class_conf_overrides.get(class_name, confidence)
            if score < threshold:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "class": class_name,
                "confidence": round(score, 3),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            })

        return detections
