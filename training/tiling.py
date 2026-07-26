"""
=========================================================
Tiled Inference Utilities

Purpose
-------
PCB defects (mouse bites, spurs, pin-holes) can be a few dozen
pixels wide on a photo that's several thousand pixels across.
A single whole-image YOLO pass resizes everything down to
`imgsz` (1280 by default here), so on a high-resolution board
photo a tiny defect can shrink below what the network can
resolve — this is the single biggest recall bottleneck for
small-object detection on large images, well documented in the
tiled/sliced-inference literature (e.g. SAHI).

This module provides the two pure, dependency-light building
blocks tiled inference needs:

  - generate_tiles()       — cover an image with overlapping
                              tiles so no defect near a tile
                              boundary is missed by both
                              neighbors.
  - merge_detections_nms() — collapse duplicate detections of
                              the same defect that show up in
                              more than one overlapping tile (or
                              in both a tile and the whole-image
                              pass), via per-class greedy NMS.

Both functions are plain Python/NumPy — no torch, no
ultralytics — so they can be unit-tested in isolation. The
actual model calls live in training/inference.py, which uses
these as building blocks.

Author : Manjeet
=========================================================
"""

from __future__ import annotations


def generate_tiles(img_w: int, img_h: int, tile_size: int, overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """Covers a (img_w, img_h) image with (x0, y0, x1, y1) tiles
    of `tile_size` pixels, stepping by tile_size * (1 - overlap)
    so adjacent tiles share a border strip wide enough that a
    defect sitting on a boundary is fully contained in at least
    one tile.

    If the image is smaller than tile_size in a dimension, a
    single tile covering that whole dimension is used.
    """

    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    step = max(1, int(tile_size * (1 - overlap)))

    def axis_starts(size: int) -> list[int]:
        if size <= tile_size:
            return [0]
        starts = list(range(0, size - tile_size + 1, step))
        # Make sure the final tile reaches the far edge even if
        # the last step undershoots it.
        if starts[-1] + tile_size < size:
            starts.append(size - tile_size)
        return starts

    x_starts = axis_starts(img_w)
    y_starts = axis_starts(img_h)

    tiles = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, img_w)
            y1 = min(y0 + tile_size, img_h)
            tiles.append((x0, y0, x1, y1))
    return tiles


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def merge_detections_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Greedy, per-class NMS over detections shaped like
    training/inference.py's output (dicts with "class",
    "confidence", "bbox": {x1,y1,x2,y2}). Same-class boxes that
    overlap more than iou_threshold are collapsed, keeping the
    higher-confidence one. Detections of different classes never
    suppress each other, since a genuine defect boundary can
    coincide with another defect type.
    """

    if not detections:
        return []

    by_class: dict[str, list[dict]] = {}
    for det in detections:
        by_class.setdefault(det["class"], []).append(det)

    kept: list[dict] = []
    for _cls, dets in by_class.items():
        dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        active = dets_sorted[:]
        while active:
            best = active.pop(0)
            kept.append(best)
            best_box = (
                best["bbox"]["x1"], best["bbox"]["y1"],
                best["bbox"]["x2"], best["bbox"]["y2"],
            )
            remaining = []
            for det in active:
                box = (
                    det["bbox"]["x1"], det["bbox"]["y1"],
                    det["bbox"]["x2"], det["bbox"]["y2"],
                )
                if _iou(best_box, box) < iou_threshold:
                    remaining.append(det)
            active = remaining

    return kept


def offset_detections(detections: list[dict], dx: float, dy: float) -> list[dict]:
    """Returns a new list with every bbox shifted by (dx, dy) —
    used to map a tile crop's local detections back into the
    original image's coordinate space. Does not mutate the input.
    """

    out = []
    for det in detections:
        bbox = det["bbox"]
        out.append({
            **det,
            "bbox": {
                "x1": round(bbox["x1"] + dx, 2),
                "y1": round(bbox["y1"] + dy, 2),
                "x2": round(bbox["x2"] + dx, 2),
                "y2": round(bbox["y2"] + dy, 2),
            },
        })
    return out
