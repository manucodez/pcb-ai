import pytest

from training.tiling import generate_tiles, merge_detections_nms, offset_detections, _iou


def _det(cls, conf, x1, y1, x2, y2):
    return {"class": cls, "confidence": conf, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


class TestGenerateTiles:
    def test_covers_full_image(self):
        tiles = generate_tiles(1000, 800, tile_size=400, overlap=0.2)
        max_x = max(t[2] for t in tiles)
        max_y = max(t[3] for t in tiles)
        min_x = min(t[0] for t in tiles)
        min_y = min(t[1] for t in tiles)
        assert max_x == 1000
        assert max_y == 800
        assert min_x == 0
        assert min_y == 0

    def test_small_image_returns_single_tile(self):
        tiles = generate_tiles(300, 200, tile_size=960, overlap=0.2)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 300, 200)

    def test_adjacent_tiles_overlap(self):
        tiles = generate_tiles(2000, 400, tile_size=960, overlap=0.2)
        xs = sorted({t[0] for t in tiles})
        assert len(xs) >= 2
        # consecutive tiles should share pixels (overlap > 0)
        first_end = [t[2] for t in tiles if t[0] == xs[0]][0]
        assert first_end > xs[1]

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            generate_tiles(0, 100, tile_size=50)
        with pytest.raises(ValueError):
            generate_tiles(100, 100, tile_size=0)
        with pytest.raises(ValueError):
            generate_tiles(100, 100, tile_size=50, overlap=1.0)


class TestIou:
    def test_identical_boxes(self):
        box = (0, 0, 10, 10)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 0, 15, 10))
        assert 0.3 < iou < 0.4  # intersection 5x10=50, union 150 -> 1/3


class TestMergeDetectionsNms:
    def test_empty_input(self):
        assert merge_detections_nms([]) == []

    def test_non_overlapping_same_class_both_kept(self):
        dets = [_det("short", 0.9, 0, 0, 10, 10), _det("short", 0.8, 100, 100, 110, 110)]
        merged = merge_detections_nms(dets, iou_threshold=0.5)
        assert len(merged) == 2

    def test_overlapping_same_class_keeps_higher_confidence(self):
        dets = [_det("short", 0.9, 0, 0, 10, 10), _det("short", 0.6, 1, 1, 11, 11)]
        merged = merge_detections_nms(dets, iou_threshold=0.3)
        assert len(merged) == 1
        assert merged[0]["confidence"] == 0.9

    def test_overlapping_different_classes_both_kept(self):
        dets = [_det("short", 0.9, 0, 0, 10, 10), _det("spur", 0.9, 0, 0, 10, 10)]
        merged = merge_detections_nms(dets, iou_threshold=0.3)
        assert len(merged) == 2

    def test_three_way_cluster_keeps_only_best(self):
        dets = [
            _det("short", 0.95, 0, 0, 10, 10),
            _det("short", 0.7, 1, 1, 11, 11),
            _det("short", 0.6, 2, 2, 12, 12),
        ]
        merged = merge_detections_nms(dets, iou_threshold=0.3)
        assert len(merged) == 1
        assert merged[0]["confidence"] == 0.95


class TestOffsetDetections:
    def test_shifts_bbox(self):
        dets = [_det("short", 0.9, 10, 10, 20, 20)]
        out = offset_detections(dets, dx=100, dy=50)
        assert out[0]["bbox"] == {"x1": 110, "y1": 60, "x2": 120, "y2": 70}

    def test_does_not_mutate_input(self):
        dets = [_det("short", 0.9, 10, 10, 20, 20)]
        offset_detections(dets, dx=100, dy=50)
        assert dets[0]["bbox"]["x1"] == 10

    def test_preserves_other_fields(self):
        dets = [_det("short", 0.9, 10, 10, 20, 20)]
        out = offset_detections(dets, dx=5, dy=5)
        assert out[0]["class"] == "short"
        assert out[0]["confidence"] == 0.9
