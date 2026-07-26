import cv2
import numpy as np
import pytest

from services.crop_defects import DefectCropper, enhance_clarity


def _make_test_image(path, w=400, h=300):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return img


class TestEnhanceClarity:
    def test_upscales_small_crop(self):
        # short side 60 * scale 5.33 (< max_upscale 6.0) -> ~320, so this
        # exercises the target-size path rather than the upscale cap.
        small = np.random.randint(0, 255, (60, 90, 3), dtype=np.uint8)
        out = enhance_clarity(small, target_min_side=320)
        assert min(out.shape[:2]) >= 300  # allow for rounding

    def test_leaves_large_crop_size_alone(self):
        large = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        out = enhance_clarity(large, target_min_side=320)
        assert out.shape[:2] == large.shape[:2]

    def test_respects_max_upscale(self):
        tiny = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
        out = enhance_clarity(tiny, target_min_side=320, max_upscale=6.0)
        assert min(out.shape[:2]) <= 5 * 6.0 + 1

    def test_handles_empty_image_without_raising(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert enhance_clarity(empty).size == 0

    def test_handles_none_without_raising(self):
        assert enhance_clarity(None) is None


class TestDefectCropper:
    def test_crop_defects_basic(self, tmp_path):
        img_path = tmp_path / "board.jpg"
        _make_test_image(img_path, w=400, h=300)

        detections = [
            {"class": "short", "confidence": 0.9, "bbox": {"x1": 50, "y1": 50, "x2": 100, "y2": 100}},
            {"class": "spur", "confidence": 0.7, "bbox": {"x1": 200, "y1": 150, "x2": 220, "y2": 170}},
        ]

        crops = DefectCropper(padding=10).crop_defects(img_path, detections)

        assert len(crops) == 2
        assert crops[0]["id"] == 1
        assert crops[1]["id"] == 2
        assert crops[0]["class"] == "short"
        assert crops[0]["image"].size > 0

    def test_ids_are_sequential_and_start_at_one(self, tmp_path):
        img_path = tmp_path / "board.jpg"
        _make_test_image(img_path)
        detections = [{"class": "x", "confidence": 0.5, "bbox": {"x1": i * 10, "y1": 0, "x2": i * 10 + 5, "y2": 5}} for i in range(5)]
        crops = DefectCropper().crop_defects(img_path, detections)
        assert [c["id"] for c in crops] == [1, 2, 3, 4, 5]

    def test_bbox_near_image_edge_is_clipped_not_crashed(self, tmp_path):
        img_path = tmp_path / "board.jpg"
        _make_test_image(img_path, w=100, h=100)
        detections = [{"class": "x", "confidence": 0.5, "bbox": {"x1": 95, "y1": 95, "x2": 130, "y2": 130}}]
        crops = DefectCropper(padding=20).crop_defects(img_path, detections)
        assert len(crops) == 1  # doesn't raise despite bbox exceeding image bounds

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DefectCropper().crop_defects(tmp_path / "does_not_exist.jpg", [])

    def test_no_detections_returns_empty_list(self, tmp_path):
        img_path = tmp_path / "board.jpg"
        _make_test_image(img_path)
        assert DefectCropper().crop_defects(img_path, []) == []
