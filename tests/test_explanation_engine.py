import os
from unittest.mock import MagicMock

os.environ.setdefault("GEMINI_API_KEY", "test-key-for-unit-tests")

from services.explanation_engine import ExplanationEngine  # noqa: E402


def _crop(id_, cls, conf=0.9):
    return {"id": id_, "class": cls, "confidence": conf, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}, "image": "fake-image"}


def _engine_with_crops(crops):
    engine = ExplanationEngine()
    engine.cropper.crop_defects = MagicMock(return_value=crops)
    return engine


class TestExplanationEngineHappyPath:
    def test_all_defects_get_ai_explanation(self):
        crops = [_crop(1, "short"), _crop(2, "spur")]
        engine = _engine_with_crops(crops)
        engine.gemini.generate = MagicMock(return_value=[
            {"id": 1, "defect_type": "short", "explanation": "e1", "root_cause": "r1",
             "severity": "Critical", "recommended_fix": "f1", "prevention": "p1"},
            {"id": 2, "defect_type": "spur", "explanation": "e2", "root_cause": "r2",
             "severity": "Low", "recommended_fix": "f2", "prevention": "p2"},
        ])

        results = engine.generate("fake/path.jpg", detections=[{}, {}])

        assert len(results) == 2
        assert all(r["explanation"]["ai_generated"] for r in results)
        assert results[0]["explanation"]["severity"] == "Critical"

    def test_no_detections_returns_empty_list(self):
        engine = _engine_with_crops([])
        assert engine.generate("fake/path.jpg", detections=[]) == []


class TestExplanationEnginePartialFallback:
    def test_missing_id_in_response_falls_back_only_for_that_defect(self):
        crops = [_crop(1, "short"), _crop(2, "spur")]
        engine = _engine_with_crops(crops)
        # Gemini only returned an explanation for defect 1.
        engine.gemini.generate = MagicMock(return_value=[
            {"id": 1, "defect_type": "short", "explanation": "e1", "root_cause": "r1",
             "severity": "Critical", "recommended_fix": "f1", "prevention": "p1"},
        ])

        results = engine.generate("fake/path.jpg", detections=[{}, {}])

        assert len(results) == 2  # both defects still present
        by_id = {r["id"]: r for r in results}
        assert by_id[1]["explanation"]["ai_generated"] is True
        assert by_id[2]["explanation"]["ai_generated"] is False  # fell back

    def test_unknown_id_in_response_is_ignored_not_crashed(self):
        crops = [_crop(1, "short")]
        engine = _engine_with_crops(crops)
        engine.gemini.generate = MagicMock(return_value=[
            {"id": 99, "defect_type": "short", "explanation": "e", "root_cause": "r",
             "severity": "Critical", "recommended_fix": "f", "prevention": "p"},
        ])

        results = engine.generate("fake/path.jpg", detections=[{}])
        assert len(results) == 1
        assert results[0]["explanation"]["ai_generated"] is False

    def test_missing_field_backfilled_with_na(self):
        crops = [_crop(1, "short")]
        engine = _engine_with_crops(crops)
        engine.gemini.generate = MagicMock(return_value=[
            {"id": 1, "defect_type": "short", "severity": "High"},  # missing several fields
        ])
        results = engine.generate("fake/path.jpg", detections=[{}])
        assert results[0]["explanation"]["explanation"] == "N/A"
        assert results[0]["explanation"]["severity"] == "High"


class TestExplanationEngineTotalFailure:
    def test_gemini_exception_falls_back_for_every_defect_without_raising(self):
        crops = [_crop(1, "short"), _crop(2, "spur")]
        engine = _engine_with_crops(crops)
        engine.gemini.generate = MagicMock(side_effect=RuntimeError("Gemini Service is currently unavailable."))

        results = engine.generate("fake/path.jpg", detections=[{}, {}])

        assert len(results) == 2
        assert all(r["explanation"]["ai_generated"] is False for r in results)
        # Detections themselves are still intact even though AI failed.
        assert {r["id"] for r in results} == {1, 2}

    def test_non_list_response_falls_back_for_all(self):
        crops = [_crop(1, "short")]
        engine = _engine_with_crops(crops)
        engine.gemini.generate = MagicMock(return_value={"unexpected": "shape"})

        results = engine.generate("fake/path.jpg", detections=[{}])
        assert results[0]["explanation"]["ai_generated"] is False
