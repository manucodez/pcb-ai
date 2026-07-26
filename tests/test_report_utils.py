from services.report_utils import (
    normalize_severity,
    normalize_result,
    overall_verdict,
    sorted_by_severity,
    dedupe_labels,
    build_markdown_report,
)


def _det(id_, cls, conf, severity, ai_generated=True):
    return {
        "id": id_,
        "class": cls,
        "confidence": conf,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "explanation": {
            "severity": severity,
            "explanation": "e", "root_cause": "r",
            "recommended_fix": "f", "prevention": "p",
            "ai_generated": ai_generated,
        },
    }


class TestNormalizeSeverity:
    def test_exact_matches(self):
        assert normalize_severity("Critical") == "Critical"
        assert normalize_severity("high") == "High"
        assert normalize_severity("MEDIUM") == "Medium"
        assert normalize_severity("low") == "Low"

    def test_synonyms(self):
        assert normalize_severity("severe") == "Critical"
        assert normalize_severity("major") == "High"
        assert normalize_severity("moderate") == "Medium"
        assert normalize_severity("cosmetic") == "Low"

    def test_unknown_and_empty(self):
        assert normalize_severity(None) == "Unknown"
        assert normalize_severity("") == "Unknown"
        assert normalize_severity("banana") == "Unknown"


class TestNormalizeResult:
    def test_normalizes_all_detections_in_place(self):
        result = {"detections": [_det(1, "spur", 0.9, "severe"), _det(2, "short", 0.8, "blocker")]}
        out = normalize_result(result)
        assert out["detections"][0]["explanation"]["severity"] == "Critical"
        assert out["detections"][1]["explanation"]["severity"] == "Critical"

    def test_handles_missing_explanation(self):
        result = {"detections": [{"id": 1, "class": "spur", "confidence": 0.5, "bbox": {}}]}
        out = normalize_result(result)
        assert out["detections"][0]["explanation"]["severity"] == "Unknown"


class TestOverallVerdict:
    def test_no_detections_is_pass(self):
        label, color, msg = overall_verdict([])
        assert label == "PASS"

    def test_critical_dominates(self):
        dets = [_det(1, "short", 0.9, "Low"), _det(2, "open_circuit", 0.9, "Critical")]
        label, _, _ = overall_verdict(dets)
        assert label == "FAIL"

    def test_high_without_critical(self):
        dets = [_det(1, "short", 0.9, "High"), _det(2, "spur", 0.9, "Low")]
        label, _, _ = overall_verdict(dets)
        assert "HOLD" in label

    def test_only_low_passes_with_note(self):
        dets = [_det(1, "spur", 0.9, "Low")]
        label, _, _ = overall_verdict(dets)
        assert label == "PASS — MINOR ISSUES"


class TestSortedBySeverity:
    def test_critical_first(self):
        dets = [_det(1, "spur", 0.9, "Low"), _det(2, "short", 0.9, "Critical"), _det(3, "x", 0.9, "Medium")]
        ordered = sorted_by_severity(dets)
        assert [d["id"] for d in ordered] == [2, 3, 1]

    def test_unknown_sorts_last(self):
        dets = [_det(1, "x", 0.9, "Unknown"), _det(2, "y", 0.9, "Low")]
        ordered = sorted_by_severity(dets)
        assert ordered[-1]["id"] == 1


class TestDedupeLabels:
    def test_unique_names_untouched(self):
        assert dedupe_labels(["a.jpg", "b.jpg"]) == ["a.jpg", "b.jpg"]

    def test_repeated_names_get_suffixed(self):
        # Every occurrence of a name that appears more than once is
        # numbered (including the first) so all copies are equally
        # distinguishable in the UI's tab labels.
        assert dedupe_labels(["a.jpg", "a.jpg", "a.jpg"]) == ["a.jpg (1)", "a.jpg (2)", "a.jpg (3)"]

    def test_mixed(self):
        assert dedupe_labels(["a.jpg", "b.jpg", "a.jpg"]) == ["a.jpg (1)", "b.jpg", "a.jpg (2)"]


class TestBuildMarkdownReport:
    def test_includes_key_sections(self):
        result = {
            "image": "board1.jpg",
            "total_defects": 1,
            "pipeline_time_sec": 1.23,
            "scan_settings": {"imgsz": 1280, "augment": False, "tiled": False, "confidence": 0.5, "iou": 0.4},
            "detections": [_det(1, "short", 0.91, "Critical")],
        }
        md = build_markdown_report(result)
        assert "board1.jpg" in md
        assert "FAIL" in md
        assert "Defect #1" in md
        assert "Short" in md

    def test_no_detections(self):
        result = {
            "image": "board2.jpg", "total_defects": 0, "pipeline_time_sec": 0.5,
            "scan_settings": {}, "detections": [],
        }
        md = build_markdown_report(result)
        assert "No defects detected" in md

    def test_fallback_note_shown_for_non_ai_generated(self):
        result = {
            "image": "board3.jpg", "total_defects": 1, "pipeline_time_sec": 0.5,
            "scan_settings": {}, "detections": [_det(1, "spur", 0.6, "Low", ai_generated=False)],
        }
        md = build_markdown_report(result)
        assert "standard guidance" in md.lower()
