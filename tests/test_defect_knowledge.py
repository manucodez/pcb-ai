from services.defect_knowledge import lookup, fallback_explanation, DEFECT_KB


class TestLookup:
    def test_known_class_exact(self):
        kb = lookup("mouse_bite")
        assert kb["severity"] == "Medium"

    def test_case_and_hyphen_tolerant(self):
        assert lookup("Mouse-Bite") == lookup("mouse_bite")
        assert lookup("SHORT") == lookup("short")

    def test_unknown_class_returns_generic_entry_not_raise(self):
        kb = lookup("some_future_defect_class")
        assert "severity" in kb
        assert kb["severity"]

    def test_all_six_standard_classes_present(self):
        expected = {"missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"}
        assert expected.issubset(DEFECT_KB.keys())


class TestFallbackExplanation:
    def test_shape_matches_ai_explanation_schema(self):
        exp = fallback_explanation("short")
        for field in ("defect_type", "explanation", "root_cause", "severity", "recommended_fix", "prevention"):
            assert field in exp

    def test_marked_as_not_ai_generated(self):
        exp = fallback_explanation("spur")
        assert exp["ai_generated"] is False

    def test_never_raises_for_garbage_input(self):
        for bad in [None, "", "   ", "!!!unknown!!!"]:
            exp = fallback_explanation(bad)
            assert exp["severity"]
