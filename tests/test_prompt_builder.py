import numpy as np

from services.prompt_builder import PromptBuilder


def _crop(id_, cls, conf):
    return {
        "id": id_, "class": cls, "confidence": conf,
        "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "image": np.zeros((10, 10, 3), dtype=np.uint8),
    }


class TestPromptBuilder:
    def test_contents_alternates_text_and_images(self):
        crops = [_crop(1, "short", 0.9), _crop(2, "spur", 0.6)]
        contents = PromptBuilder().build_contents(crops)

        # intro text, then (label, image) per crop
        assert isinstance(contents[0], str)
        assert isinstance(contents[1], str) and "Defect ID 1" in contents[1]
        assert isinstance(contents[2], np.ndarray)
        assert isinstance(contents[3], str) and "Defect ID 2" in contents[3]
        assert isinstance(contents[4], np.ndarray)

    def test_every_image_immediately_preceded_by_its_own_label(self):
        crops = [_crop(1, "short", 0.9), _crop(2, "spur", 0.6), _crop(3, "open_circuit", 0.4)]
        contents = PromptBuilder().build_contents(crops)

        image_positions = [i for i, item in enumerate(contents) if isinstance(item, np.ndarray)]
        assert len(image_positions) == len(crops)

        for crop, pos in zip(crops, image_positions):
            preceding_label = contents[pos - 1]
            assert isinstance(preceding_label, str)
            assert f"Defect ID {crop['id']}" in preceding_label

    def test_label_ids_match_crop_order(self):
        # Note: the intro text also mentions "Defect ID" in its
        # instructions, so filter on the "--- Defect ID" marker that
        # only appears in the per-defect label blocks.
        crops = [_crop(5, "short", 0.9), _crop(7, "spur", 0.6)]
        contents = PromptBuilder().build_contents(crops)
        labels = [c for c in contents if isinstance(c, str) and "--- Defect ID" in c]
        assert "Defect ID 5" in labels[0]
        assert "Defect ID 7" in labels[1]

    def test_includes_json_schema_and_rules(self):
        contents = PromptBuilder().build_contents([_crop(1, "short", 0.9)])
        intro = contents[0]
        assert "JSON" in intro
        assert '"severity"' in intro
        assert "Critical" in intro and "Low" in intro

    def test_includes_domain_hint_for_known_class(self):
        contents = PromptBuilder().build_contents([_crop(1, "short", 0.9)])
        label = contents[1]
        assert "Domain hint" in label
        assert len(label.strip()) > 0

    def test_unknown_class_does_not_crash(self):
        contents = PromptBuilder().build_contents([_crop(1, "totally_new_defect_type", 0.9)])
        assert any(isinstance(c, str) and "Defect ID 1" in c for c in contents)

    def test_build_prompt_backward_compat_returns_text_only(self):
        text = PromptBuilder().build_prompt([_crop(1, "short", 0.9)])
        assert isinstance(text, str)
        assert "Defect ID 1" in text

    def test_empty_defects_list(self):
        contents = PromptBuilder().build_contents([])
        assert contents == [contents[0]]  # just the intro, no labels/images
