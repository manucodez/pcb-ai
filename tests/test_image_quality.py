import numpy as np

from services.image_quality import assess_quality


def _solid_image(w, h, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _sharp_checkerboard(size=800):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    step = 20
    for y in range(0, size, step):
        for x in range(0, size, step):
            if (x // step + y // step) % 2 == 0:
                img[y:y + step, x:x + step] = 255
    return img


class TestAssessQuality:
    def test_none_image_flagged(self):
        result = assess_quality(None)
        assert result["ok"] is False
        assert result["warnings"]

    def test_low_resolution_flagged(self):
        result = assess_quality(_solid_image(200, 150))
        assert any("resolution" in w.lower() for w in result["warnings"])

    def test_high_resolution_sharp_image_not_flagged_for_size_or_blur(self):
        result = assess_quality(_sharp_checkerboard(1000))
        assert not any("resolution" in w.lower() for w in result["warnings"])
        assert not any("blur" in w.lower() for w in result["warnings"])

    def test_flat_image_flagged_as_blurry(self):
        # A perfectly flat image has zero Laplacian variance -> blurry.
        result = assess_quality(_solid_image(1000, 1000, value=128))
        assert any("blur" in w.lower() for w in result["warnings"])

    def test_dark_image_flagged(self):
        result = assess_quality(_solid_image(1000, 1000, value=5))
        assert any("dark" in w.lower() for w in result["warnings"])

    def test_bright_image_flagged(self):
        result = assess_quality(_solid_image(1000, 1000, value=250))
        assert any("overexposed" in w.lower() for w in result["warnings"])

    def test_metrics_present(self):
        result = assess_quality(_sharp_checkerboard(700))
        assert result["width"] == 700
        assert result["height"] == 700
        assert "blur_variance" in result
        assert "mean_brightness" in result
