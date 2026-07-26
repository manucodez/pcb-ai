from pathlib import Path

import pytest

import training.model_paths as model_paths
from training.model_paths import resolve_model_path, get_default_model_path


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Redirect the module's path constants into a scratch dir so
    tests never touch (or depend on) this machine's real runs/ or
    models/ directories."""
    runs_dir = tmp_path / "runs"
    models_dir = tmp_path / "models"
    monkeypatch.setattr(model_paths, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(model_paths, "MODELS_DIR", models_dir)
    monkeypatch.setattr(model_paths, "PROMOTED_MODEL_PATH", models_dir / "best.pt")
    monkeypatch.delenv("PCB_MODEL_PATH", raising=False)
    return tmp_path


class TestResolveModelPath:
    def test_explicit_path_wins_when_it_exists(self, tmp_path):
        weights = tmp_path / "custom.pt"
        weights.write_bytes(b"fake")
        assert resolve_model_path(explicit=weights) == weights

    def test_explicit_path_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_model_path(explicit=tmp_path / "nope.pt")

    def test_env_var_used_when_no_explicit_path(self, tmp_path, monkeypatch):
        weights = tmp_path / "env_model.pt"
        weights.write_bytes(b"fake")
        monkeypatch.setenv("PCB_MODEL_PATH", str(weights))
        assert resolve_model_path() == weights

    def test_env_var_missing_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PCB_MODEL_PATH", str(tmp_path / "ghost.pt"))
        with pytest.raises(FileNotFoundError):
            resolve_model_path()

    def test_promoted_model_used_when_no_explicit_or_env(self):
        model_paths.MODELS_DIR.mkdir(parents=True)
        model_paths.PROMOTED_MODEL_PATH.write_bytes(b"fake")
        assert resolve_model_path() == model_paths.PROMOTED_MODEL_PATH

    def test_falls_back_to_newest_run_when_nothing_promoted(self):
        # Bug this guards against: a hardcoded run-folder name (e.g.
        # "pcb_detector-3") would silently miss a fresh run named
        # just "pcb_detector". Auto-discovery must find it regardless
        # of the folder's exact name, and prefer the newest one.
        older = model_paths.RUNS_DIR / "pcb_detector" / "weights"
        newer = model_paths.RUNS_DIR / "pcb_detector2" / "weights"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)

        (older / "best.pt").write_bytes(b"old")
        import time
        time.sleep(0.01)
        (newer / "best.pt").write_bytes(b"new")

        result = resolve_model_path()
        assert result == newer / "best.pt"

    def test_raises_with_actionable_message_when_nothing_found(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_model_path()
        msg = str(exc_info.value)
        assert "train_yolo.py" in msg
        assert "PCB_MODEL_PATH" in msg

    def test_promoted_model_preferred_over_runs_dir(self):
        run_weights = model_paths.RUNS_DIR / "pcb_detector" / "weights"
        run_weights.mkdir(parents=True)
        (run_weights / "best.pt").write_bytes(b"run-version")

        model_paths.MODELS_DIR.mkdir(parents=True)
        model_paths.PROMOTED_MODEL_PATH.write_bytes(b"promoted-version")

        assert resolve_model_path() == model_paths.PROMOTED_MODEL_PATH


class TestGetDefaultModelPath:
    def test_never_raises_when_nothing_found(self):
        result = get_default_model_path()
        assert isinstance(result, Path)
        assert not result.exists()

    def test_returns_actual_model_when_present(self):
        model_paths.MODELS_DIR.mkdir(parents=True)
        model_paths.PROMOTED_MODEL_PATH.write_bytes(b"fake")
        assert get_default_model_path() == model_paths.PROMOTED_MODEL_PATH
