"""
=========================================================
Model Path Resolution

Purpose
-------
Pure, dependency-free (no torch/ultralytics) path resolution
for finding the trained detector weights. Split out from
training/inference.py so this logic can be unit-tested without
requiring the full ML stack to be installed, and so app.py can
import it for readiness checks without loading YOLO.

Bug fixed here
---------------
This used to be a single hardcoded constant
(runs/pcb_detector-3/weights/best.pt) — the literal folder name
Ultralytics happened to assign on one past training run. A
fresh `python training/train_yolo.py` run produces
runs/pcb_detector/weights/best.pt (no numeric suffix) and
silently would not match, breaking inference for anyone who
retrains. See training/train_yolo.py's docstring for how this
path gets created in the first place.

Resolution order, most to least specific:
  1. An explicit path passed to resolve_model_path(explicit=...).
  2. The PCB_MODEL_PATH environment variable.
  3. models/best.pt — the stable "promoted" location that
     training/train_yolo.py and scripts/promote_model.py write
     to, independent of whatever the run folder happened to be
     named.
  4. The most recently modified runs/*/weights/best.pt, so a
     brand-new training run works with zero manual steps even
     before anything has been promoted.

Author : Manjeet
=========================================================
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "runs"
MODELS_DIR = ROOT_DIR / "models"
PROMOTED_MODEL_PATH = MODELS_DIR / "best.pt"


def resolve_model_path(explicit: str | Path | None = None) -> Path:
    """Raises FileNotFoundError with an actionable message if no
    model can be found anywhere in the resolution order above."""

    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit model path does not exist: {p}")

    env_path = os.getenv("PCB_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"PCB_MODEL_PATH is set but does not exist: {p}")

    if PROMOTED_MODEL_PATH.exists():
        return PROMOTED_MODEL_PATH

    candidates = sorted(
        RUNS_DIR.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No trained model found. Checked, in order:\n"
        "  1. PCB_MODEL_PATH environment variable (not set or file missing)\n"
        f"  2. {PROMOTED_MODEL_PATH} (not found)\n"
        f"  3. {RUNS_DIR}/*/weights/best.pt (none found)\n"
        "Train a model with training/train_yolo.py (it auto-promotes the "
        "result), or point PCB_MODEL_PATH at an existing .pt file."
    )


def get_default_model_path() -> Path:
    """Never raises — for UI-facing readiness checks (app.py) that
    want to show a friendly message rather than crash on import.
    Falls back to the promoted-model path (even if it doesn't exist
    yet) so callers have somewhere actionable to point the user."""

    try:
        return resolve_model_path()
    except FileNotFoundError:
        return PROMOTED_MODEL_PATH
