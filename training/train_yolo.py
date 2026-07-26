"""
=========================================================
PCB Defect Detection

File : train_yolo.py

Purpose
-------
Train YOLO11 for PCB Defect Detection.

Input
-----
PCB_Combined Dataset

Output
------
Best YOLO Model
Training Metrics
Loss Curves
Evaluation Results
models/best.pt  — auto-promoted "production" copy + MANIFEST.json

Bug fixed here (read before touching validate())
---------------------------------------------------
validate() used to reconstruct the run directory as a hardcoded
guess: OUTPUT / RUN_NAME / "weights" / "best.pt". That is only
correct the FIRST time you ever train with a given run name.
Ultralytics auto-increments the folder (pcb_detector2,
pcb_detector3, ...) whenever the name already exists, so on any
retrain this guess silently points at a stale, older run's
weights instead of the one that just finished — while looking
like it worked, because the old file is still right there. This
is almost certainly how training/inference.py ended up with
"pcb_detector-3" hardcoded as its own guess: someone trained
three times and had to work out the real path by hand each time.

Fixed by reading the actual save directory back from the
trainer object instead of reconstructing it, and by having this
script auto-promote the result to a stable models/best.pt so
inference and validate() never need to guess a run folder name
again (see training/inference.py's resolve_model_path()).

Usage
-----
    python training/train_yolo.py
    python training/train_yolo.py --epochs 150 --imgsz 800 --batch 8
    python training/train_yolo.py --resume

Author : Jasmeen
=========================================================
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET = ROOT / "dataset" / "PCB_Combined" / "data.yaml"
DEFAULT_MODEL = "yolo11s.pt"
OUTPUT = ROOT / "runs"
MODELS_DIR = ROOT / "models"
DEFAULT_RUN_NAME = "pcb_detector"

DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ==========================================================


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                         help="Base checkpoint to fine-tune from, or a .pt to resume/continue training.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--resume", action="store_true", help="Resume the last interrupted run.")
    parser.add_argument("--no-promote", action="store_true",
                         help="Skip copying the result to models/best.pt after training.")
    return parser.parse_args()


def print_configuration(args):
    print("=" * 60)
    print("PCB DEFECT DETECTOR TRAINING")
    print("=" * 60)
    print(f"Dataset   : {args.dataset}")
    print(f"Model     : {args.model}")
    print(f"Epochs    : {args.epochs}")
    print(f"ImageSize : {args.imgsz}")
    print(f"BatchSize : {args.batch}")
    print(f"Workers   : {args.workers}")
    print(f"Device    : {args.device}")
    print(f"Run name  : {args.run_name}")
    print("=" * 60)


# ==========================================================


def train(args) -> Path:
    """Returns the actual save_dir Ultralytics used for this run
    (not a reconstructed guess — see module docstring)."""

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(args.dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(OUTPUT),
        name=args.run_name,
        patience=args.patience,
        resume=args.resume,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,
        cache=True,
        amp=True,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,

        # ---- Augmentation, tuned for small-object PCB defects ----
        # Defects (mouse bites, spurs, pin-holes) are tiny relative
        # to the board, so scale variance during training matters
        # more than usual — keep it on. Shear/perspective are left
        # at 0: PCB photos are flat, rigid boards, so these would
        # just teach the model to expect distortions that don't
        # occur in real inspection photos.
        scale=0.5,
        degrees=5.0,       # small tolerance for camera tilt
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,        # copper-trace patterns are meaningful
        fliplr=0.5,        # both ways up/mirrored; silkscreen text
                            # orientation doesn't matter for this task
        mixup=0.0,          # blending two boards tends to hurt tiny
                            # defect boundaries more than it helps
        copy_paste=0.0,     # ultralytics' copy_paste targets
                            # instance-segmentation masks; not a good
                            # fit for bbox-only detection labels here
    )

    # multi_scale (random +/-50% image size per batch) is a genuine
    # accuracy lever for datasets with wide object-size variance, but
    # its availability depends on the installed ultralytics version —
    # attempt it, and train without it rather than crash a long run
    # over one unsupported kwarg.
    try:
        model.train(multi_scale=True, **train_kwargs)
    except TypeError:
        print("Installed ultralytics version does not support multi_scale — training without it.")
        model.train(**train_kwargs)

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        # Fallback for older/newer API shapes — still better than a
        # hardcoded guess, since we at least search for what was
        # actually just written on disk.
        candidates = sorted(
            (OUTPUT).glob(f"{args.run_name}*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        save_dir = candidates[0].parent.parent if candidates else OUTPUT / args.run_name

    return Path(save_dir)


# ==========================================================


def validate(run_dir: Path):
    print("\nRunning Validation...\n")

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"⚠ Expected weights at {weights} but they don't exist — skipping validation.")
        return None

    model = YOLO(weights)
    metrics = model.val()

    print("\nValidation Complete\n")
    print(metrics)
    return metrics


# ==========================================================


def promote(run_dir: Path, args, metrics) -> None:
    """Copies the freshly trained best.pt to the stable
    models/best.pt location that training/inference.py's
    resolve_model_path() checks first, and writes a manifest so
    it's clear later which run/dataset/metrics produced it —
    without this, picking the right runs/*/weights/best.pt by
    hand is exactly the error-prone step that caused the
    hardcoded-path bug this script fixes elsewhere."""

    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"⚠ No weights found at {weights} — nothing to promote.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "best.pt"
    shutil.copy2(weights, dest)

    metric_summary = {}
    if metrics is not None:
        try:
            box = metrics.box
            metric_summary = {
                "map50": round(float(box.map50), 4),
                "map50_95": round(float(box.map), 4),
                "precision": round(float(box.mp), 4),
                "recall": round(float(box.mr), 4),
            }
        except Exception:
            metric_summary = {"note": "metrics available but not in the expected shape; see runs dir directly."}

    manifest = {
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "dataset": str(args.dataset),
        "base_model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "metrics": metric_summary,
    }
    (MODELS_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n✅ Promoted {weights} -> {dest}")
    print(f"   Manifest written to {MODELS_DIR / 'MANIFEST.json'}")


# ==========================================================


def main():
    args = parse_args()
    print_configuration(args)

    run_dir = train(args)
    metrics = validate(run_dir)

    if not args.no_promote:
        promote(run_dir, args, metrics)

    print("\nTraining Finished Successfully.")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
