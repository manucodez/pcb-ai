"""
=========================================================
PCB Model Promoter

Purpose
-------
List every past training run under runs/, show its final
validation metrics (read straight from Ultralytics' own
results.csv — no re-running validation needed), and copy a
chosen run's best.pt to the stable models/best.pt location that
training/inference.py's resolve_model_path() checks first.

Why this exists
-----------------
train_yolo.py auto-promotes the run it just finished. This tool
is for the other common case: you already have several old runs
sitting in runs/ from earlier experiments and want to promote
whichever actually had the best validation mAP, without
retraining anything or hand-editing paths.

Usage
-----
    python scripts/promote_model.py                 # list runs, auto-picks best mAP50-95
    python scripts/promote_model.py --list           # list only, promote nothing
    python scripts/promote_model.py --run pcb_detector3
=========================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"


def find_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        (p.parent.parent for p in RUNS_DIR.glob("*/weights/best.pt")),
        key=lambda p: (p / "weights" / "best.pt").stat().st_mtime,
        reverse=True,
    )


def read_final_metrics(run_dir: Path) -> dict:
    """Ultralytics writes one row per epoch to results.csv, with
    the best-epoch weights saved as best.pt — but results.csv's
    LAST row is the final epoch, which for early-stopped or
    patience-triggered runs is usually also the best (or very
    close to it). Good enough for a quick comparison across
    runs; re-run scripts/verify_dataset.py + model.val() for an
    authoritative number on any run you're about to ship."""

    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    last = rows[-1]
    out = {}
    for key in last:
        stripped = key.strip()
        if "map50-95" in stripped.lower() or "map50_95" in stripped.lower():
            out["map50_95"] = last[key].strip()
        elif "map50" in stripped.lower():
            out["map50"] = last[key].strip()
        elif "precision" in stripped.lower():
            out["precision"] = last[key].strip()
        elif "recall" in stripped.lower():
            out["recall"] = last[key].strip()
    return out


def promote(run_dir: Path):
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        print(f"⚠ No weights at {weights}")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "best.pt"
    shutil.copy2(weights, dest)

    manifest = {
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "metrics": read_final_metrics(run_dir),
        "promoted_via": "scripts/promote_model.py (manual)",
    }
    (MODELS_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"✅ Promoted {weights} -> {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=str, default=None, help="Run folder name under runs/, e.g. pcb_detector3")
    parser.add_argument("--list", action="store_true", help="List runs and exit without promoting anything.")
    args = parser.parse_args()

    runs = find_runs()
    if not runs:
        print(f"No runs with weights/best.pt found under {RUNS_DIR}")
        return

    print("\nAvailable runs (most recent first):\n")
    for run in runs:
        metrics = read_final_metrics(run)
        metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items()) or "no results.csv found"
        print(f"  {run.name:<20} {metrics_str}")

    if args.list:
        return

    if args.run:
        chosen = next((r for r in runs if r.name == args.run), None)
        if chosen is None:
            print(f"\n⚠ Run '{args.run}' not found under {RUNS_DIR}")
            return
    else:
        def map_value(run: Path) -> float:
            try:
                return float(read_final_metrics(run).get("map50_95", -1))
            except ValueError:
                return -1
        chosen = max(runs, key=map_value)
        print(f"\nNo --run given — auto-selecting best mAP50-95: {chosen.name}")

    print()
    promote(chosen)


if __name__ == "__main__":
    main()
