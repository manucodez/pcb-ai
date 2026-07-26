"""
=========================================================
PCB Dataset Merger

Purpose
-------
Merge DeepPCB + PCB Defects into one standardized dataset.

Features
--------
- Verifies each source dataset's own class order against the
  order this script's DEEPPCB_MAP / PCB_MAP assume, BEFORE
  remapping anything.
- Skips (and reports) images with no label file instead of
  silently copying them in as unlabeled "background" images.
- Remaps DeepPCB class ids to the standard 6-class scheme.
- Reports per-class instance counts for the merged set.
- Generates the final data.yaml.

Bugs fixed here (read before changing DEEPPCB_MAP / PCB_MAP)
--------------------------------------------------------------
1. Silent class-order corruption: DEEPPCB_MAP was hand-written
   against a *comment* describing DeepPCB's assumed class order.
   If a re-exported copy of the dataset ever has a different
   order (a very easy thing for a Roboflow re-export to change),
   every single label gets silently remapped to the WRONG class
   with no error — the training run would complete "successfully"
   on quietly corrupted data. verify_class_order() below checks
   the dataset's own data.yaml against the assumed order and
   refuses to proceed if they don't match.

2. Orphan unlabeled images: the previous version copied every
   source image into the merged set unconditionally, then tried
   to remap its label file — but if the label file didn't exist,
   remap_label_file() just returned early, leaving an image in
   the merged dataset with NO label file at all. YOLO treats an
   image with no label file as a background (no-object) sample,
   which is very likely wrong for a defect dataset where a
   missing label usually means the label was lost, not that the
   board is genuinely clean. Now these are skipped and counted;
   pass --include-unlabeled if you've confirmed they really are
   intentional background images.

Usage
-----
    python scripts/merge_datasets.py
    python scripts/merge_datasets.py --include-unlabeled
=========================================================
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

DEEPPCB = ROOT / "dataset" / "DeepPCB.v1i.yolov11"
PCBDEFECT = ROOT / "dataset" / "PCB Defects.v1i.yolov11"

OUTPUT = ROOT / "dataset" / "PCB_Combined"

# ==========================================================
# Standard Classes
# ==========================================================

STANDARD_CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# ==========================================================
# DeepPCB ID Mapping
#
# The class-id mappings below are only correct if each source
# dataset's OWN data.yaml lists its classes in the order given
# here. verify_class_order() enforces that before any remapping
# happens, instead of trusting it silently.
#
# DeepPCB assumed order
# 0 copper        -> spurious_copper (extraneous copper)
# 1 mousebite      -> mouse_bite
# 2 open           -> open_circuit
# 3 pin-hole       -> missing_hole  (closest standard analog)
# 4 short          -> short
# 5 spur           -> spur
# ==========================================================

DEEPPCB_ASSUMED_ORDER = ["copper", "mousebite", "open", "pin-hole", "short", "spur"]

DEEPPCB_MAP = {0: 5, 1: 1, 2: 2, 3: 0, 4: 3, 5: 4}

# PCB Defects is assumed to already follow the standard order.
PCBDEFECT_ASSUMED_ORDER = list(STANDARD_CLASSES)

PCB_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

SPLITS = ["train", "valid", "test"]


# ==========================================================


def read_names(data_yaml: Path) -> list[str]:
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda k: int(k))]
    return [str(n).strip().lower() for n in names]


def verify_class_order(dataset_path: Path, assumed_order: list[str], label: str) -> None:
    """Refuses to proceed if the dataset's actual class order
    doesn't match what the id-remapping table assumes — see
    module docstring point 1. This is a hard stop, not a
    warning, because a mismatch here corrupts every label
    silently."""

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{label}: no data.yaml found at {data_yaml}")

    actual = read_names(data_yaml)
    expected = [c.lower() for c in assumed_order]

    if actual != expected:
        raise ValueError(
            f"\n{label}: class order in {data_yaml} does not match the "
            f"order DEEPPCB_MAP/PCB_MAP assume — remapping would silently "
            f"corrupt labels, so this is refusing to proceed.\n"
            f"  Expected : {expected}\n"
            f"  Actual   : {actual}\n"
            f"Update the *_MAP dict in this script to match the actual "
            f"order above, then re-run."
        )

    print(f"✅ {label}: class order verified against data.yaml.")


# ==========================================================


def make_dirs():
    for split in SPLITS:
        (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)


def remap_label_lines(src: Path, mapping: dict[int, int]) -> list[str]:
    new_lines = []
    with open(src, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls not in mapping:
                raise ValueError(f"{src}: class id {cls} has no entry in the mapping.")
            parts[0] = str(mapping[cls])
            new_lines.append(" ".join(parts))
    return new_lines


def merge_dataset(dataset_path: Path, prefix: str, mapping: dict, include_unlabeled: bool, class_counts: Counter):
    print(f"\nMerging {prefix} ({dataset_path.name})")

    for split in SPLITS:
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"
        out_img = OUTPUT / split / "images"
        out_lbl = OUTPUT / split / "labels"

        if not image_dir.exists():
            print(f"  {split}: no images/ dir — skipping split.")
            continue

        images = list(image_dir.glob("*"))
        copied, skipped_unlabeled, skipped_empty = 0, 0, 0

        for img in images:
            lbl = label_dir / (img.stem + ".txt")

            if not lbl.exists() and not include_unlabeled:
                skipped_unlabeled += 1
                continue

            new_name = f"{prefix}_{img.name}"
            new_lbl_name = f"{prefix}_{img.stem}.txt"

            if lbl.exists():
                lines = remap_label_lines(lbl, mapping)
                if not lines:
                    skipped_empty += 1
                    # Still copy — an intentionally empty label file
                    # means "verified no defects", unlike a missing
                    # file, which means "we don't actually know".
                (out_lbl / new_lbl_name).write_text("\n".join(lines))
                for line in lines:
                    class_counts[STANDARD_CLASSES[int(line.split()[0])]] += 1
            else:
                (out_lbl / new_lbl_name).write_text("")

            shutil.copy2(img, out_img / new_name)
            copied += 1

        print(
            f"  {split}: {copied} copied, "
            f"{skipped_unlabeled} skipped (no label file), "
            f"{skipped_empty} with an empty label (kept, 0 objects)"
        )


# ==========================================================


def create_yaml():
    data = {
        "path": str(OUTPUT),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": STANDARD_CLASSES,
    }
    with open(OUTPUT / "data.yaml", "w") as f:
        yaml.dump(data, f, sort_keys=False)


# ==========================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-unlabeled", action="store_true",
        help=(
            "Copy images with no label file as background (0-object) "
            "samples instead of skipping them. Only use this if you have "
            "confirmed those images are genuinely defect-free, not just "
            "missing their label."
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PCB DATASET MERGER")
    print("=" * 60)

    verify_class_order(DEEPPCB, DEEPPCB_ASSUMED_ORDER, "DeepPCB")
    verify_class_order(PCBDEFECT, PCBDEFECT_ASSUMED_ORDER, "PCB Defects")

    make_dirs()

    class_counts: Counter = Counter()

    merge_dataset(DEEPPCB, "deep", DEEPPCB_MAP, args.include_unlabeled, class_counts)
    merge_dataset(PCBDEFECT, "pcb", PCB_MAP, args.include_unlabeled, class_counts)

    create_yaml()

    print("\n" + "=" * 60)
    print("PER-CLASS INSTANCE COUNTS (merged set)")
    print("=" * 60)
    total = sum(class_counts.values())
    for cls in STANDARD_CLASSES:
        count = class_counts.get(cls, 0)
        pct = (count / total * 100) if total else 0
        print(f"  {cls:<18} {count:>6}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<18} {total:>6}")

    if total and max(class_counts.values(), default=0) > 3 * (total / len(STANDARD_CLASSES)):
        print(
            "\n⚠ Class distribution is noticeably imbalanced. Consider "
            "class weighting or oversampling the rarer classes during "
            "training — see README for pointers."
        )

    print("\nDone.")
    print("Combined dataset created at:", OUTPUT)
    print("Run scripts/verify_dataset.py next to validate the merged output.")


if __name__ == "__main__":
    main()
