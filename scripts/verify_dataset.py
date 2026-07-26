"""
=========================================================
PCB Dataset Verifier

Purpose
-------
Sanity-check a merged YOLO dataset before spending GPU hours
training on it. Catches the kind of silent problems that don't
crash training but quietly cap the accuracy you can ever reach:

1. Corrupt / unreadable images.
2. Missing labels (image with no .txt).
3. Orphan labels (.txt with no matching image).
4. Malformed label LINES — wrong token count, a class id outside
   the dataset's declared range, or a coordinate outside [0, 1].
   None of these crash Ultralytics; they just quietly train on
   wrong or garbage boxes for that line.
5. Class imbalance across the merged set.
6. Exact-duplicate images appearing in more than one split
   (train/valid/test). This is the one most people don't think
   to check for: if a merge or a dataset's own export process
   ever puts the same image in both train and valid, your
   validation mAP is partly measuring memorization, not
   generalization — the number looks great and the model
   underperforms in the field.

Usage
-----
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --dataset dataset/PCB_Combined
=========================================================
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "dataset" / "PCB_Combined"

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
COORD_EPS = 1e-4


def load_class_names(dataset_path: Path) -> list[str]:
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"⚠ No data.yaml found at {data_yaml} — label class-id bounds will not be checked.")
        return []
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys(), key=lambda k: int(k))]
    return list(names)


def validate_label_file(lbl_path: Path, num_classes: int) -> list[str]:
    """Returns a list of human-readable problem descriptions for
    this file (empty if clean)."""

    problems = []
    with open(lbl_path, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                problems.append(f"line {lineno}: expected 5 tokens, got {len(parts)}")
                continue

            cls_str, *coords = parts

            if not cls_str.lstrip("-").isdigit():
                problems.append(f"line {lineno}: class id '{cls_str}' is not an integer")
                continue
            cls_id = int(cls_str)
            if num_classes and not (0 <= cls_id < num_classes):
                problems.append(f"line {lineno}: class id {cls_id} outside [0, {num_classes - 1}]")

            for name, val in zip(("x_center", "y_center", "width", "height"), coords):
                try:
                    f_val = float(val)
                except ValueError:
                    problems.append(f"line {lineno}: {name}='{val}' is not a number")
                    continue
                if not (-COORD_EPS <= f_val <= 1 + COORD_EPS):
                    problems.append(f"line {lineno}: {name}={f_val} outside [0, 1]")

    return problems


def hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def verify(dataset_path: Path):
    class_names = load_class_names(dataset_path)
    num_classes = len(class_names)

    total_images = 0
    total_labels = 0
    corrupt_images = []
    missing_labels = []
    orphan_labels = []
    malformed_labels: dict[str, list[str]] = {}
    class_counts: dict[str, int] = {name: 0 for name in class_names}
    image_hashes: dict[str, list[str]] = {}  # hash -> ["train:foo.jpg", ...]

    for split in SPLITS:
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"

        if not image_dir.exists():
            print(f"⚠ {split}: no images/ directory — skipping.")
            continue

        images = [p for p in image_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
        labels = {p.stem: p for p in label_dir.glob("*.txt")} if label_dir.exists() else {}

        total_images += len(images)

        for img in images:
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception as e:
                corrupt_images.append(f"{split}/{img.name}: {e}")
                continue

            digest = hash_file(img)
            image_hashes.setdefault(digest, []).append(f"{split}:{img.name}")

            lbl = labels.pop(img.stem, None)
            if lbl is None:
                missing_labels.append(f"{split}/{img.name}")
                continue

            total_labels += 1
            problems = validate_label_file(lbl, num_classes)
            if problems:
                malformed_labels[f"{split}/{lbl.name}"] = problems

            if num_classes:
                with open(lbl, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        cls_id = line.split()[0]
                        if cls_id.lstrip("-").isdigit() and 0 <= int(cls_id) < num_classes:
                            class_counts[class_names[int(cls_id)]] += 1

        # Anything left in `labels` had no matching image.
        orphan_labels.extend(f"{split}/{p.name}" for p in labels.values())

    duplicates = {h: files for h, files in image_hashes.items() if len(files) > 1}
    cross_split_dupes = {
        h: files for h, files in duplicates.items()
        if len({f.split(":")[0] for f in files}) > 1
    }

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET VERIFICATION REPORT")
    print("=" * 60)
    print(f"Dataset path     : {dataset_path}")
    print(f"Classes          : {class_names if class_names else 'unknown (no data.yaml)'}")
    print(f"Total images     : {total_images}")
    print(f"Total labels     : {total_labels}")
    print(f"Corrupt images   : {len(corrupt_images)}")
    print(f"Missing labels   : {len(missing_labels)}")
    print(f"Orphan labels    : {len(orphan_labels)}")
    print(f"Malformed labels : {len(malformed_labels)} file(s)")
    print(f"Duplicate images : {len(duplicates)} group(s), "
          f"{sum(len(f) for f in cross_split_dupes.values())} across DIFFERENT splits")

    if class_names:
        print("\nPer-class instance counts:")
        total = sum(class_counts.values())
        for name in class_names:
            count = class_counts[name]
            pct = (count / total * 100) if total else 0
            print(f"  {name:<18} {count:>6}  ({pct:5.1f}%)")
        if total and max(class_counts.values(), default=0) > 3 * (total / max(len(class_names), 1)):
            print("  ⚠ Noticeably imbalanced — consider class weighting or oversampling.")

    if corrupt_images:
        print(f"\nCorrupt images ({len(corrupt_images)}):")
        for x in corrupt_images[:10]:
            print(f"  - {x}")
        if len(corrupt_images) > 10:
            print(f"  ... and {len(corrupt_images) - 10} more")

    if missing_labels:
        print(f"\nImages with no label file ({len(missing_labels)}):")
        for x in missing_labels[:10]:
            print(f"  - {x}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    if orphan_labels:
        print(f"\nLabel files with no matching image ({len(orphan_labels)}):")
        for x in orphan_labels[:10]:
            print(f"  - {x}")
        if len(orphan_labels) > 10:
            print(f"  ... and {len(orphan_labels) - 10} more")

    if malformed_labels:
        print(f"\nMalformed label files ({len(malformed_labels)}):")
        for fname, problems in list(malformed_labels.items())[:10]:
            print(f"  - {fname}:")
            for p in problems[:3]:
                print(f"      {p}")

    if cross_split_dupes:
        print(f"\n⚠ Images duplicated ACROSS splits (data leakage risk) — "
              f"{len(cross_split_dupes)} group(s):")
        for h, files in list(cross_split_dupes.items())[:10]:
            print(f"  - {files}")
        print("  These inflate validation/test metrics. Consider removing "
              "the duplicate from all but one split.")

    print("\n" + "=" * 60)
    ok = not (corrupt_images or missing_labels or orphan_labels or malformed_labels or cross_split_dupes)
    print("✅ Dataset looks clean." if ok else "⚠ Issues found — see above.")
    print("=" * 60)

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    verify(args.dataset)


if __name__ == "__main__":
    main()
