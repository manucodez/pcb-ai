"""
=========================================================
PCB Dataset Class Checker

Compares the class list of two or more YOLO datasets so you
can confirm they line up (or see exactly how they don't) before
merge_datasets.py remaps and combines them.

Usage
-----
    python scripts/check_classes.py
    python scripts/check_classes.py --datasets path/a/data.yaml path/b/data.yaml
=========================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASETS = [
    ROOT / "dataset" / "DeepPCB.v1i.yolov11" / "data.yaml",
    ROOT / "dataset" / "PCB Defects.v1i.yolov11" / "data.yaml",
]


def read_classes(yaml_file: Path) -> list[str]:

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    if "names" not in data:
        raise ValueError(f"{yaml_file} has no 'names' key.")

    names = data["names"]

    if isinstance(names, dict):
        # Bug fixed here: dict keys from a YOLO data.yaml are
        # usually ints ("0: missing_hole"), but PyYAML will load
        # them as strings if the file happens to quote them
        # ("'0': missing_hole"). sorted(names.keys()) then sorts
        # lexicographically — "0","1","10","2",... — silently
        # returning classes in the wrong order, which is exactly
        # the kind of mistake that then corrupts every downstream
        # class-id remap without ever raising an error. Sorting by
        # int(key) is correct for both cases.
        names = [names[k] for k in sorted(names.keys(), key=lambda k: int(k))]

    return list(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", type=Path, default=DEFAULT_DATASETS,
        help="Paths to two or more data.yaml files to compare.",
    )
    args = parser.parse_args()

    all_classes: dict[str, list[str]] = {}

    print("\n" + "=" * 70)
    print("PCB DATASET CLASS COMPARISON")
    print("=" * 70)

    for dataset in args.datasets:
        if not dataset.exists():
            print(f"\nDataset : {dataset}")
            print("  ⚠ File not found — skipping.")
            continue

        print(f"\nDataset : {dataset.parent.name}")
        classes = read_classes(dataset)
        all_classes[dataset.parent.name] = classes

        print(f"Number of Classes : {len(classes)}\n")
        for idx, cls in enumerate(classes):
            print(f"{idx} -> {cls}")

    print("\n" + "=" * 70)

    names = list(all_classes.keys())
    if len(names) == 2:
        c1, c2 = all_classes[names[0]], all_classes[names[1]]

        print("\nCOMPARISON\n")
        if c1 == c2:
            print("✅ Perfect!")
            print("Both datasets have identical classes.")
            print("They can be merged directly.")
        else:
            print("⚠ Class mismatch detected.\n")
            print(f"{names[0]} :")
            print(c1)
            print()
            print(f"{names[1]} :")
            print(c2)
    elif len(names) > 2:
        print(f"\n{len(names)} datasets loaded — pairwise comparison only runs for exactly 2.")
        print("Class lists printed above; compare manually or narrow with --datasets.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
