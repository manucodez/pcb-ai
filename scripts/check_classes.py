from pathlib import Path
import yaml

# ==========================================================
# PCB Dataset Class Checker
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

DATASETS = [
    ROOT / "dataset" / "DeepPCB.v1i.yolov11" / "data.yaml",
    ROOT / "dataset" / "PCB Defects.v1i.yolov11" / "data.yaml"
]


def read_classes(yaml_file):

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]

    return names


all_classes = {}

print("\n" + "=" * 70)
print("PCB DATASET CLASS COMPARISON")
print("=" * 70)

for dataset in DATASETS:

    print(f"\nDataset : {dataset.parent.name}")

    classes = read_classes(dataset)

    all_classes[dataset.parent.name] = classes

    print(f"Number of Classes : {len(classes)}\n")

    for idx, cls in enumerate(classes):
        print(f"{idx} -> {cls}")

print("\n" + "=" * 70)

datasets = list(all_classes.keys())

if len(datasets) == 2:

    c1 = all_classes[datasets[0]]
    c2 = all_classes[datasets[1]]

    print("\nCOMPARISON\n")

    if c1 == c2:

        print("✅ Perfect!")
        print("Both datasets have identical classes.")
        print("They can be merged directly.")

    else:

        print("⚠ Class mismatch detected.\n")

        print(f"{datasets[0]} :")
        print(c1)

        print()

        print(f"{datasets[1]} :")
        print(c2)

print("\n" + "=" * 70)