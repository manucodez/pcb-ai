"""
=========================================================
PCB Dataset Verification

Checks
------
✔ Missing labels
✔ Missing images
✔ Empty labels
✔ Corrupted images
✔ Total statistics

Author : Jasmeen
=========================================================
"""

from pathlib import Path
from PIL import Image

# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset" / "PCB_Combined"

SPLITS = ["train", "valid", "test"]

# ==========================================================

total_images = 0
total_labels = 0

missing_labels = []
missing_images = []
empty_labels = []
corrupted_images = []

# ==========================================================

for split in SPLITS:

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = list(image_dir.glob("*.*"))
    labels = list(label_dir.glob("*.txt"))

    print(f"Images : {len(images)}")
    print(f"Labels : {len(labels)}")

    total_images += len(images)
    total_labels += len(labels)

    # ---------------------------------------
    # Check every image has a label
    # ---------------------------------------

    for img in images:

        label = label_dir / (img.stem + ".txt")

        if not label.exists():
            missing_labels.append(img.name)

        else:
            if label.stat().st_size == 0:
                empty_labels.append(label.name)

        # Check image integrity
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception:
            corrupted_images.append(img.name)

    # ---------------------------------------
    # Check every label has an image
    # ---------------------------------------

    for lbl in labels:

        found = False

        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:

            if (image_dir / (lbl.stem + ext)).exists():
                found = True
                break

        if not found:
            missing_images.append(lbl.name)

# ==========================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total Images : {total_images}")
print(f"Total Labels : {total_labels}")

print()

print(f"Missing Labels : {len(missing_labels)}")
print(f"Missing Images : {len(missing_images)}")
print(f"Empty Labels   : {len(empty_labels)}")
print(f"Corrupted Img  : {len(corrupted_images)}")

print()

if (
    len(missing_labels) == 0
    and len(missing_images) == 0
    and len(empty_labels) == 0
    and len(corrupted_images) == 0
):

    print("✅ DATASET VERIFIED SUCCESSFULLY")

else:

    print("⚠ DATASET HAS ISSUES")

    if missing_labels:
        print("\nMissing Labels:")
        print(missing_labels[:10])

    if missing_images:
        print("\nMissing Images:")
        print(missing_images[:10])

    if empty_labels:
        print("\nEmpty Labels:")
        print(empty_labels[:10])

    if corrupted_images:
        print("\nCorrupted Images:")
        print(corrupted_images[:10])