"""
=========================================================
PCB Dataset Merger

Purpose
-------
Merge DeepPCB + PCB Defects into one dataset.

Features
--------
✔ Merge train/valid/test
✔ Rename files
✔ Remap DeepPCB class ids
✔ Copy labels
✔ Generate final data.yaml

Author : Jasmeen
=========================================================
"""

import shutil
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
    "spurious_copper"
]

# ==========================================================
# DeepPCB ID Mapping
#
# DeepPCB
# 0 copper
# 1 mousebite
# 2 open
# 3 pin-hole
# 4 short
# 5 spur
#
# Standard
# 0 missing_hole
# 1 mouse_bite
# 2 open_circuit
# 3 short
# 4 spur
# 5 spurious_copper
# ==========================================================

DEEPPCB_MAP = {

    0:5,
    1:1,
    2:2,
    3:0,
    4:3,
    5:4

}

# PCB Defects already follows standard ids

PCB_MAP = {

    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5

}

# ==========================================================

SPLITS = ["train","valid","test"]

# ==========================================================


def make_dirs():

    for split in SPLITS:

        (OUTPUT/split/"images").mkdir(
            parents=True,
            exist_ok=True
        )

        (OUTPUT/split/"labels").mkdir(
            parents=True,
            exist_ok=True
        )


# ==========================================================


def remap_label_file(src,dst,mapping):

    if not src.exists():

        return

    new_lines=[]

    with open(src,"r") as f:

        for line in f:

            line=line.strip()

            if line=="":

                continue

            parts=line.split()

            cls=int(parts[0])

            cls=mapping[cls]

            parts[0]=str(cls)

            new_lines.append(
                " ".join(parts)
            )

    with open(dst,"w") as f:

        f.write(
            "\n".join(new_lines)
        )


# ==========================================================


def merge_dataset(dataset_path,prefix,mapping):

    print(f"\nMerging {prefix}")

    for split in SPLITS:

        image_dir=dataset_path/split/"images"

        label_dir=dataset_path/split/"labels"

        out_img=OUTPUT/split/"images"

        out_lbl=OUTPUT/split/"labels"

        images=list(image_dir.glob("*"))

        print(
            split,
            len(images),
            "images"
        )

        for img in images:

            new_name=prefix+"_"+img.name

            shutil.copy2(
                img,
                out_img/new_name
            )

            lbl=label_dir/(img.stem+".txt")

            new_lbl=out_lbl/(prefix+"_"+img.stem+".txt")

            remap_label_file(
                lbl,
                new_lbl,
                mapping
            )


# ==========================================================


def create_yaml():

    data={

        "path":str(OUTPUT),

        "train":"train/images",

        "val":"valid/images",

        "test":"test/images",

        "names":STANDARD_CLASSES

    }

    with open(
        OUTPUT/"data.yaml",
        "w"
    ) as f:

        yaml.dump(
            data,
            f,
            sort_keys=False
        )


# ==========================================================


def main():

    print("="*60)
    print("PCB DATASET MERGER")
    print("="*60)

    make_dirs()

    merge_dataset(
        DEEPPCB,
        "deep",
        DEEPPCB_MAP
    )

    merge_dataset(
        PCBDEFECT,
        "pcb",
        PCB_MAP
    )

    create_yaml()

    print("\nDone.")
    print("Combined Dataset Created Successfully.")

# ==========================================================

if __name__=="__main__":

    main()