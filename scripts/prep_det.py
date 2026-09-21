#!/usr/bin/env python3
"""Split the synthetic detection set into YOLO train/val layout and write det.yaml."""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "synth" / "det"
DST = ROOT / "data" / "det_yolo"
VAL_RATIO = 0.05

random.seed(11)

images = sorted((SRC / "images").glob("*.jpg"))
random.shuffle(images)
n_val = int(len(images) * VAL_RATIO)
splits = {"val": images[:n_val], "train": images[n_val:]}

for split, files in splits.items():
    img_dir = DST / "images" / split
    lbl_dir = DST / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img in files:
        shutil.move(str(img), img_dir / img.name)
        label = SRC / "labels" / (img.stem + ".txt")
        shutil.move(str(label), lbl_dir / label.name)
    print(f"{split}: {len(files)}")

(DST / "det.yaml").write_text(
    f"path: {DST}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: pickup_code\n",
    encoding="utf-8",
)
print(f"yaml -> {DST / 'det.yaml'}")
