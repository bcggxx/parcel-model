#!/usr/bin/env python3
"""Verify the exported rec ONNX model: accuracy on held-out synth crops."""
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CHARS = "0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ "
IMG_H, MAX_W = 32, 192


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    scale = IMG_H / img.height
    w = max(8, min(MAX_W, round(img.width * scale)))
    img = img.resize((w, IMG_H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    if w < MAX_W:
        arr = np.pad(arr, ((0, 0), (0, MAX_W - w)), constant_values=1.0)
    return arr[None, None, :, :]


def greedy(log_probs: np.ndarray) -> str:  # (T, 1, C)
    indices = log_probs.argmax(2)[:, 0]
    out, prev = [], 0
    for i in indices.tolist():
        if i != 0 and i != prev:
            out.append(CHARS[i - 1])
        prev = i
    return "".join(out)


def main():
    lines = (ROOT / "data/synth/rec/labels.txt").read_text(encoding="utf-8").strip().splitlines()
    random.seed(3)
    random.shuffle(lines)
    lines = lines[:1000]
    sess = ort.InferenceSession(str(ROOT / "export/rec_model.onnx"),
                                providers=["CPUExecutionProvider"])
    exact = 0
    shown = 0
    for line in lines:
        name, truth = line.split("\t", 1)
        output = sess.run(["log_probs"], {"image": preprocess(ROOT / "data/synth/rec/images" / name)})[0]
        pred = greedy(output)
        exact += pred == truth
        if shown < 8 and pred != truth:
            shown += 1
            print(f"  MISS truth={truth!r} pred={pred!r}")
    print(f"onnx acc on 1000 synth crops: {exact / len(lines):.4f}")


if __name__ == "__main__":
    main()
