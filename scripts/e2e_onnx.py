#!/usr/bin/env python3
"""End-to-end dry run of the on-device pipeline, ONNX models only.

For each full-frame image: YOLO det -> crop the box -> CRNN rec -> compare the
string against the generator manifest. This mirrors exactly what the Android
OnnxOcrEngine will do.
"""
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CHARS = "0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ "
IMG_H, MAX_W = 32, 192
DET_SIZE = 640


def letterbox(img: Image.Image, size: int):
    scale = size / max(img.width, img.height)
    w, h = round(img.width * scale), round(img.height * scale)
    resized = img.resize((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, ((size - w) // 2, (size - h) // 2))
    return canvas, scale, (size - w) // 2, (size - h) // 2


def detect(sess, img: Image.Image):
    canvas, scale, pad_x, pad_y = letterbox(img, DET_SIZE)
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    out = sess.run(None, {"images": arr})[0][0].T  # (anchors, 4+1cls)
    boxes = []
    for row in out:
        conf = row[4:].max()
        if conf < 0.4:
            continue
        cx, cy, w, h = row[:4]
        boxes.append((conf, (cx - w / 2 - pad_x) / scale, (cy - h / 2 - pad_y) / scale,
                      (cx + w / 2 - pad_x) / scale, (cy + h / 2 - pad_y) / scale))
    boxes.sort(key=lambda b: -b[0])
    picked = []
    for b in boxes:  # greedy NMS
        if all(iou(b, p) < 0.5 for p in picked):
            picked.append(b)
    return picked


def iou(a, b):
    x0, y0 = max(a[1], b[1]), max(a[2], b[2])
    x1, y1 = min(a[3], b[3]), min(a[4], b[4])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area = lambda r: (r[3] - r[1]) * (r[4] - r[2])
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0


def recognise(sess, crop: Image.Image) -> str:
    img = crop.convert("L")
    scale = IMG_H / img.height
    w = max(8, min(MAX_W, round(img.width * scale)))
    img = img.resize((w, IMG_H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    if w < MAX_W:
        arr = np.pad(arr, ((0, 0), (0, MAX_W - w)), constant_values=1.0)
    out = sess.run(["log_probs"], {"image": arr[None, None]})[0]
    indices = out.argmax(2)[:, 0]
    text, prev = [], 0
    for i in indices.tolist():
        if i != 0 and i != prev:
            text.append(CHARS[i - 1])
        prev = i
    return "".join(text)


def normalise(code: str) -> str:
    return re.sub(r"[\s\-]", "", code).upper()


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/synth_e2e"
    manifest = (data / "det" / "manifest.txt").read_text(encoding="utf-8").strip().splitlines()
    det = ort.InferenceSession(str(ROOT / "export/det_run/weights/best.onnx"),
                               providers=["CPUExecutionProvider"])
    rec = ort.InferenceSession(str(ROOT / "export/rec_model.onnx"),
                               providers=["CPUExecutionProvider"])

    found = exact = fuzzy = total = 0
    for line in manifest:
        name, truth = line.split("\t", 1)
        img = Image.open(data / "det" / "images" / name)
        boxes = detect(det, img)
        total += 1
        if not boxes:
            print(f"  NO-DET {name} truth={truth}")
            continue
        found += 1
        conf, x0, y0, x1, y1 = boxes[0]
        pad_x, pad_y = (x1 - x0) * 0.06, (y1 - y0) * 0.15
        crop = img.crop((max(0, x0 - pad_x), max(0, y0 - pad_y),
                         min(img.width, x1 + pad_x), min(img.height, y1 + pad_y)))
        pred = recognise(rec, crop)
        if pred == truth:
            exact += 1
        elif normalise(pred) == normalise(truth):
            fuzzy += 1
        else:
            print(f"  MISS {name} truth={truth!r} pred={pred!r} conf={conf:.2f}")

    print(f"\ntotal {total} | det found {found} | rec exact {exact} | "
          f"exact ignoring dash/space {fuzzy}")
    print(f"end-to-end exact-rate {exact / total:.4f} | loose-rate {(exact + fuzzy) / total:.4f}")


if __name__ == "__main__":
    main()
