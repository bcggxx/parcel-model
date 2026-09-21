#!/usr/bin/env python3
"""Fine-tune a YOLO nano detector to localise the pickup code on a label.

CPU-only box: keep epochs modest; the synth distribution is narrow so it
converges fast. Exports ONNX next to the run's weights.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", type=str, default="yolo11n.pt")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "data" / "det_yolo" / "det.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=4,
        project=str(ROOT / "export"),
        name="det_run",
        exist_ok=True,
        cos_lr=True,
        close_mosaic=5,      # last epochs without mosaic: sharper small-box fit
        verbose=True,
    )
    best = ROOT / "export" / "det_run" / "weights" / "best.pt"
    YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True)
    print(f"done -> {best.with_suffix('.onnx')}", flush=True)


if __name__ == "__main__":
    main()
