#!/usr/bin/env python3
"""Train a small CRNN-CTC recogniser for pickup-code text lines (CPU friendly).

Input:  data/synth/rec/images/*.png + labels.txt (name<TAB>text)
Output: export/rec_model.pt (best checkpoint) + export/rec_model.onnx

The model reads a grayscale 32px-high text line and outputs a character sequence.
Charset covers digits, dash, uppercase letters and space; CTC blank is index 0.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
CHARS = "0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ "
BLANK = 0
NUM_CLASSES = len(CHARS) + 1  # + CTC blank at index 0
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}

IMG_H = 32
MAX_W = 192


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def conv(cin, cout, k=(3, 3), s=(1, 1), p=(1, 1)):
            return nn.Sequential(
                nn.Conv2d(cin, cout, k, s, p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.cnn = nn.Sequential(
            conv(1, 32), nn.MaxPool2d(2, 2),          # 1x32xW   -> 32x16xW/2
            conv(32, 64), nn.MaxPool2d(2, 2),         #          -> 64x8xW/4
            conv(64, 128), conv(128, 128),
            nn.MaxPool2d((2, 1)),                     #          -> 128x4xW/4
            conv(128, 256), conv(256, 256),
            nn.MaxPool2d((2, 1)),                     #          -> 256x2xW/4
            conv(256, 256, k=(2, 1), p=(0, 0)),       #          -> 256x1xW/4
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):  # x: (B, 1, 32, W)
        feat = self.cnn(x)                # (B, 256, 1, T)
        feat = feat.squeeze(2).permute(0, 2, 1)  # (B, T, 256)
        seq, _ = self.rnn(feat)           # (B, T, 256)
        logits = self.fc(seq)             # (B, T, C)
        return logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C) for CTC


class RecDataset(torch.utils.data.Dataset):
    def __init__(self, samples, augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, text = self.samples[i]
        img = Image.open(path).convert("L")
        if self.augment:
            # Pipeline crops are cut from rotated labels, so the model must read
            # skewed text; deskewing on-device would just add complexity.
            if random.random() < 0.85:
                angle = random.uniform(-20, 20)
                img = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=255)
            if random.random() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.25))
            if random.random() < 0.5:
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.3))
        scale = IMG_H / img.height
        w = max(8, min(MAX_W, round(img.width * scale)))
        img = img.resize((w, IMG_H), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        if w < MAX_W:  # right-pad with the "background" value of normalised white-ish
            arr = np.pad(arr, ((0, 0), (0, MAX_W - w)), constant_values=1.0)
        target = [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]
        return torch.from_numpy(arr).unsqueeze(0), torch.tensor(target, dtype=torch.long)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return imgs, torch.cat(targets), lengths


def greedy_decode(log_probs):  # (T, B, C) -> list[str]
    indices = log_probs.argmax(2).permute(1, 0)  # (B, T)
    out = []
    for row in indices:
        chars, prev = [], BLANK
        for idx in row.tolist():
            if idx != BLANK and idx != prev:
                chars.append(CHARS[idx - 1])
            prev = idx
        out.append("".join(chars))
    return out


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
    return dp[-1]


def evaluate(model, loader, device, max_batches: int = 40):
    model.eval()
    exact = total = 0
    edits = chars = 0
    with torch.no_grad():
        for b, (imgs, targets, target_lengths) in enumerate(loader):
            if b >= max_batches:
                break
            preds = greedy_decode(model(imgs.to(device)))
            flat = targets.tolist()
            pos = 0
            for pred, length in zip(preds, target_lengths.tolist()):
                truth = "".join(CHARS[i - 1] for i in flat[pos:pos + length])
                pos += length
                total += 1
                exact += pred == truth
                edits += edit_distance(pred, truth)
                chars += max(1, len(truth))
    return exact / max(1, total), edits / max(1, chars)


def load_samples(rec_dir: Path, limit: int = 0):
    lines = (rec_dir / "labels.txt").read_text(encoding="utf-8").strip().splitlines()
    samples = []
    for line in lines:
        name, text = line.split("\t", 1)
        samples.append((rec_dir / "images" / name, text))
    rng = random.Random(7)
    rng.shuffle(samples)
    if limit:
        samples = samples[:limit]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="debug: only use N samples")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synth" / "rec")
    ap.add_argument("--out", type=Path, default=ROOT / "export")
    args = ap.parse_args()

    torch.set_num_threads(max(1, 12 - 2))
    device = torch.device("cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data, args.limit)
    split = int(len(samples) * 0.97)
    train_ds = RecDataset(samples[:split], augment=True)
    val_ds = RecDataset(samples[split:], augment=False)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                           num_workers=args.workers, collate_fn=collate,
                                           persistent_workers=args.workers > 0)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False,
                                         num_workers=2, collate_fn=collate)
    print(f"train {len(train_ds)} | val {len(val_ds)} | classes {NUM_CLASSES}", flush=True)

    model = CRNN().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"params {params/1e6:.2f}M (~{params*4/1e6:.1f} MB fp32)", flush=True)

    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    history = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for imgs, targets, target_lengths in train_ld:
            imgs = imgs.to(device)
            log_probs = model(imgs)
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, target_lengths)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        acc, cer = evaluate(model, val_ld, device)
        dt = time.time() - t0
        line = (f"epoch {epoch:02d}/{args.epochs} | loss {running/seen:.4f} | "
                f"val acc {acc:.4f} | CER {cer:.4f} | {dt:.0f}s")
        print(line, flush=True)
        history.append(dict(epoch=epoch, loss=running / seen, acc=acc, cer=cer))
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "acc": acc, "epoch": epoch},
                       args.out / "rec_model.pt")

    (args.out / "rec_history.json").write_text(json.dumps(history, indent=2))
    print(f"best val acc {best_acc:.4f}; checkpoint -> {args.out/'rec_model.pt'}", flush=True)

    # ONNX export: dynamic batch & width so the app can pick any crop size.
    model.eval()
    dummy = torch.zeros(1, 1, IMG_H, MAX_W)
    torch.onnx.export(
        model, dummy, args.out / "rec_model.onnx",
        input_names=["image"], output_names=["log_probs"],
        dynamic_axes={"image": {0: "batch", 3: "width"},
                      "log_probs": {0: "seq", 1: "batch"}},
        opset_version=17,
        dynamo=False,  # classic TorchScript exporter: handles LSTM, the new one chokes on it
    )
    print(f"onnx -> {args.out/'rec_model.onnx'}", flush=True)


if __name__ == "__main__":
    main()
