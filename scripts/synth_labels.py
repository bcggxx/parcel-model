#!/usr/bin/env python3
"""Synthetic express-label generator for pickup-code detection & recognition.

Outputs two datasets:
  det/  full label photos (JPEG) + YOLO annotations for the pickup-code box
  rec/  tight text-line crops (PNG) + labels.txt (filename<TAB>text) for CRNN

Everything is procedurally generated: unlimited data, zero manual labelling.
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = ROOT / "fonts"

CJK = FONTS_DIR / "NotoSansSC.ttf"
DIGIT_FONTS = [
    FONTS_DIR / "Oswald.ttf",
    FONTS_DIR / "RobotoMono.ttf",
    FONTS_DIR / "JetBrainsMono.ttf",
    FONTS_DIR / "BarlowCondensed-Bold.ttf",
    FONTS_DIR / "BarlowCondensed-SemiBold.ttf",
]
# System fallbacks so the script also runs before fonts are downloaded.
SYSTEM_DIGIT_FONTS = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]

SURNAMES = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭"
GIVEN = "伟芳娜敏静丽强磊军洋勇艳杰娟涛明超霞平刚桂英华玉兰梅鑫宇欣怡"
CITIES = ["杭州市", "上海市", "北京市", "广州市", "深圳市", "成都市", "武汉市", "南京市",
          "苏州市", "重庆市", "西安市", "长沙市", "郑州市", "青岛市", "合肥市", "福州市"]
STREETS = ["文一西路", "中山北路", "解放大道", "滨江东路", "高新南七道", "望江路",
           "学院路", "长安街", "天府三街", "光谷大道", "软件大道", "金鸡湖大道"]
BRANDS = ["ZT 中通快递", "YT 圆通速递", "ST 申通快递", "YD 韵达快递", "SF 顺丰速运",
          "JD 京东物流", "EMS 邮政速递", "JT 极兔速递"]
BRAND_COLORS = [(29, 78, 216), (220, 38, 38), (234, 88, 12), (2, 132, 199),
                (5, 150, 105), (124, 58, 237), (190, 18, 60), (15, 23, 42)]


_FONT_CACHE: dict[tuple[str, int, int | None], ImageFont.FreeTypeFont] = {}


def load_font(path: Path, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    key = (str(path), size, weight)
    font = _FONT_CACHE.get(key)
    if font is None:
        font = ImageFont.truetype(str(path), size)
        if weight is not None:
            try:
                font.set_variation_by_axes([weight])
            except Exception:
                pass  # static font, weight baked in
        _FONT_CACHE[key] = font
    return font


def available(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def draw_text_jittered(draw: ImageDraw.ImageDraw, xy, text, font, fill, rng,
                       tracking: float = 1.0, max_width: float | None = None):
    """Draw text char by char with small per-char advance jitter (thermal-print feel).

    Returns the ink bounding box (x0, y0, x1, y1) actually covered.
    """
    x, y = xy
    x0, y0, x1, y1 = x, y, x, y
    for ch in text:
        bbox = draw.textbbox((x, y), ch, font=font)
        if max_width is not None and bbox[2] - x0 > max_width:
            break
        draw.text((x, y), ch, font=font, fill=fill)
        x0 = min(x0, bbox[0]); y0 = min(y0, bbox[1])
        x1 = max(x1, bbox[2]); y1 = max(y1, bbox[3])
        advance = draw.textlength(ch, font=font)
        x += advance * tracking * rng.uniform(0.96, 1.06)
    return x0, y0, x1, y1


def make_pickup_code(rng: random.Random) -> str:
    """Formats seen on real shelves: 8-3-4025, A8-3-4025, 12-105-31, 834025 ..."""
    style = rng.random()
    if style < 0.55:
        code = f"{rng.randint(1, 20)}-{rng.randint(1, 9)}-{rng.randint(1000, 9999)}"
    elif style < 0.75:
        code = f"{rng.randint(1, 99)}-{rng.randint(10, 999)}-{rng.randint(10, 99)}"
    elif style < 0.85:
        code = f"{rng.randint(1, 9)}-{rng.randint(10000, 99999)}"
    else:
        code = "".join(rng.choices("0123456789", k=rng.choice([6, 7, 8])))
    if rng.random() < 0.25:
        code = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") + code
    if rng.random() < 0.12:
        code = code.replace("-", " ")
    return code


def paper_texture(w: int, h: int, rng: random.Random, base=(255, 255, 255)) -> Image.Image:
    noise = np.random.default_rng(rng.randrange(2**63)).normal(0, 1.6, (h, w, 1))
    arr = np.clip(np.array(base, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    arr = np.repeat(arr, 3, axis=2) if arr.shape[2] == 1 else arr
    return Image.fromarray(arr, "RGB")


def draw_barcode(draw: ImageDraw.ImageDraw, x, y, w, h, rng, color=(20, 20, 20)):
    cursor = x
    while cursor < x + w:
        bw = rng.choice([1, 1, 2, 2, 3, 4])
        gap = rng.choice([1, 1, 2, 3])
        if cursor + bw <= x + w and rng.random() > 0.06:
            draw.rectangle([cursor, y, cursor + bw, y + h], fill=color)
        cursor += bw + gap


def draw_qr(draw: ImageDraw.ImageDraw, x, y, size, rng, color=(20, 20, 20)):
    n = 21
    cell = size / n
    grid = np.random.default_rng(rng.randrange(2**63)).random((n, n)) < 0.45
    for finder in ((0, 0), (n - 7, 0), (0, n - 7)):
        grid[finder[1]:finder[1] + 7, finder[0]:finder[0] + 7] = False
        fx, fy = finder
        draw.rectangle([x + fx * cell, y + fy * cell, x + (fx + 7) * cell, y + (fy + 7) * cell], outline=color, width=max(1, int(cell)))
        draw.rectangle([x + (fx + 2) * cell, y + (fy + 2) * cell, x + (fx + 5) * cell, y + (fy + 5) * cell], fill=color)
    for r in range(n):
        for c in range(n):
            if grid[r, c]:
                draw.rectangle([x + c * cell, y + r * cell, x + (c + 1) * cell, y + (r + 1) * cell], fill=color)


def render_label(rng: random.Random):
    """Render one express label; returns (image, code_box) with code_box=(x0,y0,x1,y1)."""
    w = rng.randint(880, 1060)
    h = rng.randint(560, 720)
    img = paper_texture(w, h, rng, base=(252, 251, 248) if rng.random() < 0.7 else (255, 255, 255))
    draw = ImageDraw.Draw(img)
    ink = (25, 25, 25)
    brand_color = rng.choice(BRAND_COLORS)
    cjk = CJK if CJK.exists() else None

    # Header: brand + colored block
    draw.rectangle([0, 0, w, rng.randint(46, 62)], fill=brand_color)
    if cjk:
        draw.text((rng.randint(14, 30), rng.randint(8, 14)), rng.choice(BRANDS),
                  font=load_font(cjk, rng.randint(26, 34), 700), fill=(255, 255, 255))
    draw.rectangle([w - rng.randint(120, 200), 8, w - 16, rng.randint(38, 50)], outline=(255, 255, 255), width=2)

    y = rng.randint(62, 78)

    # Barcode + waybill number
    bc_h = rng.randint(52, 78)
    draw_barcode(draw, rng.randint(20, 60), y, rng.randint(int(w * 0.45), int(w * 0.7)), bc_h, rng)
    waybill = rng.choice(["YT", "ZT", "SF", "YD", ""]) + "".join(rng.choices("0123456789", k=12))
    dfont = load_font(rng.choice(available(DIGIT_FONTS) or available(SYSTEM_DIGIT_FONTS)), rng.randint(24, 32), 500)
    draw.text((rng.randint(30, 70), y + bc_h + 6), waybill, font=dfont, fill=ink)
    y += bc_h + rng.randint(44, 60)

    # Receiver / sender lines (Chinese distractor text)
    if cjk:
        name = rng.choice(SURNAMES) + rng.choice(GIVEN) + rng.choice(GIVEN)
        phone = "1" + rng.choice("356789") + "".join(rng.choices("0123456789", k=9))
        addr = rng.choice(CITIES) + rng.choice(STREETS) + f"{rng.randint(1, 200)}号{rng.randint(1, 30)}栋{rng.randint(101, 2504)}室"
        body = load_font(cjk, rng.randint(20, 26), 400)
        for line in (f"收  {name}  {phone}", addr, f"寄  {rng.choice(SURNAMES)}先生  {rng.choice(CITIES)}"):
            draw.text((rng.randint(20, 40), y), line, font=body, fill=ink)
            y += rng.randint(30, 38)
        draw.line([16, y, w - 16, y], fill=(120, 120, 120), width=2)
        y += rng.randint(10, 18)

    # The pickup code, the star of the show: big bold digits, often boxed.
    code = make_pickup_code(rng)
    code_size = rng.randint(52, 92)
    code_font = load_font(rng.choice(available(DIGIT_FONTS) or available(SYSTEM_DIGIT_FONTS)), code_size, rng.choice([500, 600, 700, 800]))
    cx = rng.randint(30, max(31, w - 420))
    cy = min(y + rng.randint(4, 20), h - code_size - 30)
    boxed = rng.random() < 0.5
    if boxed:
        pad = rng.randint(10, 22)
        est_w = draw.textlength(code, font=code_font) * 1.08
        draw.rounded_rectangle([cx - pad, cy - pad, cx + est_w + pad, cy + code_size * 1.35 + pad],
                               radius=rng.randint(6, 16), outline=ink, width=rng.randint(2, 5))
    x0, yy0, x1, yy1 = draw_text_jittered(draw, (cx, cy), code, code_font, ink, rng,
                                          tracking=rng.uniform(0.98, 1.12))

    # Extra distractors: QR, small print, dates.
    if rng.random() < 0.7:
        qr_size = rng.randint(84, 130)
        draw_qr(draw, w - qr_size - rng.randint(16, 40), rng.randint(70, max(71, h - qr_size - 20)), qr_size, rng)
    if cjk and rng.random() < 0.8:
        small = load_font(cjk, rng.randint(14, 18), 400)
        draw.text((rng.randint(20, 60), h - rng.randint(24, 34)),
                  f"已验视  2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} {rng.randint(8, 20)}:{rng.randint(10, 59)}  重量{rng.uniform(0.1, 9.9):.2f}kg",
                  font=small, fill=(80, 80, 80))

    code_box = (x0 - 3, yy0 - 3, x1 + 3, yy1 + 3)
    return img, code_box, code


def photometric_augment(img: Image.Image, rng: random.Random, harsh: bool = False) -> Image.Image:
    if rng.random() < (0.9 if harsh else 0.6):
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6 if harsh else 1.0)))
    if rng.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.35))
    if rng.random() < 0.4:  # smooth lighting falloff: low-frequency horizontal variation
        arr = np.asarray(img).astype(np.float32)
        rows = arr.shape[0]
        nprng = np.random.default_rng(rng.randrange(2**63))
        coarse = nprng.uniform(0.85, 1.0, (max(2, rows // 16), 1))
        band_img = Image.fromarray((coarse * 255).astype(np.uint8)).resize((1, rows), Image.BILINEAR)
        bands = np.asarray(band_img, dtype=np.float32).reshape(rows, 1, 1) / 255.0
        arr = np.clip(arr * bands, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if rng.random() < 0.5:
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(rng.randrange(2**63)).normal(0, rng.uniform(2, 7), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def place_on_background(label: Image.Image, code_box, rng: random.Random,
                        frame_size=(1280, 960)):
    """Rotate the label, paste on a tabletop-like background, track the code box."""
    fw, fh = frame_size
    scale = rng.uniform(0.5, 0.95)
    label = label.resize((int(label.width * scale), int(label.height * scale)), Image.LANCZOS)
    code_box = tuple(v * scale for v in code_box)

    angle = rng.uniform(-25, 25)
    rotated = label.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    rad = math.radians(-angle)  # PIL rotates counter-clockwise
    lw, lh = label.width, label.height
    rw, rh = rotated.width, rotated.height

    def to_rotated(px, py):
        dx, dy = px - lw / 2, py - lh / 2
        return (rw / 2 + dx * math.cos(rad) - dy * math.sin(rad),
                rh / 2 + dx * math.sin(rad) + dy * math.cos(rad))

    corners = [to_rotated(code_box[0], code_box[1]), to_rotated(code_box[2], code_box[1]),
               to_rotated(code_box[2], code_box[3]), to_rotated(code_box[0], code_box[3])]

    # Tabletop: gradient + noise, occasionally a different surface tone.
    tone = rng.choice([(168, 150, 132), (196, 188, 176), (142, 160, 175), (210, 206, 200), (120, 105, 92)])
    bg = paper_texture(fw, fh, rng, base=tone)
    grad = np.linspace(rng.uniform(0.75, 0.95), 1.05, fh, dtype=np.float32)[:, None, None]
    bg = Image.fromarray(np.clip(np.asarray(bg).astype(np.float32) * grad, 0, 255).astype(np.uint8))

    ox = rng.randint(0, max(1, fw - rw))
    oy = rng.randint(0, max(1, fh - rh))
    mask = rotated.convert("L").point(lambda v: 255 if v > 8 else 0)
    bg.paste(rotated, (ox, oy), mask)

    xs = [c[0] + ox for c in corners]
    ys = [c[1] + oy for c in corners]
    box = (max(0, min(xs)), max(0, min(ys)), min(fw, max(xs)), min(fh, max(ys)))
    return bg, box


def gen_det(out_dir: Path, n: int, rng: random.Random):
    img_dir = out_dir / "det" / "images"
    lbl_dir = out_dir / "det" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n):
        label, code_box, code = render_label(rng)
        frame, box = place_on_background(label, code_box, rng)
        frame = photometric_augment(frame, rng)
        name = f"det_{i:06d}"
        frame.save(img_dir / f"{name}.jpg", quality=rng.randint(70, 92))
        cx = (box[0] + box[2]) / 2 / frame.width
        cy = (box[1] + box[3]) / 2 / frame.height
        bw = (box[2] - box[0]) / frame.width
        bh = (box[3] - box[1]) / frame.height
        (lbl_dir / f"{name}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        manifest.append(f"{name}.jpg\t{code}")
    if manifest:
        (out_dir / "det" / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def render_rec_crop(rng: random.Random):
    """One text-line crop: mostly pickup codes, some digit/serial distractors."""
    kind = rng.random()
    if kind < 0.7:
        text = make_pickup_code(rng)
    elif kind < 0.85:
        text = "".join(rng.choices("0123456789", k=rng.choice([10, 12, 13])))
    else:
        text = rng.choice(["YT", "ZT", "SF", "731A"]) + "".join(rng.choices("0123456789", k=rng.choice([9, 11])))
    size = rng.randint(30, 68)
    fonts = available(DIGIT_FONTS) or available(SYSTEM_DIGIT_FONTS)
    font = load_font(rng.choice(fonts), size, rng.choice([400, 500, 600, 700, 800]))
    probe = Image.new("RGB", (10, 10))
    pd = ImageDraw.Draw(probe)
    w = int(pd.textlength(text, font=font) * 1.25) + 30
    h = int(size * 1.6) + 24
    img = paper_texture(w, h, rng, base=(252, 251, 248) if rng.random() < 0.7 else (255, 255, 255))
    draw = ImageDraw.Draw(img)
    ink = rng.choice([(20, 20, 20), (35, 35, 40), (15, 25, 45)])
    if rng.random() < 0.3:
        draw.rounded_rectangle([2, 2, w - 3, h - 3], radius=rng.randint(4, 12),
                               outline=ink, width=rng.randint(2, 4))
    draw_text_jittered(draw, (rng.randint(10, 16), rng.randint(8, 14)), text, font, ink, rng,
                       tracking=rng.uniform(0.98, 1.12))
    return img, text


def gen_rec(out_dir: Path, n: int, rng: random.Random):
    img_dir = out_dir / "rec" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        img, text = render_rec_crop(rng)
        angle = rng.uniform(-2.5, 2.5)
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(250, 250, 248))
        img = photometric_augment(img, rng, harsh=True)
        target_h = rng.choice([32, 40, 48, 56, 64])
        img = img.resize((max(16, int(img.width * target_h / img.height)), target_h), Image.LANCZOS)
        name = f"rec_{i:06d}.png"
        img.save(img_dir / name)
        lines.append(f"{name}\t{text}")
    with open(out_dir / "rec" / "labels.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-det", type=int, default=0)
    ap.add_argument("--n-rec", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synth")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.default_rng(args.seed)
    if args.n_det:
        gen_det(args.out, args.n_det, rng)
    if args.n_rec:
        gen_rec(args.out, args.n_rec, rng)
    print(f"done: {args.n_det} det + {args.n_rec} rec -> {args.out}")


if __name__ == "__main__":
    main()
