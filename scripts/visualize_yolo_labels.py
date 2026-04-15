#!/usr/bin/env python3
"""Draw YOLO-format bounding boxes on images; save previews for QA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Same as scripts/remap_labels_to_unified.py TARGET_CLASS_NAMES
CLASS_NAMES: Dict[int, str] = {
    0: "green",
    1: "turning",
    2: "ripe",
    3: "rotten_or_overripe",
}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clamp01(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(0.0, min(1.0, float(v)))


def _parse_yolo(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    items: List[Tuple[int, float, float, float, float]] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            continue
        try:
            cid = int(float(toks[0]))
            xc, yc, w, h = map(float, toks[1:])
        except ValueError:
            continue
        items.append((cid, xc, yc, w, h))
    return items


def _draw(
    image_path: Path,
    items: List[Tuple[int, float, float, float, float]],
    out_path: Path,
    class_names: Dict[int, str],
) -> None:
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for cid, xc, yc, bw, bh in items:
        xc, yc, bw, bh = _clamp01(xc), _clamp01(yc), _clamp01(bw), _clamp01(bh)
        x1 = (xc - bw / 2.0) * w_img
        y1 = (yc - bh / 2.0) * h_img
        x2 = (xc + bw / 2.0) * w_img
        y2 = (yc + bh / 2.0) * h_img
        x1 = max(0.0, min(w_img - 1.0, x1))
        y1 = max(0.0, min(h_img - 1.0, y1))
        x2 = max(0.0, min(w_img - 1.0, x2))
        y2 = max(0.0, min(h_img - 1.0, y2))
        color = (
            int((cid * 73) % 255),
            int((cid * 151) % 255),
            int((cid * 199) % 255),
        )
        lw = max(2, int(min(w_img, h_img) * 0.003))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        name = class_names.get(cid, "?")
        text = f"{cid}:{name}"
        if font is not None:
            draw.text((x1 + 2, max(0, y1 - 12)), text, fill=color, font=font)

    _safe_mkdir(out_path.parent)
    img.save(out_path, quality=95)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Save images with YOLO boxes drawn for inspection.")
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"Not a directory: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Not a directory: {labels_dir}")

    _safe_mkdir(out_dir)
    n_ok = 0
    n_skip = 0
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl = labels_dir / f"{img_path.stem}.txt"
        if not lbl.is_file():
            n_skip += 1
            continue
        items = _parse_yolo(lbl)
        out_path = out_dir / img_path.name
        _draw(img_path, items, out_path, CLASS_NAMES)
        n_ok += 1

    (out_dir / "_visualize_report.txt").write_text(
        f"images_drawn: {n_ok}\nimages_skipped_no_label: {n_skip}\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
