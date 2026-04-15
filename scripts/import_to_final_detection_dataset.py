#!/usr/bin/env python3
"""
Import images into `data/final_detection_dataset` for the curation UI.

Tasks supported:
1) Add arbitrary images (e.g. data/new_photos/...) into:
   - data/final_detection_dataset/images/
   - data/final_detection_dataset/bbox_vis/   (UI queue)
   and create empty labels:
   - data/final_detection_dataset/labels/<stem>.txt

2) Import `data/strawberries/{training,validation}` samples that already have
   non-empty YOLO labels:
   - copy image -> final_detection_dataset/images
   - copy label -> final_detection_dataset/labels
   - generate visualization with boxes -> final_detection_dataset/bbox_vis

Safety:
- Never overwrites existing files; if name collides, appends a numeric suffix.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]

DET_ROOT = REPO_ROOT / "data" / "final_detection_dataset"
DET_IMAGES = DET_ROOT / "images"
DET_LABELS = DET_ROOT / "labels"
DET_BBOX_VIS = DET_ROOT / "bbox_vis"

STRAW_ROOT = REPO_ROOT / "data" / "strawberries"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _clamp01(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(0.0, min(1.0, float(v)))


def _sanitize_stem(stem: str) -> str:
    # Keep unicode, but remove path separators and make spaces safe.
    stem = stem.replace("/", "_").replace("\\", "_")
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^\w\-\.\(\)\u0400-\u04FF]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "img"


def _non_overwriting_path(dir_path: Path, filename: str) -> Path:
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dir_path / f"{base}{ext}"
    if not candidate.exists():
        return candidate
    for k in range(1, 10_000):
        candidate = dir_path / f"{base}__{k}{ext}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Too many collisions for {filename} in {dir_path}")


def _parse_yolo_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    out: List[Tuple[int, float, float, float, float]] = []
    txt = label_path.read_text(encoding="utf-8", errors="ignore")
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            continue
        try:
            cid = int(float(toks[0]))
            xc, yc, w, h = map(float, toks[1:])
        except Exception:
            continue
        out.append((cid, xc, yc, w, h))
    return out


def _label_has_any_box(label_path: Path) -> bool:
    return len(_parse_yolo_label_file(label_path)) > 0


def _draw_boxes(image_path: Path, items: List[Tuple[int, float, float, float, float]], out_path: Path) -> None:
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
        color = (int((cid * 73) % 255), int((cid * 151) % 255), int((cid * 199) % 255))
        lw = max(2, int(min(w_img, h_img) * 0.003))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        if font is not None:
            draw.text((x1 + 2, max(0, y1 - 12)), str(cid), fill=color, font=font)
    _safe_mkdir(out_path.parent)
    img.save(out_path, quality=95)


def _copy_file(src: Path, dst: Path) -> None:
    _safe_mkdir(dst.parent)
    shutil.copy2(src, dst)


def _ensure_empty_label(dst_label: Path) -> None:
    if dst_label.exists():
        return
    _safe_mkdir(dst_label.parent)
    dst_label.write_text("", encoding="utf-8")


def add_new_photos(image_paths: Iterable[Path]) -> None:
    for src in image_paths:
        if not _is_image(src):
            raise SystemExit(f"Not an image file: {src}")
        stem = _sanitize_stem(src.stem)
        ext = src.suffix.lower()
        dst_img = _non_overwriting_path(DET_IMAGES, f"{stem}{ext}")
        dst_vis = DET_BBOX_VIS / dst_img.name
        dst_lbl = DET_LABELS / f"{Path(dst_img.name).stem}.txt"

        _copy_file(src, dst_img)
        _copy_file(src, dst_vis)  # no boxes yet; UI will draw overlay from labels (currently empty)
        _ensure_empty_label(dst_lbl)


def import_strawberries() -> None:
    for split in ("training", "validation"):
        d = STRAW_ROOT / split
        if not d.is_dir():
            continue
        for img in sorted(d.iterdir()):
            if not _is_image(img):
                continue
            lbl = d / f"{img.stem}.txt"
            if not lbl.is_file():
                continue
            if not _label_has_any_box(lbl):
                continue

            # Prefix to avoid collisions with existing final dataset names.
            stem = _sanitize_stem(f"strawberries_{split}__{img.stem}")
            ext = img.suffix.lower()
            dst_img = _non_overwriting_path(DET_IMAGES, f"{stem}{ext}")
            dst_lbl = DET_LABELS / f"{Path(dst_img.name).stem}.txt"
            dst_vis = DET_BBOX_VIS / dst_img.name

            _copy_file(img, dst_img)
            _copy_file(lbl, dst_lbl)
            items = _parse_yolo_label_file(dst_lbl)
            _draw_boxes(dst_img, items, dst_vis)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="Image paths to add (copied to images + bbox_vis, empty label created)")
    ap.add_argument("--import-strawberries", action="store_true", help="Import labeled strawberries training/validation")
    args = ap.parse_args()

    _safe_mkdir(DET_IMAGES)
    _safe_mkdir(DET_LABELS)
    _safe_mkdir(DET_BBOX_VIS)

    add_list = [Path(p).resolve() for p in args.add]
    if add_list:
        add_new_photos(add_list)
        print(f"added_images: {len(add_list)}")
    if args.import_strawberries:
        import_strawberries()
        print("imported_strawberries: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

