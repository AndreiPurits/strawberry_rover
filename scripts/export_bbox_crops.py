#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _yolo_to_xyxy(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    pad_frac: float = 0.0,
) -> Tuple[int, int, int, int]:
    # Convert normalized YOLO bbox to pixel xyxy with optional padding.
    # xc,yc,w,h are expected in [0,1] but we clamp defensively.
    if any(math.isnan(v) or math.isinf(v) for v in (xc, yc, w, h)):
        return (0, 0, 0, 0)

    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    bw = w * img_w
    bh = h * img_h
    px = pad_frac * bw
    py = pad_frac * bh

    x1 = (xc * img_w) - (bw / 2.0) - px
    y1 = (yc * img_h) - (bh / 2.0) - py
    x2 = (xc * img_w) + (bw / 2.0) + px
    y2 = (yc * img_h) + (bh / 2.0) + py

    x1i = _clamp(int(math.floor(x1)), 0, img_w - 1)
    y1i = _clamp(int(math.floor(y1)), 0, img_h - 1)
    x2i = _clamp(int(math.ceil(x2)), 0, img_w)
    y2i = _clamp(int(math.ceil(y2)), 0, img_h)

    return (x1i, y1i, x2i, y2i)


def _parse_yolo_bbox_lines(text: str) -> List[Tuple[int, float, float, float, float]]:
    out: List[Tuple[int, float, float, float, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            # Skip non-bbox lines (should not happen for strawberry_ds normalized).
            continue
        try:
            cid = int(float(toks[0]))
            xc, yc, w, h = map(float, toks[1:])
        except Exception:
            continue
        out.append((cid, xc, yc, w, h))
    return out


@dataclass
class CropStats:
    total_crops: int = 0
    total_crops_padded: int = 0
    per_class: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    per_class_padded: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    tiny_bboxes: int = 0
    errors: int = 0
    error_notes: List[str] = field(default_factory=list)


def export_crops_for_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    pad_frac: float,
    tiny_px: int,
    preview_count: int,
) -> CropStats:
    stats = CropStats()
    _safe_mkdir(out_dir)

    label_files = sorted(p for p in labels_dir.glob("*.txt") if p.is_file())
    if not label_files:
        stats.error_notes.append(f"No label files found in {labels_dir}")
        return stats

    for lf in label_files:
        stem = lf.stem
        img_path = _find_image(images_dir, stem)
        if img_path is None:
            stats.errors += 1
            stats.error_notes.append(f"Missing image for label {lf.name}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            stats.errors += 1
            stats.error_notes.append(f"Failed to open image {img_path.name}: {type(e).__name__}: {e}")
            continue

        img_w, img_h = img.size
        try:
            items = _parse_yolo_bbox_lines(lf.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            stats.errors += 1
            stats.error_notes.append(f"Failed to read label {lf.name}: {type(e).__name__}: {e}")
            continue

        for obj_i, (cid, xc, yc, w, h) in enumerate(items):
            x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, w, h, img_w, img_h, pad_frac=0.0)
            if x2 <= x1 or y2 <= y1:
                stats.errors += 1
                stats.error_notes.append(f"Invalid bbox (empty) {stem} obj{obj_i:03d} class{cid}")
                continue

            bw = x2 - x1
            bh = y2 - y1
            if bw < tiny_px or bh < tiny_px:
                stats.tiny_bboxes += 1

            crop = img.crop((x1, y1, x2, y2))

            class_dir = out_dir / f"class_{cid}"
            _safe_mkdir(class_dir)
            name = f"{stem}__obj{obj_i:03d}__cls{cid}.jpg"
            out_path = class_dir / name
            try:
                crop.save(out_path, quality=95)
            except Exception as e:
                stats.errors += 1
                stats.error_notes.append(f"Failed to save crop {out_path}: {type(e).__name__}: {e}")
                continue

            stats.total_crops += 1
            stats.per_class[cid] += 1

            # padded
            if pad_frac > 0:
                px1, py1, px2, py2 = _yolo_to_xyxy(xc, yc, w, h, img_w, img_h, pad_frac=pad_frac)
                if px2 > px1 and py2 > py1:
                    pcrop = img.crop((px1, py1, px2, py2))
                    padded_dir = class_dir / "padded"
                    _safe_mkdir(padded_dir)
                    pout_path = padded_dir / name
                    try:
                        pcrop.save(pout_path, quality=95)
                        stats.total_crops_padded += 1
                        stats.per_class_padded[cid] += 1
                    except Exception as e:
                        stats.errors += 1
                        stats.error_notes.append(f"Failed to save padded crop {pout_path}: {type(e).__name__}: {e}")

        img.close()

    # Build preview folders: first N lexicographically (stable)
    for class_dir in sorted(p for p in out_dir.glob("class_*") if p.is_dir()):
        preview_dir = class_dir / "preview"
        _safe_mkdir(preview_dir)
        crops = sorted(p for p in class_dir.glob("*.jpg") if p.is_file())
        for p in crops[:preview_count]:
            dst = preview_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

        padded_dir = class_dir / "padded"
        if padded_dir.exists():
            pprev = padded_dir / "preview"
            _safe_mkdir(pprev)
            pcrops = sorted(p for p in padded_dir.glob("*.jpg") if p.is_file())
            for p in pcrops[:preview_count]:
                dst = pprev / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)

    return stats


def write_report(report_path: Path, stats: CropStats, pad_frac: float, tiny_px: int, preview_count: int) -> None:
    lines: List[str] = []
    lines.append("BBOX CROP EXPORT REPORT")
    lines.append("")
    lines.append(f"pad_frac: {pad_frac}")
    lines.append(f"tiny_px_threshold: {tiny_px}")
    lines.append(f"preview_count_per_class: {preview_count}")
    lines.append("")
    lines.append(f"total_crops: {stats.total_crops}")
    lines.append(f"total_crops_padded: {stats.total_crops_padded}")
    lines.append(f"tiny_bboxes: {stats.tiny_bboxes}")
    lines.append(f"errors: {stats.errors}")
    lines.append("")
    lines.append("per_class:")
    for cid in sorted(stats.per_class.keys()):
        lines.append(f"  class {cid}: {stats.per_class[cid]} crops ({stats.per_class_padded.get(cid, 0)} padded)")
    if stats.error_notes:
        lines.append("")
        lines.append("error_notes (first 200):")
        for n in stats.error_notes[:200]:
            lines.append(f"  - {n}")
        if len(stats.error_notes) > 200:
            lines.append(f"  - ... truncated ({len(stats.error_notes) - 200} more)")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export bbox crops grouped by class_id for strawberry_ds.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--pad-frac", type=float, default=0.15, help="Padding fraction around bbox (e.g. 0.15 = 15%%).")
    ap.add_argument("--tiny-px", type=int, default=8, help="Count bbox as tiny if width or height < this many pixels.")
    ap.add_argument("--preview-count", type=int, default=50)
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    norm_root = repo_root / "data" / "normalized" / "strawberry_ds"
    images_dir = (args.images_dir or (norm_root / "images")).resolve()
    labels_dir = (args.labels_dir or (norm_root / "labels")).resolve()
    out_dir = (args.out_dir or (repo_root / "data" / "debug_box_crops" / "strawberry_ds")).resolve()

    _safe_mkdir(out_dir)
    stats = export_crops_for_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        out_dir=out_dir,
        pad_frac=args.pad_frac,
        tiny_px=args.tiny_px,
        preview_count=args.preview_count,
    )

    report_path = out_dir / "crop_report.txt"
    write_report(report_path, stats, pad_frac=args.pad_frac, tiny_px=args.tiny_px, preview_count=args.preview_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

