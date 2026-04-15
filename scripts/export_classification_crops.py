#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

CLASS_NAMES = {
    0: "green",
    1: "turning",
    2: "ripe",
    3: "rotten_or_overripe",
}


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
    pad_frac: float,
) -> Tuple[int, int, int, int]:
    if any(math.isnan(v) or math.isinf(v) for v in (xc, yc, w, h)):
        return (0, 0, 0, 0)

    # Clamp normalized coords defensively
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
            continue
        try:
            cid = int(float(toks[0]))
            xc, yc, w, h = map(float, toks[1:])
        except Exception:
            continue
        out.append((cid, xc, yc, w, h))
    return out


@dataclass
class SizeAgg:
    count: int = 0
    sum_w: int = 0
    sum_h: int = 0
    min_w: int = 10**9
    min_h: int = 10**9
    max_w: int = 0
    max_h: int = 0

    def add(self, w: int, h: int) -> None:
        self.count += 1
        self.sum_w += w
        self.sum_h += h
        self.min_w = min(self.min_w, w)
        self.min_h = min(self.min_h, h)
        self.max_w = max(self.max_w, w)
        self.max_h = max(self.max_h, h)

    def mean_w(self) -> float:
        return (self.sum_w / self.count) if self.count else 0.0

    def mean_h(self) -> float:
        return (self.sum_h / self.count) if self.count else 0.0


@dataclass
class ExportStats:
    total_crops: int = 0
    crops_by_class: Counter[int] = field(default_factory=Counter)
    tiny_bboxes: int = 0
    crop_errors: int = 0
    images_with_any_crop: int = 0
    missing_label_for_image: int = 0
    missing_image_for_label: int = 0
    unknown_class_ids: Counter[int] = field(default_factory=Counter)
    size_by_class: Dict[int, SizeAgg] = field(default_factory=lambda: defaultdict(SizeAgg))
    error_notes: List[str] = field(default_factory=list)


def export_classification_crops(
    det_images_dir: Path,
    det_labels_dir: Path,
    out_base: Path,
    pad_frac: float,
    tiny_px: int,
    preview_count: int,
) -> ExportStats:
    stats = ExportStats()

    out_all = out_base / "all"
    out_by_class = out_base / "by_class"
    out_preview = out_base / "preview"
    out_reports = out_base / "reports"

    _safe_mkdir(out_all)
    _safe_mkdir(out_by_class)
    _safe_mkdir(out_preview)
    _safe_mkdir(out_reports)

    # Create class dirs
    for cid, name in CLASS_NAMES.items():
        _safe_mkdir(out_by_class / name / "preview")
        _safe_mkdir(out_preview / name)

    img_by_stem: Dict[str, Path] = {p.stem: p for p in det_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
    lbl_by_stem: Dict[str, Path] = {p.stem: p for p in det_labels_dir.glob("*.txt") if p.is_file()}

    # missing label for image
    stats.missing_label_for_image = len(set(img_by_stem.keys()) - set(lbl_by_stem.keys()))
    # missing image for label
    stats.missing_image_for_label = len(set(lbl_by_stem.keys()) - set(img_by_stem.keys()))

    # Export crops
    for stem, img_path in sorted(img_by_stem.items()):
        lbl_path = lbl_by_stem.get(stem)
        if lbl_path is None:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            stats.crop_errors += 1
            stats.error_notes.append(f"failed_to_open_image: {img_path.name}: {type(e).__name__}: {e}")
            continue

        img_w, img_h = img.size
        try:
            items = _parse_yolo_bbox_lines(lbl_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            stats.crop_errors += 1
            stats.error_notes.append(f"failed_to_read_label: {lbl_path.name}: {type(e).__name__}: {e}")
            img.close()
            continue

        any_crop = False
        for obj_i, (cid, xc, yc, w, h) in enumerate(items):
            if cid not in CLASS_NAMES:
                stats.unknown_class_ids[cid] += 1
                # still export the crop into a numeric folder? spec doesn't ask; skip but warn
                continue

            x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, w, h, img_w, img_h, pad_frac=pad_frac)
            if x2 <= x1 or y2 <= y1:
                stats.crop_errors += 1
                stats.error_notes.append(f"invalid_bbox_empty: {stem} obj{obj_i:03d} cls{cid}")
                continue

            bw = x2 - x1
            bh = y2 - y1
            if bw < tiny_px or bh < tiny_px:
                stats.tiny_bboxes += 1

            crop = img.crop((x1, y1, x2, y2))
            out_name = f"{stem}__obj{obj_i:03d}__cls{cid}.jpg"

            # Save to all/
            out_path = out_all / out_name
            try:
                crop.save(out_path, quality=95)
            except Exception as e:
                stats.crop_errors += 1
                stats.error_notes.append(f"failed_to_save_crop: {out_path}: {type(e).__name__}: {e}")
                continue

            # Copy to by_class/
            cls_name = CLASS_NAMES[cid]
            dst = out_by_class / cls_name / out_name
            if not dst.exists():
                shutil.copy2(out_path, dst)

            stats.total_crops += 1
            stats.crops_by_class[cid] += 1
            stats.size_by_class[cid].add(bw, bh)
            any_crop = True

        if any_crop:
            stats.images_with_any_crop += 1

        img.close()

    # Build previews: first N by filename inside each class dir
    for cid, cls_name in CLASS_NAMES.items():
        class_dir = out_by_class / cls_name
        preview_dir = class_dir / "preview"
        _safe_mkdir(preview_dir)

        crops = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        for p in crops[:preview_count]:
            dst = preview_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

        # Also mirror previews to out_base/preview/<class>/
        top_preview_dir = out_preview / cls_name
        _safe_mkdir(top_preview_dir)
        for p in crops[:preview_count]:
            dst = top_preview_dir / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

    return stats


def write_summary(report_path: Path, stats: ExportStats) -> None:
    lines: List[str] = []
    lines.append("FINAL CLASSIFICATION CROP DATASET SUMMARY")
    lines.append("")
    lines.append("Class schema:")
    for cid in sorted(CLASS_NAMES.keys()):
        lines.append(f"  {cid} = {CLASS_NAMES[cid]}")
    lines.append("")
    lines.append(f"total_crops: {stats.total_crops}")
    lines.append(f"images_with_any_crop: {stats.images_with_any_crop}")
    lines.append(f"tiny_bboxes (<tiny_px threshold): {stats.tiny_bboxes}")
    lines.append(f"crop_errors: {stats.crop_errors}")
    lines.append("")
    lines.append(f"missing_label_for_image: {stats.missing_label_for_image}")
    lines.append(f"missing_image_for_label: {stats.missing_image_for_label}")
    lines.append("")
    lines.append("crops_by_class:")
    for cid in sorted(CLASS_NAMES.keys()):
        lines.append(f"  class {cid} ({CLASS_NAMES[cid]}): {stats.crops_by_class.get(cid, 0)}")
    lines.append("")
    lines.append("crop_size_by_class (px, padded crop):")
    for cid in sorted(CLASS_NAMES.keys()):
        agg = stats.size_by_class.get(cid)
        if not agg or agg.count == 0:
            lines.append(f"  class {cid} ({CLASS_NAMES[cid]}): (no crops)")
            continue
        lines.append(
            f"  class {cid} ({CLASS_NAMES[cid]}): "
            f"count={agg.count} "
            f"min={agg.min_w}x{agg.min_h} "
            f"mean={agg.mean_w():.1f}x{agg.mean_h():.1f} "
            f"max={agg.max_w}x{agg.max_h}"
        )
    if stats.unknown_class_ids:
        lines.append("")
        lines.append("unknown_class_ids (skipped crops):")
        for cid, cnt in stats.unknown_class_ids.most_common():
            lines.append(f"  class {cid}: {cnt} objects")
    if stats.error_notes:
        lines.append("")
        lines.append("error_notes (first 200):")
        for n in stats.error_notes[:200]:
            lines.append(f"  - {n}")
        if len(stats.error_notes) > 200:
            lines.append(f"  - ... truncated ({len(stats.error_notes) - 200} more)")

    _safe_mkdir(report_path.parent)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export padded bbox crops for classification from final detection dataset.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--labels-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--pad-frac", type=float, default=0.15)
    ap.add_argument("--tiny-px", type=int, default=8)
    ap.add_argument("--preview-count", type=int, default=100)
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    det_base = repo_root / "data" / "final_detection_dataset"
    images_dir = (args.images_dir or (det_base / "images")).resolve()
    labels_dir = (args.labels_dir or (det_base / "labels")).resolve()
    out_dir = (args.out_dir or (repo_root / "data" / "final_classification_dataset")).resolve()

    stats = export_classification_crops(
        det_images_dir=images_dir,
        det_labels_dir=labels_dir,
        out_base=out_dir,
        pad_frac=args.pad_frac,
        tiny_px=args.tiny_px,
        preview_count=args.preview_count,
    )

    report_path = out_dir / "reports" / "classification_dataset_summary.txt"
    write_summary(report_path, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

