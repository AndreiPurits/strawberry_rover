#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clamp_int(v: float, lo: int, hi: int, *, ceil: bool = False) -> int:
    if math.isnan(v) or math.isinf(v):
        return lo
    iv = int(math.ceil(v)) if ceil else int(math.floor(v))
    return max(lo, min(hi, iv))


def _find_label_dir(split_dir: Path) -> Optional[Path]:
    # Roboflow YOLO exports usually place labels in split/labels/, but this dataset has train/images/labels/.
    candidates = [
        split_dir / "labels",
        split_dir / "images" / "labels",
    ]
    for d in candidates:
        if d.is_dir():
            return d
    return None


def _iter_images(images_dir: Path) -> Iterable[Path]:
    if not images_dir.is_dir():
        return []
    for p in sorted(images_dir.iterdir(), key=lambda x: x.name):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _parse_obb_line(line: str) -> Optional[Tuple[int, List[Tuple[float, float]]]]:
    toks = line.strip().split()
    if len(toks) != 9:
        return None
    try:
        cid = int(float(toks[0]))
        pts = []
        for i in range(1, 9, 2):
            x = float(toks[i])
            y = float(toks[i + 1])
            pts.append((x, y))
        return cid, pts
    except Exception:
        return None


def _obb_to_xyxy_pixels(
    pts_norm: List[Tuple[float, float]],
    *,
    img_w: int,
    img_h: int,
    pad_total_frac: float,
) -> Tuple[int, int, int, int]:
    xs = [max(0.0, min(1.0, x)) for x, _y in pts_norm]
    ys = [max(0.0, min(1.0, y)) for _x, y in pts_norm]
    x1 = min(xs) * img_w
    x2 = max(xs) * img_w
    y1 = min(ys) * img_h
    y2 = max(ys) * img_h

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    # "+10%" requested: interpret as +10% total size → pad 5% each side.
    pad_each = max(0.0, float(pad_total_frac)) / 2.0
    x1p = x1 - (pad_each * bw)
    x2p = x2 + (pad_each * bw)
    y1p = y1 - (pad_each * bh)
    y2p = y2 + (pad_each * bh)

    x1i = _clamp_int(x1p, 0, img_w - 1, ceil=False)
    y1i = _clamp_int(y1p, 0, img_h - 1, ceil=False)
    x2i = _clamp_int(x2p, 0, img_w, ceil=True)
    y2i = _clamp_int(y2p, 0, img_h, ceil=True)
    return x1i, y1i, x2i, y2i


def _unique_out_path(out_dir: Path, base_name: str) -> Path:
    p = out_dir / base_name
    if not p.exists():
        return p
    stem = p.stem
    suf = p.suffix
    for k in range(2, 10_000):
        pp = out_dir / f"{stem}__dup{k}{suf}"
        if not pp.exists():
            return pp
    raise RuntimeError(f"Could not find unique name for {p}")


@dataclass
class ImportStats:
    images_seen: int = 0
    label_files_seen: int = 0
    obb_lines_seen: int = 0
    unripe_objects_seen: int = 0
    crops_written: int = 0
    skipped_existing: int = 0
    errors: int = 0
    error_notes: List[str] = field(default_factory=list)


def import_unripe_crops(
    dataset_root: Path,
    *,
    out_dir: Path,
    pad_total_frac: float,
    class_unripe: int,
    splits: Sequence[str],
    limit: Optional[int],
) -> ImportStats:
    stats = ImportStats()
    _safe_mkdir(out_dir)

    for split in splits:
        split_dir = dataset_root / split
        images_dir = split_dir / "images"
        label_dir = _find_label_dir(split_dir)
        if label_dir is None:
            stats.error_notes.append(f"missing_label_dir: {split_dir}")
            stats.errors += 1
            continue

        for img_path in _iter_images(images_dir):
            stats.images_seen += 1
            lbl_path = label_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                continue
            stats.label_files_seen += 1
            try:
                txt = lbl_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                stats.errors += 1
                stats.error_notes.append(f"read_label_failed: {lbl_path}: {type(e).__name__}: {e}")
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                stats.errors += 1
                stats.error_notes.append(f"open_image_failed: {img_path}: {type(e).__name__}: {e}")
                continue
            img_w, img_h = img.size

            wrote_any = False
            for obj_i, raw in enumerate(txt.splitlines()):
                if not raw.strip():
                    continue
                parsed = _parse_obb_line(raw)
                if parsed is None:
                    continue
                stats.obb_lines_seen += 1
                cid, pts = parsed
                if cid != class_unripe:
                    continue
                stats.unripe_objects_seen += 1
                x1, y1, x2, y2 = _obb_to_xyxy_pixels(pts, img_w=img_w, img_h=img_h, pad_total_frac=pad_total_frac)
                if x2 <= x1 or y2 <= y1:
                    stats.errors += 1
                    stats.error_notes.append(f"invalid_bbox_empty: {split}/{img_path.name} obj{obj_i:03d}")
                    continue
                crop = img.crop((x1, y1, x2, y2))
                base_name = f"obbv2__{split}__{img_path.stem}__obj{obj_i:03d}.jpg"
                out_path = out_dir / base_name
                if out_path.exists():
                    stats.skipped_existing += 1
                    continue
                out_path = _unique_out_path(out_dir, base_name)
                try:
                    crop.save(out_path, quality=95)
                    stats.crops_written += 1
                    wrote_any = True
                except Exception as e:
                    stats.errors += 1
                    stats.error_notes.append(f"save_crop_failed: {out_path}: {type(e).__name__}: {e}")

                if limit is not None and stats.crops_written >= int(limit):
                    img.close()
                    return stats

            if wrote_any and (limit is not None and stats.crops_written >= int(limit)):
                img.close()
                return stats

            img.close()

    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Import UNRIPE crops from YOLOv8-OBB dataset into classifier priority queue.")
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "V2_Strawberry Object Detection.v18iV2_for_detection.yolov8-obb",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "classifier_priority_queue" / "unripe")
    ap.add_argument("--pad-total-frac", type=float, default=0.10, help="Total bbox padding fraction (0.10 = +10% total size).")
    ap.add_argument("--class-unripe", type=int, default=1)
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report-path", type=Path, default=REPO_ROOT / "data" / "classifier_priority_queue" / "reports" / "import_unripe_from_obb_report.txt")
    args = ap.parse_args(list(argv) if argv is not None else None)

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    stats = import_unripe_crops(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        pad_total_frac=float(args.pad_total_frac),
        class_unripe=int(args.class_unripe),
        splits=splits,
        limit=args.limit,
    )

    lines = [
        "IMPORT UNRIPE PRIORITY CROPS (YOLOv8-OBB -> axis-aligned bbox)",
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "",
        f"dataset_root: {Path(args.dataset_root).resolve()}",
        f"splits: {', '.join(splits)}",
        f"class_unripe: {int(args.class_unripe)}",
        f"pad_total_frac: {float(args.pad_total_frac):.3f} (+{float(args.pad_total_frac)*100:.1f}% total bbox size; half on each side)",
        f"out_dir: {Path(args.out_dir).resolve()}",
        "",
        f"images_seen: {stats.images_seen}",
        f"label_files_seen: {stats.label_files_seen}",
        f"obb_lines_seen: {stats.obb_lines_seen}",
        f"unripe_objects_seen: {stats.unripe_objects_seen}",
        f"crops_written: {stats.crops_written}",
        f"skipped_existing: {stats.skipped_existing}",
        f"errors: {stats.errors}",
    ]
    if stats.error_notes:
        lines += ["", "error_notes (first 50):"]
        lines += [f"- {x}" for x in stats.error_notes[:50]]
        if len(stats.error_notes) > 50:
            lines.append(f"- ... truncated ({len(stats.error_notes) - 50} more)")

    report_path = Path(args.report_path)
    _safe_mkdir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {report_path}")
    print(f"Added crops: {stats.crops_written} -> {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

