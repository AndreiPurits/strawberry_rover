#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


TARGET_CLASS_NAMES = {
    0: "green",
    1: "turning",
    2: "ripe",
    3: "rotten_or_overripe",
}


MAPPING = {
    "strawberry_ds": {
        0: 1,
        4: 1,
        1: 0,
        5: 0,
        2: 2,
        3: 2,
    },
    "strawberry_turkey": {
        0: 2,
        1: 1,
        2: 0,
    },
}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(images_dir: Path) -> Iterable[Path]:
    if not images_dir.exists():
        return []
    return (p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _clamp01(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _parse_yolo_bbox(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
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
        except Exception:
            continue
        items.append((cid, xc, yc, w, h))
    return items


def _items_to_text(items: List[Tuple[int, float, float, float, float]]) -> str:
    lines = []
    for cid, xc, yc, w, h in items:
        lines.append(f"{cid} {xc:.10f} {yc:.10f} {w:.10f} {h:.10f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _draw_bboxes(
    image_path: Path,
    items: List[Tuple[int, float, float, float, float]],
    out_path: Path,
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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w_img, h_img) * 0.003)))
        label = f"{cid}:{TARGET_CLASS_NAMES.get(cid, '?')}"
        if font is not None:
            draw.text((x1 + 2, max(0, y1 - 10)), label, fill=color, font=font)

    _safe_mkdir(out_path.parent)
    img.save(out_path)


@dataclass
class RemapStats:
    dataset: str
    images_copied: int = 0
    label_files_written: int = 0
    images_missing_label: int = 0
    labels_missing_image: int = 0
    bbox_vis_created: int = 0

    unknown_class_ids: Counter[int] = field(default_factory=Counter)
    unknown_class_lines: int = 0

    new_class_objects: Counter[int] = field(default_factory=Counter)
    new_class_images: Counter[int] = field(default_factory=Counter)

    errors: List[str] = field(default_factory=list)


def remap_dataset(
    normalized_root: Path,
    out_root: Path,
    dataset: str,
    mapping: Dict[int, int],
) -> RemapStats:
    stats = RemapStats(dataset=dataset)
    src_base = normalized_root / dataset
    src_images = src_base / "images"
    src_labels = src_base / "labels"

    dst_base = out_root / dataset
    dst_images = dst_base / "images"
    dst_labels = dst_base / "labels"
    dst_vis = dst_base / "bbox_vis"
    _safe_mkdir(dst_images)
    _safe_mkdir(dst_labels)
    _safe_mkdir(dst_vis)

    # Copy images
    src_img_by_stem: Dict[str, Path] = {}
    for img in _iter_images(src_images):
        src_img_by_stem[img.stem] = img
        shutil.copy2(img, dst_images / img.name)
        stats.images_copied += 1

    # Map labels
    src_lbl_by_stem: Dict[str, Path] = {p.stem: p for p in src_labels.glob("*.txt") if p.is_file()}

    # labels without images
    stats.labels_missing_image = len(set(src_lbl_by_stem.keys()) - set(src_img_by_stem.keys()))
    if stats.labels_missing_image:
        for stem in sorted(set(src_lbl_by_stem.keys()) - set(src_img_by_stem.keys()))[:50]:
            stats.errors.append(f"label_without_image: {dataset} {stem}.txt")

    for stem, img_path in src_img_by_stem.items():
        lbl_path = src_lbl_by_stem.get(stem)
        if lbl_path is None:
            stats.images_missing_label += 1
            continue

        try:
            items = _parse_yolo_bbox(lbl_path)
        except Exception as e:
            stats.errors.append(f"failed_to_parse_label: {dataset} {lbl_path.name}: {type(e).__name__}: {e}")
            continue

        remapped: List[Tuple[int, float, float, float, float]] = []
        present_new_classes: set[int] = set()

        for old_cid, xc, yc, w, h in items:
            if old_cid not in mapping:
                stats.unknown_class_ids[old_cid] += 1
                stats.unknown_class_lines += 1
                continue  # drop unknown classes to keep unified scheme valid
            new_cid = mapping[old_cid]
            remapped.append((new_cid, xc, yc, w, h))
            stats.new_class_objects[new_cid] += 1
            present_new_classes.add(new_cid)

        for nc in present_new_classes:
            stats.new_class_images[nc] += 1

        # Write remapped label (even if empty after dropping unknowns)
        out_lbl = dst_labels / f"{stem}.txt"
        out_lbl.write_text(_items_to_text(remapped), encoding="utf-8")
        stats.label_files_written += 1

        # bbox_vis only when there are boxes
        if remapped:
            out_vis = dst_vis / f"{stem}.jpg"
            try:
                _draw_bboxes(dst_images / img_path.name, remapped, out_vis)
                stats.bbox_vis_created += 1
            except Exception as e:
                stats.errors.append(f"failed_to_draw_bbox: {dataset} {stem}: {type(e).__name__}: {e}")

    return stats


def write_report(report_path: Path, all_stats: Sequence[RemapStats]) -> None:
    lines: List[str] = []
    lines.append("UNIFIED LABEL REMAP REPORT")
    lines.append("")
    lines.append("Target class schema:")
    for cid in sorted(TARGET_CLASS_NAMES.keys()):
        lines.append(f"  {cid} = {TARGET_CLASS_NAMES[cid]}")
    lines.append("")

    for st in all_stats:
        lines.append(f"=== {st.dataset} ===")
        lines.append(f"images_copied: {st.images_copied}")
        lines.append(f"label_files_written: {st.label_files_written}")
        lines.append(f"images_missing_label: {st.images_missing_label}")
        lines.append(f"labels_missing_image: {st.labels_missing_image}")
        lines.append(f"bbox_vis_created: {st.bbox_vis_created}")
        lines.append("")
        lines.append("new_class_objects:")
        for cid in sorted(TARGET_CLASS_NAMES.keys()):
            lines.append(f"  class {cid} ({TARGET_CLASS_NAMES[cid]}): {st.new_class_objects.get(cid, 0)}")
        lines.append("")
        lines.append("new_class_images:")
        for cid in sorted(TARGET_CLASS_NAMES.keys()):
            lines.append(f"  class {cid} ({TARGET_CLASS_NAMES[cid]}): {st.new_class_images.get(cid, 0)}")
        lines.append("")
        if st.unknown_class_ids:
            lines.append("unknown_class_ids (dropped objects):")
            for cid, cnt in st.unknown_class_ids.most_common():
                lines.append(f"  old_class {cid}: {cnt} objects")
            lines.append(f"unknown_class_lines_total: {st.unknown_class_lines}")
            lines.append("")
        else:
            lines.append("unknown_class_ids: none")
            lines.append("")

        if st.errors:
            lines.append("errors (first 200):")
            for e in st.errors[:200]:
                lines.append(f"  - {e}")
            if len(st.errors) > 200:
                lines.append(f"  - ... truncated ({len(st.errors) - 200} more)")
            lines.append("")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Remap normalized dataset labels into a unified 4-class scheme.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--normalized-root", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    normalized_root = (args.normalized_root or (repo_root / "data" / "normalized")).resolve()
    out_root = (args.out_root or (repo_root / "data" / "unified")).resolve()

    _safe_mkdir(out_root)
    stats: List[RemapStats] = []
    for ds in ("strawberry_ds", "strawberry_turkey"):
        stats.append(remap_dataset(normalized_root, out_root, ds, mapping=MAPPING[ds]))

    report_path = out_root / "remap_report.txt"
    write_report(report_path, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

