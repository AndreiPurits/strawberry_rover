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

CLASS_NAMES = {
    0: "green",
    1: "turning",
    2: "ripe",
    3: "rotten_or_overripe",
}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(dirpath: Path) -> Iterable[Path]:
    if not dirpath.exists():
        return []
    return (p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)


def _clamp01(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _parse_yolo_bbox_file(label_path: Path) -> Tuple[List[Tuple[int, float, float, float, float]], List[int]]:
    """Return (items, unknown_class_ids_in_file). Skips malformed lines."""
    items: List[Tuple[int, float, float, float, float]] = []
    unknown: List[int] = []
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
        if cid not in CLASS_NAMES:
            unknown.append(cid)
        items.append((cid, xc, yc, w, h))
    return items, unknown


def _items_to_text(items: List[Tuple[int, float, float, float, float]]) -> str:
    lines = [f"{cid} {xc:.10f} {yc:.10f} {w:.10f} {h:.10f}" for cid, xc, yc, w, h in items]
    return "\n".join(lines) + ("\n" if lines else "")


def _draw_bbox_vis(image_path: Path, items: List[Tuple[int, float, float, float, float]], out_path: Path) -> None:
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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w_img, h_img) * 0.003)))
        if font is not None:
            draw.text((x1 + 2, max(0, y1 - 10)), f"{cid}:{CLASS_NAMES.get(cid,'?')}", fill=color, font=font)

    _safe_mkdir(out_path.parent)
    img.save(out_path)


@dataclass
class MergeStats:
    total_images: int = 0
    images_by_source: Counter[str] = field(default_factory=Counter)
    renamed_due_to_conflict: int = 0
    rename_examples: List[str] = field(default_factory=list)

    total_labels: int = 0
    pairs_image_label: int = 0
    images_without_labels: int = 0
    labels_without_images: int = 0

    total_bboxes: int = 0
    objects_by_class: Counter[int] = field(default_factory=Counter)
    images_with_class: Counter[int] = field(default_factory=Counter)
    unknown_class_ids: Counter[int] = field(default_factory=Counter)

    bbox_vis_created: int = 0
    warnings: List[str] = field(default_factory=list)


def merge_sources(
    sources: Sequence[Tuple[str, Path]],
    out_base: Path,
) -> MergeStats:
    stats = MergeStats()

    out_images = out_base / "images"
    out_labels = out_base / "labels"
    out_vis = out_base / "bbox_vis"
    _safe_mkdir(out_images)
    _safe_mkdir(out_labels)
    _safe_mkdir(out_vis)

    used_names: set[str] = set()
    out_img_by_stem: Dict[str, Path] = {}
    out_lbl_by_stem: Dict[str, Path] = {}

    # Copy images + labels with renaming on conflict
    for source_name, source_dir in sources:
        src_images = source_dir / "images"
        src_labels = source_dir / "labels"

        for img in sorted(_iter_images(src_images)):
            dst_name = img.name
            if dst_name in used_names:
                dst_name = f"{source_name}__{img.name}"
                stats.renamed_due_to_conflict += 1
                if len(stats.rename_examples) < 50:
                    stats.rename_examples.append(f"{img.name} -> {dst_name}")
            used_names.add(dst_name)

            dst_img = out_images / dst_name
            shutil.copy2(img, dst_img)
            stats.total_images += 1
            stats.images_by_source[source_name] += 1

            stem = Path(dst_name).stem
            out_img_by_stem[stem] = dst_img

            # label copy (match by original stem, but must follow dst image stem)
            src_lbl = src_labels / f"{img.stem}.txt"
            dst_lbl = out_labels / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
                stats.total_labels += 1
                out_lbl_by_stem[stem] = dst_lbl
            else:
                stats.images_without_labels += 1

    # Now compute label/image pairing sanity
    img_stems = set(out_img_by_stem.keys())
    lbl_stems = set(out_lbl_by_stem.keys())
    stats.pairs_image_label = len(img_stems & lbl_stems)
    stats.labels_without_images = len(lbl_stems - img_stems)
    if stats.labels_without_images:
        for s in sorted(lbl_stems - img_stems)[:50]:
            stats.warnings.append(f"label_without_image: {s}.txt")

    # Compute object stats + images-with-class and generate bbox_vis
    for stem in sorted(img_stems):
        lbl = out_lbl_by_stem.get(stem)
        if lbl is None:
            continue
        try:
            items, unknown = _parse_yolo_bbox_file(lbl)
        except Exception as e:
            stats.warnings.append(f"failed_to_parse_label: {lbl.name}: {type(e).__name__}: {e}")
            continue

        for cid in unknown:
            stats.unknown_class_ids[cid] += 1

        present: set[int] = set()
        for cid, xc, yc, w, h in items:
            stats.total_bboxes += 1
            stats.objects_by_class[cid] += 1
            present.add(cid)

        for cid in present:
            stats.images_with_class[cid] += 1

        # bbox_vis for all (including empty labels -> just skip)
        if items:
            try:
                img_path = out_img_by_stem[stem]
                _draw_bbox_vis(img_path, items, out_vis / f"{stem}.jpg")
                stats.bbox_vis_created += 1
            except Exception as e:
                stats.warnings.append(f"failed_to_draw_bbox_vis: {stem}: {type(e).__name__}: {e}")

    return stats


def write_outputs(repo_root: Path, out_base: Path, stats: MergeStats) -> None:
    summary_txt = out_base / "dataset_summary.txt"
    summary_md = repo_root / "docs" / "final_detection_dataset_summary.md"

    lines: List[str] = []
    lines.append("FINAL DETECTION DATASET SUMMARY")
    lines.append("")
    lines.append("Class schema:")
    for cid in sorted(CLASS_NAMES.keys()):
        lines.append(f"  {cid} = {CLASS_NAMES[cid]}")
    lines.append("")

    lines.append("A. Image stats")
    lines.append(f"  total_images: {stats.total_images}")
    for src, cnt in stats.images_by_source.items():
        lines.append(f"  from_{src}: {cnt}")
    lines.append(f"  renamed_due_to_conflict: {stats.renamed_due_to_conflict}")
    if stats.rename_examples:
        lines.append("  rename_examples (first 50):")
        for ex in stats.rename_examples[:50]:
            lines.append(f"    - {ex}")
    lines.append("")

    lines.append("B. Label stats")
    lines.append(f"  total_label_files: {stats.total_labels}")
    lines.append(f"  image+label_pairs: {stats.pairs_image_label}")
    lines.append(f"  images_without_labels: {stats.images_without_labels}")
    lines.append(f"  labels_without_images: {stats.labels_without_images}")
    lines.append("")

    lines.append("C. Object stats")
    lines.append(f"  total_bboxes: {stats.total_bboxes}")
    for cid in sorted(CLASS_NAMES.keys()):
        lines.append(f"  class_{cid} ({CLASS_NAMES[cid]}): {stats.objects_by_class.get(cid, 0)}")
    lines.append("")

    lines.append("D. Image-with-class stats")
    for cid in sorted(CLASS_NAMES.keys()):
        lines.append(f"  images_with_class_{cid} ({CLASS_NAMES[cid]}): {stats.images_with_class.get(cid, 0)}")
    lines.append("")

    lines.append(f"bbox_vis_created: {stats.bbox_vis_created}")
    if stats.unknown_class_ids:
        lines.append("")
        lines.append("WARNING: unknown class_id found in labels:")
        for cid, cnt in stats.unknown_class_ids.most_common():
            lines.append(f"  class {cid}: {cnt} objects")
    if stats.warnings:
        lines.append("")
        lines.append("warnings (first 200):")
        for w in stats.warnings[:200]:
            lines.append(f"  - {w}")
        if len(stats.warnings) > 200:
            lines.append(f"  - ... truncated ({len(stats.warnings) - 200} more)")

    summary_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    md: List[str] = []
    md.append("## Final detection dataset summary")
    md.append("")
    md.append(f"- **Path**: `{out_base}`")
    md.append("- **Class schema**:")
    for cid in sorted(CLASS_NAMES.keys()):
        md.append(f"  - `{cid}` = **{CLASS_NAMES[cid]}**")
    md.append("")
    md.append("### Totals")
    md.append(f"- **Images**: {stats.total_images} (ds: {stats.images_by_source.get('strawberry_ds', 0)}, turkey: {stats.images_by_source.get('strawberry_turkey', 0)})")
    md.append(f"- **Labels**: {stats.total_labels}")
    md.append(f"- **Image+label pairs**: {stats.pairs_image_label}")
    md.append(f"- **Bounding boxes**: {stats.total_bboxes}")
    md.append(f"- **bbox_vis created**: {stats.bbox_vis_created}")
    md.append("")
    md.append("### Objects by class")
    for cid in sorted(CLASS_NAMES.keys()):
        md.append(f"- **{cid} ({CLASS_NAMES[cid]})**: {stats.objects_by_class.get(cid, 0)}")
    md.append("")
    md.append("### Images containing class")
    for cid in sorted(CLASS_NAMES.keys()):
        md.append(f"- **{cid} ({CLASS_NAMES[cid]})**: {stats.images_with_class.get(cid, 0)}")
    md.append("")
    md.append("### Merge issues")
    md.append(f"- **Renamed due to filename conflicts**: {stats.renamed_due_to_conflict}")
    md.append(f"- **Images without labels**: {stats.images_without_labels}")
    md.append(f"- **Labels without images**: {stats.labels_without_images}")
    if stats.unknown_class_ids:
        md.append("- **Unknown class_id**: found (see `dataset_summary.txt`)")
    else:
        md.append("- **Unknown class_id**: none")

    _safe_mkdir(summary_md.parent)
    summary_md.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Merge unified strawberry datasets into a single final YOLO dataset.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    out_base = (args.out_dir or (repo_root / "data" / "final_detection_dataset")).resolve()

    sources = [
        ("strawberry_ds", repo_root / "data" / "unified" / "strawberry_ds"),
        ("strawberry_turkey", repo_root / "data" / "unified" / "strawberry_turkey"),
    ]

    _safe_mkdir(out_base)
    stats = merge_sources(sources, out_base=out_base)
    write_outputs(repo_root, out_base, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

