#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt", ".xml", ".json"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _is_label(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in LABEL_EXTS


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rel_hash(path: Path, root: Path, length: int = 8) -> str:
    rel = str(path.relative_to(root)).encode("utf-8", errors="ignore")
    return hashlib.sha1(rel).hexdigest()[:length]


def _copy_unique(src: Path, dst_dir: Path, used_names: Dict[str, int], raw_root: Path) -> Path:
    _safe_mkdir(dst_dir)
    name = src.name
    if name not in used_names:
        used_names[name] = 1
        dst = dst_dir / name
        shutil.copy2(src, dst)
        return dst

    stem, ext = src.stem, src.suffix
    h = _rel_hash(src, raw_root)
    attempt = f"{stem}__{h}{ext}"
    if attempt not in used_names:
        used_names[attempt] = 1
        dst = dst_dir / attempt
        shutil.copy2(src, dst)
        return dst

    i = used_names[name]
    while True:
        attempt = f"{stem}__{h}__dup{i}{ext}"
        if attempt not in used_names:
            used_names[attempt] = 1
            used_names[name] = i + 1
            dst = dst_dir / attempt
            shutil.copy2(src, dst)
            return dst
        i += 1


class LabelFormat:
    NONE = "none"
    YOLO_BBOX = "yolo_bbox"
    YOLO_SEG = "yolo_seg_polygon"
    UNKNOWN_TXT = "unknown_txt"
    VOC_XML = "pascal_voc_xml"
    COCO_JSON = "coco_json"


@dataclass
class DatasetStats:
    dataset: str
    raw_root: Path
    raw_dir: Path
    out_dir: Path
    images_dir: Path
    labels_dir: Path
    bbox_vis_dir: Path

    found_images: int = 0
    found_label_files: int = 0
    label_format: str = LabelFormat.NONE

    copied_images: int = 0
    written_labels: int = 0
    paired: int = 0
    images_without_labels: int = 0
    labels_without_images: int = 0

    duplicate_image_names: int = 0
    invalid_labels: int = 0
    notes: List[str] = field(default_factory=list)


def detect_label_format(label_files: Sequence[Path]) -> str:
    if not label_files:
        return LabelFormat.NONE

    exts = {p.suffix.lower() for p in label_files}
    if ".xml" in exts:
        return LabelFormat.VOC_XML
    if ".json" in exts:
        return LabelFormat.COCO_JSON

    txts = [p for p in label_files if p.suffix.lower() == ".txt"]
    if not txts:
        return LabelFormat.NONE

    # Inspect a few non-empty lines
    for p in txts[:10]:
        try:
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                toks = line.split()
                if len(toks) == 5:
                    return LabelFormat.YOLO_BBOX
                if len(toks) > 5 and (len(toks) - 1) % 2 == 0:
                    return LabelFormat.YOLO_SEG
                return LabelFormat.UNKNOWN_TXT
        except Exception:
            continue

    return LabelFormat.UNKNOWN_TXT


def parse_yolo_bbox_lines(text: str) -> List[Tuple[int, float, float, float, float]]:
    out: List[Tuple[int, float, float, float, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            raise ValueError(f"Expected 5 tokens per line, got {len(toks)}: {line[:120]}")
        cls = int(float(toks[0]))
        x, y, w, h = (float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4]))
        out.append((cls, x, y, w, h))
    return out


def parse_yolo_seg_as_bbox_lines(text: str) -> List[Tuple[int, float, float, float, float]]:
    out: List[Tuple[int, float, float, float, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) <= 5 or (len(toks) - 1) % 2 != 0:
            raise ValueError(f"Expected polygon line: class + x1 y1 ... pairs, got {len(toks)}: {line[:120]}")
        cls = int(float(toks[0]))
        coords = [float(t) for t in toks[1:]]
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)
        out.append((cls, xc, yc, w, h))
    return out


def _clamp01(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return max(0.0, min(1.0, v))


def normalize_yolo_bbox(
    items: List[Tuple[int, float, float, float, float]],
    clamp: bool = True,
) -> List[Tuple[int, float, float, float, float]]:
    out: List[Tuple[int, float, float, float, float]] = []
    for cls, x, y, w, h in items:
        if clamp:
            x, y, w, h = _clamp01(x), _clamp01(y), _clamp01(w), _clamp01(h)
        out.append((cls, x, y, w, h))
    return out


def yolo_items_to_text(items: List[Tuple[int, float, float, float, float]]) -> str:
    lines = []
    for cls, x, y, w, h in items:
        lines.append(f"{cls} {x:.10f} {y:.10f} {w:.10f} {h:.10f}")
    return "\n".join(lines) + ("\n" if lines else "")


def draw_bboxes(
    image_path: Path,
    yolo_items: List[Tuple[int, float, float, float, float]],
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    w_img, h_img = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for cls, xc, yc, bw, bh in yolo_items:
        x1 = (xc - bw / 2.0) * w_img
        y1 = (yc - bh / 2.0) * h_img
        x2 = (xc + bw / 2.0) * w_img
        y2 = (yc + bh / 2.0) * h_img

        x1 = max(0.0, min(w_img - 1.0, x1))
        y1 = max(0.0, min(h_img - 1.0, y1))
        x2 = max(0.0, min(w_img - 1.0, x2))
        y2 = max(0.0, min(h_img - 1.0, y2))

        color = (
            int((cls * 73) % 255),
            int((cls * 151) % 255),
            int((cls * 199) % 255),
        )
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(w_img, h_img) * 0.003)))
        label = str(cls)
        if font is not None:
            tx, ty = x1 + 2, max(0, y1 - 10)
            draw.text((tx, ty), label, fill=color, font=font)

    _safe_mkdir(out_path.parent)
    img.save(out_path)


def _collect_files(root: Path, predicate) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for p in root.rglob("*"):
        if predicate(p):
            out.append(p)
    return out


def _stem_key(p: Path) -> str:
    # Use file stem as the pairing key, ignoring extension.
    return p.stem


def normalize_dataset(raw_root: Path, out_root: Path, dataset: str) -> DatasetStats:
    raw_dir = raw_root / dataset
    out_dir = out_root / dataset
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    bbox_vis_dir = out_dir / "bbox_vis"

    _safe_mkdir(images_dir)
    _safe_mkdir(labels_dir)
    _safe_mkdir(bbox_vis_dir)

    stats = DatasetStats(
        dataset=dataset,
        raw_root=raw_root,
        raw_dir=raw_dir,
        out_dir=out_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        bbox_vis_dir=bbox_vis_dir,
    )

    images = _collect_files(raw_dir, _is_image)
    label_files = _collect_files(raw_dir, _is_label)
    stats.found_images = len(images)
    stats.found_label_files = len(label_files)
    stats.label_format = detect_label_format(label_files)

    if not raw_dir.exists():
        stats.notes.append(f"Raw dir missing: {raw_dir}")
        return stats

    # Build label map for txt-based per-image labels
    label_by_stem: Dict[str, Path] = {}
    for lf in label_files:
        if lf.suffix.lower() != ".txt":
            continue
        k = _stem_key(lf)
        if k not in label_by_stem:
            label_by_stem[k] = lf
        else:
            stats.notes.append(f"Duplicate label stem '{k}': kept {label_by_stem[k].relative_to(raw_root)}; ignored {lf.relative_to(raw_root)}")

    image_used_names: Dict[str, int] = {}
    label_used_names: Dict[str, int] = {}

    image_stems_in_out: Dict[str, str] = {}
    labels_written_stems: set[str] = set()

    for img_src in sorted(images):
        img_dst = _copy_unique(img_src, images_dir, image_used_names, raw_root)
        if img_dst.name != img_src.name:
            stats.duplicate_image_names += 1
        stats.copied_images += 1

        stem = img_dst.stem
        image_stems_in_out[stem] = img_dst.name

        # Find corresponding raw label (by original stem, not destination stem)
        raw_label = label_by_stem.get(_stem_key(img_src))
        if raw_label is None:
            stats.images_without_labels += 1
            continue

        try:
            raw_text = raw_label.read_text(encoding="utf-8", errors="ignore")
            if stats.label_format == LabelFormat.YOLO_BBOX:
                items = parse_yolo_bbox_lines(raw_text)
            elif stats.label_format == LabelFormat.YOLO_SEG:
                items = parse_yolo_seg_as_bbox_lines(raw_text)
            else:
                raise ValueError(f"Unsupported/unknown txt label format: {stats.label_format}")

            items = normalize_yolo_bbox(items, clamp=True)
            label_text = yolo_items_to_text(items)

            # Write label with same base name as normalized image
            label_name = f"{img_dst.stem}.txt"
            if label_name in label_used_names:
                # Very rare, but guard anyway
                label_name = f"{img_dst.stem}__{_rel_hash(img_src, raw_root)}.txt"
            label_used_names[label_name] = 1
            label_dst = labels_dir / label_name
            label_dst.write_text(label_text, encoding="utf-8")
            labels_written_stems.add(img_dst.stem)
            stats.written_labels += 1
            stats.paired += 1

            vis_out = bbox_vis_dir / f"{img_dst.stem}.jpg"
            draw_bboxes(img_dst, items, vis_out)
        except Exception as e:
            stats.invalid_labels += 1
            stats.notes.append(
                f"Label parse/convert failed for image {img_src.relative_to(raw_root)} "
                f"label {raw_label.relative_to(raw_root)}: {type(e).__name__}: {e}"
            )
            continue

    # Labels without images (only meaningful for txt-per-image label sources)
    if stats.label_format in (LabelFormat.YOLO_BBOX, LabelFormat.YOLO_SEG, LabelFormat.UNKNOWN_TXT):
        raw_image_stems = {_stem_key(p) for p in images}
        txt_label_stems = {_stem_key(p) for p in label_files if p.suffix.lower() == ".txt"}
        stats.labels_without_images = len(txt_label_stems - raw_image_stems)

    if stats.label_format == LabelFormat.VOC_XML:
        stats.notes.append("VOC XML detected. Conversion is not implemented yet; labels were not normalized.")
    if stats.label_format == LabelFormat.COCO_JSON:
        stats.notes.append("COCO JSON detected. Conversion is not implemented yet; labels were not normalized.")
    if stats.label_format == LabelFormat.UNKNOWN_TXT:
        stats.notes.append("Unknown .txt label format detected. No labels were normalized.")

    return stats


def write_reports(out_root: Path, stats: Sequence[DatasetStats]) -> None:
    _safe_mkdir(out_root)

    report_path = out_root / "dataset_report.txt"
    lines: List[str] = []
    lines.append("DATASET NORMALIZATION REPORT")
    lines.append("")
    for s in stats:
        lines.append(f"[{s.dataset}]")
        lines.append(f"raw_dir: {s.raw_dir}")
        lines.append(f"label_format_detected: {s.label_format}")
        lines.append(f"images_found: {s.found_images}")
        lines.append(f"labels_found: {s.found_label_files}")
        lines.append(f"images_copied: {s.copied_images}")
        lines.append(f"labels_written_yolo: {s.written_labels}")
        lines.append(f"paired_image+label: {s.paired}")
        lines.append(f"images_without_labels: {s.images_without_labels}")
        lines.append(f"labels_without_images: {s.labels_without_images}")
        lines.append(f"duplicate_image_names_resolved: {s.duplicate_image_names}")
        lines.append(f"invalid_labels: {s.invalid_labels}")
        if s.notes:
            lines.append("notes:")
            for n in s.notes[:200]:
                lines.append(f"  - {n}")
            if len(s.notes) > 200:
                lines.append(f"  - ... truncated ({len(s.notes) - 200} more)")
        lines.append("")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_summary_md(repo_root: Path, out_root: Path, stats: Sequence[DatasetStats]) -> None:
    docs_dir = repo_root / "docs"
    _safe_mkdir(docs_dir)
    md_path = docs_dir / "dataset_normalization_summary.md"

    def fmt(s: DatasetStats) -> str:
        return (
            f"### `{s.dataset}`\n"
            f"- **Найдено изображений**: {s.found_images}\n"
            f"- **Найдено label-файлов**: {s.found_label_files}\n"
            f"- **Формат labels (авто-детект)**: `{s.label_format}`\n"
            f"- **Скопировано изображений в normalized**: {s.copied_images}\n"
            f"- **Сконвертировано/записано YOLO labels**: {s.written_labels}\n"
            f"- **Пар image+label**: {s.paired}\n"
            f"- **Изображений без labels**: {s.images_without_labels}\n"
            f"- **Labels без изображений**: {s.labels_without_images}\n"
            + (f"- **Ошибок парсинга labels**: {s.invalid_labels}\n" if s.invalid_labels else "")
            + (
                "- **Замечания**:\n"
                + "\n".join([f"  - {n}" for n in s.notes[:15]])
                + ("\n  - ... (см. полный отчёт)\n" if len(s.notes) > 15 else "\n")
                if s.notes
                else ""
            )
        )

    body = []
    body.append("## Dataset normalization summary\n")
    body.append(f"- **Отчёт**: `{out_root / 'dataset_report.txt'}`")
    body.append(f"- **Выходная структура**: `{out_root}`\n")
    body.append("## Что было сделано\n")
    body.append("- Все изображения собраны в `data/normalized/<dataset>/images/` (без изменения `data/raw/`).")
    body.append("- Labels (если распознаны как YOLO bbox или YOLO segmentation polygon) приведены к **YOLO-bbox** и сохранены в `labels/`.")
    body.append("- Для изображений с labels созданы визуализации bbox в `bbox_vis/`.\n")
    body.append("## По датасетам\n")
    for s in stats:
        body.append(fmt(s))
    body.append("## Что проверить вручную\n")
    body.append("- Если какой-то датасет определился как `pascal_voc_xml` или `coco_json`: сейчас конвертация не реализована, нужно решить целевой пайплайн/классы и добавить конвертер.")
    body.append("- Просмотреть несколько файлов в `bbox_vis/` на предмет корректности рамок.")
    body.append("- Проверить, что ID классов соответствуют ожидаемым названиям классов (например Roboflow `data.yaml`).\n")

    md_path.write_text("\n".join(body).rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize strawberry datasets into a single YOLO-bbox compatible layout.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--raw-root", type=Path, default=None, help="Defaults to <repo-root>/data/raw")
    parser.add_argument("--out-root", type=Path, default=None, help="Defaults to <repo-root>/data/normalized")
    parser.add_argument("--datasets", nargs="*", default=["strawberry_ds", "strawberry_turkey", "my_data"])
    args = parser.parse_args(argv)

    repo_root: Path = args.repo_root
    raw_root = args.raw_root or (repo_root / "data" / "raw")
    out_root = args.out_root or (repo_root / "data" / "normalized")

    all_stats: List[DatasetStats] = []
    for ds in args.datasets:
        all_stats.append(normalize_dataset(raw_root=raw_root, out_root=out_root, dataset=ds))

    write_reports(out_root, all_stats)
    write_summary_md(repo_root, out_root, all_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

