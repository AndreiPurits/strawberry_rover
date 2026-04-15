#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_DIR = {
    0: "green",
    1: "turning",
    2: "ripe",
    3: "rotten_or_overripe",
}

CLS_RE = re.compile(r"__cls(\d+)\b", re.IGNORECASE)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images_recursive(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _extract_cls_id(filename: str) -> Optional[int]:
    m = CLS_RE.search(filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _is_corrupt_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return False
    except Exception:
        return True


@dataclass
class SplitStats:
    total_found_images: int = 0
    total_processed: int = 0
    copied_by_class: Counter[str] = field(default_factory=Counter)
    unknown_cls_in_name: int = 0
    unknown_cls_examples: List[str] = field(default_factory=list)
    corrupt_images: int = 0
    corrupt_examples: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def split_crops(
    src_root: Path,
    dst_root: Path,
    preview_count: int,
) -> SplitStats:
    stats = SplitStats()
    _safe_mkdir(dst_root)

    # Create class dirs + preview dirs up front
    for cls_name in CLASS_DIR.values():
        _safe_mkdir(dst_root / cls_name / "preview")

    files = sorted(_iter_images_recursive(src_root))
    stats.total_found_images = len(files)

    # Copy into class dirs
    for p in files:
        # Avoid re-processing files already in by_class to prevent duplication loops
        if dst_root in p.parents:
            continue

        stats.total_processed += 1

        cls_id = _extract_cls_id(p.name)
        if cls_id is None:
            stats.unknown_cls_in_name += 1
            if len(stats.unknown_cls_examples) < 50:
                stats.unknown_cls_examples.append(str(p.relative_to(src_root)))
            continue

        cls_name = CLASS_DIR.get(cls_id)
        if cls_name is None:
            stats.unknown_cls_in_name += 1
            if len(stats.unknown_cls_examples) < 50:
                stats.unknown_cls_examples.append(str(p.relative_to(src_root)))
            continue

        if _is_corrupt_image(p):
            stats.corrupt_images += 1
            if len(stats.corrupt_examples) < 50:
                stats.corrupt_examples.append(str(p.relative_to(src_root)))
            continue

        out_dir = dst_root / cls_name
        _safe_mkdir(out_dir)

        dst = out_dir / p.name
        if dst.exists():
            # Don't overwrite; keep first copy and warn
            stats.warnings.append(f"duplicate_filename_skipped: {p} -> {dst}")
            continue

        shutil.copy2(p, dst)
        stats.copied_by_class[cls_name] += 1

    # Build preview sets (first N sorted by filename)
    for cls_id, cls_name in CLASS_DIR.items():
        class_dir = dst_root / cls_name
        preview_dir = class_dir / "preview"
        _safe_mkdir(preview_dir)

        imgs = sorted(x for x in class_dir.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTS)
        for x in imgs[:preview_count]:
            dst = preview_dir / x.name
            if not dst.exists():
                shutil.copy2(x, dst)

    return stats


def write_report(report_path: Path, stats: SplitStats, src_root: Path, dst_root: Path, preview_count: int) -> None:
    lines: List[str] = []
    lines.append("CROP CLASS SPLIT REPORT")
    lines.append("")
    lines.append(f"src_root: {src_root}")
    lines.append(f"dst_root: {dst_root}")
    lines.append(f"preview_count_per_class: {preview_count}")
    lines.append("")
    lines.append(f"total_found_images: {stats.total_found_images}")
    lines.append(f"total_processed (excluding by_class/*): {stats.total_processed}")
    lines.append("")
    lines.append("copied_by_class:")
    for cls_name in ("green", "turning", "ripe", "rotten_or_overripe"):
        lines.append(f"  {cls_name}: {stats.copied_by_class.get(cls_name, 0)}")
    lines.append("")
    lines.append(f"unknown_cls_in_name: {stats.unknown_cls_in_name}")
    lines.append(f"corrupt_images: {stats.corrupt_images}")

    if stats.unknown_cls_examples:
        lines.append("")
        lines.append("unknown_cls_examples (first 50):")
        for ex in stats.unknown_cls_examples[:50]:
            lines.append(f"  - {ex}")

    if stats.corrupt_examples:
        lines.append("")
        lines.append("corrupt_examples (first 50):")
        for ex in stats.corrupt_examples[:50]:
            lines.append(f"  - {ex}")

    if stats.warnings:
        lines.append("")
        lines.append("warnings (first 200):")
        for w in stats.warnings[:200]:
            lines.append(f"  - {w}")
        if len(stats.warnings) > 200:
            lines.append(f"  - ... truncated ({len(stats.warnings) - 200} more)")

    _safe_mkdir(report_path.parent)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Split crop images by class_id extracted from filename (__clsX).")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--src-root", type=Path, default=None, help="Defaults to <repo-root>/data/final_detection_dataset")
    ap.add_argument("--dst-root", type=Path, default=None, help="Defaults to <src-root>/by_class")
    ap.add_argument("--preview-count", type=int, default=100)
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    src_root = (args.src_root or (repo_root / "data" / "final_detection_dataset")).resolve()
    dst_root = (args.dst_root or (src_root / "by_class")).resolve()

    stats = split_crops(src_root=src_root, dst_root=dst_root, preview_count=args.preview_count)
    report_path = dst_root / "class_split_report.txt"
    write_report(report_path, stats, src_root=src_root, dst_root=dst_root, preview_count=args.preview_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

