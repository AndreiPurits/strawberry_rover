#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _copy_tree_files(src: Path, dst: Path, *, exts: Optional[set] = None) -> int:
    _safe_mkdir(dst)
    n = 0
    for p in sorted(src.iterdir(), key=lambda x: x.name) if src.is_dir() else []:
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        out = dst / p.name
        if not out.exists():
            shutil.copy2(p, out)
        n += 1
    return n


def _gather_v3_flat(
    v3_root: Path,
    out_images: Path,
    out_labels: Path,
) -> Tuple[int, int]:
    img_n = 0
    lbl_n = 0
    for split in ("train", "val", "test"):
        img_n += _copy_tree_files(v3_root / "images" / split, out_images, exts=IMG_EXTS)
        lbl_n += _copy_tree_files(v3_root / "labels" / split, out_labels, exts={".txt"})
    return img_n, lbl_n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a single unified directory with v3 detection data, UI-ok images, and segmentation training data.")
    ap.add_argument("--v3-root", type=Path, default=REPO_ROOT / "data" / "yolo_detection_dataset_v3")
    ap.add_argument("--ui-manual-root", type=Path, default=REPO_ROOT / "data" / "classification_manual")
    ap.add_argument("--seg-root", type=Path, default=REPO_ROOT / "data" / "segmentation_project_dataset")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "data" / "unified_dataset_bundle")
    args = ap.parse_args(list(argv) if argv is not None else None)

    t0 = time.time()
    out_root: Path = args.out_root
    _safe_mkdir(out_root)

    # 1) v3 detection flat (no train/val/test)
    v3_out = out_root / "detector_v3_training_flat"
    v3_images = v3_out / "images"
    v3_labels = v3_out / "labels"
    _safe_mkdir(v3_images)
    _safe_mkdir(v3_labels)
    img_n, lbl_n = _gather_v3_flat(Path(args.v3_root), v3_images, v3_labels)

    # 2) UI-ok images by class (4 folders)
    ui_out = out_root / "ui_ok_images_by_class"
    for cls in ("green", "turning", "ripe", "rotten"):
        _copy_tree_files(Path(args.ui_manual_root) / cls, ui_out / cls, exts=IMG_EXTS)

    # 3) segmentation training data (images + labels as-is)
    seg_out = out_root / "segmentation_training_dataset"
    _safe_mkdir(seg_out)
    # Keep split folders as they exist (train/val/test) but under images/ and labels/ plus annotations/.
    for split in ("train", "val", "test"):
        _copy_tree_files(Path(args.seg_root) / split, seg_out / "images" / split, exts=IMG_EXTS)
    ann_dir = Path(args.seg_root) / "annotations"
    if ann_dir.is_dir():
        _safe_mkdir(seg_out / "labels")
        _copy_tree_files(ann_dir, seg_out / "labels", exts={".json"})

    # Also copy reports for traceability (non-critical).
    rep = Path(args.seg_root) / "reports"
    if rep.is_dir():
        _safe_mkdir(seg_out / "reports")
        for p in rep.iterdir():
            if p.is_file() and not (seg_out / "reports" / p.name).exists():
                shutil.copy2(p, seg_out / "reports" / p.name)

    # Copy v3 reports for traceability.
    v3_rep = Path(args.v3_root) / "reports"
    if v3_rep.is_dir():
        _safe_mkdir(v3_out / "reports")
        for p in v3_rep.iterdir():
            if p.is_file() and not (v3_out / "reports" / p.name).exists():
                shutil.copy2(p, v3_out / "reports" / p.name)

    # Minimal README for humans.
    readme = out_root / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "UNIFIED DATASET BUNDLE",
                time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "",
                "1) detector_v3_training_flat/",
                "   - images/: v3 detection images (no split)",
                "   - labels/: YOLO bbox labels (single-class, .txt) matching images by stem",
                "   - reports/: copied from data/yolo_detection_dataset_v3/reports/",
                "",
                "2) ui_ok_images_by_class/",
                "   - green/ turning/ ripe/ rotten/: images labeled via UI (manual)",
                "",
                "3) segmentation_training_dataset/",
                "   - images/train|val|test/: segmentation images (as currently used)",
                "   - labels/: COCO json annotations copied from segmentation_project_dataset/annotations/",
                "   - reports/: copied subset summary if present",
                "",
                f"Sources:",
                f"- v3_root: {Path(args.v3_root).resolve()}",
                f"- ui_manual_root: {Path(args.ui_manual_root).resolve()}",
                f"- seg_root: {Path(args.seg_root).resolve()}",
                "",
                f"Counts:",
                f"- v3 flat images copied: {img_n}",
                f"- v3 flat labels copied: {lbl_n}",
                "",
                f"Elapsed seconds: {time.time()-t0:.1f}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote unified bundle: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

