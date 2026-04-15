#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_images(dirpath: Path) -> List[Path]:
    return sorted([p for p in dirpath.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _parse_yolo_and_force_class0(text: str) -> Tuple[str, int, int]:
    """
    Return (new_text, kept_lines, skipped_lines).
    Keeps only well-formed YOLO bbox lines and rewrites class_id to 0.
    """
    out: List[str] = []
    kept = 0
    skipped = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            skipped += 1
            continue
        try:
            # Validate numeric tokens; class ignored
            float(toks[0])
            float(toks[1]); float(toks[2]); float(toks[3]); float(toks[4])
        except Exception:
            skipped += 1
            continue
        out.append(f"0 {toks[1]} {toks[2]} {toks[3]} {toks[4]}")
        kept += 1
    return ("\n".join(out) + ("\n" if out else "")), kept, skipped


@dataclass
class PrepStats:
    images_total: int = 0
    labels_total: int = 0
    pairs_total: int = 0

    train_images: int = 0
    val_images: int = 0
    train_labels: int = 0
    val_labels: int = 0

    total_boxes_kept: int = 0
    total_label_lines_skipped: int = 0
    empty_labels_after_rewrite: int = 0
    missing_label_for_image: int = 0
    missing_image_for_label: int = 0

    notes: List[str] = field(default_factory=list)


def prepare_dataset(
    src_images: Path,
    src_labels: Path,
    out_root: Path,
    train_ratio: float,
    seed: int,
) -> PrepStats:
    stats = PrepStats()

    out_images_train = out_root / "images" / "train"
    out_images_val = out_root / "images" / "val"
    out_labels_train = out_root / "labels" / "train"
    out_labels_val = out_root / "labels" / "val"
    for p in (out_images_train, out_images_val, out_labels_train, out_labels_val):
        _safe_mkdir(p)

    images = _iter_images(src_images)
    labels = sorted([p for p in src_labels.glob("*.txt") if p.is_file()])
    stats.images_total = len(images)
    stats.labels_total = len(labels)

    img_by_stem: Dict[str, Path] = {p.stem: p for p in images}
    lbl_by_stem: Dict[str, Path] = {p.stem: p for p in labels}

    img_stems = set(img_by_stem.keys())
    lbl_stems = set(lbl_by_stem.keys())
    stats.missing_label_for_image = len(img_stems - lbl_stems)
    stats.missing_image_for_label = len(lbl_stems - img_stems)
    if stats.missing_label_for_image:
        stats.notes.append(f"missing_label_for_image: {stats.missing_label_for_image}")
    if stats.missing_image_for_label:
        stats.notes.append(f"missing_image_for_label: {stats.missing_image_for_label}")

    paired_stems = sorted(img_stems & lbl_stems)
    stats.pairs_total = len(paired_stems)

    rnd = random.Random(seed)
    rnd.shuffle(paired_stems)
    split = int(round(len(paired_stems) * train_ratio))
    train_stems = set(paired_stems[:split])
    val_stems = set(paired_stems[split:])

    def copy_pair(stem: str, split_name: str) -> None:
        img_src = img_by_stem[stem]
        lbl_src = lbl_by_stem[stem]

        if split_name == "train":
            img_dst = out_images_train / img_src.name
            lbl_dst = out_labels_train / f"{stem}.txt"
        else:
            img_dst = out_images_val / img_src.name
            lbl_dst = out_labels_val / f"{stem}.txt"

        shutil.copy2(img_src, img_dst)

        new_text, kept, skipped = _parse_yolo_and_force_class0(
            lbl_src.read_text(encoding="utf-8", errors="ignore")
        )
        lbl_dst.write_text(new_text, encoding="utf-8")

        stats.total_boxes_kept += kept
        stats.total_label_lines_skipped += skipped
        if kept == 0:
            stats.empty_labels_after_rewrite += 1

    for stem in sorted(train_stems):
        copy_pair(stem, "train")
        stats.train_images += 1
        stats.train_labels += 1
    for stem in sorted(val_stems):
        copy_pair(stem, "val")
        stats.val_images += 1
        stats.val_labels += 1

    # Write data.yaml
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: strawberry",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Short prep report
    (out_root / "prep_report.txt").write_text(
        "\n".join(
            [
                "YOLO DETECTION DATASET PREP REPORT",
                "",
                f"src_images: {src_images}",
                f"src_labels: {src_labels}",
                f"out_root: {out_root}",
                f"train_ratio: {train_ratio}",
                f"seed: {seed}",
                "",
                f"images_total: {stats.images_total}",
                f"labels_total: {stats.labels_total}",
                f"pairs_total: {stats.pairs_total}",
                "",
                f"train_images: {stats.train_images}",
                f"val_images: {stats.val_images}",
                f"train_labels: {stats.train_labels}",
                f"val_labels: {stats.val_labels}",
                "",
                f"total_boxes_kept: {stats.total_boxes_kept}",
                f"total_label_lines_skipped: {stats.total_label_lines_skipped}",
                f"empty_labels_after_rewrite: {stats.empty_labels_after_rewrite}",
                "",
                f"missing_label_for_image: {stats.missing_label_for_image}",
                f"missing_image_for_label: {stats.missing_image_for_label}",
                *(["", "notes:"] + [f"  - {n}" for n in stats.notes] if stats.notes else []),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare a 1-class YOLO detection dataset (train/val split + class_id->0).")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--src-images", type=Path, default=None)
    ap.add_argument("--src-labels", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    src_base = repo_root / "data" / "final_detection_dataset"
    src_images = (args.src_images or (src_base / "images")).resolve()
    src_labels = (args.src_labels or (src_base / "labels")).resolve()
    out_root = (args.out_root or (repo_root / "data" / "yolo_detection_dataset")).resolve()

    prepare_dataset(src_images, src_labels, out_root, train_ratio=args.train_ratio, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

