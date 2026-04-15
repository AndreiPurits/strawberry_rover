#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["green", "turning", "ripe", "rotten"]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _list_class_images(src_root: Path, cls: str) -> List[Path]:
    d = src_root / cls
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if _is_image(p)], key=lambda p: p.name)


def _split_indices(n: int, *, seed: int, train_frac: float, val_frac: float, test_frac: float) -> Tuple[List[int], List[int], List[int]]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    idx = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(idx)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build classification_dataset_v2 from UI-confirmed crops with stratified split.")
    ap.add_argument("--src-root", type=Path, default=REPO_ROOT / "data" / "unified_dataset_bundle" / "ui_ok_images_by_class")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "data" / "classification_dataset_v2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args(list(argv) if argv is not None else None)

    t0 = time.time()
    src_root: Path = args.src_root
    out_root: Path = args.out_root

    reports = out_root / "reports"
    _safe_mkdir(reports)

    # Create output folders
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            _safe_mkdir(out_root / split / cls)

    manifest_path = reports / "split_manifest.csv"
    split_report_path = reports / "split_report.txt"

    src_counts: Dict[str, int] = {}
    split_counts: Dict[str, Dict[str, int]] = {sp: {c: 0 for c in CLASSES} for sp in ("train", "val", "test")}

    # Build split + copy
    rows: List[Dict[str, str]] = []
    for cls in CLASSES:
        imgs = _list_class_images(src_root, cls)
        src_counts[cls] = len(imgs)
        tr_idx, va_idx, te_idx = _split_indices(
            len(imgs),
            seed=int(args.seed),
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
            test_frac=float(args.test_frac),
        )
        by_split = {"train": tr_idx, "val": va_idx, "test": te_idx}
        for split, indices in by_split.items():
            for i in indices:
                p = imgs[i]
                dst = out_root / split / cls / p.name
                if not dst.exists():
                    shutil.copy2(p, dst)
                split_counts[split][cls] += 1
                rows.append({"split": split, "class": cls, "filename": p.name, "src_path": str(p)})

    # Write manifest
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "class", "filename", "src_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write report
    lines: List[str] = []
    lines.append("CLASSIFICATION DATASET V2 SPLIT REPORT")
    lines.append(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    lines.append("")
    lines.append(f"src_root: {src_root.resolve()}")
    lines.append(f"out_root: {out_root.resolve()}")
    lines.append(f"seed: {int(args.seed)}")
    lines.append(f"fractions: train={float(args.train_frac):.2f} val={float(args.val_frac):.2f} test={float(args.test_frac):.2f}")
    lines.append("")
    lines.append("Source counts:")
    for cls in CLASSES:
        lines.append(f"- {cls}: {src_counts.get(cls, 0)}")
    lines.append("")
    lines.append("Split counts (per class):")
    for split in ("train", "val", "test"):
        lines.append(f"{split}:")
        for cls in CLASSES:
            lines.append(f"  - {cls}: {split_counts[split][cls]}")
    lines.append("")
    tot_src = sum(src_counts.values())
    tot_train = sum(split_counts["train"].values())
    tot_val = sum(split_counts["val"].values())
    tot_test = sum(split_counts["test"].values())
    lines.append("Totals:")
    lines.append(f"- source total: {tot_src}")
    lines.append(f"- train total: {tot_train}")
    lines.append(f"- val total: {tot_val}")
    lines.append(f"- test total: {tot_test}")
    lines.append("")
    lines.append(f"Wrote: {manifest_path}")
    lines.append(f"Elapsed seconds: {time.time()-t0:.1f}")
    split_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {split_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

