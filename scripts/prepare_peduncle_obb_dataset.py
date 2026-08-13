#!/usr/bin/env python3
"""Build Ultralytics YOLO-OBB train/val split from плодоножки апрувд."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "плодоножки апрувд"
OUT_DEFAULT = REPO_ROOT / "data" / "peduncle_obb_approved_v1"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def rewrite_obb_label(src: Path, dst: Path) -> int:
    """Copy OBB lines, remap any class id → 0 (single-class peduncle)."""
    lines_out = []
    for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        toks = raw.strip().split()
        if len(toks) < 9:
            continue
        try:
            coords = [float(x) for x in toks[1:9]]
        except Exception:
            continue
        if any(c < -0.01 or c > 1.01 for c in coords):
            continue
        coords = [max(0.0, min(1.0, c)) for c in coords]
        lines_out.append("0 " + " ".join(f"{c:.6f}" for c in coords))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(("\n".join(lines_out) + ("\n" if lines_out else "")), encoding="utf-8")
    return len(lines_out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    img_dir = args.src / "images"
    lab_dir = args.src / "labels_peduncle"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        raise SystemExit(f"missing images/labels_peduncle under {args.src}")

    pairs = []
    for im in sorted(img_dir.iterdir()):
        if not im.is_file() or im.suffix.lower() not in IMG_EXTS:
            continue
        lab = lab_dir / f"{im.stem}.txt"
        if not lab.is_file() or not lab.read_text(encoding="utf-8", errors="ignore").strip():
            continue
        pairs.append((im, lab))

    if not pairs:
        raise SystemExit("no labeled images found")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * args.val_ratio)))
    if n_val >= len(pairs):
        n_val = max(1, len(pairs) // 5)
    val_set = pairs[:n_val]
    train_set = pairs[n_val:]

    out = args.out
    if out.exists() and args.force:
        shutil.rmtree(out)
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists; pass --force to rebuild")

    stats = {"train": 0, "val": 0, "train_obj": 0, "val_obj": 0}
    for split, items in (("train", train_set), ("val", val_set)):
        for im, lab in items:
            dst_im = out / "images" / split / im.name
            dst_lb = out / "labels" / split / f"{im.stem}.txt"
            dst_im.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(im, dst_im)
            nobj = rewrite_obb_label(lab, dst_lb)
            stats[split] += 1
            stats[f"{split}_obj"] += nobj

    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: peduncle",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out}")
    print(
        f"train={stats['train']} images / {stats['train_obj']} boxes · "
        f"val={stats['val']} images / {stats['val_obj']} boxes"
    )
    print(f"data.yaml → {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
