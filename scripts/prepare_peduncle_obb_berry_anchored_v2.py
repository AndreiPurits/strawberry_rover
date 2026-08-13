#!/usr/bin/env python3
"""Build berry-anchored YOLO-OBB train/val split from плодоножки апрувд."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

from peduncle_berry_geometry import (
    OUT_H,
    OUT_W,
    berry_to_yolo_line,
    clip_poly_to_image,
    make_canon_transform,
    obb_to_yolo_line,
    parse_berry_yolo_aabb,
    parse_peduncle_obb,
    transform_berry,
    warp_bgr,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "плодоножки апрувд"
OUT_DEFAULT = REPO_ROOT / "data" / "peduncle_obb_berry_anchored_v2"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Berry-anchored peduncle OBB dataset v2")
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-w", type=int, default=OUT_W)
    ap.add_argument("--out-h", type=int, default=OUT_H)
    args = ap.parse_args()

    img_dir = args.src / "images"
    berry_dir = args.src / "labels"
    ped_dir = args.src / "labels_peduncle"
    if not img_dir.is_dir():
        raise SystemExit(f"missing {img_dir}")

    items = []
    for im in sorted(img_dir.iterdir()):
        if not im.is_file() or im.suffix.lower() not in IMG_EXTS:
            continue
        berry_lab = berry_dir / f"{im.stem}.txt"
        ped_lab = ped_dir / f"{im.stem}.txt"
        if not ped_lab.is_file() or not ped_lab.read_text(encoding="utf-8", errors="ignore").strip():
            continue
        if not berry_lab.is_file():
            continue
        items.append((im, berry_lab, ped_lab))

    if not items:
        raise SystemExit("no labeled images with berry+peduncle")

    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * args.val_ratio)))
    if n_val >= len(items):
        n_val = max(1, len(items) // 5)
    val_set = items[:n_val]
    train_set = items[n_val:]

    out = args.out
    if out.exists() and args.force:
        shutil.rmtree(out)
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists; pass --force to rebuild")

    stats = {
        "train": 0,
        "val": 0,
        "train_obj": 0,
        "val_obj": 0,
        "skip_no_berry": 0,
        "skip_no_ped": 0,
        "skip_read": 0,
    }

    for split, split_items in (("train", train_set), ("val", val_set)):
        for im_path, berry_lab, ped_lab in split_items:
            bgr = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
            if bgr is None:
                stats["skip_read"] += 1
                continue
            h, w = bgr.shape[:2]
            berry = parse_berry_yolo_aabb(berry_lab, w, h)
            if berry is None or berry.h < 8:
                stats["skip_no_berry"] += 1
                continue
            peds = parse_peduncle_obb(ped_lab, w, h)
            if not peds:
                stats["skip_no_ped"] += 1
                continue

            tfm = make_canon_transform(berry, out_w=int(args.out_w), out_h=int(args.out_h))
            canon = warp_bgr(bgr, tfm)
            berry_c = transform_berry(berry, tfm)
            ped_lines = []
            ped_pts_canon = []
            for pts in peds:
                mapped = tfm.map_pts(pts)
                clipped = clip_poly_to_image(mapped, tfm.out_w, tfm.out_h)
                if clipped is None:
                    continue
                line = obb_to_yolo_line(clipped, tfm.out_w, tfm.out_h, cls=0)
                if line:
                    ped_lines.append(line)
                    ped_pts_canon.append(clipped)
            if not ped_lines:
                stats["skip_no_ped"] += 1
                continue

            dst_im = out / "images" / split / im_path.name
            dst_lb = out / "labels" / split / f"{im_path.stem}.txt"
            dst_berry = out / "labels_berry" / split / f"{im_path.stem}.txt"
            dst_meta = out / "meta" / split / f"{im_path.stem}.json"
            dst_im.parent.mkdir(parents=True, exist_ok=True)
            dst_lb.parent.mkdir(parents=True, exist_ok=True)
            dst_berry.parent.mkdir(parents=True, exist_ok=True)
            dst_meta.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(dst_im), canon, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            dst_lb.write_text("\n".join(ped_lines) + "\n", encoding="utf-8")
            dst_berry.write_text(berry_to_yolo_line(berry_c, tfm.out_w, tfm.out_h) + "\n", encoding="utf-8")
            cx, cy = berry_c.calyx
            meta = {
                "source": im_path.name,
                "out_w": tfm.out_w,
                "out_h": tfm.out_h,
                "scale": tfm.scale,
                "tx": tfm.tx,
                "ty": tfm.ty,
                "berry_xyxy": list(berry_c.as_xyxy()),
                "calyx_xy": [cx, cy],
                "n_peduncle": len(ped_lines),
            }
            dst_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            # optional preview overlay
            vis = canon.copy()
            cv2.rectangle(
                vis,
                (int(berry_c.x1), int(berry_c.y1)),
                (int(berry_c.x2), int(berry_c.y2)),
                (0, 220, 0),
                2,
            )
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 255), -1)
            for pts in ped_pts_canon:
                poly = np_int_poly(pts)
                cv2.polylines(vis, [poly], True, (0, 140, 255), 2)
            vis_dir = out / "bbox_vis" / split
            vis_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_dir / im_path.name), vis)

            stats[split] += 1
            stats[f"{split}_obj"] += len(ped_lines)

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
        f"train={stats['train']} / {stats['train_obj']} boxes · "
        f"val={stats['val']} / {stats['val_obj']} boxes"
    )
    print(
        f"skipped: no_berry={stats['skip_no_berry']} no_ped={stats['skip_no_ped']} "
        f"read={stats['skip_read']}"
    )
    print(f"data.yaml → {yaml_path}")
    return 0


def np_int_poly(pts):
    import numpy as np

    return np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)


if __name__ == "__main__":
    raise SystemExit(main())
