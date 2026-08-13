#!/usr/bin/env python3
"""Berry-anchored OBB dataset v2.1: scale/center/calyx from berry segmentation mask."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from peduncle_berry_geometry import (  # noqa: E402
    OUT_H,
    OUT_W,
    BerryBox,
    berry_from_mask,
    berry_to_yolo_line,
    clip_poly_to_image,
    make_canon_transform,
    obb_to_yolo_line,
    parse_berry_yolo_aabb,
    parse_peduncle_obb,
    transform_berry,
    warp_bgr,
    warp_mask,
)

SRC = REPO_ROOT / "data" / "плодоножки апрувд"
OUT_DEFAULT = REPO_ROOT / "data" / "peduncle_obb_berry_anchored_v21"
SEG_DEFAULT = REPO_ROOT / "models" / "model_groups" / "02_lightened_current" / "segmenter_best.pt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seg-weights", type=Path, default=SEG_DEFAULT)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-w", type=int, default=OUT_W)
    ap.add_argument("--out-h", type=int, default=OUT_H)
    args = ap.parse_args()

    try:
        from yolo_jetson_compat import apply_torchvision_nms_patch
    except ImportError:
        from scripts.yolo_jetson_compat import apply_torchvision_nms_patch  # type: ignore

    apply_torchvision_nms_patch()
    from pipelines.strawberry_ensemble import YoloSegmenterRoi

    if not args.seg_weights.is_file():
        raise SystemExit(f"missing segmenter: {args.seg_weights}")

    img_dir = args.src / "images"
    berry_dir = args.src / "labels"
    ped_dir = args.src / "labels_peduncle"
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
        raise SystemExit("no items")

    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * args.val_ratio)))
    if n_val >= len(items):
        n_val = max(1, len(items) // 5)
    val_set, train_set = items[:n_val], items[n_val:]

    out = args.out
    if out.exists() and args.force:
        shutil.rmtree(out)
    elif out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists; pass --force")

    print(f"Loading segmenter {args.seg_weights} …", flush=True)
    segmenter = YoloSegmenterRoi(str(args.seg_weights), device=str(args.device), imgsz=384, conf=0.2)

    stats = {
        "train": 0,
        "val": 0,
        "train_obj": 0,
        "val_obj": 0,
        "mask_ok": 0,
        "mask_fail_aabb": 0,
        "skip": 0,
    }

    for split, split_items in (("train", train_set), ("val", val_set)):
        for im_path, berry_lab, ped_lab in split_items:
            bgr = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
            if bgr is None:
                stats["skip"] += 1
                continue
            h, w = bgr.shape[:2]
            aabb = parse_berry_yolo_aabb(berry_lab, w, h)
            if aabb is None or aabb.h < 8:
                stats["skip"] += 1
                continue
            peds = parse_peduncle_obb(ped_lab, w, h)
            if not peds:
                stats["skip"] += 1
                continue

            mask_raw = segmenter.infer_mask_on_crop(bgr)
            berry = None
            mask_full = None
            if mask_raw is not None:
                mask_full = _resize_mask(mask_raw, w, h)
                berry = berry_from_mask(mask_full, aabb_hint=aabb)
            if berry is None:
                berry = BerryBox(aabb.x1, aabb.y1, aabb.x2, aabb.y2, calyx_xy=aabb.calyx, source="aabb")
                stats["mask_fail_aabb"] += 1
            else:
                stats["mask_ok"] += 1

            tfm = make_canon_transform(berry, out_w=int(args.out_w), out_h=int(args.out_h))
            # Anchor slightly toward calyx: shift so calyx sits above berry center in canon frame
            # (center still berry.cx/cy via make_canon_transform; calyx tip travels with affine)
            canon = warp_bgr(bgr, tfm)
            berry_c = transform_berry(berry, tfm)
            mask_c = warp_mask(mask_full, tfm) if mask_full is not None else None

            ped_lines, ped_pts = [], []
            for pts in peds:
                clipped = clip_poly_to_image(tfm.map_pts(pts), tfm.out_w, tfm.out_h)
                if clipped is None:
                    continue
                line = obb_to_yolo_line(clipped, tfm.out_w, tfm.out_h, cls=0)
                if line:
                    ped_lines.append(line)
                    ped_pts.append(clipped)
            if not ped_lines:
                stats["skip"] += 1
                continue

            dst_im = out / "images" / split / im_path.name
            dst_lb = out / "labels" / split / f"{im_path.stem}.txt"
            dst_berry = out / "labels_berry" / split / f"{im_path.stem}.txt"
            dst_meta = out / "meta" / split / f"{im_path.stem}.json"
            dst_mask = out / "masks" / split / f"{im_path.stem}.png"
            for p in (dst_im.parent, dst_lb.parent, dst_berry.parent, dst_meta.parent, dst_mask.parent):
                p.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(dst_im), canon, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            dst_lb.write_text("\n".join(ped_lines) + "\n", encoding="utf-8")
            dst_berry.write_text(berry_to_yolo_line(berry_c, tfm.out_w, tfm.out_h) + "\n", encoding="utf-8")
            if mask_c is not None:
                cv2.imwrite(str(dst_mask), mask_c)

            cx, cy = berry_c.calyx
            meta = {
                "source": im_path.name,
                "berry_source": berry.source,
                "out_w": tfm.out_w,
                "out_h": tfm.out_h,
                "scale": tfm.scale,
                "tx": tfm.tx,
                "ty": tfm.ty,
                "berry_xyxy": list(berry_c.as_xyxy()),
                "calyx_xy": [cx, cy],
                "n_peduncle": len(ped_lines),
                "has_mask": mask_c is not None,
            }
            dst_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            vis = canon.copy()
            if mask_c is not None:
                tint = vis.copy()
                tint[mask_c > 0] = (0.55 * tint[mask_c > 0] + np.array([0, 90, 0])).astype(np.uint8)
                vis = tint
            cv2.rectangle(
                vis,
                (int(berry_c.x1), int(berry_c.y1)),
                (int(berry_c.x2), int(berry_c.y2)),
                (0, 220, 0),
                2,
            )
            cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)
            for pts in ped_pts:
                poly = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)
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
        f"train={stats['train']}/{stats['train_obj']} · val={stats['val']}/{stats['val_obj']} · "
        f"mask_ok={stats['mask_ok']} aabb_fallback={stats['mask_fail_aabb']} skip={stats['skip']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
