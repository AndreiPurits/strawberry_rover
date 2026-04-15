#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int


def _load_coco(coco_json: Path) -> Tuple[Dict[int, CocoImage], Dict[int, List[List[float]]]]:
    """
    Returns:
      images_by_id: image_id -> CocoImage
      ann_bboxes: image_id -> list of COCO bbox [x,y,w,h] (pixels)
    """
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images_by_id: Dict[int, CocoImage] = {}
    for im in data.get("images", []):
        try:
            iid = int(im["id"])
            images_by_id[iid] = CocoImage(
                image_id=iid,
                file_name=str(im["file_name"]),
                width=int(im["width"]),
                height=int(im["height"]),
            )
        except Exception:
            continue

    ann_bboxes: Dict[int, List[List[float]]] = {}
    for ann in data.get("annotations", []):
        try:
            iid = int(ann["image_id"])
            bbox = ann.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            x, y, w, h = map(float, bbox)
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        ann_bboxes.setdefault(iid, []).append([x, y, w, h])

    return images_by_id, ann_bboxes


def _coco_bbox_to_yolo(b: List[float], *, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    xc = (x + w / 2.0) / float(img_w)
    yc = (y + h / 2.0) / float(img_h)
    wn = w / float(img_w)
    hn = h / float(img_h)
    # Clamp defensively
    xc = _clamp(xc, 0.0, 1.0)
    yc = _clamp(yc, 0.0, 1.0)
    wn = _clamp(wn, 0.0, 1.0)
    hn = _clamp(hn, 0.0, 1.0)
    return xc, yc, wn, hn


def main() -> int:
    ap = argparse.ArgumentParser(description="Build YOLO detection dataset v2 by adding N Roboflow COCO images.")
    ap.add_argument("--base-dataset", default=str(REPO_ROOT / "data" / "yolo_detection_dataset"))
    ap.add_argument("--roboflow-split-dir", default=str(REPO_ROOT / "data" / "roboflow_downloads" / "train"))
    ap.add_argument("--roboflow-coco", default="", help="Path to _annotations.coco.json (defaults to <split_dir>/_annotations.coco.json).")
    ap.add_argument("--add-count", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dataset", default=str(REPO_ROOT / "data" / "yolo_detection_dataset_v2"))
    ap.add_argument("--prefix", default="rf__")
    args = ap.parse_args()

    base = Path(args.base_dataset)
    out = Path(args.out_dataset)
    rf_dir = Path(args.roboflow_split_dir)
    coco = Path(args.roboflow_coco) if args.roboflow_coco else (rf_dir / "_annotations.coco.json")

    if not base.is_dir():
        raise SystemExit(f"Base dataset not found: {base}")
    if not rf_dir.is_dir():
        raise SystemExit(f"Roboflow split dir not found: {rf_dir}")
    if not coco.is_file():
        raise SystemExit(f"COCO annotations not found: {coco}")

    # Copy base dataset to v2 output (images+labels+yaml).
    if out.exists():
        raise SystemExit(f"Out dataset already exists, aborting: {out}")
    shutil.copytree(base, out)

    # Fix data.yaml absolute path to point at the new dataset root.
    yaml_path = out / "data.yaml"
    if yaml_path.is_file():
        txt = yaml_path.read_text(encoding="utf-8")
        txt = txt.replace(f"path: {base}", f"path: {out}")
        yaml_path.write_text(txt, encoding="utf-8")

    out_images_train = out / "images" / "train"
    out_labels_train = out / "labels" / "train"
    _safe_mkdir(out_images_train)
    _safe_mkdir(out_labels_train)

    images_by_id, ann_bboxes = _load_coco(coco)

    # Candidate images must exist on disk and have at least one bbox.
    candidates: List[CocoImage] = []
    for iid, im in images_by_id.items():
        p = rf_dir / im.file_name
        if not _is_image(p):
            continue
        if iid not in ann_bboxes:
            continue
        candidates.append(im)

    if not candidates:
        raise SystemExit("No usable COCO images found (need files + bboxes).")

    rng = random.Random(int(args.seed))
    rng.shuffle(candidates)
    selected = candidates[: int(args.add_count)]

    copied = 0
    total_boxes = 0
    skipped_exists = 0
    for im in selected:
        src_img = rf_dir / im.file_name
        stem = Path(im.file_name).stem
        new_stem = f"{args.prefix}{stem}"
        dst_img = out_images_train / f"{new_stem}{src_img.suffix.lower()}"
        dst_lbl = out_labels_train / f"{new_stem}.txt"

        if dst_img.exists() or dst_lbl.exists():
            skipped_exists += 1
            continue

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Write YOLO labels (single class: strawberry=0; map all categories to 0)
        bxs = ann_bboxes.get(im.image_id, [])
        lines: List[str] = []
        for b in bxs:
            xc, yc, wn, hn = _coco_bbox_to_yolo(b, img_w=im.width, img_h=im.height)
            # Skip degenerate bboxes after clamp
            if wn <= 0 or hn <= 0:
                continue
            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        copied += 1
        total_boxes += len(lines)

    report = out / "prep_report_v2.txt"
    report.write_text(
        "\n".join(
            [
                "build_detection_dataset_v2_from_roboflow_coco.py",
                f"base_dataset: {base}",
                f"roboflow_split_dir: {rf_dir}",
                f"roboflow_coco: {coco}",
                f"out_dataset: {out}",
                f"requested_add: {args.add_count}",
                f"seed: {args.seed}",
                f"prefix: {args.prefix}",
                f"copied_images: {copied}",
                f"skipped_exists: {skipped_exists}",
                f"total_boxes_written: {total_boxes}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Built dataset v2 at: {out}")
    print(f"Copied images: {copied} (requested {args.add_count})")
    print(f"Total boxes written: {total_boxes}")
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

