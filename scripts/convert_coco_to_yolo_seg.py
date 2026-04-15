#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _poly_to_yolo_seg(poly: List[float], w: int, h: int) -> List[float]:
    # COCO polygon: [x1,y1,x2,y2,...] absolute pixels
    out: List[float] = []
    for i in range(0, len(poly), 2):
        x = float(poly[i]) / float(w)
        y = float(poly[i + 1]) / float(h)
        # Clamp to [0,1] just in case
        x = 0.0 if x < 0 else (1.0 if x > 1 else x)
        y = 0.0 if y < 0 else (1.0 if y > 1 else y)
        out.extend([x, y])
    return out


def _ann_to_lines(ann: dict, img_w: int, img_h: int) -> List[str]:
    seg = ann.get("segmentation")
    if not isinstance(seg, list) or not seg:
        return []
    lines: List[str] = []
    for poly in seg:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        coords = _poly_to_yolo_seg([float(x) for x in poly], img_w, img_h)
        if len(coords) < 6:
            continue
        # YOLO-seg: cls x1 y1 x2 y2 ...
        lines.append("0 " + " ".join(f"{v:.6f}" for v in coords))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert COCO instance seg to Ultralytics YOLO-seg format (1-class).")
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "segmentation_project_dataset"))
    ap.add_argument("--dst", default=str(REPO_ROOT / "data" / "yolo_segmentation_dataset"))
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    ann_dir = src / "annotations"

    # Create layout
    for split in ("train", "val", "test"):
        _safe_mkdir(dst / "images" / split)
        _safe_mkdir(dst / "labels" / split)
    _safe_mkdir(dst / "reports")

    # Copy images (do not modify source)
    for split, src_split in (("train", "train"), ("val", "val"), ("test", "test")):
        src_dir = src / src_split
        dst_dir = dst / "images" / split
        for fn in os.listdir(src_dir):
            if fn.lower().endswith(".jpg"):
                shutil.copy2(str(src_dir / fn), str(dst_dir / fn))

    # Convert annotations
    mapping = {"train": "instances_train.json", "val": "instances_val.json", "test": "instances_test.json"}
    stats = {"images": 0, "anns": 0, "polys": 0, "empty_labels": 0}
    for split, ann_fn in mapping.items():
        coco = _load_json(ann_dir / ann_fn)
        images = {int(im["id"]): im for im in (coco.get("images") or [])}
        anns_by_img: Dict[int, List[dict]] = {}
        for a in coco.get("annotations") or []:
            anns_by_img.setdefault(int(a["image_id"]), []).append(a)

        for img_id, im in images.items():
            file_name = str(im.get("file_name") or "")
            w = int(im.get("width") or 0)
            h = int(im.get("height") or 0)
            if not file_name or w <= 0 or h <= 0:
                continue
            label_path = dst / "labels" / split / (Path(file_name).stem + ".txt")
            lines: List[str] = []
            for a in anns_by_img.get(img_id, []):
                new_lines = _ann_to_lines(a, w, h)
                if new_lines:
                    stats["polys"] += len(new_lines)
                lines.extend(new_lines)
                stats["anns"] += 1
            if not lines:
                stats["empty_labels"] += 1
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            stats["images"] += 1

    # data.yaml
    data_yaml = "\n".join(
        [
            f"path: {dst}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: strawberry",
            "",
        ]
    )
    (dst / "data.yaml").write_text(data_yaml, encoding="utf-8")

    rep = [
        "COCO -> YOLO-seg conversion report",
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "",
        f"src: {src}",
        f"dst: {dst}",
        "",
        f"images_written: {stats['images']}",
        f"annotations_seen: {stats['anns']}",
        f"polygons_written: {stats['polys']}",
        f"empty_label_files: {stats['empty_labels']}",
    ]
    (dst / "reports" / "conversion_report.txt").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"Wrote {dst}/data.yaml and labels. Report: {dst}/reports/conversion_report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

