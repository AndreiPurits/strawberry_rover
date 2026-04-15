#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ImageStats:
    image_id: int
    file_name: str
    w: int
    h: int
    has_ripe: bool
    has_unripe: bool
    obj_total: int
    obj_ripe: int
    obj_unripe: int

    @property
    def bucket(self) -> str:
        if self.has_ripe and self.has_unripe:
            return "mixed"
        if self.has_ripe:
            return "ripe_only"
        if self.has_unripe:
            return "unripe_only"
        return "no_objects"


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, data: dict) -> None:
    _safe_mkdir(p.parent)
    p.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")


def _index_coco(coco: dict) -> Tuple[Dict[int, dict], Dict[int, List[dict]], Dict[int, str]]:
    images = {int(im["id"]): im for im in (coco.get("images") or [])}
    anns_by_img: Dict[int, List[dict]] = {}
    for a in coco.get("annotations") or []:
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)
    cats = {int(c["id"]): str(c.get("name") or "") for c in (coco.get("categories") or [])}
    return images, anns_by_img, cats


def _image_stats(coco: dict) -> List[ImageStats]:
    images, anns_by_img, cats = _index_coco(coco)

    out: List[ImageStats] = []
    for img_id, im in images.items():
        anns = anns_by_img.get(img_id, [])
        has_ripe = False
        has_unripe = False
        obj_ripe = 0
        obj_unripe = 0
        for a in anns:
            name = cats.get(int(a["category_id"]), "")
            if name == "ripe":
                has_ripe = True
                obj_ripe += 1
            elif name == "unripe":
                has_unripe = True
                obj_unripe += 1
            elif name == "Strawberries":
                # Some exports include a parent category; we ignore it for balancing.
                pass
        out.append(
            ImageStats(
                image_id=img_id,
                file_name=str(im.get("file_name") or ""),
                w=int(im.get("width") or 0),
                h=int(im.get("height") or 0),
                has_ripe=has_ripe,
                has_unripe=has_unripe,
                obj_total=len(anns),
                obj_ripe=obj_ripe,
                obj_unripe=obj_unripe,
            )
        )
    return out


def _stratified_sample(stats: List[ImageStats], k: int, *, seed: int) -> List[ImageStats]:
    rng = random.Random(seed)
    buckets: Dict[str, List[ImageStats]] = {"mixed": [], "ripe_only": [], "unripe_only": [], "no_objects": []}
    for s in stats:
        buckets[s.bucket].append(s)

    # Keep "no_objects" very small; focus on berry masks.
    max_noobj = min(20, k // 50 + 1)
    target: Dict[str, int] = {
        "mixed": int(k * 0.25),
        "ripe_only": int(k * 0.375),
        "unripe_only": int(k * 0.375),
        "no_objects": 0,
    }
    target["no_objects"] = min(max_noobj, len(buckets["no_objects"]))

    # Adjust to exact k after availability.
    picked: List[ImageStats] = []
    for b in ("mixed", "ripe_only", "unripe_only", "no_objects"):
        pool = list(buckets[b])
        rng.shuffle(pool)
        take = min(target[b], len(pool))
        picked.extend(pool[:take])

    remaining = [s for s in stats if s.image_id not in {p.image_id for p in picked}]
    # Prefer images with more objects for "harder" scenes.
    remaining.sort(key=lambda s: (s.obj_total, rng.random()), reverse=True)
    if len(picked) < k:
        picked.extend(remaining[: (k - len(picked))])
    if len(picked) > k:
        picked = picked[:k]
    return picked


def _unify_to_one_class(
    coco: dict,
    selected_image_ids: set,
    *,
    new_category_id: int = 0,
    new_category_name: str = "strawberry",
) -> Tuple[dict, Dict[str, int]]:
    images, anns_by_img, cats = _index_coco(coco)

    out_images = []
    out_anns = []
    orig_counts: Dict[str, int] = {}

    ann_id = 0
    for img_id in sorted(selected_image_ids):
        im = images.get(img_id)
        if not im:
            continue
        out_images.append(
            {
                "id": len(out_images),
                "file_name": im.get("file_name"),
                "width": im.get("width"),
                "height": im.get("height"),
            }
        )
        new_img_id = out_images[-1]["id"]
        for a in anns_by_img.get(img_id, []):
            name = cats.get(int(a["category_id"]), "")
            if name not in ("Strawberries", "ripe", "unripe"):
                continue
            orig_counts[name] = orig_counts.get(name, 0) + 1
            out_anns.append(
                {
                    "id": ann_id,
                    "image_id": new_img_id,
                    "category_id": new_category_id,
                    "segmentation": a.get("segmentation"),
                    "bbox": a.get("bbox"),
                    "area": a.get("area"),
                    "iscrowd": int(a.get("iscrowd") or 0),
                }
            )
            ann_id += 1

    out = {
        "info": {
            "description": "strawberry_rover segmentation subset (1-class strawberry)",
            "version": "1.0",
            "date_created": time.strftime("%Y-%m-%d", time.gmtime()),
        },
        "licenses": coco.get("licenses") or [],
        "images": out_images,
        "annotations": out_anns,
        "categories": [{"id": new_category_id, "name": new_category_name, "supercategory": "none"}],
    }
    return out, orig_counts


def _copy_images(src_split_dir: Path, dst_split_dir: Path, file_names: List[str]) -> None:
    _safe_mkdir(dst_split_dir)
    for fn in file_names:
        src = src_split_dir / fn
        if not src.is_file():
            raise FileNotFoundError(str(src))
        shutil.copy2(str(src), str(dst_split_dir / src.name))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "roboflow_downloads"))
    ap.add_argument("--dst", default=str(REPO_ROOT / "data" / "segmentation_project_dataset"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_k", type=int, default=1800)
    ap.add_argument("--val_k", type=int, default=250)
    ap.add_argument("--test_k", type=int, default=250)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    splits = {"train": args.train_k, "valid": args.val_k, "test": args.test_k}

    # Layout
    dst_train = dst / "train"
    dst_val = dst / "val"
    dst_test = dst / "test"
    dst_ann = dst / "annotations"
    dst_reserve = dst / "reserve"
    dst_reports = dst / "reports"
    for d in (dst_train, dst_val, dst_test, dst_ann, dst_reserve, dst_reports):
        _safe_mkdir(d)

    summary_lines: List[str] = []
    summary_lines.append("segmentation_project_dataset subset summary")
    summary_lines.append(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    summary_lines.append("")
    summary_lines.append("Source: data/roboflow_downloads (COCO with segmentation)")
    summary_lines.append("Target: data/segmentation_project_dataset")
    summary_lines.append("")
    summary_lines.append("Segmentation training is 1-class: 0 = strawberry")
    summary_lines.append("Original ripeness categories are unified to strawberry for segmentation.")
    summary_lines.append("")

    total_orig_by_cat: Dict[str, int] = {}
    total_unified_objs = 0

    for split, k in splits.items():
        coco_path = src / split / "_annotations.coco.json"
        if not coco_path.is_file():
            raise FileNotFoundError(str(coco_path))
        coco = _load_json(coco_path)
        stats = _image_stats(coco)
        if k > len(stats):
            raise ValueError(f"requested {k} images for {split}, but only {len(stats)} available")

        picked = _stratified_sample(stats, k, seed=args.seed + (0 if split == "train" else (1 if split == "valid" else 2)))
        picked_ids = {p.image_id for p in picked}
        reserve_ids = {s.image_id for s in stats if s.image_id not in picked_ids}

        # Write subset + reserve annotations (unified)
        subset_coco, orig_counts = _unify_to_one_class(coco, picked_ids)
        reserve_coco, reserve_orig_counts = _unify_to_one_class(coco, reserve_ids)

        for name, n in orig_counts.items():
            total_orig_by_cat[name] = total_orig_by_cat.get(name, 0) + n
        for name, n in reserve_orig_counts.items():
            total_orig_by_cat[name] = total_orig_by_cat.get(name, 0) + 0  # keep keys stable

        subset_out = dst_ann / f"instances_{'val' if split == 'valid' else split}.json"
        reserve_out = dst_reserve / "annotations" / f"instances_{'val' if split == 'valid' else split}.json"
        _write_json(subset_out, subset_coco)
        _write_json(reserve_out, reserve_coco)

        # Copy images
        src_split_dir = src / split
        if split == "train":
            _copy_images(src_split_dir, dst_train, [p.file_name for p in picked])
            _copy_images(src_split_dir, dst_reserve / "train", [s.file_name for s in stats if s.image_id in reserve_ids])
        elif split == "valid":
            _copy_images(src_split_dir, dst_val, [p.file_name for p in picked])
            _copy_images(src_split_dir, dst_reserve / "val", [s.file_name for s in stats if s.image_id in reserve_ids])
        else:
            _copy_images(src_split_dir, dst_test, [p.file_name for p in picked])
            _copy_images(src_split_dir, dst_reserve / "test", [s.file_name for s in stats if s.image_id in reserve_ids])

        total_unified_objs += len(subset_coco.get("annotations") or [])

        # Split summary
        bcnt: Dict[str, int] = {"mixed": 0, "ripe_only": 0, "unripe_only": 0, "no_objects": 0}
        for p in picked:
            bcnt[p.bucket] += 1
        summary_lines.append(f"{split}: selected_images={len(picked)} reserve_images={len(stats) - len(picked)}")
        summary_lines.append(f"  selected buckets: mixed={bcnt['mixed']} ripe_only={bcnt['ripe_only']} unripe_only={bcnt['unripe_only']} no_objects={bcnt['no_objects']}")
        summary_lines.append(f"  selected objects (after unify to 1-class): {len(subset_coco.get('annotations') or [])}")
        summary_lines.append("")

    summary_lines.append("Original categories (source): Strawberries, ripe, unripe")
    summary_lines.append("Unified categories (target): strawberry")
    summary_lines.append("")
    summary_lines.append("Objects count (selected subsets only):")
    for k in sorted(total_orig_by_cat.keys()):
        if k in ("ripe", "unripe", "Strawberries"):
            summary_lines.append(f"  {k}: {total_orig_by_cat.get(k, 0)}")
    summary_lines.append(f"  unified(strawberry): {total_unified_objs}")
    summary_lines.append("")
    summary_lines.append("Note: reserve annotations are also unified to strawberry and stored under reserve/annotations/")

    _safe_mkdir(dst_reports)
    (dst_reports / "subset_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {dst_reports / 'subset_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

