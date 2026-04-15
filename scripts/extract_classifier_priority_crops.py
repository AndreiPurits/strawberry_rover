#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CropCandidate:
    split: str
    image_file: Path
    image_stem: str
    ann_id: int
    obj_idx: int
    label: str  # ripe/unripe
    bbox: Tuple[float, float, float, float]  # x,y,w,h


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _index_coco(coco: dict) -> Tuple[Dict[int, dict], Dict[int, List[dict]], Dict[int, str]]:
    images = {int(im["id"]): im for im in (coco.get("images") or [])}
    anns_by_img: Dict[int, List[dict]] = {}
    for a in coco.get("annotations") or []:
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)
    cats = {int(c["id"]): str(c.get("name") or "") for c in (coco.get("categories") or [])}
    return images, anns_by_img, cats


def _padded_bbox_xyxy(bbox_xywh: Tuple[float, float, float, float], pad_frac: float) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    px = w * pad_frac
    py = h * pad_frac
    x1 = x - px
    y1 = y - py
    x2 = x + w + px
    y2 = y + h + py
    return int(x1), int(y1), int(x2), int(y2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "roboflow_downloads"))
    ap.add_argument("--dst", default=str(REPO_ROOT / "data" / "classifier_priority_queue"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ripe_k", type=int, default=250)
    ap.add_argument("--unripe_k", type=int, default=250)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--min_size", type=int, default=40)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    out_ripe = dst / "ripe"
    out_unripe = dst / "unripe"
    reports = dst / "reports"
    for d in (out_ripe, out_unripe, reports):
        _safe_mkdir(d)

    rng = random.Random(args.seed)

    candidates_by_lbl: Dict[str, List[CropCandidate]] = {"ripe": [], "unripe": []}
    for split in ("train", "valid", "test"):
        coco_path = src / split / "_annotations.coco.json"
        coco = _load_json(coco_path)
        images, anns_by_img, cats = _index_coco(coco)
        for img_id, im in images.items():
            fn = str(im.get("file_name") or "")
            img_path = src / split / fn
            if not img_path.is_file():
                continue
            stem = Path(fn).stem
            anns = anns_by_img.get(img_id, [])
            for obj_idx, a in enumerate(anns):
                name = cats.get(int(a["category_id"]), "")
                if name not in ("ripe", "unripe"):
                    continue
                bbox = a.get("bbox") or None
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                cand = CropCandidate(
                    split=split,
                    image_file=img_path,
                    image_stem=stem,
                    ann_id=int(a.get("id") or 0),
                    obj_idx=obj_idx,
                    label=name,
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                )
                candidates_by_lbl[name].append(cand)

    for lbl in ("ripe", "unripe"):
        rng.shuffle(candidates_by_lbl[lbl])

    need = {"ripe": args.ripe_k, "unripe": args.unripe_k}
    picked: Dict[str, List[CropCandidate]] = {"ripe": [], "unripe": []}
    filtered_small = {"ripe": 0, "unripe": 0}
    by_split: Dict[str, Dict[str, int]] = {"train": {"ripe": 0, "unripe": 0}, "valid": {"ripe": 0, "unripe": 0}, "test": {"ripe": 0, "unripe": 0}}

    def out_name(c: CropCandidate) -> str:
        return f"{c.split}__{c.image_stem}__obj{c.obj_idx:03d}__{c.label}.jpg"

    # Start clean to avoid mixing old priority queues.
    for d in (out_ripe, out_unripe):
        for p in d.glob("*.jpg"):
            p.unlink()

    for lbl in ("ripe", "unripe"):
        out_dir = out_ripe if lbl == "ripe" else out_unripe
        for c in candidates_by_lbl[lbl]:
            if len(picked[lbl]) >= need[lbl]:
                break
            try:
                with Image.open(c.image_file) as im:
                    im = im.convert("RGB")
                    W, H = im.size
                    x1, y1, x2, y2 = _padded_bbox_xyxy(c.bbox, args.pad)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)
                    if (x2 - x1) < args.min_size or (y2 - y1) < args.min_size:
                        filtered_small[lbl] += 1
                        continue
                    crop = im.crop((x1, y1, x2, y2))
                    # Basic quality gate: avoid near-empty tiny crops (already filtered by min_size).
                    out_path = out_dir / out_name(c)
                    crop.save(out_path, format="JPEG", quality=92, optimize=True)
                picked[lbl].append(c)
                by_split[c.split][lbl] += 1
            except Exception:
                # Skip unreadable images/candidates.
                continue

    lines: List[str] = []
    lines.append("classifier_priority_queue crops report")
    lines.append(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    lines.append("")
    lines.append(f"Source COCO: data/roboflow_downloads")
    lines.append(f"Saved to: data/classifier_priority_queue/{{ripe,unripe}}")
    lines.append("")
    lines.append(f"Requested: ripe={need['ripe']} unripe={need['unripe']}")
    lines.append(f"Extracted: ripe={len(picked['ripe'])} unripe={len(picked['unripe'])}")
    lines.append(f"Filtered too-small (after padding, min {args.min_size}px): ripe={filtered_small['ripe']} unripe={filtered_small['unripe']}")
    lines.append("")
    lines.append("By split (extracted):")
    for s in ("train", "valid", "test"):
        lines.append(f"  {s}: ripe={by_split[s]['ripe']} unripe={by_split[s]['unripe']}")
    lines.append("")
    if len(picked["ripe"]) < need["ripe"] or len(picked["unripe"]) < need["unripe"]:
        lines.append("WARNING: could not reach requested counts; consider lowering min_size or increasing source pool.")
        lines.append("")

    _safe_mkdir(reports)
    (reports / "priority_crops_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {reports / 'priority_crops_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

