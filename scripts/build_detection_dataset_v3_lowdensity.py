#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _sha256_file(p: Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _coco_bbox_to_yolo(b: List[float], *, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x, y, w, h = map(float, b)
    xc = (x + w / 2.0) / float(img_w)
    yc = (y + h / 2.0) / float(img_h)
    wn = w / float(img_w)
    hn = h / float(img_h)
    return (
        _clamp(xc, 0.0, 1.0),
        _clamp(yc, 0.0, 1.0),
        _clamp(wn, 0.0, 1.0),
        _clamp(hn, 0.0, 1.0),
    )


def _count_yolo_label_boxes(lbl_path: Path) -> int:
    if not lbl_path.is_file():
        return 0
    n = 0
    for raw in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            continue
        n += 1
    return n


def _normalize_yolo_lines_to_single_class(lbl_path: Path) -> List[str]:
    out: List[str] = []
    if not lbl_path.is_file():
        return out
    for raw in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            continue
        try:
            xc, yc, w, h = map(float, toks[1:])
        except Exception:
            continue
        out.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out


@dataclass(frozen=True)
class Candidate:
    source: str  # "old" or "roboflow"
    image_path: Path
    bbox_count: int
    # Roboflow only:
    coco_width: int = 0
    coco_height: int = 0
    coco_bboxes: Optional[List[List[float]]] = None

    @property
    def image_name(self) -> str:
        return self.image_path.name


def _iter_old_candidates(old_root: Path) -> List[Candidate]:
    out: List[Candidate] = []
    for split in ("train", "val"):
        img_dir = old_root / "images" / split
        lbl_dir = old_root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir(), key=lambda p: p.name):
            if not _is_image(img):
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.is_file():
                continue
            out.append(Candidate(source="old", image_path=img, bbox_count=_count_yolo_label_boxes(lbl)))
    return out


def _load_roboflow_coco(
    split_dir: Path,
) -> Tuple[Dict[int, dict], Dict[int, List[List[float]]]]:
    coco_json = split_dir / "_annotations.coco.json"
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images_by_id: Dict[int, dict] = {}
    for im in data.get("images", []):
        try:
            iid = int(im["id"])
            images_by_id[iid] = im
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


def _iter_roboflow_candidates(rf_root: Path) -> List[Candidate]:
    out: List[Candidate] = []
    for split in ("train", "valid", "test"):
        d = rf_root / split
        if not d.is_dir():
            continue
        coco_json = d / "_annotations.coco.json"
        if not coco_json.is_file():
            continue
        images_by_id, ann_bboxes = _load_roboflow_coco(d)
        for iid, im in images_by_id.items():
            fn = str(im.get("file_name") or "")
            p = d / fn
            if not _is_image(p):
                continue
            bxs = ann_bboxes.get(iid) or []
            if not bxs:
                continue
            try:
                w = int(im.get("width"))
                h = int(im.get("height"))
            except Exception:
                continue
            out.append(
                Candidate(
                    source="roboflow",
                    image_path=p,
                    bbox_count=len(bxs),
                    coco_width=w,
                    coco_height=h,
                    coco_bboxes=bxs,
                )
            )
    return out


def _select_low_density(
    candidates: List[Candidate],
    *,
    want: int,
    seed: int,
    global_seen_sha: set,
) -> Tuple[List[Candidate], int]:
    # Primary key: bbox_count asc, secondary stable by filename.
    ordered = sorted(candidates, key=lambda c: (c.bbox_count, c.image_name))
    selected: List[Candidate] = []
    duplicates_skipped = 0

    # Deterministic tie-breaking already by name; seed only used if we need to shuffle backfill later.
    rng = random.Random(int(seed))

    for c in ordered:
        if len(selected) >= int(want):
            break
        try:
            sh = _sha256_file(c.image_path)
        except Exception:
            continue
        if sh in global_seen_sha:
            duplicates_skipped += 1
            continue
        global_seen_sha.add(sh)
        selected.append(c)

    # If we couldn't fill due to duplicates/IO errors, do a seeded shuffle of remaining and try to fill.
    if len(selected) < int(want):
        remaining = [c for c in ordered if c not in selected]
        rng.shuffle(remaining)
        for c in remaining:
            if len(selected) >= int(want):
                break
            try:
                sh = _sha256_file(c.image_path)
            except Exception:
                continue
            if sh in global_seen_sha:
                duplicates_skipped += 1
                continue
            global_seen_sha.add(sh)
            selected.append(c)

    return selected, duplicates_skipped


def _write_yolo_label_single_class_for_candidate(c: Candidate) -> List[str]:
    if c.source == "old":
        # Derive label path from old dataset layout.
        # old_root/.../images/{train|val}/X.jpg -> labels/{train|val}/X.txt
        parts = list(c.image_path.parts)
        try:
            i = parts.index("images")
            split = parts[i + 1]
        except Exception:
            split = "train"
        lbl = Path(*parts[:i]) / "labels" / split / f"{c.image_path.stem}.txt"
        return _normalize_yolo_lines_to_single_class(lbl)

    # Roboflow COCO bboxes
    lines: List[str] = []
    bxs = c.coco_bboxes or []
    for b in bxs:
        xc, yc, wn, hn = _coco_bbox_to_yolo(b, img_w=int(c.coco_width), img_h=int(c.coco_height))
        if wn <= 0 or hn <= 0:
            continue
        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return lines


def _count_lines(lines: List[str]) -> int:
    return sum(1 for x in lines if str(x).strip())


def _split_items(
    items: List[Candidate],
    *,
    seed: int,
    test_frac: float,
    val_frac: float,
) -> Dict[str, List[Candidate]]:
    assert 0 < test_frac < 1
    assert 0 < val_frac < 1
    assert test_frac + val_frac < 1
    rng = random.Random(int(seed))
    items2 = list(items)
    rng.shuffle(items2)
    n = len(items2)
    n_test = int(round(n * float(test_frac)))
    n_val = int(round(n * float(val_frac)))
    n_train = n - n_test - n_val
    return {
        "train": items2[:n_train],
        "val": items2[n_train : n_train + n_val],
        "test": items2[n_train + n_val :],
    }


def _write_dataset_yaml(out_root: Path) -> None:
    txt = "\n".join(
        [
            f"path: {out_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: strawberry",
            "",
        ]
    )
    (out_root / "data.yaml").write_text(txt, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build detector dataset v3 by selecting low-density images from old + roboflow sources.")
    ap.add_argument("--old-root", type=Path, default=REPO_ROOT / "data" / "yolo_detection_dataset")
    ap.add_argument("--roboflow-root", type=Path, default=REPO_ROOT / "data" / "roboflow_downloads")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "data" / "yolo_detection_dataset_v3")
    ap.add_argument("--old-count", type=int, default=600)
    ap.add_argument("--roboflow-count", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args(list(argv) if argv is not None else None)

    t0 = time.time()
    out_root: Path = args.out_root
    images_root = out_root / "images"
    labels_root = out_root / "labels"
    reports_root = out_root / "reports"

    for split in ("train", "val", "test"):
        _safe_mkdir(images_root / split)
        _safe_mkdir(labels_root / split)
    _safe_mkdir(reports_root)

    # Load candidates
    old_candidates = _iter_old_candidates(Path(args.old_root))
    rf_candidates = _iter_roboflow_candidates(Path(args.roboflow_root))

    # Select low-density subsets with exact-dedup (sha256) across both sources.
    seen_sha: set = set()
    sel_old, dup_old = _select_low_density(
        old_candidates, want=int(args.old_count), seed=int(args.seed), global_seen_sha=seen_sha
    )
    sel_rf, dup_rf = _select_low_density(
        rf_candidates, want=int(args.roboflow_count), seed=int(args.seed), global_seen_sha=seen_sha
    )
    selected = sel_old + sel_rf

    # Split selected into train/val/test
    splits = _split_items(
        selected,
        seed=int(args.seed),
        test_frac=float(args.test_frac),
        val_frac=float(args.val_frac),
    )

    # CSV selection report (all candidates for transparency, plus selected and split where applicable)
    selected_set = {c.image_path.resolve(): c for c in selected}
    split_of: Dict[Path, str] = {}
    for sp, items in splits.items():
        for c in items:
            split_of[c.image_path.resolve()] = sp

    csv_path = reports_root / "image_selection_v3.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_dataset",
                "image_name",
                "bbox_count",
                "selected_for_v3",
                "split",
            ],
        )
        w.writeheader()

        def _write_rows(cands: List[Candidate]) -> None:
            for c in sorted(cands, key=lambda x: (x.bbox_count, x.image_name)):
                rp = c.image_path.resolve()
                sel = "yes" if rp in selected_set else "no"
                sp = split_of.get(rp, "none")
                w.writerow(
                    {
                        "source_dataset": c.source,
                        "image_name": c.image_name,
                        "bbox_count": c.bbox_count,
                        "selected_for_v3": sel,
                        "split": sp,
                    }
                )

        _write_rows(old_candidates)
        _write_rows(rf_candidates)

    # Materialize dataset files
    split_bbox_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    split_img_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for sp, items in splits.items():
        for c in items:
            dst_img = images_root / sp / c.image_path.name
            if not dst_img.exists():
                shutil.copy2(c.image_path, dst_img)
            lines = _write_yolo_label_single_class_for_candidate(c)
            (labels_root / sp / f"{c.image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            split_bbox_counts[sp] += _count_lines(lines)
            split_img_counts[sp] += 1

    # data.yaml
    _write_dataset_yaml(out_root)

    # Stats for report
    def _range_mean(vals: List[int]) -> Tuple[int, int, float]:
        if not vals:
            return 0, 0, 0.0
        return min(vals), max(vals), sum(vals) / len(vals)

    old_counts_sel = [c.bbox_count for c in sel_old]
    rf_counts_sel = [c.bbox_count for c in sel_rf]
    all_counts_sel = [c.bbox_count for c in selected]
    old_min, old_max, old_mean = _range_mean(old_counts_sel)
    rf_min, rf_max, rf_mean = _range_mean(rf_counts_sel)
    all_min, all_max, all_mean = _range_mean(all_counts_sel)

    prep_report = reports_root / "prep_report_v3.txt"
    lines = [
        "DETECTOR DATASET V3 (low-density) PREP REPORT",
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "",
        f"old_source_root: {Path(args.old_root).resolve()}",
        f"roboflow_source_root: {Path(args.roboflow_root).resolve()}",
        f"out_root: {out_root.resolve()}",
        "",
        "Selection rules:",
        "- Compute bbox_count per image (old: YOLO label lines; roboflow: COCO annotations per image).",
        "- Sort by bbox_count ascending, then stable by filename.",
        "- Select N images per source with lowest bbox_count.",
        "- Prevent exact duplicates (sha256) across both sources during selection.",
        "",
        f"Requested selection: old={int(args.old_count)} roboflow={int(args.roboflow_count)} total={int(args.old_count)+int(args.roboflow_count)}",
        f"Selected: old={len(sel_old)} roboflow={len(sel_rf)} total={len(selected)}",
        "",
        "Selected bbox_count stats:",
        f"- old subset: min={old_min} max={old_max} mean={old_mean:.3f}",
        f"- roboflow subset: min={rf_min} max={rf_max} mean={rf_mean:.3f}",
        f"- total: min={all_min} max={all_max} mean={all_mean:.3f}",
        "",
        "Split:",
        f"- seed={int(args.seed)} test_frac={float(args.test_frac):.3f} val_frac={float(args.val_frac):.3f}",
        f"- train images: {split_img_counts['train']}  bboxes: {split_bbox_counts['train']}",
        f"- val images:   {split_img_counts['val']}  bboxes: {split_bbox_counts['val']}",
        f"- test images:  {split_img_counts['test']}  bboxes: {split_bbox_counts['test']}",
        "",
        "Duplicates handling:",
        f"- exact_duplicate_images_skipped_old: {dup_old}",
        f"- exact_duplicate_images_skipped_roboflow: {dup_rf}",
        "",
        "Label normalization:",
        "- All labels written as single-class YOLO bbox: class_id forced to 0; bbox coordinates unchanged.",
        "",
        f"Wrote CSV: {csv_path}",
        f"Wrote dataset yaml: {out_root / 'data.yaml'}",
        "",
        f"Elapsed seconds: {time.time()-t0:.1f}",
    ]
    prep_report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {prep_report}")
    print(f"Wrote: {csv_path}")
    print(f"Dataset v3 root: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

