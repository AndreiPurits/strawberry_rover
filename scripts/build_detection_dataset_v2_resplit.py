#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _dhash64(pil_img: Image.Image) -> int:
    # 8x8 dhash (64-bit)
    img = pil_img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = list(img.getdata())
    out = 0
    bit = 1 << 63
    for y in range(8):
        row = px[y * 9 : (y + 1) * 9]
        for x in range(8):
            out |= bit if row[x] > row[x + 1] else 0
            bit >>= 1
    return out


def _hamming64(a: int, b: int) -> int:
    x = a ^ b
    bc = getattr(x, "bit_count", None)
    if callable(bc):
        return int(bc())
    return bin(int(x) & ((1 << 64) - 1)).count("1")


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


@dataclass
class PoolItem:
    uid: str
    source: str  # "base" or "roboflow"
    src_image: Path
    # For base: label file exists. For roboflow: label_lines precomputed.
    src_label: Optional[Path]
    label_lines: Optional[List[str]]
    width: int
    height: int
    sha256: str
    dhash64: int


def _read_label_lines(p: Path) -> List[str]:
    if not p.is_file():
        return []
    out: List[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            continue
        try:
            # normalize class to 0 (single-class)
            xc, yc, w, h = map(float, toks[1:])
        except Exception:
            continue
        out.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return out


def _iter_base_pool(base: Path) -> List[PoolItem]:
    items: List[PoolItem] = []
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        for img in sorted(img_dir.iterdir()) if img_dir.is_dir() else []:
            if not _is_image(img):
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.is_file():
                continue
            try:
                with Image.open(img) as im:
                    w, h = im.size
                    dh = _dhash64(im)
            except Exception:
                continue
            sh = _sha256_file(img)
            uid = f"base__{img.stem}"
            items.append(
                PoolItem(
                    uid=uid,
                    source="base",
                    src_image=img,
                    src_label=lbl,
                    label_lines=None,
                    width=w,
                    height=h,
                    sha256=sh,
                    dhash64=dh,
                )
            )
    return items


def _load_roboflow_coco(rf_dir: Path, coco_json: Path) -> Tuple[Dict[int, dict], Dict[int, List[List[float]]]]:
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


def _sample_roboflow_pool(
    rf_dir: Path,
    coco_json: Path,
    *,
    add_count: int,
    seed: int,
) -> List[PoolItem]:
    images_by_id, ann_bboxes = _load_roboflow_coco(rf_dir, coco_json)
    candidates: List[Tuple[int, Path, int, int]] = []
    for iid, im in images_by_id.items():
        fn = str(im.get("file_name") or "")
        p = rf_dir / fn
        if not _is_image(p):
            continue
        if iid not in ann_bboxes:
            continue
        try:
            w = int(im.get("width"))
            h = int(im.get("height"))
        except Exception:
            continue
        candidates.append((iid, p, w, h))

    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    chosen = candidates[: int(add_count)]
    out: List[PoolItem] = []
    for iid, img, w, h in chosen:
        # compute hashes on image bytes for dedup/leak prevention
        try:
            with Image.open(img) as im:
                dh = _dhash64(im)
        except Exception:
            continue
        sh = _sha256_file(img)
        lines: List[str] = []
        for b in ann_bboxes.get(iid, []):
            xc, yc, wn, hn = _coco_bbox_to_yolo(b, img_w=w, img_h=h)
            if wn <= 0 or hn <= 0:
                continue
            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        uid = f"roboflow__{img.stem}"
        out.append(
            PoolItem(
                uid=uid,
                source="roboflow",
                src_image=img,
                src_label=None,
                label_lines=lines,
                width=w,
                height=h,
                sha256=sh,
                dhash64=dh,
            )
        )
    return out


def _group_items(items: List[PoolItem], *, near_hamming: int) -> List[List[PoolItem]]:
    # Exact sha256 groups first
    by_sha: Dict[str, List[PoolItem]] = {}
    for it in items:
        by_sha.setdefault(it.sha256, []).append(it)
    sha_groups = [g for g in by_sha.values()]

    if near_hamming <= 0:
        return sha_groups

    # Merge sha-groups by near-duplicate dhash within a coarse bucket (top 16 bits).
    # This is conservative and fast enough for ~1-2k images.
    # We treat any two items with hamming <= near_hamming as same group.
    buckets: Dict[int, List[PoolItem]] = {}
    for g in sha_groups:
        for it in g:
            buckets.setdefault(it.dhash64 >> 48, []).append(it)

    # Union-find on indices
    all_items = list(items)
    idx_of: Dict[str, int] = {it.uid: i for i, it in enumerate(all_items)}
    parent = list(range(len(all_items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _bk, its in buckets.items():
        n = len(its)
        if n < 2:
            continue
        its_sorted = sorted(its, key=lambda t: t.uid)
        for i in range(n):
            ai = its_sorted[i]
            ia = idx_of[ai.uid]
            for j in range(i + 1, n):
                bj = its_sorted[j]
                ib = idx_of[bj.uid]
                if _hamming64(ai.dhash64, bj.dhash64) <= near_hamming:
                    union(ia, ib)

    groups: Dict[int, List[PoolItem]] = {}
    for it in all_items:
        r = find(idx_of[it.uid])
        groups.setdefault(r, []).append(it)
    return list(groups.values())


def _write_split(out_root: Path, split: str, items: List[PoolItem]) -> Tuple[int, int]:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    _safe_mkdir(img_dir)
    _safe_mkdir(lbl_dir)
    boxes = 0
    for it in items:
        # keep filename stem stable by uid; avoid collisions
        ext = it.src_image.suffix.lower()
        stem = it.uid
        dst_img = img_dir / f"{stem}{ext}"
        dst_lbl = lbl_dir / f"{stem}.txt"
        shutil.copy2(it.src_image, dst_img)
        if it.source == "base":
            assert it.src_label is not None
            lines = _read_label_lines(it.src_label)
        else:
            lines = list(it.label_lines or [])
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        boxes += len(lines)
    return len(items), boxes


def main() -> int:
    ap = argparse.ArgumentParser(description="Build detector dataset v2 with stable test split from combined pool.")
    ap.add_argument("--base-dataset", default=str(REPO_ROOT / "data" / "yolo_detection_dataset"))
    ap.add_argument("--roboflow-split-dir", default=str(REPO_ROOT / "data" / "roboflow_downloads" / "train"))
    ap.add_argument("--roboflow-coco", default="", help="Defaults to <roboflow-split-dir>/_annotations.coco.json")
    ap.add_argument("--roboflow-add-count", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--near-dup-hamming", type=int, default=2, help="Prevent near-duplicate leakage by grouping dhash within this Hamming distance.")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac-of-remaining", type=float, default=0.15)
    ap.add_argument("--out-dataset", default=str(REPO_ROOT / "data" / "yolo_detection_dataset_v2"))
    args = ap.parse_args()

    base = Path(args.base_dataset)
    rf_dir = Path(args.roboflow_split_dir)
    coco = Path(args.roboflow_coco) if args.roboflow_coco else (rf_dir / "_annotations.coco.json")
    out = Path(args.out_dataset)

    if out.exists():
        raise SystemExit(f"Out dataset already exists: {out} (delete it if you want to rebuild)")

    base_items = _iter_base_pool(base)
    rf_items = _sample_roboflow_pool(rf_dir, coco, add_count=int(args.roboflow_add_count), seed=int(args.seed))
    pool = base_items + rf_items

    # Group to prevent leakage
    groups = _group_items(pool, near_hamming=max(0, int(args.near_dup_hamming)))

    # Stable shuffling by seed on group key (sorted uids)
    rng = random.Random(int(args.seed))
    group_keys = [(sorted(it.uid for it in g), g) for g in groups]
    rng.shuffle(group_keys)
    groups_shuffled = [g for _k, g in group_keys]

    total_imgs = sum(len(g) for g in groups_shuffled)
    test_target = int(round(float(args.test_frac) * total_imgs))

    test_groups: List[List[PoolItem]] = []
    rem_groups: List[List[PoolItem]] = []
    acc = 0
    for g in groups_shuffled:
        if acc < test_target:
            test_groups.append(g)
            acc += len(g)
        else:
            rem_groups.append(g)

    remaining_imgs = total_imgs - acc
    val_target = int(round(float(args.val_frac_of_remaining) * remaining_imgs))

    val_groups: List[List[PoolItem]] = []
    train_groups: List[List[PoolItem]] = []
    acc_val = 0
    for g in rem_groups:
        if acc_val < val_target:
            val_groups.append(g)
            acc_val += len(g)
        else:
            train_groups.append(g)

    # Flatten
    test_items = [it for g in test_groups for it in g]
    val_items = [it for g in val_groups for it in g]
    train_items = [it for g in train_groups for it in g]

    # Write dataset
    _safe_mkdir(out / "reports")
    n_train, b_train = _write_split(out, "train", train_items)
    n_val, b_val = _write_split(out, "val", val_items)
    n_test, b_test = _write_split(out, "test", test_items)

    # data.yaml (single class)
    (out / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: strawberry",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # split report
    def _src_counts(items: List[PoolItem]) -> Dict[str, int]:
        outc: Dict[str, int] = {}
        for it in items:
            outc[it.source] = outc.get(it.source, 0) + 1
        return outc

    dup_groups = sum(1 for g in groups if len(g) > 1)
    max_group = max((len(g) for g in groups), default=1)

    report_lines = [
        "yolo_detection_dataset_v2 split_report",
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "",
        f"seed: {args.seed}",
        f"roboflow_add_count: {args.roboflow_add_count}",
        f"near_dup_hamming: {args.near_dup_hamming}",
        "",
        f"pool_total_images: {total_imgs}",
        f"pool_sources: base={len(base_items)} roboflow={len(rf_items)}",
        f"grouping: groups={len(groups)} dup_groups(size>1)={dup_groups} max_group_size={max_group}",
        "",
        f"split_test_frac: {args.test_frac}",
        f"split_val_frac_of_remaining: {args.val_frac_of_remaining}",
        "",
        f"train_images: {n_train}",
        f"val_images: {n_val}",
        f"test_images: {n_test}",
        "",
        f"train_bboxes: {b_train}",
        f"val_bboxes: {b_val}",
        f"test_bboxes: {b_test}",
        "",
        f"train_sources: {json.dumps(_src_counts(train_items), ensure_ascii=False)}",
        f"val_sources: {json.dumps(_src_counts(val_items), ensure_ascii=False)}",
        f"test_sources: {json.dumps(_src_counts(test_items), ensure_ascii=False)}",
        "",
        "NOTE: base source includes BOTH old train+val. Roboflow sampled from COCO (train split) with seed.",
    ]
    (out / "reports" / "split_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # manifest for reproducibility
    manifest = out / "reports" / "pool_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for it in pool:
            f.write(
                json.dumps(
                    {
                        "uid": it.uid,
                        "source": it.source,
                        "src_image": str(it.src_image),
                        "src_label": str(it.src_label) if it.src_label else "",
                        "width": it.width,
                        "height": it.height,
                        "sha256": it.sha256,
                        "dhash64": f"{it.dhash64:016x}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Built: {out}")
    print(f"Pool total images: {total_imgs} (base {len(base_items)} + roboflow {len(rf_items)})")
    print(f"Splits: train={n_train} val={n_val} test={n_test}")
    print(f"Report: {out/'reports'/'split_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

