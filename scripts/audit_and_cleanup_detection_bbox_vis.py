#!/usr/bin/env python3
"""
Audit `data/final_detection_dataset/bbox_vis` and keep dataset consistent.

For each image in bbox_vis/:
- verify original exists in images/
- verify label exists in labels/ and contains at least one valid YOLO bbox line

If any check fails, move the corresponding files (bbox_vis image, original image,
label) to diagnostics/curation_trash/detection/ (non-destructive) and log one
JSONL entry to diagnostics/curation_actions.jsonl.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DETECTION_ROOT = REPO_ROOT / "data" / "final_detection_dataset"
DET_IMAGES = DETECTION_ROOT / "images"
DET_LABELS = DETECTION_ROOT / "labels"
DET_BBOX_VIS = DETECTION_ROOT / "bbox_vis"

DIAG_DIR = REPO_ROOT / "diagnostics"
TRASH_DIR = DIAG_DIR / "curation_trash" / "detection"
LOG_PATH = DIAG_DIR / "curation_actions.jsonl"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _now() -> float:
    return time.time()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _log(event: Dict) -> None:
    _safe_mkdir(LOG_PATH.parent)
    event = dict(event)
    event.setdefault("ts", _now())
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _move_to_trash_if_exists(p: Path) -> Optional[Path]:
    if not (p.exists() and p.is_file()):
        return None
    _safe_mkdir(TRASH_DIR)
    ts = int(_now() * 1000)
    dst = TRASH_DIR / f"{ts}__{p.name}"
    p.rename(dst)
    return dst


def _label_valid_box_count(label_path: Path) -> int:
    if not (label_path.exists() and label_path.is_file()):
        return 0
    txt = label_path.read_text(encoding="utf-8", errors="ignore")
    n = 0
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            continue
        try:
            float(toks[0])
            float(toks[1])
            float(toks[2])
            float(toks[3])
            float(toks[4])
        except Exception:
            continue
        n += 1
    return n


def _find_original_image(stem: str, ext: str) -> Optional[Path]:
    direct = DET_IMAGES / f"{stem}{ext}"
    if _is_image(direct):
        return direct
    # fallback: any extension with same stem
    for e in sorted(IMG_EXTS):
        cand = DET_IMAGES / f"{stem}{e}"
        if _is_image(cand):
            return cand
    return None


def main() -> int:
    if not DET_BBOX_VIS.is_dir():
        print(f"Missing dir: {DET_BBOX_VIS}")
        return 2

    vis_imgs = [p for p in DET_BBOX_VIS.iterdir() if _is_image(p)]
    vis_imgs.sort()

    total = len(vis_imgs)
    bad: List[Tuple[str, List[str]]] = []
    moved_count = 0

    for vis in vis_imgs:
        stem = vis.stem
        ext = vis.suffix.lower()
        label = DET_LABELS / f"{stem}.txt"
        orig = _find_original_image(stem, ext)

        reasons: List[str] = []
        if orig is None:
            reasons.append("missing_original_image")
        if not label.exists():
            reasons.append("missing_label")
        else:
            if _label_valid_box_count(label) <= 0:
                reasons.append("empty_or_invalid_label")

        if not reasons:
            continue

        bad.append((vis.name, reasons))

        moved = []
        moved.append({"path": str(vis), "moved_to": str(_move_to_trash_if_exists(vis))})
        if orig is not None:
            moved.append({"path": str(orig), "moved_to": str(_move_to_trash_if_exists(orig))})
        moved.append({"path": str(label), "moved_to": str(_move_to_trash_if_exists(label))})

        _log(
            {
                "action": "cleanup_bad_detection_item",
                "dataset": "detection",
                "bbox_vis": vis.name,
                "reasons": reasons,
                "moved": moved,
            }
        )
        moved_count += 1

    print(f"bbox_vis_total: {total}")
    print(f"bad_items: {len(bad)}")
    print(f"moved_groups: {moved_count}")
    if bad:
        print("bad_examples:")
        for name, reasons in bad[:20]:
            print(" -", name, "=>", ",".join(reasons))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

