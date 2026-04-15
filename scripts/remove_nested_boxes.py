#!/usr/bin/env python3
"""
Remove "small boxes fully inside another box" from YOLO label files.

This script scans the workspace for directories named `labels` containing `.txt`
files with YOLO (class xc yc w h) format (normalized floats).

Rule (default):
- Box A is removed if it is fully inside box B (with epsilon) AND
  area(A) <= area(B) * area_ratio_threshold.

Safety:
- For every modified label file, we copy its original contents to:
  diagnostics/label_backups/<timestamp_ms>/<relative_path>
- We also append JSONL actions to:
  diagnostics/nested_box_cleanup.jsonl

Usage:
  python3 scripts/remove_nested_boxes.py
  python3 scripts/remove_nested_boxes.py --root data/final_detection_dataset/labels
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = REPO_ROOT / "diagnostics"
BACKUP_ROOT = DIAG_DIR / "label_backups"
LOG_PATH = DIAG_DIR / "nested_box_cleanup.jsonl"


@dataclass(frozen=True)
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float
    raw_line: str
    line_idx: int

    def xyxy(self) -> Tuple[float, float, float, float]:
        x1 = self.xc - self.w / 2.0
        y1 = self.yc - self.h / 2.0
        x2 = self.xc + self.w / 2.0
        y2 = self.yc + self.h / 2.0
        return x1, y1, x2, y2

    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _append_jsonl(event: dict) -> None:
    _safe_mkdir(LOG_PATH.parent)
    event = dict(event)
    event.setdefault("ts", time.time())
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _parse_yolo_boxes(text: str) -> List[Box]:
    out: List[Box] = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        toks = line.split()
        if len(toks) != 5:
            continue
        try:
            cls = int(float(toks[0]))
            xc, yc, w, h = map(float, toks[1:])
        except Exception:
            continue
        # keep even if values are out of range; containment check will handle
        out.append(Box(cls=cls, xc=xc, yc=yc, w=w, h=h, raw_line=raw, line_idx=i))
    return out


def _contains(outer: Box, inner: Box, eps: float) -> bool:
    ox1, oy1, ox2, oy2 = outer.xyxy()
    ix1, iy1, ix2, iy2 = inner.xyxy()
    return (ix1 >= ox1 - eps) and (iy1 >= oy1 - eps) and (ix2 <= ox2 + eps) and (iy2 <= oy2 + eps)


def _pick_nested_to_remove(
    boxes: List[Box],
    *,
    eps: float,
    area_ratio_threshold: float,
) -> List[int]:
    """
    Return list of original line indices to remove.
    We remove the INNER (smaller) boxes that are fully contained in a larger box.
    """
    if len(boxes) < 2:
        return []
    remove_line_idxs = set()
    # Compare all pairs. O(n^2) is fine for typical label sizes.
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue
            a = boxes[i]
            b = boxes[j]
            # remove a if it's inside b and sufficiently smaller
            aa = a.area()
            ba = b.area()
            if ba <= 0 or aa <= 0:
                continue
            if aa > ba * area_ratio_threshold:
                continue
            if _contains(b, a, eps=eps):
                remove_line_idxs.add(a.line_idx)
    return sorted(remove_line_idxs)


def _rewrite_label_file(path: Path, remove_line_idxs: List[int]) -> Tuple[str, str]:
    """
    Return (before_text, after_text). Keeps original line order for kept lines.
    """
    before = path.read_text(encoding="utf-8", errors="ignore")
    if not remove_line_idxs:
        return before, before
    remove_set = set(remove_line_idxs)
    lines = before.splitlines()
    kept = []
    for idx, ln in enumerate(lines):
        if idx in remove_set:
            continue
        if ln.strip() == "":
            continue
        kept.append(ln.strip())
    after = ("\n".join(kept).rstrip() + ("\n" if kept else ""))
    return before, after


def _iter_label_files(root: Optional[Path]) -> List[Path]:
    if root is not None:
        if root.is_file() and root.suffix.lower() == ".txt":
            return [root]
        if root.is_dir():
            return sorted([p for p in root.rglob("*.txt") if p.is_file()])
        return []

    # Default: scan for directories named 'labels'
    candidates: List[Path] = []
    for d in REPO_ROOT.rglob("labels"):
        if not d.is_dir():
            continue
        # skip diagnostics trash/backups etc
        if "diagnostics" in d.parts:
            continue
        for p in d.glob("*.txt"):
            if p.is_file():
                candidates.append(p)
    return sorted(candidates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None, help="File or directory to scan (defaults to all */labels/*.txt)")
    ap.add_argument("--eps", type=float, default=1e-6, help="Containment epsilon in normalized coords")
    ap.add_argument(
        "--area-ratio-threshold",
        type=float,
        default=0.5,
        help="Remove inner if area(inner) <= area(outer) * threshold",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be removed; do not modify any files",
    )
    ap.add_argument(
        "--report-limit",
        type=int,
        default=20,
        help="How many example files to print in dry-run mode",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else None
    files = _iter_label_files(root)
    if not files:
        print("No label files found.")
        return 0

    backup_stamp = str(_now_ms())
    changed_files = 0
    removed_total = 0
    scanned = 0
    would_change_files = 0
    examples_printed = 0

    for p in files:
        scanned += 1
        before = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        boxes = _parse_yolo_boxes(before)
        to_remove = _pick_nested_to_remove(
            boxes,
            eps=float(args.eps),
            area_ratio_threshold=float(args.area_ratio_threshold),
        )
        if not to_remove:
            continue
        if args.dry_run:
            would_change_files += 1
            removed_total += len(to_remove)
            if examples_printed < int(args.report_limit):
                rel = p.relative_to(REPO_ROOT)
                print(f"would_change: {rel} remove_line_idxs={to_remove}")
                examples_printed += 1
            continue
        before_text, after_text = _rewrite_label_file(p, to_remove)
        if after_text == before_text:
            continue

        # backup
        rel = p.relative_to(REPO_ROOT)
        backup_path = BACKUP_ROOT / backup_stamp / rel
        _safe_mkdir(backup_path.parent)
        backup_path.write_text(before_text, encoding="utf-8")

        # write
        p.write_text(after_text, encoding="utf-8")

        changed_files += 1
        removed_total += len(to_remove)
        _append_jsonl(
            {
                "action": "remove_nested_boxes",
                "file": str(rel),
                "removed_line_idxs": to_remove,
                "removed_count": len(to_remove),
                "backup": str(backup_path.relative_to(REPO_ROOT)),
                "eps": float(args.eps),
                "area_ratio_threshold": float(args.area_ratio_threshold),
            }
        )

    print(f"scanned_files: {scanned}")
    print(f"removed_boxes_total: {removed_total}")
    if args.dry_run:
        print(f"would_change_files: {would_change_files}")
    else:
        print(f"changed_files: {changed_files}")
        if changed_files:
            print(f"backups: diagnostics/label_backups/{backup_stamp}/...")
            print(f"log: {LOG_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

