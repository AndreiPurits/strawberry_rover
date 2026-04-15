#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def _iter_images(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted([p for p in d.iterdir() if _is_image(p)], key=lambda p: p.name)


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
    # Difference hash 8x8 => 64-bit int.
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
    # Python 3.8+ has int.bit_count(); keep compatibility with older versions.
    bc = getattr(x, "bit_count", None)
    if callable(bc):
        return int(bc())
    return bin(int(x) & ((1 << 64) - 1)).count("1")


@dataclass
class GroupDup:
    keep: Path
    dups: List[Path]
    reason: str
    key: str


def find_exact_dups(paths: List[Path]) -> List[GroupDup]:
    by_hash: Dict[str, List[Path]] = {}
    for p in paths:
        try:
            h = _sha256_file(p)
        except Exception:
            continue
        by_hash.setdefault(h, []).append(p)
    out: List[GroupDup] = []
    for h, ps in by_hash.items():
        if len(ps) < 2:
            continue
        keep = ps[0]
        dups = ps[1:]
        out.append(GroupDup(keep=keep, dups=dups, reason="sha256", key=h))
    return out


def find_visual_dups(paths: List[Path], *, max_hamming: int) -> List[GroupDup]:
    # Bucket by dhash, then find close matches in same bucket (simple O(n^2) per bucket).
    by_hash: Dict[int, List[Tuple[Path, int]]] = {}
    for p in paths:
        try:
            with Image.open(p) as im:
                dh = _dhash64(im)
        except Exception:
            continue
        by_hash.setdefault(dh, []).append((p, dh))

    # Exact dhash duplicates
    out: List[GroupDup] = []
    used: set[Path] = set()

    # First, exact dhash collisions (max_hamming==0 covers)
    for dh, items in by_hash.items():
        if len(items) < 2:
            continue
        ps = [p for (p, _) in items]
        keep = ps[0]
        dups = ps[1:]
        out.append(GroupDup(keep=keep, dups=dups, reason="dhash", key=f"{dh:016x}"))
        used.update(ps)

    if max_hamming <= 0:
        return out

    # If max_hamming > 0: compare all dhash keys pairwise (coarse, may be slow on very large sets).
    # We keep it conservative and only compare within a small window by grouping on high bits.
    buckets: Dict[int, List[Tuple[Path, int]]] = {}
    for p, dh in [(p, dh) for items in by_hash.values() for (p, dh) in items]:
        buckets.setdefault(dh >> 48, []).append((p, dh))

    for _k, items in buckets.items():
        # compare within bucket
        n = len(items)
        if n < 2:
            continue
        items_sorted = sorted(items, key=lambda t: t[0].name)
        for i in range(n):
            pi, dhi = items_sorted[i]
            if pi in used:
                continue
            close: List[Path] = []
            for j in range(i + 1, n):
                pj, dhj = items_sorted[j]
                if pj in used:
                    continue
                if _hamming64(dhi, dhj) <= max_hamming:
                    close.append(pj)
            if close:
                used.add(pi)
                used.update(close)
                out.append(
                    GroupDup(
                        keep=pi,
                        dups=close,
                        reason=f"dhash_hamming<={max_hamming}",
                        key=f"{dhi:016x}",
                    )
                )
    return out


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def _move_to_trash(p: Path, *, trash_root: Path, tag: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst_dir = trash_root / tag
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{ts}__{p.name}"
    i = 1
    while dst.exists():
        dst = dst_dir / f"{ts}__{i:03d}__{p.name}"
        i += 1
    shutil.move(str(p), str(dst))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Find and remove duplicate classifier crops (candidates + manual).")
    ap.add_argument("--apply", action="store_true", help="Actually remove duplicates (move to trash).")
    ap.add_argument(
        "--delete",
        action="store_true",
        help="Dangerous: permanently delete duplicates (only if --apply). Default is move to diagnostics/dedup_trash.",
    )
    ap.add_argument(
        "--visual",
        action="store_true",
        help="Also find visual duplicates using dhash (can be slower).",
    )
    ap.add_argument(
        "--max-hamming",
        type=int,
        default=0,
        help="Max Hamming distance for dhash visual duplicates (default 0 = exact dhash match).",
    )
    ap.add_argument("--reports-dir", default=str(REPO_ROOT / "diagnostics" / "dedup_reports"))
    args = ap.parse_args()

    cand_all = REPO_ROOT / "data" / "classifier_candidates" / "all"
    cand_review = REPO_ROOT / "data" / "classifier_candidates" / "review_small"
    manual_root = REPO_ROOT / "data" / "classification_manual"
    manual_classes = [manual_root / c for c in ("green", "turning", "ripe", "rotten", "rejected")]

    groups: List[Tuple[str, List[Path]]] = [
        ("candidates_all", _iter_images(cand_all)),
        ("candidates_review_small", _iter_images(cand_review)),
    ]
    for d in manual_classes:
        groups.append((f"manual_{d.name}", _iter_images(d)))

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    trash_root = REPO_ROOT / "diagnostics" / "dedup_trash"

    report: Dict[str, object] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "apply": bool(args.apply),
        "delete": bool(args.delete),
        "visual": bool(args.visual),
        "max_hamming": int(args.max_hamming),
        "groups": {},
    }

    total_dups = 0
    total_removed = 0

    for tag, paths in groups:
        exact = find_exact_dups(paths)
        visual: List[GroupDup] = []
        if args.visual:
            visual = find_visual_dups(paths, max_hamming=max(0, int(args.max_hamming)))

        # Merge groups by path membership, preferring sha256 grouping.
        seen_dup: set[Path] = set()
        merged: List[GroupDup] = []
        for g in exact + visual:
            dups = [p for p in g.dups if p not in seen_dup and p != g.keep]
            if not dups:
                continue
            seen_dup.update(dups)
            merged.append(GroupDup(keep=g.keep, dups=dups, reason=g.reason, key=g.key))

        total_dups += sum(len(g.dups) for g in merged)

        removed: List[Dict[str, str]] = []
        if args.apply:
            for g in merged:
                for p in g.dups:
                    if not p.exists():
                        continue
                    if args.delete:
                        p.unlink()
                        dst = "(deleted)"
                    else:
                        dstp = _move_to_trash(p, trash_root=trash_root, tag=tag)
                        dst = _rel(dstp)
                    total_removed += 1
                    removed.append(
                        {
                            "file": _rel(p),
                            "action": "delete" if args.delete else "moved_to_trash",
                            "dest": dst,
                            "reason": g.reason,
                            "keep": _rel(g.keep),
                        }
                    )

        report["groups"].setdefault(tag, {})
        report["groups"][tag] = {
            "total_images": len(paths),
            "dup_files_found": sum(len(g.dups) for g in merged),
            "dup_groups": [
                {
                    "reason": g.reason,
                    "key": g.key,
                    "keep": _rel(g.keep),
                    "dups": [_rel(p) for p in g.dups],
                }
                for g in merged
            ],
            "removed": removed,
        }

    report["total_dup_files_found"] = total_dups
    report["total_removed"] = total_removed

    out_json = reports_dir / "dedup_classifier_report.json"
    out_txt = reports_dir / "dedup_classifier_summary.txt"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"dedup_classifier_images.py report @ {report['ts']}",
        f"apply={report['apply']} delete={report['delete']} visual={report['visual']} max_hamming={report['max_hamming']}",
        "",
        f"total_dup_files_found: {total_dups}",
        f"total_removed: {total_removed}",
        "",
        "Per-group:",
    ]
    for tag, _paths in groups:
        g = report["groups"][tag]  # type: ignore[index]
        lines.append(
            f"  - {tag}: total_images={g['total_images']} dup_files_found={g['dup_files_found']} removed={len(g['removed'])}"
        )
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote: {out_txt}")
    print(f"Wrote: {out_json}")
    if args.apply and not args.delete:
        print(f"Moved duplicates into: {trash_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

