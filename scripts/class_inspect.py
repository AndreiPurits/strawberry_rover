#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iter_files(dirpath: Path) -> Iterable[Path]:
    if not dirpath.exists():
        return []
    return (p for p in dirpath.iterdir() if p.is_file())


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _parse_yolo_labels(label_path: Path) -> List[int]:
    class_ids: List[int] = []
    for raw_line in _read_text(label_path).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        toks = line.split()
        # Accept both YOLO bbox (5 toks) and segmentation polygon (>5 toks).
        try:
            cid = int(float(toks[0]))
        except Exception:
            continue
        class_ids.append(cid)
    return class_ids


def _parse_roboflow_names_from_data_yaml(text: str) -> Optional[Dict[int, str]]:
    # Very small YAML parser for the typical Roboflow export:
    # names: ['a', 'b', ...]
    m = re.search(r"^\s*names\s*:\s*\[(.*)\]\s*$", text, flags=re.MULTILINE)
    if not m:
        return None
    inner = m.group(1).strip()
    if not inner:
        return {}
    # Split on commas not inside quotes (simple case: single quotes).
    parts = [p.strip() for p in inner.split(",")]
    names: List[str] = []
    for p in parts:
        if (p.startswith("'") and p.endswith("'")) or (p.startswith('"') and p.endswith('"')):
            names.append(p[1:-1])
        else:
            names.append(p)
    return {i: n for i, n in enumerate(names)}


@dataclass
class ClassStats:
    class_id: int
    count_objects: int = 0
    image_stems: Set[str] = field(default_factory=set)


@dataclass
class DatasetClassInspection:
    dataset: str
    base_dir: Path
    images_dir: Path
    labels_dir: Path
    bbox_vis_dir: Path
    class_stats: Dict[int, ClassStats] = field(default_factory=dict)
    class_name_map: Dict[int, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def ensure_class(self, class_id: int) -> ClassStats:
        if class_id not in self.class_stats:
            self.class_stats[class_id] = ClassStats(class_id=class_id)
        return self.class_stats[class_id]


def inspect_dataset(normalized_root: Path, dataset: str) -> DatasetClassInspection:
    base = normalized_root / dataset
    insp = DatasetClassInspection(
        dataset=dataset,
        base_dir=base,
        images_dir=base / "images",
        labels_dir=base / "labels",
        bbox_vis_dir=base / "bbox_vis",
    )

    label_files = sorted(p for p in _iter_files(insp.labels_dir) if p.suffix.lower() == ".txt")
    if not label_files:
        insp.notes.append("No labels found in normalized labels/.")
        return insp

    for lf in label_files:
        stem = lf.stem
        cids = _parse_yolo_labels(lf)
        for cid in cids:
            cs = insp.ensure_class(cid)
            cs.count_objects += 1
            cs.image_stems.add(stem)

    return insp


def find_metadata(repo_root: Path, dataset: str) -> Dict[int, str]:
    # Try a few common locations.
    raw_root = repo_root / "data" / "raw"
    normalized_root = repo_root / "data" / "normalized"

    candidates: List[Path] = []
    candidates += list((raw_root / dataset).rglob("data.yaml"))
    candidates += list((raw_root / dataset).rglob("data.yml"))
    candidates += list((normalized_root / dataset).rglob("data.yaml"))
    candidates += list((normalized_root / dataset).rglob("classes.txt"))

    # Special-case: strawberry_ds has Roboflow nested folder.
    if dataset == "strawberry_ds":
        candidates += list((raw_root / dataset).rglob("README.roboflow.txt"))

    for p in candidates:
        try:
            txt = _read_text(p)
        except Exception:
            continue
        if p.name in ("data.yaml", "data.yml"):
            m = _parse_roboflow_names_from_data_yaml(txt)
            if m is not None and len(m) > 0:
                return m
        if p.name == "classes.txt":
            names = [line.strip() for line in txt.splitlines() if line.strip()]
            if names:
                return {i: n for i, n in enumerate(names)}

    return {}


def copy_debug_examples(
    insp: DatasetClassInspection,
    out_root: Path,
    per_class_min: int,
    per_class_max: int,
    seed: int,
) -> None:
    rnd = random.Random(seed)
    ds_out = out_root / insp.dataset
    _safe_mkdir(ds_out)

    for cid in sorted(insp.class_stats.keys()):
        cs = insp.class_stats[cid]
        stems = sorted(cs.image_stems)
        if not stems:
            continue
        k = min(per_class_max, max(per_class_min, min(per_class_max, len(stems))))
        chosen = stems if len(stems) <= k else rnd.sample(stems, k)
        out_dir = ds_out / f"class_{cid}"
        _safe_mkdir(out_dir)
        for stem in chosen:
            # Prefer bbox_vis; fallback to images if missing.
            src = None
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                cand = insp.bbox_vis_dir / f"{stem}{ext}"
                if cand.exists():
                    src = cand
                    break
            if src is None:
                for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    cand = insp.images_dir / f"{stem}{ext}"
                    if cand.exists():
                        src = cand
                        break
            if src is None:
                insp.notes.append(f"Missing bbox_vis/images for stem '{stem}' (class {cid})")
                continue
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def write_reports(
    repo_root: Path,
    inspections: Sequence[DatasetClassInspection],
    out_txt: Path,
    out_md: Path,
) -> None:
    lines: List[str] = []
    md: List[str] = []

    lines.append("CLASS INSPECTION REPORT")
    lines.append("")

    md.append("## Class inspection summary")
    md.append("")
    md.append(f"- **Debug images folder**: `{repo_root / 'data' / 'debug_class_inspection'}`")
    md.append(f"- **Full report**: `{out_txt}`")
    md.append("")

    for insp in inspections:
        lines.append(f"=== {insp.dataset} ===")
        md.append(f"### `{insp.dataset}`")

        uniq = len(insp.class_stats)
        total_objs = sum(cs.count_objects for cs in insp.class_stats.values())
        total_imgs_with_any = len(set().union(*(cs.image_stems for cs in insp.class_stats.values()))) if insp.class_stats else 0

        lines.append(f"unique_classes: {uniq}")
        lines.append(f"total_objects: {total_objs}")
        lines.append(f"images_with_any_label: {total_imgs_with_any}")
        lines.append("")

        md.append(f"- **Unique classes**: {uniq}")
        md.append(f"- **Total objects**: {total_objs}")
        md.append(f"- **Images with any label**: {total_imgs_with_any}")

        # Class detail
        for cid in sorted(insp.class_stats.keys()):
            cs = insp.class_stats[cid]
            name = insp.class_name_map.get(cid)
            guessed = name if name is not None else "визуально проверить вручную (confidence: low)"

            lines.append(f"class {cid}:")
            lines.append(f"  count_objects: {cs.count_objects}")
            lines.append(f"  count_images: {len(cs.image_stems)}")
            if name is not None:
                lines.append(f"  class_name_from_metadata: {name}")
                lines.append(f"  guessed_meaning: {name} (metadata)")
            else:
                lines.append("  class_name_from_metadata: (not found)")
                lines.append(f"  guessed_meaning: {guessed}")
            lines.append("")

        if insp.notes:
            lines.append("notes:")
            for n in insp.notes[:200]:
                lines.append(f"  - {n}")
            if len(insp.notes) > 200:
                lines.append(f"  - ... truncated ({len(insp.notes) - 200} more)")
            lines.append("")

        # High-level “possible issues” section (non-visual)
        md.append("")
        if insp.class_name_map:
            md.append("- **Class mapping**: найдено из metadata.")
        else:
            md.append("- **Class mapping**: metadata не найдено, нужен ручной просмотр `debug_class_inspection/`.")
        md.append("")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    out_md.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect class_ids used in normalized datasets and prepare debug examples.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--normalized-root", type=Path, default=None)
    ap.add_argument("--datasets", nargs="*", default=["strawberry_ds", "strawberry_turkey"])
    ap.add_argument("--out-debug", type=Path, default=None)
    ap.add_argument("--per-class-min", type=int, default=10)
    ap.add_argument("--per-class-max", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(list(argv) if argv is not None else None)

    repo_root: Path = args.repo_root.resolve()
    normalized_root = (args.normalized_root or (repo_root / "data" / "normalized")).resolve()
    out_debug = (args.out_debug or (repo_root / "data" / "debug_class_inspection")).resolve()

    inspections: List[DatasetClassInspection] = []
    for ds in args.datasets:
        insp = inspect_dataset(normalized_root, ds)
        insp.class_name_map = find_metadata(repo_root, ds)
        inspections.append(insp)
        copy_debug_examples(
            insp=insp,
            out_root=out_debug,
            per_class_min=args.per_class_min,
            per_class_max=args.per_class_max,
            seed=args.seed,
        )

    out_txt = repo_root / "data" / "class_inspection_report.txt"
    out_md = repo_root / "docs" / "class_inspection_summary.md"
    write_reports(repo_root, inspections, out_txt=out_txt, out_md=out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

