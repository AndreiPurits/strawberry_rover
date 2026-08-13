#!/usr/bin/env python3
"""Build ripe close-up peduncle dataset from multiple canonical sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_peduncle_labeling_queue import (  # noqa: E402
    build_from_coco,
    build_from_precropped,
    build_queue,
)

COMMON = dict(
    layout="dataset",
    out=str(REPO_ROOT / "data" / "ripe_peduncle_closeup_dataset"),
    distance_min_cm=11.0,
    distance_max_cm=16.0,
    berry_diameter_mm=30.0,
    focal_px=0.0,
    focal_scale=0.95,
    pad_top_frac=0.85,
    pad_bottom_frac=0.15,
    pad_side_frac=0.20,
    min_crop_px=120,
    min_berry_px=120.0,
    min_berry_frac=0.22,
    min_sharpness=80.0,
    min_red_ratio=0.18,
    top_search_frac=0.90,
    min_green_ratio=0.06,
    min_vertical_span_frac=0.18,
    min_top_margin_frac=0.02,
    one_per_source=True,
    skip_existing=True,
    max_noise_std=13.0,
    preview_count=0,
    jpeg_quality=95,
)

SOURCES = [
    {
        "name": "luke",
        "src": REPO_ROOT / "data" / "strawberries",
        "src_tag": "luke",
        "ripeness_classes": "0",
        "peduncle_class": 2,
        "require_annotated_peduncle": True,
    },
    {
        "name": "fps",
        "src": REPO_ROOT / "data" / "ФПС ДАТАСЕТ",
        "src_tag": "fps",
        "ripeness_classes": "2",
    },
    {
        "name": "detection",
        "src": REPO_ROOT / "data" / "final_detection_dataset",
        "src_tag": "det",
        "ripeness_classes": "2",
    },
    {
        "name": "yolo_seg",
        "src": REPO_ROOT / "data" / "yolo_segmentation_dataset",
        "src_tag": "yseg",
        "ripeness_classes": "",
    },
    {
        "name": "det_v2",
        "src": REPO_ROOT / "data" / "yolo_detection_dataset_v2",
        "src_tag": "dv2",
        "ripeness_classes": "",
    },
    {
        "name": "det_v3",
        "src": REPO_ROOT / "data" / "yolo_detection_dataset_v3",
        "src_tag": "dv3",
        "ripeness_classes": "",
    },
    {
        "name": "new_photos",
        "src": REPO_ROOT / "data" / "new_photos_labeled",
        "src_tag": "np",
        "ripeness_classes": "2",
    },
    {
        "name": "roboflow",
        "src": REPO_ROOT / "data" / "roboflow_downloads",
        "src_tag": "rf",
        "layout": "coco",
        "ripeness_classes": "1",
    },
    {
        "name": "seg_project",
        "src": REPO_ROOT / "data" / "segmentation_project_dataset",
        "src_tag": "seg",
        "layout": "coco",
        "ripeness_classes": "0",
    },
    {
        "name": "peduncle_queue",
        "src": REPO_ROOT / "data" / "peduncle_labeling_queue",
        "src_tag": "pq",
        "ripeness_classes": "2",
        "layout": "precropped",
    },
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build ripe peduncle dataset from multiple sources.")
    ap.add_argument("--only", default="", help="Comma-separated source names to run (default: all).")
    ap.add_argument("--limit", type=int, default=0, help="Per-source image limit (0=all).")
    ap.add_argument("--reprocess", action="store_true", help="Overwrite existing crops (wider distance re-filter).")
    args = ap.parse_args()

    only = {x.strip() for x in str(args.only).split(",") if x.strip()}
    reports = []
    total_kept = 0

    for spec in SOURCES:
        if only and spec["name"] not in only:
            continue
        if not Path(spec["src"]).exists():
            print(f"skip missing source {spec['name']}: {spec['src']}", flush=True)
            continue

        extra = {k: v for k, v in spec.items() if k not in ("name", "src", "src_tag", "layout")}
        ns_kwargs = {k: v for k, v in COMMON.items() if k != "layout"}
        ns_kwargs.update(
            layout=str(spec.get("layout", "dataset")),
            src=str(spec["src"]),
            src_tag=str(spec["src_tag"]),
            ripeness_classes=str(extra.get("ripeness_classes", "")),
            peduncle_class=int(extra.get("peduncle_class", -1)),
            require_annotated_peduncle=bool(extra.get("require_annotated_peduncle", False)),
            limit=int(args.limit),
            overwrite=bool(args.reprocess),
            skip_existing=not bool(args.reprocess),
        )
        ns = argparse.Namespace(**ns_kwargs)
        print(f"=== source {spec['name']} -> {spec['src']}", flush=True)
        if ns.layout == "precropped":
            stats = build_from_precropped(ns)
        elif ns.layout == "coco":
            stats = build_from_coco(ns)
        else:
            stats = build_queue(ns)
        row = {"source": spec["name"], "stats": stats.__dict__}
        reports.append(row)
        total_kept += int(stats.kept)
        print(json.dumps({"source": spec["name"], "kept": int(stats.kept)}, indent=2), flush=True)

    out_report = REPO_ROOT / "data" / "ripe_peduncle_closeup_dataset" / "reports" / "multi_source_build.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps(
            {
                "distance_cm": [COMMON["distance_min_cm"], COMMON["distance_max_cm"]],
                "sources": reports,
                "total_kept_this_run": total_kept,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"total_kept_this_run": total_kept, "sources": len(reports)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
