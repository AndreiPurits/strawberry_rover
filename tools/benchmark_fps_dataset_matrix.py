#!/usr/bin/env python3
"""Benchmark PT vs TensorRT full pipeline on ФПС ДАТАСЕТ across resolution presets."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

    apply_torchvision_nms_patch()
except Exception:
    pass

PRESETS = {
    "baseline": {"imgsz_det": 640, "imgsz_seg": 384, "conf_det": 0.35, "max_det": 20, "seg_every": 2},
    "fast": {"imgsz_det": 512, "imgsz_seg": 320, "conf_det": 0.40, "max_det": 10, "seg_every": 2},
    "very_fast": {"imgsz_det": 480, "imgsz_seg": 320, "conf_det": 0.45, "max_det": 5, "seg_every": 3},
    "ultra_low": {"imgsz_det": 416, "imgsz_seg": 256, "conf_det": 0.45, "max_det": 5, "seg_every": 2},
}

ENGINE_DIR = REPO_ROOT / "runs" / "export_tensorrt" / "group02"


def _engine_path(role: str, imgsz: int) -> Path:
    return ENGINE_DIR / f"{role}_imgsz{imgsz}.engine"


def _run_benchmark(
    *,
    holdout: Path,
    outdir: Path,
    preset: str,
    backend: str,
    model_group: str,
    limit: int,
    warmup: int,
) -> dict:
    p = PRESETS[preset]
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{backend}_{preset}"
    per_csv = outdir / f"per_image_{tag}.csv"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "benchmark_holdout_full_pipeline_fps.py"),
        "--holdout",
        str(holdout),
        "--model-group",
        model_group,
        "--outdir",
        str(outdir / tag),
        "--warmup",
        str(warmup),
        "--imgsz-det",
        str(p["imgsz_det"]),
        "--imgsz-seg",
        str(p["imgsz_seg"]),
        "--conf-det",
        str(p["conf_det"]),
        "--max-det",
        str(p["max_det"]),
        "--seg-every",
        str(p["seg_every"]),
        "--preset",
        "custom",
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    if backend == "trt":
        det_e = _engine_path("detector", int(p["imgsz_det"]))
        seg_e = _engine_path("segmenter", int(p["imgsz_seg"]))
        if not det_e.exists():
            raise FileNotFoundError(f"Missing TRT engine: {det_e}")
        if not seg_e.exists():
            raise FileNotFoundError(f"Missing TRT engine: {seg_e}")
        cmd.extend(["--detector", str(det_e), "--segmenter", str(seg_e)])

    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    summary_json = outdir / tag / "summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(str(summary_json))
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    pp = data.get("pipeline_params") or {}

    row = {
        "backend": backend,
        "preset": preset,
        "det_imgsz": int(pp.get("imgsz_det", p["imgsz_det"])),
        "seg_imgsz": int(pp.get("imgsz_seg", p["imgsz_seg"])),
        "det_conf": float(pp.get("conf_det", p["conf_det"])),
        "max_det": int(pp.get("max_det", p["max_det"])),
        "seg_every": int(pp.get("seg_every", p["seg_every"])),
        "fps": data.get("total_end_to_end_fps", data.get("fps")),
        "ms_per_img": data.get("total_ms_per_img", data.get("ms_per_img")),
        "detector_ms_per_img": data.get("detector_ms_per_img"),
        "classifier_ms_per_img": data.get("classifier_ms_per_img"),
        "segmentation_ms_per_img": data.get("segmentation_ms_per_img"),
        "avg_detections_per_img": data.get("avg_detections_per_img"),
        "avg_masks_per_img": data.get(
            "avg_segmentation_masks_per_img", data.get("avg_masks_per_img")
        ),
        "images_count": data.get("images_count"),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--holdout",
        default=str(REPO_ROOT / "data" / "ФПС ДАТАСЕТ" / "images"),
    )
    ap.add_argument("--model-group", default="02_lightened_current")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "runs" / "fps_dataset_benchmark_matrix"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--backends",
        default="pt,trt",
        help="Comma-separated: pt, trt",
    )
    ap.add_argument(
        "--presets",
        default="baseline,fast,very_fast,ultra_low",
        help="Comma-separated preset names",
    )
    args = ap.parse_args()

    holdout = Path(args.holdout)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    backends = [b.strip() for b in str(args.backends).split(",") if b.strip()]
    presets = [p.strip() for p in str(args.presets).split(",") if p.strip()]

    rows = []
    for backend in backends:
        for preset in presets:
            if preset not in PRESETS:
                print(f"SKIP unknown preset {preset}", flush=True)
                continue
            try:
                row = _run_benchmark(
                    holdout=holdout,
                    outdir=outdir,
                    preset=preset,
                    backend=backend,
                    model_group=str(args.model_group),
                    limit=int(args.limit),
                    warmup=int(args.warmup),
                )
                rows.append(row)
                print(
                    f"DONE {backend}/{preset} fps={row['fps']:.2f} ms={row['ms_per_img']:.2f}",
                    flush=True,
                )
            except Exception as e:
                print(f"FAIL {backend}/{preset}: {e}", flush=True)
                rows.append({"backend": backend, "preset": preset, "error": str(e)})

    summary_csv = outdir / "summary_matrix.csv"
    summary_json = outdir / "summary_matrix.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    if rows:
        keys = [
            "backend",
            "preset",
            "fps",
            "ms_per_img",
            "detector_ms_per_img",
            "classifier_ms_per_img",
            "segmentation_ms_per_img",
            "det_imgsz",
            "seg_imgsz",
            "avg_detections_per_img",
            "avg_masks_per_img",
            "images_count",
            "error",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    md = outdir / "summary_matrix.md"
    lines = ["# FPS dataset benchmark matrix (PT vs TRT)", ""]
    lines.append(f"- holdout: `{holdout}`")
    lines.append(f"- model_group: `{args.model_group}`")
    lines.append("")
    lines.append("| backend | preset | det | seg | FPS | ms/img | det ms | cls ms | seg ms |")
    lines.append("|---------|--------|-----|-----|-----|--------|--------|--------|--------|")
    for r in rows:
        if "error" in r:
            lines.append(f"| {r.get('backend')} | {r.get('preset')} | ERROR | | | | | | |")
            continue
        lines.append(
            f"| {r['backend']} | {r['preset']} | {r['det_imgsz']} | {r['seg_imgsz']} | "
            f"{float(r['fps']):.2f} | {float(r['ms_per_img']):.2f} | "
            f"{float(r['detector_ms_per_img']):.2f} | {float(r['classifier_ms_per_img']):.2f} | "
            f"{float(r['segmentation_ms_per_img']):.2f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_csv}", flush=True)
    print(f"Wrote {summary_json}", flush=True)
    print(f"Wrote {md}", flush=True)

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_md = reports_dir / "fps_dataset_trt_comparison.md"
    report_json = reports_dir / "fps_dataset_trt_comparison.json"
    report_md.write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
    report_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {report_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
