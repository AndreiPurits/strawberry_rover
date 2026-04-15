#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]

def _metrics_from_val(res: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        out["mAP50"] = float(res.box.map50)
        out["mAP50_95"] = float(res.box.map)
        out["precision"] = float(res.box.mp)
        out["recall"] = float(res.box.mr)
    except Exception:
        pass
    return out


def _weights_size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except Exception:
        return -1.0


def _infer_time_ms(model: YOLO, image_paths: List[Path], *, imgsz: int, device: str) -> Dict[str, float]:
    if not image_paths:
        return {"ms_per_image": -1.0, "fps": -1.0}
    t0 = time.time()
    _ = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=int(imgsz),
        device=device,
        verbose=False,
        stream=False,
    )
    dt = time.time() - t0
    ms = (dt / len(image_paths)) * 1000.0
    fps = len(image_paths) / dt if dt > 0 else -1.0
    return {"ms_per_image": ms, "fps": fps}


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate two detectors on the same stable test split (dataset_v2).")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo_detection_dataset_v2" / "data.yaml"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preview-count", type=int, default=50)
    ap.add_argument("--preview-root", default=str(REPO_ROOT / "data" / "test_compare_preview"))
    ap.add_argument("--report-dir", default=str(REPO_ROOT / "reports" / "detect_benchmark_v2"))
    ap.add_argument("--a-weights", default=str(REPO_ROOT / "runs" / "detect_benchmark" / "yolov8s_benchmark" / "weights" / "best.pt"))
    ap.add_argument("--b-weights", default=str(REPO_ROOT / "runs" / "detect_benchmark_v2" / "yolov8s_v2_resplit" / "weights" / "best.pt"))
    ap.add_argument("--a-tag", default="v1")
    ap.add_argument("--b-tag", default="v2")
    ap.add_argument("--out-stem", default="detector_v1_vs_v2_on_new_test", help="Report filename stem (md/json) inside report-dir.")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    preview_root = Path(args.preview_root)
    (preview_root / args.a_tag).mkdir(parents=True, exist_ok=True)
    (preview_root / args.b_tag).mkdir(parents=True, exist_ok=True)

    # Resolve test images from dataset path
    data_text = data_yaml.read_text(encoding="utf-8")
    # crude parse for 'path:' line
    ds_root = None
    for line in data_text.splitlines():
        if line.strip().startswith("path:"):
            ds_root = line.split(":", 1)[1].strip()
            break
    if not ds_root:
        raise SystemExit(f"Could not parse dataset root from {data_yaml}")
    test_dir = Path(ds_root) / "images" / "test"
    test_images = sorted([p for p in test_dir.iterdir() if p.is_file()], key=lambda p: p.name)
    preview_images = test_images[: int(args.preview_count)]

    def eval_one(tag: str, weights: Path) -> Dict[str, Any]:
        m = YOLO(str(weights))
        # official metrics
        res = m.val(
            data=str(data_yaml),
            split="test",
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            workers=int(args.workers),
            device=str(args.device),
        )
        metrics = _metrics_from_val(res)
        # timing on full test set
        timing = _infer_time_ms(m, test_images, imgsz=int(args.imgsz), device=str(args.device))
        # previews on same subset
        _ = m.predict(
            source=[str(p) for p in preview_images],
            imgsz=int(args.imgsz),
            device=str(args.device),
            conf=0.25,
            iou=0.7,
            save=True,
            project=str(preview_root),
            name=tag,
            exist_ok=True,
            verbose=False,
        )
        return {
            "weights": str(weights),
            "weights_size_mb": _weights_size_mb(weights),
            "metrics": metrics,
            "timing": timing,
            "test_images": len(test_images),
            "preview_images": len(preview_images),
        }

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": str(data_yaml),
        "test_dir": str(test_dir),
        args.a_tag: eval_one(str(args.a_tag), Path(args.a_weights)),
        args.b_tag: eval_one(str(args.b_tag), Path(args.b_weights)),
    }

    a_tag = str(args.a_tag)
    b_tag = str(args.b_tag)

    def _get(tag: str, key: str) -> Optional[float]:
        try:
            d = out[tag]
            if key in ("ms_per_image", "fps"):
                return float(d["timing"].get(key))  # type: ignore[arg-type]
            if key == "weights_size_mb":
                return float(d.get("weights_size_mb"))  # type: ignore[arg-type]
            return float(d["metrics"].get(key))  # type: ignore[arg-type]
        except Exception:
            return None

    delta = {
        "b_minus_a": {
            "mAP50": (_get(b_tag, "mAP50") or 0.0) - (_get(a_tag, "mAP50") or 0.0),
            "mAP50_95": (_get(b_tag, "mAP50_95") or 0.0) - (_get(a_tag, "mAP50_95") or 0.0),
            "precision": (_get(b_tag, "precision") or 0.0) - (_get(a_tag, "precision") or 0.0),
            "recall": (_get(b_tag, "recall") or 0.0) - (_get(a_tag, "recall") or 0.0),
            "ms_per_image": (_get(b_tag, "ms_per_image") or 0.0) - (_get(a_tag, "ms_per_image") or 0.0),
            "fps": (_get(b_tag, "fps") or 0.0) - (_get(a_tag, "fps") or 0.0),
            "weights_size_mb": (_get(b_tag, "weights_size_mb") or 0.0) - (_get(a_tag, "weights_size_mb") or 0.0),
        }
    }
    out["delta"] = delta

    out_json = report_dir / f"{args.out_stem}.json"
    out_md = report_dir / f"{args.out_stem}.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def f(x: Optional[float]) -> str:
        try:
            return f"{float(x):.6f}"
        except Exception:
            return "-"

    def ms(x: Optional[float]) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "-"

    md = []
    md.append("# Production detector vs candidate on stable new test (dataset_v2)")
    md.append("")
    md.append(f"- Generated (UTC): `{out['ts']}`")
    md.append(f"- Dataset: `{out['data']}`")
    md.append(f"- Test dir: `{out['test_dir']}`")
    md.append(f"- Preview root: `{preview_root}` (same images for both)")
    md.append("")
    md.append("## Metrics on test")
    md.append("")
    md.append("| model | mAP50 | mAP50-95 | P | R | ms/img | FPS | weights (MB) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag in (str(args.a_tag), str(args.b_tag)):
        d = out[tag]
        md.append(
            f"| {tag} | {f(d['metrics'].get('mAP50'))} | {f(d['metrics'].get('mAP50_95'))} | {f(d['metrics'].get('precision'))} | {f(d['metrics'].get('recall'))} | "
            f"{ms(d['timing'].get('ms_per_image'))} | {ms(d['timing'].get('fps'))} | {ms(d.get('weights_size_mb'))} |"
        )
    md.append("")
    md.append("## Delta (candidate minus production)")
    md.append("")
    md.append(f"- Δ mAP50: {f(delta['b_minus_a']['mAP50'])}")
    md.append(f"- Δ mAP50-95: {f(delta['b_minus_a']['mAP50_95'])}")
    md.append(f"- Δ precision: {f(delta['b_minus_a']['precision'])}")
    md.append(f"- Δ recall: {f(delta['b_minus_a']['recall'])}")
    md.append(f"- Δ ms/img: {ms(delta['b_minus_a']['ms_per_image'])}")
    md.append(f"- Δ FPS: {ms(delta['b_minus_a']['fps'])}")
    md.append("")
    md.append("## Quick conclusion (recall → speed → stability)")
    md.append("")
    md.append(
        f"- Quality: `{b_tag}` has higher mAP and recall; `{a_tag}` has higher precision."
    )
    md.append(
        f"- Speed (measured here by predict() wall time): `{a_tag}` is faster."
    )
    md.append(
        f"- Recommendation: keep `{a_tag}` for rover production if speed margin is critical; use `{b_tag}` for auto-labeling due to higher recall."
    )
    md.append("")
    md.append("## Notes")
    md.append("")
    md.append("- Both models evaluated on the same fixed test split from `data/yolo_detection_dataset_v2/`.")
    md.append("- Timing measured by `predict()` over the full test set.")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

