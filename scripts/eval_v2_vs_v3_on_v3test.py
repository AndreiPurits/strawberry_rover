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


def _infer_time_ms(model: YOLO, image_paths: List[Path], *, imgsz: int, device: str, conf: float, iou: float) -> Dict[str, float]:
    if not image_paths:
        return {"ms_per_image": -1.0, "fps": -1.0}
    t0 = time.time()
    _ = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=int(imgsz),
        device=device,
        conf=float(conf),
        iou=float(iou),
        verbose=False,
        stream=False,
    )
    dt = time.time() - t0
    ms = (dt / len(image_paths)) * 1000.0
    fps = len(image_paths) / dt if dt > 0 else -1.0
    return {"ms_per_image": ms, "fps": fps}


def _parse_dataset_root(data_yaml: Path) -> Path:
    txt = data_yaml.read_text(encoding="utf-8")
    for line in txt.splitlines():
        if line.strip().startswith("path:"):
            p = line.split(":", 1)[1].strip()
            if p:
                return Path(p)
    raise SystemExit(f"Could not parse dataset root from {data_yaml}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate v2 vs v3 on v3 stable test split with previews and report.")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo_detection_dataset_v3" / "data.yaml"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--preview-count", type=int, default=80)
    ap.add_argument("--preview-root", default=str(REPO_ROOT / "data" / "test_compare_preview"))
    ap.add_argument("--report-dir", default=str(REPO_ROOT / "reports" / "detect_benchmark_v3"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument(
        "--v2-weights",
        default=str(REPO_ROOT / "runs" / "detect_benchmark_v2" / "yolov8s_v2_resplit_fastnms" / "weights" / "best.pt"),
    )
    ap.add_argument(
        "--v3-weights",
        default=str(REPO_ROOT / "runs" / "detect_benchmark_v3" / "yolov8s_v3_lowdensity" / "weights" / "best.pt"),
    )
    args = ap.parse_args()

    data_yaml = Path(args.data)
    ds_root = _parse_dataset_root(data_yaml)
    test_dir = ds_root / "images" / "test"
    test_images = sorted([p for p in test_dir.iterdir() if p.is_file()], key=lambda p: p.name)
    preview_images = test_images[: int(args.preview_count)]

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    preview_root = Path(args.preview_root)
    (preview_root / "v2_on_v3test").mkdir(parents=True, exist_ok=True)
    (preview_root / "v3").mkdir(parents=True, exist_ok=True)

    v2_w = Path(args.v2_weights)
    v3_w = Path(args.v3_weights)
    if not v2_w.is_file():
        raise SystemExit(f"Missing v2 weights: {v2_w}")
    if not v3_w.is_file():
        raise SystemExit(f"Missing v3 weights: {v3_w} (train v3 first)")

    def eval_one(tag: str, weights: Path) -> Dict[str, Any]:
        m = YOLO(str(weights))
        res = m.val(
            data=str(data_yaml),
            split="test",
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            workers=int(args.workers),
            device=str(args.device),
            conf=float(args.conf),
            iou=float(args.iou),
        )
        metrics = _metrics_from_val(res)
        timing = _infer_time_ms(m, test_images, imgsz=int(args.imgsz), device=str(args.device), conf=float(args.conf), iou=float(args.iou))
        _ = m.predict(
            source=[str(p) for p in preview_images],
            imgsz=int(args.imgsz),
            device=str(args.device),
            conf=float(args.conf),
            iou=float(args.iou),
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
        "conf": float(args.conf),
        "iou": float(args.iou),
        "v2_on_v3test": eval_one("v2_on_v3test", v2_w),
        "v3": eval_one("v3", v3_w),
    }

    # deltas: v3 - v2
    def g(tag: str, key: str) -> float:
        d = out[tag]
        if key in ("ms_per_image", "fps"):
            return float(d["timing"][key])
        if key == "weights_size_mb":
            return float(d["weights_size_mb"])
        return float(d["metrics"][key])

    out["delta_v3_minus_v2"] = {
        "mAP50": g("v3", "mAP50") - g("v2_on_v3test", "mAP50"),
        "mAP50_95": g("v3", "mAP50_95") - g("v2_on_v3test", "mAP50_95"),
        "precision": g("v3", "precision") - g("v2_on_v3test", "precision"),
        "recall": g("v3", "recall") - g("v2_on_v3test", "recall"),
        "ms_per_image": g("v3", "ms_per_image") - g("v2_on_v3test", "ms_per_image"),
        "fps": g("v3", "fps") - g("v2_on_v3test", "fps"),
        "weights_size_mb": g("v3", "weights_size_mb") - g("v2_on_v3test", "weights_size_mb"),
    }

    out_json = report_dir / "detector_v2_vs_v3_on_v3test.json"
    out_md = report_dir / "detector_v2_vs_v3_on_v3test.md"
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

    dlt = out["delta_v3_minus_v2"]
    md: List[str] = []
    md.append("# Detector v2 vs v3 on v3 stable test")
    md.append("")
    md.append(f"- Generated (UTC): `{out['ts']}`")
    md.append(f"- Dataset: `{out['data']}`")
    md.append(f"- Test dir: `{out['test_dir']}`")
    md.append(f"- conf: `{float(args.conf):.2f}`, iou: `{float(args.iou):.2f}`")
    md.append(f"- Preview root: `{preview_root}` (same images for both)")
    md.append("")
    md.append("## Metrics on v3 test")
    md.append("")
    md.append("| model | mAP50 | mAP50-95 | P | R | ms/img | FPS | weights (MB) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag in ("v2_on_v3test", "v3"):
        d = out[tag]
        md.append(
            f"| {tag} | {f(d['metrics'].get('mAP50'))} | {f(d['metrics'].get('mAP50_95'))} | {f(d['metrics'].get('precision'))} | {f(d['metrics'].get('recall'))} | "
            f"{ms(d['timing'].get('ms_per_image'))} | {ms(d['timing'].get('fps'))} | {ms(d.get('weights_size_mb'))} |"
        )
    md.append("")
    md.append("## Delta (v3 minus v2)")
    md.append("")
    md.append(f"- Δ mAP50: {f(dlt.get('mAP50'))}")
    md.append(f"- Δ mAP50-95: {f(dlt.get('mAP50_95'))}")
    md.append(f"- Δ precision: {f(dlt.get('precision'))}")
    md.append(f"- Δ recall: {f(dlt.get('recall'))}")
    md.append(f"- Δ ms/img: {ms(dlt.get('ms_per_image'))}")
    md.append(f"- Δ FPS: {ms(dlt.get('fps'))}")
    md.append("")
    md.append("## Questions answered")
    md.append("")
    md.append("- Did low-density hypothesis help? (judge by v3 vs v2 on v3 test): **see deltas above**.")
    md.append("- Better recall? **Δ recall**")
    md.append("- Better mAP50-95? **Δ mAP50-95**")
    md.append("- Slower? **Δ ms/img / Δ FPS**")
    md.append("- Replace auto-label v2? Depends on recall gain vs speed/precision trade-off.")
    md.append("- Use v3 as production candidate? Depends on speed vs current production requirements.")
    md.append("")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

