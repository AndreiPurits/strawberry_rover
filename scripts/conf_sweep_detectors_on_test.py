#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _infer_time_ms(
    model: YOLO,
    image_paths: List[Path],
    *,
    imgsz: int,
    device: str,
    conf: float,
    iou: float,
) -> Dict[str, float]:
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


def _fmt(x: Optional[float], nd: int = 6) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def _fmt_ms(x: Optional[float]) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"


def _eval_one_conf(
    *,
    tag: str,
    weights: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    test_images: List[Path],
    conf: float,
    iou: float,
) -> Dict[str, Any]:
    m = YOLO(str(weights))
    res = m.val(
        data=str(data_yaml),
        split="test",
        imgsz=int(imgsz),
        batch=int(batch),
        workers=int(workers),
        device=str(device),
        conf=float(conf),
        iou=float(iou),
    )
    metrics = _metrics_from_val(res)
    timing = _infer_time_ms(m, test_images, imgsz=imgsz, device=str(device), conf=float(conf), iou=float(iou))
    return {
        "model": tag,
        "weights": str(weights),
        "weights_size_mb": _weights_size_mb(weights),
        "conf": float(conf),
        "iou": float(iou),
        "metrics": metrics,
        "timing": timing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Conf sweep (P/R/latency tradeoff) for two detectors on stable test.")
    ap.add_argument("--data", default=str(REPO_ROOT / "data" / "yolo_detection_dataset_v2" / "data.yaml"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--confs", default="0.25,0.35,0.45")
    ap.add_argument("--a-tag", default="yolov8n_prod")
    ap.add_argument("--a-weights", default=str(REPO_ROOT / "runs" / "detect_benchmark" / "yolov8n_benchmark" / "weights" / "best.pt"))
    ap.add_argument("--b-tag", default="yolov8s_v2")
    ap.add_argument("--b-weights", default=str(REPO_ROOT / "runs" / "detect_benchmark_v2" / "yolov8s_v2_resplit_fastnms" / "weights" / "best.pt"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "reports" / "detect_benchmark_v2" / "conf_sweep.md"))
    args = ap.parse_args()

    data_yaml = Path(args.data)
    ds_root = _parse_dataset_root(data_yaml)
    test_dir = ds_root / "images" / "test"
    test_images = sorted([p for p in test_dir.iterdir() if p.is_file()], key=lambda p: p.name)

    confs: List[float] = []
    for s in str(args.confs).split(","):
        s = s.strip()
        if not s:
            continue
        confs.append(float(s))
    if not confs:
        raise SystemExit("No conf values provided")

    a_tag = str(args.a_tag)
    b_tag = str(args.b_tag)
    a_weights = Path(args.a_weights)
    b_weights = Path(args.b_weights)

    rows: List[Dict[str, Any]] = []
    for conf in confs:
        rows.append(
            _eval_one_conf(
                tag=a_tag,
                weights=a_weights,
                data_yaml=data_yaml,
                imgsz=int(args.imgsz),
                batch=int(args.batch),
                workers=int(args.workers),
                device=str(args.device),
                test_images=test_images,
                conf=float(conf),
                iou=float(args.iou),
            )
        )
        rows.append(
            _eval_one_conf(
                tag=b_tag,
                weights=b_weights,
                data_yaml=data_yaml,
                imgsz=int(args.imgsz),
                batch=int(args.batch),
                workers=int(args.workers),
                device=str(args.device),
                test_images=test_images,
                conf=float(conf),
                iou=float(args.iou),
            )
        )

    # Build markdown report
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md: List[str] = []
    md.append("## Conf sweep on stable test (dataset_v2/test)")
    md.append("")
    md.append(f"- Generated (UTC): `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`")
    md.append(f"- Dataset: `{data_yaml}`")
    md.append(f"- Test dir: `{test_dir}`")
    md.append(f"- Images: `{len(test_images)}`")
    md.append(f"- IOU: `{float(args.iou):.2f}`")
    md.append("")

    md.append("| model | conf | mAP50 | mAP50-95 | P | R | ms/img | FPS | weights (MB) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        m = r["metrics"]
        t = r["timing"]
        md.append(
            f"| {r['model']} | {float(r['conf']):.2f} | {_fmt(m.get('mAP50'))} | {_fmt(m.get('mAP50_95'))} | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
            f"{_fmt_ms(t.get('ms_per_image'))} | {_fmt_ms(t.get('fps'))} | {_fmt_ms(r.get('weights_size_mb'))} |"
        )

    md.append("")
    md.append("Notes:")
    md.append("- This uses `ultralytics.YOLO.val(conf=...)` on `split=test` plus wall-time `predict()` timing at the same `conf`.")
    md.append("- mAP is generally less sensitive to a single conf threshold than P/R and deployment behavior; use P/R + latency for threshold selection.")
    md.append("")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

