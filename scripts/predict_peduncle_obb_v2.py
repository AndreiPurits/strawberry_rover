#!/usr/bin/env python3
"""Predict peduncle OBB with berry geometry gate (AABB or mask-tip calyx) + cut vis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from peduncle_berry_geometry import (
    ASSOC_TAU,
    CUT_ALPHA,
    BerryBox,
    associate_peduncle,
    parse_berry_yolo_aabb,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from yolo_jetson_compat import apply_torchvision_nms_patch
except ImportError:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch  # type: ignore


def load_berry(stem: str, data_root: Path, split: str, w: int, h: int) -> Optional[BerryBox]:
    meta_p = data_root / "meta" / split / f"{stem}.json"
    if meta_p.is_file():
        d = json.loads(meta_p.read_text(encoding="utf-8"))
        xy = d.get("berry_xyxy")
        cal = d.get("calyx_xy")
        if isinstance(xy, list) and len(xy) == 4:
            calyx = tuple(map(float, cal)) if isinstance(cal, list) and len(cal) == 2 else None
            return BerryBox(
                float(xy[0]),
                float(xy[1]),
                float(xy[2]),
                float(xy[3]),
                calyx_xy=calyx,
                source=str(d.get("berry_source") or "meta"),
            )
    lab = data_root / "labels_berry" / split / f"{stem}.txt"
    if lab.is_file():
        return parse_berry_yolo_aabb(lab, w, h)
    return None


def draw_vis(
    bgr: np.ndarray,
    berry: BerryBox,
    assoc,
    raw_cands: List[Tuple[List[Tuple[float, float]], float]],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = bgr.copy()
    if mask is not None and mask.shape[:2] == vis.shape[:2]:
        overlay = vis.copy()
        overlay[mask > 0] = (0.6 * overlay[mask > 0] + np.array([0, 100, 0])).astype(np.uint8)
        vis = overlay
        cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 200, 80), 1)

    cv2.rectangle(
        vis,
        (int(berry.x1), int(berry.y1)),
        (int(berry.x2), int(berry.y2)),
        (0, 220, 0),
        2,
    )
    cx, cy = berry.calyx
    cv2.circle(vis, (int(cx), int(cy)), 6, (0, 255, 255), -1)
    cv2.putText(
        vis,
        f"calyx/{berry.source}",
        (int(cx) + 6, int(cy) - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
    )

    for pts, _conf in raw_cands:
        poly = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)
        cv2.polylines(vis, [poly], True, (70, 70, 70), 1)

    if assoc is not None:
        poly = np.array([[int(round(x)), int(round(y))] for x, y in assoc.points], dtype=np.int32)
        cv2.polylines(vis, [poly], True, (0, 140, 255), 2)
        cv2.circle(vis, (int(assoc.calyx_end[0]), int(assoc.calyx_end[1])), 4, (255, 180, 0), -1)
        cv2.circle(vis, (int(assoc.tip[0]), int(assoc.tip[1])), 4, (255, 0, 180), -1)
        cv2.circle(vis, (int(assoc.cut_point[0]), int(assoc.cut_point[1])), 6, (0, 0, 255), -1)
        cv2.line(
            vis,
            (int(assoc.calyx_end[0]), int(assoc.calyx_end[1])),
            (int(assoc.tip[0]), int(assoc.tip[1])),
            (0, 0, 255),
            1,
        )
        cv2.putText(
            vis,
            f"ped {assoc.conf:.2f} score={assoc.score:.2f}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"cut d/s={assoc.dist_norm:.2f}",
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(vis, "NO ASSOC", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def main() -> int:
    apply_torchvision_nms_patch()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--weights",
        default=str(
            REPO_ROOT
            / "runs"
            / "peduncle_obb"
            / "yolov8n_peduncle_v21_mask_calyx"
            / "weights"
            / "best.pt"
        ),
    )
    ap.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "peduncle_obb_berry_anchored_v21"),
    )
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "runs" / "peduncle_obb" / "yolov8n_peduncle_v2_predict_val"),
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tau", type=float, default=ASSOC_TAU)
    ap.add_argument("--cut-alpha", type=float, default=CUT_ALPHA)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    src = data_root / "images" / args.split
    if not src.is_dir():
        raise SystemExit(f"missing images: {src}")
    wpath = Path(args.weights)
    if not wpath.is_file():
        raise SystemExit(f"missing weights: {wpath}")

    out = Path(args.out)
    if out.exists():
        for old in out.glob("*.jpg"):
            old.unlink()
        for old in out.glob("*.png"):
            old.unlink()
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(wpath))
    images = sorted([p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    n_ok = n_miss = 0
    report = []

    for im_path in images:
        bgr = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        berry = load_berry(im_path.stem, data_root, args.split, w, h)
        if berry is None:
            n_miss += 1
            continue
        mask_p = data_root / "masks" / args.split / f"{im_path.stem}.png"
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) if mask_p.is_file() else None

        results = model.predict(
            source=bgr,
            imgsz=int(args.imgsz),
            conf=float(args.conf),
            device=str(args.device),
            verbose=False,
        )
        cands: List[Tuple[List[Tuple[float, float]], float]] = []
        if results and results[0].obb is not None and len(results[0].obb):
            xy = results[0].obb.xyxyxyxy.cpu().numpy()
            confs = results[0].obb.conf.cpu().numpy()
            for poly, c in zip(xy, confs):
                pts = [(float(p[0]), float(p[1])) for p in poly.reshape(-1, 2)]
                cands.append((pts, float(c)))

        assoc = associate_peduncle(cands, berry, tau=float(args.tau), cut_alpha=float(args.cut_alpha))
        vis = draw_vis(bgr, berry, assoc, cands, mask=mask)
        cv2.imwrite(str(out / im_path.name), vis)
        if assoc is not None:
            n_ok += 1
            report.append(
                {
                    "file": im_path.name,
                    "ok": True,
                    "conf": assoc.conf,
                    "score": assoc.score,
                    "berry_source": berry.source,
                }
            )
        else:
            n_miss += 1
            report.append(
                {
                    "file": im_path.name,
                    "ok": False,
                    "n_raw": len(cands),
                    "berry_source": berry.source,
                }
            )

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(images)} → {out}")
    print(f"associated={n_ok} miss={n_miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
