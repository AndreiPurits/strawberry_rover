#!/usr/bin/env python3
"""Run peduncle grasp v1 on field_photos (edited berries) → debug vis + JSON gallery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from yolo_jetson_compat import apply_torchvision_nms_patch  # noqa: E402

apply_torchvision_nms_patch()

from ultralytics import YOLO  # noqa: E402

from pipelines.peduncle_grasp import load_config  # noqa: E402
from pipelines.peduncle_grasp.calyx import resolve_calyx  # noqa: E402
from pipelines.peduncle_grasp.pipeline import PeduncleGraspPipeline  # noqa: E402
from pipelines.peduncle_grasp.roi import build_closeup_roi  # noqa: E402
from pipelines.peduncle_grasp.viz import draw_debug  # noqa: E402


def parse_berries(lab: Path, w: int, h: int):
    out = []
    if not lab.is_file():
        return out
    for raw in lab.read_text().splitlines():
        toks = raw.strip().split()
        if len(toks) < 5:
            continue
        if int(float(toks[0])) != 0:
            continue
        xc, yc, bw, bh = map(float, toks[1:5])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        out.append((x1, y1, x2, y2))
    return out


def main() -> int:
    cfg = load_config(REPO / "config" / "peduncle_grasp_v1.yaml")
    # Offline demo: still strict on READY (geometry calyx → NEED_NEW_VIEW),
    # but temporal buffer cleared per berry so VERIFY shows confirmation progress.
    src = REPO / "data" / "field_photos_primary_review"
    out = REPO / "runs" / "field_peduncle_grasp_v1"
    (out / "debug").mkdir(parents=True, exist_ok=True)
    (out / "json").mkdir(parents=True, exist_ok=True)
    (out / "side").mkdir(parents=True, exist_ok=True)

    weights = REPO / "runs" / "peduncle_obb" / "yolov8n_peduncle_v2_berry_anchored" / "weights" / "best.pt"
    model = YOLO(str(weights))
    pipe = PeduncleGraspPipeline(cfg)

    summary = []
    for im_path in sorted((src / "images").glob("*.jpg")):
        bgr = cv2.imread(str(im_path))
        h, w = bgr.shape[:2]
        berries = parse_berries(src / "labels" / f"{im_path.stem}.txt", w, h)
        before = bgr.copy()
        for bi, bb in enumerate(berries):
            cv2.rectangle(before, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 220, 0), 2)
            cv2.putText(before, f"berry#{bi}", (int(bb[0]), max(18, int(bb[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(before, "BEFORE berry", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        after = bgr.copy()
        frame_reports = []
        for bi, bb in enumerate(berries):
            pipe.verifier.reset()
            pipe._viewpoint_attempts.pop(bi, None)
            # rough mask from bbox (no segmenter required for demo)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                mask,
                (int(0.5 * (bb[0] + bb[2])), int(0.5 * (bb[1] + bb[3]))),
                (max(1, int(0.45 * (bb[2] - bb[0]))), max(1, int(0.48 * (bb[3] - bb[1])))),
                0,
                0,
                360,
                255,
                -1,
            )
            calyx = resolve_calyx(bgr, bb, mask, cfg)
            center = (0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3]))
            berry_h = max(1.0, bb[3] - bb[1])
            roi = build_closeup_roi(bgr, calyx.point_2d, berry_h, center, cfg.get("closeup_roi") or {})

            # OBB on close-up ROI only
            res = model.predict(source=roi.crop_bgr, imgsz=640, conf=0.12, device=0, verbose=False)[0]
            cands = []
            if res.obb is not None and len(res.obb):
                for poly, c in zip(res.obb.xyxyxyxy.cpu().numpy(), res.obb.conf.cpu().numpy()):
                    pts_roi = [(float(p[0]), float(p[1])) for p in poly.reshape(-1, 2)]
                    pts_full = [roi.to_full(x, y) for x, y in pts_roi]
                    cands.append((pts_full, float(c)))

            # First pass: association (single frame). If selected, replay 3x for temporal VERIFY.
            result = pipe.process_berry_frame(
                bgr,
                bb,
                track_id=bi,
                berry_mask=mask,
                detector_conf=1.0,
                obb_candidates=cands,
                push_temporal=True,
            )
            if result.berry and result.berry.selected_peduncle is not None:
                pipe.verifier.reset()
                for _ in range(3):
                    result = pipe.process_berry_frame(
                        bgr,
                        bb,
                        track_id=bi,
                        berry_mask=mask,
                        detector_conf=1.0,
                        obb_candidates=cands,
                        push_temporal=True,
                    )
            after = draw_debug(after, result, roi=roi, raw_cands=cands)
            frame_reports.append(result.to_dict())

        gap = 8
        side = np.zeros((h, w * 2 + gap, 3), dtype=np.uint8)
        side[:, :w] = before
        side[:, w + gap :] = after
        cv2.imwrite(str(out / "debug" / im_path.name), after)
        cv2.imwrite(str(out / "side" / im_path.name), side)
        (out / "json" / f"{im_path.stem}.json").write_text(
            json.dumps({"file": im_path.name, "berries": frame_reports}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        n_sel = sum(1 for r in frame_reports if r.get("berry") and r["berry"].get("selected_peduncle"))
        summary.append({"file": im_path.name, "n_berry": len(berries), "n_assoc": n_sel})
        print(im_path.name, f"assoc={n_sel}/{len(berries)}")

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OUT", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
