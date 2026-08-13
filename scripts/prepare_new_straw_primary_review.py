#!/usr/bin/env python3
"""Primary review pack for internet strawberry photos (new straw).

1) Detect berries with production YOLO detector.
2) Estimate which side of the berry has the peduncle (relative to berry, not frame-up).
3) Build future asymmetric crop bbox with extra pad on the peduncle side.
4) Write full-frame YOLO labels:
     class 0 = berry
     class 1 = future_crop
   plus preview overlays for human re-labeling.

Domain-gap note (high-quality web photos vs field cams):
- Tag every sample with source=new_straw_web and keep a separate review folder.
- Do NOT mix into production detector/classifier training without a held-out split
  and explicit downscale/aug policy — high-res sharpness/noise distribution differs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

    apply_torchvision_nms_patch()
except Exception:
    try:
        from yolo_jetson_compat import apply_torchvision_nms_patch

        apply_torchvision_nms_patch()
    except Exception:
        pass

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".tif", ".tiff"}

CLASS_BERRY = 0
CLASS_FUTURE_CROP = 1
SIDES = ("top", "bottom", "left", "right")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _slug(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-а-яА-ЯёЁ]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem[:80] or "img"


def _read_bgr(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.array(im)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = (x1 + x2) / 2.0 / float(w)
    yc = (y1 + y2) / 2.0 / float(h)
    return xc, yc, bw / float(w), bh / float(h)


def _green_ratio(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 40, 35), (95, 255, 255))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_and(mask, cv2.inRange(gray, 20, 255))
    # thin-ish vertical/horizontal structure preference via morphology
    k = max(3, min(bgr.shape[0], bgr.shape[1]) // 20)
    if k % 2 == 0:
        k += 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(np.count_nonzero(mask)) / float(mask.size)


def _green_mask(bgr: np.ndarray) -> np.ndarray:
    if bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (28, 35, 30), (100, 255, 255))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_and(mask, cv2.inRange(gray, 18, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _strip_score(frame: np.ndarray, xyxy: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = xyxy
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = frame[y1:y2, x1:x2]
    mask = _green_mask(roi)
    if mask.size == 0:
        return 0.0
    density = float(np.count_nonzero(mask)) / float(mask.size)
    # reward elongated stem-like structure along the outward axis (larger dim of strip)
    ys, xs = np.where(mask > 0)
    if ys.size < 8:
        return 0.4 * density
    span_y = float(ys.max() - ys.min() + 1) / float(max(1, mask.shape[0]))
    span_x = float(xs.max() - xs.min() + 1) / float(max(1, mask.shape[1]))
    elongation = max(span_y, span_x)
    return 0.55 * density + 0.45 * density * elongation


def estimate_peduncle_side(
    frame: np.ndarray,
    berry_xyxy: Tuple[int, int, int, int],
    *,
    search_frac: float = 1.15,
) -> Tuple[str, Dict[str, float]]:
    """Estimate which side the peduncle exits the berry.

    Uses tip vs calyx: the tip end of a ripe berry is redder; calyx/stem end is greener.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = berry_xyxy
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    berry = frame[y1:y2, x1:x2]
    if berry.size == 0:
        return "top", {s: 0.0 for s in SIDES}

    hsv = cv2.cvtColor(berry, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 50, 40), (12, 255, 255)),
        cv2.inRange(hsv, (160, 50, 40), (180, 255, 255)),
    )
    green = cv2.inRange(hsv, (28, 35, 30), (100, 255, 255))

    # Quarter bands along each edge inside berry
    t = max(2, bh // 4)
    s = max(2, bw // 4)
    regions = {
        "top": (red[:t, :], green[:t, :]),
        "bottom": (red[-t:, :], green[-t:, :]),
        "left": (red[:, :s], green[:, :s]),
        "right": (red[:, -s:], green[:, -s:]),
    }
    scores: Dict[str, float] = {}
    for side, (r, g) in regions.items():
        r_ratio = float(np.count_nonzero(r)) / float(r.size)
        g_ratio = float(np.count_nonzero(g)) / float(g.size)
        # peduncle/calyx side: high green, low red
        scores[side] = g_ratio - 0.65 * r_ratio

    # Outside thin-stem continuity soft bonus
    pad = max(6, int(0.7 * max(bw, bh)))
    outside_boxes = {
        "top": (x1 + bw // 5, max(0, y1 - pad), x2 - bw // 5, y1),
        "bottom": (x1 + bw // 5, y2, x2 - bw // 5, min(h, y2 + pad)),
        "left": (max(0, x1 - pad), y1 + bh // 5, x1, y2 - bh // 5),
        "right": (x2, y1 + bh // 5, min(w, x2 + pad), y2 - bh // 5),
    }
    for side, box in outside_boxes.items():
        scores[side] = float(scores[side]) + 0.25 * _strip_score(frame, box)

    def key(side: str) -> Tuple[float, float]:
        bias = {"top": 0.04, "left": 0.01, "right": 0.01, "bottom": 0.0}[side]
        return (scores[side] + bias, scores[side])

    best = max(SIDES, key=key)
    return best, {k: round(float(v), 4) for k, v in scores.items()}


def asymmetric_crop_for_side(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    side: str,
    pad_stem_frac: float,
    pad_opposite_frac: float,
    pad_side_frac: float,
    img_w: int,
    img_h: int,
    pad_base_frac: float = 0.55,
) -> Tuple[int, int, int, int]:
    """Expand berry with a generous base pad on all sides, then extra on stem side.

    Base pad keeps the peduncle inside future_crop even when side estimate is wrong.
    """
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    base_x = int(round(bw * pad_base_frac))
    base_y = int(round(bh * pad_base_frac))
    # start with isotropic-ish base
    left = base_x
    right = base_x
    top = base_y
    bottom = base_y

    stem = int(round(max(bw, bh) * pad_stem_frac))
    opp = int(round(max(bw, bh) * pad_opposite_frac))
    lat = int(round(max(bw, bh) * pad_side_frac))

    if side == "top":
        top = max(top, stem)
        bottom = max(bottom, opp)
        left = max(left, lat)
        right = max(right, lat)
    elif side == "bottom":
        bottom = max(bottom, stem)
        top = max(top, opp)
        left = max(left, lat)
        right = max(right, lat)
    elif side == "left":
        left = max(left, stem)
        right = max(right, opp)
        top = max(top, lat)
        bottom = max(bottom, lat)
    else:  # right
        right = max(right, stem)
        left = max(left, opp)
        top = max(top, lat)
        bottom = max(bottom, lat)

    cx1 = _clamp(x1 - left, 0, img_w - 1)
    cy1 = _clamp(y1 - top, 0, img_h - 1)
    cx2 = _clamp(x2 + right, 1, img_w)
    cy2 = _clamp(y2 + bottom, 1, img_h)
    if cx2 <= cx1:
        cx2 = min(img_w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(img_h, cy1 + 1)
    return cx1, cy1, cx2, cy2


@dataclass
class Proposal:
    image_id: str
    source_file: str
    obj_idx: int
    conf: float
    det_class: int
    berry_xyxy: Tuple[int, int, int, int]
    crop_xyxy: Tuple[int, int, int, int]
    peduncle_side: str
    side_scores: Dict[str, float]
    sharpness: float
    width: int
    height: int


def _laplacian(bgr: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main() -> int:
    ap = argparse.ArgumentParser(description="New-straw primary review: berry + future crop labels.")
    ap.add_argument("--src", default=str(Path("/home/andrei/project/new straw")))
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "data" / "new_straw_primary_review"),
    )
    ap.add_argument(
        "--weights",
        default=str(REPO_ROOT / "runs" / "detect_benchmark_v3" / "yolov8s_v3_lowdensity" / "weights" / "best.pt"),
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--pad-stem-frac", type=float, default=1.00, help="Extra pad on peduncle side.")
    ap.add_argument("--pad-opposite-frac", type=float, default=0.40, help="Keep opposite room so wrong side estimate still captures stem.")
    ap.add_argument("--pad-side-frac", type=float, default=0.35)
    ap.add_argument("--min-berry-px", type=float, default=40.0)
    ap.add_argument("--max-berries-per-image", type=int, default=12)
    ap.add_argument("--ripe-only-heuristic", action="store_true", help="Keep only red-ish detections.")
    ap.add_argument(
        "--id-prefix",
        default="web",
        help="Filename prefix for review images (e.g. web → web__name.jpg, vlad → vlad__name.jpg).",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    images_dir = out / "images"
    labels_dir = out / "labels"
    preview_dir = out / "preview"
    crops_preview = out / "crops_preview"
    meta_dir = out / "meta"
    reports_dir = out / "reports"
    for d in (images_dir, labels_dir, preview_dir, crops_preview, meta_dir, reports_dir):
        _safe_mkdir(d)

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"weights not found: {weights}")

    model = YOLO(str(weights))
    files = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    proposals: List[Proposal] = []
    per_image: Dict[str, List[Proposal]] = {}

    classes_txt = out / "classes.txt"
    classes_txt.write_text("berry\nfuture_crop\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        "names:\n  0: berry\n  1: future_crop\n"
        f"path: {out.resolve()}\n"
        "train: images\nval: images\n",
        encoding="utf-8",
    )

    for img_path in files:
        frame = _read_bgr(img_path)
        if frame is None:
            print(f"skip unreadable: {img_path.name}", flush=True)
            continue
        h, w = frame.shape[:2]
        # Downscale extreme photos for detector only; keep labels on original coords.
        det_frame = frame
        scale = 1.0
        max_side = max(h, w)
        if max_side > 4000:
            scale = 4000.0 / float(max_side)
            det_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        results = model.predict(
            source=det_frame,
            conf=float(args.conf),
            iou=float(args.iou),
            imgsz=int(args.imgsz),
            verbose=False,
        )
        if not results:
            continue
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            print(f"no dets: {img_path.name}", flush=True)
            continue

        boxes = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        # map back to original coords if scaled
        if scale != 1.0:
            boxes = boxes / scale

        cands: List[Proposal] = []
        prefix = re.sub(r"[^\w\-]+", "", str(args.id_prefix).strip()) or "web"
        image_id = f"{prefix}__{_slug(img_path.name)}"
        for i, (box, conf, cid) in enumerate(zip(boxes, confs, clss)):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1, y1 = _clamp(x1, 0, w - 1), _clamp(y1, 0, h - 1)
            x2, y2 = _clamp(x2, 1, w), _clamp(y2, 1, h)
            if x2 <= x1 or y2 <= y1:
                continue
            diam = max(x2 - x1, y2 - y1)
            if diam < float(args.min_berry_px):
                continue
            if bool(args.ripe_only_heuristic):
                hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv, (0, 50, 40), (12, 255, 255))
                m2 = cv2.inRange(hsv, (160, 50, 40), (180, 255, 255))
                red = float(np.count_nonzero(cv2.bitwise_or(m1, m2))) / float(max(1, (y2 - y1) * (x2 - x1)))
                if red < 0.08:
                    continue

            side, scores = estimate_peduncle_side(frame, (x1, y1, x2, y2))
            cx1, cy1, cx2, cy2 = asymmetric_crop_for_side(
                x1, y1, x2, y2,
                side=side,
                pad_stem_frac=float(args.pad_stem_frac),
                pad_opposite_frac=float(args.pad_opposite_frac),
                pad_side_frac=float(args.pad_side_frac),
                img_w=w,
                img_h=h,
            )
            sharp = _laplacian(frame[y1:y2, x1:x2])
            prop = Proposal(
                image_id=image_id,
                source_file=str(img_path),
                obj_idx=i,
                conf=float(conf),
                det_class=int(cid),
                berry_xyxy=(x1, y1, x2, y2),
                crop_xyxy=(cx1, cy1, cx2, cy2),
                peduncle_side=side,
                side_scores={k: round(float(v), 4) for k, v in scores.items()},
                sharpness=round(sharp, 2),
                width=w,
                height=h,
            )
            cands.append(prop)

        # keep highest-conf berries, prefer larger
        cands.sort(key=lambda p: (p.conf, (p.berry_xyxy[2] - p.berry_xyxy[0]) * (p.berry_xyxy[3] - p.berry_xyxy[1])), reverse=True)
        cands = cands[: int(args.max_berries_per_image)]
        if not cands:
            print(f"filtered all: {img_path.name}", flush=True)
            continue

        # Save original as jpg into review set
        out_img = images_dir / f"{image_id}.jpg"
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            buf.tofile(str(out_img))
        else:
            cv2.imwrite(str(out_img), frame)

        # YOLO labels: berry + future_crop for each proposal
        lines: List[str] = []
        vis = frame.copy()
        for j, p in enumerate(cands):
            bx1, by1, bx2, by2 = p.berry_xyxy
            cx1, cy1, cx2, cy2 = p.crop_xyxy
            bxc, byc, bw, bh = _xyxy_to_yolo(bx1, by1, bx2, by2, w, h)
            cxc, cyc, cw, ch = _xyxy_to_yolo(cx1, cy1, cx2, cy2, w, h)
            lines.append(f"{CLASS_BERRY} {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}")
            lines.append(f"{CLASS_FUTURE_CROP} {cxc:.6f} {cyc:.6f} {cw:.6f} {ch:.6f}")

            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 220, 0), 2)
            cv2.rectangle(vis, (cx1, cy1), (cx2, cy2), (0, 140, 255), 2)
            cv2.putText(
                vis,
                f"#{j} berry conf={p.conf:.2f} stem={p.peduncle_side}",
                (bx1, max(18, by1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            # crop preview strip
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size:
                rel_bx1, rel_by1 = bx1 - cx1, by1 - cy1
                rel_bx2, rel_by2 = bx2 - cx1, by2 - cy1
                cv = crop.copy()
                cv2.rectangle(cv, (rel_bx1, rel_by1), (rel_bx2, rel_by2), (0, 220, 0), 2)
                cv2.putText(cv, p.peduncle_side, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
                cv2.imwrite(str(crops_preview / f"{image_id}__obj{j:02d}.jpg"), cv)

            p.obj_idx = j
            proposals.append(p)
            meta_path = meta_dir / f"{image_id}__obj{j:02d}.json"
            meta_path.write_text(json.dumps(asdict(p), ensure_ascii=False, indent=2), encoding="utf-8")

        (labels_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        cv2.imwrite(str(preview_dir / f"{image_id}.jpg"), vis)
        per_image[image_id] = cands
        print(f"{img_path.name}: berries={len(cands)} sides={[c.peduncle_side for c in cands]}", flush=True)

    # manifest + domain note
    man_path = reports_dir / "proposals_manifest.csv"
    with man_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "image_id", "source_file", "obj_idx", "conf", "peduncle_side",
                "berry_xyxy", "crop_xyxy", "sharpness", "width", "height",
            ],
        )
        wr.writeheader()
        for p in proposals:
            wr.writerow(
                {
                    "image_id": p.image_id,
                    "source_file": p.source_file,
                    "obj_idx": p.obj_idx,
                    "conf": round(p.conf, 4),
                    "peduncle_side": p.peduncle_side,
                    "berry_xyxy": list(p.berry_xyxy),
                    "crop_xyxy": list(p.crop_xyxy),
                    "sharpness": p.sharpness,
                    "width": p.width,
                    "height": p.height,
                }
            )

    summary = {
        "src": str(src),
        "out": str(out),
        "images_processed": len(files),
        "images_with_proposals": len(per_image),
        "berry_proposals": len(proposals),
        "classes": {"0": "berry", "1": "future_crop"},
        "pad": {
            "stem_frac": float(args.pad_stem_frac),
            "opposite_frac": float(args.pad_opposite_frac),
            "side_frac": float(args.pad_side_frac),
        },
        "domain_gap_policy": {
            "source_tag": "new_straw_web",
            "quality": "internet_high_res_cleaner_than_field",
            "do_not_mix_into_production_yet": True,
            "recommended_next": [
                "Human re-label berry + future_crop boxes in primary review",
                "Only then cut crops into peduncle labeling queue",
                "When training: keep web samples in separate split / weight / downsample to camera-like resolution",
                "Do not raise global sharpness filters based on web photos — would kill field data",
            ],
        },
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "DOMAIN_GAP.md").write_text(
        "# Domain gap: new straw (web) vs field camera\n\n"
        "These photos are sharper, cleaner, and often studio/garden stock quality.\n"
        "Existing rover models were trained mostly on noisier Roboflow/FPS field frames.\n\n"
        "## Risk if mixed naively\n"
        "- Detector/classifier may overfit to clean texture and fail on dusty/noisy field cams.\n"
        "- Sharpness / red-ratio auto-filters tuned on field data may reject or over-accept inconsistently.\n\n"
        "## Policy for this project\n"
        "1. Keep web pack in `data/new_straw_primary_review/` until human primary approve.\n"
        "2. Tag all derived crops `web__...`.\n"
        "3. Do not put into production training sets until after peduncle labels + a mixed eval split.\n"
        "4. If fine-tuning later: balance batch (field vs web) or downsample web to ~field resolution.\n"
        "5. Do not retune global quality thresholds from web sharpness alone.\n",
        encoding="utf-8",
    )

    print(json.dumps({"images_with_proposals": len(per_image), "berry_proposals": len(proposals), "out": str(out)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
