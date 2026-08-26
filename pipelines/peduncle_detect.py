"""Peduncle OBB detector gated by confirmed close standoff (8–15 cm)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "config/peduncle_detect.yaml"
Point = Tuple[float, float]


@dataclass
class PeduncleHit:
    polygon: List[Point]
    conf: float
    cx: float
    cy: float


@dataclass
class PeduncleDetectResult:
    enabled: bool
    ran: bool
    reason: str
    depth_m: Optional[float] = None
    hits: List[PeduncleHit] = field(default_factory=list)
    roi: Optional[Tuple[int, int, int, int]] = None
    elapsed_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "ran": self.ran,
            "reason": self.reason,
            "depth_m": self.depth_m,
            "roi": list(self.roi) if self.roi else None,
            "elapsed_s": round(self.elapsed_s, 3),
            "count": len(self.hits),
            "hits": [
                {
                    "conf": round(h.conf, 3),
                    "cx": round(h.cx, 1),
                    "cy": round(h.cy, 1),
                    "polygon": [[round(x, 1), round(y, 1)] for x, y in h.polygon],
                }
                for h in self.hits
            ],
        }


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_CFG
    if not p.is_file():
        return {
            "enabled": True,
            "depth_min_m": 0.08,
            "depth_max_m": 0.15,
            "weights": "models/peduncle/peduncle_obb_yolov8n_v21_best.pt",
            "infer": {"conf": 0.15, "iou": 0.45, "imgsz": 640, "max_det": 10},
            "roi": {
                "width_berry_scale": 1.4,
                "height_above_scale": 0.55,
                "height_below_scale": 0.35,
                "min_size_px": 96,
                "max_size_px": 640,
            },
        }
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def depth_in_peduncle_gate(
    depth_m: Optional[float],
    *,
    depth_min_m: float = 0.08,
    depth_max_m: float = 0.15,
) -> bool:
    if depth_m is None:
        return False
    d = float(depth_m)
    return depth_min_m <= d <= depth_max_m


def _berry_roi(
    berry_bbox: Tuple[float, float, float, float],
    image_wh: Tuple[int, int],
    roi_cfg: Dict[str, Any],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = berry_bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    # Calyx / stem zone sits above berry center.
    calyx_y = y1 + 0.15 * bh
    w_scale = float(roi_cfg.get("width_berry_scale", 1.4))
    above = float(roi_cfg.get("height_above_scale", 0.55)) * bh
    below = float(roi_cfg.get("height_below_scale", 0.35)) * bh
    half_w = 0.5 * w_scale * bw
    rx1 = cx - half_w
    rx2 = cx + half_w
    ry1 = calyx_y - above
    ry2 = calyx_y + below
    side = max(rx2 - rx1, ry2 - ry1)
    min_sz = float(roi_cfg.get("min_size_px", 96))
    max_sz = float(roi_cfg.get("max_size_px", 640))
    side = min(max_sz, max(min_sz, side))
    rcx = 0.5 * (rx1 + rx2)
    rcy = 0.5 * (ry1 + ry2)
    img_w, img_h = image_wh
    x1i = int(max(0, min(img_w - 1, round(rcx - 0.5 * side))))
    y1i = int(max(0, min(img_h - 1, round(rcy - 0.5 * side))))
    x2i = int(max(x1i + 1, min(img_w, round(rcx + 0.5 * side))))
    y2i = int(max(y1i + 1, min(img_h, round(rcy + 0.5 * side))))
    return x1i, y1i, x2i, y2i


class PeduncleDetector:
    """Lazy-loaded YOLOv8n-OBB. Inference only inside depth gate."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, config_path: Optional[Path] = None):
        self.cfg = cfg if cfg is not None else load_config(config_path)
        self._model = None
        self._weights = REPO / str(self.cfg.get("weights") or "")

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    @property
    def depth_min_m(self) -> float:
        return float(self.cfg.get("depth_min_m", 0.08))

    @property
    def depth_max_m(self) -> float:
        return float(self.cfg.get("depth_max_m", 0.15))

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if not self._weights.is_file():
            raise FileNotFoundError(f"peduncle weights missing: {self._weights}")
        try:
            from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

            apply_torchvision_nms_patch()
        except Exception:
            pass
        from ultralytics import YOLO

        self._model = YOLO(str(self._weights))
        return self._model

    def detect(
        self,
        bgr: np.ndarray,
        *,
        depth_m: Optional[float],
        berry_bbox: Optional[Sequence[float]] = None,
        force: bool = False,
    ) -> PeduncleDetectResult:
        import time

        t0 = time.time()
        if not self.enabled:
            return PeduncleDetectResult(enabled=False, ran=False, reason="disabled", depth_m=depth_m)

        if not force and not depth_in_peduncle_gate(
            depth_m, depth_min_m=self.depth_min_m, depth_max_m=self.depth_max_m
        ):
            return PeduncleDetectResult(
                enabled=True,
                ran=False,
                reason="depth_out_of_gate",
                depth_m=depth_m,
                elapsed_s=time.time() - t0,
            )

        if bgr is None or bgr.size == 0:
            return PeduncleDetectResult(
                enabled=True, ran=False, reason="no_frame", depth_m=depth_m, elapsed_s=time.time() - t0
            )

        h, w = bgr.shape[:2]
        roi = None
        crop = bgr
        ox = oy = 0
        if berry_bbox is not None and len(berry_bbox) >= 4:
            roi = _berry_roi(
                (float(berry_bbox[0]), float(berry_bbox[1]), float(berry_bbox[2]), float(berry_bbox[3])),
                (w, h),
                self.cfg.get("roi") or {},
            )
            x1, y1, x2, y2 = roi
            crop = bgr[y1:y2, x1:x2]
            ox, oy = x1, y1
            if crop.size == 0:
                return PeduncleDetectResult(
                    enabled=True,
                    ran=False,
                    reason="empty_roi",
                    depth_m=depth_m,
                    roi=roi,
                    elapsed_s=time.time() - t0,
                )

        try:
            model = self._ensure_model()
        except Exception as exc:
            return PeduncleDetectResult(
                enabled=True,
                ran=False,
                reason=f"model_load:{exc}",
                depth_m=depth_m,
                roi=roi,
                elapsed_s=time.time() - t0,
            )

        infer = self.cfg.get("infer") or {}
        results = model.predict(
            source=crop,
            conf=float(infer.get("conf", 0.15)),
            iou=float(infer.get("iou", 0.45)),
            imgsz=int(infer.get("imgsz", 640)),
            max_det=int(infer.get("max_det", 10)),
            verbose=False,
        )
        hits: List[PeduncleHit] = []
        if results:
            r0 = results[0]
            obb = getattr(r0, "obb", None)
            if obb is not None and getattr(obb, "xyxyxyxy", None) is not None:
                polys = obb.xyxyxyxy.cpu().numpy()
                confs = obb.conf.cpu().numpy() if getattr(obb, "conf", None) is not None else np.ones(len(polys))
                for poly, conf in zip(polys, confs):
                    pts = [(float(ox + p[0]), float(oy + p[1])) for p in poly.reshape(-1, 2)]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    hits.append(
                        PeduncleHit(
                            polygon=pts,
                            conf=float(conf),
                            cx=float(sum(xs) / len(xs)),
                            cy=float(sum(ys) / len(ys)),
                        )
                    )
        hits.sort(key=lambda h: (-h.conf, h.cy))
        return PeduncleDetectResult(
            enabled=True,
            ran=True,
            reason="ok" if hits else "no_peduncle",
            depth_m=depth_m,
            hits=hits,
            roi=roi,
            elapsed_s=time.time() - t0,
        )


def maybe_detect_peduncle_on_frame(
    bgr: np.ndarray,
    *,
    depth_m: Optional[float],
    berry_px: Optional[float] = None,
    berry_py: Optional[float] = None,
    berry_size_px: float = 48.0,
    detector: Optional[PeduncleDetector] = None,
) -> PeduncleDetectResult:
    """Convenience: build a loose berry bbox from lock point and run gated detect."""
    det = detector or PeduncleDetector()
    bbox = None
    if berry_px is not None and berry_py is not None:
        half = 0.5 * float(berry_size_px)
        bbox = (berry_px - half, berry_py - half, berry_px + half, berry_py + half)
    return det.detect(bgr, depth_m=depth_m, berry_bbox=bbox)
