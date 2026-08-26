"""Peduncle v3 runtime: focus crop → calyx pose → peduncle OBB → LogReg association."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

# TensorRT (Jetson) still references np.bool removed in NumPy>=1.24
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined,assignment]

from .association import AssociationOutcome, AssociationScorer
from .geometry import build_focus_roi
from .grasp_point import propose_grasp_point
from .viz import draw_peduncle_v3_overlay

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO / "config/peduncle_v3_runtime.yaml"
Point2D = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class StageState(str, Enum):
    BERRY_SEARCH = "BERRY_SEARCH"
    BERRY_LOCKED = "BERRY_LOCKED"
    BERRY_APPROACH = "BERRY_APPROACH"
    PEDUNCLE_ACQUIRE = "PEDUNCLE_ACQUIRE"
    PEDUNCLE_CONFIRMED = "PEDUNCLE_CONFIRMED"
    PEDUNCLE_STANDOFF = "PEDUNCLE_STANDOFF"
    GRASP_APPROACH = "GRASP_APPROACH"
    GRASP_CLOSE = "GRASP_CLOSE"
    GRASP_VERIFY = "GRASP_VERIFY"
    GRASP_OK = "GRASP_OK"
    GRASP_FAILED = "GRASP_FAILED"
    GRASP_UNCERTAIN = "GRASP_UNCERTAIN"
    NO_ASSOC = "NO_ASSOC"
    NO_GRASP_POINT = "NO_GRASP_POINT"
    ABORT = "ABORT"
    DONE = "DONE"
    STOP = "STOP"


@dataclass
class PeduncleV3Result:
    state: StageState
    reason: str
    berry_bbox: Optional[BBox] = None
    roi: Optional[Tuple[int, int, int, int]] = None
    calyx_xy: Optional[Point2D] = None
    calyx_xy_crop: Optional[Point2D] = None
    calyx_conf: float = 0.0
    calyx_present: float = 0.0
    calyx_occluded: float = 0.0
    calyx_visible: float = 0.0
    association: Optional[AssociationOutcome] = None
    grasp_point: Optional[Dict[str, Any]] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)
    depth_m: Optional[float] = None
    overlay_bgr: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, Any]:
        assoc = self.association.as_dict() if self.association else None
        return {
            "state": self.state.value,
            "reason": self.reason,
            "berry_bbox": list(self.berry_bbox) if self.berry_bbox else None,
            "roi": list(self.roi) if self.roi else None,
            "calyx": {
                "x": None if self.calyx_xy is None else round(self.calyx_xy[0], 2),
                "y": None if self.calyx_xy is None else round(self.calyx_xy[1], 2),
                "x_crop": None if self.calyx_xy_crop is None else round(self.calyx_xy_crop[0], 2),
                "y_crop": None if self.calyx_xy_crop is None else round(self.calyx_xy_crop[1], 2),
                "conf": round(self.calyx_conf, 4),
                "present": self.calyx_present,
                "occluded": self.calyx_occluded,
                "visible": self.calyx_visible,
            },
            "association": assoc,
            "grasp_point": self.grasp_point,
            "decision": assoc["decision"] if assoc else "NO_ASSOC",
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "depth_m": self.depth_m,
            "n_peduncle": 0 if not self.association else len(self.association.candidates),
        }


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    import os

    if path is None:
        env = os.environ.get("PEDUNCLE_V3_CONFIG", "").strip()
        path = Path(env) if env else DEFAULT_CFG
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def resolve_weight(entry: Dict[str, Any], prefer: str, roots: Sequence[Path]) -> Path:
    order = []
    if prefer == "engine":
        order = ["engine", "onnx", "pt"]
    elif prefer == "onnx":
        order = ["onnx", "engine", "pt"]
    else:
        order = ["pt", "engine", "onnx"]
    for key in order:
        rel = entry.get(key)
        if not rel:
            continue
        for root in roots:
            cand = Path(rel) if Path(rel).is_absolute() else (root / rel)
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"no weights for {entry} under {list(roots)}")


class PeduncleV3Pipeline:
    """Run only after berry lock + successful close approach (caller-gated)."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, config_path: Optional[Path] = None):
        self.cfg = cfg if cfg is not None else load_config(config_path)
        roots = []
        for r in self.cfg.get("model_roots") or [
            "runs/deploy_orin_new",
            "models/hf_peduncle_v3",
            "models/hf_strawberry",
            ".",
        ]:
            roots.append(REPO / r if not Path(r).is_absolute() else Path(r))
        self.roots = roots
        prefer = str(self.cfg.get("runtime_preference") or "engine")
        models = self.cfg.get("models") or {}
        assoc_cfg = models.get("association") or {}
        # association is JSON-only
        assoc_rel = assoc_cfg.get("json") or "models/association_logreg_v3.json"
        assoc_file = None
        for root in self.roots:
            cand = Path(assoc_rel) if Path(assoc_rel).is_absolute() else (root / assoc_rel)
            if cand.is_file():
                assoc_file = cand
                break
        if assoc_file is None:
            raise FileNotFoundError(f"association json missing: {assoc_rel}")
        thr = assoc_cfg.get("threshold", (self.cfg.get("association") or {}).get("threshold", 0.1))
        margin = float((self.cfg.get("association") or {}).get("ambiguity_margin", 0.05))
        self.scorer = AssociationScorer(assoc_file, threshold=float(thr), ambiguity_margin=margin)
        self.prefer = prefer
        self._calyx = None
        self._peduncle = None
        self._calyx_path = resolve_weight(models.get("calyx") or {"engine": "models/calyx.engine"}, prefer, self.roots)
        self._peduncle_path = resolve_weight(
            models.get("peduncle") or {"engine": "models/peduncle.engine"}, prefer, self.roots
        )
        self.calyx_imgsz = int((models.get("calyx") or {}).get("imgsz", 640))
        self.peduncle_imgsz = int((models.get("peduncle") or {}).get("imgsz", 640))
        infer = self.cfg.get("infer") or {}
        self.calyx_conf_thr = float(infer.get("calyx_conf", 0.15))
        self.peduncle_conf_thr = float(infer.get("peduncle_conf", 0.15))
        self.peduncle_iou = float(infer.get("peduncle_iou", 0.45))
        self.peduncle_max_det = int(infer.get("peduncle_max_det", 15))
        self.roi_cfg = self.cfg.get("focus_roi") or {}

    def _patch_nms(self) -> None:
        try:
            from scripts.yolo_jetson_compat import apply_torchvision_nms_patch

            apply_torchvision_nms_patch()
        except Exception:
            pass

    def _ensure_calyx(self):
        if self._calyx is not None:
            return self._calyx
        self._patch_nms()
        from ultralytics import YOLO

        self._calyx = YOLO(str(self._calyx_path), task="pose")
        return self._calyx

    def _ensure_peduncle(self):
        if self._peduncle is not None:
            return self._peduncle
        self._patch_nms()
        from ultralytics import YOLO

        self._peduncle = YOLO(str(self._peduncle_path), task="obb")
        return self._peduncle

    def _predict_calyx(self, crop_bgr: np.ndarray) -> Tuple[Optional[Point2D], float, float, float, float]:
        model = self._ensure_calyx()
        res = model.predict(
            source=crop_bgr,
            imgsz=self.calyx_imgsz,
            conf=self.calyx_conf_thr,
            half=True,
            verbose=False,
        )
        if not res:
            return None, 0.0, 0.0, 0.0, 0.0
        r0 = res[0]
        kps = getattr(r0, "keypoints", None)
        if kps is None or getattr(kps, "xy", None) is None:
            return None, 0.0, 0.0, 0.0, 0.0
        xy = kps.xy.cpu().numpy()
        confs = kps.conf.cpu().numpy() if getattr(kps, "conf", None) is not None else None
        best_i, best_c = -1, -1.0
        for i in range(len(xy)):
            c = float(confs[i][0]) if confs is not None else 1.0
            if c > best_c:
                best_c, best_i = c, i
        if best_i < 0 or best_c < self.calyx_conf_thr:
            return None, float(max(0.0, best_c)), 0.0, 0.0, 0.0
        x, y = float(xy[best_i][0][0]), float(xy[best_i][0][1])
        # YOLO pose conf is continuous; map to present / visible / occluded soft flags.
        present = 1.0
        visible = 1.0 if best_c >= 0.50 else 0.0
        occluded = 1.0 if present and not visible else 0.0
        return (x, y), float(best_c), present, occluded, visible

    def _predict_peduncles(self, crop_bgr: np.ndarray) -> List[Tuple[List[Point2D], float]]:
        model = self._ensure_peduncle()
        res = model.predict(
            source=crop_bgr,
            imgsz=self.peduncle_imgsz,
            conf=self.peduncle_conf_thr,
            iou=self.peduncle_iou,
            max_det=self.peduncle_max_det,
            half=True,
            verbose=False,
        )
        hits: List[Tuple[List[Point2D], float]] = []
        if not res:
            return hits
        obb = getattr(res[0], "obb", None)
        if obb is None or getattr(obb, "xyxyxyxy", None) is None:
            return hits
        polys = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy() if getattr(obb, "conf", None) is not None else np.ones(len(polys))
        for poly, conf in zip(polys, confs):
            pts = [(float(p[0]), float(p[1])) for p in poly.reshape(-1, 2)]
            hits.append((pts, float(conf)))
        return hits

    def process_focused_berry(
        self,
        bgr: np.ndarray,
        berry_bbox: BBox,
        *,
        depth_m: Optional[float] = None,
        draw_overlay: bool = True,
        already_crop: bool = False,
    ) -> PeduncleV3Result:
        """Run peduncle stage on a locked target berry (full frame or canon crop)."""
        t_all = time.time()
        timings: Dict[str, float] = {}
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in berry_bbox]
        berry_h = max(1.0, y2 - y1)
        center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))

        if already_crop:
            roi = (0, 0, w, h)
            crop = bgr
            ox = oy = 0
        else:
            roi = build_focus_roi((x1, y1, x2, y2), (w, h), **{k: v for k, v in self.roi_cfg.items() if k in {
                "width_scale", "height_above_scale", "height_below_scale", "min_size_px", "max_size_px", "pad_px"
            }})
            rx1, ry1, rx2, ry2 = roi
            crop = bgr[ry1:ry2, rx1:rx2]
            ox, oy = rx1, ry1
            if crop.size == 0:
                return PeduncleV3Result(
                    state=StageState.NO_ASSOC,
                    reason="EMPTY_ROI",
                    berry_bbox=(x1, y1, x2, y2),
                    roi=roi,
                    depth_m=depth_m,
                    timings_ms={"total": (time.time() - t_all) * 1000.0},
                )

        t0 = time.time()
        calyx_crop, calyx_conf, present, occluded, visible = self._predict_calyx(crop)
        timings["calyx_ms"] = (time.time() - t0) * 1000.0
        calyx_full = None
        if calyx_crop is not None:
            calyx_full = (calyx_crop[0] + ox, calyx_crop[1] + oy)

        t0 = time.time()
        ped_crop = self._predict_peduncles(crop)
        timings["peduncle_ms"] = (time.time() - t0) * 1000.0
        ped_full = [
            ([(p[0] + ox, p[1] + oy) for p in poly], conf) for poly, conf in ped_crop
        ]

        t0 = time.time()
        # Berry bbox in the same frame coords as OBB (full frame unless already_crop).
        outcome = self.scorer.associate(
            ped_full,
            berry_bbox=(x1, y1, x2, y2),
            berry_center=center,
            berry_height=berry_h,
            calyx_xy=calyx_full,
            calyx_conf=calyx_conf,
            calyx_present=present,
            calyx_occluded=occluded,
            calyx_visible=visible,
        )
        timings["association_ms"] = (time.time() - t0) * 1000.0
        timings["total_ms"] = (time.time() - t_all) * 1000.0

        if outcome.decision == "TARGET":
            state = StageState.PEDUNCLE_CONFIRMED
            reason = outcome.reason
        else:
            state = StageState.NO_ASSOC
            reason = outcome.reason

        grasp_dict = None
        if outcome.decision == "TARGET" and outcome.selected is not None and calyx_full is not None:
            gp = propose_grasp_point(
                outcome.selected.polygon,
                outcome.selected.base,
                outcome.selected.tip,
                calyx_full,
                berry_h,
                self.cfg.get("grasp_point") or {},
            )
            grasp_dict = gp.as_dict()
            if not gp.ok:
                state = StageState.NO_GRASP_POINT
                reason = gp.reason

        result = PeduncleV3Result(
            state=state,
            reason=reason,
            berry_bbox=(x1, y1, x2, y2),
            roi=roi,
            calyx_xy=calyx_full,
            calyx_xy_crop=calyx_crop,
            calyx_conf=calyx_conf,
            calyx_present=present,
            calyx_occluded=occluded,
            calyx_visible=visible,
            association=outcome,
            grasp_point=grasp_dict,
            timings_ms=timings,
            depth_m=depth_m,
        )
        if draw_overlay:
            gxy = None
            if result.grasp_point and result.grasp_point.get("point_2d"):
                gxy = (float(result.grasp_point["point_2d"][0]), float(result.grasp_point["point_2d"][1]))
            result.overlay_bgr = draw_peduncle_v3_overlay(
                bgr,
                berry_bbox=result.berry_bbox,
                calyx_xy=result.calyx_xy,
                calyx_conf=result.calyx_conf,
                association=result.association,
                state=result.state.value,
                roi=result.roi,
                grasp_xy=gxy,
            )
        return result
