"""LogReg association scorer (association_logreg_v3.json)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .geometry import (
    aabb_iou,
    choose_base_tip,
    clamp01,
    contact_score,
    direction_score,
    obb_axis_ends,
    poly_aabb,
    work_point_along_stem,
)

Point2D = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class StemCandidate:
    polygon: List[Point2D]
    det_conf: float
    base: Point2D
    tip: Point2D
    work_point: Point2D
    features: Dict[str, float]
    score: float
    selected: bool = False


@dataclass
class AssociationOutcome:
    decision: str  # TARGET | NO_ASSOC
    reason: str
    threshold: float
    candidates: List[StemCandidate] = field(default_factory=list)
    selected: Optional[StemCandidate] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "threshold": self.threshold,
            "n_candidates": len(self.candidates),
            "selected": None
            if self.selected is None
            else {
                "det_conf": round(self.selected.det_conf, 4),
                "score": round(self.selected.score, 4),
                "base": [round(self.selected.base[0], 1), round(self.selected.base[1], 1)],
                "tip": [round(self.selected.tip[0], 1), round(self.selected.tip[1], 1)],
                "work_point": [
                    round(self.selected.work_point[0], 1),
                    round(self.selected.work_point[1], 1),
                ],
                "polygon": [[round(x, 1), round(y, 1)] for x, y in self.selected.polygon],
                "features": {k: round(float(v), 4) for k, v in self.selected.features.items()},
            },
            "candidates": [
                {
                    "det_conf": round(c.det_conf, 4),
                    "score": round(c.score, 4),
                    "selected": c.selected,
                    "base": [round(c.base[0], 1), round(c.base[1], 1)],
                    "tip": [round(c.tip[0], 1), round(c.tip[1], 1)],
                    "polygon": [[round(x, 1), round(y, 1)] for x, y in c.polygon],
                    "features": {k: round(float(v), 4) for k, v in c.features.items()},
                }
                for c in self.candidates
            ],
        }


class AssociationScorer:
    def __init__(self, model_json: Path, *, threshold: Optional[float] = None, ambiguity_margin: float = 0.05):
        raw = json.loads(Path(model_json).read_text(encoding="utf-8"))
        self.feat_keys: List[str] = list(raw["feat_keys"])
        self.scaler_mean = [float(x) for x in raw["scaler_mean"]]
        self.scaler_scale = [float(x) for x in raw["scaler_scale"]]
        self.coef = [float(x) for x in raw["coef"]]
        self.intercept = float(raw["intercept"])
        # Deployment config overrides JSON default (JSON often stores train thr 0.2).
        self.threshold = float(threshold if threshold is not None else raw.get("threshold", 0.1))
        self.ambiguity_margin = float(ambiguity_margin)

    def _sigmoid(self, z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def score_features(self, feats: Dict[str, float]) -> float:
        z = self.intercept
        for i, key in enumerate(self.feat_keys):
            x = float(feats.get(key, 0.0))
            scale = self.scaler_scale[i] if abs(self.scaler_scale[i]) > 1e-12 else 1.0
            xs = (x - self.scaler_mean[i]) / scale
            z += self.coef[i] * xs
        return float(self._sigmoid(z))

    def candidate_features(
        self,
        polygon: Sequence[Point2D],
        det_conf: float,
        *,
        berry_bbox: BBox,
        berry_center: Point2D,
        berry_height: float,
        calyx_xy: Point2D,
        calyx_conf: float,
        calyx_present: float,
        calyx_occluded: float,
        calyx_visible: float,
    ) -> Tuple[Dict[str, float], Point2D, Point2D, Point2D]:
        end0, end1, length = obb_axis_ends(polygon)
        base, tip, dist = choose_base_tip(end0, end1, calyx_xy)
        bh = max(1.0, float(berry_height))
        dist_norm = float(dist / bh)
        len_norm = float(length / bh)
        direction = direction_score(berry_center, calyx_xy, base, tip)
        iou = aabb_iou(poly_aabb(polygon), berry_bbox)
        contact = contact_score(base, calyx_xy, bh)
        cx, cy = berry_center
        feats = {
            "det_conf": clamp01(float(det_conf)),
            "dist_norm": dist_norm,
            "len_norm": len_norm,
            "direction": float(direction),
            "iou_berry": float(iou),
            "contact": float(contact),
            "base_rel_x": float((base[0] - cx) / bh),
            "base_rel_y": float((base[1] - cy) / bh),
            "tip_rel_y": float((tip[1] - cy) / bh),
            "calyx_conf": float(calyx_conf),
            "calyx_present": float(calyx_present),
            "calyx_occluded": float(calyx_occluded),
            "calyx_visible": float(calyx_visible),
        }
        work = work_point_along_stem(base, tip, bh)
        return feats, base, tip, work

    def associate(
        self,
        obb_hits: Sequence[Tuple[Sequence[Point2D], float]],
        *,
        berry_bbox: BBox,
        berry_center: Point2D,
        berry_height: float,
        calyx_xy: Optional[Point2D],
        calyx_conf: float = 0.0,
        calyx_present: float = 0.0,
        calyx_occluded: float = 0.0,
        calyx_visible: float = 0.0,
        threshold: Optional[float] = None,
    ) -> AssociationOutcome:
        thr = float(self.threshold if threshold is None else threshold)
        if calyx_xy is None or calyx_present < 0.5:
            return AssociationOutcome(
                decision="NO_ASSOC",
                reason="NO_CALYX",
                threshold=thr,
            )
        if not obb_hits:
            return AssociationOutcome(
                decision="NO_ASSOC",
                reason="NO_STEM",
                threshold=thr,
            )

        scored: List[StemCandidate] = []
        for poly, conf in obb_hits:
            if len(poly) != 4:
                continue
            feats, base, tip, work = self.candidate_features(
                poly,
                float(conf),
                berry_bbox=berry_bbox,
                berry_center=berry_center,
                berry_height=berry_height,
                calyx_xy=calyx_xy,
                calyx_conf=calyx_conf,
                calyx_present=calyx_present,
                calyx_occluded=calyx_occluded,
                calyx_visible=calyx_visible,
            )
            score = self.score_features(feats)
            scored.append(
                StemCandidate(
                    polygon=[(float(p[0]), float(p[1])) for p in poly],
                    det_conf=float(conf),
                    base=base,
                    tip=tip,
                    work_point=work,
                    features=feats,
                    score=score,
                )
            )
        if not scored:
            return AssociationOutcome(decision="NO_ASSOC", reason="NO_VALID_OBB", threshold=thr)

        scored.sort(key=lambda c: c.score, reverse=True)
        best = scored[0]
        if best.score < thr:
            return AssociationOutcome(
                decision="NO_ASSOC",
                reason="BELOW_THRESHOLD",
                threshold=thr,
                candidates=scored,
            )
        if len(scored) >= 2:
            gap = best.score - scored[1].score
            if gap < self.ambiguity_margin and scored[1].score >= thr:
                return AssociationOutcome(
                    decision="NO_ASSOC",
                    reason="AMBIGUOUS_TOP2",
                    threshold=thr,
                    candidates=scored,
                )
        best.selected = True
        return AssociationOutcome(
            decision="TARGET",
            reason="OK",
            threshold=thr,
            candidates=scored,
            selected=best,
        )
