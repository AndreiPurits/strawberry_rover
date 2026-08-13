"""Offline single-frame / multi-frame peduncle grasp pipeline (vision only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import (
    BerryTarget,
    GraspState,
    PerceptionResult,
    PerceptionStatus,
    load_config,
)
from .association_v2 import associate_candidates
from .calyx import NullCalyxDetector, resolve_calyx
from .cut import propose_cut
from .roi import build_closeup_roi
from .temporal import TemporalVerifier

Point2D = Tuple[float, float]


class PeduncleGraspPipeline:
    """Safe v1: calyx → close ROI → OBB candidates → strict association → cut → temporal."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, config_path: Optional[Path] = None):
        self.cfg = cfg if cfg is not None else load_config(config_path)
        self.calyx_detector = NullCalyxDetector()
        self.verifier = TemporalVerifier(self.cfg.get("temporal_verification") or {})
        self._viewpoint_attempts: Dict[int, int] = {}

    def process_berry_frame(
        self,
        bgr: np.ndarray,
        berry_bbox: Tuple[float, float, float, float],
        *,
        track_id: int = 0,
        berry_mask: Optional[np.ndarray] = None,
        ripeness: Optional[str] = None,
        detector_conf: float = 1.0,
        obb_candidates: Sequence[Tuple[List[Point2D], float]] = (),
        depth: Optional[np.ndarray] = None,
        push_temporal: bool = True,
    ) -> PerceptionResult:
        if not self.cfg.get("enabled", True):
            return PerceptionResult(
                status=PerceptionStatus.INCOMPLETE,
                state=GraspState.REJECT_TARGET,
                reason="FEATURE_DISABLED",
            )

        x1, y1, x2, y2 = berry_bbox
        center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
        berry_h = max(1.0, y2 - y1)
        target = BerryTarget(
            track_id=track_id,
            bbox=berry_bbox,
            mask=berry_mask,
            center_2d=center,
            center_3d=None,
            berry_height_px=berry_h,
            ripeness=ripeness,
            confidence=float(detector_conf),
            occlusion_score=None,
        )

        # DETECT_OR_ESTIMATE_CALYX
        calyx = resolve_calyx(bgr, berry_bbox, berry_mask, self.cfg, self.calyx_detector)
        target.calyx = calyx
        if calyx.confidence <= 0:
            return PerceptionResult(
                status=PerceptionStatus.NEED_NEW_VIEW,
                state=GraspState.CHANGE_VIEWPOINT,
                reason="CALYX_UNKNOWN",
                suggested_view="UP",
                berry=target,
            )

        # CAPTURE_CLOSEUP ROI
        roi = build_closeup_roi(
            bgr,
            calyx.point_2d,
            berry_h,
            center,
            self.cfg.get("closeup_roi") or {},
        )

        # Map candidates into full-frame if they were predicted in ROI coords
        # Caller may pass full-frame candidates already.
        mode = str(self.cfg.get("mode", "close"))
        scored, best = associate_candidates(
            obb_candidates,
            center,
            berry_bbox,
            berry_h,
            calyx.point_2d,
            calyx.mask,
            self.cfg.get("association") or {},
            mode=mode,
        )
        target.peduncle_candidates = scored
        target.selected_peduncle = best
        target.association_confidence = float(best.score) if best else 0.0

        debug = {
            "berry_id": track_id,
            "state": GraspState.ASSOCIATE.value,
            "calyx_source": calyx.source,
            "calyx_confidence": calyx.confidence,
            "roi": [roi.x1, roi.y1, roi.x2, roi.y2],
            "candidate_count": len(scored),
            "selected_candidate": 0 if best else None,
            "association_score": target.association_confidence,
            "features": None,
            "cut_allowed": False,
            "reason": "",
        }

        if best is None:
            attempts = self._viewpoint_attempts.get(track_id, 0) + 1
            self._viewpoint_attempts[track_id] = attempts
            max_att = int((self.cfg.get("viewpoint") or {}).get("max_attempts", 3))
            if attempts >= max_att:
                debug["reason"] = "LOW_ASSOCIATION_MAX_VIEWS"
                debug["state"] = GraspState.REJECT_TARGET.value
                return PerceptionResult(
                    status=PerceptionStatus.REJECT,
                    state=GraspState.REJECT_TARGET,
                    reason="LOW_ASSOCIATION",
                    berry=target,
                    debug=debug,
                )
            debug["reason"] = "LOW_ASSOCIATION"
            debug["state"] = GraspState.CHANGE_VIEWPOINT.value
            return PerceptionResult(
                status=PerceptionStatus.NEED_NEW_VIEW,
                state=GraspState.CHANGE_VIEWPOINT,
                reason="LOW_ASSOCIATION",
                suggested_view="LEFT" if attempts % 2 else "RIGHT",
                berry=target,
                debug=debug,
            )

        debug["features"] = {
            "d_norm": best.features.normalized_base_distance,
            "contact": best.features.calyx_contact,
            "direction": best.features.direction_consistency,
            "depth": best.features.depth_consistency,
            "model_conf": best.features.model_confidence,
            "obb_length_norm": best.features.obb_length_norm,
        }

        cut = propose_cut(
            best,
            calyx.point_2d,
            berry_h,
            berry_mask,
            self.cfg.get("cut") or {},
            depth=depth,
        )
        target.cut = cut
        if not cut.clearance_ok:
            debug["reason"] = "INVALID_CLEARANCE"
            debug["state"] = GraspState.CHANGE_VIEWPOINT.value
            return PerceptionResult(
                status=PerceptionStatus.NEED_NEW_VIEW,
                state=GraspState.CHANGE_VIEWPOINT,
                reason="INVALID_CLEARANCE",
                suggested_view="UP",
                berry=target,
                debug=debug,
            )

        # Geometry-only calyx cannot alone authorize cut
        if calyx.source == "geometry" and calyx.confidence < 0.5:
            # still allow temporal path but never READY from single frame
            pass

        min_score = float((self.cfg.get("association") or {}).get("min_score", 0.28))
        if push_temporal:
            self.verifier.push(
                track_id,
                calyx.point_2d,
                best.base,
                best.tip,
                cut.point_2d,
                best.score,
            )
        ok, reason, n_conf = self.verifier.evaluate(min_score)
        debug["reason"] = reason
        debug["temporal_count"] = n_conf

        if not ok:
            debug["state"] = GraspState.VERIFY.value
            debug["cut_allowed"] = False
            return PerceptionResult(
                status=PerceptionStatus.INCOMPLETE,
                state=GraspState.VERIFY,
                reason=reason,
                berry=target,
                debug=debug,
            )

        # Extra safety: geometry calyx alone → need new view rather than cut
        if calyx.source != "model":
            debug["state"] = GraspState.CHANGE_VIEWPOINT.value
            debug["reason"] = "CALYX_GEOMETRY_ONLY"
            debug["cut_allowed"] = False
            return PerceptionResult(
                status=PerceptionStatus.NEED_NEW_VIEW,
                state=GraspState.CHANGE_VIEWPOINT,
                reason="CALYX_GEOMETRY_ONLY",
                suggested_view="UP",
                berry=target,
                debug=debug,
            )

        debug["state"] = GraspState.READY_TO_CUT.value
        debug["cut_allowed"] = True
        return PerceptionResult(
            status=PerceptionStatus.OK,
            state=GraspState.READY_TO_CUT,
            reason="READY",
            berry=target,
            debug=debug,
        )
