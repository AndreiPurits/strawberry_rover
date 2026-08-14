"""Offline manipulator approach pipeline (planning / verify only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import (
    ApproachResult,
    ApproachState,
    ApproachStatus,
    load_config,
)
from .claw_exclude import detect_claw_exclude_regions, filter_detections_exclude_regions
from .planner import plan_one_shot
from .tracker import TargetTracker
from .verify import verify_success


class ManipulatorApproachPipeline:
    """Lock → one-shot plan → verify. Does not command actuators."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, config_path: Optional[Path] = None):
        self.cfg = cfg if cfg is not None else load_config(config_path)
        lock_cfg = self.cfg.get("lock") or {}
        self.tracker = TargetTracker(
            roi_pad_px=float(lock_cfg.get("roi_pad_px", 150.0)),
            hold_max=int(lock_cfg.get("hold_max", 14)),
            max_jump_px=float(lock_cfg.get("max_jump_px", 140.0)),
            min_conf=float(lock_cfg.get("min_conf", 0.25)),
        )

    def lock_from_detections(
        self,
        detections: List[dict],
        *,
        depth_m: Optional[float] = None,
        image_wh: Optional[tuple] = None,
        gray: Optional[np.ndarray] = None,
    ) -> ApproachResult:
        dets = list(detections)
        claw_cfg = self.cfg.get("claw_exclude") or {}
        if gray is not None and claw_cfg.get("enabled", True):
            regions = detect_claw_exclude_regions(
                gray,
                dark_threshold=int(claw_cfg.get("dark_threshold", 48)),
                bottom_frac=float(claw_cfg.get("bottom_frac", 0.62)),
                min_area=float(claw_cfg.get("min_area", 250.0)),
            )
            if image_wh is not None:
                dets = filter_detections_exclude_regions(dets, regions, int(image_wh[0]), int(image_wh[1]))
        det = self.tracker.select(dets, depth_m=depth_m)
        lock = self.tracker.as_lock()
        if det is None and not self.tracker.can_hold:
            return ApproachResult(
                status=ApproachStatus.REJECT,
                state=ApproachState.REJECT,
                reason="NO_LOCK",
                lock=lock,
            )
        if lock is None:
            return ApproachResult(
                status=ApproachStatus.INCOMPLETE,
                state=ApproachState.SEARCH,
                reason="SEARCH",
            )
        return ApproachResult(
            status=ApproachStatus.OK,
            state=ApproachState.LOCK,
            reason="LOCKED",
            lock=lock,
        )

    def plan(
        self,
        q_start: Dict[str, Any],
        berry_lock: Dict[str, Any],
        learned: Dict[str, Any],
    ) -> ApproachResult:
        if not self.cfg.get("enabled", True):
            return ApproachResult(
                status=ApproachStatus.INCOMPLETE,
                state=ApproachState.REJECT,
                reason="FEATURE_DISABLED",
            )
        plan = plan_one_shot(q_start, berry_lock, learned, cfg=self.cfg)
        return ApproachResult(
            status=ApproachStatus.OK,
            state=ApproachState.PLAN,
            reason=plan.reason,
            lock=plan.berry_lock,
            plan=plan,
        )

    def verify_move(
        self,
        berry_before: Dict[str, Any],
        berry_after: Optional[Dict[str, Any]],
        q_start: Dict[str, Any],
        q_target: Dict[str, Any],
        q_end: Dict[str, Any],
        *,
        expected_px: Optional[float] = None,
        expected_py: Optional[float] = None,
    ) -> ApproachResult:
        ok, report = verify_success(
            berry_before,
            berry_after,
            q_start,
            q_target,
            q_end,
            cfg=self.cfg,
            expected_px=expected_px,
            expected_py=expected_py,
        )
        return ApproachResult(
            status=ApproachStatus.OK if ok else ApproachStatus.INCOMPLETE,
            state=ApproachState.STANDOFF if ok else ApproachState.VERIFY,
            reason="STANDOFF" if ok else (report.reasons[0] if report.reasons else "VERIFY"),
            verify=report,
        )
