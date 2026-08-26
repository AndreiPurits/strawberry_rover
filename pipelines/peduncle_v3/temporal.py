"""Temporal confirmation before peduncle standoff / grasp (from peduncle_grasp)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

Point2D = Tuple[float, float]


@dataclass
class StemObservation:
    track_id: int
    calyx: Point2D
    base: Point2D
    tip: Point2D
    grasp: Point2D
    score: float
    angle_deg: float


def _angle(base: Point2D, tip: Point2D) -> float:
    return math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0]))


def _span(pts: List[Point2D]) -> float:
    arr = np.asarray(pts, dtype=np.float32)
    return float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))


class TemporalStemVerifier:
    """Require N stable TARGET frames before motion."""

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.required = int(cfg.get("required_frames", 3))
        self.max_base_shift_px = float(cfg.get("max_base_shift_px", 18))
        self.max_grasp_shift_px = float(cfg.get("max_grasp_shift_px", 22))
        self.max_calyx_shift_px = float(cfg.get("max_calyx_shift_px", 20))
        self.max_angle_change_deg = float(cfg.get("max_angle_change_deg", 25))
        self.max_obb_switch_px = float(cfg.get("max_obb_switch_px", 28))
        self._buf: Deque[StemObservation] = deque(maxlen=max(4, self.required * 3))

    def reset(self) -> None:
        self._buf.clear()

    def push(
        self,
        *,
        track_id: int,
        calyx: Point2D,
        base: Point2D,
        tip: Point2D,
        grasp: Point2D,
        score: float,
    ) -> StemObservation:
        obs = StemObservation(
            track_id=track_id,
            calyx=calyx,
            base=base,
            tip=tip,
            grasp=grasp,
            score=float(score),
            angle_deg=_angle(base, tip),
        )
        # Detect OBB identity switch vs previous frame (different stem).
        if self._buf:
            prev = self._buf[-1]
            if prev.track_id == track_id:
                db = math.hypot(base[0] - prev.base[0], base[1] - prev.base[1])
                if db > self.max_obb_switch_px:
                    # Hard reset — association jumped to another OBB.
                    self._buf.clear()
        self._buf.append(obs)
        return obs

    def evaluate(self, min_score: float) -> Tuple[bool, str, int]:
        if not self._buf:
            return False, "NO_OBSERVATIONS", 0
        tid = self._buf[-1].track_id
        recent = [o for o in self._buf if o.track_id == tid]
        if len(recent) < self.required:
            return False, f"TEMPORAL_CONFIRMATION_{len(recent)}_OF_{self.required}", len(recent)
        window = recent[-self.required :]
        if any(o.score < min_score for o in window):
            return False, "SCORE_BELOW_THRESHOLD", len(window)
        if _span([o.base for o in window]) > self.max_base_shift_px:
            return False, "BASE_UNSTABLE", len(window)
        if _span([o.grasp for o in window]) > self.max_grasp_shift_px:
            return False, "GRASP_UNSTABLE", len(window)
        if _span([o.calyx for o in window]) > self.max_calyx_shift_px:
            return False, "CALYX_UNSTABLE", len(window)
        angs = [o.angle_deg for o in window]
        if max(angs) - min(angs) > self.max_angle_change_deg:
            return False, "ANGLE_UNSTABLE", len(window)
        return True, "TEMPORAL_OK", len(window)

    def mean_grasp(self) -> Optional[Point2D]:
        if len(self._buf) < self.required:
            return None
        window = list(self._buf)[-self.required :]
        return (
            float(sum(o.grasp[0] for o in window) / len(window)),
            float(sum(o.grasp[1] for o in window) / len(window)),
        )
