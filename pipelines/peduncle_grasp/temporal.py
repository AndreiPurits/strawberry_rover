"""Temporal verification before READY_TO_CUT."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

Point2D = Tuple[float, float]


@dataclass
class FrameObservation:
    track_id: int
    calyx: Point2D
    base: Point2D
    tip: Point2D
    cut: Point2D
    score: float
    angle_deg: float


class TemporalVerifier:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.required = int(cfg.get("required_frames", 3))
        self._buf: Deque[FrameObservation] = deque(maxlen=max(3, self.required * 2))

    def reset(self) -> None:
        self._buf.clear()

    @staticmethod
    def _angle(base: Point2D, tip: Point2D) -> float:
        return math.degrees(math.atan2(tip[1] - base[1], tip[0] - base[0]))

    def push(
        self,
        track_id: int,
        calyx: Point2D,
        base: Point2D,
        tip: Point2D,
        cut: Point2D,
        score: float,
    ) -> FrameObservation:
        obs = FrameObservation(
            track_id=track_id,
            calyx=calyx,
            base=base,
            tip=tip,
            cut=cut,
            score=score,
            angle_deg=self._angle(base, tip),
        )
        self._buf.append(obs)
        return obs

    def evaluate(self, min_score: float) -> Tuple[bool, str, int]:
        """Return (ok, reason, confirmed_count)."""
        if not self._buf:
            return False, "NO_OBSERVATIONS", 0
        tid = self._buf[-1].track_id
        recent = [o for o in self._buf if o.track_id == tid]
        if len(recent) < self.required:
            return False, f"TEMPORAL_CONFIRMATION_{len(recent)}_OF_{self.required}", len(recent)

        window = recent[-self.required :]
        if any(o.score < min_score for o in window):
            return False, "SCORE_BELOW_THRESHOLD", len(window)

        max_base = float(self.cfg.get("max_base_shift_px", 18))
        max_cut = float(self.cfg.get("max_cut_shift_px", 22))
        max_ang = float(self.cfg.get("max_angle_change_deg", 25))

        bases = np_pts([o.base for o in window])
        cuts = np_pts([o.cut for o in window])
        if span(bases) > max_base:
            return False, "BASE_UNSTABLE", len(window)
        if span(cuts) > max_cut:
            return False, "CUT_UNSTABLE", len(window)
        angs = [o.angle_deg for o in window]
        # circular-ish simple delta
        if max(angs) - min(angs) > max_ang:
            return False, "ANGLE_UNSTABLE", len(window)
        return True, "TEMPORAL_OK", len(window)


def np_pts(pts):
    import numpy as np

    return np.array(pts, dtype=np.float32)


def span(arr) -> float:
    import numpy as np

    return float(np.linalg.norm(arr.max(axis=0) - arr.min(axis=0)))
