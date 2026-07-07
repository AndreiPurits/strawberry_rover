"""Template lock on one grid corner — local search near last (px, py), no full-frame re-pick."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from pipelines.roarm_kinematics import cam_error, normalize_uv
from pipelines.roarm_perception import (
    PerceptionConfig,
    PointTarget,
    detect_grid_corners,
    pixel_to_camera_xyz,
    sample_depth_median,
    _corner_index_depth_center,
    _corner_index_for_strategy,
)


@dataclass
class TemplateCornerTracker:
    """Lock grayscale patch on first corner; track with matchTemplate in local ROI."""

    locked: bool = False
    anchor_px: float = 0.0
    anchor_py: float = 0.0
    track_px: float = 0.0
    track_py: float = 0.0
    corner_idx: int = -1
    grid_kind: str = ""
    patch: Optional[np.ndarray] = None
    patch_half: int = 32
    search_radius_px: int = 48
    reacquire_radius_px: int = 140
    min_score: float = 0.55
    spatial_penalty: float = 0.45
    max_jump_px: float = 40.0
    lost_streak: int = 0
    max_lost: int = 6
    last_score: float = 0.0

    @classmethod
    def from_config(cls, cfg: PerceptionConfig) -> "TemplateCornerTracker":
        return cls(
            patch_half=max(12, int(cfg.template_patch_px) // 2),
            search_radius_px=int(cfg.template_search_radius_px),
            reacquire_radius_px=int(cfg.template_reacquire_radius_px),
            min_score=float(cfg.template_min_score),
            spatial_penalty=float(cfg.template_spatial_penalty),
            max_jump_px=float(cfg.template_max_jump_px),
            max_lost=int(cfg.track_lost_max_frames),
        )

    def _extract_patch(self, gray: np.ndarray, px: float, py: float) -> Optional[np.ndarray]:
        h, w = gray.shape[:2]
        cx, cy = int(round(px)), int(round(py))
        hh = self.patch_half
        x0, x1 = cx - hh, cx + hh
        y0, y1 = cy - hh, cy + hh
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        patch = gray[y0:y1, x0:x1].copy()
        if patch.size < 16 or patch.shape[0] < 8 or patch.shape[1] < 8:
            return None
        return patch

    def lock_at(
        self,
        bgr: np.ndarray,
        px: float,
        py: float,
        *,
        corner_idx: int = -1,
        grid_kind: str = "",
    ) -> bool:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        patch = self._extract_patch(gray, px, py)
        if patch is None:
            return False
        self.patch = patch
        self.anchor_px = float(px)
        self.anchor_py = float(py)
        self.track_px = float(px)
        self.track_py = float(py)
        self.corner_idx = int(corner_idx)
        self.grid_kind = str(grid_kind or "")
        self.locked = True
        self.lost_streak = 0
        self.last_score = 1.0
        return True

    def seed_from_target(self, target: PointTarget, bgr: np.ndarray) -> bool:
        if not target.valid:
            return False
        kind = ""
        if ":" in target.source:
            kind = str(target.source.split(":", 1)[1])
        return self.lock_at(
            bgr,
            float(target.px),
            float(target.py),
            corner_idx=int(target.corner_idx),
            grid_kind=kind,
        )

    def _search_center(self) -> Tuple[float, float]:
        return self.track_px, self.track_py

    def track_frame(self, bgr: np.ndarray, *, widen: bool = False) -> Tuple[Optional[float], Optional[float], float]:
        if not self.locked or self.patch is None:
            return None, None, 0.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ph, pw = self.patch.shape[:2]
        cx, cy = self._search_center()
        radius = self.reacquire_radius_px if widen else self.search_radius_px
        icx, icy = int(round(cx)), int(round(cy))
        h, w = gray.shape
        x0 = max(0, icx - radius)
        y0 = max(0, icy - radius)
        x1 = min(w, icx + radius + pw)
        y1 = min(h, icy + radius + ph)
        roi = gray[y0:y1, x0:x1]
        if roi.shape[0] < ph or roi.shape[1] < pw:
            return None, None, 0.0
        res = cv2.matchTemplate(roi, self.patch, cv2.TM_CCOEFF_NORMED)

        # Spatial prior: prefer the match nearest to last position so an identical
        # neighbouring corner cannot steal the lock. Penalize distance from expected.
        exp_x = float(icx - x0)  # expected patch-center x in roi coords
        exp_y = float(icy - y0)
        rh, rw = res.shape
        ys, xs = np.mgrid[0:rh, 0:rw]
        cand_x = xs.astype(np.float32) + pw * 0.5
        cand_y = ys.astype(np.float32) + ph * 0.5
        d2 = (cand_x - exp_x) ** 2 + (cand_y - exp_y) ** 2
        sigma = max(8.0, radius * 0.6)
        weight = np.exp(-d2 / (2.0 * sigma * sigma))  # 1 at center → 0 far
        scored = res - self.spatial_penalty * (1.0 - weight)

        loc = np.unravel_index(int(np.argmax(scored)), scored.shape)
        corr = float(res[loc])  # report true correlation, not penalized score
        px = float(x0 + int(loc[1]) + pw * 0.5)
        py = float(y0 + int(loc[0]) + ph * 0.5)

        # Hard guard against a big jump on the narrow (non-widened) search.
        if not widen:
            jump = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if jump > self.max_jump_px:
                return None, None, corr
        return px, py, corr

    def note_frame(self, bgr: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
        px, py, score = self.track_frame(bgr, widen=False)
        if px is None or score < self.min_score:
            px2, py2, score2 = self.track_frame(bgr, widen=True)
            if px2 is not None and score2 >= self.min_score:
                px, py, score = px2, py2, score2
            else:
                self.lost_streak += 1
                self.last_score = float(score2 if px2 is None else score)
                return None, None, self.last_score
        self.track_px = float(px)
        self.track_py = float(py)
        self.lost_streak = 0
        self.last_score = float(score)
        return px, py, score

    @property
    def is_lost(self) -> bool:
        return self.locked and self.lost_streak >= self.max_lost

    def tracking_kwargs(self) -> Dict[str, float]:
        if not self.locked:
            return {}
        return {
            "track_px": self.track_px,
            "track_py": self.track_py,
            "anchor_px": self.anchor_px,
            "anchor_py": self.anchor_py,
            "template_locked": 1.0,
            "template_score": round(float(self.last_score), 3),
        }


def _pick_initial_corner(
    bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    cfg: PerceptionConfig,
    *,
    forced_corner_idx: Optional[int],
) -> Tuple[Optional[np.ndarray], str, Optional[int]]:
    corners, grid_kind = detect_grid_corners(bgr, cfg)
    if corners is None:
        return None, "none", None
    h, w = bgr.shape[:2]
    if forced_corner_idx is not None and 0 <= forced_corner_idx < len(corners):
        idx = forced_corner_idx
    elif cfg.corner_strategy == "depth_center":
        idx = _corner_index_depth_center(corners, depth_m, cfg, w, h)
    else:
        idx = _corner_index_for_strategy(corners, w, h, cfg.corner_strategy)
    return corners, grid_kind, idx


def build_template_target(
    bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    cfg: PerceptionConfig,
    tracker: TemplateCornerTracker,
    *,
    prev_depth: Optional[float] = None,
    forced_corner_idx: Optional[int] = None,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
) -> PointTarget:
    """Track one corner via template lock; initial detect only when not locked."""
    h, w = bgr.shape[:2]
    source_kind = tracker.grid_kind or "template"

    if not tracker.locked:
        corners, grid_kind, idx = _pick_initial_corner(
            bgr, depth_m, cfg, forced_corner_idx=forced_corner_idx
        )
        if corners is None or idx is None:
            return PointTarget(
                px=w // 2,
                py=h // 2,
                u=0.0,
                v=0.0,
                depth_m=0.0,
                cam_err_m=999.0,
                valid=False,
                source="template",
                corner_idx=-1,
                reject_reason="no_grid_corner",
            )
        px_f, py_f = corners[idx, 0]
        if not tracker.lock_at(bgr, px_f, py_f, corner_idx=idx, grid_kind=grid_kind):
            return PointTarget(
                px=int(round(px_f)),
                py=int(round(py_f)),
                u=0.0,
                v=0.0,
                depth_m=0.0,
                cam_err_m=999.0,
                valid=False,
                source=f"template:{grid_kind}",
                corner_idx=idx,
                reject_reason="template_lock_failed",
            )
        source_kind = grid_kind

    if tracker.is_lost:
        return PointTarget(
            px=int(round(tracker.track_px)),
            py=int(round(tracker.track_py)),
            u=0.0,
            v=0.0,
            depth_m=0.0,
            cam_err_m=999.0,
            valid=False,
            source=f"template:{source_kind}",
            corner_idx=tracker.corner_idx,
            reject_reason="template_lost",
        )

    px_f, py_f, score = tracker.note_frame(bgr)
    if px_f is None:
        return PointTarget(
            px=int(round(tracker.track_px)),
            py=int(round(tracker.track_py)),
            u=0.0,
            v=0.0,
            depth_m=0.0,
            cam_err_m=999.0,
            valid=False,
            source=f"template:{source_kind}",
            corner_idx=tracker.corner_idx,
            reject_reason=f"template_low_score:{tracker.last_score:.2f}",
        )

    px, py = int(round(px_f)), int(round(py_f))
    u, v = normalize_uv(px, py, w, h)
    depth_val, reject = sample_depth_median(
        depth_m, px, py, cfg.depth_median_radius, cfg, prev_depth=prev_depth
    )
    if depth_val is None:
        return PointTarget(
            px=px,
            py=py,
            u=u,
            v=v,
            depth_m=0.0,
            cam_err_m=999.0,
            valid=False,
            source=f"template:{source_kind}",
            corner_idx=tracker.corner_idx,
            reject_reason=reject or "invalid_depth",
        )

    cerr = cam_error(depth_val, cfg.target_camera_distance_m)
    cam_x = cam_y = cam_z = None
    if intrinsics is not None:
        cam_x, cam_y, cam_z = pixel_to_camera_xyz(px, py, depth_val, intrinsics)
    return PointTarget(
        px=px,
        py=py,
        u=u,
        v=v,
        depth_m=depth_val,
        cam_err_m=cerr,
        valid=True,
        source=f"template:{source_kind}",
        corner_idx=tracker.corner_idx,
        camera_x_m=cam_x,
        camera_y_m=cam_y,
        camera_z_m=cam_z,
    )
