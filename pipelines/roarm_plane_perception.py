"""Stereo plane tracking on tarp/grid: RANSAC surface + corner on plane → PointTarget."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipelines.roarm_kinematics import cam_error, normalize_uv
from pipelines.roarm_perception import (
    PerceptionConfig,
    PointTarget,
    TargetTracker,
    _corner_index_depth_center,
    _corner_index_for_strategy,
    _corner_index_nearest,
    detect_grid_corners,
    pixel_to_camera_xyz,
)


@dataclass
class PlanePerceptionConfig:
    sample_step_px: int = 6
    ransac_iters: int = 120
    ransac_thresh_m: float = 0.028
    min_points: int = 28
    min_inlier_frac: float = 0.52
    smooth_alpha: float = 0.38
    mask_sky_frac: float = 0.18
    mask_bottom_frac: float = 0.78

    @classmethod
    def from_yaml(cls, perc: dict) -> "PlanePerceptionConfig":
        plane = perc.get("plane") or {}
        return cls(
            sample_step_px=int(plane.get("sample_step_px", 6)),
            ransac_iters=int(plane.get("ransac_iters", 120)),
            ransac_thresh_m=float(plane.get("ransac_thresh_m", 0.028)),
            min_points=int(plane.get("min_points", 72)),
            min_inlier_frac=float(plane.get("min_inlier_frac", 0.52)),
            smooth_alpha=float(plane.get("smooth_alpha", 0.38)),
            mask_sky_frac=float(plane.get("mask_sky_frac", 0.18)),
            mask_bottom_frac=float(plane.get("mask_bottom_frac", 0.78)),
        )


@dataclass
class PlaneModel:
    """Unit normal (into scene, +Z preferred) and Hessian offset n·X = d."""

    normal: np.ndarray
    d: float
    rmse_m: float
    inlier_frac: float

    def to_dict(self) -> dict:
        n = self.normal
        return {
            "normal": [round(float(n[0]), 4), round(float(n[1]), 4), round(float(n[2]), 4)],
            "d_m": round(float(self.d), 4),
            "rmse_m": round(float(self.rmse_m), 4),
            "inlier_frac": round(float(self.inlier_frac), 3),
        }


@dataclass
class PlaneTracker:
    """Track fitted plane + grid corner pixel between frames."""

    locked: bool = False
    normal: Optional[np.ndarray] = None
    d: float = 0.0
    corner_idx: int = -1
    grid_kind: str = ""
    track_px: float = 0.0
    track_py: float = 0.0
    anchor_px: float = 0.0
    anchor_py: float = 0.0
    lost_streak: int = 0
    max_lost: int = 5

    def seed_from_target(self, target: PointTarget, plane: PlaneModel) -> None:
        if not target.valid:
            return
        self.locked = True
        self.normal = plane.normal.copy()
        self.d = float(plane.d)
        self.corner_idx = target.corner_idx
        if ":" in target.source:
            self.grid_kind = str(target.source.split(":", 1)[1])
        self.track_px = float(target.px)
        self.track_py = float(target.py)
        self.anchor_px = float(target.px)
        self.anchor_py = float(target.py)
        self.lost_streak = 0

    def note_valid(self, target: PointTarget, plane: Optional[PlaneModel]) -> None:
        if target.valid and plane is not None:
            self.track_px = float(target.px)
            self.track_py = float(target.py)
            self.lost_streak = 0
            if self.locked and self.normal is not None:
                a = 0.38
                n = a * plane.normal + (1.0 - a) * self.normal
                n_norm = float(np.linalg.norm(n))
                if n_norm > 1e-6:
                    self.normal = n / n_norm
                self.d = a * plane.d + (1.0 - a) * self.d
            elif not self.locked:
                self.seed_from_target(target, plane)
        else:
            if self.locked:
                self.lost_streak += 1

    @property
    def is_lost(self) -> bool:
        return self.locked and self.lost_streak >= self.max_lost

    def tracking_kwargs(self) -> dict:
        if not self.locked:
            return {}
        kw: Dict[str, object] = {
            "track_px": self.track_px,
            "track_py": self.track_py,
            "anchor_px": self.anchor_px,
            "anchor_py": self.anchor_py,
        }
        if self.grid_kind:
            kw["prefer_grid_kind"] = self.grid_kind
        return kw


def _import_mask_helpers():
    from pipelines.roarm_perception import (
        _mask_gripper_occlusion,
        _neutralize_sky_band,
        _pick_grid_dark_contour,
        _tape_dark_mask,
    )

    return _neutralize_sky_band, _tape_dark_mask, _pick_grid_dark_contour, _mask_gripper_occlusion


def default_intrinsics(img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Approximate pinhole when CameraInfo not yet received."""
    fx = fy = float(max(img_w, img_h)) * 0.515
    return fx, fy, img_w * 0.5, img_h * 0.5


def resolve_intrinsics(
    intrinsics: Optional[Tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    if intrinsics is not None and intrinsics[0] > 0 and intrinsics[1] > 0:
        return intrinsics
    return default_intrinsics(img_w, img_h)


def build_tarp_mask(bgr: np.ndarray, cfg: PlanePerceptionConfig) -> Optional[np.ndarray]:
    """Mask white grid cells inside black tape contour (tarp / plywood board)."""
    neutralize, tape_dark, pick_contour, _mask_grip = _import_mask_helpers()
    work = neutralize(bgr)
    h, w = work.shape[:2]
    y_top = int(h * 0.02)
    y_bot = int(h * cfg.mask_bottom_frac)
    roi = work[y_top:y_bot]
    rh, rw = roi.shape[:2]
    if rh < 40 or rw < 40:
        return None

    dark = tape_dark(roi, mask_gripper=True)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = pick_contour(cnts, rw, rh)
    if cnt is None:
        return None

    bx, by, bw, bh = cv2.boundingRect(cnt)
    pad = max(8, int(min(bw, bh) * 0.06))
    x0, y0 = max(0, bx - pad), max(0, by - pad)
    x1, y1 = min(rw, bx + bw + pad), min(rh, by + bh + pad)

    local = np.zeros_like(dark)
    cv2.drawContours(local, [cnt], -1, 255, -1)
    tape = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    white = cv2.bitwise_not(tape)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y_top + y0 : y_top + y1, x0:x1] = white[y0:y1, x0:x1]
    mask[: int(h * cfg.mask_sky_frac), :] = 0
    if int(mask.sum()) < 500:
        return None
    return mask


def build_grid_roi_mask(
    bgr: np.ndarray,
    corners: np.ndarray,
    *,
    margin_frac: float = 0.12,
) -> np.ndarray:
    """Tight mask around detected grid corners (avoids gripper-dominated tarp mask)."""
    h, w = bgr.shape[:2]
    pts = corners.reshape(-1, 2)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    mw = max(24.0, (x1 - x0) * margin_frac)
    mh = max(24.0, (y1 - y0) * margin_frac)
    ix0 = int(max(0, x0 - mw))
    iy0 = int(max(0, y0 - mh))
    ix1 = int(min(w, x1 + mw))
    iy1 = int(min(h, y1 + mh))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[iy0:iy1, ix0:ix1] = 255
    return mask


def depth_gate_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    *,
    z_min_m: float = 0.32,
    z_max_m: float = 1.9,
) -> np.ndarray:
    """Keep mask pixels with valid far depth (drop gripper / near clutter)."""
    valid = np.isfinite(depth_m) & (depth_m > z_min_m) & (depth_m < z_max_m)
    out = np.zeros_like(mask)
    out[(mask > 128) & valid] = 255
    return out


def depth_mask_to_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: Tuple[float, float, float, float],
    cfg: PlanePerceptionConfig,
    perc: PerceptionConfig,
) -> np.ndarray:
    """Subsample valid depth pixels inside mask → Nx3 camera-frame points."""
    fx, fy, cx, cy = intrinsics
    h, w = depth_m.shape[:2]
    step = max(2, cfg.sample_step_px)
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    m = mask[grid_y, grid_x] > 128
    z = depth_m[grid_y, grid_x]
    valid = (
        m
        & np.isfinite(z)
        & (z > perc.depth_valid_min_m)
        & (z < perc.depth_valid_max_m)
    )
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    px = grid_x[valid].astype(np.float64)
    py = grid_y[valid].astype(np.float64)
    pz = z[valid].astype(np.float64)
    x = (px - cx) * pz / fx
    y = (py - cy) * pz / fy
    return np.stack([x, y, pz], axis=1)


def fit_plane_ransac(
    points: np.ndarray,
    cfg: PlanePerceptionConfig,
) -> Optional[PlaneModel]:
    """RANSAC plane fit; prefer wall-like surfaces (normal ≈ +Z)."""
    n_pts = len(points)
    min_pts = min(cfg.min_points, max(12, n_pts // 3))
    if n_pts < min_pts:
        return None
    z_med = float(np.median(points[:, 2]))
    keep = (points[:, 2] > max(0.08, z_med - 0.25)) & (points[:, 2] < z_med + 0.45)
    points = points[keep]
    n_pts = len(points)
    if n_pts < min_pts:
        return None

    best_score = -1.0
    best_inliers: Optional[np.ndarray] = None
    best_n: Optional[np.ndarray] = None
    best_d: float = 0.0

    def _try_ransac(pts: np.ndarray, wall_prior: bool) -> Optional[PlaneModel]:
        nonlocal best_score, best_inliers, best_n, best_d
        n_try = len(pts)
        if n_try < min_pts:
            return None
        for _ in range(cfg.ransac_iters):
            idx = rng.choice(n_try, 3, replace=False)
            p0, p1, p2 = pts[idx]
            n = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(n))
            if norm < 1e-7:
                continue
            n = n / norm
            if n[2] < 0:
                n = -n
            if wall_prior and n[2] < 0.12:
                continue
            d = float(np.dot(n, p0))
            if d < 0.08:
                continue
            dists = np.abs(pts @ n - d)
            inliers = dists < cfg.ransac_thresh_m
            n_in = int(inliers.sum())
            if n_in < max(12, cfg.min_points // 4):
                continue
            score = float(n_in) * (0.35 + float(n[2]))
            if score > best_score:
                best_score = score
                best_inliers = inliers
                best_n = n
                best_d = d
        return None

    rng = np.random.default_rng()
    _try_ransac(points, wall_prior=True)
    if best_inliers is None:
        best_score = -1.0
        _try_ransac(points, wall_prior=False)

    if best_inliers is None:
        return None

    inlier_frac = float(best_inliers.mean())
    if inlier_frac < cfg.min_inlier_frac:
        return None

    in_pts = points[best_inliers]

    centroid = in_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(in_pts - centroid)
    n = vh[2].astype(np.float64)
    if n[2] < 0:
        n = -n
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    d = float(np.dot(n, centroid))
    if d < 0.08:
        return None
    dists = np.abs(in_pts @ n - d)
    rmse = float(np.sqrt(np.mean(dists * dists)))
    return PlaneModel(normal=n, d=d, rmse_m=rmse, inlier_frac=inlier_frac)


def ray_plane_intersection(
    px: float,
    py: float,
    intrinsics: Tuple[float, float, float, float],
    plane: PlaneModel,
) -> Optional[np.ndarray]:
    """Intersect pixel ray with plane; return 3D point in camera frame."""
    fx, fy, cx, cy = intrinsics
    direction = np.array([(px - cx) / fx, (py - cy) / fy, 1.0], dtype=np.float64)
    denom = float(np.dot(plane.normal, direction))
    if abs(denom) < 1e-6:
        return None
    t = plane.d / denom
    if t < perc_valid_min_t():
        return None
    hit = direction * t
    if hit[2] < perc_valid_min_t():
        return None
    return hit


def perc_valid_min_t() -> float:
    return 0.05


def camera_to_plane_distance_m(plane: PlaneModel) -> float:
    """Distance from camera origin to plane along normal."""
    return abs(float(plane.d))


def _pick_corner_idx(
    corners: np.ndarray,
    depth_m: np.ndarray,
    perc: PerceptionConfig,
    w: int,
    h: int,
    *,
    forced_corner_idx: Optional[int],
    track_px: Optional[float],
    track_py: Optional[float],
    anchor_px: Optional[float],
    anchor_py: Optional[float],
) -> Tuple[Optional[int], str]:
    if track_px is not None and track_py is not None:
        idx = _corner_index_nearest(corners, track_px, track_py, perc.track_max_jump_px)
        if idx is None and anchor_px is not None and anchor_py is not None:
            reacquire = perc.track_max_jump_px * perc.track_reacquire_scale
            idx = _corner_index_nearest(corners, anchor_px, anchor_py, reacquire)
        if idx is None:
            return None, "track_lost"
        return idx, ""

    if forced_corner_idx is not None and 0 <= forced_corner_idx < len(corners):
        return forced_corner_idx, ""

    if perc.corner_strategy == "depth_center":
        return _corner_index_depth_center(corners, depth_m, perc, w, h), ""
    return _corner_index_for_strategy(corners, w, h, perc.corner_strategy), ""


def build_plane_target(
    bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    perc: PerceptionConfig,
    plane_cfg: PlanePerceptionConfig,
    *,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
    forced_corner_idx: Optional[int] = None,
    track_px: Optional[float] = None,
    track_py: Optional[float] = None,
    anchor_px: Optional[float] = None,
    anchor_py: Optional[float] = None,
    prefer_grid_kind: Optional[str] = None,
    prev_plane: Optional[PlaneModel] = None,
) -> Tuple[PointTarget, Dict[str, object]]:
    """Fit tarp plane from stereo depth, pick grid corner, project onto plane."""
    h, w = bgr.shape[:2]
    meta: Dict[str, object] = {"plane": None, "mask_pixels": 0, "cloud_points": 0}

    if depth_m is None:
        return (
            PointTarget(
                px=w // 2,
                py=h // 2,
                u=0.0,
                v=0.0,
                depth_m=0.0,
                cam_err_m=999.0,
                valid=False,
                source="plane",
                reject_reason="no_depth_frame",
            ),
            meta,
        )
    if intrinsics is None:
        intrinsics = default_intrinsics(w, h)
    else:
        intrinsics = resolve_intrinsics(intrinsics, w, h)

    if intrinsics is None:
        intrinsics = default_intrinsics(w, h)
    else:
        intrinsics = resolve_intrinsics(intrinsics, w, h)

    corners, grid_kind = detect_grid_corners(bgr, perc, prefer_kind=prefer_grid_kind)
    if corners is None:
        return (
            PointTarget(
                px=w // 2,
                py=h // 2,
                u=0.0,
                v=0.0,
                depth_m=0.0,
                cam_err_m=999.0,
                valid=False,
                source="plane",
                reject_reason="no_grid_corner",
            ),
            meta,
        )

    idx, reject = _pick_corner_idx(
        corners,
        depth_m,
        perc,
        w,
        h,
        forced_corner_idx=forced_corner_idx,
        track_px=track_px,
        track_py=track_py,
        anchor_px=anchor_px,
        anchor_py=anchor_py,
    )
    if idx is None:
        px_i = int(round(track_px or w // 2))
        py_i = int(round(track_py or h // 2))
        return (
            PointTarget(
                px=px_i,
                py=py_i,
                u=0.0,
                v=0.0,
                depth_m=0.0,
                cam_err_m=999.0,
                valid=False,
                source=f"plane:{grid_kind}",
                corner_idx=-1,
                reject_reason=reject or "track_lost",
            ),
            meta,
        )

    px_f, py_f = corners[idx, 0]
    px, py = int(round(px_f)), int(round(py_f))
    u, v = normalize_uv(px, py, w, h)

    from pipelines.roarm_perception import sample_depth_median

    depth_corner, depth_reject = sample_depth_median(
        depth_m, px, py, perc.depth_median_radius, perc
    )

    z_lo = 0.12
    z_hi = 1.9
    if depth_corner is not None:
        z_lo = max(0.10, float(depth_corner) - 0.18)
        z_hi = min(2.0, float(depth_corner) + 0.30)

    roi_mask = build_grid_roi_mask(bgr, corners)
    tarp_mask = build_tarp_mask(bgr, plane_cfg)
    if tarp_mask is not None:
        roi_mask = cv2.bitwise_and(roi_mask, tarp_mask)
    fit_mask = depth_gate_mask(roi_mask, depth_m, z_min_m=z_lo, z_max_m=z_hi)
    meta["mask_pixels"] = int((fit_mask > 128).sum())
    meta["depth_gate_m"] = [round(z_lo, 3), round(z_hi, 3)]

    points = depth_mask_to_points(depth_m, fit_mask, intrinsics, plane_cfg, perc)
    meta["cloud_points"] = int(len(points))
    plane = fit_plane_ransac(points, plane_cfg)
    plane_mode = "ransac"
    if plane is None and prev_plane is not None and prev_plane.normal is not None:
        plane = prev_plane
        plane_mode = "tracked"
    if plane is not None and prev_plane is not None and prev_plane.normal is not None:
        if float(np.dot(plane.normal, prev_plane.normal)) < 0:
            plane = PlaneModel(
                normal=-plane.normal,
                d=-plane.d,
                rmse_m=plane.rmse_m,
                inlier_frac=plane.inlier_frac,
            )

    if plane is not None:
        meta["plane"] = plane.to_dict()
        meta["plane_mode"] = plane_mode
        hit = ray_plane_intersection(px_f, py_f, intrinsics, plane)
        if hit is not None:
            depth_val = float(hit[2])
            plane_dist = camera_to_plane_distance_m(plane)
            cerr_plane = cam_error(plane_dist, perc.target_camera_distance_m)
            cam_x, cam_y, cam_z = float(hit[0]), float(hit[1]), float(hit[2])
            meta["plane_dist_m"] = round(plane_dist, 4)
            meta["point_on_plane_z_m"] = round(depth_val, 4)
            return (
                PointTarget(
                    px=px,
                    py=py,
                    u=u,
                    v=v,
                    depth_m=depth_val,
                    cam_err_m=cerr_plane,
                    valid=True,
                    source=f"plane:{grid_kind}",
                    corner_idx=idx,
                    camera_x_m=cam_x,
                    camera_y_m=cam_y,
                    camera_z_m=cam_z,
                ),
                meta,
            )

    if depth_corner is not None:
        meta["plane_mode"] = "corner_depth_fallback"
        cerr = cam_error(depth_corner, perc.target_camera_distance_m)
        cam_x, cam_y, cam_z = pixel_to_camera_xyz(px, py, depth_corner, intrinsics)
        return (
            PointTarget(
                px=px,
                py=py,
                u=u,
                v=v,
                depth_m=depth_corner,
                cam_err_m=cerr,
                valid=True,
                source=f"plane:{grid_kind}:fallback",
                corner_idx=idx,
                camera_x_m=cam_x,
                camera_y_m=cam_y,
                camera_z_m=cam_z,
            ),
            meta,
        )

    return (
        PointTarget(
            px=px,
            py=py,
            u=u,
            v=v,
            depth_m=0.0,
            cam_err_m=999.0,
            valid=False,
            source=f"plane:{grid_kind}",
            corner_idx=idx,
            reject_reason=depth_reject or "plane_fit_failed",
        ),
        meta,
    )
