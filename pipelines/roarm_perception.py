"""RoArm perception: chessboard corner detect, depth sample, target tracking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pipelines.roarm_kinematics import StereoError, cam_error, normalize_uv


@dataclass
class PointTarget:
    """Stereo target: pixel, normalized offset, depth, optional 3D in camera frame."""

    px: int
    py: int
    u: float
    v: float
    depth_m: float
    cam_err_m: float
    valid: bool
    source: str = "pixel"
    corner_idx: int = -1
    reject_reason: str = ""
    camera_x_m: Optional[float] = None
    camera_y_m: Optional[float] = None
    camera_z_m: Optional[float] = None

    def to_stereo_error(self) -> StereoError:
        return StereoError(
            u=self.u,
            v=self.v,
            depth_m=self.depth_m,
            cam_err_m=self.cam_err_m,
            px=self.px,
            py=self.py,
        )

    def to_dict(self) -> dict:
        return {
            "px": self.px,
            "py": self.py,
            "u": round(self.u, 4),
            "v": round(self.v, 4),
            "depth_m": round(self.depth_m, 4) if self.valid else None,
            "cam_err_m": round(self.cam_err_m, 4),
            "valid": self.valid,
            "source": self.source,
            "corner_idx": self.corner_idx,
            "reject_reason": self.reject_reason,
            "camera_xyz_m": (
                [round(self.camera_x_m, 4), round(self.camera_y_m, 4), round(self.camera_z_m, 4)]
                if self.camera_x_m is not None
                else None
            ),
        }


# Backward alias
GridTarget = PointTarget


def pixel_to_camera_xyz(
    px: int,
    py: int,
    depth_m: float,
    intrinsics: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    """ROS optical frame: X right, Y down, Z forward (depth along +Z)."""
    fx, fy, cx, cy = intrinsics
    z = float(depth_m)
    x = (float(px) - cx) * z / fx
    y = (float(py) - cy) * z / fy
    return x, y, z


def build_point_target(
    bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    cfg: PerceptionConfig,
    *,
    px: Optional[int] = None,
    py: Optional[int] = None,
    prev_depth: Optional[float] = None,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
    source: str = "pixel",
) -> PointTarget:
    """Build target at fixed pixel or image center."""
    h, w = bgr.shape[:2]
    if px is None or py is None:
        px, py = w // 2, h // 2
    px = int(max(0, min(w - 1, px)))
    py = int(max(0, min(h - 1, py)))
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
            source=source,
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
        source=source,
        camera_x_m=cam_x,
        camera_y_m=cam_y,
        camera_z_m=cam_z,
    )


@dataclass
class PerceptionConfig:
    chessboard_cols: int = 9
    chessboard_rows: int = 6
    corner_strategy: str = "nearest_center"  # nearest_center | bottom_left | top_right
    depth_median_radius: int = 5
    depth_valid_min_m: float = 0.05
    depth_valid_max_m: float = 2.5
    depth_jump_max_m: float = 0.4
    target_camera_distance_m: float = 0.11
    uv_deadzone: float = 0.06
    track_max_jump_px: float = 45.0
    track_lost_max_frames: int = 3
    track_reacquire_scale: float = 2.0
    # Template lock (stage 1): search patch only near last position.
    use_template_lock: bool = True
    template_patch_px: int = 64
    template_search_radius_px: int = 48
    template_reacquire_radius_px: int = 140
    template_min_score: float = 0.52
    template_spatial_penalty: float = 0.45
    template_max_jump_px: float = 40.0


@dataclass
class TargetTracker:
    """Lock grid corner on first valid frame; follow by nearest (px, py)."""

    locked: bool = False
    corner_idx: int = -1
    grid_kind: str = ""
    track_px: float = 0.0
    track_py: float = 0.0
    anchor_px: float = 0.0
    anchor_py: float = 0.0
    lost_streak: int = 0
    max_lost: int = 3
    max_jump_px: float = 45.0
    reacquire_scale: float = 2.0

    def seed_from_target(self, target: PointTarget) -> None:
        if not target.valid:
            return
        self.locked = True
        self.corner_idx = target.corner_idx
        if ":" in target.source:
            self.grid_kind = str(target.source.split(":", 1)[1])
        self.track_px = float(target.px)
        self.track_py = float(target.py)
        self.anchor_px = float(target.px)
        self.anchor_py = float(target.py)
        self.lost_streak = 0

    def note_valid(self, target: PointTarget) -> None:
        if target.valid:
            self.track_px = float(target.px)
            self.track_py = float(target.py)
            self.lost_streak = 0
            if not self.locked:
                self.seed_from_target(target)
        else:
            if self.locked:
                self.lost_streak += 1

    @property
    def is_lost(self) -> bool:
        return self.locked and self.lost_streak >= self.max_lost

    def tracking_kwargs(self) -> dict:
        if not self.locked:
            return {}
        kw = {
            "track_px": self.track_px,
            "track_py": self.track_py,
            "anchor_px": self.anchor_px,
            "anchor_py": self.anchor_py,
        }
        if self.grid_kind:
            kw["prefer_grid_kind"] = self.grid_kind
        return kw


def _pattern_size(cfg: PerceptionConfig) -> Tuple[int, int]:
    return (cfg.chessboard_cols, cfg.chessboard_rows)


def _neutralize_sky_band(bgr: np.ndarray, frac: float = 0.22) -> np.ndarray:
    """Gray out bright background above the grid so tape contours do not span full width."""
    out = bgr.copy()
    h = out.shape[0]
    out[: max(1, int(h * frac))] = (128, 128, 128)
    return out


def _pick_grid_dark_contour(
    cnts: List[np.ndarray],
    rw: int,
    rh: int,
) -> Optional[np.ndarray]:
    """Pick grid-like tape blob; skip full-width mask merged with ceiling background."""
    if not cnts:
        return None
    ranked = sorted(cnts, key=cv2.contourArea, reverse=True)
    for cnt in ranked:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw > rw * 0.88 and by < rh * 0.08:
            continue
        if bw * bh < rw * rh * 0.02:
            continue
        return cnt
    return ranked[0]


def _mask_gripper_occlusion(dark: np.ndarray) -> np.ndarray:
    """Mask gripper blob (black like tape) in bottom-center without erasing the grid."""
    out = dark.copy()
    h, w = out.shape[:2]
    y0 = int(h * 0.45)
    band = dark[y0:, :].copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    masked = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 2500 or area > 120000:
            continue
        cx = int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] * 0.5)
        if w * 0.22 < cx < w * 0.78:
            out[y0:, :][labels == i] = 0
            masked += 1
    if masked == 0:
        # Fallback: small fixed zone over fingertips only
        out[int(h * 0.68) :, int(w * 0.32) : int(w * 0.68)] = 0
    return out


def _tape_dark_mask(roi_bgr: np.ndarray, *, mask_gripper: bool = False) -> np.ndarray:
    """Black masking tape on light plywood → binary mask (255 = tape)."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = float(np.percentile(blur, 30))
    dark = ((blur < max(75.0, thresh)).astype(np.uint8)) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k3)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k9)
    if mask_gripper:
        return _mask_gripper_occlusion(dark)
    return dark


def _white_inner_from_roi(
    roi_bgr: np.ndarray,
    y_top: int,
    full_w: int,
    full_h: int,
    *,
    mask_gripper: bool,
) -> List[Tuple[float, float]]:
    """Extract inner white corners from one ROI crop."""
    rh, rw = roi_bgr.shape[:2]
    if rh < 40 or rw < 40:
        return []
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    dark = _tape_dark_mask(roi_bgr, mask_gripper=mask_gripper)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = _pick_grid_dark_contour(cnts, rw, rh)
    if cnt is None:
        return []
    bx, by, bw, bh = cv2.boundingRect(cnt)
    pad = 12
    x0, y0 = max(0, bx - pad), max(0, by - pad)
    x1, y1 = min(rw, bx + bw + pad), min(rh, by + bh + pad)
    tape = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    white = cv2.bitwise_not(tape)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    white_crop = white[y0:y1, x0:x1]
    gray_crop = gray[y0:y1, x0:x1]
    dark_crop = dark[y0:y1, x0:x1]
    ch, cw = white_crop.shape[:2]
    if ch < 30 or cw < 30:
        return []
    min_area = (cw * ch) * 0.0025
    max_area = (cw * ch) * 0.45
    candidates: List[Tuple[float, float, float]] = []
    for cnt in cv2.findContours(white_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)
        if len(approx) < 4 or len(approx) > 10:
            continue
        for pt in approx.reshape(-1, 2):
            lx, ly = float(pt[0]), float(pt[1])
            score = _white_inner_corner_score(white_crop, dark_crop, lx, ly)
            if score < 0.04:
                continue
            lx0, ly0 = lx, ly
            pts = np.array([[lx, ly]], dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.02)
            cv2.cornerSubPix(gray_crop, pts, (3, 3), (-1, -1), criteria)
            lx, ly = float(pts[0, 0]), float(pts[0, 1])
            score2 = _white_inner_corner_score(white_crop, dark_crop, lx, ly)
            if score2 < score * 0.6:
                lx, ly, score2 = lx0, ly0, score
            candidates.append((lx + x0, ly + y0 + y_top, score2))
    candidates.sort(key=lambda t: -t[2])
    merged: List[Tuple[float, float]] = []
    y_min = int(full_h * 0.20)
    rad = max(12.0, min(cw, ch) * 0.038)
    for x, y, _sc in candidates:
        if not (10 <= x < full_w - 10 and y_min <= y < full_h * 0.75):
            continue
        if any((x - mx) ** 2 + (y - my) ** 2 < rad * rad for mx, my in merged):
            continue
        merged.append((x, y))
    return merged


def _white_inner_corner_score(
    white_mask: np.ndarray,
    dark_mask: np.ndarray,
    x: float,
    y: float,
    radius: int = 14,
) -> float:
    """Inner corner: white at center, tape+white mixed in local window (rotation-invariant)."""
    h, w = white_mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    r = max(6, radius)
    if xi < r or yi < r or xi >= w - r or yi >= h - r:
        return 0.0
    if white_mask[yi, xi] < 128:
        return 0.0
    patch_w = white_mask[yi - r : yi + r + 1, xi - r : xi + r + 1]
    patch_d = dark_mask[yi - r : yi + r + 1, xi - r : xi + r + 1]
    white_frac = float((patch_w > 128).mean())
    dark_frac = float((patch_d > 128).mean())
    if white_frac < 0.30 or dark_frac < 0.12:
        return 0.0
    return white_frac * dark_frac


def detect_white_cell_inner_corners(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Inner corners of white plywood cells bordered by black tape (navigation targets)."""
    work = _neutralize_sky_band(bgr)
    h, w = work.shape[:2]
    merged: List[Tuple[float, float]] = []
    for y_bot_frac, mask_grip in ((0.72, False), (0.72, True), (0.58, True)):
        y_top = int(h * 0.02)
        y_bot = int(h * y_bot_frac)
        roi_bgr = work[y_top:y_bot]
        merged = _white_inner_from_roi(roi_bgr, y_top, w, h, mask_gripper=mask_grip)
        if len(merged) >= 4:
            break
    if len(merged) < 4:
        return None
    merged.sort(key=lambda p: (p[1], p[0]))
    return np.array(merged, dtype=np.float32).reshape(-1, 1, 2)


def detect_score_scan_inner_corners(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Fallback: score inner corners inside the grid contour (gripper-occluded frames)."""
    work = _neutralize_sky_band(bgr)
    h, w = work.shape[:2]
    y_top = int(h * 0.02)
    y_bot = int(h * 0.72)
    roi_bgr = work[y_top:y_bot]
    rh, rw = roi_bgr.shape[:2]
    if rh < 40 or rw < 40:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    dark = _tape_dark_mask(roi_bgr, mask_gripper=True)
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = _pick_grid_dark_contour(cnts, rw, rh)
    if cnt is None:
        return None

    bx, by, bw, bh = cv2.boundingRect(cnt)
    pad = 8
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(rw, bx + bw + pad)
    y1 = min(rh, by + bh + pad)

    local = np.zeros_like(dark)
    cv2.drawContours(local, [cnt], -1, 255, -1)
    local = cv2.bitwise_and(dark, local)
    tape = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    white = cv2.bitwise_not(tape)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    white_crop = white[y0:y1, x0:x1]
    dark_crop = local[y0:y1, x0:x1]
    ch, cw = white_crop.shape[:2]
    if ch < 20 or cw < 20:
        return None

    score_min = 0.11
    step = 3
    y_min = int(h * 0.20)
    merge_rad = max(16.0, min(cw, ch) * 0.035)
    scored: List[Tuple[float, float, float]] = []
    for py in range(4, ch - 4, step):
        for px in range(4, cw - 4, step):
            sc = _white_inner_corner_score(white_crop, dark_crop, float(px), float(py), radius=12)
            if sc >= score_min:
                scored.append((px + x0, py + y_top, sc))
    scored.sort(key=lambda t: -t[2])

    merged: List[Tuple[float, float]] = []
    for x, y, _sc in scored:
        if not (8 <= x < w - 8 and y_min <= y < h * 0.75):
            continue
        if any((x - mx) ** 2 + (y - my) ** 2 < merge_rad * merge_rad for mx, my in merged):
            continue
        merged.append((x, y))

    if len(merged) < 4:
        return None
    merged.sort(key=lambda p: (p[1], p[0]))
    return np.array(merged, dtype=np.float32).reshape(-1, 1, 2)


def _cross_score(mask: np.ndarray, x: float, y: float, radius: int = 12) -> float:
    """High when (x,y) sits on a tape cross (row and col both mostly tape)."""
    h, w = mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    r = max(4, radius)
    if xi < r or yi < r or xi >= w - r or yi >= h - r:
        return 0.0
    patch = mask[yi - r : yi + r + 1, xi - r : xi + r + 1]
    if patch[r, r] < 128:
        return 0.0
    row_hit = float((patch[r, :] > 128).mean())
    col_hit = float((patch[:, r] > 128).mean())
    return row_hit * col_hit


def _refine_cross_point(gray: np.ndarray, mask: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Snap to nearest strong tape cross in a small window."""
    h, w = gray.shape[:2]
    win = 14
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - win), min(w, xi + win + 1)
    y0, y1 = max(0, yi - win), min(h, yi + win + 1)
    best = (x, y)
    best_s = _cross_score(mask, x, y)
    for py in range(y0, y1, 2):
        for px in range(x0, x1, 2):
            s = _cross_score(mask, px, py)
            if s > best_s:
                best_s = s
                best = (float(px), float(py))
    if best_s < 0.35:
        return x, y
    pts = np.array([[best[0], best[1]]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    cv2.cornerSubPix(gray, pts, (7, 7), (-1, -1), criteria)
    return float(pts[0, 0]), float(pts[0, 1])


def detect_line_grid_intersections(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Detect intersections of black tape grid on light plywood."""
    work = _neutralize_sky_band(bgr)
    h, w = work.shape[:2]
    y_top = int(h * 0.02)
    y_bot = int(h * 0.72)
    roi_bgr = work[y_top:y_bot]
    rh, rw = roi_bgr.shape[:2]
    if rh < 40 or rw < 40:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    dark = _tape_dark_mask(roi_bgr, mask_gripper=True)

    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = _pick_grid_dark_contour(cnts, rw, rh)
    if cnt is None:
        return None
    bx, by, bw, bh = cv2.boundingRect(cnt)
    pad = 10
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(rw, bx + bw + pad)
    y1 = min(rh, by + bh + pad)
    crop = dark[y0:y1, x0:x1]
    crop_gray = gray[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 30 or cw < 30:
        return None

    lines = cv2.HoughLines(crop, 1, np.pi / 180, threshold=max(45, cw // 5))
    if lines is None:
        return None
    theta_ref = float(np.median(lines[:, 0, 1]))
    angle_deg = float(np.degrees(theta_ref) - 90.0)

    M = cv2.getRotationMatrix2D((cw * 0.5, ch * 0.5), angle_deg, 1.0)
    rot = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_NEAREST, borderValue=0)

    tape_w = max(8, int(cw * 0.028))
    cell = max(28, int(cw * 0.09))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (cell, tape_w))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (tape_w, cell))
    horiz = cv2.morphologyEx(rot, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(rot, cv2.MORPH_OPEN, vk)
    cross = cv2.bitwise_and(horiz, vert)
    cross = cv2.dilate(cross, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(cross, connectivity=8)
    min_area = max(20, (tape_w * tape_w) // 2)
    pts_roi: List[Tuple[float, float]] = []
    Minv = cv2.invertAffineTransform(M)

    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = centroids[i]
        p = Minv @ np.array([cx, cy, 1.0], dtype=np.float64)
        lx, ly = float(p[0]), float(p[1])
        if lx < 2 or ly < 2 or lx >= cw - 2 or ly >= ch - 2:
            continue
        gx, gy = lx + x0, ly + y0
        rx, ry = _refine_cross_point(crop_gray, crop, lx, ly)
        gx, gy = rx + x0, ry + y0
        if _cross_score(dark, gx, gy) < 0.45:
            continue
        pts_roi.append((gx, gy + y_top))

    if len(pts_roi) < 4:
        return None

    merged: List[Tuple[float, float]] = []
    y_min = int(h * 0.20)
    rad = max(20.0, min(cw, ch) * 0.07)
    for x, y in pts_roi:
        if not (8 <= x < w - 8 and y_min <= y < h * 0.70):
            continue
        if any((x - mx) ** 2 + (y - my) ** 2 < rad * rad for mx, my in merged):
            continue
        merged.append((x, y))

    if len(merged) < 4:
        return None

    merged.sort(key=lambda p: (p[1], p[0]))
    corners = np.array(merged, dtype=np.float32).reshape(-1, 1, 2)
    return corners


def detect_chessboard_corners(
    bgr: np.ndarray,
    cfg: PerceptionConfig,
) -> Optional[np.ndarray]:
    """Return Nx1x2 corner array or None (classic B&W checkerboard only)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pattern = _pattern_size(cfg)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def detect_grid_corners(
    bgr: np.ndarray,
    cfg: PerceptionConfig,
    *,
    prefer_kind: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Chessboard → white cell inner corners (tape grid) → tape crosses → score scan."""
    if prefer_kind == "white_inner":
        corners = detect_white_cell_inner_corners(bgr)
        if corners is not None:
            return corners, "white_inner"
    elif prefer_kind == "line_grid":
        corners = detect_line_grid_intersections(bgr)
        if corners is not None:
            return corners, "line_grid"
    elif prefer_kind == "score_scan":
        corners = detect_score_scan_inner_corners(bgr)
        if corners is not None:
            return corners, "score_scan"
    elif prefer_kind == "chessboard":
        corners = detect_chessboard_corners(bgr, cfg)
        if corners is not None:
            return corners, "chessboard"

    corners = detect_chessboard_corners(bgr, cfg)
    if corners is not None:
        return corners, "chessboard"
    corners = detect_white_cell_inner_corners(bgr)
    if corners is not None:
        return corners, "white_inner"
    corners = detect_line_grid_intersections(bgr)
    if corners is not None:
        return corners, "line_grid"
    corners = detect_score_scan_inner_corners(bgr)
    if corners is not None:
        return corners, "score_scan"
    return None, "none"


def _corner_index_nearest(
    corners: np.ndarray,
    px: float,
    py: float,
    max_dist_px: float,
) -> Optional[int]:
    pts = corners.reshape(-1, 2)
    d2 = (pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2
    idx = int(np.argmin(d2))
    if float(d2[idx]) > max_dist_px * max_dist_px:
        return None
    return idx


def _corner_index_for_strategy(
    corners: np.ndarray,
    img_w: int,
    img_h: int,
    strategy: str,
) -> int:
    pts = corners.reshape(-1, 2)
    cx, cy = img_w * 0.5, img_h * 0.5
    if strategy == "bottom_left":
        scores = pts[:, 0] + (img_h - pts[:, 1])
        return int(np.argmin(scores))
    if strategy == "top_right":
        scores = (img_w - pts[:, 0]) + pts[:, 1]
        return int(np.argmin(scores))
    dists = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
    return int(np.argmin(dists))


def _corner_index_depth_center(
    corners: np.ndarray,
    depth_m: Optional[np.ndarray],
    cfg: PerceptionConfig,
    img_w: int,
    img_h: int,
) -> int:
    """Pick corner near image center with valid depth (skip frame edges)."""
    pts = corners.reshape(-1, 2)
    cx, cy = img_w * 0.5, img_h * 0.5
    best_idx: Optional[int] = None
    best_dist = 1e18
    for i, (px_f, py_f) in enumerate(pts):
        px, py = int(round(px_f)), int(round(py_f))
        if px < img_w * 0.12 or px > img_w * 0.88 or py < img_h * 0.12 or py > img_h * 0.92:
            continue
        depth_val, _reject = sample_depth_median(
            depth_m, px, py, cfg.depth_median_radius, cfg
        )
        if depth_val is None:
            continue
        dist = (px_f - cx) ** 2 + (py_f - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    if best_idx is not None:
        return best_idx
    return _corner_index_for_strategy(corners, img_w, img_h, "nearest_center")


def sample_depth_median(
    depth_m: Optional[np.ndarray],
    px: int,
    py: int,
    radius: int,
    cfg: PerceptionConfig,
    prev_depth: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    if depth_m is None:
        return None, "no_depth_frame"
    h, w = depth_m.shape[:2]
    px = int(max(0, min(w - 1, px)))
    py = int(max(0, min(h - 1, py)))
    r = max(1, radius)
    y0, y1 = max(0, py - r), min(h, py + r + 1)
    x0, x1 = max(0, px - r), min(w, px + r + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > cfg.depth_valid_min_m) & (patch < cfg.depth_valid_max_m)]
    if valid.size < 3:
        return None, "insufficient_valid_depth"
    med = float(np.median(valid))
    if prev_depth is not None and abs(med - prev_depth) > cfg.depth_jump_max_m:
        return None, "depth_jump"
    return med, ""


def build_grid_target(
    bgr: np.ndarray,
    depth_m: Optional[np.ndarray],
    cfg: PerceptionConfig,
    *,
    prev_depth: Optional[float] = None,
    forced_corner_idx: Optional[int] = None,
    track_px: Optional[float] = None,
    track_py: Optional[float] = None,
    anchor_px: Optional[float] = None,
    anchor_py: Optional[float] = None,
    prefer_grid_kind: Optional[str] = None,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
) -> PointTarget:
    h, w = bgr.shape[:2]
    corners, grid_kind = detect_grid_corners(bgr, cfg, prefer_kind=prefer_grid_kind)
    if corners is None:
        return PointTarget(
            px=w // 2,
            py=h // 2,
            u=0.0,
            v=0.0,
            depth_m=0.0,
            cam_err_m=999.0,
            valid=False,
            source="grid",
            corner_idx=-1,
            reject_reason="no_grid",
        )

    idx: Optional[int] = None
    if track_px is not None and track_py is not None:
        idx = _corner_index_nearest(corners, track_px, track_py, cfg.track_max_jump_px)
        if idx is None and anchor_px is not None and anchor_py is not None:
            reacquire = cfg.track_max_jump_px * cfg.track_reacquire_scale
            idx = _corner_index_nearest(corners, anchor_px, anchor_py, reacquire)
        if idx is None:
            px_i, py_i = int(round(track_px)), int(round(track_py))
            return PointTarget(
                px=px_i,
                py=py_i,
                u=0.0,
                v=0.0,
                depth_m=0.0,
                cam_err_m=999.0,
                valid=False,
                source=f"grid:{grid_kind}",
                corner_idx=-1,
                reject_reason="track_lost",
            )
    elif forced_corner_idx is not None and 0 <= forced_corner_idx < len(corners):
        idx = forced_corner_idx
    elif cfg.corner_strategy == "depth_center":
        idx = _corner_index_depth_center(corners, depth_m, cfg, w, h)
    else:
        idx = _corner_index_for_strategy(corners, w, h, cfg.corner_strategy)

    px_f, py_f = corners[idx, 0]
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
            source="grid",
            corner_idx=idx,
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
        source=f"grid:{grid_kind}",
        corner_idx=idx,
        camera_x_m=cam_x,
        camera_y_m=cam_y,
        camera_z_m=cam_z,
    )


def episode_success(
    target: PointTarget,
    cfg: PerceptionConfig,
    *,
    initial_depth_m: float,
    success_cfg: dict,
) -> Tuple[bool, str]:
    if not target.valid:
        return False, "invalid_target"
    depth_max = float(success_cfg.get("depth_max_m", 0.22))
    cam_err_max = float(success_cfg.get("cam_err_max_m", 0.08))
    progress_min = float(success_cfg.get("depth_progress_min_m", 0.18))

    if target.depth_m >= depth_max:
        return False, f"depth_too_far:{target.depth_m:.3f}"
    if abs(target.cam_err_m) >= cam_err_max:
        return False, f"cam_err:{target.cam_err_m:.3f}"
    if abs(target.u) > cfg.uv_deadzone or abs(target.v) > cfg.uv_deadzone:
        return False, f"uv_outside_deadzone:u={target.u:.3f},v={target.v:.3f}"
    if initial_depth_m - target.depth_m < progress_min:
        return False, f"insufficient_depth_progress:{initial_depth_m - target.depth_m:.3f}"
    return True, "success"


def list_corner_candidates(bgr: np.ndarray, cfg: PerceptionConfig) -> List[Tuple[int, int, int]]:
    """Return [(idx, px, py), ...] for mastery drill."""
    corners, _kind = detect_grid_corners(bgr, cfg)
    if corners is None:
        return []
    out = []
    for i, pt in enumerate(corners.reshape(-1, 2)):
        out.append((i, int(round(pt[0])), int(round(pt[1]))))
    return out
