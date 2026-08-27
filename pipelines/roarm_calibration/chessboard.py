"""Metric chessboard pose estimation in the RealSense color optical frame."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .se3 import RigidTransform


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    frame: str = "stereo_camera_color_optical_frame"
    rectified: bool = True

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("camera_matrix must be finite 3x3")
        if not np.isfinite(distortion).all():
            raise ValueError("distortion must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        object.__setattr__(self, "camera_matrix", matrix.copy())
        object.__setattr__(self, "distortion", distortion.copy())

    @property
    def solvepnp_distortion(self) -> np.ndarray:
        # image_rect_raw is already rectified. Feeding the raw lens distortion a
        # second time would bias the board pose.
        return np.zeros_like(self.distortion) if self.rectified else self.distortion


@dataclass(frozen=True)
class ChessboardDetection:
    transform_camera_board: RigidTransform
    corners_px: np.ndarray
    reprojection_rmse_px: float


def canonicalize_corner_order(corners_px: np.ndarray) -> np.ndarray:
    """Resolve the checkerboard's 180-degree indexing ambiguity.

    The physical board has no asymmetric marker. Within the approved RoArm
    calibration envelope (camera never rolls through 90 degrees), the
    lower-right image corner is a stable canonical board origin. OpenCV SB can
    otherwise return the exact same grid in reverse order on an occasional
    frame, producing a valid reprojection error but a 180-degree board-frame
    flip that corrupts hand-eye calibration.
    """
    corners = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    if len(corners) < 2:
        raise ValueError("at least two corners are required")
    if float(np.sum(corners[0])) < float(np.sum(corners[-1])):
        corners = corners[::-1].copy()
    return corners


def object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    if cols < 3 or rows < 3:
        raise ValueError("cols and rows are inner-corner counts and must be >= 3")
    if not np.isfinite(square_size_mm) or square_size_mm <= 0.0:
        raise ValueError("exact positive square_size_mm is required")
    points = np.zeros((rows * cols, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] = grid * float(square_size_mm)
    return points


def detect_chessboard(
    image_bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    cols: int,
    rows: int,
    square_size_mm: float,
) -> Optional[ChessboardDetection]:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected BGR image")
    if image.shape[1] != intrinsics.width or image.shape[0] != intrinsics.height:
        raise ValueError("image dimensions do not match CameraInfo")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern = (int(cols), int(rows))
    corners = None
    found = False
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                60,
                1e-4,
            )
            corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
    if not found or corners is None:
        return None

    board_points = object_points(cols, rows, square_size_mm)
    image_points = canonicalize_corner_order(corners)
    ok, rvec, tvec = cv2.solvePnP(
        board_points,
        image_points,
        intrinsics.camera_matrix,
        intrinsics.solvepnp_distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    transform = RigidTransform.from_rt(
        intrinsics.frame, "calibration_board", rotation, tvec.reshape(3)
    )
    projected, _ = cv2.projectPoints(
        board_points,
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.solvepnp_distortion,
    )
    residual = projected.reshape(-1, 2) - image_points
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return ChessboardDetection(transform, image_points, rmse)


def draw_detection(
    image_bgr: np.ndarray,
    detection: ChessboardDetection,
    *,
    cols: int,
    rows: int,
) -> np.ndarray:
    overlay = np.asarray(image_bgr).copy()
    cv2.drawChessboardCorners(
        overlay,
        (int(cols), int(rows)),
        detection.corners_px.reshape(-1, 1, 2).astype(np.float32),
        True,
    )
    cv2.putText(
        overlay,
        "PnP RMSE {:.3f}px".format(detection.reprojection_rmse_px),
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay
