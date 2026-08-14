"""Online tip→camera map for Cartesian PBVS (fit only, no actuators)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def fit_tip_camera_map(
    dtip_mm: Sequence[Sequence[float]],
    dcam_mm: Sequence[Sequence[float]],
    *,
    min_probes: int = 3,
    min_tip_move_mm: float = 3.0,
) -> Optional[Tuple[np.ndarray, float]]:
    """Least-squares M in dcam = M · dtip from actual probe deltas.

    Returns (M 3x3, rmse_mm) or None.
    """
    X_rows = []
    Y_rows = []
    for dtip, dcam in zip(dtip_mm, dcam_mm):
        t = np.asarray(dtip, dtype=np.float64).reshape(3)
        c = np.asarray(dcam, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(t)) < min_tip_move_mm:
            continue
        X_rows.append(t)
        Y_rows.append(c)
    if len(X_rows) < min_probes:
        return None
    X = np.array(X_rows)
    Y = np.array(Y_rows)
    mt, *_ = np.linalg.lstsq(X, Y, rcond=None)
    m = mt.T
    pred = X @ m.T
    rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))
    return m, rmse


def cartesian_step(
    m: np.ndarray,
    cam_now_mm: Sequence[float],
    cam_goal_mm: Sequence[float],
    *,
    max_step_mm: float = 20.0,
) -> np.ndarray:
    err = np.asarray(cam_goal_mm, dtype=np.float64) - np.asarray(cam_now_mm, dtype=np.float64)
    delta = np.linalg.pinv(m) @ err
    n = float(np.linalg.norm(delta))
    if n > max_step_mm and n > 1e-9:
        delta = delta * (max_step_mm / n)
    return delta
