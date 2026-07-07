"""RoArm kinematics: stereo error → Δp → decoupled IK (base + shoulder/elbow)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class JointState:
    base: float = 0.0
    shoulder: float = 0.0
    elbow: float = 1.57
    wrist: float = 0.0
    roll: float = 0.0
    hand: float = 3.14

    def as_dict(self) -> Dict[str, float]:
        return {
            "base": self.base,
            "shoulder": self.shoulder,
            "elbow": self.elbow,
            "wrist": self.wrist,
            "roll": self.roll,
            "hand": self.hand,
        }

    @classmethod
    def from_feedback(cls, fb: Dict) -> "JointState":
        return cls(
            base=float(fb.get("b", fb.get("base", 0))),
            shoulder=float(fb.get("s", fb.get("shoulder", 0))),
            elbow=float(fb.get("e", fb.get("elbow", 1.57))),
            wrist=float(fb.get("t", fb.get("wrist", 0))),
            roll=float(fb.get("r", fb.get("roll", 0))),
            hand=float(fb.get("g", fb.get("hand", 3.14))),
        )


@dataclass
class StereoError:
    u: float
    v: float
    depth_m: float
    cam_err_m: float
    px: int
    py: int

    def cost(self) -> float:
        return math.sqrt(self.u * self.u + self.v * self.v) + abs(self.cam_err_m)


@dataclass
class KinematicsConfig:
    link_l1_mm: float = 285.0
    link_l2_mm: float = 375.0
    ik_damping: float = 0.08
    max_step_rad: float = 0.35
    base_gain_rad_per_u: float = 0.25
    base_u_deadzone: float = 0.10
    base_max_step_rad: float = 0.08
    base_only_when_fine: bool = True
    explicit_reach_cam_err_m: float = 0.10
    explicit_reach_shoulder_frac: float = 0.70
    explicit_reach_elbow_frac: float = 0.30
    stereo_kx_mm_per_m: float = 250.0
    stereo_ky_mm_per_u: float = 80.0
    stereo_kz_mm_per_v: float = 80.0
    jacobian: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml_section(cls, kin: dict, geom: dict) -> "KinematicsConfig":
        jac = kin.get("jacobian_shoulder_elbow") or {}
        cal_path = kin.get("calibration_json")
        if cal_path:
            p = Path(cal_path)
            if p.is_file():
                try:
                    cal = json.loads(p.read_text(encoding="utf-8"))
                    kin = {**kin, **{k: v for k, v in cal.items() if k != "jacobian_shoulder_elbow"}}
                    jac = cal.get("jacobian_shoulder_elbow", jac)
                except (OSError, json.JSONDecodeError):
                    pass
        return cls(
            link_l1_mm=float(kin.get("link_l1_mm", 285.0)),
            link_l2_mm=float(kin.get("link_l2_mm", 375.0)),
            ik_damping=float(kin.get("ik_damping", 0.08)),
            max_step_rad=float(kin.get("max_step_rad", 0.35)),
            base_gain_rad_per_u=float(kin.get("base_gain_rad_per_u", 0.25)),
            base_u_deadzone=float(kin.get("base_u_deadzone", 0.10)),
            base_max_step_rad=float(kin.get("base_max_step_rad", 0.08)),
            base_only_when_fine=bool(kin.get("base_only_when_fine", True)),
            explicit_reach_cam_err_m=float(kin.get("explicit_reach_cam_err_m", 0.10)),
            explicit_reach_shoulder_frac=float(kin.get("explicit_reach_shoulder_frac", 0.70)),
            explicit_reach_elbow_frac=float(kin.get("explicit_reach_elbow_frac", 0.30)),
            stereo_kx_mm_per_m=float(geom.get("stereo_kx_mm_per_m", 250.0)),
            stereo_ky_mm_per_u=float(geom.get("stereo_ky_mm_per_u", 80.0)),
            stereo_kz_mm_per_v=float(geom.get("stereo_kz_mm_per_v", 80.0)),
            jacobian=dict(jac),
        )


def normalize_uv(px: int, py: int, img_w: int, img_h: int) -> Tuple[float, float]:
    """Pixel offset from image center → u,v ∈ [-1, 1]."""
    u = (float(px) - img_w * 0.5) / max(1.0, img_w * 0.5)
    v = (float(py) - img_h * 0.5) / max(1.0, img_h * 0.5)
    return max(-1.0, min(1.0, u)), max(-1.0, min(1.0, v))


def cam_error(depth_m: float, target_camera_distance_m: float) -> float:
    """Positive = target too far along camera ray."""
    return float(depth_m) - float(target_camera_distance_m)


def stereo_delta_mm(err: StereoError, cfg: KinematicsConfig) -> Tuple[float, float, float]:
    """Map stereo error to arm-frame delta (mm): reach-X, lateral-Y, vertical-Z."""
    dx = -err.cam_err_m * cfg.stereo_kx_mm_per_m
    dy = -err.u * cfg.stereo_ky_mm_per_u
    dz = -err.v * cfg.stereo_kz_mm_per_v
    return dx, dy, dz


def forward_kinematics_2link(
    shoulder: float, elbow: float, l1_mm: float, l2_mm: float
) -> Tuple[float, float]:
    """Sagittal plane X (forward), Z (up) in mm from shoulder joint."""
    x = l1_mm * math.cos(shoulder) + l2_mm * math.cos(shoulder + elbow - math.pi)
    z = l1_mm * math.sin(shoulder) + l2_mm * math.sin(shoulder + elbow - math.pi)
    return x, z


def analytic_jacobian_2link(
    shoulder: float, elbow: float, l1_mm: float, l2_mm: float
) -> np.ndarray:
    """2×2 Jacobian ∂(x,z)/∂(shoulder, elbow) for sagittal IK."""
    s, e = shoulder, elbow
    phi = s + e - math.pi
    j = np.array(
        [
            [-l1_mm * math.sin(s) - l2_mm * math.sin(phi), -l2_mm * math.sin(phi)],
            [l1_mm * math.cos(s) + l2_mm * math.cos(phi), l2_mm * math.cos(phi)],
        ],
        dtype=np.float64,
    )
    return j


def damped_least_squares_ik(
    jacobian: np.ndarray,
    delta_xyz_mm: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Solve J @ dq ≈ delta for 2-DOF shoulder/elbow."""
    jt = jacobian.T
    n = jacobian.shape[1]
    lhs = jt @ jacobian + (damping ** 2) * np.eye(n)
    rhs = jt @ delta_xyz_mm
    try:
        dq = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        dq = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return dq


def explicit_reach_delta(cam_err_m: float, cfg: KinematicsConfig) -> Tuple[float, float]:
    """Large-error fallback: shoulder+ and elbow- (extend toward board)."""
    mag = min(cfg.max_step_rad, abs(cam_err_m) * 2.5)
    if cam_err_m > 0:
        ds = +mag * cfg.explicit_reach_shoulder_frac
        de = -mag * cfg.explicit_reach_elbow_frac
    else:
        ds = -mag * cfg.explicit_reach_shoulder_frac * 0.5
        de = +mag * cfg.explicit_reach_elbow_frac * 0.5
    return ds, de


@dataclass
class KinematicStep:
    delta_base: float
    delta_shoulder: float
    delta_elbow: float
    reason: str
    phase_hint: str
    delta_xyz_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class RoArmKinematics:
    def __init__(self, cfg: KinematicsConfig) -> None:
        self.cfg = cfg

    def compute_step(
        self,
        q: JointState,
        err: StereoError,
        *,
        use_explicit_reach: Optional[bool] = None,
    ) -> KinematicStep:
        dx, dy, dz = stereo_delta_mm(err, self.cfg)
        db = 0.0
        if abs(err.u) > self.cfg.base_u_deadzone:
            db = -err.u * self.cfg.base_gain_rad_per_u

        if use_explicit_reach is None:
            use_explicit_reach = abs(err.cam_err_m) > self.cfg.explicit_reach_cam_err_m

        if use_explicit_reach and self.cfg.base_only_when_fine:
            db = 0.0

        if use_explicit_reach:
            ds, de = explicit_reach_delta(err.cam_err_m, self.cfg)
            reason = "explicit_reach"
            phase_hint = "tri_reach"
        else:
            j = analytic_jacobian_2link(
                q.shoulder, q.elbow, self.cfg.link_l1_mm, self.cfg.link_l2_mm
            )
            if self.cfg.jacobian:
                jac = self.cfg.jacobian
                j = np.array(
                    [
                        [jac.get("ds_dx", j[0, 0]), jac.get("ds_dz", j[0, 1])],
                        [jac.get("de_dx", j[1, 0]), jac.get("de_dz", j[1, 1])],
                    ],
                    dtype=np.float64,
                )
            delta_mm = np.array([dx, dz], dtype=np.float64)
            dq = damped_least_squares_ik(j, delta_mm, self.cfg.ik_damping)
            ds, de = float(dq[0]), float(dq[1])
            reason = "ik_2link"
            phase_hint = "fine" if abs(err.cam_err_m) < 0.05 else "tri_reach"

        scale = self.cfg.max_step_rad / max(self.cfg.max_step_rad, abs(ds) + abs(de) + 1e-9)
        if scale < 1.0:
            ds *= scale
            de *= scale
            db *= scale

        cap = self.cfg.base_max_step_rad
        if abs(db) > cap:
            db = cap if db > 0 else -cap

        return KinematicStep(
            delta_base=db,
            delta_shoulder=ds,
            delta_elbow=de,
            reason=reason,
            phase_hint=phase_hint,
            delta_xyz_mm=(dx, dy, dz),
        )

    def clamp_deltas(
        self,
        q: JointState,
        step: KinematicStep,
        limits: Dict[str, Tuple[float, float]],
        shoulder_floor: float,
    ) -> KinematicStep:
        """Apply joint limits and HOME shoulder floor."""
        db = step.delta_base
        ds = step.delta_shoulder
        de = step.delta_elbow

        def _clamp_joint(name: str, cur: float, delta: float) -> float:
            lo, hi = limits.get(name, (-3.14, 3.14))
            if name == "shoulder":
                lo = max(lo, shoulder_floor)
            target = cur + delta
            if target < lo:
                return lo - cur
            if target > hi:
                return hi - cur
            return delta

        db = _clamp_joint("base", q.base, db)
        ds = _clamp_joint("shoulder", q.shoulder, ds)
        de = _clamp_joint("elbow", q.elbow, de)

        mag = max(abs(db), abs(ds), abs(de))
        if mag > self.cfg.max_step_rad:
            s = self.cfg.max_step_rad / mag
            db, ds, de = db * s, ds * s, de * s

        return KinematicStep(
            delta_base=db,
            delta_shoulder=ds,
            delta_elbow=de,
            reason=step.reason + "_clamped",
            phase_hint=step.phase_hint,
            delta_xyz_mm=step.delta_xyz_mm,
        )
