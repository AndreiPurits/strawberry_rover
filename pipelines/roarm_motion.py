"""RoArm motion: phases, guards, command builder, stall detection."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pipelines.roarm_kinematics import JointState, KinematicStep


class MotionPhase(str, Enum):
    SHOULDER_TO_HOME = "shoulder_to_home"
    UNFOLD = "unfold"
    TRI_REACH = "tri_reach"
    FINE = "fine"
    DONE = "done"


@dataclass
class PoseRef:
    base: float
    shoulder: float
    elbow: float
    wrist: float = 0.0
    roll: float = 0.0
    hand: float = 3.14

    @classmethod
    def from_dict(cls, d: dict) -> "PoseRef":
        return cls(
            base=float(d.get("base", 0)),
            shoulder=float(d.get("shoulder", 0)),
            elbow=float(d.get("elbow", 1.57)),
            wrist=float(d.get("wrist", 0)),
            roll=float(d.get("roll", 0)),
            hand=float(d.get("hand", 3.14)),
        )


@dataclass
class MotionConfig:
    command_mode: str = "staged"
    staged_acc: float = 8.0
    staged_spd: float = 0.0
    joint_stall_rad: float = 0.04
    joint_stall_timeout_s: float = 1.2
    settle_s: float = 0.5
    shoulder_home_floor: float = -1.12
    grid_view_shoulder: float = -0.52


@dataclass
class StepLog:
    step_idx: int
    phase: str
    q_before: Dict[str, float]
    delta_q_cmd: Dict[str, float]
    q_after: Optional[Dict[str, float]]
    u: float
    v: float
    depth_m: float
    cam_err_m: float
    reason: str
    blocked: bool = False
    stall: bool = False
    dry_run: bool = True
    ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "phase": self.phase,
            "q_before": self.q_before,
            "delta_q_cmd": self.delta_q_cmd,
            "q_after": self.q_after,
            "u": round(self.u, 4),
            "v": round(self.v, 4),
            "depth_m": round(self.depth_m, 4) if self.depth_m else None,
            "cam_err_m": round(self.cam_err_m, 4),
            "reason": self.reason,
            "blocked": self.blocked,
            "stall": self.stall,
            "dry_run": self.dry_run,
            "ts": self.ts,
        }


def select_phase(
    q: JointState,
    cam_err_m: float,
    cfg: MotionConfig,
    grid_view: PoseRef,
) -> MotionPhase:
    if q.shoulder < cfg.shoulder_home_floor - 0.02:
        return MotionPhase.SHOULDER_TO_HOME
    if q.shoulder < grid_view.shoulder - 0.05:
        return MotionPhase.UNFOLD
    if abs(cam_err_m) > 0.05:
        return MotionPhase.TRI_REACH
    return MotionPhase.FINE


def unfold_step(q: JointState, grid_view: PoseRef, max_step: float) -> KinematicStep:
    """Move shoulder toward GRID_VIEW and elbow toward grid pose."""
    ds = grid_view.shoulder - q.shoulder
    de = grid_view.elbow - q.elbow
    mag = max(abs(ds), abs(de), 1e-9)
    scale = min(1.0, max_step / mag)
    return KinematicStep(
        delta_base=0.0,
        delta_shoulder=ds * scale,
        delta_elbow=de * scale,
        reason="unfold_to_grid_view",
        phase_hint="unfold",
    )


def shoulder_to_home_step(q: JointState, home: PoseRef, max_step: float) -> KinematicStep:
    ds = home.shoulder - q.shoulder
    if ds < 0:
        ds = 0.0
    mag = abs(ds)
    scale = min(1.0, max_step / max(mag, 1e-9))
    return KinematicStep(
        delta_base=0.0,
        delta_shoulder=ds * scale,
        delta_elbow=0.0,
        reason="shoulder_to_home_floor",
        phase_hint="shoulder_to_home",
    )


def apply_phase_override(
    phase: MotionPhase,
    q: JointState,
    kin_step: KinematicStep,
    home: PoseRef,
    grid_view: PoseRef,
    max_step: float,
) -> KinematicStep:
    if phase == MotionPhase.SHOULDER_TO_HOME:
        return shoulder_to_home_step(q, home, max_step)
    if phase == MotionPhase.UNFOLD:
        return unfold_step(q, grid_view, max_step)
    return kin_step


def build_target_joints(q: JointState, step: KinematicStep) -> JointState:
    return JointState(
        base=q.base + step.delta_base,
        shoulder=q.shoulder + step.delta_shoulder,
        elbow=q.elbow + step.delta_elbow,
        wrist=q.wrist,
        roll=q.roll,
        hand=q.hand,
    )


def joints_to_rpc_params(q: JointState, cfg: MotionConfig) -> Dict[str, Any]:
    return {
        "base": q.base,
        "shoulder": q.shoulder,
        "elbow": q.elbow,
        "wrist": q.wrist,
        "roll": q.roll,
        "hand": q.hand,
        "spd": cfg.staged_spd,
        "acc": cfg.staged_acc,
    }


def detect_joint_stall(
    q_before: JointState,
    q_after: JointState,
    delta_cmd: KinematicStep,
    cfg: MotionConfig,
) -> bool:
    """True if commanded motion did not materialize."""
    threshold = cfg.joint_stall_rad
    for name, cmd_d, before, after in (
        ("shoulder", delta_cmd.delta_shoulder, q_before.shoulder, q_after.shoulder),
        ("elbow", delta_cmd.delta_elbow, q_before.elbow, q_after.elbow),
        ("base", delta_cmd.delta_base, q_before.base, q_after.base),
    ):
        if abs(cmd_d) < threshold:
            continue
        actual = after - before
        if abs(cmd_d) > threshold and abs(actual) < threshold * 0.5:
            return True
    return False


class RoArmMotionController:
    def __init__(
        self,
        cfg: MotionConfig,
        home: PoseRef,
        grid_view: PoseRef,
        joint_limits: Dict[str, Tuple[float, float]],
    ) -> None:
        self.cfg = cfg
        self.home = home
        self.grid_view = grid_view
        self.joint_limits = joint_limits
        self._step_logs: List[StepLog] = []

    def reset_logs(self) -> None:
        self._step_logs.clear()

    @property
    def step_logs(self) -> List[StepLog]:
        return list(self._step_logs)

    def execute_step(
        self,
        *,
        step_idx: int,
        q: JointState,
        kin_step: KinematicStep,
        phase: MotionPhase,
        u: float,
        v: float,
        depth_m: float,
        cam_err_m: float,
        dry_run: bool,
        execute_rpc_fn,
        calibrate: bool = False,
    ) -> Tuple[JointState, bool, bool]:
        """Apply one motion step. Returns (q_after, blocked, stall)."""
        if calibrate:
            phase = (
                MotionPhase.FINE
                if abs(cam_err_m) < 0.06
                else MotionPhase.TRI_REACH
            )
            step = kin_step
        else:
            step = apply_phase_override(
                phase, q, kin_step, self.home, self.grid_view, max_step=0.35
            )
        q_target = build_target_joints(q, step)

        lo, hi = self.joint_limits.get("shoulder", (-1.6, 0.6))
        shoulder_floor = max(lo, self.cfg.shoulder_home_floor)
        if q_target.shoulder < shoulder_floor:
            q_target.shoulder = shoulder_floor
            step = KinematicStep(
                delta_base=q_target.base - q.base,
                delta_shoulder=q_target.shoulder - q.shoulder,
                delta_elbow=q_target.elbow - q.elbow,
                reason=step.reason + "_shoulder_floor",
                phase_hint=step.phase_hint,
            )

        blocked = (
            abs(step.delta_shoulder) < 1e-4
            and abs(step.delta_elbow) < 1e-4
            and abs(step.delta_base) < 1e-4
            and phase not in (MotionPhase.DONE,)
        )

        q_after = q
        stall = False

        if not dry_run and not blocked:
            mode = self.cfg.command_mode
            params = joints_to_rpc_params(q_target, self.cfg)
            if mode == "simultaneous":
                result = execute_rpc_fn("joints_move", params)
            else:
                result = execute_rpc_fn("home_joints_staged", params)
            if not result.get("ok"):
                blocked = True
            else:
                time.sleep(self.cfg.settle_s)
                fb = execute_rpc_fn("feedback", {})
                if fb.get("ok") and isinstance(fb.get("feedback"), dict):
                    q_after = JointState.from_feedback(fb["feedback"])
                stall = detect_joint_stall(q, q_after, step, self.cfg)
        elif dry_run:
            q_after = q_target

        log = StepLog(
            step_idx=step_idx,
            phase=phase.value,
            q_before=q.as_dict(),
            delta_q_cmd={
                "base": step.delta_base,
                "shoulder": step.delta_shoulder,
                "elbow": step.delta_elbow,
            },
            q_after=q_after.as_dict(),
            u=u,
            v=v,
            depth_m=depth_m,
            cam_err_m=cam_err_m,
            reason=step.reason,
            blocked=blocked,
            stall=stall,
            dry_run=dry_run,
            ts=time.time(),
        )
        self._step_logs.append(log)
        return q_after, blocked, stall

    def go_to_pose(
        self,
        pose: PoseRef,
        *,
        dry_run: bool,
        execute_rpc_fn,
        label: str = "pose",
    ) -> JointState:
        params = {
            "base": pose.base,
            "shoulder": pose.shoulder,
            "elbow": pose.elbow,
            "wrist": pose.wrist,
            "roll": pose.roll,
            "hand": pose.hand,
            "spd": self.cfg.staged_spd,
            "acc": self.cfg.staged_acc,
        }
        if dry_run:
            return JointState(**{k: params[k] for k in ("base", "shoulder", "elbow", "wrist", "roll", "hand")})
        result = execute_rpc_fn("home_joints_staged", params)
        if result.get("ok"):
            time.sleep(self.cfg.settle_s * 2)
            fb = execute_rpc_fn("feedback", {})
            if fb.get("ok"):
                return JointState.from_feedback(fb["feedback"])
        return JointState(
            base=pose.base,
            shoulder=pose.shoulder,
            elbow=pose.elbow,
            wrist=pose.wrist,
            roll=pose.roll,
            hand=pose.hand,
        )
