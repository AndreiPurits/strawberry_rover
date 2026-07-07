#!/usr/bin/env python3
"""Stereo target → position + distance → RoArm approach.

Goal: read desired point from stereo camera, print (u,v,depth,XYZ), move gripper there.

Examples:
  # Only measure target (no arm motion):
  python3 scripts/run_roarm_point_approach.py --probe

  # Chessboard corner, kinematics dry-run:
  python3 scripts/run_roarm_point_approach.py --source grid --dry-run

  # Plane on tarp (default): stereo depth RANSAC + corner on plane
  python3 scripts/run_roarm_point_approach.py --probe
  python3 scripts/run_roarm_point_approach.py --source plane --move --max-steps 6

  # Fixed pixel, approach loop on bench (operator ready):
  python3 scripts/run_roarm_point_approach.py --source pixel --px 640 --py 400 --move
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "ops/axm-monitor/agent") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ops/axm-monitor/agent"))

from pipelines.roarm_kinematics import JointState, KinematicsConfig, RoArmKinematics
from pipelines.roarm_motion import MotionConfig, PoseRef, RoArmMotionController, select_phase
from pipelines.roarm_live_approach import clear_live_approach, write_live_approach
from pipelines.roarm_plane_perception import (
    PlaneModel,
    PlanePerceptionConfig,
    PlaneTracker,
    build_plane_target,
)
from pipelines.roarm_perception import (
    PerceptionConfig,
    PointTarget,
    TargetTracker,
    build_grid_target,
    build_point_target,
    episode_success,
    list_corner_candidates,
)
from pipelines.roarm_template_track import TemplateCornerTracker, build_template_target
from pipelines.roarm_knowledge import (
    DEFAULT_KB_PATH,
    load_kb,
    pick_corner_auto,
    save_kb,
    update_kb_from_run,
)
from pipelines.ros_rgb_depth import Ros2RgbDepthProvider

STEREO_RGB = "/stereo_camera/color/image_rect_raw"
STEREO_DEPTH = "/stereo_camera/depth/image_rect_raw"
STEREO_INFO = "/stereo_camera/color/camera_info"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def view_pose_from_home(home_raw: dict) -> PoseRef:
    """Perception pose = HOME (точка «Дом» / Home на dashboard)."""
    return PoseRef.from_dict(home_raw.get("HOME") or {})

def build_perception_cfg(learn: dict, target_dist_m: float) -> PerceptionConfig:
    geom = learn.get("geometry") or {}
    perc = learn.get("perception") or {}
    tol = float(perc.get("corner_track_tolerance_px", perc.get("track_max_jump_px", 95.0)))
    jump = float(perc.get("track_max_jump_px", tol))
    return PerceptionConfig(
        chessboard_cols=int(perc.get("chessboard_cols", 9)),
        chessboard_rows=int(perc.get("chessboard_rows", 6)),
        corner_strategy=str(perc.get("corner_strategy", "nearest_center")),
        depth_median_radius=int(geom.get("depth_median_radius", 5)),
        depth_valid_min_m=float(geom.get("depth_valid_min_m", 0.05)),
        depth_valid_max_m=float(geom.get("depth_valid_max_m", 2.5)),
        depth_jump_max_m=float(geom.get("depth_jump_max_m", 0.4)),
        target_camera_distance_m=float(target_dist_m),
        uv_deadzone=float(geom.get("uv_deadzone", 0.06)),
        track_max_jump_px=max(jump, tol),
        track_lost_max_frames=int(geom.get("track_lost_max_frames", perc.get("track_lost_max_frames", 5))),
        track_reacquire_scale=float(geom.get("track_reacquire_scale", perc.get("track_reacquire_scale", 3.0))),
        use_template_lock=bool(perc.get("use_template_lock", True)),
        template_patch_px=int(perc.get("template_patch_px", 64)),
        template_search_radius_px=int(perc.get("template_search_radius_px", 72)),
        template_reacquire_radius_px=int(perc.get("template_reacquire_radius_px", 140)),
        template_min_score=float(perc.get("template_min_score", 0.52)),
    )


def apply_calibrate_profile(learn: dict, *, calibrate: bool) -> Tuple[float, float, bool]:
    """Return (max_step_rad, settle_s, disable_explicit_reach)."""
    kin = learn.get("kinematics") or {}
    cur = learn.get("curriculum") or {}
    if not calibrate:
        return (
            float(kin.get("max_step_rad", 0.35)),
            float(cur.get("settle_s", 0.5)),
            False,
        )
    return (
        float(kin.get("calibrate_max_step_rad", 0.08)),
        float(cur.get("calibrate_settle_s", 0.55)),
        bool(kin.get("calibrate_disable_explicit_reach", True)),
    )


def track_tolerance_px(learn: dict, perc_cfg: PerceptionConfig) -> float:
    perc = learn.get("perception") or {}
    return float(perc.get("corner_track_tolerance_px", perc_cfg.track_max_jump_px))


def plane_model_from_seed(meta: Optional[dict], target: PointTarget) -> PlaneModel:
    p = (meta or {}).get("plane") if isinstance(meta, dict) else None
    if isinstance(p, dict) and isinstance(p.get("normal"), list) and len(p["normal"]) == 3:
        return PlaneModel(
            normal=np.array(p["normal"], dtype=np.float64),
            d=float(p.get("d_m", target.depth_m or 1.0)),
            rmse_m=float(p.get("rmse_m", 0)),
            inlier_frac=float(p.get("inlier_frac", 1)),
        )
    depth = float(target.depth_m) if target.valid and target.depth_m else 1.0
    return PlaneModel(
        normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        d=depth,
        rmse_m=0.0,
        inlier_frac=1.0,
    )


def apply_retry_tuning(
    perc_cfg: PerceptionConfig,
    base: PerceptionConfig,
    attempt_idx: int,
    retry_raw: dict,
) -> None:
    if attempt_idx <= 0:
        perc_cfg.track_max_jump_px = base.track_max_jump_px
        perc_cfg.track_reacquire_scale = base.track_reacquire_scale
        perc_cfg.track_lost_max_frames = base.track_lost_max_frames
        return
    perc_cfg.track_max_jump_px = base.track_max_jump_px + attempt_idx * float(
        retry_raw.get("tune_jump_px_per_attempt", 18.0)
    )
    perc_cfg.track_reacquire_scale = base.track_reacquire_scale + attempt_idx * float(
        retry_raw.get("tune_reacquire_per_attempt", 0.5)
    )
    perc_cfg.track_lost_max_frames = base.track_lost_max_frames + attempt_idx * int(
        retry_raw.get("tune_lost_frames_per_attempt", 1)
    )


def approach_signature(result: dict) -> str:
    ft = result.get("final_target") or {}
    tr = result.get("tracking") or {}
    steps = result.get("steps") or []
    last = steps[-1] if steps else {}
    return "|".join(
        [
            str(result.get("success_reason", "")),
            str(ft.get("reject_reason", "")),
            str(round(float(ft.get("cam_err_m") or 0), 2)),
            str(int(tr.get("lost_streak") or 0)),
            str(len(steps)),
            str(round(float(last.get("cam_err_m") or 0), 2)),
        ]
    )


def publish_live_target(
    *,
    target: PointTarget,
    meta: dict,
    tracking: Optional[dict],
    attempt: int,
    tolerance_px: float,
    status: str = "running",
    status_text: str = "",
) -> None:
    write_live_approach(
        target=target.to_dict(),
        meta=meta,
        tracking=tracking,
        status=status,
        status_text=status_text,
        attempt=attempt,
        track_tolerance_px=tolerance_px,
        image_wh=meta.get("image_wh"),
    )


def build_plane_cfg(learn: dict) -> PlanePerceptionConfig:
    return PlanePerceptionConfig.from_yaml(learn.get("perception") or {})


def perceive_once(
    provider: Ros2RgbDepthProvider,
    perc_cfg: PerceptionConfig,
    plane_cfg: PlanePerceptionConfig,
    *,
    source: str,
    px: Optional[int],
    py: Optional[int],
    corner_idx: Optional[int],
    tracker: Optional[TargetTracker],
    plane_tracker: Optional[PlaneTracker],
    template_tracker: Optional[TemplateCornerTracker],
    timeout_s: float,
    prev_depth: Optional[float] = None,
) -> Tuple[PointTarget, Dict[str, Any]]:
    pair = provider.read(timeout_s=timeout_s)
    h, w = pair.rgb_bgr.shape[:2]
    intrinsics = provider.get_intrinsics()
    meta: Dict[str, Any] = {"image_wh": [w, h], "intrinsics": intrinsics}
    track_kw = tracker.tracking_kwargs() if tracker is not None else {}
    plane_kw = plane_tracker.tracking_kwargs() if plane_tracker is not None else {}
    forced = corner_idx
    if (
        (tracker is not None and tracker.locked)
        or (plane_tracker is not None and plane_tracker.locked)
        or (template_tracker is not None and template_tracker.locked)
    ):
        forced = None

    if source == "template" or (source == "grid" and perc_cfg.use_template_lock and template_tracker is not None):
        tt = template_tracker or TemplateCornerTracker.from_config(perc_cfg)
        target = build_template_target(
            pair.rgb_bgr,
            pair.depth_m,
            perc_cfg,
            tt,
            prev_depth=prev_depth,
            forced_corner_idx=forced if not tt.locked else None,
            intrinsics=intrinsics,
        )
        meta["corners_found"] = target.valid or target.reject_reason not in (
            "no_grid_corner",
            "template_lost",
            "template_lock_failed",
        )
        meta["candidates"] = list_corner_candidates(pair.rgb_bgr, perc_cfg)
        meta["tracking"] = {
            "locked": tt.locked,
            "lost_streak": tt.lost_streak,
            "track_px": round(tt.track_px, 1),
            "track_py": round(tt.track_py, 1),
            "anchor_px": round(tt.anchor_px, 1),
            "anchor_py": round(tt.anchor_py, 1),
            "template_score": round(tt.last_score, 3),
            "is_lost": tt.is_lost,
            "mode": "template",
        }
        meta["target_standoff_m"] = perc_cfg.target_camera_distance_m
        return target, meta

    if source == "plane":
        prev_plane = None
        if plane_tracker is not None and plane_tracker.locked and plane_tracker.normal is not None:
            from pipelines.roarm_plane_perception import PlaneModel

            prev_plane = PlaneModel(
                normal=plane_tracker.normal,
                d=plane_tracker.d,
                rmse_m=0.0,
                inlier_frac=1.0,
            )
        target, plane_meta = build_plane_target(
            pair.rgb_bgr,
            pair.depth_m,
            perc_cfg,
            plane_cfg,
            intrinsics=intrinsics,
            forced_corner_idx=forced,
            prev_plane=prev_plane,
            **{**track_kw, **plane_kw},
        )
        meta.update(plane_meta)
        meta["corners_found"] = target.valid or target.reject_reason not in (
            "no_tarp_mask",
            "plane_fit_failed",
            "no_grid_corner",
            "track_lost",
        )
        if target.valid or target.reject_reason != "no_grid_corner":
            meta["candidates"] = list_corner_candidates(pair.rgb_bgr, perc_cfg)
        if plane_tracker is not None:
            from pipelines.roarm_plane_perception import PlaneModel

            plane_model = None
            if isinstance(meta.get("plane"), dict):
                p = meta["plane"]
                n = p.get("normal")
                if isinstance(n, list) and len(n) == 3:
                    plane_model = PlaneModel(
                        normal=np.array(n, dtype=np.float64),
                        d=float(p.get("d_m", 0)),
                        rmse_m=float(p.get("rmse_m", 0)),
                        inlier_frac=float(p.get("inlier_frac", 0)),
                    )
            plane_tracker.note_valid(target, plane_model)
            meta["tracking"] = {
                "locked": plane_tracker.locked,
                "lost_streak": plane_tracker.lost_streak,
                "track_px": round(plane_tracker.track_px, 1),
                "track_py": round(plane_tracker.track_py, 1),
                "is_lost": plane_tracker.is_lost,
                "plane_d_m": round(plane_tracker.d, 4) if plane_tracker.locked else None,
            }
    elif source == "grid":
        target = build_grid_target(
            pair.rgb_bgr,
            pair.depth_m,
            perc_cfg,
            forced_corner_idx=forced,
            intrinsics=intrinsics,
            **track_kw,
        )
        meta["corners_found"] = target.valid or target.reject_reason not in ("no_grid", "track_lost")
        meta["candidates"] = list_corner_candidates(pair.rgb_bgr, perc_cfg)
    else:
        target = build_point_target(
            pair.rgb_bgr,
            pair.depth_m,
            perc_cfg,
            px=px,
            py=py,
            intrinsics=intrinsics,
            source="pixel",
        )
    if tracker is not None and source == "grid":
        tracker.note_valid(target)
        meta["tracking"] = {
            "locked": tracker.locked,
            "lost_streak": tracker.lost_streak,
            "track_px": round(tracker.track_px, 1),
            "track_py": round(tracker.track_py, 1),
            "is_lost": tracker.is_lost,
        }
    meta["target_standoff_m"] = perc_cfg.target_camera_distance_m
    return target, meta


def approach_loop(
    *,
    provider: Ros2RgbDepthProvider,
    perc_cfg: PerceptionConfig,
    plane_cfg: PlanePerceptionConfig,
    kin: RoArmKinematics,
    motion: RoArmMotionController,
    home: PoseRef,
    grid_view: PoseRef,
    view_label: str = "HOME",
    shoulder_floor: float,
    joint_limits: Dict[str, Tuple[float, float]],
    success_cfg: dict,
    source: str,
    px: Optional[int],
    py: Optional[int],
    corner_idx: Optional[int],
    max_steps: int,
    dry_run: bool,
    move_arm: bool,
    skip_home: bool,
    timeout_s: float,
    execute_rpc,
    seed_target: Optional[PointTarget] = None,
    seed_plane_meta: Optional[dict] = None,
    attempt: int = 1,
    track_tol_px: float = 95.0,
    template_tracker: Optional[TemplateCornerTracker] = None,
    calibrate: bool = False,
    disable_explicit_reach: bool = False,
) -> Dict[str, Any]:
    actually_move = move_arm and not dry_run
    q = JointState(
        base=grid_view.base,
        shoulder=grid_view.shoulder,
        elbow=grid_view.elbow,
        wrist=grid_view.wrist,
        roll=grid_view.roll,
        hand=grid_view.hand,
    )

    if actually_move:
        fb = execute_rpc("feedback", {})
        if fb.get("ok"):
            q = JointState.from_feedback(fb["feedback"])
        if not skip_home:
            q = motion.go_to_pose(home, dry_run=False, execute_rpc_fn=execute_rpc, label=view_label)
        time.sleep(motion.cfg.settle_s)
    elif not dry_run:
        fb = execute_rpc("feedback", {})
        if fb.get("ok"):
            q = JointState.from_feedback(fb["feedback"])

    motion.reset_logs()
    pick_corner_idx = corner_idx
    use_plane = source == "plane"
    use_template = source == "template" or (source == "grid" and perc_cfg.use_template_lock)
    tracker: Optional[TargetTracker] = None
    plane_tracker: Optional[PlaneTracker] = None
    tt = template_tracker or (
        TemplateCornerTracker.from_config(perc_cfg) if use_template else None
    )
    if use_template and tt is not None:
        if seed_target is not None and seed_target.valid and seed_target.corner_idx >= 0:
            pick_corner_idx = seed_target.corner_idx
    elif use_plane:
        plane_tracker = PlaneTracker(max_lost=perc_cfg.track_lost_max_frames)
        if seed_target is not None and seed_target.valid:
            plane_tracker.seed_from_target(
                seed_target,
                plane_model_from_seed(seed_plane_meta, seed_target),
            )
            pick_corner_idx = None
    else:
        tracker = TargetTracker(
            max_lost=perc_cfg.track_lost_max_frames,
            max_jump_px=perc_cfg.track_max_jump_px,
            reacquire_scale=perc_cfg.track_reacquire_scale,
        )
        if seed_target is not None and seed_target.valid:
            tracker.seed_from_target(seed_target)
            if pick_corner_idx is None and seed_target.corner_idx >= 0:
                pick_corner_idx = seed_target.corner_idx
    initial_depth: Optional[float] = None
    prev_depth: Optional[float] = None
    last_target: Optional[PointTarget] = None
    success = False
    success_reason = "max_steps"

    def _is_lost() -> bool:
        if use_template and tt is not None:
            return tt.is_lost
        if use_plane and plane_tracker is not None:
            return plane_tracker.is_lost
        if tracker is not None:
            return tracker.is_lost
        return False

    def _tracking_snapshot() -> dict:
        if use_template and tt is not None:
            return {
                "locked": tt.locked,
                "anchor_px": tt.anchor_px,
                "anchor_py": tt.anchor_py,
                "final_track_px": tt.track_px,
                "final_track_py": tt.track_py,
                "lost_streak": tt.lost_streak,
                "template_score": round(tt.last_score, 3),
                "mode": "template",
            }
        if use_plane and plane_tracker is not None:
            return {
                "locked": plane_tracker.locked,
                "anchor_px": plane_tracker.anchor_px,
                "anchor_py": plane_tracker.anchor_py,
                "final_track_px": plane_tracker.track_px,
                "final_track_py": plane_tracker.track_py,
                "lost_streak": plane_tracker.lost_streak,
                "plane_d_m": plane_tracker.d,
            }
        if tracker is not None:
            return {
                "locked": tracker.locked,
                "anchor_px": tracker.anchor_px,
                "anchor_py": tracker.anchor_py,
                "final_track_px": tracker.track_px,
                "final_track_py": tracker.track_py,
                "lost_streak": tracker.lost_streak,
            }
        return {}

    for step_idx in range(max_steps):
        if _is_lost():
            success_reason = "target_lost"
            break

        target, step_meta = perceive_once(
            provider,
            perc_cfg,
            plane_cfg,
            source=source,
            px=px,
            py=py,
            corner_idx=pick_corner_idx,
            tracker=tracker,
            plane_tracker=plane_tracker,
            template_tracker=tt,
            timeout_s=timeout_s,
            prev_depth=prev_depth,
        )
        last_target = target
        tr_snapshot = _tracking_snapshot()
        publish_live_target(
            target=target,
            meta=step_meta,
            tracking=tr_snapshot,
            attempt=attempt,
            tolerance_px=track_tol_px,
            status="running" if target.valid else "searching",
            status_text=f"шаг {step_idx + 1}/{max_steps} · попытка {attempt}",
        )

        if source in ("grid", "plane", "template") and target.valid:
            if use_template:
                pick_corner_idx = None
            elif use_plane and plane_tracker is not None and not plane_tracker.locked:
                pass
            elif tracker is not None and not tracker.locked:
                tracker.seed_from_target(target)
                pick_corner_idx = None
            else:
                pick_corner_idx = None

        if _is_lost():
            success_reason = "target_lost"
            break

        if initial_depth is None and target.valid:
            initial_depth = target.depth_m

        if target.valid and initial_depth is not None:
            ok, reason = episode_success(
                target,
                perc_cfg,
                initial_depth_m=initial_depth,
                success_cfg=success_cfg,
            )
            if ok:
                success = True
                success_reason = reason
                break

        if not target.valid:
            continue

        phase = select_phase(
            q,
            target.cam_err_m,
            motion.cfg,
            home if calibrate else grid_view,
        )
        kin_step = kin.compute_step(
            q,
            target.to_stereo_error(),
            use_explicit_reach=False if disable_explicit_reach else None,
        )
        kin_step = kin.clamp_deltas(q, kin_step, joint_limits, shoulder_floor)
        q, _blocked, _stall = motion.execute_step(
            step_idx=step_idx,
            q=q,
            kin_step=kin_step,
            phase=phase,
            u=target.u,
            v=target.v,
            depth_m=target.depth_m,
            cam_err_m=target.cam_err_m,
            dry_run=not actually_move,
            execute_rpc_fn=execute_rpc,
            calibrate=calibrate,
        )
        prev_depth = target.depth_m

    return {
        "success": success,
        "success_reason": success_reason,
        "initial_depth_m": initial_depth,
        "final_target": last_target.to_dict() if last_target else None,
        "q_final": q.as_dict(),
        "steps": [s.to_dict() for s in motion.step_logs],
        "tracking": _tracking_snapshot(),
        "attempt": attempt,
        "calibrate": calibrate,
    }


def run_approach_with_retries(
    *,
    provider: Ros2RgbDepthProvider,
    perc_cfg: PerceptionConfig,
    base_perc_cfg: PerceptionConfig,
    plane_cfg: PlanePerceptionConfig,
    kin: RoArmKinematics,
    motion: RoArmMotionController,
    home: PoseRef,
    grid_view: PoseRef,
    shoulder_floor: float,
    joint_limits: Dict[str, Tuple[float, float]],
    success_cfg: dict,
    retry_cfg: dict,
    source: str,
    px: Optional[int],
    py: Optional[int],
    corner_idx: Optional[int],
    max_steps: int,
    move_arm: bool,
    timeout_s: float,
    execute_rpc,
    track_tol_px: float,
    continue_from_pose: bool = True,
    template_tracker: Optional[TemplateCornerTracker] = None,
    calibrate: bool = False,
    disable_explicit_reach: bool = False,
) -> Dict[str, Any]:
    max_attempts = 2 if calibrate else int(retry_cfg.get("max_attempts", 4))
    max_same = int(retry_cfg.get("max_same_failures", 3))
    attempts_out: list = []
    same_streak = 0
    last_sig = ""
    final: Dict[str, Any] = {"success": False, "success_reason": "no_attempt"}
    tt = template_tracker or (
        TemplateCornerTracker.from_config(perc_cfg)
        if source in ("template",) or perc_cfg.use_template_lock
        else None
    )

    for attempt_idx in range(max_attempts):
        apply_retry_tuning(perc_cfg, base_perc_cfg, attempt_idx, retry_cfg)
        if move_arm and not continue_from_pose:
            motion.go_to_pose(home, dry_run=False, execute_rpc_fn=execute_rpc, label="HOME")
            time.sleep(motion.cfg.settle_s)

        seed_target, seed_meta = perceive_once(
            provider,
            perc_cfg,
            plane_cfg,
            source=source,
            px=px,
            py=py,
            corner_idx=corner_idx,
            tracker=None,
            plane_tracker=None,
            template_tracker=tt,
            timeout_s=timeout_s,
        )
        publish_live_target(
            target=seed_target,
            meta=seed_meta,
            tracking=None,
            attempt=attempt_idx + 1,
            tolerance_px=track_tol_px,
            status="probe",
            status_text=f"проба · попытка {attempt_idx + 1}/{max_attempts}",
        )
        if not seed_target.valid:
            final = {
                "success": False,
                "success_reason": "probe_invalid",
                "probe_reject": seed_target.reject_reason,
                "attempt": attempt_idx + 1,
            }
            attempts_out.append(final)
            sig = approach_signature(final)
        else:
            final = approach_loop(
                provider=provider,
                perc_cfg=perc_cfg,
                plane_cfg=plane_cfg,
                kin=kin,
                motion=motion,
                home=home,
                grid_view=grid_view,
                shoulder_floor=shoulder_floor,
                joint_limits=joint_limits,
                success_cfg=success_cfg,
                source=source,
                px=px,
                py=py,
                corner_idx=corner_idx,
                max_steps=max_steps,
                dry_run=False,
                move_arm=move_arm,
                skip_home=continue_from_pose or attempt_idx > 0,
                timeout_s=timeout_s,
                execute_rpc=execute_rpc,
                seed_target=seed_target,
                seed_plane_meta=seed_meta,
                attempt=attempt_idx + 1,
                track_tol_px=track_tol_px,
                template_tracker=tt,
                calibrate=calibrate,
                disable_explicit_reach=disable_explicit_reach,
            )
            final["attempt"] = attempt_idx + 1
            final["tuning"] = {
                "track_max_jump_px": perc_cfg.track_max_jump_px,
                "track_reacquire_scale": perc_cfg.track_reacquire_scale,
                "track_lost_max_frames": perc_cfg.track_lost_max_frames,
            }
            attempts_out.append(final)
            sig = approach_signature(final)

        if final.get("success"):
            break

        if sig == last_sig:
            same_streak += 1
        else:
            same_streak = 1
            last_sig = sig

        if same_streak >= max_same:
            final["success_reason"] = "repeated_failure_stop"
            final["same_failure_streak"] = same_streak
            break

        if final.get("success_reason") in ("target_lost", "max_steps", "probe_invalid"):
            continue
        break

    stopped = {
        k: v
        for k, v in final.items()
        if k != "attempts"
    }
    stopped["attempts"] = [
        {kk: vv for kk, vv in att.items() if kk != "attempts"}
        for att in attempts_out
    ]
    publish_live_target(
        target=PointTarget(
            px=0,
            py=0,
            u=0,
            v=0,
            depth_m=0,
            cam_err_m=999,
            valid=False,
            source="plane",
            reject_reason=str(stopped.get("success_reason", "")),
        ),
        meta={},
        tracking=stopped.get("tracking"),
        attempt=int(stopped.get("attempt", 1)),
        tolerance_px=track_tol_px,
        status="stopped",
        status_text=str(stopped.get("success_reason", "stopped")),
    )
    return stopped


def main() -> int:
    ap = argparse.ArgumentParser(description="Stereo point → RoArm approach")
    ap.add_argument("--config", default=str(REPO_ROOT / "config/roarm_learn.yaml"))
    ap.add_argument("--home-config", default=str(REPO_ROOT / "config/roarm_home_joints.yaml"))
    ap.add_argument("--source", choices=["template", "plane", "grid", "pixel", "center"], default="template")
    ap.add_argument("--px", type=int, default=None)
    ap.add_argument("--py", type=int, default=None)
    ap.add_argument(
        "--corner-idx",
        default="auto",
        help="Grid corner index, or 'auto' to pick any valid corner",
    )
    ap.add_argument("--target-dist-m", type=float, default=None, help="Standoff along camera ray")
    ap.add_argument("--probe", action="store_true", help="Only read target pose, no approach")
    ap.add_argument("--dry-run", action="store_true", help="Plan motion, no arm commands")
    ap.add_argument("--move", action="store_true", help="Move arm (operator ready)")
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Stage 1: small joint steps, template lock, no explicit_reach",
    )
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--rgb-topic", default=STEREO_RGB)
    ap.add_argument("--depth-topic", default=STEREO_DEPTH)
    ap.add_argument("--camera-info-topic", default=STEREO_INFO)
    ap.add_argument("--timeout-s", type=float, default=8.0)
    ap.add_argument("--skip-home", action="store_true", help="Skip HOME (already in view)")
    ap.add_argument(
        "--no-continue",
        dest="continue_pose",
        action="store_false",
        help="Reset to HOME each retry instead of continuing from pose",
    )
    ap.set_defaults(continue_pose=True)
    ap.add_argument(
        "--reset-home",
        action="store_true",
        help="Go HOME before approach (ignore knowledge warm-start)",
    )
    ap.add_argument("--kb-path", default=str(DEFAULT_KB_PATH), help="Knowledge base JSON path")
    ap.add_argument("--save-log", default=None, help="JSON log path")
    args = ap.parse_args()

    learn = load_yaml(Path(args.config))
    home_raw = load_yaml(Path(args.home_config))
    home = PoseRef.from_dict(home_raw.get("HOME") or {})
    view_pose = view_pose_from_home(home_raw)
    shoulder_floor = float(home_raw.get("shoulder_home_floor", -1.12))

    target_dist = args.target_dist_m
    if target_dist is None:
        target_dist = float((learn.get("geometry") or {}).get("target_camera_distance_m", 0.11))

    perc_cfg = build_perception_cfg(learn, target_dist)
    base_perc_cfg = build_perception_cfg(learn, target_dist)
    plane_cfg = build_plane_cfg(learn)
    track_tol = track_tolerance_px(learn, perc_cfg)
    retry_cfg = learn.get("approach_retry") or {}
    source = "pixel" if args.source in ("pixel", "center") else str(args.source)
    px, py = args.px, args.py

    provider = Ros2RgbDepthProvider(
        rgb_topic=str(args.rgb_topic),
        depth_topic=str(args.depth_topic),
        sync_slop_s=0.15,
    )
    provider.open(camera_info_topic=str(args.camera_info_topic))

    t0 = time.time()
    while provider.get_intrinsics() is None and (time.time() - t0) < 6.0:
        time.sleep(0.05)

    report: Dict[str, Any] = {"ts": time.time()}
    exit_code = 0
    kb_path = Path(args.kb_path)
    kb = load_kb(kb_path)
    report["knowledge"] = {
        "path": str(kb_path),
        "session_id": kb.get("session_id"),
        "total_runs": (kb.get("stats") or {}).get("total_runs", 0),
    }

    corner_raw = str(args.corner_idx or "auto").strip().lower()
    corner_idx: Optional[int] = None
    if corner_raw not in ("auto", "", "none"):
        try:
            corner_idx = int(corner_raw)
        except ValueError:
            corner_idx = None

    continue_from_pose = bool(args.continue_pose) and not bool(args.reset_home)
    skip_home = bool(args.skip_home) or continue_from_pose
    calibrate = bool(args.calibrate) or source == "template"
    max_step_rad, settle_s, disable_explicit_reach = apply_calibrate_profile(
        learn, calibrate=calibrate
    )
    template_tracker = TemplateCornerTracker.from_config(perc_cfg) if source in (
        "template",
        "grid",
    ) else None
    # Continue calibrate from last known corner anchor in KB.
    last_tgt = kb.get("last_target") or {}
    if corner_idx is None and last_tgt.get("corner_idx") is not None:
        corner_idx = int(last_tgt["corner_idx"])
        print(f"[kb] resume corner_idx={corner_idx}")

    try:
        if corner_idx is None and source in ("template", "plane", "grid"):
            pair0 = provider.read(timeout_s=float(args.timeout_s))
            picked = pick_corner_auto(
                pair0.rgb_bgr,
                pair0.depth_m,
                perc_cfg,
                kb=kb,
            )
            if picked is not None:
                corner_idx = picked
                print(f"[kb] auto corner_idx={corner_idx}")
            report["picked_corner_idx"] = corner_idx

        target, meta = perceive_once(
            provider,
            perc_cfg,
            plane_cfg,
            source=source,
            px=px,
            py=py,
            corner_idx=corner_idx,
            tracker=None,
            plane_tracker=None,
            template_tracker=template_tracker,
            timeout_s=float(args.timeout_s),
        )
        report["perception"] = target.to_dict()
        report["meta"] = meta
        tr_probe = (meta.get("tracking") or {}) if isinstance(meta, dict) else {}
        publish_live_target(
            target=target,
            meta=meta,
            tracking=tr_probe,
            attempt=1,
            tolerance_px=track_tol,
            status="probe",
            status_text="проба цели",
        )

        print("=== TARGET (stereo) ===")
        print(json.dumps({"perception": report["perception"], "meta": meta}, indent=2, ensure_ascii=False))

        if not target.valid:
            print(f"Target invalid: {target.reject_reason}", file=sys.stderr)
            exit_code = 2
            if args.probe:
                return exit_code

        if args.probe:
            return exit_code

        dry_run = bool(args.dry_run) or not bool(args.move)
        kin_cfg = KinematicsConfig.from_yaml_section(
            learn.get("kinematics") or {},
            learn.get("geometry") or {},
        )
        kin = RoArmKinematics(kin_cfg)
        kin.cfg.max_step_rad = max_step_rad
        motion_cfg = MotionConfig(
            command_mode=str((learn.get("motion") or {}).get("command_mode", "staged")),
            staged_acc=float((learn.get("motion") or {}).get("staged_acc", 8.0)),
            settle_s=settle_s,
            shoulder_home_floor=shoulder_floor,
            grid_view_shoulder=view_pose.shoulder,
        )
        limits_raw = learn.get("joint_limits") or {}
        joint_limits = {
            k: (float(v[0]), float(v[1]))
            for k, v in limits_raw.items()
            if isinstance(v, (list, tuple)) and len(v) >= 2
        }
        motion = RoArmMotionController(
            motion_cfg, home, home if calibrate else view_pose, joint_limits
        )

        from roarm_proxy import execute_rpc

        if bool(args.move) and not dry_run and bool(args.reset_home):
            motion.go_to_pose(home, dry_run=False, execute_rpc_fn=execute_rpc, label="HOME")
            time.sleep(motion.cfg.settle_s)
            continue_from_pose = False
            skip_home = False

        if bool(args.move) and not dry_run:
            approach = run_approach_with_retries(
                provider=provider,
                perc_cfg=perc_cfg,
                base_perc_cfg=base_perc_cfg,
                plane_cfg=plane_cfg,
                kin=kin,
                motion=motion,
                home=home,
                grid_view=view_pose,
                shoulder_floor=shoulder_floor,
                joint_limits=joint_limits,
                success_cfg=learn.get("success") or {},
                retry_cfg=retry_cfg,
                source=source,
                px=px,
                py=py,
                corner_idx=corner_idx,
                max_steps=int(args.max_steps),
                move_arm=True,
                timeout_s=float(args.timeout_s),
                execute_rpc=execute_rpc,
                track_tol_px=track_tol,
                continue_from_pose=continue_from_pose,
                template_tracker=template_tracker,
                calibrate=calibrate,
                disable_explicit_reach=disable_explicit_reach,
            )
        else:
            approach = approach_loop(
                provider=provider,
                perc_cfg=perc_cfg,
                plane_cfg=plane_cfg,
                kin=kin,
                motion=motion,
                home=home,
                grid_view=view_pose,
                view_label="HOME",
                shoulder_floor=shoulder_floor,
                joint_limits=joint_limits,
                success_cfg=learn.get("success") or {},
                source=source,
                px=px,
                py=py,
                corner_idx=corner_idx,
                max_steps=int(args.max_steps),
                dry_run=dry_run,
                move_arm=bool(args.move),
                skip_home=skip_home,
                timeout_s=float(args.timeout_s),
                execute_rpc=execute_rpc,
                seed_target=target if target.valid else None,
                seed_plane_meta=meta if target.valid else None,
                attempt=1,
                track_tol_px=track_tol,
                template_tracker=template_tracker,
                calibrate=calibrate,
                disable_explicit_reach=disable_explicit_reach,
            )
        report["approach"] = approach
        report["dry_run"] = dry_run
        if not dry_run and "approach" in report:
            kb = update_kb_from_run(kb, report=report, approach=report["approach"])
            kb_path_saved = save_kb(kb, kb_path)
            report["knowledge"]["saved"] = str(kb_path_saved)
            report["knowledge"]["total_runs"] = (kb.get("stats") or {}).get("total_runs", 0)
            print(f"\n[kb] updated → {kb_path_saved}")

        print("\n=== APPROACH ===")
        print(json.dumps(approach, indent=2, ensure_ascii=False))
        for step in approach.get("steps") or []:
            dq = step["delta_q_cmd"]
            print(
                f"  step {step['step_idx']}: ds={dq['shoulder']:+.3f} de={dq['elbow']:+.3f} "
                f"depth={step.get('depth_m')} cam_err={step.get('cam_err_m')} [{step.get('reason')}]"
            )

        if not approach.get("success"):
            exit_code = 3 if exit_code == 0 else exit_code

    except KeyboardInterrupt:
        print("\nStopped.")
        exit_code = 130
    finally:
        provider.close()
        if not args.probe and bool(args.move):
            time.sleep(1.0)
            clear_live_approach()

    if args.save_log:
        log_path = Path(args.save_log)
    else:
        log_dir = REPO_ROOT / "runs/roarm_learn/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"point_{uuid.uuid4().hex[:10]}.json"
    log_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLog: {log_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
