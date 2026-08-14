#!/usr/bin/env python3
"""Unit tests for manipulator approach v1 (no GPU, no actuators)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pipelines.arm_approach.cartesian import cartesian_step, fit_tip_camera_map  # noqa: E402
from pipelines.arm_approach.claw_exclude import detect_claw_exclude_regions  # noqa: E402
from pipelines.arm_approach.phases import next_center_reach_step  # noqa: E402
from pipelines.arm_approach.pipeline import ManipulatorApproachPipeline  # noqa: E402
from pipelines.arm_approach.planner import nearest_demo, plan_one_shot, solve_delta_with_prior  # noqa: E402
from pipelines.arm_approach.rollback import push_history, rollback_until_visible  # noqa: E402
from pipelines.arm_approach.tracker import pick_detection  # noqa: E402
from pipelines.arm_approach.verify import verify_success  # noqa: E402
from arm_approach_geometry import berry_error  # noqa: E402


def _learned():
    return {
        "demos": [
            {
                "source": "a",
                "success": True,
                "q_start": {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
                "berry_start": {"px": 400.0, "py": 220.0, "depth_m": 0.40},
                "berry_success": {"px": 320.0, "py": 230.0, "depth_m": 0.13},
                "delta_q": {"base": -0.12, "shoulder": 0.15, "elbow": -0.15},
            },
            {
                "source": "b",
                "success": True,
                "q_start": {"base": 0.8, "shoulder": 0.1, "elbow": 2.0},
                "berry_start": {"px": 120.0, "py": 300.0, "depth_m": 0.55},
                "berry_success": {"px": 320.0, "py": 240.0, "depth_m": 0.13},
                "delta_q": {"base": 0.3, "shoulder": 0.4, "elbow": -0.4},
            },
        ],
        "local_jacobian": {
            "quality_ok": True,
            "matrix": [
                [0.0, 0.0, 900.0],
                [140.0, -80.0, 0.0],
                [-0.20, 0.12, 0.0],
            ],
        },
    }


class TestDemoSelect(unittest.TestCase):
    def test_nearest_demo(self):
        q = {"base": 0.02, "shoulder": 0.41, "elbow": 1.58}
        berry = {"px": 395.0, "py": 221.0, "depth_m": 0.39}
        demo = nearest_demo(_learned(), q, berry)
        self.assertIsNotNone(demo)
        self.assertEqual(demo["source"], "a")


class TestJacobianPrior(unittest.TestCase):
    def test_clip_keeps_prior_dominant(self):
        prior = np.array([0.15, -0.15, -0.12])
        error = np.array([400.0, 200.0, 0.5])
        j = np.eye(3) * 10.0
        sol = solve_delta_with_prior(error, prior, j, prior_weight=1.4, correction_clip_rad=0.35)
        self.assertTrue(np.all(np.abs(sol - prior) <= 0.35 + 1e-9))

    def test_no_jacobian_returns_prior(self):
        prior = np.array([0.1, -0.1, 0.05])
        error = np.array([80.0, 10.0, -0.1])
        sol = solve_delta_with_prior(error, prior, None)
        np.testing.assert_array_equal(sol, prior)


class TestPlan(unittest.TestCase):
    def test_plan_uses_demo(self):
        plan = plan_one_shot(
            {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
            {"px": 400.0, "py": 220.0, "depth_m": 0.40, "conf": 0.9},
            _learned(),
        )
        self.assertIn("nearest_demo", plan.reason)
        self.assertLess(plan.q_target["base"], plan.q_start["base"])


class TestVerify(unittest.TestCase):
    def test_standoff_ok(self):
        ok, report = verify_success(
            {"px": 400.0, "py": 220.0, "depth_m": 0.40, "conf": 0.9},
            {"px": 320.0, "py": 230.0, "depth_m": 0.13, "conf": 0.88},
            {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
            {"base": -0.12, "shoulder": 0.55, "elbow": 1.45},
            {"base": -0.11, "shoulder": 0.54, "elbow": 1.46},
        )
        self.assertTrue(ok)
        self.assertEqual(report.reasons, [])

    def test_no_progress(self):
        ok, report = verify_success(
            {"px": 400.0, "py": 220.0, "depth_m": 0.40, "conf": 0.9},
            {"px": 398.0, "py": 221.0, "depth_m": 0.39, "conf": 0.9},
            {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
            {"base": -0.12, "shoulder": 0.55, "elbow": 1.45},
            {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
        )
        self.assertFalse(ok)
        self.assertTrue(report.reasons)


class TestTracker(unittest.TestCase):
    def test_prefers_near_lock(self):
        dets = [
            {"x1": 10, "y1": 10, "x2": 40, "y2": 50, "conf": 0.99},
            {"x1": 190, "y1": 190, "x2": 230, "y2": 240, "conf": 0.6},
        ]
        best = pick_detection(dets, 210.0, 215.0, max_jump_px=80.0)
        self.assertAlmostEqual((best["x1"] + best["x2"]) * 0.5, 210.0, delta=5)


class TestClawSlots(unittest.TestCase):
    def test_two_slots(self):
        gray = np.full((480, 640), 180, dtype=np.uint8)
        gray[340:470, 170:250] = 20
        gray[340:470, 390:470] = 20
        regions = detect_claw_exclude_regions(gray, min_area=80.0)
        self.assertGreaterEqual(len(regions), 1)


class TestPhases(unittest.TestCase):
    def test_center_then_reach(self):
        cfg = {"aim_mode": "lock_y", "center_tol_px": 20.0, "standoff_m": 0.12, "depth_floor_m": 0.02}
        phase, dq = next_center_reach_step(
            400.0, 220.0, 0.40, aim_px=320.0, aim_py=220.0, phase="CENTER", cfg=cfg
        )
        self.assertEqual(phase, "CENTER")
        self.assertNotEqual(dq["base"], 0.0)
        phase, dq = next_center_reach_step(
            322.0, 220.0, 0.40, aim_px=320.0, aim_py=220.0, phase="CENTER", cfg=cfg
        )
        self.assertEqual(phase, "REACH")
        self.assertNotEqual(dq["shoulder"], 0.0)


class TestCartesian(unittest.TestCase):
    def test_identity_map(self):
        dtip = [[10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0], [0, 0, 10], [0, 0, -10]]
        dcam = dtip
        out = fit_tip_camera_map(dtip, dcam)
        self.assertIsNotNone(out)
        m, rmse = out
        self.assertLess(rmse, 0.1)
        step = cartesian_step(m, [0, 0, 400], [20, 0, 400], max_step_mm=50)
        self.assertGreater(step[0], 5)


class TestRollback(unittest.TestCase):
    def test_rollback(self):
        hist = []
        push_history(hist, {"base": 0.0}, {"px": 320})
        push_history(hist, {"base": 0.2}, None)
        q, reason = rollback_until_visible(hist, visible=False)
        self.assertEqual(reason, "rollback")
        self.assertEqual(q["base"], 0.0)


class TestPipeline(unittest.TestCase):
    def test_plan_and_verify(self):
        pipe = ManipulatorApproachPipeline()
        lock_res = pipe.lock_from_detections(
            [{"x1": 380, "y1": 200, "x2": 420, "y2": 250, "conf": 0.9}],
            depth_m=0.40,
        )
        self.assertEqual(lock_res.state.value, "LOCK")
        plan_res = pipe.plan(
            {"base": 0.0, "shoulder": 0.4, "elbow": 1.6},
            {"px": 400.0, "py": 225.0, "depth_m": 0.40, "conf": 0.9},
            _learned(),
        )
        self.assertIsNotNone(plan_res.plan)
        err = berry_error({"px": 400.0, "py": 225.0, "depth_m": 0.40}, plan_res.plan.target_image)
        self.assertGreater(abs(float(err[0])), 1.0)


if __name__ == "__main__":
    unittest.main()
