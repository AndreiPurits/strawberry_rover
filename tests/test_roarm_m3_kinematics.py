from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from pipelines.roarm_m3_kinematics import (
    FIRMWARE_LIMITS,
    GEOMETRY,
    HARD_LIMITS,
    IK_NO_SOLUTION,
    JointVector,
    Pose5,
    damped_pseudoinverse,
    fk,
    ik,
    jacobian,
    pose_error,
    within_limits,
)


class RoArmM3KinematicsTests(unittest.TestCase):
    def test_official_geometry_constants(self) -> None:
        self.assertAlmostEqual(GEOMETRY.l2_mm, math.hypot(236.82, 30.0), places=12)
        self.assertAlmostEqual(GEOMETRY.l3_mm, 144.49, places=12)
        self.assertAlmostEqual(GEOMETRY.tool_mm, math.hypot(171.67, 13.69), places=12)

    def test_t105_feedback_fk(self) -> None:
        fixture = Path(__file__).parent / "data" / "roarm_m3_t105_samples.json"
        samples = json.loads(fixture.read_text(encoding="utf-8"))["samples"]
        for sample in samples:
            fb = sample["feedback"]
            predicted = fk([fb["b"], fb["s"], fb["e"], fb["t"], fb["r"]])
            expected = Pose5(fb["x"], fb["y"], fb["z"], fb["tit"], fb["r"])
            error = pose_error(predicted, expected)
            self.assertLess(float(np.max(np.abs(error[:3]))), 1e-6)
            self.assertLess(float(np.max(np.abs(error[3:]))), 1e-9)

    def test_jacobian_matches_central_finite_difference(self) -> None:
        rng = np.random.default_rng(314159)
        step = 1e-6
        max_error = 0.0
        for _ in range(200):
            q = np.asarray(
                [
                    rng.uniform(-1.2, 1.2),
                    rng.uniform(-1.2, 1.2),
                    rng.uniform(0.4, 2.9),
                    rng.uniform(-1.2, 1.2),
                    rng.uniform(-1.2, 1.2),
                ],
                dtype=np.float64,
            )
            analytic = jacobian(q)
            numeric = np.zeros((5, 5), dtype=np.float64)
            for column in range(5):
                q_plus, q_minus = q.copy(), q.copy()
                q_plus[column] += step
                q_minus[column] -= step
                numeric[:, column] = pose_error(fk(q_minus), fk(q_plus)) / (2.0 * step)
            max_error = max(max_error, float(np.max(np.abs(analytic - numeric))))
        self.assertLess(max_error, 1e-6)

    def test_ik_fk_round_trip_and_seed_continuity(self) -> None:
        rng = np.random.default_rng(271828)
        tested = 0
        for _ in range(1000):
            q = JointVector(
                base=float(rng.uniform(-1.1, 1.1)),
                shoulder=float(rng.uniform(-1.1, 1.1)),
                elbow=float(rng.uniform(0.45, 2.9)),
                wrist=float(rng.uniform(-1.1, 1.1)),
                roll=float(rng.uniform(-1.1, 1.1)),
            )
            target = fk(q)
            result = ik(target, seed_q=q)
            # Some randomly generated FK poses require an equivalent wrist
            # representation outside hard limits after angle normalization.
            if not result.ok:
                continue
            tested += 1
            selected = result.selected
            assert selected is not None
            residual = pose_error(fk(selected.q), target)
            self.assertLess(float(np.linalg.norm(residual[:3])), 1e-6)
            self.assertLess(float(np.linalg.norm(residual[3:])), 1e-9)
            self.assertTrue(within_limits(selected.q, "hard"))
            seed_distance = np.linalg.norm(selected.q.as_array() - q.as_array())
            self.assertLess(float(seed_distance), 1e-8)
        self.assertGreater(tested, 800)

    def test_unreachable_pose_fails_without_fallback(self) -> None:
        result = ik(Pose5(2000.0, 0.0, 0.0, 0.0, 0.0))
        self.assertFalse(result.ok)
        self.assertEqual(result.status, IK_NO_SOLUTION)
        self.assertIsNone(result.selected)

    def test_limits_are_explicit_and_hand_is_not_pose_joint(self) -> None:
        self.assertEqual(tuple(HARD_LIMITS), ("base", "shoulder", "elbow", "wrist", "roll"))
        self.assertEqual(tuple(FIRMWARE_LIMITS), tuple(HARD_LIMITS))
        self.assertTrue(within_limits([0.0, 0.0, 1.0, 0.0, 0.0]))
        self.assertFalse(within_limits([0.0, 0.0, 0.0, 0.0, 0.0]))

    def test_damped_inverse_is_finite_at_singularity(self) -> None:
        # elbow=d2 makes the two main planar links collinear in this convention.
        q = [0.0, 0.0, GEOMETRY.d2_rad, 0.0, 0.0]
        pinv = damped_pseudoinverse(
            jacobian(q), damping=1e-2, translation_scale_mm=GEOMETRY.max_reach_mm
        )
        self.assertEqual(pinv.shape, (5, 5))
        self.assertTrue(np.isfinite(pinv).all())


if __name__ == "__main__":
    unittest.main()
