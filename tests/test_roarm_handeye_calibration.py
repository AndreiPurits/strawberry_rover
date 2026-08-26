import json
import math
import unittest
from pathlib import Path

import numpy as np

from pipelines.roarm_calibration.handeye import HandEyeObservation, solve_handeye
from pipelines.roarm_calibration.se3 import (
    RigidTransform,
    rotation_angle_rad,
    rotation_from_rpy,
)
from pipelines.roarm_calibration.urdf_fk import base_to_hand_tcp, base_to_link5
from pipelines.roarm_calibration.validation import KnownPointObservation, validate_known_points


def synthetic_observations():
    rng = np.random.default_rng(20260826)
    true_link5_camera = RigidTransform.from_rt(
        "link5",
        "stereo_camera_color_optical_frame",
        rotation_from_rpy(1.48, -0.12, 1.61),
        [-32.0, 18.0, 47.0],
    )
    base_board = RigidTransform.from_rt(
        "base_link",
        "calibration_board",
        rotation_from_rpy(0.1, -0.25, 0.35),
        [360.0, 40.0, 170.0],
    )
    observations = []
    for index in range(18):
        q = [
            rng.uniform(-1.1, 1.1),
            rng.uniform(-1.0, 1.2),
            rng.uniform(0.35, 2.8),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.2, 1.2),
        ]
        base_link5 = base_to_link5(q)
        camera_board = true_link5_camera.inverse().then(base_link5.inverse()).then(base_board)
        observations.append(
            HandEyeObservation(
                "synthetic_{:02d}".format(index),
                base_link5,
                camera_board,
                split="train" if index < 14 else "validation",
                reprojection_rmse_px=0.0,
            )
        )
    return true_link5_camera, observations


class RoArmHandEyeCalibrationTests(unittest.TestCase):
    def test_named_transform_composition_and_frame_gate(self):
        a_b = RigidTransform.from_rt("a", "b", np.eye(3), [1, 2, 3])
        b_c = RigidTransform.from_rt("b", "c", np.eye(3), [4, 5, 6])
        a_c = a_b.then(b_c)
        np.testing.assert_allclose(a_c.translation_mm, [5, 7, 9])
        np.testing.assert_allclose(a_c.inverse().then(a_c).matrix, np.eye(4), atol=1e-12)
        with self.assertRaisesRegex(ValueError, "frame mismatch"):
            b_c.then(a_b)
        with self.assertRaisesRegex(ValueError, "point frame"):
            a_b.transform_point([0, 0, 0], frame="wrong")

    def test_non_rigid_matrix_rejected(self):
        matrix = np.eye(4)
        matrix[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            RigidTransform("a", "b", matrix)

    def test_urdf_fk_matches_official_ikfast_translation_and_tool_direction(self):
        path = Path(__file__).parent / "data" / "roarm_m3_ikfast_full_fk.json"
        fixture = json.loads(path.read_text(encoding="utf-8"))
        translation_errors_mm = []
        direction_errors = []
        for sample in fixture["samples"]:
            transform = base_to_hand_tcp(sample["q"])
            expected_translation_mm = 1000.0 * np.asarray(sample["translation"])
            expected_direction = np.asarray(sample["tool_z_direction"])
            translation_errors_mm.append(float(np.linalg.norm(transform.translation_mm - expected_translation_mm)))
            direction_errors.append(float(np.linalg.norm(transform.rotation[:, 2] - expected_direction)))
        self.assertLess(max(translation_errors_mm), 0.015)
        self.assertLess(max(direction_errors), 1e-5)

    def test_handeye_recovers_known_transform_and_stays_fail_closed(self):
        expected, observations = synthetic_observations()
        result = solve_handeye(observations)
        self.assertTrue(result.ok)
        self.assertEqual(result.report["KIN_GATE"], "FAIL")
        self.assertEqual(result.report["independent_known_point_validation"], "NOT_RUN")
        translation_error = np.linalg.norm(
            result.transform_link5_camera.translation_mm - expected.translation_mm
        )
        rotation_error_deg = math.degrees(
            rotation_angle_rad(expected.rotation.T @ result.transform_link5_camera.rotation)
        )
        self.assertLess(translation_error, 1e-6)
        self.assertLess(rotation_error_deg, 1e-6)
        self.assertEqual(result.report["motion_diversity_train"]["rotation_rank"], 3)

    def test_handeye_rejects_insufficient_observations(self):
        _expected, observations = synthetic_observations()
        result = solve_handeye(observations[:5])
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "INSUFFICIENT_OBSERVATIONS")
        self.assertEqual(result.report["reason"], "INSUFFICIENT_CALIBRATION_OBSERVATIONS")
        self.assertEqual(result.report["KIN_GATE"], "FAIL")

    def test_independent_known_point_validation(self):
        transform, observations = synthetic_observations()
        point_base = np.asarray([330.0, -20.0, 190.0])
        known = []
        for index, handeye_obs in enumerate(observations[:4]):
            base_camera = handeye_obs.transform_base_link5.then(transform)
            point_camera = base_camera.inverse().transform_point(point_base, frame="base_link")
            known.append(
                KnownPointObservation(
                    "point_{:02d}".format(index),
                    handeye_obs.transform_base_link5,
                    point_camera,
                    point_base,
                )
            )
        report = validate_known_points(transform, known, max_error_mm=0.01)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["KIN_GATE"], "PASS")
        self.assertLess(report["max_error_mm"], 1e-9)

        report = validate_known_points(transform, known[:2], max_error_mm=0.01)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["reason"], "INSUFFICIENT_VALIDATION_OBSERVATIONS")


if __name__ == "__main__":
    unittest.main()
