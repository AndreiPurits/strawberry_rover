import unittest

import numpy as np

from pipelines.roarm_calibration import base_to_grasp, link5_to_grasp
from pipelines.roarm_m3_kinematics import JointVector, Pose5, grasp_fk, grasp_ik, grasp_jacobian, pose_error


class RoArmGraspFrameTest(unittest.TestCase):
    def test_link5_grasp_cad_transform_and_axes(self):
        transform = link5_to_grasp()
        np.testing.assert_allclose(transform.translation_mm, [-8.0046325, 0.0, 115.428], atol=1e-9)
        np.testing.assert_allclose(transform.rotation[:, 0], [0.0, 0.0, 1.0], atol=5e-6)
        np.testing.assert_allclose(transform.rotation[:, 1], [-1.0, 0.0, 0.0], atol=5e-6)
        np.testing.assert_allclose(transform.rotation[:, 2], [0.0, -1.0, 0.0], atol=5e-6)

    def test_grasp_fk_matches_transform(self):
        q = JointVector(0.12, -0.66, 2.70, 0.08, -0.05)
        pose = grasp_fk(q)
        np.testing.assert_allclose(pose.as_array()[:3], base_to_grasp(q.as_array()).translation_mm, atol=1e-9)
        self.assertAlmostEqual(pose.roll, q.roll, places=5)

    def test_grasp_jacobian_matches_wider_finite_difference(self):
        q = JointVector(0.08, -0.65, 2.72, 0.03, 0.04)
        analytic = grasp_jacobian(q)
        step = 1e-5
        numeric = np.empty((5, 5))
        center = q.as_array()
        for index in range(5):
            plus, minus = center.copy(), center.copy()
            plus[index] += step
            minus[index] -= step
            numeric[:, index] = pose_error(grasp_fk(minus), grasp_fk(plus)) / (2.0 * step)
        np.testing.assert_allclose(analytic, numeric, atol=2e-5)

    def test_robot_xyz_to_ik_to_fk_grasp_tcp(self):
        seed = JointVector(0.0, -0.65, 2.75, 0.0, 0.0)
        current = grasp_fk(seed)
        target = Pose5(current.x_mm + 6.0, current.y_mm + 4.0, current.z_mm - 3.0, current.pitch + 0.02, current.roll + 0.01)
        result = grasp_ik(target, seed)
        self.assertTrue(result.ok, result.reason)
        residual = pose_error(grasp_fk(result.q), target)
        self.assertLess(float(np.linalg.norm(residual[:3])), 1e-4)
        self.assertLess(float(np.linalg.norm(residual[3:])), 1e-6)


if __name__ == "__main__":
    unittest.main()
