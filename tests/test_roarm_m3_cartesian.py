import unittest

import numpy as np

from pipelines.roarm_m3_kinematics import JointVector, Pose5, fk
from pipelines.roarm_m3_kinematics.cartesian import (
    pixel_depth_to_camera_mm,
    plan_cartesian_pose,
)


class CartesianControllerTest(unittest.TestCase):
    def test_deprojection(self):
        point = pixel_depth_to_camera_mm(420.0, 230.0, 0.12, (440.0, 440.0, 420.0, 230.0))
        np.testing.assert_allclose(point, [0.0, 0.0, 120.0])

    def test_meaningful_step_gate(self):
        q = JointVector(0.0, -0.65, 2.75, 0.0, 0.0)
        pose = fk(q)
        target = Pose5(pose.x_mm + 2.0, pose.y_mm, pose.z_mm, pose.pitch, pose.roll)
        plan = plan_cartesian_pose(target, q, stage="visual", requested_delta=(2, 0, 0, 0, 0))
        self.assertEqual(plan.reason, "XY_BELOW_MEANINGFUL_STEP")

    def test_ik_fk_round_trip_with_deadband(self):
        q = JointVector(0.0, -0.65, 2.75, 0.0, 0.0)
        pose = fk(q)
        target = Pose5(pose.x_mm, pose.y_mm + 8.0, pose.z_mm, pose.pitch, pose.roll)
        plan = plan_cartesian_pose(
            target,
            q,
            stage="visual",
            requested_delta=(0, 8, 0, 0, 0),
            max_joint_step_rad=0.20,
        )
        self.assertTrue(plan.ok)
        self.assertLess(np.linalg.norm(plan.executable_error[:3]), 1.0)


if __name__ == "__main__":
    unittest.main()
