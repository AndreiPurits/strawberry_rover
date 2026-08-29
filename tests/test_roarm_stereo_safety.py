import numpy as np

from pipelines.roarm_m3_kinematics.stereo_safety import (
    feedback_motion_health,
    fit_dominant_plane,
)


def test_dominant_plane_faces_camera_and_rejects_outliers():
    rng = np.random.RandomState(4)
    xy = rng.uniform(-250.0, 250.0, size=(1000, 2))
    wall = np.column_stack((xy, 400.0 + rng.normal(0.0, 1.2, size=len(xy))))
    outliers = rng.uniform([-300.0, -200.0, 100.0], [300.0, 200.0, 900.0], size=(120, 3))
    normal, d, ratio, rmse = fit_dominant_plane(np.vstack((wall, outliers)))
    assert d > 0.0
    assert normal[2] < -0.99
    assert 390.0 < d < 410.0
    assert ratio > 0.80
    assert rmse < 3.0


def test_feedback_health_rejects_post_smoke_sentinel_values():
    result = feedback_motion_health(
        [np.pi, -np.pi, -np.pi / 2.0, -np.pi, np.pi],
        [0, 0, 0, 0, 0, 0],
    )
    assert not result.ok
    assert result.reason == "JOINT_FEEDBACK_OUT_OF_LIMITS"


def test_feedback_health_rejects_all_torque_off():
    result = feedback_motion_health([0.0, -0.7, 2.8, 0.0, 0.0], [0, 0, 0, 0, 0, 0])
    assert not result.ok
    assert result.reason == "ALL_TORQUE_OFF"


def test_feedback_health_accepts_normal_feedback():
    result = feedback_motion_health([0.0, -0.7, 2.8, 0.0, 0.0], [1, 1, 1, 1, 1, 1])
    assert result.ok
