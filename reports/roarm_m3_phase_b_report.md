# RoArm-M3 Phase B report — camera/tool calibration

Date: 2026-08-26

## Outcome

The camera topology is resolved and the Phase B calibration/validation stack is
implemented. No robot movement was commanded. A production extrinsic is not yet
available because no metric board observations or independent known-point
measurements were supplied. Therefore `KIN_GATE=FAIL` and Phase C/D motion is
not authorized.

## Camera mounting conclusion

- Device: RealSense D405 (`8086:0b5b`).
- Configuration: eye-in-hand.
- Fixed kinematic parent: `link5` (the central distal tool/gripper carrier after
  the roll joint).
- The camera follows all five pose joints: base, shoulder, elbow, wrist, roll.
- The camera does not follow `hand`; jaw opening is excluded from camera FK.
- The USB edge visible in both photographs shows that the optical axis is
  approximately along the gripper's longitudinal direction, toward the jaw
  tips. Exact translation/rotation remains a calibration result.
- Calibration frame: `stereo_camera_color_optical_frame`.

## Live Orin camera facts

- Color: `/stereo_camera/color/image_rect_raw`
- Color CameraInfo: `/stereo_camera/color/camera_info`
- Aligned depth: `/stereo_camera/aligned_depth_to_color/image_raw`
- Resolution: 848 x 480
- K: `[437.5350646972656, 0, 422.8022155761719; 0, 436.4046325683594, 230.43247985839844; 0, 0, 1]`
- Distortion model: `plumb_bob`
- D: `[-0.053496066480875015, 0.057618286460638046, 0.000026341296688769944, -0.00014131066564004868, -0.019024858251214027]`

## Implemented

- Named, validated SE(3) transforms using `T_parent_child` notation.
- Full official-Xacro orientation FK to `link5` and `hand_tcp`.
- Metric chessboard detection, subpixel corners, solvePnP, and reprojection RMSE.
- Static one-observation capture from D405 + read-only T:105 feedback; the tool
  contains no movement command.
- Multi-method OpenCV eye-in-hand solver for `T_link5_camera`.
- Train/validation board-consistency residuals and motion-diversity ranks/SVD.
- Independent known-point validation with an explicit measured threshold.
- Fail-closed config: no transform, `NOT_CALIBRATED`, `KIN_GATE=FAIL`.

## Quantitative offline validation

Official IKFast cross-check, 5 poses:

- maximum TCP translation difference: 0.009705 mm;
- mean TCP translation difference: 0.006736 mm;
- maximum tool +Z direction-vector difference: 5.044e-6.

Exact synthetic hand-eye dataset, 14 train + 4 validation poses:

- selected solver: PARK;
- recovered translation error: 4.780e-14 mm;
- recovered rotation error: 0 degrees;
- validation board translation residual median: 6.417e-14 mm;
- validation board rotation residual median: 1.457e-6 degrees;
- train relative-motion rotation rank: 3;
- train relative-motion translation rank: 3.

Phase B unit tests: 6/6 PASS. Combined Phase A + B regression after deployment:
13/13 PASS.

## Remaining Phase B physical work

1. Print or mount a rigid chessboard and provide the exact inner-corner counts
   and measured square edge in millimetres.
2. Keep it fixed to the bench/base reference.
3. Manually place the arm into diverse safe poses and capture one observation
   at a time (minimum 8 train + 3 validation; use wrist and roll diversity).
4. Solve offline and inspect all methods/residuals/conditioning.
5. Measure an independent point in `base_link`, capture it from at least three
   poses, choose an acceptance tolerance from grasp needs/repeatability, and run
   the validation gate.

Until step 5 passes: do not start pixel-to-base runtime integration and do not
perform Cartesian stem approach.
