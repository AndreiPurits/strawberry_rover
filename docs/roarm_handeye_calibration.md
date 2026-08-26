# RoArm-M3 / D405 hand–eye calibration

## Fixed frame model

The photographs and the RoArm-M3 chain establish an eye-in-hand topology:

`base_link -> link1 -> link2 -> link3 -> link4 -> link5 -> D405 color optical frame`

The D405 is bolted to the fixed central gripper/tool carrier. It follows
`base`, `shoulder`, `elbow`, `wrist`, and `roll`, while the `hand` command only
changes jaw opening and is not part of camera FK. In the photographs the USB
side of the D405 is visible; the optical axis is approximately parallel to the
longitudinal gripper direction and points toward the jaw tips. The exact offset
and rotation must come from calibration, not image measurement.

Canonical transform notation is `T_parent_child`: it maps a point expressed in
the child frame into the parent frame. Translation is always millimetres.

The transform to solve is:

`T_link5_camera`, where `camera = stereo_camera_color_optical_frame`.

For a fixed board and each static observation `i`:

`T_base_board_i = T_base_link5(q_i) * T_link5_camera * T_camera_board_i`

A valid solution makes all `T_base_board_i` agree.

## Target and camera assumptions

- Use a rigid, flat chessboard with exact measured square size.
- `--cols` and `--rows` are counts of **inner corners**, not squares.
- Keep the board fixed relative to the robot base for the entire dataset.
- The acquisition topic is `/stereo_camera/color/image_rect_raw`.
- The camera frame must be `stereo_camera_color_optical_frame`.
- Because the image is rectified, PnP uses its rectified intrinsics and does not
  apply lens distortion a second time.
- Raw images, corners, PnP pose, reprojection error, T:105 joint feedback, and
  full `T_base_link5` are retained per observation.

The Orin OpenCV build has standard chessboard and hand-eye functions, but no
`cv2.aruco`. The implementation therefore uses a chessboard without replacing
Jetson's OpenCV. ChArUco can be added later in an isolated environment.

## Safe collection procedure

The capture utility has no movement API. The operator manually puts the arm in
one safe, static pose, then captures exactly one observation:

```bash
source scripts/local_ros_env.sh
python3 tools/roarm_handeye_capture.py \
  --dataset runs/roarm_handeye/observations.json \
  --id pose_001 --split train \
  --cols <INNER_COLS> --rows <INNER_ROWS> --square-mm <MEASURED_MM>
```

Repeat with substantially different safe views. Change translation and
orientation, including roll and wrist, while keeping the whole board visible.
Do not use a sequence of nearly identical poses. Collect at least 8 training
and 3 withheld validation observations; more diverse observations are better.

Then solve offline:

```bash
python3 tools/roarm_handeye_calibrate.py \
  runs/roarm_handeye/observations.json \
  --output runs/roarm_handeye/calibration_result.json
```

The solver evaluates Tsai, Park, Horaud, Andreff, and Daniilidis methods. It
reports transform, board consistency on train/validation data, reprojection
residuals, and motion-diversity singular values.

## Independent validation and gate

Solver validation is not physical validation. At least three observations of a
point whose `base_link` position is independently measured are required. Run:

```bash
python3 tools/roarm_handeye_validate.py \
  runs/roarm_handeye/calibration_result.json \
  runs/roarm_handeye/known_points.json \
  --max-error-mm <MEASURED_ACCEPTANCE_LIMIT> \
  --output runs/roarm_handeye/known_point_validation.json
```

The acceptance limit must come from the intended grasp tolerance and measured
repeatability; the software does not invent it. Until this check passes,
`KIN_GATE=FAIL` and Cartesian approach is prohibited.

## What is deliberately not used

- `runs/roarm_kinematics/handeye_M.json`: it is a local 3x3 response fit, not a
  rigid SE(3) camera extrinsic.
- `scripts/roarm_cartesian_approach.py` calibration probes: they physically move
  the arm and estimate another local response matrix.
- Hand/gripper position in camera FK.
- A transform estimated from photographs or nominal bracket dimensions.
