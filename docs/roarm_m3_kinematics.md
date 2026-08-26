# Canonical RoArm-M3 kinematics (Phase A)

This package is an offline, opt-in mathematical model. It contains no HTTP,
RPC, serial, ROS publisher, or servo command code. It deliberately coexists
with the legacy `pipelines/roarm_kinematics.py` and is not imported by the
production berry approach.

## Public API

```python
from pipelines.roarm_m3_kinematics import fk, ik, jacobian

q = [base, shoulder, elbow, wrist, roll]
pose = fk(q)                              # x/y/z mm, pitch/roll rad
result = ik(pose, seed_q=q)              # explicit OK / IK_NO_SOLUTION
J = jacobian(q)                          # 5x5, xyz mm/rad
```

Also exported:

- `within_limits(q, kind="soft|hard|firmware")`
- `joint_limit_margins(q, kind=...)`
- `pose_error(current, target)`
- `select_ik_solution(candidates, seed_q=None)`
- `jacobian_condition(q)`
- `damped_pseudoinverse(J, damping=..., translation_scale_mm=...)`

`hand` is a separate gripper-opening command and is never part of pose IK.

## Sources

The canonical firmware endpoint model comes from the official Waveshare
RoArm-M3 Arduino example distributed from the product wiki:

- `RoArm-M3_config.h`: link vectors and servo command saturation
- `RoArm-M3_module.h`: `RoArmM3_computePosbyJointRad`, T:105 feedback, and the
  official coordinate IK implementation

The independent model is the official ROS2 `ros2-humble` workspace:

- `roarm_description/urdf/roarm_m3/roarm_m3.xacro`
- `roarm_moveit_ikfast_plugins/src/roarm_m3_hand_ikfast_solver.cpp`

Public references:

- https://www.waveshare.com/wiki/RoArm-M3
- https://github.com/waveshareteam/roarm_ws/tree/ros2-humble

The firmware and URDF end frames are not identical: T:105 reports the firmware
edge/tool endpoint, while MoveIt/IKFast targets `hand_tcp`. Raw XYZ must not be
mixed without an explicit fixed-frame/tool transform.

## Joint and pose convention

All joint inputs are firmware radians in this exact order:

`q = [base, shoulder, elbow, wrist, roll]`

The output pose is:

`[x_mm, y_mm, z_mm, pitch_rad, roll_rad]`

- At base=0, positive radial reach lies on +X.
- Positive base rotates +X toward +Y (firmware describes this as left).
- Positive shoulder leans the arm forward.
- Positive elbow increases the firmware elbow angle (firmware: down/fold).
- Wrist contributes one-for-one to tool pitch.
- Roll is rotation about the longitudinal tool axis and does not change XYZ in
  the firmware endpoint model.
- No extra shoulder bias or learned correction coefficient exists here.
- T:105 `x/y/z` are millimetres; `b/s/e/t/r/tit` are radians.

## Geometry

The official firmware stores link vectors rather than idealized scalar links:

| Vector | A (mm) | B (mm) | length (mm) | angular offset (rad) |
|---|---:|---:|---:|---:|
| L2 | 236.82 | 30.00 | 238.7126146646 | 0.1260073188 |
| L3 | 144.49 | 0.00 | 144.4900000000 | 0.0000000000 |
| tool/edge | 171.67 | 13.69 | 172.2149964434 | 0.0795776201 |

The old project values 285/375 mm are not used.

## FK

Let:

```text
phi2 = shoulder + d2
phi3 = shoulder + elbow + d3
phiE = shoulder + elbow + wrist + dE
rho  = L2 sin(phi2) + L3 sin(phi3) + LE sin(phiE)
z    = L2 cos(phi2) + L3 cos(phi3) + LE cos(phiE)
x    = rho cos(base)
y    = rho sin(base)
pitch = shoulder + elbow + wrist - pi/2
roll  = roll_joint
```

This reproduces the paired real T:105 feedback to floating-point/printed-JSON
precision.

## Analytic IK

For target `(x,y,z,pitch,roll)` and a chosen signed radial/base branch:

```text
rho_w = rho - LE cos(pitch + dE)
z_w   = z   + LE sin(pitch + dE)
D     = (rho_w^2 + z_w^2 - L2^2 - L3^2) / (2 L2 L3)
psi   = +/- acos(D)
phi2  = atan2(rho_w,z_w) - atan2(L3 sin(psi), L2 + L3 cos(psi))
shoulder = phi2 - d2
elbow    = psi + d2 - d3
wrist    = pitch + pi/2 - shoulder - elbow
base     = atan2(y,x) (plus the signed backward radial branch)
roll     = target roll
```

Every candidate is FK-verified. The selector rejects hard-limit violations,
penalizes soft-limit violations and edges, and minimizes wrapped joint travel
from `seed_q`. If no hard-valid solution exists, the result is explicitly
`IK_NO_SOLUTION`; there is no joint-delta fallback.

## Jacobian and conditioning

The Jacobian maps canonical joint increments to
`[dx_mm,dy_mm,dz_mm,dpitch,droll]`. Translation and orientation have different
units, so condition-number diagnostics normalize XYZ by total model reach
(555.4176111080 mm). This normalization is diagnostic, not a physical-movement
gate or tuned safety threshold.

Exact rank-loss sets include:

- `rho(q) = 0`: base rotation produces no Cartesian translation;
- `sin(elbow + d3 - d2) = 0`: the two main planar links are collinear.

The in-envelope elbow singularity is approximately `elbow = d2 =
0.1260073188 rad`, below the current operational hard minimum of 0.20 rad but
close enough that the lower-elbow region remains poorly conditioned. A second
practical problem region occurs when a folded configuration drives `rho` near
zero.

## Limits

Three distinct meanings are retained rather than silently conflated:

| Joint | firmware/model saturation | operational hard | planning soft |
|---|---|---|---|
| base | [-pi, pi] | [-1.57, 1.57] | [-1.25, 1.25] |
| shoulder | [-pi/2, pi/2] | [-pi/2, pi/2] | [-1.35, 1.45] |
| elbow | approximately [0, pi] | [0.20, pi] | [0.35, 3.05] |
| wrist | [-pi/2, pi/2] | [-pi/2, pi/2] | [-1.30, 1.30] |
| roll | [-pi, pi] | [-pi/2, pi/2] | [-1.30, 1.30] |

The operational limits are deliberately conservative. Roll remains restricted
until cable and tool clearance are physically validated. The official URDF has
different elbow limits `[-1.0, 2.95]`; it describes the MoveIt model, while the
firmware servo command path clamps the installed arm near `[0, pi]`.

The latest read-only T:105 shoulder feedback was 1.636757501 rad, about 0.066
rad above the firmware shoulder command saturation. This is evidence of the
known feedback/command offset under load, not a reason to widen command limits.

## Phase boundary

Phase A does not add a runtime `kinematic` motion path. Production therefore
remains legacy by construction. A future integration switch must default to
legacy and must refuse kinematic motion until camera/tool calibration and all
later gates pass.
