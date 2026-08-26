# RoArm-M3 Phase A report — canonical FK / IK / Jacobian

## Outcome

Phase A is implemented and passes offline validation. No physical movement was
performed. The new package has no transport/control dependency and is not wired
into the working strawberry runtime, so production remains legacy.

Baseline commit before Phase A: `b04ad48fb` (`Freeze working strawberry approach baseline`).

## New files

- `pipelines/roarm_m3_kinematics/__init__.py`
- `pipelines/roarm_m3_kinematics/conventions.py`
- `pipelines/roarm_m3_kinematics/model.py`
- `tests/test_roarm_m3_kinematics.py`
- `tests/data/roarm_m3_t105_samples.json`
- `tools/validate_roarm_m3_kinematics.py`
- `tools/roarm_m3_ikfast_reference.cpp`
- `docs/roarm_m3_kinematics.md`
- `reports/roarm_m3_phase_a_validation.json`
- `reports/roarm_m3_ikfast_crosscheck.json`
- this report

The package name avoids a collision with the existing legacy file
`pipelines/roarm_kinematics.py`; that file and the berry planner are unchanged.

## Geometry and formulas

Canonical dimensions come from the official Waveshare firmware
`RoArm-M3_config.h`; FK/T:105 equations come from
`RoArmM3_computePosbyJointRad` in `RoArm-M3_module.h`:

- L2 = hypot(236.82, 30.00) = 238.7126146646 mm; d2 = 0.1260073188 rad
- L3 = 144.49 mm; d3 = 0 rad
- tool/edge = hypot(171.67, 13.69) = 172.2149964434 mm;
  dE = 0.0795776201 rad

Independent references are the official ROS2 Xacro and generated IKFast solver
from `waveshareteam/roarm_ws`, branch `ros2-humble`.

## Convention

`q = [base, shoulder, elbow, wrist, roll]` in firmware radians.

`pose = [x_mm, y_mm, z_mm, pitch_rad, roll_rad]`, where
`pitch = shoulder + elbow + wrist - pi/2` and pose roll equals the roll joint.
`hand` is separate and never enters FK/IK/Jacobian.

Three limit profiles are explicit: firmware saturation, conservative
operational hard limits, and planning soft limits. Full tables and the
URDF/firmware elbow discrepancy are documented in
`docs/roarm_m3_kinematics.md`.

## Quantitative results

### FK against real T:105

Two identical read-only feedback replies at one unchanged physical pose were
captured; no movement command was sent.

| Metric | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| XYZ norm error, mm | 9.7446e-8 | 9.7446e-8 | 9.7446e-8 | 9.7446e-8 |
| pitch abs error, rad | 2.0510e-10 | 2.0510e-10 | 2.0510e-10 | 2.0510e-10 |
| roll abs error, rad | 0 | 0 | 0 | 0 |

The repo's existing run logs contain many joint states but do not preserve the
paired T:105 `x/y/z/tit` fields. Therefore this proves exact agreement for one
unique real pose, not broad real-pose coverage. Phase D should log paired
feedback for every later static target.

### Jacobian finite differences

2,000 random in-envelope states, central step 1e-6 rad, 50,000 matrix elements:

| Absolute element error | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| analytic vs numeric | 6.0053e-9 | 1.3978e-10 | 3.0591e-8 | 1.1264e-7 |

The largest derivative magnitude in the worst comparison was about 212.26
mm/rad, so the worst absolute numerical discrepancy is about 5.3e-10 relative.

### Analytic IK → FK round-trip

10,000 deterministic random targets generated from in-envelope q:

- success: 10,000 / 10,000
- explicit failure: 0
- selected outside hard limits: 0
- XYZ residual max: 4.7580e-13 mm; p95: 1.7112e-13 mm
- orientation residual max: 9.9301e-16 rad
- selected-to-seed joint distance max: 6.8580e-15 rad
- candidates per target: 2 candidates for 5,811 targets and 4 candidates for
  4,189 targets; the seed selector returned the continuous branch

An unreachable 2,000 mm target returns `IK_NO_SOLUTION` with no fallback.

### Official IKFast cross-check

The official generated solver was compiled standalone and evaluated on 2,000
states in the shared safe/URDF range:

- official IKFast FK→IK success: 2,000 / 2,000
- solution count: min 2, median 4, max 8
- nearest original-q error: mean 7.7519e-15 rad, p95 2.9920e-14 rad,
  max 1.5945e-13 rad
- our analytic firmware FK→IK on the same q rows: 2,000 / 2,000, max nearest-q
  error 5.4404e-15 rad

This independently confirms joint ordering/sign continuity and multiple-solution
handling. Raw endpoint XYZ are not directly interchangeable: IKFast uses URDF
`hand_tcp`, T:105 uses the firmware edge/tool endpoint. Across the 2,000 rows,
the uncalibrated raw endpoint difference reached 54.85 mm, which is expected
from the distinct fixed frames/tool lengths and must not be hidden as an offset.

### Conditioning and problem regions

A 50,000-state scan used a dimensionless Jacobian with XYZ divided by total
model reach (555.4176 mm): median condition 15.62, p95 54.38, maximum 79,369.5.
The worst sampled pose had radial reach about 0.013 mm, exposing the exact
`rho=0` base singularity.

Exact rank-loss sets:

- `rho(q)=0`;
- `sin(elbow + d3 - d2)=0`.

The relevant elbow singularity is `elbow≈0.1260 rad`, just below the
conservative hard minimum 0.20 rad. Near-zero radial reach remains reachable in
otherwise allowed joint ranges, so a future motion gate must inspect actual
conditioning. Phase A intentionally does not invent a pass threshold; that
requires measured accuracy/repeatability before physical Cartesian tests.

## Regression and safety

- Existing legacy kinematics/planner files: unchanged after baseline commit.
- New package imports no arm client and sends no commands.
- Unit tests: 7/7 pass.
- No T:101, T:102, T:104, gripper, cut, yank, or homing command was sent.
- The only live-arm access was read-only T:105 feedback.

## Ready for Phase B

Ready:

- canonical firmware endpoint FK;
- closed-form multi-branch IK with seed/limit-aware selection;
- FK round-trip gating and explicit `IK_NO_SOLUTION`;
- physical 5x5 Jacobian, finite-difference proof, conditioning diagnostic, and
  damped pseudoinverse utility;
- explicit frame/units convention and conservative limit profiles;
- reproducible offline validation scripts/data.

Still blocked before any Cartesian stem motion:

- determine actual camera mount and the correct parent link;
- establish a real `T_tool_camera` (or correct alternative) from diverse board
  observations;
- independently validate the extrinsic and then define `T_tool_grasp`;
- set motion conditioning/delta thresholds from measured accuracy, not guesses.

Phase B has not been started.
