# RoArm-M3 grasp TCP (`link5_grasp_v1`)

The official Waveshare `ros2-humble` Xacro defines `hand_tcp` at
`xyz=[0, 0, 115.428] mm`, `rpy=[1.5708, -1.5708, 0]` from `link5`.
The official closed-jaw STL contact faces are at `link5.x=-10.025849 mm`
and `link5.x=-5.983416 mm`. Their physical mid-plane is therefore
`link5.x=-8.0046325 mm`. The project grasp frame keeps the official
longitudinal position and orientation but moves the origin into this jaw
mid-plane:

`T_link5_grasp: xyz=[-8.0046325, 0, 115.428] mm, rpy=[1.5708, -1.5708, 0]`.

Axes are fixed and independent of the hand opening command:

- `+X_grasp`: approach, approximately `+Z_link5`;
- `+Y_grasp`: jaw-closing direction, approximately `-X_link5`;
- `+Z_grasp`: right-handed normal, approximately `-Y_link5`.

`grasp_fk`, `grasp_jacobian`, and seeded damped `grasp_ik` operate on this
frame. They are opt-in and are not imported by the legacy berry planner.

Source: <https://github.com/waveshareteam/roarm_ws>, branch `ros2-humble`,
commit `40dbd84b553695212fab713e8465f817ba95454d`.

Physical board-only validation used three independent -8 mm Cartesian moves.
Fresh T:105 and 54/54 checkerboard observations were acquired after every
move. Board-relative known-point error was 1.45, 1.63, and 1.49 mm. A
conservative 3 mm TCP uncertainty is retained for planning.
