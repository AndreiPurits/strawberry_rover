## rover_description

ROS 2 robot description package for the strawberry rover.

Use this package to store URDF/Xacro files, meshes, and related configuration describing the rover's structure.

Current URDF includes:
- `base_link`
- `lidar_link`
- `camera_front_link`
- `stereo_link`
- `bottom_rgb_camera_link`
- arch leg links (`left_arch_leg_link`, `right_arch_leg_link`) for over-bed clearance

Current rover concept (simplified but field-oriented):
- arch-over-bed geometry with approximate dimensions `1.3m (L) x 1.2m (W) x 1.2m (H)`;
- front camera mount near `x=+0.45, z=+0.85`;
- stereo mount near `x=+0.40, z=+0.80` (`stereo_link`);
- bottom RGB mount near `x=0.0, z=+0.55`, looking down to the bed area.

These sensor links are static children of `base_link` and are used by simulation and perception nodes.

