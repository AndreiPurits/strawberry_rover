## rover_fake_lidar

ROS 2 package providing a fake LiDAR sensor interface for the strawberry rover.

Intended for simulation, testing, and development without real hardware.

### Published topics

- `/scan` (`sensor_msgs/LaserScan`): simulated LiDAR scan for navigation/perception development.
- `/sim/scene_markers` (`visualization_msgs/MarkerArray`): scene debug markers designed for RViz2 and future server streaming.

### Subscribed topics

- `/sim/rover_pose` (`geometry_msgs/PoseStamped`): simulated rover pose used to place the rover marker in the scene.

### Scene markers included (baseline simplified scene)

- five field rows/beds
- rover body (updated from `/sim/rover_pose`)

Optional:
- explicit obstacle markers (`include_explicit_obstacles: true`)

### Field-row layout used in simulation

- 5 beds (rows) with configurable spacing
- no solid corridor walls
- no pole/post lines in baseline mode (simplified deterministic route scene)
- marker frame defaults to `sim_world`
- scan rays are computed against optional explicit obstacles when enabled

