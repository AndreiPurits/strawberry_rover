## rover_navigation

ROS 2 package for row-following navigation in field simulation using fake LiDAR.

### Implemented node

- `rover_navigation_node`
  - subscribes: `/scan` (`sensor_msgs/msg/LaserScan`)
  - subscribes: `/sim/rover_pose` (`geometry_msgs/msg/PoseStamped`)
  - publishes: `/cmd_vel` (`geometry_msgs/msg/Twist`)
  - publishes: `/debug/steering` (`std_msgs/msg/Float32`)
  - publishes: `/debug/centerline` (`visualization_msgs/msg/Marker`)
  - publishes: `/debug/heading` (`visualization_msgs/msg/Marker`)
  - publishes: `/debug/row_confidence` (`visualization_msgs/msg/Marker`)
  - publishes: `/debug/nav_state` (`visualization_msgs/msg/Marker`)

### Behavior

- computes average left/right probe distances from scan
- steering error = `right - left` with temporal smoothing
- runs deterministic 5-row FSM:
  - `FOLLOW_ROW`
  - `END_OF_ROW`
  - `TURN_TO_NEXT_ROW`
  - `ALIGN_NEXT_ROW`
  - `FINISHED`
- end-of-row is detected by pose boundary + front distance threshold
- next-row transition uses deterministic turn and cross-row alignment by `/sim/rover_pose`
- clamps steering change per update (`max_steer_delta`)
- stops in `FINISHED` after completing row 5
- publishes centerline/heading debug marker for RViz
- publishes color-coded row confidence + explicit nav-state marker

### Launch

```bash
ros2 launch rover_navigation rover_navigation.launch.py
```
