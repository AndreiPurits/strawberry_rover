## rover_fake_camera

ROS 2 package providing a fake monocular camera interface for the strawberry rover.

Node:
- `fake_camera_node`

Published topics:
- `/sim/camera/front/image_raw` (`sensor_msgs/msg/Image`, frame `camera_front_link`)
- `/sim/camera/bottom_rgb/image_raw` (`sensor_msgs/msg/Image`, frame `bottom_rgb_camera_link`)

The node generates:
- front navigation view (current row perspective, row progression),
- bottom RGB view (bed zone under rover arch, lateral offset relative to bed center, leaves/berries),
both tied to `/sim/rover_pose`.

Notes:
- Default image QoS is `reliable` (`image_qos_reliability: "reliable"`), which is RViz-friendly in ROS2 Foxy.
- Set `image_qos_reliability: "best_effort"` only if a consumer explicitly requires sensor-data QoS.

