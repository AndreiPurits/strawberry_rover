## rover_fake_stereo

ROS 2 package providing a fake stereo camera interface for the strawberry rover.

Node:
- `fake_stereo_node`

Published topics:
- `/sim/stereo/left/image_raw` (`sensor_msgs/msg/Image`)
- `/sim/stereo/right/image_raw` (`sensor_msgs/msg/Image`)
- `/sim/stereo/debug/combined` (`sensor_msgs/msg/Image`)

Synthetic stereo frames are tied to `/sim/rover_pose` and use two baseline-separated virtual viewpoints.
Left and right frames are intentionally non-identical to provide simple logical parallax cues.

RViz2 note:
- Stereo display plugin is not supported in many ROS2 Foxy RViz2 builds.
- Use two separate `Image` displays (left and right topics), or one `Image` display on `/sim/stereo/debug/combined`.

Notes:
- Default image QoS is `reliable` (`image_qos_reliability: "reliable"`), so RViz Image display works with default QoS in ROS2 Foxy.
- If needed for testing, QoS can be switched to `best_effort`.

