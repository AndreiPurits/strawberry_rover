## rover_perception

ROS 2 perception package for integrating the real RGB camera on the strawberry rover.

### Implemented node

- `rgb_camera_node`: captures frames from a real camera device and publishes ROS image interfaces.
- `camera_preprocess_node`: subscribes to `/camera/image_raw` and publishes grayscale + edge-preprocessed outputs.
- `sensor_fusion_node`: fuses `/scan` and `/camera/preprocessed/edges` into a lightweight semantic/debug output.

### Published topics

- `/camera/image_raw` (`sensor_msgs/msg/Image`): live RGB stream for downstream perception.
- `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`): basic camera metadata (placeholder intrinsics until calibration is available).
- `/camera/debug/image_annotated` (`sensor_msgs/msg/Image`): debug stream with center crosshair and camera label.
- `/camera/preprocessed/gray` (`sensor_msgs/msg/Image`, `mono8`): grayscale preprocessing output.
- `/camera/preprocessed/edges` (`sensor_msgs/msg/Image`, `mono8`): Canny edge output for fast downstream experiments.
- `/perception/fusion_score` (`std_msgs/msg/Float32`): fused confidence score based on front LiDAR + edge density.
- `/perception/plant_candidates` (`visualization_msgs/msg/MarkerArray`): fused RViz2 marker candidates for downstream navigation/debug.

### Launch

```bash
ros2 launch rover_perception rgb_camera.launch.py
ros2 launch rover_perception camera_preprocess.launch.py
ros2 launch rover_perception sensor_fusion.launch.py
```

### Notes

- Camera parameters are configurable with `device_index`, `width`, `height`, `fps`, and `use_v4l2_backend`.
- Camera reconnect logic is enabled by default (`auto_reconnect: true`).

### Troubleshooting (current setup)

- `/dev/video0` works for the connected USB webcam.
- `/dev/video1` does not open for this setup.
- Prefer V4L2 backend (`use_v4l2_backend: true`) and avoid GStreamer for this webcam unless explicitly needed.
