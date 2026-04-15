# Strawberry Rover ROS2 Topic Contract

Single source of truth for ROS2 interfaces across packages.  
This contract is used by all agents to keep interfaces stable for RViz2 now and server/web monitoring later.

Last updated from codebase state: simplified field-row simulation (5 beds + rover), deterministic scripted bed traversal by default, pole-free baseline scene, virtual cameras integrated in bringup, and optional navigation node enablement.

Status labels:
- `implemented`: present in current code/runtime
- `in_progress`: part of active roadmap phase but not fully integrated yet
- `planned`: defined by roadmap/context, not implemented yet

## Core active topics

| Topic | Message type | Publisher package | Publisher node | Main subscribers/consumers | frame_id | Expected rate | QoS assumptions | Classification | Status | Purpose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/scan` | `sensor_msgs/msg/LaserScan` | `rover_fake_lidar` | `FakeLidarNode` | future navigation node, RViz2 LaserScan display | `lidar_link` | `10 Hz` (default `publish_rate`) | keep-last depth `10`, reliable, volatile (from `create_publisher(..., 10)`) | fake/simulated sensor | `implemented` | Simulated LiDAR for navigation/perception development without hardware |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | `rover_perception` | `rgb_camera_node` | perception pipeline, RViz2 image display, future web stream bridge | `camera_link` (configurable via `frame_id`) | `15 Hz` default (`publish_rate`) | `qos_profile_sensor_data` | real sensor | `implemented` | Primary real RGB image stream (used in real-hardware mode, not in simulation bringup) |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | `rover_perception` | `rgb_camera_node` | perception calibration consumers, image pipeline, RViz2 | `camera_link` | `15 Hz` default when frames are available | `qos_profile_sensor_data` | real sensor | `implemented` | Camera metadata stream (placeholder intrinsics until calibration; real mode) |
| `/perception/fusion_score` | `std_msgs/msg/Float32` | `rover_perception` | `sensor_fusion_node` | future navigation logic, tuning tools, monitoring | n/a | `10 Hz` default (`fusion_rate`) | keep-last depth `10` | perception fusion | `implemented` | Lightweight fused confidence from front LiDAR and camera edge density |
| `/robot_description` | `std_msgs/msg/String` | `robot_state_publisher` (via `rover_bringup`) | `robot_state_publisher` | RViz2, tools introspection, future model-aware monitoring | n/a | startup and on parameter updates | transient-local style behavior from robot_state_publisher | debug/visualization, future server/web monitoring | `implemented` | Exposes active robot model description to ROS graph consumers |
| `/joint_states` | `sensor_msgs/msg/JointState` | joint state source TBD (none in stack yet) | node TBD | `robot_state_publisher` | joint frame references | TBD | sensor-data style expected when added | TF support | `in_progress` | Input contract for dynamic TF updates; currently static TF works without active joint publisher |
| `/parameter_events` | `rcl_interfaces/msg/ParameterEvent` | all active ROS2 nodes | ROS2 internal per node | tooling, introspection, debugging | n/a | event-driven | ROS2 default parameter event QoS | debug/visualization | `implemented` | Parameter change visibility and runtime introspection |
| `/rosout` | `rcl_interfaces/msg/Log` | all active ROS2 nodes | ROS2 logger per node | `ros2 topic echo`, logging tools, operators | n/a | event-driven | ROS2 default logging QoS | debug/visualization, future server/web monitoring | `implemented` | Centralized runtime logs |

## Simulation topics

| Topic | Message type | Publisher package | Publisher node | Main subscribers/consumers | frame_id | Expected rate | QoS assumptions | Classification | Status | Purpose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/sim/scene_markers` | `visualization_msgs/msg/MarkerArray` | `rover_fake_lidar` | `FakeLidarNode` | RViz2 Marker display, future monitoring bridge/backend | `sim_world` | `10 Hz` (published on LiDAR timer) | keep-last depth `10`, reliable, volatile | debug/visualization, future server/web monitoring | `implemented` | Simplified field markers: 5 beds/rows + rover marker coupled to `/sim/rover_pose`; pole/post marker lines removed from baseline scene; explicit obstacle markers remain optional |
| `/sim/rover_pose` | `geometry_msgs/msg/PoseStamped` | `rover_simulation` | `rover_pose_simulator` | `rover_fake_lidar` marker placement, RViz2 pose tools, future monitoring backend | `base_link` | `20 Hz` | keep-last depth `10` | simulation state, debug/visualization, future server/web monitoring | `implemented` | Simulated rover pose in bringup defaults to deterministic endless scripted traversal along bed centerlines with end-of-row shift to adjacent bed and direction flip |
| `/sim/camera/front/image_raw` | `sensor_msgs/msg/Image` | `rover_fake_camera` | `fake_camera_node` | RViz2 Image display, perception prototyping, future web stream bridge | `camera_front_link` | `10 Hz` default | reliable keep-last depth `10` by default (`image_qos_reliability` configurable) | fake/simulated sensor, debug/visualization | `implemented` | Clean pose-driven front image (no helper overlays): depends only on rover pose + yaw + bed geometry inside field bounds (baseline scene has no poles/posts) |
| `/sim/camera/bottom_rgb/image_raw` | `sensor_msgs/msg/Image` | `rover_fake_camera` | `fake_camera_node` | RViz2 Image display, berry/plant perception prototyping | `bottom_rgb_camera_link` | `10 Hz` default | reliable keep-last depth `10` by default (`image_qos_reliability` configurable) | fake/simulated sensor, debug/visualization | `implemented` | Bottom camera renders bed only where bed footprint truly exists in world coordinates; outside bed/field footprint it renders soil/background (no hallucinated bed pattern) |
| `/sim/stereo/left/image_raw` | `sensor_msgs/msg/Image` | `rover_fake_stereo` | `fake_stereo_node` | stereo/depth prototyping, RViz2, future perception | `stereo_link` | `10 Hz` default | reliable keep-last depth `10` by default (`image_qos_reliability` configurable) | fake/simulated sensor | `implemented` | Dynamic stereo-left image from virtual camera offset by `-baseline/2` in rover lateral axis; depends on rover pose and bed geometry |
| `/sim/stereo/right/image_raw` | `sensor_msgs/msg/Image` | `rover_fake_stereo` | `fake_stereo_node` | stereo/depth prototyping, RViz2, future perception | `stereo_link` | `10 Hz` default | reliable keep-last depth `10` by default (`image_qos_reliability` configurable) | fake/simulated sensor | `implemented` | Dynamic stereo-right image from virtual camera offset by `+baseline/2`; not identical to left, with logical parallax cues over bed geometry |
| `/sim/stereo/debug/combined` | `sensor_msgs/msg/Image` | `rover_fake_stereo` | `fake_stereo_node` | RViz2 Image display, operator debug | `stereo_link` | `10 Hz` default | reliable keep-last depth `10` | debug/visualization | `implemented` | Left+right combined stereo debug image for RViz2 builds where Stereo display is not supported |

## TF-related topics

| Topic | Message type | Publisher package | Publisher node | Main subscribers/consumers | frame_id | Expected rate | QoS assumptions | Classification | Status | Purpose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/tf` | `tf2_msgs/msg/TFMessage` | `rover_simulation` + `robot_state_publisher` | `rover_pose_simulator` + `robot_state_publisher` | RViz2, localization/navigation/perception nodes | dynamic `sim_world -> base_link`; static sensor frames under `base_link` | `20 Hz` dynamic + static broadcasts | TF default QoS (volatile) | TF | `implemented` | Runtime transform tree coupling rover virtual motion and sensor frames |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | `rover_description` (when launched) | `robot_state_publisher` | RViz2 and all TF consumers | static transforms such as `base_link -> lidar_link` | startup + latched behavior | TF static QoS (transient local) | TF | `implemented` | Static robot frame tree for consistent sensor geometry |

Known TF frames in current URDF/simulation:
- `sim_world` (dynamic parent frame from simulator)
- `base_link`
- `lidar_link` (fixed joint from `base_link`)
- `camera_front_link` (fixed joint from `base_link`)
- `stereo_link` (fixed joint from `base_link`)
- `bottom_rgb_camera_link` (fixed joint from `base_link`)

## Visualization/monitoring topics

| Topic | Message type | Publisher package | Publisher node | Main subscribers/consumers | frame_id | Expected rate | QoS assumptions | Classification | Status | Purpose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/sim/scene_markers` | `visualization_msgs/msg/MarkerArray` | `rover_fake_lidar` | `FakeLidarNode` | RViz2, future web/server monitoring bridge | `sim_world` | `10 Hz` | keep-last depth `10`, reliable, volatile | debug/visualization, future server/web monitoring | `implemented` | Shared simplified field visualization contract (5 rows + rover; no posts in baseline, optional explicit obstacles only when enabled) |
| `/sim/camera/front/image_raw` | `sensor_msgs/msg/Image` | `rover_fake_camera` | `fake_camera_node` | RViz2, future web monitoring bridge | `camera_front_link` | `10 Hz` | reliable keep-last depth `10` by default | debug/visualization, future server/web monitoring | `implemented` | Virtual front camera stream for simulation mode |
| `/sim/camera/bottom_rgb/image_raw` | `sensor_msgs/msg/Image` | `rover_fake_camera` | `fake_camera_node` | RViz2, future web monitoring bridge | `bottom_rgb_camera_link` | `10 Hz` | reliable keep-last depth `10` by default | debug/visualization, future server/web monitoring | `implemented` | Bottom RGB camera stream under rover arch |
| `/sim/stereo/left/image_raw` | `sensor_msgs/msg/Image` | `rover_fake_stereo` | `fake_stereo_node` | RViz2 (Image display), future stereo pipeline | `stereo_link` | `10 Hz` | reliable keep-last depth `10` by default | debug/visualization | `implemented` | Dynamic stereo left image tied to rover pose and field geometry |
| `/sim/stereo/right/image_raw` | `sensor_msgs/msg/Image` | `rover_fake_stereo` | `fake_stereo_node` | RViz2 (Image display), future stereo pipeline | `stereo_link` | `10 Hz` | reliable keep-last depth `10` by default | debug/visualization | `implemented` | Dynamic stereo right image with baseline-based viewpoint offset |
| `/sim/stereo/debug/combined` | `sensor_msgs/msg/Image` | `rover_fake_stereo` | `fake_stereo_node` | RViz2 Image display | `stereo_link` | `10 Hz` | reliable keep-last depth `10` | debug/visualization | `implemented` | RViz-friendly combined stereo stream (left|right) |
| `/debug/nav_state` | `visualization_msgs/msg/Marker` | `rover_navigation` | `rover_navigation_node` | RViz2, future web monitoring bridge | `base_link` | `10 Hz` | reliable keep-last depth `10` | debug/visualization, future server/web monitoring | `implemented` | Text marker with current navigation FSM state, active row index and traversal direction |
| `/camera/debug/image_annotated` | `sensor_msgs/msg/Image` | `rover_perception` | `rgb_camera_node` | RViz2 image debug view, future monitoring backend | `camera_link` | `15 Hz` default when enabled | `qos_profile_sensor_data` | debug/visualization, future server/web monitoring | `implemented` | Perception debug image with center crosshair and camera label |
| `/camera/preprocessed/gray` | `sensor_msgs/msg/Image` | `rover_perception` | `camera_preprocess_node` | perception modules, RViz2 image view, future backend | `camera_link` | source camera rate (up to configured RGB rate) | `qos_profile_sensor_data` | debug/visualization, perception preprocessing | `implemented` | Grayscale preprocessing output for lightweight downstream algorithms |
| `/camera/preprocessed/edges` | `sensor_msgs/msg/Image` | `rover_perception` | `camera_preprocess_node` | perception modules, RViz2 image view, future backend | `camera_link` | source camera rate (up to configured RGB rate) | `qos_profile_sensor_data` | debug/visualization, perception preprocessing | `implemented` | Canny edge preprocessing output for row/plant structure experiments |
| `/perception/plant_candidates` | `visualization_msgs/msg/MarkerArray` | `rover_perception` | `sensor_fusion_node` | RViz2 marker display, future monitoring backend, navigation debug | `lidar_link` | `10 Hz` default (`fusion_rate`) | keep-last depth `10` | debug/visualization, perception fusion, future server/web monitoring | `implemented` | Fused candidate markers and confidence labels from RGB preprocessing and LiDAR |
| `/rosout` | `rcl_interfaces/msg/Log` | all active ROS2 nodes | ROS2 logger per node | ops dashboards, backend log ingest (future) | n/a | event-driven | ROS2 default logging QoS | future server/web monitoring | `implemented` | Candidate source for remote runtime status/log forwarding |

## Planned future topics (conservative)

| Topic | Message type | Publisher package | Publisher node | Main subscribers/consumers | frame_id | Expected rate | QoS assumptions | Classification | Status | Purpose |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `rover_navigation` (manual publishers still possible for testing) | `rover_navigation_node` | `rover_simulation` (`rover_pose_simulator`), future rover base controller | `base_link` command frame assumption | `10 Hz` default from nav loop | keep-last depth `10`, reliable | control | `implemented` | Navigation motion command output when navigation is enabled; simplified bringup defaults to scripted `rover_pose_simulator` traversal and does not require `/cmd_vel` as primary route source |
| `/debug/centerline` | `visualization_msgs/msg/Marker` | `rover_navigation` | `rover_navigation_node` | RViz2, future monitoring backend | `base_link` (configurable via `debug_frame_id`) | `10 Hz` default | keep-last depth `10`, reliable | debug/visualization, future server/web monitoring | `implemented` | Smoothed row-centerline heading marker for navigation debug |
| `/debug/steering` | `std_msgs/msg/Float32` | `rover_navigation` | `rover_navigation_node` | RViz2/tools/backend | n/a | `10 Hz` default from nav loop | keep-last depth `10`, reliable | debug/visualization, future server/web monitoring | `implemented` | Base LiDAR-derived steering signal debug output; final FOLLOW_ROW command additionally applies bed-centerline-primary correction and bounded LiDAR guard weighting |
| `/debug/heading` | `visualization_msgs/msg/Marker` | `rover_navigation` | `rover_navigation_node` | RViz2, future monitoring backend | `base_link` | `10 Hz` default | keep-last depth `10`, reliable | debug/visualization, future server/web monitoring | `implemented` | Rover heading vector marker for current steering direction |
| `/debug/row_confidence` | `visualization_msgs/msg/Marker` | `rover_navigation` | `rover_navigation_node` | RViz2, future monitoring backend | `base_link` | `10 Hz` default | keep-last depth `10`, reliable | debug/visualization, future server/web monitoring | `implemented` | Color-coded navigation confidence/state marker (green/yellow/red) for row alignment quality |
| `/debug/nav_state` | `visualization_msgs/msg/Marker` | `rover_navigation` | `rover_navigation_node` | RViz2, future monitoring backend | `base_link` | `10 Hz` default | keep-last depth `10`, reliable | debug/visualization, future server/web monitoring | `implemented` | Explicit FSM state text for autonomous 5-bed cycle (`FOLLOW_ROW`, `END_OF_ROW`, `TURN_TO_NEXT_ROW`, `ALIGN_NEXT_ROW`, `FINISHED`) |
| `/perception/row_centerline` | message type TBD | planned perception or navigation package | planned node | navigation controller and RViz2 debug | frame TBD | TBD | likely reliable, keep-last | planned perception output | `planned` | Next semantic signal after first fusion stage for corridor/row following |

## Contract rules for all agents

1. Do not rename or repurpose `implemented` topics without updating this file and dependent packages.
2. Prefer standard ROS message types unless roadmap demands custom interfaces.
3. Any new debug topic should be designed so it can be consumed by RViz2 now and a server/web bridge later.
4. If QoS or frame assumptions change, update this contract in the same change set.
5. Keep planned topics conservative; only promote to `implemented` when code publishes them.
