import base64
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def _ros_time_to_sec(stamp) -> float:
    return float(stamp.sec) + (float(stamp.nanosec) * 1e-9)


class RosWebBridge(Node):
    """ROS2 -> web bridge state cache for FastAPI layer."""

    def __init__(self) -> None:
        super().__init__("rover_web_bridge")

        self.declare_parameter("row_spacing", 1.8)
        self.declare_parameter("trail_max_points", 300)
        self.declare_parameter("scan_downsample", 8)
        self.declare_parameter("control_publish_period_s", 0.05)
        self.declare_parameter("route_record_min_dt_s", 0.15)
        self.declare_parameter("route_record_min_dist_m", 0.05)

        self._row_spacing = float(self.get_parameter("row_spacing").value)
        self._trail_max_points = int(self.get_parameter("trail_max_points").value)
        self._scan_downsample = max(1, int(self.get_parameter("scan_downsample").value))
        self._control_publish_period_s = float(
            self.get_parameter("control_publish_period_s").value
        )
        self._route_record_min_dt_s = float(
            self.get_parameter("route_record_min_dt_s").value
        )
        self._route_record_min_dist_m = float(
            self.get_parameter("route_record_min_dist_m").value
        )

        self._lock = threading.Lock()
        self._trail: Deque[Tuple[float, float]] = deque(maxlen=self._trail_max_points)
        self._beds: List[Dict[str, float]] = []
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._control_started_pub = self.create_publisher(Bool, "/web/control/started", 10)
        self._control_mode_pub = self.create_publisher(String, "/web/control/mode", 10)

        self._pose = {"x": 0.0, "y": 0.0, "z": 0.0}
        self._heading_rad = 0.0
        self._speed_mps = 0.0
        self._angular_velocity = 0.0
        self._current_row = 0
        self._last_pose_time: Optional[float] = None
        self._last_pose_xy: Optional[Tuple[float, float]] = None
        self._last_heading: Optional[float] = None

        self._nav_state = "unknown"
        self._scan_summary: Dict[str, Any] = {
            "angle_min": 0.0,
            "angle_max": 0.0,
            "range_min": 0.0,
            "range_max": 0.0,
            "points": [],
        }
        self._images: Dict[str, Dict[str, Any]] = {
            "front": {},
            "bottom": {},
            "stereo": {},
        }
        self._control_started = False
        self._control_mode = "auto"
        self._last_cmd = {
            "linear_x": 0.0,
            "angular_z": 0.0,
            "source": "none",
            "stamp": 0.0,
        }
        self._pending_cmd = {"linear_x": 0.0, "angular_z": 0.0}
        self._force_zero_once = False
        self._route_recording = False
        self._current_route: Dict[str, Any] = {
            "id": "draft",
            "name": "draft_route",
            "created_at": 0.0,
            "saved_at": None,
            "metadata": {},
            "points": [],
        }
        self._saved_routes: List[Dict[str, Any]] = []
        self._active_route_id: Optional[str] = None
        self._route_counter = 0
        self._route_last_sample_time = 0.0
        self._route_last_sample_xy: Optional[Tuple[float, float]] = None
        self._current_route = self._new_route_draft(time.time())

        self.create_subscription(PoseStamped, "/sim/rover_pose", self._on_pose, 10)
        self.create_subscription(MarkerArray, "/sim/scene_markers", self._on_scene, 10)
        self.create_subscription(Marker, "/debug/nav_state", self._on_nav_state, 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.create_subscription(
            Image,
            "/sim/camera/front/image_raw",
            lambda msg: self._on_image("front", msg),
            10,
        )
        self.create_subscription(
            Image,
            "/sim/camera/bottom_rgb/image_raw",
            lambda msg: self._on_image("bottom", msg),
            10,
        )
        self.create_subscription(
            Image,
            "/sim/stereo/debug/combined",
            lambda msg: self._on_image("stereo", msg),
            10,
        )
        self.create_timer(self._control_publish_period_s, self._control_publish_tick)
        self._publish_control_state()

        self.get_logger().info("rover_web_bridge started.")

    def _on_pose(self, msg: PoseStamped) -> None:
        now = _ros_time_to_sec(msg.header.stamp)
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        heading = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

        speed = 0.0
        w = 0.0
        if self._last_pose_time is not None and self._last_pose_xy is not None:
            dt = now - self._last_pose_time
            if dt > 1e-4:
                dx = x - self._last_pose_xy[0]
                dy = y - self._last_pose_xy[1]
                speed = math.hypot(dx, dy) / dt
                if self._last_heading is not None:
                    d_heading = self._normalize_angle(heading - self._last_heading)
                    w = d_heading / dt

        with self._lock:
            self._pose = {"x": x, "y": y, "z": z}
            self._heading_rad = heading
            self._speed_mps = speed
            self._angular_velocity = w
            self._current_row = int(round(y / max(1e-6, self._row_spacing))) + 1
            self._trail.append((x, y))
            self._last_pose_time = now
            self._last_pose_xy = (x, y)
            self._last_heading = heading

            if self._route_recording and self._should_append_route_point(now, x, y):
                self._append_route_point(now, x, y, heading)

    def _on_scene(self, msg: MarkerArray) -> None:
        beds: List[Dict[str, float]] = []
        for marker in msg.markers:
            if marker.ns != "beds":
                continue
            beds.append(
                {
                    "x": float(marker.pose.position.x),
                    "y": float(marker.pose.position.y),
                    "length": float(marker.scale.x),
                    "width": float(marker.scale.y),
                }
            )
        with self._lock:
            if beds:
                self._beds = beds

    def _on_nav_state(self, msg: Marker) -> None:
        with self._lock:
            self._nav_state = msg.text if msg.text else "unknown"

    def _on_scan(self, msg: LaserScan) -> None:
        points: List[Dict[str, float]] = []
        angle = float(msg.angle_min)
        for idx, rng in enumerate(msg.ranges):
            if idx % self._scan_downsample == 0 and math.isfinite(rng):
                points.append({"a": angle, "r": float(rng)})
            angle += float(msg.angle_increment)
        with self._lock:
            self._scan_summary = {
                "angle_min": float(msg.angle_min),
                "angle_max": float(msg.angle_max),
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
                "points": points,
            }

    def _on_image(self, camera_name: str, msg: Image) -> None:
        # Keep bridge simple and format-agnostic for MVP:
        # forward raw frame bytes as base64 and render in browser JS.
        encoded = base64.b64encode(bytes(msg.data)).decode("ascii")
        with self._lock:
            self._images[camera_name] = {
                "stamp": _ros_time_to_sec(msg.header.stamp),
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": str(msg.encoding),
                "step": int(msg.step),
                "data_b64": encoded,
            }

    def get_state_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            pose = dict(self._pose)
            heading = float(self._heading_rad)
            speed = float(self._speed_mps)
            w = float(self._angular_velocity)
            trail = [{"x": p[0], "y": p[1]} for p in self._trail]
            beds = list(self._beds)
            nav_state = self._nav_state
            current_row = int(self._current_row)
            control = self._build_control_snapshot()
            routes = self._build_route_snapshot()

        now = time.time()
        sensors = self._mock_sensor_grid(now)
        analytics = self._mock_analytics(now)
        telemetry = self._build_telemetry(now, speed, current_row)

        return {
            "timestamp": now,
            "field": {
                "beds": beds,
                "sensor_grid": sensors,
            },
            "rover": {
                "pose": pose,
                "heading_rad": heading,
                "heading_deg": math.degrees(heading),
                "route_trail": trail,
                "nav_state": nav_state,
            },
            "telemetry": {
                **telemetry,
                "speed_mps": speed,
                "angular_velocity_rps": w,
                "current_row": current_row,
            },
            "analytics": analytics,
            "control": control,
            "routes": routes,
        }

    def get_scan_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._scan_summary)

    def get_camera_snapshot(self, camera_name: str) -> Dict[str, Any]:
        if camera_name not in self._images:
            return {}
        with self._lock:
            return dict(self._images[camera_name])

    def _build_control_snapshot(self) -> Dict[str, Any]:
        return {
            "mode": self._control_mode,
            "started": bool(self._control_started),
            "manual_allowed": bool(self._control_started and self._control_mode == "manual"),
            "last_command": dict(self._last_cmd),
        }

    def _build_route_snapshot(self) -> Dict[str, Any]:
        current = {
            "id": self._current_route["id"],
            "name": self._current_route["name"],
            "created_at": self._current_route["created_at"],
            "saved_at": self._current_route["saved_at"],
            "metadata": dict(self._current_route["metadata"]),
            "points": list(self._current_route["points"]),
            "point_count": len(self._current_route["points"]),
        }
        saved = []
        for route in self._saved_routes:
            saved.append(
                {
                    "id": route["id"],
                    "name": route["name"],
                    "created_at": route["created_at"],
                    "saved_at": route["saved_at"],
                    "metadata": dict(route["metadata"]),
                    "points": list(route["points"]),
                    "point_count": len(route["points"]),
                }
            )
        active = None
        for route in saved:
            if route["id"] == self._active_route_id:
                active = route
                break
        return {
            "recording": bool(self._route_recording),
            "current_route": current,
            "saved_routes": saved,
            "active_route_id": self._active_route_id,
            "active_route": active,
        }

    def get_route_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._build_route_snapshot()

    def _route_metadata_template(self) -> Dict[str, Any]:
        return {
            "bed_length_m": 22.0,
            "row_spacing_m": self._row_spacing,
            "snake_structure": True,
            "notes": "",
            "row_count": 5,
            "spacing_m": self._row_spacing,
            "rows": [],
        }

    def _new_route_draft(self, now_s: float) -> Dict[str, Any]:
        self._route_counter += 1
        route_id = f"route_{self._route_counter:03d}"
        return {
            "id": route_id,
            "name": route_id,
            "created_at": now_s,
            "saved_at": None,
            "metadata": self._route_metadata_template(),
            "points": [],
        }

    def _should_append_route_point(self, now_s: float, x: float, y: float) -> bool:
        if not self._current_route["points"]:
            return True
        dt = now_s - self._route_last_sample_time
        if dt < self._route_record_min_dt_s:
            return False
        if self._route_last_sample_xy is None:
            return True
        dist = math.hypot(x - self._route_last_sample_xy[0], y - self._route_last_sample_xy[1])
        return dist >= self._route_record_min_dist_m

    def _append_route_point(self, now_s: float, x: float, y: float, heading: float) -> None:
        point = {
            "x": float(x),
            "y": float(y),
            "yaw": float(heading),
            "timestamp": float(now_s),
            "row_index": int(self._current_row),
        }
        self._current_route["points"].append(point)
        self._route_last_sample_time = now_s
        self._route_last_sample_xy = (x, y)

    def start_route_recording(self) -> Dict[str, Any]:
        with self._lock:
            if not self._route_recording:
                now_s = time.time()
                if self._current_route["saved_at"] is not None:
                    self._current_route = self._new_route_draft(now_s)
                elif not self._current_route["points"]:
                    self._current_route["created_at"] = now_s
                    self._current_route["metadata"] = self._route_metadata_template()
                self._route_recording = True
                self._route_last_sample_time = 0.0
                self._route_last_sample_xy = None
            return self._build_route_snapshot()

    def stop_route_recording(self) -> Dict[str, Any]:
        with self._lock:
            self._route_recording = False
            return self._build_route_snapshot()

    def save_current_route(self) -> Dict[str, Any]:
        with self._lock:
            self._route_recording = False
            if not self._current_route["points"]:
                return self._build_route_snapshot()
            route = {
                "id": self._current_route["id"],
                "name": self._current_route["name"],
                "created_at": self._current_route["created_at"],
                "saved_at": time.time(),
                "metadata": dict(self._current_route["metadata"]),
                "points": list(self._current_route["points"]),
            }
            self._saved_routes.append(route)
            self._active_route_id = route["id"]
            self._current_route = self._new_route_draft(time.time())
            self._route_last_sample_time = 0.0
            self._route_last_sample_xy = None
            return self._build_route_snapshot()

    @staticmethod
    def _copy_route(route: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": route["id"],
            "name": route["name"],
            "created_at": route["created_at"],
            "saved_at": route["saved_at"],
            "metadata": dict(route["metadata"]),
            "points": list(route["points"]),
        }

    @staticmethod
    def _normalize_metadata(existing: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(existing)
        if "notes" in patch:
            meta["notes"] = str(patch.get("notes", ""))
        if "row_count" in patch:
            try:
                meta["row_count"] = max(0, int(patch.get("row_count", 0)))
            except (TypeError, ValueError):
                pass
        if "spacing_m" in patch:
            try:
                meta["spacing_m"] = float(patch.get("spacing_m", meta.get("spacing_m", 0.0)))
            except (TypeError, ValueError):
                pass
        if "rows" in patch and isinstance(patch["rows"], list):
            meta["rows"] = list(patch["rows"])
        return meta

    def _find_saved_route_index(self, route_id: str) -> int:
        for idx, route in enumerate(self._saved_routes):
            if route["id"] == route_id:
                return idx
        return -1

    def select_route(self, route_id: str) -> Dict[str, Any]:
        with self._lock:
            idx = self._find_saved_route_index(route_id)
            if idx < 0:
                raise ValueError("route_not_found")
            self._active_route_id = self._saved_routes[idx]["id"]
            return self._build_route_snapshot()

    def rename_route(self, route_id: str, new_name: str) -> Dict[str, Any]:
        clean_name = str(new_name).strip()
        if not clean_name:
            raise ValueError("route_name_empty")
        with self._lock:
            idx = self._find_saved_route_index(route_id)
            if idx < 0:
                raise ValueError("route_not_found")
            self._saved_routes[idx]["name"] = clean_name
            return self._build_route_snapshot()

    def delete_route(self, route_id: str) -> Dict[str, Any]:
        with self._lock:
            idx = self._find_saved_route_index(route_id)
            if idx < 0:
                raise ValueError("route_not_found")
            del self._saved_routes[idx]
            if self._active_route_id == route_id:
                self._active_route_id = self._saved_routes[-1]["id"] if self._saved_routes else None
            return self._build_route_snapshot()

    def update_route_metadata(self, route_id: str, metadata_patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            idx = self._find_saved_route_index(route_id)
            if idx < 0:
                raise ValueError("route_not_found")
            route = self._saved_routes[idx]
            route["metadata"] = self._normalize_metadata(route["metadata"], metadata_patch)
            return self._build_route_snapshot()

    def add_route_row_metadata(self, route_id: str, row_meta: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            idx = self._find_saved_route_index(route_id)
            if idx < 0:
                raise ValueError("route_not_found")
            route = self._saved_routes[idx]
            metadata = dict(route["metadata"])
            rows = list(metadata.get("rows", []))
            rows.append(
                {
                    "row_id": str(row_meta.get("row_id", f"row_{len(rows) + 1}")),
                    "label": str(row_meta.get("label", f"Row {len(rows) + 1}")),
                    "length_m": float(row_meta.get("length_m", metadata.get("bed_length_m", 22.0))),
                }
            )
            metadata["rows"] = rows
            metadata["row_count"] = max(int(metadata.get("row_count", 0)), len(rows))
            route["metadata"] = metadata
            return self._build_route_snapshot()

    def remove_route_row_metadata(self, route_id: str, row_index: int) -> Dict[str, Any]:
        with self._lock:
            idx = self._find_saved_route_index(route_id)
            if idx < 0:
                raise ValueError("route_not_found")
            route = self._saved_routes[idx]
            metadata = dict(route["metadata"])
            rows = list(metadata.get("rows", []))
            if row_index < 0 or row_index >= len(rows):
                raise ValueError("row_index_out_of_range")
            del rows[row_index]
            metadata["rows"] = rows
            metadata["row_count"] = max(0, int(metadata.get("row_count", 0)))
            route["metadata"] = metadata
            return self._build_route_snapshot()

    def trim_route_last_points(self, route_id: str, points_to_trim: int = 20) -> Dict[str, Any]:
        trim_n = max(1, int(points_to_trim))
        with self._lock:
            idx = self._find_saved_route_index(route_id)
            if idx < 0:
                raise ValueError("route_not_found")
            route = self._saved_routes[idx]
            pts = list(route["points"])
            if trim_n >= len(pts):
                pts = []
            else:
                pts = pts[:-trim_n]
            route["points"] = pts
            return self._build_route_snapshot()

    def get_control_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._build_control_snapshot()

    def _control_publish_tick(self) -> None:
        with self._lock:
            started = bool(self._control_started)
            mode = str(self._control_mode)
            zero_once = bool(self._force_zero_once)
            linear_x = float(self._pending_cmd["linear_x"])
            angular_z = float(self._pending_cmd["angular_z"])
            if zero_once:
                self._force_zero_once = False

        self._publish_control_state()

        if zero_once:
            twist = Twist()
            self._cmd_pub.publish(twist)
            return

        if started and mode == "manual":
            twist = Twist()
            twist.linear.x = linear_x
            twist.angular.z = angular_z
            self._cmd_pub.publish(twist)

    def start_control(self) -> Dict[str, Any]:
        with self._lock:
            self._control_started = True
            self._last_cmd["stamp"] = time.time()
        self._publish_control_state()
        with self._lock:
            return self._build_control_snapshot()

    def stop_control(self) -> Dict[str, Any]:
        with self._lock:
            self._control_started = False
        self._publish_control_state()
        self.publish_zero_cmd(source="stop")
        with self._lock:
            return self._build_control_snapshot()

    def set_control_mode(self, mode: str) -> Dict[str, Any]:
        cleaned = str(mode).strip().lower()
        if cleaned not in ("manual", "auto"):
            raise ValueError("mode must be 'manual' or 'auto'")
        with self._lock:
            self._control_mode = cleaned
        self._publish_control_state()
        if cleaned != "manual":
            # Safety rule: leaving manual mode immediately zeroes manual motion.
            self.publish_zero_cmd(source="mode_auto")
        with self._lock:
            return self._build_control_snapshot()

    def apply_manual_command(
        self,
        linear_x: float,
        angular_z: float,
        source: str = "web",
    ) -> Dict[str, Any]:
        with self._lock:
            manual_allowed = self._control_started and self._control_mode == "manual"
        if not manual_allowed:
            return self.get_control_snapshot()

        with self._lock:
            self._pending_cmd = {
                "linear_x": float(linear_x),
                "angular_z": float(angular_z),
            }
            self._last_cmd = {
                "linear_x": float(linear_x),
                "angular_z": float(angular_z),
                "source": str(source),
                "stamp": time.time(),
            }
            return self._build_control_snapshot()

    def publish_zero_cmd(self, source: str = "zero") -> Dict[str, Any]:
        # Publish an immediate zero command, then keep one extra zero on the next
        # control tick to override any in-flight non-zero command safely.
        self._cmd_pub.publish(Twist())
        with self._lock:
            self._pending_cmd = {"linear_x": 0.0, "angular_z": 0.0}
            self._force_zero_once = True
            self._last_cmd = {
                "linear_x": 0.0,
                "angular_z": 0.0,
                "source": str(source),
                "stamp": time.time(),
            }
            return self._build_control_snapshot()

    def _publish_control_state(self) -> None:
        with self._lock:
            started = bool(self._control_started)
            mode = str(self._control_mode)
        started_msg = Bool()
        started_msg.data = started
        mode_msg = String()
        mode_msg.data = mode
        self._control_started_pub.publish(started_msg)
        self._control_mode_pub.publish(mode_msg)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _mock_sensor_grid(now_s: float) -> List[Dict[str, float]]:
        grid: List[Dict[str, float]] = []
        idx = 0
        for row in range(5):
            for col in range(4):
                phase = (row * 0.7) + (col * 0.4)
                grid.append(
                    {
                        "id": idx,
                        "x": -8.0 + (col * 5.0),
                        "y": row * 1.8,
                        "soil_moisture": 48.0 + 8.0 * math.sin(now_s * 0.05 + phase),
                        "air_humidity": 62.0 + 6.0 * math.cos(now_s * 0.03 + phase),
                        "soil_temp": 21.0 + 2.0 * math.sin(now_s * 0.02 + phase),
                        "air_temp": 25.0 + 2.5 * math.cos(now_s * 0.015 + phase),
                        "daily_illumination": 72.0 + 10.0 * math.sin(now_s * 0.01 + phase),
                    }
                )
                idx += 1
        return grid

    @staticmethod
    def _mock_analytics(now_s: float) -> Dict[str, float]:
        return {
            "berries_collected_today": 320.0 + 40.0 * math.sin(now_s * 0.004),
            "working_time_hours": 5.2 + 0.3 * math.sin(now_s * 0.002),
            "energy_consumption_kwh": 3.8 + 0.2 * math.cos(now_s * 0.003),
            "avg_harvest_speed_berries_per_hour": 58.0 + 5.0 * math.sin(now_s * 0.005),
            "productivity_index": 0.82 + 0.06 * math.sin(now_s * 0.006),
        }

    @staticmethod
    def _build_telemetry(now_s: float, speed: float, current_row: int) -> Dict[str, Any]:
        return {
            "battery_level_pct": 81.0 + 4.0 * math.cos(now_s * 0.003),
            "avg_energy_consumption_wh": 690.0 + 25.0 * math.sin(now_s * 0.004),
            "battery_temperature_c": 34.0 + 1.8 * math.sin(now_s * 0.008),
            "berries_collected": 95.0 + 12.0 * math.sin(now_s * 0.005),
            "collection_speed_per_hour": 52.0 + 6.0 * math.cos(now_s * 0.004),
            "berry_density_estimate": 0.65 + 0.08 * math.sin(now_s * 0.007),
            "rear_lidar_status": "mock",
            "current_row": current_row,
            "speed_mps": speed,
            "arm_cameras": [
                {"id": i + 1, "label": f"arm_cam_{i + 1}", "status": "mock"}
                for i in range(6)
            ],
        }


def create_bridge_node() -> RosWebBridge:
    if not rclpy.ok():
        rclpy.init(args=None)
    return RosWebBridge()
