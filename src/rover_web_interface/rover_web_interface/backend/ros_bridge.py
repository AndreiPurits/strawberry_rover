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
from rclpy.qos import qos_profile_sensor_data
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
        self.declare_parameter("lidar_arc_sectors", 21)
        self.declare_parameter("lidar_arc_fov_deg", 160.0)
        # UI forward direction offset (degrees in LaserScan frame).
        # 180 means lidar is mounted backwards relative to rover forward.
        self.declare_parameter("lidar_forward_angle_deg", 180.0)
        # RPLidar scan order vs rover-left: flip sector index for UI + guard.
        self.declare_parameter("lidar_mirror_lr", True)
        self.declare_parameter("real_front_camera_topic", "/camera/image_raw")
        self.declare_parameter("real_stereo_camera_topic", "/stereo_camera/image_raw")
        self.declare_parameter("control_publish_period_s", 0.05)
        self.declare_parameter("route_record_min_dt_s", 0.15)
        self.declare_parameter("route_record_min_dist_m", 0.05)

        self._row_spacing = float(self.get_parameter("row_spacing").value)
        self._trail_max_points = int(self.get_parameter("trail_max_points").value)
        self._scan_downsample = max(1, int(self.get_parameter("scan_downsample").value))
        self._lidar_arc_sectors = max(5, int(self.get_parameter("lidar_arc_sectors").value))
        self._lidar_arc_fov_rad = math.radians(
            float(self.get_parameter("lidar_arc_fov_deg").value)
        )
        self._lidar_forward_rad = math.radians(
            float(self.get_parameter("lidar_forward_angle_deg").value)
        )
        self._lidar_mirror_lr = bool(self.get_parameter("lidar_mirror_lr").value)
        self._real_front_camera_topic = str(
            self.get_parameter("real_front_camera_topic").value
        ).strip()
        self._real_stereo_camera_topic = str(
            self.get_parameter("real_stereo_camera_topic").value
        ).strip()
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
        self._lidar_arc: Dict[str, Any] = {
            "connected": False,
            "sectors": [],
            "sector_count": self._lidar_arc_sectors,
            "fov_deg": math.degrees(self._lidar_arc_fov_rad),
            "stamp": 0.0,
            "min_dist_m": None,
        }
        self._images: Dict[str, Any] = {
            "front": {},
            "bottom": {},
            "stereo": {},
        }
        self._front_camera_jpeg: Dict[str, Any] = {}
        self._front_camera_jpeg_hub: Dict[str, Any] = {}
        self._stereo_camera_jpeg: Dict[str, Any] = {}
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
        self._arduino_status: str = '{"connected":false}'
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
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        if self._real_front_camera_topic:
            self.create_subscription(
                Image,
                self._real_front_camera_topic,
                lambda msg: self._on_image("front", msg),
                qos_profile_sensor_data,
            )
        if self._real_stereo_camera_topic:
            self.create_subscription(
                Image,
                self._real_stereo_camera_topic,
                lambda msg: self._on_image("stereo", msg),
                qos_profile_sensor_data,
            )
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
        self.create_subscription(String, "/rover/arduino/status", self._on_arduino_status, 10)
        self.create_timer(self._control_publish_period_s, self._control_publish_tick)
        self.create_timer(0.5, self._publish_control_state)
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

    def _lidar_level(self, dist_m: float, range_max: float) -> int:
        if dist_m >= range_max * 0.98:
            return 0
        if dist_m < 0.4:
            return 3
        if dist_m < 0.8:
            return 2
        if dist_m < 1.5:
            return 1
        return 0

    def _compute_lidar_arc(self, msg: LaserScan) -> Dict[str, Any]:
        sectors_n = self._lidar_arc_sectors
        fov_half = self._lidar_arc_fov_rad * 0.5
        mins = [float(msg.range_max)] * sectors_n
        angle = float(msg.angle_min)
        inc = float(msg.angle_increment)
        range_max = float(msg.range_max)
        range_min = float(msg.range_min)

        for rng in msg.ranges:
            if math.isfinite(rng) and range_min <= float(rng) <= range_max:
                rel_angle = self._normalize_angle(angle - self._lidar_forward_rad)
                if -fov_half <= rel_angle <= fov_half:
                    rel = rel_angle + fov_half
                    idx = int(rel / max(1e-6, self._lidar_arc_fov_rad) * sectors_n)
                    idx = max(0, min(sectors_n - 1, idx))
                    if self._lidar_mirror_lr:
                        idx = sectors_n - 1 - idx
                    mins[idx] = min(mins[idx], float(rng))
            angle += inc

        sectors: List[Dict[str, Any]] = []
        global_min = range_max
        for i, dist in enumerate(mins):
            level = self._lidar_level(dist, range_max)
            if dist < global_min:
                global_min = dist
            sectors.append(
                {
                    "i": i,
                    "dist_m": None if dist >= range_max * 0.98 else round(dist, 2),
                    "level": level,
                }
            )

        return {
            "connected": True,
            "sectors": sectors,
            "sector_count": sectors_n,
            "fov_deg": round(math.degrees(self._lidar_arc_fov_rad), 1),
            "range_max_m": round(range_max, 2),
            "display_max_m": round(min(4.0, range_max), 2),
            "stamp": time.time(),
            "min_dist_m": None if global_min >= range_max * 0.98 else round(global_min, 2),
        }

    def _on_scan(self, msg: LaserScan) -> None:
        points: List[Dict[str, float]] = []
        angle = float(msg.angle_min)
        for idx, rng in enumerate(msg.ranges):
            if idx % self._scan_downsample == 0 and math.isfinite(rng):
                points.append({"a": angle, "r": float(rng)})
            angle += float(msg.angle_increment)
        arc = self._compute_lidar_arc(msg)
        with self._lock:
            self._scan_summary = {
                "angle_min": float(msg.angle_min),
                "angle_max": float(msg.angle_max),
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
                "points": points,
            }
            self._lidar_arc = arc

    def _crop_center_aspect(self, arr: Any, aspect: float) -> Any:
        import numpy as np

        src_h, src_w = arr.shape[:2]
        if src_w <= 0 or src_h <= 0 or aspect <= 0:
            return arr
        current = src_w / src_h
        if current > aspect:
            new_w = max(1, int(round(src_h * aspect)))
            x0 = max(0, (src_w - new_w) // 2)
            return arr[:, x0 : x0 + new_w]
        if current < aspect:
            new_h = max(1, int(round(src_w / aspect)))
            y0 = max(0, (src_h - new_h) // 2)
            return arr[y0 : y0 + new_h, :]
        return arr

    def _encode_jpeg_bgr(self, arr: Any, width: int, height: int, quality: int) -> Dict[str, Any]:
        import cv2

        return self._encode_jpeg_bgr_fit(arr, width, height, quality, letterbox=True)

    def _encode_jpeg_bgr_fit(
        self,
        arr: Any,
        max_width: int,
        max_height: int,
        quality: int,
        *,
        letterbox: bool = True,
    ) -> Dict[str, Any]:
        import cv2
        import numpy as np

        src_h, src_w = arr.shape[:2]
        if src_w <= 0 or src_h <= 0:
            return {"ok": False, "reason": "bad_geometry"}

        scale = min(max_width / src_w, max_height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if letterbox and (new_w != max_width or new_h != max_height):
            canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            x0 = (max_width - new_w) // 2
            y0 = (max_height - new_h) // 2
            canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
            out_img = canvas
        else:
            out_img = resized

        ok, jpg = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return {"ok": False, "reason": "jpeg_encode_failed"}
        return {
            "ok": True,
            "jpeg_b64": base64.b64encode(jpg.tobytes()).decode("ascii"),
            "width": int(out_img.shape[1]),
            "height": int(out_img.shape[0]),
        }

    def _make_jpeg_from_image(self, msg: Image, *, hub: bool = False) -> Dict[str, Any]:
        try:
            import cv2
            import numpy as np

            height = int(msg.height)
            width = int(msg.width)
            encoding = str(msg.encoding or "bgr8")
            if height <= 0 or width <= 0:
                return {"ok": False, "reason": "bad_geometry"}
            raw = bytes(msg.data)
            if encoding in ("bgr8", "rgb8"):
                channels = 3
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, channels))
                if encoding == "rgb8":
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif encoding in ("mono8", "8UC1"):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                return {"ok": False, "reason": f"unsupported_encoding:{encoding}"}
            if hub:
                # Keep full camera FOV for hub stream; avoid center-crop.
                out = self._encode_jpeg_bgr_fit(arr, 800, 600, 38, letterbox=True)
            else:
                out = self._encode_jpeg_bgr(arr, 640, 480, 45)
            if not out.get("ok"):
                return out
            out["stamp"] = _ros_time_to_sec(msg.header.stamp)
            return out
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    def _on_image(self, camera_name: str, msg: Image) -> None:
        if camera_name in ("front", "stereo"):
            jpeg = self._make_jpeg_from_image(msg, hub=False)
            if jpeg.get("ok"):
                with self._lock:
                    if camera_name == "front":
                        self._front_camera_jpeg = jpeg
                        hub_jpeg = self._make_jpeg_from_image(msg, hub=True)
                        if hub_jpeg.get("ok"):
                            self._front_camera_jpeg_hub = hub_jpeg
                    else:
                        self._stereo_camera_jpeg = jpeg
            return

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

    def _on_arduino_status(self, msg: String) -> None:
        with self._lock:
            self._arduino_status = str(msg.data)

    def get_arduino_snapshot(self) -> str:
        with self._lock:
            return str(self._arduino_status)

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
            arduino_status = str(self._arduino_status)

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
            "arduino": arduino_status,
        }

    def get_scan_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._scan_summary)

    def get_lidar_arc_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._lidar_arc)

    def get_stereo_camera_jpeg(self) -> Dict[str, Any]:
        with self._lock:
            cached = dict(self._stereo_camera_jpeg)
        if cached.get("ok") and cached.get("jpeg_b64"):
            return cached

        with self._lock:
            img = dict(self._images.get("stereo") or {})
        if not img.get("data_b64"):
            return {"ok": False, "reason": "no_frame"}
        try:
            import cv2
            import numpy as np

            raw = base64.b64decode(img["data_b64"])
            height = int(img.get("height") or 0)
            width = int(img.get("width") or 0)
            encoding = str(img.get("encoding") or "bgr8")
            if height <= 0 or width <= 0:
                return {"ok": False, "reason": "bad_geometry"}
            if encoding in ("bgr8", "rgb8"):
                channels = 3
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, channels))
                if encoding == "rgb8":
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif encoding in ("mono8", "8UC1"):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                return {"ok": False, "reason": f"unsupported_encoding:{encoding}"}
            small = cv2.resize(arr, (640, 360), interpolation=cv2.INTER_AREA)
            small = cv2.convertScaleAbs(small, alpha=1.28, beta=18)
            ok, jpg = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 62])
            if not ok:
                return {"ok": False, "reason": "jpeg_encode_failed"}
            result = {
                "ok": True,
                "jpeg_b64": base64.b64encode(jpg.tobytes()).decode("ascii"),
                "width": 640,
                "height": 360,
                "stamp": float(img.get("stamp") or time.time()),
            }
            with self._lock:
                self._stereo_camera_jpeg = result
            return result
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

    def get_front_camera_jpeg(self, hub: bool = False) -> Dict[str, Any]:
        with self._lock:
            cached = dict(self._front_camera_jpeg_hub if hub else self._front_camera_jpeg)
        if cached.get("ok") and cached.get("jpeg_b64"):
            return cached

        with self._lock:
            img = dict(self._images.get("front") or {})
        if not img.get("data_b64"):
            return {"ok": False, "reason": "no_frame"}
        try:
            import cv2
            import numpy as np

            raw = base64.b64decode(img["data_b64"])
            height = int(img.get("height") or 0)
            width = int(img.get("width") or 0)
            encoding = str(img.get("encoding") or "bgr8")
            if height <= 0 or width <= 0:
                return {"ok": False, "reason": "bad_geometry"}
            if encoding in ("bgr8", "rgb8"):
                channels = 3
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, channels))
                if encoding == "rgb8":
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif encoding in ("mono8", "8UC1"):
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                return {"ok": False, "reason": f"unsupported_encoding:{encoding}"}
            out = (
                self._encode_jpeg_bgr_fit(arr, 800, 600, 36, letterbox=True)
                if hub
                else self._encode_jpeg_bgr_fit(arr, 640, 480, 45, letterbox=False)
            )
            if not out.get("ok"):
                return out
            result = {
                **out,
                "stamp": float(img.get("stamp") or time.time()),
            }
            with self._lock:
                if hub:
                    self._front_camera_jpeg_hub = result
                else:
                    self._front_camera_jpeg = result
            return result
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

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

    def apply_track_command(
        self,
        left: float,
        right: float,
        source: str = "tracks",
    ) -> Dict[str, Any]:
        """Tank tracks -1..1 -> cmd_vel (opposite tracks = turn in place)."""
        l = max(-1.0, min(1.0, float(left)))
        r = max(-1.0, min(1.0, float(right)))
        linear_x = (l + r) * 0.5
        angular_z = (r - l) * 0.5
        return self.apply_manual_command(linear_x, angular_z, source=source)

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
