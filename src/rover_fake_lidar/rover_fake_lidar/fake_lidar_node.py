import math
import random
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class FakeLidarNode(Node):
    """ROS 2 node that publishes fake LiDAR data on /scan."""

    def __init__(self) -> None:
        super().__init__("FakeLidarNode")

        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("angle_min", -3.14159)
        self.declare_parameter("angle_max", 3.14159)
        self.declare_parameter("angle_increment", 0.0174533)
        self.declare_parameter("range_min", 0.1)
        self.declare_parameter("range_max", 10.0)
        self.declare_parameter("scene_type", "field_rows")
        self.declare_parameter("scene_frame_id", "sim_world")
        self.declare_parameter("scene_markers_topic", "/sim/scene_markers")
        self.declare_parameter("rover_pose_topic", "/sim/rover_pose")
        self.declare_parameter("field_length", 20.0)
        self.declare_parameter("start_side_extension", 0.0)
        self.declare_parameter("row_count", 5)
        self.declare_parameter("row_spacing", 1.8)
        self.declare_parameter("bed_width", 0.8)
        self.declare_parameter("bed_height", 0.2)
        self.declare_parameter("rover_length", 1.3)
        self.declare_parameter("rover_width", 1.2)
        self.declare_parameter("rover_height", 1.2)
        self.declare_parameter("include_explicit_obstacles", False)

        publish_rate = float(self.get_parameter("publish_rate").value)
        if publish_rate <= 0.0:
            publish_rate = 10.0

        self._scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        markers_topic = str(self.get_parameter("scene_markers_topic").value)
        rover_pose_topic = str(self.get_parameter("rover_pose_topic").value)
        self._scene_pub = self.create_publisher(MarkerArray, markers_topic, 10)
        self.create_subscription(PoseStamped, rover_pose_topic, self._on_rover_pose, 10)
        self.create_timer(1.0 / publish_rate, self._publish)

        self._rover_x = 0.0
        self._rover_y = 0.0
        self._rover_yaw = 0.0
        self._clear_sent = False

        self.get_logger().info(
            f"FakeLidarNode started. /scan + {markers_topic}, rover pose from {rover_pose_topic}."
        )

    def _on_rover_pose(self, msg: PoseStamped) -> None:
        self._rover_x = float(msg.pose.position.x)
        self._rover_y = float(msg.pose.position.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        self._rover_yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)

    def _publish(self) -> None:
        angle_min = float(self.get_parameter("angle_min").value)
        angle_max = float(self.get_parameter("angle_max").value)
        angle_increment = float(self.get_parameter("angle_increment").value)
        range_min = float(self.get_parameter("range_min").value)
        range_max = float(self.get_parameter("range_max").value)
        scene_type = str(self.get_parameter("scene_type").value).lower()

        if angle_increment <= 0.0:
            return

        count = int((angle_max - angle_min) / angle_increment) + 1
        if count <= 0:
            return

        ranges = self._generate_ranges(
            angle_min=angle_min,
            angle_increment=angle_increment,
            range_min=range_min,
            range_max=range_max,
            scene_type=scene_type,
            count=count,
        )

        publish_rate = float(self.get_parameter("publish_rate").value)
        if publish_rate <= 0.0:
            publish_rate = 10.0

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_link"
        msg.angle_min = angle_min
        msg.angle_max = angle_max
        msg.angle_increment = angle_increment
        msg.time_increment = (1.0 / publish_rate) / float(count)
        msg.scan_time = 1.0 / publish_rate
        msg.range_min = range_min
        msg.range_max = range_max
        msg.ranges = ranges
        msg.intensities = []
        self._scan_pub.publish(msg)
        self._scene_pub.publish(self._build_scene_markers())

    def _generate_ranges(
        self,
        *,
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        scene_type: str,
        count: int,
    ) -> List[float]:
        out: List[float] = []
        for i in range(count):
            a = angle_min + i * angle_increment
            if scene_type == "field_rows":
                v = self._raycast_posts_and_obstacles(a, range_max)
            elif scene_type == "noisy":
                base = range_max * 0.9
                v = base + random.uniform(-0.05 * range_max, 0.05 * range_max)
            else:
                v = range_max * 0.95
            out.append(max(range_min, min(range_max, v)))
        return out

    def _raycast_posts_and_obstacles(self, local_angle: float, range_max: float) -> float:
        gx = self._rover_x
        gy = self._rover_y
        ga = self._rover_yaw + local_angle
        dx = math.cos(ga)
        dy = math.sin(ga)

        circles: List[Tuple[float, float, float]] = []
        if bool(self.get_parameter("include_explicit_obstacles").value):
            circles += self._build_explicit_obstacles()
        best: Optional[float] = None
        for cx, cy, radius in circles:
            hit = self._ray_circle_intersection(gx, gy, dx, dy, cx, cy, radius)
            if hit is None:
                continue
            if best is None or hit < best:
                best = hit

        return range_max if best is None else min(range_max, best)

    def _build_explicit_obstacles(self) -> List[Tuple[float, float, float]]:
        # Small explicit obstacles (e.g. crates/boxes) to exercise avoidance logic.
        return [
            (3.0, 1.8, 0.18),
            (-4.0, 5.4, 0.18),
        ]

    @staticmethod
    def _ray_circle_intersection(
        ox: float, oy: float, dx: float, dy: float, cx: float, cy: float, r: float
    ) -> Optional[float]:
        fx = ox - cx
        fy = oy - cy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        s = math.sqrt(disc)
        t1 = (-b - s) / 2.0
        t2 = (-b + s) / 2.0
        hits = [t for t in (t1, t2) if t >= 0.0]
        if not hits:
            return None
        return min(hits)

    def _build_scene_markers(self) -> MarkerArray:
        frame_id = str(self.get_parameter("scene_frame_id").value)
        field_len = max(4.0, float(self.get_parameter("field_length").value))
        start_side_extension = max(
            0.0, float(self.get_parameter("start_side_extension").value)
        )
        row_count = max(1, int(self.get_parameter("row_count").value))
        row_spacing = max(0.6, float(self.get_parameter("row_spacing").value))
        bed_width = max(0.2, float(self.get_parameter("bed_width").value))
        bed_height = max(0.05, float(self.get_parameter("bed_height").value))
        rover_length = max(0.1, float(self.get_parameter("rover_length").value))
        rover_width = max(0.1, float(self.get_parameter("rover_width").value))
        rover_height = max(0.1, float(self.get_parameter("rover_height").value))

        stamp = self.get_clock().now().to_msg()
        markers: List[Marker] = []
        marker_id = 0

        if not self._clear_sent:
            clear = Marker()
            clear.header.frame_id = frame_id
            clear.header.stamp = stamp
            clear.action = Marker.DELETEALL
            markers.append(clear)
            self._clear_sent = True

        x_min = -(field_len / 2.0) - start_side_extension
        x_max = field_len / 2.0
        bed_len = x_max - x_min
        bed_center_x = 0.5 * (x_min + x_max)

        # Five row/bed areas.
        for r in range(row_count):
            mk = Marker()
            mk.header.frame_id = frame_id
            mk.header.stamp = stamp
            mk.ns = "beds"
            mk.id = r
            mk.type = Marker.CUBE
            mk.action = Marker.ADD
            mk.pose.position.x = bed_center_x
            mk.pose.position.y = r * row_spacing
            mk.pose.position.z = bed_height / 2.0
            mk.pose.orientation.w = 1.0
            mk.scale.x = bed_len
            mk.scale.y = bed_width
            mk.scale.z = bed_height
            mk.color.r = 0.20
            mk.color.g = 0.60
            mk.color.b = 0.20
            mk.color.a = 0.6
            markers.append(mk)

        if bool(self.get_parameter("include_explicit_obstacles").value):
            for idx, (ox, oy, rad) in enumerate(self._build_explicit_obstacles()):
                obs = Marker()
                obs.header.frame_id = frame_id
                obs.header.stamp = stamp
                obs.ns = "obstacles"
                obs.id = idx
                obs.type = Marker.CYLINDER
                obs.action = Marker.ADD
                obs.pose.position.x = ox
                obs.pose.position.y = oy
                obs.pose.position.z = 0.15
                obs.pose.orientation.w = 1.0
                obs.scale.x = 2.0 * rad
                obs.scale.y = 2.0 * rad
                obs.scale.z = 0.3
                obs.color.r = 0.9
                obs.color.g = 0.2
                obs.color.b = 0.2
                obs.color.a = 0.9
                markers.append(obs)

        # Rover body marker.
        rover = Marker()
        rover.header.frame_id = frame_id
        rover.header.stamp = stamp
        rover.ns = "rover"
        rover.id = 0
        rover.type = Marker.CUBE
        rover.action = Marker.ADD
        rover.pose.position.x = self._rover_x
        rover.pose.position.y = self._rover_y
        rover.pose.position.z = rover_height / 2.0
        rover.pose.orientation.z = math.sin(self._rover_yaw / 2.0)
        rover.pose.orientation.w = math.cos(self._rover_yaw / 2.0)
        rover.scale.x = rover_length
        rover.scale.y = rover_width
        rover.scale.z = rover_height
        rover.color.r = 0.2
        rover.color.g = 0.43
        rover.color.b = 0.8
        rover.color.a = 0.9
        markers.append(rover)

        arr = MarkerArray()
        arr.markers = markers
        return arr


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

