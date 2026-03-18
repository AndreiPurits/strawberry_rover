import math
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class SensorFusionNode(Node):
    """Simple RGB + LiDAR fusion node for obstacle/candidate confidence."""

    def __init__(self) -> None:
        super().__init__("sensor_fusion_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("edges_topic", "/camera/preprocessed/edges")
        self.declare_parameter("plant_candidates_topic", "/perception/plant_candidates")
        self.declare_parameter("fusion_score_topic", "/perception/fusion_score")
        self.declare_parameter("fusion_rate", 10.0)
        self.declare_parameter("fov_half_angle_deg", 30.0)
        self.declare_parameter("max_scan_for_scoring", 5.0)
        self.declare_parameter("edges_threshold", 0.08)
        self.declare_parameter("marker_frame_id", "lidar_link")

        scan_topic = str(self.get_parameter("scan_topic").value)
        edges_topic = str(self.get_parameter("edges_topic").value)
        candidates_topic = str(self.get_parameter("plant_candidates_topic").value)
        score_topic = str(self.get_parameter("fusion_score_topic").value)

        self._fusion_rate = float(self.get_parameter("fusion_rate").value)
        if self._fusion_rate <= 0.0:
            self._fusion_rate = 10.0

        self._half_angle_rad = math.radians(
            float(self.get_parameter("fov_half_angle_deg").value)
        )
        self._max_scan = max(0.5, float(self.get_parameter("max_scan_for_scoring").value))
        self._edges_threshold = max(
            0.0, min(1.0, float(self.get_parameter("edges_threshold").value))
        )
        self._marker_frame = str(self.get_parameter("marker_frame_id").value)

        self._last_scan: Optional[LaserScan] = None
        self._last_edges_density = 0.0

        self.create_subscription(
            LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, edges_topic, self._on_edges, qos_profile_sensor_data
        )

        self._score_pub = self.create_publisher(Float32, score_topic, 10)
        self._candidates_pub = self.create_publisher(MarkerArray, candidates_topic, 10)
        self._timer = self.create_timer(1.0 / self._fusion_rate, self._on_timer)

        self.get_logger().info(
            "sensor_fusion_node started. "
            f"scan={scan_topic}, edges={edges_topic}, score={score_topic}, candidates={candidates_topic}"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        self._last_scan = msg

    def _on_edges(self, msg: Image) -> None:
        if msg.encoding != "mono8":
            self.get_logger().warn(
                f"Unsupported edges encoding '{msg.encoding}', expected 'mono8'."
            )
            return

        frame = np.frombuffer(msg.data, dtype=np.uint8)
        expected = int(msg.height) * int(msg.width)
        if frame.size != expected:
            self.get_logger().warn("Edges image has unexpected buffer size.")
            return

        image = frame.reshape((int(msg.height), int(msg.width)))
        # Use central ROI density as a simple proxy for structured scene content.
        h, w = image.shape
        x0 = int(w * 0.3)
        x1 = int(w * 0.7)
        y0 = int(h * 0.3)
        y1 = int(h * 0.8)
        roi = image[y0:y1, x0:x1]
        non_zero = cv2.countNonZero(roi)
        total = max(1, roi.shape[0] * roi.shape[1])
        self._last_edges_density = float(non_zero) / float(total)

    def _on_timer(self) -> None:
        if self._last_scan is None:
            return

        scan_score = self._compute_scan_score(self._last_scan)
        edge_score = min(1.0, self._last_edges_density / max(0.01, self._edges_threshold))
        fusion_score = float((0.6 * scan_score) + (0.4 * edge_score))

        score_msg = Float32()
        score_msg.data = fusion_score
        self._score_pub.publish(score_msg)

        self._candidates_pub.publish(self._build_candidates_marker(fusion_score, scan_score))

    def _compute_scan_score(self, scan: LaserScan) -> float:
        closest = self._max_scan
        angle = scan.angle_min
        for r in scan.ranges:
            valid = np.isfinite(r) and scan.range_min <= r <= scan.range_max
            if valid and -self._half_angle_rad <= angle <= self._half_angle_rad:
                if r < closest:
                    closest = r
            angle += scan.angle_increment

        normalized = 1.0 - min(1.0, closest / self._max_scan)
        return max(0.0, min(1.0, normalized))

    def _build_candidates_marker(self, fusion_score: float, scan_score: float) -> MarkerArray:
        stamp = self.get_clock().now().to_msg()
        distance = max(0.4, self._max_scan * (1.0 - scan_score))

        marker = Marker()
        marker.header.frame_id = self._marker_frame
        marker.header.stamp = stamp
        marker.ns = "fusion"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(distance)
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = float(min(1.0, fusion_score))
        marker.color.g = float(max(0.0, 1.0 - fusion_score))
        marker.color.b = 0.15
        marker.color.a = 0.85

        label = Marker()
        label.header.frame_id = self._marker_frame
        label.header.stamp = stamp
        label.ns = "fusion"
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(distance)
        label.pose.position.y = 0.0
        label.pose.position.z = 0.5
        label.pose.orientation.w = 1.0
        label.scale.z = 0.18
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = f"fusion_score={fusion_score:.2f}"

        msg = MarkerArray()
        msg.markers = [marker, label]
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
