"""Publish fake strawberry targets to `/roarm/target_pose` for MVP testing."""

from __future__ import annotations

from typing import List, Tuple

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node


class FakeTargetPublisherNode(Node):
    """Publishes a short list of virtual strawberry coordinates."""

    def __init__(self) -> None:
        super().__init__("fake_target_publisher")
        self._publisher = self.create_publisher(PoseStamped, "/roarm/target_pose", 10)

        self.declare_parameter("frame_id", "roarm_base")
        self.declare_parameter("publish_period_sec", 2.0)
        self._frame_id = str(self.get_parameter("frame_id").value)
        period_sec = float(self.get_parameter("publish_period_sec").value)

        self._targets: List[Tuple[float, float, float]] = [
            (0.18, 0.04, 0.14),
            (0.16, -0.05, 0.12),
            (0.20, 0.00, 0.16),
            (0.15, 0.06, 0.11),
            (0.19, -0.03, 0.13),
        ]
        self._index = 0

        self._timer = self.create_timer(period_sec, self._on_timer)
        self.get_logger().info("Fake target publisher started for /roarm/target_pose")

    def _on_timer(self) -> None:
        if self._index >= len(self._targets):
            self.get_logger().info("All fake targets published. Stopping timer.")
            self._timer.cancel()
            return

        x, y, z = self._targets[self._index]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0

        self._publisher.publish(msg)
        self.get_logger().info(
            f"Published fake strawberry target #{self._index + 1}: "
            f"({x:.3f}, {y:.3f}, {z:.3f})"
        )
        self._index += 1


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FakeTargetPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

