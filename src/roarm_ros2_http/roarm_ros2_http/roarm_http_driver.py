"""ROS2 node: subscribe PoseStamped and forward safe targets to RoArm HTTP API.

Build:
  colcon build --packages-select roarm_ros2_http
Run:
  ros2 launch roarm_ros2_http demo_pick.launch.py roarm_ip:=192.168.4.1
Manual target test:
  ros2 topic pub --once /roarm/target_pose geometry_msgs/msg/PoseStamped \
    "{header: {frame_id: 'roarm_base'}, pose: {position: {x: 0.16, y: 0.00, z: 0.12}, orientation: {w: 1.0}}}"
"""

from __future__ import annotations

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node

from .roarm_client import RoArmClient, RoArmClientError, RoArmHttpConfig
from .safety import WorkspaceLimits, sanitize_target


class RoArmHttpDriverNode(Node):
    """Translate `/roarm/target_pose` into HTTP JSON move commands."""

    def __init__(self) -> None:
        super().__init__("roarm_http_driver")

        self.declare_parameter("roarm_ip", "192.168.4.1")
        self.declare_parameter("api_path", "/js")
        self.declare_parameter("http_timeout_sec", 8.0)

        self.declare_parameter("default_t", 0.0)
        self.declare_parameter("default_r", 0.0)
        self.declare_parameter("default_g", 3.14)
        self.declare_parameter("default_spd", 0.25)
        self.declare_parameter("default_acc", 0)

        self.declare_parameter("x_min", 0.05)
        self.declare_parameter("x_max", 0.35)
        self.declare_parameter("y_min", -0.20)
        self.declare_parameter("y_max", 0.20)
        self.declare_parameter("z_min", 0.02)
        self.declare_parameter("z_max", 0.30)
        self.declare_parameter("min_planar_radius", 0.08)
        self.declare_parameter("max_planar_radius", 0.32)

        config = RoArmHttpConfig(
            roarm_ip=self.get_parameter("roarm_ip").value,
            api_path=self.get_parameter("api_path").value,
            timeout_sec=float(self.get_parameter("http_timeout_sec").value),
        )
        self._client = RoArmClient(config)
        self._limits = WorkspaceLimits(
            x_min=float(self.get_parameter("x_min").value),
            x_max=float(self.get_parameter("x_max").value),
            y_min=float(self.get_parameter("y_min").value),
            y_max=float(self.get_parameter("y_max").value),
            z_min=float(self.get_parameter("z_min").value),
            z_max=float(self.get_parameter("z_max").value),
            min_planar_radius=float(self.get_parameter("min_planar_radius").value),
            max_planar_radius=float(self.get_parameter("max_planar_radius").value),
        )

        self._default_t = float(self.get_parameter("default_t").value)
        self._default_r = float(self.get_parameter("default_r").value)
        self._default_g = float(self.get_parameter("default_g").value)
        self._default_spd = float(self.get_parameter("default_spd").value)
        self._default_acc = int(self.get_parameter("default_acc").value)

        self.create_subscription(
            PoseStamped,
            "/roarm/target_pose",
            self._on_target_pose,
            10,
        )
        self.get_logger().info(
            f"RoArm HTTP driver started. endpoint={config.endpoint}, topic=/roarm/target_pose"
        )

    def _on_target_pose(self, msg: PoseStamped) -> None:
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)

        safe_target, reason = sanitize_target(
            x=x,
            y=y,
            z=z,
            limits=self._limits,
            reject_if_unsafe=True,
        )
        if safe_target is None:
            self.get_logger().warning(
                f"Rejected unsafe target ({x:.3f}, {y:.3f}, {z:.3f}): {reason}"
            )
            return

        sx, sy, sz = safe_target
        try:
            response = self._client.move_xyz(
                x=sx,
                y=sy,
                z=sz,
                t=self._default_t,
                r=self._default_r,
                g=self._default_g,
                spd=self._default_spd,
                acc=self._default_acc,
            )
            self.get_logger().info(
                f"move_xyz success for ({sx:.3f}, {sy:.3f}, {sz:.3f}) -> {response}"
            )
        except RoArmClientError as exc:
            self.get_logger().error(f"move_xyz failed for ({sx:.3f}, {sy:.3f}, {sz:.3f}): {exc}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RoArmHttpDriverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

