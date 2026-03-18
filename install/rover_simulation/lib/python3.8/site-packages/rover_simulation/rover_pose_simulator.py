import math
from typing import List, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class RoverPoseSimulator(Node):
    """Simulate 2D rover pose from /cmd_vel."""

    def __init__(self) -> None:
        super().__init__("rover_pose_simulator")

        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("pose_topic", "/sim/rover_pose")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("world_frame_id", "sim_world")
        self.declare_parameter("x0", 0.0)
        self.declare_parameter("y0", 0.0)
        self.declare_parameter("yaw0", 0.0)
        self.declare_parameter("motion_mode", "scripted_rows")
        self.declare_parameter("row_count", 5)
        self.declare_parameter("row_spacing", 1.8)
        self.declare_parameter("row_length", 20.0)
        self.declare_parameter("start_side_extension", 0.0)
        self.declare_parameter("scripted_speed", 0.35)
        self.declare_parameter("scripted_turn_speed", 0.8)
        self.declare_parameter("waypoint_tolerance", 0.2)
        self.declare_parameter("loop_rows", True)

        self._publish_rate = float(self.get_parameter("publish_rate").value)
        if self._publish_rate <= 0.0:
            self._publish_rate = 20.0

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._world_frame_id = str(self.get_parameter("world_frame_id").value)
        self._x = float(self.get_parameter("x0").value)
        self._y = float(self.get_parameter("y0").value)
        self._yaw = float(self.get_parameter("yaw0").value)
        self._v = 0.0
        self._w = 0.0
        self._last_time = self.get_clock().now()
        self._motion_mode = str(self.get_parameter("motion_mode").value).lower()

        self._row_count = max(1, int(self.get_parameter("row_count").value))
        self._row_spacing = max(0.6, float(self.get_parameter("row_spacing").value))
        self._row_length = max(2.0, float(self.get_parameter("row_length").value))
        self._start_side_extension = max(
            0.0, float(self.get_parameter("start_side_extension").value)
        )
        self._scripted_speed = max(0.05, float(self.get_parameter("scripted_speed").value))
        self._scripted_turn_speed = max(
            0.1, float(self.get_parameter("scripted_turn_speed").value)
        )
        self._waypoint_tol = max(
            0.05, float(self.get_parameter("waypoint_tolerance").value)
        )
        self._loop_rows = bool(self.get_parameter("loop_rows").value)
        self._waypoints: List[Tuple[float, float]] = []
        self._waypoint_idx = 0
        if self._motion_mode == "scripted_rows":
            self._waypoints = self._build_row_waypoints()

        pose_topic = str(self.get_parameter("pose_topic").value)
        cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)

        self._pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Twist, cmd_vel_topic, self._on_cmd_vel, 10)
        self.create_timer(1.0 / self._publish_rate, self._on_timer)

        self.get_logger().info(
            f"rover_pose_simulator started. Subscribing {cmd_vel_topic}, publishing {pose_topic} "
            f"at {self._publish_rate:.2f} Hz."
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._v = float(msg.linear.x)
        self._w = float(msg.angular.z)

    def _on_timer(self) -> None:
        if self._motion_mode == "scripted_rows":
            self._update_scripted_velocity()

        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds / 1e9
        self._last_time = now
        if dt <= 0.0:
            return

        self._x += self._v * math.cos(self._yaw) * dt
        self._y += self._v * math.sin(self._yaw) * dt
        self._yaw += self._w * dt

        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = self._x
        msg.pose.position.y = self._y
        msg.pose.position.z = 0.0

        half = self._yaw * 0.5
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = math.sin(half)
        msg.pose.orientation.w = math.cos(half)

        self._pose_pub.publish(msg)
        self._publish_tf(now)

    def _publish_tf(self, now) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = now.to_msg()
        tf_msg.header.frame_id = self._world_frame_id
        tf_msg.child_frame_id = self._frame_id
        tf_msg.transform.translation.x = self._x
        tf_msg.transform.translation.y = self._y
        tf_msg.transform.translation.z = 0.0

        half = self._yaw * 0.5
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = math.sin(half)
        tf_msg.transform.rotation.w = math.cos(half)
        self._tf_broadcaster.sendTransform(tf_msg)

    def _build_row_waypoints(self) -> List[Tuple[float, float]]:
        waypoints: List[Tuple[float, float]] = []
        x_start = -(self._row_length / 2.0) - self._start_side_extension
        x_end = self._row_length / 2.0
        # Deterministic endless boustrophedon cycle:
        # row0 -> row1 -> ... -> rowN -> rowN-1 -> ... -> row0 -> repeat.
        row_visit_order = list(range(self._row_count))
        if self._row_count > 1:
            row_visit_order += list(range(self._row_count - 2, -1, -1))

        direction = 1
        for idx, row_idx in enumerate(row_visit_order):
            y = row_idx * self._row_spacing
            x_target = x_end if direction > 0 else x_start
            # Traverse along current bed centerline.
            waypoints.append((x_target, y))

            # Shift to neighboring bed at the row end.
            if idx < len(row_visit_order) - 1:
                next_row = row_visit_order[idx + 1]
                next_y = next_row * self._row_spacing
                waypoints.append((x_target, next_y))
                direction *= -1
        return waypoints

    def _update_scripted_velocity(self) -> None:
        if not self._waypoints:
            self._v = 0.0
            self._w = 0.0
            return

        target_x, target_y = self._waypoints[self._waypoint_idx]
        dx = target_x - self._x
        dy = target_y - self._y
        dist = math.hypot(dx, dy)
        if dist < self._waypoint_tol:
            self._waypoint_idx += 1
            if self._waypoint_idx >= len(self._waypoints):
                if self._loop_rows:
                    self._waypoint_idx = 0
                else:
                    self._waypoint_idx = len(self._waypoints) - 1
                    self._v = 0.0
                    self._w = 0.0
                    return
            target_x, target_y = self._waypoints[self._waypoint_idx]
            dx = target_x - self._x
            dy = target_y - self._y
            dist = math.hypot(dx, dy)

        target_yaw = math.atan2(dy, dx)
        yaw_err = self._normalize_angle(target_yaw - self._yaw)
        if abs(yaw_err) > 0.25:
            self._v = 0.0
            self._w = max(-self._scripted_turn_speed, min(self._scripted_turn_speed, yaw_err))
        else:
            self._v = self._scripted_speed
            self._w = max(
                -self._scripted_turn_speed, min(self._scripted_turn_speed, yaw_err * 1.5)
            )

    @staticmethod
    def _normalize_angle(value: float) -> float:
        while value > math.pi:
            value -= 2.0 * math.pi
        while value < -math.pi:
            value += 2.0 * math.pi
        return value


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoverPoseSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
