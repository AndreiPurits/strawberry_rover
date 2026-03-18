import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from std_msgs.msg import String
from visualization_msgs.msg import Marker


class RoverNavigationNode(Node):
    """Deterministic row follower using smoothed left-right LiDAR probes."""

    def __init__(self) -> None:
        super().__init__("rover_navigation_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("debug_steering_topic", "/debug/steering")
        self.declare_parameter("debug_centerline_topic", "/debug/centerline")
        self.declare_parameter("debug_heading_topic", "/debug/heading")
        self.declare_parameter("debug_row_confidence_topic", "/debug/row_confidence")
        self.declare_parameter("debug_nav_state_topic", "/debug/nav_state")
        self.declare_parameter("sim_pose_topic", "/sim/rover_pose")
        self.declare_parameter("debug_frame_id", "base_link")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("forward_speed", 0.35)
        self.declare_parameter("slow_speed", 0.12)
        self.declare_parameter("steer_gain", 1.2)
        self.declare_parameter("max_angular_speed", 0.8)
        self.declare_parameter("max_steer_delta", 0.08)
        self.declare_parameter("steer_smoothing_alpha", 0.35)
        self.declare_parameter("center_smoothing_alpha", 0.25)
        self.declare_parameter("front_smoothing_alpha", 0.40)
        self.declare_parameter("confidence_smoothing_alpha", 0.30)
        self.declare_parameter("front_stop_distance", 0.45)
        self.declare_parameter("blocked_turn_speed", 0.35)
        self.declare_parameter("left_probe_angle_deg", 65.0)
        self.declare_parameter("right_probe_angle_deg", -65.0)
        self.declare_parameter("side_window_deg", 18.0)
        self.declare_parameter("centerline_length", 2.0)
        self.declare_parameter("row_count", 5)
        self.declare_parameter("row_spacing", 1.8)
        self.declare_parameter("row_length", 22.0)
        self.declare_parameter("row_start_side_extension", 0.0)
        self.declare_parameter("end_row_margin", 0.8)
        self.declare_parameter("end_row_front_distance", 0.75)
        self.declare_parameter("end_hold_time", 0.35)
        self.declare_parameter("transition_turn_speed", 0.55)
        self.declare_parameter("transition_turn_gain", 1.6)
        self.declare_parameter("align_speed", 0.20)
        self.declare_parameter("align_xy_tolerance", 0.18)
        self.declare_parameter("align_heading_tolerance_deg", 10.0)
        self.declare_parameter("transition_backoff", 0.8)
        self.declare_parameter("bed_center_gain", 0.45)
        self.declare_parameter("follow_row_lidar_steer_weight", 0.15)
        self.declare_parameter("low_conf_lidar_steer_weight", 0.05)
        self.declare_parameter("follow_row_lidar_steer_max", 0.30)
        self.declare_parameter("bed_center_deadband", 0.10)
        self.declare_parameter("recovery_trigger_distance", 0.75)
        self.declare_parameter("recovery_keep_current_distance", 0.45)
        self.declare_parameter("recovery_speed", 0.14)
        self.declare_parameter("recovery_target_x_margin", 0.8)
        self.declare_parameter("web_control_started_topic", "/web/control/started")
        self.declare_parameter("web_control_mode_topic", "/web/control/mode")

        self._scan_topic = str(self.get_parameter("scan_topic").value)
        self._cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self._debug_topic = str(self.get_parameter("debug_steering_topic").value)
        self._debug_centerline_topic = str(
            self.get_parameter("debug_centerline_topic").value
        )
        self._debug_heading_topic = str(self.get_parameter("debug_heading_topic").value)
        self._debug_row_conf_topic = str(
            self.get_parameter("debug_row_confidence_topic").value
        )
        self._debug_nav_state_topic = str(
            self.get_parameter("debug_nav_state_topic").value
        )
        self._sim_pose_topic = str(self.get_parameter("sim_pose_topic").value)
        self._debug_frame_id = str(self.get_parameter("debug_frame_id").value)

        self._publish_rate = float(self.get_parameter("publish_rate").value)
        if self._publish_rate <= 0.0:
            self._publish_rate = 10.0

        self._forward_speed = float(self.get_parameter("forward_speed").value)
        self._slow_speed = max(0.05, float(self.get_parameter("slow_speed").value))
        self._steer_gain = float(self.get_parameter("steer_gain").value)
        self._max_angular = max(0.1, float(self.get_parameter("max_angular_speed").value))
        self._max_steer_delta = max(
            0.01, float(self.get_parameter("max_steer_delta").value)
        )
        self._steer_alpha = self._clamp(
            float(self.get_parameter("steer_smoothing_alpha").value), 0.01, 1.0
        )
        self._center_alpha = self._clamp(
            float(self.get_parameter("center_smoothing_alpha").value), 0.01, 1.0
        )
        self._front_alpha = self._clamp(
            float(self.get_parameter("front_smoothing_alpha").value), 0.01, 1.0
        )
        self._confidence_alpha = self._clamp(
            float(self.get_parameter("confidence_smoothing_alpha").value), 0.01, 1.0
        )
        self._front_stop_distance = max(
            0.1, float(self.get_parameter("front_stop_distance").value)
        )
        self._blocked_turn_speed = max(
            0.1, float(self.get_parameter("blocked_turn_speed").value)
        )
        self._left_probe_angle = math.radians(
            float(self.get_parameter("left_probe_angle_deg").value)
        )
        self._right_probe_angle = math.radians(
            float(self.get_parameter("right_probe_angle_deg").value)
        )
        self._side_window = math.radians(float(self.get_parameter("side_window_deg").value))
        self._centerline_length = max(
            0.5, float(self.get_parameter("centerline_length").value)
        )
        self._row_count = max(1, int(self.get_parameter("row_count").value))
        self._row_spacing = max(0.6, float(self.get_parameter("row_spacing").value))
        self._row_length = max(4.0, float(self.get_parameter("row_length").value))
        self._row_start_side_extension = max(
            0.0, float(self.get_parameter("row_start_side_extension").value)
        )
        self._half_row_length = 0.5 * self._row_length
        self._end_row_margin = max(0.2, float(self.get_parameter("end_row_margin").value))
        self._end_row_front_distance = max(
            0.25, float(self.get_parameter("end_row_front_distance").value)
        )
        self._end_hold_time = max(0.05, float(self.get_parameter("end_hold_time").value))
        self._transition_turn_speed = max(
            0.1, float(self.get_parameter("transition_turn_speed").value)
        )
        self._transition_turn_gain = max(
            0.1, float(self.get_parameter("transition_turn_gain").value)
        )
        self._align_speed = max(0.05, float(self.get_parameter("align_speed").value))
        self._align_xy_tolerance = max(
            0.05, float(self.get_parameter("align_xy_tolerance").value)
        )
        self._align_heading_tolerance = math.radians(
            float(self.get_parameter("align_heading_tolerance_deg").value)
        )
        self._transition_backoff = max(
            0.1, float(self.get_parameter("transition_backoff").value)
        )
        self._bed_center_gain = max(0.0, float(self.get_parameter("bed_center_gain").value))
        self._follow_row_lidar_steer_weight = self._clamp(
            float(self.get_parameter("follow_row_lidar_steer_weight").value), 0.0, 1.0
        )
        self._low_conf_lidar_steer_weight = self._clamp(
            float(self.get_parameter("low_conf_lidar_steer_weight").value), 0.0, 1.0
        )
        self._follow_row_lidar_steer_max = max(
            0.0, float(self.get_parameter("follow_row_lidar_steer_max").value)
        )
        self._bed_center_deadband = max(
            0.0, float(self.get_parameter("bed_center_deadband").value)
        )
        self._recovery_trigger_distance = max(
            0.15, float(self.get_parameter("recovery_trigger_distance").value)
        )
        self._recovery_keep_current_distance = max(
            0.05, float(self.get_parameter("recovery_keep_current_distance").value)
        )
        self._recovery_speed = max(
            0.05, float(self.get_parameter("recovery_speed").value)
        )
        self._recovery_target_x_margin = max(
            0.0, float(self.get_parameter("recovery_target_x_margin").value)
        )
        self._web_control_started_topic = str(
            self.get_parameter("web_control_started_topic").value
        )
        self._web_control_mode_topic = str(
            self.get_parameter("web_control_mode_topic").value
        )

        self._last_scan: Optional[LaserScan] = None
        self._has_pose = False
        self._sim_x = 0.0
        self._sim_y = 0.0
        self._sim_yaw = 0.0
        self._smoothed_error = 0.0
        self._smoothed_front = 2.0
        self._steering_cmd = 0.0
        self._row_confidence = 0.0
        self._nav_state = "FOLLOW_ROW"
        self._state_enter_time = self.get_clock().now()
        self._row_index = 0
        self._row_direction = 1
        self._next_row_index = 0
        self._align_phase = "CROSS_ROW"
        self._recover_row_index = 0
        self._row_initialized = False
        # If web control bridge is not running, autonomous navigation stays enabled.
        self._web_control_started = True
        self._web_control_mode = "auto"
        self._auto_cmd_enabled_last = True

        self.create_subscription(LaserScan, self._scan_topic, self._on_scan, 10)
        self.create_subscription(PoseStamped, self._sim_pose_topic, self._on_pose, 10)
        self.create_subscription(Bool, self._web_control_started_topic, self._on_web_started, 10)
        self.create_subscription(String, self._web_control_mode_topic, self._on_web_mode, 10)
        self._cmd_pub = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._debug_pub = self.create_publisher(Float32, self._debug_topic, 10)
        self._centerline_pub = self.create_publisher(
            Marker, self._debug_centerline_topic, 10
        )
        self._heading_pub = self.create_publisher(Marker, self._debug_heading_topic, 10)
        self._row_confidence_pub = self.create_publisher(
            Marker, self._debug_row_conf_topic, 10
        )
        self._nav_state_pub = self.create_publisher(
            Marker, self._debug_nav_state_topic, 10
        )
        self.create_timer(1.0 / self._publish_rate, self._on_timer)

        self.get_logger().info(
            "rover_navigation_node started. "
            f"scan={self._scan_topic}, cmd_vel={self._cmd_vel_topic}, "
            f"debug={self._debug_topic}, centerline={self._debug_centerline_topic}, "
            f"heading={self._debug_heading_topic}, row_conf={self._debug_row_conf_topic}, "
            f"state={self._debug_nav_state_topic}, pose={self._sim_pose_topic}"
        )

    def _on_scan(self, msg: LaserScan) -> None:
        self._last_scan = msg

    def _on_pose(self, msg: PoseStamped) -> None:
        self._has_pose = True
        self._sim_x = float(msg.pose.position.x)
        self._sim_y = float(msg.pose.position.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        self._sim_yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        if not self._row_initialized:
            self._row_index = self._nearest_row_index(self._sim_y)
            self._next_row_index = min(self._row_count - 1, self._row_index + 1)
            self._row_direction = -1 if abs(self._normalize_angle(self._sim_yaw - math.pi)) < 1.4 else 1
            self._row_initialized = True
            self._start_recovery_if_needed()

    def _on_web_started(self, msg: Bool) -> None:
        self._web_control_started = bool(msg.data)
        if not self._web_control_started:
            self._cmd_pub.publish(Twist())

    def _on_web_mode(self, msg: String) -> None:
        mode = str(msg.data).strip().lower()
        if mode in ("manual", "auto"):
            prev_mode = self._web_control_mode
            self._web_control_mode = mode
            if prev_mode == "auto" and mode != "auto":
                self._cmd_pub.publish(Twist())

    def _on_timer(self) -> None:
        if self._last_scan is None or not self._has_pose:
            return

        scan = self._last_scan
        left_dist, left_count = self._mean_distance(
            scan, self._left_probe_angle, self._side_window
        )
        right_dist, right_count = self._mean_distance(
            scan, self._right_probe_angle, self._side_window
        )
        front_dist, _ = self._mean_distance(scan, 0.0, math.radians(10.0))

        raw_error = right_dist - left_dist
        self._smoothed_error = self._ema(self._smoothed_error, raw_error, self._center_alpha)
        self._smoothed_front = self._ema(self._smoothed_front, front_dist, self._front_alpha)

        raw_steering = self._clamp(
            self._steer_gain * self._smoothed_error, -self._max_angular, self._max_angular
        )
        filtered_steering = self._ema(self._steering_cmd, raw_steering, self._steer_alpha)
        delta = self._clamp(
            filtered_steering - self._steering_cmd,
            -self._max_steer_delta,
            self._max_steer_delta,
        )
        self._steering_cmd = self._clamp(
            self._steering_cmd + delta, -self._max_angular, self._max_angular
        )
        self._row_confidence = self._ema(
            self._row_confidence,
            self._compute_confidence(scan, left_count, right_count, raw_error),
            self._confidence_alpha,
        )

        cmd = self._compute_fsm_command()
        auto_cmd_enabled = bool(
            self._web_control_started and self._web_control_mode == "auto"
        )
        if auto_cmd_enabled:
            self._cmd_pub.publish(cmd)
        elif self._auto_cmd_enabled_last:
            # Publish one zero command when leaving auto control, then stop
            # publishing so manual web commands are not overridden.
            self._cmd_pub.publish(Twist())
        self._auto_cmd_enabled_last = auto_cmd_enabled

        dbg = Float32()
        dbg.data = self._steering_cmd
        self._debug_pub.publish(dbg)
        self._centerline_pub.publish(
            self._build_centerline_marker(self._smoothed_error, self._smoothed_front)
        )
        self._heading_pub.publish(self._build_heading_marker(cmd.angular.z))
        self._row_confidence_pub.publish(self._build_row_confidence_marker())
        self._nav_state_pub.publish(self._build_nav_state_marker())

    @staticmethod
    def _clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(max_v, value))

    def _compute_fsm_command(self) -> Twist:
        cmd = Twist()
        if not self._web_control_started or self._web_control_mode != "auto":
            return cmd

        end_x = self._target_end_x(self._row_direction)

        if self._nav_state == "FOLLOW_ROW":
            if self._is_off_bed(self._row_index):
                self._start_recovery_if_needed()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                return cmd

            row_center_y = self._row_index * self._row_spacing
            y_error = row_center_y - self._sim_y
            if abs(y_error) <= self._bed_center_deadband:
                y_error = 0.0
            row_center_steer = self._clamp(
                self._bed_center_gain
                * y_error
                * (1.0 if self._row_direction > 0 else -1.0),
                -self._max_angular,
                self._max_angular,
            )
            # LiDAR is a secondary guard signal in FOLLOW_ROW.
            lidar_guard_steer = self._clamp(
                self._steering_cmd,
                -self._follow_row_lidar_steer_max,
                self._follow_row_lidar_steer_max,
            )
            lidar_weight = self._follow_row_lidar_steer_weight
            if self._row_confidence < 0.35:
                lidar_weight = self._low_conf_lidar_steer_weight
            steer_cmd = self._clamp(
                row_center_steer + (lidar_weight * lidar_guard_steer),
                -self._max_angular,
                self._max_angular,
            )
            if self._row_confidence < 0.35:
                cmd.linear.x = self._slow_speed
                cmd.angular.z = steer_cmd * 0.7
            else:
                cmd.linear.x = self._forward_speed
                cmd.angular.z = steer_cmd

            reached_end_pose = (
                (self._row_direction > 0 and self._sim_x >= end_x)
                or (self._row_direction < 0 and self._sim_x <= end_x)
            )
            reached_end_scan = self._smoothed_front < self._end_row_front_distance
            if reached_end_pose or reached_end_scan:
                if self._row_index >= self._row_count - 1:
                    self._set_state("FINISHED")
                else:
                    self._next_row_index = self._row_index + 1
                    self._align_phase = "CROSS_ROW"
                    self._set_state("END_OF_ROW")
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
            return cmd

        if self._nav_state == "RECOVER_ROW":
            row_center_y = self._recover_row_index * self._row_spacing
            row_start_x = -(
                self._half_row_length
                + self._row_start_side_extension
                - self._recovery_target_x_margin
            )
            row_end_x = self._half_row_length - self._recovery_target_x_margin
            target_x = self._clamp(self._sim_x, row_start_x, row_end_x)
            target_y = row_center_y
            dx = target_x - self._sim_x
            dy = target_y - self._sim_y
            dist = math.hypot(dx, dy)

            if self._align_phase == "RETURN_CENTER":
                desired_yaw = math.atan2(dy, dx) if dist > 1e-6 else self._sim_yaw
                yaw_err = self._normalize_angle(desired_yaw - self._sim_yaw)
                cmd.linear.x = self._recovery_speed if dist > self._align_xy_tolerance else 0.0
                cmd.angular.z = self._clamp(
                    1.2 * yaw_err, -self._transition_turn_speed, self._transition_turn_speed
                )
                if dist <= self._align_xy_tolerance:
                    self._align_phase = "ALIGN_HEADING"
                return cmd

            target_heading = 0.0 if self._row_direction > 0 else math.pi
            yaw_err = self._normalize_angle(target_heading - self._sim_yaw)
            cmd.linear.x = 0.0
            cmd.angular.z = self._clamp(
                self._transition_turn_gain * yaw_err,
                -self._transition_turn_speed,
                self._transition_turn_speed,
            )
            if abs(yaw_err) <= self._align_heading_tolerance:
                self._row_index = self._recover_row_index
                self._set_state("FOLLOW_ROW")
            return cmd

        if self._nav_state == "END_OF_ROW":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            elapsed = self._elapsed_in_state()
            if elapsed >= self._end_hold_time:
                self._set_state("TURN_TO_NEXT_ROW")
            return cmd

        if self._nav_state == "TURN_TO_NEXT_ROW":
            target_yaw = math.pi / 2.0
            yaw_err = self._normalize_angle(target_yaw - self._sim_yaw)
            cmd.linear.x = 0.0
            cmd.angular.z = self._clamp(
                self._transition_turn_gain * yaw_err,
                -self._transition_turn_speed,
                self._transition_turn_speed,
            )
            if abs(yaw_err) <= self._align_heading_tolerance:
                self._align_phase = "CROSS_ROW"
                self._set_state("ALIGN_NEXT_ROW")
            return cmd

        if self._nav_state == "ALIGN_NEXT_ROW":
            cross_x = self._target_end_x(self._row_direction)
            target_x = cross_x - (self._row_direction * self._transition_backoff)
            target_y = self._next_row_index * self._row_spacing

            if self._align_phase == "CROSS_ROW":
                dx = target_x - self._sim_x
                dy = target_y - self._sim_y
                dist = math.hypot(dx, dy)
                desired_yaw = math.atan2(dy, dx)
                yaw_err = self._normalize_angle(desired_yaw - self._sim_yaw)
                cmd.linear.x = self._align_speed if dist > self._align_xy_tolerance else 0.0
                cmd.angular.z = self._clamp(
                    1.2 * yaw_err, -self._transition_turn_speed, self._transition_turn_speed
                )
                if dist <= self._align_xy_tolerance:
                    self._align_phase = "ALIGN_HEADING"
                return cmd

            next_direction = -self._row_direction
            target_heading = 0.0 if next_direction > 0 else math.pi
            yaw_err = self._normalize_angle(target_heading - self._sim_yaw)
            cmd.linear.x = 0.0
            cmd.angular.z = self._clamp(
                self._transition_turn_gain * yaw_err,
                -self._transition_turn_speed,
                self._transition_turn_speed,
            )
            if abs(yaw_err) <= self._align_heading_tolerance:
                self._row_index = self._next_row_index
                self._row_direction = next_direction
                self._set_state("FOLLOW_ROW")
            return cmd

        # FINISHED (default fallback)
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

    def _mean_distance(
        self, scan: LaserScan, center: float, half_window: float
    ) -> Tuple[float, int]:
        total = 0.0
        count = 0
        angle = scan.angle_min
        lower = center - half_window
        upper = center + half_window

        for r in scan.ranges:
            if lower <= angle <= upper:
                if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                    total += r
                    count += 1
            angle += scan.angle_increment

        if count == 0:
            return scan.range_max, 0
        return total / float(count), count

    def _compute_confidence(
        self, scan: LaserScan, left_count: int, right_count: int, raw_error: float
    ) -> float:
        expected = max(1, int((2.0 * self._side_window) / max(1e-6, scan.angle_increment)))
        side_coverage = 0.5 * (
            min(1.0, left_count / float(expected)) + min(1.0, right_count / float(expected))
        )
        balance = 1.0 - min(1.0, abs(raw_error) / 2.0)
        return self._clamp(side_coverage * balance, 0.0, 1.0)

    def _build_centerline_marker(self, error: float, front_dist: float) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._debug_frame_id
        marker.ns = "navigation"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0

        yaw = self._clamp(-error * 0.7, -1.0, 1.0)
        marker.scale.x = self._centerline_length
        marker.scale.y = 0.07
        marker.scale.z = 0.07
        marker.color.a = 0.95
        if front_dist < self._front_stop_distance:
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.2
        else:
            marker.color.r = 0.2
            marker.color.g = 0.9
            marker.color.b = 0.2

        half = yaw * 0.5
        marker.pose.orientation.z = math.sin(half)
        marker.pose.orientation.w = math.cos(half)
        return marker

    def _build_heading_marker(self, steering: float) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._debug_frame_id
        marker.ns = "navigation"
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.35
        marker.scale.x = 1.2
        marker.scale.y = 0.06
        marker.scale.z = 0.06
        marker.color.r = 0.1
        marker.color.g = 0.6
        marker.color.b = 1.0
        marker.color.a = 0.95
        yaw = self._clamp(steering * 1.2, -1.0, 1.0)
        half = yaw * 0.5
        marker.pose.orientation.z = math.sin(half)
        marker.pose.orientation.w = math.cos(half)
        return marker

    def _build_row_confidence_marker(self) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._debug_frame_id
        marker.ns = "navigation"
        marker.id = 2
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.8
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.7
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.18
        marker.color.a = 1.0
        if self._row_confidence >= 0.7:
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.1
        elif self._row_confidence >= 0.4:
            marker.color.r = 1.0
            marker.color.g = 0.85
            marker.color.b = 0.1
        else:
            marker.color.r = 1.0
            marker.color.g = 0.2
            marker.color.b = 0.2
        marker.text = f"{self._nav_state} conf={self._row_confidence:.2f}"
        return marker

    def _build_nav_state_marker(self) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._debug_frame_id
        marker.ns = "navigation"
        marker.id = 3
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 1.1
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.95
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.16
        marker.color.r = 0.9
        marker.color.g = 0.9
        marker.color.b = 0.95
        marker.color.a = 1.0
        marker.text = (
            f"state={self._nav_state} row={self._row_index + 1}/{self._row_count} "
            f"dir={'+' if self._row_direction > 0 else '-'} "
            f"x={self._sim_x:.1f} y={self._sim_y:.1f}"
        )
        return marker

    def _set_state(self, state: str) -> None:
        if self._nav_state == state:
            return
        self._nav_state = state
        self._state_enter_time = self.get_clock().now()

    def _elapsed_in_state(self) -> float:
        now = self.get_clock().now()
        delta = now - self._state_enter_time
        return delta.nanoseconds / 1e9

    def _target_end_x(self, direction: int) -> float:
        if direction > 0:
            return self._half_row_length - self._end_row_margin
        neg_edge = self._half_row_length + self._row_start_side_extension
        return -(neg_edge - self._end_row_margin)

    def _nearest_row_index(self, y_value: float) -> int:
        idx = int(round(y_value / self._row_spacing))
        return max(0, min(self._row_count - 1, idx))

    def _is_off_bed(self, row_index: int) -> bool:
        center_y = row_index * self._row_spacing
        return abs(self._sim_y - center_y) > self._recovery_trigger_distance

    def _select_recovery_row(self) -> int:
        current_center = self._row_index * self._row_spacing
        if abs(self._sim_y - current_center) <= self._recovery_keep_current_distance:
            return self._row_index
        return self._nearest_row_index(self._sim_y)

    def _start_recovery_if_needed(self) -> None:
        target_row = self._select_recovery_row()
        center_y = target_row * self._row_spacing
        if abs(self._sim_y - center_y) <= self._align_xy_tolerance:
            self._row_index = target_row
            return
        self._recover_row_index = target_row
        self._align_phase = "RETURN_CENTER"
        self._set_state("RECOVER_ROW")

    @staticmethod
    def _ema(previous: float, current: float, alpha: float) -> float:
        return ((1.0 - alpha) * previous) + (alpha * current)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoverNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
