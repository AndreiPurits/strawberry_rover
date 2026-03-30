from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    roarm_ip_arg = DeclareLaunchArgument(
        "roarm_ip",
        default_value="192.168.4.1",
        description="RoArm-M3 IP address for HTTP JSON API.",
    )
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="transport",
        description="Self-test mode: status, transport, or axes.",
    )

    roarm_ip = LaunchConfiguration("roarm_ip")
    mode = LaunchConfiguration("mode")

    selftest_node = Node(
        package="roarm_ros2_http",
        executable="roarm_selftest",
        name="roarm_selftest",
        output="screen",
        parameters=[
            {
                "roarm_ip": roarm_ip,
                "mode": mode,
            }
        ],
    )

    return LaunchDescription(
        [
            roarm_ip_arg,
            mode_arg,
            selftest_node,
        ]
    )

