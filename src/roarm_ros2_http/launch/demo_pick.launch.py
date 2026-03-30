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
    api_path_arg = DeclareLaunchArgument(
        "api_path",
        default_value="/js",
        description="HTTP path for RoArm JSON endpoint.",
    )

    roarm_ip = LaunchConfiguration("roarm_ip")
    api_path = LaunchConfiguration("api_path")

    driver_node = Node(
        package="roarm_ros2_http",
        executable="roarm_http_driver",
        name="roarm_http_driver",
        output="screen",
        parameters=[
            {
                "roarm_ip": roarm_ip,
                "api_path": api_path,
            }
        ],
    )

    demo_node = Node(
        package="roarm_ros2_http",
        executable="demo_pick_sequence",
        name="demo_pick_sequence",
        output="screen",
        parameters=[
            {
                "roarm_ip": roarm_ip,
                "api_path": api_path,
            }
        ],
    )

    return LaunchDescription(
        [
            roarm_ip_arg,
            api_path_arg,
            driver_node,
            demo_node,
        ]
    )

