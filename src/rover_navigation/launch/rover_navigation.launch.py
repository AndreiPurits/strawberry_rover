import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("rover_navigation")
    config_file = os.path.join(package_share, "config", "rover_navigation.yaml")

    nav = Node(
        package="rover_navigation",
        executable="rover_navigation_node",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription([nav])
