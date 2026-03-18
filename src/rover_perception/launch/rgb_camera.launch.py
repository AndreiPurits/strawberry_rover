import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("rover_perception")
    config_file = os.path.join(package_share, "config", "rgb_camera.yaml")

    rgb_camera = Node(
        package="rover_perception",
        executable="rgb_camera_node",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription([rgb_camera])
