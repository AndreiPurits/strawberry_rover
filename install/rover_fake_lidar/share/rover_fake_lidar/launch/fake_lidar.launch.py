import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("rover_fake_lidar")
    config_file = os.path.join(package_share, "config", "fake_lidar.yaml")

    fake_lidar = Node(
        package="rover_fake_lidar",
        executable="fake_lidar_node",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription([fake_lidar])

