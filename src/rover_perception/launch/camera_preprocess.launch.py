import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("rover_perception")
    config_file = os.path.join(package_share, "config", "camera_preprocess.yaml")

    preprocess_node = Node(
        package="rover_perception",
        executable="camera_preprocess_node",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription([preprocess_node])
