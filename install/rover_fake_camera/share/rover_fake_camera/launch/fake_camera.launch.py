import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("rover_fake_camera")
    config = os.path.join(share, "config", "fake_camera.yaml")
    node = Node(
        package="rover_fake_camera",
        executable="fake_camera_node",
        name="fake_camera_node",
        output="screen",
        parameters=[config],
    )
    return LaunchDescription([node])
