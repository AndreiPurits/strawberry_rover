import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("rover_fake_stereo")
    config = os.path.join(share, "config", "fake_stereo.yaml")
    node = Node(
        package="rover_fake_stereo",
        executable="fake_stereo_node",
        name="fake_stereo_node",
        output="screen",
        parameters=[config],
    )
    return LaunchDescription([node])
