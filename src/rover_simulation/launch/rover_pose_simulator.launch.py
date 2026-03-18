import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("rover_simulation")
    config_file = os.path.join(package_share, "config", "rover_pose_simulator.yaml")

    simulator = Node(
        package="rover_simulation",
        executable="rover_pose_simulator",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription([simulator])
