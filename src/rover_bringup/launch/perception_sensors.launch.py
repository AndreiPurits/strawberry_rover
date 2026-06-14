"""Front RGB camera + RPLidar only (no sim / navigation)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    perception_share = get_package_share_directory("rover_perception")
    rplidar_share = get_package_share_directory("rplidar_ros")
    camera_config = os.path.join(perception_share, "config", "rgb_camera.yaml")

    lidar_port_arg = DeclareLaunchArgument(
        "lidar_serial_port",
        default_value="/dev/ttyUSB1",
        description="RPLidar serial port (Mega usually on /dev/ttyUSB0).",
    )
    camera_device_arg = DeclareLaunchArgument(
        "camera_device_index",
        default_value="0",
        description="V4L2 camera device index (/dev/videoN).",
    )
    use_fake_lidar_arg = DeclareLaunchArgument(
        "use_fake_lidar",
        default_value="false",
        description="Use simulated lidar instead of hardware.",
    )

    rgb_camera = Node(
        package="rover_perception",
        executable="rgb_camera_node",
        name="rgb_camera_node",
        output="screen",
        parameters=[
            camera_config,
            {"device_index": LaunchConfiguration("camera_device_index")},
        ],
    )

    rplidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rplidar_share, "launch", "rplidar_c1_launch.py")
        ),
        launch_arguments={
            "serial_port": LaunchConfiguration("lidar_serial_port"),
            "frame_id": "lidar_link",
        }.items(),
        condition=UnlessCondition(LaunchConfiguration("use_fake_lidar")),
    )

    fake_lidar_share = get_package_share_directory("rover_fake_lidar")
    fake_lidar_config = os.path.join(fake_lidar_share, "config", "fake_lidar.yaml")
    fake_lidar = Node(
        package="rover_fake_lidar",
        executable="fake_lidar_node",
        name="fake_lidar_node",
        output="screen",
        parameters=[fake_lidar_config],
        condition=IfCondition(LaunchConfiguration("use_fake_lidar")),
    )

    return LaunchDescription(
        [
            lidar_port_arg,
            camera_device_arg,
            use_fake_lidar_arg,
            rgb_camera,
            rplidar,
            fake_lidar,
        ]
    )
