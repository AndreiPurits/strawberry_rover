from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    host_arg = DeclareLaunchArgument(
        "host",
        default_value="0.0.0.0",
        description="Web server host bind address.",
    )
    port_arg = DeclareLaunchArgument(
        "port",
        default_value="8080",
        description="Web server TCP port.",
    )

    web_server = Node(
        package="rover_web_interface",
        executable="rover_web_server",
        output="screen",
        arguments=[
            "--host",
            LaunchConfiguration("host"),
            "--port",
            LaunchConfiguration("port"),
        ],
    )

    return LaunchDescription([host_arg, port_arg, web_server])
