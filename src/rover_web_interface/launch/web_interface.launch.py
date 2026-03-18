from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import TimerAction
from launch.conditions import IfCondition
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
    auto_cleanup_arg = DeclareLaunchArgument(
        "auto_cleanup_before_start",
        default_value="true",
        description="Kill previous rover_web_server process before launch.",
    )

    cleanup_old_web = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            "pkill -f '[r]over_web_interface/lib/rover_web_interface/rover_web_server' || true",
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("auto_cleanup_before_start")),
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
    delayed_web_server = TimerAction(period=1.0, actions=[web_server])

    return LaunchDescription(
        [host_arg, port_arg, auto_cleanup_arg, cleanup_old_web, delayed_web_server]
    )
