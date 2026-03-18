import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess
from launch.actions import DeclareLaunchArgument
from launch.actions import RegisterEventHandler
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Integrated bringup for field-row simulation and fake sensors."""
    description_share = get_package_share_directory("rover_description")
    fake_lidar_share = get_package_share_directory("rover_fake_lidar")
    simulation_share = get_package_share_directory("rover_simulation")
    fake_camera_share = get_package_share_directory("rover_fake_camera")
    fake_stereo_share = get_package_share_directory("rover_fake_stereo")
    navigation_share = get_package_share_directory("rover_navigation")
    bringup_share = get_package_share_directory("rover_bringup")

    urdf_file = os.path.join(description_share, "urdf", "rover.urdf")
    fake_lidar_config = os.path.join(fake_lidar_share, "config", "fake_lidar.yaml")
    rover_sim_config = os.path.join(
        simulation_share, "config", "rover_pose_simulator.yaml"
    )
    fake_camera_config = os.path.join(fake_camera_share, "config", "fake_camera.yaml")
    fake_stereo_config = os.path.join(fake_stereo_share, "config", "fake_stereo.yaml")
    nav_config = os.path.join(navigation_share, "config", "rover_navigation.yaml")
    rviz_config = os.path.join(bringup_share, "config", "field_sim_debug.rviz")

    with open(urdf_file, "r", encoding="utf-8") as urdf_fp:
        robot_description = urdf_fp.read()

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="false",
        description="Start RViz2 with preconfigured field simulation profile.",
    )
    simulation_config_arg = DeclareLaunchArgument(
        "simulation_config",
        default_value=rover_sim_config,
        description="Path to rover_simulation parameter file.",
    )
    navigation_config_arg = DeclareLaunchArgument(
        "navigation_config",
        default_value=nav_config,
        description="Path to rover_navigation parameter file.",
    )
    enable_navigation_arg = DeclareLaunchArgument(
        "enable_navigation",
        default_value="false",
        description="Start rover_navigation_node (disabled by default for scripted route mode).",
    )
    auto_cleanup_arg = DeclareLaunchArgument(
        "auto_cleanup_before_start",
        default_value="true",
        description="Kill stale rover simulation nodes before startup to avoid duplicates in RViz.",
    )

    pre_cleanup = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            (
                "pkill -f '__node:=robot_state_publisher|__node:=fake_lidar_node|"
                "__node:=rover_pose_simulator|__node:=fake_camera_node|"
                "__node:=fake_stereo_node|__node:=rover_navigation_node' || true"
            ),
        ],
        shell=False,
        condition=IfCondition(LaunchConfiguration("auto_cleanup_before_start")),
        output="screen",
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    fake_lidar = Node(
        package="rover_fake_lidar",
        executable="fake_lidar_node",
        name="fake_lidar_node",
        output="screen",
        parameters=[fake_lidar_config],
    )

    rover_pose_simulator = Node(
        package="rover_simulation",
        executable="rover_pose_simulator",
        name="rover_pose_simulator",
        output="screen",
        parameters=[LaunchConfiguration("simulation_config")],
    )

    fake_camera = Node(
        package="rover_fake_camera",
        executable="fake_camera_node",
        name="fake_camera_node",
        output="screen",
        parameters=[fake_camera_config],
    )

    fake_stereo = Node(
        package="rover_fake_stereo",
        executable="fake_stereo_node",
        name="fake_stereo_node",
        output="screen",
        parameters=[fake_stereo_config],
    )

    rover_navigation = Node(
        package="rover_navigation",
        executable="rover_navigation_node",
        name="rover_navigation_node",
        output="screen",
        parameters=[LaunchConfiguration("navigation_config")],
        condition=IfCondition(LaunchConfiguration("enable_navigation")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("use_rviz")),
    )

    startup_actions = [
        robot_state_publisher,
        fake_lidar,
        rover_pose_simulator,
        fake_camera,
        fake_stereo,
        rover_navigation,
        rviz,
    ]

    start_after_cleanup = RegisterEventHandler(
        OnProcessExit(
            target_action=pre_cleanup,
            on_exit=startup_actions,
        )
    )

    return LaunchDescription(
        [
            use_rviz_arg,
            simulation_config_arg,
            navigation_config_arg,
            enable_navigation_arg,
            auto_cleanup_arg,
            pre_cleanup,
            start_after_cleanup,
        ]
    )

