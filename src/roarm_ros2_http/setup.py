from glob import glob
from setuptools import setup

package_name = "roarm_ros2_http"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="Minimal ROS2 HTTP bridge for Waveshare RoArm-M3.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "roarm_http_driver = roarm_ros2_http.roarm_http_driver:main",
            "fake_target_publisher = roarm_ros2_http.fake_target_publisher:main",
            "demo_pick_sequence = roarm_ros2_http.demo_pick_sequence:main",
            "roarm_selftest = roarm_ros2_http.roarm_selftest:main",
            "smoke_test = roarm_ros2_http.smoke_test:main",
        ],
    },
)
