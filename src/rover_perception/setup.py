from glob import glob
from setuptools import setup

package_name = "rover_perception"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="Perception package for strawberry rover RGB camera integration.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rgb_camera_node = rover_perception.rgb_camera_node:main",
            "camera_preprocess_node = rover_perception.camera_preprocess_node:main",
            "sensor_fusion_node = rover_perception.sensor_fusion_node:main",
        ],
    },
)
