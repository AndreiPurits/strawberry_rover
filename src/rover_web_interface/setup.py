from glob import glob
from setuptools import setup

package_name = "rover_web_interface"

setup(
    name=package_name,
    version="0.0.0",
    packages=[
        package_name,
        f"{package_name}.backend",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/frontend", glob("rover_web_interface/frontend/*")),
    ],
    install_requires=[
        "setuptools",
        "fastapi",
        "uvicorn",
        "websockets",
    ],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="Web visualization interface bridge for Strawberry Rover.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rover_web_server = rover_web_interface.backend.api_server:main",
        ],
    },
)
