import glob

from setuptools import find_packages, setup

package_name = "safegait_ros"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob.glob("launch/*.py")),
        ("share/" + package_name + "/config", glob.glob("config/*")),
        ("share/" + package_name + "/urdf", glob.glob("urdf/*")),
        ("share/" + package_name + "/worlds", glob.glob("worlds/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "safegait_node = safegait_ros.safegait_node:main",
            "fake_robot = safegait_ros.fake_robot:main",
            "make_world = safegait_ros.make_world:main",
        ]
    },
)
