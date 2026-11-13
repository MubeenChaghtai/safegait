import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("safegait_ros")
    urdf = open(os.path.join(share, "urdf", "mobile_manipulator.urdf")).read()
    params = os.path.join(share, "config", "scene.yaml")
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[{"robot_description": urdf}]),
        Node(package="safegait_ros", executable="fake_robot", output="screen"),
        Node(package="safegait_ros", executable="safegait_node", parameters=[params], output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "config", "safegait.rviz")]),
    ])
