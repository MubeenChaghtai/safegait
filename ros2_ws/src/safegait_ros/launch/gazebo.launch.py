import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("safegait_ros")
    gz_share = get_package_share_directory("ros_gz_sim")
    urdf = open(os.path.join(share, "urdf", "mobile_manipulator.urdf")).read()
    params = os.path.join(share, "config", "scene.yaml")
    sim = {"use_sim_time": True}

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-r {os.path.join(share, 'worlds', 'safegait.sdf')}"}.items(),
    )
    spawn = Node(package="ros_gz_sim", executable="create",
                 arguments=["-topic", "robot_description", "-name", "safegait_bot", "-x", "0", "-y", "0", "-z", "0.12"])
    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge", output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/arm_joint_1/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double",
            "/arm_joint_2/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double",
            "/arm_joint_3/cmd_vel@std_msgs/msg/Float64]gz.msgs.Double",
        ],
    )
    return LaunchDescription([
        gazebo,
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}, sim]),
        spawn,
        bridge,
        Node(package="safegait_ros", executable="safegait_node", parameters=[params, sim], output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "config", "safegait.rviz")],
             parameters=[sim]),
    ])
