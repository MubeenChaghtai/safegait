from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster

from safegait.kinematics import RobotSpec, step

ARM_JOINTS = ("arm_joint_1", "arm_joint_2", "arm_joint_3")
RATE = 50.0


class FakeRobot(Node):
    def __init__(self) -> None:
        super().__init__("fake_robot")
        self.declare_parameter("start_xy_yaw_q", [0.0, 0.0, 0.0, 0.5, -0.9, 0.5])
        self.spec = RobotSpec()
        self.s = np.array(self.get_parameter("start_xy_yaw_q").value, dtype=float)
        self.u = np.zeros(5)
        self.last_cmd = self.get_clock().now()
        self.create_subscription(Twist, "/cmd_vel", self._on_twist, 10)
        for i, name in enumerate(ARM_JOINTS):
            self.create_subscription(Float64, f"/{name}/cmd_vel", lambda m, i=i: self._on_joint(i, m), 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.tf = TransformBroadcaster(self)
        self.create_timer(1.0 / RATE, self._tick)

    def _on_twist(self, m: Twist) -> None:
        self.u[0], self.u[1] = m.linear.x, m.angular.z
        self.last_cmd = self.get_clock().now()

    def _on_joint(self, i: int, m: Float64) -> None:
        self.u[2 + i] = m.data
        self.last_cmd = self.get_clock().now()

    def _tick(self) -> None:
        if (self.get_clock().now() - self.last_cmd).nanoseconds * 1e-9 > 0.5:
            self.u[:] = 0.0
        self.s = step(self.spec, self.s, np.clip(self.u, -self.spec.cmd_limits(), self.spec.cmd_limits()), 1.0 / RATE)
        stamp = self.get_clock().now().to_msg()
        x, y, yaw = self.s[:3]
        qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)

        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = float(x), float(y)
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = float(self.u[0]), float(self.u[1])
        self.odom_pub.publish(odom)

        tfm = TransformStamped()
        tfm.header.stamp, tfm.header.frame_id, tfm.child_frame_id = stamp, "odom", "base_link"
        tfm.transform.translation.x, tfm.transform.translation.y, tfm.transform.translation.z = float(x), float(y), 0.1
        tfm.transform.rotation.z, tfm.transform.rotation.w = qz, qw
        self.tf.sendTransform(tfm)

        js = JointState()
        js.header.stamp = stamp
        js.name = [*ARM_JOINTS, "left_wheel_joint", "right_wheel_joint"]
        js.position = [float(q) for q in self.s[3:]] + [0.0, 0.0]
        js.velocity = [float(v) for v in self.u[2:]] + [0.0, 0.0]
        self.js_pub.publish(js)


def main() -> None:
    rclpy.init()
    node = FakeRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
