from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Float64, String
from visualization_msgs.msg import Marker, MarkerArray

from safegait.kinematics import RobotSpec, arm_points, step
from safegait.scene import Scene
from safegait_ros.core import SafeGaitController

ARM_JOINTS = ("arm_joint_1", "arm_joint_2", "arm_joint_3")
DT = 0.05


def _color(r, g, b, a=1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _pt(x, y, z=0.05) -> Point:
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


class SafeGaitNode(Node):
    def __init__(self) -> None:
        super().__init__("safegait_node")
        self.declare_parameter("policy", "planned")
        self.declare_parameter("use_validator", True)
        self.declare_parameter("model_path", "")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("initial_command", "")
        self.declare_parameter("object_names", ["red_cube"])
        self.declare_parameter("object_xy", [2.0, -1.0])
        self.declare_parameter("obstacle_xyr", [1.2, 0.0, 0.3])

        p = lambda n: self.get_parameter(n).value
        names, xy, xyr = list(p("object_names")), list(p("object_xy")), list(p("obstacle_xyr"))
        scene = Scene(
            obstacles=np.array(xyr, dtype=float).reshape(-1, 3),
            objects={n: np.array(xy[2 * i : 2 * i + 2], dtype=float) for i, n in enumerate(names)},
        )
        self.spec = RobotSpec()
        self.frame = p("frame_id")
        self.ctrl = SafeGaitController(scene, self.spec, p("policy"), bool(p("use_validator")), p("model_path"), DT)

        self.pose: tuple[float, float, float] | None = None
        self.q: dict[str, float] = {}
        self.last_status = ""
        self.tick_count = 0

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.create_subscription(String, "/safegait/command", self._on_command, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.arm_pubs = [self.create_publisher(Float64, f"/arm_joint_{i + 1}/cmd_vel", 10) for i in range(3)]
        self.marker_pub = self.create_publisher(MarkerArray, "/safegait/markers", 10)
        self.status_pub = self.create_publisher(String, "/safegait/status", 10)
        self.create_timer(DT, self._tick)

        if p("initial_command"):
            self._handle_command(p("initial_command"))
        self.get_logger().info(
            f"safegait ready: policy={p('policy')}, validator={'on' if p('use_validator') else 'OFF'}, "
            f"objects={names}, obstacles={len(scene.obstacles)}. Send a command on /safegait/command."
        )

    def _on_odom(self, msg: Odometry) -> None:
        o = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z))
        self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)

    def _on_joints(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in ARM_JOINTS:
                self.q[name] = float(pos)

    def _on_command(self, msg: String) -> None:
        self._handle_command(msg.data)

    def _handle_command(self, text: str) -> None:
        task = self.ctrl.set_command(text)
        if task is None:
            self.get_logger().warn(f"did not understand '{text}' (known objects: {list(self.ctrl.scene.objects)})")
        else:
            self.get_logger().info(f"command accepted: {task.kind} {task.target}")

    def _tick(self) -> None:
        if self.pose is None or len(self.q) < 3:
            return
        s = np.array([*self.pose, *(self.q[n] for n in ARM_JOINTS)])
        d = self.ctrl.step(s)
        twist = Twist()
        twist.linear.x, twist.angular.z = float(d.u[0]), float(d.u[1])
        self.cmd_pub.publish(twist)
        for pub, rate in zip(self.arm_pubs, d.u[2:]):
            msg = Float64()
            msg.data = float(rate)
            pub.publish(msg)
        if d.status != self.last_status:
            self.last_status = d.status
            self.status_pub.publish(String(data=f"{d.status} ({self.ctrl.task.target if self.ctrl.task else '-'})"))
        self.tick_count += 1
        if self.tick_count % 4 == 0:
            self._publish_markers(s, d)

    def _marker(self, mid: int, mtype: int, ns: str) -> Marker:
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    def _publish_markers(self, s: np.ndarray, d) -> None:
        arr = MarkerArray()
        for i, (cx, cy, r) in enumerate(self.ctrl.scene.obstacles):
            m = self._marker(i, Marker.CYLINDER, "obstacles")
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(cx), float(cy), 0.25
            m.scale.x = m.scale.y = float(2 * r)
            m.scale.z = 0.5
            m.color = _color(0.5, 0.55, 0.65, 0.9)
            arr.markers.append(m)
        for i, (name, xy) in enumerate(self.ctrl.scene.objects.items()):
            is_target = self.ctrl.task is not None and self.ctrl.task.target == name
            m = self._marker(i, Marker.SPHERE, "objects")
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(xy[0]), float(xy[1]), 0.1
            m.scale.x = m.scale.y = m.scale.z = 0.2 if is_target else 0.12
            m.color = _color(0.2, 0.8, 0.3) if is_target else _color(0.8, 0.8, 0.2, 0.7)
            arr.markers.append(m)
        if d.path:
            m = self._marker(0, Marker.LINE_STRIP, "path")
            m.scale.x = 0.03
            m.color = _color(0.2, 0.4, 0.9)
            m.points = [_pt(p[0], p[1]) for p in d.path]
            arr.markers.append(m)
        if d.chunk is not None:
            cur, pts = s.copy(), [_pt(s[0], s[1], 0.08)]
            for u in d.chunk:
                cur = step(self.spec, cur, u, DT)
                pts.append(_pt(cur[0], cur[1], 0.08))
            for k, (a, b, col) in enumerate(
                [(0, d.n_safe + 1, _color(0.1, 0.8, 0.2)), (d.n_safe, len(pts), _color(0.9, 0.1, 0.1))]
            ):
                m = self._marker(k, Marker.LINE_STRIP, "chunk")
                m.scale.x = 0.05
                m.color = col
                m.points = pts[a:b]
                arr.markers.append(m)
        arm = self._marker(0, Marker.LINE_STRIP, "arm_prediction")
        arm.scale.x = 0.02
        arm.color = _color(0.9, 0.5, 0.1)
        arm.points = [_pt(p[0], p[1], 0.2) for p in arm_points(self.spec, s)]
        arr.markers.append(arm)
        txt = self._marker(0, Marker.TEXT_VIEW_FACING, "status")
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(s[0]), float(s[1]), 0.8
        txt.scale.z = 0.15
        txt.color = _color(1, 1, 1)
        txt.text = f"{d.status}" + (f"  [would violate: {', '.join(d.violations)}]" if d.violations else "")
        arr.markers.append(txt)
        self.marker_pub.publish(arr)


def main() -> None:
    rclpy.init()
    node = SafeGaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
