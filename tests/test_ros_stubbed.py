import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROS_PKG = Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "safegait_ros"


class Msg:
    _lists = {"points", "markers"}

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, n):
        if n.startswith("__"):
            raise AttributeError(n)
        v = [] if n in self._lists else 0.0 if n in {"x", "y", "z", "w", "r", "g", "b", "a"} else Msg()
        object.__setattr__(self, n, v)
        return v


class JointStateMsg(Msg):
    _lists = {"name", "position", "velocity", "effort"}


class Bus:
    def __init__(self):
        self.subs, self.sent, self.now_ns = {}, {}, 0

    def publish(self, topic, msg):
        self.sent.setdefault(topic, []).append(msg)
        for cb in self.subs.get(topic, []):
            cb(msg)


class Duration:
    def __init__(self, ns):
        self.nanoseconds = ns


class Time:
    def __init__(self, ns):
        self.nanoseconds = ns

    def __sub__(self, other):
        return Duration(self.nanoseconds - other.nanoseconds)

    def to_msg(self):
        return Msg()


@pytest.fixture()
def ros(monkeypatch):
    bus = Bus()

    class Param:
        def __init__(self, value):
            self.value = value

    class Node:
        overrides: dict = {}

        def __init__(self, name):
            self._params = {}

        def declare_parameter(self, name, default):
            self._params[name] = self.overrides.get(name, default)

        def get_parameter(self, name):
            return Param(self._params[name])

        def create_subscription(self, typ, topic, cb, qos):
            bus.subs.setdefault(topic, []).append(cb)

        def create_publisher(self, typ, topic, qos):
            return types.SimpleNamespace(publish=lambda m, t=topic: bus.publish(t, m))

        def create_timer(self, period, cb):
            return None

        def get_clock(self):
            return types.SimpleNamespace(now=lambda: Time(bus.now_ns))

        def get_logger(self):
            return types.SimpleNamespace(info=lambda *a: None, warn=lambda *a: None)

    class Marker(Msg):
        ADD = 0
        CYLINDER, SPHERE, LINE_STRIP, TEXT_VIEW_FACING = 3, 2, 4, 9

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    mod("rclpy")
    mod("rclpy.node", Node=Node)
    mod("geometry_msgs.msg", Twist=Msg, Point=Msg, TransformStamped=Msg)
    mod("nav_msgs.msg", Odometry=Msg)
    mod("sensor_msgs.msg", JointState=JointStateMsg)
    mod("std_msgs.msg", String=Msg, Float64=Msg, ColorRGBA=Msg)
    mod("visualization_msgs.msg", Marker=Marker, MarkerArray=Msg)
    mod("tf2_ros", TransformBroadcaster=lambda node: types.SimpleNamespace(sendTransform=lambda m: None))
    monkeypatch.syspath_prepend(str(ROS_PKG))
    for k in [k for k in sys.modules if k.startswith("safegait_ros")]:
        monkeypatch.delitem(sys.modules, k)
    yield types.SimpleNamespace(
        bus=bus, Node=Node,
        node_mod=importlib.import_module("safegait_ros.safegait_node"),
        fake_mod=importlib.import_module("safegait_ros.fake_robot"),
    )
    for k in [k for k in sys.modules if k.startswith("safegait_ros")]:
        sys.modules.pop(k, None)


def _run(ros, params, command, seconds=45.0):
    from safegait.kinematics import RobotSpec
    from safegait.safety import clearance
    from safegait.scene import Scene

    ros.Node.overrides = params
    fake = ros.fake_mod.FakeRobot()
    node = ros.node_mod.SafeGaitNode()
    ros.bus.publish("/safegait/command", Msg(data=command))
    spec = RobotSpec()
    scene = Scene(obstacles=np.array(params["obstacle_xyr"], dtype=float).reshape(-1, 3))
    worst_gap, max_cmd = np.inf, np.zeros(5)
    for i in range(int(seconds / 0.01)):
        ros.bus.now_ns += 10_000_000
        if i % 2 == 0:
            fake._tick()
            worst_gap = min(worst_gap, clearance(spec, scene, fake.s))
        if i % 5 == 0:
            node._tick()
    for m in ros.bus.sent.get("/cmd_vel", []):
        max_cmd[:2] = np.maximum(max_cmd[:2], [abs(m.linear.x), abs(m.angular.z)])
    return fake, node, worst_gap, max_cmd


PARAMS = {
    "policy": "planned", "use_validator": True, "object_names": ["red_cube", "blue_cube"],
    "object_xy": [2.6, -0.8, 2.2, 1.4], "obstacle_xyr": [1.1, -0.4, 0.3, 2.6, 0.9, 0.4],
}


def test_closed_loop_reaches_target_without_collision(ros):
    fake, node, worst_gap, max_cmd = _run(ros, PARAMS, "pick up the red cube")
    statuses = [m.data for m in ros.bus.sent["/safegait/status"]]
    assert statuses[-1].startswith("arrived"), statuses
    assert worst_gap > 0.0
    assert max_cmd[0] <= 0.8 + 1e-9 and max_cmd[1] <= 1.5 + 1e-9


def test_publishes_markers_and_arm_commands(ros):
    _run(ros, PARAMS, "pick up the red cube", seconds=5.0)
    markers = ros.bus.sent["/safegait/markers"][-1].markers
    namespaces = {m.ns for m in markers}
    assert {"obstacles", "objects", "chunk", "arm_prediction", "status"} <= namespaces
    assert len(ros.bus.sent["/arm_joint_1/cmd_vel"]) > 10 and len(ros.bus.sent["/arm_joint_3/cmd_vel"]) > 10


def test_unknown_command_leaves_the_robot_still(ros):
    fake, node, _, _ = _run(ros, PARAMS, "sing a song", seconds=3.0)
    assert np.allclose(fake.s[:2], [0.0, 0.0]) and node.ctrl.task is None


def test_straight_policy_is_held_back_by_the_validator(ros):
    params = dict(PARAMS, policy="controller", obstacle_xyr=[1.6, -0.8, 0.4])
    fake, node, worst_gap, _ = _run(ros, params, "pick up the red cube", seconds=40.0)
    statuses = [m.data for m in ros.bus.sent["/safegait/status"]]
    assert any(s.startswith("stalled") for s in statuses) and worst_gap > 0.0
