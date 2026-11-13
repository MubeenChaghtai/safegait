import numpy as np

from safegait.kinematics import RobotSpec, ee_position, step
from safegait.scene import Scene
from safegait_ros.core import SafeGaitController

SPEC = RobotSpec()
START = np.array([0.0, 0.0, 0.0, 0.5, -0.9, 0.5])


def _scene(obstacles=((1.6, 0.0, 0.4),)):
    return Scene(obstacles=np.array(obstacles, dtype=float).reshape(-1, 3),
                 objects={"red_cube": np.array([2.6, 0.0]), "blue_cube": np.array([2.0, 1.5])})


def _drive(ctrl, s, ticks=600):
    for _ in range(ticks):
        d = ctrl.step(s)
        if d.status in ("arrived", "stalled"):
            return s, d
        s = step(SPEC, s, d.u, ctrl.dt)
    return s, d


def test_idle_until_commanded_and_unknown_command_ignored():
    c = SafeGaitController(_scene())
    assert c.step(START).status == "idle" and not c.step(START).u.any()
    assert c.set_command("dance please") is None
    assert c.step(START).status == "idle"


def test_command_selects_object():
    c = SafeGaitController(_scene())
    assert c.set_command("pick up the blue cube").target == "blue_cube"
    assert np.allclose(c.target, [2.0, 1.5])


def test_planned_policy_drives_around_the_obstacle_to_the_target():
    c = SafeGaitController(_scene(), policy="planned")
    c.set_command("reach the red cube")
    s, d = _drive(c, START)
    assert d.status == "arrived"
    assert np.linalg.norm(ee_position(SPEC, s) - c.target) < 0.06


def test_validator_blocks_a_straight_policy_instead_of_crashing():
    c = SafeGaitController(_scene(), policy="controller")
    c.set_command("reach the red cube")
    s, d = _drive(c, START)
    assert d.status == "stalled"
    from safegait.safety import clearance
    assert clearance(SPEC, c.scene, s) >= 0.0


def test_commands_never_exceed_limits_with_validator_on():
    c = SafeGaitController(_scene(), policy="planned")
    c.set_command("reach the red cube")
    s = START
    for _ in range(300):
        d = c.step(s)
        assert np.all(np.abs(d.u) <= SPEC.cmd_limits() + 1e-9)
        s = step(SPEC, s, d.u, c.dt)


def test_decision_exposes_chunk_and_path_for_visualisation():
    c = SafeGaitController(_scene(), policy="planned")
    c.set_command("reach the red cube")
    d = c.step(START)
    assert d.chunk.shape == (10, 5) and d.n_safe > 0 and d.path is not None
