import numpy as np

from safegait.controller import WholeBodyController
from safegait.kinematics import RobotSpec, ee_position, step
from safegait.safety import clearance
from safegait.scene import Scene
from safegait.validator import ChunkValidator

SPEC = RobotSpec()
DT = 0.05
HOME = (0.5, -0.9, 0.5)


def _s0(theta=0.0):
    return np.array([0.0, 0.0, theta, *HOME])


def test_controller_reaches_a_free_target():
    ctrl = WholeBodyController(SPEC)
    s, target = _s0(), np.array([2.0, 1.0])
    for _ in range(800):
        s = step(SPEC, s, ctrl.command(s, target), DT)
    assert np.linalg.norm(ee_position(SPEC, s) - target) < 0.05


def test_controller_respects_command_limits():
    ctrl = WholeBodyController(SPEC)
    u = ctrl.command(_s0(), np.array([5.0, -4.0]))
    assert np.all(np.abs(u) <= SPEC.cmd_limits() + 1e-9)


def test_validator_accepts_stopped_chunk():
    val = ChunkValidator(SPEC, Scene(), DT)
    assert val.safe_prefix(_s0(), np.zeros((10, 5))) == 10
    assert val.check(_s0(), np.zeros((10, 5))) == []


def test_validator_flags_velocity_violation():
    val = ChunkValidator(SPEC, Scene(), DT)
    chunk = np.zeros((10, 5))
    chunk[4, 0] = 3 * SPEC.v_max
    kinds = {(v.step, v.kind) for v in val.check(_s0(), chunk)}
    assert (4, "velocity") in kinds
    assert val.safe_prefix(_s0(), chunk) == 4


def test_validator_flags_joint_limit():
    val = ChunkValidator(SPEC, Scene(), DT)
    s = _s0()
    s[3] = SPEC.q_hi[0] - 0.01
    chunk = np.zeros((10, 5))
    chunk[:, 2] = SPEC.qd_max
    assert any(v.kind == "joint_limit" for v in val.check(s, chunk))
    assert val.safe_prefix(s, chunk) < 10


def test_validator_stops_before_an_obstacle():
    scene = Scene(obstacles=np.array([[2.0, 0.0, 0.3]]))
    val = ChunkValidator(SPEC, scene, DT)
    chunk = np.zeros((40, 5))
    chunk[:, 0] = SPEC.v_max
    n = val.safe_prefix(_s0(), chunk)
    assert 0 < n < 40
    s = _s0()
    for u in chunk[:n]:
        s = step(SPEC, s, u, DT)
    assert clearance(SPEC, scene, s) >= 0.0


def test_validator_lets_a_robot_inside_the_margin_drive_away():
    gap = 0.03
    scene = Scene(obstacles=np.array([[-(SPEC.base_radius + 0.3 + gap), 0.0, 0.3]]))
    s = np.zeros(6)
    val = ChunkValidator(SPEC, scene, DT)
    assert 0.0 < clearance(SPEC, scene, s) < val.margin

    away = np.zeros((5, 5))
    away[:, 0] = SPEC.v_max
    assert val.safe_prefix(s, away) == 5

    toward = np.zeros((5, 5))
    toward[:, 0] = -SPEC.v_max
    assert val.safe_prefix(s, toward) == 0
