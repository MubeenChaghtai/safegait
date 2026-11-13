import numpy as np

from safegait.kinematics import RobotSpec
from safegait.planner import GridPlanner, PlannedPolicy
from safegait.policy import JacobianPolicy
from safegait.runner import run_episode
from safegait.safety import clearance
from safegait.scene import Scene

SPEC = RobotSpec()
START = np.array([0.0, 0.0, 0.0, 0.5, -0.9, 0.5])


def _blocked_scene():
    return Scene(obstacles=np.array([[1.4, 0.0, 0.45]]), objects={"red_cube": np.array([2.6, 0.0])})


def test_path_keeps_the_base_clear_of_obstacles():
    scene = _blocked_scene()
    path = GridPlanner(scene, SPEC).plan(START[:2], np.array([2.6, 0.0]))
    assert path is not None and len(path) >= 3
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, 30):
            p = a + t * (b - a)
            gap = np.linalg.norm(scene.obstacles[0, :2] - p) - scene.obstacles[0, 2] - SPEC.base_radius
            assert gap > 0.0


def test_path_ends_within_reach_of_the_target():
    target = np.array([2.6, 0.0])
    path = GridPlanner(_blocked_scene(), SPEC).plan(START[:2], target)
    assert 0.4 <= np.linalg.norm(path[-1] - target) <= 0.7 + 0.1


def test_no_path_through_a_wall():
    ys = np.arange(-3.4, 3.5, 0.5)
    wall = Scene(obstacles=np.array([[1.5, y, 0.4] for y in ys]))
    assert GridPlanner(wall, SPEC).plan(START[:2], np.array([3.0, 0.0])) is None


def test_planned_policy_respects_cruise_limits_while_travelling():
    scene = _blocked_scene()
    pol = PlannedPolicy(scene=scene, spec=SPEC)
    chunk = pol.propose(START, scene.objects["red_cube"])
    assert np.all(np.abs(chunk) <= pol.cruise * SPEC.cmd_limits() + 1e-9)


def test_planned_policy_reaches_where_a_straight_line_cannot():
    scene = _blocked_scene()
    target = scene.objects["red_cube"]
    straight = run_episode(JacobianPolicy(SPEC), SPEC, scene, START, target, use_validator=True)
    planned = run_episode(PlannedPolicy(scene=scene, spec=SPEC), SPEC, scene, START, target, use_validator=True)
    assert straight.stalled and not straight.success
    assert planned.success and not planned.collided and not planned.limit_violation


def test_arm_folds_the_way_with_more_room():
    scene = Scene(obstacles=np.array([[0.5, -0.55, 0.2]]))
    pol = PlannedPolicy(scene=scene, spec=SPEC)
    assert pol._choose_fold(START) == 1.0
    mirror = Scene(obstacles=np.array([[0.5, 0.55, 0.2]]))
    assert PlannedPolicy(scene=mirror, spec=SPEC)._choose_fold(START) == -1.0


def test_folded_arm_stays_inside_the_base_disc():
    from safegait.kinematics import arm_points
    from safegait.planner import Q_TRAVEL

    for sign in (-1.0, 1.0):
        s = np.array([0.0, 0.0, 0.0, *(sign * np.abs(Q_TRAVEL))])
        pts = arm_points(SPEC, s)
        worst = max(np.linalg.norm(a + t * (b - a)) for a, b in zip(pts[:-1], pts[1:]) for t in np.linspace(0, 1, 21))
        assert worst < SPEC.base_radius
        assert clearance(SPEC, Scene(obstacles=np.array([[0.7, 0.0, 0.3]])), s) > 0.0
