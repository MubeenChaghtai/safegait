import numpy as np

from safegait.policy import JacobianPolicy
from safegait.kinematics import RobotSpec
from safegait.runner import run_episode
from safegait.scene import Scene, random_episode

SPEC = RobotSpec()


def test_clean_policy_succeeds_in_empty_scene():
    s0 = np.array([0.0, 0.0, 0.0, 0.5, -0.9, 0.5])
    target = np.array([2.0, 0.8])
    for use_validator in (False, True):
        res = run_episode(JacobianPolicy(SPEC), SPEC, Scene(), s0, target, use_validator)
        assert res.success and not res.collided and not res.limit_violation


def test_validated_runs_never_collide():
    rng = np.random.default_rng(3)
    for _ in range(15):
        scene, s0, target = random_episode(rng, SPEC, n_obstacles=4)
        res = run_episode(JacobianPolicy(SPEC), SPEC, scene, s0, target, use_validator=True)
        assert not res.collided and not res.limit_violation
