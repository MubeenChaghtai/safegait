from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from safegait.physics_g1 import DEFAULT_XML, ArmEnv, ArmPolicy, ArmValidator, random_arm_episode, run_arm_episode

pytestmark = pytest.mark.skipif(not Path(DEFAULT_XML).exists(), reason="G1 model not fetched (scripts/fetch_g1.sh)")


@pytest.fixture(scope="module")
def env():
    return ArmEnv()


def test_clearance_sign_matches_physics_contacts(env):
    env.reset([], np.zeros(3))
    hand = env.hand_pos()
    for offset, touching in ((0.25, False), (0.0, True)):
        env.reset([(hand + np.array([offset, 0, 0]), 0.05)], hand + 0.2)
        mujoco.mj_forward(env.m, env.d)
        assert (env.clearance(env.q()) < 0.0) == touching
        assert (env.d.ncon > 0) == touching


def test_scratch_sees_the_same_obstacles_as_the_simulation(env):
    env.reset([], np.zeros(3))
    hand = env.hand_pos()
    env.reset([(hand + np.array([0.15, 0, 0]), 0.05)], hand + 0.2)
    assert env.clearance(env.q()) < 0.2


def test_reaches_a_free_target_in_physics(env):
    rng = np.random.default_rng(0)
    wins = 0
    for _ in range(5):
        _, target = random_arm_episode(env, rng, 0)
        wins += run_arm_episode(env, ArmPolicy(env), [], target, use_validator=False).success
    assert wins >= 4


def test_kinematic_prediction_tracks_physics(env):
    rng = np.random.default_rng(1)
    _, target = random_arm_episode(env, rng, 0)
    res = run_arm_episode(env, ArmPolicy(env), [], target, use_validator=False)
    assert np.mean(res.hand_gap) < 0.02


def test_validator_prevents_physics_collisions(env):
    rng = np.random.default_rng(2)
    raw = valid = 0
    for _ in range(40):
        obstacles, target = random_arm_episode(env, rng, 2)
        raw += run_arm_episode(env, ArmPolicy(env), obstacles, target, False).collided
        valid += run_arm_episode(env, ArmPolicy(env), obstacles, target, True, margin=0.03).collided
    assert raw > 0 and valid == 0


def test_validator_rejects_out_of_limit_chunk(env):
    env.reset([], np.zeros(3))
    chunk = np.zeros((10, 4))
    chunk[3, 0] = 10.0
    assert ArmValidator(env).safe_prefix(env.q(), chunk) == 3
