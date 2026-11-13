import numpy as np

from safegait.evaluate_learned import make_episode
from safegait.kinematics import RobotSpec
from safegait.learned import MLP, LearnedPolicy, collect_demos, featurize
from safegait.policy import SaturatedPolicy

SPEC = RobotSpec()


def test_features_are_egocentric():
    s = np.array([0.0, 0.0, 0.3, 0.5, -0.9, 0.5])
    target = np.array([2.0, 1.0])
    a = featurize(SPEC, s, target)
    th, shift = 1.1, np.array([5.0, -3.0])
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    s2 = s.copy()
    s2[:2] = shift + rot @ s[:2]
    s2[2] = s[2] + th
    assert np.allclose(a, featurize(SPEC, s2, shift + rot @ target), atol=1e-9)


def test_mlp_learns_a_simple_function():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (2000, 3))
    y = np.stack([x[:, 0] + x[:, 1], x[:, 2] ** 2], axis=1)
    net = MLP([3, 32, 32, 2], rng)
    hist = net.fit(x, y, epochs=60, rng=rng, log=lambda *_: None)
    assert hist["val"][-1] < 0.01 < hist["val"][0]


def test_save_load_roundtrip_and_policy_shape(tmp_path):
    rng = np.random.default_rng(1)
    x, y = collect_demos(SPEC, 20, rng)
    net = MLP([x.shape[1], 16, y.shape[1]], rng)
    net.fit(x, y, epochs=3, rng=rng, log=lambda *_: None)
    net.save(tmp_path / "m.npz")
    s, target = np.array([0.0, 0.0, 0.2, 0.5, -0.9, 0.5]), np.array([1.5, 0.5])
    a = LearnedPolicy(SPEC, net).propose(s, target)
    b = LearnedPolicy(SPEC, MLP.load(tmp_path / "m.npz")).propose(s, target)
    assert a.shape == (10, 5) and np.allclose(a, b)


def test_saturation_clips_to_limits():
    class Loud:
        def propose(self, s, target):
            return np.full((10, 5), 99.0)

    out = SaturatedPolicy(Loud(), SPEC).propose(np.zeros(6), np.zeros(2))
    assert np.all(np.abs(out) <= SPEC.cmd_limits() + 1e-12)


def test_commands_select_the_commanded_object():
    for seed in range(30):
        scene, _, target, command = make_episode(seed, SPEC, n_obstacles=3)
        assert any(np.allclose(target, p) for p in scene.objects.values()), command
