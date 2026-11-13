import numpy as np

from safegait.kinematics import RobotSpec, ee_position, jacobian, step

SPEC = RobotSpec()


def test_fk_straight_arm():
    s = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    expected = np.array([1.0 + 0.10 + 0.35 + 0.30 + 0.20, 2.0])
    assert np.allclose(ee_position(SPEC, s), expected)


def test_fk_rotates_with_base():
    s = np.array([0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0])
    assert np.allclose(ee_position(SPEC, s), [0.0, 0.95])


def test_jacobian_matches_numerical_derivative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = np.concatenate([rng.uniform(-2, 2, 2), [rng.uniform(-3, 3)], rng.uniform(-2, 2, 3)])
        jac = jacobian(SPEC, s)
        eps = 1e-6
        for i in range(5):
            u = np.zeros(5)
            u[i] = eps
            numeric = (ee_position(SPEC, step(SPEC, s, u, 1.0)) - ee_position(SPEC, s)) / eps
            assert np.allclose(jac[:, i], numeric, atol=1e-4)


def test_base_cannot_move_sideways():
    s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    nxt = step(SPEC, s, np.array([0.5, 0.0, 0, 0, 0]), 0.1)
    assert nxt[1] == 0.0 and nxt[0] > 0.0
