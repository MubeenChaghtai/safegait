import numpy as np

from safegait.kinematics import RobotSpec, step
from safegait.localisation import ActuationNoise, PoseEKF, SensorSpec, measure, nees, wrap
from safegait.localised_runner import run_localised_episode
from safegait.policy import JacobianPolicy
from safegait.scene import Scene

SPEC = RobotSpec()
SENSOR = SensorSpec()
DT = 0.05


def _ekf(pose=(0.5, -0.3, 0.7)):
    return PoseEKF(np.array(pose), np.diag([0.05, 0.05, 0.03]) ** 2, SENSOR)


def test_predict_matches_robot_motion_model():
    ekf = _ekf()
    u = np.array([0.6, 0.4, 0, 0, 0])
    full = np.array([*ekf.x, 0, 0, 0])
    ekf.predict(u, DT)
    assert np.allclose(ekf.x, step(SPEC, full, u, DT)[:3])


def test_predict_jacobian_matches_numerical_derivative():
    u = np.array([0.6, 0.4, 0, 0, 0])
    base = np.array([0.5, -0.3, 0.7])
    eps = 1e-6
    for i in range(3):
        ekf = _ekf(tuple(base))
        ekf.v_std, ekf.w_std = (0.0, 0.0), 0.0
        ekf.P = np.zeros((3, 3))
        ekf.P[i, i] = 1.0
        ekf.predict(u, DT)
        d = np.zeros(3)
        d[i] = eps
        shifted = step(SPEC, np.array([*(base + d), 0, 0, 0]), u, DT)[:3]
        nominal = step(SPEC, np.array([*base, 0, 0, 0]), u, DT)[:3]
        col = (shifted - nominal) / eps
        assert np.allclose(ekf.P, np.outer(col, col), atol=1e-5)


def test_update_pulls_estimate_toward_truth_and_shrinks_covariance():
    truth = np.array([1.0, 0.5, 0.3])
    ekf = PoseEKF(truth + np.array([0.15, -0.10, 0.08]), np.diag([0.2, 0.2, 0.1]) ** 2, SENSOR)
    err0 = np.linalg.norm(ekf.x[:2] - truth[:2])
    trace0 = np.trace(ekf.P)
    rng = np.random.default_rng(0)
    lms = np.array([[2.0, 1.5], [0.0, 2.0], [2.5, -1.0]])
    for i, z in measure(SENSOR, truth, lms, rng):
        ekf.update(z, lms[i])
    assert np.linalg.norm(ekf.x[:2] - truth[:2]) < err0
    assert np.trace(ekf.P) < trace0


def test_covariance_stays_symmetric_positive_definite():
    ekf = _ekf()
    rng = np.random.default_rng(1)
    lms = np.array([[2.0, 1.5], [0.0, 2.0]])
    for _ in range(200):
        ekf.predict(np.array([0.5, 0.3, 0, 0, 0]), DT)
        for i, z in measure(SENSOR, ekf.x, lms, rng):
            ekf.update(z, lms[i])
    assert np.allclose(ekf.P, ekf.P.T)
    assert np.all(np.linalg.eigvalsh(ekf.P) > 0)


def test_measure_respects_max_range():
    lms = np.array([[1.0, 0.0], [10.0, 0.0]])
    out = measure(SENSOR, np.zeros(3), lms, np.random.default_rng(0))
    assert [i for i, _ in out] == [0]


def test_wrap():
    assert abs(wrap(3 * np.pi) - np.pi) < 1e-9 or abs(wrap(3 * np.pi) + np.pi) < 1e-9


def _run(use_landmarks, seed):
    rng = np.random.default_rng(seed)
    lms = rng.uniform([-0.5, -2.5], [3.5, 2.5], size=(8, 2))
    scene = Scene(landmarks=lms)
    s0 = np.array([0.0, 0.0, 0.0, 0.5, -0.9, 0.5])
    target = np.array([3.0, 1.0])
    return run_localised_episode(
        JacobianPolicy(SPEC), SPEC, scene, s0, target, "raw", rng, use_landmarks=use_landmarks
    )


def test_landmarks_reduce_position_error():
    with_lm = np.mean([np.mean(_run(True, s).pos_err) for s in range(10)])
    dead = np.mean([np.mean(_run(False, s).pos_err) for s in range(10)])
    assert with_lm < dead


def test_filter_is_roughly_consistent():
    vals = np.concatenate([_run(True, s).nees for s in range(10)])
    assert 0.5 < vals.mean() < 12.0
