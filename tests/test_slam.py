import numpy as np
import pytest

from safegait.kinematics import RobotSpec
from safegait.localisation import SensorSpec, wrap
from safegait.localised_runner import run_localised_episode
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.scene import Scene, random_episode
from safegait.slam import EKFSLAM

SPEC = RobotSpec()
SENSOR = SensorSpec()
DT = 0.05
POSE = np.array([0.5, -0.3, 0.7])


def _slam(**kw):
    return EKFSLAM(POSE, np.diag([0.02, 0.02, 0.01]) ** 2, SENSOR, **kw)


def _z(pose, lm):
    d = np.asarray(lm) - pose[:2]
    return np.array([np.linalg.norm(d), wrap(np.arctan2(d[1], d[0]) - pose[2])])


LMS = np.array([[2.0, 1.0], [1.0, 2.0], [-0.5, 1.5]])


def test_measurement_jacobian_matches_numerical_derivative():
    f = _slam()
    f.observe([(0, _z(POSE, LMS[0]))])
    f.x[3:5] = LMS[0]
    z0, h, idx = f._expected(0)
    eps = 1e-6
    for col, i in enumerate(idx):
        g = _slam()
        g.observe([(0, _z(POSE, LMS[0]))])
        g.x[3:5] = LMS[0]
        g.x[i] += eps
        z1, _, _ = g._expected(0)
        d = z1 - z0
        d[1] = wrap(d[1])
        assert np.allclose(d / eps, h[:, col], atol=1e-5), col


def test_landmark_initialisation_covariance_matches_numerical_jacobians():
    f = _slam()
    z = _z(POSE, LMS[0])
    f.observe([(0, z)])
    assert np.allclose(f.landmarks[0], LMS[0])

    def lm_of(pose, zz):
        a = pose[2] + zz[1]
        return pose[:2] + zz[0] * np.array([np.cos(a), np.sin(a)])

    eps = 1e-6
    g_pose = np.column_stack(
        [(lm_of(POSE + eps * np.eye(3)[i], z) - lm_of(POSE, z)) / eps for i in range(3)]
    )
    g_z = np.column_stack(
        [(lm_of(POSE, z + eps * np.eye(2)[i]) - lm_of(POSE, z)) / eps for i in range(2)]
    )
    p_rr = f.P[:3, :3]
    assert np.allclose(f.landmark_cov(0), g_pose @ p_rr @ g_pose.T + g_z @ f.rm @ g_z.T, atol=1e-8)
    assert np.allclose(f.P[3:5, :3], g_pose @ p_rr, atol=1e-8)
    assert np.allclose(f.P, f.P.T)


def test_predict_moves_only_robot_block_and_cross_terms():
    f = _slam()
    f.observe([(i, _z(POSE, lm)) for i, lm in enumerate(LMS)])
    lm_before, p_ll_before = f.landmarks.copy(), f.P[3:, 3:].copy()
    f.predict(np.array([0.6, 0.4, 0, 0, 0]), DT)
    assert np.allclose(f.landmarks, lm_before)
    assert np.allclose(f.P[3:, 3:], p_ll_before)
    assert np.allclose(f.P, f.P.T)
    assert np.all(np.linalg.eigvalsh(f.P) > -1e-12)


def test_stationary_exact_measurements_recover_map_without_duplicates():
    f = _slam()
    for _ in range(20):
        f.predict(np.zeros(5), DT)
        f.observe([(i, _z(POSE, lm)) for i, lm in enumerate(LMS)])
    assert f.n_landmarks == 3 and f.n_duplicates() == 0
    assert np.allclose(f.landmarks, LMS, atol=1e-9)
    assert f.n_wrong == 0 and f.n_ambiguous == 0


def test_gating_new_associated_and_ambiguous():
    f = _slam()
    f.observe([(0, _z(POSE, LMS[0]))])
    assert f.n_landmarks == 1

    f.observe([(0, _z(POSE, LMS[0]))])
    assert f.n_landmarks == 1 and f.n_associated == 1

    f.observe([(1, _z(POSE, LMS[1]))])
    assert f.n_landmarks == 2

    z_off = _z(POSE, LMS[0] + np.array([0.15, 0.0]))
    d2 = f._mahalanobis2(0, z_off)
    g = _slam(gate=0.5 * d2, new_gate=2.0 * d2)
    g.observe([(0, _z(POSE, LMS[0]))])
    n_before = g.n_landmarks
    g.observe([(0, z_off)])
    assert g.n_landmarks == n_before and g.n_ambiguous == 1 and g.n_associated == 0


def test_a_landmark_is_matched_by_at_most_one_detection_per_scan():
    f = _slam()
    f.observe([(0, _z(POSE, LMS[0]))])
    near = [(0, _z(POSE, LMS[0])), (0, _z(POSE, LMS[0]) + np.array([0.01, 0.0]))]
    f.observe(near)
    assert f.n_associated == 1


def test_unknown_association_never_reads_true_ids():
    rng = np.random.default_rng(0)
    scans = []
    for _ in range(15):
        pose = POSE + rng.normal(0, 0.01, 3)
        scans.append([(i, _z(pose, lm) + rng.normal(0, [0.03, 0.02])) for i, lm in enumerate(LMS)])

    a, b = _slam(), _slam()
    junk = np.random.default_rng(99)
    for scan in scans:
        a.predict(np.array([0.2, 0.1, 0, 0, 0]), DT)
        b.predict(np.array([0.2, 0.1, 0, 0, 0]), DT)
        a.observe(scan)
        b.observe([(int(junk.integers(10_000)), z) for _, z in scan])
    assert np.array_equal(a.x, b.x) and np.array_equal(a.P, b.P)


def test_known_association_uses_the_label_even_when_geometry_would_not():
    f = _slam(association="known")
    f.observe([(5, _z(POSE, LMS[0]))])
    f.observe([(5, _z(POSE, LMS[1]))])
    assert f.n_landmarks == 1 and f.n_associated == 1


def test_covariance_stays_symmetric_positive_definite_over_a_run():
    f = _slam()
    rng = np.random.default_rng(1)
    pose = POSE.copy()
    for _ in range(200):
        u = np.array([0.4, 0.2, 0, 0, 0])
        f.predict(u, DT)
        pose = np.array(
            [pose[0] + 0.4 * np.cos(pose[2]) * DT, pose[1] + 0.4 * np.sin(pose[2]) * DT, pose[2] + 0.2 * DT]
        )
        scan = [
            (i, _z(pose, lm) + rng.normal(0, [SENSOR.range_std, SENSOR.bearing_std]))
            for i, lm in enumerate(LMS)
            if np.linalg.norm(lm - pose[:2]) < SENSOR.max_range
        ]
        f.observe(scan)
    assert np.allclose(f.P, f.P.T)
    assert np.all(np.linalg.eigvalsh(f.P) > 0)


def _run(estimator, seed, use_landmarks=True):
    spec = RobotSpec()
    scene, s0, target = random_episode(np.random.default_rng(seed), spec, 0, 8)
    return run_localised_episode(
        JacobianPolicy(spec), spec, scene, s0, target, "raw", np.random.default_rng(50_000 + seed),
        use_landmarks=use_landmarks, estimator=estimator,
    )


def test_slam_beats_dead_reckoning():
    slam = np.mean([np.mean(_run("slam", s).pos_err) for s in range(10)])
    dead = np.mean([np.mean(_run("ekf", s, use_landmarks=False).pos_err) for s in range(10)])
    assert slam < dead


def test_slam_map_is_close_to_truth_and_filter_is_roughly_consistent():
    runs = [_run("slam", s) for s in range(10)]
    assert np.nanmean([r.map_err for r in runs]) < 0.10
    nees = np.concatenate([r.nees for r in runs])
    assert 0.5 < nees.mean() < 15.0


def test_slam_without_detections_is_rejected():
    with pytest.raises(ValueError):
        _run("slam", 0, use_landmarks=False)


def test_validated_slam_runs_never_collide_or_exceed_limits():
    spec = RobotSpec()
    for ep in range(10):
        scene, s0, target = random_episode(np.random.default_rng(ep), spec, 3, 8)
        policy = NoisyPolicy(JacobianPolicy(spec), spec, np.random.default_rng(10_000 + ep))
        r = run_localised_episode(
            policy, spec, scene, s0, target, "oracle", np.random.default_rng(50_000 + ep), estimator="slam"
        )
        assert not r.collided and not r.limit_violation
