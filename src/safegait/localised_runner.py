from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safegait.kinematics import RobotSpec, ee_position, step
from safegait.localisation import ActuationNoise, PoseEKF, SensorSpec, measure, nees
from safegait.policy import Policy
from safegait.safety import step_violations
from safegait.scene import Scene
from safegait.slam import EKFSLAM
from safegait.validator import ChunkValidator

MODES = ("raw", "oracle", "est", "est_inflated")
ESTIMATORS = ("ekf", "slam", "slam_known")


@dataclass
class LocResult:
    success: bool = False
    collided: bool = False
    limit_violation: bool = False
    stalled: bool = False
    steps: int = 0
    blocked_chunks: int = 0
    final_error: float = float("nan")
    pos_err: list[float] = field(default_factory=list)
    nees: list[float] = field(default_factory=list)
    sigma: list[float] = field(default_factory=list)
    true_xy: list[np.ndarray] = field(default_factory=list)
    est_xy: list[np.ndarray] = field(default_factory=list)
    slam: EKFSLAM | None = None
    n_seen: int = 0
    map_err: float = float("nan")
    n_duplicates: int = 0
    n_assoc: int = 0
    n_assoc_wrong: int = 0
    n_ambiguous: int = 0


def run_localised_episode(
    policy: Policy,
    spec: RobotSpec,
    scene: Scene,
    s0: np.ndarray,
    target: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    use_landmarks: bool = True,
    estimator: str = "ekf",
    sensor: SensorSpec = SensorSpec(),
    noise: ActuationNoise = ActuationNoise(),
    dt: float = 0.05,
    exec_n: int = 5,
    max_steps: int = 600,
    tol: float = 0.05,
    success_tol: float = 0.10,
    base_margin: float = 0.05,
    inflate_k: float = 2.0,
    max_consecutive_blocks: int = 6,
) -> LocResult:
    assert mode in MODES
    assert estimator in ESTIMATORS
    if estimator != "ekf" and not use_landmarks:
        raise ValueError("SLAM without detections is just dead reckoning; use estimator='ekf'")
    res = LocResult()
    s_true = s0.copy()
    slam_mode = estimator != "ekf"
    seen: set[int] = set()

    init_std = np.array([0.05, 0.05, 0.03])
    init_noise = rng.normal(0.0, init_std)
    if slam_mode:
        ekf = EKFSLAM(
            s0[:3], np.diag([1e-3, 1e-3, 1e-3]) ** 2, sensor,
            association="known" if estimator == "slam_known" else "nn",
        )
    else:
        ekf = PoseEKF(s0[:3] + init_noise, np.diag(init_std**2), sensor)
    v_scale = 1.0 + rng.normal(0.0, noise.v_scale_std)
    w_bias = rng.normal(0.0, noise.w_bias_std)
    consecutive = 0

    while res.steps < max_steps:
        s_hat = np.concatenate([ekf.pose, s_true[3:]])
        chunk = policy.propose(s_hat, target)

        if mode == "raw":
            n = exec_n
        else:
            margin = base_margin
            if mode == "est_inflated":
                margin += inflate_k * (ekf.position_sigma() + 0.95 * ekf.heading_sigma())
            validator = ChunkValidator(spec, scene, dt, margin)
            n = min(validator.safe_prefix(s_true if mode == "oracle" else s_hat, chunk), exec_n)

        if n == 0:
            res.blocked_chunks += 1
            consecutive += 1
            res.steps += exec_n
            if consecutive >= max_consecutive_blocks:
                res.stalled = True
                break
            continue
        consecutive = 0

        for u in chunk[:n]:
            u_real = u.copy()
            u_real[0] = u[0] * v_scale + rng.normal(0.0, noise.v_step_std)
            u_real[1] = u[1] + w_bias + rng.normal(0.0, noise.w_step_std)
            nxt = step(spec, s_true, u_real, dt)
            kinds = step_violations(spec, scene, s_true, nxt, u, margin=0.0)
            res.limit_violation |= "velocity" in kinds or "joint_limit" in kinds
            res.collided |= "collision" in kinds
            s_true = nxt
            res.steps += 1

            ekf.predict(u, dt)
            if use_landmarks:
                scan = measure(sensor, s_true, scene.landmarks, rng)
                if slam_mode:
                    seen.update(i for i, _ in scan)
                    ekf.observe(scan)
                else:
                    for i, z in scan:
                        ekf.update(z, scene.landmarks[i])
            res.pos_err.append(float(np.linalg.norm(ekf.pose[:2] - s_true[:2])))
            res.nees.append(nees(ekf.pose, s_true, ekf.pose_cov))
            res.sigma.append(ekf.position_sigma())
            res.true_xy.append(s_true[:2].copy())
            res.est_xy.append(ekf.pose[:2].copy())

            if res.collided:
                return _finish(res, spec, scene, s_true, target, ekf, seen, success=False)
            est_ee = ee_position(spec, np.concatenate([ekf.pose, s_true[3:]]))
            if np.linalg.norm(est_ee - target) < tol:
                return _finish(res, spec, scene, s_true, target, ekf, seen, success_tol=success_tol)
    return _finish(res, spec, scene, s_true, target, ekf, seen, success=False)


def _finish(
    res: LocResult,
    spec: RobotSpec,
    scene: Scene,
    s_true: np.ndarray,
    target: np.ndarray,
    ekf: PoseEKF | EKFSLAM,
    seen: set[int],
    success: bool | None = None,
    success_tol: float = 0.10,
) -> LocResult:
    res.final_error = float(np.linalg.norm(ee_position(spec, s_true) - target))
    res.success = res.final_error < success_tol if success is None else success
    if isinstance(ekf, EKFSLAM):
        res.slam = ekf
        res.n_seen = len(seen)
        errs = ekf.map_errors(scene.landmarks)
        res.map_err = float(errs.mean()) if len(errs) else float("nan")
        res.n_duplicates = ekf.n_duplicates()
        res.n_assoc, res.n_assoc_wrong, res.n_ambiguous = (
            ekf.n_associated, ekf.n_wrong, ekf.n_ambiguous,
        )
    return res
