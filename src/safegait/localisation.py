from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass(frozen=True)
class SensorSpec:
    range_std: float = 0.05
    bearing_std: float = 0.03
    max_range: float = 3.0


@dataclass(frozen=True)
class ActuationNoise:
    v_scale_std: float = 0.06
    w_bias_std: float = 0.04
    v_step_std: float = 0.02
    w_step_std: float = 0.02


def motion_model(
    pose: np.ndarray, v: float, w: float, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    th = pose[2]
    mid = th + 0.5 * w * dt
    nxt = np.array(
        [pose[0] + v * np.cos(mid) * dt, pose[1] + v * np.sin(mid) * dt, wrap(th + w * dt)]
    )
    f = np.array([[1, 0, -v * np.sin(mid) * dt], [0, 1, v * np.cos(mid) * dt], [0, 0, 1]])
    g = np.array(
        [
            [np.cos(mid) * dt, -0.5 * v * np.sin(mid) * dt**2],
            [np.sin(mid) * dt, 0.5 * v * np.cos(mid) * dt**2],
            [0.0, dt],
        ]
    )
    return nxt, f, g


def measure(
    sensor: SensorSpec, pose: np.ndarray, landmarks: np.ndarray, rng: np.random.Generator
) -> list[tuple[int, np.ndarray]]:
    out = []
    for i, lm in enumerate(landmarks):
        d = lm - pose[:2]
        r = float(np.linalg.norm(d))
        if r > sensor.max_range or r < 1e-6:
            continue
        z = np.array([r, wrap(np.arctan2(d[1], d[0]) - pose[2])])
        z += rng.normal(0.0, [sensor.range_std, sensor.bearing_std])
        z[1] = wrap(z[1])
        out.append((i, z))
    return out


class PoseEKF:
    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        sensor: SensorSpec,
        v_std: tuple[float, float] = (0.08, 0.03),
        w_std: float = 0.05,
    ):
        self.x = np.asarray(mean, dtype=float).copy()
        self.P = np.asarray(cov, dtype=float).copy()
        self.sensor, self.v_std, self.w_std = sensor, v_std, w_std

    @property
    def pose(self) -> np.ndarray:
        return self.x

    @property
    def pose_cov(self) -> np.ndarray:
        return self.P

    def predict(self, u: np.ndarray, dt: float) -> None:
        v, w = float(u[0]), float(u[1])
        self.x, f, g = motion_model(self.x, v, w, dt)
        q_u = np.diag([(self.v_std[0] * abs(v) + self.v_std[1]) ** 2, self.w_std**2])
        self.P = f @ self.P @ f.T + g @ q_u @ g.T

    def update(self, z: np.ndarray, landmark: np.ndarray) -> None:
        d = landmark - self.x[:2]
        q = float(d @ d)
        r = np.sqrt(q)
        z_hat = np.array([r, wrap(np.arctan2(d[1], d[0]) - self.x[2])])
        h = np.array([[-d[0] / r, -d[1] / r, 0.0], [d[1] / q, -d[0] / q, -1.0]])
        innov = z - z_hat
        innov[1] = wrap(innov[1])
        rm = np.diag([self.sensor.range_std**2, self.sensor.bearing_std**2])
        s_mat = h @ self.P @ h.T + rm
        k = self.P @ h.T @ np.linalg.inv(s_mat)
        self.x = self.x + k @ innov
        self.x[2] = wrap(self.x[2])
        i_kh = np.eye(3) - k @ h
        self.P = i_kh @ self.P @ i_kh.T + k @ rm @ k.T

    def position_sigma(self) -> float:
        return float(np.sqrt(np.linalg.eigvalsh(self.P[:2, :2])[-1]))

    def heading_sigma(self) -> float:
        return float(np.sqrt(self.P[2, 2]))


def nees(est: np.ndarray, truth: np.ndarray, cov: np.ndarray) -> float:
    e = est - truth[:3]
    e[2] = wrap(e[2])
    return float(e @ np.linalg.solve(cov, e))
