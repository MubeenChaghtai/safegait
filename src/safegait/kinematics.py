from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_STATE = 6
N_CMD = 5


@dataclass(frozen=True)
class RobotSpec:
    link_lengths: tuple[float, ...] = (0.35, 0.30, 0.20)
    mount_offset: tuple[float, float] = (0.10, 0.0)
    base_radius: float = 0.30
    arm_radius: float = 0.03
    q_lo: tuple[float, ...] = (-2.6, -2.6, -2.6)
    q_hi: tuple[float, ...] = (2.6, 2.6, 2.6)
    v_max: float = 0.8
    w_max: float = 1.5
    qd_max: float = 2.0

    def cmd_limits(self) -> np.ndarray:
        return np.array([self.v_max, self.w_max, self.qd_max, self.qd_max, self.qd_max])


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def arm_points(spec: RobotSpec, s: np.ndarray) -> np.ndarray:
    theta = s[2]
    p = s[:2] + _rot(theta) @ np.asarray(spec.mount_offset)
    pts = [p]
    angle = theta
    for length, qi in zip(spec.link_lengths, s[3:]):
        angle += qi
        p = p + length * np.array([np.cos(angle), np.sin(angle)])
        pts.append(p)
    return np.array(pts)


def ee_position(spec: RobotSpec, s: np.ndarray) -> np.ndarray:
    return arm_points(spec, s)[-1]


def jacobian(spec: RobotSpec, s: np.ndarray) -> np.ndarray:
    pts = arm_points(spec, s)
    ee = pts[-1]
    theta = s[2]
    jac = np.zeros((2, N_CMD))
    jac[:, 0] = [np.cos(theta), np.sin(theta)]
    r = ee - s[:2]
    jac[:, 1] = [-r[1], r[0]]
    for i in range(3):
        r = ee - pts[i]
        jac[:, 2 + i] = [-r[1], r[0]]
    return jac


def step(spec: RobotSpec, s: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    v, w = u[0], u[1]
    nxt = s.copy()
    mid = s[2] + 0.5 * w * dt
    nxt[0] += v * np.cos(mid) * dt
    nxt[1] += v * np.sin(mid) * dt
    nxt[2] = (s[2] + w * dt + np.pi) % (2 * np.pi) - np.pi
    nxt[3:] += u[2:] * dt
    return nxt
