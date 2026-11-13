from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safegait.kinematics import N_CMD, RobotSpec, ee_position, jacobian


@dataclass
class WholeBodyController:
    spec: RobotSpec
    k_p: float = 2.0
    damping: float = 0.08
    weights: tuple[float, ...] = (3.0, 2.0, 1.0, 1.0, 1.0)
    q_home: tuple[float, ...] = (0.5, -0.9, 0.5)
    k_posture: float = 0.5
    max_ee_speed: float = 0.7

    def command(self, s: np.ndarray, target: np.ndarray) -> np.ndarray:
        v_des = self.k_p * (target - ee_position(self.spec, s))
        speed = np.linalg.norm(v_des)
        if speed > self.max_ee_speed:
            v_des *= self.max_ee_speed / speed

        jac = jacobian(self.spec, s)
        w_inv = np.diag(1.0 / np.asarray(self.weights))
        gram = jac @ w_inv @ jac.T + self.damping**2 * np.eye(2)
        j_pinv = w_inv @ jac.T @ np.linalg.inv(gram)

        u = j_pinv @ v_des
        posture = np.zeros(N_CMD)
        posture[2:] = self.k_posture * (np.asarray(self.q_home) - s[3:])
        u += (np.eye(N_CMD) - j_pinv @ jac) @ posture

        ratio = np.max(np.abs(u) / self.spec.cmd_limits())
        if ratio > 1.0:
            u = u / ratio
        return u
