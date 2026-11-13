from __future__ import annotations

import numpy as np

from safegait.localisation import SensorSpec, motion_model, wrap

GATE_ASSOCIATE = 9.21
GATE_NEW = 18.42


class EKFSLAM:
    def __init__(
        self,
        pose: np.ndarray,
        pose_cov: np.ndarray,
        sensor: SensorSpec,
        association: str = "nn",
        v_std: tuple[float, float] = (0.08, 0.03),
        w_std: float = 0.05,
        gate: float = GATE_ASSOCIATE,
        new_gate: float = GATE_NEW,
    ):
        assert association in ("nn", "known")
        self.x = np.asarray(pose, dtype=float).copy()
        self.P = np.asarray(pose_cov, dtype=float).copy()
        self.sensor, self.v_std, self.w_std = sensor, v_std, w_std
        self.association, self.gate, self.new_gate = association, gate, new_gate
        self.rm = np.diag([sensor.range_std**2, sensor.bearing_std**2])

        self.origin: list[int] = []
        self._id_to_idx: dict[int, int] = {}
        self.n_associated = 0
        self.n_wrong = 0
        self.n_ambiguous = 0


    @property
    def n_landmarks(self) -> int:
        return (len(self.x) - 3) // 2

    @property
    def pose(self) -> np.ndarray:
        return self.x[:3]

    @property
    def pose_cov(self) -> np.ndarray:
        return self.P[:3, :3]

    @property
    def landmarks(self) -> np.ndarray:
        return self.x[3:].reshape(-1, 2)

    def landmark_cov(self, j: int) -> np.ndarray:
        i = 3 + 2 * j
        return self.P[i : i + 2, i : i + 2]

    def position_sigma(self) -> float:
        return float(np.sqrt(np.linalg.eigvalsh(self.P[:2, :2])[-1]))

    def heading_sigma(self) -> float:
        return float(np.sqrt(self.P[2, 2]))


    def predict(self, u: np.ndarray, dt: float) -> None:
        v, w = float(u[0]), float(u[1])
        pose, f, g = motion_model(self.x[:3], v, w, dt)
        q_u = np.diag([(self.v_std[0] * abs(v) + self.v_std[1]) ** 2, self.w_std**2])
        p_rl = f @ self.P[:3, 3:]
        self.P[:3, :3] = f @ self.P[:3, :3] @ f.T + g @ q_u @ g.T
        self.P[:3, 3:] = p_rl
        self.P[3:, :3] = p_rl.T
        self.x[:3] = pose


    def _expected(self, j: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        i = 3 + 2 * j
        d = self.x[i : i + 2] - self.x[:2]
        q = float(d @ d)
        r = np.sqrt(q)
        z_hat = np.array([r, wrap(np.arctan2(d[1], d[0]) - self.x[2])])
        h = np.array(
            [
                [-d[0] / r, -d[1] / r, 0.0, d[0] / r, d[1] / r],
                [d[1] / q, -d[0] / q, -1.0, -d[1] / q, d[0] / q],
            ]
        )
        return z_hat, h, [0, 1, 2, i, i + 1]

    def _mahalanobis2(self, j: int, z: np.ndarray) -> float:
        z_hat, h, idx = self._expected(j)
        innov = z - z_hat
        innov[1] = wrap(innov[1])
        s = h @ self.P[np.ix_(idx, idx)] @ h.T + self.rm
        return float(innov @ np.linalg.solve(s, innov))

    def _update(self, j: int, z: np.ndarray) -> None:
        z_hat, h5, idx = self._expected(j)
        h = np.zeros((2, len(self.x)))
        h[:, idx] = h5
        innov = z - z_hat
        innov[1] = wrap(innov[1])
        s = h @ self.P @ h.T + self.rm
        k = self.P @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ innov
        self.x[2] = wrap(self.x[2])
        i_kh = np.eye(len(self.x)) - k @ h
        self.P = i_kh @ self.P @ i_kh.T + k @ self.rm @ k.T
        self.P = 0.5 * (self.P + self.P.T)

    def _initialise(self, z: np.ndarray, true_id: int) -> int:
        r, b = float(z[0]), float(z[1])
        ang = self.x[2] + b
        c, s = np.cos(ang), np.sin(ang)
        lm = self.x[:2] + r * np.array([c, s])
        g_pose = np.array([[1.0, 0.0, -r * s], [0.0, 1.0, r * c]])
        g_z = np.array([[c, -r * s], [s, r * c]])

        n = len(self.x)
        p_lx = g_pose @ self.P[:3, :]
        p_ll = g_pose @ self.P[:3, :3] @ g_pose.T + g_z @ self.rm @ g_z.T
        big = np.zeros((n + 2, n + 2))
        big[:n, :n] = self.P
        big[n:, :n] = p_lx
        big[:n, n:] = p_lx.T
        big[n:, n:] = p_ll
        self.P = big
        self.x = np.concatenate([self.x, lm])

        j = self.n_landmarks - 1
        self.origin.append(true_id)
        self._id_to_idx[true_id] = j
        return j


    def observe(self, scan: list[tuple[int, np.ndarray]]) -> None:
        n_old = self.n_landmarks
        taken: set[int] = set()
        for true_id, z in scan:
            if self.association == "known":
                j = self._id_to_idx.get(true_id)
                if j is None:
                    self._initialise(z, true_id)
                else:
                    self._update(j, z)
                    self.n_associated += 1
                continue

            best_j, best_d2 = None, float("inf")
            for j in range(n_old):
                if j in taken:
                    continue
                d2 = self._mahalanobis2(j, z)
                if d2 < best_d2:
                    best_j, best_d2 = j, d2

            if best_j is None or best_d2 > self.new_gate:
                self._initialise(z, true_id)
            elif best_d2 <= self.gate:
                self._update(best_j, z)
                taken.add(best_j)
                self.n_associated += 1
                self.n_wrong += self.origin[best_j] != true_id
            else:
                self.n_ambiguous += 1


    def map_errors(self, true_landmarks: np.ndarray) -> np.ndarray:
        if not self.origin:
            return np.zeros(0)
        truth = np.asarray(true_landmarks)[self.origin]
        return np.linalg.norm(self.landmarks - truth, axis=1)

    def n_duplicates(self) -> int:
        return len(self.origin) - len(set(self.origin))
