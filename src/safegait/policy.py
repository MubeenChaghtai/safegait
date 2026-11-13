from __future__ import annotations

from typing import Protocol

import numpy as np

from safegait.controller import WholeBodyController
from safegait.kinematics import RobotSpec, step


class Policy(Protocol):
    def propose(self, s: np.ndarray, target: np.ndarray) -> np.ndarray: ...


class JacobianPolicy:
    def __init__(self, spec: RobotSpec, horizon: int = 10, dt: float = 0.05):
        self.spec, self.horizon, self.dt = spec, horizon, dt
        self.ctrl = WholeBodyController(spec)

    def propose(self, s: np.ndarray, target: np.ndarray) -> np.ndarray:
        chunk = np.zeros((self.horizon, 5))
        cur = s.copy()
        for k in range(self.horizon):
            chunk[k] = self.ctrl.command(cur, target)
            cur = step(self.spec, cur, chunk[k], self.dt)
        return chunk


class NoisyPolicy:
    def __init__(
        self,
        base: Policy,
        spec: RobotSpec,
        rng: np.random.Generator,
        sigma: float = 0.25,
        p_glitch: float = 0.15,
    ):
        self.base, self.spec, self.rng = base, spec, rng
        self.sigma, self.p_glitch = sigma, p_glitch

    def propose(self, s: np.ndarray, target: np.ndarray) -> np.ndarray:
        chunk = self.base.propose(s, target)
        limits = self.spec.cmd_limits()
        chunk = chunk + self.rng.normal(0.0, self.sigma, chunk.shape) * limits * 0.3
        if self.rng.random() < self.p_glitch:
            k0 = int(self.rng.integers(0, len(chunk)))
            channel = int(self.rng.integers(0, chunk.shape[1]))
            chunk[k0:, channel] += self.rng.choice([-1.0, 1.0]) * 2.5 * limits[channel]
        return chunk


class SaturatedPolicy:
    def __init__(self, base: Policy, spec: RobotSpec):
        self.base, self.spec = base, spec

    def propose(self, s: np.ndarray, target: np.ndarray) -> np.ndarray:
        lim = self.spec.cmd_limits()
        return np.clip(self.base.propose(s, target), -lim, lim)
