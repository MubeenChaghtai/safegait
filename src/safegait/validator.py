from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safegait.kinematics import RobotSpec, step
from safegait.safety import step_violations
from safegait.scene import Scene


@dataclass(frozen=True)
class Violation:
    step: int
    kind: str


class ChunkValidator:
    def __init__(self, spec: RobotSpec, scene: Scene, dt: float, margin: float = 0.05):
        self.spec, self.scene, self.dt, self.margin = spec, scene, dt, margin

    def check(self, s: np.ndarray, chunk: np.ndarray) -> list[Violation]:
        found: list[Violation] = []
        cur = s
        for k, u in enumerate(chunk):
            nxt = step(self.spec, cur, u, self.dt)
            found += [
                Violation(k, kind)
                for kind in step_violations(self.spec, self.scene, cur, nxt, u, self.margin)
            ]
            cur = nxt
        return found

    def safe_prefix(self, s: np.ndarray, chunk: np.ndarray) -> int:
        cur = s
        for k, u in enumerate(chunk):
            nxt = step(self.spec, cur, u, self.dt)
            if step_violations(self.spec, self.scene, cur, nxt, u, self.margin):
                return k
            cur = nxt
        return len(chunk)
