from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safegait.kinematics import RobotSpec, ee_position, step
from safegait.policy import Policy
from safegait.safety import step_violations
from safegait.scene import Scene
from safegait.validator import ChunkValidator


@dataclass
class EpisodeResult:
    success: bool = False
    collided: bool = False
    limit_violation: bool = False
    stalled: bool = False
    steps: int = 0
    blocked_chunks: int = 0
    trace: list[np.ndarray] = field(default_factory=list)


def run_episode(
    policy: Policy,
    spec: RobotSpec,
    scene: Scene,
    s0: np.ndarray,
    target: np.ndarray,
    use_validator: bool,
    dt: float = 0.05,
    exec_n: int = 5,
    max_steps: int = 600,
    tol: float = 0.05,
    max_consecutive_blocks: int = 6,
    record: bool = False,
) -> EpisodeResult:
    validator = ChunkValidator(spec, scene, dt)
    res = EpisodeResult()
    s = s0.copy()
    consecutive = 0
    if record:
        res.trace.append(s.copy())

    while res.steps < max_steps:
        chunk = policy.propose(s, target)
        n = min(validator.safe_prefix(s, chunk), exec_n) if use_validator else exec_n

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
            nxt = step(spec, s, u, dt)
            kinds = step_violations(spec, scene, s, nxt, u, margin=0.0)
            if "velocity" in kinds or "joint_limit" in kinds:
                res.limit_violation = True
            if "collision" in kinds:
                res.collided = True
            s = nxt
            res.steps += 1
            if record:
                res.trace.append(s.copy())
            if res.collided:
                return res
            if np.linalg.norm(ee_position(spec, s) - target) < tol:
                res.success = True
                return res
    return res
