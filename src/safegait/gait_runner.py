from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safegait.gait import (
    GaitSpec,
    StepTrace,
    WalkerState,
    execute_step,
    init_walker,
    plan_steps,
    step_commands,
)
from safegait.gait_validator import GaitValidator
from safegait.kinematics import RobotSpec, ee_position
from safegait.policy import Policy
from safegait.safety import clearance, step_violations
from safegait.scene import Scene

STOP_STEPS = 2


@dataclass
class GaitResult:
    success: bool = False
    collided: bool = False
    fell: bool = False
    limit_violation: bool = False
    stalled: bool = False
    ticks: int = 0
    blocked_chunks: int = 0
    pushes: int = 0
    saturated_ticks: int = 0
    final_error: float = float("inf")
    traces: list[StepTrace] = field(default_factory=list)
    base_path: list[np.ndarray] = field(default_factory=list)
    ee_path: list[np.ndarray] = field(default_factory=list)


def start_is_feasible(spec: RobotSpec, gspec: GaitSpec, scene: Scene, s0: np.ndarray, steps: int = 6) -> bool:
    q = np.asarray(s0[3:], dtype=float)
    if clearance(spec, scene, s0) < 0.0:
        return False
    w = init_walker(gspec, s0[:3])
    for _ in range(steps):
        w, trace = execute_step(gspec, w, plan_steps(gspec, w, np.zeros((STOP_STEPS, 2))))
        if any(clearance(spec, scene, np.concatenate([b, q])) < 0.0 for b in trace.base):
            return False
    return True


def run_gait_episode(
    policy: Policy,
    spec: RobotSpec,
    gspec: GaitSpec,
    scene: Scene,
    s0: np.ndarray,
    target: np.ndarray,
    use_validator: bool,
    rng: np.random.Generator,
    zmp_margin: float = 0.02,
    margin: float = 0.05,
    validator_stop_steps: int = STOP_STEPS,
    use_backup: bool = True,
    zmp_noise: float = 0.005,
    push_prob: float = 0.0,
    push_max: float = 0.0,
    max_ticks: int = 600,
    tol: float = 0.05,
    max_consecutive_blocks: int = 6,
    record: bool = False,
) -> GaitResult:
    t_ticks = gspec.ticks_per_step
    res = GaitResult()
    walker: WalkerState = init_walker(gspec, s0[:3])
    q = np.asarray(s0[3:], dtype=float).copy()
    validator = GaitValidator(spec, gspec, scene, margin, zmp_margin, validator_stop_steps)
    consecutive = 0
    backup: tuple[np.ndarray, np.ndarray] | None = None

    while res.ticks < max_ticks:
        s = np.concatenate([walker.base_pose(), q])
        chunk = policy.propose(s, target)
        n_total = chunk.shape[0] // t_ticks
        n = validator.safe_steps(walker, q, chunk) if use_validator else n_total
        if n == 0:
            res.blocked_chunks += 1
            consecutive += 1
        else:
            consecutive = 0

        if n > 0:
            plan_cmds = np.vstack([step_commands(chunk, t_ticks)[:n], np.zeros((STOP_STEPS, 2))])
            arm = np.vstack([chunk[: n * t_ticks], np.zeros((STOP_STEPS * t_ticks, 5))])
        elif use_backup and backup is not None:
            plan_cmds, arm = backup
        else:
            plan_cmds, arm = np.zeros((STOP_STEPS, 2)), np.zeros((STOP_STEPS * t_ticks, 5))
        plan = plan_steps(gspec, walker, plan_cmds)
        if plan.clamped[0]:
            res.limit_violation = True
        tail = plan_cmds[1:]
        backup = (
            (np.vstack([tail, np.zeros((max(0, STOP_STEPS - len(tail)), 2))]), np.vstack([arm[t_ticks:], np.zeros((t_ticks, 5))]))
            if len(plan_cmds) > 1
            else (np.zeros((STOP_STEPS, 2)), np.zeros((STOP_STEPS * t_ticks, 5)))
        )

        push = None
        if push_prob > 0.0 and rng.random() < push_prob:
            ang = rng.uniform(0.0, 2.0 * np.pi)
            push = (int(rng.integers(t_ticks)), rng.uniform(0.0, push_max) * np.array([np.cos(ang), np.sin(ang)]))
            res.pushes += 1
        new, trace = execute_step(gspec, walker, plan, rng, zmp_noise, push)
        res.saturated_ticks += trace.zmp_saturated
        if record:
            res.traces.append(trace)

        s_prev = s
        for i in range(t_ticks):
            u = arm[i].copy()
            u[:2] = plan_cmds[0]
            q = q + u[2:] * gspec.dt
            s_next = np.concatenate([trace.base[i], q])
            kinds = step_violations(spec, scene, s_prev, s_next, u, 0.0)
            res.limit_violation |= "velocity" in kinds or "joint_limit" in kinds
            res.collided |= "collision" in kinds
            s_prev = s_next
            res.ticks += 1
            if record:
                res.base_path.append(trace.base[i].copy())
                res.ee_path.append(ee_position(spec, s_next))
            if res.collided:
                res.final_error = float(np.linalg.norm(ee_position(spec, s_next) - target))
                return res
            if np.linalg.norm(ee_position(spec, s_next) - target) < tol:
                res.success = True
                res.final_error = float(np.linalg.norm(ee_position(spec, s_next) - target))
                return res
        if trace.fell:
            res.fell = True
            break
        walker = new
        if n == 0 and consecutive >= max_consecutive_blocks:
            res.stalled = True
            break

    res.final_error = float(np.linalg.norm(ee_position(spec, np.concatenate([walker.base_pose(), q])) - target))
    return res
