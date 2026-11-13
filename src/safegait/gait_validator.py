from __future__ import annotations

import numpy as np

from safegait.gait import GaitSpec, WalkerState, execute_step, plan_steps, step_commands, zmp_excess
from safegait.kinematics import RobotSpec
from safegait.safety import step_violations
from safegait.scene import Scene


class GaitValidator:
    def __init__(
        self,
        spec: RobotSpec,
        gspec: GaitSpec,
        scene: Scene,
        margin: float = 0.05,
        zmp_margin: float = 0.02,
        stop_steps: int = 2,
    ):
        self.spec, self.gspec, self.scene = spec, gspec, scene
        self.margin, self.zmp_margin, self.stop_steps = margin, zmp_margin, stop_steps

    def first_violation(
        self, walker: WalkerState, q: np.ndarray, chunk: np.ndarray, m: int | None = None
    ) -> tuple[int, str] | None:
        gs, t_ticks = self.gspec, self.gspec.ticks_per_step
        cmds = step_commands(chunk, t_ticks)
        m = len(cmds) if m is None else m
        plan_cmds = np.vstack([cmds[:m], np.zeros((self.stop_steps, 2))])

        sim, q = walker.copy(), np.asarray(q, dtype=float).copy()
        s_prev = np.concatenate([sim.base_pose(), q])
        for j in range(m + self.stop_steps):
            plan = plan_steps(gs, sim, plan_cmds[j:])
            if j == 0 and zmp_excess(gs, plan.feet[0], plan.z0, self.zmp_margin) > 0.0:
                return j, "balance"
            if j < m and plan.clamped[0]:
                return j, "step_limit"
            sim_next, trace = execute_step(gs, sim, plan)
            for i in range(t_ticks):
                u = chunk[j * t_ticks + i].copy() if j < m else np.zeros(5)
                u[:2] = cmds[j] if j < m else 0.0
                q = q + u[2:] * gs.dt
                s_next = np.concatenate([trace.base[i], q])
                kinds = step_violations(self.spec, self.scene, s_prev, s_next, u, self.margin)
                if kinds:
                    return j, kinds[0]
                s_prev = s_next
            sim = sim_next
        return None

    def safe_steps(self, walker: WalkerState, q: np.ndarray, chunk: np.ndarray) -> int:
        m = chunk.shape[0] // self.gspec.ticks_per_step
        while m > 0:
            bad = self.first_violation(walker, q, chunk, m)
            if bad is None:
                return m
            m = min(m - 1, bad[0])
        return 0
