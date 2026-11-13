from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safegait.commands import Task, parse_command
from safegait.kinematics import RobotSpec, ee_position
from safegait.planner import PlannedPolicy
from safegait.policy import JacobianPolicy, SaturatedPolicy
from safegait.scene import Scene
from safegait.validator import ChunkValidator

POLICIES = ("planned", "controller", "learned")


@dataclass
class Decision:
    u: np.ndarray
    status: str
    chunk: np.ndarray | None = None
    n_safe: int = 0
    violations: list[str] = field(default_factory=list)
    path: list[np.ndarray] | None = None


class SafeGaitController:
    def __init__(self, scene: Scene, spec: RobotSpec | None = None, policy: str = "planned",
                 use_validator: bool = True, model_path: str = "", dt: float = 0.05, exec_n: int = 5,
                 tol: float = 0.05, max_consecutive_blocks: int = 6):
        assert policy in POLICIES, policy
        self.scene, self.spec = scene, spec or RobotSpec()
        self.policy_kind, self.use_validator, self.model_path = policy, use_validator, model_path
        self.dt, self.exec_n, self.tol, self.max_blocks = dt, exec_n, tol, max_consecutive_blocks
        self.task: Task | None = None
        self.target: np.ndarray | None = None
        self._policy = None
        self._queue: list[np.ndarray] = []
        self._blocks = 0
        self.last = Decision(np.zeros(5), "idle")

    def _make_policy(self):
        if self.policy_kind == "planned":
            return PlannedPolicy(self.spec, self.scene)
        if self.policy_kind == "controller":
            return JacobianPolicy(self.spec)
        from safegait.learned import MLP, LearnedPolicy

        return SaturatedPolicy(LearnedPolicy(self.spec, MLP.load(self.model_path)), self.spec)

    def set_command(self, text: str) -> Task | None:
        task = parse_command(text, list(self.scene.objects))
        if task is None:
            return None
        self.task, self.target = task, np.asarray(self.scene.objects[task.target], dtype=float)
        self._policy = self._make_policy()
        self._queue, self._blocks = [], 0
        self.last = Decision(np.zeros(5), "moving")
        return task

    def step(self, s: np.ndarray) -> Decision:
        if self.target is None or self._policy is None:
            self.last = Decision(np.zeros(5), "idle")
            return self.last
        if np.linalg.norm(ee_position(self.spec, s) - self.target) < self.tol:
            self.last = Decision(np.zeros(5), "arrived")
            self._queue = []
            return self.last
        if self._blocks >= self.max_blocks:
            self.last = Decision(np.zeros(5), "stalled", self.last.chunk, 0, self.last.violations)
            return self.last
        if not self._queue:
            chunk = self._policy.propose(s, self.target)
            validator = ChunkValidator(self.spec, self.scene, self.dt)
            kinds = sorted({v.kind for v in validator.check(s, chunk)})
            n = min(validator.safe_prefix(s, chunk), self.exec_n) if self.use_validator else self.exec_n
            path = getattr(self._policy, "_path", None)
            if n == 0:
                self._blocks += 1
                self.last = Decision(np.zeros(5), "blocked", chunk, 0, kinds, path)
                return self.last
            self._blocks = 0
            self._queue = [u for u in chunk[:n]]
            self.last = Decision(self._queue[0], "moving", chunk, n, kinds, path)
        u = self._queue.pop(0)
        self.last = Decision(u, "moving", self.last.chunk, self.last.n_safe, self.last.violations, self.last.path)
        return self.last
