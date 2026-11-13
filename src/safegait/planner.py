from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from safegait.controller import WholeBodyController
from safegait.kinematics import RobotSpec, step
from safegait.safety import clearance
from safegait.scene import Scene

Q_TRAVEL = (-2.5, -2.5, -2.5)

_NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class GridPlanner:
    def __init__(
        self,
        scene: Scene,
        spec: RobotSpec,
        resolution: float = 0.1,
        clearance: float = 0.08,
        bounds: tuple[float, float, float, float] = (-1.5, -3.5, 4.5, 3.5),
    ):
        self.res = resolution
        self.x0, self.y0, x1, y1 = bounds
        self.nx = int(np.ceil((x1 - self.x0) / resolution))
        self.ny = int(np.ceil((y1 - self.y0) / resolution))
        xs = self.x0 + (np.arange(self.nx) + 0.5) * resolution
        ys = self.y0 + (np.arange(self.ny) + 0.5) * resolution
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.centres = np.stack([gx, gy], axis=-1)
        self.occupied = np.zeros((self.nx, self.ny), dtype=bool)
        for cx, cy, r in scene.obstacles:
            d = np.hypot(gx - cx, gy - cy)
            self.occupied |= d < r + spec.base_radius + clearance

    def cell(self, p: np.ndarray) -> tuple[int, int]:
        i = int((p[0] - self.x0) / self.res)
        j = int((p[1] - self.y0) / self.res)
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def point(self, c: tuple[int, int]) -> np.ndarray:
        return self.centres[c]

    def _free(self, c: tuple[int, int], free_near: np.ndarray, free_radius: float) -> bool:
        if not self.occupied[c]:
            return True
        return bool(np.linalg.norm(self.centres[c] - free_near) < free_radius)

    def line_free(self, a: np.ndarray, b: np.ndarray, free_near: np.ndarray, free_radius: float) -> bool:
        n = max(2, int(np.linalg.norm(b - a) / (0.5 * self.res)))
        return all(self._free(self.cell(a + (b - a) * t), free_near, free_radius) for t in np.linspace(0, 1, n))

    def plan(
        self, start: np.ndarray, target: np.ndarray, reach_lo: float = 0.4, reach_hi: float = 0.7
    ) -> list[np.ndarray] | None:
        free_radius = 0.45
        s = self.cell(start)

        def is_goal(c: tuple[int, int]) -> bool:
            d = float(np.linalg.norm(self.centres[c] - target))
            return reach_lo <= d <= reach_hi and not self.occupied[c]

        def h(c: tuple[int, int]) -> float:
            return max(0.0, float(np.linalg.norm(self.centres[c] - target)) - reach_hi)

        open_heap = [(h(s), 0.0, s)]
        g = {s: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        closed: set[tuple[int, int]] = set()
        while open_heap:
            _, gc, c = heapq.heappop(open_heap)
            if c in closed:
                continue
            closed.add(c)
            if is_goal(c):
                path = [self.point(c)]
                while c in parent:
                    c = parent[c]
                    path.append(self.point(c))
                return self._smooth(path[::-1], start, free_radius)
            for di, dj in _NEIGHBOURS:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or n in closed:
                    continue
                if not self._free(n, start, free_radius):
                    continue
                if di and dj and (self.occupied[c[0] + di, c[1]] or self.occupied[c[0], c[1] + dj]):
                    continue
                ng = gc + self.res * (1.41421356 if di and dj else 1.0)
                if ng < g.get(n, float("inf")):
                    g[n] = ng
                    parent[n] = c
                    heapq.heappush(open_heap, (ng + h(n), ng, n))
        return None

    def _smooth(self, path: list[np.ndarray], start: np.ndarray, free_radius: float) -> list[np.ndarray]:
        pts = [np.asarray(start, dtype=float)] + [p for p in path[1:]]
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j], start, free_radius):
                j -= 1
            out.append(pts[j])
            i = j
        return out


@dataclass
class PlannedPolicy:
    spec: RobotSpec
    scene: Scene
    horizon: int = 10
    dt: float = 0.05
    lookahead: float = 0.5
    reach_radius: float = 0.72
    replan_distance: float = 0.3
    min_base_distance: float = 0.45
    unfold_distance: float = 0.0
    q_reach: tuple[float, float, float] = (0.5, -0.9, 0.5)
    cruise: float = 0.7

    def __post_init__(self) -> None:
        self.planner = GridPlanner(self.scene, self.spec)
        self.ctrl = WholeBodyController(self.spec)
        self._path: list[np.ndarray] | None = None
        self._path_target: np.ndarray | None = None
        self.replans = 0
        self.failed_plans = 0
        self._fold_sign: float | None = None
        self._reaching = False

    def _distance_to_path(self, p: np.ndarray) -> float:
        assert self._path is not None
        best = float("inf")
        for a, b in zip(self._path[:-1], self._path[1:]):
            ab = b - a
            t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0, 1))
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        return best

    def _ensure_path(self, p: np.ndarray, target: np.ndarray) -> bool:
        stale = (
            self._path is None
            or self._path_target is None
            or np.linalg.norm(self._path_target - target) > 1e-9
            or self._distance_to_path(p) > self.replan_distance
        )
        if stale:
            self._path = self.planner.plan(p, target)
            self._path_target = target.copy()
            self.replans += 1
            if self._path is None:
                self.failed_plans += 1
        return self._path is not None

    def _lookahead_point(self, p: np.ndarray) -> np.ndarray:
        assert self._path is not None
        best_i, best_t, best_d = 0, 0.0, float("inf")
        for i, (a, b) in enumerate(zip(self._path[:-1], self._path[1:])):
            ab = b - a
            t = float(np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0, 1))
            d = float(np.linalg.norm(p - (a + t * ab)))
            if d < best_d:
                best_i, best_t, best_d = i, t, d
        remaining = self.lookahead
        a = self._path[best_i] + best_t * (self._path[best_i + 1] - self._path[best_i])
        for k in range(best_i, len(self._path) - 1):
            seg_end = self._path[k + 1]
            seg = float(np.linalg.norm(seg_end - a))
            if seg >= remaining:
                return a + (seg_end - a) * remaining / max(seg, 1e-12)
            remaining -= seg
            a = seg_end
        return self._path[-1]

    def propose(self, s: np.ndarray, target: np.ndarray) -> np.ndarray:
        d = float(np.linalg.norm(s[:2] - target))
        if d <= self.reach_radius:
            self._reaching = True
        elif d > self.reach_radius + 0.5:
            self._reaching = False
        if self._reaching or not self._ensure_path(s[:2], target):
            return self._reach_chunk(s, target)
        return self._travel_chunk(s, target)

    def _reach_chunk(self, s: np.ndarray, target: np.ndarray) -> np.ndarray:
        chunk = np.zeros((self.horizon, 5))
        cur = s.copy()
        sign = -1.0 if self._fold_sign is None else self._fold_sign
        self.ctrl.q_home = tuple(-sign * np.asarray(self.q_reach))
        lo, hi = np.asarray(self.spec.q_lo) + 0.02, np.asarray(self.spec.q_hi) - 0.02
        for k in range(self.horizon):
            u = self.ctrl.command(cur, target)
            u[2:] = np.clip(u[2:], (lo - cur[3:]) / self.dt, (hi - cur[3:]) / self.dt)
            if np.linalg.norm(cur[:2] - target) < self.min_base_distance:
                u[0] = min(u[0], 0.0)
            chunk[k] = u
            cur = step(self.spec, cur, chunk[k], self.dt)
        return chunk

    def _choose_fold(self, s: np.ndarray) -> float:
        best_sign, best_gap = -1.0, -float("inf")
        lim = self.spec.cmd_limits()
        for sign in (-1.0, 1.0):
            goal = sign * np.abs(np.asarray(Q_TRAVEL))
            cur = s.copy()
            gap = clearance(self.spec, self.scene, cur)
            for _ in range(40):
                qd = np.clip(2.0 * (goal - cur[3:]), -lim[2:], lim[2:])
                cur = step(self.spec, cur, np.array([0.0, 0.0, *qd]), self.dt)
                gap = min(gap, clearance(self.spec, self.scene, cur))
            if gap > best_gap:
                best_sign, best_gap = sign, gap
        return best_sign

    def _travel_chunk(self, s: np.ndarray, target: np.ndarray) -> np.ndarray:
        chunk = np.zeros((self.horizon, 5))
        cur = s.copy()
        goal = self._path[-1]
        lim = self.spec.cmd_limits()
        if self._fold_sign is None:
            self._fold_sign = self._choose_fold(s)
        q_fold = self._fold_sign * np.abs(np.asarray(Q_TRAVEL))
        for k in range(self.horizon):
            la = self._lookahead_point(cur[:2])
            d = la - cur[:2]
            alpha = _wrap(float(np.arctan2(d[1], d[0])) - cur[2])
            w = float(np.clip(2.5 * alpha, -self.cruise * self.spec.w_max, self.cruise * self.spec.w_max))
            v = self.cruise * self.spec.v_max * float(np.clip(1.0 - abs(alpha) / 1.0, 0.0, 1.0))
            v *= float(np.clip(np.linalg.norm(goal - cur[:2]) / 0.5, 0.15, 1.0))
            folded = np.linalg.norm(cur[:2] - target) > self.unfold_distance
            q_goal = q_fold if folded else -self._fold_sign * np.asarray(self.q_reach)
            qd = np.clip(2.0 * (q_goal - cur[3:]), -self.cruise * lim[2:], self.cruise * lim[2:])
            if float(np.max(np.abs(q_fold - cur[3:]))) > 1.0 and folded:
                v = 0.0
            chunk[k] = [v, w, *qd]
            cur = step(self.spec, cur, chunk[k], self.dt)
        return chunk
