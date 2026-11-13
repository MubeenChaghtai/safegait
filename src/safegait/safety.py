from __future__ import annotations

import numpy as np

from safegait.kinematics import RobotSpec, arm_points
from safegait.scene import Scene


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0.0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def clearance(spec: RobotSpec, scene: Scene, s: np.ndarray) -> float:
    if len(scene.obstacles) == 0:
        return float("inf")
    pts = arm_points(spec, s)
    best = float("inf")
    for cx, cy, r in scene.obstacles:
        c = np.array([cx, cy])
        best = min(best, float(np.linalg.norm(s[:2] - c)) - r - spec.base_radius)
        for i in range(len(pts) - 1):
            best = min(best, _point_segment_distance(c, pts[i], pts[i + 1]) - r - spec.arm_radius)
    return best


def step_violations(
    spec: RobotSpec,
    scene: Scene,
    s_prev: np.ndarray,
    s_next: np.ndarray,
    u: np.ndarray,
    margin: float,
) -> list[str]:
    kinds: list[str] = []
    if np.any(np.abs(u) > spec.cmd_limits() + 1e-9):
        kinds.append("velocity")
    if np.any(s_next[3:] < np.asarray(spec.q_lo)) or np.any(s_next[3:] > np.asarray(spec.q_hi)):
        kinds.append("joint_limit")
    c_next = clearance(spec, scene, s_next)
    if c_next < margin and c_next < clearance(spec, scene, s_prev):
        kinds.append("collision")
    return kinds
