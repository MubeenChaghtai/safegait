from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from safegait.kinematics import RobotSpec


EXTRA_OBJECT_NAMES = ("blue_cube", "green_box", "yellow_ball")


@dataclass
class Scene:
    obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    objects: dict[str, np.ndarray] = field(default_factory=dict)
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def random_episode(
    rng: np.random.Generator,
    spec: RobotSpec,
    n_obstacles: int = 3,
    n_landmarks: int = 0,
    n_extra_objects: int = 0,
    q_home: tuple[float, ...] = (0.5, -0.9, 0.5),
) -> tuple[Scene, np.ndarray, np.ndarray]:
    heading = rng.uniform(-0.8, 0.8)
    s0 = np.array([0.0, 0.0, heading, *q_home])
    bearing = rng.uniform(-1.2, 1.2)
    dist = rng.uniform(1.5, 3.0)
    target = dist * np.array([np.cos(bearing), np.sin(bearing)])

    obstacles: list[list[float]] = []
    tries = 0
    while len(obstacles) < n_obstacles and tries < 500:
        tries += 1
        c = rng.uniform([-0.5, -2.0], [3.5, 2.0])
        r = rng.uniform(0.2, 0.4)
        if np.linalg.norm(c) < r + spec.base_radius + 0.3:
            continue
        if np.linalg.norm(c - target) < r + 0.5:
            continue
        obstacles.append([c[0], c[1], r])

    landmarks = rng.uniform([-0.5, -2.5], [3.5, 2.5], size=(n_landmarks, 2))
    objects = {"red_cube": target}
    for name in EXTRA_OBJECT_NAMES[:n_extra_objects]:
        for _ in range(200):
            b, d = rng.uniform(-1.2, 1.2), rng.uniform(1.5, 3.0)
            p = d * np.array([np.cos(b), np.sin(b)])
            if all(np.linalg.norm(p - o[:2]) >= o[2] + 0.5 for o in obstacles):
                objects[name] = p
                break
    scene = Scene(np.array(obstacles).reshape(-1, 3), objects, landmarks)
    return scene, s0, target
