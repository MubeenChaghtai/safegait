from __future__ import annotations

import argparse

import numpy as np

from safegait.gait import GaitSpec, humanoid_spec
from safegait.gait_runner import run_gait_episode, start_is_feasible
from safegait.kinematics import RobotSpec
from safegait.planner import PlannedPolicy
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.runner import run_episode
from safegait.scene import random_episode

HEAD = (
    "| policy | validator | success | collision | limit violation | stalled | blocked chunks/ep |\n"
    "|---|---|---|---|---|---|---|\n"
)


def _policy(kind: str, noisy: bool, spec: RobotSpec, scene, horizon: int, seed: int):
    base = PlannedPolicy(spec, scene, horizon=horizon) if kind == "planned" else JacobianPolicy(spec, horizon=horizon)
    return NoisyPolicy(base, spec, np.random.default_rng(10_000 + seed)) if noisy else base


def _row(label: str, validator: bool, runs: list) -> str:
    n = len(runs)
    f = lambda attr: sum(getattr(r, attr) for r in runs) / n
    return (
        f"| {label} | {'on' if validator else 'off'} | {f('success'):.0%} | {f('collided'):.0%} | "
        f"{f('limit_violation'):.0%} | {f('stalled'):.0%} | {f('blocked_chunks'):.1f} |"
    )


def wheeled(n: int, seed: int, n_obstacles: int) -> str:
    spec, rows = RobotSpec(), []
    for noisy in (False, True):
        for kind in ("straight", "planned"):
            for use_val in (False, True):
                runs = []
                for ep in range(n):
                    scene, s0, target = random_episode(np.random.default_rng(seed + ep), spec, n_obstacles)
                    pol = _policy(kind, noisy, spec, scene, 10, seed + ep)
                    runs.append(run_episode(pol, spec, scene, s0, target, use_val))
                rows.append(_row(f"{'noisy ' if noisy else ''}{kind}", use_val, runs))
    return HEAD + "\n".join(rows)


def walker(n: int, seed: int, n_obstacles: int) -> tuple[str, int]:
    spec, gs, rows = humanoid_spec(), GaitSpec(), []
    cases, ep, skipped = [], 0, 0
    while len(cases) < n:
        scene, s0, target = random_episode(np.random.default_rng(seed + ep), spec, n_obstacles, 0)
        if start_is_feasible(spec, gs, scene, s0):
            cases.append((seed + ep, scene, s0, target))
        else:
            skipped += 1
        ep += 1
    for noisy in (False, True):
        for kind in ("straight", "planned"):
            for use_val in (False, True):
                runs = []
                for ep_seed, scene, s0, target in cases:
                    pol = _policy(kind, noisy, spec, scene, 3 * gs.ticks_per_step, ep_seed)
                    runs.append(run_gait_episode(pol, spec, gs, scene, s0, target, use_val, np.random.default_rng(ep_seed)))
                rows.append(_row(f"{'noisy ' if noisy else ''}{kind}", use_val, runs))
    return HEAD + "\n".join(rows), skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=3)
    ap.add_argument("--only", choices=["wheeled", "walker"])
    args = ap.parse_args()
    if args.only != "walker":
        print(f"### Wheeled base ({args.episodes} episodes, {args.obstacles} obstacles)\n")
        print(wheeled(args.episodes, args.seed, args.obstacles), "\n")
    if args.only != "wheeled":
        table, skipped = walker(args.episodes, args.seed, args.obstacles)
        print(f"### Walking base ({args.episodes} episodes, {args.obstacles} obstacles; {skipped} infeasible starts skipped)\n")
        print(table)


if __name__ == "__main__":
    main()
