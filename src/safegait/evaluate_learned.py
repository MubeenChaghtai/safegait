from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from safegait.commands import parse_command
from safegait.kinematics import RobotSpec
from safegait.learned import MLP, LearnedPolicy
from safegait.policy import JacobianPolicy, SaturatedPolicy
from safegait.runner import run_episode
from safegait.scene import random_episode

HEAD = (
    "| policy | validator | success | collision | limit violation | stalled | blocked chunks/ep |\n"
    "|---|---|---|---|---|---|---|\n"
)


def make_episode(seed: int, spec: RobotSpec, n_obstacles: int):
    rng = np.random.default_rng(seed)
    scene, s0, _ = random_episode(rng, spec, n_obstacles, n_extra_objects=2)
    name = list(scene.objects)[int(np.random.default_rng(90_000 + seed).integers(len(scene.objects)))]
    command = f"Please pick up the {name.replace('_', ' ')}"
    task = parse_command(command, list(scene.objects))
    assert task is not None and task.target == name, (command, task)
    return scene, s0, scene.objects[task.target], command


def evaluate(n: int, seed: int, n_obstacles: int, models: dict[str, Path]) -> str:
    spec = RobotSpec()
    policies = {"controller (expert)": JacobianPolicy(spec)}
    for label, p in models.items():
        learned = LearnedPolicy(spec, MLP.load(p))
        policies[label] = learned
        policies[f"{label} + saturation"] = SaturatedPolicy(learned, spec)
    rows = []
    for label, pol in policies.items():
        for use_val in (False, True):
            runs = []
            for ep in range(n):
                scene, s0, target, _ = make_episode(seed + ep, spec, n_obstacles)
                runs.append(run_episode(pol, spec, scene, s0, target, use_val))
            f = lambda a: sum(getattr(r, a) for r in runs) / n
            rows.append(
                f"| {label} | {'on' if use_val else 'off'} | {f('success'):.0%} | {f('collided'):.0%} | "
                f"{f('limit_violation'):.0%} | {f('stalled'):.0%} | {f('blocked_chunks'):.1f} |"
            )
    return HEAD + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=3)
    ap.add_argument("--model", action="append", default=None, help="label=path, repeatable")
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in (args.model or ["learned (4000 demos)=models/bc_policy.npz",
                                                          "learned (100 demos)=models/bc_small.npz"]))
    print(f"### Command-conditioned reaching ({args.episodes} episodes, {args.obstacles} obstacles, 3 objects)\n")
    print(evaluate(args.episodes, args.seed, args.obstacles, {k: Path(v) for k, v in models.items()}))


if __name__ == "__main__":
    main()
