from __future__ import annotations

import argparse

import numpy as np

from safegait.kinematics import RobotSpec
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.runner import run_episode
from safegait.scene import random_episode


def evaluate(n_episodes: int, seed: int, n_obstacles: int) -> list[dict]:
    spec = RobotSpec()
    rows = []
    for policy_name in ("clean", "noisy"):
        for use_validator in (False, True):
            stats = dict(success=0, collided=0, limit=0, stalled=0, blocked=0)
            for ep in range(n_episodes):
                scene, s0, target = random_episode(np.random.default_rng(seed + ep), spec, n_obstacles)
                policy = JacobianPolicy(spec)
                if policy_name == "noisy":
                    policy = NoisyPolicy(policy, spec, np.random.default_rng(10_000 + seed + ep))
                r = run_episode(policy, spec, scene, s0, target, use_validator)
                stats["success"] += r.success
                stats["collided"] += r.collided
                stats["limit"] += r.limit_violation
                stats["stalled"] += r.stalled
                stats["blocked"] += r.blocked_chunks
            rows.append(
                dict(
                    policy=policy_name,
                    validator="on" if use_validator else "off",
                    **{k: v / n_episodes for k, v in stats.items() if k != "blocked"},
                    blocked_per_episode=stats["blocked"] / n_episodes,
                )
            )
    return rows


def to_markdown(rows: list[dict]) -> str:
    head = "| policy | validator | success | collision | limit violation | stalled | blocked chunks/ep |\n"
    head += "|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r['policy']} | {r['validator']} | {r['success']:.0%} | {r['collided']:.0%} | "
        f"{r['limit']:.0%} | {r['stalled']:.0%} | {r['blocked_per_episode']:.1f} |\n"
        for r in rows
    )
    return head + body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=3)
    args = ap.parse_args()
    print(to_markdown(evaluate(args.episodes, args.seed, args.obstacles)))


if __name__ == "__main__":
    main()
