from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from safegait.physics_g1 import (
    DEFAULT_XML,
    ArmEnv,
    ArmPolicy,
    NoisyArmPolicy,
    random_arm_episode,
    run_arm_episode,
)

CONDITIONS = {
    "nominal (70% gravity comp, kp 60/60/40/40)": dict(),
    "degraded (30% gravity comp, half the gains)": dict(gravity_comp=0.3, kp=(30.0, 30.0, 20.0, 20.0)),
}
VALIDATORS = (("off", False, 0.0), ("on, margin 0 cm", True, 0.0), ("on, margin 3 cm", True, 0.03))


def run(n: int, seed: int, xml: Path, n_obstacles: int) -> str:
    out = []
    for cond, kw in CONDITIONS.items():
        env = ArmEnv(xml, **kw)
        rows, all_gaps = [], []
        for noisy in (False, True):
            for vlabel, use_val, margin in VALIDATORS:
                runs = []
                for ep in range(n):
                    obstacles, target = random_arm_episode(env, np.random.default_rng(seed + ep), n_obstacles)
                    pol = ArmPolicy(env)
                    if noisy:
                        pol = NoisyArmPolicy(pol, np.random.default_rng(10_000 + seed + ep))
                    runs.append(run_arm_episode(env, pol, obstacles, target, use_val, margin))
                f = lambda a: sum(getattr(r, a) for r in runs) / n
                gaps = np.concatenate([r.hand_gap for r in runs if r.hand_gap])
                if use_val and margin == 0.03:
                    all_gaps.append(gaps)
                rows.append(
                    f"| {'noisy' if noisy else 'clean'} | {vlabel} | {f('success'):.0%} | {f('collided'):.1%} | "
                    f"{f('limit_violation'):.0%} | {f('stalled'):.0%} | {100 * gaps.mean():.2f} / {100 * np.percentile(gaps, 95):.2f} |"
                )
        out.append(
            f"**{cond}** ({n} episodes per row, {n_obstacles} obstacles)\n\n"
            "| policy | validator | success | collision (physics) | limit violation | stalled | prediction gap, hand (cm): mean / p95 |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(rows)
        )
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=2)
    ap.add_argument("--model", type=Path, default=DEFAULT_XML)
    args = ap.parse_args()
    print(run(args.episodes, args.seed, args.model, args.obstacles))


if __name__ == "__main__":
    main()
