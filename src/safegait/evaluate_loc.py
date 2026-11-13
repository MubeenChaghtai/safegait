from __future__ import annotations

import argparse

import numpy as np

from safegait.kinematics import RobotSpec
from safegait.localisation import ActuationNoise, SensorSpec
from safegait.localised_runner import MODES, run_localised_episode
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.scene import random_episode

NEES_LO, NEES_HI = 0.216, 9.348


def experiment_a(n: int, seed: int, n_landmarks: int) -> str:
    spec = RobotSpec()
    rows = []
    for use_lm in (False, True):
        mean_err, final_err, nees_all, inside = [], [], [], []
        for ep in range(n):
            scene, s0, target = random_episode(
                np.random.default_rng(seed + ep), spec, n_obstacles=0, n_landmarks=n_landmarks
            )
            r = run_localised_episode(
                JacobianPolicy(spec), spec, scene, s0, target, "raw",
                np.random.default_rng(50_000 + seed + ep), use_landmarks=use_lm,
            )
            mean_err.append(np.mean(r.pos_err))
            final_err.append(r.pos_err[-1])
            nees_all += r.nees
        nees_arr = np.array(nees_all)
        rows.append(
            f"| {'EKF + landmarks' if use_lm else 'dead reckoning'} | {np.mean(mean_err)*100:.1f} | "
            f"{np.mean(final_err)*100:.1f} | {nees_arr.mean():.2f} | "
            f"{np.mean((nees_arr > NEES_LO) & (nees_arr < NEES_HI)):.0%} |"
        )
    head = (
        "| estimator | mean position error (cm) | final position error (cm) | mean NEES (ideal 3) | steps inside 95% NEES bound |\n"
        "|---|---|---|---|---|\n"
    )
    return head + "\n".join(rows)


def experiment_b(n: int, seed: int, n_obstacles: int, n_landmarks: int, policy_name: str,
                 sensor: SensorSpec = SensorSpec(), noise: ActuationNoise = ActuationNoise()) -> str:
    spec = RobotSpec()
    rows = []
    for mode in MODES:
        stats = dict(success=0, collided=0, limit=0, stalled=0, blocked=0, ee_err=0.0)
        for ep in range(n):
            scene, s0, target = random_episode(
                np.random.default_rng(seed + ep), spec, n_obstacles, n_landmarks
            )
            policy = JacobianPolicy(spec)
            if policy_name == "noisy":
                policy = NoisyPolicy(policy, spec, np.random.default_rng(10_000 + seed + ep))
            r = run_localised_episode(
                policy, spec, scene, s0, target, mode,
                np.random.default_rng(50_000 + seed + ep), sensor=sensor, noise=noise,
            )
            stats["success"] += r.success
            stats["collided"] += r.collided
            stats["limit"] += r.limit_violation
            stats["stalled"] += r.stalled
            stats["blocked"] += r.blocked_chunks
        rows.append(
            f"| {mode} | {stats['success']/n:.0%} | {stats['collided']/n:.0%} | "
            f"{stats['limit']/n:.0%} | {stats['stalled']/n:.0%} | {stats['blocked']/n:.1f} |"
        )
    head = (
        "| validator mode | success | collision | limit violation | stalled | blocked chunks/ep |\n"
        "|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=3)
    ap.add_argument("--landmarks", type=int, default=8)
    ap.add_argument("--policy", choices=["clean", "noisy"], default="noisy")
    ap.add_argument("--stress", action="store_true", help="few landmarks, short range, larger slip")
    args = ap.parse_args()
    if args.stress:
        sensor = SensorSpec(range_std=0.10, bearing_std=0.06, max_range=1.5)
        noise = ActuationNoise(v_scale_std=0.10, w_bias_std=0.08)
        args.landmarks = 3
    else:
        sensor, noise = SensorSpec(), ActuationNoise()
    if not args.stress:
        print("### A. Localisation quality (no obstacles)\n")
        print(experiment_a(args.episodes, args.seed, args.landmarks))
    print(f"\n### B. Validator on the estimated pose ({args.policy} policy, {args.obstacles} obstacles{', STRESS' if args.stress else ''})\n")
    print(experiment_b(args.episodes, args.seed, args.obstacles, args.landmarks, args.policy, sensor, noise))


if __name__ == "__main__":
    main()
