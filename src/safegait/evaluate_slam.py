from __future__ import annotations

import argparse

import numpy as np

from safegait.evaluate_loc import NEES_HI, NEES_LO
from safegait.kinematics import RobotSpec
from safegait.localisation import ActuationNoise, SensorSpec
from safegait.localised_runner import MODES, run_localised_episode
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.scene import random_episode

ESTIMATORS_A = (
    ("dead reckoning", "ekf", False),
    ("EKF, map given", "ekf", True),
    ("SLAM, association given", "slam_known", True),
    ("SLAM, association estimated", "slam", True),
)


def _episodes_a(n, seed, n_landmarks, estimator, use_lm, sensor, noise):
    spec = RobotSpec()
    for ep in range(n):
        scene, s0, target = random_episode(
            np.random.default_rng(seed + ep), spec, n_obstacles=0, n_landmarks=n_landmarks
        )
        yield run_localised_episode(
            JacobianPolicy(spec), spec, scene, s0, target, "raw",
            np.random.default_rng(50_000 + seed + ep),
            use_landmarks=use_lm, estimator=estimator, sensor=sensor, noise=noise,
        )


def experiment_a(n, seed, n_landmarks, sensor=SensorSpec(), noise=ActuationNoise()) -> str:
    rows = []
    for label, est, use_lm in ESTIMATORS_A:
        mean_err, final_err, nees_all = [], [], []
        map_err, dup, assoc, wrong, amb, seen, mapped = [], 0, 0, 0, 0, 0, 0
        for r in _episodes_a(n, seed, n_landmarks, est, use_lm, sensor, noise):
            mean_err.append(np.mean(r.pos_err))
            final_err.append(r.pos_err[-1])
            nees_all += r.nees
            if r.slam is not None:
                if not np.isnan(r.map_err):
                    map_err.append(r.map_err)
                dup += r.n_duplicates
                assoc += r.n_assoc
                wrong += r.n_assoc_wrong
                amb += r.n_ambiguous
                seen += r.n_seen
                mapped += r.slam.n_landmarks
        nees_arr = np.array(nees_all)
        slam_cols = (
            f"{np.mean(map_err) * 100:.1f} | {mapped / n:.1f} / {seen / n:.1f} | {dup / n:.2f} | "
            f"{wrong / max(assoc, 1):.1%} | {amb / n:.1f}"
            if map_err
            else "– | – | – | – | –"
        )
        rows.append(
            f"| {label} | {np.mean(mean_err) * 100:.2f} | {np.mean(final_err) * 100:.2f} | "
            f"{nees_arr.mean():.2f} | {np.mean((nees_arr > NEES_LO) & (nees_arr < NEES_HI)):.0%} | {slam_cols} |"
        )
    head = (
        "| estimator | mean pos err (cm) | final pos err (cm) | mean NEES (ideal 3) | inside 95% bound "
        "| map err (cm) | map size / landmarks seen | duplicates/ep | wrong associations | ambiguous/ep |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(rows)


def experiment_b(n, seed, n_obstacles, n_landmarks, sensor, noise) -> str:
    spec = RobotSpec()
    rows = []
    for est in ("ekf", "slam"):
        for mode in MODES:
            st = dict(success=0, collided=0, limit=0, stalled=0, blocked=0)
            for ep in range(n):
                scene, s0, target = random_episode(
                    np.random.default_rng(seed + ep), spec, n_obstacles, n_landmarks
                )
                policy = NoisyPolicy(JacobianPolicy(spec), spec, np.random.default_rng(10_000 + seed + ep))
                r = run_localised_episode(
                    policy, spec, scene, s0, target, mode, np.random.default_rng(50_000 + seed + ep),
                    estimator=est, sensor=sensor, noise=noise,
                )
                st["success"] += r.success
                st["collided"] += r.collided
                st["limit"] += r.limit_violation
                st["stalled"] += r.stalled
                st["blocked"] += r.blocked_chunks
            rows.append(
                f"| {'EKF (map given)' if est == 'ekf' else 'SLAM'} | {mode} | {st['success'] / n:.0%} | "
                f"{st['collided'] / n:.0%} | {st['limit'] / n:.0%} | {st['stalled'] / n:.0%} | {st['blocked'] / n:.1f} |"
            )
    head = (
        "| estimator | validator mode | success | collision | limit violation | stalled | blocked chunks/ep |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(rows)


def experiment_c(n, seed, counts) -> str:
    rows = []
    for k in counts:
        pe, assoc, wrong, amb, dup, seen = [], 0, 0, 0, 0, 0
        for r in _episodes_a(n, seed, k, "slam", True, SensorSpec(), ActuationNoise()):
            pe.append(np.mean(r.pos_err))
            assoc += r.n_assoc
            wrong += r.n_assoc_wrong
            amb += r.n_ambiguous
            dup += r.n_duplicates
            seen += r.n_seen
        rows.append(
            f"| {k} | {seen / n:.1f} | {np.mean(pe) * 100:.2f} | {wrong / max(assoc, 1):.2%} | "
            f"{dup / n:.2f} | {amb / n:.1f} |"
        )
    head = (
        "| landmarks in world | seen per episode | mean pos err (cm) | wrong associations | duplicates/ep | ambiguous/ep |\n"
        "|---|---|---|---|---|---|\n"
    )
    return head + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=3)
    ap.add_argument("--landmarks", type=int, default=8)
    ap.add_argument("--stress", action="store_true", help="3 landmarks, short range, noisier sensor, larger slip (A and B)")
    ap.add_argument("--no-slip", action="store_true", help="remove the per-episode slip bias the filters do not model (A and B)")
    ap.add_argument("--only", choices=["a", "b", "c"], help="run a single experiment")
    args = ap.parse_args()

    if args.stress:
        sensor = SensorSpec(range_std=0.10, bearing_std=0.06, max_range=1.5)
        noise = ActuationNoise(v_scale_std=0.10, w_bias_std=0.08)
        lm = 3
    else:
        sensor, noise, lm = SensorSpec(), ActuationNoise(), args.landmarks
    if args.no_slip:
        noise = ActuationNoise(v_scale_std=0.0, w_bias_std=0.0, v_step_std=noise.v_step_std, w_step_std=noise.w_step_std)
    tag = ", ".join(t for t, on in (("STRESS", args.stress), ("NO SLIP BIAS", args.no_slip)) if on)
    tag = f", {tag}" if tag else ""

    run = lambda key: args.only in (None, key)
    if run("a"):
        print(f"### A. Estimator quality (no obstacles, {lm} landmarks{tag})\n")
        print(experiment_a(args.episodes, args.seed, lm, sensor, noise))
    if run("b"):
        print(f"\n### B. Validator on the estimated pose (noisy policy, {args.obstacles} obstacles, {lm} landmarks{tag})\n")
        print(experiment_b(args.episodes, args.seed, args.obstacles, lm, sensor, noise))
    if run("c") and not (args.stress or args.no_slip):
        print("\n### C. Landmark density\n")
        print(experiment_c(args.episodes, args.seed, (4, 8, 16, 32, 64)))


if __name__ == "__main__":
    main()
