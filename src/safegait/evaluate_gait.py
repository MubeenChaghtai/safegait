from __future__ import annotations

import argparse
import collections

import numpy as np

from safegait.gait import (
    GaitSpec,
    execute_step,
    humanoid_spec,
    init_walker,
    plan_steps,
    step_commands,
    to_foot_frame,
)
from safegait.gait_runner import STOP_STEPS, run_gait_episode, start_is_feasible
from safegait.gait_validator import GaitValidator
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.scene import random_episode

SPEC, GS = humanoid_spec(), GaitSpec()


def episodes(n: int, seed: int, n_obstacles: int, all_starts: bool):
    out, skipped, ep = [], 0, 0
    while len(out) < n:
        scene, s0, target = random_episode(np.random.default_rng(seed + ep), SPEC, n_obstacles, 0)
        if all_starts or start_is_feasible(SPEC, GS, scene, s0):
            out.append((seed + ep, scene, s0, target))
        else:
            skipped += 1
        ep += 1
    return out, skipped


def _policy(noisy: bool, ep_seed: int):
    pol = JacobianPolicy(SPEC, horizon=3 * GS.ticks_per_step)
    return NoisyPolicy(pol, SPEC, np.random.default_rng(10_000 + ep_seed)) if noisy else pol


def _survives(v, direction, delta, parity, tick) -> bool:
    w, fell = init_walker(GS, np.zeros(3)), False
    cmds = np.tile([v, 0.0], (4, 1))
    for k in range(15 + parity):
        push = (tick, delta * np.array(direction)) if k == 6 + parity else None
        w, tr = execute_step(GS, w, plan_steps(GS, w, cmds), push=push)
        fell |= tr.fell
    return not fell


def push_tolerance(v: float, direction) -> float:
    worst = 0.8
    for parity in (0, 1):
        for tick in (0, 4, 7):
            if _survives(v, direction, 0.8, parity, tick):
                continue
            lo, hi = 0.0, 0.8
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if _survives(v, direction, mid, parity, tick) else (lo, mid)
            worst = min(worst, lo)
    return worst


def experiment_a() -> str:
    rows = []
    for v in (0.0, 0.25, 0.5):
        w, cmds, xs, use = init_walker(GS, np.zeros(3)), np.tile([v, 0.0], (4, 1)), [], 0.0
        for k in range(16):
            plan = plan_steps(GS, w, cmds)
            w, tr = execute_step(GS, w, plan)
            xs.append(w.c[0])
            if k >= 8:
                use = max(use, np.abs(to_foot_frame(tr.foot, tr.zmp)[:, 0]).max() / GS.foot_half_len)
        speed = (xs[-1] - xs[7]) / (8 * GS.step_time)
        plan = plan_steps(GS, w, np.zeros((STOP_STEPS, 2)))
        brake = abs(to_foot_frame(plan.feet[0], plan.z0)[0]) / GS.foot_half_len
        tol = [push_tolerance(v, d) for d in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        rows.append(
            f"| {v:.2f} | {speed:.3f} | {use:.0%} | {brake:.0%} | "
            + " | ".join(f"{100 * t:.1f}" for t in tol)
            + " |"
        )
    head = (
        "| commanded speed (m/s) | achieved (m/s) | max ZMP use along foot, steady walking | ZMP shift needed to brake in one step | "
        "push tol. forward (cm) | backward | left | right |\n|---|---|---|---|---|---|---|---|\n"
    )
    foot = f"\nFoot half-size: {100 * GS.foot_half_len:.0f} cm along, {100 * GS.foot_half_wid:.0f} cm across."
    return head + "\n".join(rows) + foot


def _row(label, runs) -> str:
    f = lambda a: np.mean([getattr(r, a) for r in runs])
    pushes = sum(r.pushes for r in runs)
    fell_per_push = f"{sum(r.fell for r in runs) / pushes:.1%}" if pushes else "–"
    return (
        f"| {label} | {f('success'):.0%} | {f('collided'):.1%} | {f('fell'):.0%} | {f('limit_violation'):.0%} | "
        f"{f('stalled'):.0%} | {f('blocked_chunks'):.1f} | {pushes / len(runs):.1f} | {fell_per_push} |"
    )


HEAD = (
    "| policy / validator | success | collision | fell | limit violation | stalled | blocked chunks/ep | shoves/ep | falls per shove |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def _runs(eps, noisy, use_validator, **kw):
    return [
        run_gait_episode(_policy(noisy, s), SPEC, GS, scene, s0, tgt, use_validator, np.random.default_rng(50_000 + s), **kw)
        for s, scene, s0, tgt in eps
    ]


def experiment_b(eps) -> str:
    rows = []
    for noisy in (False, True):
        name = "noisy" if noisy else "clean"
        rows.append(_row(f"{name}, validator off", _runs(eps, noisy, False)))
        rows.append(_row(f"{name}, validator, no safe-stop check", _runs(eps, noisy, True, validator_stop_steps=0)))
        rows.append(_row(f"{name}, validator, full", _runs(eps, noisy, True)))
    return HEAD + "\n".join(rows)


def experiment_c(eps, push_max, push_prob=0.3) -> str:
    kw = dict(push_prob=push_prob, push_max=push_max)
    rows = [_row("noisy, validator off", _runs(eps, True, False, **kw))]
    for zm in (0.0, 0.02, 0.04):
        rows.append(_row(f"noisy, validator, zmp_margin {zm:.2f}", _runs(eps, True, True, zmp_margin=zm, **kw)))
    return HEAD + "\n".join(rows)


def experiment_d(eps) -> str:
    tally, total = collections.Counter(), 0
    for s, scene, s0, tgt in eps:
        pol, val = _policy(True, s), GaitValidator(SPEC, GS, scene, 0.05, 0.02, STOP_STEPS)
        w, q, rng = init_walker(GS, s0[:3]), s0[3:].copy(), np.random.default_rng(50_000 + s)
        for _ in range(40):
            ch = pol.propose(np.concatenate([w.base_pose(), q]), tgt)
            n = val.safe_steps(w, q, ch)
            bad = val.first_violation(w, q, ch)
            total += 1
            if bad is None:
                tally["accepted whole chunk"] += 1
            else:
                tally[f"{bad[1]}{' (braking steps)' if bad[0] >= ch.shape[0] // GS.ticks_per_step else ''}"] += 1
            cmds = step_commands(ch, GS.ticks_per_step)[:n].reshape(-1, 2)
            w, _ = execute_step(GS, w, plan_steps(GS, w, np.vstack([cmds, np.zeros((STOP_STEPS, 2))])), rng, 0.005)
            q = q + (ch[: GS.ticks_per_step, 2:].sum(0) * GS.dt if n > 0 else 0.0)
            if np.linalg.norm(w.c - tgt) < 0.15:
                break
    rows = [f"| {k} | {100 * v / total:.1f}% |" for k, v in tally.most_common()]
    return f"({total} chunks)\n\n| first violation in the proposed chunk | share |\n|---|---|\n" + "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", type=int, default=3)
    ap.add_argument("--all-starts", action="store_true", help="keep starts with no safe action (see start_is_feasible)")
    ap.add_argument("--push-max", type=float, nargs="+", default=[0.04, 0.06], help="shove sizes (m of DCM) for experiment C")
    ap.add_argument("--only", choices=["a", "b", "c", "d"])
    args = ap.parse_args()
    run = lambda k: args.only in (None, k)

    if run("a"):
        print("### A. The gait\n")
        print(experiment_a())
    if args.only != "a":
        eps, skipped = episodes(args.episodes, args.seed, args.obstacles, args.all_starts)
        print(f"\n({args.episodes} episodes, {args.obstacles} obstacles; {skipped} starts skipped as having no safe action"
              f"{' -- NOT skipped, --all-starts' if args.all_starts else ''})")
        if run("b"):
            print("\n### B. Validator on the walking robot (no shoves)\n")
            print(experiment_b(eps))
        if run("c"):
            for pm in args.push_max:
                print(f"\n### C. Shoves up to {100 * pm:.0f} cm of DCM, probability 0.3 per step (noisy policy)\n")
                print(experiment_c(eps, pm))
        if run("d"):
            print("\n### D. Why chunks are blocked (noisy policy, validator on)\n")
            print(experiment_d(eps))


if __name__ == "__main__":
    main()
