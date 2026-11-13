import numpy as np

from safegait.gait import (
    GaitSpec,
    WalkerState,
    execute_step,
    humanoid_spec,
    init_walker,
    plan_steps,
    step_commands,
    to_foot_frame,
    zmp_excess,
)
from safegait.gait_runner import run_gait_episode, start_is_feasible
from safegait.gait_validator import GaitValidator
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.scene import Scene, random_episode

GS = GaitSpec()
SPEC = humanoid_spec()
T = GS.ticks_per_step
Q0 = np.array([0.5, -0.9, 0.5])


def _walk(v, steps, gs=GS, pose=(0.0, 0.0, 0.0), wz=0.0):
    w, cmds, traces = init_walker(gs, np.array(pose)), np.tile([v, wz], (4, 1)), []
    for _ in range(steps):
        w, tr = execute_step(gs, w, plan_steps(gs, w, cmds))
        traces.append(tr)
    return w, traces


def _chunk(v, wz=0.0, steps=3):
    c = np.zeros((steps * T, 5))
    c[:, 0], c[:, 1] = v, wz
    return c


def test_lipm_tick_matches_numerical_integration():
    om, dt = GS.omega, GS.dt
    c, cd, z = np.array([0.1, -0.05]), np.array([0.3, 0.2]), np.array([0.02, 0.01])
    ch, sh = np.cosh(om * dt), np.sinh(om * dt)
    closed = (z + (c - z) * ch + cd / om * sh, om * (c - z) * sh + cd * ch)

    def f(y):
        return np.concatenate([y[2:], om**2 * (y[:2] - z)])

    y, h = np.concatenate([c, cd]), dt / 200
    for _ in range(200):
        k1 = f(y)
        k2 = f(y + h / 2 * k1)
        k3 = f(y + h / 2 * k2)
        k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(closed[0], y[:2], atol=1e-9) and np.allclose(closed[1], y[2:], atol=1e-9)


def test_marching_orbit_is_periodic():
    w = init_walker(GS, np.array([0.3, -0.2, 0.7]))
    states = []
    for _ in range(12):
        plan = plan_steps(GS, w, np.zeros((3, 2)))
        assert np.linalg.norm(plan.z0 - w.foot[:2]) < 1e-9
        w, _ = execute_step(GS, w, plan)
        states.append(np.concatenate([w.c, w.cd]))
    assert np.allclose(states[-1], states[-3], atol=1e-8)


def test_executed_step_lands_exactly_on_the_planned_dcm():
    w, _ = _walk(0.4, 6)
    plan = plan_steps(GS, w, np.tile([0.4, 0.0], (4, 1)))
    assert zmp_excess(GS, plan.feet[0], plan.z0) == 0.0
    w2, _ = execute_step(GS, w, plan)
    assert np.allclose(w2.dcm(GS), plan.xi_eos[0], atol=1e-9)


def test_speed_and_yaw_rate_follow_the_command():
    w, _ = _walk(0.4, 14)
    w_later, _ = _walk(0.4, 22)
    assert abs((w_later.c[0] - w.c[0]) / (8 * GS.step_time) - 0.4) < 0.004
    turned, _ = _walk(0.2, 10, wz=0.5)
    assert abs(turned.foot[2] - 10 * 0.5 * GS.step_time) < 1e-9


def test_zmp_stays_inside_the_foot_when_walking_steadily():
    _, traces = _walk(0.5, 16)
    for tr in traces[6:]:
        assert tr.zmp_saturated == 0
        assert all(zmp_excess(GS, tr.foot, z) == 0.0 for z in tr.zmp)


def test_reach_limit_clamps_and_flags():
    w = init_walker(GS, np.zeros(3))
    assert not plan_steps(GS, w, np.tile([0.5, 0.0], (3, 1))).clamped[0]
    big = plan_steps(GS, w, np.tile([2.0, 0.0], (3, 1)))
    assert big.clamped[0]
    assert to_foot_frame(big.feet[0], big.feet[1, :2])[0] <= GS.max_step_len + 1e-9


def test_push_tolerance_equals_the_foot_half_size():
    def tolerance(gs, direction):
        worst = 0.8
        for parity in (0, 1):
            for tick in (0, 7):

                def survives(delta):
                    w, cmds, fell = init_walker(gs, np.zeros(3)), np.tile([0.25, 0.0], (4, 1)), False
                    for k in range(15 + parity):
                        push = (tick, delta * np.array(direction)) if k == 6 + parity else None
                        w, tr = execute_step(gs, w, plan_steps(gs, w, cmds), push=push)
                        fell |= tr.fell
                    return not fell

                lo, hi = 0.0, 0.8
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if survives(mid) else (lo, mid)
                worst = min(worst, lo)
        return worst

    for half_len, half_wid in ((0.11, 0.05), (0.20, 0.08), (0.08, 0.03)):
        gs = GaitSpec(foot_half_len=half_len, foot_half_wid=half_wid)
        assert abs(tolerance(gs, (1, 0)) - half_len) < 0.005
        assert abs(tolerance(gs, (0, 1)) - half_wid) < 0.005


def test_clean_chunk_in_an_empty_scene_is_accepted():
    w, _ = _walk(0.3, 8)
    val = GaitValidator(SPEC, GS, Scene(), 0.05, 0.02, 2)
    assert val.first_violation(w, Q0, _chunk(0.3)) is None
    assert val.safe_steps(w, Q0, _chunk(0.3)) == 3


def test_balance_check_fires_when_the_zmp_would_leave_the_foot():
    w, _ = _walk(0.5, 10)
    val = GaitValidator(SPEC, GS, Scene(), 0.05, 0.02, 2)
    plan = plan_steps(GS, w, np.vstack([np.tile([-0.2, 0.0], (3, 1)), np.zeros((2, 2))]))
    assert not plan.clamped[0]
    assert val.first_violation(w, Q0, _chunk(-0.2)) == (0, "balance")
    assert val.first_violation(w, Q0, _chunk(-0.1)) is None
    assert GaitValidator(SPEC, GS, Scene(), 0.05, 0.0, 2).first_violation(w, Q0, _chunk(-0.2)) is None


def test_reach_check_fires_on_an_unreachable_step():
    w, _ = _walk(0.3, 8)
    assert GaitValidator(SPEC, GS, Scene(), 0.05, 0.0, 2).first_violation(w, Q0, _chunk(2.0))[1] in (
        "step_limit",
        "velocity",
    )


def test_safe_stop_check_rejects_a_prefix_the_robot_cannot_brake_after():
    w, _ = _walk(0.5, 10)
    scene = Scene(obstacles=np.array([[w.c[0] + 1.9, 0.0, 0.3]]))
    with_stop = GaitValidator(SPEC, GS, scene, 0.05, 0.02, 2)
    without = GaitValidator(SPEC, GS, scene, 0.05, 0.02, 0)
    chunk = _chunk(0.5)
    assert without.safe_steps(w, Q0, chunk) == 3
    assert with_stop.safe_steps(w, Q0, chunk) == 2
    step, kind = with_stop.first_violation(w, Q0, chunk)
    assert kind == "collision" and step >= 3


def test_base_velocity_is_judged_on_the_step_average_not_per_tick():
    w, _ = _walk(0.3, 8)
    chunk = _chunk(0.3)
    chunk[0, 0] = 0.9
    assert GaitValidator(SPEC, GS, Scene(), 0.05, 0.0, 2).first_violation(w, Q0, chunk) is None
    chunk[:T, 0] = 0.9
    assert GaitValidator(SPEC, GS, Scene(), 0.05, 0.0, 2).first_violation(w, Q0, chunk) is not None


def test_step_commands_average_each_step_window():
    chunk = np.zeros((16, 5))
    chunk[:8, 0], chunk[8:, 0] = 0.2, 0.4
    chunk[:4, 1] = 0.8
    assert np.allclose(step_commands(chunk, 8), [[0.2, 0.4], [0.4, 0.0]])


def _episode(seed, noisy, validator, **kw):
    scene, s0, target = random_episode(np.random.default_rng(seed), SPEC, 3, 0)
    pol = JacobianPolicy(SPEC, horizon=3 * T)
    if noisy:
        pol = NoisyPolicy(pol, SPEC, np.random.default_rng(10_000 + seed))
    return scene, s0, run_gait_episode(pol, SPEC, GS, scene, s0, target, validator, np.random.default_rng(50_000 + seed), **kw)


def test_validated_walking_never_collides_or_violates_limits_from_feasible_starts():
    n_checked = 0
    for seed in range(40):
        scene, s0, _ = _episode(seed, True, True)
        if not start_is_feasible(SPEC, GS, scene, s0):
            continue
        n_checked += 1
        _, _, r = _episode(seed, True, True)
        assert not r.collided and not r.limit_violation and not r.fell, seed
    assert n_checked >= 30


def test_without_the_validator_the_same_episodes_do_collide():
    runs = [_episode(seed, False, False)[2] for seed in range(40)]
    assert sum(r.collided for r in runs) >= 3


def test_start_feasibility_detects_a_start_with_no_safe_action():
    scene, s0, _ = random_episode(np.random.default_rng(194), SPEC, 3, 0)
    assert not start_is_feasible(SPEC, GS, scene, s0)
    assert start_is_feasible(SPEC, GS, Scene(), s0)


def test_a_shove_inside_the_foot_is_survived_and_one_beyond_it_is_not():
    w, cmds = init_walker(GS, np.zeros(3)), np.tile([0.25, 0.0], (4, 1))
    for _ in range(6):
        w, _ = execute_step(GS, w, plan_steps(GS, w, cmds))
    for size, expect_fall in ((0.9 * GS.foot_half_wid, False), (1.3 * GS.foot_half_wid, True)):
        sim, fell = w.copy(), False
        sim, tr = execute_step(GS, sim, plan_steps(GS, sim, cmds), push=(0, np.array([0.0, size])))
        fell |= tr.fell
        for _ in range(8):
            sim, tr = execute_step(GS, sim, plan_steps(GS, sim, cmds))
            fell |= tr.fell
        assert fell == expect_fall


def test_a_perfectly_tracked_maximum_reach_gait_is_not_called_a_fall():
    w, cmds, worst = init_walker(GS, np.zeros(3)), np.tile([1.25, 0.0], (4, 1)), 0.0
    for _ in range(14):
        plan = plan_steps(GS, w, cmds)
        assert plan.clamped[0] or worst == 0.0
        w, tr = execute_step(GS, w, plan)
        assert not tr.fell
        worst = max(worst, np.linalg.norm(tr.dcm - tr.foot[:2], axis=1).max())
    assert 0.40 < worst < GS.fall_radius
