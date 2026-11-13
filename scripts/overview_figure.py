import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

from safegait import RobotSpec
from safegait.kinematics import arm_points
from safegait.planner import PlannedPolicy, Q_TRAVEL
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.runner import run_episode
from safegait.scene import random_episode

spec = RobotSpec()


def draw_robot(ax, s, alpha=1.0, label=False, lw=3):
    ax.add_patch(Circle(s[:2], spec.base_radius, fc="#cfe3f7", ec="#2b6cb0", lw=1.5, alpha=alpha))
    h = np.array([np.cos(s[2]), np.sin(s[2])]) * spec.base_radius
    ax.add_patch(Polygon([s[:2] + h * 1.0, s[:2] + 0.5 * h + 0.5 * np.array([-h[1], h[0]]), s[:2] + 0.5 * h - 0.5 * np.array([-h[1], h[0]])], fc="#2b6cb0", alpha=alpha))
    pts = arm_points(spec, s)
    ax.plot(pts[:, 0], pts[:, 1], "-", color="#c53030", lw=lw, alpha=alpha, solid_capstyle="round")
    ax.plot(pts[:-1, 0], pts[:-1, 1], "o", color="#742a2a", ms=5, alpha=alpha)
    ax.plot(*pts[-1], "o", color="#f6ad55", mec="#c05621", ms=8, alpha=alpha)
    return pts


def box(ax, xy, w, h, text, fc):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec="#444", lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, a, b, text="", color="#444", rad=0.0, dy=0.05):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", fontsize=7.5, color=color)


for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 0, 400):
    scene, s0, target = random_episode(np.random.default_rng(seed), spec, 3)
    straight = run_episode(JacobianPolicy(spec), spec, scene, s0, target, False)
    if not straight.collided or straight.steps < 20:
        continue
    pol = NoisyPolicy(PlannedPolicy(spec, scene), spec, np.random.default_rng(10_000 + seed))
    res = run_episode(pol, spec, scene, s0, target, True, record=True)
    tr = np.array(res.trace)[:, :2]
    line = target / np.linalg.norm(target)
    detour = np.max(np.abs(tr[:, 0] * line[1] - tr[:, 1] * line[0]))
    if res.success and res.blocked_chunks >= 1 and detour > 0.35:
        break
print("seed", seed, "success", res.success, "blocked chunks", res.blocked_chunks, "steps", res.steps)
out = sys.argv[2] if len(sys.argv) > 2 else "overview.png"

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1, 1.35], hspace=0.18, wspace=0.08)

ax = fig.add_subplot(gs[0, 0])
s = np.array([0.0, 0.0, 0.0, 0.7, -0.8, 0.5])
draw_robot(ax, s)
ax.annotate("base: differential drive\n(v, w) - cannot move sideways", (-0.2, -0.22), (-0.95, -0.85), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
ax.annotate("arm: 3 joints (q1, q2, q3)\nrates dq1, dq2, dq3", arm_points(spec, s)[1], (0.35, 0.95), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
ax.annotate("end effector\n(must reach the object)", arm_points(spec, s)[-1], (0.95, -0.45), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
ax.annotate("heading", (0.25, 0.0), (-0.15, 0.55), fontsize=8.5, arrowprops=dict(arrowstyle="->"))
fold = np.array([0.0, 0.0, 0.0, *Q_TRAVEL])
draw_robot(ax, fold + np.array([-1.5, 0, 0, 0, 0, 0]), alpha=0.9)
ax.text(-1.5, -0.55, "arm folded for travel\n(inside the base disc)", ha="center", fontsize=8.5)
ax.set_xlim(-2.2, 1.7)
ax.set_ylim(-1.1, 1.3)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("A. The robot: a mobile base with a 3-joint arm", loc="left", fontsize=11)

ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("B. How one decision is made (every 0.25 s)", loc="left", fontsize=11)
box(ax, (0.1, 4.6), 2.0, 0.9, '"pick up the\nred cube"', "#fefcbf")
box(ax, (2.7, 4.6), 2.1, 0.9, "command parser\n-> target position", "#e9d8fd")
box(ax, (5.4, 4.6), 2.5, 0.9, "policy: planner / learned\n-> chunk of 10 commands", "#c6f6d5")
box(ax, (5.4, 2.7), 2.5, 1.0, "validator: predict the\nchunk, keep the safe part", "#fed7d7")
box(ax, (8.5, 2.7), 1.5, 1.0, "robot runs\nfirst 5 steps", "#cfe3f7")
box(ax, (5.4, 0.6), 2.5, 1.0, "state estimate:\nEKF / SLAM", "#e2e8f0")
box(ax, (8.5, 0.6), 1.5, 1.0, "sensors:\nodometry,\nlandmarks", "#e2e8f0")
box(ax, (2.2, 2.7), 2.3, 1.0, "nothing safe: hold still,\nask the policy again", "#feebc8")
arrow(ax, (2.1, 5.05), (2.7, 5.05))
arrow(ax, (4.8, 5.05), (5.4, 5.05))
arrow(ax, (6.65, 4.6), (6.65, 3.7))
arrow(ax, (7.9, 3.2), (8.5, 3.2), "safe part", dy=0.22)
arrow(ax, (5.4, 3.2), (4.5, 3.2), "", color="#c05621")
ax.text(4.95, 3.38, "none\nsafe", ha="center", va="bottom", fontsize=6.5, color="#c05621")
arrow(ax, (3.35, 3.7), (5.5, 4.65), "", color="#c05621", rad=-0.3)
arrow(ax, (9.25, 2.7), (9.25, 1.6), "", dy=0.0)
arrow(ax, (8.5, 1.1), (7.9, 1.1))
arrow(ax, (6.65, 1.6), (6.65, 2.7), "estimated pose", dy=-0.4, color="#2b6cb0")
ax.text(0.1, 1.15, "The policy never commands the robot\ndirectly. Each chunk is predicted first;\ncollisions, joint limits and speed limits\nare checked, and only the safe part runs.", fontsize=8.5, va="center")

ax = fig.add_subplot(gs[1, 0])
for cx, cy, r in scene.obstacles:
    ax.add_patch(Circle((cx, cy), r, fc="#a0aec0", ec="#4a5568"))
trace = np.array(res.trace)
ax.plot(trace[:, 0], trace[:, 1], "-", color="#2b6cb0", lw=1.2, label="base path")
idx = np.linspace(0, len(trace) - 1, 7).astype(int)
for k, i in enumerate(idx):
    draw_robot(ax, trace[i], alpha=0.25 + 0.75 * k / (len(idx) - 1), lw=2)
ax.plot(*target, "*", color="#38a169", ms=22, mec="#22543d", label="red cube (target)", zorder=5)
straight_tr = run_episode(JacobianPolicy(spec), spec, scene, s0, target, False, record=True).trace
st = np.array(straight_tr)
ax.plot(st[:, 0], st[:, 1], "--", color="#c53030", lw=1.2, label="straight-line policy (crashes)")
ax.plot(*st[-1][:2], "X", color="#c53030", ms=12)
ax.set_aspect("equal")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title(
    f"C. One episode: noisy planned policy + validator\nreaches the cube ({res.blocked_chunks} unsafe chunks blocked,\n0 collisions); driving straight crashes (X)",
    loc="left", fontsize=9)
ax.grid(alpha=0.25)
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
ax.set_title("D. What the experiments show (this repository)", loc="left", fontsize=11)
rows = [
    ("Wheeled base, 300 held-out episodes, 3 obstacles", ""),
    ("  straight-line, no validator", "74% success, 26% collide"),
    ("  straight-line + validator", "72% success, 0% collide, 28% stall"),
    ("  A* planner + validator", "98% success, 0% collide, 2% stall"),
    ("Walking base (pendulum model), planner + validator", "97% success, 0% collide (150 episodes)"),
    ("Learned policy (BC) + clipping + validator", "73% success, 0% collide (= controller)"),
    ("Unitree G1 arm in MuJoCo, 300 episodes", ""),
    ("  validator, 3 cm margin", "0% physics collisions (20% without)"),
    ("  validator, 0 cm margin", "3 to 4% still collide"),
]
for i, (a_, b_) in enumerate(rows):
    y = 0.93 - i * 0.095
    ax.text(0.0, y, a_, fontsize=9, fontweight="bold" if not b_ else "normal", transform=ax.transAxes)
    ax.text(0.0, y - 0.04, b_, fontsize=8.5, color="#2c5282", transform=ax.transAxes) if b_ else None
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
