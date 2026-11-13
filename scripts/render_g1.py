import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from safegait.physics_g1 import ArmEnv, ArmPolicy, random_arm_episode, run_arm_episode

out = sys.argv[1] if len(sys.argv) > 1 else "g1_physics.png"
env = ArmEnv()
renderer = mujoco.Renderer(env.m, 360, 480)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.2, -0.2, 0.85]
cam.distance, cam.azimuth, cam.elevation = 0.85, 105, -8


def frame():
    renderer.update_scene(env.d, cam)
    return renderer.render().copy()


for seed in range(200):
    obstacles, target = random_arm_episode(env, np.random.default_rng(seed), 2)
    raw = run_arm_episode(env, ArmPolicy(env), obstacles, target, False)
    val = run_arm_episode(env, ArmPolicy(env), obstacles, target, True, margin=0.03)
    if raw.collided and not val.collided and val.stalled:
        break
print("seed", seed, "raw collided", raw.collided, "validated stalled", val.stalled)

rows = []
for use_val in (False, True):
    frames, ticks = [], []
    run_arm_episode(env, ArmPolicy(env), obstacles, target, use_val, margin=0.03,
                    on_tick=lambda e, r: ticks.append(frame()))
    pick = sorted({0, len(ticks) // 3, 2 * len(ticks) // 3, len(ticks) - 1})
    env.reset(obstacles, target)
    rows.append([frame()] + [ticks[i] for i in pick[1:]])

n = max(len(r) for r in rows)
fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.6), squeeze=False)
for r, (row, label) in enumerate(zip(rows, ("no validator: the arm hits the obstacle", "with validator: the arm stops short"))):
    for c in range(n):
        ax = axes[r][c]
        ax.axis("off")
        if c < len(row):
            ax.imshow(row[c])
    axes[r][0].set_title(label, loc="left", fontsize=10)
fig.suptitle("Unitree G1 right arm in MuJoCo (red: obstacles, green: target)", fontsize=11)
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
