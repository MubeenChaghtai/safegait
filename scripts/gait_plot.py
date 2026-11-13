import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from safegait.gait import GaitSpec, humanoid_spec, to_foot_frame
from safegait.gait_runner import run_gait_episode
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.scene import random_episode

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
out = sys.argv[2] if len(sys.argv) > 2 else "gait.png"

spec, gs = humanoid_spec(), GaitSpec()
scene, s0, target = random_episode(np.random.default_rng(seed), spec, 3, 0)
policy = NoisyPolicy(JacobianPolicy(spec, horizon=3 * gs.ticks_per_step), spec, np.random.default_rng(10_000 + seed))
res = run_gait_episode(policy, spec, gs, scene, s0, target, True, np.random.default_rng(50_000 + seed), record=True)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})

for cx, cy, r in scene.obstacles:
    ax.add_patch(Circle((cx, cy), r, color="0.75"))
for k, tr in enumerate(res.traces):
    x, y, yaw = tr.foot
    corners = np.array([[-gs.foot_half_len, -gs.foot_half_wid], [gs.foot_half_len, -gs.foot_half_wid],
                        [gs.foot_half_len, gs.foot_half_wid], [-gs.foot_half_len, gs.foot_half_wid]])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    left = (k % 2 == 0)
    ax.add_patch(Polygon(corners @ rot.T + [x, y], closed=True, fc="tab:blue" if left else "tab:orange", ec="k", lw=0.6, alpha=0.8))
base = np.array(res.base_path)
ee = np.array(res.ee_path)
ax.plot(base[:, 0], base[:, 1], "g-", lw=1.5, label="pelvis (CoM)")
ax.plot(ee[:, 0], ee[:, 1], "m--", lw=1, label="end effector")
ax.scatter(*target, marker="*", s=180, c="gold", ec="k", zorder=6, label="target")
ax.plot([], [], "s", c="tab:blue", label="left foot"); ax.plot([], [], "s", c="tab:orange", label="right foot")
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
outcome = "success" if res.success else ("collision" if res.collided else ("fell" if res.fell else "stalled/timeout"))
ax.set_title(f"footsteps, seed {seed}: {outcome}, {res.blocked_chunks} blocked chunks")

pts = np.vstack([to_foot_frame(tr.foot, tr.zmp) for tr in res.traces])
ax2.add_patch(Rectangle((-gs.foot_half_len, -gs.foot_half_wid), 2 * gs.foot_half_len, 2 * gs.foot_half_wid, fill=False, ec="k", lw=2))
ax2.scatter(pts[:, 0], pts[:, 1], s=14, c="tab:red", alpha=0.7)
ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
ax2.set_xlim(-0.15, 0.15); ax2.set_ylim(-0.09, 0.09)
ax2.set_xlabel("along the foot (m)"); ax2.set_ylabel("across the foot (m)")
ax2.set_title("ZMP in the stance foot frame (box = the foot)")

fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}: {outcome}, {len(res.traces)} steps, ZMP saturated on {res.saturated_ticks} ticks")
