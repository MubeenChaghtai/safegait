import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from safegait import RobotSpec
from safegait.localised_runner import run_localised_episode
from safegait.policy import JacobianPolicy
from safegait.scene import random_episode

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
out = sys.argv[2] if len(sys.argv) > 2 else "slam.png"

spec = RobotSpec()
scene, s0, target = random_episode(np.random.default_rng(seed), spec, n_obstacles=0, n_landmarks=8)


def run(estimator, use_lm=True):
    return run_localised_episode(
        JacobianPolicy(spec), spec, scene, s0, target, "raw",
        np.random.default_rng(50_000 + seed), use_landmarks=use_lm, estimator=estimator,
    )


dead, ekf, slam = run("ekf", False), run("ekf"), run("slam")
f = slam.slam

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})

ax.plot(*np.array(slam.true_xy).T, color="green", lw=2, label="true path")
ax.plot(*np.array(slam.est_xy).T, color="tab:blue", lw=1.2, ls="--", label="SLAM path")
ax.scatter(*scene.landmarks.T, marker="*", s=160, c="k", zorder=5, label="true landmarks")
ax.scatter(*f.landmarks.T, marker="x", s=45, c="tab:red", zorder=6, label="SLAM map")
for j in range(f.n_landmarks):
    vals, vecs = np.linalg.eigh(f.landmark_cov(j))
    w, h = 2 * 2 * np.sqrt(np.maximum(vals, 0))
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(f.landmarks[j], h, w, angle=ang, fill=False, ec="tab:red", alpha=0.6))
ax.scatter(*target, marker="o", s=70, c="orange", zorder=7, label="target")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"map built online ({f.n_landmarks} landmarks, {slam.map_err * 100:.1f} cm mean error)")
ax.legend(fontsize=7, loc="best")
ax.grid(alpha=0.3)

for r, c, lab in ((dead, "r", "dead reckoning"), (ekf, "tab:orange", "EKF, map given"), (slam, "b", "SLAM")):
    ax2.plot(np.arange(len(r.pos_err)) * 0.05, np.array(r.pos_err) * 100, c, label=lab)
ax2.plot(np.arange(len(slam.sigma)) * 0.05, 3 * np.array(slam.sigma) * 100, "b--", alpha=0.5, label="SLAM 3-sigma bound")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("base position error (cm)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_title("pose error")

fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
