import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safegait import RobotSpec
from safegait.localised_runner import run_localised_episode
from safegait.policy import JacobianPolicy
from safegait.scene import random_episode

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
out = sys.argv[2] if len(sys.argv) > 2 else "localisation.png"

spec = RobotSpec()
scene, s0, target = random_episode(np.random.default_rng(seed), spec, n_obstacles=0, n_landmarks=8)
runs = {
    use_lm: run_localised_episode(
        JacobianPolicy(spec), spec, scene, s0, target, "raw",
        np.random.default_rng(50_000 + seed), use_landmarks=use_lm,
    )
    for use_lm in (False, True)
}

fig, ax = plt.subplots(figsize=(7, 3.5))
t = np.arange(len(runs[True].pos_err)) * 0.05
ax.plot(np.arange(len(runs[False].pos_err)) * 0.05, np.array(runs[False].pos_err) * 100, "r", label="dead reckoning")
ax.plot(t, np.array(runs[True].pos_err) * 100, "b", label="EKF + landmarks")
ax.plot(t, 3 * np.array(runs[True].sigma) * 100, "b--", alpha=0.5, label="EKF 3-sigma bound")
ax.set_xlabel("time (s)")
ax.set_ylabel("base position error (cm)")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
