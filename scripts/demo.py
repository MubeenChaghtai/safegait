import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from safegait import RobotSpec, parse_command
from safegait.kinematics import arm_points
from safegait.policy import JacobianPolicy, NoisyPolicy
from safegait.runner import run_episode
from safegait.scene import random_episode

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
out = sys.argv[2] if len(sys.argv) > 2 else "demo.png"

spec = RobotSpec()
scene, s0, target = random_episode(np.random.default_rng(seed), spec, n_obstacles=3)

task = parse_command("Please pick up the red cube", list(scene.objects))
assert task is not None
goal = scene.objects[task.target]

policy = NoisyPolicy(JacobianPolicy(spec), spec, np.random.default_rng(10_000 + seed))
res = run_episode(policy, spec, scene, s0, goal, use_validator=True, record=True)

fig, ax = plt.subplots(figsize=(7, 5))
for cx, cy, r in scene.obstacles:
    ax.add_patch(Circle((cx, cy), r, color="0.7"))
trace = np.array(res.trace)
ax.plot(trace[:, 0], trace[:, 1], "b-", lw=1, label="base path")
for k in np.linspace(0, len(trace) - 1, 6).astype(int):
    ax.add_patch(Circle(trace[k, :2], spec.base_radius, fill=False, ec="b", alpha=0.3))
    pts = arm_points(spec, trace[k])
    ax.plot(pts[:, 0], pts[:, 1], "r-", lw=1.5, alpha=0.4 + 0.6 * k / len(trace))
ax.plot(*goal, "g*", ms=15, label=task.target)
ax.set_aspect("equal")
ax.legend(loc="upper left")
ax.set_title(f"{task.kind} {task.target}: success={res.success}, blocked chunks={res.blocked_chunks}")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"success={res.success} collided={res.collided} stalled={res.stalled} steps={res.steps} -> {out}")
