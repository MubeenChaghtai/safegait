from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from safegait.kinematics import RobotSpec
from safegait.learned import MLP, collect_demos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/bc_policy.npz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    spec = RobotSpec()
    t0 = time.time()
    x, y = collect_demos(spec, args.episodes, rng)
    print(f"collected {len(x)} samples from {args.episodes} episodes in {time.time() - t0:.0f} s")
    net = MLP([x.shape[1], args.hidden, args.hidden, y.shape[1]], rng)
    t0 = time.time()
    net.fit(x, y, args.epochs, rng)
    print(f"trained in {time.time() - t0:.0f} s")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    net.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
