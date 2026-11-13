from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from safegait.controller import WholeBodyController
from safegait.kinematics import RobotSpec, ee_position, step

N_FEATURES = 8


def featurize(spec: RobotSpec, s: np.ndarray, target: np.ndarray) -> np.ndarray:
    c, sn = np.cos(s[2]), np.sin(s[2])
    to_base = lambda v: np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1]])
    d_target = to_base(target - s[:2])
    d_ee = to_base(target - ee_position(spec, s))
    return np.array([*d_target, *d_ee, *s[3:], np.linalg.norm(d_target)])


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.w = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mx = np.zeros(sizes[0])
        self.sx = np.ones(sizes[0])

    def forward(self, x: np.ndarray, keep: bool = False):
        a = (x - self.mx) / self.sx
        acts = [a]
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            a = a @ w + b
            if i < len(self.w) - 1:
                a = np.maximum(a, 0.0)
            acts.append(a)
        return (a, acts) if keep else a

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int, rng: np.random.Generator,
            lr: float = 1e-3, batch: int = 256, val_frac: float = 0.1, log=print) -> dict:
        self.mx, self.sx = x.mean(0), x.std(0) + 1e-6
        idx = rng.permutation(len(x))
        n_val = int(len(x) * val_frac)
        va, tr = idx[:n_val], idx[n_val:]
        m = [np.zeros_like(p) for p in self.w + self.b]
        v = [np.zeros_like(p) for p in self.w + self.b]
        t = 0
        history = {"train": [], "val": []}
        for ep in range(epochs):
            lr_ep = lr * (0.5 * (1 + np.cos(np.pi * ep / epochs)))
            order = rng.permutation(tr)
            for k in range(0, len(order), batch):
                bi = order[k : k + batch]
                pred, acts = self.forward(x[bi], keep=True)
                grad = 2 * (pred - y[bi]) / len(bi)
                gw, gb = [], []
                for i in reversed(range(len(self.w))):
                    gw.insert(0, acts[i].T @ grad)
                    gb.insert(0, grad.sum(0))
                    if i > 0:
                        grad = (grad @ self.w[i].T) * (acts[i] > 0)
                t += 1
                for j, (p, g) in enumerate(zip(self.w + self.b, gw + gb)):
                    m[j] = 0.9 * m[j] + 0.1 * g
                    v[j] = 0.999 * v[j] + 0.001 * g * g
                    p -= lr_ep * (m[j] / (1 - 0.9**t)) / (np.sqrt(v[j] / (1 - 0.999**t)) + 1e-8)
            tr_loss = float(np.mean((self.forward(x[tr[:5000]]) - y[tr[:5000]]) ** 2))
            va_loss = float(np.mean((self.forward(x[va]) - y[va]) ** 2))
            history["train"].append(tr_loss)
            history["val"].append(va_loss)
            if ep % 10 == 0 or ep == epochs - 1:
                log(f"epoch {ep:3d}  train {tr_loss:.5f}  val {va_loss:.5f}")
        return history

    def save(self, path: str | Path) -> None:
        np.savez(path, mx=self.mx, sx=self.sx, **{f"w{i}": w for i, w in enumerate(self.w)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        z = np.load(path)
        n = sum(1 for k in z.files if k.startswith("w"))
        net = cls([z["w0"].shape[0]] + [z[f"w{i}"].shape[1] for i in range(n)], np.random.default_rng(0))
        net.w = [z[f"w{i}"] for i in range(n)]
        net.b = [z[f"b{i}"] for i in range(n)]
        net.mx, net.sx = z["mx"], z["sx"]
        return net


@dataclass
class LearnedPolicy:
    spec: RobotSpec
    net: MLP
    horizon: int = 10

    def propose(self, s: np.ndarray, target: np.ndarray) -> np.ndarray:
        out = self.net.forward(featurize(self.spec, s, target)[None])[0]
        return out.reshape(self.horizon, 5) * self.spec.cmd_limits()


def collect_demos(spec: RobotSpec, n_episodes: int, rng: np.random.Generator, horizon: int = 10,
                  dt: float = 0.05, exec_n: int = 5, act_noise: float = 0.3, max_steps: int = 300):
    from safegait.policy import JacobianPolicy

    expert = JacobianPolicy(spec, horizon=horizon, dt=dt)
    lim = spec.cmd_limits()
    xs, ys = [], []
    for _ in range(n_episodes):
        q = np.array([0.5, -0.9, 0.5]) + rng.normal(0, 0.5, 3)
        s = np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi), *np.clip(q, spec.q_lo, spec.q_hi)])
        b, d = rng.uniform(-np.pi, np.pi), rng.uniform(0.3, 4.0)
        target = d * np.array([np.cos(b), np.sin(b)])
        steps = 0
        while steps < max_steps and np.linalg.norm(ee_position(spec, s) - target) > 0.05:
            chunk = expert.propose(s, target)
            xs.append(featurize(spec, s, target))
            ys.append((chunk / lim).reshape(-1))
            for u in chunk[:exec_n] + rng.normal(0, act_noise, (exec_n, 5)) * lim * 0.3:
                s = step(spec, s, np.clip(u, -lim, lim), dt)
            steps += exec_n
    return np.array(xs), np.array(ys)
