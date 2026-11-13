from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safegait.kinematics import RobotSpec
from safegait.localisation import wrap


def humanoid_spec() -> RobotSpec:
    return RobotSpec(v_max=0.5, w_max=0.8)


@dataclass(frozen=True)
class GaitSpec:
    com_height: float = 0.65
    dt: float = 0.05
    ticks_per_step: int = 8
    step_width: float = 0.20
    foot_half_len: float = 0.11
    foot_half_wid: float = 0.05
    max_step_len: float = 0.30
    max_step_back: float = 0.15
    min_step_width: float = 0.12
    max_step_width: float = 0.30
    max_step_yaw: float = 0.35
    max_reach: float = 0.40
    k_dcm: float = 3.0
    g: float = 9.81

    @property
    def fall_radius(self) -> float:
        return self.max_reach + self.foot_half_len

    @property
    def omega(self) -> float:
        return float(np.sqrt(self.g / self.com_height))

    @property
    def step_time(self) -> float:
        return self.ticks_per_step * self.dt

    @property
    def growth(self) -> float:
        return float(np.exp(self.omega * self.step_time))


def _rot(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def _left(psi: float) -> np.ndarray:
    return np.array([-np.sin(psi), np.cos(psi)])


@dataclass
class WalkerState:
    foot: np.ndarray
    side: int
    c: np.ndarray
    cd: np.ndarray

    def copy(self) -> WalkerState:
        return WalkerState(self.foot.copy(), self.side, self.c.copy(), self.cd.copy())

    def dcm(self, spec: GaitSpec) -> np.ndarray:
        return self.c + self.cd / spec.omega

    def base_pose(self) -> np.ndarray:
        return np.array([self.c[0], self.c[1], wrap(self.foot[2])])


def init_walker(spec: GaitSpec, pose: np.ndarray) -> WalkerState:
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    c0 = np.array([x, y])
    foot = c0 + (spec.step_width / (spec.growth + 1.0)) * _left(th)
    return WalkerState(np.array([foot[0], foot[1], th]), +1, c0, np.zeros(2))


def _next_foot(spec: GaitSpec, foot: np.ndarray, side: int, vx: float, wz: float) -> tuple[np.ndarray, bool]:
    w, big_t = spec.step_width, spec.step_time
    psi = foot[2]
    c_ref = foot[:2] - side * (w / 2.0) * _left(psi)
    psi_next = psi + wz * big_t
    c_next = c_ref + _rot(psi + 0.5 * wz * big_t) @ np.array([vx * big_t, 0.0])
    raw = c_next - side * (w / 2.0) * _left(psi_next)

    dx, dy = _rot(psi).T @ (raw - foot[:2])
    out = -side * dy
    dpsi = wrap(psi_next - psi)

    cdx = float(np.clip(dx, -spec.max_step_back, spec.max_step_len))
    cout = float(np.clip(out, spec.min_step_width, spec.max_step_width))
    radius = float(np.hypot(cdx, cout))
    if radius > spec.max_reach:
        cdx, cout = cdx * spec.max_reach / radius, cout * spec.max_reach / radius
    cpsi = float(np.clip(dpsi, -spec.max_step_yaw, spec.max_step_yaw))
    clamped = abs(cdx - dx) > 1e-9 or abs(cout - out) > 1e-9 or abs(cpsi - dpsi) > 1e-9

    xy = foot[:2] + _rot(psi) @ np.array([cdx, -side * cout])
    return np.array([xy[0], xy[1], psi + cpsi]), clamped


@dataclass
class Plan:
    feet: np.ndarray
    sides: list[int]
    xi_eos: np.ndarray
    z0: np.ndarray
    clamped: list[bool]

    @property
    def n_steps(self) -> int:
        return len(self.xi_eos)


def plan_steps(spec: GaitSpec, state: WalkerState, cmds: np.ndarray) -> Plan:
    cmds = np.asarray(cmds, dtype=float).reshape(-1, 2)
    n = len(cmds)
    assert n >= 1
    feet, sides, clamped = [state.foot.copy()], [state.side], []
    for j in range(n + 2):
        vx, wz = cmds[j] if j < n else (0.0, 0.0)
        nxt, was = _next_foot(spec, feet[-1], sides[-1], float(vx), float(wz))
        feet.append(nxt)
        sides.append(-sides[-1])
        clamped.append(was)
    feet_arr = np.array(feet)

    e = spec.growth
    xi = np.zeros((n, 2))
    s1 = feet_arr[n + 1, :2] - feet_arr[n, :2]
    s2 = feet_arr[n + 2, :2] - feet_arr[n + 1, :2]
    xi[n - 1] = feet_arr[n, :2] + (s1 * e + s2) / (e * e - 1.0)
    for k in range(n - 2, -1, -1):
        xi[k] = feet_arr[k + 1, :2] + (xi[k + 1] - feet_arr[k + 1, :2]) / e
    z0 = (state.dcm(spec) * e - xi[0]) / (e - 1.0)
    return Plan(feet_arr, sides, xi, z0, clamped[:n])


def to_foot_frame(foot: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (np.asarray(p) - foot[:2]) @ _rot(foot[2])


def zmp_excess(spec: GaitSpec, foot: np.ndarray, z: np.ndarray, margin: float = 0.0) -> float:
    f = to_foot_frame(foot, z)
    return float(
        max(abs(f[0]) - (spec.foot_half_len - margin), abs(f[1]) - (spec.foot_half_wid - margin), 0.0)
    )


@dataclass
class StepTrace:
    base: np.ndarray
    zmp: np.ndarray
    dcm: np.ndarray
    foot: np.ndarray
    fell: bool
    zmp_saturated: int


def execute_step(
    spec: GaitSpec,
    state: WalkerState,
    plan: Plan,
    rng: np.random.Generator | None = None,
    zmp_noise: float = 0.0,
    push: tuple[int, np.ndarray] | None = None,
) -> tuple[WalkerState, StepTrace]:
    om, dt, n = spec.omega, spec.dt, spec.ticks_per_step
    foot0, foot1 = plan.feet[0], plan.feet[1]
    rot0 = _rot(foot0[2])

    def clip(z: np.ndarray) -> np.ndarray:
        f = np.clip(rot0.T @ (z - foot0[:2]), [-spec.foot_half_len, -spec.foot_half_wid],
                    [spec.foot_half_len, spec.foot_half_wid])
        return foot0[:2] + rot0 @ f

    z_ref = clip(plan.z0)
    xi_ref0 = state.dcm(spec)
    c, cd = state.c.copy(), state.cd.copy()
    ch, sh = np.cosh(om * dt), np.sinh(om * dt)
    yaw_step = wrap(foot1[2] - foot0[2])

    base, zmps, dcms = np.zeros((n, 3)), np.zeros((n, 2)), np.zeros((n, 2))
    fell, saturated = False, 0
    for i in range(n):
        if push is not None and push[0] == i:
            cd = cd + om * np.asarray(push[1])
        xi = c + cd / om
        xi_ref = z_ref + (xi_ref0 - z_ref) * np.exp(om * i * dt)
        wanted = z_ref + (1.0 + spec.k_dcm) * (xi - xi_ref)
        z = clip(wanted)
        saturated += int(np.linalg.norm(z - wanted) > 1e-9)
        if zmp_noise > 0.0 and rng is not None:
            z = clip(z + rng.normal(0.0, zmp_noise, 2))
        c, cd = z + (c - z) * ch + cd / om * sh, om * (c - z) * sh + cd * ch
        xi_new = c + cd / om
        fell |= bool(np.linalg.norm(xi_new - foot0[:2]) > spec.fall_radius)
        base[i] = [c[0], c[1], wrap(foot0[2] + yaw_step * (i + 1) / n)]
        zmps[i], dcms[i] = z, xi_new

    new = WalkerState(foot1.copy(), -state.side, c, cd)
    return new, StepTrace(base, zmps, dcms, foot0.copy(), fell, saturated)


def step_commands(chunk: np.ndarray, ticks_per_step: int) -> np.ndarray:
    h = chunk.shape[0]
    assert h % ticks_per_step == 0, "chunk length must be a whole number of steps"
    return chunk[:, :2].reshape(h // ticks_per_step, ticks_per_step, 2).mean(axis=1)
