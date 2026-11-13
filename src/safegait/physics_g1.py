from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

DEFAULT_XML = Path("third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml")
ARM_JOINTS = ("right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint")
HAND_BODY = "right_wrist_yaw_link"
ARM_ROOT = "right_shoulder_pitch_link"
Q_START = np.array([0.0, -0.15, 0.0, 0.5])
N_OBS = 3
QD_MAX = 1.5
DT = 0.05


def _build_xml(xml_path: Path) -> str:
    text = xml_path.read_text()
    text = re.sub(r'<joint name="floating_base_joint"[^>]*/>', "", text)
    text = text.replace('meshdir="meshes"', f'meshdir="{(xml_path.parent / "meshes").resolve()}"')
    extra = "".join(
        f'<body name="obs{i}" mocap="true" pos="0 0 -5"><geom name="obs{i}" type="sphere" size="0.08" '
        f'contype="0" conaffinity="1" rgba="0.9 0.2 0.2 0.7"/></body>'
        for i in range(N_OBS)
    )
    extra += '<body name="target" mocap="true" pos="0 0 -5"><geom name="target" type="sphere" size="0.03" contype="0" conaffinity="0" rgba="0.1 0.8 0.2 0.9"/></body>'
    return text.replace("</worldbody>", extra + "</worldbody>")


@dataclass
class ArmEnv:
    xml_path: Path = DEFAULT_XML
    gravity_comp: float = 0.7
    kp: tuple[float, ...] = (60.0, 60.0, 40.0, 40.0)
    kd: tuple[float, ...] = (3.0, 3.0, 2.0, 2.0)
    m: mujoco.MjModel = field(init=False)
    d: mujoco.MjData = field(init=False)

    def __post_init__(self) -> None:
        xml = _build_xml(Path(self.xml_path))
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir=tempfile.gettempdir()) as f:
            f.write(xml)
            tmp = f.name
        try:
            self.m = mujoco.MjModel.from_xml_path(tmp)
        finally:
            os.unlink(tmp)
        m = self.m
        self.d = mujoco.MjData(m)
        self.scratch = mujoco.MjData(m)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.arm_qadr = np.array([m.jnt_qposadr[jid(n)] for n in ARM_JOINTS])
        self.arm_dadr = np.array([m.jnt_dofadr[jid(n)] for n in ARM_JOINTS])
        self.q_lo = np.array([m.jnt_range[jid(n)][0] for n in ARM_JOINTS])
        self.q_hi = np.array([m.jnt_range[jid(n)][1] for n in ARM_JOINTS])
        self.hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, HAND_BODY)
        self.act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.act_qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.is_arm_act = np.isin(self.act_dof, self.arm_dadr)
        self.arm_act = np.array([int(np.where(self.act_dof == a)[0][0]) for a in self.arm_dadr])
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ARM_ROOT)
        in_arm = np.zeros(m.nbody, dtype=bool)
        for b in range(m.nbody):
            k = b
            while k > 0 and k != root:
                k = m.body_parentid[k]
            in_arm[b] = k == root
        self.obs_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obs{i}") for i in range(N_OBS)]
        self.arm_geoms = []
        for g in range(m.ngeom):
            collidable = m.geom_contype[g] != 0 and in_arm[m.geom_bodyid[g]] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            if g in self.obs_geoms or g == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target"):
                continue
            if collidable:
                m.geom_contype[g], m.geom_conaffinity[g] = 1, 0
                self.arm_geoms.append(g)
            else:
                m.geom_contype[g] = m.geom_conaffinity[g] = 0
        self.obs_mocap = [m.body_mocapid[m.geom_bodyid[g]] for g in self.obs_geoms]
        self.target_mocap = m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")]
        self.hold_q = np.zeros(m.nq)
        self.q_ref = Q_START.copy()
        self.obstacles: list[tuple[np.ndarray, float]] = []
        self.n_active = 0

    def reset(self, obstacles: list[tuple[np.ndarray, float]], target: np.ndarray, q0: np.ndarray = Q_START) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm_qadr] = q0
        self.q_ref = q0.copy()
        self.obstacles = obstacles
        self.n_active = len(obstacles)
        for i, mid in enumerate(self.obs_mocap):
            if i < len(obstacles):
                pos, r = obstacles[i]
                self.d.mocap_pos[mid] = pos
                self.m.geom_size[self.obs_geoms[i], 0] = r
                self.m.geom_rbound[self.obs_geoms[i]] = r
            else:
                self.d.mocap_pos[mid] = [0, 0, -5]
        self.d.mocap_pos[self.target_mocap] = target
        mujoco.mj_forward(self.m, self.d)

    def q(self) -> np.ndarray:
        return self.d.qpos[self.arm_qadr].copy()

    def hand_pos(self, d: mujoco.MjData | None = None) -> np.ndarray:
        return (self.d if d is None else d).xpos[self.hand].copy()

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.scratch.qpos[:] = self.d.qpos
        self.scratch.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.m, self.scratch)
        return self.scratch.xpos[self.hand].copy()

    def hand_jacobian(self, q: np.ndarray) -> np.ndarray:
        self.scratch.qpos[:] = self.d.qpos
        self.scratch.qpos[self.arm_qadr] = q
        mujoco.mj_kinematics(self.m, self.scratch)
        mujoco.mj_comPos(self.m, self.scratch)
        jp = np.zeros((3, self.m.nv))
        mujoco.mj_jac(self.m, self.scratch, jp, None, self.scratch.xpos[self.hand], self.hand)
        return jp[:, self.arm_dadr]

    def clearance(self, q: np.ndarray, distmax: float = 0.3) -> float:
        if self.n_active == 0:
            return distmax
        self.scratch.qpos[:] = self.d.qpos
        self.scratch.qpos[self.arm_qadr] = q
        self.scratch.mocap_pos[:] = self.d.mocap_pos
        mujoco.mj_kinematics(self.m, self.scratch)
        best = distmax
        fromto = np.zeros(6)
        for go in self.obs_geoms[: self.n_active]:
            for ga in self.arm_geoms:
                best = min(best, mujoco.mj_geomDistance(self.m, self.scratch, ga, go, distmax, fromto))
        return float(best)

    def execute(self, qdot_cmd: np.ndarray) -> tuple[bool, bool]:
        m, d = self.m, self.d
        limit_violation = bool(np.any(np.abs(qdot_cmd) > QD_MAX + 1e-9))
        nxt = self.q_ref + qdot_cmd * DT
        if np.any(nxt < self.q_lo) or np.any(nxt > self.q_hi):
            limit_violation = True
        self.q_ref = np.clip(nxt, self.q_lo, self.q_hi)
        qd_ref = qdot_cmd.copy()
        collided = False
        kp, kd = np.asarray(self.kp), np.asarray(self.kd)
        n_sub = int(round(DT / m.opt.timestep))
        for k in range(n_sub):
            q_all, v_all = d.qpos[self.act_qadr], d.qvel[self.act_dof]
            tau = 40.0 * (self.hold_q[self.act_qadr] - q_all) - 2.0 * v_all + d.qfrc_bias[self.act_dof]
            q_arm, v_arm = d.qpos[self.arm_qadr], d.qvel[self.arm_dadr]
            tau[self.arm_act] = kp * (self.q_ref - q_arm) + kd * (qd_ref - v_arm) + self.gravity_comp * d.qfrc_bias[self.arm_dadr]
            d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            mujoco.mj_step(m, d)
            if d.ncon > 0 and self.n_active > 0:
                collided = True
                break
        return collided, limit_violation


class ArmPolicy:
    def __init__(self, env: ArmEnv, horizon: int = 10, gain: float = 2.0, damping: float = 0.05, max_speed: float = 0.5):
        self.env, self.horizon, self.gain, self.damping, self.max_speed = env, horizon, gain, damping, max_speed

    def command(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        v = self.gain * (target - self.env.fk(q))
        n = np.linalg.norm(v)
        if n > self.max_speed:
            v *= self.max_speed / n
        jac = self.env.hand_jacobian(q)
        qd = jac.T @ np.linalg.solve(jac @ jac.T + self.damping**2 * np.eye(3), v)
        return np.clip(qd, -0.7 * QD_MAX, 0.7 * QD_MAX)

    def propose(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        chunk, cur = np.zeros((self.horizon, 4)), q.copy()
        for k in range(self.horizon):
            chunk[k] = self.command(cur, target)
            cur = cur + chunk[k] * DT
        return chunk


class NoisyArmPolicy:
    def __init__(self, base: ArmPolicy, rng: np.random.Generator, sigma: float = 0.25, p_glitch: float = 0.15):
        self.base, self.rng, self.sigma, self.p_glitch = base, rng, sigma, p_glitch

    def propose(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        chunk = self.base.propose(q, target)
        chunk = chunk + self.rng.normal(0.0, self.sigma, chunk.shape) * QD_MAX * 0.3
        if self.rng.random() < self.p_glitch:
            k0, j = int(self.rng.integers(0, len(chunk))), int(self.rng.integers(0, chunk.shape[1]))
            chunk[k0:, j] += self.rng.choice([-1.0, 1.0]) * 2.5 * QD_MAX
        return chunk


class ArmValidator:
    def __init__(self, env: ArmEnv, margin: float = 0.03):
        self.env, self.margin = env, margin

    def safe_prefix(self, q: np.ndarray, chunk: np.ndarray) -> int:
        cur, prev_gap = q.copy(), self.env.clearance(q)
        for k, u in enumerate(chunk):
            nxt = cur + u * DT
            if np.any(np.abs(u) > QD_MAX + 1e-9) or np.any(nxt < self.env.q_lo) or np.any(nxt > self.env.q_hi):
                return k
            gap = self.env.clearance(nxt)
            if gap < self.margin and gap < prev_gap:
                return k
            cur, prev_gap = nxt, gap
        return len(chunk)


@dataclass
class ArmResult:
    success: bool = False
    collided: bool = False
    limit_violation: bool = False
    stalled: bool = False
    steps: int = 0
    blocked_chunks: int = 0
    hand_gap: list[float] = field(default_factory=list)
    joint_gap: list[float] = field(default_factory=list)


def random_arm_episode(env: ArmEnv, rng: np.random.Generator, n_obstacles: int = 2):
    q_t = np.array([rng.uniform(-1.4, -0.3), rng.uniform(-0.5, 0.2), rng.uniform(-0.5, 0.5), rng.uniform(0.6, 1.7)])
    start, target = env.fk(Q_START), env.fk(q_t)
    obstacles = []
    for _ in range(n_obstacles):
        for _ in range(100):
            t = rng.uniform(0.35, 0.8)
            c = start + t * (target - start) + rng.normal(0, 0.02, 3)
            r = rng.uniform(0.05, 0.08)
            if np.linalg.norm(c - target) > r + 0.07 and np.linalg.norm(c - start) > r + 0.12:
                obstacles.append((c, r))
                break
    return obstacles, target


def run_arm_episode(env: ArmEnv, policy, obstacles, target, use_validator: bool, margin: float = 0.03,
                    exec_n: int = 5, max_ticks: int = 300, tol: float = 0.05, max_blocks: int = 6, on_tick=None) -> ArmResult:
    env.reset(obstacles, target)
    validator = ArmValidator(env, margin)
    res, consecutive = ArmResult(), 0
    while res.steps < max_ticks:
        q = env.q()
        chunk = policy.propose(q, target)
        n = min(validator.safe_prefix(q, chunk), exec_n) if use_validator else exec_n
        if n == 0:
            res.blocked_chunks += 1
            consecutive += 1
            res.steps += exec_n
            if consecutive >= max_blocks:
                res.stalled = True
                break
            continue
        consecutive = 0
        q_pred = q.copy()
        for u in chunk[:n]:
            q_pred = q_pred + u * DT
            collided, lim = env.execute(u)
            res.limit_violation |= lim
            res.collided |= collided
            res.steps += 1
            if on_tick is not None:
                on_tick(env, res)
            if collided:
                return res
        res.hand_gap.append(float(np.linalg.norm(env.fk(np.clip(q_pred, env.q_lo, env.q_hi)) - env.hand_pos())))
        res.joint_gap.append(float(np.max(np.abs(np.clip(q_pred, env.q_lo, env.q_hi) - env.q()))))
        if np.linalg.norm(env.hand_pos() - target) < tol:
            res.success = True
            return res
    return res
