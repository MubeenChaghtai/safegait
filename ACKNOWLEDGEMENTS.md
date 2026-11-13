# Acknowledgements

Ideas this project builds on, with credit:

- **Grove-G1** by Adyansh Gupta (BSD-3-Clause), https://github.com/Adyansh04/grove-g1:
  the pattern of a policy proposing action chunks that are validated against the scene before execution (`g1_vla`).
- **humanoid_vla** by Ozkan Ceylan (MIT), https://github.com/ozkanceylan/humanoid_vla:
  the structure of a natural-language command selecting a task for a chunk-producing policy, and the evaluation framing.

No source code from these repositories is copied here. If any is added later, its license text
and copyright notice must be kept alongside it.

Standard methods used: damped least squares inverse kinematics with null-space projection (Nakamura & Hanafusa; Chiaverini),
unicycle kinematics for a differential-drive base; EKF-SLAM (Smith, Self & Cheeseman 1990; Durrant-Whyte & Bailey 2006) with nearest-neighbour
data association by chi-square gating; NEES for filter consistency (Bar-Shalom et al.);
the linear inverted pendulum model (Kajita et al. 2001) and divergent-component-of-motion walking (Englsberger, Ott & Albu-Schaeffer 2015).

Third-party models and tools used by the physics check (not redistributed in this repository):

- **unitree_mujoco** by Unitree Robotics (BSD-3-Clause), https://github.com/unitreerobotics/unitree_mujoco: the G1 29-DoF MuJoCo model and meshes.
  `scripts/fetch_g1.sh` downloads it into `third_party/`; its license applies to those files.
- **MuJoCo** by Google DeepMind (Apache-2.0).
