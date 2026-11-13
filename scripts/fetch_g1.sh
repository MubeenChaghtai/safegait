#!/usr/bin/env bash
set -euo pipefail
mkdir -p third_party
if [ ! -d third_party/unitree_mujoco ]; then
  git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git third_party/unitree_mujoco
fi
echo "G1 model: third_party/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
