from __future__ import annotations

import sys

import yaml

HEADER = """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="safegait_world">
    <physics name="1ms" type="ignored"><max_step_size>0.002</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><direction>-0.5 0.1 -0.9</direction></light>
    <model name="ground"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
        <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
    </link></model>
"""

CYL = """    <model name="obstacle_{i}"><static>true</static><pose>{x} {y} 0.25 0 0 0</pose><link name="link">
      <collision name="c"><geometry><cylinder><radius>{r}</radius><length>0.5</length></cylinder></geometry></collision>
      <visual name="v"><geometry><cylinder><radius>{r}</radius><length>0.5</length></cylinder></geometry>
        <material><ambient>0.5 0.55 0.65 1</ambient><diffuse>0.5 0.55 0.65 1</diffuse></material></visual>
    </link></model>
"""
OBJ = """    <model name="{name}"><static>true</static><pose>{x} {y} 0.1 0 0 0</pose><link name="link">
      <visual name="v"><geometry><sphere><radius>0.08</radius></sphere></geometry>
        <material><ambient>0.9 0.8 0.2 1</ambient><diffuse>0.9 0.8 0.2 1</diffuse></material></visual>
    </link></model>
"""


def main() -> None:
    scene_yaml, out = sys.argv[1], sys.argv[2]
    p = yaml.safe_load(open(scene_yaml))["safegait_node"]["ros__parameters"]
    xyr, xy, names = p["obstacle_xyr"], p["object_xy"], p["object_names"]
    body = "".join(CYL.format(i=i, x=xyr[3 * i], y=xyr[3 * i + 1], r=xyr[3 * i + 2]) for i in range(len(xyr) // 3))
    body += "".join(OBJ.format(name=n, x=xy[2 * i], y=xy[2 * i + 1]) for i, n in enumerate(names))
    with open(out, "w") as f:
        f.write(HEADER + body + "  </world>\n</sdf>\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
