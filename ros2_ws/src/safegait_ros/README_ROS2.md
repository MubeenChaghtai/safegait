# Running safegait on ROS 2, RViz and Gazebo

**Status.** The package has not yet been run on a ROS 2 installation. What *has* been tested: the decision logic
(`core.py`, 6 tests) and the node plus fake robot in a closed loop against an in-memory stand-in for ROS
(`tests/test_ros_stubbed.py`, 4 tests). What has *not* been tested: real ROS 2 (QoS, TF timing, launch files, the RViz
config), and everything about Gazebo (URDF plugins, bridge, world). The Gazebo part should be treated as a starting point that
may need debugging; Step 1 needs no Gazebo and is the recommended first check.

## What the ROS 2 side is (and is not)

It is the **wheeled planar mobile manipulator** (the model in `safegait.kinematics`) with the A* planner, or the
controller, or the learned policy, behind the validator. It reads `/odom` and `/joint_states`, and publishes `/cmd_vel`
and `/arm_joint_{1,2,3}/cmd_vel`. It is **not** the walking model and **not** the Unitree G1 (that one lives in the MuJoCo
experiment, `safegait.physics_g1`). It uses the pose from `/odom` as truth: the EKF and SLAM are not wired into ROS yet.
Obstacles and object positions come from `config/scene.yaml`.

## What you need (Ubuntu 24.04)

Ubuntu 24.04 pairs with **ROS 2 Jazzy Jellyfish** and **Gazebo Harmonic**.

| Need | Why | Install |
|---|---|---|
| ROS 2 Jazzy desktop | `rclpy`, message packages, **RViz 2**, `robot_state_publisher`, `tf2_ros` | follow docs.ros.org (Jazzy, Ubuntu deb packages), then `sudo apt install ros-jazzy-desktop` |
| ROS dev tools | `colcon`, `rosdep` | `sudo apt install ros-dev-tools python3-colcon-common-extensions` |
| Gazebo Harmonic + ROS bridge | simulation, `ros_gz_sim`, `ros_gz_bridge` | `sudo apt install ros-jazzy-ros-gz` |
| PyYAML | the world generator | `sudo apt install python3-yaml` |
| the `safegait` Python package | the policy, planner and validator | see below |
| a working OpenGL setup | RViz and Gazebo draw 3-D | a GPU driver, or software rendering (slow) |

Not needed yet, but you will meet them for the real humanoid work: `ros2_control` (hardware and joint controllers),
Nav2 and `slam_toolbox` (navigation and a standard SLAM; this repository's EKF-SLAM is a separate study), MoveIt 2
(manipulation planning), and Unitree's own ROS 2 packages and SDK for a real G1.

### Installing safegait next to ROS

ROS 2 Jazzy uses the system Python 3.12 and its apt NumPy. Do **not** let pip upgrade NumPy: it can break ROS packages.

```bash
cd safegait
python3 -m venv --system-site-packages ~/safegait_venv && source ~/safegait_venv/bin/activate
pip install --no-deps -e .
```

(`--no-deps` keeps the system NumPy; safegait needs only NumPy >= 1.26, which Jazzy provides.)

## Step 1: RViz only (no Gazebo)

```bash
source /opt/ros/jazzy/setup.bash
source ~/safegait_venv/bin/activate
cd safegait/ros2_ws
colcon build --packages-select safegait_ros
source install/setup.bash
ros2 launch safegait_ros rviz_only.launch.py
```

In another terminal (same two `source` lines plus `source install/setup.bash`):

```bash
ros2 topic pub --once /safegait/command std_msgs/msg/String "{data: 'pick up the red cube'}"
ros2 topic echo /safegait/status
```

You should see the robot drive around the grey obstacles and reach the green sphere. In RViz: **green line** = the part of
the proposed chunk the validator accepted, **red line** = the part it rejected, **blue line** = the A* path, **orange** = the
arm, text above the robot = status (and which limits the proposed chunk would have broken). To watch the validator work,
set `policy: controller` in `config/scene.yaml` (a straight-line policy): the robot stops in front of an obstacle with status
`stalled` instead of hitting it. Set `use_validator: false` to see what happens without it (in the fake robot there is no
collision physics, so it simply drives through, and the red markers show where it would have crashed).

If RViz opens but shows nothing: set *Fixed Frame* to `odom`, add *RobotModel* (topic `/robot_description`) and *MarkerArray*
(topic `/safegait/markers`). The saved config is hand-written and untested.

## Step 2: Gazebo Harmonic

```bash
ros2 run safegait_ros make_world install/safegait_ros/share/safegait_ros/config/scene.yaml \
    install/safegait_ros/share/safegait_ros/worlds/safegait.sdf      # only if you changed scene.yaml
ros2 launch safegait_ros gazebo.launch.py
# then the same ros2 topic pub command as above
```

Points that are likely to need attention (untested):

1. **Joint-state topic.** The URDF asks the Gazebo plugin to publish on `joint_states`. If `ros2 topic echo /joint_states` is empty,
   run `gz topic -l | grep joint` and change the bridge line in `launch/gazebo.launch.py` to that topic.
2. **TF.** Both Gazebo (through the bridge) and `robot_state_publisher` publish transforms. If RViz complains about the
   `odom -> base_link` transform, check `ros2 run tf2_tools view_frames`.
3. **Arm joint controllers.** The URDF uses `gz-sim-joint-controller-system` in velocity mode on `arm_joint_N/cmd_vel`.
   If the arm does not move, check `gz topic -l | grep arm_joint` and that the topics match the bridge.
4. **Base speed.** Wheel and caster friction can make the base slip; the validator's model assumes an ideal unicycle, so
   tracking error is expected. This is the same model-mismatch question as in the localisation and G1 experiments: if
   collisions appear, increase `margin` in `ChunkValidator` (the G1 experiment shows why a margin larger than the tracking error matters).
5. **Simulation time.** Everything runs with `use_sim_time`; if nothing moves, check `ros2 topic echo /clock`.

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/safegait/command` | `std_msgs/String` | in: "pick up the red cube" |
| `/odom`, `/joint_states` | `nav_msgs/Odometry`, `sensor_msgs/JointState` | in |
| `/cmd_vel` | `geometry_msgs/Twist` | out: base (v, w) |
| `/arm_joint_{1,2,3}/cmd_vel` | `std_msgs/Float64` | out: joint rates |
| `/safegait/markers`, `/safegait/status` | `visualization_msgs/MarkerArray`, `std_msgs/String` | out |
