# Configuration and Development Guide

> **Status:** This document has not been updated or tested; it will be updated in the future.

## Configuration

Robot-specific, method-specific, and robot-method profile values are configured in YAML files instead of being edited directly in Python source.

| Config | Purpose |
| --- | --- |
| `configs/base.yaml` | Unified Hydra entry config; select an application with `app=<name>`. |
| `configs/app/offline_retarget.yaml` | Offline retargeting app defaults and required config groups. |
| `configs/app/replay.yaml` | Saved-artifact replay viewer defaults. |
| `configs/app/benchmark.yaml` | Benchmark app defaults. |
| `configs/robots/panda_leap_paxini.yaml` | Panda arm + Leap hand with Paxini fingertips. |
| `configs/robots/panda_shadow.yaml` | Panda arm + Shadow hand. |
| `configs/retargeting_methods/vector_wrist_joint.yaml` | Vector wrist joint method metadata and optimizer defaults. |
| `configs/retargeting_profiles/vector_wrist_joint_panda_leap_paxini.yaml` | Panda+Leap profile binding robot, method, target links, objective weights, retargeting runtime weights, and teleoperation command limits. |
| `configs/retargeting_profiles/vector_wrist_joint_panda_shadow.yaml` | Panda+Shadow profile binding robot, method, target links, objective weights, retargeting runtime weights, and teleoperation command limits. |
| `configs/solvers/nlopt_slsqp.yaml` | NLopt SLSQP backend and stopping/runtime parameters. |
| `configs/solvers/scipy_slsqp.yaml` | SciPy SLSQP backend and stopping/runtime parameters. |

Replay accepts an offline retarget runtime name and viewer overrides:

```bash
python -m retargeting.main app=replay \
  run_name=quickstart_leap \
  viewer.port=8090 \
  viewer.no_robot_mesh=true
```

When adding a new robot, prefer this route:

1. Add robot assets under `assets/robots/<robot_name>/`.
2. Add a robot config under `configs/robots/<robot_name>.yaml`.
3. Put joints, frames, model paths, initial qpos, and hand scale in the config.
4. Put robot-method-specific target links, objective weights, and retargeting runtime weights in `configs/retargeting_profiles/<method>_<robot_name>.yaml`; keep command limits in its `teleoperation` section.
5. Reuse or add method-level optimizer defaults under `configs/retargeting_methods/`.

Avoid hard-coding robot-specific joint names, link names, URDF paths, or initial poses in core Python modules.

## Robot Assets

The current core/offline asset layout is:

```text
assets/
├── meshes/
│   ├── leap_hand/
│   ├── panda/
│   └── shadow_hand/
├── robots/
│   ├── panda_leap_paxini/
│   │   ├── manifest.yaml
│   │   ├── meshes/
│   │   ├── mjcf/
│   │   └── urdf/
│   └── panda_shadow/
└── scenes/
```

Robot configs should point to stable paths under `assets/robots/`. `panda_leap_paxini` is a self-contained portable
bundle: its manifest exposes both URDF and MJCF entry points, both descriptions resolve only bundle-local meshes, and
the bundle contains no symlinks. `panda_shadow` continues to reuse component meshes under `assets/meshes/` through
robot-local `panda` and `shadow_hand` symlinks. The shared component directories remain available for that layout and
compatibility with older assets.

For ROS robot description work, Xacro/URDF files are still available under `ws_ros2/src/my_robot_description/`.

## Data And Outputs

The repository includes a promoted replay fixture for tests and quickstart:

```text
tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz
```

Large experiment data and full benchmark datasets are not bundled as the default quickstart path. The `data/` directory is treated as a local or historical experiment-data location. See [data/README.md](../data/README.md) for the current boundary.

New generated files should go under `outputs/`, for example:

```text
outputs/teleop/
outputs/simulation/
outputs/<run_name>/retargeting/
outputs/<run_name>/benchmark/
outputs/<run_name>/plots/
```

Saved offline retargeting, benchmark, and plot artifacts use this layout:

```text
outputs/<run_name>/
  retargeting/
    result.npz
    metadata.yaml
  benchmark/
    metrics.json
    summary.csv
  plots/
    benchmark_metric_means.png
    benchmark_metric_means.pdf
```

`result.npz` stores qpos, human keypoints, wrist poses, robot frame poses, and per-frame optimization errors. `metadata.yaml` stores the replay source, robot config, retargeting config, frame range, and result schema version.

`outputs/` is gitignored.

## ROS And RViz

ROS/RViz is optional and is not required for offline replay.

The ROS path expects Ubuntu 22.04 and ROS2 Humble. When using ROS2 with conda, keep the conda Python version consistent with the system Python used by ROS2.

Install ROS2 Humble by following the [official ROS2 Humble instructions](https://docs.ros.org/en/humble/Installation.html), then install the required ROS packages:

```bash
sudo apt-get install python3-colcon-common-extensions
sudo apt-get install ros-humble-xacro
sudo apt-get install ros-humble-robot-state-publisher
sudo apt-get install ros-humble-joint-state-publisher
sudo apt-get install ros-humble-joint-state-publisher-gui
```

Build the ROS2 workspace:

```bash
cd ws_ros2
colcon build --symlink-install
source install/setup.bash
```

Example RViz launch:

```bash
ros2 launch retargeting_benchmark rviz_vis_paxini.py
```

Python-level ROS integration lives in `src/retargeting_ros/`. Compatibility scripts are still kept under `ws_ros2/src/retargeting_benchmark/src/` so existing launch files and older user commands do not break immediately.

## Live Teleoperation

Live teleoperation is an advanced path. It may require ROS, a camera or Vision Pro live stream, optional visualization dependencies, and robot-specific setup.

The legacy compatibility entrypoint is:

```bash
source ws_ros2/install/setup.bash
python ws_ros2/src/retargeting_benchmark/src/main_robot_teleoperation.py
```

Generated teleoperation recordings now default to `outputs/teleop/`.

Optional live-input dependencies:

- RGB hand detection: `pip install -e ".[vision]"`
- Vision Pro streaming: `pip install -e ".[avp]"`
- MuJoCo-related paths: `pip install -e ".[mujoco]"`
- ROS/RViz/hardware: ROS2 Humble workspace and robot drivers

## Real Robot Control

Real robot control is lab-specific and is not required for offline replay. Confirm robot safety, ROS networking, drivers, and emergency-stop procedures before running any hardware command.

The original lab setup targeted a Franka Panda arm with a Leap hand. The IP addresses below are historical examples from that environment, not portable defaults.

1. Unlock the Panda arm and activate FCI in the robot desk UI.

2. Launch the Franka driver on the Franka control PC:

   ```bash
   ssh robotics@192.168.52.5
   cd franka_emika_panda/ws_ros2/
   source install/setup.bash
   ros2 launch franka_bringup low_level_joint_impedance_controller.launch.py arm_id:=fer robot_ip:=192.168.52.3
   ```

3. Launch Leap hand bringup:

   ```bash
   conda activate retargeting
   ros2 launch leap_hand leap_bringup.py
   ```

4. Prepare real robot ROS nodes:

   ```bash
   conda activate retargeting
   ros2 launch retargeting_benchmark real_prepare.py
   ```

5. Run the teleoperation entrypoint only after confirming the hardware state and ROS topics.

Leap hand hardware details are in [ws_ros2/src/leaphand_ros2_module/readme.md](../ws_ros2/src/leaphand_ros2_module/readme.md).

## Development

Run the headless test suite from the repository root:

```bash
python -m pytest tests
```

Check the core and ROS adapter import boundary:

```bash
python -c "import retargeting; import retargeting_ros"
```

Check Hydra replay config composition without starting the viewer:

```bash
python -c "from retargeting.main import compose_hydra_base_config; cfg = compose_hydra_base_config(['app=replay','run_name=quickstart_leap','viewer.port=8090']); print(cfg['run_name'], cfg['viewer']['port'])"
```

Check Hydra offline retarget config composition:

```bash
python -c "from retargeting.main import compose_hydra_base_config; cfg = compose_hydra_base_config(['app=offline_retarget','end=1','run_name=smoke']); print(cfg['data'], cfg['run_name'])"
```

Default tests should not start ROS, RViz, cameras, Vision Pro live streaming, real robots, Open3D GUI, or MuJoCo viewer.

## Online MuJoCo Simulation

`app=mujoco_simulation` is a headless live execution path, not a saved-trajectory replay. The app takes the latest AVP
frame, retargets it once, applies the configured joint-speed and actuator-range policy, sends the resulting qpos to
the MJCF position actuators, and advances one 20 Hz command period before accepting the next frame. With the default
`0.002 s` physics timestep, one command period is exactly 25 MuJoCo steps.

Robot-specific MJCF paths live under `simulation_model` in the robot config. Runtime timing and range behavior live
in `configs/simulators/mujoco.yaml`; viewer dependencies are intentionally absent from the backend. For headless
diagnostics, override `simulator.realtime=false` so the same fixed simulated time is advanced without wall-clock
sleeping.

For raw offline human input, select `app=mujoco_offline_simulation`. This app loads only the input NPZ's `stream_*`
arrays and explicitly ignores any existing `retarget_qpos`. It initializes wrist alignment from the first valid raw
human frame, retargets that same frame, and then advances one command period for every selected source frame. Frames
without a valid hand hold the preceding robot command while still advancing simulation time, preserving the source
trajectory timeline. Offline simulation defaults to `realtime=false`; `start` and `end` select a contiguous interval.
The configured `source_hz` must match the 20 Hz command rate until timestamp-based resampling is added.
