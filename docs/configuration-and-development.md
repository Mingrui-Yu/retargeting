# Configuration and Development Guide

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
python -m retargeting_apps.main app=replay \
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
- MuJoCo Web visualization: `pip install -e ".[mujoco-web]"`
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

### Package Boundaries

The Python packages follow one dependency direction:

```text
retargeting_apps -> teleoperation -> retargeting
       |                              ^
       +------------------------------+

retargeting_ros -> teleoperation / retargeting
```

- `retargeting` owns canonical domain types, core config, kinematics, optimizers, solvers, and pure metrics.
- `teleoperation` owns device adapters, calibration, output filtering, robot backends, and online/offline execution.
- `retargeting_apps` is the only Hydra/CLI composition root and owns artifacts, reports, and replay visualization.
- Every `configs/app/<id>.yaml` maps to `retargeting_apps.apps.<id>.run(config, argv)`. The dispatcher imports only the
  selected whitelisted task. Reusable workflow execution and runtime builders live in `retargeting_apps.pipelines`,
  not in `retargeting_apps.apps`.
- `retargeting_ros` owns optional ROS, RViz, and real-robot adapters. Compatibility scripts under `ws_ros2/` import
  these canonical packages but are not imported by them.

Core code must not import `teleoperation`, `retargeting_apps`, `retargeting_ros`, or optional runtime/viewer modules.
Keep `retargeting.core.*` as the public algorithm path; do not flatten it as part of unrelated changes.

After installation, both entry forms below use the same app registry and Hydra overrides:

```bash
retargeting app=offline_retarget end=1 run_name=smoke
python -m retargeting_apps.main app=offline_retarget end=1 run_name=smoke
```

### Headless Checks

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
python -c "from retargeting_apps.main import compose_hydra_base_config; cfg = compose_hydra_base_config(['app=replay','run_name=quickstart_leap','viewer.port=8090']); print(cfg['run_name'], cfg['viewer']['port'])"
```

Check Hydra offline retarget config composition:

```bash
python -c "from retargeting_apps.main import compose_hydra_base_config; cfg = compose_hydra_base_config(['app=offline_retarget','end=1','run_name=smoke']); print(cfg['data'], cfg['run_name'])"
```

Default tests should not start ROS, RViz, cameras, Vision Pro live streaming, real robots, Open3D GUI, or MuJoCo viewer.

## Online MuJoCo Simulation

`app=mujoco_online_simulation` is a headless live execution path, not a saved-trajectory replay. The app takes the
latest AVP
frame, retargets it once, applies the actuator-range policy, sends the resulting qpos to the MJCF position actuators,
and advances one 20 Hz command period before accepting the next frame. With the default `startup_move_frames=0`,
there is no explicit target-speed limit. The `0.002 s` physics timestep makes one command period exactly 25 MuJoCo
steps.

Robot-specific MJCF paths live under `simulation_model` in the robot config. Runtime timing and range behavior live
in `configs/simulators/mujoco.yaml`; viewer dependencies are intentionally absent from the backend. For headless
diagnostics, override `simulator.realtime=false` so the same fixed simulated time is advanced without wall-clock
sleeping.

For raw offline human input, select `app=mujoco_offline_simulation`. This app loads only the input NPZ's `stream_*`
arrays and explicitly ignores any existing `retarget_qpos`. It initializes wrist alignment from the first valid raw
human frame and retargets that same frame. Its default `startup_move_frames=10` synchronously moves the actuator
target to each of the first 10 valid retarget results before consuming the next source frame. The runtime selects a
shared integer waypoint count using `ceil(max(abs(delta) / (max_joint_speed / command_hz)))`; all joints therefore
finish together and every commanded increment respects its configured speed. After startup, each valid target is
sent directly without an explicit speed limit and advances one command period. Actuator ranges and MuJoCo force,
servo, contact, and dynamics constraints remain active in both phases.

Frames without a valid hand hold the preceding robot command for one period and do not consume the startup count.
Offline simulation defaults to `realtime=false`; `start` and `end` select a contiguous interval. Set `loop=true` to
repeat that selected interval until Ctrl+C. Each repeated cycle resets MuJoCo to the configured robot initial qpos,
clears retargeter/filter/startup temporal state, and initializes relative wrist alignment again from the first valid
frame. The configured `source_hz` must match the 20 Hz command rate until timestamp-based resampling is added. During
startup, one source frame can advance multiple command periods, so simulation time intentionally exceeds the source
timeline.

### Offline MuJoCo Web Visualization

`app=mujoco_offline_simulation` can publish its live MuJoCo state through the optional passive `mjviser` adapter:

```bash
pip install -e ".[mujoco-web]"
python -m retargeting_apps.main app=mujoco_offline_simulation \
  viewer.enabled=true simulator.realtime=true loop=true
```

The adapter uses the same `MjModel` and `MjData` owned by `MujocoRobotBackend` and calls
`ViserMujocoScene.update_from_mjdata()` after every runtime command period, including every startup interpolation
waypoint. It never calls `mj_step`; the runtime remains the sole owner of physics advancement. Viewer settings are
application configuration rather than simulator-backend configuration:

The returned mjviser tab group is extended with a read-only `Joint angles` tab. The adapter enumerates compiled
MuJoCo joint ids, resolves each hinge joint through `model.jnt_qposadr`, and displays the corresponding actual
`data.qpos` value in radians. The current Panda+Leap MJCF has 23 hinge joints, all of which are updated atomically
after every viewer frame. No input callback is registered, so these fields cannot modify simulation state.

```yaml
viewer:
  enabled: false
  host: 0.0.0.0
  port: 9219
  wait_for_client: true
  keep_open_after_completion: false
  camera_distance: -1.0
  camera_azimuth: 120.0
  camera_elevation: 20.0
```

With `wait_for_client=true`, no source frame is consumed until a browser connects. Use
`keep_open_after_completion=true` to retain the final scene until Ctrl+C. Binding `0.0.0.0` exposes the server on
all interfaces; bind `127.0.0.1` and forward the configured port over SSH when direct network exposure is not
appropriate. Offline simulation still defaults to faster-than-wall-clock `simulator.realtime=false`; explicitly
enable realtime pacing for a human-observable 20 Hz run. Viewer publication occurs before each period's remaining
sleep, so its cost is included in realtime pacing without changing simulated time. In continuous mode, the same
viewer remains bound to the existing `MjModel` and `MjData`; the app publishes the reset state before processing the
next cycle. MuJoCo simulation time restarts at zero for every cycle, and Ctrl+C closes the viewer without applying
`keep_open_after_completion`.
