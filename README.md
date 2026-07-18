# Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation

<p align="center">
  <a href="https://mingrui-yu.github.io/retargeting/">Project Website</a>
  &middot;
  <a href="https://arxiv.org/abs/2506.09384">arXiv</a>
</p>

This repository contains the code for the paper "Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation".

It provides:

- A Python core package for human-to-robot dexterous-hand retargeting.
- Config-driven robot, asset, and retargeting setup.
- Offline replay tooling for quick inspection without ROS or hardware.
- Optional ROS2/RViz, live input, and real robot adapters.

<div align="center">
  <img src="./docs/overview.jpg" alt="retargeting" width="50%" />
</div>

## What's New

**2026-07-13** — The codebase has been comprehensively **reorganized** with clearer boundaries between the retargeting core, runtime adapters, configuration, and applications. The new structure also makes quick offline replay easier to discover and run.

We welcome reproductions of this work and use of this codebase as a baseline. Please open an issue with any questions; we will address them and update the repository promptly.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/retargeting/` | Pure retargeting domain types, config, kinematics, optimizers, and evaluation metrics. |
| `src/teleoperation/` | Sensor-first inputs, observation mapping, output/command policies, robot backends, and flows. |
| `src/retargeting_apps/` | CLI composition, offline artifacts, benchmark reports, and replay visualization. |
| `src/retargeting_ros/` | Optional ROS adapter package for ROS messages, RViz, and real robot integration. |
| `configs/` | Robot, retargeting, and app-level YAML configs. |
| `assets/` | Robot URDF/MJCF assets and shared component meshes. |
| `tests/fixtures/` | Replay fixtures used by smoke tests and quickstart examples. |
| `outputs/` | Default location for generated teleop, simulation, benchmark, and plot outputs. This path is gitignored. |
| `ws_ros2/` | ROS2 workspace packages, launch files, robot descriptions, and compatibility entrypoints. |

The intended dependency direction is:

```text
retargeting_apps -> teleoperation -> retargeting
       |                              ^
       +------------------------------+

retargeting_ros -> teleoperation / retargeting
```

`retargeting.core.Retargeter` consumes a canonical `RetargetingHandObservation` and produces raw qpos. It has no
dependency on ROS, cameras, RViz, hardware control, or output smoothing. `teleoperation` exposes one sensor-first
boundary and one top-level execution owner:

```text
HandInput -> SensorHandSample -> HandObservationMapper -> RetargetingHandObservation
          -> Retargeter -> output/command policy -> RobotBackend
```

`ExecutionFlow` owns mapping state, frame processing, atomic command periods, timing, observers, and reset. Offline
artifact generation uses the parallel backend-free `BatchRetargetFlow`. `retargeting_apps` is the outer CLI and
file/viewer composition layer.

## Install

The quickest path is the offline replay path. It does not require ROS, cameras, or robot hardware.

```bash
git clone --recurse-submodules https://github.com/Mingrui-Yu/retargeting.git
cd retargeting

# Required when working from an existing clone that did not initialize submodules.
git submodule update --init --recursive

conda create -n retargeting -c conda-forge python=3.10.12 pinocchio
conda activate retargeting

pip install -e ".[replay]"
```

`mr_utils` is vendored from the pinned `third_party/utils_python` submodule and
is installed together with `retargeting`; do not install it separately.

The install also provides the `retargeting` console command. For example,
`retargeting app=replay run_name=quickstart_leap` is equivalent to
`python -m retargeting_apps.main app=replay run_name=quickstart_leap`.

Optional dependency groups are defined in `pyproject.toml`. Install only what you need:

```bash
pip install -e ".[mujoco]"
pip install -e ".[mujoco-web]"
pip install -e ".[vision]"
pip install -e ".[avp]"
pip install -e ".[dev]"
```

For live AVP retargeting directly into headless MuJoCo, install both optional groups and run the unified execution
app with an online AVP input and MuJoCo backend:

```bash
pip install -e ".[avp,mujoco]"
python -m retargeting_apps.main app=teleop_exe \
  teleoperation_modes=online_mujoco input.avp_ip=192.168.52.6
```

The app retargets at 20 Hz. By default, each detected target is commanded directly without an explicit target-speed
limit, followed by 25 MuJoCo physics steps at `0.002 s`; it does not create or replay a joint trajectory. Set
`teleoperation_mode.pipeline.realtime=false` for deterministic faster-than-wall-clock headless debugging, or
`input.max_frames=N` for a bounded run.

To load a raw offline human trajectory and retarget each frame directly into MuJoCo, use the same app with archived
input:

```bash
pip install -e ".[mujoco]"
python -m retargeting_apps.main app=teleop_exe teleoperation_modes=offline_mujoco \
  input.data=tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz
```

This path reads only the raw `stream_*` human arrays and ignores any existing `retarget_qpos` in the input file. By
default, the first valid retargeted frame uses synchronized linear actuator-target interpolation: every waypoint
respects `profile.teleoperation.max_joint_speed`, advances one `0.05 s` command period, and completes before the next
source frame is consumed. Later frames send their targets directly without an explicit speed limit and advance one
command period. Set `teleoperation_mode.pipeline.startup_move_frames=N` to tune the startup length or `0` to disable
it. Frames without a valid target hold the previous command for one period and do not consume the startup count.

Use `start` and `end` to select a contiguous source interval. Set `loop=true` to repeat that interval continuously
until Ctrl+C. Before every repeated cycle, MuJoCo, temporal retargeting references, startup counters, and wrist
alignment are reset to the configured robot initial qpos. The source is required to be 20 Hz until timestamp-based
resampling is implemented; startup interpolation intentionally advances more simulated time than the source timeline.

To watch the same frame-by-frame simulation in a browser, install the dedicated Web viewer extra and enable the
passive `mjviser` adapter:

```bash
pip install -e ".[mujoco-web]"
python -m retargeting_apps.main app=teleop_exe teleoperation_modes=offline_mujoco \
  input.data=tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz \
  viewer.enabled=true teleoperation_mode.pipeline.realtime=true input.loop=true
```

The app prints the Viser server address and, by default, waits for the first browser client before consuming source
frames. `mjviser` reads the same MuJoCo `model/data` stepped by the execution flow; it does not own or duplicate
physics stepping. Startup interpolation is published after every internal command period, so the browser shows the
complete move rather than only its final state. Set `viewer.wait_for_client=false` for unattended runs or
`viewer.keep_open_after_completion=true` to retain the final state until Ctrl+C. The default
`viewer.host=0.0.0.0` listens on all interfaces; use `viewer.host=127.0.0.1` with SSH port forwarding when the server
should not be exposed directly. Continuous playback exits and closes the viewer on Ctrl+C, so
`viewer.keep_open_after_completion` applies only to finite runs.

The mjviser UI includes a read-only `Joint angles` tab. It reports all 23 Panda+Leap hinge-joint values in radians
from the live MuJoCo `data.qpos` after each completed command period. These are actual simulated joint angles rather
than retargeting requests or actuator targets; the fields cannot write back to the simulation.

Install the GPU-enabled PyTorch build that matches your CUDA driver and runtime from the [official PyTorch instructions](https://pytorch.org/). PyTorch is required for optimizer paths, but it is not pinned in `pyproject.toml` because the correct wheel depends on your CUDA environment. This codebase has been tested with CUDA 12.8 and PyTorch `2.11.0+cu128`.

ROS/RViz and real robot paths additionally require ROS2 Humble and the ROS packages listed in [ROS And RViz](#ros-and-rviz).

## Quickstart: Offline Replay

Generate a reusable offline retargeting result from the repository root:

```bash
python -m retargeting_apps.main app=offline_retarget end=200 run_name=quickstart_leap
```

The same command can optionally run follow-up steps immediately after saving the result:

```bash
python -m retargeting_apps.main app=offline_retarget end=200 run_name=quickstart_leap post.benchmark.enabled=true
python -m retargeting_apps.main app=offline_retarget end=200 run_name=quickstart_leap post.visualize.enabled=true
```

Visualize the saved result in the `viser` web viewer:

```bash
python -m retargeting_apps.main app=replay run_name=quickstart_leap
```

Compute benchmark statistics and plots from the same result:

```bash
python -m retargeting_apps.main app=benchmark run_name=quickstart_leap
```

Replay only plays artifacts saved by `app=offline_retarget`. The saved `metadata.yaml` supplies the robot, profile, and detection calibration needed to reconstruct the viewer context; replay does not rerun retargeting from raw AVP data.

If you only want to verify the package in a headless environment, run:

```bash
python -m pytest tests
```

## Detailed Documentation

Configuration, asset layout, data and outputs, ROS/RViz, teleoperation, real robot control, and development notes are in [docs/configuration-and-development.md](docs/configuration-and-development.md).

## Citation

```bibtex
@article{xin2026analyzing,
  title={Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation},
  author={Xin, Chendong and Yu, Mingrui and Jiang, Yongpeng and Zhang, Zhefeng and Li, Xiang},
  journal={IEEE Robotics and Automation Practice},
  volume={1},
  pages={29--34},
  year={2026},
  doi={10.1109/RAP.2026.3656110}
}
```

## Contact

For questions, contact Mingrui Yu at [mingruiyu98@gmail.com](mailto:mingruiyu98@gmail.com).
