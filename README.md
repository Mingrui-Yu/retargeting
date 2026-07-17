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
| `src/retargeting/` | Core Python package for offline replay, config loading, kinematics, and retargeting logic. |
| `src/teleoperation/` | Input adapters, output filters, and runtime composition around the pure retargeting core. |
| `src/retargeting_ros/` | Optional ROS adapter package for ROS messages, RViz, and real robot integration. |
| `configs/` | Robot, retargeting, and app-level YAML configs. |
| `assets/` | Robot URDF/MJCF assets and shared component meshes. |
| `tests/fixtures/` | Replay fixtures used by smoke tests and quickstart examples. |
| `outputs/` | Default location for generated teleop, simulation, benchmark, and plot outputs. This path is gitignored. |
| `ws_ros2/` | ROS2 workspace packages, launch files, robot descriptions, and compatibility entrypoints. |

The intended dependency direction is:

```text
configs/assets -> input adapters -> retargeting.core -> output adapters -> apps/CLI / retargeting_ros
```

`retargeting.core.Retargeter` consumes a canonical `HandObservation` and produces raw qpos. It has no dependency on ROS, cameras, RViz, hardware control, or output smoothing. `teleoperation` owns detector adaptation, coordinate calibration, command smoothing, and the live runtime composition.

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

Optional dependency groups are defined in `pyproject.toml`. Install only what you need:

```bash
pip install -e ".[mujoco]"
pip install -e ".[vision]"
pip install -e ".[avp]"
pip install -e ".[dev]"
```

For live AVP retargeting directly into headless MuJoCo, install both optional groups and run the online app:

```bash
pip install -e ".[avp,mujoco]"
python -m retargeting.main app=mujoco_simulation avp_ip=192.168.52.6
```

The app retargets at 20 Hz. Each detected frame is commanded immediately, followed by 25 MuJoCo physics steps at
`0.002 s`; it does not create or replay a joint trajectory. Set `simulator.realtime=false` for deterministic
faster-than-wall-clock headless debugging, or `max_frames=N` for a bounded run.

To load a raw offline human trajectory and retarget each frame directly into MuJoCo, use the offline simulation app:

```bash
pip install -e ".[mujoco]"
python -m retargeting.main app=mujoco_offline_simulation \
  data=tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz
```

This path reads only the raw `stream_*` human arrays. It ignores any existing `retarget_qpos` in the input file and
does not create an intermediate robot trajectory. Each source frame is retargeted once and advances exactly `0.05 s`
of simulated time; use `start` and `end` to select a contiguous source interval.
The source is required to be 20 Hz until timestamp-based resampling is implemented.

Install the GPU-enabled PyTorch build that matches your CUDA driver and runtime from the [official PyTorch instructions](https://pytorch.org/). PyTorch is required for optimizer paths, but it is not pinned in `pyproject.toml` because the correct wheel depends on your CUDA environment. This codebase has been tested with CUDA 12.8 and PyTorch `2.11.0+cu128`.

ROS/RViz and real robot paths additionally require ROS2 Humble and the ROS packages listed in [ROS And RViz](#ros-and-rviz).

## Quickstart: Offline Replay

Generate a reusable offline retargeting result from the repository root:

```bash
python -m retargeting.main app=offline_retarget end=200 run_name=quickstart_leap
```

The same command can optionally run follow-up steps immediately after saving the result:

```bash
python -m retargeting.main app=offline_retarget end=200 run_name=quickstart_leap post.benchmark.enabled=true
python -m retargeting.main app=offline_retarget end=200 run_name=quickstart_leap post.visualize.enabled=true
```

Visualize the saved result in the `viser` web viewer:

```bash
python -m retargeting.main app=replay run_name=quickstart_leap
```

Compute benchmark statistics and plots from the same result:

```bash
python -m retargeting.main app=benchmark run_name=quickstart_leap
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
