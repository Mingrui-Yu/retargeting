# Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation

[[Project website](https://star-xcd.github.io/retargeting/)] [[arXiv](https://arxiv.org/abs/2506.09384)]

This repository contains the code for the paper "Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation".

It provides:

- A Python core package for human-to-robot dexterous-hand retargeting.
- Config-driven robot, asset, and retargeting setup.
- Offline replay tooling for quick inspection without ROS or hardware.
- Optional ROS2/RViz, live input, and real robot adapters.

<div align="center">
  <img src="./docs/overview.jpg" alt="retargeting" width="50%" />
</div>

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/retargeting/` | Core Python package for offline replay, config loading, kinematics, and retargeting logic. |
| `src/retargeting_ros/` | Optional ROS adapter package for ROS messages, RViz, and real robot integration. |
| `configs/` | Robot, retargeting, and app-level YAML configs. |
| `assets/robots/` | Robot URDF/MJCF assets organized by robot. |
| `assets/meshes/` | Shared component meshes used by robot assets. |
| `tests/fixtures/` | Replay fixtures used by smoke tests and quickstart examples. |
| `outputs/` | Default location for generated teleop, simulation, benchmark, and plot outputs. This path is gitignored. |
| `ws_ros2/` | ROS2 workspace packages, launch files, robot descriptions, and compatibility entrypoints. |

The intended dependency direction is:

```text
configs/assets -> retargeting core -> apps/CLI -> retargeting_ros / hardware / visualization
```

The core `retargeting` package should be usable without ROS, cameras, RViz, or real robot hardware.

## Install

The quickest path is the offline replay path. It does not require ROS, cameras, or robot hardware.

```bash
git clone --recurse-submodules https://github.com/Mingrui-Yu/retargeting.git
cd retargeting

conda create -n retargeting -c conda-forge python=3.10.12 pinocchio
conda activate retargeting

pip install -e ".[replay]"
```

Optional dependency groups are defined in `pyproject.toml`. Install only what you need:

```bash
pip install -e ".[mujoco]"
pip install -e ".[vision]"
pip install -e ".[avp]"
pip install -e ".[dev]"
```

`nlopt` is installed as a core dependency because the retargeting optimizer is a primary part of this package. Install the GPU-enabled PyTorch build that matches your CUDA driver and runtime from the [official PyTorch instructions](https://pytorch.org/). PyTorch is required for optimizer paths, but it is not pinned in `pyproject.toml` because the correct wheel depends on your CUDA environment. This codebase has been tested with CUDA 12.8 and PyTorch `2.11.0+cu128`.

ROS/RViz and real robot paths additionally require ROS2 Humble and the ROS packages listed in [ROS And RViz](#ros-and-rviz).

## Quickstart: Offline Replay

Generate a reusable offline retargeting result from the repository root:

```bash
python -m retargeting.offline_retarget end=200 run_name=quickstart_leap
```

The same command can optionally run follow-up steps immediately after saving the result:

```bash
python -m retargeting.offline_retarget end=200 run_name=quickstart_leap post.benchmark.enabled=true
python -m retargeting.offline_retarget end=200 run_name=quickstart_leap post.visualize.enabled=true
```

Visualize the saved result in the `viser` web viewer:

```bash
python -m retargeting.viser_retargeting_visualize result=outputs/quickstart_leap/retargeting
```

Compute benchmark statistics and plots from the same result:

```bash
python -m retargeting.benchmark_trajectory result=outputs/quickstart_leap/retargeting
```

For quick inspection, the viewer can still retarget directly without saving an intermediate result:

```bash
python -m retargeting.viser_retargeting_visualize end=200
```

The default Hydra replay config uses:

- replay data: `tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz`
- robot config: `configs/robots/panda_leap_paxini.yaml`
- retargeting config: `configs/retargeting/vector_wrist_joint.yaml`

These paths load a promoted offline replay fixture. They do not start ROS, RViz, cameras, live Vision Pro streaming, or a real robot.

If you only want to verify the package in a headless environment, run:

```bash
python -m pytest tests
```

## Configuration

Robot-specific and retargeting-specific values are configured in YAML files instead of being edited directly in Python source.

| Config | Purpose |
| --- | --- |
| `configs/offline_retarget.yaml` | Hydra entry config for producing saved retargeting results. |
| `configs/replay.yaml` | Hydra entry config for offline replay composition. |
| `configs/benchmark.yaml` | Hydra entry config for benchmark summaries and plots from saved results. |
| `configs/robots/panda_leap_paxini.yaml` | Panda arm + Leap hand with Paxini fingertips. |
| `configs/robots/panda_shadow.yaml` | Panda arm + Shadow hand. |
| `configs/retargeting/vector_wrist_joint.yaml` | Vector wrist joint retargeting settings and link pairs. |
| `configs/solvers/nlopt_slsqp.yaml` | NLopt SLSQP backend and stopping/runtime parameters. |
| `configs/solvers/scipy_slsqp.yaml` | SciPy SLSQP backend and stopping/runtime parameters. |
| `configs/apps/replay_avp.yaml` | Offline replay app defaults. |

Replay uses Hydra config groups. Override groups and values from the command line:

```bash
python -m retargeting.viser_retargeting_visualize \
  robots=panda_shadow \
  solvers=scipy_slsqp \
  data=tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz \
  viewer.port=8090 \
  viewer.no_robot_mesh=true
```

Legacy argparse-style flags remain available for older scripts:

```bash
python -m retargeting.viser_retargeting_visualize \
  --robot configs/robots/panda_shadow.yaml \
  --retarget configs/retargeting/vector_wrist_joint.yaml \
  --hand-type shadow
```

When adding a new robot, prefer this route:

1. Add robot assets under `assets/robots/<robot_name>/`.
2. Add a robot config under `configs/robots/<robot_name>.yaml`.
3. Put joints, frames, model paths, initial qpos, and hand scale in the config.
4. Keep retargeting link pairs and objective settings in `configs/retargeting/`.

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
│   └── panda_shadow/
└── scenes/
```

Robot configs should point to stable paths under `assets/robots/`. Shared component meshes live under `assets/meshes/`; robot URDF asset folders use local symlinks such as `panda`, `leap_hand`, and `shadow_hand` so URDF mesh paths stay relative and portable.

For ROS robot description work, Xacro/URDF files are still available under `ws_ros2/src/my_robot_description/`.

## Data And Outputs

The repository includes a promoted replay fixture for tests and quickstart:

```text
tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz
```

Large experiment data and full benchmark datasets are not bundled as the default quickstart path. The `data/` directory is treated as a local or historical experiment-data location. See [data/README.md](data/README.md) for the current boundary.

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

Leap hand hardware details are in [ws_ros2/src/leaphand_ros2_module/readme.md](ws_ros2/src/leaphand_ros2_module/readme.md).

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
python -c "from retargeting.viser_retargeting_visualize import compose_hydra_replay_config; cfg = compose_hydra_replay_config(['robots=panda_shadow','solvers=scipy_slsqp','viewer.port=8090']); print(cfg['robot']['name'], cfg['solver']['name'], cfg['viewer']['port'])"
```

Check Hydra offline retarget config composition:

```bash
python -c "from retargeting.offline_retarget import compose_hydra_offline_retarget_config; cfg = compose_hydra_offline_retarget_config(['end=1','run_name=smoke']); print(cfg['data'], cfg['run_name'])"
```

Default tests should not start ROS, RViz, cameras, Vision Pro live streaming, real robots, Open3D GUI, or MuJoCo viewer.

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
