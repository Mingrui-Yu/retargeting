# Current Architecture Context

The repository is organized around a dependency direction from offline core to optional runtime adapters:

- `src/retargeting/core/` owns retargeting types, kinematics, optimizers, and solver adapters. It must not import ROS, live detectors, viewers, or command smoothing.
- `src/retargeting/config/`, `inputs/`, `pipelines/`, `artifacts/`, and `apps/` provide configuration, offline replay, artifact handling, and CLI application paths around the core.
- `src/teleoperation/` owns detector adaptation, calibration, output filtering, and live runtime composition. It may depend on `retargeting`.
- `src/retargeting_ros/` is the optional Python-level ROS/RViz/real-robot adapter. It may depend on `retargeting` and `teleoperation`; neither may depend on it.
- `ws_ros2/` contains ROS2 packages, launch files, robot descriptions, and compatibility entrypoints. Its Python compatibility modules forward to the packages under `src/`.

Robot-specific assets, joints, frame names, limits, initial states, and retargeting profiles belong in `assets/` and `configs/`, rather than new core hard-coding. Generated runs belong in the gitignored `outputs/` directory. Keep compact, deterministic fixtures in `tests/fixtures/`.

For a change that crosses these boundaries, preserve the dependency direction and add or update a headless boundary/regression test before considering ROS or hardware validation.
