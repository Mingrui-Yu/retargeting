"""Offline human trajectory retargeting directly into headless MuJoCo."""

from __future__ import annotations

from typing import Any

from retargeting_apps.config import (
    load_mujoco_web_viewer_config,
    resolve_project_path,
    to_plain_config_data,
)
from retargeting_apps.pipelines import mujoco_runtime_builder
from retargeting_apps.visualization.mjviser_live import (
    MujocoWebVisualizer,
    create_mujoco_web_visualizer,
)
from teleoperation.config import load_mujoco_simulation_config
from teleoperation.inputs.offline_avp import (
    iter_frame_indices,
    load_offline_avp_trajectory,
)
from teleoperation.avp_alignment import initialize_avp_alignment
from teleoperation.mujoco_runtime import AlignedMujocoTeleoperationDriver


def run(
    config: Any,
    argv: list[str],
) -> dict[str, float]:
    """Retarget each raw offline human frame and execute it immediately.

    Args:
        config: Composed offline-human MuJoCo application configuration.
        argv: CLI arguments accepted for the common app-runner interface.

    Returns:
        Last-frame diagnostics plus processed source-frame metadata.
    """
    del argv
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected offline MuJoCo simulation config to be a mapping.")
    loop = config_data.get("loop", False)
    if not isinstance(loop, bool):
        raise ValueError("loop must be a boolean.")
    source_hz = float(config_data.get("source_hz", 20.0))
    if source_hz <= 0:
        raise ValueError(f"source_hz must be positive, got {source_hz}.")
    simulator_config = load_mujoco_simulation_config(config_data.get("simulator"))
    if abs(1.0 / source_hz - simulator_config.control_period) > 1e-12:
        raise ValueError(
            "Offline human source_hz must match the MuJoCo command rate until timestamp resampling is supported: "
            f"source_hz={source_hz}, command_hz={simulator_config.command_hz}."
        )
    data_path = resolve_project_path(str(config_data["data"]))
    trajectory = load_offline_avp_trajectory(data_path)
    start = int(config_data.get("start", 0))
    end = int(config_data.get("end", -1))
    frame_indices = list(iter_frame_indices(trajectory.n_frames, start=start, end=end, stride=1))
    if not frame_indices:
        raise ValueError(
            f"No offline human frames selected from {trajectory.source}: start={start}, end={end}."
        )
    log_every_frames = int(config_data.get("log_every_frames", 20))
    if log_every_frames <= 0:
        raise ValueError("log_every_frames must be positive.")

    viewer_config = load_mujoco_web_viewer_config(config_data.get("viewer"))
    runtime = mujoco_runtime_builder.build_mujoco_runtime(config_data)
    driver = AlignedMujocoTeleoperationDriver(
        runtime,
        alignment_initializer=initialize_avp_alignment,
    )
    visualizer: MujocoWebVisualizer | None = None
    try:
        if viewer_config.enabled:
            model = getattr(runtime.backend, "model", None)
            data = getattr(runtime.backend, "data", None)
            if model is None or data is None:
                raise TypeError("MuJoCo Web visualization requires a backend exposing model and data.")
            visualizer = create_mujoco_web_visualizer(model, data, viewer_config)
            visualizer.update()
            runtime.set_post_command_step(visualizer.update)
            visualizer.wait_for_client()

        last_diagnostics: dict[str, float] = {}
        source_frames_processed = 0
        cycle = 1
        try:
            while True:
                for processed_count, frame_idx in enumerate(frame_indices, start=1):
                    sensor_data = trajectory.get_frame(frame_idx)
                    result = driver.step(sensor_data)
                    last_diagnostics = result.diagnostics
                    source_frames_processed += 1
                    if processed_count % log_every_frames == 0 or processed_count == len(frame_indices):
                        print(
                            f"cycle={cycle} source_frame={frame_idx} "
                            f"processed={processed_count}/{len(frame_indices)} "
                            f"sim_time={last_diagnostics.get('simulation_time', 0.0):.3f} "
                            f"tracking_max={last_diagnostics.get('tracking_error_max', 0.0):.6f}"
                        )
                if not loop:
                    break
                driver.reset()
                cycle += 1
                if visualizer is not None:
                    visualizer.update()
        except KeyboardInterrupt:
            if not loop:
                raise

        summary = dict(last_diagnostics)
        summary.update(
            {
                "source_frames_processed": float(source_frames_processed),
                "source_frame_start": float(frame_indices[0]),
                "source_frame_end": float(frame_indices[-1]),
            }
        )
        if visualizer is not None and not loop:
            visualizer.wait_after_completion()
        return summary
    finally:
        if visualizer is not None:
            runtime.set_post_command_step(None)
            visualizer.close()
