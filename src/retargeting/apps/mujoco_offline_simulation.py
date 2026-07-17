"""Offline human trajectory retargeting directly into headless MuJoCo."""

from __future__ import annotations

from typing import Any

from retargeting.apps.mujoco_simulation import build_mujoco_runtime, initialize_avp_alignment
from retargeting.config import load_mujoco_simulation_config, resolve_project_path, to_plain_config_data
from retargeting.inputs.offline_avp_replay import (
    iter_frame_indices,
    load_offline_human_trajectory,
)


def run_mujoco_offline_simulation_from_config(
    config: Any,
    argv: list[str] | None = None,
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
    trajectory = load_offline_human_trajectory(data_path)
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

    runtime = build_mujoco_runtime(config_data)
    alignment_initialized = False
    last_diagnostics: dict[str, float] = {}
    for processed_count, frame_idx in enumerate(frame_indices, start=1):
        sensor_data = trajectory.get_frame(frame_idx)
        if not alignment_initialized:
            alignment_initialized = initialize_avp_alignment(runtime, sensor_data)
            if not alignment_initialized:
                result = runtime.step_hold()
            else:
                result = runtime.step_sensor_data(sensor_data)
        else:
            result = runtime.step_sensor_data(sensor_data)
        last_diagnostics = result.diagnostics
        if processed_count % log_every_frames == 0 or processed_count == len(frame_indices):
            print(
                f"source_frame={frame_idx} processed={processed_count}/{len(frame_indices)} "
                f"sim_time={last_diagnostics.get('simulation_time', 0.0):.3f} "
                f"tracking_max={last_diagnostics.get('tracking_error_max', 0.0):.6f}"
            )

    summary = dict(last_diagnostics)
    summary.update(
        {
            "source_frames_processed": float(len(frame_indices)),
            "source_frame_start": float(frame_indices[0]),
            "source_frame_end": float(frame_indices[-1]),
        }
    )
    return summary
