"""Configuration-driven live retargeting into headless MuJoCo simulation."""

from __future__ import annotations

from typing import Any

from retargeting_apps.config import to_plain_config_data
from retargeting_apps.pipelines import mujoco_runtime_builder
from teleoperation.avp_alignment import initialize_avp_alignment
from teleoperation.mujoco_runtime import AlignedMujocoTeleoperationDriver


def run(
    config: Any,
    argv: list[str],
) -> dict[str, float]:
    """Run live AVP retargeting directly into headless MuJoCo at 20 Hz.

    Args:
        config: Composed MuJoCo simulation app configuration.
        argv: CLI arguments accepted for the common app-runner interface.

    Returns:
        Diagnostics from the last completed online frame.
    """
    del argv
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected MuJoCo simulation config to be a mapping.")
    runtime = mujoco_runtime_builder.build_mujoco_runtime(config_data)
    driver = AlignedMujocoTeleoperationDriver(
        runtime,
        alignment_initializer=initialize_avp_alignment,
    )
    avp_ip = str(config_data["avp_ip"])
    max_frames_value = config_data.get("max_frames")
    max_frames = None if max_frames_value is None else int(max_frames_value)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")
    log_every_frames = int(config_data.get("log_every_frames", 20))
    if log_every_frames <= 0:
        raise ValueError("log_every_frames must be positive.")

    detector = runtime.session.detector
    detector.connect(avp_ip=avp_ip)
    last_diagnostics: dict[str, float] = {}
    try:
        while max_frames is None or runtime.frame_count < max_frames:
            sensor_data = detector.get_raw_stream()
            result = driver.step(sensor_data)
            last_diagnostics = result.diagnostics
            if runtime.frame_count % log_every_frames == 0:
                print(
                    f"frame={runtime.frame_count} sim_time={last_diagnostics.get('simulation_time', 0.0):.3f} "
                    f"tracking_max={last_diagnostics.get('tracking_error_max', 0.0):.6f} "
                    f"overrun={last_diagnostics.get('runtime_overrun', 0.0):.6f}"
                )
    except KeyboardInterrupt:
        pass
    return last_diagnostics
