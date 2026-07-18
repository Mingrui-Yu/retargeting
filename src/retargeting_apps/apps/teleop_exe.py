"""Unified teleoperation execution app for online/offline inputs and backends."""

from __future__ import annotations

from typing import Any

from retargeting_apps.composition import build_execution_flow
from retargeting_apps.config import to_plain_config_data
from retargeting_apps.visualization.execution.manager import (
    ExecutionVisualizer,
    create_optional_execution_visualizer,
)
from teleoperation.config import load_execution_backend_config
from teleoperation.types import ExecutionStepResult


def _input_runtime_data(config_data: dict[str, Any]) -> dict[str, Any]:
    """Return nested input runtime settings when present.

    Args:
        config_data: Plain composed execution app config.

    Returns:
        Nested input mapping or an empty mapping.
    """
    input_data = config_data.get("input", {})
    return input_data if isinstance(input_data, dict) else {}


def _runtime_value(config_data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a runtime value from new nested input config or legacy root fields.

    Args:
        config_data: Plain composed execution app config.
        key: Runtime field name to resolve.
        default: Value returned when neither location defines the key.

    Returns:
        Resolved config value.
    """
    return config_data.get(key, _input_runtime_data(config_data).get(key, default))


def _is_offline_input(config_data: dict[str, Any]) -> bool:
    """Return whether the app config selects archived finite input.

    Args:
        config_data: Plain composed execution app config.

    Returns:
        True for offline input mode, otherwise False.
    """
    mode = _runtime_value(config_data, "mode")
    if mode is not None:
        return str(mode) == "offline"
    return _runtime_value(config_data, "data") is not None


def _add_progress_logger(config_data: dict[str, Any], flow: Any) -> None:
    """Attach a passive progress logger to the execution flow.

    Args:
        config_data: Plain composed execution app config.
        flow: Execution flow receiving a step observer.

    Returns:
        None.
    """
    log_every_frames = int(config_data.get("log_every_frames", 20))
    if log_every_frames <= 0:
        raise ValueError("log_every_frames must be positive.")
    selected_indices = getattr(flow.input, "frame_indices", None)

    def log_step(result: ExecutionStepResult) -> None:
        """Print compact source-frame progress without owning execution.

        Args:
            result: Immutable result from one completed source frame.

        Returns:
            None.
        """
        processed = flow.source_frame_count
        is_last_offline_frame = selected_indices is not None and result.source_index == selected_indices[-1]
        if processed % log_every_frames == 0 or is_last_offline_frame:
            prefix = (
                f"source_frame={result.source_index} processed={processed}/{len(selected_indices)}"
                if selected_indices is not None
                else f"frame={processed}"
            )
            print(
                f"{prefix} "
                f"sim_time={result.diagnostics.get('simulation_time', 0.0):.3f} "
                f"tracking_max={result.diagnostics.get('tracking_error_max', 0.0):.6f} "
                f"overrun={result.diagnostics.get('runtime_overrun', 0.0):.6f}"
            )

    flow.add_step_observer(log_step)


def _validate_runtime_options(config_data: dict[str, Any]) -> None:
    """Validate app-owned runtime options before expensive component creation.

    Args:
        config_data: Plain composed execution app config.

    Returns:
        None.
    """
    log_every_frames = int(config_data.get("log_every_frames", 20))
    if log_every_frames <= 0:
        raise ValueError("log_every_frames must be positive.")
    loop = _runtime_value(config_data, "loop", False)
    if not isinstance(loop, bool):
        raise ValueError("loop must be a boolean.")
    if not _is_offline_input(config_data):
        return
    backend_config = load_execution_backend_config(config_data.get("backend"))
    source_hz = float(_runtime_value(config_data, "source_hz", backend_config.command_hz))
    if source_hz <= 0:
        raise ValueError(f"source_hz must be positive, got {source_hz}.")
    if abs(1.0 / source_hz - backend_config.control_period) > 1e-12:
        raise ValueError(
            "Offline human source_hz must match the backend command rate until timestamp resampling is supported: "
            f"source_hz={source_hz}, command_hz={backend_config.command_hz}."
        )


def _add_offline_source_bounds(flow: Any, diagnostics: dict[str, Any]) -> None:
    """Append archived input bounds to summary diagnostics when available.

    Args:
        flow: Completed execution flow.
        diagnostics: Mutable summary diagnostics returned to the caller.

    Returns:
        None.
    """
    selected_indices = getattr(flow.input, "frame_indices", None)
    if selected_indices is None:
        return
    diagnostics.update(
        {
            "source_frame_start": float(selected_indices[0]),
            "source_frame_end": float(selected_indices[-1]),
        }
    )


def run(config: Any, argv: list[str]) -> dict[str, Any]:
    """Run one configured teleoperation execution flow.

    Args:
        config: Composed execution app configuration.
        argv: CLI arguments accepted for the common app-runner interface.

    Returns:
        Last-step diagnostics augmented with aggregate flow counters.
    """
    del argv
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected teleoperation execution config to be a mapping.")
    _validate_runtime_options(config_data)
    flow = build_execution_flow(config_data)
    _add_progress_logger(config_data, flow)
    visualizer: ExecutionVisualizer | None = None
    try:
        visualizer = create_optional_execution_visualizer(config_data, flow)
        summary = flow.run()
        diagnostics = summary.diagnostics
        _add_offline_source_bounds(flow, diagnostics)
        if visualizer is not None and _is_offline_input(config_data) and not flow.loop:
            visualizer.wait_after_completion()
        return diagnostics
    finally:
        if visualizer is not None:
            visualizer.close()
