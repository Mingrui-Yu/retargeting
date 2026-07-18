"""Saved-artifact replay application runner."""

from __future__ import annotations

from typing import Any

from retargeting_apps.config import to_plain_config_data
from retargeting_apps.artifacts.trajectory import resolve_runtime_result_dir


def _resolve_camera_vector(
    viewer_data: dict[str, Any],
    option_name: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Read one three-dimensional initial-camera setting from viewer config.

    Args:
        viewer_data: Plain viewer configuration mapping.
        option_name: Name of the requested camera option.
        default: Default three-dimensional camera vector.

    Returns:
        The configured camera vector converted to a float tuple.
    """
    value = viewer_data.get(option_name, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Replay viewer {option_name} must contain exactly three numbers.")
    return tuple(float(component) for component in value)


def resolve_replay_options_from_config(config: Any) -> dict[str, Any]:
    """Build saved-artifact viewer options from a composed application config.

    Args:
        config: Hydra/OmegaConf config object or equivalent plain mapping.

    Returns:
        Runtime option mapping consumed by the Viser renderer.
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected replay config to be a mapping.")
    run_name = config_data.get("run_name")
    if run_name is None or not str(run_name).strip():
        raise ValueError("app=replay requires run_name=<offline_retarget runtime name>.")
    viewer_data = config_data.get("viewer", {})
    if not isinstance(viewer_data, dict):
        raise ValueError("Replay viewer configuration must be a mapping.")
    return {
        "result": str(resolve_runtime_result_dir(str(run_name), config_data.get("runtime_root", "outputs"))),
        "fps": float(viewer_data.get("fps", 30.0)),
        "port": int(viewer_data.get("port", 8080)),
        "no_robot_mesh": bool(viewer_data.get("no_robot_mesh", False)),
        "trail_length": int(viewer_data.get("trail_length", 120)),
        "human_keypoint_size": float(viewer_data.get("human_keypoint_size", 0.018)),
        "initial_camera_position": _resolve_camera_vector(
            viewer_data, "initial_camera_position", (1.5, 1.5, 1.2)
        ),
        "initial_camera_look_at": _resolve_camera_vector(
            viewer_data, "initial_camera_look_at", (0.0, 0.0, 0.45)
        ),
    }


def run(config: Any, argv: list[str]) -> None:
    """Run the saved-artifact replay task.

    Args:
        config: Composed replay application configuration.
        argv: Command-line overrides accepted by the common app interface.

    Returns:
        None. The viewer runs until interrupted.
    """
    del argv
    from retargeting_apps.visualization.viser_replay import run_replay_viewer

    run_replay_viewer(resolve_replay_options_from_config(config))
