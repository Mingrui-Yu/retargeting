"""Saved-artifact replay application configuration adapter."""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Use `python -m retargeting.main app=replay` instead.")

from typing import Any

from retargeting.config import to_plain_config_data
from retargeting.artifacts.trajectory import resolve_runtime_result_dir


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
    }
