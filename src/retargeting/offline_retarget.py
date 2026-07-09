from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from retargeting.config import (
    load_retargeting_config,
    load_robot_config,
    resolve_project_path,
    to_plain_config_data,
)
from retargeting.retargeting_replay import run_offline_retargeting
from retargeting.trajectory_result import save_retargeting_trajectory


def compose_hydra_offline_retarget_config(overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose the offline retarget app config with Hydra.

    Args:
        overrides: Hydra override strings supplied after the module name.

    Returns:
        Resolved plain dictionary containing app, robot, retargeting, and output settings.
    """
    try:
        import hydra
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "hydra-core is required for the offline retarget entrypoint. "
            "Install the project dependencies, for example with `pip install -e .`."
        ) from exc

    config_dir = resolve_project_path("configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(config_name="offline_retarget", overrides=list(overrides or []))
    return OmegaConf.to_container(config, resolve=True)


def _default_run_name(robot_name: str, retargeting_type: str) -> str:
    """Create a timestamped default run directory name.

    Args:
        robot_name: Robot config name.
        retargeting_type: Retargeting objective type.

    Returns:
        Filesystem-friendly run name.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}_{robot_name}_{retargeting_type.lower()}"


def resolve_output_dir(config_data: dict[str, Any], robot_name: str, retargeting_type: str) -> Path:
    """Resolve the output directory for a retargeting artifact.

    Args:
        config_data: Plain composed app config.
        robot_name: Robot config name used when generating a default run name.
        retargeting_type: Retargeting objective type used when generating a default run name.

    Returns:
        Output directory path.
    """
    output_dir = config_data.get("output_dir")
    if output_dir is not None:
        return resolve_project_path(str(output_dir))

    output_root = resolve_project_path(str(config_data.get("output_root", "outputs")))
    run_name = config_data.get("run_name") or _default_run_name(robot_name, retargeting_type)
    return output_root / str(run_name) / "retargeting"


def run_offline_retarget_from_config(config: Any, argv: list[str] | None = None) -> Path:
    """Run offline retargeting from a composed config and save the artifact.

    Args:
        config: Hydra/OmegaConf config object or equivalent plain dictionary.
        argv: Optional command-line arguments to record in metadata.

    Returns:
        Directory containing result.npz and metadata.yaml.
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected offline retarget config to be a mapping.")

    app_data = config_data.get("app", {})
    robot_source = config_data.get("robot", app_data.get("robot"))
    retargeting_source = config_data.get("retargeting", app_data.get("retargeting"))
    robot_config = load_robot_config(robot_source)
    retargeting_config = load_retargeting_config(retargeting_source)

    data_file = str(config_data.get("data", app_data.get("data")))
    start = int(config_data.get("start", app_data.get("start", 0)))
    end = int(config_data.get("end", app_data.get("end", -1)))
    stride = int(config_data.get("stride", app_data.get("stride", 1)))

    _, trajectory, metadata = run_offline_retargeting(
        data_file=data_file,
        hand_type=robot_config.hand_type,
        start=start,
        end=end,
        stride=stride,
        robot_config=robot_config,
        retargeting_config=retargeting_config,
    )
    metadata.command = ["python", "-m", "retargeting.offline_retarget", *(argv or [])]
    output_dir = resolve_output_dir(config_data, robot_config.name, retargeting_config.type)
    return save_retargeting_trajectory(output_dir, trajectory, metadata)


def main(argv: list[str] | None = None) -> None:
    """Run the offline retarget CLI.

    Args:
        argv: Optional command-line arguments after the module name.

    Returns:
        None.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    output_dir = run_offline_retarget_from_config(compose_hydra_offline_retarget_config(argv), argv=argv)
    print(f"Saved retargeting result to {output_dir}")


if __name__ == "__main__":
    main()
