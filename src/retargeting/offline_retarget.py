from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from retargeting.config import (
    load_detection_source_config,
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
    load_teleoperation_mode_config,
    resolve_project_path,
    to_plain_config_data,
)
from retargeting.retargeting_replay import DEFAULT_DETECTION_SOURCE_CONFIG_PATH, run_offline_retargeting
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


def _post_action_enabled(config_data: dict[str, Any], action_name: str) -> bool:
    """Return whether one offline retarget post action is enabled.

    Args:
        config_data: Plain composed offline retarget config.
        action_name: Post action key under `post`, such as `benchmark` or `visualize`.

    Returns:
        True when the action has `enabled: true`.
    """
    post_data = config_data.get("post", {})
    action_data = post_data.get(action_name, {}) if isinstance(post_data, dict) else {}
    return bool(action_data.get("enabled", False)) if isinstance(action_data, dict) else False


def build_post_benchmark_config(config_data: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Build the benchmark config used after offline retargeting.

    Args:
        config_data: Plain composed offline retarget config.
        output_dir: Directory containing the just-saved retargeting artifact.

    Returns:
        Plain benchmark config consumed by `run_benchmark_from_config`.
    """
    post_data = config_data.get("post", {})
    benchmark_data = post_data.get("benchmark", {}) if isinstance(post_data, dict) else {}
    if not isinstance(benchmark_data, dict):
        benchmark_data = {}
    output_root = benchmark_data.get("output_root", config_data.get("output_root", "outputs"))
    return {
        "result": str(output_dir),
        "output_root": output_root,
        "output_dir": benchmark_data.get("output_dir"),
        "plot": bool(benchmark_data.get("plot", True)),
        "plot_root": benchmark_data.get("plot_root", output_root),
        "plot_dir": benchmark_data.get("plot_dir"),
    }


def build_post_visualize_config(
    config_data: dict[str, Any],
    output_dir: Path,
    profile_source: Any,
    solver_source: Any,
) -> dict[str, Any]:
    """Build the replay viewer config used after offline retargeting.

    Args:
        config_data: Plain composed offline retarget config.
        output_dir: Directory containing the just-saved retargeting artifact.
        profile_source: Retargeting profile config source already used for retargeting.
        solver_source: Solver config source already used for retargeting.

    Returns:
        Plain replay config consumed by `resolve_replay_options_from_config`.
    """
    post_data = config_data.get("post", {})
    visualize_data = post_data.get("visualize", {}) if isinstance(post_data, dict) else {}
    if not isinstance(visualize_data, dict):
        visualize_data = {}
    viewer_data = visualize_data.get("viewer", {})
    if not isinstance(viewer_data, dict):
        viewer_data = {}
    return {
        "result": str(output_dir),
        "data": config_data.get("data"),
        "detection_source": config_data.get(
            "detection_source", config_data.get("app", {}).get("detection_source", DEFAULT_DETECTION_SOURCE_CONFIG_PATH)
        ),
        "start": int(config_data.get("start", 0)),
        "end": int(config_data.get("end", -1)),
        "stride": int(config_data.get("stride", 1)),
        "profile": profile_source,
        "solver": solver_source,
        "viewer": {
            "fps": float(viewer_data.get("fps", 20.0)),
            "port": int(viewer_data.get("port", 9217)),
            "no_robot_mesh": bool(viewer_data.get("no_robot_mesh", False)),
            "trail_length": int(viewer_data.get("trail_length", 120)),
        },
    }


def run_benchmark_post_action(config: dict[str, Any]) -> tuple[Path, Path | None]:
    """Run the benchmark post action.

    Args:
        config: Plain benchmark config with `result` pointing at a saved artifact.

    Returns:
        Tuple of benchmark output directory and optional plot output directory.
    """
    from retargeting.benchmark_trajectory import run_benchmark_from_config

    return run_benchmark_from_config(config)


def run_visualize_post_action(config: dict[str, Any]) -> None:
    """Run the replay viewer post action.

    Args:
        config: Plain replay config with `result` pointing at a saved artifact.

    Returns:
        None. The viewer runs until interrupted.
    """
    from retargeting.viser_retargeting_visualize import resolve_replay_options_from_config, run_replay_viewer

    run_replay_viewer(resolve_replay_options_from_config(config))


def run_post_retarget_actions(
    config_data: dict[str, Any],
    output_dir: Path,
    profile_source: Any,
    solver_source: Any | None = None,
) -> list[str]:
    """Run enabled offline retarget post actions.

    Args:
        config_data: Plain composed offline retarget config.
        output_dir: Directory containing the just-saved retargeting artifact.
        profile_source: Retargeting profile config source already used for retargeting.
        solver_source: Solver config source already used for retargeting.

    Returns:
        Names of post actions that were started.
    """
    completed_actions: list[str] = []
    if _post_action_enabled(config_data, "benchmark"):
        benchmark_output_dir, plot_output_dir = run_benchmark_post_action(
            build_post_benchmark_config(config_data, output_dir)
        )
        print(f"Saved benchmark summary to {benchmark_output_dir}")
        if plot_output_dir is not None:
            print(f"Saved benchmark plots to {plot_output_dir}")
        completed_actions.append("benchmark")

    if _post_action_enabled(config_data, "visualize"):
        print(f"Starting replay viewer for {output_dir}")
        run_visualize_post_action(
            build_post_visualize_config(config_data, output_dir, profile_source, solver_source)
        )
        completed_actions.append("visualize")

    return completed_actions


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
    profile_source = config_data.get("profile", app_data.get("profile"))
    detection_source = config_data.get(
        "detection_source", app_data.get("detection_source", DEFAULT_DETECTION_SOURCE_CONFIG_PATH)
    )
    solver_source = config_data.get("solver", app_data.get("solver"))
    teleoperation_mode_source = config_data.get("teleoperation_mode", app_data.get("teleoperation_mode"))
    profile_config = load_retargeting_profile_config(profile_source)
    detection_source_config = load_detection_source_config(detection_source)
    robot_config = load_robot_config(profile_config.robot)
    retargeting_config = load_retargeting_config(profile_config.method)
    solver_config = load_solver_config(solver_source)
    teleoperation_mode_config = load_teleoperation_mode_config(teleoperation_mode_source)

    data_file = str(config_data.get("data", app_data.get("data")))
    start = int(config_data.get("start", app_data.get("start", 0)))
    end = int(config_data.get("end", app_data.get("end", -1)))
    stride = int(config_data.get("stride", app_data.get("stride", 1)))

    _, trajectory, metadata = run_offline_retargeting(
        data_file=data_file,
        start=start,
        end=end,
        stride=stride,
        robot_config=robot_config,
        retargeting_config=retargeting_config,
        retargeting_profile_config=profile_config,
        detection_source_config=detection_source_config,
        teleoperation_mode_config=teleoperation_mode_config,
        solver_config=solver_config,
    )
    metadata.command = ["python", "-m", "retargeting.offline_retarget", *(argv or [])]
    output_dir = resolve_output_dir(config_data, robot_config.name, retargeting_config.type)
    output_dir = save_retargeting_trajectory(output_dir, trajectory, metadata)
    run_post_retarget_actions(config_data, output_dir, profile_source, solver_source)
    return output_dir


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
