"""CLI and configuration composition for the Viser retargeting replay viewer."""

from __future__ import annotations

import argparse
import sys
from typing import Any, List

from retargeting.config import (
    load_detection_source_config,
    load_replay_app_config,
    load_retargeting_config,
    load_retargeting_profile_config,
    load_robot_config,
    load_solver_config,
    load_teleoperation_mode_config,
    resolve_project_path,
    to_plain_config_data,
)
from retargeting.visualization.viser_replay import run_replay_viewer


DEFAULT_REPLAY_DATA = "tests/fixtures/avp_teleop_2025-01-16_20-27-43.npz"


def parse_args(argv: List[str] | None = None):
    """Parse the legacy argparse-compatible Viser replay options.

    Args:
        argv: Optional arguments after the module name.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Visualize offline hand retargeting replay with viser.")
    parser.add_argument("--config", default=None, help="Replay app config path.")
    parser.add_argument("--profile", default=None, help="Retargeting profile config path.")
    parser.add_argument("--detection-source", default=None, help="Detection source config path.")
    parser.add_argument("--robot", default=None, help="Robot config path.")
    parser.add_argument("--retarget", default=None, help="Retargeting method config path.")
    parser.add_argument("--data", default=None)
    parser.add_argument("--result", default=None, help="Saved retargeting result directory or result.npz path.")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-robot-mesh", action="store_true", default=None, help="Disable URDF mesh rendering.")
    parser.add_argument("--trail-length", type=int, default=None, help="Use 0 to show the full trajectory up to current frame.")
    return parser.parse_args(argv)


def resolve_replay_options(args):
    """Resolve legacy CLI arguments into viewer runtime options.

    Args:
        args: Parsed legacy CLI arguments.

    Returns:
        Runtime option mapping consumed by the Viser renderer.
    """
    app_config = load_replay_app_config(args.config) if args.config is not None else None
    viewer_config = app_config.viewer if app_config is not None else None
    profile_config_path = args.profile if args.profile is not None else (app_config.profile if app_config is not None else None)
    detection_source_path = args.detection_source if args.detection_source is not None else (
        app_config.detection_source if app_config is not None else None
    )
    profile_config = load_retargeting_profile_config(profile_config_path) if profile_config_path is not None else None
    detection_source_config = load_detection_source_config(detection_source_path) if detection_source_path is not None else None
    robot_config_path = args.robot if args.robot is not None else (profile_config.robot if profile_config is not None else None)
    retargeting_config_path = args.retarget if args.retarget is not None else (
        profile_config.method if profile_config is not None else None
    )
    solver_config_path = app_config.solver if app_config is not None else None
    robot_config = load_robot_config(robot_config_path) if robot_config_path is not None else None
    retargeting_config = load_retargeting_config(retargeting_config_path) if retargeting_config_path is not None else None
    return {
        "result": args.result if args.result is not None else getattr(app_config, "result", None),
        "data": args.data if args.data is not None else (app_config.data if app_config is not None else DEFAULT_REPLAY_DATA),
        "start": args.start if args.start is not None else (app_config.start if app_config is not None else 0),
        "end": args.end if args.end is not None else (app_config.end if app_config is not None else -1),
        "stride": args.stride if args.stride is not None else (app_config.stride if app_config is not None else 1),
        "fps": args.fps if args.fps is not None else (viewer_config.fps if viewer_config is not None else 30.0),
        "port": args.port if args.port is not None else (viewer_config.port if viewer_config is not None else 8080),
        "no_robot_mesh": args.no_robot_mesh if args.no_robot_mesh is not None else (
            viewer_config.no_robot_mesh if viewer_config is not None else False
        ),
        "trail_length": args.trail_length if args.trail_length is not None else (
            viewer_config.trail_length if viewer_config is not None else 120
        ),
        "robot_config": robot_config,
        "retargeting_config": retargeting_config,
        "retargeting_profile_config": profile_config,
        "detection_source_config": detection_source_config,
        "teleoperation_mode_config": load_teleoperation_mode_config(None),
        "solver_config": load_solver_config(solver_config_path),
    }


def compose_hydra_replay_config(overrides: List[str] | None = None) -> dict[str, Any]:
    """Compose the replay application config with Hydra.

    Args:
        overrides: Hydra override strings supplied after the module name.

    Returns:
        Resolved plain replay configuration.
    """
    try:
        import hydra
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise SystemExit("hydra-core is required for the Hydra replay entrypoint.") from exc
    config_dir = resolve_project_path("configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(config_name="replay", overrides=list(overrides or []))
    return OmegaConf.to_container(config, resolve=True)


def resolve_replay_options_from_config(config: Any) -> dict[str, Any]:
    """Build renderer runtime options from a Hydra-composed config.

    Args:
        config: Hydra/OmegaConf config object or equivalent plain mapping.

    Returns:
        Runtime option mapping consumed by the Viser renderer.
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected Hydra replay config to be a mapping.")
    app_data = config_data.get("app", {})
    viewer_data = config_data.get("viewer", app_data.get("viewer", {}))
    profile_source = config_data.get("profile", app_data.get("profile"))
    detection_source = config_data.get("detection_source", app_data.get("detection_source"))
    solver_source = config_data.get("solver", app_data.get("solver"))
    teleoperation_mode_source = config_data.get("teleoperation_mode", app_data.get("teleoperation_mode"))
    profile_config = load_retargeting_profile_config(profile_source) if profile_source is not None else None
    return {
        "result": config_data.get("result", app_data.get("result")),
        "data": config_data.get("data", app_data.get("data", DEFAULT_REPLAY_DATA)),
        "start": int(config_data.get("start", app_data.get("start", 0))),
        "end": int(config_data.get("end", app_data.get("end", -1))),
        "stride": int(config_data.get("stride", app_data.get("stride", 1))),
        "fps": float(viewer_data.get("fps", 30.0)),
        "port": int(viewer_data.get("port", 8080)),
        "no_robot_mesh": bool(viewer_data.get("no_robot_mesh", False)),
        "trail_length": int(viewer_data.get("trail_length", 120)),
        "robot_config": load_robot_config(profile_config.robot) if profile_config is not None else None,
        "retargeting_config": load_retargeting_config(profile_config.method) if profile_config is not None else None,
        "retargeting_profile_config": profile_config,
        "detection_source_config": load_detection_source_config(detection_source) if detection_source is not None else None,
        "teleoperation_mode_config": load_teleoperation_mode_config(teleoperation_mode_source),
        "solver_config": load_solver_config(solver_source),
    }


def _has_legacy_cli_args(argv: List[str]) -> bool:
    """Detect legacy argparse-style replay arguments.

    Args:
        argv: Command-line arguments after the program name.

    Returns:
        True when at least one argument starts with a dash.
    """
    return any(arg.startswith("-") for arg in argv)


def main(argv: List[str] | None = None) -> None:
    """Run the Viser replay application using Hydra or legacy argparse options.

    Args:
        argv: Optional command-line arguments after the module name.

    Returns:
        None.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    options = resolve_replay_options(parse_args(argv)) if _has_legacy_cli_args(argv) else (
        resolve_replay_options_from_config(compose_hydra_replay_config(argv))
    )
    run_replay_viewer(options)


if __name__ == "__main__":
    main()
