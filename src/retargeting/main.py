"""Unified configuration-driven entrypoint for retargeting applications."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from retargeting.config import resolve_project_path, to_plain_config_data

AppRunner = Callable[[dict[str, Any], list[str]], Any]


def compose_hydra_base_config(overrides: list[str] | None = None) -> dict[str, Any]:
    """Compose the base application configuration with Hydra.

    Args:
        overrides: Hydra override strings supplied after the module name.

    Returns:
        Resolved plain dictionary for the selected application.
    """
    try:
        import hydra
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "hydra-core is required for the retargeting entrypoint. "
            "Install the project dependencies, for example with `pip install -e .`."
        ) from exc

    config_dir = resolve_project_path("configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(config_name="base", overrides=list(overrides or []))
    config_data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_data, dict):
        raise ValueError("Expected base config to compose to a mapping.")
    return config_data


def _run_offline_retarget(config: dict[str, Any], argv: list[str]) -> Any:
    """Run the offline retarget application without importing optional viewer code.

    Args:
        config: Composed offline retarget application configuration.
        argv: Command-line overrides to record in result metadata.

    Returns:
        Saved retargeting result directory.
    """
    from retargeting.apps.offline_retarget import run_offline_retarget_from_config

    output_dir = run_offline_retarget_from_config(config, argv=argv)
    print(f"Saved retargeting result to {output_dir}")
    return output_dir


def _run_replay(config: dict[str, Any], argv: list[str]) -> None:
    """Run the Viser replay application only when it is selected.

    Args:
        config: Composed replay application configuration.
        argv: Command-line overrides accepted for a uniform app-runner interface.

    Returns:
        None. The viewer runs until interrupted.
    """
    del argv
    from retargeting.apps.viser_retargeting_visualize import resolve_replay_options_from_config
    from retargeting.visualization.viser_replay import run_replay_viewer

    run_replay_viewer(resolve_replay_options_from_config(config))


def _run_benchmark(config: dict[str, Any], argv: list[str]) -> Any:
    """Run the benchmark application without importing replay or hardware adapters.

    Args:
        config: Composed benchmark application configuration.
        argv: Command-line overrides accepted for a uniform app-runner interface.

    Returns:
        Benchmark output directory and optional plot output directory.
    """
    from retargeting.apps.benchmark import run_benchmark_app

    return run_benchmark_app(config, argv=argv)


def _run_mujoco_simulation(config: dict[str, Any], argv: list[str]) -> Any:
    """Run live frame-by-frame retargeting into headless MuJoCo.

    Args:
        config: Composed online MuJoCo application configuration.
        argv: Command-line overrides accepted by the shared app interface.

    Returns:
        Diagnostics from the last completed simulation frame.
    """
    from retargeting.apps.mujoco_simulation import run_mujoco_simulation_from_config

    return run_mujoco_simulation_from_config(config, argv=argv)


def _run_mujoco_offline_simulation(config: dict[str, Any], argv: list[str]) -> Any:
    """Retarget raw offline human frames directly into headless MuJoCo.

    Args:
        config: Composed offline-human MuJoCo application configuration.
        argv: Command-line overrides accepted by the shared app interface.

    Returns:
        Diagnostics from the final processed human frame.
    """
    from retargeting.apps.mujoco_offline_simulation import run_mujoco_offline_simulation_from_config

    return run_mujoco_offline_simulation_from_config(config, argv=argv)


APP_RUNNERS: dict[str, AppRunner] = {
    "mujoco_offline_simulation": _run_mujoco_offline_simulation,
    "mujoco_simulation": _run_mujoco_simulation,
    "offline_retarget": _run_offline_retarget,
    "replay": _run_replay,
    "benchmark": _run_benchmark,
}


def resolve_app_runner(config: Any) -> AppRunner:
    """Resolve a configured application identifier to its whitelisted runner.

    Args:
        config: Composed application configuration containing ``app.id``.

    Returns:
        Runner for the selected application.
    """
    config_data = to_plain_config_data(config)
    if not isinstance(config_data, dict):
        raise ValueError("Expected application config to be a mapping.")
    app_config = config_data.get("app")
    if not isinstance(app_config, dict):
        raise ValueError("Base config requires an app mapping.")
    app_id = app_config.get("id")
    if app_id not in APP_RUNNERS:
        supported = ", ".join(sorted(APP_RUNNERS))
        raise ValueError(f"Unsupported app.id {app_id!r}. Supported values: {supported}.")
    return APP_RUNNERS[app_id]


def main(argv: list[str] | None = None) -> None:
    """Compose the selected application and dispatch it through the runner registry.

    Args:
        argv: Optional Hydra overrides after the module name.

    Returns:
        None.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    config = compose_hydra_base_config(argv)
    resolve_app_runner(config)(config, argv)


if __name__ == "__main__":
    main()
