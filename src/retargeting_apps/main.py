"""Unified configuration-driven entrypoint for retargeting applications."""

from __future__ import annotations

import sys
from importlib import import_module
from collections.abc import Callable
from types import ModuleType
from typing import Any

from retargeting_apps.config import resolve_project_path, to_plain_config_data

AppRunner = Callable[[Any, list[str]], Any]

APP_MODULES: dict[str, str] = {
    "benchmark": "retargeting_apps.apps.benchmark",
    "mujoco_offline_simulation": "retargeting_apps.apps.mujoco_offline_simulation",
    "mujoco_online_simulation": "retargeting_apps.apps.mujoco_online_simulation",
    "offline_retarget": "retargeting_apps.apps.offline_retarget",
    "replay": "retargeting_apps.apps.replay",
}


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


def _load_app_module(module_name: str) -> ModuleType:
    """Import one selected app module without eagerly loading other tasks.

    Args:
        module_name: Fully qualified module name from the app registry.

    Returns:
        Imported app module.
    """
    return import_module(module_name)


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
    if app_id not in APP_MODULES:
        supported = ", ".join(sorted(APP_MODULES))
        raise ValueError(f"Unsupported app.id {app_id!r}. Supported values: {supported}.")
    app_module = _load_app_module(APP_MODULES[app_id])
    runner = getattr(app_module, "run", None)
    if not callable(runner):
        raise TypeError(f"Application module {app_module.__name__!r} must expose callable run(config, argv).")
    return runner


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
