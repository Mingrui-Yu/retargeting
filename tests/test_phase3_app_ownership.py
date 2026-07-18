from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGETING_SRC = REPO_ROOT / "src" / "retargeting"
RETARGETING_APPS_SRC = REPO_ROOT / "src" / "retargeting_apps"


def _imported_modules(source_path: Path) -> set[str]:
    """Parse direct module imports from one Python source file.

    Args:
        source_path: Python source file to inspect.

    Returns:
        Fully qualified modules referenced by import statements.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_application_modules_have_one_canonical_package_owner():
    """Require app composition to live only under retargeting_apps.

    Args:
        None.

    Returns:
        None.
    """
    for removed_path in ("main.py", "apps", "pipelines", "artifacts", "visualization"):
        assert not (RETARGETING_SRC / removed_path).exists()

    for owned_path in (
        "main.py",
        "config.py",
        "composition.py",
        "offline_retargeting.py",
        "benchmark_report.py",
        "apps",
        "artifacts",
        "visualization",
    ):
        assert (RETARGETING_APPS_SRC / owned_path).exists()
    assert not (RETARGETING_APPS_SRC / "pipelines").exists()

    for removed_module in (
        "retargeting.main",
        "retargeting.apps",
        "retargeting.pipelines",
        "retargeting.artifacts",
        "retargeting.visualization",
    ):
        assert importlib.util.find_spec(removed_module) is None


def test_console_script_and_package_discovery_use_retargeting_apps():
    """Keep the public command name while selecting the new composition root.

    Args:
        None.

    Returns:
        None.
    """
    project_config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'retargeting = "retargeting_apps.main:main"' in project_config
    assert '"retargeting_apps*"' in project_config
    assert 'retargeting = "retargeting.main:main"' not in project_config


def test_app_configs_modules_and_entry_contracts_align():
    """Require one uniformly callable task module for every Hydra app config.

    Args:
        None.

    Returns:
        None.
    """
    import yaml

    from retargeting_apps.main import APP_MODULES

    app_dir = RETARGETING_APPS_SRC / "apps"
    config_dir = REPO_ROOT / "configs" / "app"
    app_module_ids = {path.stem for path in app_dir.glob("*.py") if path.name != "__init__.py"}
    config_ids = {path.stem for path in config_dir.glob("*.yaml")}

    assert set(APP_MODULES) == app_module_ids == config_ids
    assert (RETARGETING_APPS_SRC / "composition.py").is_file()
    assert not (RETARGETING_APPS_SRC / "pipelines").exists()
    assert not (app_dir / "mujoco_runtime_builder.py").exists()
    assert not (app_dir / "viser_retargeting_visualize.py").exists()

    for app_id, module_name in APP_MODULES.items():
        app_source = (app_dir / f"{app_id}.py").read_text(encoding="utf-8")
        config_data = yaml.safe_load((config_dir / f"{app_id}.yaml").read_text(encoding="utf-8"))
        assert config_data["app"]["id"] == app_id
        assert module_name == f"retargeting_apps.apps.{app_id}"
        assert 'if __name__ == "__main__":' not in app_source

        app_module = importlib.import_module(module_name)
        signature = inspect.signature(app_module.run)
        assert list(signature.parameters) == ["config", "argv"]
        assert all(parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values())


def test_core_config_facade_has_no_application_schema():
    """Keep replay and viewer config owned by retargeting_apps.

    Args:
        None.

    Returns:
        None.
    """
    import retargeting.config as core_config
    from retargeting_apps.config import (
        MujocoWebViewerConfig,
        ReplayAppConfig,
        ViewerConfig,
        load_mujoco_web_viewer_config,
        load_replay_app_config,
    )

    assert not hasattr(core_config, "ReplayAppConfig")
    assert not hasattr(core_config, "ViewerConfig")
    assert not hasattr(core_config, "MujocoWebViewerConfig")
    assert not hasattr(core_config, "load_replay_app_config")
    assert not hasattr(core_config, "load_mujoco_web_viewer_config")

    replay_config = load_replay_app_config({"run_name": "smoke", "viewer": {"port": 9321}})
    assert isinstance(replay_config, ReplayAppConfig)
    assert isinstance(replay_config.viewer, ViewerConfig)
    assert replay_config.viewer.port == 9321
    mujoco_viewer_config = load_mujoco_web_viewer_config({"enabled": True, "port": 9322})
    assert isinstance(mujoco_viewer_config, MujocoWebViewerConfig)
    assert mujoco_viewer_config.port == 9322


def test_teleoperation_does_not_import_application_or_ros_packages():
    """Protect the dependency direction below the application layer.

    Args:
        None.

    Returns:
        None.
    """
    forbidden_roots = {"retargeting_apps", "retargeting_ros"}
    offenders = {
        (source_path.relative_to(REPO_ROOT).as_posix(), module_name)
        for source_path in (REPO_ROOT / "src" / "teleoperation").rglob("*.py")
        for module_name in _imported_modules(source_path)
        if module_name.split(".", 1)[0] in forbidden_roots
    }

    assert offenders == set()


def test_import_application_entrypoint_succeeds_without_optional_adapters():
    """Keep app selection lazy until a concrete optional runner is dispatched.

    Args:
        None.

    Returns:
        None.
    """
    blocked_roots = ("avp_stream", "cv2", "mediapipe", "mjviser", "mujoco", "open3d", "viser")
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            \"\"\"Reject optional adapter imports while loading the app entrypoint.\"\"\"

            def find_spec(self, fullname, path=None, target=None):
                \"\"\"Reject blocked modules and defer all other lookups.

                Args:
                    fullname: Fully qualified module being resolved.
                    path: Optional parent package search path.
                    target: Optional reload target.

                Returns:
                    None when the module is not blocked.
                \"\"\"
                del path, target
                if fullname.split(\".\", 1)[0] in {blocked_roots!r}:
                    raise ModuleNotFoundError(f\"Blocked optional dependency: {{fullname}}\")
                return None

        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        sys.meta_path.insert(0, BlockOptionalImports())

        import retargeting_apps
        import retargeting_apps.apps.teleop_exe
        import retargeting_apps.main
        import retargeting_apps.visualization.execution.mjviser
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
