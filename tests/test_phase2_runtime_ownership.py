from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TELEOPERATION_SRC = REPO_ROOT / "src" / "teleoperation"


def _imported_modules(source_path: Path) -> set[str]:
    """Parse direct imports from one Python source file.

    Args:
        source_path: Python source file to inspect.

    Returns:
        Fully qualified module names referenced by import statements.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_legacy_retargeting_input_and_backend_packages_are_removed():
    """Require runtime adapters to have one canonical package owner.

    Args:
        None.

    Returns:
        None.
    """
    assert not (REPO_ROOT / "src" / "retargeting" / "inputs").exists()
    assert not (REPO_ROOT / "src" / "retargeting" / "backends").exists()


def test_runtime_config_types_are_owned_by_teleoperation():
    """Keep detector, mode, command, and simulator config outside core.

    Args:
        None.

    Returns:
        None.
    """
    import retargeting.config as core_config
    from teleoperation.config import (
        DetectionSourceConfig,
        MujocoRobotBindingConfig,
        MujocoSimulationConfig,
        TeleoperationCommandConfig,
        TeleoperationModeConfig,
        TeleoperationOutputConfig,
        TeleoperationRobotControlConfig,
    )

    runtime_types = (
        DetectionSourceConfig,
        MujocoRobotBindingConfig,
        MujocoSimulationConfig,
        TeleoperationCommandConfig,
        TeleoperationModeConfig,
        TeleoperationOutputConfig,
        TeleoperationRobotControlConfig,
    )
    assert {config_type.__module__ for config_type in runtime_types} == {"teleoperation.config"}
    for name in (
        "DetectionSourceConfig",
        "MujocoSimulationConfig",
        "TeleoperationModeConfig",
        "TeleoperationOutputConfig",
        "TeleoperationRobotControlConfig",
    ):
        assert not hasattr(core_config, name)


def test_canonical_runtime_packages_import_without_optional_dependencies():
    """Import runtime package surfaces without loading device or simulator extras.

    Args:
        None.

    Returns:
        None.
    """
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            \"\"\"Reject optional runtime modules during canonical package imports.\"\"\"

            blocked_roots = {{"avp_stream", "cv2", "mediapipe", "mujoco"}}

            def find_spec(self, fullname, path=None, target=None):
                \"\"\"Reject blocked roots and defer all other lookups.

                Args:
                    fullname: Fully qualified module name being resolved.
                    path: Optional parent package search path.
                    target: Optional module target used during reload.

                Returns:
                    None when the module is not blocked.
                \"\"\"
                del path, target
                if fullname.split(\".\", 1)[0] in self.blocked_roots:
                    raise ModuleNotFoundError(f\"Blocked optional dependency: {{fullname}}\")
                return None

        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        sys.meta_path.insert(0, BlockOptionalImports())

        import teleoperation.backends
        import teleoperation.config
        import teleoperation.flow
        import teleoperation.inputs
        import teleoperation.inputs.avp.offline
        import teleoperation.observation_mapping
        import teleoperation.types
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


def test_teleoperation_does_not_import_app_or_ros_packages():
    """Protect runtime adapters from outer composition and ROS dependencies.

    Args:
        None.

    Returns:
        None.
    """
    forbidden_roots = {"retargeting_apps", "retargeting_ros"}
    offenders = {
        (source_path.relative_to(TELEOPERATION_SRC).as_posix(), module_name)
        for source_path in TELEOPERATION_SRC.rglob("*.py")
        for module_name in _imported_modules(source_path)
        if module_name.split(".", 1)[0] in forbidden_roots
    }

    assert offenders == set()
