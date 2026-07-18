from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGETING_SRC = REPO_ROOT / "src" / "retargeting"
PACKAGE_SOURCE_ROOTS = {
    package_name: REPO_ROOT / "src" / package_name
    for package_name in ("retargeting", "teleoperation", "retargeting_apps", "retargeting_ros")
}
ALLOWED_PACKAGE_DEPENDENCIES = {
    "retargeting": frozenset(),
    "teleoperation": frozenset({"retargeting"}),
    "retargeting_apps": frozenset({"retargeting", "teleoperation"}),
    "retargeting_ros": frozenset({"retargeting", "teleoperation"}),
}
REMOVED_MODULE_PREFIXES = (
    "retargeting.apps",
    "retargeting.artifacts",
    "retargeting.backends",
    "retargeting.inputs",
    "retargeting.main",
    "retargeting.pipelines",
    "retargeting.visualization",
    "retargeting_apps.pipelines",
    "teleoperation.avp_alignment",
    "teleoperation.input",
    "teleoperation.inputs.adapter",
    "teleoperation.inputs.offline_avp",
    "teleoperation.mujoco_runtime",
    "teleoperation.session",
)

ROS_IMPORT_ROOTS = frozenset(
    {
        "builtin_interfaces",
        "cv_bridge",
        "geometry_msgs",
        "rclpy",
        "sensor_msgs",
        "std_msgs",
        "tf2_ros",
        "visualization_msgs",
    }
)
OPTIONAL_RUNTIME_IMPORT_ROOTS = frozenset(
    {
        "avp_stream",
        "cv2",
        "mediapipe",
        "mjviser",
        "mujoco",
        "open3d",
        "viser",
    }
)
FORBIDDEN_RETARGETING_IMPORT_ROOTS = frozenset(
    {
        "retargeting_apps",
        "retargeting_ros",
        "teleoperation",
    }
) | OPTIONAL_RUNTIME_IMPORT_ROOTS | ROS_IMPORT_ROOTS


def _imported_modules(source_path: Path) -> set[str]:
    """Parse all direct imports in one Python source file.

    Args:
        source_path: Python source file to inspect.

    Returns:
        Fully qualified module names referenced by import statements.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    return imported_modules


def _find_forbidden_imports(source_root: Path, forbidden_roots: frozenset[str]) -> set[tuple[str, str]]:
    """Find imports whose top-level package crosses a declared boundary.

    Args:
        source_root: Package directory to scan recursively.
        forbidden_roots: Top-level module names that the package must not import.

    Returns:
        Relative source paths and fully qualified forbidden module names.
    """
    offenders = set()
    for source_path in source_root.rglob("*.py"):
        relative_path = source_path.relative_to(source_root).as_posix()
        for module_name in _imported_modules(source_path):
            if module_name.split(".", 1)[0] in forbidden_roots:
                offenders.add((relative_path, module_name))
    return offenders


def _package_dependency_graph() -> dict[str, set[str]]:
    """Build direct dependencies among the repository's Python packages.

    Args:
        None.

    Returns:
        Mapping from each top-level package to imported repository packages.
    """
    package_names = set(PACKAGE_SOURCE_ROOTS)
    graph: dict[str, set[str]] = {package_name: set() for package_name in package_names}
    for package_name, source_root in PACKAGE_SOURCE_ROOTS.items():
        for source_path in source_root.rglob("*.py"):
            for module_name in _imported_modules(source_path):
                imported_root = module_name.split(".", 1)[0]
                if imported_root in package_names and imported_root != package_name:
                    graph[package_name].add(imported_root)
    return graph


def _dependency_graph_is_acyclic(graph: dict[str, set[str]]) -> bool:
    """Check a directed dependency graph with Kahn-style elimination.

    Args:
        graph: Mapping from package to its direct package dependencies.

    Returns:
        True when every node can be removed without encountering a cycle.
    """
    remaining = {package_name: set(dependencies) for package_name, dependencies in graph.items()}
    while remaining:
        dependency_free = {package_name for package_name, dependencies in remaining.items() if not dependencies}
        if not dependency_free:
            return False
        for package_name in dependency_free:
            remaining.pop(package_name)
        for dependencies in remaining.values():
            dependencies.difference_update(dependency_free)
    return True


def test_retargeting_core_import_graph_has_no_runtime_adapter_dependency():
    """Keep the algorithm package free of outer runtime adapters.

    Args:
        None.

    Returns:
        None.
    """
    offenders = _find_forbidden_imports(RETARGETING_SRC / "core", FORBIDDEN_RETARGETING_IMPORT_ROOTS)

    assert offenders == set()


def test_retargeting_core_imports_only_the_pure_config_schema():
    """Prevent core modules from loading the broad config facade.

    Args:
        None.

    Returns:
        None.
    """
    forbidden_modules = {"retargeting.config", "retargeting.config.schema"}
    offenders = {
        (source_path.relative_to(RETARGETING_SRC).as_posix(), module_name)
        for source_path in (RETARGETING_SRC / "core").rglob("*.py")
        for module_name in _imported_modules(source_path)
        if module_name in forbidden_modules
    }

    assert offenders == set()


def test_retargeting_package_has_no_outer_or_optional_runtime_imports():
    """Enforce the final zero-import boundary for the retargeting package.

    Args:
        None.

    Returns:
        None.
    """
    offenders = _find_forbidden_imports(RETARGETING_SRC, FORBIDDEN_RETARGETING_IMPORT_ROOTS)

    assert offenders == set()


def test_repository_package_dependency_graph_is_declared_and_acyclic():
    """Enforce package ownership for the complete repository dependency graph.

    Args:
        None.

    Returns:
        None.
    """
    graph = _package_dependency_graph()
    violations = {
        package_name: dependencies - ALLOWED_PACKAGE_DEPENDENCIES[package_name]
        for package_name, dependencies in graph.items()
        if not dependencies <= ALLOWED_PACKAGE_DEPENDENCIES[package_name]
    }

    assert violations == {}
    assert graph["retargeting"] == set()
    assert graph["teleoperation"] == {"retargeting"}
    assert graph["retargeting_apps"] == {"retargeting", "teleoperation"}
    assert _dependency_graph_is_acyclic(graph)


def test_source_and_ros_compatibility_use_only_canonical_module_paths():
    """Reject imports from every package path removed by the reorganization.

    Args:
        None.

    Returns:
        None.
    """
    source_roots = (
        REPO_ROOT / "src",
        REPO_ROOT / "ws_ros2" / "src" / "retargeting_benchmark" / "src",
    )
    offenders = {
        (source_path.relative_to(REPO_ROOT).as_posix(), module_name)
        for source_root in source_roots
        for source_path in source_root.rglob("*.py")
        for module_name in _imported_modules(source_path)
        if any(
            module_name == removed_prefix or module_name.startswith(f"{removed_prefix}.")
            for removed_prefix in REMOVED_MODULE_PREFIXES
        )
    }

    assert offenders == set()


def test_transitional_symbols_and_duplicate_config_loaders_are_removed():
    """Require one canonical API and one shared config-source loader.

    Args:
        None.

    Returns:
        None.
    """
    from retargeting.core.types import RetargetingHandObservation
    from teleoperation.inputs.avp import offline

    assert not hasattr(RetargetingHandObservation, "hand_kps_in_wrist")
    assert not hasattr(RetargetingHandObservation, "wrist_pose_in_world")
    assert not hasattr(offline, "OfflineReplay")
    assert not hasattr(offline, "load_offline_replay")
    assert not hasattr(offline, "OfflineAvpTrajectory")
    assert not hasattr(offline, "load_offline_avp_trajectory")

    config_sources = [
        REPO_ROOT / "src" / "retargeting" / "config" / "io.py",
        REPO_ROOT / "src" / "retargeting" / "config" / "core.py",
        REPO_ROOT / "src" / "teleoperation" / "config.py",
        REPO_ROOT / "src" / "retargeting_apps" / "config.py",
    ]
    config_text = "\n".join(path.read_text(encoding="utf-8") for path in config_sources)
    backend_text = (
        REPO_ROOT / "src" / "teleoperation" / "backends" / "mujoco.py"
    ).read_text(encoding="utf-8")

    assert config_text.count("def load_config_source(") == 1
    assert "def _load_config_source(" not in config_text
    assert "RobotMujoco =" not in backend_text


def test_import_package_roots_succeeds_without_optional_runtime_dependencies():
    """Import every package root while optional runtime modules are unavailable.

    Args:
        None.

    Returns:
        None.
    """
    blocked_roots = tuple(sorted(OPTIONAL_RUNTIME_IMPORT_ROOTS | ROS_IMPORT_ROOTS))
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        class BlockOptionalRuntimeImports(importlib.abc.MetaPathFinder):
            \"\"\"Reject optional runtime imports during the isolated package import.\"\"\"

            blocked_roots = {blocked_roots!r}

            def find_spec(self, fullname, path=None, target=None):
                \"\"\"Reject a blocked root and defer every other module lookup.

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
        sys.meta_path.insert(0, BlockOptionalRuntimeImports())

        import retargeting
        import retargeting_apps
        import retargeting_ros
        import teleoperation
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
