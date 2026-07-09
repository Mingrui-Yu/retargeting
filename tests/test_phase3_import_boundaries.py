from __future__ import annotations

import importlib.abc
import sys
from pathlib import Path


class _BlockedRosFinder(importlib.abc.MetaPathFinder):
    _BLOCKED_ROOTS = {
        "builtin_interfaces",
        "cv_bridge",
        "geometry_msgs",
        "rclpy",
        "sensor_msgs",
        "std_msgs",
        "tf2_ros",
        "visualization_msgs",
    }

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in self._BLOCKED_ROOTS:
            raise ModuleNotFoundError(f"Blocked ROS import during core import boundary test: {fullname}")
        return None


def test_core_import_has_no_ros_dependency():
    finder = _BlockedRosFinder()
    sys.meta_path.insert(0, finder)
    try:
        import retargeting  # noqa: F401
        from retargeting import offline_replay, retargeting_replay  # noqa: F401
        from retargeting.robot_adaptor import RobotAdaptor  # noqa: F401
        from retargeting.robot_pinocchio import RobotPinocchio  # noqa: F401
    finally:
        sys.meta_path.remove(finder)


def test_retargeting_ros_package_import_has_no_runtime_side_effects():
    import retargeting_ros  # noqa: F401


def test_core_source_tree_has_no_direct_ros_imports():
    forbidden = (
        "import rclpy",
        "from rclpy",
        "sensor_msgs",
        "std_msgs",
        "geometry_msgs",
        "visualization_msgs",
        "cv_bridge",
        "tf2_ros",
        "builtin_interfaces",
    )
    src_root = Path(__file__).resolve().parents[1] / "src" / "retargeting"
    offenders = []
    for path in src_root.rglob("*.py"):
        text = path.read_text()
        for token in forbidden:
            if token in text:
                offenders.append((path.relative_to(src_root), token))

    assert offenders == []
