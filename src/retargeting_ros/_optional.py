from __future__ import annotations


def missing_ros_error(feature: str, original_error: ModuleNotFoundError | None = None) -> ImportError:
    message = f"{feature} requires ROS Python packages. Run it from a sourced ROS environment."
    if original_error is not None:
        message = f"{message} Missing module: {original_error.name}."
    return ImportError(message)
