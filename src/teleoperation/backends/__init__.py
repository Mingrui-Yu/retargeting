"""Robot execution backends with lazy optional simulator loading."""

from teleoperation.backends.base import BackendStepResult, RobotBackend
from teleoperation.backends.kinematic import KinematicRobotBackend

__all__ = ["BackendStepResult", "KinematicRobotBackend", "MujocoRobotBackend", "RobotBackend"]


def __getattr__(name: str):
    """Load optional backends only when their public names are requested.

    Args:
        name: Attribute requested from this package.

    Returns:
        Requested optional backend class.
    """
    if name == "MujocoRobotBackend":
        from teleoperation.backends.mujoco import MujocoRobotBackend

        return MujocoRobotBackend
    raise AttributeError(name)
