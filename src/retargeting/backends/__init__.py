from retargeting.backends.base import RobotBackend

__all__ = ["MujocoRobotBackend", "RobotBackend"]


def __getattr__(name: str):
    """Load optional backends only when their public names are requested.

    Args:
        name: Attribute requested from this package.

    Returns:
        Requested optional backend class.
    """
    if name == "MujocoRobotBackend":
        from retargeting.backends.mujoco import MujocoRobotBackend

        return MujocoRobotBackend
    raise AttributeError(name)
