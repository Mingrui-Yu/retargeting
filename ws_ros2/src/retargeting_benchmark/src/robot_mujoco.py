from _retargeting_compat import ensure_retargeting_package

ensure_retargeting_package()

from retargeting.backends.mujoco import *  # noqa: F401,F403
