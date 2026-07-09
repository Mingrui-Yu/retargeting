from _retargeting_compat import ensure_retargeting_package

ensure_retargeting_package()

from retargeting.robot_teleoperation import *  # noqa: F401,F403
from retargeting.robot_teleoperation import main


if __name__ == "__main__":
    main()
