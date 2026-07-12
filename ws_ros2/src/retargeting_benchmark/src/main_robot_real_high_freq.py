#!/usr/bin/env python3
from _retargeting_compat import ensure_retargeting_package

ensure_retargeting_package()

from retargeting_ros.nodes.robot_real_high_freq import *  # noqa: F401,F403
from retargeting_ros.nodes.robot_real_high_freq import main


if __name__ == "__main__":
    main()
