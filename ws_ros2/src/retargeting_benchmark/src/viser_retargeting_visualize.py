from _retargeting_compat import ensure_retargeting_package

ensure_retargeting_package()

from retargeting.viser_retargeting_visualize import *  # noqa: F401,F403
from retargeting.viser_retargeting_visualize import main


if __name__ == "__main__":
    main()
