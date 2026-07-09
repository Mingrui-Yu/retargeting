from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = REPO_ROOT / "src"
RETARGETING_SRC = REPO_ROOT / "ws_ros2" / "src" / "retargeting_benchmark" / "src"


def pytest_configure(config):
    sys.path.insert(0, str(RETARGETING_SRC))
    sys.path.insert(0, str(PACKAGE_SRC))


def pytest_runtest_setup(item):
    # Existing scripts resolve assets and data relative to the repository root.
    import os

    os.chdir(REPO_ROOT)
