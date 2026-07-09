from pathlib import Path
import sys


def ensure_retargeting_package() -> None:
    repo_src = Path(__file__).resolve().parents[4] / "src"
    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.insert(0, repo_src_str)

