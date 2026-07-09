from __future__ import annotations

from pathlib import Path


def default_robot_config_path(hand_type: str) -> Path:
    if hand_type == "leap":
        return Path("configs/robots/panda_leap_paxini.yaml")
    if hand_type == "shadow":
        return Path("configs/robots/panda_shadow.yaml")
    raise ValueError(f"Unsupported hand type: {hand_type}")
