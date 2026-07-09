from __future__ import annotations

from typing import Protocol

import numpy as np


class RobotBackend(Protocol):
    """Minimal command/state interface for runtime robot backends."""

    def get_joint_pos(self) -> np.ndarray:
        ...

    def command_joint_pos(self, qpos: np.ndarray) -> None:
        ...

    def step(self) -> None:
        ...
