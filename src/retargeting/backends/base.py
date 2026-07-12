from __future__ import annotations

from typing import Protocol

import numpy as np


class RobotBackend(Protocol):
    """Minimal command/state interface for runtime robot backends."""

    def get_joint_pos(self) -> np.ndarray:
        ...

    def ctrl_joint_pos(self, qpos: np.ndarray) -> None:
        """Command a target joint configuration using the shared runtime convention.

        Args:
            qpos: Target joint configuration in backend joint order.

        Returns:
            None.
        """
        ...

    def step(self) -> None:
        ...
