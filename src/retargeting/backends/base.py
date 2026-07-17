from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class RobotBackend(Protocol):
    """Minimal command/state interface for runtime robot backends."""

    @property
    def control_period(self) -> float:
        """Return seconds advanced by one high-level backend step.

        Args:
            None.

        Returns:
            High-level command period in seconds.
        """
        ...

    def reset(self, qpos: Sequence[float] | None = None) -> None:
        """Reset backend state to an optional joint configuration.

        Args:
            qpos: Optional reset state in backend joint order.

        Returns:
            None.
        """
        ...

    def get_joint_pos(self) -> np.ndarray:
        """Return current joint positions in backend joint order.

        Args:
            None.

        Returns:
            Current joint-position vector.
        """
        ...

    def ctrl_joint_pos(self, qpos: np.ndarray) -> np.ndarray:
        """Command a target joint configuration using the shared runtime convention.

        Args:
            qpos: Target joint configuration in backend joint order.

        Returns:
            Applied joint command after backend validation.
        """
        ...

    def get_target_joint_pos(self) -> np.ndarray:
        """Return the last joint-position command accepted by the backend.

        Args:
            None.

        Returns:
            Last applied target vector.
        """
        ...

    def step(self) -> None:
        """Advance one complete high-level command period.

        Args:
            None.

        Returns:
            None.
        """
        ...
