from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class BackendStepResult:
    """State published after one atomic backend command period."""

    command_qpos: np.ndarray
    actual_qpos: np.ndarray
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        """Detach published backend state from mutable device buffers.

        Args:
            None.

        Returns:
            None.
        """
        command_qpos = np.asarray(self.command_qpos, dtype=float).copy()
        actual_qpos = np.asarray(self.actual_qpos, dtype=float).copy()
        command_qpos.setflags(write=False)
        actual_qpos.setflags(write=False)
        object.__setattr__(self, "command_qpos", command_qpos)
        object.__setattr__(self, "actual_qpos", actual_qpos)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType({str(key): float(value) for key, value in self.diagnostics.items()}),
        )


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

    def get_target_joint_pos(self) -> np.ndarray:
        """Return the last joint-position command accepted by the backend.

        Args:
            None.

        Returns:
            Last applied target vector.
        """
        ...

    def execute(self, qpos: np.ndarray) -> BackendStepResult:
        """Apply one target and advance one complete command period atomically.

        Args:
            qpos: Target joint configuration in backend joint order.

        Returns:
            Applied command, measured state, and backend diagnostics.
        """
        ...
