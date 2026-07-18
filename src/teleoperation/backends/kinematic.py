"""Ideal joint-position backend without physics, hardware, or visualization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from teleoperation.backends.base import BackendStepResult


class KinematicRobotBackend:
    """Realize every accepted joint-position command exactly in one period."""

    def __init__(
        self,
        *,
        initial_qpos: Sequence[float],
        control_period: float,
    ) -> None:
        """Create an ideal position-controlled robot state.

        Args:
            initial_qpos: Initial positions in actuated-joint command order.
            control_period: Seconds represented by one atomic command period.

        Returns:
            None.
        """
        values = np.asarray(initial_qpos, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("initial_qpos must be a non-empty finite one-dimensional vector.")
        if isinstance(control_period, bool):
            raise ValueError("control_period must be a positive finite number.")
        period = float(control_period)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("control_period must be a positive finite number.")
        self.initial_qpos = values.copy()
        self._control_period = period
        self._target_qpos = values.copy()
        self._actual_qpos = values.copy()

    @property
    def control_period(self) -> float:
        """Return seconds represented by one ideal command period.

        Args:
            None.

        Returns:
            Fixed command period in seconds.
        """
        return self._control_period

    def _validate_qpos(self, qpos: Sequence[float]) -> np.ndarray:
        """Validate and detach one joint-position vector.

        Args:
            qpos: Requested positions in actuated-joint command order.

        Returns:
            Independent finite vector matching the configured robot shape.
        """
        values = np.asarray(qpos, dtype=float)
        if values.shape != self.initial_qpos.shape or not np.isfinite(values).all():
            raise ValueError(f"qpos must be finite and have shape {self.initial_qpos.shape}.")
        return values.copy()

    def reset(self, qpos: Sequence[float] | None = None) -> None:
        """Synchronize ideal target and actual state to one configuration.

        Args:
            qpos: Optional reset positions; defaults to the configured initial state.

        Returns:
            None.
        """
        reset_qpos = self.initial_qpos.copy() if qpos is None else self._validate_qpos(qpos)
        self._target_qpos = reset_qpos.copy()
        self._actual_qpos = reset_qpos.copy()

    def get_joint_pos(self) -> np.ndarray:
        """Return the current ideal actual positions.

        Args:
            None.

        Returns:
            Independent actual-position vector.
        """
        return self._actual_qpos.copy()

    def get_target_joint_pos(self) -> np.ndarray:
        """Return the most recently accepted position command.

        Args:
            None.

        Returns:
            Independent target-position vector.
        """
        return self._target_qpos.copy()

    def execute(self, qpos: np.ndarray) -> BackendStepResult:
        """Accept one command and realize it exactly in the current period.

        Args:
            qpos: Requested positions in actuated-joint command order.

        Returns:
            Immutable command result whose target and actual positions are equal.
        """
        command = self._validate_qpos(qpos)
        self._target_qpos = command.copy()
        self._actual_qpos = command.copy()
        return BackendStepResult(
            command_qpos=self._target_qpos,
            actual_qpos=self._actual_qpos,
            diagnostics={},
        )
