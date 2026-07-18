"""ROS-framework robot backends that implement the shared execution contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from teleoperation.backends.base import BackendStepResult


class RosCommandBackend:
    """Adapt one atomic ROS command callback into a robot backend."""

    def __init__(
        self,
        *,
        initial_qpos: Sequence[float],
        control_period: float,
        execute_callback: Callable[[np.ndarray], np.ndarray | None],
        reset_callback: Callable[[np.ndarray], np.ndarray | None] | None = None,
    ) -> None:
        """Configure callback-driven robot command and reset operations.

        Args:
            initial_qpos: Initial command and measured state.
            control_period: Seconds represented by one callback command.
            execute_callback: Operation that publishes/executes one complete period.
            reset_callback: Optional operation that synchronizes a requested reset state.

        Returns:
            None.
        """
        values = np.asarray(initial_qpos, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("initial_qpos must be a finite one-dimensional vector.")
        if control_period <= 0:
            raise ValueError("control_period must be positive.")
        self.initial_qpos = values.copy()
        self._control_period = float(control_period)
        self._execute_callback = execute_callback
        self._reset_callback = reset_callback
        self._target_qpos = values.copy()
        self._actual_qpos = values.copy()

    @property
    def control_period(self) -> float:
        """Return seconds represented by one ROS command callback.

        Args:
            None.

        Returns:
            Configured command period in seconds.
        """
        return self._control_period

    def _validate(self, qpos: Sequence[float]) -> np.ndarray:
        """Validate one callback command against the configured qpos shape.

        Args:
            qpos: Requested robot positions.

        Returns:
            Independent finite command vector.
        """
        values = np.asarray(qpos, dtype=float)
        if values.shape != self.initial_qpos.shape or not np.isfinite(values).all():
            raise ValueError(f"qpos must be finite and have shape {self.initial_qpos.shape}.")
        return values.copy()

    def reset(self, qpos: Sequence[float] | None = None) -> None:
        """Synchronize callback backend state for a new flow cycle.

        Args:
            qpos: Optional reset target; defaults to the configured initial qpos.

        Returns:
            None.
        """
        command = self.initial_qpos.copy() if qpos is None else self._validate(qpos)
        actual = self._reset_callback(command.copy()) if self._reset_callback is not None else command
        self._target_qpos = command
        self._actual_qpos = command.copy() if actual is None else self._validate(actual)

    def get_joint_pos(self) -> np.ndarray:
        """Return the last measured state from a completed callback period.

        Args:
            None.

        Returns:
            Current robot positions.
        """
        return self._actual_qpos.copy()

    def get_target_joint_pos(self) -> np.ndarray:
        """Return the last command accepted by the callback backend.

        Args:
            None.

        Returns:
            Current target positions.
        """
        return self._target_qpos.copy()

    def execute(self, qpos: np.ndarray) -> BackendStepResult:
        """Execute one complete ROS callback command period atomically.

        Args:
            qpos: Requested robot positions.

        Returns:
            Immutable command and measured callback state.
        """
        command = self._validate(qpos)
        actual = self._execute_callback(command.copy())
        self._target_qpos = command
        self._actual_qpos = command.copy() if actual is None else self._validate(actual)
        return BackendStepResult(
            command_qpos=self._target_qpos,
            actual_qpos=self._actual_qpos,
            diagnostics={},
        )
