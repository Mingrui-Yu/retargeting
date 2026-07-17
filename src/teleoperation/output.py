"""Teleoperation output post-processing independent of retargeting objectives."""

from __future__ import annotations

import numpy as np

from retargeting.config import TeleoperationModeConfig


class QposOutputFilter:
    """Apply execution-layer filtering to raw retargeting joint commands."""

    def __init__(self, initial_qpos: np.ndarray, mode_config: TeleoperationModeConfig) -> None:
        """Create the configured output filter.

        Args:
            initial_qpos: Command used before the first retargeted result.
            mode_config: Teleoperation mode containing output filtering settings.

        Returns:
            None.
        """
        mode_config.validate()
        self.mode_config = mode_config
        self.previous_qpos = np.asarray(initial_qpos, dtype=float).copy()

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset the previous command retained by the output filter.

        Args:
            qpos: Optional command to retain after reset.

        Returns:
            None.
        """
        if qpos is not None:
            self.previous_qpos = np.asarray(qpos, dtype=float).copy()

    def apply(self, raw_qpos: np.ndarray) -> np.ndarray:
        """Filter a raw retargeting solution for robot execution.

        Args:
            raw_qpos: Unfiltered qpos returned by the retargeting core.

        Returns:
            Command qpos after the configured output filtering.
        """
        raw_qpos = np.asarray(raw_qpos, dtype=float)
        output = raw_qpos
        if self.mode_config.output.smooth_output_qpos:
            alpha = self.mode_config.output.smoothing_alpha
            output = alpha * raw_qpos + (1.0 - alpha) * self.previous_qpos
        self.previous_qpos = np.asarray(output, dtype=float).copy()
        return self.previous_qpos.copy()


class QposCommandLimiter:
    """Apply per-frame velocity and actuator-range limits before execution."""

    def __init__(
        self,
        initial_qpos: np.ndarray,
        max_joint_speed: np.ndarray,
        command_hz: float,
        lower: np.ndarray,
        upper: np.ndarray,
        ctrlrange_policy: str = "clip",
    ) -> None:
        """Create a stateful online command limiter.

        Args:
            initial_qpos: Initial command in retargeting joint order.
            max_joint_speed: Per-joint speed limits in radians per second.
            command_hz: Retargeting command frequency.
            lower: Per-joint actuator control lower bounds.
            upper: Per-joint actuator control upper bounds.
            ctrlrange_policy: Whether out-of-range requests are clipped or rejected.

        Returns:
            None.
        """
        self.previous_qpos = np.asarray(initial_qpos, dtype=float).copy()
        self.max_joint_speed = np.asarray(max_joint_speed, dtype=float).copy()
        self.lower = np.asarray(lower, dtype=float).copy()
        self.upper = np.asarray(upper, dtype=float).copy()
        expected_shape = self.previous_qpos.shape
        for name, values in {
            "max_joint_speed": self.max_joint_speed,
            "lower": self.lower,
            "upper": self.upper,
        }.items():
            if values.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}.")
        if self.previous_qpos.ndim != 1 or not np.isfinite(self.previous_qpos).all():
            raise ValueError("initial_qpos must be a finite one-dimensional vector.")
        if command_hz <= 0:
            raise ValueError(f"command_hz must be positive, got {command_hz}.")
        if (self.max_joint_speed <= 0).any() or not np.isfinite(self.max_joint_speed).all():
            raise ValueError("max_joint_speed must contain positive finite values.")
        if (self.lower > self.upper).any():
            raise ValueError("lower bounds must not exceed upper bounds.")
        if ctrlrange_policy not in {"clip", "error"}:
            raise ValueError("ctrlrange_policy must be 'clip' or 'error'.")
        self.command_hz = float(command_hz)
        self.max_delta = self.max_joint_speed / self.command_hz
        self.ctrlrange_policy = ctrlrange_policy

    def reset(self, qpos: np.ndarray) -> None:
        """Reset the previous applied command used by velocity limiting.

        Args:
            qpos: Applied command to retain as limiter state.

        Returns:
            None.
        """
        values = np.asarray(qpos, dtype=float)
        if values.shape != self.previous_qpos.shape or not np.isfinite(values).all():
            raise ValueError("qpos must match the limiter shape and contain only finite values.")
        self.previous_qpos = values.copy()

    def apply(self, requested_qpos: np.ndarray) -> np.ndarray:
        """Range-check and velocity-limit one requested position command.

        Args:
            requested_qpos: Desired positions in retargeting joint order.

        Returns:
            Bounded command for the current 20 Hz frame.
        """
        requested = np.asarray(requested_qpos, dtype=float)
        if requested.shape != self.previous_qpos.shape:
            raise ValueError(f"requested_qpos must have shape {self.previous_qpos.shape}, got {requested.shape}.")
        if not np.isfinite(requested).all():
            raise ValueError("requested_qpos must contain only finite values.")
        outside = (requested < self.lower) | (requested > self.upper)
        if outside.any() and self.ctrlrange_policy == "error":
            outside_indices = np.flatnonzero(outside).tolist()
            raise ValueError(f"requested_qpos exceeds actuator ctrlrange at indices {outside_indices}.")
        ranged = np.clip(requested, self.lower, self.upper)
        delta = np.clip(ranged - self.previous_qpos, -self.max_delta, self.max_delta)
        self.previous_qpos = self.previous_qpos + delta
        return self.previous_qpos.copy()
