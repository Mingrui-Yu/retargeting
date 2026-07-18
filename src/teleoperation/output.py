"""Teleoperation output post-processing independent of retargeting objectives."""

from __future__ import annotations

import numpy as np

from teleoperation.config import TeleoperationModeConfig


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
    """Apply actuator ranges and plan velocity-limited startup commands."""

    def __init__(
        self,
        initial_qpos: np.ndarray,
        max_joint_speed: np.ndarray,
        command_hz: float,
        lower: np.ndarray,
        upper: np.ndarray,
        ctrlrange_policy: str = "clip",
    ) -> None:
        """Create a stateful command limiter.

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

    def _validate_qpos(self, qpos: np.ndarray, field_name: str) -> np.ndarray:
        """Validate one command vector against the configured command shape.

        Args:
            qpos: Joint-position vector to validate.
            field_name: Name used in validation errors.

        Returns:
            Finite float command vector.
        """
        values = np.asarray(qpos, dtype=float)
        if values.shape != self.previous_qpos.shape:
            raise ValueError(f"{field_name} must have shape {self.previous_qpos.shape}, got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{field_name} must contain only finite values.")
        return values

    def apply_range(self, requested_qpos: np.ndarray) -> np.ndarray:
        """Apply only the configured actuator-range policy to one request.

        Args:
            requested_qpos: Desired positions in retargeting joint order.

        Returns:
            Range-bounded target without a velocity limit.
        """
        requested = self._validate_qpos(requested_qpos, "requested_qpos")
        outside = (requested < self.lower) | (requested > self.upper)
        if outside.any() and self.ctrlrange_policy == "error":
            outside_indices = np.flatnonzero(outside).tolist()
            raise ValueError(f"requested_qpos exceeds actuator ctrlrange at indices {outside_indices}.")
        return np.clip(requested, self.lower, self.upper)

    def plan_move(self, start_qpos: np.ndarray, requested_qpos: np.ndarray) -> np.ndarray:
        """Plan synchronized linear waypoints that respect every speed limit.

        Args:
            start_qpos: Previously applied actuator target.
            requested_qpos: Desired final positions in retargeting joint order.

        Returns:
            Non-empty waypoint array whose final row is the range-bounded target.
        """
        start = self._validate_qpos(start_qpos, "start_qpos")
        target = self.apply_range(requested_qpos)
        delta = target - start
        required_steps = np.abs(delta) / self.max_delta
        step_count = max(1, int(np.ceil(float(np.max(required_steps)))))
        fractions = np.arange(1, step_count + 1, dtype=float)[:, None] / float(step_count)
        return start[None, :] + fractions * delta[None, :]

    def apply(self, requested_qpos: np.ndarray) -> np.ndarray:
        """Range-check and velocity-limit one requested position command.

        Args:
            requested_qpos: Desired positions in retargeting joint order.

        Returns:
            Bounded command for the current 20 Hz frame.
        """
        ranged = self.apply_range(requested_qpos)
        delta = np.clip(ranged - self.previous_qpos, -self.max_delta, self.max_delta)
        self.previous_qpos = self.previous_qpos + delta
        return self.previous_qpos.copy()
