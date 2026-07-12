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
