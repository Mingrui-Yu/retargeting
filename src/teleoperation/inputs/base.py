"""Lifecycle protocol for pull-based sensor-first hand inputs."""

from __future__ import annotations

from typing import Protocol

from teleoperation.types import SensorHandSample


class HandInput(Protocol):
    """Pull-based source that distinguishes missing detection from end-of-stream."""

    def open(self) -> None:
        """Acquire source resources and prepare the first read.

        Args:
            None.

        Returns:
            None.
        """
        ...

    def read(self) -> SensorHandSample | None:
        """Read one sensor sample or report finite end-of-stream.

        Args:
            None.

        Returns:
            Sensor sample, including missing detections, or None at end-of-stream.
        """
        ...

    def reset(self) -> None:
        """Reset acquisition and decoder state to the start of a new cycle.

        Args:
            None.

        Returns:
            None.
        """
        ...

    def close(self) -> None:
        """Release source resources.

        Args:
            None.

        Returns:
            None.
        """
        ...
