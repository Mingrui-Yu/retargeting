"""Data returned by the retargeting algorithm boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class HandObservation:
    """Canonical hand motion consumed by device-independent retargeting."""

    keypoints_wrist: np.ndarray
    wrist_pose_world: np.ndarray
    timestamp: float | None = None
    handedness: Literal["left", "right"] = "right"
    keypoint_2d: np.ndarray | None = None
    raw: Any = None

class HandInput(Protocol):
    """Source of canonical human-hand observations."""

    def get_observation(self) -> HandObservation:
        """Return the next canonical hand observation.

        Args:
            None.

        Returns:
            Device-independent hand observation.
        """
        ...


@dataclass(frozen=True)
class RetargetingResult:
    """Result of solving one canonical human-hand observation."""

    qpos: np.ndarray
    diagnostics: Mapping[str, float]
