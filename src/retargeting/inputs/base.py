from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

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

    @property
    def hand_kps_in_wrist(self) -> np.ndarray:
        """Expose the legacy keypoint name during the API migration.

        Args:
            None.

        Returns:
            Hand keypoints expressed in the wrist frame.
        """
        return self.keypoints_wrist

    @property
    def wrist_pose_in_world(self) -> np.ndarray:
        """Expose the legacy wrist-pose name during the API migration.

        Args:
            None.

        Returns:
            Wrist pose expressed in robot world coordinates.
        """
        return self.wrist_pose_world


class HandInput(Protocol):
    """Source of canonical human-hand observations."""

    def get_observation(self) -> HandObservation:
        ...
